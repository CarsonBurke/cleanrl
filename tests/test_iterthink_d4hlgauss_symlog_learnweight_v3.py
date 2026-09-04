"""Unit tests for the v3 STOCHASTIC BETA weight meta-policy (learnweight_v3).

v3 replaces v1's deterministic weight head with a Beta meta-policy (same template as
the actor: alpha,beta = 1+softplus, unimodal). The applied weight is a softmax over the
logit of the Beta sample, normalized to mean 1; the policy is trained by REINFORCE on
the held-out first-order policy-improvement reward (w_i*A_i*s_i) with an entropy bonus.

These tests pin: defaults; the weight params being a separate group (excluded from the
policy/value/main optimizers); the Beta being valid+unimodal; the softmax-logit weight
being positive + mean-1; REINFORCE/entropy gradient routing to the weight params only;
the finite-diff directional derivative and exact param restore; and that REINFORCE moves
the Beta toward higher-reward samples (non-degenerate learning).
"""
import numpy as np
import torch
from torch.distributions.beta import Beta

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v3 import (
    Agent,
    Args,
)


class _Box:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _Envs:
    def __init__(self, obs_dim, act_dim):
        self.single_observation_space = _Box((obs_dim,))
        self.single_action_space = _Box((act_dim,))


def _make_agent(obs_dim=5, act_dim=3):
    args = Args()
    torch.manual_seed(0)
    return Agent(_Envs(obs_dim, act_dim), args), args


def _zb(B, A):
    return torch.rand(B, A).clamp(1e-6, 1 - 1e-6)


def test_v3_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10"
    assert a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0
    assert a.lw_eps > 0.0
    assert a.lw_lr > 0.0
    assert a.actor_dist == "beta"
    assert not hasattr(a, "lw_reg")          # v3 dropped the L2-to-1 tax
    assert not hasattr(a, "lw_ent_coef")     # no entropy bonus (Beta sampling is the noise)


def test_weight_params_are_separate_group():
    agent, args = _make_agent()
    w_ids = {id(p) for n, p in agent.named_parameters() if n.startswith("weight_")}
    assert w_ids  # weight_body + alpha + beta heads
    actor_ids = {id(p) for p in agent.actor_parameters()}
    critic_ids = {id(p) for p in agent.critic_parameters()}
    main_ids = {id(p) for n, p in agent.named_parameters() if not n.startswith("weight_")}
    assert not (w_ids & actor_ids) and not (w_ids & critic_ids) and not (w_ids & main_ids)


def test_weight_dist_is_unimodal_beta():
    agent, args = _make_agent()
    B = 16
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    adv = torch.randn(B)
    dist = agent.weight_dist(actor_feat.detach(), adv)
    assert isinstance(dist, Beta)
    assert dist.concentration1.shape == (B,) and dist.concentration0.shape == (B,)
    assert (dist.concentration1 >= 1.0).all() and (dist.concentration0 >= 1.0).all()  # unimodal


def test_softmax_logit_weight_is_mean_one_and_positive():
    agent, args = _make_agent()
    B = 64
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    adv = torch.randn(B)
    dist = agent.weight_dist(actor_feat.detach(), adv)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    logit = torch.log(b) - torch.log1p(-b)
    w = torch.softmax(logit, dim=0) * B
    assert (w > 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-5      # softmax*N => mean exactly 1


def test_reinforce_and_entropy_grad_route_to_weight_params_only():
    agent, args = _make_agent()
    B = 16
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    adv = torch.randn(B)
    dist = agent.weight_dist(actor_feat.detach(), adv)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    logit = torch.log(b) - torch.log1p(-b)
    w = (torch.softmax(logit, dim=0) * B).detach()
    reward = w * torch.randn(B)
    adv_w = reward - reward.mean()
    loss = -(dist.log_prob(b) * adv_w).mean() - 0.01 * dist.entropy().mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_alpha_head.weight.grad is not None
    assert agent.weight_beta_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    # actor/critic must be untouched (weight features were detached)
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_finite_diff_matches_analytic_directional_derivative():
    agent, args = _make_agent()
    B, A = 24, 3
    x, z = torch.randn(B, 5), _zb(B, A)
    actor_params = agent.actor_parameters()
    logp = agent.lw_logprob(x, z)
    g = [gi.detach() for gi in torch.autograd.grad(logp.sum(), actor_params)]
    gnorm = torch.sqrt(sum((gi * gi).sum() for gi in g)) + 1e-12
    direction = [gi / gnorm for gi in g]
    analytic = sum((gi * di).sum() for gi, di in zip(g, direction)).item()
    eps = 1e-3
    logp0 = agent.lw_logprob(x, z).detach()
    with torch.no_grad():
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=eps)
        logp_pert = agent.lw_logprob(x, z)
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=-eps)
    fd_total = ((logp_pert - logp0) / eps).sum().item()
    assert abs(fd_total - analytic) / (abs(analytic) + 1e-6) < 1e-2


def test_params_restored_after_finite_diff():
    agent, args = _make_agent()
    x, z = torch.randn(16, 5), _zb(16, 3)
    actor_params = agent.actor_parameters()
    before = [p.detach().clone() for p in actor_params]
    direction = [torch.randn_like(p) for p in actor_params]
    with torch.no_grad():
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=1e-3)
        _ = agent.lw_logprob(x, z)
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=-1e-3)
    for p, bf in zip(actor_params, before):
        assert torch.allclose(p, bf, atol=1e-7)


def test_reinforce_moves_beta_toward_high_reward_samples():
    # With a fixed per-sample reward signal c_i, REINFORCE should raise the Beta mean
    # (=> higher applied weight) for samples with larger c. Check rank correlation.
    torch.manual_seed(4)
    agent, args = _make_agent()
    B = 64
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    actor_feat = actor_feat.detach()
    adv = torch.randn(B)
    c = torch.randn(B)                         # fixed reward-per-sample signal
    opt = torch.optim.Adam(
        [p for n, p in agent.named_parameters() if n.startswith("weight_")], lr=1e-2
    )
    for _ in range(300):
        dist = agent.weight_dist(actor_feat, adv)
        b = dist.sample().clamp(1e-6, 1 - 1e-6)
        logit = torch.log(b) - torch.log1p(-b)
        w = (torch.softmax(logit, dim=0) * B).detach()
        reward = w * c
        adv_w = reward - reward.mean()
        loss = -(dist.log_prob(b) * adv_w).mean() - 0.001 * dist.entropy().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        dist = agent.weight_dist(actor_feat, adv)
        mean_b = dist.mean                       # alpha/(alpha+beta) per sample
    corr = torch.corrcoef(torch.stack([mean_b, c]))[0, 1].item()
    assert corr > 0.3, corr
