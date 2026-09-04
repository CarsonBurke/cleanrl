"""Unit tests for learnweight_v4: stochastic Beta weight meta-policy with LINEAR
mean-1 normalization and action-augmented inputs.

v4 fixes v3's over-concentration (softmax(logit(b)) -> ESS ~10%) by using LINEAR
mean-1 (w = b*N/sum(b), bounded), and feeds the head the per-sample SAMPLED ACTION
(plus value) since the reward A*s depends on the action. Pure REINFORCE, no entropy.

These tests pin: defaults; weight-param isolation; the Beta inputs/shape; linear
mean-1 being bounded + mean-1; REINFORCE gradient routing with the new inputs; the
finite-diff signal + exact restore; and that REINFORCE moves the Beta toward
higher-reward samples.
"""
import numpy as np
import torch
from torch.distributions.beta import Beta

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v4 import (
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


def _wdist(agent, B, act_dim=3, obs_dim=5):
    z = _zb(B, act_dim)
    _, vlog, feat = agent.lw_forward(torch.randn(B, obs_dim), z)
    adv = torch.randn(B)
    value = torch.randn(B)
    return agent.weight_dist(feat.detach(), adv, value, z), z


def test_v4_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10" and a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0 and a.lw_eps > 0.0 and a.lw_lr > 0.0
    assert a.actor_dist == "beta"
    assert not hasattr(a, "lw_reg")
    assert not hasattr(a, "lw_ent_coef")


def test_weight_params_isolated():
    agent, args = _make_agent()
    w_ids = {id(p) for n, p in agent.named_parameters() if n.startswith("weight_")}
    actor_ids = {id(p) for p in agent.actor_parameters()}
    critic_ids = {id(p) for p in agent.critic_parameters()}
    main_ids = {id(p) for n, p in agent.named_parameters() if not n.startswith("weight_")}
    assert w_ids and not (w_ids & actor_ids) and not (w_ids & critic_ids) and not (w_ids & main_ids)


def test_weight_dist_unimodal_beta_with_action_inputs():
    agent, args = _make_agent()
    B = 16
    dist, _ = _wdist(agent, B)
    assert isinstance(dist, Beta)
    assert dist.concentration1.shape == (B,) and dist.concentration0.shape == (B,)
    assert (dist.concentration1 >= 1.0).all() and (dist.concentration0 >= 1.0).all()


def test_weight_head_input_includes_action_dim():
    # First linear layer must accept H + 2 (adv,value) + act_dim (sampled action).
    agent, args = _make_agent(obs_dim=5, act_dim=3)
    in_features = agent.weight_body[0].in_features
    assert in_features == args.hidden + 2 + 3


def test_linear_mean_one_is_bounded_and_mean_one():
    agent, args = _make_agent()
    B = 256
    dist, _ = _wdist(agent, B)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    w = b * (B / (b.sum() + 1e-8))
    assert (w >= 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4
    # bounded by the Beta range: max weight = b_max/mean(b) < 1/mean(b), no exp blowup
    assert w.max().item() <= (1.0 / (b.mean().item())) + 1e-3


def test_reinforce_grad_routes_to_weight_params_only():
    agent, args = _make_agent()
    B = 16
    dist, _ = _wdist(agent, B)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    w = (b * (B / (b.sum() + 1e-8))).detach()
    reward = w * torch.randn(B)
    adv_w = reward - reward.mean()
    loss = -(dist.log_prob(b) * adv_w).mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_alpha_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_finite_diff_matches_analytic():
    agent, args = _make_agent()
    x, z = torch.randn(24, 5), _zb(24, 3)
    ap = agent.actor_parameters()
    logp = agent.lw_logprob(x, z)
    g = [gi.detach() for gi in torch.autograd.grad(logp.sum(), ap)]
    gn = torch.sqrt(sum((gi * gi).sum() for gi in g)) + 1e-12
    d = [gi / gn for gi in g]
    analytic = sum((gi * di).sum() for gi, di in zip(g, d)).item()
    eps = 1e-3
    logp0 = agent.lw_logprob(x, z).detach()
    with torch.no_grad():
        for p, di in zip(ap, d):
            p.add_(di, alpha=eps)
        lp = agent.lw_logprob(x, z)
        for p, di in zip(ap, d):
            p.add_(di, alpha=-eps)
    assert abs(((lp - logp0) / eps).sum().item() - analytic) / (abs(analytic) + 1e-6) < 1e-2


def test_params_restored_after_finite_diff():
    agent, args = _make_agent()
    x, z = torch.randn(16, 5), _zb(16, 3)
    ap = agent.actor_parameters()
    before = [p.detach().clone() for p in ap]
    d = [torch.randn_like(p) for p in ap]
    with torch.no_grad():
        for p, di in zip(ap, d):
            p.add_(di, alpha=1e-3)
        _ = agent.lw_logprob(x, z)
        for p, di in zip(ap, d):
            p.add_(di, alpha=-1e-3)
    for p, bf in zip(ap, before):
        assert torch.allclose(p, bf, atol=1e-7)


def test_reinforce_moves_beta_toward_high_reward():
    torch.manual_seed(4)
    agent, args = _make_agent()
    B = 64
    z = _zb(B, 3)
    _, _, feat = agent.lw_forward(torch.randn(B, 5), z)
    feat = feat.detach()
    adv = torch.randn(B)
    value = torch.randn(B)
    c = torch.randn(B)
    opt = torch.optim.Adam(
        [p for n, p in agent.named_parameters() if n.startswith("weight_")], lr=1e-2
    )
    for _ in range(300):
        dist = agent.weight_dist(feat, adv, value, z)
        b = dist.sample().clamp(1e-6, 1 - 1e-6)
        w = (b * (B / (b.sum() + 1e-8))).detach()
        reward = w * c
        adv_w = reward - reward.mean()
        loss = -(dist.log_prob(b) * adv_w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mean_b = agent.weight_dist(feat, adv, value, z).mean
    corr = torch.corrcoef(torch.stack([mean_b, c]))[0, 1].item()
    assert corr > 0.3, corr
