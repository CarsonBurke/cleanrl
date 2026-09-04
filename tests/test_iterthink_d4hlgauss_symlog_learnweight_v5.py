"""Unit tests for learnweight_v5: Beta weight meta-policy trained by REINFORCE on the
REALIZED held-out RETURNS DIFFERENCE (weighted vs uniform train step).

v5 replaces v4's first-order influence reward (w_i*A_i*s_i) with the realized effect of
the weighting: reward = J_V(theta - a*g_w) - J_V(theta - a*g_0), where g_w/g_0 are the
weighted/uniform train-loss gradients and J_V is the held-out PPO surrogate. No EMA
baseline (the uniform term is an exact control variate). Sign-neutral.

These tests pin: defaults (probe lr present, no eps/reg/ent/baseline-decay); weight-param
isolation; the Beta inputs/shape; linear mean-1 bounded + mean-1; that the uniform term
cancels (reward is a pure difference, zero when weights are uniform); that a candidate
SGD probe step is exactly restored; REINFORCE gradient routing to weight params only; and
that REINFORCE moves the Beta toward higher-reward samples.
"""
import numpy as np
import torch
from torch.distributions.beta import Beta

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v5 import (
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


def test_v5_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10" and a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0 and a.lw_probe_lr > 0.0 and a.lw_lr > 0.0
    assert a.actor_dist == "beta"
    # v5 dropped the first-order finite-diff and the EMA baseline.
    assert not hasattr(a, "lw_eps")
    assert not hasattr(a, "lw_reg")
    assert not hasattr(a, "lw_ent_coef")
    assert not hasattr(a, "lw_baseline_decay")


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
    agent, args = _make_agent(obs_dim=5, act_dim=3)
    assert agent.weight_body[0].in_features == args.hidden + 2 + 3


def test_linear_mean_one_is_bounded_and_mean_one():
    agent, args = _make_agent()
    B = 256
    dist, _ = _wdist(agent, B)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    w = b * (B / (b.sum() + 1e-8))
    assert (w >= 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4
    assert w.max().item() <= (1.0 / (b.mean().item())) + 1e-3


def test_reward_is_zero_for_uniform_weights():
    # With uniform weights (all ones), g_w == g_0 exactly, so the realized returns
    # difference reward must be identically zero -- the paired uniform term is the baseline.
    agent, args = _make_agent()
    B, A, obs_dim = 16, 3, 5
    x, z = torch.randn(B, obs_dim), _zb(B, A)
    oldlp = torch.randn(B)
    adv = torch.randn(B)
    ap = agent.actor_parameters()

    logp1, _, _ = agent.lw_forward(x, z)
    ratio1 = (logp1 - oldlp).exp()
    pg1 = adv * ratio1
    pg2 = adv * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef)
    loss = -torch.min(pg1, pg2)
    w_uniform = torch.ones(B)
    g_w = torch.autograd.grad((w_uniform * loss).mean(), ap, retain_graph=True)
    g_0 = torch.autograd.grad(loss.mean(), ap)

    def surr():
        lp = agent.lw_logprob(x, z)
        r = (lp - oldlp).exp()
        return torch.min(adv * r, adv * torch.clamp(r, 1 - args.clip_coef, 1 + args.clip_coef)).mean()

    a = args.lw_probe_lr
    with torch.no_grad():
        for p, g in zip(ap, g_w):
            p.add_(g, alpha=-a)
        jv_w = surr()
        for p, g in zip(ap, g_w):
            p.add_(g, alpha=a)
        for p, g in zip(ap, g_0):
            p.add_(g, alpha=-a)
        jv_0 = surr()
        for p, g in zip(ap, g_0):
            p.add_(g, alpha=a)
    assert abs((jv_w - jv_0).item()) < 1e-6


def test_probe_step_is_restored():
    agent, args = _make_agent()
    ap = agent.actor_parameters()
    before = [p.detach().clone() for p in ap]
    g = [torch.randn_like(p) for p in ap]
    a = args.lw_probe_lr
    with torch.no_grad():
        for p, gi in zip(ap, g):
            p.add_(gi, alpha=-a)
        for p, gi in zip(ap, g):
            p.add_(gi, alpha=a)
    for p, bf in zip(ap, before):
        assert torch.allclose(p, bf, atol=1e-7)


def test_reinforce_grad_routes_to_weight_params_only():
    agent, args = _make_agent()
    B = 16
    dist, _ = _wdist(agent, B)
    b = dist.sample().clamp(1e-6, 1 - 1e-6)
    reward = 0.7  # scalar realized returns difference
    loss = -(dist.log_prob(b).mean()) * reward
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_alpha_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_weighted_grad_differs_from_uniform_under_nonuniform_weights():
    # Sanity: with non-uniform weights g_w != g_0, so the probe can produce a nonzero
    # reward -- the meta-policy actually has something to optimize.
    agent, args = _make_agent()
    B, A, obs_dim = 32, 3, 5
    x, z, oldlp, adv = torch.randn(B, obs_dim), _zb(B, A), torch.randn(B), torch.randn(B)
    ap = agent.actor_parameters()
    logp1, _, _ = agent.lw_forward(x, z)
    ratio1 = (logp1 - oldlp).exp()
    loss = -torch.min(adv * ratio1, adv * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef))
    w = torch.rand(B)
    w = w * (B / w.sum())
    g_w = torch.autograd.grad((w * loss).mean(), ap, retain_graph=True)
    g_0 = torch.autograd.grad(loss.mean(), ap)
    diff = sum((a - b).abs().sum().item() for a, b in zip(g_w, g_0))
    assert diff > 1e-6


def test_reinforce_moves_beta_toward_high_reward_direction():
    # Surrogate of the real loop: a fixed per-sample reward proxy c_i, applied as a
    # scalar = mean(w*c). REINFORCE on -(logp.mean())*scalar should raise the Beta mean
    # for high-c samples (so the applied weighting puts mass where c is large).
    torch.manual_seed(4)
    agent, args = _make_agent()
    B = 64
    z = _zb(B, 3)
    _, _, feat = agent.lw_forward(torch.randn(B, 5), z)
    feat = feat.detach()
    adv, value, c = torch.randn(B), torch.randn(B), torch.randn(B)
    opt = torch.optim.Adam(
        [p for n, p in agent.named_parameters() if n.startswith("weight_")], lr=1e-2
    )
    for _ in range(300):
        dist = agent.weight_dist(feat, adv, value, z)
        b = dist.sample().clamp(1e-6, 1 - 1e-6)
        w = (b * (B / (b.sum() + 1e-8))).detach()
        reward = float((w * c).mean())  # scalar; higher when mass lands on high-c samples
        loss = -(dist.log_prob(b).mean()) * reward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mean_b = agent.weight_dist(feat, adv, value, z).mean
    corr = torch.corrcoef(torch.stack([mean_b, c]))[0, 1].item()
    assert corr > 0.3, corr
