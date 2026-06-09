"""Unit tests for learnweight_v6: LEARNED-AWR advantage weight.

The weight is an AWR/exp-tilt softmax over the batch advantages, but the softmax LOGIT
for each sample is a learned per-sample reweight mu_i = phi(actor_feat_i, A_i, value_i)
(signed A_i in, so phi chooses its own sign behavior). z_i ~ Normal(mu_i, sigma);
w_i = N * softmax(z_i); effective adv = w_i * A_i. Trained by REINFORCE on the realized
held-out returns difference (probe reused from v5).

These tests pin: defaults (logit_sigma present, no eps/reg/ent/baseline-decay); weight-param
isolation; the logit head shape + input dim (H+2, signed adv accepted); N*softmax is
mean-1/positive and CAN concentrate; softmax-over-T sums to |T|; reward is zero for uniform
logits (exact control variate); probe restore; REINFORCE routes to weight params only;
weighted!=uniform grad under non-uniform weights; near-uniform start; and REINFORCE moves
the logits toward higher-reward samples.
"""
import numpy as np
import torch
from torch.distributions.normal import Normal

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v6 import (
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


def _logit(agent, B, act_dim=3, obs_dim=5, adv=None):
    z = _zb(B, act_dim)
    _, _, feat = agent.lw_forward(torch.randn(B, obs_dim), z)
    adv = torch.randn(B) if adv is None else adv
    value = torch.randn(B)
    return agent.weight_logit(feat.detach(), adv, value)


def test_v6_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10" and a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0 and a.lw_probe_lr > 0.0 and a.lw_lr > 0.0
    assert a.lw_logit_sigma > 0.0
    assert a.actor_dist == "beta"
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


def test_weight_logit_shape_and_input_dim():
    agent, args = _make_agent(obs_dim=5, act_dim=3)
    mu = _logit(agent, 16)
    assert mu.shape == (16,)
    # Inputs are [actor_feat (H), advantage (1), value (1)] -- no action term.
    assert agent.weight_body[0].in_features == args.hidden + 2


def test_signed_advantage_is_accepted():
    # phi must accept signed advantages (incl. large negatives) without error or nan.
    agent, args = _make_agent()
    adv = torch.linspace(-5.0, 5.0, 32)
    mu = _logit(agent, 32, adv=adv)
    assert mu.shape == (32,) and torch.isfinite(mu).all()


def test_softmax_weight_is_mean_one_positive_and_can_concentrate():
    B = 256
    z = torch.randn(B) * 1.5
    w = B * torch.softmax(z, dim=0)
    assert (w > 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4          # N*softmax => mean exactly 1
    # Unlike a bounded mean-1 multiplicative weight, softmax CAN put large mass on a few:
    z2 = torch.zeros(B)
    z2[0] = 12.0
    w2 = B * torch.softmax(z2, dim=0)
    assert w2.max().item() > 0.5 * B                  # near-total concentration possible


def test_softmax_over_T_sums_to_T():
    nT = 100
    z = torch.randn(nT)
    w_T = nT * torch.softmax(z, dim=0)
    assert abs(w_T.sum().item() - nT) < 1e-3          # == sum(uniform ones over T)


def test_reward_is_zero_for_uniform_logits():
    # Equal logits => softmax uniform => w == 1 => g_w == g_0 => realized reward == 0.
    agent, args = _make_agent()
    B, A, obs_dim = 16, 3, 5
    x, z = torch.randn(B, obs_dim), _zb(B, A)
    oldlp, adv = torch.randn(B), torch.randn(B)
    ap = agent.actor_parameters()

    logp1, _, _ = agent.lw_forward(x, z)
    ratio1 = (logp1 - oldlp).exp()
    pg1 = adv * ratio1
    pg2 = adv * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef)
    loss = -torch.min(pg1, pg2)
    w_uniform = B * torch.softmax(torch.zeros(B), dim=0)   # == ones
    assert torch.allclose(w_uniform, torch.ones(B), atol=1e-5)
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
    mu = _logit(agent, B)
    dist = Normal(mu, args.lw_logit_sigma)
    z = dist.sample()
    reward = 0.7
    loss = -(dist.log_prob(z).mean()) * reward
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_logit_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_weighted_grad_differs_from_uniform_under_nonuniform_weights():
    agent, args = _make_agent()
    B, A, obs_dim = 32, 3, 5
    x, z, oldlp, adv = torch.randn(B, obs_dim), _zb(B, A), torch.randn(B), torch.randn(B)
    ap = agent.actor_parameters()
    logp1, _, _ = agent.lw_forward(x, z)
    ratio1 = (logp1 - oldlp).exp()
    loss = -torch.min(adv * ratio1, adv * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef))
    w = B * torch.softmax(torch.randn(B) * 1.5, dim=0)
    g_w = torch.autograd.grad((w * loss).mean(), ap, retain_graph=True)
    g_0 = torch.autograd.grad(loss.mean(), ap)
    diff = sum((a - b).abs().sum().item() for a, b in zip(g_w, g_0))
    assert diff > 1e-6


def test_starts_near_uniform():
    # Small head init => logits ~ 0 => softmax ≈ uniform => w ≈ 1 (ESS ≈ 1) at start.
    agent, args = _make_agent()
    B = 256
    mu = _logit(agent, B)
    assert mu.abs().mean().item() < 0.1
    w = B * torch.softmax(mu, dim=0)
    ess = (w.sum() ** 2 / (w.pow(2).sum())).item() / B
    assert ess > 0.95


def test_reinforce_moves_logits_toward_high_reward():
    # Fixed per-sample reward proxy c; scalar reward = mean(w*c). REINFORCE on
    # -(logp.mean())*reward should raise mu for high-c samples (mass lands on high c).
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
        mu = agent.weight_logit(feat, adv, value)
        dist = Normal(mu, args.lw_logit_sigma)
        zs = dist.sample()
        w = (B * torch.softmax(zs, dim=0)).detach()
        reward = float((w * c).mean())
        loss = -(dist.log_prob(zs).mean()) * reward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mu = agent.weight_logit(feat, adv, value)
    corr = torch.corrcoef(torch.stack([mu, c]))[0, 1].item()
    assert corr > 0.3, corr
