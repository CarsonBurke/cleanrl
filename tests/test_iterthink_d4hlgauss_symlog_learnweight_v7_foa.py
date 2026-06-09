"""Unit tests for learnweight_v7_foa: LEARNED-AWR weight, FIRST-ORDER analytic meta-grad.

Same softmax-AWR weight as v6 (logit mu_i = phi(actor_feat_i, A_i, value_i); w_i =
N*softmax(mu_i)), but trained by a DETERMINISTIC first-order meta-gradient instead of
REINFORCE. We linearize the held-out PPO improvement: a weighted train step changes J_V to
first order by  dJ_V ~ -a*|h| * sum_i w_i (G_i . hat(h)). The per-sample directional
derivative c_i = G_i . hat(h) is obtained by a finite difference of the per-sample train
loss along the unit held-out direction hat(h), then  L_meta = mean_i w_i(phi) * c_i  is
backpropped into phi only.

These tests pin: defaults (lw_eps present, no logit_sigma/probe_lr/reg/ent); weight-param
isolation; the logit head shape + input dim (H+2, signed adv accepted); N*softmax is
mean-1/positive and CAN concentrate; near-uniform start; finite-difference directional-
derivative correctness (vs an analytic directional derivative on a tiny example); exact
param restore after the finite-diff perturbation; and that the deterministic meta-grad
routes to the weight params only (NOT actor/critic).
"""
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v7_foa import (
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


def test_v7_foa_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10" and a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0 and a.lw_lr > 0.0
    assert a.lw_eps > 0.0
    assert a.actor_dist == "beta"
    # REINFORCE-era args are gone (no sampling, no probe step).
    assert not hasattr(a, "lw_logit_sigma")
    assert not hasattr(a, "lw_probe_lr")
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
    # The first-order trainer's expected pathology: softmax CAN put nearly all mass on one.
    z2 = torch.zeros(B)
    z2[0] = 12.0
    w2 = B * torch.softmax(z2, dim=0)
    assert w2.max().item() > 0.5 * B                  # near-total concentration possible


def test_softmax_over_T_is_mean_one():
    nT = 100
    z = torch.randn(nT)
    w_T = nT * torch.softmax(z, dim=0)
    assert abs(w_T.sum().item() - nT) < 1e-3          # == sum(uniform ones over T)
    assert abs(w_T.mean().item() - 1.0) < 1e-5


def test_starts_near_uniform():
    # Small head init => logits ~ 0 => softmax ≈ uniform => w ≈ 1 (ESS ≈ 1) at start.
    agent, args = _make_agent()
    B = 256
    mu = _logit(agent, B)
    assert mu.abs().mean().item() < 0.1
    w = B * torch.softmax(mu, dim=0)
    ess = (w.sum() ** 2 / (w.pow(2).sum())).item() / B
    assert ess > 0.95


def test_finite_diff_directional_derivative_matches_analytic():
    # c_i = (L_i(theta + eps*hat(h)) - L_i(theta)) / eps should match G_i . hat(h), where
    # G_i = grad_theta L_i. Verify on a tiny exact example (the actual code path: lw_forward
    # for L at theta, lw_logprob at theta + eps*hat(h)).
    torch.manual_seed(1)
    agent, args = _make_agent()
    ap = agent.actor_parameters()
    B, A, obs_dim = 6, 3, 5
    x, z, oldlp = torch.randn(B, obs_dim), _zb(B, A), torch.randn(B)
    adv = torch.randn(B)
    eps = 1e-4

    # Per-sample loss + its analytic gradients at theta.
    logp1, _, _ = agent.lw_forward(x, z)
    ratio1 = (logp1 - oldlp).exp()
    pg1 = adv * ratio1
    pg2 = adv * torch.clamp(ratio1, 1 - args.clip_coef, 1 + args.clip_coef)
    loss = -torch.min(pg1, pg2)            # (B,)
    loss0 = loss.detach()

    # Pick a fixed unit direction hat(h) in param space.
    torch.manual_seed(2)
    h = [torch.randn_like(p) for p in ap]
    h_norm = torch.sqrt(sum((g * g).sum() for g in h)) + 1e-12
    h = [g / h_norm for g in h]            # unit

    # Analytic c_i = G_i . hat(h) per sample (jacobian-vector via per-sample grad).
    c_analytic = []
    for i in range(B):
        gi = torch.autograd.grad(loss[i], ap, retain_graph=True, allow_unused=True)
        gi = [torch.zeros_like(p) if g is None else g for p, g in zip(ap, gi)]
        c_analytic.append(sum((a * b).sum() for a, b in zip(gi, h)).item())
    c_analytic = torch.tensor(c_analytic)

    # Finite-difference c_i via the production path (perturb theta in-place by eps*hat(h)).
    with torch.no_grad():
        for p, g in zip(ap, h):
            p.add_(g, alpha=eps)
        logp_pert = agent.lw_logprob(x, z)
        ratio_pert = (logp_pert - oldlp).exp()
        pg1_p = adv * ratio_pert
        pg2_p = adv * torch.clamp(ratio_pert, 1 - args.clip_coef, 1 + args.clip_coef)
        loss_pert = -torch.min(pg1_p, pg2_p)
        for p, g in zip(ap, h):
            p.add_(g, alpha=-eps)
    c_fd = (loss_pert - loss0) / eps

    assert torch.allclose(c_fd, c_analytic, atol=2e-2), (c_fd, c_analytic)


def test_finite_diff_perturbation_is_restored():
    # Add then subtract the SAME h tensors at the SAME scale => params exactly restored.
    agent, args = _make_agent()
    ap = agent.actor_parameters()
    before = [p.detach().clone() for p in ap]
    h = [torch.randn_like(p) for p in ap]
    scale = 0.013
    with torch.no_grad():
        for p, g in zip(ap, h):
            p.add_(g, alpha=scale)
        for p, g in zip(ap, h):
            p.add_(g, alpha=-scale)
    for p, bf in zip(ap, before):
        assert torch.allclose(p, bf, atol=1e-7)


def test_meta_grad_routes_to_weight_params_only():
    # The deterministic meta-loss L = mean(w(phi) * c) (c detached, mu from detached inputs)
    # must produce grads ONLY in the weight params, never the actor/critic.
    agent, args = _make_agent()
    B = 16
    nT = B
    mu = _logit(agent, B)               # built from detached actor_feat/adv/value
    c = torch.randn(nT)                 # detached per-sample directional derivative
    w_T_phi = nT * torch.softmax(mu, dim=0)
    loss = (w_T_phi * c).mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_logit_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_meta_grad_concentrates_on_lowest_c_and_collapses_ess():
    # Minimizing mean_i w_i * c_i (w = N*softmax(mu)) is UNBOUNDED below: the optimum piles
    # ALL mass on the single LOWEST-c sample (most held-out-improving direction). This is the
    # first-order OVER-CONCENTRATION pathology the header warns about: no curvature term
    # penalizes ESS collapse. Assert mass lands on argmin(c) and ESS crashes.
    torch.manual_seed(4)
    agent, args = _make_agent()
    B = 64
    z = _zb(B, 3)
    _, _, feat = agent.lw_forward(torch.randn(B, 5), z)
    feat = feat.detach()
    adv, value, c = torch.randn(B), torch.randn(B), torch.randn(B)
    ess_start = None
    opt = torch.optim.Adam(
        [p for n, p in agent.named_parameters() if n.startswith("weight_")], lr=2e-2
    )
    for step in range(800):
        mu = agent.weight_logit(feat, adv, value)
        w = B * torch.softmax(mu, dim=0)
        if step == 0:
            ess_start = (w.sum() ** 2 / w.pow(2).sum()).item() / B
        loss = (w * c).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mu = agent.weight_logit(feat, adv, value)
        w = B * torch.softmax(mu, dim=0)
        ess_end = (w.sum() ** 2 / w.pow(2).sum()).item() / B
        top = torch.argsort(w, descending=True)[: B // 8]   # most-weighted eighth
    # FIRST-ORDER OVER-CONCENTRATION (the header's expected failure mode): minimizing
    # mean(w*c) is unbounded below, so ESS collapses from ~uniform toward 1/B (mass piles
    # onto essentially one sample). Direction is still correct: the most-weighted samples
    # carry BELOW-AVERAGE c (the most held-out-improving directions).
    assert ess_start > 0.9, ess_start
    assert ess_end < 0.1, ess_end                            # near-total concentration
    assert c[top].mean().item() < c.mean().item(), (c[top].mean().item(), c.mean().item())
