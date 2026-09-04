"""Unit tests for learnweight_v7_hyper: LEARNED-AWR weight, REALIZED HYPERGRADIENT.

The weight is the same AWR/softmax-over-batch weight as v6 (logit mu_i = phi(feat,A,value),
w_i = N*softmax(mu_i)), but trained DETERMINISTICALLY by the analytic meta-gradient of the
realized held-out objective J_V(theta - a*g_w(phi)) -- no sampling/REINFORCE.

The math: J_V depends on phi only through theta' = theta - a*g_w(phi), so
    dJ_V/dphi = -a * h'^T dg_w/dphi = -a * grad_phi sum_i w_i(phi) (G_i . h'),
with G_i = grad_theta L_i^PPO and h' = grad_theta J_V AT theta'. The code computes per-sample
c_i = G_i . hat(h') by a finite-diff forward, sets lw_loss = mean_i w_i(phi)*c_i, and steps phi
to MINIMISE it (= ascend J_V). With g_w = (w_T*loss_T).mean() the exact relation is

    dJ_V/dphi = -a * |h'| * grad_phi(lw_loss).

The decisive test (test_metagrad_matches_realized_finite_difference) confirms this by
finite-differencing the TRUE realized objective vs the analytic meta-gradient.

Other tests pin: defaults (lw_eps present, no lw_logit_sigma); weight-param isolation; the
logit head shape + input dim (H+2, signed adv accepted); N*softmax mean-1 / sums-to-T /
can-concentrate; near-uniform start; meta-grad routes to weight params ONLY; and exact
theta restore after the probe + finite-diff.
"""
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v7_hyper import (
    Agent,
    Args,
)


def _value_s(value_logits1):
    # The block uses hl_support.to_scalar(value_logits).detach(); for the meta-grad MATH
    # value_s is just a detached constant input to phi, so any deterministic detached map
    # works here. Kept identical across the realized/analytic helpers below.
    return value_logits1.mean(dim=-1).detach()


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


# ---- a fixed deterministic minibatch + the helpers that mirror the training block ----
def _fixed_batch(B=12, obs_dim=5, act_dim=3, seed=3):
    torch.manual_seed(seed)
    obs = torch.randn(B, obs_dim)
    z = _zb(B, act_dim)
    oldlp = torch.randn(B) * 0.1
    adv = torch.randn(B)
    return obs, z, oldlp, adv


def _surr_V(agent, args, obs_V, z_V, oldlp_V, adv_V, clip_hi):
    lp = agent.lw_logprob(obs_V, z_V)
    r = (lp - oldlp_V).exp()
    return torch.min(adv_V * r, adv_V * torch.clamp(r, 1 - args.clip_coef, 1 + clip_hi)).mean()


def _realized_J(agent, args, obs, z, oldlp, adv, nT, clip_hi):
    """The TRUE realized objective J_V(theta - a*g_w(phi)) as a scalar (deterministic)."""
    a = args.lw_probe_lr
    M = obs.shape[0]
    actor_params = agent.actor_parameters()
    T = slice(0, nT)
    V = slice(nT, M)
    logp1, value_logits1, feat1 = agent.lw_forward(obs, z)
    ratio1 = (logp1 - oldlp).exp()
    adv1 = (adv - adv.mean()) / (adv.std() + 1e-8) if args.norm_adv else adv
    value_s = _value_s(value_logits1)
    mu = agent.weight_logit(feat1.detach(), adv1.detach(), value_s)
    w_T = (nT * torch.softmax(mu.detach()[T], dim=0))
    pg1 = adv1[T] * ratio1[T]
    pg2 = adv1[T] * torch.clamp(ratio1[T], 1 - args.clip_coef, 1 + clip_hi)
    loss_T = -torch.min(pg1, pg2)
    g_w = torch.autograd.grad((w_T * loss_T).mean(), actor_params, allow_unused=True)
    g_w = [torch.zeros_like(p) if g is None else g for p, g in zip(actor_params, g_w)]
    with torch.no_grad():
        for p, g in zip(actor_params, g_w):
            p.add_(g, alpha=-a)
        jv = _surr_V(agent, args, obs[V], z[V], oldlp[V], adv1[V].detach(), clip_hi).item()
        for p, g in zip(actor_params, g_w):
            p.add_(g, alpha=a)
    return jv


def _analytic_metagrad(agent, args, obs, z, oldlp, adv, nT, clip_hi):
    """Replicates the training block's meta-grad; returns (h_norm, a, grads-by-weight-param)."""
    a = args.lw_probe_lr
    M = obs.shape[0]
    actor_params = agent.actor_parameters()
    T = slice(0, nT)
    V = slice(nT, M)
    logp1, value_logits1, feat1 = agent.lw_forward(obs, z)
    ratio1 = (logp1 - oldlp).exp()
    adv1 = (adv - adv.mean()) / (adv.std() + 1e-8) if args.norm_adv else adv
    value_s = _value_s(value_logits1)
    mu = agent.weight_logit(feat1.detach(), adv1.detach(), value_s)
    pg1 = adv1[T] * ratio1[T]
    pg2 = adv1[T] * torch.clamp(ratio1[T], 1 - args.clip_coef, 1 + clip_hi)
    loss_T = -torch.min(pg1, pg2)
    loss_T0 = loss_T.detach()
    w_T_det = nT * torch.softmax(mu.detach()[T], dim=0)
    g_w = torch.autograd.grad((w_T_det * loss_T).mean(), actor_params, allow_unused=True)
    g_w = [torch.zeros_like(p) if g is None else g for p, g in zip(actor_params, g_w)]
    # h' at theta'
    with torch.no_grad():
        for p, g in zip(actor_params, g_w):
            p.add_(g, alpha=-a)
    jv = _surr_V(agent, args, obs[V], z[V], oldlp[V], adv1[V].detach(), clip_hi)
    hprime = torch.autograd.grad(jv, actor_params, allow_unused=True)
    hprime = [torch.zeros_like(p) if g is None else g for p, g in zip(actor_params, hprime)]
    with torch.no_grad():
        for p, g in zip(actor_params, g_w):
            p.add_(g, alpha=a)
    h_norm = torch.sqrt(sum((g * g).sum() for g in hprime)) + 1e-12
    scale = (args.lw_eps / h_norm).item()
    # c_i = G_i . hat(h') via finite diff (one-sided, as in the code)
    with torch.no_grad():
        for p, g in zip(actor_params, hprime):
            p.add_(g, alpha=scale)
        lp_p = agent.lw_logprob(obs[T], z[T])
        r_p = (lp_p - oldlp[T]).exp()
        pg1_p = adv1[T].detach() * r_p
        pg2_p = adv1[T].detach() * torch.clamp(r_p, 1 - args.clip_coef, 1 + clip_hi)
        loss_T_pert = -torch.min(pg1_p, pg2_p)
        for p, g in zip(actor_params, hprime):
            p.add_(g, alpha=-scale)
    c = ((loss_T_pert - loss_T0) / args.lw_eps).detach()
    w_T_phi = nT * torch.softmax(mu[T], dim=0)
    lw_loss = (w_T_phi * c).mean()
    agent.zero_grad(set_to_none=True)
    lw_loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in agent.named_parameters()
             if n.startswith("weight_") and p.grad is not None}
    return float(h_norm), a, grads


def test_v7_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10" and a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0 and a.lw_probe_lr > 0.0 and a.lw_lr > 0.0
    assert a.lw_eps > 0.0
    assert a.actor_dist == "beta"
    assert not hasattr(a, "lw_logit_sigma")   # no sampling in the deterministic trainer
    assert not hasattr(a, "lw_reg")
    assert not hasattr(a, "lw_ent_coef")


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
    assert agent.weight_body[0].in_features == args.hidden + 2  # [actor_feat (H), adv, value]


def test_signed_advantage_is_accepted():
    agent, args = _make_agent()
    adv = torch.linspace(-5.0, 5.0, 32)
    mu = _logit(agent, 32, adv=adv)
    assert mu.shape == (32,) and torch.isfinite(mu).all()


def test_softmax_weight_is_mean_one_positive_and_can_concentrate():
    B = 256
    z = torch.randn(B) * 1.5
    w = B * torch.softmax(z, dim=0)
    assert (w > 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4
    z2 = torch.zeros(B)
    z2[0] = 12.0
    w2 = B * torch.softmax(z2, dim=0)
    assert w2.max().item() > 0.5 * B


def test_softmax_over_T_sums_to_T():
    nT = 100
    z = torch.randn(nT)
    w_T = nT * torch.softmax(z, dim=0)
    assert abs(w_T.sum().item() - nT) < 1e-3


def test_starts_near_uniform():
    agent, args = _make_agent()
    B = 256
    mu = _logit(agent, B)
    assert mu.abs().mean().item() < 0.1
    w = B * torch.softmax(mu, dim=0)
    ess = (w.sum() ** 2 / (w.pow(2).sum())).item() / B
    assert ess > 0.95


def test_metagrad_routes_to_weight_params_only():
    # The deterministic meta-loss must only touch weight params (mu from detached inputs, c detached).
    agent, args = _make_agent()
    args.lw_eps = 1e-3
    clip_hi = args.clip_coef
    obs, z, oldlp, adv = _fixed_batch()
    nT = obs.shape[0] // 2
    _analytic_metagrad(agent, args, obs, z, oldlp, adv, nT, clip_hi)
    assert agent.weight_logit_head.weight.grad is not None
    assert agent.weight_body[0].weight.grad is not None
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_theta_restored_after_probe_and_finite_diff():
    agent, args = _make_agent()
    args.lw_eps = 1e-3
    clip_hi = args.clip_coef
    obs, z, oldlp, adv = _fixed_batch()
    nT = obs.shape[0] // 2
    before = [p.detach().clone() for p in agent.actor_parameters()]
    _analytic_metagrad(agent, args, obs, z, oldlp, adv, nT, clip_hi)
    for p, b in zip(agent.actor_parameters(), before):
        assert torch.allclose(p, b, atol=1e-7)


def test_metagrad_matches_realized_finite_difference():
    # DECISIVE: the analytic meta-grad must equal the finite-difference of the TRUE realized
    # objective J_V(theta - a*g_w(phi)). Exact relation: dJ/dphi = -a*|h'|*grad_phi(lw_loss).
    # Fully deterministic, so the only errors are O(eps) (c) and O(delta^2) (the FD below).
    agent, args = _make_agent()
    args.lw_eps = 1e-3
    clip_hi = args.clip_coef
    obs, z, oldlp, adv = _fixed_batch(B=12)
    nT = obs.shape[0] // 2

    h_norm, a, grads = _analytic_metagrad(agent, args, obs, z, oldlp, adv, nT, clip_hi)
    # analytic dJ/dp = -a * |h'| * p.grad
    pname = "weight_logit_head.weight"
    p = dict(agent.named_parameters())[pname]
    analytic_dJ = (-a * h_norm) * grads[pname].flatten()

    # central finite-difference of the realized J over each element of this weight param
    delta = 2e-3
    flat = p.data.view(-1)
    fd = torch.zeros_like(flat)
    for k in range(flat.numel()):
        orig = flat[k].item()
        flat[k] = orig + delta
        Jp = _realized_J(agent, args, obs, z, oldlp, adv, nT, clip_hi)
        flat[k] = orig - delta
        Jm = _realized_J(agent, args, obs, z, oldlp, adv, nT, clip_hi)
        flat[k] = orig
        fd[k] = (Jp - Jm) / (2 * delta)

    # direction must match almost exactly; magnitude within FD/eps tolerance
    cos = torch.nn.functional.cosine_similarity(analytic_dJ, fd, dim=0).item()
    assert cos > 0.97, f"cosine {cos}"
    ratio = (analytic_dJ.norm() / (fd.norm() + 1e-12)).item()
    assert 0.6 < ratio < 1.6, f"norm ratio {ratio}"
