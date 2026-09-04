"""Unit tests for the LEARNED advantage-weight variant (learnweight_v1).

The idea: replace the static rankgauss advantage transform with a per-sample weight
g_phi(actor_feat, A) learned ON-POLICY from a held-out POLICY-IMPROVEMENT signal.
Each minibatch is split T (train) / V (held-out). h = grad of the held-out PPO
surrogate wrt actor params is the direction that improves the policy on V. A
finite-difference forward gives s_i = grad log pi_i . (h/|h|) (first-order, no
second-order graph). The weight head maximizes sum_i w_i A_i s_i (so it up-weights
training samples whose advantage-scaled gradient generalizes to the held-out gain),
with an L2 pull toward 1 to prevent collapse. Weights are mean-1 normalized and
DETACHED when multiplying the actual PPO surrogate.

These tests pin: defaults; the weight head being trained separately (excluded from
the policy/value param groups and the main optimizer); positive mean-1 weights;
weight-head gradient routing; the held-out h routing to actor params; that the
finite-difference s matches the analytic directional derivative; and that the
weight-head objective moves weight toward high A*s samples (non-degenerate).
"""
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_learnweight_v1 import (
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


def test_learnweight_defaults():
    a = Args()
    assert a.learnweight is True
    assert a.adv_transform == "v10"     # clean GAE; the LEARNED weight is the only shaping
    assert a.norm_adv is True
    assert 0.0 < a.lw_holdout < 1.0
    assert a.lw_eps > 0.0
    assert a.lw_reg >= 0.0
    assert a.lw_lr > 0.0
    assert a.lw_hidden > 0
    assert a.actor_dist == "beta"


def test_weight_head_excluded_from_policy_value_groups():
    agent, args = _make_agent()
    wh_ids = {id(p) for p in agent.weight_head.parameters()}
    actor_ids = {id(p) for p in agent.actor_parameters()}
    critic_ids = {id(p) for p in agent.critic_parameters()}
    assert wh_ids and not (wh_ids & actor_ids) and not (wh_ids & critic_ids)
    # the main-optimizer param set (built by name in __main__) must also exclude it
    main = [p for n, p in agent.named_parameters() if not n.startswith("weight_head")]
    assert wh_ids and not (wh_ids & {id(p) for p in main})


def test_lw_forward_and_logprob_shapes():
    agent, args = _make_agent()
    B, A = 16, 3
    x, z = torch.randn(B, 5), _zb(B, A)
    logp, value_logits, actor_feat = agent.lw_forward(x, z)
    assert logp.shape == (B,)
    assert value_logits.shape == (B, args.num_bins)
    assert actor_feat.shape[0] == B
    lp2 = agent.lw_logprob(x, z)
    assert torch.allclose(logp, lp2, atol=1e-5)   # same path, same logpi


def test_weight_raw_positive_and_shape():
    agent, args = _make_agent()
    B = 16
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    adv = torch.randn(B)
    w = agent.weight_raw(actor_feat.detach(), adv)
    assert w.shape == (B,)
    assert (w >= 0).all()                          # softplus output is non-negative


def test_weight_raw_grad_routes_to_weight_head_only():
    agent, args = _make_agent()
    B = 16
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    adv = torch.randn(B)
    w = agent.weight_raw(actor_feat.detach(), adv)
    loss = (w * adv).sum()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    assert agent.weight_head[0].weight.grad is not None
    # the policy/value params must NOT receive gradient from the weight-head loss
    assert agent.actor_alpha_head.weight.grad is None
    assert agent.critic_head.weight.grad is None
    assert agent.trunk.out_proj.weight.grad is None


def test_mean_one_normalization():
    torch.manual_seed(3)
    w_raw = torch.rand(64) + 0.1
    w = w_raw * (w_raw.numel() / (w_raw.sum() + 1e-8))
    assert abs(w.mean().item() - 1.0) < 1e-5
    assert (w >= 0).all()


def test_heldout_grad_routes_to_actor_params():
    # h = grad of the held-out PPO surrogate wrt the actor params (trunk + heads),
    # which is the direction the finite-difference probes.
    agent, args = _make_agent()
    B, A = 16, 3
    x, z = torch.randn(B, 5), _zb(B, A)
    actor_params = agent.actor_parameters()
    logp, _, _ = agent.lw_forward(x, z)
    old_logp = logp.detach() - 0.01
    ratio = (logp - old_logp).exp()
    adv = torch.randn(B)
    surr = torch.min(adv * ratio, adv * torch.clamp(ratio, 0.8, 1.2)).mean()
    h = torch.autograd.grad(surr, actor_params, allow_unused=True)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in h)


def test_finite_diff_matches_analytic_directional_derivative():
    # s computed by perturbing the actor params by eps*dir then re-forwarding must
    # match the autograd directional derivative grad(sum logpi).dir to first order.
    agent, args = _make_agent()
    B, A = 24, 3
    x, z = torch.randn(B, 5), _zb(B, A)
    actor_params = agent.actor_parameters()

    # analytic: grad of sum_i logpi_i wrt actor params, dotted with a unit direction
    logp = agent.lw_logprob(x, z)
    g = torch.autograd.grad(logp.sum(), actor_params, retain_graph=False)
    g = [gi.detach() for gi in g]
    gnorm = torch.sqrt(sum((gi * gi).sum() for gi in g)) + 1e-12
    direction = [gi / gnorm for gi in g]           # unit direction == h/|h|
    analytic = sum((gi * di).sum() for gi, di in zip(g, direction)).item()

    # finite difference along the same unit direction
    eps = 1e-3
    logp0 = agent.lw_logprob(x, z).detach()
    with torch.no_grad():
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=eps)
        logp_pert = agent.lw_logprob(x, z)
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=-eps)
    s = (logp_pert - logp0) / eps
    fd_total = s.sum().item()                       # sum_i grad logpi_i . dir
    rel = abs(fd_total - analytic) / (abs(analytic) + 1e-6)
    assert rel < 1e-2, (fd_total, analytic)


def test_params_restored_after_finite_diff():
    # the in-place perturb/restore must leave the network bit-identical, otherwise
    # forward #2 (the real policy update) would run at the wrong parameters.
    agent, args = _make_agent()
    B, A = 16, 3
    x, z = torch.randn(B, 5), _zb(B, A)
    actor_params = agent.actor_parameters()
    before = [p.detach().clone() for p in actor_params]
    direction = [torch.randn_like(p) for p in actor_params]
    with torch.no_grad():
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=1e-3)
        _ = agent.lw_logprob(x, z)
        for p, d in zip(actor_params, direction):
            p.add_(d, alpha=-1e-3)
    for p, b in zip(actor_params, before):
        assert torch.allclose(p, b, atol=1e-7)


def test_weight_objective_moves_mass_toward_high_As():
    # Minimizing -(w*A*s).mean() + reg*(w-1)^2 should give larger weight to samples
    # with larger A*s. Take one gradient step and check rank correlation improves.
    torch.manual_seed(4)
    agent, args = _make_agent()
    B = 64
    _, _, actor_feat = agent.lw_forward(torch.randn(B, 5), _zb(B, 3))
    actor_feat = actor_feat.detach()
    adv = torch.randn(B)
    s = torch.randn(B)
    As = adv * s
    opt = torch.optim.Adam(agent.weight_head.parameters(), lr=1e-2)
    for _ in range(50):
        w_raw = agent.weight_raw(actor_feat, adv)
        w = w_raw * (w_raw.numel() / (w_raw.sum() + 1e-8))
        loss = -(w * As).mean() + 0.1 * ((w - 1.0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        w_raw = agent.weight_raw(actor_feat, adv)
        w = w_raw * (w_raw.numel() / (w_raw.sum() + 1e-8))
    # positive correlation between learned weight and the target alignment signal
    corr = torch.corrcoef(torch.stack([w, As]))[0, 1].item()
    assert corr > 0.3, corr
    assert abs(w.mean().item() - 1.0) < 1e-4
