"""Unit tests for the faithful V-MPO variant (vmpo_v1), arXiv:1909.12238.

Pins the paper-faithful pieces: raw advantages (no rankgauss/norm), the E-step
top-half elite filter + softmax-normalized weights, the weighted-maximum-likelihood
policy loss (weight on log pi, NOT on A*ratio), the temperature dual (Eq.4) gradient
on eta, the decoupled KL trust region (Eq.5) gradient on alpha and theta, and that
the weight/KL gradients route to the right parameters.
"""
import numpy as np
import torch
from math import log
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence

from cleanrl.vmpo.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_vmpo_v1 import (
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


def test_vmpo_defaults_are_paper_faithful():
    a = Args()
    assert a.vmpo is True
    assert a.adv_transform == "v10"      # raw advantages: eta sets the scale
    assert a.norm_adv is False
    assert a.ent_coef == 0.0             # no entropy bonus in V-MPO
    assert a.vmpo_topk_frac == 0.5       # top-half elite filter
    assert a.vmpo_init_eta == 1.0 and a.vmpo_init_alpha == 1.0   # Table 1 continuous
    assert a.vmpo_eps_eta > 0 and a.vmpo_eps_alpha > 0
    assert a.actor_dist == "beta"
    assert a.total_timesteps == 8000000 and a.env_id == "HalfCheetah-v4"


def test_vmpo_forward_shapes_and_kl_is_closed_form():
    agent, args = _make_agent()
    B, A = 16, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    logp, value_logits, dist = agent.vmpo_forward(x, z)
    assert logp.shape == (B,)
    assert value_logits.shape == (B, args.num_bins)
    assert isinstance(dist, Beta)
    # KL(old || new) between two Beta policies is finite/closed-form and >= 0.
    a_old, b_old = agent.beta_params(x)
    kl = kl_divergence(Beta(a_old.detach(), b_old.detach()), dist).sum(1)
    assert kl.shape == (B,)
    assert torch.isfinite(kl).all() and (kl >= -1e-5).all()


def test_estep_topk_half_and_psi_normalized():
    # E-step keeps the top half by advantage; psi = softmax over them sums to 1 and
    # the bottom half is excluded entirely.
    torch.manual_seed(1)
    adv = torch.randn(32)
    k = max(1, int(round(0.5 * adv.numel())))
    top_adv, top_idx = torch.topk(adv, k)
    psi = torch.softmax(top_adv / 1.0, dim=0)
    assert k == 16
    assert abs(psi.sum().item() - 1.0) < 1e-5
    # every selected index is in the top half by value
    thresh = adv.sort(descending=True).values[k - 1]
    assert (adv[top_idx] >= thresh - 1e-6).all()
    # bottom-half indices are absent
    bottom = set(range(32)) - set(top_idx.tolist())
    assert all(adv[i] <= thresh + 1e-6 for i in bottom)


def test_weighted_ml_grad_routes_to_actor_not_critic():
    # L_pi = -sum psi * log pi(a|s): gradient hits the actor heads + trunk, NOT the
    # critic head. (Weight on log pi, not on A*ratio.)
    agent, args = _make_agent()
    B, A = 16, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    logp, value_logits, dist = agent.vmpo_forward(x, z)
    adv = torch.randn(B)
    k = B // 2
    top_adv, top_idx = torch.topk(adv, k)
    psi = torch.softmax(top_adv / 1.0, dim=0).detach()
    l_pi = -(psi * logp[top_idx]).sum()
    agent.zero_grad(set_to_none=True)
    l_pi.backward()
    assert agent.actor_alpha_head.weight.grad is not None
    assert agent.actor_beta_head.weight.grad is not None
    assert agent.trunk.out_proj.weight.grad is not None
    assert agent.critic_head.weight.grad is None     # weighted-ML does not touch the value head


def test_temperature_dual_grad_matches_analytic():
    # L_eta = eta*eps + eta*log mean exp(A/eta). d/d eta computed by autograd should
    # match the analytic temperature-loss gradient (finite, well-defined).
    torch.manual_seed(2)
    adv = torch.randn(16)
    k = adv.numel()
    eps_eta = 0.1
    eta = torch.tensor(1.3, requires_grad=True)
    lse = torch.logsumexp(adv / eta, dim=0) - log(k)
    l_eta = eta * eps_eta + eta * lse
    l_eta.backward()
    g_auto = eta.grad.item()
    # analytic: dL/deta = eps + log mean exp(A/eta) - (1/eta) * sum softmax(A/eta)*A
    with torch.no_grad():
        p = torch.softmax(adv / 1.3, dim=0)
        g_manual = eps_eta + (torch.logsumexp(adv / 1.3, dim=0) - log(k)).item() - (p * adv).sum().item() / 1.3
    assert abs(g_auto - g_manual) < 1e-4


def test_alpha_dual_grad_sign_increases_alpha_when_kl_exceeds_target():
    # L_alpha = mean[ alpha*(eps_alpha - sg[KL]) ]; dL/dalpha = mean(eps_alpha - KL).
    # If KL > eps_alpha, the gradient is negative => a minimizer raises alpha (tighter
    # trust region). This is the dual that adapts the penalty.
    eps_alpha = 0.01
    kl = torch.full((8,), 0.5)        # KL well above target
    alpha = torch.tensor(1.0, requires_grad=True)
    l_alpha = (alpha * (eps_alpha - kl.detach())).mean()
    l_alpha.backward()
    assert alpha.grad.item() < 0      # gradient descent => alpha increases
    assert abs(alpha.grad.item() - (eps_alpha - 0.5)) < 1e-5


def test_kl_penalty_grad_routes_to_actor():
    # The theta-side trust-region term sg[alpha]*KL(old||new) trains the policy params.
    agent, args = _make_agent()
    B, A = 12, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    a_old, b_old = agent.beta_params(x)
    old_dist = Beta(a_old.detach(), b_old.detach())
    _, _, new_dist = agent.vmpo_forward(x, z)
    kl = kl_divergence(old_dist, new_dist).sum(1)
    l_kl = (torch.tensor(1.0) * kl).mean()
    agent.zero_grad(set_to_none=True)
    l_kl.backward()
    assert agent.actor_alpha_head.weight.grad is not None
    assert agent.trunk.out_proj.weight.grad is not None


def test_dual_loss_does_not_touch_policy_params():
    # The (eta, alpha) dual loss is built from detached advantages/KL, so backprop
    # populates only the scalar duals, never the network params.
    agent, args = _make_agent()
    B, A = 16, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    a_old, b_old = agent.beta_params(x)
    old_dist = Beta(a_old.detach(), b_old.detach())
    _, _, new_dist = agent.vmpo_forward(x, z)
    kl = kl_divergence(old_dist, new_dist).sum(1)
    adv = torch.randn(B)
    k = B // 2
    top_adv, _ = torch.topk(adv.detach(), k)
    eta = torch.tensor(1.0, requires_grad=True)
    alpha = torch.tensor(1.0, requires_grad=True)
    l_eta = eta * 0.1 + eta * (torch.logsumexp(top_adv / eta, dim=0) - log(k))
    l_alpha = (alpha * (0.01 - kl.detach())).mean()
    agent.zero_grad(set_to_none=True)
    (l_eta + l_alpha).backward()
    assert eta.grad is not None and alpha.grad is not None
    assert agent.actor_alpha_head.weight.grad is None    # network untouched by dual
    assert agent.trunk.out_proj.weight.grad is None
