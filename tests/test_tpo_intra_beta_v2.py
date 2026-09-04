import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_v2.py"
    spec = importlib.util.spec_from_file_location("ppo_continuous_action_tpo_intra_beta_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()
GUARD = 20.0


# --- utility whitening -------------------------------------------------------


def test_utility_clipping_actually_binds():
    advantages = torch.tensor([-8.0, -1.0, 0.0, 1.0, 8.0])
    clipped = MODULE.tpo_utility(advantages, utility_clip=0.5)
    unclipped = MODULE.tpo_utility(advantages, utility_clip=None)
    # The clip must bind on at least one element, or this asserts nothing.
    assert unclipped.abs().max() > 0.5
    torch.testing.assert_close(clipped.abs().max(), torch.tensor(0.5))
    assert (clipped != unclipped).any()
    torch.testing.assert_close(unclipped.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(unclipped.std(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_utility_is_permutation_equivariant():
    torch.manual_seed(0)
    advantages = torch.randn(256)
    utilities = MODULE.tpo_utility(advantages, utility_clip=3.0)
    perm = torch.randperm(256)
    # Whitening the permuted advantages must equal the permuted whitening.
    torch.testing.assert_close(MODULE.tpo_utility(advantages[perm], 3.0), utilities[perm])
    # Per-minibatch whitening genuinely disagrees with the fixed global target.
    assert not torch.allclose(MODULE.tpo_utility(advantages[perm[:64]], 3.0), utilities[perm[:64]])


def test_variance_free_batch_is_actor_neutral():
    utility = MODULE.tpo_utility(torch.full((16,), 3.25), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(16))
    logratio = torch.zeros(16, requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=4.0, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(16))


def test_single_sample_utility_is_finite_in_value_and_gradient():
    lone = torch.tensor([2.0], requires_grad=True)
    utility = MODULE.tpo_utility(lone, utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(1))
    utility.sum().backward()
    assert torch.isfinite(lone.grad).all()


# --- the TPO loss ------------------------------------------------------------


def test_loss_is_zero_and_minimal_at_the_target_ratio():
    utility = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    eta = 4.0
    at_target = utility / eta
    torch.testing.assert_close(
        MODULE.tpo_intra_loss(at_target, utility, eta=eta, logratio_guard=GUARD),
        torch.zeros(5),
        atol=1e-6,
        rtol=0,
    )
    for offset in (-1.0, -0.3, 0.3, 1.0):
        assert torch.all(MODULE.tpo_intra_loss(at_target + offset, utility, eta, GUARD) > 0.0)


def test_gradient_is_ratio_minus_target_ratio():
    utility = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    eta = 2.0
    logratio = torch.tensor([-0.7, -0.2, 0.0, 0.2, 0.7], requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, logratio.detach().exp() - (utility / eta).exp())


def test_gradient_vanishes_only_at_the_target_and_is_convex():
    utility = torch.tensor([1.0])
    eta = 2.0
    # Offset grid so no point can land exactly on utility/eta = 0.5.
    grid = (torch.linspace(-2.0, 2.0, 400) + 1e-3).requires_grad_(True)
    loss = MODULE.tpo_intra_loss(grid, utility.expand(400), eta=eta, logratio_guard=GUARD)
    (grad,) = torch.autograd.grad(loss.sum(), grid, create_graph=True)
    assert torch.all(grad != 0.0)
    assert (grad[:-1].sign() != grad[1:].sign()).sum().item() == 1
    (curvature,) = torch.autograd.grad(grad.sum(), grid)
    assert torch.all(curvature > 0.0)


def test_first_step_matches_the_policy_gradient_up_to_eta():
    """At ratio 1 with small utility, -dL/dlogratio ~= utility / eta."""
    utility = torch.tensor([-0.02, -0.01, 0.01, 0.02])
    eta = 4.0
    logratio = torch.zeros(4, requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(-logratio.grad, utility / eta, atol=1e-4, rtol=0)


def test_utility_is_detached_from_the_graph():
    utility = torch.tensor([1.0], requires_grad=True)
    logratio = torch.zeros(1, requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=2.0, logratio_guard=GUARD).sum().backward()
    assert utility.grad is None


# --- the guard: the v1 bug this replaces -------------------------------------


def test_suppressed_samples_keep_the_full_restoring_gradient():
    """Far below the guard the gradient must still be -target, not zero.

    v1 clamped logratio inside the linear term too, so anything past -guard was
    frozen with no signal in either direction.
    """
    utility = torch.tensor([0.0, 0.0, 0.0])
    logratio = torch.tensor([-19.0, -21.0, -1000.0], requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=1.0, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6, rtol=0)


def test_gradient_above_the_guard_saturates_instead_of_vanishing():
    utility = torch.tensor([0.0])
    beyond = torch.tensor([25.0], requires_grad=True)
    MODULE.tpo_intra_loss(beyond, utility, eta=1.0, logratio_guard=GUARD).backward()
    # Finite, positive (pushing the runaway sample back down), and saturated.
    torch.testing.assert_close(beyond.grad, torch.tensor([math.exp(GUARD) - 1.0]), rtol=1e-6, atol=0)


def test_loss_is_continuous_and_c1_across_the_guard():
    """Straddling the guard must move the loss by slope*2eps, not by a jump."""
    utility = torch.tensor([0.0, 0.0], dtype=torch.float64)
    eps = 1e-6
    at = torch.tensor([GUARD - eps, GUARD + eps], dtype=torch.float64, requires_grad=True)
    loss = MODULE.tpo_intra_loss(at, utility, eta=1.0, logratio_guard=GUARD)
    expected_step = math.exp(GUARD) * 2 * eps  # the local slope, not a discontinuity
    assert abs((loss[1] - loss[0]).item()) < 2 * expected_step
    loss.sum().backward()
    torch.testing.assert_close(at.grad[0], at.grad[1], rtol=1e-5, atol=0)


def test_guard_keeps_loss_and_gradient_finite_at_extremes():
    utility = torch.tensor([0.0, 0.0])
    huge = torch.tensor([200.0, -200.0], requires_grad=True)
    loss = MODULE.tpo_intra_loss(huge, utility, eta=1.0, logratio_guard=GUARD)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(huge.grad).all()
    # And the suppressed one is still being restored, not frozen.
    assert huge.grad[1].item() < 0.0


# --- end-to-end through the Beta policy --------------------------------------


def _envs_stub():
    import gymnasium as gym
    import numpy as np

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


def test_beta_policy_actions_stay_inside_the_action_box():
    torch.manual_seed(0)
    agent = MODULE.Agent(_envs_stub())
    obs = torch.randn(64, 5)
    action, z, logprob, entropy, value, concentration = agent.get_action_and_value(obs)
    assert torch.all(action >= agent.action_low) and torch.all(action <= agent.action_high)
    assert torch.all((z > 0.0) & (z < 1.0))
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()
    assert concentration.item() >= 2.0  # alpha, beta >= 1 each


def test_replaying_the_stored_variate_reproduces_the_behaviour_logprob():
    """Ratio must be exactly 1 when the policy has not moved."""
    torch.manual_seed(0)
    agent = MODULE.Agent(_envs_stub())
    obs = torch.randn(64, 5)
    _, z, logprob, *_ = agent.get_action_and_value(obs)
    _, _, replayed, *_ = agent.get_action_and_value(obs, z)
    torch.testing.assert_close(replayed, logprob)


def test_gradient_flows_through_the_beta_policy_as_ratio_minus_target():
    torch.manual_seed(0)
    agent = MODULE.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
    for p in agent.actor_alpha.parameters():
        p.data.add_(0.05)  # move the policy so the ratio is not trivially 1

    utility = torch.linspace(-2.0, 2.0, 8)
    eta = 3.0
    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    logratio = logprob - old_logprob
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta).exp()).detach()
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)
