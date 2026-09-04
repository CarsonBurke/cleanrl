import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
GUARD = 20.0


def _load(name, relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load("tpo_intra_beta_factored_v3", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_factored_v3.py")
V2 = _load("tpo_intra_beta_v2", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_v2.py")


# --- the two properties that make this a clean ablation of v2 ----------------


def test_joint_fixed_point_matches_v2():
    """Every coordinate at its target => joint ratio is exactly v2's exp(u/eta)."""
    d, eta = 6, 2.0
    utility = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    per_coord_target = (utility / (eta * d)).unsqueeze(-1).expand(-1, d)
    loss = V3.tpo_intra_loss_factored(per_coord_target, utility, eta=eta, logratio_guard=GUARD)
    torch.testing.assert_close(loss, torch.zeros(5), atol=1e-5, rtol=0)
    joint_logratio = per_coord_target.sum(-1)
    torch.testing.assert_close(joint_logratio, utility / eta, atol=1e-6, rtol=0)


def _gradient_gap(utility_scale, d=6, eta=2.0):
    """Max |v3 per-coordinate grad - v2 joint grad| at ratio 1, in float64."""
    utility = torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=torch.float64) * utility_scale
    factored = torch.zeros(4, d, dtype=torch.float64, requires_grad=True)
    V3.tpo_intra_loss_factored(factored, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    joint = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    V2.tpo_intra_loss(joint, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    return (factored.grad - joint.grad.unsqueeze(-1)).abs().max().item()


def test_linearised_gradient_matches_v2_to_first_order():
    """v3 and v2 agree on the ratio-1 gradient to first order in utility.

    They must differ at second order -- exactly u^2/(2 eta^2) * (1 - 1/d) -- so
    a fixed tolerance would either be vacuous or spuriously fail. Instead check
    the gap is genuinely quadratic: shrinking utility 10x must shrink it ~100x.
    """
    # Scales must sit inside the quadratic regime; at utility/eta ~ 1.5 the
    # cubic term is already comparable and the ratio is not informative.
    gap_big = _gradient_gap(0.1)
    gap_small = _gradient_gap(0.01)
    assert gap_big > 0.0
    ratio = gap_big / gap_small
    assert 90.0 < ratio < 110.0, f"gap is not quadratic in utility (ratio {ratio:.1f})"


def test_first_order_coefficient_is_exactly_minus_utility_over_eta():
    """Both forms have the same leading term, so both reduce to scaled PG."""
    d, eta = 6, 2.0
    utility = torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=torch.float64) * 1e-5
    factored = torch.zeros(4, d, dtype=torch.float64, requires_grad=True)
    V3.tpo_intra_loss_factored(factored, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    joint = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    V2.tpo_intra_loss(joint, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    expected = -utility / eta
    for i in range(d):
        torch.testing.assert_close(factored.grad[:, i], expected, rtol=1e-4, atol=0)
    torch.testing.assert_close(joint.grad, expected, rtol=1e-4, atol=0)


def test_gradient_is_d_times_ratio_minus_per_coordinate_target():
    d, eta = 4, 3.0
    utility = torch.tensor([-2.0, 1.5])
    logratio = torch.tensor(
        [[-0.4, 0.1, 0.25, -0.05], [0.3, -0.2, 0.0, 0.15]], requires_grad=True
    )
    V3.tpo_intra_loss_factored(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    target = (utility.unsqueeze(-1) / (eta * d)).exp()
    torch.testing.assert_close(logratio.grad, d * (logratio.detach().exp() - target))


def test_coordinates_stop_independently():
    """A coordinate already at its target gets zero gradient regardless of others."""
    d, eta = 4, 2.0
    utility = torch.tensor([2.0])
    at_target = utility.item() / (eta * d)
    logratio = torch.tensor([[at_target, 1.0, -1.0, 0.5]], requires_grad=True)
    V3.tpo_intra_loss_factored(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad[0, 0], torch.tensor(0.0), atol=1e-5, rtol=0)
    assert torch.all(logratio.grad[0, 1:].abs() > 1e-3)


def test_joint_form_cannot_stop_a_single_coordinate():
    """Contrast: in v2 no coordinate can stop while the joint ratio is off target."""
    eta = 2.0
    utility = torch.tensor([2.0])
    joint = torch.tensor([0.0], requires_grad=True)  # joint ratio 1, target > 1
    V2.tpo_intra_loss(joint, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    assert joint.grad.abs().item() > 1e-3


# --- shared TPO properties, re-checked in the factored form ------------------


def test_loss_is_non_negative_and_zero_only_at_the_target():
    d, eta = 6, 2.0
    utility = torch.tensor([-1.0, 0.5])
    base = (utility / (eta * d)).unsqueeze(-1).expand(-1, d).clone()
    for offset in (-0.5, -0.1, 0.1, 0.5):
        assert torch.all(V3.tpo_intra_loss_factored(base + offset, utility, eta, GUARD) > 0.0)


def test_variance_free_batch_is_actor_neutral():
    utility = V3.tpo_utility(torch.full((8,), -1.75), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(8))
    logratio = torch.zeros(8, 6, requires_grad=True)
    V3.tpo_intra_loss_factored(logratio, utility, eta=2.0, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(8, 6))


def test_suppressed_coordinates_keep_the_restoring_gradient():
    d, eta = 6, 2.0
    utility = torch.tensor([0.0])
    logratio = torch.tensor([[-19.0, -21.0, -1000.0, 0.0, 0.0, 0.0]], requires_grad=True)
    V3.tpo_intra_loss_factored(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    # target is exp(0) = 1, so the restoring gradient is d * (0 - 1) = -d.
    torch.testing.assert_close(
        logratio.grad[0, :3], torch.full((3,), -float(d)), atol=1e-5, rtol=0
    )


def test_gradient_above_the_guard_saturates_instead_of_vanishing():
    d = 3
    utility = torch.tensor([0.0])
    logratio = torch.tensor([[25.0, 0.0, 0.0]], requires_grad=True)
    V3.tpo_intra_loss_factored(logratio, utility, eta=1.0, logratio_guard=GUARD).backward()
    torch.testing.assert_close(
        logratio.grad[0, 0], torch.tensor(d * (math.exp(GUARD) - 1.0)), rtol=1e-6, atol=0
    )


def test_utility_is_detached_from_the_graph():
    utility = torch.tensor([1.0], requires_grad=True)
    logratio = torch.zeros(1, 4, requires_grad=True)
    V3.tpo_intra_loss_factored(logratio, utility, eta=2.0, logratio_guard=GUARD).sum().backward()
    assert utility.grad is None


# --- policy wiring -----------------------------------------------------------


def _envs_stub():
    import gymnasium as gym
    import numpy as np

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


def test_policy_returns_unsummed_per_coordinate_logprobs():
    torch.manual_seed(0)
    agent = V3.Agent(_envs_stub())
    obs = torch.randn(16, 5)
    action, z, logprob, entropy, value, concentration = agent.get_action_and_value(obs)
    assert logprob.shape == (16, 3), "the factored loss needs per-coordinate log-probs"
    assert entropy.shape == (16,)
    assert torch.all(action >= agent.action_low) and torch.all(action <= agent.action_high)
    assert torch.isfinite(logprob).all()


def test_relusq_activation_is_used():
    agent = V3.Agent(_envs_stub())
    assert any(isinstance(m, V3.ReluSq) for m in agent.actor.modules())
    assert any(isinstance(m, V3.ReluSq) for m in agent.critic.modules())
    assert not any(isinstance(m, torch.nn.Tanh) for m in agent.modules())
    x = torch.tensor([-2.0, 0.0, 3.0])
    torch.testing.assert_close(V3.ReluSq()(x), torch.tensor([0.0, 0.0, 9.0]))


def test_replaying_the_stored_variate_reproduces_the_behaviour_logprob():
    torch.manual_seed(0)
    agent = V3.Agent(_envs_stub())
    obs = torch.randn(32, 5)
    _, z, logprob, *_ = agent.get_action_and_value(obs)
    _, _, replayed, *_ = agent.get_action_and_value(obs, z)
    torch.testing.assert_close(replayed, logprob)


def test_gradient_flows_through_the_beta_policy_per_coordinate():
    torch.manual_seed(0)
    agent = V3.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        _, z, old_logprob, *_ = agent.get_action_and_value(obs)
    for p in agent.actor_alpha.parameters():
        p.data.add_(0.05)

    d, eta = 3, 2.0
    utility = torch.linspace(-2.0, 2.0, 8)
    _, _, logprob, *_ = agent.get_action_and_value(obs, z)
    logratio = logprob - old_logprob
    V3.tpo_intra_loss_factored(logratio, utility, eta=eta, logratio_guard=GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    target = (utility.unsqueeze(-1) / (eta * d)).exp()
    coefficient = (d * (logratio.detach().exp() - target)).detach()
    _, _, logprob2, *_ = agent.get_action_and_value(obs, z)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)
