import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_ghm_prefix_v2.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_ghm_prefix_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reward_coordinate_of_normalized_innovation_is_one_step_td_advantage():
    gamma = 0.97
    reward = torch.tensor([[2.5], [-1.0]])
    value = torch.tensor([[4.0], [3.0]])
    next_value = torch.tensor([[5.0], [7.0]])
    outcome = reward.unsqueeze(-1)
    mean = ((1.0 - gamma) * value).unsqueeze(-1)
    next_mean = ((1.0 - gamma) * next_value).unsqueeze(-1)
    termination = torch.tensor([[0.0], [1.0]])
    valid = torch.ones_like(termination)

    credit = MODULE.successor_innovation(
        outcome, mean, next_mean, termination, valid, gamma
    ).squeeze(-1)
    expected = reward + gamma * (1.0 - termination) * next_value - value
    torch.testing.assert_close(credit, expected)


def test_outcome_proximal_is_rotation_invariant_and_flat_at_behavior():
    torch.manual_seed(2)
    credit = torch.randn(19, 7)
    q, _ = torch.linalg.qr(torch.randn(7, 7))
    logratio = torch.randn(19) * 0.1
    actual = MODULE.outcome_proximal_loss(logratio, credit)
    rotated = MODULE.outcome_proximal_loss(logratio, credit @ q)
    torch.testing.assert_close(actual, rotated, atol=1e-6, rtol=1e-6)

    zero = torch.zeros(19, requires_grad=True)
    loss = MODULE.outcome_proximal_loss(zero, credit)
    gradient = torch.autograd.grad(loss, zero)[0]
    assert loss.item() == 0.0
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_outcome_proximal_preserves_direction_and_cross_sample_cancellation():
    logratio = torch.full((2,), torch.log(torch.tensor(2.0)))
    cancelling = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    coherent = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert MODULE.outcome_proximal_loss(logratio, cancelling).item() == 0.0
    torch.testing.assert_close(
        MODULE.outcome_proximal_loss(logratio, coherent), torch.tensor(1.0)
    )


def test_reward_projection_is_exact_32_step_td_advantage():
    torch.manual_seed(11)
    steps = 40
    gamma = 0.97
    reward = torch.randn(steps, 1)
    value = torch.randn(steps + 1, 1)
    outcome = reward.unsqueeze(-1)
    mean = ((1.0 - gamma) * value[:-1]).unsqueeze(-1)
    next_mean = ((1.0 - gamma) * value[1:]).unsqueeze(-1)
    one_step = MODULE.successor_innovation(
        outcome,
        mean,
        next_mean,
        torch.zeros(steps, 1),
        torch.ones(steps, 1),
        gamma,
    )
    credit = MODULE.fixed_prefix_credit(
        one_step, torch.zeros(steps, 1), gamma, credit_steps=32
    )
    discounts = gamma ** torch.arange(32)
    expected = (discounts * reward[:32, 0]).sum() + gamma**32 * value[32, 0] - value[0, 0]
    torch.testing.assert_close(credit[0, 0, 0], expected, atol=2e-5, rtol=2e-5)


def test_prefix_cuts_resets_and_keeps_truncation_final_observation_bootstrap():
    gamma = 0.9
    steps = 8
    reward = torch.arange(1.0, steps + 1).unsqueeze(1).repeat(1, 2)
    value = torch.arange(20.0, 20.0 + steps).unsqueeze(1).repeat(1, 2)
    next_value = torch.cat([value[1:], value[-1:]], dim=0)
    next_value[3, 0] = 50.0  # final observation at a time-limit truncation
    next_value[3, 1] = 70.0  # ignored at a true termination
    boundaries = torch.zeros(steps, 2)
    boundaries[3] = 1.0
    terminations = torch.zeros_like(boundaries)
    terminations[3, 1] = 1.0
    valid = torch.ones_like(boundaries)
    one_step = MODULE.successor_innovation(
        reward.unsqueeze(-1),
        ((1.0 - gamma) * value).unsqueeze(-1),
        ((1.0 - gamma) * next_value).unsqueeze(-1),
        terminations,
        valid,
        gamma,
    )
    credit = MODULE.fixed_prefix_credit(one_step, boundaries, gamma, credit_steps=32)
    prefix_reward = sum(gamma**i * reward[i, 0] for i in range(4))
    expected_truncation = prefix_reward + gamma**4 * 50.0 - value[0, 0]
    expected_termination = prefix_reward - value[0, 1]
    torch.testing.assert_close(credit[0, 0, 0], expected_truncation)
    torch.testing.assert_close(credit[0, 1, 0], expected_termination)


def test_prefix_at_rollout_tail_keeps_last_observed_next_state_bootstrap():
    gamma = 0.8
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    value = torch.tensor([[5.0], [7.0], [11.0], [13.0]])
    one_step = MODULE.successor_innovation(
        reward.unsqueeze(-1),
        ((1.0 - gamma) * value[:-1]).unsqueeze(-1),
        ((1.0 - gamma) * value[1:]).unsqueeze(-1),
        torch.zeros(3, 1),
        torch.ones(3, 1),
        gamma,
    )
    credit = MODULE.fixed_prefix_credit(
        one_step, torch.zeros(3, 1), gamma, credit_steps=32
    )
    expected = reward[1, 0] + gamma * reward[2, 0] + gamma**2 * value[3, 0] - value[1, 0]
    torch.testing.assert_close(credit[1, 0, 0], expected)


def test_explicit_midpoint_solver_uses_midpoint_state():
    class ExponentialFlow:
        @staticmethod
        def velocity(time_, x, obs):
            return x

    x0 = torch.ones(3, 1)
    end_time = torch.ones(3, 1)
    actual = MODULE.rk2_midpoint_flow(
        ExponentialFlow(), torch.zeros(3, 1), x0, end_time, steps=4
    )
    dt = 0.25
    expected = torch.full_like(x0, (1.0 + dt + 0.5 * dt * dt) ** 4)
    torch.testing.assert_close(actual, expected)
    assert actual.mean() > (1.0 + dt) ** 4


def test_geometric_future_sampling_never_crosses_boundaries():
    boundaries = torch.tensor(
        [[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]
    )
    offsets = torch.tensor([[1], [1], [2], [2], [0], [0]])
    indices, valid = MODULE.sample_geometric_future_indices(
        boundaries, gamma=0.9, offsets=offsets
    )
    torch.testing.assert_close(indices.squeeze(1), torch.tensor([1, 2, 4, 5, 4, 5]))
    assert valid.squeeze(1).tolist() == [True, False, True, False, True, True]


def test_geometric_future_uses_zero_absorbing_outcome_after_true_terminal():
    outcomes = torch.arange(1.0, 7.0).reshape(6, 1, 1)
    boundaries = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0]])
    terminations = torch.tensor([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    valids = torch.ones(6, 1)
    offsets = torch.tensor([[2], [1], [2], [1], [2], [0]])
    future, valid = MODULE.geometric_future_outcomes(
        outcomes,
        boundaries,
        terminations,
        valids,
        gamma=0.9,
        offsets=offsets,
    )
    # t=0 and t=1 cross the true terminal at t=1 and therefore sample the
    # absorbing zero. t=2 and t=3 cross a truncation and remain censored.
    torch.testing.assert_close(future[:2], torch.zeros(2, 1, 1))
    assert valid.squeeze(1).tolist() == [True, True, False, False, False, True]


def test_outcome_has_detached_next_embedding_action_square_and_reward():
    embedding = torch.randn(4, 3, requires_grad=True)
    action = torch.randn(4, 2)
    reward = torch.randn(4)
    outcome = MODULE.outcome_features(embedding, action, reward)
    assert outcome.shape == (4, 8)
    torch.testing.assert_close(outcome[:, :3], embedding.detach())
    torch.testing.assert_close(outcome[:, 3:5], action)
    torch.testing.assert_close(outcome[:, 5:7], action.square())
    torch.testing.assert_close(outcome[:, -1], reward)
    assert not outcome[:, :3].requires_grad


def test_fresh_block_scaling_is_isotropic_inside_embedding_block():
    torch.manual_seed(5)
    outcomes = torch.randn(37, 8)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = outcomes.clone()
    rotated[:, :3] = outcomes[:, :3] @ q
    scales = MODULE.fresh_block_scales(outcomes, emb_dim=3, act_dim=2)
    rotated_scales = MODULE.fresh_block_scales(rotated, emb_dim=3, act_dim=2)
    torch.testing.assert_close(scales, rotated_scales, atol=1e-6, rtol=1e-6)


def test_frame_transport_recovers_rotation_and_translation():
    torch.manual_seed(7)
    before = torch.randn(128, 6)
    rotation, _ = torch.linalg.qr(torch.randn(6, 6))
    translation = torch.randn(6)
    after = before @ rotation + translation
    current_rotation = torch.eye(6)
    current_offset = torch.zeros(6)
    transport, offset = MODULE.affine_frame_transport(
        after, before, current_rotation, current_offset
    )
    actual = after @ transport + offset
    expected = before
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_source_has_fixed_prefix_joint_ppo_and_mc_only_ghm():
    source = SCRIPT.read_text()
    assert "one_step_credits = successor_innovation(" in source
    assert "reward_credit = fixed_prefix_credit(" in source
    assert "vector_credit_raw = fixed_prefix_credit(" in source
    assert "loss_unclipped = -b_adv[mb] * ratio" in source
    assert "outcome_proximal_loss(logratio, b_vector_credit[mb])" in source
    assert "hard_update(ghm_target, ghm)" in source
    assert "credit_steps: int = 32" in source
    assert "td2_cfm_loss(" not in source
    assert "flow_mode:" not in source
    assert "gae_lambda" not in source
    assert "lastgaelam" not in source
    assert "replay_buffer" not in source.lower()
    assert "ReplayBuffer" not in source
    assert tuple(inspect.signature(MODULE.GeometricHorizonModel.mean).parameters) == (
        "self",
        "obs",
    )
    assert tuple(inspect.signature(MODULE.GeometricHorizonModel.velocity).parameters) == (
        "self",
        "time_",
        "x",
        "obs",
    )
