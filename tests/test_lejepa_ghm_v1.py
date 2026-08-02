import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_ghm_v1.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_ghm_v1", SCRIPT)
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


def test_mc_phase_directly_supervises_successor_mean_with_geometric_outcome():
    mc_future = torch.tensor([[3.0, 4.0]])
    mc_valid = torch.tensor([True])
    bellman_outcome = torch.tensor([[100.0, 200.0]])
    next_mean = torch.tensor([[50.0, 60.0]])
    target, valid = MODULE.mean_regression_target(
        "td2_cfm",
        iteration=2,
        warmup_iterations=4,
        mc_future=mc_future,
        mc_valid=mc_valid,
        outcome=bellman_outcome,
        next_mean=next_mean,
        termination=torch.zeros(1),
        valid=torch.ones(1),
        gamma=0.99,
    )
    torch.testing.assert_close(target, mc_future)
    torch.testing.assert_close(valid, mc_valid)


def test_td2_phase_uses_hard_bellman_mean_target_after_warmup():
    outcome = torch.tensor([[2.0]])
    next_mean = torch.tensor([[3.0]])
    target, valid = MODULE.mean_regression_target(
        "td2_cfm",
        iteration=5,
        warmup_iterations=4,
        mc_future=torch.tensor([[99.0]]),
        mc_valid=torch.tensor([True]),
        outcome=outcome,
        next_mean=next_mean,
        termination=torch.zeros(1),
        valid=torch.ones(1),
        gamma=0.9,
    )
    torch.testing.assert_close(target, torch.tensor([[2.9]]))
    torch.testing.assert_close(valid, torch.ones(1))


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


def test_td2_flow_target_is_hard_frozen():
    torch.manual_seed(3)
    online = MODULE.GeometricHorizonModel(3, 5, 16)
    target = MODULE.GeometricHorizonModel(3, 5, 16)
    MODULE.hard_update(target, online)
    obs = torch.randn(8, 3)
    next_obs = torch.randn(8, 3)
    outcomes = torch.randn(8, 5)
    termination = torch.zeros(8)
    valid = torch.ones(8)
    scales = torch.ones(4)
    loss, _, _ = MODULE.td2_cfm_loss(
        online,
        target,
        obs,
        next_obs,
        outcomes,
        termination,
        valid,
        gamma=0.9,
        scales=scales,
        emb_dim=2,
        act_dim=1,
        flow_steps=3,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in online.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(not parameter.requires_grad for parameter in target.parameters())


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


def test_source_has_single_innovation_joint_ppo_and_no_trace_or_q_condition():
    source = SCRIPT.read_text()
    assert "credits = successor_innovation(" in source
    assert "loss_unclipped = -b_adv[mb] * ratio" in source
    assert "outcome_proximal_loss(logratio, b_vector_credit[mb])" in source
    assert "hard_update(ghm_target, ghm)" in source
    assert "args.flow_mode == \"mc_cfm\"" in source
    assert "td2_cfm_loss(" in source
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
