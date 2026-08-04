import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_detached_intact_world_direct_v11.py"
)
SPEC = importlib.util.spec_from_file_location(
    "detached_intact_world_direct_v11", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args(**overrides):
    values = dict(
        latent_dim=4,
        hidden_dim=16,
        model_horizon=3,
        chunk_horizon=3,
        controller_horizon=3,
        intact_fixed_std=0.2,
    )
    values.update(overrides)
    return MODULE.Args(**values)


def make_batch(batch_size=8, horizon=3):
    observations = torch.randn(batch_size, horizon + 1, 17)
    return MODULE.ReplayBatch(
        observations=observations,
        actions=torch.randn(batch_size, horizon, 2).clamp(-1.0, 1.0),
        previous_actions=torch.randn(batch_size, horizon, 2).clamp(-1.0, 1.0),
        interval_velocities=torch.randn(batch_size, horizon),
        valid=torch.ones(batch_size, horizon, dtype=torch.bool),
    )


def clear_gradients(agent):
    for parameter in agent.parameters():
        parameter.grad = None


def has_nonzero_gradient(parameters):
    return any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


def has_any_gradient(parameters):
    return any(parameter.grad is not None for parameter in parameters)


def test_chunk_law_has_fixed_std_and_first_action_weighting():
    law = MODULE.IntentActionLaw(2, 1, 3, 8, fixed_std=1.0, tail_weight=0.1)
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.zero_()
    z = torch.zeros(1, 2)
    previous = torch.zeros(1, 1)
    mean, log_std = law.parameters_for(z, z, previous)
    assert mean.shape == log_std.shape == (1, 3)
    torch.testing.assert_close(log_std, torch.zeros_like(log_std))
    baseline = law.nll(z, z, previous, torch.zeros(1, 3, 1))
    first_error = torch.zeros(1, 3, 1)
    first_error[:, 0] = 1.0
    tail_error = torch.zeros(1, 3, 1)
    tail_error[:, 1:] = 1.0
    first_penalty = law.nll(z, z, previous, first_error) - baseline
    tail_penalty = law.nll(z, z, previous, tail_error) - baseline
    torch.testing.assert_close(first_penalty, 10.0 * tail_penalty)


def test_objective_intent_is_direction_normalized_to_factual_h12_rms_norm():
    agent = MODULE.Agent(17, 2, small_args())
    factual_intents = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    agent.update_factual_intent_scale(factual_intents, ema=0.9)
    expected_rms_norm = factual_intents.square().sum(dim=-1).mean().sqrt()
    intents = agent.objective_intent(5)
    torch.testing.assert_close(
        intents.norm(dim=-1), expected_rms_norm.expand(5)
    )
    assert not agent.factual_intent_mean_square.requires_grad
    second = torch.full((2, 4), 2.0)
    agent.update_factual_intent_scale(second, ema=0.5)
    expected_mean_square = 0.5 * expected_rms_norm.square() + 0.5 * 16.0
    torch.testing.assert_close(
        agent.factual_intent_mean_square, expected_mean_square
    )


def test_world_objective_gradients_only_world_model_partition():
    agent = MODULE.Agent(17, 2, small_args())
    loss, metrics = MODULE.world_model_objective(
        agent,
        MODULE.SIGReg(projections=8, knots=5),
        make_batch(),
        small_args(),
    )
    loss.backward()
    assert has_nonzero_gradient(MODULE.world_model_parameters(agent))
    assert not has_any_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.objective_token_parameters(agent))
    assert set(metrics) == {
        "wm_loss",
        "wm_forward_loss",
        "wm_sigreg_loss",
        "wm_state_velocity_huber",
        "wm_factual_interval_huber",
        "wm_predicted_interval_huber",
    }


def test_detached_physical_intact_gradients_only_shared_law():
    agent = MODULE.Agent(17, 2, small_args())
    factual_z = torch.randn(8, 4, 4, requires_grad=True)
    previous_action = torch.randn(8, 2, requires_grad=True)
    action_chunk = torch.randn(8, 3, 2, requires_grad=True)
    loss, metrics = MODULE.detached_intact_objective(
        agent, factual_z, previous_action, action_chunk
    )
    loss.backward()
    assert has_nonzero_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.world_model_parameters(agent))
    assert not has_any_gradient(MODULE.objective_token_parameters(agent))
    assert factual_z.grad is None
    assert previous_action.grad is None
    assert action_chunk.grad is None
    torch.testing.assert_close(
        metrics["fixed_std"], torch.tensor(small_args().intact_fixed_std)
    )
    assert torch.isfinite(metrics["detached_local_nll"])


def test_combined_controller_gradients_law_and_token_but_not_world_model():
    torch.manual_seed(5)
    args = small_args(controller_horizon=3)
    agent = MODULE.Agent(17, 2, args)
    batch = make_batch()
    with torch.no_grad():
        factual_z = agent.encode(batch.observations)
    loss, metrics = MODULE.combined_controller_objective(
        agent,
        factual_z,
        batch.previous_actions[:, 0],
        batch.actions,
        args,
    )
    loss.backward()
    token = agent.forward_objective_intent
    assert token.grad is not None and token.grad.abs().sum() > 0
    assert has_nonzero_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.world_model_parameters(agent))
    assert not factual_z.requires_grad
    assert torch.isfinite(metrics["controller_loss"])
    assert metrics["objective_token_use"] > 0


def test_direct_reward_itself_gradients_law_and_token_and_is_horizon_normalized():
    torch.manual_seed(6)
    args = small_args(controller_horizon=3)
    agent = MODULE.Agent(17, 2, args)
    z_start = torch.randn(8, 4, requires_grad=True)
    previous_action = torch.randn(8, 2, requires_grad=True)
    loss, metrics = MODULE.direct_controller_objective(
        agent, z_start, previous_action, args
    )
    loss.backward()
    assert has_nonzero_gradient(MODULE.intact_law_parameters(agent))
    assert has_nonzero_gradient(MODULE.objective_token_parameters(agent))
    assert not has_any_gradient(MODULE.world_model_parameters(agent))
    assert z_start.grad is None and previous_action.grad is None
    torch.testing.assert_close(
        loss.detach() * args.controller_horizon,
        -metrics["imagined_score"],
    )


def test_frozen_world_reward_has_nonzero_action_credit_without_model_grads():
    torch.manual_seed(7)
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(8, 4)
    action = torch.randn(8, 2, requires_grad=True)
    successor, velocity, cost = MODULE._frozen_world_step(
        agent,
        z,
        action,
        0.1,
        MODULE._detached_module_state(agent.dynamics),
        MODULE._detached_module_state(agent.interval_velocity_head),
    )
    assert successor.shape == z.shape
    action_gradient = torch.autograd.grad((velocity - cost).sum(), action)[0]
    assert action_gradient.abs().sum() > 0
    assert not has_any_gradient(MODULE.world_model_parameters(agent))


def test_two_optimizer_partitions_are_disjoint_and_complete():
    agent = MODULE.Agent(17, 2, small_args())
    groups = [
        {id(parameter) for parameter in parameters}
        for parameters in (
            MODULE.world_model_parameters(agent),
            MODULE.controller_parameters(agent),
        )
    ]
    assert not groups[0] & groups[1]
    assert set.union(*groups) == {id(parameter) for parameter in agent.parameters()}


def test_closed_loop_score_has_exact_h1_h6_h12_lengths_and_reward_sum():
    torch.manual_seed(7)
    args = small_args(model_horizon=12, chunk_horizon=12)
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(4, 4)
    previous = torch.randn(4, 2)
    for horizon in (1, 6, 12):
        score, velocities, costs, raw_actions, actions = (
            MODULE.closed_loop_direct_score(agent, z, previous, horizon, 0.1)
        )
        assert score.shape == (4,)
        assert velocities.shape == costs.shape == (4, horizon)
        assert raw_actions.shape == actions.shape == (4, horizon, 2)
        torch.testing.assert_close(score, (velocities - costs).sum(dim=1))
        assert torch.all(actions.abs() <= 1.0)


def test_one_step_imagination_matches_runtime_action_and_successor_timing():
    torch.manual_seed(11)
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    runtime_action, _, _ = MODULE.direct_action_chunk(agent, z, previous)
    score, velocities, costs, _, imagined_actions = MODULE.closed_loop_direct_score(
        agent, z, previous, horizon=1, action_cost_coef=0.1
    )
    torch.testing.assert_close(imagined_actions[:, 0], runtime_action)
    successor = agent.predict_next(z, runtime_action)
    expected_velocity = agent.predict_interval_velocity(successor)
    expected_cost = 0.1 * runtime_action.square().sum(dim=-1)
    torch.testing.assert_close(velocities[:, 0], expected_velocity)
    torch.testing.assert_close(costs[:, 0], expected_cost)
    torch.testing.assert_close(score, expected_velocity - expected_cost)


def test_runtime_direct_controller_is_deterministic_bounded_and_one_law_call():
    torch.manual_seed(13)
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(4, 4)
    previous = torch.randn(4, 2)
    original = agent.intact_action_law.parameters_for
    call_count = 0

    def counted_parameters_for(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    agent.intact_action_law.parameters_for = counted_parameters_for
    first_action, first_chunk, first_metrics = MODULE.direct_action_chunk(
        agent, z, previous
    )
    assert call_count == 1
    call_count = 0
    second_action, second_chunk, second_metrics = MODULE.direct_action_chunk(
        agent, z, previous
    )
    assert call_count == 1
    torch.testing.assert_close(first_action, second_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_chunk, second_chunk, rtol=0.0, atol=0.0)
    assert torch.all(first_chunk.abs() <= 1.0)
    assert first_metrics.keys() == second_metrics.keys()
    source = inspect.getsource(MODULE.direct_action_chunk).lower()
    for forbidden in ("for ", "while ", "rand", "sample", "rollout", "search"):
        assert forbidden not in source


def test_closed_loop_training_replans_and_recurrs_previous_action():
    source = inspect.getsource(MODULE.closed_loop_direct_score)
    assert "for _ in range(horizon):" in source
    assert "[:, 0]" in source
    assert "previous_action = action" in source
    assert "predicted_z, interval_velocity, action_cost = _frozen_world_step(" in source
    assert "agent.intact_action_law.parameters_for(" in source
    assert "_detached_module_state(agent.intact_action_law)" not in source
    step_source = inspect.getsource(MODULE._frozen_world_step)
    assert step_source.index("successor_z = predicted_z + delta") < step_source.index(
        "interval_velocity = functional_call("
    )


def test_executable_has_only_random_warmup_and_no_old_command_or_search_path():
    source = SCRIPT.read_text()
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    assert "if global_step < args.warmup_steps" in source
    for forbidden in (
        "VelocityCommandTracker",
        "velocity_goal_encoder",
        "command_normalizer",
        "target_velocity",
        "goal_condition",
        "sample_cem_action_sequences",
        "rti_action_sequence",
        "optimize_action_sequence",
        "terminal_value",
        "trust_region",
        "law_optimizer",
        "token_optimizer",
        "_detached_module_state(agent.intact_action_law)",
    ):
        assert forbidden not in source
    main = source[source.index('if __name__ == "__main__":') :]
    assert "direct_action_chunk(\n                    agent,\n                    z,\n                    previous_action," in main
    assert "closed_loop_direct_score(" not in main


def test_exact_interval_velocity_merges_live_and_final_infos():
    infos = {
        "x_velocity": np.asarray([1.25, 0.0, -0.5], dtype=np.float32),
        "_x_velocity": np.asarray([True, False, True]),
        "final_info": np.asarray(
            [None, {"x_velocity": 2.75}, None], dtype=object
        ),
        "_final_info": np.asarray([False, True, False]),
    }
    np.testing.assert_array_equal(
        MODULE.exact_interval_velocities(infos, 3),
        np.asarray([1.25, 2.75, -0.5], dtype=np.float32),
    )


def test_exact_interval_velocity_handles_simultaneous_time_limits():
    infos = {
        "final_info": np.asarray(
            [{"x_velocity": 1.0}, {"x_velocity": -2.0}], dtype=object
        ),
        "_final_info": np.asarray([True, True]),
    }
    np.testing.assert_array_equal(
        MODULE.exact_interval_velocities(infos, 2),
        np.asarray([1.0, -2.0], dtype=np.float32),
    )


def _add_transition(replay, step, done=False):
    replay.add(
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step - 1)]]),
        torch.tensor([[float(step + 1)]]),
        torch.tensor([float(step) + 0.25]),
        torch.tensor([done]),
    )


def test_replay_preserves_exact_labels_episode_seams_and_wraparound():
    replay = MODULE.SequenceReplayBuffer(6, 1, 1, 1, device="cpu")
    for step in range(8):
        _add_transition(replay, step, done=(step == 4))
    batch = replay.sample(128, 2, "cpu")
    differences = batch.observations[:, 1:, 0] - batch.observations[:, :-1, 0]
    torch.testing.assert_close(differences, torch.ones_like(differences))
    torch.testing.assert_close(
        batch.interval_velocities, batch.actions[..., 0] + 0.25
    )
    assert torch.all(batch.observations[:, 0, 0] >= 2.0)


def test_replay_recent_sampling_uses_recent_logical_starts():
    replay = MODULE.SequenceReplayBuffer(100, 1, 1, 1, device="cpu")
    for step in range(100):
        _add_transition(replay, step)
    batch = replay.sample(
        256,
        model_horizon=2,
        device="cpu",
        recent_fraction=1.0,
        recent_steps=10,
    )
    assert torch.all(batch.observations[:, 0, 0] >= 89.0)


def test_factual_terminal_observation_replaces_autoreset_state():
    autoreset = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    infos = {
        "final_observation": np.asarray(
            [np.asarray([9.0, 10.0], dtype=np.float32), None], dtype=object
        ),
        "_final_observation": np.asarray([True, False]),
    }
    factual = MODULE.factual_transition_observations(autoreset, infos)
    np.testing.assert_array_equal(factual[0], np.asarray([9.0, 10.0]))
    np.testing.assert_array_equal(factual[1], autoreset[1])


def test_sigreg_remains_time_major_and_batch_independent():
    sigreg = MODULE.SIGReg(projections=1, knots=5)
    latents = torch.tensor(
        [[[0.0], [0.5], [1.0], [1.5]], [[-1.0], [0.25], [9.0], [9.0]]]
    )
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    combined = sigreg(latents, mask)
    separate = torch.stack(
        [sigreg(latents[:1]), sigreg(latents[1:2, :2])]
    ).mean()
    torch.testing.assert_close(combined, separate)


def test_defaults_are_h12_direct_with_two_isolated_optimizers():
    args = MODULE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert args.model_horizon == args.chunk_horizon == args.controller_horizon == 12
    assert args.intact_fixed_std == 0.2 and args.chunk_tail_weight == 0.1
    assert args.detached_intact_coef == 0.1
    assert args.direct_objective_coef == 1.0
    assert args.controller_lr == 5e-4


def test_main_has_two_optimizers_one_combined_backward_and_metric_clones():
    source = SCRIPT.read_text()
    for name in ("wm_optimizer", "controller_optimizer"):
        assert f"{name} = optim.AdamW(" in source
    assert "controller_loss_function = combined_controller_objective" in source
    assert source.count("controller_loss.backward()") == 1
    assert source.count("assert_parameters_have_no_gradients(") >= 3
    assert source.count("torch.compiler.cudagraph_mark_step_begin()") == 2
    assert source.count("key: value.clone() for key, value in") == 2
    assert "clip_grad_norm_(law_parameters" in source
    assert "clip_grad_norm_(token_parameters" in source
    assert "clip_grad_norm_(controller_params" not in source
