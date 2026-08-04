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
    / "ppo_continuous_action_counterfactual_goal_intact_direct_v7.py"
)
SPEC = importlib.util.spec_from_file_location(
    "counterfactual_goal_intact_direct_v7", SCRIPT
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
        intact_fixed_std=0.2,
    )
    values.update(overrides)
    return MODULE.Args(**values)


def make_batch(batch_size=8, horizon=3):
    return MODULE.ReplayBatch(
        observations=torch.randn(batch_size, horizon + 1, 17),
        actions=torch.randn(batch_size, horizon, 2).clamp(-1.0, 1.0),
        previous_actions=torch.randn(batch_size, horizon, 2).clamp(-1.0, 1.0),
        interval_velocities=torch.randn(batch_size, horizon),
        valid=torch.ones(batch_size, horizon, dtype=torch.bool),
    )


def test_counterfactual_changes_only_forward_velocity_index():
    observation = torch.randn(4, 17)
    observation[:, 8] = torch.tensor([0.5, 7.5, 8.5, -2.0])
    original = observation.clone()
    goal, desired = MODULE.counterfactual_velocity_observation(
        observation, target_velocity_delta=1.0, target_velocity_max=8.0
    )
    torch.testing.assert_close(desired, torch.tensor([1.5, 8.0, 8.0, -1.0]))
    torch.testing.assert_close(goal[:, 8], desired)
    indices = [index for index in range(17) if index != 8]
    torch.testing.assert_close(goal[:, indices], original[:, indices])
    torch.testing.assert_close(observation, original)
    assert goal.data_ptr() != observation.data_ptr()


def test_chunk_law_has_h12_shape_fixed_variance_and_shared_routes():
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(5, 4)
    local_intent = torch.randn(5, 4)
    goal_intent = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    local_mean, local_log_std = agent.intact_action_law.parameters_for(
        z, local_intent, previous
    )
    goal_mean, goal_log_std = agent.intact_action_law.parameters_for(
        z, goal_intent, previous
    )
    assert local_mean.shape == goal_mean.shape == (5, 6)
    expected = torch.full_like(local_log_std, np.log(0.2))
    torch.testing.assert_close(local_log_std, expected)
    torch.testing.assert_close(goal_log_std, expected)
    source = inspect.getsource(MODULE.chunk_intact_nll_losses)
    assert source.count("agent.intact_action_law.nll") == 2


def test_chunk_nll_makes_first_action_primary_and_tail_auxiliary():
    law = MODULE.IntentActionLaw(2, 1, 3, 8, fixed_std=1.0, tail_weight=0.1)
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.zero_()
    z = torch.zeros(1, 2)
    previous = torch.zeros(1, 1)
    baseline = law.nll(z, z, previous, torch.zeros(1, 3, 1))
    first_error = torch.zeros(1, 3, 1)
    first_error[:, 0] = 1.0
    tail_error = torch.zeros(1, 3, 1)
    tail_error[:, 1:] = 1.0
    first_penalty = law.nll(z, z, previous, first_error) - baseline
    tail_penalty = law.nll(z, z, previous, tail_error) - baseline
    torch.testing.assert_close(first_penalty, 10.0 * tail_penalty)


def test_local_endpoint_attaches_but_goal_future_endpoint_is_detached():
    agent = MODULE.Agent(17, 2, small_args())
    z_start = torch.randn(6, 4, requires_grad=True)
    z_future = torch.randn(6, 4, requires_grad=True)
    previous = torch.randn(6, 2)
    chunk = torch.randn(6, 3, 2)
    local, goal = MODULE.chunk_intact_nll_losses(
        agent, z_start, z_future, previous, chunk
    )
    local_grads = torch.autograd.grad(
        local.sum(), (z_start, z_future), retain_graph=True
    )
    assert all(gradient.abs().sum() > 0 for gradient in local_grads)
    goal_grads = torch.autograd.grad(
        goal.sum(), (z_start, z_future), allow_unused=True, retain_graph=True
    )
    assert goal_grads[0] is not None and goal_grads[0].abs().sum() > 0
    assert goal_grads[1] is None

    shared_parameter = next(agent.intact_action_law.predictor.parameters())
    local_parameter_grad = torch.autograd.grad(
        local.sum(), shared_parameter, retain_graph=True
    )[0]
    goal_parameter_grad = torch.autograd.grad(goal.sum(), shared_parameter)[0]
    assert local_parameter_grad.abs().sum() > 0
    assert goal_parameter_grad.abs().sum() > 0


def test_direct_counterfactual_path_is_deterministic_bounded_and_search_free():
    torch.manual_seed(1)
    agent = MODULE.Agent(17, 2, small_args())
    observation = torch.randn(4, 17)
    previous = torch.randn(4, 2)
    first_action, first_chunk, first_metrics = MODULE.direct_action_chunk(
        agent, observation, previous, 1.0, 8.0
    )
    second_action, second_chunk, second_metrics = MODULE.direct_action_chunk(
        agent, observation, previous, 1.0, 8.0
    )
    torch.testing.assert_close(first_action, second_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_chunk, second_chunk, rtol=0.0, atol=0.0)
    assert first_action.shape == (4, 2)
    assert first_chunk.shape == (4, 3, 2)
    assert torch.all(first_chunk.abs() <= 1.0)
    assert set(first_metrics) == set(second_metrics)
    torch.testing.assert_close(
        first_metrics["desired_velocity"],
        (observation[:, 8] + 1.0).clamp(max=8.0),
    )
    assert torch.all(first_metrics["goal_intent_norm"] >= 0.0)
    source = inspect.getsource(MODULE.direct_action_chunk).lower()
    for forbidden in ("rand", "sample", "cem", "optim", "terminal", "proposer"):
        assert forbidden not in source


def test_executable_has_only_random_warmup_and_no_postwarm_exploration():
    source = SCRIPT.read_text()
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    assert "if global_step < args.warmup_steps" in source
    for forbidden in (
        "random_action_probability",
        "plan_action_sequence",
        "optimize_action_sequence",
        "sample_cem_action_sequences",
        "VelocityCommandTracker",
        "velocity_goal_encoder",
    ):
        assert forbidden not in source


def test_training_updates_world_model_velocity_heads_and_shared_chunk_law():
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    batch = make_batch()
    loss, metrics = MODULE.training_objective(
        agent, MODULE.SIGReg(projections=8, knots=5), batch, args
    )
    loss.backward()
    for component in (
        agent.encoder,
        agent.dynamics,
        agent.velocity_head,
        agent.interval_velocity_head,
        agent.intact_action_law,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in component.parameters()
        )
    for key in (
        "local_nll",
        "goal_nll",
        "local_mean_mae",
        "goal_mean_mae",
        "local_first_action_mae",
        "goal_first_action_mae",
        "goal_condition_use",
        "goal_intent_norm",
    ):
        assert key in metrics and torch.isfinite(metrics[key])


def test_sigreg_is_time_major_and_computes_each_batch_independently():
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


def test_exact_interval_velocity_merges_live_and_autoreset_final_info():
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


def test_replay_preserves_exact_labels_episode_boundaries_and_wraparound():
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


def test_replay_recent_sampling_selects_only_recent_logical_starts():
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


def test_defaults_are_h12_counterfactual_direct():
    args = MODULE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert args.model_horizon == args.chunk_horizon == 12
    assert args.local_nll_coef == 0.1 and args.goal_nll_coef == 0.05
    assert args.intact_fixed_std == 0.2 and args.chunk_tail_weight == 0.1
    assert args.target_velocity_delta == 1.0
    assert args.target_velocity_max == 8.0
    assert not hasattr(args, "target_velocity_quantile")


def test_compiled_metrics_are_cloned_before_cuda_graph_reuse():
    source = SCRIPT.read_text()
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "{key: value.clone() for key, value in metrics.items()}" in source
