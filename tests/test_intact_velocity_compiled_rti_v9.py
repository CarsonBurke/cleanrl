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
    / "ppo_continuous_action_intact_velocity_compiled_rti_v9.py"
)
SPEC = importlib.util.spec_from_file_location("intact_velocity_compiled_rti_v9", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

V8_SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_intact_velocity_rti_v8.py"
)
V8_SPEC = importlib.util.spec_from_file_location("intact_velocity_rti_v8", V8_SCRIPT)
V8_MODULE = importlib.util.module_from_spec(V8_SPEC)
sys.modules[V8_SPEC.name] = V8_MODULE
V8_SPEC.loader.exec_module(V8_MODULE)


def small_args(**overrides):
    values = dict(latent_dim=4, hidden_dim=16, model_horizon=3, planner_horizon=3)
    values.update(overrides)
    return MODULE.Args(**values)


def make_batch(batch_size=8, horizon=3):
    observations = torch.randn(batch_size, horizon + 1, 17)
    return MODULE.ReplayBatch(
        observations=observations,
        actions=torch.randn(batch_size, horizon, 2),
        previous_actions=torch.randn(batch_size, horizon, 2),
        interval_velocities=observations[:, :-1, 8] + 0.2 * torch.randn(
            batch_size, horizon
        ),
        valid=torch.ones(batch_size, horizon, dtype=torch.bool),
    )


def test_agent_decomposes_interval_velocity_into_state_and_action_delta():
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(5, 4)
    action = torch.randn(5, 2)
    expected = agent.predict_velocity(z) + agent.predict_forward_delta(z, action)
    torch.testing.assert_close(agent.predict_interval_velocity(z, action), expected)


def test_forward_delta_label_uses_exact_interval_velocity():
    observations = torch.zeros(3, 17)
    observations[:, 8] = torch.tensor([1.0, -0.5, 2.0])
    interval = torch.tensor([1.4, 0.25, 1.0])
    torch.testing.assert_close(
        MODULE.forward_delta_target(observations, interval),
        torch.tensor([0.4, 0.75, -1.0]),
    )


class QuadraticIntervalModel:
    def __init__(self, target):
        self.target = torch.as_tensor(target)

    def predict_interval_velocity(self, z, action):
        return 3.0 - (action - self.target.to(action)).square().sum(dim=-1)

    def predict_next(self, z, action):
        return z


def test_h12_score_sums_exact_interval_reward_and_action_cost():
    model = QuadraticIntervalModel([0.5, -0.25])
    z = torch.zeros(1, 1)
    actions = torch.tensor([[[0.5, -0.25]]]).expand(-1, 12, -1)
    score, velocities = MODULE.action_sequence_score(model, z, actions, 0.1)
    torch.testing.assert_close(score, torch.tensor([35.625]))
    torch.testing.assert_close(velocities, torch.full((1, 12), 3.0))


def test_h12_score_backpropagates_future_rewards_through_dynamics():
    agent = MODULE.Agent(17, 2, small_args(model_horizon=12, planner_horizon=12))
    z = torch.randn(5, 4)
    actions = torch.randn(5, 12, 2, requires_grad=True)
    score, _ = MODULE.action_sequence_score(agent, z, actions)
    score.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.dynamics.parameters()
    )
    assert actions.grad[:, 0].abs().sum() > 0


def test_rti_takes_one_sign_step_guards_per_env_and_calls_scorer_twice():
    targets = torch.tensor([0.5, 0.05, 2.0]).view(3, 1, 1)
    calls = []

    def score_function(agent, z, sequence, action_cost_coef):
        calls.append(sequence.detach().clone())
        score = -(sequence - targets).square().sum(dim=(-1, -2))
        velocities = sequence[..., 0]
        return score, velocities

    args = small_args(rti_step_size=0.16, action_cost_coef=0.0)
    warm = torch.zeros(3, 3, 1)
    warm[2] = 0.95
    selected, metrics = MODULE.rti_action_sequence(
        None, torch.zeros(3, 1), warm, args, score_function
    )
    assert len(calls) == 2
    torch.testing.assert_close(selected[0], torch.full((3, 1), 0.16))
    torch.testing.assert_close(selected[1], warm[1])
    torch.testing.assert_close(selected[2], torch.ones(3, 1))
    torch.testing.assert_close(
        metrics["accept_fraction"], torch.tensor([1.0, 0.0, 1.0])
    )
    assert torch.all(metrics["optimizer_improvement"] >= 0.0)
    assert torch.all((selected >= -1.0) & (selected <= 1.0))
    assert torch.all(metrics["gradient_sign_active_fraction"] == 1.0)


def test_rti_is_deterministic_and_contains_no_search_loop_rng_or_action_adam():
    targets = torch.full((2, 1, 1), 0.5)

    def score_function(agent, z, sequence, action_cost_coef):
        return (
            -(sequence - targets).square().sum(dim=(-1, -2)),
            sequence[..., 0],
        )

    args = small_args(rti_step_size=0.16, action_cost_coef=0.0)
    warm = torch.zeros(2, 3, 1)
    first, first_metrics = MODULE.rti_action_sequence(
        None, torch.zeros(2, 1), warm, args, score_function
    )
    second, second_metrics = MODULE.rti_action_sequence(
        None, torch.zeros(2, 1), warm, args, score_function
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_metrics["selected_score"],
        second_metrics["selected_score"],
        rtol=0.0,
        atol=0.0,
    )
    source = inspect.getsource(MODULE.rti_action_sequence).lower()
    for forbidden in ("for ", "while ", "rand", "sample", "cem", "adam", "moment"):
        assert forbidden not in source
    assert source.count("score_function(") == 2


def test_eager_rti_is_numerically_identical_to_validated_v8():
    targets = torch.tensor([0.4, -0.3, 1.2]).view(3, 1, 1)

    def score_function(agent, z, sequence, action_cost_coef):
        velocities = 2.0 - (sequence - targets).square().sum(dim=-1)
        score = velocities.sum(dim=-1) - action_cost_coef * sequence.square().sum(
            dim=(-1, -2)
        )
        return score, velocities

    args = small_args(rti_step_size=0.20, action_cost_coef=0.1)
    z = torch.randn(3, 4)
    warm = torch.tensor(
        [
            [[0.0], [0.2], [-0.1]],
            [[-0.2], [-0.1], [0.0]],
            [[0.9], [0.95], [1.0]],
        ]
    )
    expected_sequence, expected_diagnostics = V8_MODULE.rti_action_sequence(
        None, z, warm, args, score_function
    )
    actual_sequence, actual_diagnostics = MODULE.rti_action_sequence(
        None,
        z,
        warm,
        args,
        score_function=score_function,
        compiled_score=False,
    )
    torch.testing.assert_close(actual_sequence, expected_sequence)
    assert actual_diagnostics.keys() == expected_diagnostics.keys()
    for key in actual_diagnostics:
        torch.testing.assert_close(
            actual_diagnostics[key], expected_diagnostics[key]
        )


def test_compiled_rti_snapshots_graph_outputs_before_second_scorer_call():
    source = inspect.getsource(MODULE.rti_action_sequence)
    first_call = source.index("score_function(")
    second_call = source.rindex("score_function(")
    gradient = source.index("torch.autograd.grad")
    score_snapshot = source.index("initial_score.detach().clone()")
    velocity_snapshot = source.index("initial_velocities.detach().clone()")
    gradient_snapshot = source.index("gradient.detach().clone()")
    first_mark = source.index("torch.compiler.cudagraph_mark_step_begin()")
    second_mark = source.index(
        "torch.compiler.cudagraph_mark_step_begin()", first_mark + 1
    )
    assert first_mark < first_call < gradient
    assert gradient < score_snapshot < second_call
    assert gradient < velocity_snapshot < second_call
    assert gradient < gradient_snapshot < second_call
    assert gradient_snapshot < second_mark < second_call
    assert source.count("torch.compiler.cudagraph_mark_step_begin()") == 2
    assert source.count("score_function(") == 2
    sequence_start = source.index("selected_sequence =")
    score_start = source.index("selected_score =")
    velocities_start = source.index("selected_velocities =")
    diagnostics_start = source.index("diagnostics =")
    assert ").clone()" in source[sequence_start:score_start]
    assert ").clone()" in source[score_start:velocities_start]
    assert ").clone()" in source[velocities_start:diagnostics_start]


def test_main_compiles_only_planner_scorer_and_wires_eager_rti_fallback():
    source = SCRIPT.read_text()
    assert "planner_score_is_compiled = args.compile and args.compile_planner" in source
    assert "planner_score_function = torch.compile(\n            action_sequence_score" in source
    assert "torch.compile(\n            rti_action_sequence" not in source
    assert "score_function=planner_score_function" in source
    assert "compiled_score=planner_score_is_compiled" in source


def test_warm_start_modes_have_exact_shift_tail_and_reset_semantics():
    previous = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    reset = torch.tensor([False, True])
    repeated = MODULE.shift_action_sequence(previous, reset, "repeat")
    repeated_expected = torch.tensor(
        [
            [[2.0], [3.0], [3.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    zero_tail = MODULE.shift_action_sequence(previous, reset, "zero_tail")
    zero_tail_expected = torch.tensor(
        [
            [[2.0], [3.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    reset_plan = MODULE.shift_action_sequence(previous, reset, "reset")
    torch.testing.assert_close(repeated, repeated_expected)
    torch.testing.assert_close(zero_tail, zero_tail_expected)
    torch.testing.assert_close(reset_plan, torch.zeros_like(previous))


def test_executable_has_only_random_warmup_and_no_postwarm_exploration():
    source = SCRIPT.read_text()
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    assert "if global_step < args.warmup_steps" in source
    for forbidden in (
        "random_action_probability",
        "sample_cem_action_sequences",
        "plan_action_sequence",
        "action_opt_iterations",
        "action_opt_beta",
    ):
        assert forbidden not in source


def test_velocity_objective_updates_encoder_dynamics_and_both_velocity_heads():
    args = small_args(
        forward_coef=0.0,
        sigreg_coef=0.0,
        local_nll_coef=0.0,
        goal_nll_coef=0.0,
        velocity_coef=1.0,
    )
    agent = MODULE.Agent(17, 2, args)
    loss, _ = MODULE.training_objective(
        agent, MODULE.SIGReg(projections=8, knots=5), make_batch(), args
    )
    loss.backward()
    for component in (
        agent.encoder,
        agent.dynamics,
        agent.velocity_head,
        agent.forward_delta_head,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in component.parameters()
        )


def test_intact_endpoint_gradients_remain_asymmetric_and_shared():
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(6, 4, requires_grad=True)
    z_next = torch.randn(6, 4, requires_grad=True)
    z_future = torch.randn(6, 4, requires_grad=True)
    action = torch.randn(6, 2)
    previous = torch.randn(6, 2)
    local, goal = MODULE.intact_nll_losses(
        agent, z, z_next, z_future, action, previous
    )
    local_grad = torch.autograd.grad(
        local.sum(), (z, z_next, z_future), allow_unused=True, retain_graph=True
    )
    goal_grad = torch.autograd.grad(
        goal.sum(), (z, z_next, z_future), allow_unused=True
    )
    assert local_grad[0] is not None and local_grad[1] is not None
    assert local_grad[2] is None
    assert goal_grad[0] is not None
    assert goal_grad[1] is None and goal_grad[2] is None


def test_sigreg_is_time_major_and_batch_independent():
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


def test_exact_interval_velocity_handles_live_and_final_infos():
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


def _add_transition(replay, step, done=False):
    replay.add(
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step - 1)]]),
        torch.tensor([[float(step + 1)]]),
        torch.tensor([float(step) + 0.25]),
        torch.tensor([done]),
    )


def test_replay_preserves_exact_labels_boundaries_and_wraparound():
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


def test_defaults_are_h12_single_step_rti():
    args = MODULE.Args()
    assert args.model_horizon == args.planner_horizon == 12
    assert args.rti_step_size == 0.20
    assert args.warm_start_mode == "zero_tail"
    assert args.compile_planner
    assert args.local_nll_coef == 0.1 and args.goal_nll_coef == 0.05
    assert args.total_timesteps == 8_000_000 and args.seed == 1 and args.cuda
    for removed in (
        "planner",
        "action_opt_iterations",
        "action_opt_lr",
        "cem_population",
        "random_action_probability",
    ):
        assert not hasattr(args, removed)


def test_compiled_metrics_are_cloned_before_cuda_graph_reuse():
    source = SCRIPT.read_text()
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "{key: value.clone() for key, value in metrics.items()}" in source
