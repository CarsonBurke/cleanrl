import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_intact_velocity_h5_v4.py"
)
SPEC = importlib.util.spec_from_file_location("intact_velocity_h5_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args(**overrides):
    values = dict(latent_dim=4, hidden_dim=16, model_horizon=3)
    values.update(overrides)
    return MODULE.Args(**values)


def test_agent_decomposes_interval_velocity_into_state_and_action_delta():
    torch.manual_seed(1)
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(5, 4)
    action = torch.randn(5, 2)
    expected = agent.predict_velocity(z) + agent.predict_forward_delta(z, action)
    torch.testing.assert_close(agent.predict_interval_velocity(z, action), expected)
    assert agent.predict_interval_velocity(z, action).shape == (5,)


def test_forward_delta_label_uses_exact_interval_velocity_not_next_qvel():
    observations = torch.zeros(3, 17)
    observations[:, 8] = torch.tensor([1.0, -0.5, 2.0])
    interval = torch.tensor([1.4, 0.25, 1.0])
    torch.testing.assert_close(
        MODULE.forward_delta_target(observations, interval),
        torch.tensor([0.4, 0.75, -1.0]),
    )


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


class QuadraticIntervalModel(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = torch.as_tensor(target)

    def predict_interval_velocity(self, z, action):
        return 3.0 - (action - self.target.to(action)).square().sum(dim=-1)

    def predict_next(self, z, action):
        return z


def optimizer_args(**overrides):
    values = dict(
        action_cost_coef=0.0,
        action_opt_iterations=30,
        action_opt_lr=0.08,
        action_opt_beta1=0.9,
        action_opt_beta2=0.999,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_h5_score_sums_each_predicted_reward_and_exact_action_cost():
    model = QuadraticIntervalModel([0.5, -0.25])
    z = torch.zeros(1, 1)
    actions = torch.tensor([[[0.5, -0.25]]]).expand(-1, 5, -1)
    score, velocities = MODULE.action_sequence_score(model, z, actions, 0.1)
    torch.testing.assert_close(score, torch.tensor([14.84375]))
    torch.testing.assert_close(velocities, torch.full((1, 5), 3.0))


def test_h5_score_backpropagates_future_rewards_through_dynamics():
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(5, 4)
    actions = torch.randn(5, 5, 2, requires_grad=True)
    score, _ = MODULE.action_sequence_score(agent, z, actions)
    score.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.dynamics.parameters()
    )
    assert actions.grad[:, 0].abs().sum() > 0


def test_sequence_optimizer_is_deterministic_bounded_and_improves_score():
    model = QuadraticIntervalModel([0.6, -0.4])
    z = torch.zeros(4, 1)
    start = torch.zeros(4, 5, 2)
    first, first_metrics = MODULE.optimize_action_sequence(
        model, z, start, optimizer_args()
    )
    second, second_metrics = MODULE.optimize_action_sequence(
        model, z, start, optimizer_args()
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_metrics["selected_score"],
        second_metrics["selected_score"],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(first_metrics["optimizer_improvement"] > 0)
    assert torch.all(first.abs() <= 1.0)


class NonconcaveIntervalModel(nn.Module):
    def predict_interval_velocity(self, z, action):
        return torch.sin(15.0 * action).sum(dim=-1)

    def predict_next(self, z, action):
        return z


def test_sequence_optimizer_never_returns_below_warm_start_score():
    model = NonconcaveIntervalModel()
    z = torch.zeros(1, 1)
    start = torch.full((1, 5, 1), 0.94)
    selected, metrics = MODULE.optimize_action_sequence(
        model, z, start, optimizer_args(action_opt_iterations=12)
    )
    initial_score, _ = MODULE.action_sequence_score(model, z, start, 0.0)
    selected_score, _ = MODULE.action_sequence_score(model, z, selected, 0.0)
    assert torch.all(selected_score >= initial_score)
    torch.testing.assert_close(metrics["selected_score"], selected_score)
    assert torch.all(metrics["optimizer_improvement"] >= 0.0)


def test_deployed_sequence_optimizer_has_no_rng_cem_or_intact_dependency():
    source = inspect.getsource(MODULE.optimize_action_sequence)
    source += inspect.getsource(MODULE.action_sequence_score)
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "intact_action_law" not in attributes
    assert "rand" not in source.lower()
    assert "cem" not in source.lower()
    assert "sample" not in source.lower()
    assert "predict_interval_velocity" in attributes
    assert "predict_next" in attributes
    assert "autograd" in attributes


def test_sequence_warm_start_shifts_last_action_and_zeros_resets():
    previous = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    shifted = MODULE.shift_action_sequence(previous, torch.tensor([False, True]))
    expected = torch.tensor(
        [
            [[2.0], [3.0], [3.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    torch.testing.assert_close(shifted, expected)


def test_default_planner_horizon_is_five():
    assert MODULE.Args().planner_horizon == 5


def test_default_execution_has_no_post_warmup_exploration_switch():
    source = SCRIPT.read_text()
    assert "random_action_probability" not in source
    assert "explore =" not in source
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    assert "if global_step < args.warmup_steps" in source


def test_velocity_objective_updates_encoder_dynamics_and_both_heads():
    args = small_args(
        forward_coef=0.0,
        sigreg_coef=0.0,
        local_nll_coef=0.0,
        goal_nll_coef=0.0,
        velocity_coef=1.0,
    )
    agent = MODULE.Agent(17, 2, args)
    observations = torch.randn(8, 4, 17)
    interval = observations[:, :-1, 8] + 0.2 * torch.randn(8, 3)
    batch = MODULE.ReplayBatch(
        observations=observations,
        actions=torch.randn(8, 3, 2),
        previous_actions=torch.randn(8, 3, 2),
        interval_velocities=interval,
        valid=torch.ones(8, 3, dtype=torch.bool),
    )
    loss, _ = MODULE.training_objective(
        agent, MODULE.SIGReg(projections=8, knots=5), batch, args
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


def _add_transition(replay, step, done=False):
    replay.add(
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step)]]),
        torch.tensor([[float(step - 1)]]),
        torch.tensor([[float(step + 1)]]),
        torch.tensor([float(step) + 0.25]),
        torch.tensor([done]),
    )


def test_replay_preserves_interval_labels_boundaries_and_wraparound():
    replay = MODULE.SequenceReplayBuffer(6, 1, 1, 1, device="cpu")
    for step in range(8):
        _add_transition(replay, step, done=(step == 4))
    batch = replay.sample(128, 2, "cpu")
    differences = batch.observations[:, 1:, 0] - batch.observations[:, :-1, 0]
    torch.testing.assert_close(differences, torch.ones_like(differences))
    torch.testing.assert_close(
        batch.interval_velocities,
        batch.actions[..., 0] + 0.25,
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


def test_search_free_intact_direct_interface_is_still_exposed():
    class Law:
        def parameters_for(self, z, intent, previous_action):
            return previous_action + 0.1, torch.zeros_like(previous_action)

    class World:
        intact_action_law = Law()

        def predict_next(self, z, action):
            return z + action[..., :1]

    actions = MODULE.direct_action_sequence(
        World(), torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(2, 1), 3
    )
    expected = torch.tensor([[[0.1], [0.2], [0.3]]]).expand(2, -1, -1)
    torch.testing.assert_close(actions, expected)
