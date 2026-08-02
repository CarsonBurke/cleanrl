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
    / "ppo_continuous_action_intact_velocity_mpc_v1.py"
)
SPEC = importlib.util.spec_from_file_location("intact_velocity_mpc_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args(**overrides):
    values = dict(latent_dim=4, hidden_dim=16, model_horizon=3)
    values.update(overrides)
    return MODULE.Args(**values)


def test_model_and_full_four_slot_gaussian_shapes():
    agent = MODULE.Agent(obs_dim=17, action_dim=6, args=small_args())
    observations = torch.randn(5, 17)
    actions = torch.randn(5, 6)
    z = agent.encode(observations)
    z_next = agent.predict_next(z, actions)
    distribution = agent.intact_action_law(z, z_next - z, actions)
    assert z.shape == (5, 4)
    assert z_next.shape == (5, 4)
    assert distribution.mean.shape == (5, 6)
    assert distribution.stddev.shape == (5, 6)
    first_predictor_linear = next(
        layer
        for layer in agent.intact_action_law.predictor
        if isinstance(layer, nn.Linear)
    )
    assert first_predictor_linear.in_features == 4 * agent.latent_dim
    _, log_std = agent.intact_action_law.parameters_for(z, z_next - z, actions)
    assert torch.all(log_std >= -5.0)
    assert torch.all(log_std <= 2.0)


def test_sigreg_requires_explicit_time_batch_latent_layout():
    sigreg = MODULE.SIGReg(projections=8, knots=5)
    assert torch.isfinite(sigreg(torch.randn(3, 7, 4)))
    try:
        sigreg(torch.randn(7, 4))
    except ValueError as error:
        assert "time,batch" in str(error)
    else:
        raise AssertionError("SIGReg silently accepted a layout without time")


def test_sigreg_computes_each_time_batch_statistic_independently():
    # In one latent dimension the random projection can only change sign, which
    # leaves the characteristic-function error unchanged. This makes the
    # batched-vs-per-time comparison deterministic without exposing internals.
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


def test_cem_sequence_update_refits_each_horizon_coordinate():
    candidates = torch.tensor(
        [
            [
                [[0.0], [0.0]],
                [[1.0], [0.0]],
                [[0.0], [0.8]],
                [[1.0], [0.8]],
            ]
        ]
    )
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    center = torch.zeros(1, 2, 1)
    std = torch.ones_like(center)
    next_center, next_std, best, elites = MODULE.cem_sequence_update(
        candidates,
        scores,
        elite_count=2,
        center=center,
        std=std,
        update_rate=1.0,
        minimum_std=0.1,
    )
    torch.testing.assert_close(next_center, torch.tensor([[[0.5], [0.8]]]))
    torch.testing.assert_close(best, torch.tensor([[[1.0], [0.8]]]))
    assert next_std.shape == center.shape
    assert elites.shape == (1, 2, 2, 1)


class AdditiveVelocityWorld(nn.Module):
    def predict_next(self, z, action):
        return z + action[..., :1]

    def predict_velocity(self, z):
        return z[..., 0]


def test_sequence_score_sums_predicted_successor_velocities_over_horizon():
    world = AdditiveVelocityWorld()
    z = torch.zeros(1, 1)
    actions = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    score, velocities = MODULE.score_action_sequences(
        world, z, actions, return_velocities=True
    )
    torch.testing.assert_close(velocities, torch.tensor([[[1.0, 3.0, 6.0]]]))
    torch.testing.assert_close(score, torch.tensor([[10.0]]))
    cost_score = MODULE.score_action_sequences(
        world, z, actions, action_cost_coef=0.5
    )
    torch.testing.assert_close(cost_score, torch.tensor([[3.0]]))


def test_local_and_goal_intents_have_exact_asymmetric_endpoint_gradients():
    torch.manual_seed(3)
    agent = MODULE.Agent(obs_dim=17, action_dim=2, args=small_args())
    z = torch.randn(6, 4, requires_grad=True)
    z_next = torch.randn(6, 4, requires_grad=True)
    z_future = torch.randn(6, 4, requires_grad=True)
    action = torch.randn(6, 2)
    previous_action = torch.randn(6, 2)
    local_nll, goal_nll = MODULE.intact_nll_losses(
        agent, z, z_next, z_future, action, previous_action
    )

    local_gradients = torch.autograd.grad(
        local_nll.sum(), (z, z_next, z_future), allow_unused=True, retain_graph=True
    )
    assert local_gradients[0].abs().sum() > 0
    assert local_gradients[1].abs().sum() > 0
    assert local_gradients[2] is None

    goal_gradients = torch.autograd.grad(
        goal_nll.sum(), (z, z_next, z_future), allow_unused=True
    )
    assert goal_gradients[0].abs().sum() > 0
    assert goal_gradients[1] is None
    assert goal_gradients[2] is None


def test_velocity_supervision_updates_encoder_dynamics_and_head():
    args = small_args(
        forward_coef=0.0,
        sigreg_coef=0.0,
        local_nll_coef=0.0,
        goal_nll_coef=0.0,
        velocity_coef=1.0,
    )
    agent = MODULE.Agent(obs_dim=17, action_dim=2, args=args)
    batch = MODULE.ReplayBatch(
        observations=torch.randn(8, 4, 17),
        actions=torch.randn(8, 3, 2),
        previous_actions=torch.randn(8, 3, 2),
        valid=torch.ones(8, 3, dtype=torch.bool),
    )
    sigreg = MODULE.SIGReg(projections=8, knots=5)
    loss, _ = MODULE.training_objective(agent, sigreg, batch, args)
    loss.backward()
    for module in (agent.encoder, agent.dynamics, agent.velocity_head):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_planner_is_actor_disabled_and_has_no_policy_q_or_direct_dependency():
    functions = (MODULE.score_action_sequences, MODULE.plan_velocity_cem)
    source = "\n".join(inspect.getsource(function) for function in functions)
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "intact_action_law" not in attributes
    assert "policy" not in source.lower()
    assert "q_function" not in source.lower()
    assert "direct_action_sequence" not in source


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


def test_ring_replay_samples_full_contiguous_windows_without_reset_seams():
    replay = MODULE.SequenceReplayBuffer(8, 1, 1, 1, device="cpu")
    transitions = [
        (0.0, 1.0, False),
        (1.0, 2.0, True),
        (10.0, 11.0, False),
        (11.0, 12.0, False),
        (12.0, 13.0, False),
    ]
    for step, (obs, next_obs, done) in enumerate(transitions):
        replay.add(
            torch.tensor([[obs]]),
            torch.tensor([[float(step)]]),
            torch.tensor([[float(step - 1)]]),
            torch.tensor([[next_obs]]),
            torch.tensor([done]),
        )
    batch = replay.sample(128, model_horizon=2, device="cpu")
    differences = batch.observations[:, 1:, 0] - batch.observations[:, :-1, 0]
    torch.testing.assert_close(differences, torch.ones_like(differences))
    # The stored prior action is used even when the sampled window begins in
    # the middle of an episode; it is not blindly zeroed at window boundaries.
    assert torch.any(batch.previous_actions[:, 0, 0] != 0)


def test_ring_replay_remains_chronological_after_capacity_wraparound():
    replay = MODULE.SequenceReplayBuffer(5, 1, 1, 1, device="cpu")
    for step in range(7):
        replay.add(
            torch.tensor([[float(step)]]),
            torch.tensor([[float(step)]]),
            torch.tensor([[float(step - 1)]]),
            torch.tensor([[float(step + 1)]]),
            torch.tensor([False]),
        )
    batch = replay.sample(128, model_horizon=3, device="cpu")
    differences = batch.observations[:, 1:, 0] - batch.observations[:, :-1, 0]
    torch.testing.assert_close(differences, torch.ones_like(differences))
    assert torch.all(batch.observations[:, 0, 0] >= 2.0)
    assert torch.all(batch.observations[:, 0, 0] <= 4.0)


def test_shifted_center_is_receding_horizon_warm_start():
    sequence = torch.tensor([[[1.0], [2.0], [3.0]]])
    shifted = MODULE.shift_action_sequence(sequence)
    torch.testing.assert_close(shifted, torch.tensor([[[2.0], [3.0], [0.0]]]))


def test_velocity_index_is_exact_halfcheetah_qvel_zero():
    observations = torch.arange(34, dtype=torch.float32).reshape(2, 17)
    torch.testing.assert_close(
        MODULE.forward_velocity(observations), observations[:, 8]
    )
    assert MODULE.VELOCITY_OBSERVATION_INDEX == 8


class PreviousActionDirectLaw:
    def parameters_for(self, z, intent, previous_action):
        return previous_action + 0.1, torch.zeros_like(previous_action)


class DirectWorld(AdditiveVelocityWorld):
    def __init__(self):
        super().__init__()
        self.intact_action_law = PreviousActionDirectLaw()


def test_search_free_direct_interface_recurs_through_g_mean_and_f():
    actions = MODULE.direct_action_sequence(
        DirectWorld(),
        z=torch.zeros(2, 1),
        goal=torch.ones(2, 1),
        previous_action=torch.zeros(2, 1),
        horizon=3,
    )
    expected = torch.tensor([[[0.1], [0.2], [0.3]]]).expand(2, -1, -1)
    torch.testing.assert_close(actions, expected)


def test_cem_samples_include_incumbent_center_and_global_sequences():
    center = torch.zeros(2, 3, 1)
    incumbent = torch.full_like(center, 0.25)
    candidates = MODULE.sample_cem_action_sequences(
        center,
        torch.ones_like(center),
        population=8,
        global_fraction=0.25,
        incumbent=incumbent,
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(candidates[:, 0], incumbent)
    torch.testing.assert_close(candidates[:, 1], center)
    assert candidates.shape == (2, 8, 3, 1)
    assert torch.all(candidates.abs() <= 1.0)
