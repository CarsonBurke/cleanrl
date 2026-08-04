import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_matched_prescribed_detached_intact_v16.py"
)
SPEC = importlib.util.spec_from_file_location(
    "matched_prescribed_detached_intact_v16", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args(**overrides):
    values = dict(
        latent_dim=4,
        hidden_dim=16,
        model_horizon=3,
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


def test_single_action_law_has_four_slot_grammar_fixed_std_and_proper_nll():
    law = MODULE.IntentActionLaw(4, 2, 16, fixed_std=0.2)
    z = torch.randn(5, 4)
    intent = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    action = torch.randn(5, 2)
    mean, log_std = law.parameters_for(z, intent, previous)
    assert mean.shape == log_std.shape == action.shape
    torch.testing.assert_close(
        log_std, torch.full_like(log_std, np.log(0.2))
    )
    expected = -Normal(mean, log_std.exp()).log_prob(action).sum(dim=-1)
    torch.testing.assert_close(law.nll(z, intent, previous, action), expected)
    source = inspect.getsource(MODULE.IntentActionLaw.parameters_for)
    assert "[z, intent, z * intent, embedded_previous_action]" in source


def test_matched_objective_uses_detached_predicted_local_and_factual_goal_intents():
    torch.manual_seed(2)
    args = small_args(physical_intact_fraction=0.4)
    agent = MODULE.Agent(17, 2, args)
    factual_z = torch.randn(8, 4, 4, requires_grad=True)
    previous = torch.randn(8, 2, requires_grad=True)
    actions = torch.randn(8, 3, 2, requires_grad=True)
    loss, metrics = MODULE.matched_intact_objective(
        agent, factual_z, previous, actions, args
    )
    loss.backward()
    assert has_nonzero_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.world_model_parameters(agent))
    assert not has_any_gradient(MODULE.prescriber_parameters(agent))
    assert factual_z.grad is None and previous.grad is None and actions.grad is None
    for key in (
        "physical_nll",
        "goal_nll",
        "physical_action_mae",
        "goal_action_mae",
        "goal_physical_action_gap",
        "goal_physical_action_alignment",
        "goal_intent_use",
    ):
        assert key in metrics and torch.isfinite(metrics[key])
    source = inspect.getsource(MODULE.matched_intact_objective)
    assert "predicted_z1 = agent.predict_next(z_start, demonstrated_action)" in source
    assert "physical_intent = (predicted_z1.detach() - z_start).detach()" in source
    assert "goal_intent = (factual_z[:, -1].detach() - z_start).detach()" in source
    assert source.count("agent.intact_action_law.nll(") == 2


def test_physical_fraction_normalizes_matched_objective_scale():
    args = small_args(physical_intact_fraction=0.25)
    agent = MODULE.Agent(17, 2, args)
    factual_z = torch.randn(6, 4, 4)
    previous = torch.randn(6, 2)
    actions = torch.randn(6, 3, 2)
    total, metrics = MODULE.matched_intact_objective(
        agent, factual_z, previous, actions, args
    )
    torch.testing.assert_close(
        total.detach(),
        args.physical_intact_fraction * metrics["physical_nll"]
        + (1.0 - args.physical_intact_fraction) * metrics["goal_nll"],
    )


def test_world_objective_gradients_only_prediction_partition():
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    loss, _ = MODULE.world_model_objective(
        agent, MODULE.SIGReg(projections=8, knots=5), make_batch(), args
    )
    loss.backward()
    assert has_nonzero_gradient(MODULE.world_model_parameters(agent))
    assert not has_any_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.prescriber_parameters(agent))


def test_prescriber_reward_gradients_only_prescriber_through_frozen_law_and_wm():
    torch.manual_seed(4)
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(8, 4, requires_grad=True)
    previous = torch.randn(8, 2, requires_grad=True)
    loss, metrics = MODULE.direct_controller_objective(agent, z, previous, args)
    loss.backward()
    assert has_nonzero_gradient(MODULE.prescriber_parameters(agent))
    assert not has_any_gradient(MODULE.intact_law_parameters(agent))
    assert not has_any_gradient(MODULE.world_model_parameters(agent))
    assert z.grad is None and previous.grad is None
    torch.testing.assert_close(
        loss.detach() * args.controller_horizon, -metrics["imagined_score"]
    )


def test_prescribed_intent_is_state_dependent_and_h12_rms_bounded():
    agent = MODULE.Agent(17, 2, small_args())
    factual_intents = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    agent.update_factual_intent_scale(factual_intents, ema=0.9)
    expected = factual_intents.square().sum(dim=-1).mean().sqrt()
    z = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    intent, raw = agent.prescribe_intent(z, previous)
    torch.testing.assert_close(intent.norm(dim=-1), expected.expand(5))
    assert raw.shape == intent.shape
    shuffled, _ = agent.prescribe_intent(z.roll(1, dims=0), previous)
    assert (intent - shuffled).abs().sum() > 0


def test_three_parameter_partitions_are_disjoint_and_complete():
    agent = MODULE.Agent(17, 2, small_args())
    groups = [
        {id(parameter) for parameter in parameters}
        for parameters in (
            MODULE.world_model_parameters(agent),
            MODULE.intact_law_parameters(agent),
            MODULE.prescriber_parameters(agent),
        )
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == {id(parameter) for parameter in agent.parameters()}


def test_closed_loop_credit_has_exact_h1_h6_h12_lengths_and_reward_sum():
    args = small_args(model_horizon=12)
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(4, 4)
    previous = torch.randn(4, 2)
    for horizon in (1, 6, 12):
        score, velocities, costs, raw_actions, actions = (
            MODULE.closed_loop_direct_score(agent, z, previous, horizon, 0.1)
        )
        assert velocities.shape == costs.shape == (4, horizon)
        assert raw_actions.shape == actions.shape == (4, horizon, 2)
        torch.testing.assert_close(score, (velocities - costs).sum(dim=1))
        assert torch.all(actions.abs() <= 1.0)


def test_one_step_imagination_matches_runtime_action_and_successor_timing():
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    runtime_action, _ = MODULE.direct_action(agent, z, previous)
    score, velocities, costs, _, imagined_actions = MODULE.closed_loop_direct_score(
        agent, z, previous, 1, 0.1
    )
    torch.testing.assert_close(imagined_actions[:, 0], runtime_action)
    successor = agent.predict_next(z, runtime_action)
    expected_velocity = agent.predict_interval_velocity(successor)
    expected_cost = 0.1 * runtime_action.square().sum(dim=-1)
    torch.testing.assert_close(velocities[:, 0], expected_velocity)
    torch.testing.assert_close(costs[:, 0], expected_cost)
    torch.testing.assert_close(score, expected_velocity - expected_cost)


def test_runtime_is_deterministic_and_calls_prescriber_and_law_once():
    agent = MODULE.Agent(17, 2, small_args())
    z = torch.randn(4, 4)
    previous = torch.randn(4, 2)
    original_prescribe = agent.prescribe_intent
    original_law = agent.intact_action_law.parameters_for
    counts = {"prescriber": 0, "law": 0}

    def counted_prescribe(*args, **kwargs):
        counts["prescriber"] += 1
        return original_prescribe(*args, **kwargs)

    def counted_law(*args, **kwargs):
        counts["law"] += 1
        return original_law(*args, **kwargs)

    agent.prescribe_intent = counted_prescribe
    agent.intact_action_law.parameters_for = counted_law
    first, first_metrics = MODULE.direct_action(agent, z, previous)
    assert counts == {"prescriber": 1, "law": 1}
    counts.update(prescriber=0, law=0)
    second, second_metrics = MODULE.direct_action(agent, z, previous)
    assert counts == {"prescriber": 1, "law": 1}
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first_metrics.keys() == second_metrics.keys()
    assert torch.all(first.abs() <= 1.0)


def test_executable_has_random_warmup_only_and_no_chunk_or_search_path():
    source = SCRIPT.read_text()
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    assert "if global_step < args.warmup_steps" in source
    for forbidden in (
        "chunk",
        "sample_cem",
        "rti_action",
        "trust_region",
        "terminal_value",
        "random_action_probability",
    ):
        assert forbidden not in source.lower()
    runtime_source = inspect.getsource(MODULE.direct_action).lower()
    for forbidden in ("for ", "while ", "rand", "sample", "rollout", "search"):
        assert forbidden not in runtime_source


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


def test_defaults_and_main_have_h12_three_isolated_optimizers():
    args = MODULE.Args()
    assert args.model_horizon == args.controller_horizon == 12
    assert args.physical_intact_fraction == 0.5
    assert args.intact_fixed_std == 0.2
    assert not hasattr(args, "chunk_horizon")
    source = SCRIPT.read_text()
    for name in ("wm_optimizer", "law_optimizer", "prescriber_optimizer"):
        assert f"{name} = optim.AdamW(" in source
    assert "intact_loss_function = matched_intact_objective" in source
    assert source.count("torch.compiler.cudagraph_mark_step_begin()") == 3
    assert source.count("key: value.clone() for key, value in") == 3
    assert source.count("assert_parameters_have_no_gradients(") >= 6
