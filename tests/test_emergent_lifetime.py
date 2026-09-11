"""Actual CUDA/native lifetimes; constructed controllers are test fixtures only."""

import json

import numpy as np
import pytest
import torch

from cleanrl.collective_control.control import CollectiveController, ObservationAdapter
from cleanrl.collective_control.genome import Genome, Node, Source
from cleanrl.collective_control.lifetime import (
    HORIZON,
    LifetimeEvaluator,
    LifetimeSuite,
    _capture_preserving_state,
    make_suite,
)


def test_suite_independent_streams_and_json_round_trip():
    suite = make_suite(17, 23, 8, 128)
    assert suite.to_json() == make_suite(17, 23, 8, 128).to_json()
    restored = LifetimeSuite.from_json(json.loads(json.dumps(suite.to_json())))
    assert restored.to_json() == suite.to_json()
    children = np.random.SeedSequence([17, 23, 8]).spawn(4)
    assert suite.support_seeds == np.random.default_rng(children[0]).integers(0, 2**32, 128, dtype=np.uint32).tolist()
    assert suite.query_seeds == np.random.default_rng(children[1]).integers(0, 2**32, 128, dtype=np.uint32).tolist()
    signs = 2 * np.random.default_rng(children[2]).integers(0, 2, 128) - 1
    magnitudes = np.random.default_rng(children[3]).uniform(0.5, 1.0, 128)
    np.testing.assert_array_equal(suite.gains, signs * magnitudes)
    for args in ((18, 23, 8), (17, 24, 8), (17, 23, 9)):
        other = make_suite(*args, 128)
        assert suite.support_seeds != other.support_seeds
        assert suite.query_seeds != other.query_seeds
        assert not np.array_equal(suite.gains, other.gains)
    # Count changes cannot balance/condition previous lives or advance reset streams.
    prefix = make_suite(17, 23, 8, 3)
    assert prefix.support_seeds == suite.support_seeds[:3]
    assert prefix.query_seeds == suite.query_seeds[:3]
    np.testing.assert_array_equal(prefix.gains, suite.gains[:3])


@pytest.mark.parametrize("support,query,gains", [
    ([], [], []), ([1], [2, 3], [0.5]), ([1], [2], [[0.5]]),
    ([1], [2], [0.0]), ([1], [2], [float("nan")]), ([1], [2], [float("inf")]),
    ([1], [2], [1.01]), ([1], [2], [0.5j]), ([1], [2], [True]),
    ([-1], [2], [0.5]), ([1.5], [2], [0.5]), ([1], [2**32], [0.5]),
])
def test_suite_rejects_invalid_reset_or_gain_contract(support, query, gains):
    with pytest.raises(ValueError):
        LifetimeSuite(support, query, np.asarray(gains))


def _adapter():
    return ObservationAdapter(np.zeros(19, dtype=np.float32), np.ones(19, dtype=np.float32))


def test_evaluator_rejects_cpu_and_invalid_adapters_before_allocation():
    with pytest.raises(ValueError, match="CUDA"):
        LifetimeEvaluator(_adapter(), 4, torch.device("cpu"))
    adapter = _adapter()
    adapter.mean[-1] = 0.5
    with pytest.raises(ValueError, match="mean zero"):
        LifetimeEvaluator(adapter, 4, torch.device("cuda"))
    adapter = _adapter()
    adapter.scale[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LifetimeEvaluator(adapter, 4, torch.device("cuda"))


@pytest.fixture
def evaluator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("mujoco")
    # These cases deliberately vary execution shapes. Compiler specializations
    # are process-global state, not part of another test's controller lifetime.
    torch._dynamo.reset()
    try:
        with LifetimeEvaluator(_adapter(), 4, torch.device("cuda"), num_threads=2) as value:
            yield value
    finally:
        torch._dynamo.reset()


def _clock(initial=0.0, target=1.0):
    return Genome([Node(0, (Source(4), Source(4)), np.full(4, target, dtype=np.float32),
                        0.001, initial)], [0] * 6, 1)


def _feedback():
    identity = np.array([0, 0, 1, 1], dtype=np.float32)
    return Genome([
        Node(0, (Source(0, 17), Source(4)), identity.copy(), 1.0, 0.5),
        Node(1, (Source(0, 18), Source(4)), identity.copy(), 1.0, 0.5),
        Node(2, (Source(2, 0), Source(4)), identity.copy(), 1.0, 0.5),
        Node(3, (Source(0, 0), Source(4)), identity.copy(), 0.002, 0.3),
    ], [0, 1, 2, 3, 3, 3], 4)


@pytest.mark.cuda
def test_lifetime_reset_episode_carry_recent_and_population_isolation(evaluator):
    suite = LifetimeSuite([5, 9], [31, 37], np.array([0.7, -0.8]))
    genomes = [_clock(), _clock(1.0, 0.0)]
    intact = evaluator.evaluate(genomes, suite)
    reset = evaluator.evaluate(genomes, suite, "reset")
    recent = evaluator.evaluate(genomes, suite, "recent")
    for result, ticks in ((intact, HORIZON + 1), (reset, 1), (recent, 9)):
        upward = 2 * (1 - 0.999**ticks) - 1
        np.testing.assert_allclose(result.first_actions[0], upward, atol=3e-5)
        np.testing.assert_allclose(result.first_actions[1], -upward, atol=3e-5)
        np.testing.assert_allclose(result.query_blocks.sum(axis=2), result.query_returns, rtol=1e-12, atol=1e-10)
        assert result.scores is result.query_returns
    assert evaluator.evaluation_transitions == 3 * 2 * HORIZON * 4
    assert evaluator.replay_transitions == 8 * 4
    assert evaluator.node_updates == evaluator.evaluation_transitions
    assert evaluator.replay_node_updates == evaluator.replay_transitions
    # Reuse a cached graph after different state and genotype contents.
    evaluator.evaluate(list(reversed(genomes)), suite, "donor")
    repeated = evaluator.evaluate(genomes, suite)
    np.testing.assert_array_equal(repeated.first_actions, intact.first_actions)
    np.testing.assert_array_equal(repeated.query_returns, intact.query_returns)
    for index, genome in enumerate(genomes):
        for life in range(2):
            singleton = LifetimeSuite([suite.support_seeds[life]], [suite.query_seeds[life]], suite.gains[life:life + 1])
            isolated = evaluator.evaluate([genome], singleton)
            np.testing.assert_allclose(isolated.first_actions[0, 0], intact.first_actions[index, life], atol=1e-6)
            np.testing.assert_allclose(isolated.query_returns[0, 0], intact.query_returns[index, life], rtol=1e-6, atol=1e-5)


@pytest.mark.cuda
def test_native_feedback_causality_donor_query_resets_and_actual_recent_inputs(evaluator, monkeypatch):
    from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv

    policy_calls, physics_calls, reset_calls = [], [], []
    real_action = CollectiveController._action_buffer
    real_step = NativeMujocoVectorEnv.step
    real_reset = NativeMujocoVectorEnv.reset

    def observe_action(controller, inputs):
        call = len(policy_calls)
        previous = controller.previous_action.cpu().numpy().copy() if call == 0 or 992 <= call <= 1009 else None
        observed = np.asarray(inputs).copy()
        action = real_action(controller, inputs)
        policy_calls.append((observed, action.copy(), previous))
        return action

    def observe_step(env, action):
        executed = np.asarray(action).copy()
        result = real_step(env, action)
        physics_calls.append((executed, np.asarray(result[1]).copy()))
        return result

    def observe_reset(env, **kwargs):
        result = real_reset(env, **kwargs)
        reset_calls.append((kwargs["seed"], np.asarray(result[0]).copy()))
        return result

    monkeypatch.setattr(CollectiveController, "_action_buffer", observe_action)
    monkeypatch.setattr(NativeMujocoVectorEnv, "step", observe_step)
    monkeypatch.setattr(NativeMujocoVectorEnv, "reset", observe_reset)
    suite = LifetimeSuite([101, 103], [211, 223], np.array([0.75, -0.625]))
    reference_query = None
    reference_support = None
    for intervention in ("intact", "reset", "donor", "recent", "no_reward"):
        policy_calls.clear()
        physics_calls.clear()
        reset_calls.clear()
        result = evaluator.evaluate([_feedback()], suite, intervention)
        query_index = HORIZON + (8 if intervention == "recent" else 0)
        assert len(policy_calls) == 2 * HORIZON + (8 if intervention == "recent" else 0)
        assert len(physics_calls) == 2 * HORIZON
        assert [call[0] for call in reset_calls] == [suite.support_seeds, suite.query_seeds]
        query_inputs, query_action, query_previous = policy_calls[query_index]
        if reference_query is None:
            reference_query = query_inputs.copy()
            reference_support = result.support_returns.copy()
        np.testing.assert_array_equal(query_inputs, reference_query)
        np.testing.assert_array_equal(query_previous, 0.5)
        for index in (0, query_index):
            inputs, action, previous = policy_calls[index]
            np.testing.assert_array_equal(inputs[:, 17], 0)
            np.testing.assert_array_equal(inputs[:, 18], 1)
            np.testing.assert_array_equal(previous, 0.5)
            np.testing.assert_allclose(action[:, :3], np.tile([0, np.tanh(1), 0], (2, 1)), atol=2e-7)
        for policy_index, physical_index in ((1, 0), (query_index + 1, HORIZON)):
            inputs, action, _ = policy_calls[policy_index]
            expected_reward = (np.zeros(2) if intervention == "no_reward" and policy_index == 1
                               else np.arcsinh(physics_calls[physical_index][1]))
            np.testing.assert_allclose(inputs[:, 17], expected_reward, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(inputs[:, 18], 0)
            np.testing.assert_allclose(action[:, 0], np.tanh(expected_reward), atol=2e-7)
        for step, (executed, _) in enumerate(physics_calls):
            policy_index = step + (8 if intervention == "recent" and step >= HORIZON else 0)
            gain = -suite.gains if intervention == "donor" and step < HORIZON else suite.gains
            np.testing.assert_allclose(executed, policy_calls[policy_index][1] * gain[:, None], atol=1e-7)
        if intervention in {"reset", "recent"}:
            np.testing.assert_array_equal(result.support_returns, reference_support)
        if intervention == "recent":
            for index in range(8):
                support_input, _, support_previous = policy_calls[HORIZON - 8 + index]
                replay_input, _, replay_previous = policy_calls[HORIZON + index]
                np.testing.assert_array_equal(replay_input, support_input)
                np.testing.assert_array_equal(replay_previous, support_previous)
        if intervention == "no_reward":
            assert all(np.all(call[0][:, 17] == 0) for call in policy_calls[:HORIZON])
        np.testing.assert_array_equal(result.first_actions[0], query_action)


@pytest.mark.cuda
def test_capture_preserves_nonpristine_state_and_previous_action(evaluator):
    # Capture after restoring an experimental state must not silently reset it.
    controller = CollectiveController([_feedback()], evaluator.adapter, -np.ones(6, dtype=np.float32),
                                      np.ones(6, dtype=np.float32), 4, "uniform", evaluator.device)
    with torch.inference_mode():
        controller.state.fill_(0.2)
        controller.previous_action.fill_(0.8)
    _capture_preserving_state(controller)
    np.testing.assert_array_equal(controller.state.cpu().numpy(), np.float32(0.2))
    np.testing.assert_array_equal(controller.previous_action.cpu().numpy(), np.float32(0.8))
    action = controller.action(np.zeros((1, 19), dtype=np.float32))
    np.testing.assert_allclose(action[0, 2], 0.6, atol=1e-7)
    np.testing.assert_allclose(action[0, 3:], 2 * (0.998 * 0.2 + 0.002 * 0.5) - 1, atol=1e-7)


@pytest.mark.cuda
def test_evaluation_rejects_invalid_inputs_without_physics(evaluator):
    suite = make_suite(1, 2, 3, 1)
    with pytest.raises(ValueError, match="intervention"):
        evaluator.evaluate([_clock()], suite, "unknown")
    with pytest.raises(ValueError, match="at least one"):
        evaluator.evaluate([], suite)
    malformed = _feedback()
    malformed.nodes[0].sources = (Source(0, 19), Source(4))
    with pytest.raises(ValueError, match="interface"):
        evaluator.evaluate([malformed], suite)
    suite.gains[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate([_clock()], suite)
    assert evaluator.evaluation_transitions == 0


@pytest.mark.cuda
def test_nominal_cold_query_is_independent_of_support_history(evaluator):
    genome = _feedback()
    first = LifetimeSuite([41, 43], [71, 73], np.ones(2))
    second = LifetimeSuite([47, 53], first.query_seeds, np.ones(2))
    original = evaluator.evaluate([genome], first, "reset")
    # Insert an intact rollout to contaminate every cached recurrent slot.
    evaluator.evaluate([_clock()], second, "intact")
    changed_support = evaluator.evaluate([genome], second, "reset")
    np.testing.assert_array_equal(original.first_actions, changed_support.first_actions)
    np.testing.assert_array_equal(original.query_returns, changed_support.query_returns)
    np.testing.assert_array_equal(original.query_blocks, changed_support.query_blocks)
