"""Full 1000-step CUDA/native contracts; constructed programs are test fixtures."""

import json

import numpy as np
import pytest
import torch

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.evolving_programs.genome import Genome, Node, Op, Source, SourceKind
from cleanrl.evolving_programs.lifetime import HORIZON, ProgramEvaluator, ProgramResult, ProgramSuite, make_suite
from cleanrl.evolving_programs.policy import ProgramController


def _adapter():
    return ObservationAdapter(np.zeros(19, dtype=np.float32), np.ones(19, dtype=np.float32))


def _node(identity, op, left, right=Source(SourceKind.ZERO), literal=0.0, initial=0.0):
    return Node(identity, op, (left, right), literal, initial)


def _clock(initial=0.0, increment=0.0001):
    return Genome([_node(0, Op.ADD, Source(SourceKind.NODE, 0), Source(SourceKind.LITERAL),
                         increment, initial)], [0] * 6, 1)


def _feedback():
    return Genome([
        _node(0, Op.COPY, Source(SourceKind.OBS, 17)),
        _node(1, Op.COPY, Source(SourceKind.OBS, 18)),
        _node(2, Op.COPY, Source(SourceKind.PREVIOUS_ACTION, 0)),
        _node(3, Op.ADD, Source(SourceKind.NODE, 3), Source(SourceKind.LITERAL), 0.0001),
    ], [0, 1, 2, 3, 3, 3], 4)


def test_suite_independent_streams_roundtrip_and_profile_contract():
    suite = make_suite(17, 23, 8, 128, "positive")
    assert suite.to_json() == make_suite(17, 23, 8, 128, "positive").to_json()
    assert ProgramSuite.from_json(json.loads(json.dumps(suite.to_json()))).to_json() == suite.to_json()
    assert suite.support_seeds != suite.query_seeds
    assert np.all((suite.gains >= 0.5) & (suite.gains <= 1))
    for args in ((18, 23, 8), (17, 24, 8), (17, 23, 9)):
        other = make_suite(*args, 128, "positive")
        assert suite.support_seeds != other.support_seeds
        assert suite.query_seeds != other.query_seeds
        assert not np.array_equal(suite.gains, other.gains)
    prefix = make_suite(17, 23, 8, 3, "positive")
    assert prefix.support_seeds == suite.support_seeds[:3]
    assert prefix.query_seeds == suite.query_seeds[:3]
    np.testing.assert_array_equal(prefix.gains, suite.gains[:3])
    nominal = make_suite(17, 23, 8, 128, "nominal")
    assert nominal.support_seeds == suite.support_seeds
    assert nominal.query_seeds == suite.query_seeds
    np.testing.assert_array_equal(nominal.gains, 1)
    with pytest.raises(ValueError, match="nominal"):
        suite.validate("nominal")
    with pytest.raises(ValueError, match="profile"):
        make_suite(1, 2, 3, 1, "signed")


@pytest.mark.parametrize("support,query,gains", [
    ([], [], []), ([1], [2, 3], [0.5]), ([1], [2], [[0.5]]),
    ([1], [2], [0]), ([1], [2], [-0.75]), ([1], [2], [0.499]),
    ([1], [2], [1.01]), ([1], [2], [np.nan]), ([1], [2], [np.inf]),
    ([1], [2], [0.5j]), ([1], [2], [True]), ([-1], [2], [0.5]),
    ([True], [2], [0.5]), ([1.5], [2], [0.5]), ([1], [2**32], [0.5]),
])
def test_suite_rejects_invalid_contract(support, query, gains):
    with pytest.raises(ValueError):
        ProgramSuite(support, query, np.asarray(gains))


def test_result_never_promotes_invalid_safe_zero_returns():
    support = np.array([[True, False, True, False]])
    query = np.array([[True, True, False, False]])
    result = ProgramResult(np.zeros((1, 4)), np.array([[1., 100., 200., 300.]]), support, query,
                           np.zeros((1, 4, 6)), np.zeros((1, 4, 10)), np.zeros((1, 4)), {})
    np.testing.assert_array_equal(result.scores, [[1, -np.inf, -np.inf, -np.inf]])


def test_evaluator_rejects_cpu_and_invalid_adapters_before_allocation():
    with pytest.raises(ValueError, match="CUDA"):
        ProgramEvaluator(_adapter(), 4, 4, torch.device("cpu"))
    adapter = _adapter()
    adapter.mean[-1] = 0.5
    with pytest.raises(ValueError, match="mean zero"):
        ProgramEvaluator(adapter, 4, 4, torch.device("cuda"))
    adapter = _adapter()
    adapter.scale[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ProgramEvaluator(adapter, 4, 4, torch.device("cuda"))
    with pytest.raises(ValueError, match="ticks"):
        ProgramEvaluator(_adapter(), 4, 0, torch.device("cuda"))


@pytest.fixture
def evaluator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("mujoco")
    torch._dynamo.reset()
    try:
        with ProgramEvaluator(_adapter(), 4, 4, torch.device("cuda"), num_threads=2) as value:
            yield value
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
def test_positive_carry_reset_recent_population_isolation_and_work(evaluator):
    suite = ProgramSuite([5, 9], [31, 37], [0.7, 0.8])
    genomes = [_clock(), _clock(0.0, -0.0001)]
    intact = evaluator.evaluate(genomes, suite, "positive")
    reset = evaluator.evaluate(genomes, suite, "positive", "reset")
    recent = evaluator.evaluate(genomes, suite, "positive", "recent")
    for result, physical_actions in ((intact, HORIZON + 1), (reset, 1), (recent, 9)):
        expected = 0.0001 * 4 * physical_actions
        np.testing.assert_allclose(result.first_actions[0], expected, atol=3e-5)
        np.testing.assert_allclose(result.first_actions[1], -expected, atol=3e-5)
        np.testing.assert_allclose(result.query_blocks.sum(axis=2), result.query_returns, rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(result.scores, result.query_returns)
        assert np.all(result.support_valid & result.query_valid)
    assert evaluator.evaluation_transitions == 3 * 2 * HORIZON * 4
    assert evaluator.logical_node_updates == evaluator.evaluation_transitions * 4
    assert evaluator.capacity_node_updates == evaluator.evaluation_transitions * 4 * 4
    assert evaluator.replay_transitions == 8 * 4
    assert evaluator.replay_node_updates == evaluator.replay_transitions * 4
    evaluator.evaluate(list(reversed(genomes)), suite, "positive", "donor")
    repeated = evaluator.evaluate(genomes, suite, "positive")
    np.testing.assert_array_equal(repeated.first_actions, intact.first_actions)
    np.testing.assert_array_equal(repeated.query_returns, intact.query_returns)
    for index, genome in enumerate(genomes):
        for life in range(2):
            singleton = ProgramSuite([suite.support_seeds[life]], [suite.query_seeds[life]], suite.gains[life:life + 1])
            isolated = evaluator.evaluate([genome], singleton, "positive")
            np.testing.assert_allclose(isolated.first_actions[0, 0], intact.first_actions[index, life], atol=1e-6)
            np.testing.assert_allclose(isolated.query_returns[0, 0], intact.query_returns[index, life], rtol=1e-6, atol=1e-5)


@pytest.mark.cuda
def test_native_raw_reward_causality_interventions_replay_and_query_identity(evaluator, monkeypatch):
    from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv

    policy_calls, physics_calls, reset_calls = [], [], []
    real_action = ProgramController._action_buffer
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

    monkeypatch.setattr(ProgramController, "_action_buffer", observe_action)
    monkeypatch.setattr(NativeMujocoVectorEnv, "step", observe_step)
    monkeypatch.setattr(NativeMujocoVectorEnv, "reset", observe_reset)
    suite = ProgramSuite([101, 103], [211, 223], [0.75, 0.625])
    reference_query, reference_support, alternate_seeds = None, None, None
    for intervention in ("intact", "reset", "donor", "recent", "no_reward", "same_task", "same_task"):
        policy_calls.clear()
        physics_calls.clear()
        reset_calls.clear()
        result = evaluator.evaluate([_feedback()], suite, "positive", intervention)
        query_index = HORIZON + (8 if intervention == "recent" else 0)
        assert len(policy_calls) == 2 * HORIZON + (8 if intervention == "recent" else 0)
        assert len(physics_calls) == 2 * HORIZON
        assert reset_calls[1][0] == suite.query_seeds
        assert reset_calls[0][0] == result.metadata["actual_support_seeds"]
        if intervention == "same_task":
            assert all(a != b for a, b in zip(reset_calls[0][0], suite.support_seeds))
            if alternate_seeds is not None:
                assert alternate_seeds == reset_calls[0][0]
            alternate_seeds = reset_calls[0][0]
        else:
            assert reset_calls[0][0] == suite.support_seeds
        query_inputs, query_action, query_previous = policy_calls[query_index]
        if reference_query is None:
            reference_query = query_inputs.copy()
            reference_support = result.support_returns.copy()
        np.testing.assert_array_equal(query_inputs, reference_query)
        np.testing.assert_array_equal(query_previous, 0)
        for index in (0, query_index):
            inputs, action, previous = policy_calls[index]
            np.testing.assert_array_equal(inputs[:, 17], 0)
            np.testing.assert_array_equal(inputs[:, 18], 1)
            np.testing.assert_array_equal(previous, 0)
            np.testing.assert_allclose(action[:, :3], np.tile([0, np.arcsinh(1), 0], (2, 1)), atol=2e-7)
        for policy_index, physical_index in ((1, 0), (query_index + 1, HORIZON)):
            inputs, action, _ = policy_calls[policy_index]
            expected_reward = (np.zeros(2) if intervention == "no_reward" and policy_index == 1
                               else physics_calls[physical_index][1])
            np.testing.assert_allclose(inputs[:, 17], expected_reward, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(inputs[:, 18], 0)
            np.testing.assert_allclose(action[:, 0], np.clip(np.arcsinh(expected_reward), -1, 1), atol=2e-7)
            np.testing.assert_array_equal(action[:, 2], policy_calls[policy_index - 1][1][:, 0])
        support_gain = 1.5 - suite.gains if intervention == "donor" else suite.gains
        np.testing.assert_allclose(result.metadata["actual_support_gains"], support_gain, atol=1e-7)
        np.testing.assert_allclose(result.metadata["query_gains"], suite.gains, atol=1e-7)
        for step, (executed, _) in enumerate(physics_calls):
            policy_index = step + (8 if intervention == "recent" and step >= HORIZON else 0)
            gain = support_gain if step < HORIZON else suite.gains
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
        rewards = np.stack([call[1] for call in physics_calls])
        np.testing.assert_allclose(result.support_returns[0], rewards[:HORIZON].sum(axis=0), rtol=1e-12)
        np.testing.assert_allclose(result.query_returns[0], rewards[HORIZON:].sum(axis=0), rtol=1e-12)
        query_actions = np.stack([call[1] for call in policy_calls[query_index:]]).astype(np.float64)
        np.testing.assert_allclose(result.action_square_sum[0], np.square(query_actions).sum(axis=(0, 2)), rtol=1e-12)


@pytest.mark.cuda
def test_nominal_is_one_fresh_native_query_with_raw_return(evaluator, monkeypatch):
    from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv

    real_step = NativeMujocoVectorEnv.step
    rewards = []

    def observe_step(env, action):
        result = real_step(env, action)
        rewards.append(np.asarray(result[1]).copy())
        return result

    monkeypatch.setattr(NativeMujocoVectorEnv, "step", observe_step)
    suite = ProgramSuite([41, 43], [71, 73], np.ones(2))
    original = evaluator.evaluate([_feedback()], suite, "nominal")
    assert len(rewards) == HORIZON
    np.testing.assert_allclose(original.query_returns[0], np.stack(rewards).sum(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(original.support_returns, 0)
    np.testing.assert_array_equal(original.support_valid, True)
    assert original.metadata["actual_support_seeds"] == []
    assert original.metadata["actual_support_gains"] == []
    assert evaluator.evaluation_transitions == HORIZON * 2
    evaluator.evaluate([_clock()], suite, "positive")
    changed = evaluator.evaluate([_feedback()], ProgramSuite([47, 53], suite.query_seeds, np.ones(2)), "nominal")
    np.testing.assert_array_equal(original.first_actions, changed.first_actions)
    np.testing.assert_array_equal(original.query_returns, changed.query_returns)
    np.testing.assert_array_equal(original.query_blocks, changed.query_blocks)


@pytest.mark.cuda
def test_invalid_lifetimes_cannot_be_resurrected_and_inactive_overflow_is_isolated(evaluator):
    dead = Genome([_node(0, Op.DIV, Source(SourceKind.ONE), Source(SourceKind.ZERO))], [0] * 6, 1)
    inactive = Genome([
        _node(0, Op.CONST, Source(SourceKind.ZERO), literal=0.1),
        _node(1, Op.MUL, Source(SourceKind.NODE, 1), Source(SourceKind.NODE, 1), initial=1e20),
    ], [0] * 6, 2)
    suite = ProgramSuite([5], [31], [0.7])
    for mode in ("intact", "reset", "recent"):
        result = evaluator.evaluate([dead, inactive], suite, "positive", mode)
        np.testing.assert_array_equal(result.support_valid[:, 0], [False, True])
        np.testing.assert_array_equal(result.query_valid[:, 0], [False, True])
        assert result.scores[0, 0] == -np.inf
        assert np.isfinite(result.scores[1, 0])
        np.testing.assert_array_equal(result.first_actions[0], 0)
        np.testing.assert_array_equal(result.action_square_sum[0], 0)
        np.testing.assert_allclose(result.first_actions[1], 0.1)


@pytest.mark.cuda
def test_recent_replay_death_is_ineligible_without_extra_physics(evaluator):
    # The full support boundary builds a nonzero divisor; replay has no boundary.
    genome = Genome([
        _node(0, Op.ADD, Source(SourceKind.NODE, 0), Source(SourceKind.OBS, 18)),
        _node(1, Op.ADD, Source(SourceKind.NODE, 0), Source(SourceKind.OBS, 18), initial=1),
        _node(2, Op.DIV, Source(SourceKind.ONE), Source(SourceKind.NODE, 1), initial=1),
    ], [2] * 6, 3)
    suite = ProgramSuite([5], [31], [0.7])
    intact = evaluator.evaluate([genome], suite, "positive")
    recent = evaluator.evaluate([genome], suite, "positive", "recent")
    assert intact.support_valid[0, 0] and intact.query_valid[0, 0]
    assert recent.support_valid[0, 0] and not recent.query_valid[0, 0]
    assert recent.scores[0, 0] == -np.inf
    assert evaluator.evaluation_transitions == 4 * HORIZON
    assert evaluator.replay_transitions == 8
    assert evaluator.replay_node_updates == 8 * 3 * 4


@pytest.mark.cuda
def test_cache_evicts_closes_and_context_close_is_idempotent(evaluator, monkeypatch):
    from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv

    closed = []
    real_close = NativeMujocoVectorEnv.close

    def observe_close(env):
        closed.append(id(env))
        return real_close(env)

    monkeypatch.setattr(NativeMujocoVectorEnv, "close", observe_close)
    for lives in range(1, 5):
        evaluator.evaluate([_clock()], make_suite(1, 2, 3, lives), "nominal")
    first_env = evaluator._slots[(1, 1)][0]
    second_env = evaluator._slots[(1, 2)][0]
    evaluator.evaluate([_clock()], make_suite(1, 2, 3, 1), "nominal")
    evaluator.evaluate([_clock()], make_suite(1, 2, 3, 5), "nominal")
    assert len(evaluator._slots) == 4
    assert id(second_env) in closed
    assert id(first_env) not in closed
    live_ids = [id(env) for env, _ in evaluator._slots.values()]
    evaluator.close()
    assert not evaluator._slots
    assert all(closed.count(identity) == 1 for identity in live_ids)
    before = closed.copy()
    evaluator.close()
    assert closed == before


@pytest.mark.cuda
def test_rejects_invalid_inputs_before_physics_and_early_terminal_after_counting(evaluator, monkeypatch):
    from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv

    suite = make_suite(1, 2, 3, 1)
    with pytest.raises(ValueError, match="intervention"):
        evaluator.evaluate([_clock()], suite, intervention="unknown")
    with pytest.raises(ValueError, match="positive"):
        evaluator.evaluate([_clock()], suite, intervention="reset")
    with pytest.raises(ValueError, match="at least one"):
        evaluator.evaluate([], suite)
    invalid_suite = ProgramSuite([1], [2], [0.75])
    with pytest.raises(ValueError, match="nominal"):
        evaluator.evaluate([_clock()], invalid_suite)
    suite.gains[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate([_clock()], suite)
    assert evaluator.evaluation_transitions == 0
    real_step = NativeMujocoVectorEnv.step

    def early_terminal(env, action):
        observation, reward, terminated, truncated, info = real_step(env, action)
        return observation, reward, terminated, np.ones_like(truncated, dtype=bool), info

    monkeypatch.setattr(NativeMujocoVectorEnv, "step", early_terminal)
    with pytest.raises(RuntimeError, match="exactly at step 1000"):
        evaluator.evaluate([_clock()], make_suite(1, 2, 3, 1))
    assert evaluator.evaluation_transitions == 1
    assert evaluator.logical_node_updates == 4
    assert evaluator.capacity_node_updates == 16
