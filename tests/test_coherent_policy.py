import numpy as np
import pytest
import torch

from cleanrl.coherent_control.policy import AffineEvaluator, _AffinePolicy
from cleanrl.collective_control.control import ObservationAdapter


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("compiled coherent policies require CUDA")
    return torch.device("cuda")


def _numpy_actions(parameters, observations, adapter, low, high, episodes, *, blind=False):
    """Independent per-policy matrix multiplication, not tensor broadcasting."""
    observation_dim, action_dim = len(adapter.mean), len(low)
    actions = []
    for index, observation in enumerate(observations):
        row = parameters[index // episodes]
        weights = row[:action_dim * observation_dim].reshape(action_dim, observation_dim)
        bias = row[action_dim * observation_dim:]
        value = adapter.mean if blind else np.asarray(observation, dtype=np.float32)
        raw = weights @ ((value - adapter.mean) / adapter.scale) + bias
        actions.append(np.clip((low + high) / 2 + (high - low) / 2 * raw, low, high))
    return np.asarray(actions, dtype=np.float32)


@pytest.mark.cuda
@pytest.mark.parametrize("low,high", [
    ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
    ([-2.0, 0.2, -0.5], [0.7, 2.0, 3.0]),
])
def test_compiled_affine_mapping_reload_blind_and_owned_actions(cuda_device, low, high):
    low, high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    adapter = ObservationAdapter(np.linspace(-0.3, 0.3, 5, dtype=np.float32),
                                 np.linspace(0.5, 1.5, 5, dtype=np.float32))
    rng = np.random.default_rng(41)
    parameters = rng.normal(0, 0.3, size=(2, 18)).astype(np.float32)
    # Exercise both clipping ends as well as unsaturated affine proposals.
    parameters[0, -3:] = [-3.0, 0.0, 3.0]
    observations = rng.normal(size=(6, 5)).astype(np.float32)
    policy = _AffinePolicy(parameters, 3, adapter, low, high, cuda_device)
    expected = _numpy_actions(parameters, observations, adapter, low, high, 3)
    first = policy.action(observations)
    np.testing.assert_allclose(first, expected, atol=1e-6, rtol=1e-5)
    snapshot = first.copy()

    replacement = parameters[::-1].copy()
    replacement[:, :15] *= -0.5
    policy.reload(replacement)
    np.testing.assert_allclose(policy.action(observations),
                               _numpy_actions(replacement, observations, adapter, low, high, 3),
                               atol=1e-6, rtol=1e-5)
    policy.reload(replacement, blind=True)
    np.testing.assert_allclose(policy.action(observations),
                               _numpy_actions(replacement, observations, adapter, low, high, 3, blind=True),
                               atol=1e-6, rtol=1e-5)
    policy.reload(parameters)
    np.testing.assert_array_equal(policy.action(observations), first)
    np.testing.assert_array_equal(first, snapshot)


@pytest.mark.cuda
def test_every_normalized_sensor_affects_every_actuator_without_saturation(cuda_device):
    observation_dim, action_dim = 17, 6
    adapter = ObservationAdapter(np.linspace(-2, 2, observation_dim, dtype=np.float32),
                                 np.linspace(0.25, 4, observation_dim, dtype=np.float32))
    weights = np.linspace(0.01, 0.1, observation_dim * action_dim, dtype=np.float32).reshape(action_dim, -1)
    parameters = np.concatenate([weights.ravel(), np.zeros(action_dim, dtype=np.float32)])[None]
    observations = np.repeat(adapter.mean[None], observation_dim + 1, axis=0)
    observations[1:] += np.diag(adapter.scale * 0.25)
    policy = _AffinePolicy(parameters, observation_dim + 1, adapter,
                           -np.ones(action_dim, dtype=np.float32), np.ones(action_dim, dtype=np.float32), cuda_device)
    actions = policy.action(observations)
    response = actions[1:] - actions[0]
    assert np.all(np.isfinite(response))
    assert np.all(response > 0.002)
    np.testing.assert_allclose(response, weights.T * 0.25, atol=1e-7, rtol=1e-5)
    # A caller mutating calibration data cannot change the already-fixed policy.
    adapter.mean[:] = 100
    adapter.scale[:] = 100
    np.testing.assert_array_equal(policy.action(observations), actions)
    policy.reload(parameters, blind=True)
    np.testing.assert_array_equal(policy.action(observations), np.zeros_like(actions))


def _native_reference(env_id, parameters, seeds, horizon, adapter):
    """Raw independent single-episode rollouts; never accumulate an autoreset."""
    from cleanrl.shared.mujoco_env import make_mujoco_vector_env

    env = make_mujoco_vector_env(env_id, 1, backend="native", num_threads=1, copy=False)
    returns = np.zeros((len(parameters), len(seeds)), dtype=np.float64)
    lengths = np.zeros(returns.shape, dtype=np.int64)
    try:
        low = env.single_action_space.low.astype(np.float32)
        high = env.single_action_space.high.astype(np.float32)
        for p, row in enumerate(parameters):
            for e, seed in enumerate(seeds):
                observation, _ = env.reset(seed=[seed])
                for step in range(horizon):
                    action = _numpy_actions(row[None], observation, adapter, low, high, 1)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    returns[p, e] += reward[0]
                    lengths[p, e] = step + 1
                    if terminated[0] or truncated[0]:
                        break
    finally:
        env.close()
    return returns, lengths


@pytest.mark.cuda
def test_raw_halfcheetah_thousand_step_returns_seed_mapping_and_reload(cuda_device):
    adapter = ObservationAdapter(np.zeros(17, dtype=np.float32), np.ones(17, dtype=np.float32))
    parameters = np.zeros((2, 108), dtype=np.float32)
    parameters[0, -6:] = [0.25, -0.125, 0.5, -0.25, 0.125, -0.5]
    parameters[1, -6:] = [-0.125, 0.25, -0.25, 0.5, -0.5, 0.125]
    seeds = [7, 19]
    expected, lengths = _native_reference("HalfCheetah-v4", parameters, seeds, 1000, adapter)
    np.testing.assert_array_equal(lengths, np.full((2, 2), 1000))
    with AffineEvaluator("HalfCheetah-v4", adapter, cuda_device, 1) as evaluator:
        assert (evaluator.observation_dim, evaluator.action_dim, evaluator.parameter_dim) == (17, 6, 108)
        first = evaluator.evaluate(parameters, seeds, 1000)
        np.testing.assert_allclose(first, expected, atol=1e-9, rtol=1e-10)
        assert evaluator.evaluation_transitions == 4000
        np.testing.assert_array_equal(evaluator.evaluate(parameters, seeds, 1000), first)
        np.testing.assert_allclose(evaluator.evaluate(parameters[::-1].copy(), seeds[::-1], 1000),
                                   expected[::-1, ::-1], atol=1e-9, rtol=1e-10)
        # A separate singleton shape uses the retained dimension environment.
        np.testing.assert_allclose(evaluator.evaluate(parameters[:1], seeds[:1], 1000),
                                   expected[:1, :1], atol=1e-9, rtol=1e-10)
        blind_parameters = parameters.copy()
        blind_parameters[:, :-6] = 0.25
        np.testing.assert_allclose(evaluator.evaluate(blind_parameters, seeds, 1000, blind=True),
                                   expected, atol=1e-9, rtol=1e-10)
        # Horizon beyond TimeLimit still returns only the complete first episode.
        np.testing.assert_array_equal(evaluator.evaluate(parameters, seeds, 1200), first)
        assert evaluator.evaluation_transitions == 21000


@pytest.mark.cuda
def test_native_terminal_reward_masking_counts_finished_vector_slots(cuda_device):
    adapter = ObservationAdapter(np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32))
    parameters = np.zeros((2, 36), dtype=np.float32)
    parameters[0, -3:] = [-0.25, 0.125, -0.5]
    parameters[1, -3:] = [0.0, 0.125, 0.25]
    seeds = [1, 5, 17]
    expected, lengths = _native_reference("Hopper-v4", parameters, seeds, 1000, adapter)
    assert np.any(lengths < lengths.max()), "exercise mixed active and completed episodes"
    with AffineEvaluator("Hopper-v4", adapter, cuda_device, 1) as evaluator:
        actual = evaluator.evaluate(parameters, seeds, 1000)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)
        assert evaluator.evaluation_transitions == parameters.shape[0] * len(seeds) * lengths.max()
        np.testing.assert_array_equal(evaluator.evaluate(parameters, seeds, 1000), actual)


@pytest.mark.parametrize("scale", [np.zeros(2), np.array([1.0, -1.0]), np.array([1.0, np.nan])])
def test_invalid_adapter_scale_is_rejected_before_cuda_allocation(scale):
    with pytest.raises(ValueError, match="scale"):
        _AffinePolicy(np.zeros((1, 3)), 1, ObservationAdapter(np.zeros(2), scale),
                      np.array([-1.0]), np.array([1.0]), torch.device("cuda"))


def test_adapter_dimension_mismatch_is_rejected_before_cuda_allocation():
    with pytest.raises(ValueError, match="dimensions"):
        _AffinePolicy(np.zeros((1, 3)), 1, ObservationAdapter(np.zeros(2), np.ones(3)),
                      np.array([-1.0]), np.array([1.0]), torch.device("cuda"))


def test_cpu_policy_fallback_is_rejected():
    adapter = ObservationAdapter(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="CUDA"):
        AffineEvaluator("HalfCheetah-v4", adapter, torch.device("cpu"), 1)
