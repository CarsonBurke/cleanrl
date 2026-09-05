"""Semantic parity tests; run MuJoCo stepping through the machine-wide mlq.

These are fixed-action correctness cases, never reduced training runs or speed
evidence. Throughput measurement lives in benchmark_mujoco_throughput.py.
"""

import gymnasium as gym
import numpy as np
import pytest

from cleanrl.shared.legacy_normalization import CanonicalLegacyNormalization, is_canonical_clip
from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv, ThreadedMujocoVectorEnv, make_mujoco_vector_env
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm, make_raw_continuous_env


ENV_IDS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key != "t":  # Wall-clock episode durations necessarily differ.
                assert_tree_equal(left[key], right[key])
    elif isinstance(left, np.ndarray) and left.dtype == object:
        assert left.shape == right.shape
        for a, b in zip(left.flat, right.flat):
            assert_tree_equal(a, b)
    elif left is None:
        assert right is None
    else:
        np.testing.assert_array_equal(left, right)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("backend,threads", [("native", 1), ("native", 2), ("threaded", 2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_task_and_normalization_parity(env_id, backend, threads, dtype):
    kwargs = dict(max_episode_steps=7, forward_reward_weight=1.3,
                  ctrl_cost_weight=0.123, reset_noise_scale=0.007,
                  exclude_current_positions_from_observation=False)
    reference = make_mujoco_vector_env(env_id, 3, backend="sync", **kwargs)
    candidate = make_mujoco_vector_env(env_id, 3, backend=backend, num_threads=threads, **kwargs)
    try:
        raw_a, info_a = reference.reset(seed=1)
        raw_b, info_b = candidate.reset(seed=1)
        assert_tree_equal((raw_a, info_a)[0], raw_b)
        assert_tree_equal(info_a, info_b)
        obs_a = VectorObsNorm(3, reference.single_observation_space.shape)
        obs_b = VectorObsNorm(3, candidate.single_observation_space.shape)
        rew_a, rew_b = VectorRewardNorm(3, 0.99), VectorRewardNorm(3, 0.99)
        assert_tree_equal(obs_a.normalize(raw_a), obs_b.normalize(raw_b))
        rng = np.random.default_rng(1)
        for step in range(23):
            # Includes clipping on both sides of the action range.
            actions = rng.normal(size=reference.action_space.shape).astype(dtype)
            a, b = reference.step(actions), candidate.step(actions)
            for expected, actual in zip(a, b):
                assert_tree_equal(expected, actual)
            norm_a = obs_a.normalize_step(a[0], a[2], a[3], a[4])
            norm_b = obs_b.normalize_step(b[0], b[2], b[3], b[4])
            for expected, actual in zip(norm_a, norm_b):
                assert_tree_equal(expected, actual)
            assert_tree_equal(rew_a.normalize(a[1], a[2]), rew_b.normalize(b[1], b[2]))
            if step == 3:
                # Staggered warmup still calls a wrapped single-env reset.
                reset_a, _ = reference.envs[1].reset()
                reset_b, _ = candidate.envs[1].reset()
                assert_tree_equal(reset_a, reset_b)
                obs_a.normalize(reset_a[None], rows=slice(1, 2))
                obs_b.normalize(reset_b[None], rows=slice(1, 2))
        for a, b in zip(reference.envs, candidate.envs):
            for attribute in ("episode_returns", "episode_lengths", "episode_count",
                              "return_queue", "length_queue", "_elapsed_steps"):
                assert_tree_equal(np.asarray(a.get_wrapper_attr(attribute)),
                                  np.asarray(b.get_wrapper_attr(attribute)))
    finally:
        reference.close()
        candidate.close()


@pytest.mark.parametrize("env_id", ["Hopper-v4", "Walker2d-v4"])
@pytest.mark.parametrize("terminate", [True, False])
def test_unhealthy_reward_and_termination(env_id, terminate):
    kwargs = dict(terminate_when_unhealthy=terminate, healthy_reward=2.7,
                  healthy_z_range=(5.0, 6.0), max_episode_steps=2)
    reference = make_mujoco_vector_env(env_id, 2, backend="sync", **kwargs)
    candidate = make_mujoco_vector_env(env_id, 2, backend="native", num_threads=2, **kwargs)
    try:
        reference.reset(seed=[1, 9])
        candidate.reset(seed=[1, 9])
        for _ in range(3):
            actions = np.zeros(reference.action_space.shape, dtype=np.float32)
            for expected, actual in zip(reference.step(actions), candidate.step(actions)):
                assert_tree_equal(expected, actual)
    finally:
        reference.close()
        candidate.close()


def test_native_infers_id_and_rejects_unknown_wrappers():
    thunk = make_raw_continuous_env("HalfCheetah-v4", 0, False, "parity")
    env = NativeMujocoVectorEnv([thunk])
    try:
        assert env._env_id == "HalfCheetah-v4"
    finally:
        env.close()
    with pytest.raises(ValueError, match="wrapper stack"):
        NativeMujocoVectorEnv([lambda: gym.Wrapper(thunk())])


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("observation_mode", ["original", "cast", "declared_float32", "reshape"])
def test_legacy_normalizer_wrapper_stack_exact_parity(env_id, observation_mode):
    def create():
        env = gym.make(env_id, max_episode_steps=4)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env, epsilon=1e-7)
        def transform(obs):
            obs = np.clip(obs, -10, 10)
            if observation_mode == "reshape":
                return obs.reshape(1, -1)
            return obs.astype(np.float32) if observation_mode != "original" else obs
        env = gym.wrappers.TransformObservation(env, transform)
        if observation_mode == "declared_float32":
            env.observation_space = gym.spaces.Box(-10, 10, env.observation_space.shape, dtype=np.float32)
        elif observation_mode == "reshape":
            env.observation_space = gym.spaces.Box(-10, 10, (1,) + env.observation_space.shape, dtype=np.float64)
        env = gym.wrappers.NormalizeReward(env, gamma=0.97, epsilon=1e-7)
        return gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    reference = gym.vector.SyncVectorEnv([create, create])
    candidate = NativeMujocoVectorEnv([create, create], num_threads=2)
    try:
        for expected, actual in zip(reference.reset(seed=1), candidate.reset(seed=1)):
            assert_tree_equal(expected, actual)
        rng = np.random.default_rng(1)
        for step in range(11):
            action = rng.normal(size=reference.action_space.shape).astype(np.float32)
            for expected, actual in zip(reference.step(action), candidate.step(action)):
                assert_tree_equal(expected, actual)
            if step == 2:
                for expected, actual in zip(reference.envs[0].reset(), candidate.envs[0].reset()):
                    assert_tree_equal(expected, actual)
        for env_a, env_b in zip(reference.envs, candidate.envs):
            for attribute in ("obs_rms", "return_rms"):
                a, b = env_a.get_wrapper_attr(attribute), env_b.get_wrapper_attr(attribute)
                for field in ("mean", "var", "count"):
                    assert_tree_equal(getattr(a, field), getattr(b, field))
            assert_tree_equal(env_a.get_wrapper_attr("returns"), env_b.get_wrapper_attr("returns"))
    finally:
        reference.close()
        candidate.close()


def test_native_rejects_heterogeneous_reward_configuration():
    def create(weight):
        def thunk():
            env = gym.make("HalfCheetah-v4", ctrl_cost_weight=weight)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return gym.wrappers.ClipAction(env)
        return thunk

    with pytest.raises(ValueError, match="homogeneous"):
        NativeMujocoVectorEnv([create(0.1), create(0.2)])


def test_terminal_transform_observes_completed_episode_statistics():
    def factories(events):
        def create():
            env = gym.make("HalfCheetah-v4", max_episode_steps=2)
            env = gym.wrappers.FlattenObservation(env)
            statistics = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(statistics)
            def observe(obs):
                events.append({
                    "count": statistics.episode_count,
                    "returns": statistics.episode_returns.copy(),
                    "lengths": statistics.episode_lengths.copy(),
                    "return_queue": np.asarray(statistics.return_queue),
                    "length_queue": np.asarray(statistics.length_queue),
                    "elapsed": statistics.get_wrapper_attr("_elapsed_steps"),
                })
                return obs
            return gym.wrappers.TransformObservation(env, observe)
        return [create]

    left_events, right_events = [], []
    reference = gym.vector.SyncVectorEnv(factories(left_events))
    candidate = NativeMujocoVectorEnv(factories(right_events))
    try:
        reference.reset(seed=1)
        candidate.reset(seed=1)
        for _ in range(5):
            actions = np.zeros(reference.action_space.shape, dtype=np.float32)
            for expected, actual in zip(reference.step(actions), candidate.step(actions)):
                assert_tree_equal(expected, actual)
        assert len(left_events) == len(right_events)
        for expected, actual in zip(left_events, right_events):
            assert_tree_equal(expected, actual)
    finally:
        reference.close()
        candidate.close()


@pytest.mark.parametrize("method", ["control_cost", "_step_mujoco_simulation", "state_vector"])
def test_native_rejects_instance_task_overrides(method):
    def create():
        env = make_raw_continuous_env("HalfCheetah-v4", 0, False, "parity")()
        setattr(env.unwrapped, method, lambda *args: None)
        return env

    with pytest.raises(ValueError, match="unmodified"):
        NativeMujocoVectorEnv([create])


def test_threaded_composite_observations():
    class CompositeEnv(gym.Env):
        observation_space = gym.spaces.Dict({"position": gym.spaces.Box(-1, 1, (2,), dtype=np.float32)})
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"position": np.zeros(2, dtype=np.float32)}, {}

        def step(self, action):
            return {"position": action.copy()}, 1.0, True, False, {}

    reference = gym.vector.SyncVectorEnv([CompositeEnv, CompositeEnv])
    candidate = ThreadedMujocoVectorEnv([CompositeEnv, CompositeEnv], num_threads=2)
    try:
        reference.reset(seed=1)
        candidate.reset(seed=1)
        action = np.ones((2, 2), dtype=np.float32)
        for expected, actual in zip(reference.step(action), candidate.step(action)):
            assert_tree_equal(expected, actual)
    finally:
        reference.close()
        candidate.close()


def test_native_copy_false_and_action_shape_validation():
    env = make_mujoco_vector_env("HalfCheetah-v4", 2, backend="native", copy=False)
    try:
        obs, _ = env.reset(seed=1)
        next_obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert obs is next_obs
        with pytest.raises(ValueError, match="action shape"):
            env.step(np.zeros(env.single_action_space.shape))
    finally:
        env.close()


def test_unknown_backend_and_invalid_thread_count():
    with pytest.raises(ValueError, match="unknown MuJoCo"):
        make_mujoco_vector_env("HalfCheetah-v4", 2, backend="typo")
    with pytest.raises(ValueError, match="num_threads"):
        make_mujoco_vector_env("HalfCheetah-v4", 2, num_threads=0)


def _normalization_pipeline(base, index=0):
    obs = gym.wrappers.NormalizeObservation(base, epsilon=1e-8 * (index + 1))
    obs_clip = gym.wrappers.TransformObservation(obs, lambda x: np.clip(x, -10, 10))
    reward = gym.wrappers.NormalizeReward(obs_clip, gamma=0.95 + index * 0.01,
                                          epsilon=1e-8 * (index + 1))
    reward_clip = gym.wrappers.TransformReward(reward, lambda x: np.clip(x, -10, 10))
    return (obs, obs_clip, reward, reward_clip)


def test_batched_legacy_recognizer_does_not_execute_callbacks():
    assert is_canonical_clip(lambda arbitrary_name: np.clip(arbitrary_name, -10, 10))
    events = []

    def custom(value):
        events.append(value)
        return np.clip(value, -10, 10)

    assert not is_canonical_clip(custom)
    assert not is_canonical_clip(lambda value: np.clip(value, -10, 10).astype(np.float32))
    assert not is_canonical_clip(lambda value: np.clip(value, -5, 5))
    assert events == []


@pytest.mark.parametrize("change", ["callback", "normalizer", "float32_statistics"])
def test_batched_legacy_runtime_changes_fall_back_before_mutation(change):
    base = gym.Env()
    base.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
    base.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    row = _normalization_pipeline(base)
    adapter = CanonicalLegacyNormalization.from_wrappers((row,))
    if change == "callback":
        row[1].f = lambda value: value * 2
    elif change == "normalizer":
        row[0].normalize = lambda value: value * 2
    else:
        row[0].obs_rms.var = row[0].obs_rms.var.astype(np.float32)
    before_count = row[0].obs_rms.count
    assert adapter.normalize(np.ones((1, 3)), np.ones(1), np.zeros(1, dtype=bool)) is None
    assert row[0].obs_rms.count == before_count


def test_batched_legacy_normalization_matches_original_wrappers_without_physics():
    class DataEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64)
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.np_random.normal(size=7), {}

        def step(self, action):
            return self.transition

    reference = tuple(_normalization_pipeline(DataEnv(), i) for i in range(3))
    candidate = tuple(_normalization_pipeline(DataEnv(), i) for i in range(3))
    adapter = CanonicalLegacyNormalization.from_wrappers(candidate)
    assert adapter is not None
    for i, (left, right) in enumerate(zip(reference, candidate)):
        assert_tree_equal(left[-1].reset(seed=i)[0], right[-1].reset(seed=i)[0])
    rng = np.random.default_rng(1)
    previous_means = None
    previous_snapshot = None
    for step in range(25):
        observations = rng.normal(size=(3, 7))
        rewards = rng.normal(size=3)
        terms = rng.random(3) < 0.15
        truncs = rng.random(3) < 0.3
        expected_obs = np.empty_like(observations)
        expected_rewards = np.empty_like(rewards)
        for i, row in enumerate(reference):
            row[-1].unwrapped.transition = (observations[i], rewards[i], terms[i], truncs[i], {})
            expected_obs[i], expected_rewards[i], *_ = row[-1].step(None)
        got_obs, got_rewards = adapter.normalize(observations, rewards, terms)
        assert_tree_equal(expected_obs, got_obs)
        assert_tree_equal(expected_rewards, got_rewards)
        if previous_means is not None:
            assert_tree_equal(previous_means, previous_snapshot)
        previous_means = candidate[0][0].obs_rms.mean
        previous_snapshot = previous_means.copy()
        for i, (left, right) in enumerate(zip(reference, candidate)):
            for state_a, state_b in ((left[0].obs_rms, right[0].obs_rms),
                                      (left[2].return_rms, right[2].return_rms)):
                for field in ("mean", "var", "count"):
                    assert_tree_equal(getattr(state_a, field), getattr(state_b, field))
            assert_tree_equal(left[2].returns, right[2].returns)
            if terms[i] or truncs[i] or (step == 3 and i == 1):
                assert_tree_equal(left[-1].reset()[0], right[-1].reset()[0])
        if step == 7:
            # Restored running statistics and changed wrapper options must be
            # read from the original objects, not a stale adapter snapshot.
            for row in (reference[1], candidate[1]):
                row[0].obs_rms.mean = np.full(7, 0.25)
                row[0].obs_rms.count = 123.0
                row[2].return_rms.var = np.float64(7.0)
                row[2].gamma = 0.7
                row[0].epsilon = 1e-5


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("enabled", [True, False])
def test_native_canonical_batched_normalization_exact_parity(env_id, enabled):
    def factory(index):
        def create():
            env = gym.make(env_id, max_episode_steps=4)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            return _normalization_pipeline(env, index)[-1]
        return create

    factories = tuple(factory(i) for i in range(3))
    reference = gym.vector.SyncVectorEnv(factories)
    candidate = NativeMujocoVectorEnv(factories, num_threads=2,
                                     batch_legacy_normalization=enabled)
    assert candidate.uses_batched_normalization is enabled
    try:
        for expected, actual in zip(reference.reset(seed=1), candidate.reset(seed=1)):
            assert_tree_equal(expected, actual)
        rng = np.random.default_rng(1)
        for step in range(19):
            actions = rng.normal(size=reference.action_space.shape).astype(np.float32)
            for expected, actual in zip(reference.step(actions), candidate.step(actions)):
                assert_tree_equal(expected, actual)
            for left, right in zip(reference.envs, candidate.envs):
                for name in ("obs_rms", "return_rms"):
                    for field in ("mean", "var", "count"):
                        assert_tree_equal(getattr(left.get_wrapper_attr(name), field),
                                          getattr(right.get_wrapper_attr(name), field))
            if step == 2:
                assert_tree_equal(reference.envs[1].reset()[0], candidate.envs[1].reset()[0])
    finally:
        reference.close()
        candidate.close()
