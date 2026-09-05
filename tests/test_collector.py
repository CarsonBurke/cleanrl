"""Queued collector semantics across synchronous/asynchronous execution."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.shared.collector import OnPolicyCollector
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.age = 0
        return np.array([0.0, self.np_random.normal()]), {}

    def step(self, action):
        self.age += 1
        return np.array([float(self.age), float(action[0])]), float(self.age), False, self.age == 3, {}


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ordered_collection_preserves_buffers_normalization_and_boundaries():
    collectors = []
    try:
        for asynchronous in (False, True):
            env = gym.vector.SyncVectorEnv([CountingEnv, CountingEnv])
            obs_norm = VectorObsNorm(2, (2,))

            def policy(observations):
                return {"action": torch.tanh(observations[:, :1]), "value": observations.sum(-1)}

            collector = OnPolicyCollector(env, 7, policy, obs_norm, VectorRewardNorm(2, 0.99),
                                           non_blocking=asynchronous, async_env=asynchronous)
            collector.set_observation(obs_norm.normalize(env.reset(seed=1)[0]))
            collectors.append(collector)
        for _ in range(3):
            left, right = [collector.collect() for collector in collectors]
            for a, b in ((left.observations, right.observations),
                         (left.policy["value"], right.policy["value"]),
                         (left.next_observation, right.next_observation),
                         *zip(left.transitions, right.transitions)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert left.transitions_collected == right.transitions_collected == 14
            assert len(left.bootstraps) == len(right.bootstraps)
            for collector_a, collector_b in ((collectors[0], collectors[1]),):
                np.testing.assert_array_equal(collector_a.obs_norm.means, collector_b.obs_norm.means)
    finally:
        for collector in collectors:
            collector.close()
