"""Collector semantics across blocking/non-blocking rollout transfers."""

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
        for non_blocking in (False, True):
            env = gym.vector.SyncVectorEnv([CountingEnv, CountingEnv])
            obs_norm = VectorObsNorm(2, (2,))

            def policy(observations):
                return {"action": torch.tanh(observations[:, :1]), "value": observations.sum(-1)}

            collector = OnPolicyCollector(env, 7, policy, obs_norm, VectorRewardNorm(2, 0.99),
                                           non_blocking=non_blocking)
            collector.set_observation(obs_norm.normalize(env.reset(seed=1)[0]))
            collectors.append(collector)
        for _ in range(3):
            left, right = [collector.collect() for collector in collectors]
            for a, b in ((left.observations, right.observations),
                         (left.policy["value"], right.policy["value"]),
                         (left.next_observation, right.next_observation),
                         *zip(left.transitions[:4], right.transitions[:4])):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert left.transitions_collected == right.transitions_collected == 14
            assert len(left.bootstraps) == len(right.bootstraps) > 0
            # Truncated slots carry the final observation; the graph stored the reset observation next.
            truncated = left.transitions.truncations.bool()[:-1]
            assert truncated.any()
            assert not torch.equal(left.transitions.transition_observations[:-1][truncated],
                                   left.observations[1:][truncated])
            np.testing.assert_array_equal(collectors[0].obs_norm.means, collectors[1].obs_norm.means)
    finally:
        for collector in collectors:
            collector.close()
