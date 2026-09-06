"""Shared synchronous-policy collector: one policy version, fully batched envs.

Every step runs the whole vector environment on the current policy through a
captured :class:`RolloutStepGraph`; no stale actors, queued future actions,
reordered transitions or split batches. The policy callback returns CUDA
tensors in a mapping containing ``action``; every other tensor is recorded in
rollout storage owned by the graph (e.g. native_action, alpha, beta, value).
Returned storage is overwritten by the next collect.
"""

from dataclasses import dataclass
import time

import numpy as np
import torch

from cleanrl.shared.ppo_loop import TruncationBootstrapCache
from cleanrl.shared.rollout_graph import RolloutStepGraph
from cleanrl.shared.rollout_transfer import RolloutBatch, RolloutTransfer
from cleanrl.shared.timing import PhaseTimer


@dataclass
class CollectedRollout:
    observations: torch.Tensor
    policy: dict[str, torch.Tensor]
    transitions: RolloutBatch
    next_observation: torch.Tensor
    bootstraps: TruncationBootstrapCache
    transitions_collected: int
    wall_seconds: float


class OnPolicyCollector:
    """Compose environment, normalization, graph stepping and rollout transfer.

    ``policy_fn`` must satisfy the :class:`RolloutStepGraph` contract (device
    only, shape static, no CUDA-graph-tree compilation). ``episode_callback
    (infos, total_steps)`` runs on the controlling thread with the original
    raw episode statistics; ``total_steps`` can start at the number of
    phase-warmup transitions. Call ``set_observation`` with the normalized
    warmup result before the first collection. The caller owns and closes the
    environment unless it uses this object's ``close`` method.
    """

    def __init__(self, envs, num_steps, policy_fn, obs_norm, reward_norm, *,
                 device="cuda", non_blocking=True, store_transition_observations=True,
                 episode_callback=None):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("OnPolicyCollector requires CUDA")
        self.envs = envs
        self.num_steps = num_steps
        self.num_envs = envs.num_envs
        self.obs_norm = obs_norm
        self.reward_norm = reward_norm
        self.episode_callback = episode_callback
        self.store_transitions = store_transition_observations
        obs_shape = envs.single_observation_space.shape
        self.graph = RolloutStepGraph(policy_fn, num_steps, self.num_envs, obs_shape, self.device)
        self.transfer = RolloutTransfer(num_steps, self.num_envs, obs_shape, self.device,
                                        non_blocking=non_blocking,
                                        store_transition_observations=store_transition_observations)
        self.bootstraps = TruncationBootstrapCache(num_steps, self.num_envs, obs_shape)
        self.next_observation_host = None
        self.total_steps = 0
        self.timer = PhaseTimer()

    @property
    def observations(self):
        return self.graph.observations

    @property
    def policy_buffers(self):
        return self.graph.outputs

    @property
    def next_observation(self):
        """Device copy of the observation the next collection starts from."""
        if self.next_observation_host is None:
            return None
        return torch.as_tensor(self.next_observation_host, device=self.device)

    def set_observation(self, normalized_observation, *, total_steps=0):
        self.next_observation_host = np.array(normalized_observation, dtype=np.float32, copy=True)
        self.total_steps = total_steps

    def collect(self):
        if self.next_observation_host is None:
            raise RuntimeError("set the normalized initial observation before collecting")
        started = time.perf_counter()
        self.bootstraps.reset()
        self.graph.reset()
        next_obs = self.next_observation_host
        for step in range(self.num_steps):
            self.timer.start("rollout", use_cuda=False)
            host_action = self.graph.step(next_obs)
            if not np.isfinite(host_action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            self.timer.stop()
            self.timer.start("env", use_cuda=False)
            raw_obs, raw_reward, terms, truncs, infos = self.envs.step(host_action)
            self.timer.stop()
            self.timer.start("normalize_transfer", use_cuda=False)
            reward = self.reward_norm.normalize(raw_reward, terms) if self.reward_norm is not None else raw_reward
            next_obs, transition_obs = self.obs_norm.normalize_step(raw_obs, terms, truncs, infos)
            self.bootstraps.push_normalized(step, truncs, transition_obs)
            if self.store_transitions:
                # The explicit-next-value path preserves v30's termination behavior.
                transition_next = next_obs
                if truncs.any():
                    transition_next = next_obs.copy()
                    transition_next[truncs] = transition_obs[truncs]
                self.transfer.push(step, reward, terms, truncs, transition_next)
            else:
                self.transfer.push(step, reward, terms, truncs)
            self.total_steps += self.num_envs
            if self.episode_callback is not None:
                self.episode_callback(infos, self.total_steps)
            self.timer.stop()
        self.next_observation_host = next_obs
        batch = self.transfer.upload()
        next_observation = self.graph.stage_observation(next_obs)
        return CollectedRollout(self.graph.observations, self.graph.outputs, batch, next_observation,
                               self.bootstraps, self.num_steps * self.num_envs,
                               time.perf_counter() - started)

    def close(self):
        try:
            self.envs.close()
        finally:
            self.transfer.close()
