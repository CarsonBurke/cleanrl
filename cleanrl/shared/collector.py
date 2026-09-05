"""Shared synchronous-policy collector with overlapped CPU environment work.

One policy version supplies the entire rollout. No stale actors, queued future
actions, reordered transitions, or changed batch sizes are introduced. The
policy callback returns CUDA tensors in a mapping containing ``action``; every
other tensor is recorded into reusable rollout storage (e.g. native_action,
alpha, beta and value). Returned storage is overwritten by the next collect.
"""

from dataclasses import dataclass
import time

import numpy as np
import torch

from cleanrl.shared.async_env import AsyncEnvStepper
from cleanrl.shared.ppo_loop import TruncationBootstrapCache
from cleanrl.shared.rollout_transfer import ActionTransfer, RolloutBatch, RolloutTransfer
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
    """Compose environment, normalization and transfer utilities for any policy.

    ``episode_callback(infos, total_steps)`` runs on the controlling thread;
    it sees original raw episode statistics. ``total_steps`` can start at the
    number of phase-warmup transitions. Call ``set_observation`` with the
    normalized warmup result before the first collection. The caller owns and
    closes the environment unless it uses this object's ``close`` method.
    """

    def __init__(self, envs, num_steps, policy_fn, obs_norm, reward_norm, *,
                 device="cuda", non_blocking=True, async_env=True,
                 store_transition_observations=True, episode_callback=None,
                 mark_cuda_steps=True):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("OnPolicyCollector requires CUDA")
        self.envs = AsyncEnvStepper(envs) if async_env else envs
        self.async_env = async_env
        self.num_steps = num_steps
        self.num_envs = envs.num_envs
        self.policy_fn = policy_fn
        self.obs_norm = obs_norm
        self.reward_norm = reward_norm
        self.episode_callback = episode_callback
        self.mark_cuda_steps = mark_cuda_steps
        self.store_transitions = store_transition_observations
        obs_shape = envs.single_observation_space.shape
        self.transfer = RolloutTransfer(num_steps, self.num_envs, obs_shape, self.device,
                                        non_blocking=non_blocking,
                                        store_transition_observations=store_transition_observations)
        self.action_transfer = ActionTransfer((self.num_envs,) + envs.single_action_space.shape, self.device)
        self.observations = torch.empty((num_steps, self.num_envs) + obs_shape, device=self.device)
        self.policy_buffers = {}
        self._policy_keys = None
        self.bootstraps = TruncationBootstrapCache(num_steps, self.num_envs, obs_shape)
        self.next_observation = None
        self.total_steps = 0
        self.timer = PhaseTimer()

    def set_observation(self, normalized_observation, *, total_steps=0):
        self.next_observation = self.transfer.observation(normalized_observation)
        self.total_steps = total_steps

    @torch.no_grad()
    def collect(self):
        if self.next_observation is None:
            raise RuntimeError("set the normalized initial observation before collecting")
        started = time.perf_counter()
        self.bootstraps.reset()
        for step in range(self.num_steps):
            self.timer.start("rollout", use_cuda=False)
            if self.mark_cuda_steps:
                torch.compiler.cudagraph_mark_step_begin()
            outputs = self.policy_fn(self.next_observation)
            if "action" not in outputs:
                raise ValueError("policy callback must return an action tensor")
            keys = set(outputs) - {"action"}
            if self._policy_keys is not None and keys != self._policy_keys:
                raise ValueError("policy output keys changed during collection")
            self._policy_keys = keys
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor) or value.device != self.next_observation.device or value.ndim == 0 or value.shape[0] != self.num_envs:
                    raise ValueError("policy outputs must be batched tensors on the collector CUDA device")
                if key != "action" and key not in self.policy_buffers:
                    self.policy_buffers[key] = torch.empty((self.num_steps,) + value.shape, dtype=value.dtype, device=value.device)
            # D2H begins before independent rollout storage kernels are queued.
            self.action_transfer.submit(outputs["action"])
            host_action = self.action_transfer.wait()
            if not np.isfinite(host_action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            self.timer.stop()
            self.timer.start("env", use_cuda=False)
            if self.async_env:
                self.envs.step_async(host_action)
            self.observations[step].copy_(self.next_observation)
            for key, buffer in self.policy_buffers.items():
                if (outputs[key].shape, outputs[key].dtype) != (buffer.shape[1:], buffer.dtype):
                    raise ValueError("policy output shape or dtype changed")
                buffer[step].copy_(outputs[key])
            result = self.envs.step_wait() if self.async_env else self.envs.step(host_action)
            self.timer.stop()
            self.timer.start("normalize_transfer", use_cuda=False)
            raw_obs, raw_reward, terms, truncs, infos = result
            reward = self.reward_norm.normalize(raw_reward, terms) if self.reward_norm is not None else raw_reward
            next_obs, transition_obs = self.obs_norm.normalize_step(raw_obs, terms, truncs, infos)
            self.bootstraps.push_normalized(step, truncs, transition_obs)
            # The explicit-next-value path preserves v30's termination behavior.
            if self.store_transitions:
                transition_next = next_obs
                if np.any(truncs):
                    transition_next = next_obs.copy()
                    transition_next[truncs] = transition_obs[truncs]
                self.transfer.push(step, reward, terms, truncs, transition_next)
            else:
                self.transfer.push(step, reward, terms, truncs)
            self.next_observation = self.transfer.observation(next_obs)
            self.total_steps += self.num_envs
            if self.episode_callback is not None:
                self.episode_callback(infos, self.total_steps)
            self.timer.stop()
        batch = self.transfer.upload()
        return CollectedRollout(self.observations, self.policy_buffers, batch,
                                self.next_observation, self.bootstraps,
                                self.num_steps * self.num_envs, time.perf_counter() - started)

    def close(self):
        try:
            self.envs.close()
        finally:
            try:
                self.action_transfer.close()
            finally:
                self.transfer.close()
