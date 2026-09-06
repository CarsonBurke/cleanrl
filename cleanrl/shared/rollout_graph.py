"""One CUDA graph per environment step for synchronous, fully batched rollouts.

STANDARD: new PPO-family versions MUST step the policy through
:class:`RolloutStepGraph` instead of per-step compiled calls plus eager
sampling, storage copies and separate transfers. Do not retrofit frozen files.

Why
---
With a small MLP and ``num_envs`` in the tens, the GPU link of the rollout
chain is pure launch and synchronization latency: two ``torch.compile`` graph
launches, a dozen eager kernels for Beta sampling and log-probabilities, four
storage copies, a device-to-host action copy with its own sync, and a blocking
host-to-device observation upload. Measured at 16 envs on an idle RTX 5090 that
is ~200us per step; the same work captured into one graph replays in ~50us and
needs exactly one host synchronization. Under GPU contention from other
processes every extra sync waits for a time slice, so the single sync matters
even more than the kernel count.

What the graph contains
-----------------------
1. ``static_obs.copy_(pinned_obs)`` -- host-to-device of the policy input.
2. ``policy_fn(static_obs)`` -- returns ``{"action": ..., <key>: ...}``.
3. ``index_copy_`` of ``static_obs`` and every non-action output into
   ``(num_steps, ...)`` rollout storage at a device-side step index.
4. ``pinned_action.copy_(action)`` -- device-to-host of the action.
5. ``step_index = (step_index + 1) % num_steps`` -- storage is a ring.

The CPU per step does ``np.copyto`` into pinned memory, ``graph.replay()``,
and one event wait. Sampling inside the graph uses CUDA's graph-aware default
generator (replays advance the Philox offset), so seeded runs stay
deterministic given the same capture.

Contracts
---------
- ``policy_fn`` must be shape-static and device-only (no host syncs, no
  ``.item()``, no data-dependent control flow). It may be Inductor-compiled
  via :func:`graph_compile` (CUDA-graph trees are disabled there; a
  ``mode="reduce-overhead"`` callable cannot be captured inside another graph).
- Outputs must be CUDA tensors with leading dim ``num_envs``; ``action`` is
  copied to the host in its own dtype, the rest are stored in theirs.
- ``step`` returns the pinned action array, overwritten by the next ``step``.
  Environments that keep a reference must copy it first (``SyncVectorEnv``
  and the native MuJoCo backend clip into a new array).
- Replays run on the caller's current stream. Consume rollout storage on that
  stream. Warmup and capture use a private stream and leave the storage
  contents unspecified; ``reset`` zeroes the step index before a rollout.
"""

import numpy as np
import torch


def graph_compile(fn):
    """Inductor-compile ``fn`` for capture inside a manual CUDA graph."""
    return torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})


class RolloutStepGraph:
    def __init__(self, policy_fn, num_steps, num_envs, obs_shape, device, *, warmup=3):
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("RolloutStepGraph requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(obs_shape)
        self.policy_fn = policy_fn
        self._pinned_obs = torch.zeros((self.num_envs,) + self.obs_shape, dtype=torch.float32, pin_memory=True)
        self._pinned_obs_array = self._pinned_obs.numpy()
        self._static_obs = torch.zeros_like(self._pinned_obs, device=self.device)
        self._step_index = torch.zeros((), dtype=torch.long, device=self.device)
        self._index_view = self._step_index.view(1)
        self.observations = torch.empty((self.num_steps,) + self._static_obs.shape, dtype=torch.float32, device=self.device)
        self.outputs = {}
        self._pinned_action = None
        self._pinned_action_array = None
        self._event = torch.cuda.Event()
        self._graph = torch.cuda.CUDAGraph()
        self._capture(warmup)

    def _allocate_outputs(self, outputs):
        if not isinstance(outputs, dict) or "action" not in outputs:
            raise ValueError("policy_fn must return a dict containing an 'action' tensor")
        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor) or value.device != self.device:
                raise ValueError(f"policy output {key!r} must be a tensor on {self.device}")
            if value.ndim == 0 or value.shape[0] != self.num_envs:
                raise ValueError(f"policy output {key!r} must have leading dimension {self.num_envs}")
            if key == "action":
                self._pinned_action = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                self._pinned_action_array = self._pinned_action.numpy()
            else:
                self.outputs[key] = torch.empty((self.num_steps,) + tuple(value.shape), dtype=value.dtype, device=self.device)

    @torch.no_grad()
    def _body(self):
        self._static_obs.copy_(self._pinned_obs, non_blocking=True)
        outputs = self.policy_fn(self._static_obs)
        if self._pinned_action is None:
            self._allocate_outputs(outputs)
        elif set(outputs) != set(self.outputs) | {"action"}:
            raise ValueError("policy output keys changed between warmup calls")
        index = self._index_view
        self.observations.index_copy_(0, index, self._static_obs.unsqueeze(0))
        for key, buffer in self.outputs.items():
            value = outputs[key]
            if value.shape != buffer.shape[1:] or value.dtype != buffer.dtype:
                raise ValueError(f"policy output {key!r} changed shape or dtype between warmup calls")
            buffer.index_copy_(0, index, value.unsqueeze(0))
        action = outputs["action"]
        if action.shape != self._pinned_action.shape or action.dtype != self._pinned_action.dtype:
            raise ValueError("policy action changed shape or dtype between warmup calls")
        self._pinned_action.copy_(action, non_blocking=True)
        self._step_index.add_(1).remainder_(self.num_steps)

    def _capture(self, warmup):
        with torch.cuda.device(self.device):
            # Warmup and capture draw from the generator; restore it so building
            # the graph is invisible to the sampling stream (replays then consume
            # exactly what the equivalent eager calls would).
            rng = torch.cuda.get_rng_state(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self._body()
            stream.synchronize()
            with torch.cuda.graph(self._graph, stream=stream):
                self._body()
            stream.synchronize()
            self._step_index.zero_()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.set_rng_state(rng, self.device)

    def reset(self):
        """Point the next ``step`` at rollout slot 0."""
        self._step_index.zero_()

    def step(self, host_observations):
        """Store obs/outputs at the current slot; return the host action array."""
        np.copyto(self._pinned_obs_array, host_observations, casting="unsafe")
        self._graph.replay()
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        return self._pinned_action_array

    def stage_observation(self, host_observations):
        """Upload observations without stepping; returns the static input tensor.

        Used for the bootstrap value of the observation after the final step.
        The copy completes before returning so the pinned buffer can be reused.
        """
        np.copyto(self._pinned_obs_array, host_observations, casting="unsafe")
        self._static_obs.copy_(self._pinned_obs, non_blocking=False)
        return self._static_obs

    @property
    def observation(self):
        """Static device input: the observations of the most recent ``step``."""
        return self._static_obs
