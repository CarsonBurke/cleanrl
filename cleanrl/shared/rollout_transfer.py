"""Reusable pinned staging for ordered continuous-control rollouts.

Only the next policy observation is needed on the GPU at each environment
step (none at all with a host actor). Rewards, termination flags, optional
transition observations and any extra named float32 fields (observations,
native actions, ...) are copied into one pinned host block and uploaded
together once after the rollout. This removes several small transfers and
device allocations from every step while preserving their float32 conversion
and the transition/reset distinction.
"""

from typing import NamedTuple

import numpy as np
import torch


class RolloutBatch(NamedTuple):
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    transition_observations: torch.Tensor | None
    fields: dict[str, torch.Tensor]


class RolloutTransfer:
    """Pack rollout fields into one allocation and one end-of-rollout transfer.

    ``observation`` returns reusable device storage: snapshot it into the rollout
    buffer before the next call. ``upload`` similarly returns views overwritten
    by the next upload. On CUDA, host allocations are pinned. By default copies complete
    before returning so callers may immediately reuse/mutate their NumPy inputs;
    no implicit dependency on later action transfers or stream events is needed.

    With ``non_blocking=True``, CUDA copies are queued on the current stream.
    Consume returned tensors on that same stream, and finish enqueueing their
    consumers before calling the corresponding method again. Pinned observation
    slots use events to prevent the CPU overwriting an in-flight DMA source.
    Rollout host storage is protected similarly on its next ``push``. NumPy
    inputs may always be mutated immediately after either method returns.

    ``fields`` declares extra per-step float32 arrays by ``{name: shape}``;
    ``push`` takes them as keywords and ``upload`` returns them in
    ``RolloutBatch.fields`` as ``(num_steps, num_envs, *shape)`` views.

    This helper is for fields not consumed until the rollout ends. Recurrent
    policies that need boundary flags each step must transfer those separately.
    """

    def __init__(
        self, num_steps, num_envs, obs_shape, device, *, store_transition_observations=False,
        non_blocking=False, staging_slots=2, fields=None,
    ):
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")
        if staging_slots <= 0:
            raise ValueError("staging_slots must be positive")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.non_blocking = bool(non_blocking and self.device.type == "cuda")
        self._upload_event = None
        self._upload_pending = False
        observation_size = int(np.prod(self.obs_shape, dtype=np.int64))
        field_size = num_steps * num_envs
        packed_size = 3 * field_size
        if store_transition_observations:
            packed_size += field_size * observation_size
        extra_shapes = {name: tuple(shape) for name, shape in (fields or {}).items()}
        extra_offsets = {}
        for name, shape in extra_shapes.items():
            extra_offsets[name] = packed_size
            packed_size += field_size * int(np.prod(shape, dtype=np.int64))
        pinned = self.device.type == "cuda"
        self._host = torch.empty(packed_size, dtype=torch.float32, pin_memory=pinned)
        self._device = torch.empty(packed_size, dtype=torch.float32, device=self.device)
        host = self._host.numpy()
        self._host_fields = host[: 3 * field_size].reshape(3, num_steps, num_envs)
        fields = self._device[: 3 * field_size].view(3, num_steps, num_envs)
        self._host_transitions = None
        transitions = None
        transitions_end = 3 * field_size
        if store_transition_observations:
            shape = (num_steps, num_envs) + self.obs_shape
            transitions_end += field_size * observation_size
            self._host_transitions = host[3 * field_size : transitions_end].reshape(shape)
            transitions = self._device[3 * field_size : transitions_end].view(shape)
        self._host_extra = {}
        extra = {}
        for name, shape in extra_shapes.items():
            start = extra_offsets[name]
            stop = start + field_size * int(np.prod(shape, dtype=np.int64))
            full_shape = (num_steps, num_envs) + shape
            self._host_extra[name] = host[start:stop].reshape(full_shape)
            extra[name] = self._device[start:stop].view(full_shape)
        self._batch = RolloutBatch(fields[0], fields[1], fields[2], transitions, extra)
        # ``push`` writes one row of each host field per step. Materializing the
        # row views once turns each write into a plain ``row[...] = value``
        # (~0.18us) instead of integer-indexing the block per call (~0.41us),
        # and drops the per-name dict lookups from the hot path entirely.
        self._step_rewards = tuple(self._host_fields[0])
        self._step_terminations = tuple(self._host_fields[1])
        self._step_truncations = tuple(self._host_fields[2])
        self._step_transitions = (
            None if self._host_transitions is None else tuple(self._host_transitions)
        )
        self._step_extra = tuple(
            (name, tuple(block)) for name, block in self._host_extra.items()
        )
        self._extra_count = len(self._host_extra)
        self._observation_hosts = tuple(
            torch.empty((num_envs,) + self.obs_shape, dtype=torch.float32, pin_memory=pinned)
            for _ in range(staging_slots if self.non_blocking else 1)
        )
        self._observation_events = [None] * len(self._observation_hosts)
        self._observation_arrays = tuple(host.numpy() for host in self._observation_hosts)
        self._observation_slot = 0
        self._observation_host = self._observation_hosts[0]
        self._observation_array = self._observation_host.numpy()
        self._observation_device = torch.empty_like(self._observation_host, device=self.device)

    def observation(self, observations):
        """Transfer the policy's next observation into reusable device storage."""
        if np.shape(observations) != self._observation_array.shape:
            raise ValueError(f"observations must have shape {self._observation_array.shape}")
        slot = self._observation_slot
        event = self._observation_events[slot]
        if event is not None:
            event.synchronize()
        array = self._observation_arrays[slot]
        # Shape is validated above, so a direct assignment is the same
        # unsafe-casting copy np.copyto would do, at a third of the dispatch.
        array[...] = observations
        self._observation_device.copy_(self._observation_hosts[slot], non_blocking=self.non_blocking)
        if self.non_blocking:
            if event is None:
                event = torch.cuda.Event()
                self._observation_events[slot] = event
            event.record(torch.cuda.current_stream(self.device))
            self._observation_slot = (slot + 1) % len(self._observation_hosts)
        return self._observation_device

    def push(self, step, rewards, terminations, truncations, transition_observations=None, **fields):
        """Snapshot one step; await prior host reads, without new device transfers."""
        if not 0 <= step < self.num_steps:
            raise IndexError(f"rollout step {step} is outside [0, {self.num_steps})")
        if len(fields) != self._extra_count:
            raise ValueError(f"push expects exactly the declared fields {sorted(self._host_extra)}")
        if self._upload_pending:
            self._upload_event.synchronize()
            self._upload_pending = False
        # Assignment into the prebuilt row views casts exactly as the previous
        # np.copyto(..., casting="unsafe") did; float64 rewards and bool flags
        # land as float32 identically.
        self._step_rewards[step][...] = rewards
        self._step_terminations[step][...] = terminations
        self._step_truncations[step][...] = truncations
        transitions = self._step_transitions
        if transitions is not None:
            if transition_observations is None:
                raise ValueError("transition observations are required for this rollout")
            transitions[step][...] = transition_observations
        elif transition_observations is not None:
            raise ValueError("enable store_transition_observations to record transitions")
        for name, rows in self._step_extra:
            try:
                value = fields[name]
            except KeyError:
                raise ValueError(
                    f"push expects exactly the declared fields {sorted(self._host_extra)}"
                ) from None
            rows[step][...] = value

    def upload(self):
        """Upload a fully populated rollout in one copy; return contiguous views."""
        self._device.copy_(self._host, non_blocking=self.non_blocking)
        if self.non_blocking:
            if self._upload_event is None:
                self._upload_event = torch.cuda.Event()
            self._upload_event.record(torch.cuda.current_stream(self.device))
            self._upload_pending = True
        return self._batch

    def close(self):
        """Finish outstanding host reads before releasing or repurposing storage."""
        if self._upload_pending:
            self._upload_event.synchronize()
            self._upload_pending = False
        for event in self._observation_events:
            if event is not None:
                event.synchronize()


class ActionTransfer:
    """Reusable asynchronous device-to-host action staging.

    ``submit`` enqueues the copy after policy inference on the current CUDA
    stream. Perform independent CPU bookkeeping, then call ``wait`` before
    reading the returned NumPy array or submitting it to an environment. This
    waits for the action event only, not unrelated CUDA streams. The array is
    overwritten by the next ``submit``: an asynchronous environment must first
    snapshot it, or finish consuming it before that next submission.

    The source tensor must be produced on the current stream or explicitly
    synchronized with it. Enqueue any source mutation on that same stream.
    Dtype is explicit so the helper never silently changes policy actions.
    """

    def __init__(self, action_shape, device, *, dtype=torch.float32):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._host = torch.empty(tuple(action_shape), dtype=dtype, pin_memory=self.device.type == "cuda")
        self._array = self._host.numpy()
        self._event = None
        self._pending = False

    def submit(self, actions):
        if self._pending:
            raise RuntimeError("wait for the pending action transfer before submitting another")
        if tuple(actions.shape) != tuple(self._host.shape) or actions.dtype != self._host.dtype:
            raise ValueError(f"actions must have shape {tuple(self._host.shape)} and dtype {self._host.dtype}")
        if actions.device != self.device:
            raise ValueError(f"actions must be on device {self.device}")
        self._host.copy_(actions.detach(), non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            actions.record_stream(stream)
            if self._event is None:
                self._event = torch.cuda.Event()
            self._event.record(stream)
        self._pending = True

    def wait(self):
        if not self._pending:
            raise RuntimeError("no action transfer is pending")
        if self._event is not None:
            self._event.synchronize()
        self._pending = False
        return self._array

    def close(self):
        if self._pending:
            self.wait()
