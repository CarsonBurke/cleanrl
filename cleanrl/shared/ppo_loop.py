"""Vectorized PPO rollout/update-loop primitives.

STANDARD: all new PPO-family versions MUST use these instead of hand-rolled
equivalents. Do not retrofit frozen files.

Contents
--------
- :func:`compute_gae` -- terminal-aware GAE with distinct termination /
  truncation semantics (matches ``ppo_continuous_action.py``): terminations
  cut bootstrapping AND the trace; truncations keep the bootstrap value (from
  the transition's actual final observation) but cut the trace. Truncation
  bootstrap values arrive precomputed via :class:`TruncationBootstrapCache`.
  Masks are hoisted out of the backwards loop; use :func:`get_gae_fn` for the
  ``torch.compile``-cached variant (compile happens lazily on first call).
- :class:`TruncationBootstrapCache` -- accumulate raw final observations
  during the rollout; resolve them with ONE batched value forward after the
  loop instead of one small forward per step containing a truncation.
- :func:`device_minibatches` -- GPU-side ``randperm`` minibatch indices.
  Replaces ``np.arange`` + ``np.random.shuffle`` + host-side advanced
  indexing.
- :func:`explained_variance` -- on-device explained variance (biased ``var``,
  matching the NumPy reference). Returns a 0-d tensor: one sync only when the
  caller logs it, never inside the optimizer path.
- :func:`gather_metrics` -- single-sync device-to-host transfer for a dict of
  log-time scalar tensors. Replaces per-minibatch ``.item()`` calls (e.g. the
  classic ``clipfrac`` sync-per-minibatch): accumulate 0-d tensors, gather
  once at logging time.
"""

import numpy as np
import torch

_gae_fn_cache = {}


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    tail_value,
    gamma,
    gae_lambda,
):
    """Terminal-aware GAE over ``(T, N)`` device tensors. See module docstring."""
    truncated = truncations.bool()
    bootstrap_nonterminal = 1.0 - terminations
    trace_nonterminal = 1.0 - torch.maximum(terminations, truncations)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(tail_value)
    last_t = rewards.shape[0] - 1
    for t in range(last_t, -1, -1):
        next_value = tail_value if t == last_t else values[t + 1]
        next_value = torch.where(
            truncated[t], truncation_bootstrap_values[t], next_value
        )
        delta = rewards[t] + gamma * bootstrap_nonterminal[t] * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal[t] * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def compute_gae_from_next_values(
    rewards, values, terminations, truncations, next_values, gamma, gae_lambda
):
    """GAE using independently evaluated transition values, as in V-MPO v30.

    ``next_values[t]`` must describe the actual transition observation, including
    the final observation before autoreset. Keeping these evaluations explicit
    preserves the critic's batch layout (and its mixed-precision rounding).
    The multiplication order and ``where`` recurrence match the v30 reference.
    """
    boundaries = torch.logical_or(terminations.bool(), truncations.bool())
    bootstrap_nonterminal = 1.0 - terminations
    advantages = torch.empty_like(rewards)
    running_advantage = torch.zeros_like(next_values[-1])
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * bootstrap_nonterminal[t] - values[t]
        continuing = delta + gamma * gae_lambda * running_advantage
        running_advantage = torch.where(boundaries[t], delta, continuing)
        advantages[t] = running_advantage
    return advantages, advantages + values


def get_gae_fn(compiled=False, mode="reduce-overhead", *, explicit_next_values=False):
    """Return cached GAE; select explicit transition values for v30 semantics."""
    key = (bool(compiled), mode if compiled else None, bool(explicit_next_values))
    if key not in _gae_fn_cache:
        fn = compute_gae_from_next_values if explicit_next_values else compute_gae
        if compiled:
            fn = torch.compile(fn, mode=mode)
        _gae_fn_cache[key] = fn
    return _gae_fn_cache[key]


class TruncationBootstrapCache:
    """Snapshot final observations on the host, then evaluate/scatter in batches.

    Observation storage starts at one vector step and grows geometrically with
    actual truncations; a compact integer map deduplicates rollout slots.
    ``reset()`` reuses storage for subsequent rollouts. All mutable environment
    buffers are copied, without Python tuples or per-observation CUDA scatters.
    """

    def __init__(self, num_steps, num_envs, obs_shape=None):
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self._slot_to_entry = np.full(num_steps * num_envs, -1, dtype=np.int32)
        self._capacity = num_envs
        self._size = 0
        self._entry_slots = np.empty(self._capacity, dtype=np.int64)
        self._observations = None
        if obs_shape is not None:
            self._allocate(tuple(obs_shape))

    def _allocate(self, obs_shape):
        self._observations = np.empty(
            (self._capacity,) + obs_shape, dtype=np.float32
        )

    def _entries_for(self, slots):
        entries = self._slot_to_entry[slots]
        new = entries == -1
        needed = self._size + int(np.count_nonzero(new))
        if needed > self._capacity:
            capacity = min(max(needed, 2 * self._capacity), self._slot_to_entry.size)
            observations = np.empty(
                (capacity,) + self._observations.shape[1:], dtype=np.float32
            )
            observations[: self._size] = self._observations[: self._size]
            entry_slots = np.empty(capacity, dtype=np.int64)
            entry_slots[: self._size] = self._entry_slots[: self._size]
            self._observations = observations
            self._entry_slots = entry_slots
            self._capacity = capacity
        entries[new] = np.arange(self._size, needed, dtype=np.int32)
        self._entry_slots[entries[new]] = slots[new]
        self._slot_to_entry[slots[new]] = entries[new]
        self._size = needed
        return entries

    def reset(self):
        """Clear recorded transitions while retaining the observation storage."""
        self._slot_to_entry[self._entry_slots[: self._size]] = -1
        self._size = 0

    def _slots(self, step, truncations):
        if not 0 <= step < self.num_steps:
            raise IndexError(f"rollout step {step} is outside [0, {self.num_steps})")
        truncations = np.asarray(truncations, dtype=bool)
        if truncations.shape != (self.num_envs,):
            raise ValueError(f"truncations must have shape ({self.num_envs},)")
        indices = np.flatnonzero(truncations)
        return indices, int(step) * self.num_envs + indices

    def push_normalized(self, step, truncations, transition_observations):
        """Snapshot normalized final rows from ``VectorObsNorm.normalize_step``.

        Supply its second return value, which contains finals before autoreset.
        No object-array ``infos`` reconstruction is needed on this fast path.
        """
        indices, slots = self._slots(step, truncations)
        if indices.size == 0:
            return
        observations = np.asarray(transition_observations)
        if observations.shape[0] != self.num_envs:
            raise ValueError("transition observations must have one row per environment")
        if self._observations is None:
            self._allocate(observations.shape[1:])
        if observations.shape[1:] != self._observations.shape[1:]:
            raise ValueError("transition observation shape changed during the rollout")
        entries = self._entries_for(slots)
        self._observations[entries] = observations[indices]

    def push(self, step, truncations, infos):
        """Record raw final observations for steps where ``truncations`` holds."""
        indices, slots = self._slots(step, truncations)
        if indices.size == 0:
            return
        finals = infos.get("final_observation")
        masks = infos.get("_final_observation")
        if finals is None:
            raise RuntimeError("truncated transition missing final_observation")
        for i, slot in zip(indices, slots):
            i = int(i)
            if masks is not None and not masks[i]:
                raise RuntimeError(f"truncated environment {i} has no final observation")
            if finals[i] is None:
                raise RuntimeError(f"truncated environment {i} has no final observation")
            observation = np.asarray(finals[i])
            if self._observations is None:
                self._allocate(observation.shape)
            if observation.shape != self._observations.shape[1:]:
                raise ValueError("final observation shape changed during the rollout")
            entries = self._entries_for(np.asarray([slot]))
            self._observations[entries[0]] = observation

    def resolve(self, value_fn, device, *, batch_size=None):
        """Single batched forward; ``(T, N)`` float32 tensor on ``device``.

        ``value_fn`` maps a ``(B, *obs)`` float32 tensor to ``(B,)`` values
        (call under ``torch.no_grad``). Rows without truncation stay zero.
        Optional ``batch_size`` evaluates padded fixed-size chunks, useful for
        compiled pointwise critics. It is only appropriate when the critic has
        no batch-dependent behavior (e.g. training-mode batch normalization).
        Padding duplicates a valid observation and never enters the result.
        """
        out = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, device=device
        )
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        if self._size == 0:
            return out
        order = np.argsort(self._entry_slots[: self._size])
        slots = self._entry_slots[order]
        count = self._size
        padded_count = count if batch_size is None else ((count + batch_size - 1) // batch_size) * batch_size
        observations = np.empty(
            (padded_count,) + self._observations.shape[1:], dtype=np.float32
        )
        np.take(self._observations, order, axis=0, out=observations[:count])
        observations[count:] = observations[count - 1]
        batch = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
        index = torch.as_tensor(slots, dtype=torch.long, device=device)
        chunk_size = padded_count if batch_size is None else batch_size
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            flat = value_fn(batch[start : start + chunk_size]).flatten()
            if flat.numel() != chunk_size:
                raise ValueError("value_fn must return one value per observation")
            out.view(-1).index_copy_(0, index[start:stop], flat[: stop - start].to(out.dtype))
        return out

    def __len__(self):
        return self._size


def device_minibatches(batch_size, minibatch_size, device, generator=None):
    """Shuffled minibatch index rows covering ``range(batch_size)``, on device.

    The permutation is drawn with ``generator`` (default: nondeterministic,
    on the target device). When the generator lives on another device, the
    perm is drawn there and moved once (async, no sync) instead of failing.
    """
    target = torch.device(device)
    if generator is None:
        perm = torch.randperm(int(batch_size), device=target)
    else:
        gen_device = torch.device(getattr(generator, "device", "cpu"))
        perm = torch.randperm(int(batch_size), device=gen_device, generator=generator)
        if gen_device != target:
            perm = perm.to(target, non_blocking=True)
    return list(perm.split(int(minibatch_size)))


def gather_metrics(named):
    """Transfer ``{name: 0-d tensor}`` to host with exactly one sync.

    Entries may live on different devices (e.g. a CPU-computed scalar next
    to CUDA losses); everything is moved to the first CUDA device found
    (else the first entry's device) before the single stacked transfer.
    Per-tensor ``.to`` is a no-op when already in place.
    """
    keys = list(named)
    flat = [named[k].detach().reshape(()) for k in keys]
    target = next((v.device for v in flat if v.is_cuda), flat[0].device)
    stacked = torch.stack([v.to(target, dtype=torch.float64) for v in flat])
    return dict(zip(keys, stacked.cpu().tolist()))


def explained_variance(y_pred, y_true):
    """On-device explained variance; NaN 0-d tensor when target variance is 0."""
    var = y_true.var(unbiased=False)
    score = 1.0 - (y_true - y_pred).var(unbiased=False) / var
    return torch.where(
        var == 0,
        torch.tensor(float("nan"), device=y_true.device, dtype=score.dtype),
        score,
    )
