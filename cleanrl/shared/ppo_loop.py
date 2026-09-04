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


def get_gae_fn(compiled=False, mode="reduce-overhead"):
    """Return the GAE callable, ``torch.compile``-cached when requested."""
    key = (bool(compiled), mode if compiled else None)
    if key not in _gae_fn_cache:
        fn = compute_gae
        if compiled:
            fn = torch.compile(compute_gae, mode=mode)
        _gae_fn_cache[key] = fn
    return _gae_fn_cache[key]


class TruncationBootstrapCache:
    """One batched value forward for all truncation bootstraps in a rollout."""

    def __init__(self, num_steps, num_envs):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self._entries = []  # (step, env_index, final_obs)

    def push(self, step, truncations, infos):
        """Record raw final observations for steps where ``truncations`` holds."""
        indices = np.flatnonzero(np.asarray(truncations, dtype=bool))
        if indices.size == 0:
            return
        finals = infos.get("final_observation")
        masks = infos.get("_final_observation")
        if finals is None:
            raise RuntimeError("truncated transition missing final_observation")
        for i in indices:
            i = int(i)
            if masks is not None and not masks[i]:
                raise RuntimeError(f"truncated environment {i} has no final observation")
            if finals[i] is None:
                raise RuntimeError(f"truncated environment {i} has no final observation")
            self._entries.append((int(step), i, np.asarray(finals[i])))

    def resolve(self, value_fn, device):
        """Single batched forward; ``(T, N)`` float32 tensor on ``device``.

        ``value_fn`` maps a ``(B, *obs)`` float32 tensor to ``(B,)`` values
        (call under ``torch.no_grad``). Rows without truncation stay zero.
        """
        out = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, device=device
        )
        if not self._entries:
            return out
        batch = torch.as_tensor(
            np.stack([e[2] for e in self._entries]),
            dtype=torch.float32,
            device=device,
        )
        flat = value_fn(batch).flatten()
        for (step, i, _), v in zip(self._entries, flat):
            out[step, i] = v
        return out

    def __len__(self):
        return len(self._entries)


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
