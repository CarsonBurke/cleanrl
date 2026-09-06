"""Per-call cost of the normalization + rollout-staging path, before vs after.

At 16 envs x 17 observations the whole path is a few hundred flops, so its cost
is per-call overhead: NumPy dispatch (~0.45us per ufunc, of which ``out=`` saves
~0.05us), temporary allocation, and Python-level branching, all paid 2048 times
per rollout iteration.

This script measures the four calls that path is made of -- reward normalize,
observation normalize_step on the no-boundary fast path and on a boundary-heavy
path, and rollout staging push -- against a frozen copy of the previous
implementation, in the same process, back to back, and asserts that the new
implementation is *bit-identical* to it over a randomized trace with autoreset
boundaries (these runs are compared against frozen baselines, so "close" is not
good enough).

    .venv/bin/python scripts/benchmark_vector_norm.py --num-envs 16
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm, ufunc_clip

_RMS_COUNT_INIT = 1e-4


# --------------------------------------------------------------------------- #
# Frozen reference copies of the pre-optimization implementations. These exist
# only so this script can prove bitwise equivalence and quote a before number;
# they are deliberately verbatim and must not be "fixed" or shared.
# --------------------------------------------------------------------------- #


class ReferenceObsNorm:
    """Frozen copy of the previous ``VectorObsNorm`` (equivalence reference)."""

    def __init__(self, num_envs, obs_shape, epsilon=1e-8, clip=10.0):
        self.epsilon = epsilon
        self.clip = clip
        self.means = np.zeros((num_envs,) + tuple(obs_shape), dtype=np.float64)
        self.variances = np.ones_like(self.means)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        obs = np.asarray(obs, dtype=np.float64)
        if rows is None:
            means, variances, counts = self.means, self.variances, self.counts
        else:
            means = self.means[rows]
            variances = self.variances[rows]
            counts = self.counts[rows]

        count_axes = (slice(None),) + (None,) * (obs.ndim - 1)
        old_counts = counts[count_axes]
        total_counts = counts + 1.0
        new_totals = total_counts[count_axes]
        delta = obs - means
        new_means = means + delta / new_totals
        new_variances = (
            variances * old_counts + np.square(delta) * old_counts / new_totals
        ) / new_totals

        if rows is None:
            self.means[...] = new_means
            self.variances[...] = new_variances
            self.counts[...] = total_counts
        else:
            self.means[rows] = new_means
            self.variances[rows] = new_variances
            self.counts[rows] = total_counts

        out = (obs - new_means) / np.sqrt(new_variances + self.epsilon)
        if self.clip is not None:
            out = ufunc_clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)

    def normalize_step(self, raw_next_obs, terminations, truncations, infos):
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        boundaries = np.flatnonzero(np.logical_or(terminations, truncations))
        if boundaries.size:
            raw_transition = np.array(raw_next_obs, copy=True)
            finals = infos.get("final_observation")
            masks = infos.get("_final_observation")
            if finals is None:
                raise RuntimeError("completed transition missing final_observation")
            for i in boundaries:
                if masks is not None and not masks[i]:
                    raise RuntimeError(f"completed environment {i} has no final observation")
                if finals[i] is None:
                    raise RuntimeError(f"completed environment {i} has no final observation")
                raw_transition[i] = finals[i]
        else:
            raw_transition = raw_next_obs

        transition_obs = self.normalize(raw_transition)
        if not boundaries.size:
            return transition_obs, transition_obs
        next_obs = np.array(transition_obs, copy=True)
        next_obs[boundaries] = self.normalize(
            np.asarray(raw_next_obs)[boundaries], rows=boundaries
        )
        return next_obs, transition_obs


class ReferenceRewardNorm:
    """Frozen copy of the previous ``VectorRewardNorm`` (equivalence reference)."""

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.means = np.zeros(num_envs, dtype=np.float64)
        self.variances = np.ones(num_envs, dtype=np.float64)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)

    def normalize(self, rewards, terminations, out_dtype=np.float32):
        raw = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminations, dtype=np.float64)
        self.returns = self.returns * self.gamma * (1.0 - terminated) + raw

        total_counts = self.counts + 1.0
        delta = self.returns - self.means
        self.means = self.means + delta / total_counts
        self.variances = (
            self.variances * self.counts + np.square(delta) * self.counts / total_counts
        ) / total_counts
        self.counts = total_counts

        out = raw / np.sqrt(self.variances + self.epsilon)
        if self.clip is not None:
            out = ufunc_clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)


def reference_push(transfer, step, rewards, terminations, truncations,
                   transition_observations=None, **fields):
    """Frozen copy of the previous ``RolloutTransfer.push`` body."""
    if not 0 <= step < transfer.num_steps:
        raise IndexError(f"rollout step {step} is outside [0, {transfer.num_steps})")
    if fields.keys() != transfer._host_extra.keys():
        raise ValueError(f"push expects exactly the declared fields {sorted(transfer._host_extra)}")
    if transfer._upload_pending:
        transfer._upload_event.synchronize()
        transfer._upload_pending = False
    packed = transfer._host_fields[:, step]
    np.copyto(packed[0], rewards, casting="unsafe")
    np.copyto(packed[1], terminations, casting="unsafe")
    np.copyto(packed[2], truncations, casting="unsafe")
    if transfer._host_transitions is not None:
        if transition_observations is None:
            raise ValueError("transition observations are required for this rollout")
        np.copyto(transfer._host_transitions[step], transition_observations, casting="unsafe")
    elif transition_observations is not None:
        raise ValueError("enable store_transition_observations to record transitions")
    for name, value in fields.items():
        np.copyto(transfer._host_extra[name][step], value, casting="unsafe")


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def timeit(fn, iters, warmup=200):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6, min(samples) * 1e6


def make_trace(rng, steps, num_envs, obs_dim, boundary_rate):
    """Randomized rollout trace with terminations, truncations and finals."""
    trace = []
    for _ in range(steps):
        draw = rng.random(num_envs)
        terminations = draw < boundary_rate
        truncations = np.logical_and(draw >= boundary_rate, draw < 2 * boundary_rate)
        boundaries = np.logical_or(terminations, truncations)
        finals = [None] * num_envs
        for i in np.flatnonzero(boundaries):
            finals[i] = rng.normal(size=obs_dim)
        trace.append({
            "obs": rng.normal(size=(num_envs, obs_dim)) * 3.0,
            "rewards": rng.normal(size=num_envs) * 2.0,
            "terminations": terminations,
            "truncations": truncations,
            "infos": {"final_observation": finals, "_final_observation": boundaries},
            "reset_row": int(rng.integers(num_envs)) if rng.random() < 0.02 else None,
            "reset_obs": rng.normal(size=obs_dim),
            "float64_out": bool(rng.random() < 0.05),
            # An out_dtype the kernel does not serve must fall back to the
            # reference NumPy path, which now updates statistics in place.
            "fallback": bool(rng.random() < 0.03),
            # Envs that hand back float32 observations/rewards exercise the
            # widening cast in the staging buffer.
            "float32_input": bool(rng.random() < 0.1),
        })
    return trace


def check_equivalence(trace, num_envs, obs_dim, gamma):
    """Bitwise-compare optimized vs frozen reference over the whole trace."""
    fast_obs = VectorObsNorm(num_envs, (obs_dim,))
    ref_obs = ReferenceObsNorm(num_envs, (obs_dim,))
    fast_rew = VectorRewardNorm(num_envs, gamma)
    ref_rew = ReferenceRewardNorm(num_envs, gamma)
    # Unclipped reward normalization is a live trainer configuration (clip=None).
    fast_open = VectorRewardNorm(num_envs, gamma, clip=None)
    ref_open = ReferenceRewardNorm(num_envs, gamma, clip=None)

    boundary_steps = 0
    checks = 0

    def same(label, step, left, right):
        nonlocal checks
        checks += 1
        if type(left) is not type(right) or np.asarray(left).dtype != np.asarray(right).dtype:
            raise AssertionError(f"{label} dtype/type mismatch at step {step}")
        if not np.array_equal(left, right):
            worst = float(np.max(np.abs(np.asarray(left, dtype=np.float64)
                                        - np.asarray(right, dtype=np.float64))))
            raise AssertionError(f"{label} differs at step {step} (max |delta| {worst:.3e})")

    for step, entry in enumerate(trace):
        dtype = np.float64 if entry["float64_out"] else np.float32
        if np.any(np.logical_or(entry["terminations"], entry["truncations"])):
            boundary_steps += 1
        if entry["float32_input"]:
            obs_input = entry["obs"].astype(np.float32)
            reward_input = entry["rewards"].astype(np.float32)
        else:
            obs_input, reward_input = entry["obs"], entry["rewards"]

        same("reward", step,
             fast_rew.normalize(reward_input, entry["terminations"], out_dtype=dtype),
             ref_rew.normalize(reward_input, entry["terminations"], out_dtype=dtype))
        same("reward(clip=None)", step,
             fast_open.normalize(reward_input, entry["terminations"]),
             ref_open.normalize(reward_input, entry["terminations"]))
        if entry["fallback"]:
            same("reward(numpy fallback)", step,
                 fast_rew.normalize(reward_input, entry["terminations"], out_dtype=np.float16),
                 ref_rew.normalize(reward_input, entry["terminations"], out_dtype=np.float16))
        for field in ("returns", "means", "variances", "counts"):
            same(f"reward.{field}", step, getattr(fast_rew, field), getattr(ref_rew, field))
            same(f"reward(clip=None).{field}", step,
                 getattr(fast_open, field), getattr(ref_open, field))

        if entry["fallback"]:
            same("obs(numpy fallback)", step,
                 fast_obs.normalize(obs_input, out_dtype=np.float16),
                 ref_obs.normalize(obs_input, out_dtype=np.float16))
        fast_next, fast_trans = fast_obs.normalize_step(
            obs_input, entry["terminations"], entry["truncations"], entry["infos"])
        ref_next, ref_trans = ref_obs.normalize_step(
            obs_input, entry["terminations"], entry["truncations"], entry["infos"])
        same("next_obs", step, fast_next, ref_next)
        same("transition_obs", step, fast_trans, ref_trans)

        row = entry["reset_row"]
        if row is not None:
            # Manual single-env reset (staggered warmup uses exactly this call).
            rows = slice(row, row + 1)
            same("reset_row", step,
                 fast_obs.normalize(entry["reset_obs"][None, ...], rows=rows),
                 ref_obs.normalize(entry["reset_obs"][None, ...], rows=rows))
        for field in ("means", "variances", "counts"):
            same(f"obs.{field}", step, getattr(fast_obs, field), getattr(ref_obs, field))

    return boundary_steps, checks


def check_push_equivalence(trace, num_envs, obs_dim, act_dim, device):
    """Bitwise-compare staged host bytes for optimized vs reference push."""
    steps = min(len(trace), 256)
    kwargs = dict(store_transition_observations=True,
                  fields={"observations": (obs_dim,), "native_actions": (act_dim,)})
    fast = RolloutTransfer(steps, num_envs, (obs_dim,), device, **kwargs)
    reference = RolloutTransfer(steps, num_envs, (obs_dim,), device, **kwargs)
    rng = np.random.default_rng(7)
    try:
        for step in range(steps):
            entry = trace[step]
            observations = entry["obs"].astype(np.float32)
            native = rng.normal(size=(num_envs, act_dim)).astype(np.float32)
            fast.push(step, entry["rewards"], entry["terminations"], entry["truncations"],
                      entry["obs"], observations=observations, native_actions=native)
            reference_push(reference, step, entry["rewards"], entry["terminations"],
                           entry["truncations"], entry["obs"],
                           observations=observations, native_actions=native)
        if not torch.equal(fast._host, reference._host):
            raise AssertionError("push staged different host bytes than the reference")
        return steps
    finally:
        fast.close()
        reference.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--act-dim", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--equivalence-steps", type=int, default=5000)
    parser.add_argument("--boundary-envs", type=int, default=4,
                        help="envs at a boundary in the boundary-heavy timing path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    n, dim = args.num_envs, args.obs_dim

    trace = make_trace(rng, args.equivalence_steps, n, dim, boundary_rate=0.01)
    boundary_steps, checks = check_equivalence(trace, n, dim, args.gamma)
    push_steps = check_push_equivalence(trace, n, dim, args.act_dim, device)
    print(f"equivalence: BITWISE IDENTICAL over {len(trace)} randomized steps "
          f"({boundary_steps} with terminations/truncations, {checks} exact array "
          f"comparisons incl. running statistics: means/variances/counts/returns)")
    print(f"push staging: BITWISE IDENTICAL host bytes over {push_steps} steps\n")

    # ---- timing inputs (one fixed step, reused so only the call is measured) --
    obs = rng.normal(size=(n, dim))
    rewards = rng.normal(size=n)
    quiet = np.zeros(n, dtype=bool)
    quiet_infos = {}
    hot_terms = np.zeros(n, dtype=bool)
    hot_truncs = np.zeros(n, dtype=bool)
    rows = rng.permutation(n)[: max(0, min(args.boundary_envs, n))]
    hot_terms[rows[: len(rows) // 2]] = True
    hot_truncs[rows[len(rows) // 2:]] = True
    boundaries = np.logical_or(hot_terms, hot_truncs)
    finals = [rng.normal(size=dim) if flag else None for flag in boundaries]
    hot_infos = {"final_observation": finals, "_final_observation": boundaries}

    observations = obs.astype(np.float32)
    native = rng.normal(size=(n, args.act_dim)).astype(np.float32)

    fast_obs = VectorObsNorm(n, (dim,))
    ref_obs = ReferenceObsNorm(n, (dim,))
    fast_hot = VectorObsNorm(n, (dim,))
    ref_hot = ReferenceObsNorm(n, (dim,))
    fast_rew = VectorRewardNorm(n, args.gamma)
    ref_rew = ReferenceRewardNorm(n, args.gamma)
    transfer = RolloutTransfer(args.num_steps, n, (dim,), device,
                               fields={"observations": (dim,), "native_actions": (args.act_dim,)})
    plain = RolloutTransfer(args.num_steps, n, (dim,), device)
    steps = args.num_steps
    counter = {"i": 0}

    def next_step():
        step = counter["i"] % steps
        counter["i"] += 1
        return step

    # Written out with literal keywords, exactly how the trainers call push.
    def reference_push_extra():
        reference_push(transfer, next_step(), rewards, quiet, quiet,
                       observations=observations, native_actions=native)

    def fast_push_extra():
        transfer.push(next_step(), rewards, quiet, quiet,
                      observations=observations, native_actions=native)

    def reference_push_plain():
        reference_push(plain, next_step(), rewards, quiet, quiet)

    def fast_push_plain():
        plain.push(next_step(), rewards, quiet, quiet)

    measurements = [
        ("reward_normalize",
         lambda: ref_rew.normalize(rewards, quiet),
         lambda: fast_rew.normalize(rewards, quiet)),
        ("normalize_step (no boundary)",
         lambda: ref_obs.normalize_step(obs, quiet, quiet, quiet_infos),
         lambda: fast_obs.normalize_step(obs, quiet, quiet, quiet_infos)),
        (f"normalize_step ({int(boundaries.sum())}/{n} boundaries)",
         lambda: ref_hot.normalize_step(obs, hot_terms, hot_truncs, hot_infos),
         lambda: fast_hot.normalize_step(obs, hot_terms, hot_truncs, hot_infos)),
        ("push (+2 extra fields)", reference_push_extra, fast_push_extra),
        ("push (no extra fields)", reference_push_plain, fast_push_plain),
    ]

    print(f"num_envs={n} obs_dim={dim} act_dim={args.act_dim} iters={args.iters} "
          f"(median of {args.iters}, back-to-back before/after)")
    print(f"  {'call':<30s} {'before_us':>10s} {'after_us':>9s} {'speedup':>8s} "
          f"{'before_min':>11s} {'after_min':>10s}")
    combined = [0.0, 0.0]
    for name, before_fn, after_fn in measurements:
        before = timeit(before_fn, args.iters)
        after = timeit(after_fn, args.iters)
        print(f"  {name:<30s} {before[0]:10.2f} {after[0]:9.2f} "
              f"{before[0] / after[0]:7.2f}x {before[1]:11.2f} {after[1]:10.2f}")
        if name in ("reward_normalize", "normalize_step (no boundary)"):
            combined[0] += before[0]
            combined[1] += after[0]
    print(f"  {'combined normalize path':<30s} {combined[0]:10.2f} {combined[1]:9.2f} "
          f"{combined[0] / combined[1]:7.2f}x")

    transfer.close()
    plain.close()


if __name__ == "__main__":
    main()
