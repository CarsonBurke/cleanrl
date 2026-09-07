"""Attribute the ``env_s`` span of ``NativeMujocoVectorEnv`` per region.

Production accounting (16-env HalfCheetah, sphere run) says ``timing/env_s`` is
148us per vector step while the C physics floor is ~107us at four threads. This
script splits the remaining ~40us across every Python/NumPy bookkeeping region
of ``step_async``/``step_wait``, separates steps that cross an autoreset
boundary from steps that do not, and re-measures the physics floor as a
function of thread count on the current (contended) box.

Nothing here modifies the environment: the region attribution runs a *shadow*
reimplementation of ``step_wait`` whose outputs are checked bitwise against the
real one before any timing is reported.

Modes:
  regions   in-situ region timers + isolated per-op microbenchmarks
  boundary  cost of an autoreset boundary (per boundary and amortized)
  threads   physics floor vs num_threads
  parity    shadow-vs-real bitwise equality check only
"""

from __future__ import annotations

import argparse
import gc
from copy import deepcopy
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

from cleanrl.shared.mujoco_env import make_mujoco_vector_env  # noqa: E402
from cleanrl.shared.vector_norm import ufunc_clip  # noqa: E402

PC = time.perf_counter


def loadavg():
    with open("/proc/loadavg") as handle:
        return handle.read().split()[0:3]


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timeit(fn, iters, warmup=200):
    """Median/min/mean wall microseconds for a single call of ``fn``."""
    for _ in range(warmup):
        fn()
    samples = np.empty(iters)
    for i in range(iters):
        start = PC()
        fn()
        samples[i] = PC() - start
    return (float(np.median(samples)) * 1e6, float(samples.min()) * 1e6,
            float(samples.mean()) * 1e6)


def batch_timeit(fn, iters, inner=50, warmup=5):
    """Amortized microseconds per call, timing ``inner`` calls per sample.

    Removes the ~50-70ns perf_counter pair from very cheap regions.
    """
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    samples = np.empty(iters)
    for i in range(iters):
        start = PC()
        for _ in range(inner):
            fn()
        samples[i] = (PC() - start) / inner
    return float(np.median(samples)) * 1e6, float(samples.min()) * 1e6


def timer_overhead():
    n = 20000
    start = PC()
    for _ in range(n):
        PC()
    single = (PC() - start) / n
    return single * 1e6


def make_env(env_id, num_envs, threads, seed=1):
    envs = make_mujoco_vector_env(env_id, num_envs, backend="native", num_threads=threads)
    envs.reset(seed=seed)
    return envs


def action_stream(envs, count, seed=0):
    rng = np.random.default_rng(seed)
    shape = (count, envs.num_envs) + envs.single_action_space.shape
    return rng.uniform(-1.2, 1.2, size=shape).astype(np.float32)


# --------------------------------------------------------------------------
# Shadow step: an exact clone of step_async + step_wait for the production
# configuration (no legacy postprocessors, no batched normalization, copy=True)
# with a perf_counter around each bookkeeping region.
# --------------------------------------------------------------------------

REGION_NAMES = [
    "async_asarray_shapecheck",
    "async_clip",
    "copyto_controls",
    "native_pool_step",
    "velocity_forward",
    "ctrl_cost",
    "reward_term",
    "obs_assembly",
    "obs_publish",
    "episode_stats",
    "elapsed_steps_loop",
    "trunc_boundary",
    "infos_build",
    "flatnonzero",
    "boundary_rows",
    "return_copy",
]


class LegacyStep:
    """Verbatim clone of the pre-optimization ``step_async``/``step_wait``.

    Identical to :class:`Shadow` but without the region timers, so it is a
    fair timing baseline for the realized delta after the real backend is
    changed. Covers the production configuration only (no legacy
    postprocessors, no batched normalization, ``copy=True``).
    """

    def __init__(self, envs):
        self.e = envs

    def step(self, actions, record=False):
        e = self.e
        base = e._bases[0]
        actions = np.asarray(actions)
        if actions.shape != e.action_space.shape:
            raise ValueError("shape")
        clipped = e._clipped.get(actions.dtype)
        if clipped is None:
            clipped = np.empty(e.action_space.shape, dtype=actions.dtype)
            e._clipped[actions.dtype] = clipped
        actions = ufunc_clip(actions, e.single_action_space.low,
                             e.single_action_space.high, out=clipped)
        np.copyto(e._controls, actions)
        e._native.cleanrl_pool_step(e._pool)
        x = e._positions[:, 0]
        velocity = (x - e._before) / base.dt
        forward = base._forward_reward_weight * velocity
        costs = np.sum(np.square(actions), axis=1)
        cost_dtype = e._cost_dtypes.get(costs.dtype)
        if cost_dtype is None:
            cost_dtype = np.asarray(costs[0] * base._ctrl_cost_weight).dtype
            e._cost_dtypes[costs.dtype] = cost_dtype
        costs = costs.astype(cost_dtype, copy=False) * base._ctrl_cost_weight
        rewards = forward - costs
        terms = np.zeros(e.num_envs, dtype=bool)
        offset = int(base._exclude_current_positions_from_observation)
        width = base.model.nq - offset
        e._raw_observations[:, :width] = e._positions[:, offset:]
        e._raw_observations[:, width:] = e._velocities
        e.observations[...] = e._raw_observations
        e._episode_returns += rewards.astype(np.float32)
        e._episode_lengths += 1
        for limit, elapsed in zip(e._limits, e._episode_lengths.tolist()):
            limit._elapsed_steps = elapsed
        truncs = e._episode_lengths >= e._horizons
        boundaries = terms | truncs
        active = ~boundaries
        infos = {}
        columns = {"x_position": x, "x_velocity": velocity}
        columns.update(reward_run=forward, reward_ctrl=-costs)
        if active.any():
            for key, values in columns.items():
                infos[key] = np.where(active, values, 0.0)
                infos["_" + key] = active.copy()
        processed_observations = e._raw_observations
        for i in np.flatnonzero(boundaries):
            observation = processed_observations[i].copy()
            final_info = {key: values[i] for key, values in columns.items()}
            final_info["episode"] = {
                "r": e._episode_returns[i:i + 1].copy(),
                "l": e._episode_lengths[i:i + 1].copy(),
                "t": np.round(PC() - e._episode_starts[i:i + 1], 6),
            }
            stats = e._statistics[i]
            stats.return_queue.append(e._episode_returns[i:i + 1].copy())
            stats.length_queue.append(e._episode_lengths[i:i + 1].copy())
            stats.episode_count += np.int64(1)
            e._episode_returns[i] = 0
            e._episode_lengths[i] = 0
            e._episode_starts[i] = PC()
            e.observations[i] = observation
            reset_observation, reset_info = e.envs[i].reset()
            e.observations[i] = reset_observation
            reset_info["final_observation"] = observation
            reset_info["final_info"] = final_info
            e._add_info(infos, reset_info, i)
        return (e.observations.copy(), rewards, terms, truncs, infos)


class Shadow:
    def __init__(self, envs):
        self.e = envs
        self.acc = {name: 0.0 for name in REGION_NAMES}
        self.samples = []  # (n_boundaries, total_us, per-region us)
        # (final_info, stats, reset, add_info) microseconds per boundary row
        self.boundary_samples = []

    def step(self, actions, record=True):
        e = self.e
        base = e._bases[0]
        t = np.empty(len(REGION_NAMES) + 1)
        k = 0
        t[k] = PC(); k += 1

        # --- step_async ---
        actions = np.asarray(actions)
        if actions.shape != e.action_space.shape:
            raise ValueError("shape")
        t[k] = PC(); k += 1
        clipped = e._clipped.get(actions.dtype)
        if clipped is None:
            clipped = np.empty(e.action_space.shape, dtype=actions.dtype)
            e._clipped[actions.dtype] = clipped
        actions = ufunc_clip(actions, e.single_action_space.low,
                             e.single_action_space.high, out=clipped)
        t[k] = PC(); k += 1

        # --- step_wait ---
        np.copyto(e._controls, actions)
        t[k] = PC(); k += 1
        e._native.cleanrl_pool_step(e._pool)
        t[k] = PC(); k += 1

        x = e._positions[:, 0]
        velocity = (x - e._before) / base.dt
        forward = base._forward_reward_weight * velocity
        t[k] = PC(); k += 1

        costs = np.sum(np.square(actions), axis=1)
        cost_dtype = e._cost_dtypes.get(costs.dtype)
        if cost_dtype is None:
            cost_dtype = np.asarray(costs[0] * base._ctrl_cost_weight).dtype
            e._cost_dtypes[costs.dtype] = cost_dtype
        costs = costs.astype(cost_dtype, copy=False) * base._ctrl_cost_weight
        t[k] = PC(); k += 1

        rewards = forward - costs
        terms = np.zeros(e.num_envs, dtype=bool)
        t[k] = PC(); k += 1

        offset = int(base._exclude_current_positions_from_observation)
        width = base.model.nq - offset
        e._raw_observations[:, :width] = e._positions[:, offset:]
        e._raw_observations[:, width:] = e._velocities
        t[k] = PC(); k += 1

        e.observations[...] = e._raw_observations
        t[k] = PC(); k += 1

        e._episode_returns += rewards.astype(np.float32)
        e._episode_lengths += 1
        t[k] = PC(); k += 1

        for limit, elapsed in zip(e._limits, e._episode_lengths.tolist()):
            limit._elapsed_steps = elapsed
        t[k] = PC(); k += 1

        truncs = e._episode_lengths >= e._horizons
        boundaries = terms | truncs
        active = ~boundaries
        t[k] = PC(); k += 1

        infos = {}
        columns = {"x_position": x, "x_velocity": velocity,
                   "reward_run": forward, "reward_ctrl": -costs}
        if active.any():
            for key, values in columns.items():
                infos[key] = np.where(active, values, 0.0)
                infos["_" + key] = active.copy()
        t[k] = PC(); k += 1

        processed_observations = e._raw_observations
        processed_rows = np.flatnonzero(boundaries)
        t[k] = PC(); k += 1

        for i in processed_rows:
            b0 = PC()
            observation = processed_observations[i].copy()
            final_info = {key: values[i] for key, values in columns.items()}
            final_info["episode"] = {
                "r": e._episode_returns[i:i + 1].copy(),
                "l": e._episode_lengths[i:i + 1].copy(),
                "t": np.round(PC() - e._episode_starts[i:i + 1], 6),
            }
            b1 = PC()
            stats = e._statistics[i]
            stats.return_queue.append(e._episode_returns[i:i + 1].copy())
            stats.length_queue.append(e._episode_lengths[i:i + 1].copy())
            stats.episode_count += np.int64(1)
            e._episode_returns[i] = 0
            e._episode_lengths[i] = 0
            e._episode_starts[i] = PC()
            e.observations[i] = observation
            b2 = PC()
            reset_observation, reset_info = e.envs[i].reset()
            b3 = PC()
            e.observations[i] = reset_observation
            reset_info["final_observation"] = observation
            reset_info["final_info"] = final_info
            e._add_info(infos, reset_info, i)
            b4 = PC()
            self.boundary_samples.append(
                ((b1 - b0) * 1e6, (b2 - b1) * 1e6, (b3 - b2) * 1e6, (b4 - b3) * 1e6))
        t[k] = PC(); k += 1

        out = e.observations.copy()
        t[k] = PC(); k += 1

        deltas = np.diff(t[:k]) * 1e6
        if record:
            for name, value in zip(REGION_NAMES, deltas):
                self.acc[name] += value
            self.samples.append((len(processed_rows), float(t[k - 1] - t[0]) * 1e6,
                                 deltas.copy()))
        return out, rewards, terms, truncs, infos


class OptShadow:
    """Same outputs as ``Shadow``, with only *bitwise-safe* Python changes.

    Applied: every ``base.*`` lookup hoisted out of the step (``base.dt`` alone
    is a property costing 0.44us/step), ``out=`` buffers for the five
    intermediate allocations, one ``count_nonzero`` instead of
    ``active.any()`` + ``flatnonzero``, ``values.copy()`` instead of
    ``np.where(active, values, 0.0)`` on the (98.4%) steps with no boundary, a
    persistent ``columns`` dict, and precomputed ``"_" + key`` names.

    NOT applied (parity risk): reassociating ``(x - before) / dt * weight``,
    ``einsum`` for the control cost, dtype narrowing, dropping unused
    ``infos`` columns.
    """

    def __init__(self, envs, legacy_lookup=False, legacy_buffers=False,
                 legacy_infos=False):
        self.legacy_lookup = legacy_lookup
        self.legacy_buffers = legacy_buffers
        self.legacy_infos = legacy_infos
        e = self.e = envs
        base = e._bases[0]
        n = e.num_envs
        self.dt = base.dt
        self.fw = base._forward_reward_weight
        self.cw = base._ctrl_cost_weight
        self.offset = int(base._exclude_current_positions_from_observation)
        self.width = base.model.nq - self.offset
        self.action_shape = e.action_space.shape
        self.low = e.single_action_space.low
        self.high = e.single_action_space.high
        self.vbuf = np.empty(n, dtype=np.float64)
        self.fbuf = np.empty(n, dtype=np.float64)
        self.nbuf = np.empty(n, dtype=np.float64)
        self.rbuf = np.empty(n, dtype=np.float64)
        self.f32 = np.empty(n, dtype=np.float32)
        self.sq = {}
        self.keys = ("x_position", "x_velocity", "reward_run", "reward_ctrl")
        self.mask_keys = tuple("_" + key for key in self.keys)
        self.columns = dict.fromkeys(self.keys)
        self.samples = []

    def step(self, actions, record=True):
        e = self.e
        t0 = PC()
        actions = np.asarray(actions)
        if actions.shape != self.action_shape:
            raise ValueError("shape")
        clipped = e._clipped.get(actions.dtype)
        if clipped is None:
            clipped = np.empty(self.action_shape, dtype=actions.dtype)
            e._clipped[actions.dtype] = clipped
        actions = ufunc_clip(actions, self.low, self.high, out=clipped)
        np.copyto(e._controls, actions)
        e._native.cleanrl_pool_step(e._pool)

        if self.legacy_lookup:
            base = e._bases[0]
            dt, fw, cw = base.dt, base._forward_reward_weight, base._ctrl_cost_weight
            offset = int(base._exclude_current_positions_from_observation)
            width = base.model.nq - offset
        else:
            dt, fw, cw = self.dt, self.fw, self.cw
            offset, width = self.offset, self.width
        x = e._positions[:, 0]
        if self.legacy_buffers:
            velocity = (x - e._before) / dt
            forward = fw * velocity
        else:
            velocity = np.divide(np.subtract(x, e._before, out=self.vbuf), dt,
                                 out=self.vbuf)
            forward = np.multiply(fw, velocity, out=self.fbuf)

        if self.legacy_buffers:
            costs = np.sum(np.square(actions), axis=1)
        else:
            square = self.sq.get(actions.dtype)
            if square is None:
                square = np.empty(self.action_shape, dtype=actions.dtype)
                self.sq[actions.dtype] = square
            costs = np.sum(np.square(actions, out=square), axis=1)
        cost_dtype = e._cost_dtypes.get(costs.dtype)
        if cost_dtype is None:
            cost_dtype = np.asarray(costs[0] * cw).dtype
            e._cost_dtypes[costs.dtype] = cost_dtype
        costs = costs.astype(cost_dtype, copy=False) * cw

        # `rewards` is handed to the caller, so it must be a fresh array; an
        # `out=` buffer would alias across steps.
        rewards = forward - costs
        terms = np.zeros(e.num_envs, dtype=bool)

        raw = e._raw_observations
        raw[:, :width] = e._positions[:, offset:]
        raw[:, width:] = e._velocities
        e.observations[...] = raw

        if self.legacy_buffers:
            e._episode_returns += rewards.astype(np.float32)
        else:
            np.copyto(self.f32, rewards)
            e._episode_returns += self.f32
        e._episode_lengths += 1
        for limit, elapsed in zip(e._limits, e._episode_lengths.tolist()):
            limit._elapsed_steps = elapsed

        truncs = e._episode_lengths >= e._horizons
        boundaries = terms | truncs
        active = ~boundaries
        columns = self.columns
        columns["x_position"] = x
        columns["x_velocity"] = velocity
        columns["reward_run"] = forward
        # `-costs` keeps the control-cost dtype; an out= buffer would pin it.
        columns["reward_ctrl"] = -costs
        infos = {}
        if self.legacy_infos:
            if active.any():
                for key, values in columns.items():
                    infos[key] = np.where(active, values, 0.0)
                    infos["_" + key] = active.copy()
            crossings = len(np.flatnonzero(boundaries))
        else:
            crossings = np.count_nonzero(boundaries)
            if crossings == 0:
                for key, mask_key in zip(self.keys, self.mask_keys):
                    infos[key] = columns[key].copy()
                    infos[mask_key] = active.copy()
            elif crossings < e.num_envs:
                for key, mask_key in zip(self.keys, self.mask_keys):
                    infos[key] = np.where(active, columns[key], 0.0)
                    infos[mask_key] = active.copy()

        if crossings:
            for i in np.flatnonzero(boundaries):
                observation = raw[i].copy()
                final_info = {key: values[i] for key, values in columns.items()}
                final_info["episode"] = {
                    "r": e._episode_returns[i:i + 1].copy(),
                    "l": e._episode_lengths[i:i + 1].copy(),
                    "t": np.round(PC() - e._episode_starts[i:i + 1], 6),
                }
                stats = e._statistics[i]
                stats.return_queue.append(e._episode_returns[i:i + 1].copy())
                stats.length_queue.append(e._episode_lengths[i:i + 1].copy())
                stats.episode_count += np.int64(1)
                e._episode_returns[i] = 0
                e._episode_lengths[i] = 0
                e._episode_starts[i] = PC()
                e.observations[i] = observation
                reset_observation, reset_info = e.envs[i].reset()
                e.observations[i] = reset_observation
                reset_info["final_observation"] = observation
                reset_info["final_info"] = final_info
                e._add_info(infos, reset_info, i)

        out = e.observations.copy()
        if record:
            self.samples.append((crossings, (PC() - t0) * 1e6, None))
        return out, rewards, terms, truncs, infos


class CeilShadow:
    """Timing probe for the all-in-C ceiling. NOT semantically correct.

    Keeps only the work that cannot leave Python even if a C kernel produced
    every array: the ndarray/shape check, one FFI call, the 16
    ``TimeLimit._elapsed_steps`` object writes, allocation of the twelve
    result arrays a caller may retain, and the ``infos`` dict itself. All task
    arithmetic is assumed folded into ``cleanrl_pool_step``. Conservative: it
    still pays ``np.copyto(controls, actions)``, which C would absorb, and it
    excludes the amortized autoreset boundary (1.9us/step), which must be
    added back.
    """

    def __init__(self, envs):
        e = self.e = envs
        n = e.num_envs
        self.action_shape = e.action_space.shape
        self.obs_src = e._raw_observations
        self.f64 = np.empty(n, dtype=np.float64)
        self.b8 = np.empty(n, dtype=bool)
        self.keys = ("x_position", "x_velocity", "reward_run", "reward_ctrl")
        self.mask_keys = tuple("_" + key for key in self.keys)
        self.samples = []

    def step(self, actions, record=True):
        e = self.e
        actions = np.asarray(actions)
        if actions.shape != self.action_shape:
            raise ValueError("shape")
        np.copyto(e._controls, actions)
        e._native.cleanrl_pool_step(e._pool)
        e._episode_lengths += 1
        for limit, elapsed in zip(e._limits, e._episode_lengths.tolist()):
            limit._elapsed_steps = elapsed
        infos = {}
        for key, mask_key in zip(self.keys, self.mask_keys):
            infos[key] = self.f64.copy()
            infos[mask_key] = self.b8.copy()
        rewards = self.f64.copy()
        terms = self.b8.copy()
        truncs = self.b8.copy()
        out = self.obs_src.copy()
        return out, rewards, terms, truncs, infos


def tree_equal(a, b, path="root"):
    if isinstance(a, dict) != isinstance(b, dict):
        return [f"{path}: dict mismatch"]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        bad = []
        for key in a:
            bad += tree_equal(a[key], b[key], f"{path}.{key}")
        return bad
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}"]
        if a.dtype == object:
            bad = []
            for i, (u, v) in enumerate(zip(a.ravel(), b.ravel())):
                bad += tree_equal(u, v, f"{path}[{i}]")
            return bad
        return [] if np.array_equal(a, b) else [f"{path}: values differ"]
    if isinstance(a, (list, tuple)):
        bad = []
        for i, (u, v) in enumerate(zip(a, b)):
            bad += tree_equal(u, v, f"{path}[{i}]")
        return bad
    if a is None and b is None:
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


FREEZE_PATH = "/tmp/env_step_freeze.pkl"


def _drop_wall_clock(problems):
    """``episode["t"]`` is a wall-clock timestamp and can never be replayed."""
    return [p for p in problems
            if not p.startswith("infos.final_info") or "episode.t" not in p]


def freeze_reference(env_id, num_envs, threads, steps, path=FREEZE_PATH):
    """Record the real backend's full output tree for a fixed action stream.

    Deterministic inputs: env seeded with 7, actions from
    ``default_rng(3)``. Everything is deep-copied before storing so later
    in-place buffer reuse cannot corrupt the reference.
    """
    import pickle

    envs = make_env(env_id, num_envs, threads, seed=7)
    stream = action_stream(envs, steps, seed=3)
    frames = []
    boundary_steps = []
    for step in range(steps):
        out = envs.step(stream[step])
        frames.append(deepcopy(out))
        if "final_info" in out[4]:
            boundary_steps.append((step, int(np.count_nonzero(out[3]))))
    payload = {"env_id": env_id, "num_envs": num_envs, "steps": steps,
               "gym": gym.__version__, "numpy": np.__version__,
               "frames": frames, "boundary_steps": boundary_steps}
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    envs.close()
    print(f"froze {steps} steps of real-backend output to {path}")
    print(f"  boundary steps (step, #crossings): {boundary_steps}")
    return payload


def replay_reference(env_id, num_envs, threads, path=FREEZE_PATH, cls=None):
    """Diff the current implementation against a frozen reference."""
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload["env_id"] != env_id or payload["num_envs"] != num_envs:
        raise ValueError("frozen reference was recorded for a different setup")
    steps = payload["steps"]
    envs = make_env(env_id, num_envs, threads, seed=7)
    stream = action_stream(envs, steps, seed=3)
    stepper = envs if cls is None else cls(envs)
    problems = []
    for step in range(steps):
        out = stepper.step(stream[step])
        for name, u, v in zip(("obs", "rewards", "terms", "truncs", "infos"),
                              payload["frames"][step], out):
            problems += _drop_wall_clock(tree_equal(u, v, name))
        if problems:
            problems = [f"step {step}: " + p for p in problems]
            break
    envs.close()
    label = "real backend" if cls is None else cls.__name__
    print(f"replay vs frozen reference ({steps} steps, {label}): "
          f"mismatches={len(problems)}")
    print(f"  reference boundary steps: {payload['boundary_steps']}")
    for p in problems[:10]:
        print("   ", p)
    return problems


def _reference_module():
    """Import the pre-change ``mujoco_env`` straight out of git HEAD.

    The strongest available reference: the two classes coexist in one process,
    so old and new can be stepped side by side on identical action streams.
    """
    import importlib.util
    import subprocess

    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", "HEAD:cleanrl/shared/mujoco_env.py"],
        cwd=root, capture_output=True, text=True, check=True).stdout
    path = Path("/tmp/mujoco_env_head_reference.py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("mujoco_env_head_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # _native_library() locates mujoco_batch.c next to its own __file__, which
    # the /tmp copy cannot do. mujoco_batch.c is untouched by this change, so
    # both classes must load the exact same shared object anyway.
    from cleanrl.shared.mujoco_env import _native_library
    module._native_library = _native_library
    return module


def run_refparity(env_id, num_envs, threads, steps, stagger_stride=0):
    """Step the new class and the git-HEAD class side by side and diff."""
    from cleanrl.shared.vector_norm import make_raw_continuous_env

    reference = _reference_module()
    print(f"\n=== HEAD-reference parity: {env_id}, {num_envs} envs, "
          f"{steps} steps, stagger={stagger_stride} ===")
    fns = [make_raw_continuous_env(env_id, i, False, "") for i in range(num_envs)]
    new = make_mujoco_vector_env(env_id, num_envs, backend="native", num_threads=threads)
    old = reference.NativeMujocoVectorEnv(fns, env_id=env_id, num_threads=threads,
                                         copy=True, batch_legacy_normalization=True)
    new.reset(seed=7)
    old.reset(seed=7)
    if stagger_stride:
        stagger(new, stagger_stride)
        stagger(old, stagger_stride)
    stream = action_stream(new, steps, seed=3)
    problems = []
    histogram = {}
    for step in range(steps):
        a = stream[step]
        out_new = new.step(a)
        out_old = old.step(a)
        crossings = int(np.count_nonzero(out_new[2] | out_new[3]))
        histogram[crossings] = histogram.get(crossings, 0) + 1
        for name, u, v in zip(("obs", "rewards", "terms", "truncs", "infos"),
                              out_old, out_new):
            problems += _drop_wall_clock(tree_equal(u, v, name))
        if problems:
            problems = [f"step {step}: " + p for p in problems]
            break
    print(f"  crossings-per-step histogram: "
          f"{dict(sorted(histogram.items()))}")
    print(f"  mismatches={len(problems)}")
    for p in problems[:10]:
        print("   ", p)
    new.close()
    old.close()
    return problems


def stagger(envs, stride):
    """Give each env a distinct elapsed-step phase via per-env reset().

    Env ``k`` ends with ``(num_envs - k) * stride`` elapsed steps, so exactly
    one env crosses its TimeLimit every ``stride`` vector steps -- the same
    pattern ``--staggered-starts`` produces in the training script.
    """
    actions = np.zeros((envs.num_envs,) + envs.single_action_space.shape, dtype=np.float32)
    for index in range(envs.num_envs):
        envs.reset_at(index)
        for _ in range(stride):
            envs.step(actions)


def run_parity(env_id, num_envs, threads, steps, verbose=True, cls=Shadow):
    real = make_env(env_id, num_envs, threads, seed=7)
    shadow_env = make_env(env_id, num_envs, threads, seed=7)
    shadow = cls(shadow_env)
    stream = action_stream(real, steps, seed=3)
    # cross a full 1000-step boundary plus a staggered single-env boundary
    problems = []
    for step in range(steps):
        a = stream[step]
        out_real = real.step(a)
        out_shadow = shadow.step(a, record=False)
        for name, u, v in zip(("obs", "reward", "term", "trunc", "infos"),
                              out_real, out_shadow):
            # episode["t"] is a wall-clock timestamp relative to each env's own
            # construction time, so it can never match across two instances.
            problems += [p for p in tree_equal(u, v, name)
                         if not p.startswith("infos.final_info") or "episode.t" not in p]
        if problems:
            problems = [f"step {step}: " + p for p in problems]
            break
    equal_states = np.array_equal(real._positions, shadow_env._positions)
    real.close()
    shadow_env.close()
    if verbose:
        print(f"parity: {steps} steps, mismatches={len(problems)} states_equal={equal_states}")
        for p in problems[:10]:
            print("   ", p)
    return problems


def run_opt(env_id, num_envs, threads, steps, repeats=5, only=None):
    """Head-to-head: real step_wait vs the bitwise-safe Python rewrite."""
    print(f"\n=== optimized-python head-to-head ({env_id}, {num_envs} envs, "
          f"{threads} threads) ===")
    print(f"loadavg={loadavg()}")
    problems = run_parity(env_id, num_envs, threads, 1100, verbose=False, cls=OptShadow)
    print(f"parity vs real backend over 1100 steps (crosses a 16-env boundary): "
          f"mismatches={len(problems)}")
    for p in problems[:6]:
        print("   ", p)

    envs = make_env(env_id, num_envs, threads, seed=31)
    stream = action_stream(envs, 512, seed=37)
    plain = Shadow(envs)
    variants = {
        "opt (all)": OptShadow(envs),
        "opt -hoisting": OptShadow(envs, legacy_lookup=True),
        "opt -out= buffers": OptShadow(envs, legacy_buffers=True),
        "opt -infos fastpath": OptShadow(envs, legacy_infos=True),
        "ceiling (C does math)": CeilShadow(envs),
    }
    idx = [0]

    def real():
        envs.step(stream[idx[0] % 512]); idx[0] += 1

    def make(shadow):
        def run():
            shadow.step(stream[idx[0] % 512], record=False); idx[0] += 1
        return run

    cases = [("real envs.step", real), ("shadow (clone)", make(plain)),
             ("legacy (untimed clone)", make(LegacyStep(envs)))]
    cases += [(name, make(shadow)) for name, shadow in variants.items()]
    if only:
        keep = set(only) | {"opt (all)"}
        cases = [(name, fn) for name, fn in cases if name in keep]
    # Round-robin so that machine-load drift hits every variant identically;
    # sequential blocks put the whole effect inside the drift envelope.
    rounds = max(1, steps)
    samples = {name: np.empty(rounds) for name, _ in cases}
    for _, fn in cases:
        for _ in range(300):
            fn()
    # Rotate the order every round: the slot a variant occupies inside a round
    # is worth ~2us (cache/pool-wake state), which would otherwise be charged
    # to whichever variant sits in that slot.
    width = len(cases)
    for r in range(rounds):
        for slot in range(width):
            name, fn = cases[(r + slot) % width]
            start = PC()
            fn()
            samples[name][r] = PC() - start
    print(f"{'variant':22s} {'median':>9s} {'min':>9s} {'p25':>9s} "
          f"{'vs opt(all) med':>16s}")
    summary = {}
    for name, _ in cases:
        column = samples[name] * 1e6
        summary[name] = float(np.median(column))
        print(f"{name:22s} {np.median(column):9.2f} {column.min():9.2f} "
              f"{np.percentile(column, 25):9.2f} "
              f"{np.median(column) - np.median(samples['opt (all)'] * 1e6):16.2f}")
    # paired per-round differences: immune to slow load drift
    base_col = samples["opt (all)"]
    print("  paired median difference vs opt (all):")
    for name, _ in cases:
        if name == "opt (all)":
            continue
        diff = (samples[name] - base_col) * 1e6
        boot = np.array([np.median(np.random.choice(diff, diff.size))
                         for _ in range(200)])
        print(f"    {name:22s} {np.median(diff):7.2f}us/step "
              f"[95% CI {np.percentile(boot, 2.5):6.2f}, {np.percentile(boot, 97.5):6.2f}]"
              f"  -> {np.median(diff)*2048/1000:6.1f}ms/iteration")
    envs.close()


def run_regions(env_id, num_envs, threads, steps):
    print(f"\n=== regions ({env_id}, {num_envs} envs, {threads} threads) ===")
    print(f"loadavg={loadavg()}  perf_counter={timer_overhead()*1000:.1f}ns")

    # 1. real env_s baseline: exactly what the training loop times.
    envs = make_env(env_id, num_envs, threads, seed=11)
    stream = action_stream(envs, steps + 400, seed=5)
    i = [0]

    def real_step():
        envs.step(stream[i[0] % len(stream)])
        i[0] += 1

    med, mn, mean = timeit(real_step, steps, warmup=400)
    print(f"real envs.step()            median {med:8.2f}us  min {mn:8.2f}  mean {mean:8.2f}")
    baseline = med

    # 2. shadow with region timers
    shadow = Shadow(envs)
    for step in range(300):
        shadow.step(stream[step % len(stream)], record=False)
    shadow.acc = {name: 0.0 for name in REGION_NAMES}
    shadow.samples = []
    for step in range(steps):
        shadow.step(stream[step % len(stream)])
    clean = [s for s in shadow.samples if s[0] == 0]
    arr = np.stack([s[2] for s in clean])
    totals = np.array([s[1] for s in clean])
    print(f"shadow total (0-boundary)   median {np.median(totals):8.2f}us "
          f"min {totals.min():8.2f}  n={len(clean)}")
    print(f"{'region':28s} {'median':>9s} {'min':>9s} {'mean':>9s}  {'%of total':>9s}")
    med_sum = 0.0
    region_medians = {}
    for j, name in enumerate(REGION_NAMES):
        column = arr[:, j]
        m = float(np.median(column))
        region_medians[name] = m
        med_sum += m
        print(f"{name:28s} {m:9.3f} {column.min():9.3f} {column.mean():9.3f} "
              f"{100*m/np.median(totals):9.1f}")
    print(f"{'SUM of medians':28s} {med_sum:9.3f}")
    print(f"{'timer overhead (16 pairs)':28s} {16*timer_overhead():9.3f}")
    envs.close()
    return baseline, region_medians, float(np.median(totals))


def run_isolated(env_id, num_envs, threads, iters):
    """Isolated microbenchmarks of each bookkeeping region, no timer overhead."""
    print(f"\n=== isolated region microbenchmarks ({num_envs} envs) ===")
    print(f"loadavg={loadavg()}")
    envs = make_env(env_id, num_envs, threads, seed=13)
    stream = action_stream(envs, 64, seed=9)
    for step in range(64):
        envs.step(stream[step])
    e = envs
    base = e._bases[0]
    n = e.num_envs

    positions = e._positions.copy()
    velocities = e._velocities.copy()
    before = e._before.copy()
    raw = np.empty_like(e._raw_observations)
    observations = np.empty_like(e.observations)
    controls = np.empty_like(e._controls)
    actions_f32 = stream[0]
    clipped = np.empty_like(actions_f32)
    returns = e._episode_returns.copy()
    lengths = np.zeros(n, dtype=np.int32)
    horizons = e._horizons
    limits = e._limits
    low, high = e.single_action_space.low, e.single_action_space.high
    space_shape = e.action_space.shape
    dt = base.dt
    fw = base._forward_reward_weight
    cw = base._ctrl_cost_weight
    offset = int(base._exclude_current_positions_from_observation)
    width = base.model.nq - offset

    x = positions[:, 0]
    velocity = (x - before) / dt
    forward = fw * velocity
    costs = np.sum(np.square(actions_f32), axis=1).astype(np.float64) * cw
    rewards = forward - costs
    terms = np.zeros(n, dtype=bool)
    truncs = np.zeros(n, dtype=bool)
    boundaries = terms | truncs
    active = ~boundaries
    columns = {"x_position": x, "x_velocity": velocity,
               "reward_run": forward, "reward_ctrl": -costs}
    cost_dtypes = {}

    def f_asarray():
        a = np.asarray(actions_f32)
        if a.shape != space_shape:
            raise ValueError

    def f_clip():
        ufunc_clip(actions_f32, low, high, out=clipped)

    def f_copyto():
        np.copyto(controls, clipped)

    def f_velocity():
        xx = positions[:, 0]
        v = (xx - before) / dt
        return fw * v

    def f_cost():
        c = np.sum(np.square(clipped), axis=1)
        d = cost_dtypes.get(c.dtype)
        if d is None:
            d = np.asarray(c[0] * cw).dtype
            cost_dtypes[c.dtype] = d
        return c.astype(d, copy=False) * cw

    def f_reward():
        return forward - costs, np.zeros(n, dtype=bool)

    def f_obs_assembly():
        raw[:, :width] = positions[:, offset:]
        raw[:, width:] = velocities

    def f_obs_publish():
        observations[...] = raw

    def f_episode_stats():
        returns.__iadd__(rewards.astype(np.float32))
        lengths.__iadd__(1)

    def f_elapsed_loop():
        for limit, elapsed in zip(limits, lengths.tolist()):
            limit._elapsed_steps = elapsed

    def f_elapsed_tolist():
        lengths.tolist()

    def f_trunc():
        tr = lengths >= horizons
        bd = terms | tr
        return ~bd

    def f_infos():
        infos = {}
        if active.any():
            for key, values in columns.items():
                infos[key] = np.where(active, values, 0.0)
                infos["_" + key] = active.copy()
        return infos

    def f_columns_dict():
        return {"x_position": x, "x_velocity": velocity,
                "reward_run": forward, "reward_ctrl": -costs}

    def f_flatnonzero():
        return np.flatnonzero(boundaries)

    def f_return_copy():
        return observations.copy()

    def f_rewards_astype():
        return rewards.astype(np.float32)

    def f_activeany():
        return active.any()

    def f_where():
        return np.where(active, x, 0.0)

    def f_activecopy():
        return active.copy()

    cases = [
        ("async: asarray+shapecheck", f_asarray),
        ("async: ufunc_clip", f_clip),
        ("copyto controls", f_copyto),
        ("velocity+forward", f_velocity),
        ("ctrl cost", f_cost),
        ("reward+terms", f_reward),
        ("obs assembly (2 slices)", f_obs_assembly),
        ("obs publish (observations[...])", f_obs_publish),
        ("episode stats (+=)", f_episode_stats),
        ("  rewards.astype(f32) alone", f_rewards_astype),
        ("elapsed_steps loop (16)", f_elapsed_loop),
        ("  lengths.tolist() alone", f_elapsed_tolist),
        ("truncs+boundaries+active", f_trunc),
        ("columns dict build", f_columns_dict),
        ("infos build (8 arrays)", f_infos),
        ("  active.any() alone", f_activeany),
        ("  one np.where alone", f_where),
        ("  one active.copy() alone", f_activecopy),
        ("flatnonzero", f_flatnonzero),
        ("observations.copy() return", f_return_copy),
    ]
    print(f"{'op':36s} {'us/step':>9s} {'min':>9s}")
    for name, fn in cases:
        m, mn = batch_timeit(fn, iters, inner=100)
        print(f"{name:36s} {m:9.3f} {mn:9.3f}")

    # pure native call cost in isolation, with realistic controls
    np.copyto(e._controls, clipped)

    def f_native():
        e._native.cleanrl_pool_step(e._pool)

    m, mn, mean = timeit(f_native, 3000, warmup=300)
    print(f"{'isolated cleanrl_pool_step':36s} {m:9.3f} {mn:9.3f} (mean {mean:.2f})")
    envs.close()


def run_boundary(env_id, num_envs, threads, steps):
    print(f"\n=== autoreset boundary cost ({env_id}, {num_envs} envs, {threads} threads) ===")
    print(f"loadavg={loadavg()}")
    horizon = 1000
    stride = horizon // num_envs
    envs = make_env(env_id, num_envs, threads, seed=17)
    stagger(envs, stride)
    shadow = Shadow(envs)
    stream = action_stream(envs, 512, seed=21)
    for step in range(200):
        shadow.step(stream[step % 512], record=False)
    shadow.samples = []
    for step in range(steps):
        shadow.step(stream[step % 512])
    groups = {}
    for count, total, deltas in shadow.samples:
        groups.setdefault(count, []).append((total, deltas))
    print(f"{'#boundaries':>12s} {'n':>5s} {'total med':>10s} {'boundary_rows med':>18s}")
    zero_total = None
    for count in sorted(groups):
        rows = groups[count]
        totals = np.array([r[0] for r in rows])
        br = np.array([r[1][REGION_NAMES.index("boundary_rows")] for r in rows])
        if count == 0:
            zero_total = float(np.median(totals))
        print(f"{count:12d} {len(rows):5d} {np.median(totals):10.2f} {np.median(br):18.2f}")
    per_boundary = {}
    for count in sorted(groups):
        if count == 0:
            continue
        rows = groups[count]
        totals = np.array([r[0] for r in rows])
        per_boundary[count] = (float(np.median(totals)) - zero_total) / count
        print(f"  extra per boundary at {count} boundary/step: "
              f"{per_boundary[count]:.2f}us")
    if 1 in per_boundary:
        cost = per_boundary[1]
        amortized = cost * num_envs / horizon
        print(f"  amortized at horizon={horizon}, {num_envs} envs: "
              f"{amortized:.3f}us/step ({amortized*2048/1000:.3f}ms/iteration)")
    if shadow.boundary_samples:
        arr = np.array(shadow.boundary_samples)
        labels = ("final_info+episode", "stats+counters", "envs[i].reset()", "_add_info")
        print(f"  in-situ boundary breakdown (n={len(arr)}):")
        for j, label in enumerate(labels):
            print(f"    {label:24s} med {np.median(arr[:, j]):8.2f}us "
                  f"min {arr[:, j].min():8.2f}  mean {arr[:, j].mean():8.2f}")
        print(f"    {'SUM of medians':24s} med "
              f"{sum(float(np.median(arr[:, j])) for j in range(4)):8.2f}us")

    # break the boundary block down
    e = envs
    columns_src = {"x_position": e._positions[:, 0].copy(),
                   "x_velocity": e._before.copy(),
                   "reward_run": e._before.copy(),
                   "reward_ctrl": e._before.copy()}
    obs_row = e._raw_observations[0]
    stats = e._statistics[0]
    ret, ln, st = e._episode_returns, e._episode_lengths, e._episode_starts

    def f_final_info():
        info = {key: values[0] for key, values in columns_src.items()}
        info["episode"] = {"r": ret[0:1].copy(), "l": ln[0:1].copy(),
                           "t": np.round(PC() - st[0:1], 6)}
        return info

    def f_queues():
        stats.return_queue.append(ret[0:1].copy())
        stats.length_queue.append(ln[0:1].copy())
        stats.episode_count += np.int64(1)

    def f_obs_copy():
        return obs_row.copy()

    def f_reset():
        e.envs[0].reset()

    def f_add_info():
        info = {"final_observation": obs_row, "final_info": {"episode": {}}}
        e._add_info({}, info, 0)

    print(f"{'boundary sub-op':36s} {'us':>9s} {'min':>9s}")
    for name, fn, iters in (("final_info dict", f_final_info, 300),
                            ("stats queues append", f_queues, 300),
                            ("obs row .copy()", f_obs_copy, 300),
                            ("_add_info", f_add_info, 300),
                            ("envs[i].reset()", f_reset, 200)):
        m, mn = batch_timeit(fn, iters, inner=20)
        print(f"{name:36s} {m:9.3f} {mn:9.3f}")
    envs.close()


def run_threads(env_id, num_envs, thread_list, iters):
    print(f"\n=== physics floor vs threads ({env_id}, {num_envs} envs) ===")
    print(f"loadavg={loadavg()}")
    print(f"{'threads':>8s} {'step med':>10s} {'step min':>10s} {'native med':>11s} "
          f"{'native min':>11s} {'cpu us/step':>12s} {'load':>7s}")
    results = {}
    for threads in thread_list:
        envs = make_env(env_id, num_envs, threads, seed=23)
        stream = action_stream(envs, 256, seed=29)
        idx = [0]

        def full():
            envs.step(stream[idx[0] % 256])
            idx[0] += 1

        for _ in range(300):
            full()
        cpu0 = cpu_seconds()
        wall0 = PC()
        med, mn, _ = timeit(full, iters, warmup=0)
        wall = PC() - wall0
        cpu = cpu_seconds() - cpu0

        np.copyto(envs._controls, stream[0])

        def native():
            envs._native.cleanrl_pool_step(envs._pool)

        nmed, nmn, _ = timeit(native, iters, warmup=200)
        results[threads] = (med, mn, nmed, nmn, cpu / iters * 1e6)
        print(f"{threads:8d} {med:10.2f} {mn:10.2f} {nmed:11.2f} {nmn:11.2f} "
              f"{cpu/iters*1e6:12.1f} {loadavg()[0]:>7s}")
        envs.close()
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--thread-list", default="1,2,4,6,8")
    parser.add_argument("--only", default="",
                        help="';'-separated opt-mode variants to keep")
    parser.add_argument("--stagger", type=int, default=0,
                        help="refparity: per-env reset stride for partial crossings")
    parser.add_argument("--mode", default="regions",
                        choices=("regions", "boundary", "threads", "parity",
                                 "isolated", "opt", "freeze", "replay", "refparity"))
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if args.mode == "parity":
        problems = run_parity(args.env_id, args.num_envs, args.threads, args.steps)
        sys.exit(1 if problems else 0)
    if args.mode == "freeze":
        print(f"loadavg={loadavg()}")
        freeze_reference(args.env_id, args.num_envs, args.threads, args.steps)
        sys.exit(0)
    if args.mode == "replay":
        print(f"loadavg={loadavg()}")
        problems = replay_reference(args.env_id, args.num_envs, args.threads)
        sys.exit(1 if problems else 0)
    if args.mode == "refparity":
        print(f"loadavg={loadavg()}")
        problems = run_refparity(args.env_id, args.num_envs, args.threads,
                                 args.steps, args.stagger)
        sys.exit(1 if problems else 0)
    if args.mode == "regions":
        run_regions(args.env_id, args.num_envs, args.threads, args.steps)
    elif args.mode == "isolated":
        run_isolated(args.env_id, args.num_envs, args.threads, args.iters)
    elif args.mode == "boundary":
        run_boundary(args.env_id, args.num_envs, args.threads, args.steps)
    elif args.mode == "threads":
        run_threads(args.env_id, args.num_envs,
                    [int(t) for t in args.thread_list.split(",")], args.iters)
    elif args.mode == "opt":
        run_opt(args.env_id, args.num_envs, args.threads, args.steps,
                only=[v for v in args.only.split(";") if v] if args.only else None)


if __name__ == "__main__":
    main()
