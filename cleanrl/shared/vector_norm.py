"""Vectorized (parallel) observation and reward normalization.

STANDARD: all new PPO-family continuous-control versions must use this module
instead of per-environment ``gym.wrappers.NormalizeObservation`` /
``NormalizeReward``. Do not retrofit old versioned files; they are frozen
benchmark references.

Why this exists
--------------
The CleanRL convention wraps each sub-environment independently *before*
vectorization, so every step runs N copies of the wrapper stack (Python call
overhead + one N=1 ``RunningMeanStd`` update each, in float64). This module
keeps the exact same semantics -- independent running statistics per
environment -- but performs one batched NumPy update for all envs on the
runner side, with raw (unwrapped) sub-environments. Typical usage::

    from cleanrl.shared.vector_norm import (
        VectorObsNorm,
        VectorRewardNorm,
        make_raw_continuous_env,
    )

    envs = gym.vector.SyncVectorEnv(
        [make_raw_continuous_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    obs_norm = VectorObsNorm(args.num_envs, envs.single_observation_space.shape)
    rew_norm = VectorRewardNorm(args.num_envs, gamma=args.gamma)

    raw_next_obs, _ = envs.reset(seed=args.seed)
    next_obs = obs_norm.normalize(raw_next_obs)
    ...
    raw_next_obs, raw_rew, terms, truncs, infos = envs.step(action)
    rew = rew_norm.normalize(raw_rew, terms)
    next_obs, transition_obs = obs_norm.normalize_step(
        raw_next_obs, terms, truncs, infos
    )

Equivalence contract (do not "improve" these without renaming):
- One independent ``RunningMeanStd`` row per env (count init 1e-4, eps 1e-8),
  updated with the singleton (batch-of-1, zero batch-var) Welford step, exactly
  as ``NormalizeObservation`` on a single env would.
- Reward path tracks discounted returns keyed on ``terminated`` only
  (``ret = ret * gamma * (1 - terminated) + rew``) and scales
  ``rew / sqrt(ret_var + eps)``, exactly as ``NormalizeReward``.
- Both outputs are clipped to ``[-clip, clip]`` (default 10), matching the
  ``TransformObservation`` / ``TransformReward`` wrappers in legacy ``make_env``.
- On autoreset boundaries, final (transition) observations are updated and
  normalized *before* the post-reset observations of the same step, matching
  per-env wrapper order. ``normalize_step`` returns ``(next_obs, trans_obs)``:
  ``next_obs`` (post-reset rows at boundaries) is what enters the rollout
  buffer; ``trans_obs`` (final-obs rows at boundaries) is what truncation
  bootstrap values must be computed from.

Why there is a C kernel
-----------------------
At the sizes this runs at (16 envs x 17 observations) the NumPy formulation is
*entirely* per-ufunc dispatch: ~30 ufunc calls costing ~0.45us each to perform a
few hundred flops, and ``out=`` saves only ~0.05us of that per call. The only
way to remove the overhead without changing the arithmetic is to issue the whole
update as one call, so the common case (whole batch, no autoreset boundary) runs
in a native kernel that performs *exactly* the reference operation sequence:
same order, same intermediates, double precision throughout, IEEE divide/sqrt,
``-ffp-contract=off`` so nothing is fused into an FMA. Output is bit-identical,
not merely close -- ``scripts/benchmark_vector_norm.py`` asserts that against a
frozen copy of the previous NumPy implementation over a randomized trace.

The NumPy path below is still the reference and still runs verbatim for row
subsets (the autoreset second pass, manual resets), unusual dtypes and
broadcasting inputs. Statistics arrays (``means`` / ``variances`` / ``counts``
/ ``returns``) are allocated once and mutated in place by both paths, since the
kernel holds their addresses; never rebind them.
"""

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import gymnasium as gym
import numpy as np

try:  # ``np.clip`` spends ~3us per call in Python-side deprecation checks; the ufunc is identical.
    from numpy._core.umath import clip as ufunc_clip
except ImportError:  # NumPy < 2
    from numpy.core.umath import clip as ufunc_clip

_RMS_COUNT_INIT = 1e-4

# Bit-identical native mirror of the NumPy reference below. Every expression is
# written in the reference's evaluation order; the compiler is not allowed to
# reassociate or contract it (see _KERNEL_FLAGS).
_KERNEL_SOURCE = r"""
/* Generated from cleanrl/shared/vector_norm.py -- do not edit in the cache. */

typedef struct {
    double *means;
    double *variances;
    double *counts;
    double *returns;
    double epsilon;
    double low;
    double high;
    double gamma;
    long num_envs;
    long dim;
} cleanrl_norm_state;

/* Clipping is written as two independent compares rather than a nested ternary
   or fmin/fmax: GCC if-converts and vectorizes this form, and refuses the other
   two ("unsupported control flow"), which costs 3x. It is min(max(x, low),
   high) -- numpy's clip -- including NaN passthrough, and low/high are +-inf
   when clipping is disabled, which leaves every value (including infinities)
   untouched. */
#define CLEANRL_CLIP(value, low, high)                                         \
    if (value < low) value = low;                                             \
    if (value > high) value = high

#define CLEANRL_OBS_KERNEL(NAME, OUT_T)                                        \
void NAME(const cleanrl_norm_state *restrict state,                            \
          const double *restrict obs, OUT_T *restrict out) {                   \
    const long num_envs = state->num_envs;                                     \
    const long dim = state->dim;                                               \
    const double epsilon = state->epsilon;                                     \
    const double low = state->low;                                             \
    const double high = state->high;                                           \
    double *restrict means = state->means;                                     \
    double *restrict variances = state->variances;                             \
    double *restrict counts = state->counts;                                   \
    for (long i = 0; i < num_envs; ++i) {                                      \
        const double count = counts[i];                                        \
        const double total = count + 1.0;                                      \
        const double *restrict row = obs + i * dim;                            \
        double *restrict mean = means + i * dim;                               \
        double *restrict variance = variances + i * dim;                       \
        OUT_T *restrict destination = out + i * dim;                           \
        for (long j = 0; j < dim; ++j) {                                       \
            const double delta = row[j] - mean[j];                             \
            const double updated_mean = mean[j] + delta / total;               \
            const double updated_variance =                                    \
                (variance[j] * count + delta * delta * count / total) / total; \
            mean[j] = updated_mean;                                            \
            variance[j] = updated_variance;                                    \
            /* Subtract then divide, one division per element, and likewise a  \
               real division by `total` above: hoisting 1/total or a per-row   \
               reciprocal scale and multiplying is NOT bit-identical (it       \
               differs on ~26% of random elements), so it is not allowed here. \
               These are the only "slow" ops in the loop and they vectorize.  */\
            double value = (row[j] - updated_mean)                             \
                / __builtin_sqrt(updated_variance + epsilon);                  \
            CLEANRL_CLIP(value, low, high);                                    \
            destination[j] = (OUT_T)value;                                     \
        }                                                                      \
        counts[i] = total;                                                     \
    }                                                                          \
}

#define CLEANRL_REWARD_KERNEL(NAME, OUT_T)                                     \
void NAME(const cleanrl_norm_state *restrict state,                            \
          const double *restrict rewards, const double *restrict terminated,   \
          OUT_T *restrict out) {                                               \
    const long num_envs = state->num_envs;                                     \
    const double epsilon = state->epsilon;                                     \
    const double low = state->low;                                             \
    const double high = state->high;                                           \
    const double gamma = state->gamma;                                         \
    double *restrict means = state->means;                                     \
    double *restrict variances = state->variances;                             \
    double *restrict counts = state->counts;                                   \
    double *restrict returns = state->returns;                                 \
    for (long i = 0; i < num_envs; ++i) {                                      \
        const double reward = rewards[i];                                      \
        const double discounted =                                              \
            returns[i] * gamma * (1.0 - terminated[i]) + reward;               \
        returns[i] = discounted;                                               \
        const double count = counts[i];                                        \
        const double total = count + 1.0;                                      \
        const double delta = discounted - means[i];                            \
        means[i] = means[i] + delta / total;                                   \
        const double updated_variance =                                        \
            (variances[i] * count + delta * delta * count / total) / total;    \
        variances[i] = updated_variance;                                       \
        counts[i] = total;                                                     \
        double value = reward / __builtin_sqrt(updated_variance + epsilon);    \
        CLEANRL_CLIP(value, low, high);                                        \
        out[i] = (OUT_T)value;                                                 \
    }                                                                          \
}

CLEANRL_OBS_KERNEL(cleanrl_vector_obs_normalize_f32, float)
CLEANRL_OBS_KERNEL(cleanrl_vector_obs_normalize_f64, double)
CLEANRL_REWARD_KERNEL(cleanrl_vector_reward_normalize_f32, float)
CLEANRL_REWARD_KERNEL(cleanrl_vector_reward_normalize_f64, double)
"""

# -ffp-contract=off is load-bearing: without it GCC fuses `a * b + c` into an
# FMA and the result stops being bit-identical to NumPy. -fno-math-errno only
# drops sqrt's errno side effect (results are unchanged) and is what lets the
# inner loop vectorize at all. Nothing here relaxes IEEE arithmetic.
_KERNEL_FLAGS = (
    "-O3", "-march=native", "-std=c11", "-fPIC", "-shared",
    "-ffp-contract=off", "-fno-math-errno",
)
_KERNEL_NAMES = (
    "cleanrl_vector_obs_normalize_f32",
    "cleanrl_vector_obs_normalize_f64",
    "cleanrl_vector_reward_normalize_f32",
    "cleanrl_vector_reward_normalize_f64",
)

_native = None


class _NormState(ctypes.Structure):
    """Kernel-side view of one normalizer: addresses plus per-step invariants."""

    _fields_ = [
        ("means", ctypes.c_void_p),
        ("variances", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("returns", ctypes.c_void_p),
        ("epsilon", ctypes.c_double),
        ("low", ctypes.c_double),
        ("high", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("num_envs", ctypes.c_long),
        ("dim", ctypes.c_long),
    ]


def _host_isa():
    """Identify the CPU a ``-march=native`` build is valid for.

    The build cache may be shared between hosts; the CPU model keeps one
    machine's AVX-512 binary from being loaded on a machine without it.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def _native_kernels():
    """Build (once, cached on disk) and load the normalization kernels."""
    global _native
    if _native is not None:
        return _native
    fingerprint = hashlib.sha256(
        _KERNEL_SOURCE.encode()
        + " ".join(_KERNEL_FLAGS).encode()
        + _host_isa().encode()
    ).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = cache / "cleanrl" / "vector-norm-native" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "vector_norm_kernels.so"
    if not output.exists():
        # Concurrent processes compile separate files and atomically publish.
        library_fd, library_path = tempfile.mkstemp(suffix=".so", dir=directory)
        os.close(library_fd)
        source_fd, source_path = tempfile.mkstemp(suffix=".c", dir=directory)
        with os.fdopen(source_fd, "w") as handle:
            handle.write(_KERNEL_SOURCE)
        try:
            subprocess.run(
                ["cc", *_KERNEL_FLAGS, source_path, "-o", library_path, "-lm"],
                check=True, capture_output=True, text=True,
            )
            os.replace(library_path, output)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", str(error))
            raise RuntimeError(f"Unable to build native vector_norm kernels: {detail}") from error
        finally:
            for path in (library_path, source_path):
                if os.path.exists(path):
                    os.unlink(path)
    library = ctypes.CDLL(str(output))
    pointer = ctypes.c_void_p
    for name in _KERNEL_NAMES:
        kernel = getattr(library, name)
        kernel.argtypes = [pointer] * (4 if "reward" in name else 3)
        kernel.restype = None
    _native = library
    return library


class _Staged:
    """Private float64 staging buffer for one kernel input.

    Copying the input costs ~0.2us at these sizes -- less than extracting a
    pointer from the caller's array (``arr.ctypes.data`` alone is ~0.6us) -- and
    it applies the exact same widening cast ``np.asarray(x, dtype=np.float64)``
    does, so bool flags, float32 rows and lists all arrive identical to the
    reference path. Broadcasting inputs broadcast here exactly as they would
    there; anything the buffer cannot accept falls back to the NumPy path and
    raises NumPy's own error.
    """

    __slots__ = ("_buffer", "pointer")

    def __init__(self, shape):
        self._buffer = np.empty(shape, dtype=np.float64)
        self.pointer = self._buffer.ctypes.data

    def stage(self, values):
        try:
            self._buffer[...] = values
        except (ValueError, TypeError):
            return False
        return True


def make_raw_continuous_env(env_id, idx, capture_video, run_name):
    """Legacy ``make_env`` minus the normalization wrappers.

    Keeps ``FlattenObservation`` / ``RecordEpisodeStatistics`` / ``ClipAction``.
    Normalization is the caller's job via :class:`VectorObsNorm` /
    :class:`VectorRewardNorm`.
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


class _NativeNormalizer:
    """Shared kernel plumbing: staging buffers, output buffers, clip bounds."""

    def _setup_native(self, num_envs, out_shape, dim, inputs, returns=None):
        self._state = _NormState(
            self.means.ctypes.data, self.variances.ctypes.data, self.counts.ctypes.data,
            None if returns is None else returns.ctypes.data,
            0.0, 0.0, 0.0, 0.0, int(num_envs), int(dim),
        )
        self._state_pointer = ctypes.addressof(self._state)
        self._inputs = tuple(_Staged(shape) for shape in inputs)
        library = _native_kernels()
        # One reusable output buffer per supported dtype: the kernel writes it,
        # then a single ~0.2us copy hands the caller a freshly owned array (the
        # reference path allocated one per call, so callers may keep it).
        self._native = {}
        for dtype, suffix in ((np.float32, "f32"), (np.float64, "f64")):
            buffer = np.empty(out_shape, dtype=dtype)
            kernel = getattr(library, f"{self._kernel_prefix}_{suffix}")
            entry = (kernel, buffer, buffer.ctypes.data)
            # Scalar type and dtype object hash differently; accept both call
            # forms. Anything else falls back to the NumPy reference path.
            self._native[dtype] = entry
            self._native[np.dtype(dtype)] = entry
        self._configure()

    def _configure(self):
        """Re-read the public clip/epsilon/gamma attributes into kernel state.

        The reference path read them per call, so they stay late-bound; the
        hot path only pays two identity comparisons to notice a change.
        """
        clip = self.clip
        self._state.epsilon = float(self.epsilon)
        self._state.low = -np.inf if clip is None else -float(clip)
        self._state.high = np.inf if clip is None else float(clip)
        self._clip = clip
        self._epsilon = self.epsilon
        self._gamma = getattr(self, "gamma", None)
        self._state.gamma = 0.0 if self._gamma is None else float(self._gamma)


class VectorObsNorm(_NativeNormalizer):
    """Batched equivalent of N independent ``NormalizeObservation`` + clip.

    ``means`` / ``variances`` / ``counts`` are updated in place; the kernel
    holds their addresses, so read and mutate them but never rebind them.
    """

    _kernel_prefix = "cleanrl_vector_obs_normalize"

    def __init__(self, num_envs, obs_shape, epsilon=1e-8, clip=10.0):
        self.epsilon = epsilon
        self.clip = clip
        obs_shape = tuple(obs_shape)
        self.means = np.zeros((num_envs,) + obs_shape, dtype=np.float64)
        self.variances = np.ones_like(self.means)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)
        shape = self.means.shape
        self._setup_native(num_envs, shape, int(np.prod(obs_shape, dtype=np.int64)), (shape,))

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        """Update the selected rows' moments and return clipped normalization.

        Args:
            obs: ``(N, *obs_shape)`` batch, or ``(K, *obs_shape)`` with ``rows``.
            rows: ``None`` (whole batch), or a slice / int array / int selecting
                which stat rows the ``K`` observations belong to (autoreset
                second pass, manual resets).
        """
        if rows is None:
            native = self._native.get(out_dtype)
            if native is not None:
                staged = self._inputs[0]
                if staged.stage(obs):
                    if self.clip is not self._clip or self.epsilon is not self._epsilon:
                        self._configure()
                    kernel, buffer, pointer = native
                    kernel(self._state_pointer, staged.pointer, pointer)
                    return buffer.copy()
        return self._normalize_numpy(obs, rows, out_dtype)

    def _normalize_numpy(self, obs, rows, out_dtype):
        """Reference implementation: row subsets, odd dtypes, broadcasting."""
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
        # Singleton update: batch variance is 0, so only the delta term remains.
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

        # Subtract-then-divide, never a multiply by a reciprocal scale: the
        # reciprocal is not bit-identical to the division the wrappers do.
        out = (obs - new_means) / np.sqrt(new_variances + self.epsilon)
        if self.clip is not None:
            out = ufunc_clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)

    def normalize_step(self, raw_next_obs, terminations, truncations, infos):
        """Normalize one autoreset vector step in per-env wrapper order.

        Returns ``(next_obs, transition_obs)``; see module docstring. When no
        boundary occurred both entries are the SAME array (no copy): treat
        them as read-only, or copy before mutating.
        """
        # Overwhelmingly the common case: no env finished, so there is no
        # final observation to order before a reset and the whole step is one
        # batched update. np.count_nonzero has the cheapest dispatch of any
        # "did anything happen" reduction, and short-circuits on terminations.
        if not (np.count_nonzero(terminations) or np.count_nonzero(truncations)):
            transition_obs = self.normalize(raw_next_obs)
            return transition_obs, transition_obs
        return self._normalize_boundary_step(raw_next_obs, terminations, truncations, infos)

    def _normalize_boundary_step(self, raw_next_obs, terminations, truncations, infos):
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        boundaries = np.flatnonzero(np.logical_or(terminations, truncations))
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

        transition_obs = self.normalize(raw_transition)
        next_obs = np.array(transition_obs, copy=True)
        next_obs[boundaries] = self.normalize(
            np.asarray(raw_next_obs)[boundaries], rows=boundaries
        )
        return next_obs, transition_obs


class VectorRewardNorm(_NativeNormalizer):
    """Batched equivalent of N independent ``NormalizeReward`` + clip.

    ``returns`` / ``means`` / ``variances`` / ``counts`` are updated in place;
    the kernel holds their addresses, so never rebind them.
    """

    _kernel_prefix = "cleanrl_vector_reward_normalize"

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.means = np.zeros(num_envs, dtype=np.float64)
        self.variances = np.ones(num_envs, dtype=np.float64)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)
        shape = self.returns.shape
        self._setup_native(num_envs, shape, 1, (shape, shape), returns=self.returns)

    def normalize(self, rewards, terminations, out_dtype=np.float32):
        """Update discounted returns and return clipped normalized rewards."""
        native = self._native.get(out_dtype)
        if native is not None:
            staged_rewards, staged_terminated = self._inputs
            if staged_rewards.stage(rewards) and staged_terminated.stage(terminations):
                if (self.clip is not self._clip or self.epsilon is not self._epsilon
                        or self.gamma is not self._gamma):
                    self._configure()
                kernel, buffer, pointer = native
                kernel(self._state_pointer, staged_rewards.pointer,
                       staged_terminated.pointer, pointer)
                return buffer.copy()
        return self._normalize_numpy(rewards, terminations, out_dtype)

    def _normalize_numpy(self, rewards, terminations, out_dtype):
        """Reference implementation: odd dtypes and broadcasting inputs."""
        raw = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminations, dtype=np.float64)
        # In place: the kernel holds these addresses, so they must never be rebound.
        self.returns[...] = self.returns * self.gamma * (1.0 - terminated) + raw

        total_counts = self.counts + 1.0
        delta = self.returns - self.means
        self.means[...] = self.means + delta / total_counts
        # Singleton update: batch variance is 0.
        self.variances[...] = (
            self.variances * self.counts + np.square(delta) * self.counts / total_counts
        ) / total_counts
        self.counts[...] = total_counts

        out = raw / np.sqrt(self.variances + self.epsilon)
        if self.clip is not None:
            out = ufunc_clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)
