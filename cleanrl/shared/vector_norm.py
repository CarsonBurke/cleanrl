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
"""

import gymnasium as gym
import numpy as np

_RMS_COUNT_INIT = 1e-4


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


class VectorObsNorm:
    """Batched equivalent of N independent ``NormalizeObservation`` + clip."""

    def __init__(self, num_envs, obs_shape, epsilon=1e-8, clip=10.0):
        self.epsilon = epsilon
        self.clip = clip
        self.means = np.zeros((num_envs,) + tuple(obs_shape), dtype=np.float64)
        self.variances = np.ones_like(self.means)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        """Update the selected rows' moments and return clipped normalization.

        Args:
            obs: ``(N, *obs_shape)`` batch, or ``(K, *obs_shape)`` with ``rows``.
            rows: ``None`` (whole batch), or a slice / int array / int selecting
                which stat rows the ``K`` observations belong to (autoreset
                second pass, manual resets).
        """
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

        out = (obs - new_means) / np.sqrt(new_variances + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)

    def normalize_step(self, raw_next_obs, terminations, truncations, infos):
        """Normalize one autoreset vector step in per-env wrapper order.

        Returns ``(next_obs, transition_obs)``; see module docstring. When no
        boundary occurred both entries are the SAME array (no copy): treat
        them as read-only, or copy before mutating.
        """
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
            # normalize() never mutates its input. Avoid copying every ordinary
            # transition merely to handle the rare autoreset case.
            raw_transition = raw_next_obs

        transition_obs = self.normalize(raw_transition)
        if not boundaries.size:
            return transition_obs, transition_obs
        next_obs = np.array(transition_obs, copy=True)
        next_obs[boundaries] = self.normalize(
            np.asarray(raw_next_obs)[boundaries], rows=boundaries
        )
        return next_obs, transition_obs


class VectorRewardNorm:
    """Batched equivalent of N independent ``NormalizeReward`` + clip."""

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.means = np.zeros(num_envs, dtype=np.float64)
        self.variances = np.ones(num_envs, dtype=np.float64)
        self.counts = np.full(num_envs, _RMS_COUNT_INIT, dtype=np.float64)

    def normalize(self, rewards, terminations, out_dtype=np.float32):
        """Update discounted returns and return clipped normalized rewards."""
        raw = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminations, dtype=np.float64)
        self.returns = self.returns * self.gamma * (1.0 - terminated) + raw

        total_counts = self.counts + 1.0
        delta = self.returns - self.means
        self.means = self.means + delta / total_counts
        # Singleton update: batch variance is 0.
        self.variances = (
            self.variances * self.counts + np.square(delta) * self.counts / total_counts
        ) / total_counts
        self.counts = total_counts

        out = raw / np.sqrt(self.variances + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(out_dtype, copy=False)
