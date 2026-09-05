"""Batch the canonical CleanRL Gymnasium normalizers without replacing state.

Only the exact four-wrapper normalization/clip pipeline is eligible. Arbitrary
callbacks must retain their original per-environment execution order. The
recognizer inspects function code and globals; it never probes user callbacks.
"""

from types import FunctionType

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd


def _canonical_clip(value):
    return np.clip(value, -10, 10)


_CLIP_CODE = _canonical_clip.__code__
_NUMPY_CLIP = np.clip


def is_canonical_clip(function):
    """Recognize only ``lambda x: np.clip(x, -10, 10)`` and its def equivalent."""
    if not isinstance(function, FunctionType):
        return False
    code = function.__code__
    return (
        code.co_code == _CLIP_CODE.co_code
        and code.co_consts == _CLIP_CODE.co_consts
        and code.co_names == _CLIP_CODE.co_names
        and code.co_argcount == 1
        and code.co_kwonlyargcount == 0
        and not code.co_freevars
        and not function.__defaults__
        and not function.__kwdefaults__
        and function.__globals__.get("np") is np
        and np.clip is _NUMPY_CLIP
    )


class _Moments:
    """Reusable gather buffers; published arrays remain historical snapshots."""

    def __init__(self, states, shape):
        self.states = tuple(states)
        self.means = np.empty((len(states),) + shape, dtype=np.float64)
        self.variances = np.empty_like(self.means)
        self.counts = np.empty(len(states), dtype=np.float64)
        self.count_axes = (slice(None),) + (None,) * len(shape)

    def update(self, samples):
        # Read the actual objects on every step: callers may restore checkpoints
        # or individually reset an environment between vector steps.
        for i, state in enumerate(self.states):
            self.means[i] = state.mean
            self.variances[i] = state.var
            self.counts[i] = state.count
        # Retain Gymnasium's singleton reduction and arithmetic order, including
        # the zero-variance term (and NaN propagation for nonfinite inputs).
        singleton = samples[:, None]
        batch_mean = np.mean(singleton, axis=1)
        batch_var = np.var(singleton, axis=1)
        delta = batch_mean - self.means
        total = self.counts + 1
        count = self.counts[self.count_axes]
        denominator = total[self.count_axes]
        means = self.means + delta * 1 / denominator
        m_a = self.variances * count
        m_b = batch_var * 1
        m2 = m_a + m_b + np.square(delta) * count * 1 / denominator
        variances = m2 / denominator
        for i, state in enumerate(self.states):
            state.mean = means[i]
            state.var = variances[i]
            state.count = float(total[i])
        return means, variances


class CanonicalLegacyNormalization:
    """Batched exact float64 math over the original normalizer state objects.

    Inputs are final transition observations, before autoreset. Reset continues
    through the original wrappers, so final/reset ordering and independently
    restored normalizer states remain unchanged. Per-environment fields are
    read/published in small loops; all statistics arithmetic runs in batches.
    ``normalize`` returns ``None`` before updating any state if a runtime change
    makes the pipeline ineligible; the caller must then use the original path.
    """

    @classmethod
    def from_wrappers(cls, processors):
        expected = (
            gym.wrappers.NormalizeObservation,
            gym.wrappers.TransformObservation,
            gym.wrappers.NormalizeReward,
            gym.wrappers.TransformReward,
        )
        if not processors:
            return None
        shape = None
        for row in processors:
            if tuple(type(wrapper) for wrapper in row) != expected:
                return None
            obs, obs_transform, reward, reward_transform = row
            if not is_canonical_clip(obs_transform.f) or not is_canonical_clip(reward_transform.f):
                return None
            if any("normalize" in wrapper.__dict__ for wrapper in (obs, reward)):
                return None
            for state in (obs.obs_rms, reward.return_rms):
                if type(state) is not RunningMeanStd or any(
                    name in state.__dict__ for name in ("update", "update_from_moments")
                ):
                    return None
            current_shape = np.shape(obs.obs_rms.mean)
            if shape is None:
                shape = current_shape
            if current_shape != shape or np.shape(reward.return_rms.mean) != ():
                return None
        return cls(processors, shape)

    def __init__(self, processors, obs_shape):
        self.processors = tuple(processors)
        self.clip_functions = tuple((row[1].f, row[3].f) for row in processors)
        self.clip_codes = tuple((row[1].f.__code__, row[3].f.__code__) for row in processors)
        self.observation_wrappers = tuple(row[0] for row in processors)
        self.reward_wrappers = tuple(row[2] for row in processors)
        self.observation_moments = _Moments(
            tuple(wrapper.obs_rms for wrapper in self.observation_wrappers), obs_shape
        )
        self.reward_moments = _Moments(
            tuple(wrapper.return_rms for wrapper in self.reward_wrappers), ()
        )
        self.observation_epsilon = np.empty(len(processors), dtype=np.float64)
        self.reward_epsilon = np.empty(len(processors), dtype=np.float64)
        self.gamma = np.empty(len(processors), dtype=np.float64)
        self.returns = np.empty(len(processors), dtype=np.float64)
        self.observation_axes = (slice(None),) + (None,) * len(obs_shape)

    def normalize(self, observations, rewards, terminations):
        # Replacing a callback or normalizer implementation after construction
        # must immediately restore row-wise execution, without invoking it here.
        if np.clip is not _NUMPY_CLIP:
            return None
        for row, functions, codes in zip(self.processors, self.clip_functions, self.clip_codes):
            if (row[1].f is not functions[0] or row[3].f is not functions[1]
                    or functions[0].__code__ is not codes[0] or functions[1].__code__ is not codes[1]
                    or functions[0].__globals__.get("np") is not np
                    or functions[1].__globals__.get("np") is not np
                    or "normalize" in row[0].__dict__ or "normalize" in row[2].__dict__):
                return None
            for state in (row[0].obs_rms, row[2].return_rms):
                if type(state) is not RunningMeanStd or any(
                    name in state.__dict__ for name in ("update", "update_from_moments")
                ):
                    return None
                if np.asarray(state.mean).dtype != np.float64 or np.asarray(state.var).dtype != np.float64:
                    return None
            if row[2].returns.dtype != np.float64:
                return None
        for i, (obs, reward) in enumerate(zip(self.observation_wrappers, self.reward_wrappers)):
            self.observation_epsilon[i] = obs.epsilon
            self.reward_epsilon[i] = reward.epsilon
            self.gamma[i] = reward.gamma
            self.returns[i] = reward.returns[0]
            # Checkpoint restoration may replace RMS objects, not just fields.
            if obs.obs_rms is not self.observation_moments.states[i]:
                self.observation_moments.states = tuple(
                    wrapper.obs_rms for wrapper in self.observation_wrappers
                )
            if reward.return_rms is not self.reward_moments.states[i]:
                self.reward_moments.states = tuple(
                    wrapper.return_rms for wrapper in self.reward_wrappers
                )
        means, variances = self.observation_moments.update(observations)
        normalized_obs = np.clip(
            (observations - means) / np.sqrt(variances + self.observation_epsilon[self.observation_axes]),
            -10, 10,
        )
        returns = self.returns * self.gamma * (1 - terminations) + rewards
        for i, wrapper in enumerate(self.reward_wrappers):
            wrapper.returns = returns[i:i + 1]
        _, variances = self.reward_moments.update(returns)
        normalized_rewards = np.clip(rewards / np.sqrt(variances + self.reward_epsilon), -10, 10)
        return normalized_obs, normalized_rewards
