"""Distribution sampling without redundant device-to-host argument checks.

Use this when the policy guarantees positive finite Beta parameters by
construction (for example, ``1 + softplus(logits)``). Keep finite-value health
checks at an existing host synchronization, such as the action transfer or
metric logging. This preserves PyTorch's sampler and random-number stream;
it does not approximate a Beta using a different distribution.

``sample_beta_actions_host`` is the NumPy counterpart for host actors
(``host_actor.HostMLP``): same ``1 + softplus`` head, same clamp, same
rescaling, drawn from a ``numpy.random.Generator`` instead of the CUDA stream.
``make_beta_sampler`` is the same function with its arithmetic fused into two
native kernel calls around an untouched ``rng.beta``.
"""

import warnings

import numpy as np
from torch.distributions import Beta

from cleanrl.shared.host_graph import BetaHeadGraph

try:  # ``np.clip`` spends ~3us per call in Python-side deprecation checks; the ufunc is identical.
    from numpy._core.umath import clip as ufunc_clip
except ImportError:  # NumPy < 2
    from numpy.core.umath import clip as ufunc_clip


def sample_beta_actions(alpha, beta, low, high, *, epsilon=1e-6):
    """Return native [0,1] and rescaled actions using the standard Beta sampler.

    Parameters must already satisfy the Beta distribution's contract. Setting
    ``validate_args=False`` locally avoids the nested Dirichlet and Beta
    constructors synchronizing CUDA to inspect constraints in Python. Other
    distributions' validation settings are unaffected.
    """
    native = Beta(alpha, beta, validate_args=False).sample().clamp(
        epsilon, 1.0 - epsilon
    )
    return native, low + (high - low) * native


def sample_beta_actions_host(logits, low, high, rng, *, epsilon=1e-6):
    """NumPy Beta head: ``alpha, beta = 1 + softplus(logits)`` split on the last axis.

    ``logits`` is float32 ``(N, 2 * act_dim)``; ``low``/``high`` are float32
    ``(act_dim,)``. Returns float32 ``(native, action)``; native is clipped to
    ``[epsilon, 1 - epsilon]`` after the float32 cast, matching the device path.
    """
    concentration = np.logaddexp(0.0, logits, dtype=np.float32)
    concentration += 1.0
    # Slicing the last axis is what np.split does here (even split, 2D input)
    # without its ~3us of Python-side bookkeeping and list construction.
    half = concentration.shape[-1] // 2
    alpha, beta = concentration[..., :half], concentration[..., half:]
    native = rng.beta(alpha, beta).astype(np.float32)
    ufunc_clip(native, epsilon, 1.0 - epsilon, out=native)
    return native, low + (high - low) * native


class FusedBetaHead:
    """``sample_beta_actions_host`` with its arithmetic in two native calls.

    ``rng.beta`` is left exactly where it was: same generator, same alpha and
    beta values, same shape and dtype, one call per step, so the random stream
    is unchanged. What moves into ``host_kernel.c`` is the NumPy dispatch
    around it -- ``logaddexp``/``+= 1``/split before, and cast/clip/rescale
    after: 2.8us of the 13.4us Beta head at 16 envs and act_dim 6, paired
    A/B/A on a contended box (``scripts/benchmark_rollout_chain.py``).

    ``native`` and ``action`` are the graph's permanent buffers, overwritten by
    the next call (``HostGraphActor.__call__`` returns its logits the same
    way). Callers that keep a step's actions past the next one must copy.
    """

    fused = True
    fallback_reason = None

    def __init__(self, graph):
        self._graph = graph

    def __call__(self, logits, rng):
        alpha, beta = self._graph.concentration(logits)
        return self._graph.rescale(rng.beta(alpha, beta))


class NumpyBetaHead:
    """``sample_beta_actions_host`` bound to fixed bounds, for the fallback."""

    fused = False

    def __init__(self, low, high, epsilon, reason=None):
        self._low, self._high, self._epsilon = low, high, epsilon
        self.fallback_reason = reason

    def __call__(self, logits, rng):
        return sample_beta_actions_host(
            logits, self._low, self._high, rng, epsilon=self._epsilon)


def make_beta_sampler(num_envs, act_dim, low, high, *, epsilon=1e-6, fused=True):
    """Return the fastest available ``(logits, rng) -> (native, action)`` Beta head.

    Prefers :class:`FusedBetaHead` and falls back to the plain NumPy
    :class:`NumpyBetaHead` when the kernel cannot take these bounds or cannot
    be built at all. Both produce bit-identical results from a bit-identical
    random stream, so callers never branch on the choice -- but the fallback
    costs ~2.8us per step, which is silent and easy to ship by accident, so it
    warns once with the kernel's own reason. The result carries ``fused`` and
    ``fallback_reason`` for logging.
    """
    reason = None
    if fused:
        try:
            graph = BetaHeadGraph(num_envs, act_dim, low, high, epsilon=epsilon)
        except (TypeError, ValueError, RuntimeError, OSError) as error:
            reason = str(error)
        else:
            return FusedBetaHead(graph)
    if reason is not None:
        warnings.warn(
            f"BetaHeadGraph cannot fuse this Beta head ({reason}); falling back to "
            "sample_beta_actions_host, which costs roughly 2.8us more per step",
            RuntimeWarning, stacklevel=2,
        )
    return NumpyBetaHead(low, high, epsilon, reason)
