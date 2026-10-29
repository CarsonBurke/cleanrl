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

Both take ``parameterization``: ``"softplus"`` (``1 + softplus``, the default),
``"mode"`` (:func:`mode_beta_concentrations`, sigmoid mode times exp
concentration) or ``"mode_softplus"`` (the same with ``2 softplus``
concentration). The mode heads' fused and NumPy concentrations agree to 1 ulp
rather than bitwise; see ``beta_concentrations_host``.
"""

import warnings

import numpy as np
import torch.nn.functional as F
from torch.distributions import Beta

from cleanrl.shared.host_graph import BetaHeadGraph

try:  # ``np.clip`` spends ~3us per call in Python-side deprecation checks; the ufunc is identical.
    from numpy._core.umath import clip as ufunc_clip
except ImportError:  # NumPy < 2
    from numpy.core.umath import clip as ufunc_clip


BETA_PARAMETERIZATIONS = ("softplus", "mode", "mode_softplus")


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


def beta_concentrations_host(logits, parameterization="softplus"):
    """NumPy ``(alpha, beta)`` for float32 ``(N, 2 * act_dim)`` logits split on the last axis.

    ``softplus`` is bit-identical to the fused kernel. The mode heads evaluate
    in float64 and round each concentration to float32 once, as the kernel does;
    NumPy's float64 exp and glibc's differ below FP64 resolution, so the two
    agree to at most 1 float32 ulp.
    """
    # Slicing the last axis is what np.split does here (even split, 2D input)
    # without its ~3us of Python-side bookkeeping and list construction.
    half = logits.shape[-1] // 2
    if parameterization == "softplus":
        concentration = np.logaddexp(0.0, logits, dtype=np.float32)
        concentration += 1.0
        return concentration[..., :half], concentration[..., half:]
    if parameterization not in BETA_PARAMETERIZATIONS:
        raise ValueError(f"parameterization must be one of {BETA_PARAMETERIZATIONS}")
    z = logits[..., :half].astype(np.float64)
    s = logits[..., half:].astype(np.float64)
    with np.errstate(over="ignore"):
        if parameterization == "mode":
            precision = np.exp(s)
        else:  # 2 softplus(s), in the kernel's log1p(exp(-|s|)) + max(s, 0) form
            precision = 2.0 * (np.maximum(s, 0.0) + np.log1p(np.exp(-np.abs(s))))
        alpha = 1.0 + precision / (1.0 + np.exp(-z))
        beta = 1.0 + precision / (1.0 + np.exp(z))
    return alpha.astype(np.float32), beta.astype(np.float32)


def sample_beta_actions_host(logits, low, high, rng, *, epsilon=1e-6, parameterization="softplus"):
    """NumPy Beta head over ``beta_concentrations_host(logits, parameterization)``.

    ``logits`` is float32 ``(N, 2 * act_dim)``; ``low``/``high`` are float32
    ``(act_dim,)``. Returns float32 ``(native, action)``; native is clipped to
    ``[epsilon, 1 - epsilon]`` after the float32 cast, matching the device path.
    """
    alpha, beta = beta_concentrations_host(logits, parameterization)
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

    def __init__(self, low, high, epsilon, reason=None, parameterization="softplus"):
        self._low, self._high, self._epsilon = low, high, epsilon
        self._parameterization = parameterization
        self.fallback_reason = reason

    def __call__(self, logits, rng):
        return sample_beta_actions_host(
            logits, self._low, self._high, rng, epsilon=self._epsilon,
            parameterization=self._parameterization)


def make_beta_sampler(num_envs, act_dim, low, high, *, epsilon=1e-6, fused=True,
                      parameterization="softplus"):
    """Return the fastest available ``(logits, rng) -> (native, action)`` Beta head.

    Prefers :class:`FusedBetaHead` and falls back to the plain NumPy
    :class:`NumpyBetaHead` when the kernel cannot take these bounds or cannot
    be built at all. Both produce bit-identical results from a bit-identical
    random stream (the mode head's concentrations: within 1 ulp), so callers
    never branch on the choice -- but the fallback
    costs ~2.8us per step, which is silent and easy to ship by accident, so it
    warns once with the kernel's own reason. The result carries ``fused`` and
    ``fallback_reason`` for logging.
    """
    if parameterization not in BETA_PARAMETERIZATIONS:
        raise ValueError(f"parameterization must be one of {BETA_PARAMETERIZATIONS}")
    reason = None
    if fused:
        try:
            graph = BetaHeadGraph(num_envs, act_dim, low, high, epsilon=epsilon,
                                  parameterization=parameterization)
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
    return NumpyBetaHead(low, high, epsilon, reason, parameterization)


def mode_beta_concentrations(logits, precision_map="exp"):
    """Mode/concentration Beta head: ``alpha = 1 + m nu``, ``beta = 1 + (1 - m) nu``.

    ``logits`` split on the last axis into ``(z_m, z_nu)``: ``m = sigmoid(z_m)``
    is the mode and ``nu`` the concentration above the uniform: ``exp(z_nu)``
    (parameterization ``"mode"``, the Beta analogue of (mean, log std)) or
    ``2 softplus(z_nu)`` (``"mode_softplus"``, linear growth like the
    ``1 + softplus`` head, and equal to it at zero logits). Unlike
    ``1 + softplus``, the mode's sensitivity to ``z_m`` does not shrink as the
    policy sharpens. ``1 - m`` is ``sigmoid(-z_m)``, which avoids float32
    cancellation near the bounds.
    """
    first, second = logits.chunk(2, dim=-1)
    if precision_map == "exp":
        precision = second.exp()
    elif precision_map == "softplus":
        precision = 2.0 * F.softplus(second)
    else:
        raise ValueError("precision_map must be 'exp' or 'softplus'")
    return 1.0 + first.sigmoid() * precision, 1.0 + (-first).sigmoid() * precision

