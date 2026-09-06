"""Distribution sampling without redundant device-to-host argument checks.

Use this when the policy guarantees positive finite Beta parameters by
construction (for example, ``1 + softplus(logits)``). Keep finite-value health
checks at an existing host synchronization, such as the action transfer or
metric logging. This preserves PyTorch's sampler and random-number stream;
it does not approximate a Beta using a different distribution.

``sample_beta_actions_host`` is the NumPy counterpart for host actors
(``host_actor.HostMLP``): same ``1 + softplus`` head, same clamp, same
rescaling, drawn from a ``numpy.random.Generator`` instead of the CUDA stream.
"""

import numpy as np
from torch.distributions import Beta

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
