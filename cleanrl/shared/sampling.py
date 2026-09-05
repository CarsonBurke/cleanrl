"""Distribution sampling without redundant device-to-host argument checks.

Use this when the policy guarantees positive finite Beta parameters by
construction (for example, ``1 + softplus(logits)``). Keep finite-value health
checks at an existing host synchronization, such as the action transfer or
metric logging. This preserves PyTorch's sampler and random-number stream;
it does not approximate a Beta using a different distribution.
"""

from torch.distributions import Beta


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
