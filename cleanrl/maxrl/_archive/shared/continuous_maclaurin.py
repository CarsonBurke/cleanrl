"""Continuous-reward generalisation of MaxRL's truncated Maclaurin weights.

MaxRL (arXiv 2602.02710) expands log p = -sum_k fail@k / k and truncates at order T.
Its released estimator (verl/trainer/ppo/maclaurin.py) is binary-only: with C successes
out of N rollouts and f = N - C failures,

    w_succ = 1/N                            (the k=1, REINFORCE, term)
    w_fail = -(1/N) * sum_{k=2..min(T,f)} C(f-1, k-1) / C(N-1, k-1)

The ratio C(f-1,k-1)/C(N-1,k-1) is exactly the probability that a uniformly random
(k-1)-subset of the OTHER N-1 rollouts is entirely failures -- i.e. the normalised
elementary symmetric polynomial of degree k-1 in the other rollouts' failure
INDICATORS. That reading is what generalises: for a non-binary reward r in [0,1],
the failure indicator 1{rollout j failed} becomes the failure MASS q_j = 1 - r_j, and

    e_m^(-i) := ( sum over |S|=m, S subset of others, prod_{j in S} q_j ) / C(N-1, m)
    w_i      := (1/N) * [ r_i - (1 - r_i) * sum_{k=2..T} e_{k-1}^(-i) ]

Setting r in {0,1} recovers the released formula element for element (pinned by
tests/test_continuous_maclaurin.py), so this is a strict generalisation rather than a
different estimator that happens to agree in spirit.

Why it matters here: the paper's own Appendix M.4 extends MaxRL to non-binary rewards
only as the crude ratio advantage (r - mu)/mu, reports that it beats GRPO by a large
margin on a continuous-reward maze (Fig. 31, where GRPO's Best@32 collapses), and then
explicitly leaves "the full theoretical treatment and broader empirical evaluation of
non-binary rewards to future work". The truncated series is what makes the binary
estimator bounded and low-variance; this supplies the same for the continuous case, so
a dense-reward task need never be thresholded into a Bernoulli to use MaxRL.
"""

import torch


def failure_mass_symmetric_means(failure_mass, order):
    """Normalised elementary symmetric polynomials of the OTHER rollouts, per rollout.

    Args:
        failure_mass: (..., N) tensor of q_j = 1 - r_j in [0, 1].
        order: highest degree m to return (the series' T - 1).

    Returns:
        (..., N, order + 1) tensor whose [..., i, m] entry is e_m^(-i), the MEAN of
        prod_{j in S} q_j over all m-subsets S of the N-1 rollouts other than i.
        Degree 0 is 1 by convention (the empty product).

    Computed by an explicit leave-one-out DP in float64 rather than by dividing the
    full-set polynomial, because the deflation recurrence E_m^(-i) = E_m - q_i E_{m-1}^(-i)
    alternates signs and loses precision exactly where q_i -> 1, which is the
    all-failing group MaxRL cares most about. N is a group size (single digits), so
    the O(N^2 T) direct form is free.
    """
    if failure_mass.shape[-1] < 1:
        raise ValueError("need at least one rollout per group")
    if order < 0:
        raise ValueError("order must be non-negative")
    group_size = failure_mass.shape[-1]
    q = failure_mass.to(torch.float64)
    degrees = min(order, group_size - 1)

    columns = []
    for excluded in range(group_size):
        keep = [j for j in range(group_size) if j != excluded]
        # poly[..., m] accumulates the UNNORMALISED sum over m-subsets of `keep`.
        poly = torch.zeros(q.shape[:-1] + (degrees + 1,), dtype=torch.float64, device=q.device)
        poly[..., 0] = 1.0
        for j in keep:
            # Descending m so each step consumes the previous iteration's values.
            for m in range(min(degrees, group_size - 1), 0, -1):
                poly[..., m] = poly[..., m] + q[..., j] * poly[..., m - 1]
        columns.append(poly)
    stacked = torch.stack(columns, dim=-2)

    # Normalise degree m by C(N-1, m) to turn each sum into a mean over m-subsets.
    counts = torch.ones(degrees + 1, dtype=torch.float64, device=q.device)
    for m in range(1, degrees + 1):
        counts[m] = counts[m - 1] * (group_size - m) / m
    normalised = stacked / counts

    if degrees < order:
        # Degrees above N-1 have no subsets to average, so their mean is 0.
        pad = torch.zeros(normalised.shape[:-1] + (order - degrees,), dtype=torch.float64, device=q.device)
        normalised = torch.cat((normalised, pad), dim=-1)
    return normalised


def continuous_maclaurin_weights(rewards, order):
    """Per-rollout score weights for the order-T continuous MaxRL gradient.

    Args:
        rewards: (..., N) tensor of r_i in [0, 1]. Group axis is the last one.
        order: truncation order T >= 1. T=1 is REINFORCE (w_i = r_i / N).

    Returns:
        (..., N) float64 tensor w, to be used as  ghat = sum_i w_i * grad log m(z_i).

    Reduces exactly to maclaurin.maclaurin_weights on binary rewards.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    r = rewards.to(torch.float64)
    if torch.any(r < -1e-9) or torch.any(r > 1.0 + 1e-9):
        raise ValueError("continuous MaxRL requires rewards in [0, 1]")
    r = r.clamp(0.0, 1.0)
    group_size = r.shape[-1]

    weights = r.clone()
    if order >= 2 and group_size >= 2:
        # Degrees 1..T-1 are the k=2..T terms of the series.
        means = failure_mass_symmetric_means(1.0 - r, order - 1)
        tail = means[..., 1:order].sum(dim=-1)
        weights = weights - (1.0 - r) * tail
    return weights / group_size
