# What score the physical sampler requires

Analytical interpretation contract, not a new measured bias or proposed trainer change. The queued diagnostics already disclose this approximation. This note makes the missing quantity explicit so a small clipping probability cannot silently certify the physical PG oracle.

## Density, clamped atom and finite-precision laws differ

For an ideal continuous X~Beta(alpha,beta), let Y=clamp(X,e,1-e), with fixed e. The law has interior density p(y), left mass P_L=F(e), and right mass P_R=1-F(1-e). The likelihood scores are:

    interior: grad log p(y)
    left atom: grad log F(e)
    right atom: grad log [1-F(1-e)]

Differentiating the integral gives the useful identity:

    grad log P_L = E[grad log p(X) | X<=e]

The right tail has the analogous identity. Evaluating the continuous density score at y=e is generally different. It is not the atom's likelihood score.

For a fixed parameter direction d, write s_e=d·grad log p(e) and s_L=d·grad log P_L. Conditional on other action coordinates and a fixed state, the environment receives the same clamped torque for every latent X in the left tail. If Q_e is its expected continuation return, replacing the atom score with the endpoint density score changes the directional expectation by:

    P_L * (Q_e-b) * (s_e-s_L)

The right tail contributes the corresponding term. This formula holds for that ideal clamped law and a fixed state-only baseline b. Summing marginal corrections is possible for independent action coordinates by conditioning on the other coordinates; it does not turn observed atom counts into a confidence bound.

A probability bound alone does not control Q_e, the directional score discrepancy, or their time-weighted accumulation. In particular, score factors contain logarithms near the boundary. The missing numerical contract is a directional correction or a justified bound at the actual rounded shapes and motion. An observed zero atom count is evidence about the sampled trajectories, not proof of zero expectation or unbiased derivatives.

The real host sampler casts to float32 before clipping. Its atom preimage therefore includes a rounding interval around the threshold. For an ideal continuous Beta draw followed by quantization, any interior float32 bin [l,u) has exact mass F(u)-F(l), and its exact score is the derivative of that mass divided by that mass. This does not assert an exact description of the finite PRNG and rng.beta implementation. The familiar continuous-density score is a numerical approximation even away from the two clipping atoms. Analytic backprop through float32 shapes is itself a smooth surrogate, rather than the literal derivative of a parameter-to-rounded-probability step function. This is the usual finite-precision interpretation, not evidence of a consequential failure in this trainer.

## The diagnostic inverse-CDF law has another small mismatch

The dense and closed-loop diagnostics draw uniform quantiles on [delta,1-delta], delta=1e-7, invert the Beta CDF, then clamp native actions at e=1e-6. Before float32 rounding, its left atom mass is:

    P_L_delta = clip([F(e)-delta] / (1-2*delta), 0, 1)
    grad log P_L_delta = grad F(e) / [F(e)-delta], when delta<F(e)<1-delta

The right expression is analogous using 1-F(1-e). A saturated unit atom has zero score away from a saturation threshold; a zero-probability atom has no sampled score contribution. These are not the untruncated Beta atom masses. If a quantile cutoff lies in the unclamped interior, the support endpoint moves with the shapes; a naive continuous-score expectation also omits moving-support terms. The truncated mass differs from an ideal tail probability p by delta*(2*p-1)/(1-2*delta) when delta<p<1-delta; it is bounded above by p only when p<=.5. The absolute mass difference is at most delta/(1-2*delta), including a removed tail when p<=delta or a saturated unit atom when p>=1-delta. Reporting ideal tail probability is not an exact score correction for the declared truncated sampling law.

Common quantiles make finite-return arm comparisons paired. They do not make a mismatched likelihood-score estimator exact. Finite closed-loop return differences remain directly interpretable for the declared diagnostic sampler; their agreement or disagreement with continuous-score predictions needs this approximation caveat along with horizon, state occupancy and parameter-chord checks.

## Decision rule

Do not repair or replace the frozen trainer based on this possibility alone. First inspect pending artifacts for actual/theoretical atoms, rounded alpha/beta ranges and directional signal margins. If consequential atoms or unresolved directional cancellation prevent a sign conclusion, retain the finite-return effect and withhold the stronger score-PG interpretation. If an exact score correction becomes necessary, validate it as a separately versioned diagnostic against the declared sampling law before attributing a mechanism or selecting training.

Source scope: cleanrl/shared/sampling.py (float32 cast then clamp); scripts/audit_sepclip_dense_pg.py (truncated inverse CDF and continuous score); scripts/audit_sepclip_closed_loop_update.py (same noise law and atom caveat). No execution or source modification accompanies this note.
