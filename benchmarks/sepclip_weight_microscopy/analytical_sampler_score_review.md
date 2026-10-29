# Independent sampler-score review

Pure analytical/source review. No model/test execution, new queue jobs, trainer edits or measured-bias claim.

## Verified identities

For an ideal continuous Beta variable X with a fixed clamp threshold e, differentiating the left-tail integral gives grad log F(e)=E[grad log p(X) | X<=e]. The right-tail identity follows analogously. The atom score is generally different from the continuous density score evaluated at the clamped endpoint.

For a fixed state, fixed continuation law, state-only baseline b and parameter direction d, the left-atom contribution to NAIVE endpoint-score expectation minus CORRECT atom-score expectation is P_L*(Q_e-b)*(s_e-s_L), with the stated sign. Conditioning on other independent action coordinates permits summing their coordinate score corrections without asserting that marginal atom counts bound joint-return bias. For a full closed-loop PG, the same local score contribution is one term of the all-time likelihood-ratio derivative; later-policy changes are accounted for by later score terms rather than differentiated inside this conditional Q_e.

Small atom probability alone supplies no error guarantee without return magnitude and directional score discrepancy. A zero observed count supplies no proof of zero population mass. The note therefore adds a justified interpretation contract beyond a generic warning, while establishing no consequential bias in any actual diagnostic.

## Quantile truncation correction

For U uniform on[delta,1-delta] and X=F^(-1)(U), the pre-rounding clamped left mass is max(F(e)-delta,0)/(1-2*delta), assuming e lies below the upper quantile. When positive, its score is grad F(e)/(F(e)-delta). The stated right formula is analogous. If a quantile endpoint remains in the unclamped interior, the interior support moves with parameters and derivative boundary terms are missing from a naive ordinary density-score expectation. Common quantiles improve paired finite-return comparisons; they cannot remove score-law mismatch.

One statement in the reviewed draft needs correction: ideal tail probability p is NOT universally an upper bound on its truncated atom mass. For p>delta with the threshold inside the retained quantile interval:

    p_delta - p = delta*(2*p-1)/(1-2*delta)

It is nonpositive only when p<=.5, and positive when p>.5. For a threshold beyond the upper retained quantile p_delta saturates at one. Using the exact clipped formula across all threshold regimes avoids this issue. A generic absolute difference bound is delta/(1-2*delta); the direction of the inequality depends on p. Observed shape ranges might justify p<=.5 for these checkpoints, but that must be established rather than assumed from e being small.

## Float32 scope

F(u)-F(l) is the exact mass of a fixed rounding bin if an IDEAL continuous Beta draw is subsequently quantized. This is a useful reference law, not literally the complete law of finite pseudorandom rng.beta or finite-precision inverse-CDF arithmetic. Their generator/algorithm errors and discrete random inputs add further distinctions. The host path casts before clipping; the diagnostic helper clips inverse-CDF output before its float32 cast. Effective atom/bin boundaries should follow each actual arithmetic path.

Similarly, backprop through the shape construction uses smooth operation derivatives evaluated at rounded intermediates. It is not a derivative of a literal parameter-to-floating-point probability step map. This ordinary numerical qualification does not prove that precision error or sampler mismatch materially changes an actual PG sign.

## Interpretation decision

The justified new content is the exact missing atom directional term and moving-support distinction. Preserve directly measured finite-policy effects under the declared sampler. Withhold exact physical-score sign claims if atom corrections, cancellation margins or numerical law differences remain consequential/unbounded. No corrective training or new experiment follows from this derivation alone.

## Reviewed hashes

- `benchmarks/sepclip_weight_microscopy/analytical_sampler_score_contract.md`: `6a5e9966da701d2be32ac1f5993c4583953cbb53ea183d8cd8b66b963cc0e086`
- `cleanrl/shared/sampling.py`: `1d7ebb4ff054c73bc7a6cfdd701c8cdf34e6d8e848dfe96f1dbe8ed4ad9aa610`
- `scripts/audit_sepclip_dense_pg.py`: `1f4053ebe4854063ce19e36b05be37dc3e036d94e368b9b424595c5b98b4c058`
- `scripts/audit_sepclip_closed_loop_update.py`: `59dec8199edcadb6d02264ad3a8f9545d7a4d5785ad98a25660b1ed05619c208`
- `scripts/audit_sepclip_physics.py`: `1d53db1d68f6fb188fb7207513b606a652f70682029e278031cd3adb25605114`
