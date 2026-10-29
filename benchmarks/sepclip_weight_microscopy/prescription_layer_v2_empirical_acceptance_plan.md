# Layer Fisher V2: empirical extraction and acceptance plan

Prepared before numerical gate 11926 and actual layer outputs 11928–11930, 11933–11934. No execution or submission was performed here. Source remains frozen at c1455aae7ebb5c54cde98d0eca5cf198617e07cf86ba398276544245b83b2bcb; test remains frozen at 27f47d43c7ab6f71a13fe06ecc10a77756615921120fd6bc2557777aff70aa7a. Optimizer forensic gate 11937 and actual forensic jobs 11938–11942 are separate evidence, not assumed successes.

## Artifact acceptance

For each completed artifact, record job/attempt state, artifact byte hash, execution manifest byte hash, executed source hashes, frozen trainer/loader hashes, snapshot/trace/training-manifest pins and architecture/optimizer/step. Compare executed hashes with the frozen inventory; successful exit alone does not establish the expected source. Match the independently reviewed captured-chord artifact to the same exact snapshot/trace. Failed attempts and missing outputs remain visible and receive no inferred measurements.

Gate 11926 must pass its 44 numerical contracts before interpreting V2 metric telemetry. Preserve V1 gate 11905's uniform Fisher failure and 17 passing cases. A successful V2 gate repairs a measured method defect; it does not retrospectively validate V1 actual outputs or certify arbitrary concentration ranges.

## Conditional scalar metric precision

Extract the maximum entry absolute-error budget, minimum computed normalized gap lower bound, minimum true gap lower bound and maximum conditional relative quadratic error from fisher_numerical_contract. Require finite values, positive true gap and maximum conditional error at most the reported 0.01 limit. Verify the denominator label is eta/(gap_lower-eta), relative to the exact mathematical Fisher quadratic. Retain the RN FP64/gradual-underflow engineering assumptions alongside the numbers.

The script withholds the entire geometry on an unresolved coordinate's small mode; do not treat a missing artifact as a zero metric or a subset measurement. Positive Fisher entries or a passing Cholesky factor are insufficient on their own. The scalar metric bound does not bound compiled JVP error, Gram contraction error, finite KL digamma/lgamma arithmetic, or full hardware behavior. No ridge or small-eigenvalue clamp is authorized to replace missing evidence.

## Full joint layer response

For both first and endpoint chords, independently recompute the Gram diagonal sum, full matrix sum and cross sum = joint minus diagonal. Action coordinates are summed and saved batch states are averaged; do not divide by action count. Report joint/diagonal when diagonal is positive, the sign and size of cross terms, and each layer's self energy. A self energy is the response from moving that layer alone, not its unique allocation of the coupled total. Signed row sums can be reported as cross-inclusive descriptive allocations with that caveat.

Check matrix symmetry, finite entries and numerical minimum eigenvalue against a scale-aware FP64 contraction allowance. A tiny negative eigenvalue does not prove a negative true Fisher direction; a materially negative one requires withholding. Report the actual allowance and ratio to matrix scale instead of silently coercing an eigenvalue to zero.

Recompute and report separately: Gram sum versus summed-layer response energy; Gram sum versus independently evaluated full JVP energy; layer-summed shape and logit JVP L2 residuals. Normalize energy discrepancies by their positive reference energies and response discrepancies by full response norms where available. Null full responses need absolute residual reporting because a relative denominator is undefined. Cancellations can amplify relative residuals even when individual layers are large; include diagonal energy and the layer norm scale.

No new numerical cutoff is preregistered by this note for these contraction residuals. Record exact measurements and identify anomalies before accepting an interpretation. The separate captured full-chord numerical contract remains necessary: passing scalar Fisher precision cannot establish derivative accuracy or a correct actor implementation.

## First versus endpoint motion and finite distribution change

Report full logit JVP norm, finite observed logit-delta norm, absolute finite-minus-linear residual and relative-to-JVP residual for both chords. First is old plus the stored first retracted delta; endpoint is the independently saved model after all ten epochs. Do not label endpoint as an independent tenth-epoch increment. Floor counts are separate alpha/beta action-position counts and do not certify the derivatives of the rounded storage map.

Report finite old-to-bank FP64 Beta KL and half the full joint Fisher energy, with their ratio only when the latter is positive. FP64 evaluation of actual FP32 shapes repairs a precision convention, but the KL helper still uses CUDA lgamma/digamma and has no independently derived uniform error budget. Preserve any negative KL or implausible tiny-motion discrepancy; withhold KL-ratio interpretation when its numerical sign/scale is unresolved. A KL/Fisher mismatch combines finite-path curvature, rounded shape behavior and measurement error. It cannot alone identify which mechanism caused a return difference.

Use the captured full-chord audit's complete step-size table, adjacent representable directions, independent finite differences and output-storage heuristic. Do not choose a monotone smaller h or the most favorable h after seeing the result. First and endpoint magnitudes differ; each needs its own rounding and linearization assessment.

## Stored objective and all ten update traces

Independently sum old_stored_clipped_ppo_ascent_credit_by_layer and compare with the full old-policy covector pairing. This is the actual sign-conditioned clipped stored surrogate, with saved GAE and the source advantage-normalization convention. It is a fixed-old-objective first-order pairing, not a fitted physical gradient or finite return difference.

Require ten ordered epoch/update rows. For every layer and epoch extract clipped gradient norm, ambient and retracted delta norms, normalization-correction norm, relative weight motion and both negative clipped-loss-gradient dot delta values. Positive values mean predicted decrease of that epoch's saved clipped loss. Pairings at different epochs use different gradients and preceding weights; summing them is descriptive path accounting, not the old-policy derivative of the endpoint or a finite loss telescoping identity.

Trace base weights accumulate saved retracted deltas and are not independently saved intermediate models. Exact endpoint/optimizer continuation belongs to the separate forensic acceptance gate. In v15, recorded ambient_delta means the actual intrinsic optimizer endpoint before global normalization, not a hypothetical Euclidean master proposal.

Validate actual geometry labels and constraint axes. Additive v14 actor input projection and all six body matrices are free Euclidean; head rows are constrained. Residual mixing/gain/bias are free parameters. Unit critic/body axes apply only where the actual architecture defines them. Report constrained normal/tangent gradient components only on those axes. No radial loss or tangent projection should be invented for free v14 body matrices. Constrained basepoint unit residuals reflect reconstructed weights and rounding, not independent epoch checkpoint certification.

## Reporting boundary

These outputs can establish where actual captured parameter motion produces smooth local distribution response, constructive or destructive layer interaction, stored-objective credit and normalization effects. They do not supply reliable physical-PG replicas, retained-range efficiency, a global optimum claim, causal benchmark attribution or chart selection. Compare captured models/steps as measured cohorts; differing training paths and visited-state distributions remain confounders. Keep actual measurements, conditional numerical acceptance, source-backed interpretation and missing causal evidence separate.
