# Independent static review: layer Fisher numerical V2

Static preparation APPROVED, no blocking numerical/source/test-contract defect found. No models, numerical helpers, CUDA work, tests or jobs were executed by this reviewer. V1 source/test and failed11905 remain frozen;11905 passed17 other cases and failed the exact uniform-Fisher case at5.501474831248743e-10 absolute diagonal error. Skipped11906/11907/11908 remain separate. This approval does not assert queued CUDA contract success.

Final NEW script SHA256: c1455aae7ebb5c54cde98d0eca5cf198617e07cf86ba398276544245b83b2bcb
Final NEW test SHA256: 27f47d43c7ab6f71a13fe06ecc10a77756615921120fd6bc2557777aff70aa7a
Static-expanded CUDA cases44: original18 test AST bodies/parametrizations unchanged, plus26 independent numerical/domain cases. Original uniform rtol/atol1e-13 are unchanged. V1 hashes independently remain41787fc45d28760786843e579c474ec387168f9c6c5a8076aa0df63a460b199e / d22c07402947d7b7204591c8c851227204e13fabd9913378d4b102c8b3edf66c.

## Mathematical/source review

The recurrence and Bernoulli coefficients agree with [NIST recurrence](https://dlmf.nist.gov/5.15.E5) and [trigamma expansion](https://dlmf.nist.gov/5.15.E8). Independent positive-integral/finite-geometric remainder derivation is in layer_fisher_v2_numerical_bound_derivation.md. No remainder is inferred by differentiating a log-Gamma error bound.

Every argument shifts by32, with exact mathematical recurrence and repeated-multiplication polynomial1/x+1/(2x²)+B2/x³ throughB14/x15. Coefficients1/6,−1/30,1/42,−1/30,5/66,−691/2730,7/6 are correct. First omitted term has magnitude(3617/510)/x17. Positive integral proof establishes0<=S_B14−T<=this bound for real positive arguments. Shifted shapes>=33 make truncation far below binary64 rounding; improved arithmetic addresses the failure without relaxing uniform tolerance.

Fisher diagonals use stable paired differences, never rounded standaloneT(a)−T(a+b). The reciprocal difference retains b as(b/(a+b+j))/(a+j); power differences use a positive geometric sum. a1e30,b1 retains the exact≈1/a² diagonal even when a+b rounds to a. Paired remainder has stronger positive bound|B16|*((a+32)^−17−(a+b+32)^−17), computed stably. Offdiagonal uses−T(a+b). Rounded concentration-input error belongs to the conditional elementary-operation budget.

The declared gamma512 absolute-work budget covers the fixed scalar arithmetic paths, rounded arguments/coefficient inputs and alternating-series work, with separate omitted remainder and4096 minimum-subnormal allowance. This remains conditional engineering under standard RN binary64 elementary arithmetic and gradual underflow, not a hardware/libdevice proof. Domain[1,1e39] includes finite actualFP32 shapes/sums. Helpers avoid backendpolygamma/transcendentalpow.

Computed-positive-diagonal congruence scaling produces diagonal1 and offdiagonal corr. Comparison gap subtracts8u from1−abs(corr), and normalized entry error gives eta. TRUEgap_lower=gap_lower−eta must be positive; the exact-relative quadratic error bound is eta/(gap_lower−eta), correctly using the EXACT mathematical Fisher quadratic denominator. The initial eta/gap convention was corrected beforefreeze; it would bound relative to computed quadratic instead. The<=1% retention gate plus numerical Cholesky prevents an unsupported universalSPD assertion. It preserves highly asymmetric tiny diagonals and withholds unresolved balanced gigantic-concentration modes without ridge, shapeclamp or regularization. Gap/budget telemetry appears in actual outputs.

## Independent meaningful contracts

All18 previous bodies/decorators remain AST-identical, including strict uniform identities and architecture/partition/Gram/PPO-sign/full-SiTU tests. The26 additions are8 standalone trigamma integrals,7 paired-difference integrals,6 actualFP32 Fisher/quadratic references,2 balanced-unresolved rejection fixtures and3 invalid-domain checks.

Standalone and paired references use100-digit-work-precision scaled Laplace quadrature and expm1, independently of implementation recurrence/asymptotics/polygamma. Each error must lie inside its advertised budget and below1e-13 relative error. a1e30,b1 has a separate exact recurrence identity and binary64 lost-addend witness. Arguments cover floor/fractional/shift/moderate/asymmetric/large regimes.

Six Fisher fixtures bind mpmath100-digit references to actual stored CUDAFP32 shapes: uniform,1/1e8, reverse,1e30/1, balanced1e6, and nextabove1=1+2^-23 againstbeta2. Congruence-scaled plus/minus/e1 directions test near-small-mode and tiny-diagonal quadratic errors against the REPORTED exact-relative bound; this is stronger than checking all entries against one global absolute tolerance. Balanced1e14/1e30 must be withheld. No implementation computation occurred in this review.

Scalar high-precision references execute only inside the MLQ contracts; evaluated tensors remain CUDA. Quadrature/mpmath100-digit working precision is not formal interval certification; analytic uniform/recurrence identities provide separate anchors. Sampled reference success is not an all-domain hardware theorem.

## Limits retained

Frozen trainers/updates/sampler/replayloader/V1 attempts remain untouched. Prior independent layer-review architecture/full-partition/sign/scope findings apply. Fisher entry/quadratic budgets DO NOT certify compiled JVPs, Gram contractions/accumulation, eigenvalue/Cholesky kernels, FP64 log-Gamma/digamma KL cancellation, surrogate autograd, or the native physical gradient law. Require the separate actual captured-chord finite-difference audit, layer/full-JVP agreement and Gram cancellation telemetry before interpreting small layer credits. Shape1 remains valid Beta geometry; smooth shapeAD is not a derivative of quantized output changes. No actual measurement, optimizer efficiency, chart preference, benchmark improvement or causal interpretation is certified by this static review.
