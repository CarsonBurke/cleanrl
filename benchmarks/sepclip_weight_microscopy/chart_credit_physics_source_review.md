# Independent physics source review: saved chart credit

Read-only static review of the new chart-credit script and focused CUDA contracts. No model/test execution or queue submission; existing trainers and other reports preserved. Current preparation passes independent static physics review after the numerical-domain repair described below. This is not a numerical pass or causal result.

## Sound source interpretation and mathematics

- Input requires oracle.first_action_only explicitly True, initial Sx2D logits/SxD shapes, selected t,row,episode_age identity metadata, nonempty horizons and at least two complete finite replica fields. Recognized objectives remain separate finite_raw, finite_normalized and bootstrap_normalized. No actor/environment/fit or actor-reachability claim is introduced.
- Recorded FP32 shapes must match separately re-evaluated FP32 1+softplus(recorded logits) within two ULP. Recorded values remain unchanged. This binds law conventions while allowing arithmetic differences, but is not bitwise proof of the historical sampler law.
- The implemented actual-excess Jacobian is [[AB/(A+B),A],[-AB/(A+B),B]], equivalent to the r/ell chart derivative. Shape covector is g_z divided by FP64 sigmoid of recorded FP32 logits, and chart covector is Jacobian transpose times shape covector. Shapes/Fisher are actual-rounded; the sigmoid chain is explicitly a hybrid smooth surrogate. This distinction is honest and required.
- Actual shape1 is valid Beta but invalid interior r/ell. Such rows serialize coordinates as null and remain visible, rather than replacing them with intended pre-add softplus. Relative latent-minus-actual excess and r/ell discrepancies expose cancellation near shape1; absolute excess errors alone would conceal this.
- Shape/chart Fisher Cholesky/finite failures are withheld without regularization. Static raw geometry can still be retained for diagnosis. Transformed credit overflow is masked across mean and ALL replicas so a coordinate's reported replica dispersion uses one common domain.
- Aggregate energy sums valid per-joint contributions and divides by ALL selected states, not only surviving rows. The report labels absent withheld coordinates explicitly; this is a partial-domain noisy point energy, not a completed full-field estimate. If every conversion is nonfinite the aggregate is null rather than0.
- Replica chart covectors/slopes, SD and SE are descriptive conditional action/continuation uncertainty. They are not independent state/seed uncertainty, signal-only energy, held-out direction validation or a causal training gate. No significance certificate or optimal-chart claim is generated.

## Numerical-domain gap found statically

The pre-conversion SPD gate checks shape and chart Fishers, but transform_credit also solves the logit Fisher diag(sigmoid)*F_shape*diag(sigmoid). That third solve can fail before the later finite-field mask runs.

A concrete accepted-input example is recorded z_alpha=-700, actual alpha=1+one_FP32_ULP, with beta2 and its corresponding finite logit. Expected FP32 alpha is1; the recorded one-ULP excess passes the two-ULP guard. Its actual-shape chart is interior and shape/chart Fishers can be SPD. FP64 sigmoid(-700) is positive around1e-304, so the current usable derivative test passes; its squared scale underflows the logit Fisher entry to0. The logit matrix is singular and torch.linalg.solve can throw instead of returning a withheld coordinate. No execution is needed to establish this underflow sequence.

This synthetic accepted-JSON robustness defect is not evidence of an actual trained policy error: genuine captured FP32 shape1 rows are already withheld. Before declaring source prep complete, guard finite/SPD of the SAME logit Fisher used by the solve, or use an equally explicit coordinate-wise guarded solve policy, and add a focused synthetic contract. No ridge/clamp should silently alter the geometry.

## Focused contracts and remaining limits

The prepared contracts compare chart density gradients to independent autograd, shape Fisher to a logpartition Hessian, shape Jacobians to centered finite differences, and quadratic/covector invariance including cross terms. They cover moderate, asymmetric/near-bound, extreme uncapped, actualshape1/null, false mean/shape identity, partial withholding/all-state divisor and relative rounding discrepancies. They are meaningful preparation contracts, not executed results.

Numerical SPD does not certify adequate inverse accuracy; recorded duality and mapped-natural-direction residuals remain interpretation gates. Exact natural-coordinate invariance holds for an invertible local chart and a matching smooth-law Fisher; coordinate slope or noisy dual energy alone cannot establish superiority of practical coordinate-dependent optimization. Objective, fixed-continuation/selected-state scope, tail and sampler approximations inherited from dense PG remain unchanged.

No physical gradient reliability, causal benchmark gain, global optimum, actor function-space reachability or candidate selection follows from this static review.

Reviewed source SHA256 at initial static pass:

- `scripts/audit_sepclip_chart_credit.py`: `89d821323c352df5141adf9a1adc6ae350f037202a9bd260a6714f980269fe6c`
- `tests/test_sepclip_chart_credit.py`: `ab789012a21abc4862d3ceaac8bf4ed8ca0c49ecc2f7b5d7e72de16b4fe1f0f3`

## Final repair receipt and static disposition

Geometry owner added finite/SPD checking of the SAME smooth logit Fisher before all transform solves, unioned its exclusions with shape/chart failures, retained that raw metric and separate smooth-logit withheld coordinates, and added the accepted-two-ULP z=-700/actualshape1+ULP fixture. I inspected the repair read-only: it closes the concrete underflow/unguarded-solve sequence without ridge, clamps or changing frozen data. No remaining blocking source/formula/objective-interpretation defect was found in this static review. Preparation approved, execution unvalidated.

Final reviewed source hashes:

- `scripts/audit_sepclip_chart_credit.py`: `89d821323c352df5141adf9a1adc6ae350f037202a9bd260a6714f980269fe6c`
- `tests/test_sepclip_chart_credit.py`: `ab789012a21abc4862d3ceaac8bf4ed8ca0c49ecc2f7b5d7e72de16b4fe1f0f3`
