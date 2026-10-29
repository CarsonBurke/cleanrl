# Independent chart-credit source review

Static review only; no model/test execution, new submission or source mutation by this reviewer. Reviewed `scripts/audit_sepclip_chart_credit.py`, its focused tests, the dense-PG producer and `analytical_chart_identification.md`. Algebra and normal producer layout are correct. This is a coordinate report for existing conditional first-action credit, not an actual candidate gradient experiment or causal optimum oracle.

## Algebra and layout

For actual interior shapes a,b promoted unchanged from FP32, x=a-1,y=b-1,tau=x+y and q=x/tau. The implemented Jacobian is L=[[x*y/tau,x],[-x*y/tau,y]], correctly mapping mode-logit/excess-log-concentration increments into shape increments. Shape Fisher uses the Beta log-normalizer Hessian: diagonal trigamma(a/b)-trigamma(a+b), offdiagonal -trigamma(a+b). Chart Fisher is L^T*F_shape*L, retaining cross terms.

Credit conversion g_shape=g_logit/sigmoid(recorded_logits), g_chart=L^T*g_shape is correctly labeled a hybrid smooth-logit chain evaluated on the actual rounded shape geometry. Joint dual energy and mapped natural-direction invariance are mathematically correct where the transforms are invertible and the arithmetic sufficiently conditioned. It is not literal differentiation of rounded FP32 arithmetic, nor bit-exact v13's latent-before-addition anchor chart. The reported latent/actual-excess discrepancy and this limitation are material and correctly preserved.

`paired` converts S-by2D alpha-half/beta-half arrays to S-byD-by2 and R-byS-by2D replica arrays to R-byS-byD-by2. Boolean usable masks then select matching state/action pairs in shapes, sigmoid and mean/replica fields. Scatter restores S-byD plus trailing geometry axes, preserving withheld entries as null. This layout agrees with the dense producer.

The producer explicitly casts replicas to NumPy float64 before computing their stored mean. Thus checking the saved mean against the recomputed FP64 replica mean with the declared tolerance is compatible with this producer; it is not incorrectly demanding FP64 agreement of a stored FP32 aggregate.

Shape-one coordinates, nonfinite/below-bound shapes, zero sigmoid derivatives and numerically non-SPD Fishers are withheld without a clamp. The implementation requires explicit `first_action_only=True`; missing or differently scoped oracle metadata cannot silently become initial-action credit. Aggregate energy sums all usable action dimensions and divides by ALL selected state count, keeping uniform-state weights and avoiding survivor renormalization. It honestly reports absent coordinates and identifies this as noisy point-estimated energy rather than signal energy, actor reachability, candidate selection or fit efficiency.

## Independent tests

The tests use an independently differentiated Beta log-normalizer Hessian, direct chart-density autograd and finite differences of separately defined shape coordinates. Asymmetric, near-boundary and high-shape fixtures are useful. Natural/covector and Fisher pairing invariance tests retain cross terms. Shape-one and extreme precision cases test withheld domains, while permanent null serialization prevents concealed numerical replacement. Metadata rejection tests cover missing scope/state count and nonfinite logits.

Two additional focused assertions would strengthen the contract: reject an inconsistent saved mean/replica average, and verify that partial withholding still divides aggregate energy by all selected states rather than only surviving states. A multi-state/multi-action fixture with deliberately distinct alpha/beta credit values would independently pin the half-layout and scatter mapping, beyond simple single-state fixtures.

## Input and numerical caveats sent to the owner

The inspected version does not validate that recorded alpha/beta numerically agree with1+softplus(recorded logits). Normally the dense producer supplies this relationship, and the input content hash pins what was read. Hashing input alone, however, does not detect an internally inconsistent triplet. Finite but impossible shape/logit combinations can make the hybrid conversion arbitrarily ill-conditioned; very small sigmoid division or a highly conditioned solve may return nonfinite transformed entries while `usable_chart_mask` still says usable. SPD success alone does not guarantee reliable inversion.

Suggested precise hardening before contract freeze: validate shape/logit consistency with an explicitly documented FP32 arithmetic tolerance, or explicitly require validated upstream triplets and report nonfinite/poor-invariance conversions as withheld numerical credit. Do not clamp shapes, sigmoid factors or inverses to manufacture a result. Tiny-R SD/SE remains descriptive conditional Monte Carlo uncertainty and not a distribution-free confidence guarantee.

No blocking algebra or paired-layout defect was found for the intended valid dense-PG input. The above input/numerical guard is a reliability qualification for malformed or extreme inputs, not evidence that current saved dense results are wrong. No completed validation or candidate-selection claim is made by this static approval.

Inspected hashes (owner may tighten sources afterwards):

- `scripts/audit_sepclip_chart_credit.py`: `48754fe9c11864067fddad7fabe11e6f9b6ab0d0b79709e3e936238f64a81b0f`
- `tests/test_sepclip_chart_credit.py`: `019bbaaca0c16c7d77cce0719edd86751b87198ca471c882de149f622a8a9650`
- dense-PG producer: `1f4053ebe4854063ce19e36b05be37dc3e036d94e368b9b424595c5b98b4c058`

## Final static rereview after owner repairs

The earlier input/numerical caveats above describe the initial reviewed version and have now been addressed. Current source binds recorded shapes to separately evaluated FP32 `1+softplus(recorded logits)` within an explicitly reported two-ULP tolerance, preserves the recorded shapes, and rejects inconsistent replica means. New focused fixtures exercise impossible triplets and the full selected-state denominator under partial withholding.

The final repair checks finite/SPD smooth-logit Fisher BEFORE entering credit transforms, in addition to shape/chart Fisher checks. It records a distinct smooth-logit-Fisher withholding mask and keeps raw shape/chart/logit geometry visible for diagnosis. The accepted one-ULP fixture with recorded logit-700 and shape1+ULP specifically exposes squared-sigmoid underflow in FP64 and now requires withholding rather than a singular solve. Field-specific nonfinite transformed mean/replica coordinates are also withheld, with explicit mask/count and no survivor renormalization.

Read-only inspection found no remaining blocking source defect. Approved for queued numerical contract execution; this is not a test-pass, measured physical-credit, actual candidate, or causal-selection claim. Finite but ill-conditioned solves still require interpreting the reported duality/natural-direction invariant errors. Tiny-replica statistical uncertainty and inherited conditional-objective/sampling-law limits remain substantive.

Final inspected and statically approved hashes:

- `scripts/audit_sepclip_chart_credit.py`: `89d821323c352df5141adf9a1adc6ae350f037202a9bd260a6714f980269fe6c`
- `tests/test_sepclip_chart_credit.py`: `ab789012a21abc4862d3ceaac8bf4ed8ca0c49ecc2f7b5d7e72de16b4fe1f0f3`

No new code changes, model/test executions or submissions were performed by this reviewer.
