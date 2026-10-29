# Independent effective-readout optimizer-geometry review

Pure mathematical red-team plus scalar-JSON reconciliation. No models, tests, jobs or original-source changes.

## Verdict

No blocking algebra, sign, metric-convention or interpretation defect found. The note correctly labels the equal-rate SGD map hypothetical and keeps actual Adam, physical credit, actor Fisher cost and benchmark conclusions separate.

## Mobility, spectrum and initialization

For effective affine parameters a=rho*w,c=rho*b, with unit w and fixed hidden features, the differential map applied to equal-rate sphere/bias/gain LOSS-gradient descent gives

    delta_a = -eta*[rho^2*P*u+w*(w dot u+b*v)]
    delta_c = -eta*[rho^2*v+b*(w dot u+b*v)]

Hence M=rho^2*diag(P,1)+[w;b]*transpose([w;b]) is correct, including both bias cross blocks and their positive signs. M maps the effective loss COVECTOR into a descent VECTOR; it is a mobility/preconditioner. The gradient metric is inverse(M), where rho!=0. Calling M itself that gradient metric would reverse this convention, which the note avoids.

The tangent eigenvalue rho^2 has multiplicity hidden_width-1. In the orthonormal radial-weight/bias basis, B=[[1,b],[b,rho^2+b^2]], trace1+rho^2+b^2 and determinant rho^2 are correct. Its stated eigenvalues/product and positive definiteness at nonzero rho follow. The full determinant is rho^(2*hidden_width), another check on dimensions and nonsingularity. At rho=0, since w is unit, the remaining outer product actually has rank exactly1; saying at most1 is true but weaker.

At rho=+1 or-1 with b=0, P+w*w^T=I and M=I in effective affine coordinates. Thus the joint head reference step is free-affine SGD to first order, despite separately different chart gradients. Common positive clipping of ALL head covectors multiplies eta and preserves this instantaneous identity. Changing features, finite row retraction, unequal coordinate rates or Adam histories do not follow from that identity. The calculation does not certify a good initial Beta policy.

## Stationarity and normal-force scope

For rho!=0, v=0 follows from the bias equation; P*u=0 makes u radial; the gain equation then forces w dot u=0 and therefore u=0. Joint constrained head stationarity is exactly zero effective-affine loss covector. A nonzero head normal force cannot coexist with stationary gain AND bias in this nonsingular unregularized chart.

The distinction from merely stationary tangent head weights is correct. A radial u can leave gain/bias credit alive. These are raw gradients of one scalar loss, not optimizer-displacement stationarity: lagged moments can move with zero fresh gradient. Direct regularization or parameter-dependent auxiliary losses would require additional terms. Hidden body constraints do not have the same independent row-gain routes, so the head proof cannot be copied to gate/up/down banks.

## Artifact arithmetic

Verified `representation_historical.json` SHA256 and all four reported gain extrema. Squared extrema are:

| Checkpoint | rho^2 minimum | rho^2 maximum |
| --- | ---: | ---: |
| half4M | 4.0856501477 | 8.7029729288 |
| half8M | 5.8849548654 | 13.5222019582 |
| full4M | 3.5348406029 | 6.1245801203 |
| full8M | 4.5912571039 | 9.2783934425 |

These agree with the table's rounding. They characterize only hypothetical tangent mobility in the reference formula, not Adam's realized update or physical conditioning. Without measured b, they do not determine radial/bias eigenvalues. Beta cotangents change with the policy, preventing interpretation as measured universal upstream gradient amplification.

## Actual optimizer limitations

The first-order effective map of an arbitrary ambient weight increment is delta_a=rho*P*delta_w_ambient+w*delta_rho and delta_c=rho*delta_b+b*delta_rho. The note correctly distinguishes this kinematic map from the equal-rate SGD mobility. Elementwise preconditioning before projection can convert normal input into tangent output, and stored moments/denominators prevent assuming intrinsic SGD. This supplies a diagnostic question, not an optimizer ranking or a causal explanation of the remaining score gap.

## Reviewed hashes

- `benchmarks/sepclip_weight_microscopy/analytical_readout_optimizer_metric.md`: `a99090c5643dccca2e2d6ecdb36e57a5f42e072c5e7857c1ed93a2f3c8523468`
- `benchmarks/sepclip_weight_microscopy/representation_historical.json`: `df18acf6ec60be52f3ab8a34488eae7d3330a4cc9c1014950ae4ca42546c75e2`
