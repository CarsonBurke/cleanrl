# Chart credit interpretation contract

Read-only review checklist for the new saved-dense-field chart diagnostic. No model execution, new job or existing-source change. A coordinate transform changes how a fixed covector is expressed; it does not provide new physical PG evidence.

## Objective/provenance gates

- Pin dense JSON content/source/checkpoint hashes and its selected state mapping. Identify the stored initial logits, actual FP32 alpha/beta, sampled law and replica count. Do not synthesize missing states, replica fields or terminal objectives.
- Keep finite_raw, finite_normalized and bootstrap_normalized as separate fields. All are conditional FIRST-action derivatives at selected states with old continuation policy and frozen selected normalizers; none is the learner's full stored-occupancy covector or all-time closed-loop derivative.
- Transform every state and every replica before reporting aggregate pairings. Coordinate Jacobians depend on that state's shapes/logits. Averaging covectors first and using average shapes is generally invalid.
- For fixed reward std sigma_i and no clipping, raw field at state i is sigma_i times normalized field. Aggregated fields cannot be converted using one average sigma. With clipping, even per-state equality need not hold. Use independently stored raw/normalized fields and reported clip counts.
- A bootstrap-minus-finite difference is the gradient of the added estimated old-critic terminal reward, with first-action scope. Pair the SAME replica before subtraction. It is not true critic error or evidence that the terminal tail is correct. A raw bootstrap field is unavailable unless explicitly recorded; do not create one by an aggregate scale conversion.

## Exact covector chain, with law labels

Original coordinates z_a,z_b use alpha=1+softplus(z_a), beta=1+softplus(z_b). Let s_a=sigmoid(z_a), s_b=sigmoid(z_b). A saved original logit covector g_z transforms to shape covector

g_alpha=g_za/s_a; g_beta=g_zb/s_b.

For mode/excess-precision coordinates r=log((alpha-1)/(beta-1)), ell=log(alpha+beta-2), define A=alpha-1, B=beta-1, tau=A+B and m=A/tau. Their shape Jacobian is

T_shape = [[tau*m*(1-m), A], [-tau*m*(1-m), B]],

with columns r,ell. Consequently

g_r=tau*m*(1-m)*(g_alpha-g_beta),
g_ell=A*g_alpha+B*g_beta.

Equivalently T_z=diag(1/s_a,1/s_b)*T_shape and g_chart=T_z^T*g_z. Verify both routes within the declared arithmetic regime and retain reconstruction residuals. Do not confuse this r/ell chart with physical-mean/total-concentration coordinates: changing ell at fixed r also changes mean because of the shape floor.

The saved score field uses actual FP32 shapes plus the softplus sigmoid chain, i.e. a smooth-score surrogate evaluated at rounded forward values. The prelatent law instead takes A=softplus(z_a), B=softplus(z_b) before adding1. Shapes recomputed from logits in FP64 are a third numerical convention unless explicitly used only as a reference. Label all conventions; do not combine an actual-shape Fisher, intended-latent coordinate Jacobian and an incompatible logit field while calling the result exact.

If actual alpha=1 or beta=1, its excess is zero and finite interior r may not exist. Both1 is uniform, ell=-infinity and r undefined. These are VALID Beta laws but INVALID interior chart rows. Preserve the rows with invalid/null reasons; do not clamp excess, replace their coordinates with prelatent values, drop them silently or interpret their chart efficiency. Sigmoid underflow also invalidates division; report it instead of manufacturing g_shape. Finite but ill-conditioned rows require scale/condition telemetry and actual numerical sensitivity, not just isfinite.

## Statistical gates

- Linear transforms of independent replica fields remain replica fields. A mean chart covector without replica dispersion does not establish useful mode/precision credit.
- Scalar directional pairings against a fixed, independently selected physical motion should report each replica, mean and conditional uncertainty. Dense default TWO replicas have one variance degree of freedom; Student-t95 critical value is about12.706. Closed-loop FOUR replicas have critical3.182, but those are different observations/objectives. Do not transfer their confidence strength to this diagnostic.
- Squared field norm, Fisher dual energy and fitted efficiency are nonlinear and contain noise energy. Preserve signed independent-replica cross-energy and noise differences. Pairs sharing replicas are dependent, not additional sample count. Positive cross-energy alone does not resolve a scalar direction sign.
- A direction selected from a replica must be paired with held-out replicas before calling it useful. State/noise selection, multiple axes and multiple horizons weaken isolated signs; report the full pattern. Conditional replicas cover neither selected-state population nor training-seed uncertainty.
- Compare raw versus normalized, H256 versus H512, and finite versus estimated tail before using a sign to select an algorithm. State/units/tail scope must match before attributing disagreement to original weights.

## Geometry and permissible conclusions

For invertible local T_z and matched smooth-law Fisher, F_chart=T_z^T*F_z*T_z and g_chart=T_z^T*g_z. Exact natural-gradient motion is invariant after mapping back. Therefore a larger chart gradient, smaller coordinate condition number or larger fitted chart energy does not prove an ideal natural chart beats original coordinates.

A practical coordinate-dependent optimizer, finite steps, constraints, numerical arithmetic or changed readout function family can produce different motion. This saved-field diagnostic does not measure those actual motions or their closed-loop benchmark consequences. Report any Cauchy comparison at the SAME achieved joint Fisher cost and retained modes, with uncertainty; do not call a noisy coordinate fit an optimum.

The strongest permissible output is a convention-explicit, state/objective-matched description of resolved or unresolved mode/excess-precision credit. It cannot select causal training, establish a20k optimum, or bypass independent physical-gradient reliability.
