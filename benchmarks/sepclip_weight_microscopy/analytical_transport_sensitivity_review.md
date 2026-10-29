# Independent transport-sensitivity review

Read-only analytical review and scalar-JSON recomputation. No model/test execution, queue submission or original-source change.

## Algebra and domains

The two-row full-rank square polar factor, tangent projections, retracted descent signs and correlation derivative are correct. With c=.6,s=.8, 2*c*s/sqrt(2*(1+s))=.5059644256. Reversing G reverses the first-order tangent motion. The constrained norm objective has zero tangent gradient, so this is a legitimate normal-force-to-tangent witness, not evidence that actual PPO normal forces have these signs.

One terminology correction is recommended: for negative c, dot product becomes more negative and the ordinary angle between the two rows INCREASES toward pi. Thus “angular diversity shrinks” is ambiguous. Their absolute correlation increases and directional linear independence/Gram conditioning deteriorates for either sign of c. For two rows the Gram eigenvalues are1+|c| and1-|c|, which makes that exact interpretation explicit. This does not imply all bank feature arrangements collapse in the same manner.

The tall formula Q=Lambda*W*(transpose(W)*Lambda^2*W)^(-1/2) is correct for real diagonal Lambda and full COLUMN rank of Lambda*W. The weighted Gram must be positive definite. Isotropy makes each surviving row radial; anisotropy only generically induces tangency, not necessarily for rows aligned with eigendirections. The wide full-ROW-rank formula is the transpose analogue Q=(Lambda*W*transpose(W)*Lambda)^(-1/2)*Lambda*W. For down-column constraints the constraint/projection axis also must be transposed; do not transfer a row example without that change.

For square nonsingular G=Q*H with H positive definite, the Sylvester equation Omega*H+H*Omega=A-transpose(A), A=transpose(Q)*deltaG, and deltaQ=Q*Omega are correct. The entries with denominators sigma_i+sigma_j are in the eigenbasis of H, with A transformed to that same basis. The tall full-column-rank derivative additionally contains (I-Q*transpose(Q))*deltaG*inverse(H). For wide full-row-rank matrices apply the transposed result. Rank-deficient polar completions can be nonunique; even a uniquely specified partial factor needs separate rank-stratum/domain treatment. No derivative bound here should be extrapolated across a rank change.

The persistent-normal-gradient Adam example is correct under the explicitly stated steady-state, negligible-epsilon, fixed-gradient assumptions: g=lambda*(.8,.6) gives normalized direction approximately(1,1), tangent(-.12,.16), descent displacement eta*(.12,-.16). As the row rotates, its new gradient and stored moments change, so this is a local/asymptotic direction witness, not an exact sustained trajectory or optimizer-ranking result.

## Scalar reconciliation

Recomputed projected-input versus actual-bank ratios from `layer_historical.json` using stored-subset first-order gains and Fisher quadratics. All six4M rows and the quoted8M first-up/second-down ratios agree to stated precision. The8M values are1.249322 and.801882. The tiny legacy FP32 exact KL is not used in these ratios.

The counterfactual projects both current gradient and stored rails before their current-point blend, adapts using a newly computed lane statistic, then ALSO projects the final adapted direction before finite retraction. The final-only comparison addresses the secondary finite-retraction difference; it does not isolate gradient projection from history projection. Projecting rails at one base point is current-tangent projection, not exact parallel transport along the historical weight path. Normal proposal fractions cannot bound the effect of changing the preconditioner inputs.

All reported ratios remain stored-surrogate, isolated-bank, first-order functional measurements. Joint Fisher cross terms and independently resolved physical credit are required before selecting an intervention.

## Singular-mode coefficients are not a complete decomposition

The audit SVD is of the ambient momentum LOOKAHEAD G, not the adapted proposal P and not the physical-PG operator. It records c_i=inner(ui*transpose(vi),P) only for selected leading/trailing indices. These are valid Frobenius projection coefficients onto selected diagonal dyads of the lookahead basis.

Even retaining every diagonal dyad would generally not reconstruct P. Before lane adaptation, an exact orthogonally equivariant spectral map would stay in those diagonal modes in exact arithmetic. The actual row/column lane scaling is orientation dependent and mixes them. Square banks can acquire off-diagonal coefficients C_ij=transpose(ui)*P*vj. Tall banks can additionally acquire components outside the thin lookahead U column space; wide banks can acquire components outside the thin V row space. BF16 cascade/adaptation rounding, approximate spectral arithmetic and FP32 weight-subtraction quantization add further residual. Momentum means singular values and raw clipped-gradient coefficients refer to different quantities even before that mixing.

For a complete Frobenius audit, compute the full basis coefficients and/or P_diag=sum_i c_i*ui*transpose(vi), and report residual energy ||P-P_diag||_F^2=||P||_F^2-sum_i c_i^2, with selected and all-diagonal energies distinguished. This is a suggested diagnostic contract, not a new job or modification.

The current per-mode metric retracts each isolated coefficient dyad separately at the same base weight. Finite retraction is nonlinear: those isolated parameter deltas need not sum to the actual joint retracted delta. Their Fisher costs also have cross terms and cannot be added. They therefore describe selected isolated modal counterfactuals, NOT an exhaustive attribution of proposal norm, functional KL, surrogate gain, or return.

## Evidence interpretation

The note correctly separates proof of a possible transport effect from measured optimizer failure. Resolving independent physical credit, comparing the joint realized functional chord at matched cost, checking finite-step fidelity and deploying it through closed-loop physics remain necessary. No source algebra or stored-subset coefficient proves that radial preprojection will improve benchmarks.

## Reviewed hashes

- `benchmarks/sepclip_weight_microscopy/analytical_transport_sensitivity.md`: `eb22c2a76b0d3d316497cd15b1cdf40b70d1c3222cee3f5a9150c25249fd7f68`
- `scripts/audit_sepclip_layer_transport.py`: `23fb1162c8c9c26d14320e4b744b9667feef29d5a143bf38d7eee6fccdb87b0e`
- `cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py`: `07f5f9636103dd5fdc9896de3eb781078df8bb1cb9b3ad87bcb79907fc619c8d`
- `benchmarks/sepclip_weight_microscopy/layer_historical.json`: `ba19fa734771e4eb15378403876f0599caf5b081bc95cb368a9218014e387d80`
