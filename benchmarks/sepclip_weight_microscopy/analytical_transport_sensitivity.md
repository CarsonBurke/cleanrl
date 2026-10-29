# Normal forces can alter feature linear independence

This note extends the verified pure-radial example in root_analytical_transport.md. It is hand algebra and arithmetic from existing scalar artifacts, not a new model measurement. Ideal polar identities are not substituted for the implemented finite cascade, lane adaptation, BF16 arithmetic, or momentum.

## The direction of the spurious motion matters

Take two unit rows W=[(1,0),(c,s)], with s=sqrt(1-c*c)>0, and ambient descent gradient G=W. The constrained gradient is zero because this is the gradient of a constant norm objective on the product of unit-row spheres. Its unique square polar factor is:

    Q = [[1+s,-c],[c,1+s]] / sqrt(2*(1+s))

Let d=sqrt(2*(1+s)). After retraction, the first-order descent displacement is -eta*P_W(Q). Therefore:

    delta_w1 = eta*[0,c/d]
    delta_w2 = eta*[c*s/d,-c*c/d]
    delta(w1·w2) = 2*eta*c*s/d

For c=.6,s=.8, the dot product increases by about .505964*eta, despite zero constrained loss gradient. Positive row correlations grow and negative row correlations become more negative: absolute correlation grows and Gram conditioning deteriorates in this particular positive-normal-force example. Negative correlations correspond to increasing ordinary angular separation, so this is a claim about linear independence, not uniformly shrinking angles. For G=-W the first-order sign reverses. Thus normal-to-tangent conversion is not merely discarded update energy; it can create a sign-dependent bias in the arrangement of features. Nothing here proves that PPO's actual normal multipliers have the required signs or that its banks collapse this way.

For full-column-rank tall G=Lambda*W, the exact polar factor is Lambda*W*(W^T*Lambda^2*W)^(-1/2). If that weighted Gram is isotropic, each row remains radial and its tangent projection vanishes. If it is anisotropic, a generic row is not an eigenvector of the inverse square root and tangent motion can remain. For a wide full-row-rank bank, apply the transposed formula. Orthogonal rows with diagonal normal multipliers provide a special case where the ideal polar map remains radial. Learning can change that special structure; row normalization preserves lengths, not pairwise orthogonality.

## Small normal norm is not a general bound on preconditioner distortion

For square full-rank G=Q*H, H positive definite, the differential polar rotation obeys:

    A = Q^T*delta_G
    Omega*H + H*Omega = A-A^T
    delta_Q = Q*Omega

In the singular basis, Omega_ij=(A_ij-A_ji)/(sigma_i+sigma_j). Weak singular pairs can therefore be sensitive to a perturbation that is modest in Frobenius norm. Tall full-column-rank maps additionally have the term (I-Q*Q^T)*delta_G*H^(-1), which can be sensitive to small sigma_j outside the old column space. Rank deficiency requires separate, potentially nonunique treatment.

This means a small final proposal radial fraction does not establish that removing radial components BEFORE preconditioning would have little effect. Conversely, it does not establish that this alternative has a better physical direction. The implemented polynomial map and momentum can regularize or change this sensitivity, and functional Jacobians can suppress or amplify the resulting motion.

Adam has its own extrinsic bias. For one unit row w=(.8,.6), a persistent positive normal gradient g=lambda*w gives a steady elementwise-normalized Adam direction approximately(1,1), ignoring epsilon and bias-correction transients. Its retracted descent tangent is eta*(.12,-.16): the larger coordinate increases and the smaller decreases. This exhibits coordinate preference even though the constrained norm objective is constant. It does not show an advantage for either optimizer.

## Existing projection counterfactuals are mixed after accounting for cost

The historical layer audit includes isolated-bank variants that project the current gradient AND stored momentum rails before the polar/cascade path. These simultaneously alter credit input and history, so they do not isolate moment transport. They also tangent-project the final lane-adapted proposal before retraction; projecting stored rails into the current tangent is not historical parallel transport. They also change policy cost. For v7-half4M, ratios against the actual ambient-then-retract bank are:

| Bank | Stored-subset gain ratio | Isolated Fisher-cost ratio | Ratio of gain / sqrt(cost) |
|---|---:|---:|---:|
| first gate | 1.171 | 1.458 | 0.970 |
| first up | 1.288 | 1.321 | 1.120 |
| first down | 1.123 | 1.161 | 1.042 |
| second down | 1.279 | 1.639 | 0.999 |
| second gate | 1.059 | 1.210 | 0.963 |
| second up | 1.162 | 1.239 | 1.044 |

These ratios are scalar calculations from layer_historical.json; Fisher quadratic costs are used, not its superseded tiny FP32 exact KL. At8M, first-up's corresponding gain/sqrt(cost) ratio is about1.249, while second-down's is about.802. A larger raw surrogate gain is therefore not a uniform equal-functional-budget improvement.

The table is not a physical oracle or a confidence estimate. It uses a subset of stored advantages, separate bank directions, and first-order Fisher costs. The strong measured joint cross terms prevent adding these isolated budgets to infer a joint-policy intervention. The relevant missing identification is: which part of supported physical credit enters the optimizer, how normal/tangent credit and old moments alter the JOINT functional chord, and whether its finite closed-loop effect is useful.

## Falsifiable implications rather than a selected optimizer fix

If extrinsic normal conversion is a material source of poor updates, projecting only the FINAL direction should have little first-order effect while changing preconditioner inputs can rotate functional motion; the historical final-only counterfactual is indeed near the actual update. That is consistent with the algebra, not a causal performance result.

To attribute failure to this mechanism, normal multipliers or their moment descendants must generate a functionally consequential direction poorly paired with independently supported physical credit, after matching joint policy cost and checking finite retraction. If a supposedly intrinsic update improves only raw stored gain while spending more policy cost, or if physical credit is unresolved, the interpretation fails. The current isolated-bank results do not select tangent projection as a universal cure.
