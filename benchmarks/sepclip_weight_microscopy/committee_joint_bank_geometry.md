# Joint geometry of fixed optimizer-bank proposals

This appendix interprets the accepted historical layer review analytically. It does not optimize its numerical artifacts, estimate a new physical gradient, run a model, or propose another natural-gradient trainer. The historical inputs are six next-rollout proposals on stored training states, not complete optimizer trajectories or independent physical-return interventions.

## What reinforcing cross terms establish

For each parameter bank i, let u_i(s) be its local actor-logit displacement. On the same states and with the same Beta Fisher, define v_i as the Fisher-whitened displacement, including the state-average normalization. Then:

    Q_joint = 0.5 * norm(sum_i v_i)^2
    Q_diagonal = 0.5 * sum_i norm(v_i)^2
    R = Q_joint / Q_diagonal

The accepted review reports R between 3.34 and 4.53. This establishes substantial NET reinforcement in that state-policy metric. It rules out aggregate layer cancellation as the explanation for these particular proposals. It does not establish that every pair reinforces, that all banks are collinear, or that the resulting displacement improves return.

For K banks, Cauchy-Schwarz gives 0 <= R <= K. Equality R=K requires identical whitened bank displacements; a large R alone does not establish that equality or rank-one structure. With equal bank norms, the average off-diagonal correlation is (R-1)/(K-1). Unequal norms require the actual norm-weighted correlations; substituting an equal-norm formula would be an unsupported measurement claim.

The Gram is positive semidefinite because it is a Gram of v_i. Positive semidefiniteness constrains the TOTAL quadratic; it does not require positive off-diagonal entries. Different matrices can change overlapping action-mean fields even when their ambient parameter coordinates are disjoint. This is a possible duplication of functional response, not duplication of weights or proof that a bank is redundant.

## Effective amplitude, relative to a specified comparison

Relative to the hypothetical quadratic obtained by discarding cross terms, the joint Fisher distance is amplified by sqrt(R), approximately 1.83 to 2.13 for the reported interval. If all fixed bank chords were scaled equally, matching the diagonal-only budget would require scale 1/sqrt(R).

This is NOT evidence that the trainer learning rate is intrinsically twice too large. The comparison discards measured correlations; it is not another deployed policy. Reinforcement may be exactly how several layers implement a useful common action response. Adam also updates banks independently and can produce such reinforcement. A Muon-specific explanation needs matched Adam geometry and reliable physical credit, not this ratio alone.

Nor does a scalar reduction repair direction. It preserves which banks reinforce, their relative weights, and any unfavorable physical pairing. The historical finite KL agrees closely with joint Q, supporting its local amplitude interpretation at those proposals; it does not validate arbitrary new scales or a gait-level conclusion.

## Fixed-bank credit and the joint trust-region comparison

Keep the bank directions fixed. Allow only scalar coefficients a_i. On one common state/objective law, let g_z(s) be an independently supported physical score covector and define:

    H_ij = E[u_i(s)^T * F_z(s) * u_j(s)]
    c_i = E[g_z(s)^T * u_i(s)]
    Q(a) = 0.5 * a^T * H * a
    predicted_gain(a) = c^T * a

The historical Gram entries include the factor 0.5, so H is twice that stored Gram. The actual unscaled proposal corresponds to a=ones when the same chord convention is used.

If H is positive definite and c is nonzero, maximizing linear gain subject to Q(a)<=q gives:

    a_star = sqrt(2*q / (c^T * H^-1 * c)) * H^-1 * c
    maximum_predicted_gain = sqrt(2*q * c^T * H^-1 * c)

The derivation is Cauchy-Schwarz after the coordinate change b=H^(1/2)*a. This is a diagnostic within the tiny span of ALREADY proposed bank directions. It does not select arbitrary network directions, certify the optimizer gradient, or supply a training algorithm.

Ignoring cross terms replaces H by its diagonal and answers a different question. A bank can have positive isolated credit but poor MARGINAL utility alongside other banks. At current coefficients a, the quadratic derivative for bank i is (H*a)_i; its marginal gain is c_i. The ratio between isolated credit and isolated cost misses those interactions.

Unrestricted coefficients can be negative, reversing an optimizer bank. If a proposed intervention permits only nonnegative scaling, coefficients near one, or bounded magnitudes, the unconstrained formula is only an upper comparison. Those restrictions require their own constrained problem and finite policy checks.

## Singular and nearly singular bank geometry

If H is singular and c lies in its range, use H's pseudoinverse in the preceding formula on the retained range. Null coefficients do not change the measured functional direction; the parameter coefficients are not unique even when that direction is unique.

For an exact positive-definite output Fisher on the SAME states, a null combination of u_i has zero pairing with ANY covector on those states. Consequently an exact c cannot have a component in H's exact null space. Apparent null-space credit indicates numerical truncation, inconsistent state/metric conventions, approximation, or a mismatched credit estimate; it is not a free useful direction with zero policy cost.

Nearly null combinations are more dangerous. Large opposing bank coefficients can fit small residual functional directions while making large parameter changes. A local quadratic can then remain small while finite retraction or nonlinear activation changes invalidate the comparison. Numerical inverse thresholds, required parameter norms, independent-credit checks and finite exact KL must be reported together. A noisy estimated c does not turn the algebraic maximum into a confidence bound on true gain.

## Head, trunk, and retraction conventions

The historical largest isolated contributor changes with snapshot: first-block matrices dominate at 4M, whereas a head weight dominates some 8M proposals. The joint bank model therefore cannot assume a universal head bottleneck or one fixed effective learning rate. Readout gains, head orientation and normalized features make each u_i depend on the current full network.

For unit rows/columns, an actual endpoint chord contains a second-order normal component. The historical Gram is built from its ambient Jacobian product. A strictly first-order constrained bank model instead uses tangent-projected chords consistently in BOTH H and c. Mixing projected credit with an ambient Gram, or changing which SVD range is retained between numerator and denominator, is invalid.

Even when coefficients are modest, a new candidate must apply the actual bankwise retraction and evaluate its finite distribution change. Combining linear logit displacements is exact only for the local model. A compensating head/trunk combination can be cheap locally yet alter the function appreciably after a larger finite step or on different visited states.

## Falsifiable comparisons

The following separate competing interpretations without treating R as a verdict:

- **Useful shared response:** reinforcing banks have consistent independently estimated physical credit, and their joint actual chord improves matched closed-loop return. Large R is then productive coordination.
- **Amplitude excess:** the original joint direction has reliable positive physical pairing, but matched smaller finite steps improve return while preserving the direction. A scalar amplitude explanation is supported only by that finite comparison.
- **Bank-allocation mismatch:** reliable c supports a different weighting of the FIXED bank directions at the same true joint cost. Independent-credit and finite-retraction checks must confirm the change; diagonal rankings do not establish it.
- **Locally misleading coordination:** banks reinforce an action field on stored states, but full closed-loop feedback reverses or erases its gain. Local Fisher coherence remains true while the causal return explanation fails.

The accepted historical subset credits are stored-action/GAE pairings. Their varying signs and the reinforcing Gram do not resolve these alternatives. No numerical bank optimization or training selection is performed here. An unresolved physical-credit estimate blocks a claim that the optimizer's coordination is efficient or defective.
