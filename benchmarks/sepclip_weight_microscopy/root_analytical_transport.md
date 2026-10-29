# Analytical transport review

This is a derivation from the frozen v7 implementation, not a new experiment or a claim that the performance gap is solved. No TD7 runs are proposed. The purpose is to identify quantities that an optimizer audit must distinguish while measurements are queued.

## PPO produces a masked score covector

For one stored state/action, let r = pi_theta(a|s)/pi_old(a|s), A be the fixed stored advantage, and S_theta = grad_theta log pi_theta(a|s). Away from clipping boundaries, the ascent contribution is A*r*S_theta times an active indicator:

- A > 0: active if r < 1 + epsilon_upper.
- A < 0: active if r > 1 - epsilon_lower.
- A = 0: zero contribution.

The logged clip fraction counts BOTH ratio tails regardless of advantage sign. It is not the fraction of samples whose gradient is zero. A positive-advantage action far below the lower bound still contributes; a negative-advantage action above the upper bound still contributes. Consequently a large logged clip fraction alone does not establish clipping starvation, and a small global norm does not establish weak physical improvement.

The matrix gradient is a sum of score-weighted feature outer products after all network Jacobians. It combines a state-dependent control response with signed credit and clipping selection. Its singular vectors are statistical modes of this sum, not automatically gait modes or independently useful action directions.

For an unclipped importance-weighted expectation, a state-only baseline cancels at any current theta: E_(a~pi_old)[r*score_theta*b(s)] = b(s)*E_(a~pi_theta)[score_theta] = 0, assuming support and integrability. Cancellation is not restricted to the first epoch. Finite samples need not cancel, and the sign-conditioned clipping mask generally destroys the expectation identity because it depends on action and advantage. Future-state critic errors in GAE can also be action-dependent even before clipping, unlike its current-state baseline term.

## Polar preconditioning changes the optimization geometry

For an ideal full-rank gradient G = U diag(sigma) V^T, its polar factor is Q = U V^T. At equal Frobenius update norm, the cosine between G and Q is sum(sigma)/(sqrt(rank)*sqrt(sum(sigma^2))). Equal singular values yield cosine 1; a dominant single singular value with the remaining positive singular values tiny approaches 1/sqrt(rank). An exact rank-deficient partial polar factor needs separate treatment; finite polynomial maps do not necessarily realize that limit.

The polar factor maximizes the linear pairing with G under a spectral-norm constraint. It is not generally the maximizer at fixed policy Fisher cost, nor at fixed physical return perturbation. Thus loss of Euclidean alignment can be an intentional consequence of a different geometry, rather than an algebra defect. Its benefit depends on whether the geometry matches the network and the uncertainty of the estimated gradient modes.

For an approximate map f(sigma), the corresponding pairing is sum(sigma*f(sigma)), and norm is sqrt(sum(f(sigma)^2)). The actual cascade, lane adaptation, momentum and BF16 rounding must be measured rather than replaced with ideal Q when making a performance claim.

## Retraction and preconditioning do not commute

For unit rows w_i, tangent projection is P_W(X)_i = X_i - w_i*(w_i dot X_i). The code computes an ambient optimizer direction D and then normalizes W - eta*D. Its first-order surviving step is -eta*P_W(D). Computing D from P_W(G) would generally give a different direction.

A counterexample makes this distinction exact. Take two unit rows:

W = [[1, 0], [0.6, 0.8]], G = W.

G is the ambient gradient of 0.5*||W||_F^2. That objective is constant on the unit-row constraint, and P_W(G) is exactly zero. Nonetheless, the polar factor of G is

Q = [[3/sqrt(10), -1/sqrt(10)], [1/sqrt(10), 3/sqrt(10)]].

Its projected first row is [0, -1/sqrt(10)] and second row is [-0.8/sqrt(10), 0.6/sqrt(10)]. Both have norm 1/sqrt(10), so the retracted polar step rotates the rows even though the constrained objective has zero gradient. Row normalization cannot remove this induced tangential motion after the polar map.

This proves a possible extrinsic transport effect, not its dominance in PPO. Gate-row radial gradients are not themselves necessarily pure gauge in the unconstrained forward function. At a constrained stationary point, however, radial ambient gradients are normal forces and do not imply a useful constrained descent direction. Per-row scalar rescaling of nonorthogonal rows can change a polar factor's tangent direction.

Adam's elementwise preconditioning can likewise turn radial gradients into tangential motion. The example therefore does not prove Anvil is worse than Adam. It identifies a comparison that must separate raw tangent credit, radial-to-tangent conversion, and the optimizer's remaining tangent transformation at matched functional cost.

Momentum adds another distinction: a direction tangent at a previous W need not be tangent at current W. Projecting stored velocity into the current tangent plane removes its current normal part but is not exact parallel transport. The slow rollout clock changes which clipped gradients accumulate; it does not by itself make momentum intrinsic to the constraint.

For NorMuon specifically, v_k=mu*v_previous+(1-mu)*G_k and the lookahead entering the cascade is q_k=(1-mu)*G_k+mu*v_k. Thus q_k=(1-mu²)*G_k+mu²*v_previous. At mu=.95, the immediate coefficient is .0975; at mu=.85 it is .2775. These are coefficients before nonlinear spectral transformation and lane adaptation, not fractions of the final policy step. The historical .95 long-axis run achieved approximately5907 final100 return, while the newer .85 setting achieved8189; the sources also differ in version, so this is not automatically a strictly isolated momentum intervention without source/parity qualification. Changing Anvil's slow-rail clock also helped modestly. These findings support examining field lag, but do not identify it as the remaining gait limitation or guarantee an improvement from further shortening memory.

## Functional motion is the next test, not matrix norm

For actual parameter chords Delta_theta_l, the first-order logit displacement is u = sum_l J_l*Delta_theta_l. Its local policy cost is 0.5*E[u^T F_z u], which includes cross-layer terms. The existing measured cross terms reinforce rather than cancel. A bank with a small isolated cost can therefore contribute a large joint cost through alignment with other banks.

Comparing per-layer Frobenius norm or isolated Fisher cost is insufficient. Pairing with a reliable held-out physical covector must use the SAME joint functional direction and the SAME retained subspace when a Jacobian SVD truncates modes. Finite difference agreement between the actual real chord and JVP is required before treating the local prediction as a finite update explanation.

## Credit, transport and feedback are separate causal links

For a deterministic mean/mode action law a_theta(x), closed-loop dynamics are x_next = f(x, a_theta(x)). Linearized state feedback is f_x + f_a*J_a,x. A parameter direction injects f_a*J_a,theta*Delta_theta, then future effects propagate through those closed-loop Jacobians. For sampled Beta actions this formula is conditional on a fixed exogenous-noise realization, a_theta(x, epsilon); the mean-policy Jacobian is not the full stochastic feedback operator. Expected sampled return needs the score or paired stochastic trajectory diagnostics. A small action change may damp a useful stride response or disrupt contact timing; the immediate torque response is not its full return effect.

Full-trajectory score derivatives and paired exact old/first-update/ten-epoch policies address this distinction. A first-action conditional oracle addresses only that conditional intervention. Multiplying its effect by horizon cannot recover the full policy gradient.

GAE versus finite Monte Carlo with the SAME fixed terminal critic tests their difference under that specified terminal objective. It does not independently certify the critic tail as correct. A discrepancy between a finite update and its derivative also combines curvature, feedback and sampling error; it is not automatically critic bias.

The analytical decision order is therefore: resolve independent physical-gradient signal; establish the target objective and tail; compare stored PG with that covector; compare actual retracted optimizer motion with supported PG at joint functional cost; validate finite chord linearization; finally measure closed-loop speed/cost effects. An unresolved earlier link prevents attributing a downstream efficiency score to optimizer failure.

## What current evidence does and does not establish

The strongest frozen sepclip mode policy reaches approximately 10.58 m/s and 10,207 return. Its direct control penalty is about 373 return, so removing that penalty cannot supply the remaining approximately 9,793 points. Sampling explains roughly one thousand points, not the full gap. The issue therefore concerns the learned motion and its physical feedback, beyond direct noise cost.

Existing TD7 logs demonstrate much stronger local scalar performance but lack saved actors and normalizers. They cannot provide weight-level or gait-level evidence. A saved longer-trained PPO policy around 14.5k uses the same 1+softplus Beta shape law but a different pre-RMS residual trunk and unconstrained head. If compatibility validates, it can provide a stronger physical comparator; it cannot isolate architecture, optimizer, or training duration by score alone.

No analytical result here establishes a 20k optimum, a single primary bottleneck, or an improvement from a proposed coordinate change. Those conclusions require the already planned causal measurements and evidence-based intervention.

## Homogeneity and the branch normalization nullspace

For b(x) nonzero, the normalized branch y(x)=b(x)/||b(x)|| obeys J_y*x = (I-y*y^T)*J_b*x/||b||. If b is homogeneous of degree k, Euler's identity gives J_b*x=k*b, and the input-radial derivative of y is exactly zero. This follows for any homogeneous degree, not just the near-origin quadratic approximation.

For the bias-free SiTU branch, near zero f(g,u) is approximately 0.5*g*u, making b approximately degree two. Gate/up saturation and sigmoid transitions break homogeneity and can restore radial input information. Therefore the architecture places radial amplitude information in its nonhomogeneous responses, while directional information survives directly. This is a precise architectural inductive bias, not proof of amplitude blindness on visited states. Current visited gate statistics already reject a blanket near-origin approximation.

A global radial direction in normalized observation space is also not a pure forward-velocity change: it simultaneously alters joint positions and other velocities. Its suppression cannot be equated to missing speed feedback without measuring physically meaningful state perturbations. An additive identity stream can retain amplitude differently, but a stronger score from a different longer-trained model does not identify that as the cause.

There is an additional exact residual geometry in this actor: both h and the second branch b are unit directions, and the final feature is y=normalize(h+b). For angle phi between them, ||h+b||=2*cos(phi/2) and y has angle phi/2 to each direction when their sum is nonzero. Thus the residual is an angular bisector, not an amplitude-weighted correction. Scaling the second branch's raw output does not smoothly turn its contribution off after its own normalization. The network must change its direction (for example align it with h) to obtain an effectively small correction. The final normalization Jacobian includes 1/||h+b|| and can amplify perturbations when the directions nearly oppose. No current artifact here establishes that near-opposition occurs often or that this architecture prevents a faster gait; it identifies a specific difference from an additive pre-RMS identity stream.

## Actual candidate coordinates must match selection evidence

The v13 proposal uses r=log((alpha-1)/(beta-1)) and ell=log(alpha+beta-2). Let tau=exp(ell), m=sigmoid(r). Its native action mean is mu=(1+tau*m)/(2+tau). At fixed r, dmu/dell=tau*(2*m-1)/(2+tau)^2: excess-precision motion changes mean. A mean/total-concentration decomposition is therefore not the candidate's mode/excess-precision decomposition. Covectors and Fisher costs must transform into the actual candidate coordinates before evidence can select that intervention. Preserving the initial policy and hidden Jacobian does not preserve its later readout family.

Independent review verified the 2x2 polar algebra and emphasized the sampled-feedback distinction. Positive cross-replica energy alone is insufficient to resolve physical gradient signal with four replicas; actual direction intervals, independent held-out pairings, threshold sensitivity and finite-chord validity remain necessary guards.

## What a Beta policy gradient measures physically

For fixed state, a fixed continuation policy and a native action U~Beta(alpha,beta), define Q(U) as the conditional expected return after that action. For integrable Q independent of alpha/beta within this conditional intervention,

grad_alpha E[Q] = Cov(Q(U), log(U)); grad_beta E[Q] = Cov(Q(U), log(1-U)).

The digamma terms in the score subtract the expectation of the log action. The logit covector then multiplies these covariances by sigmoid(z_alpha) and sigmoid(z_beta). This makes the specific distinction from an action-value actor clear: PPO estimates return/log-action covariance, whereas a deterministic Q actor receives an explicit torque derivative through its learned Q. Neither estimator is automatically more accurate or universally superior.

If Q is smooth over a narrow action distribution, E[Q] is approximately Q(mu)+0.5*Q''(mu)*Var(U). The two shape gradients consequently mix changing the mean with changing the variance; they are not two pure torque commands. In native coordinates dmu/dalpha=beta/kappa^2 and dmu/dbeta=-alpha/kappa^2. Physical mean derivatives multiply these by two. In an asymmetric near-bound state with beta close to one and alpha large, increasing alpha gives a physical-mean derivative near 2/alpha^2. Lowering beta is restricted by the shape floor and its softplus derivative also approaches zero. This is a concrete conditioning issue in original logit coordinates, not a fixed action or gait capacity ceiling.

The narrow-distribution expansion is not reliable across discontinuous contact responses, strongly curved Q, or broad action noise. There the exact covariance formulation remains applicable, while a local torque derivative alone may miss the expected-return consequence of noise and contact transitions. Nor does a finite first-action covariance substitute for the all-time closed-loop policy derivative.

An invertible output coordinate change by itself cannot improve an exact natural-gradient step at first order: g_c=T^T*g_z, F_c=T^T*F_z*T and T*F_c^(-1)*g_c=F_z^(-1)*g_z. Any observed benefit must therefore arise from the actual coordinate-dependent optimizer, constraints, finite steps, approximation, or changed reachable function family. A chart can improve Adam's numerical conditioning without establishing a new physical ascent direction. This is why the chart proposal needs matched actual-gradient/metric analysis rather than a large coordinate gradient as evidence.

## Matrix spectrum contains signal and sampling noise together

Write a stochastic matrix gradient as G_hat=G_signal+E. For the exact square nonsingular polar map, small perturbations of its rotational part involve denominators sigma_i+sigma_j; rectangular off-subspace components can involve sigma_i. Small singular modes can therefore have unstable orientation. Flattening the spectrum does not distinguish a weak useful control mode from an uncertain noise mode. The finite polynomial cascade modifies this sensitivity and must not be assumed to be the exact polar map.

In the scale-dominated regime, the cascade approximately removes overall gradient scale before its polynomial maps. The actual implementation has additive 1e-6 normalization epsilon, BF16 rounding and second-moment floors, so exact positive-scale invariance and a discontinuity at zero must NOT be claimed for this port. Nevertheless, above the epsilon regime, substantially reducing gradient magnitude need not proportionally reduce the matrix proposal norm. This separates three quantities: estimated gradient magnitude, direction reliability, and actual policy displacement. A small PPO gradient cannot certify a small or safe matrix step.

The earlier noise-filtering and fractional-mode experiments do not establish this as the primary score limitation: they did not beat the better controls. Their negative outcomes reject those particular remedies; they do not logically demonstrate that every current gradient mode is informative. Conversely, historical noisy physical-oracle replicas cannot be used to condemn the optimizer until their direction uncertainty is resolved.

## Causal reading rules from the independent PPO review

| Measurement combination | What it supports | Remaining limitation |
|---|---|---|
| Paired GAE-minus-augmented-MC directional difference excludes zero | Credit transport differs for the specified finite frozen-critic-tail objective | Not proof of true infinite-horizon critic bias |
| Direction changes with H256/H512 or zero/critic tail | Attribution is tail-sensitive | Longer horizon is not automatically ground truth |
| Independent directional gains change sign or intervals contain zero | Physical-gradient direction unresolved | Cannot diagnose optimizer or representation failure |
| Reliable physical direction is reachable, but actual motion has negative held-out pairing at matched joint cost | Local optimizer transport loses useful direction | Requires finite closed-loop validation and does not predict benchmark superiority |
| Independently useful field remains outside stable retained Jacobian ranges | Local tangent representation restriction for these states | Not global capacity failure or a rank-only conclusion |
| Full-trajectory derivative positive but finite closed-loop gain negative | Local prediction fails for the finite step | Curvature, feedback and noise are not automatically critic bias |
| First update helps while full update hurts | The later update trajectory is implicated | Clipping, credit and optimizer effects still need separation |

With four independent noise replicas, a conventional approximate 95% scalar interval uses mean plus/minus 3.182*standard_error. Pairwise cross-energies share replicas and cannot be treated as six independent samples. These are conditional intervals for fixed selected states, not training-seed or full state-population uncertainty. Searching many modes and horizons requires care with isolated significance claims. At gamma=.99, gamma^256 is about .0763 and gamma^512 about .00582; these are discount factors, not numerical bounds on omitted tail bias.

## Surrogate occupancy is not a full-trajectory objective

The first-epoch learner gradient is an empirical average of score*GAE over uniformly weighted stored rollout states; it does not attach an outer gamma^t to each stored row. The all-time closed-loop diagnostic instead differentiates expected discounted return from its specified fixed initial-state distribution. Its score contributions are weighted by gamma^t and its own propagated occupancy.

For an infinite discounted objective J_rho, the policy-gradient theorem uses discounted occupancy d_(rho,gamma), yielding grad J_rho = (1/(1-gamma))*E_d[score*Q_pi]. Replacing this occupancy with the empirical stored-state distribution creates a different surrogate gradient unless matching assumptions are established. For finite horizons and terminal tails the exact occupancy and remaining-horizon Q also change with trajectory time.

The dense conditional first-action oracle at actual stored states aligns its initial occupancy with the PPO surrogate but intervenes only at that action. The full closed-loop oracle changes all actions and assesses the physical effectiveness of an actual policy step, but is not automatically the same gradient target as the training surrogate. Both are useful precisely because they address different links. Their disagreement alone is not critic bias, and the shared gamma=.99 in strong local TD7 configurations already prevents identifying discount alone as a unique explanation of the score gap.
