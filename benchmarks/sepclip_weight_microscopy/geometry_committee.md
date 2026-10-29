# Geometry committee: precise model and falsification criteria

Status: source-derived claims and predictions only; no GPU experiment was executed by this committee. Checkpoint quantitative results must be attached by the integrated runner. Frozen source audited: `cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py` and `cleanrl/shared/host_actor.py`. Prior numerical evidence read from `benchmarks/sepclip_mechanisms/investigation_report.json`.

## 1. The constrained model actually trained

Every gate and up matrix has unit rows; every down matrix has unit columns; actor and critic head matrices have unit rows. Parameters therefore live on a product of spheres, together with unconstrained head biases and readout gains. Projection happens after both optimizers step. It is not a mere numerical stabilization: it removes functional radial degrees of freedom.

For one branch, write g=Gx, u=Ux, c_g=4 tanh(g/4), c_u=25 tanh(u/25), a(g)=c_g sigmoid(g), f=a(g)c_u, and y=Df. Its exact differential is:

    dg = dG x + G dx
    du = dU x + U dx
    df = c_u a'(g) dg + a(g) sech²(u/25) du
    a'(g) = sech²(g/4) sigmoid(g) + c_g sigmoid(g)(1-sigmoid(g))
    dy = dD f + D df

For normalized h=y/||y|| away from zero, dh=(I-h hᵀ)dy/||y||. The second branch receives 8h, so its differential includes that factor and its own gate derivatives and normalization projector. If b is its unit output and q=(h+b)/||h+b||, dq=(I-q qᵀ)(dh+db)/||h+b||. These factors are not captured by singular values of an isolated matrix gradient.

The actual head equation is z_i = gain_i * (8 w_i·q + bias_i). The gain multiplies the bias as well as the matrix output. Its differential is:

    dz_i = (8 w_i·q + bias_i) d_gain_i
           + gain_i (8 dw_i·q + 8 w_i·dq + d_bias_i).

Consequently head, gain, and hidden changes can reinforce or cancel. Summing separate layer KL costs loses these cross terms. This is an exact architectural statement, not a proposed cause of low benchmark return.

## 2. Initialization changes before any learning

The actor head's orthogonal initialization has scale 0.01; immediate row normalization removes it. Gain starts at one, so the final hidden norm of eight gives |z_i|<=8 at initialization, rather than the intended scale's |z_i|<=0.08. This is exactly a factor 100 in head logit scale with the same hidden features. RMS still requires measurement on actual observations; a hard bound is not an RMS estimate. Critic head initialization already has scale one.

The shared branch initializer aims for gate/up marginal variance two. Its 43x64 wide matrix has orthogonal rows of norm sqrt(2). For the 43x17 tall first stage, orthogonal columns and the shape correction instead make the *average squared* row norm two; individual row norms depend on random leverage. Immediate normalization makes every row norm one, additionally reweighting first-stage rows relative to one another. With isotropic unit-variance inputs every marginal variance is then one. Actual observations are neither independent nor isotropic, so the visited preactivation covariance must be measured. The down initializer's common positive scale cancels in the subsequent branch output normalization.

A corrected small head gain is a falsifiable architecture/initialization intervention, not an optimizer fix. With gain=0.01, upstream hidden gradients initially shrink approximately 100-fold for a fixed score cotangent, while head-gain gradients remain of order one. Beta score/Fisher changes invalidate a blanket 100-fold gradient assertion. An intervention must therefore measure gain growth, upstream gradients, and initial action distributions rather than assume restoring a conventional initializer is beneficial.

## 3. Exact finite retraction, and what it removes

For one unit row or column w and a descent proposal d, split d=r w+t where r=w·d and w·t=0. The actual retraction is:

    w_new = ((1-eta*r)w - eta*t)
            / sqrt((1-eta*r)² + eta²||t||²).

When 1-eta*r>0, the angular step satisfies tan(theta)=eta||t||/(1-eta*r). First order gives delta_w=-eta*t. Second order gives:

    delta_w = -eta*t - eta²*r*t - (eta²/2)||t||²*w + O(eta³||d||³).

Thus radial proposal energy is discarded at first order, but radial proposal components also alter the effective tangent step at second order. A comparison of ambient-retract against final-project-retract should be small whenever eta times the maximum individual row/column proposal norm is small. A large result at that scale suggests an implementation or scaling issue, not automatically profound geometry.

For an ideal unit-singular-value polar direction before lane adaptation, early eta=0.0025 produces characteristic average angular budgets about 0.0025 for first gate/up (43x17) and second gate/up (43x64), and 0.00305 for down (64x43), before tangent loss. These follow from the implemented sqrt(max(1,out/in)) multiplier and Frobenius norms; lane adaptation, approximate cascade, row concentration, and actual retraction must be measured. At the 4M checkpoint linear LR annealing approximately halves those budgets. These small one-step angles predict that final projection alone will usually have little effect. Pre-polar gradient projection can have a much larger directional effect.

The constrained loss gradient is P_w(g)=g-w(w·g), not the ambient g. An ambient polar optimizer maximizes a pairing under an ambient spectral constraint; its optimization problem differs from the constrained, functionally weighted problem. Spectral flattening and tangent projection do not commute. Moreover lane adaptation is orientation-dependent and is followed by global Frobenius renormalization. It is not an exact polar direction after adaptation.

A simple one-vector witness quantifies the issue: with w=(1,0) and g=(r,t), normalized ambient descent has useful tangent magnitude |t|/sqrt(r²+t²), whereas normalizing the tangent gradient gives unit useful magnitude. At |r|=3|t|, only 31.6% of the ambient direction survives. This example demonstrates a possible inefficiency; it does not prove that actual layer gradients have that ratio or that a larger tangent motion improves return.

## 4. Radial directions are not uniformly gauges

Uniform positive scaling of the entire down matrix is an exact gauge of its normalized branch output. Individual down-column scaling changes relative channel contributions and generally changes the branch direction. Uniform up scaling is only approximately a gauge when u/25 is small, because tanh(u/25) is then nearly linear. Individual up-row scaling is functional even in that limit. Gate scaling changes both sigmoid(g) and tanh(g/4); it is generally functional, including global scaling.

Radial fractions alone therefore cannot distinguish wasted optimization from a representational constraint. Required decomposition:

1. Split the raw gradient and optimizer proposal into row/column radial and tangent components.
2. Push each through the actor Jacobian on visited states.
3. Compute score pairing, Beta Fisher energy, and radial/tangent cross terms.
4. Probe common down scaling separately to verify the exact gauge numerically.

A large radial gradient with zero functional radial JVP is a coordinate gauge. A large radial gradient with large score-aligned radial JVP indicates a useful degree of freedom the architecture forbids. No optimizer confined to those spheres can recover it; this would support an explicitly learned row/channel scale intervention, conditioned on physical action-oracle agreement.

## 5. Momentum transport: necessary distinctions

Current code stores ambient Euclidean gradient EMAs without any post-retraction modification. This is a valid extrinsic optimizer, not automatically a bug. A manifold optimizer could instead store tangent cotangents/vectors, project each gradient before spectral transformation, project the resulting direction again, retract, and transport its moments.

On a unit sphere, exact shortest-geodesic transport from w to w_new of a tangent vector v is:

    PT(v) = v - [v·w_new / (1+w·w_new)] (w+w_new).

Applying P_w to an old ambient EMA is current-tangent projection, not exact parallel transport. The checkpoint lacks the historical weights necessary to reconstruct a truly transported EMA. A frozen replay contrast that projects stored rails must be labeled accordingly. Fresh instrumented traces can measure exact transport versus repeated projection and preserve the distinction.

Slow rail uses beta_slow=0.98^10=0.8170728 once at the first fresh rollout gradient, then is held for the remaining nine optimizer updates. Its mean age is beta/(1-beta)=4.466 rollouts; half-life is approximately 3.431 rollouts. This matches decay memory in epoch units but replaces ten changing full-batch gradients with one beginning-of-rollout gradient. V7 improvement therefore does not isolate transport, nor prove that stale policy gradients dominate. Late fast rail beta=0.85 has mean age 5.667 optimizer steps, while its blend weight is 0.4385; slow rail contributes 0.5615. Lookahead is separately scheduled and must not be confused with fast-rail beta.

Typical one-step angular motion is small; transport failure should therefore be judged against observed accumulated motion over rail memory. A high radial stored-rail fraction can arise from functional radial input gradients rather than from the weights rotating under stale tangent moments. Distinguish those by instrumented traces before committing a transport intervention.

## 6. The metric the policy actually sees

For each independent native Beta dimension, alpha=1+softplus(z_a), beta=1+softplus(z_b). The logit score is sigmoid(z_a)[log(x)-digamma(alpha)+digamma(alpha+beta)] and the corresponding beta term uses log(1-x). The logit Fisher is the trigamma Beta Fisher sandwiched by diag(sigmoid(z_a),sigmoid(z_b)). It includes an off-diagonal term -trigamma(alpha+beta).

For parameter displacement delta and logit JVP u=J delta, local KL is (1/2) E[uᵀ F_z u]. For layers i,j, keep K_ij=(1/2)E[u_iᵀ F_z u_j], including signs. Joint local KL is the sum over all i,j. Useful first-order surrogate gain is E[A score_z·u]. For a physical-oracle advantage A_oracle, replace A with that oracle rather than assume the stored GAE is correct.

The normalized efficiency E[A score_z·u]/sqrt(E[uᵀF_z u]) is invariant to global step rescaling and tests direction quality. It still is a local score-weighted quantity on a chosen state/action distribution, not a return guarantee. If a claimed transport change improves only stored-GAE efficiency but worsens physical-oracle pairing, it optimizes a biased estimator better and should be rejected as a performance explanation.

## 7. Predictions and intervention decision gates

Prediction G1: actor-head initialization logit scale changes exactly 100-fold for identical hidden features when head row normalization is applied; all-matrix normalization adds a separate feature change. Reject a head-scale explanation of persistent sub-10k returns if matched gain initialization changes transient metrics but converges to the same physical gait and score.

Prediction G2: final-project-retract versus ambient-retract differs at second order in small per-row/column step size. Pre-polar projection can substantially alter direction. Reject a geometry explanation based solely on large discarded norm if functional tangent score efficiency and exact finite KL remain unchanged.

Prediction G3: uniform down scaling has near-zero actor JVP away from numerical epsilon; row/column-specific radial directions need not. If gate radial score energy is large and agrees with physical-oracle PG, a constrained-scale architecture intervention is more directly motivated than momentum transport.

Prediction G4: current-tangent projection of rails will outperform ambient rails locally only when surviving tangent score pairing improves at matched joint Fisher cost. Exact transport needs fresh traces. A improvement in raw gradient cosine or singular spectrum alone does not pass this gate.

Prediction G5: hidden matrix and Adam head/gain/bias changes can have non-negligible joint Fisher cross terms. Scalar matrix-only calibration can therefore miss the effective trust region even when locally accurate. This is compatible with V11's observed matched matrix budget and worse fresh benchmark; it does not establish cross terms as the cause without measurements.

Targeted causal test, conditional on G3/G4 evidence: choose one supported mechanism, then run architecture-matched Adam and Anvil controls with identical seed/protocol and full 8M training. Do not conflate a new architecture with an optimizer advantage. Use physical-oracle score pairing and closed-loop sampled/mean/mode behavior to connect the local effect to return. Do not promise 20k from changing a matrix metric.

## 8. Existing evidence limits

Selected V7 primary return is 8781.57 versus fresh Adam 8308.73, +5.69%. V7 half-rate is 8632.02. V11 matched-control entire training curve is an exact deterministic reproduction, not independent evidence. V11 fractional direction with locally matched matrix Fisher budget reached 8074.10, below its 8632.02 control. Natural-head and sample-Fisher head variants also regressed against stock Adam. These results reject simple promises that singular flattening, head curvature, or scalar policy calibration alone unlocks the missing performance.

A complete explanation still needs (a) unbiased or explicitly horizon-limited physical PG evidence, (b) sampled/mean/mode return and gait measurements under correctly restored row-wise observation normalization, and (c) a fresh causal intervention that changes the identified bottleneck. Without those, geometry quantifies the learner's motion but does not establish the actual optimum or the reason for the remaining return gap.

## Round 2: oracle discriminator and observational audit

Reviewed observational v12 trainer and the repaired physics oracle. No additional concrete RNG/optimizer parity or restoration defect was established by source inspection. The supplied compiled parity result is separate evidence; this committee did not execute it. MuJoCo integration fallback stores time, qpos/qvel, activation, controls/applied forces, solver warmstarts, mocap, user/plugin state and equality flags when present. Restoration recomputes derived arrays then restores integration inputs again. Exact zero-action replay guards consistency.

The oracle's mean/concentration covector transformation is correct. Let g_a and g_b be logit score PG, and s_a,s_b softplus derivatives. Then:

    g_mu = kappa*(g_a/s_a - g_b/s_b)
    g_kappa = mu*g_a/s_a + (1-mu)*g_b/s_b.

These magnitudes are coordinate dependent. Compare coordinate directions at equal Beta-Fisher cost: d_z/d_mu=(kappa/s_a,-kappa/s_b), d_z/d_kappa=(mu/s_a,(1-mu)/s_b). The coordinate Fisher is Tᵀ F_z T, including its off-diagonal term. An isolated coordinate's best local score slope per unit Fisher length is g_coordinate/sqrt(F_coordinate_coordinate). Near negative shape logits, inverse-softplus derivatives amplify coordinate numbers without establishing an easily realizable parameter movement. Finite mean feasibility at fixed kappa is 1/kappa<mu<1-1/kappa; at fixed mu, kappa must keep both shapes greater than one.

`scripts/audit_sepclip_oracle_geometry.py` implements this discriminator and a compiled actor Jacobian fit. It computes the product-sphere tangent operator B=F_z^(1/2) J/sqrt(number of states), fits the natural-oracle response F_z^(-1/2) g with ridge-free SVD, and reports rank, effective spectrum, residual, per-layer Fisher contributions/cross terms and finite post-retraction KL/error. FP64 tangent projection renormalizes unit constraint vectors to avoid FP32 norm-rounding being mislabeled as extra freedoms. Jacobian/SVD caches are shared across oracle horizons and replicas at exactly identical states. Tests cover coordinate finite differences/Fisher equivalence, radial projection and an intentionally unreachable linear-actor score pattern. All execution is delegated to mlq by the parent.

Interpretation gate: hundreds of functional coordinates with thousands of parameters can fit an arbitrary small saved-state score pattern. Near-perfect fit is finite-state interpolation, not evidence that the architecture learns/generalizes the desired gait. A large minimum-norm parameter movement or poor finite-retraction fidelity likewise means the local linear solution is not a realizable small intervention. Held-out physical states and fresh closed-loop training remain decisive.

The physics PG estimates are conditional *initial-action* gradients with continuation policy held fixed. Mapping them through the actor Jacobian does not produce the derivative of the whole finite-horizon rollout policy: later-action score terms are absent. Averaging age-250/500 evaluation states also is not the trainer's discounted occupancy. The new dense oracle addresses state matching but keeps the explicitly conditional score scope. Leave-one-out baselines remain conditionally unbiased with shared continuation noise because different initial action samples are independent conditional on that noise. Replicas measure continuation Monte Carlo uncertainty.

V12 raw_next_obs at reset boundaries is the autoreset observation; normalized_transition_obs preserves the learner's true final observation. This is sufficient for masked TD/truncation analysis but not exact reconstruction of every raw physical final observation. Future instrumentation should copy final_observation separately; executing frozen v12 should not be retrofitted.

Discriminating predictions: if oracle mean motion has strong equal-Fisher slope, is reachable with small tangent parameter motion, and the actual update spends its budget on concentration or points against that field, head coupling/update direction is implicated. If only a large/unstable fit or deficient held-out signed-coordinate response can realize the oracle field, representation is implicated. If GAE and oracle agree but the post-retraction optimizer update loses that pairing, optimizer geometry is implicated. If horizon or independent replicas change these signs, none of these causal attributions is yet justified.

## Completed physical-accounting evidence

From `physics_complete.json`, V7 half-rate end checkpoint frozen-row paired sample/mean/mode scores are 8856.74, 9582.11 and 9750.18. Sample-to-mean gain 725.37 decomposes exactly into +707.00 from higher average forward velocity (9.18293 to 9.88993 m/s) and +18.37 from reduced control cost (0.326193 to 0.307821 per step over 1000 steps). Mean-to-mode gain 168.08 decomposes into +234.24 velocity contribution minus 66.16 extra control cost. Sampling therefore harms velocity/gait substantially more than it increases the torque penalty.

The full-rate end mode score is 10207.08 with average velocity 10.58034 and control cost 0.37326 per step. Holding that cost, an intended 20k score over 1000 steps requires average velocity 20.37326 m/s, or 9.79292 above this checkpoint. Eliminating its entire torque cost would add only 373.26 score. These are reward-accounting identities, not an attainable-optimum oracle or a claim that the missing velocity can be learned by an optimizer modification.

Parent-reported layer data show large positive interlayer Fisher cross terms and approximately 0.5% concentration-only Fisher allocation; the hypothesis of dominant concentration-budget waste is rejected for those observed steps. Parent-reported physical-mean input Jacobian has full action rank six but effective rank 1.589. Coordinated locomotion can legitimately depend on a few phase coordinates, so this low effective rank alone is not a representation defect. Final feature cosine 0.13 under global observation sign reversal also rejects a simple globally even actor story; that synthetic transformation is not an on-manifold intervention. Causal architecture selection must use the oracle correction field and actual visited-state transport.

Geometry contracts: parent reports all four CUDA contracts passed in job 11755, after replacing an unseeded, potentially ill-conditioned toy fixture with explicit constraint vectors and observation design plus Taylor-scaling checks. Actual oracle fits retain unrestricted minimum-norm solutions and quantify large-step invalidity rather than hiding it. Exact finite KL now uses FP32 learner softplus+1 shapes, then FP64 KL algebra; the FP64 Fisher re-evaluation remains explicitly labeled as a diagnostic metric.

## Round 4: anchored mode/precision chart review

Source-reviewed root-owned `ppo_continuous_action_sepclip_anchored_precision_chart_v13.py` and its contracts; no source edits or GPU execution by this committee. Let A=softplus(anchor_alpha), B=softplus(anchor_beta), c=A+B, p=A/c, and residual coordinates dm=trainable_alpha-anchor_alpha, dl=trainable_beta-anchor_beta. Decoded excess shapes are:

    p_new = sigmoid(log(A/B)+dm)
    c_new = c*exp(dl)
    alpha_new = 1+c_new*p_new
    beta_new = 1+c_new*(1-p_new).

At initialization the frozen readout and trainable readout are equal *functions of the current hidden features*. Thus dm=dl=0 and their hidden derivatives vanish. The initial decoded policy and hidden/input Jacobian match the original mathematically. This cancellation is not stop-gradient: the frozen anchor still differentiates through its evolving input features, which is necessary to preserve hidden sensitivity.

The head/gain parameter gradients deliberately change. With s_a=sigmoid(anchor_alpha), s_b=sigmoid(anchor_beta), the derivative of decoded old-shape logits with respect to the two trainable residual slots is:

    T = [[ A*(1-p)/s_a, A/s_a ],
         [ -B*p/s_b,    B/s_b ]].

Consequently the head receives Tᵀ times the original shape-logit cotangent. At a symmetric zero anchor, T=[[log(2),2log(2)],[-log(2),2log(2)]]. An antisymmetric original PG goes into the mode slot; a symmetric original PG goes into the precision slot. Adam's elementwise moments and first-step sign/epsilon update are not invariant to this mixing/scaling. The first head proposal should change immediately.

The initial hidden gradient is unchanged mathematically, but actor clipping includes head gradients. Different head norms can change the shared actor clip factor; then Anvil's hidden gradients differ by that factor. Its cascade is approximately scale invariant, but finite normalization epsilon and BF16 boundaries preclude blanket exact equality. A full first-PPO-gradient/clip-norm/hidden-proposal contract is stronger than the existing input-Jacobian contract. Without active clipping, hidden Anvil proposals are expected to agree up to the decode's finite-precision error; subsequent hidden updates need not agree because policies and heads have diverged.

Mode and excess precision are separated, but physical *mean* and precision remain coupled. With k=2+c and physical native mean mu=(1+c*p)/(2+c):

    d_mu/d_dm = c*p*(1-p)/(2+c)
    d_mu/d_dl = c*(2*p-1)/(2+c)^2.

Increasing precision moves the mean toward the mode whenever p differs from one half. This is a useful falsifiable coordinate prediction, not independent control of mean and noise. Compare realized mean/concentration Fisher budgets rather than label the chart decoupled in those physical coordinates.

The frozen anchor also changes the readout function family after initialization: decoded shapes mix a fixed readout on current features and a learned residual readout nonlinearly. Equal active parameter count and identical initial policy/hidden tangent do not make this a global reparameterization of the old linear-head family. A benchmark gain must be attributed to an anchored-chart architecture/conditioning intervention, with matched Adam/Anvil controls, rather than to transport improvement alone.

The inverse-softplus expression is stable for positive finite excess. The upstream exponential imposes the actual FP32 domain: log excess sufficiently negative underflows to zero, yielding -inf logits and potentially 0*inf backward NaNs; log precision above approximately 88 overflows FP32 shape values. Existing wide numerical tests use FP64 and do not establish FP32 runtime backward stability. The initial unit-head bounded-anchor region is safe. A FP32 decode/backward grid should document the intended numerical range without adding a functional precision cap.

## Adversarial statistical round: no primary chart mechanism established

Current reported first-step Fisher cost is approximately 99.8% mean motion; historical full-rate mode-versus-mean gain is only about 177 points. These measurements contradict dominant concentration-budget waste and near-bound mode access as primary explanations for a roughly 10k remaining return gap. The chart remains an architecture/conditioning hypothesis, not the diagnosed main cause. It needs a predeclared chart-specific direction or observable prediction with independent physical gain at matched whole-policy Fisher cost; a new basis improving a noisy fitted surrogate is insufficient.

The dense oracle estimates conditional initial-action PG at selected states. Its leave-one-out estimator is unbiased conditional on common continuation noise for independent first actions and an exact continuous law. Clipping atoms/excluded inverse-CDF tails introduce separately reported law approximations. Current estimated Q minus current V comparisons also contain frozen-normalization, finite-horizon, and sampled-action effects; they do not identify true critic bias alone.

Two replicas with 16 first actions per state do not establish a high-dimensional optimal direction. For whitened estimates w1 and w2, independent-replica identities give E[w1·w2]=||true conditional field||² and E[0.5||w1-w2||²]=Monte Carlo noise trace. The observed squared norm of a fitted field overestimates true signal energy. A Cauchy bound based on that noisy norm can make an otherwise correct actual direction appear inefficient. An overparameterized actor can interpolate noise, and one held-out replica supplies one validation realization. Pairwise quantities for more than two replicas share replicas and are not independent CI samples.

The actual-step audit now reports signed cross-energy, noise trace, whitening-invariant pair cosines, noise-corrected mean-field energy without positivity clamping, and conditional scalar actual-gain replica values/Student-t intervals. A negative cross-energy marks unresolved full-field signal. This does not automatically invalidate a predeclared scalar direction: large noise can lie orthogonal to that direction. Its own replicated scalar gain is the relevant test. With only two replicas a nominal 95% Student-t interval uses multiplier 12.706, not approximately two; normality remains a hypothesis, not a guarantee under chaotic continuations.

Jacobian precision matters independently of Monte Carlo noise. Derivatives are evaluated in FP32, then projected/decomposed in FP64. Default FP64 arithmetic rank can retain modes below derivative accuracy. Both geometry scripts preserve this exploratory fit but report same-SVD sensitivity at relative cutoffs 1e-7 through 1e-4: rank, reachable oracle energy, parameter norm needed at the full Fisher budget, and maximum constrained efficiency. Actual motion is projected into each retained range before comparison against that range's bound. Neither full rank nor a tiny singular mode alone is a capacity or optimizer diagnosis.

Predeclared decision gates for additional evidence:

1. Use actual first-step and proposed fixed-direction scalar physical pairings, not an optimized noisy field's training fit. Require independent continuation replicas and signs consistent at the selected horizons. Report conditional intervals and raw versus normalized reward disagreement. Eight total replicas with unchanged states/action draws is a focused next measurement if two-replica conclusions are unresolved; it is not another training seed or sweep.
2. Treat fitted-direction selection as unresolved when its held-out pairing flips sign or when plausible FP32 numerical cutoffs materially change the conclusion. Large fitted parameter displacement or finite-retraction discrepancy invalidates the local fit as an actionable small intervention.
3. A useful operational local-linear gate is less than 10% logit prediction error and roughly 20% agreement of exact KL with the quadratic estimate, excluding cases near the numeric floor. These are declared diagnostic tolerances, not universal mathematical thresholds or sufficient evidence of return improvement.
4. Dense max-age 400 selection systematically omits later episode states included in PPO. To test a purported global update bias, add states spanning later steady ages (up to 700 for horizons 64/256). State matching at an early subset does not equal occupation-distribution matching. Correlated deterministic state selections do not supply independent-state or across-seed uncertainty.
5. Inspect actual ground-contact group counts. Age-stratified selection can miss sparse stance/torque phases and mostly sample flight. If phase coverage is poor, a contact/flight-stratified matched dataset is more informative than increasing generic actor capacity or interpreting effective input-Jacobian rank as a defect.

These gates can reject the chart as the primary intervention. No existing finite fork, Cauchy bound, or deterministic score establishes an attainable global optimum or predicts 20k. The missing physical performance remains principally a forward-velocity/gait gap; the investigation must identify a reproducible correction to that gait rather than defend a predetermined output chart.

## Closed-loop actual-weight intervention attribution

A proposed diagnostic changing checkpoint weights for every future action addresses a different quantity from the current initial-action oracle. For restored initial-state distribution rho and a finite horizon H, the full stochastic policy gradient contains the sum over trajectory times:

    E_old[sum_t gamma^t * (score_t·J_t delta) * Q_(H-t)(s_t,a_t)].

The existing conditional first-action score pairing supplies only t=0 at the chosen restored states. Multiplication by H or 1/(1-gamma) is invalid: later states, action scores, tail horizons and feedback differ. A negative whole-policy physical effect despite positive t=0 pairing does not establish critic bias.

The strongest inexpensive causal decomposition uses four paired arms:

- A: old policy throughout.
- B: new policy for the first action, old policy afterward.
- C: old policy for the first action, new policy afterward.
- D: new policy throughout.

B-A is the exact finite first-action intervention, C-A is the exact future-policy intervention given an old first action, and D-B-C+A is their finite feedback interaction. Common uniforms pair all arms, while independent continuation replicas measure Monte Carlo uncertainty. Reconstruct/validate first-step weights separately from the exact saved ten-epoch endpoint; do not renormalize or alter the actual endpoint silently.

Required measurements are per-state/per-replica raw and frozen-normalized return differences, time-resolved velocity versus control-cost changes, and old/new Beta KL evaluated on both old and new trajectories. Initial-state KL alone cannot detect later occupation amplification. Complete simulator state, episode age, and exactly corresponding frozen normalization rows must be identical before the first action. Mean/mode behavior arms change the stochastic policy law and therefore diagnose behavior, not the correctness of the sampled-law PG.

Attribution rules:

- Negative B-A agreeing with negative independent t=0 pairing demonstrates a harmful first-action direction on that state subset. It still does not isolate critic bias from sampled-GAE noise or finite-horizon/objective mismatch.
- Positive B-A but negative C-A identifies harmful later policy effects absent from the initial-state gradient comparison. Those later effects could have their own wrong PG or distribution shift; initial t=0 evidence cannot decide.
- Positive separate B-A and C-A with a large negative interaction implicates finite nonlinear feedback. The first-order predictor is not adequate for that update; calling the failure critic bias is unsupported.
- Positive first actual step but negative ten-epoch endpoint separates a locally useful early change from later within-update motion. Inspect changing PG, clipping masks, momentum and functional allocations before proposing a causal remedy.

To identify PG estimator bias directly, collect old-policy trajectories and independently estimate the full finite-horizon scalar directional PG using all-time score-return terms, with a state-only baseline. Compare GAE score pairing computed on the same trajectories, rewards, normalization and explicit horizon boundaries. The current dense first-action oracle plus a closed-loop finite return comparison cannot substitute for this full directional test. For numerical/curvature uncertainty, a symmetric smaller weight-direction experiment can estimate a local derivative, but its actual FP32 parameter motion and retraction must be recorded; shrinking already tiny steps can hit parameter quantization.

These comparisons remain conditional finite-horizon diagnostics at restored states. They are neither a fresh training improvement nor the full benchmark initial-state evaluation, and they do not establish global optimality.

## NorMuon named-optimizer coverage and exact fresh control

The best historical NorMuon result (final100 8189.4458) is `sepclip_anvil2_v4_normuon_fast`, whose wrapper `scripts/run_sepclip_anvil2_v4.sh` sets lr .005, momentum .85, **lane beta2 .95**, BF16 polar cascade, actor hidden matrices only, and auxiliary Adam lr .0024. The earlier original-row and long-axis v3 controls are different implementations/settings and are not the best matched reference.

The v12 candidate flags must explicitly include `--normuon-beta2 .95`; its default .9 would be a new optimizer variant. Correct matching flags are `--actor-optimizer anvil2 --matrix-method normuon --normuon-lr .005 --normuon-momentum .85 --normuon-beta2 .95 --matrix-precision bf16 --normuon-network actor`, in addition to the shared seed1/8M/16env/2threads/compiled benchmark protocol. The actor_optimizer name selects the matrix-versus-Adam plumbing; matrix_method selects the actual NorMuon algorithm.

Source inspection finds v4 and v12 NorMuon branches identical: velocity EMA, Nesterov-like input `grad.lerp(velocity,.85)`, Polar Express maps and BF16 boundaries, long-axis second-moment lanes, Frobenius norm preservation after lane adaptation, multiplier sqrt(max(1,rows/cols)), head/critic Adam, separate .5 gradient clips, and post-step row/column retraction. Architecture blocks are text-identical. v12 begin_rollout/slow_clock changes only metadata and an unused scalar in this branch. Its rollout-clock Anvil behavior is therefore irrelevant to NorMuon equivalence. No momentum parallel transport exists in either reference; introducing it would be a new intervention.

A compiled v12-versus-v4 parity contract is still necessary. The existing observational contract exercises v7 Anvil and Adam, not v4 NorMuon. Use exact best flags and identical seeds, compare all initial weights, then at least two rollouts with ten updates each; assert exact weights and optimizer velocity/second after each update, compiled PPO gradients, unchanged inactive second rail, and irrelevance of begin_rollout to NorMuon motion. This can validate an instrumentation port without retraining from a checkpoint. One fresh full8M run would supply the missing correctly normalized NorMuon weights, update traces and physical diagnostics. Until then, Anvil/Adam microscopic measurements cannot be reported as measurements of NorMuon, even where normalized-layer geometry is shared.

## Final matched-GAE closed-loop source review

The revised closed-loop diagnostic correctly closes the earlier objective mismatch. Its all-time likelihood-ratio derivative uses gamma^t times the local actor chord score and the return starting at t. For the augmented objective, the terminal old critic is held fixed as a function, but evaluated at each arm's actual terminal state. Its derivative enters through trajectory likelihood, with tail contribution gamma^H V_H times the sum of all earlier scores; it does not require differentiating frozen critic weights. GAE(lambda=1) equals augmented MC minus state baselines pathwise. The tests also verify the exact future-error transport coefficients for lambda<1. No blocking mathematical or implementation defect was found in this read-only review.

Raw and fixed-normalized objectives are individually consistent: raw rewards pair with old normalized critic times the selected fixed reward std; clipped fixed-normalized rewards pair with the unscaled old critic. They are constant rescalings of one another only absent clipping and numerical rounding. The raw baseline need not be an accurate raw value to remain an unbiased state-only control variate. The lambda1 identity checks recursion, not true critic accuracy. The lambda.95 discrepancy describes the selected finite old-critic-terminal objective, not infinite-horizon ground truth.

The derivative follows the ambient chord between actual saved actor weights. The chord has a small radial component from sphere retraction; an exactly retracted path has a different tangent at second order in the actual update. This is an accurately labeled local chord diagnostic, not an exact Riemannian derivative. A finite-gain minus derivative residual combines curvature, closed-loop interaction and Monte Carlo error. Continuous Beta score approximation also omits sampler clamp atoms and truncated-uniform mass; substantial recorded atom probabilities would prevent strong exact-PG sign claims without an atom-aware score. Four noise replicas quantify conditional error on the fixed age<=400 states, not uncertainty over state occupation, seeds or benchmark returns.
