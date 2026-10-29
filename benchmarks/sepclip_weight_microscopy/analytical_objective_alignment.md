# Objective alignment contract for physical direction interpretation

Independent read-only source review of dense_pg, actual_oracle_step and closed_loop_update, read together with ppo_epoch_credit_appendix.md and analytical_feedback_credit.md. No model/GPU execution, queue submission or existing-source change. This document separates the quantities the existing jobs estimate before any sign disagreement is assigned to weights, clipping, credit or optimizer geometry.

## Five distinct directional quantities

Use theta0 for the saved old policy and d for a fixed actual first-update or full-endpoint parameter chord. Let u_d(x)=J_z,theta(theta0,x)*d. Let S_z(x,a) be the exact old Beta logit score, with continuous-law and clipping qualifications. Write D_i(d)=S_z(x_i,a_i) dot u_d(x_i).

1. Learner empirical old-gradient pairing:

   P_B,GAE(d) = mean over ALL stored rollout rows i of Ahat_i*D_i(d).

   This uses saved normalized observation/native action and raw stored GAE, uniform empirical row weights, no outer gamma-to-rollout-time. For default norm_adv=false and ent_coef=0 it is the initial unmasked ascent covector. It is not the current later-epoch clipped gradient or a raw physical episode objective. A comparison at later theta needs that theta score/Jacobian, ratio and advantage-sign mask explicitly.

2. Dense SELECTED observed-action pairing:

   P_S,GAE(d) = mean over selected rows i in S of Ahat_i*D_i(d).

   Dense parameter_gradient_comparison reconstructs this selected observed-action gradient, and separately reports the full-rollout PG norm and selected/full cosine. Selection is round-robin environment rows with stratified eligible timesteps whose episode age<=400. Even though each selected row is exactly replayed, the empirical measure d_S is not d_B. It emphasizes early/middle ages and may omit late/contact states.For the default 16-env/1024-step trainer, 32 selected rows compared with 16384 full-batch rows is about 0.1953% of empirical rows, not a justified probability sample of every age. The source does not report a general importance weight from d_S to d_B.

3. Dense independent conditional first-action physical pairing:

   P_S,H,first(d) = mean_i in S E_(a~pi0, continuation~pi0)[(S_z(x_i,a) dot u_d(x_i))*Q_H(i,a)].

   Only the first action law is varied in this derivative; every later action is drawn from the old controller with that selected state's frozen observation normalizer and reward scale. Q_H has inner discount gamma^t from that initial action. No outer discount weights the selected states. Dense policy_logit_pg stores finite_raw, finite_normalized and bootstrap_normalized fields. Independent first actions plus leave-one-out action baseline estimate each conditional field without empirical self-baseline shrinkage. The existing dense submission uses the script default TWO independently redrawn action/continuation replicas (the closed-loop submission uses FOUR). Dense replicas assess conditional action/noise sensitivity, not selected-state uncertainty.

   Dense observed_action instead holds ONE originally collected action per selected state, averages continuation Q replicates, and subtracts V_old at that state. Its parameter_gradient_comparison and clipping sensitivity compare that conditional Q-minus-V estimate to the stored GAE. They do not measure an action-integrated true advantage. The same observed score reused in every continuation replica retains original action sampling noise.

4. Actual-step report pairing:

   P_actual,H(d) = mean_i in S g_logit,H,finite_normalized(i) dot u_d(x_i).

   The actual_step script explicitly reads ONLY dense policy_logit_pg.finite_normalized for its physical field. It does not silently switch to raw or critic-tail fields. selected_observed_gae_first_order_gain is P_S,GAE(d); independent_replica_actual_first_order_gains are replicas of the normalized, zero-tail conditional first-action P_S,H,first(d). Both first and full ten-epoch chords are linearized at theta0. Per-layer pairings sum to the joint pairing, and joint Fisher cost retains cross terms. Tangent projection and retained-range projection define further, labeled directions; they are not identical to the unprojected ambient chord.

   finite_actual_logit_change pairs g_old with z_endpoint-z_old instead of J_old*d. This is still a score-field pairing, not measured finite physical return gain. Cross-replica fitted directions validate held-out NOISE on the same selected states, not new states or the full controller gradient. Reachable rank and Cauchy efficiency concern this finite normalized first-action field.

5. Closed-loop all-time physical pairing:

   P_S,H,all(d) = mean_i in S E[sum_(t=0)^(H-1) gamma^t*(S_z(x_it,a_it) dot u_d(x_it))*(G_it^C-b_it)].

   Here x_it follows the old controller from selected initial state i. G_it^C uses remaining horizon H-t, and optional fixed old-critic terminal C. The actor chord is evaluated at every reached observation; it is not the initial u_d repeatedly applied. Old trajectory rewards, normalized values and directional scores are saved. Baseline and no-baseline estimators are both available. Matched GAE transport compares lambda credit versus augmented MC on this SAME old trajectory and SAME fixed terminal function, unlike comparison to the stored training rollout.

   Actual paired finite returns deploy old, first-update and endpoint controllers throughout the trajectory. Hybrid arms switch only at t0 or only after t0. Finite all-minus-t0-minus-after interaction is exact under the chosen coupling but is not a score-gradient decomposition. Deterministic mean/mode returns belong to different controller laws and cannot independently certify stochastic PG signs.

## An objective signature must match before interpreting signs

For each claimed pairing, record:

| Axis | Learner stored old covector | Actual-step physical pairing | Closed-loop physical pairing |
|---|---|---|---|
| Initial/state measure | Uniform all rollout rows d_B, or explicitly selected d_S | Uniform selected d_S | Selected initial d_S, then propagated old occupancy |
| Observation transform | Each row's actual recorded online transform | Initial recorded transform; thereafter fixed selected-row transform | Same frozen selected-row transform on every continuation |
| Action law | Original saved native samples and old FP32 log probabilities | Redrawn old first actions; frozen old continuation | All old continuation actions via inverse-CDF coupling |
| Reward units | Actual time-varying applied training normalized rewards inside stored GAE | Frozen per-state reward std and clip10 | Raw or same frozen normalized objective separately |
| Credit/horizon | Stored GAE with rollout/end/truncation bootstraps | Finite H conditional first-action zero-tail field | All-time gamma-weighted finite H remaining credit; zero or frozen critic tail |
| Parameter direction | Actual d; at theta0 or specifically later theta | Old-policy ambient/tangent/retained direction separately | Fixed ambient chord at old policy versus finite deployed new controller |
| Uncertainty | One fixed empirical training sample | Independent first-action/continuation replicas at fixed states | Conditional continuation replicas at fixed initial states |

A matching checkpoint hash and exact state replay close provenance axes, not occupancy/horizon/objective axes. Numerical JVP agreement closes a derivative axis, not a physical-credit axis. Resolve axes in this order before assigning a mechanism.

## Why exact first-action and all-time derivatives can disagree

Define the conditional old-policy horizon field h_K(x;d)=E_a[(S_z dot u_d)*Q_K(x,a)] with the appropriate frozen normalizer/units. Then

P_S,H,first(d) = E_(x~d_S)[h_H(x;d)],

P_S,H,all(d) = sum_t gamma^t E_(x~d_t(d_S,pi0))[h_(H-t)(x;d)].

The t0 term is the first-action field, but later terms change both state distribution and remaining horizon. No positive scalar turns the former into the latter in general. Starting all-time evaluation from d_B would still propagate that distribution; it does not keep it uniform. Equality up to 1/(1-gamma) requires demonstrated stationary occupancy and a compatible infinite-horizon Q, not merely a long rollout or a high explained-variance critic.

The appendix's exact two-state example yields old uniform derivative+.125 and infinite physical derivative-74.125 at gamma=.99 with an exact critic. That possibility alone prevents classifying a first-versus-all-time sign flip as critic bias or weight failure. It does not quantify this run's occupancy difference. More narrowly, an actual direction that pairs negatively against a reliable FIRST-action field can be called locally adverse for that conditional intervention, but not harmful to every propagated trajectory.

## Reward scale changes selected-state weighting even without clipping

Let sigma_i be the frozen reward std at selected initial state i. With no reward clipping and the same physical continuation, normalized per-state directional pairing equals raw per-state pairing/sigma_i. Therefore

P_normalized = mean_i[P_raw,i/sigma_i].

It is generally NOT P_raw divided by a single average sigma. If different states have opposite-sign physical pairings, heterogeneous positive scales alone can flip their aggregate sign. The code's raw critic baseline multiplies the old normalized value by sigma_i per state; this is correct local unit handling, not a universal aggregate conversion. Reward clipping further breaks even per-state proportionality. Online training's future applied scales also differ from the selected frozen sigma_i, so matching initial normalizer metadata does not match the later reward objective.

Observation statistics can similarly drift on the original trajectory, but dense normalization_drift reports only the saved available future steps, sometimes shorter than the requested H. It does not evaluate the counterfactual original online normalizer on each new fork. A low recorded drift summary cannot bound omitted adaptive-feedback influence without a sensitivity model.

## Finite and estimated-terminal objectives differ explicitly

For J_H^0, the omitted future return is zero by definition. J_H^V instead adds gamma^H*V_old(X_H), with V_old a fixed function in chosen units. Their directional difference is

gamma^H * E[V_old(X_H)*sum_(t=0)^(H-1) D_t(d)].

Closed-loop matched_gae_transport stores this terminal_tail_directional_pg; augmented MC and GAE(1) agree algebraically. This is the derivative of the ADDED critic-terminal reward, not a certificate that J_H^V approximates infinite physical return. Dense bootstrap_normalized PG adds only the initial-action score paired with that tail, because later policy laws are held fixed. These two tail gradients also differ in scope.

A sign reversal between zero-tail and critic-tail fields implicates objective sensitivity. A sign reversal between H256 and H512 can involve later reward, changed remaining horizons, occupancy and MC noise. It should not be silently resolved by declaring the longer field true. Comparing dense zero-tail actual-step pairing to closed-loop fixed-critic-tail GAE without first separating these tail terms creates a mismatch even when every estimator is correct.

## Scalar sensitivity limits available without new measurements

At gamma=.99:

| H | gamma^H | Sum of all-time outer gamma weights | Conservative omitted normalized return bound if every future reward stays clipped to10 |
|---|---|---|---|
|1|.990000|1.0000|990.000|
|16|.851458|14.8542|851.458|
|64|.525596|47.4404|525.596|
|256|.076315|92.3685|76.315|
|512|.005824|99.4176|5.824|

The bound 10*gamma^H/(1-gamma) holds for a hypothetical continued clipped normalized objective; it is not a raw-return bound and does not identify the unobserved continuation. A conditional first-action logit PG tail can conservatively be bounded by 1000*gamma^H*E[norm(S_z)] under that same bounded-reward continuation and integrability. That factor remains unmeasured, and an all-time tail score has the sum of all trajectory scores rather than only the initial score. A small discount multiplier by itself is not a directional sign certificate.

For H512 the initial action's outer weight is1 of99.4176 total, around 1.006% if terms had comparable size; this is weight mass, not a claim that its return derivative is exactly1% of the full one. The remaining terms may cancel or reinforce.

Existing historical v7-half end critic has normalized value mean 3.5136/std .2036, stored GAE std .06144, and self-target explained variance .91361. The conservative normalized H512 omitted-return bound 5.824 exceeds that observed value mean and dwarfs the observed advantage scale; it cannot justify a decisive near-zero physical gradient sign. These are descriptive comparisons on old rollout states, not a bound on true value error or an observed actual tail size. In particular the very loose rewardclip bound should not be substituted for the measured terminal_tail_directional_pg.

Historical strongest first-action replica cosines -.029/-.041 at H64/H256 already show unresolved direction stability. Existing artifacts do not yet supply a trustworthy numeric occupancy-distance bound, a closed-loop state-population confidence interval, or a full unmatched-unit correction. Those quantities must not be invented from scalar returns or feature spectra.

## Interpretation contract for incoming existing jobs

1. Label every reported gain with first-action versus all-time scope, d_B/d_S/propagated measure, raw/frozen-normalized units, zero/critic tail, horizon and direction linearization point.
2. Compare P_S,GAE to observed-action Q-minus-old-V at the SAME selected saved actions before calling a credit difference; retain continuation and original-action uncertainty. Compare selected/full PG cosine before extrapolating selected findings to all learner rows.
3. For actual-step physical signs, use its exact finite_normalized zero-tail field and inspect independent replica directional intervals, rank sensitivity and actual joint Fisher cost. A fitted Cauchy ratio on noisy field is not evidence against weights.
4. For closed-loop signs, compare raw versus normalized per-state contributions, zero versus fixed-tail objectives, baseline versus no-baseline estimators, H256 versus H512, and all-time versus t0 interventions. Disagreement has multiple explicit objective explanations before critic bias.
5. Same-trajectory GAE-minus-augmented-MC isolates a paired credit-estimator difference for that objective. It does not identify V_pi, old-rollout credit bias, or the full training population.
6. A positive measured derivative with a negative paired finite gain establishes failure of that local prediction at this chord, not a defective local Jacobian or critic by itself. Inspect independent nonlinear JVP contract, occupancy feedback/contact changes and finite-step curvature.
7. Claim local optimizer transport failure only when the SAME independent physical field supports the raw training direction yet opposes actual combined motion at comparable joint functional cost, with resolved uncertainty and finite validation. If state/horizon/unit/objective axes differ, report objective mismatch unresolved.

A score sign disagreement is an observation. An attribution to layers, weights, critic or optimizer requires the matched signature and subsequent causal evidence; neither source correctness nor a stronger unmatched reference automatically provides it.

## GAE filtering weights and matched-score error sensitivity

At gamma=.99 and lambda=.95, gamma*lambda=.9405.

| H | gamma^H objective-tail attenuation | (gamma*lambda)^H residual-filter attenuation | Critic-error terminal coefficient gamma*(gamma*lambda)^(H-1) | Sum of future-error coefficients gamma*(1-lambda)*sum_(j=1)^(H-1)(gamma*lambda)^(j-1) |
|---|---|---|---|---|
|64|.525596|.0197226|.0207606|.814487|
|256|.076315|1.51305e-7|1.59268e-7|.8319326|
|512|.005824|2.28931e-14|2.40981e-14|.8319328|

GAE's direct far-terminal error coefficient attenuates much faster than the physical objective tail, while its accumulated FUTURE intermediate value-error mass approaches .83193. It therefore does not imply negligible critic transport merely because lambda filtering kills the last residual. Nor does that coefficient mass measure actual score alignment. These are exact filter attenuation factors, not terminal-value accuracy, bounded raw-return error, or measured PG bias. The current-state baseline coefficient-1 cancels only under the stated matched unclipped conditional score expectation.

For a matched continuous-law, initial-policy, unclipped state/action measure and fixed direction d, write s_d=S_z dot u_d, and epsilon for the matched advantage/credit error. Cauchy gives

abs(E[s_d*epsilon]) <= sqrt(E[s_d^2])*sqrt(E[epsilon^2]).

If c=.5*E_x[u_d^T*F_z*u_d] is the EXACT joint Fisher cost under that SAME state/action law, E[s_d^2]=2*c, so abs(error pairing)<=sqrt(2*c)*RMS(epsilon). Cross-layer terms belong in c. This bounds magnitude only; it neither establishes sign nor proves the error component caused the actual step. A fixed state-only error cancels exactly, even if this loose magnitude bound is large. Matching epsilon to a future-filtered GAE transport term requires those future conditional errors, not just the reported stored advantage standard deviation or self-target residual.

A weighted sum of future errors has RMS at most sum_j abs(w_j)*RMS(e_j) by the L2 triangle inequality, with each e_j under the appropriate matched propagated distribution. No such independent error RMS is established in existing scalar artifacts. Calling .06144 GAE standard deviation a critic-error RMS would be unsupported.

At later PPO epochs the relevant covector is rho*M*s_d, whose squared second moment under old action sampling is E_old[(rho*M*s_d)^2]. It generally does NOT equal2*c computed from the unweighted old-policy Fisher. Advantage-sign-dependent masks and action importance weighting must remain in the bound. The baseline-cancellation condition and held-independent evaluation requirement likewise remain explicit. This prevents importing the initial Fisher error bound into later clipped optimizer attribution unchanged.

Reviewed source SHA256:

- `scripts/audit_sepclip_dense_pg.py`: `1f4053ebe4854063ce19e36b05be37dc3e036d94e368b9b424595c5b98b4c058`
- `scripts/audit_sepclip_actual_oracle_step.py`: `80e4dbc117808da90e9b5b87febc856a2c26d708ac3a4e9c2a56542ffa64c8e5`
- `scripts/audit_sepclip_closed_loop_update.py`: `59dec8199edcadb6d02264ad3a8f9545d7a4d5785ad98a25660b1ed05619c208`

Precision addendum: actual_step builds its Fisher with beta_geometry(logits.double()), so softplus shapes are re-evaluated in FP64 from actual FP32 logits. This is a documented diagnostic re-evaluation, not simply FP64 algebra on already-rounded FP32 shapes. Its saved precision label correctly says this. The exact matched-law Fisher equality above is theoretical; it should not be asserted bitwise for sampled/clamped FP32 implementation solely because the diagnostic matrix uses FP64. No magnitude of this discrepancy is established here, and this source fact alone does not establish a causal or blocking defect.

Fixed-baseline cancellation also assumes the evaluated direction/baseline is held independent of the current evaluation action conditional on its state/history. A direction selected using the same original training actions is not an independently held-out test of that cancellation. Actual saved chords can be treated as fixed when scored with independently redrawn oracle actions/noise; fitted oracle directions require the existing held-out replica checks.
