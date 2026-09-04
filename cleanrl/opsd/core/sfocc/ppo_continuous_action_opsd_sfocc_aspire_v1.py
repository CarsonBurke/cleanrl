# OPSD-SFOcc ASPIRE v1 -- the improvement REQUEST is calibrated by ACHIEVEMENT RATE.
# No PPO anywhere: no ratio, no clipped surrogate, no advantage, no GAE, no policy gradient.
# Sibling of ppo_continuous_action_opsd_sfocc_v3.py (§17.5 of OPSD_FAMILY.md). One flag apart.
# =====================================================================================
# WHY THIS FILE EXISTS
#
# v3 built a vector occupancy channel and a half-space improvement query
#
#   psi(s,a) = E[ sum_k gamma^k phi(s_{t+k}, a_{t+k}) ],  r ~ w . phi,  V = w . psi
#   c        = (realized future footprint) - (predicted future footprint), standardized
#   c_query  = delta * w_unit,      delta = occ_boost * std(w_unit . c),  occ_boost = 1.0
#
# and it PASSED its channel gate (chan_action_r2 0.2078 against a state-only control of
# 0.1435) while DISCLOSING two defects of the query itself in its own telemetry:
#
#   sf/teacher_conc_ratio  ~1.00      The query SHIFTS the policy and barely SHARPENS it.
#                                     The family has measured repeatedly that mass-covering
#                                     distillation against a teacher whose concentration
#                                     does not change has no sharpening channel at all.
#   sf/query_support       ~0.18      The query zeroes every channel component orthogonal
#                                     to w. NOT this file's brief; unchanged here.
#
# and one structural defect inherited from every prior arm in the family:
#
#   A FIXED REQUEST IS ABSORBABLE. §6.4's own record: debug/cond_gap decayed 0.030 -> 0.010
#   over 1.75M steps as the student closed the gap on a stationary bar. One sigma of the
#   batch's own spread is a relative reference in SCALE but not in DIFFICULTY: nothing in
#   `std` knows whether one sigma is something the policy manages constantly or never, so
#   the request can be absorbed (met almost always -> no information left in it) or drift
#   out of reach (met almost never -> the conditional has no rows there) and v3 cannot tell
#   which, because it never asks.
#
# ------------------------------------------------------------------------------------
# THE CHANGE, AND NOTHING ELSE
#
# Ask. The realized gain along the improvement direction, w_unit . c, is already in hand
# in BLOCK D -- v3 takes its std -- so the achievement rate is one comparison and one mean.
# Integrate the error in LOG space:
#
#   achieved   = mean( (c @ w_unit) > delta )
#   log_delta <- log_delta + aspire_lr * (achieved - aspire_target)
#   delta      = exp(log_delta)
#
# THE FIXED POINT is where the drift is zero, i.e. P(w_unit . c > delta) = aspire_target:
# delta converges to the (1 - aspire_target) QUANTILE of the student's OWN realized-gain
# distribution, re-derived every iteration. That is the structural property a fixed target
# cannot have -- ABSORPTION RAISES THE BAR instead of ending the drive.
#
# THE FEEDBACK CONTRACTS GLOBALLY, which is why the band below is a DETECTOR and not a
# stabilizer: `achieved` is monotonically non-increasing in delta, so too-large a delta
# drives log_delta down and too-small drives it up. This is Robbins-Monro stochastic
# quantile estimation; there is no gain at which it diverges, only one at which it is noisy.
# LOG space, not linear: delta is a magnitude and exp keeps it positive with no clamping,
# and a step becomes a fixed MULTIPLICATIVE move, which matters because the channel's EMA
# normalizer and psi's standardization both move its scale during a run.
#
# Everything else is v3, unmodified: phi's losses, the FUTURE-ONLY channel construction and
# its anti-leakage property, the SF critic's separate optimizer / backward / clip, both
# actor losses. Three new args, one new scalar of controller state, three new tags.
#
# ------------------------------------------------------------------------------------
# WHAT IS UNCHANGED FROM v3 (the parts you need to read this file)
#
# One network, two contexts. Teacher = student under privileged conditioning. Two
# actor-side losses only, both supervised regressions onto DETACHED targets:
# clone_loss = -log p(a_taken | s, c) and distill_loss = an exact per-dim clipped Beta KL.
# No target networks, no twin critics, no replay, no ensembles, no simulator probing, no
# counterfactual stepping, no coefficient grids. No advantages and no GAE: value is a
# linear readout w . psi of a vector critic, so there is nothing for an advantage to BE.
# psi(s,a) is action-conditioned, which is not a Q(s,a) violation of §2: no max/argmax
# backup (the bootstrap is the student's own mean action), no target network, no twin, and
# w . psi never multiplies a gradient and never enters an actor loss.
#
# THE TWO INHERITED HAZARDS, engineered against rather than hoped away (full derivations in
# v3's BLOCK C and in OPSD_FAMILY.md §17.4):
#  1. ACTION LEAKAGE / IDENTITY COLLAPSE. clone_loss is -log p(a_t | s_t, c_t), so if a_t
#     is recoverable from c_t the fit collapses to the identity, drives the loss to -inf
#     and teaches nothing. The naive channel `tgt - psi` contains phi(s_t, a_t) explicitly.
#     The channel used here is FUTURE-ONLY, so a_t enters solely through the real
#     transition s_{t+1}. Gate: sf/cond_action_r2. UNTOUCHED BY THIS FILE.
#  2. PHI COLLAPSE. Trained on reward alone phi converges to reward and psi becomes a
#     scalar critic in a vector coat. Decorrelation penalty + sf/phi_eff_rank, which is
#     NECESSARY AND NOT SUFFICIENT (uncorrelated noise has full rank). UNTOUCHED.
#
# ------------------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS
#
#  A1 CONTROLLER WORKS. sf/aspire_achieved converges toward aspire_target = 0.25 and
#     sf/aspire_clamp_frac stays ~0.
#  A2 SHARPENING APPEARS. sf/teacher_conc_ratio moves materially above v3's ~1.00. IF IT
#     DOES NOT, THE QUERY STILL ONLY SHIFTS AND THIS ARM HAS NOT FIXED THE DISCLOSED
#     DEFECT, and that is what will be written here.
#  A3 NO CHANNEL DAMAGE. sf/chan_action_r2 does not regress below v3's 0.2078, read
#     against sf/state_action_r2 -- the RATIO is what matters, not the absolute, because as
#     the policy sharpens R2(action | anything correlated with s) drifts up for reasons
#     that have nothing to do with the channel.
#  A4 ANTI-ABSORPTION IS THE POINT AND IS NOT TESTABLE AT 65,536 STEPS. The claim is about
#     cond_gap's LONG-RUN trend. The decay it exists to prevent took the parent 1.75M
#     steps; 16 iterations cannot show it and MUST NOT be quoted as if it could. A flat or
#     rising cond_gap over 16 points is not evidence of a fix.
#
#  A PRE-REGISTERED RISK OF THE FIX ITSELF, stated before the run rather than after. If the
#  realized projection were zero-mean Gaussian its 0.75 quantile is 0.674 sigma, BELOW v3's
#  1.0 sigma -- so calibrating to the top quartile could make the request SMALLER, and a
#  smaller request is a weaker sharpening drive, not a stronger one. The mean of w_unit . c
#  is not actually zero (the level cancels 83%, not 100%) and the distribution is not
#  actually Gaussian, so the sign is an empirical question, which sf/aspire_achieved at
#  iteration 1 answers directly: it measures P(realized > v3's request) under v3's request
#  exactly. IT MATERIALIZED IN SIGN AND NOT IN MAGNITUDE: one sigma turned out to sit at the
#  0.772 quantile at iteration 1 (achieved 0.2283), i.e. barely above the 0.25 target, so the
#  controller had only a small correction to make and delta fell 8% rather than the ~33% the
#  Gaussian reading implied. See A1 and A2 below for what actually dominated.
#
# WHAT 65,536 STEPS CAN AND CANNOT TEST HERE
#   CAN: A1's clamp half and the controller's arithmetic; A2 (a concentration ratio is a
#        per-iteration measurement, not an asymptote); A3; the --no-aspire reproduction;
#        gradient isolation; NaN/Inf freedom; speed.
#   CANNOT: A4. Also CANNOT show A1's convergence half, and that is a consequence of the
#        controller's own time constant rather than of noise: see A1 below. And A2's
#        NEGATIVE, while decisive over the request magnitudes these runs actually visited
#        (a 2.40x range), says nothing about the +-10x the band permits -- the boundary is
#        drawn explicitly in A2 rather than left to the reader.
#
# ------------------------------------------------------------------------------------
# VERIFIED IN THIS FILE. Real measurements, this disk, HalfCheetah-v4, seed 1, CUDA,
# --num-envs 16 --num-steps 256 --num-minibatches 32 --actor-epochs 4 --critic-epochs 4
# --total-timesteps 65536 --cond-mode occ == 16 iterations and 2048 actor optimizer steps
# (4 actor epochs x 32 minibatches x 16 iterations). START = iteration 1, END = iteration 16.
# THE GPU WAS SHARED with an unrelated foreign job for the whole session and ITS LOAD MOVED
# (~100% utilization / ~20 GiB early, ~3% / ~8.7 GiB late), so every SPS figure below is a
# LOWER BOUND and only within-batch arm-vs-arm comparisons mean anything; see SPEED AND
# MEMORY. Nothing else in this block depends on wall clock. 12 runs of the three arms were
# recorded in total: 8112 scalar points, ZERO non-finite.
#
#   GRADIENT ISOLATION, MEASURED BOTH WAYS, on this file's own modules.
#     After sf_total.backward():   0 of 227 agent.parameters() have a non-None grad, and
#                                  the agent's total grad norm is exactly 0.0; 14 of 14
#                                  sf.parameters() do have grads.
#     After clone_loss.backward(): 0 of 14 sf.parameters() have a non-None grad, sf total
#                                  grad norm exactly 0.0; all 227 agent params do.
#     b_c, c_query, w_unit and psi_target all carry requires_grad=False.
#   The controller adds nothing to this: log_delta is a python float, never a tensor in an
#   autograd graph, with no optimizer and no state_dict entry.
#
#   --no-aspire REPRODUCES v3 EXACTLY. Both files run at the configuration above, tag by
#   tag and point by point out of the two TensorBoard event files:
#     34 tags / 640 points / 16 differing -- the 34 tags v3 emits (this file emits 37, the
#     three new sf/aspire_* being the difference) -- and all 16 differing points are
#     charts/SPS, which is wall-clock throughput and cannot be reproduced by construction.
#     Algorithmic tags only: 33 tags / 624 points / 0 DIFFERING. K = 0, bit-exact.
#   Directly checked in isolation as well: with --no-aspire, delta is bitwise equal to v3's
#   expression occ_boost * (c @ w_unit).std() (1.0123512744903564 both sides), the function
#   returns no controller state, and a deliberately poisoned stale state is ignored. With
#   --aspire, iteration 1's delta is bitwise equal to the --no-aspire value.
#   My v3 baseline run reproduces the campaign's recorded v3 END table to four decimals on
#   every tag (chan_action_r2 0.2078, state_action_r2 0.1435, cond_action_r2 0.3478,
#   psi_r2 0.2124, psi_bias_frac 0.0048, w_r2 0.9599, phi_eff_rank 20.6792, delta 0.6510,
#   cond_gap 0.0089, distill_kl 0.0200, clone_nll -4.9724, entropy -4.5496, EV 0.3453), so
#   the side-by-side below is against a verified, not a remembered, baseline.
#
#   CONTROLLER ARITHMETIC, verified end to end AGAINST THE LOGGED SERIES ALONE, which is a
#   stronger check than a single step: log_delta[i] == log_delta[i-1] + 0.05 *
#   (achieved[i-1] - 0.25) holds at all 15 steps to a maximum absolute error of 8.3e-08,
#   and max | exp(log_delta[i]) - sf/delta[i] | = 1.6e-08 -- so the delta used at iteration
#   i is exactly the exponential of the state written at i-1, with no off-by-one. Worked
#   first step: achieved[0] = 0.228271484375 exactly (935 of 4096 rows), the step is
#   0.05 * (0.228271484375 - 0.25) = -0.00108642578125, log_delta goes -0.563509 ->
#   -0.564596, and iteration 2 logs -0.564596 (agreement 1.2e-08; the residual is because
#   the seed takes a float32 log).
#   On a stand-alone harness against the final code, a FROZEN batch driven to convergence
#   (400 controller steps) settles at delta 0.7066 with achieved 0.2495 against a target of
#   0.25, while the empirical 0.75-quantile of that batch's realized gain is 0.7046 -- the
#   controller finds its analytic fixed point to 0.3%. An independent reviewer reproduced
#   the same property on a different frozen batch: delta 0.6778, achieved 0.2490, empirical
#   quantile 0.6767, error 0.17%.
#
#   THE BAND AND ITS ANTI-WINDUP, MEASURED. A deliberately poisoned state of log_delta + 50
#   is caught at delta 10.0183 (exp(log(spread) + 2.3), i.e. a factor of 10 above v3's
#   choice) with clamped=True, and because the clamp is written BACK into the state the very
#   next iteration reads clamped=False and resumes ordinary control (9.8938, then 9.7709).
#   Integrating from the UNCLAMPED value instead would have taken thousands of iterations to
#   re-enter the band while reporting a clamp the whole way.
#
#   TWO BUGS FOUND IN REVIEW, BOTH IN THE CONTROLLER, BOTH FIXED, BOTH DISCLOSED.
#     (i) the windup above -- the first draft used the clamp only at read time and kept
#         integrating the unclamped state.
#     (ii) A NON-FINITE STATE LATCHED THE WHOLE RUN AND WAS INVISIBLE. Measured on the real
#         sf_query with one NaN in the batch: spread -> NaN, so every band comparison was
#         False (NaN compares False both ways), min(max(NaN, lo), hi) returned NaN,
#         math.exp(NaN) = NaN, and the state stayed NaN for every subsequent iteration --
#         with sf/aspire_clamp_frac still reading 0.0000. v3 cannot have this failure
#         because it rebuilds delta from scratch each iteration. The fix drops back to
#         v3's self-healing expression on a non-finite state or band, reseeds from it, and
#         COUNTS IT AS A CLAMP so the tag shows it. MEASURED RECOVERY, driven through the
#         real sf_query: two NaN batches give delta NaN with clamped=True BOTH times (so the
#         tag now shows what the first draft hid), then the first CLEAN batch reseeds to
#         delta 1.0044 = occ_boost * spread with clamped=True, and the iteration after that
#         reads clamped=False and resumes ordinary control. Fully self-healing.
#   Both fixes are measured behaviour-neutral on the runs reported here: the FINAL file's
#   --aspire run matches the pre-fix --aspire run on 36 algorithmic tags / 672 points / 0
#   differing, and the FINAL file's --no-aspire run still matches unmodified v3 on 33
#   algorithmic tags / 624 points / 0 differing. That is expected -- sf/aspire_clamp_frac is
#   0 at every iteration and neither new branch fires in this configuration. They matter for
#   a long run, which is the run this arm is built for.
#
#   NO NaN OR Inf IN ANY TAG OF ANY RUN: 12 runs across the three arms, 8112 scalar points,
#   zero non-finite values.
#
#     tag                        v3 START -> END       aspire START -> END    aspire min..max
#     sf/chan_action_r2           0.0709 ->  0.2078     0.0709 ->  0.1753    0.0709.. 0.2125
#     sf/state_action_r2          0.0036 ->  0.1435     0.0036 ->  0.1605    0.0036.. 0.1605
#     sf/cond_action_r2           0.0923 ->  0.3478     0.0923 ->  0.3361    0.0923.. 0.3361
#     sf/phi_eff_rank            15.7204 -> 20.6792    15.7204 -> 21.7424   15.7204..24.5310
#     sf/psi_r2                   0.0001 ->  0.2124     0.0001 ->  0.2525    0.0001.. 0.5264
#     sf/psi_bias_frac            0.0000 ->  0.0048     0.0000 ->  0.0019    0.0000.. 0.0899
#     sf/w_r2                    -0.8893 ->  0.9599    -0.8893 ->  0.9578   -0.8893.. 0.9631
#     sf/delta                    0.5692 ->  0.6510     0.5692 ->  0.5240    0.5240.. 0.5692
#     sf/query_support            0.1009 ->  0.1725     0.1009 ->  0.1519    0.0865.. 0.1519
#     sf/teacher_conc_ratio       1.0005 ->  0.9937     1.0005 ->  0.9933    0.9933.. 1.0100
#     sf/channel_nats             0.1822 ->  0.6766     0.1822 ->  0.6748    0.1822.. 0.8436
#     debug/cond_gap              0.0099 ->  0.0089     0.0099 ->  0.0089    0.0089.. 0.0155
#     losses/distill_kl           0.0095 ->  0.0200     0.0095 ->  0.0164    0.0095.. 0.0319
#     losses/clone_nll           -0.6687 -> -4.9724    -0.6687 -> -4.1897   -4.1897..-0.6687
#     losses/entropy             -0.4692 -> -4.5496    -0.4692 -> -3.7944   -3.7944..-0.4692
#     losses/explained_variance    0.0002 ->  0.3453     0.0002 ->  0.3589   0.0002.. 0.4256
#     sf/aspire_achieved             --                  0.2283 ->  0.0955    0.0459.. 0.2388
#     sf/aspire_log_delta            --                 -0.5635 -> -0.6462   -0.6462..-0.5635
#     sf/aspire_clamp_frac           --                  0.0000 ->  0.0000    0.0000.. 0.0000
#
#   A1 SPLITS, AND THE HALF THAT FAILS FAILS FOR A MEASURABLE REASON.
#     PASSES: sf/aspire_clamp_frac is 0.0000 at every one of the 16 iterations. The band
#       never bound; the controller never went near a runaway.
#     UNMET AS STATED: sf/aspire_achieved does NOT converge to 0.25 in 16 iterations. It
#       reads 0.2283, 0.0735, 0.0459, 0.1411, 0.2388, 0.2102, 0.2302, 0.1907, 0.2288,
#       0.1184, 0.0935, 0.0759, 0.0557, 0.0940, 0.0713, 0.0955 -- it recovers to within
#       0.01 of the target for iterations 5-9 and then slides away to ~0.09.
#     THE CAUSE IS THE CONTROLLER'S OWN AUTHORITY, NOT NOISE, and it is arithmetic. The
#       error signal lives in [-0.25, +0.75] and is negative here, so a step is at most
#       0.05 * 0.25 = 0.0125 in log space, ~1.2% of delta per iteration. Over 16 iterations
#       the controller can move delta by at most ~20% and it actually moved it by 8%
#       (0.5692 -> 0.5240, log_delta -0.5635 -> -0.6462). Over the same 16 iterations the
#       upper tail of the realized-gain distribution contracted far harder than that: in
#       the --no-aspire arm, where delta is FIXED, achieved collapses 0.2283 -> 0.0215 with
#       a run minimum of 0.0144. The controller is under-powered against this timescale by
#       roughly an order of magnitude. That is the deliberate design point of
#       aspire_lr = 0.05 -- with the error signal bounded in [-0.25, +0.75] a saturated step
#       is one e-fold per 80 iterations downward or 27 upward, chosen to be slow against the
#       policy and fast enough against an 8M-step run (244 iterations at file defaults,
#       1953 at this batch) -- colliding with a 16-iteration verification budget. IT IS
#       RECORDED AS UNMET. The gain was NOT retuned to make this table look better: tuning
#       a time constant against a 16-iteration proxy is exactly the coefficient search §2
#       forbids, and 16 iterations is the wrong timescale to tune it on.
#     WHAT IT DOES SHOW, stated without the flattering framing: at END the controller arm's
#       request is met 0.0955 of the time against the fixed arm's 0.0215 -- 4.4x MORE OFTEN,
#       which cuts the shortfall from the 0.25 target from 0.229 to 0.155, i.e. 1.5x closer
#       in absolute terms -- and on trajectories that had already diverged for 15 iterations,
#       the same caveat A3 carries. That every step had the right SIGN is arithmetic, not
#       evidence: achieved was below target at all 16 iterations, so with the band never
#       binding every step had to be a decrease and the delta series is monotone by
#       construction. What is evidence is the MAGNITUDE, and it is too small: 0.0827 of
#       log-delta moved out of the ~0.20 available, i.e. 41% of the controller's
#       16-iteration authority, against a tail that moved several times further.
#
#   A2 FAILS. STATED PLAINLY: sf/teacher_conc_ratio END 0.9933 against v3's 0.9937, with a
#   full-run range of 0.9933-1.0100. There is no movement. The query still SHIFTS the
#   policy and does not SHARPEN it, and THIS ARM HAS NOT FIXED THE DISCLOSED DEFECT.
#   The controller LOWERED delta rather than raising it (0.5692 -> 0.5240, while v3's rose
#   0.5692 -> 0.6510), which is the pre-registered risk in SIGN; but as noted there the
#   magnitude is small (8%), so the smaller request is not on its own an explanation.
#     THE STRONGER RESULT IS A WITHIN-ARM MAGNITUDE TEST THAT FELL OUT OF THE SAME RUNS,
#     and it costs nothing extra. v3's OWN delta swings 2.21x during its 16 iterations
#     (0.5692 -> 1.2596 at iteration 5 -> 0.6510 at 16) purely because the batch spread
#     moves, and across both arms the request magnitude spans 2.40x (0.5240 to 1.2596).
#     Over that entire 2.40x range teacher_conc_ratio never leaves 0.9933-1.0115. So the
#     concentration response to request magnitude is indistinguishable from zero over every
#     magnitude these runs visited, and recalibrating delta WITHIN THAT RANGE -- which is
#     what this arm does, moving it 8% -- cannot deliver sharpening. Whatever does has to
#     change how the CONDITIONAL responds to the query (the concentration parameterization,
#     or a term that actually pays for confidence), not how big the query is.
#     WHAT IS NOT ESTABLISHED, and the boundary matters: this says nothing about magnitudes
#     outside 2.40x. The +-10x band aspire_log_band permits is untested, and so is the
#     region where query_support (measured 0.1009-0.1725, i.e. the query at ~15% of the
#     channel's own RMS) would stop being small. The direct experiment is cheap and is NOT
#     in this file: one extra teacher forward at 2 * delta on a single iteration would turn
#     "the concentration does not respond" into a measured derivative instead of an
#     endpoint comparison. Also NOT measured here: the mechanism, i.e. that the trunk maps
#     c_query to a mean shift with no concentration component. That is the natural reading
#     of these numbers, not something this file probed.
#     A supporting number, weaker than the above and quoted as such: sf/aspire_achieved in
#     the --no-aspire arm shows v3's fixed request falling to the 97.85th percentile of
#     realized gains (achieved 0.0215, minimum 0.0144), so v3 was asking for something far
#     rarer than the top quartile and still got no sharpening. It is weaker because the
#     percentile is a property of the TAIL's shape, not of the request's size, and the two
#     arms' policies had diverged by then.
#
#   A3 NOT MET AT THE END POINT, MIXED ACROSS THE TRAJECTORY. chan_action_r2 END 0.1753
#   against v3's 0.2078, and read against the state-only control as the prediction requires:
#     channel/state ratio, iterations 1-8 then 9-16
#       v3      19.90 24.62 31.28 18.97  9.95  4.89  4.24 3.58
#               4.01  2.65  2.84  2.41   2.07  1.64  1.40 1.45
#       aspire  19.90 24.62 31.48 24.36 15.59 10.73  6.94 4.20
#               4.33  3.53  3.41  2.72   2.30  1.70  1.27 1.09
#     The aspire arm's ratio is STRICTLY HIGHER for iterations 3-14 (12 of 16 points, by
#     1.6-2.2x at iterations 5-7) and crosses below only at 15-16, where it ends at 1.09
#     against 1.45. AND THAT MID-RUN ADVANTAGE IS THE DENOMINATOR, NOT THE CHANNEL, which a
#     reader would otherwise take the other way: the aspire arm's chan_action_r2 is higher
#     than v3's at only 2 of 16 iterations (3 and 10). At iterations 5-7, where the ratio
#     advantage is largest, the channel reads 0.1730 / 0.1734 / 0.1769 against v3's
#     0.1840 / 0.1835 / 0.2093 -- LOWER -- while the state-only control reads
#     0.0111 / 0.0162 / 0.0255 against v3's 0.0185 / 0.0375 / 0.0494, and the control is
#     lower in the aspire arm at 12 of 16 iterations. This is precisely the confound A3's
#     own pre-registration warns about, arriving from the unexpected side.
#     So the END point regresses on both the absolute and the ratio, the mid-run ratio
#     advantage is not evidence of a better channel, and A3 is recorded as NOT MET. The run
#     maxima (aspire 0.2125, v3 0.2244) sit inside each other's spread; this is ONE seed and
#     16 points, and the two arms differ by 8% in a single scalar plus the chaotic divergence
#     of two RL runs. The honest reading: A3 is unproven, leaning negative at END, with no
#     channel DAMAGE mechanism identified (the channel construction is untouched here).
#
#   A4 NOT TESTED, AS PRE-REGISTERED. debug/cond_gap ends at 0.0089 in both arms (0.008921
#   v3 against 0.008859 aspire -- equal to four decimals, 0.7% apart). The aspire arm reads
#   higher than v3 at exactly iterations 7-15 (e.g. 0.0155 vs 0.0123 at iteration 9, 0.0120
#   vs 0.0091 at 13) and losses/distill_kl at 7 of those same 9 iterations, which is the
#   direction anti-absorption would predict -- AND THAT IS NOT EVIDENCE. The decay this
#   mechanism targets took the parent 1.75M steps; 65,536 steps is 3.7% of that, the
#   16-point series sits inside its own iteration-to-iteration range in both arms, and the
#   controller has used only 41% of its 16-iteration authority (8% of delta out of the ~20%
#   available) by the end. Quoting these nine points as anti-absorption would be exactly the
#   flattering misreading A4 was pre-registered to prevent. The mechanism's central claim
#   remains UNTESTED and needs a long run.
#
#   SPEED AND MEMORY. THE GPU WAS SHARED THE WHOLE SESSION with an unrelated foreign job
#   whose load MOVED, from ~100% utilization / ~20 GiB down to ~3% / ~8.7 GiB, so every SPS
#   figure here is a lower bound AND the between-batch differences are contention, not code.
#   All numbers cumulative @ 65536, batch 4096:
#     back-to-back triple at ~100% foreign load:  v3 577  |  --no-aspire 567  |  --aspire 576
#     interleaved 2 x 3 sweep as the load fell:   v3 1664, 1853  |  --aspire 1611, 1609
#                                                 --no-aspire 1846, 1489
#   THE RANGES OVERLAP COMPLETELY AND NO ARM SEPARATES FROM ANOTHER. Read as an upper bound
#   on the controller's cost, that is all this can support -- and it is enough, because the
#   a priori cost is one dot product that BLOCK D already computed, plus one comparison, one
#   mean and one exp, ONCE PER ITERATION (16 times in this whole run). There is no plausible
#   mechanism for it to be measurable, and the measurement does not contradict that.
#   Do NOT read these against v3's recorded ~9450 SPS as a contention factor: that figure
#   was taken at batch 32768 while these runs use batch 4096, so part of the gap is the 8x
#   smaller batch and part is the foreign job.
#   PEAK VRAM is identical to four decimals in EVERY run of every arm, 0.0877 GiB allocated
#   / 0.1445 GiB reserved, because the controller allocates one 0-dim tensor. NOTE FOR
#   ANYONE RE-CHECKING: VRAM is NOT in the TensorBoard event files -- it was measured out of
#   band by an atexit harness reading torch.cuda.max_memory_allocated/reserved -- whereas
#   every other number above came out of TensorBoard event files, all of which were scratch
#   runs and were deleted after being read; re-running the three commands regenerates them.
#
# ------------------------------------------------------------------------------------
# HONEST SUMMARY OF THIS FILE'S STATUS. The mechanism is built, correct, exactly reversible
# (K = 0 on 33 tags / 624 points), free, and gradient-isolated. Of its four pre-registered
# predictions one passes in half (A1: no clamping ever, but no convergence at this length,
# and the reason is the controller's own time constant rather than noise), one FAILS (A2: no
# sharpening -- and the same runs show that request MAGNITUDE is not the lever, over the
# 2.40x range they visited), one is NOT MET at the end point with a mid-run advantage that
# turns out to be a denominator artifact (A3), and the one the file exists for was
# pre-registered as untestable at this length and remains untested (A4).
# This is a NULL-TO-NEGATIVE result at 65,536 steps. Its value is the A2 redirection --
# stop calibrating the query's size, start changing how the conditional's CONCENTRATION
# responds to it -- plus two controller bugs caught before any GPU slot was spent. It is
# written down as such rather than dressed up.
# =====================================================================================
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 1e-3
    num_envs: int = 16
    num_steps: int = 2048
    # --- throughput. Neither flag changes the algorithm: identical math, identical
    # per-env seeding. Inherited unchanged from the parent, where the acting forward was
    # measured LAUNCH-bound rather than compute-bound. This file's acting path is strictly
    # cheaper than the parent's -- the MTP critic head is gone, so `act` is policy heads
    # only -- and the update path no longer pays for a per-sample autograd Jacobian or a
    # 6x511-bin softmax. Measured, batch held at 32768 so only env count varies, 3 repeats,
    # cumulative @65536 steps / marginal between iterations (parent convention):
    #   16 envs 9451 / 11678   32 envs 10222 / 13664   64 envs 12327 / 17035
    # against the parent's 4300 at 16 envs and 6013 at 128. GPU was shared throughout.
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95      # NO GAE IN THIS FILE. Retained under its historical name
    #                               solely as the default horizon that sf_lambda < 0 reuses,
    #                               so the feature-space lambda-return introduces no new knob
    #                               and stays numerically comparable to the lineage.
    num_minibatches: int = 32
    actor_epochs: int = 1         # passes over the batch for the rationalization + distill
    #                               losses. Reuse here re-fits the SAME action per state, so
    #                               it sharpens the conditional (entropy drops).
    critic_epochs: int = 4        # passes for the SF regression only. Same rationale as the
    #                               parent's split budget, and it survives the critic swap
    #                               unchanged: the SF losses are plain supervised regression
    #                               onto FIXED detached targets, so extra passes only reduce
    #                               fitting error, and they no longer touch the actor trunk
    #                               at all -- so unlike the parent's shared-trunk critic they
    #                               cannot fight the policy update.

    # --- OPSD occupancy conditioning ---
    cond_mode: str = "occ"        # "occ" | "occ_w". THE decisive ablation. occ feeds the
    #                               whole phi_dim-vector occupancy surprise; occ_w feeds ONLY
    #                               its scalar projection onto w, through the same Fourier
    #                               embedding the champion used. Everything else is
    #                               byte-identical, so this is a strictly better controlled
    #                               scalar-vs-vector comparison than the parent's legacy
    #                               advantage channel could ever be. §6.4's diagnosis
    #                               predicts occ_w degrades toward a no-op.
    occ_boost: float = 1.0        # THE QUERY MAGNITUDE, in units of the batch's own spread
    #                               along w. One value, no sweep (§2 forbids coefficient
    #                               grids). 1.0 is the defensible default because the channel
    #                               is a standardized RESIDUAL: a one-sigma request is the
    #                               smallest ask that is not inside the channel's own noise
    #                               and the largest that is still interpolation almost
    #                               everywhere. It also matches the champion's kappa=1, which
    #                               is the value that actually won at 8M (8819 vs kappa=2's
    #                               8249), so the arm is not smuggling in a bigger dose than
    #                               the thing it is compared against.
    #                               ITS ROLE CHANGED IN THIS FILE. Under --aspire it is no
    #                               longer THE query magnitude: it only SEEDS iteration 1
    #                               (so both arms share their first point exactly) and it
    #                               centres the runaway band. From iteration 2 on the
    #                               magnitude is whatever the achievement-rate controller
    #                               says it is. Under --no-aspire it keeps v3's meaning
    #                               exactly. It is NOT swept in either mode.

    # --- aspire v1: ACHIEVEMENT-RATE CALIBRATED QUERY MAGNITUDE ---------------------
    # THE ONE MECHANISM THIS FILE ADDS. v3's occ_boost above is fixed, and its own
    # telemetry disclosed the two consequences: teacher_conc_ratio ~1.00 (the query SHIFTS
    # the policy and barely SHARPENS it, and the family has measured repeatedly that
    # mass-covering distillation against a teacher whose concentration does not change has
    # no sharpening channel at all), and a stationary request that the student can ABSORB
    # (every prior OPSD arm died this way: cond_gap 0.030 -> 0.010 as the gap on a fixed
    # bar closed). The replacement is a scalar feedback controller that holds delta at the
    # (1 - aspire_target) quantile of the student's OWN realized-gain distribution, so the
    # bar moves when the student moves. See BLOCK D for the full derivation.
    aspire: bool = True          # --no-aspire restores v3 EXACTLY (same code path for
    #                              delta, same op order, bit-identical; VERIFIED, see the
    #                              header's REPRODUCTION line). Default ON: this file IS
    #                              the aspire arm and its default must be the thing it
    #                              tests, with the off-switch as the control.
    aspire_target: float = 0.25  # THE ACHIEVEMENT RATE the request is held at. One value,
    #                              no sweep (§2). The top quartile is the defensible
    #                              default because it is the only region that is both
    #                              REACHABLE and NOT TYPICAL: it demonstrably happened,
    #                              25% of the time, under the current policy, so the
    #                              conditional has real data at that magnitude -- while
    #                              still being something the policy does not do by
    #                              default. A target near 0.5 requests the MEDIAN outcome,
    #                              which asks for nothing (the teacher becomes the typical
    #                              student and cond_gap goes to 0 by construction); a
    #                              target near 0.0 requests an outlier the conditional has
    #                              almost no rows for, which is v1's tail-extrapolation
    #                              failure re-derived from the other end.
    aspire_lr: float = 0.05      # Controller step, in log-delta per iteration. One value,
    #                              no sweep. IMPLIED TIME CONSTANT, computed with the
    #                              error signal's ACTUAL bound rather than +-1: the signal
    #                              lives in [-aspire_target, 1 - aspire_target], so a
    #                              saturated step is 0.0125 DOWN or 0.0375 UP, i.e. one
    #                              e-fold in 80 iterations downward or 27 upward. That is
    #                              deliberately SLOW against the policy -- 80 iterations is
    #                              80 rollouts and 2560-10240 actor optimizer steps
    #                              depending on actor_epochs, so the controller cannot
    #                              chase within-iteration policy noise and become a second
    #                              learning rate -- and fast ENOUGH against training
    #                              length: an 8M-step run is 244 iterations at this file's
    #                              defaults and 1953 at the verification batch, so the
    #                              controller can travel exp(-244 * 0.0125) = 0.047x at the
    #                              worst, which is ample. MEASURED CONSEQUENCE, disclosed:
    #                              over the 16-iteration verification run this is far too
    #                              slow to track -- see A1 in the header. The gain was NOT
    #                              retuned to fix that, because tuning a time constant
    #                              against a 16-iteration proxy is the coefficient search
    #                              §2 forbids and 16 iterations is the wrong timescale.
    aspire_log_band: float = 2.3 # RUNAWAY DETECTOR, not a stabilizer (BLOCK D proves the
    #                              feedback contracts globally). Log-space half-width of
    #                              the band delta is confined to, centred on v3's own
    #                              choice log(std(w_unit . c)); 2.3 ~ log(10), so delta
    #                              may travel a factor of 10 either side of v3 before the
    #                              band binds and sf/aspire_clamp_frac reports it. The two
    #                              ends are the two documented failure modes: below
    #                              spread/10 the request is inside the channel's own noise
    #                              (v3's argument for one sigma being the SMALLEST
    #                              meaningful ask), and above spread*10 it is a 10-sigma
    #                              request that at batch 4096 no row can have realized, so
    #                              achieved is identically 0 and the tag would otherwise
    #                              show a silent slide instead of a saturation.
    cond_clip: float = 3.0        # clamp on the standardized channel fed to the trunk
    cond_ema_beta: float = 0.99   # EMA horizon for the channel's per-dim RMS (~100 iters)
    cond_embed_freqs: int = 8     # sinusoidal features per phase, occ_w only
    clone_coef: float = 1.0       # weight on the rationalization fit p(a | s, c)
    distill_coef: float = 1.0     # weight on the per-dim teacher->student divergence
    distill_kl_clip: float = 2.0  # tau: the paper's per-token pointwise divergence clip

    # --- v1: SUCCESSOR-FEATURE OCCUPANCY CRITIC ------------------------------------
    phi_dim: int = 32             # Dp. 32 is 2x HalfCheetah's action dim and ~2x its obs
    #                               dim, i.e. enough basis to span the reward AND retain
    #                               geometry, small enough that the covariance telemetry
    #                               (eigvalsh on 32x32) is free.
    sf_hidden: int = 0            # 0 = use args.hidden. Same width as the trunk, so the SF
    #                               heads are a rounding error next to the env loop.
    sf_lambda: float = -1.0       # < 0 = reuse gae_lambda. The feature-space lambda-return's
    #                               horizon. Kept as a knob because it is the one quantity
    #                               that trades channel action-attributability (short) against
    #                               occupancy horizon (long) -- but it is NOT swept here.
    sf_coef: float = 0.5          # psi TD regression. Half the reward term because psi's
    #                               target has the larger magnitude by ~1/(1-gamma*lam) and
    #                               would otherwise dominate the shared phi gradient.
    sf_rew_coef: float = 1.0      # w . phi -> r. This is what makes w mean anything, and w
    #                               IS the improvement direction, so it gets full weight.
    sf_cov_coef: float = 1.0      # phi decorrelation. Full weight: rank collapse is one of
    #                               the two named hazards and the penalty is bounded in
    #                               [0, 1] by construction (it is squared correlations), so
    #                               it cannot outrun the regression terms.
    sf_grad_clip: float = 0.5     # matches max_grad_norm; separate knob because the SF
    #                               module has its own optimizer and must not be able to
    #                               change what the actor's clip does.

    max_grad_norm: float = 0.5

    normalize_reward: bool = False
    clip_reward: bool = False

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(  # pyright: ignore[reportCallIssue]
            env, lambda observation: np.asarray(observation).clip(-10, 10)
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(
                env, lambda reward: min(max(float(reward), -10.0), 10.0)
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form."""

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            self.blocks.append(ThinkBlock(H * (k + 1), H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class AdvEmbed(nn.Module):
    """Fixed Fourier features of a SCALAR privileged channel. Used only by
    --cond-mode occ_w, where the channel is the occupancy surprise's projection onto w.

    A single raw scalar channel among 17 observation dims is trivially IGNORABLE: the
    rationalization loss can be driven down almost entirely by modelling the marginal
    p(a|s) and dropping it, because the extra likelihood it buys is small. Measured on
    HalfCheetah with raw scalar conditioning: cond_gap 0.001 and distill_kl 0.0001, i.e.
    the teacher WAS the student and the method was a no-op. Sinusoidal features fix this
    the way diffusion timestep embeddings do -- the scalar now occupies many channels and
    separates nearby values at high frequency, so it is both easy to use and expensive to
    ignore. Frequencies are fixed, not learned, so the channel cannot be switched off by
    driving weights to zero.

    RETAINED DELIBERATELY, because it makes occ_w the STRONGEST possible scalar arm. If
    the vector channel wins anyway, the win cannot be attributed to the scalar arm having
    been given a lazy encoding.
    """

    def __init__(self, n_freq):
        super().__init__()
        self.n_freq = n_freq
        self.freqs: torch.Tensor
        self.register_buffer("freqs", torch.logspace(-1.0, 3.0, n_freq, base=2.0))

    @property
    def dim(self):
        return 2 * self.n_freq

    def forward(self, chan):
        x = chan * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# =====================================================================================
# BLOCK A -- THE phi / psi / w MODULE. Sibling arms replace pieces of this and nothing
# else, so it is kept whole and self-contained.
# =====================================================================================
class SFCritic(nn.Module):
    """Successor features on RAW obs+action: phi(s,a), psi(s,a), and the reward readout w.

    DELIBERATELY NOT ON THE SHARED TRUNK, for exactly the reason the parent gave for its
    TransitionHead: routing these gradients through the actor's representation would
    confound "the occupancy channel helps" with "predicting the future is a good auxiliary
    task", and this family has already been burned by that class of confound. Separate
    module, separate optimizer, separate grad clip, and the isolation is MEASURED (see the
    header's GRADIENT ISOLATION line) rather than assumed.

    phi ends in a NON-AFFINE LayerNorm. Two facts about that, both load-bearing:
      - It is a PER-SAMPLE operation, so it introduces no batch-statistic nondeterminism
        and no train/eval divergence -- unlike BatchNorm, which would make phi depend on
        who else is in the minibatch and quietly couple the channel to the shuffle.
      - It kills SCALE collapse (phi cannot shrink toward zero to cheat the decorrelation
        penalty, nor blow up to cheat the reward fit) but NOT RANK collapse: a LayerNormed
        output can still live on a 1-dim curve. Rank is the decorrelation penalty's job,
        and even that is necessary-not-sufficient (uncorrelated noise has full rank).

    psi has no LayerNorm -- it is an unbounded discounted SUM of phi's, so normalizing it
    would destroy exactly the magnitude information the value readout needs -- and its
    final layer starts at std=0.01 so the bootstrap begins near zero rather than injecting
    a random occupancy field into iteration 1's channel.

    The channel's standardization buffers mirror TransitionHead.update_stats' EMA pattern
    for its stated reason: MuJoCo observation blocks differ ~26x in action sensitivity
    (measured median |ds/da| 0.0052 on qpos rows vs 0.1359 on qvel rows), so an
    unstandardized channel would be dominated by one block and the trunk would see 32
    channels of which a few carry all the amplitude.
    """

    def __init__(self, obs_dim, act_dim, phi_dim, hidden):
        super().__init__()
        self.phi_dim = phi_dim
        self.phi_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, phi_dim)),
            nn.LayerNorm(phi_dim, elementwise_affine=False),
        )
        self.psi_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, phi_dim), std=0.01),
        )
        # NOT zero-initialized, unlike the parent's critic head: w_unit = w/||w|| is the
        # improvement direction and must be well defined at iteration 1. PyTorch's default
        # uniform init is exactly the right thing here -- a random unit direction that the
        # reward regression then rotates into place (measured: w_r2 -0.68 -> 0.956 in 16
        # iterations, with sf_rew_mse 0.1135 -> 0.0056).
        self.w_head = nn.Linear(phi_dim, 1, bias=True)
        # Per-dim mean SQUARE of the channel, not mean and variance. The family measured
        # that a conditioning input must keep its natural zero: c = 0 means "the future came
        # out exactly as predicted", and subtracting a batch/EMA mean relabels the
        # least-bad row of an all-bad batch as positive -- a false sign. Measured on the
        # champion: ema_rms 5365 @2M vs batch-standardized 4390 vs raw 2368.
        self.c_ms: torch.Tensor
        self.register_buffer("c_ms", torch.ones(phi_dim))
        # v2: PSI TARGET STANDARDIZATION. v1 measured psi_r2 -1.7557 against a >0.5 bar with
        # psi_bias_frac 0.5974, i.e. 60% of the residual was pure per-dim LEVEL error. The
        # cause is arithmetic, not a modelling failure: LayerNorm centers phi ACROSS dims,
        # so each phi dim keeps a nonzero per-dim mean, and the lambda-return's fixed point
        # multiplies it by 1/(1 - gamma*lam) ~ 17.5. A head initialized at std=0.01 outputs
        # ~0 and therefore has to travel ~17 per dim on 512 Adam steps before R2 can even
        # reach 0. So psi_net now predicts in STANDARDIZED target space and the level is
        # supplied by a buffer -- the head starts at R2 = 0 (predicting the target's mean)
        # instead of R2 = -3.3, and only has to learn VARIATION, which is what R2 scores.
        # This follows the family's own TransitionHead precedent, and for the same stated
        # reason: an unstandardized MSE over dims whose scales differ fits only the loud ones.
        # Mean-and-variance here, unlike c_ms's mean-square-only: a REGRESSION TARGET has no
        # natural zero to protect, whereas the CHANNEL does (c = 0 means "the future came out
        # as predicted") -- that is why the two use different standardizations.
        self.psi_mu: torch.Tensor
        self.psi_var: torch.Tensor
        self.register_buffer("psi_mu", torch.zeros(phi_dim))
        self.register_buffer("psi_var", torch.ones(phi_dim))

    def phi(self, obs, action):
        return self.phi_net(torch.cat([obs, action], dim=-1))

    def psi_std(self, obs, action):
        """psi in STANDARDIZED target space. This is what the regression loss compares, so
        every feature dim contributes equally regardless of its level or scale."""
        return self.psi_net(torch.cat([obs, action], dim=-1))

    def psi(self, obs, action):
        """psi in RAW feature-occupancy units -- the public readout. Every consumer (the
        bootstrap in BLOCK B, the channel in BLOCK C, the value readout) uses this, so v2
        changes the PARAMETERIZATION of psi and its loss scaling, never its semantics."""
        return self.psi_std(obs, action) * self.psi_std_dev() + self.psi_mu

    def psi_std_dev(self):
        return self.psi_var.sqrt().clamp_min(1e-6)

    def standardize_psi_target(self, tgt):
        return (tgt - self.psi_mu) / self.psi_std_dev()

    @torch.no_grad()
    def update_psi_stats(self, tgt, beta):
        """Per-dim mean and variance of psi's target. Called with beta=0 (pure batch
        statistics) from a single site placed AFTER the channel is built; see the call site
        for why both of those are load-bearing and were wrong in v2. The beta argument is
        kept so the EMA form stays available without a signature change.

        NOT PopArt: the output layer is not compensated when the stats move. That is
        deliberate and it is safe here for a measured reason -- phi is LayerNorm'd, so the
        target's fixed-point scale is bounded by phi's scale times 1/(1 - gamma*lam) and the
        statistics are near-stationary; the one large update is the iteration-1 warmup, which
        precedes any learning. If sf/psi_r2 ever shows a sawtooth synchronized with the EMA,
        PopArt-style weight compensation is the escalation, not a larger beta."""
        self.psi_mu.mul_(beta).add_((1.0 - beta) * tgt.mean(0))
        self.psi_var.mul_(beta).add_((1.0 - beta) * tgt.var(0))

    def value(self, obs, action):
        """V(s,a) = w . psi(s,a). Linear in the occupancy measure BY CONSTRUCTION, which is
        the entire point: grad_psi V = w, constant, never vanishing.

        This is the module's public value readout and part of the contract the sibling arms
        copy. The update loop deliberately does NOT call it: losses/explained_variance needs
        the SAME linear functional applied to psi AND to psi's target, and psi is already in
        hand there, so the telemetry applies self.w_head directly rather than paying for a
        second psi forward over the whole batch."""
        return self.w_head(self.psi(obs, action)).squeeze(-1)

    def w_vec(self):
        """The reward direction in feature space, detached. Improvement points along w."""
        return self.w_head.weight.detach().reshape(-1)

    def chan_std(self):
        return self.c_ms.sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def update_chan_stats(self, c_raw, beta):
        self.c_ms.mul_(beta).add_((1.0 - beta) * c_raw.square().mean(0))


class Agent(nn.Module):
    """One network, two contexts. The privileged block is the TRAILING input channels.

    Present  -> the standardized occupancy surprise (occ), or Fourier features of its
                projection onto w (occ_w). Teacher context.
    Absent   -> that whole block is zeroed. Student context, and the acting context.
    Zeroing rather than feeding a zero-valued channel through an embedding keeps "no
    privileged information" a distinct code, since cos(0)=1 means the embedding of zero is
    not the zero vector. In occ mode zeroing and feeding zero coincide (the encoding is the
    identity) -- and that coincidence is itself meaningful: c = 0 already means "as
    predicted", so the student context is the honest neutral query rather than an
    out-of-distribution code.

    NO CRITIC HEAD. Value is w . psi from SFCritic, so the trunk carries no value
    regression at all. That removes the parent's measured shared-trunk conflict (four
    passes of actor gradient left its critic WORSE: value_loss 24.9 vs 20.5) as a matter of
    structure rather than of tuning.
    """

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        if args.cond_mode == "occ":
            self.adv_embed = None
            self.cond_dim = args.phi_dim
        elif args.cond_mode == "occ_w":
            self.adv_embed = AdvEmbed(args.cond_embed_freqs)
            self.cond_dim = self.adv_embed.dim
        else:
            raise ValueError(f"unknown cond_mode {args.cond_mode!r}")
        self.trunk = ThinkTrunk(obs_dim + self.cond_dim, H, args.k_blocks, args.n_experts)
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer(
            "action_low",
            torch.as_tensor(envs.single_action_space.low, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(envs.single_action_space.high, dtype=torch.float32).reshape(-1),
        )

    def _feat(self, obs, cond):
        return self.trunk(torch.cat([obs, cond], dim=-1))

    def cond_present(self, chan):
        """Privileged context. occ: the standardized occupancy-surprise vector itself.
        occ_w: Fourier features of its scalar projection onto w. Clamping happens where the
        channel is built, so this is pure encoding."""
        return chan if self.adv_embed is None else self.adv_embed(chan)

    def _zero_cond(self, obs):
        """Privileged context ABSENT: the whole block is zero."""
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def policy(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def act(self, obs):
        """Acting policy == the STUDENT context (privileged slot zeroed)."""
        alpha, beta = self.policy(obs, self._zero_cond(obs))
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        return self.z_to_action(z), z

    def mean_action(self, obs):
        """a_bar(s): the STUDENT's mean action. This is what psi bootstraps on, and it is
        what future_pred is evaluated at -- so the channel's predicted term carries no
        information about the action actually taken. The Beta mean is used rather than a
        sample so the channel is a deterministic function of the state, adding no sampling
        noise on top of the environment's own."""
        alpha, beta = self.policy(obs, self._zero_cond(obs))
        return self.z_to_action(alpha / (alpha + beta))


def _chunked(fn, *tensors, chunk):
    """Apply `fn` over row chunks. The full batch is 32768 rows; chunking at minibatch_size
    mirrors the parent's teacher query and keeps peak activation memory flat in batch size."""
    n = tensors[0].shape[0]
    return torch.cat([fn(*[t[i : i + chunk] for t in tensors]) for i in range(0, n, chunk)])


def phi_corr(phi):
    """Batch CORRELATION matrix of phi. Correlation, not covariance: LayerNorm already
    fixes scale, so penalizing covariance would double-charge for amplitude."""
    x = phi - phi.mean(0, keepdim=True)
    sd = x.square().mean(0).sqrt().clamp_min(1e-6)
    xn = x / sd
    return (xn.T @ xn) / x.shape[0]


def offdiag_decorr(phi):
    """Mean squared OFF-DIAGONAL entry of phi's batch correlation matrix.

    The diagonal is not penalized because it is identically 1 as a FUNCTION of phi, so its
    contribution carries exactly zero gradient and would only add a constant. Bounded in
    [0, 1], which is why sf_cov_coef can be 1.0 without any risk of outrunning the
    regression terms.

    THERE IS A HARD STRUCTURAL FLOOR AND IT MUST BE QUOTED WHENEVER THIS TAG IS READ.
    phi's non-affine LayerNorm forces sum_i phi_i(x) = 0 for EVERY sample, so
    Var(sum_i phi_i) = 0, so the off-diagonal correlations must sum to -d: at equal per-dim
    variance the mean off-diagonal correlation is exactly -1/(d-1). For phi_dim = 32 that
    puts mean|offdiag| >= 1/31 = 0.0323 and this penalty >= 1/31^2 = 0.00104 no matter what
    the network does. So a small value here means "decorrelated up to the constraint", NOT
    "the penalty did all the work", and sf/phi_offdiag_absmean must be compared to 0.0323
    rather than to 0."""
    corr = phi_corr(phi)
    d = corr.shape[0]
    off = corr - torch.diag_embed(torch.diagonal(corr))
    return off.square().sum() / (d * (d - 1))


def _ls_r2(feats, targets, ridge=1e-6):
    """Pooled R2 of the best LINEAR predictor of `targets` from `feats` (+ intercept).

    Float64 normal equations with a TRACE-RELATIVE ridge, so the number is scale-free and
    stays finite when a channel dim is degenerate -- which matters because these are the
    file's leakage and information gates and a NaN there would read as "no finding".
    In-sample by construction; with <= 50 features against 4096-32768 rows the optimistic
    bias is O(p/n) <= 1.2%, well below the thresholds it is compared against."""
    x = torch.cat([feats, torch.ones_like(feats[:, :1])], dim=-1).double()
    y = targets.double()
    xtx = x.T @ x
    xtx = xtx + (ridge * torch.diagonal(xtx).mean()) * torch.eye(
        xtx.shape[0], device=xtx.device, dtype=xtx.dtype
    )
    coef = torch.linalg.solve(xtx, x.T @ y)
    sse = (y - x @ coef).square().sum()
    sst = (y - y.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float((1.0 - sse / sst).item())


def _r2_per_dim(pred, target):
    """R2 of `pred` against `target`, averaged over the last dimension."""
    sse = (target - pred).square().sum(0)
    sst = (target - target.mean(0, keepdim=True)).square().sum(0).clamp_min(1e-12)
    return float((1.0 - sse / sst).mean().item())


# =====================================================================================
# BLOCK B -- THE FEATURE-SPACE LAMBDA-RETURN (psi's regression target).
# =====================================================================================
@torch.no_grad()
def sf_feature_targets(sf, agent, b_obs, b_next_obs, b_act, nonterm, lam_nonterm, lam, gamma, chunk):
    """phi at the taken action, psi's regression target, and the realized future footprint.

    This is a lambda-return ON FEATURES. THERE IS NO ADVANTAGE ANYWHERE IN IT: nothing is
    subtracted from anything, there is no baseline, and the result is a target for a
    vector regression rather than a weight on a gradient. That is what replacing a scalar
    critic with a vector one buys -- with V linear in psi there is no scalar residual left
    for an advantage to be.

    WHAT OPERATOR IS ACTUALLY BEING REGRESSED, STATED EXACTLY. At lam = 0 the target is the
    one-step occupancy flow constraint
        psi(s,a) = phi(s,a) + gamma * psi(s', a_bar(s')),
    i.e. "the footprint of this step plus the footprint of everything downstream". At
    lam > 0 the bootstrap is a lam-BLEND of that MEAN-action successor feature and the
    realized SAMPLED-action chain (`carry` carries phi(s_{t+k}, a_{t+k}) at the actions the
    policy actually took), so the fixed point is the blended operator's, and it coincides
    with the mean-action flow constraint only at lam = 0 or for a deterministic policy. The
    blend is deliberate -- the sampled chain is where the channel's action-attributable
    information comes from -- but it is not the textbook SF fixed point and should not be
    described as one.

    Either way it is a CONSERVATION LAW rather than a fitted optimum: contraction in
    gamma*lam-weighted expectation, unique fixed point at a fixed policy, and so NO target
    network is required. The usual reason for one is a max/argmax backup, and there is none
    here -- the bootstrap is the student's own mean action, never an optimized one.

    Boundary handling is the parent's, exactly:
      nonterm[t]     = (1 - terminations[t]) * valids[t]   -- may we bootstrap at all
      lam_nonterm[t] = 1 - boundaries[t]                   -- may the lambda chain cross t
    so a truncation bootstraps off the recorded final observation and a true termination
    contributes phi alone. NOTE the parent scales its carry by lam_nonterm ONLY while this
    multiplies the whole blended bootstrap by nonterm[t]; the two agree because the rollout
    guarantees nonterm[t] == 0 => lam_nonterm[t] == 0 (both a termination and a missing
    final observation imply boundaries[t] == 1). That invariant is what makes the shorter
    form here equivalent, so it is written down rather than assumed.
    """
    num_steps, num_envs = nonterm.shape
    phi_taken = _chunked(sf.phi, b_obs, b_act, chunk=chunk)
    a_bar_next = _chunked(agent.mean_action, b_next_obs, chunk=chunk)
    psi_next = _chunked(sf.psi, b_next_obs, a_bar_next, chunk=chunk)

    shape = (num_steps, num_envs, sf.phi_dim)
    phi_s = phi_taken.view(shape)
    psi_n = psi_next.view(shape)
    nt = nonterm.unsqueeze(-1)
    ln = lam_nonterm.unsqueeze(-1)

    tgt = torch.zeros_like(phi_s)
    future = torch.zeros_like(phi_s)
    carry = torch.zeros(num_envs, sf.phi_dim, device=phi_s.device)
    for t in reversed(range(num_steps)):
        # t == num_steps - 1 has no tgt[t+1] to mix with, so the chain weight is 0 there.
        lam_eff = 0.0 if t == num_steps - 1 else lam * ln[t]
        boot = (1.0 - lam_eff) * psi_n[t] + lam_eff * carry
        future[t] = gamma * nt[t] * boot
        tgt[t] = phi_s[t] + future[t]
        carry = tgt[t]
    return phi_taken, tgt.reshape(-1, sf.phi_dim), future.reshape(-1, sf.phi_dim)


# =====================================================================================
# BLOCK C -- THE PRIVILEGED CHANNEL: the occupancy surprise. The load-bearing part.
# =====================================================================================
@torch.no_grad()
def sf_channel(sf, agent, b_obs, future_realized, chunk, ema_beta, warmup):
    """c = (realized future footprint) - (predicted future footprint), per-dim standardized.

    THE CHANNEL IS NOT `tgt - psi(s,a)`. That leaks: tgt contains phi(s_t, a_t) EXPLICITLY,
    so the network could invert phi to recover a_taken and collapse clone_loss to the
    identity map (the parent's own warning, its lines ~327-333). Instead both terms are
    FUTURE-ONLY:

        future_realized[t] = gamma * nonterm[t] * boot[t]          == tgt[t] - phi(s_t,a_t)
        future_pred[t]     = psi(s_t, a_bar(s_t)) - phi(s_t, a_bar(s_t))
        c_raw[t]           = future_realized[t] - future_pred[t]

    phi(s_t, a_t) cancels ALGEBRAICALLY out of the first line, and the second is evaluated
    at the student's MEAN action, so a_t reaches c ONLY through the environment's response
    to it: s_{t+1}, and (on envs that terminate) the termination flag, which zeroes the
    whole future term. There is no closed-form inversion path from c back to a_t -- the
    network would have to invert the simulator. That is exactly the legitimate hindsight
    privilege §6.4 measured at R2(action | s_{t+1}) = +0.4661, and it is the difference
    between privileged information and leakage. Both are audited every iteration:
    sf/chan_action_r2 says the privilege is there, sf/cond_action_r2 says it is not a
    giveaway -- and both are LINEAR probes, so they bound recoverability from below while
    the trunk is nonlinear. A LINEAR gate is the right instrument anyway, because §6.4's
    0.0000 was measured the same way and the comparison is the point.

    Note `future_pred` subtracts phi at the SAME mean action psi was evaluated at, not at
    the taken action. Using the taken action there would reintroduce phi(s_t, a_t) with a
    sign, i.e. reintroduce the leak in the term meant to remove it.

    WHAT DOES NOT FULLY CANCEL, MEASURED. psi carries a large per-dim LEVEL (see
    sf/psi_bias_frac). A constant level L enters future_pred with weight 1 but enters
    future_realized only with weight gamma*(1-lam)/(1-gamma*lam) = 0.832 at gamma 0.99,
    lam 0.95, so 16.8% of L survives in c_raw as a CONSTANT OFFSET (1.0% at lam = 0). It is
    invisible to the R2 gates -- their intercept absorbs it -- but it does shift the
    channel's natural zero, which is the thing the RMS-only standardization and the
    absolute query in BLOCK D both lean on, and it inflates c_ms. So "the level cancels" is
    a 83% cancellation, not an identity, and the residual shrinks as psi's level converges.
    """
    a_bar = _chunked(agent.mean_action, b_obs, chunk=chunk)
    psi_bar = _chunked(sf.psi, b_obs, a_bar, chunk=chunk)
    phi_bar = _chunked(sf.phi, b_obs, a_bar, chunk=chunk)
    c_raw = future_realized - (psi_bar - phi_bar)
    # Updated from THIS batch before standardizing. Unlike the parent's held-out jac_r2,
    # this is a normalizer and not a diagnostic, so there is nothing to keep held out; and
    # iteration 1 must not divide by the init value of 1.0 when the true scale differs by
    # orders of magnitude. beta = 0 on the first iteration is the parent's own warm-start
    # convention (jac_model.update_stats(..., 0.0 if iteration == 1 else 0.99)).
    sf.update_chan_stats(c_raw, 0.0 if warmup else ema_beta)
    return c_raw, c_raw / sf.chan_std()


# =====================================================================================
# BLOCK D -- THE IMPROVEMENT QUERY: a half-space request, not a gradient step.
# =====================================================================================
@torch.no_grad()
def sf_query(sf, c, args, log_delta):
    """c_query = delta * w_unit. v3 fixed delta; THIS FILE CALIBRATES IT BY ACHIEVEMENT.

    SEMANTICS OF THE QUERY ITSELF (unchanged from v3): "what would you have done if your
    future footprint had come out `delta` better along the reward-increasing direction
    than you predicted." The channel is already a RESIDUAL with a meaningful zero -- c = 0
    means "exactly as predicted" -- so an ABSOLUTE query in c-space is a PER-STATE
    RELATIVE request in footprint space. That is what v1 lacked (a global constant query
    in absolute advantage units, which was 3-sigma extrapolation for badly-scoring states)
    and what v2 had to hack around by adding a margin to each state's own realized delta.

    WHY THIS DOES NOT VANISH. Value is LINEAR in psi, so the improvement direction is
    exactly w -- constant, and independent of how sharp the policy has become. Every
    scalar-gradient mechanism in this family carries the opposite property: the parent's
    channel got 3.7x louder and 3x less used over 1.75M steps because its direction was a
    score function whose signal-to-noise decays as the policy concentrates. There is
    nothing in the DIRECTION here to decay. The MAGNITUDE is a different story, and it is
    what this file changes.

    ================================================================================
    THE DEFECT THIS FILE FIXES. v3's delta = occ_boost * std(w_unit . c) with occ_boost
    fixed at 1.0 leaves two things on the table. MEASURED ON v3 IN THIS SESSION at the
    16-iteration configuration in the header (v3's own header table records 0.9963 ->
    1.0021 and 0.0106 -> 0.0081 for the v1 arm; these are the v3 numbers):

      sf/teacher_conc_ratio  1.0005 -> 0.9937   the teacher SHIFTS and does not SHARPEN
      debug/cond_gap         0.0099 -> 0.0089   and in the parent, 0.030 -> 0.010 over
                                                1.75M steps: the student ABSORBS a
                                                stationary request and the drive ends.

    One sigma of the batch's own spread is already a relative reference in SCALE, but it
    is NOT a relative reference in DIFFICULTY. Nothing in `std` knows whether one sigma is
    something the policy manages constantly or never. So the request can be absorbed
    (achieved almost always -> no information left in it) or be unreachable (achieved
    almost never -> the conditional has no data there) and v3 cannot tell which, because
    it never asks.

    THE FIX: ask. `achieved` is the fraction of the batch that actually realized the gain
    that was requested, and it is free -- the realized projection w_unit . c is already in
    hand, and this is one comparison and one mean over it. Then integrate in LOG space:

      achieved   = mean( (c @ w_unit) > delta )
      log_delta <- log_delta + aspire_lr * (achieved - aspire_target)
      delta      = exp(log_delta)

    WHY LOG SPACE. delta is a magnitude and must stay positive; exp enforces that with no
    clamping and no special case. It also makes the controller's steps SCALE-FREE: a step
    of aspire_lr is a fixed MULTIPLICATIVE move in delta, so the same gains work whether
    the channel's RMS is 0.1 or 10, which matters because the channel's EMA normalizer and
    psi's standardization both change its scale during a run.

    WHAT THE FIXED POINT IS, EXACTLY. The update has zero expected drift iff
    P(w_unit . c > delta) = aspire_target, i.e. delta converges to the
    (1 - aspire_target) QUANTILE of the student's own realized-gain distribution. That is
    the structural property the fixed target lacked: the bar is re-derived from current
    capability every iteration, so ABSORPTION RAISES THE BAR instead of ending the drive.
    A student that gets better at realizing gains along w pushes its own achievement rate
    up, and the controller answers by asking for more.

    WHY THE FEEDBACK IS GLOBALLY STABLE, so the clamp below is a DETECTOR and not a
    stabilizer. achieved is monotonically NON-INCREASING in delta, so the error signal
    (achieved - target) is monotonically non-increasing too: too-large delta gives
    achieved -> 0 and drives log_delta DOWN by aspire_lr * aspire_target per iteration;
    too-small delta gives achieved -> 1 and drives it UP by aspire_lr * (1 - target). This
    is exactly Robbins-Monro stochastic quantile estimation and it contracts from any
    initialization. There is no gain for which it can diverge, only a step size for which
    it is noisy -- which is what aspire_lr is chosen small for.

    THE BAND, AND WHY IT IS RELATIVE. Stable in expectation is not the same as bounded in
    finite samples, and a runaway must be VISIBLE rather than silently reinterpretable, so
    delta is confined to a factor of exp(aspire_log_band) either side of v3's own choice
    (log of the batch spread along w). Anchoring the band to the spread rather than to an
    absolute number keeps it meaningful as the channel's scale moves, and it makes the
    reported clamp fraction read "the controller left v3's neighbourhood", which is the
    interesting event. ANTI-WINDUP: the clamped value is what the state BECOMES, not merely
    what this iteration uses. Integrating from the UNCLAMPED value would let the state drift
    arbitrarily far outside the band -- the band recentres on log(spread) every iteration,
    so a fast move in the channel's scale can put the state far outside it -- and then take
    1 / (aspire_lr * aspire_target) = 80 iterations per unit of log-delta to crawl back,
    reporting clamped=True long after the cause was gone. Writing the clamp back to the
    state means it is never more than one step outside the band and recovery is immediate.

    WHY iteration 1 IS BIT-FOR-BIT v3. log_delta arrives as None on the first iteration
    and the v3 expression is used verbatim for the query, with log(delta) recorded as the
    seed. So both arms have an identical first point and every later difference is
    attributable to the controller alone.

    KNOWN STRUCTURAL RISK, INSTRUMENTED RATHER THAN HIDDEN (inherited). ||c_query|| =
    delta while a typical observed ||c|| ~ sqrt(phi_dim), because the query zeroes every
    component orthogonal to w. sf/query_support reports ||c_query|| / rms||c|| for exactly
    this and v3 measures 0.1009 -> 0.1725. The controller changes the NUMERATOR of that
    ratio and nothing else; it does not restore the orthogonal components, so it cannot fix
    this second disclosed defect and does not claim to.

    A PRE-REGISTERED RISK OF THE FIX ITSELF, stated before the run rather than after. If
    the realized projection were zero-mean Gaussian, its 0.75 quantile is 0.674 sigma,
    BELOW v3's 1.0 sigma -- so calibrating to the top quartile could make the request
    SMALLER, not larger, and a smaller request is a weaker sharpening drive, not a
    stronger one. The mean of w_unit . c is not actually zero (BLOCK C: the level cancels
    83%, not 100%) and the distribution is not actually Gaussian, so the sign of the move
    is an empirical question -- which sf/aspire_achieved at iteration 1 answers directly,
    since it measures P(realized > v3's request) under v3's request exactly.

    NON-FINITE STATE IS A RUNAWAY AND IS TREATED AS ONE. From iteration 2 on the state is
    the ONLY source of delta, so unlike v3 -- which recomputes delta from scratch every
    iteration and therefore heals itself on the next clean batch -- a single NaN in the
    channel would latch delta at NaN for the whole run. Worse, it would do so INVISIBLY:
    every band comparison against NaN is False, so the clamp detector would keep reporting
    clamped=False. So a non-finite state or band drops back to v3's self-healing expression
    for this iteration, reseeds from it, and COUNTS AS A CLAMP so sf/aspire_clamp_frac
    shows it. (Found in review, not in a run: all four verification runs are NaN-free.)

    Returns (c_query, delta, w_unit, log_delta_next, achieved, clamped). Under --no-aspire
    log_delta is passed through untouched and never read, so a stale state cannot change
    delta; in practice it is None because the caller never sets it.
    """
    w = sf.w_vec()
    w_unit = w / w.norm().clamp_min(1e-8)
    # The REALIZED gain along the improvement direction, per row. Already paid for: it is
    # the same projection v3 takes the std of, and it is what `achieved` is measured on.
    proj = c @ w_unit
    spread = proj.std()
    clamped = False
    # Is there a live, finite controller state to take the magnitude from? Iteration 1 has
    # none by construction, and --no-aspire never has one.
    use_state = args.aspire and log_delta is not None
    if use_state:
        log_spread = float(spread.clamp_min(1e-30).log().item())
        lo, hi = log_spread - args.aspire_log_band, log_spread + args.aspire_log_band
        if math.isfinite(log_delta) and math.isfinite(lo):
            clamped = log_delta < lo or log_delta > hi
            # ANTI-WINDUP: the clamp is written BACK into the state, so the integrator can
            # never accumulate outside the band and the controller cannot latch.
            log_delta = min(max(log_delta, lo), hi)
        else:
            use_state, clamped, log_delta = False, True, None
    if use_state:
        delta = spread.new_full((), math.exp(log_delta))
    else:
        # v3's expression, VERBATIM and in the same op order, so --no-aspire is a
        # bit-identical reproduction and --aspire's iteration 1 is a bit-identical seed.
        # The seed is NOT band-checked: on iteration 1 occ_boost * spread IS the band's
        # centre, so there is nothing yet to be outside of.
        delta = args.occ_boost * spread
    # MEASURED IN BOTH MODES. Under --no-aspire nothing consumes it, but it is the exact
    # quantification of the fixed-request defect (how often v3's bar is actually cleared),
    # it costs one comparison, and it does not touch delta -- so the arms stay identical.
    achieved = float((proj > delta).to(proj.dtype).mean().item())
    if args.aspire:
        # Seed from the delta ACTUALLY used -- which is v3's, so the seed is v3's choice
        # exactly -- then integrate from the (clamped) state. Sign: achieving the request
        # MORE often than the target means it was too easy, so raise it.
        if log_delta is None:
            log_delta = float(delta.clamp_min(1e-30).log().item())
        log_delta += args.aspire_lr * (achieved - args.aspire_target)
    return delta * w_unit, float(delta.item()), w_unit, log_delta, achieved, clamped


# =====================================================================================
# BLOCK E -- SF LOSS ASSEMBLY. Its own optimizer, backward and clip; see SFCritic.
# =====================================================================================
def sf_losses(sf, obs, action, psi_target, reward, args):
    """psi TD regression + reward readout + phi decorrelation.

    `psi_target` is detached (built under no_grad in BLOCK B), so this is plain supervised
    regression -- the shape the charter's §3 requires. The reward term regresses w on the
    raw environment reward: with the file's defaults there is no entropy shaping and no
    reward normalization anywhere, so w means "the direction in feature space that the
    environment pays for" with no reinterpretation needed. Passing --normalize-reward or
    --clip-reward (both off by default, both inherited from the chassis) would silently
    redefine w to point at the WRAPPED reward, and every w-derived quantity -- the query
    direction, sf/w_r2, explained_variance -- would follow it.
    """
    # v2: both sides in STANDARDIZED target space. Equivalent to a per-dim reweighting of
    # v1's raw MSE by 1/var, which is what stops the loud dims from owning the gradient.
    psi_pred = sf.psi_std(obs, action)
    phi_pred = sf.phi(obs, action)
    psi_mse = (psi_pred - sf.standardize_psi_target(psi_target)).square().mean()
    rew_mse = (sf.w_head(phi_pred).squeeze(-1) - reward).square().mean()
    cov = offdiag_decorr(phi_pred)
    total = args.sf_coef * psi_mse + args.sf_rew_coef * rew_mse + args.sf_cov_coef * cov
    return total, psi_mse, rew_mse, cov


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.occ_boost > 0.0, "a non-positive request makes the teacher no better"
    assert args.cond_mode in ("occ", "occ_w"), f"unknown cond_mode {args.cond_mode!r}"
    # offdiag_decorr divides by phi_dim * (phi_dim - 1), so phi_dim = 1 would silently make
    # the SF loss NaN. A one-dimensional occupancy measure is also exactly --cond-mode
    # occ_w's channel with none of its encoding, i.e. the thing this file exists to beat.
    assert args.phi_dim >= 2, "phi_dim must be >= 2; a 1-dim occupancy measure is a scalar"
    # delta is a BATCH std (BLOCK D), which is NaN for a single row and would silently
    # propagate NaN into the query, the teacher and every downstream tag.
    assert args.batch_size >= 2, "delta is a batch std; it needs at least two rows"
    # THE CONTROLLER'S DOMAIN. A target of exactly 0 asks for an event that never happens
    # (the error signal saturates at -aspire_target forever and delta collapses into the
    # band floor); exactly 1 asks for an event that always happens and it runs to the
    # ceiling. Both are open-ended integrators, not fixed points, so both are excluded
    # rather than clamped, and the assert says which end was passed.
    assert 0.0 < args.aspire_target < 1.0, "aspire_target must be a strict quantile in (0,1)"
    # A non-positive step is not "the controller off" -- that is --no-aspire, which takes a
    # different code path -- it is a controller that silently freezes delta at iteration
    # 1's value while still claiming to be calibrated. Negative would climb the wrong way.
    assert args.aspire_lr > 0.0, "aspire_lr must be > 0; use --no-aspire to disable"
    assert args.aspire_log_band > 0.0, "aspire_log_band must be > 0 (it is a log-space half-width)"
    sf_lambda = args.sf_lambda if args.sf_lambda >= 0.0 else args.gae_lambda
    sf_hidden = args.sf_hidden if args.sf_hidden > 0 else args.hidden

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    vector_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_cls(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))

    agent = Agent(envs, args).to(device)
    # Rollout forward only. The update stays eager: it is a small share of wall clock, and
    # graphing it would complicate the telemetry paths for little gain.
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # SFCritic's init draws happen inside fork_rng AND from an explicitly reseeded stream.
    # fork_rng alone stops the SF init from advancing the outer stream (so phi_dim cannot
    # silently shift every later minibatch permutation -- the parent's discipline for its
    # TransitionHead, and the bug class conseq_v1's probe had). But it does not fix the
    # other direction, and that direction matters for THIS file's primary claim: cond_mode
    # changes cond_dim, hence the trunk's input width, hence how many normal draws
    # orthogonal_ consumes building the agent -- so without the reseed the two arms would
    # start from DIFFERENT phi/psi/w as well as different trunks, and the iteration-1
    # channel comparison would not be a single-variable one. Reseeding pins phi, psi and w
    # identically across arms. The TRUNK still differs between them: its input width is the
    # ablation, so that difference is irreducible and is stated rather than papered over.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed)
        sf = SFCritic(obs_shape[0], act_dim, args.phi_dim, sf_hidden).to(device)
    # NOT annealed, deliberately, and unlike `optimizer` below. The SF losses are stationary
    # supervised regressions onto detached targets, so there is no trust-region reason to
    # decay their step size; the parent likewise annealed only the actor optimizer and left
    # its jac_optimizer at a constant lr.
    sf_optimizer = optim.Adam(sf.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    # ===== ACHIEVEMENT-RATE CONTROLLER STATE (BLOCK D) ==============================
    # None means "not yet initialized": BLOCK D seeds it on iteration 1 from v3's own
    # formula, log(occ_boost * std(w_unit . c)), which cannot be known before the first
    # rollout exists. Carried across iterations by hand rather than living on a module
    # because it is a control variable, not a weight: no gradient, no optimizer, no
    # state_dict. aspire_clamp_hits counts band saturations for sf/aspire_clamp_frac.
    aspire_log_delta = None
    aspire_clamp_hits = 0.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z = act_fn(next_obs)
            latent_zs[step] = z

            env_action = action.reshape((args.num_envs,) + action_shape)
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [item is not None for item in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_obses[step] = torch.as_tensor(
                transition_next_obs, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        b_obs = obs.reshape((-1,) + obs_shape)
        b_next_obs = next_obses.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_rewards = rewards.reshape(-1)
        with torch.no_grad():
            b_act = agent.z_to_action(b_z)
        nonterm = (1.0 - transition_terminations) * transition_valids
        lam_nonterm = 1.0 - transition_boundaries

        # ===== BLOCK B: psi's target, a lambda-return on FEATURES =======================
        phi_taken, b_psi_target, b_future = sf_feature_targets(
            sf,
            agent,
            b_obs,
            b_next_obs,
            b_act,
            nonterm,
            lam_nonterm,
            sf_lambda,
            args.gamma,
            args.minibatch_size,
        )

        # ===== BLOCK C: the privileged channel ==========================================
        c_raw, b_c = sf_channel(
            sf,
            agent,
            b_obs,
            b_future,
            args.minibatch_size,
            args.cond_ema_beta,
            iteration == 1,
        )

        # v3: refresh psi's standardization HERE -- strictly AFTER the channel, never
        # between BLOCK B and BLOCK C. v2 updated it in between and measurably corrupted the
        # channel: c differences a realized footprint built in BLOCK B (old stats) against a
        # predicted footprint evaluated in BLOCK C (new stats), so the two footprints sat in
        # different psi parameterizations and their difference carried a spurious offset.
        # Measured cost of that ordering, v2 vs v1 at the identical 16-iteration config:
        # sf/delta 0.6726 -> 1.3970 (c inflated ~2x) and chan_action_r2 0.2402 -> 0.2260.
        # Placing the update after BLOCK C makes every psi consumer inside one iteration
        # share one parameterization, and the loss below picks up the refreshed stats.
        #
        # PURE BATCH STATISTICS, no EMA. v2 used the 0.99 channel horizon (~100 iterations)
        # and measured psi_bias_frac climbing straight back 0.0000 -> 0.4751: the target is
        # a BOOTSTRAP, so it drifts as psi learns, and any smoothing lag becomes exactly the
        # per-dim level error this standardization exists to remove. The target is rebuilt
        # from scratch every iteration over the full batch (4096 x 32 here), so its mean and
        # variance are a property of that batch with a standard error of std/64 -- there is
        # nothing to smooth and a lag is pure harm. This REMOVES a knob rather than adding
        # one; no beta is exposed.
        sf.update_psi_stats(b_psi_target, 0.0)

        # ===== BLOCK D: the improvement query ===========================================
        # aspire_log_delta is CONTROLLER STATE carried across iterations, not a parameter:
        # it is a plain python float, it is never a tensor in an autograd graph, it has no
        # optimizer and it receives no gradient. It enters as None on iteration 1 so that
        # the first query is bit-for-bit v3's.
        (
            c_query,
            sf_delta,
            w_unit,
            aspire_log_delta,
            aspire_achieved,
            aspire_clamped,
        ) = sf_query(sf, b_c, args, aspire_log_delta)
        aspire_clamp_hits += float(aspire_clamped)

        with torch.no_grad():
            # In occ_w the standardized channel is ONE scalar, the projection onto w, which
            # the trunk then sees through AdvEmbed's Fourier lift. Two probes are therefore
            # logged and they answer different questions:
            #   sf/chan_action_r2 -- on the standardized channel BEFORE encoding. This is
            #     literally the vector analogue of §6.4's 0.0000 scalar measurement, and it
            #     is 32 predictors in occ against 1 in occ_w, which is the ablation.
            #   sf/enc_action_r2  -- on the encoded block the trunk actually receives (32-d
            #     identity in occ, 16-d Fourier in occ_w). This is the FAIR-TO-occ_w number:
            #     a linear probe on 16 Fourier features can extract far more from a scalar
            #     than a linear probe on the scalar itself, which is exactly why AdvEmbed
            #     exists. Quoting only the first would let a dimensionality artifact stand
            #     in for an information claim.
            if args.cond_mode == "occ_w":
                b_chan = (b_c @ w_unit).unsqueeze(-1)
                query_row = (c_query @ w_unit).reshape(1, 1)
            else:
                b_chan = b_c
                query_row = c_query.reshape(1, -1)
            cond_clipped = (b_chan.abs() >= args.cond_clip).float().mean().item()
            b_chan = b_chan.clamp(-args.cond_clip, args.cond_clip)
            # The query is ONE row (it is state-independent by construction), so clamp it
            # before broadcasting: clamping after `expand` would materialize a full
            # batch_size x cond_dim copy of a single vector for nothing.
            query_all = query_row.clamp(-args.cond_clip, args.cond_clip).expand(
                args.batch_size, -1
            )
            # The query is one row, so its clip fraction is a per-iteration scalar. If the
            # request has been pushed into the clamp on most dims it has degenerated back
            # into v1's constant query, which is what this detects.
            query_clip_frac = float(
                (query_all[0].abs() >= args.cond_clip - 1e-6).float().mean().item()
            )
            chan_rms = b_chan.square().mean().sqrt().item()

        # ===== PAPER FIDELITY: the teacher is FIXED for the whole update ================
        # Snapshot the teacher once per iteration so both actor losses are fixed supervised
        # targets rather than targets that move every minibatch because the student moved.
        # (The paper anchors to the INITIAL policy; that cannot port to RL from scratch --
        # our init is random, so an init-anchored teacher distills noise forever. The
        # portable half is the staleness.)
        with torch.no_grad():
            t_alpha, t_beta = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                a_t, b_t = agent.policy(b_obs[sl], agent.cond_present(query_all[sl]))
                t_alpha.append(a_t)
                t_beta.append(b_t)
            b_t_alpha = torch.cat(t_alpha)
            b_t_beta = torch.cat(t_beta)

        # ===== SF TELEMETRY, measured PRE-UPDATE so it describes the critic that produced
        # ===== this rollout's channel -- the same convention the parent used for EV. =====
        with torch.no_grad():
            psi_taken = _chunked(sf.psi, b_obs, b_act, chunk=args.minibatch_size)
            w_full = sf.w_vec()
            # Necessary-not-sufficient collapse detector: participation ratio of phi's
            # batch covariance spectrum. Its range is [1, phi_dim - 1], NOT [1, phi_dim]:
            # phi's non-affine LayerNorm forces every row to sum to zero, so the covariance
            # always has one exactly-zero eigenvalue and 31 of 32 is the ceiling. Cast to
            # double BEFORE the 32768-row accumulation, so that null direction is resolved
            # in float64 rather than being fp32 rounding noise inside the eigensolver.
            xc = (phi_taken - phi_taken.mean(0, keepdim=True)).double()
            eig = torch.linalg.eigvalsh((xc.T @ xc) / xc.shape[0]).clamp_min(0.0)
            phi_eff_rank = float(
                (eig.sum().square() / eig.square().sum().clamp_min(1e-30)).item()
            )
            corr = phi_corr(phi_taken)
            phi_offdiag = float(
                (corr - torch.diag_embed(torch.diagonal(corr))).abs().sum().item()
                / (args.phi_dim * (args.phi_dim - 1))
            )
            psi_r2 = _r2_per_dim(psi_taken, b_psi_target)
            # WHY psi_r2 CAN BE VERY NEGATIVE EARLY, MADE DECIDABLE. LayerNorm centers phi
            # ACROSS dims, not across samples, so each feature dim keeps a nonzero mean and
            # psi's lambda-return target is "a large per-dim LEVEL plus a small variation".
            # psi starts at ~0 (final layer std=0.01) and must travel to that level first,
            # which R2 charges brutally. This tag reports the share of psi's MSE that is
            # pure per-dim LEVEL error, so "not converged yet" is distinguishable from
            # "cannot fit" instead of being an argument. Near 1 = still climbing the level.
            # It also explains why the level does not poison the CHANNEL: c is a difference
            # of two future footprints, so 83% of the level cancels (see BLOCK C for the
            # exact residual, and why the remainder is an offset rather than a signal).
            psi_err = b_psi_target - psi_taken
            psi_bias_frac = float(
                (
                    psi_err.mean(0).square().mean() / psi_err.square().mean().clamp_min(1e-12)
                ).item()
            )
            # The MODEL's own reward fit, not the best affine rescaling of w . phi: no free
            # scale or intercept is granted, so this is the number the loss actually pays for.
            w_r2 = _r2_per_dim(sf.w_head(phi_taken), b_rewards.unsqueeze(-1))
            # THE LEAKAGE GATE. If a_taken is linearly recoverable from [c, s] the identity
            # collapse is live and clone_nll will run away to -inf. Threshold: <= 0.9.
            cond_action_r2 = _ls_r2(torch.cat([b_chan, b_obs], dim=-1), b_z)
            # THE INFORMATION GATE. §6.4 measured the scalar version of this at 0.0000.
            # Threshold for the hypothesis to be alive at all: >= 0.15.
            chan_action_r2 = _ls_r2(b_chan, b_z)
            # The same gate on the block the trunk ACTUALLY receives -- occ_w's best case,
            # since a linear probe on 16 Fourier features is strictly stronger than one on
            # the scalar they encode. Identical to chan_action_r2 in occ, where the encoding
            # is the identity, which is itself a useful consistency check on this pair.
            enc_action_r2 = _ls_r2(agent.cond_present(b_chan), b_z)
            # THE CONTROL THAT MAKES THE GATES ABOVE MEAN ANYTHING. As the policy sharpens,
            # a_taken becomes a deterministic function of s, so R2(action | ANY quantity
            # correlated with s) drifts up for a reason that has nothing to do with the
            # channel. §6.4's own method quotes the GAIN in R2 over state alone for exactly
            # this reason. Read every gate above against this one.
            state_action_r2 = _ls_r2(b_obs, b_z)
            # (Targets are the latent z rather than the environment action. R2 is invariant
            # to the fixed affine map between them, so this is R2 on the action.)
            # Value is a LINEAR readout of the vector critic, so EV is computed on the
            # scalar pair (V(s,a), w . tgt) and stays comparable to the parent's tag. The
            # IDENTICAL functional -- SFCritic.value's own w_head, bias included -- is
            # applied to both sides; the bias cancels in EV but sharing the call keeps the
            # tag definitionally "EV of V" rather than "EV of a related projection".
            v_pred = sf.w_head(psi_taken).squeeze(-1).cpu().numpy()
            v_true = sf.w_head(b_psi_target).squeeze(-1).cpu().numpy()
            variance = np.var(v_true)
            explained_variance = (
                np.nan if variance == 0 else 1.0 - np.var(v_true - v_pred) / variance
            )
            # How far off the channel's own support the query sits (see BLOCK D).
            query_support = float(
                (
                    query_all[0].norm()
                    / b_chan.square().sum(-1).mean().sqrt().clamp_min(1e-8)
                ).item()
            )

        clone_losses, distill_kls, ents, gaps, tea_ents = [], [], [], [], []
        stu_nlls, conc_ratios = [], []
        sf_totals, sf_psis, sf_rews, sf_covs = [], [], [], []
        # ===== DECOUPLED ACTOR / SF BUDGETS ============================================
        # Actor reuse re-fits the SAME sampled action for the same state: it adds no
        # information, only sharpens the conditional, and is paid for in entropy. SF reuse
        # is ordinary supervised regression onto FIXED detached targets, where extra passes
        # simply reduce fitting error -- and it now runs on a separate module, so unlike the
        # parent's shared-trunk critic it cannot fight the policy update at all.
        for epoch in range(max(args.actor_epochs, args.critic_epochs)):
            do_actor = epoch < args.actor_epochs
            do_sf = epoch < args.critic_epochs
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                obs_mb, z_mb = b_obs[mb], b_z[mb]

                if do_actor:
                    n = obs_mb.shape[0]
                    a_tea, b_tea = b_t_alpha[mb], b_t_beta[mb]
                    chan_mb = b_chan[mb]
                    # One forward for both contexts:
                    #   [privileged at the OBSERVED c_t ; privileged ABSENT (the student)].
                    cond_absent = chan_mb.new_zeros((n, agent.cond_dim))
                    alpha, beta = agent.policy(
                        torch.cat([obs_mb, obs_mb], 0),
                        torch.cat([agent.cond_present(chan_mb), cond_absent], 0),
                    )
                    a_cl, b_cl = alpha[:n], beta[:n]
                    a_stu, b_stu = alpha[n:], beta[n:]

                    # 1. Rationalization / hindsight MLE: fit p(a_t | s_t, c_t). "What it
                    #    did" is the TARGET; "what its future footprint did" is the INPUT.
                    clone_loss = -(
                        Beta(a_cl, b_cl, validate_args=False).log_prob(z_mb).sum(-1).mean()
                    )

                    # 2. Per-dim clipped forward KL from the detached teacher into the
                    #    student. Gradients flow through the student context only.
                    kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                    distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()

                    loss = args.clone_coef * clone_loss + args.distill_coef * distill_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        clone_losses.append(clone_loss.item())
                        # THE EXACT CHANNEL GAIN. The same z_mb scored in the STUDENT
                        # context: (student_nll - clone_nll) is the extra log-likelihood the
                        # privileged channel buys, in nats, on identical data with identical
                        # weights. §6.4's "a channel that predicts nothing about the action
                        # is a channel MLE drops for free" becomes a number here rather than
                        # an argument, and unlike comparing clone_nll against the student's
                        # own ENTROPY this compares two values of the same functional.
                        stu_nlls.append(
                            -(
                                Beta(a_stu, b_stu, validate_args=False)
                                .log_prob(z_mb)
                                .sum(-1)
                                .mean()
                                .item()
                            )
                        )
                        # UNCLIPPED, deliberately: the loss charges the tau-clipped sum but
                        # the parent logs the raw sum under this exact tag, and every number
                        # this file is compared against is the parent's. The two diverge only
                        # when a per-dim KL exceeds distill_kl_clip.
                        distill_kls.append(kl_dims.sum(-1).mean().item())
                        ents.append(
                            Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean().item()
                        )
                        tea_ents.append(
                            Beta(a_tea, b_tea, validate_args=False).entropy().sum(-1).mean().item()
                        )
                        # Does the trunk actually USE the privileged slot? If this is 0 the
                        # teacher is the student and the whole method is a no-op. NOTE it is
                        # averaged over all actor_epochs against a teacher frozen before the
                        # loop, so with actor_epochs > 1 the later passes partly report "the
                        # distillation has already closed the gap" rather than "the trunk
                        # ignores the slot". Same convention as the parent, deliberately, so
                        # the numbers stay comparable to its recorded no-op signature.
                        gaps.append(
                            (
                                a_tea / (a_tea + b_tea) - a_stu / (a_stu + b_stu)
                            ).abs().mean().item()
                        )
                        # Does the teacher SHARPEN or only SHIFT? A pure mass-covering
                        # distillation against a fixed-concentration teacher has no
                        # sharpening channel at all; > 1 means the query buys confidence.
                        conc_ratios.append(
                            (
                                (a_tea + b_tea).mean() / (a_stu + b_stu).mean()
                            ).item()
                        )

                if do_sf:
                    # Separate optimizer and separate grad clip. Sharing either would let
                    # the SF gradient norm change what the actor's clip does, which is a
                    # confound, and would break the gradient-isolation guarantee this
                    # file's whole "not an auxiliary task" claim rests on.
                    sf_total, sf_psi, sf_rew, sf_cov = sf_losses(
                        sf, obs_mb, b_act[mb], b_psi_target[mb], b_rewards[mb], args
                    )
                    sf_optimizer.zero_grad(set_to_none=True)
                    sf_total.backward()
                    nn.utils.clip_grad_norm_(sf.parameters(), args.sf_grad_clip)
                    sf_optimizer.step()
                    with torch.no_grad():
                        sf_totals.append(sf_total.item())
                        sf_psis.append(sf_psi.item())
                        sf_rews.append(sf_rew.item())
                        sf_covs.append(sf_cov.item())

        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/clone_nll", float(np.mean(clone_losses)), global_step)
        writer.add_scalar("losses/student_nll", float(np.mean(stu_nlls)), global_step)
        # THE EXACT CHANNEL GAIN IN NATS. Two values of the SAME functional on the SAME
        # actions: how much log-likelihood the privileged block buys. ~0 is §6.4's "a
        # channel that predicts nothing about the action is a channel MLE drops for free",
        # seen from inside the loss instead of from a probe.
        writer.add_scalar(
            "sf/channel_nats",
            float(np.mean(stu_nlls)) - float(np.mean(clone_losses)),
            global_step,
        )
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("losses/sf_loss", float(np.mean(sf_totals)), global_step)
        writer.add_scalar("losses/sf_psi_mse", float(np.mean(sf_psis)), global_step)
        writer.add_scalar("losses/sf_rew_mse", float(np.mean(sf_rews)), global_step)
        writer.add_scalar("losses/sf_cov", float(np.mean(sf_covs)), global_step)
        writer.add_scalar("debug/cond_gap", float(np.mean(gaps)), global_step)
        writer.add_scalar("debug/query_clip_frac", query_clip_frac, global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/cond_scale_rms", chan_rms, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/reward_mean", b_rewards.mean().item(), global_step)
        # --- the SF block's own instruments. This file's job is to be DECIDABLE. ---
        writer.add_scalar("sf/phi_eff_rank", phi_eff_rank, global_step)
        writer.add_scalar("sf/phi_offdiag_absmean", phi_offdiag, global_step)
        writer.add_scalar("sf/psi_r2", psi_r2, global_step)
        writer.add_scalar("sf/psi_bias_frac", psi_bias_frac, global_step)
        writer.add_scalar("sf/w_r2", w_r2, global_step)
        writer.add_scalar("sf/w_norm", float(w_full.norm().item()), global_step)
        # THE LEAKAGE GATE: near 1 means the identity collapse is live.
        writer.add_scalar("sf/cond_action_r2", cond_action_r2, global_step)
        # THE INFORMATION GATE: the vector analogue of §6.4's 0.0000 scalar measurement.
        writer.add_scalar("sf/chan_action_r2", chan_action_r2, global_step)
        # The same gate on the ENCODED block the trunk receives: occ_w's best case.
        writer.add_scalar("sf/enc_action_r2", enc_action_r2, global_step)
        # THE CONTROL for both R2 gates: how much of them is policy sharpening alone.
        writer.add_scalar("sf/state_action_r2", state_action_r2, global_step)
        writer.add_scalar("sf/delta", sf_delta, global_step)
        # --- THE ACHIEVEMENT-RATE CONTROLLER'S OWN INSTRUMENTS (BLOCK D) --------------
        # THE ERROR SIGNAL, and the primary diagnostic for this arm: the fraction of the
        # batch whose realized gain along w exceeded the request that was actually made.
        # Under --aspire it must converge toward args.aspire_target; under --no-aspire it
        # is a pure measurement of how often v3's FIXED request was met, which is the
        # quantity the fixed-target defect is about, so it is logged in both modes.
        writer.add_scalar("sf/aspire_achieved", aspire_achieved, global_step)
        # log of the query magnitude == the controller state when the controller is on.
        writer.add_scalar("sf/aspire_log_delta", math.log(max(sf_delta, 1e-30)), global_step)
        # Fraction of iterations SO FAR in which the band bound the request. Cumulative on
        # purpose: a per-iteration 0/1 flag is unreadable, and this makes "the controller
        # ran away once at iteration 3" visible for the rest of the run instead of silent.
        writer.add_scalar("sf/aspire_clamp_frac", aspire_clamp_hits / iteration, global_step)
        writer.add_scalar("sf/query_support", query_support, global_step)
        writer.add_scalar("sf/teacher_conc_ratio", float(np.mean(conc_ratios)), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
