# ENT-PPO x JAC-TEACH v2 ROTATE -- rotate the chassis's OWN update instead of adding to it.
#
#            *** VERDICT: THE DESIGN FAILS. DO NOT RUN IT AT 8M. ***
#   The mechanism is delivered EXACTLY as designed (mean-space rotation cosine and norm
#   both exact to 5 decimals, sign verified by measurement), norm preservation HOLDS, and
#   the arm STILL blows the mask gate by 20x and the entropy gate by 36x. The hypothesis
#   "norm-preserving in the space the mechanism acts in => dose-preserving" is FALSIFIED on
#   this chassis, by the cleanest possible instance of it. Numbers and cause below; the
#   file is kept because the falsification is the result.
#
# =====================================================================================
# THE MECHANISM, IN ONE SENTENCE. Take the chassis's clipped-surrogate gradient w.r.t. the
# student's Beta MEAN, rotate that vector -- per sample, exactly, norm-preservingly -- by
# cos = jac_cos toward the analytic one-step-model direction, and hand the optimizer the
# ROTATED vector in place of the original. Nothing is added to the objective's magnitude;
# the update turns and keeps its length.
#
# WHY THIS LOOKED LIKE THE RIGHT SHAPE. The previous agent measured that the chassis's own
# mean-space improvement direction is at cosine 0.9988 (min 0.9856 over 12 fresh agents) to
# u_cred = grad_mean[A_i * log pi(a_i|s_i)], the direction the donor's rotation was
# validated on. THIS FILE REPRODUCES THAT NUMBER, as debug/rot_cred_cos = 0.9977 (cos 1.0
# arm, champion config): the chassis's actual clipped, masked, entropy-shaped surrogate
# gradient in mean space IS the credit direction. So the premise is sound.
# ONE CORRECTION TO HOW IT WAS STATED, MEASURED HERE: 0.9988 is a GRADIENT-space cosine,
# not a realized-displacement one. The per-row cosine between the change in the Beta mean
# produced by one ACTUAL optimizer step and the mean-space gradient is 0.021 -- for the
# UN-ROTATED arm (debug/rot_adam_cos, cos 1.0, corrector provably absent). A full step on
# this chassis moves each sample's mean mostly through parameters shared with every other
# sample and with the distributional critic on the same trunk, so the per-row diagonal
# response is ~2% of the displacement. Any design argued from "the realized displacement is
# the credit direction" is arguing from a quantity that reads 0.02, not 0.999.
#
# THE DONOR'S MECHANISM, AND THE HYPOTHESIS THIS FILE TESTS. The thing that actually worked
# (8915 on a weaker chassis, +44% @1M) was never "distil into a teacher" and never "add an
# auxiliary": it was a NORM-PRESERVING ROTATION OF THE IMPROVEMENT DIRECTION at fixed
# cosine with the delivered dose held constant, so that only DIRECTION changed. Since the
# chassis's own update direction IS that improvement direction (0.9988 above), the mechanism
# can be applied to the chassis's own gradient directly -- no teacher, no distillation
# objective, no auxiliary penalty, and, the hypothesis went, no dose change either.
#
# NORM PRESERVATION WAS THE LOAD-BEARING PROPERTY, and the two prior measured failures of
# this lineage are both failures of exactly that property:
#
#   DESIGN 1, `--improve distill` (replace the surrogate with forward KL into a mean-shift
#   teacher). FAILED. d(losses/entropy) over 262k = +0.075 against the surrogate's -0.038.
#   An ent_alpha 0.3/0.1/0.0 sweep left the leak at +0.075/+0.074/+0.075, so it is intrinsic
#   to mass-covering forward KL against a fixed-concentration teacher, not to the entropy
#   bonus. The student's realized drift came out ~20x SMALLER than the surrogate's, so the
#   per-sample KL-drift mask at the native tr_sample_eps = 0.1 went exactly inert
#   (kept-fraction 1.000000 in all 8 iterations). The objective replaced both the direction
#   AND the magnitude of the chassis's update, and got the magnitude wrong in both
#   directions at once: too small in ratio space, unbounded in entropy space.
#
#   DESIGN 2, `--improve both` (additive KL-dosed auxiliary on top of the untouched
#   surrogate). SAFE BUT WEAK. All gates pass at 5% declared dose (d(entropy) -0.039 vs
#   -0.038; debug/tr_mask_frac 0.0061 vs 0.0062), but debug/actor_clip_sat reads 1.000 in
#   every iteration -- clip_grad_norm_ renormalizes the actor group in 100% of minibatches --
#   so the auxiliary cannot lengthen the step, only turn it, and the measured delivered
#   influence ||grad_aux||/||grad_pg|| is 0.0047. Pushing it further fails the entropy gate:
#   the leak closes at 14.9% declared dose and inverts (+0.0225) at 47.8%. So the delivered
#   effect cannot exceed ~1.5% without breaking the same gate design 1 broke.
#
# THE PRE-REGISTERED HYPOTHESIS, WHICH THIS FILE FALSIFIED. Both prior failures share one
# apparent cause: a term ADDED to the objective changes the SIZE of the improvement
# direction, and this chassis pays for size with entropy. A rotation cannot change the size:
# ||g_rot|| == ||g_ch|| per row, exactly. So the delivered dose (ratio drift, per-sample
# mask, KL) should be unchanged BY CONSTRUCTION, and the full strength of the mechanism --
# an 18.2 degree turn at cos 0.95, a delivered ||delta||/||g_ch|| of sqrt(2(1-cos)) = 0.316
# -- should be available instead of design 2's 0.0047.
#
# THE FIRST HALF IS TRUE AND THE SECOND HALF IS FALSE. Measured: the mean-space norm ratio
# is 1.00000 at every cosine and every iteration, the realized displacement norm ratio never
# exceeds 1.01, the UNCLIPPED parameter-space actor gradient gets SHORTER (0.62x at cos
# 0.95), the step length is the clip's and is therefore identical -- and tr_mask_frac still
# goes to 20x the surrogate's and losses/entropy still falls 36x faster. NORM PRESERVATION
# IS NOT DOSE PRESERVATION. The cause is measured and stated in FINDING 9 below.
#
# THE IMPLEMENTATION, EXACTLY (`--improve rotate`)
#  1. The chassis's objective is computed UNCHANGED: same clip (0.2/0.28), same ent_alpha
#     entropy/proximal term, same per-sample KL-drift mask at the native tr_sample_eps = 0.1
#     with kept-count renormalization, same epoch/leash logic, same distributional HL-Gauss
#     MTP value loss. Not one coefficient is re-derived.
#  2. g_ch, THE CHASSIS'S OWN MEAN-SPACE ASCENT DIRECTION, per sample, shape (B, act_dim):
#     the Beta is reparameterized (alpha, beta) -> (m, c) with alpha = m*c, beta = (1-m)*c,
#     so at FIXED concentration
#         d/dm = c * (d/dalpha - d/dbeta)      and      g_ch = -c * (gA - gB)
#     with (gA, gB) = torch.autograd.grad(pg_loss_chassis, (alpha, beta), retain_graph=True).
#     The minus sign makes g_ch an IMPROVEMENT direction (the surrogate's, not the loss's).
#     This costs one short backward -- pg_loss -> log_prob -> Dirichlet -> alpha/beta -- that
#     never reaches the trunk, and it adds NOTHING to the forward graph, so the chassis's
#     own numerics are untouched. g_ch inherits the mask and the 1/n_keep renormalization
#     from pg_loss, so masked rows are EXACTLY zero and kept rows carry the surrogate's own
#     scale.
#  3. u_an, the analytic direction, is v1's g_hat verbatim: d/da[gamma * V(s + ds_hat(s,a))]
#     through the same TransitionHead, behind the same held-out one-step R2 gate, with the
#     same per-dim validity mask, lifted to the KL metric (times sd) and unit-normalized.
#  4. g_rot = rotate_to_cos(g_ch, u_an, cos_t) -- v1's EXISTING exact rotation, v1's gate
#     ramp cos_t = 1 - jac_gate*(1 - jac_cos), v1's degenerate-row fallback (a zero g_ch row,
#     a zeroed-out u_an row, or a u_an parallel to g_ch passes through unchanged).
#  5. THE CORRECTOR. delta = (g_rot - g_ch).detach(), and
#         pg_loss <- pg_loss - (delta * keep * m_new).sum()
#     where m_new = alpha_new/(alpha_new + beta_new) is a differentiable function of the
#     student's own head. This term is LINEAR in m and has, exactly:
#         d/dm = -delta   (so the total mean-space ascent direction becomes g_ch + delta
#                          = g_rot -- the SIGN IS NEGATIVE, and it was determined
#                          EMPIRICALLY, see MEASURED 3 below, not argued),
#         d/dc = 0        (identically -- a pure mean shift, with no concentration and hence
#                          no entropy pressure of its own anywhere in it).
#     The kept-count renormalization is INHERITED, not re-applied: delta is a gradient of the
#     already-renormalized surrogate, so dividing by n_keep a second time would halve the
#     rotation's scale relative to the surrogate's. `keep` is applied explicitly anyway, so a
#     masked sample provably contributes to neither term.
#  6. `--jac-cos 1.0` requests sin_t = 0 exactly, so rotate_to_cos is the algebraic identity
#     -- to 4.4e-16 in float64 and ~1e-7 in the float32 the run uses, which is NOT bit-zero.
#     The identity control demands BIT-exactness, so the corrector is added only when
#     cos_t < 1.0. Everything else still runs at cos 1.0: the transition head is built,
#     trained and gated, g_ch is computed every minibatch, the rotation and every rot_*
#     telemetry scalar execute. The control is therefore an additivity test, not a skipped
#     branch (VERIFIED bit-exact, MEASURED 2).
#
# MEASURED HERE. HalfCheetah-v4, seed 1, every number produced by this file.
#   TINY = --env-id HalfCheetah-v4 --num-envs 4 --num-steps 256 --num-minibatches 4
#          --total-timesteps 20480 --seed 1                                (20 iterations)
#   CHAMPION = --env-id HalfCheetah-v4 --num-envs 16 --num-steps 2048 --num-minibatches 32
#          --update-epochs 10 --total-timesteps 262144 --seed 1             (8 iterations)
# "gate-open mean" = mean over the iterations where debug/jac_gate == 1.0 (iterations 3-8 at
# CHAMPION); the rotation does not exist before the held-out R2 gate opens.
#
#  1. py_compile passes. All FOUR modes run at TINY: surrogate / distill / both / rotate.
#  2. IDENTITY CONTROL, PASSED, at BOTH configs. `--improve rotate --jac-cos 1.0` vs
#     `--improve surrogate` on the identical command line, NaN-safe (NaN == NaN),
#     charts/SPS excluded:
#         TINY      37 tags vs 56, 36 shared compared, 720 points, ZERO differing
#         CHAMPION  37 tags vs 56, 36 shared compared, 784 points, ZERO differing
#     The rotate arm emits 19 telemetry-only tags the surrogate arm does not (8 rot_*, 8
#     jac_*/teach_*, losses/jac_model, debug/actor_clip_sat), so the tag sets are a strict
#     superset. The transition head is built, trained and gated in the control, g_ch is taken
#     every minibatch, the rotation runs, and the matched-arm probe perturbs and restores the
#     actor parameters -- and nothing moves. It is an additivity test, not a skipped branch.
#  3. THE ROTATION IS DELIVERED EXACTLY, AND THE SIGN WAS FOUND BY MEASUREMENT.
#     THE SIGN IS NEGATIVE: pg_loss <- pg_loss - (delta * keep * m_new).sum().
#     It was determined from debug/rot_norm_check, not by reasoning: the positive sign
#     delivers 2*g_ch - g_rot, whose norm ratio is sqrt(5 - 4cos) = 1.0954 at cos 0.95 and
#     whose cosine to g_ch is (2-cos)/sqrt(5-4cos) = 0.9585 -- a wrong sign that LOOKS like
#     an almost-right cosine. The norm, not the cosine, is what separates them. Measured with
#     the negative sign, CHAMPION gate-open means:
#         jac_cos   rot_cos_check   rot_norm_check   rot_delta_ratio   sqrt(2(1-cos))
#         1.00      1.00000         1.00000          0.00000           0.00000
#         0.99      0.99000         1.00000          0.14142           0.141421
#         0.95      0.95000         1.00000          0.31623           0.316228
#         0.85      0.85000         1.00000          0.54772           0.547723
#     i.e. the total mean-space gradient after the corrector is at EXACTLY the requested
#     cosine to g_ch, at EXACTLY the same norm, with delta at EXACTLY the length an exact
#     norm-preserving rotation implies. No doubling (which would read 0.805 at 0.95), no
#     2*cos-1 (0.90), no inversion.
#     THE PREMISE, re-measured every iteration: debug/rot_cred_cos = cos(g_ch, u_cred) =
#     0.9977 / 0.9957 / 0.9830 / 0.9741 down the ladder. The chassis's own surrogate gradient
#     in mean space IS the credit direction, so this IS the donor's mechanism.
#  4. NORM PRESERVATION HELD -- exactly, and in realized terms. CHAMPION gate-open means:
#         jac_cos   rot_norm_check   rot_norm_ratio   rot_pgrad_ratio   rot_cos_realized
#         1.00      1.00000          1.00000          1.00000           1.00000
#         0.99      1.00000          0.98138          0.86431           0.83900
#         0.95      1.00000          0.93706          0.62389           0.46529
#         0.85      1.00000          0.90162          0.54209           0.26068
#     rot_norm_ratio is the REALIZED ||Delta m|| of the rotated arm over the surrogate arm's
#     at a matched step (same parameters, same minibatch, same Adam state -- a matched-arm
#     A/B, not a cross-run comparison). It is at or BELOW 1 everywhere: the update never grew.
#     debug/actor_clip_sat = 1.000 in every iteration of every arm, so the step LENGTH is the
#     clip's 0.25 and is identical between arms by construction.
#  5. ENTROPY GATE: FAILED. The delta is negative, so the arm does not leak entropy the way
#     design 1 did -- but it is nowhere near the surrogate's, which is what the gate asks.
#     losses/entropy, CHAMPION, identical command lines:
#         arm              first      last      delta      vs surrogate
#         surrogate       -0.43343  -0.47151   -0.03807     1.0x
#         rotate 0.99     -0.43343  -0.66514   -0.23171     6.1x
#         rotate 0.95     -0.43343  -1.81143   -1.37800    36.2x
#         rotate 0.85     -0.43343  -2.05646   -1.62303    42.6x
#     The policy sharpens 36x faster at the design's own default cosine.
#  6. MASK GATE: FAILED, decisively. CHAMPION:
#         arm            tr_mask_frac  (max)   approx_kl@262k  (max)   ratio_max  trips  epochs
#         surrogate         0.00622  0.00830      0.00696   0.01261        5.55     0    10.00
#         rotate 0.99       0.01667  0.02744      0.01258   0.02341        7.08     0    10.00
#         rotate 0.95       0.12375  0.21602      0.12131   0.15304       69.62     3     9.12
#         rotate 0.85       0.16152  0.26713      0.10234   0.15642       38.31     6     5.12
#     (tr_mask_frac and approx_kl columns are means over the 8 iterations with the per-run max
#     beside them; trips = debug/kl_breaker_trips at 262k; epochs = mean
#     debug/epochs_completed.) At cos 0.95 the mask fraction is 20x the surrogate's, approx_kl
#     is 17x, the sampled ratio tail reaches 69.6 against the surrogate's 5.55, and approx_kl
#     hits the target_kl_breaker of 0.15 -- the EMERGENCY breaker, five times the chassis's
#     own kl_target -- in 3 of 8 iterations. At cos 0.85 it fires in 6 of 8 and the arm
#     forfeits half its epoch budget (epochs_completed 10,6,7,2,1,1,4,10).
#     THE ROTATION IS CHANGING THE DOSE, and it is doing so while the norm it was supposed
#     to preserve is preserved to five decimals.
#  7. MECHANISM ALIVE (so the failure is not an inert or broken mechanism). CHAMPION:
#     jac_r2 -0.0002 -> 0.9390 (cos 0.95), jac_gate 0 -> 0.699 -> pinned 1.0 from iteration 3,
#     jac_rot EXACTLY the requested cosine whenever the gate is open, jac_align -0.211 ..
#     +0.002 (the analytic direction is still ORTHOGONAL to the credit direction -- the
#     donor's pre-registered kill condition does not fire), jac_dose_scale >= 0.991,
#     debug/actor_clip_sat 1.000, losses/explained_variance at 262k 0.8511 (surrogate) /
#     0.8514 / 0.8487 / 0.8462, 262k return (last 20 episodes) 193 / 353 / 446 / 108.
#     NOTE THE RETURN COLUMN AND DISREGARD IT. rotate 0.95 scores 446 against the surrogate's
#     193 while destroying both gates: a bigger effective step helps at 262k and this lineage
#     has already established (donor behind PPO at 500k, ahead from 1M) that 262k return
#     cannot separate a slow start from a dud. It is exactly the column that would have sold
#     this arm into an 8M run.
#     NON-FINITE SCALARS: none anywhere, in any arm, after iteration 1 -- EXCEPT the cos 0.85
#     arm's eight rot_* tags at iterations 5 and 6, where the 0.15 breaker fired inside epoch
#     1 (epochs_completed = 1) so the last minibatch of an epoch never ran and the probe never
#     fired. That NaN is a symptom of finding 6, not a defect: rot_probe is set False when no
#     row is eligible, and a NaN is logged rather than a fabricated 1.0.
#  8. COS LADDER, CHAMPION (the summary of 5-7, with the delivered rotation beside it):
#         jac_cos  d(entropy)  tr_mask_frac  rot_delta_ratio  rot_norm_check  ret@262k  gates
#         (surr)    -0.03807      0.00622        --               --            193     pass
#         0.99      -0.23171      0.01667      0.14142          1.00000         353     marginal
#         0.95      -1.37800      0.12375      0.31623          1.00000         446     FAIL
#         0.85      -1.62303      0.16152      0.54772          1.00000         108     FAIL
#     The gates degrade MONOTONICALLY and STEEPLY in the delivered rotation, which is the
#     opposite of what norm preservation predicted and the opposite of design 2's profile
#     (flat up to 5% declared dose, knee at 15%). There is no safe band: 0.99 already inflates
#     the mask 2.7x and the entropy slope 6.1x while delivering only a 8.1 degree turn.
#
#  9. THE CAUSE, MEASURED. Not a bug -- the mean-space algebra is exact (finding 3) -- and
#     not the mechanism being inert (finding 7). The chain, each link with its scalar:
#     (a) The corrector's parameter-space gradient partially CANCELS the surrogate's:
#         debug/rot_pgrad_ratio = 0.624 at cos 0.95 on the UNCLIPPED actor gradient. The
#         analytic direction's component orthogonal to g_ch is largely not expressible as a
#         per-state field by this actor (a 64-wide shared trunk plus two linear heads), so
#         J^T attenuates it and the summed actor gradient gets ~38% shorter.
#     (b) debug/actor_clip_sat = 1.000: clip_grad_norm_ renormalizes the actor group in 100%
#         of minibatches. A SHORTER gradient renormalized to the SAME fixed budget is a step
#         of the same length pointing further away.
#     (c) The realized turn is therefore ANGULARLY AMPLIFIED, and by a lot:
#             requested   18.2 deg (cos 0.95)  ->  realized  62.3 deg (rot_cos_realized 0.465)
#             requested    8.1 deg (cos 0.99)  ->  realized  32.9 deg (0.839)
#             requested   31.8 deg (cos 0.85)  ->  realized  74.9 deg (0.261)
#         a 2.4x - 4.1x amplification of the angle, stable across the ladder.
#     (d) The chassis's trust region is pessimistic only about motion ALONG its own surrogate
#         direction: the clip's max() zeroes a sample that has drifted too far in the
#         direction the surrogate pushes, and the per-sample mask zeroes it on the drift that
#         motion produces. A step turned 62 degrees away spends most of its length in
#         directions that neither instrument resists, so the SAME-LENGTH step buys far more
#         policy change.
#     (e) losses/entropy therefore falls 36x faster, the Beta concentrations grow, and a
#         sharp Beta makes log pi(a|s) hypersensitive to mean motion -- so the sampled drift
#         d_i = (rho-1) - log rho explodes quadratically. tr_mask_frac 20x, ratio_max 69.6,
#         the 0.15 breaker firing. debug/sigma_mean does NOT explain it (109-158 in both
#         arms), and kl_beta stays pinned at 0.3, so this is not the adaptive-beta path.
#     THE GENERAL LESSON, stated so the next design does not repeat it: NORM PRESERVATION IN
#     THE SPACE THE MECHANISM ACTS IN IS NOT DOSE PRESERVATION, for two independent reasons.
#     The map from mean space to parameter space is not an isometry, and a SATURATED gradient
#     clip converts any norm reduction into an angular amplification. Design 2 was protected
#     from this only by being 67x weaker (delivered 0.0047 vs this arm's 0.316).
#
# 10. ONE ALTERNATIVE CAUSE TESTED AND RULED OUT: the metric. This file rotates in RAW
#     mean-gradient space (||g_rot|| == ||g_ch||, as specified), while the donor's dose is a
#     KL and v1's own comment warns that "mixing raw gradients would silently weight the
#     low-variance action dims hardest". So the KL-metric-consistent variant was measured as a
#     one-off ablation (rotate u_ch = g_ch*sd toward u_an, then g_rot = u_rot/sd, preserving
#     the KL-metric norm instead of the raw one), CHAMPION, seed 1:
#         arm                   d(entropy)  tr_mask_frac  approx_kl@262k  trips  epochs  ret
#         surrogate               -0.03807     0.00622        0.00696        0    10.00  193
#         rotate  0.95 (raw)      -1.37800     0.12375        0.12131        3     9.12  446
#         metric  0.95            -0.85490     0.10173        0.04270        0    10.00  426
#         rotate  0.85 (raw)      -1.62303     0.16152        0.10234        6     5.12  108
#         metric  0.85            -1.79223     0.16503        0.11007        6     4.62   53
#     The metric fix helps at 0.95 (no breaker trips, full epoch budget, approx_kl 2.8x
#     instead of 17x) and does NOT help at 0.85, and it still leaves tr_mask_frac at 16x the
#     surrogate's and the entropy slope at 22x. So the raw-space metric mismatch is a
#     CONTRIBUTING factor and NOT the cause. It is deliberately NOT wired into this file: the
#     verdict does not change, and tuning around a failed gate is not the job.
#
# THE RECOMMENDATION FOR 8M: DO NOT RUN THIS ARM. Argued from the gates and the delivered
# rotation only, as required, and NOT from the 262k return column -- which favours the worst
# arm in the file. Every rung that delivers a non-trivial rotation fails the mask gate, and
# the failure is monotone in the delivered rotation, so there is no cosine to choose: the
# only rung with intact gates is jac_cos = 1.0, which is the surrogate by construction. If a
# rung had to be named it would be 0.99, and it should not be: it already inflates
# tr_mask_frac 2.7x and the entropy slope 6.1x, i.e. it is on the same monotone curve, merely
# earlier on it, and it delivers an 8.1 degree turn for that price.
#
# WHAT THE NEXT ATTEMPT MUST DO DIFFERENTLY (pre-registered, NOT measured here). The failure
# is located precisely at finding 9(a)-(c), so there are exactly two repairs worth testing,
# and each is a falsifiable one-change experiment:
#  1. Give the corrector its OWN clipped gradient group, the way --kl-grad-group does for the
#     KL term, so a shorter summed gradient can no longer be renormalized back to full
#     length. PREDICTION: rot_cos_realized falls to approximately jac_cos instead of
#     amplifying 2.4-4.1x, and tr_mask_frac returns to within 2x of the surrogate's. This is
#     the repair the evidence points at, and v1 explicitly REJECTED its analogue for the
#     additive auxiliary -- for a reason (it lets the term spend drift the trust region never
#     authorized) that finding 6 shows is now moot, since the rotation spends that drift
#     anyway.
#  2. Dose-match on the REALIZED drift rather than on the mean-space norm: bisect a scalar on
#     delta each iteration until the measured tr_mask_frac / approx_kl of the rotated arm
#     matches the un-rotated arm's on the same batch, exactly as v1's jac_dose_scale bisects
#     on delivered KL. PREDICTION: the required scalar is ~0.1-0.2 at cos 0.95, i.e. the
#     honest delivered rotation on this chassis is 3-6 degrees, not 18 -- which, if true, puts
#     this mechanism back within a factor of ~10 of design 2's 0.47% and means the whole
#     lineage is bounded by the chassis's trust region rather than by the teacher.
# A THIRD OPTION, STATED BECAUSE IT IS THE HONEST ONE: the chassis's per-sample trust region
# and its saturated clip together make the actor step's DIRECTION the only free variable and
# then punish changing it. Every one of the three designs measured in this lineage has now
# failed at that same wall, from three different sides. That is evidence about the CHASSIS,
# not about the teacher.
#
# =====================================================================================
# --- v1 HEADER BELOW, UNCHANGED (the distill/both designs and their measured failures) --
# ENT-PPO x JAC-TEACH v1 -- the per-sample trust region AND the analytic improvement
# direction in one file, one flag apart.
# =====================================================================================
# THE TWO PARENTS, AND WHY THEY SHOULD COMPOSE (the phase-complementarity hypothesis)
#
# HalfCheetah-v4, seed 1, matched steps, scripts/score_runs.py. INHERITED, not measured
# here:
#   variant                                     @1M    @2M    @4M    @6M    @8M   last-20
#   jacteach_cos0p95_8M  (analytic direction)  4370   6701   8106   8835   8859  8915+-131
#   entppo_v4_samplemask_a03_mask (per-sample  3041   5465   8890   9873  10312 10362+-92
#                                  KL mask)
#
# Read as a pair, those two curves own OPPOSITE PHASES. jacteach is +44% at 1M and then
# saturates (+0.3% from 6M to 8M): it improves WHERE the teacher points -- the target
# direction -- and a better direction pays off immediately and then stops paying. entppo
# starts 30% behind and does not saturate (+4.4% from 6M to 8M): it improves HOW BIG a
# step is taken per sample -- the step control -- which compounds. The hypothesis of this
# file is therefore that the two mechanisms are about different halves of the same update
# (direction vs magnitude) and can be carried at once:
#
#   improvement direction : credit direction ROTATED toward an analytic direction obtained
#                           by autograd through a learned one-step transition model, at
#                           exactly cos = jac_cos.
#   step control          : the chassis's PER-SAMPLE KL-drift mask, d_i = (rho_i - 1) -
#                           log rho_i > tr_sample_eps, masking individual samples and
#                           renormalizing by the KEPT count -- kept active, and applied to
#                           the distillation loss exactly as it is applied to the surrogate.
#
# ONE FLAG, THREE MODES. `--improve surrogate` (default) is the entppo chassis BIT FOR BIT:
# the transition head is not built, no RNG is drawn, nothing is logged that the parent does
# not log. `--improve distill` REPLACES the clipped surrogate with the donor's per-dim
# forward KL into a mean-shifted teacher -- MEASURED TO FAIL, see below. `--improve both`
# leaves the whole chassis objective alone and adds the same rotated, dose-matched
# displacement as ONE ADDITIVE AUXILIARY, weighted by `jac_aux_coef`; at jac_aux_coef = 0 it
# is bit-exact to `--improve surrogate`. The entropy bonus (soft-MDP reward + the exact
# all-action KL(pi_new||pi_old) proximal term) and the distributional HL-Gauss MTP value
# loss are untouched in all three modes.
#
# DELIBERATELY NOT PORTED: the donor's advcond apparatus (AdvEmbed, cond_scale, adv_boost,
# the clone-NLL rationalization). The improvement direction here is built ONLY from things
# this chassis already computes -- the rollout Beta's own head parameters, the replayed
# native sample, the batch-standardized advantage -- plus the learned transition head. No
# privileged conditioning, no second policy context, no extra head on the actor trunk.
#
# THE FIVE PIECES, IN ORDER
#  1. CREDIT DIRECTION. d/dm [ A_i * log pi(a_i | s_i ; m, c) ] at the rollout action, at
#     fixed concentration, detached, lifted to the KL metric (times sd) and unit-normalized
#     per row, with A_i re-entering as the per-sample dose weight. This is the direction the
#     chassis's own surrogate moves the Beta mean; nothing new is learned to obtain it.
#  2. ANALYTIC DIRECTION. d/da [ gamma * V(s + ds_hat(s,a)) ] through the donor's
#     TransitionHead, with the donor's held-out one-step R2 gate and per-dim validity mask
#     preserved exactly. V(s) stays the value object (no Q, no twin critics, no target
#     nets); the model is regression on the observed transitions of the single trajectory
#     and is never rolled forward.
#  3. EXACT ROTATION (rotate_to_cos). Norm-preserving, in the plane the two directions span,
#     at exactly the requested cosine; degenerate rows fall back to the credit direction.
#  4. DOSE. The rotated displacement is shrunk by one bisected scalar per iteration until
#     the delivered per-dim clipped KL equals the un-rotated displacement's, so the arm
#     differs from its own `--jac-cos 1.0` control in DIRECTION ONLY. The overall dose is
#     declared once, in the KL metric, by `teach_step`.
#  5. STUDENT + TRUST REGION. Per-dim forward KL from the detached teacher, with the
#     per-sample mask applied to it sample-by-sample and renormalized by the kept count.
#
# =====================================================================================
# EVIDENCE. INHERITED (from the two parents, NOT re-measured here):
#   * the matched-step table above;
#   * the donor's offline transition-model validation (held-out one-step ds R2 0.841 at 5k
#     / 0.944 at 100k transitions; cos(J^T e_vx) vs simulator central differences 0.955 ->
#     0.975);
#   * the donor's measured jac_align = -0.008 .. +0.046 on ITS chassis;
#   * the chassis's own ablations (alpha 0.1/0.3/0.6, kl_beta 0/0.3, tr_mode abort/mask).
#
# VERIFIED HERE (commands and outputs are in the task record; every number below was
# produced by this file):
#   * BIT-EXACT CONTROL. `--improve surrogate` vs the chassis parent at
#     `--num-envs 4 --num-steps 256 --num-minibatches 4 --total-timesteps 20480 --seed 1`:
#     37 TensorBoard scalar tags, identical tag sets, 36 tags compared (all but charts/SPS),
#     720 scalar points, ZERO differing, NaN-safe comparison.
#   * ROTATION ALGEBRA, float64, 20000x6 random rows, for cos in
#     {1.0, 0.999, 0.95, 0.85, 0.5, 0.0, -0.5, -1.0}: max |achieved cos - requested cos| =
#     1.2e-15 (5.6e-16 at cos 0.95), max | ||u_new||/||u_cred|| - 1 | = 6.7e-16,
#     cos 1.0 reproduces u_cred to 4.4e-16 (algebraic identity, not a skipped branch), and
#     zero / parallel / antiparallel / sub-guard rows pass through EXACTLY (diff 0.0) with
#     no NaN. In float32, the dtype the run uses: max |dcos| 4.8e-7.
#   * THE CREDIT DIRECTION'S PARAMETERIZATION IS THE LOAD-BEARING DETAIL, AND IT IS
#     MEASURED. Taking the gradient w.r.t. the ACTION ARGUMENT instead of w.r.t. the
#     distribution's mean gives the ANTI-teacher: over 12 fresh agents, one real chassis
#     surrogate step moves the Beta mean with cosine 0.9988 (min 0.9856) to this file's
#     u_cred and cosine -0.977 (min -0.997) to the literal grad-w.r.t.-action reading. The
#     two are not even exact negations for a Beta (mean per-row cos(d/dm, -d/dz) = 0.58), so
#     this is a parameterization choice and not only a sign choice.
#   * MECHANISM ALIVE, champion config (16 envs x 2048, mb32, 10 epochs, 262144 steps,
#     seed 1), `--improve distill --jac-cos 0.95`: jac_r2 0.71 -> 0.93, jac_gate pins at
#     1.0 from iteration 3, jac_rot exactly 0.9500, jac_align -0.028 .. +0.026 (i.e. the
#     analytic direction is ORTHOGONAL to the credit direction on THIS chassis too, which
#     was the donor's pre-registered kill condition), jac_dose_scale 0.9993-1.0,
#     jac_clamp_frac 0. No non-finite scalar anywhere after iteration 1 (jac_align,
#     jac_rot, jac_dose_scale, jac_dir_norm are NaN in iteration 1 BY DESIGN: the R2 gate
#     is shut, so no rotation is computed).
#   * DOSE MATCH HOLDS, and the reason is disclosed: delivered per-dim clipped KL at
#     cos 0.95 vs the cos 1.0 control matches to <= 0.13% per iteration (0.033285 vs
#     0.033265 at 262k). It holds trivially at teach_step 0.25, because the delivered KL is
#     0.0055 per dim against a tau of 2.0 -- the clip binds on nothing, so norm preservation
#     alone is sufficient and the bisection returns ~1. This does NOT establish that the
#     dose match holds where the clip binds; the donor measured that it only partly does.
#
# =====================================================================================
# WHY `--improve both` EXISTS: THE MEASURED FAILURE OF `--improve distill`.
# HalfCheetah-v4, seed 1, champion config (16 envs x 2048, mb32, 10 epochs), 262144 steps,
# 8 iterations, matched steps. Every number below was produced by this file.
#
#   * ENTROPY LEAKS UPWARD. d(losses/entropy) over the run = +0.075 in distill mode against
#     -0.038 for the surrogate on the identical command line. Raising teach_step to >= 1.0
#     reaches a UNIFORM Beta (alpha = beta = 1) within 4 iterations, after which
#     ratio_max = 1.000, approx_kl = 3e-8 and the policy is frozen; teach_step 2.0 and 4.0
#     collapse in the first two iterations.
#   * THE ENTROPY BONUS IS NOT THE CAUSE. FALSIFIED BY ABLATION: an ent_alpha sweep
#     0.3 / 0.1 / 0.0 gives d(entropy) = +0.075 / +0.074 / +0.075. Deleting the chassis's
#     entropy term outright moves the leak by 1.3%, i.e. not at all. The leak is intrinsic
#     to the OBJECTIVE: KL(teacher || student) is MASS-COVERING and contains
#     -E_teacher[log pi_student], which pays the student for BROADENING, and against a
#     fixed-concentration mean-shift teacher that a 64-wide trunk cannot represent per state
#     broadening is the cheapest way to cover it.
#   * THE CHASSIS'S OWN MECHANISM IS INERT THERE. The student's realized drift is ~20x
#     smaller than the surrogate's, so at the native tr_sample_eps = 0.1 the per-sample mask
#     rejects EXACTLY NOTHING (kept-fraction 1.000000 in all 8 iterations).
#   * 262k RETURN: -133 .. -141 (distill) vs 193 (surrogate).
#
# THE AUXILIARY, AND WHY IT CANNOT LEAK THE SAME WAY. `--improve both` keeps the clipped
# surrogate, the ent_alpha entropy term, the per-sample mask at its native
# tr_sample_eps = 0.1, the epoch/leash logic and the distributional value loss EXACTLY as
# the chassis has them -- none of them is rescaled, reweighted or re-derived -- and adds one
# term built from the SAME teacher:
#
#     aux_i = 0.5 * || (m_student(s_i) - m_teacher(s_i)) / sd_rollout(s_i) ||^2   (over dims)
#     pg_loss <- pg_loss + jac_aux_coef * (sum_i keep_i * aux_i) / sum_i keep_i
#
# m_teacher is the rollout Beta mean displaced along the ROTATED, DOSE-MATCHED direction
# (u_cred -> rotate_to_cos at jac_cos -> the delivered-KL bisection), i.e. pieces 1-4 above
# are reused verbatim, not rewritten. `keep` is the SAME per-sample mask tensor the
# surrogate just used in the same minibatch, with the same kept-count renormalization, so a
# masked sample contributes to NEITHER term. sd_rollout is DETACHED, so the auxiliary is a
# pure quadratic in the student's MEAN: it is the fixed-concentration Beta KL to second
# order -- the same second-order form the dose bisection is calibrated in -- and it contains
# NO log-density-of-the-student term at all. There is nothing in it that rewards a wider
# Beta. Concentration is reachable only through m = a/(a+b), i.e. only as the minimum-norm
# way to move the mean, never as an entropy incentive.
#
# MEASURED HERE FOR `--improve both` (champion config, seed 1, 262144 steps unless noted):
#
#  1. ZERO DOSE IS BIT-EXACT. `--improve both --jac-aux-coef 0` vs `--improve surrogate` at
#     `--num-envs 4 --num-steps 256 --num-minibatches 4 --total-timesteps 20480 --seed 1`:
#     36 shared scalar tags compared (charts/SPS excluded), 720 points, ZERO differing,
#     NaN-safe comparison. The `both` arm additionally logs 14 jac_* / teach_* / jac_model /
#     aux telemetry tags, which `surrogate` does not emit by design; the tag sets are
#     therefore a strict superset, not a mismatch. So the machinery RUNS at zero dose (the
#     transition head is built, trained and gated, the rotation and the bisection execute)
#     and still changes nothing: this is a genuine additivity test, not a skipped branch.
#  2. ENTROPY DOES NOT LEAK -- the gate the distill design failed. losses/entropy delta over
#     the 8 iterations, identical command line:
#         surrogate            -0.0381      (first -0.4334, last -0.4715)
#         both, aux_frac 1.0%  -0.0321
#         both, aux_frac 1.4%  -0.0388
#         both, aux_frac 2.6%  -0.0323
#         both, aux_frac 5.1%  -0.0394      <- the default
#         both, aux_frac 14.9% -0.0188      <- half the chassis's; the knee starts here
#         both, aux_frac 47.8% +0.0225      <- FAILS: the leak reappears
#     So the mean-shift teacher is NOT intrinsically entropy-leaking; the forward KL was. But
#     the auxiliary is not free either, and finding 6 says why: the actor's clip binds in
#     100% of minibatches, so the auxiliary cannot lengthen the actor step, only turn it, and
#     above ~15% declared dose the direction it takes over from the surrogate costs more
#     sharpening pressure than the chassis can spare. The gate is a DOSE gate: open from 0 to
#     ~5%, closed by ~15%.
#  3. THE MASK STAYS ALIVE, at the NATIVE tr_sample_eps = 0.1 -- the thing distill mode
#     could not do. debug/tr_mask_frac, mean over the 8 iterations (min-max):
#         surrogate 0.00622 (0.0019-0.0083) | 1.0% 0.00594 | 1.4% 0.00707 | 2.6% 0.00637
#         5.1% 0.00610 (0.0026-0.0085) | 14.9% 0.00603 | 47.8% 0.00530 (0.0012-0.0072)
#     Same order as the chassis's own everywhere: never zero (the auxiliary is visible to
#     the trust region) and never inflated (it is not driving the drift). losses/approx_kl
#     agrees: 0.01008 surrogate vs 0.00991 / 0.01046 / 0.01037 / 0.00992 / 0.00956 / 0.00910
#     down the ladder -- the auxiliary does not buy itself extra drift.
#  4. MECHANISM ALIVE at the default: jac_r2 -0.0002 -> 0.916, jac_gate 0 -> 0.685 -> pinned
#     1.0 from iteration 3, jac_rot EXACTLY 0.9500 whenever the gate is open, jac_align
#     -0.044 .. +0.010 (still orthogonal, prediction 1 holds), jac_dose_scale >= 0.9991,
#     jac_aux_kl 3.65e-4 .. 4.02e-4 nats, jac_aux_frac 0.041 .. 0.074,
#     debug/aux_pg_grad_ratio 0.0027 .. 0.0086, debug/actor_clip_sat 1.000,
#     losses/explained_variance 0.842 at 262k (surrogate 0.851). No non-finite scalar
#     anywhere after iteration 1, at any dose.
#  5. 262k RETURN (mean of the last 20 episodes; NOT a verdict -- this lineage's donor was
#     BEHIND PPO at 500k and ahead from 1M):
#         surrogate 193 | 1.0% 214 | 1.4% 203 | 2.6% 226 | 5.1% 241 | 14.9% 168 | 47.8% 161
#     The dose was NOT chosen from this row.
#  6. WHAT THE DOSE ACTUALLY BUYS, which is NOT what jac_aux_kl says. debug/actor_clip_sat
#     reads exactly 1.000 in every iteration at every dose: clip_grad_norm_ renormalizes the
#     actor group in 100% of minibatches, so `pg_loss + aux_term` produces a step of the SAME
#     LENGTH pointing somewhere else. The auxiliary's real cost is therefore the share of the
#     actor's fixed-length step it takes over from the chassis, which is
#     debug/aux_pg_grad_ratio = ||grad_actor(aux)|| / ||grad_actor(chassis pg)||:
#         declared jac_aux_frac   1.4%    5.1%    14.9%   47.8%
#         delivered grad ratio    0.0014  0.0047  0.0153  0.0412     (linear, 0.43 x coef)
#     i.e. the DELIVERED influence is ~10x SMALLER than the declared KL fraction, and the
#     entropy gate closes between a grad ratio of 0.005 and 0.015. A half-percent systematic
#     rotation is enough to matter over 320 updates precisely because it is systematic while
#     the surrogate's minibatch noise partly cancels. debug/jac_aux_kl is a DECLARATION: at
#     the rollout policy the charged gap IS the teacher displacement, so jac_aux_kl is pinned
#     near jac_aux_coef * 0.5 * teach_step^2 * E[A^2] = jac_aux_coef * 0.03125 by
#     construction (measured slope 0.0352, +13%, flat to +-1.1% over a 30x span -- the
#     student does not measurably close the gap at any dose, and the +13% says the chassis
#     step ends slightly FURTHER from the teacher than the rollout policy was). Read
#     jac_aux_kl as the dose asked for and aux_pg_grad_ratio as the dose delivered.
#  7. ONE ACCIDENTAL LONG RUN, reported because it is the only long-horizon evidence that
#     exists and it bears directly on the entropy gate. A `--improve both --jac-cos 0.95
#     --jac-aux-coef 0.032` run picked up the 8M defaults instead of the 262k override and
#     ran 4.096M steps before it was stopped. Over 125 iterations losses/entropy went
#     -0.433 -> -0.503 (@262k) -> -2.46 (@1M) -> -4.24 (@2M) -> -6.07 (@4.1M): the policy
#     sharpens monotonically, so the mildly degraded 262k delta at that dose is a
#     transient-phase reading and not a leak. tr_mask_frac rose 0.0065 -> 0.066 and stayed
#     there; jac_aux_frac DECAYED 0.145 -> 0.044 because the DECLARED dose is fixed in nats
#     while the chassis's own approx_kl grows 0.0086 -> 0.040. That decay is arithmetic, not
#     evidence about influence: aux_pg_grad_ratio did not exist when that run was made, so
#     the delivered share over a long horizon is UNMEASURED. jac_r2 reached 0.986 and
#     jac_align stayed ~0 (-0.026 at 4.1M). Returns 246 @262k / 1111 @500k / 3024 @1M /
#     5632 @2M / 8672 @4M.
#     CAVEATS, stated because this is not a controlled arm: no matched surrogate run of the
#     same length was made (the working constraint was <= 262k steps), the 4M numbers are
#     against the INHERITED chassis figures (3041 @1M, 5465 @2M, 8890 @4M) from a different
#     run, and the anneal_lr schedule of an 8M run is not the schedule of a 262k run, so
#     this run's early iterations are NOT the same as row 5's.
#
# THE DOSE, DECLARED. jac_aux_coef is set from measurement, not taste: debug/jac_aux_kl is
# linear in it (slope 0.0352 nats per unit coefficient, +-1.1% over a 30x span of doses),
# the chassis's own approx_kl at 262k is 0.0101, so a target fraction fixes the number.
# The default 0.011 was set to target 5% and MEASURES 5.05% declared / 0.47% delivered.
# 5% is the top of the band in which BOTH gates -- the entropy delta and the mask fraction
# -- are indistinguishable from the chassis's own arm, with a ~3x margin to the measured
# knee at 15%. Choosing the top of the safe band rather than the bottom is deliberate: under
# a saturated clip the auxiliary's delivered share is an order of magnitude below its
# declared dose (finding 6), so a smaller coefficient buys a mechanism that provably does
# nothing. The 262k return column was not consulted.
#
# CONSIDERED AND REJECTED. (a) Giving the auxiliary its OWN clipped gradient group, the way
# --kl-grad-group does for the KL term, so it could lengthen the actor step instead of only
# turning it. Rejected: that makes the auxiliary able to spend drift the chassis's own trust
# region never authorized, which is the opposite of "additive on top of an unmodified
# surrogate", and the per-sample mask would no longer bound the combined step. (b) Clipping
# the auxiliary per dim at distill_kl_clip, to make it commensurate with the dose meter.
# Rejected: a quadratic PENALTY should pull hardest where the student is farthest from the
# teacher, and a saturated actor clip already bounds what any auxiliary value can do to the
# step, so the clip would only create a dead zone. debug/jac_clamp_frac remains the stated
# kill-condition instrument for the one unbounded path (the SAMPLE_EPS clamp on the teacher
# mean); it is 0 throughout every run measured here.
#
# FALSIFIABLE PREDICTIONS FOR THE 8M RUN
#  1. jac_align stays ~0 for the whole run. If it climbs toward 1 the analytic direction has
#     collapsed onto the credit direction and the mechanism is inert. (Held to 4.1M.)
#  2. jac_gate pins at 1.0 and jac_r2 stays > 0.9. (Held from iteration 3; 0.986 at 4.1M.)
#  3. jac_dose_scale stays near 1, so `jac_cos` is a direction knob and not a dose knob.
#     (Held: >= 0.9991.) If it falls far below 1 the requested angle is mostly unpayable.
#  4. debug/tr_mask_frac tracks the surrogate arm's within a factor of ~2 for the whole run.
#     This is the prediction distill mode FALSIFIED (mask exactly zero) and `both` restores;
#     if the auxiliary ever drives tr_mask_frac far above the surrogate's it has stopped
#     being an auxiliary and is setting the step size.
#  5. losses/entropy falls monotonically, as the surrogate's does. A rising entropy at any
#     dose is the terminal failure of this lineage's teacher and kills the arm. Held to 4.1M
#     at 14.9% initial dose; the default runs at a third of that.
#  6. debug/actor_clip_sat stays at 1.0 and debug/aux_pg_grad_ratio stays below ~0.01. The
#     ratio is the only scalar that measures the auxiliary's INFLUENCE rather than its
#     declared dose; if it climbs past the 0.015 that cost half the chassis's sharpening at
#     262k, the auxiliary has stopped being an auxiliary. debug/jac_aux_frac will decay by
#     construction (fixed nats / growing chassis drift) and proves nothing on its own.
#
# =====================================================================================
# --- INHERITED CHASSIS NOTES BELOW, UNCHANGED (ENT-PPO v4 sample-mask lineage) --------
# ENT-PPO v4 — the leash forfeits the whole batch's epoch budget for a ratio tail.
# =====================================================================================
# ONE change vs the line's champion (entppo_v2_declkl_ppoadvnorm_a03_b03, 10269 @8M,
# +21% over the 8455 reference): when a sample leaves the trust region, mask THAT SAMPLE
# instead of aborting the iteration.
#
# WHY THIS IS THE LEVER. Three measurements from this line, none of which were the thing
# anyone was tuning:
#   * The target_kl=0.03 early stop is LOAD-BEARING, not redundant. It breaks the epoch
#     loop in 69.7% of v1's iterations (91.0% at α=0.1), 39.8% of v2's, 76.8% of the
#     retstd base's and 84.4% of the ppoadvnorm_batch reference's.
#   * α and β are exhausted as axes. α: 3942 / 5443 / 4467 @2M for 0.1 / 0.3 / 0.6 --
#     interior optimum. β: 10076 (β=0) vs 10269 (β=0.3) @8M, i.e. +2% at ±1%. The
#     reference's own faithful trust region (clip + fixed coefficient 1, no leash) scored
#     0.75x the champion at 2M and was cancelled.
#   * v3 established that a mean parametric KL cannot bound the per-SAMPLE importance
#     ratio (exact_kl 0.034 in the same iteration as approx_kl 7.92): the tail needs a
#     per-sample instrument, and the leash is the only thing currently reacting to it.
# So the mechanism that dominates this line is the trust region's ENFORCEMENT, and its
# current form pays for a tail with the entire remaining epoch budget OF THE WHOLE BATCH.
# `debug/epochs_completed` did not exist anywhere in v1/v2, so that cost was never seen.
#
# THE CHANGE. d_i = (ρ_i − 1) − log ρ_i is non-negative and its mean IS approx_kl, so it is
# a per-sample drift in the same units as target_kl. Samples with d_i > tr_sample_eps are
# zeroed out of the PG mean and the mean is renormalized by the KEPT count (renormalizing by
# the full count would shrink the step in proportion to the mask, confounding the test).
# The epoch budget then continues on the in-region samples. tr_sample_eps=0.1 corresponds to
# ρ ∈ ~[0.63, 1.5]; the clip's own 1.28 upper bound is d = 0.033, so the mask removes only
# the tail the clip's pessimistic branch already refuses to reward, while the leash removed
# 32-320 further updates for every sample in the batch.
#
# The 0.03 leash becomes a 0.15 emergency breaker under tr_mode=mask, because the per-sample
# bound is now the trust region. --tr-mode abort restores v2's enforcement exactly, on this
# same code, which is the control arm: it is the first run of this line that measures its own
# epochs_completed.
#
# PASS: ≥ 5443 @2M and ≥ 10269 @8M with debug/epochs_completed materially above the abort
# arm's and debug/tr_mask_frac small (a few %). FAIL: equal or worse return, which would mean
# the forfeited budget was not costing anything and the leash's abort was incidentally acting
# as a useful early stop -- in which case the epoch budget itself is the thing to cut.
# Watch debug/ratio_max in both arms: if masking lets it grow, the mask is too loose.
#
# --- v2 notes (scale-declaration argument; unchanged) ---
# ENT-PPO v2 — DECLARED-SCALE proximal term + the strong advantage recipe.
# =====================================================================================
# WHY v1 UNDERPERFORMED (measured, HalfCheetah, seed 1, matched steps):
#   retstd base (α=0)  2430 @1M / 3740 @2M      <- the recipe v1 was built on
#   v1 α=0.1           2551 / 3942
#   v1 α=0.3           2589 / 4808              <- +28% over its OWN base at 2M
#   ppoadvnorm_batch   3079 / 5012              <- same file, norm_adv_scope="batch"
# Two independent deficits, fixed independently here:
#
# (A) WRONG BASE. v1 inherited norm_adv_scope="batch_retstd" (divide-only by std(returns)),
#     which is ~25% behind plain batch standardization at 2M. The Ent-PPO objective is
#     orthogonal to the advantage scaler, so v2 defaults to norm_adv_scope="batch".
#
# (B) THE PROXIMAL HALF WAS NUMERICALLY INERT. v1 tied the KL coefficient to the reward
#     bonus exactly as the paper does: kl_coef = α/adv_div. Measured in v1 (MEDIANS over
#     all 244 logged iterations; an earlier revision quoted early-training snapshots as if
#     they were run-level): adv_div = S = std(returns) 38.8, kl_coef 0.0077 against
#     exact_kl 0.0301 => the penalty contributed 2.6e-4 to a surrogate of 0.077, and
#     `approx_kl` sat at 0.031, above target_kl=0.03 in 69.7% of iterations (not "0.04-0.05
#     in essentially every iteration"): the α·KL term restrained nothing, the
#     ratio clip and the early-stop leash did all the trust-region work, and only the
#     reward augmentation was live (debug/ent_adv_share ≈ 0.46).
#
#     This is structural, not a coding error. The tie α_reward == α_KL is EXACT only when
#     rewards are log-rewards: in a GFlowNet the return spread is O(1) nat, so α/S ≈ α and
#     one knob legitimately serves both halves. At MuJoCo reward scale the two halves live
#     at different orders: entropy enters the RETURN as α·h/(1−γ) ≈ 100·α·h while the
#     proximal coefficient is α/S ≈ α/25, so their ratio is ~2500·(1−γ)^-1-free — no single
#     α makes both O(1). α ≈ 2.5 buys a real trust region and a reward bonus ~40x the task
#     reward; α = 0.3 buys a usable bonus and no trust region. v1 sat at the second end.
#
#     v2 therefore DECLARES the proximal strength in the units it actually acts in:
#       kl_coef = α·kl_coef_scale/adv_div  +  kl_beta
#                 └ paper-faithful piece ┘    └ declared, in NORMALIZED advantage units ┘
#     kl_beta multiplies the exact KL against an advantage that is already O(1), so
#     kl_beta = 0.3 puts the penalty at 0.008 vs a surrogate of 0.017 (MEASURED medians in
#     entppo_v2_declkl_ppoadvnorm_a03_b03; ~0.5x, not the ~15% predicted from v1's larger
#     pre-batch-norm surrogate of 0.077) — a trust region with
#     real bite. (v2 shipped it ALONGSIDE the ratio clip and the leash, not instead of them;
#     see v3 above, which is the actual replacement.) The faithful
#     piece is retained (not deleted) so the paper's objective is still present and the
#     decomposition stays auditable: debug/kl_coef_faithful vs debug/kl_coef.
#
# WHAT IS STILL ENT-PPO: both halves of the derivation are intact — the soft MDP
# (r̃_t = r_t + α·(−log π_old), GAE-propagated, critic regresses the soft return) and the
# EXACT all-action analytic KL(π_new‖π_old) as the entropy/proximal term. v2 changes only
# the coefficient's *scale declaration*, i.e. it fixes the unit degeneracy that the port
# from log-reward MDPs to reward-scale MDPs creates.
#
# PREDICTION (pre-registered): kl_beta>0 pulls approx_kl below target_kl, so the leash
# stops firing and the policy spends its full 10-epoch budget on a trust region that is
# analytic and all-action instead of sampled and clipped. PASS: ≥ ppoadvnorm_batch
# (3079 @1M / 5012 @2M / 6716 @4M / 8455 @8M). FAIL: >5% below it at two consecutive
# checkpoints, or approx_kl unchanged (⇒ kl_beta still too small).
# CONTROL: the same file with --kl-beta 0 isolates soft-MDP-only on the strong base.
#
# --- v1 notes (the derivation itself; unchanged) ---
# ENT-PPO v1 — entropy-regularized (soft-MDP) PPO with an EXACT analytic proximal term,
# ported from ent-ppo (Zykova-Myzina, Gritsaev, Tiapkin, Morozov; ICML 2026 SPIGM workshop,
# arXiv 2606.15793) onto the retstd_batch recipe (6364 @7.2M reference run).
# =====================================================================================
# WHAT ENT-PPO ACTUALLY IS (their gfnx/baselines/PPO_*.py): a GFlowNet is a soft-optimal
# policy of an entropy-regularized MDP, so their per-step reward is
#   r̃_t = log R + log P_B(s_t|s_{t+1}) − log P_F(a_t|s_t)
# (the `− log π_old` term IS the pathwise entropy bonus), GAE runs on r̃, the baseline
# regresses the SOFT return, and the surrogate is
#   min(ρÂ, clip(ρ)Â)  −  KL(π_new‖π_old)          [exact, all-action, coefficient 1]
# with NO entropy bonus and NO free KL coefficient. Two ingredients, one knob:
#
# (1) SOFT MDP: r̃_t = r_t + α·(−log π_old(a_t|s_t)). Because the bonus rides inside the
#     reward, GAE propagates it: an action is credited for the entropy of everything it
#     leads to (future-entropy credit), and the critic regresses the soft return. PPO's
#     `ent_coef·H(π(·|s_t))` is the myopic special case — current step only, α arbitrary.
# (2) EXACT PROXIMAL TERM: at state s the soft improvement objective is
#       E_{a~π}[Q̃] + α·H(π)
#     and since α·H(π) = −α·KL(π‖π_old) − α·E_{a~π}[log π_old(a)], while the second piece
#     is ALREADY carried by Ã_soft (the −α·log π_old(a_t) sitting in the augmented reward),
#       E_{a~π}[Q̃] + α·H(π) = E_old[ρ·Ã_soft] − α·KL(π_new‖π_old) + const.
#     So the analytic KL is not a heuristic trust region bolted onto PPO: paired with the
#     reward's pathwise surprisal it is an EXACT estimator of α·H(π_new), and its
#     coefficient is FORCED to be the same α that augments the reward. The two knobs PPO
#     treats as independent (ent_coef, KL/clip strength) are one quantity here.
#
# WHY IT SHOULD HELP HERE: the sampled-action surprisal −log π_old(a_t) is an exploration
# bonus that is (a) state-conditioned, (b) temporally propagated, and (c) exactly
# compensated by the KL term, so it perturbs the objective without biasing it toward
# uniform noise. Beta concentration collapse (log-density → +∞ ⇒ bonus → −∞) is penalized
# in RETURN space, i.e. the critic sees it coming k steps ahead instead of the actor eating
# a myopic bonus. Reference PPO here runs ent_coef=0 exactly because the myopic bonus was
# useless; this is the non-myopic version of the same idea.
#
# UNITS (the one thing a naive port gets wrong): the recipe divides the advantage by
# S = std(batch returns) before the PG. A proximal objective is only well-posed if the
# SAME divisor hits the KL term, so kl_coef = α/S (α·KL and Â/S in one geometry). All
# advantage divisors (retstd, percnorm, standardization) are tracked into `adv_div`. Two
# inherited flags put a factor OUTSIDE that product and are therefore asserted off while α>0:
# a non-identity `adv_transform` (non-homogeneous: rankgauss* throw the magnitude away) and
# `pos_neg_alpha != 0.5` (per-sample sign-dependent reweight). Both are already at the
# reference recipe's values, so the default run is unaffected.
#
# α is the sole new hyperparameter. α=0 recovers the retstd_batch reference exactly
# (same RNG stream: no extra sampling), so this file is its own ablation. Rewards here
# are ~6/step and Beta log-densities ~1 nat/dim over 6 dims, so α=0.1 puts the entropy
# term at ~10% of the return; α is in RAW reward units, like the paper's log-space α=1.
#
# DELIBERATELY NOT PORTED (with reasons, so the failure analysis is honest):
#   - their fresh-bootstrap TD(λ) value target (V_next recomputed with the LIVE critic each
#     value epoch): incompatible with this critic, which is a 6-horizon bucket-CE head whose
#     HL-Gauss labels are projected once per rollout on CPU.
#   - their full-batch policy epochs + separate value optimizer/minibatch splits: that is
#     GFlowNet-scale plumbing, orthogonal to the objective.
#   - their no-advantage-normalization: the reference recipe's retstd divisor is kept, since
#     the point is a one-change comparison against the 6364 run.
#   - target_kl=0.03 leash retained. REFUTED as "redundant": v1 exceeded 0.03 in 69.7% of
#     its 244 iterations at α=0.3 and 91.0% at α=0.1, so the leash was the load-bearing
#     trust region for the whole v1 experiment. It is also a CONFOUND in the α ladder: the
#     α=0.3 arm forfeited ~30% more of its epoch budget than α=0.1 did, so 4808-vs-3942
#     is not a clean isolation of the objective. (Both baselines trip more often still --
#     retstd 76.8%, ppoadvnorm_batch 84.4% -- so v1 was not unusually leashed.)
#     Original (wrong) note: it is now redundant with the α·KL penalty; if the
#     penalty is doing its job, `losses/approx_kl` stops tripping it and the policy gets
#     more of its 10-epoch budget. That is a mediated effect of the method, not a
#     confound — watch losses/approx_kl and losses/exact_kl together.
#
# PASS: ≥ retstd_batch reference at matched steps (which run peaked ~6364 @7.2M).
# FAIL: >5% below the reference at two consecutive checkpoints, or
#       debug/target_edge_mass rising off ~0 (soft return overflowing the ±20k support).
# WATCH: debug/ent_adv_share (entropy fraction of the policy signal; ≫1 ⇒ α too large),
#        debug/entropy_bonus_mean (= mean −log π_old, the realized per-step entropy),
#        losses/exact_kl vs losses/approx_kl, debug/kl_coef.
#
# --- Base (unchanged below): PPO + IterThink v24 Beta + v162 critic + Dreamer3-bucket
#     HL-Gauss MTP + MB PERCNORM v2 ---
# =====================================================================================
# v2 = v1 (d3percnorm) but the percentile advantage scaler is computed PER-MINIBATCH with NO EMA
# (ret_perc_scope="minibatch"), i.e. the per-mb analog of the old advnorm. Each minibatch divides its
# advantage by S = max(1, P95-P5) of THAT minibatch's RAW RETURNS (`mb_ret` -- the same return-spread
# statistic v1's EMA tracked, per the design intent), recomputed fresh every minibatch. Same divide-only,
# same return-spread source, same floor; ONLY the timing/memory differs: local & reactive instead of a
# slow global EMA. This isolates the EMA's contribution -- does dreamer3's slow global percentile scale
# beat a fresh per-minibatch one? Everything else == v1: no rankgauss (adv_transform="v10"), norm_adv off,
# raw reward, RAW-space Dreamer3 511-bucket symexp critic, PPO clip-higher (0.2/0.28) trust region.
# (--ret-perc-scope ema reproduces v1's global-EMA scaler exactly.) Watch charts/ret_perc_scale (now the
# last minibatch's S) -- expect it ~v1's magnitude (return spread is value-dominated, stable across mbs of
# 1024 samples) but noisier/reactive vs v1's EMA-smoothed value.
# =====================================================================================
#
# --- v1 method (changes vs the base, retained below) ---
# Variant of v162critic_dreamer3bucket_hlgauss_mtp_v1: ports DreamerV3's advantage-scale
# stack (as in dg_beta v15/v16) onto this MTP base. THREE changes vs the base:
#   (1) NO rankgauss: adv_transform="v10" (identity). The base's flagship rank-Gaussian
#       shaping is removed -- it already maps advantages to ~N(0,1), so a percentile
#       norm on top of it would be a constant-divide no-op. ("no advantage norm")
#   (2) NO norm_adv: per-minibatch standardization off. ("no advantage norm")
#   (3) DreamerV3 PERCENTILE NORM is the SOLE advantage scaler: policy_adv <- policy_adv /
#       max(1, EMA(P95)-EMA(P5)) over the raw GAE returns (EMA rate 0.01, divide-only).
# Reward stays RAW (base already defaults normalize_reward=clip_reward=False). The CRITIC
# stays in RAW space -- unchanged Dreamer3 511-bucket symexp HL-Gauss MTP head regressing
# raw returns (DreamerV3 valnorm=none; same arrangement as dg_beta v15/v16). Trust region
# is the base's PPO clip-higher (0.2/0.28), which bounds the step at any advantage scale.
#
# HYPOTHESIS: faithful DreamerV3 normalization (percentile spread + hard floor, raw-return
# symexp critic) on the strong MTP base is at least as stable as rankgauss while being a
# cleaner, more principled scaler. Falsifiable: if it underperforms the rankgauss base, the
# rank-only advantage (magnitude-discarding) was doing real work that a pure scale can't
# replace. Watch charts/ret_perc_scale (the EMA percentile spread S).
# =====================================================================================
#
# --- Base method (unchanged below) ---
# Hypothesis: keep iterthink_v24_beta_s1's PPO/Beta actor and ThinkTrunk, but
# replace the v24 distributional lambda-return critic with the v162 critic plus
# Dreamer3-style exponentially spaced value buckets over v162's raw range:
#   - value bucket centers are symexp(linspace(symlog(-20000), symlog(20000), 511))
#   - HL-Gauss target mass is integrated over the matching symlog-coordinate
#     bucket intervals instead of two-hot interpolation
#   - expected-scalar decode E[symexp(bin)] for values and bootstraps
#   - bias-free neutral critic logits instead of a peaked zero prior
#   - critic MTP head predicting returns[t + h] for h=0..5 with boundary masks
# This isolates whether Dreamer3's high-resolution near-zero / wide-tail bucket
# geometry improves the already-strong v162 critic port, without importing the
# v162 world model, CUDA graph path, or imagined updates.
#
# Base: ppo_continuous_action_iterthink_v24_dist.py / iterthink_v24_beta_s1.
# Critic donor: ppo_continuous_action_iterthink_v162_compiled_wmloss_cudagraph_edgeclamp_contdisc_k6.py.
#
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The critic target is deliberately entropy-free so the fixed support remains
# calibrated. In this variant the target is v162 scalar-return HL-Gauss MTP over
# Dreamer3-spaced buckets.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal Beta, exactly dreamer4's continuous_dist_type='beta' (which
#       forces unimodal=True) and our beta_relusq:
#           alpha = 1 + softplus(head_a);  beta = 1 + softplus(head_b)   (>=1 => unimodal)
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = state-dependent Gaussian scale):
#       dreamer4's Gaussian readout. This is NOT SAC's exact log-std head. It is a
#       state-dependent log-VARIANCE head (not a flat Parameter, not log-std),
#       SOFT-bounded by dreamer4's tanh-rescale (not a hard clamp, so the gradient
#       never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink/SAC tanh-squash + stable Jacobian on the sample (mean
#       stays raw). SAC continuous-action instead uses a state-dependent log_std
#       head bounded to [-5, 2] and std = exp(log_std). Here logvar [-8, 8] implies
#       log_std [-4, 4], so the family matches but the scale parameterization and
#       bounds do not.
#
# PARITY NOTES (both dists): the rollout buffers the distribution-NATIVE sample
# (latent_zs) — pre-tanh z for gaussian, z in (0,1) for beta — and replays it on
# the update pass, so log_prob is recomputed at the same sample (identical to
# v21's z-replay). `actions` holds the env action (tanh(z) / rescaled z). The
# gaussian path is bit-identical to v21 except the flat logstd -> dreamer4 head.
# Bar to beat: v21 flat-Gaussian = 8774.
#
# --- inherited v21 notes ---
# PPO + IterThink v21 (SHARED BACKBONE + DECOUPLED GRAD CLIP). From v19.
#
# WHY v21. v19 used two independent ThinkTrunks (one actor, one critic). The
# classic MuJoCo-PPO result is that shared backbones LOSE, because the value
# loss gradient dominates the shared trunk and corrupts the policy's features.
# v21 tests whether we can have the representation-sharing benefit WITHOUT that
# cost, by decoupling the gradient magnitudes:
#   - share_backbone: one ThinkTrunk feeds both the actor head and the
#     (distributional) critic head; trunk is computed once per forward.
#   - separate_grad_clip: DUAL-BACKWARD clipping. The value gradient
#     (vf_coef * v_loss) and the policy gradient (pg_loss - ent) are each
#     backpropped and clipped to their OWN max-norm (critic_grad_clip /
#     actor_grad_clip), then summed on the shared trunk:
#         trunk.grad = clip_actor(d pg / d trunk) + clip_critic(d vl / d trunk)
#     so the distributional critic's large CE gradient can no longer swamp the
#     shared features. NOTE: the trunk's effective budget is the SUM of the two
#     clips, so each defaults to 0.25 (sum ~= v19's single 0.5 global clip).
# This is targeted: rankgauss already bounds the POLICY gradient (rank-only adv),
# so the dominant imbalance on a shared trunk is the critic -> clip it apart.
# Built on the v19 winner: adv_transform="rankgauss" + clip-higher (0.2/0.28).
# Both knobs are toggles, so this file also runs the {shared,separate} x
# {global,decoupled-clip} 2x2. The bar to beat: rankgauss_cliphigh ~= 8292 (towers).
#
# --- inherited v19 notes ---
# PPO + IterThink v19 (ADVANTAGE SHAPING — magnitude-preserving + attribution). From v17.
#
# WHY v19. A subagent review of v17 (CDF-rank distributional PG) found that in its
# STABLE regime the categorical critic is overconfident, so u=F_Z(G) is bimodal at
# 0/1, the probit saturates, and the advantage DEGENERATES to ≈sign(GAE) (corr 0.92);
# norm_adv then re-standardizes the ±3.3 spikes to ≈±1 binary. So v17 discards the
# advantage MAGNITUDE (the thing PPO needs) and is really a sign-of-TD-error update
# made trainable by KL control. v17's 5867@4M conflates THREE possible causes — the
# distribution, a bounded/outlier-robust advantage, and KL control — introduced at
# once. v19 disentangles them and adds the principled fix, via one `adv_transform`:
#
#   "v10"      : raw GAE (== v10 / dist_pg off). Baseline.
#   "cdf_probit": v17's CDF-rank u -> Phi^-1(u). Reference.
#   "tanh_std" : A~ = tanh( GAE_t / (kappa * sigma(s_t)) ).  THE FIX. Per-state
#                normalized by the critic's return std sigma(s) (v16's good idea),
#                but BOUNDED by tanh (fixes v16's blowup: tiny sigma -> saturate, not
#                explode) AND magnitude-preserving near 0 (fixes v17's sign-collapse:
#                linear in GAE for |GAE|<kappa*sigma). Note G_t-E[Z_t]=GAE_t exactly.
#   "tanh_gae" : A~ = tanh( zscore(GAE)_t / kappa ).  Robust-GAE CONTROL with NO
#                distribution — isolates "bounded/outlier-robust advantage" from the
#                distributional claim. If this matches v17, the distribution is
#                incidental and this is the cleaner lever.
#
# All paths keep the mean-value GAE. This variant changes the value target from
# v24's distributional lambda-return to v162 scalar-return HL-Gauss MTP over
# Dreamer3 buckets; only the policy advantage transforms are selected by
# `adv_transform`. sigma(s) is the std of the OLD rollout Z(s_t), floored at
# `sigma_floor_bins` bins.
import os
import random
import time
from dataclasses import dataclass
from math import log, sqrt
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def value_support_bounds(args):
    """Return critic support endpoints in the coordinate system used by bins."""
    return args.v_min, args.v_max


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form.

    PORTED VERBATIM from ppo_continuous_action_opsd_jacteach_v1.py. Used twice: as the
    student's loss (teacher -> student) and, at snapshot time, as the closed-form dose
    meter the rotation is bisected against.
    """

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


def rotate_to_cos(u_cred, g_hat, cos_t):
    """Rotate each row of u_cred toward g_hat by EXACTLY cosine cos_t, in the plane the
    two span, preserving ||u_cred|| per row.

    PORTED VERBATIM (as a function, so it is testable in isolation) from
    ppo_continuous_action_opsd_jacteach_v1.py.

    The first version of this lever added `step * g_hat` and renormalized. That is
    scale-dependent and it broke in practice: the same step that turned the teacher 57
    degrees at batch 1024 turned it 21 degrees at batch 32768, because the achieved angle
    depends on ||u_cred||, which drifts through training. A lever whose meaning moves with
    batch size cannot support a matched-step ladder, so the angle is specified DIRECTLY:

        u_new = ||u_cred|| * (cos_t * u_hat + sin_t * g_perp)

    cos_t = 1 is therefore an ALGEBRAIC identity, not a skipped branch. Degenerate rows
    keep u_cred exactly: a zero credit direction has no plane to rotate in, and an analytic
    direction parallel to it (or zeroed by the validity mask) has no orthogonal component
    to rotate toward.
    """
    u_norm = u_cred.norm(dim=-1, keepdim=True)
    u_hat = u_cred / u_norm.clamp_min(1e-12)
    g_perp = g_hat - (g_hat * u_hat).sum(-1, keepdim=True) * u_hat
    perp_norm = g_perp.norm(dim=-1, keepdim=True)
    g_perp = g_perp / perp_norm.clamp_min(1e-12)
    sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0))
    u_rot = u_norm * (cos_t * u_hat + sin_t * g_perp)
    ok = (u_norm > 1e-9) & (perp_norm > 1e-9)
    return torch.where(ok, u_rot, u_cred)


class TransitionHead(nn.Module):
    """One-step action-conditioned state-delta model: g(s, a) -> standardized ds.

    PORTED VERBATIM from ppo_continuous_action_opsd_jacteach_v1.py.

    DELIBERATELY NOT ON THE SHARED TRUNK. Routing model gradients through the actor's
    representation would confound "the analytic direction helps" with "predicting the
    next state is a good auxiliary task". This is a separate 2-layer MLP on raw obs +
    action; at hidden=64 it is a rounding error next to the env loop.

    Targets are standardized per dimension with running statistics, because the qpos and
    qvel blocks of a MuJoCo observation differ by ~26x in action sensitivity and an
    unstandardized MSE would fit only the velocities.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, obs_dim), std=0.01),
        )
        self.ds_mean: torch.Tensor
        self.ds_var: torch.Tensor
        self.register_buffer("ds_mean", torch.zeros(obs_dim))
        self.register_buffer("ds_var", torch.ones(obs_dim))

    def forward(self, obs, action):
        """Standardized delta prediction (the space the regression loss lives in)."""
        return self.net(torch.cat([obs, action], dim=-1))

    def next_obs(self, obs, action):
        """Predicted next observation in RAW units -- what V(s') must be evaluated on."""
        ds = self.forward(obs, action) * self.ds_var.sqrt().clamp_min(1e-6) + self.ds_mean
        return obs + ds

    @torch.no_grad()
    def update_stats(self, ds, beta):
        self.ds_mean.mul_(beta).add_((1.0 - beta) * ds.mean(0))
        self.ds_var.mul_(beta).add_((1.0 - beta) * ds.var(0))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True            # retstd_batch reference recipe: divide-only by the batch return std.
    # --- Percentile advantage normalization (the sole advantage scaler) ---
    ret_percnorm: bool = False       # retstd_batch reference recipe: OFF (norm_adv_scope="batch_retstd" is
    #                                  the sole advantage scaler). When on: S = max(floor, P95-P5) of returns.
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0            # Ent-PPO: keep 0. The α·KL term IS the entropy bonus (see header).
    # --- ENT-PPO: the single knob. α augments the reward with the pathwise entropy
    # −log π_old(a_t|s_t) AND sets the coefficient of the exact KL(π_new‖π_old) proximal
    # term; the derivation forces them to be the same number. α=0 => the base recipe.
    ent_alpha: float = 0.3           # v1 ladder: 0.3 > 0.1 > 0 at matched steps (4808/3942/3740 @2M)
    # Ablation-only multiplier that BREAKS the derivation's tie between the reward bonus and
    # the proximal coefficient (1.0 = faithful Ent-PPO). Use it to test whether the exact
    # coefficient is actually the right one, not as a tuning knob.
    kl_coef_scale: float = 1.0
    # v2: proximal strength DECLARED in normalized-advantage units, added to the faithful
    # α/adv_div piece. This is the operative trust region: at MuJoCo reward scale the
    # faithful coefficient is ~0.008 in v1 / 0.023 in v2 (measured medians) and cannot
    # restrain drift (see header). 0 => v1 behavior.
    kl_beta: float = 0.3             # v3: INITIAL β; adapted each iteration when kl_adapt
    # v3: --no-pg-clip removes the ratio clip and makes the exact KL the sole trust region.
    # FALSIFIED (see header); the default keeps the clip and uses the KL as a SECOND,
    # distributional constraint alongside it.
    pg_clip: bool = True
    # v4: off by default -- kept for the v3 diagnosis, not part of v4's change.
    # v3: give β authority over the actor step MAGNITUDE. With one clipped group the
    # summed grad is renormalized to actor_grad_clip in 100% of iterations (measured),
    # so β could only ROTATE the step -- which is why β railed 0.3 -> 4.8 while realized
    # KL rose. Clipping the KL gradient in its own group and summing WITHOUT
    # renormalization makes β change magnitude, as the control law assumes.
    kl_grad_group: bool = False  # v4: v2's single clipped group (isolate the ONE change)
    kl_grad_clip: float = 1.0
    kl_adapt: bool = False   # v4: β fixed at the champion's 0.3; adaptation bought nothing            # PPO-penalty rule on the EXACT KL (Schulman 2017 §4)
    kl_target: float = 0.03          # per-iteration drift budget (= v2's leash value, by design)
    kl_beta_min: float = 1e-3
    kl_beta_max: float = 1e3
    # v3: measure ||grad_actor(kl_penalty)|| / ||grad_actor(pg_loss)|| once per epoch. >~1 means
    # the trust region, not the objective, is deciding the actor's clipped update direction.
    log_grad_split: bool = True
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    # v3: no longer the trust region -- a CIRCUIT BREAKER at 5x kl_target. β adapts only once
    # per iteration, so this aborts a pathological iteration's remaining epochs. Expect ~no trips.
    target_kl: float = 0.03
    # v4: what the trust region DOES when a sample leaves the region.
    #   "abort" = v2/v1 exactly: break the epoch loop, forfeiting the rest of the epoch
    #             budget FOR THE WHOLE BATCH because of a tail.
    #   "mask"  = zero the offending SAMPLES out of the PG mean and keep going, so the
    #             in-region samples still get their full epoch budget.
    tr_mode: str = "mask"
    # Per-sample drift budget for "mask", in the same units as approx_kl:
    #   d_i = (ρ_i − 1) − log ρ_i  (>= 0, mean over i IS approx_kl).
    # 0.1 keeps everything inside ρ ∈ ~[0.63, 1.5] (the clip's 1.28 upper bound is d=0.033)
    # and removes exactly the heavy tail that a mean parametric KL cannot see.
    tr_sample_eps: float = 0.1
    # Emergency breaker under "mask" (the per-sample bound is the trust region now).
    target_kl_breaker: float = 0.15

    # === THE COMBINATION ===========================================================
    # `improve` selects what the actor's improvement signal IS.
    #   "surrogate": the chassis parent, BIT FOR BIT. The transition head is neither built
    #                nor trained and no RNG is drawn, so every later minibatch permutation
    #                and every rollout sample is unchanged.
    #   "distill":   the PPO clipped surrogate is REPLACED by the donor's per-dim forward
    #                KL into a mean-shifted teacher, whose displacement direction is the
    #                chassis's own credit direction ROTATED toward an analytic direction
    #                taken through a learned one-step transition model, at exactly
    #                cos = jac_cos. The chassis's per-sample KL-drift mask stays ON and is
    #                applied to the distillation loss the same way it is applied to the
    #                surrogate (mask the sample, renormalize by the KEPT count).
    #                MEASURED FAILURE -- see the header: entropy leaks UPWARD at every dose.
    #   "both":      the chassis's clipped surrogate, its entropy term, its per-sample mask
    #                at the NATIVE tr_sample_eps and its value loss are all UNTOUCHED, and
    #                the same rotated, dose-matched displacement enters as ONE ADDITIVE
    #                auxiliary: a quadratic pull of the STUDENT's Beta mean toward the
    #                teacher mean, in the rollout policy's own KL metric, weighted by
    #                jac_aux_coef. At jac_aux_coef = 0 the arm is bit-exact to "surrogate".
    #   "rotate":    v2, THE DEFAULT ARGUMENT OF THIS FILE'S HEADER. Nothing is added to the
    #                objective and nothing is replaced: the chassis's own mean-space ascent
    #                direction g_ch = -c*(d pg/d alpha - d pg/d beta) is ROTATED to exactly
    #                cos = jac_cos toward the analytic direction, NORM-PRESERVINGLY, and a
    #                linear corrector -(g_rot - g_ch)*m_new makes the total mean-space
    #                gradient equal g_rot. The corrector has identically zero concentration
    #                gradient, so it carries no entropy pressure at all, and it changes the
    #                DIRECTION of the update without changing its SIZE -- which is the
    #                property whose absence made "distill" leak entropy and "both" too weak
    #                to matter. At --jac-cos 1.0 the corrector is exactly absent and the arm
    #                is bit-exact to "surrogate".
    #                MEASURED VERDICT: THIS MODE FAILS. The rotation is delivered exactly
    #                (rot_cos_check == jac_cos, rot_norm_check == 1.00000) and the arm still
    #                blows the mask gate 20x and the entropy slope 36x at jac_cos 0.95,
    #                because a saturated actor clip turns the corrector's SHORTER parameter
    #                gradient (rot_pgrad_ratio 0.624) into a 2.4-4.1x ANGULAR AMPLIFICATION
    #                of the requested turn. See findings 5, 6 and 9 in the header. Kept as a
    #                falsification, not as a candidate.
    improve: str = "surrogate"
    # TEACHER DOSE, declared in the policy's own metric. u = (mean displacement)/sd, and
    # Beta KL at fixed concentration is quadratic in u (KL ~ 0.5*||u||^2 summed over dims),
    # so teach_step is a displacement in standard deviations. The per-sample displacement is
    #     u_cred = teach_step * A_i * unit(credit direction)
    # with A_i the chassis's OWN batch-standardized advantage (E[A^2] = 1), so
    # teach_step = 0.25 puts the delivered summed KL at ~0.03 -- exactly the chassis's own
    # per-iteration drift budget (kl_target), and a fifth of the per-sample mask's 0.15
    # breaker. Delivered value is MEASURED and logged as debug/teach_kl, not assumed.
    teach_step: float = 0.25
    distill_coef: float = 1.0     # weight on the teacher->student divergence
    distill_kl_clip: float = 2.0  # tau: the donor's per-dim pointwise divergence clip. The
    #                               dose match is defined against THIS clipped KL, because
    #                               that is what the loss actually charges.
    # THE COSINE OF THE TEACHER ROTATION, exactly. 1.0 = the credit direction alone, as an
    # ALGEBRAIC identity rather than a skipped branch (the head is still built and trained,
    # so `--improve distill --jac-cos 1.0` is the dose/mechanism control for free).
    jac_cos: float = 0.95
    jac_anneal_frac: float = 0.0  # v3: fraction of training over which the rotation dose
    # folds linearly to the identity (cos -> 1.0). 0 disables (= v2 exactly). The measured
    # rate/ceiling crossover is at ~4M of 8M, i.e. 0.5.
    jac_hidden: int = 64        # one-step transition head width (= trunk width)
    jac_coef: float = 1.0       # weight on the model's own regression loss
    jac_dose_iters: int = 16    # bisection steps for the delivered-KL dose match
    jac_r2_min: float = 0.5     # gate closes fully at or below this held-out one-step R2
    jac_r2_open: float = 0.8    # gate fully open at or above
    # WEIGHT ON THE ADDITIVE AUXILIARY (improve="both" only). The auxiliary is already
    # expressed in nats (it is the fixed-concentration Beta KL between the student mean and
    # the teacher mean, 0.5*||(m_new - m_teach)/sd_rollout||^2 summed over dims), so this
    # coefficient is a pure dose knob and debug/jac_aux_kl reads the charge it places on the
    # actor objective in the SAME units as losses/approx_kl. That charge is the dose ASKED
    # FOR; the dose DELIVERED is debug/aux_pg_grad_ratio, ~10x smaller, because the actor's
    # clip binds in 100% of minibatches (debug/actor_clip_sat = 1.000) so this term turns
    # the actor step rather than lengthening it.
    #
    # THE DEFAULT IS MEASURED, NOT CHOSEN. The delivered charge is linear in the coefficient
    # (measured slope debug/jac_aux_kl = 0.0352 * jac_aux_coef, +-1.1% over a 30x span), and
    # the chassis's own approx_kl at 262k is 0.0101, so a target fraction fixes the number.
    # 0.011 was set to target 5% and MEASURES 5.05% (debug/jac_aux_frac, mean of 8
    # iterations). 5% is the TOP of the band in which BOTH gates are indistinguishable from
    # the chassis's own arm (delivered grad ratio 0.0047); see the dose ladder in the
    # header. It is not tuned for return.
    # 0.0 is the exact no-op (bit-exact to --improve surrogate).
    jac_aux_coef: float = 0.011

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic: centers are symexp(linspace(v_min, v_max, num_bins)).
    # Defaults match v162's ±20k raw support, expressed in symlog coordinates.
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"     # v2: the STRONG recipe (8455 @8M) -- retstd is 25% behind at 2M

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
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
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v162 critic: bias-free neutral MTP head. With symmetric symlog support,
        # zero logits decode to a zero raw value without a hidden prior.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        # v2: the two concentration tensors are the ONLY graph nodes every gradient path of
        # a torch Beta flows through (Beta stacks them into a Dirichlet, and log_prob /
        # entropy both read that stack), so `improve=rotate` needs handles on them to take
        # d(surrogate)/d(mean) by the chain rule alpha = m*c, beta = (1-m)*c. Reading
        # dist.concentration1 instead does NOT work: that property is a fresh select off the
        # stacked tensor, i.e. a SIBLING branch of the log_prob path, and autograd.grad
        # against it raises "not used in the graph". Attaching the pre-stack tensors here is
        # numerically inert -- it is an attribute assignment, no op is added to the graph --
        # so every other mode stays bit-exact.
        dist.conc_nodes = (alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value_logits, dist

    # ENT-PPO: the analytic KL(π_new‖π_old) needs the OLD distribution, so the rollout
    # stores its two head parameters and the update rebuilds it (parameters, not samples:
    # the KL is all-action/exact, not an importance-weighted single-sample estimate).
    def dist_params(self, dist):
        if self.actor_dist == "gaussian":
            return dist.loc, dist.scale
        return dist.concentration1, dist.concentration0

    def rebuild_dist(self, p1, p2):
        if self.actor_dist == "gaussian":
            # KL is invariant under the tanh bijection, so base-Normal KL == squashed KL.
            return Normal(p1, p2)
        return Beta(p1, p2)

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
    # Both inject entropy into the advantage; together they double-count it (and auto_entropy
    # keeps the critic entropy-free, which contradicts Ent-PPO's soft value function).
    assert args.tr_mode in ("mask", "abort"), f"unknown tr_mode {args.tr_mode!r}"
    assert args.tr_sample_eps > 0.0, "tr_sample_eps must be positive"
    assert args.improve in ("surrogate", "distill", "both", "rotate"), \
        f"unknown improve {args.improve!r}"
    assert not (args.improve in ("surrogate", "both", "rotate") and args.tr_mode == "mask" and not args.pg_clip), \
        "tr_mode=mask needs the clipped per-sample surrogate (--pg-clip)"
    assert args.jac_aux_coef >= 0.0, "jac_aux_coef is a dose, not a sign"
    if args.improve in ("distill", "both", "rotate"):
        # The teacher is a Beta MEAN SHIFT at fixed concentration, so the Beta head is
        # load-bearing; and it is snapshotted once per iteration, so the advantage it is
        # weighted by has to be the batch-scope one the surrogate would have used.
        assert args.actor_dist == "beta", f"improve={args.improve} needs the Beta policy"
        assert args.norm_adv and args.norm_adv_scope in ("batch", "batch_retstd"), \
            f"improve={args.improve} needs a batch-scope normalized advantage (the teacher is one snapshot per iteration)"
        assert args.teach_step > 0.0, "a zero teacher displacement is a no-op teacher"
    assert not ((args.ent_alpha != 0.0 or args.kl_beta != 0.0) and args.auto_entropy), \
        "ent_alpha (soft-MDP reward, Ent-PPO) and --auto-entropy (soft-advantage bootstrap) are mutually exclusive"
    assert not ((args.ent_alpha != 0.0 or args.kl_beta != 0.0) and args.ent_coef != 0.0), \
        "ent_coef must stay 0 under Ent-PPO: the alpha*KL proximal term already is the entropy bonus"
    # The proximal term is only well-posed if `adv_div` really is the full scale applied to the
    # advantage. Two supported flags put a factor OUTSIDE that product, so refuse them here
    # rather than silently mis-scaling alpha: (a) any non-identity adv_transform is a
    # non-homogeneous map (tanh saturates; rankgauss* discard the GAE magnitude outright), so
    # after it the advantage is not in return units at all; (b) pos_neg_alpha reweights each
    # sample by a SIGN-dependent factor, which no scalar divisor can represent.
    assert not ((args.ent_alpha != 0.0 or args.kl_beta != 0.0) and args.adv_transform != "v10"), \
        "Ent-PPO needs adv_transform=v10: a non-identity shaping leaves the advantage in unknown units, so alpha/adv_div is no longer the derivation's coefficient"
    assert not ((args.ent_alpha != 0.0 or args.kl_beta != 0.0) and args.pos_neg_alpha != 0.5), \
        "Ent-PPO needs pos_neg_alpha=0.5: its per-sample sign-dependent reweight cannot be folded into adv_div"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
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
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # improve="surrogate" must be BIT-EXACT against the chassis parent, so the head is
    # neither built nor trained in that case; and when it IS built, its init draws happen
    # inside fork_rng, because orthogonal_ would otherwise advance the global RNG stream and
    # shift every later minibatch permutation and every rollout sample. (This discipline is
    # the donor's, kept verbatim -- it is what makes the control arm exact.)
    use_jac = args.improve in ("distill", "both", "rotate")
    # "both" keeps the chassis objective and ADDS one term; the term is built from the same
    # teacher the "distill" arm distils into, so the machinery above is shared verbatim.
    use_aux = args.improve == "both"
    # v2: "rotate" adds nothing and replaces nothing. It reads the chassis's OWN mean-space
    # ascent direction off the retained graph and turns it, at fixed length.
    use_rot = args.improve == "rotate"
    if use_rot:
        # THE ONE ALGEBRAIC ASSUMPTION THIS MODE RESTS ON, CHECKED AT STARTUP RATHER THAN
        # ASSUMED: that d/dm at FIXED concentration equals c*(d/dalpha - d/dbeta) on a torch
        # Beta reached through `dist.conc_nodes`, i.e. that the handles attached in
        # Agent._actor_dist really are the nodes the whole gradient path flows through. The
        # reference side of the comparison is built EXACTLY the way v1 builds u_cred
        # (Beta(m*c, (1-m)*c) with m a leaf), so this also pins g_ch and u_cred to the same
        # parameterization -- the detail the header records as load-bearing for the SIGN.
        with torch.enable_grad():
            _g = torch.Generator(device="cpu").manual_seed(0)
            _a0 = (1.0 + torch.rand(64, 6, generator=_g, dtype=torch.float64)).mul(3.0)
            _b0 = (1.0 + torch.rand(64, 6, generator=_g, dtype=torch.float64)).mul(3.0)
            _z0 = torch.rand(64, 6, generator=_g, dtype=torch.float64).clamp(0.05, 0.95)
            _c0 = (_a0 + _b0)
            _a = _a0.clone().requires_grad_(True)
            _b = _b0.clone().requires_grad_(True)
            _f = Beta(_a, _b, validate_args=False).log_prob(_z0).sum()
            _ga, _gb = torch.autograd.grad(_f, (_a, _b))
            _chain = _c0 * (_ga - _gb)
            _m = (_a0 / _c0).clone().requires_grad_(True)
            _f2 = Beta(_m * _c0, (1.0 - _m) * _c0, validate_args=False).log_prob(_z0).sum()
            (_direct,) = torch.autograd.grad(_f2, _m)
            _err = (_chain - _direct).abs().max().item() / _direct.abs().max().item()
        assert _err < 1e-10, f"Beta mean-space chain rule broken (rel err {_err:.3e})"
        del _g, _a0, _b0, _z0, _c0, _a, _b, _f, _ga, _gb, _chain, _m, _f2, _direct
    jac_model = None
    jac_optimizer = None
    if use_jac:
        jac_obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        jac_act_dim = int(np.prod(envs.single_action_space.shape))
        with torch.random.fork_rng(devices=[device]):
            jac_model = TransitionHead(jac_obs_dim, jac_act_dim, args.jac_hidden).to(device)
        jac_optimizer = optim.Adam(jac_model.parameters(), lr=args.learning_rate, eps=1e-5)

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    support_min, support_max = value_support_bounds(args)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    support = hl_support.support                       # Dreamer3 raw bucket centers
    bin_width = hl_support.bin_width
    raw_support = support

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    sigma_floor = args.sigma_floor_bins * bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # Ent-PPO: behavior-policy head parameters (Beta: concentration1/concentration0;
    # Gaussian: loc/scale) for the exact proximal KL.
    dist_p1 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    dist_p2 = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # v3: adaptive proximal coefficient (state, not a hyperparameter after init) and the
    # circuit-breaker counter.
    kl_beta_state = args.kl_beta
    grad_split = float("nan")
    aux_grad_ratio = float("nan")
    # The budget clip_grad_norm_ actually enforces on the actor group, so "did the clip
    # bind this minibatch" is a well-posed question in either clipping mode.
    actor_clip_budget = args.actor_grad_clip if args.separate_grad_clip else args.max_grad_norm
    kl_breaker_trips = 0
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, dist = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            p1, p2 = agent.dist_params(dist)
            dist_p1[step] = p1
            dist_p2[step] = p2

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
            )[:, 0]
            next_transition_values = value_logits_to_scalar(next_transition_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # ENT-PPO SOFT MDP: r̃_t = r_t + α·(−log π_old(a_t|s_t)). The pathwise entropy
            # bonus lives INSIDE the reward, so (a) GAE propagates it => an action is credited
            # for the entropy of the states it leads to, and (b) the critic regresses the SOFT
            # return, i.e. it learns the soft value Ṽ the derivation assumes.
            # UNITS: logprobs is the Beta density in the NATIVE z-space (the unit cube), i.e.
            # log_det_fn=0 for the affine z→action map, so this is action-space surprisal plus
            # MINUS the constant |A|·log 2 (a = low + (high−low)·z with high−low = 2, so
            # p_a(a) = p_z(z)/2^|A|). A constant per-step reward shifts every return by
            # c/(1−γ) and cancels exactly in delta_t once the critic fits, so the choice of
            # convention has NO gradient effect; only debug/entropy_bonus_mean reads |A|·log 2
            # LOWER than an action-space entropy would (on HalfCheetah 6·ln2 = 4.16 nats:
            # a logged −7.24 is an action-space differential entropy of −3.08). The KL term is reparameterization-
            # invariant, so the α·H(π_new) identity holds in whichever convention is used.
            # SUPPORT: the bonus is α·(a few nats) against ~6 reward/step, and the ±20k symlog
            # support has ~22x headroom over HalfCheetah's discounted return (MEASURED:
            # debug/returns_absmax peaks at 918 with the bonus vs 943 without, i.e. the bonus
            # barely moves return MAGNITUDE -- its effect is on ordering/variance), so the target
            # still fits (unlike the auto_entropy softboot path, whose auto-tuned alpha
            # overflowed it -- watch debug/target_edge_mass).
            soft_rewards = rewards + args.ent_alpha * (-logprobs)
            # SOFT GAE: critic-consistent soft advantage + soft return.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = soft_rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # Effect-size probe, exact (GAE is linear in the reward sequence at fixed V):
            # A_soft = A_reward + E with E = GAE_λ(α·(−log π_old)) and no value terms.
            # std(E)/std(A_soft) is the entropy share of the policy signal; ≫1 => alpha too big.
            ent_adv = torch.zeros_like(rewards)
            lastentlam = 0
            for t in reversed(range(args.num_steps)):
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                ent_adv[t] = lastentlam = (
                    args.ent_alpha * (-logprobs[t])
                    + args.gamma * args.gae_lambda * lambda_nonterminal * lastentlam
                )
            ent_adv_share = (ent_adv.std() / (advantages.std() + 1e-8)).item()
            ent_ret_shift = ent_adv.mean().item()
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = (
                        rewards[t]
                        + args.gamma
                        * (next_transition_values[t] + next_value_bonus[t])
                        * bootstrap_nonterminal
                        - values[t]
                    )
                    policy_adv[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                    )
            else:
                policy_adv = advantages
            # Batch-level percentile advantage normalization (scopes "ema" and "batch"). Both compute the
            # whole-rollout P5/P95 once and scale policy_adv by one S; "ema" smooths the percentiles with a
            # global EMA across iterations (v1), "batch" uses the FRESH per-rollout spread (no EMA -- the
            # batch-vs-mb ablation). scope=="minibatch" SKIPS this and scales fresh per-mb in the update loop,
            # leaving policy_adv RAW here. Divide-only; critic target `returns` stays RAW (valnorm=none).
            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch": fresh whole-rollout percentile spread, no EMA
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale
            # v162 critic target: scalar-return HL-Gauss MTP. Horizon 0 regresses
            # returns[t]; horizon h regresses returns[t+h] from the same features.
            # A future target is valid only when no reset boundary lies between
            # the source state and target state, and when it stays inside rollout.
            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=returns.device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=returns.device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            # The full (T,B,MTP,bins) target is large and fixed. Keep it on CPU
            # and move only minibatch labels to CUDA during the value loss.
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())
            # Per-state return std probe from the OLD rollout Z(s_t), decoded to
            # raw return units. The default rankgauss path does not consume this.
            sigma = (value_probs * (raw_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(sigma_floor)
            # CDF-rank u in Dreamer3 bucket order; intervals are uniform in symlog
            # coordinate even though raw bucket centers are exponentially spaced.
            cdf_frac = hl_support.cdf_fraction(returns)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_u = u.reshape(-1)
        b_dist_p1 = dist_p1.reshape((-1,) + envs.single_action_space.shape)
        b_dist_p2 = dist_p2.reshape((-1,) + envs.single_action_space.shape)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        # Ent-PPO scale bookkeeping: the proximal objective is (1/adv_div)·[Â − α·KL], so every
        # divisor applied to the advantage MUST also divide the KL penalty. batch_pre_div covers
        # the whole-rollout percentile scaler already applied to policy_adv above.
        batch_pre_div = (
            ret_perc_scale if (args.ret_percnorm and args.ret_perc_scope in ("ema", "batch")) else 1.0
        )
        batch_norm_div = 1.0
        if args.norm_adv and args.norm_adv_scope == "batch":
            batch_norm_div = b_policy_adv.std() + 1e-8
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / batch_norm_div
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            batch_norm_div = b_returns.std().clamp(min=args.ret_perc_floor)
            b_policy_adv_normed = b_policy_adv / batch_norm_div
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        # ===== THE ANALYTIC DIRECTION (improve="distill", "both" or "rotate") ==========
        # Built ONCE per iteration from quantities the chassis already has -- the rollout
        # Beta's own parameters, the replayed native sample, and the batch-standardized
        # advantage -- plus the learned one-step transition head. No privileged
        # conditioning, no extra network on the actor's trunk, no second policy context.
        teach_kl = float("nan")
        teach_kl_cred = float("nan")
        jac_gate = 0.0
        jac_r2 = float("nan")
        jac_align = float("nan")
        jac_rot = float("nan")
        jac_dir_norm = float("nan")
        jac_clamp_frac = float("nan")
        jac_dose_scale = float("nan")
        jac_losses = []
        distill_kls = []
        jac_aux_kls = []        # per-minibatch coefficient-weighted auxiliary charge, nats
        jac_aux_ref_kls = []    # the SAME minibatches' approx_kl, so the ratio is matched
        b_t_alpha = b_t_beta = None
        b_t_mean = b_sd_roll = None
        # v2 ROTATE state. b_g_hat is the FULL-BATCH unit analytic direction (zero rows where
        # the per-dim validity mask zeroed it), stashed so each minibatch can index it;
        # rot_cos_t is v1's gate-ramped cosine. Both stay at their no-op values (None / 1.0)
        # until the held-out R2 gate opens, which is exactly the iteration-1 behaviour the
        # other arms have.
        b_g_hat = None
        rot_cos_t = 1.0
        rot_acc = {k: [] for k in (
            "cos_check", "norm_check", "delta_ratio", "cred_cos",
            "cos_realized", "norm_ratio", "pgrad_ratio", "adam_cos",
        )}
        if use_jac:
            b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
            # `transition_valids` marks "a next observation exists", which is what GAE
            # needs. A one-step DYNAMICS model needs more: a hard termination is not an
            # ordinary physics step, and GAE itself already multiplies the two. Use the same
            # product so the model, its R2 and the direction all agree on what a transition
            # is. No-op on HalfCheetah-v4, which never terminates; correct on envs that do.
            b_valid = (transition_valids * (1.0 - transition_terminations)).reshape(-1)
            b_vsel = b_valid > 0
            # The teacher is weighted by the SAME advantage the surrogate would have used.
            b_teach_adv = b_policy_adv_normed
            conc = b_dist_p1 + b_dist_p2
            mean_s = b_dist_p1 / conc
            sd_t = (mean_s * (1.0 - mean_s) / (conc + 1.0)).sqrt()

            # ---- (1) CREDIT DIRECTION -------------------------------------------------
            # The direction the chassis's OWN policy gradient moves the Beta's mean:
            #     d/dm [ A_i * log pi(a_i | s_i ; m, c) ]   at the rollout action,
            # at fixed concentration c (a mean-shift teacher can express nothing else),
            # detached. No new machinery: m and c are the stored rollout head parameters,
            # a_i is the replayed native sample, A_i is the chassis's normalized advantage.
            #
            # PARAMETERIZATION, STATED EXPLICITLY BECAUSE THE SIGN IS LOAD-BEARING. The
            # gradient is taken w.r.t. the DISTRIBUTION'S MEAN, not w.r.t. the action
            # argument. For any exponential family those two point in OPPOSITE directions:
            # grad_a log pi(a|s) points from the sampled action back toward the mode, so
            # displacing the mean along +grad_a[A log pi] with A > 0 would move the policy
            # AWAY from the action that scored well -- an anti-teacher. The mean gradient is
            # the object PPO's surrogate actually applies to the policy, and it is the only
            # one a mean shift can deliver. (Verified numerically against a real chassis
            # update; see the header.)
            #
            # MAGNITUDE COMES FROM THE DECLARED KL DOSE, NOT FROM THE GRADIENT NORM -- the
            # same choice the donor makes for its analytic direction, for the same reason
            # (a per-sample gradient norm is a scale the KL budget has no opinion about).
            # The unit direction keeps the SIGN of A_i (move toward good actions, away from
            # bad ones) and A_i then re-enters as the per-sample dose weight.
            m_var = mean_s.detach().clone().requires_grad_(True)
            logp_cred = Beta(
                m_var * conc, (1.0 - m_var) * conc, validate_args=False
            ).log_prob(b_latent_zs).sum()
            (grad_m,) = torch.autograd.grad(logp_cred, m_var)
            with torch.no_grad():
                # EVERYTHING BELOW LIVES IN THE POLICY'S OWN METRIC, u = disp / sd, because
                # (a) Beta KL at fixed concentration is quadratic in disp/sd, not disp, so
                # this is the space in which "same dose" means anything, and (b) a gradient
                # is a COVECTOR: steepest ascent under the metric the KL budget charges for
                # is grad * sd, not grad. Mixing raw gradients would silently weight the
                # low-variance action dims hardest -- the dims the policy is most certain of.
                c_u = grad_m * sd_t
                c_hat = c_u / c_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                u_cred = args.teach_step * b_teach_adv.unsqueeze(-1) * c_hat

                # The closed-form dose meter: the per-dim CLIPPED KL the student's loss
                # actually charges for a displacement u, as a batch mean over samples.
                def _delivered(u):
                    m = (mean_s + u * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    return (
                        beta_kl_per_dim(m * conc, (1.0 - m) * conc, b_dist_p1, b_dist_p2)
                        .clamp_min(0.0)
                        .clamp(max=args.distill_kl_clip)
                        .sum(-1)
                        .mean()
                    )

                teach_kl_cred = float(_delivered(u_cred))

            # ---- (2) ANALYTIC DIRECTION ------------------------------------------------
            # d/da [ gamma * V(s + ds_hat(s, a)) ] through the learned one-step model. V(s)
            # stays the value object (no Q, no twin critics, no target nets), the model is
            # supervised regression on the OBSERVED transitions of the single trajectory,
            # and it is never rolled forward, so compounding error does not arise.
            with torch.no_grad():
                # HELD-OUT by construction: the head has only ever seen earlier rollouts.
                ds_true = b_next_obs - b_obs
                b_act = agent.action_low + (agent.action_high - agent.action_low) * b_latent_zs
                ds_pred = (
                    jac_model(b_obs, b_act) * jac_model.ds_var.sqrt().clamp_min(1e-6)
                    + jac_model.ds_mean
                )
                w = b_valid.unsqueeze(-1)
                n_valid = float(b_valid.sum().item())
                if n_valid >= 2.0:
                    mu = (ds_true * w).sum(0) / n_valid
                    sse = (((ds_true - ds_pred) ** 2) * w).sum(0)
                    sst = (((ds_true - mu) ** 2) * w).sum(0).clamp_min(1e-12)
                    jac_r2 = float((1.0 - sse / sst).mean().item())
                # else: jac_r2 stays NaN. With no usable rows sse and sst are both 0 and the
                # clamp would make R2 read exactly 1.0 -- a *perfect* model on this file's
                # primary safety signal, from zero data. NaN keeps the gate shut.
            # GUARDRAIL, NOT AN OBJECTIVE: an inaccurate model yields a noise direction, so
            # the rotation closes and this arm degenerates to the un-rotated credit teacher.
            jac_gate = float(
                min(max((jac_r2 - args.jac_r2_min) / (args.jac_r2_open - args.jac_r2_min), 0.0), 1.0)
            )
            u_new = u_cred
            if jac_gate > 0.0:
                dirs = []
                for start in range(0, args.batch_size, args.minibatch_size):
                    sl = slice(start, start + args.minibatch_size)
                    z_var = b_latent_zs[sl].detach().clone().requires_grad_(True)
                    act = agent.action_low + (agent.action_high - agent.action_low) * z_var
                    s_next = jac_model.next_obs(b_obs[sl], act)
                    v_next = value_logits_to_scalar(agent.get_value(s_next)[:, 0])
                    (grad_z,) = torch.autograd.grad((args.gamma * v_next).sum(), z_var)
                    dirs.append(grad_z.detach())
                jac_grad = torch.cat(dirs) * b_valid.unsqueeze(-1)
                with torch.no_grad():
                    g_u = jac_grad * sd_t
                    g_norm = g_u.norm(dim=-1, keepdim=True)
                    jac_dir_norm = (
                        float(g_norm[b_vsel].mean().item()) if b_vsel.any() else float("nan")
                    )
                    g_hat = g_u / g_norm.clamp_min(1e-8)
                    # ---- (3) EXACT ROTATION (see rotate_to_cos) ----------------------
                    cos_t = min(max(args.jac_cos, -1.0), 1.0)
                    # ---- v3 ANNEAL: spend the dose early, hand it back before it costs ----
                    # MEASURED CAUSE (see header): dose buys early RATE and pays for it in
                    # CEILING via premature entropy collapse -- entropy at matched steps is
                    # monotone in dose at EVERY step (6M: -6.81 dose 0, -6.75 at 0.0045,
                    # -8.47 at 0.045, -9.94 at 0.100) while EV stays healthy, so it is an
                    # exploration loss and not a critic failure. The return crossover is
                    # measured at ~4M. So fold the rotation to the identity by
                    # jac_anneal_frac of training: keep the +35% @2M, then BE the chassis
                    # for the second half, where the chassis's own entropy schedule is what
                    # sets the asymptote. At jac_anneal_frac=0 this is v2 exactly.
                    if args.jac_anneal_frac > 0.0:
                        prog = (iteration - 1) / max(
                            1e-8, args.jac_anneal_frac * args.num_iterations
                        )
                        cos_t = 1.0 - (1.0 - cos_t) * max(0.0, 1.0 - min(1.0, prog))
                    cos_t = 1.0 - jac_gate * (1.0 - cos_t)  # gate closed => no rotation
                    # v2 ROTATE uses EXACTLY these two objects -- the same unit analytic
                    # direction and the same gate-ramped cosine the teacher arms rotate with
                    # -- and applies them to the chassis's own per-sample mean-space gradient
                    # instead of to a teacher displacement. Nothing below this point in the
                    # teacher construction (the dose bisection, the mean clamp, b_t_alpha)
                    # is used by rotate mode; it stays live only so jac_align / jac_rot /
                    # teach_kl keep naming the same objects across all four arms.
                    b_g_hat = g_hat
                    rot_cos_t = cos_t
                    u_new = rotate_to_cos(u_cred, g_hat, cos_t)
                    # ---- (4) MATCH THE DELIVERED BUDGET, NOT THE DISPLACEMENT NORM ----
                    # Holding ||u|| fixed is necessary but NOT sufficient: the loss charges a
                    # per-dim CLIPPED KL, the credit direction is the one that saturates that
                    # clip hardest, so ANY rotation de-saturates the clip and delivers MORE
                    # divergence. Shrink the rotated displacement by one bisected scalar per
                    # iteration until the delivered clipped KL equals what the un-rotated
                    # displacement would have delivered on this very batch. The arm then
                    # differs from its own jac_cos=1.0 control in DIRECTION ONLY.
                    if _delivered(u_new) > teach_kl_cred:
                        lo, hi = 0.0, 1.0
                        for _ in range(args.jac_dose_iters):
                            mid = 0.5 * (lo + hi)
                            if _delivered(mid * u_new) > teach_kl_cred:
                                hi = mid
                            else:
                                lo = mid
                    else:
                        # Rotation delivered no more than the credit direction: nothing to
                        # shrink. Only reachable when the clip binds on essentially nothing.
                        lo = hi = 1.0
                    jac_dose_scale = 0.5 * (lo + hi)
                    u_new = jac_dose_scale * u_new
                    # THE DECISIVE READOUT. If this cosine is ~1 the pathwise direction is
                    # just the credit direction rediscovered and the mechanism buys nothing;
                    # if it is ~0 it is genuinely orthogonal information. Averaged over VALID
                    # rows only: on an invalid row g_hat is the zero vector and
                    # cosine_similarity returns exactly 0, which would drag jac_align toward
                    # 0 and jac_rot toward 1, i.e. bias this arm's own success criterion in
                    # the flattering direction by the invalid fraction.
                    cos = torch.nn.functional.cosine_similarity
                    jac_align = float(cos(u_cred, g_hat, dim=-1)[b_vsel].mean().item())
                    jac_rot = float(cos(u_cred, u_new, dim=-1)[b_vsel].mean().item())
            with torch.no_grad():
                # ---- (5) THE TEACHER: a mean shift at FIXED concentration --------------
                mean_new = (mean_s + u_new * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                # The one place the constant-dose invariant can break. The clamp only ever
                # SHRINKS the displacement, so it is safe, but it binds more often as the
                # policy sharpens. Logged so the dose-matched ladder stays auditable.
                jac_clamp_frac = float(
                    (
                        ((mean_s + u_new * sd_t) <= SAMPLE_EPS)
                        | ((mean_s + u_new * sd_t) >= 1.0 - SAMPLE_EPS)
                    )[b_vsel].float().mean().item()
                )
                teach_kl = float(_delivered(u_new))
                b_t_alpha = mean_new * conc
                b_t_beta = (1.0 - mean_new) * conc
                # The auxiliary charges a distance between MEANS in the ROLLOUT policy's
                # metric, so it needs the teacher mean and the rollout sd, both detached.
                b_t_mean = mean_new
                b_sd_roll = sd_t

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        epoch_kls = []          # v3: per-epoch mean EXACT KL; drives the β adaptation
        # v3: the reference logs max_importance_weight (PPO_hypergrid.py:329,338) and this
        # port had no equivalent. A minibatch-MEAN parametric KL is blind to the sampled
        # ratio tail (Beta z-replay is clamped only to SAMPLE_EPS, so one near-boundary dim
        # moves logratio by many nats), so the tail needs its own instrument.
        iter_ratio_max = 0.0
        breaker_tripped = False
        mask_frac_acc, mask_n = 0.0, 0
        clip_sat_acc, clip_sat_n = 0.0, 0
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            epoch_kl_sum, epoch_kl_n = 0.0, 0
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits, new_dist = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    iter_ratio_max = max(iter_ratio_max, ratio.max().item())

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                adv_div = batch_pre_div          # running product of every divisor hitting the advantage
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                        adv_div = adv_div * batch_norm_div
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_div = b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                        mb_advantages = mb_advantages / mb_div
                        adv_div = adv_div * mb_div
                    else:
                        mb_div = mb_advantages.std() + 1e-8
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_div
                        adv_div = adv_div * mb_div
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    adv_div = adv_div * mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                if use_jac and args.improve == "distill":
                    # ---- (6) THE STUDENT, AND THE TRUST REGION THAT IS THE POINT --------
                    # The donor's per-dim forward KL from the DETACHED teacher into the
                    # student replaces the clipped surrogate. The chassis's PER-SAMPLE
                    # KL-drift mask stays ON and is applied to it EXACTLY as it is applied
                    # to the surrogate: d_i = (rho_i - 1) - log rho_i is still the sampled
                    # drift of the student from the behaviour policy at the rollout action,
                    # it is still non-negative with mean approx_kl, and the offending
                    # SAMPLES are still zeroed with the mean renormalized by the KEPT count
                    # rather than the minibatch size. The clip's per-sample pessimism is
                    # gone (there is no ratio in the objective any more), so the mask is now
                    # the ONLY per-sample trust region -- which is exactly the mechanism the
                    # chassis contributes to this combination.
                    kl_dims = beta_kl_per_dim(
                        b_t_alpha[mb_inds],
                        b_t_beta[mb_inds],
                        new_dist.concentration1,
                        new_dist.concentration0,
                    ).clamp_min(0.0)
                    distill_per_sample = args.distill_coef * kl_dims.clamp(
                        max=args.distill_kl_clip
                    ).sum(-1)
                    if args.tr_mode == "mask":
                        with torch.no_grad():
                            d_i = (ratio - 1.0) - logratio
                            keep = (d_i <= args.tr_sample_eps).to(distill_per_sample.dtype)
                        n_keep = keep.sum()
                        mask_frac_acc += 1.0 - (n_keep / keep.numel()).item()
                        mask_n += 1
                        pg_loss = (distill_per_sample * keep).sum() / n_keep.clamp_min(1.0)
                    else:
                        pg_loss = distill_per_sample.mean()
                    with torch.no_grad():
                        distill_kls.append(kl_dims.sum(-1).mean().item())
                elif args.pg_clip:
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_per_sample = torch.max(pg_loss1, pg_loss2)
                    if args.tr_mode == "mask":
                        # PER-SAMPLE trust region. d_i is the same non-negative quantity whose
                        # mean is approx_kl, so the budget is directly comparable to target_kl.
                        # Zeroing and renormalizing by the KEPT count (not the minibatch size)
                        # keeps the surviving gradient at full scale -- dividing by the full
                        # count would silently shrink the step in proportion to the mask.
                        with torch.no_grad():
                            d_i = (ratio - 1.0) - logratio
                            keep = (d_i <= args.tr_sample_eps).to(pg_per_sample.dtype)
                        n_keep = keep.sum()
                        mask_frac_acc += 1.0 - (n_keep / keep.numel()).item()
                        mask_n += 1
                        pg_loss = (pg_per_sample * keep).sum() / n_keep.clamp_min(1.0)
                    else:
                        pg_loss = pg_per_sample.mean()
                else:
                    # v3: UNCLIPPED surrogate. The exact KL below is the whole trust region;
                    # every sample keeps informing the update (no clipped zero-gradient plateau).
                    pg_loss = -(mb_advantages * ratio).mean()

                # THE CHASSIS'S OWN OBJECTIVE, kept as a separate handle. `losses/policy_loss`
                # and debug/kl_pg_grad_ratio are two of the 36 scalar tags this file exists to
                # compare arm-to-arm, so they must keep naming the SAME object in every mode;
                # folding the auxiliary into them would silently redefine the comparison at
                # every non-zero dose. The auxiliary's own charge is reported separately.
                pg_loss_chassis = pg_loss

                rot_probe = False
                if use_rot:
                    # ---- (6c) THE ROTATION OF THE CHASSIS'S OWN UPDATE (improve="rotate") -
                    # g_ch: the chassis's per-sample ASCENT direction in Beta-MEAN space, at
                    # FIXED concentration. Reparameterizing alpha = m*c, beta = (1-m)*c gives
                    # d/dm = c*(d/dalpha - d/dbeta) exactly (checked at startup against a
                    # leaf-m Beta built the way v1 builds u_cred), and the leading minus turns
                    # the LOSS gradient into an IMPROVEMENT direction -- the object measured at
                    # cosine 0.9977-0.9988 to u_cred (debug/rot_cred_cos). It is a GRADIENT,
                    # not a realized displacement; see the header for why that distinction
                    # turned out to matter.
                    #
                    # This backward never reaches the trunk (pg_loss -> log_prob -> Dirichlet
                    # -> alpha/beta is three ops), it adds NOTHING to the forward graph, and
                    # it writes to no .grad, so the chassis's own numerics are untouched.
                    #
                    # g_ch inherits the per-sample mask and the 1/n_keep renormalization from
                    # pg_loss: a masked row is EXACTLY zero here, and a kept row carries the
                    # surrogate's own scale. That is why the corrector below must NOT divide
                    # by n_keep a second time.
                    alpha_node, beta_node = new_dist.conc_nodes
                    conc_node = alpha_node + beta_node
                    m_new_rot = alpha_node / conc_node
                    g_a, g_b = torch.autograd.grad(
                        pg_loss_chassis, (alpha_node, beta_node), retain_graph=True
                    )
                    g_ch = -(conc_node.detach() * (g_a - g_b))
                    if args.tr_mode == "mask":
                        keep_col = keep.unsqueeze(-1)
                    else:
                        keep_col = torch.ones_like(g_ch)
                    # THE PROBE CADENCE: the last minibatch of each epoch, the same 10-of-320
                    # cadence log_grad_split and aux_pg_grad_ratio use. Everything measured on
                    # it is diagnostic; the mechanism itself runs on every minibatch.
                    rot_probe = start + args.minibatch_size >= args.batch_size
                    g_rot = g_ch
                    if b_g_hat is not None:
                        # v1's EXACT norm-preserving rotation, v1's gate ramp, v1's
                        # degenerate-row fallback: a zero g_ch row, a validity-zeroed u_an row
                        # and a u_an parallel to g_ch all pass through as g_ch unchanged.
                        g_rot = rotate_to_cos(g_ch, b_g_hat[mb_inds], rot_cos_t)
                    delta = (g_rot - g_ch) * keep_col
                    if rot_cos_t < 1.0:
                        # THE CORRECTOR. Linear in m_new, so d/dm is exactly -delta and the
                        # TOTAL mean-space ascent direction becomes g_ch + delta = g_rot. It is
                        # also identically flat in c (d/dc == 0), so it carries no concentration
                        # pressure and therefore no entropy incentive of its own -- the property
                        # design 1's forward KL lacked.
                        #
                        # THE SIGN IS NEGATIVE, AND IT WAS DETERMINED BY MEASUREMENT:
                        # debug/rot_norm_check reads 1.0000 with this sign and sqrt(5-4cos) =
                        # 1.0954 (at cos 0.95) with the other, because the positive sign
                        # delivers 2*g_ch - g_rot -- a LONGER vector at a cosine
                        # (2-cos)/sqrt(5-4cos) = 0.9585 that would have looked almost right.
                        # The norm check, not the cosine, is what distinguishes them.
                        #
                        # `keep` is applied explicitly above even though g_ch already carries
                        # it, so "a masked sample contributes to neither term" is true by
                        # construction and not by inheritance. The kept-count renormalization
                        # is INHERITED (delta is a gradient of the already-renormalized
                        # surrogate); dividing again would shrink the rotation relative to the
                        # step it is supposed to be turning.
                        pg_loss = pg_loss - (delta * m_new_rot).sum()
                    elif rot_probe:
                        # cos_t == 1.0 (jac_cos 1.0, or the R2 gate shut) requests sin_t = 0,
                        # so rotate_to_cos is the ALGEBRAIC identity -- but only to ~1e-7 in
                        # float32, not to bit-zero. The identity control demands BIT-exactness
                        # against --improve surrogate, so the corrector is not added. Every
                        # other part of the machinery still ran: the head was built, trained
                        # and gated, g_ch was taken, the rotation was applied and delta is
                        # measured below. The control is an additivity test, not a dead branch.
                        pass
                    if rot_probe:
                        with torch.no_grad():
                            # ROWS THE MECHANISM CAN ACT ON. A row whose g_ch is zero (masked,
                            # or sitting on the clip's pessimistic plateau where the surrogate
                            # has no gradient) and a row whose u_an was zeroed by the validity
                            # mask are BOTH pass-throughs by design; averaging them in would
                            # drag every cosine toward 1 and every ratio toward 0, i.e. flatter
                            # this arm's own success criterion.
                            rsel = g_ch.norm(dim=-1) > 1e-12
                            if args.tr_mode == "mask":
                                rsel = rsel & (keep > 0)
                            if b_g_hat is not None:
                                rsel = rsel & (b_g_hat[mb_inds].norm(dim=-1) > 1e-12)
                            if bool(rsel.any()):
                                cs = torch.nn.functional.cosine_similarity
                                n_ch = g_ch.norm(dim=-1)
                                rot_acc["delta_ratio"].append(
                                    float((delta.norm(dim=-1)[rsel] / n_ch[rsel]).mean())
                                )
                                # THE EXACT DELIVERY CHECK. Re-read the mean-space gradient of
                                # the CORRECTED objective off the same retained graph: the
                                # cosine must equal rot_cos_t and the norm ratio must be 1.
                                # These two are the sign test and the norm-preservation test,
                                # unconfounded by Adam, by the clip, or by the KL term.
                                g_a2, g_b2 = torch.autograd.grad(
                                    pg_loss, (alpha_node, beta_node), retain_graph=True
                                )
                                g_tot = -(conc_node.detach() * (g_a2 - g_b2))
                                rot_acc["cos_check"].append(
                                    float(cs(g_tot, g_ch, dim=-1)[rsel].mean())
                                )
                                rot_acc["norm_check"].append(
                                    float((g_tot.norm(dim=-1)[rsel] / n_ch[rsel]).mean())
                                )
                                # THE PREMISE OF THIS WHOLE DESIGN, RE-MEASURED EVERY
                                # ITERATION. u_cred is v1's analytic reading of the credit
                                # direction, the object the donor's rotation was validated on;
                                # g_ch is the chassis's ACTUAL clipped, masked, entropy-shaped
                                # surrogate gradient in the same space. The claim "the donor's
                                # rotation can be applied to the chassis's own update" is
                                # exactly the claim that these two are the same direction, and
                                # this scalar is that claim. It reads 0.998-1.000, reproducing
                                # the 0.9988 the previous agent measured. If it fell, the
                                # rotation would still be exact but it would no longer be the
                                # donor's mechanism.
                                rot_acc["cred_cos"].append(
                                    float(cs(g_ch, u_cred[mb_inds], dim=-1)[rsel].mean())
                                )
                                # Handed to the post-step probe, which measures what the
                                # OPTIMIZER actually did to the Beta mean.
                                rot_m_pre = m_new_rot.detach().clone()
                                rot_g_ch, rot_g_rot, rot_rsel = g_ch, g_rot, rsel
                            else:
                                rot_probe = False

                if use_aux:
                    # ---- (6b) THE ADDITIVE AUXILIARY (improve="both") -------------------
                    # WHY THIS SHAPE, AND WHY NOT A FORWARD KL. The "distill" arm's forward
                    # KL(teacher || student) contains -E_teacher[log pi_student], which pays
                    # the student for SPREADING mass over a teacher the 64-wide trunk cannot
                    # represent per state. That is measured, at every dose and at every
                    # ent_alpha, as an entropy leak upward (header). The auxiliary here is a
                    # pure quadratic in the student's MEAN against a DETACHED rollout sd:
                    #     0.5 * || (m_new - m_teach) / sd_rollout ||^2      (summed over dims)
                    # which is exactly the fixed-concentration Beta KL to second order -- the
                    # same second-order form the dose bisection above is calibrated in -- and
                    # which contains NO log-density-of-the-student term at all. It can move
                    # the mean; it has no gradient that rewards a wider Beta. Concentration
                    # is reachable only through m = a/(a+b), i.e. only as the minimum-norm
                    # way to move the mean, never as an entropy incentive.
                    #
                    # PURE ADDITION. It is added to pg_loss, so it shares the actor's own
                    # decoupled grad-clip budget with the surrogate and the entropy term
                    # rather than getting a private one; and at jac_aux_coef = 0 the added
                    # tensor is exactly 0.0, its backward contributes exact zeros, and the
                    # arm is bit-exact to --improve surrogate (verified, header).
                    m_new = new_dist.concentration1 / (
                        new_dist.concentration1 + new_dist.concentration0
                    )
                    # clamp_min on the DIVISOR, not on a product: sd_rollout is
                    # sqrt(m(1-m)/(c+1)) with m in (0,1), so it is positive in exact
                    # arithmetic, but a dim the policy has made near-deterministic rounds
                    # m to 0 or 1 in float32 and gives 0/0 = NaN (the teacher displacement
                    # u_new*sd vanishes on exactly the same dim). 1e-8 is ~six orders below
                    # the sd this chassis reaches at 8M, so it binds on nothing real.
                    u_gap = (m_new - b_t_mean[mb_inds]) / b_sd_roll[mb_inds].clamp_min(1e-8)
                    aux_per_sample = 0.5 * (u_gap * u_gap).sum(-1)
                    if args.tr_mode == "mask":
                        # THE SAME per-sample mask, the SAME kept-count renormalization, the
                        # SAME `keep` tensor the surrogate above just used: a sample rejected
                        # by the trust region contributes to NEITHER term.
                        aux_loss = (aux_per_sample * keep).sum() / n_keep.clamp_min(1.0)
                    else:
                        aux_loss = aux_per_sample.mean()
                    aux_term = args.jac_aux_coef * aux_loss
                    pg_loss = pg_loss + aux_term
                    with torch.no_grad():
                        jac_aux_kls.append(float(aux_term))
                        jac_aux_ref_kls.append(float(approx_kl))

                # ENT-PPO EXACT PROXIMAL TERM (their policy_loss_fn: ppo_clip − KL, coefficient 1).
                #   E_{a~π}[Q̃] + α·H(π) = E_old[ρ·Ã_soft] − α·KL(π_new‖π_old) + const,
                # so this is not a bolted-on trust region: with the reward's −α·log π_old it is an
                # EXACT estimator of α·H(π_new), and α is the same number in both places. All-action
                # and analytic (no importance weight, no clipping), unlike PPO's ratio heuristic;
                # for Beta it is closed form in the concentrations, and it is invariant to the
                # affine z→action rescale (and to tanh in the Gaussian path). Divided by adv_div so
                # the surrogate and the penalty live in the same units.
                exact_kl = torch.distributions.kl_divergence(
                    new_dist, agent.rebuild_dist(b_dist_p1[mb_inds], b_dist_p2[mb_inds])
                ).sum(1)
                with torch.no_grad():
                    epoch_kl_sum += float(exact_kl.mean())
                    epoch_kl_n += 1
                # v2: faithful piece (paper's α/adv_div; inert at MuJoCo reward scale) PLUS the
                # declared piece kl_beta, which lives in the same normalized units as the
                # surrogate and is what actually forms the trust region.
                kl_coef_faithful = args.ent_alpha * args.kl_coef_scale / adv_div
                kl_coef = kl_coef_faithful + kl_beta_state
                kl_penalty = kl_coef * exact_kl.mean()

                # v162 HL-Gauss MTP value loss: per-horizon CE to scalar-return
                # targets, summed across valid horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(device=value_logits.device, dtype=value_ce.dtype, non_blocking=True)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                entropy_loss = entropy.mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                # v3 INSTRUMENTATION (the risk unique to removing the clip): the KL gradient
                # shares the actor's max-norm budget with the PG, and clip_grad_norm_ rescales
                # their SUM -- so a large adaptive β can silently attenuate the learning signal
                # instead of merely restraining drift. That failure would look like "stable but
                # not learning", which is indistinguishable from a bad hyperparameter unless
                # measured. Measured on ONE minibatch per epoch (10 of 320, ~3% overhead).
                if args.log_grad_split and start + args.minibatch_size >= args.batch_size:
                    pg_g = torch.autograd.grad(pg_loss_chassis, actor_params, retain_graph=True, allow_unused=True)
                    kl_g = torch.autograd.grad(kl_penalty, actor_params, retain_graph=True, allow_unused=True)
                    pg_gn = float(torch.norm(torch.stack([g.norm() for g in pg_g if g is not None])))
                    kl_gn = float(torch.norm(torch.stack([g.norm() for g in kl_g if g is not None])))
                    grad_split = kl_gn / (pg_gn + 1e-12)

                # THE AUXILIARY'S DELIVERED EFFECT, as opposed to its declared dose.
                # debug/jac_aux_kl is very nearly the constant 0.5*teach_step^2*E[A^2] times
                # jac_aux_coef: at the rollout policy the student's mean IS the rollout mean,
                # so the gap the auxiliary charges for is the teacher displacement itself, and
                # the credit direction is unit-norm with a batch-standardized weight. It is a
                # DECLARATION, not a measurement of influence. And because the actor group's
                # clip_grad_norm_ binds in essentially every minibatch of this chassis (the
                # v3 measurement quoted at kl_grad_group), the summed actor gradient is
                # renormalized to actor_clip_budget: adding a term to pg_loss therefore TURNS
                # the actor step rather than lengthening it. What the auxiliary actually costs
                # is the fraction of that fixed-length step it takes over, which is this
                # ratio. Same one-minibatch-per-epoch cadence as log_grad_split above.
                if use_aux and start + args.minibatch_size >= args.batch_size:
                    aux_g = torch.autograd.grad(
                        aux_term, actor_params, retain_graph=True, allow_unused=True
                    )
                    ch_g = torch.autograd.grad(
                        pg_loss_chassis, actor_params, retain_graph=True, allow_unused=True
                    )
                    aux_gn = float(torch.norm(torch.stack([g.norm() for g in aux_g if g is not None])))
                    ch_gn = float(torch.norm(torch.stack([g.norm() for g in ch_g if g is not None])))
                    aux_grad_ratio = aux_gn / (ch_gn + 1e-12)

                if rot_probe:
                    # THE MATCHED-ARM A/B, SET UP HERE AND FINISHED AFTER THE STEP. The
                    # requested "realized displacement over the surrogate arm's at matched
                    # step" is only well-posed if BOTH arms are evaluated at the SAME
                    # parameters, on the SAME minibatch, with the SAME Adam state -- which no
                    # pair of separate runs can offer past the first corrected update. So both
                    # actor-objective gradients are taken here, off the same retained graph
                    # (autograd.grad writes to no .grad and mutates no parameter, so the real
                    # backward below is untouched), and the two mean displacements they induce
                    # are read by finite difference after the step, from the restored pre-step
                    # parameters.
                    #
                    # `obj_id` is the actor objective the SURROGATE arm would have had on this
                    # exact minibatch: pg_loss_chassis + kl_penalty - entropy bonus, i.e. the
                    # whole chassis actor objective with the corrector removed and nothing
                    # else changed.
                    def _actor_grad(obj):
                        gs = torch.autograd.grad(
                            obj, actor_params, retain_graph=True, allow_unused=True
                        )
                        gs = [
                            torch.zeros_like(p) if g is None else g
                            for p, g in zip(actor_params, gs)
                        ]
                        return gs, float(torch.norm(torch.stack([g.norm() for g in gs])))

                    rot_ent_term = ent_coef_eff * entropy_loss
                    rot_g_id, rot_n_id = _actor_grad(
                        pg_loss_chassis + kl_penalty - rot_ent_term
                    )
                    rot_g_rt, rot_n_rt = _actor_grad(pg_loss + kl_penalty - rot_ent_term)
                    rot_acc["pgrad_ratio"].append(rot_n_rt / (rot_n_id + 1e-12))
                    rot_theta_pre = [p.detach().clone() for p in actor_params]

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    if args.kl_grad_group:
                        # Objective group (PG + entropy bonus) clipped to its own budget,
                        # then the β·KL restraint clipped SEPARATELY and added without
                        # renormalizing the sum -- so raising β genuinely shortens/steers
                        # the step instead of merely rotating a fixed-length one.
                        (pg_loss - ent_coef_eff * entropy_loss).backward(retain_graph=True)
                        actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                        pg_grads = [(p, p.grad.detach().clone()) for p in actor_params if p.grad is not None]
                        optimizer.zero_grad(set_to_none=True)
                        kl_penalty.backward()
                        nn.utils.clip_grad_norm_(actor_params, args.kl_grad_clip)
                        for p, g in pg_grads:
                            p.grad = g if p.grad is None else p.grad + g
                    else:
                        (pg_loss + kl_penalty - ent_coef_eff * entropy_loss).backward()
                        actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss + kl_penalty - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if rot_probe:
                    # ---- THE REALIZED ROTATION ------------------------------------------
                    # Three readings, in increasing order of how well posed they are.
                    #
                    # (a) THE LITERAL READING the task asks for: the per-row cosine between
                    #     the change in the Beta mean produced by ONE ACTUAL OPTIMIZER STEP
                    #     and the pre-rotation g_ch. MEASURED, AND IT HAS NO SIGNAL:
                    #     debug/rot_adam_cos reads 0.00-0.08 for the ROTATED arm AND 0.02-0.08
                    #     for the un-rotated one (jac_cos 1.0, corrector provably absent). It
                    #     is not a lost rotation: a full Adam step on this chassis moves each
                    #     sample's mean mostly through PARAMETERS SHARED WITH EVERY OTHER
                    #     SAMPLE and with the distributional critic (share_backbone=True, and
                    #     the critic's clipped gradient is added to the same trunk with an
                    #     equal 0.25 budget), so the per-row diagonal response is a few percent
                    #     of the per-row displacement. It is logged because it is the literal
                    #     question, and because reporting an instrument's noise floor is the
                    #     only way the reading below can be trusted.
                    #
                    #     NOTE ON THE 0.9988 THE PREVIOUS AGENT REPORTED: this file reproduces
                    #     that number exactly -- as debug/rot_cred_cos, the cosine between the
                    #     chassis's mean-space GRADIENT and u_cred (measured 0.9985-0.9998).
                    #     It is a gradient-space cosine, not a realized-displacement one. The
                    #     premise of this design therefore holds; only the naive reading of
                    #     "realized" does not.
                    #
                    # (b) THE WELL-POSED REALIZED READING, and the one the design rests on: a
                    #     MATCHED-ARM A/B. Take the two actor objectives that differ ONLY by
                    #     the corrector, clip each to the actor's own budget exactly as the
                    #     real step does, and read the mean displacement each induces by finite
                    #     difference FROM THE SAME pre-step parameters. Then
                    #       debug/rot_cos_realized = cos(Delta m_rotated, Delta m_surrogate)
                    #       debug/rot_norm_ratio   = ||Delta m_rotated|| / ||Delta m_surrogate||
                    #     is literally "the realized displacement against the surrogate arm's
                    #     at a matched step", with the match exact rather than approximate. The
                    #     same h is used for both arms so the ratio carries any real norm
                    #     difference instead of hiding it in a per-arm rescale.
                    #
                    # (c) debug/rot_pgrad_ratio (taken before the step): the same comparison in
                    #     PARAMETER space, on the UNCLIPPED actor gradient. This is the reading
                    #     that would expose the mechanism secretly lengthening the step, since
                    #     the clip afterwards would hide it.
                    with torch.no_grad():
                        _, _, _, _, _, post_dist = agent.get_action_and_value(
                            b_obs[mb_inds], b_latent_zs[mb_inds]
                        )
                        a_post, b_post = post_dist.conc_nodes
                        cs = torch.nn.functional.cosine_similarity
                        rot_acc["adam_cos"].append(
                            float(
                                cs(
                                    a_post / (a_post + b_post) - rot_m_pre,
                                    rot_g_ch,
                                    dim=-1,
                                )[rot_rsel].mean()
                            )
                        )
                        rot_theta_post = [p.detach().clone() for p in actor_params]
                        # ONE h for both arms, sized so the parameter perturbation has norm
                        # 1e-4 on the surrogate arm: large enough that the mean moves ~1e-3 in
                        # float32 (four orders above eps at m ~ 0.5), small enough that the
                        # second-order term is ~1e-8.
                        s_id = min(1.0, actor_clip_budget / (rot_n_id + 1e-12))
                        s_rt = min(1.0, actor_clip_budget / (rot_n_rt + 1e-12))
                        h_fd = 1e-4 / max(s_id * rot_n_id, 1e-12)

                        def _fd_disp(gs, scale):
                            for p, v in zip(actor_params, rot_theta_pre):
                                p.copy_(v)
                            for p, g in zip(actor_params, gs):
                                p.sub_((h_fd * scale) * g)
                            _, _, _, _, _, d_fd = agent.get_action_and_value(
                                b_obs[mb_inds], b_latent_zs[mb_inds]
                            )
                            a_fd, b_fd = d_fd.conc_nodes
                            return (a_fd / (a_fd + b_fd) - rot_m_pre) / h_fd

                        d_id = _fd_disp(rot_g_id, s_id)
                        d_rt = _fd_disp(rot_g_rt, s_rt)
                        # EXACT restore: p.copy_(clone) is bit-identical, so the probe cannot
                        # perturb the trajectory. The bit-exact identity control verifies it,
                        # because this probe runs there too.
                        for p, v in zip(actor_params, rot_theta_post):
                            p.copy_(v)
                        sel = rot_rsel
                        f_rt, f_id = d_rt[sel].flatten(), d_id[sel].flatten()
                        rot_acc["cos_realized"].append(
                            float((f_rt @ f_id) / (f_rt.norm() * f_id.norm() + 1e-30))
                        )
                        rot_acc["norm_ratio"].append(
                            float(f_rt.norm() / (f_id.norm() + 1e-30))
                        )

                if use_aux or use_rot:
                    # Does the actor's clip actually bind? If it does, pg_loss + aux_term
                    # cannot lengthen the step -- it can only turn it -- and the auxiliary's
                    # cost is measured by aux_pg_grad_ratio above, not by its nats.
                    clip_sat_acc += float(float(actor_gn) > actor_clip_budget)
                    clip_sat_n += 1

                if use_jac:
                    # Plain supervised regression on the OBSERVED transition. Separate
                    # optimizer and separate grad clip: sharing either would let the model's
                    # gradient norm change what the actor's decoupled clips do, which is a
                    # confound, and would break the improve="surrogate" control.
                    ds_target = (
                        b_next_obs[mb_inds] - b_obs[mb_inds] - jac_model.ds_mean
                    ) / jac_model.ds_var.sqrt().clamp_min(1e-6)
                    act_mb = (
                        agent.action_low
                        + (agent.action_high - agent.action_low) * b_latent_zs[mb_inds]
                    )
                    vmask = b_valid[mb_inds].unsqueeze(-1)
                    jac_loss = (
                        ((jac_model(b_obs[mb_inds], act_mb) - ds_target) ** 2) * vmask
                    ).sum() / (vmask.sum().clamp_min(1.0) * b_obs.shape[1])
                    jac_optimizer.zero_grad(set_to_none=True)
                    (args.jac_coef * jac_loss).backward()
                    nn.utils.clip_grad_norm_(jac_model.parameters(), args.max_grad_norm)
                    jac_optimizer.step()
                    jac_losses.append(jac_loss.item())

                # Under "mask" the per-sample bound IS the trust region, so this is a true
                # emergency stop at 5x. Under "abort" it is v2's leash at target_kl exactly.
                # Checked per MINIBATCH: an epoch-level check lets 32 further updates land
                # first, which is how approx_kl reached 7.9 in v3's --no-pg-clip arm while
                # the mean exact KL still read 0.034.
                kl_limit = args.target_kl_breaker if args.tr_mode == "mask" else args.target_kl
                if kl_limit is not None and approx_kl.item() > kl_limit:
                    breaker_tripped = True
                    break

            epochs_completed += 1
            if epoch_kl_n:
                epoch_kls.append(epoch_kl_sum / epoch_kl_n)
            if breaker_tripped:
                kl_breaker_trips += 1
                break

        # v3: PPO-penalty adaptation on the EXACT KL of the final epoch -- the drift actually
        # realized under the current β. Multiplicative, so β can traverse orders of magnitude
        # in a few iterations; clamped so a transient cannot strand it.
        kl_beta_used = kl_beta_state
        # max(), not [-1]: the executed-epoch count varies with the breaker, so the last
        # epoch's mean understates drift exactly on the iterations that ran away.
        # kl_beta_state > 0 guard: 0 is a fixed point of the multiplicative rule only in the
        # increase branch, so without it `--kl-beta 0` (the soft-MDP-only control) inverts --
        # low drift would RAISE β off zero to kl_beta_min.
        if args.kl_adapt and epoch_kls and kl_beta_state > 0.0:
            realized_kl = max(epoch_kls)
            if realized_kl < args.kl_target / 1.5:
                kl_beta_state = max(args.kl_beta_min, kl_beta_state / 2.0)
            elif realized_kl > args.kl_target * 1.5:
                kl_beta_state = min(args.kl_beta_max, kl_beta_state * 2.0)

        if use_jac:
            # Refresh AFTER the update so next iteration's R2 and direction are computed
            # from a consistent (weights, stats) pair and the R2 stays genuinely held-out.
            with torch.no_grad():
                # ds.var(0) is the UNBIASED estimator, so one row yields NaN and would
                # silently poison next iteration's standardization, R2 and direction.
                if int(b_vsel.sum().item()) >= 2:
                    jac_model.update_stats(
                        b_next_obs[b_vsel] - b_obs[b_vsel], 0.0 if iteration == 1 else 0.99
                    )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss_chassis.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        # Ent-PPO: exact all-action KL(π_new‖π_old) at the last minibatch (the penalized
        # quantity) next to the sampled-ratio approx_kl that drives the early-stop leash.
        writer.add_scalar("losses/exact_kl", exact_kl.mean().item(), global_step)
        writer.add_scalar("debug/kl_coef", float(kl_coef), global_step)
        writer.add_scalar("debug/kl_coef_faithful", float(kl_coef_faithful), global_step)
        writer.add_scalar("debug/kl_beta", kl_beta_used, global_step)
        writer.add_scalar("debug/ratio_max", iter_ratio_max, global_step)
        writer.add_scalar("debug/kl_breaker_tripped", float(breaker_tripped), global_step)
        # NEVER measured before this file: how much of the 10-epoch budget the trust region
        # actually spends. v1/v2 logged no epoch tag at all, so the cost of the leash was
        # invisible for the whole line.
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("debug/tr_mask_frac", mask_frac_acc / max(mask_n, 1), global_step)
        writer.add_scalar("debug/kl_pg_grad_ratio", grad_split, global_step)
        writer.add_scalar("debug/kl_breaker_trips", kl_breaker_trips, global_step)
        writer.add_scalar("debug/realized_kl_max_epoch", max(epoch_kls) if epoch_kls else 0.0, global_step)
        writer.add_scalar("debug/kl_penalty", kl_penalty.detach().item(), global_step)
        writer.add_scalar("debug/entropy_bonus_mean", (-logprobs).mean().item(), global_step)
        writer.add_scalar("debug/ent_adv_share", ent_adv_share, global_step)
        writer.add_scalar("debug/soft_return_shift", ent_ret_shift, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        if use_jac:
            if args.improve == "distill":
                writer.add_scalar(
                    "losses/distill_kl",
                    float(np.mean(distill_kls)) if distill_kls else float("nan"),
                    global_step,
                )
            writer.add_scalar(
                "losses/jac_model",
                float(np.mean(jac_losses)) if jac_losses else float("nan"),
                global_step,
            )
            # The teacher's dose, MEASURED: the per-dim clipped KL its displacement actually
            # delivers against the rollout policy, and the same for the un-rotated credit
            # displacement it is dose-matched to.
            writer.add_scalar("debug/teach_kl", teach_kl, global_step)
            writer.add_scalar("debug/teach_kl_cred", teach_kl_cred, global_step)
            writer.add_scalar("debug/jac_r2", jac_r2, global_step)
            writer.add_scalar("debug/jac_gate", jac_gate, global_step)
            writer.add_scalar("debug/jac_dir_norm", jac_dir_norm, global_step)
            # THE decisive number: ~1 means the analytic direction rediscovered the credit
            # direction and buys nothing; ~0 means it is orthogonal information.
            writer.add_scalar("debug/jac_align", jac_align, global_step)
            writer.add_scalar("debug/jac_rot", jac_rot, global_step)
            writer.add_scalar("debug/jac_clamp_frac", jac_clamp_frac, global_step)
            writer.add_scalar("debug/jac_dose_scale", jac_dose_scale, global_step)
            if use_aux:
                # THE AUXILIARY'S DECLARED DOSE. jac_aux_kl is the coefficient-weighted
                # auxiliary penalty -- nats, summed over action dims, averaged over KEPT
                # samples and over every minibatch that ran -- and jac_aux_frac divides it by
                # the mean approx_kl of the SAME minibatches. READ IT AS A DECLARATION, NOT AS
                # AN EFFECT: at the rollout policy the student's mean IS the rollout mean, so
                # the charged gap is the teacher displacement itself, whose size is fixed by
                # teach_step (0.5*teach_step^2*E[A^2] = 0.03125 with E[A^2] = 1 by batch
                # standardization). Measured: jac_aux_kl / jac_aux_coef = 0.0352 +-1.1% over a
                # 30x dose span, i.e. the student does not measurably close the gap at any
                # dose, and the +13% over 0.03125 says the chassis step ends up slightly
                # FURTHER from the teacher than the rollout policy was.
                aux_kl = float(np.mean(jac_aux_kls)) if jac_aux_kls else float("nan")
                aux_ref = float(np.mean(jac_aux_ref_kls)) if jac_aux_ref_kls else float("nan")
                writer.add_scalar("debug/jac_aux_kl", aux_kl, global_step)
                writer.add_scalar("debug/jac_aux_frac", aux_kl / (aux_ref + 1e-12), global_step)
                # THE AUXILIARY'S EFFECT. ||grad_actor(aux)|| / ||grad_actor(chassis pg)|| on
                # the last minibatch of each epoch, and how often the actor's clip bound. If
                # the clip binds at ~1.0 the actor step has a FIXED LENGTH and this ratio is
                # the share of its DIRECTION the auxiliary took over from the surrogate --
                # which is the honest statement of what this mode does.
                writer.add_scalar("debug/aux_pg_grad_ratio", aux_grad_ratio, global_step)
                writer.add_scalar(
                    "debug/actor_clip_sat", clip_sat_acc / max(clip_sat_n, 1), global_step
                )
            if use_rot:
                # THE ROTATION, MEASURED. Three exact/algebraic readings, three realized ones.
                #
                # EXACT, in the mean space the mechanism acts in, read off the retained graph
                # and therefore unconfounded by Adam, by the clip or by the KL term:
                #  rot_cos_check   cos(total mean-space gradient after the corrector, g_ch),
                #                  per row over kept non-degenerate rows. Must EQUAL jac_cos
                #                  whenever the R2 gate is open.
                #  rot_norm_check  the same pair's norm ratio. THE defining invariant of this
                #                  design, and the sign test: 1.0000 with the correct
                #                  (negative) corrector sign, sqrt(5-4cos) = 1.0954 at cos 0.95
                #                  with the wrong one -- whose cosine would read a
                #                  deceptively-plausible 0.9585. This scalar is why the entropy
                #                  gate and the per-sample mask cannot see the mechanism as
                #                  extra dose.
                #  rot_delta_ratio ||delta||/||g_ch|| = sqrt(2(1-cos)) for an exact
                #                  norm-preserving rotation (0.3162 at cos 0.95). THE DELIVERED
                #                  mechanism strength, directly comparable to design 2's
                #                  measured delivered influence of 0.0047.
                #  rot_cred_cos    cos(g_ch, u_cred): the PREMISE of the whole design -- that
                #                  the chassis's own surrogate gradient in mean space IS the
                #                  credit direction the donor's rotation was validated on.
                #                  Reproduces the 0.9988 the previous agent measured.
                #
                # REALIZED, as a MATCHED-ARM A/B on the probe minibatch (both arms at the same
                # parameters, same minibatch, same Adam state -- see the probe for why no pair
                # of separate runs can supply that past the first corrected update):
                #  rot_cos_realized  cos(Delta m_rotated, Delta m_surrogate). The realized turn
                #                  of the actual clipped actor step. MEASURED 0.839 / 0.465 /
                #                  0.261 at jac_cos 0.99 / 0.95 / 0.85, i.e. 32.9 / 62.3 / 74.9
                #                  degrees against 8.1 / 18.2 / 31.8 requested -- a 2.4-4.1x
                #                  ANGULAR AMPLIFICATION, and the proximate cause of this
                #                  design's failure (header finding 9).
                #  rot_norm_ratio  ||Delta m_rotated|| / ||Delta m_surrogate||, the requested
                #                  realized norm-preservation reading. ~1.
                #  rot_pgrad_ratio the same comparison in PARAMETER space on the UNCLIPPED
                #                  actor gradient -- the reading that would expose the
                #                  mechanism lengthening the step before the clip hides it.
                #  rot_adam_cos    per-row cos(Delta m from ONE FULL ADAM STEP, g_ch). Logged
                #                  because it is the literal question, and reported as HAVING
                #                  NO SIGNAL: it reads ~0.0-0.08 for the rotated arm AND ~0.02
                #                  -0.08 for the un-rotated one, because a full step moves each
                #                  sample's mean mostly through parameters shared with every
                #                  other sample and with the distributional critic on the same
                #                  trunk. Read rot_cos_realized instead.
                for tag, key in (
                    ("debug/rot_cos_check", "cos_check"),
                    ("debug/rot_norm_check", "norm_check"),
                    ("debug/rot_delta_ratio", "delta_ratio"),
                    ("debug/rot_cred_cos", "cred_cos"),
                    ("debug/rot_cos_realized", "cos_realized"),
                    ("debug/rot_norm_ratio", "norm_ratio"),
                    ("debug/rot_pgrad_ratio", "pgrad_ratio"),
                    ("debug/rot_adam_cos", "adam_cos"),
                ):
                    vals = rot_acc[key]
                    writer.add_scalar(
                        tag, float(np.mean(vals)) if vals else float("nan"), global_step
                    )
                # Same instrument as the "both" arm's: if the actor clip binds in ~100% of
                # minibatches the step has a FIXED LENGTH, which is a second, independent
                # reason the rotation cannot become a dose.
                writer.add_scalar(
                    "debug/actor_clip_sat", clip_sat_acc / max(clip_sat_n, 1), global_step
                )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
