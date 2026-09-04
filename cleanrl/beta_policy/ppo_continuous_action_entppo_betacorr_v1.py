# ENT-PPO BETA-CORR v1 -- temporally coherent exploration that KEEPS the chassis's Beta actor.
# =====================================================================================
# THE MECHANISM. A per-env, per-action-dim AR(1) latent, private to the agent, perturbs the
# chassis's Beta PRE-ACTIVATIONS, and the policy is CONDITIONED ON the latent it actually used:
#
#   z_t = rho * z_{t-1} + sqrt(1 - rho^2) * xi_t,        xi_t ~ N(0, I_d)
#   raw_alpha' = raw_alpha(s_t) + g (*) z_t              [(*) = elementwise, one g per dim]
#   raw_beta'  = raw_beta(s_t)  - g (*) z_t
#   a_t ~ Beta(1 + softplus(raw_alpha'), 1 + softplus(raw_beta'))     [the chassis, verbatim]
#
# z_t -- the latent USED at step t, not z_{t-1} -- is stored in the rollout buffer and replayed in
# the update pass, so pi(a_t | s_t, z_t) is EXACTLY a Beta: log_prob, entropy and the alpha*KL
# proximal term stay the chassis's own closed forms, with no approximation and no new numerics.
# The shift is ANTISYMMETRIC (+s on alpha's pre-activation, -s on beta's), so it moves the Beta's
# MODE; a COMMON shift would modulate the concentration, i.e. inject noise into the per-step
# entropy, which is not the mechanism. Because softplus is convex, alpha+beta can only RISE under
# the perturbation, so the shift can never secretly buy per-step entropy.
# g = corr_gain_max * tanh(raw_gain), one per action dim, init EXACTLY 0 (tanh(0) == 0), learned by
# the ordinary PG + KL gradient; --corr-gain-fixed pins it for a guaranteed full-dose arm.
#
# WHY NOT THE BETA-QUANTILE GAUSSIAN COPULA (a_t = F^-1_Beta(Phi(z_t))), which preserves the
# marginal exactly. Two reasons, one measured and one structural:
#   * torch 2.12 has NO betainc / betaincinv: torch.special exposes nothing beta-related and
#     torch.distributions.Beta.cdf / .icdf both raise NotImplementedError (VERIFIED). scipy is not
#     even installed in this repo's venv. The copula's density needs the regularized incomplete
#     beta DIFFERENTIABLE in (alpha, beta) on device, so it would require a hand-rolled Lentz
#     continued fraction (~1300 elementwise CUDA kernels per log_prob, 320 log_prob calls per
#     champion iteration) plus a hand-rolled Newton inverse for the sampler.
#   * FATAL, independent of numerics: the exact conditional KL of a copula policy has NO closed
#     form. KL(pi_new||pi_old) after pushing forward by T_new lands on
#     Phi^-1(F_new(F_old^-1(Phi(.)))), which is not Gaussian, so the chassis's LOAD-BEARING exact
#     all-action KL (kl_coef ~0.35) would have to be replaced by a hybrid or a sampled estimate.
#     Trading an exact proximal term for an exact marginal is the wrong trade.
#   * And the copula's conditional covariance carries the factor (1 - rho^2), i.e. exactly v1's
#     killer (below). Conditioning on z_t instead of z_{t-1} removes that factor ENTIRELY.
#
# WHY v1 DIED, AND WHY THIS CANNOT DIE THE SAME WAY. Two measured facts about
# ppo_continuous_action_entppo_corrnoise_v1.py (RETIRED, both 8M arms culled):
#   (1) --actor-dist gaussian ALONE, with no correlation at all, runs at 0.55-0.61x the Beta
#       chassis (@3M 3607 vs 6052, @4M 4308 vs 7954). The Beta -> Gaussian family switch costs
#       ~45% on its own, and v1 forced Gaussian whenever correlation was enabled, so it started in
#       a 45% hole. THIS FILE NEVER CHANGES THE ACTION FAMILY: the conditional is a Beta with
#       Beta concentrations, the affine [-1,1] map is untouched, and --no-beta-corr is the chassis
#       BIT-FOR-BIT (gate B).
#   (2) rho pinned at 0.8 was separately destructive (return still -74 at 896k vs the reference's
#       +1102 gain), because conditioning on z_{t-1} contracts the per-step CONDITIONAL entropy by
#       -0.5*sum log(1-rho^2) = 3.07 nats at rho = 0.8: the policy went nearly deterministic per
#       step while drifting in a persistent random direction -- a randomly-biased policy, not
#       exploration. HERE THAT TERM DOES NOT EXIST. H(a_t | s_t, z_t) is a FULL Beta entropy and
#       contains no rho at all, so temporal coherence is FREE: MEASURED losses/entropy at
#       iteration 1 on config C is -0.9370 / -0.9297 / -0.9511 / -0.9529 at rho = 0 / 0.5 / 0.9 /
#       0.95 -- flat in rho to +-0.02 nats. Exploration is ADDITIVE (undiminished per-step Beta
#       noise PLUS a slowly-varying coherent drift of the mode), not traded.
#   The residual per-step entropy cost here is a function of the DOSE g, not of rho: at g = 1.0 the
#   conditional entropy sits 0.52 nats below the control at iteration 1 and recovers to 0.04 nats
#   below by 262k, because a Beta whose mode is displaced inside a FIXED support [0,1] is
#   necessarily narrower than a centred one. That is 6x smaller than v1's rho-0.8 contraction, it
#   is bounded by the dose rather than by rho -> 1, and it is MORE than repaid in the marginal:
#   debug/corr_cond_minus_marg_gap = 0.42 nats/step at g = 1.0, so the z_t-marginal entropy runs
#   0.375 nats ABOVE the control's. Stated plainly rather than claimed to be zero.
#
# DELIBERATE DEVIATIONS FROM THE ORIGINAL SPEC, all forced by the redesign, all measured:
#   * "the Beta MARGINAL is preserved EXACTLY" -> the CONDITIONAL pi(a|s,z_t) is exactly Beta (gate
#     C(i): KS p 0.235..0.978 against the analytic shifted Beta on 1e6 samples). The z_t-MARGINAL
#     is a Beta MIXTURE, not a Beta, and gate C measures the difference instead of hiding it.
#     The conditional is what PPO consumes -- every log_prob, ratio, entropy and KL in this file is
#     a conditional object -- but the claim is weaker than the spec's and is not overclaimed.
#   * "rho is one LEARNED parameter per dim" -> rho is a HYPERPARAMETER (--corr-rho, scalar).
#     pi(a_t|s_t,z_t) contains no rho because z_t is conditioned on, so d(every loss)/d(rho) is
#     structurally identically 0: there is no tensor to differentiate. Making rho learnable
#     requires DECLARING z_t as part of the action, which puts log N(z_t; rho z_{t-1}, 1-rho^2)
#     back into logprobs and reinstates exactly the -0.5 sum log(1-rho^2) contraction that killed
#     v1 (at rho 0.9 that is -4.98 nats, i.e. -1.49 reward/step against ~6 task reward/step
#     through the soft-MDP bonus alpha*(-log pi_old), a strong artificial force driving rho to 0).
#     REJECTED on those grounds. The learned dose is g, and gate D is re-pointed at g.
#   * "--corr-rho-fixed" -> --corr-rho (rho is always fixed) and --corr-gain-fixed (pins g:
#     raw_gain frozen, requires_grad False, excluded from Adam AND from the actor clip group).
#   * "sweep rho 0.2/0.4/0.6, rho must be SMALL" -> swept 0 / 0.5 / 0.9 / 0.95; ALL of them are
#     safe (gate G), because the destructive mechanism is gone. Default rho = 0.9.
#
# GATING. --beta-corr (default ON) constructs raw_gain and the latent; --no-beta-corr constructs
# NOTHING and restores args.actor_dist verbatim, so it is the chassis's default Beta champion
# (10362 @8M) bit-for-bit. Three controls, and they are different experiments:
#   * --no-beta-corr                              == the chassis. Nothing new exists.
#   * --beta-corr --corr-gain-fixed 0             == the mechanism fully LIVE (the latent runs and
#     is autocorrelated at 0.897) but numerically INERT. This is the strong bit-exactness test.
#   * --beta-corr --corr-gain-fixed G --corr-rho 0 == iid parameter noise at the SAME dose and the
#     SAME per-step marginal law, zero temporal coherence. This is the matched control for the
#     AR(1) term itself, and P4 is the falsifier built on it.
# xi and the boundary redraw come from a DEDICATED torch.Generator seeded off args.seed, so they
# cannot perturb the global stream that Beta.sample() draws from: every arm is RNG-MATCHED on the
# action sampler and differs only in the transformation applied.
#
# THE PROXIMAL TERM IS STILL EXACT. exact_kl is torch's closed-form Beta-Beta KL between the new
# and old CONDITIONALS at the SAME recorded z_t, summed over dims: all-action and analytic in a
# (no importance weight, no sampling in the action), and single-sample in (s_t, z_t) -- which is
# precisely the status the chassis already has for s_t, since s_t is likewise a draw from the
# behavior distribution. So the KL is not degraded by the mechanism; it is the same estimator with
# one more conditioning variable, and it gives g its own trust-region anchor (gate D). Likewise
# `entropy` is the EXACT conditional entropy H(a_t|s_t,z_t) in closed form (ent_coef is 0, so it is
# telemetry, but it is the right object).
#
# INHERITED, NOT RE-VERIFIED: the MTP critic horizons, the Dreamer3-bucket HL-Gauss value readout
# and its projection, the per-sample sample-mask trust region (tr_mode=mask, tr_sample_eps=0.1),
# the soft-MDP reward r~ = r + alpha*(-log pi_old), the exact-KL proximal term and kl_beta, batch
# advantage normalization, the epochs_completed leash / target_kl_breaker, dual-backward decoupled
# grad clipping, and EVERY inherited hyperparameter value (max_grad_norm, actor_grad_clip,
# kl_coef, learning_rate ... all untouched). VERIFIED IN THIS FILE: the block below.
#
# PRE-REGISTERED PREDICTIONS (chassis reference 10362 +-92 @8M; matched checkpoints @1M 3041 /
# @2M 5465 / @3M 7281 / @4M 8890; benchmark noise floor ~400; Gaussian-actor control @3M 3607 /
# @4M 4308, which this file does NOT invoke).
#   P1 RETURN. PASS: >= 10762 @8M (chassis + one noise floor) AND >= 5865 @2M.
#      FAIL: <= 9962 @8M, or (<= 5065 @2M AND <= 8490 @4M). In between = INCONCLUSIVE, say so.
#   P2 THE MECHANISM IS ALIVE. PASS: debug/corr_gain_absmean >= 0.20 AND
#      debug/corr_mode_shift_absmean >= 0.010 (>= 2% of the [-1,1] action range) by 2M.
#      FAIL: debug/corr_gain_absmax < 0.05 at 4M -- g never left its init (it is 0.019 at 262k),
#      the perturbation is inert, and any return difference is noise rather than the mechanism.
#   P3 EXPLORATION IS ADDITIVE, NOT TRADED. PASS: debug/corr_cond_minus_marg_gap >= 0.10 nats
#      while losses/entropy stays within 0.30 nats of the --no-beta-corr control at matched steps.
#      FAIL: losses/entropy runs > 0.60 nats BELOW the control's -- then the dose is buying
#      coherence with per-step stochasticity after all, i.e. v1's trade, and the premise is wrong.
#   P4 THE COHERENCE IS WHAT PAYS, NOT THE EXTRA NOISE. PASS: --corr-gain-fixed 1.0 --corr-rho 0.9
#      beats --corr-gain-fixed 1.0 --corr-rho 0.0 by >= 400 @2M. The two arms have the same dose
#      and the same per-step law in distribution and differ ONLY in temporal coherence.
#      FAIL: within 400 @2M -- the AR(1) recursion adds nothing over iid parameter noise and
#      should be dropped in favour of plain parameter noise.
#
# --- VERIFIED IN THIS FILE (measured, not claimed) -----------------------------------
# Every number below was produced by this exact file. NOTHING here is an 8M result: no 8M run was
# performed, and no return claim is made. Short config S = --num-envs 4 --num-steps 256
# --num-minibatches 4 --total-timesteps 20480 --seed 1. Champion config C = --num-envs 16
# --num-steps 2048 --num-minibatches 32 --update-epochs 10 --total-timesteps 262144 --seed 1.
#
# A COMPILE. `python -m py_compile` clean.
#
# B BIT-EXACT CONTROL vs the parent chassis on S. All shared scalar tags, charts/SPS excluded:
#   36 shared tags, 720 points compared, 0 DIFFERING, in BOTH off-paths:
#     parent (default beta) vs --no-beta-corr                    -> 0/720, 0 unique tags
#     parent (default beta) vs --beta-corr --corr-gain-fixed 0    -> 0/720, 21 tags unique to this
#       file (all debug/corr_*; nothing shared is missing or extra)
#   charts/episodic_return (20 points) is among the compared tags, so the TRAJECTORIES are
#   identical, not merely the summary statistics.
#   In the --corr-gain-fixed 0 arm the mechanism is FULLY LIVE -- debug/corr_z_lag1_ac_mean =
#   0.8970 at rho 0.9 -- while debug/corr_gain_absmax, corr_mode_shift_absmean, corr_mode_shift
#   _absmax and corr_shift_lag1_ac_mean are 0.0 in every one of the 20 iterations and the
#   conditional-minus-marginal gap is 1.3e-8. That is why this is EXACT and not merely close:
#   g == 0.0 makes the shift `x + (+-0.0)` and `y - (+-0.0)`, IEEE-exact no-ops; the frozen
#   raw_gain is excluded from the actor clip group (a live zero-grad tensor in the group would
#   still be summed into the clip norm); and the dedicated generator keeps Beta.sample()'s RNG
#   stream untouched.
#
# C DENSITY CORRECTNESS -- the load-bearing gate. g pinned to 1.0, rho 0.9 (rho does not enter the
#   density), ONE fixed state and ONE fixed z_t, N = 1e6 samples drawn through the ACTUAL sampling
#   path (100 chunks x 10000 real get_action_and_value calls; actions re-checked equal to the
#   chassis's affine map of the native sample on every chunk).
#   (i)   THE FAMILY IS UNCHANGED. KS of the sample against the ANALYTIC Beta(shifted alpha,
#         shifted beta), 200k samples: D = 0.00106..0.00231, p = 0.235..0.978 (the 5% critical D
#         is 0.00304) -- accepted on all 6 dims. The SAME samples against the UNSHIFTED Beta:
#         D = 0.068..0.258, p = 0.0 on all 6 dims, so the perturbation is real and large.
#         Moments vs analytic: |mean err| 3.0e-5..3.5e-4 = 0.13..1.51 MC sigma; |var| rel err
#         3.2e-4..3.1e-3.
#   (ii)  NORMALIZATION. E[-log p] = -0.83576 +- 0.00099 (MC SE) against the closed-form Beta
#         entropy -0.83461 -> diff -1.15e-3 = -1.16 MC sigma. scipy's analytic per-dim entropy
#         summed = -0.83461237 vs this file's closed form -0.83461183 (5.4e-7), so the entropy
#         formula itself is right to fp32.
#   (iii) INDEPENDENT IMPLEMENTATION. log_prob vs scipy.stats.beta.logpdf (scipy 1.18.1, float64,
#         a SEPARATE interpreter): MAX ABS DIFF 1.088e-6, mean 1.72e-7, over 200k samples with
#         |log p| up to 6.757 (WORST-CASE |log p| over the full 1e6 = 10.431). NONFINITE: 0 of
#         1e6, and no NaN/Inf anywhere. Independent MC E[-log p] from the scipy pdf = -0.83832426
#         vs torch -0.83832441 (1.5e-7).
#   (iv)  ROLLOUT vs REPLAY: |log_prob(rollout) - log_prob(replay)| = 0.0 EXACTLY (torch.equal
#         True), actions bitwise equal. The PPO ratio replays the same conditional that generated
#         the sample.
#   HONEST, AND THE POINT OF THE DEVIATION: the z_t-MARGINAL is a Beta MIXTURE. Its best-fit Beta
#   is REJECTED by KS (D 0.0037..0.0066, p 7.7e-3 down to 4.6e-8) and its variance is
#   0.0669..0.0673 against the unshifted Beta's 0.0569..0.0571, i.e. +17.6% marginal action
#   variance at g = 1.0. The mechanism therefore does add marginal exploration; it does not
#   preserve the marginal.
#   Pinning is fp-exact where it matters: --corr-gain-fixed 0 gives g = 0.0 EXACTLY
#   (arctanh(0) == 0, tanh(0) == 0); --corr-gain-fixed 1.0 gives 0.99999994 (fp32 atanh/tanh).
#
# D THE GAIN IS ALIVE. rho IS A HYPERPARAMETER.
#   init: raw_gain is a zeros Parameter, g = corr_gain_max*tanh(0) -> max|g| = 0.000e+00 EXACTLY.
#     raw_gain IS in actor_parameters() when learned (it rides the actor's own clip budget) and is
#     excluded when pinned.
#   grad(pg_loss) wrt raw_gain at init, 4096 REAL on-policy states:
#     [-0.0281 +0.0030 -0.0195 -0.0400 +0.0123 +0.0197] -- finite, nonzero, max 4.0e-2.
#     exact_kl at init = -1.3e-10 (new == old, so the KL is at its minimum, as it must be).
#   after 800 real Adam steps (S): g = [-0.0386 +0.0212 -0.0202 +0.0110 +0.0170 -0.0181],
#     absmax 0.0386 (peak 0.0420 mid-run), absmean 0.0210, off 0 by iteration 1 and rising.
#   after 2560 real Adam steps (C): absmax 0.0193, absmean 0.0108, mode displacement 0.0010.
#   HONEST READING: g is unambiguously off 0 and moving, but at 262k steps (3% of the 8M horizon)
#     it is only ~2e-2, i.e. a mode displacement of 0.4% of the action range. This file CANNOT
#     locate the equilibrium; P2 is checked at 2M.
#   THREE-WAY GRADIENT DECOMPOSITION on raw_gain, g = 1.0, 4096 real on-policy states after 20
#     real Adam steps so the policy has drifted to exact_kl = 0.0311 (the champion's own realized
#     drift is 0.023-0.031), harness kl_coef 0.60:
#       per-dim |PG channel|      0.0513 .. 0.0706
#       per-dim |KL entropy half| 0.1251 .. 0.1320      (the -H(new) half of KL = -H + CE)
#       per-dim |KL cross half|   0.1302 .. 0.1474
#       per-dim |NET KL|          0.0051 .. 0.0154
#       ||NET KL|| / ||half|| = 0.0965      ||PG|| / ||NET KL|| = 5.07
#     So the same qualitative result v1 found, with a WEAKER cancellation: the two halves of the
#     same KL cancel to ~10% (v1 measured ~1% for rho), and the net trust-region force is
#     out-weighted 5.1x by the policy gradient -- 8.7x at the run's true kl_coef ~0.35, since both
#     halves scale linearly with kl_coef. NO, the net force is NOT ~1% here; it is ~10%, and it is
#     still not the thing that decides g. The KL is a proximal anchor, not an entropy bonus.
#   rho: pi(a_t|s_t,z_t) contains no rho, so d(loss)/d(rho) == 0 identically -- there is no tensor
#     to differentiate and no gradient to report. See the DEVIATIONS section for why this is the
#     right trade and what the alternative would cost.
#
# E TEMPORAL CORRELATION IS REAL. From REAL training rollouts on C, within-episode pairs only
#   (transition_boundaries[t] == 0), last iteration; the latent's lag-1 AC and the lag-1 AC of the
#   EXECUTED Beta-mean displacement:
#     rho      z_lag1      shift_lag1     per-dim shift_lag1
#     0.00    -0.0005       -0.0002       -0.0059 .. +0.0071
#     0.50    +0.5004       +0.4972       +0.4877 .. +0.5026
#     0.90    +0.9002       +0.8975       +0.8914 .. +0.9005
#     0.95    +0.9502       +0.9478       +0.9458 .. +0.9506
#   The realized action-space coherence equals rho to 3 digits, so the plumbing carries the
#   intended dynamics end to end. Against a 1/sqrt(n_pairs) scale of 0.0055 (n_pairs = 32752) the
#   rho = 0 arm is indistinguishable from white (max |.| = 0.0071 = 1.3 sigma), so the coherence
#   comes from the AR(1) term and not from the parameter noise.
#   --no-beta-corr has no mode shift at all; --corr-gain-fixed 0 has it identically 0.0 while the
#   latent is still autocorrelated at 0.8970.
#   LEARNED arm at 262k: z_lag1 0.9002, shift_lag1 0.8972 -- the dynamics are there, but the DOSE
#   (mode|shift|mean 0.0010) is still tiny, so the demonstration above is the pinned-gain one.
#   STATED PLAINLY, not papered over.
#
# F RESET CORRECTNESS. The harness extracts the SHIPPED latent-advance statement from this file by
#   regex and prints it before executing it, so the lines under test are the lines that run.
#   20000 steps x 16 envs with staggered 40-step boundaries (short limit purely to make
#   boundaries) at rho = 0.95, so a carried latent would be unmistakable. 7984 boundary events,
#   312016 non-boundary steps:
#     non-boundary corr(z_t, z_{t-1}) per dim = 0.9493 .. 0.9507  (== rho, as it must be)
#     BOUNDARY     corr(z_t, z_{t-1}) per dim: max |.| = 0.0163 vs 1/sqrt(7984) = 0.0112
#                  (1.46 sigma) -> independent
#     boundary entries numerically consistent with a STALE carry: 0 / 47904 (must be none)
#     the post-reset latent IS a standard normal: per-dim |mean| <= 0.0188 (MC SE 0.0112),
#       std 0.9886 .. 1.0071, KS p-values vs N(0,1) 0.180 .. 0.834
#     stationarity (the reason for fresh-draw rather than zeros): latent variance 0.9950 at
#       boundary steps vs 0.9948 elsewhere. Zeroing would give 0.0 vs ~1.0 and would delete the
#       first post-reset step's drift entirely.
#   End to end, debug/corr_z_reset_frac = 0.00098 in every arm on C = 32 boundaries per 32768-step
#   rollout, which is exactly HalfCheetah's 1000-step limit (16 x 2048 / 1000 = 32.8).
#   ONE draw per step feeds BOTH branches of the torch.where, so RNG consumption is independent of
#   the done pattern and the run stays reproducible.
#
# G NO NaN/Inf over 262144 steps at champion config C. Six arms: --no-beta-corr (control), learned
#   g, and --corr-gain-fixed 1.0 at rho = 0.0 / 0.5 / 0.9 / 0.95. EVERY scalar of EVERY arm was
#   scanned (960 points over 58 tags per corr arm; 792 over 37 for the control): NONFINITE = NONE
#   anywhere. debug/epochs_completed = 10/10 in EVERY iteration of EVERY arm;
#   debug/kl_breaker_tripped 0 everywhere; debug/target_edge_mass exactly 0 everywhere.
#   SAMPLE-MASK SENSITIVITY (the chassis's known-fragile axis, and how v1 died), mean/max over the
#   8 iterations; actor_grad_norm is PRE-CLIP against actor_grad_clip = 0.25:
#     arm            tr_mask_frac      approx_kl        ratio_max  actor_gn      brk  epochs
#     control        0.0062 / 0.0083   0.0101 / 0.0126    5.55     1.33 / 1.60    0   10/10
#     learned g      0.0058 / 0.0078   0.0098 / 0.0128    5.42     1.41 / 1.78    0   10/10
#     g1.0 rho 0.0   0.0054 / 0.0087   0.0084 / 0.0106    4.35     1.19 / 1.54    0   10/10
#     g1.0 rho 0.5   0.0052 / 0.0073   0.0086 / 0.0102    5.27     1.10 / 1.43    0   10/10
#     g1.0 rho 0.9   0.0064 / 0.0088   0.0083 / 0.0095    5.46     1.23 / 1.80    0   10/10
#     g1.0 rho 0.95  0.0057 / 0.0072   0.0083 / 0.0107    4.91     1.26 / 1.65    0   10/10
#   Ratios to control (means): tr_mask_frac 0.84x .. 1.03x, pre-clip actor_grad_norm 0.82x ..
#   1.06x, approx_kl 0.82x .. 0.98x. LARGEST SAFE rho = 0.95, the largest TESTED: every rho sits
#   INSIDE the control's own band on every fragility metric, and ratio_max is LOWER than the
#   control's in all six arms. This is the axis that killed v1 (control tr_mask 0.0062 vs rho-0.8
#   0.0342, pre-clip actor_grad_norm 5.9 against a 0.25 clip) and it is simply not engaged here,
#   because there is no conditional-entropy contraction to widen the per-sample ratio spread. rho
#   is therefore not the dangerous knob in this design; the dose g is, and g is bounded by
#   corr_gain_max and clipped with the rest of the actor.
#   losses/value_loss first->last: control 24.00->22.90, learned 24.00->22.66, rho0.0 22.36->17.05,
#     rho0.5 22.74->15.44, rho0.9 22.25->14.59, rho0.95 22.43->15.31 -- all finite and falling.
#   losses/explained_variance: control 0->0.851, learned 0->0.857, rho0.0 0->0.706, rho0.5 0->0.735,
#     rho0.9 0->0.685, rho0.95 0->0.653. Finite and sane, BUT the pinned-dose arms are materially
#     WORSE than the control at 262k: a mode that wanders makes the value target harder. Reported
#     as a warning, not explained away; whether it recovers by 1M is untested.
#   losses/entropy first->last: control -0.4334->-0.4715, learned -0.4328->-0.4696, and the four
#     pinned arms all start at -0.93..-0.95 (flat in rho) and RECOVER to -0.62/-0.55/-0.52/-0.54 --
#     the actor de-concentrates to pay for the displaced mode. debug/corr_cond_minus_marg_gap at
#     the last iteration: 0.4927 / 0.4530 / 0.4189 / 0.4528 nats at rho 0 / 0.5 / 0.9 / 0.95, so
#     the z_t-marginal entropy is ~0.375 nats ABOVE the control's while the conditional is 0.044
#     nats below it. debug/corr_mode_shift_absmean 0.079..0.084 of the unit cube (0.16..0.17 of
#     the [-1,1] action range), absmax up to 0.433 (0.87 of the range) -- g = 1.0 is a HEAVY dose.
#
# H THROUGHPUT AND VRAM at champion config C. The GPU was SHARED throughout with an unrelated
#   16.8 GB training job pinning it at 100% utilization, so every SPS number here is a CONTENDED
#   LOWER BOUND; the arms were run STRICTLY SEQUENTIALLY so they are at least comparable to each
#   other.
#   Peak VRAM (nvidia-smi, process-attributed): control 2204 MiB, every corr arm 2210 MiB. The
#     mechanism costs 6 MiB.
#   SPS (mean over the 8 iterations): control 520, learned 509, rho0.0 614, rho0.5 566, rho0.9 497,
#     rho0.95 505. Wall clock for 262144 steps: 480-518 s across all six arms, i.e. the spread
#     between arms (7%) is smaller than the spread the contention imposes.
#   Directly measured cost, one champion-size (B = 1024) minibatch fwd+bwd of the full actor
#     objective (PG + clip + mask + exact KL): beta-corr g=1.0 22.96 ms, beta-corr g=0 23.72 ms,
#     --no-beta-corr 23.35 ms; one rollout forward at B = 16: 10.886 / 10.952 / 10.855 ms. The
#     mechanism is FREE to within measurement noise -- it adds two elementwise adds and one
#     broadcast multiply per forward and no new distribution object.
#   TIME LIMIT for an 8M queue slot: 8e6/262144 = 30.5 x ~500 s = 4.3 h at this contention.
#     Budget >= 6 h.
#
# I The parent chassis was never modified: md5 f6c545f51d6c49901bd739f6f11aabca before and after.
#   Every scratch run directory created for these measurements was deleted.
#
# NOT VERIFIED HERE, AND NOT CLAIMED: any return at 1M/2M/4M/8M; that the mechanism helps at all.
#   At 262k (3% of the horizon, where the chassis itself is only at ~3041 by 1M) the last-16
#   episodic-return means are: control +225, learned g +243, and the pinned g = 1.0 arms +32 /
#   -65 / -85 / -59 at rho 0 / 0.5 / 0.9 / 0.95. The learned arm is at the control (its g is
#   ~0.01, so it IS the control). The FULL DOSE is BEHIND at 262k. That is weak evidence, but it
#   is evidence pointing the wrong way, and it is recorded here rather than omitted -- with the
#   caveat that g = 1.0 is a heavy dose by gate G's own measurement and no dose was tuned (a
#   hyperparameter grid was out of scope), so the pinned arms are an upper-bound stress test of
#   the mechanism, not a recommended configuration. The DEFAULT path learns g from 0.
# ------------------------------------------------------------------------------------
#
# --- inherited chassis header (unchanged below) ---
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
from math import log
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

    # --- BETA-CORR: temporally correlated exploration by perturbing the Beta's own
    # pre-activations with an AR(1) latent that is CONDITIONED ON, not integrated out.
    # See the header for why this and not a Beta-quantile Gaussian copula.
    beta_corr: bool = True
    # rho is a HYPERPARAMETER, not a learned parameter: pi(a|s,z_t) does not contain rho
    # (z_t is conditioned on), so every loss has d/d rho == 0 identically. The learned dose
    # is the gain g. 0.9 => a coherence length 1/(1-rho) = 10 steps = 0.5 s of HalfCheetah,
    # the scale of one gait cycle; chosen by that argument plus gate G's safety table, NOT
    # by any measured return.
    corr_rho: float = 0.9
    # g = corr_gain_max * tanh(raw_gain). The bound stops a runaway gain from railing the
    # Beta concentrations; 2.0 is already far past a full dose (|g z| ~ 2 shifts the Beta
    # mean by ~0.29 of the unit cube, i.e. ~0.6 of the [-1,1] action range).
    corr_gain_max: float = 2.0
    # PIN the gain (raw_gain frozen, requires_grad False, excluded from Adam AND from the
    # actor clip group) for a guaranteed full-dose arm. None => g is learned from 0.
    corr_gain_fixed: Optional[float] = None
    # Telemetry budget for the conditional-vs-marginal log-density gap (= I(a_t; z_t | s_t)).
    # 0 on either disables the probe.
    corr_mi_states: int = 1024
    corr_mi_draws: int = 32

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
        # BETA-CORR: one learned per-dim gain on the AR(1) latent's perturbation of the Beta
        # PRE-ACTIVATIONS. raw_gain init zeros => g = corr_gain_max*tanh(0) == 0.0 EXACTLY,
        # so the shift is `x + (+-0.0)` and the actor is the chassis BITWISE at init.
        self.beta_corr = args.beta_corr
        self.corr_gain_max = args.corr_gain_max
        if self.beta_corr:
            self.raw_gain = nn.Parameter(torch.zeros(act_dim))
            self.corr_gain_fixed = args.corr_gain_fixed
            if self.corr_gain_fixed is not None:
                with torch.no_grad():
                    self.raw_gain.fill_(
                        float(np.arctanh(self.corr_gain_fixed / self.corr_gain_max))
                    )
                self.raw_gain.requires_grad_(False)

    def corr_gain(self):
        # g in (-corr_gain_max, corr_gain_max), EXACTLY 0 at init since tanh(0.0) == 0.0.
        return self.corr_gain_max * torch.tanh(self.raw_gain)

    def _actor_dist(self, actor_feat, corr_z=None, probe=False):
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
        raw_alpha0 = self.actor_alpha_head(actor_feat)
        raw_beta0 = self.actor_beta_head(actor_feat)
        raw_alpha, raw_beta, shift = raw_alpha0, raw_beta0, None
        if self.beta_corr and corr_z is not None:
            # ANTISYMMETRIC pre-activation shift. +s on alpha's pre-activation and -s on
            # beta's moves the Beta's MODE and leaves the concentration alpha+beta nearly
            # fixed; because softplus is convex, alpha+beta can only RISE by O(s^2), so the
            # perturbation can never secretly BUY entropy. A COMMON shift would instead
            # modulate the concentration, i.e. inject noise into the per-step entropy, which
            # is not the mechanism. At g == 0.0 these are `x + (+-0.0)` and `y - (+-0.0)`,
            # which return x and y BITWISE -- that is why --corr-gain-fixed 0 is the chassis.
            shift = self.corr_gain() * corr_z
            raw_alpha = raw_alpha0 + shift
            raw_beta = raw_beta0 - shift
        alpha = 1.0 + F.softplus(raw_alpha)
        beta = 1.0 + F.softplus(raw_beta)
        dist = Beta(alpha, beta)
        if probe and shift is not None:
            # DOSE telemetry: how far the latent moved the Beta mean, in unit-cube units
            # (multiply by 2 for the [-1,1] action range). Rollout-only (probe=False in the
            # update pass), no_grad, two extra softplus calls at B = num_envs.
            with torch.no_grad():
                a0 = 1.0 + F.softplus(raw_alpha0)
                b0 = 1.0 + F.softplus(raw_beta0)
                dist.corr_mode_shift = alpha / (alpha + beta) - a0 / (a0 + b0)
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

    def get_action_and_value(self, x, z=None, corr_z=None, probe=False):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # corr_z is the BETA-CORR conditioning latent z_t -- the latent that was USED at
        # this step. It is replayed from the buffer in the update pass, so the recomputed
        # log_prob / entropy / KL are the SAME exact Beta objects that generated the sample.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat, corr_z=corr_z, probe=probe)
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
            # raw_gain rides the actor's OWN clip budget (it is an actor parameter). A PINNED
            # gain is frozen and must stay OUT of the group: including a zero-grad tensor
            # would not change the norm, but including a LIVE one at g == 0 would -- and that
            # is exactly the arm gate B declares bit-exact.
            if self.beta_corr and self.raw_gain.requires_grad:
                heads = heads + [self.raw_gain]
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def within_episode_lag1_ac(x, boundaries):
    """Per-dim lag-1 autocorrelation of x (T,B,d) over WITHIN-EPISODE pairs only.
    boundaries is (T,B): 1.0 at a step that ENDED an episode, so the pair (t, t+1) is
    valid iff boundaries[t] == 0. Returns a numpy (d,) array; identically 0 for a
    constant-zero signal (the g == 0 arm) rather than NaN."""
    pair_ok = (boundaries[:-1] == 0).unsqueeze(-1).to(x.dtype)
    n = pair_ok.sum().clamp_min(1.0)
    x0, x1 = x[:-1], x[1:]
    m0 = (x0 * pair_ok).sum((0, 1)) / n
    m1 = (x1 * pair_ok).sum((0, 1)) / n
    d0, d1 = x0 - m0, x1 - m1
    v0 = ((d0 * d0) * pair_ok).sum((0, 1)) / n
    v1 = ((d1 * d1) * pair_ok).sum((0, 1)) / n
    cov = ((d0 * d1) * pair_ok).sum((0, 1)) / n
    return (cov / (v0.sqrt() * v1.sqrt() + 1e-12)).cpu().numpy()


def inv_softplus(y):
    """Stable inverse of softplus for y > 0: x = y + log(-expm1(-y)); exact to fp for both
    tiny y (-> log y) and large y (-> y)."""
    return y + torch.log(-torch.expm1(-y))


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
    assert not (args.tr_mode == "mask" and not args.pg_clip), \
        "tr_mode=mask needs the clipped per-sample surrogate (--pg-clip)"
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
    if args.beta_corr:
        assert args.actor_dist == "beta", \
            "--beta-corr perturbs the BETA pre-activations; use --no-beta-corr for the gaussian arm"
        assert abs(args.corr_rho) < 1.0, \
            "corr_rho must lie strictly inside (-1,1): |rho| = 1 makes the latent non-stationary"
        assert args.corr_gain_max > 0.0, "corr_gain_max must be positive"
        assert args.corr_gain_fixed is None or abs(args.corr_gain_fixed) < args.corr_gain_max, \
            "corr_gain_fixed must lie strictly inside (-corr_gain_max, corr_gain_max)"
    else:
        assert args.corr_gain_fixed is None, "--corr-gain-fixed requires --beta-corr"
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
    if args.beta_corr:
        # DEDICATED RNG STREAM. xi must NOT perturb the global torch generator, or
        # Beta.sample() would receive different numbers and the g == 0 arm would stop being
        # bit-exact with the chassis. One generator, seeded off args.seed, keeps EVERY arm
        # RNG-MATCHED on the action sampler: arms differ only in the transformation applied.
        corr_gen = torch.Generator(device=device)
        corr_gen.manual_seed(args.seed + 0x5EED)
        corr_act_dim = int(np.prod(envs.single_action_space.shape))
        # z_t (the latent USED at step t) and the Beta-mean displacement it produced.
        corr_zs = torch.zeros((args.num_steps, args.num_envs, corr_act_dim)).to(device)
        mode_shifts = torch.zeros((args.num_steps, args.num_envs, corr_act_dim)).to(device)
        corr_inn = float(np.sqrt(1.0 - args.corr_rho ** 2))
        # N(0, I) is the stationary law of the recursion for ANY rho, so seed from it.
        z_carry = torch.randn(args.num_envs, corr_act_dim, device=device, generator=corr_gen)
        z_reset_acc = torch.zeros((), device=device)
        corr_lag1_z = np.zeros(corr_act_dim)
        corr_lag1_shift = np.zeros(corr_act_dim)
        corr_mi = float("nan")

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # v3: adaptive proximal coefficient (state, not a hyperparameter after init) and the
    # circuit-breaker counter.
    kl_beta_state = args.kl_beta
    grad_split = float("nan")
    kl_breaker_trips = 0
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.beta_corr:
            z_reset_acc.zero_()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if args.beta_corr:
                # AR(1) advance, ONE draw per step so the stream is independent of the done
                # pattern (reproducibility). At an episode boundary z_t IS the fresh draw:
                # exactly N(0, I), which is also the recursion's stationary law, so no
                # previous episode's momentum survives a reset AND the first post-reset step
                # is drawn from the same law as every other step. `next_done` is the chassis's
                # own terminations|truncations of the PREVIOUS step (== dones[step]).
                xi = torch.randn(
                    args.num_envs, corr_act_dim, device=device, generator=corr_gen
                )
                z_carry = torch.where(
                    next_done.unsqueeze(1) > 0, xi, args.corr_rho * z_carry + corr_inn * xi
                )
                corr_zs[step] = z_carry
                z_reset_acc += next_done.sum()

            with torch.no_grad():
                action, z, logprob, ent, value_logits, dist = agent.get_action_and_value(
                    next_obs,
                    corr_z=z_carry if args.beta_corr else None,
                    probe=args.beta_corr,
                )
                p = torch.softmax(value_logits[:, 0], dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            p1, p2 = agent.dist_params(dist)
            dist_p1[step] = p1
            dist_p2[step] = p2
            if args.beta_corr:
                mode_shifts[step] = dist.corr_mode_shift

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
        if args.beta_corr:
            b_corr_zs = corr_zs.reshape(-1, corr_act_dim)
            # The BEHAVIOR gain. No optimizer step runs during a rollout, so the live gain is
            # still exactly the one that generated every sample in this buffer.
            gain_old = agent.corr_gain().detach()
            with torch.no_grad():
                # TEMPORAL-CORRELATION PROBES. corr_lag1_z should reproduce corr_rho (it is
                # the latent process itself); corr_lag1_shift is the realized autocorrelation
                # of the Beta MEAN displacement, i.e. of the coherent drift the actor actually
                # executed. Both read ~0 for the --no-beta-corr control.
                corr_lag1_z = within_episode_lag1_ac(corr_zs, transition_boundaries)
                corr_lag1_shift = within_episode_lag1_ac(mode_shifts, transition_boundaries)
                # CONDITIONAL-vs-MARGINAL log-density gap of the BEHAVIOR policy, in
                # nats/step: E[log pi(a|s,z_used)] - E[log (1/K) sum_k pi(a|s,z_k)] = a
                # K-sample estimate of the mutual information I(a_t; z_t | s_t) >= 0, i.e.
                # how much the latent actually tells you about the action. lp_cond is the
                # STORED log_prob (exact, nothing recomputed); the unshifted pre-activations
                # needed for the marginal are recovered EXACTLY from the stored SHIFTED
                # concentrations by inverting 1 + softplus. The K-sample logsumexp biases
                # lp_marg DOWN, so the gap is an upper bound with an O(1/K) bias.
                n_mi = min(args.corr_mi_states, args.batch_size)
                if n_mi > 0 and args.corr_mi_draws > 0:
                    midx = torch.randint(
                        0, args.batch_size, (n_mi,), device=device, generator=corr_gen
                    )
                    s_used = gain_old * b_corr_zs[midx]
                    ra = inv_softplus(b_dist_p1[midx] - 1.0) - s_used
                    rb = inv_softplus(b_dist_p2[midx] - 1.0) + s_used
                    zk = torch.randn(
                        (args.corr_mi_draws, n_mi, corr_act_dim),
                        device=device,
                        generator=corr_gen,
                    )
                    sk = gain_old * zk
                    ak = 1.0 + F.softplus(ra.unsqueeze(0) + sk)
                    bk = 1.0 + F.softplus(rb.unsqueeze(0) - sk)
                    wk = b_latent_zs[midx].unsqueeze(0).expand_as(ak)
                    lp_k = Beta(ak, bk).log_prob(wk)
                    lp_marg = (
                        torch.logsumexp(lp_k, dim=0) - log(float(args.corr_mi_draws))
                    ).sum(-1)
                    corr_mi = (b_logprobs[midx] - lp_marg).mean().item()
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
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            epoch_kl_sum, epoch_kl_n = 0.0, 0
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits, new_dist = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_latent_zs[mb_inds],
                    corr_z=b_corr_zs[mb_inds] if args.beta_corr else None,
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
                if args.pg_clip:
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
                    pg_g = torch.autograd.grad(pg_loss, actor_params, retain_graph=True, allow_unused=True)
                    kl_g = torch.autograd.grad(kl_penalty, actor_params, retain_graph=True, allow_unused=True)
                    pg_gn = float(torch.norm(torch.stack([g.norm() for g in pg_g if g is not None])))
                    kl_gn = float(torch.norm(torch.stack([g.norm() for g in kl_g if g is not None])))
                    grad_split = kl_gn / (pg_gn + 1e-12)

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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
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
        if args.beta_corr:
            gain_post = agent.corr_gain().detach().cpu().numpy()
            writer.add_scalar("debug/corr_rho", args.corr_rho, global_step)
            writer.add_scalar("debug/corr_gain_absmax", float(np.abs(gain_post).max()), global_step)
            writer.add_scalar("debug/corr_gain_absmean", float(np.abs(gain_post).mean()), global_step)
            for i, gv in enumerate(gain_post):
                writer.add_scalar(f"debug/corr_gain_{i}", float(gv), global_step)
            writer.add_scalar("debug/corr_z_lag1_ac_mean", float(corr_lag1_z.mean()), global_step)
            writer.add_scalar(
                "debug/corr_shift_lag1_ac_mean", float(corr_lag1_shift.mean()), global_step
            )
            for i, a in enumerate(corr_lag1_shift):
                writer.add_scalar(f"debug/corr_shift_lag1_ac_{i}", float(a), global_step)
            writer.add_scalar(
                "debug/corr_mode_shift_absmean", mode_shifts.abs().mean().item(), global_step
            )
            writer.add_scalar(
                "debug/corr_mode_shift_absmax", mode_shifts.abs().max().item(), global_step
            )
            writer.add_scalar("debug/corr_cond_minus_marg_gap", corr_mi, global_step)
            writer.add_scalar(
                "debug/corr_z_reset_frac",
                (z_reset_acc / float(args.batch_size)).item(),
                global_step,
            )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
