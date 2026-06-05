# PPO + IterThink v48 (SAC-Beta max entropy). From v47 + v24.5 + ../sac-beta.
# Goal: make SAC's max-entropy policy-gradient decomposition real for the Beta actor instead of
# silently disabling auto-alpha outside the Gaussian path.
#     soft_r_t   = r_t + gamma * alpha_eff * H_pi(s_{t+1})          (alpha_eff = alpha/return_std)
#     V_soft     = lambda_return(soft_r_t, V_soft)                  (the C51 critic target)
#     A_soft     = R_soft_lambda - V_soft(s_t)                      (the policy advantage)
# Changes from v47:
#   (1) Beta is the default and auto-alpha now supports it. The v24/v47 Beta path previously ran as plain
#       PPO because `auto_alpha = auto_entropy and actor_dist == "gaussian"`.
#   (2) The Beta actor uses the SAC-Beta/Tianshou shifted log-concentration parameterization:
#           alpha = exp(clamp(head_a, min, max)) + 1; beta = exp(clamp(head_b, min, max)) + 1
#       with the widened v24.5 cap (exp(8)+1) so HalfCheetah can become sharp when reward demands it.
#   (3) The default target is SAC parity, target_entropy = -|A|, with log_alpha initialized at 0.
#       Beta entropy is measured in native [0,1] space and the constant [-1,1] action-scale Jacobian is
#       omitted, matching the PPO ratio and the ../sac-beta reference convention.
#   (4) Defaults use the v24 Beta actor with sign-preserving rankgauss_signmag shaping while
#       keeping v47's state-entropy soft critic, time-limit bootstrap, and live alpha fixes.
#   (5) Direct entropy is no longer rescaled by shaped-advantage std by default; with Beta/rankgauss that
#       amplification was spending KL budget on a tiny but persistent off-clip entropy gradient.
# Inherited from v43: time-limit truncations bootstrap final observations and gamma on the entropy term.
# Inherited from v42: alpha/return_std scaling. Inherited from v41: single-counted soft-value label and one alpha.
#
# ---- inherited v43 notes ----
# PPO + IterThink v43 (SAC soft value in the C51 critic + correct TIME-LIMIT handling). From v42.
# v42 fixed the entropy SCALE; v43 fixes two SAC-backup alignment defects, leaving the entropy mechanism
# (alpha/return_std scaling, ONE alpha, reachable target, single-counted denoised label) UNCHANGED.
#
# WHY v43.
#   (1) TIME-LIMIT TRUNCATIONS WERE TREATED AS TRUE TERMINALS (the big one). All prior versions folded
#       truncations into `done` and zeroed the value bootstrap there. But MuJoCo episodes end by a 1000-step
#       TIME LIMIT, not a real terminal -- HalfCheetah has NO termination at all, so EVERY episode boundary
#       was wrongly zeroing V near the limit, biasing the value low. SAC bootstraps the soft value of the
#       cut-off state. v43 does the same: it stashes infos["final_observation"] for truncated steps, values
#       them (V and Z), and bootstraps gamma*V(final_obs) for truncation while only TRUE termination
#       (done & ~trunc) zeroes the bootstrap. Both the scalar soft-GAE and the distributional lambda-return
#       carry this (numerically verified: distributional mean-matches the scalar truncation-aware return).
#   (2) NEXT-STATE ENTROPY SAT BEFORE THE DISCOUNT. soft_rewards added alpha_eff*H(s_{t+1}) at reward level;
#       SAC's backup r + gamma*(Q - alpha*logpi(s')) puts that entropy INSIDE gamma. v43 multiplies the
#       entropy bonus by gamma (was a ~1% / (1/gamma) overweight). Exact, free.
# NOT changed: the per-rollout staleness of the soft-value labels through PPO epochs is inherent to PPO's
# batched targets (the reward targets are equally fixed; the importance ratio + fresh direct-entropy term
# are the on-policy corrections) -- refreshing them every epoch would be abandoning PPO for SAC, not a fix.
#
# ---- inherited v42 notes ----
# PPO + IterThink v42 (SAC soft value IN THE C51 CRITIC -- entropy folded in on the REWARD'S OWN SCALE).
# From v41. Same SAC structure (future entropy = single-counted denoised label in the value, ONE alpha,
# reachable target); v42 ONLY fixes how the entropy bonus is SCALED relative to the reward.
#
# WHY v42. v41 normalized the entropy bonus by ent_level = mean(discounted entropy return), a scale
# SEPARATE from the reward's. Two coupled failures (seen in the 8M trajectories, worst on HalfCheetah):
#   (1) SCALE MISMATCH. The reward is NormalizeReward-normalized (divided by return_std ~85-95), but the
#       entropy was divided by ent_level (~6 on HC). With return_std >> ent_level the entropy bonus was
#       effectively weighted ~15x the reward, so the soft reward became entropy-DOMINATED and HC's reward
#       value learning was corrupted (return stalled ~2000 vs v40's 7175).
#   (2) SINGULARITY AT THE TARGET. The reachable target is 0 nats, so the dual drives H -> 0; but ent_level
#       -> 0 there too, so alpha_eff = alpha/ent_level EXPLODED exactly at the target. HC's ent_bonus_frac
#       blew up 0.04 -> 5.5 as H crossed 0 to negative; Walker hit the same spike transiently at 1.9M.
# THE FIX: divide the entropy bonus by the SAME return_std the reward uses -- alpha_eff = alpha/return_std.
# Now H and r live in ONE common normalized frame: entropy is a controlled fraction (~alpha) of the reward,
# balanced for every env, and the entropy value VANISHES smoothly as H -> 0 (no singularity). Support safety
# (the reason v41 avoided return_std) is recovered by raising return_std_floor 1 -> 20, which bounds the
# only overflow regime -- the EARLY transient where return_std ~ 0 while H is still large. The dual,
# reachable target, single-counting, and C51 denoising are all UNCHANGED from v41 (they worked: alpha
# equilibrated ~0.08, edge_mass ~ 0).
#
# ---- inherited v41 notes ----
# PPO + IterThink v41 (SAC soft value IN THE C51 CRITIC -- entropy as a single-counted, denoised label).
# From v40. This is the real SAC structure, not the v35-40 "entropy in a side GAE" approximations.
#
# WHY v41. A three-agent adversarial review of v40 found its "dynamic SAC" story was largely fiction:
#   (1) DUAL WAS DECORATIVE. target = -|A| is UNREACHABLE on-policy (squashed H stays far above it for
#       8M steps; PPO contracts entropy ~100x slower than off-policy SAC), so the dual gradient is
#       constant-sign and alpha just bled to ~0.037 on all three envs. No fixed point, no equilibration.
#   (2) THE ONLY LIVE ENTROPY CHANNEL WAS ALPHA-INDEPENDENT. v38-40's soft channel weighted the
#       future-entropy advantage by a FIXED 0.25 (alpha absent), so the one channel that survived the
#       alpha bleed was a hand-set constant -- the opposite of "dynamic like SAC". There was no single
#       temperature governing the objective.
#   (3) ENTROPY SIGNAL WAS NOISY + RANK-LAUNDERED. A separate GAE accumulated SINGLE-SAMPLE -logpi over
#       the horizon (std grew 7->30 on HC) and rankgauss_signmag then discarded its magnitude.
# Net: v40 was ~pure-reward PPO + a constant rank-ordered nudge + a dead dual. Hopper (775) stalled
# because, once alpha bled, NO entropy support remained and the policy collapsed to a hop-and-fall gait.
#
# THE FIX (the SAC mechanism done properly): put the future entropy WHERE SAC puts it -- inside the
# learned value -- as a single-counted, denoised LABEL, governed by ONE alpha, with a REACHABLE target
# so the dual actually equilibrates. SAC's soft-Q target is r + gamma*(Q(s',a') - alpha*logpi(a'|s')):
# the next-state entropy is added to the BOOTSTRAP, regressed into Q, and averaged (denoised) over many
# samples. v41 translates this to the on-policy C51 critic:
#
#   soft_rewards[t] = rewards[t] + (alpha / ent_level) * H(s_{t+1}) * nonterminal_{t+1}
#       H(s_{t+1}) = -logpi(a_{t+1}|s_{t+1})  (single sample; the C51 critic DENOISES it by regression)
#       ent_level = mean discounted entropy return. SUPPORT SAFETY: the entropy contribution to the soft
#       VALUE has level ~alpha*E[sum g^k H]; the ~1/(1-gamma)~100 horizon amplification means dividing by
#       the REWARD return_std under-normalizes ~25x and overflows the fixed C51 support (the historical
#       "softboot" failure). Normalizing by the entropy return's own LEVEL pins the entropy value level to
#       ~alpha (<= alpha_max), inside the support for ANY env/horizon. The constant level cancels in the
#       advantage (present in both return and V_soft baseline), so the future-entropy exploration SIGNAL
#       (the per-state variation) is preserved. The CURRENT-state entropy is handled by the analytic actor
#       term -(alpha/return_std)*H(s_t) (a per-step quantity -> per-step reward normalizer).
#       * nonterminal gates the bonus at episode boundaries (also fixes v40's cross-episode leak bug).
#
# soft_rewards feeds BOTH the scalar GAE (-> policy advantage) AND the distributional lambda-return
# (-> critic target). So the critic learns V_soft, the advantage = soft GAE is automatically baselined
# by V_soft (no separate V_ent, no centering, no scale-match), and future-entropy credit is
# single-counted and denoised -- exactly SAC's soft-Q. The CURRENT-state entropy is handled by SAC's
# analytic actor term (pg_loss -= (alpha/return_std)*H(s_t)) at the SAME effective temperature.
#
# REACHABLE TARGET: target_entropy_coef = 0.0 (target = 0 nats), the on-policy free-descent level the
# v32 notes identified as crossable. Now the dual gets a TWO-SIDED gradient: H>0 lowers alpha, H<0
# raises it -> alpha equilibrates at a positive value (NOT a dead floor) -> the soft value keeps a
# meaningful, dynamically-tuned entropy term alive. alpha is clamped to [alpha_min, alpha_max]; alpha_max
# bounds the soft return inside the C51 support (monitor debug/edge_mass; widen v_min/v_max if it grows).
# The v36-40 V_ent head is RETIRED (entropy now lives in the main critic); it remains defined but
# untrained/unused to keep call signatures stable, to be removed in a cleanup pass.
#
# ---- inherited v40/v38 notes ----
# PPO + IterThink v38 (SAC max-entropy IN GAE -- SCALE-CONTROLLED future-entropy advantage). From v37.
#
# WHY v38. v37's mean-centering killed v35/v36's entropy PEDESTAL (ent_adv_mean -> 0 on all envs,
# distpg_sign_agree recovered 0.35 -> 0.73 on HC) but exposed a second failure: the entropy
# advantage's VARIANCE runs away. Logged over 0-425k: alpha barely anneals (0.099 -> 0.088; the
# once/epoch dual at lr 1e-3 is ~20x too slow), while the GAE-accumulated SINGLE-SAMPLE-logprob
# entropy noise GROWS with the horizon (ent_adv_std 7 -> 30 on HC) AND the reward advantage SHRINKS
# as the critic fits (EV -> 0.72). So soft_adv_std_ratio climbed monotonically to ~8.8 on HC / ~4.9
# on Walker -- the entropy advantage's std reached ~9x the reward advantage's, swamping the policy
# gradient: HC stuck ~-220, Walker stuck ~2 (only Hopper, where entropy genuinely varies by state,
# climbed). This is the v28-37 tension stated cleanly: ALPHA-WEIGHTING THE GAE-ACCUMULATED ENTROPY
# IS SCALE-UNSTABLE -- inert when the dual bleeds alpha to the floor (v28-31, ratio ~1.0), dominant
# when alpha stays meaningful (v35-37, ratio >> 1) -- and the slow dual cannot thread the needle.
# v38 FIX: set the in-GAE entropy scale EXPLICITLY instead of via alpha. After centering, rescale
# ent_adv to the reward advantage's std and add a FIXED fraction: policy_adv = reward_adv +
# soft_adv_coef * (ent_adv / ent_adv.std() * reward_adv.std()). Now soft_adv_std_ratio ~ sqrt(1+coef^2)
# (= ~1.03 at coef 0.25) FOREVER -- bounded across alpha, horizon, and env. The entropy can never
# dominate; it is a controlled exploration NUDGE that reorders reward-similar actions toward higher
# future entropy (signal where entropy varies by state, minor noise where ~constant like HC, so HC
# reverts to ~pure-reward PPO and should recover toward ~8800). The alpha dual + binding floor STILL
# drive the DIRECT -alpha*logpi term (the sigma-maintenance / true max-ent channel that rescued the
# terminating envs in v35). One isolated change from v37 (the in-GAE channel's magnitude rule).
#
# ---- inherited v37 notes ----
# PPO + IterThink v37 (SAC max-entropy IN GAE -- CENTERED future-entropy advantage). From v36.
#
# WHY v37. v36 tried to fix v35's pedestal by learning a future-entropy value V_ent and forming a
# baselined entropy GAE (ent_adv = GAE(H(s') + gamma*V_ent(s') - V_ent(s))). The IDEA is right but
# relying on V_ent's ABSOLUTE level to center the advantage FAILED on HalfCheetah: empirically (65k
# steps) ent_adv_mean ~ 60 >> ent_adv_std ~ 9, soft_adv_std_ratio ~ 2.2, distpg_sign_agree ~ 0.35,
# return stuck ~ -250 -- the v35 pedestal RETURNED. Root cause: HC's squashed entropy is ~constant
# across states (~4 everywhere), so the future-entropy return has almost NO per-state variance =>
# the (detached, linear) V_ent head has nothing to learn (ent explained-var ~ 0) and predicts ~a
# constant that does NOT satisfy the Bellman level needed to cancel the per-step entropy reward's
# large positive mean (E[H]~4 vs a constant-V TD residual of only -V*(1-gamma)). The GAE then
# re-accumulates that ~3/step residual into a ~60 pedestal that flips rankgauss_signmag's sign.
# v37 FIX: MEAN-CENTER ent_adv per rollout (ent_adv -= ent_adv.mean()) before policy_adv =
# reward_adv + alpha*ent_adv. This GUARANTEES a zero-mean entropy advantage -- the pedestal cannot
# survive a mean subtraction -- independent of how well V_ent fits, while KEEPING the per-state
# variation (the genuine "leads-to-higher-future-entropy" signal). V_ent is retained as a
# variance-reducing baseline where it DOES fit (terminating envs, ent EV ~0.2-0.3); its absolute
# level no longer matters for centering. On HC, where entropy barely varies by state, the centered
# alpha*ent_adv is a small near-noise perturbation => HC reverts to ~pure-reward PPO and should
# recover toward its ~8800, while the direct -alpha*logpi term still sustains exploration.
# Everything else (binding floor, V_ent head + own optimizer + Welford normalization, no alpha/2
# split, all v35 machinery) is unchanged from v36 -- this is a one-line, isolated correction.
#
# ---- inherited v36 notes ----
# PPO + IterThink v36 (SAC max-entropy IN GAE -- BASELINED future-entropy advantage). From v35.
#
# WHY v36. v35's binding floor RESCUED the terminating envs (Hopper ~990, Walker ~1790) but the
# soft-adv-in-GAE channel COLLAPSED HalfCheetah (~8800 -> ~350). Three independent red-team reviews
# (math, SAC-translation, empirical) converged on ONE implementation bug -- NOT "max-ent hurts HC"
# (SAC gets ~10-12k on HC WITH max-ent, so the mechanism is right; the translation was wrong):
#   v35 added the future-entropy bonus b_t = alpha*H(s_{t+1}) to the GAE *forward return* (the
#   nextvalues + b_t bootstrap) but the BASELINE values[t] and the critic target are ENTROPY-FREE.
#   So  A_soft[t] = A_rew[t] + gamma*alpha * SUM_k (gamma*lambda)^k H(s_{t+1+k}) -- the entropy term
#   is a discounted sum of FUTURE entropies that is NEVER baselined (SAC baselines Q_soft with V_soft;
#   we added the return half but never built the baseline half). This residual is mean-POSITIVE and
#   HORIZON-ACCUMULATING: on HalfCheetah (non-terminating, full 1000-step horizon, nonterminal==1)
#   it saturates to a near-constant pedestal ~0.8 normalized units at EVERY state, swamping HC's
#   small zero-mean reward advantage and -- because rankgauss_signmag takes sign(gae) BEFORE norm --
#   FLIPPING the advantage sign to +1 almost everywhere => the policy gradient direction is destroyed
#   (empirically: soft_adv_std_ratio 1.1->2.5, distpg_corr_with_gae -> ~0/negative, sign_agree ->
#   coin-flip, while the critic EV stayed healthy => the bug is purely in HOW THE ADVANTAGE IS FORMED).
#   Terminating Hopper/Walker truncate the sum at done (nonterminal=0) => smaller, signal-bearing bias
#   => they survived. The v35 alpha/2 split only SCALED the pedestal; it never CENTERED it.
# THE FIX (SAC-faithful): the SAME entropy added to the return must be SUBTRACTED by a baseline.
# v36 learns the future-entropy value with a separate SCALAR head V_ent (MSE; no categorical support
# to overflow -- which is exactly what sank the "softboot" attempt to put entropy in the C51 critic)
# and forms a properly BASELINED entropy advantage via its own GAE:
#     entropy reward  e_t = H(s_{t+1}) = -logpi(a_{t+1}|s_{t+1})        (raw, alpha-free)
#     ent_delta_t     = e_t + gamma*V_ent(s')*nonterm - V_ent(s_t)
#     ent_adv         = GAE(ent_delta)            # CENTERED: realized future entropy - predicted
#     policy_adv      = reward_adv + alpha * ent_adv      # zero-mean perturbation, NO pedestal
# V_ent regresses its own lambda-return (ent_adv + V_ent). The reward critic + reward GAE are
# UNTOUCHED (proven; EV stays healthy). This is SAC's decomposition Q_soft = Q_reward + V_entropy.
# NO DOUBLE-COUNT, so NO alpha/2 split: ent_adv credits action a_t for the entropy of FUTURE states
# (s_{t+1} onward); the direct -alpha*logpi(a_t) term supplies the CURRENT action's entropy gradient
# (which a detached advantage cannot). Disjoint timesteps -- each channel gets the FULL alpha, exactly
# as SAC (future entropy in the value, current entropy in the actor's -alpha*logpi).
# Hypothesis: with the pedestal removed, max-ent-in-GAE helps ALL THREE envs -- HC recovers toward
# its ~8800 (and beyond, if future-entropy credit genuinely aids exploration) while Hopper/Walker
# keep v35's gains. The binding floor + all other v35 machinery are unchanged (isolated change).
#
# ---- inherited v35 notes ----
# PPO + IterThink v35 (SAC max-entropy IN GAE + BINDING entropy floor). From v34.
#
# WHY v35. An empirical audit of v32-v34 on the TERMINATING envs (Hopper/Walker2d, stuck ~480-660)
# found the max-entropy mechanism was IMPLEMENTED but INERT, for one root cause: the alpha-dual
# NEVER BINDS. With the SAC parity target (target_entropy = -1.0*|A| = -3 Hopper / -6 Walker), the
# squashed entropy H lives FAR ABOVE the target the entire run (H/dim ~ 0.67 early, collapsing to
# ~ -0.18 Hopper / +0.26 Walker), so the dual gradient alpha*(H - target) is CONSTANT-SIGN POSITIVE
# => alpha only ever DECREASES => it bleeds to the floor. Two failures, ONE cause:
#   * alpha -> floor => the soft credit alpha*H -> ~0 => the soft-advantage-in-GAE channel is INERT
#     (exactly v28-31's measured soft_adv_std_ratio ~ 1.00). Entropy never reaches the policy.
#   * No entropy floor => sigma COLLAPSES prematurely (data: 0.5 -> 0.14 on Hopper) => the policy
#     converges to a deterministic never-survives gait while EV stays 0.95-0.99.
# THE FIX (pure SAC max-ent, no vanilla-PPO entropy): make the dual BIND as an entropy FLOOR. In
# SAC's fast off-policy regime -|A| is reachable from above; in our SLOW on-policy regime it is not,
# so the dual must instead target a REACHABLE level ABOVE the free-collapse floor. Then when H falls
# below target, alpha(H-target)<0 => alpha RISES => the actor is pushed to RE-EXPAND entropy: a real
# fixed point. With alpha held meaningful, alpha*H in the soft bootstrap V(s')+alpha*H(s') is on-scale
# and the soft-advantage GAE actually REORDERS the policy advantage (no longer inert).
# v35'S CHANGES from v34 (all serving the SAC-max-ent-in-GAE thesis):
#   (1) FULL SAC SOFT OBJECTIVE (both channels, decoupled). SAC's soft objective has TWO entropy
#       terms: future entropy in the soft-Q backup, and the direct -alpha*log_pi term on the current
#       action. Our channels are exactly that decomposition and do NOT double-count (different
#       timesteps): soft_adv=True puts FUTURE entropy IN THE GAE via the soft bootstrap V(s')+alpha*H(s')
#       ("max entropy in GAE"); direct_entropy=True adds the CURRENT-step -alpha*H actor gradient that
#       directly holds sigma up (the soft channel alone cannot -- current-state entropy is action-
#       independent and cancels in the PG, so it credits only high-entropy FUTURES, never inflating
#       sigma now). v28-34 forced these mutually exclusive; v35 decouples them so both run, as in SAC.
#       ALPHA-SPLIT (correctness): on an on-policy GAE, running both channels DOUBLE-COUNTS each
#       interior state's entropy H(s_k) -- once directly at step k, once as the successor entropy of
#       step k-1 in the soft GAE (SAC avoids this because future entropy is a LABEL inside Q, not
#       stacked additively on the same rollout). v35 splits alpha (soft_alpha_frac=0.5 each) so each
#       state's total entropy weight stays ~alpha, matching SAC's single budget.
#   (2) BINDING FLOOR TARGET: target_entropy_coef = +0.5 (per dim). Set ABOVE the measured free-collapse
#       level so the dual binds (Hopper holds H~1.5 vs collapse to -0.5; Walker holds H~3.0 vs 1.5).
#       This REPLACES the -1.0 SAC-parity target, which is unreachable here and un-binds the dual.
#       A live (non-bled) alpha is also what makes the soft-adv channel non-inert (v28-31's bled alpha
#       drove alpha*H -> 0 => soft_adv_std_ratio ~ 1.00).
#   (3) (kept from v34) rankgauss_signmag advantage, share_backbone=True, dual_freq=epoch (gentle
#       cadence), alpha_min=0.02 -- the v32 equilibrium machinery, now with a target it can equilibrate to.
# Hypothesis: the dual equilibrates (alpha dips then RISES to hold H at target), the soft-adv channel
# becomes live, the entropy floor sustains exploration, and Hopper/Walker2d climb past the ~660 ceiling.
# If HC regresses (its v32 win equilibrated near H=0), make the target env-aware.
#
# ---- inherited v34 notes ----
# PPO + IterThink v34 (SIGN-CORRECT advantage ONLY -- minimal change from v32). From v32.
#
# WHY v34. v33 stacked THREE changes (signmag transform + unshared trunk + batch-scope norm_adv)
# and HalfCheetah COLLAPSED 8800 -> 5042 (-43%), though the terminating envs improved (Hopper 905,
# Walker 1344). The regression couldn't be attributed because three levers moved at once. The HC
# win in v32 was built ON the shared ThinkTrunk + distributional-critic synergy, so unsharing the
# trunk (v33's biggest *Hopper* lever) is the prime suspect for breaking HC -- a genuine env-dependent
# conflict that should NOT be a global default.
# v34 isolates the ONE change that should help the terminating envs WITHOUT touching HC's winning
# ingredients: switch ONLY the advantage transform to "rankgauss_signmag" (sign-correct, zero-crossing
# at gae=0 not the batch median) and keep EVERYTHING ELSE at v32 (share_backbone=True, norm_adv_scope
# =minibatch). Rationale: on HalfCheetah the advantages are ~symmetric (median~=0) so signmag is
# numerically ~identical to rankgauss => HC should be PRESERVED; on the left-skewed terminating envs
# the sign correction fixes the ~10-20% wrong-signed gradients (the clip_z diagnostic, also shared
# backbone, already lifted Hopper->785 / Walker->1334; signmag is the sign-EXACT version).
# Hypothesis: HC recovers to ~8800 AND Hopper/Walker improve over v32 -- one global config, no
# per-env trunk sharing. If terminating envs still lag, unshared-trunk becomes an env-specific knob.
#
# ---- inherited v32 notes ----
# PPO + IterThink v32 (EQUILIBRIUM dual -- make alpha look/behave like SAC's). From v31.
#
# WHY v28-v31's alpha looked NOTHING like SAC's. SAC's alpha dips then EQUILIBRATES (jitters
# around a steady positive value) because its dual gradient (H - target) FLIPS SIGN: off-policy
# Q-maximization drives policy entropy to target_entropy within ~tens of thousands of steps, so
# H crosses target early and the dual oscillates around a fixed point. Our ports produced a
# MONOTONE BLEED to the floor instead, for a confirmed, measured reason:
#   * TIMESCALE MISMATCH. PPO contracts entropy ~100x SLOWER than SAC (squashed H descends
#     +4 -> -4 over ~4M steps). With SAC's target -6 (or even -2), H stays ABOVE target for the
#     ENTIRE alive-alpha window, so (H - target) > 0 always => the dual gradient is CONSTANT-SIGN.
#   * ADAM SIGN-STEPPING + 320x CADENCE. Adam on a scalar moves log_alpha by ~+-lr per step
#     regardless of |grad|. Stepping the dual per-minibatch = update_epochs*num_minibatches =
#     320 same-sign steps/rollout on REUSED data => Delta log_alpha ~ -0.32/rollout => alpha rails
#     to its floor by ~230k, MILLIONS of steps before H could ever cross target. (Measured: v29
#     H crosses -6 at 5.57M but alpha floored at 229k; v30/v31 never reach -2 before alpha dies.)
#   NOTE: the entropy ESTIMATE was faithful (summed over dims, tanh Jacobian, fresh rsample,
#   batch-meaned) -- scale was never the bug. And we deliberately keep alpha OUT of the C51 critic
#   target (SAC couples it there) because that resurrects the v26/v27 soft-critic blowup (support
#   overflow / symlog-Jensen, scored 187-288). The DIRECT actor bonus already gives the negative
#   feedback (alpha^ => H^ => grad v) needed for a fixed point, without touching the critic.
#
# v32's three fixes so the dual binds while alpha is alive, then equilibrates:
#   (1) REACHABLE TARGET: target_entropy = 0.0 -- where the policy's free descent sits at ~1-1.5M,
#       i.e. INSIDE the alive-alpha window. H crosses 0 there, the sign flips, alpha turns around.
#   (2) GENTLE CADENCE: step the dual ONCE PER EPOCH (10x/rollout) not per-minibatch (320x), on the
#       epoch-mean fresh entropy. Far fewer same-sign steps => alpha decays gently instead of railing.
#   (3) REAL FLOOR: alpha_min = 0.02 (not 1e-6) keeps temperature alive through the pre-bind transient
#       so it can RISE when the sign flips. alpha_init = 0.1 (SAC-ish), normalized frame, direct channel.
# Hypothesis: alpha now traces SAC's dip-then-equilibrate shape AND functionally holds entropy at 0,
# preventing the late over-collapse seen in v29 (return peaked ~5850 then fell to ~3170 as H->-6).
#
# ---- inherited v28 notes ----
# PPO + IterThink v28 (EQUIVALENT-ALPHA soft max-ent). From v24 (proven 4868 base).
#
# WHY v28. Porting SAC's entropy temperature to this PPO pipeline failed repeatedly for ONE
# root cause: a UNITS mismatch. SAC's alpha lives in RAW-reward units (HalfCheetah per-step
# reward ~O(1)). Here gym.NormalizeReward divides rewards by the running return-std (measured
# ~16 early, growing to ~100+ as returns grow), so the reward the loop sees is ~0.02/step.
# An alpha calibrated in normalized units is therefore ~return_std (16-100x) off from SAC's.
# Consequences we observed:
#   * v26's "alpha collapse" 0.097->0.003 was NOT a healthy equilibrium -- it was a ONE-SIDED
#     bleed toward 0 (squashed H stayed ABOVE target -6, so the dual could only decrease alpha).
#     It happened to pass through a tolerable EFFECTIVE scale (0.003*return_std ~ SAC's 0.1-0.3)
#     where it learned (3795), but it was switching max-ent OFF, not equilibrating.
#   * Holding alpha at 0.08 (v27) = 0.08*return_std ~ 1.3-4 raw = 6-20x SAC -> the entropy term
#     swamped the (tiny, normalized) reward and PINNED the policy at the SQUASHED-ENTROPY PEAK
#     (H_sq is NON-monotone in sigma: peaks +4 at mu~0,std~1 = max-variance CENTERED actions =
#     near random; H_sq only goes very negative once mu SATURATES, e.g. mu=2.5 -> H~-17).
# v28'S FIX (SUPERSEDED by v30 -- see top): inject entropy in RAW units by scaling the bonus by 1/return_std, so the
# learned alpha sits in SAC's native 0.1-0.3 range AND means the same thing. This keeps the
# EFFECTIVE entropy weight small/SAC-like, so the reward gradient can SATURATE the policy
# (instead of pinning it at the H_sq peak); a saturated policy's H drops BELOW -6, at which
# point the dual finally BINDS (two-sided) and alpha equilibrates instead of bleeding to 0.
# v48 differs from this inherited v28/v24 path: direct_entropy is active again, future entropy
# is in the soft critic target, and the critic is no longer entropy-free. The actor bonus is
# kept small by alpha/return_std and no longer amplified by shaped-advantage std by default.
#
# ---- inherited v24 notes ----
# SOFT MAX-ENTROPY add-on (--auto-entropy; v48 supports both Beta and Gaussian). Tests whether SAC's
# entropy temperature needs a SOFT VALUE to be coherent on-policy. We bake the
# closed-form policy entropy into the RETURN: r~_t = r_t + alpha*H(pi(.|s_t)),
# feeding r~ into BOTH the scalar GAE and the distributional lambda-return, so the
# categorical critic learns the soft value and the bootstrap stays consistent
# (SAC's soft-Q, done on-policy). alpha is auto-tuned to hold H at target. Rationale:
# an earlier attempt added the entropy temperature as an actor bonus on top of an
# entropy-BLIND advantage (incoherent: the bonus pinned H while the advantage pulled
# greedy) and regressed hard. Baking entropy into the return reconciles the two into
# one max-ent objective. Use --no-norm-adv --adv-transform v10 to isolate the soft
# advantage from rankgauss/adv-norm. Off (default) => byte-identical to the v24 base.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal SAC-Beta/Tianshou shifted-log concentration:
#           alpha = exp(clamp(head_a, min, max)) + 1
#           beta  = exp(clamp(head_b, min, max)) + 1
#       The widened max log-shift gives HalfCheetah enough concentration range for sharp actions.
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = "direct log std" done right):
#       dreamer4's Gaussian readout. A state-dependent log-VARIANCE head (not a
#       flat Parameter, not log-std), SOFT-bounded by dreamer4's tanh-rescale
#       (not a hard clamp, so the gradient never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink tanh-squash + SAC Jacobian on the sample (mean stays
#       raw), base-Normal entropy. (#1 soft-clamp + #2 log-var from the dreamer4
#       parity review; the standing entropy bonus #3 was judged not relevant.)
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
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
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

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-7  # v24.5/SAC-Beta AD compatibility: clamp Beta samples just off the open boundary.


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
    norm_adv: bool = True
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: Optional[float] = 0.28  # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0            # legacy CleanRL PPO entropy coefficient; v48 requires this to stay zero.
    # SAC-style max entropy. Entropy enters the soft critic target through H_pi(s_{t+1})
    # and enters the actor through a current-state analytic entropy bonus. The same learned
    # alpha governs both channels. For Beta, H is the native [0,1] Beta entropy; the constant
    # linear rescale to [-1,1] is omitted just like the PPO log-ratio.
    auto_entropy: bool = True       # v28: equivalent-alpha soft max-ent ON by default
    soft_adv: bool = True           # v47: TRUE => future entropy enters the soft critic target:
                                    #      soft_r_t = r_t + gamma * (alpha/return_std) * H_pi(s_{t+1}).
                                    #      The same soft reward feeds scalar GAE and the distributional
                                    #      lambda-return, so the critic baseline is V_soft.
    direct_entropy: bool = True     # v47: SAC's current-state -alpha*log_pi actor term. The alpha used here is
                                    #      frozen to the same rollout alpha used in the soft critic target; the
                                    #      dual update is applied after each PPO epoch for the next rollout.
    target_entropy: Optional[float] = None   # absolute override; if None, resolved PER-DIM from
                                    #      target_entropy_coef * action_dim.
    target_entropy_coef: float = -1.0  # SAC parity: target = -|A| unless target_entropy overrides it.
    alpha_lr: float = 1e-3          # CleanRL SAC uses q_lr=1e-3 for the temperature optimizer.
    alpha_init: float = 1.0         # SAC auto-alpha parity: log_alpha initializes at 0.
    alpha_min: float = 1e-6         # clamp only for numerical safety; SAC baseline itself is unclamped.
    alpha_max: float = 1.0          # baseline starts at 1 and typically decays; cap prevents soft-critic blowups.
                                    #      With alpha_eff=alpha/return_std the soft-return entropy level is
                                    #      ~alpha*H_bar*100/return_std.
                                    #      Monitor debug/edge_mass; if it climbs, lower alpha_max or widen support.
    soft_entropy_min_coef: Optional[float] = -1.0  # clamp H(s') in the soft critic label to coef*|A|. Beta entropy
                                    #      can become much more negative than the C51 support can absorb; the direct
                                    #      actor entropy term still sees the unclamped current-policy entropy.
    dual_freq: str = "minibatch"    # SAC-like visible alpha dynamics; "epoch" is the gentler PPO-stability ablation.
                                    #      (320x). 320 same-sign Adam steps/rollout on reused data is the
                                    #      accelerant that rails alpha to the floor before the target binds.
    scale_direct_entropy_by_adv: bool = False  # Beta/rankgauss default: avoid amplifying the off-clip entropy term.
    return_std_floor: float = 20.0  # v42: clamp the entropy-scaling divisor. Raised 1->20 so the EARLY transient
                                    #      (return std ~0 while H is large) cannot push alpha*H_bar*100/std past the
                                    #      C51 support. Only affects the entropy scaling, NOT the wrapper's reward
                                    #      normalization. Below the ~85 steady-state std, so reward/entropy stay balanced.
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    diag: bool = True               # v44: extra per-rollout instrumentation (toggle). raw A_soft moments;
                                    #      reward-return vs entropy-return std split; pre-tanh |z| quantiles;
                                    #      policy mean/std min-max; mean-head vs logvar-head grad norms.
    adv_transform: str = "rankgauss_signmag" # v48: Beta stack with sign-preserving shaping for soft advantages.

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"
    norm_adv_center: bool = False   # signmag/tanh_std preserve the raw advantage sign; std-only keeps that property.

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 direct-log-std: state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "beta"       # v48: SAC-Beta is the default max-ent actor
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]
    entropy_samples: int = 2         # v47: antithetic samples for lower-variance squashed entropy estimates.
    beta_log_shift_min: float = -20.0  # beta: SAC-Beta shifted log-concentration clip
    beta_log_shift_max: float = 8.0    # beta: PPO-wide cap, exp(max)+1 ~= 2982 concentration cap

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


def make_env(env_id, idx, capture_video, run_name, gamma):
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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def get_return_std(envs, floor=1.0):
    """Mean running return-std across the per-env gym.NormalizeReward wrappers.

    This is the divisor NormalizeReward applies to rewards (reward_seen = reward_raw /
    sqrt(return_rms.var)). To inject the entropy bonus in SAC's RAW-reward units, we
    divide alpha*H by this so it lands in the SAME normalized scale as the reward the
    loop sees. Floored (raw |r|~1) to avoid blow-up while return_rms.var is ~0 at the start.
    """
    stds = []
    for e in getattr(envs, "envs", []):
        w = e
        while w is not None and not isinstance(w, gym.wrappers.NormalizeReward):
            w = getattr(w, "env", None)
        if w is not None:
            stds.append(float(np.sqrt(w.return_rms.var + 1e-8)))
    return max(float(np.mean(stds)) if stds else floor, floor)


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
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        # v24: action distribution. Both parameterizations are dreamer4-faithful.
        self.actor_dist = args.actor_dist
        self.entropy_samples = max(1, args.entropy_samples)
        if self.actor_dist == "gaussian":
            # dreamer4 direct-log-std: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # SAC-Beta/Tianshou shifted log-concentration: exp(clamped head) + 1.
            # The +1 keeps each marginal unimodal; the wide cap lets successful
            # HalfCheetah policies become sharp instead of underfitting at the bounds.
            self.actor_alpha_logshift_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_logshift_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.beta_log_shift_min = args.beta_log_shift_min
            self.beta_log_shift_max = args.beta_log_shift_max
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
        alpha = self.actor_alpha_logshift_head(actor_feat).clamp(
            min=self.beta_log_shift_min, max=self.beta_log_shift_max
        ).exp() + 1.0
        beta = self.actor_beta_logshift_head(actor_feat).clamp(
            min=self.beta_log_shift_min, max=self.beta_log_shift_max
        ).exp() + 1.0
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def get_beta_stats(self, x):
        if self.actor_dist != "beta":
            raise RuntimeError("get_beta_stats is only valid for beta actor_dist")
        actor_feat, _ = self._trunks(x)
        raw_alpha = self.actor_alpha_logshift_head(actor_feat)
        raw_beta = self.actor_beta_logshift_head(actor_feat)
        log_shift_alpha = raw_alpha.clamp(min=self.beta_log_shift_min, max=self.beta_log_shift_max)
        log_shift_beta = raw_beta.clamp(min=self.beta_log_shift_min, max=self.beta_log_shift_max)
        alpha = log_shift_alpha.exp() + 1.0
        beta = log_shift_beta.exp() + 1.0
        return raw_alpha, raw_beta, log_shift_alpha, log_shift_beta, alpha, beta

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value logits (B, num_bins). Caller converts logits via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

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
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            if self.entropy_samples == 1:
                zr = dist.rsample()
                entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
            else:
                half = (self.entropy_samples + 1) // 2
                eps = torch.randn((half,) + dist.loc.shape, device=dist.loc.device, dtype=dist.loc.dtype)
                eps = torch.cat([eps, -eps], dim=0)[: self.entropy_samples]
                zr = dist.loc.unsqueeze(0) + dist.scale.unsqueeze(0) * eps
                entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(-1).neg().mean(0)
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value_logits

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_logshift_head.parameters()) + list(self.actor_beta_logshift_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, truncs, final_probs, value_probs, bootstrap_probs,
    support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·[ trunc·Z(final) + (1-done)·((1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1}) ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones/truncs (T, B);
    value_probs/final_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    TIME-LIMIT (v43): a TRUNCATED step bootstraps from Z(final_obs) (the soft value of the
    cut-off state) instead of being zeroed like a true terminal -- so only true TERMINATION
    (done & ~trunc) collapses the atoms to the reward. truncs/final_probs select that path.
    Entropy/soft-value terms are NOT injected here -- the soft reward folds entropy into
    `rewards`; the critic regresses the soft λ-return.
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            done_t = next_done                          # (B,)  done at END of step t
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            done_t = dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        trunc_t = truncs[t]                             # (B,)  time-limit at END of step t
        # boot_mask = 1 - term, term = done & ~trunc = done - trunc (mutually exclusive, done>=trunc).
        # True termination (0) collapses atoms to the reward; continuing AND truncation bootstrap (1).
        boot_mask = 1.0 - (done_t - trunc_t)            # (B,)
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n) continuing dist
        tm = trunc_t.unsqueeze(-1)                      # (B, 1)
        dist_t = tm * final_probs[t] + (1.0 - tm) * mix # truncated -> Z(final); else -> mix
        gn = (gamma * boot_mask).unsqueeze(-1)          # (B, 1); 0 only on true termination
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(dist_t, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


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
        return (2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c))
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
        return (2.0 ** 0.5) * torch.erfinv(centered)
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
                out[side] = (2.0 ** 0.5) * torch.erfinv(ctr)
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
        z = (2.0 ** 0.5) * torch.erfinv(centered)
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
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c))).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs > 0
    assert args.num_steps > 0
    assert args.num_minibatches > 0
    assert args.total_timesteps > 0
    assert args.update_epochs > 0
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.dual_freq in ("epoch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    assert args.actor_dist in ("gaussian", "beta")
    assert args.ent_coef == 0.0, "v48 uses SAC alpha for entropy; CleanRL ent_coef must stay zero"
    assert args.batch_size % args.num_minibatches == 0, "batch must divide evenly into minibatches"
    assert args.minibatch_size > 0
    assert 0.0 < args.alpha_min <= args.alpha_init <= args.alpha_max
    assert args.soft_entropy_min_coef is None or args.soft_entropy_min_coef <= 0.0
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
        raise RuntimeError("CUDA is required for this research script; run with --cuda on a CUDA machine.")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    act_dim = int(np.prod(envs.single_action_space.shape))
    soft_entropy_min = (
        args.soft_entropy_min_coef * float(act_dim)
        if args.soft_entropy_min_coef is not None
        else None
    )

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # v44 diag: separate the two gaussian heads so their grad norms can be logged independently
    # (does the policy gradient flow into the MEAN head or the LOG-VAR head?). Captured pre-clip.
    if args.diag and args.actor_dist == "gaussian":
        mean_head_params = list(agent.actor_head.parameters())
        logvar_head_params = list(agent.actor_logvar_head.parameters())

        def _grad_norm(params):  # L2 norm over a param group's grads, on-device (no host sync)
            gs = [p.grad.detach().norm() for p in params if p.grad is not None]
            return torch.norm(torch.stack(gs)) if gs else torch.zeros((), device=device)
    mean_head_gn = logvar_head_gn = torch.zeros((), device=device)

    # Soft max-entropy temperature. log_alpha is learned so the soft-value bootstrap
    # weight and the actor entropy bonus self-tune to the same target entropy. For
    # Gaussian this is squashed tanh entropy; for Beta it is native [0,1] entropy
    # with the constant action-rescale Jacobian omitted.
    auto_alpha = args.auto_entropy
    if auto_alpha:
        # SAC heuristic: target entropy = -|A| unless explicitly overridden.
        target_entropy = args.target_entropy if args.target_entropy is not None else args.target_entropy_coef * float(act_dim)
        log_alpha = torch.full((1,), float(np.log(args.alpha_init)), requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    entropies = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    # v43 TIME-LIMIT BOOTSTRAP: truncs[t] = time-limit truncation at END of step t (vs true termination),
    # and final_obs_buf[t] = the (normalized) cut-off observation for those steps, so the value can
    # bootstrap from V(final_obs) instead of treating the time limit as a true terminal.
    truncs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    final_obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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
                action, z, logprob, ent, value_logits = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = (p * support).sum(dim=-1)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            entropies[step] = ent

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # v43: record time-limit truncations and stash the (already-normalized) cut-off obs so the
            # value bootstraps from V(final_obs). SyncVectorEnv auto-resets, so next_obs is the NEW
            # episode's reset -- the real final state lives in infos["final_observation"].
            time_limit_truncations = np.logical_and(truncations, np.logical_not(terminations))
            truncs[step] = torch.as_tensor(time_limit_truncations.astype(np.float32)).to(device)
            final_obs_buf[step].zero_()
            if time_limit_truncations.any():
                if "final_observation" not in infos:
                    raise RuntimeError("time-limit truncation missing infos['final_observation']; cannot soft-bootstrap")
                fobs_arr = infos["final_observation"]
                for i in np.nonzero(time_limit_truncations)[0]:
                    if fobs_arr[i] is None:
                        raise RuntimeError("time-limit truncation missing final_observation entry")
                    final_obs_buf[step, i] = torch.as_tensor(fobs_arr[i], dtype=torch.float32, device=device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            # v43 TIME-LIMIT BOOTSTRAP: value the cut-off observations of TRUNCATED steps in one batched
            # pass. final_vals[t] = V(final_obs) (scalar, soft value) and final_probs[t] = Z(final_obs)
            # (distribution) feed the GAE and the categorical λ-return so a time limit bootstraps instead of
            # being treated as a true terminal. final_obs is already normalized (it passed the obs wrappers).
            truncs_b = truncs.bool()
            final_vals = torch.zeros((args.num_steps, args.num_envs), device=device)
            final_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins), device=device)
            final_ents = torch.zeros((args.num_steps, args.num_envs), device=device)
            if truncs_b.any():
                _, _, _, f_entropy, f_logits = agent.get_action_and_value(final_obs_buf[truncs_b])
                f_probs = torch.softmax(f_logits, dim=-1)
                final_probs[truncs_b] = f_probs
                final_vals[truncs_b] = (f_probs * support).sum(dim=-1)
                final_ents[truncs_b] = f_entropy
            # v41 SAC SOFT VALUE IN THE C51 CRITIC. The future entropy is a single-counted, denoised LABEL
            # inside the value: soft_rewards[t] = rewards[t] + (alpha/return_std)*H(s_{t+1})*nonterminal.
            # This soft reward feeds BOTH the scalar GAE (-> policy advantage, auto-baselined by V_soft = E[Z])
            # AND the distributional lambda-return (-> critic target), so the critic LEARNS V_soft and the
            # advantage is the coherent soft GAE. No separate V_ent, no centering, no scale-match: the
            # critic denoises the single-sample logprob by regression, exactly as SAC's soft-Q averages
            # next_state_log_pi over replay. SAC's decomposition: future entropy in the value (here), current
            # entropy in the analytic actor term (-alpha_eff*H(s_t), in the update loop).
            if auto_alpha:
                # Estimate H_pi(s_T) for the bootstrap state and value the same state.
                _, _, _, boot_entropy, boot_logits = agent.get_action_and_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()                              # RAW (SAC) temperature
                return_std = get_return_std(envs, args.return_std_floor)        # per-step reward normalizer
                alpha_eff = alpha_r / return_std
            else:
                boot_logits = agent.get_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
            # SOFT REWARD: fold the next-state entropy into the reward, gated at episode boundaries by
            # nonterminal (truncates the entropy return at `done` AND fixes v40's cross-episode +1-shift leak).
            if auto_alpha and args.soft_adv:
                ent_bonus = torch.zeros_like(rewards)
                # True terminals get no entropy bootstrap. Time-limit truncations use H_pi(final_obs),
                # matching the value bootstrap; ordinary continuing steps use H_pi(s_{t+1}).
                ent_bonus[:-1] = torch.where(
                    truncs[:-1].bool(),
                    final_ents[:-1],
                    entropies[1:] * (1.0 - dones[1:]),
                )
                ent_bonus[-1] = torch.where(
                    truncs[-1].bool(),
                    final_ents[-1],
                    boot_entropy * (1.0 - next_done),
                )
                ent_bonus_raw = ent_bonus
                if soft_entropy_min is not None:
                    ent_bonus = ent_bonus.clamp_min(soft_entropy_min)
                ent_bonus_clip_frac = (ent_bonus_raw < ent_bonus).float().mean().item()
                # SCALING (v42). Fold entropy into the reward on the REWARD'S OWN SCALE: alpha_eff = alpha/return_std.
                # The rewards stored here are already NormalizeReward-normalized (divided by the running return std),
                # so dividing the RAW entropy bonus by the same return_std puts H and r in ONE common frame --
                # entropy becomes a controlled fraction (~alpha) of the reward per step, balanced for ANY env. This
                # fixes v41's two coupled bugs: (1) v41 divided by ent_level = mean(discounted entropy return), a
                # SEPARATE scale, so where return_std >> ent_level (HalfCheetah: 95 vs ~6) the bonus was weighted ~15x
                # the reward and the soft reward became entropy-dominated; (2) ent_level -> 0 as H -> target 0, a
                # SINGULARITY that made the entropy weight explode exactly at the target (HC ent_bonus_frac 0.04 -> 5.5,
                # return stalled ~2000). With /return_std the entropy value level is ~alpha*H_bar/(1-gamma)/return_std
                # and VANISHES smoothly as H -> 0 (no singularity). SUPPORT SAFETY: the discounted entropy return
                # amplifies per-step H by ~1/(1-g) (~100x); alpha*H_bar is self-limiting (the dual lowers alpha when H
                # is high) and return_std_floor=20 bounds the EARLY transient (small std, large H). At steady state
                # return_std~85 matches the reward scale. Watch debug/edge_mass; if it climbs, lower alpha_max.
                # v43: the gamma factor places H(s_{t+1}) one discount deep, matching SAC's backup
                # r + gamma*(Q(s',a') - alpha*logpi(a'|s')) -- the entropy term lives INSIDE gamma.
                soft_rewards = rewards + args.gamma * alpha_eff * ent_bonus
            else:
                soft_rewards = rewards
                ent_bonus_clip_frac = 0.0
            # SOFT GAE: critic-consistent advantage from the soft reward + soft value (values = E[Z] = V_soft).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # v43 TIME-LIMIT: continuing -> bootstrap V(s_{t+1}); truncated -> bootstrap V(final_obs)
                # (truncs[t] gates final_vals[t], which is 0 elsewhere); terminated -> 0 (nextnonterminal=0,
                # not truncated). The lambda recursion still resets at ANY boundary via nextnonterminal.
                boot = nextvalues * nextnonterminal + final_vals[t] * truncs[t]
                delta = soft_rewards[t] + args.gamma * boot - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            policy_adv = advantages          # the soft GAE IS the policy advantage (single coherent channel)
            # v44 diag: decompose the soft advantage into reward vs entropy contributions. Rerun the SAME
            # GAE recursion on the REWARD only (same value baseline); since GAE is linear and the gamma*V
            # terms cancel in the difference, A_soft - A_rew is EXACTLY the entropy contribution. Report the
            # spread (std) of each -- shows how large the entropy signal is relative to the reward signal.
            if args.diag and auto_alpha and args.soft_adv:
                rew_adv = torch.zeros_like(rewards)
                _lg = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nnt, nv = 1.0 - next_done, next_value
                    else:
                        nnt, nv = 1.0 - dones[t + 1], values[t + 1]
                    boot_r = nv * nnt + final_vals[t] * truncs[t]
                    d = rewards[t] + args.gamma * boot_r - values[t]
                    rew_adv[t] = _lg = d + args.gamma * args.gae_lambda * nnt * _lg
                rew_ret_std = rew_adv.std().item()
                ent_ret_std = (advantages - rew_adv).std().item()
            else:
                rew_ret_std = ent_ret_std = 0.0
            # Critic target: SOFT reward λ-return -- entropy folded in and DENOISED by the categorical critic.
            # alpha_eff (=alpha/return_std) + alpha_max keep the soft return inside the support; watch edge_mass.
            target_probs = distributional_lambda_returns(
                soft_rewards, dones, next_done, truncs, final_probs, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
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
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        direct_entropy_norm = torch.ones((), device=device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            if args.norm_adv_center:
                b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
            else:
                b_policy_adv_normed = b_policy_adv / (b_policy_adv.std() + 1e-8)
            if auto_alpha and args.direct_entropy and args.scale_direct_entropy_by_adv:
                direct_entropy_norm = b_policy_adv.std().detach().clamp_min(1e-6)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        clipfracs_low = []
        clipfracs_high = []
        direct_entropy_norms = []
        direct_entropy_coefs = []
        old_approx_kl = approx_kl = torch.zeros((), device=device)
        pg_loss = v_loss = entropy_loss = torch.zeros((), device=device)
        alpha_loss_log = alpha_grad_log = torch.zeros((), device=device)
        actor_gn = critic_gn = torch.zeros((), device=device)

        def _step_dual(entropy_mean):
            # SAC dual in log-alpha parameterization. This is equivalent to
            # -exp(log_alpha) * (log_pi + target_entropy) with H = -log_pi.
            # Gradient on log_alpha is alpha * (H - target): when H is above
            # target alpha falls; when H is below target alpha rises.
            entropy_error = entropy_mean.detach() - target_entropy
            alpha_loss = log_alpha.exp() * entropy_error
            alpha_grad = log_alpha.exp().detach() * entropy_error
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            with torch.no_grad():
                log_alpha.clamp_(min=float(np.log(args.alpha_min)), max=float(np.log(args.alpha_max)))
            return alpha_loss.detach(), alpha_grad.detach()

        stop_update = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            epoch_entropy_sum = torch.zeros((), device=device)
            epoch_entropy_n = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                mb_direct_entropy_norm = direct_entropy_norm
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
                        if auto_alpha and args.direct_entropy and args.scale_direct_entropy_by_adv:
                            mb_direct_entropy_norm = mb_raw_adv.std().detach().clamp_min(1e-6)
                            direct_entropy_norms.append(mb_direct_entropy_norm)
                        if args.norm_adv_center:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        else:
                            mb_advantages = mb_advantages / (mb_advantages.std() + 1e-8)

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
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                with torch.no_grad():
                    clipfracs_low += [(ratio < 1.0 - args.clip_coef).float().mean().detach()]
                    clipfracs_high += [(ratio > 1.0 + clip_hi).float().mean().detach()]
                    clipfracs += [
                        ((ratio < 1.0 - args.clip_coef) | (ratio > 1.0 + clip_hi)).float().mean().detach()
                    ]

                entropy_loss = entropy.mean()  # logging and alpha update
                if auto_alpha:
                    # Accumulate the fresh current-policy entropy so the dual can be stepped once per epoch.
                    epoch_entropy_sum += entropy_loss.detach()
                    epoch_entropy_n += 1
                    if args.dual_freq == "minibatch":
                        alpha_loss_log, alpha_grad_log = _step_dual(entropy_loss.detach())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                    break

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # v41 DIRECT analytic actor entropy term = SAC's current-state -alpha*logpi (current
                # entropy CARRIES GRAD; alpha is DETACHED). Future entropy is already in the soft value
                # (soft_rewards); this adds the current-state entropy maximization -- SAC's exact split.
                # This is a PER-STEP quantity (a loss addend, no support constraint), so it uses the per-step
                # reward normalizer return_std (NOT the return-LEVEL ent_level used for the in-critic bonus).
                if auto_alpha and args.direct_entropy:
                    direct_entropy_coef = (alpha_r / return_std) / mb_direct_entropy_norm
                    direct_entropy_coefs.append(direct_entropy_coef.detach())
                    pg_loss = pg_loss - direct_entropy_coef * entropy.mean()

                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

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
                    pg_loss.backward()
                    if args.diag and args.actor_dist == "gaussian":
                        mean_head_gn = _grad_norm(mean_head_params)      # pre-clip; last-mb value logged
                        logvar_head_gn = _grad_norm(logvar_head_params)
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    if args.diag and args.actor_dist == "gaussian":
                        mean_head_gn = _grad_norm(mean_head_params)      # pre-clip; last-mb value logged
                        logvar_head_gn = _grad_norm(logvar_head_params)
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if auto_alpha and args.dual_freq == "epoch" and epoch_entropy_n > 0:
                # ONE dual step per epoch on the epoch-mean fresh entropy: 10x/rollout, not 320x.
                # Far fewer same-sign Adam steps during the one-sided pre-bind transient => alpha
                # survives (decays gently from init toward the 0.02 floor) instead of railing dead,
                # and turns around to equilibrate once H crosses target.
                alpha_loss_log, alpha_grad_log = _step_dual(epoch_entropy_sum / epoch_entropy_n)

            if stop_update:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0. CRITICAL for v41 -- the soft
        # return adds the entropy bonus, so if alpha_eff*sum(H) pushes returns past the support this climbs.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()
        # How big is the entropy contribution to the soft return, relative to the reward return? (the
        # in-critic analog of the old soft_adv_std_ratio; should be a meaningful but bounded fraction.)
        with torch.no_grad():
            if auto_alpha:
                alpha_eff_log = float(alpha_eff)
                ent_bonus_frac = (
                    ((alpha_eff * ent_bonus).abs().mean() / (rewards.abs().mean() + 1e-8)).item()
                    if args.soft_adv
                    else 0.0
                )
                if args.direct_entropy:
                    fallback_direct_coef = (alpha_eff / direct_entropy_norm).detach()
                    direct_entropy_coef_log = (
                        torch.stack(direct_entropy_coefs).mean().item()
                        if direct_entropy_coefs
                        else fallback_direct_coef.item()
                    )
                    direct_entropy_norm_log = (
                        torch.stack(direct_entropy_norms).mean().item()
                        if direct_entropy_norms
                        else direct_entropy_norm.item()
                    )
                else:
                    direct_entropy_coef_log, direct_entropy_norm_log = 0.0, 1.0
            else:
                ent_bonus_frac, alpha_eff_log = 0.0, 0.0
                direct_entropy_coef_log, direct_entropy_norm_log = 0.0, 1.0

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)          # current RAW temperature
            writer.add_scalar("losses/alpha_loss", alpha_loss_log.item(), global_step)
            writer.add_scalar("losses/alpha_grad_log_alpha", alpha_grad_log.item(), global_step)
            writer.add_scalar("losses/alpha_rollout", alpha_r.item(), global_step)          # rollout-frozen RAW alpha
            writer.add_scalar("losses/alpha_eff", alpha_eff_log, global_step)               # alpha_r / return_std
            writer.add_scalar("debug/return_std", return_std, global_step)                  # NormalizeReward divisor
            writer.add_scalar("debug/direct_entropy_coef", direct_entropy_coef_log, global_step)
            writer.add_scalar("debug/direct_entropy_norm", direct_entropy_norm_log, global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            writer.add_scalar("debug/soft_entropy_min", soft_entropy_min if soft_entropy_min is not None else 0.0, global_step)
            writer.add_scalar("debug/soft_entropy_clip_frac", ent_bonus_clip_frac, global_step)
            writer.add_scalar("debug/rollout_state_entropy", entropies.mean().item(), global_step)
            writer.add_scalar("debug/rollout_action_nll", (-logprobs).mean().item(), global_step)
            # entropy's share of the soft return; pairs with edge_mass to confirm the bonus is meaningful
            # yet support-safe.
            writer.add_scalar("debug/ent_bonus_frac", ent_bonus_frac, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().item() if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/clipfrac_low", torch.stack(clipfracs_low).mean().item() if clipfracs_low else 0.0, global_step)
        writer.add_scalar("losses/clipfrac_high", torch.stack(clipfracs_high).mean().item() if clipfracs_high else 0.0, global_step)
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
        # v43 diagnostic: realized POLICY std (distinct from b_sigma = value-dist std). Disambiguates
        # "negative squashed entropy from small sigma" vs "from mu saturation": if policy_sigma_mean rides
        # near the floor exp(0.5*logvar_min)=0.018 (and floor_frac climbs), the bound is binding and a
        # collapse is real; if it sits at a healthy spread, the negative H is mu saturation, not sigma.
        if args.actor_dist == "gaussian":
            with torch.no_grad():
                _af, _ = agent._trunks(b_obs)
                _pdist, _, _ = agent._actor_dist(_af)
                _psig = _pdist.scale
                _pmu = _pdist.loc
            writer.add_scalar("debug/policy_sigma_mean", _psig.mean().item(), global_step)
            writer.add_scalar("debug/policy_sigma_p05", _psig.flatten().quantile(0.05).item(), global_step)
            writer.add_scalar("debug/policy_sigma_floor_frac",
                              (_psig < 1.1 * float(np.exp(0.5 * args.logvar_min))).float().mean().item(), global_step)
            if args.diag:
                # mean/std min-max across the batch: bang-bang (mu saturation) shows as |mu|_max large
                # with sigma collapsing; healthy exploration keeps sigma_max well above the floor.
                writer.add_scalar("debug/policy_mu_min", _pmu.min().item(), global_step)
                writer.add_scalar("debug/policy_mu_max", _pmu.max().item(), global_step)
                writer.add_scalar("debug/policy_mu_absmean", _pmu.abs().mean().item(), global_step)
                writer.add_scalar("debug/policy_sigma_min", _psig.min().item(), global_step)
                writer.add_scalar("debug/policy_sigma_max", _psig.max().item(), global_step)
        elif args.actor_dist == "beta" and args.diag:
            with torch.no_grad():
                raw_alpha, raw_beta, log_shift_alpha, log_shift_beta, beta_alpha, beta_beta = agent.get_beta_stats(b_obs)
                alpha_hi = raw_alpha >= args.beta_log_shift_max
                beta_hi = raw_beta >= args.beta_log_shift_max
                alpha_lo = raw_alpha <= args.beta_log_shift_min
                beta_lo = raw_beta <= args.beta_log_shift_min
            writer.add_scalar("debug/beta_alpha_mean", beta_alpha.mean().item(), global_step)
            writer.add_scalar("debug/beta_beta_mean", beta_beta.mean().item(), global_step)
            writer.add_scalar("debug/beta_alpha_min", beta_alpha.min().item(), global_step)
            writer.add_scalar("debug/beta_beta_min", beta_beta.min().item(), global_step)
            writer.add_scalar("debug/beta_alpha_max", beta_alpha.max().item(), global_step)
            writer.add_scalar("debug/beta_beta_max", beta_beta.max().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_alpha_mean", log_shift_alpha.mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_beta_mean", log_shift_beta.mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_alpha_hi_frac", alpha_hi.float().mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_beta_hi_frac", beta_hi.float().mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_sample_hi_frac", (alpha_hi | beta_hi).any(dim=1).float().mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_alpha_lo_frac", alpha_lo.float().mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_beta_lo_frac", beta_lo.float().mean().item(), global_step)
            writer.add_scalar("debug/beta_log_shift_sample_lo_frac", (alpha_lo | beta_lo).any(dim=1).float().mean().item(), global_step)
        if args.diag:
            # raw soft advantage moments (pre-shaping; A_soft IS b_advantages with adv_transform="v10").
            writer.add_scalar("debug/A_soft_mean", b_advantages.mean().item(), global_step)
            writer.add_scalar("debug/A_soft_std", b_advantages.std().item(), global_step)
            writer.add_scalar("debug/A_soft_absmax", b_advantages.abs().max().item(), global_step)
            # reward-return vs entropy-return spread (the soft advantage decomposition).
            writer.add_scalar("debug/rew_ret_std", rew_ret_std, global_step)
            writer.add_scalar("debug/ent_ret_std", ent_ret_std, global_step)
            if args.actor_dist == "gaussian":
                # pre-tanh |z| quantiles: large |z| => tanh saturated (bang-bang); the squashing log-det
                # blows up and squashed entropy goes negative even with a healthy base sigma.
                _absz = b_latent_zs.abs().flatten()
                writer.add_scalar("debug/pretanh_absz_p50", _absz.quantile(0.50).item(), global_step)
                writer.add_scalar("debug/pretanh_absz_p90", _absz.quantile(0.90).item(), global_step)
                writer.add_scalar("debug/pretanh_absz_p99", _absz.quantile(0.99).item(), global_step)
                writer.add_scalar("debug/mean_head_grad_norm", float(mean_head_gn), global_step)
                writer.add_scalar("debug/logvar_head_grad_norm", float(logvar_head_gn), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
