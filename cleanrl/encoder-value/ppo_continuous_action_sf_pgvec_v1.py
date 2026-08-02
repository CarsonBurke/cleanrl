# PER-ACTUATOR POLICY GRADIENT FROM THE VECTOR RESIDUAL (pgvec_v1).
# Parent: lejepa_sf_v3.py at defaults -- the same parent as cla_v1 and sf_gadv_v1.
#
# WHERE THE COLLAPSE ACTUALLY IS. This family widened the CRITIC to a 62-dim TD(lambda)
# residual E_t and then funnelled it back through a 1-dimensional aperture before the policy
# ever saw it. The policy loss is
#
#     pg = max( -A_t * r_t , -A_t * clip(r_t) ),   r_t = exp(logpi_new(a_t) - logpi_old(a_t))
#
# and BOTH factors are scalars. The entire channel from critic to policy is ONE number per
# sample multiplying ONE direction, grad log pi(a_t|s_t). Rank-1 credit. Vector routing
# (A = E.w_r) and a state-conditional readout (A = E.g(s)) both produce a BETTER scalar; the
# interface is untouched. That is why every one of them was self-limiting.
#
# THE BOTTLENECK IS `.sum(1)`, NOT `@ w_r`. grad log pi is the only surface an advantage can
# pair with, so the rank of the credit signal is bounded by the number of independent such
# directions supplied -- and there is exactly one, because the parent's line 894 computes
# `(dist.log_prob(z) - log_det_fn(z)).sum(1)` and collapses the per-actuator log-probs BEFORE
# the ratio is formed. The Beta policy factorizes exactly across action dimensions. Splitting
# that sum is what gives the vector residual somewhere to land.
#
# THE CHANGE.
#
#   l_k(t)  = E_t . d_k / res_std               act_dim readouts of the SAME vector residual
#   zhat_k  = (z_k - E_pi_old[z_k]) / sd_pi_old[z_k]        executed action's standardized deviation
#   rho_k   = shrink * ( l_k*zhat_k - mean_j( l_j*zhat_j ) )  mean-zero ACROSS ACTUATORS, exactly
#   A^(k)   = A_t + rho_k                                  => mean_k A^(k) == A_t, exactly
#   r_k     = exp( logpi_k^new(z_k) - logpi_k^old(z_k) )
#   pg      = sum_k max( -A^(k) r_k , -A^(k) clip_k(r_k) )
#
# The policy now consumes act_dim = 6 numbers per sample instead of 1, each reading the
# successor-feature residual along a DIFFERENT direction. A scalar critic structurally cannot
# do this: it has one coordinate, so all d_k would necessarily be parallel.
#
# WHY ADDITIVE AND MEAN-CENTERED. The regression identifies a SIGNED quantity: rho_k > 0 means
# actuator k's deviation predicted a HIGHER return. A first draft used A_t*(1 + rho_k), which
# has the right sign only where A_t > 0 -- and norm_adv makes exactly half the batch negative,
# so on that half it PUNISHED the actuator the model said helped. Measured against the
# identified decomposition: corr(applied, identified) = +0.80 on A>0, -0.80 on A<0, -0.003
# overall. The multiplicative form carried ZERO net signal. Every property below holds for the
# additive form too.
#   * CONSERVATION. mean_k A^(k) == A_t exactly, by construction, for any d, because rho is
#     mean-zero across actuators. The split cannot change the TOTAL credit at a state, only its
#     DISTRIBUTION. Total advantage scale, the norm_adv statistics and `returns` are untouched.
#   * PPO-EXACT AT INIT. d's output layer is zero-init, so rho == 0 and A^(k) == A_t for every
#     k. In the unclipped region grad(sum_k -A r_k) = -A sum_k grad log pi_k = -A grad log pi:
#     BIT-FOR-BIT the parent's policy gradient. (Note A_t, not A_t/act_dim -- dividing would
#     have shrunk the policy gradient by a factor of act_dim, since sum_k r_k is an arithmetic
#     mean where the joint ratio is a product.)
#   * IT IS UNIT-MATCHED. rho is fit against the MC advantage divided by its own spread, and is
#     added to the batch-STANDARDIZED policy advantage, so both are unit-variance. Asserted at
#     startup (norm_adv on, ret_percnorm off) rather than assumed.
#
# WHAT IDENTIFIES d -- the part that makes this a measurement rather than a free reweighting.
# An unconstrained per-actuator split is degenerate: any rho summing to zero is admissible.
# The identifying signal is that the realized advantage should be explained by the actuators
# that actually DEVIATED from the policy mean in a direction that paid off. So d is fit, once
# per iteration and outside the PPO loop, by CLOSED-FORM RIDGE
#
#     min_d  E_t [ ( sum_k l_k(t) * zhat_k(t)  -  Ahat_MC(t) )^2 ],   Ahat_MC = (mc_ret - V)/sd
#
# on the same truncated-MC subset the parent's EV gate already computes. The model is linear in
# d, so this is one act_dim*sf_dim = 372-unknown least-squares system -- solved, not gradient-fit,
# for exactly the reasons w_r is (no LR, no cold start, optimal every iteration). It also removes
# a MEASURED failure: an SGD-fit MLP head equilibrates at rho_absmean ~ 0.19-0.27 against a
# PURE-NOISE target at every LR that learns anything, i.e. Adam's noise ball manufactures a credit
# split out of nothing. A 372-parameter ridge over ~14k rows has no such ball.
#
# Under a factorized policy the zhat_k are independent regressors, so the problem is well-posed
# with a UNIQUE optimum -- l_k is the local sensitivity of the realized return to actuator k's
# deviation at s_t. If the return at a state is insensitive to actuator k, l_k goes to zero
# and that actuator's advantage reverts to A_t. Nothing here is free.
#
# WHAT THIS IS NOT. It is not per-actuator VALUE estimation (no act_dim critics), and it is
# not an action-dependent baseline: A^(k) is the parent's advantage PLUS a mean-zero
# redistribution, shrunk by how well it reproduces on data it was not fit on.
#
# THE SHRINKAGE IS SPLIT-HALF RELIABILITY, NOT EV -- and getting this wrong would have made the
# run vacuous. A first draft used shrink = clamp(held-out EV of the MC fit, 0, 1). That statistic
# is capped near the ATTRIBUTABLE-VARIANCE FRACTION of a single-sample 500-step MC return, which
# is a few percent; measured on a replica at production sizes, held-out EV only crosses zero when
# ~48% of the target's variance is attributable to the action. It would have read ~-0.7 all run,
# pinned shrink at 0, and delivered 8M steps of the CONTROL while logging it as a falsification.
# The right question about rho is not how much of the RETURN it explains but how much of ITSELF
# REPRODUCES. So two independent ridge solves are taken on disjoint halves of the training rows
# and their rho predictions correlated on rows neither saw: a reliability ratio, the optimal
# shrinkage for a noisy estimator, and NOT capped by the MC noise floor. EMA'd across iterations
# because mc_mask keeps 27-49% of rows depending on where the rollout window lands relative to
# the 1000-step episode, which would otherwise oscillate the shrinkage deterministically.
#
# THE TRUST REGION IS HELD FIXED, so the arm is ONE variable. Clipping becomes per-actuator -- a
# dimension that moved too far is zeroed while the other five stay live -- but at the act_dim-th
# ROOT of the parent's bounds, so the PRODUCT of the per-dim ratios (which is exactly the parent's
# joint ratio) is confined to exactly the parent's region. Clipping each dimension at the raw
# clip_coef instead was measured to widen the joint log-ratio ~6x and inflate joint approx_kl 1.7x
# on a real Beta actor; with target_kl=0.03 live that changes the NUMBER OF EPOCHS per iteration,
# so the arm would have differed in optimizer steps as well as in credit assignment.
# --no-pgvec-clip-match is the loose-region arm, deliberately not the default.
#
# NULL / KILL SIGNALS, cheapest first:
#   pgvec/split_r ~ 0                the credit split does not REPRODUCE across independent
#                                    halves: it is noise, shrink drives rho to 0, and this run
#                                    IS the parent. The cleanest possible falsification, and
#                                    readable within the first few iterations.
#   pgvec/rho_absmean_raw ~ 0        the solve found no per-actuator structure at all.
#   raw large with split_r ~ 0       structure that does not generalize.
#   pgvec/ev_mc                      DIAGNOSTIC ONLY. Small or negative is the MC target's noise
#                                    floor, not evidence about the split. Do not gate on it.
#   losses/approx_kl inflating vs the parent -- the per-dim trust region, not the credit split.
#
# HYPOTHESIS. Per-actuator credit reduces the variance of each component of the policy
# gradient (actuator k stops being scored on return variation caused by the other five), and
# per-actuator clipping stops one saturated dimension from throttling all six. Target: the
# family best, 10,071 +/-67 @8M on HalfCheetah-v4.
#
# --no-pgvec recovers the parent EXACTLY: nothing is solved and the scalar ratio/clip path is
# restored. There is no module and no optimizer anywhere in this file, so nothing can leak into
# PPO's clip budgets and no RNG is consumed -- both arms start from identical weights by
# construction rather than by a save/restore dance.
# =====================================================================================
# PPO + successor features, v3: MAKE THE ENCODER EARN ITS PLACE.
#
# Two levers, independently flagged, on top of the current best configuration.
#
# WHY THIS FILE EXISTS. Across this family the encoder has been dead weight. The decisive
# measurement is sf_noe_v1, which DELETES the encoder, the predictor and SIGReg outright
# (phi = [s, a, a*a, 1]) and beat the full-encoder v2 at every checkpoint:
#
#   step         500k     1M    1.5M     2M
#   noe (no e)   1907   3822    5093   6292
#   v2 (with e)  1738   3562    4915   5869
#
# So the family's gains came from SUCCESSOR FEATURES, not from the learned latent. v2's own
# header already predicted why: at K=1 the prediction loss is satisfiable by a whitened
# near-linear map of s, because one-step MuJoCo prediction is close to trivial. A latent
# under no real pressure is a latent that adds parameters and noise.
#
# Both levers attack exactly that.
#
#   LEVER 1 (pred_horizon K>1) -- train the predictor AUTOREGRESSIVELY. Round k eats round
#   k-1's own output, so error compounds and one-step memorization stops being sufficient.
#   This is also the hard prerequisite IDEAS.md records for idea C, which needs a rollable
#   model to differentiate: d(e_{t+k})/d(a_t) is meaningless if the predictor was only ever
#   trained at k=1.
#
#   LEVER 2 (hist_len H>1) -- e_t = f(s_{t-H+1..t}). HalfCheetah's observation is Markov, so
#   history is NOT free information; the specific thing it buys is ACCELERATION. The reward
#   uses frame-skip-averaged velocity (x_{t+1}-x_t)/dt while obs carries instantaneous
#   qvel[0], and that gap is a finite difference -- not a function of s_t alone. It is
#   precisely the residual v2's header blames for structured error in every value estimate.
#
# BASE IS THE WINNING ARM, NOT v2. per_dim_lambda defaults OFF here, on measured evidence:
# vector routing with a UNIFORM lambda (2032/3977/5209 at 500k/1M/1.5M) beat both v2 and the
# per-dimension variant (1559/3407/4444). See per_dim_lambda's note for the mechanism.
#
# WHAT VECTOR ROUTING ACTUALLY IS, now that it has been measured. adv_vec_corr runs 0.97 ->
# 0.99 against scalar GAE, which is the identity below behaving as derived: with a uniform
# lambda, w_r.E is scalar GAE with r_t replaced by the probe's linear reconstruction w_r.phi_t,
# and V = w_r.psi as a correspondingly reconstructed baseline. The advantage is therefore
# GAE ON A DENOISED REWARD -- a 44-dim linear filter that projects out whatever in r_t the
# feature space cannot express. A 2-6% perturbation of the advantage is worth +6 to +17%
# return. That also means it removes REAL signal (the acceleration term again), and lever 2
# is the direct attack on that: put acceleration in the feature space so the filter stops
# discarding it.
# =====================================================================================
# Inherited from the B lineage: five correctness fixes found by review before v1 finished
# (v1 was cancelled at 1.6M; its runs/ dir is kept but its numbers are void):
#   1. w_r is zero-init, so adv_vector was IDENTICALLY ZERO on iteration 1 -> pg_loss == 0
#      and the actor got no gradient from the first 32,768 transitions. Now falls back to
#      scalar GAE until w_r has been solved once.
#   2. The tau estimator was ungated. At tau_stat_count == 1 the warmup rate is exactly
#      1.0, so a covariance off a handful of autocorrelated samples could overwrite every
#      mixing time and persist ~20 iterations through the EMA -- into the critic target.
#      Now gated at n_mc >= 256, with the warmup counter inside the guard.
#   3. Degenerate (deterministic) coordinates fell back to gae_lambda. Their discounted
#      sum is deterministic, so MC is exact: they want lambda = 1, and this is the exact
#      coordinate the header names as the motivating example.
#   4. lam_min was 0.50, pinning tau >= 2 and censoring the near-white action blocks --
#      one of the two blocks whose heterogeneity this variant exists to exploit.
#   5. `returns` became w_r.Lambda under vector_adv (no reward tensor in it at all), so
#      losses/explained_variance silently meant different things in the two arms. It is
#      now built from the scalar GAE unconditionally.
# =====================================================================================
# Base: ppo_continuous_action_lejepa_sf_v2.py (9,612 @6.4M on HalfCheetah-v4, vs the
# scalar-critic base's 8,278 @8M). Two changes, both in the advantage pathway; the
# critic head, the SSL path, the encoder, w_r, and PPO itself are untouched. (The critic
# TARGET does change -- per-dimension lambda -- but the head, loss and masks do not.)
#
# THE OBSERVATION THIS IS BUILT ON. v2 computes a 62-dimensional TD(lambda) residual
# E_t = Lambda_t - psi_0(s_t) for the critic, and SEPARATELY a scalar GAE for the policy.
# Those are not two different objects. The identity
#
#       w_r . E_t  ==  A_t      exactly whenever  w_r . phi_t == r_t
#
# says the scalar GAE was always a projection of the vector residual. v2 just computed
# the projection along its own recursion, with one lambda for everything, and threw the
# other 61 dimensions away.
#
# WHY THAT MATTERS. lambda is not a free knob -- it is an assumption about ONE number,
# the effective credit horizon tau = 1/(1 - gamma*lambda). The default lambda=0.95 at
# gamma=0.99 asserts tau = 16.8 steps (0.84 s) for EVERY coordinate. Measured, that is
# false and not by a little:
#
#   obs block      tau ~ 12     joint angles / velocities under a gait
#   e block        tau ~ 7      SIGReg-whitened, decorrelates faster
#   action blocks  tau ~ 1      nearly white given the state
#   constant       tau -> 100   its discounted sum is DETERMINISTIC given the episode
#                               end, so Monte Carlo is exact and zero-variance for it
#
# Accumulating raw samples past a coordinate's mixing time adds variance with NO bias
# reduction: past that point E[phi_{t+k}|s_t] is just the stationary mean, which the
# critic learns trivially. So the bias/variance trade should be made PER COORDINATE.
# A scalar critic structurally cannot do this -- one target, one lambda. A vector target
# has K mixing times and can carry K lambdas.
#
# THE TWO CHANGES.
#   1. per_dim_lambda: lambda_d = (1 - 1/tau_d)/gamma, with tau_d MEASURED each iteration
#      as std(sum_k gamma^k phi_d) / std(phi_d) -- no AR(1) or exponential-decay
#      assumption, so oscillatory coordinates (joint angles under a periodic gait) cancel
#      correctly in the sum where a lag-1 autocorrelation estimate would be badly wrong.
#      tau is computed from the lambda=1 MC sum, NOT from sf_std, precisely so the
#      estimate is lambda-independent: deriving lambda from the spread of Lambda(lambda)
#      closes a feedback loop whose only stable point is lambda_max.
#      Sanity: tau = 16.8 recovers lambda = 0.95 exactly, so this REDUCES to v2 when the
#      heterogeneity is absent.
#   2. vector_adv: the policy advantage becomes w_r . E_t. Without this, per-dimension
#      lambda would change only the critic target and never reach the policy.
#
# HONEST COSTS, both real:
#   - The projection identity is exact only if the probe is exact. It is not
#     (reward_probe_r2 ~ 0.98), so routing through w_r injects the accumulated probe
#     residual sum_k (gamma*lam)^k eps_{t+k} into the advantage. sf/value_err_frac
#     already tracks that quantity; --no-vector-adv is the control that prices it alone.
#   - lambda_d is fit on the policy-induced distribution and moves with the policy. It is
#     EMA'd (tau_ema) and bounded to [lam_min, lam_max] rather than free.
#
# KILL SIGNAL. advB/adv_vec_corr ~ 1.0 means the vector path reproduced scalar GAE and
# the run is a null experiment. advB/lam_spread ~ 0 means the per-coordinate heterogeneity
# this variant exists to exploit is not there.
#
# EXPERIMENTS. Treatment = defaults. Control = --no-vector-adv (per-dimension lambda on
# the critic target only, policy on scalar GAE) isolates the w_r projection cost from the
# lambda effect. A third arm, --no-per-dim-lambda, is v2's advantage with the vector
# routing and nothing else.
# =====================================================================================
# Base: ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_mbpercnorm_v2.py run with
# --norm-adv --norm-adv-scope batch --no-ret-percnorm (8,278 @8M on HalfCheetah-v4).
# Those three flags are BAKED IN here; everything else is untouched except the critic.
#
# THESIS. Scalar V^pi(s) conflates two objects with different timescales: STRUCTURE
# (where the agent is headed in the MDP -- slow, largely policy-independent) and
# EVALUATION (what that is worth under the current pi and return scale -- moves every
# policy step). GAE bootstraps the evaluation through time, so every actor update
# redefines the multi-step target. The base attacks this only on the CODOMAIN
# (Dreamer3-bucket HL-Gauss pins the range; critic MTP adds horizons) -- neither makes
# the bootstrapped object stationary.
#
#   e_t   = f(s_t)                     encoder, SIGReg-constrained toward N(0, I)
#   phi_t = [ e_t , s_t , a_t , a_t*a_t , 1 ]   reward features; dim = emb_dim + obs_dim
#                                               + 2*act_dim + 1
#   psi_h(s_t) ~ Lambda_{t+h}          CRITIC HEAD on the ThinkTrunk, h = 0..5
#   Lambda_t  = E_pi[ sum_k gamma^k phi_{t+k} ]   successor features -- the ONLY
#                                                 bootstrapped object, vector-valued
#   V(s)  = w_r . psi_0(s)             evaluation, a dot product
#   w_r   solved from w_r.phi_t ~ r_t  IMMEDIATE reward only, closed-form ridge,
#                                      no bootstrapping, recomputed every iteration
#
# The only scalar regression left anywhere is w_r against immediate reward. Everything
# temporal happens in a space whose marginal is externally clamped by SIGReg.
#
# WHY REPLACE THE HEAD, NOT ADD A PATHWAY. share_backbone=True: actor and critic fully
# share the ThinkTrunk. Deleting the critic would strip ~half the trunk's gradient, and
# bolting SF on in parallel would have psi read a near-raw-obs latent while a scalar
# critic reads the policy-shaped trunk -- a regression then would not be attributable to
# "SF vs scalar" at all. So the swap is surgical: same head shape, same MTP semantics,
# same boundary masks; only K goes from 511 bucket logits to emb_dim + obs_dim +
# 2*act_dim + 1 = 62 SF dimensions (HalfCheetah: 32 + 17 + 12 + 1), and the loss from
# cross-entropy to masked MSE on standardized Lambda.
# It also deletes the 402MB CPU target_probs tensor and its ~4GB/iteration H2D traffic.
#
# HONEST ACCOUNTING (the strong stationarity claim does NOT survive):
#   - psi is still policy-dependent; its fixed point moves with pi exactly as V^pi does.
#     What changes is that the bootstrap target's MARGINAL is pinned, not that
#     pi-dependence is gone.
#   - w_r is only approximately policy-independent -- it is fit over the policy-induced
#     distribution, which is pi-invariant only if the regression were well-specified.
#   - a*a is the MOST policy-dependent feature here: with a Beta actor
#     E_pi[a*a|s] = Var_pi + mean^2, so that block tracks policy entropy directly.
#     Correct (ctrl cost genuinely depends on pi) but the vector is not uniformly stable.
#   - SIGReg pins the DISTRIBUTION, not the coordinate FRAME. N(0,I) is rotation-
#     invariant and the prediction loss cannot pin the frame either (the predictor
#     co-rotates), while psi_old and w_r are functions of coordinates. This is a NEW
#     non-stationarity the design introduces. Measured (ssl/frame_drift_*), not
#     pre-emptively patched with an EMA.
#
# SSL PATH. Encoder + causal action-conditioned predictor + SIGReg, in a SEPARATE
# top-level module with its own AdamW, fully detached from the trunk and from w_r. Its
# only job is to define what e means. Exactly two loss terms, LeJEPA-faithful:
# pred MSE with BOTH branches attached + lambda * SIGReg. No EMA teacher, no stop-grad,
# no asymmetry -- SIGReg is what prevents collapse, and a teacher would re-add the second
# unanchored timescale this design exists to remove. Runs ONCE PER ITERATION, outside the
# 320-minibatch PPO loop (+10-20% wall clock; inside would be +100-200%).
#
# INSTRUMENTATION. The obvious EV metric is rigged: b_returns = advantages + values
# contains the critic's own bootstrapped values, and at dt=0.05 errors are strongly
# state-autocorrelated, so a critic scores well against a target built from itself.
# gate/* instead scores three predictors against a common TRUNCATED-MC return:
#   (1) ev_trunk_probe  detached scalar probe on sg(trunk_feat)  -- what a scalar critic
#                       on this trunk would achieve (replaces the lost HL-Gauss comparator)
#   (2) ev_sf_online    w_r.psi_0, the value the actor actually consumed -- the treatment
#   (3) ev_latent_cap   closed-form ridge of MC return onto sg(e) -- how much reward
#                       signal is LINEARLY present in the instantaneous embedding
# (3) is NOT a ceiling on (2): psi reads the trunk, not e_t, and accumulates phi over
# time, so (2) > (3) means the successor-feature construction is buying something a
# linear readout of e_t cannot. The kill signal is (2) << (1) sustained -- the SF critic
# losing to what a plain scalar critic on the same trunk would achieve.
#
# HYPOTHESES. H1 latent-path loss stays smooth across policy shifts where scalar-critic
# EV spikes. H2 latent occupancy captures gait/contact structure a scalar critic lacks.
# H3 (the falsifier) EV(w_r.psi_0) vs a scalar critic's EV on a COMMON target.
#
# SCOPING. At k=1 the predictor only keeps e dynamics-grounded; one-step prediction in
# MuJoCo is near-trivial, so a whitened near-linear map of s can satisfy it. If v1 works,
# do NOT credit the JEPA predictor for it -- k=1 makes this a clean test of the
# SUCCESSOR-FEATURE claim. Multi-horizon k and history-dependent embeddings are the v2
# levers where H2 actually gets tested.
#
# Naming: `z`/`latent_zs` is the Beta actor's native action sample (inherited). The
# encoder latent is `e`/`emb` throughout.
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

from cleanrl.shared.lejepa import (
    ActionEncoder,
    ARPredictor,
    CompiledModule,
    MLP,
    SIGReg,
    StateEncoder,
)

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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
    # NOTE: the reference run this variant is built against (exp-name
    # "iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1", 8278 @8M) was NOT a separate
    # file -- it was mbpercnorm_v2 launched with --norm-adv --norm-adv-scope batch
    # --no-ret-percnorm. Those three are baked in as defaults here so the baseline is
    # reproduced by default rather than by remembering flags.
    norm_adv: bool = True            # ppoadvnorm: plain PPO advantage standardization...
    # --- Percentile advantage normalization (disabled: superseded by norm_adv) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
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
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional latent occupancy vector.
    critic_mtp_horizon: int = 6

    # --- LeJEPA successor-feature value pathway -------------------------------------
    emb_dim: int = 32                # d. At hist_len=1 the obs manifold is 17-dim, so e is rank
    #                                  <= 17 regardless and effective rank ~17 is NOT a failure
    #                                  signal. At hist_len>1 that ceiling no longer binds (the
    #                                  encoder sees H*17 inputs), so a RISE in ssl/emb_effective_rank
    #                                  is the direct evidence lever 2 did something.
    ssl_hidden: int = 256            # encoder / projector MLP width
    pred_depth: int = 2              # causal transformer depth
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    # -- v3 LEVER 1: autoregressive multi-horizon prediction -------------------------
    pred_horizon: int = 1            # K. Number of AUTOREGRESSIVE rollout rounds. K=1 is v2
    #                                  (single-step, teacher-forced). K>1 feeds the predictor's
    #                                  OWN output back as its input, so error COMPOUNDS and the
    #                                  encoder is pressured to make e ROLLABLE rather than
    #                                  merely one-step-predictable. The reference (../le-wm)
    #                                  rolls out only at EVAL (jepa.py:61), never in training --
    #                                  there is no implementation to copy for this.
    pred_ctx: int = 3                # context frames before the first predicted frame.
    #                                  seq_len is DERIVED as pred_ctx + pred_horizon, so K=1
    #                                  gives seq_len=4 = LeWM's history_size(3)+num_preds(1).
    # -- v3 LEVER 2: history-dependent embeddings ------------------------------------
    hist_len: int = 1                # H. Observation frames the ENCODER sees: e_t = f(s_{t-H+1..t}).
    #                                  H=1 is v2 (single state; e is rank <= obs_dim by
    #                                  construction). H>1 breaks that ceiling AND -- the concrete
    #                                  mechanism, which matters because HalfCheetah's obs is
    #                                  otherwise MARKOV so history would add nothing -- lets e
    #                                  carry ACCELERATION. The reward uses the frame-skip
    #                                  AVERAGED velocity (x_{t+1}-x_t)/dt while obs carries
    #                                  INSTANTANEOUS qvel[0] at s_t. That gap is an acceleration
    #                                  term: a finite difference, so NOT a function of s_t alone.
    #                                  It is exactly what caps the reward probe's R^2 and gives
    #                                  its residual structure -- and v2's header already blames
    #                                  that structured residual for polluting every value estimate.
    #                                  NOTE phi keeps only the RAW s_t block, deliberately: if
    #                                  lagged obs went into phi as well, a linear w_r could pick
    #                                  up acceleration directly and this would no longer be a
    #                                  test of the ENCODER.
    sigreg_weight: float = 0.09      # lambda. NOTE: the Epps-Pulley statistic scales with batch
    #                                  size, so the reference value does not transfer verbatim.
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256     # exact (statistic is a mean over directions); bounds memory
    ssl_lr: float = 5e-5             # LeWM reference optimizer
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 1024            # sequences per SSL minibatch (SIGReg is ~1.1GB here, ~10GB at 8192)
    ssl_steps_target: int = 64       # SSL optimizer steps per iteration, HELD CONSTANT across K.
    #   Replaces v2's fixed ssl_epochs=8, which would have silently confounded the headline
    #   experiment. Steps per iteration are ssl_epochs * (n_seq // ssl_batch), and
    #   n_seq = (num_steps // seq_len) * num_envs SHRINKS as the horizon grows: at K=1,
    #   seq_len=4 -> 8192 seqs -> 64 steps; at K=4, seq_len=7 -> 4672 seqs -> only 32. The
    #   K=4 arm would have received HALF the encoder gradient (and half the total SIGReg
    #   pull, since SIGReg applies per step), so any regression would be unattributable
    #   between "the AR objective is bad" and "the encoder was trained half as much".
    #   The epoch count is DERIVED from this instead. Note the fix deliberately adjusts
    #   epochs and not ssl_batch: the Epps-Pulley statistic scales with the batch size, so
    #   rescaling ssl_batch would hold steps constant while quietly changing the effective
    #   sigreg_weight -- trading one confound for a subtler one.
    ssl_grad_clip: float = 1.0       # own clip (reference lewm.yaml gradient_clip_val), fully
    #                                  separate from PPO's -- see ssl/grad_norm for the pre-clip value
    sf_alpha: float = 1.0            # 1.0 = pure latent prediction, ZERO scalar regression in the
    #                                  network. <1 mixes in MSE(w_r.psi, w_r.Lambda) -- the fallback
    #                                  if the pure form underfits, not the starting point.
    # --- B: vector-routed advantage + per-dimension lambda ---------------------------
    vector_adv: bool = True          # policy advantage = w_r . E_t (the vector TD(lambda)
    #                                  residual) instead of the separate scalar GAE
    #                                  recursion. Required for per_dim_lambda to reach the
    #                                  policy at all; --no-vector-adv is the control that
    #                                  prices the w_r projection on its own.
    per_dim_lambda: bool = False     # v3 DEFAULT FLIPPED OFF, on measured evidence. Deriving
    #                                  lambda_d from the phi coordinate's mixing time LOST at
    #                                  every checkpoint (500k/1M/1.5M: 1559/3407/4444) against
    #                                  the identical file with this off (2032/3977/5209).
    #                                  tau_d is a STATE-PREDICTABILITY horizon -- how far ahead
    #                                  s_t forecasts phi_{t+k} -- but the credit kernel needs an
    #                                  ACTION-INFLUENCE horizon: how much a_t CAUSED phi_{t+k}.
    #                                  An oscillatory coordinate decorrelates fast while an
    #                                  action perturbation still shifts the whole trajectory, so
    #                                  shortening lambda_d there discards causal credit that was
    #                                  never redundant. Damage is concentrated EARLY (the
    #                                  estimator reads near-white data under a random policy:
    #                                  measured tau_obs=2.2 at iteration 1, climbing to 9 by 2M),
    #                                  and early damage in RL compounds. Kept as a flag, not
    #                                  deleted, because the ESTIMATOR is correct for what it
    #                                  measures -- see IDEAS.md, it is why idea C differentiates
    #                                  the predictor w.r.t. the ACTION instead.
    lam_min: float = 0.0             # floor. NOT a safety margin: a white coordinate's
    #                                  correct lambda IS 0 (past k=0 its conditional mean
    #                                  is the stationary mean, which the critic learns
    #                                  trivially, so MC there is pure variance). A floor of
    #                                  0.5 would pin tau >= 2 and censor the action blocks,
    #                                  i.e. clamp away half the heterogeneity being tested.
    lam_max: float = 0.99            # ceiling: tau -> 1/(1-gamma) coordinates (the constant
    #                                  intercept) would otherwise ask for exactly lambda=1
    tau_ema: float = 0.05            # EMA rate for the per-dimension mixing-time estimate,
    #                                  with the same 1/count warmup as the Lambda standardizer
    pgvec: bool = True               # PER-ACTUATOR policy gradient. OFF restores the parent's
    #                                  scalar ratio/clip path exactly and solves nothing.
    pgvec_ridge: float = 1e-2        # ridge coefficient for the CLOSED-FORM solve of the readout
    #                                  matrix d (act_dim x sf_dim). SOLVED, not gradient-fit, for
    #                                  the same reason w_r is: it removes a learning rate, removes
    #                                  the cold-start phase, and -- decisively here -- removes
    #                                  Adam's noise ball. A gradient-fit MLP head was MEASURED to
    #                                  equilibrate at rho_absmean ~ 0.19-0.27 against a PURE-NOISE
    #                                  target at every LR that learned anything, i.e. it
    #                                  manufactured a credit split out of nothing. A ridge solve
    #                                  on 372 parameters over ~14k rows has no such ball.
    pgvec_val_frac: float = 0.2      # held-out fraction of the MC-masked subset. Used for the
    #                                  REPORTED ev_mc only; the shrinkage comes from split-half
    #                                  reliability (below), not from this.
    pgvec_clip_match: bool = True    # match the PARENT's JOINT trust region exactly by clipping
    #                                  each dimension at (1+c)^(1/act_dim). Measured: per-dim
    #                                  clipping at the raw clip_coef lets the joint log-ratio run
    #                                  ~6x wider and inflates joint approx_kl 1.7x on a real Beta
    #                                  actor -- which, with target_kl=0.03 live, changes the NUMBER
    #                                  OF EPOCHS per iteration and confounds the credit-split
    #                                  measurement with a trust-region change. OFF is the
    #                                  loose-region arm, deliberately not the default.
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    sf_target_ema: float = 0.01      # EMA rate for per-dimension Lambda standardization
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

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
    norm_adv_scope: str = "batch"    # ...standardized once over the whole rollout ("batch")

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- torch.compile ---------------------------------------------------------------
    # reduce-overhead (CUDA graph trees) is NOT usable on the actor/critic path: v_loss and
    # pg_loss share the trunk forward, and the update runs backward twice with
    # retain_graph=True, cloning and re-adding grads in between. Cudagraph outputs live in
    # the graph pool, clip_grad_norm_ mutates them in place, and the second backward replays
    # over live refs -> either "accessing tensor output of CUDAGraphs that has been
    # overwritten" or a re-record per minibatch (x320/iter), which is SLOWER than eager.
    # So the trunk gets inductor fusion without graphs; the SSL nets (separate optimizer,
    # single plain backward) do get reduce-overhead.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # applies to the SSL nets only; see above for why the
    #                                        actor/critic trunk cannot take cudagraphs at all
    compile_ssl_cudagraphs: bool = False   # DEFAULT OFF, and not out of caution: LeJepaSSL.forward
    #   chains encoder -> action_encoder -> predictor, and each cudagraph-wrapped call issues its
    #   own cudagraph_mark_step_begin(), which invalidates the still-pending backward of the
    #   modules called earlier in the SAME forward. It raises "accessing tensor output of
    #   CUDAGraphs that has been overwritten" on the FIRST ssl_loss.backward() -- reproduced --
    #   and suppress_errors does NOT catch it (that only covers dynamo compile-time errors).
    #   Cloning outputs does not help either: the invalidated tensors are the saved intermediates.
    #   The SSL path is ~3% of wall clock, so inductor fusion without graphs costs ~nothing.

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @property
    def seq_len(self) -> int:
        """Derived, not a flag: pred_ctx + pred_horizon.

        A property rather than a field so it cannot drift out of sync with the levers --
        an independent seq_len would silently truncate the AR rollout the moment
        pred_horizon was raised without it. Properties carry no annotation, so this is
        not a dataclass field and tyro does not surface it as a CLI argument.
        """
        return self.pred_ctx + self.pred_horizon


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


class LeJepaSSL(nn.Module):
    """Encoder + causal action-conditioned predictor + SIGReg.

    Trained by EXACTLY two terms: a next-embedding prediction MSE with BOTH branches
    attached (no stop-gradient on the target -- SIGReg is what prevents collapse, and a
    stop-grad or EMA teacher would re-introduce the second unanchored timescale this
    design exists to remove), and SIGReg itself.

    Its only job is to define what the embedding e means. It never sees a reward, and it
    never receives gradient from the value path -- psi and w_r both read sg(e).
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.horizon = args.pred_horizon
        # LEVER 2: the encoder reads H stacked frames. e_t = f(s_{t-H+1..t}).
        self.encoder = StateEncoder(args.hist_len * obs_dim, args.emb_dim, args.ssl_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.predictor = ARPredictor(
            num_frames=args.seq_len,
            depth=args.pred_depth,
            heads=args.pred_heads,
            mlp_dim=args.pred_mlp_dim,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            dim_head=args.pred_dim_head,
            dropout=0.0,
            emb_dropout=0.0,
        )
        self.pred_proj = MLP(args.emb_dim, args.ssl_hidden, args.emb_dim)
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk
        )

    def forward(self, obs_seq, act_seq, cum_cont, sigreg_weight):
        """obs_seq (N,L,H*obs); act_seq (N,L,act); cum_cont (N,L) = cumprod of 1-boundary.

        AUTOREGRESSIVE multi-horizon rollout. Round k consumes round k-1's OUTPUT as its
        input, so prediction error compounds exactly as it would at inference -- which is
        the whole point: a teacher-forced one-step loss is satisfiable by a near-linear
        whitened map of s (one-step MuJoCo prediction is close to trivial), and that is
        why v2's encoder failed to earn its place against the no-encoder ablation.

        THE SHIFT IS WHAT MAKES IT MULTI-STEP, and it is the easy thing to get wrong. A
        causal transformer's output at position l is always "predict l+1 given context
        through l". Feeding predictions back WITHOUT shifting the action conditioning just
        re-predicts l+1 from a noisier context -- still one step, no compounding, and the
        loss curve looks entirely plausible. To reach l+2 the context must ADVANCE: round
        k runs on inputs that stand for frames k..L-1 and actions a[k..L-1], so its output
        at position l predicts emb[l+k+1]. Each round drops one position.

        Density is preserved: EVERY position contributes at EVERY horizon, so raising K
        costs K predictor passes but loses only K-1 supervised pairs per sequence.
        """
        emb = self.encoder(obs_seq)                              # (N,L,d)
        act_emb = self.action_encoder(act_seq)                   # (N,L,d)
        L = emb.shape[1]
        err_terms, mask_terms = [], []
        x = emb                                                  # round 0 input: all TRUE
        for k in range(self.horizon):
            lk = L - k                                           # x is (N, lk, d)
            # pos_offset=k: this round's tokens stand for frames k..L-1, so they must carry
            # those positional embeddings rather than restarting the window at 0.
            out = self.pred_proj(self.predictor(x, act_emb[:, k : k + lk], pos_offset=k))
            n_valid = lk - 1                                     # out[l] ~ emb[l+k+1]
            if n_valid <= 0:
                break
            # Target is ATTACHED on both branches -- no stop-grad, no EMA teacher. SIGReg
            # is what prevents collapse; a stop-grad here would re-add the second
            # unanchored timescale this design exists to remove.
            err_terms.append((out[:, :n_valid] - emb[:, k + 1 : k + 1 + n_valid]).pow(2).mean(-1))
            # Valid iff the whole window obs[0 .. k+1+l] is one episode. cum_cont[j] is the
            # product of cont[0..j], and cont[i]==1 means obs[i]->obs[i+1] is intra-episode,
            # so cum_cont[j]==1 <=> obs[0..j+1] are contiguous. Target frame k+1+l therefore
            # needs cum_cont[k+l], NOT cum_cont[k+1+l]: the latter additionally demands the
            # transition OUT of the target frame be intra-episode, which no prediction
            # depends on, and it silently drops one valid pair per horizon whenever a chunk's
            # exit transition happens to be a reset.
            mask_terms.append(cum_cont[:, k : k + n_valid])
            x = out[:, : lk - 1]                                 # feed predictions forward
        err = torch.cat(err_terms, dim=1)
        mask = torch.cat(mask_terms, dim=1)
        # Masked mean, NOT .mean() over a zeroed tensor (which would divide by the full count).
        # Pooled uniformly across horizons: later horizons carry larger error and so dominate
        # the gradient, which is what "make e rollable" MEANS. Per-horizon values are logged
        # separately (ssl/pred_loss_h*) so compounding is visible rather than inferred.
        pred_loss = (err * mask).sum() / mask.sum().clamp_min(1.0)
        # Stacked as a TENSOR, not floats: this runs 64x per iteration, and a float() per
        # horizon here would add K host syncs to every SSL step and break the compiled graph.
        per_horizon = torch.stack(
            [(e * m).sum() / m.sum().clamp_min(1.0) for e, m in zip(err_terms, mask_terms)]
        ).detach()
        # THE TRANSPOSE IS LOAD-BEARING. SIGReg reduces the empirical characteristic
        # function over dim -3 and scales by size(-2); both resolve to the BATCH only in
        # (T, B, D) layout. Passing (N, L, d) would average the CF over just L samples,
        # silently disabling collapse protection while still logging a plausible number.
        sigreg_loss = self.sigreg(emb.transpose(0, 1))           # (L, N, d)
        return pred_loss + sigreg_weight * sigreg_loss, pred_loss, sigreg_loss, per_horizon


def pgvec_design(resid_flat, res_std, zhat):
    """Regressor matrix for the per-actuator credit solve: X[i, k*sf_dim + d] = zhat_k * E_d.

    The model is  Ahat_MC(t) ~ sum_k l_k(t) * zhat_k(t)  with  l_k = E_t . d_k / res_std,
    which is LINEAR in d -- so the whole thing is one least-squares problem in act_dim*sf_dim
    unknowns and needs no optimizer at all. res_std makes l dimensionless: phi's blocks differ
    by orders of magnitude (a whitened e block, a strictly positive a*a block, an intercept
    whose discounted sum is ~1/(1-gamma)) and an unscaled readout would be dominated by
    whichever coordinate is loudest.

    d is a CONSTANT matrix, not a state-conditional network, and that is not a weakening:
    l_k(t) is still state-dependent through E_t, exactly as V = w_r . psi is state-dependent
    through psi with a constant w_r. What is learned is the READOUT DIRECTION per actuator --
    act_dim different directions into the same vector residual -- which is precisely the
    object a scalar critic cannot have, because it has one coordinate and all d_k would be
    forced parallel.
    """
    e = resid_flat / res_std                                  # (N, sf_dim)
    return (zhat.unsqueeze(-1) * e.unsqueeze(1)).reshape(zhat.shape[0], -1)


def pgvec_solve(X, y, ridge):
    """Closed-form ridge solve, float64, scale-aware -- same form as solve_reward_probe."""
    X64 = X.double()
    gram = X64.T @ X64
    rhs = X64.T @ y.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(X.dtype)


def pgvec_logits(beta, resid_flat, res_std, act_dim):
    """l_k = E_t . d_k / res_std -- act_dim scalar readouts of the SAME vector residual.

    STATE ONLY on the d side: the executed action enters the credit split exactly once, at the
    call site, through the detached zhat deviation. If d itself saw a_t the readout direction
    would be action-dependent and l_k would stop being a state-local sensitivity.
    """
    return (resid_flat / res_std) @ beta.view(act_dim, -1).T   # (N, act_dim)


def pgvec_rho(logits, zhat):
    """rho_k = l_k*zhat_k - mean_j(l_j*zhat_j): the credit REDISTRIBUTION across actuators.

    Mean-zero across the actuator axis by construction, so mean_k (A_t + rho_k) == A_t exactly
    for ANY d -- the split can never change the total credit at a state, only its allocation.
    """
    contrib = logits * zhat
    return contrib - contrib.mean(-1, keepdim=True)


def corr_score(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(
        ((a - a.mean()) * (b - b.mean())).mean()
        / (a.std().clamp_min(1e-12) * b.std().clamp_min(1e-12))
    )


def phi_features(emb, obs, action):
    """phi = [e, s, a, a*a, 1].

    The raw (already NormalizeObservation-scaled) state block is v2's whole change. See
    the header: without it a linear w_r cannot recover the velocity term from a whitened
    nonlinear latent, and v1 measured the resulting residual at lag-1 autocorrelation
    0.474 -- structured error, the expensive kind, injected into every value estimate.

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on e alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so e has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = emb[..., :1].new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, obs, action, action * action, ones], dim=-1)


def solve_reward_probe(phi, reward, ridge):
    """Closed-form ridge solve of w_r from phi -> immediate reward.

    Solved, not gradient-fit: it removes a learning rate, removes the cold-start phase
    where a random V would drive the actor, and makes "thin, fast, stationary readout"
    literal -- w_r is recomputed optimally every iteration. Done in float64; the normal
    equations square the condition number and phi's blocks have very different scales.
    """
    phi64 = phi.double()
    gram = phi64.T @ phi64
    rhs = phi64.T @ reward.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(phi.dtype)


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def stack_history(obs, cont, hist_len):
    """(T,B,obs) -> (T,B,hist_len*obs), oldest-last, boundary-aware.

    `cont[t] = 1 - transition_boundaries[t]`, i.e. obs[t] and obs[t+1] are the SAME episode
    iff cont[t]==1, so the link reaching back from t to t-h requires cont[t-1..t-h] all set.

    Computed on the FULL buffer before any chunking, so history is correct across SSL chunk
    boundaries rather than restarting every seq_len frames.

    Across a reset (or at t<h, where the previous rollout's frames are gone) the lagged slot
    falls back to obs[t] ITSELF, not zeros. Zeros would be a wild out-of-distribution vector
    under NormalizeObservation -- the encoder would see a state no policy ever visits -- while
    repeating s_t encodes "zero relative motion", which is both in-distribution and the least
    committal thing an unavailable history can say. The t<h case costs hist_len-1 of 2048
    steps per env per rollout.
    """
    if hist_len <= 1:
        return obs
    frames = [obs]
    valid = torch.ones_like(cont)
    for h in range(1, hist_len):
        # cont[t-h], zero-padded (an unavailable link is a broken link).
        c_lag = torch.cat([torch.zeros_like(cont[:h]), cont[:-h]], dim=0)
        valid = valid * c_lag
        prev = torch.cat([obs[:1].expand(h, *obs.shape[1:]), obs[:-h]], dim=0)   # obs[t-h]
        frames.append(torch.where(valid.unsqueeze(-1) > 0, prev, obs))
    return torch.cat(frames, dim=-1)


def chunk_sequences(x, seq_len):
    """(T, B, D) -> (n*B, L, D) with time contiguous within a chunk and env fixed.

    The buffer is T-MAJOR, so the usual flatten gives index t*B + e. Chunking the
    FLATTENED tensor would produce sequences that walk across envs at a fixed timestep
    rather than across time -- and because adjacent envs look similar under
    NormalizeObservation, that yields a perfectly plausible prediction loss that no curve
    would ever catch. Chunk before flattening.

    Round-trip: x[n*L + l, b] == out[n*B + b, l].
    """
    t, b = x.shape[0], x.shape[1]
    tail = x.shape[2:]
    n = t // seq_len
    return (
        x[: n * seq_len]
        .view(n, seq_len, b, *tail)
        .permute(0, 2, 1, *range(3, 3 + len(tail)))
        .reshape(n * b, seq_len, *tail)
    )


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
        # SUCCESSOR-FEATURE critic head. Same shape and MTP semantics as the HL-Gauss head
        # it replaces -- only the per-horizon target space changed, from 511 scalar-return
        # bucket logits to K = emb_dim + obs_dim + 2*act_dim + 1 occupancy dimensions.
        # Horizon h predicts Lambda_{t+h}, the vector-valued lambda-return of phi.
        # Zero-init is deliberate and not merely neutral: with psi_old == 0 the first
        # rollout's Lambda degenerates to the plain discounted sum of phi, i.e. a clean
        # Monte-Carlo target with no bootstrap from a random head.
        self.sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1  # [e, s, a, a*a, 1]
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * self.sf_dim, bias=False), std=0.1
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
        # Returns successor features (B, mtp, sf_dim); horizon 0 is psi(s_t), whose
        # scalar readout w_r . psi_0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)

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
        # PER-ACTUATOR log-probs, kept BEFORE the .sum(1) that is the parent's real
        # bottleneck: this is what lets the ratio -- and therefore the advantage -- be
        # act_dim-dimensional. log_det_fn is elementwise for gaussian and 0.0 for beta, so
        # log_prob_dim is (B, act_dim) in both cases and .sum(1) recovers log_prob exactly.
        log_prob_dim = dist.log_prob(z) - log_det_fn(z)
        log_prob = log_prob_dim.sum(1)
        # The executed sample's deviation from the policy mean, standardized per dimension
        # under the SAME distribution that produced it. Distribution-agnostic (Beta and
        # Normal both expose mean/stddev); mean 0, variance 1 under pi by construction, so
        # it is the natural regressor basis for an additive per-actuator credit model.
        z_hat = (z - dist.mean) / dist.stddev.clamp_min(1e-6)
        value_sf = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)
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
        return action, z, log_prob, entropy, value_sf, log_prob_dim, z_hat

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


def shape_advantage(gae, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has no distributional critic to read -- the head
    predicts successor features. Feeding them a constant sigma=1 / u=0 placeholder would
    make cdf_probit return the same constant for every sample (a zero policy gradient
    after norm_adv) while logging a perfectly healthy-looking curve, so the branches are
    deleted and the arg is rejected at startup instead.
    """
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
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

    # pred_ctx=0 does NOT crash -- it corrupts. The AR loop's `break` leaves per_horizon with
    # K-1 entries while the logging accumulator has K, and at K=2 that broadcasts a length-1
    # tensor across both slots, so every horizon reports the same number and the compounding
    # ratio reads 1.0 (i.e. "the rollout isn't compounding") for a purely cosmetic reason.
    if args.pred_ctx < 1:
        raise ValueError(f"pred_ctx must be >= 1, got {args.pred_ctx}")

    # Fail loud at startup rather than degrading silently mid-run.
    if args.adv_transform in ("tanh_std", "cdf_probit"):
        raise ValueError(
            f"adv_transform={args.adv_transform!r} reads the critic's per-state value "
            "DISTRIBUTION, which this variant does not have (the head predicts successor "
            "features, not bucket logits). Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    n_seq_per_iter = (args.num_steps // args.seq_len) * args.num_envs
    if n_seq_per_iter < args.ssl_batch:
        raise ValueError(
            f"ssl_batch={args.ssl_batch} exceeds the {n_seq_per_iter} sequences a rollout "
            "yields ((num_steps // seq_len) * num_envs). The SSL loop drops the last ragged "
            "minibatch (SIGReg's statistic scales with batch size), so it would take ZERO "
            "steps: the encoder would stay at random init and every sf/ and gate/ metric "
            "would be computed on a frozen random projection."
        )

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
    if auto_alpha and args.vector_adv:
        raise ValueError(
            "vector_adv routes the policy advantage through w_r . E_t, which is a "
            "projection of the REWARD occupancy residual. The soft-advantage max-ent path "
            "adds alpha*H(s') to the bootstrap, and entropy has no phi coordinate, so "
            "there is no vector analogue -- the bonus would be silently dropped. Use "
            "--no-vector-adv, or actor_dist=beta (which disables auto_alpha anyway)."
        )
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # --- LeJEPA successor-feature value pathway -------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )

    if args.compile:
        # retain_graph in the dual-backward is incompatible with donated buffers.
        import torch._functorch.config
        import torch._dynamo

        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.suppress_errors = True   # never let compile break the run
        torch._dynamo.config.recompile_limit = 64     # trunk alone sees 4 (shape, grad) combos
        if agent.share_backbone:
            agent.trunk = CompiledModule(agent.trunk, cudagraphs=False)
        else:
            agent.actor_trunk = CompiledModule(agent.actor_trunk, cudagraphs=False)
            agent.critic_trunk = CompiledModule(agent.critic_trunk, cudagraphs=False)
        for name in ("encoder", "action_encoder", "predictor"):
            setattr(ssl, name, CompiledModule(
                getattr(ssl, name), mode=args.compile_mode, cudagraphs=args.compile_ssl_cudagraphs
            ))

    # Per-dimension EMA standardization of the SF target. The blocks of phi are wildly
    # heteroscedastic: for a whitened e block std(sum gamma^k e) ~ tau ~ 20 (the effective
    # autocorrelation horizon, NOT 1/(1-gamma)=100), while the a*a block is strictly
    # positive with a large mean and small variance, and the constant block is ~1/(1-gamma)
    # exactly. A single global rescale leaves the MSE dominated by the wrong block.
    sf_mean = torch.zeros(sf_dim, device=device)
    sf_std = torch.ones(sf_dim, device=device)
    sf_stat_count = 0
    # Per-dimension mixing time, initialised at the horizon the scalar gae_lambda implies
    # (tau = 1/(1-gamma*lambda) = 16.8 at the defaults) so iteration 1 starts from the
    # base's behaviour rather than an arbitrary one. The 1/count warmup overwrites it
    # exactly on the first measurement.
    tau_vec = torch.full(
        (sf_dim,), 1.0 / (1.0 - args.gamma * args.gae_lambda), device=device
    )
    tau_stat_count = 0
    lam_vec = torch.full((sf_dim,), args.gae_lambda, device=device)
    # w_r starts at zero => V == 0 for the first rollout, which is the correct "no
    # information" baseline rather than random noise driving the actor.
    w_r = torch.zeros(sf_dim, device=device)
    # ... which makes adv_vector = sf_residual @ w_r IDENTICALLY ZERO on iteration 1, and
    # a zero advantage means pg_loss == 0 and the actor gets no gradient at all from the
    # first 32,768 transitions. v2 was not degenerate here (its scalar GAE reduces to the
    # raw discounted reward sum when values == 0). Fall back to scalar GAE until w_r has
    # been solved once; from iteration 2 the projection is live.
    act_dim_pg = int(np.prod(envs.single_action_space.shape))
    # d starts at ZERO: rho == 0, A^(k) == A_t for every actuator, and the unclipped policy
    # gradient is the parent's bit-for-bit until the first solve. There is no nn.Module and
    # no optimizer here at all -- nothing that can leak into agent.parameters() or PPO's clip
    # budgets, and no RNG consumed, so --pgvec and --no-pgvec start from identical weights
    # everywhere by construction rather than by a save/restore dance.
    pgvec_beta = torch.zeros(act_dim_pg * sf_dim, device=device)
    pgvec_shrink_ema = 0.0
    if args.pgvec:
        # The additive form is only unit-matched to `rho` -- which is fit against the MC
        # advantage divided by its own std -- if the policy advantage is standardized too.
        assert args.norm_adv and args.norm_adv_scope in ("batch", "minibatch"), (
            "pgvec adds rho to the NORMALIZED advantage, so norm_adv must be on"
        )
        assert not args.ret_percnorm, "ret_percnorm rescales the advantage after norm_adv"
    w_r_solved = False

    # Fixed INDEX set (not a fixed obs batch) for the frame-drift probe: the states are
    # re-drawn from each rollout, so the probe never goes stale w.r.t. the observation
    # normalizer. Stride 31 is COPRIME with num_envs; the buffer is T-major (i = t*B + e),
    # so a stride sharing a factor with num_envs (e.g. 32 at num_envs=16) would alias onto
    # a single environment and measure drift on one trajectory instead of the state marginal.
    drift_probe_idx = torch.arange(0, args.num_steps * args.num_envs, 31, device=device)[:1024]

    def psi_raw(sf_standardized):
        """Un-standardize the head output back into phi-accumulation units."""
        return sf_standardized * sf_std + sf_mean

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_dim = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    z_hats = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            # The SSL optimizer is separate, so the base's anneal (which writes
            # param_groups[0] of `optimizer` only) does not reach it. Anneal it too: it
            # freezes the encoder frame late in training, which is exactly when the policy
            # is refining and a drifting frame would do the most damage to the bootstrap.
            ssl_optimizer.param_groups[0]["lr"] = frac * args.ssl_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_sf, logprob_dim, z_hat = agent.get_action_and_value(next_obs)
                values[step] = psi_raw(value_sf[:, 0]) @ w_r
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            logprobs_dim[step] = logprob_dim
            z_hats[step] = z_hat

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
            flat_obs_shape = (-1,) + envs.single_observation_space.shape
            psi_next = psi_raw(agent.get_value(next_obses.reshape(flat_obs_shape))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            # Kept as trunk features, not just psi: the EV gate below needs the SAME
            # features to fit its detached scalar probe, and this variant already does two
            # full-batch trunk forwards per iteration where the base did one.
            critic_feat_buf = agent._trunks(obs.reshape(flat_obs_shape))[1]
            psi_cur = psi_raw(
                agent.critic_head(critic_feat_buf).view(-1, args.critic_mtp_horizon, sf_dim)[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)
            next_transition_values = psi_next @ w_r
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
                _, _, boot_logprob, _, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # ---- SUCCESSOR-FEATURE basis --------------------------------------------
            # Built ONCE per iteration and reused by the SSL chunker below, so the encoder
            # sees byte-identical inputs in both places (a second, independently built stack
            # is exactly how a lagged-frame off-by-one hides).
            obs_hist = stack_history(obs, 1.0 - transition_boundaries, args.hist_len)
            emb_buf = ssl.encoder(obs_hist.reshape(-1, obs_hist.shape[-1])).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            # phi keeps the RAW s_t block, NOT obs_hist -- see hist_len's note: putting lagged
            # obs in phi would let a linear w_r capture acceleration without the encoder, which
            # would confound the lever with a much cheaper change.
            phi = phi_features(emb_buf, obs, actions)         # (T, B, sf_dim)

            # ---- truncated-MC scaffolding (moved AHEAD of Lambda) --------------------
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition. The EV
            # gate below still consumes mc_ret/mc_mask; the tau estimator needs them here.
            #
            # mc_phi is the SAME recursion run element-wise on phi, i.e. the lambda=1
            # discounted feature sum. It is what makes the tau estimate below
            # LAMBDA-INDEPENDENT, which is the whole reason it is computed separately
            # instead of being read off sf_std: sf_std is the spread of Lambda(lambda),
            # so deriving lambda from it would close a feedback loop whose only stable
            # point is lambda_max.
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_phi = torch.zeros_like(phi)
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            phi_run = torch.zeros_like(phi[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                phi_run = phi[t] + args.gamma * cont.unsqueeze(-1) * phi_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
                mc_phi[t] = phi_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())

            # ---- PER-DIMENSION lambda from measured mixing time ----------------------
            # THE IDEA. GAE's single lambda is not a free knob -- it is an assumption about
            # ONE number, the effective credit horizon: tau = 1/(1 - gamma*lambda). The
            # default lambda=0.95 at gamma=0.99 asserts tau = 16.8 steps (0.84 s) for
            # EVERY coordinate of the problem. That is false here and measurably so: the
            # obs block mixes at tau ~ 12, the SIGReg-whitened e block at tau ~ 7, the
            # action blocks are nearly white (tau ~ 1), and the constant coordinate never
            # decorrelates at all (its discounted sum is deterministic given the episode
            # end, so Monte Carlo is EXACT and zero-variance for it -- tau -> 1/(1-gamma)).
            #
            # A scalar critic cannot express this: it has one target, so it gets one
            # lambda. A VECTOR target has K coordinates with K different mixing times, so
            # the bias/variance trade should be made per coordinate. Accumulating raw
            # samples past a coordinate's mixing time adds variance with NO bias
            # reduction, because past that point E[phi_{t+k}|s_t] is just the stationary
            # mean, which the critic learns trivially.
            #
            # tau_d is measured, not assumed: for coordinate d the ratio of the discounted
            # sum's spread to the feature's own spread IS the effective horizon,
            #     tau_d = std(sum_k gamma^k phi_d) / std(phi_d),
            # with no AR(1) or exponential-decay assumption -- oscillatory coordinates
            # (joint angles under a periodic gait) cancel correctly in the sum, which a
            # lag-1 autocorrelation estimate would get badly wrong.
            #
            # Inverting tau = 1/(1 - gamma*lambda) gives lambda_d = (1 - 1/tau_d)/gamma.
            # Sanity: tau = 16.8 -> lambda = 0.95 exactly, i.e. this REDUCES to the base's
            # scalar GAE when every coordinate happens to mix at the assumed rate.
            # GATED ON SAMPLE COUNT, at the same threshold the EV gate uses. Two distinct
            # failure modes, both silent:
            #   n_mc == 0  -- mc_avail only reaches mc_window on segments >= 500 steps. On
            #     Hopper/Walker early training that is NO segment, so per_dim_lambda would
            #     be a no-op for many iterations while advB/* reported measured values.
            #   0 < n_mc < 256 -- var_phi passes the degeneracy test, and at
            #     tau_stat_count == 1 the warmup rate is exactly 1.0, so a covariance
            #     estimated from a few dozen consecutive, highly autocorrelated samples off
            #     one or two trajectories would OVERWRITE all sf_dim mixing times outright
            #     and then persist for ~1/tau_ema = 20 iterations through the EMA -- into
            #     the critic target, not just the advantage.
            # tau_stat_count is incremented INSIDE the guard so "exact init at count == 1"
            # stays true: a dead iteration must not consume a warmup slot.
            if args.per_dim_lambda and n_mc >= 256:
                mcp = mc_phi.reshape(-1, sf_dim)[mc_mask]
                php = phi.reshape(-1, sf_dim)[mc_mask]
                php_c = php - php.mean(0)
                mcp_c = mcp - mcp.mean(0)
                var_phi = php_c.pow(2).mean(0)
                # COVARIANCE, NOT A RATIO OF STANDARD DEVIATIONS. This distinction is the
                # whole estimator and it is easy to get wrong:
                #     std(sum_k gamma^k phi_d) / std(phi_d)
                # measures the spread of the REALIZED return, which is inflated by the
                # accumulated future noise that lambda exists to avoid. For a white
                # coordinate it returns 1/sqrt(1-gamma^2) ~ 7.1 instead of the true 1 --
                # it would ask for heavy Monte Carlo on exactly the coordinates where MC
                # is pure variance. Verified numerically against AR(1) ground truth.
                #
                # What is wanted is the spread of the CONDITIONAL EXPECTATION,
                # Lambda(s) = sum_k gamma^k E[phi_{t+k}|s_t]. Since the future noise is
                # independent of phi_t it drops out of a covariance:
                #     Cov(phi_d[t], sum_k gamma^k phi_d[t+k]) / Var(phi_d[t])
                #       = sum_k gamma^k rho_d(k)  =  tau_d
                # i.e. the discounted integrated autocorrelation time, estimated as a
                # per-dimension OLS slope. White -> 1. AR(1) -> 1/(1-gamma*rho).
                # Oscillatory -> small, because the alternating rho_d(k) cancel in the sum,
                # which is the failure mode a lag-1 estimate cannot see.
                tau_raw = (php_c * mcp_c).mean(0) / var_phi.clamp_min(1e-8)
                # A coordinate with Var(phi_d) = 0 -- the intercept, and any dead obs dim --
                # gives 0/0. The estimator cannot see it, but its lambda is NOT irrelevant:
                # lambda_d multiplies the trace of delta_sf[...,d], and psi_cur[...,d] is a
                # learned state-varying quantity (for the intercept it encodes expected
                # discounted remaining episode length), so this coordinate moves both
                # adv_vector and the critic target.
                # The right answer is available analytically instead of empirically: if
                # phi_d is deterministic then so is sum_k gamma^k phi_d, i.e. Monte Carlo is
                # EXACT and zero-variance for it, so lambda_d = 1 with tau = 1/(1-gamma).
                # This is the coordinate the header names as the motivating example; falling
                # back to the scalar default here would silently contradict it.
                tau_det = 1.0 / (1.0 - args.gamma)
                tau_raw = torch.where(
                    var_phi > 1e-8, tau_raw, torch.full_like(tau_raw, tau_det)
                )
                # NOTE, load-bearing: this same branch is what makes an all-NaN var_phi
                # (n_mc == 0, mean over an empty tensor) safe, since NaN > 1e-8 is False.
                # The n_mc guard above makes that unreachable; do not invert the predicate.
                # tau < 1 means anti-persistent (the discounted sum partially cancels the
                # current value); the shortest meaningful credit horizon is one step.
                tau_raw = tau_raw.clamp(1.0, 1.0 / (1.0 - args.gamma))
                # Same 1/count warmup idiom as the Lambda standardizer: an exact running
                # mean early (and an exact init at count == 1), decaying into the EMA.
                tau_stat_count += 1
                tau_rate = max(args.tau_ema, 1.0 / tau_stat_count)
                tau_vec = tau_vec + tau_rate * (tau_raw - tau_vec)
                lam_vec = ((1.0 - 1.0 / tau_vec.clamp_min(1.0 + 1e-6)) / args.gamma).clamp(
                    args.lam_min, args.lam_max
                )
            else:
                lam_vec = torch.full((sf_dim,), args.gae_lambda, device=device)

            # ---- vector TD(lambda) ---------------------------------------------------
            # Element-wise on phi, mirroring the reward GAE. THE TWO MASKS ARE NOT THE SAME
            # MASK. They differ exactly at a TRUNCATION (term=0, valid=1, boundary=1):
            # bootstrap through it, but CUT the lambda trace. Collapsing them would let the
            # next episode's discounted phi-sum bleed backward into this episode's target
            # from a fresh, near-zero-velocity reset state -- ~1.6% of samples corrupted by
            # roughly the reset-vs-running value gap. Written in residual space so the
            # rollout tail (E_T = 0) is handled for free.
            # lam_vec is (sf_dim,) and broadcasts over (B, sf_dim); with per_dim_lambda off
            # it is a constant vector and this is bit-identical to v2.
            sf_residual = torch.zeros_like(phi)
            last_sf = torch.zeros_like(phi[0])
            for t in reversed(range(args.num_steps)):
                boot = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
                cont = (1.0 - transition_boundaries[t]).unsqueeze(-1)
                delta_sf = phi[t] + args.gamma * boot * psi_next[t] - psi_cur[t]
                last_sf = delta_sf + args.gamma * lam_vec * cont * last_sf
                sf_residual[t] = last_sf
            sf_target = sf_residual + psi_cur                 # Lambda_t

            # ---- THE POLICY ADVANTAGE, PROJECTED OUT OF THE VECTOR RESIDUAL ----------
            # This is the change that lets per-dimension lambda reach the policy at all.
            # sf_residual IS a vector-valued advantage, and the identity
            #     w_r . E_t == A_t   exactly whenever   w_r . phi_t == r_t
            # means v2's scalar GAE was already a projection of it -- just one computed
            # along a separate scalar recursion with a single lambda. Routing the policy
            # advantage through the vector path instead makes lambda a per-coordinate
            # object, which no scalar recursion can express.
            #
            # w_r here is DELIBERATELY the PREVIOUS iteration's solve, not this one's: it
            # is the same w_r that produced values[] during the rollout and
            # next_transition_values above, so the projection stays self-consistent with
            # the baseline it is differenced against. Re-solving first would mix two
            # different reward readouts into one advantage.
            #
            # HONEST COST. The identity is exact only if the probe is exact. It is not
            # (reward_probe_r2 ~ 0.98), so this injects the accumulated probe residual
            # sum_k (gamma*lam)^k eps_{t+k} into the advantage -- the quantity
            # sf/value_err_frac already tracks. --no-vector-adv is the control that prices
            # exactly this, holding per-dimension lambda out of the policy path.
            adv_vector = sf_residual @ w_r

            # ---- PER-ACTUATOR CREDIT: fit d, then form rho ---------------------------
            # Runs ONCE per iteration, in the same slot as the SSL step -- NOT inside the
            # 320-minibatch PPO loop, and on the rollout just collected, using the SAME
            # (previous) w_r that produced values[] and adv_vector, so the sensitivity model
            # is fit against the baseline it is actually differenced against.
            #
            # Gated on w_r_solved (on iteration 1 w_r == 0, so E_t carries no reward direction
            # and the regression would be fitting the residual's raw geometry) and on the
            # parent's own n_mc >= 256 threshold for a usable truncated-MC target.
            rho = torch.zeros(args.num_steps, args.num_envs, act_dim_pg, device=device)
            pg_ev_mc = pg_shrink = pg_split_r = pg_rho_raw = 0.0
            pgvec_active = args.pgvec and w_r_solved and n_mc >= 256
            if pgvec_active:
                with torch.no_grad():
                    resid_flat = sf_residual.reshape(-1, sf_dim)
                    res_std = resid_flat.std(0).clamp_min(1e-6)   # spread of E_t itself
                    zhat_flat = z_hats.reshape(-1, act_dim_pg)
                    # THE TARGET: the truncated-MC ADVANTAGE, divided by its own spread so rho
                    # lands in the same units as the batch-standardized policy advantage it is
                    # ADDED to. The parent's EV gate already computes mc_ret; `values` is the
                    # same (previous) w_r's value that produced adv_vector, so the target and
                    # the baseline it is differenced against share one frame.
                    mc_adv = (mc_ret - values).reshape(-1)
                    mc_idx = mc_mask.nonzero(as_tuple=True)[0]
                    tgt = mc_adv / mc_adv[mc_idx].std().clamp_min(1e-6)
                    perm = torch.randperm(mc_idx.numel(), device=device)
                    n_val = max(64, int(args.pgvec_val_frac * mc_idx.numel()))
                    val_idx, tr_idx = mc_idx[perm[:n_val]], mc_idx[perm[n_val:]]

                    X_tr = pgvec_design(resid_flat[tr_idx], res_std, zhat_flat[tr_idx])
                    X_val = pgvec_design(resid_flat[val_idx], res_std, zhat_flat[val_idx])
                    pgvec_beta = pgvec_solve(X_tr, tgt[tr_idx], args.pgvec_ridge)

                    # ---- SHRINKAGE BY SPLIT-HALF RELIABILITY, not by EV ------------------
                    # A first draft shrank rho by clamp(EV of the MC-advantage fit, 0, 1). That
                    # is the WRONG statistic and would have pinned the treatment at zero for the
                    # whole run: a single-sample 500-step MC return is dominated by irreducible
                    # policy+env noise, so EV against it is capped near the attributable-variance
                    # fraction (a few percent) no matter how well the SENSITIVITY is estimated.
                    # What actually needs shrinking is rho, and the right question about rho is
                    # how much of it REPRODUCES rather than how much of the return it explains.
                    # So: solve two independent halves of the training rows and correlate their
                    # predictions on data NEITHER saw. That is a reliability ratio -- signal
                    # variance over total variance IN THE ESTIMATOR -- which is the optimal
                    # shrinkage here and is NOT capped by the MC noise floor.
                    h = tr_idx.numel() // 2
                    b1 = pgvec_solve(X_tr[:h], tgt[tr_idx[:h]], args.pgvec_ridge)
                    b2 = pgvec_solve(X_tr[h:], tgt[tr_idx[h:]], args.pgvec_ridge)
                    r1 = pgvec_rho(pgvec_logits(b1, resid_flat[val_idx], res_std, act_dim_pg),
                                   zhat_flat[val_idx])
                    r2 = pgvec_rho(pgvec_logits(b2, resid_flat[val_idx], res_std, act_dim_pg),
                                   zhat_flat[val_idx])
                    pg_split_r = corr_score(r1, r2)
                    # EMA'd across iterations: n_val swings ~2x deterministically with the
                    # rollout's phase relative to the 1000-step episode (mc_mask keeps 27-49%
                    # of rows depending on where the window lands), so a per-iteration
                    # reliability would carry that oscillation straight into the policy
                    # gradient as nonstationarity.
                    pgvec_shrink_ema = 0.9 * pgvec_shrink_ema + 0.1 * min(max(pg_split_r, 0.0), 1.0)
                    pg_shrink = pgvec_shrink_ema

                    lg_v = pgvec_logits(pgvec_beta, resid_flat[val_idx], res_std, act_dim_pg)
                    # Held-out EV, now DIAGNOSTIC ONLY. Expected to be small or negative even
                    # when the split is real -- see the note above. It is logged so a reader can
                    # see the noise floor, not to gate anything.
                    pg_ev_mc = ev_score((lg_v * zhat_flat[val_idx]).sum(-1), tgt[val_idx])

                    lg_all = pgvec_logits(pgvec_beta, resid_flat, res_std, act_dim_pg)
                    rho_raw = pgvec_rho(lg_all, zhat_flat)
                    pg_rho_raw = float(rho_raw.abs().mean())
                    rho = (pg_shrink * rho_raw).reshape(
                        args.num_steps, args.num_envs, act_dim_pg
                    )

            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            # Always computed: it is the control arm under --no-vector-adv, and the
            # correlation between the two is the headline diagnostic either way.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            adv_scalar_gae = advantages
            if args.vector_adv and w_r_solved:
                advantages = adv_vector
            # corr(w_r.E, scalar GAE): 1.0 would mean the vector path changed nothing and
            # the run is a null experiment; a large drop means per-dimension lambda (or the
            # probe residual) genuinely moved the policy signal.
            _a, _b = adv_vector.reshape(-1), adv_scalar_gae.reshape(-1)
            adv_vec_corr = float(
                ((_a - _a.mean()) * (_b - _b.mean())).mean()
                / (_a.std().clamp_min(1e-12) * _b.std().clamp_min(1e-12))
            )
            # DELIBERATELY built from the scalar GAE, not from `advantages`. Under
            # vector_adv, advantages + values = w_r.(Lambda - psi) + w_r.psi = w_r.Lambda,
            # in which NO reward tensor appears at all -- losses/explained_variance and
            # debug/returns_* would silently start describing the vector residual's
            # magnitude, differ in meaning between the treatment and control arms, and stop
            # being comparable to v2's curve. `returns` is not the critic target here (the
            # critic regresses sf_target), so decoupling the two is free.
            returns = adv_scalar_gae + values
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
            # leaving policy_adv RAW here. Divide-only. NOTE `returns` is NOT the critic target in this
            # file (the critic regresses sf_target); it is the reward lambda-return, kept for
            # the percentile scale and the EV diagnostics.
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
            # Closed-form w_r on IMMEDIATE reward -- no bootstrapping, no multi-step. This
            # is the only scalar regression anywhere in the design. Solved after the GAE so
            # values[] and next_transition_values[] both use the same (previous) w_r and the
            # advantage stays self-consistent.
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            w_r = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            w_r_solved = True
            reward_resid = flat_rew - flat_phi @ w_r
            reward_r2 = 1.0 - (reward_resid.var() / flat_rew.var().clamp_min(1e-12)).item()
            # R^2 alone is the WRONG gate: r_t uses the frame-skip AVERAGED velocity while the
            # observation carries instantaneous qvel, so a structural residual of ~1-3% is
            # expected and harmless IF it is white. What costs EV is a gait-phase-locked,
            # speed-correlated residual, which shows up as autocorrelation, not as R^2.
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            reward_resid_ac = [
                ((resid_c[:-k] * resid_c[k:]).mean() / resid_var).item() for k in (1, 5, 10, 20)
            ]

            # Per-dimension EMA standardization of the target (see setup for why).
            # The 1/count warmup is NOT cosmetic. e's own scale moves fast for the first
            # few dozen iterations while SIGReg whitens it, but 1/sf_target_ema = 100
            # iterations is 3.3M steps of lag -- the standardization would spend a third of
            # the run tracking a distribution the encoder left behind. rate = 1/count is an
            # exact running mean early (and an exact init at count == 1, replacing a
            # separate first-iteration branch) and decays into the plain EMA once the
            # encoder settles.
            sf_stat_count += 1
            sf_rate = max(args.sf_target_ema, 1.0 / sf_stat_count)
            flat_tgt = sf_target.reshape(-1, sf_dim)
            tgt_mean, tgt_std = flat_tgt.mean(0), flat_tgt.std(0).clamp_min(1e-6)
            sf_mean = sf_mean + sf_rate * (tgt_mean - sf_mean)
            sf_std = sf_std + sf_rate * (tgt_std - sf_std)

            # MTP: horizon h regresses Lambda_{t+h}. Masks are the base's, verbatim -- a
            # future target is valid only when no reset boundary lies between source and
            # target state and it stays inside the rollout.
            mtp = args.critic_mtp_horizon
            sf_mtp = sf_target.new_zeros((args.num_steps, args.num_envs, mtp, sf_dim))
            return_mtp_mask = torch.zeros(
                (args.num_steps, args.num_envs, mtp), dtype=torch.bool, device=device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                sf_mtp[:valid_len, :, h] = sf_target[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            sf_mtp = (sf_mtp - sf_mean) / sf_std              # head regresses standardized units

            # ---- three-way EV gate: the falsifier, computed on an UNRIGGED target ----
            # The usual EV target `b_returns = advantages + values` CONTAINS the critic's
            # own bootstrapped values, and at dt=0.05 the errors are strongly
            # state-autocorrelated -- a critic scores well against a target built from
            # itself. Every predictor below is scored against the same truncated
            # Monte-Carlo discounted return instead.
            #
            # mc_ret / mc_avail / mc_mask are computed EARLIER now (the per-dimension
            # tau estimator needs them before Lambda is formed). Only the probe-residual
            # recursion stays here: it needs reward_resid, i.e. the w_r solved this
            # iteration, which does not exist until after Lambda.
            mc_resid = torch.zeros_like(rewards)          # discounted PROBE residual
            resid_run = torch.zeros_like(rewards[0])
            resid_tb2 = reward_resid.reshape(args.num_steps, args.num_envs)
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                resid_run = resid_tb2[t] + args.gamma * cont * resid_run
                mc_resid[t] = resid_run
            if n_mc >= 256:
                flat_mc = mc_ret.reshape(-1)[mc_mask]
                feat_mc = critic_feat_buf[mc_mask]
                ones_mc = feat_mc.new_ones(feat_mc.shape[0], 1)
                # (1) reference: what a scalar critic reading THIS trunk could achieve.
                #     Detached closed-form probe, so no scalar-return gradient ever enters
                #     the network -- it replaces the in-run HL-Gauss comparator that the
                #     head swap removed.
                trunk_feat_mc = torch.cat([feat_mc, ones_mc], dim=-1)
                ev_trunk_probe = ev_score(
                    trunk_feat_mc @ solve_reward_probe(trunk_feat_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # (3) how much reward-relevant signal is LINEARLY present in the
                #     INSTANTANEOUS embedding. Not a ceiling on psi: psi reads the trunk,
                #     not e_t, and accumulates phi over time, so (2) > (3) is expected and
                #     is in fact the successor-feature construction earning its keep.
                #     Treat (3) as a floor-diagnostic on the encoder, not an upper bound.
                e_mc = torch.cat([emb_buf.reshape(-1, args.emb_dim)[mc_mask], ones_mc], dim=-1)
                ev_latent_cap = ev_score(
                    e_mc @ solve_reward_probe(e_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # THE decisive reward-probe metric, and the reason R^2 and the AC lags are
                # both kept only as secondary reads. Since V = w_r.Lambda and
                # Lambda = sum gamma^k phi, we have w_r.Lambda = G_t - sum gamma^k eps_{t+k}:
                # the value error IS the discounted sum of probe residuals. R^2 alone is
                # actively misleading here -- measured offline, adding s' to phi raises R^2
                # from 0.61 to 0.94 while making THIS number WORSE (0.77 -> 0.84), because
                # the discounted sum amplifies a small correlated residual far more than a
                # large white one. Ratio to the value's own spread; lower is better.
                value_err_frac = float(
                    mc_resid.reshape(-1)[mc_mask].std() / flat_mc.std().clamp_min(1e-12)
                )
                # (2) the treatment: the ONLINE value the actor actually consumed.
                ev_sf = ev_score(values.reshape(-1)[mc_mask], flat_mc)
                # (1) and (3) are IN-SAMPLE ridge fits, i.e. optimistic upper bounds; (2) is
                # an honest out-of-sample prediction. So (3) vs (1) is the apples-to-apples
                # comparison -- (3) << (1) means the LATENT is the bottleneck and no amount
                # of psi quality can rescue this. (2) << (3) with (3) ~ (1) instead points
                # at the SF machinery (recursion, standardization), not the encoder.
            else:
                ev_trunk_probe = ev_latent_cap = ev_sf = value_err_frac = float("nan")

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_logprobs_dim = logprobs_dim.reshape(-1, act_dim_pg)
        b_rho = rho.reshape(-1, act_dim_pg)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_sf, newlogprob_dim, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                # NOTE: at the default pos_neg_alpha=0.5 this whole block is the identity. It
                # is left BEFORE the per-actuator split deliberately: splitting on the scalar
                # sign and then adding rho would give an actuator with A^(k) < 0 the positive
                # weight, but moving it after the split would make the weighting depend on rho
                # and stop the arm being one variable. Flagged rather than silently either way.
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                if args.pgvec:
                    # ---- THE DE-COLLAPSE ------------------------------------------------
                    # act_dim independent (advantage, ratio) pairs instead of one. rho is
                    # applied HERE, after the entire normalization pipeline, because it is
                    # dimensionless and multiplicative: none of shape_advantage / norm_adv /
                    # ret_percnorm / pos_neg_alpha has to be duplicated or threaded, and
                    # mean_k A^(k) == mb_advantages exactly regardless of what they did.
                    # ADDITIVE, not multiplicative. The regression identifies a SIGNED quantity
                    # -- rho_k > 0 means actuator k's deviation predicted a HIGHER return -- so
                    # the credit it identifies is A_t + rho_k. A_t*(1 + rho_k) has the right
                    # sign only where A_t > 0; on the negative half of the batch (and norm_adv
                    # makes it exactly half, since batch standardization is zero-mean) it
                    # PUNISHES the actuator the model says helped. Measured on the identified
                    # decomposition: corr(applied deviation, identified deviation) is +0.80 on
                    # A>0, -0.80 on A<0, and -0.003 overall -- the multiplicative form carries
                    # ZERO net signal. Additive conserves the mean just as exactly (rho is
                    # mean-zero across k) and is still PPO-exact at rho == 0.
                    adv_pa = mb_advantages.unsqueeze(1) + b_rho[mb_inds]          # (B, act_dim)
                    logratio_dim = newlogprob_dim - b_logprobs_dim[mb_inds]
                    ratio_dim = logratio_dim.exp()
                    # PER-ACTUATOR clipping: a dimension that moved too far is zeroed while the
                    # others stay live. SUM over k, not mean -- at rho == 0 and ratio == 1 the
                    # gradient is -A * sum_k grad log pi_k = -A * grad log pi, i.e. the parent's
                    # policy gradient exactly. A mean would divide it by act_dim.
                    #
                    # The per-dim bounds are the act_dim-th ROOT of the parent's, so the product
                    # of the per-dim ratios -- which IS the parent's joint ratio -- is confined
                    # to exactly the parent's region. Clipping each dimension at the raw
                    # clip_coef instead was measured to widen the joint log-ratio ~6x and inflate
                    # joint approx_kl 1.7x on a real Beta actor; with target_kl=0.03 live that
                    # changes the NUMBER OF EPOCHS per iteration, so the arm would have differed
                    # in optimizer steps as well as in credit assignment.
                    if args.pgvec_clip_match:
                        lo_k = (1.0 - args.clip_coef) ** (1.0 / act_dim_pg)
                        hi_k = (1.0 + clip_hi) ** (1.0 / act_dim_pg)
                    else:
                        lo_k, hi_k = 1.0 - args.clip_coef, 1.0 + clip_hi
                    pg_loss1 = -adv_pa * ratio_dim
                    pg_loss2 = -adv_pa * torch.clamp(ratio_dim, lo_k, hi_k)
                    pg_loss = torch.max(pg_loss1, pg_loss2).sum(1).mean()
                    with torch.no_grad():
                        clipfrac_dim = (
                            ((ratio_dim < lo_k) | (ratio_dim > hi_k)).float().mean().item()
                        )
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    clipfrac_dim = clipfracs[-1]

                # SUCCESSOR-FEATURE value loss: per-horizon masked MSE against the
                # standardized latent lambda-return, summed over valid horizons per row.
                # No scalar-return regression anywhere at the default sf_alpha=1.
                sf_tgt = b_sf_target[mb_inds]
                sf_err = value_sf - sf_tgt                                    # (B, mtp, sf_dim)
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)          # (B, mtp)
                v_loss = (sf_err.pow(2).mean(-1) * value_mask).sum(dim=-1).mean()
                if args.sf_alpha < 1.0:
                    # Reward-direction term. Its target w_r.Lambda still comes from the
                    # LATENT lambda-return, so the bootstrap stays in latent space and the
                    # thesis is intact; sf_alpha only decides how much of psi's capacity is
                    # spent on the one direction that is actually read out.
                    scalar_err = (sf_err @ (w_r * sf_std)).pow(2)             # (B, mtp)
                    v_loss = args.sf_alpha * v_loss + (1.0 - args.sf_alpha) * (
                        scalar_err * value_mask
                    ).sum(dim=-1).mean()
                # Per-horizon psi MSE (last minibatch of the last epoch is what gets logged).
                sf_per_h_mse = (sf_err.detach().pow(2).mean(-1) * value_mask).sum(0) / value_mask.sum(
                    0
                ).clamp_min(1)

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
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- LeJEPA SSL step -----------------------------------------------------------
        # ONCE PER ITERATION, OUTSIDE the 320-minibatch PPO loop: inside it costs +100-200%
        # wall clock, outside it costs +10-20%. Placed AFTER the PPO update so the encoder
        # frame is frozen across the entire target-construction + critic-fitting phase. The
        # residual drift is one ITERATION's worth (ssl_steps_target = 64
        # steps at defaults, not one), which is exactly what ssl/frame_drift_* measures.
        with torch.no_grad():
            # obs_hist, not obs: the SAME stack the phi/emb_buf path used above.
            seq_obs = chunk_sequences(obs_hist, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            # Masking ONLY the crossing transition (1 - boundaries[l]) is not enough: chunks
            # are cut on fixed multiples of seq_len in rollout time, not on episode
            # boundaries, so a reset at intra-chunk position 0 still leaves later positions
            # predicting normally while causal attention reads position 0 -- a state from the
            # PREVIOUS episode. The CUMPROD requires the whole context to be same-episode,
            # which is the actual precondition for the prediction to be well-posed. Prefix
            # positions before the first boundary are kept, so a straddling chunk is
            # degraded, not discarded.
            # Passed to the SSL forward UNSLICED now: with an AR rollout each horizon needs a
            # different window of it, so the slicing belongs where the horizons are known.
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
            seq_cum_cont = seq_cont.cumprod(dim=1)
        n_seq = seq_obs.shape[0]
        ssl_pred_l = ssl_sig_l = ssl_gn_sum = 0.0
        ssl_steps = 0
        # Accumulated as a TENSOR on device and synced once at the end of the iteration,
        # rather than .item()'d inside the 64-step inner loop.
        ssl_per_h = torch.zeros(args.pred_horizon, device=device)
        # e BEFORE this iteration's SSL step, on states from THIS rollout. Paired with the
        # post-step embedding of the SAME inputs below, this isolates frame movement from
        # distribution shift without needing to keep a copy of the old encoder.
        probe_emb_before = emb_buf.reshape(-1, args.emb_dim)[drift_probe_idx].clone()
        # Derived so the SSL gradient budget is invariant to pred_horizon -- see
        # ssl_steps_target. At K=1 this reproduces v2 exactly: 8192//1024 = 8 steps/epoch,
        # 64/8 = 8 epochs.
        ssl_steps_per_epoch = max(1, n_seq // args.ssl_batch)
        ssl_n_epochs = max(1, round(args.ssl_steps_target / ssl_steps_per_epoch))
        for _ in range(ssl_n_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l, per_h = ssl(
                    seq_obs[idx], seq_act[idx], seq_cum_cont[idx], args.sigreg_weight
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_gn = nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_optimizer.step()
                ssl_pred_l += pred_l.item()
                ssl_sig_l += sig_l.item()
                ssl_per_h += per_h
                ssl_gn_sum += float(ssl_gn)
                ssl_steps += 1
        ssl_pred_l /= max(ssl_steps, 1)
        ssl_sig_l /= max(ssl_steps, 1)
        ssl_gn_sum /= max(ssl_steps, 1)

        # ---- encoder health + frame drift ----------------------------------------------
        # SIGReg pins the DISTRIBUTION of e, not its coordinate frame: N(0,I) is
        # rotation-invariant and the prediction loss cannot pin the frame either (the
        # predictor co-rotates). But psi_old and w_r are functions of COORDINATES, so a
        # drifting frame makes the bootstrap target stale. This is a non-stationarity the
        # design INTRODUCES, so it is measured rather than pre-emptively patched.
        #   frame_drift_raw ~ 1 and frame_drift_rot ~ 1  -> stable, bootstrap is sound
        #   frame_drift_raw << 1 but frame_drift_rot ~ 1 -> pure rotation, the failure mode
        # frame_drift_rot is in [0,1] by von Neumann's trace inequality; frame_drift_raw is
        # in [-1,1] (it goes negative under anti-correlation). Both are 1 iff no drift.
        #
        # Both embeddings are of the SAME inputs drawn from THIS rollout, taken before and
        # after this iteration's SSL step. A probe frozen at iteration 1 would be wrong in
        # two ways: those are post-NormalizeObservation vectors carrying iteration-1
        # running stats, so by mid-training they decode to raw states the agent never
        # visits -- and OOD inputs drift MORE, so the metric would cry wolf about a frame
        # instability the actual bootstrap never sees.
        with torch.no_grad():
            # obs_hist, matching probe_emb_before (which came from emb_buf, itself built from
            # obs_hist). Re-encoding raw obs here would silently compare embeddings of two
            # DIFFERENT inputs under hist_len>1 and report frame drift that is really a
            # difference in input width.
            probe_emb_after = ssl.encoder(
                obs_hist.reshape(-1, obs_hist.shape[-1])[drift_probe_idx]
            ).float()
            a = probe_emb_before - probe_emb_before.mean(0)
            b_ = probe_emb_after - probe_emb_after.mean(0)
            na, nb = a.pow(2).sum(), b_.pow(2).sum()
            denom = (0.5 * (na + nb)).clamp_min(1e-12)
            drift_raw = float(1.0 - (b_ - a).pow(2).sum() / (2.0 * denom))
            drift_rot = float(torch.linalg.svdvals(b_.T.double() @ a.double()).sum() / denom.double())
            # e's marginal: SIGReg should drive these to (0, I). Read from the POST-step
            # embedding, so this panel and ssl/sigreg_epps_pulley describe the same encoder
            # -- emb_buf is the PRE-step encoder and would lag by a full 64 SSL steps,
            # making the very first ssl/emb_std point describe the random init.
            # Effective rank is expected near 17 (the observation dimension), NOT 32: a
            # single-state encoder cannot exceed the manifold's rank, and that is not a
            # failure -- random 1-D projections of a rank-17 pushforward still look
            # near-Gaussian by mixing, so the statistic converges anyway.
            emb_mean_abs = float(probe_emb_after.mean(0).abs().mean())
            emb_std = float(probe_emb_after.std(0).mean())
            cov_e = (b_.T @ b_) / b_.shape[0]
            eig = torch.linalg.eigvalsh(cov_e.double()).clamp_min(0)
            p_eig = eig / eig.sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p_eig * (p_eig + 1e-12).log()).sum()))

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        # ---- successor-feature diagnostics -------------------------------------
        # R^2 is the weak gate; a structural ~1-3% residual is expected (frame-skip
        # averaged reward velocity vs instantaneous qvel). The residual's
        # AUTOCORRELATION is what actually costs EV: white residual at R^2=0.98
        # costs ~0.2% EV, a gait-phase-locked one at the same R^2 costs ~2%.
        writer.add_scalar("sf/value_err_frac", value_err_frac, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        for lag, ac in zip((1, 5, 10, 20), reward_resid_ac):
            writer.add_scalar(f"sf/reward_resid_ac_lag{lag}", ac, global_step)
        for h in range(args.critic_mtp_horizon):
            writer.add_scalar(f"sf/psi_mse_h{h}", sf_per_h_mse[h].item(), global_step)
        # ||Lambda|| per block against the tau ~= 20 prediction for a whitened,
        # zero-mean e block under gamma=0.99, lambda=0.95.
        writer.add_scalar("sf/lambda_std_emb", sf_std[: args.emb_dim].mean().item(), global_step)
        # The raw-state block's scale is pinned by NormalizeObservation, not SIGReg; watch
        # it for the slow normalizer drift that replaces v1's (dead) frame-drift risk.
        writer.add_scalar("sf/lambda_std_obs",
                          sf_std[args.emb_dim : args.emb_dim + obs_dim].mean().item(), global_step)
        writer.add_scalar("sf/lambda_std_act",
                          sf_std[args.emb_dim + obs_dim : -1].mean().item(), global_step)
        writer.add_scalar("sf/lambda_absmean_emb", sf_mean[: args.emb_dim].abs().mean().item(), global_step)
        writer.add_scalar("sf/w_r_norm", w_r.norm().item(), global_step)

        # ---- B: per-dimension lambda and the vector-routed advantage ------------------
        # corr ~ 1.0 => the vector path reproduced scalar GAE and the run is a null
        # experiment. The lam_* spread is the whole premise: if the three blocks converge
        # to the same lambda, the heterogeneity this variant exploits does not exist.
        writer.add_scalar("advB/adv_vec_corr", adv_vec_corr, global_step)
        writer.add_scalar("advB/lam_emb", lam_vec[: args.emb_dim].mean().item(), global_step)
        writer.add_scalar("advB/lam_obs",
                          lam_vec[args.emb_dim : args.emb_dim + obs_dim].mean().item(), global_step)
        writer.add_scalar("advB/lam_act",
                          lam_vec[args.emb_dim + obs_dim : -1].mean().item(), global_step)
        writer.add_scalar("advB/lam_const", lam_vec[-1].item(), global_step)
        writer.add_scalar("advB/lam_spread", (lam_vec.max() - lam_vec.min()).item(), global_step)
        writer.add_scalar("advB/tau_emb", tau_vec[: args.emb_dim].mean().item(), global_step)
        writer.add_scalar("advB/tau_obs",
                          tau_vec[args.emb_dim : args.emb_dim + obs_dim].mean().item(), global_step)
        writer.add_scalar("advB/tau_act",
                          tau_vec[args.emb_dim + obs_dim : -1].mean().item(), global_step)
        # ---- PER-ACTUATOR CREDIT DIAGNOSTICS ------------------------------------------
        # Written unconditionally every iteration so a flat line is visible as a flat line
        # rather than as a missing tag.
        with torch.no_grad():
            rho_abs = rho.abs()
            # THE NULL DETECTOR. ~0 means the split never moved off uniform and the run is a
            # null experiment -- read it BEFORE reading the return curve.
            writer.add_scalar("pgvec/rho_absmean", float(rho_abs.mean()), global_step)
            writer.add_scalar("pgvec/rho_max", float(rho_abs.max()), global_step)
            # Conservation, asserted continuously rather than trusted: mean_k rho_k must be 0
            # to machine precision or the head is silently changing TOTAL credit, not just its
            # distribution, and every advantage-scale statistic downstream is compromised.
            writer.add_scalar(
                "pgvec/rho_mean_resid", float(rho.mean(-1).abs().max()), global_step
            )
            # Spread of the per-actuator multiplier across actuators at a state: 0 = uniform
            # (PPO), large = strongly differentiated credit.
            writer.add_scalar("pgvec/rho_spread", float(rho.std(-1).mean()), global_step)
            # How often a per-actuator advantage INVERTS relative to the total (rho < -1).
            # Not clamped -- an actuator that moved against the realized advantage arguably
            # should reverse -- but a large fraction alongside a falling ev_mc is the runaway.
            # Tail of the redistribution. A trained Beta policy is skewed, so zhat has a
            # one-sided ~6-sigma tail and rho is NOT bounded. Left unclamped -- an actuator that
            # moved hard against the realized advantage arguably should be pushed hard the other
            # way, and PPO's per-dim clip already bounds the step it can produce -- but the p99
            # is logged so a heavy tail is visible rather than inferred.
            writer.add_scalar("pgvec/rho_p99", float(rho_abs.flatten().quantile(0.99)), global_step)
        # THE PREMISE TEST, on the HELD-OUT MC subset only: does an additive per-actuator
        # sensitivity model predict the realized advantage at all? <= 0 falsifies the idea
        # that credit here is attributable to individual actuators.
        # DIAGNOSTIC ONLY now: EV against a single-sample 500-step MC advantage is capped near
        # the attributable-variance fraction, so a small or negative number here is the noise
        # floor of the TARGET, not evidence about the credit split. pgvec/split_r is the test.
        writer.add_scalar("pgvec/ev_mc", pg_ev_mc, global_step)
        # THE PREMISE TEST. Split-half reliability of rho: two independent ridge solves on
        # disjoint halves of the training rows, correlated on rows NEITHER saw. ~0 means the
        # credit split does not reproduce and is noise; the shrinkage then drives rho to 0 and
        # this run IS the parent. Large means the per-actuator structure is real.
        writer.add_scalar("pgvec/split_r", pg_split_r, global_step)
        writer.add_scalar("pgvec/shrink", pg_shrink, global_step)
        # PRE-shrink magnitude. Logged separately so "the head found nothing" is distinguishable
        # from "the head found something the shrinkage discarded" -- with only the post-shrink
        # number both read as zero.
        writer.add_scalar("pgvec/rho_absmean_raw", pg_rho_raw, global_step)
        writer.add_scalar("pgvec/active", float(pgvec_active), global_step)
        # The shrinkage actually applied to rho this iteration = clamp(ev_mc, 0, 1). Read
        # WITH rho_absmean: a small rho at a large shrink means the head found little to
        # redistribute; a small rho at ~0 shrink means the premise is failing.
        # Per-actuator clip rate against the parent's JOINT clipfrac. The per-dim trust region
        # is looser in aggregate; this pair plus losses/approx_kl is how that is separated from
        # the credit split when reading a result.
        writer.add_scalar("pgvec/clipfrac_dim", clipfrac_dim, global_step)
        writer.add_scalar("advB/adv_vector_std", adv_vector.std().item(), global_step)
        writer.add_scalar("advB/adv_scalar_std", adv_scalar_gae.std().item(), global_step)

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        writer.add_scalar("gate/ev_latent_cap", ev_latent_cap, global_step)    # (3) ceiling
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)

        # ---- SSL path -------------------------------------------------------------------
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        # THE diagnostic for lever 1. Horizon-1 error alone cannot tell a rollable encoder
        # from a one-step-memorizing one; the RATIO h{K}/h1 is the compounding rate, and it
        # is the number that says whether the AR pressure did anything. If it sits near 1.0
        # the rollout is not actually compounding and the shift is wrong.
        per_h_mean = (ssl_per_h / max(ssl_steps, 1)).tolist()
        for h, val in enumerate(per_h_mean, start=1):
            writer.add_scalar(f"ssl/pred_loss_h{h}", val, global_step)
        if len(per_h_mean) > 1:
            writer.add_scalar(
                "ssl/pred_compound_ratio", per_h_mean[-1] / max(per_h_mean[0], 1e-12), global_step
            )
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
