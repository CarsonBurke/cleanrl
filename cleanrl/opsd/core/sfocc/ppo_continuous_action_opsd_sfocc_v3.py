# OPSD-SFOcc v1 -- the privileged channel becomes a VECTOR OCCUPANCY MEASURE.
# No PPO anywhere: no ratio, no clipped surrogate, no advantage, no GAE, no policy gradient.
# =====================================================================================
# WHY THIS FILE EXISTS
#
# The family charter's §6.3 corollary is the whole programme: "pressure and channel
# informativeness are THE SAME QUANTITY". §6.4 then measured that the champion's channel
# has none. Its own telemetry, verbatim:
#
#   R2(action | credit scalar) = 0.0000        R2(action | s_{t+1}) = +0.4661
#   advantage_std  3.94 -> 14.72  (3.7x LOUDER)    cond_gap 0.030 -> 0.010 (3x LESS USED)
#
# Four unrelated actor-side mechanisms then all plateaued at 8812-8882. That is not four
# coincidences; it is one channel carrying no action-attributable information, four times.
#
# THE CHANGE. Replace the scalar advantage channel with the LONG-HORIZON version of the
# quantity that scored 0.4661, delivered as a 32-dim vector instead of the scalar that
# scored 0.0000. Successor features:
#
#   psi(s,a) = E[ sum_k gamma^k phi(s_{t+k}, a_{t+k}) ],   r ~ w . phi,   V = w . psi
#
# psi is a discounted OCCUPANCY MEASURE and value is LINEAR in it. So
#
#   grad_psi J  =  w      exactly, and CONSTANTLY.
#
# It never shrinks, never vanishes at a local optimum, and never becomes noise-dominated as
# the policy sharpens. Every scalar-gradient mechanism this family has tried does all three:
# that is what the 0.030 -> 0.010 decay IS. The improvement operator here is not a gradient
# at all, it is a HALF-SPACE QUERY: ask the network what it would have done had its future
# footprint come out `delta` better along the fixed direction w.
#
# WHAT IS DELETED RELATIVE TO THE PARENT (ppo_continuous_action_opsd_jacteach_v1.py)
#   - TransitionHead and every jac_* arg / loss / backward / clip / telemetry tag. The
#     parent paid for a PER-SAMPLE action-Jacobian of a learned transition model
#     (torch.autograd.grad inside a chunk loop). That entire cost is gone; the replacement
#     is two forward-only MLP heads.
#   - The HL-Gauss / MTP distributional scalar critic (critic_head, num_bins,
#     critic_mtp_horizon, the bin support, value_ce, b_target_mask, vf_coef). Value is now a
#     linear readout of the vector critic, V(s,a) = w . psi(s,a). A 6x511-bin softmax over
#     32768 rows per pass was also a large share of the update cost.
#   - GAE, `advantages`, `returns`, debug/advantage_std, debug/adv_boost. Deleted rather
#     than left unused: the point of a vector critic is that there is nothing for an
#     advantage to BE. `AdvEmbed` survives because --cond-mode occ_w reuses it.
#   - The parent's inherited 240-line note chain. Most of it documents ladders over
#     advantage conditioning, HL-Gauss and GAE, i.e. machinery this file removes; carrying
#     it would be decoration. The lineage evidence lives in OPSD_FAMILY.md §4/§6 and in the
#     parent file, both unmodified.
#
# WHAT IS UNCHANGED. One network, two contexts. Teacher = student under privileged
# conditioning. Two actor-side losses only, both supervised regressions onto DETACHED
# targets: clone_loss (a negative log-likelihood) and distill_loss (a per-dim clipped KL).
# No target networks, no twin critics, no replay, no ensembles, no simulator probing, no
# counterfactual stepping, no coefficient grids.
#
# psi(s,a) is action-conditioned, which is NOT a Q(s,a) violation of §2: there is no max or
# argmax backup anywhere (the bootstrap is the student's own mean action), no target
# network, no twin, and the scalar readout w . psi is never used as a baseline, never
# multiplies a gradient, and never appears in an actor loss. It exists only to give the
# occupancy channel a reward-aligned direction.
#
# ------------------------------------------------------------------------------------
# THE TWO HAZARDS, ENGINEERED AGAINST RATHER THAN HOPED AWAY
#
# 1. ACTION LEAKAGE / IDENTITY COLLAPSE. clone_loss is -log p(a_t | s_t, c_t). If a_t is
#    recoverable from c_t the fit collapses to the identity map -- a Dirac at a_t -- driving
#    the loss to -inf while teaching nothing, and making the teacher exactly the rollout
#    policy (see the parent's own warning, its lines ~327-333). The NAIVE channel
#    `tgt - psi(s,a)` fails this immediately: `tgt` contains phi(s_t, a_t) EXPLICITLY, so
#    the network can invert phi and read the action off. The channel built below is
#    FUTURE-ONLY, so a_t enters solely through the real environment transition s_{t+1}.
#    That is precisely the legitimate hindsight privilege §6.4 measured at +0.4661, and it
#    is the difference between privileged information and leakage. Gate: sf/cond_action_r2.
#
# 2. PHI COLLAPSE. If phi is trained only to predict reward it converges toward reward
#    itself, psi degenerates into a scalar critic in disguise, and the geometry evaporates.
#    The instrument is sf/phi_eff_rank -- and it is NECESSARY, NOT SUFFICIENT: a
#    decorrelation penalty prevents RANK collapse but not UNINFORMATIVENESS, because
#    uncorrelated noise has full rank. High rank does not license the mechanism; low rank
#    kills it.
#
# ------------------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS
#
#  P1 CHANNEL. sf/chan_action_r2 >= 0.15 by 65k steps, AND materially above the state-only
#     control sf/state_action_r2 -- otherwise the number is just the policy sharpening, and
#     §6.4's own method quotes the GAIN over state alone for precisely that reason. §6.4
#     measured the SCALAR version of this at 0.0000. If the vector version is also ~0, the
#     hypothesis is dead and no GPU slot is warranted. This is the single most informative
#     number in the file.
#  P2 NO LEAKAGE. sf/cond_action_r2 <= 0.9 and losses/clone_nll not diverging to -inf.
#  P3 NO COLLAPSE. sf/phi_eff_rank > 4 of 32; sf/psi_r2 > 0.5; sf/w_r2 > 0.5.
#  P4 TEACHER IS NOT THE STUDENT. debug/cond_gap and losses/distill_kl both materially
#     above 0. The parent recorded cond_gap 0.001 / distill_kl 0.0001 as the signature of a
#     no-op channel.
#  P5 SPEED. >= 6000 SPS at 16 envs (the parent measures 4300), and state which env count
#     reaches ~10k.
#
# WHAT 65,536 STEPS CAN AND CANNOT TEST
#   CAN: P1, P2, P4, P5, the non-collapse half of P3 (eff_rank, w_r2), NaN/Inf freedom, and
#        the scalar-vs-vector ablation (--cond-mode occ_w) since everything else is
#        byte-identical.
#   CANNOT: (a) return. 65k steps is ~2% of the 8M benchmark; the parent's own numbers
#        (994 @500k) show the early curve does not rank arms. (b) sf/psi_r2 as an
#        asymptotic claim -- psi's regression target is a lambda-return whose fixed point
#        has magnitude ~1/(1-gamma*lam) times phi's, and the head starts at std=0.01, so a
#        16-iteration run measures the START of that climb, not its end. (c) THE DECAY,
#        which is the thing this file exists to fix. cond_gap collapsing 0.030 -> 0.010 took
#        the parent 1.75M steps. Channel decay is structurally untestable at this length
#        and needs a long run. A flat cond_gap over 16 iterations is not evidence of a fix.
#
# ------------------------------------------------------------------------------------
# VERIFIED IN THIS FILE. Real measurements, this disk, HalfCheetah-v4, seed 1, CUDA. THE
# GPU WAS SHARED throughout with an unrelated 9.7 GiB training job holding ~92%
# utilization, so every SPS figure is a LOWER bound.
#
#   GRADIENT ISOLATION, MEASURED BOTH WAYS. After sf_loss.backward(): 0 of 227
#   agent.parameters() have a non-None grad, and the agent's total grad norm is exactly
#   0 (printed at 17 significant digits). After the actor's clone_loss.backward(): 0 of 14
#   sf.parameters() have a grad, and all 227 agent params do. psi's target, the channel and
#   the query all carry requires_grad=False. So "the occupancy channel helps" cannot be
#   confounded with "predicting the future is a good auxiliary task".
#
#   65,536 steps, --num-envs 16 --num-steps 256 --num-minibatches 32 --actor-epochs 4
#   --critic-epochs 4 == 16 iterations, 512 actor steps. START = iteration 1 (identical
#   fresh policies, entropy -0.48 in both arms), END = iteration 16. Same seed, one flag
#   apart. NO NaN or Inf in ANY tag of ANY run.
#
#     tag                      occ START -> END        occ_w START -> END
#     sf/chan_action_r2          0.0655 ->  0.2240       0.0032 ->  0.0003     <- P1
#     sf/state_action_r2         0.0036 ->  0.0477       0.0036 ->  0.0933     <- P1 CONTROL
#     sf/cond_action_r2          0.0838 ->  0.2840       0.0082 ->  0.0939     <- P2
#     sf/phi_eff_rank           15.2760 -> 20.8564      15.3873 -> 22.7740     <- P3
#     sf/psi_r2                 -3.2827 -> -1.7557      -1.6103 ->  0.1142     <- P3 UNMET
#     sf/psi_bias_frac           0.7608 ->  0.5974       0.6570 ->  0.3018
#     sf/w_r2                   -0.6799 ->  0.9563      -0.9716 ->  0.8948     <- P3
#     sf/w_norm                  0.4965 ->  0.5122       0.5192 ->  0.5484
#     sf/phi_offdiag_absmean     0.1410 ->  0.0374       0.1392 ->  0.0407
#     sf/delta                   0.5491 ->  0.7489       0.9228 ->  1.2623
#     sf/query_support           0.0973 ->  0.1769       0.9999 ->  1.0211
#     sf/teacher_conc_ratio      0.9963 ->  1.0021       0.9923 ->  1.0313
#     debug/cond_gap             0.0106 ->  0.0081       0.0123 ->  0.0175     <- P4
#     losses/distill_kl          0.0110 ->  0.0122       0.0165 ->  0.0442     <- P4
#     losses/clone_nll          -0.6664 -> -4.0348      -0.5683 -> -1.2801     <- P2
#     losses/entropy            -0.4769 -> -3.3800      -0.4830 -> -1.2834
#     losses/explained_variance -0.0001 ->  0.2812       0.0001 ->  0.5176
#     losses/sf_psi_mse         26.4473 ->  5.0885      18.8971 -> 16.5239
#     losses/sf_rew_mse          0.1135 ->  0.0056       0.1013 ->  0.0280
#     losses/sf_cov              0.0286 ->  0.0100       0.0279 ->  0.0097
#
#   P1 PASSES, AND IT IS THE RESULT. Read against the state-only control, which is what
#   makes it a claim rather than an artifact:
#
#     arm     R2(a | channel)   R2(a | state)   channel / state
#     occ         0.2240           0.0477            4.7x   (3.6x vs the state's own 0.0629
#                                                            run maximum -- still decisive)
#     occ_w       0.0003           0.0933            0.003x
#
#   The VECTOR channel carries 4.7x what the state carries. The SCALAR channel carries
#   LESS THAN THE STATE -- it is, to three decimals, nothing, which reproduces §6.4's
#   R2(action | credit scalar) = 0.0000 in a completely different encoding of a completely
#   different quantity. At iteration 1, with matched fresh policies, the same ordering
#   holds unconfounded: 0.0655 vector vs 0.0032 scalar against 0.0036 state-only.
#
#   AN INDEPENDENT CONFIRMATION FELL OUT OF THE LIKELIHOODS, and it is worth more than the
#   R2 because it is measured inside the loss the network actually minimizes. -clone_nll is
#   the mean log-density in the PRIVILEGED context; -entropy is the same quantity for the
#   unconditioned marginal. The difference is exactly what the channel buys:
#     occ:    4.0348 - 3.3800 = +0.655 nats
#     occ_w:  1.2801 - 1.2834 = -0.003 nats
#   §6.4's sentence was "a channel that predicts nothing about the action is a channel MLE
#   drops for free". occ_w drops it for free. occ does not.
#
#   P2 PASSES. cond_action_r2 0.2840, far below the 0.9 threshold, and clone_nll is not
#   diverging: at student entropy -3.3800 a perfectly calibrated conditional would read
#   about -3.38, and it reads -4.03, i.e. 0.65 nats of channel gain, not a Dirac. The
#   FUTURE-ONLY construction is doing its job -- phi(s_t, a_t) cancels algebraically out of
#   the channel and a_t survives only through s_{t+1}.
#
#   P3 PASSES ON TWO OF THREE. STATED PLAINLY: psi_r2 IS UNMET, NOT PASSED.
#     - phi_eff_rank 20.86 of 32, NET-RISING from 15.28 (peak 23.44 mid-run, so this is a
#       rise-then-settle, not a monotone climb). No rank collapse; hazard 2 is not live at
#       this length. phi_offdiag_absmean falls 0.1410 -> 0.0374 monotonically, so the
#       decorrelation term is working. NEITHER licenses phi as informative (uncorrelated
#       noise has full rank) -- they only exclude one specific failure.
#     - w_r2 -0.6799 -> 0.9563 with sf_rew_mse 0.1135 -> 0.0056. Reward is linear in phi to
#       96%, which is the assumption the whole construction rests on. Emphatic.
#     - psi_r2 -1.7557 MISSES the >0.5 bar. It NET-rises from -3.2827 but non-monotonically
#       (range -4.5996 to -0.7081 over the 16 iterations), and psi_bias_frac shows
#       60% of the remaining error is pure per-dim LEVEL error, i.e. psi is still climbing
#       toward its target's mean rather than failing to fit its variation. That is the
#       predicted-untestable item: 16 iterations is 512 SF steps against a target whose
#       fixed point is ~1/(1-gamma*lam) times phi's, from a head initialized at std=0.01.
#       IT IS RECORDED AS UNMET. A long run must clear it or the occupancy claim is only
#       half-built. The level does NOT contaminate the channel -- c is a difference of two
#       future footprints, so the level cancels -- which is why P1 can pass while this
#       misses, and losses/explained_variance rising -0.0001 -> 0.2812 says the reward-
#       relevant projection w . psi is being learned first.
#
#   P4 PASSES ON LEVEL AND MY READING OF IT WAS WRONG. occ's cond_gap 0.0081 and
#   distill_kl 0.0122 are 8x and 122x the parent's recorded no-op signature (0.001 /
#   0.0001), so the teacher is not the student. cond_gap NET-FELL 0.0106 -> 0.0081, which
#   sits inside its own 0.0072-0.0147 range over 16 iterations and therefore CANNOT be read
#   as the parent's decay -- that was pre-registered as untestable at this length.
#   BUT I ALSO PREDICTED occ_w WOULD SHOW THE NO-OP
#   SIGNATURE AND IT DOES NOT -- it reads HIGHER on both (0.0175 / 0.0442). The cause is
#   mechanical: AdvEmbed's fixed frequencies run to 8, so a constant query of ~1.26 moves
#   16 Fourier channels a long way, while occ's query moves 32 channels by delta * w_unit_i,
#   which is small per dim (query_support 0.18). So cond_gap and distill_kl measure the
#   AMPLITUDE OF THE TRUNK'S RESPONSE TO THE QUERY, not the information in the channel.
#   occ_w responds MORE while carrying ~750x LESS action information. P4 is therefore a
#   no-op detector and NOT a scalar-vs-vector discriminator; that job belongs entirely to
#   sf/chan_action_r2 and the clone_nll-vs-entropy gap. Recording this because it would
#   otherwise be a very easy number to quote in the flattering direction.
#
#   ALSO MEASURED, ALSO NOT FLATTERING. teacher_conc_ratio 0.9963 -> 1.0021: the teacher is
#   0.2% more concentrated than the student, i.e. the query SHIFTS the policy and barely
#   SHARPENS it. The family's observation that mass-covering distillation against a
#   fixed-concentration teacher has no sharpening channel is confirmed here, on this
#   mechanism, quantitatively. And query_support 0.0973 -> 0.1769 confirms the structural
#   risk documented in BLOCK D: the occ query sits at ~18% of the channel's own RMS norm
#   because it zeroes every component orthogonal to w. Both are the `aspire` sibling's
#   brief, and neither is hidden here.
#
#   P5 PASSES. Batch held at 32768 across the sweep so only env count varies (16x2048,
#   32x1024, 64x512), --total-timesteps 65536, 3 repeats each. Cumulative includes the
#   one-off torch.compile warmup; marginal is between iterations 1 and 2, the same
#   "marginal end-to-end" convention the parent quotes:
#
#     envs   cumulative @65536 (min-max)   marginal      parent
#       16      9451  (8875-9799)           11678         4300
#       32     10222  (9728-10693)          13664            --
#       64     12327 (11282-12858)          17035     6013 @128 envs
#
#   2.2x the parent at 16 envs, and past the parent's 128-env figure with 16. ~10k SPS is
#   reached AT 16 envs on the cumulative measure and comfortably at 32-64; nothing further
#   is needed. Peak VRAM 0.218 GiB allocated / 0.277 GiB reserved, flat across all three
#   env counts. Deleting the per-sample autograd Jacobian and the 6x511-bin critic softmax
#   is where this came from, and per §3 that is a diagnostic (no search was smuggled in),
#   not an achievement.
#
#   THE LITERAL ACCEPTANCE COMMAND (--num-envs 16 --total-timesteps 65536, file defaults,
#   so num_steps 2048 == only 2 iterations and 64 actor steps) also runs clean, 9408 SPS:
#   chan_action_r2 0.0858 -> 0.1270 against state_action_r2 0.0007 -> 0.0011 (a 115x ratio,
#   but SHORT OF P1's 0.15 absolute bar at 64 gradient steps), cond_action_r2 0.1077 ->
#   0.1562, phi_eff_rank 15.36 -> 21.79, w_r2 -0.678 -> 0.873, psi_r2 -3.36 -> -2.62,
#   cond_gap 0.0094 -> 0.0083, distill_kl 0.0092 -> 0.0063, EV -0.0000 -> -0.0492.
#   The 16-iteration configuration above is reported as primary because 2 points cannot
#   show a trend.
#
#   NULL CONTROL --distill-coef 0 (occ, 16 iterations): machinery ALIVE and INERT, exactly
#   as intended. Channel still measured: chan_action_r2 0.0655 -> 0.0654, phi_eff_rank
#   15.28 -> 16.68, w_r2 -0.68 -> 0.9847, psi_r2 -3.28 -> -0.7778. No distillation
#   pressure: distill_kl runs AWAY, 0.1085 -> 0.8078, because nothing pulls the student
#   toward the teacher, and cond_gap grows 0.0304 -> 0.0425 for the same reason. The clone
#   term alone collapses entropy to -5.4160 (clone_nll -5.4497) against the distilled arm's
#   -3.3800, so distillation is measurably restraining the collapse rather than causing it.
#   Note this arm's channel R2 (0.0654) sits BELOW its state-only control (0.2926): its far
#   sharper policy inflates the control while leaving the channel flat, which is the
#   opposite direction to a sharpening artifact and so is a third independent argument that
#   P1 is not one.
#
#   ===================================================================================
#   v2 CHANGE (this file): psi regresses in STANDARDIZED target space.
#   ===================================================================================
#   v1's ONLY unmet gate was P3's psi_r2 > 0.5, which read -1.7557 with psi_bias_frac
#   0.5974. That is 60% pure per-dim LEVEL error, and the cause is arithmetic: phi's
#   LayerNorm centers ACROSS dims, so each dim keeps a nonzero mean, and the lambda-return
#   multiplies it by 1/(1 - gamma*lam) ~ 17.5, while psi_net starts at std=0.01 and outputs
#   ~0. The head had to travel ~17 per dim on 512 Adam steps before R2 could reach 0.
#
#   v2 supplies the level from an EMA buffer and regresses only the variation:
#     psi_std(s,a)           -- network output, standardized space; what the loss compares
#     psi(s,a)               -- psi_std * sigma + mu; RAW units; unchanged semantics for
#                               every consumer (BLOCK B bootstrap, BLOCK C channel, value)
#     loss                   -- mse(psi_std, (tgt - mu)/sigma), i.e. v1's raw MSE reweighted
#                               per dim by 1/var, so loud dims stop owning the gradient
#   At init the head predicts the target's MEAN, so psi_r2 starts at ~0 rather than ~-3.3.
#
#   WHY THIS IS NOT COSMETIC, AND WHY IT SHOULD HELP P1 TOO. v1's own BLOCK C docstring
#   records that "the level cancels" in the channel is an 83% cancellation, not an identity:
#   c = gamma*nonterm*boot - (psi - phi) weights the level by gamma*nonterm on one side and
#   by 1 on the other, so psi's level error leaks into the channel as a bias and inflates
#   c_ms. Fixing the level should therefore make the channel cleaner, not just the critic.
#
#   PRE-REGISTERED FOR v2, against v1's measured numbers at the identical 16-iteration
#   configuration (--num-envs 16 --num-steps 256 --num-minibatches 32 --actor-epochs 4
#   --critic-epochs 4 --total-timesteps 65536 --seed 1, cond-mode occ):
#     - psi_r2       v1 -1.7557 -> v2 must be > 0.5 (P3's bar) to call P3 fully passed.
#     - psi_bias_frac v1 0.5974 -> v2 must fall well below it; the level is now free.
#     - chan_action_r2 v1 0.2240 -> v2 must NOT regress. This is the load-bearing gate: v2
#       touches only psi's parameterization, so a drop here means the standardization
#       perturbed the channel and the change should be reverted.
#     - SPS v1 9451 cumulative @16 envs -> v2 within noise; the change is two buffers.
#   A sawtooth in psi_r2 synchronized with the EMA would indicate the missing PopArt
#   compensation matters; that is the documented escalation and it is NOT implemented here.
# =====================================================================================
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
def sf_query(sf, c, occ_boost):
    """c_query = delta * w_unit, with delta = occ_boost * std(w_unit . c) over the batch.

    SEMANTICS: "what would you have done if your future footprint had come out `delta`
    better along the reward-increasing direction than you predicted." The channel is
    already a RESIDUAL with a meaningful zero -- c = 0 means "exactly as predicted" -- so
    an ABSOLUTE query in c-space is a PER-STATE RELATIVE request in footprint space. That
    is what v1 lacked (a global constant query in absolute advantage units, which was
    3-sigma extrapolation for badly-scoring states) and what v2 had to hack around by
    adding a margin to each state's own realized delta. Here the parameterization gives it
    for free.

    WHY THIS DOES NOT VANISH. Value is LINEAR in psi, so the improvement direction is
    exactly w -- constant, and independent of how sharp the policy has become. Every
    scalar-gradient mechanism in this family carries the opposite property: the parent's
    channel got 3.7x louder and 3x less used over 1.75M steps because its direction was a
    score function whose signal-to-noise decays as the policy concentrates. There is
    nothing here to decay.

    WHY delta IS MEASURED AGAINST THE CURRENT BATCH. A fixed absolute target is absorbable:
    once the policy reliably achieves it, the request stops being a request. Quoting delta
    in units of the batch's own spread along w means the reference moves as the policy
    improves. occ_boost = 1.0, one value, no sweep (§2).

    KNOWN STRUCTURAL RISK, INSTRUMENTED RATHER THAN HIDDEN. ||c_query|| = delta ~ 1 while a
    typical observed ||c|| ~ sqrt(phi_dim), because the query zeroes every component
    orthogonal to w. So in occ mode the query sits in a LOW-DENSITY region of the channel:
    not a tail extrapolation like v1's, but an unusually small-norm one. sf/query_support
    reports ||c_query|| / rms||c|| for exactly this, and it MEASURES 0.0973 -> 0.1769 over
    16 iterations, i.e. the occ query really does sit at ~18% of typical channel magnitude.
    It reads ~1 in occ_w (one dimension, nothing to zero) -- not exactly 1, because the
    numerator uses the UNCLAMPED delta while the denominator is the RMS of the CLAMPED
    channel, which is why the measured occ_w series is 0.9999 -> 1.0211 rather than a flat
    1.0. That asymmetry is genuinely in occ_w's favour and is disclosed rather than buried.
    Recalibrating delta against achievement is the `aspire`
    sibling's entire job.
    """
    w = sf.w_vec()
    w_unit = w / w.norm().clamp_min(1e-8)
    delta = occ_boost * (c @ w_unit).std()
    return delta * w_unit, float(delta.item()), w_unit


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
        c_query, sf_delta, w_unit = sf_query(sf, b_c, args.occ_boost)

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
        writer.add_scalar("sf/query_support", query_support, global_step)
        writer.add_scalar("sf/teacher_conc_ratio", float(np.mean(conc_ratios)), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
