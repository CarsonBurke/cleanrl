# PPO + IterThink v217 post-temporal SIGReg.
#
# Preserve v216's fresh-interface joint imagination PPO and exact Beta trust
# region, but move SIGReg to the representation it must keep non-collapsed:
# the action-conditioned transition features produced after the causal temporal
# trunk and immediately before the shared next/MTP projectors. v216 inherited
# encoder-side SIGReg, so the affine frame embeddings were forced Gaussian before
# the temporal model could reorganize them while the consumed predictive features
# received only indirect pressure through MSE. v217 regularizes only observation
# feature slots (outcome queries remain free to represent near-constant
# continuation), uses the same per-step validity mask, and removes the synthetic
# z0 anchor because every regularized row now corresponds to an actual outbound
# transition. Gradients remain attached through query readout, temporal trunk,
# and encoder.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v216 affine-RMS fresh-interface joint imagination PPO.
#
# Previous header:
# PPO + IterThink v213 affine-RMS policy-moment outcomes.
#
# Preserve v198's raw affine-RMS encoder, attached LeJEPA/SIGReg, action-free
# temporal state, outbound transition query, exact-Beta PPO/KL, persistent 4D/64
# exploration, and imagination. A separate WM-owned, action-conditioned query
# reads temporal state memory and forecasts the successor pure-base policy as a
# valid Beta distribution. It trains by exact forward KL to fixed rollout-version
# targets; no policy gradient enters the WM and no target gradient enters the
# agent. The derived native mean + clamped log-std is carried into s_(t+1),
# detached, through zero-init actor/critic residual adapters that stay disabled
# for 100k steps. Reset states receive a zero token; normal, time-limit-final,
# and imagined successors receive the forecast from (s_t,a_t). A second exact-KL
# trust anchor protects the pure v198/base policy independently of the behavior
# policy KL. All new modules use forked RNG, preserving every v198 parameter and
# initial policy/value output exactly.
#
# -----------------------------------------------------------------------------
# PPO + IterThink v198 affine-RMS temporal-query world model.
#
# Keep v197's raw affine-RMS observation encoder, attached encoder-side SIGReg,
# persistent Beta exploration, plain GAE, and exact-Beta-KL trust region. Replace
# the action-conditioned STST belief with two action-free causal temporal blocks.
# PPO stores their detached current-frame tokens and trains separate actor/critic
# query readouts. A distinct WM-owned query decoder receives the outbound action
# only after it is chosen and predicts the next state, reward, and continuation.
# This removes the neutral-action policy query and all post-SIGReg spatial
# self-attention while preserving the proven observation encoder.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v197 affine-RMSNorm Beta exploration + exact-KL trust region.
#
# Preserve v196, but replace its epoch-end sampled-action KL check with the exact
# conditional KL between stored behavior Beta distributions and the current policy.
# Check before every real-buffer actor minibatch. Once the trust budget is exhausted,
# stop reusing that behavior batch and freeze the shared trunk against its remaining
# critic gradients; the critic head may keep fitting. Fresh dream blocks are regenerated
# after every step but share the same real-state KL budget. Every real or dream actor step
# is rolled back together with its Adam state if a fixed real-state anchor breaches the
# budget. This prevents v196's stale-batch overshoot without discarding safe dream updates.
#
# -----------------------------------------------------------------------------
# PPO + IterThink v145 v129-noobsnorm-embednorm-k6 (v144 with input embed norm).
#
# Key change from v129:
#   - Add input embedding RMSNorm after v144's per-feature affine observation
#     tokens and semantic tokens are concatenated. Everything else remains v144:
#     no env obs norm/clip, symmetric reward/critic supports, rolling k=6 prompt
#     bank, and per-feature affine observation tokens.
#
# Hypothesis:
#   Raw observations make the per-feature affine token stream poorly scaled. A
#   standard transformer input embed norm may recover stable geometry without
#   restoring Gym's running observation normalizer.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v128 symsupport (v121 + symmetric reward/critic supports).
#
# Key change from v121:
#   - Keep the v120/v121 outcome-token algorithm unchanged, but make both scalar
#     distribution supports symmetric in raw space before symlog:
#       critic return support: [-20000, 20000]
#       WM reward support:    [-50, 50]
#   - Uniform/high-entropy logits now decode to zero for both reward and value
#     under the consumed E[symexp(z)] scalar path, removing the asymmetric
#     off-manifold optimism prior while preserving zero-projection initialization.
#
# Hypothesis:
#   v120 already initializes reward/value heads to zero, but broad off-manifold
#   dream distributions can still decode optimistic because the supports are
#   asymmetric. Symmetric supports should make uncertainty neutral and reduce
#   dream bootstrap artifacts without changing the losses or token layout.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v121 metriccleanup (v120 + honest dream metrics / diagnostics).
#
# Key change from v120:
#   - Keep the algorithm unchanged, but clean up misleading reward diagnostics.
#     Actual consumed dream statistics stay under `imagine/*`; teacher-forced
#     supervised reward calibration stays under `lewm/*`; labeled AR diagnostics
#     move under `diagnostics/*`.
#   - The old length-1 AR reward probe is renamed to
#     `diagnostics/length1_ar_*`, because it is a stress test rather than the
#     distribution consumed by imagination.
#   - Add `diagnostics/warm_ar_*`, a labeled AR rollout that starts from the same
#     full real context window used by actual imagination. This is the right
#     labeled check for whether dream rewards are calibrated on consumed states.
#
# Hypothesis:
#   v120's negative `imagine_ar_reward_bias` was partly a bad metric: actual
#   imagination is warm-started from real context, while the diagnostic started
#   from a single state. v121 makes the dashboard distinguish consumed dream
#   health from labeled stress tests without changing learning.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v120 outcometokens (v119 + dedicated reward/continue readout tokens).
#
# Key change from v119:
#   - Reward and continuation heads no longer flatten/read the whole predicted
#     summary. The two semantic slots are explicitly treated as reward and
#     continuation readout tokens; reward CE reads token 0, continuation BCE reads
#     token 1. Action conditioning is projected into token width for the reward
#     token only.
#
# Hypothesis:
#   v119 fixed the Bellman scalar decode, but outcome gradients were still
#   entangled across all obs and semantic tokens through flattened heads. Dedicated
#   readout tokens should make outcome roles identifiable, route CE/BCE gradients
#   more cleanly, and reduce reward/continue interference with obs latent MSE.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v119 bellmandecode (v118 + Bellman scalar expectation decode).
#
# Key change from v118:
#   - Keep symlog HL-Gauss CE targets and finite zero log-priors, but consume
#     reward/value scalars as E[symexp(z)] for Bellman quantities. The
#     hl-gauss-pytorch decode symexp(E[z]) round-trips deterministic transformed
#     targets, but under broad/off-manifold distributions it is a certainty
#     equivalent, not the raw reward/return expectation needed by GAE.
#
# Hypothesis:
#   v118 cleaned up the library HL-Gauss contract but broke imagined reward means
#   when self-fed dreams produced broad/mixed reward logits. v119 restores the
#   RL-consumed scalar to the expected raw reward/return while retaining v118's
#   symlog bucket density and zero-prior initialization.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v118 hlgaussdecode (v117 + library-style symlog HL-Gauss decode).
#
# Key change from v117:
#   - Critic and reward scalar consumption now use the hl-gauss-pytorch decode:
#     inverse_transform(E[transformed bin]), i.e. symexp(E[z]) for symlog.
#     v117 used E[symexp(z)], which is an expected scalar value but not the
#     HL-Gauss transform/inverse-transform contract.
#
# Hypothesis:
#   Matching the library decode makes projected symlog HL-Gauss targets decode
#   back to their original scalar values and removes Jensen bias from broad
#   initial/category distributions, while retaining v117's finite zero priors.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v117 zerocriticprior (v116 + explicit symlog scalar decode + critic zero prior).
#
# Key change from v116:
#   - The critic head now gets a finite zero-return log-prior. With asymmetric
#     symlog return support [-2000, 20000], uniform
#     critic logits decode to a large positive scalar value; initializing to
#     project(0) removes that early GAE/bootstrap bias.
#   - Critic and reward scalar consumption used HLGaussSupport.to_expected_scalar,
#     the explicit E[symexp(bin)] decode. v118 switches that to the library-style
#     HLGaussSupport.to_scalar decode.
#
# Hypothesis:
#   v116 fixed reward drift from the asymmetric reward bucket prior, but left the
#   critic with the same asymmetry in a larger return support. A finite log
#   project(0) value prior should clean early advantages and imagined bootstraps
#   without changing the reward/continue MTP objective.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v116 symlogrew-initbias (v115 + symlog reward support + zero prior).
#
# Key change from v115:
#   - WM reward HL-Gauss support is now symlog-coordinate over raw reward range
#     [-20, 50]. v116/v117 decode details are superseded by v118's library-style
#     HL-Gauss decode above.
#   - The reward bucket head is initialized to a zero-reward log-prior:
#     final weights are zero and bias is finite log(project(0)). This removes the
#     old implicit uniform linear-support midpoint prior (~+15 reward) without
#     making early nonzero rewards pay enormous CE against zero-tail logits.
#
# Hypothesis:
#   v115 fixed the MTP consumed-path issue, but the linear asymmetric reward
#   support still starts badly biased and is coarse near typical rewards. Symlog
#   support plus a zero-reward bucket prior should improve early reward scale and
#   reduce scalar drift in dreams.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v115 mtpoutcomes (v114 + MTP reward/continuation).
#
# Key change from v114:
#   - Reward/continuation are MTP again. Each predicted summary projection
#     pred_mtp[..., k] is decoded with CE/BCE against the reward/continue label
#     k steps ahead, using the same shifted masks as the previous MTP outcome
#     objective.
#   - Imagination consumes the k=0 projected-summary outcome path, so the scalar
#     reward used by dreamed PPO is trained by the same CE/BCE projection head.
#
# Hypothesis:
#   v114 correctly moved outcomes onto the consumed transition path, but making
#   rewards one-step-only discarded the stabilizing MTP supervision. v115 keeps
#   the consumed path aligned while restoring multi-horizon CE/BCE.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v114 beliefoutcomes (v113 + D4-style consumed reward path).
#
# Key change from v113:
#   - Reward/continuation are decoded from the same action-conditioned predictor
#     belief representation that drives one-step imagination, not from the
#     predicted next-summary semantic readout tokens. The CE/BCE loss now trains
#     exactly the scalar path consumed by imagined PPO. AR rollout remains a
#     no-grad diagnostic only; there is no auxiliary state-prediction loss and no
#     separate target-summary reward probe objective.
#
# Hypothesis:
#   v113 fixed gradient-carrying AR training but left reward calibration on a
#   loosely constrained semantic readout. Decoding transition outcomes directly
#   from the consumed belief + action path should reduce scalar reward drift in
#   dreams without reintroducing off-policy self-fed training.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v113 d4detached (v112 with AR rollout restored to diagnostics only).
#
# Change from v112:
#   - Dreamer4-style detached WM training: the autoregressive self-fed rollout is
#     no longer a gradient-carrying loss path. It remains as the imagine_ar_*
#     diagnostic computed under no_grad, while the optimized WM reward/continuation
#     losses stay on the predicted MTP path from v108. This avoids keeping H full
#     predictor graphs alive per WM minibatch; Dreamer4 similarly generates dreams
#     under no_grad and trains reward/value from one main forward, not from a
#     self-fed full-model rollout loss. Default WM minibatch 1024 should fit again
#     without gradient accumulation.
#
# PPO + IterThink v112 no-ar-sigreg (v111 with the AR SIGReg term removed entirely).
#
# Change from v111:
#   - DROP AR SIGReg. The encoder-side SIGReg stays; the AR predictions get their
#     on-manifold pressure solely through the per-step attached MSE toward SIGReg'd
#     encoder targets. Rationale: (a) redundancy -- the anchor already transmits the
#     geometry; (b) it fights the MSE-optimal predictor: the conditional mean
#     E[z|history] has a strictly lower-variance marginal than the target marginal
#     (law of total variance), so forcing the prediction marginal to match the
#     target's isotropic-Gaussian statistic demands hallucinating the unpredictable
#     component. Counter-hypothesis this ablation tests: in a deterministic-latent
#     WM, that variance forcing usefully counteracts regression-to-the-mean
#     shrinkage of deep rollouts (dream over-smoothing) and is part of why v111 ran
#     fast. v111 vs v112 at matched steps decides.
#
# PPO + IterThink v111 arsigreg-obsonly (v110 with AR SIGReg off the semantic tokens).
#
# Change from v110:
#   - AR SIGReg covers OBS TOKENS ONLY, matching the encoder-side pool (which never
#     included semantic tokens). v110 SIGReg'd the full AR summary; measured harm:
#     continuation BCE ~20x worse than v109 at matched steps (0.0114 vs 0.00059
#     @700k) and reward MAE mildly worse (0.30 vs 0.17 @500k). Mechanism: HalfCheetah
#     never terminates, so the optimal continuation token is a CONSTANT embedding
#     (zero batch variance) -- maximally anti-Gaussian; SIGReg force-injects variance
#     that the linear unproj turns into logit noise. The reward token fights the same
#     pressure partially. Obs tokens keep AR SIGReg: their MSE anchor points at
#     SIGReg'd encoder targets, so the two losses agree there (and v110's returns
#     were well ahead of baseline at matched steps: 1277 vs 1136 @1M).
#
# PPO + IterThink v110 artrain-detached (v109 with the through-time chain cut).
#
# Change from v109:
#   - NO BPTT: the AR rollout's seed summary and every fed-back prediction are
#     DETACHED. Each step h trains only its own predictor application + outcome
#     decode, DAgger-style: predict well GIVEN a corrupted (self-generated)
#     history. The multi-step-consistency gradient ("be a good input for step
#     h+1") is dropped -- it is largely redundant with the per-step MSE anchor
#     that already pins every step's output to the encoder target, and through-
#     time gradients in latent rollouts are unstable. The MSE TARGETS stay
#     attached (encoder feedback preserved, as in the teacher-forced path).
#   - Practical driver: v109's joint backward retained 5 chained growing-context
#     predictor graphs and OOM'd marginally at ~25 GiB even at microbatch 512.
#   - SIGREG ON THE AR PREDICTIONS: the AR-predicted summaries are the population
#     imagination actually consumes, but previously got isotropy pressure only
#     secondhand (per-step MSE toward SIGReg'd encoder targets). Now they enter
#     their own per-step masked SIGReg term (lewm/ar_sigreg), scaled under the
#     then-present AR loss coefficient. FULL token set: fed-back obs tokens AND the semantic
#     outcome tokens -- the latter are embeddings read out by decode_outcomes
#     (only their decoded reward/continuation logits are categorical), so they
#     get the same treatment as obs tokens.
#
# PPO + IterThink v109 artrain (v108 + the AR rollout is a training path).
#
# Change from v108:
#   - TRAIN ON THE PATH IMAGINATION CONSUMES: the autoregressive rollout along the
#     real action sequence (previously a no-grad probe) now carries gradient. At
#     each step h: latent MSE vs the (attached) real target summary, reward CE vs
#     the projected scalar label, continuation BCE. Weighted by the then-present AR loss coefficient
#     (0.5) on top of the unchanged teacher-forced losses.
#   - ATTACHED, le-wm style: the seed encoder summary and every fed-back predicted
#     summary stay in-graph (no detach anywhere, mirroring the attached obs context
#     of the teacher-forced path). Gradient through the rollout teaches step h to be
#     a good INPUT for step h+1 (multi-step consistency) and lets the encoder hear
#     about rollout failures. Per-step losses inject gradient at every depth, so
#     effective backprop path lengths are 1..H with most mass on short paths
#     (vanishing-gradient risk low at H=5); each step's own latent MSE pins outputs
#     to the real manifold (bounds self-cooperation drift).
#   - Root cause being fixed (measured on the v105 baseline): exposure bias. The
#     predictor was trained only teacher-forced but consumed autoregressively;
#     AR latent MSE grew 0.13->0.55 and AR reward MAE 0.066->0.229 over h=1..5
#     (near-linear drift accumulation), while h=1 AR MAE matched teacher-forced
#     (0.066 vs 0.059) -- the decoder was fine; the latents drift. DAgger-style:
#     train the one-step map on the input distribution it generates, labels from
#     real data.
#   - The per-horizon imagine_ar_* probes are now TRAIN metrics (same pass), no
#     longer held-out; dream quality and returns are the unbiased signals.
#
# PPO + IterThink v108 predoutcomes-only (v105 + remove target-outcome training).
#
# Change from v105:
#   - Train reward and continuation only on the predicted-summary path consumed by
#     imagination. The target-summary outcome CE/BCE is kept as a no-grad
#     diagnostic, but no longer shapes the shared outcome decoder or encoder.
#     This isolates the suspected two-path mismatch: real encoded future summaries
#     were learning an easy reward readout while dreams consume predicted summaries.
#
# PPO + IterThink v105 raw-reward symlog20k (v104 + no reward norm).
#
# Change from v104:
#   - Disable NormalizeReward and reward clipping so WM/imagination train on raw
#     HalfCheetah rewards rather than stateful normalized rewards.
#   - Move the critic HL-Gauss support to symlog coordinates for a raw-return
#     linear range of [-2k, +20k]: [symlog(-2000), symlog(20000)].
#   - Widen the LeWM immediate reward support to a raw per-step HalfCheetah range
#     so imagined rewards are not clipped by the old normalized [-3, 3] head.
#   - Reward head: softmax HL-Gauss CE over the raw support, scalar decode for
#     consumption (same pattern as the critic). A cumulative-CDF/BCE variant was
#     tried (archived as _v105_cdfrew_symlog20k): better loss geometry won early
#     (led at every matched step to ~2.7M) but per-threshold BCE meaned over 100
#     thresholds dilutes the fit -- decoded reward MAE converged 2.7x worse
#     (0.27 vs 0.10) and returns plateaued ~3.7k vs 6.2k here.
#   - Dead-plumbing removal: encode_summary_with_outcomes ignored its
#     reward_probs/continuations args entirely (verified vestigial since v92 made
#     outcomes readout-only), so the prev_reward_probs/prev_outcome_continues
#     buffers, neutral_reward_probs, and the boundary-reset label logic that fed
#     it are deleted; the encoder is a function of obs alone (as it always was).
#
# PPO + IterThink v104 (v103 + le-wm AdaLN-zero action conditioning + partial RoPE). From v103.
#
# Two changes, both tightening fidelity to le-wm's predictor:
#   (A) ACTION CONDITIONING via AdaLN-zero (le-wm's ConditionalBlock). v103 injected
#       the action TWICE and weakly: as per-dim input tokens (action_in +
#       action_dim_embed) AND as an ungated additive FiLM (cond_proj). le-wm injects
#       the action ONLY as the conditioning vector c = action_cond(a_t), which a
#       zero-init modulation MLP maps to (shift, scale, gate) for both the attention
#       and FFN sublayers of every predictor block: x = x + gate*sublayer(modulate(
#       norm(x), shift, scale)), norms affine-free. Zero init => each block starts at
#       identity and the action's influence ramps in, instead of perturbing the WM
#       from step 0. The action is no longer an input token at all (removes the
#       double-injection the audit flagged). relu^2 replaces le-wm's SiLU in the
#       modulation MLP to match this codebase's activation.
#   (B) PARTIAL RoPE 0.25 on the time axis. v103 rotated ALL head_dim channels; over a
#       5-step window the high-frequency pairs rotate several radians with no
#       positional payoff. v104 rotates only round(0.25*head_dim) channels (GPT-NeoX
#       partial RoPE), leaving the rest position-agnostic for content matching.
#
# --- inherited v103 notes (changes (1),(3),(6) below still apply; (7) full-RoPE
#     superseded by (B) above) ---
#
# Four targeted alignments to the le-wm reference, found by an adversarial
# code-vs-code audit (v102 fixed encoder SIGReg/detach symmetry; v103 continues):
#   (1) SIGReg projections 256 -> 1024 (lewm_sigreg_num_proj). The Epps-Pulley ECF
#       statistic is Monte-Carlo over random projections; le-wm uses 1024. At 256
#       the isotropy-gradient std-error is ~2x larger. Free fidelity.
#   (3) WARM-START imagination context. v102 seeded each dream from a LENGTH-1
#       context (single encoded state), the weakest regime the predictor sees. v103
#       seeds from the real preceding rollout window (up to lewm_context states +
#       their real actions), restricting seeds to boundary-clean windows (no episode
#       reset inside the window; falls back to length-1 if none). The first dreamed
#       step now sees a full real context, matching le-wm rollouts (which never query
#       with fewer than history_size frames).
#   (6) WM weight decay + larger clip. le-wm trains the WM with AdamW wd=1e-3 and
#       grad-clip 1.0; v102 shared the policy's plain Adam (no wd) and clipped the WM
#       to 0.5. v103 uses AdamW with a WM-only weight-decay param group
#       (lewm_weight_decay, others 0 so PPO is unchanged) and clips the WM to
#       lewm_grad_clip=1.0. SIGReg + weight decay jointly bound embedding norm in
#       le-wm; v102 relied on SIGReg alone.
#   (7) TEMPORAL POSITION via RoPE on the predictor time axis. v102 had NO temporal
#       position signal (causal mask only). le-wm adds a learned ABSOLUTE temporal
#       pos embedding; we instead use ROTARY (relative) on the time-axis attention
#       because imagination is an autoregressive SLIDING-WINDOW process -- a learned
#       absolute-within-window code would relabel the same physical state as the
#       window advances, whereas RoPE depends only on relative offset (i-j) and is
#       slide-invariant (and parameter-free). Space-axis/encoder attention unchanged;
#       obs tokens already carry per-feature identity so the space axis needs none.
#
# --- inherited v102 notes ---
#
# PPO + IterThink v102 (v100 + LeWM encoder-symmetry fixes). From v100.
#
# MOTIVATION. The LeWM design (see paper diagram) is encoder-SYMMETRIC: a single
# shared encoder produces z_t (predictor input/context) and z_{t+1} (target), and
# BOTH receive (a) prediction-loss gradient — le-wm computes pred_loss on
# emb[:ctx] and emb[n_preds:] with NEITHER detached — and (b) SIGReg, applied to
# the FULL embedding sequence (sigreg(emb) over all timesteps). v100 was
# target-side-only on both counts: the predictor context was detached (so the
# encoder got no "good input rep" gradient) and SIGReg covered only the z_1..z_H
# targets, leaving the z_0 rollout anchor — the state imagination actually rolls
# FROM — unconstrained. v102 restores both symmetries:
#   (1) Context un-detached: the encoder is shaped as both a good input and a good
#       target representation (faithful to le-wm). NOTE: this reverses v85, which
#       detached the context specifically so the predicted-outcome (reward/cont)
#       head losses would not push gradient through historical encoder summaries.
#       le-wm has no outcome heads, so this confound is ours alone — watch for the
#       outcome heads destabilizing the encoder via the now-live context path.
#   (2) SIGReg on BOTH z_0 and z_1..z_H (anchor + targets), matching sigreg(emb).
# Everything else is identical to v100.
#
# --- inherited v100 notes ---
#
# PPO + IterThink v100 (v24 Beta + LeWM dreamer4-style imagination training).
# From v99.
#
# KEY IDEA. v99's LeWM is a detached world model that is *only* trained as a
# representation/dynamics learner; its imagine_step API was used purely for
# diagnostics. v100 promotes WM imagination to a FULL-STRENGTH agent training path
# — the same actor/critic that PPO trains on real rollouts is also trained, at
# full coefficient (not a 0.05 auxiliary), on on-policy rollouts dreamed inside the
# frozen WM. This is the dreamer4 recipe: a learned simulator cheaply generates a
# large amount of fresh on-policy training data per iteration.
#
# PER-ITERATION PIPELINE (extends v99):
#   1. Collect a real rollout.
#   2. Train the WM on LE-JEPA dynamics/outcomes (unchanged from v99).
#   3. Train the actor/critic with real-data PPO (unchanged from v99).
#   4. IMAGINATION: for c = `imagine_batches` generate->train cycles, each:
#        a. sample `imagine_batch_size` real rollout states, encode WM summaries;
#        b. roll the policy forward `imagine_horizon` steps THROUGH THE FROZEN WM:
#           the actor samples an action from the (detached) summary belief, the WM
#           predicts next summary / reward / continuation (all detached simulator
#           outputs), repeat;
#        c. compute SCALAR HL-Gauss lambda-returns over the imagined horizon using
#           the WM's soft continuation as the discount mask and the existing critic
#           to bootstrap the final imagined state (critic target, see CRITIC below);
#        d. take ONE 1-epoch gradient step: train the existing critic on the
#           imagined return target (CE) and the existing actor on the imagined
#           advantage (PPO/clip), at full strength.
#      Each cycle regenerates fresh data from the just-updated policy, so the
#      imagined data is on-policy and the PPO ratio starts at 1 (clean PG).
#
# CRITIC (ported from iterthink_v24_beta_d4hlgauss_nocriticbias_mtp_v1):
#   The distributional categorical C51-projected lambda-return critic was replaced
#   by the Dreamer4-style "d4hlgauss nocriticbias mtp" critic:
#     - Target: SCALAR GAE lambda-return projected onto a fixed support with HL-Gauss
#       (Gaussian-smoothed two-hot, support_is_edges, sigma=value_sigma_to_bin_ratio
#       bins), NOT a categorical Bellman backup. Applied in BOTH the real and the
#       imagined critic-training paths.
#     - MTP: the critic head emits critic_mtp_horizon return rows; horizon 0 is
#       V(s_t), horizons 1..H-1 predict returns[t+h] from the same features. Loss is
#       per-horizon CE summed over a validity mask (episode-boundary + tail masked in
#       the real path; tail-only in the dream, where continuation is folded into GAE).
#     - nocriticbias: the peaked zero-return logit-bias prior is removed; the head
#       keeps layer_init's plain zero bias so the scalar HL-Gauss target sets the scale.
#
# DESIGN CHOICES / FAITHFULNESS:
#   - WM stays detached from agent gradients (detach_world_model_from_agent). The
#     imagined latents/rewards/continuations are .detach()ed; the actor is trained
#     by policy gradient (advantage-weighted), NOT by analytic dynamics backprop.
#   - Imagination is RECURRENT: the dream carries a growing (latent, action) history
#     and conditions the predictor on it via `imagine_step_from_history`, exactly like
#     the teacher-forced training path (context grows 1 -> lewm_context across the
#     horizon). This keeps the WM in-distribution in the dream — querying it with a
#     length-1 context at every step (as the original cut did) is a regime it only saw
#     at position 0 of training and drives imagined latents off-manifold.
#   - The critic is shared with real PPO and remains anchored by the ~320 real
#     gradient steps/iteration; the 8 imagined steps add model-based credit without
#     a separate value head.
#
# --- inherited v99 notes ---
#
# PPO + IterThink v99 (v24 Beta + direct latent semantic MTP). From v98.
#
# This keeps v24's PPO/Beta/distributional-critic machinery and keeps the
# MLP/MoE ThinkTrunk as the actor/critic trunk. A separate le-wm style auxiliary
# module learns a detached world-model representation:
#   per-feature embeddings -> transformer encoder/predictor -> MTP future tokens.
# The world model is trained on rollout transitions with multi-token prediction:
# teacher-forced latent/action history predicts future obs+outcome target
# embeddings. Summaries contain obs tokens only; transition outcomes are predicted
# by heads over action-conditioned predicted obs latents. The v24 agent trunk
# reads detached predictor-trunk belief latents, matching the latest le-wm control
# path while preserving the v24 PPO/ThinkTrunk backend.
# v76 removes WM-only midpoint warmup for the PPO-backed agent. The actor collects
# valid behavior-policy actions from step one while the detached WM trains online;
# this avoids training the WM only on midpoint-action dynamics and then dropping an
# untrained random actor into an out-of-distribution latent/action regime.
# v77 keeps low-memory WM microbatches, but accumulates them into PPO-sized
# effective minibatches before stepping Adam. v76 accidentally turned 32 WM Adam
# steps per rollout into 512, making the detached latent interface move too fast.
# v78 stops replacing the v24 observation interface with a cold latent-only
# interface. The v24 ThinkTrunk still receives an obs_dim vector; a zero-initialized
# detached-WM adapter adds a learnable residual to raw observations, so behavior
# starts as v24 and PPO can opt into WM features when useful.
# v79 initializes the v24 trunk/heads before the WM so the zero-adapter control
# path also has v24 RNG parity.
# v80 isolates WM initialization/update randomness from PPO/action randomness and
# uses transition outcome labels for final-observation bootstrap beliefs before
# neutralizing labels for reset observations.
# v81 masks recurrent WM history per env across episode boundaries so reset
# observations do not receive residuals conditioned on a previous episode.
# v82 builds the transition-next bootstrap belief before invalidating history for
# the next reset observation, preserving valid same-episode context at truncations.
# v83 scales the zero-initialized WM observation residual so PPO remains close to
# the v24 control path while the auxiliary latent becomes useful.
# It also bootstraps PPO from transition-next beliefs, so time-limit truncations
# use final_observation value targets while lambda carryover still stops at reset
# boundaries.
# v84 fixes the predicted outcome objective so reward/continuation losses train
# predicted MTP tokens instead of only detached decoder heads, and logs scalar
# reward prediction probes by MTP offset.
# v85 detaches the teacher-forced predictor context so the predicted outcome
# losses train the predictor/output path instead of pushing through historical
# encoder summaries, and aggregates reward probes on GPU to avoid per-minibatch
# synchronization overhead.
# v86 makes the class-local neutral reward summary use the same HL-Gauss projected
# zero distribution as rollout/reset labels, instead of a hard center one-hot.
# v87 adds a one-step recurrent imagination API:
#   summary_t, action_t -> summary_{t+1}, reward_logits_t, continuation_logits_t.
# The training loop logs direct imagine_step one-step diagnostics against rollout
# targets without yet adding imagined actor/value losses.
# v89 moves value into the WM summary as a dedicated outcome token. Online/current
# summaries use a neutral zero-value distribution to avoid future leakage; target
# summaries use the same projected lambda-return target as v24's critic. Predicted
# value tokens are trained by summary MSE and decoded by a value probe CE.
# v90 makes reward/continuation/value decoders detached probes instead of
# non-detached semantic gradients on the latent tokens, and applies SIGReg to
# outcome tokens with the same coefficient as obs tokens. Closed-loop roll-forward
# still predicts full embeddings; probes only decode/log their semantics.
# v91 removes outcome-token MSE and outcome SIGReg. Obs tokens remain
# embedding-predicted with MSE+SIGReg; reward/continuation/value tokens are
# trained exclusively by non-detached CE/BCE outcome heads, including target-side
# autoencoding so teacher-forced label tokens and predicted tokens share decoder
# semantics.
# v92 removes outcome tokens from the recurrent/autoregressive state entirely.
# The WM summary/history/belief contains obs tokens only; reward, continuation,
# and value are still predicted by non-detached heads over action-conditioned
# predicted obs latents.
# v93 sums MTP offset losses instead of averaging them. Each predicted offset is
# an additional supervised objective; averaging made longer horizons dilute the
# one-step dynamics/outcome signal rather than add training signal.
# v94 replaces mean-pooled outcome readouts with learned recurrent semantic CLS
# slots. Reward, continuation, and value tokens are initialized as learned empty
# latents, carried through the encoder/predictor, exposed to the PPO adapter, and
# decoded by separate heads. They are not initialized from labels. Target-side
# semantic head losses train the encoder CLS slots, while obs MSE/SIGReg remain
# restricted to obs tokens.
# v95 removes multi-offset obs latent MSE. Obs dynamics are trained one-step
# only, while reward/continuation/value semantic heads keep MTP supervision across
# offsets. This avoids forcing obs tokens to directly match underconditioned
# far-future states while preserving multi-horizon outcome gradients.
# v96 removes the final encoder, predictor, outcome, and flattened adapter
# RMSNorms so the agent can consume latent magnitude information directly. The
# internal transformer block pre-norms remain intact.
# v97 restores tokenwise WM final norms and gives the PPO adapter the same
# tokenwise treatment: latent tokens are RMS-normalized per token before the
# adapter projection, instead of globally normalizing the flattened latent vector.
# v98 swaps those final tokenwise RMSNorms for LayerNorms. The global flattened
# adapter RMSNorm remains removed and internal transformer block pre-norms remain
# unchanged.
# v99 removes the obs-space residual adapter bottleneck. The v24 PPO trunk reads
# the full detached flattened LeWM belief latent directly, with no raw observation
# addition, no 1280->obs_dim compression, and no residual scale. Final tokenwise
# readout norms are RMSNorms again. WM value prediction is removed; the PPO critic
# remains the only value learner.
#
# --- inherited v24 notes ---
#
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The distributional critic target is deliberately entropy-free so the fixed support
# remains calibrated. Off (default) => byte-identical to the v24 base.
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

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def log_labeled_rollout_diagnostic(writer, prefix, global_step, latent_mse_sum,
                                   reward_abs_sum, reward_err_sum,
                                   cont_bce_sum, weight):
    # Labeled replay-action rollout diagnostic. These are not consumed-dream
    # metrics: they compare a replay-action latent rollout against replay labels.
    # Use `imagine/*` for actual generated dream statistics.
    eps = 1e-8
    total_w = weight.sum().clamp_min(eps)
    writer.add_scalar(f"{prefix}/latent_mse", (latent_mse_sum.sum() / total_w).item(), global_step)
    writer.add_scalar(f"{prefix}/reward_mae", (reward_abs_sum.sum() / total_w).item(), global_step)
    writer.add_scalar(f"{prefix}/reward_bias", (reward_err_sum.sum() / total_w).item(), global_step)
    writer.add_scalar(f"{prefix}/continuation_bce", (cont_bce_sum.sum() / total_w).item(), global_step)
    for h in range(weight.shape[0]):
        w_h = weight[h]
        if w_h.item() <= 0:
            continue
        writer.add_scalar(f"{prefix}/latent_mse_h{h + 1}", (latent_mse_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/reward_mae_h{h + 1}", (reward_abs_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/reward_bias_h{h + 1}", (reward_err_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/continuation_bce_h{h + 1}", (cont_bce_sum[h] / w_h).item(), global_step)


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
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
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
    # the soft advantage supplies FUTURE-entropy credit. With plain GAE its magnitude
    # remains meaningful before the standard minibatch normalization.
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # exact conditional Beta KL budget per real PPO rollout

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = False     # v198: separate query readouts/trunks prevent critic-driven actor drift
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    # Raw-reward return support in symlog coordinates:
    # symlog(-2000)=-7.6014023346, symlog(20000)=9.9035375513.
    # symlog(-20000)=-9.9035375513, symlog(20000)=9.9035375513.
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    critic_prior_floor: float = 1e-20
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 2.0  # HL-Gauss projection sigma (Dreamer4 / hl_gauss default)
    critic_mtp_horizon: int = 6            # critic MTP: predict H future return rows per state

    # v196 uses the actual GAE magnitude; the optimizer performs only canonical
    # per-minibatch mean/std normalization below.
    adv_transform: str = "v10"

    # v10 is the identity; only canonical minibatch normalization is active.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]
    explore_code_dim: int = 4        # persistent actor-only exploration coordinates
    explore_resample_steps: int = 64 # hold a direction long enough to produce coherent motion
    explore_logit_scale: float = 0.5 # antisymmetric Beta alpha/beta logit shift

    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    lewm_dim: int = 64
    lewm_encoder_layers: int = 2
    lewm_predictor_layers: int = 2   # v198: two action-free causal temporal blocks; no predictor spatial blocks
    lewm_heads: int = 4
    lewm_kv_heads: int = 2
    lewm_ffn_mult: int = 2
    lewm_context: int = 5
    lewm_dyn_horizon: int = 5
    lewm_mtp_len: int = 1            # v198: only offset 0 has the required outbound-action conditioning
    policy_forecast_loss_coef: float = 0.5
    policy_forecast_logstd_min: float = -8.0
    policy_forecast_logstd_max: float = 0.0
    policy_forecast_adapter_warmup_steps: int = 100000
    lewm_update_epochs: int = 2       # 1->2: WM was getting 10x fewer grad steps than the agent
    # Transition cross-attention raises activation memory; accumulate two 512-row
    # microbatches to preserve the proven 1024-row effective WM update exactly.
    lewm_minibatch_size: int = 512
    lewm_effective_minibatch_size: int = 1024
    wm_warmup_steps: int = 0
    lewm_loss_coef: float = 1.0
    lewm_reward_loss_coef: float = 0.5
    lewm_termination_loss_coef: float = 0.5
    lewm_reward_num_bins: int = 101
    lewm_reward_raw_v_min: float = -50.0
    lewm_reward_raw_v_max: float = 50.0
    # symlog(-50)=-3.9318256327, symlog(50)=3.9318256327.
    lewm_reward_v_min: float = -3.9318256327243257
    lewm_reward_v_max: float = 3.9318256327243257
    lewm_reward_prior_floor: float = 1e-20
    lewm_sigreg_coef: float = 0.09
    lewm_sigreg_num_proj: int = 1024  # v103: 256->1024, matches le-wm (halves ECF estimator std-error)
    lewm_sigreg_knots: int = 17
    lewm_sigreg_min_valid: int = 32
    lewm_weight_decay: float = 1e-3   # v103: AdamW weight decay on WM params only (le-wm parity); 0 elsewhere
    lewm_grad_clip: float = 1.0       # v103: WM grad-norm clip (le-wm uses 1.0); PPO keeps max_grad_norm=0.5
    lewm_rope_fraction: float = 0.25  # v104: partial RoPE on time axis (rotate 25% of head_dim, rest position-agnostic)
    detach_world_model_from_agent: bool = True
    diagnostics_length1_ar: bool = False
    diagnostics_warm_ar: bool = True

    # v100 dreamer4-style imagination training. The frozen (detached) WM is used as
    # a simulator to generate fresh on-policy rollouts that train the SAME
    # actor/critic at full strength.
    imagine_enable: bool = False       # v216 replaces the old separate H8 update with joint H5 PPO
    imagine_batches: int = 3          # c: generate->train cycles per iteration (each a 1-epoch step).
                                      # Scaled 8->3: at 8 the imagined updates swamped real-PPO and
                                      # drove the entropy collapse (return stuck -180 while base v99
                                      # reached +2200); fewer cycles reduce that update-volume pressure.
    imagine_batch_size: int = 4096    # starting states per imagined block (parallel envs in dream)
    imagine_horizon: int = 8          # imagined rollout length; block = horizon * batch_size transitions
    imagine_warmup_steps: int = 100000  # let real PPO + WM establish before imagining
    imagine_prompt_bank_rollouts: int = 0  # v216 has no H8 prompt bank
    joint_imagine_horizon: int = 5
    joint_imagine_batch_size: int = 4096
    private_forecast_refresh_minibatch_size: int = 1024

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
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


def relu_sq(x):
    return torch.relu(x).pow(2)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer over token latent samples."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def sample_projection(self, dim, device, dtype, generator=None):
        A = torch.randn(dim, self.num_proj, device=device, dtype=dtype, generator=generator)
        return A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8))

    def forward(self, proj, A=None, generator=None, n_scale=None):
        # proj: (tokens, valid_samples, dim). n_scale overrides the extensive
        # *N factor (default proj.size(-2)) so SIGReg strength can be decoupled
        # from the actual sample count and pinned to a fixed reference.
        if A is None:
            A = self.sample_projection(proj.size(-1), proj.device, proj.dtype, generator=generator)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * (proj.size(-2) if n_scale is None else n_scale)
        return statistic.mean()


def xavier_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class LeWMTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.ffn_scale = nn.Parameter(torch.ones(dim))
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, causal=False):
        attn_mask = None
        if causal:
            seq = x.shape[1]
            attn_mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_scale.view(1, 1, -1).to(x.dtype) * attn_out
        h = self.ffn_norm(x)
        x = x + self.ffn_scale.view(1, 1, -1).to(x.dtype) * self.w2(relu_sq(self.w1(h)))
        return x


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def modulate(x, shift, scale):
    # DiT/AdaLN modulation: scale is a zero-centred multiplicative perturbation,
    # so (1 + scale) keeps the block at identity when the modulation MLP is zero.
    return x * (1 + scale) + shift


class _SpaceAttention(nn.Module):
    # Non-causal self-attention over the SPACE (feature-token) axis. No positional
    # code: feature tokens have no spatial order. Bare attention -- norm/residual/gate
    # live in the AdaLN block so the action conditioning owns them.
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class _RoPEAttention(nn.Module):
    # Causal self-attention over the TIME axis with PARTIAL rotary position embedding
    # (GPT-NeoX style): only the first `rotary_dim = round(head_dim * rope_fraction)`
    # channels per head are rotated; the rest stay position-agnostic. Over a short
    # 5-step imagination window full RoPE wastes its high-frequency pairs (the top
    # pair rotates ~4 rad across the window -- churn with no positional payoff), so a
    # 0.25 fraction keeps relative-position structure on a few low-frequency channels
    # while leaving most of the head free for content matching. norm/residual/gate are
    # supplied by the AdaLN block.
    def __init__(self, dim, num_heads, rope_fraction, rope_theta=10000.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        rot = int(round(self.head_dim * rope_fraction))
        rot -= rot % 2  # rotary_dim must be even (cos/sin come in pairs)
        rot = max(2, min(rot, self.head_dim))
        self.rotary_dim = rot
        self.qkv = xavier_linear(nn.Linear(dim, 3 * dim, bias=False))
        self.proj = xavier_linear(nn.Linear(dim, dim, bias=False))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot, 2).float() / rot))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))  # (T, rotary_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (T, rotary_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x, causal=True):
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        rd = self.rotary_dim
        cos, sin = self._rope_cos_sin(seq_len, x.device, q.dtype)
        cos = cos.view(1, 1, seq_len, rd)
        sin = sin.view(1, 1, seq_len, rd)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
        k = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        return self.proj(attn_out)


class AdaLNAxialPredictorBlock(nn.Module):
    # le-wm's ConditionalBlock (AdaLN-zero), generalised to the axial predictor.
    # The action enters ONLY here, as the conditioning vector c = action_features
    # (one vector per timestep) -- NOT as input tokens. A zero-init modulation MLP
    # maps c -> (shift, scale, gate) for both the attention and FFN sublayers; the
    # zero init makes every block start at identity, so the action's influence ramps
    # in during training rather than perturbing the world model from step 0. The
    # per-timestep modulation broadcasts over the space tokens (the action conditions
    # the whole frame uniformly). relu^2 replaces le-wm's SiLU in the modulation MLP
    # to match this codebase's activation. Norms are affine-free LayerNorm (the affine
    # role is played by AdaLN's shift/scale), per DiT/le-wm.
    def __init__(self, dim, num_heads, ffn_mult, axis, rope_fraction):
        super().__init__()
        if axis not in {"space", "time"}:
            raise ValueError(f"unknown predictor axis {axis}")
        self.axis = axis
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if axis == "time":
            self.attn = _RoPEAttention(dim, num_heads, rope_fraction)
        else:
            self.attn = _SpaceAttention(dim, num_heads)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        # AdaLN-zero: relu^2 nonlinearity then a zero-init linear to 6*dim
        # (shift/scale/gate for attn and FFN). Zero init => identity at start.
        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, action_features):
        # x: (B, T, S, D), action_features (conditioning c): (B, T, D)
        batch, time_len, space_len, width = x.shape
        mod = self.adaLN(relu_sq(action_features))                  # (B, T, 6D)
        sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, dim=-1)           # each (B, T, D)
        if self.axis == "space":
            y = x.reshape(batch * time_len, space_len, width)

            def bc(p):  # (B, T, D) -> (B*T, 1, D): broadcast over space tokens
                return p.reshape(batch * time_len, 1, width)

            h = modulate(self.norm1(y), bc(sh1), bc(sc1))
            y = y + bc(g1) * self.attn(h)
            h = modulate(self.norm2(y), bc(sh2), bc(sc2))
            y = y + bc(g2) * self.w2(relu_sq(self.w1(h)))
            return y.reshape(batch, time_len, space_len, width)
        y = x.permute(0, 2, 1, 3).contiguous().reshape(batch * space_len, time_len, width)

        def bc(p):  # (B, T, D) -> (B*S, T, D): each space stream shares per-step c
            return p.unsqueeze(1).expand(batch, space_len, time_len, width).reshape(batch * space_len, time_len, width)

        h = modulate(self.norm1(y), bc(sh1), bc(sc1))
        y = y + bc(g1) * self.attn(h, causal=True)
        h = modulate(self.norm2(y), bc(sh2), bc(sc2))
        y = y + bc(g2) * self.w2(relu_sq(self.w1(h)))
        return y.reshape(batch, space_len, time_len, width).permute(0, 2, 1, 3).contiguous()


class TemporalStateBlock(nn.Module):
    """Action-free causal history mixing along each observation-token stream."""

    def __init__(self, dim, num_heads, ffn_mult, rope_fraction):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)
        self.attn = _RoPEAttention(dim, num_heads, rope_fraction)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        # Start close to the already-useful frame encoder while leaving a direct
        # gradient path into both temporal sublayers from the first WM update.
        self.attn_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x):
        # x: (B, T, S, D). Attention is causal over T independently per slot.
        batch, time_len, space_len, width = x.shape
        y = x.permute(0, 2, 1, 3).contiguous().reshape(batch * space_len, time_len, width)
        y = y + self.attn_scale.view(1, 1, -1).to(y.dtype) * self.attn(
            self.attn_norm(y), causal=True
        )
        y = y + self.ffn_scale.view(1, 1, -1).to(y.dtype) * self.w2(
            relu_sq(self.w1(self.ffn_norm(y)))
        )
        return y.reshape(batch, space_len, time_len, width).permute(0, 2, 1, 3).contiguous()


class TransitionQueryReadout(nn.Module):
    """WM-owned action-conditioned queries for the outbound transition."""

    def __init__(self, num_obs_tokens, num_semantic_tokens, dim, num_heads, ffn_mult):
        super().__init__()
        self.num_obs_tokens = num_obs_tokens
        self.num_semantic_tokens = num_semantic_tokens
        self.semantic_query = nn.Parameter(torch.empty(num_semantic_tokens, dim))
        self.obs_query = nn.Parameter(torch.empty(num_obs_tokens, dim))
        nn.init.xavier_uniform_(self.semantic_query)
        nn.init.xavier_uniform_(self.obs_query)
        self.query_norm = nn.RMSNorm(dim)
        self.memory_norm = nn.RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        for name, param in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
        self.ffn_norm = nn.RMSNorm(dim)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        self.action_shift = xavier_linear(nn.Linear(dim, dim, bias=False))
        self.action_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.action_gate.weight)
        nn.init.zeros_(self.action_gate.bias)

    def forward(self, state_memory, action_features):
        # state_memory: (B,T,S,D), action_features: (B,T,D). Observation queries
        # retain their aligned state token as a residual; all queries then read all
        # current-frame memory slots. No query writes back into the state stream.
        batch, time_len, space_len, width = state_memory.shape
        if space_len != self.num_obs_tokens:
            raise ValueError(f"expected {self.num_obs_tokens} state tokens, got {space_len}")
        memory = state_memory.reshape(batch * time_len, space_len, width)
        semantic = self.semantic_query.to(memory.dtype).view(1, self.num_semantic_tokens, width)
        semantic = semantic.expand(batch * time_len, -1, -1)
        obs_query = self.obs_query.to(memory.dtype).view(1, self.num_obs_tokens, width)
        obs_query = obs_query.expand(batch * time_len, -1, -1) + memory
        queries = torch.cat([semantic, obs_query], dim=1)
        action = action_features.reshape(batch * time_len, 1, width)
        queries = queries + self.action_shift(action)
        attn_out, _ = self.cross_attn(
            self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )
        queries = queries + attn_out
        gate = torch.tanh(self.action_gate(relu_sq(action)))
        queries = queries + (1.0 + gate) * self.w2(relu_sq(self.w1(self.ffn_norm(queries))))
        return queries.reshape(batch, time_len, self.num_semantic_tokens + self.num_obs_tokens, width)


class AgentQueryReadout(nn.Module):
    """Agent-owned learned query over detached current-frame WM memory."""

    def __init__(self, num_tokens, dim, num_heads, ffn_mult):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.query)
        self.query_norm = nn.RMSNorm(dim)
        self.memory_norm = nn.RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        for name, param in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
        self.ffn_norm = nn.RMSNorm(dim)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        self.out_norm = nn.RMSNorm(dim)

    def forward(self, flat_memory):
        memory = flat_memory.reshape(flat_memory.shape[0], self.num_tokens, self.dim)
        query = self.query.to(memory.dtype).expand(memory.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(
            self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )
        query = query + attn_out
        query = query + self.w2(relu_sq(self.w1(self.ffn_norm(query))))
        return self.out_norm(query[:, 0])


class PolicyMomentOutcomeQuery(nn.Module):
    """WM-owned read-only query forecasting the successor pure-base Beta policy."""

    def __init__(self, dim, act_dim, num_heads, ffn_mult, logstd_min, logstd_max):
        super().__init__()
        self.dim = dim
        self.act_dim = act_dim
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.queries = nn.Parameter(torch.empty(2, dim))
        nn.init.xavier_uniform_(self.queries)
        self.action_embed = xavier_linear(nn.Linear(act_dim, dim, bias=False))
        self.query_norm = nn.RMSNorm(dim)
        self.memory_norm = nn.RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        for name, param in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
        self.ffn_norm = nn.RMSNorm(dim)
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult, bias=False))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim, bias=False))
        self.alpha_head = xavier_linear(nn.Linear(dim, act_dim))
        self.beta_head = xavier_linear(nn.Linear(dim, act_dim))

    @staticmethod
    def beta_native_moments(alpha, beta):
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total.square() * (total + 1.0))
        return mean, variance.clamp_min(1e-12).sqrt()

    def forward(self, state_memory, outbound_action):
        if state_memory.ndim != 4:
            raise ValueError("policy forecast memory must be (B,T,S,D)")
        batch, time_len, space_len, width = state_memory.shape
        if outbound_action.shape[:2] != (batch, time_len):
            raise ValueError("policy forecast action must align with state memory")
        memory = state_memory.reshape(batch * time_len, space_len, width)
        action = self.action_embed(outbound_action).reshape(batch * time_len, 1, width)
        queries = self.queries.to(memory.dtype).unsqueeze(0).expand(
            batch * time_len, -1, -1
        ) + action
        memory_n = self.memory_norm(memory)
        attn_out, _ = self.cross_attn(
            self.query_norm(queries), memory_n, memory_n, need_weights=False
        )
        tokens = queries + attn_out
        tokens = tokens + self.w2(relu_sq(self.w1(self.ffn_norm(tokens))))
        alpha = 1.0 + F.softplus(self.alpha_head(tokens[:, 0]))
        beta = 1.0 + F.softplus(self.beta_head(tokens[:, 1]))
        mean, std = self.beta_native_moments(alpha, beta)
        logstd = std.log().clamp(self.logstd_min, self.logstd_max)
        feature = torch.cat([mean, logstd], dim=-1)
        return (
            tokens.reshape(batch, time_len, 2, width),
            alpha.reshape(batch, time_len, self.act_dim),
            beta.reshape(batch, time_len, self.act_dim),
            feature.reshape(batch, time_len, 2 * self.act_dim),
        )


class LeWMBackbone(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dim = args.lewm_dim
        self.num_obs_tokens = obs_dim
        self.reward_num_bins = args.lewm_reward_num_bins
        self.num_semantic_tokens = 2
        self.reward_token_idx = 0
        self.continuation_token_idx = 1
        self.obs_token_start = self.num_semantic_tokens
        self.num_outcome_tokens = 0
        self.num_latent_tokens = self.num_semantic_tokens + self.num_obs_tokens
        self.context = args.lewm_context
        self.mtp_len = args.lewm_mtp_len

        # Latest le-wm tokenizer stance: one learned affine token per observation scalar.
        self.semantic_tokens = nn.Parameter(torch.empty(self.num_semantic_tokens, self.dim))
        nn.init.xavier_uniform_(self.semantic_tokens)
        self.obs_feature_weight = nn.Parameter(torch.empty(obs_dim, self.dim))
        self.obs_feature_bias = nn.Parameter(torch.empty(obs_dim, self.dim))
        nn.init.xavier_uniform_(self.obs_feature_weight)
        nn.init.zeros_(self.obs_feature_bias)
        self.embed_norm = nn.RMSNorm(self.dim)
        self.encoder_layers = nn.ModuleList(
            [LeWMTransformerBlock(self.dim, args.lewm_heads, args.lewm_ffn_mult) for _ in range(args.lewm_encoder_layers)]
        )
        self.encoder_norm = nn.RMSNorm(self.dim)
        self.semantic_target_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        self.obs_target_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        # v120: dedicated outcome readout tokens. The predictor's semantic slots
        # are role-specific query/readout tokens: token 0 -> reward CE, token 1 ->
        # continuation BCE. Obs tokens still carry dynamics state and are the only
        # recurrent stream.
        self.reward_token_norm = nn.RMSNorm(self.dim)
        self.continuation_token_norm = nn.RMSNorm(self.dim)
        self.reward_action_proj = xavier_linear(nn.Linear(self.act_dim, self.dim, bias=False))
        self.reward_token_unproj = xavier_linear(nn.Linear(self.dim, self.reward_num_bins))
        self.continuation_token_unproj = xavier_linear(nn.Linear(self.dim, 1))

        # v198 boundary: the shared state memory is action-free. Outbound a_t is
        # embedded only by the separate transition queries after the policy acts.
        self.action_cond = xavier_linear(nn.Linear(act_dim, self.dim))
        self.predictor_layers = nn.ModuleList(
            [
                TemporalStateBlock(
                    self.dim, args.lewm_heads, args.lewm_ffn_mult, args.lewm_rope_fraction
                )
                for _ in range(args.lewm_predictor_layers)
            ]
        )
        self.transition_readout = TransitionQueryReadout(
            self.num_obs_tokens,
            self.num_semantic_tokens,
            self.dim,
            args.lewm_heads,
            args.lewm_ffn_mult,
        )
        self.predictor_norm = nn.RMSNorm(self.dim)
        self.pred_next_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        self.pred_mtp_projs = nn.ModuleList(
            [xavier_linear(nn.Linear(self.dim, self.dim)) for _ in range(max(0, self.mtp_len - 1))]
        )
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 73000)
            self.policy_moment_outcome = PolicyMomentOutcomeQuery(
                self.dim,
                act_dim,
                args.lewm_heads,
                args.lewm_ffn_mult,
                args.policy_forecast_logstd_min,
                args.policy_forecast_logstd_max,
            )

    def encode_summary(self, obs):
        # v105: the encoder is a function of obs alone. The former
        # encode_summary_with_outcomes never read its reward/continuation args
        # (outcomes have been readout-only since v92), so the label plumbing was
        # vestigial and is removed.
        batch = obs.shape[0]
        obs_flat = obs.reshape(batch, -1)
        obs_tokens = obs_flat.unsqueeze(-1) * self.obs_feature_weight + self.obs_feature_bias
        semantic_tokens = self.semantic_tokens.to(obs_tokens.dtype).unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([semantic_tokens, obs_tokens], dim=1)
        tokens = self.embed_norm(tokens)
        for layer in self.encoder_layers:
            tokens = layer(tokens, causal=False)
        tokens = self.encoder_norm(tokens)
        semantic_tokens = self.semantic_target_proj(tokens[:, : self.num_semantic_tokens])
        obs_tokens = self.obs_target_proj(tokens[:, self.obs_token_start :])
        return torch.cat([semantic_tokens, obs_tokens], dim=1)

    def decode_belief_outcomes(self, belief, action=None, detach_belief=True):
        if detach_belief:
            belief = belief.detach()
        reward_feat = self.reward_token_norm(belief[..., self.reward_token_idx, :])
        continuation_feat = self.continuation_token_norm(
            belief[..., self.continuation_token_idx, :]
        )
        if action is not None:
            reward_feat = reward_feat + self.reward_action_proj(action)
        reward_logits = self.reward_token_unproj(reward_feat)
        continuation_logits = self.continuation_token_unproj(continuation_feat).squeeze(-1)
        return reward_logits, continuation_logits

    def _temporal_state_trunk(self, latent_history):
        # Observation-only recurrent stream. Its last-frame output is the exact
        # detached memory buffered for PPO; no synthetic/current action is needed.
        batch, context_len, num_obs, width = latent_history.shape
        if context_len > self.context:
            latent_history = latent_history[:, -self.context :]
        tokens = latent_history
        for layer in self.predictor_layers:
            tokens = layer(tokens)
        return self.predictor_norm(tokens)

    def belief_from_history(self, latent_history):
        return self._temporal_state_trunk(latent_history)[:, -1]

    def policy_forecast_from_state_memory(self, state_memory, outbound_actions):
        return self.policy_moment_outcome(state_memory, outbound_actions)

    def policy_forecast_from_history(self, latent_history, outbound_action):
        if latent_history.shape[1] > self.context:
            latent_history = latent_history[:, -self.context :]
        state_memory = self._temporal_state_trunk(latent_history)
        return self.policy_moment_outcome(
            state_memory[:, -1:], outbound_action[:, None]
        )

    def predict_mtp_from_history(self, latent_history, action_history, return_training_features=False):
        if latent_history.shape[1] > self.context:
            latent_history = latent_history[:, -self.context :]
            action_history = action_history[:, -self.context :]
        state_memory = self._temporal_state_trunk(latent_history)
        features = self.transition_readout(state_memory, self.action_cond(action_history))
        preds = [self.pred_next_proj(features)]
        preds.extend(proj(features) for proj in self.pred_mtp_projs)
        preds = torch.stack(preds, dim=2)
        if return_training_features:
            return preds, state_memory, features
        return preds

    def imagine_step_from_history(self, latent_history, action_history):
        # Same path as teacher forcing: action-free causal memory first, then the
        # actual outbound action conditions only the final transition queries.
        if latent_history.shape[1] > self.context:
            latent_history = latent_history[:, -self.context :]
        state_memory = self._temporal_state_trunk(latent_history)
        last_action = action_history[:, -1:]
        features = self.transition_readout(
            state_memory[:, -1:], self.action_cond(last_action)
        )[:, -1]
        next_summary = self.pred_next_proj(features)
        reward_logits, continuation_logits = self.decode_belief_outcomes(
            next_summary, last_action[:, -1], detach_belief=False
        )
        return next_summary, reward_logits, continuation_logits


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.obs_dim = obs_dim
        self.detach_world_model_from_agent = args.detach_world_model_from_agent
        if not self.detach_world_model_from_agent:
            raise ValueError("v198 requires a strictly detached world-model/agent interface")
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            raise ValueError("v198 requires separate actor/critic query readouts and trunks")
        self.explore_code_dim = args.explore_code_dim
        self.explore_logit_scale = args.explore_logit_scale
        self.forecast_dim = 2 * act_dim
        self.forecast_adapters_enabled = False
        # PPO buffers the detached action-free temporal memory, not a post-query
        # vector, so the agent-owned queries are recomputed and trained on updates.
        self.agent_latent_dim = obs_dim * args.lewm_dim
        self.agent_input_dim = self.agent_latent_dim + self.forecast_dim
        self.actor_readout = AgentQueryReadout(
            obs_dim, args.lewm_dim, args.lewm_heads, args.lewm_ffn_mult
        )
        self.critic_readout = AgentQueryReadout(
            obs_dim, args.lewm_dim, args.lewm_heads, args.lewm_ffn_mult
        )
        self.critic_trunk = ThinkTrunk(args.lewm_dim, H, args.k_blocks, args.n_experts)
        self.actor_trunk = ThinkTrunk(args.lewm_dim, H, args.k_blocks, args.n_experts)
        # Categorical critic, HL-Gauss MTP head:
        # outputs critic_mtp_horizon * num_bins logits; horizon 0 is V(s_t), later
        # horizons are critic-only MTP predictions of returns[t+h]. v117 restores
        # a finite zero-return log-prior after support construction; otherwise the
        # asymmetric symlog return support makes near-uniform logits decode to a
        # large positive scalar value before learning starts.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins), std=0.1
        )
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
            # A persistent code selects a state-dependent direction in action-logit
            # space. The antisymmetric shift primarily moves the Beta mean instead
            # of indiscriminately changing its concentration/entropy.
            with torch.random.fork_rng():
                torch.manual_seed(args.seed + 30000)
                self.actor_explore_basis = layer_init(
                    nn.Linear(H, act_dim * self.explore_code_dim, bias=False), std=0.1
                )
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        main_rng_state = torch.random.get_rng_state()
        self.world_model = LeWMBackbone(obs_dim, act_dim, args)
        torch.random.set_rng_state(main_rng_state)
        assert self.agent_latent_dim == self.world_model.num_obs_tokens * self.world_model.dim
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 74000)
            self.actor_forecast_residual = layer_init(
                nn.Linear(self.forecast_dim, args.lewm_dim, bias=False), std=0.0
            )
            self.critic_forecast_residual = layer_init(
                nn.Linear(self.forecast_dim, args.lewm_dim, bias=False), std=0.0
            )

    def _actor_dist(self, actor_feat, explore_code=None):
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
        raw_alpha = self.actor_alpha_head(actor_feat)
        raw_beta = self.actor_beta_head(actor_feat)
        if explore_code is not None:
            if explore_code.shape[-1] != self.explore_code_dim:
                raise ValueError("exploration code has the wrong width")
            basis = self.actor_explore_basis(actor_feat).reshape(
                actor_feat.shape[0], raw_alpha.shape[-1], self.explore_code_dim
            )
            raw_delta = torch.einsum("bar,br->ba", basis, explore_code) / (
                self.explore_code_dim ** 0.5
            )
            shift = self.explore_logit_scale * torch.tanh(raw_delta)
            raw_alpha = raw_alpha + shift
            raw_beta = raw_beta - shift
        alpha = 1.0 + F.softplus(raw_alpha)
        beta = 1.0 + F.softplus(raw_beta)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def pack_agent_input(self, latent, incoming_forecast):
        if self.detach_world_model_from_agent:
            latent = latent.detach()
            incoming_forecast = incoming_forecast.detach()
        return torch.cat(
            [latent.reshape(latent.shape[0], -1), incoming_forecast], dim=-1
        )

    def split_agent_input(self, agent_input):
        if agent_input.shape[-1] != self.agent_input_dim:
            raise ValueError("agent input has the wrong width")
        return (
            agent_input[..., : self.agent_latent_dim],
            agent_input[..., self.agent_latent_dim :],
        )

    def agent_input_from_latent(self, latent, incoming_forecast=None):
        if incoming_forecast is None:
            incoming_forecast = latent.new_zeros(latent.shape[0], self.forecast_dim)
        return self.pack_agent_input(latent, incoming_forecast)

    def agent_input_from_history(self, latent_history, incoming_forecast=None):
        latent = self.world_model.belief_from_history(latent_history)
        return self.agent_input_from_latent(latent, incoming_forecast)

    def agent_input_from_obs(self, x):
        latent = self.world_model.encode_summary(x)[:, self.world_model.obs_token_start :]
        return self.agent_input_from_history(latent.unsqueeze(1))

    def _trunks_from_agent_input(self, obs, agent_input):
        if self.detach_world_model_from_agent:
            agent_input = agent_input.detach()
        return self._actor_feat_from_agent_input(agent_input), self._critic_feat_from_agent_input(agent_input)

    def _base_actor_feat_from_agent_input(self, agent_input):
        if self.detach_world_model_from_agent:
            agent_input = agent_input.detach()
        memory, _ = self.split_agent_input(agent_input)
        return self.actor_trunk(self.actor_readout(memory))

    def _actor_feat_from_agent_input(self, agent_input):
        if self.detach_world_model_from_agent:
            agent_input = agent_input.detach()
        memory, forecast = self.split_agent_input(agent_input)
        trunk_input = self.actor_readout(memory)
        if self.forecast_adapters_enabled:
            trunk_input = trunk_input + self.actor_forecast_residual(forecast)
        return self.actor_trunk(trunk_input)

    def _critic_feat_from_agent_input(self, agent_input):
        if self.detach_world_model_from_agent:
            agent_input = agent_input.detach()
        memory, forecast = self.split_agent_input(agent_input)
        trunk_input = self.critic_readout(memory)
        if self.forecast_adapters_enabled:
            trunk_input = trunk_input + self.critic_forecast_residual(forecast)
        return self.critic_trunk(trunk_input)

    def _trunks(self, x):
        return self._trunks_from_agent_input(x, self.agent_input_from_obs(x))

    def get_value(self, x):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t), later
        # horizons are critic-only MTP predictions. Caller converts via support.
        critic_feat = self._critic_feat_from_agent_input(self.agent_input_from_obs(x))
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_value_from_agent_input(self, obs, agent_input):
        critic_feat = self._critic_feat_from_agent_input(agent_input)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_beta_params_from_agent_input(self, agent_input, explore_code):
        actor_feat = self._actor_feat_from_agent_input(agent_input)
        dist, _, _ = self._actor_dist(actor_feat, explore_code)
        return dist.concentration1, dist.concentration0

    def get_base_beta_params_from_agent_input(self, agent_input):
        actor_feat = self._base_actor_feat_from_agent_input(agent_input)
        dist, _, _ = self._actor_dist(actor_feat, None)
        return dist.concentration1, dist.concentration0

    def get_action_and_value(self, x, z=None, explore_code=None):
        return self.get_action_and_value_from_agent_input(
            x, self.agent_input_from_obs(x), z, explore_code
        )

    def get_action_and_value_from_agent_input(
        self, obs, agent_input, z=None, explore_code=None, return_beta_params=False
    ):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks_from_agent_input(obs, agent_input)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat, explore_code)
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
        outputs = (action, z, log_prob, entropy, value_logits)
        if return_beta_params:
            if self.actor_dist != "beta":
                raise ValueError("exact Beta trust region requires the Beta actor")
            return outputs + (dist.concentration1, dist.concentration0)
        return outputs

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = (
                list(self.actor_alpha_head.parameters())
                + list(self.actor_beta_head.parameters())
                + list(self.actor_explore_basis.parameters())
            )
        return (
            list(self.actor_readout.parameters())
            + list(self.actor_forecast_residual.parameters())
            + list(trunk.parameters())
            + heads
        )

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        return (
            list(self.critic_readout.parameters())
            + list(self.critic_forecast_residual.parameters())
            + list(self.critic_trunk.parameters())
            + list(self.critic_head.parameters())
        )

    @torch.no_grad()
    def assert_zero_forecast_residual_init(self):
        for name, module in (
            ("actor", self.actor_forecast_residual),
            ("critic", self.critic_forecast_residual),
        ):
            if torch.count_nonzero(module.weight).item() != 0:
                raise AssertionError(f"{name} forecast residual must initialize zero")


def shape_advantage(gae, sigma, u, args, device):
    del sigma, u, device
    if args.adv_transform != "v10":
        raise ValueError("v197 supports plain GAE only")
    return gae


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    assert args.lewm_dyn_horizon <= args.lewm_context, "v73 MTP expects dyn_horizon <= lewm_context"
    assert args.lewm_mtp_len <= args.lewm_dyn_horizon, "v73 MTP expects mtp_len <= dyn_horizon"
    assert args.lewm_mtp_len == 1, "v198 transition queries condition only the one-step outbound action"
    assert args.detach_world_model_from_agent, "v198 requires PPO to be detached from the WM"
    assert args.policy_forecast_adapter_warmup_steps >= 0
    assert args.lewm_dyn_horizon == args.lewm_context, (
        "v213 forecast targets require the teacher-forced final row to have the "
        "same full causal context as the buffered successor policy"
    )
    assert args.adv_transform == "v10", "v197 deliberately forbids ranked advantage transforms"
    assert args.norm_adv and args.norm_adv_scope == "minibatch", (
        "v197 uses canonical per-minibatch advantage normalization"
    )
    assert args.pos_neg_alpha == 0.5, "v197 preserves the plain GAE sign and weighting"
    assert args.actor_dist == "beta", "v197 exact trust-region accounting requires Beta"
    assert not args.share_backbone, "v198 separates actor/critic query readouts and trunks"
    assert args.separate_grad_clip, "v197 trust-region freezing requires decoupled gradients"
    assert args.target_kl is not None and args.target_kl > 0
    assert args.explore_code_dim > 0
    assert args.explore_resample_steps > 0
    assert not args.imagine_enable, "v216 disables H8 imagination PPO"
    assert args.joint_imagine_horizon == 5 and args.joint_imagine_batch_size > 0
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    agent.assert_zero_forecast_residual_init()
    # v103: AdamW with a WM-only weight-decay group (le-wm uses AdamW wd=1e-3 on the
    # world model). Non-WM params keep weight_decay=0, where AdamW is identical to
    # the previous plain Adam, so PPO dynamics are unchanged. eps preserved at 1e-5.
    world_model_params = list(agent.world_model.parameters())
    wm_param_ids = {id(p) for p in world_model_params}
    non_wm_params = [p for p in agent.parameters() if id(p) not in wm_param_ids]
    optimizer = optim.AdamW(
        [
            {"params": non_wm_params, "weight_decay": 0.0},
            {"params": world_model_params, "weight_decay": args.lewm_weight_decay},
        ],
        lr=args.learning_rate,
        eps=1e-5,
    )
    # Param groups for decoupled clipping. v198 deliberately has no shared
    # actor/critic parameters, so critic updates cannot consume policy KL budget.
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    actor_param_ids = {id(p) for p in actor_params}
    shared_trunk_params = [p for p in critic_params if id(p) in actor_param_ids]
    assert not shared_trunk_params, "v198 actor and critic paths must be disjoint"

    def snapshot_actor_optimizer_state():
        param_values = [p.detach().clone() for p in actor_params]
        state_values = []
        for p in actor_params:
            state_values.append(
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in optimizer.state.get(p, {}).items()
                }
            )
        return param_values, state_values

    @torch.no_grad()
    def restore_actor_optimizer_state(snapshot):
        param_values, state_values = snapshot
        for p, value, saved_state in zip(actor_params, param_values, state_values):
            p.copy_(value)
            current_state = optimizer.state[p]
            current_state.clear()
            current_state.update(saved_state)
    wm_outcome_io_params = (
        list(agent.world_model.reward_token_norm.parameters())
        + list(agent.world_model.continuation_token_norm.parameters())
        + list(agent.world_model.reward_action_proj.parameters())
        + list(agent.world_model.reward_token_unproj.parameters())
        + list(agent.world_model.continuation_token_unproj.parameters())
    )
    wm_outcome_io_param_ids = {id(p) for p in wm_outcome_io_params}
    wm_latent_params = [p for p in world_model_params if id(p) not in wm_outcome_io_param_ids]
    policy_forecast_params = list(agent.world_model.policy_moment_outcome.parameters())
    policy_forecast_param_ids = {id(p) for p in policy_forecast_params}
    base_wm_params = [
        p for p in wm_latent_params if id(p) not in policy_forecast_param_ids
    ]

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

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,  # HL-Gauss projection sigma (D4 scalar-return target)
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    support = hl_support.support                       # (num_bins,) bin centers
    scalar_support = symexp(support) if args.value_symlog else support

    def value_logits_to_scalar(logits):
        return hl_support.to_expected_scalar(logits)

    def value_probs_to_scalar(probs):
        return hl_support.probs_to_expected_scalar(probs)

    bin_width = hl_support.bin_width
    scalar_bin_width = (
        (scalar_support[1:] - scalar_support[:-1]).abs().min()
        if args.value_symlog
        else bin_width
    )

    reward_support = HLGaussSupport(
        args.lewm_reward_num_bins,
        args.lewm_reward_v_min,
        args.lewm_reward_v_max,
        0.5,
        device,
        use_symlog=True,
    )
    def reward_logits_to_scalar(logits):
        return reward_support.to_expected_scalar(logits)

    with torch.no_grad():
        agent.critic_head.weight.zero_()
        agent.critic_head.bias.copy_(
            hl_support.project_to_logprobs(
                torch.zeros(1, device=device),
                eps=args.critic_prior_floor,
            )[0]
            .repeat(args.critic_mtp_horizon)
        )
        agent.world_model.reward_token_unproj.weight.zero_()
        agent.world_model.reward_token_unproj.bias.copy_(
            reward_support.project_to_logprobs(
                torch.zeros(1, device=device),
                eps=args.lewm_reward_prior_floor,
            )[0]
        )
    sigreg = SIGReg(knots=args.lewm_sigreg_knots, num_proj=args.lewm_sigreg_num_proj).to(device)
    wm_np_rng = np.random.default_rng(args.seed + 20000)
    wm_torch_generator = torch.Generator(device=device)
    wm_torch_generator.manual_seed(args.seed + 20000)

    def masked_token_sigreg(token_latents, token_valids):
        # token_latents: (B, H, S, D), token_valids: (B, H)
        # Faithful to le-wm: whiten each rollout step's batch-marginal
        # independently (pool batch B per step, keep per-step), then average
        # over steps -- so the regularizer enforces per-step isotropy (the
        # constraint an imagination simulator needs) rather than a single
        # marginal pooled over batch*horizon (which lets per-step drift hide).
        # N is pinned to lewm_sigreg_ref_n so SIGReg/MSE balance is batch- and
        # horizon-invariant. One projection sampled per call, shared across steps.
        horizon = token_latents.shape[1]
        A = sigreg.sample_projection(
            token_latents.shape[-1], token_latents.device, token_latents.dtype,
            generator=wm_torch_generator,
        )
        step_losses = []
        for h in range(horizon):
            valid_mask = token_valids[:, h].bool()
            if int(valid_mask.sum().item()) < args.lewm_sigreg_min_valid:
                continue
            # (B, S, D) -> (valid_b, S, D) -> (S, valid_b, D)
            # N = proj.size(-2) = valid_b (the per-step batch), faithful to le-wm
            # which scales the statistic by its batch size. Steps with more valid
            # samples get proportionally more weight (precision weighting).
            step_tokens = token_latents[:, h][valid_mask].transpose(0, 1).contiguous()
            step_losses.append(sigreg(step_tokens, A=A))
        if not step_losses:
            return token_latents.sum() * 0.0
        return torch.stack(step_losses).mean()

    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    agent_inputs = torch.zeros((args.num_steps, args.num_envs, agent.agent_input_dim)).to(device)
    next_transition_agent_inputs = torch.zeros((args.num_steps, args.num_envs, agent.agent_input_dim), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    behavior_alphas = torch.ones_like(latent_zs)
    behavior_betas = torch.ones_like(latent_zs)
    base_behavior_alphas = torch.ones_like(latent_zs)
    base_behavior_betas = torch.ones_like(latent_zs)
    successor_policy_alphas = torch.ones_like(latent_zs)
    successor_policy_betas = torch.ones_like(latent_zs)
    incoming_policy_forecasts = torch.zeros(
        args.num_steps, args.num_envs, agent.forecast_dim, device=device
    )
    explore_codes = torch.zeros(
        (args.num_steps, args.num_envs, args.explore_code_dim), device=device
    )
    next_transition_explore_codes = torch.zeros_like(explore_codes)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rollout_latent_history = []
    rollout_obs_history = []
    rollout_action_history = []
    rollout_history_valid = []
    explore_generator = torch.Generator(device=device)
    explore_generator.manual_seed(args.seed + 40000)
    current_explore_code = torch.zeros(
        args.num_envs, args.explore_code_dim, device=device
    )
    current_incoming_forecast = torch.zeros(
        args.num_envs, agent.forecast_dim, device=device
    )
    imagine_prompt_bank = []

    def build_belief_agent_input(obs_tensor, incoming_forecast=None):
        current_latent = agent.world_model.encode_summary(obs_tensor)
        if agent.detach_world_model_from_agent:
            current_latent = current_latent.detach()
        # Recurrent stream stores obs tokens only; semantic slots are never fed back.
        current_latent = current_latent[:, agent.world_model.obs_token_start :]
        context_len = min(args.lewm_context, len(rollout_latent_history) + 1)
        n_past = context_len - 1
        past_latents = rollout_latent_history[-n_past:] if n_past > 0 else []
        past_valids = rollout_history_valid[-n_past:] if n_past > 0 else []
        if n_past > 0:
            past_latents = [
                latent * valid.view(-1, 1, 1).to(latent.dtype)
                for latent, valid in zip(past_latents, past_valids)
            ]
        belief_latents = torch.stack(past_latents + [current_latent], dim=1)
        if incoming_forecast is None:
            incoming_forecast = current_latent.new_zeros(
                current_latent.shape[0], agent.forecast_dim
            )
        return (
            agent.agent_input_from_history(belief_latents, incoming_forecast),
            current_latent,
            belief_latents,
        )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for pg in optimizer.param_groups:  # v103: anneal both the non-WM and WM groups
                pg["lr"] = lrnow

        rollout_actor_active = global_step >= args.wm_warmup_steps
        # Raw pre-rollout prefix is the source of truth for reconstructing this
        # rollout after the WM changes. Latent caches are intentionally excluded.
        iteration_prefix_obs = [x.detach().clone() for x in rollout_obs_history]
        iteration_prefix_actions = [x.detach().clone() for x in rollout_action_history]
        iteration_prefix_valid = [x.detach().clone() for x in rollout_history_valid]
        agent.forecast_adapters_enabled = (
            global_step >= args.policy_forecast_adapter_warmup_steps
        )
        for step in range(0, args.num_steps):
            if step % args.explore_resample_steps == 0:
                current_explore_code = torch.randn(
                    args.num_envs,
                    args.explore_code_dim,
                    device=device,
                    generator=explore_generator,
                )
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            explore_codes[step] = current_explore_code
            incoming_policy_forecasts[step] = current_incoming_forecast

            with torch.no_grad():
                agent_input, current_latent, belief_latents = build_belief_agent_input(
                    next_obs, current_incoming_forecast
                )
                base_behavior_alpha, base_behavior_beta = (
                    agent.get_base_beta_params_from_agent_input(agent_input)
                )
                if not rollout_actor_active and args.actor_dist == "beta":
                    action = ((agent.action_low + agent.action_high) * 0.5).expand(args.num_envs, -1)
                    z = torch.full_like(action, 0.5)
                    logprob = torch.zeros(args.num_envs, device=device)
                    behavior_alpha = torch.ones_like(action)
                    behavior_beta = torch.ones_like(action)
                    value_logits = agent.get_value_from_agent_input(next_obs, agent_input)
                else:
                    (
                        action,
                        z,
                        logprob,
                        ent,
                        value_logits,
                        behavior_alpha,
                        behavior_beta,
                    ) = agent.get_action_and_value_from_agent_input(
                        next_obs,
                        agent_input,
                        explore_code=current_explore_code,
                        return_beta_params=True,
                    )
                p = torch.softmax(value_logits[:, 0], dim=-1)   # horizon 0 = V(s_t)
                agent_inputs[step] = agent_input
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
                _, _, _, predicted_successor_forecast = (
                    agent.world_model.policy_forecast_from_history(
                        belief_latents, action
                    )
                )
                predicted_successor_forecast = predicted_successor_forecast[:, 0]
            actions[step] = action
            latent_zs[step] = z
            behavior_alphas[step] = behavior_alpha
            behavior_betas[step] = behavior_beta
            base_behavior_alphas[step] = base_behavior_alpha
            base_behavior_betas[step] = base_behavior_beta
            logprobs[step] = logprob
            rollout_latent_history.append(current_latent.detach())
            rollout_obs_history.append(obs[step].detach().clone())
            rollout_action_history.append(action.detach())
            rollout_history_valid.append(torch.ones(args.num_envs, device=device, dtype=torch.bool))
            if len(rollout_latent_history) > args.lewm_context - 1:
                rollout_latent_history = rollout_latent_history[-(args.lewm_context - 1) :]
                rollout_obs_history = rollout_obs_history[-(args.lewm_context - 1) :]
                rollout_action_history = rollout_action_history[-(args.lewm_context - 1) :]
                rollout_history_valid = rollout_history_valid[-(args.lewm_context - 1) :]

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_termination = terminations
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                transition_next_obs = np.array(next_obs_np, copy=True)
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
                transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            else:
                transition_next_obs_t = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            termination_tensor = torch.as_tensor(transition_termination, device=device, dtype=torch.float32)
            boundary_tensor_f = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            rewards[step] = reward_tensor
            transition_terminations[step] = termination_tensor
            transition_boundaries[step] = boundary_tensor_f
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            next_obses[step] = transition_next_obs_t
            # The transition successor belongs to the same persistent policy context
            # as the action that produced it, including terminal final observations.
            next_transition_explore_codes[step] = current_explore_code
            boundary_tensor = boundary_tensor_f.bool()
            with torch.no_grad():
                next_transition_agent_input, _, _ = build_belief_agent_input(
                    transition_next_obs_t, predicted_successor_forecast
                )
                next_transition_agent_inputs[step] = next_transition_agent_input
                successor_alpha, successor_beta = (
                    agent.get_base_beta_params_from_agent_input(
                        next_transition_agent_input
                    )
                )
                successor_policy_alphas[step] = successor_alpha
                successor_policy_betas[step] = successor_beta
            current_incoming_forecast = predicted_successor_forecast.detach()
            if bool(boundary_tensor.any()):
                for hist_idx in range(len(rollout_history_valid)):
                    rollout_history_valid[hist_idx] = rollout_history_valid[hist_idx] & (~boundary_tensor)
                current_explore_code = current_explore_code.clone()
                current_explore_code[boundary_tensor] = torch.randn(
                    int(boundary_tensor.sum().item()),
                    args.explore_code_dim,
                    device=device,
                    generator=explore_generator,
                )
                current_incoming_forecast = current_incoming_forecast.clone()
                current_incoming_forecast[boundary_tensor] = 0.0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if args.imagine_prompt_bank_rollouts > 0:
            imagine_prompt_bank.append(
                {
                    "obs": obs.detach().clone(),
                    "actions": actions.detach().clone(),
                    "dones": dones.detach().clone(),
                    "policy_forecasts": incoming_policy_forecasts.detach().clone(),
                }
            )
            if len(imagine_prompt_bank) > args.imagine_prompt_bank_rollouts:
                del imagine_prompt_bank[: len(imagine_prompt_bank) - args.imagine_prompt_bank_rollouts]

        with torch.no_grad():
            next_transition_value_logits = agent.get_value_from_agent_input(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                next_transition_agent_inputs.reshape(-1, agent.agent_input_dim),
            )[:, 0]  # horizon 0 = V(s')
            next_transition_value_probs = torch.softmax(
                next_transition_value_logits, dim=-1
            ).reshape(args.num_steps, args.num_envs, args.num_bins)
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
                _, _, boot_logprob, _, _ = agent.get_action_and_value_from_agent_input(
                    next_obses[-1],
                    next_transition_agent_inputs[-1],
                    explore_code=next_transition_explore_codes[-1],
                )
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). The resulting magnitude and zero crossing are
            # preserved until canonical per-minibatch standardization.
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
            # Critic target: Dreamer4-style scalar HL-Gauss returns with Orbit-Wars
            # MTP. Horizon 0 regresses returns[t]; horizon h regresses returns[t+h]
            # from the SAME critic features. Horizons crossing an episode boundary
            # (transition_boundaries) or running off the rollout tail are masked; the
            # loss sums valid horizons per row. returns is the entropy-free RAW reward
            # λ-return (== advantages + values), so the fixed support never overflows.
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
                # A boundary at index j separates s_j from s_{j+1} (s_{j+1} is the
                # first state of a fresh episode). Row t' predicts returns[t'+h], which
                # is in the same episode as s_{t'} iff NONE of steps t'..t'+h-1 is a
                # boundary. So check boundaries at offsets k=0..h-1 (NOT 1..h): a
                # boundary AT the target step t'+h is fine, but one at t' itself leaks.
                for k in range(0, h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            target_probs = hl_support.project(return_mtp)   # (T,B,mtp,num_bins)
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (scalar_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        wm_losses = []
        wm_latent_losses = []
        wm_reward_losses = []
        wm_termination_losses = []
        wm_sigreg_losses = []
        wm_policy_forecast_losses = []
        forecast_mean_abs_sum = torch.zeros((), device=device)
        forecast_logstd_abs_sum = torch.zeros((), device=device)
        forecast_moment_count = torch.zeros((), device=device)
        forecast_mean_stats = torch.zeros(5, device=device)
        forecast_logstd_stats = torch.zeros(5, device=device)
        wm_grad_norms = []
        wm_policy_forecast_grad_norms = []
        wm_outcome_io_grad_norms = []
        wm_reward_abs_error_sums = torch.zeros(args.lewm_mtp_len, device=device)
        wm_reward_error_sums = torch.zeros(args.lewm_mtp_len, device=device)
        wm_reward_pred_edge_sums = torch.zeros(args.lewm_mtp_len, device=device)
        wm_reward_target_edge_sums = torch.zeros(args.lewm_mtp_len, device=device)
        wm_reward_probe_weights = torch.zeros(args.lewm_mtp_len, device=device)
        # Labeled AR diagnostics, accumulated per horizon offset h=0..H-1.
        # length1_ar is a harsh stress test from a single replay state. warm_ar
        # starts from the same full real context window used by actual imagination.
        wm_length1_ar_latent_mse_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_reward_abs_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_reward_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_continuation_bce_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_weight = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_latent_mse_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_reward_abs_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_reward_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_continuation_bce_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_weight = torch.zeros(args.lewm_dyn_horizon, device=device)
        if args.lewm_loss_coef > 0.0:
            wm_b_inds = np.arange(args.batch_size)
            horizon = args.lewm_dyn_horizon
            mtp_len = args.lewm_mtp_len
            max_start = args.num_steps - 1
            obs_shape = envs.single_observation_space.shape
            wm_minibatch_size = min(args.minibatch_size, args.lewm_minibatch_size)
            wm_accum_steps = max(
                1,
                (args.lewm_effective_minibatch_size + wm_minibatch_size - 1) // wm_minibatch_size,
            )
            for _ in range(args.lewm_update_epochs):
                wm_np_rng.shuffle(wm_b_inds)
                wm_accum_count = 0
                wm_accum_divisor = wm_accum_steps
                for start in range(0, args.batch_size, wm_minibatch_size):
                    end = start + wm_minibatch_size
                    mb_inds_np = wm_b_inds[start:end]
                    mb_step_inds = torch.as_tensor(mb_inds_np // args.num_envs, device=device, dtype=torch.long)
                    mb_env_inds = torch.as_tensor(mb_inds_np % args.num_envs, device=device, dtype=torch.long)
                    mb_size = mb_step_inds.numel()
                    hist_offsets = torch.arange(horizon, device=device)
                    hist_step_inds = mb_step_inds[:, None] + hist_offsets[None, :]
                    hist_in_rollout = hist_step_inds < args.num_steps
                    safe_hist_step_inds = hist_step_inds.clamp(max=max_start)
                    env_inds = mb_env_inds[:, None].expand_as(safe_hist_step_inds)

                    future_actions = actions[safe_hist_step_inds, env_inds]
                    future_rewards = rewards[safe_hist_step_inds, env_inds]
                    future_terminations = transition_terminations[safe_hist_step_inds, env_inds]
                    future_boundaries = transition_boundaries[safe_hist_step_inds, env_inds]
                    future_valids = transition_valids[safe_hist_step_inds, env_inds]
                    future_next_obs = next_obses[safe_hist_step_inds, env_inds]

                    initial_summary = agent.world_model.encode_summary(
                        obs[mb_step_inds, mb_env_inds],
                    )
                    reward_target_probs_flat = reward_support.project(future_rewards.reshape(-1))
                    target_summaries = agent.world_model.encode_summary(
                        future_next_obs.reshape((-1,) + obs_shape),
                    ).reshape(
                        mb_size,
                        horizon,
                        agent.world_model.num_latent_tokens,
                        agent.world_model.dim,
                    )
                    # Recurrent stream is obs-only; semantic slots of the (teacher-forced)
                    # encoder summaries are not fed back as predictor input.
                    # v102: context is NOT detached -- the encoder receives gradient
                    # through the predictor-input path too (faithful to le-wm, where
                    # emb feeds both ctx_emb and tgt_emb with no detach). The encoder
                    # is now shaped as both a good *input* and a good *target* rep.
                    obs_start = agent.world_model.obs_token_start
                    teacher_history = torch.cat(
                        [
                            initial_summary[:, obs_start:].unsqueeze(1),
                            target_summaries[:, :-1, obs_start:],
                        ],
                        dim=1,
                    )
                    pred_mtp, state_memory, preprojection_features = (
                        agent.world_model.predict_mtp_from_history(
                            teacher_history, future_actions, return_training_features=True
                        )
                    )

                    prev_continues = torch.cat(
                        [
                            torch.ones(mb_size, 1, device=device),
                            1.0 - future_boundaries[:, :-1],
                        ],
                        dim=1,
                    )
                    step_weight = torch.cumprod(prev_continues, dim=1) * hist_in_rollout.float()
                    latent_weight = step_weight * future_valids
                    # Only the final teacher-forced row has v198's full five-state
                    # causal context, matching the fixed rollout successor target.
                    # Earlier rows would train a short-context forecast against a
                    # full-context behavior-policy label.
                    forecast_weight = (
                        latent_weight[:, -1:] * (1.0 - future_terminations[:, -1:])
                    )
                    target_policy_alpha = successor_policy_alphas[
                        safe_hist_step_inds[:, -1:], env_inds[:, -1:]
                    ].detach()
                    target_policy_beta = successor_policy_betas[
                        safe_hist_step_inds[:, -1:], env_inds[:, -1:]
                    ].detach()
                    (
                        _,
                        pred_policy_alpha,
                        pred_policy_beta,
                        _,
                    ) = agent.world_model.policy_forecast_from_state_memory(
                        state_memory[:, -1:], future_actions[:, -1:]
                    )
                    policy_forecast_kl = torch.distributions.kl_divergence(
                        Beta(target_policy_alpha, target_policy_beta),
                        Beta(pred_policy_alpha, pred_policy_beta),
                    ).sum(dim=-1)
                    forecast_denom = forecast_weight.sum().clamp_min(1.0)
                    wm_policy_forecast_loss = (
                        policy_forecast_kl * forecast_weight
                    ).sum() / forecast_denom
                    with torch.no_grad():
                        pred_mean, pred_std = PolicyMomentOutcomeQuery.beta_native_moments(
                            pred_policy_alpha, pred_policy_beta
                        )
                        target_mean, target_std = PolicyMomentOutcomeQuery.beta_native_moments(
                            target_policy_alpha, target_policy_beta
                        )
                        action_scale = (agent.action_high - agent.action_low).view(1, 1, -1)
                        pred_env_mean = agent.action_low.view(1, 1, -1) + action_scale * pred_mean
                        target_env_mean = agent.action_low.view(1, 1, -1) + action_scale * target_mean
                        pred_env_logstd = (action_scale * pred_std).clamp_min(1e-8).log().clamp(
                            args.policy_forecast_logstd_min,
                            args.policy_forecast_logstd_max + log(2.0),
                        )
                        target_env_logstd = (action_scale * target_std).clamp_min(1e-8).log().clamp(
                            args.policy_forecast_logstd_min,
                            args.policy_forecast_logstd_max + log(2.0),
                        )
                        moment_weight = forecast_weight.unsqueeze(-1).expand_as(pred_env_mean)
                        forecast_mean_abs_sum += (
                            (pred_env_mean - target_env_mean).abs() * moment_weight
                        ).sum()
                        forecast_logstd_abs_sum += (
                            (pred_env_logstd - target_env_logstd).abs() * moment_weight
                        ).sum()
                        forecast_moment_count += moment_weight.sum()

                        def accumulate_corr(stats, pred_value, target_value):
                            valid_pred = pred_value * moment_weight
                            valid_target = target_value * moment_weight
                            stats[0] += valid_pred.sum()
                            stats[1] += valid_target.sum()
                            stats[2] += (pred_value.square() * moment_weight).sum()
                            stats[3] += (target_value.square() * moment_weight).sum()
                            stats[4] += (pred_value * target_value * moment_weight).sum()

                        accumulate_corr(forecast_mean_stats, pred_env_mean, target_env_mean)
                        accumulate_corr(
                            forecast_logstd_stats, pred_env_logstd, target_env_logstd
                        )
                    reward_target_probs = reward_target_probs_flat.reshape(mb_size, horizon, -1)
                    # Labeled AR diagnostics are detached. They measure replay-action
                    # rollouts against real labels without changing the training path.
                    with torch.no_grad():
                        def accumulate_labeled_ar(
                            hist_latents,
                            hist_actions,
                            extra_weight,
                            latent_mse_sum,
                            reward_abs_error_sum,
                            reward_error_sum,
                            continuation_bce_sum,
                            weight_sum,
                        ):
                            for h in range(horizon):
                                hist_actions.append(future_actions[:, h])
                                (
                                    ar_next_summary,
                                    ar_reward_logits,
                                    ar_continuation_logits,
                                ) = agent.world_model.imagine_step_from_history(
                                    torch.stack(hist_latents, dim=1),
                                    torch.stack(hist_actions, dim=1),
                                )
                                w_h = latent_weight[:, h] * extra_weight
                                ar_latent_mse = F.mse_loss(
                                    ar_next_summary[:, obs_start:],
                                    target_summaries[:, h, obs_start:],
                                    reduction="none",
                                ).mean(dim=(-1, -2))
                                ar_reward_scalar = reward_logits_to_scalar(ar_reward_logits)
                                ar_reward_error = ar_reward_scalar - future_rewards[:, h]
                                ar_continuation_bce = F.binary_cross_entropy_with_logits(
                                    ar_continuation_logits,
                                    1.0 - future_terminations[:, h],
                                    reduction="none",
                                )
                                latent_mse_sum[h] += (ar_latent_mse * w_h).sum()
                                reward_abs_error_sum[h] += (ar_reward_error.abs() * w_h).sum()
                                reward_error_sum[h] += (ar_reward_error * w_h).sum()
                                continuation_bce_sum[h] += (ar_continuation_bce * w_h).sum()
                                weight_sum[h] += w_h.sum()
                                hist_latents.append(ar_next_summary[:, obs_start:].detach())

                        # Harsh stress diagnostic: one replay state, then self-fed obs
                        # latents. This intentionally does not match actual imagination's
                        # real-context warm start.
                        if args.diagnostics_length1_ar:
                            accumulate_labeled_ar(
                                [initial_summary[:, obs_start:].detach()],
                                [],
                                torch.ones(mb_size, device=device),
                                wm_length1_ar_latent_mse_sum,
                                wm_length1_ar_reward_abs_error_sum,
                                wm_length1_ar_reward_error_sum,
                                wm_length1_ar_continuation_bce_sum,
                                wm_length1_ar_weight,
                            )

                        # Consumed-path diagnostic: start from the same clean full real
                        # context window as actual imagination, then roll on replay
                        # actions so labels remain available.
                        if args.diagnostics_warm_ar:
                            warm_context = min(agent.world_model.context, args.num_steps)
                            warm_valid = mb_step_inds >= (warm_context - 1)
                            for d in range(0, warm_context - 1):
                                warm_valid &= dones[(mb_step_inds - d).clamp_min(0), mb_env_inds] == 0
                            if bool(warm_valid.any()):
                                win_d = torch.arange(warm_context - 1, -1, -1, device=device)
                                win_steps = (mb_step_inds[:, None] - win_d[None, :]).clamp(0, max_start)
                                win_envs = mb_env_inds[:, None].expand_as(win_steps)
                                win_summaries = agent.world_model.encode_summary(
                                    obs[win_steps, win_envs].reshape((-1,) + obs_shape)
                                ).reshape(
                                    mb_size,
                                    warm_context,
                                    agent.world_model.num_latent_tokens,
                                    agent.world_model.dim,
                                )
                                win_actions = actions[
                                    win_steps[:, :-1],
                                    win_envs[:, :-1],
                                ]
                                accumulate_labeled_ar(
                                    [win_summaries[:, d, obs_start:].detach() for d in range(warm_context)],
                                    [win_actions[:, d] for d in range(max(0, warm_context - 1))],
                                    warm_valid.to(dtype=latent_weight.dtype),
                                    wm_warm_ar_latent_mse_sum,
                                    wm_warm_ar_reward_abs_error_sum,
                                    wm_warm_ar_reward_error_sum,
                                    wm_warm_ar_continuation_bce_sum,
                                    wm_warm_ar_weight,
                                )
                    one_step_valid = latent_weight
                    one_step_denom = one_step_valid.sum().clamp_min(1.0)
                    one_step_pred = pred_mtp[:, :, 0, agent.world_model.obs_token_start :]
                    one_step_target = target_summaries[:, :, agent.world_model.obs_token_start :]
                    one_step_latent_loss = F.mse_loss(
                        one_step_pred, one_step_target, reduction="none"
                    ).mean(dim=(-1, -2))
                    wm_latent_loss = (one_step_latent_loss * one_step_valid).sum() / one_step_denom

                    # v115: reward/continue MTP mirrors the predicted-summary
                    # projections. Each offset k decodes pred_mtp[..., k] and is
                    # compared against labels k steps ahead. The losses are summed
                    # across offsets, then scaled by lewm_*_loss_coef.
                    pred_outcome_actions = torch.zeros(
                        mb_size,
                        horizon,
                        mtp_len,
                        future_actions.shape[-1],
                        device=device,
                        dtype=future_actions.dtype,
                    )
                    for mtp_idx in range(mtp_len):
                        valid_horizon = horizon - mtp_idx
                        if valid_horizon > 0:
                            pred_outcome_actions[:, :valid_horizon, mtp_idx] = future_actions[:, mtp_idx:]
                    pred_reward_logits_all, pred_continuation_logits_all = agent.world_model.decode_belief_outcomes(
                        pred_mtp,
                        pred_outcome_actions,
                        detach_belief=False,
                    )

                    reward_losses = []
                    termination_losses = []
                    for mtp_idx in range(mtp_len):
                        valid_horizon = horizon - mtp_idx
                        if valid_horizon <= 0:
                            continue
                        offset_valid = latent_weight[:, mtp_idx:]
                        denom = offset_valid.sum().clamp_min(1.0)
                        reward_loss = -(
                            reward_target_probs[:, mtp_idx:].detach()
                            * torch.log_softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
                        ).sum(dim=-1)
                        reward_losses.append((reward_loss * offset_valid).sum() / denom)
                        with torch.no_grad():
                            pred_reward_scalar = reward_logits_to_scalar(
                                pred_reward_logits_all[:, :valid_horizon, mtp_idx]
                            )
                            target_reward_scalar = future_rewards[:, mtp_idx:]
                            reward_error = pred_reward_scalar - target_reward_scalar
                            wm_reward_abs_error_sums[mtp_idx] += (reward_error.abs() * offset_valid).sum()
                            wm_reward_error_sums[mtp_idx] += (reward_error * offset_valid).sum()
                            pred_reward_probs = torch.softmax(
                                pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1
                            )
                            pred_edge_mass = pred_reward_probs[..., 0] + pred_reward_probs[..., -1]
                            wm_reward_pred_edge_sums[mtp_idx] += (pred_edge_mass * offset_valid).sum()
                            target_edge_frac = (
                                (target_reward_scalar <= args.lewm_reward_raw_v_min)
                                | (target_reward_scalar >= args.lewm_reward_raw_v_max)
                            ).to(offset_valid.dtype)
                            wm_reward_target_edge_sums[mtp_idx] += (target_edge_frac * offset_valid).sum()
                            wm_reward_probe_weights[mtp_idx] += denom
                        termination_loss = F.binary_cross_entropy_with_logits(
                            pred_continuation_logits_all[:, :valid_horizon, mtp_idx],
                            1.0 - future_terminations[:, mtp_idx:],
                            reduction="none",
                        )
                        termination_losses.append((termination_loss * offset_valid).sum() / denom)
                    if not reward_losses:
                        continue
                    wm_pred_reward_loss = torch.stack(reward_losses).sum()
                    wm_pred_termination_loss = torch.stack(termination_losses).sum()
                    wm_reward_loss = wm_pred_reward_loss
                    wm_termination_loss = wm_pred_termination_loss
                    # v217: regularize the consumed predictive representation,
                    # after causal temporal mixing and action-conditioned query
                    # readout but before the shared next/MTP projectors. Keeping
                    # only observation slots preserves the prior protection for
                    # near-constant continuation/reward query semantics.
                    wm_sigreg_loss = masked_token_sigreg(
                        preprojection_features[:, :, obs_start:],
                        latent_weight,
                    )
                    wm_loss = (
                        args.lewm_loss_coef * wm_latent_loss
                        + args.lewm_reward_loss_coef * wm_reward_loss
                        + args.lewm_termination_loss_coef * wm_termination_loss
                        + args.lewm_sigreg_coef * wm_sigreg_loss
                        + args.policy_forecast_loss_coef * wm_policy_forecast_loss
                    )
                    if wm_accum_count == 0:
                        optimizer.zero_grad(set_to_none=True)
                        remaining_microbatches = (
                            args.batch_size - start + wm_minibatch_size - 1
                        ) // wm_minibatch_size
                        wm_accum_divisor = min(wm_accum_steps, remaining_microbatches)
                    (wm_loss / wm_accum_divisor).backward()
                    wm_accum_count += 1
                    if wm_accum_count >= wm_accum_steps or end >= args.batch_size:
                        wm_gn = nn.utils.clip_grad_norm_(base_wm_params, args.lewm_grad_clip)
                        forecast_gn = nn.utils.clip_grad_norm_(
                            policy_forecast_params, args.lewm_grad_clip
                        )
                        wm_outcome_io_gn = nn.utils.clip_grad_norm_(
                            wm_outcome_io_params,
                            args.lewm_grad_clip,
                        )
                        optimizer.step()
                        wm_grad_norms.append(float(wm_gn))
                        wm_policy_forecast_grad_norms.append(float(forecast_gn))
                        wm_outcome_io_grad_norms.append(float(wm_outcome_io_gn))
                        wm_accum_count = 0
                    wm_losses.append(wm_loss.item())
                    wm_latent_losses.append(wm_latent_loss.item())
                    wm_reward_losses.append(wm_reward_loss.item())
                    wm_termination_losses.append(wm_termination_loss.item())
                    wm_sigreg_losses.append(wm_sigreg_loss.item())
                    wm_policy_forecast_losses.append(wm_policy_forecast_loss.item())

        def log_policy_forecast_diagnostics():
            if not wm_policy_forecast_losses:
                return
            count = forecast_moment_count.clamp_min(1.0)

            def correlation(stats):
                pred_mean = stats[0] / count
                target_mean = stats[1] / count
                pred_var = (stats[2] / count - pred_mean.square()).clamp_min(0.0)
                target_var = (stats[3] / count - target_mean.square()).clamp_min(0.0)
                covariance = stats[4] / count - pred_mean * target_mean
                return covariance / (pred_var * target_var).sqrt().clamp_min(1e-8)

            writer.add_scalar(
                "policy_forecast/beta_forward_kl",
                float(np.mean(wm_policy_forecast_losses)),
                global_step,
            )
            writer.add_scalar(
                "policy_forecast/env_mean_mae",
                (forecast_mean_abs_sum / count).item(),
                global_step,
            )
            writer.add_scalar(
                "policy_forecast/env_logstd_mae",
                (forecast_logstd_abs_sum / count).item(),
                global_step,
            )
            writer.add_scalar(
                "policy_forecast/env_mean_correlation",
                correlation(forecast_mean_stats).item(),
                global_step,
            )
            writer.add_scalar(
                "policy_forecast/env_logstd_correlation",
                correlation(forecast_logstd_stats).item(),
                global_step,
            )
            if wm_policy_forecast_grad_norms and wm_grad_norms:
                forecast_gn = float(np.mean(wm_policy_forecast_grad_norms))
                base_gn = float(np.mean(wm_grad_norms))
                writer.add_scalar(
                    "policy_forecast/private_grad_norm", forecast_gn, global_step
                )
                writer.add_scalar(
                    "policy_forecast/private_to_base_grad_ratio",
                    forecast_gn / max(base_gn, 1e-12),
                    global_step,
                )

        world_model_only = not rollout_actor_active
        if world_model_only:
            writer.add_scalar("charts/world_model_only", 1.0, global_step)
            if wm_losses:
                log_policy_forecast_diagnostics()
                writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
                writer.add_scalar("lewm/obs_latent_mse", float(np.mean(wm_latent_losses)), global_step)
                writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
                writer.add_scalar("lewm/continuation_bce", float(np.mean(wm_termination_losses)), global_step)
                writer.add_scalar("lewm/obs_sigreg", float(np.mean(wm_sigreg_losses)), global_step)
                writer.add_scalar("lewm/grad_norm", float(np.mean(wm_grad_norms)), global_step)
                writer.add_scalar(
                    "lewm/outcome_io_grad_norm",
                    float(np.mean(wm_outcome_io_grad_norms)),
                    global_step,
                )
                total_probe_weight = wm_reward_probe_weights.sum()
                if total_probe_weight.item() > 0:
                    writer.add_scalar(
                        "lewm/reward_scalar_mae",
                        (wm_reward_abs_error_sums.sum() / total_probe_weight).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "lewm/reward_scalar_bias",
                        (wm_reward_error_sums.sum() / total_probe_weight).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "lewm/reward_pred_edge_mass",
                        (wm_reward_pred_edge_sums.sum() / total_probe_weight).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "lewm/reward_target_edge_frac",
                        (wm_reward_target_edge_sums.sum() / total_probe_weight).item(),
                        global_step,
                    )
                    for mtp_idx in range(args.lewm_mtp_len):
                        offset_weight = wm_reward_probe_weights[mtp_idx]
                        if offset_weight.item() > 0:
                            writer.add_scalar(
                                f"lewm/reward_scalar_mae_mtp{mtp_idx + 1}",
                                (wm_reward_abs_error_sums[mtp_idx] / offset_weight).item(),
                                global_step,
                            )
                            writer.add_scalar(
                                f"lewm/reward_scalar_bias_mtp{mtp_idx + 1}",
                                (wm_reward_error_sums[mtp_idx] / offset_weight).item(),
                                global_step,
                            )
                            writer.add_scalar(
                                f"lewm/reward_pred_edge_mass_mtp{mtp_idx + 1}",
                                (wm_reward_pred_edge_sums[mtp_idx] / offset_weight).item(),
                                global_step,
                            )
                            writer.add_scalar(
                                f"lewm/reward_target_edge_frac_mtp{mtp_idx + 1}",
                                (wm_reward_target_edge_sums[mtp_idx] / offset_weight).item(),
                                global_step,
                            )
                if wm_length1_ar_weight.sum().item() > 0:
                    log_labeled_rollout_diagnostic(
                        writer, "diagnostics/length1_ar", global_step,
                        wm_length1_ar_latent_mse_sum, wm_length1_ar_reward_abs_error_sum,
                        wm_length1_ar_reward_error_sum, wm_length1_ar_continuation_bce_sum,
                        wm_length1_ar_weight,
                    )
                if wm_warm_ar_weight.sum().item() > 0:
                    log_labeled_rollout_diagnostic(
                        writer, "diagnostics/warm_ar", global_step,
                        wm_warm_ar_latent_mse_sum, wm_warm_ar_reward_abs_error_sum,
                        wm_warm_ar_reward_error_sum, wm_warm_ar_continuation_bce_sum,
                        wm_warm_ar_weight,
                    )
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            continue
        writer.add_scalar("charts/world_model_only", 0.0, global_step)

        # ===================== CURRENT-WM INTERFACE REPAIR =====================
        # Rebuild every policy/value input from raw causal history after the WM
        # update. This makes representation drift visible to the same stored-
        # behavior trust region as PPO instead of silently changing the policy.
        with torch.no_grad():
            fresh_agent_inputs = torch.zeros_like(agent_inputs)
            fresh_next_agent_inputs = torch.zeros_like(next_transition_agent_inputs)
            fresh_values = torch.zeros_like(values)
            fresh_value_probs = torch.zeros_like(value_probs)

            def encode_repair_rollout(raw_obs):
                flat_obs = raw_obs.reshape(
                    (-1,) + envs.single_observation_space.shape
                )
                encoded = []
                for encode_start in range(0, flat_obs.shape[0], 4096):
                    encoded.append(
                        agent.world_model.encode_summary(
                            flat_obs[encode_start : encode_start + 4096]
                        )
                    )
                return torch.cat(encoded, dim=0).reshape(
                    args.num_steps,
                    args.num_envs,
                    agent.world_model.num_latent_tokens,
                    agent.world_model.dim,
                )

            repaired_current_summaries = encode_repair_rollout(obs)
            repaired_successor_summaries = encode_repair_rollout(next_obses)

            raw_history = []
            raw_history_valid = [x.clone() for x in iteration_prefix_valid]
            for prefix_obs in iteration_prefix_obs:
                prefix_latent = agent.world_model.encode_summary(prefix_obs)
                raw_history.append(
                    prefix_latent[:, agent.world_model.obs_token_start :].detach()
                )
            prefix_actions = [x.clone() for x in iteration_prefix_actions]

            if raw_history and prefix_actions:
                masked_prefix = [
                    latent * valid.view(-1, 1, 1).to(latent.dtype)
                    for latent, valid in zip(raw_history, raw_history_valid)
                ]
                prefix_latent_history = torch.stack(masked_prefix, dim=1)
                _, _, _, repaired_forecast = agent.world_model.policy_forecast_from_history(
                    prefix_latent_history, prefix_actions[-1]
                )
                repaired_forecast = repaired_forecast[:, 0]
            else:
                repaired_forecast = torch.zeros(
                    args.num_envs, agent.forecast_dim, device=device
                )
            repaired_forecast = torch.where(
                dones[0].bool().unsqueeze(-1),
                torch.zeros_like(repaired_forecast),
                repaired_forecast,
            )

            for repair_t in range(args.num_steps):
                current_summary = repaired_current_summaries[repair_t]
                current_latent = current_summary[:, agent.world_model.obs_token_start :].detach()
                n_past = min(args.lewm_context - 1, len(raw_history))
                past_latents = raw_history[-n_past:] if n_past > 0 else []
                past_valids = raw_history_valid[-n_past:] if n_past > 0 else []
                masked_past = [
                    latent * valid.view(-1, 1, 1).to(latent.dtype)
                    for latent, valid in zip(past_latents, past_valids)
                ]
                belief_history = torch.stack(masked_past + [current_latent], dim=1)
                repaired_input = agent.agent_input_from_history(
                    belief_history, repaired_forecast
                )
                fresh_agent_inputs[repair_t] = repaired_input
                repaired_value_logits = agent.get_value_from_agent_input(
                    obs[repair_t], repaired_input
                )
                fresh_value_probs[repair_t] = torch.softmax(
                    repaired_value_logits[:, 0], dim=-1
                )
                fresh_values[repair_t] = value_logits_to_scalar(
                    repaired_value_logits[:, 0]
                )
                _, _, _, successor_forecast = agent.world_model.policy_forecast_from_history(
                    belief_history, actions[repair_t]
                )
                successor_forecast = successor_forecast[:, 0]

                raw_history.append(current_latent)
                raw_history_valid.append(
                    torch.ones(args.num_envs, device=device, dtype=torch.bool)
                )
                if len(raw_history) > args.lewm_context - 1:
                    raw_history = raw_history[-(args.lewm_context - 1) :]
                    raw_history_valid = raw_history_valid[-(args.lewm_context - 1) :]

                successor_summary = repaired_successor_summaries[repair_t]
                successor_latent = successor_summary[
                    :, agent.world_model.obs_token_start :
                ].detach()
                successor_past = [
                    latent * valid.view(-1, 1, 1).to(latent.dtype)
                    for latent, valid in zip(raw_history, raw_history_valid)
                ]
                successor_history = torch.stack(
                    successor_past + [successor_latent], dim=1
                )
                fresh_next_agent_inputs[repair_t] = agent.agent_input_from_history(
                    successor_history, successor_forecast
                )

                boundary = transition_boundaries[repair_t].bool()
                if bool(boundary.any()):
                    raw_history_valid = [valid & (~boundary) for valid in raw_history_valid]
                repaired_forecast = torch.where(
                    boundary.unsqueeze(-1),
                    torch.zeros_like(successor_forecast),
                    successor_forecast,
                )

            agent_inputs.copy_(fresh_agent_inputs)
            next_transition_agent_inputs.copy_(fresh_next_agent_inputs)
            values.copy_(fresh_values)
            value_probs.copy_(fresh_value_probs)

            fresh_next_logits = agent.get_value_from_agent_input(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                fresh_next_agent_inputs.reshape(-1, agent.agent_input_dim),
            )[:, 0]
            next_transition_values = value_logits_to_scalar(fresh_next_logits).reshape(
                args.num_steps, args.num_envs
            )
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for repair_t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (
                    (1.0 - transition_terminations[repair_t])
                    * transition_valids[repair_t]
                )
                lambda_nonterminal = 1.0 - transition_boundaries[repair_t]
                delta = (
                    rewards[repair_t]
                    + args.gamma * next_transition_values[repair_t] * bootstrap_nonterminal
                    - values[repair_t]
                )
                advantages[repair_t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            policy_adv = advantages

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            target_probs = hl_support.project(return_mtp)
            sigma = (
                value_probs
                * (scalar_support - values.unsqueeze(-1)).square()
            ).sum(-1).clamp_min(0).sqrt()
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = (
                (returns_coord.unsqueeze(-1) - support) / bin_width + 0.5
            ).clamp(0.0, 1.0)
            u = (value_probs * cdf_frac).sum(dim=-1)
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()

        fresh_invalid_fraction = 1.0 - transition_valids.mean().item()

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_agent_inputs = agent_inputs.reshape(-1, agent.agent_input_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_behavior_alphas = behavior_alphas.reshape((-1,) + envs.single_action_space.shape)
        b_behavior_betas = behavior_betas.reshape((-1,) + envs.single_action_space.shape)
        b_base_behavior_alphas = base_behavior_alphas.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_base_behavior_betas = base_behavior_betas.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_explore_codes = explore_codes.reshape(-1, args.explore_code_dim)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        b_u = u.reshape(-1)

        @torch.no_grad()
        def measure_behavior_beta_kl(indices):
            measured_alpha, measured_beta = agent.get_beta_params_from_agent_input(
                b_agent_inputs[indices], b_explore_codes[indices]
            )
            behavior_dist = Beta(b_behavior_alphas[indices], b_behavior_betas[indices])
            measured_dist = Beta(measured_alpha, measured_beta)
            return torch.distributions.kl_divergence(
                behavior_dist, measured_dist
            ).sum(dim=-1).mean()

        @torch.no_grad()
        def measure_base_beta_kl(indices):
            measured_alpha, measured_beta = (
                agent.get_base_beta_params_from_agent_input(b_agent_inputs[indices])
            )
            base_behavior_dist = Beta(
                b_base_behavior_alphas[indices], b_base_behavior_betas[indices]
            )
            measured_dist = Beta(measured_alpha, measured_beta)
            return torch.distributions.kl_divergence(
                base_behavior_dist, measured_dist
            ).sum(dim=-1).mean()

        @torch.no_grad()
        def measure_full_real_anchor_beta_kls():
            behavior_sum = torch.zeros((), device=device)
            base_sum = torch.zeros((), device=device)
            count = 0
            for anchor_start in range(0, args.batch_size, 4096):
                anchor_end = min(args.batch_size, anchor_start + 4096)
                inds = torch.arange(anchor_start, anchor_end, device=device)
                behavior_sum += measure_behavior_beta_kl(inds) * (anchor_end - anchor_start)
                base_sum += measure_base_beta_kl(inds) * (anchor_end - anchor_start)
                count += anchor_end - anchor_start
            return behavior_sum / count, base_sum / count

        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        trust_anchor_n = min(4096, args.batch_size)
        trust_anchor_inds = torch.as_tensor(
            np.linspace(0, args.batch_size - 1, trust_anchor_n, dtype=np.int64),
            device=device,
            dtype=torch.long,
        )
        clipfracs = []
        exact_beta_kls = []
        base_exact_beta_kls = []
        actor_grad_norms = []
        actor_updates_used = 0
        actor_updates_rejected = 0
        joint_dream_pg_losses = []
        joint_dream_value_losses = []
        joint_real_pg_losses = []
        joint_real_value_losses = []
        joint_dream_minibatch_kls = []
        joint_dream_full_kls = []
        ppo_wm_grad_isolated = 1.0

        def audit_wm_gradient_isolation():
            if any(param.grad is not None for param in world_model_params):
                raise RuntimeError("PPO/value gradient leaked into WM policy forecast")

        fresh_pre_repair_behavior_kl, fresh_pre_repair_base_kl = (
            measure_full_real_anchor_beta_kls()
        )

        # ======================= FROZEN-WM H5 DREAM DATA =======================
        H_dream = args.joint_imagine_horizon
        B_dream = args.joint_imagine_batch_size
        W_dream = min(agent.world_model.context, args.num_steps)
        dream_clean = (
            torch.arange(args.num_steps, device=device) >= (W_dream - 1)
        ).view(args.num_steps, 1).expand(-1, args.num_envs).clone()
        for d in range(W_dream - 1):
            idx = (torch.arange(args.num_steps, device=device) - d).clamp_min(0)
            dream_clean &= dones[idx] == 0
        clean_dream_flat = dream_clean.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if clean_dream_flat.numel() == 0:
            raise RuntimeError("v216 requires at least one clean full-context dream prompt")
        chosen = clean_dream_flat[
            torch.randint(0, clean_dream_flat.numel(), (B_dream,), device=device)
        ]
        dream_step = chosen // args.num_envs
        dream_env = chosen % args.num_envs
        win_d = torch.arange(W_dream - 1, -1, -1, device=device)
        win_steps = dream_step[:, None] - win_d[None, :]
        win_envs = dream_env[:, None].expand_as(win_steps)
        with torch.no_grad():
            dream_summary = agent.world_model.encode_summary(
                obs[win_steps, win_envs].reshape(
                    (-1,) + envs.single_observation_space.shape
                )
            ).reshape(
                B_dream,
                W_dream,
                agent.world_model.num_latent_tokens,
                agent.world_model.dim,
            )
            dream_history = dream_summary[
                :, :, agent.world_model.obs_token_start :
            ].detach()
            dream_input = b_agent_inputs[chosen]
            dream_code = b_explore_codes[chosen]
            dream_inputs = []
            dream_zs = []
            dream_logprobs = []
            dream_alphas = []
            dream_betas = []
            dream_value_probs = []
            dream_rewards = []
            dream_continues = []
            for _ in range(H_dream):
                (
                    dream_action,
                    dream_z,
                    dream_logprob,
                    _,
                    dream_value_logits,
                    dream_alpha,
                    dream_beta,
                ) = agent.get_action_and_value_from_agent_input(
                    None,
                    dream_input,
                    explore_code=dream_code,
                    return_beta_params=True,
                )
                next_summary, reward_logits, continue_logits = (
                    agent.world_model.imagine_step_from_history(
                        dream_history, dream_action[:, None]
                    )
                )
                _, _, _, next_forecast = agent.world_model.policy_forecast_from_history(
                    dream_history, dream_action
                )
                next_history = torch.cat(
                    [
                        dream_history,
                        next_summary[:, agent.world_model.obs_token_start :].unsqueeze(1),
                    ],
                    dim=1,
                )
                next_belief = agent.world_model.belief_from_history(next_history)
                next_input = agent.agent_input_from_latent(
                    next_belief, next_forecast[:, 0]
                )
                dream_inputs.append(dream_input)
                dream_zs.append(dream_z)
                dream_logprobs.append(dream_logprob)
                dream_alphas.append(dream_alpha)
                dream_betas.append(dream_beta)
                dream_value_probs.append(torch.softmax(dream_value_logits[:, 0], dim=-1))
                dream_rewards.append(reward_logits_to_scalar(reward_logits))
                dream_continues.append(torch.sigmoid(continue_logits))
                dream_history = next_history
                dream_input = next_input
            dream_bootstrap_probs = torch.softmax(
                agent.get_value_from_agent_input(None, dream_input)[:, 0], dim=-1
            )
            dream_inputs = torch.stack(dream_inputs, dim=0)
            dream_zs = torch.stack(dream_zs, dim=0)
            dream_logprobs = torch.stack(dream_logprobs, dim=0)
            dream_alphas = torch.stack(dream_alphas, dim=0)
            dream_betas = torch.stack(dream_betas, dim=0)
            dream_value_probs = torch.stack(dream_value_probs, dim=0)
            dream_rewards = torch.stack(dream_rewards, dim=0)
            dream_continues = torch.stack(dream_continues, dim=0)
            dream_values = value_probs_to_scalar(dream_value_probs)
            dream_next_probs = torch.cat(
                [dream_value_probs[1:], dream_bootstrap_probs.unsqueeze(0)], dim=0
            )
            dream_next_values = value_probs_to_scalar(dream_next_probs)
            dream_advantages = torch.zeros_like(dream_rewards)
            last_dream_gae = torch.zeros(B_dream, device=device)
            for dream_t in reversed(range(H_dream)):
                dream_delta = (
                    dream_rewards[dream_t]
                    + args.gamma
                    * dream_continues[dream_t]
                    * dream_next_values[dream_t]
                    - dream_values[dream_t]
                )
                last_dream_gae = dream_delta + (
                    args.gamma
                    * args.gae_lambda
                    * dream_continues[dream_t]
                    * last_dream_gae
                )
                dream_advantages[dream_t] = last_dream_gae
            dream_returns = dream_advantages + dream_values
            dream_mtp_returns = dream_returns.new_zeros(
                H_dream, B_dream, args.critic_mtp_horizon
            )
            dream_mtp_mask = torch.zeros(
                H_dream,
                B_dream,
                args.critic_mtp_horizon,
                dtype=torch.bool,
                device=device,
            )
            for h in range(args.critic_mtp_horizon):
                valid_len = H_dream - h
                if valid_len <= 0:
                    break
                dream_mtp_returns[:valid_len, :, h] = dream_returns[h:]
                dream_mtp_mask[:valid_len, :, h] = True
            dream_target_probs = hl_support.project(dream_mtp_returns)
            dream_sigma = (
                dream_value_probs
                * (scalar_support - dream_values.unsqueeze(-1)).square()
            ).sum(-1).clamp_min(0).sqrt()
            dream_sigma = dream_sigma.clamp_min(
                args.sigma_floor_bins * scalar_bin_width
            )
            dream_returns_coord = (
                symlog(dream_returns) if args.value_symlog else dream_returns
            )
            dream_cdf_frac = (
                (dream_returns_coord.unsqueeze(-1) - support) / bin_width + 0.5
            ).clamp(0.0, 1.0)
            dream_u = (dream_value_probs * dream_cdf_frac).sum(dim=-1)

        dream_n = H_dream * B_dream
        d_agent_inputs = dream_inputs.reshape(dream_n, agent.agent_input_dim)
        d_zs = dream_zs.reshape(dream_n, *envs.single_action_space.shape)
        d_logprobs = dream_logprobs.reshape(dream_n)
        d_alphas = dream_alphas.reshape(dream_n, *envs.single_action_space.shape)
        d_betas = dream_betas.reshape(dream_n, *envs.single_action_space.shape)
        d_codes = dream_code.unsqueeze(0).expand(H_dream, -1, -1).reshape(
            dream_n, args.explore_code_dim
        )
        d_advantages = dream_advantages.reshape(dream_n)
        d_sigma = dream_sigma.reshape(dream_n)
        d_u = dream_u.reshape(dream_n)
        d_target_probs = dream_target_probs.reshape(
            dream_n, args.critic_mtp_horizon, args.num_bins
        )
        d_target_mask = dream_mtp_mask.reshape(
            dream_n, args.critic_mtp_horizon
        )
        d_inds = np.arange(dream_n)
        dream_minibatch_size = int(np.ceil(dream_n / args.num_minibatches))

        @torch.no_grad()
        def measure_full_dream_behavior_kl():
            kl_sum = torch.zeros((), device=device)
            count = 0
            for dream_start in range(0, dream_n, 4096):
                dream_end = min(dream_n, dream_start + 4096)
                alpha, beta = agent.get_beta_params_from_agent_input(
                    d_agent_inputs[dream_start:dream_end],
                    d_codes[dream_start:dream_end],
                )
                kl_sum += torch.distributions.kl_divergence(
                    Beta(d_alphas[dream_start:dream_end], d_betas[dream_start:dream_end]),
                    Beta(alpha, beta),
                ).sum(dim=-1).sum()
                count += dream_end - dream_start
            return kl_sum / max(1, count)

        fresh_entry_behavior_kl = fresh_pre_repair_behavior_kl
        fresh_entry_base_kl = fresh_pre_repair_base_kl
        dream_entry_behavior_kl = measure_full_dream_behavior_kl()
        real_actor_active = bool(
            torch.isfinite(fresh_entry_behavior_kl)
            and fresh_entry_behavior_kl <= args.target_kl
            and torch.isfinite(fresh_entry_base_kl)
            and fresh_entry_base_kl <= args.target_kl
            and torch.isfinite(dream_entry_behavior_kl)
            and dream_entry_behavior_kl <= args.target_kl
        )
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            np.random.shuffle(d_inds)
            dream_cursor = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                if dream_cursor + dream_minibatch_size > dream_n:
                    np.random.shuffle(d_inds)
                    dream_cursor = 0
                dream_mb_inds = d_inds[
                    dream_cursor : min(dream_n, dream_cursor + dream_minibatch_size)
                ]
                dream_cursor += dream_minibatch_size

                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_logits,
                    current_alpha,
                    current_beta,
                ) = agent.get_action_and_value_from_agent_input(
                    b_obs[mb_inds],
                    b_agent_inputs[mb_inds],
                    b_latent_zs[mb_inds],
                    b_explore_codes[mb_inds],
                    return_beta_params=True,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                (
                    _,
                    _,
                    dream_newlogprob,
                    dream_entropy,
                    dream_value_logits,
                    dream_current_alpha,
                    dream_current_beta,
                ) = agent.get_action_and_value_from_agent_input(
                    None,
                    d_agent_inputs[dream_mb_inds],
                    d_zs[dream_mb_inds],
                    d_codes[dream_mb_inds],
                    return_beta_params=True,
                )
                dream_logratio = dream_newlogprob - d_logprobs[dream_mb_inds]
                dream_ratio = dream_logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    behavior_dist = Beta(
                        b_behavior_alphas[mb_inds], b_behavior_betas[mb_inds]
                    )
                    current_dist = Beta(current_alpha, current_beta)
                    exact_beta_kl = torch.distributions.kl_divergence(
                        behavior_dist, current_dist
                    ).sum(dim=-1).mean()
                    exact_beta_kls.append(exact_beta_kl.item())
                    base_exact_beta_kl = measure_base_beta_kl(mb_inds)
                    base_exact_beta_kls.append(base_exact_beta_kl.item())
                    dream_exact_beta_kl = torch.distributions.kl_divergence(
                        Beta(d_alphas[dream_mb_inds], d_betas[dream_mb_inds]),
                        Beta(dream_current_alpha, dream_current_beta),
                    ).sum(dim=-1).mean()
                    joint_dream_minibatch_kls.append(dream_exact_beta_kl.item())
                    if real_actor_active and (
                        not torch.isfinite(exact_beta_kl)
                        or exact_beta_kl > args.target_kl
                        or not torch.isfinite(base_exact_beta_kl)
                        or base_exact_beta_kl > args.target_kl
                        or not torch.isfinite(dream_exact_beta_kl)
                        or dream_exact_beta_kl > args.target_kl
                    ):
                        real_actor_active = False
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                dream_mb_raw_adv = shape_advantage(
                    d_advantages[dream_mb_inds],
                    d_sigma[dream_mb_inds],
                    d_u[dream_mb_inds],
                    args,
                    device,
                )
                dream_mb_advantages = dream_mb_raw_adv
                if args.norm_adv:
                    dream_mb_advantages = (
                        dream_mb_advantages - dream_mb_advantages.mean()
                    ) / (dream_mb_advantages.std() + 1e-8)

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
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                dream_pg_loss1 = -dream_mb_advantages * dream_ratio
                dream_pg_loss2 = -dream_mb_advantages * torch.clamp(
                    dream_ratio, 1 - args.clip_coef, 1 + clip_hi
                )
                dream_pg_loss = torch.max(dream_pg_loss1, dream_pg_loss2).mean()
                joint_pg_loss = 0.5 * (pg_loss + dream_pg_loss)
                joint_dream_pg_losses.append(dream_pg_loss.item())
                joint_real_pg_losses.append(pg_loss.item())

                # HL-Gauss MTP value loss: per-horizon CE to the scalar-return target,
                # summed across valid future horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                value_ce = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(dtype=value_ce.dtype)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()
                dream_value_log_probs = torch.log_softmax(dream_value_logits, dim=-1)
                dream_value_ce = -(
                    d_target_probs[dream_mb_inds] * dream_value_log_probs
                ).sum(dim=-1)
                dream_value_mask = d_target_mask[dream_mb_inds].to(
                    dtype=dream_value_ce.dtype
                )
                dream_v_loss = (
                    dream_value_ce * dream_value_mask
                ).sum(dim=-1).mean()
                joint_v_loss = 0.5 * (v_loss + dream_v_loss)
                joint_dream_value_losses.append(dream_v_loss.item())
                joint_real_value_losses.append(v_loss.item())

                entropy_loss = entropy.mean()
                dream_entropy_loss = dream_entropy.mean()
                joint_entropy_loss = 0.5 * (entropy_loss + dream_entropy_loss)

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

                attempted_actor_update = real_actor_active
                actor_snapshot = (
                    snapshot_actor_optimizer_state() if attempted_actor_update else None
                )
                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * joint_v_loss).backward(retain_graph=True)
                    audit_wm_gradient_isolation()
                    if not real_actor_active:
                        # The critic head may continue fitting, but once the exact
                        # policy trust budget is spent its gradient cannot move the
                        # shared representation underneath the frozen actor.
                        for p in shared_trunk_params:
                            p.grad = None
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    if attempted_actor_update:
                        (
                            joint_pg_loss - ent_coef_eff * joint_entropy_loss
                        ).backward()
                        audit_wm_gradient_isolation()
                        actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                        actor_grad_norms.append(float(actor_gn))
                    else:
                        actor_gn = torch.zeros((), device=device)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                    if attempted_actor_update:
                        post_step_beta_kl, post_step_base_kl = (
                            measure_full_real_anchor_beta_kls()
                        )
                        post_step_dream_kl = measure_full_dream_behavior_kl()
                        exact_beta_kls.append(post_step_beta_kl.item())
                        base_exact_beta_kls.append(post_step_base_kl.item())
                        joint_dream_full_kls.append(post_step_dream_kl.item())
                        if (
                            not torch.isfinite(post_step_beta_kl)
                            or post_step_beta_kl > args.target_kl
                            or not torch.isfinite(post_step_base_kl)
                            or post_step_base_kl > args.target_kl
                            or not torch.isfinite(post_step_dream_kl)
                            or post_step_dream_kl > args.target_kl
                        ):
                            restore_actor_optimizer_state(actor_snapshot)
                            real_actor_active = False
                            actor_updates_rejected += 1
                        else:
                            actor_updates_used += 1
                else:
                    loss = (
                        joint_pg_loss
                        - ent_coef_eff * joint_entropy_loss
                        + joint_v_loss * args.vf_coef
                    )
                    optimizer.zero_grad(set_to_none=True)  # v103: None (not 0) so AdamW wd skips ungraded WM params
                    loss.backward()
                    audit_wm_gradient_isolation()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

        @torch.no_grad()
        def measure_real_anchor_beta_kls():
            return measure_full_real_anchor_beta_kls()

        post_real_anchor_beta_kl, post_real_anchor_base_kl = (
            measure_real_anchor_beta_kls()
        )
        post_joint_dream_kl = measure_full_dream_behavior_kl()
        exact_beta_kls.append(post_real_anchor_beta_kl.item())
        base_exact_beta_kls.append(post_real_anchor_base_kl.item())
        if (
            not torch.isfinite(post_real_anchor_beta_kl)
            or post_real_anchor_beta_kl > args.target_kl
            or not torch.isfinite(post_real_anchor_base_kl)
            or post_real_anchor_base_kl > args.target_kl
            or not torch.isfinite(post_joint_dream_kl)
            or post_joint_dream_kl > args.target_kl
        ):
            real_actor_active = False

        # =========================== IMAGINATION TRAINING ===========================
        # dreamer4-style: with the WM frozen as a DETACHED simulator, generate fresh
        # on-policy rollouts dreamed from real rollout states and train the SAME
        # actor/critic on them at full strength. c = args.imagine_batches
        # generate->train cycles; each is a single 1-epoch gradient step over a
        # horizon x batch block of imagined transitions, regenerated from the
        # just-updated policy so the data stays on-policy (PPO ratio starts at 1).
        imagine_metrics = None
        dream_actor_updates_used = 0
        dream_actor_updates_rejected = 0
        if args.imagine_enable and global_step >= args.imagine_warmup_steps:
            H_im = args.imagine_horizon
            B_im = args.imagine_batch_size
            imagine_ent_coef = log_alpha.exp().detach() if auto_alpha else args.ent_coef
            im_pg_losses, im_v_losses, im_entropies = [], [], []
            im_actor_gns, im_critic_gns = [], []
            im_return_sum = torch.zeros((), device=device)
            im_return_sq_sum = torch.zeros((), device=device)
            im_reward_sum = torch.zeros((), device=device)
            im_reward_sq_sum = torch.zeros((), device=device)
            im_continue_sum = torch.zeros((), device=device)
            im_edge_sum = torch.zeros((), device=device)
            # v129: seed imagination from a rolling bank of raw recent rollout
            # buffers, not just the current one. A seed at (rollout r, step j,
            # env e) is clean if [j-W+1 .. j] lies within one episode and j>=W-1.
            # dones[s]=1 marks an episode-start obs, so a reset between window
            # steps p-1,p shows as dones[p]=1; require dones==0 at j, j-1, ...,
            # j-(W-2). We re-encode sampled obs windows with the current WM.
            W_im = min(agent.world_model.context, args.num_steps)
            if imagine_prompt_bank:
                bank_obs = torch.stack([entry["obs"] for entry in imagine_prompt_bank], dim=0)
                bank_actions = torch.stack([entry["actions"] for entry in imagine_prompt_bank], dim=0)
                bank_dones = torch.stack([entry["dones"] for entry in imagine_prompt_bank], dim=0)
                bank_policy_forecasts = torch.stack(
                    [entry["policy_forecasts"] for entry in imagine_prompt_bank], dim=0
                )
            else:
                bank_obs = obs.unsqueeze(0)
                bank_actions = actions.unsqueeze(0)
                bank_dones = dones.unsqueeze(0)
                bank_policy_forecasts = incoming_policy_forecasts.unsqueeze(0)
            bank_rollouts = bank_obs.shape[0]
            step_arange = torch.arange(args.num_steps, device=device)
            clean_seed = (step_arange >= (W_im - 1)).view(1, args.num_steps, 1)
            clean_seed = clean_seed.expand(bank_rollouts, args.num_steps, args.num_envs).clone()
            for d in range(0, W_im - 1):
                step_lookup = (step_arange - d).clamp_min(0).view(1, args.num_steps, 1)
                step_lookup = step_lookup.expand(bank_rollouts, args.num_steps, args.num_envs)
                clean_seed &= bank_dones.gather(1, step_lookup) == 0
            clean_flat = clean_seed.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            bank_flat_size = bank_rollouts * args.num_steps * args.num_envs
            for _ in range(args.imagine_batches):
                anchor_beta_kl, anchor_base_kl = measure_real_anchor_beta_kls()
                exact_beta_kls.append(anchor_beta_kl.item())
                base_exact_beta_kls.append(anchor_base_kl.item())
                if real_actor_active and (
                    not torch.isfinite(anchor_beta_kl)
                    or anchor_beta_kl > args.target_kl
                    or not torch.isfinite(anchor_base_kl)
                    or anchor_base_kl > args.target_kl
                ):
                    real_actor_active = False
                with torch.no_grad():
                    # Start dreams from recent real rollout states, encoded by
                    # the freshly-updated WM to avoid stale latent caches.
                    if clean_flat.numel() > 0:
                        sel = torch.randint(0, clean_flat.numel(), (B_im,), device=device)
                        flat_idx = clean_flat[sel]
                    else:
                        flat_idx = torch.randint(0, bank_flat_size, (B_im,), device=device)
                    rollout_idx = flat_idx // (args.num_steps * args.num_envs)
                    rem_idx = flat_idx % (args.num_steps * args.num_envs)
                    step_idx = rem_idx // args.num_envs
                    env_idx = rem_idx % args.num_envs
                    dream_incoming_forecast = bank_policy_forecasts[
                        rollout_idx, step_idx, env_idx
                    ]
                    ag_inputs, im_zs, im_lps, im_rs, im_cs, im_vps = [], [], [], [], [], []
                    im_explore_code = torch.randn(
                        B_im,
                        args.explore_code_dim,
                        device=device,
                        generator=explore_generator,
                    )
                    # Recurrent imagination: carry a GROWING (latent, action) history and
                    # condition the predictor on it, exactly like the teacher-forced
                    # training path (predict_mtp_from_history, whose context grows up to
                    # self.context). v103 WARM-START: seed the history with the real
                    # preceding rollout window (W states + W-1 real actions) instead of a
                    # length-1 context, so the first dreamed step sees a full real context
                    # — the regime the predictor is strongest in. The temporal trunk truncates
                    # to self.context as the window then slides forward through the dream.
                    obs_start = agent.world_model.obs_token_start
                    if clean_flat.numel() > 0:
                        win_d = torch.arange(W_im - 1, -1, -1, device=device)         # [W-1,...,0]
                        win_steps = step_idx.unsqueeze(0) - win_d.unsqueeze(1)        # (W, B_im): l_{j-W+1}..l_j
                        win_rollouts = rollout_idx.unsqueeze(0).expand(W_im, -1)
                        win_envs = env_idx.unsqueeze(0).expand(W_im, -1)
                        win_summ = agent.world_model.encode_summary(
                            bank_obs[win_rollouts, win_steps, win_envs].reshape((-1,) + obs_shape),
                        ).detach().reshape(
                            W_im, B_im, agent.world_model.num_latent_tokens, agent.world_model.dim
                        )
                        # W real states l_{j-W+1}..l_j; W-1 real actions a_{j-W+1}..a_{j-1}
                        # (action[d] is taken FROM state[d], pairing as the predictor expects).
                        hist_latents = [win_summ[d][:, obs_start:] for d in range(W_im)]
                        win_actions = bank_actions[win_rollouts[:-1], win_steps[:-1], win_envs[:-1]]
                        hist_actions = [win_actions[d] for d in range(W_im - 1)]
                    else:
                        s = agent.world_model.encode_summary(
                            bank_obs[rollout_idx, step_idx, env_idx],
                        ).detach()
                        hist_latents = [s[:, obs_start:]]   # obs-only recurrent state l_t
                        hist_actions = []                   # no real history available
                    for t in range(H_im):
                        latent_hist = torch.stack(hist_latents, dim=1)               # (B, t+1, obs_tok, dim)
                        # Action-free temporal state: the outbound action is introduced
                        # only by imagine_step_from_history after the policy samples it.
                        belief = agent.world_model.belief_from_history(latent_hist)
                        agent_input = agent.agent_input_from_latent(
                            belief, dream_incoming_forecast
                        )
                        action, z, logprob, _, value_logits = (
                            agent.get_action_and_value_from_agent_input(
                                None, agent_input, explore_code=im_explore_code
                            )
                        )
                        hist_actions.append(action)
                        _, _, _, next_dream_forecast = (
                            agent.world_model.policy_forecast_from_history(
                                latent_hist, action
                            )
                        )
                        dream_incoming_forecast = next_dream_forecast[:, 0]
                        # Dynamics: same latent history, actions now include the taken a_t.
                        dyn_actions = torch.stack(hist_actions, dim=1)               # (B, t+1, act)
                        next_s, reward_logits, cont_logits = (
                            agent.world_model.imagine_step_from_history(latent_hist, dyn_actions)
                        )
                        ag_inputs.append(agent_input)
                        im_zs.append(z)
                        im_lps.append(logprob)
                        im_vps.append(torch.softmax(value_logits[:, 0], dim=-1))   # Z(s_t), horizon 0
                        im_rs.append(reward_logits_to_scalar(reward_logits))
                        im_cs.append(torch.sigmoid(cont_logits))
                        hist_latents.append(next_s[:, obs_start:].detach())
                    # Bootstrap value at the final imagined state s_H (same recurrent belief).
                    latent_hist = torch.stack(hist_latents, dim=1)
                    belief_H = agent.world_model.belief_from_history(latent_hist)
                    value_probs_H = torch.softmax(
                        agent.get_value_from_agent_input(
                            None,
                            agent.agent_input_from_latent(
                                belief_H, dream_incoming_forecast
                            ),
                        )[:, 0],  # horizon 0 = V(s_H)
                        dim=-1,
                    )
                    value_probs_t = torch.stack(im_vps, dim=0)                  # (H,B,n) Z(s_0..s_{H-1})
                    next_value_probs = torch.stack(im_vps[1:] + [value_probs_H], dim=0)  # (H,B,n) Z(s_1..s_H)
                    im_rewards = torch.stack(im_rs, dim=0)                      # (H,B)
                    im_continues = torch.stack(im_cs, dim=0)                    # (H,B)
                    values_t = value_probs_to_scalar(value_probs_t)
                    next_values = value_probs_to_scalar(next_value_probs)
                    # Scalar GAE over the dream (soft continuation as the discount mask).
                    adv = torch.zeros(H_im, B_im, device=device)
                    lastgae = torch.zeros(B_im, device=device)
                    for t in reversed(range(H_im)):
                        nonterminal = im_continues[t]
                        delta = im_rewards[t] + args.gamma * nonterminal * next_values[t] - values_t[t]
                        lastgae = delta + args.gamma * args.gae_lambda * nonterminal * lastgae
                        adv[t] = lastgae
                    returns_t = adv + values_t
                    # Dreamer4-style scalar HL-Gauss returns with MTP over the dream
                    # horizon: horizon h regresses returns_t[t+h]. Soft continuation is
                    # already folded into returns_t via the GAE discount, so horizons
                    # are masked only where they run off the rollout tail (t+h >= H_im).
                    mtp = args.critic_mtp_horizon
                    return_mtp_im = returns_t.new_zeros((H_im, B_im, mtp))
                    return_mtp_mask_im = torch.zeros(
                        (H_im, B_im, mtp), dtype=torch.bool, device=device
                    )
                    for h in range(mtp):
                        valid_len = H_im - h
                        if valid_len <= 0:
                            break
                        return_mtp_im[:valid_len, :, h] = returns_t[h : h + valid_len]
                        return_mtp_mask_im[:valid_len, :, h] = True
                    target_probs_im = hl_support.project(return_mtp_im)  # (H,B,mtp,num_bins)
                    sigma_im = (
                        value_probs_t * (scalar_support - values_t.unsqueeze(-1)) ** 2
                    ).sum(-1).clamp_min(0).sqrt().clamp_min(args.sigma_floor_bins * scalar_bin_width)
                    returns_coord_im = symlog(returns_t) if args.value_symlog else returns_t
                    cdf_frac_im = ((returns_coord_im.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)
                    u_im = (value_probs_t * cdf_frac_im).sum(-1)
                    f_ag = torch.stack(ag_inputs, dim=0).reshape(H_im * B_im, -1)
                    f_z = torch.stack(im_zs, dim=0).reshape(H_im * B_im, -1)
                    f_lp = torch.stack(im_lps, dim=0).reshape(-1)
                    f_explore_code = im_explore_code.unsqueeze(0).expand(
                        H_im, -1, -1
                    ).reshape(H_im * B_im, args.explore_code_dim)
                    f_adv = adv.reshape(-1)
                    f_sigma = sigma_im.reshape(-1)
                    f_u = u_im.reshape(-1)
                    f_target = target_probs_im.reshape(-1, args.critic_mtp_horizon, args.num_bins)
                    f_target_mask = return_mtp_mask_im.reshape(-1, args.critic_mtp_horizon)
                    im_return_sum += returns_t.mean()
                    im_return_sq_sum += returns_t.square().mean()
                    im_reward_sum += im_rewards.mean()
                    im_reward_sq_sum += im_rewards.square().mean()
                    im_continue_sum += im_continues.mean()
                    im_edge_per_h = target_probs_im[..., 0] + target_probs_im[..., -1]
                    im_edge_mask_f = return_mtp_mask_im.to(im_edge_per_h.dtype)
                    im_edge_sum += (im_edge_per_h * im_edge_mask_f).sum() / im_edge_mask_f.sum().clamp_min(1)

                # Single 1-epoch gradient step on the imagined block (ratio starts at 1).
                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value_from_agent_input(
                    None, f_ag, f_z, f_explore_code
                )
                logratio = newlogprob - f_lp
                ratio = logratio.exp()
                shaped = shape_advantage(f_adv, f_sigma, f_u, args, device)
                if args.norm_adv:
                    shaped = (shaped - shaped.mean()) / (shaped.std() + 1e-8)
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -shaped * ratio
                pg_loss2 = -shaped * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss_im = torch.max(pg_loss1, pg_loss2).mean()
                value_log_probs_im = torch.log_softmax(value_logits, dim=-1)
                value_ce_im = -(f_target * value_log_probs_im).sum(dim=-1)
                v_loss_im = (value_ce_im * f_target_mask.to(value_ce_im.dtype)).sum(dim=-1).mean()
                entropy_loss_im = entropy.mean()
                attempted_dream_actor_update = real_actor_active
                dream_actor_snapshot = (
                    snapshot_actor_optimizer_state()
                    if attempted_dream_actor_update
                    else None
                )
                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss_im).backward(retain_graph=True)
                    audit_wm_gradient_isolation()
                    if not real_actor_active:
                        for p in shared_trunk_params:
                            p.grad = None
                    im_critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    if attempted_dream_actor_update:
                        (pg_loss_im - imagine_ent_coef * entropy_loss_im).backward()
                        audit_wm_gradient_isolation()
                        im_actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    else:
                        im_actor_gn = torch.zeros((), device=device)
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                    if attempted_dream_actor_update:
                        post_dream_anchor_kl, post_dream_base_kl = (
                            measure_real_anchor_beta_kls()
                        )
                        exact_beta_kls.append(post_dream_anchor_kl.item())
                        base_exact_beta_kls.append(post_dream_base_kl.item())
                        if (
                            not torch.isfinite(post_dream_anchor_kl)
                            or post_dream_anchor_kl > args.target_kl
                            or not torch.isfinite(post_dream_base_kl)
                            or post_dream_base_kl > args.target_kl
                        ):
                            restore_actor_optimizer_state(dream_actor_snapshot)
                            real_actor_active = False
                            dream_actor_updates_rejected += 1
                        else:
                            dream_actor_updates_used += 1
                else:
                    loss_im = pg_loss_im - imagine_ent_coef * entropy_loss_im + v_loss_im * args.vf_coef
                    optimizer.zero_grad(set_to_none=True)  # v103: None (not 0) so AdamW wd skips ungraded WM params
                    loss_im.backward()
                    audit_wm_gradient_isolation()
                    im_critic_gn = im_actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                im_pg_losses.append(pg_loss_im.item())
                im_v_losses.append(v_loss_im.item())
                im_entropies.append(entropy_loss_im.item())
                im_actor_gns.append(float(im_actor_gn))
                im_critic_gns.append(float(im_critic_gn))
            final_anchor_beta_kl, final_anchor_base_kl = (
                measure_real_anchor_beta_kls()
            )
            exact_beta_kls.append(final_anchor_beta_kl.item())
            base_exact_beta_kls.append(final_anchor_base_kl.item())
            if (
                not torch.isfinite(final_anchor_beta_kl)
                or final_anchor_beta_kl > args.target_kl
                or not torch.isfinite(final_anchor_base_kl)
                or final_anchor_base_kl > args.target_kl
            ):
                real_actor_active = False
            n_im = max(1, args.imagine_batches)
            im_return_mean = im_return_sum / n_im
            im_reward_mean = im_reward_sum / n_im
            im_return_std = (im_return_sq_sum / n_im - im_return_mean.square()).clamp_min(0).sqrt()
            im_reward_std = (im_reward_sq_sum / n_im - im_reward_mean.square()).clamp_min(0).sqrt()
            imagine_metrics = {
                "imagine/policy_loss": float(np.mean(im_pg_losses)),
                "imagine/value_loss": float(np.mean(im_v_losses)),
                "imagine/entropy": float(np.mean(im_entropies)),
                "imagine/return_mean": im_return_mean.item(),
                "imagine/return_std": im_return_std.item(),
                "imagine/reward_mean": im_reward_mean.item(),
                "imagine/reward_std": im_reward_std.item(),
                "imagine/continue_mean": (im_continue_sum / n_im).item(),
                "imagine/target_edge_mass": (im_edge_sum / n_im).item(),
                "imagine/prompt_bank_rollouts": float(bank_rollouts),
                "imagine/prompt_clean_seeds": float(clean_flat.numel()),
                "imagine/actor_grad_norm": float(np.mean(im_actor_gns)),
                "imagine/critic_grad_norm": float(np.mean(im_critic_gns)),
            }

        # ================= PRIVATE FORECAST-HEAD REFRESH =================
        private_refresh_kls = []
        private_refresh_grad_norms = []
        private_refresh_isolated = 1.0
        refresh_memory_flat, _ = agent.split_agent_input(b_agent_inputs)
        refresh_memory = refresh_memory_flat.reshape(
            args.batch_size,
            agent.world_model.num_obs_tokens,
            agent.world_model.dim,
        ).detach()
        refresh_actions = b_actions.detach()
        refresh_next_inputs = fresh_next_agent_inputs.reshape(
            -1, agent.agent_input_dim
        )
        refresh_weight = (
            transition_valids * (1.0 - transition_terminations)
        ).reshape(-1)
        refresh_mb = args.private_forecast_refresh_minibatch_size
        for refresh_start in range(0, args.batch_size, refresh_mb):
            refresh_end = min(args.batch_size, refresh_start + refresh_mb)
            sl = slice(refresh_start, refresh_end)
            weight = refresh_weight[sl]
            if weight.sum().item() <= 0:
                continue
            with torch.no_grad():
                target_alpha, target_beta = agent.get_base_beta_params_from_agent_input(
                    refresh_next_inputs[sl]
                )
            _, pred_alpha, pred_beta, _ = (
                agent.world_model.policy_forecast_from_state_memory(
                    refresh_memory[sl, None], refresh_actions[sl, None]
                )
            )
            refresh_kl = torch.distributions.kl_divergence(
                Beta(target_alpha, target_beta),
                Beta(pred_alpha[:, 0], pred_beta[:, 0]),
            ).sum(dim=-1)
            refresh_loss = (refresh_kl * weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            refresh_loss.backward()
            if any(p.grad is not None for p in base_wm_params):
                private_refresh_isolated = 0.0
                raise RuntimeError("private forecast refresh leaked into shared WM")
            if any(p.grad is not None for p in non_wm_params):
                private_refresh_isolated = 0.0
                raise RuntimeError("private forecast refresh leaked into agent")
            refresh_gn = nn.utils.clip_grad_norm_(
                policy_forecast_params, args.lewm_grad_clip
            )
            optimizer.step()
            private_refresh_kls.append(refresh_loss.item())
            private_refresh_grad_norms.append(float(refresh_gn))

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]   # (N, mtp)
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        with torch.no_grad():
            diag_n = min(1024, b_agent_inputs.shape[0])
            diag_memory, diag_forecast = agent.split_agent_input(
                b_agent_inputs[:diag_n]
            )
            actor_base_input = agent.actor_readout(diag_memory)
            critic_base_input = agent.critic_readout(diag_memory)
            actor_forecast_delta = agent.actor_forecast_residual(diag_forecast)
            critic_forecast_delta = agent.critic_forecast_residual(diag_forecast)

            def residual_ratio(delta, base):
                return delta.square().mean().sqrt() / base.square().mean().sqrt().clamp_min(1e-8)

            actor_forecast_ratio = residual_ratio(
                actor_forecast_delta, actor_base_input
            )
            critic_forecast_ratio = residual_ratio(
                critic_forecast_delta, critic_base_input
            )
            actor_feat_diag, _ = agent._trunks_from_agent_input(
                None, b_agent_inputs[:diag_n]
            )
            dist_code, to_action_diag, _ = agent._actor_dist(
                actor_feat_diag, b_explore_codes[:diag_n]
            )
            dist_zero, _, _ = agent._actor_dist(actor_feat_diag, None)
            explore_mean_shift = (
                to_action_diag(dist_code.mean) - to_action_diag(dist_zero.mean)
            ).abs()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("explore/code_norm", b_explore_codes.norm(dim=-1).mean().item(), global_step)
        writer.add_scalar("explore/action_mean_shift", explore_mean_shift.mean().item(), global_step)
        writer.add_scalar("explore/action_mean_shift_max", explore_mean_shift.max().item(), global_step)
        writer.add_scalar(
            "policy_forecast/actor_adapter_rms_ratio",
            actor_forecast_ratio.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/critic_adapter_rms_ratio",
            critic_forecast_ratio.item(),
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/actor_adapter_weight_rms",
            agent.actor_forecast_residual.weight.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/critic_adapter_weight_rms",
            agent.critic_forecast_residual.weight.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/adapters_enabled",
            float(agent.forecast_adapters_enabled),
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/ppo_wm_grad_isolated",
            ppo_wm_grad_isolated,
            global_step,
        )
        writer.add_scalar(
            "fresh_interface/pre_repair_behavior_kl",
            fresh_pre_repair_behavior_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "fresh_interface/pre_repair_base_kl",
            fresh_pre_repair_base_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "fresh_interface/entry_behavior_kl",
            fresh_entry_behavior_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "fresh_interface/entry_base_kl",
            fresh_entry_base_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "fresh_interface/invalid_fraction", fresh_invalid_fraction, global_step
        )
        writer.add_scalar(
            "joint_imagine/loss_weight", 0.5, global_step
        )
        dream_pg_abs = float(np.mean(np.abs(joint_dream_pg_losses))) if joint_dream_pg_losses else 0.0
        real_pg_abs = float(np.mean(np.abs(joint_real_pg_losses))) if joint_real_pg_losses else 0.0
        dream_value_abs = float(np.mean(np.abs(joint_dream_value_losses))) if joint_dream_value_losses else 0.0
        real_value_abs = float(np.mean(np.abs(joint_real_value_losses))) if joint_real_value_losses else 0.0
        writer.add_scalar(
            "joint_imagine/policy_abs_contribution_fraction",
            dream_pg_abs / max(1e-12, dream_pg_abs + real_pg_abs),
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/value_abs_contribution_fraction",
            dream_value_abs / max(1e-12, dream_value_abs + real_value_abs),
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/policy_loss",
            float(np.mean(joint_dream_pg_losses)) if joint_dream_pg_losses else 0.0,
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/value_loss",
            float(np.mean(joint_dream_value_losses)) if joint_dream_value_losses else 0.0,
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/minibatch_behavior_kl",
            float(np.mean(joint_dream_minibatch_kls)) if joint_dream_minibatch_kls else 0.0,
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/full_behavior_kl",
            post_joint_dream_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/entry_behavior_kl",
            dream_entry_behavior_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/actor_updates_used",
            actor_updates_used,
            global_step,
        )
        writer.add_scalar(
            "joint_imagine/actor_updates_rejected",
            actor_updates_rejected,
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/private_refresh_kl",
            float(np.mean(private_refresh_kls)) if private_refresh_kls else 0.0,
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/private_refresh_grad_norm",
            float(np.mean(private_refresh_grad_norms)) if private_refresh_grad_norms else 0.0,
            global_step,
        )
        writer.add_scalar(
            "policy_forecast/private_refresh_isolated",
            private_refresh_isolated,
            global_step,
        )
        writer.add_scalar("losses/value_loss", joint_v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", joint_pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", joint_entropy_loss.item(), global_step)
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
        writer.add_scalar("losses/exact_beta_kl", exact_beta_kls[-1], global_step)
        writer.add_scalar("losses/base_exact_beta_kl", base_exact_beta_kls[-1], global_step)
        writer.add_scalar(
            "losses/attempted_exact_beta_kl_max", max(exact_beta_kls), global_step
        )
        writer.add_scalar(
            "losses/attempted_base_exact_beta_kl_max",
            max(base_exact_beta_kls),
            global_step,
        )
        writer.add_scalar("losses/actor_updates_used", actor_updates_used, global_step)
        writer.add_scalar("losses/actor_updates_rejected", actor_updates_rejected, global_step)
        writer.add_scalar(
            "losses/trust_region_exhausted", float(not real_actor_active), global_step
        )
        writer.add_scalar(
            "losses/actor_update_fraction",
            actor_updates_used / float(args.update_epochs * args.num_minibatches),
            global_step,
        )
        writer.add_scalar(
            "losses/total_actor_update_fraction",
            actor_updates_used / float(args.update_epochs * args.num_minibatches),
            global_step,
        )
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "losses/actor_grad_norm",
            float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            global_step,
        )
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        if wm_losses:
            log_policy_forecast_diagnostics()
            writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
            writer.add_scalar("lewm/obs_latent_mse", float(np.mean(wm_latent_losses)), global_step)
            writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
            writer.add_scalar("lewm/continuation_bce", float(np.mean(wm_termination_losses)), global_step)
            writer.add_scalar("lewm/obs_sigreg", float(np.mean(wm_sigreg_losses)), global_step)
            writer.add_scalar("lewm/grad_norm", float(np.mean(wm_grad_norms)), global_step)
            writer.add_scalar(
                "lewm/outcome_io_grad_norm",
                float(np.mean(wm_outcome_io_grad_norms)),
                global_step,
            )
            total_probe_weight = wm_reward_probe_weights.sum()
            if total_probe_weight.item() > 0:
                writer.add_scalar(
                    "lewm/reward_scalar_mae",
                    (wm_reward_abs_error_sums.sum() / total_probe_weight).item(),
                    global_step,
                )
                writer.add_scalar(
                    "lewm/reward_scalar_bias",
                    (wm_reward_error_sums.sum() / total_probe_weight).item(),
                    global_step,
                )
                writer.add_scalar(
                    "lewm/reward_pred_edge_mass",
                    (wm_reward_pred_edge_sums.sum() / total_probe_weight).item(),
                    global_step,
                )
                writer.add_scalar(
                    "lewm/reward_target_edge_frac",
                    (wm_reward_target_edge_sums.sum() / total_probe_weight).item(),
                    global_step,
                )
                for mtp_idx in range(args.lewm_mtp_len):
                    offset_weight = wm_reward_probe_weights[mtp_idx]
                    if offset_weight.item() > 0:
                        writer.add_scalar(
                            f"lewm/reward_scalar_mae_mtp{mtp_idx + 1}",
                            (wm_reward_abs_error_sums[mtp_idx] / offset_weight).item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"lewm/reward_scalar_bias_mtp{mtp_idx + 1}",
                            (wm_reward_error_sums[mtp_idx] / offset_weight).item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"lewm/reward_pred_edge_mass_mtp{mtp_idx + 1}",
                            (wm_reward_pred_edge_sums[mtp_idx] / offset_weight).item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"lewm/reward_target_edge_frac_mtp{mtp_idx + 1}",
                            (wm_reward_target_edge_sums[mtp_idx] / offset_weight).item(),
                            global_step,
                        )
            if wm_length1_ar_weight.sum().item() > 0:
                log_labeled_rollout_diagnostic(
                    writer, "diagnostics/length1_ar", global_step,
                    wm_length1_ar_latent_mse_sum, wm_length1_ar_reward_abs_error_sum,
                    wm_length1_ar_reward_error_sum, wm_length1_ar_continuation_bce_sum,
                    wm_length1_ar_weight,
                )
            if wm_warm_ar_weight.sum().item() > 0:
                log_labeled_rollout_diagnostic(
                    writer, "diagnostics/warm_ar", global_step,
                    wm_warm_ar_latent_mse_sum, wm_warm_ar_reward_abs_error_sum,
                    wm_warm_ar_reward_error_sum, wm_warm_ar_continuation_bce_sum,
                    wm_warm_ar_weight,
                )
        if imagine_metrics is not None:
            for k, v in imagine_metrics.items():
                writer.add_scalar(k, v, global_step)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
