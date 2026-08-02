# PPO + IterThink v192 flowwarmstart-cleanctx (state-dependent policy input from step 0; deterministic policy/bootstrap encodes).
#
# v192 = v191 with two DEFECT REMOVALS (not new hypotheses). v191's return curve is v186
#   right-shifted ~400k: the representation fix worked (flow_mean_mse 0.90 -> 0.39 by 622k,
#   falling) but two mechanical defects delay policy learning from step 0.
#   WHAT CHANGED vs v191.
#     - FLOW-HEAD WARM START (kill the dead-input phase). The head was full AdaLN-zero
#       (mod/final_mod/out_proj all zero), so the mean-path policy input f(z=0 | belief)
#       was EXACTLY 0 -- a state-INDEPENDENT constant -- until the head trained up: the
#       actor/critic's predicted-next-latent input channel was guaranteed dead early. Now
#       (mirroring the v186 trunk warm start): block gate biases = lewm_flow_gate_init
#       (0.1); block SHIFT-slice WEIGHTS xavier-init (NOT bias-only -- at z=0 the residual
#       stream is exactly 0, so the belief can only enter through the shift path; a
#       bias-only gate warm start provably leaves x_pred == 0, verified numerically);
#       out_proj xavier * lewm_flow_outproj_gain (0.1). scale slices + final_mod stay
#       zero. Init x_pred: deterministic, state-DEPENDENT, std ~0.10.
#     - CLEAN-CONTEXT POLICY/BOOTSTRAP ENCODES. The 10% ctx-noise stays for WM TRAINING
#       and DREAM DYNAMICS (the regime the trunk trains in), but every belief the
#       actor/critic CONSUME is now deterministic (mix_ctx=False threaded through
#       belief_from_history): rollout agent input + next-transition value bootstrap
#       (build_belief_agent_input), the single-obs convenience path, and the dream
#       agent-input / dream-bootstrap beliefs. Before, duplicate encodes of the same
#       state (values[t+1] vs next_transition_values[t]) disagreed by fresh zero-mean
#       ctx-noise jitter, adding variance to every GAE delta. Dream dynamics
#       (imagine_step_from_history) and the warm-AR / flow_mean_mse diagnostics keep the
#       noisy context (metric continuity; the WM is exercised as trained).
#   RISK. Warm-started gates change early flow-loss dynamics slightly: grounding now sees
#   a small random state-dependent function instead of exactly 0 (out_proj gain 0.1 keeps
#   it small). Clean-context policy encodes create a mild train/act ctx-noise mismatch
#   for the trunk (trains noisy, acts clean; the reference also acts noisy -- deliberate
#   deviation justified by MuJoCo determinism and GAE variance).
#
# -----------------------------------------------------------------------------
# v191 (inherited) pertoken-sigreg (drop cross-token SIGReg decorrelation pressure).
#
# v191 = v190 with ONE functional change (v190's in-path BN projector and temp-0 dreams
#   stay -- they are reference-faithful; v190 eliminated the projector as the culprit).
#   WHAT CHANGED vs v190. Obs-token SIGReg goes JOINT -> PER-TOKEN: instead of one
#   projection over the flattened (17*64=1088)-dim obs frame, each obs token's 64-dim
#   marginal is pushed toward N(0,I) with the token axis FOLDED INTO THE SAMPLE/BATCH
#   axis (each of the 17 obs tokens is a separate 64-dim sample), exactly how the 2
#   outcome tokens are already handled via sigreg_A_out. This matches the trading-bot
#   reference: one token per frame there, so its per-token SIGReg == per-frame and no
#   cross-token constraint exists. sigreg_A_obs is now a D-dim projection (separate draw
#   from A_out). lewm_sigreg_coef=0.09 and lewm_sigreg_num_proj=4096 are UNCHANGED (one
#   variable at a time; 4096 is more-than-adequate density for 64-d).
#   WHY. Joint isotropy over the 1088-dim frame demands MUTUAL DECORRELATION of 17 tokens
#   that all encode the same 17-dim obs -- only satisfiable by a space-filling/hashing
#   encoder. Measured: effrank_obs_joint ~14 with SIGReg off-token (v186) vs ~197-200
#   with joint on-token SIGReg (v187/v188/v190), with conditional-mean predictability
#   pinned at chance (flow_mean_mse ~0.93-0.98 ~= token variance) in all the latter.
#   v190 reproduced the exact signature with the projector restored, isolating the joint
#   SIGReg as the cause. Per-token Gaussianity keeps the anti-collapse pressure but
#   leaves cross-token correlation free, so the encoder can stay smooth and predictable.
#   RISK. Per-token marginals from a 17-dim obs still overparameterize 64 dims, so a soft
#   space-filling pressure remains within each token. effrank_obs_joint is now free to
#   drop -- watch it and lewm/flow_mean_mse (the go/no-go signal for re-raising dream
#   temperature).
#
# -----------------------------------------------------------------------------
# v190 (inherited) projtoken-meandream (restore reference in-path BN projector; deterministic dreams).
#
# v190 = v188 with two surgical changes targeting why the v187/v188 flow never learned its
#   conditional mean (lewm/flow_mean_mse pinned at ~0.93 ~= token variance).
#   WHAT CHANGED vs v188.
#     - IN-PATH PROJECTOR RESTORED. The reference encoder (trading_bot v3-plainflow) is
#       proj -> enrich -> PROJECTOR MLP (fc -> BatchNorm1d -> GELU -> fc) -> latent_bound,
#       and the POST-projector bounded outputs ARE the tokens (SIGReg targets, flow
#       targets, trunk inputs). v187 deleted the projector entirely while moving SIGReg
#       onto the raw encoder tokens -- that is the bug being fixed. encode_summary now
#       applies a per-token Linear(D->lewm_projector_hidden=512, 8x dim like the
#       reference's 2048=8x256) -> BN -> GELU -> Linear(->D) right before the tanh bound;
#       batch and token axes are folded for BN (v186 JEPAProjector convention,
#       track_running_stats=False). NOTE the difference vs v183/v186: there the projector
#       was a separate LOSS-SPACE head; here it sits IN the token path, so its output
#       feeds everything (targets, trunk, SIGReg, flow, policy input).
#     - DREAM TEMPERATURE 0: lewm_flow_dream_temperature 1.0->0.0 -- dreams use the
#       deterministic one-shot conditional mean (v186-like) instead of temp-1 8-step
#       sampled Euler paths, until lewm/flow_mean_mse proves the flow out.
#   WHY. Diagnosis of the v187/v188 failure: joint-frame SIGReg on the RAW tokens made the
#   raw encoder map do ALL of the Gaussianization -- a space-filling/hashing solution
#   (effrank_obs_joint exploded 14 -> ~200) whose tokens carry no predictable structure,
#   so the flow's conditional mean never beat the prior (flow_mean_mse ~= token variance)
#   and temp-1 dreams (3:1 vs real data) fed pure noise to policy/value learning. The BN
#   projector absorbs the per-dim scaling/whitening cheaply, so the upstream encoder can
#   stay smooth and predictable while SIGReg still gets its isotropic target. Joint obs
#   SIGReg + num_proj=4096 are UNCHANGED (the reference has one token per frame, so its
#   per-token SIGReg is per-frame; the joint obs-token SIGReg is the legitimate analog).
#   RISK. BatchNorm in the token path makes every encoding depend on its batch's stats;
#   the thinnest call is the per-env-step rollout encode (16 envs x 19 tokens = 304 BN
#   rows -- adequate, but stats come from 16 correlated states; agent inputs are buffered
#   once, so PPO ratio replay is unaffected). Temp-0 dreams reintroduce v186's
#   deterministic-mean compounding; revisit temperature once flow_mean_mse separates from
#   token variance.
#
# -----------------------------------------------------------------------------
# v188 (inherited) plainflow-dit-k6 (ablate shortcut -> PLAIN conditional flow matching).
#
# v188 = v187 with the shortcut machinery REMOVED, mirroring trading_bot_0's v3-plainflow
#   ablation (commit 0a9ef9e2), which significantly outperformed the shortcut version there.
#   WHAT CHANGED vs v187.
#     - NO step-size-log2 embedding: the flow head is conditioned on belief + signal level
#       only (cond_in = D + signal_embed).
#     - NO self-consistency distillation, no dyadic step sampling, no signal snapping:
#       EVERY (row, timestep) trains plain conditional flow matching -- signal ~ U{0..63}
#       unsnapped, z_t = (1-t)*eps + t*x, x-space grounding MSE at full uniform weight.
#       (lewm_flow_consistency_coef / lewm_flow_step_embed_dim args are gone.)
#     - SAMPLING: temp>0 paths are an 8-STEP PLAIN EULER integration over the equally
#       spaced signal grid (d = 64/8 = 8; signals 0,8,...,56); the temp-0 one-shot
#       conditional-mean path (z=0, signal=0) is unchanged, as is the mean-path anchor
#       (lewm_flow_meanpath_prob rows trained at exactly that query).
#     - Everything else inherited from v187: 3-block AdaLN-zero x-pred DiT, diffusion
#       forcing, 10% noisy context, p=0.25 self-conditioning, tanh latent bound B=3,
#       SIGReg on clean bounded tokens, temp-1 sampled dreams, mean-path policy input.
#   WHY. In the trading-bot reference the shortcut objective's extra losses bought only
#   cheaper sampling (4 steps vs 8), and hurt prediction quality: the self-consistency
#   term competes with grounding for head capacity and trains 5/6 of rows at snapped
#   (coarser) signal levels. Plain flow matching spends the whole model on the actual
#   denoising task; 8 Euler steps on a tiny per-token MLP head are still cheap for dreams.
#   RISK. Dream generation costs 8 head passes per step instead of 4 (the head is a
#   3-block D=64 MLP -- negligible vs the trunk). lewm/flow_consistency metric is gone.
#
# -----------------------------------------------------------------------------
# v187 (inherited) flowmatch-shortcut-dit-k6 (stochastic next-latent via shortcut flow matching).
#
# v187 = v186 with the deterministic next-latent predictor (pred_next_proj + JEPA-projector
#   MSE) replaced by a SHORTCUT FLOW-MATCHING head (dreamer4 shortcut forcing; ported from
#   trading_bot_0's flow-diffusion pretraining, which is itself dreamer4-informed).
#   WHAT CHANGED.
#     - FLOW HEAD. A 3-block AdaLN-zero DiT (MLP-only, per-token) x-predicts the CLEAN next
#       summary token from a noised input z_t = (1-t)*eps + t*x, conditioned on the belief
#       token + a 64-level signal embedding (t = k/64) + a log2 step-size embedding (d in
#       {1..32}). mod/final_mod/out_proj are zero-init: at init x_pred == 0 (identity-mean).
#     - LOSS (replaces the v183 JEPA-projector MSE). Per (row, timestep): step_log2 ~ U{0..5}
#       (marginal identical to the trading-bot batch-Bernoulli(5/6) shortcut branch), signal
#       ~ U{0..63} snapped down to a multiple of d=2^step_log2. Always-on x-GROUNDING at
#       FULL UNIFORM weight across signal levels: MSE(x_pred, x_clean) with ATTACHED targets
#       (gradient reaches the encoder through target and noised input -- le-wm/JEPA style).
#       Plus SELF-CONSISTENCY distillation on d>=2 rows: two half-steps (step_log2-1) with
#       the SAME online net under no_grad give v_target = (v1+v2)/2; the one-full-step
#       student velocity matches it under an MSE weighted (1-t)^2. No EMA teacher anywhere.
#     - DIFFUSION FORCING. Noise level is independent per (row, timestep), shared across the
#       19 tokens of a frame (dreamer4 per-frame levels).
#     - 10% NOISY CONTEXT. Every predictor-trunk input token is mixed x.lerp(eps, 0.1) at
#       train AND inference (rollout/dream), so the belief never assumes a clean context.
#     - SELF-CONDITIONING p=0.25. A no_grad clean-context pre-pass produces one-shot preds;
#       each teacher-forced context token (except the anchor) is REPLACED by the model's own
#       shifted prediction w.p. 0.25 before the training pass (scheduled sampling).
#     - tanh LATENT BOUND. encode_summary output and the flow head's x_pred both pass
#       B*tanh(x/B), B=3: targets and predictions live in (-3,3), matching the N(0,I) prior
#       scale (SIGReg pushes tokens toward isotropy; the JEPA projector/BatchNorm is GONE --
#       SIGReg now acts directly on the clean bounded encoder tokens, trading-bot recipe).
#     - CONSUMERS. Policy input (combine_agent_input) uses the TEMP-0 CONDITIONAL MEAN:
#       one-shot x_pred at (z=0, signal=0, d=1) -- deterministic, one MLP pass, buffered and
#       detached exactly like v182's predicted embed. Dream recurrence samples a 4-step
#       shortcut Euler path (d=16, signals 0/16/32/48, z0 ~ N(0,I)*tau, tau=1): dreams see
#       the model's actual generative distribution. Labeled AR diagnostics use the mean path
#       (comparable to v186); a per-iteration diagnostic logs mean-path vs K=16-sampled-path
#       next-latent MSE (lewm/flow_mean_mse vs lewm/flow_sampled_mse).
#     - MTP offsets 1..3 keep their deterministic linear heads (they only feed the detached
#       outcome probes); the offset-0 probe reads the detached one-shot mean x_pred.
#   WHY. The deterministic predictor regresses to the conditional mean of a stochastic
#   transition kernel, blurring multi-modal futures; dreams then compound mean-latents. A
#   flow head models the full conditional distribution, the shortcut objective keeps
#   sampling to 4 Euler steps (cheap enough for the dream loop), and the always-on grounding
#   preserves a well-trained one-shot mean for the (deterministic) policy-input path.
#   RISK. The flow x-MSE in raw token space replaces the BatchNorm'd projector-emb MSE, so
#   lewm/obs_latent_* scales are not comparable to v186. Zero-init out_proj means the policy
#   input is exactly 0 for the first iteration's rollout (predictor warm-start gates keep the
#   BELIEF alive; the flow head wakes up within the first WM update). Sampled dreams inject
#   fresh stochasticity into imagination -- watch imagine/* stability and early KL.
#   REVIEW NOTES (v187.1, from the adversarial + fidelity reviews):
#     - MEAN-PATH ANCHOR (fix applied): training only ever fed z_t = eps at signal 0, so the
#       exact policy-input query f(z=0, s=0, d=1) was off-support. lewm_flow_meanpath_prob
#       (0.1) of rows are now forced to that query with eps=0 -- a direct conditional-mean
#       regression through the same head.
#     - FACTORIZED FLOW: the head denoises each of the 19 tokens independently given its
#       belief token (z never enters the trunk), so sampled frames have per-token-independent
#       noise -- marginal spread, not jointly-consistent futures. On near-deterministic MuJoCo
#       this mostly injects encoder/ctx-noise slop into dreams (which outweigh real data
#       3:1). Gauge: lewm/flow_sampled_mse - lewm/flow_mean_mse; ablation lever:
#       --lewm-flow-dream-temperature 0 (v186-like deterministic dreams).
#     - PROBE MISMATCH: outcome probes train on the detached one-shot MEAN but are consumed
#       on SAMPLED outcome tokens in dreams; adds dream reward/continuation variance.
#     - CTX-NOISE JITTER: fresh 10% context noise per trunk call means duplicate encodings
#       of the same state (values[t+1] vs next_transition_values[t]) disagree by zero-mean
#       jitter -- a new critic-target variance source (PPO ratio replay is unaffected:
#       agent inputs are buffered once).
#
# -----------------------------------------------------------------------------
# v186 (inherited) doublenorm-predwarmstart-k6 (fix B for v184's regression vs nGPT).
#
# v186 = v184 (std pre-LN transformer WM) with two ARCHITECTURE-FIDELITY fixes that target
#   the two mechanisms by which the de-nGPT conversion under-performed the nGPT v182. Both
#   were found by auditing v184's WM against ../le-wm directly.
#   (B1) DOUBLE-NORM. le-wm runs TWO LayerNorms per sublayer: an OUTER affine-free pre-norm
#     (in the block) and an INNER affine norm (inside Attention / FeedForward). v184 kept
#     only one affine pre-norm per sublayer, dropping the inner affine gain/bias. v186
#     restores le-wm's double-norm: encoder + predictor outer norms are affine-free
#     (eps 1e-6); inner affine LayerNorms now live in every attention (StdSelfAttention /
#     _RoPEAttention / _SpaceAttention `.norm`) and in each MLP (`mlp_norm`, le-wm's
#     FeedForward[0]). For the predictor the modulate(shift,scale) sits between the outer
#     and inner norm, exactly as le-wm's ConditionalBlock -> Attention/FeedForward composes.
#   (B2) PREDICTOR WARM-START. v184's AdaLN-zero gates were EXACTLY 0 at init, so the
#     predictor's temporal attention/MLP contributed NOTHING until the gates trained up -- a
#     cold start nGPT v182 never had (its eigen-lr eta ~= 0.05 kept attention active from
#     step 0). v186 keeps the adaLN WEIGHT zero-init (no action signal at init) but warm-
#     starts the GATE bias slices to a small positive constant (lewm_predictor_gate_init=0.1)
#     so each predictor sublayer is active (~0.1x) from the first step, while shift/scale
#     stay 0 (identity modulation) and action influence still ramps in via the zero weight.
#   WHY. v184 regressed to ~1413 @ 3.4M vs nGPT v182 ~3113. Two deviations were load-bearing:
#     a missing inner affine norm (less per-dim conditioning capacity) and a dead predictor
#     at init (no temporal mixing until gates wake up, starving the agent's belief signal
#     early). v186 isolates fix B; v185 isolates fix A (projector emb as the working rep).
#   RISK. The inner norms add affine params under wd=1e-3; warm-started gates make the
#   predictor active immediately, so re-check early KL / wm-loss stability at ~1M.
#
# -----------------------------------------------------------------------------
# v184 = v183 with the REST of the nGPT machinery removed from the world model. v183 had
#   already moved the JEPA objective off the unit sphere (BatchNorm'd projector emb) and
#   re-enabled weight decay; v184 finishes the job -- the WM is now a STANDARD pre-LayerNorm
#   transformer matching ../le-wm's normalization/residual/init regime.
#   WHAT CHANGED.
#     - Residual blocks: nGPT eigen-learning-rate interpolation `h=unit_norm(h+eta*(f-h))`
#       -> canonical pre-LN residual ADD. Encoder block: `h=h+attn(LN(h)); h=h+mlp(LN(h))`.
#       Predictor block: le-wm DiT AdaLN-zero `h=h+gate*attn(modulate(LN(h),shift,scale))`
#       (+ same for the MLP), gates zero-init so each sublayer starts at identity.
#     - Attention: removed sqk + per-head q/k unit-normalization + the cosine-logit scale
#       (head_dim**0.5); now ordinary scaled-dot-product (scale head_dim**-0.5). RoPE in the
#       temporal predictor is KEPT (positional scheme, orthogonal to nGPT).
#     - MLP: nGPT gated-SwiGLU-with-sqrt(D) + learned act_scale -> plain Linear->GELU->Linear.
#     - Norms: the unit_norm(...) wrappers throughout encode_summary / _predictor_trunk /
#       MTP heads / imagine_step are dropped. embed_norm stays Identity (first block pre-LNs);
#       encoder_norm and predictor_norm become real LayerNorms (le-wm final norms). adaLN
#       conditioning activation relu^2 -> SiLU (le-wm).
#     - Per-step weight projection: normalize_world_model_matrices() and ALL its call sites
#       are DELETED. AdamW weight decay (lewm_weight_decay=1e-3) now bounds the WM weights.
#   WHY. The unit-sphere constraint and the JEPA objective fought each other (a sphere-bound
#   token cannot be isotropic Gaussian -- the whole reason v183 added the BN projector). With
#   the objective already in emb space, keeping the recurrent/belief path on the sphere was a
#   half-measure; a standard transformer + weight decay is le-wm's actual, validated regime.
#   SCALE CONSISTENCY. Everything stays ~unit-variance per dim: xavier projections preserve
#   variance, LayerNorms (block pre-norms + encoder/predictor final norms) bound it, and the
#   recurrent feedback (pred_next_proj output) matches the encoder-summary scale, so train-
#   vs-imagine input distributions agree. The agent input is unaffected: it still passes
#   through agent_input_ln (LayerNorm, scale-invariant), so the actor/critic see the same
#   standardized signal. SIGReg/MSE already live in the BatchNorm'd projector emb (v183), so
#   they are unchanged by de-norming the tokens.
#   RISK. Without the unit sphere or weight renorm, attention/residual magnitudes are bounded
#   only by LayerNorm + wd; watch early-training stability (KL, wm losses) and imagination
#   magnitude drift over the context=5 window. The loss SCALES shift again vs v183 (raw token
#   variance changed), so re-check lewm_sigreg_coef / lewm_reward_loss_coef balance at ~1M.
#
# -----------------------------------------------------------------------------
# v183 = v182 with the JEPA SELF-PREDICTION OBJECTIVE realigned to ../le-wm.
#   IDEA. v179.x trains the WM latent stream with a TOKEN-SPACE cosine loss on the
#   unit-sphere summaries and runs SIGReg on those same unit-norm tokens (rescaled by
#   sqrt(D) to fake unit per-dim variance). le-wm instead trains JEPA in the output of a
#   PROJECTOR HEAD -- Linear(D->H) -> BatchNorm1d(H) -> GELU -> Linear(H->D) -- with an
#   MSE loss in that BatchNorm'd "emb" space, and SIGReg on the SAME emb (no sqrt(D)).
#   v183 adopts exactly that for the objective while leaving the rest of the WM on the
#   unit sphere.
#   MECHANISM.
#     - New JEPAProjector (jepa_projector) on world_model. Per update we project the
#       encoder summaries AND the predicted next-token through ONE shared BatchNorm call
#       (flatten both to (N, D), concat, project, split) so pred_emb and target_emb share
#       batch statistics and are consistently normalized -- essential, since BN is
#       batch-dependent. track_running_stats=False (always-train BN == le-wm; also
#       torch.compile/cudagraph-safe, no buffer mutation to capture).
#     - wm_latent_loss = MSE(pred_emb, target_emb) in emb space (replaces token cosine as
#       the TRAINING loss; cosine/MSE-in-token kept only as diagnostics).
#     - SIGReg now runs on emb_sequence (the projected [initial, targets]); the sqrt(D)
#       rescale is dropped -- BN makes the N(0,I) target well-posed without it.
#   WEIGHT DECAY. lewm_weight_decay 0.0 -> 1e-3, applied (AdamW) to ALL WM param groups
#     (latent stream + detached outcome-probe IO + the new projector). PPO/non_wm stays
#     0.0. Re-enabled now that the JEPA objective lives in BatchNorm'd (non-sphere) emb
#     space, where decay no longer fights the unit-norm constraint.
#   PATH UNCHANGED. The recurrent/belief/agent path stays unit-sphere: belief_from_history,
#     pred_next_proj, the v182 detached-predicted-embed agent input + agent_input_ln, and
#     imagination are all untouched. The projector is OFF the control path -- it shapes the
#     representation the WM is trained against, not what the agent reads.
#   RISK. Switching the latent loss from cosine (range [0,2]) to emb-MSE and SIGReg from
#     unit*sqrt(D) to BN-emb changes loss SCALES; lewm_sigreg_coef(0.09)/
#     lewm_reward_loss_coef(0.5) balance is now uncertain. Watch wm/sigreg vs wm/latent at
#     1M and rebalance if SIGReg dominates or vanishes.
#
# -----------------------------------------------------------------------------
# v182 = v179.3 with ONE architectural change to the AGENT INPUT.
#   IDEA. v179.x feeds the actor/critic the raw recurrent BELIEF
#   (belief_from_history(...)[:, -1], the predictor-trunk output at the last step).
#   v182 instead feeds the WM's DETACHED PREDICTED next-summary embed --
#   unit_norm(pred_next_proj(belief)), i.e. exactly imagine_step's `next_summary` --
#   followed by a per-token LayerNorm. So the agent acts on "where the WM predicts I
#   am about to be" (the JEPA prediction-target space) rather than the raw predictor
#   state, while training the WM purely from JEPA+SIGReg as before.
#   MECHANISM / GRADIENT ISOLATION.
#     - pred_next_proj is applied inside combine_agent_input at ROLLOUT/DREAM buffer
#       time; agent_input_from_latent then .detach()es the result, so the buffered
#       agent input is the stop-grad predicted embed and pred_next_proj's WM params
#       NEVER receive agent (policy/value) gradient -- consistent with the detached
#       probes / detach_world_model_from_agent philosophy.
#     - the new agent-side nn.LayerNorm(lewm_dim) (agent_input_ln) is applied in the
#       grad-enabled UPDATE forward (_trunks_from_agent_input), per token over dim D
#       (lejepa probe protocol), so the LN trains with the agent. It lives in the
#       non_wm optimizer group (base LR).
#   WHY. The predicted embed is the quantity the WM is explicitly optimized to make
#   predictive (the MSE+SIGReg target); reading the policy off it -- standardized by a
#   trained LayerNorm -- may give a cleaner, more forward-looking control signal than
#   the raw belief. All four agent-input sites (rollout act, real bootstrap, dream act,
#   dream bootstrap) and the standalone agent_input_from_obs route through
#   combine_agent_input, so the swap is centralized there. agent_input_dim unchanged
#   (pred_next_proj is D->D).
#
# -----------------------------------------------------------------------------
# v179.3 clip075-probehighlr-lnprobe-detachedprobes-pretemporalwm-beliefonly-ppoadvnorm-actor1-critic10-cliphi-criticsigma075-dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v179.3 = v179.2 with ONE change: actor_grad_clip and critic_grad_clip 0.25 -> 0.75.
#   - MOTIVATION. Imagined training does ONE actor step per 32768-transition dream block
#     (3 blocks/iter, no minibatching), measured post-step => imagined approx_kl ~5.3e-4 vs
#     real ~0.0197 (~37x lower). With separate_grad_clip the imagined actor grad is max-norm
#     clipped to 0.25, so that single big-block step barely moves the policy -- dreams were
#     near-inert for the actor. Loosen the clip to 0.75 (real path too) so each step counts.
#   - RISK. Looser clip raises step size on BOTH real and imagined updates; watch for KL
#     blow-up / instability vs v179.2's strong curve (4228 @ 4.7M). max_grad_norm=0.5 is the
#     separate_grad_clip=False fallback and is untouched.
#
# -----------------------------------------------------------------------------
# v179.2 = v179.1 with TWO bias-free changes that fix the slow detached-probe FIT (the
# representation was already excellent; the linear heads were the bottleneck). NO bias-init.
#   - DIAGNOSIS. With a detached probe the only DOF that can express the prediction is the
#     head weights/bias, and they move at ~lr/step under Adam. HalfCheetah's continuation
#     target is a CONSTANT logit(gamma)=logit(0.995)=5.29, and the SIGReg'd belief token is
#     mean-zero, so the weights can't shortcut it -- the bias alone must walk 0->5.29. The
#     budget is ~15.6k WM steps at lr=3e-4 ANNEALED (avg 1.5e-4) => max ~2.3 logits =>
#     discount saturates ~sigmoid(2.3)=0.91 (matches the observed imagine/discount_mean~0.86).
#     It is NOT a hard-learning problem; it is traversal-distance / step-budget. The reward
#     101-bin CE lags for the same reason (its scalar MEAN converges first, so the decoded
#     reward is accurate while the distribution stays under-sharpened -- cosmetic for PPO).
#   - FIX 1 (parameter-golf HEAD_LR). The probe heads get a DEDICATED optimizer param group
#     at lewm_probe_lr_mult x base lr (25x => 7.5e-3, mirroring parameter-golf's HEAD_LR=0.008
#     for its readout heads). 25x base annealed traverses ~58 logits of budget => the bias
#     reaches 5.29 in a few hundred steps and then just tracks. Per-group proportional anneal
#     preserves the higher base. This is the real lever; removing SIGReg from the continuation
#     token would NOT help (the weight norm would face the same 5.29 traversal at lr/step).
#   - FIX 2 (lejepa probe protocol). A LayerNorm precedes each probe Linear on the DETACHED
#     belief token (lejepa reads scalars off frozen JEPA embeds via nn.Sequential(LayerNorm,
#     Linear)). Replaces the unit_norm on the reward feat; standardises each of the D dims to
#     unit variance so the linear readout is well-conditioned (the unit-norm token had
#     magnitude ~1, forcing large weight norms). LN params are detached-input only -> they
#     train with the probe, never touch the representation (purity preserved). Both LayerNorms
#     join the dedicated probe LR group + the probe grad-clip group.
#   - Everything else identical to v179.1 (separate outcome/obs SIGReg, detached HL-Gauss
#     reward probe, incoming-convention outcome tokens, pure-JEPA rep, PPO-clip / actor1).
#
# -----------------------------------------------------------------------------
# v179.1 = v179 with ONE change to the SIGReg geometry: the outcome tokens are whitened
# SEPARATELY from the obs tokens instead of in one joint 19-token flatten.
#   - PROBLEM with v179's joint SIGReg. Whitening the full 19-token frame to isotropic N(0,1)
#     pushes ALL dims (incl. the outcome-token dims) to be mutually uncorrelated -- including
#     outcome<->obs. But reward/continuation are genuine FUNCTIONS of state, so the joint
#     whitening actively FIGHTS the reward/continuation signal the detached probe must read
#     off the outcome tokens (and, for the constant HalfCheetah continuation, manufactures
#     spurious obs-derived variance the probe then can't decode -> discount_bce ~17x the
#     non-detached baseline at 500k).
#   - FIX. SIGReg the obs tokens JOINTLY (flatten num_obs*D, keeps v175 cross-obs decorrelation)
#     and EACH outcome token SEPARATELY over its own D dims; outcome tokens are NOT cross-
#     whitened with obs, so they may correlate with state while staying anti-collapsed within
#     themselves. Combined as a token-count-weighted average so total SIGReg magnitude is
#     ~unchanged (isolates the decoupling, not the strength). Two projections per minibatch:
#     A_obs (num_obs*D, P), A_out (D, P).
#   - Everything else identical to v179 (detached HL-Gauss reward probe, incoming-convention
#     outcome tokens, pure-JEPA representation, PPO-clip / belief-only / actor1).
#
# -----------------------------------------------------------------------------
# v179 outcomejepatokens-detachedprobes-jointsigreg (inherited).
#
# v179 = v176.1 with a WORLD-MODEL-ONLY change: the 2 outcome (reward/continuation) tokens
# become first-class JEPA tokens. The WM is now a PURE JEPA over all 19 tokens (2 outcome +
# 17 obs); the PPO-clip policy objective, belief-only agent input, actor1, norm_adv, critic,
# gamma, num_envs are all UNCHANGED. There is no PMPO.
#   - CONTENT-BEARING OUTCOME TOKENS. The outcome tokens were learned-constant query
#     embeddings read off the obs belief by an aux cross-attention (semantic_readout). Now
#     they are tokenized (affine, symmetric to obs) from the ACTUAL incoming transition
#     outcome [symlog(r_in), c_in] at the encoder input, so the embedding already carries
#     reward/continuation info.
#   - INCOMING CONVENTION. Summary at position k encodes the outcome (r_{k-1}, c_{k-1}) of
#     the transition that led INTO state k (episode-start => grounded 0). This is the only
#     convention under which imagination works: predict summary_{t+1} (conditioned on a_t via
#     AdaLN), whose outcome token encodes r_t; the detached probe reads r_t off it.
#   - JOIN THE RECURRENT STREAM. The predictor is purely temporal over the FULL 19 tokens;
#     the semantic_readout / predictor_semantic_query aux path is GONE. JEPA (cosine+mse) and
#     joint flattened SIGReg now cover all 19 tokens (S*D = 19*64 = 1216, was 17*64=1088).
#   - DETACHED SCALAR PROBES. reward CE / continuation BCE come from DETACHED linear probes
#     on the (stop-grad) predicted outcome-token belief: the CE/BCE loss trains ONLY the
#     probe heads, never the encoder/predictor. The representation is shaped PURELY by JEPA
#     (predict non-detached embed) + SIGReg. reward_action_proj is dropped (action
#     conditioning already lives in the predictor AdaLN).
#   - HYPOTHESIS. A pure-JEPA outcome representation with detached scalar probes gives
#     cleaner, less reward-polluted dynamics features and better imagined credit assignment
#     than a loosely-constrained CE/BCE-shaped readout.
#   Built on v176.1 (PPO-clip, belief-only obs_cond_mode="none", actor1, norm_adv).
#
# -----------------------------------------------------------------------------
# v176.1 (inherited) = v176 but with a BELIEF-ONLY agent input (obs_cond_mode="none"), reverting the
# v173 obs-embed FiLM grounding. The actor/critic trunk now receives ONLY the world-model
# recurrent belief (belief_from_history), NOT the concat/FiLM of belief + current-obs embed.
#   - CHANGE. obs_cond_mode "film" -> "none". combine_agent_input buffers the belief alone
#     (1x agent_latent_dim), _fuse_agent_input is a pass-through, and the BeliefConditioner
#     is not built. Mirrors the v168/v171-era input. Everything else (joint SIGReg, purely
#     temporal WM, PPO advnorm) is identical to v176.
#   - WHY. Ablate the obs-embed/FiLM grounding against v176 under the now-stronger v176
#     backbone: does feeding the raw current-obs embed alongside the belief actually help,
#     or is the recurrent belief sufficient? Clean A/B (v176 = belief+obs FiLM, v176.1 = belief only).
#
# -----------------------------------------------------------------------------
# v176 (inherited): swap the percentile return-range advantage normalizer (batchpercnorm /
# ret_percnorm) for the STANDARD PPO-baseline per-minibatch advantage normalization:
#   - CHANGE. norm_adv=True, ret_percnorm=False (mutually exclusive). Per minibatch,
#     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8),
#     exactly ppo_continuous_action.py. adv_transform stays "v10" (identity = raw GAE),
#     norm_adv_scope stays "minibatch", so the advantage pipeline is plain-PPO: raw GAE
#     -> per-minibatch standardize. Applied to BOTH the real and imagined PG streams
#     (the imagined path's norm_adv branch standardizes each imagined minibatch too, so
#     both streams share the same per-minibatch zero-mean/unit-std convention).
#   - WHY. percnorm was scale-only (divide by P95-P5 of returns, no centering); PPO
#     advnorm CENTERS and scales per minibatch. This is the canonical PPO baseline and a
#     cleaner control for the joint-SIGReg WM change — isolates the WM contribution from
#     the bespoke retnorm. NOTE: the advantage itself may still be the soft (entropy-
#     augmented) GAE when auto_alpha is on; only the NORMALIZER changed here.
#
# -----------------------------------------------------------------------------
# v175 (inherited): joint (flattened) SIGReg over the obs-token frame vector.
#
# v174, but make SIGReg JOINT over the flattened obs-token frame vector instead of
# per-token-slot — restoring le-wm's "whiten the WHOLE per-frame latent" semantics.
#   - MOTIVATION. v174's SIGReg whitened each obs-token slot's batch-marginal
#     independently, then averaged over slots. That enforces per-slot marginal
#     isotropy but NEVER cross-token joint isotropy: the slots can each look Gaussian
#     while their joint is low-rank (redundant/collapsed tokens). le-wm has ONE latent
#     per frame, so its SIGReg is inherently joint; the per-slot factorization was a
#     v174-only weakening. With v174's predictor now purely temporal (no cross-token
#     mixing in dynamics) and the latent loss per-token cosine, NOTHING in v174 pressures
#     cross-token rank — a blind spot precisely where collapse would hide.
#   - MECHANISM. Per rollout step, flatten the obs tokens (B, S, D) -> (B, S*D) and run
#     SIGReg on that joint frame vector (sample axis = batch B, faithful per-step
#     marginal preserved, averaged over steps). Random projections now mix dims ACROSS
#     tokens, so cross-token correlation is detected and penalized; isotropic target =>
#     MAXIMIZES cross-token effective rank, forcing the per-scalar obs tokens to be
#     mutually decorrelated (the anti-collapse we want — tokens shouldn't be redundant).
#     This is structurally IDENTICAL to le-wm SIGReg on (T, B, D_frame); v174 just
#     happened to factor D_frame into S tokens of D, and v175 concatenates them back.
#   - SCALING UNCHANGED. Each token is unit-norm (per-dim var ~1/D); concatenating S of
#     them keeps per-dim var 1/D, so the existing * sqrt(lewm_dim) rescale still lands
#     at unit per-dim variance for the N(0,1) target. sqrt(D), NOT sqrt(S*D).
#   - num_proj 1024 -> 4096: the projected dim grows D=64 -> S*D (~1088 on HalfCheetah),
#     so denser slicing is needed to cover cross-token directions. (Mean over projections
#     => loss scale unchanged, only lower estimator variance; flattening also DROPS the S
#     axis from the SIGReg intermediate, so this is net cheaper in memory than v174.)
#   - PROBE. Adds charts/effrank_obs_{joint,pertoken}: spectral-entropy effective rank of
#     the encoder obs summaries, jointly (S*D) and mean per-slot (D). joint/pertoken in
#     [1, S] measures cross-token diversity — the direct readout of whether joint SIGReg
#     raises rank vs v174's per-slot version. Read-only (no_grad), no training effect.
#
# -----------------------------------------------------------------------------
# v174 (inherited): purely-temporal predictor; all spatial mixing pre-SIGReg.
#
# v173, but MOVE ALL CROSS-TOKEN (spatial / non-temporal) MIXING TO BEFORE SIGReg:
# the encoder absorbs the predictor's space-attention, and the predictor becomes
# PURELY TEMPORAL over the per-token recurrent stream.
#   - MOTIVATION (le-wm fidelity + SIGReg coverage). SIGReg regularizes the ENCODER
#     summary (the prediction targets). Previously the predictor re-mixed tokens
#     spatially (axes = [space, time, ...]) AFTER SIGReg, so the belief the agent and
#     outcome heads consume was a non-regularized recombination, and the obs targets
#     were matched against a representation that had only seen PART of the spatial
#     mixing. Concentrating all token mixing in the encoder means SIGReg governs the
#     full spatial representation the predictor consumes, and each one-step obs target
#     is a PURE temporal function of that token's own history — le-wm's
#     "encode (non-temporal) -> SIGReg -> predict (temporal)" split, kept token-factored.
#   - MECHANISM.
#       * Encoder: lewm_encoder_layers 2 -> 4 (absorbs the moved spatial depth).
#       * Predictor: axes = ["time"] * lewm_predictor_layers (2), all _RoPEAttention;
#         the obs tokens (the only recurrent stream) are propagated PER TOKEN across
#         time. Each obs token is already cross-contextualized by the deeper encoder,
#         so this is NOT a per-scalar predictor — each token's history carries
#         whole-frame info (cf. le-wm collapsing a frame to one emb vector).
#       * Readout: reward/continuation semantic tokens can no longer reach obs tokens
#         via predictor space-attn, so a single auxiliary cross-attention
#         (_SpaceCrossReadout) reads them off the time-mixed obs belief. It sits
#         OUTSIDE the recurrent stream (only obs tokens feed back), is DOWNSTREAM of
#         the obs prediction, and never touches the SIGReg'd targets — an outcome-head
#         readout, not one of the moved space blocks. num_latent_tokens (=[2 semantic,
#         obs]) is unchanged, so agent_latent_dim / the v173 FiLM / decode_belief_outcomes
#         are all untouched (zero agent-side blast radius).
#   - HYPOTHESIS. SIGReg now shapes the full spatial representation; per-token temporal
#     dynamics on a contextualized basis are better-conditioned and more le-wm-faithful
#     -> steadier WM latent loss and a cleaner belief for the actor. RISK: the predictor
#     cannot form NEW cross-token combinations during imagined rollout beyond what the
#     encoder baked in; the deeper encoder mitigates this. If WM latent loss regresses
#     vs v173, that is the signal a late in-predictor space block is needed.
#
# -----------------------------------------------------------------------------
# v173 (inherited): condition the WM belief on the raw current-obs embed (FiLM).
#
# v172, but CONDITION the WM belief on the raw current-obs embed (FiLM) instead of
# CONCATENATING them — fusing both into one agent_latent_dim vector to keep the
# trunk input width (and entry-projection params) at v171's level.
#   - MOTIVATION. v172 concatenated obs-embed and belief -> 2*agent_latent_dim, so
#     the ThinkTrunk entry Linear doubled (38:1 squeeze on HalfCheetah vs 19:1).
#     Conditioning the belief on the obs embed instead returns ONE agent_latent_dim
#     vector: the belief, affinely corrected by the current observation, so the
#     "where am I now" signal grounds the (possibly drifted, predict-next) belief
#     without inflating the trunk input.
#   - MECHANISM (FiLM / AdaLN-style). belief, obs_embed are both unit-norm token
#     stacks (B, num_latent_tokens, lewm_dim). A per-feature (scale, shift) =
#     to_mod(obs_embed) modulates the belief: fused = unit_norm(belief*(1+gamma)+beta).
#     to_mod is ZERO-INITIALIZED, so at step 0 gamma=beta=0 and fused == belief
#     EXACTLY (v171 belief-only behavior); the obs correction is purely learned, so
#     v173 cannot regress from v171 at init.
#   - TRAINABILITY. The agent input is built under no_grad in rollout and replayed
#     as a buffered leaf in the update, so a conditioner with params must run INSIDE
#     the trunk forward to receive gradient. We therefore BUFFER both latents
#     (concat, as in v172) and fuse them at the trunk entrance in _fuse_agent_input
#     (called by _trunks_from_agent_input) — recomputed every update so to_mod trains.
#     Buffer width stays 2*agent_latent_dim (cheap stored data); TRUNK width is 1x.
#   - Flag `obs_cond_mode`: "film" (default) | "concat" (recover v172) | "none"
#     (recover v171, belief only). detach behavior unchanged.
#
# -----------------------------------------------------------------------------
# v171 (inherited): percentile return-range advantage normalization.
#
# v169, but swap the Dreamer4 std-based advantage normalizer for the percentile
# return-range "retnorm" used in iterthink_v24_beta_d3bucket_mtp_mbpercnorm_v2,
# defaulting to a PER-BATCH (whole-rollout) scope (no EMA):
#   - Each iteration, compute ONE scale S = max(ret_perc_floor, P95 - P5) over the
#     full rollout's RAW GAE returns (b_returns), and divide every minibatch's
#     advantage by that same S for the whole update. Stable within an iteration,
#     reactive across iterations (recomputed fresh each rollout, no EMA warmup).
#   - DreamerV3 semantics: divide-only (no offset), floor 1.0, percentiles 5/95.
#     Reward/critic targets stay RAW (valnorm/advnorm = none).
#   - ret_perc_scope also supports "minibatch" (fresh per-mb spread, more local but
#     noisier) and "ema" (v1's slow global EMA(P95)-EMA(P5), rate 0.01, warms up
#     from a floored 1.0). Default "batch" = the per-rollout middle ground.
#   - The dream (imagined) advantage is scaled by the shared per-iteration
#     ret_perc_scale under batch/ema, or the analogous per-dream-block percentile
#     of the dream returns under "minibatch" scope.
# Flag `ret_percnorm` (default True); the old `norm_adv` CleanRL path stays off.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v169 emaadvnorm-actor1-critic10-cliphi-criticsigma075-dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v168 + restore advantage normalization, but in the Dreamer4 style instead of
# CleanRL per-minibatch whitening. We keep one persistent EMA of the *return*
# spread and divide the policy advantage by it:
#   - Each iteration, take the real-rollout GAE returns, clamp to their own
#     [5%, 95%] quantiles (outlier-robust), and lerp a slow EMA of their mean/var.
#   - Normalized advantage = raw_advantage / ema_returns_std. (Dreamer4 normalizes
#     returns AND old_values by (mean, std); the mean cancels in returns - values,
#     so this is exactly advantage / return_std — a scale-only, slow, global
#     normalizer, NOT per-minibatch recentering.)
#   - The SAME ema_returns_std scales both the real-PPO advantage and the imagined
#     (dream) advantage, since both live in the same raw value/return units.
# This is purely an actor-PG scale tool. NO reward/return-target normalization:
# the reward head and critic stay in raw Dreamer3-bucket/HL-Gauss space, matching
# Dreamer4 (which keeps heads raw symexp/twohot and only EMA-normalizes advantage).
# Decay default is faster than Dreamer4's 0.998 because we update the EMA once per
# outer iteration (~hundreds of times per run), not per optimization step.
# Flag `ema_adv_norm` (default True); the old `norm_adv` CleanRL path stays off.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v168 actor1-critic10-cliphi-noadvnorm-criticsigma075-dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v167 + restore the previous asymmetric clip-higher actor objective in both
# real and imagined policy losses:
#   max(-A*r, -A*clamp(r, 1-clip_coef, 1+clip_coef_high)).
# Actor updates still run only for one epoch while critic trains for all
# update_epochs=10. KL early-stop remains removed; KL/ratio/clipfrac diagnostics
# are still logged for actor updates.
# With the default shared actor/critic trunk, critic-only epochs update only the
# critic head so the policy representation is not moved after the actor epoch.
# Value loss was already unclipped. Advantage handling is raw GAE only:
# adv_transform="v10" with norm_adv=False.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v166 normaladv-criticsigma075-dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v165 + remove rankgauss policy-advantage shaping. adv_transform="v10" feeds
# raw GAE into PPO, while the existing norm_adv=True / norm_adv_scope="minibatch"
# applies ordinary PPO advantage normalization.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v165 criticsigma075-dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v164 + set the critic Dreamer3-bucket HL-Gauss projection sigma ratio to 0.75
# bins, matching the current reward-head sigma ratio. This keeps the same bucket
# support/decode/projection math, but makes critic CE targets sharper.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v164 dreamer3bucket-hlgauss-wmloss1024-cudagraph-contdisc-k6.
#
# v163 + restore the HalfCheetah-v4__v24_beta_v162critic_dreamer3bucket_hlgauss_mtp_v1
# HL-Gauss semantics for both critic returns and WM rewards:
# - bucket centers are raw scalar values generated as symexp(evenly spaced
#   symlog coordinates), with one exact zero bucket;
# - targets are projected by Gaussian CDF mass over the symlog-coordinate
#   intervals, clamped to the coordinate support;
# - scalar consumers decode E[raw bucket center], matching Dreamer3 scalar
#   bucket consumption while retaining HL-Gauss smoothing.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v163 wmloss1024-cudagraph-edgeclamp-contdisc-k6.
#
# v162 + restore the WM supervised minibatch to 1024 with no gradient
# accumulation. Each WM minibatch now does exactly one forward, one backward,
# one clip, and one optimizer step. The le-wm-style combined WM loss remains
# compiled without CUDA graphs; agent policy/value calls keep reduce-overhead
# CUDA graph compile.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v162 compiled-wmloss-cudagraph-edgeclamp-contdisc-k6.
#
# v161 + le-wm-style compiled supervised WM loss. The WM training path now uses
# one callable that packages encoder(obs_t), encoder(obs_{t+1:t+H}), teacher
# history construction, predictor, outcome decode, latent/reward/discount losses,
# and SIGReg before backward. This mirrors ../le-wm/train.py's training wrapper
# shape and avoids multiple separately compiled WM forwards whose saved CUDA graph
# outputs must survive until one later backward.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v161 cudagraph-edgeclamp-contdisc-k6.
#
# v160 algorithmic surface, but torch.compile is carried forward explicitly with
# CUDA graphs enabled. WM training uses fixed 512-sample microbatches while
# preserving the 1024 effective batch size; this keeps capture shapes stable and
# lowers the eager/backward peak that OOMed v160 at the WM loss backward.
#
# Key inherited choices:
# - v152 raw affine obs tokenizer, no env obs symlog.
# - Critic/reward scalar decode as E[symexp(z_center)].
# - Reward HL-Gauss edge support with clamped real-space targets.
# - Continuation target/consumption in discounted gamma*(1-terminal) space.
#
# Compile policy:
# - torch.compile default ON, reduce-overhead mode, cudagraphs ON.
# - dynamic=False to favor stable graph capture for fixed CleanRL batch shapes.
# - scalar-output capture ON for Dynamo bookkeeping paths.
# - compiled callable invocations mark CUDA graph step boundaries before entry,
#   matching ../parameter-golf's captured microstep pattern.
# - because this RL loop compiles smaller callables instead of one whole loss,
#   compiled outputs are cloned immediately outside the graph before they are
#   stored in rollout buffers or fed into a later compiled callable.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v160 v152obs-expecteddecode-edgeclamp-contdisc-k6 (v159 + reward edge support).
#
# Key change from v159:
#   - Keep v159's real-space expected scalar decode, but use edge-support
#     HL-Gauss projection for reward with target clamping. Edge support defines
#     interval mass assignment; scalar consumers still read E[symexp(center)].
#     This keeps the cleaner projection geometry without the unclamped
#     out-of-support fragility from strict hl-gauss-pytorch parity.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v159 v152obs-expecteddecode-contdisc-k6 (v158 + restore v152 obs/decode choices).
#
# Key change from v158:
#   - Restore the v152 raw affine observation tokenizer: one learned weight and
#     bias vector per scalar, no symlog/multiscale preprocessing.
#   - Restore v152 scalar consumption for critic values and WM rewards:
#     `E[symexp(z)]` instead of library-style `symexp(E[z])`.
#   - Restore v152 reward HL-Gauss support shape: center support with clamped
#     labels. This deliberately backs away from strict hl-gauss-pytorch parity
#     because the goal is to recover the v152 learning profile.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v158 hlgaussref-compile-contdisc-multiscale-obs-dreamkl-k6 (v157 + hl-gauss-pytorch parity).
#
# Key change from v157:
#   - Align fixed-support critic and reward HL-Gauss with ../hl-gauss-pytorch:
#     supports are edges, centers decode by inverse_transform(E[center]), Gaussian
#     CDF labels are normalized over support mass without clamping targets to the
#     edge first, and auxiliary spread/CDF probes stay in the transformed support
#     coordinates instead of mixing in E[symexp(center)] raw-space statistics.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v157 compile-contdisc-multiscale-obs-dreamkl-hlscalar-k6 (v156 + idiomatic torch.compile).
#
# Key change from v156:
#   - Compile the hot callables that this single-file agent actually invokes:
#     WM encode/predict/decode/imagine helpers and actor/critic agent-input
#     entry points. Compiling the module's `forward` would be mostly inert here
#     because the training loop calls custom methods directly. Default mode stays
#     eager-training friendly and disables CUDA graph capture to avoid adding graph
#     pools on top of the already tight WM backward memory budget.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v156 contdisc-multiscale-obs-dreamkl-hlscalar-k6 (v155 + D3-style continuation discount).
#
# Key change from v155:
#   - Train the WM continuation head as the predicted dream discount
#     `gamma * (1 - terminal)`, matching Dreamer3's `contdisc` semantics, and
#     consume that scalar directly in imagined GAE. This avoids an infinite-logit
#     target for normal nonterminal MuJoCo transitions while preserving real-env
#     GAE's explicit gamma and terminal handling.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v155 multiscale-obs-dreamkl-hlscalar-k6 (v154 + tokenizer/KL/decode cleanup).
#
# Key changes from v154:
#   - Replace the raw affine observation tokenizer with a shared symlog
#     multiscale scalar basis plus learned observation slot embeddings, so raw
#     observation magnitude is encoded as token direction before nGPT unit
#     normalization.
#   - Add post-step KL/clipfrac logging and target-KL early stopping for dream
#     PPO batches. Dream batches are generated by the current policy, so the
#     meaningful trust-region check is after the update on the just-used batch.
#   - Consume HL-Gauss scalars with library-style `symexp(E[z])` decode for both
#     critic values and world-model rewards, while leaving CE targets unchanged.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v154 agent-transfer-residuals-k6 (v152 + indexed agent trunk transfer residuals).
#
# Key change from v152:
#   - Add the obs-idx-transfer idea from
#     ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_mlp10_obsidxtransfer_v1.py
#     to the current ThinkTrunk while keeping its existing residual machinery:
#     each ThinkBlock receives learned same-index transfer residuals from x0 and
#     earlier block outputs in its preactivation, then still applies the existing
#     convex x/x0 residual, dense branch, and soft-MoE branch.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v152 no-outcome-readout-norms-k6 (v151 + remove reward/continue final norms).
#
# Key change from v151:
#   - Remove the role-specific RMSNorm adapters directly in front of the WM
#     reward and continuation final logit heads. Outcome tokens are already
#     nGPT-unit latents; final supervised logit heads should read them directly
#     instead of getting an extra readout normalization stage.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v151 outcome-logits-unbounded-k6 (v150 + unconstrained reward CE logits).
#
# Key change from v150:
#   - Treat both WM outcome decoders as final logit heads, not representation
#     projections. Reward keeps the v149 normalized token+action feature and
#     bias-free zero initialization, but its HL-Gauss CE logits are now produced
#     by an unconstrained linear head with no learned temperature.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v150 cont-unbounded-k6 (v149 + unconstrained continuation BCE readout).
#
# Key change from v149:
#   - Keep the reward/critic HL-Gauss and latent projection fixes, but revert the
#     continuation BCE head to the v148-style unconstrained scalar readout. The
#     continuation target is almost always true in MuJoCo and needs to saturate
#     logits quickly; nGPT row-normalizing this one-logit readout made dreams
#     underconfident (continue ~=0.85).
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v149 nGPT-WM-readout-projfix-k6 (v148 + nGPT readout/projection cleanup).
#
# Key changes from v148:
#   - Compose reward token and action embedding first, then L2-normalize before
#     the HL-Gauss reward logits. The reward HL-Gauss head is bias-free,
#     row-normalized, and uses an nGPT-style learned logit scale.
#   - Remove bias from the critic HL-Gauss logits too; symmetric supports make
#     zero logits neutral.
#   - Row-normalize latent target/prediction projection heads as embedding-like
#     maps instead of column-normalizing them as residual branch outputs.
#   - Log observation-token bias/signal diagnostics without changing the
#     tokenizer.
#
# Hypothesis:
#   v148 improved latent cosine, but some readout/projection paths still violated
#   the target hypersphere geometry. Cleaning these paths should reduce outcome
#   gradient spikes without weakening latent learning.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v148 nGPT-WM-coslat-k6 (v147 + raw cosine latent loss).
#
# Key change from v147:
#   - Train obs-token latent prediction with explicit raw cosine loss
#     `1 - dot(pred, target)` instead of per-coordinate MSE. Since nGPT latents
#     are L2-unit vectors, the old MSE was exactly cosine loss scaled by 2/dim;
#     this intentionally makes the latent objective ~dim/2 stronger.
#   - Keep logging the old per-coordinate MSE as a diagnostic, and add direct
#     `lewm/obs_latent_cosine` plus `lewm/obs_latent_cosine_loss`.
#
# Hypothesis:
#   v147's latent MSE looked small mostly because unit-vector MSE has a compressed
#   scale. Raw cosine loss gives WM latent prediction gradients the intended
#   magnitude under nGPT geometry.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v147 nGPT-WM-sigregscale-k6 (v146 + SIGReg scale fix).
#
# Key change from v146:
#   - Keep nGPT world-model latents on the L2 unit sphere, but feed SIGReg
#     sqrt(dim)-scaled copies of obs tokens. SIGReg's Gaussian ECF target expects
#     random unit projections to have variance ~=1; nGPT unit tokens have variance
#     ~=1/dim and otherwise sit at a large irreducible SIGReg floor.
#
# Hypothesis:
#   v146's high obs_sigreg is a scale-convention conflict, not poor learning.
#   Scaling only the auxiliary SIGReg view restores angular/isotropy pressure
#   without fighting nGPT's consumed unit-token geometry.
#
# -----------------------------------------------------------------------------
# Previous header:
# PPO + IterThink v146 nGPT-WM-k6 (v145 + L2-normalized WM token geometry).
#
# Key change from v145:
#   - Replace the world-model encoder's RMSNorm/MHA residual stack with
#     nGPT-style hypersphere updates: U(U(x) + eta * (U(branch) - U(x))).
#   - Use explicit L2 token normalization at world-model token input/output
#     boundaries and per-head L2 q/k normalization after projection. Time-axis
#     predictor attention applies RoPE before q/k normalization.
#
# Hypothesis:
#   Raw observation token scales remain a bottleneck even with input RMSNorm.
#   Keeping WM latents, residual updates, and attention comparisons on a unit
#   sphere should stabilize covariance geometry without changing PPO actor/critic
#   code or restoring environment observation normalization.
#
# -----------------------------------------------------------------------------
# Previous header:
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
# CRITIC (ported from iterthink_v24_beta_v162critic_dreamer3bucket_hlgauss_mtp_v1):
#   The distributional categorical C51-projected lambda-return critic was replaced
#   by a Dreamer3-bucket HL-Gauss MTP critic:
#     - Target: SCALAR GAE lambda-return projected onto a fixed support with HL-Gauss
#       in symlog-coordinate space, with raw scalar bucket centers generated by
#       symexp(linspace(v_min, v_max, num_bins)). Applied in BOTH the real and
#       imagined critic-training paths.
#     - MTP: the critic head emits critic_mtp_horizon return rows; horizon 0 is
#       V(s_t), horizons 1..H-1 predict returns[t+h] from the same features. Loss is
#       per-horizon CE summed over a validity mask (episode-boundary + tail masked in
#       the real path; tail-only in the dream, where continuation is folded into GAE).
#     - nocriticbias: the peaked zero-return logit-bias prior is removed; v149 makes
#       the critic logits bias-free so the scalar HL-Gauss target sets the scale.
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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def binary_target_entropy(target):
    eps = torch.finfo(target.dtype).eps
    target = target.clamp(eps, 1.0 - eps)
    return -(target * target.log() + (1.0 - target) * (1.0 - target).log())


def configure_torch_compile(args):
    if not args.torch_compile:
        return
    try:
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = args.torch_compile_cache_size_limit
        dynamo.config.capture_scalar_outputs = args.torch_compile_capture_scalar_outputs
    except Exception as exc:
        print(f"warning: failed to configure torch._dynamo: {exc}")
    if args.torch_compile_disable_cudagraphs:
        try:
            import torch._inductor.config as inductor_config

            if hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
        except Exception as exc:
            print(f"warning: failed to disable inductor cudagraphs: {exc}")


def maybe_compile_callable(fn, name, args):
    if not args.torch_compile:
        return fn
    if not hasattr(torch, "compile"):
        print(f"torch.compile unavailable; running {name} eagerly")
        return fn
    print(
        f"compiling {name} with torch.compile("
        f"mode={args.torch_compile_mode!r}, fullgraph={args.torch_compile_fullgraph}, "
        f"dynamic={args.torch_compile_dynamic})"
    )
    compiled_fn = torch.compile(
        fn,
        mode=args.torch_compile_mode,
        fullgraph=args.torch_compile_fullgraph,
        dynamic=args.torch_compile_dynamic,
    )

    def clone_graph_output(output):
        if torch.is_tensor(output):
            return output.clone()
        if isinstance(output, tuple):
            return tuple(clone_graph_output(item) for item in output)
        if isinstance(output, list):
            return [clone_graph_output(item) for item in output]
        if isinstance(output, dict):
            return {key: clone_graph_output(value) for key, value in output.items()}
        return output

    def compiled_call(*call_args, **call_kwargs):
        if args.torch_compile_mark_step_begin and not args.torch_compile_disable_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        output = compiled_fn(*call_args, **call_kwargs)
        if args.torch_compile_clone_outputs and not args.torch_compile_disable_cudagraphs:
            output = clone_graph_output(output)
        return output

    return compiled_call


def compile_agent_callables(agent, args):
    if not args.torch_compile:
        return
    wm = agent.world_model
    # Keep original handles for the le-wm-style supervised WM loss wrapper. That
    # wrapper is compiled as one region, so it must call the unwrapped methods
    # internally rather than nesting several compiled CUDA-graph callables.
    wm._eager_encode_summary = wm.encode_summary
    wm._eager_predict_mtp_from_history = wm.predict_mtp_from_history
    # Do not separately compile WM inference methods in this variant. The
    # supervised WM loss owns one CUDA-graph compiled region over the WM params;
    # separately captured encode/predict/imagine methods share those params and
    # can invalidate graph-owned saved tensors before backward. This mirrors
    # parameter-golf's single compiled training owner more closely.
    agent.get_value_from_agent_input = maybe_compile_callable(
        agent.get_value_from_agent_input, "agent.get_value_from_agent_input", args
    )
    agent.get_action_and_value_from_agent_input = maybe_compile_callable(
        agent.get_action_and_value_from_agent_input, "agent.get_action_and_value_from_agent_input", args
    )


def log_labeled_rollout_diagnostic(writer, prefix, global_step, latent_mse_sum,
                                   reward_abs_sum, reward_err_sum,
                                   discount_bce_sum, discount_entropy_sum, weight):
    # Labeled replay-action rollout diagnostic. These are not consumed-dream
    # metrics: they compare a replay-action latent rollout against replay labels.
    # Use `imagine/*` for actual generated dream statistics.
    eps = 1e-8
    total_w = weight.sum().clamp_min(eps)
    writer.add_scalar(f"{prefix}/latent_mse", (latent_mse_sum.sum() / total_w).item(), global_step)
    writer.add_scalar(f"{prefix}/reward_mae", (reward_abs_sum.sum() / total_w).item(), global_step)
    writer.add_scalar(f"{prefix}/reward_bias", (reward_err_sum.sum() / total_w).item(), global_step)
    discount_bce = discount_bce_sum.sum() / total_w
    discount_bce_excess = (discount_bce_sum.sum() - discount_entropy_sum.sum()) / total_w
    writer.add_scalar(f"{prefix}/discount_bce", discount_bce.item(), global_step)
    writer.add_scalar(f"{prefix}/discount_bce_excess", discount_bce_excess.item(), global_step)
    for h in range(weight.shape[0]):
        w_h = weight[h]
        if w_h.item() <= 0:
            continue
        discount_bce_h = discount_bce_sum[h] / w_h
        discount_bce_excess_h = (discount_bce_sum[h] - discount_entropy_sum[h]) / w_h
        writer.add_scalar(f"{prefix}/latent_mse_h{h + 1}", (latent_mse_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/reward_mae_h{h + 1}", (reward_abs_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/reward_bias_h{h + 1}", (reward_err_sum[h] / w_h).item(), global_step)
        writer.add_scalar(f"{prefix}/discount_bce_h{h + 1}", discount_bce_h.item(), global_step)
        writer.add_scalar(f"{prefix}/discount_bce_excess_h{h + 1}", discount_bce_excess_h.item(), global_step)


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
    torch_compile: bool = True
    torch_compile_mode: Optional[str] = "reduce-overhead"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: Optional[bool] = False
    torch_compile_cache_size_limit: int = 256
    torch_compile_capture_scalar_outputs: bool = True
    torch_compile_disable_cudagraphs: bool = False
    torch_compile_mark_step_begin: bool = True
    torch_compile_clone_outputs: bool = True
    torch_float32_matmul_precision: str = "high"

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
    actor_update_epochs: int = 1
    norm_adv: bool = True  # v176: standard PPO-baseline per-minibatch advantage normalization
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
    target_kl: Optional[float] = 0.03  # diagnostic reference only; v167 does not early-stop on KL

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.75    # v179.3: 0.25->0.75. Imagined actor takes ONE clipped step per 32768-row dream block; at 0.25 imagined KL was ~5e-4 (vs 0.0197 real) => dreams barely moved the policy. Loosen so imagined (and real) steps actually count.
    critic_grad_clip: float = 0.75   # v179.3: 0.25->0.75 (matched).

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    # Raw-reward return support endpoints, expressed in the symlog coordinate
    # system used to generate Dreamer3 symexp-spaced raw bucket centers:
    # symlog(-2000)=-7.6014023346, symlog(20000)=9.9035375513.
    # symlog(-20000)=-9.9035375513, symlog(20000)=9.9035375513.
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75  # HL-Gauss projection sigma in Dreamer3 bucket coordinates
    critic_mtp_horizon: int = 6            # critic MTP: predict H future return rows per state

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"
    # Dreamer4-style EMA advantage normalization (scale-only). Divides the policy
    # advantage by a slow EMA of the return spread; no reward/return-target norm.
    # Percentile return-range advantage normalization (DreamerV3 retnorm, mbpercnorm_v2 style).
    # adv /= S = max(ret_perc_floor, P95 - P5) of the RAW GAE returns; divide-only, no offset.
    ret_percnorm: bool = False  # v176: disabled in favor of norm_adv (mutually exclusive)
    ret_perc_scope: str = "batch"  # "batch": fresh per-rollout spread (no EMA, default);
    #                                "minibatch": fresh per-mb spread; "ema": slow global EMA(P95)-EMA(P5)
    ret_perc_rate: float = 0.01       # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05         # P5
    ret_perc_hi: float = 0.95         # P95
    ret_perc_floor: float = 1.0       # scale floor S=max(floor, .) (DreamerV3 limit=1.0)

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
    lewm_dim: int = 64
    lewm_encoder_layers: int = 4  # v174: 2->4, absorbs the predictor's moved spatial depth
    lewm_predictor_layers: int = 2  # v174: now PURELY TEMPORAL (was 4 = 2 space + 2 time)
    lewm_heads: int = 4
    lewm_kv_heads: int = 2
    lewm_ffn_mult: int = 4
    lewm_context: int = 5
    lewm_dyn_horizon: int = 5
    lewm_mtp_len: int = 4
    lewm_update_epochs: int = 2       # 1->2: WM was getting 10x fewer grad steps than the agent
    lewm_minibatch_size: int = 1024   # v163: no WM grad accumulation; one 1024-row minibatch per optimizer step.
    lewm_probe_lr_mult: float = 25.0  # v179.2: dedicated LR multiplier for the detached probe heads (reward/continuation readout + their LayerNorms). 25x base (=7.5e-3) ~ parameter-golf HEAD_LR=0.008. Fixes the lr-budget bottleneck so the constant continuation logit(0.995)=5.29 is reachable.
    wm_warmup_steps: int = 0
    lewm_reward_loss_coef: float = 0.5
    lewm_discount_loss_coef: float = 0.5
    lewm_reward_num_bins: int = 101
    lewm_reward_raw_v_min: float = -50.0
    lewm_reward_raw_v_max: float = 50.0
    # symlog(-50)=-3.9318256327, symlog(50)=3.9318256327.
    lewm_reward_v_min: float = -3.9318256327243257
    lewm_reward_v_max: float = 3.9318256327243257
    lewm_sigreg_coef: float = 0.09
    lewm_sigreg_num_proj: int = 4096  # v175: 1024->4096, joint SIGReg projects S*D (~1088) not D=64, needs denser slicing
    lewm_sigreg_knots: int = 17
    lewm_sigreg_min_valid: int = 32
    lewm_weight_decay: float = 1e-3   # v183: align with le-wm (AdamW wd=1e-3, applied uniformly to the WM params incl. the new projector). Was 0.0 (unit-sphere shrinkage concern) -- re-enabled now that the JEPA objective lives in BatchNorm'd (non-sphere) emb space.
    lewm_grad_clip: float = 1.0       # v103: WM grad-norm clip (le-wm uses 1.0); PPO keeps max_grad_norm=0.5
    lewm_rope_fraction: float = 0.25  # v104: partial RoPE on time axis (rotate 25% of head_dim, rest position-agnostic)
    lewm_predictor_gate_init: float = 0.1  # v186: AdaLN gate-bias warm-start (>0 keeps predictor sublayers active at init)
    lewm_projector_hidden: int = 512  # v190: IN-PATH token projector hidden width (8x dim, mirroring the reference's 2048 = 8x256)
    # v188: PLAIN conditional flow-matching next-latent head (v3-plainflow ablation).
    lewm_flow_blocks: int = 3               # AdaLN-zero DiT MLP blocks in the flow head
    lewm_flow_hidden_mult: int = 4          # per-block MLP hidden = mult * lewm_dim
    lewm_flow_cond_dim: int = 128           # conditioning MLP width (belief tok + signal embed)
    lewm_flow_signal_levels: int = 64       # K_MAX: discrete signal grid, t = k / K_MAX
    lewm_flow_signal_embed_dim: int = 16    # learned Embedding(K_MAX, this)
    lewm_flow_latent_bound: float = 3.0     # B*tanh(x/B) on encoder summaries AND flow x-preds
    lewm_flow_ctx_noise: float = 0.1        # noisy-context lerp weight on ALL predictor-trunk inputs (train + inference)
    lewm_flow_self_cond_prob: float = 0.25  # scheduled-sampling: replace a context token by the model's own shifted one-shot pred
    lewm_flow_meanpath_prob: float = 0.1    # fraction of rows forced to the EXACT policy-input query (z=0, signal=0): anchors f(0|belief) to E[x|belief] (training otherwise only sees z_t=eps at signal 0)
    lewm_flow_gate_init: float = 0.1        # v192: flow-block AdaLN gate-bias warm start (mirrors lewm_predictor_gate_init); with the shift-weight xavier init this makes x_pred state-dependent from step 0
    lewm_flow_outproj_gain: float = 0.1     # v192: xavier gain on the flow out_proj (was zero-init => x_pred==0 at init, a dead state-independent policy input); 0.1 keeps init x_pred std ~0.1
    lewm_flow_sample_steps: int = 8         # plain Euler steps for sampled paths (d = K_MAX / steps; v3-plainflow uses 8)
    lewm_flow_dream_temperature: float = 0.0  # v190: 1.0->0.0. Deterministic one-shot conditional-mean dreams (v186-like) until lewm/flow_mean_mse proves the flow out; >0 => z0 ~ N(0,I)*tau + 8-step Euler
    lewm_flow_eval_samples: int = 16        # K sampled paths in the per-iteration mean-vs-sampled MSE diagnostic (0 disables)
    detach_world_model_from_agent: bool = True
    # v173: how to fuse the raw current-obs embed with the history belief before the trunk.
    #   "film":   FiLM-condition the belief on the obs embed -> 1x agent_latent_dim (default).
    #   "concat": concatenate both -> 2x agent_latent_dim (v172).
    #   "none":   belief only -> 1x agent_latent_dim (v171).
    obs_cond_mode: str = "none"  # v176.1: belief-ONLY agent input (no obs embed / FiLM), like v168/v171
    diagnostics_length1_ar: bool = False
    diagnostics_warm_ar: bool = True

    # v100 dreamer4-style imagination training. The frozen (detached) WM is used as
    # a simulator to generate fresh on-policy rollouts that train the SAME
    # actor/critic at full strength.
    imagine_enable: bool = True
    imagine_batches: int = 3          # c: generate->train cycles per iteration (each a 1-epoch step).
                                      # Scaled 8->3: at 8 the imagined updates swamped real-PPO and
                                      # drove the entropy collapse (return stuck -180 while base v99
                                      # reached +2200); fewer cycles reduce that update-volume pressure.
    imagine_batch_size: int = 4096    # starting states per imagined block (parallel envs in dream)
    imagine_horizon: int = 8          # imagined rollout length; block = horizon * batch_size transitions
    imagine_warmup_steps: int = 100000  # let real PPO + WM establish before imagining
    imagine_prompt_bank_rollouts: int = 6  # v129: total recent raw rollout buffers available for dream prompts

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

    def __init__(self, in_dim, H, n_experts, transfer_slots=0):
        super().__init__()
        self.n_experts = n_experts
        self.transfer_slots = transfer_slots

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        if transfer_slots:
            self.transfer_weight = nn.Parameter(torch.empty(transfer_slots, H))
            nn.init.normal_(self.transfer_weight, mean=0.0, std=np.sqrt(2.0 / (H + transfer_slots)))
        else:
            self.register_parameter("transfer_weight", None)
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

    def forward(self, cat_feats, x0, transfer_feats=None):
        x = self.in_proj(cat_feats)                                   # (B, H)
        if self.transfer_slots:
            if transfer_feats is None or len(transfer_feats) != self.transfer_slots:
                raise ValueError(f"expected {self.transfer_slots} transfer feats, got {0 if transfer_feats is None else len(transfer_feats)}")
            history = torch.stack(transfer_feats, dim=1)               # (B, slots, H)
            transfer = (history * self.transfer_weight.to(dtype=history.dtype).unsqueeze(0)).sum(dim=1)
            x = x + transfer
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
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts, transfer_slots=k))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0, transfer_feats=feats[:-1]))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


def relu_sq(x):
    return torch.relu(x).pow(2)


def symlog(x):
    # Signed log compression. v179 grounds the outcome tokenizer's reward scalar in
    # symlog space (continuation is already in {0,1}), matching how raw rewards are
    # compressed elsewhere so the affine tokenizer sees a well-conditioned input.
    return torch.sign(x) * torch.log1p(x.abs())


def unit_norm(x, dim=-1, eps=1e-12):
    dtype = x.dtype
    x = x.float()
    return (x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)).to(dtype)


@torch.no_grad()
def spectral_effective_rank(feats, eps=1e-12):
    # v175 probe: effective rank = exp(spectral entropy) of the centered feature
    # covariance, p_i = s_i^2 / sum s_i^2. feats: (..., N, F); contracts the (N, F)
    # tail and returns one rank per leading batch dim. N = samples, F = features.
    feats = feats.float()
    feats = feats - feats.mean(dim=-2, keepdim=True)
    s = torch.linalg.svdvals(feats)                       # (..., min(N, F))
    p = s.square()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(p * p.clamp_min(eps).log()).sum(dim=-1)
    return entropy.exp()


# v184: the nGPT residual primitives (ngpt_residual / ngpt_modulated_residual /
# ngpt_gated_silu_mlp) are removed. The world model is now a STANDARD pre-LayerNorm
# transformer (h = h + sublayer(LN(h)); DiT-zero gated residual for the conditioned
# predictor), with GELU MLPs and ordinary scaled-dot-product attention. The unit-sphere
# constraint (and the per-step weight renormalization) is gone; AdamW weight decay (1e-3,
# le-wm) now bounds the weights instead.


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


def latent_bound(x, bound):
    # v187 (dreamer4/JEDI): soft-bound latents to (-bound, bound) via bound*tanh(x/bound).
    # Near-identity for |x| < bound; applied to encoder summaries (flow targets) and to
    # the flow head's x-predictions so both live on the same bounded scale as the
    # N(0, I) flow prior (SIGReg pushes the tokens toward isotropic Gaussian).
    return bound * torch.tanh(x / bound)


class FlowMatchingHead(nn.Module):
    """v188: 3-block AdaLN DiT flow head (MLP-only, per-token), x-parameterization,
    PLAIN conditional flow matching (no shortcut step-size conditioning).

    Predicts the CLEAN next summary token x from a noised input z_t = (1-t)*eps + t*x,
    conditioned on c = MLP(belief token, signal-level embed). The signal level
    k in {0..K_MAX-1} discretizes time t = k/K_MAX (k=0 pure noise, k->K_MAX clean).
    No attention / token mixing: the belief already integrated the context,
    so the head is a pure conditional per-token denoiser.

    v192 WARM START (was full AdaLN-zero: mod/final_mod/out_proj all zero => x_pred == 0
    at init, so the mean-path policy input f(z=0 | belief) was a state-INDEPENDENT
    constant until the head trained up -- a guaranteed dead-input phase). Now, mirroring
    the v186 trunk warm start (lewm_predictor_gate_init):
      - each block_mods GATE bias slice = lewm_flow_gate_init (0.1); gate weights zero;
      - each block_mods SHIFT slice WEIGHT is xavier-init (scale slice stays fully zero,
        all shift/scale biases zero). NOTE: this deviates from a bias-only warm start
        deliberately -- at the mean-path query z=0 the residual stream h is exactly 0
        (in_proj bias is zero), so LN(h)=0 and the belief can enter ONLY through the
        shift path; with shift weights zero the gate bias would multiply MLP(0)=0 and
        x_pred would remain a belief-independent 0 (verified). Shift-xavier makes
        m = W_s c, so the gated block MLPs are belief-modulated from step 0;
      - out_proj is xavier * lewm_flow_outproj_gain (0.1) instead of zero (LN before it
        renormalizes h, so the output scale ~0.1 is set by this gain; measured init
        x_pred std ~0.10 -- small vs the bounded targets, state-dependent);
      - final_mod stays fully zero (identity modulation; LN output passes at scale 1).
    Velocity is recovered analytically as v = (x_pred - z_t) / (1 - t); callers keep
    t <= (K_MAX-1)/K_MAX so 1-t >= 1/K_MAX."""

    def __init__(self, dim, args):
        super().__init__()
        self.dim = dim
        self.k_max = args.lewm_flow_signal_levels
        assert self.k_max >= 2, "K_MAX must be >= 2"
        self.latent_bound = args.lewm_flow_latent_bound
        hidden = dim * args.lewm_flow_hidden_mult
        cond_dim = args.lewm_flow_cond_dim
        self.signal_embed = nn.Embedding(self.k_max, args.lewm_flow_signal_embed_dim)
        cond_in = dim + args.lewm_flow_signal_embed_dim
        self.cond_fc1 = xavier_linear(nn.Linear(cond_in, cond_dim))
        self.cond_fc2 = xavier_linear(nn.Linear(cond_dim, cond_dim))
        self.in_proj = xavier_linear(nn.Linear(dim, dim))

        def zero_linear(linear):
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
            return linear

        def warmstart_mod(linear):
            # v192: see class docstring. Layout of the 3*dim output: [shift, scale, gate]
            # (forward chunks in that order). shift WEIGHT xavier, gate BIAS 0.1, rest 0.
            zero_linear(linear)
            with torch.no_grad():
                shift_w = torch.empty(dim, linear.weight.shape[1])
                nn.init.xavier_uniform_(shift_w)
                linear.weight[:dim].copy_(shift_w)
                linear.bias[2 * dim : 3 * dim].fill_(args.lewm_flow_gate_init)
            return linear

        self.block_mods = nn.ModuleList(
            [warmstart_mod(nn.Linear(cond_dim, 3 * dim)) for _ in range(args.lewm_flow_blocks)]
        )
        self.block_fc1s = nn.ModuleList(
            [xavier_linear(nn.Linear(dim, hidden)) for _ in range(args.lewm_flow_blocks)]
        )
        self.block_fc2s = nn.ModuleList(
            [xavier_linear(nn.Linear(hidden, dim)) for _ in range(args.lewm_flow_blocks)]
        )
        self.final_mod = zero_linear(nn.Linear(cond_dim, 2 * dim))
        self.out_proj = xavier_linear(nn.Linear(dim, dim), gain=args.lewm_flow_outproj_gain)

    def forward(self, z, signal, ctx):
        # z, ctx: (..., D); signal: long tensor of shape z.shape[:-1].
        cond_in = torch.cat([ctx, self.signal_embed(signal)], dim=-1)
        c = F.silu(self.cond_fc2(F.silu(self.cond_fc1(cond_in))))
        h = self.in_proj(z)
        for mod, fc1, fc2 in zip(self.block_mods, self.block_fc1s, self.block_fc2s):
            shift, scale, gate = mod(c).chunk(3, dim=-1)
            m = F.layer_norm(h, (self.dim,)) * (1.0 + scale) + shift
            h = h + gate * fc2(F.gelu(fc1(m)))
        shift_f, scale_f = self.final_mod(c).chunk(2, dim=-1)
        x_pred = self.out_proj(F.layer_norm(h, (self.dim,)) * (1.0 + scale_f) + shift_f)
        return latent_bound(x_pred, self.latent_bound)


class StdSelfAttention(nn.Module):
    # v186: standard multi-head self-attention with le-wm's INNER affine LayerNorm
    # (module.Attention.norm). le-wm runs a DOUBLE pre-norm per sublayer: an outer
    # affine-free LayerNorm in the block PLUS this inner affine LayerNorm at the
    # attention's own input. v184 kept only the outer norm; v186 restores the inner one for
    # le-wm fidelity (its learnable per-dim gain/bias is the rescale a single norm lacks).
    # scale = head_dim**-0.5 (sdpa default, omitted); no nGPT sqk / q-k unit-norm.
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)  # v186: le-wm inner affine norm
        self.qkv = xavier_linear(nn.Linear(dim, 3 * dim, bias=False))
        self.proj = xavier_linear(nn.Linear(dim, dim, bias=False))

    def forward(self, x, causal=False):
        x = self.norm(x)
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        return self.proj(attn_out)


class LeWMTransformerBlock(nn.Module):
    # v186: le-wm's `Block` -- a DOUBLE-norm pre-LayerNorm transformer block (encoder).
    # le-wm wraps each sublayer in an OUTER affine-free LayerNorm (eps 1e-6) and the
    # attention/FeedForward carry their own INNER affine LayerNorm. v184 collapsed this to a
    # single affine pre-norm per sublayer; v186 restores the double-norm: outer norms here
    # are affine-free (eps 1e-6) and the inner affine norms live in StdSelfAttention.norm /
    # self.mlp_norm. Residual add is the canonical x = x + sublayer(norm(x)).
    def __init__(self, dim, num_heads, ffn_mult):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = StdSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(dim)  # v186: le-wm inner affine norm (FeedForward[0])
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim))

    def forward(self, x, causal=False):
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.w2(F.gelu(self.w1(self.mlp_norm(self.norm2(x)))))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def modulate(x, shift, scale):
    # DiT/AdaLN modulation: scale is a zero-centred multiplicative perturbation,
    # so (1 + scale) keeps the block at identity when the modulation MLP is zero.
    return x * (1 + scale) + shift


class _SpaceAttention(nn.Module):
    # v184: standard non-causal self-attention over the SPACE (feature-token) axis (no
    # positional code; feature tokens have no spatial order). nGPT sqk/q-k-unit-norm and
    # cosine-logit scale removed -> ordinary scaled-dot-product. Norm/residual/gate live in
    # the AdaLN block. (Only used if a predictor layer's axis == "space"; default is all
    # "time", so this is currently dormant but kept correct.)
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)  # v186: le-wm inner affine norm
        self.qkv = xavier_linear(nn.Linear(dim, 3 * dim, bias=False))
        self.proj = xavier_linear(nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        return self.proj(attn_out)


class _RoPEAttention(nn.Module):
    # Causal self-attention over the TIME axis with PARTIAL rotary position embedding
    # (GPT-NeoX style): only the first `rotary_dim = round(head_dim * rope_fraction)`
    # channels per head are rotated; the rest stay position-agnostic. Over a short
    # 5-step imagination window full RoPE wastes its high-frequency pairs (the top
    # pair rotates ~4 rad across the window -- churn with no positional payoff), so a
    # 0.25 fraction keeps relative-position structure on a few low-frequency channels
    # while leaving most of the head free for content matching. v184: standard
    # scaled-dot-product after RoPE (no nGPT q/k unit-norm); norm/residual/gate from AdaLN.
    def __init__(self, dim, num_heads, rope_fraction, rope_theta=10000.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        rot = int(round(self.head_dim * rope_fraction))
        rot -= rot % 2  # rotary_dim must be even (cos/sin come in pairs)
        rot = max(2, min(rot, self.head_dim))
        self.rotary_dim = rot
        self.norm = nn.LayerNorm(dim)  # v186: le-wm inner affine norm
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
        x = self.norm(x)
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
        # v184: standard scaled-dot-product (default scale = head_dim**-0.5); the nGPT
        # sqk / per-head q-k unit-norm / cosine-logit scale are removed. RoPE is retained
        # (it is a positional scheme, orthogonal to the nGPT normalization being dropped).
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        return self.proj(attn_out)


class AdaLNAxialPredictorBlock(nn.Module):
    # le-wm's ConditionalBlock (AdaLN-zero), generalised to the axial predictor.
    # The action enters ONLY here, as the conditioning vector c = action_features
    # (one vector per timestep) -- NOT as input tokens. A zero-init modulation MLP
    # maps c -> (shift, scale, residual modulation) for both sublayers; zero init
    # leaves the base nGPT block active while action influence ramps in. The
    # per-timestep modulation broadcasts over the space tokens (the action conditions
    # the whole frame uniformly). v184: standard DiT gated residual add (no unit sphere);
    # AdaLN supplies shift/scale on the pre-norm and a gate on each sublayer's output.
    def __init__(self, dim, num_heads, ffn_mult, axis, rope_fraction, gate_init=0.1):
        super().__init__()
        if axis not in {"space", "time"}:
            raise ValueError(f"unknown predictor axis {axis}")
        self.axis = axis
        self.dim = dim
        # v186: le-wm ConditionalBlock (DiT AdaLN-zero) with two changes vs v184.
        # (1) DOUBLE-norm: the outer pre-norms here stay affine-free (eps 1e-6); the inner
        #     affine LayerNorms live in self.attn.norm and self.mlp_norm (le-wm fidelity).
        #     Modulate sits BETWEEN the outer affine-free norm and the inner affine norm,
        #     exactly as le-wm's ConditionalBlock -> Attention/FeedForward stack does.
        # (2) WARM-START: v184 zero-init'd the whole adaLN bias, so the gates were EXACTLY 0
        #     at init and the predictor's temporal attention/MLP contributed nothing until
        #     the gates trained up -- a cold start the nGPT v182 (eta ~= 0.05, always active)
        #     never had. v186 keeps the adaLN WEIGHT zero (no action signal at init) but
        #     initialises the GATE bias slices to a small positive constant `gate_init`, so
        #     each sublayer is active (output scaled by ~gate_init) from step 0 while shift /
        #     scale stay 0 (identity modulation). Action influence still ramps in via the
        #     zero-init weight.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if axis == "time":
            self.attn = _RoPEAttention(dim, num_heads, rope_fraction)
        else:
            self.attn = _SpaceAttention(dim, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)  # v186: le-wm inner affine norm (FeedForward[0])
        self.w1 = xavier_linear(nn.Linear(dim, dim * ffn_mult))
        self.w2 = xavier_linear(nn.Linear(dim * ffn_mult, dim))
        # AdaLN-zero: SiLU then a zero-WEIGHT linear to 6*dim = (shift, scale, gate) x2.
        # chunk order is (sh1, sc1, g1, sh2, sc2, g2): gate slices are [2D:3D] and [5D:6D].
        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        with torch.no_grad():
            self.adaLN.bias[2 * dim:3 * dim].fill_(gate_init)  # g1 warm-start
            self.adaLN.bias[5 * dim:6 * dim].fill_(gate_init)  # g2 warm-start

    def _mlp(self, h):
        return self.w2(F.gelu(self.w1(self.mlp_norm(h))))

    def forward(self, x, action_features):
        # x: (B, T, S, D), action_features (conditioning c): (B, T, D)
        batch, time_len, space_len, width = x.shape
        mod = self.adaLN(F.silu(action_features))                   # (B, T, 6D)
        sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, dim=-1)           # each (B, T, D)
        if self.axis == "space":
            y = x.reshape(batch * time_len, space_len, width)

            def bc(p):  # (B, T, D) -> (B*T, 1, D): broadcast over space tokens
                return p.reshape(batch * time_len, 1, width)

            y = y + bc(g1) * self.attn(modulate(self.norm1(y), bc(sh1), bc(sc1)))
            y = y + bc(g2) * self._mlp(modulate(self.norm2(y), bc(sh2), bc(sc2)))
            return y.reshape(batch, time_len, space_len, width)
        y = x.permute(0, 2, 1, 3).contiguous().reshape(batch * space_len, time_len, width)

        def bc(p):  # (B, T, D) -> (B*S, T, D): each space stream shares per-step c
            return p.unsqueeze(1).expand(batch, space_len, time_len, width).reshape(batch * space_len, time_len, width)

        y = y + bc(g1) * self.attn(modulate(self.norm1(y), bc(sh1), bc(sc1)), causal=True)
        y = y + bc(g2) * self._mlp(modulate(self.norm2(y), bc(sh2), bc(sc2)))
        return y.reshape(batch, space_len, time_len, width).permute(0, 2, 1, 3).contiguous()


class TokenProjector(nn.Module):
    """v190: reference-faithful IN-PATH projector (Linear -> BatchNorm1d -> GELU -> Linear).

    In the trading-bot reference encoder the projector sits INSIDE the token path
    (proj -> enrich -> PROJECTOR -> latent_bound): its output, after the tanh bound, IS
    the token that SIGReg, the flow targets, and the trunk all consume. The BN absorbs
    per-dim scaling/whitening cheaply, so SIGReg's push toward N(0, I) does not force the
    upstream encoder map into space-filling contortions (the v187/v188 failure mode).
    Applied per token: the caller folds batch and token axes into the BN batch dim (v186
    JEPAProjector pooling convention). track_running_stats=False (also v186 convention):
    the module always normalizes with batch statistics -- this file never toggles
    .train()/.eval(), and having no running-stat buffers keeps the CUDA-graph-compiled WM
    loss region free of buffer mutations. The thinnest call is the per-env-step rollout
    encode (16 envs x 19 tokens = 304 BN rows per feature -- statistically adequate);
    training/dream/diagnostic calls all fold thousands of rows."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = xavier_linear(nn.Linear(dim, hidden))
        self.bn = nn.BatchNorm1d(hidden, affine=True, track_running_stats=False)
        self.fc2 = xavier_linear(nn.Linear(hidden, dim))

    def forward(self, x):  # x: (N, dim), N = batch * tokens
        return self.fc2(F.gelu(self.bn(self.fc1(x))))


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
        # v179: the 2 outcome tokens are now content-bearing encoder tokens (grounded
        # in the incoming reward/continuation) and join the recurrent/predicted stream.
        self.num_outcome_tokens = 2
        self.num_latent_tokens = self.num_semantic_tokens + self.num_obs_tokens
        self.context = args.lewm_context
        self.mtp_len = args.lewm_mtp_len

        # v179 outcome tokenizer: one learned affine token per outcome scalar
        # (idx0 = symlog incoming reward, idx1 = incoming continuation), SYMMETRIC to
        # the obs tokenizer. Replaces the v152 learned-constant semantic query tokens:
        # the outcome embedding is now grounded in the ACTUAL incoming transition so a
        # detached probe can read it out without back-shaping the representation.
        self.outcome_feature_weight = nn.Parameter(torch.empty(self.num_semantic_tokens, self.dim))
        self.outcome_feature_bias = nn.Parameter(torch.empty(self.num_semantic_tokens, self.dim))
        nn.init.xavier_uniform_(self.outcome_feature_weight)
        nn.init.zeros_(self.outcome_feature_bias)
        # v152 tokenizer: one learned affine token per observation scalar.
        self.obs_feature_weight = nn.Parameter(torch.empty(obs_dim, self.dim))
        self.obs_feature_bias = nn.Parameter(torch.empty(obs_dim, self.dim))
        nn.init.xavier_uniform_(self.obs_feature_weight)
        nn.init.zeros_(self.obs_feature_bias)
        self.embed_norm = nn.Identity()
        self.encoder_layers = nn.ModuleList(
            [LeWMTransformerBlock(self.dim, args.lewm_heads, args.lewm_ffn_mult) for _ in range(args.lewm_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(self.dim)  # v184: le-wm Transformer final norm
        self.outcome_target_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        self.obs_target_proj = xavier_linear(nn.Linear(self.dim, self.dim))
        # v179: DETACHED scalar probes. token 0 -> reward CE, token 1 -> continuation
        # BCE, both read off the (stop-grad) predicted outcome-token belief. The probe
        # is no longer action-conditioned (action conditioning lives in the predictor
        # AdaLN), so reward_action_proj is dropped.
        # v179.2: lejepa probe protocol -- a LayerNorm standardises the (detached) belief
        # token before the linear readout (nn.Sequential(LayerNorm, Linear) in lejepa). The
        # unit-norm token had magnitude ~1; LN gives each dim unit variance so the linear
        # head is well-conditioned. LN sees detached input only -> never touches the rep.
        self.reward_token_ln = nn.LayerNorm(self.dim)
        self.reward_token_unproj = xavier_linear(nn.Linear(self.dim, self.reward_num_bins, bias=False))
        self.continuation_token_ln = nn.LayerNorm(self.dim)
        self.continuation_token_unproj = xavier_linear(nn.Linear(self.dim, 1))

        # Action enters the predictor ONLY as AdaLN conditioning (le-wm fidelity):
        # action_cond embeds the per-timestep action vector into the conditioning
        # space c; there are no action input tokens.
        self.action_cond = xavier_linear(nn.Linear(act_dim, self.dim))
        # v174: PURELY TEMPORAL predictor. All cross-token (space) mixing now lives in
        # the (deeper) encoder, pre-SIGReg; the predictor only propagates per-token
        # dynamics through time. Every block is the time axis (causal RoPE attention).
        # v179: the predictor stream is now the FULL 19-token summary (2 outcome + 17
        # obs); the outcome tokens are predicted purely temporally like obs tokens.
        axes = ["time"] * args.lewm_predictor_layers
        self.predictor_layers = nn.ModuleList(
            [
                AdaLNAxialPredictorBlock(
                    self.dim, args.lewm_heads, args.lewm_ffn_mult, axis, args.lewm_rope_fraction,
                    gate_init=args.lewm_predictor_gate_init,
                )
                for axis in axes
            ]
        )
        self.predictor_norm = nn.LayerNorm(self.dim)  # v184: final predictor norm before belief readout
        # v179: the semantic_readout / predictor_semantic_query aux cross-attention is
        # GONE. Outcome tokens are first-class recurrent tokens, not a downstream readout.
        # v187: the deterministic pred_next_proj is REPLACED by the shortcut flow-matching
        # head (offset-0 next-latent prediction). MTP offsets 1..mtp_len-1 keep their
        # deterministic linear heads: they exist only to feed the detached outcome probes.
        self.flow_head = FlowMatchingHead(self.dim, args)
        self.pred_mtp_projs = nn.ModuleList(
            [xavier_linear(nn.Linear(self.dim, self.dim)) for _ in range(max(0, self.mtp_len - 1))]
        )
        # v187: 10% noisy context (dreamer4/trading-bot): every predictor-trunk input token
        # is mixed x.lerp(eps, ctx_noise_mix) at BOTH train and inference time, so the
        # belief never assumes a perfectly clean context (closes the dream-time gap where
        # the context is the model's own imperfect generations).
        self.ctx_noise_mix = args.lewm_flow_ctx_noise
        self.latent_bound_scale = args.lewm_flow_latent_bound
        self.flow_sample_steps = args.lewm_flow_sample_steps
        self.dream_temperature = args.lewm_flow_dream_temperature
        # v190: the projector (Linear->BatchNorm->GELU->Linear) is BACK, but IN-PATH
        # (v187 deleted v183/v186's loss-space JEPAProjector while moving SIGReg onto the
        # raw tokens -- the bug v190 fixes). It runs inside encode_summary right before
        # the tanh bound, so its output feeds EVERYTHING: flow targets, SIGReg, trunk
        # inputs, and the policy path. See TokenProjector.
        self.token_projector = TokenProjector(self.dim, args.lewm_projector_hidden)
        # v184: normalize_world_model_matrices() is GONE. The world model is a standard
        # (non-nGPT) transformer; AdamW weight decay (lewm_weight_decay=1e-3, le-wm) now
        # bounds the weights instead of a per-step unit-norm projection.

    def encode_summary(self, obs, incoming_outcome):
        # v179: the encoder is a function of obs AND the incoming outcome. The 2 outcome
        # tokens are tokenized (affine, symmetric to obs) from the actual incoming
        # transition outcome -- incoming_outcome: (B, 2) = [symlog(r_in), c_in] -- so
        # the embedding already carries reward/continuation info for a detached probe to
        # read out. JEPA (non-detached target) + SIGReg keep it predictable & high-variance.
        batch = obs.shape[0]
        obs_flat = obs.reshape(batch, -1)
        obs_tokens = obs_flat.unsqueeze(-1) * self.obs_feature_weight + self.obs_feature_bias  # (B,17,D)
        outcome_tokens = (
            incoming_outcome.unsqueeze(-1) * self.outcome_feature_weight + self.outcome_feature_bias
        )  # (B,2,D)
        tokens = torch.cat([outcome_tokens, obs_tokens], dim=1)  # (B,19,D)
        # v184: no unit-norm. embed_norm is Identity (the first block pre-LNs the input);
        # encoder_norm is the le-wm final LayerNorm; target projections are raw Linears.
        tokens = self.embed_norm(tokens)
        for layer in self.encoder_layers:
            tokens = layer(tokens, causal=False)
        tokens = self.encoder_norm(tokens)
        outcome_tokens = self.outcome_target_proj(tokens[:, : self.num_semantic_tokens])
        obs_tokens = self.obs_target_proj(tokens[:, self.obs_token_start :])
        summary = torch.cat([outcome_tokens, obs_tokens], dim=1)  # (B,19,D)
        # v190: IN-PATH projector (fc -> BN -> GELU -> fc), reference-faithful. Batch and
        # token axes are folded into the BatchNorm batch dim; the POST-projector bounded
        # outputs are the tokens everything downstream consumes.
        b, s, d = summary.shape
        summary = self.token_projector(summary.reshape(b * s, d)).reshape(b, s, d)
        # v187: tanh latent bound -- summaries (the flow targets AND the recurrent stream)
        # live in (-B, B), the scale the flow's N(0, I) prior interpolates against.
        return latent_bound(summary, self.latent_bound_scale)

    @torch.no_grad()
    def obs_bias_diagnostics(self, obs):
        obs_flat = obs.reshape(-1, self.obs_dim)
        signal = obs_flat.unsqueeze(-1) * self.obs_feature_weight
        bias = self.obs_feature_bias.unsqueeze(0).expand_as(signal)
        signal_norm = signal.norm(dim=-1)
        bias_norm = bias.norm(dim=-1)
        pretoken_norm = (signal + bias).norm(dim=-1)
        return {
            "obs_token_signal_norm": signal_norm.mean(),
            "obs_token_bias_norm": bias_norm.mean(),
            "obs_token_signal_to_bias": (signal_norm / bias_norm.clamp_min(1e-8)).mean(),
            "obs_token_bias_fraction": (bias_norm / (signal_norm + bias_norm).clamp_min(1e-8)).mean(),
            "obs_token_pretoken_norm": pretoken_norm.mean(),
        }

    def decode_belief_outcomes(self, belief, detach_belief=True):
        # v179: DETACHED linear probe. The belief is stop-grad by default so the CE/BCE
        # outcome losses train ONLY the probe heads -- the outcome-token representation is
        # shaped purely by JEPA (predict non-detached embed) + SIGReg. No action term:
        # action conditioning already lives in the predictor AdaLN.
        if detach_belief:
            belief = belief.detach()
        # v179.2: LayerNorm (lejepa probe protocol) replaces unit_norm; standardises each
        # dim before the linear readout. Detached input => LN trains with the probe only.
        reward_feat = self.reward_token_ln(belief[..., self.reward_token_idx, :])
        reward_logits = self.reward_token_unproj(reward_feat)
        continuation_feat = self.continuation_token_ln(belief[..., self.continuation_token_idx, :])
        continuation_logits = self.continuation_token_unproj(continuation_feat).squeeze(-1)
        return reward_logits, continuation_logits

    def _predictor_trunk(self, latent_history, action_history, ctx_noise=None, mix_ctx=True):
        # v179: PURELY TEMPORAL predictor over the FULL 19-token recurrent stream
        # (B, T, num_latent_tokens, dim) = [2 outcome, 17 obs]. Every token (outcome and
        # obs) is propagated per token across time (cross-token mixing lives in the
        # encoder, pre-SIGReg). The action is injected as AdaLN conditioning (c), never
        # as an input token. There is no longer a separate semantic readout.
        batch, context_len, num_tok, width = latent_history.shape
        if context_len > self.context:
            latent_history = latent_history[:, -self.context :]
            action_history = action_history[:, -self.context :]
            if ctx_noise is not None:
                ctx_noise = ctx_noise[:, -self.context :]
            context_len = self.context
        # v187: 10% noisy context on the trunk INPUT tokens. The compiled WM loss passes
        # ctx_noise explicitly (its region must stay randomness-free); eager noisy
        # callers (dream DYNAMICS, warm-AR / flow-eval diagnostics) draw fresh noise.
        # mix_ctx=False callers (v192): the self-conditioning pre-pass, and every belief
        # the ACTOR/CRITIC consumes (rollout agent input, value-bootstrap encodes, dream
        # agent inputs) -- deterministic, so ctx-noise jitter never enters GAE deltas.
        if mix_ctx and self.ctx_noise_mix > 0.0:
            noise = ctx_noise if ctx_noise is not None else torch.randn_like(latent_history)
            latent_history = latent_history.lerp(noise, self.ctx_noise_mix)
        # v184: no unit-norm. The predictor blocks pre-LN their residual stream, so the raw
        # latent history (encoder targets at train time, flow-generated summaries at imagine
        # time -- both bounded, ~unit-variance) feeds straight in. The action
        # conditioning is raw (the AdaLN block applies SiLU); predictor_norm is the le-wm
        # final LayerNorm before the belief readout.
        tokens = latent_history
        action_features = self.action_cond(action_history)
        for layer in self.predictor_layers:
            tokens = layer(tokens, action_features)
        belief = self.predictor_norm(tokens)  # (B, T, 19, dim)
        return belief

    def belief_from_history(self, latent_history, action_history, mix_ctx=True):
        # v192: mix_ctx passthrough. Policy-input / value-bootstrap callers pass
        # mix_ctx=False so the beliefs the actor/critic consume (and the GAE deltas
        # built from them) are DETERMINISTIC functions of state; WM training and dream
        # DYNAMICS keep the default noisy context (the regime the trunk trains in).
        return self._predictor_trunk(latent_history, action_history, mix_ctx=mix_ctx)[:, -1]

    def flow_mean_next(self, belief):
        # TEMP-0 CONDITIONAL MEAN: one-shot x-prediction at (z=0, signal=0). Trained at
        # full uniform weight by the grounding term (plus the explicit mean-path anchor
        # rows), so this single forward IS the conditional mean of the next-summary
        # distribution. Deterministic, one MLP pass: policy-input / probe / diagnostics.
        zeros = torch.zeros(belief.shape[:-1], dtype=torch.long, device=belief.device)
        return self.flow_head(torch.zeros_like(belief), zeros, belief)

    def flow_sample_next(self, belief, temperature=1.0, num_steps=None):
        # v188 sampled path: z0 ~ N(0, I) * temperature, then a deterministic PLAIN Euler
        # ODE over the equally spaced signal grid (d = K_MAX / num_steps; signals 0, d,
        # 2d, ...). x-parameterization: v = (x_pred - z) / (1 - t). The final step lands
        # exactly on the (bounded) x_pred. The init draw is the ONLY stochasticity
        # (dreamer4: no inter-step noise injection).
        if temperature <= 0.0:
            return self.flow_mean_next(belief)
        if num_steps is None:
            num_steps = self.flow_sample_steps
        k_max = self.flow_head.k_max
        assert (
            1 <= num_steps <= k_max and k_max % num_steps == 0
        ), f"num_steps must divide K_MAX={k_max}, got {num_steps}"
        d = k_max // num_steps
        lead_shape = belief.shape[:-1]
        z = torch.randn_like(belief) * temperature
        for step in range(num_steps):
            sig_val = step * d
            signal = torch.full(lead_shape, sig_val, dtype=torch.long, device=belief.device)
            x_pred = self.flow_head(z, signal, belief)
            t = sig_val / k_max
            z = z + (x_pred - z) / (1.0 - t) * (d / k_max)
        return z

    def predict_mtp_from_history(self, latent_history, action_history, ctx_noise=None):
        # v187: offset 0 is the flow head's one-shot conditional mean (feeds the offset-0
        # probe and diagnostics; the flow LOSS reuses the returned belief features
        # directly). Offsets 1..mtp_len-1 stay deterministic linear probes-feeders.
        features = self._predictor_trunk(latent_history, action_history, ctx_noise=ctx_noise)
        preds = [self.flow_mean_next(features)]
        preds.extend(proj(features) for proj in self.pred_mtp_projs)
        preds = torch.stack(preds, dim=2)
        return preds, features

    def imagine_step_from_history(self, latent_history, action_history, temperature=None):
        # History-conditioned one-step imagination. Mirrors the teacher-forced
        # training path: the predictor sees the accumulated (latent, action) context and
        # generates the next summary at the last position. action_history[:, -1] is the
        # action just taken from the current state, which conditions both next-state and
        # outcome prediction. _predictor_trunk auto-truncates to self.context.
        # v188: generation is a flow sample. temperature=None -> the configured dream
        # temperature (default 1.0: sampled 8-step plain Euler); 0.0 -> one-shot
        # conditional mean (deterministic; used by the labeled AR diagnostics).
        if temperature is None:
            temperature = self.dream_temperature
        belief = self._predictor_trunk(latent_history, action_history)[:, -1]   # (B,19,D)
        next_summary = self.flow_sample_next(belief, temperature=temperature)   # (B,19,D) bounded
        # The outcome tokens [0,1] of next_summary are the predicted outcome embeddings;
        # the DETACHED probe reads reward/continuation scalars off them.
        reward_logits, continuation_logits = self.decode_belief_outcomes(next_summary, detach_belief=True)
        return next_summary, reward_logits, continuation_logits


class BeliefConditioner(nn.Module):
    # v173: FiLM-condition the belief on the current-obs embed. Both are unit-norm
    # token stacks (B, T, dim). A per-feature (scale, shift) read off the obs embed
    # modulates the belief: fused = unit_norm(belief*(1+gamma) + beta). to_mod is
    # ZERO-INITIALIZED so at init gamma=beta=0 => fused == belief (identity start);
    # the obs-grounding correction is learned from zero, so no regression vs belief-only.
    def __init__(self, dim):
        super().__init__()
        self.to_mod = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, belief, obs_embed):
        gamma, beta = self.to_mod(obs_embed).chunk(2, dim=-1)
        # v184: no unit-norm (FiLM path, only live under obs_cond_mode=="film"; the agent's
        # downstream agent_input_ln LayerNorm standardizes the fused vector regardless).
        return belief * (1.0 + gamma) + beta


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.detach_world_model_from_agent = args.detach_world_model_from_agent
        self.share_backbone = args.share_backbone
        # Direct latent control state: reward/continuation CLS tokens + one token per obs scalar.
        self.agent_latent_dim = (2 + obs_dim) * args.lewm_dim
        # v173: how the current-obs embed is fused with the belief.
        #   agent_input_dim = BUFFER width (what get_*_from_agent_input receives): 2x when
        #     both latents are stored (film/concat), 1x for belief-only (none).
        #   trunk_in_dim    = TRUNK input width after fusion: 2x only for raw concat;
        #     film fuses down to 1x, none is already 1x.
        self.obs_cond_mode = args.obs_cond_mode
        assert self.obs_cond_mode in ("film", "concat", "none")
        self._buffer_both = self.obs_cond_mode in ("film", "concat")
        self.agent_input_dim = self.agent_latent_dim * (2 if self._buffer_both else 1)
        trunk_in_dim = self.agent_latent_dim * (2 if self.obs_cond_mode == "concat" else 1)
        # v182: the agent acts on the WM's DETACHED PREDICTED next-summary embed
        # (v187: the flow head's one-shot conditional mean) rather than the raw belief. This
        # per-token LayerNorm standardises that predicted embed at the (grad-enabled)
        # update forward -- lejepa probe protocol -- so the LN trains with the agent
        # while the predicted embed stays stop-grad w.r.t. the world model.
        self.agent_input_ln = nn.LayerNorm(args.lewm_dim)
        if self.obs_cond_mode == "film":
            self.belief_conditioner = BeliefConditioner(args.lewm_dim)
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(trunk_in_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(trunk_in_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(trunk_in_dim, H, args.k_blocks, args.n_experts)
        # Categorical critic, HL-Gauss MTP head:
        # outputs critic_mtp_horizon * num_bins logits; horizon 0 is V(s_t), later
        # horizons are critic-only MTP predictions of returns[t+h]. v149 removes
        # the logit bias: symmetric supports make zero logits neutral, and the
        # critic must not carry a hidden scalar prior in its distribution head.
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
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
        assert self.agent_latent_dim == self.world_model.num_latent_tokens * self.world_model.dim
        torch.random.set_rng_state(main_rng_state)

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

    def agent_input_from_latent(self, latent):
        if self.detach_world_model_from_agent:
            latent = latent.detach()
        return latent.reshape(latent.shape[0], -1)

    def combine_agent_input(self, obs_embed_latent, belief_latent):
        # v173: BUFFER representation handed to get_*_from_agent_input. For film/concat
        # we store both latents (obs first, belief second) and fuse later in
        # _fuse_agent_input; for "none" we store the belief alone (v171). Both components
        # are flattened+detached identically via agent_input_from_latent.
        # v182: project the belief into the WM's PREDICTED next-summary embed BEFORE
        # buffering, so the agent acts on the prediction, not the raw belief. v187: the
        # prediction is the flow head's TEMP-0 CONDITIONAL MEAN (one-shot x-pred at z=0,
        # signal=0, d=1) -- deterministic and cheap, so the buffered agent input stays a
        # single fixed vector per state (cudagraph-captured shapes unchanged). This runs
        # at rollout/dream buffer time; agent_input_from_latent then .detach()es it, so
        # the flow head's WM params never receive agent gradient.
        # NOTE: deliberately NO unit_norm here. The agent-side agent_input_ln
        # (LayerNorm, applied at the update forward) is scale-invariant --
        # LN(c*x) == LN(x) for c>0 -- so a preceding unit_norm (a positive per-token
        # rescale) is a no-op it immediately erases. One trained norm suffices; this
        # mirrors the v179.2 probe protocol ("LayerNorm replaces unit_norm"). v184: the WM
        # is now a standard transformer, so next_summary is raw too; the agent-input
        # LayerNorm standardizes the (raw) predicted embed before the actor/critic.
        belief_latent = self.world_model.flow_mean_next(belief_latent)
        belief_in = self.agent_input_from_latent(belief_latent)
        if not self._buffer_both:
            return belief_in
        return torch.cat([self.agent_input_from_latent(obs_embed_latent), belief_in], dim=-1)

    def _fuse_agent_input(self, agent_input):
        # v173: turn the BUFFER representation into the TRUNK input. "concat"/"none"
        # pass through unchanged; "film" splits the stored [obs_embed | belief] halves,
        # FiLM-conditions the belief on the obs embed (params train here, inside the
        # update forward), and flattens back to one agent_latent_dim vector.
        if self.obs_cond_mode != "film":
            return agent_input
        D = self.world_model.dim
        T = self.world_model.num_latent_tokens
        obs_flat, belief_flat = agent_input.chunk(2, dim=-1)
        obs_embed = obs_flat.reshape(obs_flat.shape[0], T, D)
        belief = belief_flat.reshape(belief_flat.shape[0], T, D)
        fused = self.belief_conditioner(belief, obs_embed)
        return fused.reshape(fused.shape[0], -1)

    def agent_input_from_obs(self, x):
        # Standalone single-obs path (no recurrent history): build a length-1 belief
        # from this obs + a neutral action, mirroring build_belief_agent_input with an
        # empty history, then concat the raw summary. Used by the convenience
        # get_value/get_action_and_value(x) entry points.
        # Standalone single-obs convenience path: no incoming transition is available,
        # so ground the outcome tokens with zeros (episode-start convention). The FULL
        # 19-token summary is the recurrent state (v179: outcome tokens are recurrent).
        incoming_outcome = x.new_zeros(x.shape[0], self.world_model.num_semantic_tokens)
        latent = self.world_model.encode_summary(x, incoming_outcome)
        latent_hist = latent.unsqueeze(1)
        neutral = x.new_zeros(x.shape[0], 1, self.act_dim)
        # v192: policy-input path -> clean context (matches build_belief_agent_input).
        belief = self.world_model.belief_from_history(latent_hist, neutral, mix_ctx=False)
        return self.combine_agent_input(latent, belief)

    def _trunks_from_agent_input(self, obs, agent_input):
        if self.detach_world_model_from_agent:
            agent_input = agent_input.detach()
        # v173: fuse the buffered representation into the trunk input here, INSIDE the
        # (compiled) forward, so the FiLM conditioner trains on the replayed buffer leaf.
        agent_input = self._fuse_agent_input(agent_input)
        # v182: LayerNorm the buffered (detached) predicted-embed tokens per token (dim
        # D) here in the grad-enabled update forward, so the LN trains with the agent.
        # In beliefonly/film the fused width is num_latent_tokens*D, so reshape->LN->flat.
        wm = self.world_model
        bsz = agent_input.shape[0]
        agent_input = self.agent_input_ln(
            agent_input.reshape(bsz, wm.num_latent_tokens, wm.dim)
        ).reshape(bsz, -1)
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(agent_input)
            return feat, feat
        return self.actor_trunk(agent_input), self.critic_trunk(agent_input)

    def _trunks(self, x):
        return self._trunks_from_agent_input(x, self.agent_input_from_obs(x))

    def get_value(self, x):
        # Returns value LOGITS (B, mtp, num_bins); horizon 0 is V(s_t), later
        # horizons are critic-only MTP predictions. Caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_value_from_agent_input(self, obs, agent_input):
        _, critic_feat = self._trunks_from_agent_input(obs, agent_input)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        return self.get_action_and_value_from_agent_input(x, self.agent_input_from_obs(x), z)

    def get_action_and_value_from_agent_input(self, obs, agent_input, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks_from_agent_input(obs, agent_input)
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
        return action, z, log_prob, entropy, value_logits

    def _fusion_params(self):
        # v173: the FiLM belief-conditioner is a SHARED pre-trunk module (it feeds both
        # the actor and critic trunks), so — exactly like the shared backbone — it
        # belongs in BOTH the actor and critic clip groups. Under the default
        # share_backbone=True it is NOT in critic_only_params (=critic_head only), so it
        # freezes on critic-only epochs like the shared trunk. (With share_backbone=False
        # critic_only_params==critic_params, so — like each separate trunk — it also
        # takes the value grad on critic-only epochs; consistent with that config.)
        # v182: agent_input_ln is likewise a SHARED pre-trunk module (it standardises
        # the predicted-embed input feeding both trunks), so it belongs in BOTH clip
        # groups exactly like the FiLM conditioner -- otherwise the decoupled dual
        # backward would drop its value grad and apply its policy grad unclipped.
        params = list(self.agent_input_ln.parameters())
        if self.obs_cond_mode == "film":
            params += list(self.belief_conditioner.parameters())
        return params

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
        return list(trunk.parameters()) + heads + self._fusion_params()

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters()) + self._fusion_params()


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
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    # EMA (Dreamer4) and per-minibatch (CleanRL) advantage norm are alternatives; both
    # at once would double-normalize (whiten to unit std, then divide by return std).
    assert not (args.norm_adv and args.ret_percnorm), \
        "use either norm_adv (CleanRL per-minibatch) or ret_percnorm (percentile range), not both"
    assert args.ret_perc_scope in ("minibatch", "ema", "batch")
    assert args.lewm_dyn_horizon <= args.lewm_context, "v73 MTP expects dyn_horizon <= lewm_context"
    assert args.lewm_mtp_len <= args.lewm_dyn_horizon, "v73 MTP expects mtp_len <= dyn_horizon"
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
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    configure_torch_compile(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    # v146: AdamW keeps PPO/non-WM behavior unchanged while WM decay is disabled;
    # WM representation matrices are projected explicitly after optimizer steps.
    world_model_params = list(agent.world_model.parameters())
    wm_param_ids = {id(p) for p in world_model_params}
    non_wm_params = [p for p in agent.parameters() if id(p) not in wm_param_ids]
    # v179.2: split the WM params so the DETACHED probe heads (reward/continuation readout +
    # their LayerNorms) get a dedicated higher-LR optimizer group. The continuation target is
    # a constant logit(gamma)=5.29 on HalfCheetah and the SIGReg'd belief token is mean-zero,
    # so the head must WALK that distance at ~lr/step; base lr*anneal over ~15.6k WM steps only
    # covers ~2.3 logits. parameter-golf uses HEAD_LR=0.008 for its readout heads; mirror it
    # with lewm_probe_lr_mult x base. Per-group proportional anneal (below) keeps the higher base.
    wm_outcome_io_params = (
        list(agent.world_model.reward_token_ln.parameters())
        + list(agent.world_model.reward_token_unproj.parameters())
        + list(agent.world_model.continuation_token_ln.parameters())
        + list(agent.world_model.continuation_token_unproj.parameters())
    )
    wm_outcome_io_param_ids = {id(p) for p in wm_outcome_io_params}
    wm_latent_params = [p for p in world_model_params if id(p) not in wm_outcome_io_param_ids]
    optimizer = optim.AdamW(
        [
            {"params": non_wm_params, "weight_decay": 0.0},
            # v183: align with le-wm -- AdamW wd=1e-3 on all WM params (latent stream + the
            # detached outcome probe IO + the new JEPA projector). PPO (non_wm) stays at 0.0.
            {"params": wm_latent_params, "weight_decay": args.lewm_weight_decay},
            {"params": wm_outcome_io_params, "weight_decay": args.lewm_weight_decay,
             "lr": args.learning_rate * args.lewm_probe_lr_mult},
        ],
        lr=args.learning_rate,
        eps=1e-5,
    )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]  # v179.2: per-group base lr for proportional anneal
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH actor/critic lists for epochs where both losses update.
    # Critic-only epochs use critic_only_params so the shared policy trunk is not
    # moved after the single actor epoch.
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    critic_only_params = (
        list(agent.critic_head.parameters()) if agent.share_backbone else critic_params
    )
    critic_only_param_ids = {id(p) for p in critic_only_params}

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

    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,  # HL-Gauss projection sigma (D4 scalar-return target)
        device,
    )
    support = hl_support.support                       # (num_bins,) raw Dreamer3 bucket centers

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    def value_probs_to_scalar(probs):
        return hl_support.probs_to_scalar(probs)

    bin_width = hl_support.bin_width

    reward_support = Dreamer3BucketHLGaussSupport(
        args.lewm_reward_num_bins,
        args.lewm_reward_v_min,
        args.lewm_reward_v_max,
        0.5,
        device,
    )
    def reward_logits_to_scalar(logits):
        return reward_support.to_scalar(logits)
    reward_raw_edge_min = reward_support.edges[0]
    reward_raw_edge_max = reward_support.edges[-1]

    with torch.no_grad():
        agent.critic_head.weight.zero_()
        agent.world_model.reward_token_unproj.weight.zero_()
    compile_agent_callables(agent, args)
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
        # v175: joint SIGReg over the flattened (S*D) frame vector (see static variant).
        A = sigreg.sample_projection(
            token_latents.shape[-2] * token_latents.shape[-1],
            token_latents.device, token_latents.dtype,
            generator=wm_torch_generator,
        )
        step_losses = []
        for h in range(horizon):
            valid_mask = token_valids[:, h].bool()
            if int(valid_mask.sum().item()) < args.lewm_sigreg_min_valid:
                continue
            # (B, S, D) -> (valid_b, S, D) -> (1, valid_b, S*D): one joint "token" per
            # frame, batch as the sample axis. N = proj.size(-2) = valid_b, faithful to
            # le-wm (statistic scaled by batch size; steps with more valid samples get
            # proportionally more weight).
            valid_b = int(valid_mask.sum().item())
            step_flat = token_latents[:, h][valid_mask].reshape(valid_b, -1).unsqueeze(0)
            step_losses.append(sigreg(step_flat, A=A))
        if not step_losses:
            return token_latents.sum() * 0.0
        return torch.stack(step_losses).mean()

    def _sigreg_group_static(token_latents, token_valids, A):
        # Core: JOINT (flattened) SIGReg over a token GROUP. Per rollout step, flatten
        # the group's K token slots into one (K*D) per-frame vector and whiten its JOINT
        # batch-marginal (le-wm "whiten the whole per-frame latent"; here D_frame=K*D).
        # Random projections mix dims ACROSS the group's tokens, so cross-token
        # correlation WITHIN the group is penalized. Static-shape / tensor-masked for
        # CUDA-graph capture; per-step batch marginal preserved, averaged over steps.
        # token_latents: (B, H, K, D), token_valids: (B, H), A: (K*D, P)
        step_losses = []
        t = sigreg.t.to(device=token_latents.device, dtype=token_latents.dtype)
        phi = sigreg.phi.to(device=token_latents.device, dtype=token_latents.dtype)
        weights = sigreg.weights.to(device=token_latents.device, dtype=token_latents.dtype)
        batch = token_latents.shape[0]
        for h in range(token_latents.shape[1]):
            step_flat = token_latents[:, h].reshape(batch, -1)  # (B, K*D)
            valid = token_valids[:, h].to(dtype=token_latents.dtype)
            valid_count = valid.sum().clamp_min(1.0)
            valid_view = valid.view(-1, 1, 1)                   # (B, 1, 1)
            x_t = (step_flat @ A).unsqueeze(-1) * t             # (B, P, knots)
            cos_mean = (x_t.cos() * valid_view).sum(dim=0) / valid_count  # (P, knots)
            sin_mean = (x_t.sin() * valid_view).sum(dim=0) / valid_count
            err = (cos_mean - phi).square() + sin_mean.square()
            statistic = (err @ weights) * valid_count
            step_loss = statistic.mean()
            active = (valid.sum() >= args.lewm_sigreg_min_valid).to(dtype=token_latents.dtype)
            step_losses.append(step_loss * active)
        active_count = (
            (token_valids.sum(dim=0) >= args.lewm_sigreg_min_valid)
            .to(dtype=token_latents.dtype)
            .sum()
            .clamp_min(1.0)
        )
        return torch.stack(step_losses).sum() / active_count

    def masked_token_sigreg_static(token_latents, token_valids, A_obs, A_out):
        # v191: PER-TOKEN SIGReg for the obs tokens, replacing v175/v179.1's JOINT
        # flattened (K*D) frame whitening -- the confirmed v187/v188/v190 failure mode
        # (joint isotropy demands mutual decorrelation of 17 tokens encoding the same
        # 17-dim obs, forcing a space-filling encoder; effrank_obs_joint 14 -> ~200).
        # Each obs token's D-dim marginal is pushed toward N(0,I) with the TOKEN AXIS
        # FOLDED INTO THE SAMPLE/BATCH axis: per rollout step the obs group contributes
        # valid_b*K 64-dim samples to ONE ECF statistic, mirroring how each outcome token
        # already contributes valid_b samples via A_out. Cross-token correlation is now
        # unconstrained. Outcome tokens ([:obs_token_start]) are UNCHANGED: each whitened
        # separately over its own D dims (A_out, v179.1 rationale -- reward/continuation
        # are genuine functions of state, so they are not cross-whitened with obs either).
        # Scaling (verified numerically): the Epps-Pulley statistic (valid_count *
        # mean-sq ECF error) is O(1) under H0 regardless of sample count and grows
        # ~N*delta^2 under a fixed deviation, so the folded call (N = valid_b*K) exerts
        # the SAME per-token deviation pressure as K separate per-slot calls (the outcome
        # pathway) while computing one pooled statistic; v179.1's external
        # num_obs_tokens multiplier is therefore dropped. Net vs v190: same
        # deviation-regime magnitude, lower constant null floor (~(1+n_out)/19 instead of
        # ~(K+n_out)/19 -- a sampling-noise offset with no systematic gradient), and the
        # POOLED marginal (mixture over the 17 tokens) replaces the joint frame
        # distribution as SIGReg's target (reference positions-batched semantics).
        # token_latents: (B, H, S, D); A_obs: (D, P); A_out: (D, P).
        os = agent.world_model.obs_token_start
        obs_tokens = token_latents[:, :, os:, :]  # (B, H, K, D)
        b, h, k, d = obs_tokens.shape
        obs_folded = obs_tokens.permute(0, 2, 1, 3).reshape(b * k, h, 1, d)
        obs_valids = token_valids.unsqueeze(1).expand(b, k, h).reshape(b * k, h)
        obs_loss = _sigreg_group_static(obs_folded, obs_valids, A_obs)
        out_losses = torch.stack(
            [_sigreg_group_static(token_latents[:, :, i : i + 1, :], token_valids, A_out) for i in range(os)]
        )
        return (obs_loss + out_losses.sum()) / agent.world_model.num_latent_tokens

    wm_supervised_encode_summary = getattr(
        agent.world_model, "_eager_encode_summary", agent.world_model.encode_summary
    )
    wm_supervised_predict_mtp_from_history = getattr(
        agent.world_model,
        "_eager_predict_mtp_from_history",
        agent.world_model.predict_mtp_from_history,
    )

    def wm_supervised_loss_forward(
        obs_anchor,
        future_next_obs_flat,
        incoming_outcomes,
        future_actions,
        future_rewards,
        future_discounts,
        latent_weight,
        reward_target_probs,
        pred_outcome_actions,
        sigreg_weight,
        sigreg_A_obs,
        sigreg_A_out,
        flow_eps,
        flow_signal,
        flow_selfcond_mask,
        flow_ctx_noise,
    ):
        # le-wm-style training wrapper: all differentiable WM supervised pieces
        # live in one callable before backward.
        # v179: incoming_outcomes (mb, H+1, 2) grounds the per-frame outcome tokens.
        # v187/v188: ALL flow randomness is sampled OUTSIDE this (compiled) region and
        # passed in as tensors (same pattern as the SIGReg projections): flow_eps
        # (mb,H,S,D) standard normal, flow_signal (mb,H) long ~ U{0..K_MAX-1} (unsnapped;
        # plain flow), flow_selfcond_mask (mb,H) bool ~ Bernoulli(p_selfcond),
        # flow_ctx_noise (mb,H,S,D) standard normal for the 10% noisy-context mix on the
        # training trunk pass.
        mb_size = obs_anchor.shape[0]
        horizon = future_actions.shape[1]
        mtp_len = pred_outcome_actions.shape[2]
        obs_sequence = torch.cat(
            [
                obs_anchor.unsqueeze(1),
                future_next_obs_flat.reshape((mb_size, horizon) + obs_anchor.shape[1:]),
            ],
            dim=1,
        )
        encoded_sequence = wm_supervised_encode_summary(
            obs_sequence.reshape((-1,) + obs_anchor.shape[1:]),
            incoming_outcomes.reshape(-1, agent.world_model.num_semantic_tokens),
        ).reshape(
            mb_size,
            horizon + 1,
            agent.world_model.num_latent_tokens,
            agent.world_model.dim,
        )
        # v179: the recurrent/predicted/JEPA stream is the FULL 19-token summary.
        initial_summary = encoded_sequence[:, 0]
        target_summaries = encoded_sequence[:, 1:]
        teacher_history = torch.cat(
            [
                initial_summary.unsqueeze(1),
                target_summaries[:, :-1],
            ],
            dim=1,
        )
        # v187: SELF-CONDITIONING pre-pass (p=0.25 scheduled sampling). Under no_grad, a
        # CLEAN-context trunk pass + one-shot conditional-mean flow prediction gives the
        # model's own prediction of every target; shifted right one step, it REPLACES the
        # teacher-forced context token where flow_selfcond_mask is set. Position 0 is the
        # real anchor by construction (shifted_pred[:, 0] == teacher_history[:, 0]). The
        # replaced tokens are detached; real tokens keep their encoder gradient.
        with torch.no_grad():
            pre_belief = agent.world_model._predictor_trunk(
                teacher_history, future_actions, mix_ctx=False
            )
            pre_pred = agent.world_model.flow_mean_next(pre_belief)          # (mb,H,S,D)
        shifted_pred = torch.cat([teacher_history[:, :1], pre_pred[:, :-1].detach()], dim=1)
        history_sc = torch.where(
            flow_selfcond_mask[:, :, None, None], shifted_pred, teacher_history
        )
        pred_mtp, belief_features = wm_supervised_predict_mtp_from_history(
            history_sc, future_actions, ctx_noise=flow_ctx_noise
        )

        one_step_denom = latent_weight.sum().clamp_min(1.0)
        # v188 PLAIN CONDITIONAL FLOW-MATCHING loss. Targets are the clean bounded
        # encoder summaries, ATTACHED (encoder gradient flows through both the x-target
        # and the noised input, le-wm/JEPA style). Diffusion forcing: signal level
        # independent per (row, timestep), shared across the frame's tokens. GROUNDING:
        # x-space MSE at FULL UNIFORM weight over signal levels, on every row. No
        # shortcut self-consistency / step-size machinery (v3-plainflow ablation).
        flow_head = agent.world_model.flow_head
        k_max = flow_head.k_max
        num_tok = agent.world_model.num_latent_tokens
        targets = target_summaries                                            # (mb,H,S,D)
        t1 = (flow_signal.to(targets.dtype) / k_max)[:, :, None, None]        # (mb,H,1,1)
        z_t = (1.0 - t1) * flow_eps + t1 * targets
        sig_tok = flow_signal[:, :, None].expand(-1, -1, num_tok)
        x_pred = flow_head(z_t, sig_tok, belief_features)
        grounding = F.mse_loss(x_pred, targets, reduction="none").mean(dim=(-1, -2))  # (mb,H)
        one_step_latent_loss = grounding
        wm_latent_loss = (one_step_latent_loss * latent_weight).sum() / one_step_denom
        wm_latent_mse = (grounding * latent_weight).sum() / one_step_denom

        # v179: DETACHED probe -- pred_mtp is stop-grad inside decode, so CE/BCE reach
        # only the probe heads, never the encoder/predictor. v187: offset 0 of pred_mtp
        # is the flow head's detached one-shot conditional mean.
        pred_reward_logits_all, pred_continuation_logits_all = agent.world_model.decode_belief_outcomes(
            pred_mtp,
            detach_belief=True,
        )

        reward_loss_sum = pred_mtp.new_zeros(())
        discount_bce_sum = pred_mtp.new_zeros(())
        discount_entropy_sum = pred_mtp.new_zeros(())
        reward_abs_error_sums = []
        reward_error_sums = []
        reward_pred_edge_sums = []
        reward_target_edge_sums = []
        reward_probe_weights = []
        for mtp_idx in range(mtp_len):
            valid_horizon = horizon - mtp_idx
            offset_valid = latent_weight[:, mtp_idx:]
            denom = offset_valid.sum().clamp_min(1.0)
            reward_loss = -(
                reward_target_probs[:, mtp_idx:].detach()
                * torch.log_softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
            ).sum(dim=-1)
            reward_loss_sum = reward_loss_sum + (reward_loss * offset_valid).sum() / denom

            discount_target = future_discounts[:, mtp_idx:]
            discount_bce = F.binary_cross_entropy_with_logits(
                pred_continuation_logits_all[:, :valid_horizon, mtp_idx],
                discount_target,
                reduction="none",
            )
            discount_entropy = binary_target_entropy(discount_target)
            discount_bce_sum = discount_bce_sum + (discount_bce * offset_valid).sum() / denom
            discount_entropy_sum = discount_entropy_sum + (discount_entropy * offset_valid).sum() / denom

            pred_reward_scalar = reward_logits_to_scalar(pred_reward_logits_all[:, :valid_horizon, mtp_idx])
            target_reward_scalar = future_rewards[:, mtp_idx:]
            reward_error = pred_reward_scalar - target_reward_scalar
            pred_reward_probs = torch.softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
            pred_edge_mass = pred_reward_probs[..., 0] + pred_reward_probs[..., -1]
            target_edge_frac = (
                (target_reward_scalar <= reward_raw_edge_min)
                | (target_reward_scalar >= reward_raw_edge_max)
            ).to(offset_valid.dtype)
            reward_abs_error_sums.append((reward_error.abs() * offset_valid).sum())
            reward_error_sums.append((reward_error * offset_valid).sum())
            reward_pred_edge_sums.append((pred_edge_mass * offset_valid).sum())
            reward_target_edge_sums.append((target_edge_frac * offset_valid).sum())
            reward_probe_weights.append(denom)

        # v187: SIGReg directly on the CLEAN tanh-bounded encoder summaries (trading-bot
        # recipe; the v183 projector emb is gone). SIGReg pushes the tokens toward
        # isotropic N(0,I) -- the same distribution the flow prior interpolates against --
        # and is the anti-collapse device now that grounding lives in raw token space.
        sigreg_latents = encoded_sequence
        wm_sigreg_loss = masked_token_sigreg_static(sigreg_latents, sigreg_weight, sigreg_A_obs, sigreg_A_out)
        wm_discount_bce = discount_bce_sum
        wm_discount_bce_excess = discount_bce_sum - discount_entropy_sum
        wm_loss = (
            wm_latent_loss
            + args.lewm_reward_loss_coef * reward_loss_sum
            + args.lewm_discount_loss_coef * wm_discount_bce
            + args.lewm_sigreg_coef * wm_sigreg_loss
        )
        return wm_loss

    @torch.no_grad()
    def wm_supervised_metrics_forward(
        obs_anchor,
        future_next_obs_flat,
        incoming_outcomes,
        future_actions,
        future_rewards,
        future_discounts,
        latent_weight,
        reward_target_probs,
        pred_outcome_actions,
        sigreg_weight,
        sigreg_A_obs,
        sigreg_A_out,
        flow_eps,
        flow_signal,
        flow_selfcond_mask,
        flow_ctx_noise,
    ):
        mb_size = obs_anchor.shape[0]
        horizon = future_actions.shape[1]
        mtp_len = pred_outcome_actions.shape[2]
        obs_sequence = torch.cat(
            [
                obs_anchor.unsqueeze(1),
                future_next_obs_flat.reshape((mb_size, horizon) + obs_anchor.shape[1:]),
            ],
            dim=1,
        )
        encoded_sequence = wm_supervised_encode_summary(
            obs_sequence.reshape((-1,) + obs_anchor.shape[1:]),
            incoming_outcomes.reshape(-1, agent.world_model.num_semantic_tokens),
        ).reshape(
            mb_size,
            horizon + 1,
            agent.world_model.num_latent_tokens,
            agent.world_model.dim,
        )
        # v179: full 19-token recurrent/predicted/JEPA stream.
        initial_summary = encoded_sequence[:, 0]
        target_summaries = encoded_sequence[:, 1:]
        teacher_history = torch.cat(
            [
                initial_summary.unsqueeze(1),
                target_summaries[:, :-1],
            ],
            dim=1,
        )
        # v188: mirror supervised_loss -- self-conditioning pre-pass, then the plain
        # flow-matching loss on the same passed-in randomness (already all-no_grad here).
        pre_belief = agent.world_model._predictor_trunk(
            teacher_history, future_actions, mix_ctx=False
        )
        pre_pred = agent.world_model.flow_mean_next(pre_belief)
        shifted_pred = torch.cat([teacher_history[:, :1], pre_pred[:, :-1]], dim=1)
        history_sc = torch.where(
            flow_selfcond_mask[:, :, None, None], shifted_pred, teacher_history
        )
        pred_mtp, belief_features = wm_supervised_predict_mtp_from_history(
            history_sc, future_actions, ctx_noise=flow_ctx_noise
        )
        denom_latent = latent_weight.sum().clamp_min(1.0)
        flow_head = agent.world_model.flow_head
        k_max = flow_head.k_max
        num_tok = agent.world_model.num_latent_tokens
        targets = target_summaries
        t1 = (flow_signal.to(targets.dtype) / k_max)[:, :, None, None]
        z_t = (1.0 - t1) * flow_eps + t1 * targets
        sig_tok = flow_signal[:, :, None].expand(-1, -1, num_tok)
        x_pred = flow_head(z_t, sig_tok, belief_features)
        grounding = F.mse_loss(x_pred, targets, reduction="none").mean(dim=(-1, -2))
        one_step_latent_loss = grounding
        wm_latent_loss = (one_step_latent_loss * latent_weight).sum() / denom_latent
        wm_latent_mse = (grounding * latent_weight).sum() / denom_latent

        pred_reward_logits_all, pred_continuation_logits_all = agent.world_model.decode_belief_outcomes(
            pred_mtp,
            detach_belief=True,
        )
        reward_loss_sum = pred_mtp.new_zeros(())
        discount_bce_sum = pred_mtp.new_zeros(())
        discount_entropy_sum = pred_mtp.new_zeros(())
        reward_abs_error_sums = []
        reward_error_sums = []
        reward_pred_edge_sums = []
        reward_target_edge_sums = []
        reward_probe_weights = []
        for mtp_idx in range(mtp_len):
            valid_horizon = horizon - mtp_idx
            offset_valid = latent_weight[:, mtp_idx:]
            denom = offset_valid.sum().clamp_min(1.0)
            reward_loss = -(
                reward_target_probs[:, mtp_idx:]
                * torch.log_softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
            ).sum(dim=-1)
            reward_loss_sum = reward_loss_sum + (reward_loss * offset_valid).sum() / denom
            discount_target = future_discounts[:, mtp_idx:]
            discount_bce = F.binary_cross_entropy_with_logits(
                pred_continuation_logits_all[:, :valid_horizon, mtp_idx],
                discount_target,
                reduction="none",
            )
            discount_entropy = binary_target_entropy(discount_target)
            discount_bce_sum = discount_bce_sum + (discount_bce * offset_valid).sum() / denom
            discount_entropy_sum = discount_entropy_sum + (discount_entropy * offset_valid).sum() / denom
            pred_reward_scalar = reward_logits_to_scalar(pred_reward_logits_all[:, :valid_horizon, mtp_idx])
            target_reward_scalar = future_rewards[:, mtp_idx:]
            reward_error = pred_reward_scalar - target_reward_scalar
            pred_reward_probs = torch.softmax(pred_reward_logits_all[:, :valid_horizon, mtp_idx], dim=-1)
            pred_edge_mass = pred_reward_probs[..., 0] + pred_reward_probs[..., -1]
            target_edge_frac = (
                (target_reward_scalar <= reward_raw_edge_min)
                | (target_reward_scalar >= reward_raw_edge_max)
            ).to(offset_valid.dtype)
            reward_abs_error_sums.append((reward_error.abs() * offset_valid).sum())
            reward_error_sums.append((reward_error * offset_valid).sum())
            reward_pred_edge_sums.append((pred_edge_mass * offset_valid).sum())
            reward_target_edge_sums.append((target_edge_frac * offset_valid).sum())
            reward_probe_weights.append(denom)

        # v187: SIGReg on the clean tanh-bounded encoder summaries (mirror).
        sigreg_latents = encoded_sequence
        wm_sigreg_loss = masked_token_sigreg_static(sigreg_latents, sigreg_weight, sigreg_A_obs, sigreg_A_out)
        wm_discount_bce = discount_bce_sum
        wm_discount_bce_excess = discount_bce_sum - discount_entropy_sum
        wm_loss = (
            wm_latent_loss
            + args.lewm_reward_loss_coef * reward_loss_sum
            + args.lewm_discount_loss_coef * wm_discount_bce
            + args.lewm_sigreg_coef * wm_sigreg_loss
        )
        return (
            wm_loss,
            wm_latent_loss,
            wm_latent_mse,
            reward_loss_sum,
            wm_discount_bce,
            wm_discount_bce_excess,
            wm_sigreg_loss,
            torch.stack(reward_abs_error_sums),
            torch.stack(reward_error_sums),
            torch.stack(reward_pred_edge_sums),
            torch.stack(reward_target_edge_sums),
            torch.stack(reward_probe_weights),
            initial_summary,
            target_summaries,
        )

    if args.torch_compile:
        print(
            "compiling world_model.supervised_loss with torch.compile("
            "mode='default', fullgraph=False, dynamic=False, cudagraphs=False)"
        )
        try:
            import torch._inductor.config as inductor_config

            with inductor_config.patch({"triton.cudagraphs": False}):
                wm_supervised_loss_forward_compiled = torch.compile(
                    wm_supervised_loss_forward,
                    mode="default",
                    fullgraph=False,
                    dynamic=False,
                )
        except Exception as exc:
            print(f"warning: failed to patch WM loss cudagraphs off ({exc}); compiling with mode='default'")
            wm_supervised_loss_forward_compiled = torch.compile(
                wm_supervised_loss_forward,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )

        def wm_supervised_loss_forward(*call_args, **call_kwargs):
            return wm_supervised_loss_forward_compiled(*call_args, **call_kwargs)

    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    agent_inputs = torch.zeros((args.num_steps, args.num_envs, agent.agent_input_dim)).to(device)
    next_transition_agent_inputs = torch.zeros((args.num_steps, args.num_envs, agent.agent_input_dim), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    # v179: per-step incoming outcome [symlog(r_in), c_in] used to ground obs[step]'s
    # outcome tokens at the encoder. prev_* persist across iterations (continuous
    # rollout); reset-to-0 at episode boundaries is handled by the *(1-next_done).
    outcomes_in = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
    prev_reward_in = torch.zeros(args.num_envs, device=device)
    prev_cont_in = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rollout_latent_history = []
    rollout_action_history = []
    rollout_history_valid = []
    neutral_action = torch.zeros(args.num_envs, int(np.prod(envs.single_action_space.shape)), device=device)
    imagine_prompt_bank = []

    # Dreamer4-style EMA of the return spread (scale-only advantage normalizer).
    # Persists across iterations; lazily initialized to the first batch's stats.
    # Percentile retnorm state: EMA of return percentiles (scope="ema") + last logged scale.
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    def log_obs_token_diagnostics():
        for name, value in agent.world_model.obs_bias_diagnostics(obs).items():
            writer.add_scalar(f"lewm/{name}", value.item(), global_step)

    def build_belief_agent_input(obs_tensor, incoming_outcome):
        current_latent_full = agent.world_model.encode_summary(obs_tensor, incoming_outcome)
        if agent.detach_world_model_from_agent:
            current_latent_full = current_latent_full.detach()
        # v173: the FULL summary is the raw current-state embed handed to combine_agent_input
        # (buffered alongside the belief, then FiLM-fused at the trunk under obs_cond_mode).
        # v179: the recurrent stream is the FULL 19-token summary (outcome tokens included).
        current_latent = current_latent_full
        context_len = min(args.lewm_context, len(rollout_latent_history) + 1)
        n_past = context_len - 1
        past_latents = rollout_latent_history[-n_past:] if n_past > 0 else []
        past_actions = rollout_action_history[-n_past:] if n_past > 0 else []
        past_valids = rollout_history_valid[-n_past:] if n_past > 0 else []
        if n_past > 0:
            past_latents = [
                latent * valid.view(-1, 1, 1).to(latent.dtype)
                for latent, valid in zip(past_latents, past_valids)
            ]
            past_actions = [
                action * valid.view(-1, 1).to(action.dtype)
                for action, valid in zip(past_actions, past_valids)
            ]
        belief_latents = torch.stack(past_latents + [current_latent], dim=1)
        belief_actions = torch.stack(past_actions + [neutral_action], dim=1)
        # v192: CLEAN context (mix_ctx=False). This function builds every belief the
        # actor/critic consume on the real rollout -- the decision-path agent input AND
        # the next-transition value-bootstrap encode -- so fresh 10% ctx-noise here
        # injected zero-mean jitter into every GAE delta. Deterministic now.
        belief = agent.world_model.belief_from_history(belief_latents, belief_actions, mix_ctx=False)
        agent_input = agent.combine_agent_input(current_latent_full, belief)
        return agent_input, current_latent

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for pg in optimizer.param_groups:  # v179.2: proportional anneal preserves each group's base lr (incl. the higher probe-head group)
                pg["lr"] = frac * pg["initial_lr"]

        rollout_actor_active = global_step >= args.wm_warmup_steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # v179: ground next_obs's outcome tokens in the incoming transition
                # (r_{step-1}, c_{step-1}). At an episode's first state prev_* are 0.
                incoming = torch.stack([symlog(prev_reward_in), prev_cont_in], dim=-1)  # (num_envs,2)
                outcomes_in[step] = incoming
                agent_input, current_latent = build_belief_agent_input(next_obs, incoming)
                if not rollout_actor_active and args.actor_dist == "beta":
                    action = ((agent.action_low + agent.action_high) * 0.5).expand(args.num_envs, -1)
                    z = torch.full_like(action, 0.5)
                    logprob = torch.zeros(args.num_envs, device=device)
                    value_logits = agent.get_value_from_agent_input(next_obs, agent_input)
                else:
                    action, z, logprob, ent, value_logits = agent.get_action_and_value_from_agent_input(
                        next_obs,
                        agent_input,
                    )
                p = torch.softmax(value_logits[:, 0], dim=-1)   # horizon 0 = V(s_t)
                agent_inputs[step] = agent_input
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits[:, 0])
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            rollout_latent_history.append(current_latent.detach())
            rollout_action_history.append(action.detach())
            rollout_history_valid.append(torch.ones(args.num_envs, device=device, dtype=torch.bool))
            if len(rollout_latent_history) > args.lewm_context - 1:
                rollout_latent_history = rollout_latent_history[-(args.lewm_context - 1) :]
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
            boundary_tensor = boundary_tensor_f.bool()
            with torch.no_grad():
                # v179: transition_next_obs_t is the REAL next state s' (final_obs at a
                # boundary), so ground it with the actual transition outcome
                # (symlog(reward), 1-termination) -- matching the WM-loss fut_incoming.
                transition_incoming = torch.stack(
                    [symlog(reward_tensor), 1.0 - termination_tensor], dim=-1
                )
                next_transition_agent_input, _ = build_belief_agent_input(
                    transition_next_obs_t, transition_incoming
                )
                next_transition_agent_inputs[step] = next_transition_agent_input
            # v179: the next decision-path obs is a RESET state at a boundary, so its
            # incoming outcome is 0 there (achieved by *(1-next_done)); next_done is the
            # boundary (terminated|truncated) for this transition.
            next_done_f = next_done.float()
            prev_reward_in = reward_tensor * (1.0 - next_done_f)
            prev_cont_in = 1.0 - next_done_f
            if bool(boundary_tensor.any()):
                for hist_idx in range(len(rollout_history_valid)):
                    rollout_history_valid[hist_idx] = rollout_history_valid[hist_idx] & (~boundary_tensor)

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
                    # v179: store the per-step incoming outcome so dream prompt obs are
                    # grounded consistently with how they were encoded at rollout time.
                    "outcomes_in": outcomes_in.detach().clone(),
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
                    next_transition_agent_inputs[-1]
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
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = hl_support.cdf_fraction(returns)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        wm_losses = []
        wm_latent_losses = []
        wm_latent_mse_losses = []
        wm_flow_mean_mse_metric = None
        wm_flow_sampled_mse_metric = None
        wm_reward_losses = []
        wm_discount_bce_losses = []
        wm_discount_bce_excess_losses = []
        wm_sigreg_losses = []
        wm_effrank_joint = []
        wm_effrank_pertoken = []
        wm_grad_norms = []
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
        wm_length1_ar_discount_bce_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_discount_entropy_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_length1_ar_weight = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_latent_mse_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_reward_abs_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_reward_error_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_discount_bce_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_discount_entropy_sum = torch.zeros(args.lewm_dyn_horizon, device=device)
        wm_warm_ar_weight = torch.zeros(args.lewm_dyn_horizon, device=device)
        # WM training is always enabled; latent loss coefficient is fixed at 1.0.
        wm_b_inds = np.arange(args.batch_size)
        horizon = args.lewm_dyn_horizon
        mtp_len = args.lewm_mtp_len
        max_start = args.num_steps - 1
        obs_shape = envs.single_observation_space.shape
        wm_minibatch_size = min(args.minibatch_size, args.lewm_minibatch_size)
        for _ in range(args.lewm_update_epochs):
            wm_np_rng.shuffle(wm_b_inds)
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
                future_discounts = args.gamma * (1.0 - future_terminations)
                future_boundaries = transition_boundaries[safe_hist_step_inds, env_inds]
                future_valids = transition_valids[safe_hist_step_inds, env_inds]
                future_next_obs = next_obses[safe_hist_step_inds, env_inds]
                # v179: incoming-outcome grounding for the encoded sequence
                # [anchor, next_obs_0..H-1]. Position k (=1..H) is future_next_obs[k-1],
                # whose INCOMING reward = future_rewards[:, k-1] (symlog) and incoming
                # continuation = 1 - future_terminations[:, k-1]. The anchor's incoming was
                # stored at rollout time in outcomes_in. cat(anchor, fut) lines up exactly.
                anchor_incoming = outcomes_in[mb_step_inds, mb_env_inds]                       # (mb,2)
                fut_incoming = torch.stack(
                    [symlog(future_rewards), 1.0 - future_terminations], dim=-1
                )                                                                              # (mb,H,2)
                incoming_outcomes = torch.cat([anchor_incoming.unsqueeze(1), fut_incoming], dim=1)  # (mb,H+1,2)
                prev_continues = torch.cat(
                    [
                        torch.ones(mb_size, 1, device=device),
                        1.0 - future_boundaries[:, :-1],
                    ],
                    dim=1,
                )
                step_weight = torch.cumprod(prev_continues, dim=1) * hist_in_rollout.float()
                latent_weight = step_weight * future_valids
                reward_target_probs_flat = reward_support.project(future_rewards.reshape(-1))
                reward_target_probs = reward_target_probs_flat.reshape(mb_size, horizon, -1)
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
                sigreg_weight = torch.cat(
                    [torch.ones(mb_size, 1, device=device), latent_weight],
                    dim=1,
                )
                # v191: BOTH projections are per-token D-dim now (obs tokens fold their
                # token axis into the sample axis; see masked_token_sigreg_static). A_obs
                # stays a SEPARATE draw from A_out so the two groups' ECF statistics are
                # not coupled through a shared projection, and the loss/metrics signature
                # is unchanged.
                sigreg_A_obs = sigreg.sample_projection(
                    agent.world_model.dim,
                    device,
                    future_actions.dtype,
                    generator=wm_torch_generator,
                )
                sigreg_A_out = sigreg.sample_projection(
                    agent.world_model.dim,
                    device,
                    future_actions.dtype,
                    generator=wm_torch_generator,
                )
                # v188: flow randomness, sampled OUTSIDE the compiled loss region (same
                # pattern as the SIGReg projections) so the region stays graph-safe and
                # deterministic. PLAIN flow: signal ~ U{0..K_MAX-1} per (row, timestep),
                # unsnapped; no step sizes.
                n_tok = agent.world_model.num_latent_tokens
                tok_d = agent.world_model.dim
                flow_k_max = agent.world_model.flow_head.k_max
                flow_signal = torch.randint(
                    0, flow_k_max, (mb_size, horizon),
                    device=device, generator=wm_torch_generator,
                )
                flow_eps = torch.randn(
                    mb_size, horizon, n_tok, tok_d, device=device, generator=wm_torch_generator
                )
                # v187.1 (review fix): MEAN-PATH ANCHOR rows. The policy input / probes /
                # self-conditioning all consume f(z=0, signal=0), but the flow loss
                # otherwise only ever feeds z_t = eps at signal 0 (a full N(0,I) draw,
                # ||z|| ~ 8) -- the zero VECTOR is off-support and nothing ties it to the
                # conditional mean. Force a fraction of rows to that exact query
                # (signal=0, eps=0): their grounding term becomes a direct regression
                # f(0|belief) -> x, anchoring the temp-0 mean path.
                if args.lewm_flow_meanpath_prob > 0.0:
                    meanpath_mask = (
                        torch.rand(mb_size, horizon, device=device, generator=wm_torch_generator)
                        < args.lewm_flow_meanpath_prob
                    )
                    flow_signal = torch.where(
                        meanpath_mask, torch.zeros_like(flow_signal), flow_signal
                    )
                    flow_eps = flow_eps * (~meanpath_mask)[:, :, None, None].to(flow_eps.dtype)
                flow_selfcond_mask = (
                    torch.rand(mb_size, horizon, device=device, generator=wm_torch_generator)
                    < args.lewm_flow_self_cond_prob
                )
                flow_ctx_noise = torch.randn(
                    mb_size, horizon, n_tok, tok_d, device=device, generator=wm_torch_generator
                )
                wm_loss = wm_supervised_loss_forward(
                    obs[mb_step_inds, mb_env_inds],
                    future_next_obs.reshape((-1,) + obs_shape),
                    incoming_outcomes,
                    future_actions,
                    future_rewards,
                    future_discounts,
                    latent_weight,
                    reward_target_probs,
                    pred_outcome_actions,
                    sigreg_weight,
                    sigreg_A_obs,
                    sigreg_A_out,
                    flow_eps,
                    flow_signal,
                    flow_selfcond_mask,
                    flow_ctx_noise,
                )
                optimizer.zero_grad(set_to_none=True)
                wm_loss.backward()
                with torch.no_grad():
                    (
                        wm_loss_metric,
                        wm_latent_loss,
                        wm_latent_mse,
                        wm_reward_loss,
                        wm_discount_bce,
                        wm_discount_bce_excess,
                        wm_sigreg_loss,
                        reward_abs_error_batch,
                        reward_error_batch,
                        reward_pred_edge_batch,
                        reward_target_edge_batch,
                        reward_probe_weight_batch,
                        initial_summary,
                        target_summaries,
                    ) = wm_supervised_metrics_forward(
                        obs[mb_step_inds, mb_env_inds],
                        future_next_obs.reshape((-1,) + obs_shape),
                        incoming_outcomes,
                        future_actions,
                        future_rewards,
                        future_discounts,
                        latent_weight,
                        reward_target_probs,
                        pred_outcome_actions,
                        sigreg_weight,
                        sigreg_A_obs,
                        sigreg_A_out,
                        flow_eps,
                        flow_signal,
                        flow_selfcond_mask,
                        flow_ctx_noise,
                    )

                # Labeled AR diagnostics are detached. They run after the compiled
                # supervised-loss backward so CUDA graph outputs from the loss do
                # not need to survive across later compiled diagnostic calls.
                # v179: the recurrent stream is the FULL 19-token summary.
                with torch.no_grad():
                    def accumulate_labeled_ar(
                        hist_latents,
                        hist_actions,
                        extra_weight,
                        latent_mse_sum,
                        reward_abs_error_sum,
                        reward_error_sum,
                        discount_bce_sum,
                        discount_entropy_sum,
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
                                temperature=0.0,  # v187: mean path -> comparable to v186 AR MSE
                            )
                            w_h = latent_weight[:, h] * extra_weight
                            ar_latent_mse = F.mse_loss(
                                ar_next_summary,
                                target_summaries[:, h],
                                reduction="none",
                            ).mean(dim=(-1, -2))
                            ar_reward_scalar = reward_logits_to_scalar(ar_reward_logits)
                            ar_reward_error = ar_reward_scalar - future_rewards[:, h]
                            ar_discount_bce = F.binary_cross_entropy_with_logits(
                                ar_continuation_logits,
                                future_discounts[:, h],
                                reduction="none",
                            )
                            ar_discount_entropy = binary_target_entropy(future_discounts[:, h])
                            latent_mse_sum[h] += (ar_latent_mse * w_h).sum()
                            reward_abs_error_sum[h] += (ar_reward_error.abs() * w_h).sum()
                            reward_error_sum[h] += (ar_reward_error * w_h).sum()
                            discount_bce_sum[h] += (ar_discount_bce * w_h).sum()
                            discount_entropy_sum[h] += (ar_discount_entropy * w_h).sum()
                            weight_sum[h] += w_h.sum()
                            hist_latents.append(ar_next_summary.detach())

                    if args.diagnostics_length1_ar:
                        accumulate_labeled_ar(
                            [initial_summary.detach()],
                            [],
                            torch.ones(mb_size, device=device),
                            wm_length1_ar_latent_mse_sum,
                            wm_length1_ar_reward_abs_error_sum,
                            wm_length1_ar_reward_error_sum,
                            wm_length1_ar_discount_bce_sum,
                            wm_length1_ar_discount_entropy_sum,
                            wm_length1_ar_weight,
                        )

                    if args.diagnostics_warm_ar:
                        warm_context = min(agent.world_model.context, args.num_steps)
                        warm_valid = mb_step_inds >= (warm_context - 1)
                        for d in range(0, warm_context - 1):
                            warm_valid &= dones[(mb_step_inds - d).clamp_min(0), mb_env_inds] == 0
                        if bool(warm_valid.any()):
                            win_d = torch.arange(warm_context - 1, -1, -1, device=device)
                            win_steps = (mb_step_inds[:, None] - win_d[None, :]).clamp(0, max_start)
                            win_envs = mb_env_inds[:, None].expand_as(win_steps)
                            # v179: ground each window obs with its stored incoming outcome.
                            win_incoming = outcomes_in[win_steps, win_envs]
                            win_summaries = wm_supervised_encode_summary(
                                obs[win_steps, win_envs].reshape((-1,) + obs_shape),
                                win_incoming.reshape(-1, agent.world_model.num_semantic_tokens),
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
                                [win_summaries[:, d].detach() for d in range(warm_context)],
                                [win_actions[:, d] for d in range(max(0, warm_context - 1))],
                                warm_valid.to(dtype=latent_weight.dtype),
                                wm_warm_ar_latent_mse_sum,
                                wm_warm_ar_reward_abs_error_sum,
                                wm_warm_ar_reward_error_sum,
                                wm_warm_ar_discount_bce_sum,
                                wm_warm_ar_discount_entropy_sum,
                                wm_warm_ar_weight,
                            )

                    # v188: TEMP-0 CONDITIONAL-MEAN path vs K sampled 8-step plain Euler paths
                    # (one-step, teacher-forced belief on the standard noisy-context
                    # inference path). Run once per iteration on a small slice; if the
                    # sampled MSE tracks the mean MSE the flow is well-calibrated
                    # (samples cost accuracy only through genuine stochasticity).
                    if wm_flow_mean_mse_metric is None and args.lewm_flow_eval_samples > 0:
                        n_eval = min(256, mb_size)
                        eval_history = torch.cat(
                            [initial_summary[:n_eval].unsqueeze(1), target_summaries[:n_eval, :-1]],
                            dim=1,
                        )
                        eval_belief = agent.world_model._predictor_trunk(
                            eval_history, future_actions[:n_eval]
                        )
                        eval_targets = target_summaries[:n_eval]
                        eval_w = latent_weight[:n_eval]
                        eval_denom = eval_w.sum().clamp_min(1.0)
                        mean_pred = agent.world_model.flow_mean_next(eval_belief)
                        mean_mse = F.mse_loss(mean_pred, eval_targets, reduction="none").mean(dim=(-1, -2))
                        wm_flow_mean_mse_metric = ((mean_mse * eval_w).sum() / eval_denom).item()
                        sampled_acc = 0.0
                        for _ in range(args.lewm_flow_eval_samples):
                            sample_pred = agent.world_model.flow_sample_next(eval_belief, temperature=1.0)
                            sample_mse = F.mse_loss(sample_pred, eval_targets, reduction="none").mean(dim=(-1, -2))
                            sampled_acc += ((sample_mse * eval_w).sum() / eval_denom).item()
                        wm_flow_sampled_mse_metric = sampled_acc / args.lewm_flow_eval_samples

                wm_gn = nn.utils.clip_grad_norm_(wm_latent_params, args.lewm_grad_clip)
                wm_outcome_io_gn = nn.utils.clip_grad_norm_(
                    wm_outcome_io_params,
                    args.lewm_grad_clip,
                )
                optimizer.step()
                wm_grad_norms.append(float(wm_gn))
                wm_outcome_io_grad_norms.append(float(wm_outcome_io_gn))
                wm_reward_abs_error_sums += reward_abs_error_batch
                wm_reward_error_sums += reward_error_batch
                wm_reward_pred_edge_sums += reward_pred_edge_batch
                wm_reward_target_edge_sums += reward_target_edge_batch
                wm_reward_probe_weights += reward_probe_weight_batch
                wm_losses.append(wm_loss_metric.item())
                wm_latent_losses.append(wm_latent_loss.item())
                wm_latent_mse_losses.append(wm_latent_mse.item())
                wm_reward_losses.append(wm_reward_loss.item())
                wm_discount_bce_losses.append(wm_discount_bce.item())
                wm_discount_bce_excess_losses.append(wm_discount_bce_excess.item())
                wm_sigreg_losses.append(wm_sigreg_loss.item())
                # v175 probe: cross-token effective rank of the encoder summaries.
                # joint = rank of the flattened (S*D) frame vector; pertoken = mean
                # per-slot (D) rank. joint/pertoken in [1, S] => cross-token diversity.
                # v179: S is now the full 19-token summary.
                _summ = target_summaries[:, 0]               # (B, S, D), anchor-next step
                _nb = min(512, _summ.shape[0])
                _summ = _summ[:_nb]
                wm_effrank_joint.append(spectral_effective_rank(_summ.reshape(_nb, -1)).item())
                wm_effrank_pertoken.append(
                    spectral_effective_rank(_summ.transpose(0, 1).contiguous()).mean().item()
                )

        world_model_only = not rollout_actor_active
        if world_model_only:
            writer.add_scalar("charts/world_model_only", 1.0, global_step)
            if wm_losses:
                writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
                writer.add_scalar("lewm/obs_latent_loss", float(np.mean(wm_latent_losses)), global_step)
                writer.add_scalar("lewm/obs_latent_mse", float(np.mean(wm_latent_mse_losses)), global_step)
                if wm_flow_mean_mse_metric is not None:
                    writer.add_scalar("lewm/flow_mean_mse", wm_flow_mean_mse_metric, global_step)
                    writer.add_scalar("lewm/flow_sampled_mse", wm_flow_sampled_mse_metric, global_step)
                log_obs_token_diagnostics()
                writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
                writer.add_scalar("lewm/discount_bce", float(np.mean(wm_discount_bce_losses)), global_step)
                writer.add_scalar("lewm/discount_bce_excess", float(np.mean(wm_discount_bce_excess_losses)), global_step)
                writer.add_scalar("lewm/obs_sigreg", float(np.mean(wm_sigreg_losses)), global_step)
                if wm_effrank_joint:
                    writer.add_scalar("charts/effrank_obs_joint", float(np.mean(wm_effrank_joint)), global_step)
                    writer.add_scalar("charts/effrank_obs_pertoken", float(np.mean(wm_effrank_pertoken)), global_step)
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
                        wm_length1_ar_reward_error_sum, wm_length1_ar_discount_bce_sum,
                        wm_length1_ar_discount_entropy_sum,
                        wm_length1_ar_weight,
                    )
                if wm_warm_ar_weight.sum().item() > 0:
                    log_labeled_rollout_diagnostic(
                        writer, "diagnostics/warm_ar", global_step,
                        wm_warm_ar_latent_mse_sum, wm_warm_ar_reward_abs_error_sum,
                        wm_warm_ar_reward_error_sum, wm_warm_ar_discount_bce_sum,
                        wm_warm_ar_discount_entropy_sum,
                        wm_warm_ar_weight,
                    )
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            continue
        writer.add_scalar("charts/world_model_only", 0.0, global_step)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_agent_inputs = agent_inputs.reshape(-1, agent.agent_input_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        b_u = u.reshape(-1)
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

        # Percentile retnorm, global scopes ("ema"/"batch"): compute one return-spread
        # scale S = max(floor, P95 - P5) per iteration. "ema" smooths the percentiles
        # across iterations (rate 0.01); "batch" uses the fresh per-rollout spread.
        # scope=="minibatch" SKIPS this and recomputes S fresh per minibatch below.
        if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
            with torch.no_grad():
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(b_returns, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch"
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        actor_epochs_used = 0
        pg_loss = torch.zeros((), device=device)
        entropy_loss = torch.zeros((), device=device)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        actor_gn = torch.zeros((), device=device)
        critic_gn = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            actor_active = epoch < args.actor_update_epochs
            if actor_active:
                actor_epochs_used = epoch + 1
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if actor_active:
                    _, _, newlogprob, entropy, value_logits = agent.get_action_and_value_from_agent_input(
                        b_obs[mb_inds],
                        b_agent_inputs[mb_inds],
                        b_latent_zs[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio < 1.0 - args.clip_coef) | (ratio > 1.0 + clip_hi)).float().mean().item()
                        ]

                    if args.adv_transform_scope == "minibatch":
                        mb_raw_adv = shape_advantage(
                            b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device
                        )
                    else:
                        mb_raw_adv = b_policy_adv[mb_inds]
                    mb_advantages = mb_raw_adv
                    if args.norm_adv:
                        if args.norm_adv_scope == "batch":
                            mb_advantages = b_policy_adv_normed[mb_inds]
                        else:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    if args.ret_percnorm:
                        # Percentile retnorm: divide advantage by a return-spread scale.
                        # minibatch scope recomputes S = max(floor, P95-P5) fresh from this
                        # minibatch's returns; ema/batch use the iteration-level ret_perc_scale.
                        if args.ret_perc_scope == "minibatch":
                            mb_ret = b_returns[mb_inds]
                            qs = torch.tensor(
                                [args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device
                            )
                            lo, hi = torch.quantile(mb_ret, qs)
                            mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                            mb_advantages = mb_advantages / mb_perc_scale
                            ret_perc_scale = mb_perc_scale.item()
                        else:
                            mb_advantages = mb_advantages / ret_perc_scale

                    # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                    # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                    # Split on the shaped advantage's sign (pre-norm = the true advantage sign).
                    if args.pos_neg_alpha != 0.5:
                        mb_advantages = mb_advantages * torch.where(
                            mb_raw_adv >= 0,
                            2.0 * args.pos_neg_alpha,
                            2.0 * (1.0 - args.pos_neg_alpha),
                        )

                    ratio_clamped = torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + clip_hi,
                    )
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * ratio_clamped
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                else:
                    value_logits = agent.get_value_from_agent_input(
                        b_obs[mb_inds],
                        b_agent_inputs[mb_inds],
                    )

                # HL-Gauss MTP value loss: per-horizon CE to the scalar-return target,
                # summed across valid future horizons per row. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                value_ce = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(dtype=value_ce.dtype)
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                if auto_alpha and actor_active:
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
                    active_critic_params = critic_params if actor_active else critic_only_params
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=actor_active)
                    critic_gn = nn.utils.clip_grad_norm_(active_critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in active_critic_params if p.grad is not None]
                    if actor_active:
                        optimizer.zero_grad(set_to_none=True)
                        (pg_loss - ent_coef_eff * entropy_loss).backward()
                        actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                        # trunk.grad currently = clip_actor(d pg / d trunk); add the
                        # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                        for p, g in value_grads:
                            p.grad = g if p.grad is None else p.grad + g
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        for p, g in value_grads:
                            p.grad = g
                    optimizer.step()
                else:
                    loss = v_loss * args.vf_coef
                    if actor_active:
                        loss = loss + pg_loss - ent_coef_eff * entropy_loss
                    optimizer.zero_grad(set_to_none=True)  # v103: None (not 0) so AdamW wd skips ungraded WM params
                    loss.backward()
                    if actor_active:
                        critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    else:
                        critic_gn = nn.utils.clip_grad_norm_(critic_only_params, args.max_grad_norm)
                        for p in agent.parameters():
                            if id(p) not in critic_only_param_ids:
                                p.grad = None
                    optimizer.step()

        # =========================== IMAGINATION TRAINING ===========================
        # dreamer4-style: with the WM frozen as a DETACHED simulator, generate fresh
        # on-policy rollouts dreamed from real rollout states and train the SAME
        # actor/critic on them at full strength. v167 mirrors real training:
        # each generated block gets actor_update_epochs actor epochs and
        # update_epochs critic epochs.
        imagine_metrics = None
        if args.imagine_enable and global_step >= args.imagine_warmup_steps:
            H_im = args.imagine_horizon
            B_im = args.imagine_batch_size
            act_dim_im = agent.world_model.act_dim
            imagine_neutral = torch.zeros(B_im, act_dim_im, device=device)
            imagine_ent_coef = log_alpha.exp().detach() if auto_alpha else args.ent_coef
            im_pg_losses, im_v_losses, im_entropies = [], [], []
            im_actor_gns, im_critic_gns = [], []
            im_return_sum = torch.zeros((), device=device)
            im_return_sq_sum = torch.zeros((), device=device)
            im_reward_sum = torch.zeros((), device=device)
            im_reward_sq_sum = torch.zeros((), device=device)
            im_discount_to_gamma_ratio_sum = torch.zeros((), device=device)
            im_discount_sum = torch.zeros((), device=device)
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
                bank_outcomes_in = torch.stack([entry["outcomes_in"] for entry in imagine_prompt_bank], dim=0)
            else:
                bank_obs = obs.unsqueeze(0)
                bank_actions = actions.unsqueeze(0)
                bank_dones = dones.unsqueeze(0)
                bank_outcomes_in = outcomes_in.unsqueeze(0)
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
            im_old_approx_kls, im_approx_kls, im_clipfracs = [], [], []
            im_blocks = 0
            im_actor_updates = 0
            im_critic_updates = 0
            for _ in range(args.imagine_batches):
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
                    ag_inputs, im_zs, im_lps, im_rs, im_cs, im_vps = [], [], [], [], [], []
                    # Recurrent imagination: carry a GROWING (latent, action) history and
                    # condition the predictor on it, exactly like the teacher-forced
                    # training path (predict_mtp_from_history, whose context grows up to
                    # self.context). v103 WARM-START: seed the history with the real
                    # preceding rollout window (W states + W-1 real actions) instead of a
                    # length-1 context, so the first dreamed step sees a full real context
                    # — the regime the predictor is strongest in. _predictor_trunk truncates
                    # to self.context as the window then slides forward through the dream.
                    if clean_flat.numel() > 0:
                        win_d = torch.arange(W_im - 1, -1, -1, device=device)         # [W-1,...,0]
                        win_steps = step_idx.unsqueeze(0) - win_d.unsqueeze(1)        # (W, B_im): l_{j-W+1}..l_j
                        win_rollouts = rollout_idx.unsqueeze(0).expand(W_im, -1)
                        win_envs = env_idx.unsqueeze(0).expand(W_im, -1)
                        # v179: ground each prompt obs with its stored incoming outcome (the
                        # CONSUMED imagination path uses the real stored incoming, not zeros).
                        win_incoming = bank_outcomes_in[win_rollouts, win_steps, win_envs]
                        win_summ = agent.world_model.encode_summary(
                            bank_obs[win_rollouts, win_steps, win_envs].reshape((-1,) + obs_shape),
                            win_incoming.reshape(-1, agent.world_model.num_semantic_tokens),
                        ).detach().reshape(
                            W_im, B_im, agent.world_model.num_latent_tokens, agent.world_model.dim
                        )
                        # W real states l_{j-W+1}..l_j; W-1 real actions a_{j-W+1}..a_{j-1}
                        # (action[d] is taken FROM state[d], pairing as the predictor expects).
                        # v179: full 19-token recurrent state.
                        hist_latents = [win_summ[d] for d in range(W_im)]
                        win_actions = bank_actions[win_rollouts[:-1], win_steps[:-1], win_envs[:-1]]
                        hist_actions = [win_actions[d] for d in range(W_im - 1)]
                        cur_full = win_summ[W_im - 1]       # v172: full current-state embed l_j
                    else:
                        s = agent.world_model.encode_summary(
                            bank_obs[rollout_idx, step_idx, env_idx],
                            bank_outcomes_in[rollout_idx, step_idx, env_idx],
                        ).detach()
                        hist_latents = [s]                  # v179: full 19-token recurrent state l_t
                        hist_actions = []                   # no real history available
                        cur_full = s                        # v172: full current-state embed l_t
                    for t in range(H_im):
                        latent_hist = torch.stack(hist_latents, dim=1)               # (B, t+1, 19, dim)
                        # Belief the actor acts on: current action slot is neutral (the
                        # action is not chosen yet), matching build_belief_agent_input.
                        # v192: dream AGENT INPUT -> clean context (the dream DYNAMICS
                        # below keep the noisy context the trunk trains in).
                        belief_actions = torch.stack(hist_actions + [imagine_neutral], dim=1)
                        belief = agent.world_model.belief_from_history(latent_hist, belief_actions, mix_ctx=False)
                        # v172: concat the full current-state embed (cur_full) with the belief.
                        agent_input = agent.combine_agent_input(cur_full, belief)
                        action, z, logprob, _, value_logits = (
                            agent.get_action_and_value_from_agent_input(None, agent_input)
                        )
                        hist_actions.append(action)
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
                        hist_latents.append(next_s.detach())   # v179: full 19-token summary
                        cur_full = next_s.detach()           # v172: dreamed next state becomes current embed
                    # Bootstrap value at the final imagined state s_H (same recurrent belief).
                    latent_hist = torch.stack(hist_latents, dim=1)
                    belief_actions_H = torch.stack(hist_actions + [imagine_neutral], dim=1)
                    # v192: dream bootstrap value belief -> clean context (agent-consumed).
                    belief_H = agent.world_model.belief_from_history(latent_hist, belief_actions_H, mix_ctx=False)
                    value_probs_H = torch.softmax(
                        agent.get_value_from_agent_input(
                            None, agent.combine_agent_input(cur_full, belief_H)
                        )[:, 0],  # horizon 0 = V(s_H)
                        dim=-1,
                    )
                    value_probs_t = torch.stack(im_vps, dim=0)                  # (H,B,n) Z(s_0..s_{H-1})
                    next_value_probs = torch.stack(im_vps[1:] + [value_probs_H], dim=0)  # (H,B,n) Z(s_1..s_H)
                    im_rewards = torch.stack(im_rs, dim=0)                      # (H,B)
                    im_discounts = torch.stack(im_cs, dim=0)                    # (H,B)
                    values_t = value_probs_to_scalar(value_probs_t)
                    next_values = value_probs_to_scalar(next_value_probs)
                    # Scalar GAE over the dream. The WM continuation head is trained
                    # Dreamer3-contdisc style as the full predicted discount
                    # gamma * survival, so do not multiply by gamma again here.
                    adv = torch.zeros(H_im, B_im, device=device)
                    lastgae = torch.zeros(B_im, device=device)
                    for t in reversed(range(H_im)):
                        discount = im_discounts[t]
                        delta = im_rewards[t] + discount * next_values[t] - values_t[t]
                        lastgae = delta + args.gae_lambda * discount * lastgae
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
                        value_probs_t * (support - values_t.unsqueeze(-1)) ** 2
                    ).sum(-1).clamp_min(0).sqrt().clamp_min(args.sigma_floor_bins * bin_width)
                    cdf_frac_im = hl_support.cdf_fraction(returns_t)
                    u_im = (value_probs_t * cdf_frac_im).sum(-1)
                    f_ag = torch.stack(ag_inputs, dim=0).reshape(H_im * B_im, -1)
                    f_z = torch.stack(im_zs, dim=0).reshape(H_im * B_im, -1)
                    f_lp = torch.stack(im_lps, dim=0).reshape(-1)
                    f_adv = adv.reshape(-1)
                    f_ret = returns_t.reshape(-1)
                    f_sigma = sigma_im.reshape(-1)
                    f_u = u_im.reshape(-1)
                    f_target = target_probs_im.reshape(-1, args.critic_mtp_horizon, args.num_bins)
                    f_target_mask = return_mtp_mask_im.reshape(-1, args.critic_mtp_horizon)
                    im_return_sum += returns_t.mean()
                    im_return_sq_sum += returns_t.square().mean()
                    im_reward_sum += im_rewards.mean()
                    im_reward_sq_sum += im_rewards.square().mean()
                    im_discount_to_gamma_ratio_sum += (im_discounts / args.gamma).mean()
                    im_discount_sum += im_discounts.mean()
                    im_edge_per_h = target_probs_im[..., 0] + target_probs_im[..., -1]
                    im_edge_mask_f = return_mtp_mask_im.to(im_edge_per_h.dtype)
                    im_edge_sum += (im_edge_per_h * im_edge_mask_f).sum() / im_edge_mask_f.sum().clamp_min(1)
                    im_blocks += 1

                # Train actor once and critic for all epochs on the generated dream block.
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                for im_epoch in range(args.update_epochs):
                    actor_active_im = im_epoch < args.actor_update_epochs
                    if actor_active_im:
                        _, _, newlogprob, entropy, value_logits = agent.get_action_and_value_from_agent_input(
                            None, f_ag, f_z
                        )
                        logratio = newlogprob - f_lp
                        ratio = logratio.exp()
                        shaped = shape_advantage(f_adv, f_sigma, f_u, args, device)
                        if args.norm_adv:
                            shaped = (shaped - shaped.mean()) / (shaped.std() + 1e-8)
                        if args.ret_percnorm:
                            # Same percentile retnorm as the real path (shared raw return units).
                            # The dream block is processed whole (no minibatching), so for the
                            # "minibatch" scope we take the percentile spread over the entire
                            # imagined-return block; ema/batch reuse the iteration-level scale.
                            if args.ret_perc_scope == "minibatch":
                                qs = torch.tensor(
                                    [args.ret_perc_lo, args.ret_perc_hi], device=f_ret.device
                                )
                                lo, hi = torch.quantile(f_ret, qs)
                                im_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                                shaped = shaped / im_perc_scale
                            else:
                                shaped = shaped / ret_perc_scale
                        ratio_clamped = torch.clamp(
                            ratio,
                            1.0 - args.clip_coef,
                            1.0 + clip_hi,
                        )
                        pg_loss1_im = -shaped * ratio
                        pg_loss2_im = -shaped * ratio_clamped
                        pg_loss_im = torch.max(pg_loss1_im, pg_loss2_im).mean()
                        entropy_loss_im = entropy.mean()
                    else:
                        value_logits = agent.get_value_from_agent_input(None, f_ag)
                        pg_loss_im = torch.zeros((), device=device)
                        entropy_loss_im = torch.zeros((), device=device)

                    value_log_probs_im = torch.log_softmax(value_logits, dim=-1)
                    value_ce_im = -(f_target * value_log_probs_im).sum(dim=-1)
                    v_loss_im = (value_ce_im * f_target_mask.to(value_ce_im.dtype)).sum(dim=-1).mean()
                    if args.separate_grad_clip:
                        active_critic_params = critic_params if actor_active_im else critic_only_params
                        optimizer.zero_grad(set_to_none=True)
                        (args.vf_coef * v_loss_im).backward(retain_graph=actor_active_im)
                        im_critic_gn = nn.utils.clip_grad_norm_(active_critic_params, args.critic_grad_clip)
                        value_grads = [(p, p.grad.detach().clone()) for p in active_critic_params if p.grad is not None]
                        if actor_active_im:
                            optimizer.zero_grad(set_to_none=True)
                            (pg_loss_im - imagine_ent_coef * entropy_loss_im).backward()
                            im_actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                            for p, g in value_grads:
                                p.grad = g if p.grad is None else p.grad + g
                        else:
                            im_actor_gn = torch.zeros((), device=device)
                            optimizer.zero_grad(set_to_none=True)
                            for p, g in value_grads:
                                p.grad = g
                        optimizer.step()
                    else:
                        loss_im = v_loss_im * args.vf_coef
                        if actor_active_im:
                            loss_im = loss_im + pg_loss_im - imagine_ent_coef * entropy_loss_im
                        optimizer.zero_grad(set_to_none=True)  # v103: None (not 0) so AdamW wd skips ungraded WM params
                        loss_im.backward()
                        if actor_active_im:
                            im_critic_gn = im_actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        else:
                            im_actor_gn = torch.zeros((), device=device)
                            im_critic_gn = nn.utils.clip_grad_norm_(critic_only_params, args.max_grad_norm)
                            for p in agent.parameters():
                                if id(p) not in critic_only_param_ids:
                                    p.grad = None
                        optimizer.step()

                    im_v_losses.append(v_loss_im.item())
                    im_critic_gns.append(float(im_critic_gn))
                    im_critic_updates += 1
                    if actor_active_im:
                        with torch.no_grad():
                            _, _, post_logprob, _, _ = agent.get_action_and_value_from_agent_input(
                                None, f_ag, f_z
                            )
                            post_logratio = post_logprob - f_lp
                            post_ratio = post_logratio.exp()
                            old_approx_kl_im = (-post_logratio).mean()
                            approx_kl_im = ((post_ratio - 1.0) - post_logratio).mean()
                            clipfrac_im = (
                                ((post_ratio < 1.0 - args.clip_coef) | (post_ratio > 1.0 + clip_hi))
                                .float()
                                .mean()
                            )
                        im_pg_losses.append(pg_loss_im.item())
                        im_entropies.append(entropy_loss_im.item())
                        im_actor_gns.append(float(im_actor_gn))
                        im_old_approx_kls.append(old_approx_kl_im.item())
                        im_approx_kls.append(approx_kl_im.item())
                        im_clipfracs.append(clipfrac_im.item())
                        im_actor_updates += 1
            n_im = max(1, im_blocks)
            im_return_mean = im_return_sum / n_im
            im_reward_mean = im_reward_sum / n_im
            im_return_std = (im_return_sq_sum / n_im - im_return_mean.square()).clamp_min(0).sqrt()
            im_reward_std = (im_reward_sq_sum / n_im - im_reward_mean.square()).clamp_min(0).sqrt()
            imagine_metrics = {
                "imagine/policy_loss": float(np.mean(im_pg_losses)) if im_pg_losses else 0.0,
                "imagine/value_loss": float(np.mean(im_v_losses)) if im_v_losses else 0.0,
                "imagine/entropy": float(np.mean(im_entropies)) if im_entropies else 0.0,
                "imagine/return_mean": im_return_mean.item(),
                "imagine/return_std": im_return_std.item(),
                "imagine/reward_mean": im_reward_mean.item(),
                "imagine/reward_std": im_reward_std.item(),
                "imagine/discount_mean": (im_discount_sum / n_im).item(),
                "imagine/discount_to_gamma_ratio_mean": (im_discount_to_gamma_ratio_sum / n_im).item(),
                "imagine/target_edge_mass": (im_edge_sum / n_im).item(),
                "imagine/prompt_bank_rollouts": float(bank_rollouts),
                "imagine/prompt_clean_seeds": float(clean_flat.numel()),
                "imagine/old_approx_kl": float(np.mean(im_old_approx_kls)) if im_old_approx_kls else 0.0,
                "imagine/approx_kl": float(np.mean(im_approx_kls)) if im_approx_kls else 0.0,
                "imagine/max_approx_kl": float(np.max(im_approx_kls)) if im_approx_kls else 0.0,
                "imagine/clipfrac": float(np.mean(im_clipfracs)) if im_clipfracs else 0.0,
                "imagine/target_kl_stop": 0.0,
                "imagine/batches_used": float(im_blocks),
                "imagine/actor_updates": float(im_actor_updates),
                "imagine/critic_updates": float(im_critic_updates),
                "imagine/actor_grad_norm": float(np.mean(im_actor_gns)) if im_actor_gns else 0.0,
                "imagine/critic_grad_norm": float(np.mean(im_critic_gns)) if im_critic_gns else 0.0,
            }

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_per_h = b_target_probs[..., 0] + b_target_probs[..., -1]   # (N, mtp)
        edge_mask_f = b_target_mask.to(dtype=edge_per_h.dtype)
        edge_mass = ((edge_per_h * edge_mask_f).sum() / edge_mask_f.sum().clamp_min(1)).item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
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
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("losses/actor_epochs_used", float(actor_epochs_used), global_step)
        writer.add_scalar("losses/critic_epochs_used", float(args.update_epochs), global_step)
        if args.ret_percnorm:
            writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if wm_losses:
            writer.add_scalar("lewm/loss", float(np.mean(wm_losses)), global_step)
            writer.add_scalar("lewm/obs_latent_loss", float(np.mean(wm_latent_losses)), global_step)
            writer.add_scalar("lewm/obs_latent_mse", float(np.mean(wm_latent_mse_losses)), global_step)
            if wm_flow_mean_mse_metric is not None:
                writer.add_scalar("lewm/flow_mean_mse", wm_flow_mean_mse_metric, global_step)
                writer.add_scalar("lewm/flow_sampled_mse", wm_flow_sampled_mse_metric, global_step)
            log_obs_token_diagnostics()
            writer.add_scalar("lewm/reward_ce", float(np.mean(wm_reward_losses)), global_step)
            writer.add_scalar("lewm/discount_bce", float(np.mean(wm_discount_bce_losses)), global_step)
            writer.add_scalar("lewm/discount_bce_excess", float(np.mean(wm_discount_bce_excess_losses)), global_step)
            writer.add_scalar("lewm/obs_sigreg", float(np.mean(wm_sigreg_losses)), global_step)
            if wm_effrank_joint:
                writer.add_scalar("charts/effrank_obs_joint", float(np.mean(wm_effrank_joint)), global_step)
                writer.add_scalar("charts/effrank_obs_pertoken", float(np.mean(wm_effrank_pertoken)), global_step)
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
                    wm_length1_ar_reward_error_sum, wm_length1_ar_discount_bce_sum,
                    wm_length1_ar_discount_entropy_sum,
                    wm_length1_ar_weight,
                )
            if wm_warm_ar_weight.sum().item() > 0:
                log_labeled_rollout_diagnostic(
                    writer, "diagnostics/warm_ar", global_step,
                    wm_warm_ar_latent_mse_sum, wm_warm_ar_reward_abs_error_sum,
                    wm_warm_ar_reward_error_sum, wm_warm_ar_discount_bce_sum,
                    wm_warm_ar_discount_entropy_sum,
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
