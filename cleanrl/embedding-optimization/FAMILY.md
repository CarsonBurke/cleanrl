# Embedding Optimization

Successor family to `cleanrl/pan-goal-solver/`. That family's contract (no critic,
no action optimization through the model, hindsight BC only) was falsified in
stages: hindsight behavior cloning has no improvement operator, and every channel
that routed goal information through one-step prediction heads trained on
goal-unconditioned data measured goal-blind (goal_removal ~0.001 across
v19/v20/v21/v24). This family keeps what survived — the LeJEPA world model,
frames/embeddings as goals, hindsight relabeling as the source of supervision —
and replaces the dead policy channel with the one thing the old contract forbade:
**an analytic improvement operator**. The policy is trained by backpropagating
value through the world model.

## The core loss

The policy is deterministic and the entire policy objective is one differentiable
scalar:

```
L_pi = -[ r_hat(z, pi(z, g)) + gamma * V(f(z, pi(z, g)), g) ]
```

Gradients flow backward through V, through f, into the action, into theta.
No sampling, no argmax, no scoring sets, no contrastive term, no likelihood
ratio. The world model and critic jointly define the target *implicitly*: the
loss surface the policy descends IS the value landscape composed with dynamics.
This is teacher forcing in the original sense — regression toward the
value-ascending action — without ever materializing that action as a label.

The old family's Phase-A objective, minimize `d(f(z, pi), g)`, is the special
case: with hindsight-trained V and lambda=1, `V(z, g) = gamma^k` where k is the
step distance, so `-log_gamma V(z, g) ~ d(z, g)` and value ascent on V is
distance descent on d. V generalizes d by (a) bootstrapping (lambda < 1
sharpens beyond pure regression) and (b) composing with the reward head so the
policy is not purely goal-reaching.

The one non-negotiable detail: **f, r_hat, and V are frozen during the policy
step**. The policy must not be able to bend the landscape it is descending;
that is the model-exploitation channel (the same failure FAMILY.md flagged for
frozen-reward-head goal ascent, and the reason Dreamer-style methods stop
world-model gradients at the policy boundary).

## Architecture

1. **Encoder + dynamics** — LeJEPA-style: `E(obs) -> z`, `f(z, a) -> z'_hat`,
   trained by prediction MSE against the attached online embedding of the next
   observation plus SIGReg isotropy regularization (no EMA target, no stop-grad
   asymmetry). Family invariant inherited from pan-goal-solver: the world model
   is trained by nothing but its own losses; reward heads read detached z.
2. **Reward heads** — `r_hat(z, a)`: one-step (normalized) reward, MSE. On
   detached z. `W_hat(z)`: windowed forward reward rate ("how good is being in
   this regime"), MSE on detached z; used only to score goals.
3. **Critic `V(z, g)`** — TD(lambda) on hindsight-relabeled pairs, lambda
   returns computed along the *actually experienced* path (no GAE, no
   advantage). Reward structure is sparse-at-goal: 0 until arrival, 1 at the
   achieved frame, so `lambda=1` gives pure regression `V(z_t, z_{t+k}) <-
   gamma_g^k` (the Phase-A distance regression) and `lambda<1` adds TD
   sharpening/stitching. Separate goal discount `gamma_g` defines the metric.
   Two anchors close the extrapolation holes hindsight-only training leaves:
   cross-trajectory negative pairs regressed to `gamma_g^T` (without them V is
   never asked about a goal off the achieved future, so the "reachability
   prior" would be vacuous extrapolation), and a `V(g,g)=0.99` self-anchor
   (0.99, not 1.0, so the near-goal sigmoid logit cannot drift into
   saturation and kill dV/dz' exactly where the policy needs gradient).
4. **Goal proposer `G(z) -> g`** — regresses toward the embedding maximizing
   the value-based score: `max W_hat(g) + alpha * V(z, g)` with W_hat and V
   frozen. The `V(z, g)` term is the reachability prior (unreachable
   embeddings have V ~ 0, pulling proposals back toward the achievable set);
   a small `||g||^2` penalty keeps proposals inside the SIGReg-shaped
   (isotropic Gaussian) embedding distribution — both are the guards against
   the off-manifold frozen-head exploitation the old family documented.
5. **Policy `pi(z, g) -> a`** — tanh-squashed deterministic head, trained by
   L_pi above, acting with additive Gaussian exploration noise. Trained on a
   mixture of hindsight goals (goal-reaching generalization) and proposed
   goals (task alignment with what it sees at act time).

Data flow per iteration (PPO-shaped outer loop, on-policy segments):
rollout with `a = pi(z, G(z)) + noise` -> world-model epochs -> critic epochs
(hindsight lambda-return targets recomputed once per iteration) -> proposer
step -> policy step. Freezing is implemented as `backward(inputs=<module
params>)` + per-module optimizers, never `requires_grad` flips (compile-safe).

## Why this can work where pan-goal-solver could not

- **Improvement operator**: GCSL/hindsight BC can only reproduce the data's
  action-goal mutual information, which was ~0 under OU/random collection.
  Here goal information reaches the policy analytically through dV/da — it
  exists as soon as V has any g-dependence, which hindsight TD guarantees by
  construction (targets gamma_g^k depend on g through k). No identifiability
  bootstrap needed.
- **The critic is the stitcher**: TD(lambda<1) composes paths that were never
  executed end-to-end; BC never could.
- **One-step model only**: gradients traverse f exactly once (SVG(1)-style),
  so compounding model error — the classic backprop-through-dynamics failure —
  is bounded; long-horizon credit comes from V, not from unrolling.

## Standing objections from the repo's own record (answered, and instrumented)

- *"One-step goal-conditioned channels measured dead three times (v19/v20/v21;
  v23 post-mortem)."* Those were **supervised** heads trained on
  goal-unconditioned data: the goal signal had to exist as mutual information
  I(action; goal | belief) in the replay, and it didn't. Here the policy loss
  consumes no action labels at all — the goal signal is the analytic gradient
  dV/da through f, which exists as soon as V is g-dependent, and hindsight TD
  makes V g-dependent by construction (targets gamma_g^k vary with g through k).
  The claim is falsifiable the same way the old one was: if
  `goal_removal_action_mse` stays pinned at ~0.001, this family's thesis is
  wrong too. Both aliveness diagnostics (goal-shuffle and null-goal) are logged
  every iteration from v1.
- *"SIGReg whitening is active pressure against the scale structure value is
  made of"* (IDEAS.md kill-switch measurement: ~25 EV points lost regressing
  return on SIGReg'd latents; encoder-value/DESIGN_cla.md). Acknowledged and
  deliberately accepted for v1: SIGReg is the sole anti-collapse mechanism of
  the attached-target JEPA, and V here regresses gamma_g^k (a *distance*
  functional, plausibly less scale-hungry than env return) — but r_hat and
  W_hat DO regress env reward from the whitened z, so `losses/reward_mse` /
  `losses/rate_mse` are the tell. First planned ablation if they plateau high:
  SIGReg on a projector head instead of z itself, or raw-obs skip input to the
  reward heads.
- *"Do not route non-world-model losses into the world model"* (v18 reversion).
  Honored: critic, proposer, and policy all read detached z; the encoder and
  dynamics are trained by prediction + SIGReg only; reward heads read detached
  z. The policy backprops *through* f and V but updates neither
  (`backward(inputs=policy_params)`).

## Known risks (measure, don't argue)

- Value-gradient methods inherit rough loss surfaces; dV/da can be noisy early.
  Mitigation: frozen targets per policy step, grad clip, tanh bounding.
- Deterministic policy + Gaussian noise may underexplore vs. PPO's learned
  covariance. Watch early return curves.
- Mixed units in L_pi (normalized reward vs. V in [0,1]) — coefficients
  `reward_coef`/`value_coef` exist and their balance is a real degree of
  freedom, not a nuisance.
- Goal-channel aliveness must be measured, not assumed (family lesson):
  `diagnostics/goal_sensitivity_action_mse` (matched vs shuffled goal) is
  mandatory in every version.

## Versions

- `ppo_continuous_action_embopt_v1.py` — initial construction as specified
  above. HalfCheetah-first; termination prediction deliberately omitted
  (HalfCheetah is truncation-only).
  Result (job 688, cancelled externally at 1.3M): returns -133 peak @672k then
  degrading to -212. Autopsy: THE CORE THESIS HELD — goal_removal_action_mse
  0.037-0.047 (alive; the old family's dead runs pinned at ~0.001), r_hat/value
  action-gradient terms balanced (~0.7-0.9 each), reward readable from whitened
  z (reward_mse 0.03, ~97% EV — SIGReg concern not binding for one-step
  reward). The failure was the PROPOSER: free gradient ascent on frozen
  W_hat + V with a 0.5*||g||^2 prior collapsed to a near-zero constant goal
  (goal_sqnorm ~0.5 vs on-manifold ||z||^2 ~ 64; proposed_goal_sensitivity
  ~1e-4). A mode-seeking prior is wrong in high dim — the typical N(0,I_64)
  sample lives on the shell ||g||~8, and W_hat's ascent gradient can't beat a
  prior gradient linear in ||g||. The origin of a whitened space is the mean
  replay state = "stand still"; the policy faithfully reached it
  (policy_rhat -0.9: sacrificing reward to climb V). Lesson: with an alive
  goal channel, returns track goal quality — fix the proposer, not the policy.
- `ppo_continuous_action_embopt_v2.py` — RETIRED OFF-FAMILY without data
  (job 695 cancelled at ~1M, user decision). It replaced the proposer with a
  per-state argmax over replay embeddings — retrieval, not optimization: goals
  capped at achieved states (the no-improvement-operator failure reintroduced
  at the goal level) and the analytic gradient — the family's thesis —
  abandoned for one component. Kept only for its telemetry additions
  (rollout_goal_v / rollout_goal_dv, z_sqnorm), which v3 inherits.
- `ppo_continuous_action_embopt_v3.py` — the correct fix for v1's proposer
  collapse: same analytic ascent on the frozen score W_hat(g) +
  alpha_reach * V(z,g), but proposals reparameterized onto the typical shell
  of the SIGReg'd marginal, g = sqrt(dz) * unit(MLP(z)). v1's bug was the
  mode-seeking 0.5*||g||^2 prior (the N(0,I_64) mode at the origin is the
  mean replay state; its typical set is the ||g||=8 shell). The shell
  constraint is structural — no norm gradient fighting the score, no origin
  attractor, in-distribution by construction, still open-ended beyond replay.
  (job 698)
  Result @1M: returns -437 -> -37 but noisy; shell fixed the collapse and
  exposed the next exploit: proposer_what 5-8 (6+ sigma above any real rate —
  the frozen W_hat's fantasy on the measure-zero-data shell complement),
  rollout_goal_dv < 0 (goals unreachable, policy not climbing V), goal_removal
  decaying (policy learning to ignore useless goals). Improvement was carried
  by the myopic r_hat term alone.
- `ppo_continuous_action_embopt_v4.py` — anti-fantasy grounding for W_hat:
  random shell points (w.h.p. far off the data submanifold in 63-dim)
  regressed toward the minibatch-minimum rate, weight 0.25 — the same
  negative-anchor medicine that fixed V's extrapolation, applied symmetrically.
  Standing family principle: every frozen head that gets ascended needs its
  extrapolation anchored on the complement of the data. Adds
  what_real_mean/max telemetry (fantasy gap = proposer_what vs what_real_max).
  (job 699)
  Result @1.75M: FALSIFIED — sampling-based anchors cannot cap the max of an
  unconstrained MLP over a 63-dim shell (proposer_what 6-11 vs what_real_max
  <= 2.6; ascent finds ridges between anchors). V's own grounding held
  (proposer_v ~ 0: the critic correctly called the goals fantasy) but the
  ADDITIVE score let W_hat=10 outvote V=0. Returns unchanged (~-180).
  Principle amended: grounding-by-samples is insufficient for an ascended
  head — the score STRUCTURE must make exploitation self-defeating.
- `ppo_continuous_action_embopt_v5.py` — two fronts:
  (1) Proposer score made multiplicative and support-clamped:
  V(z,g) * (clamp(W_hat(g), r_lo, r_hi) - r_lo) with r_lo/r_hi the observed
  min/max windowed rate. Fantasy is structurally capped (nothing scores above
  the best OBSERVED regime; the cap ratchets up with real performance), and
  at the clamp only V's gradient survives — exploitation self-defeating.
  (2) WM convergence audit against ../le-wm found two quantitative bugs, not
  taste: the SIGReg statistic scales linearly with batch N, so coef 0.09
  (balanced at le-wm's batch 128) was ~16x too strong at our minibatch 2048 —
  now pinned to sigreg_ref_n=128 (batch-invariant, same pinning trick as
  panlejepa's lewm_sigreg_ref_n); and replayed observations carried stale
  wrapper normalization — now raw obs stored, normalized by CURRENT running
  stats at every use site (symmetric to the v1 reward-scale fix). Plus
  le-wm-style stabilization: WM on AdamW(1e-4, wd 1e-3), LayerNorm in
  encoder/dynamics hidden layers. le-wm's BatchNorm asymmetric prediction
  head deliberately not adopted (collapse is not our failure mode; BN state
  mutation is cudagraph-hostile). (job 702)
  Result @1.7M: WM FIXED (wm_pred 0.455 -> 0.012, converging; sigreg ~1.2;
  z_sqnorm stable) and the proposer became diverse, state-dependent, and
  fantasy-free (out_std 0.85, what within support). But the pendulum swung to
  TRIVIALITY: proposer_v 0.96 = goals ~2 steps away — the multiplicative gate
  over-rewards V and found the self-anchor exploit (V(g,g) trained to 0.99, so
  g ~ z scores near-perfectly). Held 16 steps, such goals immediately go stale
  (rollout_goal_dv -0.05/step): the value term pulled the policy BACK toward
  its recent state while r_hat pushed forward — equal-magnitude opposing
  gradients = flat returns. Lesson: reachability must be a CONSTRAINT (where),
  not a reward (whether); any monotone-in-V score term is exploitable at one
  end or the other.
- `ppo_continuous_action_embopt_v6.py` — aspiration-band proposer:
  score = (clamp(W_hat, r_lo, r_hi) - r_lo) - beta_band*(V(z,g) - gamma_g^H)^2
  with H = goal_refresh (gamma_g^16 ~ 0.72): triviality and fantasy both cost;
  rate is optimized freely at the pinned travel horizon; the policy-loss terms
  align instead of cancelling. Restarted as v6b (job 704) with LeJEPA health
  telemetry: z_eff_rank (participation ratio of the z covariance),
  z_top_eig_share, and wm_delta_var (per-dim Var(z'-z), the predict-no-change
  baseline that calibrates wm_pred).
- `ppo_continuous_action_embopt_v7.py` — v6b + V-where-the-policy-queries-it:
  signal-gap audit found V trains only on encoder outputs while the policy
  differentiates it exclusively at dynamics outputs f(z, pi(z,g)) — every
  policy gradient samples V out-of-distribution in a direction pi controls.
  Fix: critic consistency term V(f(z_t,a_t), g) -> lambda-target of z_{t+1}
  (arrival -> 1), weight 0.5; replay actions are policy+noise so the training
  query distribution tracks the policy's. Runs alongside v6b to isolate the
  term's effect. Remaining known gap (accepted for now): off-manifold goals
  only get the negatives' floor signal, not a graded distance — a graded
  signal would need imagined-rollout scoring (future version). (job 705)
  v6b result @2.3M: WM stays fixed (wm_pred 0.011 vs delta-var 1.35), the
  band holds exactly (proposer_v ~0.70 vs 0.72 target), proposer_what ~2.5-3.0
  is inside real support (what_real_max ~3.0) — goal QUALITY is finally
  non-pathological. But returns still plateau ~-180 and rollout_goal_dv stays
  negative (~-0.025): the policy does not close distance on held goals. Goal
  channel is alive (goal_removal_action_mse 0.03-0.05), so the residual
  suspects are V's off-distribution error at f(z,pi) (v7's target) and the
  detached-aux clutter (v8's target). z_eff_rank dipped to 9 early then
  recovered to ~23 — watch, not yet alarming.
- `ppo_continuous_action_embopt_v8.py` — v7 minus falsified auxiliary losses.
  Principle (user directive, adopted as family law): an auxiliary loss is
  usually a patch over a design flaw; the LeJEPA WM receives ONLY prediction
  MSE + SIGReg, and every other consumer reads detached embeddings (audit
  confirmed this already held — reward heads take z.detach(), freezing is via
  backward(inputs=...)). Pruned by evidence: (1) W_hat shell-negatives
  (falsified in v4 — cannot pin an MLP's max on a 63-dim shell by regressing
  samples; redundant since the v6 support clamp bounds the score structurally);
  (2) V(g,g)=0.99 self-anchor (root cause of the v5 triviality exploit; the
  lambda-recursion writes the literal constant 1 at arrival, so targets never
  consult V(g,g) — the anchor only fed the exploit). Kept with justification:
  V cross-trajectory negatives (V's ONLY off-manifold signal; the band penalty
  can reject fantasy goals only because of it) and the v7 V.f consistency term
  (trains V on its actual query distribution). Three-arm attribution:
  v6b (704) = band baseline, v7 (705) = +consistency, v8 = +consistency-aux.
  Result (v6b @2.3M, v7 @1.5M, v8 @2.2M; 704/705 cancelled externally):
  v8 best-in-family early (-154 avg @672k) then regressed to the same plateau.
  v7 == v6b at matched steps => consistency term FALSIFIED (pruned in v9).
  The decisive pattern: policy_next_v climbs 0.52 -> 0.70 while realized V
  stays ~0.41 and rollout_goal_dv < 0 and worsening — the policy exploits the
  frozen one-step lookahead faster than any patch retrains it. And with W_hat
  unpinned, proposer_what left real support within 400k (4.5 vs max mostly
  < 0): analytic ascent reliably finds whichever head is LEAST grounded.
  v8 final @8M (job 710, completed): -160 (last 20 eps) — plateau held to the
  end, no late takeoff. Best in family; still ~two orders below PPO baseline.
- `ppo_continuous_action_embopt_v9.py` — H-step unroll + commanded-goal TD
  (job TBD). Four changes, all aimed at the same failure (frozen-surface
  exploitation), no new aux losses:
  (1) Policy = SVG(8): L_pi = -[sum gamma^i r_hat(z_i,a_i) + gamma^8 V(z_8,g)]
  through frozen f — compounding model error makes fantasy self-punishing;
  8 steps of r_hat credit puts gait inside the policy's own horizon (v8's
  policy_rhat ~ -0.9 showed one-step greedy has no gait to find).
  (2) Proposer = goal-SEEKER u(z') -> action unrolled 16 steps through frozen
  f; goal := imagined endpoint. Reachability is STRUCTURAL (on f's manifold,
  exactly 16 model-steps away by construction) — deletes the aspiration band
  and every V term in the score. Score = discounted imagined path reward +
  gamma^16 * window-sum-scaled support-clamped W_hat(endpoint).
  (3) Critic pruned to lambda-return main + cross-trajectory negatives.
  (4) COMMANDED-GOAL TD (answers the standing "accepted gap"): segments are
  also relabeled with the goals the policy actually pursued (stored in the
  ring buffer; 8 window-start columns beside the 8 hindsight columns), same
  lambda-return recursion with no arrival event — bootstrap from segment-end
  V, floor at episode cuts. Unfulfilled promises are devalued by REAL
  experience; V finally trains on the exact (z, g_commanded) pairs behind
  policy_next_v and rollout_goal_dv. Answered along the way (user Q): obs/rew
  wrapper EMAs have been gone since v5 (raw storage + current-stats renorm);
  unfreezing f+V inside the policy step would be wireheading (policy gradient
  would train V to lie) — the fix is V learning FROM rollout behavior, which
  is (4). New telemetry: imag_err_h (imagined vs realized 8-step unroll under
  executed actions); proposer_v is now a free V-calibration probe (structural
  16-step goals => honest V should read ~gamma^16 ~ 0.72).
  Result @2.4M (job 711): BREAKTHROUGH — first locomotion in family history.
  Return -453 -> +1154 (vs v8 terminal -160; family had never crossed zero).
  Attribution from telemetry: policy_rhat flipped -0.9 -> +0.8-0.9 (the
  SVG(8) path credit found a gait one-step greedy could not represent);
  the optimism gap collapsed (policy_next_v 0.09 vs realized 0.04, was
  0.70 vs 0.41) — commanded-goal TD devalued unfulfilled promises exactly
  as designed; imag_err_h 0.15-0.30 vs delta-var 1.8 (model valid at the
  policy's operating point). Note the goal channel is now honest but
  MARGINAL: action_grad_value 0.04 vs action_grad_rhat 0.53, proposer_v
  0.035 (V calls seeker endpoints near-unreachable — over-contraction by
  the no-arrival recursion, or seeker-imagined endpoints diverge from
  what policy trajectories reach). The unroll's reward term is carrying
  the run; the goal machinery is no longer harmful but not yet helpful.
  Open question for v10+: make the goal channel earn its keep or prune it.
  Standing user concern (frozen f): freezing is per-phase only (f gets 16
  fresh MSE updates each iteration before the policy differentiates it);
  policy-loss gradients into f would train it to lie (predict transitions to
  high-V states), so the separation stays. The actual risk — policy operating
  where f is wrong — is instrumented (imag_err_h, optimism gap), currently
  healthy; if it re-emerges, candidate mitigations are f trained
  preferentially on the policy's imagined-state distribution or an f-ensemble
  with gradient through the min.
- `ppo_continuous_action_embopt_v10.py` — v9 cost recovery, mechanism intact
  (user: SPS 7800 -> 6000 unacceptable). One H=8 unrolled update carries ~8
  depths of action credit per backward, so v9's 16 updates were ~8x v8's
  per-iteration signal; halving still leaves ~4x. policy_updates 16 -> 8,
  proposer_updates 8 -> 4, and the rollout proposal is computed only on
  actual refresh steps (the old torch.where pattern ran the 16-step proposal
  unroll under CUDA graphs EVERY env step, wasting 15/16 of them; split into
  rollout_act + rollout_propose compiled separately). Semantic delta: env
  resetting mid-window keeps its stale goal <= 15 steps (~1/1000 steps on
  HalfCheetah). Expected ~7000-7400 SPS. Queued --after-success v9 (711) to
  keep the v9 8M benchmark solo and clean. CANCELLED before start (job 712):
  user redirected — SVG unrolling is not this family's mechanism.
- `ppo_continuous_action_embopt_v11.py` — unrolling removed, honest V kept
  (user direction: "SVG is not the answer; unrolling is a different thing we
  don't want to touch here", after v9 proved unstable: +1154 @2.4M but
  523 +/- 118 @4.75M — 8-step gradient chains through a learned f with no
  trust region are high-variance improvement steps). v9 changed two things;
  only one was the unroll. v11 keeps COMMANDED-GOAL TD (the piece that
  collapsed the optimism gap and matches the family spec) and restores the
  ORIGINAL one-step objective L_pi = -[r_hat + gamma*V(f(z,pi),g)] and the
  v6 shell/band proposer. The band's V is now honest, closing the grounding
  loop v6-v8 lacked: fantasy proposal -> commanded -> fails -> devalued by
  real experience -> band pushes proposer away. Attribution ladder this
  completes: v8 = one-step, no honest V (-160); v9 = unroll + honest V
  (~1000, unstable); v11 = one-step + honest V. Family thesis on the line:
  long-horizon credit belongs in V, not in unrolling. imag_err_h kept as a
  pure telemetry probe (imag_probe_h=8, no training unroll).
  v9 endpoint: cancelled at 6.2M per user direction. Final read: 1008 +/- 17
  (last 20) — matched steps 894 @2M / 823 @4M / 1031 @6M, i.e. noisy
  oscillation around ~900-1000 with a mild upward drift, not collapse; the
  523 +/- 118 snapshot @4.75M was a trough of that oscillation. Still the
  family record and proof the plateau was breakable — v11 tests WHICH half
  broke it, at one-step variance. v11 = job 714, launched clean.
  v11 verdict @2.2M (cancelled): -162 = the v8 plateau. Honest V alone does
  NOT restore locomotion; the unroll was the load-bearing half. Component
  audit (user's core-idea framing): V trains AND tests well (no optimism
  gap); the POLICY's one-step signal is starved — honest V of a 16-step goal
  changes little per action, the sigmoid floor attenuates it ~2x more, and
  r_hat drowns it 6:1 (action_grad 0.58 vs 0.09) while one-step r_hat greed
  has no gait in horizon (policy_rhat -0.88, exactly v8); the proposer is
  ground into conservatism by the honest test it cannot pass (proposer_what
  ~ mean rate, proposer_v 0.25 << 0.72 and falling — the grounding loop
  works, but grounds the proposer into the dirt when the policy can't
  deliver). f itself is fine (imag_err_h 0.05 vs 1.5).
- `ppo_continuous_action_embopt_v12.py` — DISTANCE-SPACE CRITIC (job TBD).
  Fix the starved policy signal while staying strictly one-step: the value
  head's GEOMETRY was hostile to its main consumer. d_hat(z,g) =
  softplus(critic + d_init) predicts steps-to-go; V := gamma^d_hat kept for
  bootstrap/telemetry. Same lambda-return recursion, targets converted once
  (d = log G/log gamma; arrival 0, floor 128), Huber in step units;
  negatives -> d = 128. Policy: L = -[r_hat - d_hat(f(z,pi),g)] — "one step
  closer" and "one step of normalized reward" share units, and grad(d_hat)
  is range-independent (no sigmoid tail). Proposer band in d-space:
  (clamp(W)-r_lo) - 4*((d-16)/16)^2. New "test them well" telemetry:
  dist_calib_bias/abs (hindsight pairs have exact known k = j - t) and
  goal_promise vs goal_delivery (promised W_hat(g) vs realized hold rate).
  Result @1.85M (job 715, cancelled): "better in some ways" (user) — returns
  touched -113 (best one-step at matched steps) but oscillated; SPS 10800;
  the starvation FLIPPED (action_grad_value 9.9 vs rhat 2.2). But the new
  calibration probe caught a structural flaw: dist_calib_bias +77 — the head
  saturated at d ~ 127 everywhere, so the strong gradient was noise. Root
  cause: commanded columns are a no-arrival gamma-RATCHET (targets can only
  decay; V can never be proven right about a proposed goal, only wrong) and
  negatives also say d=128; in gamma-space those floor labels carried tiny
  MSE gradients (benign), but d-space Huber gives every sample unit gradient
  — label volume (~75% "far") collapsed the head onto the majority label.
  LESSON (units): a reparameterization changes WHICH ERRORS GET HEARD, not
  just gradient geometry — rebalance label sources when changing spaces.
- `ppo_continuous_action_embopt_v13.py` — evidence-weighted critic: hindsight
  pairs (exact k) weight 1.0; commanded pairs (prior, not evidence — the
  ratchet structurally cannot register success) cmd_weight 0.15; negatives
  coef 0.05. sample_pairs returns the is-hindsight mask; Huber
  reduction="none" + weighted mean. Honest-devaluation loop kept at reduced
  gain; exact labels reclaim the head.
  Result @1.6M (job 716, running to 8M as benchmark): CONSISTENTLY LEARNING
  (user-confirmed) — returns climbing through -32 (best one-step), the
  promise/delivery loop functions for the first time (promise 0.46-0.50
  ABOVE mean rate 0.13, delivery 0.20-0.23 following), proposer_v on the
  0.72 band target, policy_rhat crossing positive, SPS 11000. Calibration
  improved but still the bottleneck: bias +77 -> +13, abs ~28 on mean-45
  distances — hindsight training targets are lambda=0.95 BOOTSTRAP-diffused
  while the exact label k = j - t sits unused.
- `ppo_continuous_action_embopt_v14.py` — exact-k hindsight targets (the
  spec's lambda=1 / Phase-A special case, applied exactly where evidence is
  exact): after the recursion, hindsight-column targets := gamma^(j - t) —
  pure regression to realized path distance (V^pi semantics). Commanded
  columns keep the lambda-bootstrap ratchet at cmd_weight 0.15. Expected:
  dist_calib_abs -> single digits so the policy's dominant d-gradient
  (12 vs 2) points at truth. Chained --after-success v13 (716).
- HINDSIGHT BRANCH VERDICT + DIRECTION CHANGE (user): v13 finished 8M FLAT
  (final -3.9; @2M/4M/6M/8M all within -20..+20 — the 1.6M "consistently
  learning" read did not extrapolate; job 716 succeeded). v14 tracked it at
  2.1M (~-64, oscillating -370..+183; job 718 cancelled). User direction:
  NO hindsight relabeling, NO counterfactual constructs of any kind (the
  briefly-drafted v15 "real-anchored" policy step z' = E(o_{t+1}) +
  f(z,a) - f(z,a_exec) was rejected and deleted before ever running).
  The mental model going forward: take an action, observe z_{t+1}, and let
  every critic label be a fact about that executed transition.
- `ppo_continuous_action_embopt_v15.py` — TD(0)-on-executed-transitions
  critic; the entire relabeling tower deleted (ring-buffer segment
  recursions, sentinel arrival indices, lambda, evidence weights, negatives,
  hindsight args). Three fact losses, Huber in d-space:
    TD(0): d(z_t, g_cmd) <- 1 + d(z_{t+1}, g_cmd), g_cmd = goal actually
      pursued (buf_goal); bootstrap detached AND capped at num_steps=128
      (1 + d is not a contraction and commanded goals have no arrival event;
      the cap defines the head's range [0,T] by label construction).
    one-step: d(z_t, z_{t+1}) -> 1 (the user's "V(z, z+1)").
    identity: d(z, z) -> 0 (arrival semantics WITHOUT an arrival event:
      [z,g,g-z] with small g-z is the anchors' own input region, so
      near-arrival generalizes; TD chains terminate through it).
  Episode cuts masked, not floored. Goal-conditioned phases (critic,
  policy) sample only the 8 most recent segments — buf_goal stores latent
  COORDINATES and the frame drifts under WM training (review finding; in
  v13 stale goals were the 0.15-weight minority, here they'd be 100% of
  labels). Policy goals: fresh proposals + stored commanded. dist_calib_*
  is now a PURE MEASUREMENT (observed z_t vs z_{t+k}, k in {2,4,8,16},
  never trained on) — the first fully independent critic test. New logs:
  losses/critic_onestep, losses/critic_self. Review flagged (accepted
  risks, watch): per-goal gauge freedom off-manifold is only bounded by
  the cap + anchors; TD(0) propagation is slow (1 step/update vs dense
  multi-step hindsight labels). Job 719, 8M HalfCheetah.
- v15 VERDICT (job 719, cancelled @2.3M): TRIVIAL-FIXED-POINT COLLAPSE, the
  mirror image of v13's ratchet. d_hat -> 0 everywhere within 400k
  (critic_self exactly 0, td/onestep parked at Huber 0.5 = constant
  off-by-one, critic_v_mean pinned at 1.0, action_grad_value exactly 0,
  goal channel dead, returns at the r_hat-greedy plateau ~-150). LESSONS:
  (1) the absolute level of a goal-distance critic is NOT identifiable from
  facts about executed transitions — TD(0) is a pure difference constraint
  and the constant solution is an attractor once every far-label source is
  removed; (2) latent geometry: wm_delta_var ~1.4/dim x 64 dims => per-step
  ||dz||^2 ~ 91 vs typical random-pair distance^2 = 2*dz = 128 — the
  SIGReg'd latent mixes in ~2 steps, so NO labeling scheme can calibrate a
  multi-step absolute distance head here; only ~1-step structure exists.
- `ppo_continuous_action_embopt_v16.py` — DELTA-ONLY VALUE, PER-STEP GOALS
  (user direction: policy optimizes how well it optimizes embed MSE to the
  goal; goals re-proposed every step for max signal; no absolute head).
  Design evolution within v16 pre-run (review-driven): a 16-step
  window-progress head was caught degenerate pre-submission (endpoint
  mse mixed out => h ~ mse(z,g) - const => "farthest goal wins"); fixed by
  the per-step horizon, where the target's g-dependence is CAUSAL.
    h(z_t,g_t) <- mse(z_t,g_t) - mse(z_{t+1},g_t)   (g = commanded at t;
      pure outcome regression, no bootstrap; random motion INCREASES
      expected MSE so unaligned goals earn negative labels)
    L_pi = -[r_hat + (mse(z,g) - mse(f(z,pi),g)) + h(f(z,pi),g)]
    proposer score = W_hat(g) + h(z,g), UNCLAMPED — support clamp replaced
      by grounding W_hat at its query points: what(g_cmd) <- delivered rate
      (promise-vs-delivery as training, not telemetry). Standard scalar
      critic per user; the v8 unpinning failure is answered by training
      where we query instead of clamping.
  Pool = K*T*E ~ 16k measured pairs/iter (recent 8 segments only).
  Jobs: 720 (per-step, gr=1) + 721 (--goal-refresh 16 window ablation,
  expected to reproduce the reviewed degeneracy — cheap falsification).
- v16 VERDICT (jobs 720 per-step / 721 gr16 window, both cancelled @1.1M):
  per-step spiked -369 -> -67 @512k then flatlined ~-100; gr16 strictly
  worse (-200, confirms per-step; window variant retired for good).
  Root cause of the flatline — THE TREADMILL EXPLOIT: measured one-step
  progress scales with goal distance (Delta-mse ~ (g-z).dz), so the
  additive h term in the proposer score buys distance, not value. Proposer
  proposed anti-aligned far goals (rollout_goal_dist ~3.0 vs random-pair
  baseline 2.0) and SOLD the rate term (proposer_what negative while
  proposer_prog 1.5); policy flailed to generate latent displacement
  (policy_rhat pinned -0.7). All progress labels were honest — the
  objective was wrong. LESSON: reachability is not a reward; it must enter
  only as grounding (promise dragged to delivered) or a constraint, never
  as an additive score term with open-ended magnitude.
- `ppo_continuous_action_embopt_v17.py` — proposer score = grounded
  W_hat(g) ONLY (unclamped; what(g_cmd) <- delivered rate keeps it honest).
  h keeps its two proper jobs: policy-loss local credit + telemetry.
  Expected if right: rollout_goal_dist <= ~2, goal_promise positive with
  delivery chasing. Jobs: 722 (default) + 723 (--value-coef 0.5 probe:
  v16 action-grad balance showed the goal term dominating r_hat ~2.6:1).
- `ppo_continuous_action_embopt_v18.py` — v17 + review-driven derivative
  grounding (submitted concurrently with v17 as a probe): (1) GOAL NOISE
  (goal_noise 0.25, shell re-projected; the noisy goal IS the commanded
  goal, labels stay factual) — v17 trains h and the delivery map only on
  the 1-D manifold (z, propose(z)), so every dg-gradient the proposer
  ascends points where no data varies (DDPG no-action-noise pathology);
  (2) SEPARATE goal_rate_head gwhat(g) <- delivered rate; proposer score =
  gwhat only (v17's shared rate_head made state-rate and goal-delivery
  regressions collide at ||x|| ~ 8, blurring exactly the surface the
  proposer ascends). Job 724, alongside 722 (v17) and 723 (v17_vc05).
- v17/v18 VERDICT (jobs 722/723/724, all cancelled @~2.2M): the treadmill
  fix WORKED (rollout_goal_dist back to ~2.0 from the 3.0 exploit; v18's
  goal noise + separate delivery head changed nothing vs v17 — within
  noise on every signature), but all three flatlined at ~-200/-170.
  Fine-grained curves overturned the "flatline" framing: each run LEARNED
  TO RUN and was then destroyed — v17 hit -41 @96k (crashed by 144k),
  vc05 hit +327 @208k (crashed by 240k), then a dead ~-200 plateau.
  Two structural causes diagnosed (user-directed on both):
  (1) SUCCESS-COUPLED UNITS: reward scale (buf_rew.std 0.7 -> ~0.15 right
      at the spike) deflates every normalized-reward target 5x exactly
      when the policy finds something worth keeping, while the goal term
      keeps fixed latent units and takes over; the obs RMS shifts the
      encoder's input frame at the same moment (velocity stats jump).
      The policy's own success rescales its objective and moves its frame.
      LESSON: no running normalization anywhere in this family — success
      must not change the units of the objective.
  (2) MODEL-ERROR CROSS-TERM EXPLOIT: the policy goal term read progress
      through f; progress is LINEAR in prediction error while wm_pred is
      QUADRATIC (err 0.016/dim at dist^2 ~ 2 admits ~0.36/dim fake
      progress once ascent aligns f's error with (g - z')). Observed:
      policy_prog 0.46-0.60 vs 0.03-0.06 measured (11-15x over-claim), a
      drag of equal action-gradient magnitude to r_hat (~1.0 vs ~1.0);
      vc05 (half goal weight) doing better confirmed the term was net
      harmful. LESSON: never put the dynamics model between the action
      and a maximized score; small MSE does not mean small exploitable
      bias. (Answers the user's value_coef question: it is the exchange
      rate between r_hat and the goal pull on the same action — it only
      "matters" while the goal term is dishonest.)
- `ppo_continuous_action_embopt_v19.py` — STATIONARY UNITS + REGRESSION-
  ONLY GOAL CREDIT (user: "should be able to do regression", "reward ema
  and obs ema need to be axed"). Axed: obs RMS (raw obs; encoder LayerNorm
  absorbs scale), reward/rate scaling (raw targets), v18 goal noise
  (v18==v17 empirically; with per-step re-proposal the delivery map
  already trains exactly where the proposer queries). Replaced the policy
  goal term with an action-conditioned regression head:
    q(z,a,g) = critic([z,g,g-z,a]) <- measured progress-to-go of the
      EXECUTED transition (same pool as h, now action-conditioned; the
      exact rhat pattern — expl_noise ball keeps dq/da labeled at the
      policy's actions; warmup's uniform actions are free action coverage)
    L_pi = -[reward_coef * r_hat(z,pi) + value_coef * q(z,pi,g)]  (no f)
  Expected if right: no crash after the first spike (stationary units);
  policy_prog ~ rollout_goal_prog (honest, no over-claim). Jobs: v19
  default + --value-coef 0 control (first clean reward-only baseline of
  the family: does the goal channel help AT ALL under honest units?) +
  --value-coef 4 probe (goal term likely under-weighted in raw units).
  Post-review fix before submission: separate clip budgets for encoder+dyn
  vs the reward-unit heads (shared global clip would have relocated the
  success-coupling — raw-reward residuals grow ~10x with returns and would
  throttle the encoder's LeJEPA gradient through the shared norm).
  Jobs: 725 (v19) + 726 (vc0 control) + 727 (vc4 probe).
- v19 VERDICT (jobs 725/726/727, cancelled @~3.2M): the stationarity fix
  held as designed (no unit-driven crash; vc0 perfectly stable) and the
  coefficient sweep became a clean CAUSAL experiment on the goal channel:
    vc0 (goal off): monotone -377 -> -171, ZERO swings — but plateaued;
    vc1: repeated spike-crash cycles (-123 -> -482, -45 -> -500);
    vc4: dormant at -516, a -1068 excursion, +60 @2.2M, crash to -600.
  Instability monotone in value_coef under stationary units => the goal
  channel itself destabilizes. Mechanism: SPURIOUS ATTRIBUTION — the
  delivery map credits commanded goals with delivered rate, but measured
  rollout_goal_prog ~ 0.001-0.01 (the policy achieves ~zero real progress
  toward goals). Success gets attributed to whatever goals were fashionable,
  the proposer chases them, goals shift, and the goal-dominated action
  gradient (0.7-3.2 vs reward's 0.1-0.7) drags the policy off its gait.
  Also: q(z,a,g) still over-claims (policy_prog ~0.32 vs ~0.01 measured) —
  bounded by re-labeling but not closed (ball-edge extrapolation).
  vc0's stability exposed the complementary gap: r_hat is IMMEDIATE reward;
  pathwise ascent on it is myopic (cannot learn momentum-building), hence
  the -175 plateau. LESSONS: (1) a goal-value map is only as good as the
  causal link goal->behavior; if measured goal-following is ~zero, delivery
  regression is pure confounding and the proposer amplifies noise;
  (2) the family needs a long-horizon credit channel that does not route
  through goals.
- `ppo_continuous_action_embopt_v20.py` — LONG-HORIZON REWARD CREDIT BY
  REGRESSION; GOAL TERM DEFAULT OFF. New head qrate(z,a) <- buf_rate
  (measured mean reward over next rate_window=16 executed steps; the
  what(z) target, action-conditioned; rhat pattern at horizon 16 — no
  bootstrap, no TD).
    L_pi = -[reward_coef*rhat + rate_coef*qrate + value_coef*qprog], vc=0
  Goal machinery keeps training (telemetry + future re-entry). Probes:
  default (rate on, goal off), --rate-window 64 (horizon probe), and
  --value-coef 0.5 (does a small goal term help on top of qrate?).
- RAW BACKPORTS (user direction): `_v13_raw` / `_v14_raw` / `_v16_raw` —
  byte-identical algorithms to v13/v14/v16 with ONLY the v19 stationarity
  fixes applied: no obs RMS (raw obs), no reward scaling (raw targets),
  split clip budgets (encoder+dyn vs reward-unit heads). Rationale: those
  versions posted the family's best transients (v14 +183 @1.7M, v16 -67)
  UNDER success-coupled normalization that then destroyed them — this
  isolates how much of their failure was the units, not the algorithms.
  Caveat to watch: raw units shift each version's internal term balances
  (policy reward-vs-value, W_hat clamps trained in normalized units).
  Jobs: 728 (v13_raw) + 729 (v14_raw) + 730 (v16_raw). v20 (qrate) held
  ready, queued after these.
  v20 post-review fixes: (1) action_grad_* diagnostics now log UNSCALED
  grad norms (value_coef=0 was silencing exactly the goal-channel gradient
  curve needed to decide re-entry); (2) qrate regression masks segment-
  tail steps whose forward window is truncated below rate_window (~12.5%
  of labels were short-horizon; tolerable for telemetry surfaces, not for
  a head feeding the policy gradient). Jobs: 731 (v20) + 732 (rw64) +
  733 (vc05), queued behind the raw backports 728-730.
- `ppo_continuous_action_embopt_v21.py` — PERSISTENT EXPERIENCE-ANCHORED
  GOALS (a-priori redesign; user's human-goal analogy: goal = predicted
  best joy-per-time destination, refining as the world model improves,
  re-evaluated pursuit not re-invented goal, deterministic best-next-step).
  Mechanism: goal = REAL replay state (top measured rate -> best what(z))
  stored as its raw OBSERVATION, re-encoded per iteration (fixed referent,
  refining representation — also kills stale-latent-goal drift at the
  root), switched only by hysteresis (challenger must beat incumbent's
  predicted rate by goal_switch_margin). Proposer net, shell projection,
  gwhat delivery head all REMOVED — the state-rate map what(z), grounded
  densely on real states, is the goal evaluator. Persistence fixes label
  coherence: v16-v20 re-invented goals per step so every progress label
  answered a different pursuit question (rollout_goal_prog ~0.01).
  L_pi = -[rhat + qrate + q(z,a,goal)], goal term ON (vc=1).
  Controls: job 733 (v20_vc05, old shell-goal machinery on same chassis)
  is the direct A/B for the goal redesign. Early raw-backport signal:
  v13_raw hit -106 @2.4M — already above every EMA-era plateau.
  v21 post-review fixes: (1) truncated-window rates masked to -inf before
  the candidate topk (max over heteroscedastic estimates is biased toward
  noisy segment-tail entries; the regression mask had to reach selection
  too); (2) fantasy guard — incumbent's self-evaluation capped at the best
  MEASURED recent rate (once its segment ages out of the ring, only
  extrapolation constrains what() at the anchor; uncapped, a high-fantasy
  incumbent locks in forever). Jobs: 734 (v21) + 735 (v21_rc0, rate_coef 0
  — goal channel as the sole long-horizon signal), alongside control 733
  (v20_vc05, old shell goals).
- RAW-BACKPORT VERDICT (jobs 728/729/730; NOTE: killed at ~5.5M by an mlq
  protected-job quirk when queued v20 jobs were cancelled — not a judgment
  call; 5.5M suffices for the verdict): stationary units alone do NOT
  rescue the old goal designs. v13_raw -103 @2M then declined to ~-222
  (its ratchet pathology persists in raw units); v14_raw oscillated
  (-176 @2M, -61 @4M, -107 @5.5M, CI +/-54 — the old spike-crash cycle,
  slower); v16_raw stable but stuck ~-162. None broke out. Confirms the
  v19 causal sweep from the other direction: the EMA fix was necessary
  (raw variants all beat their EMA-era plateaus) but NOT sufficient — the
  shell-goal machinery itself is the remaining destabilizer.
- v21 RETIRED PRE-DATA (user correction; jobs 734/735 cancelled early):
  replay-anchored goals are a museum curator — they cap the goal at the
  best REMEMBERED state and structurally cannot represent the 20k-return
  state nobody has visited; the doctor prediction is made WITHOUT having
  been a doctor. Hysteresis imposed fake persistence: humans re-derive
  the goal continually from a STABLE BELIEF; persistence must emerge from
  belief stability. Per-step goal re-invention is correct; what thrashed
  in v16-v19 was the SCORING (delivered-rate association), not the
  re-invention.
- `ppo_continuous_action_embopt_v22.py` — ASPIRATIONAL GOALS: proposer
  net back, re-proposing EVERY step, scored by the BELIEF map itself:
  score = what(g) (predicted rate OF the goal state, shell-ascended,
  extrapolating beyond experience). No replay anchor, no hysteresis, no
  delivery head. Grounding = pursuit loop (what regresses measured rates
  at real states; pursuit drags the visited distribution toward the goal
  region, correcting belief where the goal lives; "doctor ->
  anesthesiologist" = the argmax sharpening as what trains). Watched
  risk: goal_promise >> what_real_max persistently = runaway fantasy.
  L_pi = -[rhat + qrate + q(z,a,g)], vc=1. Control: 733 (v20_vc05,
  delivered-credit scoring, same chassis).
- v22 REVIEW (pre-launch): no coding defects. One structural risk flagged:
  proposer objective -what(propose(z)) has NO z-dependence, so its optimum
  is a CONSTANT shell point g*; and unlike v16-v19's gwhat (regressed at
  commanded goals toward delivered rates = negative feedback), NOTHING
  trains what() at proposed goals — collapse pressure kept, correction
  deleted. If pursuit never closes the distance (v20 measured
  rollout_goal_prog ~0.001-0.01), what(g*) is never corrected and
  goal_promise ascends unbounded -> constant-direction action bias =
  v17-style crash signature. DECISION: test the extrapolation thesis
  UNGUARDED (guards all trade it away: gwhat = the rejected scoring;
  support clamp = the rejected v21 aspiration cap; goal noise = smearing)
  but with a hard early-kill criterion at ~300k: kill if proposer_out_std
  decaying toward 0 AND goal_promise diverging above what_real_max AND
  rollout_goal_prog pinned at ~0. Persistence of ONE goal is not itself
  failure (doctor analogy: stable belief -> stable goal); UNGROUNDED
  promise growth is.
- v20_vc05 VERDICT (job 733, full 8M, delivered-credit control): stable
  but flat at the myopic plateau — -273@2M -> -243@4M -> -184@8M, final
  mean -184 (vs v19 vc0 rhat-only plateau ~-175). No crash (stationary
  units held), but the goal channel did NOTHING REAL: rollout_goal_prog
  -0.002 (zero actual pursuit) while policy_prog 0.65 (critic claims
  large progress at policy actions) — the spurious-attribution /
  extrapolation over-claim signature again, just too weak at vc=0.5 to
  destabilize. goal_promise -0.19: delivered-credit scoring learned goals
  are worth nothing, confirming it cannot express aspiration. Chassis
  (stationarity + qrate) is sound; delivered-credit goal SCORING is a
  dead end -> v22's belief scoring is the live hypothesis.
- v20_vc05 ADDENDUM (reviewer): policy_prog 0.65 vs rollout_goal_prog
  -0.002 are the SAME quantity measured two ways (qprog at policy actions
  vs realized progress under executed actions) — a ~300x over-claim
  exactly where the policy queries. Structurally the same failure as
  v19's forward_dyn cross-term (0.46-0.60 claimed vs 0.03-0.06 measured):
  swapping the dynamics model for a regression head moved the over-claim
  from model error to EXTRAPOLATION error, reducing neither direction nor
  magnitude. Reading v22's telemetry: prog_calib_bias/abs measure qprog
  at EXECUTED (in-distribution) actions; if calib ~0 while policy_prog >>
  rollout_goal_prog, the head is fine where it trains and fantasises
  where the policy queries it — a CRITIC-exploitability failure,
  independent of Finding 1's proposer collapse (which the out_std /
  promise / goal_prog triple catches). The two failures point to
  different v23s: fix the proposer's grounding vs fix the critic's
  off-policy over-claim.
- v22 CHECKPOINT (job 736 @2.5M): TAKEOFF. last-20 return +575 (CI ±26)
  at 2.54M, up from -267@2M — sharpest and largest improvement in family
  history (prior best: +327 transient pre-crash v17_vc05; +60@2.2M v19
  vc4). Finding 1's proposer collapse DID occur (proposer_out_std ~0.004
  = constant goal) and the kill triple technically fired (goal_promise
  4.7 vs what_real_max 1.1, rollout_goal_prog ~0), but OVERRIDDEN: the
  criterion existed to abort a doomed run, and this one is succeeding —
  goal_delivery rose -0.34 -> +0.62 across 1.7M->2.5M, what_real_max
  0.72 -> 1.12, promise-real gap stable (~3.6) not diverging, policy_rhat
  and policy_qrate flipped negative -> +0.57/+0.58 with real reward.
  Constant goal = persistence from stable belief (the design intent);
  what collapsed is per-step VARIETY only. Critic over-claim signature
  present (policy_prog 0.33 vs realized ~0; prog_calib_bias -0.002
  in-distribution) yet not blocking learning. Open question: is the
  fixed aspirational pull causal for the takeoff (directed-exploration
  bias toward a believed-high-rate state) or is rhat+qrate doing the
  work? Watch to 8M for the family's signature post-spike crash;
  stationary units should prevent the v17 mechanism.
- `ppo_continuous_action_embopt_v23.py` — PESSIMISTIC TWIN PROGRESS
  CRITIC + REACHABILITY-AWARE ASPIRATION. Fixes the two structural
  defects v22 succeeded DESPITE: (1) qprog extrapolation over-claim at
  policy queries (policy_prog 0.33 vs realized ~0, calib clean
  in-distribution) -> twin heads critic/critic2 regressed on the same
  measured targets from different inits, ALL consumers (policy, proposer,
  telemetry) read min(q1,q2); pure regression pessimism, no
  bootstrap/target nets; new diagnostics/qprog_gap = |q1-q2| at policy
  queries. (2) proposer z-independence -> constant-goal collapse
  (out_std ~0.004) -> L_prop = -[what(g) + prop_reach_coef *
  min-q(z, act(z,g), g)] (prop_reach_coef=1.0): aspiration must also be
  pursuable from HERE; reachability read through the min so the proposer
  cannot harvest critic fantasy. Hypothesis: v22's takeoff under a
  constant fantasy pull + exploitable critic is a LOWER BOUND; decisive
  metric is rollout_goal_prog lifting off ~0 (flat zero in every version
  to date). Probe: job 737 (embopt_v22_vc0) = goal-channel causal
  ablation on the unchanged v22 chassis.
- v22 FINAL VERDICT (job 736, 8M complete): +584 last-20 (571 CI±19) —
  FAMILY BEST by a wide margin. Trajectory oscillated (+441@4M, -252@6M,
  -103@7M, +584@8M): instability remains but RECOVERS — stationary units
  prevent the permanent v17-style crash; the excursions likely track the
  exploitable-critic pull. goal_promise drifted 4.7 -> 5.7 vs
  what_real_max 1.2 (fantasy gap slowly widening but bounded-ish);
  proposer stayed collapsed (constant goal) throughout the success.
- v22_vc0 CAUSAL VERDICT (job 737, cancelled at 5.1M): goal term OFF on
  the identical chassis = FLAT -149 for 5M straight (the rhat+qrate
  myopia plateau). THE GOAL CHANNEL IS CAUSAL for the takeoff: the
  constant aspirational pull toward a believed-high-value shell point IS
  the exploration mechanism (+584 vs -149). LESSON: a fixed
  fantasy-direction bias, even with near-zero per-step realized goal
  progress, breaks the pathwise-myopia plateau that no facts-only action
  channel escaped.
- USER CLARIFICATION (belief-map semantics): the goal score is NOT the
  per-step reward rate AT the goal state; it is REWARD EFFICIENCY —
  cumulative reward collected en route to the state divided by steps to
  reach it ("joy per use of my time"). Also: not sold on multiple
  critics or the q machinery generally — v23 may run as a probe, but the
  direction is efficiency-scored aspiration, not critic hardening.
- `ppo_continuous_action_embopt_v24.py` — REWARD-EFFICIENCY BELIEF MAP
  (mainline, user-directed). v22 chassis untouched (single q, belief-only
  proposer — the config that took off); ONLY the what() target changes:
  eff(s_t) = (sum of rewards from episode start to arrival) / t, masked
  at reset states; accumulators ep_cum/ep_len zeroed on done. Efficiency
  prices travel cost INTO the aspiration score itself — the principled,
  critic-free version of v23's bolt-on reachability term. goal_delivery
  now = realized segment efficiency (same units as promise). Jobs:
  738 = embopt_v23 (critic-machinery probe), 739 = embopt_v24 (mainline).
- v23 REVIEW + EARLY DATA (job 738 kept running): review found (1) twin
  heads share ONE clip budget -> each head's step ~29% smaller than v22's
  (0.5/sqrt2) — a real A/B confound, noted here, not fixable mid-run;
  (1b) losses/critic now sums both heads (~2x v22's scale — not a
  regression when overlaying curves); (2) the reachability term is
  STRUCTURALLY v16's treadmill score (progress linear in (g-z).dz, so
  magnitude is bought by DISTANCE) and twin-min gives NO protection —
  the treadmill is honest, densely-labeled signal both heads agree on;
  watch proposer_what going negative + rollout_goal_dist > 2.0; (3)
  prog_calib_bias is now min-shifted by -(in-distribution gap)/2 BY
  CONSTRUCTION (E[min] = mu - E|X1-X2|/2) — a negative reading is the
  pessimism working, NOT decalibration; v22's "calib ~0" will not
  reproduce. EARLY DATA (928k) contradicts the treadmill prediction:
  proposer_what +6.4, rollout_goal_dist 1.66 < 2.0 random baseline,
  proposer_out_std 0.22 (z-dependence restored, no collapse), and
  rollout_goal_prog 0.167 — FIRST-EVER liftoff of realized pursuit from
  ~0 in family history. Return -466@720k -> -46@928k and climbing.
  Meanwhile v24 (efficiency map): +417@736k — v22 needed ~2.3M to go
  positive. If v23 merits iteration: per-head clips, qprog_gap_data
  (executed-action gap baseline), goal_grad_belief/reach norms.
- v24 REVIEW (post-launch, fix-forward): NO correctness bug — accumulator
  ordering, autoreset/truncation semantics, arity, units all verified
  correct. Actioned in-file for next iteration (job 739 unchanged):
  losses/rate_mse renamed losses/eff_mse (tag collision would silently
  compare efficiency-MSE vs forward-rate-MSE across runs — read 739's
  rate_mse tag with that caveat); new diagnostics/eff_var (eff_mse /
  eff_var ~ 1-R^2: -> 1 means the efficiency target is unlearnable from
  z and what() collapsed flat); rate_window docstring fixed. FLAGGED
  RISK: since-episode-start 1/t averaging makes labels heteroscedastic
  and washes out the state-local component at large t (HalfCheetah
  always truncates at exactly 1000) — belief map may compress. Fallback
  if flat: fixed-length BACKWARD window (mean reward over last W steps
  en route) — still "reward paid per step to get here", but state-local
  and TimeLimit-independent. Early read @2.5M: what spread (-0.17 mean,
  0.61 max) NOT flat; goal_promise 1.39 vs v22's 5.7 — efficiency
  naturally tempers fantasy, as intended.
- 2M+ CHECKPOINT (jobs 738/739 running): v24 -394@500k -> +539@2M ->
  EXCURSION -334@2.5M (v22's 6M excursion arriving earlier; v22
  recovered — watch). v23 STEADY: -481@500k -> +143@2M -> +239@2.6M, no
  excursion, rollout_goal_prog 0.42-0.51 (real pursuit, family first),
  proposer_out_std 0.45, qprog_gap 0.05. Hypothesis forming: the
  excursion cycles are critic-exploitation episodes; v23's pessimistic
  min + reachability-grounded goals damp exactly that. Verdict at 8M.
- REVIEWER CORRECTION + TWO CONFIRMED PREDICTIONS (mid-run, ~3.4M):
  (1) Treadmill call corrected: pessimism penalizes distribution SHIFT on
  the GOAL axis whether or not the quantity is honest — buying progress
  magnitude requires pushing g where the twins have no shared data, so
  the min prunes it; v16 had no such guard. Early v23 goals pulled
  INWARD (dist 1.66 < 2.0). BUT the guard decays: as data support
  widens, qprog_gap -> 0 and the min becomes a no-op — treadmill arrives
  LATE. CONFIRMED IN FLIGHT: v23 qprog_gap 0.055 -> 0.035, dist 1.66 ->
  2.31 -> 2.64 (through the 2.0 baseline), proposer_what 6.4 -> 1.78
  falling while proposer_prog climbs. Returns still rising (+248@3.5M)
  but the exploit is engaging. (2) v24 prediction: eff at large t is
  quasi-episode-constant -> what() partly fits EPISODE IDENTITY ->
  fast early gains (association is a good proxy early) then
  destabilization as success self-confirms — v24 crashed +539@2M ->
  -305@3M and is NOT recovering at 3.4M (v22's excursions recovered in
  ~2M). Discriminator added to v24 file: eff_var_between (episode
  component) vs eff_var_within (state component). v25 direction
  crystallizing: efficiency semantics with a state-local BACKWARD window
  (kills episode-identity association AND heteroscedasticity), plus
  v23's grounding if finals support it.
- CORRECTION + RETROSPECTIVE TEST (supersedes the "excursions =
  critic-exploitation, min damps them" hypothesis two entries up —
  that used the WRONG PAIR: v23 vs v24 differ in both critic machinery
  AND belief target, so steadiness cannot be attributed to the min):
  (1) Retrospective over-claim test through both excursions (logged
  policy_prog vs rollout_goal_prog): the gap is CHRONIC (~0.25-0.5
  throughout both runs) and does NOT spike at excursions — v24's gap
  SHRANK into its crash (0.39@1.6M -> 0.24@2.6M), v22's stayed ~0.45-0.5
  through excursion AND recovery. Critic over-claim is background, not
  the excursion trigger. (2) The clean one-variable A/B (v22 vs v24,
  same chassis, only what() target changed) attributes v24's
  faster-positive AND earlier/deeper/unrecovered crash to the
  EFFICIENCY TARGET: episode-identity association — strong early proxy,
  self-confirming destabilization later. (3) what_real numbers show a
  LEVEL SHIFT not compression (v24 spread 0.78 vs v22 0.67 — goal
  channel NOT starved); fantasy tempering is ESTABLISHED: extrapolation
  excess (promise - what_real_max) 0.78 vs v22's 4.47, a 5.7x reduction.
  Net v25 guidance: keep efficiency SEMANTICS (tempering is real and
  wanted), make the label STATE-LOCAL via a fixed backward window
  (kills the episode-identity channel — the actual crash mechanism);
  the min's value remains UNPROVEN pending a clean single-variable test.
- `ppo_continuous_action_embopt_v25.py` — SYNTHESIS (job 740). Four
  evidence-backed decisions: (1) belief target = BACKWARD-WINDOW
  efficiency (eff_window=16, full-window mask): state-local, kills v24's
  episode-identity association while keeping the established fantasy
  tempering; (2) progress target = PROJECTION (g-z).(z'-z)/(||g-z||
  sqrt(dz)) — reviewer design: the old distance-delta target was linear
  in ||g-z|| (v16 treadmill; v23's min only deferred it since the guard
  decays with data support), the projection divides the inflation out BY
  CONSTRUCTION and drops the goal-independent ||d||^2 noise term;
  horizon-matching was considered and REJECTED (fast-mixing latent:
  per-step ||dz||^2 ~ 91 vs random-pair 128, so H-step targets
  degenerate to pure distance = treadmill in pure form); (3) SINGLE q
  head — twins dropped deliberately: treadmill now structurally gone,
  chronic action-axis over-claim proven non-triggering, user not sold;
  (4) proposer = belief + reachability (v23's z-dependence fix, now safe
  under projection), goal_grad_belief/reach logged to measure the
  balance. UNITS WARNING (stated before first run per reviewer):
  rollout_goal_prog, policy_prog, prog_calib_*, losses/critic are in
  PROJECTION units — NOT comparable to any v16-v24 curve; value_coef and
  prop_reach_coef start at 1.0 and must be read against the grad-norm
  telemetry, not against old-unit intuitions.
- v25 amendment pre-data: projection denominator floor raised from 1e-6
  to 0.1*sqrt(dz) (10% of shell radius) — with reachability pulling
  goals toward reachable territory, g near z is a live regime, and a
  1e-6 clamp would make near-zero-gap targets explosive (~1e6 * proj)
  and sign-noisy; with the floor they go NEUTRAL (goal effectively
  reached). No effect on typical gaps (~sqrt(2*dz)). Job 740 cancelled
  pre-data, resubmitted as job 741.
- USER PRINCIPLE (design law going forward): multiple losses on the same
  graph = probably doing something wrong. The tell in v25: grad-norm
  balance telemetry and four blend coefficients — instrumentation that
  exists only because the objective was a committee.
- `ppo_continuous_action_embopt_v26.py` — ONE LOSS PER MODULE (job 742).
  Belief what(z) = the ONLY value function (backward-window efficiency).
  Proposer: max what(g), nothing else (WHERE; constant-goal convergence
  accepted as belief-persistence per v22's causal result). Policy: max
  qprog(z,a,g), nothing else (HOW; pure pursuit — reward reaches the
  policy exclusively through the goal). rhat/qrate heads and buf_rate
  machinery DELETED, not zero-weighted; reward_coef/rate_coef/value_coef/
  prop_reach_coef gone. wm phase: heads regress on detached z, so the
  summed backward routes exactly one objective per parameter set.
  Stakes the family thesis: the goal channel carries EVERYTHING. Risk
  accepted: no direct reward gradient on the policy — miscalibrated
  belief or pursuit has no safety net. Reference: v25 (blended
  objective, same chassis) running as job 741. v23/v24 FINALS for the
  record: -309 / -321 (both late-collapsed per the diagnosed mechanisms;
  v22's +584 stands).
- v25 REVIEW (no correctness bugs; jobs 741/742 unchanged, fixes applied
  to both files fix-forward): (a) rolling-window ordering verified
  correct incl. boundaries. (b) Cauchy-Schwarz caps |tgt| <=
  ||disp||/sqrt(dz) unconditionally — no blow-up regime existed. (c)
  REVIEWER SELF-CORRECTION: the raw projection is degree-ZERO in the gap
  (direction-only) — g~z is NOT repulsive; the gap-norm floor is what
  makes parked goals neutral and creates the interior optimum. The floor
  is LOAD-BEARING (marked so in code). (d) goal_grad_* must project out
  the RADIAL component (shell normalization annihilates it) — tangential
  projection now in both files; prior readings overstate belief. (e) NEW
  READING RULES: rollout_goal_dist is no longer a treadmill indicator
  (no distance reward; both shells at sqrt(dz) -> dist parks near 2
  regardless). Replacement: rollout_goal_prog / sqrt(wm_delta_var) ~
  mean cosine alignment in [-1,1]. NEW failure mode: alignment -> 1 with
  flat returns = self-consistent INERT fixed point (proposer points
  where the policy already goes; goal channel contributes nothing) —
  corroborate with goal_removal_action_mse. Especially critical for v26
  (pursuit-only policy). (f) DESIGN FINDING + ANSWERED TEST: backward-16
  window ~ v22's forward rate with a lag, NOT v24's whole-journey
  average; the tempering test (promise - what_real_max settling ~0.8 vs
  ~4.5) is ALREADY ANSWERED by v25@2M: gap ~14.9 >> 4.5 — THE TEMPERING
  WAS THE EPISODE-AVERAGING, not the efficiency framing. The user's
  whole-journey concept is NOT what W=16 implements; a longer window
  (sub-episode, e.g. 64-128) is the open lever if fantasy runs away.
  Directional subtlety recorded: backward-window aspiration = "seek
  states typically PRECEDED by high reward" (association shape). (g)
  Known hole: first eff_window states of each episode are unlabeled for
  what() — pure extrapolation exactly where episodes start; accepted
  (partial windows would reintroduce heteroscedasticity). (h) stale v24
  doc comments fixed in both files.
- FARM EXPLOIT (v26 review, CONFIRMED live): the projection target is
  only the FIRST-ORDER term of distance reduction. Exact identity:
  exact_reduction = projection - excess, excess ~ (perp displacement)^2
  / (2*||gap||) >= 0, and the excess does NOT telescope — a closed orbit
  near the goal collects ~||d||^2/(2*D*sqrt(dz)) per step FOREVER, vs an
  honest pursuit budget bounded by D/sqrt(dz) TOTAL (~1.4). In this
  fast-mixing latent (per-step ||d||^2 ~ 91 vs pair distance^2 ~ 2*dz =
  128, i.e. one step covers ~2/3 of a typical separation) the farm rate
  is ~0.5/step — worth ~45x genuine pursuit over an episode. Polarity is
  INVERTED vs the v16 treadmill: this exploit rewards NEAR goals + fast
  perpendicular motion, so LOW/flat goal distance is the warning sign.
  The floor caps the rate but cannot remove the exploit.
- v26 VERDICT: CANCELLED at ~4.5M (job 742) — FARMING CONFIRMED by the
  4-signal gate: proposer_out_std 0.002-0.013 (constant goal),
  rollout_goal_prog 0.62-0.79 SUSTAINED (impossible under a telescoping
  target: total claimed >> any honest budget), rollout_goal_dist flat
  ~2.0, returns -703 << the -149 reward-only backstop. Pure pursuit of a
  farmable target is worse than no goal channel at all: the policy
  perfected orbiting. The one-loss-per-module thesis is NOT refuted —
  the target was corrupted, so the arm never tested the thesis.
- v25 FINAL (job 741 SUCCEEDED): last-20 mean 682.6 +/- 211 = NEW FAMILY
  BEST (prev v22 +584); mid-run peaks 1099@4M, 1176@6M. ATTRIBUTION
  CAVEAT: v25's goal channel shows the SAME farm signature (prog ~0.78
  sustained, dist flat ~2.0), so its +682.6 was carried by rhat+qrate
  with the goal channel acting as a (possibly farming) directional bias
  — the causal claim "projection-goal channel helps" is NOT established
  by v25. Watch item on the blended line: promise-evidence gap ~14.9.
- `ppo_continuous_action_embopt_v27.py` — EXACT-REDUCTION TARGET (job
  743). v26 chassis unchanged (one loss per module); the pursuit target
  becomes the EXACT one-step distance reduction, tgt = (||g-z|| -
  ||g-z'||)/sqrt(dz). TELESCOPES (closed loops collect exactly zero —
  perpendicular-motion farm dead) and is bounded by ||d||/sqrt(dz) by the
  triangle inequality (v16 distance inflation dead) — both known
  exploits die by construction; no denominator, so no floor. New
  diagnostic prog_farm_rate = (old projection - exact) on the fresh
  slice = what the old target would have overpaid (expect ~0.5 early).
  Reading rules: sustained rollout_goal_prog now REQUIRES falling
  rollout_goal_dist (telescoping) — sustained prog with flat dist is
  itself an anomaly; inert-fixed-point check (alignment ratio -> 1 +
  goal_removal ~ 0 + flat returns) still applies. This is the CLEAN
  goal-only ablation arm: v22 both-channels +584, v22_vc0 reward-only
  -149, v27 goal-only with an uncorrupted target. Backstop: must clearly
  beat -149.
- READING-RULE CORRECTION (reviewer, supersedes v25 review rule (e)):
  NO value of the alignment ratio rollout_goal_prog/sqrt(wm_delta_var)
  is evidence of pursuit — under the PROJECTION target it is farmable at
  every level from ~0.37 to 1. Exact hold-distance geometry: a step of
  length ||d|| that keeps distance D from g has projection p =
  sqrt(D^2+||d||^2) - D; at family values (D=11.31, ||d||=9.54) that is
  p=3.49 -> target 0.44/step sustained at ratio 0.37 — healthy-looking
  pursuit with ZERO approach. Two null regimes: ratio->1 = carrot
  (needs z-dependent proposer, self-terminates by arrival); ratio ~0.37
  + constant goal = orbit/hold (stable — the regime v26 fell into,
  claimed prog 0.62-0.79 ~ the predicted 0.44 band). Rule going
  forward: the ONLY honest pursuit evidence is rollout_goal_dist
  actually falling / exact reduction positive. "ratio->1 = inert" stays
  as necessary-not-sufficient, demoted under the distance check. In
  v27 this is enforced at the source: rollout_goal_prog IS the exact
  reduction, so sustained positive prog REQUIRES falling dist by
  construction, and hold-distance orbits score exactly zero.
- v25 final telemetry (precision on the farm signature): dist 2.279
  (@2.6M) -> 1.597 (@8M), a TOTAL decline of ~0.68 latent units over
  ~5.4M steps, vs claimed prog 0.68-0.83 PER STEP throughout — claimed
  cumulative approach exceeds the honest exhaustible budget by ~5
  orders of magnitude. Farming attribution unchanged. Other finals:
  promise 11.2 vs delivery 0.89 (fantasy gap persisting), out_std 0.12
  (proposer near-constant), eff_var between/within 0.14/0.15 (healthy
  state-discriminative labels — the belief map itself is fine).
- v27 REVIEW (delta verified clean, job 743 unblocked). Key findings:
  (a) CONDITIONAL SAFETY: the exact target telescopes only for FIXED g.
  With per-step re-proposal the carrot regime (place g ahead of motion,
  collect, discard) is farmable at up to ||d||/sqrt(dz) ~ 1.19/step vs
  honest budget ~1.4 TOTAL. v27's guard is ARCHITECTURAL, not
  mathematical: the proposer's only loss is -what(g) — it is BLIND to
  the pursuit target — and it empirically collapses near-constant
  (out_std ~0.002-0.013), under which the sum telescopes exactly.
  CONSEQUENCES: (i) proposer_out_std is SAFETY-CRITICAL telemetry (the
  precondition for the conservation bound), not just a collapse watch.
  (ii) DESIGN LAW: reachability/pursuit-derived terms in the proposer
  score (v23/v25 qprog term) are KNOWN-DANGEROUS — they hand the
  proposer the incentive to build the carrot (~1.19/step vs 1.4 total).
  Future proposer z-dependence must come from something other than the
  pursuit head. (iii) g=z is strictly REPULSIVE under the exact target
  (gapn1=0 forces tgt = -||g-z'||/sqrt(dz) < 0) — the old "g~z
  repulsive" claim is finally true here (was false for projection).
  (b) SNR: no concern. |tgt| cap identical (1.19); action sensitivity
  d(tgt)/d(cos) = 0.91 at cos 0, 1.28 at cos 0.5 — sharpens where
  pursuit works. The offset moved: random motion scores ~ -0.44
  (absorbed into head bias; no effect on dq/da).
  (c) READING RULE (mandatory): prog_random_baseline =
  sqrt(rollout_goal_dist) - sqrt(rollout_goal_dist + wm_delta_var)
  ~ -0.435 at family values — derivable from two logged scalars.
  Three-level read of rollout_goal_prog: ~-0.44 = no pursuit (chance);
  ~0 = PARKED (distance held, closest approach reached); >0 sustained =
  genuine approach (REQUIRES falling dist by conservation). A raw -0.3
  at 2M is +0.14 above chance, not a dead channel.
  (d) GATE CORRECTIONS: alignment->1 inert gate is structurally
  FORBIDDEN under the exact target (conservation self-limits it) —
  v27's real inert/freeze signature is prog ~ 0 with dist flat at a low
  floor. action_grad_value is the budget-exhaustion gauge (finite
  credit per goal-epoch under constant goal; decay-to-zero precedes
  behavioral flattening — earliest freeze signal). SUCCESS signature:
  SAWTOOTH in rollout_goal_dist (approach -> exhaust -> what() retrains
  at newly-visited region -> argmax moves -> goal jumps -> dist steps
  up), prog positive in bursts. Sawtooth period = direct measurement of
  the belief-retraining loop throughput — the actual question v27 tests.
  (e) OPTIMISTIC CASE (why the 16.5-vs-1.6 fantasy gap may not matter):
  the policy consumes only the goal's DIRECTION; argmax of what() on
  the shell points along measured-efficiency ascent, so an overvalued
  distant goal is still far-field hill-climbing on the belief map.
  Risk is THROUGHPUT (loop stutters vs a direct reward gradient), not
  correctness. (f) BACKSTOP CALIBRATION: v26's -703 = orbiting was
  actively costly. A PARKED v27 should land ~-300..0 from control cost
  alone — a ~-250 reading is a freeze (informative), not a v26-class
  catastrophe; distinguish in the writeup.
- v26 POST-MORTEM CLOSED QUANTITATIVELY (reviewer's hold-identity check,
  run on v26's own logged scalars): prog_hold = sqrt(ldist+wm_delta_var)
  - sqrt(ldist). Over 2-4.5M: observed prog 0.744 at ldist 1.996 with
  wm_delta_var 2.127 (~1.5x the v16-era 1.42 — elevated latent step
  magnitude, as predicted for control-expensive orbiting) -> prog_hold
  = 0.618. The hold mechanism explains ~83% of the claimed progress;
  residual +0.127 (stable: +0.128 over 1-2M; required dvar for full
  closure 2.66 vs measured 2.13). NOTE: wm_delta_var is the raw second
  moment E[(z'-z)^2] per dim (comment says Var but code is uncentered),
  so coherent drift is already included — the residual is NOT hidden
  mean motion. Jensen (prog_hold concave in step size) biases the
  batch-mean BELOW the formula, so +0.13 is a lower bound on the true
  excess: a small secondary mechanism was active (candidates: goal
  motion beyond out_std ~0.002-0.013, or slight systematic alignment
  above pure hold). Verdict: orbit/hold farm confirmed as the DOMINANT
  mechanism by number, not pattern-match; secondary term minor.
- THE LINEAGE IDENTITY (cleanest statement of v16-v26): the projection's
  hold-distance farm rate and the exact target's random-motion baseline
  are the SAME quantity, opposite sign:
    projection at constant distance: +[sqrt(ldist+dvar) - sqrt(ldist)]
    exact under random motion:       -[sqrt(ldist+dvar) - sqrt(ldist)]
  The projection paid out, every step, precisely the amount by which
  random motion increases distance (it omitted second-order distance
  growth, so ANY motion collected that growth as "progress"). The exact
  target debits the same term correctly — hence its negative baseline
  and zero-scoring orbits. Every version v16-v26 measured an honest
  quantity; it just wasn't the quantity anyone thought it was.
- BASELINE IS A FORMULA, NOT A CONSTANT (reviewer amendment to the v27
  reading rule): prog_random_baseline must be recomputed at each read
  from the CONTEMPORANEOUS (rollout_goal_dist, wm_delta_var) pair. It
  scales with latent step size — v26's own inflation (dvar 1.42 ->
  ~2.1-2.8) moves it from -0.44 toward -0.78. Hazard pre-empted: a
  policy whose latent motion speeds up shows RAW prog falling while
  performance-vs-chance is flat/improving; against a stale -0.44 that
  misreads as pursuit decay (the lineage's signature error shape:
  honest quantity, wrong attribution). "-0.44" is an illustration at
  (dist 2.0, dvar 1.42), nothing more.
- CORRECTION (reviewer, supersedes the two entries above AND the hold
  math in the earlier reading-rule entry): the +0.13 residual was
  APPROXIMATION ERROR in the hold formula, not a v26 mechanism. Exact
  hold rate: ||g-z'|| = ||g-z|| = D forces (g-z).d = ||d||^2/2 with no
  expansion, so the projection collected under distance-hold is
    tgt_hold = v / (2*sqrt(l))   [v = wm_delta_var, l = rollout_goal_dist]
  Against v26's scalars: 2-4.5M -> 0.753 vs observed 0.744 (-1.2%);
  1-2M -> 0.744 vs 0.739 (-0.7%). CLOSURE ~99%, observed slightly BELOW
  hold (distance not perfectly rigid — the right side). The earlier
  sqrt(l+v)-sqrt(l) hold formula was the truncated expansion (valid only
  v << l; at v26's v/l ~ 1.07 it errs by ~20% = the whole "residual").
  RETIRED: both secondary-mechanism candidates (goal drift beyond
  out_std; systematic alignment above hold) — nothing to hunt in v28.
  The Jensen caveat dissolves (exact rate is LINEAR in v). "Required
  dvar 2.66" was a correct inversion of the wrong formula. The earlier
  illustration (p=3.49, 0.44/step, ratio 0.37) was likewise truncated;
  exact: p = ||d||^2/2D = 4.02, 0.50/step, ratio 0.42 — the qualitative
  rule (no ratio value is evidence of pursuit) unchanged.
  CONDITION LABELS on the sign identity, fixed: the identity pairs the
  projection's OVERPAYMENT (projection - exact) under RANDOM motion,
  +[sqrt(l+v)-sqrt(l)], with the exact target's random baseline,
  -[same] — exact, stands. (Precision note: the RAW projection under
  random motion averages ~0; it is the overpayment relative to the
  exact reduction that equals +0.44 — random motion loses ground at
  0.44/step and the projection scores it as neutral.) Under DISTANCE
  HOLD (what v26 actually did) the overpayment is v/(2*sqrt(l)) = the
  full projection, since exact = 0 there. The v27 random-motion
  baseline sqrt(l)-sqrt(l+v) was derived under the correct condition
  and STANDS unchanged in watcher/header. Verdict upgraded: v26 fully
  explained by distance-hold orbiting at its own step size, to ~1%,
  no secondary term. Lineage note (reviewer's own): same failure shape
  again — an honest, nearly-right quantity carrying a label that does
  not match what it measures.
- v27 FINAL (job 743 SUCCEEDED, 8M): last-20 -338.7 +/- 7.0. VERDICT:
  PARKED FREEZE, exactly the predicted band (~-300..0 control-cost
  floor; distinct from v26's -703 active orbiting). Full-run telemetry:
  (a) THE EXACT TARGET IS EXPLOIT-FREE, CONFIRMED: rollout_goal_prog
  0.0001-0.0006 the ENTIRE run while prog_farm_rate tracked the hold
  prediction v/(2*sqrt(l)) to ~4% in every 800k bin (e.g. 0.445 vs
  0.468 late) — the policy distance-held all run and was paid ~nothing
  for it, where the old projection would have paid ~0.45/step. The
  instrument and the fix both worked. (b) NO SAWTOOTH: dist pinned
  1.70-2.08, no goal jumps; proposer out_std 0.031 -> 0.0019 monotone;
  promise INFLATED 1.8 -> 6.0 while what_real_max stayed ~0.4 — the
  belief map at the never-visited goal region grows unchecked (no
  delivery correction), so the argmax never moves: RUNAWAY FANTASY
  ANCHORS THE GOAL PERMANENTLY. The belief-retraining loop v27 was
  testing never turned once in 8M steps. (c) goal conditioning stayed
  alive (goal_removal 0.13-0.47) and action_grad_value halved then
  flattened ~0.27 — the policy pursued; pursuit itself has no signal
  left at closest approach. prog ~ 0 vs contemporaneous baseline ~
  -0.41 = the policy holds distance (above chance), but a static shell
  goal is UNREACHABLE from the gait manifold — closest approach is a
  limit cycle at ldist ~ 2.0, and an HONEST target provides exactly
  zero gradient there. CAUSAL LADDER COMPLETE (HalfCheetah 8M, seed 1):
    reward-only (v22_vc0)        -149
    goal-only, honest (v27)      -339
    both, corrupted goal (v22)   +584
    blended, corrupted (v25)     +683
  The family thesis "the goal channel can carry everything" is REFUTED:
  goal-only is worse than reward-only. REINTERPRETATION (the big one):
  the goal channel's contribution in v22/v25 was almost certainly NOT
  aspiration/approach — their pursuit was farming (prog ~ hold rate,
  dist flat) — it was the projection's OVERPAYMENT ~ ||d||^2/(2D
  sqrt(dz)): a dense bonus proportional to SQUARED LATENT STEP SIZE,
  i.e. an accidental latent-velocity/kinetic-energy reward. Large
  per-step obs change in HalfCheetah correlates with moving fast, which
  correlates with the true reward. This explains the whole ladder at
  once: why the "goal" helped despite being constant (the bonus is
  goal-independent to first order), why v25 farmed AND set the family
  best (+683: kinetic bonus + reward heads), why v26 collapsed (-703:
  kinetic bonus ALONE selects fast perpendicular flailing with no
  reward direction), and why honest v27 froze (-339: remove the
  corruption and the channel has nothing to give at closest approach).
  The corruption was load-bearing. v28 should test this directly:
  replace the goal machinery with what it accidentally was — an honest,
  explicit dense motion/velocity term alongside measured reward heads —
  or find the single-objective formulation that captures both.
- CORRECTION TO THE REINTERPRETATION (self-caught, supersedes the
  kinetic claim as applied to v22): v22's goal target was the ONE-STEP
  DISTANCE-DELTA in D^2 units, ldist(z,g) - ldist(z',g) (v22:743), NOT
  the projection — the projection entered in v25. The distance-delta
  TELESCOPES for a fixed goal (sum of D^2_t - D^2_{t+1}), so orbiting
  paid ZERO under v22; its overpayment structure is dd = 2(gap.d)/dz -
  ||d||^2/dz, i.e. it contains a kinetic PENALTY, not a bonus. The
  kinetic-bonus story therefore CANNOT explain v22's +584. Scope of the
  kinetic hypothesis narrowed to the projection era: it explains v26's
  -703 (kinetic-only flailing) and possibly v25's goal channel, but
  v25's margin over v22 (+683 vs +584, CI ~211) is not significant, so
  the kinetic bonus has NO demonstrated positive contribution anywhere.
  What stands: goal-channel-as-adjuvant is causal (+584 vs -149) with a
  telescoping, orbit-proof target and near-zero realized pursuit — the
  channel's value is in its GRADIENT (directional action bias + kinetic
  penalty shaping), not in payments actually collected. v27's freeze
  then says: that gradient only matters when a reward gradient coexists.
- v28 PROBE PAIR (2x2 decomposition of the goal term on the v22
  chassis; existing rungs: dd +584, none -149):
  * embopt_v28: goal target dd -> EXACT reduction (||g-z||-||g-z'||)/
    sqrt(dz), value_coef 3.0 (scale-match: dd's action-gradient
    d(dd)/dcos = 2*sqrt(l*v) ~ 3.4 vs exact's ~0.9-1.3 — ratio ~2.6-3.7,
    midpoint 3.0). Tests: does the honest-in-D-units adjuvant retain
    the +584? If yes, v27 failed only for lacking reward heads and the
    family keeps the exact target. If no (-149-ish), the D^2 units /
    kinetic-penalty structure of dd was load-bearing.
  * embopt_v28_kin: qprog TARGET replaced by the goal-INDEPENDENT
    kinetic quantity ||z'-z||^2/dz (one-line delta; head may ignore g).
    Tests: is a pure latent-speed bonus an adjuvant at all? If yes
    (+584-ish), the goal directional content never mattered. If no,
    kinetic-bonus is dead as a mechanism and direction is what matters.
- v27 POST-MORTEM REVIEW (crossed with the dd self-correction; the
  reviewer's endorsement of the kinetic story predates it and inherits
  the v22-attribution error — but four elements stand on their own):
  (a) SHARPENED OVERPAYMENT FORM: overpay = ||d_perp||^2/(2D sqrt(dz))
  exactly (squared PERPENDICULAR step; at v25's cos ~0.54 that is ~70%
  of ||d||^2). Mildly anisotropic: rewards motion ACROSS the goal
  direction, debits the along-goal component. projection = honest
  telescoping term (bounded, ~nothing over a long run) + this bonus =
  the bonus was the projection channel's ENTIRE sustained payment.
  (b) SYNTHESIS (needs re-exam post-dd-correction for v22, holds for
  v25/v26): magnitude pressure alone flails (-703); reward direction
  alone plateaus (-149); the combination runs. NOTE v22's dd expands to
  directional-linear MINUS kinetic — v22's +584 is evidence FOR the
  perturbation/direction reading, AGAINST pure-kinetic.
  (c) GENERALITY EXPOSURE (biggest open risk): every family result is
  HalfCheetah-only. HalfCheetah cannot terminate, so state-change
  pressure is free and correlates with reward. In Hopper/Walker2d large
  state change correlates with FALLING and ends episodes — magnitude
  pressure is selected AGAINST, predicting failure for the combined
  form too. TEST RUNNING: v25 on Hopper (job 747; mechanism-specific
  prediction: v25 well below PPO-baseline Hopper ~2000+ confirms the
  magnitude-pressure reading AND caps the family's generality).
  (d) OVER-CLAIM SCOPED: "goal-only worse than reward-only" holds for
  the aspiration ARCHITECTURE AS BUILT (frozen proposer: promise 1.8->
  6.0 unchecked, loop turned 0 times). An honest target under a
  proposer that actually moved is UNTESTED. Thesis logged as refuted
  for-the-architecture, honest-target question kept open.
  PROBE SET COMPLETED (reviewer C2 added): v28 exact-adjuvant (744),
  v28_kin C1 kinetic (745), v28_dir C2 directional-linear u.(z'-z)
  fixed random u (746) — C1 >> C2 confirms kinetic; C1 ~ C2 says any
  large structured gradient perturbation escapes the plateau. 2x2+C2
  rungs: dd +584 | none -149 | exact ? | kinetic ? | direction ?.
- FOR THE USER (one-loss law, surfacing not designing-around): the law
  as stated is about multiple losses on one GRAPH; one-loss-PER-MODULE
  was our interpretation. Ledger: interpretation produced -703 (v26)
  and -339 (v27); blended policy objectives produced +584 (v22) and
  +683 (v25). A single scalar -(rhat + qrate + lambda*adjuvant) is
  arguably ONE loss on one graph, with cross-module routing already
  handled by detach + backward(inputs=). Decision belongs to the user;
  the reviewer-recommended v28+ destination satisfies the law under ANY
  reading: policy loss = one term, exploration pressure moved into the
  ROLLOUT as state-correlated/latent-directed noise (shaped by whatever
  the probes confirm) — a genuinely novel mechanism vs an accident
  rediscovered.
- REVIEWER VERIFICATION + THE LOAD-BEARING NUMBERS FOR THE ADJUVANT
  READING (log permanently): v22 FINAL telemetry — proposer_out_std
  0.002456 (goal effectively constant -> telescoping applied),
  rollout_goal_prog -0.002034 (realized pursuit zero, slightly
  negative). The goal channel COLLECTED NOTHING over 8M and still moved
  -149 -> +584: near-proof the value is in the GRADIENT, not payments.
  The dd gradient, exactly: d(dd)/d(z'-z) = 2(g-z')/dz — a persistent
  latent-direction pull of magnitude ~D plus velocity damping -2d/dz.
  A regularizer/symmetry-breaker, not a goal.
- value_coef 3.0 caveats for reading v28 (reviewer): the dd/exact
  scale match is ALIGNMENT-DEPENDENT (ratio 3.7 at chance, 2.6 at cos
  0.5) — a null could be a scale artifact at either end; and 3.0
  multiplies the head's approximation error too (3x gradient NOISE) —
  if v28 underperforms, "noise amplification" is a live alternative to
  "dd structure load-bearing"; prog_calib_abs separates them.
- v28_kin (745) CAVEAT: proposer left live — g=propose(z) is a second
  encoding of z the head can exploit, and nothing constrains the
  policy's use of the drifting g input. Read 745 as "kinetic + live
  drift", not clean kinetic. Clean arms added on the frozen-goal
  chassis (g = one random shell point at init, proposer_updates=0,
  [z,g,g-z] then reparameterizes z — no architecture change):
  * embopt_v28_fixg (748): dd target + value_coef UNCHANGED, only the
    apparatus removed. THE decisive arm: ~+584 => proposer/belief/
    efficiency machinery from v20 forward was decorative, v29 = reward
    heads + constant latent-direction bias; ~-149 => goal content
    matters, aspiration line alive.
  * embopt_v28_kinf (749): fixg chassis + goal-independent kinetic
    target — true "kinetic bonus, no goal channel".
  Probe fleet: 744 exact | 745 kin-live | 746 dir-live | 747 v25-Hopper
  | 748 fixg | 749 kinf; rungs dd +584, none -149.
- SCALE CONFOUND CAUGHT PRE-DATA (reviewer, time-critical): C1 and C2
  were matched on target SPREAD (1.4 vs 1.19) but differ 3.4x in
  d(target)/d(motion) — the only quantity the policy consumes (C1
  quadratic: 2||d||/dz ~ 0.30; C2 linear: |u| = 1). The 745-vs-746
  (live) comparison is DEPRECATED for mutual reading (scale-confounded
  AND drift-leaky); each still reads individually against the rungs.
  Clean discrimination pair rebuilt on the frozen-goal chassis with
  matched gradient norms: embopt_v28_kinf relaunched as job 750 at
  --value-coef 3.4 (749 cancelled pre-start), embopt_v28_dirf NEW (job
  751, vc 1.0): u.(z'-z), u one fixed random unit direction.
- C2 READING RULE (reviewer): u.d TELESCOPES (closed loop = u.(z_end -
  z_start) = 0; steady gaits are periodic in egocentric obs, hence in
  latent). Read dirf in the gradient frame; realized target mean should
  sit ~0. SUSTAINED positive mean = net latent drift along u = the
  policy stopped cycling: an ANOMALY signal, not success.
- PRE-REGISTERED BRANCH (kinetic-penalty falsifier): v22 carried a
  kinetic PENALTY (+584); v26 was bonus-dominated (-703); reviewer
  predicts kinf@3.4 lands BELOW -149 (bonus is anti-useful with reward
  heads present). IF kinf < -149 (or kinf ~ dirf ambiguous), RUN
  v28_penf = kinf file at --value-coef -1.0 — exactly v22's implicit
  penalty scale (dd's -||d||^2/dz term at vc 1.0) with zero directional
  content. Tests "the penalty was the active ingredient and the
  direction was inert" — which v22's rollout_goal_prog -0.002 leaves
  fully open. Decision rule logged BEFORE data so the branch is not
  chosen after seeing the number.
- Fleet after corrections: 744 exact-adjuvant | 745 kin-live (indiv
  only) | 746 dir-live (indiv only) | 747 v25-Hopper | 748 fixg
  (decisive, one-variable) | 750 kinf@3.4 | 751 dirf@1.0; rungs dd
  +584, none -149. First-checkpoint duty: verify action_grad_value
  parity between 750 and 751 EARLY (relaunch is cheaper than
  interpreting a confounded 8M pair).
- SCALE ALARM PARTIALLY RETRACTED + PAIRING STRUCTURE SETTLED (reviewer
  wrote pre-750/751; reconciled here): at vc 1.0 fixg's dd gradient
  (2||g-z'||/dz ~ 0.353) and kinf's kinetic gradient (2||d||/dz ~
  0.298) are naturally within 18% — the 3.4x confound was confined to
  the quadratic-vs-LINEAR pair. Final structure: {748 fixg vc1.0 <->
  v22 rung} one-variable apparatus test, naturally scaled; {750
  kinf@3.4 <-> 751 dirf@1.0} direction-vs-magnitude at matched ~1.0
  gradient norm; 745/746 exploratory only (746 formally ORPHANED — no
  clean twin; do not use for direction-vs-magnitude). Asymmetry noted:
  bonus arm tested at 3.4 (matched to dirf), penalty branch
  pre-registered at -1.0 (matched to v22's implicit penalty scale) —
  different questions, different scale anchors. kinf@3.4 also carries
  the 3.4x noise-amplification caveat (same shape as v28's 3.0).
- PRE-REGISTERED CONFOUND, fixg NEGATIVE BRANCH (log BEFORE the number
  lands): g_fix is frozen in LATENT coordinates while the encoder
  trains 8M steps — constant vector, continuously drifting semantic
  referent (the v15/v21 staleness, maximal case). POSITIVE branch
  (~+584) is STRENGTHENED by this (arbitrary vector with wandering
  meaning does the whole job => apparatus decorative). NEGATIVE branch
  (~-149) is AMBIGUOUS: "aspiration content mattered" vs "latent-frozen
  goal went stale; a live proposer at least tracks the moving frame" —
  opposite v29 implications. Disambiguator pre-committed: v21-style
  persistent referent — anchor the goal to one fixed OBSERVATION drawn
  at init, re-encode it every iteration (referent fixed, representation
  tracks the encoder). Run ONLY if fixg lands negative.
- TABLE-READING RULE for the neutralized arms (reviewer, logged before
  the finals table exists): 748/750/751 are NOT a comparable triple —
  fixg runs at gradient norm 0.353 (correctly matched to its v22 rung
  at vc 1.0) while the kinf/dirf pair runs at ~1.0 (matched to each
  other, verified 1.013 vs 1.000). The ONLY valid comparisons are
  fixg-vs-v22 (apparatus) and kinf-vs-dirf (magnitude vs direction);
  nothing legitimate crosses between the two groups. Do not build the
  side-by-side triple. penf spec re-verified by reviewer: vc -1.0 on
  the kinf file = gradient norm 0.298 = v22's implicit penalty
  magnitude exactly, zero directional content.
- PRE-REGISTERED SIGN CONTROL (reviewer; closes the last pre-data gap):
  if kinf@3.4 < -149 AND penf@-1.0 reproduces ~the v22 rung, the
  inviting reading "the SIGN of the kinetic term is the active
  variable" is MAGNITUDE-CONFOUNDED (3.4x apart: 1.013 vs 0.298);
  "small helps, large hurts" explains the same pair. Rule: that sign
  claim REQUIRES kinf@+1.0 (gradient 0.298, exactly matching
  penf@-1.0) before it may be logged — run it only on that branch, not
  speculatively. Partial mitigation: v26 (-703) is bonus-at-large-
  scale corroboration, but differs too widely to serve as control.
- OPERATING-POINT RISK + SECOND DISCRIMINATION PAIR (reviewer; acted
  pre-data). Common axis = goal-channel gradient norm at the action:
  v22 dd 0.353 -> +584 | 744 exact@3.0 0.375 -> +434@5M climbing |
  745 kin-live 0.298 -> -185 | 746 dir-live 1.000 -> -387. The two
  WORKING arms sit at ~0.35-0.38 (independent confirmation the vc 3.0
  match landed); the high discrimination pair (750/751) sits at ~1.0 =
  dir-live's operating point, the worst arm so far. PRE-REGISTERED
  NULL: if kinf@3.4 AND dirf@1.0 both land in the -150..-400 band, the
  pair is UNINFORMATIVE about direction-vs-magnitude (scale artifact,
  not science) and no conclusion may be drawn from it. Insurance
  spent (1 job each): job 752 embopt_v28_kinf1 (kinf @ vc 1.0, norm
  0.298 — TRIPLE duty: sign control for the penalty branch, low-point
  kinetic arm, and the null-branch re-run) and job 753 embopt_v28_dirf03
  (dirf @ vc 0.3, norm 0.300 — matched <1%): the discrimination now
  exists at BOTH operating points regardless of how the high pair
  lands. INTERIM FINDING (more than expected): kin-live 0.298 vs v22
  dd 0.353 are accidentally near-scale-controlled and differ by ~769
  return — at matched scale, direction-plus-penalty beats kinetic
  bonus decisively; CONTENT is implicated over scale, and the
  kinetic-bonus hypothesis is in serious trouble before the
  neutralized wave reports.
- PRE-REGISTERED COMPARISON STATISTIC (reviewer; logged before fleet
  finals). Evidence: v22's own trailing-20 swung >800 points within one
  run (-50.6@3M, +440.6@4M, +396.2@5M, -252.4@6M, -103.0@7M,
  +583.7@8M) — "+584" is an oscillation phase, and last-20@8M across 9
  single-seed arms would manufacture orderings from phase. RULES:
  (1) PRIMARY statistic = mean episodic return over the FINAL 1M steps
  (~1000 episodes). (2) SECONDARY = trailing-20 trajectory at
  5/6/7/8M; a claimed difference must be visible in >=3 of 4
  checkpoints, not the endpoint alone. (3) THRESHOLD: no
  direction-vs-magnitude or apparatus conclusion from a final-1M gap
  under ~250 points. RUNGS RECOMPUTED under the primary statistic:
    v22      final-1M = +423.0   (traj +405 / -254 / -123 / +596)
    v22_vc0  final-1M = -157.5   FLAT, window 4.5-5.5M (run ended
             5.5M; any @6/7/8M "-151.3" is carry-forward, NOT data)
    v25      final-1M = +752.7   (traj +708 / +860 / +678 / +492)
             — still family best; MORE stable than its last-20
    v26      final-1M = -589.5   (cancelled ~5M)
    v27      final-1M = -332.9   (traj -343/-338/-328/-336; freeze =
             near-zero variance, statistic barely matters)
  Note for fixg-vs-v22: the apparatus rung is +423 +/- an ~850-point
  internal swing — the trajectory requirement (3 of 4) does the real
  work there; a fixg final-1M within ~250 of +423 with a similar
  oscillation shape reads as reproduction.
- DIAGNOSIS CORRECTED: "PARKED" -> "FUTILE PUSH" (reviewer arithmetic;
  supersedes the passive-park language in the v27 verdict). HalfCheetah
  reward = fwd_velocity - 0.1*sum(a^2). A truly idle policy earns ~0,
  NOT -333. v27's -332.9/1000-step episode = 0.333/step of control
  cost at ~zero velocity => sum(a^2) ~ 3.33 => RMS |a| ~ 0.75/actuator:
  the policy pushed at 3/4 actuation into an unreachable goal for 8M
  steps. The freeze is a freeze of LEARNING (dq/da ~ 0 at closest
  approach, CI collapses), NOT of motion — the converged action is a
  maximal futile push. This explains what passive-park could not:
  v28's late -296 << vc0's -157.5 — a dead channel cannot make things
  WORSE than no channel; a channel commanding 0.75-RMS actuation into
  a wall can, and the deficit size matches. CONFIRMATION INSTRUMENT:
  diagnostics/ctrl_cost = (0.1*sum a^2).mean() added to the four
  pre-launch files (v28_fixg/_kinf/_dirf + v25->Hopper); ~0 = idle,
  ~0.33 = futile push; decomposes every failure into "didn't move" vs
  "moved expensively". (Running jobs 744/745/746 predate the line.)
- v29 DESIGN CONSTRAINT (from the above): FARM-PROOF AND COST-FREE ARE
  DIFFERENT PROPERTIES. The exact target's VALUE vanishes at closest
  approach (farm-proof) but its ARGMAX ACTION does not — and the action
  is what gets executed. An unreachable goal is a permanent control-
  cost tax proportional to push effort. Any v29 keeping a goal needs
  (a) reachable/on-manifold goals, or (b) a pursuit term whose ACTION
  decays at closest approach, not just its payment. Nothing in the
  current fleet satisfies either.
- OPEN TENSION (logged, not smoothed): under the pre-registered
  final-1M statistic, v25 (+752.7) vs v22 (+423.0) = +330, CLEARING
  the 250 threshold — the statistic change converts "indistinguishable"
  (last-20: +99 gap) into "v25 significantly family best", and v25 is
  the arm whose goal channel farms. Not necessarily the farming story
  rescued: v25 also differs in belief window (backward-16 vs forward
  rate). Discriminator = 744's final-1M: ~+423 => target form
  irrelevant, backward-window belief becomes the candidate for v25's
  edge; ~-296 => exact target actively harmful at 8M and v25's edge
  needs another account. Flagged: this tension was CREATED by a
  methodology change the reviewer recommended — kept visible.
- FIRST-WAVE FINALS (pre-registered statistic; jobs 744/745 done):
  * 744 exact@3.0: final-1M = -155.3, traj +205/-302/+38/-244. Reads:
    (i) does NOT retain the dd adjuvant (-578 vs the +423 rung, clears
    threshold decisively); (ii) the NOISE-AMPLIFICATION branch is
    REFUTED: prog_calib_abs 0.029-0.044 vs v22's 0.073-0.108 —
    relative-to-range calibration comparable (0.029 vs 0.026); the
    head fit the exact target WELL and vc 3.0 amplified a clean
    gradient; (iii) the pre-registered instability branch is the
    descriptive fit: mean pinned at the no-channel floor with +-500
    swings — "oscillates around floor", neither stable adjuvant nor
    stable harm; (iv) same zero-collection regime as v22 (prog 0.0008,
    dist ~2.0, out_std ~0.005): only the gradient ever acted.
  * 745 kin-live@0.298: final-1M = -161.2, traj -191/-194/-171/-172 —
    FLAT at the vc0 floor (-157.5). Kinetic bonus at low scale is
    INERT: no help, no harm. (High-scale harm prediction remains
    kinf@3.4's question.)
  * v25-tension discriminator landed BETWEEN its branches: exact
    contributes net ~zero at 8M (not +423, not -296) — v25's +330 edge
    over v22 still unexplained; belief-window remains a candidate.
- NEW LEADING CANDIDATE — THE DAMPING TERM (logged with readings
  BEFORE its test runs): dd's action-gradient = 2(g-z')/dz = DIRECTION
  (~D-weighted) + DAMPING (-2d/dz, the latent velocity penalty);
  exact's = direction only, normalized, NO damping. 744 shows
  direction-only at matched scale and clean calibration = floor.
  Ergo the surviving structural difference carrying v22's +423 is the
  DAMPING term (alone or in synergy). Direct test = v28_penf (kinf
  file @ vc -1.0, norm 0.298 = v22's implicit penalty magnitude,
  zero direction): submitted NOW on the strength of two independent
  routes (744's result + the original penalty branch), readings
  pre-registered: fixg ~+423 AND penf ~positive => damping is the
  active ingredient (direction decorative); fixg ~+423 but penf ~
  floor => direction+damping synergy required; fixg ~floor => dd
  needed the live apparatus after all and the damping story dies with
  it. penf name: embopt_v28_penf.
- v28 HINGE READ (reviewer's sawtooth discriminator, run at 250k grain
  over 4M-7.75M): the STRICT discriminator fires NEGATIVE — dist flat
  1.9-2.13 through both hinges, no jump. But dist is structurally
  UNINFORMATIVE here: with zero collection (prog 0.0008) and both z,g
  near shells, dist sits ~2.0 regardless of which goal is commanded —
  the dist-jump signature presupposed approach dynamics that never
  existed. The channels that CAN see goal identity tell a different
  story: crash hinge (5.0->5.5M) promise 5.01->2.47 and what_real_max
  1.12->0.32 as returns crash; trough (5.5-6.5M) promise deflates to
  1.85, out_std pinned 0.0025; RECOVERY hinge (6.75-7.0M) out_std
  SPIKES 4-5x (0.0025->0.0112->0.0137 — the argmax landscape
  reorganizing), promise re-inflates 1.85->5.80, wmax 0.28->1.49,
  returns -278->+319. VERDICT: not the approach-budget sawtooth, and
  not bare instability either — a BELIEF-DEFLATION CYCLE: crash ->
  realized efficiency falls -> belief map deflates -> proposer argmax
  reorganizes (goal moves in DIRECTION, invisible to dist) -> returns
  recover -> belief re-inflates -> new anchor -> next crash. Period
  ~1.5M (provisional, ~2 cycles). FIRST EVIDENCE IN THE FAMILY that
  the aspiration loop can turn at all (v27: zero turns, promise
  monotone up, out_std monotone down — the contrast is exact).
  Consequence kept from the reviewer's framing: the v29 question for
  this line is loop THROUGHPUT and stability, not coefficient tuning.
- PERMITTED-READINGS ADDITION (reviewer): v28-vs-fixg QUALITATIVE
  comparison is sanctioned despite the scale/target mismatch (which
  still blocks numeric comparison): live proposer can move the goal,
  frozen goal structurally cannot. If v28 recovers from futile-push
  troughs repeatedly while fixg crashes once and plateaus forever,
  the proposer earns its keep independently of scale. fixg's version
  of the signal: return recoveries + what_real_max cycling (its
  out_std/promise are meaningless — proposer untrained).
- DAMPING HYPOTHESIS CHALLENGED (reviewer; logged before penf/fixg
  report). (a) SIGN-SYMMETRY PROBLEM: 745 says +||d||^2/dz @0.298 is
  inert; penf tests the SAME channel at -0.298. Damping-active requires
  sign asymmetry with no proposed mechanism. penf branches
  pre-registered BOTH ways: floor => symmetry confirmed, damping dead;
  positive => asymmetry established, mechanism then owed. (b) NOTE:
  damping and farm-proofing are the SAME term (-||d||^2/dz is exactly
  what makes dd telescope) — penf isolates an anti-farm correction
  stripped of what it corrected; may not be a meaningful object alone
  (second route to expecting floor). (c) COMPETITOR — DISTANCE-DECAY:
  dd's gradient magnitude 2||g-z'||/dz SHRINKS on approach (satisfies
  the v29 cost-free constraint); exact's 1/sqrt(dz) NEVER does —
  predicting exactly the measured futile-push tax difference. The
  fleet cannot separate decay from damping (dd carries both).
  (d) SEPARATING ARM, PRE-SPECIFIED AND HELD until fixg + kinf@3.4
  land: dd-minus-kinetic, target = 2(g-z).d/dz on the v22 file
  (D-weighted decaying direction, NO damping; deliberately
  reintroduces farming). ~+423 => distance-decay active, damping
  decorative; floor => damping load-bearing, penf becomes the
  interesting arm.
- FARM/TELESCOPE AXIS — THE UNCOMFORTABLE ORDERING (reviewer): among
  reward-head arms: v25 farmable/never-exhausts +752.7 | v22
  telescoping+decay +423.0 | 744 telescoping+constant-pull -155.3.
  One axis, ~900 points, orders all three: telescoping credit EXHAUSTS
  (bounded by initial distance) and ends in futile push; a farmable
  target supplies PERPETUAL gradient. Implication to take seriously:
  the farm payment may be the FEATURE — the family's only perpetual
  goal-gradient mechanism (v26's -589 only shows perpetual gradient
  WITHOUT direction flails, not that perpetual is bad). CONFOUND
  (keeps this a candidate, not a conclusion): v25's goal gradient norm
  is 0.125 — smallest in the fleet (vs 0.353/0.375) — so "small
  perpetual nudge beats large exhausting pull" fits the same numbers.
  Both readings agree on v29 direction: sustained (non-exhausting),
  modest-magnitude, direction-carrying goal gradient with cost decay
  near anchor.
- 746 dir-live FINAL: final-1M = +328.2 (n=1008); traj@5/6/7/8M =
  -425.6 / -340.8 / +346.8 / +287.6. LATE FLIP ~6.5M after 5.7M of
  floor — my "worst arm" call at 5.7M was wrong under the primary
  statistic. Reads (single seed, exploratory arm, proposer-live leak
  caveats apply): (a) NULL-RULE PREMISE DEAD: the 750/751 null rule
  assumed operating point ~1.0 = dir-live's floor regime; dir-live is
  not floor. Null rule RETIRED — kinf@3.4 and dirf@1.0 are read on
  their own values against ledger rungs, no auto-uninformative branch.
  (b) POINT vs DIRECTION refinement of futile push: 744 constant-
  magnitude pull toward an unreachable POINT = -155.3 (perpetual
  conflict at closest approach); 746 constant-magnitude pull along a
  fixed random DIRECTION = +328.2. A direction never "arrives" — no
  closest-approach conflict exists; within an episode the credit
  telescopes to (z_T - z_0).u_dir (oscillation-proof) yet across
  episodes it never exhausts. Supports the perpetual-gradient reading
  on the farm/telescope axis. (c) SCALE-ALONE ORDERING NOW UNTENABLE:
  norm 0.125 (+752.7), 1.0 (+328.2), 0.375 (-155.3), 0.353 (+423.0) —
  no monotone scale story; target FORM (exhausting-point vs
  perpetual-direction/farm) carries the ordering. (d) v29 spec gains a
  concrete candidate: fixed or slowly-varying latent DIRECTION targets
  are farm-proof-per-episode, perpetual-across-episodes, and dodge the
  reachability constraint entirely (no point to reach).
- 747 v25-Hopper mid-read @3M: trailing-20 = 66.1 (Hopper PPO ref
  ~2000+). Tracking the pre-registered mechanism prediction (v25's
  farm does not generalize off-HalfCheetah). Final read at 8M stands.
- 744 FLOOR CLAIM AMENDED (reviewer challenge, verified): trajectory
  +205.3 / -302.1 / +38.2 / -244.3 at 5/6/7/8M; trajectory mean -75.7;
  final-2M -98.6. The final-1M window (-155.3) lands on a DOWNSWING —
  worst-case sampling for an oscillating arm. 744 is hereafter
  reported as "final-1M -155.3, trajectory mean -75.7, UNSETTLED", not
  as a settled floor. ~80 points above vc0's -157.5 on trajectory
  mean, with ~500-point swings. The pre-registered statistic is kept
  (no retroactive changes) but is acknowledged as designed for settled
  runs; unsettled runs get the three-number report + the flag.
- 744 HINGE CHECK (250k bins, 4M-8M — closes reviewer item 7856ba4e):
  rollout_goal_dist FLAT 1.90-2.13 through the crash (+491 -> -300 @
  5.0-5.25M), the trough, and the recovery (-278 -> +319 @ 6.5-7M);
  rollout_goal_prog 0.0007-0.0010 throughout (zero collection).
  NOT A SAWTOOTH — no goal-loop turnover; policy/value-loop
  instability with the goal channel along for the ride (reviewer's
  branch 2). Recovery coincides exactly with proposer_out_std spike
  0.0025 -> 0.0137 (belief-deflation cycle signature, matching the
  earlier v28 hinge read). CONSEQUENCE: the damping/decay inference's
  premise survives the sawtooth attack, but its statement softens to:
  constant-direction pull at matched scale = unsettled oscillation in
  the -76..-155 band — still 500-600 points below dd's +423, so the
  residual attributed to damping-or-decay stands in magnitude; the
  "at the floor" phrasing does not.
- DIST BLIND SPOT QUANTIFIED (reviewer's concession, recorded at
  their request as their error — third instance of the family's
  label-vs-measurement error shape): with shell normalization
  ||g|| = ||z|| = sqrt(dz), rollout_goal_dist = 2 - 2*cos(theta) —
  a pure ANGLE, nothing else. The observed 1.9-2.13 band is
  cos(theta) in [-0.065, +0.05], entirely inside the 1/sqrt(64) =
  0.125 spread of two RANDOM directions in 64-d. dist cannot
  distinguish "anchor relocated to a fresh direction" from "anchor
  never moved." Any future goal-motion claim must use a quantity not
  pinned by dimensionality (see instrumentation entry below).
- 744 LEAD-LAG @50k GRAIN (reviewer's causal discriminator,
  pre-registered readings): returns LEAD. Return recovery begins
  ~6.25M (-301 -> -277 over 6.25-6.45M) while proposer_out_std is
  flat at 0.00244; out_std's first movement is 6.45-6.50M and its
  peak (0.0323 @6.70-6.75M) coincides with the return PEAK (+621),
  decaying with it. Per the pre-registration: READOUT story wins —
  the belief map passively fits improved delivery; the proposer
  reorganization FOLLOWS recovery, it does not drive it.
  CONSEQUENCE: belief-loop interventions (delivery-corrected
  promise) fix a thermometer; demoted for v29. The crash originates
  in the reward/policy loop. "Belief-deflation cycle" is renamed:
  the cycle is real, the belief channels are its READOUT.
- CYCLE = BUDGET EXHAUSTION (reviewer's free test, matched on reward
  heads, pre-registered): v22 telescoping-dd CYCLES — crash onsets
  ~3.0M (ret 604 -> -39) and ~5.25M (ret 359 -> -215; promise 4.51
  -> 2.28 trough @6.5M; wmax 0.88 -> 0.26) with full recovery
  7.0-7.5M (ret +511; promise 5.77; wmax 1.19). v25 farmable
  projection DOES NOT — promise 16.3 -> 18.4 monotone-to-plateau,
  wmax up to ~2.1 and holds, returns plateau 730-820, and
  proposer_out_std PINNED 0.081-0.086 all run (never collapses; v22
  collapses to 0.002). Pre-registered branch 1 confirmed: the cycle
  IS telescoping-budget exhaustion (map promises, policy chases,
  bounded budget runs out, delivery collapses, re-anchor buys a
  fresh budget). The farm/telescope axis now has a MECHANISM, and it
  agrees with the lead-lag verdict independently: fix the TARGET
  (non-exhausting), not the belief. v27 is disposed as a
  counterexample (telescoping but no reward heads — nothing to
  exhaust FROM). v29 spec: non-exhausting target is now doubly
  motivated (direction-form 746 result + exhaustion mechanism).
- GOAL-DRIFT INSTRUMENTATION (pre-registered, CANNOT attach to
  750-754): reviewer proposed logging cos(g_t, g_{t-Delta}) / drift
  of the goal mean — the quantity dist integrates away. All queued
  arms (751-754) are fixg copies with FIXED g (proposer_updates=0),
  so there is no goal motion to measure; 750 already running. The
  instrumentation is pre-committed for ANY future live-proposer run
  (v29 or v22-family rerun): log cosine between consecutive
  iterations' proposed-goal means. out_std alone is a SPREAD, not a
  location — a 4-5x move on a collapsed spread is compatible with
  the argmax barely moving (reviewer caveat, adopted).
- fixg READING SOFTENED (reviewer): fixg cannot reorganize by
  construction, so a one-crash-plateau outcome is consistent with
  "reorganization requires a live proposer" AND with "a fixed anchor
  is a different task with a different ceiling." Positive result is
  phrased "consistent with the proposer earning its keep," not
  established. ~1.5M period stays provisional (fitted to two
  oscillations; amplitude same order as 744's swing throughout).
- 746 ALIGNMENT-LOTTERY HYPOTHESIS (reviewer; competitor to the
  direction-form reading, SUPERSEDES the celebratory parts of the 746
  final entry): u_dir is drawn ONCE from a Gaussian; whether paying
  the policy to move the latent along it helps depends entirely on
  how that arbitrary latent direction maps onto physical forward
  progress — a coin flip, run once. The trajectory is the SIGNATURE
  OF A SIGN FLIP, not a mechanism engaging: first 5.7M at ~-400 is
  270 BELOW the no-channel floor (-157.5) — a dead channel gives the
  floor (745 did, -161.2); 270 below floor means the channel actively
  drove the policy somewhere bad (natural candidate: paid to run
  BACKWARD). Then +480 above floor. Mechanism: u_dir is fixed in
  LATENT coordinates but the encoder keeps training — the physical
  meaning of the fixed direction rotates, and at ~6.5M it rotated
  past orthogonal into positive alignment with forward progress.
- ANTIPODE CONTROL SUBMITTED — job 755 embopt_v28_dirfneg, dirf @
  value_coef=-1.0. The target is LINEAR in u_dir and u_dir is drawn
  before value_coef is consulted, so -1.0 is the SAME draw mirrored
  (perfect same-seed antipode, not a new lottery ticket).
  PRE-REGISTERED: both 751 (+1.0) and 755 (-1.0) well above floor =>
  direction targets structurally good, sign-independent, v29
  direction line vindicated. They MIRROR (one high, one ~-400) =>
  alignment lottery confirmed; 746's +328.2 was a won coin flip and
  says nothing about direction targets as a class. NOTE 751 is only
  a partial replicate (different RNG stream position => independent
  direction; two favorable draws = 25% under coin flip —
  informative, not decisive; the antipode is decisive).
- dir_reward_corr INSTRUMENTATION added to v28_dirf.py (pre-start
  for 751/753/755; no RNG consumed, u_dir draw unchanged): masked
  Pearson corr(u_dir displacement, env reward) on the fresh segment
  per iteration. Converts the alignment argument from inference to
  measurement: a zero-crossing near a return flip on any direction
  arm closes the case.
- POINT-vs-DIRECTION MECHANISM CORRECTED (reviewer, from our own
  dist result — supersedes the "closest-approach conflict" clause of
  the 746 entry): 744's dist was PINNED at the random-pair baseline
  ~2.0 all run — z never approached g, there was no closest-approach
  event, so arrival dynamics CANNOT separate 744 from 746. Surviving
  candidate: STATIONARITY — 744's proposer-driven goal moves (policy
  chases a non-stationary objective, never consolidates a gait);
  746's u_dir is constant 8M steps. 748 fixg is exactly the
  stationary-POINT arm that separates stationarity from
  directionality: fixg well above floor => operative variable is
  stationarity and "point" was never the problem. Point/direction
  conclusion ON HOLD until fixg lands. v29 TENSION: if stationarity
  carried 746, a proposer that proposes DIRECTIONS reintroduces the
  moving-objective problem under a new name — the v29 direction line
  must keep the direction fixed or slowly-varying.
- CROWN CAVEAT: +752.7 is currently a HALFCHEETAH-LOCAL result.
  v25-Hopper @3M trailing-20 = 66 (vs PPO ref ~2000+) is failing to
  stand up, tracking the farm-doesn't-generalize prediction. If the
  Hopper final stays near this level, v25's status changes from
  "king" to "best on the tuning environment," and v29 should not be
  specified to reproduce a possibly environment-local artifact.
- NULL-RULE RETIREMENT PRECEDENT (for the changelog): the 750/751
  null rule was retired because its stated PREMISE (dir-live sits at
  floor) was falsified as a matter of fact — NOT because the reading
  was unfavorable. Retiring a rule with a false premise is
  legitimate; retiring one whose reading is disliked is what the
  discipline prevents. These look similar in a changelog and are
  opposite in kind.
- SCALE CAVEAT NARROWED (reviewer concession): scale is dead as a
  PRIMARY explanation of the arm ordering; survives only as "scale
  may matter within a fixed target form," which no arm currently
  tests.
- 748 fixg FINAL — SETTLED, BELOW FLOOR: final-1M = -344.7 (n=1008),
  final-2M -345.9, trajmean -347.9 — all three agree, genuinely
  settled (flat -342..-354 from 3.5M on; single improvement
  -505 -> -345 over 3.0-4.0M, then a dead plateau; NO cycles, NO
  recovery, never positive). That is 187 BELOW the no-channel floor
  (-157.5): a stationary random latent point with the dd target is
  ACTIVELY HARMFUL, not neutral. Telemetry: dist pinned 2.07-2.09
  (random-pair angle baseline — never approached g_fix), prog
  ~0.002-0.003, out_std frozen 0.9295 (apparatus confirmed).
  ctrl_cost MISSING — 748 launched before the ctrl_cost patch; the
  -345 cannot be decomposed into futile-push tax vs backward motion.
  READS (per pre-registration this is the negative/ambiguous
  branch): (a) v22 live-proposer dd +423.0 vs fixg frozen-proposer
  dd -344.7 => the live proposer is worth ~ +770 points on the SAME
  target — "consistent with the proposer earning its keep"
  (softened phrasing per pre-registration; staleness + lottery
  confounds below). (b) Sanctioned qualitative v28-vs-fixg read:
  v28 cycles and repeatedly recovers; fixg cannot re-anchor and
  NEVER recovers — coheres with cycle=budget-exhaustion (re-anchor
  buys the fresh budget; fixg has no re-anchor step, so no fresh
  budget, so a permanent plateau at the exhausted level).
  (c) Stationarity hypothesis read: fixg did NOT recover above
  floor, so stationarity alone does not rescue a point target — BUT
  the alignment-lottery confound applies to fixg too (g_fix is one
  random draw, exactly like u_dir), so a single bad draw cannot
  condemn points as a class either. Current single-draw lattice:
  stationary point -344.7 settled | stationary direction +328.2
  after a 6.5M flip | live-proposer point (dd) +423.0 cycling.
- fixg DISAMBIGUATOR TRIGGER FIRED BUT HELD: the pre-registered
  v21-style persistent-referent arm (fixed OBSERVATION re-encoded
  each iteration) was conditioned on fixg negative — trigger met.
  HELD rather than submitted: the alignment-lottery framework now
  contaminates what it would test (a reset-state referent is stable
  in MEANING but both "bad content" and "staleness" stories predict
  harm for it — weak separation), and the queued lattice
  (751 dirf / 755 antipode / 753 / 752 / 754) resolves the live
  questions first. Revisit after the antipode lands.
- SAWTOOTH: DEAD VIA THE PROG CHANNEL (robust route, reviewer;
  supersedes the angle argument as the basis): rollout_goal_prog at
  0.0007-0.0010 is a DIRECT measurement of collection, 2-3 orders
  below target scale — nothing was ever collected, no budget
  consumed, nothing to exhaust-and-re-anchor. Also the reviewer's
  geometry concession was an OVER-concession: dist is blind to
  RELOCATION only, not to APPROACH (10% closing => dist ~1.8, outside
  the whole 1.90-2.13 band) — so the flat band independently excludes
  approach, and the sawtooth (which requires approach as phase one)
  is doubly dead.
- CONTRADICTION REPAIR (reviewer): the "branch 2 / along for the
  ride" conclusion and the belief-deflation-cycle reading cannot both
  stand — the dichotomy was false (dist-flat is consistent with
  passive instability AND with relocation-cycling that dist cannot
  see). Status line replacing both: SAWTOOTH EXCLUDED;
  PASSIVE-INSTABILITY vs BELIEF-DRIVEN UNRESOLVED — now updated by
  the crash-hinge analysis below.
- CRASH-HINGE LEAD-LAG @50k GRAIN WITH MEASURED SEs (reviewer's
  better-powered test; 3 crashes: 744 @~4.95M, v22 @~2.9M, v22
  @~5.25M): (1) PROPOSER REORGANIZATION NEVER LEADS — out_std rises
  only AFTER the return cliff at all three crashes (744: flat 0.0023
  until returns already negative; v22c1: rise at 3.0M+ post-crash;
  v22c2: flat 0.0021 until after the 5.25M cliff), and follows the
  recovery too. RELOCATION-DRIVEN CYCLING IS EXCLUDED AT EVERY HINGE.
  (2) Leading indicators are HETEROGENEOUS: 744's crash is led by
  wmax (realized efficiency) declining monotonically ~350k+ before
  returns peak-and-crash, promise following wmax — delivery-side
  lead, readout-consistent. v22 CRASH 1 shows the one clean ACTIVE
  signature: promise fell 5.72 -> 4.69 (~18%, resolved) during
  2.25-2.5M while returns ROSE (499 -> 659) and wmax ROSE
  (0.98 -> 1.13) — a belief-INTERNAL downward revision ~400k before
  the crash, not a fit to delivery. v22 crash 2: promise/wmax/returns
  co-decline gradually from 4.85M, cliff 5.25M, no distinguishable
  leader. VERDICT: passive-vs-active remains unresolved and may be
  heterogeneous across crashes; the settled facts are (i) relocation
  never leads anywhere, (ii) v22c1's belief-leads event is real.
- RECOVERY-ONSET POWER CRITIQUE ADJUDICATED BY MEASUREMENT (reviewer
  asked for computed SEs rather than their estimate): floor-region
  bins have SE 0.5-2.0 (episodes near-uniform at the floor), NOT ~20.
  The 744 recovery onset -301.5 -> -276.6 over 6.0-6.45M is >10 SE
  with out_std flat at 0.00244-0.00251 throughout — RETURNS-LEAD AT
  THE RECOVERY IS RESOLVED after all. Standing results: recovery =
  returns lead (resolved); crashes = mixed per above.
- BELIEF-FIDELITY DEMOTION, PRIMARY ARGUMENT (reviewer — replaces
  lead-lag as the basis): v25, the family best at +752.7, runs
  promise 16.3-18.4 against wmax ~2.1 — a ~9x overestimate, never
  correcting, for 8M steps, with no crash. By our own diagnostic
  definition ("promise >> what_real_max persistently = runaway
  fantasy") the BEST arm is maximally pathological on belief
  fidelity all run. Belief-map fidelity is therefore not the
  performance lever; delivery-corrected promise stays demoted for
  v29 on this timing-free argument, with the recovery lead-lag as
  support.
- CYCLE TEST PHRASING DOWNGRADED (reviewer): "consistent with
  exhaustion, one run per cell, carried by the out_std collapse
  signature" — not "branch 1 established." v22/v25 differ on target
  form AND gradient norm AND performance level, and crash-pattern
  evidence is near-circular with returns. The carrying channel:
  out_std is spread ACROSS STATES (shell-normalized coords ~1.0 per
  dim) — v22's 0.002 = ONE state-independent goal (total collapse),
  v25's 0.081 = weakly state-dependent all run; parameterization
  confound checked clean by reviewer (same head, init, statistic).
- 744-RESIDUAL ATTRIBUTION WIDENED (reviewer): "constant-direction
  pull = -76..-155 band" is falsified by 746 (+328.2, also a
  constant-magnitude direction pull, no damping, no decay). The
  500-600 point gap 744-to-dd stands in magnitude but is now
  three-way: damping | distance-decay | STATIONARITY (744 chases a
  proposer-moved point; 746's vector fixed 8M) — with 746's own
  legitimacy contingent on the antipode (755). fixg's landed -344.7
  bears on stationarity (stationary point NOT rescued) subject to
  its single-draw lottery caveat.
- INSTRUMENT VERIFIED (direct tfevents check, not timestamp
  inference): 751's run carries diagnostics/dir_reward_corr — mlq
  reads the script at exec, not submit. All three direction arms
  (751/753/755) are instrumented. EARLY SIGNAL, consistent-with only:
  751 @3.3M has dir_reward_corr ~ -0.15 (stable, n=1541) with
  last-500k return -513.8 — a misaligned draw actively harming, the
  predicted early phase. The registered test remains the ZERO
  CROSSING tracking a return flip.
- fixg EXHAUSTION MISAPPLIED (reviewer; corrects read (b) of the 748
  entry): fixg's prog is 0.0020-0.0033 — two orders below target
  scale, same zero-collection regime as 744 — so no budget was ever
  collected and there was NOTHING TO EXHAUST. "Plateau at the
  exhausted level" is corrected to "plateau at the NEVER-STARTED
  level." The same zero-collection argument that killed 744's
  sawtooth cuts here identically; exhaustion cannot explain fixg.
- LOTTERY+SEARCH UNIFIED CANDIDATE (reviewer; POST-HOC SYNTHESIS of
  four single-run arms, invented after seeing all four — CANDIDATE
  ONLY until 753/755 report; earns its keep solely via the
  pre-registered zero-crossing prediction): a fixed random latent
  referent's PHYSICAL meaning drifts as the encoder trains. v25
  farmable = pays for ANY motion, no referent alignment needed, no
  lottery played, no crash, +752.7. v22 live proposer = SEARCHES
  referent space by ascending the belief map, finds and RE-FINDS
  aligned referents as the encoder rotates — cycling = drift-out /
  re-anchor, +423. 746 fixed direction = one draw, no search, lost
  5.7M, rescued by chance encoder drift at 6.5M, +328.2. fixg fixed
  point = one draw, no search, never rescued, -344.7. REFRAME of the
  +770 read: the live proposer is worth +770 as a SEARCH PROCEDURE
  over referents, not as apparatus — removing it strands the policy
  on its draw. Registered prediction: dir_reward_corr negative
  early, crossing zero near any performance flip (point-arm analog:
  Pearson of dd target vs env reward). Confirmation would make
  exhaustion unnecessary everywhere.
- POINT-LOTTERY CONTROL SPEC (registered, NOT queued): fixg @
  value_coef=-1.0. On the SIGReg shell, maximizing distance from
  g_fix approximates pursuing the antipode -g_fix, so negation is a
  second point draw — with the caveat that the equivalence is only
  as tight as the shell constraint (unlike the direction case, where
  linearity makes it exact). Right arm if point-vs-direction ever
  needs firming; not ahead of the current lattice.
- DISAMBIGUATOR HOLD CONVERTED TO A TRIGGER (so held cannot decay
  into dropped): if 755 MIRRORS 746 (lottery confirmed), the
  persistent-referent disambiguator is RETIRED AS UNREADABLE and
  logged as such; if 755 ALSO CLEARS THE FLOOR (structural), it is
  SUBMITTED IMMEDIATELY. Decision made now, before the number
  exists. (Reviewer's process ruling adopted: a pre-registered
  trigger commits to a READING, not a launch date; holding because
  the reading became contaminated is legitimate.)
- CRASH-HINGE CONCLUSION NARROWED (reviewer; supersedes "relocation-
  driven cycling is excluded at every hinge"): out_std is spread
  ACROSS STATES — a proposer whose output MEAN drifts to a different
  shell region with unchanged spread shows flat out_std. The
  statistic is structurally blind to relocation (the very caveat
  adopted earlier, then misapplied by me). Correct statement: THE
  SPREAD OF PROPOSED GOALS NEVER LEADS — proposer COLLAPSE or
  DISPERSION is excluded as a leading driver at every hinge; where
  the goals WENT is unmeasured. Sawtooth stays dead (killed on the
  prog channel, not on out_std).
- v22c1 REREAD — THE UNRESOLVED PAIR (reviewer): promise =
  what(propose(zb)).mean() and wmax = what(zb).max() come from the
  SAME head. At v22c1 the head's valuation went DOWN on proposed
  goals while going UP on real states, simultaneously. Two
  producers, neither passive-readout: (a) RELOCATION — proposer mean
  moved into a lower-valued region, spread unchanged (invisible to
  out_std by construction); (b) HEAD REVISION localized to the goal
  region. My "belief-internal downward revision" read was (b)
  presented as the only option; (a) fits the same numbers and is
  exactly what the narrowed point above no longer excludes. Logged
  as unresolved (a)-vs-(b). Note (a) is also the lottery+search
  candidate's predicted re-anchor precursor.
- REGIME-DEPENDENT SE LESSON (reviewer): bin SE in this family
  depends on regime — the floor regime (futile constant push,
  return ~ -0.1*sum(a^2)*1000, near-zero velocity) is close to
  deterministic (SE 0.5-2), the learning regime is 10x+ looser.
  Consequences: 744's +-500 swings are astronomically resolved, the
  -155.3-vs--75.7 window distinction is well above noise, and small
  moves are readable at the floor but NOT in the learning region.
- v29 INSTRUMENTATION (registered NOW for any live-proposer run;
  neither scalar consumes RNG — cannot attach to the queued fleet,
  all fixg copies at proposer_updates=0): (1) cos(mean proposed
  goal, frozen iteration-0 reference) — measures relocation
  DIRECTLY; frozen-anchor drift accumulates and is readable, unlike
  consecutive-iteration cosine which is dominated by jitter
  (supersedes the earlier consecutive-cosine spec). (2) what()
  evaluated on a FROZEN probe set of latents captured at iteration 0
  — head revision detector. Together: probe values move = head
  revising; promise moves while probe holds = goals relocated.
  Separates (a) from (b) cleanly.
- 747 v25-HOPPER FINAL — KINETIC-BIAS MECHANISM CONFIRMED ON ITS
  REGISTERED TEST: final-1M return = 41.3 (PPO ref ~2000+);
  wm_delta_var 1.85-1.89 = 86% of the HalfCheetah level (2.15) —
  NOT low; episodic_length PINNED at 23-29 of 1000 (termination
  floor) for the entire run. High latent motion + instant death =
  the registered signature (the refuting branch required
  wm_delta_var LOW). MECHANISM (reviewer): v25's farm rate is
  proportional to wm_delta_var — to first order a bonus on LATENT
  KINETIC ENERGY. HalfCheetah (no termination, no alive bonus):
  "move a lot" is nearly collinear with reward. Hopper (+1.0
  alive, terminates when unhealthy): "move a lot" is the fastest
  route to falling, forfeiting all future reward. CROWN CAVEAT
  UPGRADED from prediction-tracking to MECHANISM-CONFIRMED: +752.7
  is an artifact of HalfCheetah's reward being nearly collinear
  with latent motion. v25 = best-on-tuning-environment, not king.
- WALKER2D PREDICTION ON RECORD (registered BEFORE data; job 756
  embopt_v25_walker submitted, queued): Walker2d has an alive
  bonus with termination, so the same mechanism predicts v25 fails
  there too — signature: wm_delta_var near HC level + episodic
  length pinned near termination floor. Confirmation would make
  the crown a one-environment result on two independent counts;
  refutation (Walker2d succeeds) would localize the failure to
  Hopper's dynamics instead of the mechanism.
- 745 TENSION RESOLVED BY DECOMPOSITION (reviewer): if the farm is
  a kinetic bonus, why was 745 kin-live (explicit kinetic bonus at
  LARGER norm 0.298) inert at -161.2? Because v25's target =
  directional pull toward a proposer-SEARCHED goal + farm residual.
  The farm supplies PERPETUITY (stops the telescoping), the
  searched direction supplies CONTENT. Strip direction => 745,
  inert. Strip search => 746/fixg, lottery. v25 keeps both.
  Consistent with lottery+search, and it sharpens v29: PERPETUITY
  WITHOUT THE KINETIC BIAS — the two have been the same term in
  every version so far, which is why the family kept choosing
  between farming and exhaustion.
- v29 CONSEQUENCE, WRITTEN NOW: DO NOT INHERIT THE FARM. The farm
  term is the part that does not travel between environments.
  v29 = searched (live-proposer) referent + perpetual
  direction-carrying credit WITHOUT latent-kinetic payment + the
  registered instrumentation (frozen-reference goal cosine; frozen
  probe set). FROZEN-PROBE PURPOSE WIDENED: what() on iteration-0
  probe latents separates relocation from head revision AND
  directly measures encoder-drift staleness (decay of frozen-probe
  values IS staleness — the concrete no-foresight mechanism for
  v22c1's promise fall: what() is only held up where training
  signal arrives, so a vacated region decays while real states
  rise). Do not cut it as narrow-use.
- SIGN CORRECTION (reviewer; supersedes my "re-anchor precursor"
  connective note): the proposer ASCENDS the frozen belief map, so
  relocation RAISES promise by construction. A promise dip against
  rising wmax is the STIMULUS for re-anchoring (the landscape moved
  out from under a lagging ascent, 8 updates/iter), not the
  re-anchor; the re-anchor shows up as promise RECOVERING. This
  gives lottery+search a two-phase signature: stale referent
  devalued (promise dips, wmax holds) -> proposer re-climbs
  (promise recovers) -> delivery follows (returns recover).
  Discriminating prediction: promise recovery LEADS return
  recovery.
- v22 RECOVERY HINGE @50k GRAIN (the untested hinge; registered
  readings applied): PROMISE RECOVERY LAGS. Returns began a
  RESOLVED slow climb from ~5.9M (-262 -> -198 by 6.6M at floor
  SEs ~1) while promise was STILL FALLING; promise trough is
  6.65-6.75M (2.266), upturn from 6.75-6.80M. Promise's rise does
  precede the steep acceleration (7.05M+, -86 -> +433 in 250k) by
  ~250-300k, and out_std's peak (0.00467) is coincident with the
  steepest return bin (7.20-7.25M) — but the registered reading
  concerned recovery ONSET, and returns turn first. PER THE
  REGISTRATION: the belief channel is downstream at the recovery
  too; lottery+search LOSES the re-anchor mechanism it was invoked
  to supply and reverts to a story about single-draw arms only.
  Its one remaining leg is the dir_reward_corr zero-crossing
  prediction on 751/753/755. Accumulating cross-hinge picture: at
  every hinge tested (three crashes, two recoveries), no
  belief/proposer channel has ever led a return turn — the goal
  channel's +580 value (v22 vs vc0) must come from continuous
  gradient shaping, not from the visible cycle turns.
- (a)/(b) WIDENED TO A TRIPLE (reviewer): promise =
  what(propose(zb)).mean() over the CURRENT batch — during v22c1
  the policy was provably moving (returns 499 -> 659), so (c)
  STATE-DISTRIBUTION SHIFT (different states -> different
  proposals -> different map regions) is a third producer of a
  promise fall, live in that exact window. (a) relocation is now
  additionally DISFAVORED on structural grounds (ascent raises
  promise). v29 INSTRUMENT ADDED: log what(propose(z_probe)).mean()
  on the SAME frozen probe set — live-promise minus probe-promise
  isolates (c), frozen-probe what() isolates (b), frozen-reference
  cosine isolates (a). Three scalars, one probe set, decision
  table closed.
- HOPPER MEASUREMENT VERIFIED CLEAN (reviewer): wm_delta_var is
  done-masked (v25:935-936 — dmask drops transitions where done
  fires at t+1), so reset jumps are excluded and 1.85-1.89 is real
  WITHIN-EPISODE motion. The circularity worry (dying fast inflating
  the statistic) is closed; MECHANISM-CONFIRMED stands. Refinement:
  for a hopper, FALLING is the motion-maximizing trajectory — the
  kinetic bias selects falling DIRECTLY, not via incidental
  flailing; 23-29 step episodes are what that selection looks like.
- 750 DECOUPLING TEST (registered readings applied; wm_delta_var vs
  returns, 500k bins): FIRST BRANCH CONFIRMED — wm_delta_var stays
  flat-to-rising (2.10 -> 2.17) while returns decay +484 -> -140
  over 1-6M; ctrl_cost holds ~0.40-0.41 throughout. The policy kept
  buying latent motion and it stopped buying reward: early phase =
  collinearity (latent near init tracks obs change ~ velocity) that
  EXPIRED as SIGReg/prediction reshaped the representation. Caveat
  on record: wm_delta_var is a screen (also moves for world-model
  reasons). Clean instrument kin_reward_corr (masked Pearson of
  per-step ||d||^2 vs env reward) ADDED to the kinf file — 754 penf
  (still queued) carries it; 752 kinf1 (already running) does not.
- TWO NON-STATIONARITIES (reviewer; makes the floor-vs-below-floor
  lattice structural): SIGReg pushes the latent marginal toward
  N(0,I), which is ROTATION-SYMMETRIC — the encoder's basis can
  rotate freely. (1) BASIS ROTATION breaks any target defined by
  fixed latent STRUCTURE (u_dir, g_fix pick out coordinates):
  746's flip, fixg's permanent loss. (2) REPRESENTATIONAL CHANGE
  (what the encoder spends variance on) breaks targets defined by
  latent MAGNITUDE: 750's decay. ||d||^2 is rotation-INVARIANT, so
  kinetic arms cannot lose the rotation lottery — structural
  prediction, holds across four arms: direction-free 745 (-161.2)
  and 750 (-109.5) sit AT floor; fixed-referent fixg (-344.7) and
  746-early (~-425) sit far BELOW it. v29 SPEC, SHARPENED FORM:
  every latent-geometry-anchored target inherits one of the two
  non-stationarities because the encoder never stops training. The
  only immune referents are (i) continuously re-derived against
  current data — the search procedure — or (ii) anchored to REWARD
  rather than latent geometry.
- MAGNITUDE-vs-DIRECTION CAUTION (reviewer): 751's -514 is a single
  lottery DRAW, not comparable to lottery-free kinetic. The right
  statistic at norm ~1.0 is kinetic vs the MEAN over the antipode
  pair (751+755) — the exact-antipode pair estimates the
  expectation over draws. If 755 is symmetric around floor the
  honest split shrinks far below 600 and might vanish.
- SYNTHESIS SPLIT (reviewer, accepting their prediction's failure):
  LOTTERY (fixed latent referents = single draws whose alignment
  drifts) is ALIVE, unaffected, retains the zero-crossing test on
  751/753/755. SEARCH (proposer value = re-anchoring stale
  referents) is DEAD at five of five hinges — failed its first
  test. Logged separately so the dead half cannot borrow
  credibility from the living half.
- "DOWNSTREAM" OVER-REACH CORRECTED (reviewer): at v22's recovery
  the belief channel moved OPPOSITE delivery for ~800k steps
  (returns climbing from ~5.9M while promise still fell to its
  6.65-6.75M trough). A downstream readout co-moves; it does not
  anti-correlate for 800k. The test excluded LEADING only. Status:
  at this hinge the belief channel neither leads NOR tracks —
  UNRESOLVED third state, same over-reach shape as the out_std
  relocation entry.
- AMPLIFIER RE-FILED WITH PRE-DATA STANDING (reviewer): "re-anchor
  amplifies a recovery already underway" is not a rescue of dead
  SEARCH — it is a PREDICTION of continuous-shaping (stale referent
  = useless gradient; re-anchor restores a useful one => speeds
  learning in progress without initiating it). Registered as a
  consequence of the adopted hypothesis; adjudicated on a future
  live-proposer run by the frozen-reference cosine (relocation
  between grind onset and acceleration => amplifier; none => not).
- CONTINUOUS-SHAPING TEST (registered readings applied; v22 cycle
  2, 250k bins, three gradient-quality channels): BRANCH 2 (all
  flat) DECISIVELY EXCLUDED — the channels swing hard with the
  cycle, so the cycle IS goal-channel-entangled and the +580 is
  not bought by a constant. BRANCH 1 SPLITS: the ANTICIPATION
  clause FAILS — nothing degrades approaching the crash
  (calib_abs improves 0.068 -> 0.062, act_grad holds 0.60-0.62;
  degradation begins AT the crash bin) — while the RESTORATION
  clause is met by action_grad_value: collapse 0.62 -> 0.42 at the
  crash/trough, steady climb through the floor grind (parallel to
  the slow return climb from ~5.9M), 92% of pre-crash level BEFORE
  the 7.05M acceleration, peak 0.628 within it. prog_calib_abs
  stays degraded through the whole recovery (worst bin 7.0-7.25M),
  recovering only after. UNREGISTERED OBSERVATIONS (flagged, not
  concluded): goal_removal_action_mse — the goal channel's grip on
  the ACTION is maximal during the floor (0.53-0.58) and FALLS
  during recovery (0.49 -> 0.42), i.e. the floor phase is
  goal-channel-DOMINATED action-wise; if anything this reads as
  the channel holding the policy at the floor, not lifting it out.
  NET: continuous-shaping survives with a mechanism for
  restoration-tracks-recovery, gains no crash-anticipation
  support, and inherits a new tension (influence peaked at the
  floor). The goal channel varies with the cycle; which direction
  the causality runs remains open — same unresolved status as the
  belief channel.
- CONVERGENCE CLAIM CORRECTED (reviewer): "three independent lines"
  was ONE AND A HALF. Farm environment-locality (Hopper) and the
  non-stationarity taxonomy (750 decay) are the SAME fact — LATENT
  GEOMETRY IS NOT REWARD GEOMETRY — cross-sectioned across
  environments and across training time respectively. The
  five-hinge line rested on continuous-shaping, which won by
  elimination; its registered test has NOW run (see shaping-test
  entry): branch-2 excluded (channel is cycle-entangled), but
  causality open and grip-maximal-at-floor tension added. Spec
  basis, honest form: ONE well-confirmed fact twice measured, plus
  one test that came back entangled-but-unresolved. "The spec is
  starting to write itself" is retracted as the overcommitment
  feeling this lineage's failure mode rides in on.
- "REWARD-ANCHORED" IS WEAKER THAN IT SOUNDS (reviewer): v22's
  proposer already ascends what(g) — the referent is ALREADY
  selected on reward criteria — and it still went stale, cycled,
  lost to v25. Reward-anchoring the referent's VALUE does not
  reward-anchor its DOMAIN: the referent is still a point in a
  space that never stops moving. STRENGTHENED CONSTRAINT: the
  target must not NAME A POINT IN LATENT SPACE AT ALL — any
  commanded latent referent inherits the non-stationarity through
  its domain regardless of how it is chosen.
- v29 CANDIDATE NAMED — POTENTIAL-BASED: target = what(z') -
  what(z), the belief map used as a POTENTIAL, recomputed every
  step. Checklist vs everything established: no referent => no
  lottery; gradient of what rotates WITH the basis => immune to
  basis rotation; what retrains against measured efficiency =>
  tracks representational change; no distance => no farm; no
  arrival => no futile push; direction-carrying and DERIVED (not
  drawn); deletes proposer/goal-input/referent/farm/exhaustion in
  one move while keeping the one component plausibly doing work
  (reward-rate map supplying directional gradient). SAFETY
  PROPERTY: potential-based shaping in the Ng sense — structurally
  incapable of dragging the policy below the no-channel floor,
  which is precisely how every catastrophic arm failed (fixg
  -344.7, 746-early -425, v26 -589). CAVEAT (mine): Ng's theorem
  is for additive reward shaping with gamma-discounted F =
  gamma*Phi(s')-Phi(s); here the term enters via a learned Q-head
  ascended in a weighted policy objective — the invariance
  argument is structural/heuristic, not imported as a proof. Also
  on record: upside capped (buys learning speed, not a better
  optimum; will not reproduce +752.7 — which was HalfCheetah-local
  anyway); telescopes per episode (746's perpetuity profile
  without the draw); quality bounded by what's local monotonicity
  — instrumentable day one via prog_calib_abs-style calibration.
  PRIORITY: reviewer's registered conditional (flat => natural
  replacement) did NOT fire — the channel is entangled — but the
  grip-maximal-at-floor observation independently supports a term
  that structurally cannot hold the policy at the floor. In the
  candidate set; priority to be settled against the shaping-test
  tension.
- GRIP-vs-PERFORMANCE CANDIDATE LAW — CROSS-ARM TEST NULL
  (registered readings applied): goal_removal_action_mse (grip = a
  direct measure of goal conditioning of behavior; g is a policy
  input) final-1M vs final-1M return, vc=1.0 arms only per the
  registered caveat (744@3.0 / 750@3.4 held out; vc0 zero point
  grip 0.145 — input-perturbation floor):
    v25 +752.7 grip 0.603 | v22 +423.0 grip 0.444 | 746 +328.2
    grip 0.260 | 745 -161.2 grip 0.557 | fixg -344.7 grip 0.350 |
    v26 -589.5 grip 0.733
  Spearman rho = -0.14 (n=6): NO relationship. The BEST arm has
  near-TOP grip; the law's exemplar (v26, 0.733) coexists with its
  counterexample (v25). PER THE REGISTRATION: the within-v22
  anti-correlation (grip maximal at floor, falls in recovery) is a
  WITHIN-RUN PHASE DYNAMIC, not a design principle, and constrains
  v29 much less. The potential-based candidate loses this support
  and stands on the non-stationarity argument alone (its
  registered fallback). Cross-arm grip LEVEL reflects design;
  within-run grip CHANGE reflects phase — do not mix the
  cross-sections.
- ACCELERATION-BIN CONFOUND (reviewer): inside the 7.05M+
  acceleration the critic's targets move fast, so calib_abs
  degrades MECHANICALLY (critic lag) and act_grad's peak there
  carries no causal content. The clean window is the FLOOR (static
  policy). Consequences: the shaping test's "restoration met by
  act_grad" is WEAKENED (drawn partly from the contaminated bin);
  the anticipation FAILURE is unambiguous (clean window). The
  within-v22 grip anti-correlation rests on the floor phase and
  survives.
- SCALAR-ARCHAEOLOGY STOP (agreed, both parties): eight scalars on
  finished runs are exhausted as of the cross-arm test. Remaining
  questions (relocation vs head-revision vs distribution-shift,
  amplifier, grip causality) all require the v29 instruments
  (frozen probe set, frozen-reference cosine, probe-promise) and
  cannot be recovered retroactively. Further decomposition of
  existing scalars risks MANUFACTURING structure — the lineage's
  known failure mode. No more re-analysis of finished runs; new
  conclusions come from new instrumented runs.
- THEOREM DOES NOT TRANSFER — Q-HEAD SAFETY CLAIM REFUTED (reviewer,
  reversing their own claim after checking v25:684-695): the family's
  policy update is a deterministic one-step ascent on three frozen
  regression surfaces (obj = reward_coef*rhat + rate_coef*qrate +
  value_coef*qprog) — NO ratio, NO advantage, NO trajectory sum, NOT
  PPO. Ng invariance comes entirely from the potential telescoping
  inside the return; with no trajectory sum nothing telescopes. Under
  this objective what(z')-what(z) is ONE-STEP LOOKAHEAD, correct only
  if what ~ V*; what is a measured efficiency surface, so a bad
  potential CAN drive below floor. Below-floor-impossibility is
  REFUTED for the Q-head route.
- v29 FORK TAKEN — LITERAL ADDITIVE SHAPING IN ACTUAL PPO: F =
  gamma*Phi(s') - Phi(s) added to the env reward inside
  ppo_continuous_action's update, Phi = what(z). The decisive reason
  (stronger than theorem-availability): under potential-based shaping
  ANY Phi yields the same optimal policy, so encoder drift in Phi —
  the lineage's central pathology, killer of every fixed-referent
  arm — changes only the learning signal and CANNOT move the optimum.
  Drift becomes structurally harmless instead of engineered-around.
  Single PPO loss (satisfies the user's one-loss law). Residual risk
  on record: what is trained on visited states only; ascending it
  pushes off-distribution where it is unreliable (v22's 9x runaway
  fantasy) — under literal shaping this degrades to NO-HELP, not
  active harm; meter = promise-vs-wmax gap (already exists).
- FIVE IMPLEMENTATION TRAPS (registered for the v29 file; #1 severe):
  (1) TERMINAL: Phi(terminal)=0 — terminating transitions use
  F = -Phi(s), NOT gamma*Phi(s')-Phi(s); a sign error here
  manufactures die/survive incentives exactly where Hopper/Walker2d's
  alive bonus and the confirmed kinetic mechanism live, invalidating
  the run. (2) TRUNCATION != TERMINATION: on time-limit truncation
  Phi(s') is NOT zeroed — reuse the baseline's bootstrap distinction.
  (3) SAME gamma as GAE (0.99). (4) NORMALIZE Phi by running std
  (what's scale is arbitrary; unnormalized Phi makes lambda
  meaningless across envs). (5) ORDER: add shaping AFTER
  NormalizeReward so the normalizer's statistics stay about the true
  task.
- BASELINE GAP — THE QUESTION THAT OUTRANKS THE FORK (reviewer): the
  family's policy update has not been PPO for many versions, and the
  actual baseline ppo_continuous_action.py has NEVER been run at
  matched settings in this investigation. No matched run exists in
  runs/. JOB 757 ppo_baseline_hc8m SUBMITTED (HalfCheetah-v4, 8M,
  seed 1, num-envs 16, same final-1M statistic). REGISTERED READINGS:
  baseline >> +752.7 => the entire lineage has been optimizing far
  below the reference; every "floor" is the floor of a broken policy
  update; the +580 goal-channel value is relative to that; mechanisms
  (lottery, two non-stationarities, Hopper collinearity) survive but
  v29's foundation changes and the fork becomes MANDATORY. baseline
  <= ~+752.7 => family is competitive and the fork is a clean
  improvement. Number goes in the ledger BEFORE the v29 spec is
  frozen.
- 751 dirf@1.0 FINAL: -513.8, DEAD FLAT (every 1M bin -513/-514,
  settled from bin 1) — deepest sustained floor in the fleet, 169
  below fixg. dir_reward_corr NEGATIVE ALL RUN (-0.20 -> -0.14,
  no zero crossing). NO CROSSING + NO FLIP = the contrapositive leg
  of the registered zero-crossing prediction — consistent with
  lottery. Contrast 746 (live proposer, different draw): flipped at
  6.5M. 751 (frozen apparatus, this draw): never. Note 751 vs 746
  differ in BOTH draw and apparatus, so 755 (same draw mirrored,
  same apparatus) remains the only clean control.
- THIRD ANTIPODE BRANCH REGISTERED PRE-DATA (751's -514 makes it
  live; the original registration covered only both-above and
  mirror): if 755 ALSO lands far below floor, then direction
  conditioning at norm ~1.0 on the frozen apparatus is harmful
  INDEPENDENT of draw — kills structural-good AND the lottery
  explanation for this operating point, and indicts the
  conditioning channel itself (g input dragging actions) rather
  than the draw. Readings now cover: both above / mirror / both
  below.
- 753 DOSE-RESPONSE REGISTRATION (pre-data; realized only now):
  dirf03 runs the SAME file and seed, hence the IDENTICAL u_dir
  draw as 751, at value_coef 0.3 — a dose-response probe on the
  same bad draw. Readings: 753 near floor (-157) => the harm needs
  strong coupling; 753 near -513 => conditioning harm saturates
  below 0.3; intermediate => graded dose-response.
- 752 kinf1@1.0 FINAL: -224.3 (shape: +10/+50 early, settled ~-225
  by 3M) — 67 BELOW the vc0 floor, denting "kinetic sits AT floor."
  Kinetic trio is non-monotone in coefficient: 745@1.0-live -161 |
  750@3.4 -109 (early +415) | 752@1.0 -224 (weak early). AMENDED
  structural claim: direction-free targets occupy a BAND AROUND the
  floor (-109..-224), fixed-referent arms a band far below
  (-345..-514) — the rotation-invariance separation SURVIVES as
  band-vs-band with a ~120-point gap, but "cannot lose the lottery"
  no longer means "lands exactly at floor." Single-seed caveat on
  every band edge.
- CROSS-ARM ENTRY AMENDED (reviewer): the rho is DROPPED from the
  argument — at n=6 its SE is ~0.45, indistinguishable from 0, -0.6,
  and +0.3 alike; report only as "n=6, no power." What kills the law
  is the DESIGN-MATCHED PAIR: v25 and v26 are both
  farmable-projection + live-proposer (matched on target form,
  apparatus, AND g-informativeness), grips 0.603 vs 0.733, returns
  1342 points apart, only substantive difference the reward heads. A
  counterexample inside a matched pair needs no correlation
  coefficient. Null stands on that pair.
- removal_mse INSTRUMENT RULE (reviewer; instrument lesson, NOT a
  law rescue — and their disclosure is on record: the informativeness
  confound was noticed AFTER the unfavorable result, so the
  registered null stands regardless; the confound explains the ORDER
  without rescuing the CLAIM, since v25/v26 are matched on the
  confound and the law fails inside the pair): grip clusters by how
  much each arm's DESIGN makes g informative — 746 lowest (target
  u_dir.d ignores g, policy learns to ignore an irrelevant input),
  fixg low (constant input absorbed into bias), v25/v26 highest (g
  fully determines the objective). removal_mse is therefore a
  WITHIN-ARM instrument only: valid for phase tracking inside one
  run (the within-v22 anti-correlation survives on exactly that
  footing), INVALID for ranking designs. Carried into v29.
- POTENTIAL-BASED CANDIDATE STANDING (stated plainly, both parties):
  selected by CONSTRAINT-ARGUMENT ALONE, zero measurement support —
  the weakest footing of anything promoted today, and elimination-won
  hypotheses are 1-for-3 against direct tests in this investigation.
  REQUIREMENTS BEFORE THE SPEC FREEZES: (1) the 757 baseline number
  in the ledger first; (2) a first-class registered KILL GATE in the
  v29 spec (success/failure thresholds vs the true PPO baseline plus
  a shaping-off ablation arm), written before the first v29 run
  starts, not after.
- USER DIRECTIVE (2026-07-29) + SCALE REALITY CHECK: NO cross-env
  runs — the family is nowhere near ready (756 walker CANCELLED
  mid-run; no further Hopper/Walker until competitive on
  HalfCheetah). The repo's own frontier: dg_beta family scores
  4,100-5,700 last-20 at only 1-3M steps on HalfCheetah (61 runs
  scored), user reports ~8k attainable at full length. Family best
  is +752.7 at 8M — roughly 10x BELOW the in-repo frontier. This
  confirms the baseline warning WITHOUT waiting for 757: every
  comparative number in this ledger ("+580 vs vc0", "best in
  family", every "floor") was internal to a family whose policy
  update is not PPO, in a regime an order of magnitude below what
  this repo demonstrably achieves. The within-family MECHANISMS
  survive (they are internal contrasts); the VALUE judgments do
  not. 757 (plain ppo_continuous_action.py, matched settings) still
  queued — its number goes at the TOP of this ledger and sets the
  scale for everything else. v29 is built on a real PPO chassis or
  not at all.
- v29 SPEC ITEMS (reviewer, final pre-watch batch): (1) ABLATION !=
  BASELINE BY DEFAULT — at lambda=0 the v29 file still trains the
  world model, whose torch.randint minibatch sampling advances the
  global RNG stream and diverges the realization from plain PPO
  after the first WM update (single-seed swings exceed any effect
  size). FIX BY CONSTRUCTION: world-model sampler gets its OWN
  torch.Generator + env wrapper stack identical to baseline =>
  lambda=0 is bit-identical to ppo_continuous_action.py; the method
  provably degenerates to the reference; no seed-realization
  defense available. (2) LEARNING-SPEED GATE: potential shaping
  provably cannot move the optimum, so a final-score-only gate
  tests the one thing the method cannot improve. Gate at matched
  1M/2M/4M checkpoints + final. PRE-REGISTERED: helps-early +
  neutral-late = SUCCESS for this method class; neutral-early = the
  substantive death (potential carries nothing the critic lacks).
  (3) FALSIFIABLE HYPOTHESIS (spec header): shaping with what(z)
  helps IFF the world model's latent value surface generalizes
  across states FASTER than the PPO critic, transferring value
  information the critic has not yet acquired. Meter: gap between
  what(z)'s implied ordering and the critic's on the same states.
  (4) Phi MUST NOT BE THE PPO CRITIC (r + gamma*V(s')-V(s) is the
  TD error — shaping with the critic double-counts the advantage);
  the potential must be an INDEPENDENT value estimate, which is
  exactly what the what(z) head trained on measured efficiency is.
- BASELINE CANCELLED PER USER; TRUE FRONTIER IDENTIFIED: 757 plain-
  PPO baseline cancelled ("waste of time" — user). The reference
  point is the repo frontier, now measured: iterthink_v24 tpo
  family = 10,410 last-20 at ~7-8M (multiple runs >10k); pmpo =
  9,222; dg_beta = 4,100-5,700 at 1-3M. Family best +752.7 final-1M
  is ~14x below frontier (estimator note: frontier figures are
  last-20, family figure final-1M; v25's last-20 @8M ~492-729 —
  the gap survives any estimator by an order of magnitude).
  Reviewer's rebase argument ADOPTED and superseded in the same
  stroke: v29's base is not plain PPO and not dg_beta — it is the
  FRONTIER chassis (iterthink v24 tpo lineage), since potential
  shaping is base-agnostic by construction and a success condition
  measured against any lesser chassis produces a regression.
- SMALLEST DECISIVE EXPERIMENT FIRST (reviewer, adopted): before
  any world-model apparatus, test the bare shaping hypothesis on
  the frontier chassis: a small independent MLP regressing
  discounted return from raw observations as Phi, shaped in
  additively with the five traps honored. Helps => mechanism real;
  only then is "does a latent world-model potential beat a raw-obs
  potential" a well-posed question worth the apparatus. No help =>
  the hypothesis dies at fifty times less surface area. Kill gates
  at 1M/2M/4M against the same-chassis lambda=0 control
  (bit-identity construction via separate torch.Generator).
- BAND MECHANISM QUANTITATIVE (reviewer arithmetic on landed
  finals): return = velocity_total - ctrl_total decomposes: 750
  kinetic@3.4 ctrl 407 / velocity +298 | 752 kinetic@1.0 ctrl 349 /
  velocity +125 | 751 direction@1.0 ctrl 516 / velocity +2.
  Kinetic arms BUY REAL MOTION and overpay; the anti-aligned arm
  pays the fleet's largest effort for ZERO motion. vc0 lower bound
  recovered from its own return: ctrl >= 0.157/step if velocity
  non-negative; instrumented arms pay 2-3x that. TAX CONFIRMED,
  not merely uncontradicted. Anti-alignment is not expensive
  movement, it is full price for no movement — that is the
  band-vs-band gap's mechanism.
- ADVANTAGE-INVARIANCE THEOREM (reviewer; verified independently —
  supersedes the v29 mechanism story): with V' = V - c*Phi the TD
  residual is invariant TERM BY TERM (delta' = delta), so every GAE
  advantage is bit-identical and potential shaping has NO channel to
  a PPO policy gradient except CRITIC APPROXIMATION ERROR: it
  changes the critic's fitting problem from V to V - c*Phi, nothing
  else. Corrected hypothesis: shaping helps IFF V - c*Phi is easier
  to fit than V and the value-error reduction improves advantage
  quality. CONSEQUENCES (inverting two earlier recommendations,
  reviewer's own): (1) Phi should be as CLOSE TO V as possible —
  best construction is a FROZEN PERIODIC SNAPSHOT of the chassis's
  own critic (residual learning; independent within each phase; in
  return units for free); (2) the raw-obs-MLP "cheap version" was
  the WRONG first move (a second worse critic enlarges the
  residual); (3) primary kill gate = EARLY EXPLAINED VARIANCE and
  value loss vs control — returns are downstream confirmation only;
  no early expl-var gain = mechanism absent regardless of returns.
- CHASSIS FACTS (tpoprobe_v1): normalize_reward=False (trap 5 moot;
  Phi must be in RETURN units, NOT unit-normalized — trap 4
  inverted for this base); gamma 0.99 / gae_lambda 0.95; wrapper
  order keeps RecordEpisodeStatistics BEFORE reward transforms so
  episodic_return stays raw and frontier-comparable (shaping must
  be added in the training loop, never as a wrapper); critic =
  HL-Gauss 511 bins on FIXED raw-return support [-20000,20000] with
  a documented edge-mass failure mode — watch edge_mass during any
  c sweep. Trap 1 (terminations-only mask) stands and is silent on
  HalfCheetah (no termination) — write it correctly NOW or it hides
  until a cross-env run.
- v29 OPEN RISKS (mine, flagged to reviewer): (a) HL-GAUSS
  RESOLUTION: with c=1 the critic's targets shrink toward a
  near-zero residual; on a FIXED raw-space support the usable bins
  collapse to a few ~78-wide central bins (coarse), whereas a
  symlog-space support gives near-zero the FINEST resolution —
  whether the mechanism is expressible on this head depends on
  which space the bins live in (impl agent instructed to report).
  (b) PROBE-CONSISTENCY: tpoprobe scores candidates by r + gamma*
  (1-term)*V_next; once the critic represents V - c*Phi, mixing raw
  r with shaped-critic V corrupts candidate ordering by
  -gamma*c*Phi(s'_cand) unless the site is shaped consistently —
  every reward-value mixing site must close the invariance algebra
  (impl agent instructed per-site). (c) SNAPSHOT REFRESH changes
  the potential every N iters — invariance holds within each phase;
  refresh boundaries are the residual non-stationarity.
  (d) HONEST NARROWNESS (reviewer): on a well-tuned distributional
  critic there may be NO residual worth removing — the substantive
  death, gated by early expl-var.
- PROBE PROGRAM COMPLETE — FINAL THREE ARMS: 754 penf -231.1 | 755
  antipode -477.4 | 753 dirf03 -686.9. Registered readings applied:
  (1) DAMPING DEAD, CLEANLY: penf@-1.0 (-231.1) is sign-symmetric
  with kinf1@+1.0 (-224.3) — 7 points apart, well inside single-seed
  noise. The pre-registered symmetry branch fires; -||d||^2/dz is
  not an active ingredient. The damping-vs-decay question resolves
  toward distance-decay/stationarity by elimination (moot for
  building — old chassis abandoned — but closes the ledger). penf's
  kin_reward_corr ~0 all run (-0.004 -> -0.064): on this apparatus
  the kinetic target never had reward alignment to lose.
  (2) THIRD ANTIPODE BRANCH FIRES — BOTH BELOW FLOOR: 751 -513.8 and
  755 -477.4. Direction conditioning at |vc|=1.0 on the frozen
  apparatus is harmful INDEPENDENT of draw sign: lottery is dead as
  the explanation for the frozen-arm depths, structural-good is
  dead, the conditioning channel itself (g input + pursuit pressure)
  is indicted. 746 (+328.2, live proposer) stands as the family's
  one unexplained direction anomaly.
  (3) DOSE-RESPONSE: FOURTH OUTCOME, outside all three registered
  branches — 753 (same draw as 751, coupling 0.3) = -686.9, the
  WORST number in family history, below v26. Lower coupling produced
  a DEEPER catastrophe (shape: -221 first 2M, then plunge). Harm is
  not monotone in dose; no registered story covers this; single-seed.
  (4) INSTRUMENT LESSON: 755's dir_reward_corr came back POSITIVE
  (+0.13..+0.24) vs 751's negative — OPPOSITE sign, contradicting
  the same-sign prediction. The reviewer's fallback (instrument
  folded the coefficient) is mechanically wrong — the computation
  never touches value_coef. Actual reason: the Pearson is computed
  on the VISITED distribution, which the policy's pursuit shapes;
  the sign tracks the pursued direction, not the draw's geometry.
  dir_reward_corr measures draw-alignment ON-POLICY only; the
  751/755 corr comparison is uninterpretable as geometry, as the
  registration anticipated (returns were primary).
- FINAL FAMILY TABLE (final-1M, HalfCheetah, broken-chassis regime;
  frontier reference ~10,400 last-20): v25 +752.7 | v22 +423.0 |
  746 +328.2 | 750 -109.5 | 744 -155.3(unsettled, trajmean -75.7) |
  vc0 -157.5 | 745 -161.2 | 752 -224.3 | 754 -231.1 | v27 -332.9 |
  fixg -344.7 | 755 -477.4 | 751 -513.8 | v26 -589.5 | 753 -686.9.
  All single-seed. Program closed; all further work on the frontier
  chassis (v29 potshape).
- v29 RISK LEDGER UPDATE (reviewer): (1) HL-GAUSS RESOLUTION RISK
  RESOLVED FAVORABLY from the file — bins uniform in SYMLOG space
  (v_min/v_max = symlog(+/-20000) = +/-9.9035..., value_symlog=True):
  near-zero residuals land where one bin spans ~0.039 raw units vs
  ~194 at frontier scale — ~5000x finer exactly where the mechanism
  concentrates mass. c<1 not forced. SUCCESSOR RISK: the HL-Gauss
  smoothing SIGMA is tuned for the current wide target spread; if
  absolute in symlog units it will over-smooth concentrated residual
  targets (symptom: unimprovable value-loss floor). Sigma is a
  design parameter under shaping, not an inherited hyperparameter.
  (2) PROBE ALGEBRA VERIFIED (reviewer): shaped-r probe score =
  r_raw + gamma*V(s'_cand) - c*Phi(s), candidate-common shift,
  ordering exactly preserved — AND the terminal-candidate case
  requires Phi(terminal)=0 in the shaped reward, so TRAP 1 AND THE
  PROBE SITE ARE THE SAME BUG AT TWO SITES; implemented and tested
  together (five-site algebra trace required in the impl report).
  (3) GATE COMPARABILITY (new, hits the primary kill gate):
  explained variance is a ratio and shaping shrinks its DENOMINATOR
  by orders of magnitude — the shaped arm can have smaller absolute
  value error yet report WORSE expl-var. Raw cross-arm expl-var
  comparison would read success as failure. HARD REQUIREMENT:
  reconstruct V_hat = V' + c*Phi and score against UNSHAPED returns
  (same for value loss) — logged as shaping/expl_var_reconstructed,
  identical computation to the chassis's own expl-var so curves
  overlay. A gate that cannot distinguish success from failure is
  worse than no gate.
- v29 REGISTERED KILL GATE (pre-launch, before any data seen).
  Run: embopt_v29_potshape_v1, HalfCheetah-v4, 8M, seed 1, CLI-
  matched to the k8_eta6_coef1 reference; CONTROL = the existing
  completed run (10,371 last-20 @7.98M) — bit-identity at
  shaping_coef=0 is verified by construction, not re-run.
  PRIMARY GATE (mechanism): shaping/expl_var_reconstructed
  (V_hat = V' + c*Phi scored vs UNSHAPED returns, computed
  identically to the chassis expl-var) overlaid on the control's
  explained-variance curve at 1M / 2M / 4M.
    SUCCESS signature: reconstructed expl-var (and value-loss)
    better than control EARLY (by 1-2M), washing out late is
    fine; returns >= control trajectory.
    DEATH: neutral-early reconstructed expl-var = no residual
    worth removing on an already-good critic (the registered
    likeliest death) -> kill without iteration.
    INFORMATIVE DEATH: expl-var improves but returns don't ->
    mechanism present, not rate-limiting; log and stop.
  SECONDARY: returns vs control at matched steps (downstream of
  the mechanism gate, never overrides an expl-var death).
  WATCH: edge_mass (documented chassis failure mode); sigma
  successor risk is REPORT-ONLY this run (symptom if real:
  unimprovable value-loss floor on concentrated residuals).
  PROCESS NOTE: impl agent built the superseded spec first
  (mailbox timing — revisions queued while it was mid-turn);
  redirected 2026-07-29 with consolidated spec (frozen-critic-
  snapshot Phi, phi_refresh_every=16, NO F normalization, coef
  1.0). Nothing submits before my review + v19-review both pass
  the file against the trap list.
- CORR INSTRUMENT: TWO ERRORS LOGGED (reviewer request), plus the
  structural resolution. Error 1: same-sign prediction failed.
  Error 2: the fallback (value_coef folding) was mechanically
  wrong. RESOLUTION (reviewer; adopted): a policy maximizing
  reward + c*(direction term) settles where marginal reward loss
  balances c * marginal directional gain -> corr(pursued-dir
  motion, reward) is FORCED NEGATIVE at equilibrium regardless of
  draw geometry. Predicts both signs exactly (751 pursues +u:
  -0.15; 755 pursues -u: +0.13..+0.24 on +u axis). CONSEQUENCE:
  the zero-crossing test was STRUCTURALLY INCAPABLE of testing
  the lottery from the start — the lottery's one remaining leg
  was never load-bearing. Instrument measures the equilibrium
  trade-off at the driven operating point (selection effect),
  not draw alignment. Returns-primary registration saved the
  reading.
- 753 VELOCITY DECOMPOSITION (registered one-number carve-out
  from the archaeology stop; logged either way): final-1M
  ctrl_cost/step = 0.320 -> ctrl_total ~ 320, velocity_total =
  -686.9 + 320 ~ -367. CLEARLY NEGATIVE -> 753 learned a
  coherent BACKWARD gait. Different failure from 751 (vel +2.2,
  ctrl 516 = futile push, zero net motion). So weak coupling
  (0.3) produced WRONG-DIRECTION coherent motion while strong
  coupling (1.0) produced expensive immobility. Single seed,
  closed chassis — no story built on it.
- TRANSFER TO v29 (registered before any v29 data): 753 is a
  concrete counterexample to "small coupling is the conservative
  end" on this codebase. Any future v29 c-sweep must treat
  small-c arms as GATED arms, not safe-by-default; a small-c arm
  underperforming the c=0 control is a result, not a fluke.
  (Current plan is a single c=1.0 arm; this binds future sweeps.)
- v29 CHASSIS FINDINGS (reviewer; each verified in-file):
  (1) TWO GAE loops exist (reward GAE ~960; soft-advantage GAE
  ~975 with entropy bonus b_t). MY VERIFICATION ADDS: the soft
  path is DORMANT in the reference config — auto_alpha =
  auto_entropy AND actor_dist=="gaussian", and this chassis runs
  Beta with auto_entropy=False. Both loops read the same rewards
  buffer, so one in-place shaping mutation before GAE is
  consistent across both BY CONSTRUCTION. Eight-site trace
  (2 GAE x {nonterm, term, trunc} + probe x {nonterm, term})
  still required, covering the dormant path.
  (2) FREE INVARIANCE SELF-TEST: running RMS of the one-step TD
  residual (feeds tpo_score_floor, coef 0.5) must match the c=0
  control — WITH MY CORRECTION: invariance holds only once the
  critic represents V - c*Phi, so refresh boundaries produce
  transients; the hard-failure criterion is PERSISTENT
  divergence away from refresh boundaries, not any divergence.
  Bug coupling: an inconsistent site silently retunes the probe
  floor -> corrupts candidate selection.
  (3) share_backbone=True -> the frozen snapshot must include
  the SHARED TRUNK, not just the critic head (trunk moves under
  actor gradients too); impl reports snapshot param count;
  refresh-boundary non-stationarity larger than assumed.
  (4) Phi scalar extraction must be the chassis's EXACT
  logits->scalar call (expectation-then-symexp vs symexp-then-
  expectation are different functions; only one keeps Phi in V's
  units). Report item.
  (5) Bit-identity precedent in-file: tpo_coef=0 => exact base
  behavior — impl follows that discipline rather than inventing
  one.
- v29 FINAL BRIEF PRECISIONS (reviewer; crossing resolved — the
  eight-site amendment had already gone out in the same queued
  batch as the consolidated redirect, so all five items deliver
  together when the impl agent wakes):
  (1) F-SCALE PRECISION: a FIXED scale k is harmless (Phi/k is
  still a potential); ONLY time-varying (running/EMA) scale
  breaks telescoping. The ban is on running normalization
  specifically, not scaling per se; moot in practice since the
  snapshot Phi is already in return units.
  (2) NATIVE EXPL-VAR IS UNINTERPRETABLE BY DESIGN: with c=1 and
  Phi ~ V_snapshot, V' = V - Phi ~ 0 by construction -> critic
  outputs near zero, native explained variance looks terrible.
  THAT IS THE DESIGN WORKING (advantage degenerates gracefully
  into GAE on the snapshot's TD residuals). This is the concrete
  reason the reconstructed-V gate is MANDATORY, not
  nice-to-have. Corollary registered into the kill-gate watch:
  near-zero targets sit in the finest symlog region far from
  support edges -> edge_mass should be QUIET in the shaped arm;
  elevated edge_mass = real signal, not noise.
- v29 KILL GATE AMENDED (pre-launch, still before any data):
  (1) SECONDARY GATE ADDED — REFRESH-TRANSIENT DECAY (reviewer;
  adopted with asymmetric reading): the magnitude of the TD-RMS/
  value-loss transient at each phi refresh (every 16 iters, many
  measurements per run, internal — no control needed).
  REGISTERED ASYMMETRICALLY: flat-or-growing transients late in
  training = HARD NEGATIVE (critic never adapts to the shifted
  target; design adds non-stationarity without buying
  simplicity). Decaying transients = necessary-but-WEAK support
  only — decay is confounded with global convergence (value
  functions settle in the control arm too), so decay alone never
  counts as mechanism confirmation; reconstructed expl-var
  remains primary.
  (2) ENABLE TRANSIENT EXPECTED (verified algebra): at enable
  the critic still represents unshifted V, so delta' = delta +
  c(gamma*Phi' - Phi) ~ 2*delta - r with c=1, Phi ~ V — a spike
  the SAME ORDER as the residual. Not a failure signal. Refresh
  transients much smaller (16 iters of critic drift only).
  (3) PROBE SITE EXPLICITLY SHAPED (reviewer catch): the
  by-construction buffer-mutation guarantee covers both GAE
  loops but NOT the probe — probe rewards come fresh from
  physics (cached e.unwrapped, frozen norm stats). Required:
  r'_cand = r_cand + c(gamma*(1-term_cand)*Phi(s'_cand) -
  Phi(s)); the candidate-common -c*Phi(s) may be omitted
  (cancels in anchored softmax), the candidate-varying
  +c*gamma*(1-term)*Phi(s'_cand) MUST be present or candidate
  ordering silently corrupts. The (1-term_cand) mask IS trap 1
  at the probe site — invisible on HalfCheetah (no termination),
  surfaces only cross-env.
- v29 BRIEF, TWO CLOSING ITEMS (reviewer; crossing again resolved
  in our favor — the probe-site item was already in the queued
  batch): (1) TENSOR REUSE CONSTRAINT: Phi(s'_cand) computed from
  the IDENTICAL preprocessed candidate-obs tensor the probe
  builds for V(s'_cand), never re-derived — the obs path is
  TWO-stage (frozen NormalizeObservation + TransformObservation
  clip [-10,10] ~line 329) and re-derivers forget the clip,
  feeding the frozen trunk OOD inputs on exactly the tail states
  where candidate ordering matters most. Construction-over-
  discipline, same principle as the buffer mutation. (2) COST
  FLAG: Phi doubles the probe's critic-side forwards (8
  candidates/state, full-trunk pass under share_backbone). Per
  no-smoke-test rule, NO short throughput run — the real 8M
  submission's SPS gets read against the control's SPS in the
  first minutes as a scheduling fact only; impl reports an
  expected-overhead estimate.
- v29 REFRESH STATISTICS SPLIT (reviewer conceded amplitude gate;
  proposed recovery time; adopted): the refresh yields TWO
  measurables with OPPOSITE confound structure.
  AMPLITUDE (= c*(Phi_new - Phi_old)) — confound and signal push
  the SAME way (shrinks under plain convergence); stays
  registered asymmetrically (flat-or-growing late = hard
  negative; decay = weak support only).
  RECOVERY TIME tau (iterations for TD-RMS to return to
  pre-refresh level) — confound INVERTED: anneal_lr=True
  (line 209, verified) decays the LR across training, so plain
  convergence with no mechanism predicts tau GROWS; the
  mechanism (residual gets easier as snapshot approaches truth)
  predicts tau SHRINKS. tau flat-or-falling late = evidence
  measured AGAINST the annealing gradient. REGISTERED
  REPORT-ONLY (per reviewer's own caveat: batch composition,
  target-KL early stop, actor-pulled trunk also move tau).
  Recoverable post-hoc from already-required tags (per-iteration
  TD-RMS + refresh markers); no new implementation.
- v29 EARLY STRUCTURAL CHECK REGISTERED (pre-launch): if tau >
  phi_refresh_every (16) — transients overlapping the next
  refresh — the potential is not piecewise-constant but
  continuously non-stationary, and the invariance the design
  rests on holds nowhere except in the limit. DETECTION: first
  ~500 iterations of the REAL 8M run (NO separate short run —
  no-smoke-test rule; same pattern as the SPS read). ACTION:
  sustained overlap = structural no-go -> cancel, raise
  phi_refresh_every to 32/64, resubmit. This is a design-
  validity check, not a performance read — cancelling on it is
  not an evidence peek. Registered remedy knob identified in
  advance: phi_refresh_every.
- v29 KILL GATE: THIRD CURVE ADDED (reviewer; pre-launch): the
  gate plots THREE expl-var curves vs UNSHAPED returns — control
  critic V, reconstructed V_hat = V' + c*Phi, and Phi ALONE.
  Two curves cannot distinguish "mechanism works" from "critic
  fit noise but the snapshot was already fine" (both show V_hat
  tracking control). Third curve separates on sight:
  V_hat > Phi and > control = critic adds value on top of the
  snapshot (mechanism present); V_hat ~ Phi = critic contributed
  nothing, residual was noise = the REGISTERED DEATH with its
  cause identified on the spot. NAMED MECHANISM for that death
  (SNR argument): shrinking the target V -> V - c*Phi shrinks
  signal toward zero while the TD-target sampling-noise floor
  stays put -> fitting SNR WORSENS; and the residual is a
  difference of strongly correlated functions — shared smooth
  structure cancels, the rough recently-updated part remains.
  SMALL DOES NOT IMPLY EASY; this is the concrete way the
  likeliest death fires. Cost: one scalar
  (shaping/expl_var_phi), Phi already computed; recoverable
  post-hoc if the tag misses the rebuild (sent nice-to-have —
  impl woke mid-exchange, batch delivered, rebuild underway).
  Reviewer also self-corrected their short-run tau phrasing to
  the real-run read — already how it was registered; no change.
- v29 THIRD CURVE: POST-HOC CLAIM RETRACTED, TAG UPGRADED TO
  REQUIRED (reviewer self-correction; I had propagated the same
  error into the nice-to-have flag): expl-var needs Phi(s) and
  the unshaped return per rollout state AT that iteration;
  neither snapshot weights nor rollout states are persisted —
  the event file holds scalars only. The curve exists ONLY if
  logged in-loop. shaping/expl_var_phi is now REQUIRED (one
  scalar/iteration from already-materialized tensors, computed
  identically to the other two expl-var metrics). Checkpoint-
  weights fallback rejected as unnecessary — the in-loop line
  is trivial. Rationale unchanged: without the third curve the
  registered death fires without its cause, which is the exact
  follow-up-investigation pattern this program exists to avoid.
- v29 RED-TEAM PASS COMPLETE (on the OLD-SPEC file; verdict
  APPROVED-with-defects, superseded by the rebuild but findings
  triaged): TRANSFERS: (1) bit-identity micro-harness (2 env x
  64 steps x 3 iters, 8-decimal scalar match, both tpo_coef
  modes) — now the REQUIRED form of bit-identity evidence;
  (2) guard-shape audit (isfinite skip-assignment leaving None
  on a later unconditional logging read); (3) DEFECT B
  TRANSFERRED AS A REGISTERED WATCH: the probe correction
  +c*gamma*(1-term)*Phi(s'_cand) is exact ONLY in the absorbed
  limit; while the critic lags, TPO group score spread inflates
  (measured +61% tpo_score_std_mean, tpo_floored_frac 0.289 ->
  0.094 in their micro-run) and TPO degenerates toward "rank by
  Phi(s')". New spec is structurally milder (shaping on from
  iteration 1 -> no mid-training enable shock; frozen Phi
  between refreshes -> gap is only Phi_new - Phi_old), but
  shaped-vs-control {tpo_score_std_mean, tpo_sigma_floor,
  tpo_floored_frac} is the registered watch instrument for it.
  MOOT: per-batch-std suggestion, defect A's site, D3 target
  de-shaping (no Phi training in new spec). REJECTED: their
  "cleaner alternative" (shape policy_adv only, critic raw) —
  that is the PRE-THEOREM design: F stays in the PG permanently
  = advantage shaping, forfeits the exact GAE invariance that IS
  the registered hypothesis. Would need its own registration as
  a different experiment.
- v19-REVIEW PASS ON THE WRONG FILE STATE (triaged): reviewer
  did a full trap-list pass on the on-disk v29 file believing it
  the rebuild; verified timeline shows it is the ORIGINAL-brief
  architecture + red-team D-fixes — the impl has NOT yet
  processed the redirect batch (its last turn predates
  delivery). TRIAGE: (1) their BLOCKING finding (all three gate
  curves absent; the stock expl-var curve is ARM-INCOMPARABLE —
  both sides in shaped units AND the denominator shrinks
  mechanically since Psi ~ Phi is positively correlated with
  R_true, so real improvement can read as critic regression =
  the embopt shrinking-denominator artifact recurring) TRANSFERS
  and is already structurally required in the redirect.
  ADJUDICATION: stock expl-var tag stays byte-identical to the
  chassis; its arm-incomparability lives in registration +
  report header; the comparable objects are the reconstructed
  and phi-alone curves vs unshaped returns. (2) EMA question
  SETTLED: phi_f_std = banned category AND time-varying-c
  telescoping breach (their within-iteration self-consistency
  verdict doesn't cover cross-iteration rescale); phi_tgt EMAs
  die with Phi training; chassis's native td_rms_ema untouched.
  (3) PASSES THAT TRANSFER (sites persisting in rebuild): probe
  correction sign + z-score cancellation, terminations-only
  mask, single gamma, pre-mutation reward_frac,
  RecordEpisodeStatistics unshaped-by-construction, frozen-stat
  consistency (Phi treated exactly like the critic it corrects).
  (4) CREDIT: fork_rng(devices=[device]) CUDA-generator subtlety
  (torch.manual_seed reseeds CUDA too) — correct and easy to get
  wrong; dies with PotentialNet but recorded.
- v29 ABSORPTION-BETA ANALYSIS (reviewer; the most load-bearing
  addition of the day — adopted at full strength): write beta =
  fraction of Psi the critic has absorbed (V_critic = V_true -
  beta*Psi). BOTH approximate-correction defects are exactly
  proportional to (1-beta): probe leaves candidate-varying
  (1-beta)*c*gamma*Phi(s'_cand) (=> TPO partly ranks by Phi);
  GAE leaves (1-beta)*F in the advantage (= UNLABELED ADVANTAGE
  SHAPING — the rejected design). Persistent low beta silently
  CONVERTS the run into the rejected experiment; the cancel
  criterion is the BOUNDARY BETWEEN THE TWO EXPERIMENTS, not
  just a validity check. Both sites fail in the SAME direction
  (toward high-Phi = high-value) so the compound failure MIMICS
  SUCCESS. CRITICAL COINCIDENCE (previously unstated anywhere):
  (1-beta) is largest EARLY — exactly the window where the
  registered hypothesis predicts its signal — so early gains
  are predicted by BOTH designs and the primary read cannot
  distinguish them without measuring beta.
- v29 GATE AMENDED ACCORDINGLY (pre-launch, no data seen):
  (1) NEW REQUIRED TAG shaping/absorption_beta =
  -Cov(b_values - R_unshaped, Psi)/Var(Psi) per iteration
  (estimator clean: frozen-snapshot Phi cannot correlate with
  current-rollout return noise). (2) PRIMARY READ IS NOW JOINT:
  early reconstructed-expl-var gains count as mechanism
  evidence ONLY with beta near 1 in the same window; early
  gains + low beta = rejected-design evidence, NOT success.
  (3) VALIDITY GATE: beta persistently far below 1 => the run
  is the other experiment; results uninterpretable under this
  registration; STOP and re-register before reading anything.
  (4) Reconstructed curve now does triple duty (mechanism gate,
  arm-comparable critic gate, coarse beta readout: at beta=0
  V_hat = V_true + Psi scores WORSE than control). (5) Reviewer
  crossing note: their "refresh not in file" check ran on the
  pre-redirect artifact (crossing with my version-state
  correction); phi_refresh_every=16 is in the redirect spec.
  Their deeper point adopted regardless: beta is MEASURED, not
  inferred from structure. Defect-A None-crash confirmed live
  in the old file (dies with the EMA machinery; guard-shape
  audit item covers the pattern in the rebuild).
- v29 LEDGER CORRECTION (red-team round 2 self-correction): the
  "+61% tpo_score_std_mean = probe-correction effect" evidence I
  logged is MISATTRIBUTED. The correction contributes ~0.18 of
  group spread (new shaping/tpo_corr_ratio instrument); the
  score-std inflation was TRAJECTORY DIVERGENCE (shaped run on a
  different policy/critic since iter 1 — different states,
  differently-trained critic). tpo_score_std_mean is NOT
  cross-run comparable; the registered watch instrument for the
  absorbed-limit defect is shaping/tpo_corr_ratio (plus
  absorption_beta). Defect B remains real as a RISK (denominator
  shrink) — now capped at 1.0*group_std and instrumented. Also:
  red-team WITHDREW its policy-adv-only alternative after impl
  pushback (the +gamma*Phi(s') term is action-dependent at the
  sampled successor = permanent PG bias; same argument I used).
- v29 RECURSION CATCH (reviewer; verified) + CHAIN ANALYSIS
  (mine) + ARCHITECTURE ADJUDICATION: raw-snapshotting a critic
  that fits SHAPED returns gives Phi_new = V_true - c*Phi_old:
  at c=1 period-2, never converges; refresh amplitude
  |V_true - 2*Phi_old| non-decaying (mimics "still learning");
  beta pinned low permanently (advantage-shaping confound never
  clears, success-shaped). Reviewer's fix (de-shaped snapshot
  Phi_new := V_critic + Psi_old = V_true) is EXACT but has an
  UNBOUNDED CHAIN they missed: Psi_old retention is recursive —
  each Phi references its predecessor; at 244 iters/refresh-16
  that is ~15 stacked frozen modules by 8M (memory + 15x Phi
  eval cost incl. the probe). Infeasible.
  ADJUDICATION — FINAL v29 ARCHITECTURE (supersedes the
  frozen-critic-snapshot redirect): KEEP the built independent-
  Phi architecture (its de-shaped training target = raw-return
  regression is EXACTLY the anti-recursion protection; twice
  red-teamed; bit-identity proven), with amendments:
  (1) USED-Phi = frozen COPY of the live Phi net, refreshed
  every phi_refresh_every=16 iters (cheap: small MLP copy, not
  a trunk) -> piecewise-constant potential, exact invariance
  within intervals, timescale separation for beta, small
  refresh jumps (16 iters of Phi drift). Live Phi trains every
  iteration on de-shaped targets (de-shaping uses the FROZEN
  used-copy's Psi — no recursion: target is a raw-return
  estimate independent of the potential).
  (2) DELETE both EMAs: phi_f_denom gone (F in raw return
  units, fixed c — telescoping + user ban); phi_tgt running
  standardization REPLACED by FIXED symlog parametrization
  (Phi = symexp(net(s)), matching the chassis's own
  value_symlog treatment) — fixed transforms are valid
  potentials, running stats are not.
  (3) shaping_coef default 1.0.
  (4) The four gate tags (3 curves + absorption_beta), refresh
  markers, and NEW shaping/phi_drift (mean |Phi_live - Phi_used|
  over batch at refresh) all required. All prior registrations
  (tau overlap, amplitude asymmetric, joint beta read, three-
  curve gate) apply unchanged.
  Beta-estimator cleanliness preserved: used-Phi is frozen >= 1
  iteration stale, cannot correlate with current-rollout noise.
- v29 FINAL PRE-BUILD ROUND (reviewer; all three items land on
  the adjudicated architecture):
  (1) GRADIENT BUDGET: refresh-16 = 5,120 critic minibatch
  updates per interval (320/iter x 16) against a constant
  potential — reviewer WITHDRAWS the no-interval-to-converge
  worst case for the rebuild. Prior updated: beta should
  recover well within each interval; persistently low beta is
  now SURPRISING-AND-DIAGNOSTIC, not expected.
  (2) FIRST-INTERVAL BLANK (adopted, schedule fix sent): the
  used-Phi at iteration 0 is an initialized net ~ meaningless,
  so a plain 16-interval runs the first ~524k steps effectively
  unshaped — the registered neutral-early death would score an
  INERT window. FIX: first refresh at iteration 2, then every
  16 (refreshes at 2, 16, 32, ...). Any Phi is a valid
  potential; scheduling change only.
  (3) SAME-KNOB ANALYSIS (registered): at c=1.0 with Phi ~
  V_true at refresh, the critic's target = V_true(now) -
  V_true(last refresh) ~ 0 over 16 iters while sampling noise
  is untouched — c=1.0 is the EXTREME of the registered SNR
  death; the mechanism knob (shrink the residual) and the death
  knob (destroy target SNR) are the SAME knob. REGISTERED
  PREDICTION: stock expl-var tag sits NEAR ZERO in the shaped
  arm BY CONSTRUCTION — not a failure signal; report header
  must preempt the dashboard-glance misread.
  (4) COEFFICIENT TENSION + DECISION: c=1.0 maximizes both the
  death and the quality of the beta instrument that detects it
  (estimator signal ~ beta*c vs fixed noise floor); c=0.5
  separates mechanism from death on the otherwise-shared knob.
  DECISION: TWO ARMS — c=1.0 primary (registered gates) and
  c=0.5 gated companion (explicitly gated per the 753
  non-monotonicity rule: sub-c=0 performance at any c is a
  result, not a fluke). Both HalfCheetah 8M seed 1, CLI-matched
  to the reference; control = existing 10,371 run. Submission
  concurrency decided at submit time per mlq guidance.
- v29 THREE HOLES IN THE ADJUDICATION (reviewer; chain defect
  conceded on their side — "composition grows depth, distill-
  into-weights is the same fix implemented so it bounds"):
  HOLE 1 — LIVE/USED AMBIGUITY, five sites, rule NOT uniform:
  rewards shaping/de-shaped target/probe correction/beta Psi =
  USED copy; only the optimized forward pass = LIVE. Highest
  risk: the de-shaped target site (old file computes it from
  the only net that existed) — a mechanical carry-forward
  de-shapes by the LIVE potential, error = c*(Phi_live -
  Phi_used) = PHI_DRIFT READING ITS OWN CAUSE. Explicit
  USED/LIVE column required in the eight-site trace.
  HOLE 2 — METRIC CHANGE, REGISTERED: at c=1.0 with Phi=V_true,
  r_shaped = delta EXACTLY (F = delta - r) — the shaped reward
  IS the TD residual; the critic regresses returns-of-residuals
  (advantage-scale). Expl-var is a RATIO and normalizes away
  the absolute reduction that is the entire claim. NEW PRIMARY:
  shaping/critic_residual_rms = RMS(b_returns - b_values),
  absolute, shift-cancels, arm-comparable. Control curve
  DERIVED from existing run: returns_std * sqrt(1 - expl_var)
  (both tags logged by chassis; caveat: assumes ~zero-mean
  residual). Mechanism present iff shaped RMS < control RMS.
  Three expl-var curves DEMOTED to arm-comparability + death-
  cause identification (they keep the beta/noise-fit
  differential role). SUCCESS SIGNATURES registered: reward_frac
  -> ~1.0 and stock expl-var -> ~0 at c=1.0 (both would read as
  catastrophe; header preempts).
  HOLE 3 — BETA IS A STABILITY PARAMETER: de-shaped target =
  V_true + (1-beta)*Psi_used, fixed point Phi* = V_true/beta at
  c=1 — imperfect absorption INFLATES the potential (1.25x at
  beta 0.8, 2x at 0.5), which hardens the absorption task:
  bounded but SELF-REINFORCING. HARD PRE-REGISTERED THRESHOLD:
  median absorption_beta < 0.7 sustained over any 1M-step
  window after the first two refresh intervals = VALIDITY
  FAILURE -> stop, re-register (rationale: at 0.7 the Phi
  inflation is 1.43x and the PG contamination is 30% of F;
  drift-with-positive-feedback needs a line, not a judgment
  call). FREE CROSS-CHECK: falling beta must co-move with
  rising Phi scale (~1/beta); beta falling with flat phi_std =
  ONE INSTRUMENT IS LYING.
  SYMLOG ROUGHNESS CHECK (derived, no new instrument): expected
  std(F)/std(Phi) ~ 0.01-0.1 for a smooth potential; order 1 =
  Phi rough at transition scale, F mostly noise (symexp
  amplifies net-space error most where values are largest).
  REASSURANCE ON RECORD: delta' = delta term-by-term regardless
  of magnitudes — every accumulated risk lives on the critic
  side, where all gate tags point; the policy gradient is
  untouched.
- v29 HOLE-2 SELF-CORRECTION ACCEPTED (reviewer): critic_
  residual_rms is REDUNDANT with the reconstructed curve —
  Var(R_true - V_hat) = Var(R_shaped - V_shaped) with a COMMON
  denominator Var(R_true), so reconstructed expl-var and
  residual RMS rank arms identically. PRIMARY reverts to
  shaping/expl_var_reconstructed as originally registered; the
  residual_rms tag is KEPT as its unnormalized twin (one line,
  plots without derivation; report language corrected at
  review). Four gate tags, not five. The surviving half:
  R' = R - Psi(s) is a shift by a deterministic function of the
  start state, so Var(R'|s) = Var(R|s) EXACTLY — sampling noise
  is a common additive floor both arms carry; the mechanism can
  only move the approximation-error term. Power ceiling applies
  to ALL metrics equally.
- v29 PRE-LAUNCH POWER CHECK (run on the existing control, no
  GPU spent): control EV @0.5M/1M/2M/4M/8M = 0.878/0.918/
  0.818/0.938/0.953; headroom (1-EV) in the 0.5-2M signal
  window = 0.08-0.18 of return variance; EV window scatter sd
  0.075 early (autocorrelated; effective window-mean SE
  ~0.02-0.03), 0.016 late. DETECTION FLOOR at single seed:
  ~+0.04 EV in a matched 1M window. VERDICT: gate has teeth —
  +0.04 = removing ~1/3 to 1/2 of early headroom, the size the
  hypothesis predicts if real. Reviewer's swap-seeds-for-
  companion condition does NOT fire; TWO-ARM plan stands
  (c=1.0 + gated c=0.5).
  QUANTIFIED PRIMARY GATE (pre-registered): MECHANISM-PRESENT
  iff shaped reconstructed EV minus control EV >= +0.04
  averaged over the matched 0.5-2M window (equivalently ~18%
  residual-RMS reduction at EV 0.88). Below +0.04 =
  indistinguishable from zero at n=1 = the registered
  neutral-early death fires. ACKNOWLEDGED CAVEAT: an unknown
  fraction of the 8-18% headroom is irreducible sampling noise;
  if noise dominates, even a perfect mechanism shows <+0.04 —
  that is the SNR death expressing at measurement level and it
  is ACCEPTED as a death mode (indistinguishable by design at
  n=1). Note: control has a mid-run EV dip to 0.82 @2M with
  returns_std nonmonotonic (43->23->59) — matched-step
  comparison is mandatory, already registered.
- v29 REVERSAL ACCEPTED — CO-PRIMARY JOINT GATE (reviewer
  reversed their own withdrawal; reason auditable): the
  redundancy argument fails because each arm computes
  Var(R_true) over ITS OWN visited distribution (different
  policies, different states) — same OBJECT, different NUMBER.
  Same trajectory-divergence failure class the red-team caught
  on tpo_score_std_mean. The two metrics carry OPPOSITE
  confounds: absolute residual RMS rides the return scale (a
  faster-learning arm shows LARGER residuals while fitting
  relatively better); reconstructed EV normalizes by the arm's
  own performance-dependent denominator. FIVE gate tags.
  REGISTERED JOINT PATTERN TABLE (matched 0.5-2M window,
  control RMS derived = returns_std*sqrt(1-EV)):
    EV >= control+0.04 AND RMS <= control  -> MECHANISM
      PRESENT, robust to both confounds (only clean positive).
    RMS smaller AND EV lower  -> smaller problem, not better
      fit: NOT mechanism.
    RMS larger AND EV higher  -> performance confound live:
      plausibly present, follow-up required, not clean.
    Neither favors shaped  -> neutral-early DEATH.
  The +0.04 floor applies to the EV side (power check); the RMS
  side has no clean floor (scale confound) and serves as the
  direction check in the joint read. Noise-floor bound
  unchanged (survives their correction): Var(R'|s) = Var(R|s)
  exactly; the floor dilutes BOTH metrics equally.
  CROSSING NOTE: their re-flag of the power check predates my
  run of it — verdict unchanged, two arms stand (headroom
  8-18% early vs +0.04 floor; swap-seeds condition does not
  fire). Impl unaffected: critic_residual_rms was already
  required; only the registration language moves (co-primary,
  not demoted twin).
- v29 GATE FINALIZED — RATIONALE CORRECTED + THRESHOLD BANDED
  (reviewer; supersedes both the "redundant" basis and the
  "co-primary" label from the crossed messages):
  RATIONALE, STATED ACCURATELY: reconstructed EV is THE gating
  metric because it is (a) SCALE-ROBUST — returns_std swings
  2.5x (43->23->59) WITHIN the control alone, so absolute
  residual RMS would swing with it for reasons unrelated to
  fit — and (b) EMPIRICALLY CALIBRATED from the control's own
  scatter, which already absorbs denominator volatility.
  critic_residual_rms INFORMS BUT CANNOT GATE (no comparable
  calibration, rides the return scale); its role is the
  direction check inside the joint pattern table (which stands:
  RMS/EV agreement-and-disagreement patterns localize the
  performance confound). NOT redundant — do not drop the tag;
  NOT a gate — do not threshold it.
  SE CORRECTION (my error, caught by reviewer): the +0.04 floor
  was calibrated as ~2 SE of ONE arm's window mean; the gate
  compares TWO arms, so SE_diff = 0.025*sqrt(2) ~ 0.035 and
  +0.04 is ~1.1 SE_diff => ~13% one-sided false-positive under
  the null. A clean 2-SE line is ~+0.07, which clears only the
  TOP of the predicted effect range (0.027-0.09) — the design
  is underpowered for the lower half of its own prediction =
  the SNR death at measurement level, with its consequence now
  explicit.
  REGISTERED BANDED GATE (matched 0.5-2M window, shaped
  reconstructed EV minus control EV):
    < +0.04            -> NEUTRAL-EARLY DEATH (as registered)
    +0.04 to +0.07     -> SUGGESTIVE (~1-2 SE_diff): NOT
      mechanism-present; pre-committed CONDITIONAL SECOND SEED
      of c=1.0 required before any conclusion
    > +0.07            -> MECHANISM-PRESENT at n=1 (~2 SE_diff)
  The false-positive rate lives next to the threshold by
  design. Launch plan unchanged: two arms (c=1.0 + gated
  c=0.5); the second seed is a conditional spend, fired only by
  the suggestive band. Matched-step comparison remains
  load-bearing (unmatched = different denominators, not noise).
  REGISTRATION NOW COMPLETE FROM BOTH PARTIES.
- v29 PHASE-SHIFT THREAT + SIGN-CONSISTENCY AMENDMENT (reviewer;
  adopted with a power-quantified implementation): the control's
  EV has 0.10 of structure INSIDE the 0.5-2M signal window
  (0.878 -> 0.918 -> 0.818), 2.5x the +0.04 floor, and the dip
  tracks a policy-improvement phase (returns_std -> 23 at the
  same point). Arms learning at different rates traverse that
  curve OUT OF PHASE, so a window-mean difference can clear
  +0.04 from phase alignment alone — and faster early learning
  is BOTH the hypothesis's prediction AND the phase-shift
  generator. THIRD instance today of the confound-aligned-with-
  hypothesis family. CHECKLIST ITEM EXTENDED (temporal twin):
  cross-arm comparisons must ask not only "own visited
  distribution?" but "own PHASE of learning?" — matched steps
  fix the x-axis, not the phase.
  REGISTERED IMPLEMENTATION (power-checked): raw per-checkpoint
  sign-majority would FAIL a true effect — per-point SE_diff ~
  0.075*sqrt(2) ~ 0.106, so a true +0.04 shows positive at only
  ~65% of checkpoints. Instead: split 0.5-2M into THREE 0.5M
  sub-windows (~15 checkpoints each, sub-window SE_diff ~
  0.027). EV GATE IS NOW: (a) mean matched-step difference >=
  +0.04 over the full window AND (b) difference POSITIVE IN ALL
  THREE sub-windows. Power ~0.80 for a true +0.04, higher above;
  a phase artifact (curves sliding past each other) alternates
  sub-window signs and fails (b). Large-mean-but-sign-flipping =
  read as PHASE ARTIFACT, not weak positive. Banding unchanged
  on the mean (0.04/0.07); (b) applies to all bands.
  COMPANION UPGRADED: the c-ordering signature c=1.0 > c=0.5 >
  control at matched sub-windows is a mechanism signature a
  phase artifact has no reason to produce (real mechanism
  monotone in c). This is now the c=0.5 arm's PRIMARY job,
  stronger than mechanism-vs-death separation, free with the
  existing allocation.
- v29 BUILD MID-WRITE STATUS + REVIEW PROTOCOL HAZARD (reviewer
  status note; no defect, no intervention): observed partial
  state = header FULLY rewritten to final spec (USED/LIVE
  five-row table, expected-signature preemptions, joint gate)
  and setup done (phi_used deepcopy + freeze, phi_refresh_every)
  while EVERY use site still references the live net, both EMAs
  still live, and NO gate tags exist yet. Normal mid-write
  order (header-first), but it creates a REVIEW HAZARD: the
  header asserts, in confident specific detail, exactly the
  properties the trap list checks — currently all FALSE in the
  body. PROTOCOL LOCKED FOR BOTH PASSES: the header is a CLAIM
  UNDER TEST, never evidence; every trap-list item verified at
  its code site; the five USED/LIVE rows are checked by the
  IDENTIFIER at the line (phi_used vs phi_net) and nothing
  else; and comments demonstrably updated in a different pass
  than the code get an extra look. General lesson worth
  keeping: documentation quality can actively work AGAINST
  review quality when docs are written from the spec rather
  than from a verification.
- v29 SIGN-CONSISTENCY AMENDED — 2-OF-3, WITH CURVE-SHAPE
  CONDITIONALITY (reviewer's third self-correction + my
  autocorrelation inconsistency, both verified):
  MY ERROR: sub-window SE used raw n=15 after I had already
  applied autocorrelation deflation at full-window level
  (effective n ~9 per 1M). Carried down: effective n ~4.5 per
  0.5M sub-window, SE_diff ~0.106/sqrt(4.5) ~0.050 — so
  ALL-THREE-positive has power 0.79^3 ~0.49 at a true +0.04,
  not the 0.80 I registered. An underpowered gate believed
  0.80-powered fires neutral-early on a REAL mechanism — the
  expensive direction.
  REVIEWER'S ERROR (unwound): the phase threat was overstated —
  a pure time shift integrates to (delta/(b-a))*[EV(b)-EV(a)],
  ENDPOINTS ONLY, internal swing cancels. On this curve
  endpoints are 0.878 -> 0.818 (net -0.06), so a shaped-ahead
  phase shift yields a SMALL NEGATIVE mean difference —
  condition (a) at +0.04 already excludes it decisively. The
  sub-window machinery was paying a power cost against a threat
  the mean already handles ON THIS CURVE SHAPE.
  REGISTERED FINAL FORM: (a) mean matched-step difference >=
  +0.04 over 0.5-2M (banding 0.04/0.07 unchanged); (b) AT LEAST
  2 OF 3 sub-windows positive (power ~0.88 at true +0.04) —
  retained to catch SHAPE differences (not clean time shifts),
  where endpoint cancellation does not apply.
  CONDITIONALITY ON RECORD: (a)'s phase protection depends on
  the window's EV FALLING end-to-end. On any future window
  where EV RISES end-to-end, the sign flips, (a) stops
  protecting, and all-three-positive becomes the right setting
  again. Curve-shape conditional, not universal.
  Unaffected: banding, c=0.5 monotonicity signature (depends on
  neither number).
- v29 MISSING-REFRESH HAZARD + REFRESH PLACEMENT (reviewer;
  forwarded to impl as final build directives):
  (1) THE FAILURE THAT PASSES EVERY TRAP-LIST ROW: deepcopy at
  init + refresh logic missing => every USED/LIVE row reads
  phi_used, EMAs gone, all tags present — clean sweep — while
  the run shapes 8M steps with a potential frozen at RANDOM
  INIT. Valid potential, nothing crashes, invariance holds,
  produces a well-behaved NULL that reads as the registered
  neutral-early death: we would record the mechanism dead
  having never tested it. DETECTOR IS BEHAVIORAL: shaping/
  phi_drift must be a SPIKE TRAIN (non-zero at refresh iters
  2/16/32/..., exactly zero between); FLAT-ZERO = this failure.
  Added to both review passes + impl report requirement.
  Grep footnote: the refresh site's own phi_net read (deepcopy/
  load_state_dict) is the ONE legitimate out-of-training-block
  reference — expected false positive, not an intervention
  trigger.
  (2) REFRESH PLACEMENT: END of iteration, after Phi training
  and logging. The four USED sites refer to two moments (reward
  write: copy about to be absorbed; de-shape/probe/beta: copy
  the pre-update critic absorbed last iteration); they coincide
  except at refresh, so mid-iteration refresh makes one
  variable two objects. End-of-iteration keeps every single
  iteration self-consistent (what the eight-site trace checks).
  (3) REGISTERED EXPECTED SIGNATURE: absorption_beta DIPS at
  the iteration after each refresh (3, 17, 33, ...) then
  recovers — three critic-side sites briefly one refresh ahead
  of what the critic absorbed. NOT a bug. The 0.7 validity
  threshold survives BECAUSE it is a median over ~30 iterations
  containing only ~2 refresh-adjacent dips — the median form
  doing work it was not chosen for.
