# Pan Goal Solver

Primary reference: [Pan-1: A General Goal-Conditioned Minecraft Model](https://pantograph.com/journal/pan-1).
See [PAN1.md](./PAN1.md) for a structured account of the article and the
boundary between its disclosed method and this family’s adaptation.
See [LEWM.md](./LEWM.md) for notes on the slower latent-planning analogue and
[GOAL_PREDICTOR.md](./GOAL_PREDICTOR.md) for the open-ended proposer design.

## Objective

Learn continuous control from online experience with a Pan-like direct goal
solver and a learned open-ended goal. On every environment step, a detached
head maps the current world-model belief to one deterministic goal latent. The
solver immediately maps `(current belief, goal) -> action`; there is no search.
On every optimization cycle, a separate reward head learns reward MSE from the
detached predicted-next latent, then supplies a frozen gradient that improves
the goal head's predicted goal. As the world representation and reward belief
improve, the point can acquire a clearer and more specific meaning.

## Hard constraints

- No pretraining data.
- Environment reward is used only as the MSE target for the reward head. Reward,
  predicted reward, and their gradients must not enter the world model or
  follower. The detached predicted point `G` is the only intended path from the
  reward-trained subsystem to the direct actioner.
- No action-value critic, advantage, PPO, actor-critic update, or reward policy
  gradient. A latent reward predictor `R(z)` and goal-conditioned successor
  model are permitted because they evaluate/select states, not actions.
- No search, planning, MPC, action optimization, imagined action evaluation, or
  rollout at inference or follower training. The one-step world model learns
  only from real replay transitions.
- Inference is a direct policy call: `(current belief, goal) -> action`.
- Goals contain observation-history latents only. Never include reward, value,
  continuation, action, or privileged simulator state in a goal.
- Reward prediction is invariant to positive rescaling of a goal latent. Goal
  distance is inconsequential; the open objective must improve latent content
  rather than exploit reward-head sensitivity to vector norm.

## Learning rule

- Collect experience with the current direct policy.
- For follower training, relabel achieved future observation latents as
  hindsight goals without using reward or ranking trajectories by utility.
- Train a goal-conditioned desired-next latent, inverse action, occupancy, and
  composed first action only from those real hindsight tuples. The successor
  predicts the actual next latent that followed the hindsight-conditioned
  state; the follower never evaluates an unexecuted action through the world
  model.
- On every goal cycle, predict the action-conditioned next latent with the world
  model, detach it, decode it with the frozen reconstruction decoder, and fit
  `R(decoder(z_hat_next))` to real scalar reward with MSE. Then freeze `R` and
  the decoder, compute `G = P(stopgrad(b))`, and update only `P` to maximize
  `R(decoder(G))`. Neither head may update the world model through this path.
- The direct follower factors through a Pan-like desired successor and inverse
  action: `z_next* = S(b,G)`, then `a = I(b,z_next*)`. Both are amortized network
  heads trained on real hindsight transitions and execute inside one fused
  action call. First-action likelihood and successor occupancy remain reward-free.
- A v215-like LeJEPA world model may learn observation representation,
  action-free temporal state, and a one-step action-conditioned successor. It
  must not predict reward or generate multi-step training rollouts.
- An aspirational goal may be unreachable. It is the single history-latent point
  that the current reward model judges most rewarding, not a distribution and
  not a target capped at what the follower has achieved.
- The goal proposer is a required learned component, not an external-goal
  assumption. Its input is detached world belief and its only preference is
  predicted reward. Do not add difficulty, novelty, reachability, uncertainty,
  counterfactual, or hand-selected target-reward objectives.
- The predicted ideal may evolve as the world model learns and epistemic
  uncertainty falls. Embodied MSE progress toward a frozen pre-action goal and
  revision of the point estimate itself must be measured separately.
- If the current control belief contains action history, use a documented
  observation-only projection for the goal and MSE space. The representation
  must canonicalize behaviorally equivalent gait phases well enough that one
  deterministic point denotes a motion regime rather than an averaged pose.

## Required checks

- Verify reward replacement, shuffling, or zeroing changes only reward-head and
  goal-head updates; world-model, successor, and action-policy gradients remain
  identical.
- Verify the predicted-next latent and goal-head belief inputs are detached,
  and that the reward head is frozen while its output trains the goal head.
- Verify hindsight goals never cross episode boundaries or contain action slots.
- Verify shuffled goals materially change occupancy predictions and actions.
- Verify action inference performs exactly one policy forward pass.
- Report reward consumed by the goal learner separately from reward-blind
  follower losses and external benchmark evaluation.

All algorithms in this folder must follow this contract. Reward use outside
`R`/`P`, or any PPO, action critic, or planning path, belongs to another family.

## Amendment (v14): staged nonparametric proposer

While follower goal-tracking is being validated, the proposer requirement may
be satisfied nonparametrically instead of by the learned `P` trained through a
frozen `R` (that gradient path invites reward-model exploitation: off-manifold
latents the frozen head scores optimistically). In this stage:

- Reward selects the replay anchor with the best sustained windowed reward
  rate (the goal-utility role, realized as an argmax over real data instead of
  a learned regressor). This is the only place reward enters goal selection.
- Aspiration is a documented observation-space transform of that anchor: all
  velocity coordinates scaled by a fixed `alpha > 1`, encoded by the EMA
  target encoder. In proprioceptive control the aspiration coordinate is
  physical, so obs-space extrapolation is better grounded than free latent
  ascent; the anchor ratchet keeps the commanded goal a fixed margin ahead of
  demonstrated capability (auto-curriculum).
- A detached windowed reward-rate head may be trained purely as a diagnostics
  probe (aspiration-sweep monotonicity); nothing differentiates through it.

Everything else in the contract is unchanged: the world model, successor, and
follower remain reward-free, inference stays one direct policy call, and the
aspiration-sweep and goal-sensitivity checks are mandatory. A learned `P`
(with support constraints) should replace this stage once the follower
demonstrably tracks and extrapolates commanded goals
(`ppo_continuous_action_pan_goal_solver_v14.py`).

v15 (`ppo_continuous_action_pan_goal_solver_v15.py`) is that learned
replacement: an inverse generator `P(y) -> G` fit by supervised MSE on
achieved (sustained-rate, target-latent) pairs and queried at a
quantile-relative aspiration `y*` above support. It satisfies the original
"learned proposer" requirement while keeping the amendment's safeguards:
reward conditions only the proposer's label/query coordinate, no gradient
flows through the diagnostic rate head, and extrapolation is a query of an
on-support-trained map, never an argmax over a learned scorer.

v18 (`ppo_continuous_action_pan_goal_solver_v18.py`) let the occupancy
InfoNCE backpropagate into the temporal belief. Although reward-free and
contract-legal, it was reverted by decision: the world model stays purely
predictive, matching Pan-1's own separation of goal-conditioned learning
from world-model learning. Do not route non-world-model losses into the
world model in this family.

v19 (`ppo_continuous_action_pan_goal_solver_v19.py`) restores the
contract's follower factorization that v16's monolithic head had dropped:
a nonlinear desired-successor S(b, f_t, g, G) conditioned on the goal in
goal-latent space but predicting the next frame summary (the goal latent's
phase-robust statistics are near one-step-invariant and cannot carry the
intermediate), dense goal-free one-step inverse dynamics I(b, f_t,
f_target), and the fused composition a = I(b, f, S(b, f, g, G)) with a
composed action loss (successor detached). Occupancy remains a
follower-side readout on a detached belief.

v16 (`ppo_continuous_action_pan_goal_solver_v16.py`) repairs the three
failures v14 measured at 1.1M steps before the learned proposer is retried:
gait-frequency explorer periods (6-30 steps; the 20-80 band could not excite
locomotion), additive forward-velocity aspiration
(`min((v_fwd + delta)/v_fwd, cap)` for forward anchors, unscaled otherwise;
fixed multiplicative alpha was a no-op on near-zero-velocity anchors), and a
nonlinear follower action head over (belief, bounded goal delta) replacing
the linear direction-Jacobian that showed chance-level frozen-goal tracking.
The goal-sensitivity check is now an explicit null-goal comparison
(`diagnostics/goal_removal_action_mse`) since the MLP head does not vanish
at zero goal delta. Contract obligations are otherwise unchanged.

v20 (`ppo_continuous_action_pan_goal_solver_v20.py`) attached the composed
loss through the successor; v21 (`..._v21.py`) made the successor target a
waypoint frame f_{t+j}, j ~ U[1, min(k, 16)], on the path to the hindsight
goal. Both were falsified at their pre-registered 1.5M checkpoints:
goal_removal_action_mse stayed dead (0.001 / 0.0002) and frozen_policy_mse
flat at ~2.3, in v21's case with the phase-aliasing signature the missing
horizon input predicts (waypoint-inverse MSE 3x the one-step inverse).

Accepted root cause (adversarial review, 2026-07-17): the family has been
trying to learn a goal-conditioned policy from goal-UNCONDITIONED data.
Half the replay is a scripted explorer whose future is screened off by the
action-history-bearing belief; the policy half was only ever commanded one
fixed latent (frozen since ~800k in every run), so I(action; goal | belief)
is ~0 in the data and no follower architecture can extract a conditioning
signal. Pan-1 never faces this: its data is goal-directed human play.
Architecture iteration on this question is now closed.

v22 (`ppo_continuous_action_pan_goal_solver_v22.py`) is the identifiability
fix, stated prospectively:
- Goal-diverse collection: every policy episode is commanded a goal drawn
  from a mixture (replay-achieved regime 50%, unscaled anchor 25%,
  velocity-scaled aspiration 25%); raw goal histories are stored and
  re-encoded online each iteration. This is what closes the GCSL bootstrap:
  any goal-dependence the policy exhibits enters the data, and hindsight
  regression amplifies it.
- Achieved-goal reaching eval (matched vs shuffled command MSE,
  `evaluation/reach_servo_gap`): the follower's unit test. A positive
  sustained gap is REQUIRED before any aspiration/extrapolation claim; a
  bias-token follower shows no gap.
- Successor conditioned on the waypoint horizon j/16; composed loss
  detached through S (S keeps waypoint-prediction semantics — v20/v21
  showed attachment cannot conjure signal the data lacks).
Pre-registered 1.5M criteria: primary — reach_servo_gap sustained > 0 and
goal_removal_action_mse lifting toward 0.05+; supporting — pursuit_mse_achieved
trending down. Failure of the primary criterion falsifies the data-side
diagnosis, not just the version.
Known debts deferred to later versions, deliberately: time-resampled
aspiration encoding (velocity-only scaling is a slow-gait/fast-gait
chimera), explorer annealing, reproduction-gated (non-lifetime-argmax)
anchor ratchet, occupancy made load-bearing for aspiration feasibility or
deleted.

Direction change (2026-07-18, user): "I basically want pan-1 but without
needing pretraining data or a specific goal. I don't think I want all this
infrastructure." v22 (goal-diverse collection over the v14-lineage stack)
was cancelled mid-run as superseded. A Forward-Backward successor-
representation alternative was designed and implemented but shelved UNRUN
by the same directive (kept for reference at
cleanrl/pan-goal-solver/fb_alternative_unrun.py); its analysis stands:
hindsight BC alone has no improvement operator.

v23 (`ppo_continuous_action_pan_goal_solver_v23.py`) is the minimal
faithful online Pan-1, stated prospectively:
- Quantity #2 as the reference states it: a K-hypothesis winner-take-all
  next-frame DISTRIBUTION head D(belief, f_t, G) (the deterministic MSE
  successors of v19-v22 contradicted the reference and mean-collapsed).
- Quantity #1 load-bearing: the contrastive occupancy readout gates
  imagined goals (largest velocity scale clearing an achieved-pool
  occupancy quantile) — Pan-1's nearest-achievable grounding.
- Thin action attachment: dense one-step inverse dynamics; act =
  invert D's most likely hypothesis. One direct pass, no planning.
- Goals are FRAMES (a specific future observation), not phase-robust
  statistics latents: the statistics goal made the follower's target
  not-a-function-of-the-goal (audit GAP 3) and is retired with the
  GoalProjector.
- Collection replaces the pretraining corpus: commanded goal frames
  switch WITHIN episodes (64 steps), 60% random achieved frames / 40%
  imagined (velocity-scaled top-reward frames, occupancy-gated), rebuilt
  fresh at every switch. No anchor, no ratchet, no reward head, no
  scripted explorer, no proposer network. Reward's only role is sorting
  which achieved frame gets scaled.
- World model: pure LeJEPA (attached target, token-space SIGReg, no EMA),
  trained by nothing but its own losses.
Pre-registered 1.5M falsifiers: reach_servo_gap sustained > 0;
goal_removal_action_mse alive (not pinned at ~0.001); imagined-vs-base
eval arms separating while episodic returns climb past the ~1500-1800
retrieval band.

v23 was cancelled BEFORE producing data (2026-07-18, user: "this is an
incredibly not well thought out solution"). Post-mortem, accepted: v23
routed the entire goal signal through the one-step conditional
P(f_{t+1} | b, G) — the exact channel measured goal-blind three separate
times (v19/v20/v21, goal_removal 0.0002-0.001) — while DELETING the only
channel this family ever measured alive: direct first-action regression
on the far (~50-step) hindsight goal (v16/v17, goal_removal 0.05-0.09
even under a fixed command). Argmax-logit hypothesis selection would
additionally have collapsed onto the behavior-marginal mode, erasing goal
conditioning at act time even if D had learned some. The occupancy gate
consumed a near-chance readout. Lesson recorded: do not re-route an
empirically-alive signal through an empirically-dead bottleneck for the
sake of architectural fidelity to the reference's decomposition.

v24 (`ppo_continuous_action_pan_goal_solver_v24.py`) is the direct
construction, stated prospectively:
- Relabel own trajectories with frames actually reached later (geometric
  offsets, discount 0.98, max 256, mean ~50 — the alive far-goal regime);
  train ONE policy pi(a | belief, f_t, G) with ONE loss: MSE against the
  action actually taken. No successor head, no inverse head, no occupancy,
  no WTA hypotheses, no composed loss, no proposer.
- G is the frame summary of a single future observation, re-encoded fresh
  each iteration. Goals-as-frames retained from v23's one correct call.
- Collection: commanded goals switch within episodes (64 steps); mixture
  60% uniform achieved / 20% top-reward / 20% velocity-scaled top-reward
  (scales 1.25/1.5/2.0, ungated). Random warmup 100k, then policy + OU
  noise. Scaled commands are the sole (minimal) exceed-replay mechanism;
  the known no-improvement-operator risk of pure hindsight BC is accepted
  and will be measured, not architected around in advance.
- World model unchanged (family invariant): pure LeJEPA, attached online
  target, token-space SIGReg, no EMA; follower reads detached memory.
Pre-registered 1.5M falsifiers: (1) reach_servo_gap sustained > 0;
(2) goal_removal_action_mse alive (not pinned at ~0.001); (3) top/scaled
eval arms separating from the uniform-goal arm while returns climb.

v24 partial data (job 102, cancelled externally at 352k of 8M — GPU
reprioritized twice by request; falsifier checkpoint at 1.5M NOT reached,
so nothing below is a verdict):
- follower/action_mse 0.334 -> 0.177, declining steadily: the policy is
  learning to predict actions from context.
- diagnostics/goal_removal_action_mse pinned at 0.0 through 270k, then
  1e-5 / 5e-5 at 311k/352k: the goal input contributes ~nothing yet —
  the policy is so far behavior-cloning the marginal. Whether the late
  uptick is the channel igniting or noise is exactly what 1.5M decides.
- reach_servo_gap noise around zero (4e-5, -1.5e-3, +9.5e-3); eval arms
  (top/scaled/uniform) nearly identical, near-zero returns — consistent
  with a goal-blind deterministic policy that barely moves.
- Training returns -157 -> -134 (max -10) under OU exploration.
Context for interpretation: at 352k, replay was still ~30% uniform-random
warmup and the follower had taken only ~2.8k updates. v16/v17's alive
signal (0.05-0.09) was measured much later in training. Inconclusive;
resume/rerun required to judge the pre-registered falsifiers.

v24 later reached ~448k on a sibling run with the same signature:
goal_removal ~2e-4, reach_servo_gap ~0, action_mse declining without goal
use. Accepted reading: pure far-horizon first-action GCSL under
random/OU data is nearly unidentified; declining action MSE is marginal
behavior cloning, not goal reproduction. Separate review also corrected
the claim that v16/v17's alive goal_removal was "far hindsight alone":
those runs co-trained one-step inverse dynamics (G = f_{t+1}) with the
far term.

v25 (`ppo_continuous_action_pan_goal_solver_v25.py`) is the online Pan
pretrain cut, stated prospectively:
- Reframe: training is pretraining-style goal reproduction on a growing
  online corpus, not reward RL. Episodic return is logged only; primary
  success is state-level goal conditioning and reach metrics.
- Qty #1 occupancy: InfoNCE over (obs-only belief, future frame G).
- Qty #2 next-frame dist D: K-hypothesis WTA on f_{t+1} | (b, f_t, G),
  with batch goal-shuffle gap diagnostics. D is NOT the act-time path
  (v23 failure: invert(argmax D)).
- Action: multi-scale pi(a | b, f_t, G) with short (G = f_{t+1}) + far
  hindsight, plus dense goal-free inverse. Act = pi only.
- Follower belief zeros incoming actions so OU/action history cannot
  screen off G under MSE.
- Collection: v24 goal-diverse mixture (ungated). Reward only ranks
  command frames (preference). No occupancy gate until occ_shuffle_gap
  is measured alive.
- Pre-registered: ~500k early kill if dist/occ shuffle gaps and
  goal_removal stay dead; ~1.5M primary pass requires sustained
  reach_servo_gap > 0 and goal_shuffle_action_mse above floor.
  Preference/return arms secondary only after primary holds.
