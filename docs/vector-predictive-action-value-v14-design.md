# Vector predictive action value: v14 design hypothesis

Status: original reviewed design; implementation details and experiment records now live in [vector TD response v14](vector-td-response-v14.md). Builds only on our v13 implementation and its recorded training evidence. In particular, the implementation uses fixed Beta(2,2) feature coordinates and integrates the full continuation critic, superseding the preliminary collection-policy centering discussion below.

## Motivation

V13 improved faster than v12 initially (1,330 versus 481 at 2M), then plateaued. This is evidence of useful early learning, not evidence that its predicted actor gains were calibrated. Its vector predictions already depend on state and action. Two concrete limitations deserve testing: coarse, noisy long-horizon targets, and an action-response family containing only singleton and pairwise actuator interactions. These are structural limitations, not established causes of the plateau.

The goal is a learned latent that supports accurate predictions of reward-producing future consequences of a state/action pair. Extra output width, generic physical-state reconstruction, and a latent anchored only by one scalar reward readout do not establish that property.

## Predictive object and learned latent

Define U(s,a,age,h)[c] as expected cumulative reward component c over the next h steps, clipped to the episode's remaining duration, with subsequent actions from the collection policy. Preserve progress and all individual actuator costs as separate observed components. Summing components at the full remaining horizon recovers the task's expected remaining return.

Learn action features phi_k(s,a) whose shapes depend on state, shared across horizon queries. Use products of Beta-integrable powers over selected actuator subsets, including subsets of size greater than two, together with exact collection-policy score features. State-dependent powers on pairwise supports alone would not remove v13's inability to express genuine higher-order action interactions.

The critic predicts a state-conditioned coefficient tensor indexed by horizon, component, and feature. Its combination with phi(s,a) is a learned state-action predictive representation. Several independent component/horizon queries supervise it; there is no scalar value head between the latent and these predictions. This expands the action function class rather than merely adding more reward bins.

Analytic Beta moments remain available for state-dependent powers because state and critic parameters are held fixed during each action expectation. Freeze powers, coefficients, collection-policy centering, and normalization throughout actor optimization. No generated future observation or physical-state decoder is needed.

## Targets from real short segments

The core learning signal is a **vector temporal-difference residual**. For the one-step form:

`delta_h = reward_components + continuation_mask * frozen expected U(next_state, next_action, next_age, h-1) - U(state, action, age, h)`

All terms retain the component axis. The next-action expectation uses the collection policy; the continuation prediction is detached and frozen for the rollout. At h=1 or an actual episode end, continuation is zero. For finite-horizon undiscounted return this expression has no discount factor; choosing gamma=0.99 would change the target objective and should be an explicit experimental choice.

Train the action-conditioned predictor to reduce this vector residual. This is ordinary semi-gradient TD with vector predictions, not full residual-gradient optimization through both sides of the Bellman equation. It is also not a scalar TD error broadcast into multiple latent channels. The residual corrects the existing prediction; it does not itself identify a useful latent basis or provide a noise-free actor advantage. The actor uses the learned action-response function described below.

The one-step form is the conceptual base. Real multi-step segments generalize it when longer credit propagation is needed; segment length remains an explicit choice rather than silently introducing GAE or repeated target sweeps.

For each requested horizon h, use a real segment of length m no longer than h, the remaining episode, or the available rollout. The target vector is:

`Y_h = observed reward-component sum over m steps + frozen expected U(actual endpoint, next action, endpoint age, h-m)`

Continuation is zero at an actual episode end or when h-m is zero. A rollout boundary is not an episode end: bootstrap from its actual final observation. Terminal observations must never be replaced by reset observations. Unknown warmup episode ages need the same explicit resolution or exclusion as v13.

Short queries entirely covered by the segment have directly observed targets. Longer queries use the same critic snapshot and collection policy for all fitting on that rollout. Horizon conditioning permits the h-m query without rounding to a stored temporal bin. Snapshot once, construct targets once, then fit the entire fresh batch with those targets fixed. This uses a Bellman bootstrap, but no repeated Bellman target sweeps, replay, or GAE.

This changes the statistical tradeoff: shorter observed segments reduce continuation sampling noise while introducing approximation and bootstrap bias. It does not supply new counterfactual observations or guarantee correct action effects. Segment length and horizon coverage are substantive algorithm choices, not details to conceal.

## Actor objective and optimization pressure

The actor maximizes the analytically integrated full-remaining-horizon vector prediction, summing true signed reward components only at the final objective. Critic learning supervises all component/horizon predictions independently. The final scalar objective is unavoidable for maximizing one benchmark return; a scalar bottleneck in critic representation or targets is avoidable.

Independent vector supervision removes the algebraic loophole of training only w-transpose-z, which leaves all orthogonal latent directions unconstrained. It cannot guarantee that every latent dimension has a distinct useful meaning, nor should dimensions be forced apart without evidence that the task requires them. Vector prediction loss is a reward-grounded inductive bias, not mathematically identical to maximizing return.

No contrastive objective, arbitrary state reconstruction, learned target whose only anchor is a scalar decoder, or additional exploration objective is proposed.

## Evaluation and remaining uncertainties

Use fresh seed 1, HalfCheetah-v4, 16 environments by 2048 steps, full-batch fitting, KL ceiling 0.03, and the same matched-step return comparisons. No checkpoint resumption or held-out promotion gate. Any implementation needs numerical checks for analytic moments and derivatives with state-conditioned powers, horizon/end/boundary handling, and frozen-target lifetime before its queued training run.

The hypothesis is that a richer joint-action response family, trained from temporally compositional vector targets, preserves v13's early learning while supporting improvement beyond the plateau. Neither part is validated yet. A combined experiment cannot separately attribute improvements to the feature family and target construction.

V13's approximate natural-gradient solve is a separate limitation: median relative residual 0.0446. Record solver accuracy and actual accepted KL in the next experiment; do not attribute all failures to the critic or present modeled positive gains as proof of environmental improvement. Long-horizon bootstrapping, finite fresh-data action coverage, policy changes between rollouts, and conditioning of the learned feature family remain material risks.
