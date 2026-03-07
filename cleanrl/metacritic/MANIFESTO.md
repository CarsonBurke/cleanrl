# Meta-Critic: Self-Aware Policy Optimization

## The Problem

PPO is a greedy depth-first search through policy space. Each gradient step follows the steepest direction on the current batch, clamps the step size, and commits. When the policy descends into a narrow basin — locally optimal but globally poor — it has no mechanism to detect this or recover. By the time returns plateau or collapse, the policy is trapped: the gradients in the basin point inward, reinforcing the bad solution.

Clipping is a band-aid. It limits the damage from bad steps but also limits progress from good ones. It treats the symptom (ratio explosion) not the cause (the critic gave a bad advantage signal that pointed the gradient into a trap).

## Root Cause Analysis

The real problem isn't that the policy moves too far per step. It's that:

1. **The critic is myopic.** It predicts V(s) under the current policy but has no model of how the policy itself is evolving. It can't anticipate that the current optimization trajectory leads to collapse.

2. **The optimizer has no memory.** Each PPO update is independent. The system doesn't learn from its own optimization history — it can't recognize "this pattern of gradient statistics preceded a collapse last time."

3. **There is no breadth.** SGD follows one path. It never considers that a structurally different policy — one reached by a different sequence of updates — might be globally better. It's descending one branch of the policy tree without mapping the others.

## The Insight

In human learning, there is no separate system watching the learning process from outside. The same system that acts is also the system that reflects on whether its learning is going well. The brain has self-awareness of its own optimization state — it notices when learning is stalling, when confidence is misplaced, when a strategy isn't working.

The agent needs this same self-awareness. Not as a bolted-on module, but as an integrated capability: the optimization trajectory is just another input stream, like proprioception. The agent should learn to be cautious not because we impose clipping, but because it has learned from experience that certain optimization regimes precede collapse.

Trust emerges from self-knowledge, not external constraints.

## Architecture

### The Meta-Critic

A small network trained online that observes the optimization trajectory and predicts returns M iterations into the future. It takes as input a window of per-iteration feature vectors:

- Mean episodic return
- Policy entropy, value loss, policy loss
- KL divergence, clip fraction
- Explained variance
- Gradient norm statistics
- Parameter delta norm, direction consistency (cosine similarity with previous delta)
- Current learning rate and its multiplier

From this window, it predicts: **what will the mean return be M iterations from now?**

Training is by hindsight: M iterations after collecting features, the actual return becomes the target. The meta-critic learns which optimization trajectories lead to improvement and which lead to decline.

### LR Modulation

The meta-critic's predictions modulate the effective learning rate:

- Predicted improvement → allow larger steps (scale LR up)
- Predicted decline → brake (scale LR down)
- Uncertain → maintain current rate

This is the minimum viable intervention. The meta-critic doesn't change the loss function, the architecture, or the exploration strategy. It just controls how fast the policy moves based on learned foresight about where the optimization is heading.

### Toward Self-Aware Critics (Future)

The deeper version: feed optimization trajectory features directly into the value network as additional input. The critic then learns to be uncertain when the optimization is unstable — it has seen that "when gradients look like this, my predictions become unreliable." This naturally reduces advantage magnitude, which naturally reduces policy step size.

No clipping needed. The system clips itself because it knows when to distrust its own signal.

### Toward Systematic Exploration in Policy Space (Future)

Human learning is noisy but targeted — like bisection. Try one extreme, then the other, then narrow down. Each experiment is designed to maximize information about which direction is best, not just follow the gradient.

The meta-critic enables this: once it can predict outcomes of optimization trajectories, it can evaluate hypothetical update directions without executing them. "If I perturb the policy this way, the meta-critic predicts improvement. That way, decline." This turns policy optimization from greedy descent into informed search.

The noise isn't noise — it's probing. And the probing should be along axes the system is most uncertain about.

## Implementation

```
cleanrl/metacritic/
  MANIFESTO.md             -- this file
  ppo_metacritic_v1.py     -- meta-critic with LR modulation
```

### Current: ppo_metacritic_v1.py

- Forks lstd_linear_tanh (best current architecture)
- Collects 16 optimization features per iteration
- Meta-critic MLP (160 → 64 → 32 → 1) takes window of W=10 feature vectors
- Predicts mean return M=10 iterations ahead
- Trained online with MSE, batch of 32 random historical samples per iteration
- Modulates LR by multiplicative factor in [0.2, 2.0]
- Standard PPO clipped loss retained (meta-critic is additive, not replacing)

### Success Criteria

1. **Meta-critic prediction accuracy**: correlation > 0.5 between predicted and actual returns M steps ahead
2. **LR modulation helps**: mean return improves by >5% over baseline lstd_linear_tanh on 2/3 environments
3. **Collapse reduction**: fewer instances of >50% return drop from recent peak
4. **Eventual goal**: remove PPO clipping entirely without performance degradation, with trust emerging from the meta-critic's learned foresight

## Open Questions

1. **Feature engineering vs learning**: are hand-picked optimization features sufficient, or should the meta-critic learn its own features from raw gradient/parameter data?
2. **Non-stationarity**: the optimization landscape changes as training progresses. The meta-critic must adapt. Weight recent data more? Use a recurrent architecture?
3. **Intervention granularity**: LR modulation is coarse. Could the meta-critic modulate per-parameter learning rates, or per-sample advantage weighting?
4. **Self-referential learning**: the meta-critic's LR modulation changes the optimization trajectory, which changes the meta-critic's training data. Does this create stable or unstable feedback loops?
5. **How far ahead**: M=10 is arbitrary. Shorter horizons are easier to predict but less useful. Longer horizons capture more but are noisier. Should M be adaptive?
