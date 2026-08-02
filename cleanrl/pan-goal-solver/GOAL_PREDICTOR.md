# Open-ended goal prediction

## Exact computation graph

The goal is one deterministic latent, predicted anew on every environment step:

```text
b_t = world_model(observation/action history)
G_t = P(stopgrad(b_t))
a_t = pan_actioner(current_goal_latent_t, G_t)
```

`P` is a detached head off the world-model belief. It does not receive reward,
predicted reward, a requested reward level, an aspiration schedule, difficulty,
novelty, reachability, or a set of candidates. Calling it on each step is one
forward pass, not search or planning. Although the point is recomputed, there is
no goal distribution and no iterative inference-time optimization.

The point can be physically impossible. It should represent the fastest,
highest-reward cheetah that the model can currently imagine. The simulator may
prevent the follower from reaching it; the point still gives the follower a
clear direction toward the physical limit.

## Reward belief

A frozen reconstruction decoder followed by a separate head predicts reward
from the world model's action-conditioned one-step prediction:

```text
z_hat_(t+1) = world_model_successor(history_t, action_t)
x_hat_(t+1) = decoder(stopgrad(z_hat_(t+1)))
r_hat_t = R(x_hat_(t+1))
L_reward = MSE(r_hat_t, r_t)
```

`R` is a state/reward model, not an action-value critic. It does not receive the
current action as a separate input, choose an action, estimate return-to-go, or
train the follower. The predicted next latent already contains the modeled
consequence of the real action. Reward gradients stop at that latent, so neither
`R` nor `P` can reshape the world representation to make their job easier.

The frozen decoder begins with RMS normalization. Goal distance was declared
inconsequential, and the follower's bounded delta feature converges to the
normalized direction from its current latent to a far `G`. Consequently the
complete reward evaluator satisfies `R(decoder(cG)) = R(decoder(G))`: the goal
head cannot manufacture higher reward merely by increasing latent norm. The
decoder is trained only by the world model's real-history reconstruction loss;
reward never changes it. This is an architectural symmetry and semantic
interface, not a reachability constraint or an extra goal preference.

## Improving the goal

Every goal optimization cycle first fits `R` on real reward MSE. It then freezes
all parameters of `R` and updates only `P`:

```text
G = P(stopgrad(b))
L_goal = -R_frozen(decoder_frozen(G))
```

Thus the goal head learns the latent its current reward belief scores highest.
There is no fixed `y*`, no best-seen-state target, no reward-conditioned ray,
and no counterfactual. Optimizing `P` is amortized during learning; inference
samples the resulting point with a single call.

The reward model can be exploited off the achieved latent manifold. That is the
intended pressure on this design, not a reason to introduce a competing novelty,
difficulty, or reachability objective. The remedy is a better world/reward
objective and better data. Predicted reward, real induced reward, and goal norm
must be logged separately so exploitation is visible rather than silently
mistaken for progress.

## The doctor analogy

At first, a person's representation of “doctor” may encode only a vague cluster
of features. Their current belief produces a correspondingly weak goal point.
As experience improves the representation, it gains meaningful dimensions:
cardiology versus anesthesiology, preferred work, competence, lifestyle, and so
on. The reward belief learns which combination the person values, and the goal
head revises the single point accordingly.

At every step the person evaluates and acts on distance from the goal they can
currently represent. Two kinds of change remain distinct:

```text
embodied progress: z_t moves toward the pre-action G_t
goal revision:     improved belief changes P(b) and therefore G
```

The first is measured by freezing `G_t` across the observed transition. The
second is measured by evaluating successive goal-head versions on the same
fixed belief histories. A changed goal can reflect a clearer conception rather
than physical regression.

## Pan-like follower

The actioner is trained without reward:

- future observation-history latents from the same episode become hindsight
  goals;
- a goal-conditioned successor predicts the real next latent that followed each
  hindsight-conditioned state;
- inverse-action prediction maps that real local transition to its executed
  action, while composed first-action prediction trains the fused path;
- bounded delta features preserve the magnitude of local achieved changes for
  inverse-action learning while saturating to direction-only far-goal control;
- goal occupancy/successor learning teaches goal-conditioned structure;
- inference emits one action directly from the current belief and `G`.

This is the strongest transferable part of Pan-1. Pan-1 itself uses externally
provided goal images and does not disclose a learned maximally rewarding goal
head. `P` and `R` are this family's open-ended online extension.

## Required invariants and diagnostics

- Exactly one goal-head call and one follower call per environment action.
- Exactly one reward-head update and one goal-head update per follower cycle.
- Reward changes no world-model or follower gradients.
- Goal maximization changes no reward-head parameters.
- The goal head receives detached world belief.
- The reward head receives detached predicted-next latent.
- Hindsight goals never cross episode boundaries.
- Removing or shuffling `G` materially changes the direct action.
- Report reward-prediction MSE, predicted reward of `G`, real episodic reward,
  frozen-goal MSE progress, and fixed-history goal revision separately.
