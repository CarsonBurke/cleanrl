# LeWorldModel notes

Local source reviewed: `../../../le-wm`, July 17, 2026.

Primary files:

- `README.md`: project claims and usage.
- `jepa.py`: JEPA representation, latent rollout, and terminal goal cost.
- `module.py`: SIGReg, action embedding, and autoregressive predictor.
- `train.py`: exact training objective.
- `eval.py` and `config/eval`: goal construction and planning configuration.

## What LeWM actually learns

LeWM is an action-conditioned latent dynamics model, not a learned goal
proposer and not a goal-conditioned policy.

For each image sequence it:

1. Encodes every frame with a ViT and takes the CLS token.
2. Projects that token to a 192-dimensional latent.
3. Embeds the corresponding actions.
4. Uses a causal, action-conditioned transformer to predict the next latent at
   each context position.

The default context contains three frames and predicts one step ahead at each
position. Training sequences therefore contain four frames. Actions condition
the transformer through zero-initialized AdaLN modulation.

Both sides of the prediction loss use the same online encoder. There is no
frozen target encoder, EMA encoder, pretrained visual model, decoder, reward
head, termination head, or auxiliary supervision.

## Training objective

The complete default objective is

```text
mean_squared_error(predicted_next_latent, encoded_next_frame)
    + 0.09 * SIGReg(all_encoded_frames)
```

SIGReg projects embeddings onto 1,024 random unit directions and penalizes
deviation of their empirical characteristic functions from a standard Gaussian
at 17 knots. It prevents representational collapse while allowing end-to-end
training with a single encoder.

The supplied datasets contain actions and are collected offline. Depending on
the task they are expert or random datasets. The learner does not consume a
reward objective, but its behavior quality can inherit the coverage and skill
of the dataset.

## How LeWM solves a goal

The goal is external. Evaluation chooses a real dataset state a fixed number of
steps after the initial state and supplies its rendered image as the goal.
LeWM encodes that image into `goal_emb`.

For each proposed action sequence, LeWM:

1. Encodes the current observation history once per cost call and shares it
   across that call's candidate sequences.
2. Autoregressively predicts future latents under the candidate actions.
3. Computes terminal squared latent distance to `goal_emb`.

An external optimizer then changes the action sequences to reduce that cost.
The default evaluation uses CEM with 300 candidate sequences, 30 optimization
iterations, and a horizon of five action blocks. Each block contains five
primitive actions, matching the training frameskip, so a plan spans 25 primitive
steps—the same as the evaluation goal offset. Only the terminal prediction is
scored, using `sum(||z_pred_final - stopgrad(z_goal)||^2)`; intermediate states
have no goal cost.

One CEM solve therefore evaluates 9,000 candidate sequences and 45,000
candidate latent transitions per environment. `get_cost` is invoked on every
CEM iteration, so invariant current and goal images are each encoded 30 times
rather than cached across the solve. With a receding horizon of all five blocks,
the policy executes 25 primitive actions before replanning; the 50-step
evaluation budget uses roughly two large solver bursts.

An alternative Adam solver uses 100 samples and 30 gradient steps through the
latent rollouts. It is also online planning, not a learned policy.

The local repository delegates solver implementation to `stable_worldmodel`,
does not pin that dependency, and uses an older solver API than current
upstream. Exact resampling behavior is therefore not reproducible from this
checkout alone. The local model and numerical configuration still make the main
inference cost explicit: many batched autoregressive world-model evaluations
are performed before an action block is selected.

## Comparison with the intended family

| Property | LeWM | Pan Goal Solver target |
|---|---|---|
| Goal source | External achieved image | Detached belief-to-goal head maximizing predicted reward |
| Representation training | Next-latent MSE + SIGReg | LeJEPA-style self-supervision plus goal learning |
| Action learning | No policy; dynamics model uses labeled actions | Direct goal-conditioned action prediction |
| Goal use in training | None | Hindsight goal/successor objective |
| Inference | Optimize action sequences through latent rollouts | One fused `belief -> goal -> action` pass |
| Planning/search | CEM or gradient optimization | Forbidden |
| Reward model | None | Detached head on predicted-next goal latent |

LeWM and the target family share a useful geometric principle: encode a goal in
the same latent space as predicted observations and learn without reconstructing
pixels. The control mechanisms are otherwise different. LeWM asks which action
sequence makes a dynamics rollout end near a supplied goal. The target system
must learn both which goal to pursue and the action directly, amortizing control
into network weights rather than solving an optimization problem online.

## Lessons worth transferring

- End-to-end JEPA training can be stabilized with a simple Gaussian latent
  regularizer instead of an EMA or pretrained encoder.
- Follower reachability should be evaluated in learned observation/successor
  space. The signal used to decide which goal is better is a separate design
  axis and must be stated explicitly.
- Actions belong in transition/control conditioning, not in the observation goal.
- Put the learned goal and predicted observations in the same coordinate space.
  An open-ended goal may exploit reward-model error; compare predicted reward
  with achieved reward so that pressure improves the model rather than being
  hidden by an unrelated preference objective.
- A short action-conditioned predictor is useful during representation learning,
  but repeatedly rolling it inside an optimizer is the inference bottleneck to
  remove.

## What LeWM does not answer

LeWM provides no mechanism for predicting a better goal. Its goal is a real
future observation selected by the evaluator. Minimizing its terminal cost over
the goal latent itself would choose an easy/current state, while unconstrained
maximization would leave the data manifold and exploit model error.

It also does not show how to train a direct policy from scratch on its own online
data. Replacing its solver requires a separate hindsight goal-conditioned action
objective and adequate exploration. These are central research problems for
this family, not implementation details inherited from LeWM.
