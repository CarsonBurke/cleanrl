# LEWM and Dreamer4 Alignment Notes

These notes summarize the local reads of `../le-wm` and `../dreamer4` for the LEJEPA imagined PPO variants in this folder.

## LeWM

- LeWM training is only next-encoder-latent prediction plus SIGReg.
- The prediction target is the future encoder embedding, not a future predictor embedding.
- LeWM rollout is autoregressive over predicted latent embeddings conditioned on action history.
- LeWM cost is an inference/planning criterion: final predicted latent MSE to a detached encoded goal latent.
- LeWM does not train reward, value, or termination heads.
- During eval/planning, the cost model is frozen and in eval mode.

Implication for our variants: `lejepa/` should mean latent prediction plus SIGReg. Reward, termination, and any task cost should be logged and reasoned about as `dynamics/` or `cost/` readouts over detached latent rollouts, not as part of the LEJEPA representation objective.

## Dreamer4

- Dreamer4 trains a generative dynamics model plus reward/terminal/action auxiliary predictions.
- Its value head is not part of world-model training. The trainer excludes value-head params from the world-model optimizer and trains value from imagined returns during policy/value learning.
- Dream generation runs the model in eval mode and restores the previous mode afterward.
- Rewards and continuation probabilities are generated from agent-token embeddings on dreamed states.
- Generated rewards/continuations are arrival-state aligned, then shifted left for policy optimization so the final dreamed state is bootstrap-only.
- Terminal handling separates two roles:
  - soft continuation probability is used in GAE;
  - Bernoulli terminal sampling is used to set dream rollout length / early stop.
- Terminal loss uses asymmetric smoothing: nonterminal targets are clamped to `1 - gamma`, while terminal targets remain 1.
- Terminal output bias is initialized strongly nonterminal to avoid early discount collapse.

Implication for our variants: the PPO critic should be the value learner. A separate dynamics value head is not useful unless it is explicitly used as a terminal cost/value in imagination. Termination should not be double-counted by multiplying soft continuation and sampled nonterminal masks into the same transition discount.

## Current Practical Priority

Most of this is architectural hygiene rather than the immediate performance bottleneck. The immediately useful changes are:

- construct dreams with the world model in eval mode;
- keep LEJEPA logs/objective separate from reward/termination/cost readouts;
- remove or disable unused dynamics value loss;
- use soft continuation for imagined GAE without also applying sampled terminal to the same continuation.

The more speculative follow-up is reward/cost conditioning. For HalfCheetah, reward contains an action-cost component, so a reward readout from next latent alone can alias actions. A better cost readout should either receive the action explicitly or include an explicit control-cost term.

## v45 Refactor Direction

The `costaware_v45` variant applies the immediate refactor:

- reward readout consumes `z_t`, `a_t`, `sum(a_t^2)`, and `z_hat_{t+1}`, all detached from the WM latent path;
- terminal readout remains detached from `z_hat_{t+1}` and is initialized toward nonterminal;
- dynamics value loss is removed; imagined PPO critic owns value learning;
- dreams are constructed in eval mode;
- dynamics diagnostics compare autoregressive predicted rewards against real rewards under recorded actions and log reward action-sensitivity.
