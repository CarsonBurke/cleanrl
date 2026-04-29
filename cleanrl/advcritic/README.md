# Advantage Critic PPO Family

This folder contains PPO variants that separate the policy-facing advantage signal from the bootstrap value baseline.

- `ppo_continuous_action_advcritic_v1.py`: pure action-conditional advantage critic trained from Monte Carlo reward-to-go. Negative ablation; removing TD/GAE was too destructive.
- `ppo_continuous_action_advcritic_gae_v2.py`: restores `V(s)` for TD/GAE and trains `A(s,a)` to distill GAE. Better on Hopper and Walker2d.
- `ppo_continuous_action_advcritic_introspective_v3.py`: adds an uncertainty head to `A(s,a)` and blends learned advantages with raw GAE according to predicted confidence.
- `ppo_continuous_action_advcritic_tdres_gae_v4.py`: keeps raw GAE, trains `A(s,a)` only as a one-step TD-residual predictor, and splices it into the current residual term without distilling from full GAE.
- `ppo_continuous_action_advcritic_fixedtd_gae_v5.py`: same non-distillation idea as v4, but trains `A(s,a)` against the fixed rollout TD residual instead of a target that moves as `V(s)` updates.
- `ppo_continuous_action_advcritic_gaepure_v6.py`: trains `A(s,a)` on full GAE targets, then uses only the critic's detached predicted advantages for the policy update.

The current working hypothesis is that an advantage critic should behave like a fallible self-model: use trajectory evidence directly when uncertain, and only denoise or reshape the policy gradient when the critic has learned a coherent local action model.
