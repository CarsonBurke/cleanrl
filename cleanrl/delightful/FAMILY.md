# Delightful

Delight-gated (`advantage * surprisal`) PPO family on MuJoCo.

- `onpolicy/` — full-batch / long-GAE on-policy HL-Gauss variants.
- `ppo/` — tanh/EMA state-sigma actor variants (bounded-reward, latent
  surprisal, native-tail mean-only).
- `qcritic/` — Q-critic + reward-EMA line (v15-v23).
- `ppo_continuous_action_delight_v1.py` — original v1 entry point.
