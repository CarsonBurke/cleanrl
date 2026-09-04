# SA-Transformer

Self-attention actor/critic backbones with SPO-asymmetric trust regions.
Benchmark: HalfCheetah-v4.

- `pma/` — shared SA→PMA (pooled multihead attention) backbone line.
- `role/` — role-decoupled transformer line (decoupled critics, latent
  readouts, transfer critics).

Distinct from `../ppo_transformer/` (transformer PPO backbones); the split is
the SPO trust region + SA/PMA readout mechanism, documented per file.
