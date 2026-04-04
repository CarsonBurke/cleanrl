# STSTS: Space-Time Shared-Trunk Transformer

Axial space-time transformer backbone (Dreamer4-style) with learned CLS tokens for actor, critic, and optionally SDE heads. Spatial blocks treat observation features as an unordered set; temporal blocks use RoPE for sequence position.

## Core idea

A single shared transformer processes observation history with separate CLS query tokens that specialize into actor/critic/SDE readouts. This gives the policy temporal context and a rich shared representation without separate networks.

## Variants

| Suffix | What it adds |
|-|-|
| `_separate` | Separate actor/critic backbones (no sharing) |
| `_shared_cls` | Base: shared backbone with actor+critic CLS tokens |
| `_shared_proj` | Shared backbone with projected readouts |
| `_sde_*` | State-dependent exploration (SDE) covariance head |
| `_sde_cholesky` | Cholesky-parameterized full covariance |
| `_sde_matrix*` | Matrix-valued noise with various state conditioning |
| `_sde_lowrank` | Low-rank covariance approximation |
| `_sde_stateent*` | State-conditioned entropy bonus |
| `_sde_mtp` | Multi-token prediction for exploration |
| `_worldmodel*` | Integrated latent world model for imagination |
| `_worldmodel_sde4cls*` | World model with dedicated SDE CLS token |
| `_worldmodel_uncgate` | World model with uncertainty gating |
| `_teacher*` | Hindsight teacher with bidirectional attention for distillation |
| `_latent_perturb` | Latent-space perturbation for exploration |
| `_signadv` | Sign-of-advantage exploration gating |
