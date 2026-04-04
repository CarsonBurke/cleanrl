# Architecture Variants

Network architecture modifications to the standard PPO MLP: alternative activations, normalization layers, observation encodings, and training structure changes.

## Variants

| File | What it does |
|-|-|
| `_silu` | SiLU activation (no normalization) |
| `_silu_rmsnorm` | SiLU activation + RMSNorm |
| `_rmssilu` | Linear -> RMSNorm -> SiLU layers (replacing Linear -> Tanh) |
| `_rmssilu_actanh` | RMS-SiLU + action tanh squashing |
| `_rmssilu_dlogstd` | RMS-SiLU + decoupled learnable log-std |
| `_rmssilu_actanh_dlogstd` | RMS-SiLU + action tanh + decoupled log-std |
| `_rmssilu_dgvl` | RMS-SiLU + dual gradient value learning |
| `_dreamerv3` | DreamerV3-style symlog observation encoding + symexp-twohot distributional critic |
| `_sepopt` | Separate optimizers for actor and critic (no shared loss) |
