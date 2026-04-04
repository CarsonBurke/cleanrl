# HL-Gauss Distributional Critic

Replaces the standard scalar MSE critic with a categorical distribution over discretized value bins, trained via cross-entropy with Gaussian-projected targets. Based on ["Stop Regressing" (Farebrother et al., 2024)](https://arxiv.org/abs/2403.03950).

## Core idea

The critic outputs `num_bins` logits instead of a single scalar. Targets are projected onto the bin support using a Gaussian kernel (HL-Gauss), and the value loss becomes cross-entropy. This yields better representation learning and more stable value estimates than MSE regression.

## Variants

| Suffix | What it adds |
|-|-|
| `hlgauss` | Base HL-Gauss critic with standard PPO actor |
| `_ablation` | Ablation study configurations |
| `_dg` | Delightful Policy Gradient gating on policy loss |
| `_dlogstd` | Decoupled learnable log-std (not baked into actor network) |
| `_dgvl` | Dual gradient value learning (separate policy/value gradient paths) |
| `_dgvl_norm` | DGVL + gradient normalization |
| `_dgvl_pctl` | DGVL + percentile-based value scaling |
| `_dgvl_stdnorm` | DGVL + standardized normalization |
| `_dgvl_sym` | DGVL + symmetric loss variant |
| `_sepopt` | Separate optimizers for actor and critic |
| `_shared` | Shared actor-critic backbone |
| `_vlconf` | Value loss with confidence weighting |
| `_vlscale` | Scaled value loss |
| `silu_*` | SiLU activation variant |
| `rmssilu_*` | RMSNorm + SiLU activation variant |
| `symhlgauss_*` | Symmetric HL-Gauss (bins centered at 0) |
| `_dpg` | Deterministic policy gradient through the distributional critic |
