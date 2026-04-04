# Tbot: Trading-Bot Inspired Architecture

Adapts architectural ideas from a trading bot for MuJoCo continuous control. Combines soft-sign bounded SDE noise, distributional critic (DreamerV3-style symlog two-hot), RMSNorm + SiLU activations, and mean-scale decoupling.

## Core idea

1. **Soft-sign SDE noise**: `sigma = |x / (|x| + 1)| * scale` -- better gradient flow than tanh, naturally bounded, fully state-dependent
2. **Distributional critic**: 255-bin categorical over symlog-spaced buckets with cross-entropy loss
3. **RMSNorm + SiLU**: modern activation/normalization replacing Tanh
4. **Mean-scale decoupling**: separate learnable scale for initial action magnitude

## Variants

| Suffix | What it adds |
|-|-|
| `tbot` | Full architecture (distributional critic + SDE) |
| `_scalar` | Scalar critic instead of distributional |
| `_scalar_lstd` | Scalar critic + learnable log-std |
| `_scalar_lstd_gated` | Gated noise modulation |
| `_scalar_lstd_mvn` | Multivariate normal distribution |
| `_scalar_lstd_noclamp` | No clamping on log-std |
| `_scalar_lstd_now` | No output weight coupling for noise |
| `_scalar_lstd_prescale1` | Prescale = 1 ablation |
| `_scalar_lstd_shared_w` | Shared weights between mean and noise |
| `_scalar_lstd_softsign` | Softsign activation for noise |
| `_scalar_lstd_wide_cov` | Wider covariance parameterization |
| `_tight` | Tighter noise bounds |
| `_tight_nonorm` | Tight bounds without normalization |
