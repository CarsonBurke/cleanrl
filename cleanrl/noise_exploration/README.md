# Noise & Correlated Exploration

Strategies that inject structured, temporally correlated noise into the exploration process. These approaches modify how noise is generated and applied rather than changing the policy distribution's parameterization.

## Approaches

| File | Strategy |
|-|-|
| `_pink_noise` | 1/f colored noise for temporally correlated exploration ([Eberhard et al.](https://arxiv.org/abs/2011.15034)) |
| `_spectral_mix` | Colored noise + state-dependent low-rank mixing matrix for cross-actuator covariance |
| `_soa*` | Spectral Observation Augmentation: OU noise + periodic signals appended to obs as a "stochastic heartbeat" |
| `_coae*` | Correlated Observation Augmentation: multi-scale OU noise appended to obs (architecture-agnostic) |
| `_ace` | Adaptive Correlated Exploration: OU noise evolution replacing discrete gSDE resampling |
| `_arou` | Autoregressive OU action noise with explicit exploration memory input |
| `_tanh_arou` | AROU with tanh squashing |
| `_tanh_msou_noise` | Tanh-squashed multi-scale OU noise |
| `_mtsor` | Multi-Timescale Stochastic Orthogonal Rotation: AR(1) noise bank + orthogonal mixing |
| `_itce` | Independent Time-Correlated Exploration: learned cross-actuator covariance + gSDE temporal reuse |
| `_tarpe` | Temporally Adaptive Rank-Projected Exploration: OU noise + low-rank state-gated mixing |
