# gSDE: Generalized State-Dependent Exploration

Variants of generalized State-Dependent Exploration (gSDE) with Ornstein-Uhlenbeck temporal correlation, multi-scale noise banks, and custom covariance parameterizations. Based on [gSDE (Raffin et al., 2022)](https://arxiv.org/abs/2005.05719).

## Core idea

Standard gSDE samples a noise matrix once per rollout and reuses it, coupling exploration to the learned features. These variants extend this with:
- **OU evolution**: continuously evolve the noise matrix with an OU process for smoother temporal correlation
- **Multi-scale banks**: maintain K noise matrices at different timescales with state-dependent mixing
- **Custom covariance**: Gram-scaled, learnable noise floors, entropy ceilings

## Variants

| Suffix | What it adds |
|-|-|
| `gsde` | Standard gSDE implementation |
| `ougsde` | OU-evolved noise matrix with state-dependent correlation |
| `msougsde` | Multi-scale OU bank with state-dependent mixture weights |
| `msougsde_fixedmix` | Fixed (non-learned) mixture weights |
| `msougsde_renewal` | Renewal-based resampling strategy |
| `msougsde_sticky*` | Sticky resampling (persist noise longer) |
| `custom_gsde` | Gram-tanh covariance + learnable noise floor/prescale |
| `custom_gsde_ent` | Custom gSDE + entropy ceiling |
| `custom_gsde_flow` | Custom gSDE + normalizing flow |
| `custom_gsde_twohot` | Custom gSDE + two-hot distributional critic |
| `space` | LSMN with K/tau annealing for exploration stability |
| `space_msou` | LSMN combined with multi-scale OU |
