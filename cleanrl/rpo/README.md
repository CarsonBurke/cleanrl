# RPO & Search-Based Methods

Random Perturbation Optimization variants and search-based exploration strategies that modify the policy optimization objective or use population/adversarial search.

## Variants

| File | Approach |
|-|-|
| `rpo_continuous_action` | Base RPO: random perturbation for exploration stability |
| `rpo_continuous_action_gsde` | RPO + generalized state-dependent exploration |
| `rpo_continuous_action_lattice` | RPO + lattice-based exploration |
| `learnable_rpo_continuous_action` | RPO with learnable perturbation parameters |
| `rpo_adaptive_search` | Adaptive search over RPO hyperparameters |
| `rpo_curiosity_search` | RPO with curiosity-driven objective |
| `rpo_multi_search` | Multi-objective RPO search |
| `ppo_adversarial_search` | Adversarial perturbation search for robust policies |
