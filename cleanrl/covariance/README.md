# Structured Covariance Parameterizations

Full or structured covariance matrix parameterizations for the policy distribution, going beyond diagonal Gaussian noise. These methods learn cross-actuator correlations directly in the policy.

## Variants

| File | Parameterization |
|-|-|
| `_logchol` | Gram-scaled covariance via log-Cholesky factorization with tanh-bounded correlations and zero-sum diagonal |
| `_sharedw_logchol` | Log-Cholesky with shared weights between actor mean and covariance |
| `_prec_eigen` | Precision eigendecomposition: covariance via eigh(A) with exp(-eigenvalues), trivial entropy gradient |
| `_lattice` | Latent exploration via learned lattice structure ([Latent Exploration for RL](https://arxiv.org/abs/2305.20065)) |
| `_lattice_tanh` | Lattice exploration with tanh-bounded actions |
