# Latent costate full-batch v12

The requested defaults are full-batch model/critic fitting and one detached critic target per rollout. The latent covector critic and direct state/action/alpha-beta derivatives are unchanged. No replay, GAE, scalar value head, generated-state bootstrap, contrastive objective, or checkpoint resume.

## Hypothesis and configuration

The original explicitly provided PPO source (`cleanrl/ppo_continuous_action_normres_beta_gradient_v2.py`) uses 16 environments × 2,048 steps = **32,768 transitions**, one minibatch, and ten optimization epochs. V12 keeps that rollout/batch size. The critic excludes unknown episode ages after warmup using weights; it does not partition the data.

With full batches there are ten model and eight critic Adam updates per rollout, versus 80 and 64 in the minibatch reference. Targets and their normalization scales are computed once before critic fitting and frozen for all eight epochs. Recomputing action credit after fitting does not perform another critic fit. Eight regression epochs are not eight Bellman target refreshes.

Hypothesis: raising the learning rates improves fitting on each fresh full batch enough to recover useful vector credit without repeated Bellman backups. This is an empirical hypothesis, not an equivalence to minibatch Adam. The two learning rates move together, so this experiment does not identify their separate contributions.

| Setting | Old-rate comparison | V12 default |
|---|---:|---:|
| Model learning rate | 0.001 | 0.003 |
| Critic learning rate | 0.0003 | 0.0009 |
| Fitting batch size | 32,768 | 32,768 |
| Model / critic fitting epochs | 10 / 8 | 10 / 8 |
| Critic target refreshes per rollout | 1 | 1 |
| Gamma | 1 | 1 |
| Actor KL ceiling | 0.03 | 0.03 |

Gamma stays at 1 because the prior isolated 0.99 experiment did not improve the reference, and the current experiment targets fitting and repeated backups. Mean-KL acceptance controls the local actor step; accepted updates are not guarantees of return improvement. Reward remains scalar at the environment interface, while the critic predicts a latent covector pulled back to a state-gradient vector. The physical transition model preserves state coordinates instead of compressing the entire learning target to reward.

## Jobs and validation

- 7545: CUDA numerical contracts, parallel limit 1, time limit 20 minutes; 23 passed. The compiled learning contract now fits the same targets for eight consecutive full-batch steps and checks that target and next-costate snapshots remain unchanged.
- 7546: fresh seed-1 HalfCheetah-v4, nominal 8M transitions, old rates; parallel limit 1, time limit 60 minutes.
- 7547: same setup, default 3× rates; parallel limit 1, time limit 60 minutes.

Both training jobs depend on successful completion of 7545. No smoke training. Monitor return trajectories and fitting diagnostics; cancel persistent, clear underperformance after sufficient data rather than treating a single decline as failure. Source is frozen once evaluated.

Independent review found no blocking defect in the v11-to-v12 changes.

Source SHA256: `e11e46e14e05d39ba4d785544053d0bc81d0e52e2bfabe6c47beb6bc3dff26ef`.

## Results

Both fresh runs completed 8,044,160 transitions (including warmup). No early stopping, resume, or additional training. These are last-100 training episode returns for seed 1.

| Configuration | Return at 2M | Return at 4M | Return at 6M | Final return |
|---|---:|---:|---:|---:|
| Prior v10 reference: minibatches, eight target refreshes | 4,073 | 5,621 | 5,845 | 5,831 |
| Full batch + one fixed target, old rates | 481 | 1,721 | 2,780 | **3,466** |
| Full batch + one fixed target, 3× rates | 309 | 1,481 | 2,364 | **2,836** |

The larger learning rates reduced final return by **18.2%** relative to the old-rate configuration. The old-rate configuration remained **40.6%** below the v10 reference but was still improving at the end of training. Both requested removals together learned more successfully than either prior separate removal; that interaction cautions against assuming their effects are additive. One seed does not establish robustness.

Each new run made exactly 2,450 model optimizer steps, 1,960 critic optimizer steps, 245 critic target refreshes, and 245 accepted actor updates. Mean actual actor KL was 0.029972 (old rates) and 0.029979 (3× rates). Thus a materially smaller realized mean KL does not explain the return gap. Accepted actor steps do not imply monotonically improving environment returns.

The higher-rate run ended with lower normalized training model loss (0.00273 versus 0.00517), despite worse return. These losses are measured on different learned policies' state distributions and do not measure Jacobian accuracy, so they cannot establish better reward-relevant modeling. They do show why training transition MSE alone is an insufficient success criterion.

**Decision:** preserve full-batch fitting and a single frozen target as the requested foundation. The joint learning-rate increase did not solve the learning problem. Prefer the old-rate invocation below for subsequent comparisons. Do not restore minibatches or repeated backups simply to recover the old score. Further progress needs a better reward-credit prediction/optimization argument; this experiment does not establish that vectorization itself causes a return gain. A matched immediate-credit or scalar control would be needed for that attribution.

V12's evaluated source remains frozen, including its experimental 3× learning-rate defaults. Reproduce the better configuration with:

```text
--model-learning-rate 0.001 --critic-learning-rate 0.0003
```

Full-batch fitting and one target refresh are defaults in either case. Gamma is 1; the KL ceiling remains 0.03. No switch to SAC/TD7 or replay was made.

Report generation: job **7548**, parallel limit 1, time limit 5 minutes, succeeded. Numerical contract job 7545 and training jobs 7546–7547 all succeeded.

Artifacts: [numeric results](latent-costate-fullbatch-v12-results.json), [learning curves](latent-costate-fullbatch-v12-results.png), and [vector figure](latent-costate-fullbatch-v12-results.svg). The numeric report includes exact run paths, counts, and matched-step measurements.
