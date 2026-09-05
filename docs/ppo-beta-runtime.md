# Base PPO: Beta actions and shared runtime

`cleanrl/ppo_continuous_action.py` now uses a bounded Beta actor and the
validated shared execution tools directly; no launcher is necessary.

The actor retains two 64-unit tanh layers. Its output is two parameters per
action dimension, `alpha, beta = 1 + softplus(head)`. Samples are generated
with PyTorch's Beta sampler and mapped to each finite Box action interval.
PPO stores native samples rather than inverse-transforming physical actions.
Log densities subtract the log action-range Jacobian; entropy adds it.
Sampling remains outside compiled graphs, and parameter checks are disabled
only locally because the head enforces positivity. Nonfinite actions/losses
fail at existing host synchronization points.

This is an intentional policy change. Old Gaussian checkpoints do not load
into this Beta actor, and historical Gaussian scores do not describe this
implementation. New checkpoints retain the bare-state-dict format and the
`Agent(envs)` / `get_action_and_value` evaluation API. The standalone evaluator
requires an explicit local checkpoint instead of downloading Gaussian weights.

## Enabled execution improvements

- Original-engine native stepping for HalfCheetah/Hopper/Walker2d-v4 with
  Gymnasium 0.29.1. `--env-backend auto` selects sync for other environments;
  rendering also uses sync. Physics threads are capped at the environment count.
- Shared vector observation/reward normalization, preserving independent
  statistics, terminated-only reward returns, clipping and final-before-reset order.
- Persistent pinned transfers, one packed reward/flag upload per rollout,
  and batched final-observation critic evaluation for timeout bootstraps.
- Compiled deterministic policy statistics, log-probabilities, PPO loss and GAE;
  fused Adam; GPU minibatch permutations with a separate RNG generator.
- On-device explained variance and clip-fraction accumulation, one scalar
  metric transfer per rollout, and phase/interval timing.
- Standard stochastic phase warmup for parallel environments, charged against
  the step budget. Single-environment defaults do not warm up. Use
  `--no-staggered-starts` to retain unstaggered parallel starts explicitly.

The defaults remain one environment, 2,048 steps per rollout, 32 minibatches,
10 epochs, learning rate 3e-4 and gradient clipping 0.5. The implementation
keeps FP32 arithmetic without silently enabling BF16 or TF32. CUDA is required.
Changing minibatch RNG implementations changes seeded permutation sequences;
neither compiled arithmetic nor this new policy promises bitwise agreement
with historical Gaussian training trajectories.

The optional `target_kl` stop still uses the final minibatch's KL after each
complete epoch. Its necessary host synchronization is retained. Rendering,
tracking, save/evaluation and optional upload remain available.

Manual whole-update capture, inference-weight caching, experimental fused
numerical kernels, GPU physics and asynchronous environment workers are not
enabled. Those paths were unsafe, numerically incompatible or slower in the
earlier audit. `--non-blocking-transfers` remains an explicit measurement option;
blocking pinned observation staging is the default.

## Run and validate

Submit training through the shared queue, for example:

```bash
mlq submit --name ppo_beta --max-parallel-runs 1 --time-limit 2h \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -u cleanrl/ppo_continuous_action.py \
  --env-id HalfCheetah-v4 --seed 1 --total-timesteps 8000000 --exp-name ppo_beta
```

`--num-envs 16 --num-steps 128` is an explicit parallel alternative with the
same 2,048-transition batch size, but changes temporal rollout geometry. It is
not silently made the baseline default. Use `--env-backend sync` or
`--no-compile` for controlled execution comparisons.

`tests/test_ppo_base_runtime.py` checks the Beta density/Jacobian, bounds, RNG,
checkpoint API, independent clipped PPO loss/gradients/Adam, and evaluation
resource cleanup. CUDA cases must run through `mlq`.
`scripts/benchmark_ppo_base.py` measures repeated fixed-work inference and
minibatch updates and persists numerical gates, JSON and TensorBoard metrics.
It is not a learning experiment and does not establish a benchmark score.

## Measured results

Validation job **4969** passed all **150 tests**, including native physics,
normalization, transfers, Beta loss/gradient/optimizer parity and repeated
interoperation of the production compiled graphs. A separate CPU-only base
configuration/runtime check passed 35 tests (overlapping that suite).

Production fixed-work benchmark **4973** completed on an RTX 5090 with PyTorch
2.12.0+cu130, FP32/highest precision and TF32 disabled. All numerical gates
passed, including three matched optimizer updates with gradient clipping and
Adam-state comparisons. Median steady-state wall-clock measurements:

| Component | Shape | Speedup versus eager |
| --- | --- | --- |
| Deterministic policy/value inference | 1 / 16 / 64 environments | 1.09x / 1.13x / 1.45x |
| Full PPO minibatch update, compiled loss + fused Adam | 64 / 256 samples | 2.69x / 2.76x |
| Shared compiled GAE versus the original public loop | 2,048 steps x 1 environment | 359.56x (82.81 ms to 0.23 ms) |
| Shared compiled GAE versus the original public loop | 128 steps x 16 environments | 104.69x (5.51 ms to 0.053 ms) |

Inference excludes sampling and physics. Update timings include loss, backward,
gradient clipping and the optimizer. These component speedups do not multiply
and do not establish total training throughput or learning-score equivalence.
No new learning run was launched for this change. The earlier 3.27x v30 result
is not a measurement of this Beta trainer.

Compilation startup is significant: first-call GAE compilation took **82.8 s**
for the default 2,048-step rollout and **5.0 s** for the 128-step rollout.
These costs are excluded from steady-state timings. The compiler-CUDA-graphs
disabled ablation (**4972**) also passed its numerical gates, but inference
was slower than eager; it is not the enabled production configuration.

Jobs 4969, 4972 and 4973 all succeeded with concurrency one, normal priority
and a 30-minute limit. Benchmark 4970 was cancelled before starting to include
the single-environment default in its replacement. Full timings, numerical
errors, tolerances and source hashes are recorded in:

- `runs/HalfCheetah-v4__ppo_base_beta_components_graphs_v2__1__1788631158/benchmark.json`
- `runs/HalfCheetah-v4__ppo_base_beta_components_v1__1__1788591846/benchmark.json`
