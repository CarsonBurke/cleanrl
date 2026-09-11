# Base PPO: Beta actions and shared runtime

`cleanrl/ppo_continuous_action.py` now uses a bounded Beta actor and the
validated shared execution tools directly; no launcher is necessary.

The actor retains two 64-unit tanh layers. Its output is two parameters per
action dimension, `alpha, beta = 1 + softplus(head)`. Training uses an FP32
host actor mirror and NumPy's Beta sampler; public CUDA evaluation uses
PyTorch's Beta sampler. Both map samples to each finite Box action interval.
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
  rendering also uses sync. Physics threads default to two and are capped at
  the environment count. Native workers also assemble the raw observations.
- Shared vector observation/reward normalization, preserving independent
  statistics, terminated-only reward returns, clipping and final-before-reset order.
- `make_host_mirror` and `make_beta_sampler` select the shared fused host
  implementations. Stage their borrowed output buffers before the next call.
  Refresh actor weights once per rollout, after the preceding learner update.
- Persistent pinned transfers, one packed rollout upload, and batched
  final-observation critic evaluation for timeout bootstraps.
- Compiled deterministic policy statistics, log-probabilities and PPO loss;
  fused Adam; GPU minibatch permutations with a separate RNG generator.
- `get_gae_fn(compiled=True)` uses a bounded-code CUDA recurrence for contiguous
  FP32 targets. It unrolls only 32 steps at a time and preserves the backwards
  recurrence, separate termination/truncation rules and FP32 multiplication
  order. Other dtypes/layouts and requested autograd retain the reference
  compiled PyTorch implementation; `compiled=False` remains the eager reference.
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

Manual whole-update capture, inference-weight caching, the rejected
projection/temperature kernels, GPU physics and asynchronous environment
workers are not enabled. Those paths were unsafe, numerically incompatible
or slower in the earlier audit. `--non-blocking-transfers` remains an explicit
measurement option; blocking pinned observation staging is the default.

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

For the standard N16 / 8M / seed-1 workload, the repository submission command
sets the queue budget and environment explicitly:

```bash
.venv/bin/python scripts/submit_mujoco.py --name ppo_beta
.venv/bin/python scripts/submit_mujoco.py --dry-run --name ppo_beta
```

It submits base PPO with two physics threads, OMP/MKL threads one, one Inductor
compiler worker, spin budget 5000, concurrency six, and a two-hour limit. This is an
explicit experiment geometry, not a change to the trainer's one-env default.
Novel scripts, nonstandard geometry, or extra trainer options default to
concurrency one and parking instead. Override resource settings only after
measuring them. `--script`, `--num-envs`, `--env-threads`, `--max-parallel-runs`,
`--compile-threads`, `--env-spin`, `--time-limit` and repeatable `--after-success` are submission
options; algorithm-specific trainer arguments follow `--`. No W&B, automatic
retries, priority increases, background children, or learning-based culling
are injected.

Shared runtime initialization also defaults `TORCHINDUCTOR_COMPILE_THREADS` to
one, respecting an explicit environment override. Compiler workers are separate
from Torch intra-op/physics threads. A measured six-process simultaneous cold
baseline launch exhausted system RAM; steady-state concurrency is not a safe
cold-compilation budget. The full-run comparison below warms each version
serially before measuring six concurrent runs and has a host-memory reserve guard.

Future trainers should use these shared factories rather than copy their
implementations. Shared internal improvements then apply to new processes
without editing frozen versioned scripts. The maintained base trainer is the
reference for standard PPO integration; algorithm-specific losses remain local.

`tests/test_ppo_base_runtime.py` checks the Beta density/Jacobian, bounds, RNG,
checkpoint API, independent clipped PPO loss/gradients/Adam, and evaluation
resource cleanup. CUDA cases must run through `mlq`.
`scripts/benchmark_ppo_base.py` measures repeated fixed-work inference and
minibatch updates and persists numerical gates, JSON and TensorBoard metrics.
It is not a learning experiment and does not establish a benchmark score.

## Historical baseline measurements

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

## Accepted shared execution measurements

The current task retained native observation assembly, fused host factories,
bounded-code GAE and conservative compiler scheduling. The PPO loss, minibatch
sequence, optimizer updates, precision and clipping remain unchanged.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full cold 8M run, one process | 512.04 s | 141.15 s |
| Six complete warm-cache 8M runs, group wall time | 200.71 s | 186.70 s |
| Six-run aggregate transitions/s | 238,510 | 256,411 |
| Standard GAE, T2048/N16 cold first call (separate component job) | 203.46 s | 5.11 s |
| Standard GAE, T2048/N16 steady call | 201.69 us | 83.52 us |

Full-run job **5328**, successful second attempt: HalfCheetah-v4, seed 1,
N16/T2048, 32 minibatches, ten epochs, two physics threads, spin 5000,
FP32/highest and TF32 disabled. Each version first completes one serial cold
8M run in its own empty caches, then six complete runs sharing only that
version's warmed cache. Compiler workers are one for both versions. All 14 runs
complete 243 planned iterations at 7,978,624 transitions, including 16,000
warmup transitions; the remainder cannot fill another rollout.

Aggregate whole-process throughput improved **7.5%**; the single cold run was
**3.63x** faster. These are different measurements: cumulative group throughput
includes process startup, while interval SPS excludes some startup/logging.
The mean per-run steady interval rates in the concurrent comparison were
roughly 44.9k before and 44.4k after; do not call the 7.5% gain a steady-learner
speedup. Six identical seeds measure concurrency, not six independent learning
replications. Final-100 returns were 6101.4 before and 7252.3 after. Host mirror
FP32 rounding changes trajectories; this is not bitwise learning equivalence
or evidence of a statistically established return improvement.

GAE job **5244** passed all six old-compiled versus new target comparisons
**bitwise** (both recurrence contracts, T2048/N1, T2048/N16 and T128/N16).
Cold first-call reductions ranged from 6.5x to 57.1x; steady reductions from
1.23x to 5.16x. The component job used fresh processes and empty caches.

Environment job **5245**, successful second attempt, matched both saved native
and original Gym outputs/RNG bitwise across all three tasks. Seven alternating
timing repetitions measured about 0.9% HalfCheetah and 2.0% Hopper gains;
Walker2d was effectively unchanged (0.2% slower by median). These small
component differences are not claimed as general end-to-end speedups.
Its first attempt failed because the benchmark reused lifetime episode
counters against a fresh oracle; the fixture now constructs a fresh pair.

Two learner candidates were rejected, not promoted: indexed gathers (**5246**)
and sparse diagnostics (**5326**). Both passed numerical checks but failed
the predeclared 3%/consistent-paired-repeat performance gate. Sparse diagnostics
regressed the batch-256 case. The original compiled loss and fused Adam remain;
no unused shared learner helper or experimental reporting API is installed.

Correctness job **5243** passed 184 tests, including candidate-independent
native/GAE/host integration contracts; **5327** passed 38 checks of the later
rejected learner experiment. Initial aggregate attempt **5328/1** suffered
SIGKILL during six simultaneous cold compilations with system-wide OOM evidence.
It is not a completed run or performance result. The corrected harness uses
serial cache warmup, bounded compiler workers, and an 8 GiB host-memory reserve.
All these jobs use queue concurrency one, normal priority and no automatic
retries. Time limits are 30 minutes except GAE and aggregate jobs at one hour.

Evidence and reproducible source snapshots:

- [Full-run JSON](../runs/HalfCheetah-v4__ppo_execution_full_aggregate_v1_1788833694__1__1788833694/benchmark.json)
- [GAE JSON](../runs/GAE__gae_execution__1__1788829606083104483/benchmark.json)
- [Environment JSON](../runs/env_execution__1788831858120036503/benchmark.json)
- [Indexed learner rejection](../runs/HalfCheetah-v4__ppo_indexed_loss_execution__1__1788830289261592462/benchmark.json)
- [Sparse learner rejection](../runs/HalfCheetah-v4__ppo_sparse_diagnostics_v2__1__1788832592987690707/benchmark.json)
- [Pre-change source ZIP](../runs/ppo_execution_reference_v1.zip)
- [Measured candidate source ZIP](../runs/ppo_execution_candidate_v1.zip)
- [Rejected learner source ZIP](../runs/ppo_rejected_learner_candidates_v2.zip)

To reproduce, extract the source ZIPs into separate directories, give each a
`runs` directory (or a symlink into this checkout's `runs` for UI discovery), and
submit `scripts/benchmark_ppo_execution.py --baseline-root <before>
--candidate-root <after> --runs 6` through `mlq` with concurrency one. The
harness uses this checkout's `.venv/bin/python`, persists source fingerprints,
requires complete rollouts, and rejects source changes during execution.

Final retained-source verification **5347 passed 181 tests in 76.38 seconds**
(concurrency one, normal priority, 30-minute limit). It covers the installed
native/GAE/host paths, base interfaces and submission resource guards after
removing the rejected learner experiments and adding bounded compiler workers.
The two independent source reviews found no remaining concrete defect after
fixing the benchmark cache-state confound and reviewing the memory-safe protocol.
