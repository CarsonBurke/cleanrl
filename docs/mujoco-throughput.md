# Shared MuJoCo throughput tools

These tools target execution without changing the policy, objective, batch
size, rollout horizon, optimizer, precision, simulator, or seed. The frozen
VMPO v30 script is a workload reference, not a new algorithm implementation.
Native scheduling/storage changes and opt-in numerical kernel candidates are
distinct: the latter can change reduction rounding and require explicit gates.

## Final acceptance status (through job 4957)

| Path | Evidence and disposition |
| --- | --- |
| Original-engine native backend | Accepted for the measured v30 workload: full 8M run, 3.27x whole-process speedup, all 40 non-timing scalar series identical to contemporaneous sync. Preferred validated path. |
| Shared collector, asynchronous physics and packed transfers | Full 8M run matches all 40 learning metric series, but is 8.3% slower than native-only (136.65 vs 126.15 seconds). Available for explicit integration and measurement; not the recommended v30 execution path. |
| Manual whole-update CUDA capture | Rejected for production: reproducible illegal memory access when combined with compiled inference; standalone numerical tests did not establish integration safety. |
| Inference parameter cache | Experimental only, not training-enabled: approximately 1.45x N16 inference speedup, but N64 fails the unchanged bitwise decoded-value gate. |
| Fused projection/temperature and GPU physics substitution | Not training-enabled: failed optimizer or simulator parity gates; measured gains do not justify adoption. |

## Existing scripts

The launcher substitutes the Gymnasium vector-environment constructor within
the launched process. It leaves the original training file unchanged:

```bash
mlq submit --name v30_native --max-parallel-runs 1 --time-limit 2h \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -m cleanrl_utils.fast_mujoco --backend native --threads 4 \
  cleanrl/vmpo/ppo_continuous_action_iterthink_v24_beta_vmpo_v30_dreamer_bucket_moment_hlgauss_reward_norm.py \
  --env-id HalfCheetah-v4 --num-envs 64 --seed 1 \
  --total-timesteps 8000000 --exp-name v30_native
```

Use `--backend sync` for the original environment path and `--backend threaded`
for a reference implementation that steps the original wrappers on worker
threads. Thread count is a benchmark parameter, not a guarantee of speed: tiny
environments can be slower with additional workers. The ML queue's concurrency
limit and the environment's physics thread count are independent.

The launcher supports scripts that construct `gym.vector.SyncVectorEnv`; it
does not alter other vector backends or subprocesses. Constructor substitution
is not compatible with scripts that subclass that symbol or use it as an
`isinstance`/`issubclass` type; use `--backend sync` (which leaves the constructor
untouched), direct execution, or explicit shared-factory wiring for those scripts.
It sets PyTorch CPU threads to one while preserving the
trainer's numerical precision and determinism choices. All TensorBoard logs,
model outputs and training arguments remain the trainer's responsibility.

The native backend supports the installed Gymnasium 0.29.1 HalfCheetah-v4,
Hopper-v4, and Walker2d-v4 classes. It uses their original MuJoCo models, data,
reset methods and random generators. One native call advances independent
environments using MuJoCo's original `mj_step` and `mj_rnePostConstraint`,
spread across a persistent worker pool. There is no GPU simulator conversion
or fast-math. Its small C bridge builds against the installed MuJoCo wheel in
`$XDG_CACHE_HOME/cleanrl/mujoco-native` (or `~/.cache/cleanrl/mujoco-native`).
It requires a C compiler with POSIX threads.

Idle workers park on a condition variable instead of spinning. A rollout
issues one batched step per ~150us of surrounding Python work, so a spinning
team costs `num_threads` whole cores while doing nothing: measured at 4
threads and 16 envs, libgomp's default burned 1630us of process CPU per step
to back 614us of real work, and stole enough throughput from the policy thread
to cost wall time too. With three concurrent rollouts, parking holds aggregate
throughput at 161.7k SPS on 5.1 cores where an unbounded spin needs 11.9 cores
for 164.8k.

`CLEANRL_ENV_SPIN` (pause iterations before parking, default 0) trades CPU
back for wall time, and the default is deliberately conservative rather than
optimal. Parking's cost is entirely tail latency: parked workers queue behind
other runnable threads, so at 16 envs / 4 threads under load the median step
is 15-25us slower than with `CLEANRL_ENV_SPIN=5000` while the minima barely
move. Spinning wins whenever cores are genuinely idle and only turns negative
once they saturate — measured, `spin=5000` still gained 6.6% aggregate at 6
concurrent runs x 2 threads, which consumes 9.3 of 12 cores. At the
documented operating point (6 runs, `num_threads=2`) set it; leave it at 0
when the box is oversubscribed.

`num_threads` is a per-run latency knob paid for in aggregate throughput, so
it must be chosen end-to-end and never from a rollout-only benchmark. On the
real trainer at 6 concurrent runs: 1 thread gives 186k aggregate SPS, 2 gives
252k, 4 gives 209k. A rollout-chain proxy that omits the optimizer step
wrongly favours 1 thread, because per-run cost there is almost pure physics.

Supported wrappers are checked explicitly. Native execution preserves the
canonical raw wrapper stack and the standard legacy observation/reward
normalization and transformation prefix. Legacy normalizers keep their original
objects and equations; new scripts should use the faster shared vectorized
normalizers. Canonical legacy clipping/normalization stacks are batched while
retaining original RMS object identity and singleton-update arithmetic. Custom
transform callbacks remain ordered per environment; changed callbacks or
normalizer state fall back to their original wrapper behavior.
As with other parallel vector backends, environments must be
independent: a transform must not change another environment's physics state.
Unsupported wrappers, custom task implementations and MuJoCo
callbacks fail explicitly. Native mode is unsuitable for custom engine error
callbacks or malformed models: its direct C calls do not use the Python
binding's fatal-error exception conversion. Use the original backend for those
cases. The shared factory uses sync for video/rendering; the launcher requires
an explicit compatible backend when a script supplies rendering wrappers.

## Maintained and new trainers

Replace vector-environment construction with the shared factory:

```python
from cleanrl.shared.mujoco_env import make_mujoco_vector_env

envs = make_mujoco_vector_env(
    args.env_id, args.num_envs, backend="native", num_threads=4,
    capture_video=args.capture_video, run_name=run_name,
)
```

Continue using `VectorObsNorm`, `VectorRewardNorm` and `run_phase_warmup`.
Per-environment statistics, terminated-only reward returns, clipping, and
final-before-reset observation normalization remain unchanged. Both ordinary
`envs.envs[i].reset()` and `reset_at(i)` keep native episode bookkeeping correct.

For rollout storage, `RolloutTransfer` stages every per-step field on the host
and uploads them in one packed copy after the rollout. With a host actor
(`HostMLP`) nothing reaches CUDA during the rollout at all:

```python
from cleanrl.shared.host_actor import HostMLP
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.sampling import sample_beta_actions_host

actor = HostMLP(agent.actor, N)              # FP32 NumPy mirror; refresh() after each update
transfer = RolloutTransfer(T, N, obs_shape, "cuda",
                           fields={"observations": obs_shape, "native_actions": (act_dim,)})
for step in range(T):
    native, action = sample_beta_actions_host(actor(obs), low, high, rng)
    # environment step; normalization.
    transfer.push(step, normalized_reward, terminated, truncated,
                  observations=obs, native_actions=native)
batch = transfer.upload()
# batch.rewards/terminations/truncations are contiguous (T, N);
# batch.fields["observations"] / ["native_actions"] are (T, N, ...).
values, old_logprobs = rollout_statistics(batch.fields["observations"].flatten(0, 1), ...)
```

Values and old log-probabilities come from one batched device forward over the
uploaded rollout (the same network and numerics the loss uses). Measured on the
baseline at 16 envs, this removes the ~200us (idle GPU) to ~400us (GPU shared
with another mlq job) per-step device round trip; `scripts/profile_rollout_step.py`
attributes the remaining per-step cost (physics, normalization, host actor).

The helper reuses pinned host and CUDA storage and performs one packed metadata
upload per rollout. Transfers complete before source storage can be reused.
Returned observations/batches are reusable views: copy them into persistent
rollout storage before the next observation/upload call. Recurrent policies
that consume episode flags immediately must still transfer those flags each
step. Set `store_transition_observations=True` when the algorithm needs all
final transition observations on the GPU, then supply the fifth `push` argument.
`non_blocking=True` enables event-protected pinned staging slots. The default
blocking mode remains useful as a parity/performance control. Always call
`close()` before releasing storage. `transfer.observation(obs)` uploads a single
observation batch (e.g. for the tail bootstrap value); `ActionTransfer` supplies
reusable pinned action downloads for device-side policies outside the graph path.

`TruncationBootstrapCache.push_normalized(step, truncated, transition_obs)` takes
the second output of `VectorObsNorm.normalize_step` directly. It snapshots
reusable environment buffers, collects finals without Python tuple lists,
and scatters batched critic results with indexed tensor operations. `reset()`
reuses storage between rollouts. Optional `resolve(..., batch_size=N)` pads
critic calls to stable shapes; use it only for critics without batch-dependent
behavior and validate the numerical consequences of changing GEMM batch sizes.

`sample_beta_actions(alpha, beta, low, high)` in `cleanrl.shared.sampling` uses
the original PyTorch Beta sampler with local argument validation disabled.
It preserves the sampler's random stream and clamp/rescale operations. Use it
only when the policy guarantees valid positive parameters, and retain finite
action/metric checks at an existing host synchronization. It never changes
global distribution validation settings.

Use `get_gae_fn(compiled=True)` for shared GAE. Algorithms such as frozen v30
that deliberately evaluate explicit next values can instead select
`explicit_next_values=True`, preserving v30's recurrence and terminal masks.
Reusing already-computed critic values is not automatically equivalent in
BF16: changing GEMM batch shapes can change rounding. The benchmark reports
that difference separately; the launcher never enables critic reuse.

Continue using `configure_runtime`, `device_minibatches`, `gather_metrics` and
`PhaseTimer` in new implementations. Measure `env`, `rollout`, `update`, and
wall-clock interval throughput. Synchronize logging metrics together; avoid
`.item()` in optimizer loops. Small Python lists outside the hot path are not
themselves a reason to rewrite code.

### Whole-pipeline integration

Policies that cannot be mirrored on the host (autocast/bf16, large networks)
use `cleanrl.shared.rollout_graph.RolloutStepGraph`: one captured CUDA graph per
step containing the pinned observation upload, the policy callback, the scatter
of every output into `(T, N, ...)` ring-buffer storage at a device-side step
index, and the pinned action download — exactly one host synchronization per
step. Compile the callback with `graph_compile` (Inductor without CUDA-graph
trees); capture leaves the CUDA RNG stream untouched, so replays draw exactly
what the eager calls would (the v30 parity test checks this bit for bit).

`cleanrl.shared.collector.OnPolicyCollector` composes the native factory,
normalizers, the step graph and packed transfers. Its policy callback returns a
mapping containing `action` plus arbitrary tensor fields such as `value`,
`alpha`, or recurrent state. One policy version produces the complete rollout;
nothing is queued, reordered, or split. The environment is stepped on the
calling thread. Offloading it does not pay: within a step the policy needs the
observation the step produces, so there is nothing to overlap, and splitting
the envs into two staggered groups to create overlap was measured 2.3x slower
because it doubles every fixed per-call cost -- NumPy dispatch, ctypes
marshalling, Gymnasium bookkeeping and future handoff -- which is precisely
what dominates at this batch size. Recurrent reset handling remains the policy
callback's responsibility;
recurrent trainers must provide their own boundary/state wiring.

### Host policy mirror

`cleanrl.shared.host_graph.make_host_mirror(sequential, num_rows)` is the entry
point; it returns the fastest mirror available for that architecture, and every
mirror exposes the same `refresh()` / `__call__(obs)` contract, so callers never
branch on the choice.

It prefers `HostGraphActor`, which walks an integer op-graph in one native call
instead of issuing ~85 NumPy ufuncs. At these shapes the NumPy mirrors are
dispatch-bound, not arithmetic-bound: on (16, 17) inputs a ufunc costs ~0.5us of
dispatch, so `out=` reuse saves ~0.05us and cannot help. Fusing the whole
forward into one call is the only thing that does. A 16-row SiTU-sphere trunk
(width 64, 3 blocks) drops from 70.6us to 10.6us; a plain tanh MLP from 6.0us to
2.6us, at the same FP32-vs-CUDA deviation as the hand-written mirrors.

Weights are re-marshalled only in `refresh()` (once per optimizer step), never
per env step; the per-step call only re-binds input and output addresses.

When the kernel cannot express a network, the factory falls back to the matching
hand-written mirror in `cleanrl.shared.host_actor` and warns once with the
kernel's reason. The warning matters: a silent fallback costs ~6x on the policy
forward and is otherwise invisible in a run's logs. A genuinely malformed or
unsupported network still raises.

### Host Beta head

`cleanrl.shared.sampling.make_beta_sampler(num_envs, act_dim, low, high)` is the
same pattern one layer down: once the policy forward is fused, the Beta head is
~60% of what is left of the rollout's host path. `rng.beta` is untouched and
irreducible -- the RNG stream is load-bearing -- so what moves into
`host_kernel.c` is only the NumPy dispatch around it: `logaddexp`/`+= 1`/split
before, cast/clip/rescale after. Paired A/B/A at 16 envs and act_dim 6: the head
falls 13.4 -> 10.5us per step and the whole chain 24.5 -> 21.6us, after which
`rng.beta` is 77% of the remainder.

The pre-sampler op is the one place here that is libm-bound rather than
dispatch-bound: bitwise identity with `np.logaddexp` forces glibc `expf` +
`log1pf` per element (~5.8ns each), so it recovers ~1.4us rather than the ~2.5us
NumPy spends. The kernel's own polynomial `exp` is far faster and 1-3 ulp off,
which is not usable when the output seeds a random draw.

### Bitwise identity is shape-dependent

Worth knowing before batching any per-tensor reduction: a CUDA row reduction
picks its accumulation split from the shape, so `X.mean(dim=1)` on a stacked
`[n, W]` buffer is NOT guaranteed to equal `x.mean()` per row. Measured, `mean`
diverges in the last ulp at (n>=4, W=1024) and (n>=16, W=256), `std` at
(n>=6, W=512) and (n>=16, W=128), while the shapes this fork actually runs
((8, 64) and (12, 43)) are clean. Batched reductions whose result reaches the
learner therefore need a construction-time equality check against the loop they
replace, with a per-group fallback -- not a shape-independent assumption.

The accepted learner path compiles the original loss and uses ordinary fused
Adam execution. `cleanrl.shared.cuda_update.CudaGraphUpdate` remains diagnostic
code, **not a production optimization**. Although standalone tests passed
parameter/optimizer/RNG restoration and first-update parity, actual v30
integration failed with illegal memory access (4942 and 4952). A tiny regression
test also reproduces the failure when manual update capture follows compiled
inference on the same parameters (4951). Later transfer failures in that process
are consequences of the poisoned CUDA context, not evidence of separate transfer
bugs. Retaining compiled callables and explicitly owning gradient buffers did
not establish safety. Do not enable manual update capture in training.

Its standalone N16 update speedup was 1.12x, but the N64 result was 0.967x
(4941), so it also offered no measured N64 update benefit. These results do not
reject PyTorch's existing `torch.compile(..., mode="reduce-overhead")` execution;
the unsafe candidate is the additional manual whole-update capture layer.

`scripts/train_mujoco_throughput.py` integrates these utilities against v30:

```bash
mlq submit --name v30_shared_pipeline --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -u scripts/train_mujoco_throughput.py \
  --env-id HalfCheetah-v4 --num-envs 64 --seed 1 --total-timesteps 8000000 \
  --exp-name v30_shared_pipeline --compile --compile-mode reduce-overhead
```

It imports the frozen model/arguments and extracts the actual frozen loss,
recording the reference SHA256. It preserves 39-step batches, full-batch next
critic evaluation, BF16 execution, top-k ties, warmup budget, percentile scaling,
dual updates and target-promotion rules. It logs all reference loss metrics,
per-phase wall time, cumulative SPS and interval SPS. Its production path does
not enable manual update capture, inference parameter caching, or experimental
fused numerical kernels. Use `--no-async-env`, `--no-non-blocking-transfers`, or
`--env-backend sync` for controlled execution comparisons.

Job 4947 passed 23 checks: the actual frozen source versus shared collector and
ordinary compiled-loss updates, plus transfer safety. The integration fixture
uses N16, a complete 1000-step stochastic phase warmup and two 39-step rollouts.
It checks actions, CUDA RNG, normalization, storage, targets, metrics, parameters
and Adam state, including a natural truncation and nonzero critic in the second
rollout. This is fixed-work correctness coverage, not a shortened training run
or evidence of equal final learning scores.

### Experimental inference parameter cache

`InferenceParameterCache` retains explicit BF16 copies of selected matmul
operands while leaving master weights and other operations FP32. Cached tensors
need static addresses for compiled CUDA graphs; otherwise the graph wrapper
copies them every call, making the initial N16 attempt slower (0.602x).
Refreshes are explicit after master updates or target promotion. Expert biases
and residual gates remain FP32; converting the whole model to BF16 is not
equivalent. The benchmark also excludes the value-head weight to retain its
original padded GEMM layout.

After static-address correction, job 4950 measured approximately **1.45x N16
inference speedup**, including approximately 1.45x for 39 calls with cache
refresh costs included. Zero/nonzero-head and master-update/refresh cases passed
bitwise output, sampled-action and RNG checks at N16. **N64 failed** the unchanged
decoded-value gate: maximum absolute difference `0.0009765625`, while policy
parameters, sampled actions and RNG matched exactly. No N64 timing or training
acceptance follows from that failed test.

Generated-code inspection found different value-head GEMM layouts in the first
candidate. Excluding that head restored all 45 GEMM operand/output layouts;
remaining Triton arithmetic matched, but independent autotuning selected
different softmax/value-reduction block sizes and warp counts. Those choices
can change FP32 reduction order. They are a concrete explanation candidate for
the residual mismatch, not proof that every intermediate logit is identical.
The strict bitwise gate was not relaxed. Policy-only caching (4953) measured
1.19x at N16 but also failed N64 decoded-value parity despite leaving critic
parameters uncached. Neither cache mode is enabled in the training proxy.

The fixed-work benchmark remains available for reproducibility, not as a
recommendation to repeat the failed experiments:

```bash
.venv/bin/python scripts/benchmark_inference_cache.py --help
# GPU execution, if explicitly revisited, must be submitted through mlq.
# --no-cache-critic selects the measured policy-only experiment.
```

Its JSON and TensorBoard reports separate compilation, inference, refresh and
39-call amortized timing. These are inference measurements, not training SPS.

Two additional kernels are **benchmark-only rejected candidates**:

- Shared HL-Gauss moment-matching projection, retaining
  all 32 bisection iterations, original log-mass cutoff and tilt bound.
- Shared V-MPO temperature solver, retaining all 32
  geometric bisection iterations and original KL expression. Its adapter
  validates the exact reference-loop AST fingerprint before substitution.

These fuse dispatch-heavy iterative reductions into Triton kernels using
precise exponential/division operations and disabled multiply-add contraction.
They are not automatically bitwise identical: labels, decoded moments, KL,
weights, effective sample size, gradients and first optimizer updates must be
checked before full learning comparisons. They do not justify changing an
algorithm's iteration count, tolerance, precision, or batch size.

Job 4934 rejected both fused-update variants at N=16: maximum first-update
parameter differences were 4.16e-6 (projection) and 7.65e-5 (projection plus
temperature), beyond the recorded tolerance. Their measured update speedups
were only 1.005x and 1.019x. The training proxy therefore no longer exposes
these options, and the dependent full fused run was skipped. Experimental
investigation remains reproducible with benchmark `--fused-updates`; the
optional update benchmark compares only original and captured execution unless
`--fused-updates` is requested. The later integration failures above override
the earlier standalone captured-update success as an adoption decision.

Compiler artifacts default to `$XDG_CACHE_HOME/cleanrl/{torchinductor,triton}`
(or `~/.cache/cleanrl/...`), respecting explicit cache environment variables.
Completed timing events are recycled instead of allocated on every interval.

### GPU physics audit

`scripts/audit_mujoco_warp.py` exports 1000-step original-v4 fixtures for all
three tasks at N=16/64, then audits modern native MuJoCo and CUDA MJWarp against
the same states/actions. It records model/version drift, reward and observation
errors, boundary disagreements, contact/constraint overflow, compilation time
and repeated CUDA-graph resident replay throughput in JSON and TensorBoard.
Checksums bind the exported model and trajectory to the audit.

The completed [audit JSON](../artifacts/mujoco-warp-audit-v1/audit.json) compares
240,000 reference transitions on an NVIDIA GeForce RTX 5090, with seed 1 and
three GPU timing repetitions per configuration. Every configuration completed
without contact/constraint overflow or nonfinite physics outputs. Rates below
are environment transitions per second; each transition includes the original
four or five MuJoCo integration steps.

| Environment | Parallel environments | MJWarp GPU resident replay | Modern native CPU physics |
| --- | ---: | ---: | ---: |
| HalfCheetah-v4 | 16 | 13,773 | 60,499 |
| HalfCheetah-v4 | 64 | 47,193 | 59,957 |
| Hopper-v4 | 16 | 5,240 | 25,731 |
| Hopper-v4 | 64 | 18,611 | 25,768 |
| Walker2d-v4 | 16 | 4,372 | 18,887 |
| Walker2d-v4 | 64 | 15,028 | 18,631 |

At these batch sizes, GPU resident replay was slower than sequential native
physics in this audit. GPU timing includes a resident fixture-load kernel and
CUDA-graph physics; CPU timing excludes fixture restoration. Both exclude
policy inference, learning, normalization and reset sampling. These are
component measurements, not training speedups or comparisons against the
optimized native parallel backend. Larger batches remain unmeasured and would
require separate algorithm-preserving integration and benchmarking.

Strict numerical parity failed on all six configurations. The largest GPU
observation differences were 0.00363 for HalfCheetah, 0.32010 for Hopper and
0.26802 for Walker2d. Hopper also changed one termination among 16,000
transitions and two among 64,000. Modern native CPU replay showed the same
Hopper termination-disagreement counts and similar state errors, suggesting
that the engine upgrade accounts for much of Hopper's drift. This is a concrete
semantic difference, not merely a strict floating-point tolerance failure.

The original fixtures use MuJoCo 2.3.3 and Gymnasium 0.29.1. The isolated audit
environment at `~/.cache/cleanrl/mjwarp-eval-v1` uses MuJoCo/MJWarp 3.12.0,
Warp 1.17.0, NumPy 1.26.4 and TensorBoard 2.20.0. MJWarp integrates in float32
and raises the solver tolerance to approximately `1e-6`; the original engine
uses float64. Its warmstart behavior can also differ; see the
[official numerical-differences documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html).
GPU stepping is not enabled in training, which continues to use the original
CPU MuJoCo engine and existing dependency versions.

The fixtures use stochastic actions rather than a trained locomotion policy.
Each replay starts from an original state, so the audit does not measure
accumulated trajectory drift or learning scores. Reward comparisons reconstruct
the original float64 Gym equations using the source pre-step position; they
include input-casting effects and do not validate a separate GPU reward kernel.
The measured state and termination differences already prevent treating this
backend as an equivalent replacement.

## Reproducible measurements

```bash
mlq submit --name mujoco_throughput --max-parallel-runs 1 --time-limit 1h \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python scripts/benchmark_mujoco_throughput.py \
  --num-envs 16 64 --backends sync threaded native --num-threads 4 \
  --thread-counts 1 2 4 8 --profile
```

Results go to `runs/{env}__{exp_name}__1__{timestamp}/benchmark.json` and local
TensorBoard so the existing harness can discover them. The benchmark records
versions, configuration, compilation startup, repeated wall-time samples,
CUDA-stream elapsed time, raw/normalized physics, transfers, sampling,
compiled v30 inference and critic evaluation, and closed-loop rollout rates.
Optional traces expose CPU dispatch and CUDA kernels. Update profiling uses
the frozen v30 loss on fixed data; it is a compute measurement, not a training
result. Full-horizon environment parity precedes performance measurements.
`--profile-update --require-numerical-parity` additionally reproduces the
standalone update comparison; its gate does not check compiled/manual-graph
interoperability and must not be interpreted as production capture approval.

CUDA-stream elapsed time includes host launch starvation and should not be
interpreted as summed kernel execution time. Closed-loop rollout speed excludes
learner updates. Neither component timings nor initial-policy physics costs
establish end-to-end training speed or learning equivalence. Compare complete
8-million-transition seed-1 runs with identical arguments, and keep compilation
startup separate from steady-state throughput.

The original shared suite passed 120 tests, including real native physics,
normalization, CUDA transfers and sampling. The expanded suite passed 79 tests,
including compiled AdamW/clipping and CUDA RNG/module-buffer restoration.
Detached parameter snapshots fixed an early test-fixture error, but later
compiled-inference/manual-capture regressions still failed. Passing the earlier
suite does not supersede those failures. The collector-only integrated path
subsequently passed 23 checks (4947).

### Measured native-backend result

The unchanged v30 workload, HalfCheetah-v4, N=64, seed 1, BF16 and compiled
reduce-overhead execution completed its full 8M budget with:

| Backend | Queue attempt wall time | Cumulative SPS near 8M | Final 100-episode return |
| --- | ---: | ---: | ---: |
| Sync (4933) | 412.45 s | 19,671 | 5,651.8 |
| Native, 4 threads (4935) | 126.15 s | 65,826 | 5,651.8 |
| Native + shared collector (4936) | 136.65 s | 60,707 | 5,651.8 |

This is **3.27x faster by whole-process wall time**, or 3.35x by the logged
cumulative training rate near 8M. All **40 non-timing scalar series** match
exactly across both complete runs, including individual episodic returns,
losses, duals and target-promotion metrics—not merely their final means.
The collector-only proxy subsequently matched all 40 shared non-timing series
as well: 27,826 scalar events, with identical steps and values, including 7,871
episodes. It was **8.3% slower than native-only** by whole-process time. Keep
native-only as the recommended path; a faster packed-transfer component does
not imply a faster complete collector. Neither result validates the rejected
capture, inference-cache or fused-kernel candidates.

A pre-existing run, `vmpo_v30_dreamer_moment_hlgauss_reward_norm`, already had
the same recorded hyperparameters (except experiment name). Repeating the
unchanged baseline was avoidable and will not be repeated. Its historical
final return was 7,744.3 versus 5,651.8 in both current executions. Recorded
hyperparameters alone do not identify the cause of that historical difference;
do not attribute it to the native backend, whose contemporaneous scalar traces
are identical to sync. Reuse historical learning evidence and fixed-work parity
tests before scheduling further full-run validation.

Recorded queue outcomes are:

| Job | Work | Final recorded outcome |
| --- | --- | --- |
| 4889 | Shared correctness suite | Passed, 120 tests |
| 4907–4908 | Original fixtures and modern-native/MJWarp audit | Completed; GPU simulator parity rejected |
| 4931 | Expanded correctness suite | Passed, 79 tests; later interoperability regression not covered |
| 4933 / 4935 | Full 8M sync / native v30 | Completed; identical non-timing scalar series |
| 4934 | N16 profiling and strict fused-update gates | Failed fused-update parameter parity; evidence retained |
| 4936 | Full shared-collector proxy | Completed; all 40 metric series identical, 8.3% slower than native-only |
| 4937 | Full fused-kernel proxy | Skipped after prerequisite failure; obsolete flags, not reusable |
| 4941 | N64 profiling and standalone update gates | Completed; capture 0.967x, eight threads fastest in measured environment sweeps |
| 4942–4943 | Captured v30 integration and blocking diagnostics | Failed with illegal memory access |
| 4944 | Sanitizer command preparation | Help-only invocation; not a correctness or sanitizer result |
| 4945 / 4948 | Sanitizer investigations | Cancelled; not successful checks |
| 4946 / 4949 | Cache and standalone update unit tests | Passed; not integrated v30 cache/capture acceptance |
| 4947 | Non-manual-capture v30 collector integration and transfers | Passed, 23 checks |
| 4950 | Full inference-cache benchmark and corrected retries | N16 faster; each attempt rejected by strict N64 value parity |
| 4951 | Tiny compiled-inference/manual-update interoperability regression | Reproducible illegal memory access |
| 4952 | Owned-gradient capture integration experiment | Failed with illegal memory access |
| 4953 | Policy-only inference-cache benchmark | N16 1.19x; failed strict N64 value parity |

All jobs declare `--max-parallel-runs 1`, normal priority, and no automatic retries.
Failed correctness jobs are manually retried only after fixing the failure.
There is no learning-based autocull in this controlled execution comparison.
Failures prevent dependent work from starting. Queue state is authoritative;
the table records this investigation's checkpoint, not future job outcomes.
Final supported-path validation **4955 passed 190 tests** in 23.46 seconds,
covering actual v30 integration, native physics/normalization, shared rollout
helpers, CUDA transfer/sampling and metadata tools. Its concurrency limit was
one, normal priority, with a 30-minute time limit. Isolated experimental
regression job **4956 passed 12 tests with one strict expected failure** in
7.58 seconds (concurrency one, five-minute limit). The expected failure is the
known compiled-peer/manual-capture illegal memory access, contained in a child
process; unrelated errors fail normally. It remains a rejected training path.

Completion hook **4957 succeeded** and includes all four 8M trainers: 4933,
4935, 4936 and 4937. It ran after their terminal states, reporting final-100-episode means,
matched-step returns at 1M/2M/4M/8M and cumulative SPS, and includes interval
SPS where available. It has concurrency 1, normal priority, one attempt and a
5-minute time limit. Read `mlq logs 4957` for its persistent score table.
It also includes the historical v30 result and explicitly reports queue
states/reasons via `score_runs.py --jobs`; missing patterns or empty runs are
warned about rather than silently omitted. Hook 4940 failed because its job
environment omitted `XDG_RUNTIME_DIR`, making its nested `mlq` client look for
the daemon socket at the wrong path. Hook 4957 explicitly inherits that variable;
only scoring was rerun, never training. Job 4936 used the ordinary compiled-loss
production path. Job 4937 was skipped after the fused
numerical gate failed; it will not be resubmitted with relaxed tolerances.
Failed, skipped or incomplete jobs cannot produce complete 8M results; absent
run data cannot produce a score. The current daemon marks unsatisfied
dependent jobs skipped. Earlier chains (4893–4897, 4912–4916, 4930, 4932) were
superseded after correctness failures, not silently counted as completed runs.

The final combined local CPU-only selection passed **93 tests** (23 CUDA cases
deselected). These checks cover AST substitution, asynchronous
ordering, transfer ownership, launcher dispatch, audit task arithmetic/options,
numerical-gate persistence/failure behavior, and timing-event reuse.
The N64 thread sweep completed: eight threads were fastest for raw,
vector-normalized and legacy-normalized environment workloads. This is not an
8M result for eight threads; the full native result used four. Collector-only
fixed-work equivalence passed; manual capture and inference caching did not
pass their complete acceptance checks. No additional profiling is required to
state these dispositions. Reuse completed learning evidence before scheduling
any further full-run validation.

For future jobs, attach a reusable score/status hook rather than relying on
the investigation's hardcoded hook 4957. Replace both placeholders:

```bash
MLQ_JOB_ID=1234                 # replace with the submitted job ID
MLQ_EXP_NAME=my_experiment      # replace with its exact experiment name
mlq submit --name "${MLQ_EXP_NAME}_scores" --max-parallel-runs 1 \
  --time-limit 5m --after-terminal "$MLQ_JOB_ID" --cwd "$PWD" \
  --inherit-env XDG_RUNTIME_DIR -- \
  .venv/bin/python scripts/score_runs.py "$MLQ_EXP_NAME" \
  --jobs "$MLQ_JOB_ID" --env HalfCheetah-v4 --last 100 \
  --at 1M,2M,4M,8M --metrics charts/SPS,charts/interval_SPS
```

An after-terminal hook reports failures and missing data as well as completed
scores. It does not convert a failed or partial run into an 8M result.

After completion, compare the training results with the existing run reader:

```bash
.venv/bin/python scripts/score_runs.py v30_shared \
  --env HalfCheetah-v4 --at 1M,2M,4M,8M --metrics charts/SPS
```

Use `mlq show JOB_ID` for execution state and wall time. The reported training
SPS includes initial warmup and compilation; component benchmark results report
startup separately. A requested 8M budget stops at the same last complete
rollout in both original v30 jobs.
