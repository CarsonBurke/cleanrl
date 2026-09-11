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

### V-MPO short-rollout bootstrap transfer correction

The 64×39 V-MPO reference exposed small blocking uploads that long-rollout
PERI timing obscured. `TruncationBootstrapCache.resolve_with_tail` packs tail
observations, final observations and scatter indices into a reusable pinned
block, uploads its active prefix asynchronously, and preserves the original
critic batch layouts. Host reuse waits for the preceding DMA event. Consume
returned tensors on the calling stream before the next cache call; `close()`
waits for staging DMA, not arbitrary downstream work on another stream.

MLQ **5605** passed 93 tests covering packed bootstrap values, GAE compatibility,
repeated reuse/growth and compiled output lifetime alongside existing shared
contracts. MLQ **5607** preserved model/PopArt/dual state bitwise after 110
fixed-rollout updates and reduced synchronized return-plus-update time from
**1.170 s to 0.366 s** per 100 timed iterations. Moving the graph boundary alone
did not help. These totals include optimizer work; individual host attribution
times are not isolated GPU-kernel timings.

Real-trainer median interval throughput over 1M–8M increased from **48.9k SPS**
(v62 normalized) to **75.7k SPS** (v63 raw), with return generation falling from
**16.17 to 2.91 ms/update**. Reward settings differ; use the fixed-rollout gate
for controlled runtime evidence, not this comparison for an algorithm claim.
The faster covariance-dual algorithm was subsequently set aside in favor of
v60. v64 carries only the proven runtime changes and an optional reward-
normalization ablation onto that selected baseline; see `cleanrl/vmpo/FAMILY.md`
for job IDs and learning outcomes. All cited profiling/validation jobs declare
parallel limit one and a 20-minute runtime limit.

### Collective-control v8 execution refactor

`cleanrl/collective_control/control.py` uses compiled FP32 recurrence inside the
shared `RolloutStepGraph`: one captured observation upload, recurrent update,
action download and host wait per step. Source addresses are decoded on genome
reload, not per observation. Fixed node capacity permits insertion/deletion
without recapture; reset restores recurrent state after capture and between
episodes. Public `action()` returns owned storage; the evaluator consumes the
borrowed graph buffer before the next step.

`TeamEvaluator` owns native environments and controllers for each
`(team_count, seed_count)` shape. Training reuses them across generations,
encodes shared genomes once per reload, and uses shallow team snapshots.
Mutation candidates and saved champions still own independent genome copies.
The proposal incumbent is evaluated in the candidate batch. Confirmation is
still sequential by victim: accepted replacements affect later confirmations.
No horizons, episode counts, mutation draws, acceptance thresholds or checkpoint
schemas were changed. Compiled arithmetic can change FP32 rounding.

MLQ **6053**, RTX 5090, HalfCheetah-v4, seed 1, eight physics threads:
16 residents, initial 32-node and heterogeneously mutated genomes, 128-node
capacity, 1,000-step horizons. Four paired calls per case alternate evaluation
order and genome payloads. Below are baseline/reused-evaluator median times,
excluding the first cold call:

| Batch | Initial genomes | Mutated genomes | Speedup |
| --- | --- | --- | --- |
| Proposal: 129 teams × 1 episode | 1025.5 → 477.0 ms | 958.7 → 440.1 ms | 2.15–2.18× |
| Confirmation: 5 teams × 4 episodes | 520.3 → 143.7 ms | 538.0 → 130.6 ms | 3.62–4.12× |
| Development: 1 team × 16 episodes | 488.5 → 119.9 ms | 453.0 → 111.0 ms | 4.07–4.08× |

Public controller action latency improved 6.56–9.62×. Shared-observation
32-step traces differed by at most `1.08e-7`; paired full-horizon episode
returns differed by at most `2.88e-6`. These are execution benchmarks, not a
training-quality result or a claim of bitwise-identical evolutionary trajectories.
Cold controller compilation/capture is reported separately by the benchmark.
MLQ **6054** passed six tests, including independent recurrent-state equations,
uniform/evolved authority, reset/remapping, genome growth/shrink, action-buffer
ownership and sequential acceptance of complementary candidates.

Before editing a future implementation, retain its `control.py` and `genome.py`
in a reference directory, then compare with:

```bash
mlq submit --name collective-v8-benchmark --max-parallel-runs 1 --time-limit 20m \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python scripts/benchmark_collective_control.py \
  --baseline-path /path/to/reference --seed 1 --threads 8 \
  --stage all --repeats 4 --output /tmp/collective-v8-benchmark.json
```

Both reported jobs used `maxParallelRuns=1`; the regression job had a
15-minute limit. No new training run was launched for this refactor.

### Collective-control v9: typed wiring and paired evolutionary selection

The learning audit found two source-address bugs, not a need for a prediction
head. Observation and previous-proposal indices were accidentally remapped
through node IDs; deleting a disconnected node could flip an action from
`+tanh(2)` to `-tanh(2)`. Only previous-proposal channel zero could be generated.
V9 uses node-ID lookup only for node sources and generates all action channels.
The recurrent feedback remains each resident's own normalized proposal, not the
collective executed action. A collective with zero total evolved authority now
has the defined neutral action-space midpoint rather than an undefined quotient.

Search remains gradient-free episodic policy evolution:

- Local mutation clones the resident being replaced. The separate
  `transplant_probability` operator (default 0.1) copies a distinct resident
  without simultaneous mutation. Operator proposals, acceptance and gains are
  logged separately; singleton populations always use local mutation.
- Proposals nominate a shortlist; screening selects its winner. Only an
  independent, full-horizon, 16-episode paired validation decides replacement.
  Its one-sided Student-t lower bound must exceed zero, at alpha
  `0.05 / residents`. Later resident decisions see prior accepted replacements
  but receive fresh screening and validation seeds.
- The Student-t bound assumes approximately normal independent paired
  differences. Bonferroni controls the per-generation family under those
  assumptions, not the entire adaptive run; it is not a distribution-free
  guarantee.
- Development uses a fixed 64-episode suite. Champions therefore compete on
  the same cases. Final test and diagnostic suites are separate and independent
  of checkpoint generation. Schema3 records seed lists and typed-source
  semantics; evaluating older checkpoints explicitly reports reinterpretation,
  not exact reproduction of the broken decoder.
- `total_transitions` counts actual stepped vector slots, including inactive
  lanes still simulated, and includes final development. Calibration is
  excluded and its upper bound recorded separately. Budget/time stops finish
  the current generation and always score/save its final population. Thus a
  transition target may overshoot by one generation plus final development.
- Plateau culling tracks both raw material progress and development EMA
  (decay 0.8, delta 0.01): 20 warmup evaluations, then 20 stale evaluations.
  Either signal resets patience; `plateau_patience=0` disables it. Cull state is
  checkpointed; a completed cull emits `AUTOCULL` and exits successfully (0).
  It can stop an architecture before the common transition target.

MLQ **6072** passed **23** focused tests, including typed deletion invariance,
all proposal channels, recurrent reset/reload, neutral zero authority,
independent winner validation, rejection of harm/uncertainty/ties, complementary
sequential improvements, pure transplantation, fixed evaluation suites, and
forced final scoring at budget/plateau stops. Independent static review found
no material correctness defect. The module CLI no longer eagerly imports its
own entrypoint through unused package reexports.

Full-horizon initialization calibration (MLQ **6074**, coherent; **6075**,
ensemble): 32 local births per dose, 16 common 1,000-step training-pool episodes,
plus a 1,000-step teacher-forced observation trace. Candidate RNG starts and
victims are paired across doses. These are frozen candidate panels, not
training results:

| Architecture | Dose | Mean team-action RMS change | Numerically neutral | Mean gain of best four candidates |
| --- | ---: | ---: | ---: | ---: |
| 16-resident ensemble | 2 | 0.000362 | 28.1% | 0.01493 |
| 16-resident ensemble | 8 | 0.001969 | 0% | 0.07655 |
| 16-resident ensemble | 32 | 0.004213 | 0% | 0.12402 |
| One coherent controller | 2 | 0.0000519 | 25.0% | 0.000102 |
| One coherent controller | 8 | 0.000509 | 0% | 0.001489 |
| One coherent controller | 32 | 0.003271 | 0% | 0.029648 |

These initialization-only diagnostics motivated dose 32 for the controlled
training experiment; they do not establish a universally optimal mutation dose.
The experiment driver is `scripts/experiment_collective_control_v9.py`:
`calibrate --arm ensemble|coherent --output PATH`, or
`train --arm ensemble|coherent --mutation-events 32 --run-dir PATH --output PATH`.
Run it through `mlq`, with `max-parallel-runs=1`. The two arms start with
4,192 versus 4,190 mutable fields (16 × 32 nodes versus 1 × 523), use only local
mutation, a common 128M-transition target, seed1, and full 1,000-step horizons.
They do not match effective mutation impact, per-decision acceptance threshold,
or allocation between proposal and validation episodes. The comparison is of
two complete search/controller configurations, not an isolated causal test of
averaging. One training seed supports conditional checkpoint comparisons only.

#### Completed v9 training evidence

Both jobs succeeded, with `max-parallel-runs=1`, priority0, and a 45-minute
runtime limit. All values below are raw 1,000-step episode returns. The same
64 fresh final-test seeds score both v9 champions and the saved v4 reference:

| Configuration | MLQ job | Actual training/evaluation transitions | Training seconds | Champion generation | Fresh mean return | Observations replaced with adapter mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved v4 reference | evaluated inside both jobs | historical budget differs | — | 52 | 3.15865 | not remeasured in these jobs |
| V9 averaged ensemble | 6078 | 128,156,000 | 693.24 | 130 | 3.81798 | 3.22006 |
| V9 coherent controller | 6079 | 101,809,000 | 760.63 | 415 | 12.07577 | 12.04234 |

The ensemble reached its transition target at generation132, accepting 234
of 2,111 paired decisions. The coherent arm accepted 96 of 525 decisions and
culled at generation525 after 20 stale development evaluations; it did **not**
consume the same actual budget as the ensemble. Its stable-suite champion
(development11.8318 at generation415) was retained even though the terminal
population had fallen to development8.7811. Independent paired acceptance
reduces selection noise but does not prevent population regressions across an
adaptive run; the stable champion and cull serve distinct purposes.

Paired final-test comparisons (mean ± SEM over 64 shared episodes):

- Ensemble minus v4: **+0.65933 ± 0.03028**, positive on64/64.
- Coherent minus v4: **+8.91712 ± 0.13087**, positive on64/64.
- Coherent minus ensemble: **+8.25779 ± 0.12340**, positive on64/64.
- Ensemble observation advantage: **+0.59792 ± 0.03324**.
- Coherent observation advantage: **+0.03342 ± 0.04297**.

Thus the coherent configuration wins raw reward for this seed, with fewer
consumed transitions, but its gain is essentially autonomous under the
observation ablation; it is **not evidence of better feedback control**.
The ensemble shows a clear observation benefit. Neither result establishes
competitive general locomotion, across-seed reliability, or an optimal dose.
The old reference used a different historical training budget, so its paired
test comparison is not a sample-efficiency claim.

Artifacts:

- `runs/collective_control_v9_ensemble_seed1/champion.json`
- `runs/collective_control_v9_coherent_seed1/champion.json`
- `runs/collective_v9_ensemble_result.json`
- `runs/collective_v9_coherent_result.json`
- `runs/collective_v9_{ensemble,coherent}_calibration.json`

Each run also retains its source-hashed manifest, TensorBoard events, latest
checkpoint, per-generation transition accounting, paired decision evidence and
development returns. The result JSON includes all fresh returns and the
generation-independent seed list, allowing paired comparisons without treating
the initial calibration or repeated development cases as a final test.

### Coherent CMA v10: direct feedback and joint parameter search

The v9 coherent champion was better than the ensemble but remained near an
autonomous low-action solution. Of its six output nodes, five reached sensors
only through three to five additional recurrent node hops. Its 523-node graph
had 399 reachable nodes but just 25 reachable observation-source edges; the
median maximum truth-table partial slope was 0.2877. Connectivity alone did not
provide a strong, easily mutable sensor-to-action map.

MLQ **6085** checked the actual old CUDA controller against stock Gymnasium over
four full 1,000-step episodes: maximum raw reward and observation differences
were both **zero**. This was not a reward-normalization or backend-units problem.
Its action RMS was0.09492 and mean late-episode temporal action SD0.002282.
The new affine CUDA evaluator also reproduced the stock Gym returns exactly
under an independently specified constant-action oracle. Evidence is retained
in `runs/coherent_cma_boundary_check.json`; the throwaway driver was removed.

`cleanrl/coherent_cma_v10.py` changes both the policy representation and search:

- One signed affine policy, `clip(midpoint + half_range * (W z + b), low, high)`,
  where `z=(observation-mean)/scale` uses a fixed calibrated adapter without an
  input tanh. HalfCheetah has108 parameters: all17 observed state dimensions
  connect directly to all6 actuators. There is no resident averaging.
- Active full-covariance CMA-ES from `cma` evolves those parameters jointly.
  All64 candidate fitnesses update the search distribution. This replaces
  isolated graph-field mutation and winner-only replacement; there is no PPO,
  critic, backpropagation, replay buffer or model-based rollout.
- Policy evaluation stays compiled FP32 CUDA, with cached graphs and native
  environments per population/episode shape. Only the small evolutionary
  distribution's statistics run through the established NumPy CMA implementation.
- Candidates share two rotating full-horizon training episodes. Every ten
  generations, the current distribution mean and current train-best candidate
  compete on the same fixed64-episode development suite. The persisted champion
  is reloaded before final evaluation on128 separate heldout episodes, including
  a paired observation-blind ablation.
- Default budget128M actual train/development transitions; whole-generation
  overshoot is reported. Culling uses development EMA0.8, material delta5,
  20 warmup evaluations and30 stale evaluations. Numerical library stops and
  budget stops also score/save the final population before final testing.

Each optimizer owns its random sampler. In particular, seed zero does not
silently invoke pycma's time-based seeding or share global NumPy state.
Run names now describe the method/version, **without a seed suffix**; the seed
remains recorded in configuration and manifests. Historical v9 artifact paths
are unchanged so existing evidence links remain valid.

Entrypoints:

```bash
mlq submit --name coherent-cma-v10 --max-parallel-runs 1 --time-limit 45m \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -m cleanrl.coherent_cma_v10 train
```

The default run directory is `runs/coherent_cma_v10`, containing
`champion.json`, `latest.json`, `manifest.json`, `metrics.jsonl`,
`tensorboard/` and `final_result.json`. Checkpoints are evaluable policies,
not resumable CMA optimizer states. Evaluate with
`python -m cleanrl.coherent_cma_v10 evaluate --checkpoint PATH --output PATH`
through `mlq`. This is a new architecture/search configuration, not an isolated
ablation of covariance adaptation.

MLQ **6090** passed **44** combined regressions, including full-horizon raw
HalfCheetah returns, native Hopper terminal masking, all sensor-to-actuator
paths, graph reload/blind toggles, real CMA convergence on a rotated
anisotropic objective, fixed-suite champion selection, every stop boundary,
checkpoint loading and isolated seed-zero reproducibility. Independent review
found the pycma seed-zero issue; its correction passed focused re-review with
no remaining material finding. The full training job is **6093**, named
`coherent-cma-v10`, with exclusive queue limit1 and a 45-minute runtime limit.

#### V10 result: four-digit feedback control

Job **6093** completed successfully at its transition budget, not a plateau
cull. The generation910 CMA mean was the development champion, scoring6607.58
on the fixed development suite. Reloading its saved checkpoint and evaluating
128 separate full-horizon episodes produced:

| Measurement | Result |
| --- | ---: |
| Heldout mean raw return | **6582.44 ± 21.85 SEM** |
| Heldout return range | 5579.29–7000.52 |
| Episodes above1000 | 128/128 |
| Observation-blind mean | −179.71 |
| Paired observation advantage | **+6762.15 ± 21.84 SEM** |
| Actual train/development transitions | 128,192,000 |
| Final evaluation transitions, including blindness | 256,000 |
| Training-loop wall time | 421.07 seconds |

The independent test suite was touched only after champion selection and the
stop checkpoint were frozen. These are conditional results from one training
seed, not an across-seed reliability estimate.

Learning milestones below are **development** scores, not repeated final tests:

| First development champion above | Generation | Actual transitions | Elapsed seconds |
| --- | ---: | ---: | ---: |
| 1000 | 10 | 1,472,000 | 8.63 |
| 3000 | 50 | 7,104,000 | 28.41 |
| 5000 | 180 | 25,408,000 | 89.13 |
| 6000 | 320 | 45,120,000 | 152.32 |

The observation ablation separates this from v9's nearly autonomous12-return
solution: removing state feedback destroys performance. This establishes a
useful direct-feedback evolutionary baseline, not a claim that affine control
or CMA-ES is globally optimal.

MLQ **6095** then reloaded the learned checkpoint and drove native MuJoCo and
stock Gymnasium with the same compiled CUDA actions for four complete1,000-step
test episodes. Returns were6481.83,6579.92,6774.03,6609.14 in **both** backends,
also identical to the corresponding original test returns; maximum per-step
reward and observation errors were both zero. That job used queue limit1 and
a 15-minute runtime limit. Its throwaway driver was removed.

Primary artifacts:

- `runs/coherent_cma_v10/champion.json`
- `runs/coherent_cma_v10/final_result.json`
- `runs/coherent_cma_v10/stock_gym_verification.json`
- `runs/coherent_cma_v10/manifest.json`
- `runs/coherent_cma_v10/metrics.jsonl`

### Emergent coherent control: neutral construction and acquired state

`cleanrl.emergent_control` is the discovery experiment, not another CMA policy
upgrade. It inherits a random 32-node two-input soft-logic graph, with capacity
256. Synchronous bounded recurrence, observation encoding and outer evolution
are supplied computational priors; no task-solving circuit, fitted controller,
estimator or within-life optimizer is seeded. Historical v9 and CMA v10 remain
unchanged baselines. There is one coherent organism, not an action ensemble.

Two mechanisms transfer from current bitlearn:

- **Neutral construction:** every eighth mutation proposal is structural-only;
  the others use two point events by default. Exact genotype copies are removed.
  A separate uniform reservoir retains genotype-distinct offspring with a
  conservative all-input equivalence proof over every action root and its
  transitive recurrent dependencies. If no positive birth is admitted, a fair
  coin can admit that neutral offspring. Equal return is not a neutrality proof.
- **Acquisition interventions:** inherited structure stays frozen while useful,
  irrelevant and restricted experience change life-local state. This measures
  what an organism acquires rather than relabeling increased return as learning.

A lifetime is two full 1,000-step HalfCheetah-v4 episodes: support followed by an
independently reset query. Both use the same hidden scalar actuator gain,
independent random sign times Uniform[0.5,1]. The gain multiplies nominal actions
before native physics; neither the gain nor the post-gain action is a sensor.
Inputs are the 17 ordinary observations, `asinh(previous raw reward)`, and an
episode-start indicator. Reward and previous-proposal feedback reset at each
physical boundary; recurrent node state persists from support to query and
resets from the inherited genome between lifetimes. Observation calibration is
fixed, task-blind random-action calibration, not a trained controller.

Fitness is **query raw return only**. Support can serve exploration but all its
physics is counted. Positive candidates pass rotating proposal lives, fresh
shortlist screening, then an independent one-sided paired Student-t lower bound.
The normal-difference approximation is not an across-run error guarantee.
Development chooses checkpoints, not births. The hidden-gain objective is a
different task and must not be plotted as ordinary HalfCheetah benchmark return.

Frozen evaluation uses 128 fresh lifetimes, paired across:

1. Intact support state.
2. Pristine state at the query boundary.
3. Support under the opposite gain, followed by the true query gain.
4. Only the last eight actual support inputs and previous proposals replayed
   from pristine state.
5. Reward withheld during support, with ordinary query feedback restored.

All branches reset the physical query with matching seeds, true gains and
immediate boundary inputs. First actions, first-100 and remaining-900-step
effects are reported separately. A separate gain-one, pristine-query control
reports ordinary raw HalfCheetah return. State carryover can be controller phase
or warmup; donor trajectories differ on-policy; recent-input replay is not a
universal context-removal control. Null late differences do not exclude rapid
reacquisition. None of these contrasts alone proves a novel learning algorithm.

```bash
mlq submit --name emergent-control-full --max-parallel-runs 1 --time-limit 45m \
  --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -m cleanrl.emergent_control train
```

The default budget is 128 million actual physics transitions, a 30-minute
internal limit, and plateau stopping after 40 development evaluations without
at least 1.0 return of raw/EMA progress, following 20 warmup evaluations. Development runs
every five generations. A completed run automatically evaluates its frozen
champion. `latest.json` saves the full inherited continuation state and
generation-indexed RNG schedule; `champion.json` is evaluation-only. Continue
through `resume --run-dir PATH --additional-transitions N`, or evaluate through
`evaluate --checkpoint PATH --lives 128`, always via mlq. Source fingerprints,
including native physics C source, reject incompatible replay. Lineage records
retain the complete genome of each admitted descendant.

#### First full run: construction enabled, learning not discovered

MLQ **6127** completed and plateau-stopped at generation 295 after
**83,776,000** train/development transitions, with a 128-million ceiling; it was
not stopped by the queue or shortened as a smoke run. Training-loop wall time
was **453.34 seconds**. The generation 285 development champion improved from
−5.58985 to −0.18920. Frozen evaluation used another **1,536,000** physics
transitions, plus separately counted recent-input replay.
Node-update telemetry counts logical genotype nodes; fixed-capacity CUDA also
executes padding. These counters are not measured FLOP counts.

| Frozen measurement, 128 independent lifetimes | Mean ± SEM |
| --- | ---: |
| Hidden-gain query return | −0.221027 ± 0.058272 |
| Intact minus pristine-query state | +0.000746 ± 0.003842 |
| Intact minus opposite-gain support | +0.0000000731 ± 0.0000000346 |
| Intact minus recent-eight-input replay | +0.000653 ± 0.001670 |
| Intact minus reward-withheld support | exactly 0 |

There were **19 positive and 146 neutral births**. Seven neutral duplications
survived, but no subsequent positive birth recruited any of those new IDs.
All six positive structural births were deletions. The champion has 25 inherited
nodes, only **three action-reachable nodes**, five dangling outputs decoding to
exact zero action, and no reward/boundary input in its reachable computation.
The final continuation genome has 23 nodes. Neutral construction is operational;
useful cumulative construction and task-specific acquired competence were not
demonstrated. Do not scale this run on the strength of neutral-birth counts.

The frozen zero-action control confirmed that the apparent improvement was
effectively **inaction**, not acquired competence. On the same 128 query lives,
zero action scored **−0.219342 ± 0.058167 SEM**; the champion's paired advantage
was **−0.001685 ± 0.001297**. On four full nominal episodes, its action RMS was
only **0.001574** on the [−1,1] action scale. Native and stock Gymnasium received
the same compiled CUDA actions and matched observations and rewards exactly,
with identical returns also matching the saved nominal-query results. This
diagnostic, MLQ **6139**, used max-parallel-runs=1 and a 15-minute hard cap.
It added 256,000 null-control and 8,000 native/stock physics transitions, never
modified a checkpoint, and never used the null controller as a training seed.

Verification **6126** passed **85 tests** on real CUDA/native physics, including
the historical addressing and affine/CMA regressions. An earlier combined run
passed 83 but exhausted Dynamo's process-global specialization cache in two later
legacy shape tests; the new CUDA fixtures now isolate compiler state without
raising production limits or enabling eager fallback. Independent reviews found
one post-development deadline overrun; a failing regression established it, the
fix passed, and focused re-review cleared it. The acquisition review found no
material defect. Verification and training used max-parallel-runs=1, with
20-minute and 45-minute hard caps respectively.

Artifacts are under `runs/emergent_control/`: `initial_genome.json`,
`latest.json`, `champion.json`, `metrics.jsonl`, `lineage.jsonl`,
`acquisition_generation_285.json`, `final_result.json`,
`postrun_verification.json`, `assessment.json`, and `tb/`.

### Arithmetic population: evolving a feedback network from scalar programs

`cleanrl/arithmetic_evolution.py` and `cleanrl/evolving_programs/` replace the
single-incumbent convex soft-logic experiment with **256 independently acting
programs**. This is not affine/CMA parameter fitting, PPO, SGD, action voting, or
an installed learning algorithm. Every initial program is random: 16 nodes,
with a capacity of 128. Evolution chooses opcodes, wiring, literals, initial
state, and motor roots. The supplied scalar operations are `COPY`, `ADD`, `SUB`,
`MUL`, `DIV`, `TANH`, and `CONST`.

The redesign addresses several coupled barriers, not one isolated hypothesis:
the old parallel proposals were not a persistent breeding population; convex
tables attenuated individual incoming signals; synchronous graph depth incurred
physical-action latency; and symmetric motor reversal rewarded inaction before
there was a useful controller to adapt. Accordingly, the new system has:

- Four synchronous internal microticks per physical observation. Previous action
  is held fixed through those ticks. This is a supplied compute allowance, not an
  evolved discovery.
- Signed float32 state and `asinh`-encoded normalized observations. Internal
  arithmetic is not implicitly clipped or squashed; `TANH` is an explicit op.
  Motor commands alone are clamped to [−1,1].
- Permanent lifetime invalidation for nonfinite active computation. Safe-zero
  motor commands protect batched physics, but the invalid organism scores
  **−infinity**, never the zero-action return. Unused arithmetic is not a death
  condition.
- Whole-program generational reproduction: 95% exponential rank selection at
  temperature 0.1 plus 5% full-support sampling, with averaged exact tie ranks.
  Ordinary sampled-parent copies occur with probability 0.1; they are not
  privileged elites. Other births get point edits or isolated duplication/deletion.
- A preregistered curriculum: cold, nominal-gain HalfCheetah first; a development
  return of **1,000** unlocks positive hidden gains sampled uniformly from [0.5,1].
  Positive-gain lives contain a full 1,000-step support episode and a separate
  1,000-step query. Only program state crosses that physical reset; previous
  action and previous reward are cleared. Reward and boundary inputs are
  available, but no estimator or update rule is provided.

Training uses two lives per organism. Every ten generations, the top eight
training candidates are scored on the same 64 development lives; development,
not final evaluation, chooses the champion and the curriculum transition.
The two frozen stage champions then receive the same reserved 128-case suite.
Its controls are intact state, reset state, support at gain `1.5−g`, independently
reset support at the same gain, replay of the last eight support inputs/actions,
and support with the reward input suppressed. The mismatched support gain stays
inside the training range. Cold-state differences alone do not establish
acquired knowledge: gait phase and warmup can change them.

#### Full run and frozen acquisition test

MLQ **6164** passed **102 tests** in 23.86 seconds on CUDA/native physics,
including legacy emergent-control regressions. Two independent source reviews
found no material defect. MLQ **6165** ran the full experiment with
max-parallel-runs=1 and a 60-minute hard cap. Seed **1** is checkpoint metadata,
not a run-directory suffix.

The nominal gate opened at **generation 10**, at development return
**1,320.4785**. The positive-gain champion was selected at generation **150**,
at development return **3,438.8667**. Training stopped at generation **232**
after **257,536,000 train/development transitions**, in **738.64 seconds**.
The configured 256-million transition threshold is checked between complete
batches, and the final development batch is still evaluated; it is not an exact
hard transition ceiling. Both frozen evaluations together added **3,328,000**
physical transitions and **2,048** replay transitions. No episode was shortened.

| Frozen query measurement | Nominal-stage champion | Positive-stage champion |
| --- | ---: | ---: |
| Cold nominal return | 1322.639 ± 0.981 | 301.192 ± 88.068 |
| Positive-gain intact return | 830.664 ± 34.306 | 2901.209 ± 149.715 |
| Intact minus reset | −2.117 ± 9.889 | −51.331 ± 129.380 |
| Intact minus mismatched support | −16.957 ± 12.974 | 28.885 ± 133.303 |
| Intact minus independently reset same-task support | −6.772 ± 7.787 | −39.679 ± 117.776 |
| Intact minus last-eight replay | exactly 0 | exactly 0 |
| Intact minus no support reward | exactly 0 | exactly 0 |

Values are means ± SEM over 128 lives; every reported champion life was valid.
This is useful evolved control, not another zero-action plateau. However,
**task-specific acquired competence is not demonstrated**, and nominal
competence was not retained through positive-gain selection. The development
and final returns also differ materially; the development-selected value must
not be substituted for held-out performance.

At the end, 97.27% of the population was viable, with 22.46 total and 10.82
action-reachable nodes on average. Realized parent effective counts ranged from
40.53 to 107.24. These are counts of contributing parent slots, not a claim of
independent genealogical founders.

Both champion paths reconstruct, with every parent/child genotype hash checked,
to **random founder 163**. Its seven action-reachable nodes became nine in the
nominal champion and eleven in the positive champion. The nominal champion is
not the positive champion's literal ancestor; their paths share the founder.
On the positive path, node 21 was neutrally duplicated at generation 86 and
became action-reachable at generations 90–96. No post-founder node is active in
the final champion. Reachability alone is not proof of useful new capacity.

Run artifacts are in `runs/arithmetic_evolution/`: `initial_population.json`,
`nominal_foundation_population.json`, `latest.json`, both `champion_*.json` and
`final_*.json` files, `metrics.jsonl`, `lineage.jsonl`, `ancestry_audit.json`, and
`tb/`. The latest checkpoint saves the full population, but this experiment has
**no resume CLI**. Frozen champion loading requires matching source fingerprints.

#### What the evolved computation actually does

The positive champion's active graph reduces to the following real-arithmetic
form, with zero-based observation/action indices and the existing encoder
`z = asinh((observation−mean)/scale)`:

```text
b[t] = tanh(a[t−1,1])
d[t] = tanh(z[t,5]) / −1.532714963 − b[t]
r[t] = d[t−1] + tanh(a[t−1,4] + z[t,15]) + z[t,0]
a[t] = clip([b[t], r[t], b[t], d[t], d[t], −0.971990108], −1, 1)
```

The one-step delay follows from the graph's four synchronous microticks.
At cold start, `d[−1]` is initial(node 9) minus initial(node 2), and previous
action is zero. At the support/query boundary, the delayed value is retained
but previous action is cleared. The observations used here are torso height,
front-thigh angle, and front-shin angular velocity. Neither reward nor the
boundary channel reaches an output.

This is a small **neural-style recurrent feedback controller**, not an installed
neural policy architecture and not evidence of an evolved optimizer. We supplied
the scalar `TANH` primitive and action-feedback interface; evolution selected
the useful wiring and constants. The reduced equations, at zero encoded sensory
input, produce a four-step action-1 cycle `−1, 0, +1, 0`. That is an algebraic
zero-input check, not a physical locomotion measurement.

The compiled CUDA algebraic shadow and full graph were compared at identical
observations and actual previous actions over four full 1,000-step native
episodes. Maximum motor discrepancy was **2.3842e−7**. This checks the local
transition, not independent long-run floating-point trajectory equivalence.
Native and stock Gym matched observations and rewards **exactly**, including
terminal observations and boundary flags.

On **512 fresh frozen cases**, the positive champion scored
**3044.130 ± 65.897 SEM**, against zero action **−0.274743 ± 0.029942**.
The nominal-stage comparison champion scored **747.994 ± 15.933** on those
same positive-gain cases. All diagnostic variants remained valid.

| Intervention on the positive champion | Paired positive-query return loss, mean ± SEM |
| --- | ---: |
| Remove all previous-action inputs | 3112.550 ± 65.963 |
| Remove torso-height input | 200.344 ± 68.664 |
| Remove front-thigh-angle input | 977.127 ± 60.688 |
| Remove front-shin-velocity input | 4.205 ± 63.959 |
| Replace the denominator by −1 | −16.620 ± 61.385 |
| Delete all nine inactive nodes | exactly 0 |

Source ablations substitute zero **after** the encoder. The feedback dependence
and front-thigh contribution are real, but neither the front-shin sensor nor
fine-tuning that denominator has a resolved benefit in this assay. Do not infer
necessity from a wire merely being reachable.

The 512-case cold-nominal scores were **194.724 ± 38.750** for the positive
champion and **1311.890 ± 3.791** for the nominal-stage champion. A separate
128-case fixed-gain sweep, with full support at the same gain, gave the positive
champion means **1827, 2778, 3943, 4324, 188** at gains
**0.5, 0.625, 0.75, 0.875, 1.0**. Its high-gain collapse therefore remains with
support; it is not explained simply by cold initialization.

An **exploratory
clairvoyant comparison**, with effective gain `min(g,0.8)`, scored
**3523.728 ± 41.981**, a paired gain of **479.599 ± 66.880** over the frozen
controller on the 512 fresh cases. The cap was fixed from the earlier 128-case
response bins before these cases were observed. A context-aware actuator wrapper
could implement this by applying `min(1,0.8/g)` after the controller clamp while
preserving its unscaled internal action feedback. This is not a learned policy,
not a training seed, and not a global optimum. On its own, this does not isolate
context value from context-independent actuator attenuation.

MLQ **6198** supplied the missing controls on those same 512 cases, with full
support/query episodes, max-parallel-runs=1, and a 20-minute cap. It added
**9,216,000** physical transitions. Every arm used the same compiled core,
preserved its unscaled internal action feedback, and applied a diagnostic
actuator multiplier. The unscaled arm reproduced the prior baseline exactly.

| Actuator-shell rule | Positive-query return, mean ± SEM |
| --- | ---: |
| Unscaled | 3044.130 ± 65.897 |
| Fixed 0.80 | 2603.662 ± 41.747 |
| Fixed 0.85 | 2893.358 ± 45.947 |
| Fixed 0.90 | 3125.963 ± 51.068 |
| Fixed 0.95 | 3217.942 ± 55.754 |
| Correct context: `min(1,0.8/g)` | 3534.010 ± 41.451 |
| Mean of three shuffled-context controls | 3013.304 ± 53.566 |

Each shuffled arm had **exactly the same multiplier multiset** as the
correct-context arm; only its association with the actual gains changed. Correct
context beat their mean by **520.707 ± 48.773**, and beat the best fixed-grid
member by **316.069 ± 40.992** in paired comparisons. The fixed-grid winner was
chosen on this evaluation, optimistically favoring the context-free comparator.
The three shuffles are controls on 512 cases, not 1,536 independent samples.

This isolates useful context information **within this frozen-core scaling
family**. It is not a bound against every context-free controller, proof that
the allowed program can construct or learn the estimator, or an evolved
learning mechanism. The explicit shell uses two float32 multiplications;
the earlier effective-gain simulation used one. Therefore compare arms within
this controlled experiment, rather than interpreting its small numerical
difference from the initial clairvoyant result. Cases, multipliers, permutations,
paired results, and verification source are in `context_control_audit.json`.

The genealogy controls also distinguish growth from new capacity. The
generation-86 duplication was exactly neutral. The generation-90 recruitment
event was **595.491 ± 154.059 worse** than its parent on a frozen positive suite.
By generation 96, deleting the recruited node cost **1247.390 ± 210.729**:
that route had become useful. But redirecting node 21's consumer to its original
producer, node 13, at generation 90 reproduced the return exactly. Thus there
was temporary causal recruitment, not proof of irreducible extra capacity;
the final champion used none of the added nodes.

MLQ **6176** completed these measurements with max-parallel-runs=1 and a
25-minute hard cap: **19,456,000** evaluation transitions plus **8,000**
native/stock parity transitions. An earlier diagnostic attempt, **6169**, passed
its algebraic check but compared the vector environment's autoreset observation
with Gym's terminal observation. The diagnostic now reads `final_observation`;
no training or policy code was changed. The repeated cases are not counted as
additional independent samples.

`mechanism_audit.json` contains the cases, variant genotypes, paired results,
reduced equations, and archived verification source. `assessment.json`
summarizes the supported conclusion. This single-seed, jointly redesigned
experiment establishes useful unseeded feedback construction, **not acquired
learning, retained nominal competence, or optimality**. More capacity alone is
not supported by these measurements.
