# Vector return response v13

Implementation of the [reviewed design](vector-return-response-v13-design.md). Source: `cleanrl/ppo_continuous_action_vector_return_response_v13.py`.

The critic predicts actual action-dependent future reward profiles, replacing v12's physics-model/Jacobian path. The actor uses analytically integrated Beta responses, with exact KL constrained nonlinear candidate evaluation.

## Representation and targets

Six future intervals: `[0,1)`, `[1,8)`, `[8,32)`, `[32,128)`, `[128,512)`, `[512,1000)`, clipped at each real episode end. Seven signed channels: forward progress, then each of six negative control expenditures. Summing channels and intervals exactly recovers raw reward-to-go for every eligible transition.

Targets use complete future suffixes within the current fixed-policy rollout. Left-partial episode prefixes are allowed; unknown warmup age is inferred from the actual next terminal. Right-censored targets are excluded, not zero-tail bootstrapped. The actor can still use known-age states from those censored suffixes with predicted remaining interval lengths. There is no GAE, replay, or Bellman backup.

A separate state-only network predicts the mean reward profile. The response network emits a `[6 intervals,7 components,39 features]` tensor per state: 12 exact Fisher-whitened Beta score coordinates plus 27 learned response kernels. Kernels include two single-actuator kernels per actuator and one kernel for every actuator pair. Global active exponents are learned within `(0,4)`; inactive exponents are zero. This bounded family avoids arbitrarily narrow kernels, but remains an approximation and does not cover arbitrary higher-order joint action interactions.

Both network heads predict per-transition component rates and multiply by actual interval lengths before raw-profile regression. Empty intervals contribute zero. Raw signed reward units are preserved. One batch-wide target-energy scalar normalizes the whole loss; there is no per-head loss normalization. Longer intervals have greater optimization sensitivity in this rate parameterization; it is not equivalent to equal-weight per-step-rate fitting.

## Fitting and actor update

- Fresh seed 1, HalfCheetah-v4, nominal 8M transitions, 16 environments × 2,048 steps; environment threads 2.
- Entire eligible rollout fitted at once; no minibatches.
- State-only profile: 10 Adam steps at 0.001. Its prediction is then frozen.
- Action response: 8 Adam steps at 0.0003 against the observed residual profile; coefficients and kernel shapes trained together.
- Both use gradient norm limit 0.5, BF16 trunks and FP32 readouts. Analytic Beta moments use FP64 for stable centering/variance/derivatives.
- One natural actor direction, CG50, with KL ceiling 0.03. Sixteen nonlinear response/KL evaluations bracket and refine the ray, including interior optima. Best measured positive feasible gain is retained; this is not a global constrained solve or a guarantee of environment-return improvement.

Kernel shape, response coefficients, old-policy score coordinates, and old-policy kernel normalization stay fixed throughout actor optimization. State-only profile cancels from expected policy improvement. Reward/profile reduction occurs at the final actor objective, not in critic supervision.

This staged baseline-then-response fitting is a concrete implementation choice beyond the design's initial joint-fit discussion. Any state-only baseline has zero population score expectation; staged fitting does not change that identity. Finite-sample overfitting and shared-network approximation remain concerns.

## Verification and experiment record

- Initial numerical job **7557**, parallel limit 1, time limit 25 minutes: 12 passed; one compiled unchanged-policy gain check failed because its 1e-10 tolerance was below FP32 path rounding (observed difference 2.85e-8). FP64 derivative contracts passed.
- Independent end-to-end review found no blocker in target boundaries, sparse kernel integration, frozen policy coordinates, full-batch update wiring, or compiled-output lifetime.
- Final numerical job **7560**, parallel limit 1, time limit 25 minutes: **14 passed**. Only the compiled unchanged-policy gain check uses an FP32-appropriate 1e-6 tolerance; unchanged KL remains 1e-10 and FP64 contracts remain strict.
- Fresh training job **7566**, parallel limit **1**, time limit **60 minutes**, depends on successful 7565. Nominal 8M transitions, seed 1. Monitor returns and cancel persistent uninformative underperformance; do not promote on fitting loss alone.

Tests cover exact polynomial moments, analytic alpha/beta derivatives versus autograd, dense/sparse kernel equivalence, learned exponent gradients, signed profiles and censoring, score/kernel credit versus integrated-objective autograd, baseline/response gradient isolation, an interior ray optimum, and two compiled learning/actor cycles. No smoke training or held-out promotion gate.

Source SHA256 before benchmark: `1bc21af403e475fbcc956a49f3451b111a752779855cee09a0a781672236ca4f`.

- Training attempt **7561** failed while compiling the first production-shape target construction, before any optimizer update. The target scans were changed to contiguous final-axis scans with identical mathematical targets.
- Production-shape regression job **7562**, parallel limit 1, time limit 25 minutes: **15 passed**, including exact 2048 × 16 × 6 input / 32768 × 6 × 7 output shapes and retained compiled outputs.
- Fresh training attempt **7563** still failed before optimization: earlier small-shape compilations in the test process had changed specialization and concealed the production Inductor `SplitScan` lowering bug. Job **7564** reproduced it after the production regression was isolated with a cleared Dynamo cache, static shapes, and training's no-gradient context (14 passed, 1 failed).
- Target accumulation now uses a hierarchical parallel prefix scan with 128-element tiles, preserving full-rollout targets and FP64 accumulation. These arithmetic tiles do not partition fitting or add Bellman sweeps. Numerical job **7565**, parallel limit 1, time limit 25 minutes: **15 passed**, including the isolated production-shape regression.

## Results

**Negative result.** Job **7566** was cancelled for sustained underperformance after **6,667,904 logged transitions**, rather than extended. Its last-100 training episode return was **1,232.23**; its peak was **1,477.14 at 3,325,568 transitions**. No final 8M score or completed-run checkpoint exists. The actual run directory contains `progress.json` and a separate `experiment_outcome.json` recording cancellation; no completion result was fabricated.

| Nominal transitions | v13 response critic | v12 full batch, fixed targets |
|---|---:|---:|
| 1M | 438.12 | -45.38 |
| 2M | 1,330.08 | 481.24 |
| 4M | 1,235.44 | 1,720.64 |
| 6M | 1,262.71 | 2,779.96 |

Scores use the nearest logged rollout at each budget, from fresh seed-1 runs. The early advantage disappeared; at 6M v13 was about 55% below v12. The older v10 minibatch/eight-sweep reference was stronger still. This is a single-seed comparison of complete algorithms, not an isolated test of vectorization.

All **203 actor proposals were accepted**, with mean joint KL **0.0299727** and minimum **0.0298673**. Thus the nonlinear search still chose almost the entire allowed KL budget throughout this run. Acceptance and positive modeled gains did not imply consistent environmental improvement. Approximately **75.6%** of collected transitions had complete suffixes available for critic fitting (**5,029,008 fitted transition samples**, before repeated optimizer passes).

An additional optimization limitation remains: the 50-step conjugate-gradient solve had a median relative residual of **0.0446**, mean **0.0478**, and maximum **0.1659**. Exact candidate KL remained feasible, but these were approximate natural directions, not fully converged solves. This run does not isolate the contribution of solver error, noisy long-horizon targets, response-family approximation, or finite neural fitting.

The principal hypothesis was not supported: removing derivatives of a learned dynamics model and analytically integrating reward responses was insufficient for sustained improvement. Raw-profile regression can fit predictable state-dependent outcomes without accurately identifying the smaller action effects. Furthermore, the response models collection-policy continuations while a candidate actor changes behavior throughout future trajectories. Neither low profile loss nor exact integration eliminates those gaps. The logged losses average optimizer-step losses within each fitting stage; they are not a matched before/after estimate of explained action variance.

The dependent report job **7567** was skipped because training was cancelled. Partial-report job **7568** succeeded; final report job **7569**, parallel limit **1**, time limit **5 minutes**, adds solver summaries and accurate loss labels. Numerical verification **7565** passed all **15 CUDA tests**; independent review found no target-boundary or scan-censoring defect.

Artifacts: [numeric results](vector-return-response-v13-results.json), [comparison plot](vector-return-response-v13-results.png), [vector plot](vector-return-response-v13-results.svg). Raw evidence: `runs/HalfCheetah-v4__vector_return_response_v13_8M__1__1789581691929940339/`. Preserve v13 as a negative result; no automatic long-run extension.
