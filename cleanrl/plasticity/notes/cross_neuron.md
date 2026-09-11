# Cross-neuron posterior (low-rank EKF)

Files: `network_bayes_stream_v2.py` (full P x P EKF, reference), `lowrank_bayes_stream_v1.py` (precision `diag(D) + M M^T`, rank r, buffer T=64, Gram-eigh truncation with dropped energy folded into D, Woodbury gain, S = M^T D^-1 M maintained incrementally), `panel_hd_ekf.py` (same learner, batched per-bar update with n observations, on the vol panel).

## Dense 17-64-64-1 stream (network_bayes_stream_v2 task), paired seed 1, 65,536 samples, batch 1

Endpoint clean test MSE (zero predictor 1.3592 / 2.9057); selection by validation endpoint over the same grids as v2 (Adam LR 1e-5..1e-2, prior scale 1e-3..10); all selections interior unless marked. Adam endpoint 0.130993 reproduces v2's record exactly.

| arm | stationary | hetero 1 + switch 0.5 | wall (5 configs) |
|---|---|---|---|
| Adam | 0.13099 | 0.29392 | 18 s |
| full P x P EKF (v2, diffusion 1e-5) | 0.09601 | 0.20232 | 490 s |
| lowrank r=0 (diagonal Kalman) | 0.12887 | 0.29310 | 25-72 s |
| lowrank r=16 | 0.09770 | - | 84 s |
| lowrank r=64 | 0.09179 | 0.24146 | 36-111 s |
| lowrank r=256 | 0.08848 | 0.21986 | 237-301 s |
| lowrank r=64 / 256, forget 1e-4 | - | 0.24436 / 0.24058, selected prior 1e-3 EDGE | 376-395 s |

Sustained clean error / null over the stream, stationary: Adam 0.1389, full 0.0947, r=0 0.1259, r=16 0.1004, r=64 0.0953, r=256 0.0928. Stress: Adam 0.1544, full 0.1234, r=0 0.1537, r=64 0.1328, r=256 0.1286.

## Vol panel (finance_panel.md task), width 256, F=257, P=132,097, one batched update per bar (n <= 180), no learning rate

| config | TEST (last 40%) | VAL (40-60%) | blocks | wall |
|---|---|---|---|---|
| ekf r=128 prior 0.1 forget 0 | **0.71999** | 0.73807 | 0.7330 0.7246 0.7415 0.7317 0.7514 0.7389 0.6812 0.6642 | 3868 s (3 concurrent) |
| ekf r=32 prior 0.1 forget 0 | 0.72482 | 0.74480 | 0.7371 0.7291 0.7479 0.7385 0.7557 0.7426 0.6867 0.6678 | 2450 s |
| ekf r=256 prior 0.1 forget 0 | 0.71868 | 0.73657 | 0.7319 0.7225 0.7394 0.7306 0.7503 0.7359 0.6822 0.6631 | 2757 s |
| ekf r=128 prior 1 forget 0 | 0.72188 | 0.74420 | 0.7369 0.7293 0.7461 0.7383 0.7512 0.7389 0.6829 0.6591 | |
| ekf r=128 prior 0.1 forget 3e-5 | 0.73562 | 0.75255 | | |

References on the same panel (finance_panel.md): Adam w256 0.7501 (own best lr), js k=3 w256 0.7255, js k=3 w1024 0.7196.

`panel_hd_floor.py` (in-sample fit of the width-256 MLP on the whole period incl. test, every 3rd bar, 7.1M samples, Adam 1e-3, 30 epochs): in-sample test-window relative MSE 0.683 -> 0.479 and still falling at epoch 30 (memorisation; not a floor estimate). Held-out offline MLP (finance_panel.md) 0.7799.

## Covariance downdate sketch v3: useful geometry, rejected accuracy claim

`covariance_sketch_stream_v3.py` / `covariance_sketch_eval_v3.py`.
Represent `P = diag(D) - U U^T`; append the exact EKF covariance downdate for
each observation. Every 64 observations, retain the largest modes of
`U^T diag(D)^-1 U`. Discarding downdates restores uncertainty in PSD order in
exact arithmetic, unlike folding lost precision correlations into a diagonal.
This does not establish optimal learning: discarding genuine certainty can hurt.

Same 5,377-parameter network, initialization, noisy observations, process variance
`1e-5` times the prior per sample, and past-only residual-scale estimate.
Complete 65,536-observation streams, seed 1. Each arm independently selects its
LR/prior by duration-weighted sustained clean validation; all decisions lock
before any test scoring. All winners are interior. These held-out sustained
MSEs differ from the endpoint-selected online/null metric above.

| Learner | Stationary sustained / endpoint | Hetero + switch sustained / endpoint |
|---|---:|---:|
| Adam | 0.173742 / 0.130993 | 0.291393 / 0.293916 |
| Neuron-block EKF | 0.166468 / 0.136692 | 0.274385 / 0.271886 |
| Full covariance EKF | **0.121055 / 0.096009** | **0.223580 / 0.202325** |
| Diagonal EKF | 0.177153 / 0.146333 | 0.289836 / 0.287068 |
| Covariance sketch r16 | 0.151911 / 0.115930 | 0.258832 / 0.245396 |
| Covariance sketch r64, fixed primary | 0.142192 / 0.108490 | 0.255045 / 0.241854 |
| Covariance sketch r256 | 0.142523 / 0.102346 | 0.253639 / 0.285250 |
| r64, mean-update direction erased | 0.214205 / 0.194007 | 0.314946 / 0.374810 |

r64 lowers sustained clean error versus Adam by 18.16% / 12.47%, but is
**17.46% / 14.07% worse than full covariance**. Erasing direction preserves
instantaneous linearized leverage while losing the gain. Increasing rank to
256 does not recover the full learner's accuracy. Reject the primary hypothesis;
do not count a cheaper approximation as an accuracy breakthrough.

Measured five-prior-grid update/report/checkpoint time: r64 6.28 / 5.58 seconds,
full 117.60 / 115.21 seconds. Compilation is separate; this is not a universal
standalone throughput claim. Steps are compiled CUDA graphs; eigentruncation
is compiled outside capture. FP32 posterior products disable TF32.

55 CUDA/reporting contracts passed (5669), including independent float64/autograd
EKF comparisons, covariance-order and whitening checks, zero-local-Jacobian
cross-neuron credit, scalar leverage, causal labels and capture restoration.
Two independent source reviews found no actionable defects. Initial experiment
5670 hit Dynamo's shared-wrapper specialization limit; 5671 was cancelled for
that known failure. Clearing the previous arm's disjoint compiler cache fixed
execution without changing the recurrence. Full jobs **5672 / 5673 succeeded**;
all eight test traces reach 65,536 in root TensorBoard events. Queue limit 1,
priority 0, one attempt, 45-minute run limits; no performance pruning.

One paired task seed, no PPO/general-optimality claim. Joint covariance is not
an autonomous scalar per-neuron gate. Source/data hashes, every locked choice,
phase curves, geometry and resource measurements are in
[v3 evidence](../../../benchmarks/plasticity/covariance_sketch_v3_evidence.json).

## Iterated assimilation v4: more faithful local fitting is not better learning

`iterated_bayes_stream_v4.py` / `iterated_bayes_eval_v4.py`.
Test the full learner's single tangent approximation, rather than another
covariance compression. Fixed-prior iterated EKF holds `theta0`, `C=P+Q` and
historical `R` throughout two or four mean refinements:

    v = C j(theta)
    theta_next = theta0 - v * (f(theta,x) - y - j(theta)'(theta-theta0)) / (R+j'v)

Add process noise once; condition covariance once after the final linearization.
One iteration reproduces full EKF; for a linear Gaussian observation, all
iteration counts give the same posterior. This is an established algorithm,
not a claimed invention or exact nonlinear Bayesian inference.

Full 65,536-observation heteroscedastic midpoint-switch stream, same data hashes
as v3, all independently validation-selected optima interior:

| Learner | Prior / LR | Sustained test MSE | Endpoint | Before / after switch |
|---|---:|---:|---:|---:|
| Adam | LR 3e-4 | 0.291393 | 0.293916 | 0.217489 / 0.365297 |
| Full EKF | prior 1 | **0.223580** | **0.202325** | **0.166161 / 0.280998** |
| Two refinements | prior 0.1 | 0.262613 | 0.237938 | 0.192385 / 0.332840 |
| Four refinements, fixed primary | prior 0.1 | 0.262596 | 0.237922 | 0.192383 / 0.332809 |

**Reject: four refinements are 17.45% worse than full EKF**, including both
phases. Their 9.88% improvement over Adam is not a new advance over this family's
incumbent. Two versus four refinements changes sustained risk by only 0.0064%.
At prior 1, sustained validation error rises from full EKF's 0.224243 to
1.111026 / 1.085616 for two / four refinements; at prior 10 both exceed 3.36.
The larger-prior failure is finite and remains in the grid, not silently dropped.
These observations reject the refinement hypothesis; they do not independently
prove that observation-noise fitting is the causal mechanism.

Five-prior-grid update/report/checkpoint times: full 105.22 s, two 142.30 s,
four 171.71 s; startup is separate. More compute produced worse learning.
The planned stationary confirmation was conditional on a stress improvement
and was not launched. No favorable endpoint or extra iteration substitutes for
the failed primary.

69 contracts passed (5677): independent CUDA float64/autograd nonlinear
recurrence, analytical linear posterior, historical-noise and clean-label
isolation, nonzero-state capture restoration, and shared reporting contracts.
Two independent source reviews found no actionable issues. Full job **5680
succeeded**, all four root TensorBoard test traces reach 65,536. Full-EKF test
curves match v3 exactly; Adam differs by at most 5.96e-8. Queue limit 1, priority
0, one attempt, 45-minute run limit, no performance cull.

The family's accuracy leader remains the full cross-neuron posterior. Neither
discarding its certainty nor repeatedly refining its local observation fit
improved it here. One paired seed does not establish generality or optimality.
Selections, phase curves, all grids, source/data/checkpoint hashes and resources:
[v4 evidence](../../../benchmarks/plasticity/iterated_bayes_v4_evidence.json).
