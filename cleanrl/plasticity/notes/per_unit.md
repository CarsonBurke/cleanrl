# Per-unit and per-connection updates

| File / mechanism / configuration | Measurement |
|---|---|
| Sphere sdplast v5–v9; `cleanrl/shared/state_plasticity.py` receives layer input, preactivation and incoming preactivation gradient | PPO files/configurations and sphere invariance measurements in `ppo_runs.md`; pre-Adam data gate modifies incoming-weight gradient only, not input gradient. |
| `/tmp/verify_v8_capability.py`; latent feature marks noisy/quiet regimes | v8 mean w noisy/quiet 0.503/1.999, ratio 3.98x; v7 1.0000/1.0000, 1.00x; within-batch ratio 3.98x. |
| `/tmp/verify_v8_perunit.py`; half units noisy on same samples | Affected/unaffected w=0.50/1.99 (3.98x); unit-averaged w sample CV 0.0004; within-sample unit CV 0.61. |
| `/tmp/verify_v9_capability.py`; asymmetric clamp | Noisy/quiet w v7 1/1, v8 0.503/1.999, v9 0.139/2.000; ratios 1/3.98/14.4x. |
| `/tmp/verify_v9_perunit.py`; half units noisy | Affected/spared w 0.133/2.000 (15.0x); sample CV of unit mean 0.0001; within-sample unit CV 0.894. |
| `hidden_stream.py`; Gaussian inputs, 1024 inputs/4 useful/64 hidden, noise_std 2, B=1, single pass, clean teacher test MSE/zero | 15 configs, each own best LR, vectorized pass 13.9 s; table below. |

| Hidden arm | Best LR | Test/zero | Useful:junk weight magnitude | Useful/junk level |
|---|---:|---:|---:|---:|
| Oracle | 3e-3 | 0.0852 | 25.9 | 1/0 |
| Mirror input-pooled | 1e-3 | 0.1457 | 18.7 | 1.0000/0.0244 |
| Mirror per-connection | 1e-3 | 0.1719 | 16.3 | 0.1847/0.0114 |
| Adam | 3e-4 | 0.2705 | 11.6 | — |
| SGD | 3e-4 | 0.3098 | 4.6 | — |
| Adam / mirror at LR 3e-3 | 3e-3 | 0.554/0.160 | — | — |

| `hidden_stream.py`; 40k steps, useful/1024 | Adam | Mirror input | Adam/mirror | Useful level |
|---|---:|---:|---:|---:|
| 4 (0.4%) | 0.3117 | 0.1952 | 1.60x | 0.854 |
| 16 (1.6%) | 0.3009 | 0.2249 | 1.34x | 0.647 |
| 64 (6%) | 0.4035 | 0.3589 | 1.12x | 0.293 |
| 256 (25%) | 0.4148 | 0.3852 | 1.08x | 0.079 |
| 1024 (100%) | 0.4384 | 0.4371 | 1.00x | 0.169 |

| `hidden_stream.py`; 4/1024 useful, noise 8, B=1, 60k samples; LR | Adam | Mirror input | Oracle |
|---|---:|---:|---:|
| 1e-4 | 0.6554 | 0.3326 | 0.3035 |
| 3e-4 | 0.6717 | 0.3228 | 0.2805 |
| 1e-3 | 0.8685 | 0.3404 | 0.2811 |
| 3e-3 | 1.5353 | 0.6503 | 0.7517 |
| 1e-2 | 3.6227 | 1.2070 | 3.6328 |

| Configuration / file | Measurement |
|---|---|
| `hidden_stream.py mirror_alloc`; mean-one levels, noise 8, B=1, 60k; cap/best test-zero/LR | 2/0.3216/3e-4; 8/0.2790/3e-5; 32/0.2351/3e-5; 128/0.2003/1e-5; oracle own-grid 0.2805 at 3e-4 (ratio 1.40x); edge status not recorded for this sweep. |
| Same allocation, 1024/1024 useful | Mirror_alloc 0.797 versus Adam 0.709. |
| `novelty_stream.py`; one shared teacher, sequential versus shuffled | Overall 0.0653/0.0655; 49.5% per-unit past-region alignments harmful; all state-feature absolute r≤0.014; shared-teacher retention/acquisition 0.133/0.027. |
| `novelty_stream.py`; per-region teachers | Retention/acquisition 0.687/0.195. |
| `novelty`, mean-one standardized distance from running preactivation mean; shared teacher | Overall 0.1404 versus Adam 0.1206; also exceeded its own uniform-scalar error (number not recorded); inverted `familiar` 0.1251. |
| `novelty_stream.py`; conflict task, per-arm LR sweeps | Adam 0.6201; learned one-step lookahead unit readout 0.6183 (level 1.000, dispersion 0.000); statebin mirror-twin per state cell 0.6137 (level 1). |
| Same conflict task; precision geometry | Per-unit precision 0.6530 at 8 init-only seeds; previous 4-seed 0.5982 versus Adam 0.6258; precision_shared 0.6210; precision_diag RMS/EMA 0.6645; later 64-task-seed expanded-grid audit: precision−Adam +0.0938 ±0.0165, Adam 0.4022 at LR 9.6e-5 (see `measurement.md`). |
| First precision run, outgoing w2 initialized zero and gated | Error exactly 1.0000; subsequent projection applies to incoming-row input space only. |
| `/tmp/probe_precision.py`; precision geometry, shift=12 | Median max/min eigenvalue ratio 543; 60.7% directions damped below 0.5; lambda=0 baseline bit-identical; median absolute cosine between units' most-damped directions 0.77. |
| `precision_audit.py`; own-step oracle, mean- and dispersion-matched controls, 64 task-resampled seeds | B=8, shift 12: 3.4% headroom, oracle−Adam -0.0046 ±0.0101; B=1: -0.0228 ±0.0052 (5.6%), level-only component 0; shared-teacher relative headroom 17.0%. |
| `hidden_stream.py`; LR effective `L*G*c`, 1024/4 useful, noise 8, B=1, paired seeds 1/2/3, each own grid 1e-5..1e-2, no edge winners | ownfree (G=1, own twin) 0.344/0.174/0.619; input (G=1, layer twin) 0.323/0.175/0.515; uniform (G=1/mean(c), no unit evidence) 0.833/0.470/0.990; ownnull (G=1/mean(c), own twin) 0.178/0.073/0.167; credit (same G, layer twin) 0.177/0.055/0.169; G=1 errors 2.0–3.7x. |
| Same harness; `softhinge_credit` makes evidence invariant to outgoing magnitude, versus `softhinge_alloc`, 6 paired seeds | Credit 0.188/0.054/0.175/0.079/0.092/0.074; alloc 0.199/0.062/0.176/0.092/0.150/0.082; credit lower on 6/6. |
| Same harness; recorded selected arms | Best mechanism 0.174, oracle-mask 0.287, Adam 0.644; linear-stream ratio ~0.04, hidden/linear ~4.5x. |
| `softhinge_free`; G proportional to t²/z² | Error 0.252 versus control 0.185. |
| `dense_boundary.py`; 8 paired seeds, no edge winners | Softhinge_alloc−Adam: 4/1024 -0.530 ±0.086; 128/1024 +0.009 ±0.024; 1024/1024 +0.248 ±0.114; 17/17 +0.008 ±0.014; mirror_alloc at 17/17 level 1.0000 everywhere. |
| Dense midpoint teacher redraw, 17/17 inputs, noise 2/8 | Adam 0.104/0.411; best mechanism 0.101/0.420; oracle-mask 0.093/0.396 (3–11% lower than Adam). |
| `pc_stream.py`; dense net, field fraction 1.0, per-arm LR from 3e-5, 16 paired seeds, corrected `u += pc_lr*du` | B=1 Adam/pc/shuffle 0.5919/0.5880/0.5927, pc−Adam -0.0039, t=-0.35; B=32 0.6530/0.6568/0.6470, delta +0.0038; B=256 0.6905/0.7464/0.7282, delta +0.0560, t=+8.74; batch correction telemetry in `measurement.md`. |
| `hetvar_stream.py bins`; per-parameter `E[delta_j\|a_i bin]`, sums and analytic `p=(S1²−S2)^+/(n*S2)`, no twin/EMA, post-Adam | Homoscedastic delta +0.064 versus shuffle +0.068; hetero -0.016 versus -0.017; signal-scales -0.227 versus -0.210; level 0.001 (~0.1% predictability); pre-Adam level 0.005 absorbed more noise than Adam. |

| Neuronal covariance / meta-descent file and configuration | Sustained clean test MSE / endpoint / execution |
|---|---|
| `unit_bayes_stream_v1.py`; own incoming covariance, input direction, output sensitivity, all-block predicted-output-uncertainty denominator; 65,536 dense stationary observations, paired seed 1, 5256 | Adam LR 0.0003 0.173745/0.130990; neuron prior 1 0.166467/0.136698; layer-shared prior 1 0.173667/0.139919; direction-erased scalar prior 1 0.170189/0.143209; all selections interior; sustained neuron−Adam -4.2%. |
| Same neuron geometry/execution | First-layer normalized directional SD within-state/cross-unit 0.0845/0.0834; shared cross-unit SD 0; five-prior neuron versus seven-LR Adam update/report 2.03/0.90 s, startup 9.70/7.70 s. |
| Corrected covariance shuffle, 5286; own histories, other-unit operators | Neuron/shared/shuffle/scalar sustained 0.166467/0.173668/0.176861/0.170189; previous permutation+inverse-update control reproduced neuron to roundoff. |
| Midpoint teacher switch, 5269; same one paired seed, all hyperparameters interior | Adam/neuron/shared/shuffle/scalar 0.268297/0.250206/0.255640/0.257040/0.263990; neuron versus Adam endpoints 0.246224/0.257118; sustained delta -6.7%. |
| `network_bayes_stream_v2.py`; full 5,377-parameter covariance, five priors | Covariance state 0.539 GiB; 5297, 65,536 observations, paired seed 1, independent validation selection; Adam LR 0.0003, all covariance priors 1, all interior. |
| 5297 Adam | 0.173742 /0.130993 /0.92 s. |
| 5297 neuron-block | 0.166468 /0.136692 /2.08 s. |
| 5297 full cross-neuron | 0.121055 /0.096009 /152.69 s; versus Adam sustained -30.3%, endpoint -26.7%; versus block sustained -27.3%, ~73x training time; peak allocated memory 3.81 GiB. |
| 5297 full covariance, mean-update direction erased | 0.200792 /0.165029 /131.64 s. |
| Full covariance stress 5343; hetero=1 plus midpoint teacher change, 65,536 observations, independent validation, all interior | Adam/block/full/direction-erased sustained 0.291393/0.274385/0.223580/0.309313; full versus Adam -23.3%, endpoints 0.202325/0.293916; full/block update-report 69.96/1.86 s. |
| `unit_metadescent_stream_v1.py`; RMS-normalized one-step hypergradient from next actual noisy loss, own incoming Adam-row positive state multiplier | Older paths, Adam-state and representation derivatives omitted; no clean-target learning; TF32 run rejected by startup parity, FP32 products used. |
| Meta-descent 5280; heteroscedastic midpoint switch | Adam/unit/bias-only/sample-shared/shuffled sustained 0.331977/0.332017/0.332011/0.332038/0.332041; all meta arms select grid floor meta LR 0.0001, base LR 0.0003; zero-meta identity is Adam. |

## Matrix-free predictive momentum transport

`predictive_transport_stream_v1.py` / `predictive_transport_eval_v1.py`;
completed mlq **5746 stationary / 5747 heteroscedastic midpoint switch**, each
limit 1, CUDA, seed 1, 65,536 observations, the same 17-64-64-1 dense network.
Each of six families independently selects LR and beta1 from the same 90
combinations by duration-weighted clean validation, with every family locked
before test scoring. All selected coordinates are interior; no invalid candidates.
The initial 60-candidate runs 5744/5745 triggered the predeclared lower-beta
extension because stress Adam selected beta=.5; adding 0/.1/.3 retained that winner.

| Optimizer | Stationary sustained / endpoint MSE | Switch sustained / endpoint MSE |
|---|---:|---:|
| Adam | 0.173742 / 0.130993 | 0.290787 / 0.293660 |
| Full same-example gradient transport | 0.141771 / 0.099118 | 0.266509 / 0.263615 |
| Prediction-change transport, no gate | **0.131350 / 0.095280** | **0.248811 / 0.250424** |
| Current per-row Jacobian-stability gate, fixed primary | 0.131350 / 0.095282 | 0.248829 / 0.250459 |
| Layer-shared gate | 0.131376 / 0.095341 | 0.248811 / 0.250333 |
| Past-only per-row gate | 0.131360 / 0.095288 | 0.248808 / 0.250365 |

Ungated predictive transport lowers sustained error **24.4% stationary / 14.4%
switch** versus jointly tuned Adam. Its selected LR is .001 in both tasks;
beta1=.9999 stationary / .999 switch, versus Adam .0003 and .9/.5.
Use the current input to evaluate both current and previous pre-update models:
`C=(f_current-f_previous)*J_current`. Add
`beta1*(1-beta1**(t-1))*C` to Adam's biased first moment, leaving its raw-gradient
second moment unchanged. This transports only the historical mass actually present.
The correction is independent of the current noisy label; the full nonlinear
gradient difference additionally contains a noisy residual times Jacobian change.
This is a prediction-based, state-dependent optimizer with linear-sized state:
no covariance, Fisher matrix, sketch, clean-target learning or noise oracle.

**Decision:** retain ungated transport; do not credit an uncertainty-gating gain.
The primary's selected final layer-average q values are nearly one, and current,
shared and history variants nearly coincide. q is deterministic Jacobian stability,
not calibrated uncertainty. These results do not establish autonomous neuron gates.
STORM/MARS-like gradient-transport precedents exist; no invention claim.

Execution contracts caught compiled previous-weight/history aliasing and FP32
cancellation in long-horizon bias correction and near-zero Jacobian ratios.
The final implementation separates compiled next-weight computation from captured
weight/history commits, uses stable expm1 bias masses, and computes forward/Jacobian
teaching signals in FP64 on CUDA for **every** arm; weights/moments/q remain FP32.
Job **5743: 59 passed**, including independent autograd/recurrence, label isolation,
causal history, candidate failure isolation and full state/capture restoration.
Strict local transition audits and exact graph replay pass for every final run.
Failed startup runs are execution diagnostics, not learning outcomes.

Not compute-matched: 90-candidate update/report time is approximately
4.85 s Adam / 9.10 s predictive / 11.60 s gated in stationary, and
4.86 / 9.09 / 11.63 s in stress; compilation and test scoring excluded.
Transport needs two forward/Jacobian evaluations per observation. Batched harness
mutable state is 7,929,826 bytes for 90 candidates, not a production memory benchmark.
One paired task seed, no PPO result and no global-optimality claim. Historical full
covariance remains lower in sustained error (0.121055 / 0.223580), but was not rerun
as a new matched control; the user-directed optimizer line deliberately excludes it.

Reproduction and source/data hashes:
`benchmarks/plasticity/predictive_transport_v1_plan.json`,
`benchmarks/plasticity/predictive_transport_v1_evidence.json`;
full candidate curves, selection locks, checkpoints and root TensorBoard events:
`runs/DenseStream__predictive_transport_v1_{stationary,stress}_bracketed__1__20260909/`.

## FP32 optimizer refinement

`predictive_transport_stream_v2.py` / `predictive_transport_eval_v2.py`;
mlq **5808 stationary / 5809 switch**, limit 1, same paired seed/data and complete
65,536-observation streams. All five arms independently select from the same
**224 LR/beta pairs**, retaining all original 90 candidates. This is adaptive
refinement on the existing task, not a fresh-seed confirmation. All selected
coordinates are interior, with zero invalid candidates.

| Optimizer | Stationary sustained / endpoint MSE | Switch sustained / endpoint MSE |
|---|---:|---:|
| FP32 Adam | 0.173742 / 0.130993 | 0.290773 / 0.293458 |
| FP32 prediction-change transport | 0.130476 / 0.099955 | 0.248811 / 0.250427 |
| Single-forward tangent transport | 0.130475 / 0.099948 | 0.248814 / 0.250425 |
| Implicit predictive transport, declared primary | 0.130247 / 0.099495 | 0.248073 / 0.248800 |
| Scale-one pseudo-Huber transport | 0.135444 / 0.099491 | 0.233763 / 0.228071 |

FP32 is sufficient here: the coarse-grid v2 predictive scores differ from v1's
FP64 teaching-signal scores by less than 4e-7 sustained MSE. Removing the inert
Jacobian-ratio gate, unused previous Jacobian and telemetry reduces matched
90-candidate update/report wall time from approximately 9.1 to 3.45 seconds.
This is a combined precision/computation/state improvement, not an isolated
arithmetic microbenchmark. Tangent uses `C=<J,theta-theta_previous>*J` and removes
the previous forward as well, with nearly unchanged risk.

**Decision:** tangent is the simpler squared-loss speed/accuracy candidate:
24.9% / 14.4% lower sustained error than jointly tuned Adam. Implicit solves a
current-example rank-one proximal quadratic in the Adam metric without storing
curvature matrices; its small error gain costs extra work. Robust transport
reduces switch error 19.6% versus Adam and 6.0% versus squared predictive transport,
but increases stationary error 3.8% versus squared predictive transport. It changes
the objective to pseudo-Huber and is not a universal replacement or calibrated
uncertainty model.

Selected-only K=1 CUDA-event replay, excluding compilation/validation/checkpoint
IO, takes approximately 1.52–1.56 s Adam, 2.25–2.26 s predictive, 2.08 s tangent,
2.55–2.58 s implicit and 2.11–2.12 s robust per full stream. This includes online
telemetry and host graph submission gaps; independently compiled K=1 timing does
not establish exact trajectory equality with the K=224 accuracy run.

Independent review caught an inherited capture-audit hole: intersecting validity
masks could conceal a compiler-only nonfinite transition. Job **5806** reproduces
both mismatch directions; both now raise before candidate masking and restore all
state. Job **5807: 38 passed** covers v2 optimizer/evaluator contracts. Original
local tolerances and exact production graph replay remain unchanged.
Evidence: `benchmarks/plasticity/predictive_transport_v2_{plan,evidence}.json`;
full artifacts: `runs/DenseStream__predictive_transport_v2_{stationary,stress}_refined__1__20260909/`.

## Fused Triton execution

`predictive_transport_stream_v3.py` / `predictive_transport_eval_v3.py`;
mlq **5812 stationary / 5813 switch**, limit 1, same 224-pair grids and data.
One CTA owns each candidate's counters and processes 16 sequential observations
inside a captured kernel, retaining weights/moments locally. No learning-rule,
loss-scale or precision change from v2. The small temporary v2 eager oracle exists
only during capture audits; production is fused FP32 Triton.

All ten selected LR/beta pairs are unchanged; maximum selected sustained-MSE
difference from v2 is **5.21e-6**, with zero invalid candidates. Job **5811:
38 passed**, covering independent trajectories, separate candidate indices,
fused-block ordering, exact graph replay, NaN isolation and fault-injected rollback.

The initial 16-warp fused kernel is **not uniformly faster**:

| Method | V2 K=1 fullstream replay | V3 K=1 replay | Decision from this comparison |
|---|---:|---:|---|
| Adam | 1.52–1.56 s | 1.28–1.29 s | Lower single-candidate latency; no grid/report gain. |
| Predictive | 2.25–2.26 s | 2.88–2.91 s | Slower; retain v2 execution. |
| Tangent | 2.08 s | 1.68–1.69 s | 1.23–1.24x single-candidate speedup; grid/report approximately neutral. |
| Implicit | 2.55–2.58 s | 3.38–3.41 s | Slower; retain v2 execution. |
| Robust | 2.11–2.12 s | 2.89 s | Slower; retain v2 execution. |

Thus launch fusion alone does not establish the dominant bottleneck. A separate
fullstream 4/8/16-warp experiment, **5817**, measures register/spill/shared-memory
use alongside K=1 latency and K=224 throughput; execution choices must follow
measurements rather than treating every fused kernel as an improvement.
Evidence: `benchmarks/plasticity/predictive_transport_v3_{plan,evidence}.json`;
full artifacts: `runs/DenseStream__predictive_transport_v3_{stationary,stress}__1__20260909/`.

Occupancy job **5817 completed all 30 fullstream cases**: five arms × K=1/224 ×
4/8/16 warps, with fixed learning configurations and capture parity enforced.
Four warps is consistently slower. Sixteen remains fastest for Adam/tangent;
eight improves predictive/implicit/robust relative to their 16-warp fused versions,
but their K=1 times (2.27/2.75/2.25 s) still do not beat v2 (2.26/2.58/2.12 s).
No new warp-tuned source version is retained. The compiler reports substantial
register spilling—for example tangent's 4/8/16-warp spill counts are 912/312/300—
so a universal launch-overhead explanation is insufficient.

**Retained frontier:** fused v3 tangent for lower single-candidate squared-loss
latency; v2 robust when its measured heteroscedastic accuracy tradeoff is wanted.
Keep v2 execution for predictive/implicit/robust. No aggregate tangent throughput
win or universal robust-loss improvement is claimed.

The occupancy controls preserve the locked candidates' endpoint validation closely,
but not every poor grid trajectory: the LR=.03/beta=.99999 corner changes greatly
with reduction order while remaining finite and much worse than the selected risk.
Those candidates and all endpoint values remain in the evidence; none is silently
filtered and no warp-specific accuracy selection is made.
Profiling artifacts, TensorBoard and source snapshot:
`runs/DenseStream__transport_warps_v3__1__20260909/`.

## Redesigned optimizer proxies and fresh confirmation

The predictive-transport PPO failure in `ppo_runs.md` invalidates promotion from
the old dense-stream result. That proxy principally rewarded averaging scalar,
single-example label noise; its Adam comparator left beta2 and decay fixed.
The PPO pilot separately used a fixed LR and only two momentum values. Neither
comparison established a tuned advantage in moving, reused policy objectives.

`optimizer_proxy_eval_v4.py` through `optimizer_proxy_eval_v8.py` separate:

| Case | Fresh observations | Batch | Passes per batch | Objective |
|---|---:|---:|---:|---|
| Online | 65,536 | 1 | 1 | Stationary noisy nonlinear regression |
| Drifting reuse | 65,536 | 64 | 8 | Interpolating nonlinear teachers and shifting inputs |
| Clipped bandit | 262,144 | 256 | 8 | Noisy quadratic reward, signed clipped Gaussian-policy objective |

The bandit freezes old log-probabilities and advantages within each batch and
draws fresh on-policy actions next round. It has no critic, GAE, multidimensional
Beta policy or environment dynamics: it is not a MuJoCo surrogate with established
ranking validity. Clean teacher error is reporting-only. Noisy validation chooses
the complete trajectory's hyperparameters before held-out scoring; checkpoints
cannot be selected by test performance.

V8 provides 2,036 identical explicit LR/beta1/beta2/decay/head-rate configurations
per family. Conditional head-rate refinements add 135 configurations on the
bandit and 512 online. These are adaptive development searches, not untouched
confirmation. The Adam decay-zero subset is retained separately from AdamW.
Boundary refinement is not a proof of globally optimal hyperparameters.

The reuse-clock transport hypothesis is rejected: freezing variance and counting
only fresh batches did not solve the policy regression, and an extreme
full-transport configuration failed the numerical capture audit. The v7 probe is
retained under `benchmarks/plasticity/optimizer_proxy_v7_capture_probe.py`, not
treated as a passing test. V8's eight-step polar online search also failed its
strict FP32 transition audit (job **5883**); the later online refinement excludes
that family. Its original partial JSON still says `training`; the recorded
terminal queue state is authoritative. No missing score is imputed.

### Retained mechanism

`optimizer_proxy_model_v8.py`: for hidden weight matrices, form bias-corrected
Nesterov momentum `q = beta1*m_hat + (1-beta1)*g`, apply three or five quintic
Newton-Schulz steps, then normalize the direction to Frobenius norm
`0.2*sqrt(out*in)`. Hidden biases and the scalar output head use AdamW; the head
has its own LR ratio, equally available to the AdamW comparator.

This is Muon-inspired matrix optimization, not a novel optimizer claim or a
per-sample neuron-state gate. The Gram matrices are temporary numerical algebra,
not learned sample covariance/Fisher state. The controls normalize either the
untransformed Nesterov direction (`matrix_rms`) or Adam-preconditioned direction
(`adam_rms`) to the same hidden-weight direction RMS. Independent LR/decay tuning
means actual parameter displacements are not identically norm-matched.

### Locked fresh-task result

Plan: `benchmarks/plasticity/optimizer_proxy_confirmation_plan.json`.
Job **5895** succeeded: seed 1, new RNG namespace 100, full original horizons,
five fixed methods per case, no retuning. Configurations were locked by minimum
noisy development validation across the original and head-refinement searches
before generating new teachers, initialization, training data or test draws.
All methods within a case share the exogenous draws. The three cases share the
underlying fresh teacher/initialization realization; they are not independent
training-seed replicates.

Sustained risk is the mean of **16 equally spaced right-endpoint** held-out clean
excess risks, not a continuously observed integral:

| Method | Online risk | Drifting-reuse risk | Clipped-bandit risk |
|---|---:|---:|---:|
| Tuned AdamW | 0.221751 | 0.341913 | 0.260806 |
| Five-step polar, predeclared primary | **0.184554 (-16.77%)** | **0.317653 (-7.10%)** | **0.237358 (-8.99%)** |
| Three-step polar | 0.185733 (-16.24%) | 0.321347 (-6.01%) | 0.240212 (-7.90%) |
| Nesterov RMS control | 0.229843 (+3.65%) | 0.344762 (+0.83%) | 0.272902 (+4.64%) |
| Adam RMS control | 0.219014 (-1.23%) | 0.342349 (+0.13%) | 0.267256 (+2.47%) |

The directional gain survives fresh draws and both RMS controls. This supports
matrix geometry over scalar normalization in these tasks, not the original
family's autonomous-neuron interpretation. There is no across-seed significance
claim, and the **>=20% fresh-confirmation target was not met**.

| Method | Online training seconds | Drifting reuse | Clipped bandit |
|---|---:|---:|---:|
| AdamW | 5.236 | 0.773 | 0.858 |
| Five-step polar | 14.075 | 1.860 | 1.995 |
| Three-step polar | 11.081 | 1.466 | 1.561 |

These K=1 wall times include host data copies and checkpoint IO, exclude capture
startup and held-out scoring, and are not aggregate-throughput measurements.
Five-step polar costs **2.32–2.69x AdamW**; three steps save approximately 21%
versus five, with slightly higher risk in every case. No equal-compute advantage
is established. Retain five steps for the measured accuracy frontier and three
for the lower-cost alternative; do not promote either to PPO on these results.

### Verification and retained evidence

- **5880: 34 passed**, no failures/errors/skips. Independent gradient/AdamW
  references, signed clipping, transport decomposition, moment clocks, matrix
  geometry and head-rate contracts; compiled CUDA execution.
- **5881/5882** completed the 2,036-configuration drift/bandit searches;
  **5889/5890** completed head-rate refinements. **5883** failed as described.
- **5895** completed all 15 locked full trajectories. All 10 recorded source
  hashes still matched at analysis; all 240 checkpoint files and complete
  observation/update counts were verified.
- All these jobs used `maxParallelRuns=1`, priority 0, one attempt, no retries.
  Time limits: contracts 20m, original searches 60m, refinements/confirmation 45m.
  No short-horizon performance culling; nonfinite grid candidates are marked
  invalid, while numerical-contract failures stop the job.

Complete queue records, development selections, confirmation curves, configs,
source checks and checkpoint hashes:
`benchmarks/plasticity/optimizer_proxy_v8_evidence.json`.
Raw confirmation:
`runs/OptimizerProxyConfirm__v1__1__1788993862680122315/results.json`.
