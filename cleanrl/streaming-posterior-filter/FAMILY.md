# Streaming Posterior Filter

## Objective

Learn from temporal streams with one observation at a time, bounded optimizer
overhead, and no dependence on batch statistics or a terminal learning-rate
schedule. The end-state is posterior plasticity over slow parameters: uncertain
weights move quickly, supported weights consolidate, and process variance keeps
the learner able to track nonstationarity.

The first benchmark carrier is the exact PPO program used by
`HalfCheetah-v4__iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_statedynlr_v2_centered__1__1787361543`.
The family removes that program's controller and replaces Adam with one
streaming-native update. Rollout, model, loss, gradient clipping, and seed stay
fixed so the optimizer thesis is falsifiable.

## What the filter claim means

A literal extended Kalman filter over a neural network requires a likelihood
residual, its parameter Jacobian, and a curvature model. PPO exposes a noisy,
clipped policy/value gradient from a moving objective instead. Calling that an
exact weight posterior would be false.

PPO does not supply those quantities. SPF v1 therefore filters the object PPO
actually exposes: a clipped stochastic gradient. Its explicit approximation is:

- each matrix row is one isotropic latent gradient state;
- `M` is the filtered gradient mean;
- `P`, `R`, and `Q` are latent-gradient posterior, observation, and process
  variance, not weight uncertainty or curvature;
- Adam's raw second moment `V` remains as the controlled coordinate
  preconditioner;
- the scalar row gain is an assumed-density Student-t Kalman update.

This is an approximate empirical-Bayes filter with named boundaries, not a
controller network relabeled as Bayesian inference. A temporal supervised model
with an explicit likelihood can later implement the stronger weight-posterior
claim.

## v1 update

For a row of width `d` and streaming gradient `g_t`:

```text
k0         = 1 - beta1
Q_t        = (k0^2 / beta1) R_(t-1)
Pminus_t   = P_(t-1) + Q_t
e_t        = g_t - M_(t-1)
delta_t    = sum(e_t^2) / (Pminus_t + R_(t-1))
w_t        = min(1, (nu + d) / (nu + delta_t))
K_t        = w_t Pminus_t / (w_t Pminus_t + R_(t-1))
M_t        = M_(t-1) + K_t e_t
P_t        = (1-K_t) Pminus_t
R_t        = beta2 R_(t-1)
             + (1-beta2) [rowmean((sqrt(w_t)(g_t-M_t))^2) + w_t P_t]
V_t        = beta2 V_(t-1) + (1-beta2) g_t^2
theta_t    = theta_(t-1) - temperature M_t / sqrt(Vhat_t)
```

`Q/R` is fixed by the nominal Gaussian responsiveness `k0`. This is
gradient-scale invariant: scaling one row's gradients scales `M`, `P`, `R`,
`Q`, and `V` consistently while leaving `delta`, `w`, `K`, and the normalized
parameter direction unchanged. It is a designed forgetting prior, not a learned
physical process variance.

`w` is one-sided. Correlated or near-identical PPO minibatches never receive
more precision than a Gaussian observation, while an extreme innovation gets a
smaller gain. The Student-t update is an assumed-density/IRLS approximation;
mixing only the measurement variance does not produce a closed-form Student-t
Kalman posterior.

v1 deliberately does not infer `Q` from the same innovation used to reject an
outlier. That feedback would turn `large innovation -> large Q -> large gain`
and cancel the robustness mechanism. No row competes with another.

The implementation stores `P` and `R` as float64 row vectors, computes row RMS
with max-rescaling, and stores `sqrt(V)` with a `hypot` update. This is
algebraically the update above but remains finite for every finite float32
gradient. There is no absolute epsilon in the update: nonzero gradient
rescaling stays equivariant across the representable float32 range, while exact
zero uses an update of zero.

## State and cost

For a parameter tensor with shape `(out, ...)`:

- full-size filtered gradient `M`;
- full-size Adam-compatible square-root second moment `sqrt(V)`;
- float64 row vectors for observation noise `R`, posterior variance `P`, and
  last diagnostics.

Memory is Adam's two parameter copies plus small row state. Work is linear in
parameter count. The implementation performs no per-minibatch host
synchronization; scalar diagnostics synchronize once per PPO iteration.

Matrix rows map to perceptrons. A vector or scalar tensor uses one filter for the
whole tensor. Sparse gradients are outside the v1 contract.

## Two-timescale architecture

The family separates three mechanisms instead of forcing one optimizer state to
do all jobs:

1. **Slow consolidation:** SPF over trainable parameters. This is v1 and must
   prove useful before architectural changes are admitted.
2. **Robust predictive likelihood:** supervised temporal models should expose
   their actual predictive residual and variance to a Student-t likelihood.
   That is more informative than inferring data corruption from gradients. PPO
   v1 cannot add it without changing the objective, so the optimizer uses robust
   gradient innovations as the weaker available signal.
3. **Fast state:** token/bar models may add DeltaNet-style key-addressed fast
   weights updated in the forward pass. Fast state absorbs transient context;
   SPF consolidates persistent structure. Fast weights are not part of the first
   PPO isolation because coupling an architecture and optimizer would make a
   result uninterpretable.

A later language/bar carrier must be single-sample-native end to end: causal
state persists across items; neither optimizer statistics nor the predictive
likelihood may read future examples or require batch moments.

## Invariants and required diagnostics

Every version must preserve:

- positive finite `P`, `R`, and `Q` by stable information-form updates;
- independent row updates; no centering, softmax, fixed budget, or other
  zero-sum plasticity;
- one-sided bounded influence through the Student-t precision weight;
- scale-invariant fixed `Q/R` in v1;
- no default learning-rate annealing;
- no controller network or cross-step autograd graph;
- no optimizer-step `.item()` or other CUDA synchronization.

Required TensorBoard series:

- `spf/gain_mean`, `spf/gain_std`, `spf/gain_p10`, `spf/gain_p90`;
- `spf/student_weight_mean`, `spf/outlier_fraction`;
- `spf/p_over_r_mean`, `spf/innovation_per_dim_mean`;
- `spf/prior_cosine_mean`, `spf/update_rms_mean`,
  `spf/relative_update_mean`;
- `debug/actor_clip_frac`, `losses/approx_kl`, and the carrier's `charts/SPS*`.

A claimed noise-robust result must eventually include a controlled temporal
stream with switching regimes and injected heavy-tailed corruption. Clean
stationary performance alone cannot establish the family thesis.

## Stable prequential metrics

`charts/episodic_return` is reserved for actual environment episode reward. The
synthetic regression and language carriers have no episodes and must not write
that tag.

Their fixed-window evaluation metrics are:

- regression: raw `stream/latent_prequential_mse` plus
  `stream/explained_energy = 1 - MSE / latent target energy`;
- language: raw `language/clean_nll` plus
  `language/code_length_skill = 1 - NLL / log(256)`.

Predictions are scored before the sample update. Metrics use the clean target,
not the corrupted observation or training loss. The PPO carrier alone reports
Gymnasium's real episode reward under `charts/episodic_return`.

## Benchmark contract

Exact plain-PPO reference:

- 1576 at 500k;
- 3079 at 1M;
- 4507 at 1.6M;
- 5012 at 2M;
- 6716 at 4M;
- 8278 in the 8M window; final-20 8455.

Exact centered carrier supplied for this family:

- 1435 at 500k;
- 3482 at 1M;
- 5310 at 1.6M, where the run ended;
- final cumulative throughput 2221 SPS.

For v1, kill on non-finite filter state, sustained gain collapse, or more than
10% below plain PPO at two consecutive matched checkpoints after 1M. The
wall-clock bar is at least 85% of the centered carrier's throughput after
startup. A step-count win that loses matched wall clock is not a win.

## Known risks

- `P` is uncertainty in a latent gradient state, not weight confidence. The PPO
  carrier cannot prove consolidation or a weight-posterior interpretation.
- PPO reuses one rollout for several epochs. Innovation agreement can mean
  repeated data, not new evidence; one-sided `w` and fixed `Q/R` limit false
  certainty but cannot restore independence.
- Shared trunk gradients combine actor and critic after separate clipping. A
  single row filter sees their sum; conflicting objectives can look like
  observation noise and lag the update.
- Student-t weighting can reject a rare useful policy gradient, especially when
  PPO's clip boundary creates a legitimate abrupt direction change.
- The exact carrier's actor norm clip is already binding. v1 can test directional
  robustness, not claim to solve raw heavy-tailed gradient magnitude.
- The global temperature remains a real scale parameter. Calling it anything
  else would not remove it; the streaming claim is that it stays constant while
  state-dependent gain comes from the filter.

## Versions

- `ppo_continuous_action_spf_v1.py` — replaced Adam and the centered activation
  controller with a row-isotropic Student-t latent-gradient filter. Global
  temperature `3e-4`, nominal gain `0.1`, square-root Adam `V`, fixed
  `Q/R=0.01111`, `nu=5`, no annealing. Job 3232 cancelled at 1.76M: 2604 at
  1M and 4006 at 1.5M versus plain PPO 3079/4356 and the supplied centered
  carrier 3482/5103. Throughput 1742 SPS versus 2221. The direction-filter
  approximation is both slower and worse; it is not the family's slow-weight
  solution.
- `streaming_switch_regression_v1.py` — first truly single-sample carrier with
  an actual supervised residual/Jacobian. Preliminary seed-1 combined-condition
  result: robust diagonal filter 0.00217 clean prequential MSE at 10,179 steady
  SPS; AdamW 0.0134 at 3,001 SPS; IDBD was 0.00820 while still running near the
  endpoint. This is promising but not attribution: online `R`, adaptive `Q`,
  robust weighting, and recursive curvature were coupled.
- `streaming_switch_regression_v2.py` — preregistered repair: exact-norm sin/cos
  temporal features, random hidden switch times, clean latent scoring, fixed
  oracle `R` in primary arms, one-sided Student weight, Joseph diagonal update,
  adaptive/fixed-Q and Gaussian-filter ablations, and stationary/switching x
  clean/outlier factorial. Stable metrics are latent MSE and explained energy.
  On clean switching data at seed 1, tuned IDBD (`alpha_0=0.3`) reached 0.00664
  MSE at 13,173 SPS. The fixed robust filter reached 0.00932 with `Q=1e-3` at
  9,854 SPS; AdamW reached 0.05145 with `lr=0.003` at 2,809 SPS. The original
  local empirical-Bayes `Q` controller regressed to 0.06814. Broader fixed-Q,
  Gaussian, contaminated, and matched-wall-clock runs remain in flight.
- `streaming_switch_regression_v3.py` — temporal-hypergradient `Q` controller.
  Forward sensitivities correctly matched a two-step finite difference, but all
  clean-switching runs regressed: 0.04185 best MSE at 7,360 SPS versus 0.00932
  for fixed `Q` and 0.00664 for IDBD. Correct hypergradients did not solve the
  credit horizon or adaptation-speed problem.
- `streaming_switch_regression_v4.py` — Bayesian process-noise mixture. Three
  robust diagonal experts (`Q=1e-8,1e-5,1e-3`) are selected independently per
  output by hazard-mixed Student-t predictive evidence. At seed 1 it reached
  0.000120/0.000122 MSE on stationary clean/outlier and 0.003745/0.003787 on
  switching clean/outlier at 7.2–7.5k SPS. Switching post-change 2k MSE was
  0.0352, matching the fixed high-Q expert while cumulative MSE beat tuned IDBD
  by 44%. Isolated corruption had negligible lasting cost.
- `streaming_fastweight_lm_v1.py` — byte-level causal GRU plus DeltaNet-style
  rank-one fast state, robust contamination categorical likelihood, Student-t
  latent/write likelihoods, and one-token updates without batching, replay, or
  BPTT. The 83,268-parameter seed-1 carrier makes 300,000 updates over 7.318
  passes of a 40,997-byte training prefix and evaluates on 4,096 never-trained
  suffix bytes. The first robust run reached 2.309 NLL but normalized its gate
  with pre-write error while adapting scale from post-write error; it is
  superseded. Plain fast memory reached 2.106 held-out NLL; the slow robust GRU
  reached 2.084; the corrected scale-consistent robust fast-memory rerun reached
  2.084 NLL (8.036 perplexity, 47.5% top-1, 77.1% top-5). Fast state therefore
  matched but did not beat the slow robust carrier and has not earned a language
  win.
- `ppo_continuous_action_spf_fastmemory_v2.py` — Adam slow weights plus a causal
  per-environment robust Delta dynamics memory. PPO stores each behavior-time
  read, preserving replay ratios; memory writes only after valid transitions.
  At 8M HalfCheetah steps it reached 8,852 at the matched window and 8,821 over
  the final 20 episodes at 4.1k SPS, versus plain PPO 8,278/8,455. This is a
  hybrid: the memory is streaming, but slow weights reuse 32,768-transition
  rollouts for 10 epochs.
- `ppo_continuous_action_spf_online_v3.py` — honest single-use control: 64
  independent streams, one-step TD, one optimizer update per fresh transition,
  no repeated epochs, replay, advantage batch normalization, or BPTT.
  Runtime validation rejects overrides that introduce rollout accumulation,
  repeated updates, batch shaping/normalization, multi-horizon targets, or a KL
  leash.
  It is an online actor-critic because a one-use ratio is exactly one. At 1M
  HalfCheetah steps it reached 1,132 versus 3,096 for the batched hybrid, with
  2.27k versus 3.52k SPS and a 21.9 versus 2.33 actor gradient norm. The first
  run was cancelled at 1.15M under the preregistered 1–2M underperformance
  rule. Single-use updates removed reuse but sharply increased policy-gradient
  variance and did not preserve PPO's score.
  Unculled seed-1 job 3381 confirmed the failure mode over 8M steps: returns
  repeatedly crossed below zero and recovered (1.5M -293, 2.5M 1,297, 3M
  -179, 6M 1,196, 7M -44), ending at 1,233 final-20 and only 796 average
  return at 2.11k SPS. The issue is long-horizon instability, not merely slower
  initial learning.
- `ppo_continuous_action_spf_online_robustadv_v4.py` — divided each TD advantage
  by a causal per-stream robust scale. Job 3342 was cancelled at 1.02M after a
  500k collapse and recovered to only 925 at 1M. Magnitude equalization behaved
  like sign-SGD and discarded useful TD confidence.
- `ppo_continuous_action_spf_online_robustgate_v5.py` — retained TD magnitude
  and applied only a Student-t outlier gate. Job 3344 reached 1,114 at 1M but
  oscillated to 653 at 1.75M and was cancelled at 1.98M. Outlier rejection alone
  did not stabilize the one-use policy gradient.
- `ppo_continuous_action_spf_online_causaladvnorm_v6.py` — used lagged
  per-stream robust location and scale, never current-batch statistics. Job
  3354 reached 757 at 1M at 1.39k SPS and was cancelled at 1.09M. The actor norm
  clip was already binding, so scalar normalization changed little while lagged
  centering damaged direction. Missing batch normalization is not the primary
  online bottleneck.
- `ppo_continuous_action_spf_online_eligibility_v7.py` and
  `ppo_continuous_action_spf_online_shorttrace_v8.py` — exact Beta actor-head
  eligibility traces with lambda 0.9 and 0.5 respectively, leaving the V critic
  and trunk one-step. v7 job 3368 improved the 1M score to 1,232 and reached
  1,570 at 2M, then collapsed below zero by 3.26M. v8 job 3373 completed 8M but
  repeatedly collapsed and recovered, ending at 854. Temporal actor credit
  helps early learning but stale score directions destabilize the policy.
- `ppo_continuous_action_spf_online_fastactor_v9.py` and
  `ppo_continuous_action_spf_online_epfastactor_v10.py` — per-stream local
  state-to-Beta-logit plasticity. v9 job 3375 reached 1,283 at 1M, then collapsed
  by 2M and ended at 1,047 after 8M. v10 made the offset smooth, used exact
  behavior-time Jacobians, and reset it at episode boundaries; job 3378 lost the
  early gain and still collapsed near 2.3M. Direct policy plasticity is useful
  early but is not long-horizon stable.
- `ppo_continuous_action_spf_online_fastvalue_v11.py` — added a causal robust
  linear V(s) residual only to the actor TD error while keeping the slow
  distributional V target raw and Q-free. Job 3380 was near v3 at 1M
  (1,006 versus 1,132), then collapsed immediately and was cancelled at 1.09M.
  A local value residual did not remove the actor instability.
- `ppo_continuous_action_spf_online_targetvalue_v12.py` — bootstrapped the
  one-step actor and raw V target from a frozen Polyak V(s) copy (`tau=0.005`),
  updated only after the single optimizer step. Job 3384 briefly reached 1,045
  at 1.5M and 1,119 at 2.5M, but fell to -292 at 3M and stayed negative through
  4.1M, where it was cancelled. Explained variance remained 0.82 at 4M and the
  actor gradient norm fell to 3.16, so inaccurate moving bootstraps and raw
  gradient magnitude are not sufficient explanations for the policy collapse.
- `ppo_continuous_action_spf_online_posteriorinfo_v13.py` — replaced direct
  policy perturbations with a causal information-gain bonus from action-
  conditioned Bayesian dynamics. Job 3386 was manually stopped at 2.62M after
  reaching 762/1,383/1,495 at 1M/2M/2.5M. This was the strongest stable online
  curve, but only 1.31k SPS and still far below the batched hybrid.
- `ppo_continuous_action_spf_online_separatedtarget_v14.py` — separated actor
  and target-V updates to remove shared-gradient interference. Its run was
  stopped at 64k during an unrelated CPU saturation incident, so it has no
  benchmark verdict.
- `ppo_continuous_action_spf_online_epistemictoken_v15.py` — injected a smooth
  episode-persistent posterior token into the Beta policy. Job 3409 reached
  861 at 1M, but the token offset saturated near its 0.5 cap and throughput fell
  to 784 SPS. It was cancelled at 1.02M.
- `ppo_continuous_action_spf_hybrid_posteriorinfo_v16.py` — moved the v13
  action-conditioned posterior bonus onto the strong replayed PPO carrier.
  Job 3424 was cancelled at 112k: its 374 SPS made the full-covariance posterior
  computationally noncompetitive.
- `ppo_continuous_action_spf_hybrid_robustsurprise_v17.py` — reused the v2
  fast dynamics memory to form an actor-only Student-t robust, lagged-
  standardized surprise bonus. Job 3426 reached 1,580/3,279/6,199/7,572 at
  500k/1M/2M/3M, beating v2 by 9%/6%/47%/38%, and averaged 6,701 across the
  run versus 6,055 for v2. The gain narrowed by 6M and reversed at 8M:
  matched-window 8,358 and final-20 8,745 versus v2's 8,852 and 8,821.
  Persistent unit-scale exploration improved sample efficiency but impeded
  final policy refinement; throughput also fell to 2.76k SPS.
- `ppo_continuous_action_spf_hybrid_robustprogress_v18.py` — removed v17's
  variance normalization so stationary centered surprise anneals to zero.
  Job 3432 tracked v2 closely: 2M/4M/6M matched-window returns were
  4,654/6,877/7,814 versus 4,211/6,765/8,114, then it ended at 8,506
  matched-window and 8,754 final-20. Its 5,974 run-wide average was below both
  v17 and v2. Raw centering annealed too quickly to preserve v17's acceleration
  and still did not recover v2's final score.
- `ppo_continuous_action_spf_hybrid_fractionalsurprise_v19.py` — divides
  centered robust surprise by the square root of its lagged standard deviation,
  interpolating between v17's persistent unit normalization and v18's rapidly
  vanishing raw progress. Job 3438 is the first strict seed-1 improvement over
  v2 at every main checkpoint: 1M/2M/3M/4M/6M/8M matched-window returns were
  3,250/5,972/7,232/7,989/8,626/9,223 versus
  3,096/4,211/5,475/6,765/8,114/8,852. It reached 9,218 final-20 and a 6,892
  run-wide average versus v2's 8,821 and 6,055. Throughput was 3.07k SPS versus
  v2's 4.29k, so the 2M sample win is also a matched-wall-clock win, while the
  endpoint gain costs throughput. Fractional self-annealing preserved v17's
  exploration acceleration and restored late extrinsic refinement.
- `ppo_continuous_action_spf_hybrid_fractionalsurprise_fused_v20.py` — keeps
  v19's objective but compiles pure functional fast-memory read/update kernels
  and writes final model/fast-state artifacts. CUDA tests prove eager v20 is
  stepwise identical to v19 and compiled state/read recurrence matches eager.
  Job 3448 was externally cancelled at 1.89M, but was already ineligible:
  500k/1M/1.5M returns were 1,340/2,755/3,864 versus v19's
  1,472/3,250/4,753, while throughput fell from 2.71k to 2.20k SPS at 1.8M.
  Default-mode Inductor compilation increased rather than removed recurrent
  step overhead; v20 should not be resumed or generalized.
- `ppo_continuous_action_spf_fastmemory_repro_v3.py` — training-identical v2
  reference carrier with final eval-compatible model and fast-state/optimizer
  sidecar writes. It exists only for the queued multi-seed/environment matrix.
