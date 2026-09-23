# Vector segment TD v15

Source: `cleanrl/ppo_continuous_action_vector_segment_td_v15.py`.

## Focused change from v14

V14's one-step vector TD learned slowly and declined after a peak near 594. This follow-up changes **only the observed TD segment**, from one step to up to 32 real transitions. The critic architecture, fixed Beta(2,2) feature reference, state-dependent joint-action kernels, horizon queries, loss normalization, eighteen full-batch fitting steps, learning rate 0.0003, and actor update remain the same as [v14](vector-td-response-v14.md).

The target at each queried horizon h is the signed reward-component sum over the first min(h,m) actual transitions, plus the frozen expected vector critic at the actual endpoint if h exceeds m. Here m is clipped by 32, the next actual episode end, and the rollout boundary. Queries shorter than the common segment have fully observed targets and no continuation. An actual episode end suppresses bootstrap; a nonterminal rollout boundary retains it.

Endpoint indices select the **last consumed transition's next observation**, so the continuation state and age refer to the same instant. FP64 hierarchical prefix sums construct all observed vectors over the full rollout, without fitting minibatches. All continuation predictions and targets are detached and computed once before any fitting.

This is a single fixed-length multi-step TD target, with gamma=1 and finite-horizon termination. There is no lambda mixture, GAE, replay, imagined trajectory, scalar value bottleneck, or repeated target sweep. The additional observed steps carry real delayed consequences into each target. They also increase trajectory sampling noise; improvement is a hypothesis, not a guarantee.

## Verification and experiment

Independent review found no blocker in the observed sums, episode/rollout boundaries, shared endpoints for short queries, or frozen-target construction. Numerical job **7582**, parallel limit **1**, time limit **25 minutes**: **7 CUDA checks passed**. They include explicit-sum reference cases, arbitrary scan lengths, terminal versus rollout-cut behavior, one-step equivalence, compiled fitting, and isolated production-shape scan/output-lifetime checks. The unchanged analytic critic/actor machinery also passed v14's 12 contracts.

Fresh seed-1 HalfCheetah job **7583**, parallel limit **1**, time limit **60 minutes**, depends on successful 7582. Standard batch: 16 × 2048 transitions, environment threads 2, KL ceiling 0.03. Nominal 8M transitions; no checkpoint resumption. Report job **7584**, parallel limit **1**, time limit **5 minutes**, runs after training reaches a terminal queue state.

Source SHA256: `d99628d799b39027a758cbb7fc27689f4e0aae5894678f22af6624d6c88dd526`.

## Results

**Negative result.** Job **7583** was cancelled after persistent low return. Graceful shutdown saved a cancelled result and checkpoint at **5,488,256 transitions**, with last-100 episode return **387.36**. The peak was **416.04 at 507,520 transitions**. Return oscillated mostly around 200–400 for several million transitions, far below the established references. This is not a completed 8M score.

| Matched budget | v13 Monte Carlo | v14 one-step TD | v15 32-step TD |
|---|---:|---:|---:|
| 1M | 438.12 | -25.99 | -0.32 |
| 2M | 1,330.08 | 116.06 | 275.09 |
| 4M | 1,235.44 | 551.90 | 305.73 |

All **167 actor updates** were accepted. Mean KL was **0.01889**, and median CG relative residual was **0.00317**. Late updates frequently chose much less than the 0.03 ceiling. Neither a more accurate natural-direction solve nor a smaller accepted KL produced strong returns here.

There is direct evidence of a critic-fitting problem beyond long-horizon bootstrap uncertainty: the final **one-step-horizon vector prediction RMSE was 3.056**, versus **0.395** at v14's endpoint. That one-step target contains only observed immediate rewards and **no continuation prediction**, even in v15. The remaining-horizon TD residual RMS was 4.351. Endpoint states and policies differ, so these RMSE values are descriptive and not a matched-distribution ablation, but v15's immediate prediction error is real and cannot be explained by bootstrap error in that target.

Longer observed segments did not recover v13's early performance or produce sustained improvement. The experiment weakens the hypothesis that one-step temporal propagation alone explains v14's failure. It does not isolate representation limits, coupled horizon fitting, target-scale/optimization effects, or bootstrap bias in the longer targets. No claim that vector TD itself is disproven follows from these two finite-model experiments.

Report job **7584** succeeded. [Numeric results](vector-segment-td-v15-results.json), [comparison plot](vector-segment-td-v15-results.png), [vector plot](vector-segment-td-v15-results.svg). Raw evidence: `runs/HalfCheetah-v4__vector_segment_td_v15_8M__1__1789583524286086634/`. No long-run extension; preserve both TD versions as negative results.
