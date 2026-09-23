# Unified vector TD v16

Source: `cleanrl/ppo_continuous_action_vector_unified_td_v16.py`.

## One predictive function

V16 removes the distinctions between horizon-specific predictions and separate state-only/action-response networks. A single shared trunk produces one learned coefficient tensor and the state-dependent shapes of its integrable action features. Its vector Q predicts the remaining episode's progress and six signed actuator-cost contributions from state, action, and episode age.

For HalfCheetah the coefficient tensor has shape `[7 components, 48 features]`. The features comprise one constant, twelve fixed-reference Beta score coordinates, and 35 learned singleton/pairwise/higher-order action kernels. The constant is an intercept within the same tensor, not a separately fitted value branch. There are no horizon queries, horizon heads, or separate prediction losses. The representation is learned, but its width does not establish that every coordinate becomes independently useful.

Observed-action prediction, exact policy expectation, actor gradient, and candidate actor gain all derive from this same fitted function. The constant participates in both prediction and TD continuation. It cancels only when computing the change in expected return caused by an actor update. Component values are summed only at that final actor objective.

The one temporal rule is a vector Bellman equation. Training uses its real 32-step sampled composition:

`target_vec = sum(observed reward components) + expected_frozen_Q_vec(actual endpoint)`

The observed segment stops at 32 steps, an actual episode end, or the rollout boundary. Actual episode ends zero continuation; a nonterminal rollout boundary retains it. The same function is zero at terminal episode age. Every target is detached and materialized once before all full-batch fitting steps. This enforces that all calculations come from the same model; fitting only encourages Bellman consistency and does not guarantee correct environmental predictions.

## Comparison with v15

Unchanged: Beta actor, fixed Beta(2,2) feature coordinates, state-dependent integrable kernel family, 32-step observed segments, gamma=1 finite-horizon objective, 18 full-batch critic steps at learning rate 0.0003, gradient clipping 0.5, CG50 and KL ceiling 0.03. No replay, GAE, imagined state decoding, contrastive objective, or repeated target sweeps.

Changed: one shared trunk replaces separate trunks; one coefficient head replaces the reference-mean and response heads; all horizon-query inputs, sampling, and losses are removed. These are an architectural bundle, not an isolated parameter-count ablation. V16 also removes v15's directly observed short-horizon supervision away from episode ends. Its reward grounding instead comes through the observed segment in the unified TD rule.

## Verification and jobs

Numerical job **7585**, parallel limit **1**, time limit **25 minutes**: **10 CUDA checks passed**. Tests cover explicit unified predictions and analytic expectations, shared-trunk gradients for each component, constant-feature cancellation only in actor gain, terminal zeros, real segment boundaries and one-step equivalence, isolated production-shaped compiled targets, and two compiled critic/actor cycles. Independent review found no blocker in the final wiring or compiled-output lifetime.

Fresh HalfCheetah-v4 seed-1 job **7586**, parallel limit **1**, time limit **60 minutes**, depends on successful 7585. Nominal 8M transitions, 16 environments × 2048 steps, environment threads 2, compiled CUDA. No checkpoint loading. Report job **7587**, parallel limit **1**, time limit **5 minutes**, depends on the training job reaching a terminal queue state.

Source SHA256 at submission: `2d1a5f94e1066fb9cf409e7766e20c1bbf38299307d2450d509e7ab15ff399e0`.

## Results

Job **7586 completed** at **8,044,160 transitions**. Final last-100 training episode return was **788.91**, also the recorded peak. This improves on the split/horizon-query TD version in this seed, but remains substantially below v13 and the stronger v12 reference. Return largely plateaued after approximately 5.5M; the final peak is only marginally above that plateau. No long-run extension was launched.

| Nominal budget | v16 unified TD | v15 horizon-query TD | v13 Monte Carlo | v12 full batch |
|---|---:|---:|---:|---:|
| 1M | 132.95 | -0.32 | 438.12 | -45.38 |
| 2M | 427.28 | 275.09 | 1,330.08 | 481.24 |
| 4M | 533.89 | 305.73 | 1,235.44 | 1,720.64 |
| 6M | 763.43 | stopped | 1,262.71 | 2,779.96 |

All **245 actor updates** were accepted; mean KL was **0.02307** and final KL **0.00914**. The final modeled gain was positive (**0.0171**) despite the return plateau. Acceptance therefore remains a model-local property, not evidence of consistent environment improvement.

Actor optimization was also imperfect: median CG relative residual **0.03437**, mean **0.07629**, maximum **0.98713**; **58/245** solves exceeded 0.1. These were feasible KL-constrained directions, not reliably converged natural-gradient solutions. The experiment does not isolate critic approximation from actor conditioning.

The final vector TD residual RMS was **4.0146**, while the bootstrapped target RMS was **263.587**. These describe fitting to the frozen target, not error against true remaining environmental returns. There is no longer an immediate-horizon head or corresponding separate prediction metric. The architectural change eliminates that distinction; it does not by itself establish useful future prediction.

Report job **7587 succeeded**. [Numeric results](vector-unified-td-v16-results.json), [learning curves](vector-unified-td-v16-results.png), [vector plot](vector-unified-td-v16-results.svg). Completed result and checkpoint: `runs/HalfCheetah-v4__vector_unified_td_v16_8M__1__1789585442529891760/`. Single-seed, training-return evidence; this is a structural improvement over v15, not a promoted benchmark improvement over the established reference.
