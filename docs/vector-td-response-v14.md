# Vector TD response v14

Source: `cleanrl/ppo_continuous_action_vector_td_response_v14.py`. Extends the [design hypothesis](vector-predictive-action-value-v14-design.md); the implementation choices below supersede that document's preliminary discussion.

## Hypothesis and implementation

Learn state-action future reward predictions with a vector one-step TD residual. Preserve progress and six signed actuator-cost channels across all horizon queries. The actor uses the full remaining horizon and sums physical components only at the final objective. There is no scalar value network, GAE, replay, imagined state decoder, contrastive objective, or repeated target sweep.

The critic uses separate state trunks for its reference mean and action response, with horizon-conditioned vector readouts. A shared state-action feature family includes 12 score coordinates and 35 learned Beta-integrable kernels: 12 singleton, 15 pairwise, six three-actuator, and two all-actuator products. Kernel powers depend on state and are trained jointly with the prediction coefficients. The corresponding 47-feature latent is queried for all seven physical components. Horizon predictions are cumulative totals, not rates multiplied by horizon lengths as in v13.

Feature centering and scaling use a fixed **Beta(2,2)** reference measure. This keeps the critic's represented function stable when the collection policy changes. The reference mean is not the current policy's value. Continuation targets integrate the **entire** frozen critic under the current collection policy, including its action response. Candidate actor gains subtract collection-policy feature expectations while retaining fixed-reference feature scaling. The actor's Fisher metric remains the separate collection-policy Fisher.

One-step target: `component_reward + expected_frozen_U(actual_next_state, h-1)`, with zero continuation at an actual episode end or h=1. At h=0 the entire prediction is identically zero. The targets are detached and materialized once before all fitting steps. An arbitrary rollout boundary keeps continuation using the actual last next observation. Real terminal observations are staged before an environment reset can replace them.

Each transition has six horizon queries: one step, the full remaining episode, and four stratified random integer queries. Every admissible integer horizon has nonzero support, avoiding an untrained h-1 chain on a sparse fixed grid. This samples prediction queries while keeping the entire rollout in every optimizer step.

## Settings and verification

- Fresh seed 1, HalfCheetah-v4, 16 environments × 2048 steps, two environment threads.
- Undiscounted finite-horizon objective, gamma=1, unchanged from v13. No additional gamma ablation is mixed into this experiment.
- Eighteen joint full-batch critic steps per rollout, Adam learning rate 0.0003, gradient clipping 0.5. One target snapshot throughout those steps.
- TD loss keeps every component/query independently supervised. One scalar normalizer uses observed immediate reward-component energy, not growing predicted value energy; it does not collapse the targets.
- Critic trunks use BF16; readouts use FP32; Beta moments use FP64 internally. CUDA compilation and host rollout standards are retained.
- One natural actor update, 50 CG iterations, exact candidate KL ceiling 0.03, nonlinear ray search. Solver residuals remain diagnostic, not a claim of exact optimization.
- Graceful cancellation finishes the current iteration and writes an explicitly cancelled result plus its checkpoint, when the process has time to complete cleanup.

Numerical job **7579**, parallel limit **1**, time limit **25 minutes**: **12 tests passed**. Tests cover state-conditioned analytic moments and derivatives, higher-order action interactions, distinct reference/collection coordinates, full policy expectations, vector target detachment and boundaries, zero horizons, production-shaped compiled target construction, and two compiled fitting/actor cycles. Independent review found no blocker in main-loop target construction, coordinate wiring, terminal observations, or compiled-output lifetimes.

Fresh training job **7580**, parallel limit **1**, time limit **60 minutes**, depends on successful 7579. No checkpoint loading. Source SHA256 at submission: `dd6bffeba488b7fe6c0be334060b55fc4c02f377dbbd692ef3041a9ab8519a8b`.

## Interpretation limits

TD trades future sampling noise for continuation approximation bias. With one target refresh per rollout, new temporal information can propagate slowly. Shared horizon conditioning does not guarantee accurate long-horizon predictions. Richer action features do not create counterfactual data for poorly sampled actions. Exact critic integration and a feasible KL do not prove positive environment-return improvement.

This version changes both the target construction and the action-response family; a combined benchmark cannot isolate their effects. The fixed-reference mean is a vector function, not a scalar control baseline. Full-vector prediction loss is reward-grounded supervision, not mathematically identical to maximizing benchmark return.

## Results

**Negative result.** Job **7580** was cancelled after sustained underperformance and decline. Its graceful shutdown saved a cancelled result and checkpoint at **7,782,016 transitions**, with last-100 episode return **227.54**. Peak return was **593.81 at 4,537,984 transitions**. Matched-step returns were -25.99 at 1M, 116.06 at 2M, 551.90 at 4M, and 232.79 at 6M. This failed to retain v13's early advantage or approach v12's later performance.

All **237 actor updates** were accepted, but mean KL was **0.01356**, substantially below the 0.03 ceiling. Median relative CG residual was **0.01064**. Thus the run was not simply exhausting the KL budget on every step, and the solver was generally more accurate than v13; neither observation certifies correct critic credit.

One plausible limitation is slow propagation of delayed consequences with just one one-step target refresh per 32,768 transitions. This is a hypothesis, not a causal finding. The focused follow-up v15 keeps the critic, loss scaling, optimizer, and actor unchanged while replacing one-step targets with a real 32-step vector residual plus one frozen continuation.

Report job **7581**, parallel limit **1**, time limit **5 minutes**, succeeded after training reached a terminal queue state. [Numeric evidence](vector-td-response-v14-results.json), [learning curves](vector-td-response-v14-results.png). Raw run: `runs/HalfCheetah-v4__vector_td_response_v14_8M__1__1789583093813423555/`. No long-run extension.
