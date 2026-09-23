# Fresh long-run full-batch vector-critic experiment

The user requested a longer run to determine whether v12's full-batch, single-target configuration continues improving. The initial request said 80M; the user later clarified the intended budget was **50M** and requested stopping after the gains stalled. Job 7549 was cancelled, with the last complete logged rollout at **56,606,336 transitions**. This was a fresh initialization, not a resumed 8M checkpoint. Training source stayed frozen.

## Configuration and jobs

- HalfCheetah-v4, seed 1. Submitted budget 80,000,000; corrected intended budget 50,000,000; actual last logged step 56,606,336.
- 16 environments × 2,048 steps: 32,768 transitions per rollout, matching the original PPO source.
- Model LR 0.001, critic LR 0.0003: the better configuration from the two completed v12 8M experiments.
- Ten full-batch model steps; eight full-batch critic steps against one fixed, detached Bellman target per rollout. No repeated inner target refreshes.
- Gamma 1, horizon 1,000, mean actor KL ceiling 0.03, one natural actor update per rollout.
- No replay buffer, GAE, scalar value head, contrastive objective, or generated-state bootstrap. The current rollout is reused for fitting, then discarded; network parameters and optimizer moments persist.
- Training job **7549**, parallel limit **1**, time limit **60 minutes**. Cancelled by request after the user corrected the intended budget and judged further training uninformative.
- Original report job **7550**, parallel limit **1**, time limit **5 minutes**, was skipped because training was cancelled. Replacement partial-run report **7551**, same limits, succeeded. No training retry or resume.
- No algorithm change or new numerical implementation: the same frozen v12 source passed 23 CUDA numerical contracts in job 7545.

## Why stability without replay is plausible

Replay is a data-reuse mechanism, not a requirement for stable actor-critic learning. V12 uses fresh behavior data, a large fitting batch, fixed critic targets within each rollout update, gradient clipping, and an actor update constrained by exact mean KL. Actor parameters stay fixed during model and critic fitting, and the critic has separate parameters. These are plausible stabilizers, not an experimentally isolated causal explanation.

There were 245 actor updates and target refreshes in the 8M run. The stopped long run completed 1,727 of each; the original 80M budget would have required 2,442. Limited update frequency can produce slow, apparently stable learning. The KL ceiling controls change on the sampled states; it does not guarantee return improvement, a per-state bound, or an entropy floor. Ongoing episodes also retain states reached under earlier policies.

Performance stability, retained exploration, and noncollapsed latent features are separate claims. Rising return establishes none of the latter two by itself.

## Comparison with SAC

The primary reference is [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/html/1812.05905v2), especially sections 4.2–6 and appendices C–D. Its practical algorithm uses twin scalar soft Q-functions, slowly averaged target critics, replay, and entropy-temperature adjustment. Its actor differentiates Q through sampled actions; the paper uses tanh-squashed Gaussian actions and gamma 0.99.

| Aspect | V12 | Implication |
|---|---|---|
| Predicted credit | A latent covector pulled back to a vector of state sensitivities | Direct sensitivity supervision avoids fitting scalar value levels, but accuracy is unproven. |
| Actor credit | Immediate reward derivative + learned dynamics action-Jacobian transpose × next-state costate | Requires useful model derivatives, not merely accurate next-state predictions. |
| Actor update | Beta implicit transport into alpha/beta, then a natural-gradient step with exact mean KL | A pathwise gradient like SAC in broad form, with different credit estimation and step control. |
| Data | Large fresh rollouts; full-batch fitting; no historical buffer | Avoids stale action-distribution fitting, but loses historical coverage and opportunities to reuse old transitions. |
| Temporal propagation | One fixed critic target per rollout | Multiple regression steps improve that fit without performing additional temporal backups. Distant reward information may spread slowly. |
| Policy exploration | Stochastic independent Beta marginals, no entropy incentive | Stochasticity can shrink cumulatively. KL control does not maintain exploration. |
| Policy family | Alpha and beta exceed 1 | Supports skewed unimodal marginals but excludes U-shaped marginals. Both our policy and SAC's policy have bounded actions; boundedness is not a distinguishing advantage. |
| Long horizon | Finite-horizon gamma 1 | Aligns with undiscounted episode return, but derivative errors can amplify through dynamics and policy Jacobians. |

SAC already supplies an action-gradient vector despite its scalar Q output. The claim worth testing here is that *directly predicting reward sensitivities* improves credit assignment, not that a vector output is automatically richer than differentiating Q.

The direct field also gives up a consistency constraint: `J_h(state)^T c(state, age)` need not be the gradient of any scalar value function. An arbitrary state-dependent covector can have inconsistent cross-partials. Identity features fix the physical coordinates but do not enforce this consistency or guarantee useful learned latent features. In contrast, differentiating a scalar approximation produces a field consistent with that approximation, although it can still be wrong about the environment.

## What looks beneficial versus costly

The strongest aspects to preserve are separate vector credit for state/action/policy parameters, use of actual observed next states, and controlled policy movement. They avoid generating whole imagined trajectories and provide a clear route from predicted reward sensitivity to actor updates.

The leading performance concern is the mismatch between the dynamics fitting objective and how its output is used: transition/progress prediction MSE supervises values, while the actor relies on derivatives. V12's latent critic likewise minimizes vector residuals uniformly in fixed state coordinates; this is not guaranteed to prioritize errors according to their effect on achieved return. Predicting many coordinates does not by itself solve that optimization-pressure problem.

Another possible limitation is the actor step itself: its constrained linearized objective usually selects a step near the full 0.03 KL budget, even when the underlying credit is weak or inaccurate. Scaling down every credit would not shrink this trust-region step. The measured near-boundary KL throughout training therefore permits late wandering; it is not a convergence certificate. SAC uses a different optimizer rather than this fixed trust-region-radius update. This does not establish that the step rule causes the observed fluctuations.

Other plausible costs are slow temporal credit propagation, decreasing exploration, and losing coverage of previously visited behaviors. The derivative recursion multiplies errors by closed-loop Jacobians; the exact discounted scalar Bellman contraction does not automatically apply to this learned vector update. Finite episode boundaries make the ideal recursion well-defined, not necessarily numerically stable.

These are hypotheses, not measured causes of a gap to SAC. No matched SAC experiment has been run here, and training returns should not be equated with published deterministic evaluation scores. Borrowing an entropy objective or slowly changing target critic would not inherently require replay, but neither is introduced during this experiment. Coordinatewise minima of vector critics would not inherit the meaning of SAC's lower scalar Q estimate.

## Results

The fresh run reproduced all 245 return measurements from the earlier 8M run exactly. Metrics below are seed-1 last-100 **training episode returns**, not independent evaluation scores.

| Point | Actual logged step | Return |
|---|---:|---:|
| Previous 8M endpoint reproduced | 8,044,160 | 3,466 |
| 16M | 16,006,784 | 4,572 |
| 24M | 24,002,176 | 5,725 |
| 32M | 31,997,568 | 5,967 |
| 40M | 39,992,960 | 6,059 |
| Corrected 50M budget | 49,987,200 | **5,719** |
| Actual last complete logged rollout | 56,606,336 | **6,101** |

Before 50M, peak logged return was **6,496.729 at 42,647,168** transitions. Extra steps collected before cancellation reached a higher peak of **6,733.023 at 56,082,048**; that is beyond the intended budget and does not establish sustained improvement.

Mean logged rolling-return values were 5,859 over 24–32M, 6,129 over 32–40M, and 5,823 over 40–50M. These are descriptive averages of overlapping training-return windows, not independent statistical estimates. The run improved considerably beyond 8M, then fluctuated around roughly 6,000. The evidence supports stopping; longer training did not recover strong sample efficiency. The earlier minibatch/eight-refresh v10 reference had already reached 5,831 at 8M.

All 1,727 attempted actor updates were accepted; their mean actual KL was **0.0299773**. The run completed 17,270 model optimizer steps and 13,816 critic optimizer steps, with 1,727 target refreshes. Repeated acceptance of positive predicted gains did not imply continuing benchmark improvement.

At 50M, mean Beta concentration was 14.182 and native joint differential entropy was -7.118. At the last logged rollout, these were 14.238 and -7.343. Concentration had already reached roughly 14 near 4M, so the later performance fluctuations do not coincide with a large change in that aggregate. These averages cannot establish per-state coverage or noncollapsed latent features.

The run did not reach normal trainer completion, so it has no final `result.json` or end-of-run model checkpoint. Its `progress.json` and separate `experiment_outcome.json` preserve observations and cancellation provenance. The report marks status cancelled and final return null rather than presenting the last logged score as a completed 80M endpoint. No jobs remain active.

**Decision:** stop this run; preserve the user's full-batch and single-target constraints. The next improvement needs a clearer connection between predicted vector credit and realized return, rather than more training of this unchanged configuration. The SAC comparison above identifies hypotheses to examine, not causal findings from an ablation.

Artifacts: [numeric report](latent-costate-fullbatch-v12-long-run-results.json), [learning curves](latent-costate-fullbatch-v12-long-run-results.png), [vector figure](latent-costate-fullbatch-v12-long-run-results.svg).
