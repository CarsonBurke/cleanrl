# Future-action control v19

Hypothesis: the unified future predictor can make source-action credit less
noisy by removing predicted effects of subsequent sampled actions from each
real TD segment. The comparison uses unweighted v18, which reached 7,751
return at 20.04M, as the reference. Source actor, Fisher solver, KL ceiling,
critic architecture and optimizer are otherwise unchanged.

## Unified target construction

Write Z(s,a) for the full 263-coordinate critic prediction: seven signed
return-component means and 128 complex joint characteristic coordinates.
At every actual rollout state, evaluate the observed action and eight
independent actions drawn from the frozen collection policy. Their average
estimates the policy expectation of Z; define the centered prediction

    D_t = Z(s_t,a_t) - mean_k Z(s_t, independent_action_k).

For a source transition t with a terminal-clipped real segment of m steps,
retain the v18 observed-segment TD target and subtract the transported controls
for t+1 through t+m-1. The source action and endpoint action are excluded.

Mean-coordinate differences add directly. Rotate each complex difference by
its frequency applied to the actual reward-vector prefix between the source
and the controlled action. This prefix is known before the action is drawn.
Consequently each transported control has conditional mean zero. The target
keeps the same conditional expectation for every coordinate for any fixed
critic; neither correct predictions nor exact Monte Carlo integration are
needed for this expectation identity.

With exact predictions, exact policy integration, and deterministic dynamics,
these controls cancel future-action randomness pathwise. Eight-action Monte
Carlo averages leave sampling noise. Inaccurate controls can increase variance;
the unit coefficient is a testable model-based hypothesis, not a guarantee.

Subtracting controls in raw return space and then taking sine/cosine would
change the distribution target. v19 instead controls the embedding directly.
Corrected sampled targets may leave the complex unit disk and are not clipped.
The critic still predicts bounded characteristic coordinates; their conditional
target expectation remains the original bounded expectation.

The mean estimator is algebraically a finite unit-weight action-value TD
trace/martingale sum. It is related to lambda=1 trace constructions. It has no
scalar V-based GAE estimator, lambda mixture, replay, generated-state decoder,
or repeated target sweeps. This relationship is explicit rather than hidden
behind a new name.

## Implementation and independence

All predictions, controls, endpoint continuations and actor credit are computed
with the critic frozen before fitting this rollout. Policy averages use the
collection policy. They remain fixed through all fitting and actor line-search
evaluations. Reusing an auxiliary action sample in other source rows creates
cross-row covariance but does not change the conditional-mean argument.

Two vectorized prefix scans compute transported sums. Rotate controls into
absolute reward-prefix coordinates, take inclusive prefix differences that
exclude the source action, then rotate back to the source. FP64 phases and
128-element tiled scans avoid long-scan compiler issues and cancellation from
long prefixes. Segment lengths come from the same real terminal/rollout-cut
handling as v18. A one-step segment has exactly zero future control.

The shared critic evaluates all candidate-action coordinates once. There is
still one prediction function and one frozen target snapshot per rollout.
The policy still maximizes expected total return; the characteristic outputs
remain indirect influences on that risk-neutral objective through shared
representation learning. v19 uses all coordinates to improve prediction
targets, but does not establish 263 independent policy-gradient directions.

## Experiment and interpretation

Fresh seed-1 HalfCheetah, nominal 20M transitions, 16 environments × 2048 steps,
unweighted v18 component/utility fitting, 18 full-batch Adam steps at 0.0003,
KL ceiling 0.03, no checkpoint loading, CUDA/BF16 compiled computation. The
20M budget makes the late comparison meaningful after v18's 8M ranking reversed.
No additional solver or learning-rate change confounds the target intervention.

In each training rollout, log original and controlled pre-fit vector residual
RMS and Fisher-whitened local residual-credit RMS using the same predictions.
These paired diagnostics describe sampled second moments, not independent
parameter-gradient variance estimates. They mix approximation error and
stochastic continuation noise. Their improvement is not a promotion gate;
actual learning return is the deciding evidence.

Record component and characteristic control sizes, corrected targets outside
the unit disk, fitting gradients, true CG residuals and importance-ratio ESS.
Checks must cover explicit transport sums, expectation preservation, ideal
cancellation, source exclusion, terminal boundaries, frozen targets, and the
production compiled tensor shape before training begins.

## Execution record

Independent design and source review found no blocking issue in source-action
exclusion, transport signs, terminal clipping, policy sampling, or frozen
compiled-output lifetimes. Syntax checks passed for trainer, tests and report.
CUDA numerical checks and the training run have not executed at submission.

- MLQ 7683: 25 CUDA cases, including the 19 inherited contracts, explicit
  transport/freeze checks, an enumerated branching environment with exact and
  misspecified critics and independent action averages, and production
  32768×263 compiled repeated-call coverage. Parallel limit 1, time limit 30m.
- MLQ 7684: fresh seed-1 nominal 20M training after 7683 succeeds. Parallel
  limit 1, time limit 60m. Explicit `--no-credit-score-weighting`.
- MLQ 7685: report comparing actual return with unweighted v18 after 7684
  terminates. Parallel limit 1, time limit 5m.

All jobs use default priority and one attempt. The queue has earlier work,
including another project's exclusive training job. No v19 test or return
result is claimed while these jobs are queued. There is no automatic pruning
threshold or promotion to a larger budget; per-rollout progress and graceful
cancellation are retained for inspection.

Trainer SHA256:
`ffc7b9ceb77d75211747eaaa77fb81e344049743cb08d3ea4a506768726132d7`.
Test SHA256:
`d110a6b6dbd6bf5d26058c60cecf10886316295f79374676ea965323189c9fee`.
