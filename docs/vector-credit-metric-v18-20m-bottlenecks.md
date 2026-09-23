# v18 at 20M: comparison and bottlenecks

This analysis concerns the unchanged v18 trainer and two fresh seed-1
HalfCheetah runs, weighted and unweighted. It uses training returns and
per-rollout diagnostics, not a held-out promotion gate. Numerical evidence is
in `vector-credit-metric-v18-20m-results.json`; the report script produces
matched-step comparisons, 4M-step diagnostic windows, and a six-panel figure.

## Completed comparison

Both jobs completed 20,037,248 collected transitions with 611 full-batch actor
updates and 10,998 critic optimizer steps. No checkpoint was loaded. Each fresh
run exactly reproduced all 245 logged episode-return means from its corresponding
earlier 8M run. This is reproducibility of one seed, not independent replication.

| Last-100 training return | Weighted | Unweighted |
|---|---:|---:|
| Earlier 8.04M endpoint | 5,309 | 5,012 |
| Nearest 12M | 5,901 | 6,293 |
| Nearest 16M | 6,529 | 7,078 |
| Final 20.04M | 7,285 | 7,751 |
| Peak through 20.04M | 7,362 | 7,773 |

Unweighted finished 466 points (6.4%) ahead and had the stronger 12–20M curve.
Use it as the current reference for this seed and budget. Both variants still
improved late in training; neither establishes a broad 20M plateau. The
unweighted variant retains the full-rank component/utility metric and increased
mean-error pressure. The comparison isolates score weighting, not all v18
changes, and does not establish a general across-seed ranking.

## Measured late-training limitations

These statistics cover the final 124 updates, from 16M through 20.04M.

| Diagnostic | Weighted | Unweighted |
|---|---:|---:|
| Median recomputed CG relative residual | 0.126 | 0.112 |
| Updates with CG relative residual above 0.1 | 87/124 | 74/124 |
| Median of rollout-average preclip critic gradient norm | 134.5 | 112.3 |
| Gradient clipping threshold | 0.5 | 0.5 |
| Median model/residual local credit RMS ratio | 0.073 | 0.072 |
| Median post-fit utility TD RMS, raw return units | 44.2 | 47.2 |
| Median importance-ratio ESS fraction | 0.926 | 0.907 |

The numerical solves deteriorate substantially: their corresponding 0–4M
median residuals were 0.014 and 0.009. The critic gradients also grow from
early medians 10.1 and 11.8 to the values above. These are demonstrated
optimization limitations, not demonstrated causes of a return ceiling.

The model-based credit remains small relative to observed-residual credit,
but the ratio cannot distinguish unpredictable future actions from critic
approximation error. Unweighted has better returns without a smaller median
utility TD error or better median late ESS. Therefore these comparisons do
not support explaining its advantage by either metric alone. Score weighting
does not optimize the complete corrected estimator or the shared actor's
parameter-space gradient error, so its early advantage had no guarantee of
persisting.

Priority is (1) make the vector predictions operationally useful for return
credit, (2) distinguish and address trajectory credit noise rather than merely
increase return-fitting pressure, and (3) improve numerical conditioning while
preserving the same full-batch/no-sweep experiment. Simply increasing output
width, score weights, or CG iteration count does not establish a model-level
solution to the first two issues. The concrete estimator hypothesis below is
unimplemented and is not an already accepted replacement for the user's
no-GAE constraint.

Training MLQ jobs: 7653 weighted and 7655 unweighted, both succeeded with
parallel limit 1 and 60-minute limits. Report 7656 succeeded with parallel
limit 1 and a 5-minute limit. Trainer SHA256 remains
`a7bfec63fcb10168b1ef43db72b511a9ba266125c774b797e7ac12eb2bda6361`.
The unchanged algorithm's 19 CUDA contracts had passed job 7645. The report
and interpretation received independent review; no further training was launched.

## What the vector currently does

The critic predicts seven reward-component means and 256 joint characteristic
coordinates from one state-action-age trunk. The critic objective retains
component errors even when their reward sum cancels. Its supervision therefore
does not collapse to scalar return regression.

However, `PredictiveCritic.components` reads only the seven means, and
`corrected_credit` sums their weights before multiplying policy scores. Since
the mean head is linear, adding its seven rows is algebraically one scalar
readout of the same trunk. The 256 characteristic predictions influence the
actor only by shaping shared features. They do not provide 256 distinct
action-credit directions. v18 has richer supervision, but has not established
the richer operational use of future predictions that motivated this work.

Expected total return is necessarily a scalar objective. The limitation is
where predictions become useful for estimating its gradient, not that the
final objective is scalar. The mean/moment loss and return comparisons do not
isolate the usefulness of the characteristic supervision.
Simply delaying the final sum over components would be algebraically identical
and would not fix this limitation. A useful extension must change what the
vector predicts or how it reduces error in the return-gradient estimate.

## Sources of residual credit

The observed target includes 32 real transitions and an endpoint continuation
averaged over eight sampled actions. A perfect conditional critic still leaves
randomness from the subsequent policy actions. Finite endpoint integration adds
sampling noise, and inaccurate endpoint predictions add bootstrap bias. Fitting
the source-state conditional mean cannot remove all these terms.

The logged model/residual RMS values are local alpha/beta credit magnitudes.
They are not whitened parameter-gradient variances and their scale changes
with the policy. A large residual is not proof of a bad critic. Likewise,
post-fit residuals measure agreement with the frozen approximate target,
not correctness of that target. v18's fitting objective weights one part of
the corrected gradient estimator; it omits the variance and covariance of
the separately sampled model integral.

## Optimization and policy update limitations

The critic's mean-output scale is fixed at 1000 steps, while v18 fits mean
errors in 32-step units. Combined with the component/utility metric, this can
produce substantial gradient clipping. The logged preclip norm is an average
over 18 epochs: it cannot recover exact clipping frequency or per-epoch
optimization behavior. Adam also prevents interpreting clipping as a simple
proportional reduction in effective learning rate. Loss traces alone cannot
distinguish noisy targets, insufficient fit, and an oscillating optimizer.

The Fisher solve uses 100 preconditioned CG iterations with fixed damping and
FP32 matrix-vector products. The recomputed residual reveals numerical solve
error. Even a small residual establishes only a good solution to the damped
system, not an accurate policy gradient or an undamped natural direction.

The direction and line search use the same sampled action estimates. Exact
average KL of at most 0.03 does not bound every importance ratio. Surrogate
selection can exploit finite-sample error; accepted positive surrogate gain
does not certify improved environmental return. The reported importance ESS
is unweighted across the candidate actions, not credit-weighted gradient ESS.

The critic receives episode age but the actor does not. This restricts the
policy to a stationary map of observations while its credit can depend on
remaining time. This is valid evaluation of that policy class, rather than
an implementation inconsistency. Its practical importance is unmeasured and
lower priority than demonstrated optimization or credit limitations.

## Candidate direction requiring a separate experiment

A more useful predictor could remove the contribution of subsequent sampled
actions from source-action credit. Under the frozen collection policy, the
quantity Q(s_t,a_t) minus its policy expectation has conditional mean zero.
The critic must be fixed before those trajectory actions, and the expectation
must use the collection policy. Reusing a critic fitted to the same trajectory
would invalidate this conditional-independence argument in general.
Subtracting those terms for t=1 through the end of the real segment preserves
the source-action target mean. With exact Q, exact policy integration, and
deterministic transitions, it removes future-action randomness pathwise.

This is a vector Q-TD/martingale control, algebraically related to finite
lambda=1 trace estimators. It should not be presented as a novel mechanism
unrelated to GAE. It requires no scalar value network, replay, or generated
state decoder, but its usefulness depends on predictive accuracy and Monte
Carlo integration noise. Approximate controls can increase variance.

For the joint distribution, subtracting raw-return controls before sine/cosine
would change the characteristic targets incorrectly. To preserve all prediction
semantics, apply centered prediction controls directly in the embedding and
rotate complex coordinates by the already observed reward prefix. That prefix
is known before the controlled action, so the control retains conditional
mean zero in every coordinate. This changes the estimator, not the predicted
future or reward objective. It remains a hypothesis, not an implemented or
validated improvement, and still does not automatically make every
characteristic coordinate necessary for the risk-neutral actor.
Individual controlled targets can leave the complex unit disk; projecting them
back into the disk would generally bias the regression target. Finite independent
action integration preserves expectation but introduces additional sampling
noise. True episode ends stop the controls and continuation; rollout cuts keep
continuation at their real endpoint.
