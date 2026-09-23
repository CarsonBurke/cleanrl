# Vector credit metric v18

Hypothesis: the v17 predictor becomes more useful to the actor when its fitting
objective emphasizes vector TD errors that contaminate alpha/beta credit.
This tests optimization pressure with the successful v17 actor held fixed as
an algorithm. It does not establish an optimal latent representation.

## Evidence motivating the change

v17 completed fresh seed-1 HalfCheetah training at 8,044,160 collected steps,
reaching 4,334 last-100 episode return and still improving. v16 reached 789;
the older full-batch v12 reference reached 3,466. v17's multiple simultaneous
changes prevent attribution to distribution prediction alone.

At the final v17 update, model alpha/beta credit RMS was 1.230 and observed
residual credit RMS was 13.657. Some residual randomness is irreducible.
These local RMS values are not measurements of actor parameter-gradient
variance. Mean and moment losses were 0.0000472 and 0.004115 respectively;
their relative sizes do not directly measure competing gradient magnitudes.

## Objective

Keep the single state-action-age trunk and 263-dimensional joint prediction:
seven signed return-component means and 128 complex characteristic moments.
Keep fixed-1000 representation units, exact complex TD composition, terminal
atoms, and raw-unit actor readout. Only critic fitting changes.

Let e be the seven-dimensional raw TD residual, C=7, A=6, and n=32 the fixed
observed TD segment length. Let z be the collection policy's Fisher-whitened
alpha/beta score for the observed action. Set w = ||z||² / (2A). The mean loss is

    w * (||e||² + (sum(e))²) / (4 * C * n²).

Add the unchanged half squared error of the joint characteristic coordinates.
Average over the same valid rollout transitions. Weights, targets and all
actor credit are materialized under the pre-fit critic/collection policy and
stay frozen through 18 full-batch optimizer steps. There are no target sweeps.

The component metric is M=(I+11ᵀ)/2: eigenvalue 4 along the reward-sum
direction, 0.5 in every orthogonal direction. It retains errors that cancel in
total reward; it is full rank, with the same trace as the original identity
metric. The final scalar actor objective remains expected total return.

For an exact Beta draw, E[w|s]=1. Because w depends only on state and action,
weighted regression preserves the unrestricted conditional target mean. A
finite shared network changes its projection intentionally. No tail clipping
or batch renormalization is applied. The fixed n scales the mean loss; it
does not introduce another prediction horizon or split the prediction task.

This is a metric for sampled observed-residual credit. It neither minimizes
the full corrected actor estimator's variance nor removes bootstrap bias.
The eight-action model integral adds noise and covariance. Representation
sharing and global gradient clipping can change characteristic learning too.
The 1000-to-32 rescaling multiplies mean pressure by about 977 before applying
the component metric and score weights. It is a substantial intervention.

## Paired experiment

1. Default v18: use the score energy w above.
2. v18 without score weighting: set w=1; keep the new component metric and scale.

This pair isolates action-score weighting. Comparison to v17 tests the combined
metric/scale intervention, without disentangling those two changes. Distribution
moments themselves remain unablated; better returns cannot prove their benefit.

Both runs use fresh initialization, seed 1, nominal 8M transitions, 16×2048
rollouts, no replay, no GAE, KL ceiling 0.03, 18 full-batch Adam steps at 0.0003,
one frozen target snapshot per rollout, CUDA/BF16 compiled computation, and
the unchanged v17 actor update. No checkpoints are loaded.

Judge actual training-return curves and matched-step comparisons. Inspect
score-weight tails, raw component/utility residuals, model/residual credit,
importance-ratio ESS, true CG residual and gradient clipping for interpretation.
No held-out promotion gate and no automatic longer-run promotion are used.
The numerical contracts must pass before either fresh run begins.

## Execution record

Independent design and implementation review found no blocking issue. The
review emphasized that this objective measures only observed-residual credit,
and that the original mean-loss diagnostic is not the new optimized mean
block. Both diagnostics are now logged explicitly. Syntax checks passed for
trainer, tests and report; CUDA numerical results remain pending.

- MLQ 7645: 19 CUDA contracts (the 16 v17 contracts adapted to v18 plus three
  explicit metric/gradient contracts), parallel limit 1, time limit 25 minutes.
- MLQ 7646: fresh default v18 nominal 8M, after 7645 succeeds; parallel limit 1,
  time limit 60 minutes.
- MLQ 7647: fresh no-score-weighting nominal 8M, after 7645 succeeds and 7646
  terminates; parallel limit 1, time limit 60 minutes.
- MLQ 7648: paired results report after both training jobs terminate; parallel
  limit 1, time limit 5 minutes. Missing training progress is reported explicitly.

All jobs have default priority and one attempt. There is no automatic pruning
threshold or extension; the bounded 8M runs retain per-rollout progress and
support graceful cancellation. At submission, the checks and experiments are
waiting behind another project's exclusive job. No v18 test or return result
is claimed at this point.

Trainer SHA256:
`a7bfec63fcb10168b1ef43db72b511a9ba266125c774b797e7ac12eb2bda6361`.
Test SHA256:
`4cddc5da96f79e1c89dc4af1cb23bee41a99454d17b2de83d9de4757ac11aaf5`.

## Completed 8M comparison and requested 20M experiments

MLQ 7645 passed all 19 CUDA contracts. Both training jobs and the report
completed successfully. At 8,044,160 collected steps, weighted v18 reached
5,309.330 last-100 training return (peak 5,320.971), and unweighted v18 reached
5,012.139 (also its peak), versus v17's 4,333.993. This is a single-seed
comparison; the unweighted run retains v18's component metric and mean scale.

The user requested weighted and unweighted comparisons at 20M, followed by
bottleneck analysis. Both use fresh initialization and the unchanged trainer
SHA256 above. KL, batch size, learning rate, prediction family, 18 full-batch
steps, and one target snapshot per rollout remain unchanged.

- MLQ 7653: weighted fresh 20M, parallel limit 1, time limit 60 minutes.
- MLQ 7655: unweighted fresh 20M, parallel limit 1, time limit 60 minutes;
  starts after 7653 terminates. Command includes `--no-credit-score-weighting`.
- MLQ 7656: paired 20M report, parallel limit 1, time limit 5 minutes, after
  both training jobs terminate.

All jobs have default priority and one attempt. No automatic pruning threshold
is applied to this explicit 20M comparison; health and progress are monitored.
Job 7654 was cancelled before it started because its submitted unweighted
command omitted the disabling flag; 7655 is the corrected submission. No
model ran and no transitions were collected by 7654.

All three 20M jobs succeeded. Both trainers collected 20,037,248 transitions.
Weighted finished at 7,284.923 last-100 return; unweighted at 7,751.143.
Both exactly reproduced their corresponding 245 logged returns from the
earlier 8M runs. Unweighted is the stronger current seed-1 reference, while
both continued improving beyond 8M. Detailed curves, diagnostic windows and
limitations are recorded in `vector-credit-metric-v18-20m-results.json` and
`vector-credit-metric-v18-20m-bottlenecks.md`.
