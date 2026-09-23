# Joint predictive distribution v17

Fresh seed-1 HalfCheetah experiment. No replay, GAE, checkpoint initialization,
minibatches, or Bellman target sweeps. Same 16 × 2048 rollout, one frozen target
per rollout, 18 full-batch critic optimizer steps, and actor KL ceiling 0.03.

## Hypothesis

The v16 critic restricted state-action effects to prescribed Beta power kernels,
whose reference standard deviations often reached their numerical floor. Its
actor trusted fitted effects without observed correction. Its late natural
linear solves also deteriorated. V17 removes those restrictions rather than
retuning their normalization.

A single joint state-action-age residual network predicts an embedding of the
remaining signed reward-component distribution. The seven physical components
are progress reward and six actuator costs; their sum is the task reward. The
embedding contains their normalized means and 128 complex characteristic
moments of joint component projections. Axes, the utility direction, and fixed
random unit directions at four frequencies give 263 prediction coordinates.
The trunk is shared by every coordinate. These are supervised future-return
predictions, with no negative pairs, state reconstruction, or generated-state
decoder.

The final sum of component means is required by the expected-return objective.
The extra coordinates change representation learning, not the definition of
optimality. They could be useless or harmful auxiliary pressure; dimension
alone does not demonstrate useful richness. This version does not prove an
advantage over an otherwise matched mean-only critic.

## One prediction rule

All returns use fixed normalization H=1000, not remaining episode length.
Features are component means divided by H sqrt(7), and cosine/sine moments
divided by sqrt(128). One output projection produces means, amplitude logits,
and phases. The complex moments have radius sigmoid(amplitude). This enforces
the necessary unit-disk condition, not full joint-distribution validity.

A real 32-step segment supplies a vector reward sum. Add it to continuation
means and rotate the continuation complex moments by its projected phase.
Continuation is the average of eight action predictions at the actual endpoint
state. At an episode end, continuation is the exact embedding of zero return.
All targets are materialized before fitting and stay fixed throughout the batch.
The loss is the squared embedding error. Gamma is one for this finite-horizon
objective. No separate short/long or outcome/effect prediction systems exist.

## Actor and numerical treatment

Before current-batch fitting, draw eight independent actions at each real state.
For each, subtract the average prediction of the other seven actions. Divide
these centered vector values by eight, and add a ninth vector weight for the
observed action: actual vector TD target minus pre-fit critic prediction.
Contract these vectors with task utility and Beta scores to obtain alpha/beta
credit. The finite-step surrogate uses the same frozen weights with joint
importance-ratio changes. There is no ratio clipping.

For a fixed candidate policy, the integration and observed correction cancel
current-action critic error in expectation. This identity does not remove
bootstrapped continuation error, realized future noise, or finite-sample policy
selection error. FP32 sampling/clamping introduces a small sampling approximation.
Mean KL does not bound the largest sampled importance ratio; ratio maxima and
empirical effective sample fraction are logged.

Natural updates use 100 preconditioned CG iterations. Eight independent
state/output Fisher probes estimate the parameter diagonal. Reductions use
FP64, and the final true linear-system residual is recomputed and logged beside
the recursive residual. The preconditioner may help conditioning; it does not
make an inaccurate Fisher matvec exact.

## Evaluation

CUDA numerical contracts cover the embedding composition and terminal atom,
Beta gradients and exact KL, leave-one-out correction, shared state/action
trunk gradients, compiled update lifetimes, production-shape segment scans,
and preconditioned solves. Benchmark success is judged by fresh training return,
not numerical tests, loss, or positive fitted surrogate gain.

Jobs and results will be appended after execution. Source:
`cleanrl/ppo_continuous_action_vector_predictive_distribution_v17.py`.

## Execution record

- MLQ 7607: 15 CUDA contracts passed in 24.88 seconds. Parallel limit 1,
  time limit 25 minutes.
- MLQ 7608: fresh nominal 8M training failed during host-actor construction,
  before warmup/collection/optimization. The already-modified shared dependency
  `cleanrl/shared/host_graph.py` raised `NameError: name '_OP_STRIDE' is not defined`.
  Parallel limit 1, time limit 60 minutes. This is an infrastructure failure,
  not evidence for or against the learning hypothesis.
- MLQ 7609: the dependent report failed because there was no progress file.
  Parallel limit 1, time limit 5 minutes. No learning curve was produced.

Source SHA256 for the attempted run:
`a359f19f9da4f2aa7abbfc4510c1ea126f02523ab4aac88e72fd1b54e77a74ad`.

The user was asked whether the original restriction on reading other files
permits a narrow inspection of this shared infrastructure failure. The shared
file has not been inspected or modified by this work.

After that startup failure, one additional numerical contract was added for the
actual public host-mirror boundary: compare the 16-row rollout actor against
CUDA before and after a parameter change and `refresh()`. This new sixteenth
case is syntax-checked but has not been executed; the preceding 15-case suite
is the one that passed job 7607. Resolving the shared startup failure and rerunning
the full suite precede any fresh training retry.

## Authorized infrastructure inspection and fresh retry

The user authorized the narrow dependency inspection. At inspection time,
`_OP_STRIDE = 8` was already present in the shared host-actor module. No shared
code was changed by this work. Its SHA256 at submission was
`d7867ec1b8f382133af3f16b46428dbc6c7b9ea8fa39b2d1839298183614c475`.
The v17 trainer still has the same SHA256 recorded above.

- MLQ 7620: full 16-case CUDA suite, now including the actual host actor boundary;
  parallel limit 1, time limit 25 minutes.
- MLQ 7621: fresh seed-1 nominal 8M training, strictly after success of 7620;
  parallel limit 1, time limit 60 minutes. No checkpoint loading.
- MLQ 7622: report after termination of 7621;
  parallel limit 1, time limit 5 minutes.

At submission these jobs are queued behind another project's exclusive job.
No additional checks or learning results are claimed until they execute.

## Completed fresh run

All 16 CUDA contracts passed in MLQ 7620. Training 7621 and report 7622
succeeded. The unchanged v17 source reached 8,044,160 collected transitions,
with final and peak last-100 training return 4,333.993. Matched returns were
955.499 at 1M, 925.921 at 2M, 2,114.139 at 4M, and 3,462.066 at 6M.
The established v12 full-batch reference finished at 3,465.798 and v16 at
788.906. These are single-seed training-return comparisons, not a causal
ablation of vector distribution prediction.

All 245 actor proposals passed the sampled surrogate/KL test; that count is
not evidence that each update improved actual return. Mean KL was 0.029841;
median recomputed CG residual ratio was 0.006412 and maximum 0.098709.
The final model-credit RMS was 1.230 versus observed-residual credit RMS
13.657, with cosine -0.023775. These are per-state local credit diagnostics,
not parameter-gradient variance measurements; the residual also contains
irreducible future-action randomness.

Final normalized mean-block loss was 0.0000472 and moment-block loss 0.004115.
Loss magnitudes alone do not establish gradient dominance or prove that the
means are underfit. They motivate the explicit v18 objective intervention:
retain the prediction family and actor, reprioritize vector mean errors in
policy-credit units, and isolate score weighting in a paired fresh run.
No v17 checkpoint is resumed. See vector-predictive-distribution-v17-results.json
and the accompanying PNG/SVG for the completed evidence.
