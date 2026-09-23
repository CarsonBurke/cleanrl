# INTACT v8: factual goals and real-return-weighted joint fitting

## Research question

Can INTACT learn useful goal selection without PPO and without trusting the
action derivatives of imagined returns? The proposed v8 fits a stochastic goal
prescriber and its conditional action law to the same positively weighted real
transitions. Prediction remains a representation-learning task. It does not
supply a reward-maximization gradient to the prescriber.

This is a hypothesis and an experiment design, not an optimality claim. No v8
benchmark result is established by this document.

## Evidence motivating the change

The relevant recent line is the root `cleanrl/ppo_continuous_action_jepa_intact_*`
family, rather than the older scripts in `cleanrl/embedding-optimization/`.

The following figures were read from TensorBoard through `scripts/score_runs.py`
on 2026-09-19, using `--at 1M,2M,8M --last 20`. All are HalfCheetah-v4, seed 1.
Matched-step columns use the scoring helper's averaging window; the final
column uses the last 20 episodes and therefore is a different statistic.

| Run | At 1M | At 2M | At 8M | Final last 20 | Last episode step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jepa_intact_model_control_v5_none` | 1,839.0 | 3,585.0 | ~4,308.7 | 4,452.4 | 7,995,008 |
| `jepa_intact_real_corrected_gradient_v6_corrected` | 2,519.1 | 4,152.7 | ~5,615.1 | 5,668.2 | 7,995,008 |
| `jepa_intact_one_step_model_gradient_v7_model` | 769.2 | ~50.1 | unavailable | 65.1 | 1,982,000 |

`~` means the run ends inside the matched-step averaging window. A second v6
run with the same experiment name ends at 32,000 steps; it is not the completed
reference. These single-seed learning curves support a research decision, not a
statistical claim of algorithm superiority.

In v5, model actor modes optimize a short imagined rollout. The `none` mode
keeps real-return PPO control, with the same broad representation and inverse
architecture. In v6, the model contribution is instead a linear action control
variate. With fixed behavior mean `mu0`, fixed response `h`, and current mean
`mu`, its correction is `E_behavior[ratio * h.(a-mu0)] - h.(mu-mu0)`.
For an exact action expectation this is zero, including its policy gradient.
The model can change estimator variance without replacing the real-return
objective. Finite samples, clipping of the separate PPO term, and representation
updates still matter; this identity does not establish monotonic improvement.

v7 removes real-return actor correction and optimizes learned one-step reward
plus learned continuation times the critic on a predicted successor. Freezing
the modules prevents actor loss from changing their parameters, but does not
make their action derivatives correct. Low factual prediction error does not
validate derivatives transverse to factual trajectories or critic values at
model-generated states. Its collapse shows that reducing imagined horizon to
one was insufficient here. It does not identify the entire causal explanation.

A matched 2M diagnostic read adds an important qualification. v6's isolated
all-module policy KL was about 0.0405; v7's was about 2.468. Value-expansion MSE
was about 0.00186 versus 0.00149 respectively, but advantage variance was
0.00463 versus 0.00115. Thus v7's superficially small model error is large
relative to its remaining return signal, and its updates move the policy much
more. These are sparse interval metrics (fewer than ten points in the window),
with v7 ending inside the 2M window. They support examining both model
derivative reliability and policy-update geometry; they do not separate those
causes. The new empirical weight KL is not an action-policy KL guarantee.

Relevant implementations:

- `cleanrl/ppo_continuous_action_jepa_intact_model_control_v5.py`: `actor_objective`, `training_loss`.
- `cleanrl/ppo_continuous_action_jepa_intact_real_corrected_gradient_v6.py`: `world_loss`, `model_action_value`, `training_loss`.
- `cleanrl/ppo_continuous_action_jepa_intact_one_step_model_gradient_v7.py`: `actor_objective`.

Older experiments expose additional risks. Detached v16 combines one-step
predicted deltas and H12 factual deltas in a shared inverse law without a horizon
indicator, then searches arbitrary fixed-radius directions in 64 latent
dimensions. Matching the norm of factual goals does not match their support.
v17 is a PPO actor reparameterization with an unconstrained hidden "intent";
it does not test a JEPA-grounded goal mechanism. Policy-free MPPI v18 removes
the inverse law but uses only H12 reward, no terminal value, and a small search
population over a 72-dimensional action sequence. At 1M its matched return was
15.0, despite low factual prediction error and similar mean predicted and
realized first-step velocities (0.049 versus 0.040). Its failure cannot be
attributed to demonstrated reward hallucination alone: exploration, search,
and horizon limitations remain competing explanations.

## Proposed objective and its exact idealized argument

Let `x` contain the observed state and previous physical action. In a real
rollout, record first action `a`, a boundary-safe future state, and a real-return
advantage estimate `A`. With an encoder frozen for this rollout's actor update,
define hindsight label `g = encode(future_state) - encode(current_state)`.

For N rollout samples, solve the finite-sample improvement problem:

```
maximize sum_i w_i * A_i
subject to w_i >= 0, sum_i w_i = 1
           sum_i w_i * log(N * w_i) <= epsilon
```

Its interior solution has `w_i = softmax(A_i / temperature)`. Choose temperature
against the actual KL constraint, planned as epsilon = 0.1. Uniform weights are
the matched ablation. ESS is a diagnostic, not a substitute for this KL
constraint. A global rollout constraint does not itself bound policy KL at
each state, or even marginal action-policy KL after approximate fitting.

Fit the joint likelihood with exactly the same detached weights in both terms:

```
L = -sum_i w_i * [log p_selector(g_i | x_i)
                  + log p_law(a_i | x_i, g_i)]
```

The idealized argument is a distribution factorization. Let `q_w(g,a | x)` be
the conditional distribution induced by weighting factual tuples. If both
factors fit it exactly, then:

```
p_new(a | x)
  = integral p_selector(g | x) * p_law(a | x,g) dg
  = integral q_w(g | x) * q_w(a | x,g) dg
  = q_w(a | x)
```

Thus the deployed action marginal is the reward-weighted factual action
distribution. No derivative through predicted reward or predicted state is
needed. Future states provide an auxiliary factorization of action selection,
not an independently verified causal command interface. In particular, the
weighted action marginal involves the expectation of the sample weights given
`x,a`; with stochastic return estimates, it is not generally the exponential
of an exact expected advantage.

The proposed approximation uses eight Gaussian mixture components for the
goal selector, learned isotropic component standard deviations initialized at
0.25, and a conditional Beta
action law. Runtime samples a component, samples its goal, then samples the
Beta action. Goals are resampled every step, matching the one-decision joint
factorization. Mixture means must not be averaged into one goal: averaging
distinct successful trajectories can produce an unsupported intermediate goal.

## What the argument does not guarantee

**Future-action confounding.** A factual future endpoint depends on the first
action, later actions, and any environment stochasticity. The conditional
inverse law is a posterior association, not proof that its first action causes
that endpoint. This does not invalidate the idealized marginalization above.
It does invalidate assuming goal persistence or long-horizon controllability
without additional evidence.

**Unsupported Gaussian samples.** The 64-dimensional endpoint representation
comes from a lower-dimensional physical state manifold; conditional one-step
reachability has at most the action dimension locally. Isotropic goal noise
can leave that support. More mixture components address distinct modes but do
not fix covariance geometry. Learned isotropic scales adapt dispersion but
cannot represent anisotropic reachable directions. The initial standard
deviation and eight components are experiment choices, not established optima.
The Beta family also limits
conditional expressivity. Exact fitting in the derivation is an idealization.

**Shared objectives.** JEPA/SIGReg and local inverse supervision can change
features or action-law parameters independently of the weighted goal objective.
Such auxiliary updates mean the final law need not be the weighted conditional
maximum-likelihood solution. Record parameter ownership and gradient routes;
do not present the factorization identity as a guarantee for the combined loss.
The current implementation multiplies weighted goal-law NLL by 0.05 and
unweighted local inverse NLL by 0.1; it clips the law together with world-model
gradients. Those choices retain prior auxiliary settings, but make the law's
effective update different from unconstrained joint maximum likelihood.
In v6/v7, reward and value heads consume detached features, so preservation of
all task-relevant distinctions is not directly enforced by those heads.

**Encoder drift.** Freeze actor features and endpoint labels once per rollout
to avoid changing labels during fitting. That does not prevent the deployed
encoder from changing through world learning, nor the next rollout from using a
different latent chart. Measure behavior changes induced by that representation
update. A target encoder or explicit coordinate transport could address this,
but is not justified as an automatic patch before measuring the effect.

**Finite data and fitted state distribution.** The global weighting also shifts
which states receive regression emphasis. In continuous states there are few
repeated identical contexts, so both conditionals depend heavily on function
approximation. Positive weights can overemphasize noisy lucky trajectories;
the KL budget controls concentration but does not remove this bias. GAE
supplies factual rewards plus estimated bootstrap values, not exact returns.

**Exploration and horizon.** Positive reweighting cannot invent successful
actions absent from the rollout. A stochastic mixture creates exploration but
does not establish useful exploration. Boundary-shortened goals also have
different effective horizons; a shared law may average incompatible contexts.

## Evaluation and decision criteria

The initial queued experiments are weighted H8 and weighted H1, each a full
8M-step HalfCheetah-v4 run, seed 1, with identical architecture, auxiliary
objectives, rollout size, and compute settings. This compares complete variants
with different factual endpoint horizons. At H1, local and goal action labels
coincide at the frozen chart, so their combined weight is `0.1 + 0.05*w`:
normalized, `(2+w)/3`. This dilutes reward selection toward behavior cloning.
At H8 the same auxiliary term acts at different inputs, with different
interference. The comparison does not isolate horizon independently of that
interaction. Uniform weighting is implemented as a subsequent matched control,
but has not been submitted. Use the harness logs and machine-wide queue. These are
benchmark experiments, not reduced smoke runs. Characterize concurrency before
using standard-PPO throughput assumptions for this architecture.

Compare learning curves at matched 1M, 2M, and 8M against each other and the
completed v6 and v5 `none` references. Report final last-20 return separately
from matched-step means. Report observed throughput and elapsed compute too.
Never compare a stopped 2M endpoint directly with a completed 8M endpoint as an
algorithm ranking.

The weighted-versus-uniform comparison tests whether factual reward selection
adds value to the same latent factorization. It does not isolate the value of
hierarchical goals versus a direct reward-weighted action policy; that requires
a separate matched direct-action control if the first experiment is promising.

Inspect weight KL, temperature, ESS, maximum weight, goal mixture occupation,
goal likelihood, conditional action likelihood, Beta concentration, latent
scale, factual goal norms, and sampled goal norms. Evaluate the action law on
factual versus sampled goals and monitor whether selector components are used.
Where possible compare pre-update and post-update action behavior on fixed
observations, separating encoder and action-law changes.

Success requires sustained real-return improvement rather than only lower
goal NLL, better prediction, or a higher imagined objective. An early weighted
advantage over uniform would support the reward-weighting mechanism; beating
both prior references would support the complete implementation. Failure of
both variants would leave manifold mismatch, fitting interference, and poor
exploration unresolved. Do not convert any one of these outcomes into a claim
that JEPA or INTACT as a whole is proven or disproven.

### Decisions after the first measured curves

- If factual inverse likelihood improves but sampled-goal action behavior
  diverges, prioritize proposal support and covariance geometry. More imagined
  rollout depth does not repair this mismatch.
- If goal likelihood improves while shuffling factual goals barely changes
  action likelihood, investigate whether the inverse law ignores goals or uses
  previous action as a shortcut. A low JEPA loss alone is not evidence of
  meaningful intent control.
- If return weighting helps but the hierarchy adds no benefit over a matched
  direct-action weighted-likelihood policy, remove the unnecessary hierarchy.
  That direct-action experiment is a follow-up, not an existing result.
- If both factor fits and goal dependence are strong but returns stagnate,
  examine exploration coverage and the real-return estimator before making the
  predictor larger. This on-policy scheme deliberately gives up replay-driven
  value learning; a real-transition off-policy critic remains another candidate,
  but would need independent checks against action-value extrapolation.
- If H1 and H8 disagree, inspect their different local-supervision interference
  before selecting a causal account. A coefficient sweep alone would not resolve
  whether the inverse representation or return-weighted control is responsible.

## Execution record

Submitted 2026-09-19:

| Job | Work | Parallel limit | Runtime limit | Dependency |
| --- | --- | ---: | --- | --- |
| 8386 | CUDA behavioral contracts, including compiled gradients and host parity | 1 | 20 minutes | none |
| 8389 | v8 weighted H8, 8M, seed 1 | 1 | 2 hours | 8386 succeeds |
| 8390 | v8 weighted H1, 8M, seed 1 | 1 | 2 hours | 8386 succeeds |

The first contract attempt passed 11 tests and failed compiled/eager gradient
equivalence for the mixture density (75 of 33,792 elements in one prescriber
weight tensor; largest absolute difference 0.00205). Explicit detached-max
centering of component log densities was added to stabilize backward
normalization without changing the mathematical likelihood. Job 8386 was
manually retried. Original training jobs 8387/8388 were skipped by their failed
prerequisite, not executed; 8389/8390 replace them and await the retry outcome.

All use the machine-wide queue at ordinary priority. Training uses 16 native
environments, two environment threads, compilation, explicit OMP/MKL thread
limits, and TensorBoard harness logging. No automatic retries are configured.
Monitor real-return progress and cancel uninformative runs through `mlq`;
there is no unattended metric-culling process attached to these jobs.

Evidence job **8407** runs `scripts/intact_objectives_report.py` after jobs
8389, 8390, and 8392 reach any terminal outcome (parallel limit 1, ten-minute
limit). It writes `docs/intact-objectives-results.{json,png,svg}` using the
shared harness scalar reader. It does not infer queue success from available
logs, extrapolate missing returns, or claim completed evaluation. At submission,
the figure contains only historical references; new runs have no return data.

At the end of the 55-minute queue observation window (2026-09-20 UTC), checks
8386 and 8391 were still queued behind higher-priority work. Training jobs
8389, 8390, and 8392 remained gated on their checks; evidence job 8407 remained
gated on the training outcomes. The observation timeout did not cancel or retry
any workload. All five new Python files passed syntax validation, and the
baseline report rendered successfully. The numerical stabilization remains
unverified on CUDA; neither new algorithm has a measured benchmark result yet.

## Precedents

The [upstream INTACT implementation](https://github.com/zju3dv/INTACT-JEPA)
grounds a shared inverse law using attached physical successors and detached
future goals. Its demonstrated goal-conditioned setting is different from
online reward maximization here; v8 is not a reproduction.

[Relative Entropy Policy Search](https://ojs.aaai.org/index.php/AAAI/article/view/7727)
and [Hierarchical REPS](https://proceedings.mlr.press/v22/daniel12/daniel12.pdf)
are precedents for information-constrained weighting and latent policy fitting.
The empirical weighting step here does not implement their full state-distribution
constraints or establish an equivalent policy improvement guarantee.
