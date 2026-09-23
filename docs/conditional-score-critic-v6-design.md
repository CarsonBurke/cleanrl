# Conditional score critic: decoder-free v6 design

Status: independently reviewed and implemented in
`cleanrl/ppo_continuous_action_conditional_score_v6.py`. All 17 final CUDA numerical
contracts passed. Fresh fixed-policy experiment MLQ 7463 completed and failed the
mechanism gate. No actor training was launched.
All eventual runs must initialize actors, critics, normalizers, and optimizers afresh.
No prior checkpoint may initialize the method.

## Motivation

V5's reward decoder fits real transition latents accurately, but this does not
establish accuracy on generated latents. Its delayed action credit is wrong.
Remove that interface entirely: predict the conditional expectation of the
actor's action statistics given an outcome that actually happened. Never generate
an outcome and never decode a predicted latent into reward or scalar value.

This changes what the latent represents. It represents action credit in the
actor's distribution parameters, rather than a future state requiring an
additional utility model. Its dimension is determined by the actor's sufficient
statistics, not by arbitrary copies of a scalar return target.

## Exact identity

Fix the behavior actor throughout fitting and validation. At time t, let S_t be
its parameter score, grad_theta log pi_theta(a_t | s_t). Define Y_(t,h) to include
a real observed outcome at t+h and its measured reward r_(t+h). Including the
reward explicitly avoids assuming that an observation contains everything needed
to determine reward; this also covers stochastic rewards.

The tower property gives:

    E[r_(t+h) S_t] = E[r_(t+h) E[S_t | s_t, Y_(t,h), h]].

For J = E[sum_u gamma^u r_u], sum these contributions with gamma^(t+h).
Equivalent occupancy sampling and horizon importance weights must reproduce
these coefficients. Uniform rollout states are not automatically the exact
discounted start-state occupancy. All derivatives below are evaluated at the
behavior actor; model outputs, sampling distributions, and weights are detached.

## A minimal sufficient vector output for a Beta actor

For D independent Beta action coordinates, use natural parameters
eta(s) = (alpha_i(s)-1, beta_i(s)-1) for each i. Let

    T(a) = (log a_i, log(1-a_i)) for each i
    b(s) = E_pi[T(a) | s]
    z(s,a) = L(s)^(-1) (T(a) - b(s)),

where L L^T is the block-diagonal Fisher covariance of T under the behavior
actor. The policy parameter score is J_eta(s)^T L(s) z(s,a).

Train a critic to output

    m_phi(s, Y, h) = E[z(s,a_t) | s,Y,h],

using vector regression to observed earlier-action statistics, with complete
trajectories held out from fitting. This is a 2D-dimensional latent, with two
nonredundant distribution directions per actuator. It is sufficient for this
actor's exact first-order gradient: no posterior density, sampled posterior
actions, generated transition, or reward decoder is necessary.

The critic is reward-conditioned because Y includes measured reward. It is
action-statistic-supervised: its target is never scalar return, scalar Q, or GAE.
A rich physical-outcome encoder may be part of its input network; its usefulness
must be measured rather than inferred from latent width. At h=0, use the observed
score exactly when the outcome includes the original action; do not train an
inverse model to approximate a deterministic identity.

Accumulate a vector for each starting state:

    c_t = sum_h gamma^h r_(t+h) L(s_t) m_phi(s_t,Y_(t,h),h)
    gradient contribution = J_eta(s_t)^T c_t.

Each observed reward weights its own inferred vector contribution before the
horizon sum. No scalar return is broadcast over all action-score directions.
The eventual scalar loss is an implementation of a vector-Jacobian product;
scalarizing the optimizer objective is not scalarizing the critic.

## Actor update and data protocol

The local surrogate sum_t gamma^t eta_theta(s_t)^T stopgrad(c_t) has the desired
gradient at the behavior actor. Use a measured joint-policy KL constraint for
updates, and refresh the data and critic for the changed actor. This identity
does not make arbitrary multi-epoch PPO clipping or large steps unbiased.

Sample temporal bands to train short and long horizons, but correct their
probabilities in the actor estimator. Terminations, truncations, and observation
censoring must have explicit semantics. Silently dropping unobserved late rewards
would change the objective. A finite-horizon gate can be exact for its declared
horizon; it cannot establish correctness for the unobserved continuing tail.

## Meaning of optimality

At the population vector-regression optimum, with sufficient conditioning and
correct sampling, the estimator equals the original policy gradient in
expectation. It therefore does not introduce an intrinsic-reward or scalarization
tradeoff into the task objective.

The conditional mean minimizes squared prediction error for the action-score
vector. Because measured reward and starting-state Jacobian are known under the
conditioning, each exact conditioned reward-gradient term is a Rao-Blackwell
projection of its sampled counterpart. Its covariance is no larger for that
individual term. This does not guarantee lower covariance for the complete
trajectory sum, whose differently conditioned terms have cross-covariances.

It is not a theorem of global policy optimality, lower sample complexity, or
finite-model unbiasedness. Conditional mean error e induces gradient bias
E[r J_eta^T L e]. Held-out action-statistic MSE alone does not certify a small
reward-weighted gradient error. Cross-fitting prevents fitting/evaluation leakage
but does not remove this approximation bias.

There is also an important representation tradeoff. If Y reveals the original
action, then m=z and conditioning offers no variance reduction. More outcome
information is not automatically better. Conversely, removing reward-relevant
information without preserving reward measurability can break the identity.

The obvious sample correction r*m + r*(z-m) equals r*z and discards the benefit.
No finite-error unbiased correction with retained improvement is established by
this design. Such a claim would need extra structure, exact integration, or
additional sampling, together with its compute cost.

## Required evidence before actor training

- Fresh fixed-policy data with fitting and evaluation separated by trajectories.
- Vector moment calibration and reward-weighted gradient residuals, by horizon
  and by policy-distribution direction; inspect mean/concentration behavior.
- Independent-reference signal checks and paired projected-gradient comparisons.
- Whole-block variance measurements, not only per-transition or per-term errors.
- A comparison of rich physical-outcome conditioning with reduced conditioning
  to test whether the critic merely reconstructs the original sampled action.
- Correct local actor derivative and declared discount/censoring semantics.

Only after these pass should fresh policy-training benchmarks assess whether the
method produces stronger policies. The design itself is not evidence of that.

## Implemented protocol

The experiment uses a freshly initialized Beta actor, seed 1, 16 HalfCheetah
environments, and 2048-step rollout chunks. It collects 32 calibration rollouts,
128 fitting rollouts, and 256 held-out rollouts. Observation statistics freeze
after calibration. The estimator uses raw reward divided by one frozen global
standard deviation, avoiding per-environment reward reweighting and clipping.
A frozen per-episode-age reward mean centers both reference and modeled credit.
This subtracts a policy-independent deterministic reward schedule for the declared
fixed-length task; it does not introduce a value network.

The objective under audit is precisely the discounted 1000-step episodic reward
from reset, including the outer gamma^age factor. Only complete episodes wholly
observed in a rollout chunk are used. HalfCheetah has fixed-length episodes, so
this inclusion criterion is action-independent. Early terminations are explicitly
rejected rather than silently censored. No continuing-task value tail is inferred.

The full physical-outcome critic is the preregistered primary model. A separately
fitted reduced-outcome critic drops the future physical transition while retaining
starting state, policy concentrations, episode age, horizon, measured reward, and
normalization context. It is an exploratory test of excess future conditioning;
its success cannot automatically promote the primary method.

Both models have joint residual context networks, a learned 32D latent per
actuator, and a shared two-component action-score head. BF16 compiled networks
train on FP32 full-vector squared residuals. Beta geometry uses FP64 special
functions before returning FP32 targets and Fisher factors. No scalar value,
reward decoder, generated outcome, GAE, or bootstrapped return is implemented.

Immediate credit is exact. Delayed horizons are stratified into bands starting at
1, 2, 4, 8, 16, 32, 64, 128, and 256, with the last band ending at episode end.
Each band uses its exact discount mass and samples a conditional geometric offset.
The Monte Carlo reference sums all observed future rewards with a vectorized
doubling scan. A second reference uses identical horizon samples to separate
model effects from sampling noise.

Primary uncertainty uses paired rollout bootstrap, preserving correlations among
the complete episodes within each rollout. Means and covariance weight episodes
consistently when counts vary. The variance comparison uses complete episode
gradient sums, retaining all cross-time covariance. The gate requires reference
SNR >=3, positive lower95 independent split-mean dot product, upper95 relative
gradient error <0.5, lower95 cosine >0.5, and upper95 episode variance ratio <0.95.
It additionally requires upper95 variance ratio divided by squared mean-gradient
norm ratio <0.95, preventing a simple shrinkage of the entire gradient from passing.
These are practical promotion thresholds, not a proof of negligible residual bias.

MLQ 7461 initially passed 14 of 15 contracts. Its remaining test held a compiled
geometry output across another graph invocation without cloning; production code
already cloned it. The corrected test and an added gradient-shrinkage regression
passed as MLQ 7462: 16/16 contracts, parallel limit 1, 20-minute limit. The dependent
full experiment is MLQ 7463, parallel limit 1, 60-minute limit, no automatic retries or priority
override. Independent review checked estimator, episode selection, geometry, and
buffer lifetimes and prompted consistent episode weighting and the shrinkage check.

The first full-run attempt stopped during calibration because `gather_metrics`
requires tensor values and a calibration metric was a Python float. The logging
boundary now converts all metrics to tensors. MLQ 7463 was manually retried from
fresh initialization; no checkpoint or optimizer state was resumed. This startup
failure supplies no learned-model evidence. No automatic retries are configured.

## Completed evidence

MLQ 7463's fresh second attempt completed 13,647,488 transitions. The 256 held-out
rollouts supplied 4,297 fully observed episodes. Actor parameters remained fixed,
and both critics remained frozen throughout holdout. Reference SNR was 5.323 and
the independent split-mean dot lower95 was positive. The reference was usable;
the primary gate status is **failed**, not inconclusive.

| Estimator | Gradient cosine | Norm ratio | Relative error (upper95) | Episode variance ratio (upper95) |
|---|---:|---:|---:|---:|
| Primary physical-outcome critic | -0.03953 | 0.74609 | 1.27107 (1.31732) | 0.01952 (0.02004) |
| Reduced-outcome ablation | 0.75323 | 1.17144 | 0.77945 (0.88678) | 0.00873 (0.00901) |
| Paired sampled-horizon reference | 0.95385 | 1.04827 | 0.31479 (0.58377) | 4.14905 (4.25093) |

Variance is relative to the exact full-return score estimator, not the noisier
sampled-horizon reference. The primary signal-normalized variance ratio was
0.03507 (upper95 0.04249), so this is not merely uniform gradient shrinkage.
Nevertheless, its mean direction is misaligned. Low variance does not establish
valid or improved policy updates when the mean-gradient criterion fails, and
neither critic is promoted. This is not a claim that its mean-squared error is
worse at every possible update batch size.

The sampled-horizon reference is analytically unbiased, yet its mean-error upper
bound also fails the threshold because it adds substantial sampling noise. Gate
failure alone is therefore not a proof of bias. For the learned critics, the large
discrepancy together with their small sampling variance is the evidence of finite
approximation error.

| Exact future horizon | Reference SNR | Primary gradient cosine | Norm ratio | Relative error |
|---|---:|---:|---:|---:|
| 1 | 60.08 | 0.99832 | 1.01647 | 0.06068 |
| 4 | 30.02 | 0.96750 | 0.90932 | 0.25947 |
| 16 | 4.51 | -0.64848 | 0.13182 | 1.09011 |
| 64 | 1.10 | unresolved | unreliable | unreliable |

The primary model's projected mean credit is accurate at one step and reasonably
aligned at four steps; this does not establish accuracy of its entire conditional
vector function. Its 16-step estimate is too small and its point-estimate direction
opposes a usable observed reference.
The 64-step reference is too weak to diagnose the true gradient or call it zero.
Horizon 0 is supplied exactly by construction and is not evidence of model fitting.

The reduced-outcome ablation performs worse at horizons 1 and 4 but better at 16
and in aggregate. This demonstrates a conditioning/fitting tradeoff in these two
implementations; it does not establish a universal preference for less information.
Neither real-outcome conditioning nor low ordinary vector-regression loss is
sufficient evidence that weak delayed reward-weighted moments have been learned.

The decoder interface is absent, so decoder extrapolation cannot explain this
version's error. The result does not invalidate the exact conditional-expectation
identity; it rejects this finite fitted implementation. It does not uniquely
separate representation, fitting, capacity, and data limitations. Comparisons with
v5 are not matched-data estimates of architectural improvement: the fresh data
stream and audited objective differ.

Final checks MLQ 7469 passed 17/17 CUDA contracts (parallel limit 1, 15-minute limit).
The final suite includes correlated action/outcome identity checks, unequal episode
weights, and rejection of pure scalar gradient shrinkage, in addition to whitening,
autograd agreement, compiled backward, horizon sampling, and boundary handling.

Evidence: [gate.json](../runs/HalfCheetah-v4__conditional_score_v6_fresh__1__1789537296647494474/gate.json),
[holdout.json](../runs/HalfCheetah-v4__conditional_score_v6_fresh__1__1789537296647494474/holdout.json).

A next hypothesis should explicitly improve fitting of delayed reward-weighted
vector moments. Ordinary score MSE can be dominated by unpredictable action noise
while missing a small, consequential correlation with reward. Positive weighting
by a quantity measurable from the conditioning outcome preserves the population
conditional-mean optimum, but whether that improves finite-sample credit learning
is a new experiment, not a conclusion of this result. Arbitrary rescaling or
deleting delayed horizons would not establish the original gradient.

## Relation to existing work

The conditional-expectation identity is related to
[Hindsight Credit Assignment](https://proceedings.neurips.cc/paper/2019/file/195f15384c2a79cedf293e4a847ce85c-Paper.pdf).
The proposed implementation directly regresses continuous-policy sufficient
score moments rather than estimating a hindsight density and evaluating scalar
values. No novelty claim is made for the underlying identity.
