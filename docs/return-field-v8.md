# Return-field policy improvement v8

Implementation: `cleanrl/ppo_continuous_action_return_field_v8.py`.
This is fresh end-to-end policy training, judged by episodic return. There is no
fixed-policy fitting stage, held-out promotion gate, GAE, scalar value network,
generated future outcome, or reward decoder.

## What the critic predicts

For each starting state, the critic outputs a tensor with axes:

    temporal band x actuator x (whitened alpha statistic, whitened beta statistic)

For Beta natural parameters eta=(alpha-1,beta-1), let T(a) contain log(a) and
log(1-a). With F=Cov(T)=L L^T, define z=L^-1(T-E[T]). The field for band b is

    c_b(s) = E[R_b * z | observation, episode age, policy concentrations, environment].

R_b is the actual discounted reward sum over that future band. The bands start at
0,1,2,4,8,16,32,64,128,256,512 and end at the episode boundary. Targets are joint
reward-score cross moments, not copies of scalar returns or inverse-action labels.
Their conditional means can point independently in mean-changing and
concentration-changing directions. CUDA contracts verify both cases.

Rewards remain scalar task observations. They multiply action-score vectors to
form tensor targets. Temporal components stay separate through critic fitting;
only the fitted vectors are added for the policy update. No scalar critic output
is broadcast over the sampled action score.

## Objective and geometry

A whitened policy displacement u corresponds to delta_eta=L^-T u. Its local
expected reward gain is c_total dot u and its local KL cost is 0.5||u||², where
c_total=sum_b c_b. Thus the exact conditional field supplies the locally optimal
direction for the Fisher approximation, subject to a KL budget.

The implemented actor surrogate is

    weighted_mean[(eta_new-eta_old) dot (L*c_total)].

This uses L*c for the ordinary natural-parameter gradient. It does not confuse
that quantity with the natural displacement L^-T*c. The actor remains in its
softplus+1 Beta family; no alpha/beta coordinate clipping constructs an invalid
target distribution.

One joint KL(old||new), summed across actuators, constrains the total update. It
is not split across temporal heads. The initial dual is

    sqrt(weighted_mean[sum_over_actuators_and_statistics(c_total²)] / (2*KL_budget)).

The sum over coordinates matters: an elementwise RMS would mis-scale the budget.
After optimizing this frozen local surrogate, parameter backtracking requires
positive predicted gain and measured exact joint KL <=0.01. Complete rejection
restores both actor weights and Adam state. Fractional acceptance retains proposal
optimizer moments; it is an approximation, not an invariant natural optimizer.

The free neural field is an approximation to the compatible gradient field, not a
guaranteed unbiased gradient critic. Positive predicted gain is not a guarantee
of improved real return. End-to-end training curves determine whether it helps.
No novelty claim is made for the underlying policy-gradient identity.

## Targets and trajectory boundaries

Default gamma=1 targets the benchmark's undiscounted 1000-step episodic return.
Optional gamma<1 also applies the outer gamma^episode_age weight; it does not
silently substitute a different state weighting for the discounted objective.

Only complete episodes within a 2048-step rollout chunk supply targets. This
uses roughly half the collected transitions and avoids cross-policy return tails.
The implementation explicitly rejects variable-length termination; the selection
argument is specific to fixed-length HalfCheetah. All collected transitions count
toward the benchmark budget, including unused boundary fragments and warmup.

The baseline is an age-specific reward mean from other environments in the same
rollout. ALL episodes from the source environment are excluded, because that
environment's observation statistics can depend on its earlier episodes. Raw
rewards are centered before computing temporal returns. No same-batch scale or
per-environment reward normalizer changes the actor's task weights. Standard
adaptive per-environment observation preprocessing remains in use.

Temporal sums use an FP64 parallel scan and telescoping return differences. A
band that starts after the episode ends is identically zero in v8, both in the
actor field and in its regression support. Partly observed bands retain their
actual truncated reward sums; no heuristic decay or fractional mask is applied.

## Optimization details

Each cycle collects data under the current actor, computes old policy statistics
once, fits the critic, freezes its output tensor, updates the actor, and collects
new data. Repeated optimization on the batch only fits that frozen local
surrogate; it is not described as repeated fresh on-policy gradient estimation.

The critic uses tensor PopArt normalization for the 11 x action_dim x 2 outputs.
When mean and scale change, its last linear weights and bias are transformed so
the raw field is preserved. The actor always receives the unnormalized field.
The trunk runs in BF16; the final PopArt head and raw-output affine transform run
in FP32 to avoid magnifying BF16 rounding into large raw-credit changes.

For a=old_scale/new_scale, head Adam first and second moments scale by a and a²
because the normalized squared-loss head gradients change in those units. This
does not preserve the entire Adam trajectory: trunk history, clipping, and
adaptive steps are not coordinate invariant. The preservation contract is the
raw predicted field, within FP32 precision.

Actor and critic use separate shuffle generators. Both training arms have the
same actor initialization, host sampling stream, and actor minibatch seed. Only
the learned arm fits a critic; the sampled arm supplies the observed return-score
tensor directly. Both use the same actor surrogate and KL constraint. Sampled
credit can inflate the dual through its noisy norm, so realized KL must accompany
return comparisons. This is an algorithm ablation, not a claim of matched wall time.

## Execution record

- MLQ 7492: v7 CUDA contracts, 12 passed; parallel limit 1, 20-minute limit.
- MLQ 7493: fresh v7 learned run, stopped at 7,388,800 transitions to enforce known
  zero support in a new version. Last logged 100-episode mean return: 2123.61.
  This is a partial-run observation, not an 8M result. Source and logs are preserved.
- MLQ 7494: fresh sampled-credit comparison, 8M requested; parallel limit 1,
  60-minute limit. Its targets already have exact zero support, so this arm is
  unchanged by the v8 field-support fix.
- MLQ 7496: v8 CUDA contracts, parallel limit 1, 20-minute limit.
- MLQ 7497: fresh v8 learned run, 8M requested; parallel limit 1, 60-minute limit,
  dependent on the v8 contracts. No checkpoint initialization or resumption.

All use seed 1, CUDA, 16 environments, two environment threads, local TensorBoard,
and normal queue priority. The 8M budgets round up to full rollout batches and
include stochastic phase warmup. No automatic retries are configured.

## Completed training evidence

MLQ 7496 passed 13/13 CUDA contracts. MLQ 7497 and the sampled-credit comparison
MLQ 7494 both completed 8,044,160 environment transitions, including 4,111,000
transitions from complete episodes used for targets. All initialization was fresh.

| Collected steps, approximately | Learned v8: mean last 100 episode returns | Sampled comparison |
|---|---:|---:|
| 1M | 4.50 | -230.18 |
| 2M | 339.79 | -206.81 |
| 4M | 1183.74 | -176.64 |
| 6M | 1611.11 | -120.61 |
| Final 8.04M | 1814.62 | -106.72 |

The return field learns useful behavior, but this comparison has a material
optimizer effect: mean realized joint KL was 0.0017564 for learned credit and
0.000005106 for sampled credit, despite the identical 0.01 ceiling. The latter's
noisy unprojected field norm inflates the initial dual, heavily penalizing policy
movement. Therefore the full return gap cannot be attributed solely to better
credit prediction. A larger common KL ceiling would not resolve this mechanism.

The next version should solve in the actor's realizable parameter geometry:
compute the frozen-gain gradient, solve against its policy Fisher with damped
conjugate gradient, scale by the undamped directional KL curvature, and verify
the step using exact joint KL. This removes the unprojected-field penalty from
both comparison arms. It remains a local approximate update, not a theorem of
monotonic return improvement.

Evidence:
[learned result](../runs/HalfCheetah-v4__return_field_v8_learned_8M__1__1789539671128294132/result.json),
[sampled result](../runs/HalfCheetah-v4__return_field_v7_sampled_8M__1__1789539554253128260/result.json).
These are single-seed training returns, not held-out evaluation or a standard-PPO
baseline comparison. No state-of-the-art performance claim is supported.
