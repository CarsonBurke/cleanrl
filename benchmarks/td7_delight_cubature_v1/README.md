# TD7 latent-Q with a delight gate

Starting point: `HalfCheetah-v4__td7_latentq_onpolicy_v8_replay_sale__1__1791395613`,
produced by `cleanrl/td7/latentq/td7_latentq_preaction_onpolicy_v8.py` with
`actor_mode=replay`, `actor_sale=True`, seed 1, 16 environments, 1024 rollout
steps, and one replay update per transition. Its requested horizon was 3M;
job 11971 was cancelled. Recorded training return averages 16,208 over its
last 20 episodes at 818k; its last deterministic evaluation is 17,082 at
802,432. These are partial-run results, not an 8M benchmark.

## Paper reading and design

Read the complete local editions, including appendices and references, and
inspected figures and relevant original-PDF equations:

- [Delightful Policy Gradient](../../../reading-list/markdown/2026-osband-delightful-policy-gradient/paper.md)
- [Delightful Distributed Policy Gradient](../../../reading-list/markdown/2026-osband-delightful-distributed-policy-gradient/paper.md)
- [Delightful Gradients Escape Corners](../../../reading-list/markdown/2026-mei-delightful-gradients-corner-escape/paper.md)
- [Delightful Exploration](../../../reading-list/markdown/2026-osband-delightful-exploration/paper.md)

The paper gate multiplies an advantage-weighted **score gradient**. Multiplying
v8's pathwise center-Q loss by a detached sigmoid would be a different method.
v1 therefore retains the v8 critic and data cadence, but replaces the actor
objective with a fresh-action score update at replay states.

For each state, draw four Gaussian preactions in two antithetic pairs. Query the
trained physical point head at their clipped actions, averaging the two Q
members as the original actor does. Estimate the state baseline with the
equal-weight Gaussian cubature nodes `+/-sqrt(action_dim) * unit_axis`.
This rule integrates all polynomials through degree three exactly before
clipping. It provides a deterministic, action-independent baseline rather
than adding a random baseline inside a nonlinear gate. It is approximate for
the actual neural critic and especially for clipped actions.

Let `A = Q(s, clip(u)) - V_cubature(s)`. The default weight is
`2 * sigmoid((A / RMS(A)) * min(0.5 * sum(eps**2), 10) / temperature)`.
Freeze the sampled preaction, critic estimate, baseline, RMS and gate before
differentiating its Gaussian log probability. Antithetic samples retain the
correct Gaussian marginal; their baseline does not use the other sample.
No replay-action importance ratio is needed because actions are freshly drawn
from the current policy. Replay **states** still have the original sampling bias.

The default `gate_scope=mean` applies this weight to the location score and
leaves the scale score as PG. The entropy loss and alpha tuner stay ungated.
For a locally linear Q and symmetric Gaussian, paired location updates equal
PG, while full DG adds a variance incentive. v8's positive alpha tuner is a
minimum-entropy constraint and cannot cap exploration. Mean-only gating avoids
that known inflation mechanism; its benefit must come from nonlinear action
quality, rather than larger noise on a linear critic.

These are deliberate adaptations, not a literal reproduction:

- Relative surprisal comes from the acting-side paper's host-mode convention.
  Signed differential surprisal averages approximately -5.30 at six dimensions
  and sigma 0.1, so it usually reverses the intuitive success/failure gate.
- RMS scaling changes only gate units, never recenters advantages or changes
  the critic targets. It is an extension; the continuous paper specifies reward
  EMA decay 0.999 but does not give an unambiguous normalization formula.
- The factor two recovers ordinary PG in the large-temperature limit.
- Mean-only scope is an extension. `--gate-scope all` exposes full-score DG;
  `--surprisal density --gate-scale raw` exposes its signed-density/raw-unit
  variant, but still uses the cubature baseline and factor two.
- New versions apply the repository's 0.5 gradient clipping standard to the
  encoder, critic and actor. Original v8 did not clip these gradients. This is
  a comparison confound against the historical run; matched controls share it.

The theory paper's global conclusions are not guarantees for this neural
system: its argument changes between score-gradient and mirror-style updates,
and treats thin inequality regions as measure-zero sets.

## Controls and diagnostics

The queued main run and `actor_objective=pg` control differ only by gate weight.
`actor_objective=pathwise` also exposes v8's original actor objective with the
new version's gradient clipping for a further control.

TensorBoard records gate mean/saturation, advantage mean/RMS, negative
surprisal fraction, clipped action fraction, cubature-versus-center value
disagreement, policy KL, sigma, alpha, return, deterministic evaluation and
phase times. The cubature-center gap is **not** an estimate of quadrature error.
Sampled advantage means can diagnose systematic baseline mismatch, with
Monte Carlo uncertainty. No performance improvement is established until runs
execute and matched-step comparisons are available.

All GPU work goes through mlq, declared concurrency 1 pending characterization.
Both jobs request 8M transitions with seed 1. The supervisor runs autocull in
the same foreground process group and terminates it with the trainer. The
historical reference stops at 818k, so autocull cannot assess relative
underperformance beyond its supported curve; health enforcement remains
active. Queue IDs and commands are recorded in `jobs.json`.

Submitted on 2026-10-10:

| Job | Objective | Declared concurrency | Budget | Submission state |
| --- | --- | --- | --- | --- |
| 12682 | Delight, mean scope | 1 | 8M, seed 1 | Queued |
| 12683 | Ungated PG control | 1 | 8M, seed 1 | Queued |

Both wait behind higher-priority machine-wide work. Analytic tests passed;
CUDA compilation/capture and learning remain unverified until these jobs run.
For matched-step comparisons, use:

```bash
mlq show 12682
mlq show 12683
.venv/bin/python scripts/score_runs.py td7_latentq_delight_cubature_v1 td7_latentq_onpolicy_v8_replay_sale --at 200k,400k,600k,800k,2M,4M,8M --metrics eval/episodic_return
```

## Floor-free follow-up (v2)

On user request, created `cleanrl/td7/delight/td7_latentq_delight_nofloor_v2.py`:
remove the entropy objective, positive alpha tuner, alpha optimizer and
`target_sigma` argument. Both PG and DG now learn scale exclusively through
the ungated action-value score. Preserve the actor's numerical log-std bounds
`[-5, 0]`; this ablation removes the soft 0.1 entropy floor, not the lower
parameterization bound near 0.0067. Critic target smoothing remains 0.2 and
does not constrain behavior noise.

The same mathematical contracts pass against both versions (24 checks).
User requested switching the active experiment to PG and a 20-minute limit:

- Cancelled floor-enabled DG job 12682 to release the active lease.
- Floor-free PG job **12687**: priority 3, concurrency 1, wall-clock limit 20m.
- Floor-free DG job **12688**: priority 3, concurrency 1, wall-clock limit 20m.
- Existing floor-enabled PG job 12683 also now has a 20m limit (priority 2).

The declared transition ceiling remains 8M, but the explicit 20-minute limit
includes test checks, compilation, initialization, evaluation and training.
These are full experiments with a user-requested wall-clock budget, not
reduced smoke runs. Compare learning curves at matched recorded steps;
time-limit termination does not mean the 8M ceiling was reached.


## Preaction query and exact throughput changes (v3)

`cleanrl/td7/delight/td7_latentq_delight_fast_preaction_v3.py` restores the
actor's distribution-conditioned critic query: Q(s, current mu, current sigma,
u), with SALE embedding evaluated at clipped u. Sample actions and Gaussian
cubature baseline nodes use the same detached current-policy context. Only
the fixed-action Gaussian score receives actor gradients. This remains a
replay-state, fresh-action score surrogate; it does not differentiate through
the critic's policy-context inputs. `--actor-query point` preserves the v2
point-Q surrogate for matched comparisons. All three critic supervision lanes
remain present in both modes.

Neither PG nor DG has PPO ratio clipping, importance-ratio clipping, a KL
penalty, or a KL stopping rule. Actor/critic/encoder gradient norms are clipped
at 0.5. Both remain free of the learned entropy floor introduced in v8; hard
log-std bounds remain [-5, 0]. Zero target smoothing means a point-action
Bellman backup, not zero behavior exploration.

Performance changes preserve the losses, architecture, update count, replay
probabilities, and precision:

- The observed-preaction and physical-point critic lanes have zero tangent
  directions. Keep their primal evaluations, but propagate Hermite derivatives
  only for the center lane. Reuse their identical frozen action embedding.
- Sample LAP replay hierarchically: choose a block by its total priority, then
  a row within that block by its priority. Recompute touched block masses after
  every priority update and rebuild after rollout insertion. Padding has zero
  mass. This preserves each row's probability exactly in real arithmetic,
  subject to normal floating-point summation differences. It uses two uniforms,
  so the seed's random-number sequence differs from v2.

An independent read-only review found no concrete preaction, replay-cache, or
CUDA-capture rollback bugs. 44 CPU mathematical checks pass, covering score derivatives, gate behavior,
zero-tangent elimination and gradients, exact categorical proportions, zero
mass prefixes, and duplicate priority updates. CUDA equivalence and repeated
throughput measurements are submitted separately. The benchmark forces point
queries in both versions to isolate performance changes and compares critic
outputs, directional derivatives, losses, and parameter gradients before
measuring full 1M-row replay training graphs. This is not evidence of an
end-to-end training speedup until measured.

| Job | Actor objective | Actor query | Target sigma | Priority | Wall limit |
| --- | --- | --- | --- | --- | --- |
| 12690 | CUDA equivalence + throughput benchmark | point | 0.2 | 4 | 20m |
| 12691 | DG | preaction | 0.2 | 3 | 20m |
| 12692 | DG | preaction | 0 | 3 | 20m |
| 12693 | DG | point | 0 | 3 | 20m |
| 12694 | PG | preaction | 0.2 | 3 | 20m |

All four training runs depend on successful benchmark job 12690, declare
machine-wide concurrency 1, request 8M HalfCheetah transitions, and use seed 1.
Each runs autocull against the original v8 reference within its job process
group. Existing v2 DG job 12688 supplies the point-query / target-sigma-0.2
corner. Queue waiting does not consume the 20-minute wall-clock budget.


## First v3 evidence

CUDA equivalence job 12690 succeeded. Repeated synthetic train-API timing was
810.0 us/update for v2 versus 763.4 us/update for v3: 6.1%
higher optimizer throughput. Critic/actor gradient comparisons passed the
specified floating-point tolerances. This is optimizer throughput, not a
measured whole-run speedup.

DG preaction with target sigma 0.2 (12691) remained at approximately -594
greedy evaluation return through 1.46M environment steps and ended at its
20-minute limit. Its first logged rollout-level policy KL was 23,361; later
KL was zero and sigma_mean was constant at 0.17228 with min exp(-5) and max 1.
About half the sampled preaction coordinates exceeded the physical bounds.
This supports early actor saturation and subsequent immobility, not a NaN
failure; the critic loss continued declining on the failed policy's data.
The logs do not directly measure mean-head gradients or prove why saturation
occurred. Critic dependence on redundant context/unclipped preaction, and the
absence of a policy-distance limit, are hypotheses rather than established
causes. Zero-smoothing preaction (12692) reached evaluation 7,321 at 409,216
steps, so preaction conditioning alone does not determine failure in these
single-seed runs. PG preaction and point/no-smoothing controls remain needed.


## Direct learned-V experiment (v4)

The user clarified that the goal was testing latent-Q's **V function**.
The v3 preaction experiment changed the action-value channel; its center V was
only a diagnostic. That did not directly test learned V as the policy baseline.

`cleanrl/td7/delight/td7_latentq_value_baseline_v4.py` now computes
`A(s,u) = Q_point(s,clip(u)) - V_latent(s;mu,sigma)`. Point-Q uses critic inputs
`(a,0,a)` and SALE at physical action a. Learned V is the center query
`(mu,sigma,mu)` with SALE at mu. All advantages, critic values, and V inputs are
detached; the only actor gradient is the fixed-u Gaussian score. Neither the
preaction-conditioned action channel nor pathwise context derivatives drive
the actor. Both learned V and Gaussian cubature V are evaluated on every
actor update for gap mean/RMS, advantage-sign disagreement, means and positive
advantage fraction. The DG cubature control changes only which baseline is
used. Keep the existing Hermite, observed-preaction and physical-point critic
training losses rather than distilling learned V toward its cubature control.

All policy-target variants bootstrap with the target actor's actual sigma.
This evaluates the learned center under the target stochastic policy instead
of substituting fixed 0.2 smoothing. The critic still uses conservative
min-twin backups and target-value range clipping. LAP prioritization,
nonlinear Huber regression, finite network capacity and replay/current-context
shift make the center an approximate V, not a guaranteed conditional mean.
The shared backup change means v4 versus historical v2 is not a pure baseline
ablation; learned-V DG versus cubature-V DG **within v4** isolates baseline.

Cancelled active v3 no-smoothing preaction job 12692 on the user's instruction.
Cancelled queued preaction PG 12694, obsolete floor-enabled PG 12683, and old
point/no-smoothing job 12693 (replaced below with fixed runtime inheritance).
Job 12691 had already ended at its time limit before the cancellation request.

Autocull had not been functional inside queued jobs: the runner removes
XDG_RUNTIME_DIR unless explicitly inherited, making nested mlq clients seek a
socket in the wrong directory. Future submissions explicitly inherit
XDG_RUNTIME_DIR (and MLQUEUE_RUNTIME_DIR when set). A supervisor preflight now
fails before GPU allocation if the queue cannot be reached. CLI failures print
stderr and failed cancellations no longer count as successes. Watch interval
is 60 seconds; opt-in collapse grace is 200k with a 50k EMA, a 50% decline from
own prior peak, and at least a 2000-point decline. Generic performance grace
and global thresholds are unchanged. Reference is the completed 1.67M point-DG
v1 curve, extending comparison coverage beyond the original 818k v8 curve.
Historical replay with these scoped settings catches the failed preaction v3
at 600k while keeping successful point-DG/PG v2 runs. The manual cancellation
of 12692 was not an automatic-cull verdict.

| Job | Objective | Baseline | Bellman std | Priority | Wall limit |
| --- | --- | --- | --- | --- | --- |
| 12698 | GPU compatibility and throughput validation | cubature | fixed 0.2 | 4 | 20m |
| 12699 | DG | learned latent V | target policy | 3 | 20m |
| 12700 | DG | cubature V control | target policy | 3 | 20m |
| 12701 | PG | learned latent V | target policy | 3 | 20m |
| 12702 | DG | cubature V | fixed zero | 3 | 20m |

Training jobs declare concurrency 1, HalfCheetah-v4, seed 1, an 8M transition
ceiling, and depend on successful validation job 12698. Its comparison forces
cubature/fixed mode in v4 to match the existing v3 point-query control. It
compares the common gate/V-gap metrics; v4-specific calibration diagnostics
are additional outputs. The new learned-V/policy-target branch is exercised
by the full training experiments. Pure mathematical tests and independent
implementation review passed before submission; GPU results remain pending.

Validation job 12698 succeeded: common critic/actor values and gradients matched
in cubature/fixed compatibility mode. 74 mathematical and supervisor checks
passed. The new learned-V/policy-target training branch is running in 12699;
its supervisor preflight and live daemon connection succeeded.

## Exact v1 PG control

On explicit user request, job 12703 reruns the original v1 script with
`--actor-objective pg`, matching the DG reference
`HalfCheetah-v4__td7_latentq_delight_cubature_v1__1__1791688686`.
Keep its learned entropy floor, cubature baseline, physical point-Q advantages,
fixed 0.2 target std, network/optimizer settings and replay/update cadence.
Only the mean-score DG gate is removed; scale already used PG in the DG run.
HalfCheetah-v4, seed 1, 8M transition ceiling, priority 3, concurrency 1 and
20-minute wall limit. Supervision uses the repaired queue runtime environment
and one-minute watch. References are pinned to the exact original DG run so
a newly created PG control cannot accidentally become the culling yardstick.


## Restore the successful v1 properties (v5)

`cleanrl/td7/delight/td7_latentq_value_floor_v5.py` starts from the original
v1, preserving its networks, full replay-priority scan and random draw count,
all critic/encoder losses and three critic supervision lanes, optimizer rates,
mean-only DG gate, alpha optimizer, entropy target sigma 0.1, fixed critic target
sigma 0.2, gradient clipping, action samples and update cadence. It does not
carry forward v4's no-floor or target-policy-sigma changes. The sole scientific
ablation is actor baseline: learned center V versus original cubature V.

Cubature V uses the physical-point critic lane `(a,0,a)` at the 12 clipped
Gaussian nodes `mu +/- sqrt(6)*sigma_i*e_i`, averaging both twin values and nodes.
Learned V is the same twin-mean critic at center `(mu,sigma,mu)`. Both are
computed for diagnostics and detached from actor gradients. The sampled point-Q
numerator is shared across the two baseline configurations. Cubature control
matches the original v1 score/entropy math; learned-V changes only baseline.
The center remains an approximate expectation under Hermite/LAP/Huber training.

Independent code/AST review found the original network classes, critic/encoder
losses, replay step, alpha tuning, targets and update cadence unchanged. Pure
score-gradient checks passed for both baselines, PG/DG, both gate scopes and
entropy coefficients. CUDA equivalence and checkpoint roundtrip validation
will precede full seed-1 HalfCheetah experiments through mlq.


v5 saves `runs/<run_name>/checkpoint.pt` atomically at completed rollouts,
every 300 seconds and at graceful termination. The saved learner contains
all seven models/targets, all four Adam states, alpha, filled replay/priority,
cursors, target extrema, counters, and Python/NumPy/host/Torch/CUDA RNG streams.
`--resume PATH` restores tensors in place so captured CUDA graphs keep their
parameter and optimizer buffer addresses. Scientific arguments must match;
run name, total transition budget and runtime settings may change. Environment
trajectories restart with stochastic current-policy phase warmup, so this is
learner continuation rather than bitwise continuation of MuJoCo trajectories.
SIGTERM requests an exit at a completed rollout; SIGKILL or a shorter queue
grace can leave only the preceding periodic checkpoint. New checkpoints do
not recover the original v1 run's lost state.

92 pure mathematical, checkpoint and supervisor tests passed. Validation job
12709 compares v1 against v5 in cubature mode, including policy gradients and
alpha entropy loss, then serializes/reloads a full 1M-row replay checkpoint
and verifies that the next captured actor/alpha/hard-target update is identical.
The validation file is on disk within benchmarks and deleted afterwards.

| Job | Experiment | Priority | Wall limit | Concurrency |
| --- | --- | --- | --- | --- |
| 12709 | CUDA equivalence + checkpoint validation | 4 | 20m | 1 |
| 12710 | DG, learned latent V, original v1 settings | 3 | 20m | 1 |
| 12711 | DG, original cubature V, original v1 settings | 3 | 20m | 1 |

Both full training jobs depend on successful validation, run HalfCheetah-v4,
seed 1, with an 8M transition ceiling, preserve the queue runtime environment,
and use the repaired 60-second autocull supervisor. The matched cubature run
also recreates the original v1 behavior with checkpointing for continuation.


### Priority and checkpoint-validation correction

The user explicitly requested priority over other queued work. New validation
jobs use priority 11; replacement full experiments use priority 10. Limits stay
20 minutes and concurrency 1. Job 12709 failed an overstrict bitwise-next-update
checkpoint check; dependent experiments 12710/12711 were skipped, not trained.
Diagnostic job 12714 restored every saved state tensor, RNG state and captured
address exactly, and post-update RNG matched too. The replay-priority update
selected slightly different rows (only two entries differed among 1M priorities),
then the networks diverged as expected from different minibatches. This suggests
the original FP32 CUDA cumulative-priority scan's nondeterministic summation;
[PyTorch documents CUDA floating-point cumsum nondeterminism](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).
The diagnostic must isolate checkpoint recovery from scan rounding without
changing the original trainer or weakening tensor equality tolerances.

Replacement submissions after the explicit priority request:

| Job | Experiment | Priority | Wall limit | Concurrency |
| --- | --- | --- | --- | --- |
| 12719 | CUDA equivalence + two-phase checkpoint validation | 11 | 20m | 1 |
| 12720 | DG, learned latent V, original v1 settings | 10 | 20m | 1 |
| 12721 | DG, original cubature V, original v1 settings | 10 | 20m | 1 |

Training jobs depend on successful 12719. Validation separately requires exact
restoration of nonuniform replay and all learner/RNG/graph addresses, then exact
next-update equality with uniform priorities in the same captured training graph.
Uniform priorities are a validation fixture only; both actual training runs retain
the original nonuniform prioritized replay. A separate sampler probe compares the
CUDA draws, cumulative sums, and selected indices. This preserves exact equality
checks without asserting determinism that the original weighted scan cannot promise.

Job 12719 succeeded. v1/v5 cubature losses and gradients passed equivalence.
The independent original-sampler probe reproduced identical random uniforms but
different floating-point cumulative sums (maximum absolute difference 0.25),
which changed 17 sampled indices. This confirms why the original bitwise weighted
next-update assertion failed. Every saved nonuniform learner/replay/RNG state and
captured address restored exactly. With uniform priorities in the validation
fixture, the next complete actor/alpha/hard-target update matched bitwise across
all learner and RNG state. Actual training retains original weighted replay;
12720/12721 were released by this successful validation at priority 10.

### Learned-V v5 diagnosis (run 1791703320)

Read-only analysis of recorded events and implementation; windowed data are in
`value_floor_v5_diagnosis.json`. These numbers are 40k-transition trailing means,
not significance estimates or averages over seeds. The matched v5 cubature run
was still below 400k when this analysis was performed.

| Step | Learned-V v5 return | Original v1 cubature return |
| --- | --- | --- |
| 200k | 8578 | 10094 |
| 400k | 13851 | 12010 |
| 600k | 14117 | 15812 |
| 800k | 16257 | 15938 |
| 900k | 16592 | 16786 |
| 950k | 15940 | 16842 |

Learned V is above its same-state/current-policy point-Q cubature estimate by
about 0.057–0.096 in these windows. Across-state RMS disagreement is 30–39%
of sampled advantage RMS; 12–15% of sampled advantage signs change. The learned
baseline produces mean gates around 0.81–0.85 versus approximately 0.89–0.91
in original v1. These are direct observations, not proof of the late decline's
cause. Its last 50 noisy episodes average 15191 at 966k, whereas the latest greedy
evaluation at 933504 was still 16333 (the 900736 evaluation was 17278). There is
no evaluation after the final decline, so a collapse of the mean policy has not
been established.

Mechanism: ordinary antithetic mean-score PG cancels an action-independent shared
baseline exactly in each pair. DG instead uses f(A)=2*A*sigmoid(k*A), where k is
shared by the pair and depends on rarity and detached global advantage RMS.
The pair mean-gradient coefficient is f(Qplus−V)−f(Qminus−V). f is nonmonotone
when k*A is below approximately −1.28; sufficiently overestimated V can suppress
or even reverse a pair's Q ranking. Exact symmetric advantages recover the PG
pair coefficient. Log-scale score gradients do not have exact finite-sample
baseline cancellation and can also be affected, despite retaining ungated PG.

Why V need not be calibrated: its center is fitted through center+tangent to
replayed single-action TD labels, using robust loss and residual-dependent LAP
sampling without importance correction. Stored behavior (mu,sigma) differs from
the current actor context. No loss enforces that the center equals the integral
of the physical point-Q branch under the current actor. Cubature and sampled Q
use the same branch, so shared additive errors cancel; the learned center query
can introduce a cross-context calibration error. These structural facts explain
susceptibility, but do not identify which loss/replay/context effect dominates.
Both variants retain the entropy tuner; mean sigma and alpha remain comparable,
so the removed-noise-floor failure is not the explanation here.

A discriminating next experiment would train/calibrate current-policy latent V
against a detached point-Q expectation (fresh integration samples, independent of
actor scoring), preserving original critic/actor/exploration settings. Pairwise
DG direction reversals and same-checkpoint baseline gradient comparisons should
be measured before claiming the late decline is caused by V miscalibration.
A fresh greedy evaluation of the final checkpoint would distinguish behavioral
noise degradation from mean-policy deterioration.

### Comparison with successful preaction v5 (run 1790309009)

The old `cleanrl/td7/latentq/td7_latentq_preaction_v5.py` actor directly differentiates
`−mean(critic(s,[mu,sigma],mu,zsa(mu))) − alpha*sum(log_sigma)` through the center's
policy inputs. Its observed-preaction lane is a critic-training channel, not a
sampled preaction score actor. New DG learned-V actor instead detaches that center,
subtracts it from physical sampled point-Q, and differentiates gated log-probability
scores. Useful policy-input derivatives in the old method do not establish
cross-query integral calibration required for DG advantages.

Additional verified confounds: old single-environment episode-adaptive training
bursts versus new 16x1024 fixed rollout bursts; old unclipped optimizer gradients
versus new 0.5 encoder/critic/actor gradient clipping; old evaluation of a selected
checkpoint policy versus new evaluation of current policy. Both retain target
sigma 0.1, target smoothing sigma 0.2, and Hermite order 2. Raw training returns
also show the old model ahead (40k windows: 600k 16728; 800k 16904; 950k 17783),
so checkpoint selection alone does not explain its advantage.

A faithful test of adding DG to the old winner must preserve its surrounding
setup and compare old direct-center actor, point-Q score PG, cubature DG, and
learned-center-V DG. Current v5 isolates actor baseline relative to v1 DG; it is
not an isolated DG intervention on old preaction v5, nor evidence against latent-Q
as an architectural idea.

### Native Hermite delight policy v6

User requested an independent design review and stewarded implementation using
latent V, without the rejected extra point-Q Monte Carlo distillation. Two
independent agents converged on the critic's native Hermite advantage. The rejected
calibration prototype was removed and never submitted.

The learned distribution value is the twin mean at
`V(s,mu,sigma)=critic(s,[mu,sigma],preaction=mu,zsa(mu),zs)`.
Its total mean derivative includes latent mu, preaction mu, and SALE's action
embedding. With detached derivatives, the actor advantage is
`H(eps)=sum(sigma*eps*dV/dmu)+0.5*sum(sigma*(eps^2−1)*dV/dsigma)`.
This is precisely the reconstruction tangent already used in critic training.
The native actor uses the existing four antithetic raw Gaussian actions only,
with no point-Q forward, cubature baseline, teacher, or additional samples.

For Gaussian noise, `E[H]=0`, `E[H*score_mu]=dV/dmu`, and
`E[H*score_logsigma]=sigma*dV/dsigma`. Thus ordinary score PG recovers the
successful direct-center actor gradient in expectation. DG gates this same local
geometry rather than subtracting potentially miscalibrated values from two critic
contexts. Only policy scores and the original entropy term receive actor gradients.
Constant additive shifts in V cannot affect H. First-order-only antithetic DG is
identical to PG for the mean; the existing second-order term makes the gate
meaningful. These guarantees describe a learned Hermite surrogate, not an unbiased
true environment advantage or a guarantee that DG improves benchmark scores.

The actor change leaves critic TD/Hermite losses, network architecture, replay/LAP,
learning rates, entropy target sigma0.1, target smoothing sigma0.2, gradient
clipping0.5, rollout cadence, and checkpointing unchanged from v5. `actor_advantage
point` retains the original actor explicitly for equivalence testing; default
`hermite` selects the new policy. Native diagnostics report center mean, odd/even
Hermite RMS, analytic surrogate RMS, pair-coefficient cosine, and pair reversal
fraction. Pair-coefficient cosine is not the full parameter-gradient cosine.
Four actions contain two independent directions per state, so finite-sample
variance remains a limitation; native PG is the matched control.

123 pure mathematical, gradient-boundary, checkpoint and supervisor checks passed.
Independent AST and scientific review found no blocking defect. CUDA validation
compares the retained point-mode actor and unchanged critic with v5, verifies the
native geometry against the trained critic JVP, verifies score-only gradients and
constant-offset invariance, then measures captured native updates and checkpoint
recovery. No production sample count is increased by these validation fixtures.

| Job | Experiment | Priority | Wall limit | Concurrency |
| --- | --- | --- | --- | --- |
| 12726 | Native CUDA/JVP/score/checkpoint validation | 13 | 20m | 1 |
| 12727 | Native Hermite DG | 12 | 20m | 1 |
| 12728 | Native Hermite ordinary PG control | 12 | 20m | 1 |

Both seed1 HalfCheetah jobs have an 8M transition ceiling and depend on successful
12726. They run the foreground-process-group autocull supervisor every60seconds,
reference the original successful DG cubature run, and save atomic learner
checkpoints. Queue admissions were resumed for the user's authorized runs.

User requested cancellation and requeue while sleeping, with mlq paused. Validation12726 was cancelled gracefully; dependent12727/12728 skipped without training. Replacement validation and both training submissions retain priorities13/12, concurrency1, 20-minute limits and successful-validation dependencies. Queue remains paused; no automatic unpause is scheduled. Replacement job IDs: 12732, 12733, 12734.

### Native Hermite v6 learning dynamics and requested ablations

Validation12732 succeeded after admissions resumed: compiled native actor,
trained-JVP agreement, score-only gradient, offset invariance, and serialized
checkpoint recovery all passed. Native ordinary-PG job12734 was cancelled before
starting, so there is no native PG learning control yet. Training12733 is run
1791738398. The user's subsequent request is two independent native-DG ablations:
remove entropy floor with target smoothing0.2 retained; or target smoothing0 with
entropy floor retained. Neither adds samples or changes the native advantage.

At approximately600k transitions, trailing40k aligned means are:

| Metric | Native Hermite v6 | Original cubature DG |
| --- | --- | --- |
| Training return | 10847 | 15812 |
| Greedy evaluation | 11760 | 16224 |
| Arithmetic sigma mean | 0.210 | 0.151 |
| Critic tangent std | 15.65 | 2.60 |
| Actor advantage RMS | 9.48 | 0.73 |
| KL per fixed rollout update burst | 106.1 | 20.3 |
| Clipped action fraction | 0.336 | 0.341 |

Critic/sigma/target tags use learner steps while actor/return tags use environment
steps; the table aligns v5/v6 critic tags with the approximately32k warmup offset.
Native odd Hermite RMS9.38 vs even1.275 means the even term contains only about
1.8% of their combined measured squared energy. Pair-coefficient cosine0.9975 and
reversal fraction0.0012 show the gate barely changes pair directions. For purely
odd antithetic advantages, complementary gates sum to2 and DG equals PG exactly.
Thus the dominant actor is a sampled reconstruction of the original latent-V
mean gradient rather than substantial delight selection.

With h=sigma*dV/dmu and M=2 independent Gaussian directions/state, the ordinary
mean-score estimate is diag(1/sigma)*mean(eps eps^T)*h. Its expectation is exactly
dV/dmu, but coordinate variance is (sum(h^2)+h_j^2)/(M*sigma_j^2). Hot coordinates
can contaminate cold-coordinate estimates; logged sigma extrema0.0067 and1 imply
possible strong anisotropy, though extrema alone do not establish per-state
conditioning. Gradient clipping and Adam are nonlinear, so equality in expectation
before those operations does not guarantee equal optimizer behavior. The scale
score cancels the odd term exactly, leaving noisy fourth-moment estimation of the
even term. Broader behavior, harder prioritized TD targets and larger center
sensitivities can reinforce actor noise; this is a hypothesis, not a proven cause.

Higher sigma alone is insufficient: original successful preaction v5 also had
sigma≈0.19 at600k, and physical clipping fractions are comparable. The entropy
floor targets average log sigma/geometric mean, not arithmetic sigma; native
alpha≈0.032 is actually lower than cubature≈0.081. Removing it tests pressure on
heterogeneous dimensions, but may worsen conditioning if cold dimensions shrink.
Zero target smoothing tests the backup query's fixed sigma0.2 assumption; it does
not directly remove rollout noise or score-estimator variance.

min_target/max_target are cumulative extrema of bootstrapped TD labels, never
reset. A lower-tail event can move min_target permanently; it is not a lower bound
on policy returns. The lower native upper-target range tracks its poorer returns.
Higher HermiteR2 can reflect broader residual variation and is not independent
proof that the policy/value estimates improved.

A later variance intervention can integrate ordinary Hermite score-PG analytically
and use the existing four samples only for the nonlinear DG-minus-PG correction.
That is variance reduction of the same score expectation, with detached critic
coefficients and no additional point-Q queries. It is not part of the requested
floor/smoothing ablations; first measure those independently.

Version7 implements only the requested independent ablations, keeping default
version6 math bit-exact. `entropy_floor=False` removes the actor entropy term and
alpha tuning; the natural Hermite sigma-score update remains. Effective alpha is
logged as zero. Disabled alpha Adam state remains empty and restores without
creating spurious entries. `target_smoothing_sigma=0` changes only the TD backup
context; entropy is retained in that experiment. Both preserve four actor samples.
161 mathematical, gradient, ablation, checkpoint and supervisor tests passed.

| Job | Experiment | Priority | Wall limit | Concurrency |
| --- | --- | --- | --- | --- |
| 12737 | No-floor native CUDA + checkpoint validation | 13 | 20m | 1 |
| 12738 | Zero-smoothing equivalence/CUDA + checkpoint validation | 13 | 20m | 1 |
| 12739 | Native DG, no entropy floor, smoothing0.2 | 12 | 20m | 1 |
| 12740 | Native DG, original entropy floor, smoothing0 | 12 | 20m | 1 |

Each training run depends on its corresponding validation. HalfCheetah seed1,
8M transition ceiling, original autocull reference and atomic learner checkpoints
remain. No combined no-floor/zero-smoothing run was submitted.

Both v7 CUDA validations succeeded (12737/12738): native geometry/JVP, gradient isolation, constant-offset invariance, full compiled graph and serialized checkpoint recovery passed, including genuinely empty disabled-alpha Adam state. Zero-smoothing v6/v7 original losses/gradients matched. Job12739 no-floor is running;12740 zero-smoothing is queued next at priority12,20-minute limits. Original v6 hit its20-minute limit and saved checkpoint at1457792 transitions; latest greedy evaluation16682.9 at1425024 transitions, last50noisy episodes14497.2, showing continuing improvement with instability. Its final min bound−298.36 and max1696.76 are cumulative target extrema, not policy-return bounds.

### Empirical check of minimum-target predictive value

The user's observation should not be dismissed from metric semantics. Across40
HalfCheetah latent-Q runs,13 have approximately1M-step training outcomes and11
have greedy evaluation outcomes at that horizon. A larger drop in min_target
between200k and400k learner updates predicts better1M scores: rank correlations
+0.58 training/+0.65 greedy. After residualizing ranks against400k training return,
those shrink to+0.29/+0.32. Absolute min_target at200k is different: correlation
with1M return is+0.43, meaning less-negative early minima tend to score better in
the full mixed-family cohort. Among9 runs already scoring>5k at200k, that absolute
association is weakly negative(−0.20). Within discussed runs, min decreases as
returns increase (rank rho roughly−0.97 native/−0.92 cubature/−0.64 original),
strongly confounded by training age.

Minimum expansion is therefore a plausible leading indicator of exploration or
critic development, not a direct best-value record or a universal lower-is-better
criterion. It also affects training: the historical minimum is the lower target
clipping bound. With y=r+gamma*clip(q,m,M), extending m downward requires observed
r<(1−gamma)*m for nonterminal transitions. Broad negative reward tails plus learned
pessimistic continuation, not arbitrary critic excursion alone, can expand it.
This may unlock bad-state discrimination/avoidance and locomotion coverage.
Native Hermite expands negative values without equally fast positive-value growth,
so both bounds and actual returns matter. These are exploratory mixed-method,
mostlyseed1 associations subject to culling/completion bias, not causal or
out-of-sample predictor validation. Data, summary, and figure are saved as
min_target_predictor_analysis.json, min_target_predictor_summary.json, and
min_target_predictor.png in this folder.

### Analytically integrated Hermite delight v8

User authorized integrating ordinary Hermite score PG and logging gradient norms.
New frozen version: `cleanrl/td7/delight/td7_latentq_delight_hermite_integrated_v8.py`.
For detached center derivatives, the ordinary Gaussian score expectations are
`g_mu` and `sigma*g_sigma` for mean and log-sigma respectively. The actor-only
linear surrogate supplies these exact expectations; the existing four antithetic
actions supply only the unchanged sampled DG-minus-PG correction. Gate RMS,
surprisal, scope, critic coefficients and action detachment remain unchanged.
No new samples, point-Q queries, or actor path through critic parameters.
Equality concerns expected raw gradients conditional on replay states; nonlinear
clipping and Adam updates can differ. Actor-loss scalar values change with the
surrogate representation and should not be compared directly to v6/v7 losses.

Defaults retain entropy tuning (target geometric sigma0.1) and use target smoothing0,
based on the completed independent v7 ablation. At600k/1M/1.4M, zero smoothing had
training returns14265/15601/16012 versus v6's11124/13580/14812. Single-seed evidence;
no claim of final superiority. No-floor job12739 was canceled at654k and performed
worse at600k; it is not a completed long-horizon comparison.

Each captured encoder/critic/actor update accumulates the pre-clip norm, update
count, clipping count, applied multiplier and maximum norm on CUDA. Interval
statistics are logged every500 learner steps at
`gradients/{encoder,critic,actor}/{preclip_norm_mean,preclip_norm_max,clipped_fraction,clip_multiplier_mean,update_count}`.
Actor counts include only delayed actor updates. Statistics participate in capture
state restoration and checkpoints. Clipping remains0.5 for all three networks.

Independent mathematical/code review found no blocker.210 checks passed, including
exact Gaussian moment agreement, gradient isolation, the unchanged sampled delight
correction across both gate scopes/scales/surprisal definitions, unchanged critic
math, point control, checkpoint buffers and accumulator statistics.
CUDA validation12773 is queued at priority14; integrated-DG training12774 depends
on its success and is queued at priority13. Both limits20minutes, max concurrency1;
training ceiling8M transitions,seed1,16envs,threads2. The foreground wrapper runs
scoped automatic culling against the original cubature DG reference. Commands and
submission records are in jobs.json. CUDA validation and learning outcomes are
pending at submission time.

V8 outcome: CUDA validation12773 succeeded. Training12774 completed its20-minute
budget (mlq marks this failed/time-limit) at~1.46M transitions. Training returns
600k/1M/1.4M were15462/16394/16091, compared with zero-smoothing v7's
14265/15601/16012. Latest greedy evaluation17538. Thus analytic integration
improves early learning, but matched1.4M noisy training scores are similar;
long-term dominance remains unestablished.

Clipping telemetry after100k learner steps: actor and critic clipped on100% of
updates. Mean interval pre-clip norms18.88 and292.16; mean applied multipliers
0.0409 and0.0024. Encoder clipped on29.61%, mean norm0.448, mean multiplier0.921.
These are raw gradient scalings, not equivalent reductions in Adam learning rate.
This establishes strong clipping in v8; old v6/v7 frequency remains unmeasured.
Agreed machine coordination window remains open for session2090fdc8's repaired
bounded trading chain; no additional Hermite GPU jobs submitted during its repair.

### Raw centered Hermite gate comparison

User questioned RMS advantage scaling and prefers raw, centered DG. Rechecking
DPG Section2 and continuous Algorithm2 confirms advantage RMS is an adaptation,
not part of that algorithm. Ours sets effective gate temperature to
`temperature*RMS(A)`; removing RMS retains absolute advantage magnitude and lets
uniformly shrinking advantages approach a neutral gate. Neither design is yet
shown optimal for this latent-Q setup. The paper additionally normalizes rewards
(EMA decay.999), whereas these trainers use original environment reward units.

Raw comparison uses the frozen v8 trainer's existing `--gate-scale raw` argument.
No algorithm file or checkpoint semantics changed. Hermite A remains centered
exactly under the Gaussian expectation. We do not subtract finite sample means:
the even term can have a nonzero4draw mean, and same-sample recentering changes
nonlinear DG and invalidates the integrated base's sampled-PG equivalence.
The existing relative surprisal, doubled gate, mean-only scope, entropy target,
zero target smoothing, integrated actor estimator and clipping instrumentation
are retained, making this an RMS-only ablation rather than exact paper DG.
A literal gate reproduction would additionally require signed density surprisal,
all-score gating and removing factor2; reward units also require explicit treatment.

114 focused checks passed; independent literature/mathematical review supports
this comparison. CUDA validation12930(priority14) and dependent training12931
(priority13) queued, each20min,maxParallelRuns1. Training8Mceiling,seed1;
run_hermite_integrated_raw_supervised.sh runs scoped culling and full tests.
The earlier trading session is no longer live and no trading chain is queued;
coordination ask could not reach that session. Jobs/commands recorded in jobs.json.

### Original cubature winner with raw gate units v9

User explicitly requested original run1791688686 with raw gating, then extended
this run's limit to30minutes. New trainer
`cleanrl/td7/delight/td7_latentq_delight_cubature_raw_v9.py` copies frozenv1.
Only scientific default change is gate_scale=raw. Original actor/critic/
encoder losses, architectures, four antithetic samples,12node baseline, mean-only
scope, relative rarity, alpha tuner sigma0.1, target smoothing0.2, optimizers,
clipping0.5 and rollout/update cadence are retained. No integrated-Hermite actor
or zero-smoothing improvement is imported. Passive allupdate clipping statistics
are copied from reviewedv8 and included in capture restoration. Norms, fractions,
multiplier and count log every500learner updates. No checkpoint functionality is
added to this original trainer.

111 checks passed, including AST equality of scientific methods, equality of all
other argument defaults and identical telemetry semantics. Prediction: original
actor A has smaller magnitude than nativeHermite; removing RMS may soften its
gate in parts of training, while retaining magnitude changes as learning progresses.
Performance benefit is uncertain; compare matchedsteps against originalv1.
GPUvalidation12958(priority14,20min,max1) is queued against originalv1 with both
using raw gating. Training12959(priority13,30min,max1) depends on its success;
seed1,HalfCheetah-v4,8Mceiling,16envs,threads2,compiled,scopedautocull retained.
Exact commands and submissionmetadata are in jobs.json. Validation/results pending.

### Raw cubature performance investigation

Job12959 reached its30minbudget at~1.95Mtransitions; no NaN/health failure identified
from available queue/scalar evidence. OriginalRMS ended~1.67M. score_runs matched
trainingreturn600k/1M/1.4M/1.6M:
RMS16160/16654/17030/17312; raw16228/16893/17032/16681.
Raw is not an established early regression; relative underperformance appears
late. Its greedy evaluation dipped around1.6M and recovered above18k later;
latest18035at1.916M vsRMS18455at1.654M are unmatched endpoints.

Observed mechanisms, not causal conclusions:
* Raw weakens saturation because point advantages commonly RMS0.6-0.9. At1.6M
matched logged snapshots, negativegate suppression3.8%raw vs8.3%RMS, positive
saturation0.84% vs1.65%. Mean gates are similar, so averageweight alone misses it.
* Late policy drift grows: raw ±30k-window KL~35.8 at1.6M vsRMS~14.1 and~98.9
around1.9M. These are rollout-burst KL(old||new), not peroptimizerstep trust regions.
Sigma around1.6M is similar(raw~.158,RMS~.162 on learnerstep windows); a simple
excesssigma story is not supported. Raw actor/critic clipping is100%; original
frequency was not logged, so a clipping difference cannot be established.
* Actor Amean consistently roughly−.08to−.10. Twelve Gaussian cubature nodes are
exact only throughdegree3; clipped actions and nonlinearpointcritic violate that
integrand assumption. Thus A is not guaranteed centered. Repeatednegativeaggregate
means suggest a cubature baseline bias, but do not identify per-state error.
A baseline error leaves ordinaryscorePG expectationunchanged yet changesDG gates.
* For antithetic A±=E±O and evenmode-relative rarity, purelyoddE=0 DGpairedmean
exactlyequalsPG regardlessraw/RMS. DGmean distinctions dependon evencomponents,
including truecurvature or baselineerror. Saturationdoesnotcertify usefulDG.
* Mean-onlyscope leaves scalePGungated. WholeSGDgradientdirectionafter sharedtrunk,
entropy andAdam is not capturedby gatefractions or Hermitepaircoefficientcosine.

Next diagnostic priority is actual actor gradient DG-minus-PG norm and cosine,
mean/scale decomposition, and ordinaryscore variance/covariance with learned
Hermite. A promising design is nativeHermite as a zero-mean controlvariate for
originalpointDG: analyticE[Hscore] + sampled[(Apoint−H)score] + unchanged
sampledDGminusPG correction gatedonApoint. It reuses the existing4actor actions
and12baseline nodes, adds no contrastivesamples, and preserves expectedpointDG
rawgradient even iflatentV is inaccurate. Variancecanincrease ifH poorlymatches
pointgeometry; measure covariancebeforeassumingbenefit. It doesnotfixbaseline
bias by itself. No training/benchmark job submitted during sharedCASMI/RLhold.
Windowedobservationssavedin cubature_raw_gate_diagnosis.json.

### Baseline-free design reconsideration

User asks whether baseline-free or nonapproximate centering is preferable. A rawQ
nonlinear gate depends on arbitraryvalue offsets; removing subtraction is not
necessarily removing the need for a meaningfulreference. Pairwiseantithetic
centering makes Aodd and causes mean-onlyDG tocollapseexactly topairedPG.
NativeHermite H alreadyhas Gaussianexpectationzero analytically, needs no extra
baselineintegration and is offsetinvariant, but still approximates physicalQ's
shape. Exact surrogatecentering is not exacttrueadvantage.

Preferred next hypothesis is native rawH with full-scoreDG, retaining analytically
integratedordinaryPG and existing4actions, rather than anotherbaselineestimator.
The main change from currentnativev8raw would be gate_scope=all. SymmetricoddH
leaves gatedmeanpairedPGunchanged but can change sigma, an explorationmechanism
omitted by our mean-only adaptation. Puretensorcheck of the realscorehelper with
1DoddA and antitheticε(.5,2,−.5,−2): PG and meanDG ascent(mu,logsigma)=(2.125,0),
allDG=(2.125,2.886). This is a finite mathematical example, notperformance evidence.
Prediction is a strongerDGsigmaeffect; excessivevariance is a materialpossible
failure and entropyfloor is no upperconstraint. Benchmark gain unknown.
The previouspointDG controlvariateproposal preserves its expectedtarget and
addressesvariance, but doesnotremove cubaturebaseline from gate; distinctobjective.

A properVcalibration alternative would revise centerlane's Huber/LAP projection
rather than add pointQintegration samples: squaredconditionalexpectation loss
with correctprioritysamplingweights. This changes critictraining and remains
subjectto bootstrap/capacity/contextcoverage, so do notclaim it guaranteestrueV
or isbetter than originalrobustloss. It is a separatetest if Vlevel accuracy is
required, notnecessary for nativeH's algebraiccentering. No jobs queued duringhold.

### User-requested independent Claude reconsideration of DG centering

User explicitly challenged centering claims and requested a Claude collaborator.
Spawned cagents Claude agentcf085a88 (sessiond9c924d2), prompt recorded separately.
Root reread the complete DPG article/appendices and inspected baselineFigure3.
CORRECTION: zero-meanadvantages are NOT a requirement ofDG. Section3 reports gains
for zero,constant,estimatedexpected andoracle baselines; Section4.2/Prop2 explicitly
uses b=0. Reward/valueoffsets changingDG is a designproperty, notproof that an
uncentered estimator is invalid. ExistingnegativeAmean isnotalone a defectdiagnosis.
Earlier exact-centering preferences were stronger than the paper/evidence supports.

Exact nativeHermite centering and firsttwo ordinaryPG moment identities do NOT
establish preservation of DG. Gate is nonlinear; higherHermite terms orthogonal
to ordinaryscores can affectDG. Witness underε~N(0,1):
A=(ε³−3ε)+0.4*(ε⁴−6ε²+3). E[A],E[Aε],E[A(ε²−1)] arezeroexactly, so a perfectV's
mean/scale derivatives yieldzero secondorderH at thiscontext. RawDG with
sigmoid(A*min(.5ε²,10)) hasnonzero expected mean/scale scores. Gaussianquadrature
128nodes gives(.29714,1.27800);256(.29547,1.27682). This is a mathematicalwitness,
nottraining evidence, and demonstrates lostDGstructure ratherthan baselineerror.

Antitheticodd cancellation is a symmetry-specific result, notgeneralDGfailure.
The paperitself includescases where preservingexpected direction with reduced
variance is useful. Full-score variance growth was a theoreticalwarning here;
no previousfull-scope trainingexperiment appears in thisjobhistory, so do not
reportit as measuredfailure. ProposedfullnativeHrun heldpendingjointreview.
No newGPU jobs submitted while sharedboundedworkloads run.

### Joint Claude/root decision: direct point-DG reference v10

The proposals immediately above are superseded. Claude read the full DG paper
and related material independently; root reread the full DG paper and challenged
both centering claims and proposed control-variate guarantees. We selected pure
point-DG, with no Hermite actor, integrated ordinary-PG base, fitted coefficients,
or additional training action draws. A control variate can help or hurt; toy
variance failures were enough to reject assuming it helps, not to prove it never
can. Neither exact centering nor a particular reference is prescribed by DG.

New trainer: cleanrl/td7/delight/td7_latentq_delight_reference_v10.py.
Four antithetic preactions are drawn at replay states, clipped for point-Q queries,
and scored with the unclipped Gaussian density. All Q/reference/gate quantities
are detached. The reference is either Q_point(s,clip(mu)) or learned V(s,mu,sigma).
Both consume five critic rows/state in training (four actions plus one reference).
Latent critic training, SALE, LAP, hard targets, cadence, gradient clipping, the
entropy floor and fixed sigma=.2 distribution-center backup are preserved.
The latter is a learned approximate query, not exact Gaussian integration, and
is not the current actor's sigma-dependent stochastic-policy backup.

Gate: 2*sigmoid(raw U * min(.5||eps||^2,10)), temperature1. Full-score is default;
mean-only is the control. Mode-relative rarity and factor2 are adaptations of the
paper's signed clipped log-density and sigmoid gate. They preserve rare-success /
rare-failure reweighting intent but are not claimed equivalent to its continuous
algorithm. Raw/RMS sets gate temperature; raw is a hypothesis, not established
optimality. DG baseline choice changes its expected gradient.

Full-score DG need not shrink sigma even at a modal quadratic optimum. In a
1D Gaussian toy U=-c*eps^2 and doubled sigmoid(U*.5eps^2), the expected scale
ascent changes from -.08082 at c=.1 to +.11866 at c=2; ordinary PG is -2c.
Suppressing large negative outer contributions can leave inner scale expansion.
This is a mathematical warning, not a measured training regression. Logs now
include signed PG/full-DG/trained scale coefficients, inner/outer components,
geometric sigma, entropy target error, actual actor-parameter DG-minus-PG norms
and cosines, entropy/reward gradients, and reference disagreement. The probe
uses saved last-step states/encodings/draws under current weights; it is finite
batch evidence, not a population gradient or a measurement of earlier weights.
Alpha being positive alone proves nothing; its trend and signed scale terms
must be examined together.

CUDA validation13002 (20m,priority14,max1) failed six diagnostic score-partition
equalities by up to3.22e-6;68 other checks passed. No training ran. The original
three dependent training jobs13003-13005 were skipped. Queued diagnostic13006
(10m) separates productionTF32, preciseFP32 and FP64, including supplied vs
reconstructed standardized noise. Full validation13008 (20m) depends on its
success. Strict FP64 algebra checks complement a precision-aware production
component-norm error bound; the numerical cause remains unconfirmed until
13006 completes. Non-network checks currently66passed/15 CUDAcases skipped.

New training chain13009 latent/all,13010 modal/all,13011 modal/mean: each30m,
priority13,max1,seed1,HalfCheetah,8M ceiling, validation-success dependent.
Autoculling uses greedy evaluation to avoid conflating exploration sigma with
training-policy reward; shared supervisor implementation is unchanged. New
run performance remains unknown. CASMI13001 already held the exclusive lease
when the diagnostic was queued; no foreign job was interrupted.

13006 numerical diagnostic completed successfully: production partition relative
norm error9.21e-5 modal /1.05e-4 latent; preciseFP32 ~4e-8; FP64 ~1.3e-16.
Supplied vs reconstructed noise changes were negligible compared with TF32 error.
This confirms separate-VJP TF32 rounding as the cause; no production arithmetic
or actor update change was needed. CUDA validation13008 is now running.

Validation13008 SUCCEEDED:81 CUDA-inclusive tests; critic values, JVPs and training
gradients exactly match the preserved v8 point/latent control. Actor loss matches
exactly, gradient maximum discrepancy1.99e-6 within the numerical contract.
Probe finite/no-state-or-gradient-or-RNG mutation checks pass for both references.
Serialized learner/replay/Adam/alpha/RNG/address restoration passes. Uniform-prefix
CDF fixture gives exact next captured update including hard targets; original
nonuniform FP32 scan association may differ in sampled indices after restore,
and that inherited limitation remains explicitly reported.

Queued single-process learner throughput was1114updates/s for v8 point/latent,
1169 for v10 latent,1214 for v10 modal at the full1M LAP scan,2000steps x3.
This is a diagnostic learner-only measurement, not aggregate or end-to-end
training speed. Training13009 started successfully;13010/13011 remain dependent
on the prior arm reaching a terminal state. Performance superiority remains
unknown. Claude final read-only review found no blocker, and its owned agent
was closed after the collaboration and validation completed.

User scheduling override: place our runs below parameter-golf. Changed queued
13010 and13011 from priority13 to -1; parameter-golf13007 is priority1.
13009 is already running; mlq only changes queued/held priorities, so its current
30-minute allocation remains. Future submissions in this task use priority-1
unless the user gives a new scheduling instruction.

User concerned sigma is harming learning. At~507k full-DG sampled log-sigma
coefficient+.174 versus ordinaryPG-.337, alpha~1.4e-11, geometric sigma~.145,
arithmetic~.39,~15% nearceiling. Positive tail contribution1.022 exceedsnegative
inner-.848. Finite-batch coefficients support altered DG scale incentive but
are not actual Adam-induced output changes. Modal-minus-latent gap~.82 and~29%
sign disagreement implicate reference sensitivity without proving Vincorrect.
Queued13014 (priority-1,10m,max1) inspects saved checkpoint's existing four
probe draws, partitions signed scale by advantage/inner-outer/action-clipping,
and computes raw-parameter-descent projection on mean sigma for eachreference.
No new training draws, optimizer updates, sigma clamp or entropy retuning.
All pending work remains below parameter-golf. Diagnostic will inspect whichever
atomic checkpoint is available when admitted; output includes checkpoint step.

### Team investigation of the v10 sigma regression

The user requested a thorough team investigation. Independent native reviewers
read the full DG article and audited critic code, and a new Claude collaborator
(agent745935df, sessionf035fd21) analyzed matched-step scalars and mathematical
fixtures. Root challenged proposed explanations and repairs; several were
revised or withdrawn. Evidence is in sigma_team_scalar_evidence.json and
claude_sigma_empirical_review.json. Toy results are mathematical examples,
not benchmark evidence. CUDA diagnostic13014 remains queued at priority-1,
with a10-minute limit, below parameter-golf. No live trainer was modified.

Measured matched-environment-step windows (600k +/-30k): greedy return v10
13248 vs original cubature16576 vs raw cubature16983; training return10461 vs
16290 vs16256. Clipped action-coordinate fraction .452 vs .341 vs .314.
The v10 geometric sigma is .1435 here, but arithmetic sigma is roughly .4.
Over450k-700k, v10 near-ceiling fraction median .196 and near-floor fraction
median .098. We cannot compare endpoint occupancy with old versions because
they did not log these fractions. All earlier versions also logged min/max
sigma near both bounds; claiming only v10 has endpoint values would be wrong.
The run was cancelled by request and saved a736896-step checkpoint; the queue
record does not establish who requested cancellation. Its culler last said keep.

The earlier positive+.174 DG scale quote was one batch. Across450k-700k the
median full-DG coefficient is+.0255, range-.363 to+1.034: mostly cancellation
of substantial inward/outward terms, while PG retains contraction. This is a
sampled output-space coefficient, not an Adam-induced output change. Advantage
RMS median6.26 (range3.39-43.22) versus original .686 and rawcubature .656.
Suppressed/saturated fractions are .210/.146 versus original .086/.030 and
rawcubature .047/.014. RMS is tail-sensitive; gate hardening does not by itself
prove that RMS normalization improves sigma.

The main structural result: if advantage A depends on other noise coordinates
but not epsilon_j, our doubled joint-rarity DG scale signal is
E[2 sigmoid(A*r/eta)*A*(epsilon_j^2-1)]
= (2/eta) E[epsilon_j^2*A^2*sigmoid_prime(A*r/eta)] >= 0.
The rarity cap retains a nonnegative term on its uncapped region. Ordinary PG
is zero for that coordinate. Thus genuine full DG can grow reward-flat-coordinate
variance even with perfect Q and exact global mean V; not a centering error.
With two Gaussian coordinates, Q=-epsilon_1^2, A=1-epsilon_1^2, numerical
quadrature verifies PG=(-2,0), relativeDG approximately(-.536,+.160).
Signed density also retains the flat-coordinate variance incentive. This
mechanism is established mathematically; actual reward-flat directions in our
trained critic are not yet established.

Our mode-relative rarity drops a consequential normalizer from the paper:
sum(log_sigma)+D/2*log(2*pi). At D6, geometric sigma .145 this is-6.0725.
Typical paper signed surprise is then negative while our rarity is positive.
Common six-dimensional fixtures were independently reproduced by Claude and
the native reviewer: Q=-.3*sum(first3 epsilon_i^2), modal reference, density
normalizer-6.0725 gives relative curved/flat(+.052,+.041), density(-.635,+.248).
With exact mean reference these become(-.281,+.051) and(-.422,+.117).
Different off-optimum/curvature fixtures produce other signs; density is no
universal repair. At a centered isotropic quadratic optimum every positive
radial gate must contract: conditional E[epsilon_j^2-1|R]=R/D-1 and
A=c(D-R), so expected scale=-cD*E[w(R)*(R/D-1)^2]<=0.
Root used that constraint to correct a misleading description of an off-optimum
Claude fixture. Full fixture parameters and reproducibility metadata were added.

Clipping creates a separate representation issue: overshoots producing the same
executed boundary action currently have different latent gates/scores. Physical
DG can score boundary tail probabilities and interior Gaussian densities using
an explicit mixed reference measure, with no extra training samples. It removes
overshoot nuisance but still has entropy-like flat-coordinate variance pressure;
it is a different full-DG action representation, not a promised cure.

Critic/encoder losses are AST-identical to cubaturev1, Hermitev6 and preactionv5.
No new critic-loss bug was found. The learned center is a shared Huber/LAP
center-plus-Hermite fit without importance correction and need not equal the
current Gaussian average point-Q. Fixed sigma=.2 continuation ignores current
policy sigma; it is inherited, but differs from stochastic-policy evaluation.
Using V as a differentiable objective in preactionv5 and as a nonlinear gate
threshold in v10 are different roles. Reference sign disagreement establishes
sensitivity, not incorrect V or a requirement for exact centering.

Two additional interactions: (1) hot/cold coordinates can degrade paired mean
credit estimates. For locally linear Q, paired variation depends on all
sigma_i*Q'_i*epsilon_i, giving cold coordinate j noise proportional to
sqrt(sum_i!=j sigma_i^2*Q'_i^2)/sigma_j. Antithetic pairing cancels a constant
offset, so 1/sigma_j alone is NOT evidence of a gradient explosion. Actual
paired coefficients and actor Jacobians must be checked. (2) alpha~1e-11 makes
the exponential log-alpha controller numerically sluggish after long slack.
It is correctly inactive while geometric sigma>.1 and cannot explain current
hot-coordinate pressure. Nonvanishing log-dual updates can still accumulate
windup. A projected nonnegative multiplier in alpha-space avoids that debt,
but changes early optimization and still constrains only global geometric
entropy. It cannot shrink hot coordinates. A zero initial multiplier would
explicitly test the observed shared early sigma=1 forcing/hysteresis; it is a
separate scientific choice, not a hidden controller-only repair.

Rejected proposals: restoring PG sigma as if pureDG; arbitrary sigma clamps;
assuming density/RMS fixes the problem; a same-four-draw empirical mean claimed
as action-independent. With two antithetic pairs, own-pair even credit is halved,
not the independent-four-draw3/4 factor. That biased self-reference was withdrawn
as first repair, and it would bypass the requested latent-V actor role.

A larger genuinelyDG candidate is coordinate-conditional credit using latent
V(s, hybrid_mean, sigma_only_j), random coordinate j, and four draws varying
only j with other actions fixed. It can keep five actor critic rows and avoids
cross-coordinate irrelevant-variance credit/hot-to-cold contamination in the
exact-reference limit. However, the current critic is not validated on those
conditional contexts; it needs explicit training/coverage and calibration.
Reference error alone can reintroduce expansion proportional to its square.
This is a future architectural hypothesis, not implemented or established best.

13014 now compares both references with relative, RMS-relative, signed-density,
RMS-density and physical clipped-policy gates, using exactly the saved four
probe draws. It reports sigma/advantage/normalizer quantiles, per-coordinate and
sigma-binned scale terms, paired mean coefficients, bounded-logstd slopes,
and identical-history cloned one-step Adam+clip+entropy output changes.
These are counterfactual current-checkpoint updates, not reconstructed historical
steps or population gradients. Pure distribution tests7passed, including flat
DG expansion and physical-law normalization/score identities. Training remains
unchanged; diagnosis is queued rather than falsely claimed complete.

Claude completed the consolidated investigation and its owned agent was closed.
Next proposed clean training arm, pending13014: projected nonnegative alpha
initialized at zero, preserving latent-V/full-DG/raw/critic settings. This is
explicitly a test of early entropy-forcing and hysteresis, not a repair for
current high sigma by adding entropy. From-scratch controller changes affect
early trajectories even if alpha initially1; the claim they are inert before
floor crossing was withdrawn. No such trainer/run was submitted in this audit.
Coordinate-conditional latent-V credit remains a larger candidate only after
checking the current checkpoint's restoring forces and saturation.

User reported current run poor. Stopped13010 modal/all around327k envsteps:
geometric sigma~.9165, near-ceiling fraction~.974, greedy~6782. Held13011
mean-onlycontrol before cancelling13010 so it would not start automatically.
13014 initially failed on sigma-only autograd because mean-head parameters were
unused; materialized zero gradients for unused parameters and retried via mlq.
Attempt2 passed. At736896checkpoint sigma quantiles(0,.1,.25,.5,.75,.9,1)
are(.00674,.00675,.0120,.241,.947,.99993,1), confirming extreme dispersion.
For latentreference, PG ceiling scale coefficient-1.030 vsrelativeDG-.0485;
floor PG-.0349 vsDG+.521. RMS relative gate makesceil+.0826, while density
makesceil-.913. All cloned one-step savedAdam candidate updates increase mean
sigma on thisbatch, includingPG, so coefficient signs cannot identify actual
optimizer-inducedoutputchanges. Paired mean coefficient RMS floor866PG/1034DG
versusceil5.17PG/5.84DG, supportingbutnotproving hot/cold credit-noise hypothesis.
Matched independent futureexperiment proposed: keeplatentV/fullDG/critic;
zero-startprojecteddualcontroller forboth relative andpaper signed-densitygate
arms, withsamegateamplitude and30m limits atpriority-1. This wouldcomparegate
geometry undercontrolledinitialexploration, notreplicatethepaper'sRetrace setup.
No newtrainer ortrainingrun submitted in this question/stop turn.

### Projected entropy controller and matched DG gates v11

The user authorized the matched pair. New trainer:
cleanrl/td7/delight/td7_latentq_delight_projected_v11.py.
Both runs retain learned latent V as actor reference, full-score DG, raw
advantages, four antithetic actions, SALE, clipping, replay, critic/encoder losses,
and the fixed sigma=.2 continuation. The factor2 gate multiplier is identical.
Relative rarity is min(.5*sum eps^2,10); density rarity adds sum log_sigma plus
D/2*log(2*pi) and clips symmetrically to[-10,10]. Gates and references are detached.
The density arm tests the paper's gate within TD7, not its entire Retrace setup.

Both arms use direct nonnegative entropy_alpha, initial0, alpha loss
alpha*(observed_logstd_entropy-target_logstd_entropy), Adam, then in-place
projection alpha>=0. This avoids exponential log-alpha gradient starvation and
log-space debt; Adam momentum remains at the boundary. It is one global
geometric-entropy constraint, not a coordinate floor or sigma ceiling.
This changes the early trajectory compared with v10 and is intentional in BOTH
arms. Actor architecture, critic loss, query definitions and training cadence
are pinned by AST tests. V11 checkpoints have a distinct format and alpha state.
Diagnostics retain gradient norms/probes, geometric sigma, endpoint fractions,
scale terms and target error, plus negative-surprisal fraction.

Independent mathematics review found no blocker and checked density identity,
detachment, PG limits, KKT signs and captured scalar storage. The critic reviewer
authored tests with CUDA-only actual network cases. Local full selected suite:
77passed,10CUDAcases skipped. Queued validation13037(priority-1,max1,20m)
will run all87 tests, then full1M-LAP compiled learner/checkpoint benchmarks
for both gates. Relative reward/critic equivalence to v10 fixes entropy0 ONLY
for the mathematical comparison; captured updates use the real controller.

Training13038 relative and13039 density are validation-success dependent,
sequential after-terminal, priority-1,max1,30m each, HalfCheetah-v4,seed1,8M
ceiling,16envs,1024steps,2envthreads. Queued commands/submissions are in jobs.json.
Both have greedy-evaluation autocull supervision. Old modal mean control13011
remains held, and bad modal/full13010 was cancelled. No performance claims
are made before these runs; CUDA validation is still pending shared-GPU lease.

Validation13037 succeeded:87 CUDA-inclusive tests passed. Relative reward loss /
actor-gradient equivalence (entropy fixedzero for mathonly), exactcriticvalues /
JVP /traininggradients, bothgradientprobes, compiledactualcontrollersteps and
serializedcheckpointrestore checks passed. Uniform-prefix priority fixture gives
exactnextcaptured update includinghardtargets; inheritednonuniformFP32CDFscan
association remainsreportedseparately. Thedensityprobe entropygradientnorm.04286
aftercompiledtraining shows thezero-initialmultiplierreactivated;relativeprobe
entropygradientnorm0 atitscheckpoint. Learner-onlythroughput~1203updates/s
relative/~1204density, matchingv10~1205; notend-to-end/aggregatebenchmark.
Training13038 started;13039 waitsforitsterminalstate. No performanceconclusions
yet. Theheld13011oldmeancontrolwillnotstart. No additionalcontrollermoment-reset
or sigma-floor/clamp adjustment was introduced.

User requested killing the projected relative run as better but stalled early.
Cancelledqueuedpartner13039 first, then running13038; bothconfirmedcancelled.
13038 saved a completed-rollout checkpoint at1195648 environmentsteps,
1163264 learnersteps, run1791874728. Densityarm13039 neverstarted, so this
iteration provides no relative-vs-density benchmark result. Noreplacementrun
submitted on this cancellation request.

### Binary Rao-Blackwellized DG transfer assessment (2026-10-14)

Inspected parameter-golf's critic_design_math_20261013 synthesis/results and
the v11 critic/actor paths; independent math and critic reviewers agree on the
distinction recorded in binary_rb_transfer_analysis.json. For binary outcomes
and no baseline, E[R sigmoid(R*l/eta)] = Q sigmoid(l/eta). Real-valued returns
require E[(G-b) sigmoid((G-b)*l/eta)], with our factor 2 if retained. Scalar Q
does not determine this coefficient: at l=2, constant G=-1 has coefficient
-0.119203, whereas equiprobable G=-3,+1 has the same mean Q=-1 but coefficient
+0.436690. These numbers were independently recomputed with scalar arithmetic.

Current mean-advantage DG is a different estimand from conditional sampled-return
DG, not automatically an implementation bug. A transfer would therefore be an
explicit algorithm change. A policy-consistent distributional return critic or
a coefficient critic with valid trajectory-return labels is needed; directly
gating a scalar Bellman mean retains the plug-in approximation. Predicting the
coefficient removes outcome sampling noise when accurate, but retains DG's
exploration pressure and does not establish a cure for excessive sigma.

The current critic already has nonlinear action-conditioned inputs and separate
parameters from the actor. Latent V integrates the present action distribution,
not the nonlinear utility of future outcomes. Fixed-sigma .2 continuation also
differs from the current actor policy. No trainer changes or new jobs submitted
for this analysis. Old job13011 is terminal (30-minute limit); current queue
inspection found no active or queued TD7 Delight jobs. No claim is made about
who released the previously held job.
