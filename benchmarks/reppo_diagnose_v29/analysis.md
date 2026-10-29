# Why latent-roll v27 slows near 12M

Analysis of `HalfCheetah-v4__reppo_latentroll_v27_h8_50M__1__1790378891`.
All three 16.1M training runs finished. Independent simulator audits finished
at 8M and 12M for the original H8 method. The user stopped the expensive oracle
during its 16M evaluation; remaining oracle and dependent analysis jobs were
cancelled or skipped. This report uses only completed evidence.

## Diagnosis

The leading explanation is an **action-value learning bottleneck with continuing
policy movement**, rather than an implementation failure that starts at 12M.
The actor relies exclusively on learned action gradients; the auxiliary model
does not directly make those gradients match counterfactual returns. Once the
easy gait improvements are exhausted, a weak action-ranking signal must compete
with entropy adjustment, KL rollback, and finite optimizer steps. That is a
plausible mechanism for slow net progress and intermittent regressions.

Confidence is moderate. The simulator confirms limited action-value
discrimination, but does not prove it caused the plateau. In particular, the
12M saved update improves actual returns. The stronger original suspicion of
uniformly harmful late gradients is **not supported**. No measured discontinuity
establishes a special failure at 12M.

Two suspected explanations fare worse: reward-prediction gradient dominance is
contradicted by the gradient probes and the reward-loss ablation; sustained
destructive value/latent gradient opposition is not observed. Latent learning
still consumes capacity and gradient magnitude, so these results do not clear
every possible auxiliary-loss effect.

## What actually happened

The requested run contains 15.854M recorded environment steps, despite `50M` in
its name. Its training-return means are 11,376 at 10–12M, 11,501 at 12–14M, and
11,514 at 14–15.854M. This is a strong slowdown with local regressions, not a
sustained collapse beginning at 12M. Sparse deterministic evaluations likewise
do not establish a sustained decline.

## Reproduction and controlled intervention

All experiments use HalfCheetah-v4, seed 1, CUDA, 16 environments, and 16.1M
requested training steps. All jobs were submitted through `mlq` with an explicit
parallel-run limit of 1. Snapshots are at 8.011392M, 12.009088M, and 16.006784M.
The final learner step is 16.088704M because training uses whole rollout batches.

| Arm | 6–8M training return | 10–12M | 14–16M |
|---|---:|---:|---:|
| v27 H8, reward prediction enabled | 10,441 | 11,376 | 11,517 |
| v27 H8, reward coefficient zero | 9,888 | 11,232 | 11,544 |
| v25 parent, lambda=.8 | 8,757 | 10,165 | 10,944 |

The v29 diagnostic trainer preserves the v27 learning computation. Every
overlapping learning scalar checked against the original run is bitwise equal;
the parent reproduction also matches its original run. See
`reproduction_parity.json`. The probes therefore describe the original learning
trajectory, rather than a merely similar rerun.

Removing reward prediction does not remove the late plateau. This is one seed,
but the direct intervention and gradient magnitudes both argue against reward
prediction being the principal bottleneck. It is not evidence that every
auxiliary loss is harmless: the latent loss remains in both H8 arms.

## Code-level mechanism

The actor minimizes `-Q(s,a) + temperature * log pi(a|s)` using reparameterized
actions. Its reward-seeking gradient comes entirely through the critic's action
input. It has no return-based policy-gradient correction. Outside the per-state
KL bound, its objective switches to KL rollback toward the rollout behavior
policy (`actor_loss`, frozen v27 lines 430–450).

The critic fits a categorical projection of soft lambda-return targets on
observed state–action pairs. The H8 auxiliary predicts future latent features
and rewards. The value head is trained on the initial latent only: the imagined
reward sequence does not directly constrain the value of an imagined action
sequence (`critic_loss`, lines 402–428). Good latent or reward prediction thus
does not establish accurate action-value differences or derivatives.

Bootstraps use the changing actor and a periodically refreshed critic. The
lambda branch's next-state entropy is explicitly included, while cut chains
bootstrap once. Inspection did not reveal double-counted entropy, crossing
episode boundaries in the latent chain, or an obvious captured-update defect.
Contract tests and exact reproduction support the implementation audit.

## What the training diagnostics establish

At 14–16M in the H8 reproduction, encoder gradient norms are approximately:

| Loss | Weighted encoder gradient norm |
|---|---:|
| Value cross-entropy | .894 |
| Latent prediction | .717 |
| Reward prediction | .0038 |

The value/latent gradient cosine averages about +.019. The latent objective
consumes substantial gradient magnitude, but the probes do not show persistent
opposition to the value objective. The reward gradient is too small to support
the earlier suspicion of reward-head dominance.

Feature RMS grows from about 3.29 at 2–4M to 5.90 at 14–16M. Raw latent MSE must
therefore be interpreted with scale. H8 MSE divided by target energy does not
show an abrupt failure at 12M, but this normalization can hide constant feature
offsets. Frozen-checkpoint diagnostics additionally measure centered variation.
That additional analysis was implemented but not executed before the user
stopped computation; it is not used as evidence in this report.

The logged held-out advantage correlation is not an independent truth test:
its return target bootstraps from a learned critic, and both advantages subtract
the same sampled baseline. Consecutive action-gradient cosine measures temporal
consistency, not correctness. Likewise, before/after next-rollout errors use
bootstrap labels and cannot establish improved true value accuracy.

## Independent simulator tests

The planned protocol for each checkpoint and arm uses 512 sampled
states, 32 continuation rollouts, two action-perturbation sizes (.2 and .05),
and 100/200/300-step horizons. These labels retain critic-bootstrap tails,
including at time limits; they are corroborating evidence, not exact Q values.
The implementation checks physics and rewards against Gymnasium.

For the completed reward-enabled H8 checkpoints at 8M and 12M, a second test
compares saved before/after policies for the last
actor update and for the whole update block. Common random numbers pair three
branches: old policy throughout, new first action then old policy, and new
policy throughout. It records raw, scaled soft, and undiscounted raw returns at
100/300/1000 steps. These returns have no learned value tail. The simulator
continues past artificial episode limits, matching the continuing bootstrap
objective; the undiscounted totals are not reset-based benchmark episodes.

The pre-update critic predicts the first-action soft improvement at a fixed
temperature. At 1000 steps the discount multiplier is about .000043. Reported
trajectory intervals resample whole environment rows; there are usually only
16 independent rows. They capture neither training-seed uncertainty nor every
possible state-distribution shift.

The legacy finite-difference metric named `cos_pooled_corrected` is mean
oracle gain along unit critic-gradient directions divided by RMS oracle
gradient magnitude. It is not ordinary pooled cosine: variation in gradient
norms can lower it even when individual directions align. At 8M, restricting
to states with no intermediate time-limit bootstrap gives H300 values .121 at
mean actions and .226 at sampled actions. The top 5% of states dominate the
noise-corrected gradient energy, so exact magnitudes are fragile. See
`oracle_8M_tail_sensitivity.json` for trajectory intervals and trimming checks.

A negative final actor update alone is not a critic failure: the KL gate may
intentionally trade predicted value for a smaller behavior-policy KL. Offline
gate analysis distinguishes these cases. The strongest critic-error evidence
would be positive predicted improvement but negative realized improvement,
with a paired actual-minus-predicted interval excluding zero. If local
first-action predictions agree with simulation while repeated policy changes
lose return, the likely mechanism instead concerns update size or induced state
distribution. Raw degradation with soft improvement indicates an objective
tradeoff. Agreement and positive late improvements would weaken the proposed
critic-fidelity bottleneck.

## Completed simulator results

The following values pool the sampled states. Brackets are 95% whole-trajectory
bootstrap intervals, with 16 environment trajectories—not independent seeds.
The finite-difference labels below retain the horizon/time-limit critic tails.

| Statistic at H300 | 8M | 12M |
|---|---:|---:|
| Within-state action-value correlation, noise corrected | .200 [.091, .313] | .154 [.060, .283] |
| Ordinary pooled gradient cosine, mean action, h=.2 | .182 [.086, .318] | .097 [.017, .471] |
| Ordinary pooled gradient cosine, sampled action, h=.2 | .260 [.151, .458] | .030 [-.054, .241] |

The point estimates are consistent with weaker useful action information later,
especially when the critic's gradient magnitudes determine the weighting.
Intervals overlap and gradient energies are heavy-tailed. Some mean-action
bootstrap denominators at 12M are nonpositive; its reported interval conditions
on valid draws (93.15%). This prevents a decisive claim of temporal degradation.
Restrictions excluding intermediate time-limit tails do not provide a clean,
high-precision rescue. The reports preserve those restrictions and sensitivity
checks rather than selecting a favorable subset.

The more independent tests measure actual policy changes without critic tails:

| Saved whole update, 1000-step discounted gain | 8M | 12M |
|---|---:|---:|
| Raw return | -5.08 [-6.40, -4.07] | +10.56 [5.63, 15.95] |
| Soft return, initial temperature | -.121 [-.246, -.024] | +.881 [.389, 1.416] |
| Soft return, final temperature | -.066 [-.190, .030] | +.870 [.379, 1.403] |

These are continuing-state discounted differences, not changes in the usual
1000-step reset-based benchmark score. The 8M update sacrifices raw reward and
increases entropy; its soft loss is not distinguishable from zero when evaluated
at the final temperature. The 12M update is beneficial under both temperatures.
Re-evaluating at the final temperature attenuates the apparent 8M soft-objective
loss; this does not show that temperature adaptation caused the policy change.
The later update contradicts a universal harmful-update explanation.

For the isolated final actor step, the critic predicts first-action soft gains
of .00122 at 8M and .000229 at 12M. Measured gains are -.0137 [-.1151, .0861]
and +.1603 [.0146, .3000], respectively. At 8M the measurement is underpowered
for such a small prediction. At 12M the critic understates a beneficial change;
that is a mismatch, not evidence of destructive ascent. The last step may also
contain intended KL rollback. Behavior-gate analysis was implemented and its
tests passed, but the model-based postprocessing was not run after cancellation;
no gate-stratified numerical claim is made here.

## Practical implication

If iterating on this architecture, prioritize correcting the actor's reliance
on critic-only action gradients with an observed-return policy-gradient signal,
or directly constraining action-value consistency through imagined rewards and
values. Merely improving latent prediction need not improve counterfactual
action ranking. These are motivated directions, not demonstrated improvements.

Do not treat this evidence as a reason to delete reward prediction, increase
the latent horizon, or lower a learning rate reflexively. Removing reward
prediction already failed to materially change the late return. There is no
evidence for support saturation, action saturation, or an entropy-controller
explosion as the principal cause. A fixed entropy target also does not impose
an 11.5k universal ceiling: the v25 parent eventually reaches roughly 13.2k
stochastic return at 48–50M under the same target, although that is a different
architecture and cannot predict v27's asymptote.

## Validation and stopping point

Completed training jobs: 10299 (parent), 10301 (H8), and 10302 (H8 no reward).
Completed test jobs: 10300 (six trainer/graph contracts), 10303 (five oracle
contracts including compiled CUDA/MuJoCo identity-policy validation), 10315
(gradient attribution), and 10325 (three behavior-gate/temperature contracts).
A subsequent pure oracle-statistics test also passed. All declared parallel
limits were 1. Full learner-scalar parity is recorded separately.

Oracle 10310 saved complete 8M/12M datasets before cancellation during 16M.
Control oracles 10309/10311 were cancelled before starting; dependent model
gradient and gate analyses were skipped. No job from this campaign remains
queued, held, or running. The autoculler was stopped. Frozen-checkpoint
centered-feature and whole-network gradient-share measurements remain
uncomputed; only the actual training-time encoder probes are used above.

## Reproducible artifacts

`manifest.json` records commands, job IDs, source hashes, design, and queue
outcomes. `source/` preserves the relevant implementation. `results.json`
contains numerical training windows and available simulator/gradient reports;
`results.md` provides generated tables; `diagnostics.png` plots the training
comparison. Snapshot files and simulator datasets remain under `runs/`.
