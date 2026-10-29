# 12 - Red team of the diagnosis (Part 1 of `11_synthesis_and_plan_DRAFT.md`)

Role: final adversarial reviewer. Read in full: 07, 08, 09, the draft; 03 / 06 only where a claim rests on them.
New computation is CPU only: tfevents dumps (`data/03_runs`), the oracle chunks (`data/03_chunk_*.npz`), and the
consecutive snapshot pairs of the reference trajectory. Simulation: 3 stochastic episodes on each of 3 existing
snapshots, twice (about 10 s in total). No training, no GPU, no edits to existing files.

Tags: **[NEW]** computed here; **[07] [08] [09]** taken from that report. One seed, one trajectory per recipe
throughout; every "run" below is a single trajectory.

| script | output | what |
|---|---|---|
| `scripts/12_longrun.py` | `data/12_tfevents_checks.txt` | stochastic / deterministic return, std, q_mse of the 50M REPPO runs at 4-50M |
| `scripts/12_return_accounting.py` | same file | crash share, tumble cost and ordinary noise cost per window from the training episode record |
| `scripts/12_sigma_jitter.py` | same file | per-iteration `policy/std` jitter and its relation to per-iteration return, by lag |
| `scripts/12_std_composition.py` | same file | logged std / entropy regressed on crashed episodes in the batch |
| `scripts/12_stage_shrinkage_rowsplit.py` | `data/12_stage_shrinkage_rowsplit.txt` | calibration slope per stage (shrinkage test); upright / inverted row split of each chunk |
| `scripts/12_rewrite_onrun.py` | `data/12_rewrite_onrun.txt` | one-iteration rewrite of the critic's gradient field on on-run states; which states the KL gate holds |
| (inline, same rollouts) | `data/12_outlier_states.txt` | what the top-1% gradient-energy states are |

---

## 0. Verdicts

| | Verdict | One line |
|---|---|---|
| D1 | **OVERCLAIMED** (headline stands, three sub-claims do not) | "The logged metric and the pooled oracle numbers misled 60 versions" stands. "Near-optimal shrinkage" holds at 1M and 8M-trimmed only; at 4M and 30M the critic is over-dispersed 2-3.5x. "1.2-1.8x PPO" is a point estimate at P 0.75-0.95 from two different recipes. The gradient field is spiky on ordinary upright states and its spike amplitudes change 3-6x per iteration. Confidence in "not broken, only information-limited": ~45%, not 80%. |
| D2 | early half **STANDS WITH EDITS**; late half **OVERCLAIMED**; one bullet **WRONG** | Tumbles cost 250-440 return directly at 4-8M (14-24% of the deterministic-minus-stochastic gap, about 0 of the slope change). The "+-3-4% sigma jitter = +-85 return = 10x progress" bullet is largely a batch-composition artifact. "Inverted rows are the outlier states" is not what on-run rollouts show. |
| D3 | **OVERCLAIMED** | "2.9-4.1x at n = 1" is conditional on deleting the 1.2% of states that hold most of the actor's gradient energy, and uses an information formula that does not apply to a noise-free biased label. Honest range 0-4x at n = 1, 0.4-1.3x at n = 5. |
| D4 | **STANDS WITH EDITS** | The memory dispute is resolved by the data without an experiment: direction persists, spike amplitude does not. "Mean KL pinned at the bound" is the in-update average; end-of-iteration KL was 0.014 / 0.028 / 0.042 on three iterations. |
| D5 | **STANDS** | Matches 06. Part 2's "4-6 min per 8M run" is the solo figure's lower edge (379 s solo; 672-755 s at three concurrent). |

---

## 1. D1 - the critic

### 1.1 Report 07 D head-to-head: not circular, but much weaker than the draft's sentence

- **Circularity: no.** The score-function reference with the critic's own E_pi Q as baseline is unbiased for any
  state baseline, every product is cross-lane, and the snapshot critic had not been fit on the chunk. The shared
  bootstrap bias exists only in the lam_ref 0.95 column.
- **Width.** Under the clean reference (lam_ref 0.99) [07 table]:
  - 4M, full parameters: critic 0.59 [0.29, 0.87], SF 0.43 [-0.11, 0.79]; P(critic > SF) 0.75-0.87. Ratio 1.2-1.4
    with an interval that includes < 1 and > 3.
  - 8M: critic 1.17 [0.04, 2.31], SF 0.67 [-0.37, 1.67]; P 0.92-0.95; and the truth itself is unresolved there
    (P(|V*|^2 > 0) 0.45-0.54). The P = 1.00 the draft's range leans on is the lam_ref 0.95 column, the one with
    the shared bias.
  - 12M: neither resolved. 30M: critic 0.2-0.75x the SF, P 0.31-0.55.
- **Recipe mixing.** 4M / 12M / 30M are v23 (lambda 0.95, no in-iteration refresh, no H8); 8M / 12M are v29-H8.
  The draft's "1.2-1.8x at 4M-8M" joins one snapshot from each recipe; "leans score function at 30M" is the old
  recipe only. Neither is the v60 reference.
- **What the comparison is not.** One rollout, unit Euclidean step, mean only, no Adam / epochs / clipping, a
  critic baseline instead of a PPO value net. PPO's noise averages across iterations; the critic's error is
  correlated (07's own caveat 4).

**Edit.** Replace "yields 1.2-1.8x the true gain per unit step of PPO's estimator at 4M-8M, and is never
significantly worse" by: "per rollout it is statistically indistinguishable from a GAE(0.95) score-function
estimate at 4M (v23) and 8M (v29); point estimates favour the critic (ratio 1.2-1.8, P 0.75-0.95), nothing is
resolved at 12M, and the point estimate favours the score function at 30M (v23)". Confidence that the critic's
gradient is not worse than PPO's per rollout at 4-8M: ~75%. Confidence that it is better: ~55%.

### 1.2 07's "95% unrelated to truth" vs 08's "near-optimal shrinkage"

At 1M these are the **same numbers**. Without state 402: r^2 = 0.05, sd ratio 0.28, so
slope(critic on truth) = r x 0.28 = 0.063 and slope(truth on critic) = r / 0.28 = 0.79. Any posterior mean with
r = 0.22 has 95% of its own variance orthogonal to the truth; "95% unrelated" is not evidence against shrinkage,
and "calibrated" is not evidence of quality. The discriminating number is the calibration slope (equivalently
amplitude / correlation), and it has to be measured per stage. 08 did it at 1M only and flagged that the logged
`probe/slope_beta` falls to 0.42 / 0.37 at 4M / 8M; the draft dropped that qualifier.

**[NEW] Per stage** (`12_stage_shrinkage_rowsplit.py`; within-state advantages over the 8 fresh actions, truth =
two independent 16-rollout halves at H = 100, bootstrap over states):

| snapshot | per-state corr, all / top 5% dropped | amplitude / corr, all / trimmed | slope truth-on-critic, all | same, top 5% dropped | P(slope < 1) |
|---|---|---|---|---|---:|
| v23 4M | 0.29 / 0.31 | 3.5 / 2.0 | 0.28 [0.16, 0.53] | 0.50 [0.38, 0.61] | 1.00 |
| v29-H8 8M | 0.20 / 0.26 | 2.2 / 1.1 | 0.46 [0.17, 0.86] | 0.88 [0.65, 1.11] | 0.99 / 0.86 |
| v23 12M | 0.04 / 0.06 | 5.9 / 2.7 | 0.17 [-0.30, 0.78] | 0.37 [0.16, 0.59] | 0.99 |
| v29-H8 12M | 0.15 / 0.14 | 1.3 / 1.3 | 0.74 [0.24, 1.82] | 0.79 [0.21, 1.62] | 0.72 |
| v23 30M | 0.22 / 0.29 | 3.2 / 1.5 | 0.32 [0.19, 0.50] | 0.68 [0.47, 0.89] | 1.00 |

- Optimal shrinkage (slope 1) is compatible with v29-H8 at 8M trimmed and 12M. It is rejected for v23 at 4M and
  30M even trimmed, and everywhere untrimmed except v29 12M. The actor consumes the untrimmed field (07 A2's own
  standard), where the critic's contrasts are **2-3.5x larger than their accuracy supports**.
- v23 at 12M has essentially no per-state information (0.04-0.06). That is a late critic failure in the old
  recipe, visible in its q_mse too (4-6 against 0.7-2.0 for the lambda-0.8 lineage; 1.4 below).
- In all five chunks the top 5 of 512 states hold 46-56% of the critic's within-state energy, and the 4M and 8M
  chunks contain **zero inverted rows**. The energy concentration is not a tumble phenomenon.

### 1.3 Rewrite per iteration, on on-distribution states

07 B used 4096 states of a different run. **[NEW]** `12_rewrite_onrun.py`: 3000 states from three stochastic
episodes of the first snapshot of each consecutive pair of the reference trajectory.

| pair | top 1% / 5% of states: share of gradient energy | per-state cosine k vs k+1, median (p10) | same, top-decile states | rms change / rms field, all (upright only) | top-1% states: rms ratio k+1 / k |
|---|---|---|---|---|---:|
| 1.02M -> 1.03M | 0.88 / 0.92 | 0.88 (0.33) | 0.98-0.99 | 1.35 (0.56) | 1.50 |
| 4.01M -> 4.03M | 0.69 / 0.84 | 0.93 (0.58) | 0.98-0.99 | 1.28 (0.68) | 0.29 |
| 7.91M -> 7.93M | 0.88 / 0.94 | 0.92 (0.48) | 0.98-0.99 | 0.88 (0.88) | 0.16 |

Reading:

- **Both reports are right about different moments of one object.** State by state the *direction* of dQ/da is
  highly persistent (median cosine 0.88-0.93, and more so where the gradient is large), which is 08's memory.
  The *energy-weighted* field changes by 56-135% of its own rms in one iteration, which is 07's rewrite, and
  07's off-run 59-113% reproduces on on-run states. The reconciliation is amplitude: the top 1% of states
  carry 69-88% of the energy and their amplitude moves by 1.5x, 0.29x, 0.16x between consecutive critics. At 4M
  the bottom half of states by |g| has an rms ratio of 8: individual states jump in and out of the spike set.
- **The spikes are ordinary gait states** (`data/12_outlier_states.txt`). At 7.9M all 3000 rows are upright and the
  top 30 states are scattered through the three episodes at median |pitch| 0.19 (all rows 0.19), Q 103 (107),
  sigma 0.44 (0.49). At 4M, with one crashed episode in the sample, inverted rows are 27% of rows and 30% of the
  energy, i.e. no more than their share; 28 of the top 30 states are upright.
- So the draft's D1 bullet "pooled sums in which one tumbled state holds 86-88%" is true of the 1M oracle and
  wrong as a general account. Extreme concentration of the action-gradient energy in about 1% of ordinary
  states, with direction-stable but amplitude-unstable spikes, is a property of this critic at every stage
  examined. Combined with slope 0.28-0.5 untrimmed, the simplest reading is that the spikes are mostly fitted
  noise in amplitude. Whether the true field is also spiky there (contact events; 03 reports the true gradient
  rms falls from 1.70 to 1.01 between h 0.1 and 0.3) is unmeasured.

### 1.4 "Information-limited": is there evidence a critic can do better on the same data?

Yes, and the draft's sentence "loss / architecture / target changes could not help, and none did" is too strong:

- Critic passes 4 -> 16: +2.0k at 1M and 8M [08, C+]. Some aux loss vs none: +1.3k [08, B-]. Decoupled
  minibatches: +4.1k [08, B]. Same data each time.
- **[NEW] Target recipe moves the late plateau** (`12_longrun.py`):

  | run | stochastic @8 / 16 / 30 / 50M | deterministic @8 / 16 / 30 / 50M | q_mse @30-50M |
  |---|---|---|---|
  | v23 (lambda 0.95, no refresh) | 10.6 / 11.3 / 11.4 / 11.2k | 10.9 / 12.4 / 12.7 / 13.0k | 5.6-6.4 |
  | v25 refresh, lambda 0.95 | 9.0 / 11.4 / 11.8 / 11.7k | 10.0 / 12.8 / 13.1 / 13.0k | 4.8-6.6 |
  | v25 refresh, lambda 0.8 | 9.4 / 11.0 / 12.5 / 13.3k | 10.8 / 12.1 / 14.5 / **15.0k** | 1.6-2.0 |

  Same data budget, a different target horizon: +1.6k stochastic and +2.0k deterministic at 50M, and a critic
  error three times smaller. Single trajectories (floor ~1.2k), and the lambda-0.8 run is the one that sat
  inverted until 1.8M, but it is the only modern-recipe run past 16M.
- More data per unit of policy change raises the (flawed) logged metric monotonically [03 Q2: KL 0.03 -> 0.01,
  rollout 1024 -> 4096]. That is what "information-limited" predicts, and it also says the critic is nowhere near
  a ceiling that is independent of the data regime.
- 07 B's own conclusion is "not provably at the pooled ceiling and not provably below it" (worth 1 to ~20
  rollouts of score-function data in the policy's tangent space). The draft reports the per-state ceiling only.

### 1.5 Does the late trajectory point at a critic failure D1 underweights?

The premise needs correcting first [NEW, `all_hc_runs.tsv`, `12_longrun.py`]:

- At 8M-16M REPPO is **ahead** of the PPO-lineage 50M runs (those read 8.0-9.4k at 8M and 9.2-10.7k at 16M;
  REPPO 9.3-11.0k and 11.0-11.7k).
- "REPPO plateaus at ~11k while PPO climbs to 13.6-14.7k" is true of v23 and v25-lambda-0.95, the lambda-0.95
  recipes. The lambda-0.8 run reaches 13.3k stochastic and 15.0k deterministic at 50M. No v27+ reference-recipe run
  exists past 16M. So the 50M gap to PPO is **not established for the current recipe**.
- Where it is established (v23), it does point at the critic: per-state correlation 0.04-0.06 at 12M, q_mse 4-6,
  deterministic return +1.0k in 38M steps, and 07's 30M head-to-head leaning to the score function.
- Within the 8M benchmark window the thing to explain is the **deterministic** slope of the reference run:
  +1.2k per M over 3-5M, +0.84k over 5-6M, +0.40k over 6-7M, about +0.3k per M at 7-7.5M (`12_return_accounting.py`).
  That is a slowdown of the mean policy, so it cannot be explained by episodes that tumble under noise except
  through the data they contaminate (section 2.2).

**D1 edits.** Keep bullets 1-2 (metric artifact, pooled oracle numbers), with "one tumbled state" restricted to
the 1M oracle. Replace bullets 3-5 by:

- "Per sample the lambda-return carries 5-7% action signal; the critic's per-state correlation (0.15-0.3; 0.04 for
  v23 at 12M) is not below a per-state pooled ceiling. Whether it is at the ceiling in the policy's tangent
  space is open (07 B)."
- "Calibration is stage- and recipe-dependent: near slope 1 at 1M and at 8M (v29, trimmed); over-dispersed
  2-3.5x at 4M and 30M (v23) and everywhere untrimmed."
- "The gradient field's energy sits in ~1% of ordinary upright states whose amplitude changes 3-6x per
  iteration."
- "Per rollout the critic's gradient is statistically indistinguishable from PPO's estimator at 4-8M."
- "Target recipe and optimisation schedule did move outcomes by 1.3-4k; HL-Gauss resolution and width did not."

Is D1 an overcorrection of the user's experience? Partly. "The critic cannot rank" was a measurement artifact.
"The critic struggles" is supported: spiky, amplitude-unstable, over-dispersed late, and it failed outright in
the lambda-0.95 recipe by 12M.

---

## 2. D2 - tumbling

### 2.1 Early flip: Beta vs tanh

Stands as "the only feature with zero events on one side", with these edits [NEW, hyperparameters and curves
of the 12 bounded-head runs]:

- The bounded-head runs do not differ from the tanh reference only at initialisation. Their action sd is
  0.46-0.49 at 50k, 0.29-0.38 at 0.3M and 0.10-0.23 at 1M; the tanh reference's pre-tanh sd is 1.04, 0.74, 0.33
  at the same points. They are quieter through the **whole** 0.1-1M window in which the branch is taken. Nine
  of the 12 have no entropy-target dual, and seven (v18-v22) use lambda 0.95, the horizon to which 09's own
  census gives a trapped rate of 0.12 against 0.28 at lambda 0.8. The bounded-head runs on the modern lambda-0.8
  recipe are v47 x3, v48 and v57: five trajectories, none with the dual. The census cannot separate "initial
  noise" from "noise over the first 0.5M" from "bounded head" from "no dual" from "lambda 0.95".
- They are faster early (5.1-6.4k at 1M against 4.3k) and **every one ends lower** (7.0-9.4k at 8M against
  10.8-11.0k). v18 with the dual: 5.4k / 6.8k / 8.7k at 1 / 4 / 8M. No bounded-head REPPO run has matched the
  tanh reference at 8M.
- `stepgeom_v48_box` (Beta, action sd held at 0.45-0.48 for 1M steps, 204 return at 1M) is counted as "upright";
  it is a run that barely learned, not evidence about flipping.
- Trapped 27/123 vs 0/12 is p = 0.12. The p = 0.003 contrast includes the "mixed" class, which 09 calls its least
  certain. PPO's 0/111 are neither independent nor the same algorithm.
- "Even upright runs spend 8-17% of their first 1M inverted" is one trajectory (path B). The other upright runs
  with snapshots read 28-80% of the first 0.5M [09 1.2].
- The decisive one-second check (random actions under tanh(N(0, 1.08^2)) vs Beta(1.69, 1.69), count inversions)
  has still not been run. I did not run it either: it is not on an existing snapshot.

Confidence that the early inversion is noise-triggered and that lowering early action noise cuts the trapped
rate: ~65%. Confidence that it can be done without the 1.6-4k late cost every quiet-start run has paid: ~35%.

### 2.2 Late somersaults: cause or symptom? Return accounting

**[NEW]** From the training episode record (1000 episodes per 1M window; crash = episode < 0.85 x window median):

| reference run (v60 base) | 3M | 4M | 5M | 6M | 7M | 7.5M |
|---|---:|---:|---:|---:|---:|---:|
| crash share | 4.6% | 4.6% | 6.0% | 7.3% | 6.4% | 5.6% |
| tumble cost = mean(non-crashed) - mean(all) | 178 | 243 | 344 | 435 | 338 | 248 |
| deterministic - stochastic mean | 1179 | 1021 | 1438 | 1839 | 1807 | 1744 |
| ordinary noise cost (deterministic - non-crashed mean) | 1001 | 778 | 1093 | 1404 | 1468 | 1497 |

- **12.5% at 7.9M is the upper edge of a 48-episode sample** (6/48, interval 6-25%). The training record, 1000
  episodes per window, gives 5.6-7.3% over 5-8M. 09's "roughly 800 of the 1840 gap is crashed episodes" becomes
  250-440 of 1740-1840, i.e. **14-24%**. The other 76-86% is the ordinary cost of sigma 0.5 on a fast gait.
- **Slope.** Between 4M and 7.5M the deterministic return rose 2.6k and the stochastic mean 1.9k. The tumble cost
  changed by about 0 (+5; +190 at its 6M peak). The ordinary noise cost grew by 720. So of the 0.7k by which the
  stochastic curve lagged the deterministic one, tumbles explain ~0-25% and ordinary noise cost the rest; of the
  deterministic slowdown (1.2k per M -> 0.3k per M) they explain nothing directly.
- **Upper bound on the direct benefit of removing every tumble at 8M: +0.25-0.45k stochastic.** The draft's E1.2
  prediction "+1-2k stochastic at 8M and the deterministic plateau moves" requires the indirect path.
- **The indirect path (crashed rows corrupt the critic) is weaker than 09 argued.** [NEW, row split]
  - The 4M and 8M oracle chunks contain no inverted rows and already have q_mse 1.41 and 1.44, the level 09
    attributes to crashed rows; the top 1% of rows hold 29% and 62% of it.
  - Where inverted rows exist they hold 38% (v23 12M, 4.4% of rows), 15% (v23 30M, 3.0%) and **0.7%** (v29-H8 12M,
    10.7% of rows; the critic fits them: 10.9 against a target of 10.7) of q_mse. Upright rows alone are at 10.2,
    16.3 and 2.95. 09's prediction "more than 70% of q_mse from fewer than 5% of rows, upright rows at their
    pre-3M values" fails in all three chunks. The error is concentrated, but in upright-tagged rows.
  - Time order in the one run that keeps climbing (v25 lambda 0.8): crash share 2-3% over 12-20M while the
    deterministic return creeps (+0.9k in 8M steps), then 8.6-11% over 30-40M while it jumps to 14.5-15.0k.
    Crash rate follows speed; a high crash rate did not stop the fastest progress in the archive.
- So: late tumbling is mostly a **symptom of speed at fixed noise**, with a direct cost of ~0.3k at 8M (rising to
  1.0k at 50M in v23, 15% crash share). 09's own 50% for "the plateau is the speed-versus-tumble frontier" should
  be ~30%. The larger and better-evidenced late term is the ordinary noise cost: 1.4-1.5k at 8M (12%), which is
  what v42 and the 8M entropy forks recover (+0.8-1.4k) without changing the crash rate.

### 2.3 The sigma-jitter bullet

**[NEW]** `12_sigma_jitter.py`, `12_std_composition.py`. Note that `noisefork_v60_base`, `regretprobe_v55_dual` and
`landscape_ref_v55dual` are the same bit-identical trajectory; 08's "two runs" are one trajectory plus v27-H8.

- The logged series does move as 08 says: after 3.5M the per-iteration difference of `policy/std` has sd 0.019
  (3.9% of 0.49; median |diff| 2.2%), lag-1 autocorrelation of differences -0.31 to -0.33, in both trajectories.
- **But `policy/std` is the batch mean over rollout rows, and the policy's sigma on inverted rows is twice the
  gait's**: 1.08 vs 0.47 (v23 12M), 0.96 vs 0.50 (v29 12M), 0.86 vs 0.50 (v23 30M). One crashed episode is ~500
  inverted rows = 3% of the batch = +0.015 in the batch mean. Regressing the detrended logged std on the number of
  crashed episodes finishing in the same and the next rollout gives +0.009 and +0.005 per crashed episode
  (sum 0.014, the predicted size), R^2 0.33-0.38 with that crude proxy; entropy +0.11-0.13 and +0.06-0.07 nats
  per crashed episode, R^2 0.19-0.26. The contemporaneous correlation of the logged std with the mean episode
  return of the same rollout is -0.61 to -0.63.
- **No forward effect.** The logged std at the end of iteration k has correlation -0.04 to -0.07 with the mean
  return of episodes finishing wholly after it, and -0.03 to -0.05 with their crash count. On the crash-robust
  median return the forward slope is -1.3k to -2.1k per unit sigma (+-440, +-260), i.e. 21-34 return per one sd of
  the logged jitter, against the 5.6k per unit sigma (85 return) that sigma^2 scaling predicts.
- "+-85 return per iteration" is therefore an inference (sigma^2 arithmetic) checked against two instrument
  readings that are each about 2 SE from zero, not an observed swing. The observed per-iteration sd of the
  median training return is 118-121 from 16 episodes, of which the part associated with the logged std is 36-49
  contemporaneously.
- **Edit.** "The dual jitters sigma by +-3-4% per iteration, ~+-85 return, ~10x net progress" should read: "the
  logged batch-mean sigma and entropy move by +-3-4% and +-0.3 nats per iteration; at least a third of that is the
  number of crashed episodes in the batch (the policy's sigma is ~1.0 on inverted rows); the gait-state sigma
  jitter is at most +-2.6% and unmeasured; its return effect is an upper bound of ~20-50 per iteration".
  Confidence that the dual's motion is a first-order cause of the late slowdown: ~20%.
- A real mechanism survives in a different form and is listed in section 6: the dual's entropy estimate is
  contaminated by crashed rows, so the temperature responds to the crash count.

### 2.4 "Inverted rows ... are the oracle's outlier states"

**WRONG as a general statement** (1.3): on on-run rollouts the gradient-energy outliers are ordinary upright
states, and inverted rows hold energy in proportion to their count. State 402 at 1M is one such case.

### 2.5 D2 edits

- Title: "the seed-averaged score is governed by the early flip; late tumbling is a symptom with a ~0.3k direct
  cost at 8M". Confidence in the early half ~65%, in "late tumbling limits the plateau" ~30%.
- Late bullet: 0% / 4% / 12.5% -> "0% / 4% [1, 14] / 12.5% [6, 25] in 48-episode rollouts; 4.6% / 5.6-7.3% in the
  training record at 4M / 5-8M".
- Replace the sigma-jitter bullet as in 2.3. Delete "are the oracle's outlier states".
- Last bullet: "noise cost 11-12%" is right (1.4-1.5k at 7-7.5M) and is the dominant late term; say so.

---

## 3. D3 - the transition channel

Numbers reproduce from 07. The framing does not survive 07's own standard.

- **Which states were dropped.** Six of 512 is 1.2%. In every dataset examined, the top ~1% of states hold
  46-88% of the critic's gradient energy and they are mostly ordinary states (1.2, 1.3). 07 A2 insists the field
  be scored "the way it is consumed", untrimmed. On that standard the n = 1 label is 0.05 [-0.18, 0.52] and the
  single-label n = 5 correlation is 0.36, not 0.54. At state 402, V'(s_1) is anti-aligned at -0.8 to -0.9.
- **The formula.** Information per step = (K - 1) / (K n) x c^2 / (1 - c^2) treats 1 - c^2 as noise that averages
  out. At n = 1 the label is deterministic: its residual is the bias of V' and more labels converge to rho = 0.45,
  not to 1. The n = 1 "information" does not accumulate; it is a one-off lift of the Q readout from 0.19 toward
  at most 0.45 per-state correlation on the non-outlier states. The critic already has a median per-state cosine
  of +0.61 with its own one-step lookahead, so part of that lift is already in it.
- **Range.**

  | | n = 1, K = 8 | n = 5, K = 8 |
  |---|---|---|
  | six states dropped, 07's CI on c and both reference values | 2.0-8.3x (central 2.9-4.1x) | 0.95-1.3x |
  | all states | 0.03x (interval 0-5x) | 0.35-0.7x |
  | as an accumulating quantity | not defined (ceiling 0.45) | valid |

- 1M only, v60 only. The critic's calibration degrades after 1M (1.2), so V' at perturbed states at 4-8M is more
  likely worse than better.
- "A branch label correlates with truth 0.90 at n = 5" is the expected label. One real branch is 0.54 trimmed,
  0.36 untrimmed; the n = 5 reward-vs-V' split is unknown.

**Edit.** "Information per env step vs an ordinary sample: 0-4x at n = 1 (2.9-4.1x only if the ~1% highest-energy
states are excluded, where V' is unreliable or anti-aligned, and only as a one-off lift to a 0.45 ceiling);
0.4-1.3x at n = 5; n = 2-4 unmeasured." Confidence that a vine at n = 2-3 gives >= 2x information per env step as
the actor consumes it at 4M: ~30%, not 65%. The oracle completion in E2.3(i) is the right gate; it must be
scored untrimmed and energy-weighted, with the reward / V' split.

---

## 4. D4, D5

- **Memory dispute.** Resolved by 1.3 without a run: direction memory is long (per-state cosine 0.9 per
  iteration), amplitude memory at the high-energy states is about one iteration. An EMA critic for the actor
  would mainly damp spike amplitude, so a positive E2.1 would not distinguish "memory helps" from "tail damping
  helps". Add an arm that clips per-state |dQ/da| at its p99 without any EMA.
- **Step size.** "Mean KL is pinned at the bound with 32-41% gated" is the average over the 32 actor steps on
  training rows. End-of-iteration KL on fresh on-run states [NEW]: 0.028 with 35% outside (1M), 0.014 with 4.5%
  (4M), 0.042 with 46% outside and 19.5% beyond twice the bound (7.9M); 08 has 0.026 and 0.043 at 8M and 12M.
  The gate is loose by up to 40% and the realised step varies 3x between iterations.
- **H8**: unresolved, as stated.
- **D5** matches 06. Add 06's own line that solo wall-clock to 8.4k is 1.7x *slower* than PPO despite the sample
  efficiency.

---

## 5. Retired beliefs

| Belief | Verdict on retiring it |
|---|---|
| "1/40 action gap is the cause" (v14 scalar head neutral) | **Mislabelled.** v14 (9.7k at 8M on an early recipe without H8) retires "HL-Gauss bin resolution is the bottleneck". The small within-state signal share (5-7% per sample) is D1's own mechanism; it cannot be both retired and the diagnosis. The heads that attacked the gap (v26, v38, v51, v63) failed for other reasons; v51 was +1.3k at 2M. Keep "action gap = low signal share" as live. |
| "self-inconsistent critic" | Retire per state (median +0.61). Keep the energy-weighted fact: at the highest-energy state it was V'(s') that was wrong, which is what a vine at n = 1 and every bootstrap target lean on. |
| "target variance is not the bottleneck" | Correctly reopened (v61 / v62 were no-ops). |
| "replay does not help" | Fine. |
| "flip is irreversible and decided by 0.5M" | Fine (10 of 37 recovered). |
| "H8 = 2.3x at 1M" | Fine. |
| "collinearity leak" | **Too hasty: demote, do not retire.** The only test is a global linear association at 4M and later. No snapshot before 1M, where the head start and the flip happen, and no nonlinear form. 04's placebo-action fit is cheap and still unrun. |
| (implicit) "the plateau is at ~11k and PPO passes it by 50M" | Should be added to the retired list for the lambda-0.8 recipe (1.5). |

---

## 6. Missing hypotheses, ranked

| # | Hypothesis | Evidence here | Cheapest check |
|---|---|---|---|
| 1 | **Spiky, amplitude-unstable action-gradient field on ordinary states.** ~1% of upright states carry 69-88% of the pathwise gradient energy; spike amplitude moves 3-6x per iteration; untrimmed calibration slope 0.28-0.46. The pre-clip actor gradient norm is 2.5-4 against a 0.5 clip, so every other state's signal is scaled down 5-8x by a handful of rows. | 1.2, 1.3, `data/12_outlier_states.txt` | Offline on existing snapshots: cosine between the actor parameter gradient with and without the top 1% of states (machinery in `audit_actor_gradient_tails.py`), and the oracle gain of each. Then one fork arm from 4M with per-state |sigma dQ/du| clipped at p99. |
| 2 | **Observation normalisation of root pitch is set by tumbles, per lane.** Cumulative per-lane pitch sd is 0.9-1.1 at 1M (early inversion), 0.66-0.84 at 4M and 0.88-1.51 at 7.9M (somersaults wind pitch to 16 rad), against a gait pitch sd of ~0.15-0.25. The one feature that predicts a tumble reaches the networks compressed 4-7x and scaled up to 1.7x differently across lanes (all-dimension scale ratio 0.90-1.55 at 7.9M). The Beta / PPO runs were never inverted and do not have this. | [NEW] snapshot statistics | Free: compare `obs_variances[:, 1]` across existing tanh and Beta snapshots. Then a from-scratch arm with pitch wrapped to [-pi, pi] (or winsorised statistics); it interacts with both tumble events. |
| 3 | **The benchmark-window slowdown is in the deterministic policy, and the late-plateau premise is recipe-specific.** | 1.5 | Free: restate the target as the deterministic slope 4-8M. One continuation of the reference recipe to 30M if the 50M horizon matters. |
| 4 | **The entropy dual and the logged sigma are contaminated by crashed rows.** Sigma is ~1.0 on inverted rows; each crashed episode adds +0.17-0.20 nats to the batch entropy the dual sees, so gait-state entropy sits below target and the temperature tracks the crash count. | 2.3 | Log std and entropy on upright rows only (Stage 0), then compute the dual on upright rows only in one fork arm. Replaces E1.2 (a) as the first dual experiment. |
| 5 | **Where the KL budget goes.** Per state, the cosine between the actual mean move and that state's own pathwise direction is +0.14 to +0.28 (median; +0.26 to +0.46 against the post-update critic). The states outside the gate are not the high-gradient ones (Spearman of KL with |sigma g| +0.06 to +0.27; outside-fraction 0.43-0.51 across all gradient quintiles at 7.9M). Most of each state's move is generalisation from other states plus the co-adapting critic. | `data/12_rewrite_onrun.txt` | Stage 0 logging of the same two numbers per iteration; no run needed. If confirmed, the per-state gate is discarding an arbitrary 35-46% of rows late, and a smaller network step is a different thing from a smaller KL bound. |
| 6 | **In-iteration co-adaptation (16 passes over 16k rows).** 08: the post-update critic's predicted gain is 3.4x the realised gain at 12M; v51 in-sample contrast correlation 0.25 vs 0.08 fresh. The per-state move aligns better with the post-update critic than the pre-update one (row 5). | [08], row 5 | 08's I3 (in-sample minus held-out realised gain), two lanes held out. Already half in Stage 0; make the held-out lanes explicit. |
| 7 | **Bootstrap V' at action-perturbed and off-gait states.** Anti-aligned at the highest-energy 1M state; governs both the lambda-return's n = 1-2 terms and the vine. | [07 A1] | Already in E2.3(i) if the reward / V' split is recorded for the top-energy states separately. |
| 8 | **Transfer to Hopper-v4 and Walker2d-v4.** No REPPO run exists on either. There a fall terminates: the flip trap (the largest term in D2) is absent, the absorbing-state corruption is absent, and noise-induced falls end the episode, so sigma 0.4-0.5 held by a -0.5 nat/dim target will cost far more than 12%. Pitch statistics (row 2) are bounded by termination. D2's fixes are HalfCheetah-specific; D1 and the noise cost are what should transfer. | none | One reference-recipe run on each task before Stage 1, with the entropy target as the only swept knob. It decides whether Stage 1 is worth doing first. |

Not supported by what I could check: per-lane normalisation mean shifts (rms 0.012-0.017 sd, negligible apart
from the scale effect in row 2).

---

## 7. Limits of this review

- Every stage comparison mixes recipes (v23, v29-H8, v55 / v60); there is one chunk per stage.
- The on-run rewrite and gate numbers use 3000 states from three episodes of lane 0's normalisation, one sampled
  action per state, three snapshot pairs. The 4M sample contains a crashed episode.
- "Crash" in the return accounting is a return rule; somersaults that land on the feet are not counted, so the
  tumble cost is slightly understated and the ordinary noise cost overstated by the same amount.
- The composition regression uses crashed-episode counts as a proxy for inverted-row counts; R^2 0.33-0.38 is a
  lower bound on the composition share.
- I did not open the PPO-lineage trainers, and did not verify 06 beyond matching the draft's numbers to it.
