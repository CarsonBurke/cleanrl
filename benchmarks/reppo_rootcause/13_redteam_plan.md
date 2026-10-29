# 13 - Red team of the experimental plan (11, Part 2)

Scope: the plan in `11_synthesis_and_plan_DRAFT.md` Part 2, read against reports 01, 06-10, the v60 trainer
(`cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py`, cited as `v60:<line>`), the repo `CLAUDE.md`
and the machine-learning skill. Read-only review. No training or simulation was run.

Tags: [CODE] read from source; [CALC] computed here (scipy / numpy, exact binomial, noncentral t);
[DOC n] taken from report n and not re-verified; [INF] my inference.

One thing I tried and could not do: the "one-second check" of 09 section 1.5 (do untrained tanh-Gaussian
actions tip the cheetah where Beta / a quieter head do not). I wrote it as
`scripts/13_random_posture.py` and submitted it as a CPU job through `mlq`; the submission was refused by
the permission system (the queue is busy with other owners' jobs). The script is therefore **unexecuted and
untested** (it only passes `py_compile`). It is specified below as E0.0.

---

## 0. Verdict

The plan is directionally right (instrument first, flip before critic, vine last) but, as written, it would
repeat the family's standard failure: too many arms at a replicate count that cannot resolve the predicted
effects, two predictions the archive has already contradicted, one gate that is self-contradictory, and no
upper-bound experiments to tell a dead direction from an under-built one. Thirteen required changes, in
priority order:

| # | Level | Change | Why (one line) |
|---|---|---|---|
| M1 | MUST | Stage 0 parity is not achievable as specified; restate it | "diagnostics every 8th iteration" removes default-CUDA-generator draws that v60 makes every iteration; shared modules changed after the reference run |
| M2 | MUST | Fix the posture instrument: `cos(obs[1]) < 0`, not `abs(obs[1]) > pi/2` | pitch is unwrapped; 3 of 8 late somersaults land upright after a full revolution |
| M3 | MUST | Realised-gain instrument: backward form is the primary; "held-out trapezoid" does not exist without withholding lanes | forward half is in-sample and memorisation-biased |
| M4 | MUST | Replicates: n = 8 per fork arm, blocked on 4 parent checkpoints; state power, not "resolves" | n = 4 has 40% power at 0.6k; the sd it rests on is n = 3 |
| M5 | MUST | Flip census: continuous primary endpoint; binary claim needs 32 per arm; do not cull the control | 16 v 16 Fisher power is 0.27 for 0.22 -> 0 |
| M6 | MUST | E1.2: drop the prediction "tumbles 12% -> < 3%"; add the truncation-on-inversion diagnostic fork | every late low-noise arm in the archive kept a 3-8% crash share |
| M7 | MUST | E1.2a: `--dual-step-per-iteration` moves the KL multiplier too; split the parameter group | [CODE] one optimizer group holds both duals |
| M8 | MUST | E1.1: specify the init (bias-only does not reach Beta's tail); add the zero-code arms 09 ranked; add an 8M non-inferiority criterion | bias-only leaves 4.4% of actions beyond 0.9 |
| M9 | MUST | E2.1: add the `frozen` (one-iteration lag) arm, replace horizon 16, log slope persistence first | EMA confounds averaging with removal of in-iteration co-adaptation |
| M10 | MUST | E2.3: budget-free upper-bound fork before the budgeted trainer; antithetic K = 2; fix the "0.9 ceiling"; compare to the full-rollout reference | plan's own numbers give x1.6-2.5 total information and a 0.45 ceiling, not 0.15 -> 0.4 |
| M11 | MUST | Move the compute decision to Stage 0 (census arms), test the assembled lean recipe first, n = 8 | non-inferiority at 0.6k with n = 4 passes a truly equal arm only half the time |
| M12 | MUST | Add what is missing: PPO reference under the same protocol, from-scratch confirmation rule, Hopper / Walker2d probe, multi-episode deterministic eval | "beats PPO" is currently assumed |
| M13 | MUST | Get the user's explicit agreement to `--rng-stream` | it is a seed in everything but name; the ML skill says "don't do multiple seeds" |
| S1-S6 | SHOULD | see section 9 | |
| C1-C4 | CONSIDER | see section 9 | |

---

## 1. Does each experiment test its hypothesis? Power, recomputed [CALC]

### 1.1 Fork arms, "n = 4 resolves ~0.6k at 8M"

The 0.6k is 2 x sd x sqrt(2/4) with sd = 0.42k. That is the difference at which a z-test has **50%** power.
With the variance estimated from the data (two-sample t, alpha 0.05 two-sided):

| per-arm n | 80%-power MDE at sd 0.42k | at sd 0.84k | at sd 0.25k (4M) |
|---|---:|---:|---:|
| 3 | 1.29k | 2.58k | 0.77k |
| 4 | 1.00k | 2.00k | 0.60k |
| 6 | 0.75k | 1.51k | 0.45k |
| 8 | 0.63k | 1.27k | 0.38k |
| 12 | 0.50k | 1.01k | 0.30k |

Power of n = 4 at sd 0.42k: 0.40 at a true 0.6k, 0.80 at 1.0k. At sd 0.84k: 0.14 / 0.30 / 0.56 / 0.80 at
0.6 / 1.0 / 1.5 / 2.0k.

The sd itself is n = 3 (paths A, B, E; one of them a fork). Its 95% interval is x0.52 to x6.3, not "a factor
of ~2" [CALC, chi-square with 2 df]. Fork control vs base already differed by 744-840 at 8M [DOC 01].

Consequence for the plan's own predictions: E1.2 "+1-2k at 8M", E2.2 "+0.8-2k at 2-3M", E2.3 "+1k at 4M" sit
at 1-2 MDE for n = 4 if sd is 0.42k and below one MDE if it is 0.84k. The falsifiers ("slope does not
change", "gain lost by 8M") are null results, which n = 4 cannot assert at all.

**Pseudo-replication.** Every fork in the plan starts from one trajectory's checkpoint (path B, upright,
11069: one of the best runs in the record). Four streams from one parent measure post-fork noise only. A
result may be a property of that critic and actor state.

**Required spec (M4).**
- The Stage 0 census saves full checkpoints at 1M and 4M for every stream (`--checkpoint-steps 1e6,4e6`;
  the first-iteration-past-threshold rule of `v60:901` applies).
- Fork experiments use **4 upright parents x 2 streams = 8 runs per arm**, same parents and streams in every
  arm, analysed as a blocked design (arm difference within parent, then across parents).
- E0.2 becomes the fork noise floor on that design (8 control forks at each fork point), and its sd, not
  0.42k, sets the n of every later fork experiment. If the measured fork sd at 8M exceeds 0.6k, 8M-endpoint
  fork experiments go to n = 12 or are read only at fork + 2M.
- Report effect with a 95% interval. "Null" means the interval excludes the pre-registered effect size, not
  "p > 0.05".
- About 25 arm-vs-control comparisons are planned; expect one false positive at alpha 0.05. Every positive is
  re-run from scratch on streams it has not used (M12).

### 1.2 Binomial designs (E0.3, E1.1)

Reference trapped rate: 0.22 [0.16, 0.30] for all tanh runs, 6/23 reference-like, 2/4 reference [DOC 09].

| Quantity | p = 0.06 | 0.10 | 0.16 | 0.22 | 0.30 | 0.45 |
|---|---:|---:|---:|---:|---:|---:|
| E0.3 "3-7 of 16 trapped" is observed | - | 0.21 | 0.48 | 0.71 | 0.83 | 0.56 |
| E1.1 "<= 1 of 16" is observed | 0.75 | 0.52 | 0.25 | 0.10 | 0.03 | 0.001 |
| "<= 1 of 32" | 0.42 | - | 0.03 | 0.004 | - | - |
| "0 of 32" | 0.14 | - | 0.004 | - | - | - |

- E0.3's prediction is close to unfalsifiable: it comes true with probability 0.48-0.83 for any rate from
  0.16 to 0.45. Treat E0.3 as an estimate (16 runs give +-0.2), not a test.
- E1.1's pass rule passes an **unchanged** arm 10% of the time at p = 0.22 and 25% at p = 0.16, and fails a
  truly fixed arm (p = 0.06) 25% of the time.
- As a two-arm comparison it is weaker still. One-sided Fisher exact, alpha 0.05:

| n per arm | 0.22 -> 0 | 0.22 -> 0.06 | 0.30 -> 0 | 0.30 -> 0.06 | 0.45 -> 0 | 0.45 -> 0.06 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 0.27 | 0.15 | 0.55 | 0.35 | 0.91 | 0.74 |
| 24 | 0.63 | 0.34 | 0.89 | 0.62 | 1.00 | 0.93 |
| 32 | 0.86 | 0.43 | 0.98 | 0.73 | 1.00 | 0.98 |
| 48 | 0.99 | 0.66 | 1.00 | 0.91 | 1.00 | 1.00 |

  16 per arm distinguishes "0.22-0.45 from < 0.06" only if the baseline is at the top of its range. 0 of 16
  bounds the rate below 0.21, 0 of 32 below 0.11 (Clopper-Pearson).

**Required spec (M5).**
- Primary endpoint of every from-scratch census: **inverted share of rows over 0-0.5M** per run (continuous,
  available at 0.5M, compared by Mann-Whitney). Archive values are 0.08-0.17 for the upright reference and
  0.91-0.99 for trapped runs [DOC 09], against a prediction of < 0.05: 16 per arm is ample for that.
- Secondary: trapped at 1M (inverted share over 0.75-1.0M > 0.5). A claim "flip rate below 0.1" needs 0-1 of
  32 pooled across the passing arm's two waves.
- The **control census is not culled**: 10 of 37 trajectories that sat on the back-running curve left it,
  two of them after 2M [DOC 09], and the all-stream 8M mean is the baseline every later claim is measured
  against. Cost of letting ~4 trapped runs finish: ~24M steps. Treatment arms may be culled at 2M when
  inverted share over the last 0.5M exceeds 0.9, and are then scored at the trapped plateau (2.4-2.5k).
  `autocull --enforce all` must not run on the control census; use `--enforce health`.
- Report flip count and upright-conditional mean separately. The all-stream mean is bimodal: with 16 runs
  its standard error is ~0.9k from the flip count alone.

### 1.3 Per-experiment confounds

| Item | What it claims to test | Confounds as written | Falsifiable at stated n? |
|---|---|---|---|
| E0.1 | instruments are non-invasive | fails for reasons unrelated to v64 (section 3) | yes, once restated |
| E0.2 | fork arms reproducible | a duplicate with identical seeds is deterministic by construction; it tests determinism, not noise | trivially; the informative quantity is the fork sd (1.1) |
| E0.3 | honest baseline | culling at 2M biases it down; init is shared (section 2) | as an estimate |
| E1.1a | initial noise chooses the basin | sigma changes the KL geometry (the same mean move costs 3.9x the KL at sigma 0.55, so early mean steps are ~half as large), the critic's early data spread, and the step at which the dual starts pushing sigma up; bias-only init is not the stated distribution (section 4) | continuous endpoint yes; binary no |
| E1.1b | head shape | Beta with a squashed-entropy dual ran at action sd 0.12-0.22 in v18 against 0.33-0.47 for the reference [DOC 08]: head shape is confounded with noise level; "large" port, unusable on resume [DOC 10] | no 8M criterion stated although the known risk is -2.2k at 8M |
| E1.2a | slosh | flag also slows the KL multiplier 10x (M7) | at n = 8 |
| E1.2b | saturation-driven inflation | three different objects under one name (section 4); predicted size ~0.3k | no |
| E1.2c | "known-positive reference" | v31 is not positive inside 8M: -0.2k / +0.3k, crash share 5.8-8.3% at cap 0.02, deterministic eval 9.2-10.7k [DOC 09, 10] | it is a null reference |
| E2.1 | does the critic accumulate slope information | EMA = averaging + lag + removal of in-iteration co-adaptation (section 5) | reading needs a pooled instrument window |
| E2.2 | early step too small | multiplier transient on the way back; early realised KL is already 2-3x the bound | n = 8 yes |
| E2.3 | transition channel | success measured against a weakened control; gate ignores displaced real steps | no (section 5) |
| Stage 3 | cheaper at equal return | one-factor arms do not bound the assembled recipe; n = 4 non-inferiority | no (section 6) |

---

## 2. `--rng-stream`

How v60 seeds [CODE]:

| Stream | Seeded at | Used for |
|---|---|---|
| `random`, `np.random`, `torch.manual_seed(seed)` | `v60:359-361` | network init (actor is the first module built, then critic); the default CUDA generator: minibatch rows (`uniform_rows`, `v60:622`), reparameterisation noise in `actor_loss`, the K = 4 bootstrap actions in `bootstrap_values` (`squashed_sample` -> `randn_like`) |
| `diagnostic_generator` | `v60:435` | `critic_diagnostics`, `information_price`, `regret_probe`, the diagnostic row permutation; no effect on learning in dual mode (`v60:835` is skipped) |
| `sampler = np.random.default_rng(seed)` | `v60:650` | host action noise, including the 1000-step phase warmup (`warmup_action` calls `act`) |
| `compute_phase_offsets(..., seed)`, `run_phase_warmup(..., seed=seed)` | `v60:727-730` | episode-age phases and env reset seeds |
| `eval_env.reset(seed=seed + 10_000 + episode)` | `v60:669` | deterministic evaluation |

Findings:

1. **Implementable in one line, and stream 0 can be bit-identical.** `np.random.default_rng([seed, k])`
   with k = 0 produces the same stream as `default_rng(seed)` (SeedSequence zero-pads its entropy pool;
   verified numerically here [CALC]); k = 1 differs. So the flag needs no special case for k = 0. Pin it with
   a test, because that equality is an implementation detail of NumPy.
2. **It perturbs the warmup too**, so streams differ in env state and observation statistics before the
   first recorded step. Good for divergence; say so in the header instead of "only the action noise".
3. **Independence is conditional on the seed-1 init.** Init weights, the minibatch index sequence, the loss
   noise and the reset-noise sequence are shared; only chaos decorrelates the runs. The record supports that
   it does (v61 control differs from v60 base only in RNG consumption and landed in the other basin), so
   binomial counts are usable. But the measured flip rate is P(flip | this init). The untrained net is not
   neutral: its pre-tanh mean has sd ~0.35 across states and its log-std sd ~0.35 [INF from the default
   `nn.Linear` init of the 512 -> 12 layer]. A seed-specific tendency cannot be seen from streams.
   - That conditional distribution is exactly what a "seed = 1" benchmark number is drawn from, so the
     protocol is within the spirit of the rule for *internal decisions*.
   - For the external claim "beats PPO" it is not the seed-averaged quantity readers assume. Report it as
     "seed 1, 16 noise streams".
4. **It is a seed in everything but name.** `CLAUDE.md` says "always use seed=1" and the ML skill says "don't
   do multiple seeds" on compute grounds. The honest position: the flip lottery makes any single-run
   comparison under ~3k uninterpretable [DOC 01], sixty versions were steered by such comparisons, and an 8M
   REPPO run costs ~4 minutes of box time. Put that to the user once and get a yes (M13). Do not let the flag
   stand in for the conversation.
5. **Better alternatives?** None that is cleaner. Varying `--seed` on forks only is equivalent in effect and
   more plainly against the rule. Perturbing the CUDA generator instead gives the same conditional replicate
   at the cost of touching a captured graph's generator. The one useful addition is the blocked parent
   design of 1.1, which turns stream replicates into replicates over trajectories.
6. **PPO needs the same flag** (M12). The frozen PPO files cannot be retrofitted; the reference arm needs a
   new versioned copy of the newest `ppo_continuous_action_*` with `--rng-stream` on its Beta host sampler.

---

## 3. Stage 0 instruments

### 3.1 Is "bit-exact 11013.6 with instruments on" achievable?

Not as specified. Three independent obstacles:

1. **The plan thins something that feeds the learning RNG [CODE].** Each iteration v60 calls
   `refresh_targets(diagnostic_lambda, ...)` twice (`v60:826-827`). Both go through the compiled
   `bootstrap_values`, whose `squashed_sample` draws `(4, 16384, 6)` normals from the **default CUDA
   generator**, the same generator that then supplies the minibatch rows and the actor's noise. "Diagnostics
   every 8th iteration" removes those draws on 7 of 8 iterations and changes every later minibatch. This is
   the mechanism by which the v61 control left path B.
2. **The reference environment has moved [CODE, file dates].** `cleanrl/shared/gae.py` and `ppo_loop.py`
   carry uncommitted edits dated 2 Oct (a `per_step` lambda branch inside the Triton GAE kernel v60 calls
   through `get_gae_fn`); `v60_base` ran on 29 Sep and no REPPO run is newer. The scalar path looks
   arithmetically unchanged, but the kernel is recompiled and bit-equality is not guaranteed. Unmodified v60
   may itself no longer reproduce 11013.6.
3. **Any edit inside a compiled loss can change floating-point order.** Row-split metrics added to
   `critic_loss` / `actor_loss`, or `kl_bound` turned into a tensor, alter the traced graph. Inductor does
   not promise the same fusion and reduction order for the outputs that were already there.

**Required spec (M1): how to guarantee non-invasiveness.**
- Rule 1: `critic_loss`, `actor_loss`, `bootstrap_values`, `encode_targets`, `critic_step`, `actor_step` and
  the update loop are textually identical to v60 when every new flag is at its default. New behaviour lives
  in separate functions selected at construction time, never in a branch inside a traced function.
- Rule 2: instruments are eager, `no_grad` (or their own `autograd.grad`), run outside the `update` span, and
  draw only from `diagnostic_generator` or a new `instrument_generator`. Nothing new touches the default CUDA
  generator or the host sampler.
- Rule 3: the two diagnostic `refresh_targets` calls stay on every iteration (5 ms). Only the consumers of
  `diagnostic_generator` (`critic_diagnostics`, `information_price`, `regret_probe`) and the new instruments
  are thinned. `diagnostics/*` then differs from v60's values; learning does not.
- Rule 4: parity is checked against the v60_base tfevents on tags the instruments cannot touch
  (`charts/episodic_return`, `losses/*`, `policy/*`, `grad/*`), value by value, by a script that follows the
  run and cancels it at the first mismatch. A mismatch shows within the first iterations, so a failed parity
  run costs seconds of GPU, and the passing one is the full 8M stream-0 run the plan needs anyway.
- Rule 5: submit unmodified v60 in the same wave. If v64 and v60 agree with each other but not with the
  archive, the environment moved and the new pair becomes the reference; record that and continue. If v64
  differs from today's v60, the instruments are invasive.
- Rule 6: the same check is repeated for every later version's control arm (flags at default vs v64
  stream 0). It is the implementation test the plan asks for and costs nothing extra.

### 3.2 The realised-gain instrument (08 section 4)

Implementable from what the trainer holds at the start of iteration t + 1 [CODE]:

| Needed | Where it is in v60 |
|---|---|
| pi_old (behaviour of rollout t) | `previous_actor` after the insert copies (`v60:794-795`) |
| pi_new (behaviour of rollout t + 1) | `old_actor` = `actor` |
| fresh rows, their noise | `rollout_actions`, `rollout_noise` |
| advantages the update did not see | `diagnostic_targets` / `hard_targets` minus the K-sample baseline under `target_critic`, which at that point has never been fit on rollout t + 1 |

Cost: one 16,384-row forward of `previous_actor` plus arithmetic, about 1 ms. No env steps.

Three things the plan gets wrong or leaves out:

1. **"Held-out ... trapezoid form" is a contradiction (M3).** The trapezoid averages the backward form (next
   rollout: held out) with the forward form (rollout t: the rows the critic and actor just trained on). 08
   names in-sample rows as "the important" bias (memorisation; possibly several times the truth). A held-out
   forward form needs 2 of 16 lanes withheld from both minibatch streams, which changes learning and breaks
   parity. Spec: log the **backward form as the primary**, the in-sample forward form as a flagged secondary,
   and their difference as the memorisation meter (08's I3). The trapezoid is available only in a separate,
   non-parity arm with withheld lanes. The backward form's own second-order bias is ~7% [DOC 08].
2. **The pre-update critic no longer exists when it is needed.** `target_critic` is refreshed every 16 critic
   steps (`v60:861-862`), so at the end of the update it is the critic of step 112. The "predicted gain of the
   pre-update critic" needs an explicit frozen copy taken at insert (`pre_critic`, 1.4M parameters). Compute
   the predicted gain right after the update, eager, with common noise from the instrument generator, for
   both `pre_critic` and the live critic; log mean part, scale part and T dH separately.
3. **Behaviour-time moments.** Training renormalises raw observations with end-of-rollout statistics, so
   `old_actor(normalize(raw))` is not exactly the behaviour policy [DOC 02 L4]. Add the host actor's
   `(mean, log_std)` and the behaviour-time normalised observation to `RolloutTransfer(fields=...)` (buffer
   layout only; no numerics) and evaluate pi_old on the behaviour-time observation.

What would make it lie:

- **It is blind where D2 says the action is.** The advantage is a bootstrapped lambda-return against a critic
  that values the inverted state at 13.4 where the truth is 3.8 [DOC 09]. A tumble's 900-step cost is mostly
  outside a lambda-0.95 trace and is mis-baselined. The instrument can report real gains while the return
  stalls on tumbles. Log it split by posture, winsorised and raw, with a lane-jackknife SE; add a fixed
  n = 10 truncated-return variant (08's own recommendation).
- **Late SNR.** Per-iteration SNR for net progress is 0.6-0.9 at 4-5M and 0.13-0.3 at 8M [DOC 08]. It cannot
  show a plateau moving. It can compare arms if the window is fixed in advance: pooled over the first 60
  fork iterations and 8 runs, the SE is ~1.3 return per iteration per arm against predicted differences of
  6-20 [CALC from 08's SE of 28].
- **Scale dominates.** Both real updates measured so far were dominated by the dual moving sigma [DOC 08].
  A total that is not split into mean / scale / T dH will read the slosh, not the mean step.
- **N = 2.** The claim "pre-update prediction matches realised gain" rests on two updates at ~2 SE each.
  Stage 0 is the first real validation; treat calibration as a hypothesis until a few hundred iterations
  are in.

### 3.3 Other instrument fixes

- **Posture (M2).** `obs[1]` is the unwrapped root pitch. Late crashes are 0.5-2.5 forward revolutions and 3
  of 8 land upright after a full one [DOC 09]. `abs(obs[1]) > pi/2` scores those as inverted for the rest of
  the episode. Use `cos(obs[1]) < 0` for inverted rows and count sign changes of `cos` per episode for
  tumbles. `scripts/reppo_posture.py` and 09's census use the `abs` rule; early half-flips are unaffected,
  late crash shares may be overstated.
- **Slope persistence (08's I5) belongs in Stage 0.** Keep a ring of the last 8 pre-update critics and log
  the cosine of their dQ/da at the same held-out rows at lags 1, 2, 4, 8 (eager, thinned). It answers the
  D4 dispute (memory 3-6 iterations vs "rewrites 59-113% per iteration") before any EMA run and tells E2.1
  which horizon to test.
- **Deterministic eval.** One episode every 25 iterations is too thin for "the deterministic plateau moves".
  Use 4 episodes (own env, own seeds, host mirror: non-invasive), and score every arm's final checkpoint
  with `scripts/fork_eval.py`.
- **E0.0 (CPU, seconds).** Run `scripts/13_random_posture.py`: 16 episodes each of untrained
  tanh(N(0, 1.08)), 0.80, 0.55, 0.40, Beta(1.69, 1.69), the seed-1 init net as is, with a log-std bias, and
  with a small-gain last layer. If the reference init is not inverted far more than the quiet ones, E1.1a's
  premise is wrong before any GPU time is spent; if the seed-1 net differs from the iid version, the init
  (not just the noise level) matters, which bears on section 2 point 3.

---

## 4. Stage 1

### 4.1 Is "initial pre-tanh sigma 0.55" a free parameter?

Yes, at initialisation only, and it persists for roughly the window that matters [CODE + CALC]:

- The actor's last layer outputs `(mean, log_std)` with no clamp and PyTorch's default init (`v60:228-229`).
  Initial sigma is therefore ~1.08 on average and varies by state; 17% of actions are beyond |a| 0.9.
- The entropy target is -0.5 nats per dimension of **squashed** entropy (`v60:397`, `:500-501`). At sigma
  1.08 the squashed entropy is +0.65 per dimension, at 0.55 it is +0.55; -0.5 corresponds to sigma 0.15 for
  an unsaturated mean. So at the start the constraint is slack by ~1 nat per dimension in either arm and the
  dual only **lowers** the temperature (0.01 -> 0.0015 over the first 0.5M [DOC 09]). It does not push sigma
  back up until entropy reaches the target, around 1M.
- Until then sigma is set by the pathwise gradient through the critic under the KL gate. A lower start is
  not undone.

Two specification problems:

1. **Bias-only is not the stated distribution.** Shifting the log-std bias by log 0.55 leaves the state-
   dependent spread of both heads: action sd 0.51 and 4.4% beyond |a| 0.9, against 0.445 / 0.75% for a
   uniform sigma of 0.55 and 0.478 / 2.6% for Beta(1.69, 1.69) [CALC]. To get the quiet, bounded start the
   census credits, the last layer needs a small gain as well (the PPO-standard 0.01), which also zeroes the
   random initial mean function. Those are two interventions.
2. **The same KL moves the mean half as far.** Per-state KL is in sigma units. At sigma 0.55 a mean step of
   a given size costs 3.9x the KL it costs at 1.08. The arm is "quieter and more timid". The archive says
   quiet-start families are faster early, so the sign is probably fine, but the attribution is not clean.

**Required spec (M8), E1.1, all from scratch, 16 streams per arm, wave 1:**

| Arm | Change | Lines |
|---|---|---|
| ctl | v64 reference (this is E0.3) | 0 |
| a1 | last-layer gain 0.01, log-std bias log 0.55 | ~5 |
| a2 | last-layer gain 0.01, log-std bias 0 (sigma 1.0): isolates the random initial mean from the noise level | same flag |
| a3 | `--gve-lambda 0.95` for the first 0.5M then 0.8 (09's intervention 2: 7/61 v 19/59 trapped, p 0.008, no early speed lost) | ~10 (device tensor, new function per 3.1 rule 1) |
| a4 | `--kl-start 1.0`: enforce the gate from iteration 1 (realised KL is 0.035-0.10 for the first 20 iterations while the multiplier climbs from 0.01 [DOC 09]) | 0 |

- Beta (the plan's 1.1b) moves to a conditional arm: run it only if a1 fails the posture endpoint. It is the
  largest port, it conflates head shape with noise level, and its 8M record is -2.2k.
- Pass, per arm: inverted share over 0-0.5M below half the control's (Mann-Whitney p < 0.05); and
  upright-conditional 8M mean not more than 0.5k below the control's (one-sided 95% interval); and no more
  trapped runs than the control.
- An arm that fails the posture endpoint at 1M is cancelled as a whole. Passing arms run to 8M.
- Falsifier of D2-early: a1 and a2 both leave the inverted share within 25% of the control's.
- The passing arm is re-run on 16 new streams as part of Stage 4 (that is where the "0-1 of 32" claim comes
  from).

### 4.2 "Constraint on pre-squash sigma instead of squashed entropy"

Well-defined only once three choices are made, because the entropy enters v60 in three places [CODE]: the
dual's constraint (`v60:500-501`), the actor's objective (`temperature * logprob`, `v60:495`, where the
`-tanh_log_det` term is the force that pushes means away from saturation), and the critic's soft bootstrap
(`v60:455`).

- Constraint only: the actor still maximises squashed entropy at a temperature set by a different quantity.
  Incoherent.
- All three: a consistent max-ent MDP with Gaussian (pre-squash) entropy. The actor loses its anti-
  saturation force; expect means to drift toward the bounds. A fork also changes the critic's target
  definition mid-run, so the first ~0.3M of the fork is re-adaptation.
- Either way a **level** must be chosen. Continuing from the 4M checkpoint's own sigma (0.46) the arm's
  entire predicted effect is the avoided inflation 0.46 -> 0.52: (0.52^2 - 0.46^2) / 0.52^2 of a ~1.4k noise
  cost, about **0.3k** [CALC from 08's sigma^2 scaling]. That is half the n = 8 MDE. The fork cannot answer
  its question.

Against the ML guidance ("do not reach reflexively for clamping, limits, annealing"): a sigma floor or
target is exactly v20 / v22's "exploration budget" (dual on mean log std at 0.1-0.2; 7.5-8.5k at 8M, on the
entropy-free Beta base) and v31's cap is a clamp. Replacing one constant by another constant in a nicer
coordinate is defensible as a re-parameterisation (same count of constants, removes the arithmetic fact that
-0.5 is unreachable for |a| >= 0.91). It is not a diagnosis-led fix, and it should not be sold as one.

What the archive already established and the plan does not use [DOC 09 section 2.4, 10 section 5]:

| Version | Result | What it means for Stage 1 |
|---|---|---|
| v31 price cap | -0.2k / +0.3k at 8M; +0.9-1.0k only at 16M; crash share 5.8-8.3% (cap 0.02), deterministic eval 9.2-10.7k | not a positive reference inside 8M; lower temperature did not lower tumbles |
| v42 forks (-1.0 / -1.5 / -2.0 from 8M) | +0.8-1.0k stochastic at 12M, crash share still 3-8%, q_mse and actor gradient worse, saturation x2-5 | the tumble rate is a fixed point of the learner; the policy re-adapts and tumbles at the same rate |
| v43 low target from 0.8M | crash share 0.1-0.3%, q_mse 0.1, **stalls at 5.9-6.9k** | "tumbles vanish but the slope does not improve" has already been observed once (TF32, never reached the fast gait) |
| v43 late drop | 11571 at 8M with the reference's deterministic return | the gain is removed evaluation noise, not learning |
| v44 sharp mean | 6.6-7.4k | reading the slope at the mode instead of under noise is worse |
| v47 no entropy | crashes 3-6% at action sd 0.06-0.08 | low noise is not sufficient for a robust gait |
| v54 / v56 IDS price | 9.2-9.7k; v56 is the only run past 7k with ~1% crash share, and the slowest | "noise priced by the critic's own measured cost against information value" **was** tried; the measured price came from the same weak slope |

So the plan's E1.2 prediction ("tumbled-episode share 12% -> < 3% ... the deterministic plateau moves") is
contradicted for arms (b) and (c) by every comparable run, and arm (a) has no mechanism for it. At most ~1k
of stochastic return is available at 8M from noise level alone, and most of it is evaluation noise.

A more principled formulation exists, in two parts:

1. **The cost side of noise is measurable on-policy without the critic's slope.** By Stein's identity the
   derivative of the return with respect to log sigma_d is E[A (eps_d^2 - 1)], which v60 already logs as
   `probe/true_curvature` from hard returns (`v60:590-592`). Its per-dimension SE is about
   sd(A) sqrt(2) / sqrt(16384) = 0.013 per iteration against a signal of roughly 0.05 per dimension late
   [CALC from 08's 1.4k noise cost]: SNR ~3 per iteration per dimension, where the mean slope's is below 1.
   Following that gradient alone is what PPO does, and it collapses sigma and stalls at ~7k (v19, v21).
2. **So the missing quantity is the information value of noise, and it has to be measured as learning, not
   as slope.** The one on-policy way to measure it is to run different noise levels at once and compare
   what the learner gets from each (bet B2 in section 8). Everything else (target, cap, floor, anneal) is a
   constant standing in for that number.

For the *benchmark*, one schedule is principled rather than a patch: the value of information goes to zero
as the remaining budget goes to zero, so exploration priced by remaining steps is a finite-horizon
statement, not a symptom fix. It buys the ~1k of evaluation noise honestly, provided deterministic and
stochastic returns are both reported.

**Required spec (M6, M7), E1.2, forks at 4M, 4 parents x 2 streams per arm:**

| Arm | Change | Pre-registered prediction |
|---|---|---|
| ctl | none | - |
| t (new, the decisive one) | **diagnostic only**: training envs end the episode (termination, no bootstrap) after 20 consecutive steps with `cos(pitch) < 0`; evaluation on the unmodified env | if D2-late is right: q_mse, actor gradient norm and KL multiplier return to their 2-3M relative levels within 0.3M, and the 4M -> 8M slope of the *unmodified-env* deterministic return rises by >= 30%. If metrics recover and the slope does not: the plateau is not tumbles; Stage 2 carries the weight. This is 09's own "decisive" test and separates its 75% claim from its 50% claim |
| a | temperature-only dual step / 10 (own optimizer group; `v60:395-401` puts `log_kl_multiplier` and `log_dual_temperature` in one group, so the existing flag would also slow the KL multiplier) | iteration-to-iteration sd of `policy/std` falls >= 3x with mean sigma within 5% of control; return effect <= 0.5k either way. A slope change >= 30% would be a surprise worth following |
| b | budget-priced entropy target (linear in remaining steps from the 4M value to -1.5 per dimension) | +0.6-1.0k stochastic at 8M, deterministic within 0.4k of control, crash share unchanged (3-8%) |

Arm t uses task knowledge and is never a method; it is the upper bound for "remove tumbles". The pre-squash
arm and the v31 cap are dropped.

---

## 5. Stage 2

### 5.1 EMA critic for the actor

Not ill-posed, but the design does not isolate what it claims to.

- **Weight averaging is sound here.** The critic is Linear + RMSNorm + SiLU with no running statistics, and
  consecutive iterates are warm-started. Averaging weights is, to first order, averaging logits, i.e. a
  geometric mixture of the categorical outputs; the decoded value is not the mean of decoded values, but the
  difference is second order while iterates stay close. A critic step moves a weight by about lr per step:
  ~0.003 per iteration as a random walk against a weight scale of ~0.03 [CALC]. Horizon 4 is safe; 16 is
  where "close" starts to fail.
- **The real hazard at long horizons is support, not weights.** The actor queries the averaged critic at the
  *current* policy's actions. The mean moves ~0.2-0.28 sigma rms per iteration [DOC 08]: 0.4-0.6 sigma after
  4 iterations, 0.8 (random walk) to 3+ (coherent) after 16. Old critics were fit on other actions and for
  other policies' Q. Horizon 16 mostly tests extrapolation.
- **EMA changes three things at once:** temporal averaging; lag; and removal of in-iteration co-adaptation
  (the actor no longer climbs a critic fit on the same rows minutes earlier; 08 measured that critic's
  predicted gain inflated 3.4x at 12M). v9 "frozen judge" is the pure-lag point and was -0.5k at 1M, +0.4k at
  8M on a weak recipe. Without a lag-only arm a positive EMA result does not say "the critic accumulates
  slope information".
- **A null would not settle D4 either**, because it could be the weight-space approximation. The function-
  space version (mean of dQ/da over K frozen critics, each under its own normalisation) is exact and costs
  +10-25% update time at K = 4 [DOC 10].

**Required spec (M9), E2.1:**
- First read the Stage 0 slope-persistence curve. If the lag-1 cosine of dQ/da at fixed rows is above ~0.8
  there is nothing to average: skip E2.1. If it is below ~0.3 with no slow component, averaging at horizon m
  recovers at most a factor sqrt(m) on noise that is mostly not signal: run it. In between, run it.
- Arms (forks at 1M, 4 parents x 2 streams): `live` (control), `frozen` (pre-update critic), `ema` at
  horizon 2 and 6 iterations (or the two horizons the persistence curve brackets).
- Readouts, pooled over the first 60 fork iterations: backward realised gain per iteration (mean part),
  pre-critic predicted / realised ratio, in-sample minus held-out gain. Return at 2M and 4M.
- If `ema` beats both `live` and `frozen` by the MDE: confirm with the K-snapshot dQ/da average, then repeat
  at 4M. If `frozen` matches `ema`: the effect is co-adaptation, and the cheaper fix is the frozen judge.
- Prediction under 07: +30-100% realised mean-part gain; under 08: null. Keep those.

### 5.2 KL window forks

"Raise for 1M -> 3M, then return to 0.03" is a reasonable test with four defects:

1. The multiplier adapts at 0.15 log units per iteration. On the way back there are ~10 iterations of
   realised KL above the bound. Log realised KL and read the "persistent effect" from 0.5M after the return.
2. `kl_bound` is a Python float inside the compiled `actor_loss` (`v60:497`, `:503`). Making it a tensor
   edits the traced graph; the control must still match v64 stream 0 bit for bit (3.1 rule 6).
3. A larger KL also lets sigma move further per iteration. With the slosh already spending part of the
   budget on sigma, log the mean part and scale part of the realised KL separately, or the arm tests "more
   slosh" as much as "bigger mean step".
4. The Stage 0 instrument answers part of this for free: realised / predicted ~1 with a persistently
   positive gain means the step is too small; a ratio well below 1 means it is not. Read that before
   choosing bounds.

Keep the arms (0.06 and 0.1, window 1M -> 3M, then 0.03; forks at 1M; 4 parents x 2 streams). Falsifier as in
the plan. The "ratio controller" is a classical trust-region test and is fine in principle, with two
conditions: it must move the bound in policy space (v58's weight-space extrapolation regressed 2.6k at 4M),
and it must use the backward (held-out) ratio. It is only usable to ~3M, so build it only if the fixed
windows are positive at 3M.

### 5.3 Vine

**Accounting.** Counting every branch control step against the 8M budget is right, and 16 x 768 real + 4,096
branch steps per iteration is consistent. Three gaps:

- With the real trajectory as branch 0, the K - 1 alternatives cost (K - 1) n steps for K - 1 contrasts:
  information per branch step is c^2 / (1 - c^2) / n **for any K**. 07's (K - 1) / (K n) factor, and its
  "larger K is better", assumed all K branches are paid for. Once branch 0 is free, K = 2 is as efficient
  per step as K = 8 and covers 7x as many states.
- "n = 1 labels are noise-free" holds only if the tail's K = 4 bootstrap actions and entropy term use the
  same noise for every branch of a state. `bootstrap_values` draws fresh noise per call (`v60:453`). Specify
  common noise in the tail, and recompute branch labels at each of the 8 target refreshes.
- The "held-out branches, ~0.9 ceiling" metric is mis-stated. 0.90 is the correlation of the *expected* n = 5
  label with the truth. A critic scored against single held-out labels has a ceiling of sqrt(label
  reliability): 1.0 at n = 1, 0.60 at n = 5, 0.40 at n = 10 [CALC from 07's r_n]. And at n = 1 the label is
  >= 96% the critic's own V' [DOC 07], so a high score there says the Q head has learned V' composed with the
  dynamics, whose correlation with the truth is capped at 0.45.

**The gate does not hold up.** "Information per env step >= 2x an ordinary sample" ignores what the branch
steps displace and what the labels are capped at. With the plan's own numbers [CALC from 07 section E]:

| Split per iteration | Ordinary information | Vine information (n = 1, c = 0.447) | Total v baseline |
|---|---:|---:|---:|
| 16,384 real (baseline) | 16,384 x 0.064 = 1,049 | 0 | 1.0 |
| 12,288 real + 4,096 branch | 786 | 4,096 x 0.25 = 1,024 | 1.7 |
| 8,192 real + 8,192 branch | 524 | 2,048 | 2.5 |

A 1.7-2.5x information gain moves a shrinkage estimator's correlation by about its square root: 0.19 ->
0.25-0.30, under a hard ceiling of 0.45, on numbers that hold at 1M only, with six top-energy states dropped
(pooled with them, n = 1 is 0.05). The plan's prediction "held-out contrast corr 0.15 -> > 0.4" is not
consistent with that. And 07 section D puts the critic's aggregate cosine in policy-parameter space at
0.67-0.74 at 4M already, so even a perfect slope is worth at most ~1.4x gain per KL-limited step.

The control is also the wrong yardstick. "+1k over the same shortened rollout without branches" can be
passed by an arm that is below the full-rollout reference, since shorter rollouts were worse at 8M in every
recipe [DOC 10].

**Is "extra rows plus a contrast loss" the right use of the labels?**

| Use | Assessment |
|---|---|
| Extra (s, a_k) rows with target G_real(s) + (y_k - y_0) (10's design) | The K rows of a state share one draw of G's noise (93-95% of its variance). The contrast is clean; the level is the same noisy label replicated K times, which over-weights those states' level noise. With K = 2 the replication is 2x and tolerable |
| Extra rows **plus** a scalar within-state contrast loss | A second loss type on decoded values next to a categorical CE: a new coefficient and a scale mismatch (the v34 confound 10 warns about). Not needed if the rows carry the contrast |
| Per-state local slope fit (K >= 7, n = 1) | Gives the exact secant gradient of r + gamma V'(f(s, a)) at ~585 states per iteration. Aggregate information is *less* than the existing critic's on 12k states (585 x 0.25 against ~460 in the same units) unless a network generalises it across states, at which point it is the row design again |
| Branches for the actor only, score-function form | Unbiased and needs no critic for the slope; same state-coverage problem; useful as the upper-bound arm's second readout |
| **Antithetic pair: a- = tanh(mu - sigma eps) against the real a+** | Recommended. One extra branch per state, the difference y+ - y- cancels the level and every even-order term, so it is the directional derivative along eps to second order. Usable as a critic row pair and as a direct actor term ((y+ - y-) / 2) eps / sigma |

**Required spec (M10), E2.3:**

- (i) Oracle completion as planned (4M and 7.9M checkpoints, n = 1..10, reward / V' split, CPU through
  `mlq`), plus: K = 2 antithetic labels, all states (no dropping; report trimmed and untrimmed), and the
  gate restated as **total** information at the proposed split >= 2x the full-rollout baseline with the
  label ceiling >= 0.6. If n = 1 is the only horizon that clears the per-step bar, stop: its ceiling is V'.
- (ii-a) **Budget-free upper bound first.** Same v66 build, flag `--branch-count-steps false`: full 16 x 1024
  rollout plus free antithetic branches at 4,096 states, n in {1, 3}; forks at 1M and 4M, 4 parents x 2
  streams, against the plain control. This asks the only question that justifies the build: does a better
  action slope produce a better actor at all? Kill vine if the free version is not >= 1k ahead at fork + 3M
  (or >= +30% realised mean-part gain). Label every such run "diagnostic, env steps not counted".
- (ii-b) Only then the budgeted version (16 x 768 real + 4,096 branch steps), with **two** controls: the
  shortened rollout without branches, and the full-rollout reference. Success is beating the second.
- Tail noise common across a state's branches; labels recomputed at every refresh; branch rows carry
  `rollout_chain` = 0.
- The held-out metric is reported with its correct ceiling per n.

**On-policy status and honest reporting.** Vine is on-policy in the data sense (current-policy actions from
states of the current policy's distribution). It needs something PPO does not: resetting the simulator to
arbitrary visited states. That is TRPO's "vine" assumption; it is unavailable through the standard Gymnasium
interface and on any real system. Rules for the write-up:
- the headline "fully on-policy REPPO vs PPO" number is vine-free;
- vine results are a separate row, "requires simulator state restore", with all branch steps in the step
  count and the budget-free runs marked diagnostic;
- for Hopper / Walker2d the branch kernel must reproduce the termination rule, and the parity test must
  cover it.

### 5.4 E2.4

Fixed n = 5 targets shorten the real-return horizon, which is the direction the census associates with more
flips (1-step 3/4, lambda 0.5 1/1, 0.8 0.28, 0.95 0.12). Run it only as a fork after 1M, never from scratch,
and only if E2.3's oracle shows n = 5 labels are mostly reward rather than V'.

---

## 6. Stage 3

**Realism against 06.**
- The <= 0.07 s update is an estimate, not a measurement, and it needs every item of 06's skeleton: aux on
  every 4th step, reward head on 2 latents, 1,024-row critic minibatches, BF16 on the aux path, diagnostics
  thinned, and refresh K = 2 every 32 steps. The plan's list omits the refresh change (14 ms) and has "H1 v
  H8" as an arm that is not part of the skeleton.
- >= 55k SPS solo needs the host actor at <= 0.045 s. At 512 wide it is 0.129 s per iteration, as much as
  the physics. Without the 256-wide actor the iteration is ~0.35 s: ~46k SPS solo, and six runs are CPU-
  bound at roughly twice PPO's CPU time per step, so nearer 120-150k aggregate than 200k [CALC from 06's
  phase table]. The 256-wide actor is an untested learning change on which the target depends.
- Missing options: GPU acting at 128-256 envs (keeps the 512-wide actor at ~10 ms per iteration; 128 x 128
  was neutral at matched precision [DOC 06]); the late-fusion critic (3.6x cheaper extra actions, and the
  only design where many reparameterised actions per state are nearly free); 06's two offline measurements
  (actor-gradient variance split, precision sensitivity by path), which should precede the BF16 arm. The ML
  skill's rule is BF16 wherever the model supports it; "strict FP32 because TF32 cost return" is one
  unexplained observation, and TF32 v FP32 flip rates (2/7 v 25/116) do not support it.
- "Timing: solo 1M-step profile arms" are reduced runs, which the repo rules exclude. Every 8M learning arm
  already logs `timing/*` per iteration and accepts `--profile-iteration 40`. Take timings from stream 0 of
  each arm, submitted with `--max-parallel-runs 1`.

**The test design cannot deliver its claim.** "Return within 0.6k of the full recipe" at n = 4 is a non-
inferiority test whose one-sided 95% margin is ~0.58k: a truly equal arm passes only when its observed
deficit is ~0, i.e. half the time [CALC]. At n = 8 the margin is 0.37k and a truly equal arm passes ~86% of
the time. And five one-factor arms each "within 0.6k" do not bound their sum.

**Order.** Do the compute decision first, not after Stage 1. REPPO at three concurrent runs saturates the
GPU at ~36k SPS aggregate and starves every other queued job; every later experiment is 2-5x cheaper on a
lean base; and the aux question (69% of the update, benefit disputed between 06 and 08) is itself a root-
cause item. The objection in the plan ("from-scratch is clean only after Stage 1") is met by running the
compute arms as census arms, where the flip count is part of the readout.

**Required spec (M11):** in wave 1, next to the control census, two more from-scratch census arms of 16
streams:
- `lean-safe`: aux roll on every 4th critic step, reward head on z_0 + 1 random latent, zero coefficients
  skip the compute, diagnostics thinned, refresh K = 2 every 32. Strict FP32, widths and minibatch
  unchanged. Predicted update ~0.19 s.
- `lean-full`: lean-safe + 1,024-row critic minibatch + 256-wide actor + BF16 on the aux path. Predicted
  update ~0.07 s.
- Decision: adopt the cheapest arm whose upright-conditional 8M mean is within 0.5k of the control (one-
  sided 95%), whose flip count is not higher, and whose 1M mean is within 0.3k. If `lean-full` fails and
  `lean-safe` passes, bisect `lean-full`'s three additions as single arms on the lean-safe base (cheap by
  then). If both fail, the aux roll is earning its cost: keep the reference and record that 08 was wrong.
- All later stages run on the adopted base; its census checkpoints are the fork parents.

---

## 7. Cost, ordering, stopping rules

Box model [DOC 06]: reference recipe saturates at ~36k SPS aggregate, so an 8M run costs ~3.7 box-minutes
and 1M costs ~28 s regardless of concurrency (`--max-parallel-runs 3`). lean-safe ~2x cheaper, lean-full up
to ~5x. The plan as drafted is ~150 runs and ~9 box-hours on the reference recipe; cost was never the
constraint, decisiveness is.

| Wave | Content | Runs | Steps | Box time at 36k | Kill / stop rule |
|---|---|---:|---:|---:|---|
| 0a | E0.0 random-posture (CPU); v64 stream 0 + unmodified v60 (parity); 06's two offline measurements; vine oracle completion (CPU) | 2 GPU | 16M | 8 min | parity follower cancels at first mismatch; no wave 1 until v64 = today's v60 |
| 1 | from scratch, 16 streams each: control census (checkpoints at 1M, 4M), lean-safe, lean-full, E1.1 a1, a2, a3, a4; PPO reference 8 streams x 2 variants (new versioned file with `--rng-stream`); Hopper-v4 and Walker2d-v4 probe, v64 reference, 4 streams each | 112 REPPO + 16 PPO + 8 | <= 960M REPPO-equivalent | <= 6.5 h, less after arm-level culls (expected ~4.5 h) | E1.1 arms cancelled whole at 1M on the posture endpoint; treatment runs culled at 2M when inverted share > 0.9; control never culled |
| 2 | forks on the adopted base, 4 parents x 2 streams: fork noise floor at 1M and 4M (16; these are also the controls); E1.2 t, a, b at 4M (24); E2.1 frozen, ema-short, ema-long at 1M (24); E2.2 KL 0.06, 0.1 at 1M (16) | 80 | ~450M | ~3.5 h (1.7 h on lean-safe) | stop an arm at fork + 2M if it is > 2 fork-sd behind control; E2.1 skipped if slope persistence > 0.8 |
| 3 | vine: budget-free upper bound, n in {1, 3}, forks at 1M and 4M (32); only if it passes, budgeted + two controls (24) | 32-56 | ~180-330M | 1.4-2.5 h | oracle gate fails -> no build; free version < 1k ahead at fork + 3M -> stop |
| 4 | integration: winners in one file; 16 new streams on HalfCheetah to 8M; 8 streams each on Hopper and Walker2d; matched PPO reference on all three | ~48 | ~380M | ~3 h (well under 1 h on lean-full) | - |

Total: roughly 270-300 runs and 12-16 box-hours on the reference recipe, about a third of that if a lean
base is adopted in wave 1.

Stage-level kill criteria:

- **Stage 0.** If v64 cannot be made bit-identical to today's v60 with instruments on, the instruments run in
  a separate arm and are never on in a scored run. If the backward realised gain and the pre-critic's
  prediction disagree by more than 2 SE over a 100-iteration window at 1-3M, drop the "ratio" readouts from
  every later experiment (they were validated on N = 2).
- **Stage 1 early.** If no arm halves the 0-0.5M inverted share: initial noise is not the trigger; run the
  conditional Beta arm once, then stop spending on the head and go to B2 (section 8).
- **Stage 1 late.** If arm t (truncation diagnostic) restores the critic metrics but not the slope: the
  plateau is not tumbles. Stop all late-noise work except the budget-priced target (it is worth its ~1k of
  evaluation noise regardless).
- **Stage 2.** EMA: stop if neither `frozen` nor `ema` moves the pooled realised gain by 20%. KL: stop if the
  3M gain is gone by 0.5M after the return to 0.03. Vine: above.
- **Stage 3.** Folded into wave 1.

**If Stage 1 succeeds and the plateau remains** (flip rate near zero, upright mean still ~11k):
1. Arm t tells which plateau it is. If t raises the slope, the remaining problem is tumble avoidance, which
   is a value-level error on rare rows (the critic over-values the inverted state 3.5x), not an action-slope
   problem: go to B1 (section 8). If t does not, the ceiling is critic information: Stage 2 and B3.
2. Take the budget-priced entropy target for the ~1k of late evaluation noise.
3. Do not start another round of critic-loss or target variants. D1 already says they cannot help.

What is missing from the plan (M12):

- **A PPO-lineage reference under the same protocol.** The 8M comparison today is 11.0k (conditional on
  upright, one init) against 8.4k (plain) or 10.2-10.4k (the maximum of ~1000 single-seed runs). Run the
  sepclip control and the best PPO-lineage variant on 8 streams each. They cost ~100 s solo per run.
- **From-scratch confirmation of every fork positive**, on streams it has not used, before it enters the
  integrated file.
- **Hopper / Walker2d earlier.** The flip diagnosis is specific to a task whose episodes cannot end. On the
  other two a fall terminates the episode, so the same initial noise produces short episodes instead of a
  trap, and nothing in the record says the reference even learns there. A 4-stream probe per env in wave 1
  costs ~30 box-minutes and prevents three stages of HalfCheetah-specific tuning.
- **Deterministic and stochastic returns reported together** for every arm (section 3.3).
- **A pre-registered table** per experiment: primary metric, window, effect size, n, decision. The plan has
  predictions; it does not say which number at which step decides.

---

## 8. Bigger picture

The plan is too incremental in one specific sense: Stages 1 and 2 are mostly knob moves on the existing
design (an init, a dual rate, an EMA, a KL window), and the one structural item (vine) is the one whose
measured upside is smallest relative to its cost and whose result cannot be the headline. The diagnosis in
Part 1 says the critic is at the ceiling of its channel and roughly level with PPO's estimator in parameter
space. If that is right, a large jump comes from one of: removing the flip (worth ~4k in expectation, and
the plan does address it), adding an independent estimator, measuring the value of noise instead of fixing
it, or making many-action updates cheap. Three bets the evidence supports, each with an inexpensive gate:

**B1. Hybrid actor: unit-norm pathwise + score-function gradient.**
- Evidence: 07 section D puts the critic's and GAE's aggregate cosine with the true projected gradient both
  near 0.65 at 4M (product 0.40-0.45), with errors from different sources (critic error is correlated
  across iterations; score-function noise is independent across iterations). The score function is the
  better of the two at 30M, which is where PPO overtakes REPPO. The control-variate form buys nothing
  (cv_combo ~ sf_gae), and the only prior mix (v40) used unnormalised advantages and destabilised the
  restart, so it was not a test [DOC 10].
- Design: each actor step, compute both parameter gradients on the same minibatch, normalise each to unit
  norm, average, then apply the existing 0.5 clip. Both are already clipped on essentially every step, so
  unit-norm mixing changes direction only. PPO-style clipped ratio on standardised advantages for the
  score-function half.
- Predicted effect: two estimators with cosine 0.65 and independent errors average to
  0.65 / sqrt(0.65^2 + (1 - 0.65^2) / 2) = 0.77 [CALC]: +18% gain per step at 4-8M, more late, and an
  unbiased component that tumble rows cannot dominate through dQ/da. Roughly +0.5-1k at 8M, larger at 16M+.
- Risk: the errors are not independent (both read the same returns); the score-function half slows the
  pathwise head start before 0.5M (mix weight from 0 at the start, tied to measured reliability rather than
  a schedule).
- Gate, CPU only, existing data: add the unit-norm average as one more estimator in `07_D_headtohead.py`
  and read its gain at 4M / 8M / 30M. If it is not above both parents, drop the bet.

**B2. Mixed-noise behaviour with a noise-conditioned critic, replacing the entropy constant.**
- Evidence: the mean policy at 0.5M is upright while its own noise tips it in 6 of 6 episodes; half noise
  cuts late tumbles from 6/48 to 1/48 on a frozen policy; the population runs (v45 / v46) are 0 of 4 flipped
  with +1.1k at 1M; every constant-based noise rule has an interior optimum nobody can locate from one
  trajectory [DOC 09, 10].
- Design: each env draws a noise scale c from a small fixed set (say 0.5, 1, 1.5 times the head's sigma)
  per episode; the critic takes c as an input and is evaluated on-policy for the mixture; the actor's mean
  is trained through all of them. The per-c returns measure the noise cost directly; the per-c realised
  gain (Stage 0 instrument, split by c) measures what each noise level contributes to learning. The mixture
  weights follow that measurement. No entropy target, no cap, no anneal.
- Predicted effect: flip rate near zero without an init change (the batch always contains near-mean-policy
  rows); late stochastic return up by the low-noise envs' share of the ~1.4k noise cost; and the first
  direct measurement of the quantity the whole entropy debate is about.
- Risk: each noise level gets a fraction of the data; the conditioned critic is a harder fit; the reported
  training return is a mixture and must be labelled as such, with the c = 1 and deterministic returns given
  separately.
- Gate: the E1.1 census with a two-level version (a quarter of envs at half noise, unconditioned critic),
  which is 09's own untested suggestion and costs a few lines.

**B3. Late-fusion critic with K-action updates.**
- Evidence: 06 section 4.6. With the action entering at the first layer, every extra action costs a full
  critic forward, so the actor uses one action per state and the bootstrap four. A state trunk evaluated
  once per row with the action entering a small head makes extra actions 3.6x cheaper and lets the CE step,
  the actor step and the bootstrap share the trunk. It is also the largest remaining compute lever after
  the aux thinning.
- Design: phi(s) trunk, [phi, a] head to the 151 bins; actor step on K = 8 reparameterised actions per
  state (antithetic); bootstrap on the same K.
- Predicted effect: update time down a further 30-40% on the lean base; within-state actor-gradient variance
  down ~K-fold at no data cost. Return effect unknown in sign: 06's offline variance split decides whether
  within-state variance is a material share.
- Risk: it changes the critic's function class exactly where it is weakest; the dueling critic (a different
  way of separating state and action) was the worst removal in the archive (-6 to -7k).
- Gate: 06's offline "actor gradient variance decomposition" on the existing checkpoints. If the within-
  state share is small, this is a compute item only.

Not one of the three, but worth a line: a short on-policy-corrected replay window is the standard route to
more information per env step and has the only replay-positive in the record (v11, grade B-). It flips more
often when active before 0.5M [DOC 09], and whether it counts as "fully on-policy" is the user's call.

---

## 9. Required changes, in priority order

### MUST

**M1. Restate the Stage 0 parity gate.** Reason and spec: section 3.1 (rules 1-6). Keep the two diagnostic
`refresh_targets` calls on every iteration; thin only `diagnostic_generator` consumers; instruments eager
with their own generator; no edits inside traced functions at default flags; run unmodified v60 alongside;
parity follower on untouched tags.

**M2. Posture by `cos(obs[1]) < 0`.** Reason: unwrapped pitch. Spec: `posture/inverted_frac` = share of
rollout rows with `cos(pitch) < 0`; `posture/tumbles_per_episode` = sign changes of `cos(pitch)` per
episode; row-split critic metrics use the same mask. Re-check 09's late crash shares with it.

**M3. Realised-gain instrument.** Reason: section 3.2. Spec: backward form primary; in-sample forward form
and their difference logged as the memorisation meter; explicit `pre_critic` copy at insert; behaviour-time
`(mean, log_std)` and normalised observation added to the transfer fields; mean / scale / T dH split;
posture split; winsorised and raw; lane-jackknife SE; fixed-n = 10 variant; slope-persistence ring of 8
critics.

**M4. Fork protocol.** Reason: section 1.1. Spec: 4 upright parents x 2 streams per arm, blocked analysis;
fork noise floor measured first at both fork points and used to set n; effects reported with intervals.

**M5. Census protocol.** Reason: section 1.2. Spec: primary endpoint inverted share over 0-0.5M; trapped-at-
1M secondary; "below 0.1" needs 0-1 of 32; control census never culled; treatment culls scored at the
trapped plateau; flip count and upright-conditional mean always separate.

**M6. Rewrite E1.2.** Reason: section 4.2. Spec: arms ctl, t (truncation-on-inversion diagnostic), a
(temperature-only dual step / 10), b (budget-priced target); predictions as tabulated there; drop the
pre-squash arm and the v31 cap.

**M7. Separate optimizer groups for the two duals.** Reason: `v60:395-401`. Spec: new flag
`--temperature-step-per-iteration` (default = `--dual-step-per-iteration`); default path must stay bit-
identical to v64 stream 0.

**M8. Rewrite E1.1.** Reason: section 4.1. Spec: arms ctl, a1, a2, a3, a4 as tabulated; Beta conditional;
three-part pass rule including 8M non-inferiority.

**M9. Rewrite E2.1.** Reason: section 5.1. Spec: gate on slope persistence; arms live / frozen / ema at two
bracketing horizons <= 8; pooled-window instrument readout; K-snapshot confirmation of a positive.

**M10. Rewrite E2.3.** Reason: section 5.3. Spec: total-information gate with ceiling; budget-free upper
bound first; antithetic K = 2 with the real action as one branch; common tail noise; two controls; correct
held-out ceilings; reporting rules for the simulator-restore assumption.

**M11. Compute decision in wave 1.** Reason: section 6. Spec: lean-safe and lean-full as 16-stream census
arms; adopt the cheapest non-inferior base; timings from the 8M arms, no separate timing runs.

**M12. Add the missing references.** PPO under `--rng-stream` (new versioned file), 8 streams x 2 variants;
from-scratch confirmation of fork positives on unused streams; Hopper / Walker2d 4-stream probes in wave 1;
4-episode deterministic eval and `fork_eval.py` on final checkpoints; a pre-registration table per
experiment.

**M13. Ask the user about `--rng-stream`.** Reason: section 2 point 4. One question, with the power numbers
of section 1 attached. Without a yes, nothing in this plan is interpretable and nothing should be queued
beyond wave 0a.

### SHOULD

- **S1. E0.0** (`scripts/13_random_posture.py`, CPU, through `mlq` when the queue allows) before wave 1.
- **S2. B1's offline gate** (unit-norm average added to 07's head-to-head) in wave 0a. It is the cheapest
  test in this document of something that could be large.
- **S3. 06's two offline measurements** (actor-gradient variance split; precision sensitivity by path) in
  wave 0a; they decide B3 and the BF16 item.
- **S4. B2's two-level gate** (a quarter of envs at half noise) as a sixth E1.1 arm if the queue allows.
- **S5. Read E2.2's bounds off the Stage 0 ratio** before fixing 0.06 / 0.1; log mean and scale parts of the
  realised KL.
- **S6. Header honesty in every new file**: declared deviations (fixed reward scale instead of
  `VectorRewardNorm`; strict FP32), the stream protocol, and for vine the simulator-restore assumption.

### CONSIDER

- **C1.** GPU acting at 128-256 envs if the 256-wide actor fails in `lean-full`.
- **C2.** A split-half critic arm (08's I4) as a separate, non-parity diagnostic run: the only simulator-free
  reliability measure, but it changes learning.
- **C3.** E2.4 only as a post-1M fork and only if the oracle shows n = 5 labels are mostly reward.
- **C4.** The on-policy-corrected replay window, if the user accepts it as on-policy.

---

## 10. Recommended final ordering

1. **Wave 0a** (2 GPU runs + CPU jobs, < 15 box-minutes): user decision on streams; v64 with rules 1-6;
   parity pair; E0.0; B1 offline gate; 06 offline pair; vine oracle completion.
2. **Wave 1** (~136 runs, 4.5-6.5 box-hours): control census x16; lean-safe x16; lean-full x16; E1.1 a1-a4
   x16 each; PPO reference 2 x 8; Hopper and Walker2d probes 2 x 4. Outcome: the honest baseline, the base
   recipe, the flip fix, the PPO yardstick, and parent checkpoints.
3. **Wave 2** (~80 forks, 1.7-3.5 box-hours): fork noise floor at 1M and 4M; E1.2 t / a / b; E2.1 if not
   gated out; E2.2. Outcome: which plateau it is, and whether averaging or step size is real.
4. **Wave 3** (32-56 runs, 1.4-2.5 box-hours): vine upper bound, then the budgeted version only on a pass;
   in parallel the first from-scratch runs of whichever of B1 / B2 / B3 cleared its gate (16 streams each).
5. **Wave 4** (~48 runs): integrated file; 16 fresh HalfCheetah streams; Hopper and Walker2d at 8 streams;
   PPO reference alongside; deterministic and stochastic, flip count and conditional mean, vine reported
   separately if present.

## 11. Limits of this review

- Nothing was run on the GPU or in the simulator. E0.0 is written but unexecuted.
- The sd behind every MDE is the archive's n = 3; my power tables inherit it. The fork noise floor in wave 2
  replaces it.
- I read 06, 08, 09 and 10 in the sections bearing on the plan and 01 / 07 in part; 02-05 were not opened.
  Numbers tagged [DOC] are theirs.
- The information arithmetic for vine uses 07's 1M figures and a square-root shrinkage rule of thumb; it
  bounds the prediction, it does not replace the oracle completion.
- The claim that the Oct 2 GAE kernel edit may break bit-reproduction is from reading the diff, not from a
  run. The scalar path looks unchanged.
- The cos-vs-abs posture point is from the definition of the observation and 09's description of full
  revolutions; I did not recount 09's crash shares.
- B1-B3 effect sizes are back-of-envelope. Each has a gate so that the estimate does not have to be trusted.
