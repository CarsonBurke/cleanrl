# 09 — The flip and the 3-4M regime change

Role: forensic analyst. Existing artifacts only: tfevents of every `runs/HalfCheetah-v4__reppo_*` run (scalar
dumps in `data/03_runs`), observation statistics inside saved snapshots, and about 45 s in total of CPU
posture rollouts on existing snapshots. No training, no GPU, no edits to existing files. Everything is seed 1.

Tags: **[OBS]** read or computed from logs / snapshots / rollouts in this session. **[INF]** my inference.
Intervals are 95% Wilson; p-values are two-sided Fisher exact. Returns are raw episodic returns.

Scripts (`benchmarks/reppo_rootcause/scripts/`), outputs in `data/09/`:

| Script | Output | What |
|---|---|---|
| `09_common.py` | - | loaders, Wilson interval |
| `09_census_table.py` | `census_raw.tsv` | one row per run: trajectory, fingerprint, hyperparameters |
| `09_census.py` | `census.tsv`, `census.txt` | posture class per distinct trajectory, feature cross-tabs, Fisher tests |
| `09_obsnorm_posture.py` | `obsnorm_posture.txt` | posture from saved observation statistics (no simulation) |
| `09_posture_sim.py` | `posture_sim.jsonl`, `posture_sim_v60base.jsonl` | per-episode posture rollouts, optional noise scale |
| `09_early_pairs.py` | `early_pairs_iter1-30.txt` | per-iteration scalars of matched trajectories |
| `09_early_cohort.py` | `early_cohort.txt` | class medians and AUC of early scalars, trapped vs upright |
| `09_regime_trace.py` | (stdout) | fine-binned trace of one run incl. crash share |
| `09_regime_cohort.py` | `regime_cohort.tsv/.txt` | regime metrics at return levels 6k / 8k / 10k / 11k, all runs |
| `09_regime_alignment.py` | `regime_alignment.txt` | step-vs-return alignment of the change, pooled return bands |
| (inline) | `regime_noise_arms.txt` | entropy forks, price cap, anneal, no-entropy, IDS arms |

---

## 0. Findings that change the picture

1. **Both events are the same physical event.** The cheetah tips forward (half a forward somersault, root
   pitch +3.6 rad) and lands on its back. HalfCheetah never terminates, so the inverted state is absorbing for
   the rest of the 1000-step episode. Early, the policy also owns a back-running gait (1.4-2.2 m/s) and the run
   can lock in (Event 1). Late, it has no back gait (0.2-0.4 m/s), so each tumble is a crashed episode that
   floods the batch with off-distribution states (Event 2). [OBS: rollouts, sections 1.4 and 2.3]
2. **Upright runs were inverted first.** Saved observation statistics show that runs which end upright spent
   8-17% (reference path B), 50-55% (v43 TF32), 64-68% (v36 e128), 76-80% (v43 0-nats, v44) of their first
   0.5-1.0M steps upside down. Trapped runs: 91-99%. The outcome is not "does it flip" but "does it get back
   up, and when". [OBS]
3. **At 0.5M the mean policy is upright while the sampled policy is on its back.** v36 e128 @505k:
   deterministic 2.66k, upright, 2.9 m/s; with its own exploration noise 6 of 6 episodes tip over within
   49-268 steps and finish on the back at 1.35-1.41 m/s. Exploration noise is the proximate trigger of each
   inversion, and the training data are the sampled policy's. [OBS]
4. **The flip is confounded with the actor distribution, not explained by pathwise-vs-score-function.** Every
   PPO-lineage trainer in this repo is a Beta policy (`ppo_continuous_action.py` l.1-2, 237). REPPO runs with a
   bounded Beta / clipped-Gaussian head and the same pathwise Q-climbing actor: 0 trapped of 12. REPPO
   tanh-Gaussian: 27 trapped of 123. The tanh-Gaussian head starts at pre-tanh std 1.08 = action sd 0.65 with
   17% of actions beyond |a| 0.9; Beta and clipped-Gaussian heads start at action sd 0.48-0.49 with 2.6%. [OBS]
5. **The 3-4M change is tied to return level (gait speed), not step count.** In five runs that cross ~7k at
   about 3.5M, 4.5M, 4.5M, 5.5M and 7.5M the same signature appears at the crossing: crash share rising to
   2-7%, q_mse x1.7-5, actor gradient norm x1.5-6, KL multiplier x1.3-4. [OBS]
6. **Lowering noise late does not lower the crash rate.** The 8M entropy forks drop pre-tanh std from 0.52 to
   0.44 / 0.38 / 0.30 and keep a 3-5% crash share, with *worse* critic metrics, because the policy re-adapts
   (sampled-action saturation 3% -> 7-16%). On a *frozen* 7.9M policy, half noise cuts tumbles from 6/48 to
   1/48. The crash rate is a fixed point of the learner; noise sets the speed at which it is reached. [OBS]

---

# Event 1 — the flip

## 1.1 (a) Census

Unit: a **distinct from-scratch trajectory**. Runs that are bit-identical for their first 300k steps count
once (the outcome is set before that; section 1.3), forks and resumes are excluded. 193 runs with logs ->
144 distinct trajectories -> 135 classifiable (9 are shorter than 1M or never learn).

Classes, from the per-episode return record (`09_census.py`):

| Class | Rule | n |
|---|---|---:|
| F trapped | rolling 48-episode median never exceeds 2800; ends at median 1.4-2.7k with IQR < 120 | 27 |
| R recovered | at least one 100k window from 0.6M on with median 1.0-2.3k and IQR < 120 (the back-running signature), later exceeds 2800 | 10 |
| M mixed at the end | left the curve, but >= 10% of the last 200 episodes sit at 1.2-2.8k while the p90 episode is > 3.5k | 16 |
| U upright | none of the above | 82 |

Why the IQR rule works [OBS]: a back-running policy produces tightly clustered episodes on a slow saturating
curve (median 1000 / 1270 / 1490 / 1630 / 1725 / 1800 / 1870 / 1880 at 0.3 ... 1.0M, IQR 40-90, identical to
within 5% in v61 control, v43 FP32, v13 r1, v25 lam 0.8). Upright runs crossing the same return band grow by
~1000 per 100k and have IQR 150-500.

Validation against posture [OBS, `09_obsnorm_posture.py`]: all four F-labelled runs that have a snapshot (v37,
v37 FP32, v38 FP32, v39) have cumulative mean root pitch 2.96-3.16 rad at 1M (91-99% of all steps inverted);
U-labelled runs have 0.1-1.65. v25 lam 0.8 (R; the v29 parent arm is the same trajectory) reads 27% of its
first 8M steps inverted from the 8M snapshot, against 1.6M / 8M = 20% on the curve plus a 10% tail of inverted
episodes until 2.5M+ from the return record: the two independent readings agree.

Trapped (27): v1 ppobatch lr1e-3, v4 lr1.2e-3, v5 (3 arms), v6 default, v7 cw1024, v7 lam0.8, v11 retrace
buf1, v12, v13 r1, v25 lam0.5, v28, v32, v33, v35 (2), v37, v37 FP32, v38 FP32, v39, v43 FP32, v50 (2),
v55 ids, v61 control, v61 l08. Six more than the ledger's 21 (v1 lr1e-3, v6 default, v33, v50 x2; v38 FP32
is inverted at a lower plateau, 1.6k).

Recovered (10), with the step at which the rolling median first exceeds 2800: v11 buf4 0.91M, v52 q 1.15M,
v63 1.23M, v11 buf8 1.24M, v16 1.31M, v26 lam0.95 1.37M, v9 ns256 1.72M, **v25 lam0.8 1.80M** (-> 13.7k at
50M), v4 ep10 2.89M, v1 ep4_mb1 4.17M. "Irreversible" is wrong: 10 of the 37 trajectories that sat on the
curve at >= 0.6M left it.

### Flip rate against design features

| Feature | Value | n | F | R | M | U | P(F) [95%] | P(F+R+M) [95%] |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Actor distribution** | tanh-Gaussian | 123 | 27 | 10 | 16 | 70 | 0.22 [0.16, 0.30] | 0.43 [0.35, 0.52] |
| | Beta (v3, v18-21, v47, v48, v57) | 11 | 0 | 0 | 0 | 11 | 0.00 [0.00, 0.26] | 0.00 [0.00, 0.26] |
| | clipped Gaussian (v22) | 1 | 0 | 0 | 0 | 1 | 0 [0, 0.79] | |
| | PPO lineage (all Beta), ledger count | 111 | 0 | | | | not independent draws | |
| *tanh only below* | | | | | | | | |
| **Target horizon** | 1-step TD (v5) | 4 | 3 | 0 | 1 | 0 | 0.75 [0.30, 0.95] | 1.00 |
| | lambda 0.5 | 1 | 1 | 0 | 0 | 0 | 1.00 [0.21, 1] | |
| | lambda 0.8 | 54 | 15 | 3 | 6 | 30 | 0.28 [0.18, 0.41] | 0.44 [0.32, 0.58] |
| | lambda 0.95 | 60 | 7 | 7 | 8 | 38 | 0.12 [0.06, 0.22] | 0.37 [0.26, 0.49] |
| | lambda 0.99 | 1 | 0 | 0 | 1 | 0 | | |
| **Replay / window / model rows** | no | 79 | 13 | 7 | 8 | 51 | 0.16 [0.10, 0.26] | 0.35 [0.26, 0.46] |
| | yes | 44 | 14 | 3 | 8 | 19 | 0.32 [0.20, 0.47] | 0.57 [0.42, 0.70] |
| **Precision** | FP32 | 116 | 25 | 10 | 15 | 66 | 0.22 [0.15, 0.30] | 0.43 |
| | TF32 | 7 | 2 | 0 | 1 | 4 | 0.29 [0.08, 0.64] | 0.43 |
| **Aux horizon** | 1 | 71 | 13 | 8 | 10 | 40 | 0.18 [0.11, 0.29] | |
| | 8 | 49 | 14 | 2 | 6 | 27 | 0.29 [0.18, 0.42] | |
| **Critic** | HL-Gauss Q | 111 | 24 | 8 | 16 | 63 | 0.22 [0.15, 0.30] | |
| | distributional (v12-17) | 6 | 2 | 1 | 0 | 3 | 0.33 [0.10, 0.70] | |
| | scalar (v14, v59) | 3 | 0 | 0 | 0 | 3 | 0 [0, 0.56] | |
| **KL bound** | 0.03 | 112 | 26 | 10 | | | 0.23 [0.16, 0.32] | |
| | 0.01-0.015 / 0.1-0.3 | 2 / 5 | 0 / 0 | | | | 0 [0, 0.66] / 0 [0, 0.43] | |
| | 0.0075 (v35 window) | 1 | 1 | | | | n = 1 | |
| **Entropy target** | -0.5/dim | 114 | 27 | 10 | | | 0.24 [0.17, 0.32] | |
| | other (0, -0.25, -1, -1.5) | 5 | 0 | | | | 0 [0, 0.43] | |
| **Temperature source** | dual | 120 | 26 | | | | 0.22 [0.15, 0.30] | |
| | ids / calibrated / drift | 1 / 1 / 1 | 1 / 0 / 0 | | | | n = 1 each | |
| **Rollout shape** | 16 x 1024 | 105 | 25 | 9 | | | 0.24 [0.17, 0.33] | |
| | 16 x 4096 | 3 | 0 | | | | 0 [0, 0.56] | |
| **Actor heads** | 1 | 119 | 27 | | | | 0.23 [0.16, 0.31] | |
| | 4 / 8 / 16 (v45, v46) | 4 | 0 | 0 | 0 | 4 | 0.00 [0.00, 0.49] | |
| **Era** | v7-v17 / v23-v31 | 38 / 19 | 5 / 2 | | | | 0.13 / 0.11 | |
| | v32-v44 | 18 | 9 | | | | 0.50 [0.29, 0.71] | |
| | v45-v63 | 25 | 5 | | | | 0.20 [0.09, 0.39] | |
| **Reference algorithm only** (ledger) | paths A, B, C, D | 4 | 2 | | | | 0.50 [0.15, 0.85] | |
| Reference-like (tanh, v27+, lambda 0.8, dual -0.5, H8, no replay, 1 head) | | 23 | 6 | 2 | 4 | 11 | 0.26 [0.13, 0.47] | 0.52 |

Fisher exact [OBS]:

| Contrast | Counts | p |
|---|---|---:|
| tanh vs Beta + clipped, trapped | 27/123 vs 0/12 | 0.12 |
| tanh vs Beta + clipped, ever on the curve or mixed (F+R+M) | 53/123 vs 0/12 | **0.003** |
| lambda >= 0.95 vs lambda <= 0.8 or 1-step, trapped | 7/61 vs 19/59 | **0.008** |
| same, trapped or recovered | 14/61 vs 22/59 | 0.11 |
| replay vs none, trapped (all tanh) | 14/44 vs 13/79 | 0.07 |
| replay vs none, trapped (tanh, v27+, 16 x 1024, 1 head) | 5/7 vs 8/36 | 0.02 |
| FP32 vs TF32 | 25/116 vs 2/7 | 0.65 |
| aux horizon 1 vs 8 | 13/71 vs 14/49 | 0.27 |

Reading, with the honest caveats:

- **Actor distribution** is the only feature with zero events on one side, and it is the feature that also
  separates the whole PPO lineage from REPPO. n = 12 gives an upper bound of 0.24, so "Beta never flips" is not
  established, but "Beta is at most as bad as tanh and probably much better" is (p 0.003 on the wider
  outcome). Confound: the Beta variants are seven designs, not one design repeated.
- **Target horizon** has a dose-response in the expected direction (1-step 3/4, lambda 0.5 1/1, 0.8 0.28,
  0.95 0.12), but lambda is confounded with era (0.95 is v1-v17, 0.8 is v25+). Within-family pairs are the
  fairer evidence: v7 (0.8 F, 0.95 U, 0.99 M), v25 (0.5 F, 0.8 R at 1.8M, 0.95 U), v61 (0.8 F, 0.8-control F,
  0.95 U), v26 (0.8 U, 0.95 R). Three of four families order as predicted; the fourth reverses. Suggestive.
- **Replay / window / model-row variants** flip more, most clearly in the modern cohort (5/7). They also are
  the designs with the shortest effective target horizon (v32 1-step, v33 mix, v35 window, v37 model
  branches), so this is partly the same lever.
- **Precision, aux horizon, critic type, KL bound, entropy target, temperature source, rollout shape:** no
  detectable effect at the available n. TF32 2/7 vs FP32 25/116 does not support "TF32 causes flips".
  Capacity (cw1024, 1/1) and the IDS temperature (1/1) are single trajectories.
- The v32-v44 era (9/18) is where FAMILY.md's "5 of the last 6 flipped" impression comes from; it is the era of
  replay / model variants on lambda 0.8.

## 1.2 Posture evidence that did not exist before

`VectorObsNorm` keeps a cumulative per-env running mean and variance of raw observations, saved in every
snapshot; obs[1] is the root pitch. A two-point mixture (upright pitch u, inverted 3.2) gives the share of all
steps so far spent inverted, f = (mean - u) / (3.2 - u); u = 0.1 is a late gait, u = 0.4 the early forward-leaning
gait (measured 0.3-0.4 in the 0.5M rollouts). The mixture's predicted sd matches the stored sd in every row
(e.g. v43: predicted 1.54, stored 1.51), so these are mixtures, not a tilted gait. All 16 envs agree to +-0.1.

| Run (end state) | snapshot | cumulative pitch mean / sd | inverted share of all steps so far (u 0.4 .. 0.1) | next interval mean pitch |
|---|---:|---|---:|---:|
| v57 actiongeom, **Beta** (upright) | 0.26M | 0.12 / 0.29 | 0.00-0.01 | 0.07 |
| v60 base = path B (upright, 11.0k) | 1.02M | 0.63 / 0.99 | **0.08-0.17** | 0.00 (to 4M) |
| v43 -0.25/dim TF32 (upright) | 0.51M | 1.17 / 1.29 | 0.28-0.35 | 0.47 |
| v43 TF32 (upright) | 0.51M | 1.79 / 1.51 | **0.50-0.55** | 0.23 |
| v36 e128 TF32 (upright) | 0.50M | 2.19 / 1.49 | 0.64-0.67 | 1.12 |
| v43 0-nats TF32 (upright) | 0.51M | 2.54 / 1.42 | 0.76-0.79 | 0.26 |
| v44 sharpmean FP32 (upright) | 0.51M | 2.57 / 1.35 | **0.78-0.80** | 0.28 |
| v38 TF32 (M, half inverted) | 0.51M | 1.49 / 1.77 | 0.39-0.45 | 1.25 |
| v39 (trapped) | 0.51M | 2.71 / 1.39 | 0.83-0.84 | 3.21 |
| v38 FP32, v37, v37 FP32 (trapped) | 1.02M | 3.09-3.16 | 0.96-0.99 | - |

[OBS] the table. [INF] Trapped runs at 96-99% of the first 1M means the cheetah was already on its back within
the first 2-4 iterations (30-65k steps), while the policy is still near its initialisation (pre-tanh std
1.0-1.08, return -370 to -250, Q ~ 0). Path B's 8-17% is 80-170k steps, i.e. roughly its first 5-10
iterations. The simplest reading consistent with all rows: **every tanh-Gaussian run starts inverted-dominated
under its initial exploration noise, and "upright" is an escape** that happens at iteration ~5-10 (A, B),
~0.25-0.4M (v43, v44), 0.9-4M (the R class), or never (F). The Beta run was never inverted.

## 1.3 (b) What differs before the outcomes diverge

Matched trajectories of the reference algorithm (`early_pairs_iter1-30.txt`): B = v60 base and A = v27 H8
(upright); C = v61 control and D = v43 FP32 (trapped). One iteration = 16,384 steps.

| iteration | 3 | 5 | 6 | 8 | 10 | 11 | 13 | 16 | 20 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median return, B / A | -271 / -267 | -232 / -242 | -177 / -194 | -75 / -114 | 22 / 46 | 137 / 195 | 412 / 475 | 789 / 983 | 1409 / 1454 | 2548 / 2502 |
| median return, C / D | -294 / -271 | -251 / -225 | -211 / -183 | -127 / -101 | 13 / 14 | 88 / 60 | 283 / 240 | 555 / 470 | 880 / 781 | 1354 / 1319 |
| return IQR, B / A | 69 / 90 | 57 / 48 | 63 / 86 | 77 / 65 | 109 / 134 | 121 / 115 | 78 / 100 | 434 / 94 | 495 / 242 | - |
| return IQR, C / D | 66 / 95 | 56 / 68 | 37 / 58 | 62 / 57 | 28 / 65 | 64 / 43 | 72 / 48 | 106 / 66 | 58 / 45 | - |
| aux_loss, B / A | .033 / .034 | .034 / .029 | .032 / .030 | .039 / .033 | .034 / .038 | .037 / .040 | .042 / .041 | .046 / .045 | .051 / .050 | - |
| aux_loss, C / D | .029 / .029 | .022 / .026 | .022 / .024 | .023 / .023 | .020 / .023 | .020 / .023 | .021 / .023 | .022 / .024 | .022 / .027 | - |
| q_mse, B / A | .058 / .057 | .058 / .036 | .051 / .036 | .058 / .046 | .048 / .063 | .057 / .065 | .083 / .087 | .108 / .112 | .140 / .130 | - |
| q_mse, C / D | .045 / .042 | .029 / .037 | .028 / .034 | .039 / .032 | .028 / .029 | .030 / .030 | .037 / .033 | .036 / .037 | .031 / .036 | - |
| dqda_norm, B / A | .063 / .056 | .080 / .059 | .076 / .072 | .107 / .085 | .116 / .110 | .139 / .139 | .161 / .159 | .173 / .172 | .206 / .198 | - |
| dqda_norm, C / D | .050 / .048 | .060 / .061 | .063 / .069 | .070 / .070 | .070 / .071 | .069 / .077 | .092 / .089 | .080 / .090 | .102 / .121 | - |
| pre-tanh std, B / A | 1.03 / 1.02 | 1.00 / 1.02 | .98 / 1.00 | .96 / .93 | .88 / .88 | .85 / .86 | .82 / .82 | .76 / .75 | .68 / .66 | - |
| pre-tanh std, C / D | 1.01 / 1.00 | .98 / .97 | .94 / .94 | .89 / .91 | .83 / .86 | .81 / .83 | .76 / .78 | .67 / .70 | .58 / .60 | - |
| mean \|tanh(mu)\|, B / A | .245 / .243 | .243 / .242 | .251 / .252 | .302 / .285 | .329 / .323 | .344 / .342 | .371 / .384 | .433 / .437 | .491 / .465 | - |
| mean \|tanh(mu)\|, C / D | .246 / .252 | .237 / .260 | .264 / .274 | .312 / .318 | .368 / .356 | .400 / .374 | .455 / .422 | .501 / .451 | .535 / .488 | - |
| realised KL / violation frac (all four) | .06-.09 / .78-.86 | .06-.07 / .78-.82 | .06-.07 / .74-.79 | .05-.06 / .70-.73 | .05-.06 / .70-.75 | | .04-.05 / .66-.72 | | .034-.041 / .52-.62 | |

[OBS]

- **Returns are indistinguishable through iteration 10 (160k steps) and separate gradually from iteration
  11-13 (180-210k).** There is no single rollout in which a trapped run drops or an upright run jumps. The
  first episodes are identical across runs (-369.9); v50 k3 shares A's first five episodes and still ends
  trapped.
- **Critic-side scalars separate earlier, at iterations 5-8 (80-130k steps):** `aux_loss` (latent
  next-feature loss), `q_mse` and `dqda_norm` stay low in trapped runs and rise in upright ones. In A and B
  `aux_loss` *rises* from iteration 7 on (0.030 -> 0.050) while in C and D it stays at 0.020-0.024.
- `aux_loss` behaves as a posture proxy [INF, checked against section 1.2]: across 12 reference-design
  trajectories, the 8 with `aux_loss` < 0.027 at iterations 8-12 are exactly the 6 trapped runs plus v43 TF32
  and v44 (upright at the end, but 50-80% inverted in their first 0.5M by the observation statistics); the 4
  with `aux_loss` > 0.030 (A, B, v61 l095, v29 no-reward) all stay upright. Lying on the back is a simpler
  dynamical regime to predict.
- Trapped runs then show faster std decay (0.58-0.60 vs 0.66-0.68 at iteration 20), higher mean |tanh(mu)|
  from iteration 8, lower sampled-action saturation (1.2-2.1% vs 2.7-3.5%), a lower KL multiplier and KL-violation
  fraction, and lower Q. Temperature is identical (0.0080 -> 0.0018 in all four; the dual only decays here).
  Actor pre-clip gradient norm (0.03-0.13, below the 0.5 clip) and the held-out advantage correlation
  (0.05-0.19) do not separate.
- Across the wider cohort (14 F vs 26 U, tanh, v23+, `early_cohort.txt`): AUC(F > U) at iteration 8 / 12 / 20 is
  q_mse 0.11 / 0.03 / 0.05, dqda_norm 0.14 / 0.17 / 0.12, return IQR 0.31 / 0.21 / 0.04, std 0.20 / 0.15 / 0.16,
  saturation 0.21 / 0.19 / 0.20, aux_loss 0.27 / 0.23 / 0.25, **median return 0.50 / 0.43 / 0.31** (0.12 only
  by iteration 30). The q_mse AUC of 0.09-0.19 at iterations 2-4 is a design confound (the F cohort is rich in
  1-step / model-target variants with a different q_mse scale) and should not be read as an iteration-2 signal;
  within the reference design the first clean separation is iteration 5-8.

Answer to "gradual pull or discrete accident": **neither as stated.** Nothing in the logs shows a single
flipping rollout followed by lock-in, and nothing shows a critic steadily dragging an upright policy onto its
back. The record fits a third description: all tanh runs begin inverted-dominated; between iterations ~5 and
~10 some begin to keep the cheetah upright (critic-side proxies move first, returns follow 3-5 iterations
later); the rest improve the back gait instead. The branch is taken while Q is ~0.03-1.2, the critic is 4-8
iterations old, and the realised per-iteration KL is 2-3x the 0.03 bound, so it is sensitive to any
perturbation (hence 2 of 4). [INF from OBS]

Leading indicator, usable now: `aux_loss` at iterations 8-12 (0.13-0.2M steps) below ~0.027 = inverted-
dominated data; return IQR at iterations 9-16 below ~70. A direct one would be free: log the share of rollout
rows with |obs[1]| > pi/2.

## 1.4 Rollouts on existing snapshots [OBS, `posture_sim.jsonl`]

| Snapshot | Deterministic | Stochastic (own noise) |
|---|---|---|
| v39 @507k (trapped) | 1266, 1314; inverted from step 20-22; 1.5 m/s on the back | 600-842; inverted from step 39-249 |
| v37 FP32 @1.02M (trapped) | 1831, 1949; inverted from step 10-11; 2.1-2.2 m/s | 1680-1780; inverted from step 11-32 |
| v36 e128 @505k (ends upright) | 2654, 2662; **never inverted**; 2.9 m/s | **6 of 6 inverted**, from step 49-268; 2.0 m/s upright before, 1.35-1.41 m/s on the back; 1076-1324 |
| v43 0-nats @507k (ends upright) | 2010, 2258; never inverted | **5 of 10 inverted**, at step 83-901; upright 1.6-2.2 m/s, back 1.3-1.6 m/s |
| v60 base @1.02M (upright) | 4365, 4390 | 0 of 48 crash; one 2-step excursion past pi/2 |

- A trapped policy flips **deliberately**: its mean action is on its back by step 10-22 of every episode.
- A policy in transition is **bimodal by state**: it runs upright from upright states and on its back from
  inverted states, the upright gait is faster (2.0-2.9 vs 1.4 m/s) but is knocked over by the policy's own
  noise (pre-tanh std 0.50-0.58) within a few hundred steps, and the back gait is stable under the same noise.
  With a tumble at step ~150 on average, about 85% of the on-policy rows come from the slower gait.
- The R-class escapes look the same in the return record [OBS]: v25 lam 0.8 sits at 1.82-2.04k (IQR 40-70)
  until 1.5M; then the p90 episode is 2137 / 2610 / 3092 / 3401 / 3658 at 1.5 / 1.6 / 1.7 / 1.8 / 1.9M while the
  p10 episode stays at 1.97-2.25k until 1.8M and the minimum stays at 1.3-2.0k through 2.5M. A second mode
  appears and takes over in ~0.3M steps (about 20 iterations); a 10% inverted tail lasts another 0.6M+.
  v5 ns64 is the reverse: mixed for 5M steps (p10 2.2k, median 3.9-4.0k) and sliding back (median 3.2k at 7.5M).

## 1.5 (c) Why this actor and not PPO: candidate mechanisms against the data

First a correction to the question. "Pathwise Q-climbing flips where score-function PPO never does" compares
two things that differ in the action distribution as well: all PPO-lineage runs are Beta policies, and
pathwise REPPO with a Beta or clipped-Gaussian head is 0 for 12. The data cannot separate "pathwise" from
"tanh-Gaussian head as configured"; what they can say is below.

| Candidate | Verdict | What the logs / snapshots show |
|---|---|---|
| High initial exploration noise from the entropy target | **Supported as the trigger, but the source is the head's initialisation, not the entropy target** | [OBS] The dual only *lowers* temperature for the first 0.5M (0.008 -> 0.0015); squashed entropy starts at 3.2-3.4 nats, far above the -3.0 target. PPO-Beta entropy is *higher* than REPPO's from 0.2M on (2.5-3.5 / 1.4-3.0 / -0.1-1.6 nats at 0.3 / 0.5 / 1M in two baselines vs REPPO about 1.0 / -1 to -2 / -3). But the noise *shape* differs: tanh(N(0, 1.08)) has action sd 0.65 with 17% of samples beyond \|a\| 0.9 and 1.4% beyond 0.99 (logged 2.7%); Beta(1.69, 1.69) and v22's clipped Gaussian start at sd 0.48-0.49 with 2.6% beyond 0.9. Every non-flipping family starts at 0.48; the flipping family at 0.65. [OBS] own-noise rollouts tip an upright gait in 6/6 and 5/10 episodes at 0.5M where the mean action never does. [INF] trapped runs are 96-99% inverted over the first 1M, so they were on their back before the policy had moved. |
| No flip penalty, back-running a decent local optimum, greedy climber commits faster | **Supported** | [OBS] Back gait 1.4-2.2 m/s, stable under noise, absorbing within an episode; trapped mean policy flips itself by step 10-22. REPPO moves much faster than PPO in the critical window: 500-1700 return at 0.2-0.3M against PPO's 20-400; realised KL per iteration 0.06-0.10 for the first ~20 iterations. Fully on-policy data (one rollout, no replay) means a posture that owns 85% of the rows owns 85% of the actor's and critic's updates. |
| Short real-return horizon of the bootstrapped target (not in the brief) | **Suggestive** | [OBS] Census dose-response: 1-step 3/4, lambda 0.5 1/1, 0.8 0.28, 0.95 0.12 (p 0.008, era-confounded; 3 of 4 within-family pairs agree). At lambda 0.8 the target carries 1 / (1 - 0.99 x 0.8) = 4.8 steps of real reward; the rest is a critic that is ~0 everywhere for the first 5-8 iterations. A tumble pays forward root velocity immediately and costs gait speed for the next 900 steps. |
| KL gate masking | **Real early anomaly, not discriminating** | [OBS] The KL multiplier starts at 0.01 and needs ~25 iterations to reach 0.1. For the first 20 iterations the realised mean KL is 0.035-0.10 against the 0.03 bound and 52-86% of states end each iteration outside the gate, where they receive only a pull of weight 0.01-0.09. The trust region is not enforced exactly while the branch is taken. Trapped runs have slightly *lower* violation (AUC 0.2-0.3), an effect of their slower policy change. No run varies `kl_start` (0.01 in all 142). |
| Actor exploits an over-estimated Q at a rarely visited flipped state | **Not supported for the early branch; supported late** | [OBS] In the window where the branch is taken inverted states are 10-99% of the data, not rare. At 1M in an upright run the oracle's flipped state has critic Q 13.4 vs true 3.8, but upright states are at 36-44: over-estimated, yet not preferred. This is an Event-2 mechanism (section 2). |
| Unnormalised huge critic gradients at off-distribution states steer the shared policy | **Not supported early; supported late** | [OBS] Early dqda_norm is 0.05-0.2 and *lower* in trapped runs; the actor's pre-clip gradient norm is 0.03-0.13, below the clip. From 3M it is 2.5-4, see section 2. |
| Observation statistics captured by the flipped regime | **Present, not sufficient** | [OBS] In trapped runs the pitch statistics are mean 3.0-3.2, sd 0.8-1.2, so an upright state sits at -3 sigma (inside the +-10 clip). But 10 runs recovered with captured statistics, and upright runs carry a pitch sd of 1.0-1.6 for the first 1M with no visible harm. |

Most likely account (confidence ~60%, [INF]): the tanh-Gaussian head's initial action noise knocks the
untrained cheetah onto its back in nearly every episode; the first few critic iterations then see mostly
inverted states; a fast, short-horizon, loosely trust-regioned Q-climber improves whatever gait owns the
data. Whether a run also establishes the upright gait before the back gait is entrenched is decided at
iterations ~5-10 and is chaotic. A slower learner with a milder, bounded initial distribution (PPO-Beta,
REPPO-Beta) is not inverted at the start and never enters the competition. The alternative that the logs
cannot exclude (~40%): something specific to the pathwise update on a tanh head during iterations 5-10, with
the initial inversion incidental.

**The one-second check that decides between them** (not run: it does not use an existing snapshot): roll out
random actions tanh(N(0, 1.08^2)) and Beta(1.69, 1.69) for a few episodes and count inversions. If the first
tips over within an episode and the second does not, the account above is confirmed at its root.

## 1.6 (d) Protocol and interventions

**Protocol: flip rate as a first-class outcome.**

1. Log `posture/inverted_frac` = share of rollout rows with |raw obs[1]| > pi/2 each iteration. Free, exact,
   and removes every return-shape heuristic used above.
2. Replicate by perturbing only the exploration-noise stream (seed stays 1, so init, env and warmup are
   shared). The trainer is bit-deterministic, so any offset to that generator gives an independent trajectory;
   v60's checkpoint / fork machinery already restores full state, and a checkpoint at iteration 0 gives every
   arm the same start.
3. Stop each replicate at **0.75M steps** (46 iterations, 9.4% of an 8M run). At that point trapped runs are at
   1.7-1.8k with IQR < 80 and upright ones above 3.2k; the classes do not overlap in any of the 135 classified
   trajectories except the slow recoverers, which is what `inverted_frac` over 0.6-0.75M > 0.5 resolves.
   Report trapped-at-0.75M, and separately the inverted share of the first 0.3M (the continuous outcome, far
   more sensitive than the binary one).
4. Size: 16 replicates per arm cost 12M steps = 1.5 standard runs. Baseline is 0.26-0.50 (reference-like 6/23,
   reference 2/4). 0 of 16 bounds the rate below 0.19, 0 of 32 below 0.11; "about zero" needs 0 of ~60. Run
   the control arm first: it has never been measured with more than 4.
5. Any arm that passes then gets 3 full 8M replicates with flip count and upright-conditional score reported
   separately (as the ledger already recommends).

**Interventions, ranked by census support and by cost to early speed.**

| # | Intervention | Census support | Early-speed evidence | Risk |
|---|---|---|---|---|
| 1 | **Start the tanh-Gaussian head at Beta's action noise**: initialise the log-std bias for pre-tanh sigma ~0.55-0.6 (action sd ~0.45-0.48, ~1% beyond \|a\| 0.9). Fallback: the Beta head itself. | Beta + clipped 0/12 [0, 0.24] vs tanh 27/123; PPO-Beta lineage 0/111; F+R+M p 0.003 | Low-noise-start families are *faster*: 2.0-2.6k at 0.3M in 8 of 10 (v21, v57: 1.1-1.3k) and 4.4-6.4k at 1M, against the reference's 1.2-1.35k and 4.2k | The sigma-init variant is untested; the Beta variants end lower at 8M (7.0-9.2k vs 10.2-11k), which is why the init on the existing head is preferred over swapping the head |
| 2 | **lambda 0.95 for the first ~0.5M**, then the usual 0.8 | 7/61 vs 19/59, p 0.008; 3 of 4 within-family pairs | None lost: v61 l095 1287 / 2733 / 4337 at 0.3 / 0.5 / 1M vs B 1192 / 2662 / 4264 | Era confound; v26 reverses |
| 3 | **Population of K = 4 actor heads, 2048 rows per head, shared critic** (v45 mb2048) | 0/4 [0, 0.49]; FAMILY: none of 8+ heads inverted | +1.1-1.2k at 1M, 11.0k at 8M = reference | 4x actor compute; n = 4 is weak (0/4 has probability 0.25 at p = 0.29) |
| 4 | **Keep replay / windows / 1-step or model targets out of the first ~0.5M** | replay arms 14/44 vs 13/79 (p 0.07); modern cohort 5/7 vs 8/36 (p 0.02) | Free | Only relevant to variants that add them |

Hypothesis-driven, no census support yet: enforce the gate from iteration 1 (`kl_start` ~1 instead of 0.01;
realised early KL is 2-3x the bound) or a 3-5 iteration critic-only warm-up; give a quarter of the envs half
the exploration noise so the batch always contains the mean policy's posture (motivated by section 1.4).
Diagnostic upper bound, not a method: truncate training episodes on sustained inversion. It should drive the
flip rate to zero and remove Event 2's crashed rows, but it uses knowledge of the task.

---

# Event 2 — the 3-4M regime change

## 2.1 What actually happens at 3-4M [OBS, `09_regime_trace.py`]

Reference path A (v27 H8) in 0.25M bins. "Crash" = share of episodes below 0.85 x the bin median.

| M steps | median | min episode | crash | q_mse | target mean | adv corr | KL mult | actor grad | pre-tanh std | temp | det. eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | 5000 | 4480 | 0.000 | 0.09 | 46.7 | 0.138 | 0.20 | 0.54 | 0.392 | 0.0104 | 5480 |
| 2.00 | 5820 | 5410 | 0.000 | 0.13 | 54.5 | 0.139 | 0.18 | 0.67 | 0.423 | 0.0167 | 6320 |
| 2.25 | 6180 | 3500 | 0.016 | 0.23 | 57.0 | 0.134 | 0.26 | 0.96 | 0.441 | 0.0195 | 7170 |
| 2.75 | 7010 | 466 | 0.028 | 0.40 | 63.8 | 0.088 | 0.52 | 2.17 | 0.450 | 0.0225 | 7820 |
| 3.25 | 7890 | **-203** | 0.044 | 0.60 | 71.3 | 0.076 | 0.65 | 2.85 | 0.459 | 0.0236 | 8740 |
| 4.00 | 8830 | -275 | 0.060 | 0.89 | 79.8 | 0.067 | 0.82 | 3.26 | 0.467 | 0.0270 | 9770 |
| 4.25 | 9090 | -305 | 0.084 | 1.18 | 79.7 | 0.103 | 0.77 | 3.26 | 0.480 | 0.0355 | - |
| 5.25 | 9960 | -242 | 0.088 | 1.71 | 87.5 | 0.069 | 0.87 | 3.68 | 0.489 | 0.0385 | 11100 |

**Episodes become heavy-tailed exactly where the critic metrics break.** Before 2.25M no episode is below
0.85 x median and the worst episode is within 10% of the median. From 2.25M crashed episodes appear (3.5k,
then 466, then negative totals), and from 3.25M they are 2-9% of episodes for the rest of the run. Path B
(v60 base) shows the same with an earlier trickle: isolated catastrophic episodes already at 1.5-2.5M (-137,
390, 248; share 0.4-2.4%), then 3-9% from 3.0M; q_mse 0.35 -> 1.06 between 2.5M and 3.25M. The v62 forks
follow B.

A second, independent trace of the same thing [OBS, section 1.2]: the rms root pitch over an interval,
reconstructed from consecutive snapshots, is 0.55 rad over 1-4M and **1.18 rad over 4-7.9M** in v60 base, with
an interval mean of only 0.03 (v29 H8: 1.24 over 8-12M; v23: 0.95 over 4-12M). A gait cannot do that; a few
percent of steps at pitch 3.6-16 rad can.

## 2.2 (e) Step count or return level?

Runs that reach the ~7k band at different steps (1M bins, `regime_noise_arms.txt`):

| Run | Before (bin, median) | After (bin, median) | crash share | q_mse | actor grad | KL mult |
|---|---|---|---|---|---|---|
| v27 H8 (reference) | 2-3M, 6.3k | 3-4M, 7.9k | 0.02 -> 0.03 | 0.25 -> 0.57 | 1.27 -> 3.28 | 0.33 -> 0.74 |
| v43 s50 (TF32, -0.5/dim until 4M) | 3-4M, 5.76k | 4-5M, 7.63k | 0.003 -> 0.017 | 0.13 -> 0.62 | 0.87 -> 5.5 | 0.23 -> 1.10 |
| v43 f1p0 (to -1.0/dim, std 0.38) | 4-5M, 6.25k | 5-6M, 7.73k | 0.002 -> 0.019 | 0.12 -> 0.64 | 1.24 -> 6.8 | 0.30 -> 1.33 |
| v43 0-nats (std 0.61-0.64) | 3-4M, 6.38k | 4-5M, 7.65k | 0.038 -> 0.068 | 0.44 -> 0.74 | 1.05 -> 1.55 | 0.35 -> 0.46 |
| v44 const (std 0.35) | 6-7M, 6.42k | 7-8M, 7.21k | 0.015 -> 0.040 | 0.18 -> 0.56 | 1.39 -> 2.97 | 0.32 -> 0.65 |
| v43 (to -2.0/dim, std 0.30 -> 0.21) | never passes 6.9k | - | 0.001-0.003 throughout | 0.09-0.12 throughout | 0.9 -> 7.4 (rises anyway) | 0.24 -> 1.31 |

Across the 27 from-scratch tanh runs that reach 9k (`regime_alignment.txt`), the step at which each marker
first crosses a fixed threshold varies about twice as much as the return at that moment:

| Marker | n | step: median, CV | return: median, CV |
|---|---:|---|---|
| actor grad norm > 2 | 21 | 3.75M, 0.49 | 7180, 0.26 |
| q_mse > 0.5 | 25 | 2.25M, 0.58 | 5380, 0.27 |
| KL multiplier > 0.5 | 22 | 1.50M, 0.67 | 4990, 0.34 |
| crash share >= 3% (0.7 x median rule, 0.5M window) | 24 | 3.62M, 0.58 | 7000, 0.25 |

Pooled 0.5M windows of the same runs, by return band:

| Return band | windows / runs | crash share, median [IQR] | pre-tanh std | q_mse | q_mse / target^2 (x 1e4) | actor grad | KL mult | adv corr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 5-7k | 111 / 27 | 0.012 [0.005, 0.030] | 0.42 | 0.40 | 1.34 | 0.82 | 0.29 | 0.116 |
| 7-9k | 128 / 27 | 0.025 [0.012, 0.051] | 0.44 | 1.04 | 2.23 | 2.12 | 0.63 | 0.101 |
| 9-10.5k | 110 / 24 | 0.040 [0.026, 0.054] | 0.46 | 1.77 | 2.41 | 3.78 | 0.91 | 0.085 |
| 10.5-12k | 283 / 11 | 0.036 [0.026, 0.052] | 0.49 | 3.70 | 3.53 | 3.11 | 0.74 | 0.073 |
| 12-14.5k | 141 / 3 | 0.052 [0.036, 0.082] | 0.51 | 4.63 | 3.81 | 1.99 | 0.53 | 0.086 |

**Answer (e):** the change is present in every upright run that gets fast enough, and it is tied to the gait
reaching ~6.5-8k return (about 7 m/s), not to a step count. The reference meets it at 3-4M because that is when
it gets there; slower runs meet it at 4.5-7.5M with the same signature; a run that stalls below 7k (v43 to
-2.0/dim) never shows the crash or q_mse part. It is graded rather than a step: crash share and relative
critic error keep rising with return. Thresholds are crude and the runs are one seed each; the 2x difference
in dispersion is the robust part.

## 2.3 What a crash is [OBS, `posture_sim_v60base.jsonl`]

48 stochastic and 2 deterministic episodes per condition on the v60 base checkpoints:

| Checkpoint | noise scale | deterministic return | stochastic median / mean | episodes with an inversion | crashes (< 0.85 x median) | pre-tanh std | sampled / mode saturation |
|---|---:|---:|---|---|---|---:|---|
| 1.02M | 1.0 | 4365, 4390 | 4099 / 4095 | 1 (2 steps) | 0/48 [0, 0.07] | 0.33 | 0.049 / 0.042 |
| 1.02M | 0.5 | | 4323 / 4307 | 0 | 0/48 | | |
| 4.01M | 1.0 | 10413, 10473 | 9406 / 9222 | 2 | 2/48 = 0.04 [0.01, 0.14] | 0.46 | 0.025 / 0.006 |
| 4.01M | 0.5 | | 10205 / 10184 | 0 | 0/48 [0, 0.07] | | 0.007 / 0.003 |
| 7.91M | 1.0 | 12490, 12613 | 11542 / 10715 | 6 = 0.125 [0.06, 0.25] | 5/48 = 0.10 [0.05, 0.22] | 0.52 | 0.029 / 0.005 |
| 7.91M | 0.5 | | 12337 / 12136 | 1 = 0.02 [0.00, 0.11] | 1/48 | | 0.006 / 0.002 |

- The crash is a **forward somersault**: in 7 of the 8 full-noise episodes with an inversion the root pitch
  runs from about -0.5 to +3.7 ... +16.3 rad (0.5 to 2.5 revolutions; the eighth is a brief excursion to 2.0),
  starting at step 46-689 from a 5-11 m/s gait. In 5 of the 8 the cheetah
  ends on its back and stays there at **0.2-0.5 m/s** for the rest of the episode (returns 1437-5365); in 3 it
  lands on its feet after a full revolution and carries on (4949-9916). The 8M policy has no back gait.
- Noise is the trigger for a fixed policy: tumbles at full vs half noise are 6/48 vs 1/48 at 7.9M (p 0.11) and
  8/96 vs 1/96 pooled over 4M and 7.9M (p 0.035). The rate rises with speed: 0 / 4% / 12.5% at 4.4k / 10.4k /
  12.5k deterministic return.
- Cost at 7.9M: mean stochastic return 10715 vs 12136 at half noise vs 12550 deterministic. Roughly 800 of the
  1840 gap between deterministic and stochastic return is crashed episodes (median 11542 vs mean 10715); the
  rest is ordinary noise cost.
- This matches the training logs' 4-8% crash share at 7-8M and the stochastic-vs-deterministic gap that
  report 03 describes as "late progress is mostly the policy becoming robust to its own noise".

## 2.4 (f) Candidate causes against the data

| Candidate | Verdict | Evidence |
|---|---|---|
| **Episode returns become heavy-tailed (stumbles), so target noise explodes** | **Supported; the best-evidenced link** | [OBS] Crash share 0 -> 4-9% coincides with the q_mse rise in A, B and five other runs at their own crossing steps; crashes are tumbles into a near-stationary inverted state. [INF] Arithmetic: a crashed episode contributes ~50 tumble rows and ~500 inverted rows whose value is 0-10 against ~80-100 for the gait. At a 5% crash share that is ~0.25% of rows in a tumble and ~2.5% inverted. A value error of 40 on 0.1% of rows is already q_mse 1.6, the whole logged level (1.1-1.7). The oracle's flipped state (critic 13.4 vs true 3.8, 80% of the squared action gradient) and report 03's "near-flip states that inflate every pooled variance" at 12M / 30M are these rows. The target-noise growth of 2.5x between 4M and 8M against 1.9x for the true within-state advantage variance is the same thing. |
| **Squashed-entropy target forces pre-tanh std up as actions grow, multiplying noise as the gait becomes noise-sensitive** | **Supported as arithmetic and as a contributor to *where* the plateau sits; not what starts the change** | [OBS] Per-dimension squashed entropy is 0.5 ln(2 pi e sigma^2) + E ln(1 - tanh^2(u)). The pre-tanh sigma that gives -0.5 nats at pre-tanh mean m is 0.150 (m 0), 0.176 (0.4), 0.240 (0.7, \|a\| 0.60), 0.274 (0.8, \|a\| 0.66), 0.371 (1.0, \|a\| 0.76); **for m >= 1.5 (\|a\| >= 0.91) no sigma reaches -0.5** (maximum -0.514 at sigma 1.25). A dimension-state whose mean sits near a bound has to be paid for by extra sigma everywhere else, through one scalar temperature. Logged: mean \|tanh(mu)\| 0.59 -> 0.65, sigma 0.39 -> 0.52, temperature 0.010 -> 0.040, entropy pinned at -3.0. The observed sigma is 2x what -0.5 nats means for an unsaturated action. But sigma rises smoothly from 1M (0.39 at 1.5M, 0.45 at 3M, 0.47 at 4M) with no step at 3-4M, and the crash onset tracks speed (same signature at sigma 0.35-0.38 in v44 const and v43 f1p0). |
| Action saturation (mode approaching +-1) | **Not supported as stated** | [OBS] Deterministic mode saturation is 0.1-0.6% from 4M (4-5% at 1M, i.e. it *falls*); sampled-action saturation 1.9% -> 3.2%; mean \|tanh(mu)\| moves 0.59 -> 0.65. The means are not at the bounds; the entropy arithmetic above bites well before they are. |
| Reward scale pushes Q toward the edge of the HL-Gauss support, or fixed bins become relatively coarser | **Not supported** | [OBS] `support_clip_frac` is 0 throughout; target mean 47 -> 102 on a [-50, 250] support. Bins are 2.0 wide and Q grows, so bins get relatively *finer*. q_mse / target^2 rises 1.3e-4 -> 2.2e-4 -> 3.5e-4 across the bands, so the error grows faster than scale; v14's scalar head shows the same level (q_mse > 0.5 at 4.8k, crash >= 3% at 9.9k). `critic_ce` rises 1.22 -> 1.34, consistent with broader target distributions. |
| KL gate binding harder as sigma changes | **Effect, not cause** | [OBS] Mean KL stays at 0.030-0.032 and the violation fraction at 0.30-0.36 from 1.5M to 16M. The multiplier (0.33 -> 0.74) and the pre-clip gradient norm (1.3 -> 3.3) rise together when the crashed rows arrive: the multiplier is the dual's response to larger raw gradients. |

**Do runs whose noise falls avoid or delay the change?** [OBS, `regime_noise_arms.txt`]

| Arm | pre-tanh std | crash share | q_mse | actor grad / KL mult | adv corr | sampled saturation | return, det. eval |
|---|---|---|---|---|---|---|---|
| fork control (-0.5/dim), 9-12M | 0.52 | 0.035-0.041 | 1.2-1.3 | 2.2-2.6 / 0.5-0.6 | 0.07-0.08 | 0.030 | 11.5-11.7k, 11.9-12.2k |
| fork -1.0/dim | 0.44 | 0.033-0.047 | 1.8-2.1 | 3.5-3.7 / 0.7 | 0.03-0.06 | 0.070 | 12.2-12.5k, 12.6-12.9k |
| fork -1.5/dim | 0.35-0.38 | 0.032-0.077 | 2.2-3.4 | 6.9-7.9 / 1.3-1.5 | 0.03-0.04 | 0.11-0.14 | 12.2-12.6k, 8.7-12.5k (unstable) |
| fork -2.0/dim | 0.30-0.33 | 0.042-0.051 | 2.1-2.8 | 5.5-9.8 / 1.0-1.8 | 0.02-0.05 | 0.09-0.16 | 12.5-12.8k, 12.9-13.6k |
| v31 price cap 0.02 (T capped), 6-10M | 0.42-0.43 | 0.058-0.083 | 2.6-2.7 | 5.2-5.6 / 1.1-1.2 | 0.054 | 0.060 | 11.4-11.7k, 9.2-10.7k |
| v31 price cap 0.03, 6-16M | 0.48 | 0.029-0.049 | 1.4-1.7 | 2.7-3.6 / 0.6-0.8 | 0.05-0.07 | 0.040 | 11.1-12.3k, 12.0-12.9k |
| v43 0-nats (more noise), 4-8M | 0.63-0.64 | 0.053-0.073 | 0.74-0.95 | 1.5-1.75 / 0.5 | 0.14-0.15 | 0.010 | 7.6-8.7k, 9.2-9.5k |
| v54 IDS belief (more noise), 3-8M | 0.60-0.66 | 0.032-0.080 | 0.7-0.9 | 0.8-1.55 / 0.3-0.4 | 0.13-0.15 | 0.011 | 6.9-9.2k, 8.6-9.8k |
| v56 calibrated IDS, 3-7M | 0.47-0.49 | **0.005-0.013** | 0.4-0.66 | 1.3-2.0 / 0.34-0.46 | 0.08-0.10 | 0.027 | 7.0-8.8k, 6.9-9.4k |
| v43 to -2.0/dim from 0.8M, 3-8M | 0.29 -> 0.21 | **0.001-0.003** | **0.09-0.12** | 1.7 -> 7.4 / 0.4 -> 1.3 | 0.08-0.11 | 0.04 -> 0.10 | stalls 5.9 -> 6.9k |
| v47 no entropy (Beta), 3-8M | 0.06-0.08 | 0.03-0.06 | 0.9-1.2 | 4.5-5.5 / 0.9-1.1 | 0.02-0.06 | 0.05-0.10 | 7.9 -> 9.2k |
| v46 late drop (K = 4), 5-8M | 0.40 -> 0.31 | 0.060-0.096 | 2.5-3.2 | 3.1-8.2 / 0.8-1.7 | 0.03-0.05 | 0.05-0.08 | 10.6-11.2k, 10.5-12.3k |

- **Less noise late does not remove the crashes or repair the critic.** Every late entropy reduction keeps a
  3-8% crash share and has *higher* q_mse, actor gradient and KL multiplier and *lower* advantage correlation
  than its control. What changes is the policy: sampled saturation doubles to quintuples and mean
  \|tanh(mu)\| rises 0.64 -> 0.71-0.75. Given less noise, the actor pushes the gait harder until it tumbles at
  the same rate. Returns still improve by 0.8-1.1k (stochastic) and 0.6-1.4k (deterministic at -2.0/dim).
- **More noise brings the change earlier and lower.** v43 0-nats and v54 crash at 3-8% from 6.4-6.9k and
  plateau at 8.7-9.2k, with a healthy-looking critic (q_mse < 1, advantage correlation 0.13-0.15, the best in
  the family). The critic metrics are not what limits these runs.
- **Runs that never reach the fast gait never crash and keep a small q_mse** (v43 to -2.0/dim: 0.1-0.3%,
  q_mse 0.09-0.12 for 5M steps). Their actor gradient norm and KL multiplier rise anyway (to 7.4 and 1.3), so
  those two also respond to shrinking sigma and are the least specific markers of the change. v56 is the only
  run that passes 7k with a crash share near 1% and q_mse < 0.7; it is also the slowest to get there.
- No-entropy runs (v47) crash at 3-6% with action sd 0.06-0.08: a collapsed-spread policy is fragile in its
  own way. Low noise is not sufficient for a robust gait.

## 2.5 (g) Most likely causal chain

1. **[OBS]** Around 6.5-8k return the gait is fast enough that a forward somersault becomes reachable from
   the policy's own exploration noise: 0% of episodes at 1M (4.4k), 4% at 4M (10.4k det.), 12% at 7.9M
   (12.5k det.); at half noise 0%, 0%, 2%.
2. **[OBS]** HalfCheetah does not terminate and the late policy has no back gait, so a tumble leaves hundreds
   of near-stationary inverted rows per crashed episode: 2-5% of rows from 3M on.
3. **[OBS + INF]** Those rows carry a value 70-100 below the gait's, a critic that over-values them (13.4 vs
   3.8) with |dQ/da| 60x the median, and lambda-returns that straddle the tumble. q_mse rises 4.5x (and 2.6x
   relative to target^2), the pooled held-out advantage correlation drops 0.12 -> 0.07, the actor's pre-clip
   gradient norm rises 2.6x with the top 1% of states holding 73-80% of its squared magnitude, and the KL
   multiplier doubles to keep the realised KL at the bound.
4. **[INF]** From here the learner sits on a speed-versus-tumble frontier. Going faster raises the tumble
   rate; the pathwise gradient is a poor guide to avoiding tumbles, because the only rows that carry that
   information are the rare, badly fit ones. The return slope halves and keeps halving.
5. **[OBS]** The entropy target fixes where on the frontier the run stops: it holds the squashed entropy at
   -0.5/dim, which by the tanh arithmetic means a pre-tanh sigma that *rises* (0.39 -> 0.52) as the actions
   grow. Lower the target at 8M and the run moves +0.8-1.4k along the frontier and settles at the same tumble
   rate; raise it and the run stops 2-3k earlier.

Confidence:

| Claim | Confidence | Why not higher |
|---|---|---|
| The 3-4M shift in q_mse / advantage correlation / actor gradient / KL multiplier is the arrival of crashed episodes | **~75%** | Timing in 6 runs at different steps, the rollouts, the row arithmetic, the oracle's outlier state and the interval pitch rms all agree; but nobody has yet split those metrics by row |
| The change is tied to gait speed, not step count | **~85%** | Single seeds, crude thresholds |
| The late return plateau is the speed-versus-tumble frontier at the noise the entropy target imposes | **~50%** | The entropy forks move the plateau by ~1k while the tumble rate stays fixed, which fits; but v43 0-nats and v54 plateau with a healthy critic and v47 crashes at tiny noise, so a critic-accuracy ceiling independent of tumbles (report 03: estimation-limited) remains a full alternative. Crashes cost ~800 return directly at 7.9M; the rest of the 50M gap to PPO is not explained here |
| Mode saturation or the HL-Gauss support causes it | < 10% | Section 2.4 |

**Checks that would confirm it** (cheap, CPU, existing checkpoints):

1. **Row split.** Roll out the 4M and 7.9M v60 checkpoints once (16k rows each), tag every row as upright /
   in a tumble / inverted from raw obs[1], and recompute q_mse, the held-out advantage correlation and the
   share of sum |sigma dQ/du|^2 per tag, using `audit_actor_gradient_tails.py`'s machinery. Prediction: more
   than 70% of q_mse and more than half of the squared actor gradient come from fewer than 5% of rows, and
   on upright rows q_mse / target^2 and the advantage correlation are at their pre-3M values (~1.3e-4,
   ~0.12). If the upright rows alone show the degradation, this chain is wrong and the critic ceiling is the
   story.
2. **Log `posture/inverted_frac` and a tumble count per iteration** in the next trainer version, next to
   q_mse. Prediction: q_mse tracks the tumble count iteration by iteration (it should already be visible as
   the spikiness of q_mse: 0.60 / 0.89 / 1.18 / 1.07 / 1.37 in consecutive bins).

The decisive but not free test is a fork from the 4M checkpoint that truncates training episodes on
sustained inversion (fork-paired with an untouched control, as in v62). Prediction: q_mse, actor gradient norm
and KL multiplier fall back toward their 2-3M relative levels within ~0.3M steps. Whether the return slope
recovers is exactly what separates the 75% claim from the 50% one.

---

## 3. How the two events connect

| | Event 1 (first 0.3M) | Event 2 (from ~7k return) |
|---|---|---|
| Trigger | initial tanh-Gaussian action noise (sd 0.65) on an untrained body [INF]; own noise on a young gait [OBS] | own exploration noise on a fast gait [OBS] |
| Motion | forward tip, pitch +3.6 | forward somersault, 0.5-2.5 revolutions |
| After | back gait at 1.4-2.2 m/s, improvable, stable under noise | no back gait, 0.2-0.5 m/s |
| Share of rows | 10-99% | 2-5% |
| Consequence | a competing basin that owns the data; 22% trapped, 43% visibly affected | outlier rows that own the critic's error and the actor's gradient energy |

One design fact produces both: an episode that cannot end, a state that cannot be left, and exploration noise
large enough to reach it. PPO-Beta in this repo avoids the first by starting quieter and learning slower, and
its entropy keeps falling (to -4.2 ... -4.5 nats at 8M against REPPO's pinned -3.0), which is the direction
the 8M entropy forks say helps.

## 4. Limits

- One seed everywhere. "Distinct trajectories" are different designs or perturbations at seed 1, not
  replicates of one design; only 4 are the reference algorithm.
- Posture is measured for 23 runs (observation statistics) and 9 snapshots (rollouts); the other
  classifications come from the return record. The M class is the least certain.
- The claim that all tanh runs start inverted is an inference from cumulative statistics at 0.26-1.0M; no
  snapshot exists before 262k for any tanh run. The one-second random-action check in section 1.5 settles it.
- Rollout counts are small (48 per condition; 2-10 on the early snapshots). The noise-scale effect is p 0.035
  pooled, p 0.11 at 7.9M alone.
- The Beta-vs-tanh contrast is confounded with seven different designs and, for PPO, with the whole
  algorithm. The lambda contrast is confounded with era.
- Crash share from the return record uses a 0.7 x or 0.85 x median rule; in fast-growing early windows it
  over-counts, which is why the pooled table starts at 5k.
- I did not split any critic metric by row. Section 2.5's first check is the gap between this report's
  75% and a measurement.
