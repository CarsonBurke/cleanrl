# OPSD-AdvCond: normalization ablations (no-advnorm / no-rewardnorm)

Status date: 2026-08-24. Env: HalfCheetah-v4, seed 1, shared `mbpercnorm_v2` chassis.
Method family: `cleanrl/opsd/core/advcond/ppo_continuous_action_opsd_advcond_v{1..7}.py` — on-policy self-distillation,
**no PPO** (no importance ratio, no clipped surrogate, no advantage-weighted policy gradient).

## HEADLINE: the conditioning-scale fix beat PPO at 8M

`cond_scale=ema_rms` (scale-only conditioning, no mean subtraction) run to the standard
8M horizon, HalfCheetah-v4, seed 1, a4c4 mb128, `adv_boost=1`:

| | @500k | @1M | @2M | @4M | @6M | @8M |
| --- | --- | --- | --- | --- | --- | --- |
| PPO baseline (same chassis) | **1576** | 3079 | 5012 | 6716 | - | 8278 |
| **ema_rms, kappa=1** | 994 | **3826** | **6267** | 7680 | **8598** | **8819** |
| ema_rms, kappa=2 | 1524 | 3993 | 6084 | **7848** | 8486 | 8249 |

**8819 @8M vs PPO's 8278: +6.5%.** `score_runs.py` last-20 reads 8718.4 +-497.7. Ahead of
PPO from 1M onward, and past PPO's *final* 8M score by 6M. First arm in the OPSD-AdvCond
lineage to beat PPO at the full horizon, and the entire margin comes from one variable:
how the advantage is scaled before being used as a conditioning input.

### The margin's optimum is horizon-dependent

kappa=2 leads through 4M (7848 vs 7680) and then loses at 8M (8249 vs 8819). At 2M the
ladder is kappa=2 (6077) > 3 (5624) > 1 (5365) > 4 (3128, after posting the lineage's best
@1M at 4255 and then destabilizing). A bigger margin buys early speed and spends late
stability; the LR anneal appears to interact. Linear kappa annealing over 2M did **not**
capture both ends (4->1: 4415, 4->2: 5864, both below constant kappa=2) - prediction
falsified. An 8M 2->1 anneal and a constant kappa=1.5 are queued at the horizon where the
flip actually happens.

### Two more falsified predictions

- `clone_coef` has a **non-monotone optimum at 1.0**: 0.5 -> 4636 @2M, 1.0 -> 5365,
  2.0 -> 4340. "More sharpening is strictly better" is wrong.
- Removing the clone term's sharpening while keeping its calibration (v9) is **worse**
  (4136 @2M) despite behaving exactly as designed - less entropy collapse, larger
  improvement steps. The champion collapses to H=-13.0 and wins. On HalfCheetah the
  sharpening is a feature, not the pathology I diagnosed.

Throughput note: the acting forward on this chassis is launch-bound, not compute-bound
(eager `act` costs 1149 us at batch 16 and 1277 us at batch 256 - flat). `--async-envs`
plus `--compile-act` and a larger `num_envs` took the marginal rate 3025 -> 6013 SPS
(2.0x), so an 8M run is ~22 min rather than ~48. Neither flag changes the algorithm.

## TL;DR - the ablations ran, and the prediction held

`ema_rms` (scale-only, no mean subtraction) **beats** the batch-standardized champion at
every matched step on HalfCheetah-v4, seed 1, a4c4 mb128:

| arm (`cond_scale`) | @250k | @500k | @750k | @1M | note |
| --- | --- | --- | --- | --- | --- |
| `batch` (PPO-style advnorm) | -15 | 354 | 1614 | 2334 | prior champion, 4452 @2M |
| **`ema_rms`** (scale only) | **65** | **664** | **1845** | **2962** | **+27% @1M** |

That is the predicted direction: mean subtraction was destroying the conditional's natural
zero, and it cost ~27% of return at 1M. Scale-only fixes it without unbounding the input.

| question | state | evidence |
| --- | --- | --- |
| no **reward** norm | **the condition of every run to date** | `normalize_reward: bool = False`, no run passes the flag |
| reward norm ON (control) | queued (3574) | - |
| scale-only, no mean subtraction (`ema_rms`) | **MEASURED, WINS** | 2962 @1M vs 2334 |
| no **adv** norm at all (`cond_scale=raw`) | **running** (3573) | 4 @250k; RMS 1.50 and rising |
| `raw` with the clip widened to +-50 (3601) | queued | see caveat below |
| PPO-style advnorm (`batch`) | measured to 2M | 4452 @2M |

**Caveat on the `raw` arm.** `adv_cond_clip` (+-3) is applied in *every* mode, and raw
`delta` reached RMS ~14.7 on the champion's run. So job 3573 will measure *clipping*, not
unnormalized conditioning, once its scale passes 3. Job 3601 is the honest version
(`--cond-scale raw --adv-cond-clip 50`). Expect it to fail differently: `AdvEmbed`'s
frequencies are fixed at 0.5..8, so an O(15) input **aliases** rather than resolves.
Early read at 400k, before its scale passed the clip: `raw` had a *stronger* channel than
`ema_rms` (cond_gap 0.0291 vs 0.0086, distill_kl 0.124 vs 0.043).

## Why advantage normalization is suspect in this method specifically

PPO normalizes advantages to control **the scale of a gradient**. This method never
multiplies a gradient by an advantage. Here `delta_t` (a 1-step TD residual) is an **input**
to a learned conditional: the teacher is `pi(.|s, delta_t + kappa)`, the student is
`pi(.|s, absent)`. Batch mean/sd is the wrong tool for an input, for two separate reasons:

1. **Mean subtraction destroys the natural zero.** `delta_t = 0` means "exactly as V(s)
   expected". After subtracting the batch mean, the least-bad action in an all-bad batch is
   relabelled *positive*. The conditional is taught a false sign.
2. **Batch sd makes the units non-stationary.** The margin `adv_boost` is quoted in those
   units, so "beat yourself by 1" is a tiny real gain when the batch sd is small and a large
   one later. We inflict input distribution shift on ourselves every iteration.

The counter-constraint: `delta` cannot simply be fed raw, because `AdvEmbed`'s Fourier
frequencies are **fixed** (`logspace(-1, 3, base=2)` = 0.5 .. 8) and only resolve an O(1)
input. So the three arms are: full standardization, scale-only, and nothing.

## Mechanism data (v4, 131072 steps, mb=32, a4c4, adv_boost=1.0)

Short-run diagnostics only — no return claim.

| mode | `cond_scale_rms` over 4 iters | `cond_clip_frac` | entropy | `distill_kl` | EV |
| --- | --- | --- | --- | --- | --- |
| `batch` | 1.00 1.00 1.00 1.00 (pinned by construction) | 0.9% | -0.708 | 0.126 | 0.406 |
| `ema_rms` | 1.00 1.28 1.40 1.18 | 2.0% | -0.654 | 0.078 | 0.445 |
| `raw` | 0.61 1.34 **2.17** 1.90 | **10.8%** | **-0.512** | 0.063 | 0.461 |

Two readings:

- **The raw residual is genuinely non-stationary.** Its RMS grew 3.5x in four iterations as
  V learned. Left unscaled it was already saturating the +-3 conditioning clamp at 10.8% and
  rising, which destroys the channel's resolution — so *some* scale tracking is required, and
  "no advnorm at all" is likely to degrade late rather than early.
- **Entropy preservation orders `raw` > `ema_rms` > `batch`**, exactly as the false-sign
  argument predicts: teaching a wrong sign has to be paid for by sharpening the conditional.

This is why `ema_rms` (keep the zero, divide by a slow de-biased RMS of `delta`, never
subtract the mean) is the arm I expect to win, and why `raw` is run anyway rather than
argued away.

## Reward normalization: current state

- `normalize_reward: bool = False` and `clip_reward: bool = False` are the defaults, and no
  submitted job overrides them. `gym.wrappers.NormalizeReward` is therefore never
  constructed. **All results in this family are already the no-reward-norm condition.**
- `gym.wrappers.NormalizeObservation` **is** active. It stays: every durable baseline on this
  chassis (PPO 8278 @8M, hopsd_v27 8171 @8M) uses it, so removing it would void the
  comparisons rather than test anything.
- Job 3574 runs the champion config with `--normalize-reward` so that "off is better" becomes
  a measurement instead of an inherited default.

## Reference points

| arm | @500k | @1M | @1.5M | @2M | slope /1M |
| --- | --- | --- | --- | --- | --- |
| PPO baseline (this chassis) | 1576 | 3079 | — | 5012 | — |
| **champion** `v5_a4c4_mb128` (`cond_scale=batch`) | 899 | 2581 | 3832 | **4452** | +1547 |

Champion is at **89% of PPO @2M**, up from 69% at 1M, but its slope decayed +3754 -> +1547,
so it is decelerating.

### Why it decelerates (measured from the champion's own logged internals)

An earlier draft of this note said `delta`'s spread collapses. **That is refuted by the
champion's own telemetry** and is corrected here.

| metric | 0k | 500k | 1M | 1.5M | 1.75M |
| --- | --- | --- | --- | --- | --- |
| `advantage_std` | 3.94 | 6.25 | 14.56 | 14.44 | **14.72** |
| `cond_gap` | 0.030 | 0.022 | 0.016 | 0.011 | **0.010** |
| `distill_kl` | 0.162 | 0.096 | 0.066 | 0.046 | **0.039** |
| teacher - student entropy | -0.146 | -0.093 | -0.068 | -0.049 | **-0.046** |
| student entropy | -0.72 | -1.90 | -3.70 | -5.23 | **-5.71** |
| explained variance | 0.38 | 0.95 | 0.97 | 0.96 | 0.94 |

`delta`'s spread **grew 3.2x**. The actual mechanism is the reverse: the **policy sharpens**
(5 nats of entropy gone) and, as it does, the teacher and the student converge onto each
other — `cond_gap` 3x down, `distill_kl` 4x down, the entropy gap between the two contexts
3x down. The improvement operator's signal *is* that gap, so the operator runs out of things
to say. The critic is not implicated (EV 0.94-0.97 throughout).

### Consequence for these ablations — a falsifiable prediction

Because `advantage_std` reaches ~14.7 while `adv_cond_clip = 3.0`:

- **`raw` (3573) should degenerate late.** Unscaled `delta` with sd ~14.7 against a +-3
  clamp means near-total saturation, so the conditioning channel loses nearly all
  resolution. The 131k data already showed 10.8% clipped and rising; this predicts it goes
  much further. If `raw` instead holds up, my model of the channel is wrong.
- **`ema_rms` (3572) should stay healthy**, dividing by a slow RMS that tracks 14.7 and
  keeping the Fourier input O(1).
- **Neither fixes deceleration.** All three modes only change how honestly the channel is
  encoded; the vanishing teacher-student gap is what `v7`'s KL-targeted dose attacks, by
  pinning that gap directly on the real distributions.

## Noise floor — read the ablations against this

`v3_mb128_e4` and `v5_a4c4_mb128` are **the same algorithm on the same seed**, differing only
in floating-point association (`clone + distill + vf*v` in one expression vs accumulated in
two). They read **2908 vs 2581 @1M**.

So roughly **300 points at 1M is pure numerical noise**, and only seed 1 exists. Differences
below ~400 between ablation arms will not be treated as findings.

## Queue state (all mine, `maxParallelRuns=1`, 2M steps, seed 1)

| job | arm | tests |
| --- | --- | --- |
| 3570 | `v5_a8c4_mb128` | more actor reuse |
| 3571 | `v5_a8c8_mb128` | more actor + critic reuse |
| 3572 | `v6_emarms_a4c4_mb128` | **scale-only conditioning** |
| 3573 | `v6_raw_a4c4_mb128` | **no advnorm at all** |
| 3574 | `v6_batch_rewnorm_a4c4` | **reward norm ON control** |
| 3575 | `v7_kl0p02_a4c4_mb128` | KL-targeted dose, target 0.02 |
| 3576 | `v7_kl0p05_a4c4_mb128` | KL-targeted dose, target 0.05 |

Every arm is a single-variable change from the champion (`a4c4`, `mb128`, `cond_scale=batch`),
so each result is attributable. `v6` with `cond_scale=batch` reproduces `v5` exactly, which is
what makes 3572/3573/3574 clean comparisons against the champion row above.

## What will settle it

1. 3572 vs champion: does dropping mean subtraction help return, or only entropy?
2. 3573 vs 3572: is scale tracking necessary, or was the clip saturation harmless?
3. 3574 vs champion: is reward-norm-off actually right, or merely inherited?

Decision rule: any gap under ~400 @1M is noise; a real winner must also hold at 2M and show
a non-decaying slope.
