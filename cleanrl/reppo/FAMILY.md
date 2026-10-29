# REPPO line

Pathwise-actor PPO-style method: HL-Gauss Q(s, a) critic fit to the fresh rollout, tanh-Gaussian actor
climbing dQ/da under a per-state KL gate and an entropy-target dual. HalfCheetah-v4, seed 1, 16 envs.
Scores are stochastic training returns (the benchmark), matched-step means from `scripts/score_runs.py`.

Best lineage: v7 -> v23 -> v25 (in-iteration target refresh, lambda 0.8) -> v27 (H-step latent-roll
critic aux, run with `--aux-horizon 8`) -> v31 (entropy price cap).

Diagnosis of the v27 plateau: `benchmarks/reppo_diagnose_v29/analysis.md`.

**Read `benchmarks/reppo_rootcause/README.md` before trusting anything below.** That audit found the
steering metric (`advantage_corr_heldout`, ceiling ~0.3, anti-predictive of return), the 1M oracle's
pooled statistics (one state holds 86-88% of the energy) and many single-trajectory verdicts in this file
unreliable; several "refuted" ideas (exact targets, hybrid gradient, replay, orthogonal advantage) were
never effectively tested.

## v30-v32 (plateau campaign after the v29 diagnosis)

| Version | Idea | @4M | @8M | @12M | End (16.1M) |
|---|---|---:|---:|---:|---:|
| v27 H8 (ref) | latent-roll aux | 8596 | 11031 | 11274 | 11282 |
| v30 modelgrad w=0.5 | actor climbs r_hat + gamma V(s + delta_hat) (SVG(1)) | 8605 | 9588 | 10919 | 10150 |
| v30 modelgrad w=1.0 | same, model only | 7214 | 8810 | 9371 | culled 14.1M |
| v27 H8 ent-target 1.0 / 1.5 nats/dim | fixed lower noise | - | - | - | stalled ~3.7k, culled 2.5M |
| v31 pricecap 0.02 | temperature <= cap | 8800 | 10832 | 11773 | 12432 |
| v31 pricecap 0.03 | temperature <= cap | 8720 | 11373 | 11978 | 12308 |
| v32 replaytwin | v27 critic on 1M replay, 1-step twin-min targets, Polyak | 2349 | - | - | culled 4.6M |
| v33 mixtarget | v32 + half fresh lambda-0.8 rows per minibatch | 1769@1M | - | - | cancelled 1.1M (~2k SPS) |
| v35 window kl 0.03 / 0.0075 | 128-step segments, critic on 8-segment window | 2226 / 2009 | - | - | flipped, ended ~9.6M |
| v34 replayaug | v27 exactly + 1024 replay 1-step rows per minibatch, v27 refresh | 4839 | 5562 | - | culled 9.2M (15k SPS) |

Findings:

- **v30**: the world model is accurate (delta R^2 0.99) yet its Q estimate's held-out advantage correlation
  (0.05-0.11) is no better than the critic's. The weak link is the value surface, not the action's
  one-step consequence; routing the gradient through dV/ds' does not help.
- **Noise**: the mean-action policy scores ~1k above the stochastic one (fixed -0.5 nats/dim target keeps
  pre-squash std ~0.5). Lower fixed targets kill early learning. v31's price cap keeps v27 early and lets
  entropy fall later: +1k at the end, but deterministic eval is unchanged (~12.1k at 12M) - it removes
  noise cost, it does not improve the policy. A patch, not a fix.
- **v32/v33**: Polyak targets (tau 0.005 over 512 updates, ~92% moved per rollout) reproduce v7's
  once-per-rollout bootstrap staleness, and both fell into the ~2.3k local optimum as v7 lambda 0.8 did
  (v25, same lambda with in-iteration refresh, escapes). At 1M v32 valued its policy at 10.2 vs ~18 implied
  by its returns. Twins x 512 updates x H8 also cost ~8x v27 (2-2.6k SPS). (An earlier reading blamed short
  target horizon; the staleness confound was not controlled.)
- **v34**: with v27's fresh bootstrap refresh the trap is gone, but adding replay rows still halves progress
  (5.6k at 8M vs 11.0k). Replay with 1-step targets hurts this critic regardless of staleness; held-out
  advantage correlation did not improve (0.06-0.08). Replay is not the lever for REPPO.

## The ~2.3k "trap" is the cheetah flipped onto its back

`scripts/reppo_posture.py` on 1.5M snapshots (seed 1): v35 (2.2k) spends 99% of each episode with root pitch
~3.3 rad (upside down, running on its back); v27 H8 (5.2k) has pitch ~0.1 and never inverts. The flip is a
discrete, irreversible early event decided within the first ~0.5M steps. With one seed per design, v32, v33
and v35 "failing" at 2.3k measures whether that seed-1 trajectory flipped, not the design's quality: those
results are inconclusive, and the mechanistic explanations offered for them (target horizon, bootstrap
staleness, replay) are unsupported. Any version at ~2k-2.5k by 1M should be checked for posture before
drawing conclusions. v34 (5.6k at 8M, upright trajectory) remains a genuine negative.

## v36-v38 (fast path, model branches, action gap)

v36 is v27 H8 on a faster path (target features cached per refresh; TF32). Speed ladder, 800k steps, one
job at a time: update phase per 16k-step iteration 0.51 s (v27, strict FP32) -> 0.47 s (v36 FP32) ->
0.28 s (v36 TF32, default) -> 0.19 s (BF16). Returns at 800k are within seed noise (3.5-3.7k), so TF32 is the
default from v36 on; env + host rollout (~0.28 s) are now the larger share.

| Version | Idea | @1M | @4M | @8M |
|---|---|---:|---:|---:|
| v37 dynabranch | v36 + world-model ensemble on replay; critic also fits CRN branch returns (A=4, H=5) | 1697 | 2274 | 2363 (flipped) |
| v38 branchadv | actor climbs a scale-normalized A(s, a) fit to CRN branch contrasts (A=8, H=12, lambda .95) | ~1.2k | 2568 | culled 4.6M |

v37 flipped (posture at 1M: pitch 3.29, 99% upside down), but its diagnostics are informative regardless:
held-out model delta R^2 0.996 and reward R^2 0.999, yet the model's advantage correlation (~0.10) equals
the critic's. Branch returns of a state's policy-sampled actions differ by ~0.04 (scaled), the critic's
branch spread is half that and ranks them at corr ~0.3.

**The action gap.** Q ~ 100-110 (scaled) on HL-Gauss bins 2.0 wide (sigma 1.5); critic fit RMSE ~1.1
(q_mse 0.9-1.4); advantage_std_ratio 0.12-0.17. The action-dependent part of Q is ~1/40 of its label width
and is learned only as a residual of state-value fitting, so dQ/da is mostly noise however good the data.
This explains why replay (v32-v34), windowing (v35) and imagined targets (v37) did not move action ranking.
v38 gives action ranking its own, scale-normalized network trained on CRN branch contrasts.

Flip note: v32, v33, v35 and v37 flipped; the v27/v31/v34/v36 runs did not. Four of four critic-data
changes flipping suggests the flip is design-correlated (slower early upright progress), not pure luck.

**v38 negative (and the end of the model-imagination route).** A fit its own imagined contrasts poorly
(normalized MSE 0.78-0.93) and ranked imagined branches better than the critic (corr ~0.4 vs ~0.15-0.25),
but ranked *real* held-out actions worse (advantage corr ~0.02 vs the critic's ~0.09); the model's own
branch ranking against real returns was ~0.03-0.06. Posture at 1M: half inverted (47%). With v30 (model
value gradient), v37 (model targets for Q) and v38 (model contrasts for a separate A) all negative: a
one-step-accurate model (R^2 0.996) does not give action contrasts that predict real outcomes on this
task. The action-dependent effect lives in small, feedback-amplified state differences the model does
not resolve. Do not pursue model imagination for action ranking further.

## v39-v40 (observed-return gradient; fork protocol)

| Run | Idea | @500k | @1M | @2M | @4M |
|---|---|---:|---:|---:|---:|
| v39 dualpg w=0.5 | actor score (1-w) Q + w clipped LR surrogate on soft lambda-0.95 advantages | 755 | 1558 | 1678 | ~1.96k (flipped by 0.5M, cancelled) |
| v36, 128 envs x 128 steps | same 16k batch from 8x more independent trajectories | 816 | 3342 | 4704 | 6082 (upright; cancelled 4.4M) |
| v27 H8 ref | | 2521 | 4174 | 5634 | 8596 |

- 128 envs: upright but ~0.7x the reference throughout. More independent trajectories per batch (with
  128-step rollouts) does not help the critic; the effective-sample-size hypothesis is not supported.
- v39 flipped by 0.5M (pitch 3.3, 98% inverted). Five of the last six novel from-scratch runs flipped
  (v32, v33, v35, v37, v39), so seed-1 from-scratch runs no longer test the plateau, which is where this line
  is stuck.

**Fork protocol (v40).** `--init-snapshot` starts from the v27 H8 8M snapshot (upright, ~11k) and runs 4M
more steps. Logged steps continue from 8M, so runs compare directly with the reference (11274 @12M). The
control is `--pg-weight 0`, which is exactly v36. Optimizer states and the KL multiplier restart in every
arm, so the restart transient is shared. Plateau ideas are screened this way first, and winners get a full
from-scratch run.

Fork screens from v27 H8 @8M (training return @9M / 10M / 11M / 12M; deterministic eval late):

| Arm | @9M | @10M | @11M | @12M | det. eval |
|---|---:|---:|---:|---:|---:|
| control (v36 algorithm) | 11121 | 11070 | 11568 | 11408 | 12.0-12.4k |
| dual-PG w=0.5 | 10744 | 10691 | 10384 | 10972 | 11.3-11.8k |
| pure LR w=1 | 7692 | 8678 | - | - | 10.4k (cancelled 10.7M) |
| kl_bound 0.01 | 10597 | 10597 | 11085 | 11127 | 12.0-12.3k |
| kl_bound 0.003 | 9389 | 8490 | 7867 | 8071 | 10.4k, falling |
| reference (unforked) | 10928 | 11460 | 11106 | 11274 | - |

- The control reproduces the reference, so the fork protocol is sound. The LR arms crash right after the
  fork and stay behind. Their steady-state clip fraction is ~30%: the 0.03 per-state KL step leaves a third
  of the LR samples clipped, and the surrogate's mean hovers near zero. The observed-return estimator adds
  noise, not signal, at this stage.
- Noise-floor hypothesis refuted: a smaller trust region does not raise the plateau. At KL 0.003 the policy
  *degrades* steadily: the critic's gradient direction stays stable (consecutive cos ~0.84) while critic fit
  worsens (q_mse 0.9 -> 3.7) and std grows. Small steady steps along a stable, biased dQ/da drift into
  critic errors. This is the pathwise-exploitation (TD3 overestimation) signature, and REPPO has no
  pessimism.

v41 twin critics, forked from 8M. The second member starts fresh, and every arm gets a 16-iteration
critic-only warmup with the actor frozen:

| Arm | @9M | @10M | @11M | @12M | det. eval late |
|---|---:|---:|---:|---:|---:|
| actor climbs member 0 (control) | 10882 | 11305 | 11569 | 11005 | 12.2k |
| mean of 2 | 10109 | 10882 | 11165 | 10924 | 12.1k |
| min of 2 (TD3-style pessimism) | 10728 | 11170 | 11401 | 11402 | 12.4k |

Pessimism gives at most a marginal, tighter finish (11.5k +- 0.13k final window) and does not lift the
plateau. Member gap ~0.3 (scaled). Across v39-v41 forks, the observed-return estimator, step size and
pessimism all land at the same deterministic ~12.2k. The plateau is not an actor-side or
overestimation artifact.

v42 critic capacity, forked from 8M with only the actor loaded. The critic is fresh and gets a 16-iteration
warmup:

| Arm | @9M | @10M | @11M | @12M | det. eval late |
|---|---:|---:|---:|---:|---:|
| fresh 512 x 1 value layer | 6867 | 9932 | 10779 | 11130 | 11.8k |
| fresh 1024 x 3 value layers | 9767 | 10022 | 10436 | 10410 | 12.1k |

The bigger critic recovers faster after the fork (det. 9.2k vs 0.9k at 8.4M) but plateaus at the same ~12.1k.
Capacity is refuted. The deterministic ~12.2k plateau is invariant to the estimator, step size,
pessimism, capacity, replay, model imagination and env parallelism. The fixed entropy target (-0.5
nats/dim, pre-squash std ~0.5) is the one constraint never varied late.

**Entropy target forks from 8M: the one lever that moves the plateau.** The target is -mult x 6 nats; the
control is -0.5/dim:

| Target | @9M | @10M | @11M | @12M | det. eval late | final entropy / T |
|---|---:|---:|---:|---:|---:|---:|
| -0.5/dim (control) | 11121 | 11070 | 11568 | 11408 | 12.0-12.4k | -3.0 / ~0.04 |
| -1.0/dim | 11947 | 11951 | 11971 | 12291 | 12.8k | -5.8 / 0.024 |
| -1.5/dim | 11550 | 12034 | 11995 | 12178 | 12.1-12.8k | -9.1 / 0.011 |
| -2.0/dim | 12191 | 12450 | 12023 | 12399 | 13.3k -> 13.7k, rising | -12.9 / 0.0003 |

The fixed -0.5/dim entropy constraint sets the ~12.2k deterministic fixed point: stochastic returns gain ~+1k,
and at -2.0/dim the temperature goes to ~0, the policy picks std 0.33 on its own and the deterministic policy
keeps improving. Combined with the from-scratch stall of low fixed targets (~3.7k), noise is necessary early
and binding late. v43 schedules the target: -0.5/dim for the first 10% of iterations, then linear to the final
level (arms: -2.0 and -1.0 per dim).

**v43 entanneal (from scratch, 8M): early entropy reduction is harmful.** Both arms stay upright
(upside_down_frac 0 at 0.5M/1M) and match the reference to ~1M, then fall behind as the target tightens:

| Final target | @1M | @2M | @4M | @6M | @8M | det. @8M | entropy / T / std @8M |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref v27 H8 (-0.5 fixed) | 4174 | 5634 | 8596 | 10176 | 11031 | 12.1k | -3.3 / 0.034 / 0.49 |
| -1.0/dim | 3832 | 4922 | 5958 | 8460 | 10153 | 11.0k | -5.9 / 0.007 / 0.35 |
| -2.0/dim | 3863 | 5005 | 6029 | 6507 | 6936 | 7.3k | -11.2 / 1e-5 / 0.20 |

The damage is monotone in how much noise is removed mid-training, while the same reduction from 8M helps.
The mid-training phase is exploration-limited and the late phase is noise-limited; the fixed -0.5/dim
target is a compromise that is wrong in both phases. The higher-entropy side (less negative target) had
never been tested; v43 screens it next with constant targets.

**v43 higher-entropy screen (from scratch, 8M).** Training return per step (point values, not windows):

| Arm | @1M | @2M | @4M | @6M | @8M | det. @8M | entropy / std @8M |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref (-0.5 fixed) | 4174 | 5634 | 8596 | 10176 | 11031 | 12.1k | -3.3 / 0.49 |
| -0.25/dim fixed | 3233 | 4785 | 7564 | 9674 | 9440 | 11.3k | -1.6 / 0.55 |
| 0/dim fixed | 3572 | 4705 | 6797 | 8047 | 6460 | 9.5k | 0.1 / 0.65 |
| 0/dim until 50%, then to -1.5/dim | 3572 | 4705 | 6797 | 10644 | 12000 (last-20 11307) | 12.2k | -8.0 / 0.36 |

More entropy does not help mid-training (-0.5/dim is already near the right mid-phase level), so the
extrapolation from v43's early-reduction damage is refuted. The late drop recovers +2k within 2M steps
and ends above the reference on training return, but its deterministic return (12.2k) equals the
reference: at 8M the gain is the removed evaluation noise, not a better mean policy. Next: keep -0.5/dim
until 50-60% and then drop (s50_f1p5, s60_f2p0).

**v44 sharpmean (queued as 10453 default and 10454 constant -0.5/dim).** The pathwise gradient of
E Q(s, tanh(mu + sigma eps)) moves the mean to the optimum of a sigma-smoothed critic. v44 gives the mean
dQ/da at tanh(mu) and gives the std the sampled objective with mu detached; the log-density keeps its
mean gradient. Hypothesis: the deterministic ~12.2k@8M is the smoothed optimum, so v44 should raise it at
unchanged exploration noise.

**v43 late-drop screens (-0.5/dim until 50/60%, then to -1.5 / -2.0 per dim):**

| Arm | @1M | @2M | @4M | @6M | @8M | last-20 |
|---|---:|---:|---:|---:|---:|---:|
| s50 -> -1.5/dim | 3847 | 4799 | 6045 | 9567 | ~9974 | 10254 |
| s60 -> -2.0/dim | 3847 | 4799 | 6036 | 9239 | ~10401 | 10284 |

**Confound: the v36 fast base itself appears to learn slower from scratch.** These arms run the reference's
exact -0.5/dim until 4M, yet sit at ~6.0k@4M against v27 H8's 8596 (v31: 8720-8800; v29 noreward:
7637). Every v36-derived from-scratch run (v43 x5, v44 pending) sits at 6.0-6.8k@4M. So the from-scratch
v43 comparisons above are against the wrong baseline, and the flips of v37/v39 may also be partly the
base. v27 is bit-deterministic (v27 and v31 share 4173.6@1M); v36 diverges by 1M (3847). Testing:
v36 with --matmul-precision highest (job 10455) and a semantic audit of v27 vs v36.

**Audit: TF32 is the only semantic v27 -> v36 difference.** Cached target features are re-encoded after
every target copy (exact); all Args, losses, schedules and RNG streams match. Strict-FP32 runs (v27, v29,
v30, v31) give 4150 / 5620-5740 / 8280-8660 at 1M/2M/4M; TF32 runs (v36_e128, v43 x5) give 3530-3820 /
4680-4960 / 5900-7500. Caveat: the FP32 runs share one seed-1 path, so TF32 may partly act as a new
seed. There is a mechanism, though. TF32 keeps ~10 mantissa bits (relative ~5e-4), which is the same
size as the within-state action contrasts the actor climbs (~0.04 on Q ~100, relative ~4e-4). The
rounding enters the critic's action encoding, the 40.9x zero-logit offset and the 8-step latent chain.
The speed-ladder parity at 800k (3.5-3.7k for all precisions) was too early to show it. Decision: new
versions default to strict FP32 (v44 now does); the TF32/BF16 speedups are not free for REPPO. Job 10455
(v36 --matmul-precision highest) checks code equivalence against v27's 4151-4174@1M.

**Confirmed: v36 with --matmul-precision highest is bit-identical to v27 H8** (4173.6 / 5633.9 / 8596.0 /
10175.6 at 1-6M, 10894@8M; SPS ~10.6k with 3 concurrent runs). TF32 alone produced the ~2.5k@4M deficit
of every v36-lineage from-scratch run, so the from-scratch results of v37-v43 (including the flips and the
entropy-schedule rankings) were measured on a handicapped base and are not conclusive. Fork results
(v40-v42, entropy forks) started from the FP32 snapshot and matched their controls, so they stand.

**v44 sharpmean (FP32): refuted.**

| Arm | @1M | @2M | @4M | @6M | @8M | det. @8M | std @4M |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref (v36 FP32) | 4174 | 5634 | 8596 | 10176 | 10894 | 12.0k | 0.45 |
| sharp + late drop | 3944 | 4803 | 5624 | 6272 | 6645 | 7.0k | 0.35 |
| sharp, -0.5 fixed | 3944 | 4803 | 5635 | 6185 | 7400 | 8.5k | 0.35 |

noise_cost = Q(s, tanh mu) - E Q(s, a) is only 0.015-0.06 (scaled Q), about the size of the action
contrasts (~0.04): noise-smoothing barely moves the optimum, so it was never the ceiling. Decoupling the
mean from the sampled objective removes the coupling that kept std near 0.45 (it settles at 0.35) and
learning slows as with the v43 early reductions. Next: the late-drop schedule in FP32 (jobs 10456 s50->-1.5, 10457 s50->-2.0).

**v43 late drop in FP32 (10456/10457): both flipped** (2157@2M, ~2.5k@8M, flat). With the same algorithm
as v36-FP32 until 4M, v43's different code path (extra modules, ops order) alone produced a different
seed-1 trajectory that took the ~2.3k back-running trap. Across recent from-scratch runs the flip
decides roughly 2 of 5 outcomes and dominates any 8M comparison. The early phase is a bistable, chaotic
basin choice. A method that escapes or avoids the trap reliably is worth more than any late-phase gain.

**v45 headpop: an in-run population against the flip.** There are K=4 independent actors, each driving
4 of the 16 envs, with a shared critic and per-head duals. From 1M to 3M, every 250k steps the worst head
is replaced by the best (weights, duals and Adam state) when best - worst > 0.2|best|; at 3M all heads
collapse to the best. Eval and snapshots use the best head. FP32. If a single run flips with p~0.4, all
four heads flip with ~3%. Cost: each head gets 1/4 of the actor data (512-row minibatches); the
mb2048 arm keeps v36's 2048 rows per head. Jobs: reppo_headpop_v45, reppo_headpop_v45_mb2048.

**v45 headpop results (FP32, 8M):**

| Arm | @1M | @2M | @4M | @6M | @8M | last-20 | det. @8M |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref v27 H8 / v36 FP32 | 4174 | 5634 | 8596 | 10176 | 10894-11031 | 11.0k | 12.0-12.1k |
| K=4, 512 rows/head | 3788 | 4137 | 6784 | 8132 | ~8868 | 8593 | 9.9k |
| K=4, 2048 rows/head | 5347 | 6347 | 8554 | 9764 | ~11002 | 10978 | 12.3k |

The logged return averages all heads until the 3M collapse. None of the 8 heads flipped (per-head returns
at 1M: 3.8-5.3k). With p(flip)~0.4 per independent run, that would be a 2% event, so the population seems
to prevent the flip itself, likely through the shared critic's broader data, not only select it away.
The selection margin never fired before the collapse (heads within 20%). With 2048 rows per head (4x the
actor updates per sample of each head) early learning is much faster (+1.2k@1M, +0.7k@2M), but after the
collapse the run converges to the same plateau: the late phase stays entropy-bound. 512 rows per head
is too noisy. v46 adds v43's late drop (-0.5/dim to 50%, then to -1.5/dim) with and without the collapse.

**v46 popanneal (K=4, 2048 rows/head, late drop to -1.5/dim):**

| Arm | @4M | @6M | @8M | last-20 | det. @8M |
|---|---:|---:|---:|---:|---:|
| collapse at 3M | 8880 | 10491 | ~10941 | 10898 | 12.4k |
| no collapse | 9173 | 10730 | ~11020 | 11045 (±994) | 11.7k |

The late drop from scratch does not reproduce the fork gain inside 8M. The low-entropy phase also produces
occasional catastrophic episodes (2598, 6532) that the wide CIs reflect. The late phase stays near 11k.

**Why v45 learned faster early.** batch = 16 x 1024 rows; the actor takes 32 steps of 2048 rows per
iteration. At 2048 rows/head each v45 head takes exactly v36's steps with v36's batch size, on 1/4 of
the env data, yet each head learned faster than v36's single actor. The pathwise actor is not
data-limited; it is critic-limited. A shared critic fit on K diverse policies' data (broader action and
state coverage near each policy) gives each actor a better dQ/da. Test: scale the population with the
fixed -0.5/dim target and no collapse. K=4 (control), K=8, K=16 (one actor per env; 1024 rows/head,
all of its data).

**Population scaling (v46, fixed -0.5/dim, no collapse): refuted beyond K=4.**

| K | @500k | @1M | @2M | @3M | @4M | last-20 @8M | det. @8M | SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ref (K=1) | 2521 | 4174 | 5634 | - | 8596 | 11031 | 12.1k | 10.6k |
| 4 | 3207 | 5347 | 6347 | 8246 | 9138 | 10257 (±892) | 12.0k | 11.6k |
| 8 | 2735 | 4970 | 5518 | 6253 | 6930 | 10295 | 11.1k | 8.7k |
| 16 (1024 rows/head) | 1884 | 3920 | 4365 | 5839 | 6425 | culled 4M | 7.0k | 6.7k |

No head flipped at any K, but learning slows monotonically beyond K=4. Each head trains only on its own
block's states, so K=8/16 heads see 2/1 envs of state coverage. The "critic-limited" reading holds only
while each head keeps enough state coverage. At 8M no population variant beats the single-actor
reference (10.3-11.0k vs 11.0k): the population buys flip robustness and a faster first 2M, not a higher
8M score.

**Re-opened: TF32 may have produced the action-gap diagnosis.** v37/v38 and the v40-v42 fork screens ran on
the TF32 base. Branch-return contrasts are ~0.04 (scaled) on returns ~100, and TF32's relative error
(~5e-4) is ~0.05 at that scale: the model's "advantage correlation ~0.10" (v37) and v38's A ranking (corr
~0.02) were measured at or below TF32's noise floor. The v29 critic diagnostics (FP32) still show the
critic's own ranking is weak, but the model-branch route was never tested in FP32. Rerun from scratch in
FP32: reppo_dynabranch_v37_fp32, reppo_branchadv_v38_fp32. The fork "invariance" conclusions (v40-v42)
are likewise TF32 and provisional.

## v47 crossspread: no entropy, the spread reads an independent critic

Stepping back from entropy schedules. PPO's Beta needs no entropy bonus because its spread only drifts
when real outcomes correlate with deviation from the mode. REPPO's pathwise spread gradient reads the
critic's curvature at the mode, and the mode sits at that same critic's argmax, so the curvature there is
negative by construction: the optimizer's curse, applied to curvature. Spurious peaks look sharp and the
spread collapses (v19 Beta no-entropy: entropy -13.7, 7.0k; v21's likelihood-ratio spread also
collapsed, -8.4). The entropy target was a crutch for this bias. v47 applies Double Q to curvature: the
Beta mode climbs one critic and the concentration takes its pathwise gradient from an independent
critic whose noise peaks are not aligned with the mode. There is no entropy term, target or temperature,
and no clip; full Q and REPPO's per-state KL region are kept. Arms: crossfold (critics on env halves;
the spread uses the critic that never saw the state), independent (both on all data), and same (the
single-critic control; should collapse like v19). The probe spread_grad_narrowing_frac is the fraction
of states where the spread critic narrows the policy.

**v47 results: fast start, then a crawl; not the answer.**

| Arm | @500k | @1M | @2M | @4M | @6M | @8M | entropy @3M |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref v27 H8 | 2521 | 4174 | 5634 | 8596 | 10176 | 11031 | -3.3 (target) |
| crossfold | 4128 | 6375 | 7186 | 7574 | 8226 | ~8259 | -9.3 |
| independent | 3767 | 5721 | 6986 | 7975 | 8714 | ~9084 | -8.7 |
| same (control) | 3846 | 5203 | 5842 | 6349 | 6810 | ~7046 | -10.7 |

The independent spread critic slows the collapse (nu 117-144 vs 196 @3M) and helps (+2k over the
control), but 78-90% of states still narrow; the spread still collapses and learning crawls after ~2M.
Without entropy, Beta REPPO learns ~1.5x faster than the Gaussian reference early and then stalls. SPS
~5k (two critics, Beta rsample).

## Research reset (see OPTIMUM.md): abstract optimum, gaps, ablations

Diagnostics of v47's crawl: the KL gate always binds (kl 0.030, violation ~0.35), so the mode's step is
~0.1 sigma; as sigma collapses the step shrinks 3-5x, and the critic's held-out ranking decays to noise
(0.01), since it only sees actions within ~sigma. v26 dueling (FP32 base): flat 3.5-4.1k, so the
contrast-scale head is deprioritized. Running:
- Iteration rate (update rate as confound of the TD7 gap): v36 FP32 with rollouts of 128 or 256 steps
  per env, same passes per sample (reppo_iterrate_v36_ns128 / ns256).
- v48 stepgeom on v47-independent: box (sigma-free mode step), floor (nu cap, std >= 0.1), boxfloor.

**Results: iteration rate and v48 step geometry.**

| Run | @500k | @1M | @2M | @4M | last-20 @8M |
|---|---:|---:|---:|---:|---:|
| ref v27 H8 | 2521 | 4174 | 5634 | 8596 | 11031 |
| v36 ns128 (8x iterations) | 3724 | 4968 | 5981 | 8606 | 9670 |
| v36 ns256 | 2478 | 2887 | 3864 | 4398 | culled 5.8M (flip-like) |
| v47 indep (control) | 3767 | 5721 | 6986 | 7975 | 9084 |
| v48 floor (nu cap, std >= 0.1 at m=.5) | 3820 | 5874 | 7492 | 8271 | 9545 |
| v48 box / boxfloor | 10 | 205 | 839 | 2378 | culled (bug, see below) |

- Update rate is not the TD7 lever: 8x more policy iterations give +0.8k@1M and a lower 8M.
- The box arms are invalid as a step test. My precision bound (|d log nu| <= 0.02/iteration) is below the
  early narrowing rate (~0.04), so nu stayed 1.6-3.7 (near-uniform policy) and nothing was learned.
- **Law found: the critic's action ranking scales with behavior spread.** Held-out advantage correlation:
  0.25-0.29 at action std ~0.45 (box), 0.07-0.14 at ~0.3 (Gaussian ref), ~0 at 0.06-0.12 (Beta floor or
  collapse). Spread is information, and on-policy REPPO pays for it in the scored episodes. A 0.1 floor
  is too narrow to restore it (+0.5k). TD7 escapes the trade-off because replay holds many past
  policies' actions, so its critic gets action diversity without the current behavior paying for it.

## Distributional probe of the v29 H8 critic (8M / 16M snapshots, strict FP32)

Scripts and outputs: session scratchpad `zprobe.py`, `gt_analyze.py` (mlq 10504-10517). Units are scaled returns.
- Z(s,a) is a location family. Quantile contrasts correlate >= 0.96 with the E[Z] contrast (0.2-7% orthogonal
  variance); sigma_Z varies 1.2% across actions; sigma_Z = HL floor 1.5 plus an action-invariant excess ~0.9.
- Probability matching on Z: independent draws give 63 of 64 effective actions (uniform), comonotone 1.5
  (greedy). Neither is a usable policy; the return spread is target noise, not uncertainty about the argmax.
- Ground truth (128 states x 8 actions x 64 rollouts, noise-corrected): true contrast std 0.33 vs critic 0.11
  (3x compressed); corr(E[Z] contrast, truth) ~0.2 against a ceiling of ~0.7. No Z statistic beats the mean on
  typical states. Only in rare (<0.3%) high-sigma multimodal "bifurcation" states do pessimistic statistics
  (mean - sigma, q05) rank better. Common random numbers do not help: HalfCheetah trajectories diverge
  chaotically.
- Conclusion: the bottleneck is the accuracy of the mean contrast, not the distribution's shape.

## v49 drifttemp: temperature = rms change of critic contrasts per iteration (refuted)

T = c * rms[A_{k-1}(s,a) - A_{k-2}(s,a)] on fresh states, where A is the contrast Q(s,a) - E_pi Q(s,.)
(max-ent over a uniform base = probability matching when T is the contrast's standard error). c=1: 1135@500k,
2316@1M, 3626@2M, then decayed to ~1.4k; c=3: negative throughout (T runs away to 0.4-0.9). Entropy stayed at
3-4 (near uniform) in both. The measured drift equals the contrast std itself (0.031/0.031 early, 0.106/0.14
at 1.6M): one iteration of fitting rewrites the critic's action ranking about as much as the ranking is
worth. The honest per-iteration uncertainty is therefore ~uniform; greedy REPPO works because a slow actor
averages critic noise across iterations.

## v50 contrastanchor: prior on the previous iteration's contrasts (refuted)

kappa * mean (C - sg C_prev)^2 / (2 Var_Z) at 8 policy actions per state. kappa 3 / 7: 1.39k / 1.40k @500k,
1.82k / 1.93k @1M (ref 2.5k / 4.2k); cancelled. The anchor does stop drift (0.008 vs 0.049) but freezes the
contrasts small (std 0.022 vs 0.071): anchoring to a compressed estimate compounds the compression.

## v51 contrastdist: categorical head over standardized contrasts (G - E_pi Q') / s_y; actor climbs it

@500k 2.34k, @1M 4.34k (ref 2.5k / 4.18k). Mechanism: on its own training rows D reaches corr 0.25 with the
targets and std ratio 0.22, about the ceiling set by the per-sample SNR (~0.3), so fitting is not the
limit. On fresh states its corr is 0.08 (Q's 0.10). The contrast problem is generalization across states,
i.e. data-limited, which points to replay (v52).

**v51 at 3M: 8343 (last 20 eps) vs 6757 for the same code path with the actor on Z (v49b_dual); @2M 6971 vs
5634.** Reproducible bit-for-bit (v51_8m rerun identical through 1.3M). Control v51_qctl (D trained, actor on
Z): 2129@500k, 3382@1M, so the gain is the actor's signal, not D as an encoder aux.

## v52 replaywindow: W=4 rollouts, V-trace soft targets (sampled-action baseline), critic only (refuted)

@1M: 3106 (actor on D) / 2272 (actor on Z) vs 4355 / 4174; ~4.9k SPS with 4 concurrent runs. Held-out contrast
corr did not rise (0.08 / 0.06). Mean trace c 0.67 (0.58 on the oldest slot); 40% of replayed rows have
pi/mu > 1. Replay did not make the contrasts generalize on this budget.

**v51 at 8M: fast start, then a stall (the run died at 7.46M: disk full).** 8014@3M, 8445@4M, 8056@5M,
7608@6M, 7792@7M vs reference 7059 / 8596 / 9266 / 10176 / 10620. After 4M, D's standardizing scale s_y rises
0.54 (1M) -> 1.3 (4M) -> 2.25 (6M) and D's held-out contrast corr falls 0.08 -> 0.03; train corr 0.25 -> 0.16.
The contrast target's noise grows faster than the contrasts as the policy improves, so D loses resolution.
Open: is s_y growth state-level error (fixable with a learned V(s) baseline) or genuine return noise
(then the actor should combine the Z and D estimators, whose gradients agree only at cos ~0.5)?

## v53 contrastlambda: D's own lambda (refuted)

3M: lam 0.0 3425, 0.5 3512, 0.95 6661 vs v51 (0.8) 8343. Gradient cos(D, Q) ~0.5 in all arms; D's contrast
std is 0.4-0.5x Q's. v51's lambda = 0.8 is a sharp optimum or a lucky draw; the λ sensitivity itself is a
warning that v51's early gain is fragile.

## v54 ids: information-directed temperature (no entropy target)

Diagnosis: entropy collapse is the right answer to the wrong objective. For a Gaussian, Stein's identity gives
d/dsigma E Q(mu + sigma eps) = sigma E[Q''], which sees only curvature and never how far mu is from the
optimum. Any myopic spread gradient therefore collapses near a concave peak. An unbiased real-return
gradient collapses too, only more slowly: v21's LR spread reached entropy -8.4, and PPO-Beta anneals. Critic
bias (v47) changes the rate, not the fixed point. Spread is only worth what it teaches.

IDS on the local quadratic model sets this price. Minimizing regret^2 / information, with information
~ sigma^2 at our SNR, gives noise regret = offset regret, i.e. sigma = |offset to the optimum|. With max-ent
curvature h_i = T_i / sigma_i^2, this is the closed form T_i = sqrt(E_s[(sigma_i g_i)^2]). g is the slope of
the mean's max-ent objective on fresh states, and two half-sample means remove the Monte-Carlo inflation.
log T moves at most 0.15 per iteration. There is no entropy target and no schedule.

Arms:
- belief (default): the slope is that of the policy that chased the previous critic, under the current
  critic. It measures lag plus how far one refit moved the argmax (Thompson-like).
- lag: the current policy under the critic it chased (the red-team review predicts this collapses).

Jobs 10584 / 10585. Reference: v36 FP32 (10894@8M, T 0.003 -> 0.034, std 0.33 -> 0.51).

**v54 results (8M, seed 1): refuted as a temperature rule; it over-explores.**

| Run | @1M | @4M | @6M | @8M | last-20 | det. @7.8M | T late | std late | held-out corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref v36 FP32 | 4174 | 8596 | 10176 | 10894 | 11085 | 12.0k | 0.03-0.04 | 0.49-0.51 | 0.05-0.07 |
| v54 belief | 4019 | 7495 | 8339 | 9159 | 8901 | 9.9k | 0.08-0.09 | 0.60-0.64 | 0.13-0.15 |
| v54 lag | 3010 | 6677 | 7945 | 8720 | 8597 | 10.2k | 0.09 | 0.64 | 0.15 |

- The balance itself holds (offset regret tracks noise regret, 0.28 vs 0.24 late), and nothing collapsed:
  both arms stay wide. The red-team review's collapse prediction for lag was wrong.
- Spread buys ranking, as the v48 law predicts: held-out corr is 2-3x the reference's. It still did not buy
  a better policy. Even the deterministic return is 2k lower.
- The premise fails. The measured offset is not the kind that spread resolves. The dqda norm at the mean
  is ~1.0 late in both v54 and the reference, despite v54's wider data and better ranking. The belief and
  lag arms give the same T, so the slope is structural (the KL gate binds on 35% of states; a slope held by
  the trust region, plus critic error that grows with Q scale), not identification noise. IDS assumes the
  information bought by sigma is about this offset. Here it is not, so T tracks Q's scale and rises
  through training (0.04 -> 0.09) instead of annealing.

## v55 regret probe: where v54's balance is wrong

v55 applies Stein's identity to the rollout's own noise, with hard lambda-0.95 returns minus E_pi Q^. This
gives real-return batch moments at the behavior mean, per dimension, from single samples, and the same
estimator applied to the critic's contrasts. Summed over dimensions, reference regime (entropy dual; the
run reproduced 11.0k):

| | 1M | 2M | 4M | 7.8M |
|---|---:|---:|---:|---:|
| true curvature E[sigma^2 Q''] | -0.054 +- 0.025 | -0.137 +- 0.037 | -0.225 +- 0.076 | -0.232 +- 0.087 |
| critic curvature (same samples) | -0.047 | -0.112 | -0.195 | -0.221 |
| max-ent identity -sum T | -0.038 | -0.110 | -0.206 | -0.241 |
| beta = real / critic slope at mean | 0.50 | 0.71 | 0.40 | 0.28 |

- The critic's curvature at its own argmax is not biased: it matches the real return's within noise. The
  mean sits on a real concave peak. "Collapse is a critic-curvature bias" is refuted in this regime, and a
  real-return (PPO-style) spread gradient would narrow at the same rate (consistent with v21). PPO
  explores because its mean moves at the evidence rate and sits on slopes, where real curvature is ~0.
- v54's error is entirely on the slope side. Only beta of the critic's slope is real, so v54 priced
  1/beta^2 = 2-12x too much offset energy. The square root of that is its measured 2-3x temperature excess.
- The ids-arm probe (10594) flipped (2.3k trap), so its moments are not the reference regime.

v56 calibids (job 10598): T = beta sqrt(E[|sigma g^|^2] / d). On the probe's numbers this is ~0.035 late,
equal to the hand-tuned target, and 0.02-0.04 before 2M, where the reference's dual fell to 0.003.

## v56 calibids: calibrated IDS price (regression)

T = C⁺ / sqrt(d·E): the IDS offset price discounted by the calibration β = E[σĝ·εA]/E[σ²ĝ²] (the share of
the critic slope that real returns confirm). Job 10598, cancelled at 7.7M.

| run | @1M | @2M | @4M | @6M | last-20 | T @1M/4M/7.8M | β @1M/4M/7.8M |
|---|---:|---:|---:|---:|---:|---|---|
| v56 calibids | 3615 | 4943 | 6890 | 8936 | 9531 | 0.045/0.063/0.073 | 0.60/0.46/0.34 |
| v55 dual (ref dynamics) | 4267 | 5839 | 9085 | 10323 | 11014 | ~0.003 → 0.035 | 0.50/0.40/0.28 |

Calibration fixed v54's 1/β² over-pricing on paper (it predicted T≈0.035 from v55-dual's probe statistics),
but closed-loop T settles 1.5-2x higher and, worse, 15x higher before 2M: a higher T keeps σ wide, which keeps
the measured slope energy and β up, so the rule prices its own noise. Every offset-regret temperature (v54,
v56) loses to the entropy-target dual. Next: measure the true noise-cost curve J(c) and the return along the
critic's ascent directly (scripts/landscape.py) instead of inferring them from first-order identities.

## v57 actiongeom: fixed physical-mode trust geometry (three-minute trial)

Base: v47-independent. Replace full Beta KL as the joint gate/dual distance with
`0.03 * mean((mode - old_mode)^2) / 0.02^2 + KL_precision`, where precision KL holds
mode at its old value. Pure precision updates retain v47's exact KL and gradient;
there is no log-concentration box, floor, or entropy term. Mode/precision share the
original trust budget. Critic architecture, optimizer and sample/update counts stay fixed.

Job **10610**, HalfCheetah-v4, seed 1, strict FP32 CUDA/compiled, max-parallel-runs **4**,
**3m wall limit including startup**. Queue terminated it with SIGTERM at the time
limit (marked failed due to timeout, not a learner exception). Logged 2,047,008 steps;
last complete update metrics at 2,031,232. Autocull supervised only this prefix and
was stopped after the job ended. Snapshots retained through 1,015,424 steps.

| run | @500k | @1M | @1.5M | @2M |
|---|---:|---:|---:|---:|
| v47 independent (historical) | 3767 | 5721 | 6819 | 6986 |
| v57 actiongeom | 2518 | 4459 | 5396 | 5797 |

Matched-step windows from score_runs.py. Update-averaged mode RMS stays ~0.018-0.019
while std falls from ~0.241 at 500k to ~0.072 at 2M. Mode RMS is intermediate
minibatch displacement relative to the old actor, not final-policy displacement.
Deterministic eval: 6311 at 1.982M versus historical v47's 7543 at 2.064M (nearby,
not identical steps/evaluation seeds). The fixed 0.02 budget is restrictive early;
by 2M concentration still narrows and larger permitted mode movements have not
recovered the performance gap. This setting regresses through 2M. The trial does
not resolve the late-plateau hypothesis, because it stops before that regime;
no additional runs were launched. Three mathematical contract tests passed and an
independent implementation review found no blocking issue.

## Return landscape of the reference (scripts/landscape.py)

Reference dynamics: v55 --temperature-source dual (deterministic; reproduces 11013.6 exactly), snapshots
one iteration apart at 1.02M, 4.01M, 7.91M. 32 episodes per point, common random numbers (paired se).

Noise cost J(0) - J(1) (policy noise scaled by c; c=0 deterministic): 5.0% @1M, 12.3% @4M, 11.0% @7.9M
(PPO sepclip @12M: 9.3%). The cost is ~quadratic in c (the Stein/curvature model of noise cost holds), and the
critic's surrogate ranks the whole c curve correctly at every stage: noise is justified only by information.

One-iteration update slice (c in units of the actual step, 1 = the step taken):

| snapshot | true peak (det / stoch) | slope at 0 (det) | critic's peak | curve corr | cos(critic grad, step) |
|---|---|---|---|---|---:|
| 1M | 7.8 / 5.4 (rises to 6x) | +32 +- 4 | > 6 | 0.96 | 0.07 |
| 4M | ~0.5-1.5, cliff at 2x | +15 +- 6 | ~1.5-2 | 0.40-0.50 | 0.03 |
| 7.9M | ~0 (-0.3 / -0.5) | -23 +- 8 | ~1.2 | 0.80-0.86 | 0.008 |

The direction the pathwise update takes is good, but its length is miscalibrated: ~5x too timid early, and
late the whole step is critic error (the critic predicts gain where the true slope is <= 0). The raw critic
gradient is almost orthogonal to the step Adam + the trust region actually take, and along it any step of
update size destroys return (sharp ridge; random filter-normalized directions are ~1000x flatter).
-> v58 evidencestep (real-return line search on the step length) and v59 scalarq (TD7 scalar critic; HL-Gauss
bins of 2 scaled units vs 0.02-0.1 unit action contrasts).

### v57 diagnosis: miscalibrated geometry, not a clean plateau ablation

Exact scalar Beta KL calculation (six identical action dimensions, fixed nu,
centered old mode) gives a v47 mode RMS radius **0.11421** at initialization
(nu = 2 log 2, KL = 0.03). V57's 0.02 radius is **5.71x smaller**. The Gaussian
approximation misses this: the sampled physical mean is mode * nu/(nu+2), so it
moves only 0.4094 times the mode at initialization. With centered mode and nu=100,
v47's radius is 0.01005; at nu=250 it is 0.00634. At mode=0.6 and nu=250, a 0.02
mode move has exact KL **0.47069**, consistent with v57's large late logged KL.
These are pure-mode illustrative calculations, not observed final-policy steps.

TensorBoard windows, same steps in each run:

| metric | v47 independent 0–250k | v57 0–250k | v47 independent 1.90–2.03M | v57 1.90–2.03M |
|---|---:|---:|---:|---:|
| mean absolute mode | 0.359 | 0.076 | 0.638 | 0.663 |
| action std | 0.433 | 0.428 | 0.085 | 0.068 |
| entropy | 2.921 | 2.920 | -8.01 | -11.73 |
| nu | 2.45 | 3.52 | 99.4 | 264.3 |
| actual Beta KL | 0.057 | 0.023 | 0.0277 | 0.513 |
| action saturation fraction | 0.0016 | 0.0012 | 0.0356 | 0.1318 |

KL and mode-displacement metrics average intermediate optimizer evaluations; they
are not final-policy displacement or guarantees. Both late advantage correlations
are weak and window-sensitive (0.011 vs 0.017 in the final window), so these data do
not establish that v57's critic is uniquely worse. Support clipping is zero in both.

The early regression is consistent with imposing a much tighter mode constraint
while concentration remains free to rise. The later metric treats large distribution
changes as small distances, especially near action bounds. It also discards Beta's
state/action anisotropy and joint mean/concentration geometry. Detaching outputs
does not separate shared-trunk parameter updates; preserving a pure-precision KL
identity does not preserve the precision trajectory or budget in a joint update.
Consequently the three-minute result **does not refute** the late-step hypothesis.
The original formula tests passed but failed to assess baseline calibration.

Rethink: retain v47's full Beta KL as the control geometry. Any mode-relaxation
ablation must preserve its original feasible region, and log both proposed and
final displacements, sampled-mean change, precision change, actual KL and tails.
A union gate (`min(full_KL, action_distance)`) would meet feasible-set containment
and remove v57's early restriction, but would still permit extrapolation and is
only a causal ablation, not an evidence-backed resolution. The substantive next
question is whether critic-proposed improvements survive a frozen-policy paired
simulator test and where trustworthy action evidence ends. Repair that evidence
before adopting a larger step as the solution. No further training was launched.

## v58 evidencestep: real-return line search on the step length (regression)

Job 10617, cancelled at 4.3M: 3817 / 4838 / 6023 @1/2/4M vs reference 4174 / 5634 / 8596. Multiplier hit
the grid cap (3) early, then sat at 1.1-1.5 (realized KL up to 9x the bound); per-iteration optima swung
0..3. Two causes: (1) away from the behavior the importance-weighted surrogate ignores the state-distribution
shift, so its curvature is optimistic and the fit is biased toward long steps (red-team finding); (2) the
one-iteration landscape optimum is myopic: repeating a longer step every iteration compounds, moving the
policy off the data the critic was fit on, which a single-step slice cannot see. The one-step landscape is a
diagnostic of calibration, not a target for step control.

Existing independent landscape measurements reinforce the rethink (Gaussian
v55-dual reference, **not v47/v57**): frozen one-iteration update slices with 32
paired evaluation episodes show that at 1.015M multiplying the update by 5 improves
deterministic return by **122.8 +- 13.8 SE**, whereas at 7.913M multiplying by 2
reduces it by **188.9 +- 50.3 SE**, even while the frozen critic's regularized
surrogate increases. Reports are `runs/landscape/*v55dual_pairs*/update_all.json`.
The stored quadratic fits use the entire coefficient grid (-2 through 6), not a
local derivative estimate; their peaks/slopes must not be treated as direct local
measurements, and their nominal parameter errors omit cross-point CRN covariance.
These slices do not identify v47's plateau, but contradict a universal larger-step
remedy in the reference regime. The stronger research direction is evidence-
calibrated policy improvement: let the critic propose a direction and independent
real outcomes determine whether and how far to follow it. This needs held-out data,
importance-ratio/ESS checks, and uncertainty-aware step selection; current-policy
Q optimism is not an independent validation signal.

## v59 scalarq (regression, 8.3k @8M)
Scalar Q head with Huber (delta 1, raw units) and no PER: Huber on these residual scales is L1, i.e. median
regression, which is not the mean-consistent Bellman target. Uninformative about scalar-vs-HL-Gauss; an MSE
head is the fair test.

## v60 noisefork
Checkpoint/resume harness. Base run reproduces v55-dual exactly (11013.6), so resume is transparent. Fork
experiments parked: red-team found dose uncontrolled, n=3 underpowered (MDE ~1000+), and arms change the
objective, not only the behavior noise.

## v61 exacttarget
REPPO's lambda-return target carries the true advantages of sampled future actions (irreducible; oracle:
~90% of a return advantage's variance). Q(lambda)-corrected target
G_t = r_t + gamma [V'_{t+1} + lam (G_{t+1} - Q'(s_{t+1}, a_{t+1}))] leaves only critic error under
deterministic MuJoCo, so its fixed point is soft Q^pi for every lambda: on-policy without a bias/variance
trade. Implemented as GAE with values = Q'(s, a_taken). V' now has weight 1 (vs 1 - lam), so its Gaussian
entropy expectation is made exact (Rao-Blackwell), in both modes. New metric bellman_rms_heldout.
Arms: lam 0.8 (10626), lam 0.95 (10627), --no-exact-targets control (10628; also isolates the RNG shift the
new diagnostic introduces). Reference: v60 base / v55-dual 11013.6.

### v61 results (from scratch, seed 1)
Control (--no-exact-targets) and lam 0.8 both flipped (~2.1k plateau); control differs from v60-base only in
RNG consumption, so seed-1 from-scratch runs are a flip lottery. lam 0.95: 9715 @8M (base 11014), final
advantage_corr_heldout -0.03 (base 0.14).

## v62 exactfork: v61 targets resumed from v60-base's upright 1M checkpoint (flip-free pairing)
| arm | 2M | 3M | 4M | 5M | bellman_rms @5M |
|---|---|---|---|---|---|
| v60 base | 5839 | 7848 | 9085 | 9856 | - |
| control (uncorrected) | 5658 | 7896 | 8762 | 9994 | 1.11 |
| exact lam 0.8 | 5824 | 8073 | 9139 | 9749 | 1.22 |
| exact lam 0.95 | 5533 | 7492 | 8738 | 9234 | 1.02 |
No effect beyond fork noise, and the held-out one-step Bellman residual (scaled units; true within-state
advantage ~0.03, HL-Gauss bin width 2.0) is unchanged: removing target noise does not make the critic more
consistent. The critic's error is fitting/representation, not target variance. Tests pinning the target math:
tests/test_reppo_exact_targets.py (exact Q^pi recovery for every lambda on a solved deterministic MDP).
Final @8M (last 20 eps +-CI95): control 10174+-866, lam 0.8 10458+-460, lam 0.95 10570+-303, v60 base
11014+-567. Arms within fork noise (control vs base alone differ by 840). Verdict: exact targets are neutral;
target variance is not REPPO's bottleneck.

## Critic oracle (scripts/reppo_critic_oracle.py, v60-base @1M; job 10632, cancelled after snapshot 1)
Truth by MuJoCo branching (CRN, 512 states x 8 fresh actions x 64 rollouts, H 400). Scaled units.
- Level: critic Q vs true corr 0.93 across states, err sd 0.90 vs between-state sd 2.3.
- Within-state: true sd 0.22 (~0.1 HL-Gauss bin), critic sd 0.21, corr 0.11 (0.15 noise-corrected);
  critic's best of 8 gains only +0.27 within-state sd.
- Action gradient (pre-tanh): slope of true directional derivative on critic's 0.37+-0.12 (h 0.1),
  0.21+-0.05 (h 0.3): the pathwise signal is ~20-37% real.
- Critic's Q(s, a) vs its own r + gamma V'(f(s, a)) within state: corr -0.33 (self-inconsistent).
- One-step model route (true f, critic V'): same slope, ascent ratio 0.79 vs 0.55 at h 0.3: V's state
  gradient is inaccurate too, so routing through V is no fix on its own.
- Chaos: branches decorrelate within ~100 steps; FD truth noise ~ sd(Q)/h.

## v63 orthoadv: orthogonal advantage critic (see header)
Q = V(s) + A(s, z), z = behavior noise (independent of s), A quadratic in z (slope + diagonal curvature), V at
tanh(mu_old). Removes the state-error leak into the action slope by construction. Jobs 10639 (fork from
v60-base 1M) and 10640 (from scratch).
v63 results: fork 4876 @2M / 5783 @3M vs v62 control 5658 / 7896; from scratch 4439 @3M (base 7848).
Regression; both cancelled. Reading: orthogonal A is unbiased but variance-limited, and REPPO's lambda-return
feeds it the future-action noise the exact target removes (v62 was neutral because the leaky critic was
bias-limited). Arms: v63 --exact-targets lam 0.8 / 0.95 forks (jobs 10643, 10644).
v63 + exact targets: lam 0.8 5836 @3M, lam 0.95 5258 @3M (v63 5783, control 7896). Cancelled. The variance
explanation is refuted: lower-noise targets do not rescue the orthogonal A. Whatever the leaky Qnet slope
carries (between-state action covariation, global nonlinear structure, saturation), the local quadratic A in
behavior coordinates loses more than the de-biasing gains. Unmeasured: A's own gradient accuracy (needs an
oracle adapter for Q = V + A).
