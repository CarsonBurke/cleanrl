# 03 — Independent re-analysis of the REPPO diagnostic artifacts

Role: data analyst. Everything below is **recomputed from raw arrays** unless tagged `[doc]` (read from
`FAMILY.md` / `analysis.md` / a saved JSON summary). No training, no GPU, no simulation was run. Units are the
trainer's scaled returns (raw reward x 0.1). CIs are 95% bootstrap over oracle states (1000-4000 draws) unless
stated; "corrected" means the oracle's own rollout noise is removed from the truth's variance.

Scripts: `benchmarks/reppo_rootcause/scripts/03_*.py`. Logs and tables: `benchmarks/reppo_rootcause/data/03_*`.

Raw sources used

| Tag | Snapshot | Truth | Has the on-policy chunk? |
|---|---|---|---|
| 1M | v60 base @1.015M, lambda 0.8 (`runs/reppo_critic_oracle/*.npz`) | 512 states x 58 actions x 64 CRN rollouts, H 400, plus n-step checkpoints t = 1..400 | no (and no observations saved) |
| 4M / 12M / 30M | v23 probe, lambda 0.95 (`runs/reppo_oracle/*.pt`) | 512 x 58 x 32, H 300 | yes: 16384 (s, a, r, s', lambda-return target) |
| 8M / 12M | v29 H8, lambda 0.8 (`runs/reppo_latentroll_oracle/*.pt`) | 512 x 58 x 32, H 300 | yes |
| landscape | v55 dual @1.02M / 4.01M / 7.91M (`runs/landscape/*`) | 32 CRN episodes per point | - |
| runs | 193 tfevents runs `runs/HalfCheetah-v4__reppo_*` | - | - |

The oracle chunks are a fresh rollout of the frozen snapshot policy with trainer-exact targets, so they are
exactly one iteration of critic training data, and held out for the snapshot critic. My reconstruction of
normalized obs, actor moments and critic values on them matches the saved oracle values to 1e-4 or better
(`03_prep_oracle_chunks.py`).

---

## Bottom line

1. **On the channel the Q critic actually uses (regress the lambda-return on (s, a)), the problem is
   ESTIMATION-limited, and the critic is at or above what I could extract from the same data.** One on-policy
   sample carries 4-7% advantage signal (4M-8M). Closed-form estimators fit on one 16384-sample chunk reach a
   within-state correlation of 0.02-0.18 with the truth; the snapshot critic reaches 0.15-0.29 on the same
   states. The same estimators recover 0.83-0.95 of a *planted* smooth advantage of identical variance, so the
   shortfall is not estimator weakness: at most ~4% of the true within-state advantage variance is smooth,
   low-order structure. Adding the chunk's information to the critic changes its score by 0.000 to -0.015 (every CI inside +-0.04).
2. **There is a second channel in which the information exists and the Q readout misses it by 2-6x** (1M only):
   the same critic's state value composed with the true dynamics ranks actions at 0.34-0.43 after one real
   step and 0.87 after five, against 0.15 for Q(s, a) directly. The true advantage is fully formed within
   ~10 steps (corr of the 10-step truth with the 400-step truth: 0.995). A linear action-effect model fit on
   one chunk does not capture this channel (0.01-0.21, at or below the critic).
3. **Several published oracle numbers are heavy-tail artifacts.** At 1M, 1% of states hold 89% of the critic's
   within-state energy. The "self-consistency -0.33" is +0.41 [+0.34, +0.47] once those states are removed;
   "critic sd 0.21 = true sd 0.22" is 0.046 vs ~0.2 on the median state; the gradient slope 0.37 is measured
   on ~26 states.
4. **The logged critic-quality metric is not predictive of return** — it is negatively rank-correlated with it
   across healthy runs (rho -0.26 / -0.34 / -0.32 at 1M / 4M / 8M). It moves with data per policy step
   (smaller KL bound, 4x longer rollouts), which is itself the signature of an estimation-limited quantity.

---

## Q1. Variance decomposition and the noise ceiling

### 1a. Decomposition (variance, scaled units^2; sd in parentheses)

| | 1M (v60, lam .8) | 4M (v23, lam .95) | 8M (v29 H8, lam .8) |
|---|---:|---:|---:|
| Between-state variance of Q_true | 5.37 (2.32) | 16.2 (4.03) | 63.7 (7.98) |
| Within-state (action-caused) true variance | 0.050 (0.223) | 0.106 (0.326) | 0.199 (0.446) |
| Target noise: lambda-return minus Q_true(s, a0) | 0.093 sampling only, approx. (lam .8); 0.41 (lam .95) | 1.52 (1.23) | 3.82 (1.95) |
| Monte-Carlo single-rollout variance (H 300-400) | 2.79 | 8.14 | 25.5 |
| Critic error variance at a0 (state level) | 0.82 (0.90) | 1.62 (1.27) | 4.55 (2.13) |
| Critic within-state variance, pooled | 0.042 (0.206) | 0.112 (0.334) | 0.039 (0.196) |
| Critic within-state sd, median state | 0.046 | - | - |
| Var(target - Q_critic) over the whole chunk | - | 1.32 | 1.42 |
| Critic bias vs truth | -0.60 | -1.01 | +2.74 |

Notes. 4M and 8M target noise is measured directly (trainer targets at the 512 oracle states minus truth, oracle
SE removed), so it contains bootstrap error as well as sampling noise. 1M has no chunk; its figure is the
rollout-to-rollout variance of a lambda-mixture built from the saved n-step checkpoints (coarse: 8 horizons).
The analytic future-action term sigma_A^2 c^2/(1 - c^2), c = gamma lambda, gives 0.084 (1M, lam .8), 0.81 (4M,
lam .95), 0.33 (8M, lam .8): at 4M and 8M most of the measured target noise is therefore NOT future-action
sampling noise but state-dependent bootstrap error. The 12M and 30M datasets contain near-flip states that
inflate every pooled variance (between-state variance 659-1183); I do not tabulate them.

The critic's state-level RMSE stays near 1.3-1.7% of the value level through training (logged `q_mse`: 0.14 at
Q~22, 1.5 at Q~95), while the true within-state sd grows only 2x. The ratio sigma_A / critic RMSE falls from
~0.6 at 1M to ~0.36-0.44 at 4M+.

### 1b. Information per sample and effective sample size

- Per-sample advantage signal share sigma_A^2 / (sigma_A^2 + target noise): 0.066 at 4M, 0.050 at 8M
  (single-sample correlation ceiling 0.26 and 0.22). At 1M, with sampling noise only: 0.24 (lam .8), 0.087
  (lam .95).
- Autocorrelation (measured on the chunks). The target residual is strongly autocorrelated along a trajectory:
  lag-1 rho 0.71-0.89, ESS factor 0.026-0.083, i.e. **N_eff ~ 430-1360 of 16384 for anything that is a function
  of state** (the value level). The score products (y - b) z_d, which are what an action-gradient or advantage
  regression averages, are white: ESS factor 0.95-0.96, **N_eff ~ 15.6k**. This matches the argument that the
  policy noise z_t is independent across time, so (noise_t x z_t) is a martingale difference.
- Total information for the advantage: N_eff x SNR ~ 15.6k x 0.05-0.07 ~ 800-1100 degrees of freedom
  resolvable at SNR 1. Analytically, a gradient field needing P free features per action dimension is recovered
  at corr 0.9 / 0.7 / 0.5 / 0.1 only while P <= 81 / 357 / 1030 / 34000 (8M numbers; 4M similar).

### 1c. The decisive test: what can be extracted from one chunk?

Estimators see only (obs, action, lambda-return target) of the 16384-sample chunk — the critic's training
signal for one iteration, with no simulator truth. A_hat(s, a) = sum_d z_d (phi(s) . W_d), z the policy's own
noise; baseline a random-feature ridge on obs. Scored leave-env-lanes-out against the same truth as the critic
(within-state correlation over 8 fresh actions, noise-corrected).

| Estimator (free parameters) | 4M | 8M | 12M (v29) | 30M |
|---|---:|---:|---:|---:|
| **Snapshot critic** | **+0.285 [+0.20, +0.42]** | **+0.200 [+0.07, +0.33]** | **+0.154 [+0.08, +0.25]** | **+0.233 [+0.14, +0.36]** |
| Constant gradient (6) | +0.06 | +0.02 | -0.04 | -0.02 |
| State-linear gradient (108) | +0.08 to +0.11 | +0.10 | +0.01 | +0.08 |
| Random-feature gradient (876) | +0.08 to +0.11 | +0.07 to +0.12 | +0.01 | +0.05 |
| Constant gradient + curvature (27) | +0.184 [+0.13, +0.25] | +0.130 [+0.06, +0.20] | +0.04 | +0.05 |
| State-linear gradient + curvature (486) | +0.17 | +0.12 | +0.01 | -0.03 |
| kNN local-linear, best k | +0.08 | +0.07 | -0.00 | +0.10 |
| Critic + kNN correction fit on (target - Q_critic), k 2048 | +0.285 (delta +0.000 [-0.02, +0.01]) | +0.201 (delta +0.001 [-0.02, +0.02]) | +0.139 (delta -0.015) | +0.234 (delta +0.001) |

Paired difference, best linear-in-z estimator minus critic: -0.20 [-0.36, -0.11] at 4M; -0.08 [-0.23, +0.06]
at 8M. No learning-curve trend is resolvable between N/4, N/2 and N (all within noise of zero).

Positive control (`03_q1c_planted_control.py`): plant a known advantage of the measured within-state variance
into the real targets and recover it with the same pipeline.

| Planted field | ridge const | ridge linear | ridge RF | kNN 2048 |
|---|---:|---:|---:|---:|
| constant gradient (4M / 8M) | 1.00 / 1.00 | 0.95 / 0.93 | 0.84 / 0.79 | 0.90 / 0.81 |
| state-linear gradient | 0.49 / 0.26 | 0.95 / 0.93 | 0.89 / 0.90 | 0.77 / 0.66 |
| smooth nonlinear (random tanh net) | 0.70 / 0.70 | 0.93 / 0.92 | 0.85 / 0.83 | 0.81 / 0.78 |
| rough (2000 high-frequency features) | 0.00 / 0.03 | 0.15 / 0.12 | 0.21 / 0.13 | 0.11 / 0.11 |

The real advantage behaves like the "rough" row. Reading: if the real advantage were a state-smooth gradient
field, one rollout would give a correlation of ~0.9 and the critic's 0.2-0.3 would be far below the ceiling.
It is not. The smooth, low-order share of the true within-state variance is at most (0.18 / 0.9)^2 ~ 4%; the
state-independent share of the true gradient energy is 0.00 (4M) and 0.00 (8M).

### 1d. The logged metric's own ceiling

`advantage_corr_heldout` correlates the critic's advantage with a single lambda=0.95 return minus the same
8-sample baseline. A perfect critic would not score 1. On the oracle chunks the perfect-critic value is
+0.24 [-0.03, +0.47] (4M), +0.39 [-0.01, +0.70] (12M v29), cruising-state 8M +0.41 [+0.09, +1.2]; these are
very imprecise because the truth for a single action has reliability 0.1-0.6. Indicatively the logged
0.07-0.13 is a third to a half of its ceiling. The shared sampled baseline inflates the logged value by about
std_ratio / 8 ~ 0.02.

### 1e. Answer

- (a) Held-out advantage correlation: the critic's true within-state correlation is 0.15 [0.13, 0.28] at 1M,
  0.29 [0.20, 0.42] at 4M, 0.20 [0.07, 0.33] at 8M, 0.15 [0.08, 0.25] at 12M. It is **at or above** every
  estimator I could build from one rollout, including ones proven able to recover a smooth signal of the same
  size. On this channel the gap to 1.0 is missing information, not failed optimization.
- (b) Action gradient: with N_eff ~ 15.6k and SNR ~ 0.05-0.07, a gradient field is recoverable to cosine 0.7
  only if it has under ~350 degrees of freedom per action dimension. The measured field has no detectable
  state-independent component and is not captured by 108-876-parameter models, so its accuracy from one
  rollout is bounded near the 0.1-0.2 the critic shows.
- Caveat: this bounds what I tried, not every conceivable estimator, and the critic also integrates earlier
  rollouts. It rules out the reading "the signal is easy and the network fails to fit it".

### 1f. The second channel: transitions x state value (1M only, true dynamics)

The 1M oracle saved the n-step truths (t real steps, then the critic's own V'). Same states, same fresh actions:

| Predictor of the within-state advantage | corr with H=400 truth (top 5% predictor-energy states removed) | all states | per-state Spearman |
|---|---:|---:|---:|
| Q(s, a) direct | +0.147 [+0.07, +0.21] | +0.149 [+0.13, +0.28] | +0.087 +- 0.019 |
| r + gamma V'(f(s, a)), 1 true step | +0.339 [+0.28, +0.40] (1% trim: +0.43) | +0.055 [-0.20, +0.52] | +0.194 +- 0.019 |
| 5 true steps then V' | +0.844 [+0.78, +0.91] | +0.872 [+0.83, +0.93] | +0.429 |
| 10 true steps then V' | +0.862 [+0.81, +0.93] | +0.909 [+0.87, +0.96] | +0.538 |

(Per-state Spearman is attenuated by truth noise; a perfect predictor would score about 0.61.) The same holds
for the pre-tanh gradient: pooled corrected corr 0.32-0.35 for the one-step lookahead's finite difference vs
0.12-0.17 for the critic's autograd (5% trim); true rate of improvement along the unit direction 0.31 [0.23,
0.38] vs 0.14 [0.08, 0.21] at h 0.3.

Split-half disattenuated correlation of the t-step truth with the 400-step truth: 0.06 (t=1, all states),
0.90 (t=5), 0.995 (t=10), 1.00 (t>=25); the within-state sd is 0.21-0.25 at every t. **The true advantage is a
short-horizon quantity**; steps beyond ~10 add rollout noise and no within-state signal.

So the critic's state-value function is good enough to rank actions well when given real next states, and its
Q(s, .) readout does not reproduce that. This is the one place where the data say REPRESENTATION, with two
limits: it is measured at 1M only (the multi-snapshot oracle run was cancelled), and it needs counterfactual
next states. A cheap learned substitute fails: a state-linear action-effect ridge fit on one chunk, composed
with the critic's V' (`03_q1f_learned_lookahead.py`), scores +0.21 [+0.15, +0.32] at 4M (critic +0.285),
+0.01 at 8M, -0.01 at 12M, +0.16 at 30M, with held-out R^2 of the action-caused next-state residual only
0.05-0.43. `[doc]` v30 / v37 / v38 learned-model routes were also negative.

---

## Q2. Which interventions moved the critic-quality metrics?

Table of all 193 runs: `data/03_q2_table.tsv` (return, advantage corr, std ratio, Bellman rms, q_mse, dqda norm,
policy std, KL, violation, temperature at 1M / 4M / 8M, and config delta vs the modal hyperparameters).
118 runs log `advantage_corr_heldout`; 10 log `bellman_rms_heldout`.

Distribution across runs: advantage corr median 0.080 / 0.080 / 0.070 at 1M / 4M / 8M, interquartile range
0.06-0.11, 90th percentile 0.12-0.13. Std ratio median 0.13-0.16, IQR 0.12-0.19. Almost every intervention sits
inside this band.

What moved it:

| Direction | Run | corr 1M / 4M / 8M | return 8M | Config delta |
|---|---|---|---:|---|
| up | v23 kl_bound 0.01 | 0.159 / 0.163 / 0.154 | 7585 | KL bound 0.03 -> 0.01 |
| up | v23 kl_bound 0.015 | 0.119 / 0.132 / 0.085 | 7857 | |
| (ref) | v23 default (0.03) | 0.098 / 0.106 / 0.044 | 10580 | |
| down | v23 ns4096 kl_bound 0.12 | 0.102 / 0.036 / 0.012 | 7416 | |
| up | v25 lam 0.8, num_steps 4096 | 0.200 / 0.169 / 0.144 | 8107 | 4x rollout, vs 0.064 / 0.086 / 0.111 at 1024 |
| up | v54 ids lag / belief | 0.167 / 0.168 / 0.162 | 8711 | |
| up | v43 ent_target_mult 0 | 0.121 / 0.144 / 0.145 | 8320 | |
| up (degenerate) | v48 box / boxfloor | 0.285 / 0.259 / - | flipped (2374 @4M) | |
| down | v26 duel | 0.00-0.03 / -0.12-0.00 / -0.08-0.08 | 3464-4134 | ratio 0.02-0.05 |
| down | v47 crossspread, v48 floor, v22 clipgauss | 0.01-0.06 | 7079-9407 | |
| none | v61 / v62 exact targets | 0.05-0.07 @4M, 0.03-0.07 @8M | 9802-10630 | Bellman rms 0.93-1.13 @4M in all arms |

A dose-response exists in exactly one family of knobs: **more data per unit of policy change raises the
metric** (KL bound 0.03 -> 0.015 -> 0.01 gives 0.106 -> 0.132 -> 0.163 at 4M; within the 4096-step pair, KL bound 0.03 -> 0.12
gives 0.079 -> 0.036; a 4x rollout gives 0.064 -> 0.200 at 1M in v25, 0.098 -> 0.150 in v23). Every one of those runs has lower return at matched steps. Nothing that changed the
critic's loss, target, architecture or auxiliary task moved the metric upward out of the band.

Is the metric predictive of return? Spearman across runs:

| Metric @T vs return @T | all runs | healthy runs (above the flip trap) |
|---|---|---|
| advantage corr, 1M / 4M / 8M | -0.15 / -0.19 / -0.27 (n 99 / 81 / 77) | **-0.26 (p .03) / -0.34 (p .009) / -0.32 (p .009)** (n 65 / 57 / 65) |
| std ratio, 4M / 8M | +0.19 / -0.01 | -0.40 (p .002) / -0.40 (p .001) |
| q_mse, 4M / 8M | +0.29 / +0.35 | +0.42 / +0.26 (scale effect: error grows with value) |
| dqda norm, 1M / 4M | +0.71 / +0.78 | +0.46 / +0.60 (scale effect) |
| temperature, 4M | +0.52 | +0.60 |
| KL violation fraction, 4M | -0.45 | -0.37 |
| Bellman rms | n = 10, rho +0.04 | - |

The critic-quality metric is **anti-predictive** of return among healthy runs. Runs are one seed each and
belong to families, so p-values overstate independence; the sign is consistent at all three steps.

---

## Q3. Trends within the reference runs

Per-1M-bin tables for v27 H8 (bit-identical to the v29 H8 arm), v29 parent, v29 no-reward, v36, v55 dual, v60 base
(identical to v55 dual through 8M), v61 lam .95, v62 arms: `data/03_q3_log.txt`.

v27 H8 / v60 base, representative bins:

| M steps | return | det. eval | d return / M | q_mse | adv corr | policy std | temperature | KL mult. | actor grad norm | KL viol. frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 4775 | 5001 | +2483 | 0.12 | 0.128 | 0.387 | 0.0165 | 0.26 | 0.67 | 0.36 |
| 2-3 | 6337 | 7104 | +1562 | 0.25 | 0.123 | 0.440 | 0.0194 | 0.33 | 1.27 | 0.37 |
| 3-4 | 7856 | 9023 | +1519 | 0.57 | 0.073 | 0.454 | 0.0246 | 0.74 | 3.28 | 0.32 |
| 4-5 | 8853 | 10054 | +997 | 1.13 | 0.068 | 0.480 | 0.0311 | 0.80 | 3.31 | 0.33 |
| 6-7 | 10180 | 12035 | +558 | 1.67 | 0.068 | 0.502 | 0.0389 | 0.69 | 2.62 | 0.34 |
| 9-10 | 11234 | 12407 | +240 | 1.35 | 0.075 | 0.509 | 0.0406 | 0.64 | 2.88 | 0.33 |
| 11-12 | 11378 | 12232 | +4 | 1.32 | 0.100 | 0.524 | 0.0362 | 0.65 | 2.84 | 0.32 |
| 14-15 | 11519 | 12618 | -28 | 1.52 | 0.073 | 0.525 | 0.0403 | 0.57 | 2.44 | 0.33 |

- The return slope decays smoothly from 1M; there is no discontinuity at 12M (agrees with `analysis.md`).
- **The one sharp change is at 3-4M**: q_mse rises 4.5x between 2-3M and 4-5M, the logged advantage corr drops
  0.123 -> 0.073 and never recovers, the KL multiplier doubles (0.33 -> 0.74) and the actor's pre-clip gradient
  norm rises 2.6x. The return slope falls from ~1500/M to ~1000/M in the next bin and keeps halving. v60 base
  and the v62 arms show the same sequence at the same steps, more gradually (q_mse 0.35 -> 0.91 -> 1.14, corr
  0.107 -> 0.089 -> 0.077, KL multiplier 0.41 -> 0.75 over 2-5M).
- The trust region is saturated for the entire run: mean KL sits at the 0.030 bound and 32-37% of states are in
  violation in every bin from 1M to 16M.
- Pre-tanh policy std **rises** monotonically (0.39 -> 0.52) while entropy is pinned at the -3.0 target; the
  temperature needed to hold it rises 2.4x and mean |action| rises 0.59 -> 0.65. As means saturate, the squashed
  entropy target is met by growing pre-tanh noise.
- The deterministic policy plateaus earlier than the stochastic one: deterministic eval is 12035 at 6-7M and
  12618 at 14-15M (+70/M), while the stochastic return gains 1340 over the same span. The gap closes from 1855
  to 1099. Late progress is mostly the policy becoming robust to its own noise.
- The v29 parent (no latent rollout) keeps advantage corr at 0.09-0.13 through 16M, has lower q_mse (0.7-1.1),
  and is 500-1700 lower in return throughout; its KL multiplier reaches 2.2 and actor gradient norm 11. Within
  runs, first differences of advantage corr vs return correlate at -0.3 to +0.1: no usable lead/lag signal.

---

## Q4. Published oracle and landscape numbers

### Critic oracle @1M (`03_q4_critic_oracle_1M.py`, `03_q4b_robust_1M.py`)

| Published `[doc]` | Recomputed | 95% CI | Robust version |
|---|---:|---|---|
| Level corr 0.93, error sd 0.90, between sd 2.3 | 0.930, 0.903, 2.318 | corr [0.83, 0.97]; error sd [0.72, 1.13] (lanes [0.46, 1.21]) | - |
| Within corr 0.11 (0.15 corrected) | 0.109 (0.149; split-half 0.148) | raw [0.09, 0.21], corrected [0.13, 0.28] | 5% critic-energy trim: 0.147 [0.07, 0.21]; per-state Pearson mean 0.100 +- 0.020 |
| True sd 0.22, critic sd 0.21 | 0.223, 0.206 | critic sd [0.07, 0.34] | critic sd on the median state 0.046; pooled value is 89% from 5 states |
| Pre-tanh slope 0.37 +- 0.12 (h 0.1) | 0.372 | [0.27, 1.71] | 5% critic-gradient-energy trim: 1.05 [0.43, 1.68] |
| Pre-tanh slope 0.21 +- 0.05 (h 0.3) | 0.211 | [0.16, 0.88] | trimmed: 0.49 [0.17, 0.81] |
| Self-consistency corr -0.33 | -0.331 | [-0.44, +0.50] | 5% trim: **+0.406 [+0.34, +0.47]**; per-state median +0.55 |

- Every pooled statistic reproduces to three digits. The published +- are rollout-noise SEs with the states held
  fixed; resampling states gives far wider and strongly asymmetric intervals, because 5% of states hold 92% of
  the critic's pre-tanh gradient energy and 1% hold 89% of its within-state energy.
- Robust gradient statements: noise-corrected pooled correlation 0.18 [0.12, 0.30] (h 0.1) and 0.17 [0.14,
  0.27] (h 0.3); the true rate of improvement along the critic's unit gradient, every state weighted equally,
  is 0.31 [0.16, 0.47] and 0.14 [0.08, 0.21] against a true gradient rms of 1.70 and 1.01. That is a cosine of
  roughly 0.14-0.18, positive with high confidence. On typical states the critic's gradient is roughly
  calibrated per unit (slope 0.5-1.0) but weakly aligned.
- Attenuation. The truth's reliability is 0.54 for fresh-action advantages (64 rollouts) and the FD-gradient
  signal fraction is 0.20 (h 0.1) / 0.43 (h 0.3). Disattenuation by the SE method and by split halves agree
  (0.149 vs 0.148; 0.182 vs 0.193; 0.174 vs 0.175).
- At 4M-30M (32 rollouts, H 300) the truth is much weaker: fresh-advantage noise share 0.70-0.79, FD-gradient
  signal fraction 0.06-0.09. The critic's z-space gradient slope there is 0.22 [0.01, 0.38] (4M), 0.09 [-0.45,
  +0.71] (8M), 0.74 [0.09, 1.11] (12M v29): **gradient accuracy after 1M is essentially unmeasured.** The
  advantage correlations (table in 1c) are the only later-stage numbers with usable intervals. They do not
  support a monotone decline: 0.15 (1M), 0.29 (4M), 0.20 (8M), 0.15 (12M), 0.23 (30M), overlapping CIs,
  different trainers.
- `[doc]` "true contrast std 0.33 vs critic 0.11 (3x compressed)" for v29 H8 is consistent with my 8M figures
  (true 0.446, critic 0.196 pooled) and with the median-state compression at 1M (4-5x).

### Return landscape (`03_q4d_landscape.py`)

| Published `[doc]` | Recomputed |
|---|---|
| Noise cost 5.0% / 12.3% / 11.0% | 5.0% [4.0, 5.8] / 12.3% [10.0, 16.1] / 11.0% [7.7, 15.5] |
| cos(critic grad, step) 0.07 / 0.03 / 0.008 | 0.0704 / 0.0270 / 0.0080 (deterministic parameter-space cosines; no sampling error) |
| Update-slice slope at 0, deterministic: +32 +- 4 / +15 +- 6 / -23 +- 8 | per-episode local fit, \|c\| <= 1: +12 [-6, +31] / +14 [-5, +33] / -83 [-221, +2] |
| Gain of the actual step J(1) - J(0), deterministic | +14 [-28, +56] / +7 [-41, +52] / -163 [-522, +54] |

- The published slope SEs come from one quadratic fit across c in [-2, 6]; locally none of the three slopes
  excludes zero. The training curve implies an expected gain of about +41 / +16 / +8 per iteration, below the
  landscape's resolution (CI half-width ~45 deterministic, 300-1000 stochastic). "Late, the whole step is critic
  error" rests on one snapshot pair and is only significant at twice the step (c = 2: -189 [-294, -103] at
  7.9M; -447 [-1220, -11] at 4M).
- The parameter-space cosine between a raw gradient and a 32-step Adam update under a KL gate is expected to be
  small whatever the critic's quality; 0.07 -> 0.008 is not evidence about the critic by itself.
- Along the raw critic-gradient direction (units of the actual update's norm), deterministic:
  - 1M: c = +0.5 gains **+141 [+104, +178]**, c = -0.5 loses 536, c = +1 loses 359. Half an update length along
    the raw critic gradient gains as much as six actual updates (c = 6 on the update slice: +128 [+87, +171]).
  - 4M: c = +-0.5 lose 2291 and 2708; 7.9M: lose 1582 and 1497. Symmetric, curvature-dominated; the local slope
    is not resolvable at this grid (+417 [-369, +1382] and -85 [-498, +495]).
  The published reading "any step of update size destroys return" is correct; what it omits is that at 1M the
  direction is a strong ascent direction at half that size.

---

## Q5. Anomalies the prose does not discuss

1. **Heavy tails dominate the pooled oracle statistics** (Q4). Three published conclusions change sign or size
   when 1-5% of states are removed. The self-inconsistency claim is the most consequential: on 95% of states
   the critic agrees with its own one-step lookahead at +0.41.
2. **The one-step lookahead with the critic's own V' ranks actions 2-3x better than Q, and a five-step one
   reaches 0.87** (1f). `[doc]` FAMILY.md states the model route has "the same slope ... no fix on its own",
   which is the pooled, outlier-dominated number.
3. **The true advantage is complete within ~10 steps** (1f). The long-horizon "chaos" contributes rollout
   noise, not within-state signal. A 10-25-step oracle would be as accurate as the 400-step one at a fraction
   of the cost and with reliability 0.9 instead of 0.54.
4. **The v61 / v62 "exact target" test did not remove the variance it was meant to.** The Q(lambda) correction
   subtracts the critic's own estimate of each future action's advantage. Given that estimate's quality at 1M,
   Var(A - A_critic) / Var(A) = 0.98-1.00 on 95-99% of states and 1.57 over all states (the outlier states add
   noise). The neutral result therefore does not show that "target variance is not the bottleneck"; the
   treatment was nearly a no-op. Consistent with the logged Bellman rms being unchanged across arms.
5. **Action-space gradient slope is zero where the pre-tanh one is positive**: -0.013 [-0.15, +0.82] (h 0.1,
   312 states) and +0.023 [-0.08, +0.62] (h 0.3, 169 states), same file, never quoted. Intervals are wide;
   65% of states have a coordinate with |a0| > 0.95 and 35% of action-space pairs were dropped, so the
   action-space sample is the unsaturated minority.
6. **None of the oracles measures the critic the actor differentiates.** Snapshots are taken at the end of an
   iteration and scored on a new rollout, i.e. a one-iteration-stale critic on states it has not fit. The actor
   reads the freshly fit critic at in-sample states. Whether in-sample fitting helps or memorizes target noise
   at a0 is unmeasured.
7. **Critic bias is not small and has no stable sign**: -0.60 (1M), -1.01 (4M v23), +2.74 (8M v29), -1.00
   (12M v29), +2.0 (12M v23), +5.7 (30M v23), with the targets carrying nearly the same bias each time (+2.63
   at 8M). Up to 2-5% of the value level late. Not investigated here; the truth bootstraps the critic at H 300
   with weight 0.05, so the magnitudes are slight under-estimates.
8. **Effective sample size for the state value is ~3-8% of the batch** (1b). The critic takes 128 minibatch
   steps (16 epochs) on what is statistically ~430-1360 independent value labels.
9. **Metric vs return is anti-correlated across runs** (Q2), and the metric moves only with data per policy
   step.
10. **Deterministic performance plateaus near 7M; the later stochastic gains are noise robustness** (Q3),
    while pre-tanh std keeps rising under the squashed-entropy target.
11. **Later-stage gradient numbers are unconstrained** (Q4): `results.md`'s "normalized directional gain
    0.13-0.17" at 8M / 12M sits on a truth with 6-9% signal.

---

## What I could not determine

- Whether the transition-channel advantage (1f) persists after 1M: only the 1M oracle saved n-step truths.
- Whether a stronger estimator than ridge / kNN extracts more from one chunk. The planted control shows the
  ones used are adequate for smooth signals; it cannot rule out structure of a kind I did not model.
- Gradient accuracy of the critic at 4M and later (truth too noisy), and anything about the in-sample critic.
- Seed variance: every run and every oracle is seed 1.

## Files

- Report: `benchmarks/reppo_rootcause/03_data_reanalysis.md`
- Scripts: `benchmarks/reppo_rootcause/scripts/03_prep_oracle_chunks.py`, `03_q1_estimation_ceiling.py`,
  `03_q1b_local_estimators.py`, `03_q1c_planted_control.py`, `03_q1d_quadratic.py`, `03_q1e_logged_metric.py`,
  `03_q1f_learned_lookahead.py`, `03_q4_critic_oracle_1M.py`, `03_q4b_robust_1M.py`, `03_q4c_lookahead_1M.py`,
  `03_q4d_landscape.py`, `03_dump_reppo_runs.py`, `03_q2_runs_table.py`, `03_q3_trends.py`
- Data: `benchmarks/reppo_rootcause/data/03_q2_table.tsv` (all runs), `03_*_log.txt`, `03_q1_results.json`,
  `03_q1b_results.json`, `03_chunk_*.npz` (flattened oracle chunks), `03_runs/` (scalar dumps, 33 MB)
