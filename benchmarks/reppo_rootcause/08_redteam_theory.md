# 08 - Red team of the theory report (04), under the corrected oracle numbers

Role: adversarial re-derivation. CPU only, no training, no simulation, no edits to existing files.
Inputs: reports 01-06, `cleanrl/reppo/FAMILY.md`, tfevents under `runs/`, the oracle chunks
(`data/03_chunk_*.npz`), and the v29-H8 snapshots, which turned out to contain the actor **before and after one
full update** together with a fresh rollout of the post-update policy.

Provenance tags: **[RT]** computed by me in this session (script named); **[01]..[06]** taken from that report;
**[D]** derivation (algebra, exact under the stated model); **[H]** heuristic. Confidence is stated per claim.

New scripts (all under `benchmarks/reppo_rootcause/scripts/`) and their outputs (`data/`):

| script | output | what it does |
|---|---|---|
| `08_thresholds.py` | `08_thresholds.txt` | steps-to-return-threshold, return gained per 16,384 steps by window, exploration / KL scale, 22 runs |
| `08_mine_runs.py`, `08_hparams.py`, `08_list_tags.py` | `08_mine_core.txt` | matched-step scalars (KL, entropy, std, Q-Prop diagnostics, slope beta) and hyperparameters |
| `08_toy_shrinkage.py` | `08_toy_shrinkage.txt` | posterior-mean slope estimator at the measured SNR: corr, amplitude, calibration slope vs pooled samples |
| `08_leak_check_real.py` | `08_leak_check_real.txt` | 04's own "cheap signature" for the collinearity leak, on the real critic at 4M / 8M / 12M / 30M |
| `08_sf_instrument_real.py` | `08_sf_instrument_real.txt` | the score-function update-gain instrument on a real REPPO update (8.01M, 12.01M), with SEs |
| `08_entropy_slosh.py` | `08_entropy_slosh.txt` | iteration-to-iteration fluctuation of entropy / std / temperature in the reference runs |

---

## 0. Verdict

1. **The contradiction is mostly a baseline artifact, plus one real effect 04 has no place for.** Against the
   two weak PPO controls REPPO-upright is 1.6-2.2x faster to every return level. Against the best PPO-lineage
   runs in the repo it is ahead only below ~5k (the first ~1M steps) and by a noise-level margin at 8M; between
   6k and 9k the two reach each level at the same step. At the same KL per iteration, REPPO and the sepclip
   PPO control gain the **same** return per iteration from 0.5M to 2M (55 vs 63, 26 vs 30). So 04's "same
   information" claim is roughly borne out where it can be tested, and its "extracts less" is not: there is no
   window in which REPPO-upright gains less per iteration than plain PPO except its own entropy dip at 1-2M.
2. **REPPO's lead over plain PPO has two separable parts**: a head start accumulated before 0.5M (+1.6k,
   present in every REPPO variant including v1 and the entropy-free ones), and a sustained higher rate from
   2M to 6M that exists only while the entropy dual holds the noise scale up. Remove the dual and the same
   critic, actor and KL bound stall at 7.0-7.3k, which is PPO's level.
3. **Under the corrected numbers the critic is close to an optimally shrunk estimator at 1M, not a broken
   one.** "Amplitude 0.3x truth, calibration slope ~0.6-1.2, correlation 0.15-0.29" is the signature of a
   posterior mean that has pooled about 2-11 on-policy samples per local slope. 04's K_eff of 0.2-1.2 and
   its "no pooling, no memory" were fitted to the outlier-dominated 0.11.
4. **The collinearity leak has no remaining support in the real system.** Its motivating number was an
   artifact; 04's own signature test, run on the real critic at four snapshots, gives correlations of
   -0.02 to +0.12 where a dominant leak predicts >= 0.3; and the premise of its late-regime amplification
   (sigma falls, sd(mu)/sigma rises to 6-25) is false for REPPO, where sigma rises and the ratio stays 2.1-2.5.
5. **04's dismissal of the model / transition channel is wrong on the information argument** and contradicted
   by 03's data (critic V' composed with true dynamics ranks at 0.34-0.43 after 1 step, 0.87 after 5).
6. **The instrument the study lacked exists and works on real data.** The score-function estimate of the
   realised first-order gain of the actor's actual update, on held-out rows, has a per-iteration standard error
   of 0.002-0.003 (20-30 return per iteration) at 8M, against a simulator paired-update probe whose 95% interval
   was +-0.10-0.15 from ~49M simulator steps. On the two real updates available it (a) agrees with the
   **pre-update** critic's predicted gain to within 1 SE both times, (b) shows the post-update critic's
   predicted gain is inflated 3.4x at 12M, and (c) shows both updates were dominated by the entropy dual moving
   the noise scale, not by the mean step.

Overall confidence in the ordering of section 3: moderate (about 60%). Every return comparison is single-seed.

---

## 1. Re-deriving 04's root causes under the corrected numbers

### 1.1 What the corrected signature means: optimal shrinkage [D, RT]

Model one local slope cell: g ~ N(0, tau^2 I_d), d = 6, sigma_A^2 = d sigma^2 tau^2; K on-policy samples
y_k = sigma g.eps_k + n_k with Var n = nu^2; per-sample signal share S = sigma_A^2 / (sigma_A^2 + nu^2).

    posterior mean:   g_hat = (E'E + kappa I)^-1 E'y / sigma,     kappa = d nu^2 / sigma_A^2 = d (1 - S) / S
    for ANY posterior mean:   Cov(g_hat, g) = Var(g_hat)
        =>  calibration slope (truth on estimate) = 1
        =>  amplitude ratio sd(hat) / sd(true) = correlation rho
    rho^2(K) = K / (K + K_half),   K_half = (d + 1) + d (1 - S) / S        (04's formula, confirmed)

So an optimal estimator at low SNR is **always** attenuated by exactly its own correlation and always
calibrated. The three alternatives are distinguishable by two numbers:

| estimator | amplitude ratio vs rho | calibration slope |
|---|---|---|
| posterior mean (right amount of shrinkage) | equal | 1 |
| unshrunk least squares / minimum-norm interpolation (noise fitted) | far above (2-10x at K <= 16) | 0.01-0.1 |
| over-shrunk / under-fit | below | above 1 |

`08_toy_shrinkage.py` (20,000 cells, 8 fresh actions each) at the measured S:

| S | K | posterior mean: corr / amplitude / slope | closed form corr | least squares: corr / amplitude / slope |
|---|---:|---|---:|---|
| 0.06 | 1 | 0.094 / 0.098 / 0.95 | 0.099 | 0.079 / 2.02 / 0.04 |
| 0.06 | 4 | 0.186 / 0.196 / 0.95 | 0.195 | 0.088 / 7.73 / 0.01 |
| 0.06 | 8 | 0.269 / 0.274 / 0.99 | 0.271 | 0.106 / 9.53 / 0.01 |
| 0.06 | 16 | 0.374 / 0.372 / 1.01 | 0.370 | 0.299 / 3.37 / 0.09 |
| 0.06 | 64 | 0.629 / 0.625 / 1.01 | 0.623 | 0.616 / 1.62 / 0.38 |

Measured, 1M, trimmed [02 C1, 03 Q4]: correlation 0.15-0.23 (0.29 at 4M), amplitude ratio 0.21-0.32,
gradient calibration slope 0.5-1.2, contrast-level slope = corr / amplitude = 0.6-0.9.

- That sits on the posterior-mean row with K between 2 and 11 (S = 0.05-0.07: rho 0.15 -> K 2.0-2.8;
  rho 0.23 -> 4.8-6.8; rho 0.29 -> 8.0-11.1; amplitude 0.32 read as rho -> 9.9-13.8).
- Amplitude / correlation is 1.1-1.5, i.e. mildly over-dispersed, far from the 2-10x of an unshrunk fit and on
  the opposite side from under-fitting.
- 02's reading "under-resolution" is correct only relative to the critic's **own one-step lookahead**
  (slope 1.69, amplitude 0.23, corr 0.39 [02 L1]: amplitude below correlation, slope above 1). Relative to
  the truth it is not under-fit. Both statements hold for a shrinkage estimator trained on noisy labels: its
  amplitude is set by the SNR of what it is shown, not by what a noise-free label would support.

**Answer to the task question.** "0.3x amplitude with roughly unit calibration slope" is shrinkage toward zero
of a well-regularised estimator at low SNR, i.e. approximately the right thing to do given the Q-regression
channel. It is not the single-sample minimum-norm structure 04 proposed (that has amplitude ~2x and slope
0.04 at K = 1) and not under-fitting. Confidence: moderate-high at 1M (two independent trimmed analyses);
low late, see next point.

**Where it stops being true.** The logged `probe/slope_beta` of the reference run falls monotonically:
0.92 (0.25M), 0.63 (0.5M), 0.60 (1M), 0.51 (2M), 0.42 (4M), 0.37 (8M) [RT, `08_mine_core.txt`], matching the
across-run corr / std_ratio slope of 0.6 -> 0.5 [01 sec 0.4]. A slope of 0.4 means contrasts ~2.5x too large
for their accuracy: late, the critic is over-dispersed, i.e. fitting noise in the action direction. Caveat:
these are pooled Pearson statistics and inherit the outlier sensitivity of [02 C1]; nobody has a trimmed
late-stage slope ([03 Q4]: "gradient accuracy after 1M is essentially unmeasured"). Grade C+.

### 1.2 The critic does pool across iterations (04 says it does not)

04 sec 2.2 row (b): "No. v49 measured that one iteration rewrites the contrasts by as much as their own sd."
Three independent measurements say otherwise, and they agree on the size:

1. **v49's number, read correctly [D].** The drift was dA = A_{k-1} - A_{k-2} on fresh rows, rms(dA) = sd(A)
   early (0.031 / 0.031) and 0.106 vs 0.14 at 1.6M [FAMILY.md l.401]. For two equal-variance fields,
   Var(dA) = 2 sd^2 (1 - rho), so rms(dA) = sd means **rho = 0.5**, and 0.106 / 0.14 means rho = 0.71.
   Independent redraws would give rms(dA) = 1.41 sd. "100% rewritten" is a misreading; half to 70% of the
   contrast variance survives each iteration. As an AR(1) memory that is (1 + rho) / (1 - rho) = 3-6
   iterations of effective pooling.
2. **03 Q1c.** The snapshot critic scores 0.285 / 0.200 where the best estimator built from one chunk scores
   0.18 / 0.13, and adding the chunk's information to the critic changes nothing. In information terms
   (rho^2) the critic holds 2.4-2.5x what one rollout affords the best alternative.
3. **02 L1.** A fresh critic gets 0.07 from one rollout after 1536 noise-free updates; the trained one has 0.39
   against the same labels. "What the trained critic knows about actions was accumulated over the preceding
   iterations."

So the warm-started critic is already an exponentially weighted pool over roughly 3-6 iterations. This is
source (b) in 04's own table, harvested. Consequence for 04's proposal: a Kalman / EMA over 10-30 iterations
has at most a 2-5x sample headroom over what exists (1.4-2.2x in correlation), and only if the slope field is
stationary that long. The policy moves 0.245 sigma per iteration (0.8 sigma as a random walk over 10
iterations, 2.5 sigma if coherent), and 03 finds the field rough in state (smooth share <= 4%), so
stationarity over 10-30 iterations is unsupported. Confidence that pooling of ~3-6 iterations exists:
moderate-high. Caveat: persistence can be persistent **error** as well as signal (v40 reports consecutive
gradient cosine 0.84 while the kl 0.003 fork degrades); instrument I5/I6 in section 4 separates them.

### 1.3 Mechanism by mechanism

| 04 mechanism | status under corrected numbers | evidence |
|---|---|---|
| RC1 per-sample memorisation, no pooling, no memory, "K_eff 0.2-1.2" | **Refuted as stated; survives in a different form.** K_eff is 2-11, pooling over ~3-6 iterations exists (1.2), the estimator is near-optimally shrunk at 1M (1.1). What survives: the Q-regression channel is information-poor (S = 5-7%, field rough in state [03]), so even an optimal estimator reaches only rho 0.15-0.3. "REPPO extracts less than PPO" is unsupported: per iteration at matched KL it extracts the same (section 2.1). | 02 C1, 03 Q1, `08_toy_shrinkage.txt`, `08_thresholds.txt` |
| RC2 collinearity leak | **Unsupported in the real system.** See 1.4. | `08_leak_check_real.txt`, 02 C1, 02 L1 arm W, v63 |
| RC3 slope horizon too short ("needs tens of steps", lambda_slope 0.95-0.98) | **Falsified as stated, a weaker form survives.** 04's E3 said "RC3 wrong if > 70% realised by n = 5". 03 1f: the 5-step truth correlates 0.90 with the 400-step truth, 10-step 0.995. The true advantage is a 5-10-step quantity. What survives: the n = 1-2 components of the lambda-0.8 mixture (36% of the weight) are weak, because the critic's one-step lookahead has half the true gradient amplitude [02 C1]. The supported fix is a fixed n of ~5-10, not lambda 0.95-0.98, which adds noise from ~17 steps for no signal. | 03 1f, 02 C1; v25 lam 0.95 vs 0.8 (8933 vs 9282 @8M, 11.9k vs 13.3k @50M), v53, v7 lam 0.99 (3308) all favour the shorter trace |
| RC4 sigma-denominated trust region | **Survives, but it is double-edged and 04 only saw one edge.** Early it is too timid (x5 step at 1M gains +123 +- 14, [01] grade B). But the same geometry is what separates REPPO from PPO after 2M (section 2.3): with sigma held at 0.4-0.5 a KL of 0.03 is a 2-4x larger absolute mean step than PPO's at 4M. | landscape, `08_thresholds.txt` |
| RC5 loss scale (HL-Gauss "3x rate handicap") | **Weak.** v14 scalar MSE is no better (9571 vs 10152, C+ null); v51's standardised contrast head gains early (+1.3k @2M, B-) then stalls. With pooling over only 3-6 iterations a rate handicap could matter, so not excluded. | 01 |
| RC6 roughness in the action at sub-sigma scale | **Open, slightly strengthened.** The h-dependence 04 cited (0.37 -> 0.21) was the artifact, but the trimmed slopes still fall with h (1.05-1.17 at h 0.1, 0.49-0.69 at h 0.3) and the true gradient rms is 1.70 at h 0.1 vs 1.01 at h 0.3 [03]. E6 (linear share of within-state variance) is still unmeasured. | 02, 03 |

Other 04 statements that do not survive:

- **Sec 2.2 row (f) and RC3's corollary: "a dynamics model composed with grad V contributes nothing
  on-policy, because grad V along the action-reachable directions is learned from the same eps."** The
  argument confuses where the two factors get their information. V is a level regression over the whole
  visited manifold (and over several iterations); its variation along action-reachable directions is
  constrained by between-state differences, which are far larger than one step of sigma eps (action-space
  between-state sd 0.66-0.70 vs within 0.28-0.33 [RT], and state differences accumulate over many steps).
  The action's effect on s' is a noise-free 17-dimensional supervised label per transition. Neither factor is
  "the SF statistic one step shifted". 03 1f measures the result: the critic's own V' after one true step
  ranks at 0.34-0.43, after five at 0.87, against 0.15 for Q(s, a). This channel is real information the
  pathwise design could use and the score function cannot. What is unresolved is whether a **learned** model is
  accurate enough (v30 / v37 / v38 negative, 03's linear model 0.01-0.21, all C-grade or flipped).
- **Sec 5 "direction efficiency ~0.06" and "predicted optimum 7.1x vs measured 5.4-7.8x, no free
  parameter".** The slope-at-0 it uses (+32 +- 4) is one quadratic fit; the local estimate is +12 [-6, +31]
  [03 Q4]. The agreement is not evidence.
- **Sec 5 "Late: sigma is smaller and means are saturated, so sd(mu)/sigma is large (25-100)".** Measured:
  pre-tanh sigma is 0.44 / 0.49 / 0.55 / 0.51 at 4M / 8M / 12M / 30M and between-state sd(mu) / sigma is
  2.5 / 2.5 / 2.1 / 2.4 [RT, `08_leak_check_real.txt`]. REPPO's noise scale rises through training.
- **Sec 2.5 (v61/v62 did not remove target noise)** survives and is confirmed twice [01 sec 0.5, 03 Q5.4].
- **Sec 3.1 (HL-Gauss is not a resolution floor)** survives [02 cleared, 01 sec 0.6].

### 1.4 Is there any evidence left for the leak?

What supported it in 04: self-consistency -0.33 (artifact: +0.41 to +0.48 trimmed), "truth-sized contrast sd"
(artifact: 0.2-0.3x), wrong-sign late slope (unmeasured: [03 Q4]), the v48 "law" (the metric's own ceiling
moves with spread [01]), and a linear-dynamics toy.

Direct test [RT, `08_leak_check_real.py`]. 04 sec 6 RC2 proposed "the cos between the critic's action gradient
and the pooled-across-states regression slope of V_hat on the action; RC2 predicts >= 0.3, otherwise ~0".
On the 512 held-out oracle states x 8 fresh actions, with w the across-state slope of the critic's Q on the
action fitted on the 16,384-row chunk (raw; linear-obs partialled out; and on tanh(mu(s)) alone), top 5% of
states by critic energy removed:

| snapshot | corr(critic contrast, association contrast): raw / partialled / on mu | partial coefficient on the association (95% CI over states) |
|---|---|---|
| v23 4M | -0.024 / -0.017 / -0.011 | -0.012 [-0.039, +0.014] / -0.016 [-0.097, +0.055] / -0.002 |
| v29 H8 8M | +0.068 / +0.030 / +0.055 | +0.011 [+0.003, +0.018] / +0.023 [-0.016, +0.057] / +0.005 |
| v29 H8 12M | +0.046 / +0.116 / -0.017 | +0.002 / +0.049 [+0.030, +0.069] / -0.000 |
| v23 30M | +0.001 / -0.017 / +0.022 | +0.000 / -0.003 [-0.011, +0.008] / +0.001 |

The critic imports 0-5% of the across-state association slope; the one significant cell (12M, partialled)
accounts for about 1.3% of the critic's contrast variance. Corroborating: 02's whitened-noise-input arm did
not help (C); v63, the leak-free estimator, regressed (B-); the trimmed self-consistency is positive.

Limits: this tests a **global linear** association, the form in 04's toy. A state-dependent nonlinear leak is
not excluded, and there is no snapshot before 1M. Confidence that a leak is not a leading cause after 4M:
moderate-high. 04's placebo-action experiment (E2) remains the clean test and is cheap.

---

## 2. Why upright REPPO beats PPO per sample

### 2.1 First, what the lead actually is [RT, `08_thresholds.txt`]

Steps (M) to reach a smoothed return level:

| run | 2k | 4k | 6k | 8k | 9k | 10k | 11k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppo_continuous_action` | 0.93 | 1.93 | 3.85 | - | - | - | - |
| `sepclip_control_8M` | 0.75 | 1.55 | 3.33 | 6.78 | - | - | - |
| best PPO-lineage (`iterthink_v24_..._tpoprobe_v1_k8_eta6`) | 0.62 | 1.20 | 2.02 | 3.02 | 4.05 | 6.05 | - |
| `entppo_v4_samplemask_a03_mask` | 0.70 | 1.38 | 2.40 | 3.30 | 4.10 | 6.38 | - |
| REPPO reference path A (`reppo_fast_v36_fp32`) | 0.42 | 0.93 | 2.25 | 3.60 | 4.55 | 5.90 | 7.97 |
| REPPO reference path B (`reppo_noisefork_v60_base`) | 0.42 | 0.88 | 2.08 | 3.20 | 4.00 | 5.25 | 7.55 |
| REPPO v7 cu128 (1-step aux, lambda 0.95, no refresh) | 0.45 | 0.85 | 1.98 | 3.92 | 5.17 | 6.55 | - |
| REPPO entropy-free Beta (`reppo_beta_noent_v19`) | 0.30 | 0.55 | 2.15 | - | - | - | - |
| REPPO v47 independent spread critic (no entropy) | 0.30 | 0.55 | 1.07 | 4.05 | 7.05 | - | - |

Return gained per 16,384 env steps (one REPPO iteration), by window:

| run | 0.25-0.5M | 0.5-1M | 1-2M | 2-3M | 3-4M | 4-6M | logged KL per iteration @1M / @4M (iteration size) |
|---|---:|---:|---:|---:|---:|---:|---|
| `sepclip_control_8M` | 54.8 | 63.1 | 30.5 | 17.4 | 12.9 | 8.6 | 0.023 / 0.017 (16,384) |
| best PPO-lineage | 78.3 | 65.5 | 41.7 | 32.9 | 15.9 | 8.6 | 0.029 / 0.036 (32,768) |
| `entppo_v4` | 66.4 | 59.1 | 39.7 | 29.8 | 26.4 | 8.1 | 0.033 / 0.042 (32,768) |
| REPPO path A | 108.4 | 54.2 | 23.9 | 23.4 | 25.2 | 12.9 | 0.031 / 0.031 (16,384) |
| REPPO path B | 118.7 | 55.5 | 25.8 | 32.9 | 20.3 | 10.1 | 0.031 / 0.028 (16,384) |
| REPPO entropy-free (v19) | 153.0 | 44.4 | 11.9 | 5.9 | 3.8 | 2.4 | 0.032 / 0.033 (16,384) |

Readings:

- **Versus the two plain PPO controls**: REPPO is 1.6-2.2x faster to 4k, 6k, 8k. The lead over sepclip is
  +1.6k at 0.5M, +1.4k at 1M, +1.1k at 2M, +2.5k at 4M, +2.7k at 8M. So it is a head start, a window of no
  advantage, then a second gain between 2M and 4M.
- **0.5M to 2M: equal gain per iteration at equal KL per iteration** (55 vs 63; 26 vs 30), REPPO vs the
  sepclip control, which is the only PPO run here with the same 16 x 1024 iteration. This is the one place the
  "same information" claim is testable against a like-for-like trust region, and it holds. (The other PPO
  runs use 16 x 2048 iterations; their logged KL is not comparable per step.)
- **Versus the best PPO-lineage runs**: ahead to 4k (0.88M vs 1.20M), level from 6k to 9k, ahead again at
  10k (5.25M vs 6.05M). At 8M, 11.0k vs 10.2-10.4k is inside the single-seed floor (~1.2k [01]) and both sides
  carry a selection bias: best of ~1000 on one side, conditional-on-upright on the other. I did not open
  those trainers; "PPO-lineage" is inferred from their names, as in [01].
- **In deterministic terms the gap is larger.** REPPO's deterministic eval is 12.6k at 8M with sigma 0.5 (the
  noise costs 11-12%); the PPO-lineage runs have a Gaussian-equivalent sigma of roughly 0.05-0.17 by 4M (from their logged
  entropies; the range covers the unknown Beta support convention), so
  their stochastic return is close to their deterministic one. REPPO's **mean** policy at 8M is about 2k
  better than the best PPO-lineage run and 4-5k better than plain PPO. The 8M stochastic score undersells it
  (v42: lowering the entropy target after 8M recovers +0.9k, grade B+).

### 2.2 The head start (before 0.5M): pathwise through an early critic that is still accurate

- Present in every REPPO variant: v1 reaches 1k at 0.20M (KL 0.26 per 4,096-step iteration), the entropy-free
  Beta variants reach 4k at 0.55M and 5k at 0.75-0.88M (the fastest runs to 5k in the repo, 2-3x PPO),
  ns128 reaches 2k at 0.23M.
- It is not the entropy dual (variants without it are faster early), not the H8 roll (v7 cu128 has it too),
  not lambda 0.8, not a KL-rate difference (sepclip's KL per iteration at 0.5M is 0.032, REPPO's 0.033).
- The only direct evidence about why: the reference run's logged real fraction of the critic slope is 0.92
  at 0.25M and 0.63 at 0.5M, declining to 0.37 by 8M, and the held-out metric is at its highest then (0.146
  at 0.25M) [RT]. Early, the action dependence of the return is dominated by short-horizon, state-smooth
  structure (reward is forward velocity minus a known quadratic in the action; 03 shows the advantage is
  complete in ~10 steps), which a regression in (s, a) can pool over the whole batch; a pathwise step then has
  almost no estimator variance while the score function is still paying for an untrained baseline (PPO's
  explained variance is 0.70-0.76 at 0.25M). This is candidate (d), state-generalisation of the slope, in the
  only regime where it is plausible. Grade C: one run's probe, no oracle before 1M. 03's "slope field is
  rough" finding is for 4M onward and does not contradict it.
- Step size matters early too: more iterations per env step are faster to 1-2k (ns128, v1) and worse later
  (ns128 -1.2k at 8M; v1 plateaus at 4.5k).

### 2.3 The second gain (2M to 6M): the entropy dual holds the noise scale, hence the absolute step

Within REPPO, same critic recipe, same actor, same KL bound 0.03 [RT]:

| run | entropy dual | policy std @1M / 2M / 4M | gain per iteration 2-3M / 3-4M / 4-6M | return @8M |
|---|---|---|---|---:|
| reference (tanh-Gaussian) | yes | 0.33 / 0.41 / 0.47 (pre-tanh) | 23-33 / 20-25 / 10-13 | 11.0k |
| v18 Beta mode | yes | 0.12 / 0.21 / 0.22 | 4.8 / 8.4 / 7.9 | 8.9k |
| v19 Beta, no entropy | no | 0.10 / 0.067 / 0.051 | 5.9 / 3.8 / 2.4 | 7.0k |
| v47 "same" (no entropy) | no | 0.10 / 0.070 / 0.061 | 6.1 / 2.2 / 3.8 | 7.0k |
| v21 hybrid (likelihood-ratio spread) | no | 0.23 / 0.15 / 0.11 | 7.3 / 3.2 / 3.5 | 7.3k |

- Every entropy-free pathwise run is as fast or faster to 5k and then stalls near 6-7k as its spread
  collapses, landing at 7.0-7.3k, which is where plain PPO lands (7.4-8.4k). [01] grades the collapse B.
  This is the cleanest "single removal" in the archive that is neither a flip nor a broken critic, and it
  removes the whole post-2M lead.
- PPO's own noise collapses the same way: the Gaussian PPO baseline's sigma is 0.32 / 0.21 / 0.14 / 0.11 at
  1M / 2M / 4M / 8M (from its entropy), and its gain per iteration decays 30 -> 19 -> 13 -> 7. REPPO's sigma
  goes the other way (0.33 -> 0.51). At 1M the two have the same sigma and the same gain per iteration.
- Mechanism, stated carefully. Both methods bound the step in units of sigma. With sigma held at 0.4-0.5 a KL
  of 0.03 moves the mean 2-4x further in action units than PPO moves at 4M, and a wider behaviour spread also
  keeps the action effect identifiable (the true within-state sd grows 0.22 -> 0.45 [03]). I cannot separate
  "bigger absolute step" from "more identifiable slope": both follow from sigma. What the data do say is that
  the gain is an exploration / step-geometry effect that needs the critic's slope to be merely usable, not
  good: v26 (action dependence collapsed, std_ratio 0.02-0.05) stalls at 3.5-4.1k.
- The PPO-lineage runs that reach 10.2-10.4k are the ones with modified exploration or entropy handling
  (`entppo_*`, `betacorr_*`) or a stronger critic; they close the 2-4M gap (30 / 26 per iteration vs REPPO's
  23-33 / 20-25). Consistent with this being something a score-function method can also have.
- The dual is not sufficient (v18 Beta with the dual stalls at 1-4M) and has an interior optimum
  (v43 0-nats 8.5k; fixed low entropy early 3.7k; lower target after 8M +0.9k).

### 2.4 Candidates, weighed

| candidate | verdict | evidence, grade |
|---|---|---|
| (a) warm-started critic pools the slope across iterations | **Real, moderate size.** ~3-6 iterations of memory; it is why the critic beats any one-chunk estimator. Necessary for the critic to be usable; not shown to be what beats PPO, since PPO's policy net also integrates across iterations. | 1.2; B- |
| (b) 32 actor steps on a smooth surrogate | **Weak lever.** au16 vs au32: 3426 vs 4583 @1M, 9493 vs 10152 @8M; au64 no gain. Matched-KL gain per iteration equals sepclip's at 0.5-2M. | 06 2.1; C+ |
| (c) lower-variance targets (lambda 0.8 Q) | **Not a source.** REPPO: v25 lam 0.95 vs 0.8 is 8933 vs 9282 @8M; v7 cu128 at lambda 0.95 is as fast early as the lambda-0.8 reference. PPO with lambda 0.8: 4742 vs 8391 (`awr_ce_ppo_sepclip_control_v1_lam0p8`), short-trace control 5150. Shorter traces alone hurt a V-baseline method. | C+ |
| (d) state-generalisation of the slope | **Plausible only before ~1M** (2.2); refuted from 4M on by 03's planted control. | C |
| (e) entropy dual keeps exploration noise high | **Largest supported contribution to the post-2M lead** (2.3). Also the source of the late noise cost and of the per-iteration slosh in section 4.3. | B for the removal; C for the mechanism |
| (f) auxiliary H = 8 latent roll | **06's "2.3x at 1M" does not hold up.** It compares against `refresh_v25_lam0p8`, which reads 1395 / 1842 at 0.5M / 1M, the flip-plateau signature, and then recovers [01 sec 1.3]. Against v7 cu128 (1-step aux) the H8 reference is -0.4k at 1M, +0.6k at 4M, +0.9k at 8M: inside noise, and confounded with the in-iteration refresh. What is supported: having **some** self-predictive aux loss (v7 noaux: 0 at 1M, -2.2k at 4M, -1.3k at 8M, at 4 critic passes). The PPO-lineage Q-Prop pair shows the same thing for a latent input (8998 vs 5648 @8M with / without). | B- for "an aux loss helps"; C for H8 |
| (g) 512-wide networks, 16 passes | Passes: 4 -> 16 is +2.0k at 1M and 8M, 32 is flat (C+). Width: never tested upright from scratch (cw1024 flipped); fresh wide critics under an 8M policy are null (C). | 06 2.1 |
| (h, added) decoupled actor / critic minibatches and lambda-returns | The largest single steps in the archive: v7 coupled 4006 vs 8152 (-4.1k, B); v5 -> v6 (B). These are conditions for the critic to be usable at all, not explanations of the lead over PPO. | 01 |
| (i, added) the pre-update critic's predicted gain is calibrated on real updates | New, N = 2 (section 4.2). If it holds, the pathwise surrogate is trustworthy at the size of step actually taken, which a clipped importance-weighted surrogate is not beyond its clip. | C |

Largest single removals within REPPO, by 8M cost (single seed, MDE ~1.2k): dueling critic -6 to -7k (B);
coupled minibatches -4.1k (B); entropy dual -1.8k within Beta and about -3 to -4k against the tanh reference
(B, policy-class confound); critic passes 16 -> 4 -2.0k (C+); aux loss -1.3k (B-); H8 + refresh vs v7 cu128
+0.9k (noise).

### 2.5 Runs that isolate "score-function actor with REPPO machinery"

| run | what it isolates | result | grade | reading |
|---|---|---|---|---|
| `reppo_qprop_v1` vs `_noqprop` | PPO (sepclip) actor; REPPO's HL-Gauss Q(s,a), lambda-returns, aux loss as a Q-Prop control variate | 9035 vs 8011 @8M; identical to 2M (2890 / 4800 vs 3076 / 4879) | C | **No head start**: with a score-function actor the REPPO critic gives nothing early. Logged corr(A, A_bar) is 0.035-0.10 and eta falls 0.3 -> 0.016 [RT], so the control variate removes <= 1% of the variance; the +1.0k cannot be variance reduction and is inside noise. |
| `sepclip_reppo_hlgauss_v1_normreward` | PPO with REPPO's HL-Gauss value loss | 2482 / 4796 / 7109 / 9369 | C | +1.0k over sepclip at 8M, nothing at 1M. The loss is not the head start. |
| v39 dual-PG from scratch | 0.5 pathwise + 0.5 clipped likelihood ratio | flipped | none | uninformative |
| v40 forks from 8M | w = 0.5: 10972 vs 11408 @12M; pure likelihood ratio: 8678 @10M | C for w 0.5 (noise); B for "pure LR at the pathwise KL step hurts" | 30% of samples clipped: the LR arm was not given PPO's step regime. Does not test a score-function actor fairly. |
| v21 hybrid | pathwise mode, likelihood-ratio precision, no entropy | 7331, spread still collapses | B- | The collapse is not a pathwise-curvature artifact. |
| v63 orthogonal A(s, eps) | leak-free critic, pathwise actor: PPO's statistic routed through a net | 4876 / 5783 vs 5658 / 7896 @2M / 3M | B- | The orthogonal form is worse than Q(s, a) at the same budget. |
| `reppo_v1_ppobatch_*` | REPPO with PPO-style full-batch epochs | 1.6-3.2k @4M | C | too few optimiser steps; not an isolation |
| `ppo_short_trace_control_v1`, `awr_..._lam0p8` | PPO with a short actor trace | 5150, 4742 @8M vs 8391 | C+ | shorter traces alone hurt PPO |

Taken together: nothing in the archive shows a score-function actor benefiting from REPPO's critic, loss or
aux machinery before 2M, and the one clean isolation (qprop) shows no early effect. The head start therefore
belongs to the pathwise actor, not to the critic's representation feeding a score-function estimator. No run
puts REPPO's entropy dual on a PPO actor from scratch; that is the missing isolation for (e).

---

## 3. Root causes re-ranked, and 04's design elements re-graded

### 3.1 Ranking (of what limits REPPO-upright, then the unconditional score)

1. **Flip rate** (unconditional): 2 of 4 reference trajectories; worth ~4k expected [01]. Untouched by any
   theory report. No PPO-family run sits on that plateau (0 of 111).
2. **An information-poor Q-regression channel, used about as well as it can be** (corrected RC1). S = 5-7%
   per sample, slope field rough in state, critic near the posterior mean with K_eff 2-11. Confidence ~65%.
3. **Entropy-dual dynamics late**: sigma 0.5 costs 11-12% of return; the dual moves sigma +-3-4% per
   iteration, with first-order return effects of +-50-80 per iteration against a net progress of +4-10
   (section 4.3). v42 is the best-graded positive result in the record. Confidence ~55% that this is the
   largest fixable late term.
4. **The unused transition channel** (03 1f): 2-6x better action ranking from the same V, at 1M only, needing
   an accurate model. Confidence ~50% that it persists past 1M; unmeasured.
5. **Step geometry early** (RC4): x5 at 1M. Real but shared with PPO, and every attempt to exploit it
   regressed.
6. Late over-dispersion of the action slope (slope 0.4): C+, pooled statistics only.
7. Loss scale, roughness in a: weak / open. **Leak: unsupported.**

### 3.2 04's "statistically optimal" elements

| element (04 sec 7) | status | already tried? (version, outcome, [01] grade) |
|---|---|---|
| 1. Score-function gradient with a critic control variate, instead of pathwise | **Unsupported, and the available evidence points the other way.** Its premise was a leaky critic that "costs direction"; there is no leak signal (1.4), the pre-update critic's predicted gain matched the realised gain on both real updates (4.2), and the pathwise actor is where the head start comes from (2.5). The unbiasedness argument still stands as insurance against the late over-dispersion. | `reppo_qprop_v1` 9035 vs 8011 (C; control variate explains <= 1% of variance); `latent_qprop_v1..v5` 8.4-9.4k @8M, 13.6-14.0k @50M, g-head "explained ~0% of A" (C); v39 flip; v40 w 0.5 -0.4k (C), pure LR -2.7k (B, clip confound); v21 7331 (B-). 04 dismisses the qprop runs as "(s, a) regressions with the leak"; with no leak that dismissal no longer holds. |
| 2. Orthogonal pooled slope head, no (s, a) input | **Unsupported.** Pooling across states is refuted from 4M on by the planted control (03: smooth share <= 4%); the orthogonal form discards the absolute-coordinate structure that makes cross-iteration memory possible, as 04 itself notes. | v63 -2.1k @3M (B-); v26 dueling collapse (B); v38 scale-normalised A fails (C+); v51 contrast head +1.3k @2M then stall (B-); 02's whitened-noise input arm worse (C). |
| 3. Kalman / EMA of the slope over 10-30 iterations | **Mostly already present, remaining headroom small.** The warm-started critic pools ~3-6 iterations (1.2). Stationarity over 10-30 is not supported. | v50 contrast anchor (stops drift 0.008 vs 0.049, 1.8-1.95k @1M, culled on the flip plateau; C); v9 frozen judge 8595 vs 8152 (C); v41 mean of two critics null (B-); v45 population 11002 = reference (C); v25 in-iteration refresh +2.8k at 50M, -0.9k at 8M (B at 50M). |
| 4. Longer horizon for the slope than for the level (lambda_slope 0.95-0.98) | **Refuted in that form; a short fixed n is supported.** The advantage is complete in 5-10 steps; the weak part of the lambda-0.8 mixture is n = 1-2. | v53 contrast head lambda 0 / 0.5 / 0.95: 3425 / 3512 / 6661 vs 8343 at 0.8 (C+): longer is worse than 0.8, much shorter is far worse; v61 lambda 0.95 9715 (C); v7 lambda 0.99 3308; v25 0.95 vs 0.8. A fixed n = 5-10 target has not been tried. |
| 5. Step from evidence rather than from sigma | **Still justified in principle, with the instrument of section 4 as the evidence.** The earlier attempts used the critic's own curvature or noisy return line searches. | v57 (miscalibrated 5.7x, C); v58 real-return line search -2.6k @4M (B-); v48 box step (bug / cull); v40 kl 0.01 null, kl 0.003 -3.3k (B); v23 kl 0.015 / 0.01 -2.3 to -2.9k @8M, equal by 33M (C+); ns128 +0.8k @1M, -1.2k @8M (C+). |
| 6. A short replay window for the slope and the dynamics-through-V route | **Justified, more than 04 argued**, because of the transition channel (03 1f), not because of collinearity. | v11 V-trace buf4 / buf8 +2.4-2.6k over its buf1 control, late-blooming (B-); v24 TD7-regime critic never ran past 0.41M; v52 culled at 1.2M. |

04's net forecast ("1.5-3x over PPO from items 1-5") rested on K_eff ~1 and on pooling to K ~ 50-100. With
K_eff already 2-11 and no poolable structure after 4M, the forecast has no remaining basis.

---

## 4. The metric problem

### 4.1 The instrument: realised first-order gain of the actual update, by score function [D]

Rows (s_i, a_i) from the rollout of pi_old, u_i the pre-tanh action, eps_i = (u_i - mu_old(s_i)) / sigma_old(s_i),
A_i an advantage estimate for pi_old that the update did not see.

    forward:   R_fwd = (1/N) sum_i A_i [ log pi_new(u_i | s_i) - log pi_old(u_i | s_i) ]
    linear:    log ratio ~ sum_d dmu_d eps_d / sigma_d  +  sum_d dlog sigma_d (eps_d^2 - 1)        (mean part + scale part)
    backward:  R_bwd = -(1/N) sum_j A'_j [ log pi_old(u_j | s_j) - log pi_new(u_j | s_j) ],   rows from the NEXT rollout (pi_new)

Expectation (Stein, then the performance-difference lemma):

    E R_fwd = E_{s ~ d_old} [ g_sigma(s) . dmu(s)  +  c_sigma(s) . dlog sigma(s) ]  +  O(KL^1.5)
            = first-order change of the per-step discounted advantage = (1 - gamma) (J(new) - J(old)) to first order

with g_sigma the sigma-smoothed slope and c_sigma,d = E[A (eps_d^2 - 1)] the smoothed curvature term. In the
trainer's scaled units, +10 return per iteration (1000-step episodes, reward scale 0.1) is +0.0010. The same
units as the critic's predicted first-action gain, P = E_s[Q(s, a_new) - Q(s, a_old)] with common noise.

Variance. x_i = A_i w_i with w_i the log ratio, Var(w_i) = 2 KL_i. To leading order

    Var(x_i) = Var(A) . 2 KL          SE_1 = sd(A) sqrt(2 KL) / sqrt(N_eff)
    SNR_1    = m / SE_1 = eta . sqrt(S' . N_eff)

where m is the true gain, eta the per-state direction efficiency of the step (cos between dmu and g_sigma)
and S' = sigma_A^2 / Var(A) the per-sample signal share. Score products are white along a trajectory
(ESS factor 0.95 [03]; lag-1 autocorrelation +0.00 to +0.08 here), but lanes are heterogeneous, so the
lane-block SE is ~1.3x the iid one: use N_eff ~ 9,000-12,000 of 16,384.

With S' = 0.05-0.07, eta = 0.06-0.10 and N_eff = 9,000-15,600: SNR_1 = 1.3-3.3 per iteration.

Per stage, using sd(A) ~ sqrt(q_mse + sigma_A^2) for a lambda-0.8 residual (this reproduces the measured 1.19
at 8M), KL 0.03, N_eff 9,000, and the training curve's gain per iteration as the signal:

| stage | sd(A) | SE_1 (units / return per iteration) | net progress per iteration | SNR_1 | iterations for SNR 3 |
|---|---:|---|---:|---:|---:|
| 1M | 0.41 | 0.0011 / 11 | 26-55 | 2.5-5 | 1-2 |
| 2-3M | 0.57 | 0.0015 / 15 | 23-33 | 1.6-2.2 | 2-4 |
| 4-5M | 1.1 | 0.0028 / 28 | 16-25 | 0.6-0.9 | 11-25 |
| 8M | 1.2 | 0.0031 / 31 | 4-10 | 0.13-0.3 | 90-500 |

So as a meter of **net progress** it is usable through ~3M with a 4-iteration EMA, marginal at 4-5M with a
16-32 iteration EMA, and useless at the plateau (where net progress is ~0 by definition). As a meter of
**whether a specific update or its predicted gain is real**, it stays useful to the end, because those
quantities are 5-20x larger than net progress late (4.2, 4.3): a predicted gain of +50-200 per iteration is
resolved in 1-4 iterations at any stage.

Biases, with sizes:

1. **Rows the critic and actor trained on (the important one).** If the critic memorises "a_i at s_i was
   followed by a high return", its slope at s_i points along eps_i (G_i - V), the actor moves that way, and
   x_i picks up A_i^2 |eps_i|^2 > 0. Any memorisation-induced correlation r_mem between w_i and A_i adds
   r_mem sd(A) sd(w), to be compared with a signal correlation eta sqrt(S') = 0.013-0.026 per row. v51 measured
   an in-sample contrast correlation of 0.25 with its own targets vs 0.08 fresh, so r_mem of several
   hundredths is plausible and the in-sample reading could be several times the truth. **Use held-out rows.**
   The backward form on the next rollout is free and exactly held-out (the trainer already runs diagnostics
   on the fresh rollout before fitting). Alternatively hold 2 of 16 lanes out of both the critic's and the
   actor's minibatches (12.5% of the data) for the forward form.
2. **State-distribution shift / second order.** Relative size O(KL x curvature): from 04's own curvature
   number, ~7% at the taken step. The trapezoid (R_fwd + R_bwd) / 2 cancels it, in the same way the trapezoid
   rule integrates a gradient along a path.
3. **Bootstrap in A.** With a lambda-return the instrument measures the lambda-mixture of n-step lookahead
   slopes. The critic's one-step lookahead has about half the true amplitude [02 C1], so at lambda 0.8 the
   reading is attenuated to roughly 75% of the truth (sign preserved); at lambda 0.95 to ~95% at ~1.8x the
   standard deviation. A fixed n = 10 truncation is the better trade given 03 1f.
4. **Soft vs hard return.** The critic's Q and the trainer's targets are soft. A state-independent entropy
   offset in A does nothing (the score has zero mean), but the actor is paid T dH directly, which the
   instrument does not include. Log the mean part, the scale part and T dH separately (4.3 shows why).
5. **Behaviour-noise mismatch.** [02 L4]: training renormalises observations after the rollout, so
   `old_actor` on renormalised observations is not exactly the behaviour policy. Use the stored behaviour
   noise and behaviour-time (mu, sigma), and centre A, so a nonzero mean of w cannot leak the level in.
6. **Heavy tails.** max |A| is 30-70 against an sd of 1.2-1.7 (flipped / off-distribution states). Winsorising
   A at 0.5-2.5% per tail lowers the SE by 20-45% and moved the estimates by 0.5-1.5 SE here. Log both, and
   take the SE from a jackknife over the 16 lanes.

### 4.2 It works on real data [RT, `08_sf_instrument_real.py`]

The v29-H8 snapshots store the actor and critic before and after one full update; the oracle file stores a
fresh 16 x 1024 rollout of the post-update policy with trainer-exact lambda-0.8 targets. That is the backward
form, exactly held-out. Reconstruction check: recovered noise has mean -0.002, sd 1.003.

| | 8.01M | 12.01M |
|---|---|---|
| KL(new, old) mean / median; rms \|dmu\| / sigma | 0.026 / 0.021; 0.20 | 0.043 / 0.026; 0.28 |
| sd(A); per-row sd of x | 1.16-1.20; 0.27 | 1.63-1.70; 0.42 |
| **realised gain**, full log ratio, pre-update critic as baseline (lane-bootstrap SE) | **-0.0061 +- 0.0034** | **+0.0082 +- 0.0040** |
| same, post-update critic as baseline | -0.0063 +- 0.0028 | +0.0058 +- 0.0028 |
| same, A winsorised 2.5% | -0.0049 to -0.0051 +- 0.0017 | +0.0044 to +0.0072 +- 0.0017-0.0034 |
| realised gain, mean part only | -0.0040 to -0.0047 +- 0.0022-0.0029 | -0.0006 to +0.0003 +- 0.0032-0.0036 |
| **predicted gain, pre-update critic** | **-0.0047 +- 0.0003** | **+0.0064 +- 0.0005** |
| predicted gain, post-update critic | +0.0001 +- 0.0002 | **+0.0220 +- 0.0016** |
| change of the squashed entropy in this one update; T dH | +0.137 nats; +0.0057 | -0.081 nats; -0.0022 |
| in return per iteration: realised / predicted (pre) | -61 +- 34 / -47 | +82 +- 40 / +64 |

Readings (N = 2 updates; each realised number is ~2 SE from zero; treat as a demonstration, grade C):

- **Precision.** One iteration of free on-policy data gives +-0.003. The simulator paired-update probe for the
  same two updates reported 95% intervals of [-0.144, +0.048] and [-0.141, +0.151] from 512 states x 3
  policies x 32 rollouts x 1000 steps [benchmarks/reppo_diagnose_v29/results.md], against predicted gains of
  0.0005-0.005. The probe was 20-50x too coarse to see what it was built to see; this instrument is not.
- **The pre-update critic's predicted gain matches the realised gain in both cases** (ratio 1.3 +- 0.7 and
  1.3 +- 0.6), including the sign flip between the two. The critic the actor starts from is, on this evidence,
  calibrated about the update that was actually taken.
- **The post-update critic's prediction is inflated**: +0.0220 predicted vs +0.0058-0.0082 realised at 12M
  (3.4 SE apart), and it turned -0.0047 into 0.0000 at 8M. The critic that co-adapts with the actor during the
  iteration overstates the gain of the actor's own move by 50-150 return per iteration. That is the
  optimiser's-curse term, measured directly; it is invisible to every metric the study logged.
- **Neither update was a mean-improvement step.** At 8M the update raised the squashed entropy by 0.137 nats in
  one iteration (the actor was paid +0.0057 by the entropy term and gave up 0.0047 of Q, as the pre-update
  critic said it would); at 12M it lowered entropy by 0.081 nats and the whole realised gain is in the scale
  part (mean part 0.000 +- 0.003).

### 4.3 What that points at: the entropy dual sloshes the noise scale [RT, `08_entropy_slosh.txt`]

Logged every iteration, reference runs, after 3.5M: squashed entropy has sd 0.22-0.30 nats around its target
and an iteration-to-iteration difference sd of 0.28-0.34 with negative lag-1 autocorrelation of the differences
(-0.17 to -0.52); `policy/std` moves +-3-4% per iteration (difference sd 0.015-0.022 on 0.47-0.51, lag-1
-0.23 to -0.44); the temperature has sd 0.008-0.010 on a mean of 0.034-0.041. Before 1.5M the same series are
smooth trends (lag-1 +0.55 to +0.91).

Size of the effect: the noise costs ~1,400 return at sigma 0.5 and scales as sigma^2, so +-3% in sigma is
about +-85 return per iteration, the size the instrument measured (-61, +82), and roughly ten times the net
progress per iteration late (+4-10). It also spends about 6 x 0.03^2 = 0.005 of the 0.03 KL budget per
iteration on moving sigma back and forth. This is a concrete, checkable candidate for the late plateau that
is neither a critic-accuracy nor an information problem, and it is consistent with the best-graded positive
result in the record (v42, entropy target, B+) and with 03's observation that late stochastic gains are
"robustness to its own noise". Confidence that the slosh is real: high (logged every iteration, two runs).
Confidence that removing it would raise the plateau: low-moderate; untested.

### 4.4 Further instruments

| instrument | definition (all from on-policy data, no resets) | what it would have revealed |
|---|---|---|
| **I1** realised gain (4.1), EMA over 4-32 iterations, split into mean / scale / T dH | backward form on each fresh rollout; lane-jackknife SE | whether a variant's updates are real before the return curve can show it: at 1-3M a 4-iteration window resolves +-8 return per iteration, against a single-seed return floor of 300-1200. It would have shown the late slosh (4.3) in the first run that logged it. |
| **I2** trust-region ratio rho_TR = I1 / predicted gain, for the pre-update and the post-update critic | same rows; predicted gain is deterministic given the critic | calibration **along the direction that is actually taken**, which is what the actor needs, rather than a pooled per-state correlation. It separates "critic wrong" (pre-update ratio far from 1) from "co-adaptation optimism" (post-update ratio << 1, the 12M case) from "step too small" (ratio ~1 with a persistently positive I1, the 1M landscape case). It would have replaced v55's beta, v57, v58 and the landscape probe, and given the evidence-based step (04 item 5) something valid to stand on. |
| **I3** in-sample minus held-out I1 | forward form on trained rows minus forward form on 2 held-out lanes | memorisation of target noise in the action direction: exactly the unmeasured item in [03 Q5.6] and the object of 02 L1's "more critic updates are predicted to hurt". It would turn the cu64 / 128 / 256 sweep from a return comparison into a measurement. |
| **I4** split-half critics | two critics (or two heads on a shared trunk) fit on disjoint halves of the lanes; on a third set of states log (i) the correlation of their within-state contrasts, per-state-normalised and trimmed, and (ii) the cosine of the two actor parameter gradients | reliability without a simulator: with independent errors the contrast correlation is rho_half^2, so it gives the accuracy that the 0.3-ceiling metric cannot, and item (ii) is the batch-mean direction quality [01 sec 7c1]. Shared history and any shared bias inflate it, so it is an upper bound on accuracy; a high split-half agreement with a low rho_TR is the signature of a stable bias (a leak would look like this). v8 and v41 had the machinery and logged neither. |
| **I5** slope persistence | cosine between consecutive iterations' critic action gradients at the same held-out states, and its decay over lags 1, 2, 4, 8 | the critic's actual memory window (section 1.2 infers 3-6 iterations from one early number). With I4 it separates noise (low agreement, low persistence) from tracking (high agreement, decaying persistence) from bias (high persistence, rho_TR below 1). It decides whether 04's 10-30 iteration filter has anything to average. |
| replacement for `advantage_corr_heldout` | the same statistic computed per state against a 10-step truncated return, state-normalised, top 1-2% of states by critic energy reported separately | removes the 0.3 ceiling's dependence on the data regime (the ceiling itself moves with spread and with lambda) and the single-state domination of [02 C1]. Still a per-state quantity; I1 / I2 are the ones tied to the update. |

Why the old metric is anti-predictive: it rises with data per unit of policy change (smaller KL, longer
rollouts, wider spread) and those same knobs slow learning [03 Q2]. I1 does not have that property: it is
proportional to the step actually taken.

---

## 5. Adversarial notes on the other first-round reports

- **01**: "REPPO conditional-on-upright is ahead of PPO by 2.6-3.6k" is right against the weak controls only;
  the lead is a head start plus a 2-4M window (2.1). Its "error structure, not error size" hypothesis
  [sec 7c1] is not supported by the two real updates: the pre-update critic's prediction along the taken
  direction was calibrated. The 8M comparison should also be made deterministically, where REPPO's margin is
  ~2k.
- **02**: "calibrated but attenuated = under-resolution" is half right; relative to the truth the critic is
  near the optimal shrinkage and slightly over-dispersed (1.1). The conclusion "more critic updates are
  predicted to hurt" is consistent with that; "get the action dependence from the dynamics" is the channel
  with measured headroom.
- **03**: the estimation-limited conclusion rests on ridge / random-feature / kNN estimators in normalised
  observation space; "rough" is relative to those features. Its own 1f shows the information exists in
  another channel, so "missing information" should read "missing in the Q-regression channel". The planted
  control has no counterpart before 4M, where the head start happens.
- **06**: the roll's "2.0-2.5x at 1M" uses a flip-like parent as the denominator (2.4 f). The cost conclusion
  (69% of the update) stands, which makes the roll's unproven benefit the cheapest thing to test.
- **This report**: the within-REPPO entropy attribution mixes policy classes (Beta vs tanh-Gaussian); the
  matched-KL comparison treats PPO's approximate KL and REPPO's exact KL as the same quantity; the two-update
  instrument result is N = 2 at about 2 SE each; every "PPO-lineage best" run was identified by name only;
  nothing here addresses the flip, which is the largest term in the expected score.

## 6. Cheapest decisive follow-ups (not run; all would go through `mlq`)

1. Log I1 / I2 (backward form, both critics, mean / scale / T dH split) in the reference trainer. No extra
   environment steps; one batched forward of the old actor per iteration.
2. Fork from the 8M snapshot with the temperature frozen at its running mean (or the dual's step cut 10x):
   tests 4.3 directly against the v42 controls.
3. Replace the lambda-0.8 mixture by a fixed n = 5-10 target (1.3, RC3) on a fork from 1M.
4. 04's placebo-action fit (E2) on a saved rollout, to close the leak question before 1M.
5. REPPO's entropy dual on the sepclip PPO actor from scratch: the missing isolation for 2.3.
