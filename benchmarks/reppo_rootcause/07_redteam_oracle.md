# 07 — Red team of the oracle claims, and the critic-vs-PPO head-to-head

Role: adversarial verifier. Everything below is recomputed from raw arrays with new code
(`benchmarks/reppo_rootcause/scripts/07_*.py`, logs in `data/07_*`). CPU only, no training, no simulation.
Units are the trainer's scaled returns. CIs are 95% bootstraps over states (and over chunk blocks where a
chunk is involved), 1000-2000 draws. One seed, one snapshot per stage throughout: every statement is about
these snapshots.

Data actually available

| Tag | Has states / obs / actor moments? | On-policy chunk? | Truth |
|---|---|---|---|
| 1M (v60, `runs/reppo_critic_oracle/*.npz`) | **no** (only a0, pre0, lane, age) | **no** | 512 x 58 x 64 CRN, H 400, n-step checkpoints at t = 1, 5, 10, 25, 50, 100, 200, 400 only |
| 4M / 12M / 30M (v23), 8M / 12M (v29 H8) (`runs/reppo_oracle`, `runs/reppo_latentroll_oracle`) | yes | yes, 16 x 1024, and the 512 oracle states are rows of it | 512 x 58 x 32 CRN, H 100 / 200 / 300 |

Consequences: A, C, E can only be done at 1M; B and D only at 4M and later. D at 1M is impossible (no policy
noise, no trajectory returns, no observations for that snapshot). C's "rewards alone vs V' alone" split for
n >= 5 and horizons n = 2-4, 6-9 are **not in any saved array** (see C).

## Verdicts

| Claim | Verdict | One line |
|---|---|---|
| A. 1M pooled oracle numbers are owned by state 402; trimmed, the gradient is "calibrated, 3x too small, r^2 ~0.05, self-consistent" | numbers **HOLD**; interpretation **WEAKENED** | Trimming hides nothing (untrimmed, actor-weighted alignment is the same 0.15-0.22). But "calibrated slope ~1" is true of any least-squares predictor and says nothing about resolution vs noise; the reverse slope is 0.03-0.06: 95% of the critic's action dependence is unrelated to the truth. |
| B. Return-regression channel is estimation-limited; critic at or above what the data supports | **HOLDS** per state; **WEAKENED** as a statement about what the actor needs | Independent bound: only ~2-15% of the true gradient field's energy lies in any policy-representable feature span, so no amount of pooling lifts the per-state correlation above ~0.15-0.4. In those spans, however, pooling 10-30 rollouts would give an aggregate cosine of 0.75-0.9, and the critic rewrites its field by 50-110% rms per iteration. |
| C. Critic V' with true dynamics ranks at 0.34-0.43 after 1 step, 0.87 after 5; advantage complete in ~10 | numbers **HOLD**; "the critic's V carries it" proven **only at n = 1** | n = 1 is >= 96% V' (reward share of the label's action gradient 0.1-0.2%). For n = 5 the V'/reward split is not recoverable from the arrays. CRN is not what makes it work, but one real branch is far noisier than the published expectation (single-label corr 0.44-0.54, not 0.87-0.90). |
| D. Critic dQ/da vs PPO score function, per rollout | **EQUAL within error; point estimates favour the critic at 4M-12M, the score function at 30M** | In policy-parameter projections the critic's true gain per unit step is 1.2-1.8x the SF's at 4M/8M (P(critic better) 0.75-1.00), unresolved at 12M, 0.2-0.75x at 30M (P 0.31-0.55). Nowhere is the critic significantly worse. |
| E. Vine economics | answered for n in {1, 5, 10, 25} | Break-even against an ordinary sample is at n ~ 5 (K = 8). n = 1 is 3-4x more informative per env step but its ceiling is 0.45; n = 10 is 0.2-0.3x. |

---

## A. The 1M oracle and state 402 (`07_A_oracle_1M.py`, `data/07_A_log.txt`)

Reproduced exactly: within corr 0.109 (0.149 corrected), self-consistency -0.331, slopes 0.372 / 0.211,
true sd 0.223, critic sd 0.206 pooled and 0.046 on the median state. State 402 holds 86% of the critic's
pre-tanh gradient energy and 88% of its within-state energy.

**A1. What state 402 actually says (not in 02 / 03).** At that state the critic's gradient is *right in
direction and 2-4x too large*: true rate along the critic's unit gradient +4.4 +- 2.3 (h 0.1) and +2.4 +- 1.0
(h 0.3) against a measured |g_true| of 8.0 / 3.8 (noise included, so the cosine is at least ~0.55-0.65); |g_critic| 17.5. What is wrong there is the
critic's **own one-step lookahead**: cos(1-step lookahead FD, truth) = -0.79 / -0.88, while the 5-step
lookahead is +0.90 / +0.91. So "Q anti-correlates with its own lookahead (-0.33)" is one flipped state where
Q is right and V'(s') is wrong. 02 / 03 are correct that the statistic is an artifact; neither noticed which
side was wrong.

**A2. Is trimming hiding what the actor experiences? No.** The actor's loss is an untrimmed sum, so I scored
the field the way it is consumed: step delta_a(s) = w(s) g_c(s), true first-order gain per unit rms step,
as a fraction of a perfect field's (pre-tanh; brackets are state-bootstrap CIs of the raw gain).

| weighting w | h 0.1, all states | h 0.1, without 402 | h 0.3, all | h 0.3, without 402 |
|---|---:|---:|---:|---:|
| raw sum (w = 1) | 0.18 [0.11, 0.31] | 0.21 | 0.17 [0.14, 0.27] | 0.20 |
| clip |g| at p99 | 0.22 | 0.21 | 0.22 | 0.20 |
| clip |g| at p90 | 0.19 | 0.18 | 0.18 | 0.17 |
| unit per state (per-state trust region) | 0.19 [0.09, 0.27] | 0.18 | 0.14 [0.07, 0.21] | 0.14 |
| drop top 5% | 0.15 | - | 0.12 | - |

Every weighting, with or without the outlier, gives 0.12-0.22. The pooled slope (0.37 / 0.21) was low
because 402 over-claims magnitude, not because it points the wrong way. Attack failed: r^2 ~ 0.03-0.05 is
robust.

**A3. "Calibrated" is regression arithmetic, not evidence of under-resolution.** Both directions, within-state
advantages over the 8 fresh actions, truth noise removed where it biases:

| states | slope truth-on-critic | slope critic-on-truth | r^2 | sd critic / sd truth |
|---|---:|---:|---:|---:|
| all | 0.18 [0.13, 0.99] | 0.120 [0.05, 0.23] | 0.022 | 0.81 [0.27, 1.33] |
| without 402 | 0.79 [0.52, 1.05] | 0.063 [0.04, 0.09] | 0.050 [0.02, 0.09] | 0.28 [0.26, 0.31] |
| drop top 5% | 0.76 [0.45, 1.05] | 0.056 [0.03, 0.08] | 0.042 | 0.27 |

Gradient (pre-tanh), without 402: truth-on-critic 1.13 [0.41, 1.93] (h 0.1), 0.66 [0.40, 0.94] (h 0.3);
critic-on-truth 0.038 and 0.063; norm ratio 0.18 and 0.31.

- Truth-on-critic is unbiased under truth noise and equals 1 for *any* conditional-mean predictor, including
  an optimally shrunk pure-noise-plus-signal estimate. A cross-entropy / L2 regressor is driven to slope 1 by
  construction. It cannot distinguish "under-resolved" from "noisy".
- The reverse slope is the informative one: a unit of true advantage moves the critic by 0.06. The critic's
  amplitude of 0.28 true sds is 0.06 aligned plus 0.27 orthogonal. "What it claims is roughly real; it claims
  little" should read: **what it claims is on average unbiased, 95% of it is unrelated to the truth, and it
  responds to 4-6% of the true signal.** The amplitude-3x-too-small figure understates the attenuation of the
  aligned part (16x) and hides the unrelated part.

**A4. Per-state distributions, no trimming, truth noise handled by split halves.**

| per-state statistic | p10 / p50 / p90 | mean | frac > 0 | perfect-predictor ceiling | disattenuated mean |
|---|---|---:|---:|---:|---:|
| critic vs truth, Pearson over 8 actions | -0.52 / +0.14 / +0.68 | 0.100 +- 0.020 | 0.60 | 0.63 | **0.16** |
| critic vs 10-step truth (reliability 0.80) | -0.65 / +0.23 / +0.78 | 0.132 +- 0.023 | 0.62 | 0.89 | **0.15** |
| 1-step lookahead vs 10-step truth | -0.49 / +0.37 / +0.88 | 0.273 | 0.69 | 0.89 | 0.31 |
| 5-step lookahead vs 10-step truth | +0.26 / +0.88 / +0.98 | 0.737 | 0.95 | 0.89 | 0.83 |
| critic gradient cosine vs truth, h 0.3 (6-D) | -0.60 / +0.11 / +0.71 | 0.078 +- 0.021 | 0.58 | 0.55 | **0.14** |
| critic gradient cosine vs 10-step truth, h 0.3 | -0.62 / +0.14 / +0.80 | 0.119 +- 0.023 | 0.59 | 0.84 | **0.14** |
| critic gradient cosine vs own 1-step lookahead | -0.33 / +0.61 / +0.94 | 0.448 +- 0.022 | 0.82 | 1.00 | 0.45 |

A random 6-D direction has per-state cosine 0 +- 0.41, so the critic's per-state cosine distribution is a
random one shifted by 0.08-0.14. Self-consistency is robustly positive per state (median +0.61): that part
of 02 / 03 holds without any trimming.

What I tried that failed to break A: untrimmed actor-weighted alignment (same as trimmed); rank instead of
Pearson (same); a 2.7x less noisy truth (same 0.14-0.16).

---

## B. Is the critic at the ceiling once iterations are pooled? (`07_B_ceiling.py`, `07_B2_signal_share.py`)

Premise re-measured: signal share of one lambda-return target 0.061 [0.037, 0.099] at 4M and 0.058 [0.039,
0.081] at 8M (matches 03). At 12M (v29) it is 0.26 [0.08, 0.44], heavy-tailed; 03 did not tabulate it.

03's argument was "ridge / kNN on one chunk reach 0.02-0.18, the critic 0.15-0.29". Ridge may be weak and
the critic pools iterations, so I built the bound differently, from the truth itself. For the mean-gradient
field G(s) (pre-tanh-mean coordinates) and a whitened feature space Phi:

- E|G|^2: cross product of two independent truths at the same state (FD at a0, rollout half A; fresh-action
  score-function truth, half B). 4M: 0.78 [0.35, 1.26]. 8M: 4.2 [1.3, 8.4].
- b = energy of the true field inside span(Phi), from cross-lane products of unbiased estimates (oracle and
  score function; see D). f = b / E|G|^2 is the share of the truth Phi can represent at all.
- tau = noise energy of one chunk's unbiased projection. Pooling M chunks of a static field gives an aggregate
  cosine sqrt(M b / (M b + tau)) and a per-state field correlation of at most sqrt(f) times that.

| 4M, feature space | f (share of true field) | aggregate cosine, M = 1 / 10 / 30 | per-state corr ceiling, M = 1 / 10 / 30 / inf |
|---|---:|---|---|
| linear in obs (108 dof) | 0.023 [-0.02, 0.08] | 0.34 / 0.75 / 0.89 | 0.05 / 0.11 / 0.14 / 0.15 |
| actor last layer, top 32 PCs (192) | 0.042 [-0.02, 0.12] | 0.35 / 0.77 / 0.90 | 0.07 / 0.16 / 0.18 / 0.20 |
| actor last layer, top 128 PCs (768) | 0.058 [-0.08, 0.20] | 0.21 / 0.57 / 0.77 | 0.05 / 0.14 / 0.19 / 0.24 |
| actor last layer, all (3078) | 0.148 [-0.17, 0.47] | 0.16 / 0.47 / 0.67 | 0.06 / 0.18 / 0.26 / 0.38 |

8M: f = 0.01-0.02 in every space (CIs include 0); per-state ceilings 0.10-0.13 at M = inf.
Critic's measured per-state field correlation on the same axis: +0.23 [0.15, 0.48] at 4M, -0.01 [-0.28,
0.19] at 8M.

Reading:
1. **The "rough field" premise survives a different method.** At most a few percent of the true gradient
   field's energy is a smooth function of state in any space the policy can represent. An ideal estimator in
   those classes pooling 10-30 iterations reaches a per-state correlation of 0.11-0.26; pooling forever,
   0.15-0.38. The critic's 0.15-0.29 is not below that. Claim B holds in the metric it was stated in.
2. **But that metric is the wrong target.** The actor can only use the projection of the field onto its own
   tangent space. There, one rollout of an unbiased estimator has cosine 0.2-0.35 with the truth, and 10-30
   pooled rollouts 0.75-0.9. The critic's aggregate cosine in the same spaces is 0.5-0.9 at 4M (lower CI
   bounds 0.1-0.5; see D). So in the part that matters, the critic is worth somewhere between 1 and ~20
   rollouts of score-function data (CI that wide), not provably at the pooled ceiling and not provably below it.
3. **The critic does not keep what it learned.** Consecutive snapshots of the v55-dual run (iterations 244 /
   245 and 482 / 483), same states and actions (4096 off-run states from the oracle chunks, 8 policy actions):

   | pair | per-state gradient cosine, median (p10) | pooled gradient cosine, all / drop top 1% | rms change of the field / rms of the field (drop top 1%) | policy KL |
   |---|---|---|---|---:|
   | 4.01M -> 4.03M | 0.94 (0.59) | 0.64 / 0.81 | 0.53 / 0.90 = 59% | 0.017 |
   | 7.91M -> 7.93M | 0.91 (0.40) | 0.38 / 0.51 | 1.20 / 1.07 = 113% | 0.038 |

   One iteration changes the action-gradient field by 60-110% of its own rms (energy-weighted), and the
   median state's direction by cos 0.91-0.94. If the change were a random walk, the memory would be ~10-15
   iterations at the median state and ~1-2 where the gradient is large. The true field of a policy that moved
   by KL 0.02-0.04 cannot change that fast, so most of this is re-fitting noise. Limits: two snapshot pairs,
   off-run states, and I cannot separate a stable component from per-iteration noise with two points.

Net: "estimation-limited" is right about one chunk and about per-state correlation. "At or above what the
data supports" is not established for the multi-iteration, policy-projected quantity, and the per-iteration
rewrite is direct evidence that the critic is not accumulating across the 60-480 iterations it has seen.

---

## C. The transition channel (`07_CE_lookahead_vine.py`, `data/07_CE_log.txt`)

**(ii) rewards + critic V', expected label vs H = 400 truth** (within-state, noise removed from both sides
by cross-half products):

| n | all 512 states | without the 6 top-energy states | sd(E label) / sd(truth) |
|---|---:|---:|---:|
| Q(s, a) direct | 0.147 [0.13, 0.28] | 0.188 [0.11, 0.26] | 0.28 |
| 1 | 0.05 [-0.18, 0.52] | **0.447 [0.38, 0.51]** | 0.54 |
| 5 | 0.889 [0.86, 0.92] | **0.897 [0.86, 0.93]** | 0.90 |
| 10 | 0.985 [0.96, 1.01] | 0.987 | 0.99 |
| 25 | 0.992 | 0.994 | 0.99 |

The published 0.34-0.43 / 0.87 / "complete by ~10" reproduce (slightly higher with two-sided disattenuation).
n = 2-4 and 6-9 were never computed by the generator.

**(i) rewards only, (iii) V' only.** Not separable from the saved arrays: `checks_soft` stores the sum, and
neither the reward partial sums, the first reward, the states nor the fresh actions were saved. What is
identifiable:

- n = 1: the generator's own JSON (`model_based.*.reward_share`) gives FD energy of r(s, a) alone over FD
  energy of r + gamma V'(f(s, a)) = 0.0011-0.0015 in both spaces and both steps. The exactly-known control
  cost part, recomputed here, is 0.0002 pooled and 0.002 on the median state. So **at n = 1 at least 96% of
  the label's action-gradient amplitude is the critic's V'**, and that label ranks actions at 0.45 where Q
  manages 0.19. At one-step action-effect scale the critic's V is informative, 2.4x better than its own Q
  readout. That part of the claim is established.
- n = 5: 0.90 could be mostly five real rewards. The arrays cannot say. The one hint points *against*
  crediting V' unconditionally: at the flipped state V'(s_1) is anti-aligned with truth (-0.8) and the
  5-step label is +0.9, so there the rewards, not V', carry it.
- Exact missing measurement: for the 4M-30M datasets (which do store `states.state`, actions and snapshots)
  512 states x 9 actions x 16 CRN rollouts x 10 steps = 0.74M env steps, recording reward partial sums and
  gamma^n V'(s_n) separately for n = 1..10. I did not run it (no-simulation rule). It would also answer
  03's open question of whether the channel survives past 1M.

**CRN.** Per-branch noise variance of a within-state-centred label, CRN vs independent noise per action
(6 states dropped): n = 5: 0.068 vs 0.120; n = 10: 0.239 vs 0.338; n = 25: 0.80 vs 0.93. CRN removes 30-45%
of the noise at n = 5-10 and nothing by n = 50. It is a modest help, not the mechanism.

**What the published curve does not say.** 0.45 / 0.90 / 0.99 are properties of the *expected* label (64
rollouts averaged). One real branch gives (correlation of a single centred label with the true advantage):

| n | CRN, K = 8 | no CRN, K = 8 | ordinary single action, critic baseline |
|---|---:|---:|---|
| 1 | 0.447 | 0.440 | - |
| 5 | 0.537 | 0.440 | - |
| 10 | 0.398 | 0.360 | - |
| 25 | 0.237 | 0.208 | - |
| lambda-return 0.8 / 0.95 | - | - | 0.22 / 0.19 |

So the transition channel's advantage over an ordinary lambda-return sample is a factor ~2 in correlation at
n = 1-5 per label, not 0.9 vs 0.15, and it needs K actions from the same state to exist at all.

---

## D. Critic dQ/da vs PPO's score function from the same chunk (`07_D_headtohead.py`, `07_D2_chunk_reference.py`, `07_D3_combined.py`, `07_D4_controls.py`)

### Setup

Field: the gradient of E_z Q_soft(s, tanh(mu + sigma z)) with respect to the pre-tanh mean, a 6-vector per
state. Estimators at an on-policy sample (s_i, a_i, z_i):

- critic: dQ_c/da(s_i, a_i) (1 - a_i^2) (REPPO's pathwise term); critic_star: the same at a = tanh(mu).
- sf_gae(lam): A_hat z_i / sigma_i, GAE(lam) on soft rewards with the snapshot critic's E_pi Q as the state
  baseline (PPO's estimator; the baseline is the critic's because no separate V exists). lam = 0.95 is "PPO".
- sf_target: (trainer lambda-return target - baseline) z / sigma. cv_combo: critic + sf_gae0.95 - (critic's
  own advantage) z / sigma, i.e. the critic gradient as a Stein control variate.

Aggregation V = mean_s g(s) (x) F(s) for F = [1, obs] (108 dims), the actor's last hidden layer (3078 dims;
exactly the gradient of the mean head), and the full actor parameter gradient J^T g (279k dims). A constant F
(6 dims) carries no resolvable truth (|V*|^2 indistinguishable from 0 at 4M) and is dropped.

Truth. Two unbiased views of V* exist and I use both: (O) the oracle at the 512 states (H = 100; mean of the
FD-at-a0 gradient and the exact fresh-action score-function identity over the 8 fresh actions), and (S) the
chunk's own score function with a near-Monte-Carlo return (lam_ref = 0.99; 0.95 as a check). **Every inner
product is taken only between units in different env lanes**, so no product shares states, action noise or
return noise. |V*|^2 comes from the three cross products OO, OS, SS (inverse-variance weighted; they agree
within noise, e.g. 4M full-parameter: 0.54 / 0.26-0.55 / 0.58-0.62). Reported:

- gain = <E V_est, V*> / |V_est|: true first-order improvement per unit-norm step along what one rollout
  delivers. Needs no estimate of |V*|, so it is the robust head-to-head quantity.
- cos = gain / |V*|. Noisy because |V*|^2 is (its P(> 0) is given).

Controls (`07_D4`): permuting the critic field's rows across the chunk drops its gain from +0.178 to
+0.009 / +0.046 / -0.013 at 4M (linear space). At 8M the permuted field keeps +0.10-0.13 (the mean gradient
alone is aligned there). The oracle-only arm (`07_D`) and the chunk-only arm (`07_D2`) give the same signs
and sizes as the combined arm.

### Result (all states; gains with 95% CI; first number lam_ref 0.99, second 0.95)

| stage | space | critic gain | sf_gae0.95 gain | P(critic > SF) | SF one-rollout cos^2 (own reliability) | |V*|^2 P(> 0) |
|---|---|---|---|---:|---:|---:|
| 4M v23 | linear | 0.178 [0.07, 0.28] / 0.173 [0.08, 0.25] | 0.152 [-0.06, 0.31] / 0.156 [-0.03, 0.29] | 0.75 / 0.75 | 0.41 | 0.84 / 0.96 |
| | last layer | 0.379 [0.13, 0.63] / 0.384 [0.20, 0.57] | 0.297 [-0.09, 0.57] / 0.303 [-0.04, 0.53] | 0.81 / 0.85 | 0.36 | 0.78 / 0.96 |
| | full params | 0.592 [0.29, 0.87] / 0.562 [0.33, 0.78] | 0.428 [-0.11, 0.79] / 0.431 [-0.05, 0.73] | 0.87 / 0.88 | 0.34 | 0.80 / 0.97 |
| 8M v29 | linear | 0.357 [-0.04, 0.78] / 0.570 [0.26, 0.89] | 0.216 [-0.19, 0.56] / 0.348 [-0.05, 0.62] | 0.92 / 1.00 | 0.55 | 0.45 / 0.89 |
| | last layer | 0.721 [-0.13, 1.57] / 1.165 [0.58, 1.79] | 0.457 [-0.39, 1.18] / 0.721 [-0.07, 1.30] | 0.94 / 1.00 | 0.52 | 0.44 / 0.89 |
| | full params | 1.173 [0.04, 2.31] / 1.841 [0.89, 2.83] | 0.669 [-0.37, 1.67] / 1.045 [-0.04, 1.86] | 0.95 / 1.00 | 0.46 | 0.54 / 0.88 |
| 12M v23 | linear | 0.084 [-0.19, 0.36] / 0.180 [-0.04, 0.39] | -0.035 [-0.41, 0.37] / -0.009 | 0.70 / 0.85 | 0.04 | 0.33 / 0.41 |
| | last layer | -0.122 [-0.74, 0.40] / 0.179 [-0.31, 0.56] | 0.099 [-0.70, 0.97] / 0.062 | 0.39 / 0.62 | ~0 | 0.68 / 0.63 |
| | full params | 0.114 [-0.51, 0.66] / 0.445 [-0.11, 0.85] | 0.087 [-0.71, 0.97] / 0.096 | 0.58 / 0.82 | 0.05 | 0.59 / 0.62 |
| 12M v29 | linear | 0.216 [-0.17, 0.66] / 0.223 [-0.08, 0.58] | -0.33 [-0.64, 0.25] / -0.30 | 0.93 / 0.94 | <= 0 | 0.15 / 0.12 |
| | full params | 1.40 [-0.39, 3.61] / 1.08 [-0.14, 2.61] | -0.32 [-1.53, 1.26] / -0.22 | 0.96 / 0.95 | ~0 | 0.44 / 0.36 |
| 30M v23 | linear | 0.081 [-0.20, 0.38] / 0.134 [-0.13, 0.40] | 0.147 [-0.31, 0.61] / 0.178 | 0.51 / 0.55 | 0.09 | 0.61 / 0.74 |
| | last layer | 0.161 [-0.28, 0.56] / 0.202 [-0.21, 0.58] | 0.522 [-0.20, 1.17] / 0.543 | 0.31 / 0.32 | 0.25 | 0.71 / 0.90 |
| | full params | 0.085 [-0.87, 1.04] / 0.314 [-0.54, 1.14] | 0.533 [-0.44, 1.39] / 0.586 | 0.32 / 0.44 | 0.22 | 0.65 / 0.85 |

Upright-only subsets at 12M / 30M change nothing material. Cosines where |V*|^2 is resolved (4M, lam_ref
0.95): critic 0.67-0.74 [0.31-0.41, 1.8], sf_gae0.95 0.56-0.61 [0.0, 0.72]; the product cos_critic x cos_sf,
which does not need |V*|, is 0.40-0.45 [0.16, 0.59] at 4M and 0.43-0.76 at 8M, i.e. both are near 0.65 if
equal.

Other estimators, 4M cosines (lam_ref 0.99 / 0.95): critic_star 0.60-0.71 / 0.51-0.54; sf_gae0.8 0.81-0.89 /
0.71-0.73; sf_target 0.55-0.57 / 0.48-0.54; cv_combo 0.61 / 0.53-0.59. At 12M and 30M critic_star is mostly
negative (-0.6 to +0.1): the gradient at the mean action is worse than the gradient averaged over sampled
actions. Using the critic as a control variate buys nothing over plain SF (cv_combo ~ sf_gae0.95) except
possibly at 30M (0.45-0.56 vs 0.40-0.51).

Per state (4M): the critic's field correlation with the true mean-gradient field is 0.23 [0.15, 0.48]; one
score-function sample's is sqrt(E|G|^2 / E|A_hat z / sigma|^2) = sqrt(0.78 / 83.8) = 0.10. Per state the
critic is worth about 5-6 score-function samples at that state.

### Answer

- **Per rollout, REPPO's critic gradient is not worse than PPO's estimator. It is statistically equal, with
  point estimates 1.2-1.8x better in true gain per unit step at 4M and 8M** (P 0.75-0.95 under the unbiased
  reference, 0.75-1.00 under lam_ref 0.95), unresolved at 12M (neither estimator has a resolvable gain against
  the v23 truth; against v29 the critic's gain is positive at P 0.92-0.97 while the SF's is not), and
  **0.2-0.75x at 30M** (P(critic better) 0.31-0.55).
- The critic's gain is positive with P >= 0.98 at 4M and 8M in the full parameter space. The oracle-only arm
  alone gives the same sign at P 0.92-0.98 (4M).
- One rollout of GAE(0.95) score function has cos^2 0.34-0.55 with the true projected gradient at 4M-8M and
  roughly 0-0.25 at 12M-30M. PPO's per-rollout direction is therefore itself poor late; this is not specific
  to REPPO.
- Consequence for the root cause: "the critic's action gradient is worse than what PPO would extract from the
  same data" is **not supported** at 4M-12M. The low per-state numbers (0.1-0.3) understate the critic,
  because most of the true field is not representable by the policy and the critic is good on the part that
  is. What the data cannot exclude is that the critic is far below what its 60-480 iterations of history
  should allow (B).

Caveats that could move this:
1. The snapshot critic has not been fit on the chunk it is scored on (one iteration stale); the trainer's
   actor reads the freshly fit critic. This biases against the critic if fitting helps, for it if it
   memorises.
2. The SF baseline is the critic's E_pi Q, not a PPO value net; lam_ref 0.95 shares a small bootstrap bias
   with the critic (the lam_ref 0.99 column is the clean one and is noisier).
3. Mean direction only (no log-std gradient), soft objective, Euclidean cosine in each parameterisation, no
   Adam, no multi-epoch reuse, no clipping.
4. PPO's noise is independent across iterations and averages out over consecutive updates; a critic's error
   is correlated across iterations. Equal per-rollout cosine does not imply equal progress over many updates.
5. One seed, one chunk per stage; the truth aggregate is resolved (P(|V*|^2 > 0) >= 0.84) only at 4M and, under
   lam_ref 0.95, at 8M and 30M.

---

## E. Vine economics at 1M (`07_CE_lookahead_vine.py`)

Label: K actions from one visited state, common random numbers, n real steps then the critic's V', centred
within the state. Cost K n env steps for K - 1 contrasts. Two numbers per n: the ceiling rho_n (what infinitely
many such labels converge to, C(ii)) and the single-label correlation c_n with the true advantage. Information
per env step = (K - 1) / (K n) x c^2 / (1 - c^2). Reference: an ordinary sample, one label per env step, with
the study's 5-7% share (0.053-0.075 per step; my own single-label correlation at 1M, 0.19-0.22, gives
0.037-0.05). Six top-energy states dropped; with them, n = 1 is useless when pooled (corr 0.05) and n = 5 is
0.36 instead of 0.54.

| n | ceiling rho_n | single-label corr (CRN) | label reliability r_n | info per env step, K = 2 / 4 / 8 | relative to an ordinary sample (K = 8) |
|---|---:|---:|---:|---|---|
| 1 | 0.45 | 0.447 | 1.00 (deterministic) | 0.125 / 0.187 / 0.219 | **2.9-4.1x** |
| 5 | 0.90 | 0.537 | 0.36 | 0.041 / 0.061 / 0.071 | **0.95-1.3x** |
| 10 | 0.99 | 0.398 | 0.16 | 0.009 / 0.014 / 0.016 | 0.22-0.31x |
| 25 | 0.99 | 0.237 | 0.05 | 0.001 / 0.002 / 0.002 | 0.03-0.04x |

- **Break-even is n ~ 5 at K = 8** (n ~ 3-4 at K = 2, interpolated; n = 2-4 were not saved). Beyond n = 5 a
  vine step is worth less than an ordinary step.
- **Best per env step is the shortest branch with the largest K**: n = 1, K = 8 is 3-4x an ordinary sample.
  K = 2 gives 57% of K = 8's efficiency, K = 4 gives 86%.
- The n = 1 label is noise-free, so it is learned immediately, but it converges to the critic's own one-step
  lookahead, whose correlation with the truth is 0.45 (and negative at off-distribution states). It can lift
  the Q readout from 0.19 to at most 0.45; it cannot exceed what V' knows.
- Without CRN the information per step falls by ~4% (n = 1), 41% (n = 5), 21% (n = 10).
- n = 2-3 is probably the optimum (ceiling rising from 0.45 toward 0.9 while noise is still below the
  signal), but that is an interpolation; the arrays have no such horizon.
- All of this is 1M only. At 4M-8M the ordinary sample's share is the same 5-7%, but the vine side is
  unmeasured.

---

## What I could not do

- D at 1M (no observations, policy moments or on-policy returns for that snapshot).
- C(i) / C(iii) for n >= 5 and any n in 2-4, 6-9; E for those n; C and E after 1M. A 0.74M-step branch run on
  the 4M-30M datasets would supply all of it.
- A PPO value-network baseline; the in-sample (freshly fit) critic; log-std gradients.
- Whether the per-iteration rewrite in B is noise or tracking, beyond the bound from the policy's KL.

## Files

- Report: `benchmarks/reppo_rootcause/07_redteam_oracle.md`
- Scripts: `benchmarks/reppo_rootcause/scripts/07_common.py`, `07_A_oracle_1M.py`, `07_B_ceiling.py`,
  `07_B2_signal_share.py`, `07_CE_lookahead_vine.py`, `07_D_headtohead.py` (oracle-only arm),
  `07_D2_chunk_reference.py` (chunk-only arm), `07_D3_combined.py` (final arm), `07_D4_controls.py`
- Logs / results: `benchmarks/reppo_rootcause/data/07_A_log.txt`, `07_B_log.txt`, `07_B2_log.txt`, `07_CE_log.txt`,
  `07_D_log.txt` and `07_D2_log.txt` (4M only), `07_D3_log.txt`, `07_D3_results_*.json`, `07_D4_log.txt`
