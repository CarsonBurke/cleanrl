# Cross-sectional vol panel (real 5-minute bars, 200 liquid US stocks)

Files: `panel_stream.py` (linear ceilings), `panel_hd.py` (`Bank`: 8 channels x 32 lags = F=257 causal features), `panel_hd_mlp.py` (offline MLP ceiling vs streaming Adam), `panel_hd_gate.py` (per-sample precision gate), `panel_hd_pp.py` (per-parameter evidence gate).

Data: `trading_bot_0/long_data/bars/*.300.bars`, top-200 median-dollar-volume non-ETF names, 112241 bars, 94.3% valid (bar, stock) samples. Target `vol`: next-bar squared unit-vol return, clamped at 25, centred on the fit-window mean. Score: relative MSE vs the constant fit-window mean on the last 40% of time (lower is better; 1.0 = nothing captured). PERM = target time-permuted (nothing learnable). Streaming = one optimizer step per bar on that bar's cross-section (~190 samples), scored online after the 60% cut, no refits.

## Ceilings

| Model | REAL | PERM |
|---|---|---|
| ridge, 16 features, hindsight fit first 60% | 0.9207 | 1.0000 |
| ridge, F=257, hindsight fit first 60% | 0.8891 | 1.0013 |
| streaming AdamW linear, F=257, best lr 3e-4 | 0.8931 | 1.0092 |
| offline MLP 257-256-256-1, AdamW, early-stopped, fixed at test | 0.7799 | - |
| offline MLP 16-feature, width 256 / 512 | 0.8072 / 0.8068 | - |
| streaming Adam MLP 16-feature, best lr 3e-4 | 0.8105 | 1.0007 |
| streaming Adam MLP 257-256-256-1, lr 1e-4 / 3e-4 / 1e-3 / 3e-3 | 0.7507 / 0.7501 / 0.7722 / 0.8616 | 1.0061 (3e-4) |
| streaming Adam MLP width 1024, lr 1e-4 / 3e-4 | 0.7523 / 0.7717 | 1.0065 |

## Per-sample precision gate (`panel_hd_gate.py --gate prec`)

RLS log-variance readout from the net's own last hidden state (float64, horizonless), sample weight `1/Var_hat(y|x)`, mean-normalised; shuffle permutes weights across samples within the bar.

| lr | prec | prec_shuffle | Adam |
|---|---|---|---|
| 1e-4 | 0.9116 | 0.7624 | 0.7507 |
| 3e-4 | 1.0076 | 0.7645 | 0.7501 |
| 1e-3 | 0.9196 | 0.7941 | 0.7722 |

## Per-parameter evidence gate (`panel_hd_pp.py --gate js`)

Each parameter keeps plain sums of its per-bar gradient, `S1 = sum g`, `S2 = sum g^2`; gate `p = (1 - S2/S1^2)^+` read before the bar's gradient is added; multiplies the Adam step post-optimizer. No EMA, no twin, no discount, no tuned scale. `js_shuffle` permutes `p` across the parameters of each tensor (level kept, correspondence broken). Width 256, seed = init seed (data fixed).

| lr | none (Adam) | js | js_shuffle | js gate level |
|---|---|---|---|---|
| 1e-4 | 0.75069 | 0.75743 | 0.81541 | 0.223 |
| 3e-4 | **0.75012** | 0.73811 | 0.76765 | 0.152 |
| 1e-3 | 0.77219 | **0.72869** | **0.75126** | 0.131 |
| 3e-3 | 0.86164 | 0.73350 | 0.75189 | 0.226 |
| 1e-2 | - | 0.76334 | 0.76740 | 0.331 |

Init seeds 1/2/3 at own best lr: Adam 0.75012 / 0.74947 / 0.74925; js 0.72869 / 0.72819 / 0.72749; js_shuffle 0.75126 / 0.75248 / 0.75141; js@3e-3 0.73350 / 0.73545 / 0.73322.

Test window split into 8 contiguous blocks (seed 1, own best lr):

| block | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Adam 3e-4 | 0.7614 | 0.7554 | 0.7770 | 0.7653 | 0.7810 | 0.7674 | 0.7106 | 0.6901 |
| js 1e-3 | 0.7433 | 0.7329 | 0.7513 | 0.7443 | 0.7632 | 0.7426 | 0.6863 | 0.6728 |
| js_shuffle 1e-3 | 0.7596 | 0.7553 | 0.7784 | 0.7638 | 0.7814 | 0.7689 | 0.7139 | 0.6953 |
| js - Adam | -0.0181 | -0.0225 | -0.0257 | -0.0210 | -0.0178 | -0.0248 | -0.0243 | -0.0173 |

js - Adam: mean -0.0214, sd 0.0032, 8/8 blocks. js - js_shuffle: 8/8 blocks. PERM stream: js@1e-3 1.00117 (gate level 0.106) vs Adam@3e-4 1.00614.

Streaming runtime: ~110-230 s per (arm, lr) on one RTX 5090 (3 concurrent).

### Variants (`--kappa`, `--discount`, `--cap`, `--width`), seed 1, REAL

Gate `(1 - kappa/t^2)^+`, `t^2 = S1^2/S2`. Discount d: `S1 *= (1-d)`, `S2 *= (1-d)^2` per bar. Cap c>0 replaces the gate with `min(t^2/kappa, c)` (linear in evidence, amplifies above kappa).

| config | lr 3e-4 | 1e-3 | 3e-3 | 1e-2 | gate level (best lr) |
|---|---|---|---|---|---|
| Adam w256 | **0.7501** | 0.7722 | 0.8616 | - | - |
| Adam w1024, lr 3e-5 / 1e-4 / 3e-4 | 0.7641 / **0.7523** / 0.7717 | | | | - |
| Adam w2048, lr 3e-5 / 1e-4 | **0.7550** / 0.7592 | | | | - |
| js k=1 w256 | 0.7381 | **0.7287** | 0.7335 | 0.7633 | 0.131 |
| js k=1 d=1e-4 w256 | | 0.7321 | 0.7445 | | 0.116 |
| js k=1 d=1e-3 w256 | | 0.7528 | 0.7721 | | 0.143 |
| js k=3 w256 | | 0.7366 | **0.7255** | 0.7292 | 0.033 |
| js_shuffle k=3 w256 | | | 0.7538 | | 0.038 |
| js k=5 w256 | | | 0.7356 | 0.7301 | 0.014 |
| js k=10 w256 | | 0.7672 | 0.7508 | | 0.011 |
| js k=3 w1024 | | 0.7256 | **0.7202** | 0.7247 | 0.097 |
| js k=3 w2048 | | | **0.7202** | 0.7238 | 0.141 |
| js k=3 cap=1 w1024 | | | 0.7525 | | 0.324 |
| js k=3 cap=4 w1024 | 0.7227 | 0.7251 | 0.7528 | | 0.220 |
| js k=3 cap=16 w1024 | 0.7227 | 0.7253 | | | 0.219 |

js k=3 w1024 lr 3e-3, seeds 1/2/3: 0.7202 / 0.7195 / 0.7192; blocks (seed 2): 0.7352 0.7205 0.7387 0.7353 0.7566 0.7343 0.6783 0.6638 (Adam w256 blocks above: worse in 8/8, mean gap -0.030).

## Per-(sample, parameter) at optimizer-state cost (`panel_hd_coh.py`, width 1024, seed 1, kappa 3)

Manual backward; per-sample gradient of a dense layer is rank-1 so `sum_t |delta_ti||x_tj|` (L1 mass) is one extra matmul. Sign-agreement gating of samples reduces algebraically to (G, L1). Runtime equal to autograd Adam (~240 s per lr, 3 concurrent). Manual gradients checked against autograd on the first bar.

| arm | lr 3e-4 | 1e-3 | 3e-3 | 1e-2 | gate level |
|---|---|---|---|---|---|
| js (S1, S2 of batch gradients; reference) | | 0.7283 | **0.7174** | | 0.028 |
| l1: t^2 = (2/pi) N S1^2 / L1^2 | | 0.7864 | 1.0352 | | 0.96-0.99 (heavy tails: L1 << sigma, null anti-conservative) |
| coh: js x within-bar coherence (1 - kappa_c (pi/2) L1_bar^2 / (n G^2))^+ | | 0.7437 | 0.7288 | | 0.032 |
| js + sample clip: output residual winsorized at 3 rms (Huber-type control) | | 0.7620 | 0.7479 | 0.7561 | 0.035 |
| js + unit clip: delta_ti winsorized per hidden unit at 3 x own running rms | | 0.8987 | 0.8642 | 0.8290 | 0.079 |
| Adam + unit clip | 0.8957 | 1.0288 | | | - |

## Alternatives to gating the current Adam step (width 1024, seed 1, bracketed unless marked EDGE)

| mechanism | file / flags | lr 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | activity |
|---|---|---|---|---|---|---|---|---|
| EVT k=3: per-parameter sequential test, step S1/sqrt(S2) when t^2>=k, then reset S1=S2=0 | `panel_hd_evt.py --arm evt --kappa 3` | 0.9072 | 0.8985 | 0.9091 | 0.9299 | | | fire rate 0.0003-0.0023 |
| EVT k=6 | `--kappa 6` | 0.9072 | 0.9071 | 0.7835 | **0.7403** | 0.7568 | 9.32 | fire rate <5e-5 |
| EVT k=12 | `--kappa 12` | 0.8811 | 0.8754 | 0.8133 | 0.7665 EDGE | | | fire rate <5e-5 |
| cum: step = S1/sqrt(S2 N) (cumulative mean over cumulative rms), no gate | `panel_hd_pp.py --step cum` | | 0.8974 | 1.7025 | 3.3067 | 8.7613 | | - |
| cum + js gate k=3 | `--step cum --gate js --kappa 3` | | | 0.7670 | 0.8166 | 56.98 | | level 0.14-0.18 |
| snr: per-parameter smoothing horizon h = kN/t^2 in [2, N], step m_h / cumulative rms | `--step snr --kappa 3` | | 1.0073 | 1.3925 | 1.5739 | | | mean 1/h ~ 0 (almost every parameter at the cumulative horizon) |
| snr k=10 | `--step snr --kappa 10` | | 1.0717 | 1.0280 | 4.0565 | | | |
| snr k=3, horizons shuffled across parameters | `--gate hshuffle` | | | 1.1775 | 2.8468 | | | |
| consol d=1e-3: w = w_slow + w_fast, ungated Adam step into w_fast leaking at d, fraction gate_js consolidated per step | `--gate consol --fast-decay 1e-3 --kappa 3` | 0.7741 | 0.9049 | 1.6973 | | | | level 0.016-0.093 |
| consol d=1e-2 | `--fast-decay 1e-2` | **0.7563** | 0.8254 | 1.3169 | | | | |
| consol d=1e-3, gate shuffled | `--gate consol_shuffle` | | 0.9070 | 1.4034 | | | | |

Reference on the same width: Adam 0.7523 (lr 1e-4), js k=3 0.7202 (lr 3e-3).

## Controlled nonlinear learner comparison: categorical teaching plus JS

`panel_distributional_model_v1.py` / `panel_distributional_eval_v1.py`,
`panel_specialist_model_v2.py` / `panel_specialist_eval_v2.py`, and
`panel_frontier_model_v3.py` / `panel_frontier_eval_v3.py`.

Same frozen panel features, targets, masks and order within each comparison.
One CUDA-compiled, graph-replayed streaming pass, predict before update, seed 1.
Each family independently selects among six learning rates on the 40–60% prefix
window; selected configurations are fixed before scoring the forward suffix.
Target centering uses the first 60%, including that selection window. Forward
labels never select models. No input/target redesign is credited as a learner gain.

The categorical head uses 33 uniform support points spanning the existing raw
target range [0,25], with expectation-preserving two-hot labels. Its decoded
mean is scored by the same MSE as scalar models. JS multiplies the Adam step
using prior-history parameter-gradient evidence; it is not a new state router.

| Stock cohort | Scalar Adam | Scalar + JS | Categorical CE + JS | MSE reduction vs scalar JS |
|---|---:|---:|---:|---:|
| Original 200 | 0.769824 | 0.748147 | **0.740715** | 0.993% |
| Next disjoint 200 | 0.745833 | 0.724875 | **0.714512** | 1.430% |
| Third disjoint 200 | not run | 0.748038 | **0.738839** | 1.230% |

Scores are sample-weighted relative MSE on each experiment's consumed forward
window, not comparable across cohorts as difficulty changes. Each has six
observed temporal blocks, with the last partial. CE+JS wins 5/6, 6/6 and 5/6
against scalar JS respectively. Actual symbols are verified disjoint; market
dates overlap and SPY is shared, so these are not independent market replications.

### Capacity and same-head loss controls (v3)

| Learner | Active parameters | Prefix-selected LR | Real relative MSE | Permuted relative MSE |
|---|---:|---:|---:|---:|
| Scalar 256 + JS | 132097 | 1e-3 | 0.748038 | 1.000646 |
| Scalar 267 + JS | 140710 | 1e-3 | 0.748392 | 1.000821 |
| Categorical MSE + JS | 140321 | 3e-4 | 0.747103 | 0.999968 |
| **Categorical CE + JS** | **140321** | **3e-3** | **0.738839** | **1.001386** |

Same categorical initialization, support and prior-history gate; only the loss
differs. Scalar width 267 is the smallest width meeting the categorical active
parameter budget. CE+JS lowers MSE **1.276% versus that scalar** and **1.106%
versus categorical MSE+JS**, winning 5/6 observed blocks against both, across
29,436 forward bars and 5,704,415 valid stock-bar samples.

The gain survives capacity and head-parameterization controls. Null noise
fitting is slightly higher for CE+JS, so this is not pure noise suppression.
CE's logit gradient is `p-q`; half-squared decoded-mean error gives
`(mean-y)*p*(support-mean)`. The former supplies direct teaching on outcomes
currently assigned low probability. This mathematical distinction is not a
separately measured causal mediation result.

Rejected hypotheses: standalone CE did not beat scalar JS in v1. Learned
state-dependent expert routing in v2 scored 0.736519, worse than learned-constant
routing's 0.728660 in all six observed blocks and worse than dense CE+JS.

Verification: 35 v1/v2 contracts passed (job 5603); 27 v3 contracts passed (5621),
including independent autograd/Adam comparisons and CUDA capture/restoration.
Independent reviews found no model defects; an unsupported NaN-feature test
fixture was corrected without changing the learner. v3 saves a prefix lock
manifest before the suffix; the null authenticates its digest and raw scores.
Both runs verify unchanged source/data/stream bytes and all group clocks.
Root TensorBoard final steps match consumed artifacts.

Jobs: v1 real/permuted 5587/5593; v2 5604/5612; v3 5622/5624. Limit 1, priority 0,
one attempt, 45-minute learning limits. All six views intentionally pruned;
no continuing pruned streams or treating partial windows as full completion.
Active-parameter matching does not establish equal FLOPs, optimizer storage or
standalone latency. No PPO, profitability, general optimizer or optimality claim.

Durable evidence:
[v1](../../../benchmarks/plasticity/panel_distributional_v1_evidence.json),
[v2](../../../benchmarks/plasticity/panel_specialist_v2_evidence.json),
[v3](../../../benchmarks/plasticity/panel_frontier_v3_evidence.json).

## Latest information, causal memory, and actual return prediction

The v4 memory study separates information timing from model architecture.
The frozen Bank uses t-1…t-32 even though forecasting occurs after observing t.
The latest frame uses t…t-31; the memory frame adds 24 causal EWMA channels
at half-lives 64/256/1024. Volume surprise already exists in the Bank.

| Volatility learner | Development relative MSE | Rank400 forward confirmation |
|---|---:|---:|
| Old-frame scalar Adam | .750123 | .697471 |
| Old-frame categorical CE+JS | .720435 | .666080 |
| Latest-frame categorical CE+JS | .668281 | .617618 |
| Memory-frame categorical CE+JS | .666223 | .614693 |
| Memory exp-MSE | .703657 | .644986 |
| Memory exp-QLIKE | .699639 | .637332 |

Latest information accounts for a 7.28% reduction versus old-frame CE+JS;
memory adds only 0.474% beyond latest-frame CE+JS, below the 10% incremental
memory target. Memory CE+JS is 11.87% below old scalar Adam, but attributing that
whole gain to memory would be wrong. Its randomized-target relative MSE is
1.001864. These are **volatility**, not signed-return, scores.

### Direct signed returns and shared representations (v5/v6)

Raw targets are uncentered, unclipped retained-close log returns. Training uses
the same returns divided by causal RMS through t; reported raw forecasts
multiply by that known denominator. The zero baseline is genuinely raw zero.
The existing abs(return)>.2/missing-data exclusion is retained and disclosed;
“unclipped targets” does not mean an unfiltered data source.

Five families test old/latest/memory Adam and memory with correctly paired or
within-bar permuted auxiliary volatility teaching. v6 broadens the prefix
search to 60 settings per family, including auxiliary weight zero. Both
auxiliary families select **zero auxiliary weight**. LR/beta2 selection uses
normalized MSE on the 40–60% prefix interval, never raw suffix scores.

| Prefix-locked family | Development raw MSE / zero | Rank400 confirmation raw MSE / zero | Signed-null raw MSE / zero |
|---|---:|---:|---:|
| Old Adam | 1.000071 | 1.000024 | 1.000017 |
| Latest Adam | .999824 | .999351 | 1.000039 |
| Memory Adam | .999817 | .999352 | 1.000039 |
| Auxiliary, weight zero | 1.000073 | .999423 | 1.000026 |
| Signed ridge1 | 1.000236 | 1.000176 | 1.000205 |
| Zero | 1 | 1 | 1 |

Memory Adam's raw confirmation gain is **0.0648%**, not a substantial return
prediction breakthrough. All three observed blocks improve on zero, but they
are not independent seeds or a significance test. The >=1% raw-MSE target is
unmet, and neither long memory nor the tested auxiliary objective improves on
latest-only Adam in confirmation. No trading costs or profitability were tested.

The confirmation interval is rank400 [90145,101182), beyond the previously
consumed v3 interval, not globally pristine market dates. The survivor-selected
universe and coverage filtering remain limitations: adjacent retained bars can
bridge gaps, and the nominal `first_session_before` argument is not enforced by
the frozen panel builder. Calendar overlap and shared SPY preclude independent
market-replication claims.

Jobs: v4 development/confirmation/null 6128/6131/6132; v5 return development
6138; v6 development/null 6154/6163; corrected confirmation 6168. Job 6162 was
cancelled before starting because its command omitted the rank400 cache; no
learning result was reused from it. All learning jobs use queue limit 1,
priority 0 and one attempt. v6 development/confirmation replay totals are
524/457 seconds for the whole candidate bank, not standalone model costs.

Full arrays, prefix locks, numerical failures, source/data authentication and
job ledgers:
[predictive optimum evidence](../../../benchmarks/plasticity/predictive_optimum_v1_evidence.json).

### Hierarchical return inference (v7): reject

The next strategy shares statistical strength rather than auxiliary gradients:
stock coefficients have a Gaussian prior around a learned common coefficient.
Pooled, independent and hierarchical families each receive 64 prefix candidates.
Positive Schur updates compute the declared static Gaussian posterior online;
independent natural-Gram/RHS batch solves audit the final coefficients.
All three families use the same 13 **unclipped** signed-return features.

| Prefix-locked learner | Development raw MSE / zero | Reused rank400 | Additional rank600 | Rank0 signed null |
|---|---:|---:|---:|---:|
| Pooled ridge | 1.000142 | 1.000425 | 1.000103 | .999993 |
| Independent ridge | 1.000229 | 1.001188 | 1.000182 | 1.000078 |
| **Hierarchical ridge** | **1.027653** | **1.011581** | **2283.778294** | **1.018349** |
| Frozen memory Adam | .999817 | .999243 | .998745 | 1.000022 |
| Zero | 1 | 1 | 1 | 1 |

The hierarchy fails catastrophically on rank600 despite finite candidates and
agreement with independent end-state coefficient solves. Numerical agreement
does not validate the likelihood, representation or statistical assumptions.
No candidate was replaced with zero, clipped after failure, or silently dropped.
The selected independent prior 1e10 and hierarchical global prior 1e7 are search
upper endpoints; this rejects the tested strategy, not every hierarchical model.

Rank400 scores its full [60709,101182) suffix, explicitly reusing observations;
rank600 scores its own [47413,79022) suffix. The stock sets are disjoint from
rank0 and each other, but dates overlap and SPY is shared. Memory Adam's 0.1255%
raw-MSE gain on rank600 remains below the 1% target. It is not evidence of
profitability or independent market replication.

A full development-only magnitude diagnostic reproduces saved target counts,
raw target energy and prediction energy before attribution. The maximum absolute
input is 182,321.55; the largest stock Gram diagonal is 7.98e11. Thus a prior
precision of 1e10 does not imply near-zero forecasts on this feature geometry.
Only 255 of 8,799,705 suffix samples (0.00290%) have an absolute feature above
1,000, but they account for 79.65% of hierarchy raw prediction energy and 99.995%
of independent-ridge energy. These are **feature-magnitude associations**, not
formal posterior leverage or proof of the cause of rank600 failure. Inconsistent
lag/forecast volatility scaling and misspecified observation uncertainty remain
plausible explanations, not established fixes.

All four full-history jobs completed: development 6189, rank400 replay 6217,
rank600 6218, signed null 6219; diagnostic 6220. All use queue limit 1.
Three final posterior audits agree in each run; source/data/lock hashes,
observation clocks and durable score bounds are verified. Full learning walls
are 160/120/112/100 seconds, with joint-bank replay 38/32/26/37 seconds.
No equal-compute or global-optimum claim.

Artifacts:
[hierarchy plan](../../../benchmarks/plasticity/structural_inference_v7_plan.json),
[full evidence](../../../benchmarks/plasticity/predictive_optimum_v1_evidence.json),
`runs/panel_return_hierarchy_v7_{development,replay_rank400,confirmation_rank600,null}/`,
and `runs/panel_return_hierarchy_v7_leverage_diagnostic/results.json`.
