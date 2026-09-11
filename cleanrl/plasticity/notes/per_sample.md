# Per-sample updates

| File / configuration | Measurement / mechanism |
|---|---|
| `hetvar_stream.py`; Linear(17,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1), all inputs useful | Held-out noise-free MSE; 8 paired seeds resample teacher/stream/noise field/init; each arm independently LR-selected on bracketed grids; hetero 2 Adam grid extended to 2e-6..1e-3 after optimum 2e-5 hit first grid edge. |
| `agree`; `cos²(g_t,Adam m)/(1-cos²)` | Error 0.3707 versus agree_shuffle 0.3702. |
| `hetvar_ta`; own last-hidden-state linear log-variance readout, Gaussian NLL, normalized LMS | `c=[1/Var_hat(r\|x)]*(nu+1)/(nu+z²)`, divide by EMA(c), cap 20; per-sample readout rate, batch×rate<2. |
| `hetvar_ta`; twin readout on random past residual² mispaired with current state | Positive-part shrinkage `1-V_twin/V_pred`; fixed decay 0.3 cost 60% of gain, decay 0 added +0.018 null-cell error; lag-1→random-past ring on SPY c_sd 0.36→0.70. |
| `hetvar_ta`; nu estimated from standardized-residual kurtosis `kappa=3(nu-2)/(nu-4)` | Gaussian nu→infinity; fixed nu=5 Gaussian delta +0.005, adaptive -0.0003. |

| `hetvar_stream.py` cell; paired error difference versus own-LR Adam, 8 seeds | Oracle 1/sigma² | hetvar | hetvar_ta | huber | shuffle |
|---|---:|---:|---:|---:|---:|
| Homoscedastic | 0.000 | -0.0000 | -0.0002 | +0.001 | 0.0000 |
| Hetero 2 | -0.133 | -0.131 | -0.141 | -0.038 | +0.025 |
| Hetero 2 + t(2.5) | -0.116 | -0.116 | -0.148 | -0.081 | +0.025 |
| Teacher switch @50% | -0.093 | -0.091 | -0.099 | -0.012 | +0.042 |
| Low noise | +0.039 | -0.022 | -0.026 | -0.012 | +0.016 |
| Batch 32 | -0.099 | -0.116 | -0.132 | -0.047 | +0.025 |
| Batch 512; bracketed grid | -0.103 | -0.107 | -0.103 | — | +0.091 |

| `hetvar_stream.py oracle_*`; gate `b²/(b²+sigma²)`, b=f_theta−f*, 8 seeds, bracketed own-LR grids; paired difference vs Adam | Oracle 1/sigma² | Oracle predictability | Predictability shuffle |
|---|---:|---:|---:|
| Signal proportional to sigma | +5.96 (9.5 versus 3.6) | -0.35 | +0.33 |
| Hetero 2 | -0.133 | -0.188 | +0.022 |
| Homoscedastic | 0.000 | -0.040 (17%) | +0.052 |
| Null stream | -0.023 | -0.024 | — |

| Estimator / file | Homoscedastic / hetero 2 / signal-scales error difference vs Adam |
|---|---|
| `hetvar_stream.py wiener/snr`; last-hidden-state linear residual probe | +0.05 / -0.06 / +0.71; approximately own shuffle. |
| `hetvar_stream.py ntk`; Nadaraya–Watson gradient-kernel estimate with sign-randomized twin | +0.09 / -0.05 / +1.85; approximately own shuffle. |

| `rethink/Rethink{A,B,C}.py`; three independent designs of per-perceptron gate from own input `a` on the current sample; paired diff vs Adam, 8 seeds, bracketed grids | hetero 0 | hetero 2 | signal ∝ σ | null | switch 0.5 |
|---|---:|---:|---:|---:|---:|
| A `wiener`: held-out linear mu(a), log-var s2(a), gate mu²/(mu²+s2), pre-Adam, mean-normalised | +0.039 (shuf +0.047) | -0.114 (shuf +0.026) | +0.745 (shuf +0.741) | +0.009 EDGE (shuf +0.009) | -0.055 (shuf +0.041) |
| A precision-only ablation 1/s2(a) | +0.001 | -0.122 (shuf +0.031) | +0.997 (shuf +0.949) | +0.0005 | -0.070 (shuf +0.069) |
| B `cond`: prequential E[r\|a], log Var(r\|a), ratio-calibrated, gate on Adam m only | +0.065 EDGE (shuf +0.078) | -0.073 (shuf +0.010) | +0.069 n.s. (shuf +2.34) | +0.0008 EDGE (shuf -0.0001) | -0.025 (shuf +0.015) |
| C `pp_var`: RLS of goal pre-activation on [1,a], log-linear RLS of held-out residual², JS-calibrated, gate sig/(sig+s²(a)), pre-Adam | +0.021 (one seed NaN at lr 5e-4; shuf +0.002) | -0.149 (shuf +0.007) | +0.195 n.s. (shuf +0.225) | +0.0009 (shuf +0.0007) | -0.109 (shuf +0.017) |
| C `pp` (conditional-mean fixable power added) | +0.055 | -0.053 | — | — | — |
| Oracle predictability, same cells | -0.040 | -0.188 | -0.346 | -0.006 | -0.166 |
| A, B, C each report the conditional-mean (predictability) term null or harmful in every cell isolating it; all measured gain from the conditional-variance term. | | | | | |

| Corrected `sample_stream.py`; 10 arms ×11 LRs ×65,536 single observations, jobs 5294/5295, all selected LRs interior | df=2 Student-t + midpoint switch sustained clean MSE | hetero=2 + midpoint switch sustained clean MSE |
|---|---:|---:|
| Adam | 0.374600 | 0.383050 |
| Known-variance sample weighting | 0.374600 | 0.290230 |
| Huber | 0.342432 | 0.357595 |
| Learned heteroscedastic weighting | 0.374370 | 0.303763 |
| Student-t learned heteroscedastic weighting | 0.332865 | 0.290765 |
| Student-t asymmetric variant | 0.331251 | 0.290395 |
| Shuffled heteroscedastic weighting | 0.374534 | 0.465480 |

| `stock_stream.py` configuration | Measurement |
|---|---|
| SPY 5-minute `trading_bot_0/long_data/bars/SPY.300.bars`; 468,054 bars, 2016-08-22..2026-08-19 | TBBARS01, 64-byte header, 36-byte `<q6fI>` records (epoch-ms, OHLC, volume, vwap, count); 224 causal features, 7 channels ×32 lags divided by trailing EWMA. Default target subtracts a known trailing return mean, divides by trailing volatility, then clips ±10; normalized zero is **not** raw-return zero. B=1 predict-before-update; real and time-permuted labels. |
| 40 configs=4 methods ×5 LRs ×real/permuted; 468,021 scored bars | 65.7 s, 3.5 µs/config-bar; previous per-cell process 4 min/cell; best mirror/Adam/SGD 1.00019/1.00085/1.00263 relative zero; all absolute real-permuted gaps ≤0.003, half opposite sign. |
| LR 1e-3, normalized target | Mirror 1.032, Adam 1.153, SGD divergent. |
| `--lr-grid 0.0`; frozen zero weights | Permuted mean level 0.696, synthetic pure noise 0.008; independent-coordinate twin signs 0.690; causal channel/target centring real/permuted 0.828/0.344. |
| `--method hetvar --raw-target --vol-feature`; 468k SPY bars, raw centred return, one causal log-vol feature | Interior own-LR optima on 1e-6..3e-5 grid: Adam 0.99941, hetvar 0.99938 (~2 null-SD). |
| Same raw-target setup, LR 1e-3 | Real hetvar/Adam 1.0259/1.0363; permuted 1.0237/1.0399; noise absorption -35%; hetvar real-minus-permuted gap smaller than Adam's. |

## Same-target forecast composition and available-bar audit

Frozen v7 conditional stacking adds calibration, forecast combination, and
state interactions with a matched-context Adam control. On its 40,960 forward
observations `[294912,335872)`, context relative MSE was **0.999338**, static
control **0.999337**, matched-context Adam **0.999363**, and raw-input linear
Adam **0.999541**. Context did not improve the static control. Random-sign
context scored **1.000105** versus static **1.000011**. Jobs 5564/5565/5567/5568;
[plan](../../../benchmarks/plasticity/conditional_stock_v4_plan.json),
[evidence](../../../benchmarks/plasticity/conditional_stock_v4_evidence.json).

`predictive_available_state_v8.py` and `predictive_available_stock_eval_v5.py`
compare original bars `t..t+31` with the shifted window `t+1..t+32`.
Both predict the **unchanged** target at `t+33`; its normalization already
requires information through `t+32`. Original feature/target hashes match v7
bitwise, including all 468,021 original labels and the last real feature row.
Four independent Adam grids have real-prefix LR locks at 335,872; no Bayesian
prior or mixture selection on the forward window.

Real forward `[335872,409600)`, 73,728 observations; normalized-target zero
has relative MSE 1. Lower is better:

| Forecast | Relative MSE |
|---|---:|
| Causal raw-return zero transformed to target units | **0.995008** |
| Latest-window Bayesian mixture | 0.998565 |
| Latest-window linear Adam | 0.998602 |
| Original-window Bayesian mixture | 0.999258 |
| Original-window linear Adam | 0.999449 |
| Latest-window MLP Adam | 1.000673 |
| Original-window MLP Adam | 1.003919 |

With `d[k]=EWMA(return)[k]` and the frozen trailing scale `v[k]`, the label is
`clip((return[t+33]-d[t+32])/v[t+32], ±10)`. The causal no-learning diagnostic
is `clip(-d[t+32]/v[t+32], ±10)`. It beat the latest Bayesian model in **9/9**
chronological 8,192-observation blocks. This exposes a benchmark confound:
learning the known centering correction can beat normalized zero without
forecasting raw returns. Clipping the raw-zero transform is not, in general,
the exact conditional mean of clipped noise.

Fresh inputs increased reduction from normalized zero by 93.4% relative to
the old-window mixture, but reduced total noisy MSE by only 0.0694%. The
latest Bayesian/Adam gap is tiny; this is an input-alignment gain, not a
demonstrated general optimizer advance. Do not promote raw-return prediction
claims unless a learner beats the transformed raw-zero control.

The randomized-sign run stopped at 376,832 (40,960 forward observations).
At the common `[335872,376832)` horizon, real/null relative MSE was
0.997890/1.000002 for latest Bayes and 0.995046/1.004848 for the raw-zero
diagnostic. The diagnostic remains unchanged on randomized labels: no
current random sign is used. Both views preserve identical input hashes,
original target magnitudes, algorithm sources, and four real-prefix locks.

Verification: **146 CUDA/host contracts passed**, two independent read-only
reviews found no actionable defects, and full-consumed-prefix FP64 normal
equations checked both input frames and every dense prior. All checkpoint
covariances stayed positive definite without repair. Jobs **5575–5578**;
maximum parallel runs 1, one attempt, 12-minute contracts, 15-minute learning
runs, 10-minute reference. Both learning runs intentionally pruned (exit 75);
these are censored single-path/seed1 results, not full-file, significance,
profitability, global-optimum, or MuJoCo evidence.

[Plan](../../../benchmarks/plasticity/available_stock_v5_plan.json) ·
[Paired evidence](../../../benchmarks/plasticity/available_stock_v5_evidence.json) ·
[Numerical reference](../../../benchmarks/plasticity/available_stock_v5_batch_reference.json).
