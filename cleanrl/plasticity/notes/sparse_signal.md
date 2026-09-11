# Sparse-signal streams

| File / configuration | Measurement / mechanism |
|---|---|
| `noisy_stream_diagnostic.py`; 4096 Bernoulli(0.01) inputs, 1 signal/4095 distractors, N(0,5) target noise plus ±1 spikes, jobs 5098 linear/5099 hidden | B=1, single pass, no replay; clean target weight 1 on input 0, 0 elsewhere; selectivity=signal weight/distractor RMS. |
| `cleanrl/shared/state_plasticity.py`; snrgate | Matched-beta first/second gradient moments updated on active observations only, post-optimizer SNR/mean(SNR); 60k selectivity 7.6 versus Adam 6.2; active signal/distractor level 1.980/0.795=2.49x; previous reference included inactive zero-SNR entries and saturated all levels at ceiling 2.0 (floor 0.125). |
| `snradam`; Adam's own m/v | 1%-sparse feature moment ratio magnitude ~1 just after firing. |

| 20k steps, 16 seeds, linear stream | Selectivity ± reported uncertainty | Raw-statistic separation |
|---|---:|---:|
| Adam | 6.01 ±0.69 | — |
| Energy, v5–v9 residual-energy objective | 5.43 ±0.99 | 0.94x |
| SNR + debias | 15.60 ±3.08 | 21.3x (before debias 14.0x) |
| Statewiener + debias | 15.13 ±3.19 | 14.6x |
| Adam LR×0.158, matching uniform factor | 5.92 ±0.56 | — |

| 20k steps, 16 seeds, regime stream | Selectivity ± reported uncertainty | Quiet/noisy signal-coordinate level |
|---|---:|---:|
| Adam | 4.08 ±0.97 | — |
| Energy | 4.60 ±1.35 | 1.02x |
| SNR running moments | 11.81 ±5.41 | 1.03x |
| Statewiener state-conditioned | 13.60 ±5.83 | 1.73x; all-4096-coordinate aggregate previously 0.79x |

| Initial 20k diagnostic arm | Signal weight | Distractor RMS | Selectivity |
|---|---:|---:|---:|
| SGD | 0.213 | 0.0288 | 7.4 |
| Adam | 0.586 | 0.0946 | 6.2 |
| Rowgate v9 | 0.590 | 0.0951 | 6.2 |
| Colgate v9 | 0.580 | 0.0941 | 6.2 |
| Snradam | 0.428 | 0.0700 | 6.1 |
| Oracle support | 0.595 | 0.000 | infinity |

| Chart command / configuration | Measurement |
|---|---|
| `.venv/bin/python cleanrl/plasticity/noisy_stream_diagnostic.py --method all --steps 20000 --seeds 4 --eval-steps 4096 --adaptive-z --plot-window 500 --plot cleanrl/plasticity/reference/noisy_stream_predictions.png` | Linear 4096-input stream, B=1; N(0,5) noise; zero-predictor clean MSE 0.0112; table below retains recorded ratios. |

| 20k chart arm | Clean MSE | Recorded multiple of zero | Signal weight | Distractor RMS |
|---|---:|---:|---:|---:|
| Oracle | 0.00219 | 0.20x | 0.562 | 0.000 |
| Veto adaptive z | 0.00604 | 0.54x | 0.426 | 0.0072 |
| Graded fixed z=5 | 0.00901 | 0.81x | 0.121 | — |
| Graded adaptive z | 0.01088 | 0.97x | 0.245 | 0.0102 |
| SGD | 0.04176 | 3.74x | 0.173 | 0.0287 |
| SNR | 0.08084 | 8.89x | 0.732 | 0.0443 |
| Statewiener | 0.08628 | 9.49x | 0.729 | 0.0457 |
| Energy | 0.26911 | 29.6x | 0.460 | 0.0802 |
| AdamW | 0.31268 | 28.0x | 0.517 | 0.0872 |
| Adam | 0.36029 | 32.3x | 0.562 | 0.0937 |
| Mirror, level=1−FDP from sign-randomized twin | 0.00397 | 0.44x | 0.418 | —; signal level 1.0000 |

| Same stream / ablation | Measurement / correction |
|---|---|
| Graded `(t²/(t²+z²))^p` versus fixed-z binary veto | Matched 20k MSE 0.0090 versus 0.0076, difference 19%; adaptive graded exponent 2 gives 0.01088; prior binary-only attribution withdrawn. |
| Remove SNR envelope floor | Matched 20k 0.0808→0.0729; 100k 0.2664→0.3050; previous 3.6x reduction compared different horizons (20k/100k). |
| `--adaptive-z`; threshold from coordinate upper quantile | Graded signal retention 0.121→0.245; veto 0.176→0.426, MSE 0.0076→0.0060. |
| GLS-weighted cumulative evidence, weight 1/sigma_hat² | Homoscedastic mirror MSE 0.00397→0.00406; heteroscedastic regime 0.00878→0.00890. |
| Input pooling `granularity="input"`; per-connection SNR 0.05, 400 steps (t~1), width 64 | Per-connection gate never opens; twin-calibrated pooled sum of t² reaches level 1.000 by step 100, useful/useless separation 10x. |

| Signal moves at midpoint; veto adaptive z | Clean MSE | New weight | Stale absolute weight |
|---|---:|---:|---:|
| Evidence decay 0 | 0.01007 | 0.110 | 0.146 |
| Decay 1e-3 | 0.01236 | 0.052 | 0.077 |
| Decay 1e-2 | 0.01712 | 0.020 | 0.029 |
| SGD | 0.04183 | 0.120 | 0.085 |

| Mirror/SGD switch configuration | MSE | Current weight | Stale absolute weight | Stale-weight error share |
|---|---:|---:|---:|---:|
| Mirror signal moves | 0.01205 | 0.0209 | 0.1011 | +1.4% |
| SGD signal moves | 0.04183 | 0.1201 | 0.0849 | +0.2% |
| Mirror signal leaves then returns | 0.00755 | 0.1449 | 0.0035 | — |
| SGD leaves then returns | 0.04287 | 0.1179 | 0.0839 | — |
| Returning versus fresh mirror coordinate | — | 7x weight with 34% less post-change time | — | — |

| Fan-in D, 20k B=1 | Mirror MSE | SGD MSE | Mirror signal/distractor RMS level |
|---|---:|---:|---:|
| 32 | 0.00324 | 0.00672 | 1.0000/0.2720 |
| 64 | 0.00241 | 0.00664 | 1.0000/0.2606 |
| 256 | 0.00325 | 0.00867 | 1.0000/0.1094 |
| 1024 | 0.00319 | 0.01382 | 1.0000/0.0545 |
| 4096 | 0.00397 | 0.04176 | 1.0000/0.0398 |

| `reference/spike_recovery.png`; 4096 features, 1 predictive, ±1 label noise | Clean MSE | Recorded multiple of zero | Signal weight | Null weight |
|---|---:|---:|---:|---:|
| SGD | 0.09194 | 8.44x | 0.634 | 0.0469 |
| Adam | 0.44816 | 41.14x | 1.027 | 0.1043 |
| Mirror | 0.00227 | 0.23x | 1.046 | 0.0070 |
| Softveto floor 0.125 | 0.00190 | 0.17x | 0.852 | 0.0059 |
| Smoothgate `(t²/(t²+z²))^4` | 0.00117 | 0.11x | 0.683 | 0.0004 |
| Veto hard z=5 | 0.00037 | 0.03x | 1.025 | 0 |
| Gradedveto hinge | 0.00039 | 0.04x | 0.893 | 0 |
| Softhinge | 0.00039 | 0.04x | 0.893 | 1.6e-12 |
| Oracle support | 0.00043 | 0.04x | 1.027 | 0 |

| Mechanism / later run | Measurement |
|---|---|
| Fan-in 64 comparison | SGD/mirror error ratio 2.8x; pooled tensor supplies H×D null draws. |
| Softhinge `certainty=softplus(k*(1-z²/t²))/k`, k=24, twin-calibrated z | Null level softplus(-k)/k=1.6e-12; large-evidence level→1; reference scale 1/sqrt(4096)=0.016, softveto floor 0.125=8x this scale, smooth ratio level 0.04=2.5x. |
| Sparse teacher switch, per-parameter amplification above level 1 | Mechanism test/zero 0.31 versus oracle-mask 0.55. |
| Corrected sparse recovery, job 5336; fixed LR 0.001, seed 1, 60,000 observations, ±1 noise | Adam MSE 0.234731 (23.473x zero); mirror 0.001327 (0.133x); softhinge 0.000332 (0.033x); softhinge amplification 0.046290 (4.629x); support-informed Adam 0.000200 (0.020x); mirror division correction in `measurement.md`. |
| `hidden_stream.py` / `novelty_stream.py` | Hidden-layer LR/density/amplification, novelty and precision rows in `per_unit.md`; null-stream stock absorption and oracle rows in `per_sample.md`. |

## Structural inference and change memory (v5–v7)

The target is the causal conditional mean, not merely smaller optimizer steps.
Brain-inspired context memory is a hypothesis: these models must retain useful
histories without hallucinating signal from noise. All views consume 60,000
seed-1 observations, with 4,096 Bernoulli(.01) features and Gaussian noise of
standard deviation 1. Clean targets score forecasts but never select AdamW.
The first 15,000 noisy stationary observations select one of 72 AdamW settings:
LR 3e-5, beta1=0, beta2=.999, decay=.01. That lock is transferred unchanged.

- **v5:** global change-point branches around the existing factorized learner.
  Nine top-mass branches fail recurrence; 32 improve it but still lose to v3.
- **v6:** average four hazard hypotheses, h=0/1e-5/1e-4/1e-3, with 32 branches
  each. Fresh namespace-200 recurrence is 29.15% worse than v3; null clean MSE
  1.59e-5 fails the 1e-5 guard. No promotion.
- **v7:** exact null/singleton Normal–Inverse-Gamma inference within each segment,
  Student-t predictive evidence, and posterior-mass-preserving log-age
  resampling. History retention remains approximate. A predictive-density hedge
  with the v6 factorized family protects against two-support misspecification.
  Singleton-only inference fails that view; it is not a general sparse optimum.

| v7 view | Development clean MSE | Fresh namespace-300 clean MSE | Fresh v3 | Fresh change versus v3 |
|---|---:|---:|---:|---:|
| Stationary | .000267088 | .000748761 | .000823075 | -9.03% |
| Change | .001102836 | .001303367 | .001369676 | -4.84% |
| Null | .000007978 | .000007240 | .000000986 | 7.34x |
| Recurrent | .001421310 | .002421099 | .001930316 | +25.42% |
| Two support | .001103052 | .001359417 | .001380888 | -1.55% |

**Reject v7 promotion.** The development change gain of 20.22% contracts to
4.84% on the fresh stream; recurrence reverses from an 11.32% improvement to a
25.42% regression. The predeclared requirement was at least 20% below v3 on
both change and recurrence, at most 5% worse on stationary/two-support, and
null MSE at most 1e-5. Passing the absolute null guard does not mean better
noise suppression than v3.

All five fresh views completed, including unfavorable ones. Each verifies
60,000 data/update/checkpoint clocks, unchanged source/data/lock hashes and
7,680,000 consumed pre-generated resampling uniforms. Joint-bank replay plus
reporting/checkpoint work took 158–177 seconds per fresh view; this is not
isolated learner latency or an equal-compute comparison. Mutable model state
is 17,703,832 bytes, excluding shared prior, tape and traces.

Jobs: v5 development 6129/6133–6136; v6 development 6149–6153 and fresh
6157–6161; v7 development 6184–6188 and fresh 6199–6203. Queue limit 1,
priority 0, one attempt. Evidence, plans, complete paths and numerical checks:
[predictive optimum evidence](../../../benchmarks/plasticity/predictive_optimum_v1_evidence.json).
These are specialized inference results, not PPO, generic optimizer gains,
independent-seed significance, or evidence that an optimum has been attained.
