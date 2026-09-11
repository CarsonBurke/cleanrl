# PPO runs

| File / run / configuration | Measurement |
|---|---|
| `ppo_32xlr_1mb_noadvnorm_stiglu_sphere_v1`; `HalfCheetah-v4__ppo_32xlr_1mb_noadvnorm_stiglu_sphere_50M__1__1788662844` | 13170 ±955 @50M; num_envs=16, num_steps=2048, B=32768, num_minibatches=1, update_epochs=10, learning_rate=9.6e-3, anneal_lr enabled, norm_adv off. |
| Same sphere base, full 50M annealing horizon | 4867 @2M, 6983 @4M, 8574 @8M, 9218 @12M, 10091 @16M, 11720 @32M, 13220 @50M. |
| `ppo_continuous_action`; seed 1, 8M, LR 3e-4 | Final-20 7468 ±124, @4M 6067. |
| `ppo_lrctl_5.1e-4`; same task, LR 1.7x | Final-20 8242 ±126, @4M 6842, final delta +774. |
| `ppo_lrctl_8.1e-4`; same task, LR 2.7x | Final-20 8369 ±292, @4M 7272, final delta +901; increment from 5.1e-4 +127 (~12% over old base). |
| `sdplast_v1`; old base | 7455 ±292 versus 7468 ±124; data_std 0.40. |
| `statedynlr` v1/v8 | v1 uniform 2.72x multiplier; v8 gain +1065, realized rate_std 0.013 versus target 0.31; recorded scores 8455→9763. |
| `sdplast_snr_v3`; 5072, LR 8.1e-4 | 8610 ±124 versus LR control 8369 ±292; lam_mean 1.354, ceiling 1.5, lam_std 0.013 (~1.35x uniform level). |
| `ppo_continuous_action_sdplast_relsnr_v4.py` | `lam_i=exp(span*(t_i-mean(t)))`, `t_i=tanh(gain*(l_i-mean(l))/span)`; geometric mean 1; 1000x global SNR rescaling unchanged; no old-base benchmark. |
| `ppo_continuous_action_sphere_sdplast_v5.py`; 5082 full /5083 `--no-snr-level`, 50M, `--max-parallel-runs 3 --priority 2` | 8 sites, 512 units at trunk.in_proj/block.down; per-sample pre-Adam GLS `w=exp(-p)` with unit batch mean 1 and bounds `[1/wmax²,wmax²]`; post-Adam relative-SNR rate. |
| v5 full / no-SNR-level; HalfCheetah-v4, seed 1, full 50M schedules | @8M 8943/9185; @12M 9699/10096; @16M 10451/10064; @20M 11018/10275 (delta +743); @34M 12482/10905 (delta +1577), base 11720 @32M (delta +762); windows ~101 episodes, CI95 ±48..279. |
| v5 `--lam-span 2.5`; 5087 | Cancelled; @12M 9282 versus v5 9699. |
| v5 `--lam-gain 2.0`; 5089 (`v7_gain2`, exponent 1, 8 sites) | @20M 10409 versus exponent-0.5 v5 11018. |
| v5 `--weight-max 4.0`; 5088 (`v6_wmax4`) | @20M 10370 versus v5 11018. |
| `ppo_continuous_action_sphere_sdplast_v7.py`; 5090 | Exponent 1, 20 sites/1028 units including gate/up/down; v5 exponent 0.5 lam_std 0.171, log bound ±0.405 (42%); probe dispersion at exponents 0.5/1/2: 0.266/0.337/0.379 (+26% from 0.5 to 1). |
| `ppo_continuous_action_sphere_sdplast_v8.py`; 5092 default /5093 `--weight-max 4.0` | `p=p_level+p_state`, `p_state=p_cap*tanh(readout/p_cap)`, `w=exp(p_ref-p_state)`, p_ref=EMA(mean(p_state)); no return measurement recorded. |
| `ppo_continuous_action_sphere_sdplast_v9.py`; 5094 default /5095 `--weight-suppress 32.0` | Default weight_suppress=8, weight_inflate=2, detached clamp `[1/8,2]`; no return measurement recorded. |
| v5/v7 endpoint KL off→on, same data | v5 8 sites: 0.00660→0.00657 (0.996x), clipfrac 0.0753→0.0755; v7 20 sites: 0.00650→0.00651 (1.001x), clipfrac 0.0743→0.0744. |
| v5 full/no-level, @12M and @20M | @12M KL 0.040/0.022; @20M KL 0.0268/0.0292 and clipfrac 0.198/0.198; base KL/clipfrac 0.0225/0.187. |
| `/tmp/kl_budget.py`; same data and seed | None: endpoint KL 0.00660; sustained random lam: KL 0.00660, step_sq ratio 1.066; resampled/update: 0.00659/1.062; measured-SNR: 0.00659/1.184 (~18% step_sq increase). |
| `/tmp/gradnorm_probe.py`; max_grad_norm 0.5 | Median pre-clip norm 0.595; 55% steps clip; median overshoot 1.19x. |
| Sphere PPO telemetry | Per-unit SNR 1.2e-4..1.3e-3; sigma²/mu² ~800–8000; batch-mean relative error ~17% at 32768; w_std 0.44/0.45; lam_std 0.171; EV base 0.985 versus arms 0.951–0.958. |
| v5 supervision, actual B=minibatch=32768 | Base/supervised update 4.00/8.24 ms (2.06x); 15,250 updates/50M, +65 s; peak 0.64 GiB, probes 64 MB; gate_every=1, 10 updates/rollout; old base 32 minibatches, gate_every=4, 1.30x update cost, +1.6% wall time. |
| v5/v7 throughput | Base→v5 SPS 37663→34109 (8–9%); v7 update cost 4.07x, +199 s/50M (~15%), peak 1.35 GiB. |
| `/tmp/verify_v5.py` | 30 checks pass; unit-uniform row-rescale policy change 9.3e-10, dispersed 4.7e-2 relative (2.5e5x); activation walk difference 0.0; host actor difference 1.9e-9; earlier v1/v2 checks 57/41 pass; neutral step and initial model bit-identical. |
| `ppo_continuous_action_pcbatch_v1.py`; standard dense trunk, per-(unit,sample) gate in minibatch sum | @114k c_mean 0.514, unit dispersion 0.098, sample dispersion 0.382 (~2x mean LR cut). |
| pcbatch jobs 5141 /5142 `--pc-shuffle` /5143 `--pc-scalar` /5144 `--pc-off`; max-parallel 3, no compile, 8M | Reference baseline 8278, incumbent 10362 (+21%); seed 1 off/pc/scalar/shuffle 6122/6049/5717/5671; off equals `ppo_continuous_action_hostactor` 6121.7 deterministic; earlier build 7468; reference 8278; spread ~1000+. |
| pcbatch seeds 2–8, jobs 5149–5176 | 44k SPS, 3 min/run; final 8-seed table below; final pc−off -914 (-13%), scalar−off -113, shuffle−off -688 (~-690), pc−shuffle -226. |

| pcbatch arm, HalfCheetah-v4, 8 seeds | @2M | @4M | @8M mean ±CI95 (SD) |
|---|---:|---:|---:|
| off | 4497 | 6137 | 7128 ±488 (704) |
| scalar | 4387 | 6034 | 7015 ±485 (700) |
| shuffle | 3838 | 5451 | 6440 ±377 (544) |
| pc | 3729 | 5099 | 6214 ±250 (360) |

| Rollout probe / configuration | Measurement / correction |
|---|---|
| RLBridge / `ppo_signal_legibility.py`; 2.5M PPO steps, 256 actor+critic units, per-(unit,sample) dL/dz from real clipped loss via retain_grad; control permutes each unit's signal column within update | Historical gain-invariant SNR F=7.20–8.08 versus permutation 0.89–0.95 (8.1–8.6x); 96% of 256 units above permutation p95. |
| Same probe; 500k windows, within-unit-centred cell/unit pattern | Cross-window r=0.20→0.27→0.32→0.35; permutation ~0. |
| Same probe; historical scalar reliability | eta²=1.0e-05; within-unit cell reliability range 3.9e-05 on [0,1]; log-SD noise 0.946, tanh slope `(1-a²)` log-SD 0.844; prior ~89% share calculation withdrawn; coarse-bin standardization removes bin-constant gain only. |
| `ppo_signal_legibility.py`; corrected aggregation/ANOVA | Nonnegative squared-SNR selects stronger positive or negative half; previous positive-only reference changed under global sign flip; standardized ANOVA uses within/total sums of squares and residual degrees of freedom; old eta_snr denominator was sample count and could exceed 1; unsupported 1/n subtraction removed, historical reference magnitudes changed. |

## Predictive transport v23: negative HalfCheetah pilot

Covariance-free FP32 output-score transport, applied to the real clipped PPO loss.
Frozen v22 Pre-RMS SiTU-GLU/Beta architecture; same LR 0.0096, raw GAE, clipped
scalar critic, 32768-sample batch, 10 epochs, no gradient clipping in every arm.
Seed 1, CUDA compiled, requested 8M steps; actual final metric step 7,978,624
(16,000 phase-warmup transitions plus 243 full rollouts). Scores below are
training episodic returns, not separate deterministic evaluation.

| Mode | beta1 | Job | @1M | @2M | @4M | Final 100 episodes | Delta vs matched Adam |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam | 0.90 | 5832 | 2503 | 4174 | 6373 | 8299 | — |
| Critic transport | 0.90 | 5833 | 2061 | 3623 | 5655 | 7739 | -6.75% |
| Actor + critic transport | 0.90 | 5834 | 998 | 2415 | 4224 | 5457 | -34.24% |
| Adam | 0.99 | 5835 | 2793 | 5246 | 7266 | 8500 | — |
| Critic transport | 0.99 | 5836 | 1458 | 2196 | 3176 | 5129 | -39.66% |
| Actor + critic transport | 0.99 | 5837 | 72 | 554 | 1432 | 3508 | -58.73% |

Matched-step windows are +/-50k transitions. All four transport arms lose at
every reported step and at the endpoint. Single seed only; within-run episode
confidence intervals would not establish seed-level significance.

- Verification job 5831: **21 contracts passed**, zero failures/skips, 37.51s.
  Includes frozen-v22 gradient equality, clipping/entropy output scores,
  zero-correction Adam identity, moment mass, owned previous snapshots, and
  compiled functional gradients on the actual residual architecture.
- All seven jobs succeeded, one attempt each. Declared max-parallel-runs 1;
  contracts time limit 20m, training 45m, priority 0. No culls or retries.
- All six runs share initialization hashes, source hashes, runtime versions,
  and non-treatment arguments. Recorded sources still match working files.
  No nonfinite logged scalars; each run saved a checkpoint.
- At 4M/beta1=0.99, actor+critic transport KL is 0.00113 versus Adam 0.01009.
  Critic explained variance is higher (0.971 versus 0.875), despite return
  1432 versus 7266: explained variance alone is not policy quality.
- Median post-first update phase (10 optimizer steps): Adam 21.35–21.44ms,
  critic transport 30.15–30.36ms, both 33.00–33.08ms. Whole serial attempts
  87.6–109.9s include startup/compilation; differing compiler cache and env
  timing prohibit an aggregate-throughput claim.

**Decision:** reject this transfer as an improvement; retain ordinary Adam.
The supervised regression gain did not transfer under these matched PPO settings.

[INFERENCE] Lower policy KL is consistent with adversely changed update
dynamics, but alignment/transport error was not measured, so causality is not
established. The correction/fresh-gradient coefficient ratio approaches
beta1/(1-beta1), i.e. 9 or 99; unweighted correction/gradient telemetry near 0.01
does not imply a negligible correction. PPO also refreshes data, advantages,
and bootstrap targets each rollout, whereas this correction covers only model
output-score drift on fixed current data and omits output-Jacobian drift.
Before another redesign, measure those approximation and objective-boundary
errors on captured PPO batches rather than blindly damp or extend a losing run.

Evidence: `benchmarks/plasticity/predictive_ppo_v23_evidence.json` retains exact
scores, window counts, diagnostics, timing, manifests, checkpoint paths and
queue outcomes; `predictive_ppo_v23_contracts.xml` retains numerical results.
