# REPPO root-cause study, part 1: evidence ledger

Role: evidence auditor / historian. Read-only audit of `cleanrl/reppo/FAMILY.md`, `cleanrl/reppo/OPTIMUM.md`,
`benchmarks/reppo_diagnose_v29/{analysis,results}.md`, the 63 version headers, the oracle JSONs under
`runs/reppo_*oracle/`, and the tfevents of every `*reppo*` HalfCheetah run. No training was launched.

Provenance tags used below:
- [RAW] = I re-derived the number from tfevents / JSON in this session.
- [DOC] = only read in a write-up or file header; not re-verified.
- [CODE] = read in the source.
- [REASONING] = my inference; not measured.

Artifacts (all under `benchmarks/reppo_rootcause/`):
- `reppo_runs.tsv` - matched-step returns (window +-50k, same as `score_runs.py --at`) at 0.5/1/2/3/4/6/8/12/16M,
  last-20 mean, and `advantage_corr_heldout`, `advantage_std_ratio`, `q_mse` at 1/2/4/8M, for all 200 REPPO runs.
- `baseline_runs.tsv` - same returns for sepclip / ppo_baseline / ppo_continuous_action / ppo_situ / awr / td7 runs.
- `all_hc_runs.tsv` - same for every HalfCheetah run in `runs/` (used for section 6).
- `scripts/dump_runs.py`, `scripts/metric_vs_return.py`, `scripts/diff_hparams.py`.

Grades: A controlled, clearly beyond noise, no known confound. B suggestive (single seed but large, or consistent
across several runs). C confounded / underpowered / metric not validated. D contradicted by other repo evidence.

---

## 0. The ten findings that change how the record should be read

1. **Runs are bit-deterministic, so "repeats" are not replicates.** `reppo_v7_q_cu128`, `_r1`, `_r2` are the same
   trajectory to the last digit (2414 / 4583 / 6030 / 8014 / 10152 at 0.5/1/2/4/8M) [RAW]. Same for `distq_v12_r1/r2`,
   and for v27 H8 = v29 reward = v36 FP32 = v49b_dual = v31 (until the cap binds), and v55-dual = v60 base =
   landscape_ref. The whole REPPO record contains only **two** independent upright from-scratch trajectories of the
   reference algorithm (call them path A and path B) plus one fork (E). There is no seed variance estimate anywhere.

2. **The reference algorithm itself flips about half the time.** Trajectories that are the reference algorithm
   with only RNG consumption / op-order changed: A upright (11.0k), B upright (11.1k), v61 control flipped (2.2k),
   v43-FP32 flipped (2.5k). 2 of 4 [RAW]. A fifth same-hyperparameter pair, `distq_hidden_v13_r1` (2.2k, flipped)
   vs `_r2` (9.2k), has **identical logged hyperparameters** [RAW, `diff_hparams.py`] (code revision between them
   unknown). So the "best ~11.0k @8M" is a conditional-on-upright number; the expected seed-averaged 8M score of
   the reference is plausibly ~6-7.5k, i.e. **below** the PPO sepclip control (8.4k). Nobody has stated this.

3. **`advantage_corr_heldout` does not predict performance and its ceiling is about 0.3, not 1.** Across 50-75
   distinct REPPO trajectories, Spearman(metric, return) is -0.28 .. +0.06 at 1/2/4/8M [RAW]. Flipped 2.3k runs
   score 0.09-0.14, the 11k reference 0.05-0.14, a near-uniform non-learning policy 0.26-0.29. The reference
   signal is a lambda-0.95 return advantage in which the row's own action accounts for only ~1/8.7 of the
   variance, so a *perfect* critic would score ~0.34 (section 3). "Held-out corr 0.01-0.14 = near noise" is a
   misreading: 0.05-0.14 is 15-40% of ceiling.

4. **The critic's action contrasts are over-dispersed for their accuracy, not just "3x compressed".** Slope of
   held-out return-advantage on critic-advantage = corr / std_ratio: median 0.6 at 1M falling to 0.5 at 8M across
   runs; reference 0.7 / 0.9 / 0.45 / 0.4 at 1/2/4/8M [RAW]. This matches v55's beta (0.50 / 0.71 / 0.40 / 0.28)
   and the oracle's pre-tanh slope (0.21-0.37) from three different estimators. A calibrated estimator with
   true correlation rho would have sd = rho x true sd; the critic's sd is ~2x that. Roughly half or more of the
   action dependence the actor climbs is fitted noise, and the fraction that is real declines through training.

5. **v61/v62 "exact targets" did not remove target variance, so claim (3) is untested.** The correction
   subtracts the critic's *own* Q'(s_{t+1}, a_{t+1}) [CODE]; it can remove at most rho^2 (a few %) of the
   future-action noise while the critic's advantages are rho ~0.2 accurate. Direct evidence: the training
   residual `q_mse` did not fall - exact lam 0.8 1.42 vs control 1.34 at 5M; exact lam 0.95 **4.18** vs 1.34,
   and 6.28 vs 1.64 at 7.7M [RAW]. A target that had lost its noise would be easier to fit, not 3x harder.

6. **The scalar-MSE-vs-HL-Gauss test that FAMILY.md says is still needed was already run (v14).** v7-cu128 with a
   plain MSE scalar head: 9571 @8M (last-20 9377) vs HL-Gauss 10152 (last-20 10505); 6333 vs 8014 @4M [RAW]. No
   gain from removing the 2.0-wide bins. Together with the fact that the HL-Gauss CE gradient is smooth in the
   target mean (no quantization; a 0.04 shift moves the projected histogram's mean by 0.04), the "action gap =
   1/40 of label width" is a description of signal-to-noise, not a demonstrated cause. Also the "1/40" uses the
   critic/model contrast sd (~0.04); the oracle's *true* within-state sd is 0.22-0.33, i.e. 1/6-1/9 of a bin.

7. **A nominally identical pair differs by 3.5k while both stay upright.** `vtrace_v11_buf1` is declared
   "identical to v7's target on fresh data"; hyperparameters match v7-cu128, logged fresh pi/mu ratio is
   0.988-0.997 [RAW]. It scores 6633 @8M vs 10152. Either there is an undiagnosed implementation difference or
   upright run-to-run spread is several k. Likewise v27 with `aux_horizon` 4 vs 8 as the only difference:
   5020 vs 11031 @8M; and with reward loss off: H4 9104, H8 10274. A non-monotone 2x2 with a 6k range.

8. **Replay has one positive result that the summaries omit.** Within v11 (V-trace, cu128): buf1 6633, buf2 6359,
   buf4 9226, buf8 9013 (last-20 9649) @8M [RAW]. The replay arms were *behind* until ~5M (buf8 1325 @1M vs 3871)
   and overtook late. v52 (the "refuted" V-trace window) was cancelled at 1.2M, exactly the regime where v11's
   winning arms looked worst.

9. **The oracle's physical-action gradient agreement is ~zero and was not reported.** `runs/reppo_critic_oracle/
   v60_base.json`: in action space slope_true_on_critic -0.01 +- 0.12 (h 0.1), 0.02 +- 0.06 (h 0.3), cos median
   0.00-0.03, sign agreement 0.50-0.52; only the pre-tanh version gives the quoted 0.37 +- 0.12 / 0.21 +- 0.05
   (sign agreement still 0.52-0.53) [RAW]. And the quoted "Q vs own one-step backup within-state corr -0.33
   (self-inconsistent)" is the pooled figure; the per-state median correlation in the same block is **+0.55**,
   gradient cos median 0.58, sign agreement 0.68 [RAW]. The pooled number is driven by a minority of states.

10. **REPPO conditional-on-upright is level with or slightly ahead of the best on-policy 8M results in the repo
    (11.0k vs 10.3-10.4k, inside the noise floor); its problems are the flip rate and the 50M asymptote.** See
    section 6.

---

## 1. Noise floor (verified from tfevents)

### 1.1 Independent trajectories of the reference algorithm (v27 H8 / v36 FP32 lineage, strict FP32)

| Path | Runs sharing it | 0.5M | 1M | 2M | 3M | 4M | 6M | 8M | last-20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | v27_h8_50M, v29_h8_reward, v36_fp32, v49b_dual, v31 (pre-cap) | 2521 | 4174 | 5634 | 7059 | 8596 | 10176 | 11031 | 11085 (8M run) |
| B | v55_dual, v60_base, landscape_ref (probes consume RNG) | 2575 | 4267 | 5839 | 7848 | 9084 | 10323 | 11069 | 11014 |
| E | v62 control: fork of B at 1.05M, uncorrected targets | - | - | 5658 | 7896 | 8762 | 9673 | 10325 | 10174 |
| C | v61 control (= B's algorithm, different RNG consumption) | 1380 | 1884 | 2125 | 2223 | - | - | - | flipped |
| D | v43_fp32_s50 (= A's algorithm until 4M, different op order) | 1353 | 1947 | 2157 | 2223 | 2254 | 2434 | 2537 | flipped |
| F | v43 TF32 s50 (TF32 precision, otherwise A until 4M) | 2111 | 3847 | 4799 | 5523 | 6045 | - | - | - |

All [RAW]. Hyperparameter diffs confirmed with `diff_hparams.py` (v36_fp32 vs v55_dual differ only in probe
arguments; v60_base vs v61_control only in checkpoint/fork arguments and `exact_targets False`).

- Upright spread (A, B, E; sample sd, n = 3): 2M 112, 3M 470, 4M 248, 6M 340, 8M 419. With n = 3 the sd is known
  only to a factor of ~2 either way.
- Measurement noise on one "@8M" cell: an 8M-step run ends inside the +-50k window, so the cell averages ~half the
  episodes. The same trajectory reads 10894 (v36_fp32, truncated window) and 11031 (v27_h8_50M, full window); v7
  cu128 reads 10152 vs 10484 the same way [RAW]. Last-20 CI95 is +-300..+-990. FAMILY.md's "10894-11031" range for
  the reference is one trajectory measured twice, not two results.
- Late forks from the common 8M snapshot (v40 w0, v41 twin_first, unforked): @12M 11407 / 11005 / 11274, sd ~200;
  per-checkpoint sd 130-250 over 9-12M [RAW].

### 1.2 Minimum detectable effect, single run vs single run

Two-sigma on a difference of two independent runs = 2.83 x sd:

| Comparison | 1-2M | 4M | 8M |
|---|---:|---:|---:|
| from scratch, **both upright** (optimistic, sd from A/B/E) | ~0.3-0.4k | ~0.7k (1.3k at 3M) | ~1.2k |
| from scratch, unconditional | none: a flip (p ~0.4-0.5) costs 6-9k | | |
| fork from the 8M snapshot, per checkpoint | - | - | ~0.6k at 12M |
| fork from the 1M checkpoint (v62 style) | ~0.5k | ~0.9k | ~1.2k+ (control vs base alone: 744-840) |

The optimistic row is a lower bound. Section 0 items 7 and the v27 H4/H8 2x2 show nominally-near-identical upright
configurations differing by 3.5-6k, and there is a second, non-flip stall band at ~3.5-5k (v27_full H4 3.7-5.0k for
16M steps, v26 3.5-4.1k, v7 coupled 4.0k, v9 ns2048 4.1k, v10 buf1 4.2k). Honest statement: **a single-seed
from-scratch 8M difference under ~1.2k is uninterpretable, and under ~3k is not safe.**

### 1.3 Flip census

Runs sitting at 1.9-2.6k and flat after >= 3M steps, by distinct trajectory [RAW]: v4 lr1.2e-3, v5 (x3 arms),
v7_cw1024, v7_lam0p8, v11_retrace_buf1, v12 (r1 = r2), v13_r1, v25_lam0p5, v28, v32, v35 (x2), v37, v37_fp32,
v39, v43_fp32 (x2 arms, one trajectory), v55_ids, v61_control, v61_l08. That is ~21 of ~135 from-scratch
trajectories overall (~16%), ~10 of the ~50 from-scratch trajectories since v32, and 2 of 4 reference-algorithm
trajectories. For contrast, **0 of 111** runs of the PPO baseline families (sepclip / ppo_* / awr, >= 3.5M steps)
sit in 1.8-2.7k at 4M [RAW, `baseline_runs.tsv`], and 43 of 1404 (3%) of all non-REPPO, non-TD7 HalfCheetah runs
do (those include many weak experimental methods and were not checked for flatness or posture). The back-running
trap is far more frequent under the REPPO actor than under likelihood-ratio actors on the same seed.

Posture (root pitch) was actually measured only for v35, v27, v37, v38, v39 and v43's upright arms [DOC]; every
other "flip" is inferred from the 2.0-2.5k return level.

"Discrete, irreversible, decided within 0.5M" is not established: `refresh_v25_lam0p8` reads 1395 / 1842 at
0.5M / 1M - indistinguishable from the flipped runs (1353-1380 / 1884-1947) - then 3467 @2M and 9282 @8M [RAW].
Either it was slow, or it flipped and recovered. No posture snapshot exists for it. `replay_v5_ns64` goes the other
way: 3703 @4M then 2254 at 8M.

---

## 2. Spot-checks of quoted numbers against tfevents

Checked ~90 quoted return cells with `scripts/dump_runs.py` (cross-validated against `scripts/score_runs.py --at`
on 9 runs: identical). The record is numerically honest; the mismatches are:

| Quoted | Where | Actual [RAW] | Comment |
|---|---|---|---|
| v38 branchadv 2568 @4M | FAMILY v36-v38 table | 3155 (2076 @3M) | run ended 4.62M at last-20 3523, still rising |
| v43 "0/dim fixed" 6460 @8M | higher-entropy screen | 8500 window, 8417 last-20 | doc used single-episode point values; off by 2k |
| v43 -0.25/dim 9440 @8M | same table | 10067 window, 9900 last-20 | same cause |
| v43 m0_late1p5 12000 @8M | same table | 11571 window, 11307 last-20 | point value; doc does give last-20 |
| v36 e128 6082 @4M | v39-v40 table | 5933 | minor |
| v56 "cancelled at 7.7M", last-20 9531 | v56 table | run reached 7.995M, last-20 9697 | minor |
| v36 FP32 "10894 @8M" vs v27 "11031" | several | same trajectory | truncated-window artifact (section 1.1) |
| "true within-state advantage ~0.03" | v62 | oracle 0.22 (1M), 0.33 (8M probe) | 0.03-0.04 is the critic/model contrast, not truth |

Verified exactly (sample): v27 H8 4174/5634/8596/11031/11274; v30 w0.5 8605/9588/10919; v31 12432 and 12308 end;
v32 2349; v34 4839/5562; v35 2226/2009; v37 1697/2274/2363; v40 forks 11408/10972/11127/8071; v41 11005/10924/
11402; v42 11130/10410 and 12291/12178/12399; v44 6645/7400; v45 8868/11002; v46 K8 10295; v47 8259/9084/7046;
ns128 9670; v48 floor 9545; v51 2333/4355/6971/8343; v52 3106/2272; v53 3425/3512/6661; v54 9159/8720; v58
3817/4838/6023; v59 8520; v61 l095 9715; v62 whole table and finals 10174/10458/10570/11014; v63 4876/5783/4439/
5836/5258; header claims v7 8.5k, cu64 9.6k, cu128 10.5k, cu256 9.7k, noaux 6.7k, coupled 4.0k, v8 5.5k vs 6.2k,
v10 buf1 4.5k, v6 buf8 4.7k, v19 7.0k, v12 2.2k; TD7 17.1k @1M (td7_sdnoise_v1 17124); best on-policy 6.4k @1M
(v47 crossfold 6375).

Oracle JSON cross-check (`runs/reppo_critic_oracle/v60_base.json`): level corr 0.930, error sd 0.903, between sd
2.32; within true sd 0.223, critic sd 0.206, corr 0.109 (0.149 corrected), pick gain 0.0596 +- 0.012 = 0.27 sd:
all as quoted. Not quoted: truth reliability 0.54; 38% of branches hit a time limit and bootstrap from the critic
with mean weight 0.27; critic level bias -0.6 (soft) / -2.2 (hard); the action-space gradient block (section 0.9).

---

## 3. Is `advantage_corr_heldout` a valid measure? (definition, ceiling, bias)

Definition [CODE: `critic_diagnostics`, e.g. `ppo_continuous_action_reppo_noisefork_v60.py` l.514-533]. On 4096
random rows of the fresh rollout, before the iteration's update:

    baseline(s)       = mean over 8 fresh policy samples of Q_target(s, a')
    critic_advantage  = Q_target(s, a_taken) - baseline(s)
    return_advantage  = G_lambda=0.95(s, a_taken) - baseline(s)       (soft, bootstrapped from the same critic)
    metric            = Pearson correlation pooled over all rows;  std_ratio = sd(critic_adv) / sd(return_adv)

Ceiling [REASONING, consistent with repo measurements]. Write the lambda-return advantage as the discounted sum of
the taken actions' true advantages: G - V(s_0) = sum_k (gamma lambda)^k A(s_k, a_k) + critic-bootstrap error. With
gamma lambda = 0.9405 the sum of squared weights is 1 / (1 - 0.9405^2) = 8.66, so the row's own action carries
1 / 8.66 = 11.5% of the variance and future actions 88.5%. The max correlation of *any* function of (s_0, a_0) -
including the true Q - is sqrt(0.115) = **0.34**, lower once bootstrap error is added. Three independent repo
numbers agree: v61's header "oracle: ~90% of a return advantage's variance" is future actions; v26's header
"oracle true spread is 0.3-0.6 of the lambda-return advantage spread"; and the one policy with huge action effects
(v48 box, near-uniform Beta) measures corr 0.26-0.29 with std_ratio 0.30-0.32, i.e. it sits on the ceiling [RAW].

Bias. The same 8-sample baseline is subtracted from both sides, so its Monte-Carlo error adds
+var(Q over actions)/8 to the covariance: a spurious +std_ratio/8, i.e. about **+0.015 to +0.03** [REASONING].
Readings of 0.01-0.03 (v47 late, v48 floor, v57) are therefore indistinguishable from zero or below.

Precision. Per-iteration standard error ~1/sqrt(4096) = 0.016; windowed means are precise. The problem is validity,
not noise.

What it measures. Converting with the ceiling: reference 0.05-0.14 -> true within-state correlation of roughly
0.15-0.4. Oracle estimates for comparison: 0.15 (v60 1M, noise-corrected), 0.20 (v29 8M FD), 0.15 (v29 12M),
~0.29 (0.2 against a ceiling 0.7, v29 8M/16M probe), 0.29 / 0.03 / 0.23 (v23 at 4M / 12M / 30M, `runs/reppo_oracle`).
So the honest summary is "true within-state rank correlation ~0.15-0.3, explained variance 2-9%", with one snapshot
(v23 12M) near zero. That is weak, but it is not "0.01".

It does not track outcomes [RAW, `metric_vs_return.py`]:

| step | n trajectories | Spearman(corr, return), all | upright only | median corr | median std_ratio | median slope corr/ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1M | 76 | -0.15 | -0.28 | 0.075 | 0.134 | 0.64 |
| 2M | 76 | +0.06 | +0.04 | 0.085 | 0.148 | 0.57 |
| 4M | 74 | -0.15 | -0.13 | 0.082 | 0.157 | 0.57 |
| 8M | 59 | -0.17 | -0.14 | 0.069 | 0.135 | 0.51 |

High readings belong to runs that move slowly or explore widely: flipped v35 0.10-0.14, v55_ids (flipped) 0.10-0.13,
v23 kl 0.01 0.13-0.16 (7.6k), v54 0.12-0.17 (8.7-9.2k), v43 0-nats 0.12-0.155 (8.5k), box 0.26-0.29 (<2.4k).
The metric is driven by (a) action spread relative to TD noise and (b) how little the policy moved since the critic
was fit. It is a property of the data regime more than of critic quality.

A PPO-relevant corollary nobody drew [REASONING]: a single GAE(0.95) sample has the same <= 0.34 correlation with the
true advantage, and PPO reaches 8.4k @8M and 14k @50M on it. So the critic's per-sample contrast accuracy
(rho ~0.15-0.3) is the same order as the raw signal PPO consumes. What differs is the error structure: GAE noise is
independent across samples and averages out over 16k rows; a critic's error is a deterministic function of (s, a),
correlated across states, and the actor takes 32 Adam steps directly up it.

---

## 4. Ledger: the six headline claims

| # | Claim | Evidence | Grade | Why |
|---|---|---|---|---|
| 1a | Critic gets the state level right (corr 0.93) | v60 1M oracle: corr 0.930, error sd 0.90 vs between sd 2.32 [RAW]. One snapshot. Truth has critic-bootstrapped time-limit tails (38% of branches, weight 0.27). v23 oracle value error/true sd 0.34-0.38 at 4-30M [RAW]. | B | consistent across 4 snapshots / 2 oracles; but 0.93 means level error sd 0.9 = **4x the true advantage sd**, hardly "right" at the scale that matters |
| 1b | Within-state ranking is near noise (held-out 0.01-0.14, oracle 0.11) | oracle 0.109 raw / 0.149 corrected, truth reliability 0.54; best-of-8 pick gains +0.27 sd (5 sigma; perfect = 1.42 sd) [RAW] | B for "weak (rho 0.15-0.3)"; D for "near noise / 0.01" | held-out metric has ceiling 0.34 and +0.02 bias (section 3) |
| 2 | Action gap: action part of Q is ~1/40 of the HL-Gauss label width, hence dQ/da is mostly noise | dist_target_spread 0.04 (v15), model branch spread 0.04 (v37, TF32), bins 2.0 [DOC] | C as description; D as cause | true sd is 0.22-0.33 not 0.04; HL-Gauss CE is smooth in the mean; v14 scalar MSE head no better (9.4-9.6k vs 10.2-10.5k) [RAW]; v59 was Huber (median) and admitted uninformative; v26 dueling and v63 orthogonal A (own scale) both regressed |
| 3 | Target variance is not the bottleneck (v61/v62 neutral) | v62 arms within fork noise [RAW] | **C/D** | the intervention did not reduce variance (section 0.5): q_mse unchanged at lam 0.8 and 3-4x higher at lam 0.95; correction uses the critic's own weak advantages; v61 from scratch: control and lam 0.8 both flipped |
| 4 | Replay / windowing / model imagination do not help | v6 buf8 4.7k vs buf1 8.2k (uncorrected chains); v10 retrace worse with buffer; v32/v33/v35/v37 flipped; v34 5.6k; v52 behind at 1M | C overall; D for the blanket form | v11 V-trace buf4/buf8 +2.4-2.6k over its buf1 control and late-blooming [RAW]; v32/v33/v35/v37 are flips; v33 cancelled at 1.1M for speed; v24 (the TD7-regime critic) stopped at **0.41M** and was never evaluated [RAW]; v52 cancelled at 1.2M; v34 mixes 1-step replay rows into every minibatch (target-type confound) |
| 5 | The ~2.3k failures are a discrete early flip, a seed lottery | posture: v35 99%, v37 99%, v39 98% inverted; v27 0% [DOC]; reference algorithm flips 2/4 [RAW] | B that 2.3k = on its back and that the reference itself flips; C for "discrete/irreversible/decided by 0.5M" and for "pure lottery" | v25 lam0.8 counter-example; FAMILY contradicts itself ("design-correlated" l.78 vs "lottery" l.256); flip rate may well differ by design (~10 of ~50 recent designs vs 2/4 reference - not distinguishable). PPO-family: 0 of 111 |
| 6 | The critic's error is "fitting/representation" | v62: held-out one-step Bellman residual unchanged by exact targets | C | follows from claim 3, which is untested. Independent support exists though: held-out one-step TD residual rms is 0.3 -> 1.1 -> 1.3 at 1.3/5/7.7M, a constant ~3x the critic's own advantage sd even in a flipped run (0.106, ratio 3.1) [RAW]. That is a noise-free measurement of fresh-data inconsistency; its cause (lag, overfit to the last rollout, capacity, optimization) is not identified |

---

## 5. Ledger: every version

Returns are [RAW] unless marked. "Ref" = path A (4174 / 5634 / 8596 / 11031). "Flip" = 2.0-2.5k plateau.

### v1-v29 (verdicts live only in file headers)

| Ver | Claim / verdict | Evidence | Grade | Notes |
|---|---|---|---|---|
| v1 | Standalone REPPO plateaus ~5k, PPO 8.4k | reppo_v1 4505; variants 3.9-5.4k | B | several arms agree |
| qprop_v1 | REPPO Q as a control variate helps PPO | 9035 vs noqprop 8011 @8M | C | +1.0k single seed, PPO-family noise unknown; header also says on-policy Q explains <1% of advantage variance |
| v2 | nGPT trunk does not help REPPO | cw64 2620, cw512 ~3.0k @6M | C | different lr / KL; erratic curves |
| v3 | Beta head on v2 fails | 643 @8M (collapsed) | C | v18 header itself calls it "no evidence against Beta" |
| v4 | Dual rates decoupled from update count | ep4_mb8 5120; lr sweeps 2.2-5.0k; ep10_mb1 collapses to -222 | C | no clean control |
| v5 | 1-step TD + 1M replay + per-iteration frozen target fails (2.3k) | all 3 arms incl. no-buffer control 2.3-2.4k | B that this configuration fails; C as evidence about replay | header: ~490 Bellman backups per run, measured propagation not data. All arms look flipped |
| v6 | lambda-return (GVE) targets; buf1 8.5k, replay hurts | buf1 8152 (l20 8547), buf8 4709, default (61 rollouts) 1978 | B | monotone dose-response over 3 arms; uncorrected stale chains |
| v7 | v6's gain = decoupling the actor rows from the critic's rows | default 8152 vs coupled 4006, all3 3962, epoch 2933, rolloutnorm 8555 | B | 4k effect, two arms agree (coupled, all3); still one seed; mechanism (imprint) later refuted by v8 |
| v7 cu | Critic updates per rollout are a lever: 32 / 64 / 128 / 256 | 8152 / 9591 / 10152 / 9819 | C+ | monotone to 128 but steps of 0.4-1.4k are at the MDE |
| v7 misc | No aux loss hurts; lam 0.8 / 0.99 hurt; critic width 1024 fails | noaux 6861; lam0p8 2327 (flip-like), lam0p99 3308; cw1024 2347 (flip-like) | B- for aux (3.3k); C for the rest | lam0p8 and cw1024 sit on the flip plateau - **capacity from scratch has never produced an upright run** |
| v8 | Cross-fit refutes state memorization | crossfit 5174 vs infold 6726 (l20 5457 vs 6216) | C | both far below v7 (half the data per critic); 0.8-1.5k difference |
| v9 | Frozen judge neutral; rollout length matters | frozen 8595 vs 8152; ns128 4227 @4M, ns256 7077, ns512 7219, ns2048 4094 | C | ns trend contradicts v36 ns128 (9871, faster early) - inconsistent across bases |
| v10 | Retrace hurts even at buf1 | 4194 / 3408 / 2038 for buf 1/4/8 | B that it fails; C on why | |
| v11 | V-trace replay at cu128 | buf1 6633, buf2 6359, buf4 9226, buf8 9013; retrace_buf1 2367 (flip) | **B- positive for replay within v11**; and an unexplained 3.5k gap to v7 cu128 | see section 0.7-0.8. Header's "memorization" hypothesis is contradicted by its own metric: taken_residual ~ sampled_residual (0.64 vs 0.65) [RAW] |
| v12 | Distilled distribution critic stalls 2.2k | r1 = r2 = 2241 @4M | C | flip plateau, bit-identical "repeat" |
| v13 | Actor hidden as critic context | r1 2218 (flip), r2 9240, identical hparams | - | **the cleanest flip-lottery exhibit in the repo** |
| v14 | Scalar MSE critic | 9571 (l20 9377) vs 10152 (l20 10505); 50M run 11017 @24M; rewnorm 9007 | C+ null | within MDE at 8M, -1.7k at 4M. The existing answer to "is HL-Gauss resolution the problem": no sign of it |
| v15 | Scalar difference head for distribution critic | 6251 @6M, erratic | C | |
| v16/v17 | Distribution-critic GVE loses to v7 cu128 | 4992 / 5287 @4M vs 8014 | B- | 3k, two related arms |
| v18 | Beta policy on cu128 | mode 8870, softplus 7708 vs 10152 | C+ | -1.3k / -2.4k |
| v19 | Beta without entropy collapses | 7027, std 0.21 -> 0.046 [DOC] | B | repeated by v47 "same" (7046) and v21 |
| v20 | TD7-style exploration budget | 8508 / 8325 | C | |
| v21 | LR gradient on precision still collapses | 7331, entropy -8.4 [DOC] | B- | supports "collapse is not purely a pathwise-curvature bias" |
| v22 | Clipped Gaussian; rules out bound saturation | 7511 / 8025 @8M | C | policy class and entropy accounting changed together |
| v23 | = v7 cu128 instrumented; oracle: dQ/da cos 0.2-0.3, advantages explain 0-10% | same trajectory as v7 cu128; oracle corrected corr 0.29 / 0.03 / 0.23, explained 8% / 0.1% / 5% at 4/12/30M [RAW] | B | snapshot-to-snapshot variation is 10x (true within sd 0.33 / 1.91 / 0.54); 50M last-20 10543 - **flat/declining to 50M** |
| v23 arms | Smaller KL / longer rollouts do not help | kl 0.015 7761, kl 0.01 7630 @8M, both ~11.0-11.2k by 33M; ns4096 9313 | C+ | |
| v24 | REPPO actor on TD7-regime critic (twin, 1-step, 1M replay, Polyak) | run has **409k steps** | untested | the single most relevant experiment for the TD7 gap was never run to a readable point |
| v25 | In-iteration target refresh + expected bootstrap; lam 0.8 | lam0.8 1842 / 3467 / 4920 / 9282, 13317 @50M; lam0.95 8933 @8M, 11936 @50M; lam0.5 flip; ns4096 13392 @50M | C at 8M (-0.9..-1.5k vs v23); **B at 50M** (+2.8k over v23's 10.5k, two arms) | at 8M the "improvement" is negative; the refresh matters for the asymptote |
| v26 | Dueling V + scalar A collapses action dependence | 3.5-4.1k, std_ratio 0.02-0.05, corr ~0 | B | large and mechanistically visible |
| v27 | Latent-roll aux, H8 is best | H8 11031 vs v25 9282 vs v23 10484; **H4 5020**; norew H4 9104 | C | +0.5k over v23 is inside noise; the H4/H8/reward 2x2 is non-monotone with a 6k range |
| v28 | Latent-Q critic | 2118 (flip plateau) | C | |
| v29 | Reward loss is not the bottleneck; gradient cos ~0; plateau diagnosis | noreward 10274 @8M / 11691 @16M vs 11031 / 11400; encoder grad norms [DOC] | B for "removing reward loss does not remove the plateau"; C for the bottleneck diagnosis | analysis.md is the most careful document: it flags the held-out metric as not independent, gives CIs, calls its own confidence moderate |

### v30-v63

| Ver | Claim / verdict | Evidence | Grade | Notes |
|---|---|---|---|---|
| v30 | Model value gradient does not help; world model accurate (R^2 0.99) yet its advantage corr 0.05-0.11 | w0.5 9588 @8M, 10919 @12M; w1.0 8810 | C+ | -1.4k / -2.2k; model advantage corr uses the same ceiling-0.34 metric |
| v27 ent 1.0/1.5 | Low fixed entropy kills early learning | 3.7k, culled 2.5M | C | culled before 3M (CLAUDE.md calls that winner-killing) |
| v31 | Price cap +1k at 16M, deterministic unchanged | 12432 / 12308 vs 11400 @16M; @8M 10832 / 11373 | B- | +0.9-1.0k at 16M, two caps agree; 8M effect nil |
| v32 | 1-step twin replay with Polyak fails | flip (2349) | C | FAMILY already retracted the mechanism |
| v33 | Mix targets | 1769 @1M, cancelled 1.1M | C | no information |
| v34 | "Genuine negative" for replay | 4839 / 5562 vs 8596 / 11031, upright | B- that this mix hurts; C about replay | 1024 one-step replay rows added to every minibatch: target-type and loss-mix confound |
| v35 | Windowed critic | flip x2 | C | |
| v36 | TF32 base slower; FP32 = v27 bit-identical | F vs A: 3847/4799/6045 vs 4174/5634/8596; v36_fp32 identical to A | A for code equivalence; **C+ for "TF32 alone halves learning"** | one TF32 trajectory vs two upright FP32 ones, while FP32 perturbations of the same size flip 2/4. FAMILY's own caveat ("TF32 may act as a new seed") was dropped two paragraphs later |
| v37 | Model-branch targets | flip (2363); **FP32 rerun also flips (2211 @4M)** | C | FP32 rerun result never written up |
| v38 | Scale-normalized A on CRN branch contrasts fails; "end of the model-imagination route" | 3155 @4M and rising (quoted 2568); **FP32 rerun 1486 @4M**, corr 0.06-0.07 | C+ | both precisions bad, so the TF32 re-opening is closed for v38; but "half inverted" posture, one seed, and A fit its own targets poorly (nMSE 0.78-0.93) so the idea was not really instantiated |
| v39 | Dual PG from scratch | flip | C | |
| v36 e128 | 128 envs x 128 steps: effective-sample-size hypothesis not supported | 3342/4704/5933, cancelled 4.4M; TF32 | C | changes rollout length and precision simultaneously; contradicts ns128 (below) |
| v40 fork | Fork protocol sound | control 11407 @12M vs unforked 11274 | B | |
| v40 | Observed-return estimator adds noise, not signal | w0.5 10972 (-0.4k); w1 8678 @10M | C for w0.5 (inside noise); B for pure LR hurting | 30% clip fraction at the pathwise KL step: the LR arm was not given PPO's own step regime |
| v40 | Smaller trust region does not raise the plateau; kl 0.003 degrades (TD3-overestimation signature) | kl01 11127; kl003 8071 | B that kl 0.003 degrades (-3.3k); C on the mechanism | "overestimation" then contradicted by v41 |
| v41 | Pessimism does not lift the plateau | first 11005, mean 10924, min 11401 | B- null | 0.4k spread = fork noise; second member trained 16 warmup iterations only |
| v42 | Capacity refuted | fresh 512 11130, fresh 1024x3 10410 @12M | C | fresh critics dropped into an 8M policy for 4M steps; never tested from scratch upright (v7_cw1024 flipped) |
| v42 ent | Entropy target is the one lever on the late plateau | -1.0 12291, -1.5 12178, -2.0 12399 vs 11408 @12M; det 12.8-13.7k vs 12.0-12.4k [DOC] | **B+** | +0.8-1.0k at every checkpoint in all three arms, ~4 fork-sd; the best-supported positive result in the record. Caveat: three arms share one snapshot |
| v43 anneal | Early entropy reduction harmful; 0 nats no better | TF32: f1p0 10153, f2p0 6936; m0 8500 (not 6460), m0p25 10067 (not 9440), m0_late1p5 11571 | C | TF32 base and point-value table errors; FAMILY already downgrades these |
| v43 FP32 | Late drop from scratch | both arms flip | C | = path D, the reference algorithm until 4M |
| v44 | Sharp-mean refuted | 6645 / 7400 vs 10894 | B- | -3.5..-4k, two arms sharing one trajectory to 4M |
| v45 | Population prevents the flip; 2048 rows/head faster early | mb2048 5347 / 6347 / 8554 / 11002; 512 rows 8868 | C for flip prevention; B- for the +1.2k @1M early gain | 0 flips in 4 trajectories has probability 0.13 at p = 0.4 (heads share a critic and are not independent; the "2%" in FAMILY treats 8 heads as independent). 8M score = reference |
| v46 | Late drop inside 8M no gain; K>4 slower | 10941 / 11020; K8 10170, K16 6425 @4M | C+ | |
| v47 | Independent spread critic slows collapse, +2k over control | crossfold 8259, indep 9084, same 7046 | B- | 3 arms ordered; still collapses; -2k vs ref at 8M but +1.5-2.2k at 1M |
| iterrate | Update rate is not the TD7 lever | ns128 3724 / 4968 / 5981 / 8606 / 9871; ns256 4398 @4M (flip-like) | C+ | 8x iterations gives +1.2k @0.5M, +0.8k @1M, -1.2k @8M. Batch per update, diagnostics rows and updates per iteration all change; ns256 contradicts monotonicity |
| v48 | "Law: critic ranking scales with behavior spread" | box 0.26-0.29, ref 0.07-0.14, floor 0.02-0.04 | C+ | the metric's ceiling itself moves with spread/TD-noise ratio; box is a non-learning policy; same-spread counter-examples: v23 kl 0.01 0.13-0.16, flipped v35 0.12-0.14; no oracle at different spreads |
| probe | Z is a location family; true contrast sd 0.33 vs critic 0.11; corr 0.2 of ceiling 0.7 | scratchpad scripts, not in repo [DOC] | B- | consistent with the other oracles; scripts not preserved |
| v49 | Drift temperature refuted | c1 peaks 3626 @2M then decays; c3 negative | B that it fails | "one iteration rewrites the ranking as much as it is worth" (drift = contrast sd) is an important measurement: C+ (one run, early) |
| v50 | Contrast anchor refuted | 1.8-1.95k @1M, cancelled 1.2M | C | culled at 1.2M on the flip plateau |
| v51 | Contrast-distribution head: fast start then stall | 4355 / 6971 / 8014 / 8445 / 7608 @1/2/3/4/6M; qctl 3382 @1M | B- for the early gain (+1.3k @2M, +0.95k @3M, bit-reproducible); stall is real | disk-full death at 7.46M; lambda sweep (v53) says the gain is fragile |
| v52 | V-trace replay window refuted | 3106 / 2272 @1M, cancelled 1.2M | C | see section 0.8 |
| v53 | D's own lambda | 3425 / 3512 / 6661 @3M vs 8343 | C+ | lam 0.0 / 0.5 curves look half-stalled |
| v54 | IDS temperature over-explores | 9159 / 8720 vs 10894; T 0.08-0.09 | B- | -1.7..-2.2k, two arms |
| v55 | Critic curvature at the mean is unbiased; only beta = 28-71% of the critic slope is real | probe table with SEs [DOC]; slope cross-checks in section 0.4 [RAW] | B | three estimators agree on the slope fraction; curvature claim rests on one run with ~35% relative SE |
| v56 | Calibrated IDS regression | 9736 @8M (l20 9697) vs 11069 | B- | -1.3k |
| v57 | Fixed mode-step geometry | 4459 / 5797 vs 5721 / 6986, 3-minute trial | C | authors' own verdict: miscalibrated by 5.7x, does not refute the hypothesis |
| landscape | Update direction good, length miscalibrated; late step is critic error; cos(critic grad, step) 0.07 / 0.03 / 0.008 | `runs/landscape/*` [DOC] | C+ | one run, 32 paired episodes; a later author notes the quadratic fits are not local measurements. The x5 @1M (+123 +- 14) and x2 @7.9M (-189 +- 50) paired results are B |
| v58 | Real-return line search regression | 3817 / 4838 / 6023, cancelled 4.3M | B- | -2.6k @4M |
| v59 | Scalar Q with Huber | 8520 | C | Huber = median regression; authors call it uninformative. v14 is the informative one |
| v60 | Checkpoint/resume transparent | base = v55_dual exactly | A | noise-fork experiments parked as underpowered (correctly) |
| v61 | Exact targets from scratch | control flip, l08 flip, l095 9908 (l20 9715) | C | lottery; and section 0.5 |
| v62 | Exact targets neutral -> target variance not the bottleneck | 10325 / 10590 / 9608 @8M, base 11069 | C (see claim 3) | the fork-pairing design itself is good and should be reused |
| oracle | 20-37% of the pathwise signal is real; Q self-inconsistent (-0.33) | v60 1M JSON | C+ | selective reporting (section 0.9); one snapshot at 1M, where T = 0.003 and the policy is far from the plateau |
| v63 | Orthogonal advantage critic regression; variance explanation refuted | fork 4876 / 5783 vs 5658 / 7896; +exact 5836 / 5258 | B- that this A fails; C on "variance refuted" | the "+exact" arms inherit v61's non-reduction of variance |

OPTIMUM.md specific claims:

| Claim | Grade | Note |
|---|---|---|
| TD7 17.1k @1M vs 6.4k best on-policy: "4x at 1M" | B (verified numbers) | 16 envs x REPPO vs single-env TD7 with ~64x critic compute/step [DOC]; a data-efficiency gap, not a diagnosis |
| "TF32 alone halved learning" | C+ | section 5, v36 |
| "held-out ranking correlation 0.01-0.14" as near the noise floor | D | section 3 |
| "KL gate binds in every run (kl 0.030, violation 0.33-0.36)" | B | v7/v11 scalars agree: kl 0.030, violation 0.33-0.39 [RAW] |
| "critic curvature at its own argmax is negative by construction (optimizer's curse)" | D | refuted by v55 inside FAMILY.md; OPTIMUM.md was not updated |
| "v46's K>4 failure was per-block states" | C | untested explanation |
| "on-policy lambda-returns on stale data is why replay failed in v32-v34" | D | v32 used 1-step targets; v32/v33 flipped; v34 used 1-step replay rows |

---

## 6. What REPPO has to beat

All HalfCheetah-v4, seed 1, matched +-50k window [RAW].

| Method | 1M | 2M | 4M | 8M | later |
|---|---:|---:|---:|---:|---|
| `ppo_continuous_action` (baseline PPO) | 2250 | 4082 | 6067 | 7428 | - |
| `sepclip_control_8M` (= adam_reference, dust_v1_backprop) | 2910 | 4770 | 6620 | 8398 | - |
| `awr_ce_ppo_sepclip_control_v1_50M` | 2401 | 3675 | 5680 | 8391 | 13576 @50M |
| `ppo_situ_reorg_v6_baseline_50M` | 2492 | 4543 | 6862 | 8030 | **14397 @50M** |
| `sepclip_mechanisms_rollout_clock_full` | 2701 | 4777 | 5903 | 8801 | - |
| `awr_adaptive_trace_ce_v19_50M` | 3321 | 5258 | 7172 | 9322 | 11814 @50M |
| `sepclip_reppo_hlgauss_v1_normreward_50M` | 2482 | 4796 | 7109 | 9369 | 11366 @50M |
| `reppo_qprop_v1` (PPO + REPPO-Q control variate) | 2890 | 4800 | 7260 | 9035 | - |
| REPPO reference, upright (path A / B) | 4174 / 4267 | 5634 / 5839 | 8596 / 9084 | 11031 / 11069 | 11.3-11.5k @16M; v25 lam0.8 13317 @50M; v23 10543 @50M |
| REPPO v31 price cap | 4174 | 5724-5767 | 8720-8800 | 10832-11373 | 12308-12432 @16M |
| best from-scratch non-REPPO in all 1015 HalfCheetah runs that reach 8M: `iterthink_v24_beta_d4hlgauss_..._tpoprobe_v1_k8` | 3410 | 5954 | 8931 | **10412** | - |
| next: `v24_beta_v162critic_mtp_v1`, `entppo_v4_samplemask_a03_mask`, `betacorr_g0p25_r09_8M`, `sf_vlam_v13` | 3.0-3.9k | 5.5-6.2k | 8.4-8.9k | 10.2-10.35k | - |
| best non-REPPO @16M: `residual_stiglu_ngpt_latent_qprop_50M_v1` | - | - | - | 8998 | 10768 @16M, 13565 end; `_v3` 14028 end |
| TD7 (`td7_sdnoise_v1`, `td7_fast_sdnoise_v1`, latentq v3) | 17.1-17.6k | - | - | - | 16.0-16.6k already at 0.5M; no TD7 run reaches 8M |

The last three non-TD7 rows come from the sweep of every HalfCheetah run (`all_hc_runs.tsv`; `fork_*` runs
excluded because they start from 12k checkpoints). I did not open those trainers; "on-policy" is inferred from
their PPO-lineage names. 12 of 1015 from-scratch non-REPPO runs reach >= 10.0k at 8M and 64 reach >= 9.0k; the
10.4k figure is the maximum over ~1000 single-seed runs and carries a winner's-curse bias of its own.

Reading:
- Conditional on staying upright, REPPO (11.0-11.1k) is ahead of the PPO baselines by 2.6-3.6k and ahead of the
  best PPO-lineage research variants by ~0.6-0.7k. The first gap is real; the second is inside the single-seed
  noise floor. REPPO is clearly faster early (4.2k vs 3.0-3.9k @1M) and ahead at 16M (11.4-12.4k vs 10.8k).
- Unconditionally (flip p ~0.4-0.5) the expected 8M score is roughly 0.55 x 11.0k + 0.45 x 2.4k ~ 7.1k, below the
  sepclip control. No PPO-baseline-family run sits on the 2.3k plateau (0 of 111). **Flip rate is worth more
  expected score (~4k) than every critic-quality idea tried so far.**
- At 50M PPO passes REPPO (14.4k vs 10.5-13.3k); REPPO's late phase is where the entropy constraint and the
  declining real fraction of dQ/da (beta 0.7 -> 0.28) bite.
- The TD7 gap (17k @1M) is 4x and has never been tested against REPPO's actor with a TD7-regime critic (v24 died
  at 0.41M).

---

## 7. Conclusions

### (a) Established (A/B)

1. Trainers are bit-deterministic; any RNG/op-order perturbation yields an independent trajectory. [A]
2. The 2.0-2.5k plateau is the cheetah running on its back (posture-verified on 3 runs), and the unmodified
   reference algorithm lands there in 2 of 4 perturbed trajectories (3 of 6 counting v13). [B]
3. Decoupling the actor's minibatch from the critic's (v7), lambda-return targets rather than once-per-rollout
   1-step targets (v5 vs v6), and the aux next-feature loss (v7 noaux) each matter by 3-4k. [B]
4. The critic is state-accurate at corr ~0.93-0.94 with level error sd ~0.9 (scaled), about 4x the true
   within-state advantage sd (0.22-0.33). [B]
5. Within-state action ranking is weak but real: true rank correlation ~0.15-0.3 (five of six oracle snapshots),
   best-of-8 gain 0.27 sd. [B]
6. Only ~30-70% of the critic's action slope is confirmed by real returns, falling to ~30-45% late
   (three estimators). The contrasts are over-dispersed relative to their accuracy. [B]
7. The held-out one-step Bellman residual on fresh data is ~3x the critic's own advantage sd in every regime
   examined, flipped runs included. [B as a measurement]
8. Lowering the entropy target late (fork from 8M) gains ~+0.9k stochastic / +0.6-1.5k deterministic; lowering it
   early or mid-run hurts. [B+ / B-]
9. Shrinking the trust region to KL 0.003 late degrades (-3.3k); a pure likelihood-ratio actor at the pathwise
   KL step degrades (-2.7k). [B]
10. Entropy-free pathwise actors collapse their spread (v19, v21, v47 same) and stall at 7.0-7.3k. [B]
11. v26 dueling with an unanchored scalar A collapses action dependence (std_ratio 0.02-0.05). [B]
12. v25's in-iteration target refresh raises the 50M asymptote (13.3k vs 10.5k) while lowering 8M. [B]
13. REPPO-upright is 2.6-3.6k above the PPO baselines at 8M and statistically level with the best PPO-lineage
    variants (10.3-10.4k); PPO-lineage wins at 50M (13.6-14.4k vs 10.5-13.3k); TD7 is 4x at 1M. [B]

### (b) Widely repeated, not established

1. "Held-out advantage correlation 0.01-0.14 means the ranking is near noise." Ceiling ~0.34, +0.02 bias, zero
   cross-run association with return.
2. "The action gap vs HL-Gauss label width is why dQ/da is noise." v14 MSE head is no better; true contrast sd is
   5-8x the quoted 0.04; CE is not quantized.
3. "Target variance is not the bottleneck." The only test did not reduce variance.
4. "Replay does not help REPPO." v11 says otherwise within its own control; most negatives are flips or early culls;
   the TD7-regime critic was never run.
5. "Model imagination cannot rank actions." v30 is a mild regression; v37 flipped twice; v38's A never fit its own
   targets. The measured facts (one-step R^2 0.996, branch ranking 0.03-0.06 vs real returns) are C-grade
   single-run numbers on a ceiling-limited metric.
6. "Capacity is refuted." Tested only as a fresh critic under an 8M policy; from scratch it flipped.
7. "TF32 alone halved learning." One TF32 trajectory.
8. "The flip is discrete, irreversible, decided by 0.5M" and "the population prevents it."
9. "v27 H8 is the best lineage / the aux horizon helps." H8 vs v23 is +0.5k; H4 is 5.0k.
10. "Update rate / effective sample size are not levers." ns128, ns256, e128 and v9's ns sweep disagree with each
    other.
11. "The plateau is invariant to estimator, step size, pessimism, capacity, replay, imagination, parallelism."
    Each leg is a single fork or a flipped run; FAMILY.md itself later calls the fork invariances provisional.
12. "The critic's curvature at its argmax is biased negative" (still in OPTIMUM.md; refuted by v55).

### (c) Most important untested or badly tested hypotheses about the critic

1. **Error structure, not error size.** The critic's per-sample contrast accuracy (rho 0.15-0.3) is comparable to
   the single-sample GAE signal PPO uses (<= 0.34). PPO's errors are independent across samples; the critic's are
   a smooth function the actor ascends for 32 Adam steps. Untested: measure how much of the critic's *batch-mean*
   policy-gradient direction is real (cosine between the pathwise parameter gradient and a large-sample
   likelihood-ratio or simulator gradient at fixed policy), instead of per-state correlations.
2. **One action per state.** Every (s, a) contrast must be inferred across states by generalization. The simulator
   can restore states (the oracle does). Training the critic with K branch actions per state for a short real
   horizon ("vine" data, real physics, not a learned model) has never been tried and directly supplies the
   within-state contrast that is missing. It also gives a clean upper bound: does a critic given low-noise
   within-state labels produce a better actor at all?
3. **Genuinely low-variance targets.** Needs a baseline that is accurate without the critic already being good:
   simulator-branched advantages, or CRN pairs of real rollouts. v61 could not do this.
4. **One-step consistency.** REPPO's Q disagrees with its own r + gamma V'(f(s, a)) in action direction (per-state
   median corr 0.55, cos 0.58) and its fresh-data TD residual is 3x its advantage sd. TD7's critic is trained on
   exactly that one-step structure. The REPPO-actor-on-TD7-critic experiment (v24) was never completed.
5. **Critic lag behind the policy.** v49 measured that one iteration of fitting changes the contrasts by as much
   as their sd; v23 kl 0.01 has the best held-out metric of any learning run. Whether the critic is chasing a
   moving policy (tracking error) vs noise-fitting has not been separated - e.g. freeze the actor for N iterations
   and watch whether held-out TD residual and oracle slope converge.
6. **Over-fitting of the action dependence** (slope 0.3-0.5 late). Never treated as such: no early stopping on a
   held-out rollout, no shrinkage of dQ/da by the measured beta, no weight decay / update-count sweep read against
   the slope metric. cu64/128/256 was read only on returns.
7. **Early-phase bistability as a critic problem.** Flipped runs have a *good-looking* critic (q_mse 0.02, held-out
   0.10-0.14). Is the flip the actor exploiting an early critic that values back-running, or pure policy-space
   bistability? PPO-family runs do not show the 2.3k plateau.
8. **Which estimand.** The soft, sampled-action lambda-return target mixes temperature (rising 10x over a run,
   shifting soft values) into every label; q_mse grows 10x from 2M to 8M while returns grow 2x. Not examined.

### (d) Negative results that deserve a re-test

| Result | Why re-test | How |
|---|---|---|
| v24 TD7-regime critic | never ran (0.41M) | fork-paired from an upright 1M checkpoint; run to >= 4M |
| v52 / v11 V-trace replay | only positive replay signal; v52 culled at 1.2M where v11's winners were behind | v11 buf1/4/8 on the v60 base, forked from an upright checkpoint, full 8M |
| v61/v62 exact targets | did not test the hypothesis | replace the critic baseline with simulator-branched or CRN advantages; verify q_mse actually falls |
| v42 capacity | fresh critic under a frozen-era policy | from scratch via the fork protocol at 1M with critic carried over through widening, or simply 2-3 perturbed starts |
| v37 / v38 model branches | flips and an A that did not fit | only after (c)2 with *real* branches says whether better contrasts help at all |
| v14 scalar head | closest thing to a test of the action-gap story, single seed | rerun on the v60 base with 2-3 perturbed trajectories; cheap and decisive for claim 2 |
| v36 TF32 | one trajectory | 2-3 TF32 trajectories via RNG perturbation; decides whether a 1.7x speedup is free |
| v43 / v46 late entropy drop from scratch | TF32 base, flips, point-value tables | fork from 4M upright checkpoint; the 8M fork result (B+) predicts a gain |
| v40 dual-PG / LR arm | run at the pathwise KL step with 30% clipping | give the LR term PPO's own step/clip regime; it is the natural fix for hypothesis (c)1 |
| v45 population | flip prevention unproven (p = 0.13) but would be worth ~4k expected | count flips over >= 8 perturbed trajectories for K = 1 vs K = 4 |

Protocol recommendation that follows from sections 1-2: no single from-scratch seed-1 run should decide anything.
Use (i) fork-pairing from a common upright checkpoint with a same-code control, (ii) at least 3 RNG-perturbed
trajectories per arm (the code is deterministic, so a perturbation flag is free), (iii) full +-50k windows that do
not straddle the end of the run, and (iv) report flip count and upright-conditional score separately.
