# 06 - Compute: where REPPO's GPU time and sample-gradients go, and the compute-optimal skeleton

Role: compute architect. Nothing was launched for this report: all numbers come from existing tfevents,
existing `update_profile.txt` kernel profiles, and static reading of the code.

Provenance tags used throughout:

- **[M]** measured from an artifact (tfevents scalar, kernel profile, FAMILY.md/CLAUDE.md table).
- **[D]** derived arithmetically from [M] numbers (the derivation is stated).
- **[E]** estimated from code (operation counts, FLOPs); not yet measured.

Reproduce the tables (CPU only, ~80 s at 3 processes):

```
.venv/bin/python benchmarks/reppo_rootcause/scripts/compute_mine_runs.py \
    benchmarks/reppo_rootcause/data/compute_runs.json reppo speed_v27 speed_v36 \
    HalfCheetah-v4__ppo_continuous_action__ sepclip_microscopy_v12 --jobs 3
.venv/bin/python benchmarks/reppo_rootcause/scripts/compute_tables.py benchmarks/reppo_rootcause/data/compute_runs.json timing
.venv/bin/python benchmarks/reppo_rootcause/scripts/compute_tables.py benchmarks/reppo_rootcause/data/compute_runs.json budget
.venv/bin/python benchmarks/reppo_rootcause/scripts/compute_profile_shares.py runs/HalfCheetah-v4__speed_v36_*/update_profile.txt
```

Caveats that apply to every throughput number: one seed; the `concurrent` column only counts overlap with
other *mined* runs (REPPO + the PPO baselines), so a run marked 1 may still have shared the box with another
family; phase timings are medians of the per-iteration PhaseTimer totals.

---

## 0. Headline

1. **REPPO is GEMM-bound on the GPU, and the GPU is the shared resource.** v60 spends 0.436 s of GPU per
   16k-step iteration in the update, plus 0.028 s in eager diagnostics [M]. PPO of the current standard spends
   0.012 s [M]. That is 27 GPU-microseconds per env step against 0.73: **36x**.
2. **Solo wall-clock ratio is 3.8x (21.4k vs 78-85k SPS) [M], but the aggregate ratio is ~7x.** Three concurrent
   REPPO runs each drop to 10.2-12.1k SPS with update = 1.10-1.30 s [M]: the box's REPPO throughput saturates at
   about 16384 / 0.45 s = 36k SPS no matter how many runs are queued [D], against 235-252k aggregate for six
   PPO runs [M, CLAUDE.md].
3. **About 69% of the update is the auxiliary H=8 latent roll and its reward head** (roll ~51%, reward ~18%)
   [D from the FP32 kernel profile, cross-checked by the v29 parent-vs-H8 artifact: 0.155 s vs 0.511 s].
   The critic's own cross-entropy path is ~13%, the actor ~4%, target refresh 4.9%, optimizer + clipping 3.1%.
4. **The archive says the critic is under-trained at 4 passes and saturated at ~16**; 32 passes buy nothing.
   No run ever varied rows-per-step at a fixed step count in the modern recipe, and no budget arm carries a
   held-out critic metric, so "which of steps / sample-gradients matters" is the open question that decides
   how cheap the skeleton can be.
5. **Target:** GPU budget <= 0.07 s per 16k-step iteration (from 0.46). That makes six concurrent runs
   CPU-bound like PPO (~200k aggregate SPS, ~6x today) and gives ~50-60k SPS solo. Exact PPO parity
   (>= 80k solo) needs update <= 0.03 s, which FP32 cannot reach at 16 passes of a 512-wide critic.

---

## 1. Cost model of the reference (v36 / v60 / v61)

### 1.1 Networks (HalfCheetah: obs 17, action 6) [E, counted from the constructors]

| Module | Layers | Parameters | Forward MAC per row |
|---|---|---:|---:|
| Actor `net` | 17->512 RMSNorm SiLU, 512->512 RMSNorm SiLU, 512->12 | 279,052 (+2 duals) | 0.277 M |
| Critic encoder | 23->512 RMSNorm SiLU, 512->512 | 275,456 | 0.274 M |
| Critic value head | SiLU, 512->512 RMSNorm SiLU, 512->151 | 340,631 | 0.339 M |
| Critic dynamics (aux) | 518->512 RMSNorm SiLU, 512->512 | 528,896 | 0.527 M per roll step |
| Critic reward head (aux) | SiLU, 512->512 RMSNorm SiLU, 512->1 | 263,681 | 0.263 M per latent |
| `zero_dist` | 151 logits | 151 | - |
| **Critic total** | | **1,408,815** | Q(s,a) forward = 0.613 M |

56% of the critic's parameters (792,577) exist only for the auxiliary loss. Unit costs used below
(MAC = multiply-accumulate; backward through weights ~ 2x forward, backward to the input only ~ 1x):
Q forward 0.613 M; Q forward + full backward 1.84 M; Q forward + backward to the action only 1.23 M;
actor forward + backward 0.83 M; one roll step forward + backward 1.58 M; reward head forward + backward 0.79 M.

### 1.2 Per-iteration schedule (16 envs x 1024 steps = 16,384 rows) [E, read from the update loop]

| Purpose | Calls / iteration | Rows / call | Network passes per call |
|---|---:|---:|---|
| Critic step (AdamW fused, clip 0.5) | 128 | 2,048 (uniform with replacement) | encoder + value head fwd/bwd (CE); 8 sequential dynamics steps fwd/bwd; reward head fwd/bwd on 8 x 2048 = 16,384 latents (one batched call); gather of 8 x 2048 x 512 cached target features |
| Actor step (AdamW fused, clip 0.5) | 32 (after every 4th critic step) | 2,048 | old-actor fwd (no grad); actor fwd/bwd; critic Q fwd + backward to the action only (`backward(inputs=actor_parameters)`); per-state KL gate; two dual losses |
| Target refresh | 8 (every 16 critic steps) | 16,384 | actor fwd; target-critic Q fwd on K=4 actions (65,536 rows); lambda-return via the GAE kernel; target-encoder fwd (cached features for the roll). v61 exact targets add one taken-action Q fwd |
| Diagnostic targets (v55+) | 2 | 16,384 | two more full refreshes (soft lam 0.95, hard) that feed only the diagnostics |
| Diagnostics (v55+, eager, uncompiled) | 1 | 4,096 / 4,096 / 16,384 | `critic_diagnostics` (11 Q-fwd equivalents per row), `information_price` (16), `regret_probe` (25, over the whole rollout) |
| Deterministic eval | every 25 iterations | 1 env, 1000 steps | host mirror, Python `gym.make` env: 0.078 s per eval [M] |

Optimizer steps per iteration: 160 (128 critic + 32 actor). Both networks are clipped on essentially every
step: median pre-clip norms are 1.84-1.97 (critic) and 2.3-2.6 (actor) against `max_grad_norm` 0.5 [M].
Under Adam that mostly rescales nothing, so the number of steps, not the batch size, sets how far parameters
move per iteration. This matters for section 4.

### 1.3 Sample-gradients per env step ("update-to-data")

| Stream | Count per env step | MMAC per env step [E] | Share of update FLOPs [E] | Share of update GPU time [D] |
|---|---:|---:|---:|---:|
| Critic CE gradients (Q fwd+bwd) | 16 | 29.4 | 8.0% | ~13% (10% GEMM + elementwise) |
| Aux: latent transition gradients | 128 (16 x H=8) | 202.5 | 55.3% | ~51% (43% GEMM + elementwise) |
| Aux: reward-head gradients | 128 | 100.8 | 27.5% | ~18% |
| Actor pathwise gradients (1 action/state) | 4 | 9.3 | 2.5% | ~4% |
| Target refresh forwards (K=4) | 32 Q fwd + 8 encoder fwd + 8 actor fwd | 24.0 | 6.6% | 4.9% [M] |
| Optimizer + clipping | 160 steps / iteration | - | - | 3.1% [M] |
| **Update total** | | **366** (6.0 TMAC = 12 TFLOP per iteration) | 100% | 100% = 0.436-0.468 s [M] |
| Diagnostics + diagnostic targets (outside `update`) | ~40 Q-fwd equivalents | 25.4 | (+7%) | 0.028 s [M, v60] |

PPO of the current standard (`sepclip_transition_microscopy_v12`, 64-wide actor and critic, full batch,
10 epochs): 10 sample-gradients per env step, ~0.3 MMAC per env step [E], 10 optimizer steps per iteration,
update 0.012 s [M]. REPPO therefore spends ~1000x the FLOPs and 36x the GPU time of PPO per env step; PPO's
update is launch-latency-bound (1.2 ms per full-batch step), REPPO's is arithmetic-bound.

How the time shares were derived. The FP32 profile of one v36 iteration
(`runs/HalfCheetah-v4__speed_v36_fp32__1__1790566603/update_profile.txt`) has 448 ms of self CUDA time
against a 468 ms update span, so the GPU is ~96% busy and CPU launch overhead is not the bottleneck
(the CUDA-graph capture works) [M]. By kernel class [M]:

| Class | FP32 | TF32 | BF16 |
|---|---:|---:|---:|
| GEMM (cutlass sgemm / tensorop, splitK) | 353.8 ms (78.9%), 30.8 us per launch | 191.1 ms (69.1%), 16.9 us | 118.4 ms (61.8%), 10.8 us |
| Triton elementwise (RMSNorm, SiLU, log-softmax, gathers, stack) | 62.5 ms (14.0%) | 54.0 ms (19.5%) | 42.7 ms (22.3%) |
| Optimizer + clip (`multi_tensor`, elementwise) | 13.8 ms (3.1%) | 10.8 ms | 11.0 ms |
| Self CUDA total | 448.2 ms | 276.7 ms | 191.6 ms |

Within the FP32 GEMMs [D from launch counts]: three kernels with 2,432 + 2,432 + 2,752 = 7,616 launches
(241 ms) are the 2,048-row 512x512 products; the roll accounts for 8 steps x 2 layers x 3 products x 128
= 6,144 of them (~194 ms, 43% of the update) and the CE path for ~1,408 (~44 ms, 10%). Three kernels with
128-160 launches at 148-229 us (76 ms, 17%) are the reward head's 16,384-row products. The two compiled
refresh regions total 8 x (2.465 + 0.271) ms = 21.9 ms (4.9%). Elementwise time was apportioned by layer
count. Large products are 30-45% more FLOP-efficient than 2,048-row ones (37 vs 26 TFLOP/s in FP32) [D],
which is why the reward head's time share (18%) is below its FLOP share (27.5%).

Independent cross-check from run-level artifacts (all solo, FP32) [M]:

| Run | Critic / actor steps | Aux | Refresh | update s / iteration |
|---|---|---|---|---:|
| `reppo_frozenjudge_v9` (v7 recipe) | 32 / 32 | 1-step | none | 0.051 |
| `reppo_probe_v23_50M` (v7 cu128 + diagnostics) | 128 / 32 | 1-step | none | 0.132 |
| `reppo_diagnose_v29_parent_lam08` (v25) | 128 / 32 | 1-step | 8 x K4 | 0.155 |
| `speed_v27h8` | 128 / 32 | H8 + reward, per-step target re-encode | 8 x K4 | 0.508 |
| `speed_v36_fp32` | 128 / 32 | H8 + reward, cached target features | 8 x K4 | 0.468 |
| `reppo_noisefork_v60_base` | 128 / 32 | H8 + reward | 8 x K4 | 0.436 (+0.023 diagnostics, +0.005 targets) |

So: 96 extra critic steps cost 0.08 s (0.8 ms each with a 1-step aux); 8 refreshes cost 0.023 s; going from
a 1-step aux to H8 + reward costs 0.31-0.36 s. The roll dominates everything else combined.
`reppo_diagnose_v29_h8_noreward` (0.497 s) is not a reward-head timing: the head is still evaluated and
only its coefficient is zero.

### 1.4 Measured throughput and phase totals (seconds per 16,384-step iteration) [M]

| Run | Precision | Concurrent (mined) | SPS (cumulative / after 1M) | env | rollout (host actor) | norm+transfer | update | diagnostics + targets | 8M wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO `sepclip_microscopy_v12_adam_control` | - | 1 | 78.1k / 84.6k | 0.125 | 0.042 | 0.007 | 0.012 | gae 0.001 | 102 s |
| PPO `sepclip_microscopy_v12_anvil_control` | - | 1 | 75.6k / 78.8k | 0.126 | 0.042 | 0.008 | 0.021 | gae 0.001 | - |
| `speed_v27h8` (800k steps) | FP32 | 1 | 19.4k / - | 0.137 | 0.144 | 0.010 | 0.508 | 0.005 | - |
| `speed_v36_fp32` (800k) | FP32 | 1 | 16.9k / - | 0.139 | 0.126 | 0.010 | 0.468 | 0.005 | - |
| `speed_v36_tf32` (800k) | TF32 | 1 | 19.7k / - | 0.147 | 0.128 | 0.010 | 0.282 | 0.004 | - |
| `speed_v36_bf16` (800k) | BF16 | 1 | 13.3k / - | 0.146 | 0.129 | 0.011 | 0.188 | 0.004 | - |
| `reppo_fast_v36_fp32` | FP32 | 3 | 10.6k / 10.3k | 0.133 | 0.127 | 0.010 | 1.303 | 0.012 | 755 s |
| `reppo_fast_v36_e128` (128 envs x 128) | TF32 | 2 | 19.9k / 20.6k | 0.127 | 0.113 | 0.003 | 0.539 | 0.006 | - |
| `reppo_regretprobe_v55_dual` | FP32 | 2 | 14.5k / 14.7k | 0.149 | 0.136 | 0.012 | 0.752 | 0.058 | - |
| `reppo_landscape_ref_v55dual` | FP32 | 1 | 21.4k / 21.5k | 0.132 | 0.128 | 0.011 | 0.432 | 0.027 | - |
| `reppo_noisefork_v60_base` | FP32 | 1 | 21.0k / 21.5k | 0.134 | 0.129 | 0.011 | 0.436 | 0.028 | 379 s |
| `reppo_exacttarget_v61_l095` | FP32 | 3 | 11.9k / 12.1k | 0.146 | 0.133 | 0.012 | 1.101 | 0.087 | 672 s |
| `reppo_exacttarget_v61_control` (to 3.7M) | FP32 | 3 | 10.7k / 10.9k | 0.140 | 0.138 | 0.011 | 1.123 | 0.075 | - |

Cumulative SPS of the 800k speed-ladder runs is dominated by compile time; use their phase columns.
There is no solo v61 run in the archive; its update cost at 3 concurrent equals v36's at 3 concurrent, so its
solo cost is expected to be v60's plus ~4 ms for the extra taken-action forward per refresh [E].

REPPO vs PPO, explicitly:

| Quantity | PPO v12 | REPPO v60 | Ratio |
|---|---:|---:|---:|
| Iteration wall, solo [M] | 0.187 s | 0.74 s | 4.0x |
| SPS, solo, after 1M [M] | 78.8-84.6k | 21.5k | 3.7-3.9x |
| GPU seconds per iteration [M] | 0.012 | 0.464 | 39x |
| Host actor seconds per iteration [M] | 0.042 | 0.129 | 3.1x |
| Env + normalize/transfer [M] | 0.132 | 0.145 | 1.1x |
| Aggregate box throughput | 235-252k (6 runs) [M, CLAUDE.md] | ~31-36k (3 runs x 10.3-12.1k) [M] | ~7x |
| Return at 8M (seed 1) [M] | 8.4-8.7k | 11.07k | - |
| Wall to reach 8.4k, solo [D] | 102 s (8M steps) | ~170 s (~3.6M steps) | 1.7x slower |

Two readings that matter for the study: REPPO's sample efficiency does not yet pay for its compute even in
solo wall-clock to PPO's final score, and in the box's real operating mode (several queued runs) REPPO
experiments are GPU-serialized: a second concurrent run buys ~35% more aggregate throughput, a third nothing.

---

## 2. What the archive says about update budget vs outcome

All rows: HalfCheetah-v4, seed 1, windowed mean return (+-50k steps) [M]. Critic metrics are the training-time
means logged by the run: `critic_ce` and `q_mse` are **in-sample** (on the minibatches being fit, against
the targets being chased), so they fall with more updates by construction and are not evidence of critic
quality. Only v23+ logs a held-out metric (`advantage_corr_heldout`: correlation of the critic's advantage
with the return advantage on the next rollout's fresh actions before any fit to them).

### 2.1 Critic and actor updates (v7 recipe: lam 0.95, 1-step aux, no refresh, 2,048-row minibatches)

| Arm | Critic steps (passes) | Actor steps | R@1M | R@4M | R@8M | critic_ce @1M/4M/8M | update s (concurrency) |
|---|---|---:|---:|---:|---:|---|---|
| `reppo_decoupled_v7` | 32 (4) | 32 | 2585 | 6218 | 8152 | 1.49 / 1.37 / 1.42 | 0.063 (5) |
| `reppo_v7_q_cu64` | 64 (8) | 32 | 3415 | 7579 | 9591 | 1.37 / 1.51 / 1.55 | 0.166 (5) |
| `reppo_v7_q_cu128` (= r1 = r2, bit-identical) | 128 (16) | 32 | 4583 | 8014 | 10152 | 1.27 / 1.37 / 1.46 | 0.127 solo-ish |
| `reppo_v7_q_cu256` | 256 (32) | 32 | 4132 | 7696 | 9819 | 1.23 / 1.35 / 1.35 | 0.479 (6) |
| `reppo_v7_q_cu128_au16` | 128 (16) | 16 | 3426 | 7584 | 9493 | 1.35 / 1.36 / 1.44 | - |
| `reppo_v7_q_cu64_au64` | 64 (8) | 64 | 3814 | 6757 | 9142 | 1.28 / 1.29 / 1.41 | - |
| `reppo_v7_q_cw1024` (width 1024) | 32 (4) | 32 | 1570 | 2187 | 2347 | 1.22 / 1.21 / 1.21 | collapse |
| `reppo_decoupled_v7_epoch` (permutation epochs instead of uniform rows) | 32 (4) | 32 | 2449 | 3887 | 2933 | - | unstable |

Reading:

- **4 -> 16 passes is a real gain** (+2.0k at 1M, +1.8k at 4M, +2.0k at 8M); 8 passes sit in between.
- **16 -> 32 passes is flat to slightly negative** (-0.45k at 1M, -0.3k at 8M) while in-sample CE keeps
  falling (1.46 -> 1.35) and the aux loss triples (0.097 -> 0.274 at 8M). That is the signature of fitting
  the rollout's target noise rather than under-training: more passes lower the training loss and do not
  raise the return. One seed, so "flat" is the defensible word, not "worse".
- Actor steps are a weak lever: 16 -> 32 helps early (3426 -> 4583 at 1M), 32 -> 64 (at 8 critic passes)
  does not.
- The 2.3k outcome (`cw1024`, and `lam0p8`/`replay_v5`/`refresh_v25_lam0p5` in other tables) always comes
  with CE pinned at 1.20-1.21 and q_mse < 0.05: the critic has collapsed onto near-constant targets. Low
  in-sample loss is the failure there, which is one more reason not to read CE as quality.

### 2.2 Aux horizon and reward head (v27-v29 recipe, 128 / 32 steps, refresh 16)

| Arm | Aux | R@1M | R@4M | R@8M | adv_corr_heldout @1M/4M/8M | update s solo |
|---|---|---:|---:|---:|---|---:|
| `reppo_diagnose_v29_parent_lam08` | 1-step | 1842 | 4920 | 9282 | 0.064 / 0.086 / 0.111 | 0.155 |
| `reppo_latentroll_v27_norew` | H4, no reward | 3632 | 5274 | 9104 | 0.115 / 0.083 / 0.133 | 0.75 at 3 concurrent |
| `reppo_latentroll_v27_full` | H4 + reward | 3907 | 3817 | 5020 (flip) | 0.088 / 0.051 / 0.062 | 0.83 at 3 concurrent |
| `reppo_diagnose_v29_h8_noreward` | H8, reward coef 0 | 4682 | 7637 | 10274 | 0.091 / 0.116 / 0.058 | 0.497 |
| `reppo_diagnose_v29_h8_reward` (= v27 H8 = v36 FP32) | H8 + reward | 4174 | 8596 | 11031 | 0.090 / 0.068 / 0.055 | 0.511 |

- The roll is where the early sample efficiency comes from: 2.0-2.5x the parent's return at 1M, +1.7k at 8M.
- It is also 70% of the update. In wall-clock terms on a solo box [D]: the parent runs 35.8k SPS and H8 19.8k,
  so by the time H8 has done 8M steps (11.0k) the parent has done ~14.4M (its 16M last-20 is 10.4k). H8 is
  still ahead per second, but by ~6%, not by the 2x its step curve suggests.
- **The roll does not improve the held-out critic metric.** `advantage_corr_heldout` is 0.05-0.13 in every
  arm and, if anything, lower for H8 at 8M (0.055-0.058) than for the parent (0.111). Whatever the roll buys,
  it is not a visibly better action ranking on fresh data.
- The reward head is unresolved at one seed (10.3k without, 11.0k with) and costs ~18% of the update.

### 2.3 Rollout shape

| Arm | Envs x steps | Passes | R@500k | R@1M | R@4M | R@8M | adv_corr @1M/4M/8M |
|---|---|---:|---:|---:|---:|---:|---|
| `reppo_v9_live_ns128` (v7 recipe) | 16 x 128 | 4 | 128 | 1478 | 4227 | - | - |
| `reppo_v9_live_ns256` | 16 x 256 | 4 | 1159 | 1777 | 5614 | 7077 | - |
| `reppo_v9_live_ns512` | 16 x 512 | 4 | 1630 | 2234 | 6005 | 7219 | - |
| `reppo_decoupled_v7` / `frozenjudge_v9` | 16 x 1024 | 4 | 1547 / 958 | 2585 / 2115 | 6218 / 5594 | 8152 / 8595 | - |
| `reppo_frozenjudge_v9_live_ns2048` | 16 x 2048 | 4 | 300 | 1496 | 3545 | 4094 | - |
| `reppo_iterrate_v36_ns128` (v36 FP32) | 16 x 128 | 16 | 3724 | 4968 | 8606 | 9871 | 0.067 / 0.088 / 0.059 |
| `reppo_iterrate_v36_ns256` | 16 x 256 | 16 | 2478 | 2887 | 4398 (flip, culled 5.8M) | - | 0.062 / 0.066 / - |
| `reppo_fast_v36_fp32` | 16 x 1024 | 16 | 2521 | 4174 | 8596 | 10894 | 0.090 / 0.068 / 0.038 |
| `reppo_fast_v36_e128` (TF32) | 128 x 128 | 16 | 816 | 3342 | 5933 | - | 0.108 / 0.111 / - |
| `reppo_refresh_v25_lam0p8` (lam 0.8, 1-step aux) | 16 x 1024 | 16 | 1395 | 1842 | 4920 | 9282 | 0.064 / 0.086 / 0.111 |
| `reppo_refresh_v25_lam0p8_ns4096` | 16 x 4096 | 16 | 614 | 2781 | 5674 | 8106 | 0.200 / 0.169 / 0.144 |

- At fixed passes, 16k rows per iteration is the best 8M shape in both recipes; 2k-8k rows are 0.5-1.0k
  lower at 8M and 64k rows are slower early with the same asymptote (v25 at 50M: 13.3k vs 13.4k last-20).
  Shorter rollouts are faster to 500k (3724 vs 2521) and then lose it.
- **More fresh data per fit is the one thing in the archive that moves the held-out metric:** 64k-row
  rollouts give `advantage_corr_heldout` 0.14-0.20 against 0.06-0.11 at 16k rows. Update count, aux horizon
  and precision do not move it. The critic's limit looks like data per policy, not optimization.
- 128 envs x 128 steps is neutral against 16 x 1024 at the same precision (TF32 references: 3.5-3.8k at 1M,
  6.0-6.8k at 4M [M, FAMILY.md]) and costs the same per sample in every phase: env 0.127 vs 0.133 s,
  host actor 0.113 vs 0.127 s per 16k rows [M]. Rollout shape is not a compute lever on this box at two
  physics threads; it is a learning lever and an enabler for GPU-side acting (section 4.5).

### 2.4 Minibatch size and full-batch updates

Only the earliest versions varied it, with a coupled actor/critic update, so the evidence is weak:
`reppo_v1_ppobatch_*` and `reppo_dualrate_v4_*` full-batch arms (4 or 10 steps of 16,384 rows at lr 3e-4 or
1e-3) reach 1.6-3.2k at 4M against 3.4-4.3k for 8 minibatches x 4 epochs; `reppo_dualrate_v4_ep10_mb1` ends
at -222. Full-batch REPPO means 10-16 Adam steps per iteration, ~5-8k steps in 8M: too few for a 1.4M-parameter
critic at lr 3e-4, and raising the rate (v4 at 1.2e-3: 2.2k) did not rescue it. `reppo_headpop_v45` (512 rows
per head) vs `_mb2048` (8868 vs 11002 at 8M) changes the population structure as well, so it does not isolate
rows per step.

### 2.5 Precision

TF32 cuts the update 40% (0.468 -> 0.282 s) and BF16 60% (0.188 s) [M]. TF32 costs ~2.5k of return at 4M
across every v36-lineage from-scratch run (FP32 4150 / 5620-5740 / 8280-8660 at 1M / 2M / 4M against TF32
3530-3820 / - / 6.0-6.8k) [M, FAMILY.md]; FP32 v36 is bit-identical to v27. BF16 was only run to 800k
(2432 at 500k vs 2521). Which path is precision-sensitive was never isolated.

### 2.6 Answer

- **Under-trained below 16 passes, saturated at 16, not helped by 32** (v7 recipe; 16 passes was never
  re-swept after the roll and refresh were added).
- **Not over-trained in the damaging sense at the reference budget**, but the last doubling only lowers the
  in-sample loss, and the held-out metric is flat at ~0.05-0.11 across every optimization-side change.
- **Insensitive to the actor's budget** within 16-64 steps.
- **Sensitive to data per fit and to aux horizon in return, but only data moves held-out critic quality.**
- Missing, and needed to design the skeleton: (a) rows per step at fixed steps; (b) a held-out critic metric
  on budget arms; (c) aux amortization (fewer rows / fewer steps carrying the roll); (d) a second seed on any
  of it - several arms differ by less than the flip-prone seed-to-seed spread.

---

## 3. Compliance / efficiency audit of v60 and v61 against the shared standards

| Item | Status in v60 / v61 | Gap and expected saving |
|---|---|---|
| `runtime.configure_runtime()` before networks | Yes (`matmul_precision`, `allow_tf32`) | none |
| `mujoco_env.make_mujoco_vector_env(..., num_threads=2)` | Yes (`--env-threads 2`) | none; env 0.134 vs PPO's 0.125 s is within noise |
| `VectorObsNorm` | Yes | Observations are stored raw and re-normalized on the GPU inside every loss with per-env statistics (`normalize(rollout_obs[rows], rows)`): three gathers + arithmetic per step for a result that is constant within the iteration. Normalize once in `insert`: < 2 ms / iteration [E] |
| `VectorRewardNorm` | Not used; fixed `reward_scale` 0.1 | Deliberate (stationary categorical support). Not a cost; flag it as a declared deviation in the header |
| `staggered_envs` (+ stochastic warmup) | Yes (`act` samples) | none |
| `ppo_loop` GAE / truncation bootstrap / metrics | `get_gae_fn(explicit_next_values=True)`, `gather_metrics` | none |
| `ppo_loop` minibatching | Not used: uniform rows with replacement drawn inside the captured graph | Deliberate and graph-safe. The one permutation-epoch arm in the archive (`v7_epoch`) was unstable; leave it |
| `.item()` in the optimizer path | None; sums accumulate on device, one `gather_metrics` per iteration | none |
| Host mirror rollout actor | Yes, `make_host_mirror(actor.net, 16)` + custom `HostSquashedGaussian` | Compliant but **3.1x PPO's cost: 0.129 vs 0.042 s** [M]. The actor is 512 wide; 16 rows x 0.277 MMAC in ~126 us is ~70 GFLOP/s on one core, so the mirror is arithmetic-bound, not dispatch-bound [D]. Saving ~0.08-0.09 s / iteration with a 256-wide actor (`--actor-hidden-dim 256`, a learning change) or with GPU acting at >= 128 envs |
| `make_beta_sampler` | n/a (tanh-Gaussian) | none |
| `RolloutTransfer(fields=...)` | Yes (`raw_obs`, `raw_next_obs`, `actions`, `noise`) | normalize+transfer 0.011 vs 0.007 s: the two raw observation copies. ~4 ms; not worth touching |
| `PhaseTimer` env / rollout / update | Yes, plus insert / targets / diagnostics / eval | none |
| Compile + CUDA graphs | `graph_compile` losses, `capture_steps` for both optimizer steps; GPU 96% busy during the update [M] | Working. `--compile-mode` is accepted and ignored (documented). The three diagnostics functions and their two extra target refreshes are **not compiled and run every iteration**: 0.028 s solo, 0.06-0.09 s under contention [M] = 6% of the GPU time now and as much as the whole update of the target skeleton. Gate to every Nth iteration: saves ~0.025 s |
| Fused optimizers | AdamW `fused=True`; `clip_grad_norm_(foreach=True)` inside the graph | 3.1% of the update; none |
| AMP / TF32 / BF16 | Flags exist; default strict FP32 since v44 because TF32 costs return | Largest unspent lever (-40% / -60%), blocked on root-causing the precision sensitivity (section 4.6) |
| Update structure | 128 critic steps each carrying the full H8 roll and an 8 x 2048-row reward batch | **The dominant gap: ~69% of the update.** Section 4.2 |
| Target refresh | 8 x (K=4 over the full rollout) + target-feature encode | 4.9% (22 ms). Fine now; 30% of the target skeleton. Section 4.4 |
| Aux coefficients | `--aux-loss-coef 0` / `--reward-loss-coef 0` still run the roll and the head | An ablation flag that saves nothing; make zero coefficients skip the compute |
| Deterministic eval | Python `gym.make` episode every 25 iterations: 0.078 s | 3 ms / iteration amortized; fine |
| v61 only | `refresh_targets(exact=True)` adds a 16,384-row Q forward per refresh; Bellman diagnostics add a `bootstrap_values` call on 4,096 rows | ~+4 ms update, ~+10 ms diagnostics [E]; no solo run exists to confirm |

Summary: v60/v61 are wired to the standard. Their cost is the algorithm's update schedule, the 512-wide host
actor and the per-iteration diagnostics, not missing infrastructure.

---

## 4. The compute-optimal skeleton

### 4.1 Budget

Floor that no REPPO design changes at 16 envs x 2 threads: env 0.125-0.134 s + normalize/transfer 0.008-0.011 s
per 16k steps = ~0.14 s, a ceiling of ~115k SPS [M].

| Tier | Iteration (solo) | SPS solo | GPU s / iteration | Aggregate ceiling, 6 runs [D] |
|---|---:|---:|---:|---:|
| v60 today [M] | 0.74 s | 21.5k | 0.46 | ~36k (GPU-bound) |
| Target A, "queue-friendly" | 0.27-0.30 s | 55-60k | <= 0.07 | ~200k+ (CPU-bound, like PPO) |
| Target B, "PPO speed" | <= 0.20 s | >= 80k | <= 0.03 | 235-250k |
| PPO v12 [M] | 0.187 s | 79-85k | 0.012 | 235-252k |

Aggregate ceiling = min(CPU-bound ~240k, 16384 / GPU-seconds-per-iteration). The second term is the one REPPO
hits today, and it is confirmed by the archive (3 runs: update 1.10-1.30 s each, 31-36k total).

Target A's phase budget: env 0.130 + transfer 0.010 + host/GPU actor <= 0.045 + update <= 0.06 + targets and
amortized diagnostics <= 0.01 + logging. Target A is the one to build to; B is only reachable with reduced
precision or fewer critic passes (4.7).

### 4.2 Aux loss: amortize it, do not carry it on every critic step (largest saving)

Today every one of the 128 critic steps runs 8 sequential transition steps and 16,384 reward predictions.
The aux loss is a dense regression (512 target dimensions x 8 steps per row), so its gradient is far less
noisy per row than the CE's; it does not need 16 passes over the rollout. Options, cheapest first:

1. Roll on every 4th critic step (32 roll steps per iteration, weight x4 or unchanged): update
   0.436 -> ~0.21 s [E: removes 3/4 of 69%].
2. Reward head on z_0 and one random later latent instead of all 8: removes ~3/4 of the 18% [E].
3. Roll on a 512-row slice of each 2,048-row minibatch: less efficient than (1) because 512-row GEMMs hit the
   ~10 us launch floor (BF16 profile) rather than scaling 4x [D].
4. Narrow latent for the roll (project 512 -> 128 before the dynamics): ~16x fewer roll FLOPs; a critic
   architecture change, so only if the winning critic fix keeps a roll at all.
5. Drop it: the parent runs at 0.155 s but loses 2.3k at 1M and 1.7k at 8M (one seed).

Expected with (1) + (2): aux 0.30 s -> ~0.07 s, update ~0.20 s, 31k SPS solo, GPU ceiling ~70k.
Learning risk: unknown, which is why it is the first measurement in section 5. Section 2.2 already shows the
roll does not raise the held-out advantage correlation, so a critic fix that works through data or targets
may make the roll removable outright.

### 4.3 Minibatch vs full batch, and steps vs sample-gradients (second largest)

- Full batch is the wrong shape for this critic: it is 30-45% more FLOP-efficient per sample [D] but gives
  8-16x fewer Adam steps, and the archive's full-batch arms are the worst of their family (2.4).
- The decisive unknown is whether the 4 -> 16 pass gain of 2.1 came from 4x more **steps** or 4x more
  **sample-gradients**. Gradients are clipped on every step and Adam normalizes scale, so parameter travel
  per iteration is set by the step count; a smaller minibatch only adds gradient noise. If steps are what
  matter, 128 steps x 1,024 rows keeps the trajectory and halves every per-row cost; 128 x 512 quarters it
  down to the launch floor.
- Recommended skeleton default until measured: 128 critic steps x 1,024 rows (8 passes), 32 actor steps x
  2,048 rows. CE path ~55 ms -> ~33 ms [E, GEMMs at 1,024 rows are ~0.6x the 2,048-row time].

### 4.4 Target refresh

22 ms now. In the lean skeleton cut it to ~8 ms: K=2 with v61's Rao-Blackwellized log-density (the Gaussian
term's expectation is exact, so most of the sampling noise K=4 was averaging is already gone), refresh every
32 steps, and encode target features only if a roll consumes them. The action enters the critic at the first
layer, so nothing of the Q forward is shared across the K actions of a state; see 4.6 for the structural fix.

### 4.5 Rollout actor and rollout shape

- The host mirror is arithmetic-bound at 512 wide (0.129 s, as much as physics). A 256-wide actor is ~4x
  cheaper in the hidden product: ~0.04-0.05 s [E], PPO's level. `--actor-hidden-dim 256` exists.
- More envs do not lower per-sample cost of physics or of the host actor at two threads (128 envs: env 0.127,
  actor 0.113 s per 16k rows [M]). They do make GPU acting viable: 256 envs x 64 steps is 64 policy calls per
  iteration; one captured `RolloutStepGraph` replay per call is ~0.1-0.15 ms [E], ~10 ms per iteration against
  113-129 ms on the host. That keeps a 512-wide actor at PPO-level rollout cost. REPPO's lam 0.8 targets
  (effective horizon ~5 steps) tolerate 64-128-step rollouts, unlike PPO's GAE 0.95; the H8 chain would lose
  8/64 of its rows at rollout ends instead of 8/1024. Learning evidence: 128 x 128 is neutral at matched
  precision (2.3). The throughput doc's N=64 sweep found eight physics threads fastest for a single run,
  which is the wrong operating point for a shared box; stay at two.
- Recommended: keep 16 x 1024 with a 256-wide actor as the default; treat 128-256 envs + GPU acting as the
  option if the wider actor proves necessary.

### 4.6 Fusing actor and critic, reusing features, and many actions per state

- **Fusing the two optimizer steps into one compiled graph buys < 3%.** The update is 96% GPU-busy and
  79% GEMM; launch and optimizer overhead together are ~3% [M]. Fusion cannot share arithmetic either: the
  critic step needs weight gradients at the data action, the actor step needs input gradients at the policy
  action with the critic's weights excluded, and a single stacked forward would mix the two halves into the
  weight gradient. The existing `backward(inputs=actor_parameters)` already skips the critic's weight-gradient
  GEMMs, which is the saving that matters. Keep two captured graphs.
- **Trunk reuse is impossible with early action fusion.** `encoder(cat(obs, action))` makes every action a
  full 0.613 MMAC forward. A late-fusion critic (state trunk phi(s) once per row; action enters a small head,
  e.g. [phi, a] -> 256 -> 151 bins, ~0.17 MMAC per action) makes extra actions 3.6x cheaper and lets the
  actor step, the K bootstrap actions and the CE step share one trunk forward per row [E]. This is the only
  design in which "many reparameterized actions per state" is close to free. It changes the critic's function
  class exactly where it is weakest (advantage_std_ratio 0.10-0.20), so it belongs to the critic-fix
  investigators to accept or reject; the compute case for it is strong.
- **What the actor can afford today:** the actor is 4% of the update (16 ms). Each extra action per state
  costs one Q forward + input backward (1.23 MMAC) and no actor passes; K=4 at 32 steps adds ~18 ms and K=8
  ~45 ms [E]. Against today's update that is +4% / +10%; against target A it is a third to most of the budget.
  More distinct states cost 1.9x as much per dQ/da as more actions at the same state (they add the actor's own
  passes). Which reduces the actor's gradient variance more per GPU-second is unmeasured; section 5 has the
  snapshot measurement that decides it.
- Old-policy moments are constant within an iteration: compute them once for the rollout (one 16,384-row
  forward) and gather, instead of 32 forwards of 2,048 rows. ~1 ms; do it for cleanliness, not speed.

### 4.7 Precision

TF32 and BF16 are the only route to target B at 16 passes, and they are blocked by a measured return cost
with no identified mechanism. A plausible one [E, hypothesis]: TF32 rounds matmul inputs to 10 mantissa bits,
and the critic's action dependence is a small difference on a large value (advantage std ~0.13 of the return
advantage), so the first layer's `cat(obs, action)` product and the 151-bin readout lose exactly the signal
the actor differentiates. If so, a split-precision update recovers most of the speed safely: BF16 for the
roll and reward head (69% of the time, stop-gradient regression targets), strict FP32 for the encoder's first
layer, the value head, the CE loss and the actor's pathwise pass. Expected update 0.436 -> ~0.25 s on its own
and ~0.05 s on top of 4.2 + 4.3 [E]. It needs a new version (the existing flags are global).

### 4.8 The skeleton, assembled

Per iteration, 16 envs x 1024 steps, strict FP32 unless noted:

1. Rollout: host mirror, 256-wide actor (or GPU acting at >= 128 envs). <= 0.045 s.
2. Insert: upload, normalize observations once, old-policy moments once, target copy. ~2 ms.
3. 128 critic steps x 1,024 rows: CE every step (~33 ms); roll + reward on every 4th step with the reward
   head on two latents (~35 ms FP32, ~15 ms BF16).
4. 32 actor steps x 2,048 rows, one action per state (16 ms), gathered old moments.
5. 4 refreshes, K=2 (8 ms).
6. Diagnostics compiled and on every 8th iteration (~3 ms amortized).

Update ~0.09 s FP32, ~0.07 s with BF16 on the aux path [E] -> iteration ~0.28 s, ~58k SPS solo, ~200k
aggregate at six runs. **Irreducible remainder:** physics (0.13 s, 46% of the iteration), then the critic's
CE sample-gradients: ~3.4 ms per pass over 16k rows at 512 wide in FP32 [D], which no engineering removes -
only fewer passes, a narrower critic, or lower precision. Everything else is below 10 ms.

Ranked savings against v60's 0.464 s of GPU per iteration:

| # | Change | Saves (GPU s / iteration) | Evidence | Learning risk |
|---|---|---:|---|---|
| 1 | Aux roll + reward amortized 4x | ~0.22 | [D] profile + v29 artifact | untested |
| 2 | 1,024 rows per critic step | ~0.06 after #1 (0.17 alone) | [E] GEMM scaling | untested (steps vs gradients) |
| 3 | BF16 on the aux path only | ~0.02-0.04 after #1 (0.19 alone) | [M] BF16 kernel times | hypothesis on TF32 mechanism |
| 4 | Diagnostics every 8th iteration, compiled | ~0.025 | [M] | none |
| 5 | Refresh K=2, every 32 steps | ~0.014 | [M] 22 ms region | low (v61 Rao-Blackwell) |
| - | 256-wide actor (CPU, not GPU) | ~0.085 s wall | [M] 0.129 vs 0.042 | untested |

---

## 5. Measurement plan (to run later, one job at a time, through `mlq`)

Nothing below has been submitted. All arms: HalfCheetah-v4, seed 1, strict FP32, the v60 script, and
`--max-parallel-runs 1` so timings are solo. Timing arms are 1M steps with a kernel profile at iteration 40
(the precedent is the 800k `speed_v36_*` ladder); learning arms are the standard 8M. Keep
`scripts/autocull.py --ref reppo_noisefork_v60_base --only rc06_ --enforce all --yes --watch 600` running
while learning arms are in flight. Read results with `compute_mine_runs.py` + `compute_tables.py timing|budget`
and `compute_profile_shares.py`.

Common prefix:

```bash
S=cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py
submit() {  # submit NAME STEPS [extra flags...]
  local name=$1 steps=$2; shift 2
  mlq submit --name "$name" --max-parallel-runs 1 --cwd "$PWD" \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env CLEANRL_ENV_SPIN=5000 -- \
    .venv/bin/python -u "$S" --env-id HalfCheetah-v4 --num-envs 16 --exp-name "$name" \
    --total-timesteps "$steps" --seed 1 --env-threads 2 --compile --compile-mode reduce-overhead "$@"
}
```

### 5.1 Validate the cost model (timing arms; each prediction is falsifiable)

```bash
submit rc06_t0_base        1000000 --profile-iteration 40
submit rc06_t1_h1          1000000 --profile-iteration 40 --aux-horizon 1
submit rc06_t2_h4          1000000 --profile-iteration 40 --aux-horizon 4
submit rc06_t3_cu64        1000000 --profile-iteration 40 --critic-updates 64 --target-refresh 8
submit rc06_t4_mb1024      1000000 --profile-iteration 40 --minibatch-size 1024
submit rc06_t5_mb512       1000000 --profile-iteration 40 --minibatch-size 512
submit rc06_t6_refresh     1000000 --profile-iteration 40 --bootstrap-samples 1 --target-refresh 32
submit rc06_t7_actor256    1000000 --profile-iteration 40 --actor-hidden-dim 256
```

| Arm | Predicted `timing/update_s` [E] | What it validates |
|---|---:|---|
| t0 base | 0.43-0.44 | reproduces v60 solo |
| t1 H=1 | 0.17-0.19 | roll + reward = ~69% (7/8 of it removed) |
| t2 H=4 | 0.28-0.30 | roll cost is linear in H |
| t3 64 critic steps | 0.23-0.26 | critic steps are ~85% of the update |
| t4 1,024 rows | 0.26-0.29 | GEMM time ~0.6x per halving of rows |
| t5 512 rows | 0.17-0.21 | approach to the ~10 us launch floor |
| t6 K=1, 4 refreshes | 0.415-0.42 | refresh = 22 ms |
| t7 256-wide actor | update ~0.43; `timing/rollout_s` 0.04-0.05 | host actor is arithmetic-bound |

The model is wrong where a prediction misses by more than ~15%; t1 and t4 are the two that the skeleton's
budget depends on.

One concurrency check of the aggregate-ceiling claim (the only arm that is not solo; submit three copies with
`--max-parallel-runs 3` when the box is otherwise idle): three `rc06_t0_base`-equivalent runs should each
report update ~1.2-1.4 s and sum to ~33-36k SPS.

### 5.2 Top three savings, as learning arms (8M each)

Existing flags cover saving #2 and the actor width directly, and bracket saving #1:

```bash
# Saving 2: steps vs sample-gradients (same 128 steps, fewer rows) and the converse (same gradients, fewer steps)
submit rc06_mb1024         8000000 --minibatch-size 1024
submit rc06_mb512          8000000 --minibatch-size 512
submit rc06_cu64_mb4096    8000000 --critic-updates 64 --target-refresh 8 --minibatch-size 4096
# Saving 1, bracketed with existing flags: half and one-eighth of the roll
submit rc06_h4             8000000 --aux-horizon 4
submit rc06_h1             8000000 --aux-horizon 1
# Rollout actor
submit rc06_actor256       8000000 --actor-hidden-dim 256
# Precision, global (the only form the flags allow); BF16 has never been run past 800k
submit rc06_bf16           8000000 --bf16 --matmul-precision high
```

Note `--minibatch-size` also changes the actor's rows in v60; `rc06_mb1024` therefore halves the actor's
sample-gradients too, which section 2.1 says is a weak lever. Decision rules against
`reppo_noisefork_v60_base` (4267 / 9084 / 11069 at 1M / 4M / 8M): an arm is "free" if it is within 0.5k at 4M
and 8M and its `diagnostics/advantage_corr_heldout` is not lower; anything that differs by less than ~1k needs
a second seed before it changes the design, because the family's flip mode is larger than that.

Savings #1 and #3 in their real form need one new version (not written here; proposed name
`ppo_continuous_action_reppo_leanupdate_v64.py`, a copy of v61 or of whichever critic fix wins) with:
`--aux-every N` (roll on every Nth critic step), `--reward-latents K` (reward head on z_0 plus K-1 random
later latents), zero aux coefficients skipping the compute, `--aux-bf16` (autocast only around the roll and
reward head), `--diagnostic-interval N`, observations and old-policy moments computed once per iteration.
Its arms, with the same `submit` prefix and `S` pointed at the new file:

```bash
submit rc06_lean_aux4      8000000 --aux-every 4 --reward-latents 2 --diagnostic-interval 8
submit rc06_lean_aux4_bf16 8000000 --aux-every 4 --reward-latents 2 --diagnostic-interval 8 --aux-bf16
submit rc06_lean_full      8000000 --aux-every 4 --reward-latents 2 --diagnostic-interval 8 --aux-bf16 \
                                   --minibatch-size 1024 --actor-hidden-dim 256 --bootstrap-samples 2 --target-refresh 32
```

`rc06_lean_full` is target A: predicted update ~0.07 s, ~58k SPS solo.

### 5.3 Two offline measurements on existing checkpoints (one short GPU job each, via `mlq`)

`runs/HalfCheetah-v4__reppo_noisefork_v60_base__1__*/checkpoint_{1015424,4013696,7913088}.pt` hold full
training state. Scripts to be written under `benchmarks/reppo_rootcause/scripts/`:

1. **Actor gradient variance decomposition.** With critic and actor frozen, draw S states x K actions, compute
   the per-sample actor gradient of -Q, and split its variance into between-state and within-state parts.
   Decides 4.6: if the within-state share is large, K actions per state (cheap) beats more states.
2. **Precision sensitivity by path.** On one fixed minibatch, compare dQ/da and the CE gradient in FP32 against
   TF32 and BF16 applied to (a) the whole critic, (b) everything except the first encoder layer and the value
   head, (c) the roll only. Report cosine and relative norm error of dQ/da. Decides 4.7 before spending an 8M
   run on a split-precision arm.

---

## 6. What this means for the rest of the study

- The critic investigators should assume a budget of **~8-16 CE passes and at most a quarter of today's aux
  compute**; fixes that need more optimization per rollout than v60 are fighting the efficiency goal, and
  the archive says more optimization does not move the held-out metric anyway.
- The only archive lever that moved held-out critic quality is more fresh data per fit (2.3). That is
  compute-cheap on the GPU (per-sample cost is flat) but costs env steps; it argues for critic fixes that
  pool information across rollouts on-policy (exact-target style) rather than more passes.
- Any proposed fix should be reported with its GPU seconds per iteration next to its return, because on this
  box that number, not solo SPS, sets how many experiments per hour the study can run.
