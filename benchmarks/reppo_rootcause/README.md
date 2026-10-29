# REPPO root cause: synthesis and plan

Twelve sub-reports (01-10 investigations, 12-13 reviews of the draft of this file) are in this folder;
scripts in `scripts/`, outputs in `data/`. HalfCheetah-v4, 16 envs x 1024 steps, seed 1. Nothing was
trained for this study: every number is recomputed from the existing archive (tfevents, oracle arrays,
snapshots) or read from code. Confidence figures are judgement, after two rounds of red-teaming.

## 1. What was wrong with the investigation itself

| Problem | Evidence | Report |
|---|---|---|
| The steering metric `advantage_corr_heldout` has a ceiling of ~0.3, not 1, and is anti-predictive of return across runs (Spearman -0.26 to -0.34). | Its reference is a return advantage in which the row's own action carries ~1/8.7 of the variance. | 01, 02, 03 |
| The 1M oracle's headline numbers are pooled sums owned by a heavy tail. | One state holds 86-88% of the critic's within-state energy. "Self-consistency -0.33" is +0.4 to +0.6 robustly; "critic sd = true sd" is 0.3x on the typical state. | 02, 03, 07 |
| Trainers are bit-deterministic, so "repeats" were one trajectory. | Only two independent upright reference trajectories exist. Single-run minimum detectable effect: ~0.4k at 1-2M, ~1.2k at 8M. | 01 |
| About a fifth of tanh-Gaussian trajectories end trapped on their back (27/123), more recover late. Verdicts on v32, v33, v35, v37, v50 and others are flip outcomes. | Census of 135 from-scratch trajectories. | 09 |
| Several "refuted" ideas were never tested. | v61/v62 exact targets removed ~0-2% of target noise by construction; v40's hybrid arm fed unnormalised advantages; v63's slope accuracy was never measured; replay arms were culled early or flipped. | 01, 03, 08, 10 |

Seed-averaged, the reference is plausibly ~7-9k at 8M rather than 11.0k, i.e. not clearly above sepclip
PPO (8.4k). No PPO run was ever measured under a replicate protocol either.

## 2. Diagnosis

### 2.1 The code is correct (confidence ~85%)
Target recursion (matches brute force to 1.8e-15), truncation bootstrap, HL-Gauss encode/decode (round
trip 4e-5, no clipping, decode slope within 0.3% of 1), FP32 precision (contrast error 5e-6), action path,
loss signs, KL gate, duals, update order: all cleared (02). HL-Gauss bin width is not a resolution floor
(04; v14's scalar head was neutral). No published component is missing (05).

### 2.2 The critic's action signal: information-poor AND badly conditioned (confidence ~55%)
- **Information-poor (solid).** A lambda-return sample carries 5-7% action-advantage signal (03, 07). Per
  state, the critic (corr 0.15-0.29 with truth) is at the ceiling an ideal estimator pooling 10-30
  iterations could reach (0.11-0.26; 07). Only more data per policy step ever moved critic quality, and
  it cost return (03). The published method uses 8x more data per evaluation and its paper reports that
  performance "drops steeply with decreasing dataset size" (05).
- **Not worse than PPO's estimator (weak).** Projected into policy-parameter space, critic gradient and
  GAE score-function gradient are statistically indistinguishable per rollout at 4-8M (critic gain 0.59
  [0.29, 0.87] vs 0.43 [-0.11, 0.79] at 4M); the critic leans worse at 30M. The snapshots are v23/v29
  recipes, not v60 (07, 12).
- **Badly conditioned (new, the leading critic-side hypothesis).** The top 1% of states hold 69-88% of
  squared action-gradient energy. After 1M these are ordinary upright gait states, not tumbles. Per-state
  gradient DIRECTION is stable between consecutive critics (cosine 0.88-0.93) but the spike AMPLITUDES
  move 3-6x per iteration, so the energy-weighted field is rewritten by 56-135% each iteration. Contrasts
  are over-dispersed 2-3.5x outside 1M (calibration slope 0.28-0.68). The actor consumes this as an
  unnormalised mean under a 0.5 grad-norm clip (pre-clip 2.5-4), and spends its KL budget mostly off each
  state's own pathwise direction (median cosine 0.14-0.28) (07, 12).
- **Untapped channel.** With true dynamics, the critic's own V' ranks actions far better than Q does
  (expected-label corr 0.45 at 1 real step, 0.90 at 5; the advantage is complete within ~10 steps). Honest
  information gain per env step from real-simulator branches: 0-4x at n=1, 0.4-1.3x at n=5, from the 1M
  oracle only (03, 07, 12). Learned models do not deliver it (v30/v37/v38).

### 2.3 Early tumbling decides the flip lottery (confidence ~65%)
Even upright runs spend 8-17% of their first 1M steps inverted. At 0.5M the mean policy runs upright and
its own noise tips it within 49-268 steps in 6/6 episodes. The tanh head starts with 17% of actions beyond
|a| 0.9; bounded heads start at 2.6% and were trapped 0/12 times (p 0.12, confounded: those runs are
quieter throughout and all end at 7.0-9.4k). The branch is taken around iterations 5-10 (09, 12).

### 2.4 The late plateau is NOT explained (confidence that tumbling causes it: ~30%)
The deterministic slope falls from 1.2k/M to 0.3k/M between 4M and 7.5M. Crash share is 5.6-7.3% at
5-8M and costs 250-440 return directly; it did not change while the slope collapsed (12). The change is
tied to return (~7k), not step (09). Candidates still standing: the conditioning problem in 2.2; the
entropy dual reading crashed rows (sigma 0.86-1.08 on inverted rows vs 0.47-0.50 upright); pitch
observation statistics set by tumbles (cumulative sd 0.88-1.51 vs a gait sd of ~0.2, lane ratio
0.90-1.55); a genuine information ceiling. The plateau premise itself is recipe-specific: v25 (lambda
0.8 with in-iteration refresh) reaches 13.3k stochastic / 15.0k deterministic at 50M.

### 2.5 Where REPPO's edge over PPO comes from (confidence ~55%)
At matched KL per iteration REPPO and sepclip gain the same return per iteration over 0.5-2M. The lead is
a head start before 0.5M that belongs to the pathwise actor (absent when a score-function actor uses the
same critic), plus a higher 2-6M rate that exists only while the entropy dual holds sigma at 0.4-0.5;
entropy-free variants stall at PPO's ~7k (08). Largest single removals at 8M: dueling critic -6k, coupled
minibatches -4.1k, entropy dual -1.8 to -4k, critic passes 16 -> 4 -2k, aux loss -1.3k.

### 2.6 Compute (confidence ~80% on shares, measured from existing profiles)
v60: 0.436 s update per iteration vs PPO 0.012 s: 36x GPU, 3.8x wall solo (21.5k vs 79-85k SPS), ~7x
aggregate. ~69% of the update is the auxiliary loss (H8 latent roll ~51%, reward head ~18%); critic CE
~13%, actor ~4%. Arithmetic-bound, so only removing work helps. Whether the H8 roll earns its cost is
unresolved (+1.7k per 06, inside noise per 08).

## 3. Plan

Rules: new versioned files only; every job through `mlq`; HalfCheetah 8M. Two items need a user decision
before anything runs (section 4).

### Wave 0 - instruments and gates
`reppo_instrument_v64`, learning-identical to v60, plus an unmodified v60 run alongside as the parity
pair (`shared/gae.py` and `ppo_loop.py` have uncommitted edits newer than the v60 base run, so v60 itself
may no longer reproduce 11013.6). Instruments must draw no RNG and remove no existing draw.
- Posture: inverted share via cos(pitch) < 0 (obs[1] is unwrapped pitch), tumbles per episode.
- Row-split metrics (upright vs inverted): q_mse, std, entropy, share of squared actor gradient.
- Gradient conditioning: top-1% energy share, per-state cosine and amplitude ratio against the previous
  critic (slope persistence), cosine of each state's move with its own pathwise direction.
- Realised gain: backward-form score-function estimate of the last update's first-order gain on the next
  rollout, against the prediction of an explicit pre-update critic copy. Usable SNR through ~3M (08).
- Multi-episode deterministic eval at fixed steps.
- Oracle script: per-state medians, trimmed statistics, top-state share, saved states, n = 1..10 reward
  partial sums.
CPU-only gates on existing data: actor-gradient cosine with and without the top 1% of states; the hybrid
gradient gate (bet B1); pitch-statistics comparison across lanes.

### Wave 1 - honest baselines, tumbling, lean compute (16 streams per arm)
| Arm | Purpose | Primary endpoint | Kill / pass |
|---|---|---|---|
| v64 control census (never culled) | Real seed-averaged baseline and flip rate | inverted share 0-0.5M; 8M mean | - |
| PPO-lineage reference, same protocol | "Beats PPO" measured, not assumed | 8M mean | - |
| Uniform sigma 0.55 at init with last-layer gain 0.01 | Is the flip caused by initial noise shape? | inverted share 0-0.5M | pass: share < 5% and 8M non-inferior |
| lambda 0.95 for the first 0.5M; `--kl-start 1.0` | Zero-code alternatives | same | same |
| lean-safe / lean-full (aux roll every 4th critic step, reward head on 2 latents; plus minibatch 1024, 256-wide actor, BF16 on aux only) | Make every later experiment 2-3x cheaper | 8M mean vs census; update s/iter | pass: within 1.0k at n=16 and update <= 0.15 s (safe) / 0.07 s (full) |
| Hopper-v4, Walker2d-v4 reference (4 each) | Does any of this transfer where falls terminate? | return | - |

Binary flip claims need 32 runs; 16 v 16 has power 0.27 for 0.22 -> 0.

### Wave 2 - conditioning, noise, step (forks: 4 parent checkpoints x 2 streams per arm, plus a noise-floor fork pair)
| Arm | Hypothesis | Prediction | Falsifier |
|---|---|---|---|
| Per-state p99 clip of dQ/da; per-state unit-norm pathwise step | 2.2 conditioning | realised gain per unit KL up; 8M +1k | gain unchanged => spikes are harmless |
| EMA critic for the actor (horizon 4, 8) and a frozen-critic arm | amplitude instability vs lag vs co-adaptation | EMA > frozen > live in realised/predicted ratio | all equal |
| Truncation-on-inversion (diagnostic only) | does the late plateau need tumbles at all? | deterministic slope after 4M unchanged (12 predicts this) | slope doubles => 2.4 is tumbling after all |
| Entropy dual and obs statistics on upright rows only; wrapped pitch | dual and normaliser contamination | sigma stops rising; q_mse < 1 | no change |
| KL bound read off the realised/predicted ratio; fixed 0.06 and 0.1 over 1-3M | early step ~5x too timid | +0.8-2k at 2-3M | gain lost by 8M |
Every fork positive is re-run from scratch before it is believed.

### Wave 3 - bets, each behind a cheap gate
- **B1 hybrid actor**: unit-norm pathwise + score-function gradient. 07 implies cosine with the true
  projected gradient 0.65 -> 0.77. Gate: CPU-only on existing data.
- **Vine labels**: first a budget-free upper bound (branches not charged) to see whether accurate
  contrasts lift the plateau at all; only then the budgeted version (antithetic K = 2, real action as one
  branch, every branch step charged, compared against the full-rollout reference). Needs simulator state
  restore; report separately from the PPO-comparable result.
- **B2 mixed-noise envs with a noise-conditioned critic**: replaces the entropy constant with an
  explicit spread of exploration scales.
- **B3 late-fusion critic with K actions per state**: compute lever; return effect unknown.

### Wave 4 - integrate
One file with the winners; 16 fresh HalfCheetah streams, 8 each on Hopper and Walker2d, PPO alongside.

Cost: ~270-300 runs, 12-16 box-hours on the reference recipe, about a third of that once a lean base
passes.

## 4. Decisions needed from the user
1. **Replicates.** Everything above needs independent trajectories. Proposal: `--rng-stream k` perturbs
   only the host action-noise stream (bit-identical at k = 0; seed 1 still sets init, resets, phases).
   It is a seed in effect and conflicts with "always seed=1" / "don't do multiple seeds".
2. **Vine.** It is on-policy but requires restoring simulator state, which PPO does not.

## 5. Not done
- No experiment was run. The box queue is occupied; fw16 was at load ~72 with no repo, venv or queue
  daemon.
- A reviewer wrote `scripts/13_random_posture.py` (random-action inversion check) and its `mlq`
  submission was refused by the permission system; it is unexecuted.
- v64 is not written.
