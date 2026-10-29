# 14 — CPU-only decision gates: hybrid actor direction, gradient spikes, observation normalisation

CPU only, no training, no `mlq`. Scripts `scripts/14_*.py`, logs `data/14_*`. HalfCheetah-v4, seed 1, one
snapshot per stage. Units are the trainers' scaled returns (reward x 0.1). CIs are 95% bootstraps (2000 draws).
Simulation used: 15 stochastic episodes x 1000 steps in total (5 per v60 checkpoint, `14_states.py`, 21 s single
thread), because no states are stored with the v60 checkpoints.

| Gate | Verdict | One line |
|---|---|---|
| 1. Hybrid actor direction (B1) | **FAIL** | Unit-norm critic + score-function sum beats the critic alone with P 0.44-0.48 at 4M and 0.04-0.13 at 8M; the held-out optimal mix with P 0.23-0.27 and 0.01-0.02. Per-state normalisation P 0.29-0.50 / 0.05-0.29; p99 clip P 0.37-0.55 / 0.49-0.66. Nothing reaches 0.8 at both stages. |
| 2. Gradient spikes | **Real, located, and not what the clip is binding on** | v60 has no spikes at 1M (top-1% energy 0.21), strong ones at 4M and 7.9M (0.79, 0.84). A spike is the action-sensitivity of a low second mode of the HL-Gauss value distribution (94% of the spike gradient at 7.9M). The pre-clip norm of 2.6 logged at 7.9M is the KL pull-back term (2.53), not dQ/da. |
| 3. Observation normalisation | **CONFIRMED and larger than stated** | Statistics are per lane, networks are shared. Gait pitch occupies 0.19-0.31 normalised sd (4-5x compressed). The same physical state read through two lanes gives policies a median KL of 0.06-0.15 apart (bound 0.03) and Q values a median 0.11-0.68 apart (median within-state action sd of Q: 0.05-0.12). |

---

## Gate 1 — hybrid actor direction (`14_gate1_hybrid.py`, `14_gate1_table.py`, `14_gate1b_v60.py`)

### Method

07 D3's framework unchanged: field = estimate of the gradient of E_z Q_soft(s, tanh(mu + sigma z)) with respect to
the pre-tanh mean; aggregates in linear (108), actor-last-layer (3078) and full actor-parameter (279k) spaces; truth
from the oracle (O) and the near-Monte-Carlo score function (S, GAE 0.99; 0.95 as the second column), every product
with the truth between different env lanes; bootstrap over chunk blocks and oracle units. 07's critic and
score-function gains reproduce to the digit (0.5924 / 0.4277 at 4M, full parameters).

Estimators: (a) critic dQ/da (1 - a^2); (b) GAE(0.95) score function; (c) unit-norm sum of the two realised
aggregates, plus the same with the norms a 2048-row minibatch would have (B1 normalises per actor step) and with
GAE(0.8); (d) non-negative weights maximising predicted cosine, fitted on 8 lanes and scored on the other 8 against
the other 8 lanes' truth only, averaged over 8 random splits x 2 directions and compared with the critic on the
same halves (an in-sample version is given as an optimistic bound); (e) each state's gradient rescaled to the batch
rms norm, in pre-tanh and in action coordinates; (f) per-state norms clipped at p99, both coordinate systems; plus
top-1% removed. Controls: critic rows permuted, score-function rows permuted.

One correction to 07's bootstrap. A resampled block set carries resampling noise on top of sampling noise, which
doubles the same-block part of |V_est|^2 in every draw and so deflates the bootstrap gain of noisy estimators (the
score function's noise share is 45-100%, the critic's 1-12%). I replace v_i^2 by v_i on the diagonal; the uncorrected
numbers are in the JSON under `naive-norm`. The correction moves P(score function > critic) at 4M from 0.13 to 0.23
and no verdict.

### Result, full actor-parameter space (first number reference GAE 0.99, second 0.95; P = P(better than critic alone))

| estimator | 4M v23 gain | P | 8M v29-H8 gain | P | 12M v23 P | 12M v29 P | 30M v23 gain | P |
|---|---|---|---|---|---|---|---|---|
| (a) critic | 0.592 [0.30, 0.89] / 0.562 [0.33, 0.80] | - | 1.173 [0.04, 2.34] / 1.841 [0.90, 2.87] | - | - | - | 0.085 [-0.90, 1.07] / 0.314 | - |
| (b) score function GAE 0.95 | 0.428 [-0.14, 0.96] / 0.431 | 0.23 / 0.24 | 0.669 [-0.51, 1.92] / 1.045 | 0.07 / 0.01 | 0.43 / 0.22 | 0.04 / 0.05 | 0.533 / 0.586 | 0.72 / 0.63 |
| (c) unit-norm sum | 0.585 [0.17, 0.97] / 0.570 | **0.44 / 0.48** | 0.979 [-0.14, 2.20] / 1.535 | **0.13 / 0.04** | 0.47 / 0.32 | 0.05 / 0.09 | 0.385 / 0.561 | 0.77 / 0.73 |
| (c) unit-norm at minibatch scale | 0.607 / 0.585 | 0.51 / 0.56 | 1.067 / 1.673 | 0.15 / 0.06 | 0.48 / 0.36 | 0.07 / 0.12 | 0.284 / 0.490 | 0.78 / 0.76 |
| (c) unit-norm sum with GAE 0.8 | 0.658 / 0.630 | 0.67 / 0.68 | 1.020 / 1.607 | 0.11 / 0.04 | 0.39 / 0.30 | 0.07 / 0.13 | 0.064 / 0.247 | 0.44 / 0.36 |
| (d) optimal mix, held-out lanes | 0.529 (critic on the same halves 0.547) | **0.23 / 0.27** | 1.196 | **0.02 / 0.01** | 0.20 / 0.09 | 0.06 / 0.11 | 0.206 / 0.388 | 0.52 / 0.46 |
| (d) optimal mix, in-sample (optimistic) | 0.609 / 0.585 | 0.63 / 0.67 | 1.173 (weight on SF = 0) | 0.19 / 0.11 | 0.52 / 0.43 | 0.12 / 0.18 | 0.533 / 0.606 | 0.83 / 0.82 |
| (e) unit per state, pre-tanh | 0.553 / 0.567 | 0.29 / 0.50 | 1.045 / 1.549 | 0.29 / 0.05 | 0.75 / 0.75 | 0.29 / 0.37 | 0.619 / 0.613 | 0.88 / 0.77 |
| (e) unit per state, action coords | 0.429 / 0.455 | 0.05 / 0.07 | 1.013 / 1.572 | 0.19 / 0.03 | 0.80 / 0.79 | 0.26 / 0.30 | 0.462 / 0.464 | 0.78 / 0.65 |
| (f) clip p99, pre-tanh | 0.575 / 0.570 | 0.37 / 0.53 | 1.179 / 1.842 | 0.56 / 0.49 | 0.41 / 0.51 | 0.29 / 0.43 | 0.150 / 0.366 | 0.74 / 0.73 |
| (f) clip p99, action coords | 0.577 / 0.573 | 0.37 / 0.55 | 1.186 / 1.850 | 0.66 / 0.62 | 0.38 / 0.45 | 0.32 / 0.42 | 0.160 / 0.373 | 0.82 / 0.80 |
| top 1% of states removed | 0.577 / 0.575 | 0.42 / 0.55 | 1.152 / 1.812 | 0.38 / 0.27 | 0.54 / 0.61 | 0.24 / 0.29 | 0.159 / 0.372 | 0.64 / 0.63 |
| control: critic rows permuted | 0.101 [-0.32, 0.53] | 0.02 | 0.479 [-0.50, 1.54] | 0.01 | | | | |
| control: SF rows permuted | 0.192 [-0.31, 0.65] | 0.07 | -0.239 | 0.03 | | | | |

Cosines with the truth are usable only at 4M (|V*|^2 > 0 in 0.80-0.97 of draws): critic 0.93 [0.36, >1],
score function 0.67 [0.13, >1], unit-norm sum 0.91 [0.48, >1], held-out mix not above the critic. At 8M the truth
norm is unresolved (P 0.45-0.54) and cosines exceed 1; the gains above do not depend on it.

Other spaces. Last layer: the same pattern (4M unit-sum P 0.51, clip 0.42-0.61; 8M unit-sum 0.07-0.16, clip
0.42-0.60). Linear-in-observation space is the one place a re-weighting clears 0.8 at 4M: p99 clip 0.212 vs 0.178, P
0.88-0.92, top 1% removed 0.89; at 8M the same estimators are at 0.40-0.60. The linear space is not the space the
actor steps in.

### Why it fails: the error structure

| stage (full parameters) | noise share of \|V\|^2, critic | noise share, score function | correlation of their same-chunk fluctuations (rho_noise) | cosine between their EXPECTED aggregates | realised cosine | error correlation given the truth (rho_error) |
|---|---:|---:|---|---|---:|---|
| 4M v23 | 0.04 | 0.66 | -0.00 [-0.38, 0.21] | 0.91 [0.45, >1] | 0.52 | -0.35 [-2.0, 0.93] / +0.18 [-0.29, 0.83] |
| 8M v29 | 0.01 | 0.54 | +0.15 [-0.51, >1] | 1.12 [0.88, >1] | 0.77 | undefined (truth norm unresolved) |
| 12M v23 | 0.09 | 0.95 | +0.05 [-0.10, 0.18] | 1.04 [0.12, >1] | 0.24 | +0.23 [-0.04, 0.63] |
| 12M v29 | 0.05 | 1.01 | -0.05 [-0.34, 0.27] | unresolved | 0.34 | undefined |
| 30M v23 | 0.09 | 0.78 | +0.01 [-0.13, 0.15] | 0.64 [-0.09, >1] | 0.29 | +0.29 [-0.67, 0.87] |

- The two estimators' sampling fluctuations are uncorrelated (rho_noise within +-0.15 of zero everywhere, needs no
  truth). The independence that B1 assumed holds, for the noise.
- But B1's arithmetic assumed two estimators at cosine 0.65 with independent errors. They are not symmetric. The
  critic's aggregate is 91-99% expectation: its error is bias, almost none of it noise. The score function's aggregate
  is 45-100% noise. And the critic's expected direction is nearly collinear with the score function's expected
  direction (0.91 at 4M, ~1 at 8M), which is the truth up to GAE bias. So at 4-8M there is little orthogonal bias for an
  unbiased partner to cancel, and the partner brings its own noise at equal weight. With the measured cosines (0.93
  and 0.67) the independent-error formula itself predicts 0.89 for the unit-norm sum, below the critic alone;
  measured 0.91.
- The held-out optimal mix puts 24% [5%, 54%] of the unit-scaled weight on the score function at 4M (the in-sample
  optimum puts none on it at 8M) and does not beat the critic out of sample.
- At 30M the picture reverses in sign but not in confidence: the critic's expected direction has drifted (0.64),
  the mix and the per-state normalisation lean better (P 0.73-0.88), consistent with 07's finding that the score
  function is the better of the two there.
- Limitation that cuts against the bet, not for it: the snapshot critic was not fit on the chunk it is scored on.
  The trainer's actor reads a critic fitted for 128 steps on the same rows, which is where "the critic has absorbed
  the same return noise" would appear. That correlation cannot be measured from these files; if present it lowers
  the value of the combination further.

### v60 recipe (`14_gate1b_v60.py`): chunk-only arm on the v60 checkpoints' own states

No oracle exists for v60, so the only truth reference is the cross-episode score function from five episodes
(4995 rows; |V*|^2 > 0 in 0.35-0.62 of draws: weak). Full-parameter space:

| | 1M | 4M | 7.9M |
|---|---|---|---|
| critic gain | 0.96 [0.27, 1.68] | 1.84 [-0.31, 4.84] | 1.26 [-0.45, 2.93] |
| unit-norm sum, P(better) | 0.08 | 0.52 | 0.25 |
| unit per state (pre), P(better) | 0.01 | 0.73 | 0.76 |
| clip p99 (pre), gain, P(better) | 0.92, 0.10 | 2.39, 0.80 | 1.88, 0.87 |
| top 1% removed, gain, P(better) | 0.78, 0.06 | 3.13, 0.83 | 2.26, 0.91 |
| critic noise share (5k rows) | 0.07 | 0.16 | 0.29 |
| rho_noise | 0.01 [-0.81, 0.33] | -0.02 [-0.49, 0.12] | 0.19 [-0.12, 0.39] |

### Verdict

**FAIL.** Neither the combination (c, d) nor the normalisation (e, f) is better than the critic alone with
P >= 0.8 at both 4M and 8M, in the space the actor steps in or in the last-layer space. The 0.65 -> 0.77 estimate
rested on a symmetry that the data do not show.

Would it hold for v60? For the hybrid, very likely yes (my estimate ~80%): the failure is structural (critic error
is bias aligned with the unbiased direction, score-function error is noise at 16k rows), the v60 chunk-only arm gives
P 0.08 / 0.52 / 0.25, and the in-training critic is fitted on the chunk, which can only add correlation. For the
p99 clip the answer is different and should be kept separate: on v60's own states clipping or removing the top 1%
leans better at 4M and 7.9M (P 0.80-0.91, intervals include zero, weak reference, several estimators tried), which
agrees with Gate 2: v60's spikes disturb the parameter gradient much more at 7.9M than the v23 / v29 snapshots'
do. That is an indication to keep the Wave-2 p99-clip fork arm, not a pass of this gate. Drop B1.

---

## Gate 2 — gradient spikes on the v60 checkpoints (`14_states.py`, `14_gate2_spikes.py`, `14_gate2b_mechanism.py`, `14_gate2c_end_of_iteration.py`)

States: 5 stochastic episodes per checkpoint, each under a different lane's statistics (lanes 0, 4, 7, 11, 14).
Returns 4.1-4.5k (1M), 9.0-9.5k plus one crashed episode at 1.9k (4M; 15% of rows inverted), 10.1-11.8k (7.9M).
dQ/da at the sampled action; the trainer draws fresh noise per actor step, statistically the same thing.

### Distribution

| | 1M | 4M | 7.9M |
|---|---:|---:|---:|
| top-1% share of squared norm, pre-tanh dQ/da (1 - a^2) | 0.21 | 0.79 | 0.84 |
| same, post-tanh dQ/da | 0.15 | 0.73 | 0.76 |
| same, upright rows only | 0.21 | 0.78 | 0.84 |
| top-5% share, pre-tanh | 0.45 | 0.91 | 0.93 |
| p50 / p99 / max of the pre-tanh norm | 0.17 / 0.95 / 2.6 | 0.28 / 4.2 / 63 | 0.33 / 5.4 / 76 |
| excess kurtosis of the norm (pre-tanh) | 20 | 1170 | 679 |
| excess kurtosis of pooled components, pre / post | 33 / 86 | 4270 / 1269 | 3092 / 2010 |
| share of var(log norm) belonging to the state, not the sampled action (8 fresh actions) | 0.73 | 0.78 | 0.65 |
| lag-1 autocorrelation of log norm; run length of top-1% membership | 0.18; 1 step | 0.51; 1 step | 0.22; 1 step |

v60 at 1M is not spiky. 12's "0.88 at 1.02M" was a different run (v55 pairs, three episodes); the concentration
there is not a property of this checkpoint. At 4M and 7.9M it reproduces.

### What the spike states are (top 1% by pre-tanh norm, 50 states; median unless noted)

| | 4M top-1% / rest | 7.9M top-1% / rest |
|---|---|---|
| cos(pitch) | 0.992 / 0.979 | 0.988 / 0.980 |
| inverted | 1 of 50 (15% of rows are) | 0 |
| x velocity | 9.97 / 9.55 | 12.4 / 12.2 |
| max_d \|tanh(mean)\| | 0.974 / 0.937 | 0.961 / 0.950 |
| dims with \|a\| > 0.99 (mean) | 0.30 / 0.15 | 0.16 / 0.19 |
| policy std | 0.37 / 0.47 (Spearman with log norm -0.60) | 0.43 / 0.49 (-0.45) |
| Mahalanobis distance of normalised obs | 4.1 / 4.0 | 3.9 / 3.9 |
| Mahalanobis of the critic input (s, a); 10-NN distance | 4.5 / 4.7; 2.1 / 2.3 | 4.5 / 4.6; 2.3 / 2.1 |
| Q | 85.6 / 88.4 | 103.4 / 108.4 |
| sd of the HL-Gauss distribution | 3.4 / 1.8 | **6.4 / 1.8** |
| feet on floor (mean); other geom on floor | 0.04 / 0.09; 0 / 0.07 | 0.04 / 0.08; 0 / 0 |
| gait phase: resultant length of spike phases (all rows 0.02) | 0.15; octant 0 holds 59% of the energy on 18% of rows | 0.39; octants 3-4 hold 60% on 43% of rows |

Spikes are upright, fast, mid-flight gait states that are in the bulk of the data (not outliers in observation or
critic-input space), where the policy is more confident than usual, and where the critic's value distribution is
2-3.5x wider than its floor (label kernel sd 1.5, bin 2.0, floor ~1.7). They last one step, are a property of the
state rather than of the sampled action, and prefer part of the gait cycle without being confined to it.

### Dimensions and tanh coordinates

- A spike is one coordinate: the largest coordinate holds 0.94 (4M) and 0.89 (7.9M) of the spike's pre-tanh energy
  (0.64-0.66 for ordinary states); participation ratio 1.1-1.2 of 6. It is the back shin (44-54% of spikes) or the
  front thigh (30-50%); those two coordinates hold 89-91% of all pre-tanh energy at 4M-7.9M.
- The spike is a property of dQ/da itself, in both coordinate systems. Top-1% sets overlap 34-37 of 50; Spearman of
  the two norms 0.79-0.81. The tanh Jacobian does not create it and does not remove it.
- Saturation is not the mechanism. dQ/da is somewhat larger on saturated components (median 0.53-0.56 at |a| > 0.99
  against 0.10-0.14 at |a| < 0.5; Spearman of |dQ/da_d| with |a_d| +0.26 to +0.29), i.e. the post-tanh gradient is
  largest where the action cannot move, but those components are 2.6-3.1% of the components and 5-7% of the
  post-tanh energy, and after the Jacobian they carry 0.0% of what reaches the actor. Components with |a| > 0.9 are
  25-29% of all components and receive 0.5-0.9% of the pre-tanh energy: a quarter of the action coordinates are
  effectively frozen for the pathwise term, which is a separate cost from the spikes. 60-71% of the pre-tanh energy
  comes from components with |a| < 0.5.
- The spike is not a narrow kink: the finite-difference slope along the gradient's own direction equals the analytic
  norm out to h = 0.2 (ratio 1.00-1.05) and is 0.87-1.12 at h = 0.5. The critic says Q changes by several units
  across the action range at those states.

### Mechanism (`14_gate2b_mechanism.py`)

Split Q - s_mode = sum_k p_k (s_k - s_mode) into bins within 6 units of the modal bin (the label kernel's own
width), a low tail and a high tail; dQ/da splits the same way, exactly.

| | 1M | 4M | 7.9M |
|---|---:|---:|---:|
| low-tail mass, spike states mean (median) / rest mean | 0.001 / 0.000 | 0.063 (0.007) / 0.001 | **0.176 (0.103) / 0.005** |
| mean position of that tail below the mode | -16 | -17 | -18 |
| share of the spike states' dQ/da that is the low-tail term (energy-weighted) | 0.41 | 0.84 | **0.94** |
| same for ordinary states | 0.00 | 0.13 | 0.33 |
| rows with tail mass > 5%, and their share of all gradient energy | 0.0%, 0% | 0.9%, 70% | **2.7%, 84%** |

A spike is the action-derivative of the probability the critic assigns to a second, low-value outcome about 17-18
units below the mode (a stumble or loss of speed; a full crash costs more). The decode makes it large: dQ/dlogit_k
= p_k (s_k - Q), so moving one percent of mass between modes 18 apart moves Q by 0.18, more than the median
within-state action sd of Q (0.12). Every stage of the critic amplifies at those states (decode gain x2.1, head x3.5,
encoder x3.4 at 7.9M); RMSNorm scale is not involved (1 / rms ratio 0.99-1.05). At 1M the distribution has no
second mode anywhere and there are no spikes.

Is the second mode real? Not decidable for v60. Realised discounted reward over the next 100 steps at spike states
is not distinguishable from elsewhere (share more than 5 below the median 0.40 vs 0.32 at 7.9M, n = 43). On the
oracle datasets (other recipes), the true slope along the critic's own direction at the critic's top-1% states is
0.29 [-0.03, 1.22] of the claimed slope at 4M (v23), 1.40 [-0.63, 4.3] at 8M (v29), 1.20 [0.41, 2.43] at 12M,
-0.47 [-1.02, -0.01] at 30M, six states each. The true field is itself heavy-tailed (states ranked on one third of
the rollouts, energy from the other two: top 1% holds at least 0.20-0.75), but the critic's large-gradient states
are not the truth's (Spearman of critic norm with split-half true energy +0.08 to +0.19), and in the 50-90% band the
true energy is 3-150x what the critic claims. So: large true action-gradients exist, the critic puts its own in
partly the wrong places, with the right sign at 4-12M and the wrong sign at 30M. "Mostly fitted noise in amplitude"
(12) is supported only at 4M.

Most likely mechanism, in one sentence: a two-hot-style categorical critic represents rare bad outcomes as a
separate mode, the expectation's gradient with respect to the action is the mode separation times the
action-sensitivity of a small mixture weight, that weight is estimated from few events and re-estimated every
iteration (12: amplitude moves 3-6x per iteration while direction holds), and the actor consumes it unnormalised.

### The actor's parameter gradient (v60 loss, first step of an iteration, 48 minibatches of 2048)

| | 1M | 4M | 7.9M |
|---|---|---|---|
| unmodified: pre-clip norm, median [p10, p90] | 0.35 [0.32, 0.37] | 1.06 [0.88, 1.28] | 1.17 [0.92, 1.58] |
| ... relative to the 0.5 clip | 0.70 (never clipped) | 2.1 | 2.3 |
| ... cosine between two minibatches' gradients | 0.91 | 0.81 | **0.50** |
| without the top 1%: norm; cosine with unmodified (minibatch, full batch); between minibatches | 0.31; 0.97, 0.97; 0.91 | 0.86; 0.84, 0.84; 0.93 | 0.87; **0.62, 0.67**; 0.84 |
| top 1% alone: norm; cosine with unmodified | 0.11; 0.37 | 0.69; 0.34 | **1.23**; 0.46 |
| (f) clip p99, pre: norm; cosine; between minibatches | 0.34; 1.00; 0.91 | 0.91; 0.95; 0.91 | 0.89; 0.81; 0.76 |
| (f) clip p99, action coords | 0.34; 1.00; 0.91 | 0.93; 0.95; 0.91 | 0.87; 0.85; 0.73 |
| clip p90, pre | 0.29; 0.98; 0.92 | 0.67; 0.84; 0.94 | 0.76; 0.62; 0.90 |
| (e) unit per state, pre: norm; cosine; between minibatches | 0.41; 0.83; 0.94 | **3.33**; 0.51; 0.97 | **5.29**; 0.43; 0.97 |
| (e) unit per state, action coords | 0.39; 0.88; 0.92 | 3.16; 0.49; 0.97 | 3.85; 0.44; 0.96 |
| entropy term's share: norm of pathwise part / entropy part | 0.34 / 0.06 | 1.13 / 0.40 | 1.32 / 0.84 |

- At 4M the top 1% of states hold 79% of the per-state energy but are a minority of the parameter gradient: the
  other 99% add coherently (norm 0.86 against 0.69 for the spikes) and the direction survives their removal at 0.84.
  At 7.9M the spikes are the larger part (1.23 against 0.87), removing them turns the gradient to cosine 0.62-0.67,
  and they are why two minibatches of the same rollout agree only at 0.50.
- Removing or clipping the spikes does not take the gradient under the clip: 0.86-0.93 at 4M, 0.87-0.89 at 7.9M,
  still 1.7-1.8x the threshold. 12's reading "every other state's signal is scaled down 5-8x by a handful of rows"
  is wrong: the clip binds without them.
- Unit-norm per state is a different optimiser, not a conditioner: norm 6-10x the clip, cosine 0.43-0.51 with
  the current direction. p99 clipping is the mild option (cosine 0.81-0.95); at 7.9M it leaves the between-minibatch
  agreement at 0.73-0.76, because p99 is still 16x the median; p90 reaches 0.90.
- What the logged pre-clip norm is (`14_gate2c`). The run logged iteration means of 0.72 / 3.08 / 2.59. Rebuilding the
  last actor step from the checkpoint (policy against its saved anchor): 36% / 16% / 42% of states are outside the
  KL gate, and the gradient is objective-on-inside-states 0.20 / 0.84 / 0.56 plus KL-pull-back-on-outside-states
  0.61 / 0.89 / **2.53**, total 0.62 / 1.38 / 2.63. Late in an iteration at 7.9M the 0.5 clip is scaling down a
  gradient that is 96% restoring force by norm, and the critic-driven part enters at about a fifth of its size.
  That is a larger conditioning problem than the spikes, and none of options (e) / (f) touches it.

---

## Gate 3 — observation normalisation (`14_gate3_obsnorm.py`; lane tables in `data/14_gate3_lane_stats_*.tsv`)

How v60 normalises: `VectorObsNorm` keeps one independent cumulative mean / variance per env lane (the
`NormalizeObservation` contract, `cleanrl/shared/vector_norm.py`); the trainer stores raw observations and, at
update time, passes each row through the current statistics of the lane that produced it (`normalize(raw, rows)`,
`env = rows % N`). Actor and critic are shared by all 16 lanes and receive no lane identity. Statistics are
cumulative from step 0 (495k samples per lane at 7.9M), so early tumbles never wash out.

### Lane statistics against the upright gait

| | 1M | 4M | 7.9M |
|---|---|---|---|
| pitch: lane cumulative sd, min / median / max | 0.91 / 0.98 / 1.09 | 0.66 / 0.74 / 0.84 | 0.88 / 0.94 / **1.51** |
| pitch: lane cumulative mean, range | 0.60 .. 0.69 | 0.13 .. 0.19 | 0.07 .. 0.23 |
| pitch: gait mean, sd | +0.19, 0.30 | -0.14, 0.16 | -0.16, 0.18 |
| pitch: gait sd in normalised units (median lane) | 0.31 | 0.22 | **0.19** (0.12 in lane 4) |
| pitch: gait p1..p99 in normalised units, median width (own-statistics width) | 1.49 (4.85) | 0.87 (3.90) | **0.78 (4.15)**, e.g. [-0.48, +0.01] |
| pitch: gait mean offset from the cumulative mean, in gait sds | -1.5 | -1.9 | -1.4 |
| pitch velocity: lane sd ratio max / min; gait sd in normalised units | 1.04; 1.04 | 1.04; 1.06 | 1.02; 1.12 |
| other compressed inputs (gait sd in normalised units) | rootz 0.48, vx 0.69 | rootz 0.66, vx 0.68 | rootz 0.66, vx 0.64 |
| all 17 dims: lane sd ratio max / min | 1.02-1.20 | 1.01-1.27 | 1.02-**1.72** (pitch), 1.19 rootz, 1.09 vx |
| lane mean spread in gait sds: pitch, vx, rootz | 0.30, 0.11, 0.39 | 0.35, 0.12, 0.11 | **0.93**, 0.34, 0.11 |

Pitch velocity is clean (the tumbles do not distort it). Pitch is the one input set by tumbles: the gait occupies a
fifth of a normalised unit, 4-5x less than every joint input, and sits 1.4-1.9 gait sds off centre. The v23 / v29
chunks show the same (gait pitch sd 0.19-0.26 normalised units at 4M / 8M, lane sd ratio 1.29-1.53, 2.10 at 30M).

A constant rescale of one input is not an information loss by itself: the first layer is linear and can undo it.
What it costs is that the network must carry 4-5x larger weights on that input than on the others, and it does:
+0.05 normalised units of pitch (0.05 rad, a quarter of a gait sd) moves the policy mean by 0.55 / 0.70 / **1.04**
policy sds (median) at 1M / 4M / 7.9M, against 0.08-0.35 for the same step in bthigh, rootz, vx or pitch velocity. That
gain is what turns small lane differences into different policies.

### The same physical state through different lanes

| | 1M | 4M | 7.9M |
|---|---|---|---|
| pitch input: sd across lanes, in within-gait sds (rms over states; largest two-lane gap) | 0.11; 0.74 | 0.09; 2.2 | **0.12; 2.3** |
| rootz, vx inputs: same | 0.10, 0.05; 0.57, 0.39 | 0.06, 0.04; 0.82, 0.18 | 0.06, 0.11; 0.85, 0.56 |
| the 14 other inputs (rms) | 0.004-0.023 | 0.003-0.016 | 0.005-0.019 |
| policy KL between two lanes at the same state: median, mean | 0.146, 0.35 | 0.060, 0.20 | **0.121, 0.62** |
| ... without the most deviant lane | 0.120, 0.26 | 0.057, 0.18 | 0.095, 0.32 |
| ... with pitch alone read through one common statistic (mean KL) | 0.13 | 0.11 | 0.23 |
| mean-action displacement from the lane-average policy, in policy sds (rms; worst lane) | 0.57; 1.17 | 0.43; 0.72 | **0.76; 2.12** |
| sd of Q across lanes at the same (s, a): rms, median | 0.20, 0.11 | 0.47, 0.25 | **1.08, 0.68** |
| within-state sd of Q over 8 policy actions: rms, median | 0.12, 0.05 | 0.45, 0.11 | 0.54, 0.12 |
| sd across lanes of the within-state advantage / rms advantage | 1.14 | 0.67 | 1.08 |
| correlation between two lanes' rankings of 8 actions: median, p10 | 0.96, 0.77 | 0.98, 0.82 | 0.93, 0.68 |
| dQ/da cosine between two lanes: median, p10; norm max / min over lanes at top-1% states | 0.99, 0.84; 1.8 | 0.99, 0.89; 8.3 | 0.99, 0.68; **12.4** |

v60's per-iteration KL bound is 0.03.

- In input space the lanes disagree by about a tenth of a gait sd on pitch, rootz and vx, which looks negligible
  (12 called the mean shifts negligible). Through the networks it is not: the policies that two lanes execute at the
  same physical state are a median 2-5 KL-bounds apart, and 0.7-2.1 policy sds from the lane-average policy in the worst lane (lane 4 at 7.9M,
  whose pitch sd is 1.51 after somersaults).
- For a critic shared across lanes: one input vector stands for 16 slightly different physical states, each
  followed by a slightly different effective policy, and the regression has to average their values. The spread of
  Q across lanes at fixed (s, a) is 2-6x the median action-induced spread of Q that the actor is trying to climb
  (0.68 vs 0.12 at 7.9M). Most of that is a common shift per state: action rankings agree at 0.93-0.98 in the median
  state, 0.68-0.82 at the tenth percentile; the direction of dQ/da agrees at 0.99 (p10 0.68-0.89); the amplitude at
  spike states does not (lanes differ 8-12x at 4M-7.9M). So lane disagreement is not what makes the typical state's
  gradient direction poor; it is a floor on Q accuracy of about one advantage sd, an extra source of
  spike-amplitude scatter, and a 16-way split of the behaviour policy that the on-policy critic is told is one policy.
- Mean-action evaluation uses lane 0's statistics only; at 7.9M lane 0 is a typical lane (KL 0.45 against the
  others), at 1M it is the most deviant (0.92).

Verdict: hypothesis 2 of report 12 is confirmed on resolution (4-5x on pitch) and its lane effect is understated
there. The cheap fixes are the ones already listed (statistics shared across lanes, or pitch wrapped / statistics
from upright rows); shared statistics alone remove the lane term and cost nothing. This is measured on network
outputs, not on return: whether it moves the 8M number is for the Wave-2 arm.

---

## Limits

- One seed, one snapshot per stage; the oracle stages are v23 / v29 recipes, the spike and normalisation numbers are
  v60. Gate 1's truth norm is resolved only at 4M.
- v60 states are 5000 per checkpoint from five lanes' statistics; top 1% = 50 states. The 4M sample has one crashed
  episode.
- Gate 2's actor-gradient numbers are the first step of an iteration (and, in `2c`, the last); the 30 steps in
  between, with a critic being fitted on the same rows, are not reconstructed.
- Gate 3 measures how much the networks' outputs change, with the learned critic standing in for the true Q across
  lanes.
- P(better) values were computed for about a dozen estimators per stage; isolated values near 0.8-0.9 (linear-space
  clip at 4M, the v60 chunk-only clip) are not corrected for that.

## Files

- Report: `benchmarks/reppo_rootcause/14_cpu_gates.md`
- Scripts: `scripts/14_common.py`, `14_states.py`, `14_gate1_hybrid.py`, `14_gate1_table.py`, `14_gate1b_v60.py`,
  `14_gate2_spikes.py`, `14_gate2b_mechanism.py`, `14_gate2c_end_of_iteration.py`, `14_gate3_obsnorm.py`
- Outputs: `data/14_gate1_log.txt`, `data/14_gate1_results_<stage>.json`, `data/14_gate1b_log.txt`,
  `data/14_gate2_log.txt`, `data/14_gate2b_log.txt`, `data/14_gate2c_log.txt`, `data/14_gate3_log.txt`,
  `data/14_gate3_lane_stats_<step>.tsv`, `data/14_states_<step>.npz`
