# REPPO root-cause study: 02 code audit

Scope: `cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py` (line by line), its diff against
`..._fast_v36.py`, the target path of `..._exacttarget_v61.py`, the critic and loss of `..._v1.py` / `..._decoupled_v7.py`,
`cleanrl/shared/{gae,ppo_loop,hl_gauss,vector_norm,rollout_transfer,graph_step,staggered_envs}.py`,
`scripts/reppo_critic_oracle.py` and `scripts/landscape.py::reppo_adapter`.

All numbers below are CPU-only re-analyses or small CPU checks; no training or GPU job was launched and no
existing file was edited. Evidence base is narrow: one run (`reppo_noisefork_v60_base`, seed 1), its three
checkpoints, and the one oracle array file that exists (`runs/reppo_critic_oracle/v60_base_1015424.npz`, the
1.0M-step checkpoint, 512 states). Treat generalisation to 8M as untested.

Helper scripts (all under `benchmarks/reppo_rootcause/scripts/`):
`audit_oracle_npz.py`, `audit_oracle_gradient_robust.py`, `audit_numeric_checks.py`,
`audit_actor_gradient_tails.py`, `audit_fit_capability.py`, `audit_tf32_emulation.py`.

## Headline

The implementation is clean: I found no bug in target construction, indexing, normalization, HL-Gauss, the actor
loss, or update ordering. The one confirmed defect is in the **oracle's statistics**, and it changes the diagnosis:

- The "Q correlates -0.33 with its own r + gamma V'(f(s,a))" result is produced by **one off-distribution state**
  out of 512. Without it the correlation is **+0.48** (per-state median +0.55, positive in 80% of states).
- The same state inflates the critic's "within-state sd" from 0.062 to 0.206, which made the critic look like it
  had the right amplitude and random direction. It actually has **about a quarter to a third of the true amplitude**.
- The same state drags the gradient slope (true-on-critic, pre-tanh) from **0.7-1.2 down to 0.2-0.37**. The
  "action gradient is only 20-37% real" figure is that artifact.

Corrected picture at 1M steps: the critic's action dependence is **calibrated but strongly attenuated** (what it
claims is roughly real; it claims little), and it is **self-consistent** with its own one-step lookahead. That is the
signature of under-resolution, not of a noisy or sign-flipped gradient.

## CONFIRMED

### C1. Oracle pooled statistics are owned by a single outlier state

`scripts/reppo_critic_oracle.py:464-488` (`within_stats`), `:491-536` (`gradient_stats`), `:562-579`
(`residual_stats`), `:590-612` (`model_based`). Every headline statistic is a ratio of sums over states
(`(ours * truth).sum() / sqrt(...)`, `dot / critic_sq`), so each state is weighted by its squared critic contrast
or squared critic gradient.

State 402 (age 691, critic Q 13.4, true Q 3.8; the rest of the sample has Q in 35.9-44.1) is a failed/flipped
cheetah. Its critic values over the 8 fresh actions span about +-6 and its |dQ/da| is 28.6 against a median of 0.455.

| Quantity (soft truth, 1M checkpoint) | All 512 states | Without state 402 / lowest-2% Q |
|---|---|---|
| Share of sum of squared critic contrast in the top 1% of states | 0.895 | - |
| Share of sum |dQ/da|^2 in state 402 alone | 0.80 | - |
| Critic vs own 1-step lookahead, within-state corr | -0.331 | **+0.476** (per-state median +0.55) |
| Critic within-state sd / true sd | 0.92 | **0.32 +- 0.02** (0.062 vs 0.265 raw) |
| Critic vs truth, within-state corr (noise-corrected) | 0.149 | 0.232 +- 0.040 |
| Slope true-on-critic, pre-tanh h=0.1 | 0.37 +- 0.44 | **1.17 +- 0.38** |
| Slope true-on-critic, pre-tanh h=0.3 | 0.21 +- 0.26 | **0.69 +- 0.13** |
| Slope true-on-critic, action space h=0.1 / 0.3 | -0.01 / 0.02 | 0.34 +- 0.33 / 0.28 +- 0.24 |
| Slope 1-step-lookahead-on-critic, pre h=0.1 / 0.3 | -0.13 / -0.15 | +0.92 / +0.86 |
| Gradient rms (pre, h=0.3): critic / 1-step lookahead / 5-step lookahead | 0.339 / 0.319 / 0.379 | **0.126 / 0.209 / 0.361** |

SEs are state bootstraps. The result is stable to the trimming rule: dropping only state 402 gives 1.13 / 0.66;
dropping the lowest 1%, 2% or 5% by Q gives 1.11-1.17 / 0.69-0.71; dropping the top 1% by |g| gives 1.02 / 0.62.
Dropping the top 5% by |g| lowers it to 0.61 / 0.52, so the slope is somewhere in 0.5-1.2, with wide error bars.
The oracle's own `ascent_ratio` (weighted by |g|, not |g|^2) already said 1.17 and 0.54 on all states.

Reproduce: `.venv/bin/python benchmarks/reppo_rootcause/scripts/audit_oracle_npz.py` and
`.../audit_oracle_gradient_robust.py`.

What the corrected numbers say:
- True-on-lookahead slopes are about 1 at both horizons (0.82-0.95 for 1 step, 1.01-1.08 for 5 steps), so
  truth >= lookahead in amplitude: critic 0.126 < 1-step 0.21 < 5-step 0.36 <= truth (about 0.41).
  Each time the action effect has to pass through the fitted function it loses roughly half its amplitude.
- The lambda=0.8 target is a mixture of n-step lookaheads with weights 0.2, 0.16, 0.128, 0.102, 0.082 (67% on n<=5).
  Using the measured lookahead amplitudes, the target's conditional mean carries roughly 0.31 of gradient rms,
  about three quarters of the truth. The critic keeps 0.126. The loss is in the regression step, not in how
  the target is built.
- In action space the slope is lower (about 0.3 +- 0.3) and 35% of pairs are dropped for saturation; the actor's
  gradient flows in pre-tanh space, so the pre-tanh rows are the relevant ones.

Minimal fix: report per-state medians and a trimmed or per-state-normalised pooled statistic next to every pooled
one, print the energy share of the top state, and exclude or separately report states whose critic Q falls below
the 1-2% quantile. Predicted effect: no change in training; the diagnosis changes from "gradient is mostly noise
and anti-consistent" to "gradient is roughly calibrated, a factor 3 too small in amplitude, r^2 about 0.05".

Not checked: the 4M and 7.9M checkpoints have no oracle arrays. `audit_actor_gradient_tails.py` shows the same
heavy tail in the critic's slope there (top 1% of on-policy states hold 73-80% of sum |sigma dQ/du|^2 at all three
checkpoints), so any pooled oracle statistic at those steps should be assumed contaminated until re-run.

### C2. Trainer diagnostics have ceilings far below 1 (definition, not a bug)

`v60:514-534`. `advantage_corr_heldout` correlates Q(s,a) - b with G_0.95 - b, where b is the same 8-sample
estimate of E_pi Q. G - b contains the state-level critic error and the full return noise: its sd is
`probe/real_advantage_std` = 0.80 at 1M against a true within-state sd of about 0.22-0.26. A perfect critic would
therefore score about 0.3 on `advantage_corr_heldout` and about 0.3 on `advantage_std_ratio`, not 1. The logged
0.096 and 0.13 are consistent with the corrected oracle (corr 0.23 x 0.3 = 0.07; amplitude 0.3 x 0.3 = 0.1).
The shared b adds a small spurious positive correlation (order Var_a(Q)/8 over the product of sds; negligible).
Both are plain Pearson statistics over rows, so they inherit C1's outlier sensitivity.
`losses/q_mse` (`v60:481`) is a training-row error against noisy targets averaged over all 128 updates including
the steps right after each target refresh; its floor is the target noise, and it is not a held-out error.

## LIKELY FLAWS (design, with evidence)

### L1. One action per state does not identify the action effect at this sample size, even without label noise

`audit_fit_capability.py`: 14 lanes x 1024 on-policy steps from the 1M checkpoint (14,336 rows, one action per
state), labels L1(s,a) = r + gamma V'_ckpt(f(s,a)) computed through the simulator, so they are a deterministic
function of (s,a) with a known action dependence. The v60 critic is fit as in the trainer (CE on HL-Gauss labels,
AdamW 3e-4, clip 0.5, 2048-row minibatches) and scored on 384 held-out states x 8 fresh actions where L1 is known
exactly (within-state sd of L1 0.301; lowest-2% Q states trimmed).

| Arm | Updates | Within-state corr (per-state median) | Amplitude sd(Q)/sd(L1) | Slope L1-on-Q | Train rms | Held-out level rms |
|---|---|---|---|---|---|---|
| Trained critic, as loaded | 0 | +0.39 (+0.59) | 0.23 | 1.69 | 0.273 | 0.397 |
| A: continued, noise-free labels | 128 | +0.31 (+0.66) | 0.23 | 1.35 | 0.214 | 0.408 |
| A | 1536 | +0.24 (+0.68) | 0.32 | 0.75 | 0.111 | 0.423 |
| N: continued, labels + N(0, 0.4^2) | 128 | +0.26 (+0.60) | 0.25 | 1.05 | 0.441 | 0.429 |
| N | 1536 | +0.08 (+0.34) | 0.53 | 0.14 | 0.267 | 0.542 |
| S: fresh critic, noise-free | 1536 | +0.07 (+0.30) | 0.28 | 0.27 | 0.305 | 0.496 |
| W: fresh critic, whitened-noise action input, noise-free | 1536 | +0.06 (+0.22) | 0.57 | 0.10 | 0.318 | 0.552 |

Readings, with their limits:
- Noise-free labels and 12x the trainer's update count drive the training error from 0.27 to 0.11 while the
  held-out level error does not move (0.40 to 0.42) and the pooled within-state correlation falls. The critic
  spends its capacity on a state-indexed fit of 14k rows; the held-out state-level error (0.4) is larger than
  the action signal (0.3). This is a generalisation limit, not an optimisation or precision limit.
- A fresh critic cannot extract the action structure from one rollout at all (corr 0.07 after 1536 noise-free
  updates). What the trained critic knows about actions (corr 0.39 against its own lookahead) was accumulated
  over the preceding ~60 iterations.
- With label noise at the trainer's residual level, extra updates convert noise into spurious action dependence
  (slope 1.05 at 128 updates, 0.14 at 1536). v60's 128 updates sit near the point where the slope is still about 1.
  More critic updates per rollout is predicted to hurt, not help.
- Feeding the action as standardized noise (atanh(a) - mu(s)) / sigma(s) did not help in this test (larger
  spurious amplitude, lower slope). I had suspected the raw action acts as a state proxy; this does not support
  an input-parametrization fix.
- Pooled correlation and per-state median disagree in direction for arm A (0.39 to 0.24 against 0.59 to 0.68), so
  "continued fitting makes ranking worse" is not established; "it does not make it clearly better" is.
- One checkpoint, one rollout, one seed per arm, a 1-step label rather than the lambda-return, no aux loss, frozen
  bootstrap. A small test, not a benchmark.

Decisive test (needs GPU, not run): keep everything in v60 and give the critic K>=4 branched actions per state for
a subset of rollout states (MuJoCo `set_state`, one step each, label r + gamma V'(s')). If the oracle amplitude
ratio and within-state correlation rise sharply, identifiability is the bottleneck. This is a diagnostic, not a
method, since it uses simulator resets.

Minimal fix direction: get the action dependence from a quantity that is supervised densely and without noise.
The oracle already measures the upper bound for this: the gradient of r + gamma V'(f(s,a)) through the true
dynamics has slope 0.82-0.95 and 1.7-2.3x the critic's amplitude; five steps gives slope 1.0 and 2.9-4.4x. Every
transition is a noise-free 17-dimensional label for f, in contrast to one noisy scalar for Q. Whether the learned
latent model is accurate enough is the open question (`modelgrad_v30` exists; I did not audit it). Predicted
effect if the model is good: 2-3x the gradient amplitude at the same calibration; because the actor is
KL-saturated (L3), the benefit would come from better direction, not a bigger step.

### L2. The critic's action sensitivity explodes off-distribution and enters the actor loss unnormalised

`v60:495-499`: the actor loss is the plain mean of -Q over the minibatch. On on-policy rollouts
(`audit_actor_gradient_tails.py`, 16,000 states per checkpoint) the top 1% of states hold 80% / 73% / 75% of
sum |sigma dQ/du|^2 at 1M / 4M / 7.9M, with a max-to-median ratio above 100.
The effect on the actual update is moderate: the actor parameter gradient without the top 0.5% of states has
cosine 0.96-0.98 with the full one, and in 2048-row minibatches the top 10 rows contribute a gradient whose norm
is 0.1-0.7 of the batch gradient (cosine of batch gradient with and without them 0.75-0.99).
So this is a real but secondary distortion, and mostly a hazard for diagnostics (C1).
Test: clip the per-state dQ/du norm at its 99th percentile in `actor_loss`. Predicted effect: small.

### L3. The actor is permanently trust-region-saturated

Logged: `policy/kl` 0.030-0.032 against a bound of 0.03 and `policy/kl_violation` 0.32-0.41 at every stage.
A third or more of the states in each actor minibatch are outside the gate and receive only the KL pull
(`v60:497-499`), and actor gradients are clipped at every stage (pre-clip norm 0.55 to 3.3 against 0.5).
The actor's step length is set by the gate, not by |dQ/da|. Attenuation of the gradient is therefore not
itself the harm; only its direction (r^2 about 0.05 against truth) matters. This is by design, not a bug.

### L4. Minor inconsistencies, each measured or bounded, none plausibly causal

- Behaviour actions are sampled from observations normalised with the running statistics at step time; training
  renormalises the whole rollout with end-of-rollout statistics (`v60:437-445`, `:790-792`). `old_actor` on the
  renormalised observations is therefore not exactly the behaviour policy, and `rollout_noise` is not exactly the
  noise of `rollout_actions` under it. At 1M the statistics move by about 1.6% of one rollout's shift per
  iteration; early iterations are affected more.
- Normalisation is per environment lane. At 1M the 16 lanes' means differ by a median 0.010 (max 0.049) pooled
  sd and their sds by up to 20% in one dimension, so the same raw state reaches the networks through 16 slightly
  different affine maps. Small next to between-state variation; not negligible next to the one-step state
  perturbation an action causes. A single shared statistic would remove it.
- The bootstrap V' uses the live actor (`v60:451`), which moves during the iteration, while the Monte-Carlo part
  of the lambda-return is under the behaviour policy. Inherited from REPPO; bounded by the KL gate.
- AdamW's default weight decay 0.01 applies to every critic parameter including `zero_dist` and the norm gains
  (`v60:403`). Total shrink over 8M steps is about exp(-0.19) on parameters with no gradient; negligible.
- v36 and seven other versions default to `matmul_precision="high"` (TF32). Emulated TF32 adds an rms error of
  0.002-0.007 to within-state value contrasts (on-distribution contrasts are 0.06-0.3) and 0.1% to autograd
  gradients. It matters slightly for value-difference diagnostics and not for the actor. v60 uses strict FP32.

## CLEARED (checked, correct; no need to re-check)

Target construction
- Lambda-return recursion (`v60:294-304`, `ppo_loop.py:80-99`, `gae.py:49-50, 67-69`): matches a brute-force
  implementation of G_t = r_t + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H_{t+1})] to 1.8e-15 on a toy chain
  with terminations, truncations and the rollout end. The value shift across episode boundaries is harmless
  because boundaries reset the trace. The Triton kernel is pinned to the reference by `tests/test_ppo_loop.py`.
- Entropy: enters at the next state only, as V' = mean_k [Q - T log pi] and as gamma lam T H on continuing steps
  (`v60:447-456, 815-823`); consistent with the actor objective -Q + T log pi and with the oracle's definition.
- Truncation and time limit: `raw_next` takes `final_observation` (`v60:763-765`); truncations and the rollout end
  bootstrap V' alone; terminations bootstrap nothing. HalfCheetah never terminates.
- Indexing: `raw_obs=raw_current` (s_t), `actions` (a_t), reward r_t, `raw_next_obs` (s_{t+1}) are aligned
  (`v60:755-769`); `RolloutTransfer.push` copies at call time, so the reused action and noise buffers are safe;
  row % num_envs is the lane.
- Bootstrap network: `target_critic` is a copy of the critic taken at insert and again every 16 updates
  (`v60:793, 858-864`). Eight fitted-Q sweeps per iteration; this is the intended REPPO schedule.

Scale and HL-Gauss
- Reward scale is the fixed 0.1; there is no running reward normaliser, so labels are stationary in scale.
- Support [-50, 250], 151 bins, width 2, sigma 1.5: encode/decode round trip error 4e-5 in FP32 in the interior,
  slope of decode against target within 0.3% of 1; a 0.03 contrast at Q=100 survives (100.0300064). The CE optimum
  for noisy targets decodes to the target mean (99.9985 for N(100, 0.4)); bias appears only within a few sigma of
  an edge (-0.13 at 245). `support_clip_frac` is 0 throughout and Q peaks near 113.
- FP32 on the trained critic: |Q32 - Q64| at most 7e-5, contrast error rms 5e-6, dQ/da relative error median 5e-6.

Action path into the critic
- The critic sees the tanh-squashed action that the environment received; no clipping mismatch (the space is
  checked to be [-1, 1]).
- First-layer column norms are equal for action and observation inputs (action 2.15-2.80, observation mean 2.48).
  The first layer does not suppress the action. RMSNorm after it rescales per sample and does not remove direction.
- Autograd and finite-difference critic gradients agree (oracle: cosine 1.0).

Actor loss and optimisation
- Sign, per-state KL gate, entropy dual and KL multiplier signs are correct (`v60:486-507`).
- No actor-loss gradient reaches critic parameters (`backward(inputs=actor_parameters)`, `v60:638`); no
  dropout or batch statistics, so there is no train/eval mismatch; BF16 is off by default.
- Gradient clipping at 0.5 is active on the critic (pre-clip 0.85-2.0), but under Adam a per-step rescale changes
  little. Not a plausible cause.
- Update order: targets, then 4 critic updates per actor update, 128 and 32 per rollout, uniform rows with
  replacement (each row about 16 critic and 4 actor visits). Optimiser state persists across iterations.

Aux latent roll
- Indices of successor rows, chain masks, reward rows and target features are correct (`v60:458-484`).
  Logged aux loss 0.04-0.09 and reward loss below 1e-3 against a reducible CE of about 0.1-0.2. I did not measure
  gradient shares on the trunk; a no-aux run exists in the family history (`reppo_v7_q_noaux`).

Oracle (apart from C1)
- Observation reconstruction, per-lane statistics, reward, entropy bonus sign, bonus timing (never at s_0),
  one-step lookahead (r + gamma V'(s') with V' noise shared across a state's actions), common random numbers and
  the leave-one-out baseline are all consistent with the trainer.
- It samples states from the post-update actor while the critic was fit on the previous policy's actions. That is
  the distribution the actor queries next, so it is the right one; just note it is not the training distribution.

## Not done

- No oracle arrays exist for 4M or 7.9M, so C1's corrected numbers are for 1M only.
- Gradient share of the aux loss on the encoder was not measured.
- The decisive test for L1 and any training-level confirmation need GPU jobs, which were out of scope.
