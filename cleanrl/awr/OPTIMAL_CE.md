# Toward the optimal CE construction (living document)

Scope: PPO-replacement policy updates for the Beta-policy MuJoCo trainers in this folder, framed as
cross-entropy (CE) projections onto a per-sample target. Constraints: V(s) critic only (no Q(s,a)),
one rollout, intra-batch, non-contrastive, no architectural change, no replay, no trust-region crutch.
Revise this file whenever a result changes what we believe. Newest revision first in the log at the bottom.

## 1. What a CE rule is

Per sample i (state s_i, one action a_i ~ pi_old, advantage estimate A_hat_i, z_i its batch z-score),
r = pi/pi_old, delta = log r. Every rule in the family is

    L = mean_i w_i (r_i - r*_i log r_i)

Two exact rewrites, both used below:

    w (r - r* log r) = -f delta + w (r - 1 - log r)       + const       f = w (r* - 1)   (log form)
                     = -f (r - 1) + k (r - 1 - log r)       + const       k = w r*         (ratio form)

- f is the **force**: the gradient at theta_old is -mean f_i grad log pi_i, a policy gradient with f in
  place of A. It decides the direction.
- w is the **curvature in delta** at theta_old (d2/d delta2 = w e^delta): a per-sample penalty
  r - 1 - log r, whose pi_old-expectation is KL(pi_old || pi) only when w is constant within a state.
  It decides how expensive it is to move each sample's log-probability.
- r* = 1 + f/w is derived, not chosen. It is the per-sample fixed point, and **only matters if the
  update actually converges toward it**.

The design space is therefore two functions of the data: (f, w). Plus one structural choice the
sample-level form hides: **which output directions each sample is allowed to move** (section 4).

There is no clean "converged regime". With far more parameters than samples, the w-weighted
empirical Fisher (one action per state) is singular: its min-norm solution fits every sample
exactly (delta_i = f_i / w_i), and where the fit actually goes is set by Adam's implicit bias and the
network's pooling across states, not by a natural gradient. In practice the update is Adam on the
force with a brake from w, stopped after 10 epochs.

## 2. Evidence so far (HalfCheetah, 50M unless noted)

| Rule | f (force) | w (curvature) | Result |
|---|---|---|---|
| PPO control | A (raw), zero past clip | 0 inside box, wall at 1 +- 0.2 | 13.58k s1; 4 seeds 12.2-13.8k, mean 13.16k |
| v1-v5 exp CE | exp-weighted | exp | ESS collapse on kurtosis ~100 tails; far below |
| v6-v8 log-ratio LS | z/eta | uniform (exact KL add-back) | ~12.2k; unconverged targets (fit R2 0.02-0.14), tail-driven |
| v10 chi2 CE | (z - lambda)/eta floored at -1 | r*-shaped | 12.57k; discarding negative magnitudes costs (ppo_chi2_weights ablation: 11.0k) |
| v14 (v10 + beta2 0.9) | as v10 | as v10 | 13.25k (single seed) |
| v15 exp target, linear force | z/eta | x/expm1(x) | 12.74k |
| v16 sign reach | z | abs(z)/abs(r*-1) | 12.17k s1; seeds 13.36k, 13.28k, 8.78k (s4 locked into a bad gait by 7M) |
| v17 sign up / exp down | z | mixed | worse (8-10k at 25M), cancelled |
| v18 (v16, actor lambda 0.8) | z from A_0.8 | as v16 | from scratch: ~5k plateau at 19M (v16 9.4k at 10M), culled. Forks: +0.34k vs v16 from PPO 24M, +0.51k from v16 20M |
| PPO actor lambda 0.8 | A_0.8 | as PPO | from scratch: ~5.9k plateau at 19.6M, culled. Forks: +0.39k from PPO 24M, -1.6k from v16 20M |
| v19 adaptive trace | z from the lambda chosen per iteration | as v16 | pre-fix (argmax over negative early signals picked the noisiest view): 5.9k at 15.8M. Fixed (0.95 until the MC reference has positive cross-fold signal): 8.3k at 5.75M, level with v16; running (job 9802) |

Diagnostics that constrain the theory:
- **Seed spread swamps the ranking.** PPO seeds span 1.5k; every CE rule's single-seed result lies
  inside that span. We cannot rank rules below ~1k differences with one seed.
- **Forks from matched states** (scripts/awr_fork_policy.py, actor Adam reset): from PPO 12M/24M and
  v16 20M/35M states, every linear-force rule is within noise of PPO over 8-15M steps. The average
  paired edge for PPO is ~0.45k, below the per-pair noise.
- **Fixed points do not matter.** v16 with 1.5-2x reach doubled or tripled per-update KL and gained
  nothing; the one-update bench showed held-out dJ flat across reach. We are in the unconverged
  regime: direction comes from f; w only brakes.
- **Actor Adam beta2 (0.9 vs 0.999) is not the late ceiling** (forks from both states).
- **Rules differ mainly in the concentration channel.** Entropy trajectories diverge strongly between
  rules and between seeds (v16 s1 stalls at -5.8, v16 s4 collapses to -10.7, PPO -8.5 to -9.4), while
  return trajectories from matched states tie. The failures we see (stall, premature gait lock) are
  concentration phenomena.

## 3. What is settled

1. **The force must be linear in A_hat, both tails intact.** Discarding negative magnitudes (v10's
   floor) costs ~2.5k; exponentiating (v1-v5) collapses ESS. The linear force is the only one that is
   unbiased under the fit's averaging, and every linear-force rule ties PPO.
2. **The per-sample fixed point (reach) is not a lever** in the regime the update actually runs in.
3. **Mass balance matters only through w, and only near convergence.** If E_old[r* | s] = 1 the
   per-state fixed point is r = r* for any w. Otherwise the normalization gives r = w r* / (w + lambda).
   When E[r*] > 1, lambda > 0 and mass drains off the low-w samples; with w proportional to abs(z)
   (v16, E[r*] ~ 1.6) those are the A ~ 0 actions. When E[r*] < 1, samples with w < -lambda are
   unbounded. With one action per state this "per-state fixed point" is only a heuristic, and our
   updates never get near it.

## 4. The open question: which directions a scalar can support

Each sample carries one scalar, A_hat_i, but the Beta head has 2 x act_dim outputs per state. What a
sample says about the per-state update is the rank-1 direction F_i^-1 s_i (s_i is the score of a_i in
the head's outputs, F_i the closed-form Beta Fisher). Averaged over samples, E[F^-1 s A | s] =
F^-1 grad J(s): the per-state natural gradient, unbiased in every output channel.

What the red team (CPU simulation, 1-D Beta, heavy-tailed noise, rare far-from-mean catastrophe)
established:
- **Noise is not channel-specific in natural units.** With action-independent noise, the per-sample
  natural-gradient noise is the same for location and log-concentration
  (Var([F^-1 s]_c A) / (Finv_cc E[A^2]) ~ 1 for both). "Concentration is intrinsically noisier" is false.
- **Signal is channel-specific.** Concentration signal is second order (Q'' sigma^2 against Q' sigma
  for location). Under a rare far-from-mean catastrophe it comes almost entirely from that
  catastrophe. Sharpening away from falls is then the correct expected gradient, not noise.
- **So the real problems are rare-event variance and credit.** The top 1% of abs(z) held 35-94% of
  the target energy. GAE also blames the last action for a fall that took several steps, so the
  catastrophic tail is both high variance and of uncertain credit. Channels differ in how much of
  their signal depends on that tail.
- **Per-channel scalars depend on the coordinates.** F is not diagonal off mu = 0.5 (corr -0.31 at
  0.85, -0.71 at 0.97), and one scalar A correlates the noise across channels. The principled
  object is a matrix.

The optimal construction, restated. What we want is the update that maximizes the realized
improvement of the fitted policy, not the best-estimated target per sample. For a fitted function
g_hat of the per-state natural gradient, that factor is its **calibration slope**,
Cov(g, g_hat) / Var(g_hat), measured out of sample. This is a per-channel (in general, per-direction)
step multiplier. Shrinking the targets by a Wiener ratio and then fitting would shrink twice, because
the network already pools across states. Adam's m / sqrt(v) already shrinks noisy head weights
somewhat, and zeroing concentration updates would freeze the entropy annealing HalfCheetah needs.
So the target is **calibrated steps per direction**, not "suppress concentration".

A second, independent lever is the **credit of the rare tail**. A catastrophe's advantage is
certainly real at the state level, but its assignment to the sampled action is uncertain. How the
rule treats the tail in each direction (location vs concentration) is where PPO's clip, v16's
abs(z) weights and exp targets actually differ, and it matches where the entropy trajectories
diverge.

### 4.1 Measured (scripts/awr_channel_reliability.py, jobs 9750/9754, runs/awr_channel_reliability.json)

Per-sample output-space natural-gradient targets F^-1 s z, 12 channels, on the PPO 12M/24M and v16
20M/35M snapshots. 262k training samples and 65k held out per snapshot. Fits are ridge regressions on
the actor's last hidden layer, split by env halves.

| snapshot | top-1% energy loc / conc / z^2 | attenuation slope loc / conc | held-out R2 loc / conc | A-from-state slope (R2) |
|---|---|---|---|---|
| PPO 12M | .64 / .58 / .60 | .02 / .10 | ~0 / 8e-5 | .93 (.015) |
| PPO 24M | .39 / .41 / .38 | .12 / .21 | 3e-5 / 5e-4 | .34 (.007) |
| v16 20M | .52 / .50 / .50 | .06 / .17 | ~0 / 2e-5 | .91 (.017) |
| v16 35M | .64 / .57 / .56 | -.03 / .10 | ~0 / 2e-5 | .87 (.009) |

The attenuation slope is the fraction of a fitted function that is signal: a fit on half the envs,
tested on the other half's samples. Standard error is about 0.12 per dim.

- **The channel hypothesis is refuted.** Concentration is at least as reliable as location in all 4
  snapshots. It is no more tail-heavy than location: both simply inherit z^2's tail. Location and
  concentration targets are nearly uncorrelated (mean abs(r) < 0.04). Dropping F^-1 (a vanilla PG
  target) changes nothing.
- **Every update is overwhelmingly noise, in every channel.** Even a fit on 131k samples keeps only
  0-20% signal. A 16k training batch keeps far less. Huber (tail-robust) slopes are about the same
  (~0.03), so no per-sample reshaping of the force recovers much. This explains the ties: any
  linear-force rule's update is mostly the same noise, whatever its (f, w).
- **The critic has state-predictable bias.** E[A | s] is predictable from the actor's features
  (slope 0.34-0.93, R2 0.4-1.7%).

### 4.2 What this implies for the construction

1. **Per-update signal, not per-sample shape, is the binding constraint.** With a 2-20% signal
   fraction, trajectories are closer to random walks with a weak drift. That also explains the
   1.5k seed spread and the gait lock in v16 s4. A construction beats PPO only if its realized
   update carries more signal, or drifts less under the same noise.
2. **Where the baseline enters.** In the ratio form L = -mean f (r - 1) + mean k (r - 1 - log r),
   a per-state shift of f by c at fixed k changes L by -c (r - 1). That has zero pi_old-expectation,
   the same invariance as PPO's surrogate. So the critic's per-state bias enters a CE rule only
   through how k is computed from the biased z. Every rule so far computes it that way:
   - v16: k = 1.58 z (good side), 1.22 abs(z) (bad side);
   - v10, v15: k = w r*, a function of z.

   The bias therefore perturbs the per-state anchor strength by roughly (dk/dz) c: a looser anchor in
   states where V overestimates (and bad-side samples dominate), a tighter one where it underestimates.
   The bias is measurably state-predictable (4.1). This is a systematic (not noise) term, second
   order in the step. In a regime where signal is 2-20% of the update, a systematic term is what
   compounds across updates. It is the one concrete mechanism found so far that separates a CE rule
   from PPO in expectation. (PPO's own asymmetry is its clip side, chosen by sign(A), so PPO is not
   fully invariant either.)
3. **Hence the candidate:** keep f = z, and compute k from the cross-fitted, per-state-centered
   advantage z - c_hat(s). Uniform k is the kce family, which already lost on the one-update bench
   (dJ-ref -0.1 to -0.3e-3, because tail reach is unbounded as f -> k). So the candidate is v16's
   abs(z)-proportional anchor, made invariant to the critic's bias.

## 4.3 The return-maximizing construction (what a meta-learner would learn, in closed form)

Imagine a network that sets every sample's CE force f_i so as to maximize the post-update return
J(theta + Delta(f)). No meta-learning is needed to find its optimum:

- **First order.** Delta = P sum_i f_i g_i (g_i = grad log pi_i, P the optimizer's preconditioner), so
  dJ/df_i = g_i^T P grad J. The optimal force is each sample's alignment with the true gradient,
  i.e. how much raising this action's probability would raise the return, **through the network's
  own generalization**. grad J is unknown. Estimated from the *other* samples it is
  sum_{j != i} K_ij A_hat_j, with K_ij = g_i^T P g_j the network's (preconditioned) tangent kernel.
  Self-terms (K_ii A_hat_i^2) are pure noise that looks like improvement, which is exactly what
  10 epochs on raw targets fit.
- **Second order.** Adding the curvature cost of a noisy step turns the choice of force into a
  Wiener filter: kernel ridge regression of A_hat on the tangent features, with the ridge penalty
  set by the data. In primal form it regresses A_hat_i onto the score features g_i. That is
  **compatible function approximation** (Kakade 2001; Peters & Schaal, NAC): the fitted weights are
  the damped natural gradient, and the prediction g_i^T w is the denoised, compatible advantage.

Why it fits every constraint and every lesson so far:
- **Linear in A_hat.** The force stays unbiased (section 3.1).
- **Exactly per-state centered.** E_old[g | s] = 0, so the critic's per-state bias cannot reach the
  targets (section 4.2), with no extra critic machinery.
- **Denoised by the batch's own evidence.** The penalty comes from cross-validation over env-grouped
  folds, and each sample's target comes from a fit that never saw its env, which removes the
  self-noise term above. When a batch has no predictable signal, the targets shrink to zero instead
  of the update fitting noise.
- **Reachable.** The targets lie in the policy's own tangent space, so the full-network CE fit can
  actually reach them. The 10 epochs converge instead of chasing per-sample noise (the v6-v8 failure).
- **The textbook target works again.** Denoised targets have light tails, so the KL-optimal
  exp(A_compat / eta) no longer collapses the ESS (the v1-v5 failure). eta keeps its meaning as the
  exchange rate between advantage and nats, and it is now a real step size, because fixed points are
  reached.

Red-team corrections (rev 5):
- **"Natural gradient" is true only of the fit's endpoint.** A fit converged onto the targets moves
  the head by the ridge coefficients, a damped natural step. The first gradient on them is
  F (F + lambda)^-1 g, and that tends to the plain policy gradient as lambda -> 0 (projecting onto the
  compatible span leaves the head gradient unchanged). What is genuinely new is (a) cross-fitting,
  which removes each sample's own noise, (b) the kernel consensus as the per-sample force, and
  (c) targets that are reachable.
- **On real batches the MSE-optimal penalty sits at the grid edge.** Cross-validated R2 is 0-3e-4 per
  16k batch (job 9764, cancelled), so the target scale was set by the grid. Fixes:
  - pick the penalty by held-out first-order improvement, sum(y y_hat) / abs(y_hat), including the
    infinite-penalty limit (the pure kernel consensus of the other folds, which is exactly the
    first-order meta-optimum);
  - normalize targets to unit std, so eta is a temperature in std units;
  - the shuffled-advantage null then moves the policy too, which makes the comparison fair.
- **P is now Adam's real preconditioner.** The features are scaled by (sqrt(v_hat) + eps)^-1/2 from the
  snapshot's moments, so the kernel is g_i^T P g_j, as in the derivation. That also replaces the
  arbitrary isotropic penalty.
- **Per-state centering holds in expectation only.** Finite-sample coefficients still carry some
  critic bias.
- **Selection is optimistic.** compat is picked from 3 eta families against PPO's 1, so judge it by
  consistency across the 4 snapshots, not by the best cell.

Implementation (bench rule `compat`, scripts/awr_update_bench.py):
- The tangent features are the exact per-sample gradients of the actor head's weight, bias and
  readout gain (12 x 65 + 12 = 792 dims, against 16k samples), preconditioned by P^1/2.
- Kernel ridge cross-fitted over 4 env-grouped folds, with the penalty chosen by held-out alignment.
- Targets r* = exp(y_hat / eta), with y_hat at unit std, fitted by plain CE.
- Synthetic check:
  - at a true R2 of 2% under t(2.5) noise, the target correlates 0.57 with the true signal
    (raw: 0.14);
  - at 0.2%, 0.18 (raw: 0.045).

Open risks:
- Last-layer tangent space only. The hidden layers' tangent directions are ignored when forming the
  target, though the full network is still free in the fit.
- The alignment criterion is itself noisy at a per-batch R2 around 1e-4.

### 4.4 Result: compat refuted on the bench (job 9766, runs/awr_update_bench_compat_v2.json)

Held-out dJ (e-3) at matched KL, compat's best eta against PPO:

| snapshot | PPO | best compat |
|---|---|---|
| v16 20M | +0.029 (KL 0.03) | -0.003 (KL 0.02) |
| v16 35M | +0.018 (KL 0.02) | -0.016 (KL 0.02) |
| PPO 12M | +0.001 (KL 0.02) | about -0.01 to -0.1 |
| PPO 24M | +0.051 (KL 0.03) | negative |

Other measurements:
- The chosen penalty swings between the grid's lower edge and inf.
- The held-out alignment of the targets with y is 0.004-0.03.
- In-sample gain is ~1/4 of PPO's.
- max abs(y_hat) reaches 7-26 std: the Beta log-score tails pass through the features.

Reading:
- **Last-layer denoising removes signal faster than noise.** The 64-d head tangent space holds
  almost none of the per-batch signal. The raw per-sample force pools through the *whole* network's
  generalization, which, weak as it is, carries more.
- **At first order the raw policy gradient is already the meta-optimum.** d = sum A_hat_i g_i is the
  unbiased gradient, and P d maximizes expected first-order improvement for a given step.
  Reweighting can only win at second order, by moving noise out of directions where J is curved.
  That needs the noise geometry of the full parameter space, which a last-layer kernel cannot see.
- **PPO's own held-out gain per update is near the noise floor** (+0.001 to +0.05e-3, sd ~0.03-0.06),
  and turns negative by 2x lr. At PPO's operating point, second-order costs already bind.

So a target construction beats PPO only by extracting information from the batch that the scalar
per-sample force discards, or by placing its noise where it costs less. Denoising the same scalar
through a fixed kernel does neither.

### 4.5 Result: the trace length is the lever (job 9776, runs/awr_update_bench_lamcombo_v1.json)

The force was the PPO surrogate on GAE advantages at several lambdas. The critic was left unchanged (it
is the snapshot's own). Table: the best held-out dJ (e-3) within KL <= 0.02 over the lr x epoch path,
scored against A_0.95 / Monte Carlo. **This table is optimistic** (red team): it takes a max over
points on the 131k trace subset, whose noise is +-0.1 in dJ and happened to favor 0.8 on all 4
snapshots. Trust the full-set figures below it.

| snapshot | lambda 0.95 (PPO) | 0.8 | 0.5 | 0 | cross-fitted combo |
|---|---|---|---|---|---|
| v16 20M | 0.124 / 0.060 | 0.260 / 0.245 | 0.245 / 0.288 | 0.221 / 0.310 | 0.251 / 0.298 |
| v16 35M | 0.029 / -0.026 | 0.098 / -0.007 | 0.124 / -0.005 | 0.131 / -0.003 | 0.128 / -0.029 |
| PPO 12M | 0.058 / 0.093 | 0.069 / 0.103 | 0.077 / 0.114 | 0.005 / 0.006 | 0.093 / 0.089 |
| PPO 24M | 0.047 / 0.033 | 0.118 / 0.159 | 0.077 / 0.109 | -0.074 / -0.125 | 0.083 / 0.096 |

Full held-out set, trainer-like endpoint (lr x1, 10 epochs), paired over 8 batches. These numbers
count batch variance only, not the shared held-out noise (dj_se ~0.02-0.04). They are the defensible
numbers:
- **lambda 0.8 minus 0.95:** +0.071 (t 2.6) at v16 20M, +0.052 (t 5.3) at v16 35M, +0.026 (t 2.1)
  at PPO 12M, +0.018 (t 0.5) at PPO 24M.
- **MC scoring on the full set:** favors 0.8 on all 4, by 0.013-0.051, pooled z ~1.5.
- **The ordering is monotone:** 1.0 < 0.95 < 0.9 < 0.8 ~ 0.5.

So: strong on the v16 snapshots and weak on PPO's. It is directionally consistent, not "1.2-3.4x
everywhere".

Other measurements:
- The cross-batch signal share of the update is 0.2-0.37 at lambda 0.5 and 0.01-0.11 at 0.95.
- First-order efficiency (shared signal per unit of Adam-geometry step) at v16 20M is 1.9 / 2.0 / 1.7 /
  1.3 / 1 / 0.42 for lambda 0 / .5 / .8 / .9 / .95 / 1, all scored against A_0.95.
  Against MC, lambda 0.5 has 5.6x MC's own efficiency.
- The optimal cross-fitted linear combination is 2.05x, no better than the best single lambda.
- c fitted intra-batch (4 env folds) fits noise and loses.

Reading:
- **This is the first construction with a consistent-sign first-order edge over PPO.** It is
  significant on the v16 snapshots only.
- **Part of the A_0.95-scored edge is rewarded critic error.** A_0.95 loads ~0.83 on nearby-state
  critic errors. At 35M, dJ - dJ_mc grows as lambda shrinks (0.95: -0.005 up to 0: +0.045). That is
  why 0 looks good under A_0.95 and not under MC.
- **Counter-evidence (training).** awr_ce_ppo_sepclip_control_v1_lam0p8_50M, with the critic *also*
  at lambda 0.8, trailed the control: 2466 vs 3674 at 2M, 4299 vs 6213 at 5M, stopped at 7.7M. v18
  keeps the critic at 0.95, so that run does not refute it. It is still the warning that an early,
  poor critic makes a short trace biased when it matters most.
- **The mechanism.** MuJoCo transitions are deterministic, so delta_t = A(s_t, a_t) + critic error.
  Every further residual in the trace adds a later sampled action's noise, which the credit for a_t
  does not need.
- **The optimum is set by critic quality.** lambda 0 wins on the v16 snapshots and loses on PPO's
  (critic bias); 0.5-0.8 is robust.
- **The GAE paper's 0.92-0.98 does not transfer.** It was chosen for whole training runs, with the
  critic trained on the same lambda.
- **The one-step bench cannot see compounding critic bias.** A shorter trace makes the actor more
  myopic about the current V, and only training runs can price that.
- **The lever is the kernel over residuals, not the weighting of the scalar.** The (w, r*) of every
  CE rule acts on a scalar that already lost this. In CE terms, the optimal f is z computed from a
  short-trace A_hat.

Test: v18 (v16 plus --actor-lambda 0.8; the critic keeps its 0.95 returns), job 9779. Its PPO twin
(ppo_short_trace_control_v1), job 9780, separates the source of the gain from the CE rule.

**Training result (rev 8).**
- **From scratch, a fixed short trace is catastrophic.** v18 and its PPO twin locked into ~5-6k gaits
  (1M: 1.3k vs v16's 3.0k; still ~5k at 19M). The earlier lambda 0.8 run, with the critic also at
  0.8, did the same.
- **From mature states it is neutral to positive.** For the CE rule it is consistently positive on
  both forks: v18 vs v16 finals +0.34k (PPO 24M) and +0.51k (v16 20M), with area gains of similar
  size. PPO is mixed.
- **So the per-update bench edge is real where the critic is good, and early critic bias dominates
  otherwise.** The optimal trace is not a constant; it tracks critic reliability.
- **v19 chooses it per iteration.** It scores each view against a critic-free reference, maximizing
  (E[g_lambda] . E[g_MC]) / |g_lambda| across env-disjoint folds, with an EMA of 0.95.
- **Bench check of that criterion (MC-scored efficiency relative to 0.95).** The best lambda is 0.5,
  0.9, 0.9 and 0.8 on the four snapshots. Efficiency is 2.1x, 1.08x, 1.12x and 1.19x of 0.95's, and
  lambda 0 is heavily penalized where the critic is biased.

### 4.6 Oracle advantages (job 9945, scripts/awr_oracle_advantage.py, runs/awr_oracle_v1.json)

MuJoCo states are exactly restorable. Setup:
- For 32k train and 16k eval states from the v16 20M and PPO 24M snapshots, A_true(s, a) comes from
  branched rollouts: the sampled action plus K=7 fresh ones, M=4 rollouts each, H=200 with a V bootstrap.
- Future action noise is coupled across branches by a counter-keyed Beta sampler (CRN).
- Physics is bitwise gymnasium, and the frozen-policy return matches the trainer (z = -0.05).

| | v16 20M | PPO 24M |
|---|---|---|
| true within-state A sd (normalized reward) | 0.031 | 0.032 |
| critic error V^pi - V, sd | 0.097 | 0.087 |
| oracle noise share at H = 25 / 200 (M=4) | 0.80 / 0.92 | 0.72 / 0.88 |
| CRN variance gain | 0.9-1.0 | 1.1-1.3 |
| R2sig of A_hat at lambda 0 / .5 / .8 / .9 / .95 / 1 | .016 / .074 / .102 / .086 / .065 / .022 | .006 / .060 / .101 / .086 / .059 / .010 |

R2sig is the noise-corrected share of A_hat's variance explained by A_true (R2 / oracle reliability).

Reading:
- **Advantage noise is the problem, and it is ~90-94% of every estimator.** At best (lambda ~0.8),
  one sampled advantage carries 10% signal variance, and PPO's 0.95 carries 6%. This independently
  confirms 4.5's ordering against a critic-free truth, not just against A_0.95 or MC.
- **Trajectories decorrelate within ~25 steps**, so CRN buys nothing. Future actions contribute the sum
  of their own true advantages, sqrt(sum (gamma lambda)^2k) x 0.031, plus chaos. The critic's error is
  3x the action signal. A short trace swaps the first for the action-dependent part of the second, and
  that is why lambda 0 carries only ~1%: V's error is not smooth at the scale of one step's action effect.
- **The update-ceiling half is inconclusive.** With a 90%-noise oracle, oracle1 is about as reliable as
  A_0.8, and every frontier difference is inside the scoring SE (~0.06-0.1e-3). A precise oracle would
  need M ~ 400 per action.
- **Structure worth exploiting.** Transitions are deterministic, so delta_t = A^pi(s_t, a_t) exactly
  under V^pi. All of A_hat's noise is therefore learnable structure: V's error, and the future actions'
  own A. Neither is irreducible sampling noise. A model fit across many batches can pool it; a per-batch
  scalar cannot. Next measurement: the R2sig a persistent A_phi(s, a) reaches against A_true, as a
  function of training batches, including the iterated future-action correction
  GAE_lambda(delta - A_phi) + A_phi.

## 5. Next steps

1. **Information the scalar force discards** (the only remaining source of a first-order edge):
   - **The trajectory structure of A_hat: measured, see 4.5.** Open: is the optimal kernel over
     residuals exponential at all? Its optimum drifts with critic quality, so a fixed lambda may be the
     wrong object. Candidates:
     - an intra-batch 1-parameter selection by cross-fold signal share (the 6-parameter combo was
       too noisy);
     - a trace that adapts to the critic's measured residual error. The critic's explained variance
       of delta is a per-batch statistic.
   - **Per-dimension structure.** One scalar credits all 6 action dims equally. With V(s) only,
     per-dim credit is unidentifiable from one sample, so there is probably no lever here without
     Q(s,a).
2. **Second-order placement of noise.** Measure which full-network directions PPO's held-out loss
   (the curvature of J) punishes: dJ's decline with lr splits by location vs concentration KL
   (kl_loc/kl_conc are logged). If one channel's KL costs return without buying signal, the
   construction should spend less of its step there.
3. The critic-bias anchor term (4.2) stays a candidate, but small.

## Log

- 2026-09-25 (rev 10): v19 finished at 11.8k (50M), tying v16; its lambda choice was noise-driven. The
  oracle advantage bench (4.6) puts every advantage estimator at <= 10% signal (best lambda ~0.8, PPO's
  0.95 at 6%). The critic's error is 3x the true action signal, and CRN does not couple HalfCheetah
  trajectories. Next: can a persistent A_phi(s, a) pool that learnable noise away?

- 2026-09-24 (rev 9): v19's first run picked views by argmax over negative, noise-level signals early
  (the noisiest view wins that argmax) and plateaued at ~5.9k, like v18. With the fallback to 0.95 it
  tracks v16. A v18 run at actor lambda 0.95 reproduced v16 exactly (it is deterministic), so v18's
  failure is the trace, not the code. The early phase is where the trace choice is most fragile.

- 2026-09-24 (rev 8): v18 trained. A fixed short trace locks into a bad gait from scratch. From
  mature states it is neutral to positive for CE (+0.3-0.5k on both forks). v19 (an adaptive trace
  chosen by critic-free cross-fold alignment with the MC gradient) is submitted.

- 2026-09-24 (rev 7): lambda views measured (4.5). A shorter actor trace (lambda 0.5-0.8) beats PPO's
  0.95 on held-out dJ at matched KL on all 4 snapshots, on both scorings. Combining views adds nothing.
  v18 and its PPO twin submitted. Red team the same day: the headline table is subset-noise-optimistic.
  On the full set the edge is significant on v16 snapshots only (t 2.6 and 5.3), weak on PPO's
  (t 2.1 and 0.5), and partly rewarded critic error. A prior run with lambda 0.8 on both actor and
  critic lagged early.

- 2026-09-24 (rev 6): compat refuted on the bench: it loses to PPO on all 4 snapshots, and last-layer
  denoising removes signal faster than noise. At first order the raw PG is already meta-optimal, so
  a win needs new information or a better placement of noise. Next steps rewritten.

- 2026-09-24 (rev 5): compat red team. The natural-step claim now applies to the endpoint only; the
  penalty is chosen by held-out alignment (with the inf limit); targets have unit std; P is Adam's.
  Bench job 9766.

- 2026-09-24 (rev 4): section 4.3, the closed-form return-maximizing construction: cross-fitted
  compatible (natural-gradient) advantages as CE targets. Bench job 9764.

- 2026-09-24 (rev 3): channel measurement. Concentration is not less reliable than location. Updates are
  2-20% signal in every channel. Identified CE's missing baseline invariance, with the critic's
  state-predictable bias acting as a per-state KL coefficient, as the one systematic CE-vs-PPO term.

- 2026-09-24 (rev 2): red team. Section 4's claim that concentration is noisier was false in natural
  units; restated around signal, calibration slopes and tail credit. Dropped the Gauss-Newton
  converged-regime claim (the w-weighted empirical Fisher is singular in the overparameterized
  regime). Fixed section 3.3's direction and the rewrite constants.
- 2026-09-24: first version. Written after the fork, seed and reach studies established that
  linear-force rules tie PPO and that rules differ mainly in the concentration channel.
