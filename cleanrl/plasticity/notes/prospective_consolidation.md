# Prospective consolidation: an energy-based streaming optimizer

Line opened 2026-09-11. Question: can a streaming optimizer that spends "energy"
the way synapses do (pay for persistent change, not for transient exploration)
and that learns toward its own prediction of its future rather than the raw
realized future, beat tuned AdamW on noisy streams and generalize better?

Code: `cleanrl/plasticity/optimizer_proxy_model_v9.py` (families `tier_adamw`,
`tier_polar`, `tier_vel_adamw`, controls `look_adamw`, `scalar_adamw`),
`optimizer_proxy_eval_v9.py` (per-family grids, auxiliary state), plan
`benchmarks/plasticity/optimizer_proxy_v9_plan.json` (generator
`make_optimizer_proxy_v9_plan.py`), contracts `tests/test_optimizer_proxy_v9.py`.
Full literature surveys (197 verified papers) are in `../reference/lit_*.md`.

## 1. What the literature says brains do with energy during learning

Recurring, quantitatively supported principles (strongest source each; details
and the rest of the citations in `reference/lit_brain_energy.md` and
`reference/lit_energy_learning.md`):

1. **The metabolic cost of learning is the path length of weight change,
   sum |dw|, not the weight magnitude.** Sequential learning random-walks the
   weights so path length exceeds the straight-line distance by 20-900x;
   "synaptic caching" (a cheap transient component consolidated into an
   expensive persistent one only when it crosses a threshold) recovers up to
   10x (Li & van Rossum 2020 eLife; Pache & van Rossum 2023 Curr Opin
   Neurobiol; Karbowski 2019 J Neurophysiol). Consolidation is literally
   gated by energy state in flies (Placais & Preat 2013 Science; Placais et
   al. 2017 Nat Commun).
2. **Update only when the error exceeds the noise floor; prune plasticity
   competitively.** Lazy learning updates only on misclassified samples at
   no accuracy cost (Pache & van Rossum 2023 arXiv); top-k competitive
   plasticity with a churning mask keeps accuracy (van Rossum & Pache 2024
   PLoS CB).
3. **Two-tier stores with slow consolidation give power-law forgetting and
   near-linear capacity** (Benna & Fusi 2016 Nat Neurosci; Fusi, Drew &
   Abbott 2005 Neuron).
4. **Plasticity rate is proportional to posterior uncertainty**, which
   shrinks with evidence and grows with drift; the same signature emerges
   from precision costs alone (Aitchison et al. 2021 Nat Neurosci; Jegminat
   et al. 2022 PLoS CB; Malkin et al. 2024 eLife).
5. **Infer the target state before changing weights.** Prospective
   configuration relaxes activity to its post-learning configuration first
   and then changes weights consistently; this is what wins at batch size 1,
   under drift, and under interference (Song et al. 2024 Nat Neurosci).
   Predictive-coding inference is an adaptive trust-region step in activity
   space (Innocenti, Singh & Buckley 2023), and inference learning is an
   implicit (proximal) gradient step (Alonso et al. 2022 NeurIPS).
6. **Neurons learn toward their own look-ahead prediction of their state**
   (prospective coding), which removes lag and permits continuous updating
   (Urbanczik & Senn 2014 Neuron; Brea et al. 2016 PLoS CB; Haider et al.
   2021 NeurIPS; Senn et al. 2024 eLife).
7. **Activity/weight energy penalties are abstraction priors**: with
   nonnegativity they provably disentangle (Whittington et al. 2023 ICLR),
   yield minimal low-dimensional codes matching PFC (Stroud et al. 2025
   eLife), and produce predictive coding by themselves (Ali et al. 2022
   Patterns). These act on representations, not on the optimizer.
8. **Sleep downscales synapses proportionally, sparing the strong ones**
   (de Vivo et al. 2017 Science; Tononi & Cirelli 2014 Neuron).

On the optimizer side, the survey in `reference/lit_streaming_optimizers.md`
finds that Bayesian-filter optimizers (Aitchison 2020 AdaBayes; Khan & Rue
2023; IVON, Shen et al. 2024) carry the noise model but have no streaming
evidence, the streaming-RL optimizers (ObGD, Elsayed et al. 2024; SwiftTD,
Javed et al. 2024) have batch-1 evidence but no noise model, and **no
existing optimizer estimates its own per-parameter forgetting rate from the
innovation sequence**; that gap is exactly where item 4 above points.
Zhou et al. 2020 NeurIPS and Yang, Tang & Tu 2023 PRL explain the Adam/SGD
generalization gap by Adam's per-coordinate whitening of gradient noise
(it destroys the anisotropy that biases SGD toward flat minima), and Balles
& Hennig 2018 isolate Adam's harm to its sign/normalization, not its
momentum.

### Why "learn toward your own prediction" works (reference/lit_self_prediction.md)

A prediction helps exactly when it is closer to E[target | input] than the
realized sample (Menon et al. 2021 ICML, distillation bias-variance; Daley
et al. 2024 ICML, compound returns; Polyak & Juditsky 1992, averaged
iterates are asymptotically efficient). Bootstrapped self-prediction avoids
collapse under a two-timescale asymmetry, a fast predictor and a slowly
moving, gradient-free predicted quantity (Tang et al. 2023 ICML; BYOL's
target EMA). Holding a lagged copy turns an ill-conditioned operator into a
contraction (Fellows et al. 2023 ICML). Refitting to your own prediction is a
spectral low-pass that kills small-eigenvalue (noise) directions first
(Mobahi et al. 2020 NeurIPS), and early learning fits signal before noise
(Arpit et al. 2017; ELR, Liu et al. 2020). Optimizer-side, stepping toward a
short-horizon self-forecast strictly lowers the noisy-quadratic variance
fixed point (Lookahead, Zhang et al. 2019 NeurIPS); stepping toward a linear
extrapolation of yourself does not survive minibatch noise (Anderson
acceleration, Pasini et al. 2021; Adam-moment extrapolation needs a
verifier).

## 2. The rule

Two tiers per parameter. The deployed, consolidated tier phi is the
optimizer's prediction of its own future; the transient tier z is the noisy
realized future. z runs the fast optimizer (AdamW, or polar for hidden
matrices) for one block, then at the boundary:

    d    = z - phi                       innovation (block displacement)
    c   <- g c  + (1-g) d * d_prev       lag-1 innovation covariance (EMA)
    nu  <- g nu + (1-g) d^2              innovation power (EMA)
    K   <- clamp(K * exp(eta * c/nu), kmin, 1)
    phi <- z - (1-K) d                    = phi + K d
    z   <- phi                            reset (transient decays to persistent)

Velocity arm: d = z - phi - vel, phi <- phi + K d + vel, vel <- vg vel +
(1-vg)(phi' - phi); phi follows its own predicted trajectory where the
measurement is untrusted (constant-velocity model).

Why the lag-1 autocorrelation. Under the local-level Kalman model the gain
is optimal exactly when successive innovations are white. With a restoring
gradient (curvature a > 0 along the parameter), consolidating noise moves
phi away from the optimum and the next block's displacement points back:
Cov(d_t, d_{t-1}) = a^2 (1-Ka) Var(e) - a K R, which is negative when the
parameter is jittering around a fixed point and positive when the truth is
drifting or still distant. So K shrinks where displacements alternate and
grows where they persist. This is scale-free (rho = c/nu), per parameter,
temporal (allocation of step size in time, which `FAMILY.md` records as the
only effect in this family that a correctly-controlled oracle did not
already dominate), and post-fast-optimizer, so Adam cannot cancel it
(MEASURED_CONSTRAINTS C2).

Energy interpretation: consolidated path length sum |K d| is spent only on
displacements with evidence of persistence; transient jitter is discarded at
zero persistent cost. This is synaptic caching with the threshold replaced by
an evidence gain.

Why this is not an Adam statistic. Adam's m and v summarize gradients inside
a block; when a block reuses one batch (8 epochs here; 10 epochs on one
32768-sample minibatch in the PPO base), every within-block gradient has the
same sign and the cross-batch consistency of realized displacements is
unobserved by Adam. The gain reads exactly that.

Mapping to the "optimizer-side JEPA" intuition: phi is the predictor of the
future, z is the realized future, and the learning signal that reaches the
deployed model is the fraction of the realized displacement that the
predictor's own error statistics say is predictable. Dense structure comes
from persistence across blocks, not from any single sample.

## 3. Controls and what would falsify the hypothesis

- `look_adamw`: eta = 0, K = k0 in {0.25, 0.5, 0.75}. Same two-tier
  structure, no adaptation (uniform Lookahead). If it matches the adaptive
  arms, the gain is a structural/level effect, not evidence-driven.
- `scalar_adamw`: one K per candidate adapted from the parameter-mean rho.
  Same time profile, spatial dispersion erased. If it matches `tier_adamw`,
  the per-parameter dispersion carries nothing (the C5 trap).
- `adamw`, `polar`: independently tuned on the retained v8 2036-config grid.
- Every arm sweeps its own lr; tier arms sweep eta {0.3, 1}, gamma {0.9,
  0.98}; kmin 0.03, k0 1 (starts as its fast tier).
- Promotion bar unchanged from v8: >= 20% lower sustained held-out clean
  excess risk than tuned AdamW, no substantial regression elsewhere, runtime
  accounted (tier cost is elementwise, well under polar's 2.3-2.7x).

## 4. Protocol and jobs

Same v8 regimes, model, noisy-validation locking, held-out scoring. Block =
epochs for the reuse regimes (one round), 32 samples for `iid_online`.
Deployment, checkpoints and bandit rollouts occur at block boundaries, where
weights equal phi. Contracts: 18 tests (fast-tier identity at eta = 0, k0 =
1; whitening direction; exact state algebra; scalar control; off-boundary
passthrough; compiled vs eager; real learner capture/audit/replay through
boundaries).

mlq 6328 contracts | 6329 drifting_reuse | 6330 clipped_bandit | 6331
iid_online (cancelled before start; max-parallel 1, one attempt, 120m each).
The full 4680-config tier grids made those jobs run 8-10 minutes, against a
~2 minute budget per run, so the remaining work uses
`benchmarks/plasticity/optimizer_proxy_v9_lean_plan.json`: tier families fix
beta1/beta2/weight decay/head scale to their fast tier's v8-locked
confirmation config for the regime and sweep the 13-point lr axis times
eta {.3, 1} x gamma {.9, .98} (52 configs; look_adamw 13 lrs x k0 {.25, .5,
.75}), one family per job via `--families` (20m limit, one attempt,
chained with --after-terminal). Jobs: iid_online 6352 adamw, 6353 polar,
6354 look, 6355 scalar, 6356 tier_adamw, 6357 tier_polar, 6358 tier_vel;
clipped_bandit 6359-6363 and drifting_reuse 6364-6368 (look, scalar,
tier_adamw, tier_polar, tier_vel in that order).

## 5. Results

### drifting_reuse (job 6329, `runs/OptimizerProxy__v9_drifting_reuse__1__1789189577292088846`)

Sustained held-out clean excess risk, selected on noisy validation, test never reranks:

| arm | sustained | endpoint | vs adamw | train s | selected | gain mean / std (checkpoints) |
|---|---|---|---|---|---|---|
| adamw | 0.30341 | 0.46868 | 0 | 23.1 | lr 3e-4, b1 .95, b2 .99, wd .1, head 1 | |
| polar | 0.28520 | 0.43005 | -6.00% | 37.2 | lr 2e-3, b1 .9, b2 .999, wd .1, head .5 | |
| look_adamw (K=.25, edge) | 0.29494 | 0.46462 | -2.79% | 53.5 | lr 2e-3, b1 .95, b2 .99, wd .1, head 1 | .25 / 0 |
| scalar_adamw | 0.30341 | 0.46868 | 0.00% | 80.5 | = adamw, eta .3, g .9 | 1.0 / 0 |
| tier_adamw | 0.30342 | 0.46860 | 0.00% | 81.4 | = adamw, eta .3, g .9 | 1.0 / 0 |
| tier_polar | 0.28377 | 0.42117 | -6.47% | 124.4 | lr 2e-3, b1 .95, b2 .999, wd .1, head .5, eta .3, g .98 | 1.0 / 0 |
| tier_vel_adamw | 0.29906 | 0.46492 | -1.43% | 90.1 | lr 5e-4, b1 .5, b2 .999, wd .1, head .5, eta .3, g .9 | .72 -> .45-.52 / .29-.37 |

Reading. The local-level whitening rule never fired: under sustained drift
(teachers change every 128 rounds, input mean moves continuously) every
parameter's consecutive block displacements are positively correlated, rho is
near +1 everywhere, K is clamped at 1 from the first boundaries, and the
adaptive arms reduce exactly to their fast tiers (tier_adamw = adamw to 5
digits; tier_polar within noise of polar). This is the rule behaving as
derived: a still-moving parameter should be consolidated fully. It also
means the plain rule cannot help in a regime where nothing converges. Only
the velocity arm measured its innovation relative to the predicted
trajectory, which removes the persistent component; its gains adapted (mean
~0.5, dispersion ~0.37) and it gained 1.4%, with eta and gamma at the low
grid edge. The uniform Lookahead control at K = 0.25 with a 6.7x higher fast
learning rate gained 2.8% (k0 at the grid edge): a structural/level effect
of the two-tier reset with a hot fast tier, not evidence-driven. Tier arms
used a reduced base grid (b1 5 values, b2 3, wd 2, head 3); several
selections sit at its edges, so the comparison is conservative for them.

### clipped_bandit (job 6330, `runs/OptimizerProxy__v9_clipped_bandit__1__1789190148769412264`)

Status caveat: every family trained, was selected and was test-scored
(`status: locked_test_scoring`, all eight selections carry test curves), but
the run ended with the source-change guard firing because I edited the
argument parsing of `optimizer_proxy_eval_v9.py` (a `--families` override and
the output tag) while the job was running. Nothing on the compute path
changed, so the numbers below are the real measurements, but the run is not
protocol-clean; the lean per-family reruns (jobs 6359-6363) replace it.

| arm | sustained | endpoint | vs adamw | train s | selected | gain mean / std (checkpoints) |
|---|---|---|---|---|---|---|
| adamw | 0.24398 | 0.38615 | 0 | 37 | lr 2e-3, b1 .5, b2 .99, wd .1, head 1 | |
| polar | 0.22351 | 0.33289 | -8.39% | 51 | lr 3e-3, b1 .9, b2 .999, wd .1, head .5 | |
| look_adamw (K=.5) | 0.24513 | 0.36433 | +0.47% | 81 | lr 5e-3, b1 .95, b2 .9, wd .1, head .5 | .5 / 0 |
| scalar_adamw | 0.24363 | 0.36718 | -0.14% | 117 | lr 1.4e-3, b1 .5, b2 .99, wd .1, head 2, eta .3, g .9 | 1.0 / 0 |
| tier_adamw | 0.24676 | 0.36810 | +1.14% | 119 | lr 2e-3, b1 .5, b2 .99, wd .1, head 2, eta .3, g .98 | .84 -> .79 / .29 -> .34 |
| tier_polar | 0.22034 | 0.33252 | -9.69% | 156 | lr 3e-3, b1 .9, b2 .99, wd .1, head .5, eta .3, g .98 | 1.0 / 0 |
| tier_vel_adamw | 0.24440 | 0.38405 | +0.17% | 120 | lr 2e-3, b1 .5, b2 .99, wd .1, head 2, eta .3, g .98 | .75 -> .65 / .34 -> .38 |

Reading. Here the plain rule does fire on the AdamW tier: a third of the
parameters see alternating block displacements under the 8-epoch clipped
surrogate and their gains fall (mean .79, dispersion .34), yet the sustained
risk is 1.1% worse than AdamW, and the scalar control, which keeps the same
profile without dispersion, is neutral. So in this regime per-parameter
whitening of the displacement sequence identifies real jitter but removing
it does not lower held-out risk: the jitter that Adam leaves in these
parameters is not what limits the clipped-bandit error, and the 8 reuse
epochs already act as a strong within-block averager. tier_polar again sits
at K = 1 (polar's normalized steps are persistent block to block) and its
1.3% over polar is a beta2 swap inside the grid, not consolidation. The
velocity arm adapts but is neutral. Combined with drifting_reuse: the
local-level rule either does not fire (sustained motion) or fires without
benefit (bandit); the remaining regime where it should matter is iid_online
with B = 1, where blocks of 32 single-sample steps are dominated by gradient
noise and the fixed point is stationary.

### iid_online (lean plan, jobs 6352-6358; one family per run)

Reference AdamW `runs/OptimizerProxy__v9_iid_online_adamw__1__1789192303214995895`.
Run time is startup plus training, 2036-config families first, 52/39-config
tier families after.

| arm | sustained | vs adamw | run s | selected | gain mean / std at end |
|---|---|---|---|---|---|
| adamw | 0.24028 | 0 | 107 | lr 3e-4, b1 .995, b2 .9, wd 0, head 1 | |
| polar | 0.18802 | -21.75% | 207 | lr 1e-4, b1 .995, b2 .95, wd 0, head 4 (edge) | |
| look_adamw (K=.25, edge) | 0.23024 | -4.18% | 15 | lr 1.4e-3, adamw betas | .25 / 0 |
| scalar_adamw | 0.24024 | -0.02% | 16 | = adamw, eta .3, g .9 | 1.0 / 0 |
| tier_adamw | 0.24017 | -0.05% | 15 | = adamw, eta .3, g .9 | 1.0 / 0 |
| tier_polar | 0.18785 | -21.82% | 44 | = polar lock (head 8), eta .3, g .9 | 1.0 / 0 |
| tier_vel_adamw | 0.24006 | -0.09% | 18 | = adamw, eta .3, g .9, vg .9 | 1.0 / 0 |

Reading. The rule did not fire once in iid_online either: every one of the
52 tier_adamw configurations (eta up to 1, gamma .9 and .98, all 13 learning
rates) ends with mean gain 1.000 and zero dispersion. The cause is not the
landscape but the fast tier's own colouring. AdamW's locked beta1 is .995, a
momentum time constant of 200 steps, while a block is 32 steps: consecutive
block displacements are dominated by the same momentum state and are
therefore positively correlated for every parameter regardless of whether
the parameter is jittering around its fixed point. The local-level model
assumes the fast tier's displacements are a noisy observation of the
parameter's drift; with heavy momentum they are a low-pass filtered
observation, so the whitening test measures the optimizer, not the
parameter. The same mechanism explains drifting_reuse (beta1 .95, block =
8 x 1024 steps, so momentum is not the cause there; sustained motion is)
and why the bandit tier only fired with beta1 .5. Even the velocity arm stays at K = 1 here: its EMA of consolidated
steps lags the momentum-driven displacement, so the residual is still
positively correlated. It adapted only where beta1 was small (bandit,
.5) or the block was long relative to the momentum horizon (drifting). Lookahead at K = .25 with a 4.7x hotter
fast tier is again the structural gain (-4.2%), and polar remains the
frontier by a wide margin (-21.8%).

### Lean same-grid reruns: clipped_bandit (6359-6363, refs 6371/6372) and drifting_reuse (6364-6368, refs 6373/6374)

Every arm on the 13-point lr axis at its fast tier's v8-locked betas; each
run 6-19 s. "vs" is against the same-grid fast tier (adamw or polar).

| regime | adamw | look_adamw | scalar | tier_adamw | tier_vel | polar | tier_polar |
|---|---|---|---|---|---|---|---|
| clipped_bandit | 0.24416 | 0.25312 (+3.7%, K=.75) | 0.24850 (+1.8%, K=1) | 0.24928 (+2.1%, K .80 +- .35) | 0.24294 (-0.5%, K .66 +- .40) | 0.22294 | 0.22249 (-0.2%, K=1) |
| drifting_reuse | 0.30341 | 0.29494 (-2.8%, K=.25) | 0.30341 (0, K=1) | 0.30342 (0, K=1) | 0.30342 (0, K=1) | 0.28520 | 0.28569 (+0.2%, K 1.0 +- .03) |
| iid_online | 0.24028 | 0.23024 (-4.2%, K=.25) | 0.24024 (0, K=1) | 0.24017 (0, K=1) | 0.24006 (0, K=1) | 0.18802 | 0.18785 (-0.1%, K=1) |

The velocity arm's 1.4% in the full-grid drifting run came with beta1 .5;
on the locked beta1 .95 it never fires and equals AdamW. tier_polar never
leaves K = 1 in any regime (polar's orthogonalized steps are persistent
block to block by construction) and is polar to within selection noise.

## 6. Conclusion for the whitening rule, and what survives

The hypothesis of section 2, that a per-parameter gain adapted on the
lag-1 autocorrelation of the fast tier's block displacements lowers
sustained held-out risk, is falsified in all three regimes with a clear
mechanism:

1. The displacement sequence of a momentum optimizer is coloured by the
   optimizer, not by the parameter's relation to its fixed point. With the
   locked AdamW betas (beta1 .995 on iid_online, .95 on drifting_reuse) and
   with polar's orthogonalized momentum, consecutive block displacements are
   positively correlated for every parameter and K is pinned at 1: the arms
   reduce to their fast tiers exactly. The scalar and velocity controls
   inherit the same blindness.
2. Where the coloring is weak (bandit, beta1 .5) the rule does fire, with a
   third of the gain mass below .5, and sustained risk rises 2.1%; the
   scalar control with the same profile and no dispersion rises 1.8%. So
   removing the displacement components the rule labels as jitter removes
   signal the deployed tier needed, and per-parameter dispersion adds
   nothing on top. This matches the family's earlier finding that
   per-parameter state-conditional plasticity is null or harmful (C5).

What survives is structural and already known: a uniform K < 1 on a 4-7x
hotter AdamW tier (Lookahead / Polyak variance reduction) gains 4.2% on
iid_online and 2.8% on drifting_reuse and loses 3.7% on the bandit, with K
at the grid edge .25; polar is the frontier in every regime (-21.8%, -8.4%,
-6.0%). Neither meets the 20% promotion bar as a two-tier effect on AdamW,
and nothing here changes the retained frontier (job 5895).

On the framing: "the optimizer learning toward its own prediction of the
future" cannot be implemented as a per-parameter gain on the fast tier's
weight displacements, because those displacements carry the fast tier's
own dynamics and the gain has no way to separate them from the landscape.
An implementation faithful to the dense-prediction idea would have to
measure the innovation in function space (the fast tier's outputs on the
block's inputs against the deployed tier's), which is a different
experiment with its own cost; it is not started here. v10 tests the one
cheap follow-up the data supports: whether the uniform two-tier reset
stacks with polar (`optimizer_proxy_model_v10.py`, `look_polar`, k0 axis
extended to .1; jobs 6375 contracts, 6376-6381 scenario x family).

## 7. v10: does the uniform two-tier reset stack with polar?

Jobs 6375 (contracts, 10 passed) and 6376-6381; lean per-scenario grids, 13
lrs x k0 {.1, .25, .5, .75} at the locked betas, 6-27 s per run. References
are the same-grid v9 fast tiers.

| regime | adamw | look_adamw (v10) | polar | look_polar (v10) |
|---|---|---|---|---|
| iid_online | 0.24028 | 0.22704 (-5.5%, K=.1 edge, lr 5e-3 = 17x) | 0.18802 | 0.18013 (-4.2%, K=.1 edge, lr 1e-3 = 10x) |
| clipped_bandit | 0.24416 | 0.25424 (+4.1%, K=.75) | 0.22294 | 0.22065 (-1.0%, K=.75, lr 5e-3) |
| drifting_reuse | 0.30341 | 0.29005 (-4.4%, K=.1 edge, lr 1e-2 = 33x) | 0.28520 | 0.28495 (-0.1%, K=.75) |

Best validation per k0 for look_polar: iid_online improves monotonically as
K falls (1.108 at .1, 1.116 at .75); bandit and drifting improve
monotonically as K rises toward 1 (.58 vs .83; 1.347 vs 1.425), i.e. the
reset costs polar in the reuse regimes and the small "gains" there are
selection noise around K = 1. Reading: the two-tier reset is a stationary-
regime variance reducer. It stacks with polar only where the fixed point
is stationary (iid_online, -4.2%), and there the selected K sits at the
grid edge with a 10-17x hotter fast tier, so the effect is still growing
toward a Polyak-averaged hot fast tier. Jobs 6383-6385 extend k0 to
{.03, .05, .1} (plan `optimizer_proxy_v10_ext_plan.json`) for the three
edge cases. None of this is an energy or prediction effect: it is the
lag-as-contraction / Lookahead variance fixed point from the
self-prediction survey (Fellows 2023; Zhang 2019), and it does not reach
the 20% promotion bar as an increment on the retained frontier.

Extension (jobs 6383-6385, k0 {.03, .05, .1}, 3-30 s per run). Combined
with the v10 grid the k0 axis is now interior for all three edge cases:

| regime / arm | k0 .03 | .05 | .1 | selected | sustained risk | vs fast tier |
|---|---|---|---|---|---|---|
| iid_online look_adamw | worse | worse | best | K .1, lr 5e-3 | 0.22704 | -5.5% |
| iid_online look_polar | 1.106 | 1.105 (best) | 1.108 | K .05, lr 2e-3 | 0.18048 | -4.0% |
| drifting_reuse look_adamw | 1.59 | 1.45 | 1.40 (best) | K .1, lr 1e-2 | 0.29005 | -4.4% |

(final-checkpoint validation shown for the k0 profile). The two-tier reset
saturates at 4-5.5% over its fast tier in the stationary and drifting
regression regimes and costs 1-4% in the bandit. Retained frontier
unchanged; no promotion.

## 8. Function-space consolidation: cheap falsification first

Before building anything into the compiled protocol, a one-step
counterfactual on plain fast-tier trajectories
(`function_space_consolidation_diag_v1.py`; contracts job 6394, 2 passed;
diagnostic job 6395, 118 s; `runs/FunctionSpaceDiag__v1__1__1789195548189881588`).
At each of the 2048 block boundaries of iid_online (block 32), d = z - phi is
split into the row-space component of the output Jacobian at phi over the
last 1/4/16 blocks (rank 32/128/512 of 5377) and its null-space remainder,
and the clean held-out risk (8192 fresh inputs, namespace 7) is evaluated at
phi, z, phi + d_par and phi + d_perp. The trajectory is never altered.
Second-half means of the per-block risk change:

| fast tier (risk) | full move | projected h1 / h4 / h16 | null-space only h1 / h4 / h16 | projected beats full h1 / h4 / h16 | energy in row space h1 / h4 / h16 |
|---|---|---|---|---|---|
| adamw locked (0.214) | -7.1e-5 | -3.1e-4 / -2.2e-4 / -8.7e-5 | +5.9e-5 / +1.5e-4 / +2.2e-5 | 63% / 78% / 65% | 13% / 25% / 37% |
| polar locked (0.164) | -1.4e-5 | -3.0e-4 / -2.1e-4 / -2.2e-5 | +2.1e-4 / +2.4e-4 / +1.8e-5 | 67% / 87% / 61% | 9% / 26% / 42% |
| adamw hot 5e-3 (3.73, degenerate at K=1) | +9.9e-4 | -6e-5 / +8.5e-4 / +1.5e-3 | +8.1e-4 / +3.0e-4 / +5.8e-4 | 51% / 46% / 43% | 0 / 0 / 0 |
| polar hot 2e-3 (1.27, degenerate at K=1) | -8.1e-5 | -5.4e-2 / -1.1e-2 / -7.6e-4 | +4.4e-2 / +2.6e-2 / +2.1e-3 | 91% / 80% / 63% | 5% / 17% / 27% |

Reading. For both locked fast tiers, 87-91% of each block's displacement
energy lies in the null space of the block's own output Jacobian, and that
part raises held-out risk on average; the row-space part alone lowers risk
4-20x more per block than the full move and wins on 63-87% of blocks. This
is what the dense-prediction framing predicts: the fast tier's weight move
is a noisy observation, its output change on the data it saw is the dense
part. The hot-lr trajectories are degenerate at K = 1 (saturated, Jacobian
near zero for AdamW) and are only informative in closed loop. The test
passed, so the closed loop (`function_space_consolidation_diag_v2.py`, job
6398: 16 trajectories, open and closed at h = 1/4/16 for the four fast
tiers, projection applied every block, moments kept) decides whether the
one-step gain survives the changed trajectory.

Closed loop (job 6398, 121 s, `runs/FunctionSpaceDiag__v2__1__1789195752225227371`).
Sustained clean held-out risk over 16 checkpoints, projection applied at
every boundary, fast tier restarted from the projected weights:

| fast tier | open (K=1) | closed h1 | closed h4 | closed h16 | path energy h1 / h4 / h16 |
|---|---|---|---|---|---|
| adamw locked 3e-4 | 0.24724 | 0.37901 (+53%) | 0.29040 (+18%) | 0.25060 (+1.4%) | x.21 / x.29 / x.37 |
| polar locked 1e-4 | 0.19303 | 0.34179 (+77%) | 0.23346 (+21%) | 0.20393 (+5.6%) | x.09 / x.26 / x.42 |
| adamw hot 5e-3 | 2.515 (degenerate) | 0.775 | 1.402 | diverged | |
| polar hot 2e-3 | 1.126 (degenerate) | 0.676 | 1.217 | 1.328 | |

Reading. The one-step gain does not survive the changed trajectory. The
null-space component of a block's displacement is harmful for the risk
right now but is what later blocks build on: discarding it every block
spends 3-10x less path energy and learns slower, and the loss shrinks
monotonically as the projection rank grows toward the identity. The hot
tiers improve over their degenerate open runs but stay 3-4x above the
locked open tiers, so no learning rate makes the hard projection
competitive; at most it would recover a Lookahead-sized effect at an
interior rank. Under the rule fixed before the run (projected must beat
open in closed loop for at least one history), function-space
consolidation as a Jacobian row-space projection is dead. Cost of the
whole test: two diagnostic scripts, two 2-minute jobs.

What this pins down for the framing. The fast tier's weight displacement
is not "noise plus signal" separable by what the block's data can see:
the invisible part is exploration that later data justifies. A
consolidation rule that wants to be energy-minimal in function space
would have to keep the null-space part as transient state and consolidate
it only once later inputs certify it, which is a memory of unexplained
displacement, not a projection. That is a different and heavier design
and is not started here.
