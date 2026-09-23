# RethinkD: reapproach from the original goal and the collected papers

Goal (verbatim intent): a streaming optimizer that handles noisy data better than Adam/SGD by
reducing an energy in a streaming manner, so that what is learned is more abstract, more memory-like
and generalizes better; one reading is a learner that optimizes toward its own denser prediction of
the future rather than the noisy realized future (an optimizer-side JEPA).

Code for the one test this document pre-registers: `cleanrl/plasticity/structured_generalization_diag_v1.py`,
contracts `tests/test_structured_generalization_diag_v1.py`. Literature: `../reference/lit_*.md`.

## 1. What the papers say brains do (the four surveys, compressed to rules)

| Rule | Source | Optimizer reading |
|---|---|---|
| Energy is path length, sum over time of abs(dw); transient stores commit to persistent weights only past a threshold (synaptic caching, cascades) | lit_brain_energy | Move persistent weights only where displacement evidence is consistent |
| Infer before learn: relax activity toward a configuration consistent with the target, then learn toward that (prospective configuration, iPC, latent equilibrium) | lit_energy_learning, lit_self_prediction | Learn toward the network's own prospective state; per-sample (I + J J^T)^-1 error preconditioning; implicit / proximal SGD |
| Plasticity proportional to posterior uncertainty (Kalman gain, AdaBayes, VOGN/IVON) | lit_streaming_optimizers Thread 1 | Per-parameter step = posterior variance, no square root, per-example g^2 valid at batch 1 |
| Adam's per-coordinate whitening isotropizes gradient noise and lightens its tails; the sign, not the variance adaptation, harms generalization; Adam's real edge is invariance to per-parameter gradient scale (rare features) | Thread 5 | Keep the SGD direction, shrink by SNR (M-SVAG); noise anisotropy is what finds flat basins |
| Self-prediction works as conditional-expectation denoising with a slow teacher; extrapolation fails under noise | lit_self_prediction | Slow tier is a denoised target only if it is actually better than the fast iterate |
| Energy penalties act as abstraction priors (Whittington 2023, Stroud 2025, Ali 2022) | lit_brain_energy | Path-length or weight-norm priors change WHAT is learned, visible only out of distribution |

## 2. What this family has measured (all seed 1, v8 protocol unless stated)

| Mechanism | Result |
|---|---|
| Polar (Muon-style five-step) hidden direction, AdamW head | Frontier: -16.8% / -7.1% / -9.0% clean excess risk vs AdamW (iid_online / drifting_reuse / clipped_bandit), job 5895 |
| Per-layer RMS normalization without the polynomial (`matrix_rms`) | +3.6% / +0.8% / +4.6%: per-layer scale alone is not the gain; the spectral direction is |
| Adam-preconditioned then polar (`adam_rms`) | -1.2% / +0.1% / +2.5% |
| v9 prospective consolidation, per-parameter innovation-whitened gains (three families, two controls, lean and full grids) | Null or harmful everywhere; momentum coloring pins the gain at its floor where beta1 is long and it hurts (+2.1%) where beta1 is short |
| v9/v10 uniform Lookahead over AdamW and polar | 4 to 5.5% in the regression regimes, harmful in the bandit; below the 20% bar |
| Function-space (Jacobian row-space) consolidation, one step and closed loop | One-step positive, closed loop +53%/+77% at one block of history: null-space displacement is exploration that later data justify |
| RethinkA/B/C per-perceptron predictability gates | Win only on heteroscedastic streams (t about -8); null or harmful on homoscedastic streams; the conditional-mean half failed in every form |
| Historical constraints C2, C3, C5 | Pre-Adam weighting is cancelled by Adam; Adam's 1/sqrt(v) amplifies distractor coordinates (selectivity 6.2 vs SGD 7.4); any uniform component is a learning-rate change |

## 3. A finding derivable from data already on disk: the proxy's excess risk is bias and tracking, not variance

Per-checkpoint clean excess risk of the locked AdamW and polar trajectories, confirmation job 5895 (16
right-endpoint checkpoints, first to last):

```
iid_online      adamw  0.354 0.297 0.272 0.267 0.230 0.249 0.242 0.197 0.197 0.187 0.175 0.173 0.170 0.196 0.175 0.168
                polar  0.308 0.262 0.217 0.230 0.191 0.200 0.201 0.154 0.157 0.142 0.150 0.144 0.141 0.156 0.155 0.144
drifting_reuse  adamw  0.253 0.408 0.199 0.406 0.291 0.636 0.315 0.460 0.271 0.414 0.228 0.363 0.184 0.351 0.223 0.469
clipped_bandit  adamw  0.186 0.307 0.168 0.265 0.221 0.518 0.221 0.383 0.196 0.300 0.162 0.316 0.154 0.239 0.170 0.370
```

* iid_online: the risk still halves across the horizon and is falling at the end (0.170 to 0.168 over
  the last quarter with checkpoint jitter of about 0.02). The sustained mean is dominated by the
  decaying bias of a learner that has not converged; the jitter, which is the only part any
  averaging or consolidation device can remove, is 10 to 15% of the level, and a slow tier pays a
  lag against a trend that is still falling.
* drifting_reuse and clipped_bandit: the risk alternates 2x between consecutive checkpoints in
  phase with the teacher interpolation and the sinusoidal input shift. The sustained mean is
  tracking error. A slower deployed tier lags a moving target; this is why Lookahead hurt in the
  bandit.

Consequences, derived without new compute:

1. Every variance-side device tested in this family (per-parameter consolidation gains, Lookahead,
   function-space projection, an EMA of iterates, a slow self-prediction target) has at most 10 to
   15% headroom in iid_online and negative headroom under drift. The v9, v10 and function-space nulls
   were predictable from these curves. A deployed EMA of iterates is dead by the same derivation and
   is not run.
2. The lever in all three regimes is progress per sample at fixed injected noise, which is a direction
   and credit-assignment problem. That is exactly what polar is (matrix_rms shows the gain is the
   spectral direction, not per-layer scale).
3. Prospective configuration's per-sample (I + J J^T)^-1 preconditioning is near-uniform across samples
   for a tanh MLP on Gaussian inputs (abs(J(x))^2 varies little), so it is a learning-rate change
   (C5), and with the locked 200-step momentum the per-sample output step is about 0.008 of the
   residual, so cross-layer interference inside one step is second order. Dead by derivation in this
   harness; it fires only with large per-sample steps.
4. A conditional-mean self-prediction target (learn toward E[r | x] estimated by a shadow model) is
   dead by the RethinkA/B/C precedent: the shadow model faces the same noise with the same variance.

## 4. The property the goal names is not measurable in the current proxy

The v8 proxy scores in-distribution clean excess risk against a dense random teacher of the student's
own architecture with Gaussian inputs. Nothing in that task can be abstracted: every input is
relevant, the target is exactly representable, and the held-out inputs share the training marginal.
"Generalizes better through abstraction or memory" therefore has no lever in it; the harness can only
reward statistical efficiency, and that is what every result above measures. The goal's claim that
Adam and SGD do not optimize for generalizable structure is a claim about WHAT is learned, and the
papers that support it (energy penalties as abstraction priors; Adam moving irrelevant coordinates at
signal scale, C3 and Kunstner 2024; SGD's anisotropic noise selecting flat basins) all predict a
difference that is only visible when (a) the teacher has structure the student can under- or
over-use and (b) the evaluation leaves the training marginal.

## 5. Candidates, ranked by expected information per cost, each with its firing condition

| Candidate | Firing condition | Status |
|---|---|---|
| Structured-teacher out-of-distribution premise test (section 6) | Teacher with irrelevant inputs; held-out off the training marginal | Run: it decides whether any energy/abstraction mechanism can have a lever in MLP regression |
| M-SVAG shrinkage on the SGD direction (Balles and Hennig; lit idea 2); lazy threshold consolidation of distractor jitter (synaptic caching); AdaBayes no-sqrt steps | Distractor or rarely-active coordinates whose Adam-equalized steps hurt generalization | Only if section 6 fires; otherwise no lever (dense inputs) |
| Prospective configuration / iPC relaxation | Large per-sample output steps; cross-layer interference | Dead by derivation (section 3.3) |
| EMA-of-iterates deployment; slow self-prediction target | Variance-dominated sustained risk | Dead by derivation (section 3.1) |
| Conditional-mean denoised residual | A cheaper estimator of E[r given x] than the learner itself | Dead by precedent (RethinkA/B/C) |
| Innovation-driven per-parameter forgetting (adaptive Kalman Q, lit idea 3) | Lag-1 innovation structure not already absorbed by momentum | Dead: v9 measured the innovation autocorrelation and found it pinned by beta1 |

## 6. Pre-registered test: does the optimizer change what is learned, or only how fast?

`structured_generalization_diag_v1.py`, one mlq job, expected well under two minutes (v8 compile plus
CUDA-graph path; the confirmation run trained a polar candidate in 14 s).

* Task: iid_online regime exactly (batch 1, 65536 samples, unit target noise, 16 checkpoints), but
  the teacher reads only the first 4 of 17 inputs (zero first-layer columns, kept columns rescaled by
  sqrt(17/4) so pre-activation variance is unchanged). Thirteen inputs are pure distractors.
* Held-out sets (fresh namespace 7, 8192 points): in_dist x ~ N(0, I); distractor with irrelevant
  inputs scaled by 3 (teacher output unchanged, so this scores invariance); relevant with relevant
  inputs scaled by 1.5 (extrapolation on the signal).
* Candidates: v8 iid_online locks for AdamW and polar at lr x {1/2, 1, 2} and weight decay {0, 0.1},
  twelve in all, each method batched through the v8 RoundLearner. Weight decay is the cheapest energy
  prior; the learning-rate axis is the cheapest path-length knob. Recorded per checkpoint: the three
  risks, path length sum abs(dw) (the papers' energy), distance from init, first-layer weight energy on
  relevant versus distractor columns.
* Decision rule, fixed before running:
  - FIRE: some candidate has sustained distractor risk at least 20% below another's while its
    in-distribution risk is equal or worse. Then the abstraction claim has a lever in MLP regression:
    add a structured/OOD regime to the protocol and test the energy-side rules (M-SVAG shrinkage on the
    SGD direction, lazy threshold consolidation, weight decay as a swept axis) against polar there.
  - KILL: the distractor/in-distribution risk ratio spans less than 10% across all finite candidates.
    Then the out-of-distribution penalty is a property of the fit alone, the optimizer only sets the
    fit, and the abstraction claim has no lever in this model class; the family keeps the progress-per-
    sample lever (polar) and says so.
  - Otherwise: report and do not build.

## 7. Results (job 6411, 25 s of training; contracts job 6410, 5 passed, `structured_generalization_diag_v1_contracts.xml`; run `runs/StructuredGeneralizationDiag__v1__1__1789197623701037300`)

Sustained clean risk over 16 checkpoints, each candidate valid. `ood` = distractor / in_dist.
`w1_rel` / `w1_dis` = final first-layer weight energy on the 4 relevant / 13 distractor columns (init: 8 / 26).

| candidate | in_dist | distractor | relevant | ood | path L1 | dist from init^2 | w1_rel | w1_dis |
|---|---|---|---|---|---|---|---|---|
| adamw lr/2 | 0.1500 | 1.358 | 0.238 | 9.05 | 2602 | 29.0 | 22.1 | 15.3 |
| adamw lr/2 wd .1 | 0.1792 | 0.947 | 0.293 | 5.29 | 2575 | 84.0 | 9.3 | 1.8 |
| adamw lock | 0.1229 | 1.033 | 0.181 | 8.41 | 5466 | 63.6 | 28.4 | 12.6 |
| adamw lock wd .1 | 0.1569 | 0.662 | 0.228 | 4.22 | 5240 | 142.6 | 8.1 | 0.2 |
| adamw 2lr | 0.1481 | 0.897 | 0.188 | 6.06 | 11131 | 139.6 | 51.8 | 11.3 |
| adamw 2lr wd .1 | 0.1936 | 0.724 | 0.252 | 3.74 | 10525 | 178.1 | 9.2 | 0.2 |
| polar lr/2 | 0.1261 | 0.828 | 0.215 | 6.57 | 2668 | 27.3 | 13.6 | 12.4 |
| polar lr/2 wd .1 | 0.1378 | 0.643 | 0.211 | 4.67 | 2614 | 51.2 | 10.4 | 3.7 |
| polar lock | 0.1043 | 0.668 | 0.158 | 6.41 | 5312 | 55.0 | 15.4 | 8.4 |
| polar lock wd .1 | 0.1046 | 0.438 | 0.148 | 4.19 | 5160 | 110.4 | 9.8 | 0.5 |
| polar 2lr | 0.1208 | 0.686 | 0.160 | 5.68 | 10416 | 134.3 | 21.0 | 3.6 |
| polar 2lr wd .1 | 0.1191 | 0.481 | 0.157 | 4.04 | 10085 | 171.7 | 9.6 | 0.6 |

Pre-registered verdict: **FIRE** (12 qualifying pairs). The cleanest pair is polar at its lock with
and without weight decay: identical in-distribution risk (ratio 1.002), distractor risk 34% lower,
relevant extrapolation 6% lower. Review caveat (post hoc, independent code review of the run file):
the kill branch of the rule as coded compares the distractor/in_dist ratio across candidates at
different learning rates, and that ratio moves with the fit even under the null, so the kill branch
could not have fired on this grid and the recorded 2.4x span is not evidence. The fire verdict does
not use the span; it rests on fit-matched pairs, and the decisive pair above is matched to 0.2%.
The run file is frozen as run; a v2 will compute the kill span among fit-matched candidates, guard
the single-candidate and zero-risk cases, and record the init column energies (8 / 26). Across methods at equal weight decay, polar has 35% lower distractor risk
than AdamW at the locks against a 15% in-distribution gain, and moves the distractor columns less
(w1_dis 8.4 vs 12.6): Adam's per-coordinate equalization spends path length on coordinates whose
gradient is mostly noise (C3), and that shows up only off the training marginal.

What the energy quantities say:
* Path length is set by the learning rate alone (2.6k / 5.3k / 10.5k for lr/2, lock, 2lr) and is
  unchanged by weight decay; the candidates that generalize best are not the shortest paths. A path-length
  penalty is not the abstraction prior here.
* Weight energy on distractor columns is. The init carries 26 units of distractor energy; every
  candidate must unlearn it. Without decay the learners leave 8 to 15 units and pay 4 to 9x off the
  marginal; with decay 0.1 they leave 0.2 to 0.6 and pay about 4x. Decay costs AdamW in distribution
  (+28% at the lock) but costs polar nothing; the spectral direction tolerates the prior.
* The remaining 4x penalty with decay is not first-layer distractor mass (0.2 units); it is the
  relevant sub-network itself being evaluated at inputs the training marginal never showed the
  deeper layers. That part is a fit property, not an optimizer property.

## 8. Verdict

The abstraction claim of the original goal has a lever in MLP regression, but not the one the
current proxy scores, and not path length. Optimizers with the same in-distribution risk differ by
a third off the training marginal, the difference is carried by how much weight mass they leave
on coordinates whose gradient is noise, and Adam is the worst of the tested directions at this.
The pre-registered consequence applies: the protocol needs a structured teacher with an
out-of-distribution held-out set as a regime, and the energy-side rules get tested against polar
there with weight decay as a swept axis for every arm.

Firing-condition check for the next test, derived from this table before building it. A gate that
freezes low-SNR coordinates (M-SVAG shrinkage gamma = m^2 / (m^2 + rho s), synaptic-caching
thresholds) would leave the init's 26 units of distractor energy in place, which is worse than any
candidate above. The distractor coordinates carry a real, consistent shrink-toward-zero signal
(their weights inject input noise into the output), so the rule must let that signal through and
only suppress the noise-driven random walk on top of it. That is a shrinkage of the step toward
the direction of the momentum, not a freeze; weight decay is its crude uniform version. The
second-cheapest test is therefore: polar and AdamW directions, decay {0, .03, .1, .3} as a swept
axis, plus one arm per direction with the M-SVAG factor applied to the momentum entering the
direction (not to the applied step, C2), on this structured regime and on the three protocol
regimes so that no in-distribution regression hides behind an out-of-distribution gain. Not built
yet; the protocol change is the user's call.

## 9. Diagnostic v2 (job 6417, 24 s; contracts 6435, 6 passed) and the pre-registered prediction for v11

v2 fixes the review's defects (kill span among fit-matched candidates only, guards, init energies
recorded: relevant 8.0 / distractor 26.0), widens decay to {0, .03, .1, .3}, and measures the one
quantity that decides whether an SNR-shrinkage rule can act: the Balles-Hennig factor
gamma = mhat^2 / (mhat^2 + rho s) from the AdamW learner's own moments at its locked betas, averaged
over the 16 checkpoints, separately on the first layer's relevant and distractor columns.

| AdamW candidate | in_dist | distractor | gamma relevant | gamma distractor |
|---|---|---|---|---|
| lr/2 | 0.150 | 1.358 | 0.377 | 0.399 |
| lock | 0.123 | 1.033 | 0.377 | 0.400 |
| lock wd .03 | 0.123 | 0.817 | 0.364 | 0.395 |
| lock wd .1 | 0.157 | 0.662 | 0.367 | 0.386 |
| 2lr | 0.148 | 0.897 | 0.387 | 0.412 |

Rule verdict again FIRE (29 fit-matched pairs, 63 qualifying pairs; span among matched pairs 1.10
in log ratio, i.e. 3x). Polar at its lock with decay .03 is better in distribution than without
(0.098 vs 0.104) and 18% better off-marginal; decay .1 costs nothing in distribution and gives 34%;
decay .3 gives 41% off-marginal at a 51% in-distribution cost. For AdamW every decay level costs in
distribution.

The firing condition for SNR shrinkage is NOT met. gamma is 0.36 to 0.41 on every column, and the
relevant and distractor columns differ by 0.02 to 0.03. Two consequences follow before any run:

1. The per-sample per-coordinate gradient SNR is about 1/650 on every coordinate (from
   rho = .0025 and gamma = .38): with unit target noise and a 200-step momentum, no coordinate's
   gradient is signal-dominated, relevant or not. A distractor column's mean gradient is a
   shrink-toward-zero force proportional to its weight; a relevant column's mean gradient points
   to a fixed nonzero target; both are consistent signals of the same size against the same
   noise. Temporal per-coordinate statistics cannot tell them apart, which is the RethinkA/B/C
   result again from the other side.
2. A factor that is 0.38 +/- 0.03 everywhere is a uniform multiplier (C5). Polar and the per-layer
   RMS direction renormalize the whole layer, so the multiplier cancels exactly up to the 0.03
   dispersion. Pre-registered prediction for jobs 6424-6428: polar_svag equals polar and rms_svag
   equals matrix_rms within selection noise on every metric. If either shrunk family beats its
   parent by the 20% bar off-marginal, the prediction is wrong and the dispersion carries more
   than it looks.

What weight decay does that shrinkage cannot: it is a prior on the answer (small weights), and it
wins here because the teacher is sparse in its inputs. The relevant columns resist it (their
gradient signal has a nonzero target), the distractor columns do not (their gradient signal agrees
with it). The energy prior the papers describe therefore acts through agreement between a
deterministic pull and the gradient's mean, not through gradient variance. That is testable with a
prior that is not uniform L2 but the same kind of object: a pull toward the coordinate's own
long-horizon mean (a slow tier), which the v9/v10 measurements already scored as at most 4 to 5.5%
in distribution and which this regime has not scored off-marginal.

## 10. v11 protocol, structured_ood regime (jobs 6424-6428; contracts 6436, 12 passed; plan `optimizer_proxy_v11_plan.json`)

Protocol: `optimizer_proxy_eval_v11.py` with `optimizer_proxy_model_v11.py`. New regime structured_ood
(iid_online's batch, horizon, noise; teacher reads 4 of 17 inputs; selection by noisy in-distribution
validation only; locked test scoring adds distractor-scaled and relevant-scaled held-out sets). Every
family sweeps lr (13) x decay {0, .03, .1, .3} x beta1 {lock, .9, .99} with beta2/head fixed at its
parent's iid_online lock. Sustained clean risk of the selected candidate, 16 checkpoints:

| family | in_dist | distractor | relevant | selected lr / wd / beta1 | train s |
|---|---|---|---|---|---|
| adamw | 0.1252 | 1.022 | 0.186 | 3e-4 / 0 / .995 | 11 |
| matrix_rms | 0.1333 | 0.864 | 0.188 | 1e-4 / .03 / .995 | 12 |
| polar | 0.0990 | 0.543 | 0.152 | 1e-4 / .03 / .995 | 22 |
| polar_svag | 0.1017 | 0.581 | 0.158 | 1e-4 / .03 / .995 | 22 |
| rms_svag | 0.1319 | 0.864 | 0.187 | 1e-4 / .03 / .995 | 12 |

* The pre-registered null holds: polar_svag is within 3% of polar in distribution and 7% worse
  off-marginal; rms_svag is within 1% of matrix_rms on both. At identical configurations the
  shrunk families' validation scores differ from their parents' by a median of -0.14%. SNR
  shrinkage of the momentum entering a per-layer direction is dead here, for the reason section 9
  derived: gamma is uniform to within 0.03.
* The regime does what section 4 said the old proxy could not: it separates optimizers by WHAT
  they learned. Under the same in-distribution selection AdamW ends 47% worse off-marginal than
  polar against 21% worse in distribution, because its selection cannot find weight decay (every
  decay level costs AdamW in distribution) while polar's selection picks .03 for free. The
  per-layer RMS direction without the polynomial sits between (0.864): the spectral direction
  carries most of the off-marginal gain, decay the rest.
* beta1 sits at the top of the swept axis (.995) for every family; the parents' full v8 grids
  selected it interior, so this is inherited, not a new edge.

Verdict on the energy-side rules from the papers, in this family's harness: per-coordinate gradient
variance carries no credit-assignment information at batch 1 with unit noise (sections 9, 10 and
RethinkA/B/C agree); the energy prior that changes what is learned is a pull toward a prior answer
that the gradient's mean either resists (relevant) or agrees with (distractor). Uniform L2 is its
crudest form and already beats every variance-side device this family built.

## 11. Pre-registered follow-up: a non-uniform pull of the same kind, off-marginal

The distractor coordinates carry no fixed target; their weight is a noise-driven random walk around
zero whose amplitude is what the off-marginal penalty measures (w1_dis in sections 7 and 9). A slow
deployed tier (uniform Lookahead, v10's look_polar, already in the v11 model) damps that walk without
a prior on the answer: it pulls every coordinate toward its own recent mean, which is zero for a
distractor and the target for a relevant coordinate. In distribution it was worth 4 to 5.5% on the
regression regimes (section 2). Firing condition: the off-marginal penalty must be dominated by
walk amplitude rather than by a biased mean; sections 7 and 9 show w1_dis 8 to 15 units without
decay against 0.2 to 0.6 with, so it is.

Test: look_polar on structured_ood, plan `optimizer_proxy_v11_look_plan.json` (polar's iid locks,
lr x k0 {.1, .25, .5, .75} x decay {0, .03, .1}, 156 configs), one job. Rule: FIRE if look_polar's
selected sustained distractor risk is >= 20% below polar's 0.543 with in-distribution risk within 5%
of 0.099; KILL if it is within 10% of polar off-marginal; otherwise report. A fire extends to the
three protocol regimes, where v10 already measured look_polar in distribution.

Result (job 6438, 25 s; plan `optimizer_proxy_v11_look_plan.json`): look_polar selected lr 1.4e-3, decay .03,
k0 .1 (the axis edge; validation 1.065 / 1.112 / 1.242 / 1.459 at k0 .1 / .25 / .5 / .75).

| family | in_dist | distractor | relevant |
|---|---|---|---|
| polar (v11 lock sweep) | 0.0990 | 0.543 | 0.152 |
| look_polar | 0.0808 | 0.432 | 0.120 |

FIRE by the rule (distractor -20.5%, in distribution -18%, relevant -21%), at the gain edge. This is
the first mechanism in the family to clear the 20% bar on any promotion metric, and it does so on
the metric the original goal names, with a hot fast tier (14x polar's locked lr) and a slow
deployed tier. In distribution v10 measured the same structure at 4 to 5.5% on the dense-teacher
regimes; the sparse teacher raises the in-distribution gain to 18% because the damped walk is
mostly distractor mass there. Per the v10 precedent an edge gain triggers an extension, not a
promotion: plan `optimizer_proxy_v11_look_ext_plan.json` (k0 {.02, .05, .1}, same lr and decay
axes). Sub-bar and harmful in-distribution results for look_polar in drifting_reuse and
clipped_bandit (v10) stand; a structured drifting regime has not been built.

Caveat from the per-checkpoint curves (locked test scoring, distractor set, 16 checkpoints):

```
polar       2.04 1.18 0.80 0.57 0.55 0.34 0.40 0.32 | 0.29 0.34 0.32 0.28 0.43 0.29 0.28 0.26
look_polar  1.49 0.84 0.59 0.45 0.45 0.24 0.31 0.25 | 0.22 0.26 0.27 0.30 0.39 0.33 0.25 0.27
```

First-half mean 0.90 vs 0.65 (-28%); second-half mean 0.311 vs 0.286 (-8%); endpoints 0.261 vs
0.270. The sustained-mean gain is mostly faster early progress from a fast tier at 14x polar's
learning rate, damped by the slow tier, which is exactly the bias-dominated structure section 3
described. The asymptote is not clearly different. In-distribution the same holds (second half
0.066 vs 0.059, -10%). So the fire is real under the protocol's metric but it is a speed
effect on a metric that rewards speed, and it should be read with the 16-checkpoint mean's
known bias in mind; a longer horizon or a second-half metric would shrink it.

Extension (job 6442, 23 s; plan `optimizer_proxy_v11_look_ext_plan.json`, k0 {.02, .05, .1}): selected
lr 3e-3, decay .03, k0 .05, interior on every axis (validation 1.085 / 1.058 / 1.082 at k0 .02 / .05 / .1).

| family | sustained in_dist | sustained distractor | sustained relevant | 2nd-half distractor | endpoint distractor |
|---|---|---|---|---|---|
| adamw | 0.1252 | 1.022 | 0.186 | | |
| polar | 0.0990 | 0.543 | 0.152 | 0.311 | 0.261 |
| look_polar k0 .1 (edge) | 0.0808 | 0.432 | 0.120 | 0.286 | 0.270 |
| look_polar k0 .05 | 0.0742 | 0.378 | 0.113 | 0.260 | 0.251 |

Against polar: -25% in distribution, -30% off-marginal, -26% on relevant extrapolation, all
interior; second-half off-marginal -16%, endpoint -4%. Against AdamW under the same in-distribution
selection: -41% in distribution, -63% off-marginal.

## 12. Verdict

* The original goal's claim is right in one specific form and measurable now: optimizers that fit
  equally in distribution differ by a third or more in what they learned, Adam is the worst of the
  tested directions at it, and the difference is carried by weight mass left on coordinates whose
  gradient is noise. The v8 proxy could not see this; the structured_ood regime can.
* Of the energy rules from the papers, the ones that read gradient VARIANCE (per-coordinate SNR
  shrinkage, per-parameter consolidation gains, the RethinkA/B/C gates) are dead in this harness for
  a derivable reason: at batch 1 with unit noise every coordinate's per-sample SNR is about 1/650,
  relevant or not. The ones that act as a PULL that the gradient's mean either resists or agrees
  with are alive: uniform L2 (free for polar, costly for AdamW) and the slow deployed tier, which
  pulls every coordinate toward its own recent mean and lets a hot fast tier (30x polar's lock)
  explore without depositing its walk in the deployed weights. That is the closest thing in the
  family to "optimize toward your own denser prediction of the future": the deployed tier is the
  learner's prediction of where the fast tier's walk is going, and the gain is largest exactly
  where the walk has no target.
* look_polar at k0 .05 is the first mechanism to clear the 20% bar on a promotion metric, on three
  metrics at once, interior on every axis, in one regime. The gain is front-loaded (a speed effect
  on a mean that rewards speed), the second-half gain is 14 to 16%, and the endpoint gain is 4%.
* It is not promotable under the family's rule: v10 measured the same mechanism in distribution at
  4 to 5.5% in the dense regression regimes, and at the gain that wins here (k0 .05 to .1) it is
  harmful in drifting_reuse and clipped_bandit on validation (section 13; the selected look_polar
  there is polar with the tier turned off), and a drifting structured regime does not exist. Whether structured_ood joins the protocol as a promotion metric,
  and whether a drifting or bandit variant with a sparse teacher is built to test the slow tier
  where it lost, is the user's call; both are one plan file and one job each on the v11 harness.

## 13. Does look_polar generally beat AdamW? No on its own; the general win is polar's

Selected sustained clean excess risk per regime (each family sweeps its own lr; v9 references, v10 look_polar, v11 structured_ood):

| regime | adamw | polar | look_polar | look vs adamw | look vs polar | selected k0 |
|---|---|---|---|---|---|---|
| iid_online | 0.2403 | 0.1880 | 0.1805 | -24.9% | -4.0% | .05 (interior) |
| drifting_reuse | 0.3034 | 0.2852 | 0.2850 | -6.1% | -0.1% | .75 (top of grid) |
| clipped_bandit | 0.2442 | 0.2229 | 0.2207 | -9.6% | -1.0% | .75 (top of grid) |
| structured_ood in_dist | 0.125 | 0.099 | 0.074 | -41% | -25% | .05 (interior) |
| structured_ood distractor | 1.022 | 0.543 | 0.378 | -63% | -30% | .05 (interior) |

Best validation per gain in the v10 look_polar sweeps (lower is better; polar's own validation 1.1772 / 1.2767 / 0.4685):

| regime | k0 .1 | k0 .25 | k0 .5 | k0 .75 |
|---|---|---|---|---|
| drifting_reuse | 1.3069 | 1.2814 | 1.2775 | 1.2763 |
| clipped_bandit | 0.6554 | 0.5125 | 0.4757 | 0.4663 |

So the slow deployed tier is a two-regime device: at the gain that wins in structured_ood and iid_online (k0 .05 to .1) it is +2.4% in drifting and +40% in the bandit on validation, and per-regime selection rescues it only by turning it off (k0 at the top of the grid, where look_polar is polar). Section 12's "harmful in clipped_bandit" is that small-gain result; the selected look_polar never lost to polar. The family's general win over AdamW is polar (-25% / -6% / -10% / -21% in distribution, -47% off-marginal), unchanged since v8. A better idea is required, and it must not be a uniform pull: the tier's lag is exactly what kills it where the target moves.

### 13.1 Candidate: consistency-gated weight decay, and its two firing conditions

Where the structured risk is. From the v11 ratios, the student's sensitivity to the 13 irrelevant inputs accounts for about 55% of polar's in-distribution risk and 90% of AdamW's (distractor excess = 8 x sensitivity). Uniform decay removes that mass at the price it charges the 4 relevant columns; v2 measured the two cancelling (polar lock, decay .1: same fit, -34% distractor). Per-coordinate gradient variance cannot separate the columns (gamma uniform, section 9). What can, in principle, is the direction of the gradient mean relative to the weight: on an irrelevant column the population gradient is a restoring force toward zero; on a relevant column it points to the teacher's value, mostly away from zero for a student initialised below the rescaled teacher. A decay applied only where a slow gradient average agrees with shrinking is a non-uniform pull that removes irrelevant mass at full strength; in dense regimes it reduces to a rescaled uniform decay (agreement about 1/2 everywhere), so it has no lag and no bandit failure mode by construction. It is the papers' tagging-gated consolidation with the tag read from direction, not variance.

Pre-registered diagnostic (`structured_generalization_diag_v3.py`, one job, batch 1, 65536 samples, both iid locks at lr x {1/2, 1, 2}):

* ceiling: oracle decay {.1, .3, 1} on the first layer's distractor columns only (decoupled, in place between graph replays) against uniform decay {0, .03, .1, .3}. FIRE if some oracle candidate is >= 20% better in distribution than the best uniform candidate, or fit-matched (within 5%) with >= 20% lower distractor risk. KILL only if at least one oracle candidate is fit-matched, none is >= 10% better in distribution, and every fit-matched one is within 10% on distractor risk (the vacuous kill was caught in review before the run). This bounds what ANY gate can buy.
* detect: at the lock without decay, second-half mean fraction of first-layer coordinates whose momentum (lock beta1), two-pole filtered momentum (second pole .99, .999) or slow realised displacement (.99, .999) says "shrink", distractor minus relevant columns. FIRE if some signal's margin >= 0.25; KILL if all <= 0.10.

Build the gated rule (model v12, all four regimes, each family sweeping its own lr and decay) only if both fire. If the ceiling kills, the structured risk is not removable by any decay gate and the direction is closed. If the ceiling fires and detection kills, the signal is not in the stream and the direction is closed for streaming optimizers.

### 13.2 Result (contracts job 6496, 8 passed; run job 6497, 9 s AdamW + 22 s polar training; `runs/StructuredGeneralizationDiag__v3__1__1789257382995869158`)

Ceiling: FIRE for both methods, and it is the largest lever measured in this family. Sustained clean excess risk, in_dist / distractor, first-layer distractor energy at the end:

| candidate | in_dist | distractor | w1 distractor energy |
|---|---|---|---|
| adamw lock, decay 0 (best uniform) | 0.123 | 1.033 | 12.6 |
| adamw lock, oracle decay 1 on distractor columns | 0.078 (-36%) | 0.219 (-79%) | 0.07 |
| polar lock, decay .03 (best uniform) | 0.098 | 0.548 | 3.6 |
| polar lock, oracle decay 1 on distractor columns | 0.073 (-26%) | 0.263 (-52%) | 0.26 |

The oracle gain is monotone in the oracle decay up to the largest value tested (1), so the ceiling is at least this. The endpoint in-distribution risk follows the same order (adamw 0.083 vs 0.069; polar 0.062 vs 0.039), so unlike the slow tier this is not a speed effect.

Detect (pre-registered, at the locks WITHOUT decay): KILL for both. Best margin 0.076 (adamw) and 0.077 (polar), both from the two-pole filtered momentum with second pole .999; every signal's distractor-column shrink agreement is 0.50 to 0.53, i.e. the restoring force on an irrelevant column is invisible even in a 2000-sample average (the excess risk it carries, about 0.05, is spread over 832 coordinates against unit per-sample gradient noise). The margin that exists comes from the RELEVANT columns, which say "grow" 57 to 63% of the time (the student starts below the rescaled teacher). So the gate I pre-registered, "shrink where the gradient mean says shrink", cannot be read from the stream: the pre-registered consequence stands for that gate.

Post-hoc observation from the same table, not pre-registered and not a verdict: under uniform decay the same signal separates the columns, because relevant columns RESIST the decay and distractor columns do not. Second-half shrink agreement of the .999-filtered momentum, relevant / distractor:

| decay | adamw lr1 | polar lr1 | polar lr2 |
|---|---|---|---|
| 0 | .43 / .50 | .43 / .50 | .42 / .48 |
| .03 | .34 / .49 | .37 / .50 | .35 / .45 |
| .1 | .20 / .45 | .27 / .48 | .31 / .42 |
| .3 | .08 / .24 | .23 / .44 | .27 / .44 |

At decay .1 the margin is 0.17 to 0.25 (it was 0.07 without decay). The readable tag is not "the gradient wants this weight at zero" but "this weight does not push back when pulled": use it or lose it. That makes the candidate a resistance-gated decay: decay everything at full strength (the prior), release it per coordinate in proportion to the evidence that the slow gradient average opposes the pull. In dense regimes every coordinate resists and the rule reduces to a weaker uniform decay, which the decay sweep already covers, so it cannot lose there by construction; under drift the gate reads a slow average and can lag, which is the risk to measure. It is tested in section 14 with its own pre-registered rules on all four regimes.

## 14. Resistance-gated decay (model v12): pre-registration

Rule (`optimizer_proxy_model_v12.py`, families gate_adamw and gate_polar). Per weight coordinate, EMAs of the raw gradient and its square with pole gate_beta give s_hat, q_hat; z = s_hat sign(w) / sqrt(rho (q_hat - s_hat^2) / (1 - rho)) is the slow mean's z-score against its own noise; gate = Phi(z); the step is the parent's at decay 0 minus layer_lr * weight_decay * gate * w. Pure noise gives gate 1/2 on average (uniform decay at half strength), a resisting coordinate gate near 0, an agreeing one gate near 1. Bias columns never decay. Contracts: parent bit-exact at decay 0; equal to the parent at full decay under a shrink-everywhere zero-variance stream; v11 families pass through bit-exact.

Why it can win: the oracle ceiling (13.2) is -26% / -36% in distribution and the readable tag is resistance (13.2 table). Why it can lose: the gate reads a 100 to 1000 sample average, so under drift it releases decay late on coordinates that became relevant and keeps decaying them meanwhile; and half-strength decay on noise coordinates may be too weak to matter at the decays the parents tolerate, in which case the sweep will push weight_decay to 1 and the gate's cost on relevant coordinates decides.

Plan `optimizer_proxy_v12_plan.json` (`make_optimizer_proxy_v12_plan.py`): parents adamw / polar on 13 lrs x decay {0, .03, .1, .3, 1}; gated families on 13 lrs x decay {.03, .1, .3, 1} x gate_beta {.99, .999}; betas and head scale at the regime locks; all four regimes; one family per job (16 jobs). Selection unchanged (noisy in-distribution validation).

Rules, fixed before the jobs ran:

* structured_ood FIRE: the gated family's selected sustained in-distribution risk is >= 20% below its parent's on this plan, or within 5% of it with >= 20% lower sustained distractor risk. Both gated families are scored; gate_polar against polar is the promotion comparison, gate_adamw against adamw answers "does the gate beat Adam's own inability to use decay".
* protocol regimes NO-REGRESSION: the gated family's selected risk within 5% of its parent's in iid_online, drifting_reuse and clipped_bandit. A regression above 5% in any regime kills promotion whatever structured_ood says.
* structured KILL: neither gated family fires and the parent's uniform decay 1 already matches the gated family's best (the gate adds nothing over a stronger uniform pull). Then the direction is closed: the tag is readable but not exploitable at streaming SNR.
* Report the endpoint in-distribution risk next to the sustained one (the slow tier's gain was front-loaded; this one should not be).

Review before the run (one agent): gate math, parent equality at gate 1 for both paths, aux slots, graph capture and test premises confirmed; one inherited off-by-one found, v11's `variance_reduction` evaluates beta^(n+1) for an n-update EMA, fixed locally in v12 with its own contract. First submission (6513-6529) cancelled before start; resubmitted as contracts 6533 and family jobs 6534-6549 (structured_ood, iid_online, drifting_reuse, clipped_bandit x adamw, polar, gate_adamw, gate_polar).

### 14.1 structured_ood result (jobs 6534-6537) and pre-registered pole extension

| family | in_dist sustained | endpoint | distractor | relevant | selected |
|---|---|---|---|---|---|
| adamw | 0.1251 | 0.0828 | 1.022 | 0.186 | lr 3e-4, decay 0 |
| gate_adamw | 0.1212 (-3.2%) | 0.0850 | 0.873 (-14.6%) | 0.181 | lr 3e-4, decay .03 (edge), pole .999 (edge) |
| polar | 0.0990 | 0.0513 | 0.543 | 0.152 | lr 1e-4, decay .03 |
| gate_polar | 0.0981 (-0.9%) | 0.0500 | 0.521 (-4.0%) | 0.151 | lr 1e-4, decay .1, pole .999 (edge) |

No fire. What the gate did do: strong decay became much cheaper (best validation at decay 1: gate_polar 1.108 vs polar 1.380; gate_adamw 1.153 vs adamw 1.531; at decay .3: 1.085 vs 1.142 and 1.116 vs 1.351), but no gated configuration beat the parent's best, so the release on relevant coordinates is still too weak at the pole .999: their measured resistance (agreement .27 vs .48) is a z of about -0.6, which Phi maps to a gate of .27 against .5 on noise, a 2x discrimination where the oracle has infinite. Derivation for the extension: at equilibrium a relevant coordinate is displaced from its optimum by about 2 sigma sqrt((1-beta)/(1+beta)) / H (the mean gradient needed for z = -2 over the curvature); at .999 that is about 0.045 sigma / H, at .9999 about 0.014 sigma / H, so a 10x slower pole cuts the in-distribution cost of the release by 3x while the window (10k of 65k samples) is still short against the horizon. Both families sat at the pole edge, which is the family's extension trigger.

Extension (`optimizer_proxy_v12_ext_plan.json`, gate_beta {.9995, .9999}, same lr and decay axes, structured_ood only, two jobs): same FIRE rule against the v12 parents. If it fires, the three protocol regimes are run for the firing family with the extended axis before any promotion; if it does not, the structured KILL clause applies (the tag is readable but not exploitable at streaming SNR).

### 14.2 Protocol regimes (jobs 6538-6549; contracts 6553, 15 passed after one fixture fix, model unchanged)

Selected sustained clean excess risk (endpoint in parentheses):

| regime | adamw | gate_adamw | polar | gate_polar |
|---|---|---|---|---|
| iid_online | 0.2403 (0.191) | 0.2504 (0.207), +4.2% | 0.1879 (0.154) | 0.1887 (0.151), +0.4% |
| drifting_reuse | 0.3034 (0.469) | 0.3032 (0.470), -0.1% | 0.2817 (0.422) | 0.2789 (0.427), -1.0% |
| clipped_bandit | 0.2465 (0.370) | 0.2435 (0.380), -1.2% | 0.2214 (0.318) | 0.2202 (0.306), -0.6% |

No regression anywhere (rule: within 5%), as derived: in dense regimes the gate is a weaker uniform decay. gate_adamw's +4.2% in iid_online is the grid, not the gate: AdamW's lock there is decay 0, which the gated grid does not contain (decays start at .03), and gate_adamw at decay .03 sits on that edge. gate_polar selected the fastest pole (.99) in all three dense regimes and the slowest (.999) in structured_ood, consistent with the lag cost under drift and the discrimination gain under a fixed sparse teacher. Training time 2 to 23 s per family.

### 14.3 Pole extension result (jobs 6561, 6562; `optimizer_proxy_v12_ext_plan.json`)

| family | in_dist sustained | endpoint | distractor | relevant | selected |
|---|---|---|---|---|---|
| adamw (v12) | 0.1251 | 0.0828 | 1.022 | 0.186 | lr 3e-4, decay 0 |
| gate_adamw | 0.1100 (-12.1%) | 0.0603 (-27.1%) | 0.482 (-52.8%) | 0.179 | lr 2e-4, decay .3, pole .9999 (edge) |
| polar (v12) | 0.0990 | 0.0513 | 0.543 | 0.152 | lr 1e-4, decay .03 |
| gate_polar | 0.0912 (-7.9%) | 0.0462 (-9.9%) | 0.441 (-18.8%) | 0.145 | lr 1e-4, decay .3, pole .9999 (edge) |

No fire (bar 20% on in-distribution, or 20% on distractor at matched fit; gate_polar is -7.9% / -18.8%). Not a kill either: the parent's uniform decay 1 validates at 1.380 (polar) and 1.531 (adamw) against 1.075 and 1.099 gated, so the gate is doing real work over a stronger uniform pull. The gain is not front-loaded (endpoint gain >= sustained gain for both families), unlike the slow tier's. The pole is at the grid edge again and the gain is monotone in it (.999 -> .9999: gate_polar -0.9% -> -7.9%, gate_adamw -3.2% -> -12.1%), as derived in 14.1; the selected decay moved from .03/.1 to .3 as the release got cheaper. One more point (.99999, a 100k-sample window against the 65k horizon, i.e. effectively the cumulative mean) closes the axis; beyond it the rule is a cumulative sign test and the structured regime cannot distinguish further poles.

### 14.4 Final pole point (jobs 6565, 6566; `optimizer_proxy_v12_ext2_plan.json`, gate_beta .99999)

| family | in_dist sustained | endpoint | distractor | relevant | selected |
|---|---|---|---|---|---|
| gate_adamw | 0.1063 (-15.1% vs adamw) | 0.0787 (-5.0%) | 0.411 (-59.8%) | 0.169 | lr 3e-4, decay .3, pole .99999 |
| gate_polar | 0.0855 (-13.7% vs polar) | 0.0455 (-11.2%) | 0.326 (-39.9%) | 0.140 | lr 1e-4, decay 1 (edge), pole .99999 |

Against AdamW, gate_polar is -31.7% in distribution and -68% off-marginal under identical in-distribution selection.

By the letter of the 14 rule this is not a fire: in-distribution -13.7% is short of 20%, and the second clause was written as "within 5% of the parent" with >= 20% lower distractor risk, which a candidate that dominates the parent on both (-13.7% and -39.9%) does not literally satisfy. The clause was meant as "no worse than 5%"; the rule did not anticipate dominance. I record both readings and do not call it a fire. Trend over the pole: gate_polar -0.9% -> -7.9% -> -13.7% in distribution, -4% -> -19% -> -40% off-marginal (.999, .9999, .99999); the gain is not front-loaded (endpoint -11%). gate_adamw's endpoint moved from -27% to -5% between the last two poles at a similar sustained gain, so its endpoint is noisy at the 16-checkpoint resolution. With the pole axis closed (a 100k window exceeds the 65k horizon), gate_polar's selected decay is now at the grid edge (1); the family precedent calls for one decay extension ({3, 10}, same pole, structured_ood, both families) before a verdict. Any promoted configuration must then be re-checked for regression in the three protocol regimes with the same extended axes.

### 14.5 Closing the axes: decay extension and one union selection (pre-registered)

Jobs 6573-6574: decay {3, 10} at pole .99999, structured_ood, both gated families (`optimizer_proxy_v12_ext3_plan.json`). Jobs 6575-6578: gate_polar on the union grid, 13 lrs x decay {.03, .1, .3, 1, 3, 10} x pole {.99, .999, .9999, .99999} (312 configurations, `optimizer_proxy_v12_full_plan.json`), in all four regimes, so that (a) the structured number is one selection rather than the best of four plans and (b) the no-regression check in iid_online, drifting_reuse and clipped_bandit sees the same axes. Rules as in section 14 with the dominance reading made explicit: structured FIRE if the union selection is >= 20% below polar in distribution, or no worse than 5% in distribution with >= 20% lower distractor risk; NO-REGRESSION if within 5% of polar's v12 selection in each dense regime. Promotion of gate_polar over polar requires both; if the union selection lands between (say -10 to -20% in distribution with a large off-marginal gain and no regression), it is reported as a strict improvement below the bar and the promotion is the user's call.

### 14.6 Results: decay extension (jobs 6573, 6574) and union selection (jobs 6575-6578)

Decay extension at pole .99999, structured_ood: gate_polar selects decay 3 (over 10) with sustained in-distribution .08479 against .08548 at decay 1 (14.4) and distractor .2276 against .3264; the in-distribution axis is flat from decay 1 to 3 and only the off-marginal risk keeps falling. gate_adamw selects decay 3 at .13057, worse than its .10630 at decay .3: the noisy validation picks the edge and the decay axis overshoots for AdamW. Decay is closed for polar (interior on the union grid); no further extension.

Union grid (312 configurations, one selection per regime, noisy in-distribution validation only):

| regime | gate_polar | polar (v12 selection) | AdamW (v12 selection) | gate vs polar | gate vs AdamW | selected (lr, decay, pole) |
|---|---|---|---|---|---|---|
| iid_online | .18809 | .18786 | .24027 | +0.1% | -21.7% | 1e-4, .03, .99999 |
| drifting_reuse | .27887 | .28170 | .30341 | -1.0% | -8.1% | 2e-3, .1, .99 |
| clipped_bandit | .21995 | .22143 | .24648 | -0.7% | -10.8% | 3e-3, .1, .99 |
| structured_ood | .08548 (distractor .3264) | .09902 (.5433) | .12514 (1.0219) | -13.7% (distractor -39.9%) | -31.7% (-68.1%) | 1e-4, 1, .99999 |

Rules of 14.5. NO-REGRESSION holds in all three dense regimes (worst +0.1%). Structured FIRE does not hold by the 20% in-distribution bar; the selection is a strict improvement below the bar (-13.7% in distribution together with -39.9% off marginal, endpoint -11.2%, relevant-scale -8.0%). The pole sits at the grid edge in every regime (.99999 above in the static regimes, .99 below under drift), as predicted by the sqrt(window) discrimination argument versus lag: the gate wants the longest window the drift allows. Extending the pole below .99 for the drift regimes was not pursued: the gains there are at most 1% and inside the selection noise. Verdict on this proxy suite: gate_polar dominates polar (never worse than +0.1%, one large structured gain) and beats AdamW in all four regimes by 8% to 32%; promotion over polar is the user's call per 14.5.

## 15. Transfer tests: stock panel and PPO (pre-registered before any result)

Shared torch implementation `cleanrl/shared/gate_polar.py` (`GatePolar`, `mlp_groups`), contracts `tests/test_gate_polar.py` (job 6624; job 6584 failed on two test-side defects, 7 of 9 passed, and a review found a crash in the stock harness's finiteness check, fixed before resubmission): reproduces `optimizer_proxy_model_v12.transition` for gate_polar and polar in float64 to 1e-12, Adam groups match `torch.optim.Adam` at decay 0, exact gate limits, compiled equals eager, device LR/step tensors. Every job below is chained on 6624.

Stock panel (`panel_hd_mlp_gate_v1.py`, jobs 6625-6646, `--max-parallel-runs 3`). Same 257-256-256-1 stream as `finance_panel.md` (streaming Adam 0.7501 at lr 3e-4, seed spread about .001, PERM 1.006). Arms: Adam lr 3e-4 anchor (reproduction check, also builds the feature cache); polar (decay 0) lr {1e-4, 3e-4, 1e-3}; gate_polar lr {1e-4, 3e-4, 1e-3} x pole {.999, .99999} x decay {.1, .3, 1}. Selection is best-of-grid on the test window for every arm, exactly as the recorded Adam number was chosen, so the comparison is best-of-grid against best-of-grid. Rule: gate_polar counts as a stock win if its best relative MSE is at least .005 below Adam's best (five seed spreads) AND below polar's best AND better than polar in at least 6 of 8 test blocks; a PERM run of the selected gate config follows and must not fall below .99 (a sub-unit PERM would mean the decay is fitting the constant, not signal). If gate_polar beats Adam but not polar, the gain is the polar direction, not the gate, and is reported as such.

PPO (`ppo_continuous_action_gatepolar_v1.py`, base Beta PPO with only the optimizer swapped, jobs 6647-6662, HalfCheetah-v4, 16 envs, 8M steps, seed 1, `--max-parallel-runs 6`). Arms: `--optimizer adam` lr 3e-4 in the same trainer (control; the recorded base run scores 7668 +-299 over its last episodes); polar (decay 0) lr {1e-4, 3e-4, 1e-3}; gate_polar lr {1e-4, 3e-4, 1e-3} x pole {.99, .9999} x decay {.1, .3}. The pole grid follows the drift regimes' preference for short windows; PPO takes about 78k optimizer steps in 244 iterations of 320, so .99 reads within-iteration consistency and .9999 spans about 30 iterations. head_lr_scale is fixed at 1 (not swept; the proxy's structured lock used 8 for polar, so the head axis is a known unswept degree of freedom). Rule: single seed, so no claim below 300 return (about the base run's own spread). gate_polar is an RL win if its best final return exceeds both the in-trainer Adam control and the best polar arm by more than 300; if polar alone clears Adam, the credit goes to the direction, not the gate; anything inside 300 is a null and is reported as one. Runs clearly behind Adam at 2M-3M steps are cancelled, per the repository's culling guidance.

Amendments while running (before any gate result). (a) The in-trainer Adam control scored 7163 +-220 against 7668 +-299 for the earlier run of the same base file at the same seed: two nominally identical runs differ by about 500, so the 300-return bar is below the replicate noise; the RL verdict uses 500 as the minimum detectable difference. (b) Polar at lr 1e-4 finished at 2725 (step magnitude 0.2 lr per coordinate is too small), and the four gate arms at lr 1e-4 tracked it exactly through 1.2M steps (690-825 vs Adam's 2586); they were cancelled (jobs 6651-6654) under the repository's culling rule, leaving the lr 3e-4 and 1e-3 gate arms. (c) The stock Adam anchor reproduced the recorded number (0.75010 vs 0.75012, identical blocks).
(d) Polar at lr 1e-3 finished at 9126 +-172, about 2000 above the Adam control, at the top of its LR grid; the polar and gate arms are extended to lr 3e-3 (jobs 6674-6678) so that no arm is selected at a grid edge.
(e) Polar alone on the stock stream is 7.3% worse than Adam (0.80503 at lr 3e-4, interior, against 0.75010; 0 of 8 blocks), so gate arms built on polar cannot isolate the gate there. A gate_adam arm (GatePolar with every group on the Adam direction, gated decay only) is added as v2 of both harnesses (`panel_hd_mlp_gate_v2.py`, `ppo_continuous_action_gatepolar_v2.py`; v1 sources are pinned by queued jobs): stock lr 3e-4 x pole {.999, .99999} x decay {.1, .3, 1}; PPO lr 3e-4, pole .9999, decay {.1, .3} (stock jobs 6683-6688, PPO jobs 6689-6690). Same rules: a gate_adam stock win is best gate_adam <= 0.7451 with >= 6/8 blocks; an RL win is > Adam + 500.

### 15.1 PPO result (HalfCheetah-v4, 8M, seed 1; jobs 6647-6662, 6674-6678; last-20-episode mean +-CI95)

| arm | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|
| Adam (base control, in this trainer) | 7163 +-220 | - | - |
| polar (decay 0) | 5963 +-44 | 9126 +-172 | 9065 +-257 |
| gate pole .99, decay .1 | 5901 | 8638 | 7910 |
| gate pole .99, decay .3 | 5532 | 7998 | 7355 |
| gate pole .9999, decay .1 | 6003 | 9292 | 9696 +-103 |
| gate pole .9999, decay .3 | 5866 | 9108 | 9210 |

Earlier base run of the same file: 7668 +-299; the two Adam runs differ by about 500, the replicate noise floor (15 amendments). lr 1e-4 arms cancelled (polar 2725; gate tracked it).

Reading. (1) The polar direction is the RL win: best polar 9126 against Adam 7163 / 7668, +1460 to +1960, three to four noise floors, with lr interior (1e-3 over 3e-3). (2) The gate at pole .99 is worse than polar at every LR and decay (-60 to -1700): a 100-step window reads within-iteration same-batch consistency, which in PPO's reused minibatches looks like signal for every coordinate, so the gate releases nothing where it should and decays where the drift has moved on. (3) At pole .9999 (30-iteration window) the gate is at or above polar at every LR (+40, +166, +630 at decay .1), monotone in LR, and its best point 9696 is +570 over polar's best: at the noise floor, and at the top of its LR and pole grids, so not a claim. Extension queued (jobs 6679-6682: polar lr 1e-2, gate lr 1e-2 at pole .9999, gate lr 3e-3 and 1e-2 at pole .99999, all decay .1) to close both edges before any verdict on the gate in RL.

Cost in this trainer (PhaseTimer update phase per iteration of 320 optimizer steps, end of run, under 3-6 concurrent runs): Adam 0.18 s; polar 0.70-0.93 s; gate 0.42-0.80 s. End-of-run SPS 64k (Adam, fewer neighbours) against 27-38k; wall 124 s against 210-297 s per 8M run. The extra time is kernel launches for five Newton-Schulz iterations on 64x64 matrices plus the compiled group updates, not FLOPs; the gate adds nothing measurable over polar. A cudagraph-captured optimizer step would remove most of it. Optimizer state: Adam 2.0x the parameters, polar 1.1x, gate 3.05x (11469-parameter agent).

### 15.2 Stock panel result (jobs 6625-6646; relative MSE on the last 40%, lower is better; seconds per run)

| arm | lr 1e-4 | lr 3e-4 | lr 1e-3 |
|---|---|---|---|
| Adam (anchor; recorded grid 0.7507 / 0.7501 / 0.7722) | - | 0.75010 (173 s) | - |
| polar (decay 0) | 0.83532 | 0.80503 | 0.83379 |
| gate pole .999, decay .1 / .3 / 1 | .83211 / .82622 / .81544 | .81143 / .81104 / .81824 | .82898 / .82872 / .84581 |
| gate pole .99999, decay .1 / .3 / 1 | .83026 / .81689 / **.77885** | .80221 / .79902 / .78914 | .84434 / .84966 / .82392 |

Blocks, best gate against Adam: .7862/.7806/.8023/.7971/.8017/.7891/.7534/.7261 against .7614/.7554/.7770/.7652/.7809/.7674/.7105/.6902, Adam better in 8 of 8.

Reading. The polar direction is the wrong direction for this stream: best polar 0.80503 is 7.3% behind Adam in 8 of 8 blocks. The 257 features have very different scales and signal levels, and Adam's per-coordinate variance normalization is exactly the preconditioner that stream wants, while the polar step equalizes all singular directions of a 257x256 matrix and so spends its fixed step budget on noise directions. The gate does what the proxy said it does: on top of polar at lr 1e-4 it lowers the risk monotonically in decay and in pole (0.835 -> 0.779, -3.25% against the best polar, 8 of 8 blocks better), and both winning axes sit at the grid edge (decay 1, pole .99999), the same corner as the structured proxy. It does not close the gap to Adam (+3.8%, 0 of 8 blocks): by the pre-registered rule, no stock win for gate_polar. The gate's own contribution on this stream is measured by the gate_adam arms (15 amendments, jobs 6683-6688), and the PERM control of the selected gate config is job 6694 (result appended below).

Cost on this stream: 175-278 s per run against 173 s for Adam (1.0x to 1.6x; the Python loop over 112k bars dominates, the compiled optimizer adds three group launches plus Newton-Schulz on 257x256 and 256x256 per bar).

15.1 addendum, gate on the Adam direction in PPO (jobs 6689-6690, lr 3e-4, pole .9999): decay .3 scored 7736 +-118 (+573 over the in-trainer Adam control, +68 over the base run), decay .1 scored 6074 +-34 (-1089). Opposite signs at adjacent decays with one seed: a null for the gate on Adam in PPO, and evidence that single-seed PPO differences of this size are noise. Every RL gain in this section belongs to the polar direction.

15.1 edges closed (jobs 6679-6682): polar lr 1e-2 6457 (polar's best 9126 at 1e-3 is interior); gate lr 1e-2 at pole .9999 6585 and at .99999 8560; gate lr 3e-3 at pole .99999 9386. The gate's best, 9696 at lr 3e-3 / pole .9999 / decay .1, is now interior on LR and pole. Final RL reading: polar beats Adam by 1460-1960 (three to four noise floors): a real single-seed win for the direction. Gate over polar: +570 at the best points, and at pole .9999 with decay .1 the gate is above polar at all four LRs (6003/5963, 9292/9126, 9696/9065, 6585/6457), a consistent sign but every margin inside the 500 replicate spread; by the amended rule this is a null for the gate in RL, with a suggestive direction that only more seeds could confirm.

15.2 addendum, gate on the Adam direction (jobs 6683-6688, lr 3e-4): pole .999 hurts monotonically in decay (0.76602 / 0.78491 / 0.80622 at decay .1 / .3 / 1); pole .99999 is flat against Adam (0.75088 / 0.75320 / 0.75455, +0.1% to +0.6%, Adam better in 7 of 8 blocks at the best point). On this stream the gate is neutral at the slowest pole and harmful at short ones; no stock win by the rule. Runtime 135-137 s against 173 s for eager torch Adam: the compiled group update is cheaper than eager Adam here, so the gate's arithmetic is not the cost, the polar Newton-Schulz is (175-278 s).

Section 15 verdict. The transfer tests separate the two ingredients. The polar direction: a large real win on PPO (+1460 to +1960 over Adam, seed 1), a 7% loss on the stock stream. The resistance gate: dominant on the proxies (never worse than +0.1%, -13.7% in-dist and -40% off-marginal in the structured regime), a consistent-sign but inside-noise +570 over polar on PPO, a monotone 3% improvement over polar on stocks that still trails Adam, and neutral-to-harmful on top of Adam on stocks. The gate is a regularizer whose payoff is proportional to how much irrelevant weight mass the stream lets accumulate; the structured proxy has a lot, PPO's 17-input trunk and the stock stream have little that Adam does not already control. It does not beat Adam on stocks, and its RL contribution is unresolved at one seed.

PERM control (job 6694), selected gate config on the time-permuted target: 1.00095 (Adam's recorded PERM 1.0061). Nothing learnable stays at the constant predictor; the decayed model is slightly closer to it than Adam, as expected from decay pulling toward zero output. Control passes; the stock reading above stands.
