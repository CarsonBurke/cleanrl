# plasticity

State-dependent plasticity: each perceptron decides, from its own state on the
current sample, how much of that sample's gradient is allowed to move its
incoming weights.

## Current base

`ppo_32xlr_1mb_noadvnorm_stiglu_sphere_v1` — **13170 ± 955 @50M**, run
`HalfCheetah-v4__ppo_32xlr_1mb_noadvnorm_stiglu_sphere_50M__1__1788662844`.
Config: `num_envs` 16, `num_steps` 2048 (batch 32768), `num_minibatches` 1,
`update_epochs` 10, `learning_rate` 9.6e-3 with `anneal_lr`, `norm_adv` off.

Matched-step trajectory to compare against: 4867 @2M, 6983 @4M, 8574 @8M,
10091 @16M, 11720 @32M, 13220 @50M. **Arms must run the full 50M**: `anneal_lr`
schedules over `total_timesteps`, so an 8M run is not comparable at 8M.

## The learning-rate finding (it invalidated the whole prior lineage)

Measured on the *old* base (`ppo_continuous_action`, lr 3e-4), seed 1, 8M:

| variant | final-20 | @4M | vs base |
|-|-|-|-|
| `ppo_continuous_action` | 7468 ± 124 | 6067 | — |
| `ppo_lrctl_5.1e-4` (1.7x) | 8242 ± 126 | 6842 | **+774** |
| `ppo_lrctl_8.1e-4` (2.7x) | 8369 ± 292 | 7272 | **+901** |

That base was undertuned by ~12% of final score, gains flattening (+774 then
+127). Consequences:

1. `sdplast_v1` is a confirmed null: 7455 ± 292 vs 7468 ± 124.
2. The `statedynlr` line is a learning-rate result. Its v1 ran an admitted
   uniform 2.72x rail; v8 gained +1065 while its gates measurably did nothing
   (realized `rate_std` 0.013 against a 0.31 target). 8455 -> 9763 is one knob.
3. `sdplast_snr_v3` (job 5072, at the tuned 8.1e-4) scored 8610 ± 124 vs the
   8369 ± 292 LR control — **overlapping CIs, and the diagnostics convict it**:
   `lam_mean` 1.354 of a 1.5 ceiling with `lam_std` **0.013**. Every unit
   saturated the tanh against the frozen reference, so the level became a
   uniform ~1.35x LR lift and dispersion collapsed to exactly v8's failure
   number. Three independent implementations, one failure mode.

Root cause of the rail: a frozen scalar reference plus a globally rising SNR
means every unit saturates, and saturation squeezes out the dispersion that was
supposed to be the mechanism.

## v4: remove the uniform degree of freedom

`ppo_continuous_action_sdplast_relsnr_v4.py`. The reference is no longer frozen;
it is the layer's own cross-unit mean of `log sqrt(SNR)` at the current step,
and the bounded log-level is re-centered after the tanh:

    lam_i = exp(span * (t_i - mean_j t_j)),  t_i = tanh(gain * (l_i - mean l) / span)

so `prod(lam)^(1/n) == 1` exactly. The uniform direction cannot be earned, a
1000x global SNR rise leaves `lam` bit-unchanged, and saturation can no longer
eat the dispersion. Not benchmarked on the old base — superseded by v5.

## v5: the first setup where the premise is testable

`ppo_continuous_action_sphere_sdplast_v5.py`, jobs **5082** (full) and **5083**
(`--no-snr-level`), both 50M, `--max-parallel-runs 3 --priority 2`.

The sphere base kills the confound *structurally*. Plastic sites are exactly the
linears whose output `justnorm` immediately renormalizes — `trunk.in_proj` and
every `block.down`, 8 sites, 512 gated perceptrons. A common-mode row rescale is
divided straight back out, so **a uniform level change cannot buy a single point
of score**; only cross-unit dispersion can move the policy. Verified: uniform
rescale changes the policy by 9.3e-10, a dispersed rescale by 4.7e-2 relative —
a ratio of **2.5e5**. Combined with v4's exact geometric-mean-one level, the
uniform direction is dead twice over.

Two factors, deliberately separable:

- **`data` (per-sample, pre-Adam)** — changes the gradient's *direction*. Unit
  `i` predicts `p_{t,i}`, the log noise energy of its own output residual in
  state `t`, from bounded scale-free features of its own pre-activation plus a
  rank-`ctx_dim` view of the layer input. GLS weight `w = exp(-p)`, renormalized
  to per-unit mean one, clamped to `[1/wmax^2, wmax^2]`.
- **`rate` (per-unit, post-Adam)** — changes the *level*, as above. Disabled by
  `--no-snr-level` (job 5083), isolating the direction factor.

Why SNR and not v1's agreement: agreement asks "does this sample agree with what
I have been doing", whose reference is the unit's own recent updates, so without
disjoint data it degenerates into shrinking the batch toward its own mean (v1:
`data_std` 0.40, effect 0). Estimator variance is a property of the estimator,
measurable in-sample with no feedback loop. Gauss-Markov also makes
inverse-variance weighting the *unbiased* minimum-variance choice under
heteroscedasticity, whereas agreement-weighting is biased: a weight predicted
from `state` is independent of the sample's realized noise, a weight read off
the realized residual is not. "Each perceptron decides from its own state" is
exactly the condition that makes GLS legal here.

Init is bit-identical to the base (`to_plastic` copies weights after the base
builder runs; predictors are zero, `w == 1`, `lam == 1`), so at step 0 this file
*is* `stiglu_sphere_v1`.

## Cost

Supervision needs the backward's `delta`, so it cannot fold into the compiled
loss. Measured at the real batch 32768 / minibatch 32768:

| | ms/update | overhead |
|-|-|-|
| base update | 4.00 | 1.00x |
| with supervision | 8.24 | 2.06x |

15250 updates over 50M, so **+65s total** on a multi-hour run. Peak VRAM 0.64
GiB (probes 64 MB + grads), which is what justifies `--max-parallel-runs 3`.
`gate_every` is 1 here: with 1 minibatch and 10 epochs there are only 10 updates
per rollout, so there is nothing to amortize (on the 32-minibatch old base the
default was 4, giving 1.30x for +1.6% wall clock).

`clear_probes()` MUST run on every minibatch, not only supervised ones:
`probe.grad` is allocated in the cudagraph pool, so leaving it set makes the
next backward accumulate into a tensor a later replay has already overwritten
(hard `RuntimeError`, caught before a real run).

## Verification

30 checks in `/tmp/verify_v5.py`, all passing. Load-bearing ones:

- uniform row rescale of any plastic site is a no-op (9.3e-10); dispersed is not
  (2.5e5x larger) — the structural claim
- the trunk walk `gate_supervision` uses matches the real modules' inputs and
  pre-activations **exactly** (0.0), so the duplicated topology cannot drift
- `dL/dW == sum_t w*delta*x` exactly; `dL/dx == W^T delta` **ungated** — gating
  is local, a unit throttles itself and never the network
- bias-free `down` sites: forward bit-identical, correct gradients, and the
  stepper plans around the `None` bias instead of striding into it
- geometric mean of `lam` is exactly one; a 1000x global SNR rise leaves `lam`
  unchanged; a neutral unit takes Adam's step bit-for-bit
- gate NLL reaches every predictor parameter and no network parameter
- fullgraph `reduce-overhead` compile of the whole PPO loss succeeds,
  `probe.grad` survives cudagraph replay, `HostSiTUSphereActor` mirrors the
  plastic actor (1.9e-9)

Earlier suites: 57 checks for v1, 41 for v2, all passing.

## Results

Matched-step on HalfCheetah-v4, seed 1, against the base trajectory. Windows
hold ~101 episodes (CI95 +-48..279), so these differences are real, not noise.

| step | base | v5 full (5082) | v5 `--no-snr-level` (5083) |
|-|-|-|-|
| 8M | 8574 | 8943 | 9185 |
| 12M | 9218 | 9699 | 10096 |
| 16M | 10091 | 10451 | 10064 |
| 20M | — | 11018 | 10275 |
| 32/34M | 11720 @32M | **12482** @34M | 10905 @34M |

**The level's benefit is LATE.** `wonly` leads to ~12M, then the full arm
crosses over and reaches +1577 over it by 34M, and +762 over the base. An
8M-horizon evaluation would have rejected the level outright — the winner-killing
trap `autocull.py`'s CALIBRATION note warns about. **Judge this family at 20M+.**

Why late: the level reallocates step size toward units whose gradients are still
reliable. Early every unit has usable gradient and redistribution is neutral; as
the base flattens (11720 @32M -> 13220 @50M) per-unit SNR differentiates and the
reallocation is what sustains progress. The bottleneck it relieves is
**late-training progress**, not early sample efficiency.

A transient KL gap at 12M (0.040 vs 0.022) was trajectory divergence between two
single-seed runs, NOT a mechanical cost; by 20M both sit at clipfrac 0.198. It
was misread as causal once — `/tmp/kl_budget.py` had already ruled it out.

## v7: correct shrinkage exponent + every perceptron included

`ppo_continuous_action_sphere_sdplast_v7.py`, job **5090**. Two corrections
found by reading v5's own telemetry.

**1. The exponent was wrong.** v5 used `lam ~ SNR^0.5`, justified as "the Wiener
amplitude". Amplitude matching is the wrong objective for a step size: the MMSE
shrinkage of a noisy gradient estimate is `|mu|^2/(|mu|^2+sigma^2) = SNR`,
exponent **one**, and for a noisy quadratic whose curvature Adam has already
absorbed into `sqrt(v)` the optimal per-unit step multiplier is likewise
proportional to SNR. v5's 0.5 halved the level's dynamic range in log space,
which is why `lam_std` sat at 0.171 against a +-0.405 bound — only 42% of the
rail. **The gain was binding, never the bound**, so `--lam-span 2.5` (5087) was
the wrong knob and measured behind v5 at matched step (9282 vs 9699 @12M);
cancelled. Measured dispersion vs exponent: 0.266 / 0.337 / 0.379 for
0.5 / 1.0 / 2.0, i.e. +26% at the derived value, with the bound only starting to
bind past it.

**2. Half the perceptrons were excluded.** v5 gated only the justnorm-fronted
linears (`in_proj`, `block.down`): 8 sites, 512 units. The `gate`/`up` linears —
the NONLINEAR hidden units, where SiTU saturation and dead-unit dynamics live
and where "plasticity loss" is normally measured — never got to decide. v7 gates
them: **20 sites, 1028 units**. Registration order inside `SiTUGLUBranch` is
gate, up, down, and `trunk_site_activations` MUST emit in that same order or
supervision is misrouted to the wrong unit.

These sites are NOT justnorm-fronted, so the level's magnitude is not
architecturally projected out there. Measured rather than assumed:

| coverage | sites | endpoint KL off -> on | clipfrac |
|-|-|-|-|
| v5 (justnorm only) | 8 | 0.00660 -> 0.00657 (x0.996) | 0.0753 -> 0.0755 |
| v7 (+ hidden) | 20 | 0.00650 -> 0.00651 (**x1.001**) | 0.0743 -> 0.0744 |

Still trust-region-free: the branch output is justnorm'd downstream, so the
magnitude is absorbed there even though the individual site is not fronted.

Factorization across live arms: **5089** (`--lam-gain 2.0` on v5 = exponent 1,
8 sites) isolates the exponent; **5090** (v7 default = exponent 1, 20 sites)
adds coverage on top; their difference is coverage alone. **5088**
(`--weight-max 4.0`) tests the direction factor's envelope.

### A correction: Adam does NOT cap state-dependence at per-batch

v7's header claimed "per-sample magnitude cannot survive Adam, so magnitude can
only be conditioned per batch". **That was wrong.** It confused a persistent
uniform rescale with state-conditional modulation. Adam cancels only the
component of a per-sample weighting that is *constant across the batch* AND
*persistent in time*: that factors out of the sum and is divided back out by
`sqrt(v)`. Two things survive:

- **composition** — reweighting changes the summed gradient's DIRECTION;
- **transient scale** — `v` is an EMA over ~1/(1-beta2) ~ 1000 steps, so a batch
  downweighted *today* really does take a smaller step today, because `v` still
  reflects the running average.

So a per-sample gate CAN throttle its own contribution to the sum. v8 acts on
this.

## v8: a sample may throttle its own contribution

`ppo_continuous_action_sphere_sdplast_v8.py`, jobs **5092** (default) and
**5093** (`--weight-max 4.0`).

v7 normalized the GLS weight three separate ways, every one along the batch
axis: the prediction was centered per unit (`raw - raw.mean(0)`), the weight was
renormalized to per-unit mean one (`w / w.mean(0)`), and the target was scaled
by its batch mean (`ell / ell.mean(0)`). Consequence: **if every sample in a
batch looked like noise, `w` was renormalized straight back to mean one.** The
model could rank samples against each other but could never say "this whole
batch is noise, contribute little" -- the most useful thing a noise predictor
can say on a heavy-tailed problem. That restriction was inherited from the old
base, where an unnormalized weight could drift into an unlabelled LR change; on
this base justnorm kills that confound architecturally, so it was pure
capability loss.

Keeping the scale anchored takes care, because `p` is tanh-capped to
`+-log(weight_max)` and so cannot represent an absolute log noise energy (which
drifts toward -27 as gradients shrink). So `p` is split:

    p = p_level + p_state,   p_state = p_cap * tanh(readout / p_cap)
    w   = exp(p_ref - p_state),  p_ref = EMA of p_state.mean(0)

`p_level` is a free per-unit parameter the NLL fits to the absolute level (the
target `ell` is no longer batch-normalized, so the optimum `exp(p) = E[ell|s]`
is absolute). The weight uses only the bounded state part against a slow
reference, so within-batch variation survives, batch-level deviation from the
running average survives (the new capability), and the long-run mean is anchored
at one. `p_state - p_ref` lies in `[-2 p_cap, 2 p_cap]`, exactly the
pre-existing `[1/weight_max^2, weight_max^2]` envelope. At init every gate
parameter is zero, so `w == 1` exactly and the file is still bit-identical to
the base.

**Measured, with v7 as the control** (`/tmp/verify_v8_capability.py`). A latent
feature marks the regime; nothing defines "noisy" -- the unit is only asked to
predict its own residual energy from its own state:

| | noisy-batch mean `w` | quiet-batch mean `w` | ratio |
|-|-|-|-|
| **v8** | **0.503** | 1.999 | **3.98x** |
| v7 | 1.0000 | 1.0000 | 1.00x |

v7's is pinned at exactly one, so the capability was absent by construction, not
merely unused. Within-batch discrimination is preserved (3.98x).

**It is still per PERCEPTRON, not per sample** (`/tmp/verify_v8_perunit.py`).
Planting noise that afflicts one half of the units and spares the other, on the
same samples: afflicted units set `w = 0.50` while unaffected units set
`w = 1.99` on that same sample (3.98x disagreement), each group throttled only
in its own bad regime. The unit-averaged `w` has CV **0.0004** across samples,
so a per-sample-only gate would have nothing to learn from that data at all.
Within-sample dispersion across units: CV 0.61.

`w` rails at exactly `1/weight_max` and `weight_max`, so the envelope now
carries real work rather than being a safety bound -- hence the 5093 arm.

## v9: asymmetric trust envelope

`ppo_continuous_action_sphere_sdplast_v9.py`, jobs **5094** (default) and
**5095** (`--weight-suppress 32.0`).

v8's envelope was symmetric in log space, which is the wrong SHAPE. `w = 1/sigma^2`
and `sigma^2` is heavy-tailed *upward* -- a state can be arbitrarily noisy -- so
`w` is heavy-tailed *downward*. Confidence, by contrast, has a hard ceiling: no
state is quieter than the irreducible noise floor. v8 spent as much range on
inflation, where there is little to gain, as on suppression, where the mass is.

    weight_suppress = 8   ->  w >= 1/8    (near-total throttling)
    weight_inflate  = 2   ->  w <= 2      (short side)

The clamp acts on a DETACHED multiplier under `no_grad`, so a hard asymmetric
clamp is clean: the predictor is trained by its own NLL and never through `w`,
so there is no dead-gradient pathology at the rails.

Measured (`/tmp/verify_v9_capability.py`, `/tmp/verify_v9_perunit.py`):

| | noisy-batch `w` | quiet-batch `w` | ratio |
|-|-|-|-|
| v7 | 1.0000 | 1.0000 | 1.00x |
| v8 (symmetric) | 0.503 | 1.999 | 3.98x |
| **v9 (asymmetric)** | **0.139** | 2.000 | **14.4x** |

Per-unit disagreement holds and widens: on the SAME sample, afflicted units sit
at `w = 0.133` while spared units sit at `2.000` (15.0x), with the unit-averaged
`w` having CV 0.0001 across samples -- so a per-sample-only gate would have
nothing to learn from that data. Within-sample CV across units 0.894.

### Why large inflation is self-limiting, not dangerous

`/tmp/gradnorm_probe.py`: `max_grad_norm` 0.5 clips the GLOBAL gradient norm,
and the pre-clip norm is median **0.595 with 55% of steps clipping** (median
overshoot 1.19x). So boosting one unit mostly REALLOCATES budget away from the
others rather than adding magnitude, and the more inflation is used the more it
saturates into the clip. Bounded by the optimizer -- but for the same reason it
cannot buy a globally larger step, only a larger share of a fixed one.

### What the envelope does NOT provide

"A big error deserves a big update" is IMPORTANCE, not noise. The residual
magnitude is already in `delta`, so that update is already present in the raw
gradient; and the heteroscedastic NLL can only learn how noisy a unit's residual
is in a given state. Importance is not in the objective, so no widening of the
clamp grants it. Real importance-weighting needs a different signal
(consequence/curvature, or `|advantage|`) and would be a separate mechanism.

## Open negative: the "corrected" exponent is measuring worse

At 20M: v5 (exponent 0.5) **11018**, `v7_gain2` (exponent 1) 10409,
`v6_wmax4` 10370.

The v7 derivation gives exponent 1 for the TRUE SNR. But `lam` is driven by an
ESTIMATED SNR, and that estimate is itself noisy; reacting fully to a noisy
estimate over-shrinks, so the optimal exponent against an estimate is strictly
below the one against the truth. v5's 0.5 is therefore defensible as a
bias-variance compromise, and the "correction" may have removed a useful extra
shrinkage. Single seed and the level pays late, so this is not final -- but the
derivation should NOT be treated as settled in its favour.

## The diagnostic that should have come first

### Rethink: what the diagnostics actually established

The mechanism now lives in `cleanrl/shared/state_plasticity.py` and is
deliberately NOT PPO-specific: it sees only `(layer input, pre-activation,
incoming pre-activation gradient)`, which exist identically in LLM
pretraining, time-series forecasting and PPO, and it works with any optimizer.
Reference material for the family is in `cleanrl/plasticity/reference/`.

`noisy_stream_diagnostic.py` was rewritten: the old version did ~6 host syncs
PER STEP, which is why 20k steps cost 11s instead of ~1s. It is now sync-free
with independent seeds run as a batch dimension, so a 20k-step, 16-seed
measurement takes seconds.

#### Results, 20k steps, 16 seeds, selectivity = w[signal] / rms w[distractors]

| linear stream | selectivity | raw statistic separation |
|-|-|-|
| adam | 6.01 +- 0.69 | -- |
| energy (the v5-v9 objective) | 5.43 +- 0.99 | **0.94x** |
| snr + debias | **15.60 +- 3.08** | 21.3x |
| statewiener + debias | 15.13 +- 3.19 | 14.6x |
| adam at 0.158x lr (C5 control) | 5.92 +- 0.56 | -- |

| regime stream | selectivity | quiet/noisy level on the signal coordinate |
|-|-|-|
| adam | 4.08 +- 0.97 | -- |
| energy | 4.60 +- 1.35 | 1.02x |
| snr (running moments) | 11.81 +- 5.41 | 1.03x |
| statewiener (state-conditioned) | 13.60 +- 5.83 | **1.73x** |

#### Four things these establish

1. **The family's objective was blind, and mildly harmful.** `energy` separates
   signal from distractor by 0.94x -- i.e. not at all -- and scores BELOW plain
   Adam. This is C1 confirmed to two decimals, and it retroactively explains
   every flat result from v5 through v9.
2. **The gain is real and not a learning-rate change.** The level's realized
   uniform factor was 0.158, and Adam run at exactly 0.158x lr scores 5.92 --
   indistinguishable from Adam at 1.0x (6.01). So none of the 15.60 comes from
   the uniform component. First confound-free result in this family.
3. **State-conditioning is the only thing that buys reliability awareness.**
   Running moments cannot express it (1.03x) because they average over regimes;
   the state-conditioned predictor gets 1.73x. Conversely, on the linear stream
   the two are equal (15.60 vs 15.13), exactly as the mu^2/E[g^2] identity
   predicts, since that stream has no state to condition on.
4. **Squaring an estimated mean is biased the wrong way.** `E[mu_hat^2] = mu^2 +
   Var(mu_hat)` and `Var(mu_hat)` grows with the noise, so the naive statistic
   is inflated exactly where data is least reliable. Undebiased, the regime
   statistic came out INVERTED. Fixed two ways: analytically for a batch
   estimator (the variance of the mean is already computed for the
   denominator), and by a sign-randomized twin for a streaming estimator, whose
   true mean is exactly zero with identical energy, sparsity and horizon.
   Debiasing raised the linear raw separation from 14.0x to 21.3x.

#### Two measurement errors worth remembering

- **A central anchor measures the envelope, not the statistic.** With 4095/4096
  coordinates unreliable, anchoring the level on its arithmetic mean pins the
  reference AT the unreliable value, so the good coordinate saturates the
  envelope and the separation reads as the ceiling regardless of the statistic's
  quality. This produced a bogus "2.49x" that was only caught because a
  reviewer questioned it. The diagnostic now reports the RAW statistic
  separately from the realized level, and anchors near the top.
- **An aggregate metric over mostly-zero quantities reports noise.** The regime
  separation averaged over all 4096 coordinates read 0.79x -- apparently
  backwards -- because 4095 of them have a debiased signal clamped to exactly
  zero in BOTH regimes. Restricted to the signal coordinate it reads 1.73x.


`cleanrl/plasticity/noisy_stream_diagnostic.py`, jobs **5098** (linear) and
**5099** (hidden). HalfCheetah score cannot say WHICH capability a mechanism
has, so this reproduces the Oak/Sutton noisy-stream task where noise rejection
IS the task: 4096 Bernoulli(0.01) features of which one carries signal, targets
carrying N(0,5) noise plus +-1 spikes. Correct behaviour is `w[0] -> 1`,
`w[1:] -> 0`. The yardstick is an ORACLE told which inputs are signal, not a
historical algorithm.

Selectivity = signal weight / distractor weight magnitude, 20k steps:

| method | signal | distractor | ratio |
|-|-|-|-|
| sgd | 0.213 | 0.0288 | **7.4** |
| adam | 0.586 | 0.0946 | **6.2** |
| rowgate (v9 rule, per unit) | 0.590 | 0.0951 | **6.2** |
| colgate (v9 rule, per input) | 0.580 | 0.0941 | **6.2** |
| snradam (Adam's own m,v as SNR) | 0.428 | 0.0700 | **6.1** |
| oracle | 0.595 | 0.000 | inf |

### Three independent defects, all measured

**1. Residual ENERGY cannot distinguish signal from noise.** For a distractor
connection the residual energy is `(delta*x_i)^2 = delta^2`; for the SIGNAL
connection it is *also* `delta^2`. Identical. What differs is whether
`delta*x_i` is consistently SIGNED. The entire v5-v9 objective is a
heteroscedastic energy/GLS predictor, so it is blind to signal by construction
-- `colgate` matches Adam to three significant figures even after the mechanism
is moved onto the axis the task rewards. No envelope, axis or capacity change
can repair this; credit assignment needs the FIRST moment.

**2. The pre-Adam channel is nearly closed** (`/tmp/channel_probe.py`). Asking
for a 0.125x step:

| channel | 1 batch | 10 batches | sustained |
|-|-|-|-|
| pre-Adam gradient weight (v8/v9 `w`) | 0.897 | 0.594 | -> 1.00 |
| post-Adam step level (`lam`) | 0.124 | 0.124 | 0.125 |

beta1=0.9 lets one batch move the step by only ~10%; beta2 cancels a sustained
rescale outright. v8/v9 were pushing on a channel that attenuates them ~10x.

**3. Adam's 1/sqrt(v) is itself hostile to credit assignment.** It equalizes
step size per coordinate, amplifying a distractor's small noisy gradient up to
signal scale. Measured: Adam is WORSE than plain SGD on selectivity (6.2 vs
7.4) despite much larger learned weights.

### What does work: consistency, matched horizon, per observation

`snradam` reuses Adam's own moments as an SNR and fails (6.1) because beta1 and
beta2 have different horizons: for a 1%-sparse feature `|m/sqrt(v)|` spikes to
~1 right after it fires, so it measures burstiness. `snrgate` fixes both
problems -- `mu` and `u` are EMAs at a MATCHED beta, and they advance only on
steps where the coordinate is ACTIVE, so the horizon counts observations rather
than wall-clock time -- and applies `SNR / mean(SNR)` POST-optimizer:

    60k steps: selectivity 7.6 (vs adam 6.2)
    realized level while active: signal 1.980, distractor 0.795 = 2.49x

First mechanism in this family to show ANY measured separation on this stream.
Note the failure mode of the first attempt: `snr_ref` averaged over INACTIVE
coordinates (SNR identically 0), which saturated every level at the ceiling and
turned the mechanism into a 2x LR change -- the same uniform-LR confound this
family has now fallen into four times. The signal level rails at the 2.0
ceiling while distractors sit at 0.795 rather than near the 0.125 floor, so the
bound is currently binding on the wrong side and the SNR estimate is not yet
confident enough to throttle noise hard.

### Where the remaining bottlenecks are

1. **Clipping, inherited from the base.** clipfrac 0.187 (base), 0.198 (arms):
   about one sample in five contributes no policy gradient, from 10 epochs over
   a single 32768 minibatch at 32x LR. Plasticity does not spend this budget,
   but no estimator improvement recovers a clipped sample, so it caps the
   ceiling.
2. **Gradient noise floor — the resource being monetized.** Per-unit SNR
   1.2e-4..1.3e-3 (`sigma^2/|mu|^2` ~ 800-8000), so the batch mean still carries
   ~17% relative error per unit at 32768 samples. `w_std` 0.44/0.45.
3. **Not the critic.** EV 0.985 base vs 0.951-0.958 arms. Mild inversion worth
   watching: plasticity slightly *worsens* value fit while improving return,
   consistent with GLS downweighting noisy return targets.
4. **Not compute.** SPS 37663 -> 34109 (v5) = 8-9%; v7 costs 4.07x on the
   update, +199s over 50M (~15%), peak 1.35 GiB.

### The level is trust-region-free (measured, not argued)

At 20M: `approx_kl` 0.0268 (full) / 0.0292 (wonly), `clipfrac` 0.198 / 0.198,
against the base's 0.0225 / 0.187. Identical. A transient gap at 12M (0.040 vs
0.022) was trajectory divergence, not a mechanical cost -- `/tmp/kl_budget.py`
settles it directly, holding data and seed fixed:

| level variant | endpoint KL | step_sq ratio |
|-|-|-|
| none | 0.00660 | — |
| sustained, random `lam` | 0.00660 | 1.066 |
| resampled every update | 0.00659 | 1.062 |
| **from real measured SNR** | **0.00659** | **1.184** |

The level inflates the squared step by 18% and moves KL by **zero**. Cause: the
plastic sites are justnorm-fronted, so the magnitude the level adds is largely
common-mode and gets projected straight back out. The property that kills the
learning-rate confound also makes the level free in trust region. Redistribution
is all that survives -- which is exactly the premise.


## Watch

`sdp/*_lam_std` and `sdp/*_w_std`. On this base `lam_mean` is pinned to one by
construction, so dispersion is the only thing left to look at. Current: lam_std
0.171 (stable), w_std 0.44.

Judge at 20M and beyond, NOT at 8M: the level is neutral-to-negative before
~14M and only pays afterwards.

FAIL: below the base's matched-step trajectory at two consecutive checkpoints
past 16M, `w_std` pinned at the `wmax` rails, or `lam_std` collapsed below v8's
0.013 -- that last would mean the mechanism is inert whatever the score says.
