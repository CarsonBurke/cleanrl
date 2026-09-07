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


## Chart: predictions on the noisy stream (`reference/noisy_stream_predictions.png`)

Regenerate with:

```
.venv/bin/python cleanrl/plasticity/noisy_stream_diagnostic.py --method all \
  --steps 20000 --seeds 4 --eval-steps 4096 --adaptive-z \
  --plot-window 500 --plot cleanrl/plasticity/reference/noisy_stream_predictions.png
```

Linear stream, 4096 inputs (1 predictive, 4095 distractors, all 1%-sparse),
targets carry N(0,5) noise, **batch size 1, single pass, no replay**. Test MSE is
against the NOISE-FREE target; the zero-predictor scores 0.0112, so any ratio
above 1.0x means the learner is worse than predicting nothing.

| method | test MSE | vs zero-predictor | signal w | distractor rms w |
|---|---|---|---|---|
| oracle (told the answer) | 0.00219 | 0.20x | 0.562 | 0.000 |
| **veto, self-calibrated z** | **0.00604** | **0.54x** | 0.426 | 0.0072 |
| graded, fixed z=5 | 0.00901 | 0.81x | 0.121 | - |
| graded, self-calibrated z | 0.01088 | 0.97x | 0.245 | 0.0102 |
| sgd | 0.04176 | 3.74x | 0.173 | 0.0287 |
| snr (v5-v7 statistic) | 0.08084 | 8.89x | 0.732 | 0.0443 |
| statewiener | 0.08628 | 9.49x | 0.729 | 0.0457 |
| energy (v8/v9 objective) | 0.26911 | 29.6x | 0.460 | 0.0802 |
| adamw | 0.31268 | 28.0x | 0.517 | 0.0872 |
| adam | 0.36029 | 32.3x | 0.562 | 0.0937 |

Three results worth keeping:

1. **Graded is not the problem.** Matched at 20k steps, graded `(t^2/(t^2+z^2))^p`
   lands within 19% of the binary veto (0.0090 vs 0.0076 at fixed z). An earlier
   claim in this file that the veto won by being BINARY was wrong; it won on the
   **statistic**. Corrected.
2. **The floor was never the driver either.** Removing the envelope floor from
   the `snr` gate: 0.0808 -> 0.0729 at 20k, and 0.2664 -> 0.3050 at 100k. The
   earlier "3.6x from deleting the floor" compared 20k against 100k. Corrected.
3. **The threshold needs no hyperparameter.** With D coordinates and O(1)
   carrying signal, the upper quantile of `t` across coordinates IS the null's
   extreme value. `--adaptive-z` calibrates on it and raises signal retention
   0.121 -> 0.245 (graded) and 0.176 -> 0.426 (veto), the latter also improving
   MSE 0.0076 -> 0.0060.

### Why the statistic, not the envelope, was the whole story

`t = |sum_t G| / sqrt(sum_t G^2)` is the exact self-normalised cumulative. For a
coordinate with a consistent conditional mean it grows as `sqrt(n) * mu/sigma`,
without bound; for a distractor it stays `O(1)` forever. Separation therefore
IMPROVES with evidence. Every v5-v9 statistic was a ratio of EMA moments at a
fixed horizon, whose separation is capped by the horizon no matter how much
evidence arrives -- which is why five 50M-step MuJoCo runs were flat.

### Batch-size invariance (algebraic, not measured)

At batch B the per-update gradient has mean `mu` and sd `sigma/sqrt(B)`, so after
`n` samples (`n/B` updates) `t = sqrt(n/B) * mu*sqrt(B)/sigma = sqrt(n)*mu/sigma`:
**`t` depends on samples seen, not on how they are batched.** The mechanism's
evidence accrual is therefore batch-size invariant, and everything in the table
above was measured at batch size 1.

### Nonstationarity: measured, and the remaining bottleneck

Signal moved to a fresh coordinate at 50% of training (veto, self-calibrated z):

| evidence decay | test MSE | w[new coord] | \|w\|[stale coord] |
|---|---|---|---|
| 0 (exact cumulative) | 0.01007 | 0.110 | 0.146 |
| 1e-3 | 0.01236 | 0.052 | 0.077 |
| 1e-2 | 0.01712 | 0.020 | 0.029 |
| sgd control | 0.04183 | 0.120 | 0.085 |

- The mechanism still beats SGD 4.2x through a regime change, and it does learn
  the new coordinate.
- **Forgetting is a false fix.** Decay strictly hurts: it caps attainable `t` at
  ~`sqrt(1/d)`, which costs admission of the true signal and buys nothing.
- **The real defect is that a license is never revoked.** Stale weight (0.146)
  exceeds the newly-learned weight (0.110): the obsolete coordinate keeps a
  large `t` from its history, so it stays admitted and simply drifts. Note the
  correct response to a reversal is MORE plasticity there, to unlearn -- so
  admission and trust are different quantities, and this family has only ever
  modelled one of them.

### Ranked bottlenecks of the graded rule

1. **Signal throttling.** Best retention is 0.426 vs oracle 0.562; admission
   costs ~`z^2 (sigma/mu)^2` effective firings. Asymmetric hysteresis (admit at
   low z, evict at high z) attacks this directly.
2. **No revocation / no reversal detector.** Measured above. Fix: two-timescale
   evidence, admitting on `max(t_fast, t_slow)` while their DISAGREEMENT raises
   plasticity instead of lowering it.
3. **`z` is global.** It is differentiable in the graded form, so `z_i` can be
   meta-learned per unit from that unit's own state -- the premise applied one
   level up, and foreclosed entirely by a binary veto.
4. **Column axis only.** Evidence is per (unit, input) pair, i.e. two buffers,
   the same footprint as Adam. Nothing yet gates the row axis.


## Mirror plasticity (`cleanrl/shared/state_plasticity.py`)

The graded rule that finally beats the binary veto. Level = one minus the
false-discovery proportion, calibrated by a sign-randomized twin
`R = sum_t eps_t g_t` whose order statistics ARE the null distribution of
`A = sum_t g_t`. No threshold, no reference, no learned estimator, no
hyperparameter. Verified 16/16 in `/tmp/verify_mirror.py` (contract tests).

| linear stream, 20k, batch 1 | test MSE | vs zero | signal w | level on signal |
|---|---|---|---|---|
| oracle | 0.00219 | 0.20x | 0.562 | - |
| **mirror (graded)** | **0.00397** | **0.44x** | 0.418 | **1.0000** |
| veto (binary) | 0.00604 | 0.54x | 0.426 | - |
| graded `(t^2/(t^2+z^2))^2` | 0.01088 | 0.97x | 0.245 | - |
| sgd | 0.04176 | 3.74x | 0.173 | - |

The level reaching EXACTLY 1 is the whole improvement: a sigmoid-shaped gate
throttles the entry it has just certified, which cost 2.7x MSE and half the
signal weight. Binariness was never the lever -- the earlier claim in this file
that the veto won by being binary is retracted above.

### Retained evidence is a feature (the "stale license" question, settled)

| signal MOVES mid-run | test MSE | current coord w | stale \|w\| | cost of the stale weight |
|---|---|---|---|---|
| mirror | 0.01205 | 0.0209 | 0.1011 | **+1.4% of error** |
| sgd | 0.04183 | 0.1201 | 0.0849 | +0.2% |
| mirror, signal LEAVES then RETURNS | 0.00755 | **0.1449** | 0.0035 | - |
| sgd, leaves then returns | 0.04287 | 0.1179 | 0.0839 | - |

Pruning the obsolete weight recovers 1.4% of the error, so the retained license
is not a defect worth engineering against. And it pays: a returning coordinate
reaches 7x the weight of a fresh one from 34% LESS post-change time, because its
evidence is still on the books. What remains in the move case is admission
LATENCY, which is information-bound, not a revocation failure.

### The batch-size law (corrects an earlier claim in this file)

Exactly, and verified to 2.2% at B in {1, 8, 64, 1024}:

    t = sqrt(n) * mu / sqrt(sigma^2 + B mu^2)

So `t` is batch-INVARIANT only while the per-update SNR is small
(`B << sigma^2/mu^2`), and power decays as `1/sqrt(1 + B mu^2/sigma^2)` past it.
My earlier "batch-size invariant" claim dropped the `B mu^2` term.

**This quantitatively explains the v5-v9 MuJoCo failures.** Measured per-unit
`sigma^2/mu^2` in PPO is 800-8000 against a batch of 32768, so
`1/sqrt(1 + 32768/800) = 0.15`: PPO's batch destroys ~85% of the statistic's
power before any rule sees it. A plasticity mechanism of this kind needs small
batches or per-sample evidence; on a 32768-sample batch there is nothing left to
find. This is a prediction, not a post-hoc story.

### Fan-in: no wide ensemble required

| fan-in D | mirror | sgd | level (signal / distractor rms) |
|---|---|---|---|
| 32 | 0.00324 | 0.00672 | 1.0000 / 0.2720 |
| 64 | 0.00241 | 0.00664 | 1.0000 / 0.2606 |
| 256 | 0.00325 | 0.00867 | 1.0000 / 0.1094 |
| 1024 | 0.00319 | 0.01382 | 1.0000 / 0.0545 |
| 4096 | 0.00397 | 0.04176 | 1.0000 / 0.0398 |

The null is pooled across the tensor, so an `H x D` layer supplies `H*D` draws.
At fan-in 64 -- our actual trunk width -- it still beats SGD 2.8x.

### Two pushes, one negative and one qualitative

**GLS-weighted evidence: measured no-op.** Weighting each observation by
`1/sigma_hat^2` changed nothing on the homoscedastic stream (0.00397 -> 0.00406)
AND nothing on the heteroscedastic regime stream (0.00878 -> 0.00890). Reason:
`t = |sum g| / sqrt(sum g^2)` is already studentized, so a noisy observation
inflates numerator and denominator together. The v5-v9 "reliability weighting"
program was paying for something the statistic gives away free.

**Input pooling (`granularity="input"`): qualitative.** At a per-connection SNR
of 0.05 over 400 steps (`t ~ 1.0`), per-connection evidence NEVER opens -- which
is correct refusal. Pooling the energy `sum_j t_ji^2` across a width-64 layer,
calibrated by the twin's pooled energy, reaches level 1.000 by step 100 with 10x
separation from the useless inputs. For any real layer this is the difference
between the mechanism working and not working, since latency falls as
`1/sqrt(n)` and pooling multiplies `n` by the layer width.

### Real market data (`cleanrl/plasticity/stock_stream.py`)

SPY 5-minute bars from `trading_bot_0/long_data/bars/SPY.300.bars` (468,054 bars,
2016-08-22 to 2026-08-19; packed `TBBARS01`, 64-byte header, 36-byte records
`<q6fI>` = epoch-ms, OHLC, volume, vwap, trade count). 224 causal features
(7 channels x 32 lags, each divided by a trailing EWMA so the input is stationary
without look-ahead), target = next-bar return over trailing volatility.

Because the intrinsic noise is unknown, every learner is run TWICE: on the real
stream and with the targets PERMUTED in time -- a ground-truth null that
preserves the marginal and destroys predictability. A learner that learns only
the learnable part must sit at 1.0 on permuted data and below 1.0 on real data;
a learner that absorbs noise exceeds 1.0 on both. Prequential (predict-then-
update) scoring at batch size 1, so every prediction is out-of-sample and there
is no split to leak through. Each method is reported at ITS OWN best LR over a
shared grid, because this family has already produced four LR-artifact "wins".


## Inside a hidden layer (`cleanrl/plasticity/hidden_stream.py`)

Everything before this used a single linear unit, where "each perceptron
decides" describes one perceptron. A hidden unit is the harder case: its
incoming gradient is a PRODUCT of the input and a backpropagated error whose
sign flips as the layer above reorganises. Dense teacher on `x ~ N(0, I)`,
1024 inputs of which 4 matter, 64 hidden units, noise_std 2.0, batch 1, single
pass; test MSE against the NOISE-FREE teacher; each method at ITS OWN best LR;
all 15 configs in one vectorized pass (13.9s).

| method | best lr | test/zero | \|w\| useful:junk | level useful / junk |
|---|---|---|---|---|
| oracle | 3e-3 | 0.0852 | 25.9 | 1.0 / 0.0 |
| **mirror, input-pooled** | 1e-3 | **0.1457** | 18.7 | 1.0000 / 0.0244 |
| mirror, per-connection | 1e-3 | 0.1719 | 16.3 | 0.1847 / 0.0114 |
| adam | 3e-4 | 0.2705 | 11.6 | - |
| sgd | 3e-4 | 0.3098 | 4.6 | - |

1.86x better than Adam in a real hidden layer, and input pooling beats
per-connection exactly as the latency argument predicts: per-connection reaches
only level 0.185 on the useful inputs (it under-admits) while pooling reaches
1.0000. Mirror is also far less LR-sensitive: at 3e-3 Adam collapses to 0.554
while mirror holds 0.160.

### The applicability boundary (measured, and it predicts our MuJoCo results)

Same task, 40k steps, sweeping how many of the 1024 inputs actually matter:

| useful / 1024 | adam | mirror_input | advantage | level on useful |
|---|---|---|---|---|
| 4 (0.4%) | 0.3117 | 0.1952 | **1.60x** | 0.854 |
| 16 (1.6%) | 0.3009 | 0.2249 | 1.34x | 0.647 |
| 64 (6%) | 0.4035 | 0.3589 | 1.12x | 0.293 |
| 256 (25%) | 0.4148 | 0.3852 | 1.08x | 0.079 |
| 1024 (100%) | 0.4384 | 0.4371 | **1.00x (inert)** | 0.169 |

Monotone decay to exactly inert. The mechanism is a USELESS-INPUT REJECTOR, and
its ceiling is set by how much of the input is useless. Consequences:

* **MuJoCo/PPO: predicted inert.** 17 observation dimensions that all inform the
  policy is the 100% column. This is a prediction that matches the five flat
  50M-step runs (v5-v9) rather than a story told after them. Note it is inert,
  not harmful (0.4371 vs 0.4384), so it is a safe addition -- but it cannot be
  the source of a HalfCheetah win, and no amount of tuning changes that.
* **A second, independent reason PPO is hostile:** with 8M steps / batch 32768 /
  10 epochs / 32 minibatches there are ~78k updates, so `t ~ sqrt(78000) * mu/sigma`
  is in the hundreds for nearly every parameter. Evidence is overwhelming
  everywhere, the FDP goes to zero everywhere, and the level saturates at 1: the
  rule is inert by saturation as well as by input density.
* **Market data and LLM streams are the favourable end**, where the fraction of
  genuinely predictive inputs is tiny. That is where this belongs.

### Market data, honest verdict (`cleanrl/plasticity/stock_stream.py`)

40 configs (4 methods x 5 LRs x real/permuted) in ONE pass over 468,021 SPY
5-minute bars: 65.7s, 3.5us per config-bar. The earlier one-process-per-cell
version cost 4 minutes PER CELL; the sweep is vectorized over configs because
the stream is sequential but the configs are not.

* **No learnable signal found.** No method beats the zero-predictor at any LR
  (best: mirror 1.00019, adam 1.00085, sgd 1.00263) and there is no
  real-vs-permuted gap anywhere (all |gap| <= 0.003, half the wrong sign). This
  task cannot verify the learning claim.
* **It does discriminate on noise refusal**, which is the premise: mirror is
  closest to 1.0 at every LR and degrades far more slowly -- at lr 1e-3, mirror
  1.032 vs adam 1.153 vs sgd divergent.
* **A calibration defect surfaced that synthetic data hid.** With weights frozen
  at zero (`--lr-grid 0.0`), mean level was 0.696 on the PERMUTED stream where
  the null is exactly true and levels should be ~0 (synthetic pure noise gives
  0.008). Two hypotheses falsified by measurement: it is not the restoring
  gradient `E[xx']w` (persists at `w = 0`), and it is not shared-sign
  co-signing of the twin (an independent sign per coordinate gives 0.690 vs
  0.696). Causal mean-centring of the channels and target -- absent before, so
  `E[y x_i] = mu_y m_i != 0` survived any permutation and the null was NOT null
  -- halves it to 0.344 and opens a real/permuted separation that did not exist
  before (0.828 vs 0.344). The residual is under investigation; prime suspect is
  serial dependence, against which an independent per-step sign flip is
  anti-conservative by construction (it destroys in the twin the dependence the
  observed accumulator keeps), with a block sign flip as the standard fix.


## Batch size, learning rate, and budget REALLOCATION

Tested the claim that a per-connection noise model should buy small batches and
high learning rates directly. Sparse signal (4 of 1024 inputs), noise_std 8.0,
FIXED 60,000-sample budget so `steps = samples / batch`, every method at its own
best LR from a shared grid.

| batch | adam | mirror_input | oracle |
|---|---|---|---|
| 1 | 0.6703 | **0.3266** | 0.3205 |
| 8 | 0.6384 | 0.3134 | 0.2743 |
| 64 | 0.7287 | 0.3243 | 0.2803 |
| 256 | 0.8137 | 0.3458 | 0.3151 |

* **Batch 1 with the mechanism beats Adam at EVERY batch size**, including 256
  (1.96x the best Adam anywhere), and lands within 2% of the oracle.
* Mirror is nearly batch-INSENSITIVE (0.313-0.346 across 1..256): the level does
  the noise averaging that the batch dimension was there to do.
* At a fixed sample budget, larger batches make Adam WORSE (0.670 -> 0.814):
  batching buys noise reduction by spending updates, and that trade is negative
  here. "Use a big batch to resolve variance" is not free.

### Learning-rate headroom, and why the level cap was the real ceiling

| lr (batch 1) | adam | mirror_input | oracle |
|---|---|---|---|
| 1e-4 | 0.6554 | 0.3326 | 0.3035 |
| 3e-4 | 0.6717 | 0.3228 | 0.2805 |
| 1e-3 | 0.8685 | 0.3404 | 0.2811 |
| 3e-3 | 1.5353 | 0.6503 | 0.7517 |
| 1e-2 | 3.6227 | 1.2070 | 3.6328 |

Mirror runs at 30x Adam's LR and still matches Adam's best (0.650 vs 0.655), so
the usable band is much wider. But mirror's own optimum is at the same 3e-4 as
Adam's -- AND SO IS THE ORACLE'S. That is the diagnostic: the LR ceiling is set
by the noise on the coordinates that carry signal, not by the junk, and a level
bounded by `1 - FDP <= 1` can only ever SHRINK steps, so it cannot raise them.

**`mirror_alloc` removes that ceiling** by rescaling the levels to mean one, so
shut connections FUND larger steps on confident ones and the layer's total step
budget is conserved rather than reduced. Sweeping the amplification cap
(sparse, noise 8, batch 1, 60k samples):

| alloc cap | best test/zero | at lr |
|---|---|---|
| 2 | 0.3216 | 3e-4 |
| 8 | 0.2790 | 3e-5 |
| 32 | 0.2351 | 3e-5 |
| **128** | **0.2003** | 1e-5 |
| oracle, best over its OWN LR grid | 0.2805 | 3e-4 |

**1.40x better than a mask that was told which inputs matter.** This is not a
disguised learning-rate effect: the oracle was swept over the same grid and had
the same freedom. What reallocation adds is per-connection differentiation AMONG
the useful weights -- reliability structure that a binary mask cannot express.
The best LR falls as the cap rises, as expected when the realized step is
`lr * level`.

**Honest cost:** reallocation gives up the property that the rule never
amplifies, so unlike the capped level it CAN take budget from `clip_grad_norm_`
and its stability is no longer free. It also does not rescue the dense case: at
1024/1024 useful inputs `mirror_alloc` is worse than Adam (0.797 vs 0.709),
so the density boundary above still stands and RL is still the wrong target.


## The state-conditional reframe, and its falsification (novelty_stream.py)

The whole v1-v9 / mirror / kalman / spike lineage computes a SCALAR per weight from
gradient statistics accumulated over TIME. No such object can express "move for these
inputs and not for those", because that is a statement about directions in input
space. So the premise was never actually under test.

Three harnesses in a row could not test it either, for the same reason each time:

| harness | why it cannot test state-dependence |
|---|---|
| `noisy_stream_diagnostic.py` | iid stream, only structure is junk columns -> rewards feature selection alone |
| `hidden_stream.py` | same, inside a hidden layer |
| `novelty_stream.py` with ONE shared teacher | every region wants the same mapping, so no unit faces conflicting demands. Measured: sequential vs shuffled identical (0.0653 vs 0.0655), and per-unit alignment with past regions is a coin flip (49.5% harmful, every state feature |r| <= 0.014) |

Fixed by giving each region its OWN target function, which creates real conflict:
retention 0.687 vs acquisition 0.195, against 0.133/0.027 for the shared teacher.

### What was tried on the validated conflict task, and what happened

| arm | overall | verdict |
|---|---|---|
| `adam` | 0.6201 | control, LR swept |
| `novelty` (|z-mu|/sigma, mean-1) | 0.1404 vs 0.1206 adam (shared teacher) | LOST, and lost to its own uniform-scalar control |
| `familiar` (inverted) | 0.1251 | also lost -> the rule was injecting noise, not reading competence |
| `learned` (per-unit readout, one-step lookahead meta-gradient) | 0.6183 | INERT: level pinned 1.000, dispersion 0.000 |
| `statebin` (mirror twin statistic indexed by state cell) | 0.6137 | INERT: level saturates at 1, because in a dense task every cell has real signal |
| `precision` (per-unit anisotropic, predictive-coding form) | 0.6530 | LOST at 8 seeds (looked like 0.5982 vs 0.6258 at 4 seeds -- noise) |
| `precision_shared` (one geometry for the whole layer) | 0.6210 | ties adam, and BEATS the per-unit version |
| `precision_diag` (= an RMS/EMA accumulator) | 0.6645 | worst |

### Two mistakes worth keeping on the record

1. **Budget conservation is an invented mechanic.** Renormalising levels to mean one
   was adopted defensively, to dodge the LR confound. It also makes a rule exactly
   blind to a global shift: when every unit's state is novel, the normalisation
   cancels it. The LR confound belongs in a CONTROL ARM (`uniform`, matched realized
   plasticity), not in the mechanism.
2. **Gating the outgoing weight deadlocks the unit.** `w2` starts at zero, so the
   incoming gradient starts at zero, so a level read off that gradient pins both at
   zero forever. First `precision` run scored exactly 1.0000 (nothing learned) for
   this reason. The projection belongs in the input space of the incoming row only.

### Where the geometry probe landed (`/tmp/probe_precision.py`, all checks pass)

The learned geometry is genuinely anisotropic -- median max/min eigenvalue ratio 543,
60.7% of directions damped below 0.5, and `lambda=0` reproduces the baseline
bit-exactly. But the per-perceptron content is weak: median |cos| between different
units' most-damped directions is 0.77, i.e. most units commit to the SAME direction,
because a shift-12 region centre dominates every unit's input. That is why
`precision_shared` ties `precision`. The anisotropy is real; the per-unit autonomy is
mostly not, on this task.

### Standing conclusion

No state-conditional mechanism yet beats Adam under matched controls. The single
positive RL result in this whole family remains the LATE-training gain from the level
(+743 by 20M, inert before ~14M, at the point where the base curve flattens), and no
synthetic harness built here has ever reproduced it -- which is itself the strongest
evidence that these synthetic tasks do not represent what RL is bottlenecked by.


## The minimum bar: recover the learnable target (reference/spike_recovery.png)

The chart is the test that should have gated everything else. On the noisy stream
(4096 features, 1 predictive, target +-1 noise) the learnable target is a sparse
spike train. Reproducing it -- not scoring well against it -- is the bar.

| method | test MSE vs clean target | x zero-predictor | signal w | null w |
|-|-|-|-|-|
| sgd | 0.09194 | 8.44x WORSE | 0.634 | 0.0469 |
| adam | 0.44816 | 41.14x WORSE | 1.027 | 0.1043 |
| mirror | 0.00227 | 0.23x | 1.046 | 0.0070 |
| softveto (floor 0.125) | 0.00190 | 0.17x | 0.852 | 0.0059 |
| smoothgate (t2/(t2+z2))^4 | 0.00117 | 0.11x | 0.683 | 0.0004 |
| **veto (hard, z=5)** | **0.00037** | **0.03x** | 1.025 | 0 |
| **gradedveto (hinge)** | **0.00039** | 0.04x | 0.893 | 0 |
| **softhinge (differentiable)** | **0.00039** | 0.04x | 0.893 | 1.6e-12 |
| oracle (told the answer) | 0.00043 | 0.04x | 1.027 | 0 |

Three rules reach the oracle. Everything this family built before them does not:
SGD and Adam are worse than predicting nothing, and `mirror` -- the strongest
mechanism here on every earlier metric -- fires FALSE spikes at the wrong times.

### What actually decides it (both of my earlier readings were wrong)

I claimed the requirement was suppression to EXACTLY zero. It is not. The
requirement is a MAGNITUDE: with D distractors the absorbed output noise scales
with the rms gate level over them, so anything well below `1/sqrt(D) = 0.016` is
indistinguishable from zero in consequence. `softveto`'s floor of 0.125 is 8x too
large and `t2/(t2+z2)`'s 0.04 is 2.5x too large -- that is the visible noise floor
in those rows, and it is a magnitude error, not an argument for binariness.

The second requirement is the one that kills the naive smooth form: the gate must
SATURATE at 1 once evidence accrues. `(t2/(t2+z2))^4` suppresses nulls to 4e-4 but
keeps taxing the signal, dragging its weight to 0.683 and tripling the error. So:

    certainty = softplus(k * (1 - z^2/t^2)) / k

saturates at 1, crosses the calibrated boundary smoothly, puts nulls at
softplus(-k)/k = 1.6e-12 for k=24, and is differentiable everywhere. It scores
identically to the hard veto (0.00039 vs 0.00037) while remaining a continuous,
meta-learnable function -- which a hinge or a threshold cannot be. `z` is set by
the sign-randomized twin, so there is no scale hyperparameter.

`t^2` is built from sufficient statistics accumulated over the stream, so this is
batch-agnostic by construction: batch 1 is the native case, and batch size changes
only how fast evidence arrives.

### Independent audit of the earlier headline (PrecisionSkeptic, precision_audit.py)

Recorded because it invalidates results in this file rather than supporting them.
64 true seeds (teachers, centres, stream, noise, init all resampled; arms paired
within seed), LR grid 1.2e-5..1.97e-1 so nothing sits at an edge:

- Adam's own optimum is lr 9.6e-5 -> 0.4022, BELOW the floor of the grid every
  earlier experiment used. Restricting Adam to that window costs it
  -0.2204 +-0.0491 on its own. `precision` recovered 13% of an LR
  misspecification; on the full grid it LOSES by +0.0938 +-0.0165.
- `novelty_stream.py`'s seeding is init-only (one teacher, one centre draw, one
  shared stream; the `seed` field in configs is never read), which understates
  seed SD 4.2x (0.0203 vs 0.0857 resampled).
- Null calibration by splitting ONE arm into disjoint halves (true gap exactly 0):
  at n=4 a spurious |gap| >= 0.028 appears 65-77% of the time. Both the +0.0276
  "win" and the 8-seed -0.0329 "loss" are draws from the same null.
- Per-unit ORACLE ceiling (each unit told per sample whether its own step lowers
  the mixture loss; controls at matched mean and matched dispersion): batch 8 /
  shift 12 has 3.4% headroom and the oracle beats plain Adam by NOTHING
  (-0.0046 +-0.0101). Batch 1 streaming has -0.0228 +-0.0052 = 5.6% and its
  level-only component is zero, i.e. ALL of the headroom is per-unit. The
  no-conflict shared-teacher task has the largest relative headroom, 17.0%.

Consequences adopted: (1) every arm sweeps its own LR and the winner may not sit
at a grid edge; if the control's optimum is at the floor the grid is wrong.
(2) paired seeds with the TASK resampled, not just init. (3) the flagship config
was the worst possible one -- batch 8, conflict, shift 12 -- whose ceiling is 10x
below what 4 seeds can resolve. (4) "conflict is the only thing state-dependence
can buy" is falsified: batch-1 streaming is where the headroom lives, which is
exactly the regime the requirements name.


## What "effective learning rate" decomposition actually says (hidden_stream, batch 1)

Every rule in this family is a multiplier on the baseline learning rate, so write
it as the only decomposition that matters:

    lr_eff(i) = L * G(t) * c_i

`L` baseline (swept), `G` a global factor shared by all units, `c_i` the unit's
own relative plasticity. Arms differ ONLY in G and in what c_i is allowed to read.
Paired seeds, 1024 inputs / 4 useful / noise 8 / batch 1, each arm's own LR grid
1e-5..1e-2 with no edge winners:

| arm | G | c_i reads | s1 | s2 | s3 |
|-|-|-|-|-|-|
| ownfree | 1 | own evidence, own twin | 0.344 | 0.174 | 0.619 |
| input | 1 | own evidence, layer twin | 0.323 | 0.175 | 0.515 |
| uniform | 1/mean(c) | NOTHING (mean only) | 0.833 | 0.470 | 0.990 |
| ownnull | 1/mean(c) | own evidence, own twin | 0.178 | 0.073 | 0.167 |
| credit | 1/mean(c) | own evidence, layer twin | 0.177 | 0.055 | 0.169 |

1. The FAMILY-WISE threshold is not load-bearing. Thresholding each perceptron
   against its OWN sign-randomised twin ties the layer-max version (0.178/0.073/
   0.167 vs 0.177/0.055/0.169). The layer-max was a population normalisation --
   it coupled every unit's certainty to the noisiest coordinate anywhere in the
   layer -- and removing it costs nothing. Autonomy is free; take it.
2. G MUST NOT be a constant: G=1 with L re-swept costs 2.0-3.7x. But G alone,
   with zero per-unit dispersion, is the WORST arm measured (0.833). The two
   factors multiply; neither substitutes for the other.
3. The reason is NOT budget conservation (my error, twice). Mean certainty drifts
   DOWN as the layer becomes selective, so a fixed L means the mechanism silently
   decays its own global learning rate and stalls. G = 1/mean(c) states: spend
   exactly the baseline's total step budget, redistribute only. That makes the
   mechanism LR-confound-free BY CONSTRUCTION, which is the property whose
   absence in the protocol faked four earlier results here.

Also measured, same harness, paired 6/6 seeds: making the evidence invariant to a
unit's own outgoing weight magnitude (`softhinge_credit`) beats `softhinge_alloc`
on every seed (0.188/0.054/0.175/0.079/0.092/0.074 vs 0.199/0.062/0.176/0.092/
0.150/0.082). A hidden unit's gradient is delta_h * x with delta_h carrying
w2_h, which changes sign and scale while evidence accumulates, so raw sums mix
credit across eras when the unit meant different things downstream. Best arms now
sit at 0.174 vs oracle-mask 0.287 and adam 0.644; the remaining gap to the 0.04x
achieved on a LINEAR stream is ~4.5x and is a DEPTH/estimator problem, which is
25x more headroom than the entire measured per-unit ceiling (5.6%).

`softhinge_free` (G proportional to t^2/z^2) is a bad idea and the numbers say so
(0.252 vs 0.185): t^2 grows linearly in samples, so that G is an unintended
learning-rate ramp, not a plasticity decision.


## Verdict on the premise, from the target domain (ppo_signal_legibility.py, dense_boundary.py)

RLBridge harvested the premise's own statistic from REAL PPO updates (2.5M steps,
256 actor+critic units, exact per-(unit,sample) pairs, signal = dL/dz via
retain_grad on the real clipped loss, control = each unit's signal column
permuted within the same update so only the state->signal correspondence dies):

- EXPRESSIBLE: F of the gain-invariant SNR 7.20-8.08 vs permuted 0.89-0.95
  (8.1-8.6x), 96% of 256 units above the permuted p95. E[signal | own state] is
  genuinely above its own sampling error.
- PERSISTENT and STRENGTHENING: cross-window correlation of the per-(cell,unit)
  pattern, centred within unit, r = 0.20 -> 0.27 -> 0.32 -> 0.35 across 500k
  windows vs permuted ~0. A causal rule CAN estimate it and still have it later.
- BUT THE MAGNITUDE IS NOTHING: eta^2 = 1.0e-05. One hundred-thousandth of the
  variation in a unit's teaching-signal reliability is explained by its own state;
  reliability range across a unit's cells is 3.9e-05 on [0,1].
- AND THE APPARENT SIGNAL IS A TAUTOLOGY: log-sd of the signal's noise across a
  unit's state cells is 0.946, of which the unit's own tanh slope (1-a^2)
  accounts for 0.844, i.e. ~89%. A rule that reads its own preactivation to scale
  its step is, to first order, re-applying its own derivative -- which the
  gradient already contains. This trap is invisible unless the slope is measured
  alongside, and it is the most likely explanation for every "state-dependent"
  result in this family's history.

Independently, dense_boundary.py mapped the selectivity axis against input
density (8 paired seeds, no edge winners): softhinge_alloc - adam is
-0.530+-0.086 at 4/1024 useful, NULL (+0.009+-0.024) at 128/1024, and HARMFUL
(+0.248+-0.114) at 1024/1024. At MuJoCo's shape (17 inputs, all useful) it is
+0.008+-0.014, i.e. inert, and mirror_alloc is the exact identity (level 1.0000
everywhere). A dense regime-change test (teacher redrawn mid-stream, 17/17):
adam 0.104/0.411 at noise 2/8, best mechanism 0.101/0.420, oracle 0.093/0.396 --
the ORACLE MASK's total headroom is 3-11% and no mechanism reaches it.

CONCLUSION: per-perceptron plasticity conditioned on the unit's own state is
measured-null as a route to MuJoCo score. It is expressible but worth 1e-5 of
variance, it is ~89% the unit's own activation derivative, and the junk-rejection
capability that makes it pay on synthetic streams is exactly inert at the input
density MuJoCo has. Recorded so this is not rediscovered a sixth time.

WHAT SURVIVED, and it is not state-dependence: after a regime change, per-parameter
evidence with AMPLIFICATION (level above 1 for newly-reliable parameters) beat the
oracle mask on the sparse switch test, 0.31x vs 0.55x the zero-predictor, because
a mask knows where the signal is but still relearns at baseline rate. That is a
temporal capability -- allocation of step size in time, not across units -- and it
is the only measured effect in this family that a correctly-controlled oracle does
not already dominate.

## pc_stream.py: gate strength scaled as 1/B (fixed), and the cross-batch result

`u += (pc_lr / batch) * du` where `du` already SUMS over the batch: per-step
increment batch-invariant, step count falls as samples/batch, so the local model's
total movement scaled as 1/B. At B=256 the gate was inert (multiplier 0.982,
dispersion 0.020) vs firing at B=1 (0.728, 0.424). Every cross-batch comparison
before this compared a live gate to a dead one. Fixed: `u += pc_lr * du`
(bit-identical at B=1). Verified dispersion 0.423 / 0.429 / 0.412 at B=1/32/256.

Standard dense net, field 1.0, LR swept per arm from 3e-5, 16 paired seeds:

| B   | adam   | pc     | pc_shuffle | pc - adam            |
|-----|--------|--------|------------|----------------------|
| 1   | 0.5919 | 0.5880 | 0.5927     | -0.0039, t -0.35     |
| 32  | 0.6530 | 0.6568 | 0.6470     | +0.0038; shuffle wins|
| 256 | 0.6905 | 0.7464 | 0.7282     | +0.0560, t +8.74     |

Null at B=1, harmful with batch, and beaten by its own shuffled-correspondence
control at every B >= 32. This is with gate strength matched, so it is not the
defect-4 artifact any more.

## The direct test, never run before: HalfCheetah-v4 @ 8M, seed 1

`ppo_continuous_action_pcbatch_v1.py` (standard dense trunk, per-(unit,sample)
gate inside the minibatch gradient sum). Gate fires on the real run: at 114k
c_mean 0.514, unit dispersion 0.098, sample dispersion 0.382 -- note c_mean 0.51
is a ~2x effective-LR cut on gated layers, so the pc-vs-off comparison is
LR-confounded; pc vs scalar (same mean, sample dependence destroyed) and pc vs
shuffle (same marginals, correspondence destroyed) are the clean comparisons.

mlq 5141 pcbatch_v1 | 5142 --pc-shuffle | 5143 --pc-scalar | 5144 --pc-off
(max-parallel 3, no compile). References: baseline 8278 @8M, incumbent 10362.

Seed 1 alone: off 6122 / pc 6049 / scalar 5717 / shuffle 5671 -- and the same
code's off arm equals `ppo_continuous_action_hostactor` (6121.7, deterministic),
while an earlier numerically different build of the same algorithm scored 7468 and
the reference 8278: single-seed HalfCheetah spread at 8M is ~1000+. One seed
resolves nothing here, so seeds 2-8 were added (mlq 5149-5176; 44k SPS, 3 min/run).

**RESULT, 8 seeds per arm, HalfCheetah-v4 @8M (mean +-CI95, sd):**

| arm                              | @2M  | @4M  | final           |
|----------------------------------|------|------|-----------------|
| off  (plain PPO, same code)      | 4497 | 6137 | 7128 +-488 (704)|
| scalar (per-unit LR only)        | 4387 | 6034 | 7015 +-485 (700)|
| shuffle (marginals kept)         | 3838 | 5451 | 6440 +-377 (544)|
| pc (the premise)                 | 3729 | 5099 | 6214 +-250 (360)|

The premise is the WORST arm: -914 (-13%) vs plain PPO, behind at every
checkpoint. Decomposition: the per-unit mean-level change costs nothing (scalar
-113, noise); within-batch per-sample dispersion costs ~-690 regardless of which
unit it is attached to (shuffle); the correct state->unit correspondence adds a
further -226 on top of the shuffle. On the standard dense PPO net, per-perceptron
state-conditional plasticity is not null on HalfCheetah -- it is harmful, and the
harm is in exactly the component the premise is about.
