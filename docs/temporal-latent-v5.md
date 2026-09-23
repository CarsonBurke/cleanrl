# Temporal-latent critic v5

Hypothesis: explicitly representing future time and balancing predictive training
across horizons helps retain action-dependent consequences that a pooled future
distribution misses. v4's reward decoder and reference passed their accuracy and
signal thresholds while its policy gradient remained nearly orthogonal; this motivates
a new predictive model, not another scalar decoder adjustment.

Every run starts all networks, normalizers, and optimizers from scratch. No checkpoint
is loaded. The encoder, full vector targets, and primary loss-scaled reward decoder
are retained. Scalar returns do not train the encoder or successor model.

The critic G(s,a,noise,h) samples a full future-transition latent at exact integer
horizon h. h=0 is the current transition. It receives a logarithmic time feature,
multiple time-decay features, and an immediate-transition indicator. Predictive
training samples equally among horizon bands starting at 0,1,2,4,8,16,32,64,128,256.
Within each band, h follows the conditional geometric law; the last band is unbounded.

With k available real transitions, h<k uses the observed latent at t+h. For h>=k,
bootstrap at the physical next state of transition t+k-1 with residual horizon h-k.
True termination gives an absorbing zero future. The residual horizon decreases,
but these exact-horizon backups have no discount attenuation; discount contraction
is not claimed for their fitted errors.

The actor uses vector coordinate-counterfactual advantages as before. Each pair
shares its horizon, donor actions, and generator noise. When a band is sampled
uniformly, multiply its coordinate advantage by B*(gamma^L-gamma^U), with gamma^U=0
for the unbounded tail. This restores the original geometric objective. **Balanced
training weights and actor integration weights have different roles.** Nonlinear
utility is evaluated on each latent outcome before any averaging.

Retain v4's independent split-reference check and same-state tail sensitivity.
Additionally compare model and observed score gradients separately at horizons
0,4,16,64 on identical eligible states. These measurements do not substitute for
the primary aggregate gate; short/long errors must not be hidden by averaging.

The fixed-policy protocol remains 32 calibration, 128 fitting, and 256 frozen holdout
rollouts. The policy is a newly initialized random policy, so this is a mechanism
experiment rather than an actor-training benchmark. No throughput or benchmark-score
improvement is asserted before evidence.

Independent review found the horizon sampling, geometric integration weights,
residual-horizon backups, and coordinate-gradient scaling consistent. Fixed-horizon
reward MSE compares sampled predictions with sampled physical outcomes and includes
sampling variance; it is not a conditional-mean error estimate.

MLQ 7429 contains CUDA numerical contracts (parallel limit 1, 20-minute limit).
MLQ 7430 is the dependent full fresh experiment (parallel limit 1, 60-minute limit).
Both use normal priority and no automatic retries. At submission they were waiting
behind another exclusive job.


## Completed evidence

MLQ 7429 passed all 14 CUDA contracts. MLQ 7430 completed 13,647,488 transitions
with fresh initialization and no checkpoint loading. The primary gate failed.
Encoder reconstruction NMSE was 0.04154, decoder NMSE 0.00990, and reference SNR
14.7829; these passed their individual thresholds. The aggregate model gradient
had cosine 0.53141 but norm ratio 11.9391 and relative error 11.4391 (upper95 11.7209).
Improved direction does not compensate for severely misestimated credit.

| Exact future horizon | Reference SNR | Gradient cosine | Model/reference norm | Relative error |
|---|---:|---:|---:|---:|
| 0 | 187.28 | 0.99507 | 1.03943 | 0.10867 |
| 4 | 47.93 | 0.71032 | 0.67413 | 0.70481 |
| 16 | 5.61 | 0.68762 | 5.12066 | 4.49210 |
| 64 | 0.86 | unresolved | unreliable ratio | unreliable relative error |

The immediate-action gradient is substantially correct despite noisy per-sample
forecasts. At horizon 16, measured action credit is much smaller than predicted.
Horizon 64 has insufficient reference signal to quantify a reliable ratio or angle;
it does not establish an exactly zero true gradient. These findings support a
failure to learn the temporal evolution of action effects. Accurate decoding on
real transitions does not establish accurate decoding on generated latents; decoder
extrapolation, inaccurate future distributions, and direct action conditioning
remain possible causes. The horizon-16 reference uses directly observed rewards,
so its discrepancy does not depend on a reference value bootstrap.

Same-state H256/H512 value-tail references had cosine 0.999981 and relative
difference 0.00621; H512 zero/value tails differed by 0.00137. Tested tail choices
therefore do not plausibly explain the aggregate 11.44 relative model discrepancy.
The reference remains conditional on this policy, state subset, and projections.

Evidence: [gate.json](../runs/HalfCheetah-v4__temporal_latent_v5_fresh__1__1789534321550448613/gate.json),
[diagnosis.json](../runs/HalfCheetah-v4__temporal_latent_v5_fresh__1__1789534321550448613/diagnosis.json).

No actor training was launched. A next model should test whether making the initial
action affect future predictions only through a predicted next state reduces these
persistent conditional errors. That is a new architectural hypothesis, not a result
established here. Arbitrary gradient rescaling or forced temporal decay would hide
the observed credit error rather than establish correct vector advantages.
