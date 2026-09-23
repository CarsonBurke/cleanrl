# Fresh successor-latent diagnosis v4

This experiment starts every parameter, normalizer, and optimizer from scratch.
It has no checkpoint-loading path. Its fixed random actor is an early-learning
diagnostic, not a trained benchmark policy or a continuation of v3.

The learned 64-dimensional encoder receives full-transition reconstruction and
covariance supervision. It freezes before successor fitting. The successor critic
predicts joint future-latent distributions through vector distributional Bellman
targets. Neither receives reward gradients. The primary actor estimator retains
coordinate-specific counterfactual advantages, with nonlinear utility applied to
each generated outcome before averaging.

## Questions and controls

1. **Loss-scale sensitivity:** two latent decoders have identical 128-wide initial
   weights and data. One fits ordinary reward residuals, the other divides residuals
   by a fixed calibration reward standard deviation. This changes gradient scale,
   clipping, and Adam epsilon effects; it does not change a Hessian condition number.
2. **Representation versus decoder fitting:** compare the primary 256-wide latent
   decoder with a 256-wide decoder receiving complete pre-encoder transition features.
   Inputs are normalized observation/action/displacement features plus a real-transition
   indicator, not raw simulator state. Different input widths mean parameter counts
   are not exactly identical.
3. **Hidden normalization context:** repeat both 256-wide cases with environment
   identity. This detects dependence on per-environment observation/reward coordinates.
   A gap does not by itself prove irreversible information loss; fitting limitations
   and finite model capacity remain possible.
4. **Reference precision:** collect 256 fresh holdout rollouts, preserve fixed projection
   directions, report noise-corrected mean-gradient norm and independent split-mean
   agreement. Low previous SNR was not assumed to scale as pure signal.
5. **Reference bias sensitivity:** compare 64/256/512-step returns with zero and frozen
   value tails on identical starting states, each having 512 real transitions before
   a reset/rollout boundary. Compare the model on the same subset. These checks omit
   boundary states and cannot prove infinite-horizon unbiasedness.
6. **Immediate versus delayed credit:** report the immediate-reward score gradient
   separately from the remaining reference. A random policy may expose mostly action
   cost reduction; passing its gate would not demonstrate learned locomotion credit.

The scalar value model is only an external reference measurement. It neither
supervises the predictive latent nor updates the actor.

## Protocol

- Seed 1; HalfCheetah-v4; 16 environments and 2 environment threads.
- 32 calibration rollouts fit the representation and reference value, and calibrate
  observation/reward normalization. Reward scale uses the last eight calibration
  rollouts. Decoder fitting begins after representation and normalizers freeze.
- 128 fitting rollouts train the successor, primary decoder, and five decoder probes.
- 256 heldout rollouts freeze all networks and normalizers. Counterfactual sampling
  uses four independent donor/noise draws per state and action coordinate.
- 13,647,488 transitions including warmup. This is a fixed-policy diagnostic, not
  an actor-training benchmark.
- Primary gate retains v3 thresholds and additionally requires a positive one-sided
  95% bootstrap lower bound on the dot product of independent half-run mean gradients.
- All frozen models, actor weights, and observation-normalization coordinates are
  saved as evidence. Future experiments must still start fresh.

## Verification and execution

MLQ 7425 passed all 13 CUDA contracts. MLQ 7427 completed the full diagnostic; it depended
on those contracts, uses parallel limit 1, normal priority, and a 60-minute time
limit. No retries or performance-based culling are configured; episodic return
does not improve under this intentionally fixed actor.

Independent review checked probe alignment, paired initialization, freeze boundaries,
coordinate advantages, and discounted horizon/tail indexing. The live full run has
exercised shared observation normalization during calibration successfully.

Known reporting limit: a nondefault configuration with no 512-transition windows
would report one rather than zero sensitivity samples because the denominator is
clamped. The configured 2048-step/1000-step-episode experiment has eligible windows;
this count does not affect its primary full-state gate.


## Completed evidence

All 13,647,488 transitions completed; actor and holdout models stayed unchanged.
The gate failed with a usable reference, not an inconclusive low-signal result.

| Decoder | Heldout normalized MSE |
|---|---:|
| Latent, 128 wide, original loss scale | 0.031340 |
| Latent, same 128 initialization, fixed residual scale | 0.010725 |
| Latent, 256 wide, fixed residual scale (primary) | 0.010499 |
| Pre-encoder features, 256 wide | 0.002929 |
| Latent plus environment context, 256 wide | 0.008810 |
| Pre-encoder features plus environment context, 256 wide | 0.001495 |

Loss scale matters; increased decoder width adds little in this comparison. The
remaining latent/pre-encoder gap is compatible with representation loss or fitting
limitations, not proof of irrecoverable information loss. The primary decoder now
passes the 0.05 threshold on this fresh random policy. This is not a direct repair
measurement on v3's different mature-policy distribution.

Reference SNR is 14.2754; independent split-mean dot lower95 is positive (0.0016063).
Model/reference gradient cosine is -0.04787, relative error is 1.33826, and its
upper95 is 1.36072. Thus better reward decoding did not fix successor action credit.
On identical eligible states, H256 and H512 value-tail references have cosine
0.999976 and relative difference 0.00699; H512 zero/value tails differ by 0.001126.
Those checks make the observed near-orthogonal model gradient difficult to explain
by the tested tail choices alone. They still do not prove an exact infinite-horizon
reference. The immediate-reward gradient opposes part of the full gradient (cosine
-0.28254), so preserving delayed consequences matters here.

Evidence: [gate.json](../runs/HalfCheetah-v4__successor_latent_diagnostic_v4_fresh__1__1789533546203702917/gate.json),
[diagnosis.json](../runs/HalfCheetah-v4__successor_latent_diagnostic_v4_fresh__1__1789533546203702917/diagnosis.json).

The next hypothesis is temporal conditioning: pooling all geometric horizons may
hide weak action-dependent structure in a broadly fitted future distribution.
Only 1% of geometric mass is the immediate transition; action effects can persist
much longer, so that percentage is not itself a diagnosis. A fresh horizon-aware
vector critic can test the hypothesis without routing reward gradients into its
representation or replacing vector supervision with scalar returns.
