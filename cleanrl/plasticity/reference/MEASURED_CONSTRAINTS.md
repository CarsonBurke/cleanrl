# Historical measurements and their scope

Interpretation corrected 2026-09-07; see `../FAMILY.md`'s proxy audit. Historical
numbers below are not universal optimizer constraints or current corrected-proxy
measurements.

Target setting: PPO on MuJoCo HalfCheetah-v4, Adam(lr~9.6e-3, beta1=0.9,
beta2=0.999, eps=1e-5), batch 32768 in ONE minibatch, 10 epochs, global
`clip_grad_norm_(., 0.5)`. Networks are small MLP trunks (width 64) on a unit
hypersphere trunk (`justnorm` re-projects every stream, so any common-mode row
rescale of a layer's weights is a NO-OP on the policy: verified 9.3e-10
relative change).

## C1. Residual energy alone is not a general credit-assignment signal
For a Bernoulli feature, `(delta*x_i)^2 = delta^2` when it is active, zero
otherwise. This does **not** prove identical conditional energies for predictive
and distractor features: conditioning on activity can change the residual law.
The historical energy rule tied Adam at selectivity 6.2 on one sparse stream;
that result does not exclude useful state-conditional variance estimation or
gradient-vector covariance.

## C2. Pre-optimizer gradient weighting is nearly cancelled by Adam
Asking for a 0.125x step by scaling the GRADIENT:
  1 batch -> realized 0.897   10 batches -> 0.594   sustained -> 1.00
Asking for it by scaling the REALIZED STEP (post-Adam):
  1 batch -> 0.124            10 batches -> 0.124   sustained -> 0.125
Adam approximately cancels a fixed positive rescale once both moments have
equilibrated (apart from epsilon). Sample-dependent weighting can change both
the summed direction and transient magnitude. It is not generally equivalent
to a persistent uniform rescale; post-Adam scaling directly controls the
realized step.

## C3. Adam's 1/sqrt(v) is hostile to credit assignment
It equalizes per-coordinate step size, amplifying a distractor's small noisy
gradient up to signal scale. Measured on the stream: Adam selectivity 6.2 is
WORSE than plain SGD's 7.4.

## C4. The global gradient clip couples units
`clip_grad_norm_` is a uniform downscale (relative weights preserved exactly,
min/max per-row scaling 1.000000/1.000000) and never scales up, so it cannot
undo suppression -- but it binds on 55% of steps, so amplifying one unit
REMOVES step budget from all others. An untouched unit's realized gradient fell
0.0628 -> 0.0158 purely because its neighbours grew.

## C5. Any uniform component is a learning-rate change, and it is a trap
This family has produced a fake "win" FOUR times by accidentally applying a
near-uniform multiplier, which is just an LR change. Controls: the tuned LR is
~8.1e-4 vs default 3e-4 on the old base and that alone was worth +12% of final
score. Report realized magnitude and dispersion, sweep each arm's LR, and use
state-independent/matched controls. A geometric mean alone does not rule out a
time-varying effective-LR explanation, and forced mean-one gates can erase
useful global adaptation.

## C6. Throughput
8M-50M env steps per experiment. A mechanism costing >~2x per update is very
hard to justify. Memory per extra full-size optimizer-state-shaped buffer is
acceptable (Adam already keeps two).

# Sparse feature-selection diagnostic, not the whole family premise

`cleanrl/plasticity/noisy_stream_diagnostic.py` in the repo implements the
blog's linear stream: 4096 Bernoulli(0.01) features, only feature 0 predictive,
target = x[0] + (+-1 spike at 1%) + N(0, 5). Batch size 1, online.
Primary metrics now are exact clean-target reconstruction error, distractor
leakage and their mean cross term. Selectivity is supplementary: it can improve
while recovering almost no target. Historical selectivity numbers at 20k steps:

  sgd 7.4 | adam 6.2 | energy-based per-unit gate 6.2 | energy-based
  per-input gate 6.2 | oracle (told which input is signal) infinite

The hidden-layer variant is implemented separately in `../hidden_stream.py`.
The sparse diagnostic's `hidden` option is unsupported. Support-informed Adam
knows the relevant input coordinates; its score is a reference, not an upper
bound on achievable optimizer performance.
