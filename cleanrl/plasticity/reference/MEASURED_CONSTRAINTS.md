# Hard constraints, all MEASURED in this repo (not opinions)

Target setting: PPO on MuJoCo HalfCheetah-v4, Adam(lr~9.6e-3, beta1=0.9,
beta2=0.999, eps=1e-5), batch 32768 in ONE minibatch, 10 epochs, global
`clip_grad_norm_(., 0.5)`. Networks are small MLP trunks (width 64) on a unit
hypersphere trunk (`justnorm` re-projects every stream, so any common-mode row
rescale of a layer's weights is a NO-OP on the policy: verified 9.3e-10
relative change).

## C1. Residual energy cannot distinguish signal from noise
For a distractor connection the per-sample residual energy is `(delta*x_i)^2 =
delta^2`. For the SIGNAL connection it is ALSO `delta^2`. Identical. Only the
first moment differs (whether `delta*x_i` is consistently signed). A mechanism
whose objective is a heteroscedastic variance/energy predictor is therefore
blind to signal by construction. Measured: such a mechanism scores 6.2
selectivity on the stream below vs Adam's 6.2 -- literally no effect -- even
after being moved onto the per-input axis.

## C2. Pre-optimizer gradient weighting is nearly cancelled by Adam
Asking for a 0.125x step by scaling the GRADIENT:
  1 batch -> realized 0.897   10 batches -> 0.594   sustained -> 1.00
Asking for it by scaling the REALIZED STEP (post-Adam):
  1 batch -> 0.124            10 batches -> 0.124   sustained -> 0.125
Reason: beta1=0.9 means one batch moves `m` by only 10%; `v` divides out any
persistent rescale. Consequence: a per-sample gradient weight can change the
summed gradient's DIRECTION (survives) but not its MAGNITUDE (does not).

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
score. So a proposal MUST either provably have no uniform component, or report
its realized geometric mean so the confound is visible.

## C6. Throughput
8M-50M env steps per experiment. A mechanism costing >~2x per update is very
hard to justify. Memory per extra full-size optimizer-state-shaped buffer is
acceptable (Adam already keeps two).

# The diagnostic you must predict against

`cleanrl/plasticity/noisy_stream_diagnostic.py` in the repo implements the
blog's linear stream: 4096 Bernoulli(0.01) features, only feature 0 predictive,
target = x[0] + (+-1 spike at 1%) + N(0, 5). Batch size 1, online.
Metric = selectivity = (learned weight on feature 0) / (rms weight over the
4095 distractors). Reference numbers at 20k steps:

  sgd 7.4 | adam 6.2 | energy-based per-unit gate 6.2 | energy-based
  per-input gate 6.2 | oracle (told which input is signal) infinite

There is also a `hidden` task: 4096 inputs -> ReLU hidden -> scalar, where only
the leading `signal_inputs` coordinates are predictive.
