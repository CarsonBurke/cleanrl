# RethinkB: per-perceptron plasticity as the conditional SNR of the unit's own error signal

Harness: `cleanrl/plasticity/rethink/RethinkB.py` (task generator copied from `hetvar_stream.py`; 8 paired
seeds, 32768 samples, batch 1, held-out MSE vs the noise-free target, every arm at its own best LR from the
grid 5e-6 .. 1e-1 (14 values, 2.2x steps), gates applied to the gradient entering Adam's first moment only).

## 1. Mechanism and derivation

Adam's step is `lr * m / sqrt(v) ~ lr * sqrt(E[g]^2 / E[g^2])`: every parameter already steps by the square
root of the TEMPORAL SNR of its gradient. Conditioning that SNR on the current sample instead of on time is
the per-parameter version of "how predictable is the target from the state". For weight `w_ji` the gradient is
`delta_j * a_i`, so `E[g|state]^2 / E[g^2|state] = E[delta_j|a]^2 / E[delta_j^2|a]`: the input factors out and
every weight into perceptron j shares one gate, the predictable fraction of j's error signal given j's own
input `a`. With a scalar output `delta_j = r * d out/d z_j`, so the unit's error IN OUTPUT UNITS is the residual
`r` itself; the unit's state view `a` (x, h1 or h2) is what differs between layers.
Per sample the Wiener gate is `c_j = mu(a)^2 / (mu(a)^2 + s^2(a))`, `mu = E[r|a]`, `s^2 = Var(r|a)`.
Estimation, self-calibrated (no twin, no random past, no second network, no oracle): two PREQUENTIAL linear
regressions per layer input, predicted before the sample trains them, normalised LMS with per-regressor rate
`(n+1)/tau`, `tau = 1/(1-beta2) = 1000` (the only horizon, shared with Adam):
`mu_hat = u.[a;1]` for `E[r|a]`; `l_hat = w.[a;1]` for `log Var(r|a)` with target `log e^2 + 1.27`
(`-E[log chi2_1]`, exact for Gaussian noise). Because a prequential prediction is independent of this
sample's noise, `E[mu_hat r] = E[mu^2]` is unbiased for the predictable power while `E[mu_hat^2]` over-counts
by the estimation noise; `rho = E[mu_hat r]/E[mu_hat^2]` (per-layer EMA, horizon tau) is both the optimal
shrinkage of the readout and its calibration: `rho -> 0` when nothing is predictable, `-> 1` when the readout
is right. Numerator = posterior `E[mu^2|mu_hat] = rho^2 mu_hat^2 + rho(1-rho) E[mu_hat^2]`; the log-variance
deviation from its mean is shrunk by the same prequential ratio `rho_v`, so spurious heteroscedasticity in a
homoscedastic stream is removed without a decay knob. The gate multiplies the gradient entering Adam's `m`
only (`v` sees the raw gradient), so its absolute level survives Adam's normalisation and the mean gate can be
far below 1 (null cell). Variant `pre`: the same gate normalised to mean 1 by its EMA, entering `m` and `v`.

## 2. Results (paired diff vs Adam at each arm's own best LR; mean, sem, t; 8 seeds)

Adam's best LR was bracketed in every cell (5e-6/1e-5 below its 2e-5 optimum in the hetero cells). Oracle =
true `b^2/(b^2+sigma^2)` under the same plumbing, bracketed everywhere.

| cell | Adam mse | oracle | oracle shuffle | **cond** (best lr) | cond shuffle | cond `pre` | `pre` shuffle |
|---|---|---|---|---|---|---|---|
| hetero 0 | 0.2258 | -0.048 (.005, -9.2) | +0.043 (.008) | **+0.065** (.011, +6.2) lr 1e-1 EDGE | +0.078 (.013, +6.0) | +0.037 (.009, +4.0) | +0.035 (.004, +8.8) |
| hetero 2 | 0.3566 | -0.192 (.022, -8.8) | +0.012 (.003) | **-0.073** (.014, -5.4) lr 5e-2 | +0.010 (.003, +2.9) | -0.082 (.020, -4.2) | +0.024 (.003, +9.2) |
| hetero 2 + signal-scales | 3.571 | -0.540 (.082, -6.6) | +0.211 (.104) | **+0.069** (.139, +0.5) lr 5e-2 | +2.34 (.59, +4.0) | +0.60 (.18, +3.3) | +1.02 (.22, +4.7) |
| hetero 0 + null | 0.0097 | -0.0069 (.002, -3.4) | -0.0013 (.003) | **+0.0008** (.001, +0.8) lr 1e-1 EDGE | -0.0001 (.001, -0.1) | +0.0020 (.001, +1.6) | +0.0005 (.001, +0.7) |
| hetero 2 + switch 0.5 | 0.4312 | -0.181 (.021, -8.7) | +0.019 (.004) | **-0.025** (.010, -2.6) lr 2e-2 | +0.015 (.004, +3.6) | -0.043 (.019, -2.3) lr 5e-4 | +0.034 (.007, +4.7) |

EDGE flags: `cond` at 1e-1 in hetero 0 and null (mean gate 0.003; it wants an even larger nominal LR). Both
are losses/nulls, so no win is voided, but the extended grid was not run. All other optima are interior.
Mean gate levels: hetero 2 0.014 (sd 0.033); hetero 0 0.003; null 0.002; oracle 0.15-0.33.

Reading: in the two cells with a real noise field (hetero 2, switch) the gate beats Adam AND its own shuffle
(shuffle is worse than Adam), i.e. the state->sample correspondence carries the gain: 38% (hetero 2) and 14%
(switch) of the oracle's gain. In signal-scales the arm is neutral vs Adam but its shuffle is +2.3, so the
per-state gate is doing real work there too (it undoes the harm of its own level trajectory). Homoscedastic
is a clear LOSS (+0.065), not null; null cell is null (no absorbed-noise reduction, unlike the oracle).

## 3. Falsified along the way (all hetero 2 unless stated, 8 seeds, own best LR, bracketed)

1. Readout rate `1/tau` per sample (first form): NLMS time constant is `n/eta` = 65k samples, readouts never
   converge, gate 0.005, -0.023 (t -1.8). Fixed by per-regressor rate `(n+1)/tau`.
2. Gate on the Adam STEP (post-optimizer): -0.006 and its shuffle -0.006, identical over the whole grid.
   A gate on a 10-sample-smoothed step has no per-sample correspondence left; the gate must weight the
   gradient. Falsified.
3. Per-UNIT readouts of the raw backprop error `delta_j` (`condraw`, my first "per-perceptron" form):
   hetero 2 -0.077 (shuffle +0.006), switch -0.043, but homoscedastic +0.043 (EDGE) and signal-scales +0.27.
   Diagnostics showed hidden-unit `rho_v` = 0.6-0.9 with hetero 0: `Var(delta_j|a) = sigma^2 (d out/d z_j)^2`,
   so the unit's OWN tanh-gain reads as a noise field and the gate divides the chain rule out of the update.
   Fixed by regressing the error in output units (`delta_j / gain_j = r`), which collapses all units of a
   layer to one gate (they share the input) - honest, but it means "per-perceptron" buys only three state views.
4. Mean readout alone (`--no-var`, precision fixed): +0.003 (t 1.8), shuffle +0.006. At Adam's steady state
   the linearly-fixable part of the residual is ~0 by construction (Adam removes it), `rho` = 0.05-0.2, so the
   mean readout carries almost no per-state signal; ALL of the hetero-2 gain comes from the conditional
   variance. `--no-mean` (unconditional numerator): -0.086; `--no-shrink` (rho = 1): -0.090; neither
   distinguishable from the full form (-0.073 .. -0.087 across runs; sem 0.013-0.020).
5. Per-WEIGHT scalar state (`param`: regress on `(1, a_i)` per weight, bias state-free): -0.002 (t -0.9),
   shuffle +0.002. A single input coordinate explains ~1/17 of the noise field; too weak to gate on.
6. Wide readouts on `[x; h1; h2]` (145 regressors) for every layer, signal-scales: +0.27 (t 2.3), no better
   than the layer-own view (run against the raw-delta form; not repeated after fix 3).
7. `pre` (mean-1) level vs absolute level: same hetero-2 gain (-0.082 vs -0.073); `pre` is worse in
   signal-scales (+0.60) and not null in hetero 0 either (+0.037). The homoscedastic loss is therefore not
   the level schedule: it is per-state structure in `Var(r|a)` that is real from the unit's view (nonlinear
   misfit of the residual is state-dependent) but is learnable by the network, so down-weighting it biases
   the fit toward low-misfit states. From a unit's local view, learnable-but-not-linearly-fixable error and
   noise are the same thing; only an oracle (or memory of past (x, r) pairs, which was ruled out) separates them.

## 4. Verdict

Not a great result; a real but narrow one. The prequentially calibrated conditional-variance gate is genuine
plasticity on the heteroscedastic stream: -0.073 vs Adam (t -5.4), shuffle +0.010, 38% of the oracle's gain,
with no tuned knobs beyond Adam's own horizon. But it fails the homoscedastic null (+0.065, a loss) and does
not reach the oracle's null-cell gain, and signal-scales is neutral. The reason is structural, not a bug:
the quantity the user wants (predictable power of the target at this state, i.e. `b(x)^2`) is not identifiable
from a unit's own state on the current sample once Adam has removed the linearly-fixable part; what remains
readable is `Var(r|a)`, and that conflates noise with not-yet-learned structure. The prequential-ratio
calibration (`rho`, `rho_v`) works as designed (rho_v ~0.05 for the output layer in hetero 0, ~0.9 in hetero 2)
and is a knob-free replacement for twins/EMAs of levels; it is the part of this design worth keeping.
Per-parameter (scalar-state) gating is null; per-perceptron gating collapses to per-layer after the
gain-normalisation that is required for correctness.
