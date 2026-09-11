# RethinkA: per-perceptron plasticity from held-out predictability of the unit's own error

## 1. Mechanism (designed form) and why it should read predictability

Unit j (hidden or output) sees its input vector a (all presynaptic activations) and receives the backprop error
d_j = r * J_j(x). Its incoming weights can only absorb the part of d_j that is LINEAR in a (the gradient of a
linear fit), so "how predictable is the target from this unit's state" was operationalised as: how much of d_j at
THIS a is linearly forecastable, relative to what is left over. Two shadow readouts per unit (normalised LMS, one
global rate eta = 0.02 for every cell, never tuned per task):

    mu_j(a)  = U_j.[a,1]           forecast of d_j at this state, read BEFORE the sample updates U (held out)
    s2_j(a)  = exp(V_j.[a,1])      forecast of the held-out squared forecast error at this state (Gaussian-NLL step)
    gate_j   = mu^2 / (mu^2 + s2)  Wiener gain in [0,1]: explained / (explained + unexplained)

With r = b + eps, mu^2 estimates b^2 restricted to what the unit can fix and s2 estimates sigma^2 + what it cannot,
so gate ~ b^2/(b^2+sigma^2) per unit: 0 where d_j is noise given the state, ->1 where it is fully forecastable.
The gate multiplies d_j BEFORE Adam (so noisy samples never enter m, v); it is divided by its running mean
(cumulative, horizonless) so mean plasticity is ~1 and capped at 20x. Shuffle control: lag-1 gate per unit.
Post-Adam application (gate x step) was also implemented and is null (momentum spreads every sample over ~10 steps).

Ablation `prec`: gate_j = 1/s2_j(a) (drop the numerator) -- a per-unit state-conditional precision. Ablation `mu`:
gate = mu^2 only. `r2`: 1 - E[e^2|a]/E[d^2|a] from two log-variance readouts (smoothed R^2). `debias`: subtract the
analytic NLMS misadjustment eta/(2-eta) s2 from mu^2 (positive part). All in `RethinkA.py`.

## 2. Results: paired diff vs Adam, 8 seeds, 32768 samples, batch 1, held-out MSE vs noise-free target

Grid for every arm: 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 (10 points). Adam's optimum in every cell is
interior (5e-4 / 2e-5 [1e-5 is worse: 0.4437 vs 0.3566] / 5e-4 / 5e-4 / 5e-5). Every arm optimum is interior unless
flagged EDGE. Oracle = b^2/(b^2+sigma^2) with the true learnable residual and true sigma, re-measured here.

| cell                     | Adam mse | oracle diff (sem, t)     | wiener arm (sem, t) [lr]         | wiener shuffle (sem, t)  |
|--------------------------|---------:|--------------------------|----------------------------------|--------------------------|
| hetero 0 (homoscedastic) | 0.2258   | -0.0400 (0.0067, -6.01)  | +0.0392 (0.0077, +5.07) [5e-4]   | +0.0465 (0.0113, +4.12)  |
| hetero 2                 | 0.3566   | -0.1877 (0.0212, -8.85)  | -0.1141 (0.0148, -7.73) [5e-4]   | +0.0258 (0.0062, +4.14)  |
| hetero 2 + signal-scales | 3.5707   | -0.3455 (0.0684, -5.05)  | +0.7450 (0.1947, +3.83) [5e-4]   | +0.7408 (0.1816, +4.08)  |
| hetero 0 + null          | 0.0097   | -0.0062 (0.0018, -3.44)  | +0.0087 (0.0017, +5.12) [1e-5] EDGE (low end: "learn nothing" is its best) | +0.0089 (0.0020, +4.56) |
| hetero 2 + switch 0.5    | 0.4312   | -0.1660 (0.0179, -9.29)  | -0.0554 (0.0139, -3.99) [5e-4]   | +0.0408 (0.0067, +6.08)  |

| cell                     | prec arm 1/s2(a) (sem, t) [lr]   | prec shuffle (sem, t)    | gate corr with oracle gate |
|--------------------------|----------------------------------|--------------------------|----------------------------|
| hetero 0                 | +0.0008 (0.0023, +0.35) [5e-4]   | +0.0011 (0.0012, +0.98)  | -0.12                      |
| hetero 2                 | -0.1216 (0.0173, -7.04) [5e-4]   | +0.0312 (0.0053, +5.89)  | +0.32                      |
| hetero 2 + signal-scales | +0.9969 (0.2208, +4.51) [2e-3]   | +0.9491 (0.2098, +4.52)  | +0.15                      |
| hetero 0 + null          | +0.0005 (0.0005, +1.12) [5e-4]   | +0.0015 (0.0006, +2.43)  | -0.31                      |
| hetero 2 + switch 0.5    | -0.0703 (0.0078, -9.00) [5e-4]   | +0.0686 (0.0070, +9.75)  | +0.29                      |

Bounded precision `bprec` (gate = sbar2/(sbar2+s2), sbar2 the unit's cumulative mean of s2; gate in (0,1), cv 0.26
instead of ~3 for raw 1/s2), 3 cells measured: hetero 2 -0.0513 (0.0096, -5.35) [2e-4], shuffle +0.0016 (0.0008);
signal-scales +0.163 (0.105, +1.55, n.s.) [1e-3], shuffle +0.099 (0.052); null -0.0001 (0.0001). Half the hetero-2
gain of raw precision, but the signal-scales loss shrinks from +1.0 to not significant: the level cost is the cv.

Readout-rate sensitivity of the wiener arm, hetero 2 (diff vs Adam / its shuffle): eta 0.002: -0.047 / +0.024;
0.005: -0.075 / +0.029; 0.02: -0.114 / +0.026; 0.05: -0.104 / +0.023; 0.1: -0.080 / +0.017. Per-unit instead of
global level normaliser (eta 0.005): -0.080; cap 100 instead of 20: -0.067. The hetero-2 win is robust to all of these.

## 3. Falsified along the way (all hetero 2 unless stated, own best LR, shuffle in parentheses)

- Post-Adam gating (`r2post`, grid extended to 1e-1): +0.0002 (+0.0004). Null: Adam's momentum carries every
  sample regardless of the gate on the step. Pre-Adam is the only place a per-sample gate acts as information.
- `mu` form (numerator only, no precision): +0.0073 (+0.0164). The linear forecast alone carries nothing usable.
- `r2` form (smoothed held-out R^2 from two log-variance readouts): +0.0288 (+0.0216) at eta 0.02, +0.0522 at 0.05.
  Worse than its own shuffle: the difference of two noisy readouts is anti-informative, not merely noisy.
- `debias` (analytic James-Stein on mu^2): h0 +0.0545, h2 -0.1055, ss +0.9196, null +0.0090, sw -0.0502. No cell
  improves; the numerator's problem is not estimation noise (see below).
- Layer diagnostic (unit-normalised, gate only layer 3 / only layers 1-2): null cell, output-unit gate correlates
  0.71 with the oracle gate (it is f^2/(f^2+s2) almost exactly, since f is linear in h2) and STILL loses
  (+0.0078, = its shuffle +0.0079); hidden-only +0.0023 (n.s.). h0: output-only +0.0055 (n.s.), hidden-only +0.0451.
  ss: output-only +0.151, hidden-only +0.626. The numerator does not help even where it is nearly exact.
- The reason, which I did not anticipate at design time: at any first-order stationary point E[d_j a] = 0 for every
  unit, i.e. the linearly-predictable part of every unit's error is zero BY DEFINITION of convergence, while the
  oracle's b(x) is a nonlinear function of the input and is not zero. So "predictability of the target from the
  unit's own state", read linearly, is a transient (init, teacher switch) and decays to estimation noise (corr with
  the oracle gate 0.03 in hetero 0). What survives is the state-conditional second moment s2(a): heteroscedasticity.
  Wiener and prec agree wherever there is a win; prec is exactly null where it must be (h0, null) and wiener is not.

## 4. Verdict

Not a great result. The honest reading of the table:
- In the two heteroscedastic-noise cells the mechanism is real plasticity, not a schedule: hetero 2 -0.114 (wiener)
  / -0.122 (prec) vs shuffle +0.026/+0.031, i.e. 61-65% of the oracle's gain, t = -7; switch -0.055/-0.070 vs shuffle
  +0.041/+0.069, 33-42% of the oracle. Grids bracket every optimum.
- It carries this from the precision term 1/E[e^2|a], a per-unit, state-conditional noise-field reader. That is
  Gauss-Markov weighting, which the user explicitly said is only a proxy for what they want. The predictability
  numerator, the part that was supposed to answer "can the state predict the target", is falsified in every cell
  that isolates it (h0, null, signal-scales): it is at best null and its noise costs ~cv^2 of effective samples.
- Signal-scales (signal ~ sigma) is a hard loss for both forms (+0.75 / +1.0, equal to their shuffles, i.e. pure
  level cost of a high-variance gate with no information in that cell); the oracle wins there only through b^2,
  which no linear per-unit forecast can supply at convergence.
- Precision-only is the deliverable if one is wanted: null in h0 (+0.0008, t 0.35) and null (+0.0005, t 1.1), wins
  hetero 2 and switch, loses signal-scales. It is not the predictability mechanism the brief asked for, and I do not
  believe a per-unit linear readout of the unit's own input can be one; the b^2 that the oracle uses lives in the
  full-network misfit, which the first-order condition hides from every unit individually.
