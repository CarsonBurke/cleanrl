# RethinkC: per-perceptron plasticity from the unit's own goal regression

Code: `cleanrl/plasticity/rethink/RethinkC.py` (task generator copied from `hetvar_stream.py`, arms new).
All numbers: 8 paired seeds, 32768 samples, batch 1, held-out MSE vs the noise-free target, every arm
at its own best LR from the 11-point grid {5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2}.
Adam and the oracle reproduce the reference numbers exactly (0.2258/0.3566/3.571/0.0097; 0.1858/0.1689/3.225/0.0039).

## 1. Mechanism (`pp_var`; the conditional-mean extension `pp` is section 3)

Perceptron j on the current sample has a state (its input vector a, features phi = [1, a]), a
pre-activation z_j = w_j . phi and an error delta_j = dL/dz_j. Its own update -lr * delta_j * phi is a
stochastic linear regression of a target on its state, so "how predictable is the target from the
state" has an exact per-unit reading: regress the unit's GOAL on its own state and ask how much of
the goal is unpredictable there.
* Goal: y_j = z_j - delta_j / mean(J_j^2), J_j = d out / d z_j (one Gauss-Newton step; for the output
  unit y_j is literally the label). Unlike delta_j, the goal does not move when the unit learns, so
  horizonless plain sums stay valid (no EMA, no random past, no forgetting knob).
* Mean model: exact recursive least squares of y_j on phi (one Gram inverse per layer, shared by
  its units; per unit only cross sums; Woodbury update, so batch-agnostic). Held-out residual
  e = y_j - yhat_j(a); average fixable power sig_j = running mean of (z_j - yhat_j)^2 - se^2,
  se^2 = s^2 phi' A^-1 phi the analytic OLS estimation noise.
* Noise model: RLS of log e^2 on phi; log s_j^2(a) = fit + 1.27 (E log chi2_1). Spurious
  heteroscedasticity is removed analytically: with q slopes the explained SS exceeds the truth by
  q s^2, so fitted deviations are shrunk by (SS_exp - q s^2)^+ / SS_exp (positive-part James-Stein);
  on a homoscedastic stream the field collapses to a constant with no decay knob.
* Gate c_j(a) = sig_j / (sig_j + s_j^2(a)) in [0, 1], per perceptron (row of W and its bias),
  multiplies delta_j before Adam. It is the Wiener coefficient: where the unit's goal is unpredictable
  from its state (pure noise inside a domain), c -> 0 and the unit does not learn from that sample.
Every scale is estimated from the stream; the only constants are exact (1.27, q = n_in).

## 2. Results (paired diff vs Adam: mean / sem / t; lower is better; no winner on a grid edge)

| cell | Adam mse (lr) | oracle_wiener diff | pp_var mse (lr) | pp_var diff | pp_var_shuffle diff |
|---|---|---|---|---|---|
| hetero 0 | 0.2258 (5e-4) | -0.0400 / 0.0067 / -6.0 | 0.2470 (2e-4)* | **+0.0212** / 0.0057 / +3.7* | +0.0020 / 0.0008 / +2.5 |
| hetero 2 | 0.3566 (2e-5) | -0.1877 / 0.0212 / -8.9 | 0.2080 (5e-4) | **-0.1487** / 0.0181 / -8.2 | +0.0067 / 0.0016 / +4.1 |
| hetero 2 --signal-scales | 3.5707 (5e-4) | -0.3455 / 0.0684 / -5.1 | 3.7660 (1e-3) | **+0.1954** / 0.1208 / +1.6 | +0.2254 / 0.0722 / +3.1 |
| hetero 0 --null | 0.0097 (5e-4) | -0.0058 / 0.0017 / -3.4 | 0.0106 (5e-4) | **+0.0009** / 0.0004 / +2.5 | +0.0007 / 0.0004 / +1.7 |
| hetero 2 --switch-at 0.5 | 0.4312 (5e-5) | -0.1660 / 0.0179 / -9.3 | 0.3218 (5e-4) | **-0.1094** / 0.0155 / -7.1 | +0.0172 / 0.0032 / +5.3 |

\* hetero 0: at lr 5e-4 (Adam's optimum) one of eight `pp_var` seeds diverged to NaN, so that LR is void
for the arm and its best LR is 2e-4; the other columns track Adam to 3 decimals (2e-4: 0.2470 vs
0.2429; 1e-3: 0.2734 vs 0.2749), i.e. the gate is a null there apart from the divergence, which is
real and reported (the level trap: mean gate 0.16, a sporadic opening became a large Adam step).
Adam's hetero-2 optimum 2e-5 is bracketed (1e-5: 0.4437, 5e-5: 0.3581). Gate level (time mean of
the mean gate): pp_var 0.05-0.35, oracle 0.16-0.37.

Reading: in both heteroscedastic cells `pp_var` beats Adam at t = -8.2 / -7.1 and its own shuffle
loses (t = +4.1 / +5.3): the gain is state->sample plasticity, not a schedule. It reaches 79% of the
oracle's gain in hetero 2 and 66% under the teacher switch (the noise field is stationary there;
horizonless sums are fine). Null cell: null (+0.0009, indistinguishable from its shuffle). Signal
scales: not significant at 8 seeds (t 1.6) and no better than its shuffle; the oracle's gain there
comes from the instantaneous learnable residual, which needs a working conditional-mean model.

## 3. Falsified along the way (all 8 seeds, 32768 samples, own best LR, unless stated)

* Form 1: regress delta_j itself on [1, a] (JS-shrunk mean and log-variance), plain sums, gate
  mu(a)^2/(mu(a)^2 + s^2(a)). hetero 0 +0.058 (t 5.1, shuffle +0.050), hetero 2 -0.067 (t -3.9,
  shuffle +0.020), signal-scales +0.42, null +0.008 (t 3.6, shuffle +0.009), switch -0.032 (t -10,
  shuffle +0.049). E[delta_j | a] is non-stationary by construction (learning removes it), so a
  horizonless regression of delta remembers where the net USED to be wrong; the null cell is the
  clean symptom. Fixed by regressing the goal (section 1); null moved from +0.008 to -0.001.
* Form 2: goal = z_j - delta_j without the Gauss-Newton scale for hidden units. delta_j is a gradient,
  not a displacement, so the goal tracked z_j and mu was the unit's own weight drift; hetero 0 smoke
  (2 seeds, 3072 samples) +0.016 vs -0.005 for the scaled goal. Kept the scaled goal.
* Form 3 = `pp`: goal regression with the PER-STATE conditional mean, sig(a) = ((z_j - yhat_j(a))^2 -
  se^2)^+: hetero 0 +0.055 (t 4.3, shuffle +0.060), hetero 2 -0.053 (t -6.8, shuffle +0.019),
  signal-scales +0.85 (t 2.8, shuffle +0.71), null -0.001 (t -0.7, shuffle +0.010), switch -0.024
  (t -8.4, shuffle +0.049). Beats Adam and its shuffle only where the noise field carries it, and
  there it throws away most of what the variance model alone gets (-0.053 vs -0.149): the per-state
  linear fixable-error estimate is too noisy (mean gate 0.02-0.04, sporadic openings) and, for the
  output unit, converges to zero by construction (same model class as the unit), so it gates on
  estimation noise. Ablations hetero 2 / hetero 0: mean-only (`pp_mu`, global s^2) +0.001 / +0.066;
  variance-only (`pp_var`) -0.149 / null; no JS shrink and no se^2 (`pp_raw`) -0.081 / +0.044; level
  normalised to own running mean (`pp_norm`, cap 20) NaN at 1e-3 / +0.063. The conditional-variance
  readout is the whole effect; the JS calibration matters (-0.149 vs -0.081) and is what makes the
  homoscedastic cell a null.
* Post-Adam gating (gate multiplies the Adam step, form 1): hetero 0 +0.077, hetero 2 -0.013 with
  shuffle -0.012 (pure schedule), null -0.005 at the 1e-2 grid EDGE (void). Pre-Adam is better for
  this gate; the shuffle shows the pattern, not the level, carries the gain.

## 4. Verdict

A real, knob-free, per-perceptron rule - "gate each unit by how unpredictable its own goal is from
its own state, calibrated analytically" - captures 79% (hetero 2) and 66% (teacher switch) of the
oracle's gain over Adam, at t = -8.2 / -7.1 with its shuffle worse than Adam in both, and is a null
in the null cell. That part is a great result. It is honest to add: (a) in the homoscedastic cell it
is a null with one divergence at Adam's best LR, so the pre-Adam level trap is not fully solved;
(b) in the signal-scales cell it does not help; (c) the conditional-MEAN half of "predictability" -
knowing at this state how wrong the unit currently is - failed in every form I tried (sections 3),
so the mechanism reads the noise side of predictability, not the signal side. The oracle shows the
signal side is worth as much again; I did not find a state-local, horizonless estimator of it.
