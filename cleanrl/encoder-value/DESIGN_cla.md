# Counterfactual Latent Advantage (CLA)

Design record for `ppo_continuous_action_cla_v1.py`. Written before implementation so the
hypothesis is falsifiable rather than retrofitted, and amended once — see "the measurement
that took the asset away" — rather than quietly rewritten to match what shipped.

## Why the previous four attempts failed, in one sentence

The encoder is trained by prediction MSE + SIGReg, neither of which is connected to the
value or policy objective, so it is *optional* — and the measured consequence is that
deleting it outright (`sf_noe_v1`, 9,953 +/-109 @8M) beats every variant that keeps it.

| arm | 500k | 1M | 1.5M | verdict |
|---|---|---|---|---|
| 628 uniform-lambda vector routing | 2032 | 3977 | 5209 | **10,071 +/-67 @8M — the target** |
| noe (encoder deleted) | 1907 | 3822 | 5093 | 9,953 @8M |
| lejepa_sf_v2 | 1738 | 3562 | 4915 | baseline |
| 633 AR K=4 | 1961 | 3673 | 4961 | -5%, cancelled |
| 635 histformer H=16 | 1877 | 3720 | — | -6.5%, cancelled |
| 627 per-dim lambda | 1559 | 3407 | 4444 | -15%, cancelled |
| 634 hist H=3 | — | — | — | mechanism falsified, cancelled |

Two further measurements constrain the design:

- **The vector-routing win is reward denoising.** `adv_vec_corr` 0.97->0.99 and
  `reward_probe_r2` ~0.992 with a strongly autocorrelated residual (lag1 0.34, lag10 0.46).
  The identity `w_r . E == A` holds exactly when `w_r . phi == r`, so *improving the probe
  removes the gain*. The current win fights its own improvement. Do not build on it.
- **Short credit horizons are lethal here.** The per-dim lambda arm cut tau from 16.8 to ~9
  and lost 15%. Any design that truncates the horizon starts from a large deficit.

## The asset — and the measurement that took it away

The original design put `psi_cf` in LeJEPA's latent, using the JEPA predictor as the
counterfactual model. Two things justified that: SIGReg pins the marginal to `N(0,I)`, so a
head reading it is not chasing a drifting representation; and the action-conditioned
predictor can evaluate actions that were never taken. Nothing else in the codebase can.

**The plan's own kill switch settled it against that version before a step was run.**

| predictor of truncated-MC return | @1M | late |
|---|---|---|
| `gate/ev_latent_cap` — ridge onto `sg(e)`; the CEILING for ANY latent critic | 0.32 | 0.66 |
| `gate/ev_trunk_probe` — the same ridge on raw trunk features | 0.61 | 0.88 |

The stated rule was: *"if (3) << (1), the latent is the bottleneck and this cannot win
however good psi gets."* It is, by ~25 EV points, at every checkpoint. SIGReg's whitening
toward `N(0,I)` is active pressure against the scale structure value is made of — which is
also the retroactive explanation for why deleting the encoder wins.

So the counterfactual next-state is valued by the **trunk**, and the model lives in
**observation space**. LeJEPA is not deleted; it survives exactly where it was measured to
pay, inside `phi`, where `w_r . phi` does reward denoising. That makes this run a test of
**counterfactual baselines**, not of LeJEPA — stated plainly rather than sold as one.

## What this replaces, and why it is not the previous lever again

Vector routing is scalar GAE with `r_t` swapped for `w_r.phi_t`: a STRICTLY LINEAR functional
of `E`, self-limiting because improving the probe destroys it. CLA is the first change in
this family that alters the advantage **non-linearly** — `psi_cf` is an expectation over a
learned model, not a projection of anything already in the buffer.

## The move

GAE's baseline `V(s)` is a *learned estimate* of `E_{a~pi}[Q(s,a)]`. With an
action-conditioned model that expectation can be **computed** instead of learned.

```
s_hat'(a) = s + D(s, a)                                 # ObsDynamics; predicts the DELTA
a^(m) ~ pi(.|s),  m = 1..M                              # counterfactual actions
psi_cf(s)  = (1/M) sum_m [ phi(s, a^m) + gamma * psi(s_hat'(a^m)) ]

psi_t     <- (1-beta)*psi(s_t)     + beta*psi_cf(s_t)   # BOTH ends of the residual
psi_{t+1} <- (1-beta)*psi(s_{t+1}) + beta*psi_cf(s_{t+1})
delta_t    = phi_t + gamma*boot*psi_{t+1} - psi_t
E_t        = delta_t + gamma*lambda*cont*E_{t+1}        # full lambda recursion, UNCHANGED
```

Why each piece:

- **Full lambda recursion retained.** The per-dim lambda result says horizon truncation is
  expensive. This changes the baseline, not the horizon.
- **Marginalized over pi, so unbiased.** `psi_cf` has no dependence on the action actually
  taken, so it is a valid state-value baseline and the policy gradient stays unbiased. If
  it depended on `a_t` it would be an action-dependent baseline and would bias the gradient.
  `counterfactual_psi` is asserted to have no free reference to the `actions` buffer.
- **Model used only at k=1.** Measured: the predictor is accurate one step ahead and gains
  nothing from AR training past that. Use it precisely where it is trustworthy.
- **Blended at BOTH ends.** GAE telescopes only if the same value function appears at `t` and
  `t+1`. Replacing `psi(s_t)` alone would break the lambda-return into a biased sum of
  mismatched baselines. For the same reason `values[]` — recorded during the rollout from the
  *unblended* psi — is rebuilt as `psi_cur @ w_r` after the blend, or the scalar residual
  reads `r_t + gamma*V_cf(s_{t+1}) - V_learned(s_t)` and `returns` silently stops telescoping.
- **Delta target + zero-init output.** An untrained model is then the *exact identity*, so a
  cold counterfactual degrades to a stale-by-one-step value rather than to noise.
- **Held-out, per-dimension R^2 gate.** Pooled R^2 lets one loud dimension (root x-velocity)
  carry the score while the joint angles `phi` needs stay unlearned; an R^2 against `s'`
  instead of `Delta` reads 0.99 for the identity predictor and would open the gate on a model
  that had learned nothing. In-sample R^2 would open it on memorisation.

## Hypotheses

- **H1 (the point).** A computed baseline beats a learned one where the critic is wrong.
  Measure: `cla/ev_sf` vs `cla/ev_sf_learned` — the SAME truncated-MC target predicted by the
  expanded and by the learned value. `ev_sf > ev_sf_learned` is direct evidence the model
  knows something the critic does not; `<` means it is injecting bias and `cla_beta` is too
  high. `cla/baseline_gap` is the weaker companion: ~0 from the start means a null experiment.
- **H2 (variance).** Advantage variance falls at matched return. Measure `advB/adv_*_std`
  and `losses/explained_variance` against 628 at matched steps.
- **H3 (control).** `--cla-m 0` or `--cla-beta 0` reverts to the parent, bitwise at the blend.
  Target to beat: `sf_vlam_v2_scalarlam` at **10,071 +/-67 @8M** on HalfCheetah-v4.

## Known risks, stated up front

- **Model error enters the baseline.** A biased `s_hat'` biases `psi_cf`. Guarded by the R^2
  gate and by `cla_beta`; `beta=0` recovers the parent exactly, so a failure degrades
  gracefully rather than catastrophically.
- **Finite M injects noise into the baseline, and GAE does NOT damp it.** In the expansion of
  `A_t`, the CURRENT state's baseline carries coefficient exactly `-1`; only the FUTURE values
  are attenuated by `(1-lambda)`. So the M-sample error reaches the advantage essentially
  undamped. It is unbiased — the `a^m` are drawn independently of the executed `a_t` — but not
  free. `cla/cf_se_frac` measures it against the advantage spread precisely so a null result
  can be attributed to *"the model knows nothing"* rather than *"the signal was buried under
  sampling noise"*. If that fraction is large, **M** is the knob, not `cla_beta`.
  (The critic's own target is safe by a different route: `psi_cf(s_t)` cancels out of
  `sf_target = E_t + psi_cur`, leaving the noise attenuated by `gamma*(1-lambda)` ~ 0.05.)
- **The gate may never open.** If held-out per-dim R^2 on `Delta s` never reaches 0.9 the run
  is just the parent with extra cost. `cla/dyn_r2` and `cla/active` are logged from iteration
  1; check by ~300k and lower the threshold rather than burning 8M steps on a closed gate.
- **Cost.** 2 * M full trunk+critic forwards over 32,768 states per iteration, forward-only
  and chunked. Expect a ~20-30% SPS hit at M=4. Measure before committing to 8M.
- **Beta actor.** Counterfactual samples go through the SAME native-z -> `to_action` path as
  the executed action, so `phi(s,a)` and `D(s,a)` see the identical action representation.

## Verification

`scratchpad/verify_cla.py`, all passing: exact identity at init; `Delta`-vs-`s'` and
per-dim-vs-pooled R^2 both demonstrated to separate a real model from a null one;
`counterfactual_psi` exactly equals `phi(s,a) + gamma*psi(s_hat')` on a controlled stand-in
agent; chunk size does not change the result; `psi_raw` applied to the critic output and to
it alone; AST check that the executed action cannot reach the baseline; MC error falls as
`1/sqrt(M)`; the SE estimator reads exactly 0 under a point-mass policy; gate closed for
`m=0`/`beta=0`/`r2 < threshold`; blend is a bitwise no-op at `beta=0`; `values[]` rebuilt
after the blend; boundary masking and the disjoint train/val split.
