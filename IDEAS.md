# IDEAS

Parked directions with enough design worked out to be picked up cold. One heading per
idea. Keep the blocker explicit — most of these are parked because of a prerequisite,
not because they are speculative.

---

## Model-based credit assignment: replace GAE's fixed kernel with a measured one

**Family:** `cleanrl/encoder-value/` · **Status:** blocked on multi-step JEPA · **Filed:** 2026-07-27

### The problem with GAE

```
Â_t = Σ_k (γλ)^k δ_{t+k}
```

`(γλ)^k` is a *guess* at how much action `a_t` influenced what happened at `t+k`. It is
fixed, geometric, and identical for every state and every action. In HalfCheetah an
action at the apex of a gait cycle has a completely different influence horizon than one
taken mid-stance, and GAE weights them the same. Every reward that lands within the
kernel gets credited to `a_t` whether or not `a_t` had anything to do with it — that
mis-attributed mass is a large part of the policy gradient's variance.

Tuning λ does not fix this. It only rescales one number that is wrong per-state.

### The move

The JEPA predictor is a differentiable action-conditioned dynamics model:
`ê_{t+1} = T(e_t, a_t)`, with the action entering through AdaLN-zero conditioning. Roll
it forward and the influence of `a_t` on the state `k` steps later is **directly
measurable** by backprop:

```
J_k = ∂ê_{t+k} / ∂a_t              (act_dim × emb_dim Jacobian, k steps of rollout)
```

Replace the assumed geometric kernel with the measured one:

```
c_k(s_t, a_t) = ‖ w_r^T J_k ‖   normalized over k        # reward-relevant influence
Â_t = Σ_k γ^k · c_k(s_t, a_t) · δ_{t+k}
```

Project the Jacobian onto `w_r` rather than taking a raw norm — we want influence on
*value*, not influence on arbitrary latent coordinates, and `w_r` is exactly the map from
occupancy to reward. `c_k` is state- and action-dependent, which `(γλ)^k` structurally
cannot be. This is "learning GAE" in the strong sense: the credit kernel becomes a
measured property of the dynamics instead of a hyperparameter.

### Why the encoder-value family is the right host

`w_r` and the vector residual `E_t = Λ_t − ψ_0(s_t)` already exist in these files, and
`w_r` is precisely the linear functional that turns latent motion into value. In a scalar
critic there is no `w_r`, so there is nothing to project the Jacobian onto — the idea does
not port to a plain PPO baseline without inventing that map first.

### Blocker

The predictor is trained **one step ahead on 4-length sequences** (`seq_len=4`,
`history_size=3 + num_preds=1`, per the `../le-wm` reference). Autoregressive rollout
compounds error immediately, so `J_k` is meaningless past `k≈1-3` and the whole point is
the shape of `c_k` over `k`.

**Prerequisite: train the predictor autoregressively at k > 1.** That is a real change,
not a config bump — feed predictions back as inputs during training, which changes what
the encoder is pressured to represent. Note the reference gets multi-step behaviour by
autoregressive rollout *at eval only* (`le-wm/jepa.py:61`), never in training, so there is
no reference implementation to copy for this.

Do the multi-step predictor first and evaluate it on its own (does `pred_loss` at k=4
stay bounded? does `emb_effective_rank` hold up? does v2's return curve move at all?),
then build the credit kernel on top.

### Honest risks

- **Bias.** Reweighting δ's breaks GAE's bias/variance contract. GAE at λ<1 is already
  biased, so the question is whether this bias is better-directed, not whether it exists.
  Needs an anneal-to-`(γλ)^k` fallback so a failure degrades to standard GAE.
- **Deterministic dynamics.** HalfCheetah is deterministic given the action, so `J_k` is a
  clean signal here. In a stochastic env the Jacobian of a *mean* prediction understates
  influence. Do not assume this transfers.
- **Cost.** `k` backward passes through the predictor per rollout step. The predictor is
  79k params on a 4-token sequence, so this is probably affordable, but it is the first
  thing in this family that puts the JEPA in the inner loop rather than once per
  iteration. Measure before committing.
- **The `c_k` normalization is a free parameter** and a bad choice silently reduces to
  uniform credit. Log the realized kernel shape against `(γλ)^k` every iteration.

### Related, already tried

- **A** (orthogonal occupancy surprise `‖E^⊥‖` as an exploration bonus) — proposed, not
  yet built.
- **B** (per-dimension λ from measured mixing times) — built, see
  `cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v2.py`, in flight. B is the cheap
  version of the same intuition: it makes the credit kernel per-*coordinate* instead of
  per-*state*.

### Why B is not a cheap substitute for C — the two horizons are different objects

B sets `λ_d` from `τ_d = Σ_k γ^k ρ_d(k)`, the discounted integrated autocorrelation of
`φ_d`. Measured on a live run, `τ_obs` starts near 2 under a random policy and climbs past
9 as a gait forms; `τ_act ≈ 1`, so `λ_act ≈ 0.02`. The estimator is doing exactly what it
was designed to do — and that is the problem.

**`τ_d` is a state-predictability horizon. The credit kernel needs an action-influence
horizon.** `τ_d` answers "how far ahead does `s_t` tell me about `φ_{t+k}`?" The advantage
needs "how much did `a_t` *cause* `φ_{t+k}`?" These coincide only if actions and states
have the same temporal reach, which in a gait they emphatically do not: a coordinate can
decorrelate fast (oscillatory, quickly forgetting its initial condition) while an action
perturbation at `t` still shifts the whole downstream trajectory. Shortening `λ_d` on a
fast-mixing coordinate therefore discards *causal* credit that was never redundant.

Concretely for HalfCheetah's `r = x_vel − 0.1‖a‖²`: `w_r` loads the control cost onto the
`a⊙a` block (`λ≈0.02`, near-1-step — defensible, ctrl cost really is instantaneous) and
the velocity term onto the obs block (`λ≈0.42`, `τ≈9` vs the default `τ=16.8`). So B
roughly halves the credit horizon for the one reward component that has extended temporal
structure, which is the mechanism to expect a slowdown from.

This *strengthens* C rather than pricing it down. `∂ê_{t+k}/∂a_t` is literally the
action-influence kernel B is missing — it differentiates the prediction with respect to
the action, so it cannot be fooled by a coordinate that is unpredictable-but-influenced.
B underperforming for this reason is a positive result for C, not a negative one. What
would price C down is B being *neutral* — that would say kernel shape does not matter.

### C's prerequisite was built and it failed — C is now priced much lower

The Blocker section above set a gate: *"Do the multi-step predictor first and evaluate it on
its own... then build the credit kernel on top."* That was done.

`cleanrl/encoder-value/ppo_continuous_action_lejepa_sf_v3.py` trains the predictor
autoregressively (round `k` consumes round `k−1`'s own output). Run `v3_mstep_k4`, K=4,
HalfCheetah-v4, seed 1, with the SSL step budget held fixed at 64 optimizer steps so the
horizon was the only variable:

| | 500k | 1M | 1.5M |
|---|---|---|---|
| K=1 (uniform-λ vector routing, 628) | 2032 | 3977 | 5209 |
| K=4 autoregressive | 1961 | 3673 | 4961 |

**−5%, cancelled.** `emb_effective_rank` moved 18 → 29, so AR training genuinely changed the
representation — it was not a no-op that failed to engage. It changed it in a direction that
cost return.

The gate's own terms therefore say: do not build C on this predictor. `J_k = ∂ê_{t+k}/∂a_t`
is only meaningful if the rollout it differentiates is trustworthy at `k>1`, and the only
evidence available says training it that way makes the representation worse for value.

### And the deeper reason, which retires the latent for value work generally

The plan's cheap kill switch was: ridge of truncated-MC return onto `sg(e)` — the ceiling for
*any* critic reading the latent — against the same ridge on raw trunk features.

| | @1M | late |
|---|---|---|
| `gate/ev_latent_cap` (onto `sg(e)`) | 0.32 | 0.66 |
| `gate/ev_trunk_probe` (onto trunk features) | 0.61 | 0.88 |

~25 EV points, at every checkpoint. **SIGReg's whitening toward `N(0,I)` is active pressure
against the scale structure value is made of.** The latent is a lossy bottleneck *for value*,
whatever it is good for representationally.

This is the retroactive explanation for the family's most awkward measurement — `sf_noe_v1`,
which deletes the encoder, the predictor and SIGReg outright, beats every variant that keeps
them (9,953 ±109 @8M). It also explains the history-encoder result, which is monotone in the
wrong direction: `reward_probe_r2` = 0.963 / 0.965 / 0.946 and `resid_ac_lag10` = 0.134 /
0.206 / 0.373 for H = 1 / 3 / 16. More history makes the linear probe *worse*. Access is not
representation — the encoder is trained by prediction MSE + SIGReg, and neither of those asks
for linear decodability.

**Consequence for C.** C routes credit through `w_r · J_k`, i.e. through the latent, and is
therefore capped by the same 0.66. Any version worth building must either put the Jacobian in
observation space (which needs an obs-space action-conditioned model — see
`cleanrl/encoder-value/DESIGN_cla.md`, which builds exactly that for a different purpose and
would supply `∂ŝ_{t+k}/∂a_t` directly), or first show that a latent trained with a
value-relevant objective clears the trunk probe. Do not build C on the SIGReg latent as it
stands.

### Where the encoder still pays

Inside `φ`, and only there. `w_r · φ` is a 44-dim linear filter on the reward, and
`reward_probe_r2 ≈ 0.992` with a strongly autocorrelated residual (lag1 0.34, lag10 0.46) —
an autocorrelated residual compounds through the λ-return, which is why <1% of reward
variance is worth +6 to +17% return. But note this lever is **self-limiting**: `w_r·E = Â`
holds exactly when `w_r·φ = r`, so improving the probe destroys the gain. Anything built here
has to be non-linear in `E` to add signal at all.
