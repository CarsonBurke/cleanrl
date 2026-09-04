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

---

## TPO-MD's realized policy step decays 5.7x in the second half of training

**Family:** `cleanrl/ppo_continuous_action_tpomd_*` · **Status:** diagnosed, two fixes built,
neither tested (needs ≥5M steps) · **Filed:** 2026-08-18

### The measurement

`tpomd_alllayer_residual_td_v25` is the best run in the repo (11,015 @7.46M HalfCheetah,
8,069 @2M — best on-policy at matched steps). Its own logs show the policy update dying:

| | 0.2M | 1.9M | 3.7M | 5.6M | 7.4M |
|---|---|---|---|---|---|
| `max_epoch_approx_kl` (realized policy KL) | .0163 | .0328 | .0339 | .0175 | **.0058** |
| `tpo_kl_achieved` (discrete surrogate) | .0300 | .0300 | .0300 | .0300 | .0300 |
| `tpo_kl_base` (natural uncapped step) | 1.37 | 0.20 | 0.076 | 0.031 | 0.037 |
| `tpo_score_std_mean` | 1.39 | 1.70 | 1.68 | 1.38 | 1.65 |
| `tpo_sigma_global` | 0.67 | 1.95 | 4.01 | 5.45 | 7.15 |
| `episodic_return` | 81 | 7,688 | 10,026 | 10,927 | 11,313 |

`tpo_cap_engaged` is 1.0 at **every** iteration and `actor_epochs_completed` is 10/10
throughout — so nothing was being restrained; the target itself decayed onto the anchor.
Return growth falls from +30%/1.8M to +13%/3.7M across exactly that window.

Alternative causes ruled out from the same logs: `explained_variance` holds 0.956–0.981,
`value_loss` flat ≈1.0, `target_edge_mass` = 0.000 (HL-Gauss support never saturates),
`tape/h16_target_rms_over_eta` = 0.72 (deep auxiliary heads still delivering), and
`tape/encoder_relative_drift_before_capture` *falls* 0.216 → 0.027. The critic and the tape
are healthy. Only the policy step decayed.

### Two mechanisms

1. **Dimensional.** Scores are divided by an EMA of the *global* TD-residual RMS, which
   grows with the return scale (11x), while the ranked quantity — within-group score spread
   across K=8 candidates at one state — is flat. `u_scores` shrink ~9x for reasons unrelated
   to ranking quality, dragging `tpo_kl_base` onto the eps=0.03 cap.
2. **Geometric, and not fixable by rescaling.** As the Beta policy sharpens (entropy −0.66 →
   −10.7) the K candidates collapse together in *action* space, so a fixed reweighting of
   near-identical actions moves the fitted continuous policy less and less. This is why the
   discrete surrogate KL and the realized policy KL decoupled by ~6x.

### Built, not tested

- `ppo_continuous_action_tpomd_alllayer_spreadtemp_v28.py` — fix (1) open loop:
  `--tpo-sigma-mode group_spread`. **Verified to engage:** at 920k it drove `sigma_global`
  to 0.461 (spread) instead of 1.088 (td_rms), giving `tpo_kl_base` = 0.768 vs v25's 0.164,
  a 4.4x larger reserve above the cap. Return was within noise of v25 (+5% at matched steps).
- `ppo_continuous_action_tpomd_realizedkl_ctrl_v29.py` — fix (1)+(2) closed loop: multiply
  scores by `u_gain`, adapted in log space to hold `max_epoch_approx_kl` at 0.033.
  **Verified to regulate:** `u_gain` settled at 0.236 with realized KL 0.029–0.037 and
  `cap_engaged` = 0. Return ~7% behind v25 at 624k.

### Blocker — why both are untested

**Neither variant can separate from v25 before ~4M, by construction.** v25's realized KL over
0–1M is already 0.016–0.031, i.e. at v29's 0.033 target, and its `kl_base` still has 5x
headroom above the cap, so both fixes are near-no-ops early. The defect only appears once
`kl_base` reaches the cap at ~5.6M. Testing this hypothesis costs a ≥5M-step run per arm and
cannot be triaged at 1–2M; both arms were cancelled at 944k and 624k for showing no early
gain, which the design predicts.

If picked up: run v29 (it subsumes v28) against v25 to at least 6M before judging, and read
`debug/tpo_u_gain` against `losses/max_epoch_approx_kl` rather than return, since the return
signature is confined to the tail.

### The independent lever this analysis turned up

The slot prior inside the same target is separately wrong and is early-testable — see
`ppo_continuous_action_tpomd_mpo_anchor_v30.py`. Candidates are sampled from `pi_old`, so
the sampling already supplies the `pi_old` factor of `q ∝ pi_old·exp(u/eta)`; v25 also
weights the slots by `log_softmax(log pi_old(a_i))`, making the effective prior `pi_old^2`.
That discounts exactly the tail candidates whose 1-step reward the MuJoCo probe measures
exactly. The exact MPO/AWR E-step is a uniform slot prior.

---

## TPO-MD's improvement operator is at a local optimum in all four measured dimensions

**Family:** `cleanrl/ppo_continuous_action_tpomd_*` · **Status:** direction closed; one arm
(v32) running to 8M · **Filed:** 2026-08-18

`tpomd_alllayer_residual_td_v25` (11,015 @7.46M HalfCheetah) scores K=8 MuJoCo-probed
candidates by `score_i = r_i + gamma*V(s'_i)`, centres them per state, divides by a global
scale, and fits `q = softmax(beta*log pi_old + u/eta)` under a KL trust region. Four
independent knobs on that operator were built and measured against v25, seed 1, HalfCheetah:

| lever | file | engaged? | vs v25 |
|---|---|---|---|
| temperature reference | `..._spreadtemp_v28.py` | yes, `kl_base` 0.164 → 0.768 | no early effect; late-only by design |
| step-size regulator | `..._realizedkl_ctrl_v29.py` | yes, realized KL pinned 0.033 | **−21%** (2,366 vs 3,003 @630k) |
| slot prior exponent | `..._anchor_beta_v31.py` | yes, anchor ESS 8.0 → 2.51 | β=1 is the argmax |
| score quality | `..._depth2probe_v32.py` | yes, `score_std` +18.5% | **±0%** (@500k/750k/1M) |

### The slot-prior sweep is the informative one

Realized policy KL was pinned at 0.033 by v29's controller so step SIZE was matched and only
DIRECTION varied. `anchor = log_softmax(beta * log pi_old)`:

| beta | 0 (exact MPO E-step) | 0.5 | 1 (v25) | 2 |
|---|---|---|---|---|
| return @~630k | 552 | 1,460 | **3,003** | −65 |
| anchor ESS (of 8) | 8.0 | 7.30 | — | 2.51 |

The MPO/AWR derivation says the prior should be uniform — candidates are drawn from `pi_old`,
so the sampling measure already supplies the `pi_old` factor of `q ∝ pi_old*exp(u/eta)`, and
v25 applies it twice. **The derivation is right and the estimator is unusable.** With K=8
samples from a 6-dim Beta spanning ~1.7 nats of log-prob, `softmax(u/eta)` alone lets one
deep-tail sample own the target; its exact 1-step probe reward is a poor proxy for its
long-run value. The `pi_old` factor is variance control, not an artefact. Bias is cheap here;
variance is not.

`beta=2` fails for a *separate* structural reason worth remembering: a sharp anchor is
mode-seeking (target → highest-`pi_old` candidate → policy sharpens → anchor sharpens), and
realized KL ran to 0.262 with `u_gain` pinned at its floor. **The gain controller scales the
advantage tilt and has no authority over the anchor, so no gain setting can stabilise β>1.**

### Why v32 does not rescue it

If the operator is variance-limited, the fix is better scores, not a more aggressive target.
v32 gives each candidate a second EXACT physics step under the policy mean:
`score_i = r_i + gamma*(1-t_i)*[r'_i + gamma*(1-t'_i)*V(s''_i)]`. This adds a second exactly
measured reward difference, cuts the critic's weight from `gamma` to `gamma^2`, and doubles
the state separation at which V is read. It worked as designed — `score_std` 1.014 → 1.202,
`kl_base` 0.451 → 0.530 — at 2.4x the wall cost (`probe_sps_overhead` 7.8 → 20.6) and 2x the
privileged simulator access, **and returned nothing.**

### Consequence

The candidate ranking was never critic-limited, and the operator is not the binding
constraint. v25's early curve is near-invariant to every operator change tried, which locates
the early bottleneck in critic fit / representation formation (the tape side — the one change
that ever delivered, ~+12% when introduced) and the late bottleneck in the step decay
documented in the previous section. Do not spend further runs on the target distribution,
its temperature, its prior, or its scores.

