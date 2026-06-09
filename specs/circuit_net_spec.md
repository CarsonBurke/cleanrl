# CircuitNet — Spec v1.1 (implementation-ready, post 2 expert rounds)

**Working name:** CircuitNet — a giant pool of small expert modules; per input a learned router
activates a tiny sparse subset and composes them over recurrent steps into an input-specific
**circuit**. Plain backprop; compute independent of pool size (no O(N²)). Successor to DSRG.

**Provenance:** v0 → 4-expert panel (sparse-MoE, comp-neuro, deep-RL, adversarial) → v1 → 2nd pass
(adversarial coherence + implementation-readiness) → **v1.1, buildable**. The dominant lesson across
both rounds: **the trunk was never the binding constraint — the actor path was throttled.** So v1.1 is
"lean + un-throttled + measured," with scaling and brain mechanisms deferred behind a hard go/no-go.

---

## 0. Goals + the hard constraint

User goals: (1) brain-inspired, **backprop-only**; (2) **giant** (modules scale freely); (3) learns to
**create circuits** (emergent reuse of perceptrons & groups); (4) **no O(N²)**; (5) beat the ~7000
HalfCheetah plateau. v1.1 constraint: **every circuit/brain claim is a logged, kill-switched metric or
it doesn't ship** — the narrative kept DSRG alive 18 versions past falsification.

### 0.1 DSRG post-mortem (do not repeat)
Contractive routing (convex-avg ⇒ smooth maps) · illusory width (1024×8×rank4 ⇒ eff. width ≈8) · inert
routing (gates static by ~200k) · throttled actor (shared sparse trunk + sparse readout; critic EV~0.95
but a *separate-tower dense MLP* scored ~8292 > DSRG 6931) · config throttles (target_kl 0.03, clip 0.25,
rankgauss).

---

## 1. Architecture (v1.1)

### 1.1 Separate actor & critic trunks — NON-NEGOTIABLE
No shared backbone. Each gets its own encoder, expert pool, router, dense readout. The HL-Gauss critic's
large CE gradient on a shared trunk specializes experts toward value features → the weak-actor failure.
Critic may be bigger (more experts / higher T); actor stays small with a **dense** readout.

### 1.2 Trunk: recurrent sparse MoE over a residual workspace
```
obs ──encoder──► w ∈ R^{D}                        (D=256; K=1 workspace in v1.1, K=4 = backlog)

for t in 1..T:                                    (T=2; expert pool WEIGHT-TIED across steps)
    ŵ   = LayerNorm(w) + phase_emb[t]             (pre-LN + per-step learned PHASE code, [T,D])
    g   = Router(ŵ)                               (Linear D→M; top-2, softmax over the 2 kept logits)
    Δ   = Σ_{e∈top2(g)} g_e · Expert_e(ŵ)         (selected experts ADD, weighted by gate)
    w   = w + γ · Δ                               (LayerScale γ, per-channel, init 0.1)

actor:  LayerNorm(w_T) ─► Linear ─► Beta(α,β)             (DENSE readout, no routing)
critic: LayerNorm(w_T) ─► Linear ─► HL-Gauss symlog bins  (DENSE readout, no routing)
```
- **Experts:** M SwiGLU MLPs `D→h→D` (h=2D=512), **weight-tied across T** = the reuse substrate.
- **"Start shallow, grow circuits" — SINGLE attenuation (v1.1 fix):** the *only* near-zero init is each
  expert's **output projection `W_out` (init 1e-3·randn)** so Δ≈0 at t=0 and the net starts as
  encoder→readout (a working shallow net). **LayerScale γ init 0.1** (a healthy residual-branch scale),
  *not* 1e-4. The v1 draft stacked γ=1e-4 *and* near-zero W_out ⇒ ~1e-8 double attenuation ⇒ the task
  gradient into the router was ~zero ⇒ a self-engineered inertness path (the exact thing §2 must catch).
  One attenuation (W_out), self-resolving as experts learn. Router liveness early comes from LB/z-loss +
  warmup noise (logit-level, Δ-magnitude-independent), so the router stays alive while Δ ramps.
- **Per-step phase embedding** `phase_emb[T,D]` (zero-init): tied router weights but a distinct per-step
  code ⇒ the T-step selection is an *ordered* circuit, not the same set re-fired. **This is the only
  mechanism that genuinely prevents the weight-tied iterated-map fixed-point** (LayerScale and small-T
  only bound exposure; delta logging only watches). Keep and make it load-bearing.
- **LayerScale, not a highway gate:** saturating gates were the v6/v7 failure already removed here.

### 1.3 Routing: top-2 softmax + the anti-collapse package (first-class, from step 0)
- top-2 of M by logit; softmax over the **top-2 logits only**; gate weights multiply expert outputs ⇒
  router gradient through selected gates (Switch/GShard). **No STE.**
- **Implementation at M=64: dense-compute-all-M then mask to top-2** (gather/scatter is pointless at this
  size; switch to true sparse dispatch only at M≥1024, backlog). The full-M `softmax(logits)` (for LB) is
  free from the same logits.
- **Load-balance**, batch-level (NOT per-sample), summed over T then /T: `M·Σ_e f_e·P_e`,
  `P_e=softmax(logits).mean(0)`, `f_e=onehot_top2.mean(0).detach()`, **coef 0.01**. Batch-level is the
  key: each input specializes (sparse), the batch keeps all experts alive.
- **Router z-loss** `mean(logsumexp(logits)²)`, **coef 1e-3** (stops logit blow-up → saturation/freeze).
- **Warmup logit noise** `N(0, σ²)` on logits, **σ: 1.0 → 0 linearly by 75k env-steps**.
- LB uses full-M `P_e`, so non-selected experts still get ranking gradient + occasional noisy selection —
  collapse loop broken without STE.

### 1.4 No O(N²)
Only mixing = the K=1 workspace. Per step: router `O(D·M)` + `k=2` experts `O(k·D·h)`. **M decoupled from
FLOPs by top-k.** No module↔module edges. Giant pool = *parameters*, not compute.

### 1.5 Exploration — iid Beta in v1.1; gSDE deferred (v1 fix)
gSDE is defined for a Gaussian mean perturbation; "modulate Beta concentration with state-dependent
noise" is **not a log-prob-consistent mechanism** and would force a Beta→Gaussian head swap that
confounds the trunk A/B. **v1.1 ships iid Beta** (the existing mechanism). gSDE becomes a separate
`_gsde` variant with a squashed-Gaussian head (backlog) where it is well-defined.

### 1.6 Plasticity — ReDo dormant-expert reset
Every **250k** env-steps: dormancy score `s_e = E[|hidden_e|]/mean_e E[|hidden_e|]` over a probe batch;
any expert with `s_e < τ=0.025` gets `W_in/W_gate` re-init, `W_out` → 1e-3·randn, and its **per-expert
rows of Adam `exp_avg`/`exp_avg_sq` zeroed** (leave `step`). Constant LR otherwise.

### 1.7 Config (start small — sample the slope, don't assume the giant)
`D=256, M=64, h=512 (SwiGLU), k=2, T=2, K=1`. M=4096 rejected: most of the pool would see <1 grad/epoch
⇒ dead experts, and a giant pool buys nothing until the structure beats the dense control. **Run an
M=256 spot-check inside v1.1** so the gate samples the *slope* dReturn/dM (a positive M=64→256 slope is
far stronger evidence for the giant hypothesis than one M=64 point). torch.compile the (fixed-shape
dense) MoE cell — no recompiles; T=2 ⇒ no grad-checkpoint needed.

### 1.8 Keep validated pieces + un-throttle config (labeled axis, for attribution)
Beta actor, HL-Gauss symlog critic, GAE, obs/reward-norm, 16 envs. Config: `target_kl=None`, clip
0.2/0.28, **global** grad-clip 0.5, **no rankgauss**, pre-LN.

---

## 2. Falsifiability — instrumentation + kill-switches (pinned)

**Guarded metrics (kill-switch attached):**
- **`routing_mi`** (conditionality) `:= H(mean_b q_b) − mean_b H(q_b)` where `q_b∈R^M` is the per-sample
  top-2 gate-mass distribution. Range [0, log M]; =0 ⇔ input-independent routing. *(This entropy-gap MI
  lower bound replaces the unshippable continuous-obs `I(selection;obs)`.)* **KILL if < 0.05·log M at
  200k** (it's DSRG inertness again).
- **`eff_rank`** = effective rank (e.g. exp(entropy of normalized singular values)) of the stacked
  expert-output matrix over a probe batch — the direct **illusory-width / collapse** detector (DSRG's
  named cause; was missing in v1). **WARN if eff_rank ≪ D** trending down.
- **`dead_expert_frac`** (using the §1.6 τ=0.025). **Shrink M / escalate ReDo if > 0.5 at 200k.**
- **`return` vs the matched-FLOP dense control** at 1M. **Stop & fix readout if < control.** Early proxy:
  **`actor_grad_norm / critic_grad_norm` ratio at 200k** (early actor-throttle warning).
- **`delta_ratio[t]` = ‖γ·Δ_t‖ / ‖w_T‖** per step (a *ratio*, not raw Δ — distinguishes "Δ small because
  γ/W_out small" from "experts agree"). If →0, recurrence isn't working.

**Diagnostic-only (logged, no kill-switch):** per-step `router_entropy`, `expert_util_entropy`,
`route_decision_var` (reuse v15 `circuit_metrics.conditionality`), `mean(γ)`.

## 3. Go/no-go control battery — run CONCURRENTLY with v1.1 (same PR)

1. **Matched-FLOP dense residual MLP** (same D, T-equiv depth, same heads, same active FLOPs, **no
   routing**). 2. **Frozen-random-router** (fixed random top-2, no router grad). 3. **T=1** (no recurrence).

**ONE unambiguous decision rule** (v1.1 fix — the spec previously stated two bars):
> **PASS** iff, over **3 seeds**, mean final-100k return of CircuitNet ≥ the **matched-FLOP dense control
> by ≥ 5%** *AND* `routing_mi ≥ 0.05·log M` at 200k *AND* `return(T=2) ≥ return(T=1) + 5%`.
>
> The ~8292 separate-tower dense MLP is **context, not the gate** (it had more compute than the matched
> control). Do not gate on 8292.
>
> **Tie-breaker (pre-registered, not ad-hoc):** if CircuitNet ties/loses the dense control BUT
> `routing_mi` and `delta_ratio` are healthy (machinery works, score capped), run **one K=4 blackboard
> confirmation** before declaring the representation axis dead — the gate is otherwise blind to the
> single-workspace bottleneck (same D=256 as the dense control), the panel's #1 expressivity risk.

If PASS → backlog. Else → §5 pivot. **Frozen-router-ties-learned or T=1-ties-T=2 ⇒ the routing/circuit
machinery is decorative regardless of absolute score.**

## 4. Backlog — one at a time, only after the gate PASSES, each measured, **re-running §3 controls at the new scale**
1. **Scale M** (64→256→1024) — the giant goal, earned. Re-run controls at each M (anti-collapse is
   load-bearing differently at scale; an M=64 pass does not transfer).
2. **K=4-slot blackboard** + attention write-competition + one broadcast/readout slot (parallel circuits,
   no N²; relieves the single-workspace bottleneck).
3. **ReMoE-style ReLU/threshold routing** (per-expert sigmoid gate + L1; allows few-or-zero experts;
   fully differentiable, anti-inertness). #1 routing alternative.
4. **Router liveness escalation** (if inert): critic-loss router warmup + 1-step latent self-prediction
   aux (always-on dense gradient so the router never starves on clipped PG).
5. **Neuromodulation:** critic value/uncertainty (stop-grad) → router temperature + low-dim FiLM on
   expert gains. Gate routing/gain only, never the action dist.
6. **gSDE** state-dependent exploration (separate squashed-Gaussian `_gsde` variant).
7. **Co-activation / temporal-consistency** routing priors (stable assemblies); divisive norm in experts;
   hierarchical group routing (only past M≈8k).

## 5. Honest dissent — is representation even the bottleneck?
Two experts argued the real 7k→11k gap is **off-policy data efficiency**, not representation (matched-FLOP
MLP tied DSRG; SAC hits ~11k with a *tiny* net). v1.1 respects this: the **§3 gate is exactly that test**
— if a matched-FLOP dense MLP ties CircuitNet, the representation axis is confirmed dead and we pivot
*fast*. If CircuitNet beats it, the likely win is "**sparse updates reduce gradient interference /
plasticity loss**" — a real on-policy lever we'll have *measured*. The named fallback (highest-EV pivot if
the gate fails): **replay buffer + Q-critic data reuse on the PPO core** (the mechanism SAC wins with).

---

## Appendix A — Implementation defaults (pinned; build against these)

| Item | Value |
|------|-------|
| D / M / h / k / T / K | 256 / 64 / 512 (SwiGLU) / 2 / 2 / 1 |
| Expert MoE forward | dense-compute-all-M then mask to top-2 (sparse dispatch = M≥1024 backlog) |
| Gate | softmax over top-2 logits only; gate·output; **no STE** |
| LayerScale γ | `Parameter(full([D], 0.1))`, applied `w = w + γ*Δ` |
| Expert W_out init | `1e-3 * randn` (the single shallow-start attenuation) |
| Phase emb | `Parameter(zeros([T, D]))`, added to `LayerNorm(w)` pre-router |
| LB loss | `M·Σ (softmax(logits).mean(0)) · (onehot_top2.mean(0).detach())`, /T, **coef 0.01** |
| z-loss | `mean(logsumexp(logits)²)`, /T, **coef 1e-3** |
| Warmup noise | logit `+= σ·randn`, **σ = max(0, 1−step/75000)** |
| Trunks | `share_backbone=False` only; new `CircuitNetTrunk`; **dense** readouts; drop PathwayReadout |
| Route backward | keep v15 triple-backward; `route_parameters()` = **both** trunks' router weights/biases (deduped); route_loss = LB+z, clipped on router params only |
| ReDo | every 250k; score on expert hidden act; τ=0.025; reset `W_in/W_gate`, `W_out`→1e-3·randn; zero per-expert Adam `exp_avg`/`exp_avg_sq` rows, keep `step` |
| routing_mi | `H(mean_b q_b) − mean_b H(q_b)`, q_b = per-sample top-2 gate masses; kill < 0.05·log M @200k |
| Exploration | iid Beta (gSDE deferred) |
| Config | target_kl=None, clip 0.2/0.28, global grad-clip 0.5, no rankgauss, pre-LN, torch.compile cell |

**v15 reference points** (`ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dsrg_v15.py`):
triple-backward clip `:1196-1226` · trunk route_loss stash `:690` · separate-trunk scaffold `:756-761` ·
param methods `:847-869` · compile-the-cell `:981-985` · Beta actor `:805-810` · reusable
`circuit_metrics` `:717-747`.

---

*Status: v1.1 implementation-ready. Next: build the lean trunk + §2 instrumentation + §3 control battery
in one file, run the 4-way battery (CircuitNet / dense-control / frozen-router / T=1) on HalfCheetah.*
