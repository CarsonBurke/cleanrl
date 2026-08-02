# IDBD Beta v2 — paper-gap ablations

## KEY FINDING (2026-07-15): IDBD's meta-learning is inert in every deep-RL run

TensorBoard `idbd/*` diagnostics on the 8M runs (sepclip, rankgauss, theta05, theta2):
α_mean stays at init 0.05 for all 8M steps (`frac_at_max`=0, `meta_dot`≈−2e-7).
Even θ=2.0 (400× stream default) moves α only 0.05→0.0447 over 8M, and meta_dot is
*negative* at every θ (successive per-param gradients anticorrelate — noise/overshoot
regime, ~20-step h memory is swamped by minibatch noise in dense deep-net gradients,
unlike sparse-feature linear LMS). So:

- Every "IDBD" result here is actually **plain per-param SGD(0.05) + per-head grad
  clip 0.5** on a tiny-batch stream. That combination is the real discovery.
- The per-head clip acts as the implicit advantage normalizer (pins update magnitude);
  that's why noadvnorm works and noactorclip collapsed.
- The late plateau = constant effective step forever (fixed α, clip-pinned scale).
  NOTE: `anneal_lr` was a silent no-op for IDBD (only re-seeds β at init).
- TD7+IDBD loses because in i.i.d. replay IDBD ≡ momentum-free SGD replacing Adam
  (no RMS preconditioning), and replay sampling destroys the temporal gradient
  correlation IDBD's h-trace feeds on. Smaller batches don't fix that (b32 run died
  at 46k, at -43 return); unleashed meta (papertheta) hurts wherever tried.

Follow-ups queued: `sgdstream_v1` (explicit SGD control), `sgdstream_anneal_v1`
(working linear anneal → plateau test), `idbd_v4_rankwin` (rank→probit advantage
shaping over sliding 4096 window on the fresh delayed stream; rankgauss robustness
without 2048-step staleness).

### Follow-up results (2026-07-15, same day)

- `sgdstream_v1`: matched sepclip exactly at 500k (2001 vs 2009) → attribution
  confirmed, killed as redundant.
- `sgdstream_anneal_v1` (8M done): 5259 last-20 (±59) vs sepclip 5173 (±658).
  +20-30% faster mid-training (4460 vs 3712 @2M) but **same endpoint** → the
  ~5.5k plateau is NOT step-size policy; it's structural (suspects: Beta-policy
  entropy collapse — βconc grows to ~14 — GAE-32 horizon, 64×64 capacity).
- `idbd_autostep_v1` (killed @2M, 2212 vs sepclip 3712): the normalized
  per-tensor meta WORKS (finally acts) and its actions HURT — actor α
  self-collapsed 20× (0.05→0.0024, approx_kl→1e-4), critic α exploded to ~0.6.
  Verdict on the whole IDBD-in-PPO line: inert at paper scale → acts when
  Autostep-normalized → actively harmful when acting, because one-step gradient
  agreement is misaligned with control performance (actor-gradient
  anticorrelation is intrinsic to PG streams, not an overshoot signal). CLOSED.
- `idbd_v4_rankwin` (2M, running): 3823 @2M ≈ sepclip pace, but far ahead of
  rankgauss's early pace (2351 vs 651 @500k); rankgauss's win was late-slope —
  keep to 8M.
- Queued `sgd_rankwin_anneal_v1`: anneal × rank-window composition.
- Next track (user direction): learned translator → TD7 rank-weighted
  soft-greedification (`td7_rankgreedy_v1`, building).

### TD7 track results (2026-07-15/16)

- `td7_rankgreedy_v1` (killed @162k, 9676 vs base ~11-12k): k=8 rank-weighted
  candidate distillation replaced the DPG actor. Diagnostics: w_cand0=0.27,
  Q-gap grew to 0.45 — signal existed but 8-sample zeroth-order search loses
  to the analytic ∇ₐQ oracle in 6-D. Lesson: TD7's actor update is not the
  bottleneck; min-twins already handle overestimation.
- `td7_stocksig_hlgauss_v1` (killed @65k, 5885 vs stocksig 8086, −30%):
  isolated `--hl-gauss-critic` (511 bins, symlog ±20k, σ=2 bins). Structural
  diagnosis: symlog resolution ∝ 1/(1+Q); at Q≈9k one bin ≈ 400 raw and σ-blur
  ≈ ±800 raw, below the ΔQ the DPG gradient needs between neighboring actions.
  CE loss flat (state ranking fine) while actor gradient starves. Fair retest
  would need: categorical as firewalled aux head (repo invariant), or linear
  support on running-scale-normalized targets, or 2047 bins + σ0.75 +
  symexp(E[z]) decode.
- `td7_stocksig_hlg_narrow_v2b` (killed @160k, @50k/100k/150k = 2081/5792/7734
  vs wide 2691/6249/— and td7_v1 5144/9247/10658): right-sized symmetric support
  (symlog ±7.8244 ≈ raw ±2500, σ_ratio 0.75; 5× better resolution at the Q≤1600
  operating point) made it slightly WORSE, not better. Support geometry
  falsified as the cause. Revised diagnosis: HL-Gauss hurts TD7 through the
  control path itself — ∇ₐQ must flow through the softmax-categorical decode
  (blurrier than a linear head), and the CE loss decouples LAP priorities from
  value error. **HL-Gauss-in-the-critic line closed for TD7**; only remaining
  fair test is categorical as a firewalled aux head.
- `td7_stocksig_hlg_narrow51_v3` (killed @69k, −234 avg — never learned).
  Diagnostics show the actual mechanism (NOT gradient starvation): runaway
  overestimation with total discrimination collapse. q_min climbed to meet
  q_max ([378, 384] over the whole batch by 47k) while true policy return
  was ≈ −200; critic CE flat at 2.4 throughout; edge_mass ~1e-19 (support
  width not the issue). Mechanism: at Q≈380 one symlog bin ≈ 117 raw, so all
  real action-value differences are sub-bin and unrepresentable in the CE
  targets; the critic cannot encode "this action is better"; DPG's only
  representable ascent direction is +1 bin (+36%, multiplicative bins);
  bootstrap targets follow → positive-feedback inflation. 511-bin runs
  escaped (bins 10× finer keep discrimination representable) and pay only
  the steady ~35% quantization/decode tax. Lesson: categorical value
  critics under DPG fail CATASTROPHICALLY once bin width exceeds
  action-value gaps — it's a cliff, and it's an overestimation loop, not a
  slowdown. HL-Gauss line closed (aux-head test also cancelled by user).
- Noise-arm mechanism checks (from tfevents, 2026-07-16):
  - `td7_sdnoise_v1` σ is genuinely state-dependent, not an annealer:
    σ_min≈0.01, σ_max≈1.0 (100× spread), σ_mean≈0.13 stable 42k→294k;
    α settled ≈0.05. Exploration is reallocated across states under a
    fixed budget, as designed. @150k 10877 vs td7_v1 10658.
    **FINAL @1M: 17,122 (last-30) vs td7_v1 16,043 — +6.7%, new best.
    Led or matched baseline at every checkpoint.**
  - `td7_betanoise_v1` deficit is ENTIRELY the entropy transient: alpha_ent
    1.0→0.04 while conc grew 1.06→18 (done ~35k); after convergence it is
    ahead of both at matched steps (10400 @98k vs td7_v1 9247, sdnoise
    9726) despite spotting ~15k steps. frac_saturated≈0.26 — quarter of
    action dims near bounds, the regime where bounded Beta support beats
    Gaussian+clamp. Entropy autotune pinned exactly at target (−9.5 vs
    −9.46).
  - Registered prediction for `td7_betanoise_v2` (warm-start:
    conc_init=49.5 — symmetric Beta exactly at the −9.46 budget, std
    ≈0.0995; my first spec of 18 was wrong, that's the mean-conc of v1's
    ASYMMETRIC equilibrium and a symmetric 18 sits above budget at std
    0.164 — alpha_ent_init=0.1): ≥5000 @50k and ≥ sdnoise within noise
    @100k;
    otherwise the Beta parameterization itself (implicit-reparam gradient
    quality) is inferior to Gaussian and the arm closes in favor of
    sdnoise.
  - **v2 RESULT (killed @179k): prediction failed on both bars** (4956
    @50k; 8515 @100k vs sdnoise 9726) AND the pre-registered inference was
    wrong — v1 (cold start) now LEADS the whole board at matched steps
    (11752 @175k vs sdnoise 11419, td7_v1 11331), so Beta isn't inferior;
    the warm start itself hurt (−1000+ at every checkpoint vs v1).
    Corrected model: v1's 25k-35k near-uniform phase was not wasted — it
    was extra exploration that filled the LAP buffer with diverse actions
    after seed; killing it costs more than the delayed convergence did.
    Cold-start Beta = an emergent wide→budget exploration schedule. Keep
    cold starts for stochastic-policy arms on this env.
- Soft-Bellman arms launched (user-directed; 2×2 with hard-backup
  siblings): `td7_softgauss_v1` (squashed Gaussian, SAC backup:
  a'~current π, minQ.clamp − α·logπ, logπ target-clamped ±50) and
  `td7_softbeta_v1` (same with v215 Beta head, ln2 Jacobian bookkeeping,
  **cold start** --conc-init 1.7 per the v2 lesson, matching softgauss's
  σ≈1 init). Prior evidence against: full-SAC swap lost 15280 vs 16454,
  but confounded. Registered bars @100k: softgauss ≥ sdnoise (9726),
  softbeta ≥ betanoise_v1 (9971), within noise. Failure signature to
  check before blaming the mechanism: target_logpi_mean/max inflation
  (entropy term corrupting the backup) vs clean underperformance.
  - **softgauss RESULT (killed @104k): bar failed — 8334 @100k vs sdnoise
    9726 (also below td7_v1 9247; 4777 vs 5404 @50k). Diagnostics rule
    out numerical corruption**: target_logpi mean 4.9 / max 15 (clamp ±50
    never touched), α converged 0.13, entropy_proxy −5.06 ≈ target −5.30;
    steady-state soft-value shift only ≈ −65 on Q≈900. The soft backup
    ITSELF is a clean ~10-14% regression on TD7/HalfCheetah — the earlier
    full-SAC-swap loss was not (only) confounds. Candidate mechanisms:
    single stochastic a′ (no clipped smoothing) raises target variance
    that TD7's min-twin/clamp machinery was tuned to suppress; α settling
    2.6× higher than sdnoise's (0.13 vs 0.05) adds actor-update noise.
    Exploration-value propagation isn't worth those costs on a
    dense-reward env. `td7_softbeta_v1` (auto-launches into the slot) is
    the remaining check: Beta's bounded logpi and boundary behavior could
    shift the trade — bar stays ≥ betanoise_v1's 9971 @100k.
  - **softbeta RESULT (killed @105k): bar failed identically — 8424 @100k
    vs betanoise_v1 9971 (−15.5%; softgauss was −14.3% vs ITS sibling).
    Diagnostics clean again** (logpi mean 5.6 / max 16.6, α 0.12, conc
    21). **Soft-Bellman line CLOSED**: the soft backup is a
    parameterization-independent ~14-15% regression on TD7/HalfCheetah
    with no numerical pathology — the mechanism is the backup change
    itself (stochastic a′ target variance + higher settled α), not the
    head. Entropy belongs in the actor loss only on this algorithm/env;
    matches the sac_v11 result, now shown in isolation.
- A priori theory of sdnoise (theorist agent, 2026-07-16): actor-loss
  stationary point gives sigma_i^2 ≈ alpha/|H_ii| (H = action-Hessian of
  Q) — curvature-matched exploration allocation under a fixed entropy
  budget. Explains the measured 100× sigma spread, the sigma_max=1.0 pin
  (flat dims want more than the ceiling; raising LOG_STD_MAX would drain
  alpha and SHRINK sigma in curved dims — don't), and why it composes
  with TD7's hard backup (behavior-only change; control path untouched).
  Diagnosed 8M bottleneck: asymptotic ceiling (eval ~17k by 500k, <3%
  gain to 890k) from (1) temporally white noise averaging out over a
  stride, (2) locality/unimodality, (3) running-max clamp slowing value
  propagation (+5%/400k). Ranked proposals: P1 OU/AR(1)-correlated
  sigma(s) noise (building as td7_sdnoise_v2, rho=0.9, behavior-only);
  P2 state-dependent low-rank exploration covariance (pending critic);
  P3 decouple greedy mu from noisy entropy path; P4 twin-disagreement
  bonus (risky); P5 second-mode head (wrong env); P6 novelty-relaxed
  clamp (REJECTED — invariant-violating, HL-Gauss death signature).
  Adversarial critic verdicts (2026-07-16): mechanism confirmed with a
  sign condition — sigma_i^2 = −alpha/H_ii holds only where H_ii<0; flat
  and convex dims run monotonically to the sigma=1 clamp, so the measured
  100× spread is BIMODAL pegged/pinched, not a smooth curvature map (and
  the first-order term dies from E[eps]=0 at every iterate, so mu being
  mid-optimization is a non-issue). P1 BUILD-WITH-CHANGES: rho 0.8 not
  0.9 (checkpoint-selector variance grows (1+rho)/(1−rho) — TD7 picks
  checkpoints by MIN training-episode return, the same variance-
  sensitivity that killed the soft arms, hitting the selector not the
  targets; HalfCheetah-only since no termination), stationary N(0,I)
  reset init (zeros under-noises early episode), diagnostics
  ep_return_var + ckpt_resets. Effect size corrected: white noise gives
  √T·sigma coherent reach, not zero — AR(1) at rho .8 gives ~3× reach
  and sustained-posture gaits. P2 DON'T-BUILD: wrong axis (spatial not
  temporal) AND a real accounting flaw — diagonal target_entropy makes
  the logdet(I+LᵀD⁻¹L)≥0 term steal budget from diagonal sigma,
  over-sharpening it. P3 BUILD: restores TD7's exact DPG fixed point for
  mu (v1's mu sits O(sigma²·∂³Q) off the greedy argmax from
  noise-smoothing); near-free, min-twins cover the point-estimate risk.
  Plan: td7_sdnoise_v2 with two orthogonal flags (noise_corr=0.8 OU;
  decouple_greedy) → arms: OU-only, then OU+decouple for attribution.
- Beta-vs-Gaussian noise gap (user obs.): even @200k, beta ahead @300k,
  gauss pulls away @400k+ (15490/15186 @400k, 16178/15414 @500k). Late
  divergence timing implicates the sharpened-policy regime, two suspects:
  (1) mean-greedy bias — deterministic/target action = Beta mean, less
  extreme than the mode on boundary-skewed dims (26% saturated;
  Beta(30,2): mean→a 0.875 vs mode→0.933), bias grows with concentration;
  (2) entropy-budget boundary loophole — Gaussian's unclamped μ can
  exceed ±1, giving free post-clamp precision at the bounds while entropy
  is charged pre-clamp; Beta pays real mass to sharpen against a
  boundary. `td7_betanoise_v3` (mode-greedy at all 3 deterministic sites
  incl. backup; charts/mode_mean_gap diagnostic) tests (1); registered
  prediction: tracks v1 to ~300k and closes most of the late gap; failure
  → (2) dominates and the lever is boundary handling. RESOLVED 2026-07-16:
  v3 (mode-greedy) killed @201k — 10,484 vs v1's 11,812 @200k, behind at
  matched steps with mode_mean_gap=0.084 (bias real, correcting it doesn't
  help) ⇒ hypothesis (1) falsified. v4 (stretch 1.1) at 77k: 8,861 vs
  v1's ~7,600 pace, frac_pinned=0.36 ⇒ initially read as hypothesis (2)
  confirmed — WRONG, interpolation artifact (linear interpolation of a
  convex curve understates v1 mid-rise; use matched-step columns only).
  v4 killed @152k: 9,624/10,376 @100k/150k vs v1's 9,971/11,428 — behind
  and diverging despite frac_pinned=0.41 (loophole heavily used, still
  loses). **BOTH single-change fixes regressed vs v1 ⇒ Beta line CLOSED:
  v1 (mean-greedy, native support, cold start) was already the best Beta
  configuration; Gaussian's late-game edge remains mechanistically
  unexplained but is robust. Focus: sdnoise advancement track.**
  betanoise_v1 killed
  @515k under kill-underperformers directive (curve recorded for
  matched-step reference).
- `td7_sdnoise_v1` (in flight, 200k+): SAC's fc_logstd state-dependent noise
  head only, entropy confined to actor loss, α autotuned to target_sigma=0.1
  budget. @150k = 10877 vs td7_v1 10658 — at/above baseline at matched steps,
  12196 @200k. First TD7 modification on this board that isn't losing.
- `td7_betanoise_v1` (in flight): v215 Beta head (conc = 1+softplus ≥ 1,
  z∈(0,1) → a = 2z−1) as state-dependent exploration; bounded support removes
  Gaussian clamp-mass distortion at |a|=1; entropy in actor loss only, α_ent
  autotuned to the σ=0.1-equivalent z-space target (−1.577/dim incl. −ln2
  scaling). Known transient: α_ent starts at 1.0 → policy near-uniform until
  ~35k (concentrations 1.2→2.9 by 33k, returns flip positive).
- P1/P3 attribution round (td7_sdnoise_v2 arms vs v1 matched-step; 2026-07-15):
  **ou alone (ρ=0.8): FAILED** — 13,538 @300k vs v1 13,970 (−3.1%); killed.
  **dg alone (decouple_greedy): FAILED** — 12,494 @265k vs v1 13,147 (−5.0%),
  gap widening after a transient catch-up @154k; killed per registered rule.
  σ-allocation diagnostics (σ_min/mean/max ≈ v1) do NOT support the
  "μ parks on sharp ridge → σ pinches" story; favored mechanism: v1's μ on
  noise-smoothed Q is (a) optimized for the *executed* noisy policy — the
  very quantity training returns and the checkpoint selector measure — and
  (b) an adaptive-width regularizer against narrow critic overestimation
  artifacts (same class TD3 target-smoothing fights); dg removes both.
  Timing fits: divergence begins ~200k when critic sharpens and σ map
  differentiates (objectives coincide while σ is large/uniform).
  **oudg (ρ=0.8 + dg): PASSED 300k bar** — 15,206 @305k vs bar 13,970
  (v1 @300k); early +4.9% lead had evaporated to a tie @286k, then late
  surge. Effects strongly non-additive (both singles hurt, combo leads) —
  n=1 seed each, so run-to-run variance remains a live alternative.
  Next bar: oudg ≥16,000 @400k (v1 15,490) → 8M candidate.
  **oudg PASSED 400k bar (2026-07-16):** 16,080 @400k matched (+3.8% vs
  v1 15,490), +7.3% @350k (15,815 vs 14,736); the 200k-300k tie resolved
  as a sustained late surge. PROMOTED: `td7_sdnoise_v2_oudg_8m` queued
  (same config: ρ=0.8 default + --decouple-greedy, seed 1, 8M); the 1M
  run rides to completion for the head-to-head vs v1's 17,122 @1M.
  Registered prediction for 1M: oudg ≥17,122; failure ⇒ the surge was
  variance, reassess before reading the 8M run as an advancement.
- `td7_sdnoise_v3_anti` (queued 2026-07-15, ρ=0, decouple off): antithetic
  σ-gradient — Q(μ̄±σε) with the same ε, odd-order terms cancel pathwise,
  σ head gets near-pure curvature signal (toy-quadratic probe: unbiased,
  variance ↓ up to 28.8× per dim, never worse). Registered predictions:
  charts/sigma_spread opens earlier than v1's ~40k; tracks-or-beats v1
  throughout (bars: ≥9,700 @100k, ≥12,000 @200k). antithetic_sigma=False
  verified byte-identical to v2; all-flags-off ≡ v1.
  **RESULT (killed @109k): BOTH predictions falsified.** 8,664 @100k vs v1
  9,726 (−10.9%), behind from 50k (−8.3%); sigma_spread only 1.83 decades
  @109k vs v1 reaching ~2.1 by ~40k — cleaner curvature signal made σ
  differentiation SLOWER, not faster. SNR-bottleneck hypothesis dead.
  Emerging pattern across the round: **actor-gradient stochasticity is
  functional, not noise.** Three variance-reduction interventions (dg:
  deterministic μ-gradient; anti: odd-term cancellation; both coupled-anti's
  O(σ²) μ-symmetrization) all regressed; the behavior-only change (ou) was
  ~neutral alone. Two candidate mechanisms, not yet discriminated:
  (a) selector harvesting — TD7's max-checkpoint selection converts
  actor-update noise into performance ES-style; cleaning the gradient
  narrows the candidate distribution the max feeds on; (b) heavy-tailed
  single-sample updates in log σ act as a random-walk widener that reaches
  useful allocations before the critic's true curvature is informative.
  Both imply the same directive: do NOT clean/denoise the actor gradient;
  advancement levers live on the behavior/selection side (oudg's late
  surge is consistent).

Base: `ppo_continuous_action_idbd_beta_v2.py` (α_init=0.05, θ=0.05, max_α=1, **epochs=10**, mb=num_envs, grad clip 0.5, unit-feature decay, joint actor+critic IDBD).

**All ablations use `update_epochs=1`** (only standard baseline keeps 10).

| ID | Exp name | Change vs v2 | Paper motivation |
|----|----------|--------------|------------------|
| baseline | `idbd_beta_v2` | epochs=10 (only multi-epoch run) | control |
| epochs1 | `idbd_beta_v2_abl_epochs1` | epochs=1 only | one-pass incremental control |
| bounds | `idbd_beta_v2_abl_bounds` | β≥−10, \|Δβ\|≤2 (+1 epoch) | § after Fig. 2 practical bounds |
| theta001 | `idbd_beta_v2_abl_theta001` | θ=0.001 (+1 epoch) | long-run fig θ |
| splitac | `idbd_beta_v2_abl_splitac` | separate actor/critic IDBD (+1 epoch) | single unit vs mixed heads |
| mb1 | `idbd_beta_v2_abl_mb1` | minibatch size 1 (+1 epoch) — **killed, too slow** | one-example incremental |
| ppomb | `idbd_beta_v2_abl_ppomb` | standard PPO `num_minibatches=32` (mb≈1024 w/ 16 envs) (+1 epoch) | CleanRL default batching |
| noclip | `idbd_beta_v2_abl_noclip` | max_grad_norm=0 (+1 epoch) | no grad clip in LMS |
| x2decay | `idbd_beta_v2_abl_x2decay` | h decay uses Linear input x² (+1 epoch) | eq. (6) |
| hlgauss | `idbd_beta_v2_abl_hlgauss` | HL-Gauss critic, support [−20,20] on **normalized** returns (+1 epoch) | v215-style CE value; reward norm ⇒ no ±20k |
| nopolyclip | `idbd_beta_v2_abl_nopolyclip` | unclipped policy surrogate −A·ratio (+1 epoch) | no PPO ratio clip |
| td7 | `idbd_td7_v1` | IDBD on TD7 baseline, **1M** steps, num_envs=1 | cross-algo |
| td7_papertheta_b32 | `idbd_td7_papertheta_b32` | TD7+IDBD + **paper θ/β bounds** + **batch=32** (default 256) | smaller batches for IDBD meta |
| pmpo | `idbd_pmpo_v1` | IDBD on `iterthink_v24_beta_d3bucket_mtp_pmpo_v1`, **1M** steps | run …pmpo_v1__1__1782021163 |
| splitac_silu_hlgauss | `idbd_beta_v2_abl_splitac_silu_hlgauss` | splitac + **SiLU** + **HL-Gauss** critic (+1 epoch) | combo of best pieces |
| splitac_silu_hlgauss_noadvnorm | `idbd_beta_v2_abl_splitac_silu_hlgauss_noadvnorm` | same + **norm_adv=False** | raw GAE advantages |
| splitac_silu_hlgauss_d3retnorm | `idbd_beta_v2_abl_splitac_silu_hlgauss_d3retnorm` | noadvnorm + **Dreamer3 retnorm** (EMA P5–P95 scale, floor=1) | `A /= max(1, P95−P5)` |
| splitac_silu_hlgauss_d3retnorm_norewnorm | `idbd_beta_v2_abl_splitac_silu_hlgauss_d3retnorm_norewnorm` | d3retnorm + **no env reward norm**; HL-Gauss **symlog** [−10,10] | raw rewards + retnorm handle scale |
| splitac_silu_hlgauss_incr | `idbd_beta_v2_abl_splitac_silu_hlgauss_incr` | **incremental**: 1×num_envs env-step → optim (1-step TD) | online IDBD timescale for \(h\) — **failed** (advnorm+rewnorm) |
| splitac_silu_hlgauss_incr_d3retnorm_norewnorm | `idbd_beta_v2_abl_splitac_silu_hlgauss_incr_d3retnorm_norewnorm` | incr + **no advnorm** + **no rewnorm** + **d3 retnorm** + HL symlog | fix scale; sole A scaler = retnorm |
| splitac_silu_hlgauss_incr_noadvnorm | `idbd_beta_v2_abl_splitac_silu_hlgauss_incr_noadvnorm` | incr + **rewnorm** + HL linear [−20,20] + **noadvnorm** | parallel to raw/d3; batch-noadvnorm recipe |
| splitac_silu_hlgauss_incr_raw | `idbd_beta_v2_abl_splitac_silu_hlgauss_incr_raw` | incr + **no rewnorm / retnorm / advnorm** + HL symlog | fully raw A scale |
| gae32_* | (batch collect-32) | superseded by delayed_gae32_* | partial logs kept |
| delayed_gae32_noadvnorm | `…_delayed_gae32_noadvnorm` | mature GAE H=32 + **rewnorm + noadvnorm** | **winner** ~4.4k @1.6M |
| delayed_gae32_raw | `…_delayed_gae32_raw` | mature GAE + no rewnorm/retnorm + symlog | underperformed; killed |
| delayed_gae32_d3retnorm_norewnorm | `…_delayed_gae32_d3retnorm_norewnorm` | mature GAE + d3 retnorm + no rewnorm | underperformed; killed |
| delayed_gae32_noadvnorm_norewnorm | `…_noadvnorm_norewnorm` | winner − rewnorm (+ HL symlog) | is rewnorm required? |
| delayed_gae32_noadvnorm_nopolyclip | `…_noadvnorm_nopolyclip` | winner − PPO ratio clip | clip vestigial? |
| delayed_gae32_noadvnorm_noactorclip | `…_noadvnorm_noactorclip` | winner − actor grad clip (critic still clipped) | clip throttling π? |
| delayed_gae32_noadvnorm_pmpo | `…_noadvnorm_pmpo` | delayed GAE + **PMPO** + rewnorm, no adv/retnorm | sign-soft + reverse KL |
| flush_gae32_noadvnorm | `…_flush_gae32_noadvnorm` | **collect-32 → optim all → throw away** (no slide) | polish / reuse vs delayed |
| delayed_gae256_noadvnorm | `…_delayed_gae256_noadvnorm` | delayed mature **GAE H=256** + noadvnorm | longer multi-step credit vs H=32 |
| delayed_hlong | `…_delayed_gae32_noadvnorm_hlong` | delayed H=32 noadvnorm + **κ=0.05** (~20× h mem) | longer IDBD eligibility |
| delayed_hlong200 | `…_delayed_gae32_noadvnorm_hlong200` | delayed H=32 noadvnorm + **κ=0.005** (~200× h mem) | very long eligibility |
| delayed_wclip | `…_delayed_gae32_noadvnorm_wclip` | delayed noadvnorm; **meta on raw g**, clip only weight step | grad-clip was corrupting IDBD meta |
| delayed_emarrms | `…_delayed_gae32_noadvnorm_emarrms` | delayed noadvnorm + **EMA-RMS(A)** (not d3 retnorm) | |A| collapse under fixed α |
| delayed_sepclip | `…_delayed_gae32_noadvnorm_sepclip` | delayed noadvnorm; **separate** actor/critic grad clips @0.5 | joint clip coupling vs independent |
| sepclip_papertheta | `idbd_sepclip_papertheta` | sepclip + **paper θ** (θ=0.05, stream 0, β bounds) | stack best clip path × paper meta |
| **v3 base** | `idbd_v3_base` | sep A/C **weight** clip + meta on **raw** g | new baseline for paper fixes |
| v3_metascale | `idbd_v3_metascale` | v3 + meta g unit-RMS | paper O(1) meta products |
| v3_papertheta | `idbd_v3_papertheta` | v3 + θ=0.05, stream 0, β bounds | paper θ / bounds |
| v3_paperx | `idbd_v3_paperx` | v3 + h decay x²∝g² | paper feature-present decay |
| v3_paperlms | `idbd_v3_paperlms` | v3 + A2C score + MSE V | closer to LMS δ² |
| v3_noclip | `idbd_v3_noclip` | v3 + max_grad_norm=0 | no clip anywhere |
| v3_td7critic | `idbd_v3_td7critic` | v3 + **EMA target clip** on bootstrap V / return targets (TD7-inspired; no AvgL1) | critic target scale (≠ actor SNR) |
| v3_onesidedA | `idbd_v3_onesidedA` | v3 + **one-sided** A floor: gain=min(max_boost, max(1, ref/EMA\|A\|)) | |A| starvation without early dampen |
| rankgauss_hlong | `…_rankgauss_hlong` | rankgauss + κ=0.05 | **wrong base; killed** |
| **sgdstream_v1** | `sgdstream_v1` | sepclip loop, IDBD → **torch.optim.SGD(0.05)** | attribution control: inert-IDBD ≡ SGD |
| **sgdstream_anneal_v1** | `sgdstream_anneal_v1` | sgdstream + **linear LR anneal → 0** | plateau = constant-step noise floor? |
| **v4_rankwin** | `idbd_v4_rankwin` | sepclip + **rank→probit** adv over sliding 4096 window | rankgauss shaping × fresh stream |
| **autostep_v1** | `idbd_autostep_v1` | sepclip + **per-tensor Autostep meta** (Δβ=θ·Σ(-gh)/vmax, θ=0.01, τ=1e4) | restore paper-scale meta dynamics, scale-free |
| splitac_silu_hlgauss_theta05 | `idbd_beta_v2_abl_splitac_silu_hlgauss_theta05` | silu_hlgauss + **θ=0.5** | force α off init plateau |
| splitac_silu_hlgauss_theta2 | `idbd_beta_v2_abl_splitac_silu_hlgauss_theta2` | silu_hlgauss + **θ=2.0** | extreme meta; adapt or blow |
| splitac_silu_hlgauss_thinktrunk | `idbd_beta_v2_abl_splitac_silu_hlgauss_thinktrunk` | silu_hlgauss loop + **ThinkTrunk** (sep. A/C) | capacity vs 64×64 MLP |
| splitac_silu_hlgauss_rankgauss | `idbd_beta_v2_abl_splitac_silu_hlgauss_rankgauss` | **rankgauss** adv instead of mean/std | magnitude-free shaping |
| splitac_silu_hlgauss_rankgauss_thinktrunk | `idbd_beta_v2_abl_splitac_silu_hlgauss_rankgauss_thinktrunk` | **rankgauss** + **ThinkTrunk** (sep A/C) | stack winner × capacity |
| splitac_silu_hlgauss_pmpo | `idbd_beta_v2_abl_splitac_silu_hlgauss_pmpo` | **PMPO** no advnorm, mb=16 | sign-soft + reverse KL |
| **gated if incr does well** (last20≥2800 @≥2M) | | | |
| incr_targetv | `…_incr_targetv` | incr + **Polyak target V** (τ=0.005) | online TD bootstrap stability |
| incr_gae16 | `…_incr_gae16` | **16-step GAE** λ=0.95, mb=16 | multi-step credit, still frequent |
| incr_vfirst | `…_incr_vfirst` | incr **critic-first** then actor | fresher A every step |
| incr_score | `…_incr_score` | incr **A2C score** (no PPO clip) | single-pass on-policy purity |

## Queue

Submit every run to the machine-wide `mlq` daemon with a deliberate global
compatibility limit. These characterized CleanRL jobs normally use 3; use 1
when a new workload's resource behavior is uncertain.

```bash
MLQ_RUN_NAME=idbd_v3_myexp
MLQ_SCRIPT=cleanrl/idbd/ppo_continuous_action_idbd_beta_v2_abl_splitac_silu_hlgauss_delayed_gae32_noadvnorm_v3.py
mlq submit --name "$MLQ_RUN_NAME" --max-parallel-runs 3 --cwd "$PWD" -- \
  .venv/bin/python -u "$MLQ_SCRIPT" \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name "$MLQ_RUN_NAME" \
  --total-timesteps 8000000 --seed 1 --compile --compile-mode reduce-overhead

mlq status
mlq show JOB_ID
mlq logs JOB_ID --follow
mlq logs JOB_ID --stderr
mlq cancel JOB_ID
# Failed or lost jobs only:
mlq retry JOB_ID
```

Record the returned job ID and declared limit. Use `mlq` rather than process
inspection for queue state, logs, retries, and cancellation.

## Results table (fill when complete)

| Exp | Steps | Mean return (last 20) | approx_kl | clipfrac | α_mean | α_p10 | α_p90 | grow_frac | notes |
|-----|-------|----------------------|-----------|----------|--------|-------|-------|-----------|-------|
| baseline | | | | | | | | | |
| bounds | | | | | | | | | |
| theta001 | | | | | | | | | |
| splitac | | | | | | | | | |
| mb1 | | | | | | | | | |
| epochs1 | | | | | | | | | |
| noclip | | | | | | | | | |
| x2decay | | | | | | | | | |
| hlgauss | | | | | | | | | |
| nopolyclip | | | | | | | | | |

**Keep** ablations that improve return / healthy KL / α differentiation.  
**Drop** those that hurt or no-op. Final combo → `idbd_beta_v3`.
