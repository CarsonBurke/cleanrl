# TD7 latent-Q: diagnose action credit before adding introspection

Analysis snapshot: 24 October 2026. Six parallel reviews covered reference/baselines, self-models, deliberation, other critic families, belief-coordinate mathematics, and primary literature. Checked `CLAUDE.md`, `cleanrl/td7/FAMILY.md`, implementation paths, and CPU TensorBoard events through `.venv/bin/python` and `scripts/_runs.py`. No training, GPU work, cancellation, or existing-code edits. Prescriptions below are hypotheses, not demonstrated improvements.

**Recommendation:** retain v27's SALE action path, replay/cadence, and target construction. Make introspection predict the reliability of *directional action credit*, validated on independent episode blocks; do not differentiate a free error predictor. The failures establish a calibration/control mismatch, not that introspection cannot work. No available evidence warrants promising a score above v27.

## 1. What the measurements actually say

Values below are exact `eval/episodic_return` checkpoints, rounded to whole returns, rather than smoothed windows. Evaluation uses checkpoint systems; `charts/episodic_return` is live stochastic behavior. Deliberation evaluation removes execution noise but still randomly samples candidates, so it is not deterministic inference in the same sense as v27.

| Method | 100k | 300k | 800k | 1M | Later checked evaluation |
|---|---:|---:|---:|---:|---|
| Original TD7 | 10,302 | 15,821 | — | 17,248 | Run ends around 1.04M |
| Fast state-dependent noise | 10,554 | 16,090 | 17,206 | 17,744 | Historical comparison |
| Preaction v5 | 11,498 | 16,301 | 17,578 | 18,445 | 18,437 at 1.2M |
| **v27 reference** | **12,029** | **15,792** | **18,525** | **18,751** | **19,254 at 1.7M; 19,242 at 1.705M** |
| Selfmodel | 9,747 | 11,648 | 14,922 | — | 15,036 at 1.2M; 16,475 at 1.345M |
| Selfcritic + deliberate | 10,674 | 12,219 | — | — | Live return −109 at 307,999 |
| Deliberate | 7,850 | 15,274 | — | — | 15,525 at 400k; 16,320 at 465k |
| Consequence v1 | 9,799 | 13,204 | 17,593 | — | 14,738 at 1.2M |
| Costate v2 | 9,467 | 13,334 | 17,926 | — | — |
| Closed costate v3 | 7,142 | 13,771 | 17,067 | — | — |
| Belief residual v4 | 10,933 | 13,792 | 16,144 | — | 18,470 at 1.7M |
| Heatbelief / chart | 8,947 / 7,685 | 11,997 / 11,881 | — | — | — |
| Heatbelief dictionary | 6,702 | — | — | — | — |

The reference is a late-learning improvement, not an early universal winner. V27 is algorithmically v5 plus an observational probe; their score differences do not show that the probe helps. Historical baselines are not controlled reruns. Several FAMILY.md/brief numbers are stale or window-dependent, notably consequence v1 at 800k.

**Measurement caveat:** losses, sigma, and Q diagnostics use optimizer `training_steps`; fresh/eval/live returns use environment steps. Deferred checkpoint assessments separate these clocks. Selfmodel windows below align training diagnostics by event wall time to environment episodes. Do not compare all tags at numerically equal steps.

### Selfmodel: useful scalar prediction, unvalidated control derivatives

Across the loaded histories, median episode fresh MAE falls **161.55 → 34.56** for selfmodel and **193.04 → 56.79** for the combined arm when subtracting signed predicted error. Correction improves 87.5% / 85.1% of episodes. Episode-mean predicted versus actual error correlations are .889 / .921. Thus “b never learned error” is false.

Within-episode `fresh/sm_corr` medians are only .0739 / .118, but p90 values are .479 / .956. Selfmodel reaches .939 at 1,299,999; combined reaches .932 at 299,999. Much useful calibration is an episode offset, but localization is not uniformly absent. Logs lack per-state Q/G/b and derivative ground truth; copying Q variation while return tails remain smooth is a possible explanation, not a checked decomposition.

More seriously, **the logged correction subtracts signed b; the actor subtracts relu(b)**. Improved corrected MAE does not validate the actor objective. At the actor query:

`grad J = grad C − 1[b>0] grad b + entropy gradient`.

Behavior-error regression does not identify that new gradient at a changed belief and zero-noise center. Even a perfectly predicted state offset should contribute no action correction; a flexible model can instead assign arbitrary off-support slopes.

| Environment-step window | Selfmodel live median | v27 live median | Selfmodel fresh bias median | Selfmodel sigma median / actor penalty median |
|---|---:|---:|---:|---|
| 250–350k | 193 | 15,032 | +42.2 | .499 / 41.9 |
| 750–850k | 13,388 | 17,309 | −285.8 | .277 / .142 |
| 1.15–1.25M | 15,557 | 17,972 | −290.9 | .078 / — |
| 1.25–1.35M | 1,200 | 18,090 | +59.0 | .472 / 44.1 |

The last window's live p10/p90 are −388 / 16,349: intermittent catastrophe, not uniform policy collapse. Checkpoint evaluation preserves competent old policies. Sigma/penalty co-instability supports an actor-objective problem but does not isolate its cause without derivative instrumentation or ablation. The combined arm's plan-gain maximum is 93.02 at 105,999; attributing that specifically to b requires unavailable per-candidate score decomposition.

**Effective samples:** gamma=.99 and tail cutoff 459 leave 541 eligible rows per 1,000-step episode. A 20k-row window contains about 20 episode blocks and 10,820 eligible rows, not 20k independent labels. However, “ESS=20” is not measured. With iid reward innovations, return correlation is approximately `.99^k`; finite-block ESS `n² / [n + 2 sum_k (n−k) rho(k)]` gives 3.33 per 541-row block, about **66.6 across 20 independent episodes**. Persistent episode effects can reduce this toward 20. Use episode-block validation/variance, not a fixed claimed ESS or one-row-per-episode dogma.

### Deliberation: policy inconsistency before any proven winner's curse

`Mind.plan` scores a deterministic first candidate, two deterministic raw-actor continuations, and terminal raw-actor twin-min Q at sigma=.2. Execution adds *new* Gaussian noise around the selected candidate. TD targets and actor optimization remain raw-actor queries, while data comes from the noisy planned policy. These are different continuation/execution objects. The planner is not evaluating repeated deployment of its own policy.

Near 300k (275–325k medians), deliberate versus v27: live return **13,410 / 15,050**, fresh bias **+29.71 / −60.71**, fresh MAE **43.82 / 68.91**, sigma **.2223 / .1981**. Plan gain is **4.045**, mean retained fraction **.003**. Small on-replay model losses—normalized dynamics MSE about .0040, reward MSE .000175—do not establish accurate candidate ranking or value-sensitive model extrapolation.

For 64 independent equal-value Gaussian score errors with standard deviation tau, expected selected error is **2.344 tau**. Gains 4–7 can arise from differential error of only **1.7–3.0 Q units**. Equicorrelation rho reduces the selection component by `sqrt(1−rho)`. This is a null calculation, not proof that selection fails: fresh absolute error contains common-mode state error and cannot estimate tau. Likewise, variance doubles only for *unselected independent* candidate/execution draws; hard selection invalidates that variance claim. Checked: another stochastic draw is added after choosing an already shifted candidate.

The **7× speed claim needs qualification**. Around 300k cumulative SPS is 103 versus 694. But deliberate rollout time per 10k abruptly falls **55.29s at 320,999 → 5.91s at 330,999**; update/eval also fall. At 460,999 the phase sum is **22.69s/10k**, versus a historical reference interval near 400k of **15.72s/10k**. Recent interval overhead is roughly 1.3–1.5×, not 7×; contemporaneous machine load and attribution were not measured. Cumulative SPS retains the slow prefix. Read-only job 14745 inspection reported running, declared limit 4; frozen event endpoint was 466,999. At that endpoint fresh bias is −19.20, so it is not invariably optimistic.

### Heatbelief and consequence-Q: reject the broad causal stories

Heatbelief v1/chart are strongly pessimistic (mean fresh bias −137.6/−144.4 over 250–350k), yet score about 12k at 300k. Pessimism is insufficient. V1 has approximately 2.87 contributing bumps near 300k; dictionary v2 has lower reported loss at 100k (4.40 versus v27 8.34) but return 6,702 versus 12,029. These structurally different losses are not a common accuracy scale.

Heatbelief changes basis expressivity, point-only TD, actor expectation, target integration, and action-Jacobian routing together. Its chart is detached `zsa(zs, mu_current)`, not differentiable `zsa(zs, queried_action)`. Its failure does not isolate the necessity of the SALE action path.

Consequence v1's fresh MAE at one episode near 800k is 9.96 versus v27 63.30. Across **700–900k**, episode-mean MAE is instead **74.5 / 58.3 / 45.7 / 47.1** for consequence v1 / costate v2 / closed v3 / v27. “4× more accurate” is not a stable mature-stage fact. Closed v3 is slightly more accurate in this window and scores lower, supporting the narrower claim that aggregate calibration does not order control quality. Consequence-Q also changes continuation sigma and removes v27's twin-min target, confounding the comparison.

Residual v4 is the clearest exploitation warning: over 1.5–1.75M, fresh bias median/p90 **+60.2/+527.5**, live return p10 **6,731**, versus v27 **−66.3/−42.0** and **17,860**. Its residual directional gradient RMS grows **.224 at 300k → 1.031 at 1.7M**. Costate supervision constrains V, not the added residual derivative; a small scalar residual can bypass the grounded gradient. Association with optimism is strong, but policy mismatch still prevents a causal “positive bias caused collapse” conclusion.

## 2. Why v27 works, and what is actually limiting it

Let `C(s,mu,sigma)` be twin-mean Q at `u=mu, zsa=zsa(zs,mu)`. Its actor receives:

`g_mu = Q_mu + Q_u + J_zsa(mu)^T Q_zsa`;

`g_logsigma = sigma * Q_sigma + alpha`.

SALE parameters are frozen during actor differentiation, but its action Jacobian remains live. SALE trains `zsa(zs(s),a)` to predict stopped next-state features, grounding a route from action changes to learned consequences. Hermite center tangents traverse the **same total mean direction**. Behavior and point TD anchor realized queries; replay, delayed policy updates, target bounds and checkpointing provide stabilizing structure. This explains plausible advantages, not experimentally isolated contributions.

Pessimism enters explicitly through `min(Q1,Q2)=mean(Q)−abs(Q1−Q2)/2` in targets; the actor ascends their mean. Target lag/bounds and function approximation also matter. **Exact Gaussian smoothing is not enforced.** Direct behavior TD wants `Q(s,mu,sigma,u)=physical_Q(s,clip(u))`; Hermite projection wants the center to represent its Gaussian average. These are generally incompatible for a curved action-value function in one shared unrestricted head. Their weighted compromise can be useful. LAP's residual-dependent sampling and Huber loss also break the ideal unweighted squared-projection moment interpretation. Gaussian clipping itself does not invalidate Stein identities when applied to `f(clip(u))`; truncating the Hermite expansion and learning inconsistent query semantics are the issues.

The entropy target fixes expected `sum(log sigma)`, hence a geometric budget near .1, **not arithmetic mean sigma=.1**. Observed arithmetic mean decreases from roughly .204 early to .161 late; extrema reach 1 and .0067. A claim of uniformly small noise is unsupported.

There is **slowing improvement, not a demonstrated hard 19k ceiling**: eval medians rise 18,558 over .8–1M, 18,876 over 1–1.2M, and 19,026 over 1.4–1.705M. A checkpoint-like jump occurs 1.670→1.675M (19,014→19,239). No 2M reference measurement exists. Late live median is 18,193, below checkpoint eval 19,026; no live deterministic evaluation distinguishes noise cost from checkpoint lag.

Late Hermite R² median falls to **.086** from .189 early; about **29.4%** of late logged minibatches have negative R². Tangent standard deviation falls **3.91→1.98**, while critic loss falls **7.22→3.12**. These establish diminished incremental tangent fit, not whether the actor direction is wrong, exploration is exhausted, or representation is saturated. Those competing explanations need directional data.

Fresh bias is not calibrated error for the deployed deterministic checkpoint policy. V27 compares behavior queries against noisy live returns while bootstrapping a different target policy; late median is −64.7, 96.4% negative. Its truncated probe omits a tail below .01 weight (up to roughly 18 value units at Q≈1800), biasing error upward. Selfmodel instead bootstraps that tail with current twin mean; its probe is consequently not identical. Age cannot separate policy drift, exploration cost and approximation error.

**Identifiability:** a function vanishing on observed `(s,mu_behavior,sigma_behavior)` can have arbitrary transverse slopes, preserving calibration while changing actor gradients. But parameter-conditioned values are not universally unidentifiable: `F(s,b)=E[Q(s,clip(mu+sigma eps))]` has precise semantics. Extra hidden coordinates have zero true gradient only if they leave the induced action law and continuation unchanged; covariance or temporal-correlation coordinates can have real effects.

## 3. Ranked prescriptions

All experiments here are future proposals, not performed work. Preserve seed 1, HalfCheetah-v4, approximately 2M steps and matched v27 cadence; use mlq and the harness for any future ML work. Evaluate checkpoint **and live deterministic** policies, log checkpoint identity/age, geometric/arithmetic sigma, clipping frequency, and total/direct/SALE gradient norms. A single seed distinguishes mechanisms here, not statistical dominance. Cheap rejection probes are not reduced training runs offered as performance evidence.

### 1 — Cross-fitted directional self-criticism in belief coordinates

**Mechanism.** Replace free `grad b` with a critic-step auditor trained on independent fresh episode blocks. Freeze/version the continuation target, and form `y=r+gamma*V_target(s')`. For Gaussian noise define normalized features `h=(eps,(eps²−1)/sqrt(2))`. Under uniform conditional sampling, `E[h*y]` identifies the whitened first/diagonal-second belief derivatives of the smoothed physical target. For v27's proposed gradient, the matching coefficients are `d=(sigma*g_mu, sigma*Q_sigma/sqrt(2))`. The innovation `h*(y−c(s)−d·h)` tests its directional error, with epsilon-independent baseline c and frozen d. This equality is for the sampling law, not LAP/Huber-weighted projection; use fresh uniform/episode-balanced auditor data.

Start with one proposed actor direction rather than all coefficients: score estimator `(y−c(s))*sum_i eps_i*v_i/sigma_i` estimates its mean-action derivative. Choose v and freeze d/c **before the evaluated noise draw**; post-selection directions leak noise. Cross-fit episodes and critic snapshots, estimate block variance, and predict reliability from state/phase and policy version. Fit and validate on separate episode blocks. Apply only a **detached state weight or trust radius** to actor improvement; keep entropy budgeting separate. No differentiated action-conditioned error field. Labels from old beliefs audit old beliefs: bound drift in Gaussian KL, refresh or abstain rather than silently treating old eps as current-policy evidence.

**Why it could beat v27.** Detect shared, systematic directional errors invisible to twin agreement, suppress harmful updates without inventing new action slopes, and preserve useful SALE credit. It directly tests the missing covariance between realized consequences and proposed improvement. This is weaker than an independent return oracle: bootstrap bias, low score SNR and state-generalization remain.

**Cheapest decisive experiment.** First add a shadow auditor to a full v27 run; compare held-out sign/reliability prediction against state-offset and twin-agreement baselines at 300k/1M. Enable detached gating only if predictive. Then compare matched 2M learning. Existing `preaction_kalman_v1` already uses twin-gradient agreement and reaches 16,871 at 300k but only 18,467 at 2M; renaming that gate is not novelty.

**Failure signature.** Held-out covariance near zero, uncertainty that never shrinks, or a gate behaving like a global learning-rate reduction; same-direction shared bootstrap error remains invisible to the auditor. Do not turn “noisy estimate” into a confident conservative bound.

**Hooks.** `stage`, `flush`, `_log_fresh` for versioned pre-update labels; a separate auditor beside `_critic_loss_fn`; `_actor_loss_fn` for detached weighting. Preserve the Q→mu/u/zsa derivative. Later coefficient correction is justified only after sign calibration succeeds.

### 2 — A flexible SALE belief potential with heat consistency

**Mechanism.** Give the center a single coherent meaning: `C(s,mu,sigma)=E[physical_Q(s,clip(mu+sigma eps))]`. Retain flexible neural capacity, the differentiable SALE action path, point labels at sigma zero, and Hermite center labels; remove the conflicting positive-sigma direct-preaction label. Tie variance credit to mean curvature through the Gaussian identity `partial_logsigma_i C = sigma_i² * partial²_mu_i C`, using **total** mean derivatives through u and zsa. Approximate the constraint by sampling one coordinate per row and batching finite differences at mean/scale perturbations. This adds critic work, not model rollout or environment branching.

**Why it could beat v27.** Eliminate unsupported independent sigma slopes while retaining rich action geometry. The actor must earn a preference for variance through the same potential's action curvature; this directly addresses the mixed-query compromise rather than penalizing its scalar errors. Unlike heatbelief/v21, no positive-atom or fixed quadratic/hinge bottleneck; unlike v16, no eight-action integration; unlike paired v25, retain v27's replay/cadence/one-step targets. The boundary/PDE identifies a Gaussian potential only under the appropriate regularity/growth assumptions; approximate training does not guarantee it.

**Cheapest decisive experiment.** Under original v27 cadence, compare removing the behavior channel alone against removing it plus heat consistency; retain an unmodified reference. Verify finite-difference total derivatives and held-out first/second noise moments, then compare 300k/1M/2M evaluation. Perturb scale in log coordinates; near action boundaries evaluate the same clipped-action convolution extension rather than clamping mean perturbations and mislabelling the derivative. This control is necessary because v27 may benefit from its inconsistent regularization. Do not bundle fresh-only replay or lambda returns.

**Failure signature.** Better PDE residual but weaker held-out action credit; excessive curvature suppression, saturation near sigma bounds, loss of early return, or critic throughput cost without later gain. Squared stochastic trace estimates introduce additional curvature-variance regularization; use coordinate residuals rather than calling that estimator unbiased.

**Hooks.** `PreActionLatentTwinCritic.forward`, `_critic_loss_fn`, `_make_step` and diagnostics; `_actor_loss_fn` retains total SALE differentiation. Preserve physical point supervision, target twin-min and checkpoint selection. Change the critic channel structure explicitly rather than silently changing the meaning of behavior queries.

### 3 — Reward-relevant differential SALE, keeping the flexible critic

**Mechanism.** Add an auxiliary predictor to the existing transition representation: use `zsa(zs,a)−zsa(zs,mu_behavior)` plus a learned state/belief intercept to predict the **centered Bellman innovation**, with an epsilon-independent baseline and frozen continuation. Do not force zero prediction at zero displacement: point value and Gaussian mean differ under curvature. Ground local latent displacement in reward-relevant consequences, not only next latent coordinates. Fit factual noisy-action labels and audit their first Hermite covariance on separate episodes; these are not measured same-state counterfactual contrasts. Optionally use a short observed multi-step auxiliary target, not imagined rollouts. Retain the main critic's direct action/belief routes and all three v27 losses.

**Why it could beat v27.** The actor's useful route is `J_zsa^T Q_zsa`; latent prediction MSE need not preserve its value-sensitive directions. Training contrast covariance can strengthen that route without consequence-Q's exclusive learned-model bottleneck or residual bypass. The substantive novelty is differential supervision of the representation used by belief credit, not merely adding a reward head.

**Cheapest decisive experiment.** One new version differing only in this auxiliary loss, with identical targets/cadence; compare held-out directional innovation fit and eval at 300k/1M/2M. First isolate the existing route with a gradient-only zsa detach control; do not infer its causal contribution from heatbelief.

**Failure signature.** Lower auxiliary error without improved held-out action-direction prediction; collapsed latent rank, inflated Jacobian norms, or conflict with SALE transition learning. Separate critic and representation improvement claims.

**Hooks.** `Encoder`, `_encoder_loss_fn`, `_make_step`, and corresponding hard-copy/host-mirror structures if encoder parameters change. `_critic_loss_fn` and `_actor_loss_fn` retain their original semantics. Keep the auxiliary predictor off the rollout path.

### 4 — Introspection controls exploration covariance, not the exploitation objective

**Mechanism.** Allocate a fixed exploration budget toward directions with unresolved but learnable directional Bellman innovations. Begin with diagonal redistribution within v27's existing Gaussian; move to low-rank correlated covariance only after the diagonal controller works. Use cross-fitted uncertainty reduction and independently estimated local value cost, not absolute TD error or return surprise alone. Keep the raw/checkpoint actor objective and twin-min target unchanged in the first test.

**Why it could beat v27.** Current geometric entropy control allows arithmetic sigma and clipping cost to vary widely. Locally, exploration changes expected value by approximately `.5*tr(H*C)`; useful allocation trades this cost against directional information gain. This gives introspection a measurable collection role without an exploitable `grad b`. Full covariance could discover coordinated joint directions that diagonal noise misses, but would require correct cross-Hermite terms and action-law accounting.

**Cheapest decisive experiment.** Match the empirical exploration-cost/budget of v27, redistribute only diagonal variance from detached auditor statistics, and check held-out derivative SNR plus live/checkpoint returns. Log saturation and distinguish geometric budget from realized clipped covariance. No claim of a gain from merely lowering noise.

**Failure signature.** Repeatedly sampling inherently noisy/model-biased directions with no learning progress; improved live returns solely from reduced total noise; deterministic eval stagnates. If labels are not reliable enough for prescription 1, they are not an exploration oracle either.

**Hooks.** Actor sigma output/collection block, `stage`, `_entropy_fn` and entropy accounting. Correlated extensions also change `_critic_loss_fn` tangents and replay noise metadata; they are a new algorithm, not an extra hidden belief coordinate.

### 5 — Conditional salvage of deliberation: paired one-step criticism, then amortize

**Mechanism.** Compare center and 2–4 antithetic local candidates using paired model/critic differences, score the **executed belief**, and require independently validated advantage. Common-mode offsets cancel. Keep planning out of every environment step: distill validated improvements at update time into the actor. A model remains an auxiliary critic of proposed changes, not the sole value path. If repeatedly collecting with an improved planner, train/evaluate its actual continuation explicitly instead of the current three-policy mixture.

**Why it could beat v27.** Reduce selection multiplicity and differential error, preserve useful local improvement and avoid three-step extrapolation. Approximate missed first-action noise cost is `.5*tr(H*C)`; shared execution-noise draws or validated covariance correction can account for it. This is lower priority until rankings are measured.

**Cheapest decisive experiment.** Future frozen-Mind audit using exact restored simulator states, paired candidate/center first beliefs, common random numbers and one fixed continuation. Measure realized discounted **differences**, sign accuracy and selected-score bias across independent episode blocks. Do not use the same terminal critic as the verdict; bound any omitted tail. Existing logs cannot perform this test because simulator states/branch returns are absent. Reject planning if real differences do not support the reported gain; only then fund a full training version.

**Failure signature.** Predicted 4–7-unit gain vanishes under paired returns; rankings work but execution noise erases gain; or distillation moves beyond validated local support. Twin agreement alone does not constitute independent validation.

**Hooks.** In the deliberate fork: `Mind.plan`, `Mind.consequence`, `Mind.belief_value`, `_make_planner`, `_model_loss_fn`, `_actor_loss_fn`, evaluation and collection. Starting from v27 requires adding the auxiliary model/distillation machinery; no such functions exist in v27 itself.

## 4. Literature: mechanisms to compose, not benchmark promises

Primary sources were checked. None establishes a comparable seed-1 HalfCheetah-v4 gain over this reference.

| Source | Relevant mechanism and judgment |
|---|---|
| [BRO, 2024](https://arxiv.org/abs/2405.16158) | Regularized residual critics and separate optimistic collection support separating exploration from exploitation. Counterexample to “optimism always loses”; not permission to replace twin-min blindly. |
| [SimBa, 2024 / ICLR 2025](https://arxiv.org/abs/2410.09754), [SimBaV2, 2025](https://arxiv.org/abs/2502.15280) | Residual/per-sample-normalized or hyperspherical geometry can condition value gradients. Useful controlled architectural baselines; a port is not the novel contribution. Audit Jacobians, rank and action sensitivity before scaling. |
| [CrossQ, ICLR 2024](https://arxiv.org/abs/1902.05605) | Wide critics, batch normalization, no target networks. Original arXiv submission was 2019. Batch-dependent Q breaks v27's per-row manual JVP assumptions unless cross-sample statistic derivatives are implemented; poor direct transplant. |
| [TD-MPC2, 2023/2024](https://arxiv.org/abs/2310.16828) | Grounded latent representations, SimNorm and policy-prior planning; supports representation quality, not arbitrary hard-max three-step planning. |
| [BBF, 2023](https://arxiv.org/abs/2305.19452) | Predictive representation learning, plasticity and regularized scaling; Atari evidence, not MuJoCo proof. Uncontrolled resets could destroy SALE/target/checkpoint consistency. |
| [HL-Gauss / Stop Regressing, 2024](https://arxiv.org/abs/2403.03950) | Classification of noisy scalar targets may improve conditioning. Synthetic histogram spread is not epistemic uncertainty. If added, keep mean/tangent semantics consistent; support saturation can kill actor gradients. |
| [MR.Q, 2025](https://arxiv.org/abs/2501.16142), [representation-scaling follow-up, 2026](https://arxiv.org/abs/2606.05555) | Predictive representations without planning are the strongest conceptual fit for prescription 3. SALE already predicts latents: novelty must improve reward-relevant differential geometry, not add another generic model. Multitask scaling results do not diagnose this plateau. |

A modest residual/LayerNorm critic with exact JVPs or an auxiliary HL-Gauss head is a conditioning control, not the central research proposal. Do not bundle it with the first introspection test.

## 5. Stop doing

- Treating scalar MAE, within-episode correlation, or positive plan gain as evidence of useful actor derivatives; treating bias sign as a universal policy-quality law.
- Differentiating freely fitted MC error models at unsupported belief-center queries. A good state offset is not a good action penalty.
- Calling v27 an exact belief expectation, calling every parameter-conditioned critic unidentifiable, or assuming all hidden belief coordinates have zero gradient.
- Repeating existing physical-integral/finite-basis fixes as new ideas: action-integral v16, analytic-moments v21, and paired/center v22–26 already explore related remedies, with cadence/target changes that complicate attribution. Exact algebra is not the missing success criterion.
- Optimizing the actor, collecting data and planning under three different policies without naming the resulting target mismatch.
- Comparing raw loss scales across different objectives, optimizer steps to environment steps, cumulative SPS to recent interval cost, or checkpoint returns to live stochastic returns as though they were the same measurement.

## Evidence provenance

All directories are under `runs/`, named `HalfCheetah-v4__<experiment>__1__<timestamp>`. Main timestamp suffixes: v27 `1792812977`; selfmodel, deliberate and combined `1792870543`; consequence v1 `1792812977`, costate v2 `1792815322`, closed v3 `1792816860`, residual v4 `1792818830`; heatbelief v1 `1792824261`, chart `1792824666`, dictionary `1792868720`; original `td7_v1` `1783448814`; `td7_fast_sdnoise_v1` `1790302633`; preaction v5 `1790309009`; preaction Kalman `1792607086`. Exact eval points, named fixed windows and episode-level aggregates above are deliberately distinguished. Running-job data is frozen at the recorded endpoint, not a claim about its eventual result.
