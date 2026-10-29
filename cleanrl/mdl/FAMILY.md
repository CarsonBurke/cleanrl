# MDL weight-posterior line

Does charging the description length of the weights (beta * KL(q||p) / N per rollout) stop PPO's K-epoch
fitting from memorising the rollout, so that more epochs stop hurting? Base: the sepclip PPO control
(`awr/ppo_continuous_action_ppo_sepclip_control_v1.py`). HalfCheetah-v4, seed 1, 16 envs, 8M steps,
`--no-norm-adv --no-clip-vloss --clip-coef-upper 0.2`. Scores are stochastic training returns, last 10 episodes,
from `scripts/score_runs.py`.

## v1 (`ppo_continuous_action_mdl_posterior_v1.py`)

`--posterior none` reproduced `sepclip_control_8M` exactly (identical returns at every step).

| Run | @2M | @4M | @8M | actor / critic pruned | actor bits/weight | sampled-vs-decoded policy KL |
|---|---:|---:|---:|---:|---:|---:|
| none e10 (control) | 4770 | 6620 | **8413** | - | - | - |
| mdl b0.1 e10 | 4311 | 6086 | 8187 | 0.5% / 0.4% | 5.2 | 0.05 |
| mdl b1 e10 | 4790 | 5643 | 6607 | 18% / 0.6% | 2.1 | 1.33 |
| svd b0.1 e10 | 3741 | 4937 | 5709 | 40% / 22% | 2.3 | 0.04 |
| svd b1 e10 | 3908 | 5233 | 5913 | 87% / 73% | 0.38 | 0.57 |
| none e40 | 3965 | 5325 | 6288 | - | - | - |
| mdl b1 e40 | 345 | 95 | 21 (collapsed) | 47% / 74% | 0.13 | 86 |
| svd b1 e40 | 2007 | 1981 | 1706 | 72% / 91% | 0.11 | 36 |

Second-half means of the critic's pre-update (held-out) and post-update (fit) MSE on each rollout:

| Run | held-out MSE | fit MSE | gap | approx KL | clipfrac | final entropy |
|---|---:|---:|---:|---:|---:|---:|
| none e10 | 0.0052 | 0.0049 | 6% | 0.013 | 0.10 | -4.1 |
| none e40 | 0.0032 | 0.0026 | 18% | 0.023 | 0.22 | -6.4 |
| svd b0.1 e10 | 0.0022 | 0.0020 | 6% | 0.011 | 0.09 | -3.4 |

Findings:

- **The premise does not hold in this PPO.** The critic barely memorises the rollout (train/held-out gap 6% at
  10 epochs). At 40 epochs the gap grows to 18%, but held-out error still *improves* (0.0052 -> 0.0032). The
  25% return loss at 40 epochs is on the actor side: twice the per-update KL and clip fraction, and entropy
  collapsing faster (-6.4 vs -4.1). That is over-stepping the trust region, not memorisation, and a bit cost does
  not target it.
- **The actor ELBO is degenerate under PPO.** The surrogate is not a likelihood. Samples pushed outside the clip
  get zero gradient, so nothing restores weight precision and the KL wins. mdl drives actor sigma to about |mu|
  (median log alpha ~0, SNR ~1). svd shrinks means inside the unit-norm rows and prunes them. The sampled acting
  policies drift from the decoded one (1.3 nats/state at mdl b1). More epochs mean more optimiser steps toward
  that degenerate optimum, so e40 collapses (86 nats). "Epochs do not change the objective" is true, and it
  does not help when the objective's optimum is itself bad.
- **On the critic, which has a real likelihood, MDL behaves as intended.** mdl leaves critic sigma near its
  initial value (the data pays for the precision). svd b0.1 pruned 22% of the critic. *Correction (v2
  analysis):* its 2.4x lower held-out MSE is not a better critic. MSE is in normalised-return units, and its
  scale follows the policy's return variance. Explained variance, which normalises that away, is *lower*
  for svd b0.1 than for the control (0.881 vs 0.906, second-half means). Compare critics by EV, not MSE.
- mdl cannot charge for the means here: nGPT row normalisation fixes sum(mu^2), so its prior term is constant
  and it reduces to learned weight noise.
- Cost: the posterior arms run at roughly half the control's SPS (per-env NumPy actor plus per-segment refreshes).

Next, if pursued: a critic-only posterior (deterministic actor) for svd/mdl at e10 and e40, which is where the
MDL objective is well posed. For the actor, a likelihood-shaped data term would be needed: an AWR/CE policy fit,
or acting with the decoded mean and charging the policy distortion E KL(pi_mu || pi_{mu+eps}) in nats.

## v2 (`ppo_continuous_action_mdl_critic_posterior_v2.py`): posterior on the critic only

The control actor (deterministic, fused host mirror) is unchanged. Critic matrices are q = N(mu, sigma^2); the loss
adds vf_coef * s^2 * KL / N, with s^2 = the rollout's pre-update (held-out) MSE, i.e. the critic's per-transition
negative ELBO in the control's value-loss units. GAE values come from the decoded critic. `--posterior none`
reproduced v1 none exactly.

| Run | @2M | @4M | @8M (±CI95) |
|---|---:|---:|---:|
| none e10 (control) | 4770 | 6620 | 8413 (±149) |
| mdl b1 e10 | **5145** | **6986** | **8565** (±69) |
| svd b0.1 e10 | 4349 | 6260 | 8365 (±104) |
| svd b1 e10 | 4052 | 5308 | 6636 (±80) |
| none e40 (v1) | 3965 | 5325 | 6288 (±58) |
| mdl b1 e40 | 4932 | 6194 | **7468** (±531) |
| svd b0.1 e40 | 4323 | 5191 | 5900 (±56) |
| svd b1 e40 | 4168 | 5132 | 5837 (±47) |

Second-half means (EV = explained variance of the pre-update critic on the new rollout; the bits and pruned
columns are final values):

| Run | EV | held-out / fit MSE (gap) | approx KL | clipfrac | final entropy | critic bits/w | pruned | median log alpha |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| none e10 | 0.906 | 0.0052 / 0.0049 (6%) | 0.013 | 0.10 | -4.1 | - | - | - |
| mdl b1 e10 | **0.947** | 0.0065 / 0.0061 (6%) | 0.014 | 0.10 | -3.9 | 5.4 | 0.6% | -6.4 |
| svd b0.1 e10 | 0.910 | 0.0050 / 0.0047 (6%) | 0.013 | 0.09 | -4.2 | 4.4 | 15% | -6.7 |
| svd b1 e10 | 0.887 | 0.0027 / 0.0026 (4%) | 0.012 | 0.09 | -4.0 | 2.3 | 64% | 5.5 |
| none e40 | 0.929 | 0.0032 / 0.0026 (18%) | 0.023 | 0.22 | -6.4 | - | - | - |
| mdl b1 e40 | 0.927 | 0.0079 / 0.0064 (18%) | 0.033 | 0.25 | -6.4 | 4.3 | 31% | 0.6 |
| svd b0.1 e40 | 0.902 | 0.0027 / 0.0021 (21%) | 0.032 | 0.23 | -6.6 | 2.4 | 39% | 1.9 |
| svd b1 e40 | 0.900 | 0.0023 / 0.0019 (20%) | 0.030 | 0.25 | -6.4 | 0.6 | 86% | 6.5 |

Findings:

- **mdl on the critic is the first posterior arm at or above the control.** At e10 it led at every checkpoint
  (+8% @2M, +6% @4M, +2% @8M; +4% averaged over the run) and had the best critic by EV (0.947 vs 0.906).
  One seed: the 8M gap (+152) is inside the control's CI95.
  *Correction (diagnostics pass):* sigma "barely moved" because it never equilibrated, not because the
  data pays for the precision. log sigma^2 rises monotonically in every mdl run, at a rate proportional
  to beta (b0.1 drifts 10x slower than b1), and never plateaus. At e40, with 4x the optimiser steps, the
  same objective reaches median log alpha +0.5 and 30% "pruned" and is still accelerating. The e10
  posterior is therefore the -12 initialisation plus drift: the arm is "control + ~4% multiplicative LRT
  weight noise" (training-time value loss 4% above the decoded fit), not a converged description length,
  and its 5.4 bits/weight is an init artefact. The EV gain is not a cleaner critic either: it comes from
  the batches, not the critic. See v3 below.
- **e40: mdl recovers part of the epoch loss (7468 vs 6288, +19%), but not through the stated mechanism.**
  Its actor over-steps exactly like the control's (approx KL 0.033, clipfrac 0.25, entropy -6.4). Its critic
  EV equals the control's (0.927 vs 0.929), and its train/held-out gap is the same 18%. Late returns are
  volatile (CI95 ±531), so this is the least certain result here. A seed would settle it. The whole gap
  opens between 1M and 2M steps (1.5M: 3739 vs 2990) and then holds at about +1000 without growing,
  which is the signature of early trajectory divergence rather than a mechanism that keeps paying.
- **svd hurts at every setting.** Pruning concentrates the unit-norm rows. At b1 it prunes 64-86% of the
  critic, loses EV, and loses 21% of return at e10. At e40 it is 6-7% below the e40 control.
- **The memorisation premise remains unsupported.** At 40 epochs every arm, with or without a bit cost, shows
  the same ~18-21% critic train/held-out gap. The return loss tracks actor over-stepping, which a critic
  description length does not touch.
- Cost: the critic-only arms keep the control's fused rollout. Their median update is 23 ms per iteration,
  against the control's 18 ms (LRT forward plus KL). Env time is about 150 ms per iteration, so the cost is
  negligible.

## v3 (`ppo_continuous_action_mdl_critic_diagnostics_v3.py`): v2 plus instruments

The v3 changes are diagnostics only:
- the critic error is decomposed into bias and residual variance, alongside the return mean and variance;
- Monte-Carlo LRT draws measure the output noise and the Jensen gap (noisy critic vs the decoded mean);
- the exact Gauss-Newton diagonal F gives the mdl sigma fixed point, 1/sigma*^2 = 1/s_p^2 + vf_coef F / kl_weight,
  checked against an unbiased Monte-Carlo pull;
- Adam's second moment is logged for the sigma group;
- checkpoints are saved at 25/50/75/100%, with optimiser, normaliser and rollout.

Every reproduction matched its v2/v1 run to the episode (none e10 8413, mdl b1 e10 8565, none e40 6288,
mdl b1 e40 7468). New arm: mdl kl_scale 0 (LRT weight noise, no description cost) e10: **7636** (±192).

Medians (75-100% of training, unless stated otherwise):

| e10 run | held-out bias | held-out residual var | batch return var | output noise std | Jensen gap |
|---|---:|---:|---:|---:|---:|
| none | -0.0018 | 0.0041 | 0.044 | - | - |
| mdl b1 | -0.0001 | 0.0058 | 0.178 | 0.014 | -0.0004 |
| mdl b0 (noise only) | -0.0019 | 0.0040 | 0.034 | 0.011 | -0.0002 |

Findings:

- **The EV gain is the batches, not the critic.** The critic bias is negligible in every run (|bias| < 0.002
  vs residual std ~0.07), so MSE is residual variance. The mdl b1 runs (e10 and e40) simply collected
  rollouts with 4-5x the within-batch return variance (second-half median 0.21 vs 0.044). Per iteration,
  EV correlates with log return variance at +0.77 to +0.90 in every run, and the mdl b1 critic's absolute
  held-out error is *higher* than the control's (0.0065 vs 0.0052 second-half mean). Do not compare critics
  across runs by EV when their batch return variance differs; compare residual variance on matched data.
- **Decoding with the means is sound.** The Jensen gap between the noisy critic the loss trains and the
  decoded mean critic is ~4e-4 at e10 (2e-3 at e40). The output noise std is ~0.014 (~3% of the MSE).
- **mdl's sigma is pinned by Adam's eps, far from its objective.** The Gauss-Newton fixed point is confirmed
  (the Monte-Carlo pull equals the GN pull to 0.99-1.02). Its target is log sigma^2 ~ -6.2 (median log
  alpha ~ -1, i.e. ~60% weight noise). The run sits 5.3 nats below that: the KL's push is ~200x the
  data's pull (median log pull/push -5.4), and >99% of weights want more noise. Yet sigma barely moves:
  - the sigma group's sqrt(v) is 3e-7 to 2e-6, far below Adam's eps = 1e-5;
  - so Adam takes raw-gradient SGD steps of lr * g / eps ~ 1e-4 per step;
  - over ~4900 annealed steps that gives about the 0.3-0.5 nat drift observed.

  That also explains why the drift is proportional to beta. The e10 mdl arm never optimised its stated
  objective. It is the -12 init plus eps-limited drift.
- **The true critic-MDL optimum is heavy noise that turns into pruning.** At e40, as sigma rises, the
  Fisher of the noisier weights collapses (median F 1.4e-4 -> 1.7e-6), and the target log alpha climbs to
  +5.8, self-reinforcing toward pruning. The objective, if reached, would strip most of the critic (the
  same direction svd took, which lost return).
- **Noise floor.** mdl b1 and mdl b0 at e10 are, in effect, the same algorithm: identical init, log sigma^2
  -11.45 vs -12.01, output noise 0.014 vs 0.011. Their 8M returns are 8565 and 7636, and the control is 8413.
  Single-seed 8M differences of ~900 are therefore within trajectory noise. That covers mdl e10 vs the
  control (+152), and is a caution on mdl e40 vs none e40 (+1180, which opens at 1-2M and then holds).
  **No MDL arm has a demonstrated return effect.**

Next, if pursued: optimise sigma properly before judging MDL. Options are eps << sqrt(v) for the sigma group
(eps 1e-12, or a gradient rescaled by N), or a closed-form per-rollout sigma from F, since the fixed point
is cheap and exact. Then compare at matched return variance, over 2-3 seeds, against the ~900 noise floor.

## v4 (`ppo_continuous_action_mdl_critic_natgrad_v4.py`): sigma by natural gradient

v4 drops Adam on log sigma^2. Before each rollout's update, a natural-gradient VI step (Bayesian learning
rule, mean-field Gauss-Newton, rho = 1) sets the precision to 1/s_p^2 + N F / (kl_scale s^2). The KL term
is gone from the means' loss: under unit-norm rows it is constant in the means.

| Run | @2M | @4M | @8M (±CI95) | log sigma^2 mean (2nd half) | prunable (log alpha > 3) | output noise std | EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| none e10 (control) | 4770 | 6620 | 8413 (±149) | - | - | - | 0.90 |
| mdl b0.1 e10 | 4502 | 6221 | 7856 (±431) | -4.3 | 32% | 0.035 | 0.92 |
| mdl b1 e10 | -298 | -277 | **-131** (never learns) | -0.5 | 64% | 12 | 0.00 |
| none e40 | 3965 | 5325 | 6288 (±58) | - | - | - | 0.92 |
| mdl b0.1 e40 | -147 | 764 | **3311** | -2.7 | 58% | 0.058 | 0.89 |
| mdl b1 e40 | -255 | -292 | **-306** | 0.0 | 69% | 8.4 | 0.00 |

Findings:

- **The optimisation problem is solved.** After each step, sigma sits on its target (the median log sigma^2
  gap after the update is 0.003-0.006 nats).
- **The objective, at the noise it selects, destroys the critic.** At beta = 1, sigma reaches ~0.8, against
  means of ~0.14 (64% of weights prunable). The output noise std is 4-12 on returns whose variance is ~0.2.
  The critic's EV falls to 0, so the advantages are noise. The actor then stops moving (approx KL ~1e-3,
  entropy stays at its maximum), and the policy never learns. At beta = 0.1 the e10 run is within the noise
  floor of the control (7856 vs 8413; ~900 single-seed spread). The e40 run loses half its return: its
  actor destabilises (approx KL ~0.1, entropy rising back above zero).
- **The Gauss-Newton approximation that sets sigma breaks down at those noise levels.** The unbiased
  Monte-Carlo pull of the data on log sigma^2 is 3-16x the Gauss-Newton pull at beta = 0.1, and up to
  ~1000x at beta = 1. In v3, near the init, it was 1.00. The quadratic model at the means misses how
  badly noise through the normalisations and SiTU gates hurts the loss. So the precision is
  underestimated, noise is over-admitted, and the means then train under that noise. That is a runaway.
  v4 therefore optimises a linearised ELBO, not the true one; the true optimum has less noise. A faithful
  version would replace F with an expected-curvature estimate under q (IVON's reparameterised Hessian
  estimator, g (w - mu) / sigma^2).
- **Conclusion at v4 (superseded by v5 and v6 below).** At this point no description-length arm had beaten
  the control beyond single-seed noise. Offline diagnosis of v4's checkpoints then showed that v4's failure
  was the curvature point and the prior, not MDL itself:
  - **Curvature point.** GN at the means misses gates that noise unsaturates.
  - **Prior.** A per-rollout zero-mean prior charges every rollout for the whole critic.

## v5 (`ppo_continuous_action_mdl_critic_online_v5.py`): VOGN precisions, sequential prior

Mental model from v4's checkpoints (scratchpad analysis, offline):

- **Per-layer GN at the means is accurate where gates are linear, and fails where noise changes the
  operating point.** At b1, gate1/up1 sat at log F ~ -14 (saturated) and were given prior-scale noise. That
  unsaturated the gates, which made the noise 50-10^5x what GN predicted.
- **Which estimator matches the exact derivative.** The exact quantity is the Monte-Carlo
  d E_q[1/2 (v - R)^2] / d sigma^2 per layer. Two GN-under-q estimators were compared against it:
  - GN at whole-batch *weight samples* (VOGN; backprop through sampled weights) matched it to 0.4-1.3x at
    b0.1;
  - LRT forwards with the noise scale detached backprop through the means, and gave 0.03-0.07x on the
    second-stage gate/up.
- **A stream needs a sequential code.** The MDL of a stream charges each rollout only what its update
  changes. The previous posterior, tempered by forgetting gamma and mixed with the base prior, is the prior:
  Lambda = gamma lambda + (1 - gamma)/s_p^2, m = gamma lambda mu / Lambda.
  - The KL's mean term, kl_weight/2 sum Lambda (mu - m)^2 ~ vf/2 gamma/(1 - gamma) sum F dmu^2, is a
    Fisher-weighted anchor.
  - sigma is refit at the updated means after each update, so the next prior is self-consistent.
  - Each rollout's first precision iteration is a full step; the first rollout gets 4x iterations (cold
    start from the init contracts at ~0.6x per step).

| Run | @2M | @4M | @8M (±CI95) | entropy (Q4) | held-out bias (Q4) | Jensen gap (Q4) | noise std |
|---|---:|---:|---:|---:|---:|---:|---:|
| none e10 (control) | 4770 | 6620 | 8413 (±149) | -4.0 | -0.001 | - | - |
| emp b0.1 e10 | 4534 | 6466 | **8547** (±258) | -3.8 | +0.008 | -0.009 | 0.014 |
| emp b1 e10 | 3576 | 3741 | 4830 (±42) | -0.3 | +0.110 | -0.114 | 0.045 |
| seq g0.5 b1 e10 | 3912 | 5102 | 6156 (±96) | -2.8 | +0.026 | -0.028 | 0.022 |
| seq g0.9 b1 e10 | 4216 | 5180 | 6137 (±62) | -3.5 | +0.003 | -0.004 | 0.011 |
| none e40 (control) | 3965 | 5325 | 6288 (±58) | -6.2 | -0.001 | - | - |
| emp b1 e40 | 982 | -211 | 2192 (±84) | - | - | -0.11 | 0.065 |
| seq g0.9 b1 e40 | 4649 | 6261 | **7804** (±145) | -3.7 | +0.021 | -0.021 | 0.021 |

Findings:

- **The posterior is now what it claims to be.** In every run, the Monte-Carlo/GN pull ratio is 0.93-1.07
  after the first quarter (v4: 3-1000x), and the update leaves log sigma^2 within 0.01 nats of its fixed
  point. No run collapses: b1 went from -131 (v4) to 4830 (empirical) and 6100 (sequential).
- **The remaining loss is a value offset, not critic noise.** The means are fit so that the *noisy* critic
  matches the returns (E_q[v] ~ R), but GAE used v(mu). Held-out bias = -Jensen gap in every run, to two
  digits.
  - Without advantage normalisation, an offset b shifts all advantages by ~-(1 - gamma) b / (1 - gamma
    lambda) = -0.17 b. That lowers every sampled action's probability, so entropy stays high and the
    policy sharpens slowly.
  - Return order follows the offset: emp b1 (+0.11, entropy -0.3) < seq g0.5 (+0.026) < control.
  - Exception: seq g0.9 e10 has negligible offset and still trails by ~2300, which is beyond the ~900
    noise floor but is one seed.
- **40 epochs: the sequential prior keeps +1500 over the control (7804 vs 6288).** v3's Adam-sigma mdl b1
  e40 also beat its control (7468), so robustness to many epochs recurs across critic posteriors. Part of
  v5's gain may come from the offset's entropy effect (entropy -3.7 vs the control's -6.2), which slows the
  over-stepping actor's collapse.
- **Next (v6):** values = the posterior predictive mean, 1/K sum_k v(x; mu + sigma eps_k), with K = 16
  weight samples drawn once per rollout. That is the prediction the ELBO trains, and it removes the offset
  by construction.

## v6 (`ppo_continuous_action_mdl_critic_predictive_v6.py`): values = posterior predictive mean

GAE, held-out MSE and truncation bootstraps use 1/K sum_k v(x; mu + sigma eps_k), with K = 16 weight samples
drawn once per rollout (shared by every state, so their Monte-Carlo error largely cancels in TD differences).
Everything else is v5. `critic/heldout_at_means_bias` keeps v5's value for comparison.

| Run | @2M | @4M | @8M (±CI95) | v5 @8M | held-out bias (Q4) | at-means bias (Q4) | entropy (Q4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none e10 (control) | 4770 | 6620 | 8413 (±149) | - | -0.001 | - | -4.0 |
| emp b0.1 e10 | 3790 | 4808 | 5611 (±105) | 8547 | -0.002 | +0.003 | -3.4 |
| emp b1 e10 | 4255 | 5965 | **7503** (±67) | 4830 | -0.001 | +0.046 | -4.0 |
| seq g0.5 b1 e10 | 4258 | 6316 | **8405** (±89) | 6156 | -0.001 | +0.027 | -3.7 |
| seq g0.9 b1 e10 | 3998 | 5153 | 6048 (±61) | 6137 | -0.002 | +0.002 | -3.9 |
| none e40 (control) | 3965 | 5325 | 6288 (±58) | - | -0.001 | - | -6.2 |
| emp b1 e40 | 4028 | 6158 | **7946** (±271) | 2192 | -0.000 | +0.061 | -6.3 |
| seq g0.9 b1 e40 | 3760 | 5443 | **7775** (±141) | 7804 | -0.001 | +0.012 | -6.0 |

Findings:

- **The offset mechanism is confirmed.**
  - Held-out bias is back at the control's -0.001 in every run, while the critic at its means still
    over-predicts by the same +0.01-0.06.
  - Entropy now tracks the control (b1 e10 -4.0 vs -4.0; v5 had -0.3).
  - The runs that carried a large v5 offset recovered: emp b1 e10 4830 -> 7503, seq g0.5 6156 -> 8405
    (= control), emp b1 e40 2192 -> 7946.
  - Runs whose v5 offset was negligible (seq g0.9, b0.1) are algorithmically near-identical to v5.
- **The single-seed noise floor at e10 is ~3000, not ~900.** emp b0.1 differs between v5 and v6 only by a
  +0.008 value offset, and scored 8547 vs 5611. With that spread, no e10 arm (seq g0.9 ~6100 twice
  included) is distinguishable from the control.
- **The 40-epoch gain is the consistent signal, and it is not an entropy artefact.** Every critic-posterior
  e40 run beats the single e40 control (6288):
  - v3 mdl b1: 7468;
  - v5 seq g0.9: 7804;
  - v6 emp b1: 7946;
  - v6 seq g0.9: 7775.

  v6's e40 entropy matches the control's (-6.0 to -6.3), so v5's offset was not the cause. Its mechanism is
  not yet identified:
  - the posterior runs' approx KL is equal or higher;
  - their held-out critic residual is *higher* (0.004-0.006 vs 0.003).

  Caveat: the e40 control is one trajectory (v1-v3 reproduce it bit-exactly, so they are not extra seeds).
- **Next:**
  - A second-seed e40 control is the cheapest way to tell whether the +1500 is real.
  - If it holds, identify the channel. Candidates:
    - noise-regularised critic trunk features;
    - advantage variance;
    - per-state value error correlated with actions.

## v7 (`ppo_continuous_action_mdl_energy_actor_v7.py`): sequential Laplace code on the actor

The "infinite epochs with an energy cost" idea applied to the actor. Train toward convergence each rollout,
and let a description-length cost for moving the policy make extra epochs harmless. The actor pays
c std(A)/N * 1/2 sum Lambda (mu - m)^2 per update, with:

- Lambda = gamma lambda_prev + (1 - gamma)/s_p^2, and lambda = Lambda + N F_pi after each update;
- F_pi the exact diagonal Beta policy Fisher (Cholesky random directions with trigamma);
- m = the previous posterior mean.

The critic is v6 emp b1 throughout. 8M steps, last 10 episodes:

| run | 8M return | Q4 entropy |
|---|---:|---:|
| control e10 / e40 | 8413 / 6288 | -4.0 / -6.2 |
| c0.1 e10 | 6137 | - |
| c0.03 e40 | 6712 | - |
| c0.1 e40 | 6043 | - |
| c0.3 e40 | 6274 | - |
| c0.1 e100 | 5764 | -7.96 |
| c0.1 e40, no clip | 4770 | -9.14 |

Every arm lost to v6 emp b1 at matched epochs (e40: 7946). Without the clip the code alone failed to hold
the policy. Diagnosis:

- **The code misses most of each update.** The Fisher-predicted update KL (1/2 d^T diag(F) d) was 12-50% of
  the exact KL with the clip (0.001-0.006 vs 0.009-0.02 at c0.1 e10), and 3-14% without it (0.08-0.28 vs
  2-3 nats).
- **Free directions.** Unit-norm head rows put the whole Beta concentration scale in readout_gain and the
  head bias (24 parameters). The trunk-level code left them unanchored.
- **Correlated moves.** The diagonal under-counts moves along correlated directions.
- **Effect:** the code slowed the priced directions and diverted the update into the free ones.
  Concentration grew and entropy collapsed with epochs (-8 at e100, -9 without the clip).

Conclusion: a parameter-space code must price every parameter, and their correlations, or it steers the
update rather than bounding it. v8 moves the code into function space.

## v8 (`ppo_continuous_action_mdl_predictive_actor_v8.py`): predictive (function-space) actor code

v7's code charged parameter moves and missed most of each update. v8 instead charges the policy's own
movement, in nats, on states from earlier rollouts:

  tau sum_{k>=1} gamma^k mean_{s in rollout t-k} KL(pi_{t-1}(.|s) || pi_theta(.|s)),  tau = c std(A).

Its second-order expansion is v7's Laplace code, with the same weights, anchor and temperature. The KLs are
exact closed-form Beta KLs, so every parameter and correlation is priced. The replay holds 32 rollouts x
512 subsampled states. Once it is full, the code is about 8.7 c std(A) per nat of mean KL.
torch.distributions.kl_divergence cannot be traced by Dynamo; the first submission (11087-11092) failed at
compile for that reason.

Setup: critic v6 emp b1, gamma 0.9. 8M steps, last 10 episodes.

| run | 8M | @2M | Q4 entropy | Q4 approx KL | update / newest-replay KL |
|---|---:|---:|---:|---:|---:|
| control e10 | 8413 | 4770 | -4.0 | 0.0096 | - |
| control e40 | 6288 | 3965 | -6.2 | 0.017 | - |
| v6 emp b1 e40 | 7946 | 4028 | -6.25 | 0.022 | - |
| c0.1 e10 | 6997 | 4214 | -3.1 | 0.0039 | 1.07 |
| c0.1 e40 | **7781** | 3625 | -4.9 | 0.0080 | 1.34 |
| c0.3 e40 | 6106 | 2683 | -3.1 | 0.0037 | 1.6 |
| c0.3 e100 | 6107 | 2416 | -4.5 | 0.0050 | 1.77 |
| c1 e40 | 4876 | 1444 | -1.1 | 0.0014 | 1.87 |
| c0.3 e40 no clip | 5138 | 2150 | -5.0 | 0.057 | 3.1 |

Findings:

- **v8 beats v7 in every matched arm.** c0.1 e40 scored 7781 vs 6043; without the clip, 5138 vs 4770 (at 3x the c).
- **Epoch invariance holds.**
  - c0.3 scored the same at e100 and e40: 6107 vs 6106.
  - c0.1 e40 moves the policy per update about as much as the e10 control (approx KL 0.020 -> 0.008 vs
    0.026 -> 0.010).
  - Entropy no longer collapses with epochs.
- **The free-direction gap is small.** The update's KL on the current, unpriced states is 1.1-1.9x its KL
  on the newest replayed rollout, and 3x without the clip.
- **The code binds.**
  - At c1 the surrogate and code gradients at the final parameters are opposed (cos -0.4 to -0.64).
  - At c0.1 they are not (cos +0.2 to +0.4). Adam's fixed-size steps leave the update short of the
    penalised optimum.
- **Returns fall monotonically in c.** Read with c -> 0 as v6 e40 (7946): c0.1 7781, c0.3 6106, c1 4876.
  Every arm also lags early (@2M), with Q1 entropy above the control's. The mass-covering direction resists
  narrowing.
- **Conclusion.** The code does what it was built for: epochs stop mattering and entropy holds. But limiting
  per-update movement did not raise return at any price tested.
  - c0.1 e40 (7781) and the e10 control (8413) are within the ~3000 single-seed spread.
  - The 40-epoch control's deficit is repaired by the critic posterior (v6 e40 7946), not by an actor code.
  - Over-stepping was not the binding constraint at 40 epochs; the price of a nat is still uncalibrated.

## v9 (`ppo_continuous_action_mdl_variational_actor_v9.py`): variational actor update, dual temperature

v8 left two parts of its code short of the Bayesian update it stood in for: the KL direction and a hand-set
price. v9 makes each actor update the variational posterior under the previous policy as prior:

  max  E_s [ E_pi A - eta KL(pi(.|s) || pi_prev(.|s)) ]

- The surrogate is the clipped ratio, on this rollout's states.
- The KL is the exact reverse Beta KL. v8's replay is gone.
- eta is the REPS dual temperature: KL(softmax(A/eta) || uniform) = epsilon over the batch, found by on-device
  bisection.

With one action per state, that joint KL also counts the advantage variance that actions do not explain. So
epsilon prices nats rather than budgeting the per-state move.

Setup: critic v6 emp b1. 8M steps, last 10 episodes.

| run | 8M | @2M | Q4 entropy | Q4 update KL (rev / fwd) | eta/std(A) Q1->Q4 | dual ESS Q4 |
|---|---:|---:|---:|---:|---:|---:|
| control e10 | 8413 | 4770 | -4.0 | - | - | - |
| v8 c0.1 e40 (0.87 std) | 7781 | 3625 | -4.9 | 0.0080 (fwd) | 0.87 | - |
| eps0.3 e10 | 6492 | 3264 | -2.6 | 0.0026 / 0.0026 | 1.4 -> 1.8 | 0.14 |
| eps0.1 e40 | 6010 | 2681 | -3.5 | 0.0032 / 0.0032 | 2.2 -> 2.5 | 0.57 |
| eps0.3 e40 | 5905 | 3122 | -3.4 | 0.0039 / 0.0039 | 1.4 -> 2.1 | 0.10 |
| eps1 e40 | **6696** | 3562 | -4.3 | 0.0038 / 0.0038 | 0.87 -> 1.7 | 0.008 |
| eps0.3 e100 | 5563 | 2996 | -4.8 | 0.0055 / 0.0055 | 1.4 -> 1.9 | 0.14 |
| eps0.3 e40 no clip | 5400 | 3335 | -5.5 | 0.021 / 0.021 | 1.4 -> 2.0 | 0.12 |

Findings:

- **Direction is irrelevant here.** Reverse and forward update KLs agree within 2% in every run. At
  0.003-0.02 nats per update both are the same quadratic form, so v8's "mass-covering resists narrowing"
  explanation for its early lag is refuted.
- **The dual's price is high, and it rises through training.** Heavy-tailed advantages give eta =
  1.4-2.5 std(A), above v8's best 0.87, and it climbs as advantages grow heavier-tailed. At this price the
  realised per-state move is about 1% of epsilon.
- **Returns again rank by price, cheapest first.** eps1 > eps0.1/0.3, matching v8's monotone c curve.
  eps1 reaches v8's price only by collapsing the tilt to an ESS below 1%: the regime where calling it a
  nonparametric posterior is nominal.
- **Epochs still cost return, less than for the control.** At eps0.3: e10 6492, e40 5905, e100 5563.
- **Without the clip, entropy collapses** (-5.5) at the same eta. The prior term alone does not stop
  concentration growth.

Conclusion across v7-v9 (Laplace, function-space, variational): pricing the actor's movement does what it
says. Per-update KL is bounded, and entropy holds or degrades slowly. But return is monotone decreasing in
the price at every form and calibration tried. The best actor-coded run (v8 c0.1 e40, 7781) does not beat
having no actor code (v6 e40, 7946) or the e10 control (8413). On HalfCheetah with this critic, the
40-epoch deficit is a critic problem (v6 repairs it), not actor over-stepping. A KL cost on the actor, in
any direction or calibration tried, trades learning speed for a stability the run did not need.
