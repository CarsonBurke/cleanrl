# 05 - Literature and reference-implementation comparison (REPPO root cause)

Scope: what the published REPPO actually is (paper + official code), how the local v60/v36 line differs,
what prior art says about learning action gradients from on-policy data, and what follows for a 16-env
HalfCheetah-v4 regime. No training was run for this report.

How to read the evidence labels:
- [code] read directly from the official repo, commit 05a5776 (2026-08-23), https://github.com/cvoelcker/reppo
- [paper] read directly from the arXiv HTML text, v3 and v4 (v4 = ICLR 2026 camera-ready, 10 Apr 2026),
  https://arxiv.org/abs/2507.11019
- [fetched] read from the cited paper's text, downloaded for this report
- [memory] recalled, arXiv ID verified against the arXiv API but the specific claim not re-read
- [local] from this repo's FAMILY.md / OPTIMUM.md / run tables (not re-measured here)
- [inference] my reasoning; not measured by anyone

One correction to the study brief up front: the published REPPO is NOT a "hundreds or thousands of policy
iterations per 1M steps" method. Its default is 1024 envs x 128 steps = 131,072 transitions per
iteration, i.e. 7.6 policy iterations per 1M steps and 381 iterations in its whole 50M budget. The local
line (61 iterations per 1M, 488 in 8M) iterates 8x MORE often per sample than the paper, on 8x LESS
data per iteration.

---

## 1. The published method

### 1.1 Algorithm (paper Alg. 1, sections 3.1-3.3; code src/torchrl/reppo.py, src/jaxrl/reppo.py)

| Item | Published value | Source |
|---|---|---|
| Data per iteration | 1024 envs x 128 steps = 131,072 transitions | [paper] Table 2, [code] mjx_dmc_large_data.yaml |
| Total budget | 50,000,000 steps (ManiSkill also reported at 100M; IsaacLab up to 250M) | [paper] |
| Epochs / minibatches | 8 epochs x 64 minibatches, minibatch 2048 -> 512 critic and 512 actor updates per iteration (1:1, interleaved, critic first) | [paper] Table 2, [code] |
| Repo default (config/reppo.yaml) | 4 epochs x 128 minibatches (minibatch 1024); paper experiments use the overrides above | [code] |
| Critic target | TD-lambda on soft rewards: G_t = r~_t + gamma * [(1-d_t) * (lambda * G_{t+1} + (1-lambda) * V_{t+1})]; at truncation bootstraps from V_{t+1} alone | [paper] Alg. 1, [code] compute_gve |
| V_{t+1} | ONE sample: Q_phi(x_{t+1}, a_{t+1}), a_{t+1} ~ behavior policy, computed during the rollout with the pre-update critic | [paper], [code] collect_fn |
| Soft reward | code: r - gamma * alpha * log pi(a_{t+1} | x_{t+1}) (single sample; paper pseudocode omits the gamma) | [code] |
| Target recomputation | None. "Crucially, the targets are computed on-policy after a new data batch is gathered, and the Q targets are not recomputed before gathering new data." No target network, no twin critics, polyak = 1.0 | [paper] 3.1, [code] |
| lambda, gamma | lambda 0.95; gamma = 1 - 10 / max_env_steps (0.99 for 1000-step episodes; 0.97 for the MJX humanoids) | [paper] Table 2, [code] |
| Critic loss | HL-Gauss cross-entropy + aux_loss_mult * self-prediction MSE, aux_loss_mult = 1.0 | [paper], [code] |
| Auxiliary loss | ONE-step latent self-prediction: |f_p(phi(x_t, a_t)) - phi(x_{t+1}, a_{t+1})|^2, target embedding computed at rollout time with the behavior action and pre-update encoder (stop-gradient by construction) | [paper] eq. 17, [code] |
| Actor loss | per state: if KL < KL_tar: -Q(x, a) + e^alpha * log pi(a|x), a ~ pi_theta reparameterized, ONE sample; else: e^beta * KL. ("clipped" mode; a "full" additive mode is reported as slightly worse) | [paper] eq. 15-16, [code] |
| KL | Monte-Carlo, 16 samples from the OLD policy: mean[log pi_old(a_j) - log pi_new(a_j)], i.e. KL(old || new). The paper's constraint text writes KL(pi_theta || pi_theta'); the estimator and the code default (reverse_kl: false) are old || new | [code], [paper] |
| KL_tar | 0.1 (DMC, ManiSkill); 0.01 for IsaacLab ("needs to be set more tightly ... the same value used for PPO") | [paper] Table 2, E.3 |
| Multipliers | alpha (entropy) and beta (KL) in log space, both init 0.01, updated by the same optimizer at lr 3e-4 every minibatch; entropy target = -0.5 * dim(A) | [code], [paper] |
| Optimizer | paper: "Adam", "weight decay and learning rate schedules ... harmful". Torch code: torch.optim.AdamW with default weight decay (0.01), lr 3e-4, max grad norm 0.5 | [paper], [code] |
| Observation norm | one shared running mean/std (EmpiricalNormalization, eps 1e-2), "important for performance" | [paper], [code] |
| Reward scaling | none for mujoco_playground DMC (reward_scaling 1.0, rewards already in [0, 1]); Brax configs use reward_scaling 0.1 | [code] config/env |

### 1.2 Networks

| Item | Published value | Source |
|---|---|---|
| Critic | input concat(obs, action); encoder = Linear-RMSNorm-SiLU, Linear (512); value head = SiLU, Linear-RMSNorm-SiLU, Linear(151 logits); predictor head same shape -> 512. Width 512 (1024 for MJX humanoids) | [code] torch_models.py |
| Normalization | paper says "layer normalization"; code uses nn.RMSNorm | [code] |
| HL-Gauss | 151 bins; sigma = 0.75 * bin width; targets clipped into [vmin, vmax]; a learned "zero_dist" logit offset scaled by 40.9 | [code] hl_gauss, Critic |
| vmin / vmax | paper rule: r_min / (1 - gamma), r_max / (1 - gamma). Shipped configs: DMC [0, 150], Brax [0, 150] (with reward x 0.1), Brax humanoid [0, 200], MJX humanoid [-10, 10], ManiSkill [-15, 15] | [paper], [code] |
| Resulting resolution | DMC/Brax: bin width 1.0, sigma 0.75, for Q up to ~100 with per-step reward <= ~1. One bin = about one step of reward | [inference from code] |
| Actor | 3 linear layers (2 hidden, 512), RMSNorm + SiLU, outputs mean and log_std; std = exp(log_std) + min_std with min_std 0.0 (no bounds); tanh-squashed Gaussian | [code] |

### 1.3 What the paper reports

- Benchmarks: 23 mujoco_playground DMC tasks, 8 ManiSkill3 tasks, IsaacLab humanoid locomotion (v4
  appendix), Atari-10 (discrete variant). All GPU-parallel. There are NO gym/gymnasium MuJoCo
  HalfCheetah/Hopper/Walker2d results anywhere in the paper or repo (Brax configs exist in the repo but no
  Brax results are published). [paper], [code]
- Headline: about 33% higher normalized return than PPO at similar wall-clock on DMC; roughly on par with
  FastTD3 (10.5M-transition replay) and better than FastTD3 restricted to a 32,768-transition buffer. [paper]
- Estimator ablation (section 4.1): pathwise through Q > score-function with the learned Q > PPO >
  PPO-style clipped GAE objective inside REPPO. Qualitative ordering only; no per-component numbers are
  given in text (figures only). [paper]
- Component ablation (v4 E.1, at 131,072 and at 32,768 transitions per iteration): KL regularization and
  HL-Gauss are "necessary ... independent of the size of the on-policy data"; removing KL is the only
  ablation that falls below PPO. Removing normalization is a minor loss that grows at the smaller data
  size. The auxiliary loss matters more at the smaller data size and is "strongly beneficial in some
  environments, but harmful in others". Again figures only. [paper]
- Data scaling (v4 E.2, the one experiment that speaks to our regime). Four settings with a fixed
  2048 minibatch and 8 epochs: 1024x128 (512 updates), 1024x32 (128), 1024x8 (32), 256x8 (8). The
  paper's table labels these "130k / 65k / 32k / 16k buffer", which does not match the env x step
  products (131,072 / 32,768 / 8,192 / 2,048); I could not resolve which is intended. Finding, verbatim:
  "The mean performance of REPPO drops steeply with decreasing dataset size. PPO on the other hand does
  best in the medium and small dataset regimes." and "Larger datasets allow the trained Q function to
  generalize better." [paper]
- The paper never varies below 256 envs, never reports a critic-accuracy measurement (no action-ranking,
  no gradient-accuracy number), and the repo's scaling scripts vary only rollout length and network
  size at 1024 envs. "Accurate surrogate" is demonstrated only on a 2-D toy function (32 vs 1024 training
  points). [paper], [code]
- Official result CSV for DMC CheetahRun (40 trials): mean about 870 of 1000 by roughly 8M steps and
  about 925-930 from 20M on. The CSV's step axis looks resampled (non-zero return at step 0), so treat
  the 8M figure as approximate. DMC CheetahRun is a different task and reward from gym HalfCheetah. [code]

---

## 2. Item-by-item diff against local v60 (base v36, origin v1)

Local values read from cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py.

| # | Item | Published | Local v60 | Status | Likely consequence |
|---|---|---|---|---|---|
| 1 | Transitions per iteration | 131,072 (1024 x 128) | 16,384 (16 x 1024) | DIFFERENT (8x less) | The paper's own E.2: REPPO "drops steeply" as this shrinks while PPO does not. Biggest regime gap. |
| 2 | Independent trajectories per batch | 1024 | 16 | DIFFERENT | Local test at fixed batch (128 envs x 128 steps) was 0.7x the reference [local], so trajectory count at fixed batch is not the lever. |
| 3 | Policy iterations by 8M | 61 | 488 | DIFFERENT (8x more) | The paper's setting is far less sample-efficient per step; at 8M it has barely started. |
| 4 | KL bound per iteration | 0.1 (0.01 on IsaacLab) | 0.03 | DIFFERENT | Samples per nat of allowed policy movement: paper 1.31M, local 0.55M (2.4x more aggressive). Local forks at 0.01 were neutral, 0.003 worse [local]. |
| 5 | Critic updates per iteration | 512 x 2048 (8 passes) | 128 x 2048 (16 passes) | DIFFERENT | Local fits each sample twice as often. |
| 6 | Actor updates per iteration | 512 (8 passes), 1:1 with critic | 32 (4 passes), 1 per 4 critic steps | DIFFERENT | Local actor takes 16x fewer steps per iteration against a critic that has moved further between them. |
| 7 | Target recomputation | never within an iteration ("crucially") | every 16 critic updates, from a re-frozen copy of the live critic and the LIVE (already updated) actor | DIFFERENT | Local targets are a moving, slightly off-policy fixed-point iteration. Locally this was an improvement (v25) [local]; it contradicts the paper's stated design. |
| 8 | V' estimate | 1 action sample, behavior policy | 4 samples, current policy, expected entropy bonus | DIFFERENT | Lower-variance bootstrap locally. Not a plausible cause of a weak critic. |
| 9 | lambda | 0.95 | 0.8 | DIFFERENT | Less future-action noise, more bootstrap weight. Local sweeps chose 0.8 [local]. |
| 10 | gamma | 0.99 for 1000-step episodes | 0.99 | SAME | - |
| 11 | HL-Gauss bins / sigma ratio | 151 / 0.75 | 151 / 0.75 | SAME | - |
| 12 | Support and reward scale | Brax: reward x 0.1, [0, 150] -> bin 1.0, sigma 0.75 | reward x 0.1, [-50, 250] -> bin 2.0, sigma 1.5; Q about 100-110 | DIFFERENT (2x coarser, about half the support never used) | Absolute smoothing is 2x the reference's for the same value scale. Farebrother et al. report the best sigma is tied to target scale, not bin count. Untested locally as far as FAMILY.md shows. Note the reference is also coarse: about 1 step of reward per bin. |
| 13 | Critic architecture | 512 wide, RMSNorm, SiLU, concat(obs, action) | same | SAME | - |
| 14 | zero_dist offset x 40.9 | yes | yes | SAME | - |
| 15 | Auxiliary loss | 1-step latent self-prediction, weight 1.0, rollout-time targets | 8-step action-conditioned latent rollout + reward head, weights 1.0 / 1.0, targets from refreshed target critic | DIFFERENT | Local is a stronger model-style aux. Paper says aux is environment-dependent and can hurt. |
| 16 | Actor network, std | 2 hidden x 512, unbounded exp std | same | SAME | - |
| 17 | Actor objective, KL gate | clipped mode, per-state gate | same | SAME | - |
| 18 | KL estimate | 16-sample MC, clipped actions | closed-form Gaussian KL(old || new) | DIFFERENT (better) | Exact; removes estimator noise. |
| 19 | Entropy target, multiplier init | -0.5 x dim(A); 0.01 / 0.01 | same ("dual" mode) | SAME | - |
| 20 | Multiplier step size | 512 x 3e-4 = 0.154 log units per iteration | 0.15 per iteration over 32 updates | SAME (by construction) | - |
| 21 | Optimizer | AdamW (default weight decay) in torch code, lr 3e-4, clip 0.5 | AdamW, lr 3e-4, clip 0.5 (duals without decay) | SAME | Reference also clips the duals together with the actor; negligible. |
| 22 | Observation normalization | one shared running mean/std | one independent running mean/std PER ENV (VectorObsNorm; rows normalized with their own env's statistics; evaluation uses env 0's) | DIFFERENT | The same physical state reaches the networks through 16 slightly different affine maps. Probably small after millions of steps, but it is state-level input jitter on a problem whose action signal is about 10% of the state-level value spread. Not measured. |
| 23 | Truncation / rollout end | torch: rows at truncation and the last rollout step are masked from the critic loss (non-partial-reset envs) | all rows used, true final observations | DIFFERENT (better) | - |
| 24 | Benchmark reward structure | DMC rewards in [0, 1]; control dt smaller than gym's (from memory, not checked) | gym HalfCheetah, unbounded reward, dt 0.05 | DIFFERENT | Not an implementation issue. A larger dt gives a larger per-step action effect (Tallec et al.), which would favor the local setting if the dt recollection is right. |
| 25 | Evaluation metric | deterministic (mean-action) evaluation | stochastic training return | DIFFERENT | Local deterministic eval is about 1k higher [local]; keep in mind when comparing numbers. |
| 26 | Exploration-noise scaling across envs with importance-clipped lambda | implemented in the JAX code, disabled by default (noise min = max = 1) | missing | MISSING (dormant upstream) | See idea B2. |

Summary of the diff: the network, loss family, policy class, KL gate and multiplier mechanics are
faithful. Every material difference is in the data regime (rows 1-6) or is a deliberate local change
(rows 7-9, 15). I found no missing component of the published method whose absence would explain a weak
critic. The two cheap, never-tested implementation deviations are rows 12 (support twice as coarse and
half unused) and 22 (per-env observation statistics).

### What the paper implies for our regime

1. The paper's default would be hopeless at an 8M budget: 61 iterations with at most 0.1 nats of KL each.
   It is designed for a 50M-250M budget. Adopting its env count and rollout shape at 8M total steps
   trades away iterations we cannot afford. [inference from paper numbers]
2. The paper's only low-data evidence (E.2) says pathwise-through-Q degrades steeply as data per
   iteration falls and score-function PPO does not. Our 16,384 sits between its two smallest 1024-env
   settings. The stated reason is Q-function generalization. [paper]
3. Normalization and the auxiliary loss gain importance as data shrinks; HL-Gauss and the KL gate matter
   at every size. All four are present locally. [paper]
4. Nothing in the paper or repo addresses low env counts, and nothing measures whether the critic ranks
   actions within a state. We therefore have no evidence that the reference critic is any better than
   ours on the oracle metric (corr 0.11); it may be equally weak and simply averaged over 8x more data
   and many more gradient steps per policy. [inference]

---

## 3. Prior art on the specific failure

The failure: the critic's level is good (corr 0.93 across states) but its within-state action ranking is
near noise (corr 0.11; action gradient 20-37% real) [local].

### 3.1 Why the action signal is small and noisy

| Work | Mechanism relevant to us | Evidence | Transfers to 16-env on-policy? |
|---|---|---|---|
| Baird 1994; Tallec, Blier, Ollivier 2019, "Making Deep Q-learning Methods Robust to Time Discretization" (arXiv 1901.09732) | Theorem: Q(s, a) = V(s) + O(dt). "The effect of individual actions on the Q-function is of order O(dt). This will make ranking of actions difficult, especially with an approximate Q-function." Fix: learn A separately with its own scale (Advantage Updating / DAU). [fetched] | Theory plus low-dimensional control experiments. | The diagnosis matches the local numbers (within-state sd 0.22 vs between-state 2.3). The fix has been tried locally in several forms (v26 dueling, v51 contrast head, v63 orthogonal A) without a durable gain [local]. |
| Tucker et al. 2018, "The Mirage of Action-Dependent Baselines" (arXiv 1802.10031) | Variance decomposition of the policy gradient into Sigma_tau (future trajectory), Sigma_a (the sampled action) and Sigma_s. With discounted returns "Sigma_tau dominates", so even an optimal action-dependent baseline removes little; earlier reported gains came from implementation bias. [fetched] | Measured on HalfCheetah and Humanoid with TRPO. | Directly applicable. It matches the local finding that about 90% of a return-advantage's variance is future-action noise. It also explains why v61/v62 "exact targets" were neutral: see the note below. |
| Bellemare et al. 2016, "Increasing the Action Gap" (arXiv 1512.04860) | Operators that widen the gap between the best and other actions while preserving the greedy policy. [memory] | Atari, discrete actions, greedy policies. | Weak. The operators act on max/argmax backups; an entropy-regularized on-policy evaluation critic has no such backup. They also bias the evaluation the pathwise gradient relies on. |
| Pan, Guertler, Schoelkopf 2022, "Direct Advantage Estimation" (arXiv 2109.06093) | Learn a pi-centered A jointly with V from multi-step returns; subtracting learned advantages of FUTURE actions from the return removes their variance. "Due to the difficulty of enforcing the centering constraint in continuous domains, we currently restrict ourselves to discrete action space domains." [fetched] | Discrete control. | Already approximated locally: v63 (centered quadratic A in behavior-noise coordinates) plus v61 exact targets, which regressed [local]. Untested remainder: DAE's single joint n-step loss. Low prior. |

Note on v61/v62 [inference, worth checking by whoever owns the target analysis]: the exact-target
correction G_t = r_t + gamma [V' + lambda (G_{t+1} - Q'(s_{t+1}, a_{t+1}))] removes future-action noise
only through the critic's own advantage estimate, as a control variate. A control variate with
correlation rho to the truth removes a fraction rho^2 of that variance. With rho about 0.11-0.15 that is
1-2%. So the neutral v62 result does not show that "target variance is not the bottleneck"; it shows that
a critic which cannot rank actions cannot denoise its own targets. The two explanations (representation
vs target noise) remain unseparated by that experiment.

### 3.2 Getting a pathwise gradient: three routes in the literature

Every method I could find that gets a usable dQ/da does it by one of three routes. None does it from
about 16k on-policy transitions per policy.

Route A: off-policy replay with one-step TD on action-diverse data.

| Work | Mechanism | Evidence | Transfers? |
|---|---|---|---|
| TD3 (arXiv 1802.09477), SAC (1801.01290), TD7 (2306.02451), TQC (2005.04269) | One-step target r + gamma Q(s', pi(s')): under deterministic dynamics the action dependence of the label is exact (no future-action noise). Replay holds actions from many past policies, so actions vary widely at a given state. | TQC 12,090 and TD3 9,656 on HalfCheetah-v3 at 1M (SB3 zoo, https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md); TD7 17.1k at 1M in this repo [local]. | Violates "no replay". Local replay attempts on the REPPO critic (v32-v35, v52) all failed, but they kept lambda-returns or V-trace on a tiny window; none reproduced the full recipe [local]. |
| CrossQ (arXiv 1902.05605) | Batch (re)normalization in the critic, no target network, UTD 1. [fetched] | MuJoCo, off-policy. | The normalization idea is already present (RMSNorm, no target net). BatchNorm on 16 autocorrelated trajectories is risky. |
| BRO (2405.16158), SimBa (2410.09754), SimbaV2 (2502.15280) | Larger regularized critics; SimbaV2 adds hyperspherical normalization and a categorical critic with rewards rescaled so targets keep unit variance inside a fixed support. [fetched for SimbaV2; memory for BRO/SimBa] | Off-policy, high UTD. SimbaV2 ablation: distributional critic plus reward scaling "improves performance, especially in environments with high reward variance like MuJoCo". | Architecture and return-scale normalization transfer in principle; all the evidence is with replay. Untested locally. |
| FastTD3 (2505.22642), PQL (2307.12983), FastSAC (2512.01996) | Thousands of envs, batch 32,768, distributional critic. FastTD3: "the randomness from parallel environments increases diversity in the data distribution"; "large-batch updates provide a more stable learning signal for the critic". Larger replay is better. FastSAC adds layer norm and drops clipped double Q. [fetched] | GPU humanoid benchmarks. REPPO's paper shows FastTD3 collapsing on hard tasks when the buffer is cut to 32,768. | These are the evidence that small data starves pathwise critics, not a fix for it. |
| AVG (arXiv 2411.15370), Stream-RAC in "Streaming Deep RL Finally Works" (2410.14606) | The closest prior art to "pathwise actor, no replay, tiny data": batch size 1, no target network. AVG's recipe is observation normalization, penultimate-layer feature normalization and TD-error scaling; target networks hurt. [fetched] | AVG: "often achieving final performance comparable to batch policy gradient methods" over 10M steps on gymnasium MuJoCo (per-task numbers are in figures; I could not extract HalfCheetah). Stream-RAC: median normalized score 0.519 over 50 tasks where 1.0 is the best batch method, vs 0.044 for incremental SAC. | Shows pathwise-without-replay can learn at all, at roughly half of batch performance. Both use one-step TD (no future-action noise) and thousands of policy updates per 1k steps. Neither measures action ranking. Their stabilizers (normalization) are already in REPPO. |
| MPO (1806.06920), V-MPO (1909.12238), AWR (1910.00177), AWAC (2006.09359) | Replace the pathwise step with a weighted regression onto sampled actions. V-MPO and AWR need only V(s), i.e. they avoid the action-ranking critic entirely. [memory] | V-MPO is fully on-policy but was run at billions of steps. | These are score-function-family estimators in disguise. This repo has dedicated vmpo/ and awr/ families; not a REPPO fix. |

Route B: first-order information from a model or simulator.

| Work | Mechanism | Evidence | Transfers? |
|---|---|---|---|
| SVG(0)/SVG(1)/SVG(inf), Heess et al. 2015 (arXiv 1510.09142) | Backpropagate value gradients along real trajectories through a (learned) model with inferred noise. [memory] | Small control tasks. | Local v30 (SVG(1) with a learned model) was neutral to negative [local]. |
| MAGE, D'Oro and Jaskowski 2020 (arXiv 2004.14309) | Train the critic on the action-gradient of the TD error, not its value: "it is the norm of the action-gradient of the policy evaluation error instead of its value that should be minimized". Needs a differentiable model. [fetched] | MuJoCo, model-based off-policy. | The principle is exactly our problem statement. The learned-model route is locally triple-negative (v30, v37, v38: model R^2 0.996 yet no better action ranking) [local]. |
| Sobolev training, Czarnecki et al. 2017 (arXiv 1706.04859) | Fit derivatives as well as values when derivative targets exist. [memory] | Supervised and distillation settings. | Needs gradient targets; see idea B7. |
| SHAC (2204.07137), AHAC (2405.17784) | Short-horizon analytic gradients through a differentiable simulator with a value bootstrap. [memory] | Differentiable simulators with smoothed contact. | Requires a differentiable simulator; gym MuJoCo only offers finite-difference Jacobians. |
| Suh et al. 2022, "Do Differentiable Simulators Give Better Policy Gradients?" (arXiv 2202.00817) | First-order estimates can have MORE variance than zeroth-order under stiffness and chaos, and are biased under near-discontinuous contact ("empirical bias"); the two estimators "may fundamentally operate on different landscapes". [fetched] | Contact-rich examples. | Caution for any long-horizon first-order route on HalfCheetah (local oracle: branches decorrelate within about 100 steps [local]). Short horizons only. |

Route C: massive on-policy batches.

| Work | Mechanism | Evidence | Transfers? |
|---|---|---|---|
| REPPO | 131k fresh transitions per policy from 1024 envs. | Section 1. | Section 2: not at an 8M budget. |
| PQN, Gallici et al. 2024 (arXiv 2407.04811) | LayerNorm makes TD stable without target networks or replay; lambda-returns from vectorized envs. "PQN can learn even with a small number of environments but clearly benefits from collecting more experiences in parallel." [fetched] | Discrete actions (max over a finite action set, so no action-gradient is needed). | Supports dropping target networks (done). Says nothing about continuous action gradients. |

Combining estimators:

| Work | Mechanism | Evidence | Transfers? |
|---|---|---|---|
| Q-Prop (arXiv 1611.02247), IPG (1706.00387), Stein control variates (1710.11198) | Use the critic's first-order Taylor term as a control variate for the score-function gradient, adding the pathwise term back analytically; unbiased for any critic. [memory] | Tucker et al. showed the published gains were largely implementation bias and the true variance reduction is small because Sigma_tau dominates. | Low. This repo already has Q-Prop and IPG variants in the residual_stiglu_ngpt line and v39 mixed the two gradients in REPPO (negative) [local]. |
| Expected Policy Gradients, Ciosek and Whiteson (arXiv 1706.05374) | Integrate the gradient over actions (analytically or by many samples) instead of using the one sampled action; removes Sigma_a for the actor. [memory] | Mostly analytic and small-scale. | Cheap: more reparameterized samples per state in the actor loss. Removes actor Monte-Carlo noise but not critic bias. |
| Compatible function approximation and natural actor-critic (Sutton et al. 1999; Kakade 2001; Peters and Schaal 2008); GProp, Balduzzi and Ghifary 2015 (arXiv 1509.03005) | With compatible features, regressing on-policy returns onto grad log pi recovers the natural gradient: the score-function estimator only needs the projection of A onto a theta-dimensional subspace, averaged over all samples. A pathwise critic must get the local slope right at every state. [memory] | Classical theory. | This is the statistical reason for the paper's E.2: with little data, projecting onto the score is efficient and a nonparametric Q is not [inference]. v63 was the compatible-style test and regressed [local]. |
| Smoothed Action Value Functions, Nachum et al. 2018 (arXiv 1803.02348) | Learn Q~(s, mu) = E over a ~ N(mu, Sigma) of Q(s, a) directly; its gradient and Hessian in mu give the mean and covariance updates of a Gaussian policy. [fetched abstract; details from memory] | Off-policy MuJoCo. | The idea of learning the policy-averaged object rather than the pointwise one is appealing, but training data still only covers actions within sigma of the mean. |

### 3.3 Classification losses

Farebrother et al. 2024, "Stop Regressing" (arXiv 2403.03950) [fetched]: sweeps bins in {21, 51, 101, 201}
and sigma / bin-width in {0.25, 0.5, 0.75, 1.0, 2.0} on 13 Atari games at a fixed value range. "A wide range
of sigma values ... outperform Two-Hot" and "the optimal value of sigma seems to be independent of number
of bins, indicating that HL-Gauss generalizes best across a specific range of target values". Two things
follow for us. First, what matters is absolute sigma relative to the target scale, and the local sigma
(1.5) is twice the reference's for the same Q scale. Second, all of that evidence is about value
accuracy for discrete argmax control; none of it concerns sub-bin action derivatives of the decoded mean.
The decoded Q is a sum over 151 bins with centers up to 250 units away, so a 1e-3 probability
perturbation on a far bin moves Q by about 0.1, which is half the true within-state sd (0.22). That is
consistent with the local finding that TF32 alone halves learning [local]. The reference shares this
weakness in kind (bin = 1 step of reward), so it is a property of the method more than of the port.
[inference]

---

## 4. Best-known fully on-policy HalfCheetah results at <= 8-10M steps

Numbers I could verify from primary sources:

| Method | Env | Steps | Return | Source |
|---|---|---|---|---|
| PPO (CleanRL) | HalfCheetah-v4 | 8M | 2637 +- 1068 | https://github.com/vwxyzjn/cleanrl/blob/master/docs/rl-algorithms/rpo.md |
| RPO (alpha 0.01) | HalfCheetah-v4 | 8M | 2717 +- 1315 | same |
| PPO (SB3 zoo, tuned) | HalfCheetah-v3 | 1M | 5819 +- 664 | https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md |
| TRPO (SB3 zoo) | HalfCheetah-v3 | 1M | 1785 +- 69 | same |
| PPO (Tianshou) | HalfCheetah (v3) | 1M / 3M | 5784 +- 1244 / 7337 +- 1508 | https://github.com/thu-ml/tianshou/blob/v0.5.0/examples/mujoco/README.md |
| PPO / SPO (Xie et al. 2025) | HalfCheetah-v4 | 10M | 4550 / 3602 (3-layer nets); 2242 / 5307 (7-layer) | https://arxiv.org/abs/2401.16025 |
| REPPO (paper) | - | - | no gym MuJoCo result | section 1.3 |
| AVG, Stream-AC / Stream-RAC | HalfCheetah-v4 | 10M / 5M | figures only; not extracted | arXiv 2411.15370, 2410.14606 |

Best verified fully on-policy number: about 7.3k at 3M (Tianshou PPO). I found no published fully
on-policy result above about 8k at <= 10M on gym HalfCheetah. The search was not exhaustive, v3 and v4
differ only in bindings, and Brax/MJX "halfcheetah" numbers are a different simulator and not comparable.
On that evidence the local REPPO line (about 11k stochastic, about 12k deterministic at 8M, single seed
[local]) is already above the published on-policy state of the art, and the gap that remains is to
off-policy methods (TQC 12.1k at 1M; local TD7 17.1k at 1M).

What the strongest methods do about critic quality: all of them are off-policy and combine (i) one-step
or short n-step TD on a large replay of action-diverse data, (ii) a mechanism against overestimation
(twin minimum, truncated quantile mixtures) or, more recently, normalization instead (CrossQ, BRO, SimBa,
FastSAC), (iii) many gradient steps per transition, and (iv) large or carefully regularized critics. The
strong on-policy methods (PPO family, V-MPO) avoid the question: they learn only V(s) and take action
information straight from returns through the score function.

---

## 5. Conclusions

### (a) Differences that could plausibly explain the weak critic

1. Data per policy evaluation, 8x less than published (16,384 vs 131,072), with the paper itself
   reporting that REPPO degrades steeply as this shrinks while PPO does not. This is the only difference
   backed by published evidence, and it is a property of the regime we chose in order to fit an 8M
   budget, not a porting error.
2. There is no evidence the published critic ranks actions well either. The paper never measures it.
   The reference setup spends about 1.3M samples per nat of policy movement and 512 actor steps per
   iteration, so a critic slope that is 20-37% real would still produce steady improvement there. The
   local observation that "a slow actor averages critic noise across iterations" [local] is probably how
   the published method works too. If so, "weak critic" is the method's normal operating point, and the
   plateau is where the averaged signal runs out.
3. Support resolution: [-50, 250] against the reference's [0, 150] for the same value scale (sigma 1.5 vs
   0.75), with about half the bins unused. Untested locally. Moderate prior, since the reference is also
   coarse.
4. Per-env observation statistics instead of one shared normalizer. Untested, cheap, small prior.
5. In-iteration target refresh with the live actor, against the paper's explicit "not recomputed". It
   helped locally, so it is not a suspect for the plateau, but it means the critic chases a moving target
   8 times per iteration, and "one iteration of fitting rewrites the critic's action ranking about as
   much as the ranking is worth" [local] is partly a statement about this churn.

Not suspects: architecture, HL-Gauss form, KL gate, multiplier mechanics, optimizer. They match.

### (b) Most promising ideas for a fast, fully on-policy REPPO at 16 envs

On env count and rollout shape: the evidence says do not change it. (i) At a fixed 16k batch, 128 envs x
128 steps was 0.7x the reference [local]. (ii) Growing the batch toward the paper's regime costs
iterations: 16 x 4096 reached 9.3k at 8M against 10.5k [local], and the paper's own default would have
done 61 iterations by 8M. (iii) Wall-clock is not the obstacle: native CPU physics runs at about 60k
transitions/s at both 16 and 64 envs (docs/mujoco-throughput.md) [local], so env count is nearly free in
time up to at least 64. One real cost at high env counts: the staggered-start warmup consumes
episode_horizon x num_envs steps of the budget (1.02M of 8M at 1024 envs, 128k at 128). The binding
constraint is samples per unit of policy change, and that cannot be improved by reshaping a fixed budget.

Ideas, ordered by my prior. Each is specified so it can be run as a single change on the v60/v36 base.
"Untested" means I found no record in cleanrl/reppo/FAMILY.md; I did not audit the other 27 families.

B1. Time-averaged critic for the actor only (Averaged-DQN, Anschel et al. 2017, arXiv 1611.01929).
    Keep training the critic as now. Maintain an EMA of its weights across iterations (for example
    decay such that the half-life is 4-8 iterations, updated once per critic step) and differentiate the
    EMA critic in actor_loss; targets and diagnostics keep using the live or target critic. Rationale:
    the per-iteration slope is signal plus fitting noise that is rewritten every iteration on fresh
    data; averaging m iterations cuts that noise by about sqrt(m) at the cost of a bias from policy
    drift (0.03 nats per iteration). This makes explicit the averaging the slow actor currently does
    implicitly, and would let the actor take more or larger steps. Check with the oracle: slope of true
    directional derivative on the EMA critic's vs the live critic's. Untested. Differs from v50, which
    anchored the critic's training and froze its contrasts.
B2. Mixed exploration scale across envs with importance-clipped lambda (dormant in the official JAX
    code; PQL's "mixed exploration"). Env i acts with std scaled by c_i (for example 4 envs at 1.0 and
    12 spread over 1.0-2.0). In the target recursion replace lambda by lambda x clip(pi(a|s) / mu_i(a|s),
    lmbda_min 0.5, 1), exactly as src/jaxrl/reppo.py does. Rationale: the standard error of a regression
    slope scales as 1 / (action sd), and the local law "ranking correlation 0 at std 0.1, 0.1 at 0.3,
    0.29 at 0.45" [local] says coverage is the lever. Cost: the wide envs score lower, and the benchmark
    is training return, so report base-env and all-env returns separately. FastTD3 found "no significant
    gains" from mixed noise with replay; PQL found gains. Untested here.
B3. Support matched to the reference and finer: v_min -20, v_max 180 with 401 bins (bin 0.5, sigma
    0.375), then 801 bins (bin 0.25). Strict FP32. Rationale: section 3.3. Predicts a higher
    advantage_std_ratio and a higher oracle slope if smoothing or tail leakage is compressing contrasts
    (the critic's contrasts are 3x compressed against truth [local]). Untested.
B4. One shared observation normalizer for all envs, as in the reference. One-line change in spirit;
    removes a per-env input offset. Untested.
B5. Larger effective batch for the critic's later passes: keep 128 updates but use the full 16,384-row
    batch for the last quarter of them (or halve the critic learning rate over the iteration). Rationale:
    FastTD3's observation that large batches give "a more stable learning signal for the critic"; the
    action signal is 1-4% of the loss, so minibatch noise from state-level terms dominates its gradient.
    Note the paper found learning-rate schedules harmful in its regime. Untested.
B6. More reparameterized samples per state in the actor loss (expected-gradient, Ciosek and Whiteson),
    for example 8 with the mean over samples before the KL gate. Cheap on GPU. Removes actor Monte-Carlo
    noise only; useful mainly in combination with B1.
B7. Diagnostic ceiling, not a fair method: first-order gradient targets from MuJoCo finite-difference
    Jacobians (mujoco.mjd_transitionFD) along the real trajectory, with a short lambda-weighted backward
    recursion (effective horizon 1 / (1 - gamma x lambda), about 5 steps at lambda 0.8), used as a
    Sobolev term |grad_a Q - g|^2 or directly as the actor's gradient. This gives a 6-number slope label
    per transition instead of one noisy scalar, which is what MAGE, SVG and SHAC rely on. It needs
    simulator state access and roughly 30-60 extra physics steps per Jacobian that are not counted as
    samples, so it breaks sample accounting. Its value is to answer the root-cause question decisively:
    if an actor fed near-true short-horizon gradients still plateaus near 11-12k, the critic is not the
    bottleneck. The local one-step version of this already moved the ascent ratio from 0.55 to 0.79
    [local]. Suh et al. warn about stiffness and contact bias; keep the horizon short.
B8. Critic architecture from the scaled off-policy line (SimbaV2 hyperspherical normalization or BRO
    residual blocks) with return-scale normalization. Lowest prior of the list: all evidence is with
    replay at high update-to-data ratios, and the local critic's problem is generalization of contrasts
    across states with little data, which more capacity does not obviously help.

If none of B1-B6 moves the oracle slope, the literature's answer is unambiguous and it is not on-policy:
accurate action gradients come from replayed one-step TD on action-diverse data (Route A), with the
full recipe rather than a window bolted onto lambda-returns.

### (c) Dead ends according to the literature (and local evidence where it agrees)

- Action-dependent baselines or control variates as a variance fix (Q-Prop, Stein, IPG as control
  variates): Tucker et al.; locally v39 and the residual-line Q-Prop variants.
- Using the critic's own advantage to denoise multi-step targets while the critic cannot rank actions
  (v61/v62): the removable variance is rho^2, about 1-2%.
- Action-gap-increasing operators: built for greedy discrete backups; bias policy evaluation.
- Learned-model imagination for action ranking (MAGE with a learned model, SVG(1), Dyna branches):
  locally triple-negative despite an accurate one-step model; Suh et al. explain why contact-rich
  first-order signals degrade.
- Exploiting the return distribution's shape (quantiles, pessimism, probability matching): Z is a
  location family here [local]; the distributional literature's gains are attributed to the loss, not
  the shape (Farebrother et al.).
- Separate advantage heads on on-policy data (dueling, DAU-style, DAE-style): the diagnosis is right
  but v26, v51, v53 and v63 show the head inherits the same data limit.
- Lower lambda or pure one-step targets without replay: v25 at lambda 0.5 fell into the flipped optimum
  and v53 at lambda 0 was worst [local]; the published on-policy Q methods use lambda 0.65-0.95.
- Reshaping the same 8M steps into many envs with short rollouts: section 5(b).
- Target networks or Polyak targets for an on-policy critic: REPPO, PQN and AVG all remove them; AVG
  finds they hurt; locally v32/v33.

### Things I could not verify

- Per-component ablation magnitudes in the REPPO paper (figures only).
- HalfCheetah numbers for AVG and the Stream-X methods (figures only).
- Which batch sizes the labels in the paper's E.2 table refer to (labels and products disagree).
- Whether the per-env observation statistics differ enough across envs to matter (no measurement).
- Claims marked [memory].
- I did not audit families other than cleanrl/reppo for prior tests of B1-B8.
