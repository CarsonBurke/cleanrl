# TD7 variants

Off-policy TD7 continuous-control line (base `../td7_continuous_action.py`
stays flat). MuJoCo benchmark, seed 1.

Subfamilies:

- `lesale/` — LEWM/SALE latent-action world-model line (compile, JEDI,
  LE-JEPA, LEWM-rollout threads).
- `stocksig/` — stock-signal + state-dependent noise + LE-JEPA reward/policy
  heads (dual-proj incl. WPPG variants, HL-Gauss reward betas, isometric/
  latent64/semantic64, PC-actor).
- `ctx/` — context-conditioned actor/critic (v1-v3).
- `noise/` — exploration-noise sweeps (betanoise, sdnoise, softbeta, softgauss).
- `fast/` — throughput base: GPU replay/LAP, CUDA-graph train steps
  (`shared/graph_step.py`), host-mirror acting. Same algorithm as
  `noise/td7_sdnoise_v1.py`; start new TD7 variants from here.
- `latentq/` — critic over the pre-action latent (mu, sigma) instead of the
  action, trained with the Hermite/Stein projection of the realized noise
  (v1), plus realized-action points at sigma=0 (v2). `preaction_onpolicy_v8`
  keeps the v5 critic on replay but collects with 16 staggered envs x 1024 and
  gives the actor one KL-sized natural-gradient step per rollout.
- `consequenceq/` — the action reaches the value only through a supervised
  one-step model: Q(s, a) = r^ + gamma (1 - d^) V(s^'(s, a)), with V a state
  value. Tests whether a dynamics-grounded actor gradient leaves less critic
  error to exploit, measured as fresh-data optimism (`fresh/*`). Its reference
  is `latentq/..._fresh_optimism_v27` (v5 plus the same probe). v1: ~4x more
  accurate fresh values than v5 but slower improvement (15.3k vs 17.0k @800k).
  `costate_v2` adds a value-gradient Bellman loss along the actor's directions
  (16.5k @800k); `costate_closed_v3` closes that recursion with random state
  tangents; `belief_residual_v4` adds A(s, b) in unit-scale belief coordinates
  on top of v2's grounded V.
- `heatbelief/` — the critic values the actor's belief N(mu, sigma^2) exactly:
  Q(s, a) = V0(s) + sum of nonnegative axis-aligned Gaussian bumps, so the
  expected value under clip(N(mu, sigma^2)) is closed form per dimension (erf)
  and obeys the heat equation. Point TD on the executed action only; actor
  ascends the exact belief value (no action path); target is the exact
  expectation under 0.2 smoothing. `--chart` adds zsa(zs, mu_current) as
  detached trunk context (the only route for SALE's zsa). Reference: v27.
  v4 (consequenceq belief_residual) post-mortem that motivated it: fresh
  optimism +60-90 median / +520 p90 and a collapsing stochastic-return tail
  (p10 3.6k vs v27 18.2k @1.7M) — optimism, not accuracy, was the problem.
  Results (eval): v1 9.3k @100k / 12.3k @300k (bumps collapse to ~3 active,
  pessimistic), chart v1 ~same, `dictionary_v2` (global learned atoms) best TD
  loss but 7.6k @100k; v27 is 12.8k / 15.9k. All cancelled: exact belief
  algebra without the zsa(zs, a) action path loses, and lower TD loss again did
  not buy return.
- `selfcritic/` — v27 plus introspection, three arms. `selfmodel_v1`: b(s, mu,
  sigma, u, age) regresses the critic's own error against realized returns on
  the last 20k transitions (age = training steps since collection, queried at
  0, so policy drift is not read as optimism); the actor ascends Q~ - relu(b).
  `deliberate_v1`: a one-step model plans 3 steps over 64 belief draws, best
  becomes the belief mean, executed action is still a Gaussian draw around it;
  evaluation deliberates with a checkpointed `Mind` (actor, critic, model,
  normalizers). `selfcritic_deliberate_v1`: both, the planner terminal being
  the actor's own objective twin-mean Q~ - relu(b). `fresh/sm_*` scores the
  self-model on unseen episodes; `charts/plan_gain` is a max over noisy draws,
  so biased upward.
  Results (eval, v27 15.8k @300k / 18.6k @1.2M): selfmodel 11.6k @300k, 15.0k
  @1.2M, with the live policy collapsing (training 1.1k @1.3M);
  selfcritic_deliberate 12.2k @300k. Both cancelled. b never learned per-state
  error (fresh sm_corr ~0): rows within an episode share their return tail, so
  20k rows are ~20 samples. b only tracked the window's mean error, which swung
  -300..+350, and relu(b) then fed the actor large, uninformative, exploitable
  gradients. deliberate 15.3k @300k (kept running), but ~7x slower (105 SPS)
  and it almost never keeps the mean (0.3%). Its training returns trail
  because the executed action adds sigma noise on top of an already-sampled
  pick.
- `beliefpotential/` — prescriptions 1-2 of `benchmarks/selfcritic_analysis_v1/
  REPORT.md` on v27. The center gets one meaning, C(s, mu, sigma) = E Q(s,
  clip(mu + sigma eps)): the behavior-query label is dropped, and the heat
  identity sigma dC/dsigma = sigma^2 d2C/dmu2 is enforced at the actor's belief.
  The curvature is a central difference of exact mean JVPs through u and zsa.
  Every arm also runs a logging-only direction audit on fresh episodes:
  realized Stein slope over the claimed slope along the critic's ascent
  directions (`audit/rho`, `audit/rho_sigma`), plus held-out skill of an auditor
  A(s) against constant and twin-agreement baselines. Arms: `v27sem`
  (`--behavior-channel --heat-weight 0`), `control` (`--heat-weight 0`), `heat`.
  First reading (~35k): v27 semantics has a sigma slope ~15x below its own
  curvature (heat residual 0.33), confirming the mixed-meaning diagnosis.
  Results (eval @300k / 500k; v27 15.8k / 17.3k): v27sem 16.2k / 17.4k, control
  16.2k / 17.3k, heat 16.0k / 17.2k. All cancelled ~600k: null. The heat loss
  made C consistent (residual 0.09) and shrank sigma to 0.14, with no return
  effect. Audit, the lasting result: rho ~0.5 in every arm, i.e. the realized
  one-step slope along the actor's ascent direction is half the claimed one.
  Per-state skill was ~0 for every predictor (auditor negative): single-sample
  Stein slopes are unidentifiable per state. With twin cosine ~0.65 most slope
  error is shared, so even an infinite ensemble caps rho near 0.64 (inferred).
- `valuegeometry/` — prescription 3 on v27. The encoder also learns
  w(zs) . (zsa(a) - zsa(mu_b)) + b(zs, mu_b, sigma_b) ~ normalized
  (y - C(s, mu_b, sigma_b)), so the zsa displacement the actor's gradient flows
  through encodes value differences to first order (vg_weight 0.03 against SALE
  MSE ~0.005-0.05). `noroute` (`--vg-weight 0 --no-actor-zsa-route`) detaches
  the actor's zsa(mu) path: how much of v27's learning flows through SALE.
  Results (eval; v27 15.8k @300k, 18.4k @700k): vg 16.4k @300k but 17.2k @700k
  (rho unchanged at 0.47, vg_r2 ~0.19); noroute 4.6k, flat from 100k to 925k.
  Cancelled. Structural finding: essentially all of v27's policy improvement
  flows through J_zsa(mu)^T dQ/dzsa (a one-step latent-model gradient); the
  critic's direct mu, sigma and u inputs alone teach the actor almost nothing.
- `latentaction/` — the critic scores the actor's stochastic final latent, not
  the action. A 64-d belief z = clip(m + sigma eps) is the "action" for replay,
  SALE zsa(zs, z), the belief critic and the actor. The head is detached and
  frozen: a = tanh(1.5 W z), with W's rows orthonormal. Latent noise becomes
  correlated action noise W Sigma(s) W^T. Logs charts/action_noise_std and
  charts/action_saturation. v1's latent entropy budget is slack: sigma_min at
  the floor, sigma_max at 1 and alpha ~0.004 from ~100k, because the
  axis-aligned dims have different reach through the head. Action noise
  settled at ~0.135 anyway. Eval 8.5k @100k, 12.0k @300k, 15.8k @480k (v27
  12.0k / 15.8k / 17.2k).
  `twocritic_v2`: the head learns, a = tanh(z W + MLP(z)), initialized to v1's
  map. It has two critics with their own SALE encoders. Qz(s, z) sees only the
  state and latent and trains the trunk. Qa(s, a) trains only the head, at the
  detached mean m, at head_lr 1e-4. The trunk's entropy is the pushforward
  action entropy 0.5 logdet(J Sigma J^T), with J the detached analytic head
  Jacobian, so latent directions the head ignores earn none. Failure
  signatures: fresh/abs_err_z growing with losses/head_correction (Qz is
  off-policy in the head), and Qz-vs-Qa fresh error.
  v1 cancelled at ~500k (user: not it). v2 went NaN at training step 43k:
  the head saturated (correction 0.7), the post-tanh J lost rows, and the
  TF32 covariance stopped being positive definite under cholesky_ex. It was
  also behind before that (3.7k vs v27 6.8k @50k). `twocritic_v3` uses a
  pre-tanh J and a float64 covariance.
  v3 ran 2M (eval): 6.4k @100k, 12.9k @300k, 14.9k @500k, 16.4k @700k, then
  flat at 16.4-16.7k (best 17.1k). v27 is 12.0k / 15.8k / 17.3k / 18.4k and
  ~19.2k. The latent critic is consistently more accurate on fresh episodes
  than the action critic: |err| 55-160 vs 230-270. Head drift from the initial
  map holds at ~0.5-0.6, saturation ~0.3-0.4, executed noise ~0.05.
  `twocritic_v4` separates the confounds. It is v1's latent side (v27 belief
  critic) plus flags: a = learned head, motor entropy; b = `--no-learn-head`;
  c = `--action-noise 0.1` (post-tanh); d = `--entropy-space latent`;
  e = `--target-sigma 0.15`. At the same target, the motor budget runs ~1.5x
  less noise than the latent one (latent sigma 0.067 vs 0.1 through the 1.5
  gain). So a vs e is the noise level and e vs d the budget geometry. A
  learned head also re-weights the belief critic's LAP sampling (priority =
  max of both TDs), so a vs b carries that too. v1 is the frozen/latent
  control.
  At 290k: b (frozen) 13.8k, a 13.2k, c 13.0k, d 12.7k, e 11.7k; v1 13.0k and
  v27 15.9k. Alpha was ~0 in every arm, so the entropy budget never binds;
  c, d and e were cancelled. The learned heads drifted 0.55-0.69 and saturated
  25-33%, against 15% for the frozen head.
  `twocritic_v5`: LAP priority comes from the belief critic only, and the arms
  are a head_lr ladder (1e-4, 1e-5, 3e-6; b is lr 0). A full-rank frozen head
  loses no expressiveness. Learning it only buys noise geometry, at the cost
  of Q~'s stationarity, so the head should run on a slower timescale.
- `dgrl/` — Distance-Guided RL (arXiv 2602.08616) on delight reference v10:
  the actor regresses onto a critic-softmax average of its own Gaussian draws
  (DBU) with a matching weighted-variance sigma fit, optionally acting by
  rank-softmax over critic-scored candidates (SDN).
- `misc/` — hop/hist/hvm memory, search, stepmem, tap, rank-greedy,
  delightful-PG port.
- `consequencenoise/` — v27 with behavior noise shaped by SALE's action
  Jacobian J = d zsa/da at mu (host NumPy forward mode): pre_action =
  mu + S sigma eps, S = det-1 ZCA whitener of (J^T J + lam I), lam =
  0.1 tr(J^T J)/d. Entropy budget unchanged; first-order Hermite tangent is
  u - mu, second order v27's on whitened eps (S does not depend on sigma).
  Tests whether consequence-isotropic exploration feeds the J^T dQ/dzsa route
  better labels. Logs consequence/* (cond, effective rank, shaped vs diagonal),
  shape/cond_S, noise/action_std*, noise/clip_frac. `--no-shape-noise` is the
  matched control.
  Cancelled at ~40k (user: not it); at that point shaped vs diagonal
  consequence cond was 81 vs 74, i.e. shaping had not yet bitten.
- `intention/` — the critic judges intentions, never actions: Q(s, zs, g) with
  g = zsa(zs, a), SALE's predicted next embedding. No action, mu, sigma or
  pre-action input, so judged intentions are reachable by construction.
  `judge_v1` is v27 mapped into g. Point TD on g(a_executed); the pre-action
  channel merges into point (weight 2/3); the center keeps its order-1 Hermite
  tangent through zsa_jvp. Order 2 needed sigma as an input, so it moves to the
  actor: sigma ascends an antithetic belief value Q(g(sg mu +- sigma eps)).
  Without it, entropy alone drives sigma to 1. The target uses TD7's sampled
  smoothing. Logs `fresh/intent_gap` (SALE error on unseen episodes) and
  `losses/belief_gap`. Reference: v27.
  `contrastive_v1`: TD-InfoNCE judge. Twin phi(s, zs, g) and psi(zs(s_f), s_f)
  with cosine logits times a learned scale. The contrast set is the batch's own
  next states plus a learned zero-reward absorbing column ⊥; terminal rows put
  their gamma mass on ⊥. Readout Q = 1/(1-gamma) E_softmax[r_j]. Two arms
  (`--judge`): `contrastive`, where the actor ascends the readout with no TD
  critic and replay is uniform, and `hybrid`, judge_v1 unchanged plus
  0.1 x TD-InfoNCE on the critic's q2 features. Logs `losses/contrastive_loss`,
  `losses/readout_q`, `charts/contrastive_acc`, `charts/logit_scale` and
  `charts/absorbing_mass`. Jobs 14795 and 14796 (HalfCheetah, 2M steps).
  Pure arm (14795) cancelled: ~3.5k eval @100k vs v27's 12k. Top-1 next-state
  accuracy fell from 1.4% to 0.39% (chance, 1/257) while the logit scale rose
  10 -> 22; readout_q sat at 300-390, about mean batch reward / (1-gamma), so the
  readout was the same for every intention and the actor had no signal
  (fresh rel_optimism 9-16). Inferred cause: the only hard positive weighs
  1-gamma = 0.01; the other 0.99 bootstraps from a near-uniform target against a
  fresh contrast set each step, too weak to grow a 100-step occupancy.
  `contrastive_v2`: v1's judge with Monte Carlo positives, as in contrastive
  RL. Row t's positive is s_{t+k}, k ~ Geometric(1-gamma), taken from the same
  episode in the whole-episode-flushed ring and carrying reward r_{t+k-1}, so the
  readout is the discounted return of the replay's behavior mixture (one-step
  evaluation, not the current policy). Past a terminated episode's end the
  positive is ⊥; past a truncated end it clips to s_T (biased). `--td-mix`
  lerps toward v1's TD label, sampled with the same k; on k > 1 rows the row's
  own column (a future of s' on the same trajectory) is masked out of the target
  weights. Arms `mc` (0) and `mix` (0.5). Logs `charts/future_offset`,
  `charts/absorbed_frac` and `charts/stale_futures` (expected 0). Jobs 14800 (mc)
  and 14801 (mix), HalfCheetah, 2M steps.
  Results: hybrid tracked v27 to 1.68M (18.9k vs 19.0k; logit scale pinned
  at 100, accuracy 1-2%), null. v2 (trajectory futures) discriminates (top-1
  12-23% vs 0.4% chance), and mc's readout is the most accurate critic in this
  repo (fresh |err| 5.6, rel optimism 0.00). Yet the policy barely moves: mc
  1.4k @185k, mix 3.4k @165k, v27 13.6k. Mechanism (inferred): batch
  negatives are futures of other states, so the classifier can be solved from
  s alone (which trajectory am I on). The intention's effect on the
  100-step occupancy is a small difference on top of that, and dQ/dg ~ 0.
  Accuracy without action-sensitivity again buys no improvement.
