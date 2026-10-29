# REPPO root-cause study, part 10: prior attempts and feasibility of the candidate directions

Role: prior-art-within-the-repo and feasibility checker. Read-only: no training, no GPU job, no edit to an
existing file. Reference implementation for every "smallest faithful implementation" below is
`cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py` (line numbers `v60:N` refer to it).

Provenance tags:
- [CODE] read in the source in this session.
- [TSV] matched-step returns from the first-round dump `reppo_runs.tsv` / `all_hc_runs.tsv` (same +-50k window
  as `scripts/score_runs.py --at`); I re-checked `vtrace_v11*` and `fork8m_v40*` against `score_runs.py` directly
  and they agree to the digit.
- [RAW] read from tfevents / checkpoints / JSON by me in this session (scalars, logged hyperparameters).
- [DOC] read only in FAMILY.md, OPTIMUM.md, a file header or a first-round report; not re-verified.
- [INFER] my reasoning; not measured.

Scripts (both read-only on `runs/`):
- `scripts/10_rows.py` - prints return rows from the TSVs for run-name regexes.
- `scripts/10_hparams.py` - dumps the logged `hyperparameters` text of 216 REPPO-related runs to
  `data/10_hparams.json` and queries it (used for every "which precision / KL / lambda did that run use" below).

Return rows are `@1M / @2M / @4M / @8M` unless stated. "Ref" = v60 base (path B): 4267 / 5839 / 9084 / 11069;
path A (v27 H8 = v36 FP32): 4174 / 5634 / 8596 / 11031. "Flip" = flat 2.0-2.5k plateau. Grades A-D as in 01.

---

## 0. Findings that change the candidate list

1. **Simulator-branch ("vine") labels have never been used in training.** v37 and v38 branch through a learned
   world-model ensemble [CODE]; no trainer in `cleanrl/` restores MuJoCo state. It is feasible at zero net
   wall-clock under honest step accounting, without touching the training vector env (section 2).
2. **An averaged critic for the actor has never been tried.** Every Polyak network in the family (v24, v32, v33)
   feeds bootstrap targets only; the actor always differentiates the live critic [CODE]. v50 is a different
   object (a loss on the trained critic) and was cancelled at 1.2M on flip-level returns.
3. **The KL bound was only ever moved down on the modern recipe.** 0.015 / 0.01 from scratch (v23), 0.01 / 0.003
   as 8M forks (v40), 0.0075 (v35, flipped). The published 0.1 was run only on the v1 recipe (16 x 256 steps,
   coupled minibatches, 4.5k) and once together with a 4x longer rollout (v23 ns4096 kl 0.12). KL 0.1 on a
   16 x 1024 decoupled recipe: never run [RAW hparams].
4. **Every 8M fork screen (v40, v41, v42, entropy forks) ran in TF32** (`matmul_precision=high` in all 13 fork
   runs [RAW]), from a strict-FP32 snapshot. FAMILY.md says both "started from the FP32 snapshot ... so they
   stand" and "likewise TF32 and provisional"; the second is the accurate one.
5. **The v40 dual-PG arm was not a test of a hybrid gradient.** Its likelihood-ratio term used unnormalised
   advantages with sd 2.3-5.2 (scaled units) against critic contrasts of ~0.15 of that, its surrogate mean was
   negative right after the fork (-0.58, -2.5), the policy std jumped 0.52 -> 0.83 and kl_violation to 0.71 in the
   first 0.3M after the fork [RAW]. The LR term dominated and destabilised the restart.
6. **`reppo_qprop_v1`'s +1.0k cannot be variance reduction.** Its batch-optimal weight eta is 0.065 -> 0.018 and
   the control variate correlates 0.04-0.05 with the advantage, i.e. it removes ~0.2% of the variance [RAW].
7. **v61's "exact" target is v10's Retrace target at buffer 1** (same taken-action baseline Q'(s_{t+1}, a_{t+1}))
   [CODE both headers and target code]. v10 collapsed (4.2k) with once-per-rollout targets; v61 l095 reached
   9.7k with in-iteration refresh. Retrace with refresh plus a replay window is therefore untried, and the v11
   "memorisation" explanation for v10 was never needed.
8. **`--resume` in v60 is not a bit-exact continuation.** Environments, the host action sampler, the CUDA
   generator and the diagnostic generator all restart from `--seed` [CODE]. A fork control differs from the
   unforked base (v62 control vs base: -181 / -322 / -744 at 2 / 4 / 8M [TSV]); a same-code control fork is
   mandatory, and fork-to-fork reproducibility has never been checked by a duplicate run.
9. **Three full-state checkpoints exist, all on path B, all upright:** `checkpoint_1015424.pt`,
   `checkpoint_4013696.pt`, `checkpoint_7913088.pt` under
   `runs/HalfCheetah-v4__reppo_noisefork_v60_base__1__1790713932/` [RAW]. No `fork_*.pt` exists anywhere.

---

## 1. Averaged critic for the actor (EMA of weights, or mean of the last K critics' dQ/da); v50

### (i) Prior versions

| Version | What the code does [CODE] | Difference from the direction |
|---|---|---|
| v9 frozenjudge | `--frozen-judge`: actor reads `target_critic` (the critic frozen at the iteration start) instead of the live one (l.410). v7 recipe, 32 critic updates, lambda 0.95, no refresh | a one-iteration *lag*, not an average: it removes within-iteration chasing but keeps one fit's noise in full |
| v24 / v32 / v33 | `torch._foreach_lerp_(target_parameters, critic_parameters, target_tau)` after each critic update (tau 0.005) | the Polyak copy feeds bootstrap targets only; `actor_loss` calls `critic.value` (live) in all three |
| v41 twinq fork | actor climbs member 0 / mean of 2 / min of 2; second member fresh, 16-iteration critic-only warmup | an average over two independently initialised critics at one time, not over time; TF32 fork |
| v8 crossfit, v47 crossspread | two critics on env halves or on all data; actor reads the out-of-fold one, or mode and spread read different critics | independence across data, not temporal averaging |
| v50 contrastanchor | adds `kappa * mean (C - sg C_prev)^2 / (2 Var_Z)` to the **critic loss**, C = Q - mean over 8 policy actions, C_prev from the critic that ended the previous iteration under the normalisation it was fit with (l.398-404, 460-463, 716-730) | see below |
| qprop_v1 | the control variate's slope g is taken before Q trains on the batch | lag for leakage control, PPO base |

No file in `cleanrl/reppo`, `awr`, `td7`, `policy_aware_critic`, `critic_uncertainty` or the top-level
qprop / ipg / latent_critic files keeps an EMA or snapshot average of a critic for the actor's gradient
(grep for polyak / lerp / ema / swa [CODE]).

### How v50 differs from an EMA-for-actor critic

1. **Where it acts.** v50 changes the trained critic's weights, so it also changes the bootstrap values V', the
   lambda-returns and the latent-roll targets. An EMA-for-actor leaves the critic and its targets untouched and
   only changes which function the actor differentiates.
2. **What it averages.** v50 pulls this iteration's contrasts toward the previous critic's contrasts *on states
   the previous critic never saw* (the fresh rollout). That is the previous critic's generalisation, its weakest
   part (held-out corr ~0.08 vs ~0.25 on its own training rows [DOC v51]). A weight EMA averages fitted and
   generalised parts alike.
3. **Shrinkage.** The anchor starts in iteration 2 against a critic whose contrasts are ~0, and each iteration
   anchors to an already compressed estimate. Measured: drift fell 0.049 -> 0.008 but contrast sd fell
   0.071 -> 0.022 [DOC]. An EMA cannot stop the live critic from learning contrasts; its amplitude converges to
   the persistent component of the live critic's.
4. **Memory.** Nominally comparable (1 + kappa = 4 or 8 iterations), so v50 is a statement about anchoring in
   the loss, not about the time scale.

### (ii) Outcomes

| Run | @0.5M | @1M | @4M | @8M | Flip? |
|---|---:|---:|---:|---:|---|
| v9 frozen vs v7 live (same recipe) | 958 vs 1547 | 2115 vs 2585 | 5594 vs 6218 | 8595 vs 8152 | no |
| v50 kappa 3 / 7 | 1398 / 1406 | 1838 / 1953 | - | - | cancelled 1.23M; these are the flipped runs' numbers (1353-1380 / 1884-1947); posture not measured |
| v41 first / mean / min @12M (fork from 8M) | | | | 11005 / 10924 / 11401 | no |

All [TSV].

### (iii) Grade and status

- v9: C (+0.44k at 8M, -0.5k at 1M, single trajectory each, weak recipe). v50: C (culled at 1.2M on the flip
  plateau; the compression diagnostic is real but says nothing about averaging). v41 mean: B- null, TF32.
- **Status: NEVER TRIED** (EMA weights for the actor; explicit K-snapshot gradient average).

### (iv) Feasibility on v60

Prediction worth writing down first [INFER from DOC]: v49 measured per-iteration contrast drift rms about equal
to the contrast sd (0.031 / 0.031 early, 0.106 / 0.14 at 1.6M), which implies consecutive fits' contrasts
correlate about 0.5-0.7. An m-iteration average should then raise `probe/slope_beta` (real / critic slope,
already logged by v60: 0.50 / 0.71 / 0.40 / 0.28) toward 1 and shrink the gradient norm. Since the actor is
KL-saturated and grad-clipped at every step, a smaller gradient norm by itself changes nothing; any effect has
to come from direction.

EMA of weights (smallest version, about 25 lines, new file v64):
- After `target_critic` (`v60:389`): `actor_critic = copy.deepcopy(critic).requires_grad_(False)`.
- `actor_loss` (`v60:492`): `q = actor_critic.value(obs, actions)`. `loss.backward(inputs=actor_parameters)`
  (`v60:638`) already differentiates the critic only through its action input.
- Update loop (`v60:865`): after `critic_update()`, `torch._foreach_lerp_(ema_parameters, critic_parameters, tau)`
  (the exact call v24 l.507 uses, outside the captured graphs).
- Flags: `--actor-critic {live,ema,frozen}`, `--actor-ema-iterations m` with tau = 1 / (128 m); arms m = 1, 3, 10.
- Checkpoint: add the EMA state to `save_checkpoint` (`v60:709`); on `--resume` from a v60 checkpoint that
  lacks the key, initialise it from `critic`. The existing three checkpoints stay usable.
- The critic has only Linear + RMSNorm (no running statistics), so weight averaging is well defined.
- Diagnostics to add: `regret_probe` and `critic_diagnostics` evaluated on the EMA critic next to the target
  critic (they are the calibrated in-run readouts), cos(dQ/da live, EMA).
- Compute: no extra forward (the actor swaps one critic for another); 128 lerps over 1.4M parameters per
  iteration, under 1% of the 0.436 s update.

Explicit K-snapshot gradient average: K frozen critics plus the normalisation each was fit under (v50 already
carries `final_norm` / `previous_norm`, l.398-400, 675-680). Each actor step costs K critic forward/backward-to-
action passes over 2048 rows instead of one; the actor step is a small share of an update dominated by 128
critic steps with the H8 roll, so K = 4 is roughly +10-25% update time [INFER; not profiled].

Caveat: per-env observation statistics move between iterations. A weight EMA is evaluated under the current
statistics only; this matters in the first ~20 iterations and is negligible later (counts 63k at 1M, 251k at
4M [RAW checkpoint]).

---

## 2. Simulator-branch ("vine") advantage labels with common random numbers

### (i) Prior versions

| Version | What the code does [CODE] | Difference |
|---|---|---|
| v37 dynabranch | world-model ensemble M_e(s, a) -> (delta, reward, continue) fit on a replay of real transitions; critic also fits imagined CRN branch lambda-returns (A = 4 branches, H = 5) | learned model, not the simulator |
| v38 branchadv | separate scale-normalised A(s, a) net fit to imagined CRN branch contrasts (A = 8, H = 12, lambda 0.95); actor climbs (1 - w) Q + w A | learned model; A fit its own targets at normalised MSE 0.78-0.93 [DOC] |
| v30 modelgrad | actor climbs r_hat + gamma V(s + delta_hat) through the learned latent model | learned model, SVG(1) |
| `scripts/reppo_critic_oracle.py`, `reppo_oracle.py`, `awr_oracle_advantage.py`, `landscape.py` | real MuJoCo branching from restored states | offline diagnostics on frozen snapshots only |

Confirmed by grep over `cleanrl/` for `set_state|qpos|qvel|qacc_warmstart|mj_step|Physics`: the only trainer hits
are attention-position variables in `td7/ctx`, a field list in a sepclip microscopy diagnostic and the word
"Physics" in v37's header. **No version trains on real-simulator branches.**

### (ii) Outcomes

| Run | @1M | @2M | @4M | @8M | Flip? |
|---|---:|---:|---:|---:|---|
| v37 (TF32) | 1697 | 2109 | 2274 | 2363 | yes (pitch 3.29, 99% inverted [DOC]) |
| v37 FP32 | 1749 | 2068 | 2211 | - | yes (flat; not written up) |
| v38 (TF32) | 1203 | 1598 | 3155 | - | "half inverted" 47% at 1M [DOC]; still rising at 4.6M |
| v38 FP32 | 1006 | 1159 | 1486 | - | not a 2.3k plateau; simply not learning |
| v30 w0.5 / w1.0 | 3522 / 3995 | 5941 / 5095 | 8605 / 7214 | 9588 / 8810 | no |

Offline evidence that the real-simulator version is a different proposition [DOC 03 section 1f, v60 base at 1M,
512 states x 58 actions]: correlation with the 400-step truth of the within-state advantage is +0.15 for Q
directly, +0.34 (5% trim) for one true step then the critic's V', **+0.84-0.87 for five true steps then V'**,
+0.86-0.91 for ten. The true advantage is complete within ~10 steps. Measured at 1M only.
FAMILY.md's "common random numbers do not help: trajectories diverge chaotically" refers to 64-400-step
rollouts; at n <= 10 that objection does not apply.

### (iii) Grade and status

v37: C (flip, twice). v38: C+ negative for the learned-model A. v30: C+ mild regression.
**Status for the direction as specified: NEVER TRIED.** The learned-model negatives do not transfer: the model's
own branch ranking against real returns was 0.03-0.06 [DOC].

### (iv) Feasibility (the key question)

What the env stack supports [CODE]:

- `cleanrl/shared/mujoco_env.py` `NativeMujocoVectorEnv` has no save/restore API. `mujoco_batch.c` exports only
  `cleanrl_pool_create`, `cleanrl_pool_step`, `cleanrl_pool_destroy`; `step_env` runs `mj_step` x frame_skip
  then `mj_rnePostConstraint` on each env's own `mjData` in place.
- The state is nevertheless readable: `envs._bases[i].data` is the gymnasium `mjData`
  (`self._bases`, l.186; pointers handed to the pool at l.257, valid for the pool's lifetime because gymnasium
  resets in place). `qpos`, `qvel`, `qacc_warmstart` are NumPy views.
- Stepping branches *through the training vector env* would corrupt its bookkeeping: `_episode_returns`,
  `_episode_lengths`, `TimeLimit._elapsed_steps`, `RecordEpisodeStatistics` queues, autoreset at the horizon
  (`step_wait` l.429-511), and `VectorObsNorm.normalize_step` would absorb branch states into the running
  statistics. Do not do this.
- The oracle already has the right primitive: `scripts/awr_oracle_advantage.py` `Physics` (l.84-235), a
  stateless threaded C kernel `oracle_step(model, scratch, threads, state_rows, ctrl, count, frame_skip)` over
  flat rows (qpos, qvel, qacc_warmstart), which "with ctrl is everything mj_step reads" for HalfCheetah (no
  actuator activations, mocap or applied forces). `data->time` is not part of the row and does not enter the
  dynamics. Asserted bitwise against gymnasium on observations and to 1e-6 on rewards
  (`validate_physics`; `tests/test_reppo_critic_oracle.py` exists).
- Throughput: 500,516 env steps/s at 12 threads in the v60 oracle run's own validation
  (`runs/reppo_critic_oracle/v60_base.json` [RAW]), about 42k steps/s per thread. The training env phase is
  0.134 s per 16,384 steps at 2 threads [DOC 06], about 61k steps/s per thread including vector-env overhead.
  So 80-120k branch steps/s at the standard 2 threads.

Compatibility:

- **Observation normaliser.** REPPO stores raw observations and renormalises every row with its env's frozen
  per-iteration statistics (`normalize`, `v60:437-439`). Branch observations are `concat(qpos[1:], qvel)` of
  the branch row and take the source row's env statistics; they never pass through `normalize_step`, so the
  running statistics remain those of the on-policy trajectory. Compatible.
- **Reward normaliser.** REPPO has none (fixed `reward_scale` 0.1). The branch reward is
  `(x_after - x_before) / dt - 0.1 |a|^2`, as `Oracle.raw_reward`. Compatible.
- **Episode-age phases.** Branches never touch the training envs, so `staggered_envs` phases, suppress masks
  and episode statistics are unaffected. A branch that would cross step 1000 can simply continue: HalfCheetah
  never terminates and the critic does not see time (the oracle's "soft" truth does the same).
- **Bit-determinism.** Kernel rows are independent, so results do not depend on thread scheduling.

Smallest faithful implementation on v60 (about 250 lines plus a shared module; the largest item here):

1. New `cleanrl/shared/mujoco_branch.py`: move `KERNEL_SOURCE`, `build_kernel`, `Physics.step` there (a trainer
   must not import from `scripts/`), plus `read_states(envs)` copying the 16 envs' (qpos, qvel, qacc_warmstart)
   into a (16, 27) float64 array. Pin with `tests/test_mujoco_branch.py`: K steps through the kernel equal K
   steps of `NativeMujocoVectorEnv` bitwise.
2. Rollout loop (`v60:755-769`): call `read_states` before `envs.step` and keep a (num_steps, 16, 27) host
   array. 16 small copies per step against ~257 us per step today.
3. After `insert` (`v60:781`): pick S start rows, draw K alternative first actions per state from the
   behaviour policy, and run n steps. **Reuse the real trajectory as branch 0**: the rollout already stores its
   noise (`rollout_noise`), so alternatives use `tanh(mu(s') + sigma(s') * eps_real[t + j])` for j >= 1. That is
   exact common random numbers against a branch that costs nothing. Actions come from one GPU actor forward of
   S x K rows per branch step (n host/device round trips per iteration).
4. Label: `y_k = sum_{j<n} gamma^j r_j + gamma^n V'(s_n)` with V' from `bootstrap_values` (`v60:447`), plus the
   soft bonus terms at intermediate states exactly as `refresh_targets` folds them in. Train the critic on the
   extra rows with target `G_real(s) + (y_k - y_0)`: the level label stays the row's own lambda-return and only
   the low-noise contrast is new. This avoids v34's confound (a second target type mixed into every minibatch).
   Branch rows have no successor rows in the rollout layout, so their `rollout_chain` mask is zero (they get
   the z_0 reward loss only).
5. Flags: `--branch-states S`, `--branch-actions K`, `--branch-steps n`, `--branch-use {critic,actor,both}`.
6. **Budget accounting.** One branch control step (5 `mj_step`s) is one timestep. `global_step += S K n` per
   iteration and `num_iterations = (total_timesteps - warmup) // (num_envs num_steps + S K n)` (`v60:364`).
   Episodic returns are logged against that `global_step`, so `score_runs.py --at` stays honest. Example at
   equal budget per iteration: 16 x 512 real steps + 512 states x 4 alternatives x 4 steps = 16,384.
7. Compute relative to v60: physics is step-for-step what the replaced real steps cost (about 0.1 s for 8,192
   branch steps at 2 threads); the tail V' is S K rows x 4 samples per refresh against 16,384 x 4 today;
   branch rows replace rollout rows inside the same 2,048-row minibatches. Net: about equal wall-clock.

Risks to state up front [INFER]: (a) halving the on-policy rollout halves the data behind the level fit, and
shorter rollouts were worse at 8M in every recipe (section 7), so an arm with the same shortened rollout and no
branches is a required control; (b) the tail relies on V' between nearby states; the +0.84 figure says it is
good enough at 1M, nothing is known later; (c) the label is on-policy for the behaviour policy only.

---

## 3. Hybrid actor gradient: score-function plus pathwise, or pathwise as a control variate

### (i) Prior versions [CODE]

| Version | Base | Estimator | Notes |
|---|---|---|---|
| v39 dualpg | v36, TF32, from scratch | `score = (1 - w) Q(s, a~pi) + w min(rho A, clip(rho, 1 +- 0.2) A)`, w = 0.5 (l.473-477). A = soft lambda-0.95 return minus E_{pi_old} Q_target (the critic's own baseline), **not normalised**. Same 32 actor steps of 2,048 uniform rows, same per-state KL gate | a plain weighted sum in return units, not a control variate; no PPO epoch structure, no trained V |
| v40 dualpg fork | v39 + `--init-snapshot` (v27 H8 at 8M); optimisers and KL multiplier restart | arms w = 0 (control), 0.5, 1.0; also kl 0.01 / 0.003 | TF32 [RAW] |
| v21 hybridgrad | v19 (Beta, no entropy) | mode: pathwise; precision nu: likelihood ratio with per-minibatch standardised A | LR used only for the spread |
| reppo_qprop_v1 | PPO sepclip control (Beta) + REPPO's Q | `clip(ratio)(A - eta A_bar) + eta pessimistic_clip(g^T (mu_new - mu_old))`, g = dQ/da at the rollout-time mean, eta = max(0, cov / var) per batch | PPO is the base; Q only supplies a first-order control variate |
| latent_qprop v1-v5 (top level) | residual SiTU-GLU nGPT PPO | first-order slope head g(s, L) regressing advantages (v3: GAE(0) for g, GAE(0.95) for PPO; v5: trajectory-wise control variates) | no Q critic |
| latent_ipg_v4 | latent-tilt PPO | `(1 - eta) clipped LR + eta d/dtheta g_snap(s, m + sigma eps)`, pathwise stopped outside the clip band | pathwise through a latent "tilt" critic U(s, T), not Q(s, a) |

### (ii) Outcomes

| Run | @1M | @2M | @4M | @8M | later | Flip? |
|---|---:|---:|---:|---:|---|---|
| v39 w0.5 | 1558 | 1678 | 2029 | - | | yes (98% inverted [DOC]) |
| v40 w0 (control) | | | | 11134 | 11407 @12M | no |
| v40 w0.5 | | | | 11140 | 10972 @12M | no |
| v40 w1 | | | | 11140 | 8678 @10M, cancelled 10.7M | no |
| v21 | 4423 | 5894 | 6533 | 7331 | | no (spread collapse) |
| reppo_qprop_v1 / noqprop | 2890 / 3076 | 4800 / 4879 | 7260 / 6607 | 9035 / 8011 | | no |
| latent_qprop v1 / nolatent v1 | 3208 / 2205 | 4911 / 3838 | 6958 / 4994 | 8998 / 5648 | v1 13565 at 48M | no |
| latent_qprop v3 / v5 | 1596 / 3105 | 4521 / 4831 | 7372 / 6452 | 9380 / 8400 | v3 14028 at 50M | no |
| latent_ipg eta 0 / 0.5 | 1541 / 1900 | 3404 / 4041 | 4878 / 4865 | 5854 / 5489 | | no |

All [TSV]. Diagnostics [RAW]:
- v40 w0.5: `pg_advantage_std` 5.2 / 5.0 / 2.3 / 2.7 at 8.3 / 9 / 10 / 11.8M; `pg_clip_frac` 0.48 -> 0.32;
  `pg_surrogate` -0.58, -2.54, -0.04, +0.02; policy std 0.83 at 8.3M (control 0.52); kl_violation 0.71 at 8.3M.
- reppo_qprop_v1: `qprop/eta` 0.065 / 0.031 / 0.023 / 0.018; `control_corr` 0.051 / 0.053 / 0.045 / 0.038;
  `fullq_corr` 0.054 / 0.063 / 0.059 / 0.043.

### (iii) Grade and status

- v39: C (flip). v40 w0.5: C (-0.4k inside fork noise, confounded as in section 0 item 5). v40 w1: B that a pure
  LR actor in the pathwise step regime degrades. v21: B- (LR on the spread alone still collapses).
- reppo_qprop_v1: C; the mechanism claimed cannot produce the gain (0.2% variance removed).
- latent_qprop / ipg: different base and no Q; they say a first-order control variate explains ~0% of A
  [DOC header: "held-out control EV ~0.2%"].
- **Status: CONFOUNDED / UNTESTED-IN-EFFECT** for "PPO-regime score-function term plus pathwise term" on the
  REPPO base. TESTED-NEGATIVE (B) only for pure LR at the pathwise step.

### (iv) Feasibility on v60

Port v39's pieces (about 60 lines): store the pre-squash action (`rollout_noise` and the old moments already
give it), a `pg_advantages` buffer filled once per iteration from `refresh_targets(0.95, ...)` minus a baseline,
and one extra term in `actor_loss` (`v60:495`). Confounds to remove relative to v39 / v40:
1. Standardise A per iteration (PPO does) and express w as a gradient-norm share, not an objective-unit share:
   with A sd 2-5 against Q contrasts of ~0.2, w = 0.5 in return units is an almost pure LR gradient that
   `max_grad_norm` 0.5 then rescales.
2. Baseline: E_pi Q carries the level error (sd 0.9, about 4x the true advantage sd [DOC 01]). It does not bias
   the gradient but adds variance; use the K-sample V' already computed for the bootstrap.
3. Do not restart the optimisers and the KL multiplier at the fork (v60's `--resume` restores both; v40's
   `--init-snapshot` did not).
4. Strict FP32.
5. The per-state gate hands outside-gate states the KL pull only; with a 0.2 ratio clip and a 0.03 KL gate two
   trust regions stack. Log the clip fraction inside the gate; if it stays ~30% the LR term is still not in its
   own regime.

Control-variate form (Q-Prop): the measured first-order explained variance is ~0.2% on the PPO base, so the
variance-reduction route has a weak prior at any cost. Compute: one extra log-density per actor step; nil.

---

## 4. Trust-region size and schedule

### (i) Prior versions

| Run(s) | Recipe [RAW hparams] | KL bound | Gate |
|---|---|---|---|
| reppo_v1, v1_ent1p5, v1_kl0p3 | v1: 16 x 256 steps, 2 minibatches x 8 epochs, coupled actor/critic rows | 0.1, 0.1, 0.3 | per state |
| v1_ppobatch_kl0p03_* (4 arms) | v1 moved to 16 x 1024, full batch or 8 x 4 epochs, lr 3e-4 / 1e-3 | 0.03 | per state |
| ngpt_v2, ngpt_beta_v3 | 16 x 256, lr 2.4e-3 | 0.1 | per state |
| v4 and everything from v6 on | 16 x 1024 | 0.03 | per state |
| v23 kl0p015, kl0p01 | v7 cu128, from scratch | 0.015, 0.01 | per state |
| v23 ns4096_kl0p12 | 16 x 4096 rows, 512 / 128 updates | 0.12 | per state |
| v35 window | 128-step segments | 0.03, 0.0075 | per state |
| v40 kl01, kl003 | fork from 8M, TF32 | 0.01, 0.003 | per state |
| v48 box / floor (on v47 Beta, no entropy) | box: gate on mode displacement per dimension and on d log nu; floor: cap on nu | box in action units | per state |
| v57 actiongeom (on v47) | `0.03 mean((mode - old)^2) / 0.02^2 + KL_precision` | action units | per state |
| v58 evidencestep (v55 dual) | whole-update multiplier in **weight space** chosen by an importance-weighted real-return line fit on the next rollout; grid 0..3 | 0.03, then extrapolated | per state |

`inside = kl < args.kl_bound` with `torch.where(inside, objective, kl * multiplier)` appears in every version
except v57 and qprop_v1 [CODE grep]. **A batch-mean gate has never been run.** The multiplier is already driven
by the batch-mean KL (`v60:503`).

### (ii) Outcomes

| Run | @0.5M | @1M | @2M | @4M | @8M | later | Flip? |
|---|---:|---:|---:|---:|---:|---|---|
| reppo_v1 (kl 0.1) | 2264 | 3023 | 3589 | 4036 | 4505 | | no |
| reppo_v1_kl0p3 | 1862 | 3297 | 3577 | 3613 | 4203 | | no |
| v1_ppobatch_kl0p03 ep4_mb8 / ep4_mb1 | 1041 / -3 | 2208 / 361 | 2396 / 274 | 3424 / 2698 | 4056 / - | | no |
| v1_ppobatch_kl0p03 lr3e-4 / lr1e-3 (10 full-batch epochs) | 137 / 548 | 1013 / 1701 | 1872 / 680 | 2604 / 1620 | - | ended 5.2M | no |
| v23 kl 0.03 | 2414 | 4583 | 6030 | 8014 | 10484 | 11342 @12M | no |
| v23 kl 0.015 | 1466 | 3420 | 4588 | 5704 | 7761 | 9612 @12M, 11023 end (33M) | no |
| v23 kl 0.01 | 1607 | 3619 | 4987 | 5418 | 7630 | 9109 @12M, 11169 end (33M) | no |
| v23 ns4096 kl 0.03 / kl 0.12 | 227 / 726 | 1421 / 3469 | 4162 / 5067 | 5771 / 5799 | 9313 / 7421 | 10299 / 8230 @12M | no |
| v40 kl 0.01 / 0.003 (from 8M) | | | | | | 11127 / 8071 @12M (control 11407) | no |
| v35 kl 0.03 / 0.0075 | 1526 / 1305 | 1771 / 1493 | 2119 / 1748 | 2226 / 2009 | 2314 / 2037 | | yes, both |
| v48 floor vs v47 indep | 3820 / 3767 | 5874 / 5721 | 7492 / 6986 | 8271 / 7975 | 9526 / 9084 | | no |
| v48 box | 10 | 205 | 839 | 2377 | - | | invalid: precision bound below the early narrowing rate [DOC] |
| v57 vs v47 indep | 2518 / 3767 | 4459 / 5721 | 5797 / 6986 | - | - | 3-minute trial | no |
| v58 | 3272 | 3817 | 4838 | 6023 | - | cancelled 4.6M | no |

All [TSV].

Runs that enlarged the *early* step, and what happened later:
- **16 x 128-step rollouts (`iterrate_v36_ns128`, FP32)**: same 0.03 KL per iteration on 1/8 of the data, i.e.
  8x the KL budget per sample; `policy/kl` 0.029-0.031 and `kl_violation` 0.33-0.37 throughout [RAW], so the
  larger budget was fully used. 3724 / 4968 / 5981 / 8606 / 9871 at 0.5 / 1 / 2 / 4 / 8M: **+1.2k at 0.5M, +0.8k
  at 1M, +0.3k at 2M, 0 at 4M, -1.2k at 8M** against path A. This is the only FP32 upright evidence, and it has
  the sign pattern the landscape data predict (step too timid early, too long late). It also changes data per
  critic fit, so it is not a clean step test.
- v23 ns4096 kl 0.12 (4x step on 4x data): +2.0k at 1M, +0.9k at 2M, 0 at 4M, -1.9k at 8M against its kl 0.03
  twin. Same sign pattern, v7 recipe.
- v45 mb2048 (4 heads, each 32 steps on a quarter of the data): +1.2k at 1M, +0.7k at 2M, level at 8M.
- v58: multiplier at the grid cap 3 early, then 1.1-1.5 with realised KL up to 9x the bound [DOC]: -0.4k at 1M,
  -2.6k at 4M. It extrapolates the whole update in weight space past the per-state gate, which is not the same
  as raising the bound.
- v1 kl 0.3 vs 0.1: level (4203 vs 4505) on a 4.5k recipe.

v60 base gate statistics [RAW]: `kl_violation` 0.65 at 0.3M, 0.39 / 0.40 / 0.32 / 0.33 at 1 / 2 / 4 / 7.8M;
`kl_multiplier` 0.06 -> 0.22 -> 0.78; pre-clip actor gradient norm 0.10 at 0.3M, 0.67 at 1M, 2.9 at 4M.

### (iii) Grade and status

- "Smaller KL does not help": B for 0.003 degrading late; C+ for 0.01-0.015 from scratch being slower to 8M and
  equal at 33M; the 0.01 fork neutral (TF32, B- null).
- "Larger early step helps early and hurts late": C+ (three single-trajectory arms on three recipes agree in
  sign; each is confounded with data per fit).
- v48 / v57 action-unit geometry: C; both on the entropy-free Beta base, v48 box invalid, v57 miscalibrated by
  5.7x by its authors' own analysis. v58: B- that weight-space extrapolation regresses.
- **Status: KL 0.1 on the modern recipe NEVER TRIED; a decaying KL schedule NEVER TRIED; batch-mean gate NEVER
  TRIED; action-unit step CONFOUNDED / UNTESTED-IN-EFFECT; smaller constant KL TESTED-NEGATIVE to neutral.**

### (iv) Feasibility on v60

- Constant bound: `--kl-bound` exists; zero lines; valid on `--resume` (the multiplier is restored from the
  checkpoint and adapts at 0.15 log units per iteration).
- Schedule: `args.kl_bound` is a Python float inside the compiled `actor_loss` (`v60:497`, `:503`), so it is a
  constant of the captured graph. Make it a device tensor exactly as `target_entropy` is (`v60:397`, filled at
  `:753`); about 10 lines plus `--kl-bound-final`, `--kl-anneal-steps`. A fork-window flag in the style of
  `--fork-ent-target-mult` / `--fork-iterations` gives "larger step for N iterations, then back to control",
  which measures the learning effect and the persistent effect separately.
- Batch-mean gate: replace `torch.where(inside, objective, kl * multiplier)` by
  `objective + multiplier * kl` with the multiplier's existing dual (3 lines, flag `--kl-gate {state,mean}`).
- Compute: none.

---

## 5. Noise control

### (i) Prior versions [CODE headers; hparams RAW]

| Version | Idea | Precision / base |
|---|---|---|
| v1_ent1p5, v27 ent 1.0 / 1.5 | lower fixed entropy target from scratch | v1; v27 FP32 |
| v3, v18 | Beta head (softplus on nGPT base; mode / softplus on v7 cu128), closed-form entropy and KL | v2 / v7 |
| v19 | Beta, no entropy term | v7 cu128 |
| v20 | v19 + TD7-style exploration budget: dual on mean log std >= log 0.1, confined to the actor | v19 |
| v21 | v19 + LR gradient for the precision | v19 |
| v22 | clipped Gaussian, budget on pre-clip sigma 0.1 / 0.2 | v20 |
| v31 pricecap | temperature capped at 0.02 / 0.03; identical to path A until the cap binds | v27, FP32 |
| v42 entropy forks | target -1.0 / -1.5 / -2.0 per dim from the 8M snapshot | TF32 fork |
| v43 entanneal | -0.5 for 10% then linear to -1.0 / -2.0; constant 0 / -0.25; 0 until 50% then -1.5; -0.5 until 50 / 60% then -1.5 / -2.0 | TF32 except the two `fp32_s50` arms |
| v44 sharpmean | mean climbs dQ/da at tanh(mu); std keeps the sampled objective | FP32 |
| v45 / v46 | population of 4 heads; late drop with and without collapse | FP32 |
| v47 crossspread | Beta, no entropy; spread reads an independent critic | FP32 |
| v49 drifttemp | T = c x rms contrast drift | FP32 |
| v54 / v56 IDS | T from offset-regret balance; calibrated by beta | FP32 |
| v60 noise fork | perturbation-window forks on the entropy target | parked, never run |

State-dependent vs global std: every REPPO actor outputs a per-state log std (or Beta concentration). v20 / v22
hold the *mean* log std with a dual but keep it state dependent. **A state-independent std has never been run.**

### (ii) Outcomes

| Run | @1M | @2M | @4M | @8M | later / det. eval [DOC] | Flip? |
|---|---:|---:|---:|---:|---|---|
| v18 mode / softplus | 5403 / 5062 | 5986 / 5452 | 6790 / 6036 | 8870 / 7708 | | no |
| v3 | 2456 | 2214 | 1802 | 643 | collapsed; lr 2.4e-3, KL 0.1 | - |
| v19 | 5223 | 5952 | 6547 | 7027 | std 0.21 -> 0.046 | no |
| v20 / nosmooth | 5186 / 5217 | 6124 / 6083 | 7564 / 6929 | 8508 / 8325 | | no |
| v21 | 4423 | 5894 | 6533 | 7331 | entropy -8.4 | no |
| v22 s0.1 / s0.2 | 5187 / 5259 | 5808 / 5643 | 6254 / 6894 | 7511 / 8025 | 7945 / 8613 @12M | no |
| v27 ent 1.0 / 1.5 | 3688 / 3728 | 3999 / 3089 | - | - | culled 2.5M | no |
| v31 cap 0.02 / 0.03 | 4174 / 4174 | 5767 / 5724 | 8800 / 8720 | 10832 / 11373 | 12432 / 12308 at 16M (ref 11400) | no |
| v42 ent -1.0 / -1.5 / -2.0 | | | | | 12291 / 12178 / 12399 @12M (control 11408); det. 12.8k / 12.1-12.8k / 13.3-13.7k vs 12.0-12.4k | no |
| v43 -1.0 / -2.0 (TF32) | 3832 / 3863 | 4922 / 5005 | 5958 / 6029 | 10153 / 6936 | | no |
| v43 0 / -0.25 fixed (TF32) | 3571 / 3233 | 4705 / 4785 | 6797 / 7564 | 8500 / 10067 | | no |
| v43 0 until 50% then -1.5 (TF32) | 3571 | 4705 | 6717 | 11571 | last-20 11307; det. 12.2k | no |
| v43 s50 -> -1.5, s60 -> -2.0 (TF32) | 3847 | 4799 | 6045 / 6036 | 9974 / 10401 | | no |
| v43 FP32 s50 -> -1.5 / -2.0 | 1947 | 2157 | 2254 / 2243 | 2537 / 2539 | | yes, before the drop |
| v44 / v44 const | 3944 | 4803 | 5624 / 5635 | 6645 / 7400 | | no |
| v46 collapse / no collapse (late drop) | 5347 | 6347 | 8880 / 9173 | 10941 / 11020 | det. 12.4k / 11.7k | no |
| v47 crossfold / indep / same | 6375 / 5721 / 5203 | 7186 / 6986 / 5842 | 7574 / 7975 / 6349 | 8259 / 9084 / 7046 | | no |
| v49b c1 / c3 | 2316 / -280 | 3626 / -262 | 2761 / -251 | - | | no |
| v54 belief / lag | 4019 / 3010 | 4762 / 4673 | 7495 / 6677 | 9159 / 8720 | | no |
| v56 | 3615 | 4943 | 6890 | 9736 | | no |
| v55 ids | 1626 | 1794 | 2116 | 2286 | | yes |

All [TSV].

### (iii) Which gave a robust positive

- **Lowering the entropy target after 8M: +0.8-1.0k stochastic at 12M in all three arms, B+** (01). Caveats: one
  shared snapshot, TF32, gain measured outside the 8M budget.
- **Price cap (v31): +0.9-1.0k at 16M, B-**; at 8M -0.2k / +0.3k, i.e. nothing.
- **Is there a variant that reduced late noise without collapsing early learning? Yes, two, but neither shows a
  gain inside 8M.** v31 is identical to the reference to ~2M by construction and never falls behind. v46
  (population + drop at 50%) ends at 10941 / 11020, level with the reference. v43 `m0_late1p5` ends at 11571
  (last-20 11307) but on the TF32 base and from a higher-entropy first half; its deterministic return equals
  the reference's, so the gain is removed evaluation noise. The FP32 late-drop arms flipped before the drop.
- Everything that sets the temperature from a measured rule (v49, v54, v56) or removes the entropy term (v19,
  v21, v47) is negative at B- to B. Sharp mean (v44) is negative at B-.
- Size of the prize [DOC landscape]: noise cost J(0) - J(1) is 5.0% at 1M, 12.3% at 4M, 11.0% at 7.9M;
  deterministic eval is ~12.0k against ~11.0k stochastic at 8M. At most ~1k is available at 8M from noise alone.

Status: late entropy reduction **TESTED-POSITIVE after 8M, UNTESTED-IN-EFFECT inside 8M on an upright FP32
trajectory**; measured-temperature rules and entropy-free actors **TESTED-NEGATIVE**; Beta vs tanh-Gaussian
**CONFOUNDED** (v18 -1.3k / -2.4k on the v7 recipe with a different early profile: +0.8k at 1M); global std
**NEVER TRIED**.

### (iv) Feasibility on v60

- Permanent late drop from a checkpoint: zero lines. `target_entropy` is refilled every iteration from
  `args.ent_target_mult` (`v60:750-753`), so `--resume checkpoint_4013696.pt --ent-target-mult 1.0
  --iterations 243` is a valid arm and `--ent-target-mult 0.5` its control. The dual temperature is restored and
  moves at 0.15 log units per iteration.
- Price cap: port v31's clamp on the dual temperature (about 5 lines, flag `--temperature-cap`).
- Global std: a learned `log_std` parameter in place of the second half of the actor output; touches `Actor`,
  `policy_moments` and `HostSquashedGaussian` (the host mirror then outputs the mean only); about 30 lines.
- Compute: none.

---

## 6. Per-state normalisation or clipping of dQ/da; advantage-standardised actor objectives

### (i) Prior versions

No version clips, winsorises or normalises the per-state action gradient in the actor loss (grep over
`cleanrl/reppo`, [CODE]). Adjacent:

| Version | What it normalises [CODE] | Difference |
|---|---|---|
| v15 distq_diff | `q = diff_scale * critic.difference(...)`, a running RMS over the batch | one global scale |
| v38 | A net trained on scale-normalised branch contrasts, mixed as `(1 - w) q + w A` | global scale; learned-model labels |
| v51 / v53 | actor climbs `contrast_scale * critic.contrast`, D trained on (G - E_pi Q') / s_y | global standardisation of the *target*, new head |
| v63 | Q = V(s) + A(s, z), A quadratic in the behaviour noise z | structural, not a clip |
| v21 | per-minibatch standardised A for the LR precision term | LR term only |
| all | `max_grad_norm` 0.5 on the parameter gradient; per-state KL gate | global clip; the gate removes a state from the objective once it has moved 0.03 nats, which already bounds how far any one state is pushed |

### (ii) Outcomes

Nothing to score for the direction itself. First-round measurement [DOC 02 L2, `audit_actor_gradient_tails.py`]:
the top 1% of on-policy states hold 80% / 73% / 75% of sum |sigma dQ/du|^2 at 1 / 4 / 7.9M, max-to-median above
100; but the actor parameter gradient without the top 0.5% of states has cosine 0.96-0.98 with the full one,
and the top 10 rows of a 2,048-row minibatch contribute 0.1-0.7 of its norm (cosine with and without them
0.75-0.99). 02's own prediction for a 99th-percentile clip: small.
Related outcomes: v51 +0.2k at 1M, +1.3k at 2M, then a stall (8445 at 4M, 7608 at 6M); v63 fork 4876 / 5783 at
2 / 3M against control 5658 / 7896; v38 above.

### (iii) Grade and status

**NEVER TRIED** as specified. Prior expectation small (C+ offline measurement at three checkpoints of one run).
"Advantage-standardised objective" exists only as new heads (v38, v51, v63), each negative or fragile at B- / C+
for reasons other than the standardisation.

### (iv) Feasibility on v60

The clean form keeps the batch-mean scale so the balance between Q and `T log pi` (`v60:495`) is unchanged:
1. In `actor_loss`: `g = grad of critic.value(obs, actions).sum() wrt actions` (detached), winsorise the
   per-state norm at a batch quantile, and use the first-order surrogate `-(g_clipped * actions).sum(-1)` in
   place of `-q`. With no clipping this has exactly `-q`'s gradient.
2. `actor_loss` is wrapped by `graph_compile` (`v60:618`); an inner `torch.autograd.grad` may not trace. The
   existing `action_gradient` (`v60:509`) runs it eagerly in diagnostics. Safest: compute `g` in an uncompiled
   prefix of `actor_step` and pass it into the compiled loss; CUDA-graph capture (`capture_steps`, `v60:848`)
   records kernels and is indifferent to that. This needs a short check that the captured replay stays
   deterministic.
3. Flags: `--dqda-clip-quantile 0.99` (0 disables). About 30 lines.
4. Per-state *unit* normalisation is a different algorithm (it discards the magnitude that sets the entropy
   balance) and should not be the first arm.
5. Compute: the critic backward-to-action already happens once per actor step; splitting it out adds at most
   one extra critic forward per actor step (32 per iteration), under 3% of the update [INFER].

---

## 7. Data per policy evaluation: bigger rollouts / lower KL, and the mirror

### (i) Prior versions [RAW hparams]

| Run | Envs x steps | Critic / actor updates | Recipe |
|---|---|---|---|
| v9_live_ns128 / ns256 / ns512, v7, v9 ns2048 | 16 x 128 ... 2048 | 4 / 4, 8 / 8, 16 / 16, 32 / 32, 64 / 64 (passes fixed at 4) | v7, lambda 0.95 |
| v23 ns4096, ns4096_r2, ns4096_kl0p12 | 16 x 4096 | 512 / 128 | v7 cu128 |
| v25 lam0p8_ns4096 | 16 x 4096 | 16 passes | v25 |
| iterrate_v36_ns128 / ns256 | 16 x 128 / 256 | 16 passes | v36 FP32 |
| v36_e128 | 128 x 128 | 16 passes | v36 **TF32** |
| v1_ppobatch, v4 arms | 16 x 1024, full batch or 8 minibatches | 4-10 epochs, coupled rows | v1 / v4 |
| v23 kl0p015 / kl0p01 | 16 x 1024 | 128 / 32 | lower KL = more data per nat |

### (ii) Outcomes

| Run | @0.5M | @1M | @2M | @4M | @8M | later | Flip? |
|---|---:|---:|---:|---:|---:|---|---|
| v9 ns128 | 128 | 1478 | 2693 | 4227 | - | ended 4.1M | no |
| v9 ns256 / ns512 | 1159 / 1630 | 1777 / 2234 | 3268 / 4265 | 5614 / 6005 | 7077 / 7219 | | no |
| v7 (ns1024) | 1547 | 2585 | 3319 | 6218 | 8152 | | no |
| v9 ns2048 | 300 | 1496 | 3046 | 3545 | 4094 | | stall band, not the 2.3k plateau |
| v23 ns4096 vs v23 | 227 / 2414 | 1421 / 4583 | 4162 / 6030 | 5771 / 8014 | 9313 / 10484 | 10299 / 11342 @12M | no |
| v25 ns4096 vs v25 | 614 / 1395 | 2781 / 1842 | 4747 / 3467 | 5674 / 4920 | 8106 / 9282 | 13392 / 13317 at 50M | no |
| iterrate ns128 vs path A | 3724 / 2521 | 4968 / 4174 | 5981 / 5634 | 8606 / 8596 | 9871 / 11031 | | no |
| iterrate ns256 | 2478 | 2887 | 3864 | 4398 | - | culled 5.8M, last-20 5949 | "flip-like" in FAMILY, but it was still rising |
| e128 (TF32) | 816 | 3342 | 4704 | 5933 | - | cancelled 5.4M | no |

All [TSV]. Held-out metric [DOC 06]: 64k-row rollouts give `advantage_corr_heldout` 0.14-0.20 against 0.06-0.11
at 16k rows; nothing else in the archive moves it.

### (iii) Grade and status

C / C+ throughout: each comparison is one trajectory against one, the ns128 / ns256 pair is non-monotone, and
rollout length changes data per fit, iterations per sample and (at fixed KL per iteration) step per sample all
at once. Robust-looking regularities: short rollouts are faster to 0.5-1M and lower at 8M; 4x rollouts are
slower early, lower at 8M and equal at 50M.
**Status: CONFOUNDED / UNTESTED-IN-EFFECT.** Not tested at all: more data per evaluation *late only* (rollout
length growing over training), and KL per sample held fixed while rollout length varies.

### (iv) Feasibility on v60

Zero lines: `--num-steps` with `--critic-updates` / `--actor-updates` scaled to keep passes (128 / 32 per 16,384
rows; `critic_updates % target_refresh == 0` and `% actor_updates == 0` are enforced, `v60:320-329`). A fork may
change `--num-steps`: the checkpoint holds no rollout buffers, only networks, optimisers and per-env statistics
(`num_envs` must stay 16). So "fork from 4M with 16 x 4096" and "fork from 1M with 16 x 256" are valid arms
today. Compute per env step is unchanged.

---

## 8. Critic-side budget and auxiliary loss

### (i) Prior versions and outcomes [TSV; update seconds DOC 06]

| Lever | Runs | @1M | @4M | @8M | Notes |
|---|---|---:|---:|---:|---|
| Critic updates 32 / 64 / 128 / 256 (v7 recipe) | v7, cu64, cu128, cu256 | 2585 / 3415 / 4583 / 4132 | 6218 / 7579 / 8014 / 7696 | 8152 / 9591 / 10152 / 9819 | never re-swept on the H8 + refresh recipe |
| Actor updates 16 / 32 (cu128); 64 (cu64) | au16, cu128, cu64_au64 | 3426 / 4583 / 3814 | 7584 / 8014 / 6757 | 9493 / 10152 / 9142 | |
| No aux loss (v7) | v7_q_noaux | 2421 | 4034 | 6861 | vs 8152 |
| Aux horizon 1 | v25 lam0.8 = v29 parent | 1842 | 4920 | 9282 | 11091 @16M |
| Aux horizon 4 + reward | v27_full | 3907 | 3817 | 5020 | stall band 3.7-5.5k for 16M steps |
| Aux horizon 4, no reward | v27_norew | 3632 | 5274 | 9104 | 10980 @16M |
| Aux horizon 8 + reward | v27 H8 (path A) | 4174 | 8596 | 11031 | 11400 @16M |
| Aux horizon 8, reward coef 0 | v29 noreward | 4682 | 7637 | 10274 | 11691 @16M |
| Width 1024 (v7 cu32) | v7_q_cw1024 | 1570 | 2187 | 2347 | flip plateau |
| Fresh 512 / fresh 1024 x 3 under the 8M actor | v42 | | | | 11130 / 10409 @12M (control 11407); TF32 |
| Twin critics from 8M | v41 first / mean / min | | | | 11005 / 10924 / 11401 @12M; TF32 |
| Scalar MSE head (v7 cu128) | v14_r1, 50M rerun, rewnorm | 4358 / 4358 / 4547 | 6333 / 6333 / 6696 | 9571 / 9653 / 9007 | plain MSE confirmed at l.345 [CODE]; vs 10152 |
| Scalar Huber head | v59 | 3573 | 7457 | 8520 | Huber, not MSE |
| TF32 | path F (v43 s50 before its drop) | 3847 | 6045 | - | one trajectory |
| BF16 | speed ladder | | | | 800k steps only |
| Minibatch size | v1 / v4 full-batch arms; v45 512 vs 2048 rows per head | | | | never isolated |

06 measured: H = 8 roll plus reward head is ~69% of the 0.436 s update (roll ~51%, reward ~18%); H8 is 2.0-2.5x
the H1 parent at 1M and +1.7k at 8M; 16 -> 32 passes lowers the training loss without raising the return.

### (iii) Grade and status

- Aux loss present vs absent: B- (3.3k, v7). H8 vs H1: C (+1.7k at 8M, one trajectory each; H4 + reward at 5.0k
  makes the 2 x 2 non-monotone with a 6k range, 01 item 7).
- Passes: C+ on the v7 recipe; untested on the reference recipe.
- Width / capacity: C (from scratch it flipped; the fork tests a fresh critic under an old actor, in TF32).
- Twin: B- null (TF32 fork, second member fresh).
- Scalar MSE: C+ null (-0.6k at 8M, -1.7k at 4M, single trajectory, v7 recipe).
- Precision: C+ (one TF32 trajectory).
**Status: aux horizon, passes, minibatch size, capacity and precision are CONFOUNDED / UNTESTED-IN-EFFECT on the
v60 recipe; twin critics and the scalar head are TESTED-NEUTRAL at low power; Huber head uninformative.**

### (iv) Feasibility on v60

All flags exist: `--aux-horizon {1,4,8}`, `--reward-loss-coef 0`, `--critic-updates`, `--actor-updates`,
`--minibatch-size`, `--target-refresh`, `--matmul-precision high`, `--bf16`. All are valid on `--resume` except
`--critic-hidden-dim` (the checkpoint's critic must load). `--aux-horizon` changes no parameter shapes (the
`dynamics` module is shared across steps). Compute per 06: H1 0.17-0.19 s update (predicted), TF32 0.28 s,
BF16 0.19 s against 0.436-0.468 s. Twin and big critics need v41 / v42 code ported (about 80 lines each) and
cost about 2x and 4x the critic share.

---

## 9. Replay-flavoured, policy-evaluation-correct data reuse

### (i) Prior versions [CODE headers; hparams RAW]

| Version | Recipe | Target | Buffer | Outcome [TSV] | Flip? |
|---|---|---|---|---|---|
| v5 | v4, 1-step, target frozen per iteration | r + gamma [Q_target(s', a'~pi) - T log pi] | 1M / none / actor replay | 2.3-2.4k all arms incl. no-buffer | all flat at the plateau |
| v6 | v4 + lambda 0.95 over the buffer, **uncorrected** chains | Peng-style | 1 / 8 / 61 rollouts | 8152 / 4709 / 1978 @8M | no / no / plateau-like |
| v10 retrace | v7 cu32, no refresh | taken-action baseline Q'(s', a_{t+1}), c = lambda min(1, pi/mu) | 1 / 4 / 8 | 4194 / 3408 / 2038 | no / no / late slide |
| v11 vtrace | v7 **cu128**, lambda 0.95, no refresh, 1-step aux | sampled-action baseline, same trace | 1 / 2 / 4 / 8 | 6633 / 6359 / 9226 / 9013 @8M (last-20 6544 / 6315 / 9190 / 9649) | no |
| v11 retrace_buf1 | as v11 | v10's baseline | 1 | 2367 | plateau |
| v24 tdreplay | v23 actor; twin HL-Gauss, 1-step, 64-rollout replay, Polyak 0.005, 512 updates | min of twins, K-sample expectation | 64 | **409k steps**, 6.9k SPS [RAW] | not evaluable |
| v32 replaytwin | v27 critic, 1-step twin-min, Polyak | as v24 | 64 | 2349 @4M | yes |
| v33 mixtarget | v32 + half fresh lambda-0.8 rows | mixed | 64 | 1769 @1M, cancelled 1.2M | not evaluable |
| v34 replayaug | v27 exactly + 1,024 one-step replay rows in every minibatch | mixed target types | 64 | 4839 / 5562 @4 / 8M | no (upright) |
| v35 window | 128-step segments, critic on an 8-segment window, 16 updates | on-policy lambda | 8 segments | 2226 / 2009 @4M | yes, both |
| v52 replaywindow | v51 (v36 FP32 + contrast head); V-trace soft targets with E_pi Q' baseline, recomputed over the window at every refresh, critic only | W = 4 | 4 rollouts | actor on D 1771 / 3106, actor on Q 1282 / 2272 at 0.5 / 1M; cancelled 1.2M | not evaluable; the Q arm's numbers are close to the flipped runs' (1353-1380 / 1884-1947) |

v11's replay arms were behind until ~5M (buf8 1325 at 1M vs 3871 for buf1) and overtook late; v52 was cancelled
exactly there.

### (iii) Grade and status

01's grades stand: v11 replay B- positive within its own control; v10 B that it fails (now explained as
unrefreshed taken-action targets, section 0 item 7); v34 B- that this mix hurts, C about replay; v5 / v32 / v35
flips; v24 / v33 / v52 no information.
**Status: CONFOUNDED / UNTESTED-IN-EFFECT.** TESTED-NEGATIVE only for uncorrected replayed chains (v6, B) and for
one-step rows mixed into lambda-return minibatches (v34, B-).

### Arms that deserve a clean re-test, and the protocol

1. **V-trace window W = 1 / 4 / 8 on the reference recipe, to 8M.** v52 already has the code on the v36 FP32 path
   (`--actor-signal q`); what it lacks is v60's checkpoint harness. Port the window into a v64 that keeps
   `--resume`, fork all three from `checkpoint_1015424.pt` (the buffer fills over the first W iterations), and
   do not judge before 6M. W = 1 must reproduce the control fork bit-for-bit; that is the implementation check
   v11 never had (its buf1 is 3.5k below v7 cu128 with matching hyperparameters and a logged pi/mu of
   0.988-0.997 [DOC 01]; one candidate cause visible in the target is that `min(1, pi/mu)` truncates only the
   upward half of a ratio that is 1 up to float error, which lowers the effective lambda slightly; too small to
   explain 3.5k on its own [INFER]).
2. **Retrace with in-iteration refresh** (v61's target with c = lambda min(1, pi/mu), over the same window). v61
   shows the taken-action baseline is viable once targets are refreshed (9.7k from scratch, 10.6k as a fork);
   v10's failure does not apply. One extra arm of the same v64.
3. **v24 TD7-regime critic.** It cannot load a v60 critic (twin, different schedule), so a fork needs v42-style
   actor-only initialisation plus a critic warmup, which adds a restart confound; run it from scratch with
   several RNG streams instead, to 4M. Cost is the obstacle: 512 twin updates per iteration ran at 6.9k SPS, so
   4M steps is about 10 minutes solo and it should run alone.
4. Not worth re-running: v5 (one-step with a per-iteration frozen target), v32 / v33 (superseded by v24),
   v34 (target-type mix), v35 (superseded by the v52-style window), v6 (uncorrected).

Compute for the window on v60 [INFER; v52 measured 4.9k SPS at 4 concurrent]: the targets phase and
`encode_targets` scale with W (bootstrap over W x 16,384 rows x 4 samples, 8 refreshes per iteration); the 128
critic steps do not. Expect roughly +30-60% update time at W = 4.

---

## 10. n-step / lambda choices; slope horizon vs level horizon; exact targets

### (i) Prior versions

| Version | What varies [CODE] | Outcome [TSV] | Flip? |
|---|---|---|---|
| v7 lam 0.8 / 0.95 / 0.99 | lambda, once-per-rollout targets, cu32 | 2327 / 8152 / 3308 @8M | 0.8: plateau |
| v25 lam 0.5 / 0.8 / 0.95 | lambda with in-iteration refresh and K-sample bootstrap, 1-step aux | 2379 / 9282 / 8933 @8M; - / 13317 / 11936 at 50M; @1M 1798 / 1842 / 4019 | 0.5: plateau |
| v27 and later | lambda 0.8 only | | |
| v26 duel | V + scalar A, same lambda | 3.5-4.1k | no; action dependence collapsed |
| v51 contrastdist | separate categorical head D on (G_0.8 - E_pi Q') / s_y; actor climbs D; Q keeps the level | 4355 / 6971 / 8445 @1 / 2 / 4M, then 7608 @6M | no |
| **v53 contrastlambda** | **D's own lambda 0.0 / 0.5 / 0.95 with Q's targets unchanged at 0.8** | 3425 / 3512 / 6661 @3M against 8343 for D at 0.8 | no; lam 0.0 / 0.5 look half-stalled |
| v37 | critic also fits model-imagined CRN branch returns, H = 5 | flip | yes |
| v38 | separate A on model branch contrasts, H = 12, lambda 0.95 | 3155 @4M | partly |
| latent_qprop_v3 (PPO base) | slope head regresses GAE(0); PPO's advantages stay GAE(0.95) | 9380 @8M, 14028 at 50M (v1 with GAE(0.95) for both: 8998, 13565) | no |
| v61 exacttarget | `G_t = r_t + gamma [V'_{t+1} + lam (G_{t+1} - Q'(s_{t+1}, a_{t+1}))]`, implemented as GAE with values = Q'(s, a_taken) | control flip, lam 0.8 flip, lam 0.95 9908 (last-20 9715) | 2 of 3 |
| v62 exactfork | the same, resumed from v60's 1M checkpoint | control 10325, lam 0.8 10590, lam 0.95 9608 @8M; last-20 10174 / 10458 / 10570 | no |
| v63 orthoadv | Q = V + quadratic A in the behaviour noise; with and without exact targets | fork 4876 / 5783 @2 / 3M; +exact 5836 / 5258 @3M; control 5658 / 7896 | no |

### (iii) Grade and status

- Lambda: C. On the H8 reference recipe only 0.8 has ever been run in uncorrected form; on v25 0.95 is +2.2k at
  1M, -0.35k at 8M and -1.4k at 50M against 0.8, one trajectory each. **CONFOUNDED / UNTESTED-IN-EFFECT.**
- Separate slope horizon: **TESTED, C+ negative, in exactly one form** (v53: a second head on standardised
  contrasts, shorter lambda worse at 3M, single trajectories). The PPO-base analogue (qprop_v3) is +0.4k at 8M,
  inside noise. Not tried: a slope critic regressing a short-horizon target that bootstraps from a *separately
  trained* level critic. Honest prior for that variant: weak. With one action per state, shortening the horizon
  trades future-action noise for V' error, and the offline table says one true step then V' is poor on all
  states (pooled 0.055; split-half correlation of the 1-step truth with the 400-step truth 0.06) while five
  steps are good (0.87). The lambda 0.8 mixture already puts 67% of its weight on n <= 5 [DOC 02]. The
  low-noise short-horizon slope label that the data support is the vine label of section 2, not a different
  lambda.
- Exact / Q(lambda) targets: C (01 claim 3). The correction subtracts the critic's own Q'(s_{t+1}, a_{t+1}), so
  it removes at most rho^2 of the future-action noise; `q_mse` did not fall. **CONFOUNDED / UNTESTED-IN-EFFECT**
  as a test of "low-variance targets"; as a target in its own right it is neutral within fork noise.

### (iv) Feasibility on v60

- Lambda: `--gve-lambda` exists and is valid on `--resume`; zero lines.
- Exact targets: use v62 as is (it carries the same resume harness, `resume` / `iterations` at l.159-162).
- Two-critic slope / level split: a second `Critic` and optimiser, a second target buffer filled by
  `refresh_targets(lam_slope, ...)` with `bootstrap_values` reading the level critic, the actor reading the
  slope critic; about 80 lines and roughly 2x the critic share of the update unless the slope critic drops the
  H8 roll.

---

## (a) Direction x status matrix

| # | Direction | Closest prior | Best evidence grade | Status | Lines on v60 | Compute vs v60 | Usable on `--resume` |
|---|---|---|---|---|---:|---|---|
| 1 | EMA / averaged critic for the actor | v9 (lag), v50 (loss anchor), v41 (twin mean) | C | **NEVER TRIED** | ~25 (EMA); ~80 (K snapshots) | +<1%; +10-25% | yes |
| 2 | Real-simulator vine labels | v37, v38 (learned model) | C / C+ (model); offline 5-step corr 0.87 | **NEVER TRIED** | ~250 + shared module + test | ~equal under honest accounting | yes |
| 3 | Hybrid score-function + pathwise | v39, v40, v21, qprop_v1 | C; B for pure LR negative | CONFOUNDED / UNTESTED-IN-EFFECT | ~60 | nil | yes |
| 4a | KL 0.1 or decaying KL on the modern recipe | v1 kl0p3, v23 ns4096 kl0p12, ns128 | C+ | **NEVER TRIED** | 0 / ~10 | nil | yes |
| 4b | Smaller constant KL | v23 kl 0.01 / 0.015, v40 forks | B (0.003), C+ | TESTED-NEGATIVE to neutral | 0 | nil | yes |
| 4c | Batch-mean gate | none | - | **NEVER TRIED** | 3 | nil | yes |
| 4d | Step in action units | v48 box, v57, v58 | C; B- (v58) | CONFOUNDED / UNTESTED-IN-EFFECT | ~40 | nil | yes |
| 5a | Late entropy reduction | v42 forks, v31, v43, v46 | B+ after 8M | TESTED-POSITIVE late; UNTESTED inside 8M in FP32 | 0 / 5 | nil | yes |
| 5b | Measured temperature rules, entropy-free actors, sharp mean | v19, v21, v44, v47, v49, v54, v56 | B / B- | TESTED-NEGATIVE | - | - | - |
| 5c | Beta vs tanh-Gaussian | v3, v18 | C+ | CONFOUNDED | large | nil | no |
| 5d | Global (state-independent) std | none | - | **NEVER TRIED** | ~30 | nil | no (actor shape) |
| 6 | Per-state dQ/da clip / normalisation | none (v15, v38, v51 global scales) | offline C+ "small" | **NEVER TRIED** | ~30 | +<3% | yes |
| 7 | Data per policy evaluation | v9 ns, v23 / v25 ns4096, ns128 / 256, e128 | C / C+ | CONFOUNDED / UNTESTED-IN-EFFECT | 0 | nil per step | yes |
| 8a | Aux horizon, passes, minibatch, precision | v27 2 x 2, v7 cu sweep, path F | C / C+ | CONFOUNDED / UNTESTED-IN-EFFECT | 0 | 0.4x-1x | yes |
| 8b | Twin critics; scalar MSE head | v41; v14 | B- null; C+ null | TESTED-NEUTRAL (low power) | ~80 | ~2x critic; ~1x | twin: partly; scalar: no |
| 8c | Capacity | v7 cw1024, v42 | C | CONFOUNDED / UNTESTED-IN-EFFECT | ~10 | up to 4x critic | no |
| 9 | Off-policy-correct replay window | v11, v52, v24, v10 | B- positive (v11) | CONFOUNDED / UNTESTED-IN-EFFECT | ~200 (port v52) | +30-60% update | yes |
| 10a | Lambda on the reference recipe | v7 / v25 sweeps | C | CONFOUNDED / UNTESTED-IN-EFFECT | 0 | nil | yes |
| 10b | Slope horizon separate from level horizon | v53, qprop_v3 | C+ negative | TESTED-NEGATIVE in one form | ~80 | ~1.3-2x critic | yes |
| 10c | Exact Q(lambda) targets | v61, v62 | C | CONFOUNDED (variance not reduced) | 0 (v62) | +4 ms | yes |

---

## (b) Top 5 by (untested in effect) and (cheap on v60)

1. **Averaged critic for the actor (direction 1).** Never tried, ~25 lines, no compute, loads the existing
   checkpoints. Confounds to avoid: do not anchor the critic's loss (v50); leave targets on the live critic;
   read `probe/slope_beta` and the oracle on the EMA critic rather than `advantage_corr_heldout`; sweep the
   memory (1 / 3 / 10 iterations) because lag bias grows with it; gradient-norm changes are invisible behind
   the clip and the KL gate, so only direction can matter.
2. **Early step size: KL 0.06 / 0.1 for a bounded window from the 1M checkpoint, then back to 0.03
   (direction 4a).** Never tried on this recipe; zero lines for a constant bound, ~10 for the window. Confounds:
   do not change rollout length at the same time (ns128 and ns4096-kl0.12 both did); do not extrapolate in
   weight space (v58); report the effect at the end of the window and again after the return to control,
   because all three confounded precedents show an early gain that reverses by 8M; watch `kl_violation` and
   policy std for a blow-up like v40's.
3. **Late entropy reduction inside the 8M budget, strict FP32, from the 4M checkpoint (direction 5a).** The only
   B+ positive in the record, never reproduced inside 8M on an upright FP32 trajectory. Zero lines
   (`--ent-target-mult 1.0 / 1.5` on resume) or 5 lines (price cap). Confounds: TF32 (all earlier forks); judge
   stochastic and deterministic returns separately (v31 / v43 gains were evaluation noise only); expected size
   at most ~1k, so it needs the replicate count of section (c).
4. **Rollout length as a schedule (direction 7, with 4a).** Zero lines: fork from 4M with 16 x 2048 or 4096 at
   fixed passes, and from 1M with 16 x 256. Untested in this form; the only quantity that moves the held-out
   metric in the archive. Confounds: KL per sample changes with rollout length at fixed KL per iteration, so
   pair each arm with a KL-matched twin (bound scaled by the length ratio); 64k-row arms are slow early, so the
   fork must start late; v25 says no asymptote change, so the claim to test is the 4-8M slope only.
5. **Critic budget on the reference recipe: `--aux-horizon 1 / 4`, `--reward-loss-coef 0`, critic passes 8 / 16,
   `--gve-lambda 0.95` (directions 8a and 10a).** Zero lines, all valid on resume, and each arm is cheaper than
   the control (H1 is ~0.4x the update). Confounds: the existing H4 / H8 / reward 2 x 2 is non-monotone with a
   6k range across single trajectories, so nothing here is readable without replicates; a fork changes the aux
   horizon under an encoder trained with H8, so add a from-scratch flip census for any arm that looks neutral.

Not in the top 5 only because they are not cheap:
- **Vine labels (direction 2)** have the strongest measured prior of anything on the list (within-state
  correlation 0.87 against 0.15, at 1M) and are compute-neutral, but need ~250 lines, a shared module and a
  parity test. It is also the only proposal that tests the upper bound "does a critic with good within-state
  labels produce a better actor at all".
- **Per-state dQ/da winsorisation (direction 6)** is cheap and untried but the offline audit predicts a small
  effect (cosine 0.96-0.98 with and without the tail); run it only as a rider on another experiment.
- **Hybrid gradient (direction 3)** and **replay window (direction 9)** are medium ports with confounded priors.

---

## (c) Protocol robust to the flip lottery and the single-seed noise floor

### How forking works in v60 [CODE]

Saving (`save_checkpoint`, `v60:707-717`; triggered by `--checkpoint-steps` at `v60:901-903`, at the end of the
first iteration whose `global_step` reaches the threshold): `actor_state` (network, `log_dual_temperature`,
`log_kl_multiplier`, the IDS temperature buffer), `critic`, `old_actor`, `previous_actor`, both AdamW states
(actor optimiser with its two parameter groups), `obs_means` / `obs_variances` / `obs_counts` (per env),
`global_step`, `iteration`, `temperature`, `target_entropy`, `args`.

Not saved: MuJoCo states and episode counters of the 16 envs, the host action sampler's generator, the CUDA
default generator (minibatch rows, policy noise in the losses), the diagnostic generator, the rollout buffers,
the target critic (it equals the critic at the end of an iteration).

Resuming (`v60:688-705`, `:726-746`): the nets, optimisers and statistics are loaded (statistics in place,
because the native normaliser holds pointers); `target_critic` is set to the critic; then the run proceeds as a
fresh process: `random`, NumPy and torch are seeded from `--seed`, `compute_phase_offsets(16, 1000, seed)`
and `run_phase_warmup` reset the envs and run 1,000 unrecorded stochastic steps per env **with the loaded
actor**, updating the observation statistics; `global_step = 16,000 + checkpoint step`. Consequences:
- The first recorded step of a fork from `checkpoint_1015424.pt` is at 1,031,424, and `--iterations 426` ends at
  8,011,008 (what v62 used [RAW hparams]). From 4,013,696 use `--iterations 243` (also ends 8,011,008).
- Without `--iterations`, `num_iterations = (total_timesteps - 16,000) // 16,384 = 487` *further* iterations
  (`v60:364`): `--iterations` must always be given on a fork.
- A fork is **not** a continuation of the base: the envs and every random stream restart. FAMILY.md's "resume
  is transparent" is true only in the sense that writing checkpoints does not perturb the base run
  (base = v55 dual to the digit [TSV]).
- `--fork-ent-target-mult` / `--fork-iterations` apply another entropy target for the first N fork iterations
  and then return to `--ent-target-mult`; `fork_<N>.pt` and `fork_<num_iterations>.pt` are written for
  `scripts/fork_eval.py`, which scores them deterministically and stochastically on common episode seeds.
- Constant learning rate, no schedule state: nothing else depends on the absolute iteration.

Upright checkpoints on disk now [RAW]: `checkpoint_1015424.pt` (iteration 61, T 0.003, KL multiplier 0.23),
`checkpoint_4013696.pt` (iteration 244, T 0.0225, multiplier 0.80), `checkpoint_7913088.pt` (iteration 482,
T 0.0357, multiplier 0.64), all strict FP32, lambda 0.8, H8, KL 0.03, from path B (11069 at 8M). Policy-only
snapshots (actor net, critic, statistics; no optimisers, not loadable by `--resume`) also exist for path A
(`reppo_diagnose_v29_h8_reward_16M`: 8.0 / 12.0 / 16.0M, the one v40's `--init-snapshot` used) and for v23
(4 / 12 / 30M).

Are fork arms reproducible? By construction yes: every stream is reseeded from `--seed` and nothing reads the
wall clock except the run name. From-scratch bit-determinism is established (v7 cu128 = r1 = r2; v55 dual =
v60 base = landscape ref). **No two identical fork submissions exist on disk, so fork determinism is inferred,
not verified**; the first job of any campaign should be a duplicate control fork, run at the concurrency the
campaign will use.

Existing fork precedents: v62 control vs v60 base -181 / +48 / -322 / -744 at 2 / 3 / 4 / 8M; six forks from the 1M
checkpoint (v62 x 3, v63 x 3), none flipped [TSV].

### Recommended protocol

Rules that bind: HalfCheetah-v4, 8M, seed 1, every job through `mlq`, `autocull` running, no smoke tests.

0. **Replicates without changing the seed.** "Always use seed=1" and "at least 3 RNG-perturbed trajectories"
   (01) conflict unless the perturbation is not the seed. Add one flag to the new version, `--rng-stream k`
   (default 0), that changes only the host action sampler (`np.random.default_rng([seed, k])`, `v60:650`). Init,
   env resets, phase offsets and the checkpoint stay those of seed 1. The record shows this is sufficient: v61's
   control differs from v60 base only in RNG consumption and landed on the other basin. If the rule is read as
   forbidding this, the alternative is `--seed k` on forks only, where the learner state is the checkpoint's
   and the seed changes just the restart streams; that needs the user's agreement.
1. **Fork noise floor, once (4 jobs).** Control forks from `checkpoint_1015424.pt`, streams 0-3, to 8M, plus
   one duplicate of stream 0. Yields the fork sd at 2 / 4 / 8M (today known only as n = 3 at 0.11k / 0.25k /
   0.42k) and verifies determinism.
2. **Screening by fork, n = 4 streams per arm, same streams in every arm.** Treat arms as unpaired: sharing a
   stream shares only the warmup and the first rollout, after which trajectories decorrelate. With sd 0.42k at
   8M the two-sigma difference of two 4-run means is 2 x 0.42 x sqrt(2 / 4) = 0.6k (0.16-0.3k at 2M, ~0.35k at
   4M). An effect below that is not a result.
   - Early-phase ideas (directions 1, 4a, 2, 3): fork at 1.015M.
   - Late-phase ideas (5a, 7, 8a, 10a): fork at 4.014M with `--iterations 243`.
   - Every arm set includes a same-code control (`flag off`), which must match the v60 control fork
     bit-for-bit; that is the implementation check.
3. **Flip census for anything that could change the basin (directions 1, 2, 3, 4a, 8a).** From scratch, 8
   streams per arm, submitted as 8M runs and cancelled at 2M when flat at 1.9-2.6k (the repo rule allows
   stopping clear underperformers at 1-2M). Report flips / 8 and the upright-conditional curve separately. Power:
   at p = 0.45, at most 1 flip in 8 has probability 0.06, so "0 or 1 of 8" is evidence of a lower flip rate and
   "3 to 5 of 8" is indistinguishable from the reference. Confirm posture with `scripts/reppo_posture.py` on a
   1.5M snapshot instead of inferring it from the return level.
4. **Readouts.** Matched-step windows that do not straddle the end of the run (end at 8.01M or read @7.9M);
   last-20; deterministic eval; `probe/slope_beta`, `policy/kl_violation`, `policy/std`, `q_mse`; and for
   critic-side arms the oracle (`scripts/reppo_critic_oracle.py`) on a checkpoint, at H 10-25 rather than 400
   (03: reliability 0.9 instead of 0.54 at a fraction of the cost). Do not use `advantage_corr_heldout` as a
   quality measure (01 section 3).
5. **Queueing.** `mlq submit ... --max-parallel-runs 3` for REPPO: three concurrent runs each make 10-12k SPS
   and the box saturates near 36k aggregate SPS on the GPU [DOC 06], so more parallelism buys nothing. A 4-arm
   x 4-stream fork screen from 1M is 16 x 7M steps, about 50-55 minutes of box time; the census is 8 x 2M per
   arm. Keep `scripts/autocull.py --only <prefix> --enforce health --yes --watch 600` running: fork runs start
   past `--min-steps` (3M for the 4M forks), so `--enforce all` against a from-scratch reference can cull them
   on the first evaluation; use `health` or pass the fork control as `--ref` [CODE: gates at autocull l.574-587;
   not exercised].
6. **Promotion.** An arm is promoted to a from-scratch 8M comparison only if it clears the fork MDE at the
   checkpoint it targets or lowers the flip count; the final claim is the mean over streams of both arms with
   flips reported separately, never a single seed-1 run against path A or B.

### What this protocol cannot do

- It conditions on path B. An idea whose value is in the first 1M (before any checkpoint) is visible only in
  the census.
- Forks from 1M inherit a critic and an encoder trained by the control algorithm; a neutral fork does not rule
  out a from-scratch effect (v42's capacity test has exactly this weakness).
- n = 4 resolves ~0.6k at 8M. Most single-run differences in the record are smaller than that.

---

## Not checked

- I did not open `cleanrl/td7/` trainers beyond its 22-line FAMILY.md, nor `cleanrl/awr/` trainers; the claim
  "no EMA critic for the actor" outside `cleanrl/reppo` rests on grep, not on reading each file.
- The latent_qprop / ipg / latent_critic files were read at header level only; their scores are from
  `all_hc_runs.tsv`.
- Update-time estimates marked [INFER] were not profiled.
- Whether `torch.autograd.grad` inside `graph_compile` traces (direction 6) was not tested; no code was run on
  the GPU.
- Posture was not measured for any run in this report; "flip" follows 01's return-level convention except where
  [DOC] cites a posture measurement.
