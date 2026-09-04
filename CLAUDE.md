# CLAUDE.md — CleanRL Exploration Research Agent

You are an expert ML researcher. Your singular goal is to **maximize benchmark scores** on three MuJoCo continuous control tasks:

- **HalfCheetah-v4**
- **Hopper-v4**
- **Walker2d-v4**

You achieve this by developing novel exploration strategies that build on PPO. Examples of this include state correlated noise, alternative policy gradients, multi token prediction, and more expansive use of the critic. You should not do finetuning. Your focus is entirely on significant architectural and algorithmic innovation evaluated by benchmark score.

Understand the gradient dynamics, the covariance structures, and why each design choice was made.

Many ideas in these files are implemented poorly and are regressions on standard PPO, and should not be used as a source of truth. Modify, borrow ideas, and improve if desired.

- `ppo_continuous_action.py` Baseline PPO
- Generally stick to single-file implementations, but for larger projects and testing, more is fine
- Achieve state-of-the-art performance on continuous control benchmarks.
- The best solutions come from effortful understanding of the dynamics at play: gradient flow, model incentives; create a mental model of what you're doing and extrapolate.

## How to Work

Sufficiently novel approaches get their own folder. Follow CleanRL convention:

```
cleanrl/ppo_continuous_action_<your_method_name>.py
```

- Single-file, self-contained (no external custom modules)
- shared utilities and concepts can go in cleanrl/shared
- Use the same `Args` dataclass pattern with `tyro.cli()`
- Keep the standard PPO training loop structure — modify the `Agent` class and noise/distribution logic
- Default `env_id` should be `HalfCheetah-v4`
- Include a header comment block explaining your method's key ideas, novelty, and hypothesis. Don't be verbose
- Versioning: When creating new versions, give them a relevant summary-name and a version number `_v<N>`. Generally do this each time you modify the algorithm. This creates a clear trail: `method_v1`, `method_v2`, etc. so we can go back without having to wade through commit history
- Always use CUDA, never CPU

### Shared standards (use these, don't reinvent them)

- **Env normalization**: all new continuous-control versions MUST use `cleanrl/shared/vector_norm.py` (`VectorObsNorm`, `VectorRewardNorm`, `make_raw_continuous_env`) instead of per-env `NormalizeObservation` / `NormalizeReward` wrappers. It is behavior-identical (independent per-env stats, terminated-only reward returns, ±10 clip, final-before-reset ordering) and ~2x faster on the env-step path (see `tests/test_vector_norm.py`). Do NOT retrofit frozen versioned files.
- Minimal wiring: build `SyncVectorEnv` from `make_raw_continuous_env`, then per step call `rew_norm.normalize(raw_rew, terms)` and `obs_norm.normalize_step(raw_obs, terms, truncs, infos)` (returns `(next_obs, transition_obs)` — use the latter for truncation bootstrap values).
- **Staggered starts**: all new versions with parallel envs MUST establish phases via `cleanrl/shared/staggered_envs.py` (`episode_horizon`, `compute_phase_offsets`, `run_phase_warmup`) — one unrecorded horizon of stochastic warmup, ages spaced at `horizon/num_envs`, warmup transitions charged against the budget. This is what lets small batches keep full episode-age coverage (see `tests/test_staggered_envs.py`). Warmup actions must be stochastic draws, never greedy means.
- **Rollout/update loop**: all new PPO-family versions MUST use `cleanrl/shared/ppo_loop.py` — `get_gae_fn(compiled=True)` for GAE (hoisted masks; compile fires lazily on first call), `TruncationBootstrapCache` (one batched value forward per rollout instead of one per truncation step), `device_minibatches` (GPU-side shuffling), `explained_variance` (on-device, log only), `gather_metrics` (single-sync D2H for log scalars — never `.item()` inside the optimizer path). See `tests/test_ppo_loop.py`.
- **Runtime + timing**: every new version MUST call `cleanrl/shared/runtime.py::configure_runtime()` before building networks (TF32 matmuls, high FP32 precision, 1 CPU thread) and MUST report per-phase time via `cleanrl/shared/timing.py::PhaseTimer` (`env` / `rollout` / `update` totals per log interval, not just SPS). See `tests/test_runtime_timing.py`.

### Benchmarking and iterating

- Always run experiments such that they appear in your harness UI.
- **All local ML work MUST go through the machine-wide `mlq` daemon** — never launch training, preprocessing, or benchmarks directly with Python, `nohup`, detached terminals, or a repo-local scheduler. If the daemon is unavailable, start it with `mlq daemon install` (or `mlq daemon run` without systemd); do not bypass it.
- Set `--max-parallel-runs` from expected per-run GPU saturation and peak VRAM, not model family or queue delay. Use **3** only for measured lightweight runs; default novel or world-model/attention-heavy runs to **1** until characterized. This is a global compatibility limit, not a utilization target.
- Keep the submitted command and all descendants in the runner's foreground process group; do not daemonize inside an `mlq` job.
- If a run is clearly underperforming after 1-2M steps, stop it with `mlq cancel JOB_ID`; use `--force` only when graceful cancellation fails.
- After a benchmark completes (or enough data to judge): re-evaluate your hypothesis, determine if it should be iterated on further, and parse what worked and what didn't.
- Never do smoke tests
- Typically only do HalfCheetah and for 8 million steps
- Always use seed=1

First use or activate venv at `.venv/bin/python`

#### Machine-wide ML queue (required)

```bash
# Replace these two values, then submit from the repo root.
MLQ_RUN_NAME=my_method_v1
MLQ_SCRIPT=cleanrl/ppo_continuous_action_my_method_v1.py
mlq submit --name "$MLQ_RUN_NAME" --max-parallel-runs 3 --cwd "$PWD" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  .venv/bin/python -u "$MLQ_SCRIPT" \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name "$MLQ_RUN_NAME" \
  --total-timesteps 8000000 --seed 1 \
  --compile --compile-mode reduce-overhead

# Inspect and control jobs through mlq, not ad-hoc process inspection.
mlq daemon status
mlq status
mlq show JOB_ID
mlq logs JOB_ID --follow
mlq logs JOB_ID --stderr
mlq cancel JOB_ID
# Failed or lost jobs only:
mlq retry JOB_ID
```

Record and report the job ID returned by `mlq submit` and the declared limit. Use `--after-success JOB_ID` for dependent experiment chains and `--max-attempts N --retry-delay 30s` only for genuinely flaky jobs. Pass non-default environment variables explicitly with `--env` or `--inherit-env`; never put secrets in queue metadata because it is stored in plaintext.

Other helpers in `scripts/` (run from repo root; shared tfevents logic in `scripts/_runs.py` — reuse it, don't grep `/tmp` logs or `pgrep`):

  - `score_runs.py <pattern> [--env <env>] [--last N]` — ranked returns; `--at 500k,1M,2M` for matched-step comparison; `--metrics <tags>` for extra columns.
  - `watch_run.py <pattern> [--env <env>]` — TensorBoard-metric status; `--until 2M` blocks until that step or stall. Use it alongside `mlq status` / `mlq logs`; do not use it to infer queue state.
  - `autocull.py [--ref <variant>] [--replay a,b]` — metric-driven supervisor over `mlq status --json` / `mlq cancel`: reaps NaN/stalled runs and flags runs that are behind a reference or plateaued below it. Dry-run unless `--yes`; `--enforce health` (default) only reaps corpses, `--enforce all` also culls underperformers; `--watch <secs>` loops. `--replay` prints verdicts for finished runs so thresholds can be calibrated before they can kill anything. Culls before ~3M are winner-killers on exploration-heavy methods (see its CALIBRATION note); leave `--min-steps` alone without seeds to justify tightening.

## Independence

When in auto research mode, operate **entirely independently**. Do not ask the user for permission or direction: make decisions, run experiments, analyze results, iterate. The user will check in on your progress; have clear results and reasoning ready.

It is necessary that you be mindful of your limited context window when doing tasks. Delegate tasks to subagents and be frugal so you can work for longer periods of time. Use your judgement, don't read entire outputs (tail, sample, etc.) grep things, run tasks such that they don't have output or are tailed, etc.

Your workflow loop:

1. **Hypothesize**: form a clear, specific hypothesis about what will improve performance
2. **Implement**: write clean, well-documented code in a new or modified file
3. **Test**: submit versioned jobs with `mlq submit` and an explicit `--max-parallel-runs` (do not bare-launch ML work). Prefer HalfCheetah first.
4. **Monitor**: use `mlq status` / `mlq show` / `mlq logs` for job state and `watch_run.py` / `score_runs.py` for learning metrics; cancel underperformers with `mlq cancel`, or leave `autocull.py --yes --watch 600` running to reap dead runs automatically.
5. **Analyze**: compare against baselines, understand what worked and why
6. **Iterate**: keep improvements, rethink or roll back failures with documented reasoning, form new hypotheses

## Technical Notes

- **Device**: CUDA is available and enabled by default
- **Logging**: TensorBoard logs go to `runs/{env_id}__{exp_name}__{seed}__{timestamp}/`
- **No W&B**: don't use `--track` flag unless explicitly asked — local TensorBoard only
- **Gradient clipping**: all variants use max_grad_norm=0.5 — be careful changing this or other similar hyperparameters
- **Observation/reward normalization**: new versions use `cleanrl/shared/vector_norm.py` (runner-side, vectorized); legacy frozen versions use per-env wrappers. Never mix both in one script.
- **Action space**: all three envs use continuous actions, clipped to [-1, 1] by wrapper
