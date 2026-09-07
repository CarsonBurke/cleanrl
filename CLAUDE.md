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

### Shared standards

Start from the newest `ppo_continuous_action_*` version — it already wires every
item below, and this list is all you should need to write a run. Throughput
rationale and measurements are in `docs/mujoco-throughput.md`; each module's
contract is pinned by its `tests/test_<module>.py`.

Required in every new continuous-control version:

- `runtime.configure_runtime()`, before building networks.
- `mujoco_env.make_mujoco_vector_env(env_id, num_envs, num_threads=2)` for the env. `num_threads` is a per-run latency knob traded against aggregate throughput, so it is set by the concurrency guidance below, not per script.
- `vector_norm.VectorObsNorm` / `VectorRewardNorm` for normalization — never the per-env `NormalizeObservation`/`NormalizeReward` wrappers, and never both in one script.
- `staggered_envs` for episode-age phases (`episode_horizon`, `compute_phase_offsets`, `run_phase_warmup`). Warmup actions must be stochastic draws, never greedy means.
- `ppo_loop` for GAE, truncation bootstrap, minibatching and metric gathering — never `.item()` inside the optimizer path.
- `host_graph.make_host_mirror(actor, num_envs)` for the rollout actor and `sampling.make_beta_sampler(num_envs, act_dim, low, high)` for the Beta head, with `rollout_transfer.RolloutTransfer(fields=...)` for the upload. Both act on the host and return permanent buffers overwritten by the next call, so stage them in the same step; values and old log-probs come from one batched forward over the uploaded rollout.
- `timing.PhaseTimer`, reporting `env`/`rollout`/`update` totals per log interval, not just SPS.

Do NOT retrofit frozen versioned files.

### Benchmarking and iterating

- Always run experiments such that they appear in your harness UI.
- **All local ML work MUST go through the machine-wide `mlq` daemon** — never launch training, preprocessing, or benchmarks directly with Python, `nohup`, detached terminals, or a repo-local scheduler. If the daemon is unavailable, start it with `mlq daemon install` (or `mlq daemon run` without systemd); do not bypass it.
- Set `--max-parallel-runs` from measured aggregate throughput, not per-run speed. For the standard 16-env MuJoCo PPO trainers use **6** with `--env-threads 2`: measured end-to-end, that is 235-252k aggregate SPS (~3.7 min per 8M-step run) against 115k for the old 3-runs/4-threads point, i.e. 2.1x the aggregate for a ~6% slower individual run. Go to 10 only when the box is exclusively yours (317k aggregate, ~4.5 min per run, ~17 GiB VRAM). Default novel or world-model/attention-heavy runs to **1** until characterized. Size VRAM on whole-process footprint (~1.7 GiB per standard run), never on `torch.cuda.max_memory_allocated`.
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
mlq submit --name "$MLQ_RUN_NAME" --max-parallel-runs 6 --cwd "$PWD" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env CLEANRL_ENV_SPIN=5000 -- \
  .venv/bin/python -u "$MLQ_SCRIPT" \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name "$MLQ_RUN_NAME" \
  --total-timesteps 8000000 --seed 1 --env-threads 2 \
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

Performance work — never tune from a single-process wall-clock number; the box
runs several trainers at once, so throughput per core is the figure of merit.
`benchmark_rollout_scale.py [--runs N] [--threads T]` spawns N independent
rollout processes and reports aggregate SPS and SPS/core; run it with the
machine otherwise idle. It is the one that decides thread counts. For
attribution use `benchmark_rollout_chain.py` (per-link) and
`benchmark_rollout_loop.py` (assembled loop, wall *and* CPU). Component
benchmarks: `benchmark_env_cpu.py`, `benchmark_host_graph.py`,
`benchmark_vector_norm.py`, `profile_rollout_step.py`.

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
- **Action space**: all three envs use continuous actions, clipped to [-1, 1] by wrapper
