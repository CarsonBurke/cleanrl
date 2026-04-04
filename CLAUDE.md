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
- The best solution may combine ideas from control theory, signal processing, information geometry, or dynamical systems in ways no one has tried.

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

### Benchmarking and iterating

- Always run experiments as background tasks so they appear in your harness UI.
- Generally do not run more than 3 experiments at once, which already saturate compute.
- If a run is clearly underperforming after 1-2M steps you may wawnt to stop it
- After a benchmark completes (or enough data to judge): re-evaluate your hypothesis, determine if it should be iterate on futher, and parse what worked and what didn't.
- Use `cleanrl/scripts/score_runs.py` to get a clear picture of results. Run with `uv run python cleanrl/scripts/score_runs.py <pattern> [--env <env>] [--last N] [--runs-dir <from root dir>]`.

First use or activate venv at `.venv/bin/python`

Run with **16 parallel environments** and **versioned experiment names**:

```bash
# Use Bash tool with run_in_background: true
.venv/bin/python -u cleanrl/ppo_continuous_action_<method>.py \
    --env-id HalfCheetah-v4 \
    --num-envs 16 \
    --exp-name <method>_v<N> \
    --total-timesteps 1000000 \
    --seed 1
```

## Independence

When in auto research mode, operate **entirely independently**. Do not ask the user for permission or direction: make decisions, run experiments, analyze results, iterate. The user will check in on your progress; have clear results and reasoning ready.

It is necessary that you be mindful of your limited context window when doing tasks. Delegate tasks to subagents and be frugal so you can work for longer periods of time. Use your judgement, don't read entire outputs (tail, sample, etc.) grep things, run tasks such that they don't have output or are tailed, etc.

Your workflow loop:

1. **Hypothesize**: form a clear, specific hypothesis about what will improve performance
2. **Implement**: write clean, well-documented code in a new or modified file
3. **Test**: run all three benchmarks as background tasks with versioned names and check in on them at set intervals
4. **Monitor**: Run as a background task, and if relevant, periodically check progress and stop underperforming runs early.
5. **Analyze**: compare against baselines, understand what worked and why
6. **Iterate**: keep improvements, rethink or roll back failures with documented reasoning, form new hypotheses

## Technical Notes

- **Device**: CUDA is available and enabled by default
- **Logging**: TensorBoard logs go to `runs/{env_id}__{exp_name}__{seed}__{timestamp}/`
- **No W&B**: don't use `--track` flag unless explicitly asked — local TensorBoard only
- **Gradient clipping**: all variants use max_grad_norm=0.5 — be careful changing this or other similar hyperparameters
- **Observation/reward normalization**: handled by env wrappers, not the agent
- **Action space**: all three envs use continuous actions, clipped to [-1, 1] by wrapper
