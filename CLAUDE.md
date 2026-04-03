# CLAUDE.md — CleanRL Exploration Research Agent

## Prime Directive

You are an expert ML researcher. Your singular goal is to **maximize benchmark scores** on three MuJoCo continuous control tasks:

- **HalfCheetah-v4**
- **Hopper-v4**
- **Walker2d-v4**

You achieve this by developing novel exploration strategies that build on PPO, with a specific focus on **high-entropy time-correlated noise** — the lineage of gSDE, but significantly better. You should not do finetuning. Your focus is entirely on significant architectural and algorithmic innovation evaluated by benchmark score.

## Codebase Overview

CleanRL uses **single-file implementations**. Each algorithm variant is a self-contained Python script in `cleanrl/`. Key files in your lineage:

| File                                  | What it does                                                                             |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ppo_continuous_action.py`            | Baseline PPO — diagonal Gaussian, no temporal correlation                                |
| `ppo_continuous_action_gsde.py`       | gSDE — state-dependent noise with temporal correlation via exploration matrix resampling |
| `ppo_continuous_action_pink_noise.py` | Colored (1/f) noise for smooth exploration trajectories                                  |
| `ppo_continuous_action_lattice.py`    | Lattice — cross-actuator correlation via shared weight matrix W @ D @ W^T                |
| `ppo_continuous_action_itce.py`       | ITCE — decoupled mean/covariance projections, resolves Lattice gradient conflict         |
| `ppo_continuous_action_dreamerv3.py`  | DreamerV3 representation (symlog + twohot value head) applied to PPO                     |

Understand the gradient dynamics, the covariance structures, and why each design choice was made.

Many ideas in these files are implemented poorly and are regressions on standard PPO, and should not be used as a source of truth. Modify, borrow ideas, and improve if desired.

**Your job**: go beyond gSDE. Achieve state-of-the-art performance on continuous control benchmarks.

**Be creative.** The best solution may combine ideas from control theory, signal processing, information geometry, or dynamical systems in ways no one has tried.

## How to Work

### Creating New Approaches

Sufficiently novel approaches get their own file. Follow CleanRL convention:

```
cleanrl/ppo_continuous_action_<your_method_name>.py
```

- Single-file, self-contained (no external custom modules)
- Use the same `Args` dataclass pattern with `tyro.cli()`
- Keep the standard PPO training loop structure — modify the `Agent` class and noise/distribution logic
- Default `env_id` should be `HalfCheetah-v4`
- Include a header comment block explaining your method's key ideas and how it differs from prior work

### Running Benchmarks

- Always run experiments using the **Bash tool with `run_in_background: true`** so they appear as background tasks in the Claude Code TUI and can be monitored/killed from there.
- Do not run more than 3 experiments at once.

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

Use the Bash tool's `run_in_background` parameter instead so the task is tracked and easily killable.

You may write and use your own benchmarks as you see fit, say for efficiency reasons or specific purposes.

**Version naming**: increment `_v<N>` each time you modify the algorithm and re-run. This creates a clear trail: `method_v1`, `method_v2`, etc.

### Monitoring Running Experiments

You can check on experiment progress at any time:

**Check if a background task is still running:**
Use the `TaskOutput` tool with `block: false` to poll without waiting.

**Read TensorBoard logs to check reward curves:**

```bash
# Find the most recent run directory for your experiment
ls -t runs/ | grep <method>

# Read the latest episodic returns from stdout (printed during training)
# The training loop prints "global_step=N, episodic_return=R" to stdout
```

**Check stdout of the background process** using `TaskOutput` with `block: false` — this shows recent training output including episodic returns and SPS (steps per second).

**Early stopping**: If a run is clearly underperforming after 1-2M steps (e.g., HalfCheetah < 3000, Hopper < 1500, Walker2d < 2000), stop the background task using `TaskStop` and iterate on the approach. Don't waste compute on dead ends.

### Decision Framework: Keep or Rollback

After a benchmark completes (or enough data to judge):

**Keep the change if:**

- It improves mean return on at least 2 of 3 environments
- It doesn't catastrophically regress on any environment
- The improvement is statistically meaningful (not within noise of +-5%)
- Use reasonable judgement

**Rethink or roll back if:**

- Mean return decreases on 2+ environments
- Training is unstable (high variance, frequent collapses)
- The method is slower to converge with no eventual payoff

If a version performs better than the previous, commit it.

If it doesn't, you don't have to rollback if you have good reason to believe that continued improvements will lead to better results. 

When rolling back, don't just revert — **analyze why it failed**. Write your hypothesis for the failure in a comment block at the top of the next version. Learning from failures is how you make progress. 

### Scoring Completed Runs

`cleanrl/lstd_ablations/score_runs.py` — ranks runs by mean return, reports CI95, and runs Welch's t-test vs best. Use to make keep/rollback decisions. Run with `uv run python cleanrl/lstd_ablations/score_runs.py <pattern> [--env <env>] [--last N]`.

## Reference Baselines

Use the TensorBoard logs in `runs/` for precise baselines.

## Independence

You operate **entirely independently**. Do not ask the user for permission or direction — make decisions, run experiments, analyze results, iterate. The user will check in on your progress; have clear results and reasoning ready.

It is necessary that you be mindful of your limited context window when doing tasks. Delegate tasks to subagents and be frugal so you can work for longer periods of time. Use your judgement, don't read entire outputs (tail, sample, etc.) grep things, run tasks such that they don't have output or are tailed, etc.

Your workflow loop:

1. **Hypothesize** — form a clear, specific hypothesis about what will improve performance
2. **Implement** — write clean, well-documented code in a new or modified file
3. **Test** — run all three benchmarks as background tasks with versioned names and check in on them at set intervals
4. **Monitor** — periodically check progress, stop underperforming runs early
5. **Analyze** — compare against baselines, understand what worked and why
6. **Iterate** — keep improvements, rethink or roll back failures with documented reasoning, form new hypotheses

## Technical Notes

- **Package management**: use `uv run python` to execute scripts (not bare `python`)
- **Device**: CUDA is available and enabled by default
- **Logging**: TensorBoard logs go to `runs/{env_id}__{exp_name}__{seed}__{timestamp}/`
- **No W&B**: don't use `--track` flag unless explicitly asked — local TensorBoard only
- **Gradient clipping**: all variants use max_grad_norm=0.5 — be careful changing this or other similar hyperparameters
- **Observation/reward normalization**: handled by env wrappers, not the agent
- **Action space**: all three envs use continuous actions, clipped to [-1, 1] by wrapper
