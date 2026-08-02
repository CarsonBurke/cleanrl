# WM Benchmarks

Submit these heavier, load-sensitive runs through the machine-wide ML queue.
Keep their global compatibility limit at 1 unless concurrent execution has
been explicitly characterized:

The `v6.x` comparisons in this folder should use the 4-env regime, not 16 envs. These scripts were tuned around `num_envs=4`, `num_steps=512`, replay-seeded imagination, and a heavier model-update budget per real rollout.

Recommended HalfCheetah commands:

```bash
mlq submit --name wm_sde_stateent_base_v6_2_v2 --max-parallel-runs 1 \
  --cwd "$PWD" --env PYTORCH_ALLOC_CONF=expandable_segments:True -- \
  .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_2.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_2_v2 \
  --total-timesteps 8000000 \
  --seed 1
```

```bash
mlq submit --name wm_sde_stateent_base_v6_3_v2 --max-parallel-runs 1 \
  --cwd "$PWD" --env PYTORCH_ALLOC_CONF=expandable_segments:True -- \
  .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_3.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_3_v2 \
  --total-timesteps 8000000 \
  --seed 1
```

```bash
mlq submit --name wm_sde_stateent_base_v6_4_v1 --max-parallel-runs 1 \
  --cwd "$PWD" --env PYTORCH_ALLOC_CONF=expandable_segments:True -- \
  .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_4.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_4_v1 \
  --total-timesteps 8000000 \
  --seed 1
```

Monitoring:

```bash
mlq status
mlq show JOB_ID
mlq logs JOB_ID --follow
mlq logs JOB_ID --stderr
.venv/bin/python scripts/score_runs.py "wm_sde_stateent_base_v6" --env HalfCheetah-v4
```
