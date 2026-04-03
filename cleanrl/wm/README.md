# WM Benchmarks

Use the local venv directly for these runs:

```bash
.venv/bin/python -u cleanrl/wm/<script>.py ...
```

The `v6.x` comparisons in this folder should use the 4-env regime, not 16 envs. These scripts were tuned around `num_envs=4`, `num_steps=512`, replay-seeded imagination, and a heavier model-update budget per real rollout.

Recommended HalfCheetah commands:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_2.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_2_v2 \
  --total-timesteps 8000000 \
  --seed 1
```

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_3.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_3_v2 \
  --total-timesteps 8000000 \
  --seed 1
```

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python -u cleanrl/wm/wm_sde_stateent_base_v6_4.py \
  --env-id HalfCheetah-v4 \
  --num-envs 4 \
  --exp-name wm_sde_stateent_base_v6_4_v1 \
  --total-timesteps 8000000 \
  --seed 1
```

Monitoring:

```bash
ls -td runs/HalfCheetah-v4__wm_sde_stateent_base_v6_* | head
```

```bash
pgrep -af 'cleanrl/wm/wm_sde_stateent_base_v6_(2|3|4)\\.py'
```
