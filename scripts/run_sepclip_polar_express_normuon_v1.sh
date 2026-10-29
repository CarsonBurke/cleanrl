#!/usr/bin/env bash
set -euo pipefail

# Submit with --inherit-env XDG_RUNTIME_DIR; mlq clears unlisted variables.
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit}"
run_name="${1:-sepclip_polar_express_normuon_v1_hc50m}"
matrix_lr="${2:-0.005}"

# The supervisor stays in the mlq foreground process group with the trainer.
.venv/bin/python -u scripts/autocull.py \
  --ref awr_ce_ppo_sepclip_control_v1_50M \
  --only "$run_name" \
  --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u cleanrl/ppo_continuous_action_sepclip_polar_express_normuon_v1.py \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" \
  --total-timesteps 50000000 --seed 1 \
  --learning-rate 0.0024 --normuon-lr "$matrix_lr" \
  --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead
