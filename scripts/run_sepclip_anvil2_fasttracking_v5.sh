#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit for autocull queue access}"
.venv/bin/python -u scripts/autocull.py \
  --ref sepclip_anvil2_v4_anvil_half --only sepclip_anvil2_fasttracking_v5_ \
  --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u cleanrl/ppo_continuous_action_sepclip_anvil2_fasttracking_v5.py \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name sepclip_anvil2_fasttracking_v5_half --total-timesteps 8000000 --seed 1 \
  --learning-rate 0.0024 --normuon-network actor --matrix-method anvil2 \
  --normuon-lr 0.0025 --matrix-precision bf16 \
  --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead
