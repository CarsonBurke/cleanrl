#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit}"
network="$1"
run_name="sepclip_normuon_split_v2_${network}_hc50m"
.venv/bin/python -u scripts/autocull.py \
  --ref awr_ce_ppo_sepclip_control_v1_50M --only "$run_name" \
  --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u cleanrl/ppo_continuous_action_sepclip_normuon_split_diagnostic_v2.py \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" --total-timesteps 50000000 --seed 1 \
  --learning-rate 0.0024 --normuon-lr 0.005 --normuon-network "$network" \
  --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead
