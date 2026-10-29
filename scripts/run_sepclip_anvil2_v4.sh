#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit for autocull queue access}"
kind="$1"
case "$kind" in
  normuon_half) extra=(--matrix-method normuon --normuon-beta2 0.95 --normuon-lr 0.0025) ;;
  normuon_double) extra=(--matrix-method normuon --normuon-beta2 0.95 --normuon-lr 0.01) ;;
  normuon_fast) extra=(--matrix-method normuon --normuon-beta2 0.95 --normuon-momentum 0.85 --normuon-lr 0.005) ;;
  anvil_half) extra=(--matrix-method anvil2 --normuon-lr 0.0025) ;;
  anvil_full) extra=(--matrix-method anvil2 --normuon-lr 0.005) ;;
  anvil_double) extra=(--matrix-method anvil2 --normuon-lr 0.01) ;;
  anvil_quad) extra=(--matrix-method anvil2 --normuon-lr 0.02) ;;
  anvil_fp32) extra=(--matrix-method anvil2 --matrix-precision fp32 --normuon-lr 0.005) ;;
  *) exit 2 ;;
esac
run_name="sepclip_anvil2_v4_${kind}"
.venv/bin/python -u scripts/autocull.py \
  --ref normuon_audit_8m_adam --only sepclip_anvil2_v4_ \
  --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u cleanrl/ppo_continuous_action_sepclip_anvil2_v4.py \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" --total-timesteps 8000000 --seed 1 \
  --learning-rate 0.0024 --normuon-network actor \
  --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead "${extra[@]}"
