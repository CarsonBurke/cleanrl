#!/usr/bin/env bash
set -euo pipefail
name=$1
script=$2
shift 2
# mlq persists a minimal environment without XDG_RUNTIME_DIR. Nested queue
# clients need the existing per-user daemon endpoint; never start another daemon.
if [[ -z ${MLQUEUE_RUNTIME_DIR:-} && -z ${XDG_RUNTIME_DIR:-} && -S /run/user/$UID/mlqueue/mlqd.sock ]]; then
  export MLQUEUE_RUNTIME_DIR="/run/user/$UID/mlqueue"
fi
# Disable only the early DEAD rule: autocull documents it as a winner-killer.
# Keep calibrated plateau/health checks, original reference, and min-steps default.
mkdir -p logs
.venv/bin/python -u scripts/autocull.py \
  --ref td7_latentq_onpolicy_v8_replay_sale --only "$name" \
  --enforce all --dead-until 0 --yes --watch 600 > "logs/${name}_autocull.log" 2>&1 &
cull_pid=$!
trap 'kill "$cull_pid" 2>/dev/null || true; wait "$cull_pid" 2>/dev/null || true' EXIT
.venv/bin/python -u "$script" --env-id HalfCheetah-v4 --num-envs 16 \
  --exp-name "$name" --total-timesteps 8000000 --seed 1 --env-threads 2 --compile "$@"
