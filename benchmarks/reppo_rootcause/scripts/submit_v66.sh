#!/usr/bin/env bash
# usage: submit_v66.sh <arm-name> <stream> [extra trainer flags...]
# Queues one HalfCheetah 8M run of reppo_lanefree_v66 on rng stream <stream>.
set -euo pipefail
arm=$1; k=$2; shift 2
name="reppo_v66_${arm}_s${k}"
mlq submit --name "$name" --max-parallel-runs 3 --cwd "$PWD" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env CLEANRL_ENV_SPIN=5000 -- \
  .venv/bin/python -u cleanrl/reppo/ppo_continuous_action_reppo_lanefree_v66.py \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name "$name" \
  --total-timesteps 8000000 --seed 1 --env-threads 2 \
  --compile --compile-mode reduce-overhead --rng-stream "$k" "$@" | tail -1
