#!/usr/bin/env bash
set -Eeuo pipefail
run_name=${1:?run name required}
objective=${2:?actor objective required}
query=${3:?actor query required}
smoothing=${4:?target smoothing required}
case "$run_name" in
    td7_latentq_delight_fast_preaction_v3*) ;;
    *) printf 'Unexpected experiment prefix\n' >&2; exit 2 ;;
esac
case "$objective:$query:$smoothing" in
    delight:point:0|delight:preaction:0|delight:preaction:0.2|pg:preaction:0.2) ;;
    *) printf 'Unsupported experiment configuration\n' >&2; exit 2 ;;
esac

.venv/bin/python -m pytest -q tests/test_td7_latentq_delight_cubature_v1.py tests/test_td7_latentq_delight_fast_preaction_v3.py
mkdir -p benchmarks/td7_delight_cubature_v1/logs
.venv/bin/python -u scripts/autocull.py \
    --ref td7_latentq_onpolicy_v8_replay_sale --only "$run_name" \
    --enforce all --yes --watch 600 \
    > "benchmarks/td7_delight_cubature_v1/logs/${run_name}_autocull.log" 2>&1 &
cull_pid=$!
cleanup() {
    kill "$cull_pid" 2>/dev/null || true
    wait "$cull_pid" 2>/dev/null || true
}
trap cleanup EXIT
.venv/bin/python -u cleanrl/td7/delight/td7_latentq_delight_fast_preaction_v3.py \
    --env-id HalfCheetah-v4 --num-envs 16 --num-steps 1024 \
    --total-timesteps 8000000 --seed 1 --env-threads 2 --compile \
    --exp-name "$run_name" --actor-objective "$objective" --actor-sale \
    --actor-query "$query" --target-smoothing-sigma "$smoothing"
