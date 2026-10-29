#!/usr/bin/env bash
set -Eeuo pipefail
run_name=${1:?run name required}
objective=${2:?actor objective required}
baseline=${3:?actor baseline required}
target_std=${4:?target std mode required}
case "$run_name" in
    td7_latentq_value_baseline_v4*) ;;
    *) printf 'Unexpected experiment prefix\n' >&2; exit 2 ;;
esac
case "$objective:$baseline:$target_std" in
    delight:latent:policy|delight:cubature:policy|pg:latent:policy|delight:cubature:fixed) ;;
    *) printf 'Unsupported experiment configuration\n' >&2; exit 2 ;;
esac

.venv/bin/python -m pytest -q \
    tests/test_td7_latentq_delight_cubature_v1.py \
    tests/test_td7_latentq_delight_fast_preaction_v3.py \
    tests/test_td7_latentq_value_baseline_v4.py tests/test_autocull.py tests/test_autocull_mlq.py
mkdir -p benchmarks/td7_delight_cubature_v1/logs
cull_args=(--ref HalfCheetah-v4__td7_latentq_delight_cubature_v1__1__1791688686 --only "$run_name"
    --enforce all --yes --collapse-ratio 0.5 --collapse-drop 2000
    --collapse-ema-halflife 50k --collapse-min-steps 200k)
# Fail before allocating GPU resources if the queue client cannot reach mlqd.
.venv/bin/python -u scripts/autocull.py "${cull_args[@]}"
.venv/bin/python -u scripts/autocull.py "${cull_args[@]}" --watch 60 \
    > "benchmarks/td7_delight_cubature_v1/logs/${run_name}_autocull.log" 2>&1 &
cull_pid=$!
cleanup() {
    kill "$cull_pid" 2>/dev/null || true
    wait "$cull_pid" 2>/dev/null || true
}
trap cleanup EXIT
.venv/bin/python -u cleanrl/td7/delight/td7_latentq_value_baseline_v4.py \
    --env-id HalfCheetah-v4 --num-envs 16 --num-steps 1024 \
    --total-timesteps 8000000 --seed 1 --env-threads 2 --compile \
    --exp-name "$run_name" --actor-objective "$objective" --actor-sale \
    --actor-baseline "$baseline" --target-std "$target_std" --target-smoothing-sigma 0
