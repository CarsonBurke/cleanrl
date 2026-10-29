#!/usr/bin/env bash
set -Eeuo pipefail
run_name=${1:?run name required}
objective=${2:?actor objective required}
baseline=${3:?actor baseline required}
case "$run_name" in
    td7_latentq_value_floor_v5*) ;;
    *) printf 'Unexpected experiment prefix\n' >&2; exit 2 ;;
esac
case "$objective:$baseline" in
    delight:latent|delight:cubature|pg:latent|pg:cubature) ;;
    *) printf 'Unsupported experiment configuration\n' >&2; exit 2 ;;
esac
.venv/bin/python -m pytest -q \
    tests/test_td7_latentq_delight_cubature_v1.py \
    tests/test_td7_latentq_value_baseline_v4.py \
    tests/test_autocull.py tests/test_autocull_mlq.py \
    tests/test_td7_value_floor_checkpoint.py tests/test_checkpointing.py
mkdir -p benchmarks/td7_delight_cubature_v1/logs
cull_args=(--ref HalfCheetah-v4__td7_latentq_delight_cubature_v1__1__1791688686 --only "$run_name"
    --enforce all --yes --collapse-ratio 0.5 --collapse-drop 2000
    --collapse-ema-halflife 50k --collapse-min-steps 200k)
.venv/bin/python -u scripts/autocull.py "${cull_args[@]}"
.venv/bin/python -u scripts/autocull.py "${cull_args[@]}" --watch 60 \
    > "benchmarks/td7_delight_cubature_v1/logs/${run_name}_autocull.log" 2>&1 &
cull_pid=$!
cleanup() {
    kill "$cull_pid" 2>/dev/null || true
    wait "$cull_pid" 2>/dev/null || true
}
trap cleanup EXIT
.venv/bin/python -u cleanrl/td7/delight/td7_latentq_value_floor_v5.py \
    --env-id HalfCheetah-v4 --num-envs 16 --num-steps 1024 \
    --total-timesteps 8000000 --seed 1 --env-threads 2 --compile \
    --exp-name "$run_name" --actor-objective "$objective" --actor-sale \
    --actor-baseline "$baseline" --target-sigma 0.1 --target-smoothing-sigma 0.2
