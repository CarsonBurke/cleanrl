#!/usr/bin/env bash
set -Eeuo pipefail
run_name=${1:?run name required}
reference=${2:?reference required}
scope=${3:-all}
case "$run_name" in
    td7_latentq_delight_reference_v10*) ;;
    *) printf 'Unexpected experiment prefix\n' >&2; exit 2 ;;
esac
case "$reference" in
    modal|latent) ;;
    *) printf 'Unknown reference\n' >&2; exit 2 ;;
esac
case "$scope" in
    all|mean) ;;
    *) printf 'Unknown gate scope\n' >&2; exit 2 ;;
esac
.venv/bin/python -m pytest -q \
    tests/test_td7_latentq_delight_cubature_v1.py \
    tests/test_td7_latentq_delight_reference_v10.py \
    tests/test_td7_value_floor_checkpoint.py tests/test_checkpointing.py \
    tests/test_autocull.py tests/test_autocull_mlq.py
mkdir -p benchmarks/td7_delight_cubature_v1/logs
cull_args=(--ref HalfCheetah-v4__td7_latentq_delight_cubature_v1__1__1791688686 --only "$run_name"
    --enforce all --yes --collapse-ratio 0.5 --collapse-drop 2000
    --collapse-ema-halflife 50k --collapse-min-steps 200k)
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/supervise_eval.py "${cull_args[@]}"
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/supervise_eval.py "${cull_args[@]}" --watch 60 \
    > "benchmarks/td7_delight_cubature_v1/logs/${run_name}_autocull.log" 2>&1 &
cull_pid=$!
cleanup() {
    kill "$cull_pid" 2>/dev/null || true
    wait "$cull_pid" 2>/dev/null || true
}
trap cleanup EXIT
.venv/bin/python -u cleanrl/td7/delight/td7_latentq_delight_reference_v10.py \
    --env-id HalfCheetah-v4 --num-envs 16 --num-steps 1024 \
    --total-timesteps 8000000 --seed 1 --env-threads 2 --compile \
    --exp-name "$run_name" --actor-objective delight --actor-sale \
    --actor-baseline "$reference" --gate-scope "$scope" --gate-scale raw \
    --target-sigma 0.1 --entropy-floor --target-smoothing-sigma 0.2
