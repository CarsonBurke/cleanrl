#!/usr/bin/env bash
set -Eeuo pipefail

run_name=${1:?run name required}
objective=${2:?actor objective required}
case "$run_name" in
    td7_latentq_delight_cubature_v1*) ;;
    *) printf 'Unexpected experiment prefix\n' >&2; exit 2 ;;
esac
case "$objective" in
    delight|pg|pathwise) ;;
    *) printf 'Unknown actor objective\n' >&2; exit 2 ;;
esac

# Analytic checks precede the full experiment; no reduced training runs.
.venv/bin/python -m pytest -q tests/test_td7_latentq_delight_cubature_v1.py

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

# Both children remain in mlq's foreground process group. The supervisor ends
# with the trainer; queue cancellation and time limits cover the complete job.
.venv/bin/python -u cleanrl/td7/delight/td7_latentq_delight_cubature_v1.py \
    --env-id HalfCheetah-v4 --num-envs 16 --num-steps 1024 \
    --total-timesteps 8000000 --seed 1 --env-threads 2 --compile \
    --exp-name "$run_name" --actor-objective "$objective" --actor-sale
