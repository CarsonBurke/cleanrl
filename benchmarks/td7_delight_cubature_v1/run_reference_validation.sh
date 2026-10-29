#!/usr/bin/env bash
set -Eeuo pipefail
.venv/bin/python -m pytest -q \
    tests/test_td7_latentq_delight_reference_v10.py \
    tests/test_td7_value_floor_checkpoint.py tests/test_checkpointing.py \
    tests/test_autocull.py tests/test_autocull_mlq.py
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/benchmark_exact.py \
    cleanrl/td7/delight/td7_latentq_delight_hermite_integrated_v8.py \
    cleanrl/td7/delight/td7_latentq_delight_reference_v10.py \
    --actor-objective delight --actor-advantage point --actor-baseline latent \
    --gate-scale raw --gate-scope all --target-smoothing-sigma 0.2 \
    --steps 2000 --repeats 3 \
    --output benchmarks/td7_delight_cubature_v1/reference_v10_latent_validation.json
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/benchmark_exact.py \
    cleanrl/td7/delight/td7_latentq_delight_reference_v10.py \
    --actor-objective delight --actor-baseline modal --gate-scale raw --gate-scope all \
    --target-smoothing-sigma 0.2 --steps 2000 --repeats 3 \
    --output benchmarks/td7_delight_cubature_v1/reference_v10_modal_validation.json
