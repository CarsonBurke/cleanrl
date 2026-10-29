#!/usr/bin/env bash
set -Eeuo pipefail
.venv/bin/python -m pytest -q \
    tests/test_td7_latentq_delight_projected_v11.py \
    tests/test_td7_sigma_checkpoint_diagnostic.py \
    tests/test_td7_value_floor_checkpoint.py tests/test_checkpointing.py \
    tests/test_autocull.py tests/test_autocull_mlq.py
# Fixed zero entropy for the reward-surrogate comparison only; captured steps
# and checkpoint validation use each trainer's actual configured controller.
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/benchmark_exact.py \
    cleanrl/td7/delight/td7_latentq_delight_reference_v10.py \
    cleanrl/td7/delight/td7_latentq_delight_projected_v11.py \
    --actor-objective delight --actor-baseline latent --gate-scale raw --gate-scope all \
    --surprisal relative --math-actor-entropy-coefficient 0 --target-smoothing-sigma 0.2 \
    --steps 2000 --repeats 3 \
    --output benchmarks/td7_delight_cubature_v1/projected_v11_relative_validation.json
.venv/bin/python -u benchmarks/td7_delight_cubature_v1/benchmark_exact.py \
    cleanrl/td7/delight/td7_latentq_delight_projected_v11.py \
    --actor-objective delight --actor-baseline latent --gate-scale raw --gate-scope all \
    --surprisal density --target-smoothing-sigma 0.2 --steps 2000 --repeats 3 \
    --output benchmarks/td7_delight_cubature_v1/projected_v11_density_validation.json
