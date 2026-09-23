#!/usr/bin/env bash
# Matched 8M HalfCheetah ablation for the target-standardized HL-Gauss critic.
# One trainer, one differing flag group per arm. Run from the repo root.
set -euo pipefail

AFTER="${1:?usage: submit_stdhlgauss_v1.sh <verification-job-id>}"
SCRIPT=cleanrl/ppo_continuous_action_normres_stdhlgauss_v1.py
COMMON=(
  --env-id HalfCheetah-v4 --seed 1 --total-timesteps 8000000
  --num-envs 16 --num-steps 2048 --num-minibatches 1 --update-epochs 10
  --learning-rate 0.0096 --anneal-lr --no-norm-adv --reward-norm
  --env-threads 2 --compile --compile-mode reduce-overhead --staggered-starts
)

submit() {
  local name="$1"
  shift
  mlq submit --json --name "$name" --max-parallel-runs 6 --time-limit 45m --priority 5 \
    --after-success "$AFTER" --cwd "$PWD" \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env CLEANRL_ENV_SPIN=5000 -- \
    .venv/bin/python -u "$SCRIPT" --exp-name "$name" "${COMMON[@]}" "$@" \
    | .venv/bin/python -c 'import json,sys; job=json.load(sys.stdin); print(job["id"], job["name"])'
}

# Control: the frozen scalar recipe (best known 50M configuration at 8M budget).
submit stdhlg_v1_mse_8m --value-loss mse
# Paper's missing control: same K-way head, MSE on its decoded mean.
submit stdhlg_v1_msesoftmax_8m --value-loss mse_softmax --value-bins 101
# The control that std_proxy_v2 made decisive: a scalar head carrying the same
# per-rollout target standardization. If it matches the categorical arms, the
# gain is drift absorption and classification contributes nothing.
submit stdhlg_v1_popart_8m --value-loss mse_popart
submit stdhlg_v1_popart_w256_8m --value-loss mse_popart --critic-width 256
# Smoothing sweep at fixed resolution (bin width 0.1 target std).
submit stdhlg_v1_hlg_s075_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 0.75
submit stdhlg_v1_hlg_s2_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 2.0
submit stdhlg_v1_hlg_s5_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 5.0
# Resolution at fixed sigma (0.2 target std): 0.05 vs 0.1 target std bins.
submit stdhlg_v1_hlg_k201_s4_8m --value-loss hlgauss --value-bins 201 --value-sigma-bins 4.0
# Gradient-scale detail that vf_coef silently changed for every previous version.
submit stdhlg_v1_hlg_s2_matched_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 2.0 --ce-scale matched
# Every previous categorical trainer silently dropped clip_vloss; isolate it.
submit stdhlg_v1_hlg_s2_noclipv_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 2.0 --no-clip-vloss
# Capacity, the paper's stated gain mechanism, with its own scalar control.
submit stdhlg_v1_hlg_s2_w256_8m --value-loss hlgauss --value-bins 101 --value-sigma-bins 2.0 --critic-width 256
submit stdhlg_v1_mse_w256_8m --value-loss mse --critic-width 256
