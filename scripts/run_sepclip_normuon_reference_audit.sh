#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit}"
kind="$1"
case "$kind" in
  adam) script=cleanrl/awr/ppo_continuous_action_ppo_sepclip_control_v1.py; extra=() ;;
  rows) script=cleanrl/ppo_continuous_action_sepclip_normuon_split_diagnostic_v2.py; extra=(--normuon-network actor --normuon-lr 0.005) ;;
  long) script=cleanrl/ppo_continuous_action_sepclip_longaxis_normuon_v3.py; extra=(--normuon-network actor --normuon-lr 0.005) ;;
  *) exit 2 ;;
esac
# Same eight-million-step LR horizon in all three fresh runs. The old 50M
# control has a different schedule, so it cannot serve as this comparison.
run_name="normuon_audit_8m_${kind}"
.venv/bin/python -u scripts/autocull.py \
  --ref normuon_audit_8m_adam --only normuon_audit_8m_ \
  --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u "$script" \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" --total-timesteps 8000000 --seed 1 \
  --learning-rate 0.0024 --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead "${extra[@]}"
