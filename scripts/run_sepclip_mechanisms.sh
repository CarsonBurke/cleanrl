#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit}"
kind="$1"
case "$kind" in
  adam_reference) script=cleanrl/awr/ppo_continuous_action_ppo_sepclip_control_v1.py; extra=() ;;
  noise_full) script=cleanrl/ppo_continuous_action_sepclip_noise_polar_v6.py; extra=(--noise-multiplier 1 --normuon-lr 0.005) ;;
  noise_low) script=cleanrl/ppo_continuous_action_sepclip_noise_polar_v6.py; extra=(--noise-multiplier 0.3 --normuon-lr 0.005) ;;
  noise_large_step) script=cleanrl/ppo_continuous_action_sepclip_noise_polar_v6.py; extra=(--noise-multiplier 1 --normuon-lr 0.01) ;;
  noise_zero_control) script=cleanrl/ppo_continuous_action_sepclip_noise_polar_v6.py; extra=(--noise-multiplier 0 --normuon-lr 0.005) ;;
  noise_equalizer) script=cleanrl/ppo_continuous_action_sepclip_noise_polar_v6.py; extra=(--noise-multiplier 1 --normuon-lr 0.005 --lane-equalize) ;;
  rollout_clock) script=cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py; extra=(--slow-clock rollout --normuon-lr 0.0025) ;;
  rollout_clock_full) script=cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py; extra=(--slow-clock rollout --normuon-lr 0.005) ;;
  fisher_anvil) script=cleanrl/ppo_continuous_action_sepclip_fisher_head_v8.py; extra=(--actor-matrix-optimizer anvil2 --normuon-lr 0.0025 --head-target-kl 0.001) ;;
  fisher_anvil_small) script=cleanrl/ppo_continuous_action_sepclip_fisher_head_v8.py; extra=(--actor-matrix-optimizer anvil2 --normuon-lr 0.0025 --head-target-kl 0.0003) ;;
  fisher_adam_control) script=cleanrl/ppo_continuous_action_sepclip_fisher_head_v8.py; extra=(--actor-matrix-optimizer adam --head-target-kl 0.001) ;;
  sample_fisher_anvil_v9) script=cleanrl/ppo_continuous_action_sepclip_sample_fisher_head_v9.py; extra=(--actor-matrix-optimizer anvil2 --normuon-lr 0.0025 --head-target-kl 0.001 --head-normalization sample) ;;
  sample_fisher_adam_v9) script=cleanrl/ppo_continuous_action_sepclip_sample_fisher_head_v9.py; extra=(--actor-matrix-optimizer adam --head-target-kl 0.001 --head-normalization sample) ;;
  kfac_radius_control_v9) script=cleanrl/ppo_continuous_action_sepclip_sample_fisher_head_v9.py; extra=(--actor-matrix-optimizer anvil2 --normuon-lr 0.0025 --head-target-kl 0.00055 --head-normalization kronecker) ;;
  policy_budget_anvil_v11) script=cleanrl/ppo_continuous_action_sepclip_policy_budget_anvil_v11.py; extra=(--spectral-power 0.5 --normuon-lr 0.0025 --policy-calibration) ;;
  policy_budget_cascade_control_v11) script=cleanrl/ppo_continuous_action_sepclip_policy_budget_anvil_v11.py; extra=(--spectral-power 0 --normuon-lr 0.0025 --policy-calibration) ;;
  *) exit 2 ;;
esac
run_name="sepclip_mechanisms_${kind}"
.venv/bin/python - "$run_name" "$script" <<'PY'
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

run_name, trainer = sys.argv[1:]
paths = [Path(trainer), Path('scripts/run_sepclip_mechanisms.sh')]
paths.extend(sorted(Path('cleanrl/shared').glob('*.py')))
paths.append(Path('cleanrl/shared/host_kernel.c'))
manifest = dict(run_name=run_name, captured_at=time.time(), python_version=sys.version,
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    package_versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'gymnasium', 'tyro')},
    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
out = Path('benchmarks/sepclip_mechanisms/execution_sources')
out.mkdir(parents=True, exist_ok=True)
(out/f'{run_name}_{time.time_ns()}.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
.venv/bin/python -u scripts/autocull.py --ref normuon_audit_8m_adam \
  --only sepclip_mechanisms_ --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u "$script" \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" --total-timesteps 8000000 --seed 1 \
  --learning-rate 0.0024 \
  --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead "${extra[@]}"
