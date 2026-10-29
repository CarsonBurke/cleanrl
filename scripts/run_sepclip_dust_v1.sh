#!/usr/bin/env bash
set -euo pipefail
: "${XDG_RUNTIME_DIR:?Pass --inherit-env XDG_RUNTIME_DIR to mlq submit}"
kind="${1:-dust}"
case "$kind" in
  dust) extra=() ;;
  backprop) extra=(--no-dust) ;;
  *) exit 2 ;;
esac
run_name="sepclip_dust_v1_${kind}"
script=cleanrl/dust/ppo_continuous_action_sepclip_activation_credit_v1.py
.venv/bin/python - "$run_name" "$script" <<'PY'
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
name, trainer = sys.argv[1:]
paths = [Path(trainer), Path('scripts/run_sepclip_dust_v1.sh')]
paths.extend(sorted(Path('cleanrl/shared').glob('*.py')))
paths.append(Path('cleanrl/shared/host_kernel.c'))
manifest = dict(run_name=name, captured_at=time.time(),
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    upstream='https://github.com/qlabs-eng/dust',
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
out=Path('benchmarks/sepclip_dust_v1/execution_sources')
out.mkdir(parents=True, exist_ok=True)
(out/f'{name}_{time.time_ns()}.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
.venv/bin/python -u scripts/autocull.py --ref normuon_audit_8m_adam \
  --only sepclip_dust_v1_ --enforce all --yes --watch 600 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; wait "$supervisor" 2>/dev/null || true' EXIT
.venv/bin/python -u "$script" \
  --env-id HalfCheetah-v4 --num-envs 16 --env-threads 2 \
  --exp-name "$run_name" --total-timesteps 8000000 --seed 1 \
  --learning-rate 0.0024 --no-norm-adv --no-clip-vloss --clip-coef-upper 0.2 \
  --compile --compile-mode reduce-overhead "${extra[@]}"
