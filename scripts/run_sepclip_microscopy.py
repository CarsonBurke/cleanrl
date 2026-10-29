"""Queue-only foreground runner, execution-source manifest and harness diagnostics."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'benchmarks/sepclip_weight_microscopy'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--name', required=True)
    cli.add_argument('--training', action='store_true')
    cli.add_argument('--output-json', type=Path)
    cli.add_argument('--source', action='append', default=[])
    cli.add_argument('command', nargs=argparse.REMAINDER)
    args = cli.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not args.name.startswith('sepclip_microscopy_'):
        cli.error('Require foreground command and sepclip_microscopy_ name')
    if not os.environ.get('XDG_RUNTIME_DIR'):
        raise RuntimeError('Submit --inherit-env XDG_RUNTIME_DIR')
    stamp = time.time_ns()
    sources = [Path(__file__), *[ROOT/p for p in args.source]]
    # Audits import one another and dispatch checkpoint architectures. Capture
    # the whole small sepclip source family so transitive local dependencies
    # remain attributable even when an explicit --source names only the entry.
    sources.extend(sorted((ROOT/'scripts').glob('audit_sepclip_*.py')))
    sources.extend(sorted((ROOT/'cleanrl').glob('ppo_continuous_action_sepclip_*.py')))
    sources.extend(sorted((ROOT/'cleanrl/shared').glob('*.py')))
    sources.extend([ROOT/'cleanrl/shared/host_kernel.c', ROOT/'cleanrl/shared/mujoco_batch.c'])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources if p.exists()}
    gym_distribution = importlib.metadata.distribution('gymnasium')
    environment_sources = [gym_distribution.locate_file(p) for p in (
        'gymnasium/envs/mujoco/assets/half_cheetah.xml',
        'gymnasium/envs/mujoco/half_cheetah_v4.py',
        'gymnasium/wrappers/time_limit.py')]
    manifest = dict(run_name=args.name, captured_at=stamp, command=command,
                    declared_max_parallel_runs=1,
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=ROOT,text=True).strip(),
                    python=sys.version,
                    packages={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','gymnasium','mujoco','tyro')},
                    environment_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in environment_sources if p.exists()},
                    source_sha256=hashes,
                    environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CLEANRL_ENV_SPIN','XDG_RUNTIME_DIR')})
    directory = ARTIFACTS/'execution_sources'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/f'{args.name}_{stamp}.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    supervisor = None
    result = None
    try:
        if args.training:
            supervisor = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/autocull.py',
                '--ref','normuon_audit_8m_adam','--only','sepclip_microscopy_',
                '--enforce','all','--yes','--watch','600'],cwd=ROOT)
        result = subprocess.run(command,cwd=ROOT).returncode
        if not args.training:
            from torch.utils.tensorboard import SummaryWriter
            payload = json.loads(args.output_json.read_text()) if result == 0 and args.output_json else {}
            writer = SummaryWriter(str(ROOT/'runs'/f'HalfCheetah-v4__{args.name}__1__{int(time.time())}'))
            writer.add_text('protocol', 'Read-only CUDA checkpoint diagnostic; not a training benchmark')
            writer.add_scalar('queue/exit_code',result,0)
            if args.output_json:
                writer.add_text('result_path',str(args.output_json))
            writer.add_text('execution_manifest',str(output))
            def log(value, prefix='audit', index=0):
                if isinstance(value, dict):
                    for k,v in value.items(): log(v,prefix+'/'+str(k),index)
                elif isinstance(value, list) and value and isinstance(value[0],dict):
                    for i,v in enumerate(value): log(v,prefix,i)
                elif isinstance(value,(int,float)) and not isinstance(value,bool):
                    writer.add_scalar(prefix,value,index)
            log(payload)
            writer.close()
    finally:
        if supervisor is not None:
            supervisor.terminate()
            try: supervisor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                supervisor.kill(); supervisor.wait()
        manifest.update(exit_code=result, finished_at=time.time_ns())
        output.write_text(json.dumps(manifest,indent=2)+'\n')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
