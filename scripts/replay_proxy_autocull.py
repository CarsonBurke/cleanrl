"""Replay proxy pruning on saved TensorBoard intervals; never launches a learner.

CPU-only analysis of recorded scalar metrics; never imports a learner or
initializes CUDA. Uses scripts._runs and the SAME ProxyCull policy the stock
runner enforces, not a second heuristic.
"""
import argparse
import json
from pathlib import Path

from scripts._runs import RunScalars
from cleanrl.shared.autocull import ProxyCull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--arm', default='covariance_original_random_sign')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/plasticity/autocull_replay.json'))
    args = parser.parse_args()
    metadata = json.loads((args.run / 'results.json').read_text())
    arm = metadata['arms'][args.arm]
    candidates = len(arm['grid'])
    names = ('error_ratio', 'prediction_energy_ratio') if args.arm.endswith('random_sign') else ('error_ratio',)
    tags = {f'{args.arm}/candidate_{i}/{name}' for i in range(candidates) for name in names}
    throughput_tag = f'{args.arm}/updates_per_second'
    tags.add(throughput_tag)
    scalars = RunScalars(args.run, tags)
    series = {(i, name): scalars.series(f'{args.arm}/candidate_{i}/{name}')
              for i in range(candidates) for name in names}
    steps = series[0, names[0]][0]
    if not len(steps) or any(len(values[0]) != len(steps) or not (values[0] == steps).all()
                             for values in series.values()):
        raise ValueError('missing or mismatched metric steps; no inferred decision')
    guard = ProxyCull(candidates, dict.fromkeys(names, 1e-4))
    switch = metadata['phases']['suffix_positive'][1]
    previous = 0
    decisions = []
    for i, step in enumerate(steps):
        after_switch = args.arm.endswith('planted') and previous >= switch
        decision = guard.observe(int(step),
                                 {name: [float(series[c, name][1][i]) for c in range(candidates)] for name in names},
                                 phase='reversed' if after_switch else 'stationary',
                                 phase_start=switch if after_switch else 0)
        if args.arm.endswith('planted') and step == switch:
            decision = None
        previous = int(step)
        if decision:
            decisions.append(decision)
            break
    report = {'source': str(args.run), 'arm': args.arm,
              'observed_horizon': int(steps[-1]), 'first_prune': decisions[0] if decisions else None,
              'observations_avoided': int(steps[-1]) - decisions[0]['step'] if decisions else 0,
              'interpretation': 'retrospective replay of default policy; no new training and not a statistical superiority test'}
    time_steps, speeds = scalars.series(throughput_tag)
    replay_times = {int(step): int(step) / float(speed) for step, speed in zip(time_steps, speeds)
                    if speed > 0}
    if decisions and decisions[0]['step'] in replay_times and int(steps[-1]) in replay_times:
        stopped = replay_times[decisions[0]['step']]
        full = replay_times[int(steps[-1])]
        report.update(replay_seconds_to_prune=stopped, full_replay_seconds=full,
                      replay_seconds_avoided=full - stopped,
                      avoided_fraction=report['observations_avoided'] / int(steps[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
