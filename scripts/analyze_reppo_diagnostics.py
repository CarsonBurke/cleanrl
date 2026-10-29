"""Summarize the checkpointed v27 diagnosis from existing TensorBoard/oracle data.

Read-only analysis, no model loading or simulation. Missing windows stay missing;
the report never infers completion from a run name or treats training errors as truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _runs import RunScalars, find_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='reppo_diagnose_v29')
    parser.add_argument('--out', type=Path, default=Path('benchmarks/reppo_diagnose_v29'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patterns = [args.prefix, 'reppo_latentroll_v27_h8_50M__1__1790378891',
                'reppo_refresh_v25_lam0p8_50M__1__1790376907']
    windows = [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16)]
    summary, curves = {}, {}
    for path in find_runs([Path('runs')], patterns, env_filter='HalfCheetah-v4'):
        if 'oracle' in path.name:
            continue
        run = RunScalars(path)
        if not run.max_step():
            continue
        entry = {'path': str(path), 'last_recorded_step': run.max_step(), 'windows': {}}
        for low, high in windows:
            values = {}
            for tag in sorted(run.tags):
                if tag.startswith(('timing/', 'charts/SPS', 'charts/interval')):
                    continue
                steps, scalar = run.series(tag)
                valid = (steps >= low * 1e6) & (steps < high * 1e6)
                if valid.any():
                    values[tag] = {'mean': float(scalar[valid].mean()),
                                   'sd': float(scalar[valid].std()), 'count': int(valid.sum())}
            entry['windows'][f'{low}-{high}M'] = values
        summary[path.name] = entry
        curves[path.name] = run
    oracle = {}
    for path in sorted(Path('runs/reppo_latentroll_oracle').glob(f'*{args.prefix}*_combined.json')):
        oracle[path.name] = json.loads(path.read_text())
    attribution = {path.name: json.loads(path.read_text()) for path in
                   sorted(Path('runs/reppo_latentroll_oracle').glob(f'*{args.prefix}*_gradients.json'))}
    gates = {path.name: json.loads(path.read_text()) for path in
             sorted(Path('runs/reppo_latentroll_oracle').glob(f'*{args.prefix}*_policy_gates.json'))}
    sensitivity = {path.name: json.loads(path.read_text()) for path in args.out.glob('oracle_*tail_sensitivity.json')}
    sensitivity_by_source = {Path(report['source']).stem: report for report in sensitivity.values()}
    result = {'training': summary, 'oracle': oracle, 'gradient_attribution': attribution, 'policy_gates': gates,
              'tail_sensitivity': sensitivity,
              'limitations': ['Single seed; windows and episode samples are correlated.',
                              'Training/next-rollout Q errors use learned bootstrap targets.',
                              'Consult mlq job metadata for completion and culling.']}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2))
    lines = ['# REPPO latent-roll diagnostic results', '',
             'Window means; absent windows are not extrapolated. Queue outcomes are separate.', '',
             '| Run | Recorded M steps | 6–8M | 10–12M | 14–16M |',
             '|---|---:|---:|---:|---:|']
    for name, entry in summary.items():
        cells = [entry['windows'][w].get('charts/episodic_return', {}).get('mean')
                 for w in ['6-8M', '10-12M', '14-16M']]
        lines.append('| ' + ' | '.join([name.split('__')[1], f"{entry['last_recorded_step']/1e6:.3f}",
                                        *['—' if c is None else f'{c:,.0f}' for c in cells]]) + ' |')
    lines += ['', f'{len(oracle)} combined oracle reports available; raw statistics are in results.json.', '',
              'Oracle conclusions must distinguish first-action changes from full-policy changes,',
              'frozen pre/post critics, soft versus raw returns, and confidence intervals.']
    if oracle:
        lines += ['', '## Simulator policy comparisons', '',
                  'Direct 1000-step continuing-state soft returns; fixed pre-update temperature, no critic tail.',
                  'Intervals resample environment trajectories; they do not include training-seed variation.', '',
                  '| Arm / checkpoint | Update | Critic predicted first-action gain | Simulator gain | Trajectory 95% interval | Actual minus predicted 95% interval | Full-policy raw discounted gain |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for report in oracle.values():
            arm = Path(report['snapshot']).parent.name.split('__')[1]
            for update, comparison in report.get('policy', {}).items():
                if not isinstance(comparison, dict) or 'results' not in comparison:
                    continue
                horizon = comparison['results'].get('1000')
                if not horizon:
                    continue
                soft = horizon['first_action']['soft']
                lo, hi = soft.get('trajectory_ci95_low', soft['ci95_low']), soft.get('trajectory_ci95_high', soft['ci95_high'])
                elo, ehi = soft['prediction_error_trajectory_ci95_low'], soft['prediction_error_trajectory_ci95_high']
                lines.append(f"| {arm} / {report['global_step']/1e6:.3f}M | {update} | "
                             f"{soft['predicted_mean']:.5f} | {soft['mean']:.5f} | [{lo:.5f}, {hi:.5f}] | "
                             f"[{elo:.5f}, {ehi:.5f}] | "
                             f"{horizon['full_policy']['raw']['mean']:.4f} |")
        lines += ['', '## Finite-difference evidence', '',
                  '300-step soft labels retain critic bootstrap tails, including at earlier time limits.', '',
                  'The legacy pooled metric is mean directional gain divided by RMS oracle gradient magnitude,',
                  'not ordinary pooled cosine. Heavy-tailed gradient norms can lower it even with aligned directions.', '',
                  '| Arm / checkpoint | Subset | Gradient center | Normalized directional gain (h=.2) | SE | Within-state advantage correlation |',
                  '|---|---|---|---:|---:|---:|']
        for report in oracle.values():
            arm = Path(report['snapshot']).parent.name.split('__')[1]
            for subset in ['all', 'cruising']:
                values = report.get('finite_difference', {}).get('results', {}).get(subset, {}).get('300')
                if values:
                    for center in ['star', 'a0']:
                        gradient = values[f'grad_{center}_h0.2']
                        lines.append(f"| {arm} / {report['global_step']/1e6:.3f}M | {subset} | {center} | "
                                     f"{gradient['cos_pooled_corrected']:.4f} | {gradient['cos_pooled_corrected_se']:.4f} | "
                                     f"{values['advantage']['corr_corrected']:.4f} |")
    if gates:
        lines += ['', '## Behavior-policy KL gate', '',
                  'Last-update strata refer to pre-update KL against the rollout behavior policy.',
                  'They describe sampled states, not independent causal interventions.', '',
                  '| Arm / checkpoint | Stratum | States | Predicted soft gain | Simulator soft gain | Actual minus predicted 95% interval |',
                  '|---|---|---:|---:|---:|---:|']
        for report in gates.values():
            arm = Path(report['snapshot']).parent.name.split('__')[1]
            for name, stratum in report['comparisons']['last_actor_update']['strata'].items():
                if 'horizons' not in stratum:
                    continue
                soft = stratum['horizons']['1000']['first_action']['soft_gain']
                lo, hi = soft['prediction_error_trajectory_ci95_low'], soft['prediction_error_trajectory_ci95_high']
                lines.append(f"| {arm} / {report['global_step']/1e6:.3f}M | {name} | {stratum['states']} | "
                             f"{soft['predicted_mean']:.5f} | {soft['mean']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    if attribution:
        lines += ['', '## Critic gradient attribution', '',
                  'Means over four disjoint minibatches from a fresh saved rollout. These are frozen-checkpoint',
                  'gradients against bootstrap labels, not historical optimizer gradients.', '',
                  '| Arm / checkpoint | Total norm | Encoder squared-norm share | Value-head share | Learned-offset share |',
                  '|---|---:|---:|---:|---:|']
        for report in attribution.values():
            arm = Path(report['snapshot']).parent.name.split('__')[1]
            stats = report['summary']
            values = [stats.get(tag, {}).get('mean', 0.) for tag in
                      ['total/grad_norm', 'group/encoder/squared_norm_share',
                       'group/value_head/squared_norm_share', 'group/zero_dist/squared_norm_share']]
            lines.append(f"| {arm} / {report['global_step']/1e6:.3f}M | " +
                         ' | '.join(f'{v:.4f}' for v in values) + ' |')
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n')
    if not curves:
        print('No campaign training data yet; wrote empty result inventory.')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('charts/episodic_return', 'Training return'),
              ('eval/deterministic_return', 'Deterministic return'),
              ('probe/encoder_ce_latent_cosine', 'Value / latent encoder gradient cosine'),
              ('probe/encoder_ce_reward_cosine', 'Value / reward encoder gradient cosine'),
              ('probe/encoder_latent_weighted_grad_norm', 'Weighted latent encoder gradient norm'),
              ('probe/h8_normalized_latent_mse', 'H8 normalized latent prediction MSE')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (tag, title) in zip(axes.flat, panels):
        for name, run in curves.items():
            steps, values = run.series(tag)
            x, y = [], []
            for low in np.arange(0, 16.5, .5):
                mask = (steps >= low * 1e6) & (steps < (low + .5) * 1e6)
                if mask.any():
                    x.append(float(steps[mask].mean() / 1e6))
                    y.append(float(values[mask].mean()))
            if x:
                ax.plot(x, y, label=run.variant, linewidth=1.4)
        ax.set(title=title, xlabel='Environment steps (millions)', xlim=(0, 16.2))
        ax.axvline(12, color='grey', linestyle=':', linewidth=.8)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(args.out / 'diagnostics.png', dpi=160)
    if oracle:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        arms = sorted({Path(report['snapshot']).parent.name.split('__')[1] for report in oracle.values()})
        for arm in arms:
            records = sorted([report for report in oracle.values()
                              if Path(report['snapshot']).parent.name.split('__')[1] == arm],
                             key=lambda report: report['global_step'])
            label = arm.replace('reppo_diagnose_v29_', '').replace('_16M', '')
            x, y, low, high = [], [], [], []
            for report in records:
                values = report.get('finite_difference', {}).get('results', {}).get('all', {}).get('300')
                if values:
                    x.append(report['global_step'] / 1e6)
                    y.append(values['advantage']['corr_corrected'])
                    source = f"{Path(report['snapshot']).parent.name}_{report['global_step']}"
                    cell = next((c for c in sensitivity_by_source.get(source, {}).get('advantage_cells', [])
                                 if c['horizon'] == 300 and c['subset'] == 'all' and c['restriction'] == 'original'), None)
                    interval = cell['advantage_corr_corrected']['trajectory_ci95'] if cell else [float('nan')] * 2
                    low.append(y[-1] - interval[0])
                    high.append(interval[1] - y[-1])
            if x:
                axes[0].errorbar(x, y, yerr=[low, high], fmt='o-', capsize=4, label=label)
            for axis, objective in zip(axes[1:], ['raw', 'soft']):
                x, y, low, high = [], [], [], []
                for report in records:
                    values = report.get('policy', {}).get('update', {}).get('results', {}).get('1000')
                    if values:
                        stats = values['full_policy'][objective]
                        x.append(report['global_step'] / 1e6)
                        y.append(stats['mean'])
                        low.append(stats['mean'] - stats['trajectory_ci95_low'])
                        high.append(stats['trajectory_ci95_high'] - stats['mean'])
                if x:
                    axis.errorbar(x, y, yerr=[low, high], fmt='o-', capsize=4, label=label)
        for axis, title in zip(axes, ['Within-state action-value correlation\n(noise corrected; critic tail)',
                                     'Whole update: raw discounted gain\n(1000 steps, no critic tail)',
                                     'Whole update: soft gain\n(fixed pre-update temperature)']):
            axis.set(title=title, xlabel='Environment steps (millions)')
            axis.axhline(0, color='grey', linewidth=.8)
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle('One seed; gain intervals resample environment trajectories, not training seeds', fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / 'oracle_comparisons.png', dpi=160)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
