"""Analyze fresh 20M v18 comparisons and matched earlier training evidence."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
RETURN = 'charts/episodic_return_mean_100'
VARIANTS = {
    'v18 score-weighted credit metric': 'vector_credit_metric_v18_20M',
    'v18 without score weighting': 'vector_credit_metric_v18_unweighted_20M',
}
REFERENCE = RUNS / 'HalfCheetah-v4__vector_predictive_distribution_v17_8M__1__1789592871354193081/result.json'


def summarize(rows, status, artifact):
    peak = max(rows, key=lambda row: row[RETURN])
    matched = {}
    for step in (2_000_000, 4_000_000, 8_000_000, 12_000_000, 16_000_000, 20_000_000):
        if step <= rows[-1]['step']:
            row = min(rows, key=lambda row: abs(row['step'] - step))
            matched[str(step)] = {'step': row['step'], 'return': row[RETURN]}
    return {
        'status': status, 'artifact': str(artifact.relative_to(ROOT)),
        'steps': rows[-1]['step'], 'final_return': rows[-1][RETURN],
        'peak_return': peak[RETURN], 'peak_step': peak['step'],
        'matched_steps': matched,
        'mean_kl': statistics.mean(row['policy/exact_joint_kl'] for row in rows),
        'median_cg_residual': statistics.median(row['natural/cg_residual_ratio'] for row in rows),
        'final_metrics': rows[-1],
        'phase_diagnostics': phase_diagnostics(rows),
    }



def phase_diagnostics(rows):
    keys = ['grad/critic_preclip_norm', 'natural/cg_residual_ratio',
            'policy/importance_max', 'policy/importance_ess_fraction',
            'policy/concentration', 'policy/entropy',
            'critic/component_td_rms', 'critic/utility_td_rms',
            'credit/observed_residual_rms', 'credit/model_alpha_beta_rms',
            'credit/residual_alpha_beta_rms', 'credit/model_residual_cosine',
            'losses/optimized_mean_block_postfit', 'losses/moment_block_postfit',
            'policy/accepted_gain', 'policy/exact_joint_kl',
            'critic/score_energy_mean', 'critic/score_energy_max']
    result = {}
    for low, high in ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20.1)):
        selected = [row for row in rows if low * 1e6 < row['step'] <= high * 1e6]
        if not selected:
            continue
        group = {'updates': len(selected),
                 'first_return': selected[0][RETURN], 'last_return': selected[-1][RETURN],
                 'mean_return': statistics.mean(row[RETURN] for row in selected),
                 'cg_above_0.1': sum(row['natural/cg_residual_ratio'] > .1 for row in selected),
                 'importance_ess_below_0.5': sum(row['policy/importance_ess_fraction'] < .5 for row in selected),
                 'importance_ess_below_0.1': sum(row['policy/importance_ess_fraction'] < .1 for row in selected),
                 'mean_epoch_gradient_above_clip': sum(row['grad/critic_preclip_norm'] > .5 for row in selected),
                 'accepted_updates': sum(row['policy/accepted_scale'] > 0 for row in selected),
                 'model_residual_credit_ratio_median': statistics.median(
                     row['credit/model_alpha_beta_rms'] / max(row['credit/residual_alpha_beta_rms'], 1e-30)
                     for row in selected)}
        for key in keys:
            values = [row[key] for row in selected if key in row]
            if values:
                group[key] = {'median': statistics.median(values), 'mean': statistics.mean(values),
                              'min': min(values), 'max': max(values)}
        result[f'{low}-{high}M'] = group
    return result


def main():
    reference = json.loads(REFERENCE.read_text())
    curves = {'v17 reference': reference['progress']}
    summaries = {'v17 reference': summarize(reference['progress'], reference['status'], REFERENCE)}
    for label, experiment in VARIANTS.items():
        paths = list(RUNS.glob(f'HalfCheetah-v4__{experiment}__1__*/progress.json'))
        if not paths:
            summaries[label] = {'status': 'no_training_progress'}
            continue
        if len(paths) != 1:
            raise RuntimeError(f'Expected one fresh run for {experiment}, found {len(paths)}')
        result_path = paths[0].parent / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows = result.get('progress') or json.loads(paths[0].read_text())
        curves[label] = rows
        summaries[label] = summarize(rows, result.get('status', 'no_final_result'), paths[0])
        old_paths = list(RUNS.glob(f"HalfCheetah-v4__{experiment.replace('20M', '8M')}__1__*/progress.json"))
        if len(old_paths) == 1:
            previous = json.loads(old_paths[0].read_text())
            paired = [(a, b) for a, b in zip(rows, previous) if a['step'] == b['step']]
            summaries[label]['fresh_prefix_comparison'] = {
                'matched_updates': len(paired),
                'max_absolute_return_difference': max((abs(a[RETURN] - b[RETURN]) for a, b in paired), default=None),
                'return_equal_updates': sum(a[RETURN] == b[RETURN] for a, b in paired)}

    fig, grid = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = grid.flatten()
    for label, rows in curves.items():
        steps = [row['step'] / 1e6 for row in rows]
        axes[0].plot(steps, [row[RETURN] for row in rows], label=label)
        axes[1].plot(steps, [row['critic/component_td_rms'] for row in rows], label=label)
        axes[2].plot(steps, [row['credit/model_alpha_beta_rms'] / max(row['credit/residual_alpha_beta_rms'], 1e-30)
                            for row in rows], label=label)
        axes[3].plot(steps, [row['natural/cg_residual_ratio'] for row in rows], label=label)
        axes[4].plot(steps, [row['grad/critic_preclip_norm'] for row in rows], label=label)
        axes[5].plot(steps, [row['policy/importance_ess_fraction'] for row in rows], label=label)
    axes[3].set_ylabel('Recomputed relative CG residual')
    axes[4].set_ylabel('Mean preclip critic gradient norm')
    axes[5].set_ylabel('Importance-ratio ESS fraction')
    axes[3].set_yscale('log')
    axes[4].set_yscale('log')
    axes[4].axhline(.5, linestyle='--', color='gray')
    axes[0].set_ylabel('Last-100 training episode return')
    axes[1].set_ylabel('Post-fit component TD RMS (raw units)')
    axes[2].set_ylabel('Model / residual local credit RMS')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Collected transitions (millions)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('v18 at 20M — fresh seed 1, HalfCheetah-v4')
    output = ROOT / 'docs/vector-credit-metric-v18-20m-results'
    output.with_suffix('.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    fig.savefig(output.with_suffix('.png'), dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    print(json.dumps({name: {key: value for key, value in item.items() if key not in ('final_metrics', 'phase_diagnostics')}
                      for name, item in summaries.items()}))


if __name__ == '__main__':
    main()
