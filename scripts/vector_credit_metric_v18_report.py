"""Compare the two fresh v18 fitting objectives against the completed v17 run."""
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
    'v18 score-weighted credit metric': 'vector_credit_metric_v18_8M',
    'v18 without score weighting': 'vector_credit_metric_v18_unweighted_8M',
}
REFERENCE = RUNS / 'HalfCheetah-v4__vector_predictive_distribution_v17_8M__1__1789592871354193081/result.json'


def summarize(rows, status, artifact):
    peak = max(rows, key=lambda row: row[RETURN])
    matched = {}
    for step in (1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000):
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
    }


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for label, rows in curves.items():
        steps = [row['step'] / 1e6 for row in rows]
        axes[0].plot(steps, [row[RETURN] for row in rows], label=label)
        axes[1].plot(steps, [row['critic/component_td_rms'] for row in rows], label=label)
        axes[2].plot(steps, [row['credit/model_alpha_beta_rms'] / max(row['credit/residual_alpha_beta_rms'], 1e-30)
                            for row in rows], label=label)
    axes[0].set_ylabel('Last-100 training episode return')
    axes[1].set_ylabel('Post-fit component TD RMS (raw units)')
    axes[2].set_ylabel('Model / residual local credit RMS')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Collected transitions (millions)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('v18 critic fitting intervention — fresh seed 1, HalfCheetah-v4')
    output = ROOT / 'docs/vector-credit-metric-v18-results'
    output.with_suffix('.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    fig.savefig(output.with_suffix('.png'), dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    print(json.dumps({name: {key: value for key, value in item.items() if key != 'final_metrics'}
                      for name, item in summaries.items()}))


if __name__ == '__main__':
    main()
