"""Summarize this task's completed and explicitly culled runs; no model loading."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
REFERENCE = RUNS / 'HalfCheetah-v4__latent_costate_v10_no_projection_8M__1__1789544944088860507/result.json'
ARMS = [('v10 reference', 7527, REFERENCE)]
for label, name, job in [('Gamma 0.99', 'gamma099', 7541),
                          ('Full batch', 'fullbatch', 7542),
                          ('No repeated sweeps', 'no_sweeps', 7543)]:
    matches = list(RUNS.glob(f'HalfCheetah-v4__latent_costate_v11_{name}_8M__1__*/progress.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one run for {name}, got {len(matches)}')
    result_path = matches[0].parent / 'result.json'
    if not result_path.exists() and not (matches[0].parent / 'ablation_outcome.json').exists():
        raise RuntimeError(f'{name} is not completed or explicitly culled')
    ARMS.append((label, job, result_path if result_path.exists() else matches[0]))

rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for label, job, path in ARMS:
    completed = path.name == 'result.json'
    result = json.loads(path.read_text()) if completed else {}
    progress = result['progress'] if completed else json.loads(path.read_text())
    returns = [p['charts/episodic_return_mean_100'] for p in progress]
    steps = [p['step'] for p in progress]
    peak = max(progress, key=lambda p: p['charts/episodic_return_mean_100'])
    row = dict(label=label, job=job, artifact=str(path.relative_to(ROOT)),
               status='completed' if completed else 'cancelled_underperforming',
               transitions=progress[-1]['step'],
               final_return=result.get('final_return_mean_100'),
               latest_return=returns[-1],
               peak_return=peak['charts/episodic_return_mean_100'], peak_step=peak['step'],
               actor_attempted=len(progress),
               actor_accepted=sum(p['policy/accepted_scale'] > 0 for p in progress),
               mean_kl=sum(p['policy/exact_joint_kl'] for p in progress) / len(progress),
               mean_cg_residual=sum(p['natural/cg_residual_ratio'] for p in progress) / len(progress),
               last_costate_rms=progress[-1]['credit/next_costate_rms'],
               max_costate_rms=max(p['credit/next_costate_rms'] for p in progress),
               last_model_loss=progress[-1]['losses/model_normalized_mse_half'],
               model_optimizer_steps=progress[-1].get('updates/model_optimizer', len(progress) * 80),
               critic_optimizer_steps=progress[-1].get('updates/critic_optimizer', len(progress) * 64),
               critic_target_refreshes=progress[-1].get('updates/critic_target_refreshes', len(progress) * 8),
               matched_steps={})
    for milestone in (1000000, 2000000, 4000000, 6000000, 8000000):
        if milestone > steps[-1]:
            continue
        match = min(progress, key=lambda p: abs(p['step'] - milestone))
        row['matched_steps'][str(milestone)] = dict(step=match['step'], value=match['charts/episodic_return_mean_100'])
    rows.append(row)
    axes[0].plot([s / 1e6 for s in steps], returns, label=label)
    axes[1].plot([s / 1e6 for s in steps], [p['credit/next_costate_rms'] for p in progress], label=label)
axes[0].set_ylabel('Last-100 training episode return')
axes[1].set_ylabel('Next-state costate RMS')
for ax in axes:
    ax.set_xlabel('Collected transitions (millions)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Three separate latent-costate ablations — fresh seed 1, HalfCheetah-v4')
output = ROOT / 'docs/latent-costate-ablation-v11-results'
fig.savefig(output.with_suffix('.png'), dpi=160)
fig.savefig(output.with_suffix('.svg'))
output.with_suffix('.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
for row in rows:
    print(json.dumps(row))
