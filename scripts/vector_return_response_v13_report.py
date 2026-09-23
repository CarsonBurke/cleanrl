"""Summarize only this task's frozen v13 run and its known references."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
paths = list(RUNS.glob('HalfCheetah-v4__vector_return_response_v13_8M__1__*/progress.json'))
if len(paths) != 1:
    raise RuntimeError(f'Expected one v13 run, got {len(paths)}')
run = paths[0].parent
result_path = run / 'result.json'
if result_path.exists():
    result = json.loads(result_path.read_text())
    progress = result['progress']
    status = 'completed'
else:
    outcome = json.loads((run / 'experiment_outcome.json').read_text())
    result = {}
    progress = json.loads(paths[0].read_text())
    status = outcome['status']
ret = 'charts/episodic_return_mean_100'
peak = max(progress, key=lambda p: p[ret])
last = progress[-1]
summary = dict(status=status, artifact=str((result_path if result else paths[0]).relative_to(ROOT)),
               transitions=last['step'], final_return=result.get('final_return_mean_100'),
               last_logged_return=last[ret], peak_return=peak[ret], peak_step=peak['step'],
               actor_attempted=len(progress), actor_accepted=sum(p['policy/accepted_scale'] > 0 for p in progress),
               mean_kl=sum(p['policy/exact_joint_kl'] for p in progress) / len(progress),
               min_kl=min(p['policy/exact_joint_kl'] for p in progress),
               mean_suffix_fraction=sum(p['data/complete_suffix_fraction'] for p in progress) / len(progress),
               mean_cg_residual=sum(p['natural/cg_residual_ratio'] for p in progress) / len(progress),
               median_cg_residual=statistics.median(p['natural/cg_residual_ratio'] for p in progress),
               max_cg_residual=max(p['natural/cg_residual_ratio'] for p in progress),
               final_metrics=last, matched_steps={})
for step in (1000000, 2000000, 4000000, 6000000, 8000000):
    if step > last['step']:
        continue
    match = min(progress, key=lambda p: abs(p['step'] - step))
    summary['matched_steps'][str(step)] = dict(step=match['step'], value=match[ret])
references = [
    ('v12 full batch, fixed target', RUNS / 'HalfCheetah-v4__latent_costate_v12_fullbatch_fixed_low_lr_8M__1__1789577645639026017/result.json'),
    ('v10 minibatches, eight refreshes', RUNS / 'HalfCheetah-v4__latent_costate_v10_no_projection_8M__1__1789544944088860507/result.json')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for label, path in references:
    reference = json.loads(path.read_text())['progress']
    axes[0].plot([p['step']/1e6 for p in reference], [p[ret] for p in reference], label=label, alpha=.7)
steps = [p['step']/1e6 for p in progress]
axes[0].plot(steps, [p[ret] for p in progress], label='v13 vector action response', linewidth=2)
axes[0].set_ylabel('Last-100 training episode return')
axes[0].legend(fontsize=7)
axes[1].plot(steps, [p['policy/exact_joint_kl'] for p in progress])
axes[1].axhline(.03, linestyle='--', color='gray')
axes[1].set_ylabel('Accepted actor KL')
axes[2].plot(steps, [p['losses/baseline_profile'] for p in progress], label='State-only profile')
axes[2].plot(steps, [p['losses/response_profile'] for p in progress], label='Response fitting')
axes[2].set_ylabel('Mean fitting loss / profile energy')
axes[2].legend(fontsize=8)
for ax in axes:
    ax.set_xlabel('Collected transitions (millions)')
    ax.grid(alpha=.2)
fig.suptitle('Vector return response v13 — fresh seed 1, HalfCheetah-v4')
output = ROOT / 'docs/vector-return-response-v13-results'
output.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
fig.savefig(output.with_suffix('.png'), dpi=160)
fig.savefig(output.with_suffix('.svg'))
print(json.dumps({k: v for k, v in summary.items() if k != 'final_metrics'}))
