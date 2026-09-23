"""Summarize the fresh v14 run and only its established local references."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
paths = list(RUNS.glob('HalfCheetah-v4__vector_td_response_v14_8M__1__*/progress.json'))
if len(paths) != 1:
    raise RuntimeError(f'Expected one actual v14 run, got {len(paths)}')
run = paths[0].parent
result_path = run / 'result.json'
result = json.loads(result_path.read_text()) if result_path.exists() else {}
progress = result.get('progress') or json.loads(paths[0].read_text())
status = result.get('status', 'interrupted_without_final_result')
ret = 'charts/episodic_return_mean_100'
last = progress[-1]
peak = max(progress, key=lambda row: row[ret])
summary = dict(status=status, artifact=str(paths[0].relative_to(ROOT)), transitions=last['step'],
               last_logged_return=last[ret], peak_return=peak[ret], peak_step=peak['step'],
               actor_attempted=len(progress), actor_accepted=sum(p['policy/accepted_scale']>0 for p in progress),
               mean_kl=statistics.mean(p['policy/exact_joint_kl'] for p in progress),
               median_cg_residual=statistics.median(p['natural/cg_residual_ratio'] for p in progress),
               mean_cg_residual=statistics.mean(p['natural/cg_residual_ratio'] for p in progress),
               max_cg_residual=max(p['natural/cg_residual_ratio'] for p in progress),
               final_metrics=last, matched_steps={})
for step in (1000000,2000000,4000000,6000000,8000000):
    if step <= last['step']:
        match = min(progress,key=lambda row:abs(row['step']-step))
        summary['matched_steps'][str(step)] = dict(step=match['step'],value=match[ret])
references = [
    ('v12 full batch, fixed targets', RUNS/'HalfCheetah-v4__latent_costate_v12_fullbatch_fixed_low_lr_8M__1__1789577645639026017/result.json'),
    ('v13 Monte Carlo vector response', RUNS/'HalfCheetah-v4__vector_return_response_v13_8M__1__1789581691929940339/progress.json')]
fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for label,path in references:
    value = json.loads(path.read_text())
    rows = value['progress'] if isinstance(value,dict) else value
    axes[0].plot([p['step']/1e6 for p in rows],[p[ret] for p in rows],label=label,alpha=.7)
steps = [p['step']/1e6 for p in progress]
axes[0].plot(steps,[p[ret] for p in progress],label='v14 vector TD',linewidth=2)
axes[0].set_ylabel('Last-100 training episode return')
axes[0].legend(fontsize=7)
axes[1].plot(steps,[p['policy/exact_joint_kl'] for p in progress])
axes[1].axhline(.03,linestyle='--',color='gray')
axes[1].set_ylabel('Accepted actor KL')
for key,label in [('critic/immediate_residual_rms','One-step horizon'),
                  ('critic/remaining_residual_rms','Full remaining horizon')]:
    axes[2].plot(steps,[p[key] for p in progress],label=label)
axes[2].set_ylabel('Post-fit vector TD residual RMS')
axes[2].legend(fontsize=8)
for ax in axes:
    ax.set_xlabel('Collected transitions (millions)')
    ax.grid(alpha=.2)
fig.suptitle('Vector TD response v14 — fresh seed 1, HalfCheetah-v4')
output = ROOT/'docs/vector-td-response-v14-results'
output.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
fig.savefig(output.with_suffix('.png'),dpi=160)
fig.savefig(output.with_suffix('.svg'))
print(json.dumps({k:v for k,v in summary.items() if k != 'final_metrics'}))
