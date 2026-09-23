"""Report the stopped long run at the user-corrected 50M budget and actual cutoff."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
paths = list(RUNS.glob('HalfCheetah-v4__latent_costate_v12_fullbatch_fixed_low_lr_80M__1__*/progress.json'))
if len(paths) != 1:
    raise RuntimeError(f'Expected one stopped long run, found {len(paths)}')
path = paths[0]
progress = json.loads(path.read_text())
result = json.loads((path.parent / 'experiment_outcome.json').read_text())
assert result['status'] == 'cancelled'
short_path = RUNS / 'HalfCheetah-v4__latent_costate_v12_fullbatch_fixed_low_lr_8M__1__1789577645639026017/result.json'
short = json.loads(short_path.read_text())
args = result['args']
assert result['fresh_initialization'] and not result['checkpoint_loaded']
assert args['total_timesteps'] == 80000000 and args['seed'] == 1
assert args['full_batch'] and args['critic_target_refreshes'] == 1
assert args['num_envs'] * args['num_steps'] == 32768
assert args['model_learning_rate'] == .001 and args['critic_learning_rate'] == .0003
assert args['gamma'] == 1 and args['trust_kl'] == .03
ret = 'charts/episodic_return_mean_100'
peak = max(progress, key=lambda p: p[ret])
last = progress[-1]
matched = {}
for millions in (1, 2, 4, 8, 16, 24, 32, 40, 48, 50, 56):
    target = millions * 1000000
    row = min(progress, key=lambda p: abs(p['step'] - target))
    matched[str(target)] = {k: row[k] for k in ('step', ret, 'credit/next_costate_rms', 'policy/entropy', 'policy/concentration', 'policy/exact_joint_kl')}
summary = dict(job=7549, artifact=str(path.relative_to(ROOT)), args=args,
               status='cancelled', submitted_budget=80000000, corrected_budget=50000000,
               transitions=last['step'], final_return=None, last_logged_return=last[ret],
               peak_return=peak[ret], peak_step=peak['step'],
               actor_attempted=len(progress), actor_accepted=sum(p['policy/accepted_scale'] > 0 for p in progress),
               model_optimizer_steps=int(last['updates/model_optimizer']), critic_optimizer_steps=int(last['updates/critic_optimizer']),
               critic_target_refreshes=int(last['updates/critic_target_refreshes']),
               mean_kl=sum(p['policy/exact_joint_kl'] for p in progress) / len(progress),
               min_kl=min(p['policy/exact_joint_kl'] for p in progress),
               max_kl=max(p['policy/exact_joint_kl'] for p in progress),
               final_metrics=last,
               identical_8m_return_prefix=all(a[ret] == b[ret] for a, b in zip(progress, short['progress'])),
               first_8m_endpoint=progress[len(short['progress']) - 1][ret],
               matched_steps=matched)
for window in (1000000, 5000000, 10000000):
    rows = [p for p in progress if p['step'] >= last['step'] - window]
    summary[f'last_{window}_mean_logged_return'] = sum(p[ret] for p in rows) / len(rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
steps = [p['step'] / 1e6 for p in progress]
for ax, key, label in zip(axes.flat,
                          (ret, 'credit/next_costate_rms', 'policy/entropy', 'policy/concentration'),
                          ('Last-100 training episode return', 'Next-state costate RMS', 'Native Beta joint differential entropy', 'Mean Beta concentration')):
    ax.plot(steps, [p[key] for p in progress], linewidth=1.2)
    ax.axvline(8.04416, color='gray', linestyle=':', linewidth=1, label='Previous run endpoint')
    ax.axvline(50., color='tab:red', linestyle=':', linewidth=1, label='Corrected 50M budget')
    ax.set_xlabel('Collected transitions (millions)')
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
axes[0, 0].axhline(5831.04849609375, color='tab:green', linestyle='--', linewidth=1,
                    label='v10 minibatch/sweeps: 8M endpoint')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Fresh long run stopped at 56.61M; corrected budget 50M — HalfCheetah-v4, seed 1')
output = ROOT / 'docs/latent-costate-fullbatch-v12-long-run-results'
output.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
fig.savefig(output.with_suffix('.png'), dpi=160)
fig.savefig(output.with_suffix('.svg'))
print(json.dumps({k: v for k, v in summary.items() if k not in ('args', 'final_metrics', 'matched_steps')}))
print(json.dumps(matched))
