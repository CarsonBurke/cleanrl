"""Compact queue-aware evidence for PPO optimizer mechanism experiments."""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _runs import RETURN_TAG, RunScalars, find_runs, latest_per_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', nargs='+', type=int, required=True)
    parser.add_argument('--out', type=Path, default=Path('benchmarks/sepclip_mechanisms'))
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(subprocess.check_output(['mlq','show',str(job),'--json'], text=True)) for job in cli.jobs]
    lookup = {}
    for job in jobs:
        lookup[job['name']] = job
    tags = {RETURN_TAG, 'losses/post_final_epoch_exact_kl', 'losses/clipfrac', 'losses/entropy',
            'policy/concentration', 'policy/mean_step_rms', 'head/predicted_kl', 'head/target_kl',
            'head/sample_predicted_kl', 'head/exact_head_only_kl', 'head/sample_to_kronecker_ratio',
            'policy_budget/reference_quadratic','policy_budget/unscaled_quadratic',
            'policy_budget/scaled_quadratic','policy_budget/scale','policy_budget/accepted',
            'charts/interval_SPS', 'grad/actor_preclip_norm'}
    for bank in range(3):
        for metric in ('noise_to_signal', 'soft_polar_norm', 'effective_polar_norm', 'grad_momentum_cosine',
                       'epsilon_fraction', 'nesterov_norm'):
            tags.add(f'matrix_last/bank{bank}_{metric}')
    rows = []
    focused = []
    focus_labels = {
        'sepclip_mechanisms_adam_reference':'Adam',
        'sepclip_mechanisms_rollout_clock':'Fresh slow rail (.0025)',
        'sepclip_mechanisms_rollout_clock_full':'Fresh slow rail (.005)',
        'sepclip_mechanisms_policy_budget_cascade_control_v11':'Fresh cascade control (.0025)',
        'sepclip_mechanisms_policy_budget_anvil_v11':'Fractional + policy budget (.0025)',
    }
    fig, ax = plt.subplots(figsize=(13, 8))
    patterns = ['sepclip_mechanisms_', 'normuon_audit_8m_adam', 'sepclip_anvil2_v4_anvil_half']
    for run in latest_per_variant(find_runs([Path('runs')], patterns, 'HalfCheetah-v4')):
        scalars = RunScalars(run, tags=tags)
        steps, values = scalars.series(RETURN_TAG)
        if not values.size:
            continue
        job = lookup.get(scalars.variant)
        historical = scalars.variant in {'normuon_audit_8m_adam','sepclip_anvil2_v4_anvil_half'}
        complete = bool((historical or (job and job['state']=='succeeded')) and steps[-1]>=7900000)
        interval_metrics = {}
        for tag in tags-{RETURN_TAG}:
            metric_steps, metric_values = scalars.series(tag)
            if not metric_values.size:
                continue
            intervals = {}
            for lower, upper in ((0,100000),(100000,500000),(500000,1000000),(1000000,2000000),
                                 (2000000,4000000),(4000000,6000000),(6000000,7900000)):
                selected = metric_values[(metric_steps>lower)&(metric_steps<=upper)]
                if selected.size:
                    intervals[f'{lower}:{upper}'] = dict(mean=float(selected.mean()),
                        maximum=float(selected.max()), count=int(selected.size))
            interval_metrics[tag] = intervals
        row = dict(variant=scalars.variant, run=str(run), job=job['id'] if job else None,
                   state=job['state'] if job else 'verified_historical_control', completed_8m=complete,
                   last_step=int(steps[-1]), final_100_return=float(values[-100:].mean()) if complete else None,
                   latest_observed_100_return=float(values[-100:].mean()),
                   matched_returns={str(step):float(values[(steps>=step-200000)&(steps<=step)].mean())
                                    if steps[-1]>=step and ((steps>=step-200000)&(steps<=step)).any() else None
                                    for step in (1000000,2000000,4000000,6000000,7900000)},
                   matched_return_counts={str(step):int(((steps>=step-200000)&(steps<=step)).sum())
                                          if steps[-1]>=step else 0
                                          for step in (1000000,2000000,4000000,6000000,7900000)},
                   legacy_centered_returns={str(step):scalars.window_mean(RETURN_TAG,step,200000)
                                            if steps[-1]>=step else None
                                            for step in (1000000,2000000,4000000,6000000,7900000)},
                   latest_metrics={tag:scalars.latest(tag) for tag in tags-{RETURN_TAG}},
                   interval_metrics=interval_metrics)
        rows.append(row)
        if steps.size>=100:
            smooth = np.convolve(values,np.ones(100)/100,'valid')
            ax.plot(steps[99:]/1e6,smooth,
                    label=scalars.variant.replace('sepclip_mechanisms_',''),
                    linestyle='--' if historical else '-', linewidth=2 if historical else 1.5)
            if scalars.variant in focus_labels:
                focused.append((focus_labels[scalars.variant]+(' [partial]' if not complete else ''),
                                steps[99:]/1e6,smooth))
    ax.set(xlabel='Environment steps (millions)',ylabel='Return, trailing 100 episodes',
           title='PPO optimizer mechanisms — HalfCheetah-v4, seed 1')
    ax.grid(alpha=.2)
    if rows:
        ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(cli.out/'learning_curves.png',dpi=160)
    plt.close(fig)
    if focused:
        fig,axes = plt.subplots(1,2,figsize=(14,5))
        for label,plot_steps,smooth in focused:
            axes[0].plot(plot_steps,smooth,label=label,linewidth=1.7)
            late = plot_steps>=6
            if late.any():
                axes[1].plot(plot_steps[late],smooth[late],label=label,linewidth=1.7)
        for axis in axes:
            axis.set(xlabel='Environment steps (millions)',ylabel='Return, trailing 100 episodes')
            axis.grid(alpha=.2)
        axes[0].set_title('Fresh controls and selected mechanisms')
        axes[1].set(title='Late training',xlim=(6,8))
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
        fig.suptitle('HalfCheetah-v4, seed 1 — descriptive curves, not independent replication')
        fig.tight_layout(rect=(0,.12,1,.94))
        fig.savefig(cli.out/'comparison_curves.png',dpi=160)
        plt.close(fig)
    completed=sorted([row for row in rows if row['completed_8m']],key=lambda row:row['final_100_return'],reverse=True)
    result=dict(jobs=[{k:job.get(k) for k in ('id','name','state','maxParallelRuns','stateReason')} for job in jobs],
                runs=rows,ranked_completed=[row['variant'] for row in completed],
                metric_definitions={'matched_returns':'Inclusive trailing [step-200000, step], only if run reaches step.',
                    'legacy_centered_returns':'Legacy [step-200000, step+200000], truncated at actual run end; retained for audit, not primary comparison.',
                    'matrix_last/bankN_noise_to_signal':
                    'Propagated noise covariance trace divided by squared Nesterov norm: a power ratio.',
                    'diagnostic/noise_to_signal':'Fold-noise norm divided by raw full-gradient norm: a norm ratio.'},
                limitations=['One seed, one environment, selected candidates; no robustness claim.',
                             'Latest optimizer metrics are not interval averages.',
                             'Cancelled/failed/partial runs have no final-8M score.',
                             'Noise covariance assumes fixed linearized noise within rollout and independent rollout noise.',
                             'Fisher head uses a Kronecker approximation, not exact total actor Fisher.'])
    (cli.out/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in completed:
        print(f'{row["variant"]}: {row["final_100_return"]:.1f}')
    for job in jobs:
        print(f'job {job["id"]}: {job["state"]} (limit {job["maxParallelRuns"]})')


if __name__=='__main__':
    main()
