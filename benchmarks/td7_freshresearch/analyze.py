"""Extract matched-step evidence without extrapolating partial runs."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from _runs import RunScalars, find_runs

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PATTERNS = ['1791405636', '1791395613', 'td7_freshresearch_', 'td7_latentq_critic_',
            'td7_latentq_integral_only_v16', 'td7_latentq_lambda_only_v17',
            'td7_latentq_lambda_future_cv_v18', 'td7_latentq_lambda_snapshot_cv_v19', 'td7_latentq_lambda_refresh_cv_v20', 'td7_latentq_analytic_moments_v21', 'td7_latentq_paired_lambda_sale_v22', 'td7_latentq_paired_lambda_sale_v23', 'td7_latentq_paired_lambda_coverage_v24', 'td7_latentq_paired_lambda_center_v25', 'td7_latentq_paired_replay_coverage_v26']
PROBE_GROUPS = ('target_refresh', 'online_encoder_swap', 'fitting_burst/current', 'fitting_burst/archive')
PROBE_METRICS = ('center_rms_change', 'point_rms_change', 'label_rms_change',
                 'gradient_rms_change', 'gradient_cosine', 'gradient_norm_ratio',
                 'gradient_rms_before', 'gradient_rms_after', 'count')
TAGS = ['charts/episodic_return', 'eval/episodic_return', 'actor/kl',
        'gradient_alignment/cosine', 'gradient_alignment/path_norm',
        'gradient_alignment/score_norm', 'diagnostics/after/convolution_mae',
        'gradient_alignment/projected_scale', 'gradient_alignment/path_score_norm_ratio',
        'gradient_alignment/lane_projection_mean', 'gradient_alignment/projection_block_se',
        'gradient_alignment/score_split_half_cosine', 'gradient_alignment/projection_lcb95',
        'diagnostics/critic/relative_parameter_displacement',
        'diagnostics/after/fit/q', 'charts/max_target', 'actor/encoder_swap_kl',
        'actor/kl_mean_term', 'actor/kl_scale_term', 'charts/optimizer_steps',
        'charts/replay_size', 'losses/critic_loss', 'losses/hermite_r2',
        'losses/smoothing_gap', 'losses/preaction_gap', 'losses/q_values',
        'losses/tangent_std', 'charts/min_target', 'critic_targets/mean', 'critic_targets/std', 'charts/SPS']
TAGS += [f'critic_probe/{group}/{metric}' for group in PROBE_GROUPS for metric in PROBE_METRICS]
CV_METRICS = ('raw_residual_variance', 'clean_residual_variance', 'variance_ratio',
              'variance_reduction_fraction', 'residual_control_covariance')
TAGS += [f'critic_cv/{metric}' for metric in CV_METRICS]
TAGS += [f'critic_cv/position_{position:02d}/{metric}'
         for position in range(32) for metric in CV_METRICS]
TAGS += [f'critic_cv/{metric}' for metric in ('raw_target_std', 'clean_target_std',
         'control_shift_mean', 'control_total_mean', 'control_total_std',
         'innovation_order1_mean', 'innovation_order1_std', 'innovation_order2_mean',
         'innovation_order2_std', 'future_control_order1_mean', 'future_control_order1_std',
         'future_control_order2_mean', 'future_control_order2_std',
         'coefficient_order1', 'coefficient_order2', 'next_coefficient_order1',
         'next_coefficient_order2', 'calibration_step', 'candidate_control_std')]
TAGS += ['critic_refresh/count', 'critic_refresh/mean_increment_mse',
         'critic_refresh/final_raw_displacement_mse']
TAGS += [f'critic_refresh/final/{metric}' for metric in CV_METRICS]
TAGS += [f'critic_refresh/final/{metric}' for metric in ('initial_raw_mean', 'final_raw_mean', 'final_raw_std')]
TAGS += ['charts/sigma_mean', 'charts/sigma_min', 'charts/sigma_max']
TAGS += ['losses/point_r2', 'losses/point_residual_std', 'losses/distribution_gap', 'losses/importance_mean']
TAGS += [f'critic_heldout/{phase}/{name}/{stat}' for phase in ('prefit','postfit')
         for name in ('return_residual', *[f'mean_score_{i}' for i in range(6)], *[f'scale_score_{i}' for i in range(6)])
         for stat in ('mean','block_se')]
TAGS += [f'critic_heldout/{phase}/{name}' for phase in ('prefit','postfit')
         for name in ('point_mse','return_residual_std','rows','temporal_blocks')]
TAGS += ['charts/ordinary_transitions', 'charts/branch_transitions', 'charts/branch_useful_transitions']
TAGS += [f'paired/{name}' for name in ('anchor_count','fit_anchor_count','unique_initial_directions',
         'draws_per_rollout','fit_reuse_per_rollout','direct_channel_pair_weight')]
TAGS += [f'paired/{phase}/{split}/{label}/{metric}'
         for phase in ('prefit','postfit') for split in ('fit','holdout') for label in ('frozen','latest')
         for metric in ('odd_mse','odd_bias','odd_covariance','odd_correlation','odd_rms_ratio',
                        'even_mse','even_bias','preaction_contrast_mse','point_contrast_mse',
                        'label_contrast_std','reward_contrast_std','bootstrap_contrast_std','bootstrap_contrast_mean')]
TAGS += [f'paired/postfit/{split}/{metric}' for split in ('fit','holdout')
         for metric in ('center_displacement_rms','odd_displacement_rms','odd_prefit_postfit_cosine')]
TAGS += [f'branch_coupling/offset_{offset:02d}/{metric}' for offset in range(32)
         for metric in ('alive_pairs','state_rms','action_rms')]
TAGS += ['paired/direct_channel_branch_weight', 'charts/branch_fit_observations']
TAGS += [f'paired/postfit/{split}/clock_matrix/{channel}_{clock}_mse'
         for split in ('fit','holdout') for channel in ('odd','even')
         for clock in ('old_initial','old_final','new_initial','new_final')]
TAGS += [f'paired/postfit/{split}/{channel}_label_movement_rms'
         for split in ('fit','holdout') for channel in ('odd','even')]
TAGS += [f'branch_direct/{phase}/offset_{offset:02d}/{clock}/{channel}_mse'
         for phase in ('prefit','postfit') for offset in (0,8,16,31)
         for clock in ('initial','latest') for channel in ('point','preaction')]
TAGS += [f'branch_direct/{phase}/offset_{offset:02d}/{metric}'
         for phase in ('prefit','postfit') for offset in (0,8,16,31)
         for metric in ('valid_holdout_count','label_movement_rms')]
TAGS += [f'history/{name}' for name in ('fresh_direct','history_direct','fresh_hermite','history_hermite','fresh_sale','history_sale','history_point_age','history_pair_age','point_occupancy','pair_occupancy')]
TAGS += [f'actor_queries/{name}' for name in ('predicted_center_gain','behavior_center','current_center','behavior_saturation','current_saturation')]
TAGS += [f'paired/postfit/{split}/crossed_encoder/{channel}_{metric}'
         for split in ('fit','holdout') for channel in ('odd','even')
         for metric in ('oldcritic_newencoder_initial_mse','oldcritic_newencoder_final_mse','encoder_only_displacement_rms','critic_after_encoder_displacement_rms')]
rows = []
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for p in find_runs([ROOT / 'runs'], PATTERNS):
    r = RunScalars(p, tags=set(TAGS))
    row = {'run': p.name, 'variant': r.variant, 'metrics': {}}
    # Paired methods spend 4096 additional interactions per ordinary rollout.
    # Their final episode precedes the final branch batch; use the counted
    # interaction clock to recognize the available partial 8M tail.
    paired_method = r.variant.startswith(('td7_latentq_paired_lambda_', 'td7_latentq_paired_replay_'))
    rollout_budget = 20_480 if paired_method else 16_384
    counted_step = r.max_step('charts/SPS')
    for tag in TAGS:
        steps, vals = r.series(tag)
        if not len(vals):
            continue
        # Undefined gradient ratios/cosines are intentionally logged as NaN.
        # Preserve their count while excluding them from numerical summaries.
        invalid_samples = int((~np.isfinite(vals)).sum())
        finite = np.isfinite(vals)
        steps, vals = steps[finite], vals[finite]
        if not len(vals):
            row['metrics'][tag] = {'invalid_samples': invalid_samples, 'matched': {}}
            continue
        # Frozen legacy trainers log optimizer-path tags against optimizer steps.
        # Do not silently interpret those as environment transitions.
        optimizer_clock = (tag == 'charts/max_target' and r.variant in {
            'td7_freshresearch_cadence_v10', 'td7_latentq_fresh_sale_v9',
            'td7_latentq_onpolicy_v8_replay_sale',
        })
        points = {}
        for target in [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]:
            partial_budget_tail = target == 8_000_000 and counted_step >= target - rollout_budget
            if optimizer_clock or (steps[-1] < target and not partial_budget_tail):
                points[str(target)] = None
                continue
            mask = (steps <= target) & (steps >= target - 100_000)
            points[str(target)] = {'mean': float(vals[mask].mean()), 'samples': int(mask.sum()), 'partial': bool(steps[-1] < target)} if mask.any() else None
        row['metrics'][tag] = {
            'invalid_samples': invalid_samples,
            'last_step': int(steps[-1]), 'last_value': float(vals[-1]),
            'step_unit': 'optimizer_steps' if optimizer_clock else 'environment_transitions',
            'last20_mean': float(vals[-20:].mean()), 'matched': points,
            'after1M': {
                'samples': int((steps >= 1_000_000).sum()),
                'mean': float(vals[steps >= 1_000_000].mean()),
                'median': float(np.median(vals[steps >= 1_000_000])),
                'negative_fraction': float((vals[steps >= 1_000_000] < 0).mean()),
            } if not optimizer_clock and (steps >= 1_000_000).any() else None,
        }
        if tag == 'charts/episodic_return':
            # Average bins contain all lanes; overlapping episode trajectories are
            # not treated as independent statistical replications.
            bins = steps // 50_000
            unique = np.unique(bins)
            x = [steps[bins == b].mean() / 1e6 for b in unique]
            y = [vals[bins == b].mean() for b in unique]
            axes[0].plot(x, y, label=r.variant)
        if tag == 'actor/kl':
            axes[1].plot(steps / 1e6, vals, alpha=.7, label=r.variant)
    rows.append(row)
axes[0].set_ylabel('Training episode return (50k bins)')
axes[0].legend(fontsize=7, loc='best')
axes[1].set_ylabel('KL(behavior || updated policy)')
axes[1].set_yscale('log')
axes[1].set_xlabel('Environment transitions (millions)')
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'learning_curves.svg')
(OUT / 'evidence.json').write_text(json.dumps(rows, indent=2))
for row in rows:
    m = row['metrics'].get('charts/episodic_return', {})
    print(row['variant'], m.get('last_step'), round(m.get('last20_mean', float('nan')), 1))
