"""Read-only CPU statistics over saved oracle labels; no model or simulation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import reppo_oracle as oracle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('oracle_8M_tail_sensitivity.json'))
args = parser.parse_args()
source = args.dataset or next((ROOT / 'runs/reppo_latentroll_oracle').glob('*h8_reward*8011392.pt'))
data = torch.load(source, map_location='cpu', weights_only=False)
states, meta = data['states'], data['meta']
age = states['age'].numpy()
rows = states['row'].numpy()
subsets = oracle.subsets(data)
report = {'source': str(source), 'cells': [], 'advantage_cells': [], 'metric_definitions': {
    'cos_pooled_corrected': 'Legacy unit-direction gain: mean(unit(c) dot g) / sqrt(mean noise-corrected ||g||²). Perfect per-state alignment need not reach 1 when oracle gradient norms vary.',
    'globally_pooled_cosine_corrected': 'Ordinary globally pooled cosine: mean(c dot g) / sqrt(mean ||c||² * mean noise-corrected ||g||²). Includes actual critic magnitudes; high-energy states carry greater weight.',
    'advantage_corr_corrected': 'Within-state-centered action-value covariance divided by critic SD and noise-corrected oracle SD; pooled over states and sampled fresh actions.',
}, 'limitations': [
    'No-intermediate-tail means age + horizon <= 1000; every label still has a critic bootstrap at horizon.',
    'Cruising uses the original longest-horizon fresh-action value definition.',
    'State bootstrap ignores serial dependence; trajectory bootstrap resamples environment rows.',
    'Trimming selects on noisy measured gradient energy and is sensitivity analysis, not an unbiased estimator.',
    'Noise-corrected estimates can have nonpositive bootstrap denominators; valid fractions are reported.',
    'Noise correction is a finite-sample ratio estimate; values and confidence limits may exceed [-1, 1] and are not clipped.',
    'All confidence intervals concern this frozen policy and sampled state/action distribution, not training-seed uncertainty.',
]}


def clustered_ratio(numerator, energy_left, energy_right, cluster):
    """Ratio of summed moments with environment-row bootstrap and paired draws."""
    groups = np.unique(cluster)
    sums = np.array([[numerator[cluster == k].sum(), energy_left[cluster == k].sum(),
                      energy_right[cluster == k].sum()] for k in groups])
    draws = np.random.default_rng(1).integers(len(groups), size=(2000, len(groups)))
    totals = sums[draws].sum(1)
    good = (totals[:, 1] > 0) & (totals[:, 2] > 0)
    values = totals[good, 0] / np.sqrt(totals[good, 1] * totals[good, 2])
    denom = energy_left.sum() * energy_right.sum()
    return {'estimate': float(numerator.sum() / np.sqrt(denom)) if energy_left.sum() > 0 and energy_right.sum() > 0 else float('nan'),
            'trajectory_ci95': np.quantile(values, [.025, .975]).tolist() if good.any() else [float('nan')] * 2,
            'trajectory_bootstrap_valid_fraction': float(good.mean()), 'trajectory_clusters': len(groups)}

for horizon in (100, 200, 300):
    q = data['returns'][str(horizon)].double().numpy()
    for subset in ('all', 'cruising'):
        for restriction in ('original', 'no_intermediate_tail'):
            mask = subsets[subset].copy()
            if restriction == 'no_intermediate_tail':
                mask &= age + horizon <= 1000
            cluster = rows[mask]
            fresh_start = meta['action_layout']['fresh'][0]
            fresh = q[mask, fresh_start:]
            centered = fresh - fresh.mean(1, keepdims=True)
            truth = centered.mean(-1)
            predicted = states['critic_q'].double().numpy()[mask, fresh_start:]
            predicted = predicted - predicted.mean(1, keepdims=True)
            divisor = fresh.shape[1] - 1
            advantage = clustered_ratio((predicted * truth).sum(1) / divisor,
                (predicted ** 2).sum(1) / divisor,
                (truth ** 2).sum(1) / divisor - centered.var(-1, ddof=1).sum(1) / (fresh.shape[-1] * divisor), cluster)
            report['advantage_cells'].append({'horizon': horizon, 'subset': subset, 'restriction': restriction,
                'states': int(mask.sum()), 'advantage_corr_corrected': advantage,
                **oracle.advantage_stats(fresh, states['critic_q'].double().numpy()[mask, fresh_start:])})
            for location in ('star', 'a0'):
                offset, stop = meta['action_layout'][f'{location}_probes']
                span = states[f'span_{location}'].double().numpy()[0, mask]
                labels = oracle.finite_difference(q[mask, offset:stop], span, 0)
                critic = states[f'critic_grad_{location}'].double().numpy()[mask]
                critic_fd = oracle.finite_difference(states['critic_q'].double().numpy()[mask, offset:stop, None], span, 0)[..., 0]
                stats = oracle.gradient_stats(labels, critic, critic_fd, np.random.default_rng(1), boots=1000)
                g = labels.mean(-1)
                signal = (g*g).sum(-1) - labels.var(-1, ddof=1).sum(-1)/labels.shape[-1]
                dot = (oracle.unit(critic)*g).sum(-1)
                legacy_ratio = clustered_ratio(dot, np.ones(len(dot)), signal, cluster)
                stats.update({key: value for key, value in legacy_ratio.items() if key != 'estimate'})
                stats['globally_pooled_cosine_corrected'] = clustered_ratio(
                    (critic * g).sum(-1), (critic ** 2).sum(-1), signal, cluster)
                order = np.argsort((g*g).sum(-1))
                stats['trim_raw_energy_top5pct'] = oracle.gradient_stats(labels[order[:int(.95*len(order))]], critic[order[:int(.95*len(order))]], critic_fd[order[:int(.95*len(order))]], np.random.default_rng(1), boots=500)
                report['cells'].append({'horizon':horizon,'subset':subset,'restriction':restriction,'location':location,'states':int(mask.sum()),**stats})

out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
for c in report['cells']:
    print(c['horizon'], c['subset'], c['restriction'], c['location'], 'n', c['states'],
          'cos', round(c['cos_pooled_corrected'],3), 'SE', round(c['cos_pooled_corrected_se'],3),
          'clusterCI', [round(x,3) for x in c['trajectory_ci95']],
          'globalcos', round(c['globally_pooled_cosine_corrected']['estimate'],3),
          'top5energy', round(c['energy_share_top5'],2),
          'trimcos', round(c['trim_raw_energy_top5pct']['cos_pooled_corrected'],3))
