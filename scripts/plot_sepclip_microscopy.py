"""Render measured sepclip checkpoint physics and layer geometry; run via mlq."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def label(row):
    run = Path(row['snapshot']).parent.name.split('__')[1]
    names = {'sepclip_mechanisms_rollout_clock': 'v7 Anvil .0025',
             'sepclip_mechanisms_rollout_clock_full': 'v7 Anvil .005',
             'sepclip_microscopy_v12_adam_control': 'v12 Adam',
             'sepclip_anvil2_v4_anvil_half': 'v4 Anvil .0025'}
    step = row.get('global_step', row.get('step'))
    return f'{names.get(run, run)} / {step / 1e6:.1f}M'


def save(fig, path, artifacts):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    artifacts.extend([str(path.with_suffix('.png')), str(path.with_suffix('.svg'))])


def physics_figures(document, directory, artifacts):
    rows = document['results']
    chosen = [{e['mode']: e for e in row['evaluations']
               if e['normalization'] == 'frozen' and e['row_offset'] == 0} for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5.3))
    for j, mode in enumerate(('sample', 'mean', 'mode')):
        ax.bar(x+(j-1)*.25, [v[mode]['return_summary']['mean'] for v in chosen], width=.25,
               yerr=[v[mode]['return_summary']['se'] for v in chosen], capsize=3, label=mode)
    ax.axhline(20000, color='#7c2530', linestyle='--', linewidth=1, label='20k ambition; no proven optimum')
    ax.set_xticks(x, [label(r) for r in rows], rotation=20, ha='right')
    ax.set_ylabel('Raw return, 1,000 steps')
    ax.set_title('Closed-loop evaluation: exact frozen per-environment normalization rows')
    ax.text(.01, .97, '16 paired reset seeds; error bars are episode SE, not across-training-seed uncertainty',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    save(fig, directory/'closed_loop_returns', artifacts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (before, after) in zip(axes, [('sample', 'mean'), ('mean', 'mode')]):
        velocity = np.array([1000*(v[after]['metrics']['velocity']['mean']-
                                  v[before]['metrics']['velocity']['mean']) for v in chosen])
        saved_cost = np.array([1000*(v[before]['metrics']['control_cost']['mean']-
                                    v[after]['metrics']['control_cost']['mean']) for v in chosen])
        ax.bar(x-.18, velocity, width=.36, label='Velocity contribution')
        ax.bar(x+.18, saved_cost, width=.36, label='Saved control cost')
        ax.axhline(0, color='#555555', linewidth=.7)
        ax.set_xticks(x, [label(r) for r in rows], rotation=40, ha='right', fontsize=8)
        ax.set_title(f'{before} → {after}')
    axes[0].set_ylabel('Contribution to raw return difference')
    axes[1].legend(fontsize=9)
    fig.suptitle('Most benefit comes from changed velocity, beyond the direct action-cost tax')
    save(fig, directory/'velocity_cost_decomposition', artifacts)


def layer_figures(document, directory, artifacts):
    for index, row in enumerate(document['snapshots']):
        labels = [v.replace('.0.', '.').replace('.weight', '').replace('readout_gain', 'gain')
                  for v in row['component_labels']]
        gram = np.asarray(row['fisher_gram'], dtype=float)
        joint = float(row['joint_fisher_quadratic'])
        relative = gram/max(abs(joint), 1e-30)
        maximum = max(abs(relative).max(), 1e-30)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.1]})
        image = axes[0].imshow(relative, cmap='RdBu_r', vmin=-maximum, vmax=maximum)
        axes[0].set_xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
        axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
        axes[0].set_title(f'Layer Fisher Gram / joint cost {joint:.3g}')
        fig.colorbar(image, ax=axes[0], shrink=.75)
        banks = row['banks']
        radial = np.array([b['proposal_radial_norm_fraction'] for b in banks])
        names = [b['parameter'].replace('.0.', '.').replace('.weight', '') for b in banks]
        positions = np.arange(len(banks))
        axes[1].bar(positions-.18, radial, width=.36, label='Removed norm fraction')
        axes[1].bar(positions+.18, radial**2, width=.36, label='Removed squared-norm fraction')
        axes[1].set_xticks(positions, names, rotation=50, ha='right', fontsize=8)
        axes[1].set_ylim(0, max(.35, radial.max()*1.15))
        axes[1].set_title('Ambient proposal removed by first-order tangent projection')
        axes[1].legend(fontsize=8)
        cross = row['cross_fisher_quadratic']/max(abs(joint), 1e-30)
        fig.suptitle(f'{label(row)} — cross-layer share {cross:.1%}; saved optimizer step, '
                     f'Fisher on {row["subset_count"]} states; no physical PG')
        save(fig, directory/f'layer_geometry_{index}', artifacts)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--physics', type=Path, required=True)
    cli.add_argument('--layers', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    directory = args.out.parent/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    artifacts = []
    physics_figures(json.loads(args.physics.read_text()), directory, artifacts)
    layer_figures(json.loads(args.layers.read_text()), directory, artifacts)
    args.out.write_text(json.dumps({'artifacts': artifacts, 'figure_count': len(artifacts)//2,
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.physics, args.layers)},
        'scope': 'Measured historical physics and layer diagnostics; no optimum or causal-training claim'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
