"""Plot accepted actual-update artifacts without loading models; queue via mlq.

Keep stored PPO ascent, clipped epoch loss gradients and physical PG separate.
JSON input byte pins and all zero/sign information are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Input byte pin differs: {path}')
    return json.loads(path.read_text())


def finite(values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite plotting input')
    return array


def short(name):
    return name.replace('actor.', '').replace('trunk.', '').replace('.weight', '')


def save(fig, stem, outputs):
    for suffix in ('.png', '.svg'):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        outputs.append({'path': str(path), 'sha256': digest(path)})
    plt.close(fig)


def plot_layers(label, doc, directory, outputs):
    names = doc['chords']['first']['layer_order']
    if names != doc['chords']['endpoint']['layer_order']:
        raise ValueError('Layer order differs between actual chords')
    traces = doc['epoch_trace']
    if len(traces) != 10:
        raise ValueError('Require complete actual ten-update trace')
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for ax, chord in zip(axes[0], ('first', 'endpoint')):
        row = doc['chords'][chord]
        gram = finite(row['full_joint_shape_fisher_gram'])
        joint = float(finite(row['joint_energy']))
        if joint <= 0 or gram.shape != (len(names), len(names)):
            raise ValueError('Invalid accepted joint Fisher geometry')
        relative = gram / joint
        scale = np.abs(relative).max()
        image = ax.imshow(relative, cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.set_xticks(range(len(names)), [short(n) for n in names], rotation=70, ha='right', fontsize=7)
        ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=7)
        kl = float(finite(row['finite_actual_old_to_bank_kl']))
        ax.set_title(f'{chord}: layer Gram / joint energy\nKL={kl:.4g}, half-Fisher={.5*joint:.4g}')
        fig.colorbar(image, ax=ax, shrink=.7)
    x = np.arange(len(names))
    for offset, chord in ((-.18, 'first'), (.18, 'endpoint')):
        credits = finite([doc['chords'][chord]['old_stored_clipped_ppo_ascent_credit_by_layer'][n] for n in names])
        axes[1, 0].bar(x+offset, credits, width=.36, label=chord)
    axes[1, 0].axhline(0, color='#555', lw=.7)
    axes[1, 0].set_xticks(x, [short(n) for n in names], rotation=70, ha='right', fontsize=7)
    axes[1, 0].set_title('Old stored clipped PPO ascent pairing with actual chord')
    axes[1, 0].set_ylabel('Signed surrogate pairing; not physical PG')
    axes[1, 0].legend()
    motion_values = [[t['layers'][n]['actual_relative_weight_motion'] for t in traces] for n in names]
    for name, row in zip(names, motion_values):
        for trace, value in zip(traces, row):
            weight_norm = float(finite(trace['layers'][name]['reconstructed_weight_norm']))
            if weight_norm < 0 or (value is None) != (weight_norm == 0):
                raise ValueError('Relative motion is undefined exactly at a zero preceding weight norm')
    finite([v for row in motion_values for v in row if v is not None])
    motion = np.array([[np.nan if v is None else v for v in row] for row in motion_values], dtype=float)
    if (motion[np.isfinite(motion)] < 0).any():
        raise ValueError('Negative relative norm')
    positive = motion[motion > 0]
    if positive.size:
        lower, upper = positive.min(), positive.max()
        if upper == lower:
            upper = np.nextafter(upper, np.inf)
        image = axes[1, 1].imshow(np.ma.masked_where((motion == 0) | np.isnan(motion), motion), aspect='auto',
                                norm=LogNorm(vmin=lower, vmax=upper), cmap='viridis')
        fig.colorbar(image, ax=axes[1, 1], shrink=.7)
    else:
        axes[1, 1].imshow(np.ma.masked_where((motion == 0) | np.isnan(motion), motion),
                          aspect='auto', cmap='Greys', vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(10), range(1, 11))
    axes[1, 1].set_yticks(range(len(names)), [short(n) for n in names], fontsize=7)
    axes[1, 1].set_xlabel('Actual update number')
    color_label = 'Log color' if positive.size else 'No positive motion'
    axes[1, 1].set_title('Actual delta norm / preceding weight norm\n'
                         f'{color_label}; {int((motion == 0).sum())} zero and {int(np.isnan(motion).sum())} undefined cells masked')
    fig.suptitle(f'{label}: actual captured update geometry', fontsize=15)
    fig.text(.01, .005, 'Full old batch; smooth shape JVP and FP64 Fisher. Cross terms retained. First bank reconstructed; endpoint independently saved.\n'
             'Stored surrogate credit and relative parameter motion do not certify physical ascent or optimizer efficiency.', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .97))
    save(fig, directory/(label+'_layers'), outputs)


def plot_moments(label, doc, directory, outputs):
    if doc.get('status') != 'accepted_actual_update_reconstruction':
        raise ValueError('Forensic reconstruction was not accepted')
    epochs = doc['epochs']
    if len(epochs) != 10:
        raise ValueError('Incomplete forensic trace')
    for row in epochs:
        if not all(c['bitwise_equal'] for c in row['ambient_parity'].values()):
            raise ValueError('Recorded ambient parity missing')
        if not all(c['bitwise_equal'] for c in row['actual_parity'].values()):
            raise ValueError('Recorded retracted parity missing')
    if not all(c['bitwise_equal'] for c in doc['endpoint_parity'].values()):
        raise ValueError('Exact endpoint parity missing')
    if not epochs[0]['matrix_banks']:
        return  # Adam has no matrix rails; actor actual-motion plot covers it.
    names = [name for bank in epochs[0]['matrix_banks'] for name in bank['members']]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, stage in zip(axes, ('gradient_blend', 'gradient_lookahead', 'gradient_cascade')):
        by_epoch = [{name: member for bank in row['matrix_banks'] for name, member in bank['members'].items()} for row in epochs]
        values = []
        for name in names:
            row = []
            for epoch in range(10):
                alignment = by_epoch[epoch][name][stage]
                value = alignment['cosine']
                if value is None:
                    norms = finite([alignment['left_norm'], alignment['right_norm']])
                    if not (norms == 0).any() or (norms < 0).any():
                        raise ValueError('Undefined cosine without a zero vector')
                elif not np.isfinite(value) or abs(value) > 1+2e-12:
                    raise ValueError('Nonfinite or invalid optimizer cosine')
                row.append(value)
            values.append(row)
        # Undefined zero-vector cosine is masked, never converted to agreement.
        array = np.ma.masked_invalid(np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float))
        # A <=2e-12 FP64 norm/dot overshoot is allowed; color saturates at1.
        image = ax.imshow(array, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(10), range(1, 11))
        ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=8)
        ax.set_xlabel('Actual update number')
        ax.set_title(stage.replace('gradient_', 'clipped loss G vs '))
    fig.colorbar(image, ax=axes.tolist(), shrink=.8, label='Cosine; matrix-space, not functional or physical alignment')
    fig.suptitle(f'{label}: source-reconstructed optimizer stages', fontsize=14)
    fig.text(.01, .015, 'Every actual delta and endpoint matched bitwise. Shadow internal stages have separate bounded checks.\n'
             'G is the clipped LOSS covector; stages are reconstructed, not independently recorded. No benchmark cause or physical-PG claim.', fontsize=9)
    fig.subplots_adjust(left=.14, right=.86, bottom=.19, top=.82, wspace=.15)
    save(fig, directory/(label+'_moments'), outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', nargs=3, action='append', default=[], metavar=('LABEL', 'JSON', 'SHA256'))
    parser.add_argument('--optimizer', nargs=3, action='append', default=[], metavar=('LABEL', 'JSON', 'SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.layers:
        parser.error('Require accepted actual layer artifacts')
    for rows in (args.layers, args.optimizer):
        labels = [row[0] for row in rows]
        if len(labels) != len(set(labels)):
            raise ValueError('Duplicate figure labels within input kind')
    directory = args.output.parent/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False})
    outputs, inputs = [], {}
    for kind, rows in (('layers', args.layers), ('optimizer', args.optimizer)):
        for label, raw_path, pin in rows:
            if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
                raise ValueError('Unsafe figure label')
            path = Path(raw_path)
            document = read_pinned(path, pin)
            inputs[str(path)] = pin
            (plot_layers if kind == 'layers' else plot_moments)(label, document, directory, outputs)
            if digest(path) != pin:
                raise ValueError('Input changed while plotting')
    args.output.write_text(json.dumps(dict(input_sha256=inputs,artifacts=outputs,
        source_sha256=digest(Path(__file__)),scope='Visualization of accepted actual-source diagnostics only; no model loading, new measurements, physical oracle, significance selection or causal training claim.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
