"""Compiled full captured actor chord JVP versus independent centered primals.

CUDA/MLQ only. Neighboring ambient chords are never sphere-normalized. This
checks local numerical derivatives on saved observations, not physical PG,
finite return, efficiency or global architecture capacity.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sepclip_prescription_replay import MODULES, load_replay, sha256, require_pin, validate_provenance


def _finite(value):
    return bool(torch.isfinite(value).all())


def _sum_squared(values):
    return sum(float(value.double().square().sum()) for value in values)


def response_metrics(derivative, finite):
    """Keep null directions and FP32 floor responses explicit, without floors."""
    left = derivative.double(); right = finite.double()
    a = float(left.norm()); b = float(right.norm()); error = float((left - right).norm())
    return dict(jvp_norm=a, centered_primal_norm=b, absolute_l2_error=error,
        absolute_max_error=float((left - right).abs().max()),
        relative_l2_error=error / a if a > 0 else None,
        cosine=float((left.flatten() @ right.flatten()) / (a * b)) if a > 0 and b > 0 else None,
        null_jvp=a == 0, zero_rounded_primal_response=b == 0,
        scope='Smooth compiled AD chain versus rounded FP32 centered primal; shape-floor plateaus can disagree with a nonzero smooth-chain derivative.')


def output_storage_floor(plus, minus, h, derivative):
    """ULP telemetry only, not a bound on accumulated compiled arithmetic error."""
    def spacing(value):
        zero = torch.zeros_like(value)
        toward_zero = (value - torch.nextafter(value, zero)).double().abs()
        at_zero = torch.nextafter(zero, torch.ones_like(value)).double().abs()
        return torch.where(value == 0, at_zero, toward_zero)
    heuristic = (spacing(plus) + spacing(minus)) / (2 * h)
    norm = float(heuristic.norm()); reference = float(derivative.double().norm())
    return dict(toward_zero_ulp_sum_over_2h_l2=norm,
        toward_zero_ulp_sum_over_2h_max=float(heuristic.max()),
        relative_to_jvp_norm=norm / reference if reference > 0 else None,
        scope='FP32 output-storage spacing heuristic only; toward-zero spacing varies across binades and excludes accumulated compiled-primal roundoff. Not a rigorous error bound or derivative certificate.')


def audit_chord(actor, old, direction, observations, hs):
    """Reusable numerical kernel; actual-capture provenance belongs to analyze."""
    names = dict(actor.named_parameters())
    if set(old) != set(names) or set(direction) != set(names) or tuple(actor.named_buffers()):
        raise ValueError('Require complete explicit actor parameter banks and no implicit buffers')
    if not hs or len(hs) < 3 or any(not isinstance(h, (int, float)) or isinstance(h, bool) or not np.isfinite(h) or h <= 0 for h in hs):
        raise ValueError('Need at least three finite positive neighboring step scales')
    if len(set(hs)) != len(hs) or any(a <= b for a, b in zip(hs, hs[1:])):
        raise ValueError('Step scales must be distinct and strictly decreasing')
    if any(h < torch.finfo(torch.float32).tiny or h > torch.finfo(torch.float32).max for h in hs):
        raise ValueError('Step scales must be in the finite normal FP32 scalar range')
    for name in names:
        for bank in (old, direction):
            value = bank[name]
            if value.shape != names[name].shape or value.dtype != torch.float32 or value.device.type != 'cuda' or not _finite(value):
                raise ValueError('Invalid actual actor parameter/chord array')
    if observations.ndim != 2 or observations.dtype != torch.float32 or observations.device.type != 'cuda' or not _finite(observations):
        raise ValueError('Require finite recorded FP32 CUDA observations')
    def primal(parameters, obs):
        logits = torch.func.functional_call(actor, parameters, (obs,), strict=True)
        alpha, beta = (1 + F.softplus(logits)).chunk(2, -1)
        return logits, alpha, beta
    def differentiated(parameters, tangent, obs):
        return torch.func.jvp(lambda p: primal(p, obs), (parameters,), (tangent,))
    compile_options = dict(fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    compiled_primal = torch.compile(primal, **compile_options)
    compiled_jvp = torch.compile(differentiated, **compile_options)
    with torch.no_grad():
        reference = tuple(value.clone() for value in compiled_primal(old, observations))
        jvp_primal, jvp = compiled_jvp(old, direction, observations)
        jvp_primal = tuple(value.clone() for value in jvp_primal)
        jvp = tuple(value.clone() for value in jvp)
    if any(not _finite(value) for value in (*reference, *jvp_primal, *jvp)):
        raise ValueError('Nonfinite independently compiled primal/JVP output')
    if any(value.dtype != torch.float32 for value in (*reference, *jvp_primal, *jvp)):
        raise ValueError('Require actual FP32 primal and AD outputs for storage-floor telemetry')
    if any(not torch.allclose(a, b, rtol=2e-5, atol=3e-6) for a, b in zip(reference, jvp_primal)):
        raise ValueError('Independent primal and JVP graph primal disagree')
    labels = ('latent_logits', 'alpha', 'beta')
    requested_squared = _sum_squared(direction.values())
    requested_norm = requested_squared ** .5
    states = observations.shape[0]
    parameter_count = sum(value.numel() for value in old.values())
    nonzero_requested = sum(int((value != 0).sum()) for value in direction.values())
    rows = []; previous = None
    for h in hs:
        plus = {name: old[name] + h * direction[name] for name in names}
        minus = {name: old[name] - h * direction[name] for name in names}
        if any(not _finite(value) for value in (*plus.values(), *minus.values())):
            rows.append(dict(h=h, status='withheld_nonfinite_neighbor_parameters'))
            previous = None
            continue
        effective = {name: (plus[name].double() - minus[name].double()) / (2 * h) for name in names}
        effective_norm = _sum_squared(effective.values()) ** .5
        displacement_error = _sum_squared(effective[name] - direction[name].double() for name in names) ** .5
        center_error = _sum_squared((plus[name].double() + minus[name].double()) / 2 - old[name].double() for name in names) ** .5
        vanished = sum(int(((direction[name] != 0) & (plus[name] == minus[name])).sum()) for name in names)
        with torch.no_grad():
            positive = tuple(value.clone() for value in compiled_primal(plus, observations))
            negative = tuple(value.clone() for value in compiled_primal(minus, observations))
        if any(not _finite(value) for value in (*positive, *negative)):
            rows.append(dict(h=h, status='withheld_nonfinite_neighbor_outputs'))
            previous = None
            continue
        finite = tuple((p.double() - m.double()) / (2 * h) for p, m in zip(positive, negative))
        row = dict(h=h, status='measured' if effective_norm > 0 else 'withheld_unrepresentable_parameter_chord',
            effective_centered_parameter_direction_norm=effective_norm,
            effective_vs_requested_direction_relative_error=displacement_error / requested_norm if requested_norm > 0 else None,
            centered_parameter_basepoint_l2_error=center_error,
            vanished_nonzero_parameter_elements=vanished,
            vanished_nonzero_parameter_fraction=vanished / nonzero_requested if nonzero_requested else None,
            response_comparisons={label: response_metrics(d, f) for label, d, f in zip(labels, jvp, finite)},
            output_storage_rounding_floors={label: output_storage_floor(p, m, h, d)
                for label, p, m, d in zip(labels, positive, negative, jvp)},
            rounded_shape_floor_counts={label: dict(old=int((r == 1).sum()), plus=int((p == 1).sum()),
                minus=int((m == 1).sum()), both_neighbors_floor=int(((p == 1) & (m == 1)).sum()))
                for label, r, p, m in zip(labels[1:], reference[1:], positive[1:], negative[1:])})
        if effective_norm == 0:
            row['comparison_scope'] = 'No nonzero requested chord was represented by neighboring FP32 parameter banks; response metrics cannot validate this chord derivative.'
        if previous is not None:
            previous_h, previous_finite, previous_row = previous
            row['neighbor_comparison'] = dict(coarser_h=previous_h, h_ratio=h / previous_h,
                both_neighbor_chords_representable=effective_norm > 0 and previous_row['effective_centered_parameter_direction_norm'] > 0,
                current_vanished_nonzero_parameter_fraction=row['vanished_nonzero_parameter_fraction'],
                coarser_vanished_nonzero_parameter_fraction=previous_row['vanished_nonzero_parameter_fraction'],
                responses={label: dict(centered_response_difference_norm=float((fine - coarse).norm()),
                    centered_response_difference_relative_to_jvp_norm=(float((fine - coarse).norm()) / float(d.double().norm()))
                        if float(d.double().norm()) > 0 else None,
                    error_ratio_to_coarser=(row['response_comparisons'][label]['absolute_l2_error'] /
                        previous_row['response_comparisons'][label]['absolute_l2_error'])
                        if previous_row['response_comparisons'][label]['absolute_l2_error'] > 0 else None)
                    for label, fine, coarse, d in zip(labels, finite, previous_finite, jvp)})
        rows.append(row)
        previous = (h, finite, row) if effective_norm > 0 else None
    return dict(status='numerical_measurement_only', requested_ambient_chord_norm=requested_norm,
        parameter_count=parameter_count, nonzero_requested_parameter_elements=nonzero_requested,
        saved_observation_count=states, jvp_norms={label: float(value.double().norm()) for label, value in zip(labels, jvp)},
        independent_graph_primal_max_abs_errors={label: float((a - b).abs().max()) for label, a, b in zip(labels, reference, jvp_primal)},
        neighboring_steps=rows, zero_requested_chord=requested_norm == 0,
        scope='Ambient actor parameter chord at old weights; no constraint retraction on neighbors. AD shape chain is a smooth-score convention, not derivative of FP32 rounding.',
        limitations=['All requested neighboring h values are retained; no post-hoc best-h selection or automated derivative certification.',
            'Relative error and cosine are undefined for null reference/finite directions; no numerical denominator floor invents a sign.',
            'Agreement concerns saved observations and this chord, not future states, physical PG, finite return or optimizer efficacy.',
            'Finite nonlinear chord error and FP32 rounding can compete as h decreases; shape-floor disagreement does not establish hidden-layer starvation.'])


def analyze(snapshot_path, trace_path, hs, **kwargs):
    sources = {str(Path(__file__).resolve()): sha256(__file__),
        str(Path(__file__).with_name('sepclip_prescription_replay.py').resolve()): sha256(Path(__file__).with_name('sepclip_prescription_replay.py'))}
    replay = load_replay(snapshot_path, trace_path, **kwargs)
    old = replay.banks['old']; observations = replay.snapshot['batch']['obs']
    results = {}
    for label, direction in zip(('first_actual_retracted_chord', 'saved_endpoint_minus_old_chord'), replay.deltas):
        results[label] = audit_chord(replay.agent.actor, old, direction, observations, hs)
    for path, pin in ((snapshot_path, kwargs['snapshot_sha256']), (trace_path, kwargs['trace_sha256']),
                      (kwargs['manifest_path'], kwargs['manifest_sha256']), (kwargs['execution_path'], kwargs['execution_sha256'])):
        require_pin(path, pin)
    for path, pin in sources.items(): require_pin(path, pin)
    validate_provenance(kwargs['module_name'], kwargs['manifest_path'], kwargs['manifest_sha256'],
                        kwargs['execution_path'], kwargs['execution_sha256'], kwargs['exp_name'])
    return dict(replay_validation=replay.validation, full_saved_batch=True, independently_recorded_logits=False,
        artifact_kind='Actual captured first/end all-actor-parameter chords; derivative numerical audit only',
        chord_arithmetic='First chord uses saved FP32 retracted delta; endpoint chord subtracts exact saved endpoint and old FP32 states in FP32. Neighboring parameters are rounded FP32 ambient additions, not retracted policies.',
        chord_audits=results, source_sha256=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    for name in ('snapshot-sha256', 'trace-sha256', 'manifest-sha256', 'execution-sha256', 'exp-name'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--module-name', choices=tuple(MODULES), required=True)
    parser.add_argument('--manifest-path', type=Path, required=True)
    parser.add_argument('--execution-path', type=Path, required=True)
    parser.add_argument('--hs', nargs='+', type=float, default=[1., .5, .25, .125, .0625])
    parser.add_argument('--output', type=Path, required=True)
    cli = vars(parser.parse_args())
    snapshot = cli.pop('snapshot'); trace = cli.pop('trace'); hs = cli.pop('hs'); output = cli.pop('output')
    result = analyze(snapshot, trace, hs, **cli)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
