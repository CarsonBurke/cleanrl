"""Host provenance/barrier contracts and actual CUDA paired StockRunner integration.

CUDA tests must be queued via mlq; this file never substitutes CPU learner execution.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import predictive_available_stock_eval_v5 as available
from cleanrl.plasticity.predictive_available_state_v8 import AvailableStateComparison, PairedAdamBank
from cleanrl.shared import runtime
from cleanrl.shared.autocull import ProxyCull


@pytest.fixture(scope='module')
def prefix_evidence():
    end = available.SELECTION_END
    names = available.output_names()
    target = np.ones(end + 1, dtype=np.float32)
    prediction = np.full((end + 1, len(names)), .25, dtype=np.float32)
    for number, family in enumerate(available.ADAM_FAMILIES):
        columns = available.candidate_columns(names, family)
        prediction[available.COLD_START:end, columns[number + 1]] = .75
        prediction[:available.COLD_START, columns[number + 1]] = 1000
        # A giant first-suffix error would reverse the intended winner if leaked.
        prediction[end, columns[number + 1]] = 1e6
    for frame in ('original', 'latest'):
        prediction[:, names.index(f'{frame}_zero')] = 0
    locks = available.select_prefix_locks(prediction[:end], target[:end], names, end)
    samples = end + 20000
    fingerprint = dict(zip(available.DATA_KEYS, (c * 64 for c in 'abcd')))
    reference = {'artifact_sha256': 'e' * 64, 'samples': samples,
                 'data_sha256': {'features': fingerprint['original_features'], 'target': fingerprint['target']},
                 'source_sha256': {'stock_stream.py': 'f' * 64},
                 'processed_observations': end,
                 'original_features_bitwise_verified': True, 'target_bitwise_verified': True}
    evidence = {'args': {'view': 'real', 'seed': 1, 'selection_end': end},
                'data_sha256': fingerprint, 'source_sha256': {'model.py': 'f' * 64},
                'reference': reference, 'view_target_sha256': fingerprint['target'],
                'raw_zero_sha256': 'a' * 64,
                'samples': samples, 'maximum_observations': samples,
                'processed': samples, 'processed_observations': samples, 'status': 'completed',
                'phases': {key: list(bounds) for key, bounds in available.phase_ranges(samples).items()},
                'adam_grid': list(available.ADAM_GRID), 'output_names': list(names),
                'adam_locks': locks, 'protocol': {'primary_output': available.PRIMARY}}
    return prediction, target, names, evidence


def load_written(tmp_path, evidence, **overrides):
    path = tmp_path / 'real.json'
    path.write_text(json.dumps(evidence), encoding='utf-8')
    return available.load_real_locks(path, overrides.get('fingerprint', evidence['data_sha256']),
                                     overrides.get('source_sha256', evidence['source_sha256']),
                                     overrides.get('reference', evidence['reference']),
                                     overrides.get('samples', evidence['samples']),
                                     overrides.get('raw_zero_sha256', evidence['raw_zero_sha256']))


@pytest.mark.parametrize('family', available.ADAM_FAMILIES)
def test_four_locks_exclude_cold_start_and_first_suffix_label(prefix_evidence, family):
    prediction, target, names, evidence = prefix_evidence
    end = available.SELECTION_END
    lock = evidence['adam_locks'][family]
    assert lock['selected_index'] == available.ADAM_FAMILIES.index(family) + 1
    assert lock['selection_start_inclusive'] == available.COLD_START
    assert lock['selection_end_exclusive'] == lock['optimizer_updates_at_lock'] == end
    assert lock['suffix_observations_used'] == 0
    assert lock['selected_lr'] == available.ADAM_GRID[lock['selected_index']]
    assert all(row['count'] == end - available.COLD_START for row in lock['candidates'])
    for consumed, barrier in ((end - 1, end), (end + 1, end), (end, end - 1)):
        with pytest.raises(ValueError, match='exact 335872 barrier'):
            available.select_prefix_locks(prediction[:consumed], target[:consumed], names, consumed, barrier)


@pytest.mark.parametrize('status,extra', [('running', 0), ('pruned', 8192), ('completed', 20000)])
def test_null_inherits_every_real_lock_without_selection(tmp_path, prefix_evidence, status, extra):
    evidence = copy.deepcopy(prefix_evidence[3])
    evidence.update(status=status, processed=available.SELECTION_END + extra,
                    processed_observations=available.SELECTION_END + extra)
    assert load_written(tmp_path, evidence) == evidence['adam_locks']


@pytest.mark.parametrize('family', available.ADAM_FAMILIES)
@pytest.mark.parametrize('mutation', ['winner', 'late', 'suffix', 'missing', 'nonfinite'])
def test_each_adam_family_rejects_fabricated_or_noncausal_lock(tmp_path, prefix_evidence, family, mutation):
    evidence = copy.deepcopy(prefix_evidence[3])
    lock = evidence['adam_locks'][family]
    if mutation == 'winner':
        lock['selected_index'] = 0
        lock['selected_lr'] = available.ADAM_GRID[0]
    elif mutation == 'late':
        lock['optimizer_updates_at_lock'] += 1
    elif mutation == 'suffix':
        lock['suffix_observations_used'] = 1
    elif mutation == 'missing':
        del evidence['adam_locks'][family]
    else:
        lock['candidates'][0]['error_ratio'] = float('nan')
    with pytest.raises(ValueError):
        load_written(tmp_path, evidence)


@pytest.mark.parametrize('field', available.DATA_KEYS)
def test_null_rejects_any_changed_input_or_target_fingerprint(tmp_path, prefix_evidence, field):
    evidence = prefix_evidence[3]
    fingerprint = dict(evidence['data_sha256'], **{field: '0' * 64})
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_written(tmp_path, evidence, fingerprint=fingerprint)


@pytest.mark.parametrize('case', ['source', 'reference', 'labels', 'unconsumed', 'null', 'failed',
                                  'wrong_seed', 'boundary', 'missing_phase', 'grid', 'names',
                                  'incomplete_completion', 'processed', 'primary', 'horizon', 'raw_zero'])
def test_null_rejects_inconsistent_real_provenance(tmp_path, prefix_evidence, case):
    evidence = copy.deepcopy(prefix_evidence[3])
    overrides = {}
    if case == 'source':
        overrides['source_sha256'] = {'model.py': '0' * 64}
    elif case == 'reference':
        overrides['reference'] = dict(evidence['reference'], artifact_sha256='0' * 64)
    elif case == 'labels':
        evidence['view_target_sha256'] = '0' * 64
    elif case == 'raw_zero':
        overrides['raw_zero_sha256'] = '0' * 64
    elif case == 'unconsumed':
        evidence.update(processed=available.SELECTION_END - 1, processed_observations=available.SELECTION_END - 1)
    elif case == 'null':
        evidence['args']['view'] = 'random_sign'
    elif case == 'failed':
        evidence['status'] = 'failed'
    elif case == 'wrong_seed':
        evidence['args']['seed'] = 2
    elif case == 'boundary':
        evidence['args']['selection_end'] -= 1
    elif case == 'missing_phase':
        del evidence['phases']['suffix_all']
    elif case == 'grid':
        evidence['adam_grid'] = evidence['adam_grid'][:-1]
    elif case == 'names':
        evidence['output_names'].reverse()
    elif case == 'incomplete_completion':
        evidence['processed'] = evidence['processed_observations'] = available.SELECTION_END
    elif case == 'processed':
        evidence['processed'] -= 1
    elif case == 'primary':
        evidence['protocol']['primary_output'] = 'latest_dense_0.0001'
    else:
        overrides['samples'] = evidence['samples'] - 1
    with pytest.raises(ValueError):
        load_written(tmp_path, evidence, **overrides)


@pytest.fixture(scope='module')
def reference_arrays():
    # Full production-width original frame, including the last row beyond the barrier.
    samples = available.SELECTION_END + 1
    original = np.zeros((samples, 224), dtype=np.float32)
    target = np.ones(samples, dtype=np.float32)
    reference = {'args': {'view': 'real', 'seed': 1}, 'status': 'pruned',
                 'processed': available.SELECTION_END, 'processed_observations': available.SELECTION_END,
                 'samples': samples, 'maximum_observations': samples,
                 'source_sha256': {'predictive_conditional_stack_v7.py': 'a' * 64,
                                   'stock_stream.py': 'b' * 64},
                 'data_sha256': {'features': available.array_sha256(original),
                                 'target': available.array_sha256(target)},
                 'stock_helper_args': {'lags': 32, 'steps': 0, 'center': True, 'raw_target': False,
                                       'vol_feature': False, 'vol_span': .01}}
    return original, target, reference


@pytest.mark.parametrize('change', ['none', 'original', 'target', 'last_target', 'horizon', 'helper', 'normalization'])
def test_reference_requires_identical_full_original_task(tmp_path, reference_arrays, change):
    original, target, template = reference_arrays
    reference = copy.deepcopy(template)
    helper_hash = 'b' * 64
    if change == 'original':
        reference['data_sha256']['features'] = '0' * 64
    elif change == 'target':
        reference['data_sha256']['target'] = '0' * 64
    elif change == 'last_target':
        target = target.copy()
        target[-1] = 2
    elif change == 'horizon':
        reference['samples'] -= 1
    elif change == 'helper':
        helper_hash = '0' * 64
    elif change == 'normalization':
        reference['stock_helper_args']['raw_target'] = True
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference), encoding='utf-8')
    if change == 'none':
        verified = available.verify_reference(path, original, target, helper_hash)
        assert verified['samples'] == len(target)
        assert verified['data_sha256'] == reference['data_sha256']
        assert verified['original_features_bitwise_verified'] and verified['target_bitwise_verified']
    else:
        with pytest.raises(ValueError):
            available.verify_reference(path, original, target, helper_hash)


def test_cli_rejects_changed_reference_before_cuda(tmp_path, reference_arrays, monkeypatch):
    original, target, template = reference_arrays
    reference = copy.deepcopy(template)
    reference['data_sha256']['features'] = '0' * 64
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference), encoding='utf-8')
    root = tmp_path / 'run'
    args = available.Args(output_dir=str(root), reference_result=str(path))
    monkeypatch.setattr(available.tyro, 'cli', lambda _: args)
    monkeypatch.setattr(available.stock_stream, 'read_bars', lambda _: object())
    monkeypatch.setattr(available, 'build_paired_stream',
                        lambda bars, config: (original, original, target, np.zeros_like(target)))
    monkeypatch.setattr(available, 'source_fingerprint', lambda: {'stock_stream.py': 'b' * 64})

    def forbidden_cuda():
        pytest.fail('CUDA was inspected before rejecting a changed reference task')

    monkeypatch.setattr(torch.cuda, 'is_available', forbidden_cuda)
    with pytest.raises(ValueError, match='reference full-file features SHA256 mismatch'):
        available.main()
    result = json.loads((root / 'results.json').read_text())
    assert result['status'] == 'failed' and result['processed_observations'] == 0
    assert not (root / 'predictions.npy').exists()


def test_columns_remain_semantic_with_extra_model_outputs_and_reordering(prefix_evidence):
    names, evidence = prefix_evidence[2:]
    widened = available.output_names((*available.MODEL_NAMES, 'extra_diagnostic'))
    assert len(names) == 69 and len(widened) == 70
    for altered in (widened, tuple(reversed(widened))):
        columns = available.selected_columns(altered, evidence['adam_locks'])
        assert altered[columns[available.PRIMARY]] == available.PRIMARY
        for family in available.ADAM_FAMILIES:
            frame, kind = family.split('_')
            lr = evidence['adam_locks'][family]['selected_lr']
            assert altered[columns[f'adam_{family}']] == f'{frame}_adam_{kind}_{lr:g}'
    unlocked = available.selected_columns(names, None)
    assert available.PRIMARY in unlocked
    assert not any(key.startswith('adam_') for key in unlocked)


@pytest.mark.parametrize('family', available.ADAM_FAMILIES)
def test_failed_prefix_arm_is_not_silently_omitted(prefix_evidence, family):
    prediction, target, names, _ = prefix_evidence
    altered = prediction[:available.SELECTION_END].copy()
    altered[available.COLD_START, available.candidate_columns(names, family)[0]] = np.nan
    with pytest.raises(FloatingPointError):
        available.select_prefix_locks(altered, target[:available.SELECTION_END], names, available.SELECTION_END)


@pytest.mark.parametrize('consumed', [0, 2000, available.SELECTION_END, available.SELECTION_END + 3])
def test_comparisons_use_only_consumed_labels_for_both_information_frames(prefix_evidence, consumed):
    names, evidence = prefix_evidence[2:]
    samples = available.SELECTION_END + 7
    columns = available.selected_columns(names, evidence['adam_locks'])
    target = np.linspace(.1, 1.2, samples, dtype=np.float32)
    factors = np.arange(len(names), dtype=np.float32) / 67
    factors[columns['original_zero']] = factors[columns['latest_zero']] = 0
    prediction = target[:, None] * factors[None, :]
    target[consumed:] = np.nan
    prediction[consumed:] = np.nan
    phases = available.phase_ranges(samples)
    summary = available.summarize_predictions(target, prediction[:consumed], names, phases, consumed)
    pairs = available.paired_comparisons(summary, names, columns)
    for phase, (start, stop) in phases.items():
        count = max(0, min(stop, consumed) - start)
        row = summary[phase]
        assert row['count'] == count
        assert row['complete'] is (consumed >= stop)
        assert row['expected_count'] == stop - start
        if not count:
            assert row['metrics'] == pairs[phase]['comparisons'] == []
            continue
        assert {pair['baseline'] for pair in pairs[phase]['comparisons']} == {
            'original_dense_bayes_mixture', 'original_zero', 'latest_zero', 'raw_zero_transformed',
            *(f'adam_{family}' for family in available.ADAM_FAMILIES)}
        yy = target[start:start + count].astype(np.float64)
        primary = prediction[start:start + count, columns[available.PRIMARY]].astype(np.float64)
        for pair in pairs[phase]['comparisons']:
            baseline = prediction[start:start + count, columns[pair['baseline']]].astype(np.float64)
            difference = np.square(yy - primary).sum() - np.square(yy - baseline).sum()
            assert pair['mse_difference'] == pytest.approx(difference / count, abs=1e-12)
            assert pair['error_ratio_difference'] == pytest.approx(difference / np.square(yy).sum(), abs=1e-12)


@pytest.mark.parametrize('status,nonfinite', [('pruned', False), ('failed', True)])
def test_partial_artifacts_preserve_raw69_and_both_complete_cheap_states(tmp_path, prefix_evidence,
                                                                      status, nonfinite):
    names, evidence = prefix_evidence[2:]
    samples, consumed = available.SELECTION_END + 2, 3
    predictions = torch.full((samples, len(names) - 1), 12345.)
    predictions[:consumed] = torch.arange(consumed, dtype=torch.float32)[:, None] / 4
    if nonfinite:
        predictions[1, names.index(available.PRIMARY)] = float('nan')
    states = [torch.tensor([float(i)]) for i in range(10)]
    clock = torch.tensor(consumed)
    model_frames, banks, trajectories = {}, {}, []
    for number, frame in enumerate(('original', 'latest')):
        frame_states = states[number * 5:(number + 1) * 5]
        model_frames[frame] = SimpleNamespace(state_tensors=lambda values=frame_states: values,
                                              observations=clock, alpha=torch.tensor(2 + consumed / 2),
                                              dense_cov=torch.eye(2).unsqueeze(0), diag_cov=torch.ones(1, 2))
        trajectory = torch.zeros(samples, 11)
        trajectories.append(trajectory)
        banks[frame] = SimpleNamespace(linear=SimpleNamespace(index=clock, steps=clock),
                                       mlp=SimpleNamespace(index=clock, steps=clock, predictions=trajectory))
    model = SimpleNamespace(state_tensors=lambda: states, configs={}, observations=clock, **model_frames)
    runner = SimpleNamespace(predictions=predictions, index=clock, model=model,
                             adam=SimpleNamespace(mutable=[*states, *trajectories],
                                                  configs=available.ADAM_GRID, **banks))
    target = np.ones(samples, dtype=np.float32)
    result = {'status': status, 'adam_grid': available.ADAM_GRID,
              'adam_locks': evidence['adam_locks'], 'exit_code': 75}
    causal_raw_zero = np.full(samples, np.nan, dtype=np.float32)
    causal_raw_zero[:consumed] = [-.2, .3, -.4]
    available.save_artifacts(tmp_path, runner, target, names, available.phase_ranges(samples), consumed,
                             ProxyCull(1, {'error_ratio': 1e-4}), result, causal_raw_zero)
    stored = np.load(tmp_path / 'predictions.npy')
    np.testing.assert_equal(stored[:, :-1], predictions[:consumed].numpy())
    np.testing.assert_equal(stored[:, -1], causal_raw_zero[:consumed])
    np.testing.assert_equal(np.load(tmp_path / 'targets.npy'), target[:consumed])
    saved = json.loads((tmp_path / 'results.json').read_text())
    assert saved['status'] == status
    assert saved['processed'] == saved['processed_observations'] == consumed
    assert saved['nonfinite_predictions'] == int(nonfinite)
    assert saved['phase_metrics']['suffix_all']['count'] == 0
    assert saved['paired_comp']['suffix_all']['comparisons'] == []
    assert len(saved['paired_comp']['prefix_all']['comparisons']) == 8
    assert set(saved['clocks'].values()) == {consumed}
    assert set(saved['covariance_health']) == {'original', 'latest'}
    assert 'checkpoint_diagnostic_failure' not in saved
    if nonfinite:
        assert saved['phase_metrics']['prefix_all']['metrics'][names.index(available.PRIMARY)]['error_ratio'] is None
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    assert checkpoint['step'] == consumed
    for key, expected in (('model_state', states), ('original_state', states[:5]),
                          ('latest_state', states[5:]), ('adam_state', states)):
        for actual, reference in zip(checkpoint[key], expected, strict=True):
            torch.testing.assert_close(actual, reference)
    for frame in ('original', 'latest'):
        assert checkpoint['trajectory_columns'][frame] == available.candidate_columns(names, f'{frame}_mlp')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; run via mlq')
@torch.no_grad()
def test_actual_paired_stock_capture_restores_nested_trajectories_and_all68_prelabel_columns(monkeypatch):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(19, 448, device='cuda', generator=generator)
    ys = torch.randn(19, device='cuda', generator=generator)
    model = AvailableStateComparison(448, 'cuda')
    adam = PairedAdamBank(xs, ys, available.ADAM_GRID)
    runner = available.StockRunner(xs, ys, model, adam)
    names = available.output_names(model.output_names, include_diagnostic=False)
    assert names == available.output_names(include_diagnostic=False)
    states = runner.mutable
    initial = [tensor.clone() for tensor in states]
    # Inspect independently: omitted nested buffers must not evade restoration checks.
    nested = (adam.original.mlp.predictions, adam.latest.mlp.predictions)
    nested_initial = [tensor.clone() for tensor in nested]
    real_compile, callables = torch.compile, []

    def recording_compile(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        callables.append(compiled)
        return compiled

    def restore(saved):
        for tensor, value in zip(states, saved, strict=True):
            tensor.copy_(value)

    def assert_state(expected):
        for tensor, reference in zip(states, expected, strict=True):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)

    monkeypatch.setattr(torch, 'compile', recording_compile)
    graphs = runner.capture(16)
    assert_state(initial)
    for tensor, reference in zip(nested, nested_initial, strict=True):
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
    update = callables[-1]
    graphs[16].replay()
    for _ in range(3):
        graphs[1].replay()
    expected = [tensor.clone() for tensor in states]
    nested_expected = [tensor.clone() for tensor in nested]
    restore(initial)
    for _ in range(len(xs)):
        update()
    torch.cuda.synchronize()
    assert_state(expected)
    for frame, actual, reference in zip(('original', 'latest'), nested, nested_expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        columns = available.candidate_columns(names, f'{frame}_mlp')
        torch.testing.assert_close(runner.predictions[:, columns], actual, rtol=0, atol=0)
    clocks, health = available.checkpoint_health(runner)
    assert set(clocks.values()) == {len(xs)}
    for row in health.values():
        assert set(row['cholesky_info']) == {0}
        assert row['diagonal_finite'] and row['diagonal_min'] > 0
    restore(initial)
    for _ in range(4):
        update()
    before = [tensor.clone() for tensor in states]
    original = ys[4].clone()
    try:
        ys[4] = -4
        update()
        forecast_negative = runner.predictions[4].clone()
        update()
        next_negative = runner.predictions[5].clone()
        restore(before)
        ys[4] = 4
        update()
        forecast_positive = runner.predictions[4].clone()
        update()
        next_positive = runner.predictions[5].clone()
        torch.testing.assert_close(forecast_negative, forecast_positive, rtol=0, atol=0)
        for family in available.ADAM_FAMILIES:
            columns = available.candidate_columns(names, family)
            assert not torch.equal(next_negative[columns], next_positive[columns])
        for frame in ('original', 'latest'):
            columns = [names.index(f'{frame}_{name}') for name in available.BASE_NAMES if name != 'zero']
            assert not torch.equal(next_negative[columns], next_positive[columns])
    finally:
        ys[4].copy_(original)
