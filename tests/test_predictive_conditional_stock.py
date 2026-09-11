"""Host provenance/censoring contracts and the real compiled stock-runner integration.

CUDA test execution belongs in mlq. These tests never substitute a shorter market
run for the real experiment; the synthetic integration defends graph/state causality.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import predictive_conditional_stock_eval_v4 as conditional
from cleanrl.plasticity.predictive_conditional_stack_v7 import StateConditionedStack
from cleanrl.shared import runtime
from cleanrl.shared.autocull import ProxyCull

FAMILIES = (('model_locks', 'calibration'), ('model_locks', 'stacking'), ('model_locks', 'context'),
            ('adam_locks', 'linear'), ('adam_locks', 'mlp'), ('adam_locks', 'context'))


@pytest.fixture(scope='module')
def prefix_evidence():
    end = conditional.SELECTION_END
    names = conditional.output_names()
    target = np.ones(end + 1, dtype=np.float32)
    prediction = np.full((end + 1, len(names)), .25, dtype=np.float32)
    for number, (key, family) in enumerate(FAMILIES):
        grid = conditional.PRIORS if key == 'model_locks' else conditional.ADAM_GRID
        columns = conditional.candidate_columns(names, family, grid, adam=key == 'adam_locks')
        prediction[conditional.COLD_START:end, columns[(number + 1) % len(grid)]] = .75
        # The intended winner would lose if the scoring interval included cold start.
        prediction[:conditional.COLD_START, columns[(number + 1) % len(grid)]] = 1000
    prediction[:, names.index('base_zero')] = 0
    locks = conditional.select_prefix_locks(prediction[:end], target[:end], names, end)
    samples = end + 20000
    evidence = {
        'args': {'view': 'real', 'seed': 1, 'selection_end': end},
        'data_sha256': {'features': 'a' * 64, 'target': 'b' * 64},
        'source_sha256': {'predictive_conditional_stack_v7.py': 'c' * 64},
        'samples': samples, 'maximum_observations': samples,
        'processed': samples, 'processed_observations': samples, 'status': 'completed',
        'phases': {name: list(bounds) for name, bounds in conditional.phase_ranges(samples).items()},
        'model_priors': list(conditional.PRIORS), 'adam_grid': list(conditional.ADAM_GRID),
        'output_names': list(names), **locks,
    }
    return prediction, target, names, evidence


def load_written(tmp_path, evidence, **kwargs):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(evidence), encoding='utf-8')
    return conditional.load_real_locks(
        path, kwargs.get('fingerprint', evidence['data_sha256']),
        kwargs.get('source_sha256', evidence['source_sha256']),
        kwargs.get('selection_end', conditional.SELECTION_END))


@pytest.mark.parametrize('key,family', FAMILIES)
def test_all_six_locks_exclude_cold_start_and_first_suffix_label(prefix_evidence, key, family):
    prediction, target, names, evidence = prefix_evidence
    end = conditional.SELECTION_END
    number = FAMILIES.index((key, family))
    grid = conditional.PRIORS if key == 'model_locks' else conditional.ADAM_GRID
    expected = evidence[key][family]
    assert expected['selected_index'] == (number + 1) % len(grid)
    assert expected['selection_start_inclusive'] == conditional.COLD_START
    assert expected['selection_end_exclusive'] == expected['optimizer_updates_at_lock'] == end
    assert expected['suffix_observations_used'] == 0
    assert {row['count'] for row in expected['candidates']} == {end - conditional.COLD_START}
    altered_prediction, altered_target = prediction.copy(), target.copy()
    altered_prediction[end] = np.nan
    altered_target[end] = np.nan
    actual = conditional.select_prefix_locks(altered_prediction[:end], altered_target[:end], names, end)
    assert actual[key][family] == expected
    # No selector can be called late, early, with an altered barrier, or with
    # full future arrays (even if its consumed argument claims the prefix).
    for consumed, barrier, length in ((end + 1, end, end + 1), (end - 1, end, end),
                                       (end - 1, end - 1, end - 1), (end, end, end + 1)):
        with pytest.raises(ValueError, match='exact 294912 barrier'):
            conditional.select_prefix_locks(altered_prediction[:length], altered_target[:length],
                                             names, consumed, barrier)


@pytest.mark.parametrize('status,extra', [('running', 0), ('pruned', 10000), ('completed', 20000)])
def test_null_inherits_all_six_real_locks_without_reselection(tmp_path, prefix_evidence, status, extra):
    evidence = copy.deepcopy(prefix_evidence[3])
    evidence.update(status=status, processed=conditional.SELECTION_END + extra,
                    processed_observations=conditional.SELECTION_END + extra)
    assert load_written(tmp_path, evidence) == {key: evidence[key] for key in ('model_locks', 'adam_locks')}


@pytest.mark.parametrize('key,family', FAMILIES)
def test_each_family_rejects_fabricated_winner_and_late_lock(tmp_path, prefix_evidence, key, family):
    for mutation in ('winner', 'late', 'suffix', 'missing'):
        evidence = copy.deepcopy(prefix_evidence[3])
        lock = evidence[key][family]
        if mutation == 'winner':
            grid = conditional.PRIORS if key == 'model_locks' else conditional.ADAM_GRID
            value_key = 'selected_prior' if key == 'model_locks' else 'selected_lr'
            wrong = (lock['selected_index'] + 1) % len(grid)
            lock.update(selected_index=wrong, **{value_key: grid[wrong]})
        elif mutation == 'late':
            lock['optimizer_updates_at_lock'] += 1
        elif mutation == 'suffix':
            lock['suffix_observations_used'] = 1
        else:
            del evidence[key][family]
        with pytest.raises(ValueError):
            load_written(tmp_path, evidence)


@pytest.mark.parametrize('field', ['features', 'target', 'source'])
def test_null_rejects_wrong_data_or_source_sha256(tmp_path, prefix_evidence, field):
    evidence = prefix_evidence[3]
    kwargs = ({'source_sha256': {'predictive_conditional_stack_v7.py': 'd' * 64}} if field == 'source'
              else {'fingerprint': {**evidence['data_sha256'], field: 'd' * 64}})
    with pytest.raises(ValueError, match='SHA256'):
        load_written(tmp_path, evidence, **kwargs)


@pytest.mark.parametrize('case', ['unconsumed', 'old_boundary', 'null', 'failed', 'wrong_seed',
                                  'output_order', 'priors', 'adam_grid', 'inconsistent_score',
                                  'nonfinite_score', 'wrong_count', 'different_targets',
                                  'incomplete_completion', 'processed_disagreement', 'cold_start'])
def test_transfer_rejects_inconsistent_provenance(tmp_path, prefix_evidence, case):
    evidence = copy.deepcopy(prefix_evidence[3])
    lock = evidence['model_locks']['context']
    kwargs = {}
    if case == 'unconsumed':
        evidence.update(status='pruned', processed=conditional.SELECTION_END - 1,
                        processed_observations=conditional.SELECTION_END - 1)
    elif case == 'old_boundary':
        evidence['args']['selection_end'] = 237568
        kwargs['selection_end'] = 237568
    elif case == 'null':
        evidence['args']['view'] = 'random_sign'
    elif case == 'failed':
        evidence['status'] = 'failed'
    elif case == 'wrong_seed':
        evidence['args']['seed'] = 2
    elif case == 'output_order':
        evidence['output_names'].reverse()
    elif case == 'priors':
        evidence['model_priors'][0] *= 2
    elif case == 'adam_grid':
        evidence['adam_grid'][0] *= 2
    elif case == 'inconsistent_score':
        lock['candidates'][0]['error_squared_sum'] *= 2
    elif case == 'nonfinite_score':
        lock['candidates'][0]['error_ratio'] = None
    elif case == 'wrong_count':
        lock['candidates'][0]['count'] -= 1
    elif case == 'different_targets':
        lock['candidates'][0]['target_squared_sum'] *= 2
    elif case == 'incomplete_completion':
        evidence['processed'] -= 1
        evidence['processed_observations'] -= 1
    elif case == 'processed_disagreement':
        evidence['processed'] -= 1
    else:
        lock['selection_start_inclusive'] = 0
    with pytest.raises(ValueError):
        load_written(tmp_path, evidence, **kwargs)


def test_semantic_columns_survive_extra_model_outputs_and_reordering(prefix_evidence):
    prediction, target, names, evidence = prefix_evidence
    assert len(names) == 61
    end = conditional.SELECTION_END
    reordered = ('unrelated_model_control', *reversed(names))
    altered = np.concatenate((np.full((end, 1), np.nan, dtype=np.float32), prediction[:end, ::-1]), axis=1)
    locks = conditional.select_prefix_locks(altered, target[:end], reordered, end)
    assert locks == {key: evidence[key] for key in ('model_locks', 'adam_locks')}
    columns = conditional.selected_columns(reordered, **locks)
    for family in conditional.MODEL_FAMILIES:
        prior = locks['model_locks'][family]['selected_prior']
        assert reordered[columns[family]] == f'{family}_{prior:g}'
    for family in conditional.ADAM_FAMILIES:
        lr = locks['adam_locks'][family]['selected_lr']
        prefix = 'context_adam' if family == 'context' else f'adam_{family}'
        assert reordered[columns[f'adam_{family}']] == f'{prefix}_{lr:g}'
    assert reordered[columns['base_static_mixture']] == 'base_static_mixture'
    assert reordered[columns['base_zero']] == 'base_zero'


@pytest.mark.parametrize('key,family', FAMILIES)
def test_nonfinite_candidate_cannot_be_silently_dropped_from_selection(prefix_evidence, key, family):
    prediction, target, names, _ = prefix_evidence
    end = conditional.SELECTION_END
    grid = conditional.PRIORS if key == 'model_locks' else conditional.ADAM_GRID
    columns = conditional.candidate_columns(names, family, grid, adam=key == 'adam_locks')
    altered = prediction[:end].copy()
    altered[conditional.COLD_START, columns[0]] = np.nan
    with pytest.raises(FloatingPointError, match='no selective omission'):
        conditional.select_prefix_locks(altered, target[:end], names, end)


@pytest.mark.parametrize('consumed', [0, 2000, conditional.SELECTION_END, conditional.SELECTION_END + 3])
def test_paired_metrics_use_only_consumed_predictions_for_every_control(prefix_evidence, consumed):
    names, evidence = prefix_evidence[2:]
    samples = conditional.SELECTION_END + 7
    columns = conditional.selected_columns(names, evidence['model_locks'], evidence['adam_locks'])
    target = np.linspace(.1, 1.2, samples, dtype=np.float32)
    factors = np.arange(len(names), dtype=np.float32) / 67
    factors[columns['base_zero']] = 0
    prediction = target[:, None] * factors[None, :]
    target[consumed:] = np.nan
    prediction[consumed:] = np.nan
    phases = conditional.phase_ranges(samples)
    summary = conditional.summarize_predictions(target, prediction[:consumed], names, phases, consumed)
    pairs = conditional.paired_comparisons(summary, names, columns)
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
            'calibration', 'stacking', 'base_static_mixture', 'base_adaptive_mixture',
            'base_dynamic_mixture', 'base_zero', 'adam_linear', 'adam_mlp', 'adam_context'}
        yy = target[start:start + count].astype(np.float64)
        primary = prediction[start:start + count, columns['context']].astype(np.float64)
        for pair in pairs[phase]['comparisons']:
            baseline = prediction[start:start + count, columns[pair['baseline']]].astype(np.float64)
            difference = np.square(yy - primary).sum() - np.square(yy - baseline).sum()
            assert pair['mse_difference'] == pytest.approx(difference / count, abs=1e-12)
            assert pair['error_ratio_difference'] == pytest.approx(difference / np.square(yy).sum(), abs=1e-12)


@pytest.mark.parametrize('status,nonfinite', [('pruned', False), ('failed', True)])
def test_partial_artifacts_keep_raw_failure_and_full_state_without_future_slots(tmp_path, prefix_evidence,
                                                                              status, nonfinite):
    names, evidence = prefix_evidence[2:]
    samples, consumed = conditional.SELECTION_END + 2, 3
    predictions = torch.full((samples, len(names)), 12345.)
    predictions[:consumed] = torch.arange(consumed, dtype=torch.float32)[:, None] / 4
    if nonfinite:
        predictions[1, names.index('context_0.1')] = float('nan')
    trajectory = torch.zeros(samples, 11)
    states = [torch.tensor([float(i)]) for i in range(10)]
    clock = torch.tensor(consumed)
    model = SimpleNamespace(state_tensors=lambda: states, configs={}, observations=clock,
                            base=SimpleNamespace(state_tensors=lambda: states[:4], observations=clock,
                                                 cov=torch.eye(2).unsqueeze(0)),
                            meta=SimpleNamespace(state_tensors=lambda: states[4:8],
                                                 alpha=torch.tensor(2 + consumed / 2),
                                                 cov=torch.eye(2).unsqueeze(0)),
                            context_adam=SimpleNamespace(state_tensors=lambda: states[8:], steps=clock))
    runner = SimpleNamespace(predictions=predictions, index=clock, model=model,
                             adam=SimpleNamespace(mutable=[states[0], trajectory],
                                                  linear=SimpleNamespace(index=clock, steps=clock),
                                                  mlp=SimpleNamespace(predictions=trajectory, index=clock, steps=clock),
                                                  configs=conditional.ADAM_GRID))
    target = np.ones(samples, dtype=np.float32)
    result = {'status': status, 'adam_grid': conditional.ADAM_GRID, 'model_priors': conditional.PRIORS,
              'adam_locks': evidence['adam_locks'], 'model_locks': evidence['model_locks'], 'exit_code': 75}
    conditional.save_artifacts(tmp_path, runner, target, names, conditional.phase_ranges(samples), consumed,
                               ProxyCull(1, {'error_ratio': 1e-4}), result)
    np.testing.assert_equal(np.load(tmp_path / 'predictions.npy'), predictions[:consumed].numpy())
    np.testing.assert_equal(np.load(tmp_path / 'targets.npy'), target[:consumed])
    saved = json.loads((tmp_path / 'results.json').read_text())
    assert saved['status'] == status
    assert saved['processed'] == saved['processed_observations'] == consumed
    assert saved['nonfinite_predictions'] == int(nonfinite)
    assert saved['phase_metrics']['suffix_all']['count'] == 0
    assert saved['paired_comp']['suffix_all']['comparisons'] == []
    assert len(saved['paired_comp']['prefix_all']['comparisons']) == 9
    assert set(saved['clocks'].values()) == {consumed}
    assert set(saved['covariance_health']) == {'base', 'meta'}
    assert 'checkpoint_diagnostic_failure' not in saved
    if nonfinite:
        assert saved['phase_metrics']['prefix_all']['metrics'][names.index('context_0.1')]['error_ratio'] is None
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    assert checkpoint['step'] == consumed
    for key, expected in (('model_state', states), ('base_state', states[:4]),
                          ('meta_state', states[4:8]), ('context_adam_state', states[8:])):
        for actual, reference in zip(checkpoint[key], expected, strict=True):
            torch.testing.assert_close(actual, reference)
    assert len(checkpoint['adam_state']) == 1


def test_unlocked_failed_prefix_never_fabricates_a_primary_selection():
    names = conditional.output_names()
    columns = conditional.selected_columns(names, None, None)
    summary = conditional.summarize_predictions(np.ones(3), np.zeros((3, len(names))), names,
                                                 {'observed': (0, 3)}, 3)
    assert 'context' not in columns
    assert conditional.paired_comparisons(summary, names, columns)['observed']['comparisons'] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; run via mlq')
@torch.no_grad()
def test_actual_stock_capture_restores_all_clocks_and_prelabel_columns(monkeypatch):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(19, 224, device='cuda', generator=generator)
    ys = torch.randn(19, device='cuda', generator=generator)
    model = StateConditionedStack(224, 'cuda')
    adam = conditional.AdamBank(xs, ys, conditional.ADAM_GRID)
    runner = conditional.StockRunner(xs, ys, model, adam)
    names = conditional.output_names(model.output_names)
    assert names == conditional.output_names()
    states = runner.mutable
    initial = [tensor.clone() for tensor in states]
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
    update = callables[-1]
    graphs[16].replay()
    for _ in range(3):
        graphs[1].replay()
    expected = [tensor.clone() for tensor in states]
    restore(initial)
    for _ in range(len(xs)):
        update()
    torch.cuda.synchronize()
    assert_state(expected)
    clocks, health = conditional.checkpoint_health(runner)
    assert set(clocks.values()) == {len(xs)}
    for row in health.values():
        assert set(row['cholesky_info']) == {0}
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
        # Every one of 61 forecasts is prelabel, including the copied base rows.
        torch.testing.assert_close(forecast_negative, forecast_positive, rtol=0, atol=0)
        for key, family in FAMILIES:
            grid = conditional.PRIORS if key == 'model_locks' else conditional.ADAM_GRID
            columns = conditional.candidate_columns(names, family, grid, adam=key == 'adam_locks')
            assert not torch.equal(next_negative[columns], next_positive[columns])
        base_columns = [i for i, name in enumerate(names) if name.startswith('base_') and name != 'base_zero']
        assert not torch.equal(next_negative[base_columns], next_positive[base_columns])
    finally:
        ys[4].copy_(original)
