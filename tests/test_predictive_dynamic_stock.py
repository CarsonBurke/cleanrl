"""Host provenance/observed-evidence contracts and actual stock-runner CUDA capture.

CUDA integration must run through mlq; it complements the model recurrence tests
by exercising the unchanged linear/MLP Adam banks and shared stock runner.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import predictive_dynamic_stock_eval_v3 as dynamic
from cleanrl.plasticity.predictive_dynamic_nig_v6 import DynamicNIG
from cleanrl.shared import runtime
from cleanrl.shared.autocull import ProxyCull


@pytest.fixture(scope='module')
def real_evidence():
    end = dynamic.SELECTION_END
    target = np.linspace(-1.25, 2.5, end, dtype=np.float32)
    grid = dynamic.ADAM_GRID
    locks = {}
    for family, winner in (('linear', 7), ('mlp', 9)):
        prediction = target[:, None] * np.full((1, len(grid)), .25, dtype=np.float32)
        prediction[:, winner] = target * .75
        locks[family] = stock.select_prefix(prediction, target, grid, dynamic.COLD_START, end, end)
    samples = end + 20000
    return {
        'args': {'view': 'real', 'seed': 1, 'selection_end': end},
        'data_sha256': {'features': 'a' * 64, 'target': 'b' * 64},
        'source_sha256': {'predictive_dynamic_nig_v6.py': 'c' * 64},
        'samples': samples, 'maximum_observations': samples,
        'processed': samples, 'processed_observations': samples, 'status': 'completed',
        'phases': {name: list(bounds) for name, bounds in dynamic.phase_ranges(samples).items()},
        'model_configs': copy.deepcopy(list(dynamic.MODEL_CONFIGS)),
        'output_names': list(dynamic.output_names()), 'adam_grid': list(grid), 'adam_locks': locks,
    }


def load_written(tmp_path, source, **kwargs):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(source), encoding='utf-8')
    return dynamic.load_real_locks(
        path, kwargs.get('fingerprint', source['data_sha256']),
        kwargs.get('source_sha256', source['source_sha256']),
        kwargs.get('selection_end', dynamic.SELECTION_END),
        kwargs.get('adam_grid', dynamic.ADAM_GRID))


@pytest.mark.parametrize('status,extra', [('running', 0), ('pruned', 10000), ('completed', 20000)])
def test_transfer_preserves_both_real_adam_locks_at_barrier_or_terminal(tmp_path, real_evidence, status, extra):
    source = copy.deepcopy(real_evidence)
    consumed = dynamic.SELECTION_END + extra
    source.update(status=status, processed=consumed, processed_observations=consumed)
    transferred = load_written(tmp_path, source)
    assert transferred == {'adam_locks': source['adam_locks']}
    assert transferred['adam_locks']['linear']['selected_index'] == 7
    assert transferred['adam_locks']['mlp']['selected_index'] == 9


@pytest.mark.parametrize('field', ['features', 'target', 'source'])
def test_null_transfer_rejects_wrong_data_or_algorithm_hash(tmp_path, real_evidence, field):
    kwargs = ({'source_sha256': {'predictive_dynamic_nig_v6.py': 'd' * 64}} if field == 'source'
              else {'fingerprint': {**real_evidence['data_sha256'], field: 'd' * 64}})
    with pytest.raises(ValueError, match='SHA256'):
        load_written(tmp_path, real_evidence, **kwargs)


@pytest.mark.parametrize('case', [
    'old_boundary', 'prefix_unconsumed', 'suffix_used', 'late_lock', 'cold_start_included',
    'null_source', 'failed_source', 'wrong_seed', 'missing_family', 'model_configs',
    'output_order', 'adam_grid', 'fabricated_winner', 'inconsistent_score',
    'nonfinite_score', 'wrong_count', 'incomplete_completion', 'processed_disagreement',
])
def test_transfer_rejects_noncausal_or_inconsistent_evidence(tmp_path, real_evidence, case):
    source = copy.deepcopy(real_evidence)
    lock = source['adam_locks']['linear']
    kwargs = {}
    if case == 'old_boundary':
        source['args']['selection_end'] = 163840
        kwargs['selection_end'] = 163840
    elif case == 'prefix_unconsumed':
        source.update(status='pruned', processed=dynamic.SELECTION_END - 1,
                      processed_observations=dynamic.SELECTION_END - 1)
    elif case == 'suffix_used':
        lock['suffix_observations_used'] = 1
    elif case == 'late_lock':
        lock['optimizer_updates_at_lock'] += 1
    elif case == 'cold_start_included':
        lock['selection_start_inclusive'] = 0
    elif case == 'null_source':
        source['args']['view'] = 'random_sign'
    elif case == 'failed_source':
        source['status'] = 'failed'
    elif case == 'wrong_seed':
        source['args']['seed'] = 2
    elif case == 'missing_family':
        del source['adam_locks']['mlp']
    elif case == 'model_configs':
        source['model_configs'][5]['halflife'] *= 2
    elif case == 'output_order':
        source['output_names'][0:2] = source['output_names'][1::-1]
    elif case == 'adam_grid':
        source['adam_grid'][0] *= 2
    elif case == 'fabricated_winner':
        lock.update(selected_index=0, selected_lr=dynamic.ADAM_GRID[0])
    elif case == 'inconsistent_score':
        lock['candidates'][0]['error_squared_sum'] *= 2
    elif case == 'nonfinite_score':
        lock['candidates'][0]['error_ratio'] = None
    elif case == 'wrong_count':
        lock['candidates'][0]['count'] -= 1
    elif case == 'incomplete_completion':
        source['processed_observations'] -= 1
        source['processed'] -= 1
    elif case == 'processed_disagreement':
        source['processed'] -= 1
    with pytest.raises(ValueError):
        load_written(tmp_path, source, **kwargs)


def test_prefix_ignores_cold_start_and_first_suffix_label_with_variable_model_width():
    end = dynamic.SELECTION_END
    grid = (.001, .01)
    # Deliberately not the default model width: fixed offsets would select experts.
    names = dynamic.output_names((*dynamic.MODEL_NAMES[:4], 'static_test', 'discount_test'), grid)
    width = len(names) - 2 * len(grid)
    target = np.ones(end + 1, dtype=np.float32)
    prediction = np.zeros((end + 1, len(names)), dtype=np.float32)
    prediction[:dynamic.COLD_START, width] = 1000
    prediction[dynamic.COLD_START:end, width:width + 2] = [.8, .2]
    prediction[dynamic.COLD_START:end, width + 2:] = [.2, .8]
    before = dynamic.select_adam_prefix(prediction[:end], target[:end], names, end, adam_grid=grid)
    prediction[end] = 1e20
    target[end] = 1e20
    after = dynamic.select_adam_prefix(prediction[:end], target[:end], names, end, adam_grid=grid)
    assert before == after
    assert before['linear']['selected_index'] == 0
    assert before['mlp']['selected_index'] == 1
    assert before['linear']['candidates'][0]['count'] == end - dynamic.COLD_START
    for consumed, endpoint, stop in ((end + 1, end, end + 1), (end - 1, end, end),
                                     (end - 1, end - 1, end - 1)):
        with pytest.raises(ValueError, match='exact 237568 barrier'):
            dynamic.select_adam_prefix(prediction[:stop], target[:stop], names, consumed,
                                       endpoint, grid)
    assert dynamic.phase_ranges(end + 1)['suffix_all'] == (end, end + 1)
    with pytest.raises(ValueError, match='237568'):
        dynamic.phase_ranges(end + 1, 163840)


@pytest.mark.parametrize('consumed', [0, 2000, dynamic.SELECTION_END, dynamic.SELECTION_END + 3])
def test_partial_pairs_compare_primary_to_every_control_on_observed_labels(consumed):
    samples = dynamic.SELECTION_END + 7
    grid = (.001, .01)
    model_names = (*dynamic.MODEL_NAMES[:4], 'static_test', 'discount_test')
    names = dynamic.output_names(model_names, grid)
    locks = {'linear': {'selected_index': 1}, 'mlp': {'selected_index': 0}}
    columns = dynamic.selected_columns(names, locks, grid)
    assert columns['adam_linear'] == 7
    assert columns['adam_mlp'] == 8
    target = np.linspace(.1, 1.2, samples, dtype=np.float32)
    factors = np.arange(len(names), dtype=np.float32) / 13
    factors[columns['zero']] = 0
    prediction = target[:, None] * factors[None, :]
    target[consumed:] = np.nan
    prediction[consumed:] = np.nan
    phases = dynamic.phase_ranges(samples)
    summary = dynamic.summarize_predictions(target, prediction[:consumed], names, phases, consumed)
    pairs = dynamic.paired_comparisons(summary, names, columns)
    for phase, (start, stop) in phases.items():
        count = max(0, min(stop, consumed) - start)
        row = summary[phase]
        assert row['count'] == count
        assert row['complete'] is (consumed >= stop)
        assert row['expected_count'] == stop - start
        if not count:
            assert row['metrics'] == []
            assert pairs[phase]['comparisons'] == []
            continue
        assert len(row['metrics']) == len(names)
        assert {pair['baseline'] for pair in pairs[phase]['comparisons']} == {
            'static_mixture', 'dynamic_mixture', 'zero', 'adam_linear', 'adam_mlp'}
        yy = target[start:start + count].astype(np.float64)
        chosen = prediction[start:start + count, columns['adaptive_mixture']].astype(np.float64)
        for pair in pairs[phase]['comparisons']:
            baseline = prediction[start:start + count, columns[pair['baseline']]].astype(np.float64)
            difference = np.square(yy - chosen).sum() - np.square(yy - baseline).sum()
            assert pair['mse_difference'] == pytest.approx(difference / count, abs=1e-12)
            assert pair['error_ratio_difference'] == pytest.approx(difference / np.square(yy).sum(), abs=1e-12)


@pytest.mark.parametrize('status,nonfinite', [('pruned', False), ('failed', True)])
def test_partial_artifacts_preserve_raw_forecasts_full_state_and_consumed_counts(tmp_path, status, nonfinite):
    samples, consumed = dynamic.SELECTION_END + 2, 3
    grid = (.001,)
    names = dynamic.output_names(dynamic.MODEL_NAMES[:4], grid)
    predictions = torch.full((samples, len(names)), 12345.)
    predictions[:consumed] = torch.arange(consumed, dtype=torch.float32)[:, None] / 4
    predictions[:consumed, names.index('zero')] = 0
    if nonfinite:
        predictions[1, 0] = float('nan')
    trajectory = torch.zeros(samples, 1)
    states = [torch.tensor([float(i)]) for i in range(8)]
    runner = SimpleNamespace(predictions=predictions, index=torch.tensor(consumed),
                             model=SimpleNamespace(state_tensors=lambda: states, configs=[]),
                             adam=SimpleNamespace(mutable=[states[0], trajectory],
                                                  mlp=SimpleNamespace(predictions=trajectory), configs=grid))
    target = np.ones(samples, dtype=np.float32)
    result = {'status': status, 'adam_grid': grid, 'adam_locks': None, 'exit_code': 75}
    dynamic.save_artifacts(tmp_path, runner, target, names, dynamic.phase_ranges(samples), consumed,
                           ProxyCull(1, {'error_ratio': 1e-4}), result)
    np.testing.assert_equal(np.load(tmp_path / 'predictions.npy'), predictions[:consumed].numpy())
    np.testing.assert_equal(np.load(tmp_path / 'targets.npy'), target[:consumed])
    saved = json.loads((tmp_path / 'results.json').read_text())
    assert saved['status'] == status
    assert saved['processed'] == saved['processed_observations'] == consumed
    assert saved['nonfinite_predictions'] == int(nonfinite)
    assert saved['phase_metrics']['suffix_all']['count'] == 0
    assert saved['paired_comp']['prefix_all']['count'] == consumed
    assert len(saved['paired_comp']['prefix_all']['comparisons']) == 3
    if nonfinite:
        assert saved['phase_metrics']['prefix_all']['metrics'][0]['error_ratio'] is None
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    assert checkpoint['step'] == consumed
    for actual, expected in zip(checkpoint['model_state'], states, strict=True):
        torch.testing.assert_close(actual, expected)
    assert len(checkpoint['adam_state']) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; run via mlq')
@torch.no_grad()
def test_dynamic_stock_capture_restores_both_adam_banks_and_is_predict_before_label(monkeypatch):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(19, 5, device='cuda', generator=generator)
    ys = torch.randn(19, device='cuda', generator=generator)
    model = DynamicNIG(xs.shape[1], 'cuda')
    adam = dynamic.AdamBank(xs, ys, (.001, .003))
    runner = dynamic.StockRunner(xs, ys, model, adam)
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
    assert all(int(clock) == len(xs) for clock in
               (runner.index, model.observations, adam.linear.index, adam.linear.steps,
                adam.mlp.index, adam.mlp.steps))
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
        width = len(model.output_names)
        for columns in (slice(0, 3), slice(4, width), slice(width, width + 2), slice(width + 2, width + 4)):
            assert not torch.equal(next_negative[columns], next_positive[columns])
    finally:
        ys[4].copy_(original)
