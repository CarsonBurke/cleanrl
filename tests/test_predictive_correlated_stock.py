"""Selection/provenance and observed-only evidence, plus a CUDA recurrence contract.

The CUDA test is an optimizer/capture unit contract, not a shortened stock run;
execute GPU tests only through the shared mlq queue.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import predictive_correlated_stock_eval_v2 as correlated
from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG
from cleanrl.shared import runtime
from cleanrl.shared.autocull import ProxyCull


@pytest.fixture(scope='module')
def real_evidence():
    end = correlated.SELECTION_END
    target = np.linspace(-1.25, 2.5, end, dtype=np.float32)
    priors, grid = correlated.PRIORS, correlated.ADAM_GRID
    fingerprint = {'features': 'a' * 64, 'target': 'b' * 64}
    model_locks = {}
    adam_locks = {}
    for family, winner in (('dense', 3), ('diagonal', 1)):
        predictions = target[:, None] * np.full((1, len(priors)), .2, dtype=np.float32)
        predictions[:, winner] = target * .8
        model_locks[family] = correlated.select_model_prefix(
            predictions, target, priors, family, end, end)
    for family, winner in (('linear', 7), ('mlp', 9)):
        predictions = target[:, None] * np.full((1, len(grid)), .25, dtype=np.float32)
        predictions[:, winner] = target * .75
        adam_locks[family] = stock.select_prefix(predictions, target, grid, correlated.COLD_START, end, end)
    samples = end + 20000
    return {
        'args': {'view': 'real', 'seed': 1, 'selection_end': end},
        'data_sha256': fingerprint, 'samples': samples, 'maximum_observations': samples,
        'processed_observations': samples, 'status': 'completed',
        'phases': {name: list(bounds) for name, bounds in correlated.phase_ranges(samples).items()},
        'model_priors': list(priors), 'adam_grid': list(grid),
        'model_locks': model_locks, 'adam_locks': adam_locks,
    }


def load_written(tmp_path, source, **kwargs):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(source), encoding='utf-8')
    return correlated.load_real_locks(
        path, kwargs.get('fingerprint', source['data_sha256']),
        kwargs.get('selection_end', correlated.SELECTION_END),
        kwargs.get('priors', correlated.PRIORS), kwargs.get('adam_grid', correlated.ADAM_GRID))


@pytest.mark.parametrize('status,extra', [('running', 0), ('pruned', 10000), ('completed', 20000)])
def test_real_locks_transfer_at_barrier_and_after_censoring(tmp_path, real_evidence, status, extra):
    source = copy.deepcopy(real_evidence)
    source.update(status=status, processed_observations=correlated.SELECTION_END + extra)
    locks = load_written(tmp_path, source)
    assert locks == {key: source[key] for key in ('model_locks', 'adam_locks')}
    assert locks['model_locks']['dense']['selected_index'] == 3
    assert locks['model_locks']['diagonal']['selected_index'] == 1
    assert locks['adam_locks']['linear']['selected_index'] == 7
    assert locks['adam_locks']['mlp']['selected_index'] == 9


@pytest.mark.parametrize('field', ['features', 'target'])
def test_lock_transfer_requires_identical_real_features_and_targets(tmp_path, real_evidence, field):
    fingerprint = {**real_evidence['data_sha256'], field: 'c' * 64}
    with pytest.raises(ValueError, match='SHA256'):
        load_written(tmp_path, real_evidence, fingerprint=fingerprint)


@pytest.mark.parametrize('case', [
    'wrong_boundary', 'old_quarter', 'prefix_unconsumed', 'suffix_used', 'late_lock',
    'cold_start_included', 'null_source', 'failed_source', 'wrong_seed', 'missing_diagonal',
    'model_grid', 'adam_grid', 'requested_model_grid', 'requested_adam_grid',
    'fabricated_model_winner', 'fabricated_adam_winner', 'wrong_prior', 'wrong_type',
    'nonfinite_candidate', 'inconsistent_error', 'inconsistent_prediction', 'wrong_count',
    'different_targets', 'incomplete_completion', 'past_file',
])
def test_rejects_noncausal_or_fabricated_real_lock_evidence(tmp_path, real_evidence, case):
    source = copy.deepcopy(real_evidence)
    lock = source['model_locks']['dense']
    kwargs = {}
    if case == 'wrong_boundary':
        source['args']['selection_end'] -= 1
        kwargs['selection_end'] = source['args']['selection_end']
    elif case == 'old_quarter':
        source['phases']['prefix_all'][1] = source['samples'] // 4
    elif case == 'prefix_unconsumed':
        source.update(status='pruned', processed_observations=correlated.SELECTION_END - 1)
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
    elif case == 'missing_diagonal':
        del source['model_locks']['diagonal']
    elif case == 'model_grid':
        source['model_priors'][0] *= 2
    elif case == 'adam_grid':
        source['adam_grid'][0] *= 2
    elif case == 'requested_model_grid':
        kwargs['priors'] = correlated.PRIORS[:-1]
    elif case == 'requested_adam_grid':
        kwargs['adam_grid'] = correlated.ADAM_GRID[:-1]
    elif case == 'fabricated_model_winner':
        lock.update(selected_index=0, selected_prior=correlated.PRIORS[0])
    elif case == 'fabricated_adam_winner':
        source['adam_locks']['linear'].update(selected_index=0, selected_lr=correlated.ADAM_GRID[0])
    elif case == 'wrong_prior':
        lock['selected_prior'] = correlated.PRIORS[0]
    elif case == 'wrong_type':
        lock['candidate_type'] = 'diagonal'
    elif case == 'nonfinite_candidate':
        lock['candidates'][0]['error_ratio'] = None
    elif case == 'inconsistent_error':
        lock['candidates'][0]['error_squared_sum'] *= 2
    elif case == 'inconsistent_prediction':
        lock['candidates'][0]['prediction_squared_sum'] *= 2
    elif case == 'wrong_count':
        lock['candidates'][0]['count'] -= 1
    elif case == 'different_targets':
        source['model_locks']['diagonal']['candidates'][0]['target_squared_sum'] *= 2
    elif case == 'incomplete_completion':
        source['processed_observations'] -= 1
    else:
        source['processed_observations'] += 1
    with pytest.raises(ValueError):
        load_written(tmp_path, source, **kwargs)


def test_first_suffix_label_cannot_enter_selection_and_coldstart_cannot_choose_winner():
    end = correlated.SELECTION_END
    priors = correlated.PRIORS[:2]
    target = np.ones(end + 1, dtype=np.float32)
    prediction = np.zeros((end + 1, 2), dtype=np.float32)
    prediction[:correlated.COLD_START, 0] = 1000  # Would reverse the winner if included.
    prediction[correlated.COLD_START:end, 0] = .8
    prediction[correlated.COLD_START:end, 1] = .2
    before = correlated.select_model_prefix(prediction[:end], target[:end], priors, 'dense', end, end)
    prediction[end] = [1e20, -1e20]
    target[end] = 1e20
    after = correlated.select_model_prefix(prediction[:end], target[:end], priors, 'dense', end, end)
    assert before == after
    assert before['selected_index'] == 0
    assert before['candidates'][0]['count'] == end - correlated.COLD_START
    with pytest.raises(ValueError):
        correlated.select_model_prefix(prediction, target, priors, 'dense', end, end + 1)
    with pytest.raises(ValueError):
        correlated.select_model_prefix(prediction[:end], target[:end], priors, 'dense', end, end - 1)
    with pytest.raises(ValueError):
        correlated.select_model_prefix(prediction[:end - 1], target[:end - 1], priors, 'dense', end - 1, end - 1)


@pytest.mark.parametrize('consumed', [0, 2000, correlated.SELECTION_END, correlated.SELECTION_END + 3])
def test_partial_phases_and_dynamic_columns_score_all_selected_pairs(consumed):
    end = correlated.SELECTION_END
    samples = end + 7
    priors, grid = (1e-6, 1e-4, 1e-2), (.001, .01)
    names = (*(f'dense_{p:g}' for p in priors), *(f'diagonal_{p:g}' for p in priors),
             'dense_bayes_mixture', 'zero', *(f'adam_linear_{lr:g}' for lr in grid),
             *(f'adam_mlp_{lr:g}' for lr in grid))
    model_locks = {'dense': {'selected_index': 2}, 'diagonal': {'selected_index': 1}}
    adam_locks = {'linear': {'selected_index': 1}, 'mlp': {'selected_index': 0}}
    columns = correlated.selected_columns(names, model_locks, adam_locks, priors, grid)
    assert columns == {'dense': 2, 'diagonal': 4, 'dense_bayes_mixture': 6,
                       'zero': 7, 'adam_linear': 9, 'adam_mlp': 10}
    target = np.linspace(.1, 1.2, samples, dtype=np.float32)
    factors = np.arange(len(names), dtype=np.float32) / 13
    factors[columns['zero']] = 0
    predictions = target[:, None] * factors[None, :]
    target[consumed:] = np.nan
    predictions[consumed:] = np.nan
    phases = correlated.phase_ranges(samples)
    summary = correlated.summarize_predictions(target, predictions[:consumed], names, phases, consumed)
    pairs = correlated.paired_comparisons(summary, names, columns)
    for phase, (start, stop) in phases.items():
        count = max(0, min(stop, consumed) - start)
        assert summary[phase]['count'] == count
        assert summary[phase]['complete'] is (consumed >= stop)
        assert summary[phase]['expected_count'] == stop - start
        if not count:
            assert summary[phase]['metrics'] == []
            assert pairs[phase]['comparisons'] == []
            continue
        yy = target[start:start + count].astype(np.float64)
        for pair in pairs[phase]['comparisons']:
            chosen = predictions[start:start + count, columns[pair['role']]].astype(np.float64)
            baseline = predictions[start:start + count, columns[pair['baseline']]].astype(np.float64)
            difference = np.square(yy - chosen).sum() - np.square(yy - baseline).sum()
            assert pair['mse_difference'] == pytest.approx(difference / count, abs=1e-12)
            assert pair['error_ratio_difference'] == pytest.approx(difference / np.square(yy).sum(), abs=1e-12)


def test_failure_artifacts_preserve_written_nan_and_exclude_unwritten_tail(tmp_path):
    samples, consumed = correlated.SELECTION_END + 2, 3
    predictions = torch.full((samples, 1), 12345.)
    predictions[:consumed, 0] = torch.tensor([0., float('nan'), .25])
    trajectory = torch.zeros(samples, 1)
    state = torch.tensor([2.])
    runner = SimpleNamespace(predictions=predictions, index=torch.tensor(consumed),
                             model=SimpleNamespace(state_tensors=lambda: [state], configs={'test': True}),
                             adam=SimpleNamespace(mutable=[state, trajectory],
                                                  mlp=SimpleNamespace(predictions=trajectory), configs=(.001,)))
    target = np.ones(samples, dtype=np.float32)
    result = {'status': 'failed'}
    correlated.save_artifacts(tmp_path, runner, target, ('dense',),
                              correlated.phase_ranges(samples), consumed,
                              ProxyCull(2, {'error_ratio': 1e-4}), result)
    stored = np.load(tmp_path / 'predictions.npy')
    np.testing.assert_equal(stored, [[0.], [np.nan], [.25]])
    np.testing.assert_equal(np.load(tmp_path / 'targets.npy'), target[:consumed])
    saved = json.loads((tmp_path / 'results.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['nonfinite_predictions'] == 1
    assert saved['phase_metrics']['suffix_all']['count'] == 0
    assert saved['phase_metrics']['prefix_all']['metrics'][0]['error_ratio'] is None
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    assert checkpoint['step'] == consumed
    torch.testing.assert_close(checkpoint['model_state'][0], state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; run via mlq')
@torch.no_grad()
def test_actual_nig_capture_restores_all_state_and_current_label_cannot_change_forecast(monkeypatch):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(19, 5, device='cuda', generator=generator)
    ys = torch.randn(19, device='cuda', generator=generator)
    model = CorrelatedNIG(xs.shape[1], 'cuda')
    adam = correlated.AdamBank(xs, ys, (.001, .003))
    runner = correlated.StockRunner(xs, ys, model, adam)
    states = [runner.index, runner.predictions, model.dense_mean, model.dense_cov,
              model.diag_mean, model.diag_cov, model.alpha, model.beta,
              model.log_weights, model.observations,
              *adam.linear.mutable, *adam.mlp.mutable, adam.prediction]
    initial = [tensor.clone() for tensor in states]
    real_compile, callables = torch.compile, []

    def recording_compile(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        callables.append(compiled)
        return compiled

    def restore(saved):
        for tensor, value in zip(states, saved):
            tensor.copy_(value)

    def assert_state(expected):
        for tensor, reference in zip(states, expected):
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
        for columns in (slice(0, len(model.priors)), slice(len(model.priors), 2 * len(model.priors)),
                        slice(width, width + 2), slice(width + 2, width + 4)):
            assert not torch.equal(next_negative[columns], next_positive[columns])
    finally:
        ys[4].copy_(original)
