"""Leakage, observed-horizon, provenance, and frozen-transfer contracts.

Short CUDA streams exercise recurrence and capture boundaries, not performance.
"""

import json

import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import network_bayes_stream_v2 as bayes
from cleanrl.plasticity import predictive_stock_transfer_eval_v1 as transfer
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.plasticity.predictive_persistent_support_v4 import PersistentSupport
from cleanrl.shared import runtime


@pytest.mark.parametrize('consumed', [0, 3, 6, 10, 20, 24])
def test_summaries_use_only_observed_rows_even_with_poisoned_unwritten_tail(consumed):
    phases = stock.phase_ranges(24, 2)
    names = ('posterior', 'adam')
    target = np.linspace(-1.3, 2.1, 24, dtype=np.float32)
    predictions = np.column_stack((target * .4, target * -.2))
    target[consumed:] = np.nan
    predictions[consumed:] = np.nan

    result = transfer.summarize_predictions(target, predictions, names, phases, consumed)
    truncated = transfer.summarize_predictions(
        target, predictions[:consumed], names, phases, consumed)
    assert result == truncated
    assert set(result) == set(phases)
    for phase, (start, stop) in phases.items():
        row = result[phase]
        count = max(0, min(stop, consumed) - start)
        assert row['count'] == count
        assert row['expected_count'] == stop - start
        assert row['complete'] is (consumed >= stop)
        if count == 0:
            assert row['metrics'] == []
        else:
            expected = stock.metric_sums(target[start:start + count], predictions[start:start + count])
            assert len(row['metrics']) == len(names)
            for actual, metrics in zip(row['metrics'], expected):
                for key, value in metrics.items():
                    assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('case', ['negative', 'past_target', 'past_predictions', 'reversed_phase', 'negative_phase'])
def test_summaries_reject_invalid_observation_or_phase_bounds(case):
    target = np.arange(1, 9, dtype=np.float32)
    predictions = np.zeros((8, 1), dtype=np.float32)
    consumed = 8
    phases = {'suffix': (2, 8)}
    if case == 'negative':
        consumed = -1
    elif case == 'past_target':
        target = target[:-1]
    elif case == 'past_predictions':
        predictions = predictions[:-1]
    elif case == 'reversed_phase':
        phases['suffix'] = (8, 2)
    else:
        phases['suffix'] = (-1, 8)
    with pytest.raises(ValueError):
        transfer.summarize_predictions(target, predictions, ('posterior',), phases, consumed)


@pytest.fixture
def real_result():
    samples = 12000
    phases = stock.phase_ranges(samples, 2000)
    prefix = phases['prefix_all'][1]
    grid = (.001, .003)
    target = np.linspace(-1.25, 2.5, prefix, dtype=np.float32)
    linear_predictions = np.column_stack((target * .25, target * .75))
    mlp_predictions = np.column_stack((target * .8, target * -.4))
    fingerprint = {'features': 'a' * 64, 'target': 'b' * 64}
    result = {
        'args': {'view': 'real', 'seed': 1},
        'data_sha256': fingerprint.copy(),
        'samples': samples,
        'maximum_observations': samples,
        'processed_observations': samples,
        'status': 'completed',
        'phases': {name: list(bounds) for name, bounds in phases.items()},
        'adam_grid': list(grid),
        'adam_locks': {
            'linear': stock.select_prefix(linear_predictions, target, grid, 2000, prefix, prefix),
            'mlp': stock.select_prefix(mlp_predictions, target, grid, 2000, prefix, prefix),
        },
    }
    return result, fingerprint, grid


def write_result(tmp_path, result):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('status,consumed', [('completed', 12000), ('pruned', 3000), ('pruned', 6500)])
def test_completed_and_pruned_real_prefix_locks_transfer_without_reselection(tmp_path, real_result, status, consumed):
    result, fingerprint, grid = real_result
    result['status'] = status
    result['processed_observations'] = consumed
    locks = transfer.load_real_locks(write_result(tmp_path, result), fingerprint, grid)
    assert locks == result['adam_locks']
    # The original prefix selects different candidates for the two model classes.
    assert locks['linear']['selected_index'] == 1
    assert locks['mlp']['selected_index'] == 0


@pytest.mark.parametrize('field', ['features', 'target'])
def test_transfer_rejects_mismatched_real_data_fingerprint(tmp_path, real_result, field):
    result, fingerprint, grid = real_result
    fingerprint = {**fingerprint, field: 'c' * 64}
    with pytest.raises(ValueError):
        transfer.load_real_locks(write_result(tmp_path, result), fingerprint, grid)


@pytest.mark.parametrize('case', [
    'null_source', 'wrong_seed', 'missing_mlp', 'unconsumed_prefix', 'past_file',
    'wrong_quarter', 'suffix_used', 'late_lock', 'cold_start_included',
    'wrong_winner', 'wrong_lr', 'inconsistent_error', 'wrong_candidate_count',
    'nonfinite_candidate', 'wrong_grid', 'incomplete_completion', 'different_mlp_targets',
])
def test_transfer_rejects_malformed_or_noncausal_real_locks(tmp_path, real_result, case):
    result, fingerprint, grid = real_result
    lock = result['adam_locks']['linear']
    if case == 'null_source':
        result['args']['view'] = 'random_sign'
    elif case == 'wrong_seed':
        result['args']['seed'] = 2
    elif case == 'missing_mlp':
        del result['adam_locks']['mlp']
    elif case == 'unconsumed_prefix':
        result['processed_observations'] = 2999
        result['status'] = 'pruned'
    elif case == 'past_file':
        result['processed_observations'] = result['samples'] + 1
    elif case == 'wrong_quarter':
        result['phases'] = stock.phase_ranges(16000, 2000)
    elif case == 'suffix_used':
        lock['suffix_observations_used'] = 1
    elif case == 'late_lock':
        lock['optimizer_updates_at_lock'] += 1
    elif case == 'cold_start_included':
        lock['selection_start_inclusive'] = 0
    elif case == 'wrong_winner':
        lock['selected_index'] = 0
        lock['selected_lr'] = grid[0]
    elif case == 'wrong_lr':
        lock['selected_lr'] = grid[0]
    elif case == 'inconsistent_error':
        lock['candidates'][0]['error_squared_sum'] *= 2
    elif case == 'wrong_candidate_count':
        lock['candidates'][0]['count'] += 1
    elif case == 'nonfinite_candidate':
        lock['candidates'][0]['error_ratio'] = None
        lock['candidates'][0]['error_squared_sum'] = None
    elif case == 'incomplete_completion':
        result['processed_observations'] = 6500
    elif case == 'different_mlp_targets':
        prefix = result['samples'] // 4
        different_target = 2 * np.linspace(-1.25, 2.5, prefix, dtype=np.float32)
        predictions = np.column_stack((different_target * .8, different_target * -.4))
        result['adam_locks']['mlp'] = stock.select_prefix(
            predictions, different_target, grid, 2000, prefix, prefix)
    else:
        result['adam_grid'] = list(reversed(grid))
    with pytest.raises(ValueError):
        transfer.load_real_locks(write_result(tmp_path, result), fingerprint, grid)


@pytest.fixture
def cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(11, 5, device='cuda', generator=generator)
    ys = torch.randn(11, device='cuda', generator=generator)
    return xs, ys


def compile_update(update):
    return torch.compile(update, fullgraph=True, mode='max-autotune-no-cudagraphs')


def snapshot(states):
    return [tensor.clone() for tensor in states]


def restore(states, saved):
    assert len(states) == len(saved)
    for tensor, value in zip(states, saved):
        tensor.copy_(value)


def assert_state(states, expected, *, rtol=0, atol=0):
    assert len(states) == len(expected)
    for tensor, value in zip(states, expected):
        torch.testing.assert_close(tensor, value, rtol=rtol, atol=atol)


def observable_state(runner):
    # Independently enumerate frozen sublearner state, not only Runner.mutable:
    # otherwise a missing optimizer moment or schedule clock could evade reset checks.
    return [runner.index, runner.predictions, runner.model.observations,
            *runner.model.v3.state_tensors(), *runner.model.v4.state_tensors(),
            *runner.adam.linear.mutable, *runner.adam.mlp.mutable, runner.adam.prediction]


@torch.no_grad()
def test_current_label_changes_learning_not_any_current_forecast(cuda_stream):
    xs, ys = cuda_stream
    model = transfer.TransferModel(xs.shape[1], 'cuda')
    runner = transfer.StockRunner(xs, ys, model, transfer.AdamBank(xs, ys, (.001, .003)))
    update = compile_update(runner.update)
    for _ in range(4):
        update()
    states = observable_state(runner)
    before = snapshot(states)
    original_label = ys[4].clone()
    try:
        ys[4] = -4.
        update()
        negative_forecast = runner.predictions[4].clone()
        update()
        negative_next_forecast = runner.predictions[5].clone()
        restore(states, before)
        assert_state(states, before)
        ys[4] = 4.
        update()  # Exactly the same compiled callable and restored prelabel state.
        positive_forecast = runner.predictions[4].clone()
        update()
        positive_next_forecast = runner.predictions[5].clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(positive_forecast, negative_forecast, rtol=0, atol=0)
        # Prevent a no-update implementation from satisfying the causal assertion.
        for columns in (slice(0, 4), slice(4, 6), slice(6, 8)):
            assert not torch.allclose(positive_next_forecast[columns], negative_next_forecast[columns])
    finally:
        ys[4].copy_(original_label)


@torch.no_grad()
def test_capture_restores_complete_state_and_matches_same_compiled_recurrence(cuda_stream, monkeypatch):
    xs, ys = cuda_stream
    model = transfer.TransferModel(xs.shape[1], 'cuda')
    # Advance the schedule clocks to cross the first frozen birth in a short
    # recurrence contract; this is not a shortened stock-performance experiment.
    model.v4.observations.fill_(4092)
    model.observations.fill_(4092)
    runner = transfer.StockRunner(xs, ys, model, transfer.AdamBank(xs, ys, (.001, .003)))
    compiled_calls = []
    real_compile = torch.compile

    def recording_compile(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        compiled_calls.append(compiled)
        return compiled

    monkeypatch.setattr(torch, 'compile', recording_compile)
    states = observable_state(runner)
    initial = snapshot(states)
    graphs = runner.capture(4)
    assert_state(states, initial)
    # Use the very callable captured by Runner, not a separately compiled twin.
    update = compiled_calls[-1]
    graphs[4].replay()
    graphs[1].replay()
    checkpoint = snapshot(states)
    graphs[4].replay()
    graphs[1].replay()
    graphs[1].replay()
    graph_final = snapshot(states)
    restore(states, checkpoint)
    for _ in range(6):
        update()
    torch.cuda.synchronize()
    assert_state(states, graph_final)
    assert int(runner.index) == int(runner.adam.linear.index) == int(runner.adam.mlp.index) == len(xs)
    assert int(model.observations) == int(model.v4.observations) == 4092 + len(xs)
    # Complete Runner.mutable restore must also recover independently enumerated
    # frozen state after capture/replay, including both Adam accumulators.
    restore(states, initial)
    runner_checkpoint = snapshot(runner.mutable)
    for _ in range(3):
        update()
    restore(runner.mutable, runner_checkpoint)
    assert_state(states, initial)


@torch.no_grad()
def test_adam_bank_matches_independent_frozen_linear_and_mlp_forecasts(cuda_stream):
    xs, ys = cuda_stream
    grid = (.001, .003)
    bank = transfer.AdamBank(xs, ys, grid)
    linear_args = sparse.Args(seed=1, input_dim=xs.shape[1], steps=len(xs))
    linear = sparse.LinearLearner('adam', grid, linear_args, xs, ys)
    mlp_args = bayes.Args(seed=1, input_dim=xs.shape[1], hidden=64)
    initial = bayes.init_weights(mlp_args, torch.Generator(device='cuda').manual_seed(1), xs.device)
    mlp = stock.MeasuredLearner('adam', grid, initial, mlp_args, xs, ys)
    bank_update = compile_update(bank.update)
    linear_update = compile_update(linear.update)
    mlp_update = compile_update(mlp.update)
    actual = []
    expected = []
    for step in range(len(xs)):
        bank_update()
        linear_update()
        mlp_update()
        actual.append(bank.prediction.clone())
        expected.append(torch.cat((linear.prediction, mlp.predictions[step])).clone())
    actual = torch.stack(actual)
    expected = torch.stack(expected)
    # Independent compilation can reorder FP32 reductions through the MLP's
    # 64-wide layers; these bounds match existing frozen-recurrence contracts.
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual[0, :len(grid)], torch.zeros(len(grid), device='cuda'), rtol=0, atol=0)
    assert not torch.allclose(actual[:, :len(grid)], actual[:, len(grid):])
    assert not torch.allclose(actual[1:, 0], actual[1:, 1])
    assert not torch.allclose(actual[1:, 2], actual[1:, 3])
    assert_state(bank.linear.mutable, linear.mutable, rtol=3e-4, atol=3e-6)
    assert_state(bank.mlp.mutable, mlp.mutable, rtol=3e-4, atol=3e-6)


@torch.no_grad()
def test_transfer_preserves_frozen_v3_v4_forecast_and_state_trajectories(cuda_stream):
    xs, ys = cuda_stream
    model = transfer.TransferModel(xs.shape[1], 'cuda')
    v3 = PredictiveMeanRisk(xs.shape[1], 'cuda')
    v4 = PersistentSupport(xs.shape[1], 'cuda')
    model.v4.observations.fill_(4092)
    model.observations.fill_(4092)
    v4.observations.fill_(4092)
    update, update_v3, update_v4 = map(compile_update, (model.update, v3.update, v4.update))
    assert tuple(model.output_names) == (
        'v3_likelihood_static', 'v3_likelihood_switching',
        'v4_likelihood_birth_only', 'v4_likelihood_switching')
    actual = []
    expected = []
    for x, y in zip(xs, ys):
        actual.append(update(x, y).clone())
        expected.append(torch.cat((update_v3(x, y)[:2], update_v4(x, y)[:2])).clone())
    # Separate fused graphs can change FP32 reduction order, not priors, rows,
    # causal statistics, or scheduled cohort retirement.
    torch.testing.assert_close(torch.stack(actual), torch.stack(expected), rtol=3e-4, atol=3e-6)
    assert_state(model.v3.state_tensors(), v3.state_tensors(), rtol=3e-4, atol=3e-6)
    assert_state(model.v4.state_tensors(), v4.state_tensors(), rtol=3e-4, atol=3e-6)
