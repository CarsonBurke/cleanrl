"""Host provenance/risk contracts and CUDA runner contracts; parent queues execution."""

import copy
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from cleanrl.plasticity import predictive_hazard_eval_v6 as v6
from cleanrl.plasticity import predictive_segment_eval_v5 as v5
from cleanrl.shared import runtime


@pytest.fixture
def original_lock(tmp_path):
    scores = [float(i + 1) for i in range(72)]
    scores[17] = .5
    lock = {'version': 'segment_v5_lock', 'phase': 'development', 'view': 'stationary',
            'locked_at': 15000, 'families': v5.family_contract(), 'source_sha256': v5.source_hashes(),
            'selected_adamw': v5.adam_grid()[17], 'prefix_noisy_mse': scores,
            'feature_sha256': 'paired-features', 'noise_sha256': 'paired-noise',
            'prefix_target_sha256': 'original-prefix'}
    path = tmp_path / 'prefix_lock.json'
    path.write_text(json.dumps(lock))
    return path, lock


def test_original_v5_prefix_selection_transfers_without_reselection_to_all_views(original_lock):
    path, original = original_lock
    for phase, namespace in (('development', 0), ('confirmation', 200)):
        for view in ('stationary', 'change', 'null', 'recurrent', 'two_support'):
            lock, digest, actual_namespace = v6.load_protocol(v6.Args(lock=str(path), phase=phase, view=view))
            assert lock == original
            assert lock['selected_adamw'] == v5.adam_grid()[17]
            assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual_namespace == namespace
    assert v6.family_contract()['namespaces']['confirmation'] != v5.family_contract()['namespaces']['confirmation']


@pytest.mark.parametrize('mutation', ['confirmation', 'change', 'suffix', 'source', 'family', 'selection', 'nonfinite'])
def test_import_rejects_nonprefix_or_changed_original_contract_before_data_draws(original_lock, monkeypatch, mutation):
    path, original = original_lock
    lock = copy.deepcopy(original)
    if mutation == 'confirmation':
        lock['phase'] = 'confirmation'
    elif mutation == 'change':
        lock['view'] = 'change'
    elif mutation == 'suffix':
        lock['locked_at'] = 60000
    elif mutation == 'source':
        lock['source_sha256']['evaluator'] = 'changed'
    elif mutation == 'family':
        lock['families']['namespaces']['confirmation'] = 200
    elif mutation == 'selection':
        lock['selected_adamw'] = v5.adam_grid()[0]
    else:
        lock['prefix_noisy_mse'][0] = float('inf')
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(v6.tyro, 'cli', lambda cls: v6.Args(lock=str(path)))
    def forbidden(*args, **kwargs):
        pytest.fail('invalid imported lock reached runtime or data generation')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(v5.sparse, 'draw_stream', forbidden)
    with pytest.raises(ValueError):
        v6.main()


@pytest.mark.parametrize('view,step,active,stale', [
    ('stationary', 60000, (0,), None), ('null', 60000, (), None),
    ('two_support', 60000, (0, 1), None), ('change', 29999, (0,), None),
    ('change', 30000, (1,), 0), ('change', 60000, (1,), 0),
    ('recurrent', 19999, (0,), None), ('recurrent', 20000, (1,), 0),
    ('recurrent', 39999, (1,), 0), ('recurrent', 40000, (0,), 1),
    ('recurrent', 60000, (0,), 1),
])
def test_exact_next_risk_matches_enumeration_and_signed_stale_partition(view, step, active, stale):
    probability = .19
    weights = torch.tensor([[1.4, .7, -.9, .1]], dtype=torch.float64)
    support = torch.tensor([float(i in active) for i in range(4)], dtype=torch.float64)
    inputs = torch.tensor(list(itertools.product((0., 1.), repeat=4)), dtype=torch.float64)
    masses = (probability ** inputs.sum(-1)) * ((1 - probability) ** (4 - inputs.sum(-1)))
    risk = v6.next_forecast_risk(weights, view, step, probability)
    expected = (((inputs @ (weights[0] - support)).square()) * masses).sum()
    torch.testing.assert_close(risk['clean_mse'][0], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(risk['clean_mse'], risk['signal_reconstruction_mse']
                               + risk['distractor_leakage_mse'] + risk['signal_distractor_cross'])
    torch.testing.assert_close(risk['distractor_leakage_mse'], risk['stale_support_leakage_mse']
                               + risk['remaining_distractor_leakage_mse'] + risk['stale_remaining_cross'])
    if stale is None:
        assert float(risk['stale_support_leakage_mse']) == 0.
        assert float(risk['stale_remaining_cross']) == 0.
    else:
        torch.testing.assert_close(risk['stale_support_leakage_mse'], probability * weights[:, stale].square())
        assert float(risk['stale_remaining_cross']) < 0.


def test_partial_recurrent_sums_stop_at_consumed_bound_not_unseen_suffix():
    count = 40002
    predictions = np.zeros((count, 2), dtype=np.float32)
    predictions[:20000] = (1., 2.)
    predictions[20000:40000] = (3., -1.)
    predictions[40000:] = (-2., 4.)
    clean = np.zeros(count, dtype=np.float32)
    noisy = np.ones(count, dtype=np.float32)
    summary, noisy_error, clean_error = v6.error_summary(predictions, noisy, clean, 'recurrent')
    assert summary['consumed_interval'] == [0, count]
    assert [(row['start'], row['end'], row['count']) for row in summary['phase_errors']] == [
        (0, 20000, 20000), (20000, 40000, 20000), (40000, 40002, 2)]
    post = summary['post_change_errors']
    assert (post['start'], post['end'], post['count']) == (20000, count, 20002)
    np.testing.assert_array_equal(post['clean_squared_error_sum'], [180008., 20032.])
    np.testing.assert_array_equal(post['noisy_squared_error_sum'], [80018., 80018.])
    np.testing.assert_allclose(post['clean_mse'], np.array([180008., 20032.]) / 20002)
    np.testing.assert_allclose(summary['full_clean_mse'], clean_error.mean(0))
    np.testing.assert_allclose(summary['full_noisy_mse'], noisy_error.mean(0))


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def runner_fixture(steps=7):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    cfg = v5.sparse.Args(steps=steps, input_dim=4, graph_steps=3)
    xs = torch.tensor([[True, i % 2 == 0, i % 3 == 0, i % 5 == 0] for i in range(steps)], device='cuda')
    ys = torch.tensor([1., -.5, 2., .25, -3., .4, 1.7][:steps], device='cuda')
    model = v6.ComparisonModel(4, 'cuda')
    return v6.TracedRunner(xs, ys, model, v5.AdamWBank(v5.adam_grid()[17:18], cfg, xs, ys))


@cuda_only
@torch.no_grad()
def test_new_runner_graph_restores_whole_state_and_aligns_all_trace_rows(tmp_path):
    runner = runner_fixture()
    control = runner_fixture()
    initial = [tensor.clone() for tensor in runner.mutable]
    graphs = runner.capture(3)
    for actual, original in zip(runner.mutable, initial):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    compiled = torch.compile(control.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    for row in range(len(runner.xs)):
        prior = runner.model.mixture.log_hazard_weights.exp().clone()
        expected_prediction = runner.model.mean_weights() @ runner.xs[row].float()
        graphs[1].replay()
        compiled()
        torch.cuda.synchronize()
        torch.testing.assert_close(runner.predictions[row, :9], expected_prediction, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(runner.hazard_weights_before[row], prior, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(runner.hazard_weights_after[row], runner.model.mixture.log_hazard_weights.exp())
        for trace, child in zip(runner.traces, runner.model.segments):
            torch.testing.assert_close(trace[row], child.trace(), rtol=3e-5, atol=2e-6)
        densities = torch.stack([child.predictive_log_prob for child in runner.model.segments])
        torch.testing.assert_close(runner.predictive_log_prob[row], densities, rtol=0, atol=0)
        torch.testing.assert_close(runner.discards[row], torch.stack([s.discarded_mass for s in runner.model.segments]))
        expected_posterior = torch.softmax(prior.log() + densities, dim=0)
        torch.testing.assert_close(runner.hazard_weights_after[row], expected_posterior, rtol=3e-5, atol=2e-6)
    assert int(runner.index) == len(runner.xs)
    for actual, expected in zip(runner.mutable, control.mutable):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    completed = [tensor.clone() for tensor in runner.mutable]
    for actual, original in zip(runner.mutable, initial):
        actual.copy_(original)
    graphs[3].replay()
    graphs[3].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    for actual, expected in zip(runner.mutable, completed):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    v6.save_analysis_state(tmp_path, runner, len(runner.xs))
    with np.load(tmp_path / 'branch_traces.npz') as archive:
        for name, child in zip(runner.model.child_names, runner.model.segments):
            np.testing.assert_allclose(archive[name][-1], child.trace().cpu().numpy(), rtol=3e-5, atol=2e-6)
        assert 'hazard_mixture' not in archive.files
        np.testing.assert_array_equal(archive['predictive_log_prob'], runner.predictive_log_prob.cpu().numpy())


@cuda_only
@torch.no_grad()
def test_comparison_forecasts_are_prelabel_and_diagnostics_do_not_mutate_state():
    left, right = v6.ComparisonModel(4, 'cuda'), v6.ComparisonModel(4, 'cuda')
    x = torch.tensor([1., 0., 1., 0.], device='cuda')
    left.update(x, torch.tensor(2., device='cuda'))
    for target, source in zip(right.state_tensors(), left.state_tensors()):
        target.copy_(source)
    saved = [tensor.clone() for tensor in left.state_tensors()]
    forecast = left.mean_weights() @ x
    left.diagnostics()
    left.mean_weights()
    for actual, original in zip(left.state_tensors(), saved):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    a = left.update(x, torch.tensor(-4., device='cuda'))
    b = right.update(x, torch.tensor(6., device='cuda'))
    torch.testing.assert_close(a, forecast, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(left.mean_weights(), right.mean_weights())
