"""Lock/risk/export host contracts and queued CUDA graph/clock contracts."""
import copy
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from cleanrl.plasticity import predictive_structure_eval_v7 as v7
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


def test_original_prefix_lock_transfers_without_reselection(original_lock):
    path, original = original_lock
    for phase, namespace in (('development', 0), ('confirmation', 300)):
        for view in ('stationary', 'change', 'null', 'recurrent', 'two_support'):
            lock, digest, actual = v7.load_protocol(v7.Args(lock=str(path), phase=phase, view=view))
            assert lock == original
            assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == namespace


@pytest.mark.parametrize('mutation', ['confirmation', 'change', 'suffix', 'source', 'family', 'selection', 'nonfinite'])
def test_bad_lock_fails_before_runtime_or_random_draws(original_lock, monkeypatch, mutation):
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
        lock['families']['namespaces']['confirmation'] = 300
    elif mutation == 'selection':
        lock['selected_adamw'] = v5.adam_grid()[0]
    else:
        lock['prefix_noisy_mse'][0] = float('inf')
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(v7.tyro, 'cli', lambda cls: v7.Args(lock=str(path)))
    def forbidden(*args, **kwargs):
        pytest.fail('invalid lock reached runtime or random draws')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(v5.sparse, 'draw_stream', forbidden)
    monkeypatch.setattr(v7, 'draw_resampling', forbidden)
    with pytest.raises(ValueError):
        v7.main()


@pytest.mark.parametrize('view,step,active,stale', [
    ('stationary', 60000, (0,), None), ('null', 60000, (), None),
    ('two_support', 60000, (0, 1), None), ('change', 29999, (0,), None),
    ('change', 30000, (1,), 0), ('change', 60000, (1,), 0),
    ('recurrent', 19999, (0,), None), ('recurrent', 20000, (1,), 0),
    ('recurrent', 39999, (1,), 0), ('recurrent', 40000, (0,), 1),
    ('recurrent', 60000, (0,), 1),
])
def test_next_forecast_risk_matches_exhaustive_bernoulli_inputs(view, step, active, stale):
    probability = .19
    weights = torch.tensor([[1.4, .7, -.9, .1]], dtype=torch.float64)
    support = torch.tensor([float(i in active) for i in range(4)], dtype=torch.float64)
    xs = torch.tensor(list(itertools.product((0., 1.), repeat=4)), dtype=torch.float64)
    masses = probability ** xs.sum(1) * (1 - probability) ** (4 - xs.sum(1))
    risk = v7.next_forecast_risk(weights, view, step, probability)
    torch.testing.assert_close(risk['clean_mse'][0], ((xs @ (weights[0] - support)).square() * masses).sum(),
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(risk['clean_mse'], risk['signal_reconstruction_mse'] +
                               risk['distractor_leakage_mse'] + risk['signal_distractor_cross'])
    torch.testing.assert_close(risk['distractor_leakage_mse'], risk['stale_support_leakage_mse'] +
                               risk['remaining_distractor_leakage_mse'] + risk['stale_remaining_cross'])
    if stale is None:
        assert float(risk['stale_support_leakage_mse']) == 0.
    else:
        torch.testing.assert_close(risk['stale_support_leakage_mse'], probability * weights[:, stale].square())


def test_partial_phase_evidence_never_scores_unconsumed_suffix():
    prediction = np.zeros((40002, 2))
    prediction[:20000] = (1., 2.)
    prediction[20000:40000] = (3., -1.)
    prediction[40000:] = (-2., 4.)
    summary, noisy, clean = v7.error_summary(prediction, np.ones(40002), np.zeros(40002), 'recurrent')
    assert summary['consumed_interval'] == [0, 40002]
    assert [(r['start'], r['end']) for r in summary['phase_errors']] == [(0, 20000), (20000, 40000), (40000, 40002)]
    np.testing.assert_array_equal(summary['post_change_errors']['clean_squared_error_sum'], [180008., 20032.])
    np.testing.assert_allclose(summary['full_noisy_mse'], noisy.mean(0))
    np.testing.assert_allclose(summary['full_clean_mse'], clean.mean(0))
    with pytest.raises(ValueError):
        v7.error_summary(prediction, np.ones(40003), np.zeros(40002), 'recurrent')


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def runner_fixture():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    cfg = v5.sparse.Args(steps=7, input_dim=4, graph_steps=3)
    xs = torch.tensor([[True, i % 2 == 0, i % 3 == 0, i % 5 == 0] for i in range(7)], device='cuda')
    ys = torch.tensor([1., -.5, 2., .25, -3., .4, 1.7], device='cuda')
    model = v7.ComparisonModel(4, 'cuda', v7.draw_resampling(7, 0, xs.device))
    return v7.TracedRunner(xs, ys, model, v5.AdamWBank(v5.adam_grid()[17:18], cfg, xs, ys))


@cuda_only
@torch.no_grad()
def test_graph_capture_restore_final_trace_and_full_randomness_consumption(tmp_path):
    runner, control = runner_fixture(), runner_fixture()
    initial = [tensor.clone() for tensor in runner.mutable]
    random_hash = v5.tensor_hash(runner.model.mixture.singleton.uniforms)
    graphs = runner.capture(3)
    for actual, saved in zip(runner.mutable, initial):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    compiled = torch.compile(control.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    for row in range(7):
        forecast = runner.model.mean_weights() @ runner.xs[row].double()
        prior = runner.model.mixture.log_structure_weights.exp().clone()
        graphs[1].replay()
        compiled()
        torch.cuda.synchronize()
        torch.testing.assert_close(runner.predictions[row, :len(runner.model.output_names)], forecast, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(runner.structure_weights_before[row], prior, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(runner.structure_weights_after[row],
                                   torch.softmax(prior.log() + runner.structure_log_prob[row], 0), rtol=1e-12, atol=1e-12)
        for trace, segment in zip(runner.traces, runner.model.segments):
            torch.testing.assert_close(trace[row], segment.trace().double(), rtol=3e-5, atol=2e-6)
        assert int(runner.randomness_consumed[row]) == (row + 1) * 128
    assert int(runner.index) == int(runner.adam.index) == int(runner.adam.legacy.index) == 7
    assert all(int(s.observations) == 7 for s in runner.model.segments)
    assert int(runner.model.mixture.singleton.index) == 7
    for actual, expected in zip(runner.mutable, control.mutable):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    completed = [t.clone() for t in runner.mutable]
    for actual, saved in zip(runner.mutable, initial):
        actual.copy_(saved)
    graphs[3].replay()
    graphs[3].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    for actual, expected in zip(runner.mutable, completed):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert v5.tensor_hash(runner.model.mixture.singleton.uniforms) == random_hash
    v7.save_analysis_state(tmp_path, runner, 7)
    with np.load(tmp_path / 'branch_traces.npz') as archive:
        for name, trace in zip(runner.model.child_names, runner.traces):
            np.testing.assert_allclose(archive[name][-1], trace[-1].cpu().numpy(), rtol=0, atol=0)
        np.testing.assert_array_equal(archive['randomness_consumed'], np.arange(1, 8) * 128)
    checkpoint = torch.load(tmp_path / 'checkpoint.pt', weights_only=False)
    assert checkpoint['step'] == 7 and checkpoint['uniforms_consumed'] == 896
    assert checkpoint['resampling_sha256'] == random_hash
    with pytest.raises(ValueError):
        v7.save_analysis_state(tmp_path, runner, 6)
    with pytest.raises(ValueError):
        v7.save_predictions(tmp_path, runner, runner.ys, runner.ys, 8, 'stationary')
    v7.save_predictions(tmp_path, runner, runner.ys, runner.ys, 7, 'stationary')
    with np.load(tmp_path / 'prequential.npz') as archive:
        np.testing.assert_array_equal(archive['predictions'][-1], runner.predictions[-1].cpu().numpy())
        np.testing.assert_array_equal(archive['noisy_target'], runner.ys.cpu().numpy())


@cuda_only
@torch.no_grad()
def test_resampling_is_reproducible_and_does_not_advance_global_generator():
    before = torch.cuda.get_rng_state().clone()
    tape = v7.draw_resampling(7, 0, 'cuda')
    torch.testing.assert_close(torch.cuda.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(tape, v7.draw_resampling(7, 0, 'cuda'), rtol=0, atol=0)
    assert v5.tensor_hash(tape) != v5.tensor_hash(v7.draw_resampling(7, 300, 'cuda'))
    assert bool(((tape >= 0) & (tape < 1)).all())
