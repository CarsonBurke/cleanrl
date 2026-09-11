"""Host-only controls: causal selection, paired streams, and safe rejection."""
import hashlib
import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import torch

from cleanrl.plasticity import predictive_mean_risk_conjugate_eval_v3 as evaluator


@pytest.fixture
def provenance():
    return {
        'args': {'view': 'stationary', 'seed': 1},
        'stream': asdict(evaluator.sparse.Args()),
        'processed_observations': 60000, 'feature_sha256': 'a' * 64,
        'adam_grid': [0.0001, 0.001],
        'adam_prefix_lock': {
            'selected_index': 1, 'selected_lr': 0.001,
            'selection_start_inclusive': 0, 'selection_end_exclusive': 15000,
            'optimizer_updates_at_lock': 15000, 'suffix_observations_used': 0,
            'criterion': 'minimum prequential squared error / zero-predictor squared error',
            'tie_break': 'first (smallest) learning rate',
            'candidates': [
                {'lr': .0001, 'count': 15000, 'target_squared_sum': 15000.,
                 'error_squared_sum': 15000., 'error_ratio': 1.},
                {'lr': .001, 'count': 15000, 'target_squared_sum': 15000.,
                 'error_squared_sum': 13500., 'error_ratio': .9}],
        },
    }


def save_lock(tmp_path, source):
    path = tmp_path / 'stationary.json'
    path.write_text(json.dumps(source))
    return path


def load(tmp_path, source):
    return evaluator.load_stationary_lock(save_lock(tmp_path, source), evaluator.sparse.Args())


def test_transfers_prefix_winner_with_auditable_source_identity(tmp_path, provenance):
    path = save_lock(tmp_path, provenance)
    _, grid, digest = evaluator.load_stationary_lock(path, evaluator.sparse.Args())
    assert grid == (.001,)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejects_a_different_scientific_stream(tmp_path, provenance):
    provenance['stream']['noise_sigma'] = .5
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


@pytest.mark.parametrize('field,value', [('selection_end_exclusive', 30000), ('suffix_observations_used', 1)])
def test_rejects_control_selection_using_future_labels(tmp_path, provenance, field, value):
    provenance['adam_prefix_lock'][field] = value
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


def test_control_requires_a_consumed_stationary_prefix(tmp_path, provenance):
    provenance['processed_observations'] = 14999
    with pytest.raises(ValueError):
        load(tmp_path, provenance)
    provenance['processed_observations'] = 60000
    provenance['args']['view'] = 'null'
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


def test_rejects_a_rate_not_in_the_recorded_grid(tmp_path, provenance):
    provenance['adam_prefix_lock']['selected_lr'] = .003
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


def test_rejects_falsified_prefix_scores_and_nonwinning_selection(tmp_path, provenance):
    choice = provenance['adam_prefix_lock']
    choice['candidates'][1]['error_ratio'] = .5
    with pytest.raises(ValueError):
        load(tmp_path, provenance)
    choice['candidates'][1]['error_ratio'] = .9
    choice.update(selected_index=0, selected_lr=.0001)
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


def test_ties_preserve_the_first_rate(tmp_path, provenance):
    choice = provenance['adam_prefix_lock']
    choice['candidates'][1].update(error_squared_sum=15000., error_ratio=1.)
    with pytest.raises(ValueError):
        load(tmp_path, provenance)
    choice.update(selected_index=0, selected_lr=.0001)
    assert load(tmp_path, provenance)[1] == (.0001,)


def test_nonfinite_loser_is_allowed_but_incomplete_evidence_is_not(tmp_path, provenance):
    candidate = provenance['adam_prefix_lock']['candidates'][0]
    candidate.update(error_squared_sum=None, error_ratio=None)
    assert load(tmp_path, provenance)[1] == (.001,)
    candidate['error_squared_sum'] = 15000.
    with pytest.raises(ValueError):
        load(tmp_path, provenance)


def test_invalid_provenance_persists_failure_without_entering_cuda(tmp_path, provenance, monkeypatch):
    provenance['stream']['noise_sigma'] = .5
    args = evaluator.Args(view='null', adam_lock=str(save_lock(tmp_path, provenance)), output=str(tmp_path))
    monkeypatch.setattr(evaluator.tyro, 'cli', lambda _: args)
    monkeypatch.setattr(evaluator.torch.cuda, 'is_available', Mock(side_effect=AssertionError('unexpected CUDA access')))
    with pytest.raises(ValueError):
        evaluator.main()
    result = json.loads(next(tmp_path.glob('*/results.json')).read_text())
    assert result['status'] == 'failed' and result['processed_observations'] == 0


def test_changed_features_fail_before_any_label_or_model_work(tmp_path, provenance, monkeypatch):
    args = evaluator.Args(view='change', adam_lock=str(save_lock(tmp_path, provenance)), output=str(tmp_path))
    monkeypatch.setattr(evaluator.tyro, 'cli', lambda _: args)
    monkeypatch.setattr(evaluator.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(evaluator.runtime, 'configure_runtime', lambda **_: None)
    monkeypatch.setattr(evaluator, 'SummaryWriter', lambda _: Mock())
    features = torch.zeros((2, 2), dtype=torch.bool)
    monkeypatch.setattr(evaluator.sparse, 'draw_stream', lambda *_: (features, None))
    monkeypatch.setattr(evaluator.sparse, 'teacher_labels', Mock(side_effect=AssertionError('unexpected label work')))
    with pytest.raises(ValueError):
        evaluator.main()
    result = json.loads(next(tmp_path.glob('*/results.json')).read_text())
    assert result['status'] == 'failed' and result['processed_observations'] == 0
