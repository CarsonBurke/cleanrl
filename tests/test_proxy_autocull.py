"""Host-only resource policy tests; no learning or CUDA work is launched."""
import json
import subprocess
import sys

import pytest

from cleanrl.shared.autocull import ProxyCull, ProxyPruned, PRUNED_EXIT_CODE, prune_proxy


def test_flat_proxy_stops_only_after_warmup_and_full_patience():
    policy = ProxyCull(1, {'loss': .0001})
    for step in (4096, 8192, 12288, 16384, 20480, 24576):
        assert policy.observe(step, {'loss': [1.]}) is None
    decision = policy.observe(28672, {'loss': [1.]})
    assert decision['status'] == 'pruned'
    assert decision['decisions'][0]['stale_evaluations'] == 3


def test_improvement_in_any_metric_or_candidate_protects_the_run():
    policy = ProxyCull(2, {'loss': .001, 'recovery': .001})
    for i in range(1, 30):
        assert policy.observe(i * 4096, {'loss': [1., 1.], 'recovery': [1., 1 / i]}) is None


def test_stale_candidate_can_recover_while_another_keeps_grid_alive():
    policy = ProxyCull(2, {'loss': .001})
    for i in range(1, 8):
        assert policy.observe(i * 4096, {'loss': [1., 1 / i]}) is None
    assert policy.hooks[0].state_dict()['culled']
    for i in range(8, 17):
        assert policy.observe(i * 4096, {'loss': [1 / i, .2]}) is None
    assert not policy.hooks[0].state_dict()['culled']


def test_new_regime_gets_its_own_warmup_not_old_stale_history():
    policy = ProxyCull(1, {'loss': .001})
    for step in (4096, 8192, 12288, 16384, 20480, 24576):
        assert policy.observe(step, {'loss': [1.]}) is None
    for step in (28672, 32768, 36864, 40960, 45056, 49152):
        assert policy.observe(step, {'loss': [2.]}, phase='moved', phase_start=24576) is None
    assert policy.observe(53248, {'loss': [2.]}, phase='moved', phase_start=24576)


def test_isolated_spike_does_not_stop_a_recovering_curve():
    policy = ProxyCull(1, {'loss': .001})
    for i in range(1, 20):
        value = 3. if i == 6 else 1 / i
        assert policy.observe(i * 4096, {'loss': [value]}) is None


def test_nonfinite_candidate_does_not_hide_an_improving_one():
    policy = ProxyCull(2, {'loss': .001})
    for i in range(1, 12):
        assert policy.observe(i * 4096, {'loss': [float('nan'), 1 / i]}) is None
    dead = ProxyCull(2, {'loss': .001}).observe(1, {'loss': [float('inf'), float('nan')]})
    assert 'nonfinite' in dead['reason']
    json.dumps(dead, allow_nan=False)


def test_duplicate_checkpoints_and_incomplete_metrics_cannot_advance_patience():
    policy = ProxyCull(1, {'loss': .001})
    policy.observe(4096, {'loss': [1.]})
    with pytest.raises(ValueError):
        policy.observe(4096, {'loss': [1.]})
    with pytest.raises(ValueError):
        policy.observe(8192, {'loss': []})
    assert policy.state_dict()['last_step'] == 4096


def test_saved_policy_history_is_a_detached_snapshot():
    policy = ProxyCull(1, {'loss': .001})
    policy.observe(4096, {'loss': [1.]})
    snapshot = policy.state_dict()
    original = json.dumps(snapshot, sort_keys=True)
    policy.observe(8192, {'loss': [.5]})
    policy.observe(12288, {'loss': [float('nan')]})
    assert json.dumps(snapshot, sort_keys=True) == original


def test_prune_decision_is_durable_before_exception(tmp_path, capsys):
    record = {'status': 'pruned', 'reason': 'plateau', 'step': 28672}
    with pytest.raises(ProxyPruned) as caught:
        prune_proxy(tmp_path, 'trial', record)
    saved = json.loads((tmp_path / 'autocull.json').read_text())
    assert saved == caught.value.record
    assert saved['arm'] == 'trial'
    assert capsys.readouterr().out.startswith('AUTOCULL ')


def test_intentional_prune_exits_nonzero_before_following_work(tmp_path):
    # Exercise an actual process exit using the same guard/exception contract.
    code = '''
import sys
from pathlib import Path
from cleanrl.shared.autocull import ProxyCull, ProxyPruned, prune_proxy, PRUNED_EXIT_CODE
root = Path(sys.argv[1])
try:
    policy = ProxyCull(1, {'loss': .001})
    for step in range(4096, 100001, 4096):
        decision = policy.observe(step, {'loss': [1.]})
        if decision:
            prune_proxy(root, 'flat', decision)
    (root / 'following_arm').touch()
except ProxyPruned:
    raise SystemExit(PRUNED_EXIT_CODE)
finally:
    (root / 'writer_closed').touch()
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == PRUNED_EXIT_CODE != 0
    assert (tmp_path / 'writer_closed').exists()
    assert not (tmp_path / 'following_arm').exists()
    assert json.loads((tmp_path / 'autocull.json').read_text())['step'] == 28672


def test_stock_boundary_saves_only_consumed_predictions_and_unwinds(tmp_path, monkeypatch):
    # Control-flow fixture, not a learner or a reduced training experiment.
    import numpy as np
    import torch
    from cleanrl.plasticity import covariance_stock_eval_v1 as stock

    class FixtureLearner:
        def __init__(self, method, grid, initial, a, xs, ys):
            self.capture_steps = 1
            self.index = torch.zeros((), dtype=torch.int64)
            self.steps = torch.zeros(())
            self.predictions = torch.full((len(xs), len(grid)), 2.)
            self.weights = [torch.zeros(1)]
            self.noise = torch.ones(1)
        def capture(self):
            return self, 0.
        def replay(self):
            self.index.add_(1)
            self.steps.add_(1)

    class Writer:
        def add_scalar(self, *args):
            pass
        def flush(self):
            pass

    monkeypatch.setattr(stock, 'MeasuredLearner', FixtureLearner)
    for name in ('empty_cache', 'reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    args = stock.Args(log_every=8192)
    result = {'arms': {}}
    n = 65536
    with pytest.raises(ProxyPruned):
        stock.run_arm('covariance_original_random_sign', 'network', (1.,), 1e-5, [],
                      torch.zeros(n, 1), np.ones(n), args, stock.phase_ranges(n, args.cold_start),
                      tmp_path, Writer(), result, np.array([0, n-1]))
    saved = json.loads((tmp_path / 'results.json').read_text())
    arm = saved['arms']['covariance_original_random_sign']
    assert saved['status'] == arm['status'] == 'pruned'
    consumed = arm['optimizer_updates_per_candidate']
    assert consumed == 40960 < n
    assert np.load(tmp_path / arm['prediction_artifact']).shape == (consumed, 1)
    assert arm['phase_metrics']['suffix_all'][0]['count'] == consumed - n // 4


@pytest.mark.parametrize('arm,candidates', [('adam_real_grid', 2), ('covariance_original_planted', 1)])
def test_replay_preserves_candidate_grid_and_scheduled_phase(tmp_path, monkeypatch, arm, candidates):
    from torch.utils.tensorboard import SummaryWriter
    from scripts import replay_proxy_autocull as replay

    (tmp_path / 'results.json').write_text(json.dumps({
        'arms': {arm: {'grid': list(range(candidates))}},
        'phases': {'suffix_positive': [16384, 32768]},
    }))
    with SummaryWriter(str(tmp_path)) as writer:
        for i, step in enumerate(range(8192, 65537, 8192), 1):
            for candidate in range(candidates):
                loss = 1. if candidate == 0 else 1 / i
                writer.add_scalar(f'{arm}/candidate_{candidate}/error_ratio', loss, step)
                writer.add_scalar(f'{arm}/candidate_{candidate}/prediction_energy_ratio', 1., step)
    report = tmp_path / 'replay.json'
    monkeypatch.setattr(sys, 'argv', ['replay', str(tmp_path), '--arm', arm, '--output', str(report)])
    replay.main()
    assert json.loads(report.read_text())['first_prune'] is None


def test_sparse_entrypoint_persists_pruned_arm_and_stops_serial_job(tmp_path, monkeypatch):
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse

    monkeypatch.setattr(sparse.tyro, 'cli', lambda _: sparse.Args(output=str(tmp_path)))
    monkeypatch.setattr(sparse.runtime, 'configure_runtime', lambda **_: None)
    monkeypatch.setattr(sparse.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(sparse, 'draw_stream', lambda *_: (None, None))

    class Writer:
        def __init__(self, *args):
            pass
        def add_text(self, *args):
            pass
        def close(self):
            (tmp_path / 'writer_closed').touch()

    def stalled_arm(a, method, grid, view, xs, noise, root, writer, **kwargs):
        # Feed a recorded control decision at the arm boundary; no learner.
        key = f'{view}/covariance_q{kwargs["diffusion"]:g}'
        sparse.save_json(root / f'{key.replace("/", "_")}.json',
                         {'status': 'pruned', 'processed_observations': 28672})
        if (tmp_path / 'started').exists():
            (tmp_path / 'following_arm').touch()
        (tmp_path / 'started').touch()
        prune_proxy(root, key, {'status': 'pruned', 'reason': 'recorded plateau', 'step': 28672})

    monkeypatch.setattr(sparse, 'SummaryWriter', Writer)
    monkeypatch.setattr(sparse, 'run_arm', stalled_arm)
    with pytest.raises(SystemExit) as exit_info:
        sparse.main()
    assert exit_info.value.code == PRUNED_EXIT_CODE
    assert (tmp_path / 'writer_closed').exists()
    assert not (tmp_path / 'following_arm').exists()
    saved = json.loads(next(tmp_path.glob('*/results.json')).read_text())
    assert saved['status'] == 'pruned'
    assert saved['arms']['stationary/covariance_q1e-05']['processed_observations'] == 28672
