"""CUDA actor contract requires mlq; pure statistics tests use synthetic saved labels."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('gate_analysis_test', ROOT / 'scripts/reppo_policy_gate_analysis.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gate_strata_and_entropy_components_use_correct_units():
    raw = np.zeros((4, 3, 2))
    soft = np.zeros_like(raw)
    raw[:, 1], raw[:, 2] = 10, 20
    soft[:, 1], soft[:, 2] = 3, 4
    paired = {'rows': np.arange(4), 'comparison': 'last_actor_update',
              'predictions': {'q_delta': np.ones((4, 2)), 'soft_objective_delta': np.full((4, 2), 2.)},
              'returns': {1000: {'raw': raw, 'soft': soft}}}
    kl = {'prelast': np.array([0., .1, 0., .1]), 'after': np.array([.1, 0., .1, 0.])}
    snapshot = {'args': {'kl_bound': .03, 'reward_scale': .1}, 'temperature': .2,
                'temperature_before_last_actor_update': .1}
    report = module.summarize_outcomes(paired, kl, snapshot)
    assert report['outside_to_inside_fraction'] == .5
    assert report['inside_to_outside_fraction'] == .5
    for stratum in report['strata'].values():
        assert stratum['predicted_first_step_entropy_bonus_gain']['mean'] == 1
        first = stratum['horizons']['1000']['first_action']
        assert first['scaled_raw_gain']['mean'] == 1
        assert first['discounted_entropy_bonus_gain_all_steps']['mean'] == 2
        assert first['soft_gain']['prediction_error_mean'] == 1
        assert first['soft_gain_at_final_temperature']['mean'] == 5
        assert 'predicted_mean' not in first['soft_gain_at_final_temperature']
    assert report['strata']['prelast_outside_gate']['behavior_kl_change_mean'] == -.1
    assert report['strata']['prelast_inside_gate']['behavior_kl_change_mean'] == .1
    paired['comparison'] = 'update'
    assert list(module.summarize_outcomes(paired, kl, snapshot)['strata']) == ['all']


def test_soft_gain_reweighting_changes_only_entropy_component():
    raw = np.array([1., 2., -1.])
    entropy = np.array([2., -3., 0.])
    np.testing.assert_allclose(module.reweight_soft_gain(raw, raw + entropy, .1, .2), raw + 2 * entropy)
    np.testing.assert_allclose(module.reweight_soft_gain(raw, raw + entropy, .1, .1), raw + entropy)
    np.testing.assert_allclose(module.reweight_soft_gain(raw, raw + entropy, .1, 0.), raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; submit through mlq')
def test_compiled_actor_gate_is_relative_to_behavior_not_adjacent_actor():
    trainer, _ = module.oracle.load_trainer({'critic': {'dynamics.0.weight': None}})
    trainer.configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    args = trainer.Args(actor_hidden_dim=32)
    snapshot = {'args': vars(args), 'obs_means': np.zeros((2, 17)),
                'obs_variances': np.ones((2, 17)), 'obs_epsilon': 1e-8}
    for key, mean in (('actor_net_before_update', 0.), ('actor_net_before_last_actor_update', 1.), ('actor_net', .5)):
        actor = trainer.Actor(17, 6, args).cuda()
        with torch.no_grad():
            for parameter in actor.net.parameters():
                parameter.zero_()
            actor.net[-1].bias[:6] = mean
        snapshot[key] = {name: value.detach().cpu().clone() for name, value in actor.net.state_dict().items()}
    paired = {'states': np.zeros((4, 27)), 'rows': np.array([0, 1, 0, 1])}
    kl = module.gate_kl(trainer, snapshot, paired, torch.device('cuda'))
    np.testing.assert_allclose(kl['prelast'], 3.)
    np.testing.assert_allclose(kl['after'], .75)
    assert np.all(kl['after'] < kl['prelast'])
