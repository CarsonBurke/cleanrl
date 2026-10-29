"""Standalone numerical chord contracts; all compiled CUDA use through MLQ."""
import importlib
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_prescription_chord_jvp import audit_chord, response_metrics, output_storage_floor


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(), 'Submit contracts through MLQ; CUDA mandatory'
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def affine_fixture(bias=0.):
    actor = nn.Linear(2, 4).cuda()
    with torch.no_grad():
        actor.weight.zero_(); actor.bias.fill_(bias)
    old = {name: value.detach().clone() for name, value in actor.named_parameters()}
    direction = {name: torch.zeros_like(value) for name, value in old.items()}
    direction['bias'].fill_(.125)
    obs = torch.tensor([[1., 2.], [-1., .5]], device='cuda')
    return actor, old, direction, obs


def test_affine_centered_primal_matches_exact_bias_derivative_at_every_h():
    actor, old, direction, obs = affine_fixture()
    result = audit_chord(actor, old, direction, obs, [1., .5, .25])
    assert result['jvp_norms']['latent_logits'] == pytest.approx(.125 * 8**.5)
    assert result['requested_ambient_chord_norm'] == pytest.approx(.25)
    for row in result['neighboring_steps']:
        assert row['status'] == 'measured'
        assert row['effective_vs_requested_direction_relative_error'] == 0
        metric = row['response_comparisons']['latent_logits']
        assert metric['relative_l2_error'] == 0
        assert metric['cosine'] == pytest.approx(1.)
    assert result['neighboring_steps'][1]['neighbor_comparison']['h_ratio'] == .5


def test_rounding_plateau_withholds_chord_even_when_ad_derivative_nonzero():
    actor, old, direction, obs = affine_fixture(bias=1.)
    direction['bias'].fill_(1e-12)
    result = audit_chord(actor, old, direction, obs, [1., .5, .25])
    assert result['jvp_norms']['latent_logits'] > 0
    for row in result['neighboring_steps']:
        assert row['status'] == 'withheld_unrepresentable_parameter_chord'
        assert row['vanished_nonzero_parameter_fraction'] == 1
        assert row['response_comparisons']['latent_logits']['cosine'] is None
        assert row['response_comparisons']['latent_logits']['zero_rounded_primal_response']


def test_beta_floor_plateau_is_distinct_from_parameter_displacement_loss():
    actor, old, direction, obs = affine_fixture(bias=-20.)
    result = audit_chord(actor, old, direction, obs, [1., .5, .25])
    assert result['jvp_norms']['alpha'] > 0
    assert result['jvp_norms']['beta'] > 0
    for row in result['neighboring_steps']:
        assert row['status'] == 'measured'
        assert row['effective_vs_requested_direction_relative_error'] == 0
        assert row['response_comparisons']['latent_logits']['relative_l2_error'] == 0
        for field in ('alpha', 'beta'):
            assert row['rounded_shape_floor_counts'][field]['both_neighbors_floor'] == 4
            assert row['response_comparisons'][field]['relative_l2_error'] == pytest.approx(1.)
            assert row['response_comparisons'][field]['cosine'] is None
            assert row['output_storage_rounding_floors'][field]['relative_to_jvp_norm'] > 1


@pytest.mark.parametrize('module_name,architecture', [
    ('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14', 'additive_prerms'),
    ('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14', 'original'),
    ('cleanrl.ppo_continuous_action_sepclip_constraint_transport_v15', 'original')])
def test_actual_nonlinear_full_parameter_chord_independent_centered_primals(module_name, architecture):
    module = importlib.import_module(module_name)
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-np.ones(6, np.float32), np.ones(6, np.float32)))
    torch.manual_seed(1729)
    agent = (module.Agent(env, architecture) if module_name.endswith('v14') else module.Agent(env)).cuda().eval()
    with torch.no_grad(): agent.normalize_matrices()
    actor = agent.actor
    old = {name: value.detach().clone() for name, value in actor.named_parameters()}
    direction = {name: (torch.arange(value.numel(), device='cuda', dtype=torch.float32).sin() * 1e-4).reshape(value.shape)
                 for name, value in old.items()}
    obs = torch.linspace(-.7, .9, 4 * 17, device='cuda').reshape(4, 17)
    result = audit_chord(actor, old, direction, obs, [1., .5, .25])
    assert result['jvp_norms']['latent_logits'] > 0
    assert result['nonzero_requested_parameter_elements'] > 1000
    for row in result['neighboring_steps']:
        assert row['status'] == 'measured'
        metric = row['response_comparisons']['latent_logits']
        # Nearby scales all retained; no choosing the best finite difference.
        assert metric['relative_l2_error'] < .05
        assert metric['cosine'] > .995
    assert 'numerical_measurement_only' == result['status']


def test_same_primal_with_detached_additive_body_fails_body_only_derivative_comparison():
    module = importlib.import_module('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14')
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-np.ones(6, np.float32), np.ones(6, np.float32)))
    torch.manual_seed(731)
    agent = module.Agent(env, 'additive_prerms').cuda().eval()
    with torch.no_grad(): agent.normalize_matrices()
    actor = agent.actor
    class DetachedBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = actor.trunk; self.head = actor.head; self.readout_gain = actor.readout_gain
        def forward(self, observations):
            return self.readout_gain * self.head(self.trunk(observations).detach())
    wrong_actor = DetachedBody()
    old = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    direction = {name: torch.zeros_like(value) for name, value in old.items()}
    direction['trunk.in_proj.weight'] = .01 * old['trunk.in_proj.weight']
    obs = torch.linspace(-.8, .9, 4 * 17, device='cuda').reshape(4, 17)
    correct = audit_chord(actor, old, direction, obs, [1., .5, .25])
    wrong = audit_chord(wrong_actor, old, direction, obs, [1., .5, .25])
    assert correct['jvp_norms']['latent_logits'] > 1e-7
    assert wrong['jvp_norms']['latent_logits'] == 0
    for row in wrong['neighboring_steps']:
        metric = row['response_comparisons']['latent_logits']
        assert metric['null_jvp']
        assert metric['centered_primal_norm'] > 1e-7
        assert metric['absolute_l2_error'] > 1e-7
        assert metric['relative_l2_error'] is None
        assert metric['cosine'] is None
        assert row['status'] == 'measured' # Measurement, never an automatic derivative pass.


def test_storage_ulp_floor_uses_actual_fp32_outputs_and_step_denominator():
    plus = torch.ones((2, 2), device='cuda'); minus = plus.clone()
    derivative = torch.ones_like(plus)
    result = output_storage_floor(plus, minus, .5, derivative)
    # Toward-zero spacing at1 is2^-24; (two spacings)/(2h) is2^-23.
    assert result['toward_zero_ulp_sum_over_2h_max'] == pytest.approx(2**-23)
    assert result['toward_zero_ulp_sum_over_2h_l2'] == pytest.approx(2 * 2**-23)
    assert result['relative_to_jvp_norm'] == pytest.approx(2**-23)


@pytest.mark.parametrize('hs', [[1., .5], [1., .5, .5], [.25, .5, 1.], [1., .5, 0.], [1., .5, float('nan')]])
def test_invalid_or_non_neighboring_step_metadata_rejected(hs):
    with pytest.raises(ValueError): audit_chord(*affine_fixture(), hs)


def test_nonfinite_neighbor_parameters_are_withheld_not_clipped_or_projected():
    actor, old, direction, obs = affine_fixture()
    old['bias'].fill_(2e38); direction['bias'].fill_(2e38)
    # Ensure finite base outputs: these negative huge logits give floor1 law.
    old['bias'].neg_(); direction['bias'].neg_()
    result = audit_chord(actor, old, direction, obs, [2., 1., .5])
    assert result['neighboring_steps'][0]['status'] == 'withheld_nonfinite_neighbor_parameters'
    assert result['neighboring_steps'][1]['status'] == 'withheld_nonfinite_neighbor_parameters'


def test_finite_neighbor_parameters_with_nonfinite_primal_outputs_are_withheld():
    class ExponentialActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(85., device='cuda'))
        def forward(self, observations):
            return -self.scale.exp().expand(observations.shape[0], 4)
    actor = ExponentialActor()
    old = {'scale': actor.scale.detach().clone()}
    direction = {'scale': torch.tensor(3., device='cuda')}
    obs = torch.zeros((2, 2), device='cuda')
    result = audit_chord(actor, old, direction, obs, [2., 1., .5])
    # Scale91 is finite, but exp(91) overflows FP32. Base85 AD remains finite.
    assert result['neighboring_steps'][0]['status'] == 'withheld_nonfinite_neighbor_outputs'
    assert result['neighboring_steps'][1]['status'] == 'measured'


def test_null_reference_response_has_no_invented_relative_error_or_cosine():
    zero = torch.zeros((2, 2), device='cuda')
    nonzero = torch.ones_like(zero)
    result = response_metrics(zero, nonzero)
    assert result['relative_l2_error'] is None
    assert result['cosine'] is None
    assert result['absolute_l2_error'] == 2
