"""CUDA algebra/architecture contracts for representation microscopy; use mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from cleanrl.ppo_continuous_action_sepclip_anvil2_v4 import Agent
from cleanrl.shared.host_actor import situ_glu
from scripts.audit_sepclip_representation import beta_quantities, head_report, spectra, stages


def make_agent():
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    return Agent(envs).cuda().requires_grad_(False)


def test_exact_situ_odd_even_components_at_nontrivial_gates():
    gate = torch.tensor([-12., -4., -1., -.01, 0., .02, 1.7, 5., 13.], device='cuda')
    up = torch.tensor([32., -.5, -10., 2., 3., -4., .2, -27., 8.], device='cuda')
    positive, negative = situ_glu(gate, up), situ_glu(-gate, -up)
    even = .5 * (4*torch.tanh(gate/4)) * (25*torch.tanh(up/25))
    odd = even*torch.tanh(gate/2)
    torch.testing.assert_close((positive+negative)*.5, even, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close((positive-negative)*.5, odd, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(positive, even+odd, atol=2e-6, rtol=2e-6)
    # Large visited gates need not have weak odd components.
    assert (odd.abs()/even.abs().clamp_min(1e-20))[0] > .99


def test_actual_actor_normalization_jacobian_is_radial_projection():
    torch.manual_seed(1)
    agent = make_agent()
    observations = torch.randn(8, 17, device='cuda')
    features = stages(agent.actor, observations)
    jac = torch.func.vmap(torch.func.jacrev(lambda x: stages(agent.actor, x)))(observations)
    for raw, raw_jac, unit_jac in (
            (features[:, 0], jac[:, 0], jac[:, 1]),
            (features[:, 2], jac[:, 2], jac[:, 3]),
            (features[:, 1]+features[:, 3], jac[:, 1]+jac[:, 3], jac[:, 4])):
        unit = F.normalize(raw, dim=-1)
        radial = torch.einsum('ni,nij->nj', unit, raw_jac)
        expected = (raw_jac-unit[..., None]*radial[:, None])/raw.norm(dim=-1)[:, None, None]
        torch.testing.assert_close(unit_jac, expected, atol=3e-6, rtol=3e-5)
        # Its output derivatives are tangent to the hidden sphere.
        torch.testing.assert_close(torch.einsum('ni,nij->nj', unit, unit_jac),
                                   torch.zeros_like(radial), atol=3e-6, rtol=0)


def test_beta_mean_mode_bounds_for_nontrivial_actor_rows():
    torch.manual_seed(2)
    agent = make_agent()
    with torch.no_grad():
        # Real actor head with state-dependent rows plus strongly asymmetric
        # biases: cover central, near-boundary and high-concentration policies.
        agent.actor.head[0].bias.copy_(torch.tensor(
            [0., 4., -10., 35., -5., 2., 0., -6., 8., -12., 10., 2.], device='cuda'))
    observations = torch.randn(20, 17, device='cuda')
    alpha, beta, concentration, mean, variance, mode = beta_quantities(agent, observations)
    assert (alpha >= 1).all() and (beta >= 1).all()
    assert (mean.abs() <= 1-2/concentration+1e-6).all()
    assert (mode.abs() <= 1+1e-6).all()
    torch.testing.assert_close(mode, (alpha-beta)/(concentration-2), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(mean, mode*(1-2/concentration), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(variance, (1-mean.square())/(concentration+1), atol=2e-6, rtol=2e-6)
    mu = (mean+1)*.5
    minimum_c = torch.maximum(mu.reciprocal(), (1-mu).reciprocal())
    assert (concentration >= minimum_c-2e-4).all()


def test_head_projection_erases_output_std_on_known_rows():
    torch.manual_seed(3)
    agent = make_agent()
    observations = torch.randn(32, 17, device='cuda')
    with torch.no_grad():
        # Fix orthonormal known rows, so initialization's intended .01 gain
        # has an exact identifiable effect independent of RNG orientation.
        head = agent.actor.head[0]
        head.weight.zero_()
        head.weight[:, :12].copy_(torch.eye(12, device='cuda')*.01)
        head.bias.zero_()
        before = agent.actor(observations)
        head.weight.div_(head.weight.norm(dim=1, keepdim=True))
        after_head_only = agent.actor(observations)
        torch.testing.assert_close(after_head_only, before*100, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(agent.actor.readout_gain, torch.ones(12, device='cuda'), atol=0, rtol=0)
        agent.normalize_matrices()
        torch.testing.assert_close(head.weight.norm(dim=1), torch.ones(12, device='cuda'))
        # Projection of the trunk changes the representation; only the exact
        # Cauchy-Schwarz bound is asserted after all matrix projections.
        assert (agent.actor(observations).abs() <= 8+1e-5).all()


def test_zero_spectra_and_uniform_beta_mode_are_not_false_signal():
    zero = spectra(torch.zeros(7, 4, device='cuda'))
    assert zero['effective_rank']['mean'] == 0
    assert zero['numerical_rank_relative_1e8']['mean'] == 0
    agent = make_agent()
    with torch.no_grad():
        agent.actor.head[0].weight.zero_()
        agent.actor.head[0].bias.fill_(-200.)
    obs = torch.randn(10, 17, device='cuda')
    _, _, concentration, mean, variance, mode = beta_quantities(agent, obs)
    torch.testing.assert_close(concentration, torch.full_like(concentration, 2.))
    torch.testing.assert_close(mean, torch.zeros_like(mean))
    torch.testing.assert_close(variance, torch.full_like(variance, 1/3))
    assert torch.isnan(mode).all()
    report = head_report(agent, obs, 1)
    assert report['mode_undefined_count'] == 60
    assert report['physical_mode']['finite_count'] == 0
    assert all(value is None for value in report['torque_proportions']['mode'].values())


def test_effective_head_row_alignment_accounts_for_gain_sign():
    agent = make_agent()
    with torch.no_grad():
        rows = torch.randn(6, 64, device='cuda')
        agent.actor.head[0].weight[:6].copy_(rows)
        agent.actor.head[0].weight[6:].copy_(rows)
        agent.actor.readout_gain[:6].fill_(1.)
        agent.actor.readout_gain[6:].fill_(-1.)
    report = head_report(agent, torch.randn(10, 17, device='cuda'), 1)
    assert abs(report['alpha_beta_head_row_cosine']['mean'] + 1.) < 1e-6
