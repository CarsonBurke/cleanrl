"""Contracts for the exact autoregressive Beta policy (cleanrl/fpo/ppo_continuous_action_sepclip_autoregressive_beta_v10.py)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Beta

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "cleanrl/fpo/ppo_continuous_action_sepclip_autoregressive_beta_v10.py"
    spec = importlib.util.spec_from_file_location("sepclip_autoregressive_beta_v10", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent(module, coupling_gain=None, act_dim=6, low=-1.0, high=1.0):
    envs = SimpleNamespace(
        single_action_space=gym.spaces.Box(low, high, (act_dim,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
    )
    torch.manual_seed(0)
    agent = module.Agent(envs)
    agent.normalize_matrices()
    if coupling_gain is not None:
        with torch.no_grad():
            agent.actor.readout_gain[2 * act_dim:] = coupling_gain
    return agent


def test_dimension_j_concentrations_depend_only_on_earlier_dimensions():
    module = _load()
    agent = _agent(module, coupling_gain=1.0)
    logits = agent.actor(torch.randn(1, 17)).detach()
    native = torch.rand(1, 6, requires_grad=True)
    alpha, beta, _ = agent.concentrations(logits, native)
    for dim in range(6):
        grad_alpha, = torch.autograd.grad(alpha[0, dim], native, retain_graph=True)
        grad_beta, = torch.autograd.grad(beta[0, dim], native, retain_graph=True)
        assert torch.all(grad_alpha[0, dim:] == 0) and torch.all(grad_beta[0, dim:] == 0)
        if dim:
            assert grad_alpha[0, :dim].abs().sum() > 0


def test_logprob_is_the_exact_chain_rule_sum_of_conditionals():
    module = _load()
    agent = _agent(module, coupling_gain=1.5, low=-2.0, high=3.0)
    logits = agent.actor(torch.randn(64, 17)).detach()
    native = torch.rand(64, 6).clamp(1e-3, 1 - 1e-3)
    expected = torch.zeros(64)
    prefix = torch.full_like(native, 0.5)  # entries >= dim must not matter
    for dim in range(6):
        alpha, beta, _ = agent.concentrations(logits, prefix)
        expected += Beta(alpha[:, dim], beta[:, dim]).log_prob(native[:, dim]) - np.log(5.0)
        prefix[:, dim] = native[:, dim]
    torch.testing.assert_close(agent.action_logprob(logits, native), expected, rtol=1e-5, atol=1e-5)


def test_zero_coupling_gain_is_the_diagonal_control_policy():
    module = _load()
    agent = _agent(module)  # default init
    logits = agent.actor(torch.randn(32, 17)).detach()
    native = torch.rand(32, 6)
    alpha, beta, term = agent.concentrations(logits, native)
    control_alpha, control_beta = (torch.nn.functional.softplus(logits[:, :12]) + 1.0).chunk(2, dim=-1)
    assert torch.equal(term, torch.zeros_like(term))
    # Same values up to softplus rounding on a strided view.
    torch.testing.assert_close(alpha, control_alpha, rtol=0, atol=1e-6)
    torch.testing.assert_close(beta, control_beta, rtol=0, atol=1e-6)


def test_host_and_device_ancestral_samplers_draw_the_same_coupled_distribution():
    module = _load()
    agent = _agent(module, coupling_gain=2.0)
    state_logits = agent.actor(torch.randn(1, 17)).detach()
    rows = 200_000
    host = module.HostAutoregressiveBeta(rows, 6, np.full(6, -1.0, np.float32), np.ones(6, np.float32))
    native, action = host(np.repeat(state_logits.numpy(), rows, axis=0), np.random.default_rng(0))
    np.testing.assert_allclose(action, 2.0 * native - 1.0, rtol=0, atol=1e-6)
    assert native.min() >= np.float32(module.SAMPLE_EPS) and native.max() <= np.float32(1 - module.SAMPLE_EPS)
    torch.manual_seed(1)
    device = agent.sample_native(state_logits.expand(rows, -1)).numpy()
    host_corr, device_corr = np.corrcoef(native.T), np.corrcoef(device.T)
    assert np.abs(host_corr - device_corr).max() < 0.02
    assert np.abs(native.mean(0) - device.mean(0)).max() < 4e-3
    assert np.abs(native.std(0) - device.std(0)).max() < 4e-3
    assert np.abs(host_corr[np.tril_indices(6, -1)]).max() > 0.1  # the coupling is real


def test_ratio_is_one_before_the_update_and_coupling_receives_gradient():
    module = _load()
    agent = _agent(module, coupling_gain=0.3)
    args = SimpleNamespace(clip_coef=0.2, clip_coef_upper=0.2, norm_adv=False, clip_vloss=False,
                           ent_coef=0.0, vf_coef=0.5)
    observations, native = torch.randn(128, 17), torch.rand(128, 6).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        logits, values = agent.get_policy_and_value(observations)
        old = agent.action_logprob(logits, native)
    loss, metrics = module.ppo_loss(agent, observations, native, old, torch.randn(128), torch.randn(128),
                                    values.view(-1), args)
    assert metrics[4].abs() < 1e-7 and metrics[5] == 0 and metrics[6] > 0
    loss.backward()
    assert agent.actor.readout_gain.grad[12:].abs().sum() > 0
