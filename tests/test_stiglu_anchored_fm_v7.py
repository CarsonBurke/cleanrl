"""Moment separation, genuine residual CFM and detached host sampling contracts."""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_stiglu_anchored_fm_v7 as flow
from cleanrl import ppo_continuous_action_stiglu_conditional_fm_v6 as previous

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def strict_runtime():
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(171)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32))


def test_identity_residual_is_gaussian_even_after_large_mean_and_scale_change():
    torch.manual_seed(1)
    old = previous.Agent(spaces(), previous.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    model = flow.Agent(spaces(), flow.Args(policy="flow")).cuda()
    assert flow.state_hash(model.critic) == previous.state_hash(old.critic)
    assert flow.state_hash(model.actor) == previous.state_hash(old.actor)
    obs = torch.randn(16, 3, device="cuda")
    noise = torch.randn_like(obs)
    with torch.no_grad():
        model.actor[-1].bias.copy_(torch.tensor([30.0, -20.0, 8.0, -0.8, 0.3, 0.7], device="cuda"))
        mean, log_std = model.gaussian_parameters(obs)
        actual = flow.sample_native(model, obs, noise, 8)
        expected = mean + log_std.exp() * noise
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Zero residual field needs no learned trajectory to move the mean.
        sampler = flow.HostSampler(model, 16, 8)
        host, _ = sampler(obs.cpu().numpy(), None, noise=noise.cpu().numpy())
        np.testing.assert_allclose(host, expected.cpu().numpy(), rtol=3e-6, atol=3e-6)


def test_weighted_moment_standardization_and_gaussian_optimum():
    targets = torch.tensor([[[1.0, -3.0], [2.0, -1.0], [6.0, 4.0]],
                            [[-8.0, 0.0], [2.0, 3.0], [5.0, 9.0]]], device="cuda", requires_grad=True)
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], device="cuda", requires_grad=True)
    mean, std = flow.candidate_moments(targets, weights)
    residual = (targets.detach() - mean[:, None]) / std[:, None]
    torch.testing.assert_close((weights.detach()[..., None] * residual).sum(1), torch.zeros_like(mean), atol=2e-7, rtol=0)
    torch.testing.assert_close((weights.detach()[..., None] * residual.square()).sum(1), torch.ones_like(mean), atol=3e-7, rtol=0)
    m = mean.clone().requires_grad_()
    l = std.log().requires_grad_()
    nll = (weights.detach()[..., None] * (0.5 * ((targets.detach() - m[:, None]) * (-l[:, None]).exp()).square() + l[:, None])).sum()
    grad_mean, grad_scale = torch.autograd.grad(nll, (m, l))
    torch.testing.assert_close(grad_mean, torch.zeros_like(grad_mean), atol=2e-7, rtol=0)
    torch.testing.assert_close(grad_scale, torch.zeros_like(grad_scale), atol=3e-7, rtol=0)
    assert not mean.requires_grad and not std.requires_grad


def test_nonlinear_odd_flow_preserves_anchor_mean_and_actual_host_parity():
    model = flow.Agent(spaces(), flow.Args()).cuda()
    with torch.no_grad():
        model.actor[-1].weight.normal_(std=0.03)
        model.shape_actor[-1].weight.normal_(std=0.3)
    single_obs = torch.randn(8, 3, device="cuda")
    obs = single_obs.repeat_interleave(2, 0)
    half_noise = torch.randn(8, 3, device="cuda")
    noise = torch.stack((half_noise, -half_noise), dim=1).reshape(16, 3)
    with torch.no_grad():
        native = flow.sample_native(model, obs, noise, 8)
        mean, log_std = model.gaussian_parameters(obs)
        torch.testing.assert_close(native.view(8, 2, 3).mean(1), mean[::2], atol=3e-7, rtol=3e-6)
        assert ((native - mean) / log_std.exp() - noise).square().mean().item() > 1e-5
        sampler = flow.HostSampler(model, 16, 8)
        host, actions = sampler(obs.cpu().numpy(), None, noise=noise.cpu().numpy())
        np.testing.assert_allclose(host, native.cpu().numpy(), rtol=3e-5, atol=4e-6)
        np.testing.assert_allclose(actions, np.tanh(flow.INITIAL_STD * host), rtol=3e-6, atol=1e-7)
        before = host.copy()
        again, _ = sampler(obs.cpu().numpy(), None, noise=-noise.cpu().numpy())
        assert host is again
        np.testing.assert_allclose(0.5 * (before + again), mean.cpu().numpy(), rtol=3e-5, atol=4e-6)


def test_residual_cfm_is_translation_scale_invariant_and_updates_only_shape():
    model = flow.Agent(spaces(), flow.Args()).cuda()
    obs = torch.randn(16, 3, device="cuda", requires_grad=True)
    targets = torch.randn(16, 3, device="cuda", requires_grad=True)
    mean = torch.randn(16, 3, device="cuda", requires_grad=True)
    std = torch.full_like(mean, 0.8, requires_grad=True)
    noise = torch.randn_like(targets, requires_grad=True)
    times = torch.rand(16, 1, device="cuda", requires_grad=True)
    loss = flow.shape_loss(model, obs, targets, mean, std, noise, times)
    transformed = flow.shape_loss(model, obs, 100 + 5 * targets, 100 + 5 * mean, 5 * std, noise, times)
    torch.testing.assert_close(transformed, loss, rtol=3e-6, atol=3e-6)
    compiled = torch.compile(lambda o, y, m, s, n, t: flow.shape_loss(model, o, y, m, s, n, t), fullgraph=True)
    actual = compiled(obs, targets, mean, std, noise, times)
    torch.testing.assert_close(actual, loss, rtol=3e-6, atol=3e-6)
    actual.backward()
    assert model.shape_actor[-1].weight.grad.norm().item() > 1e-4
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    assert all(value.grad is None for value in (obs, targets, mean, std, noise, times))


def test_gaussian_anchor_projection_gradient_matches_v6():
    torch.manual_seed(1)
    old = previous.Agent(spaces(), previous.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    model = flow.Agent(spaces(), flow.Args(policy="flow")).cuda()
    obs = torch.randn(16, 3, device="cuda")
    target = torch.randn_like(obs)
    noise, times = torch.randn_like(obs), torch.rand(16, 1, device="cuda")
    reference = previous.projection_loss(old, obs, target, noise, times)
    actual = flow.projection_loss(model, obs, target, noise, times)
    reference.backward()
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for parameter, other in zip(model.actor.parameters(), old.actor.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.shape_actor.parameters())
