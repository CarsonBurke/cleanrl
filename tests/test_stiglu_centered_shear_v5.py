"""Contracts for odd, volume-preserving exploration and its exact PPO score."""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_stiglu_centered_shear_v5 as flow

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
            torch.manual_seed(81)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def agent(initial_std=1 / 6):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32))
    result = flow.Agent(spaces, flow.Args(initial_std=initial_std, whiten_mean=True)).cuda()
    with torch.no_grad():
        result.actor[-1].weight.normal_(std=0.04)
        for coupling in result.couplings:
            coupling[-1].weight.normal_(std=0.7)
    return result


def test_nonlinear_shears_are_odd_invertible_unit_volume_and_mean_preserving():
    model = agent().double()
    observation = torch.tensor([[0.2, -0.4, 0.8]], device="cuda", dtype=torch.float64)
    source = torch.tensor([[0.3, -0.7, 1.1]], device="cuda", dtype=torch.float64)
    transformed, claimed_logdet = model.transport(observation, source)
    negative, _ = model.transport(observation, -source)
    torch.testing.assert_close(transformed, -negative, rtol=2e-12, atol=2e-12)
    assert (transformed - source).norm().item() > 1e-3
    recovered, _ = model.transport(observation, transformed, inverse=True)
    torch.testing.assert_close(recovered, source, rtol=2e-12, atol=2e-12)
    jacobian = torch.autograd.functional.jacobian(
        lambda epsilon: model.transport(observation, epsilon[None])[0][0], source[0])
    sign, actual_logdet = torch.linalg.slogdet(jacobian)
    assert sign.item() == 1
    torch.testing.assert_close(actual_logdet, torch.zeros_like(actual_logdet), rtol=0, atol=2e-10)
    torch.testing.assert_close(claimed_logdet, actual_logdet.expand(1), rtol=0, atol=2e-10)
    observations = observation.expand(2, -1)
    native, forward_density = model.sample_native(observations, torch.cat((source, -source)))
    mean, log_std = model.policy_parameters(observations)
    torch.testing.assert_close(native.mean(0), mean[0], rtol=2e-12, atol=2e-12)
    expected = (-0.5 * source.square() - 0.5 * math.log(2 * math.pi) - log_std[0]).sum(-1)
    torch.testing.assert_close(model.logprob(observations, native), expected.expand(2), rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(forward_density, expected.expand(2), rtol=2e-10, atol=2e-10)


def test_centered_inverse_score_gradient_matches_finite_difference_not_sampling_gradient():
    model = agent().double()
    obs = torch.tensor([[0.2, -0.4, 0.8]], device="cuda", dtype=torch.float64)
    native = torch.tensor([[0.1, -0.3, 0.6]], device="cuda", dtype=torch.float64, requires_grad=True)
    model.logprob(obs, native).sum().backward()
    assert native.grad is None
    for parameter, index in ((model.actor[-1].bias, 0), (model.actor[-1].bias, 4),
                             (model.couplings[0][-1].weight, (0, 3)), (model.couplings[1][-1].weight, (0, 4))):
        gradient = parameter.grad[index].item()
        with torch.no_grad():
            original = parameter[index].item()
            parameter[index] = original + 1e-5
            plus = model.logprob(obs, native).item()
            parameter[index] = original - 1e-5
            minus = model.logprob(obs, native).item()
            parameter[index] = original
        assert abs(gradient) > 1e-7
        assert gradient == pytest.approx((plus - minus) / 2e-5, rel=3e-5, abs=2e-6)


@pytest.mark.parametrize("initial_std", [1 / 6, 1 / math.sqrt(1536)])
def test_paired_host_buffers_and_compiled_ppo_preserve_true_behavior_score(initial_std):
    model = agent(initial_std)
    sampler = flow.HostSampler(model, 16)
    rng = np.random.default_rng(92)
    observations = rng.normal(size=(16, 3)).astype(np.float32)
    native_np, _ = sampler(observations, rng)
    obs = torch.from_numpy(observations).cuda()
    native = torch.from_numpy(native_np.copy()).cuda()
    source = torch.from_numpy(sampler.source.copy()).cuda()
    old_logprob = torch.from_numpy(sampler.logprob.copy()).cuda()
    with torch.no_grad():
        expected_native, expected_score = model.sample_native(obs, source)
        torch.testing.assert_close(native, expected_native, rtol=4e-5, atol=2e-6)
        torch.testing.assert_close(old_logprob, expected_score, rtol=4e-5, atol=1e-5)
        torch.testing.assert_close(model.logprob(obs, native), old_logprob, rtol=4e-5, atol=1e-5)
        values = model.get_value(obs).flatten()
        advantages = torch.linspace(-1, 1, 16, device="cuda")
        returns = values + advantages
        model.couplings[0][-1].weight.add_(0.05)
        _, denominator, drift = flow.rollout_statistics(model, obs, native, old_logprob)
        torch.testing.assert_close(denominator, old_logprob, rtol=0, atol=0)
        assert drift[1].item() > 1e-3
    args = flow.Args(norm_adv=False)
    eager, _ = flow.ppo_loss(model, obs, native, old_logprob, advantages, returns, values, args)
    expected_gradient = torch.autograd.grad(eager, model.couplings[0][-1].weight)[0]
    compiled = torch.compile(lambda o, a, lp, adv, ret, val: flow.ppo_loss(model, o, a, lp, adv, ret, val, args), fullgraph=True)
    loss, _ = compiled(obs, native, old_logprob, advantages, returns, values)
    torch.testing.assert_close(loss, eager, rtol=3e-5, atol=3e-6)
    loss.backward()
    torch.testing.assert_close(model.couplings[0][-1].weight.grad, expected_gradient, rtol=1e-4, atol=5e-6)
    sampler.refresh()
    with torch.no_grad():
        updated, _ = sampler(observations, rng)
        torch.testing.assert_close(model.logprob(obs, torch.from_numpy(updated.copy()).cuda()),
                                   torch.from_numpy(sampler.logprob.copy()).cuda(), rtol=4e-5, atol=1e-5)
