"""Exact score and detached-sampling contracts for conditional transport PPO."""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_stiglu_distribution_v2 as control
from cleanrl import ppo_continuous_action_stiglu_transport_ppo_v4 as flow

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def strict_runtime():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(73)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2.0, 0.5, -0.3], np.float32), np.array([1.0, 3.0, 0.7], np.float32)),
    )


def nontrivial_agent(initial_std=0.610376541075546):
    agent = flow.Agent(spaces(), flow.Args(initial_std=initial_std, whiten_mean=True)).cuda()
    with torch.no_grad():
        agent.actor[-1].weight.normal_(std=0.03)
        for coupling in agent.couplings:
            coupling[-1].weight.normal_(std=0.2)
            coupling[-1].bias.normal_(std=0.1)
    return agent


def test_identity_transport_matches_gaussian_density_initialization_and_score_gradient():
    torch.manual_seed(1)
    gaussian = control.Agent(spaces(), control.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    agent = flow.Agent(spaces(), flow.Args()).cuda()
    assert flow.state_hash(agent.critic) == control.state_hash(gaussian.critic)
    assert flow.state_hash(agent.actor) == control.state_hash(gaussian.actor)
    obs = torch.randn(17, 3, device="cuda")
    native = torch.randn(17, 3, device="cuda") * 0.3
    first, second = gaussian.policy_parameters(obs)
    expected = gaussian.action_logprob(first, second, native)
    actual = agent.logprob(obs, native)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    expected.sum().backward()
    actual.sum().backward()
    for candidate, reference in zip(agent.actor.parameters(), gaussian.actor.parameters()):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=3e-5, atol=3e-6)


def test_nonlinear_transport_density_matches_full_jacobian_and_inverse():
    agent = nontrivial_agent().double()
    observation = torch.tensor([[0.2, -0.4, 0.8]], device="cuda", dtype=torch.float64)
    source = torch.tensor([[0.3, -0.7, 1.1]], device="cuda", dtype=torch.float64)
    native, forward_score = agent.sample_native(observation, source)
    mean, log_std = agent.policy_parameters(observation)
    recovered, inverse_logdet = agent.transport(observation, (native - mean) / log_std.exp(), inverse=True)
    torch.testing.assert_close(recovered, source, rtol=2e-10, atol=2e-10)
    jacobian = torch.autograd.functional.jacobian(
        lambda epsilon: agent.sample_native(observation, epsilon.unsqueeze(0))[0][0], source[0])
    sign, logdet = torch.linalg.slogdet(jacobian)
    assert sign.item() == 1
    expected = (-0.5 * source.square() - 0.5 * math.log(2 * math.pi)).sum(-1) - logdet
    torch.testing.assert_close(agent.logprob(observation, native), expected, rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(forward_score, expected, rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(inverse_logdet + log_std.sum(-1), logdet.expand(1), rtol=2e-9, atol=2e-9)
    physical_jacobian = torch.autograd.functional.jacobian(
        lambda epsilon: (agent.action_bias + agent.action_scale * agent.sample_native(observation, epsilon.unsqueeze(0))[0].tanh())[0], source[0])
    physical_expected = (-0.5 * source.square() - 0.5 * math.log(2 * math.pi)).sum(-1) - torch.linalg.slogdet(physical_jacobian)[1]
    torch.testing.assert_close(agent.physical_logprob(observation, native), physical_expected, rtol=2e-8, atol=2e-8)


def test_inverse_score_gradients_keep_all_parameter_dependencies_but_not_sample_graph():
    agent = nontrivial_agent().double()
    observation = torch.tensor([[0.2, -0.4, 0.8]], device="cuda", dtype=torch.float64)
    native = torch.tensor([[0.1, -0.3, 0.6]], device="cuda", dtype=torch.float64, requires_grad=True)
    agent.logprob(observation, native).sum().backward()
    assert native.grad is None
    for parameter, index in ((agent.actor[-1].bias, 0), (agent.actor[-1].bias, 4),
                             (agent.couplings[0][-1].bias, 0), (agent.couplings[1][-1].weight, (0, 3))):
        gradient = parameter.grad[index].item()
        with torch.no_grad():
            original = parameter[index].item()
            parameter[index] = original + 1e-5
            plus = agent.logprob(observation, native).item()
            parameter[index] = original - 1e-5
            minus = agent.logprob(observation, native).item()
            parameter[index] = original
        assert abs(gradient) > 1e-7
        assert gradient == pytest.approx((plus - minus) / 2e-5, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("initial_std", [0.610376541075546, 1 / 6])
def test_host_forward_density_replay_and_borrowed_buffers_at_low_noise(initial_std):
    agent = nontrivial_agent(initial_std)
    observations = np.random.default_rng(82).normal(size=(16, 3)).astype(np.float32)
    sampler = flow.HostSampler(agent, 16)
    rng = np.random.default_rng(83)
    native_np, action_np = sampler(observations, rng)
    native = torch.from_numpy(native_np.copy()).cuda()
    old_logprob = torch.from_numpy(sampler.logprob.copy()).cuda()
    source = torch.from_numpy(sampler.source.copy()).cuda()
    obs = torch.from_numpy(observations).cuda()
    with torch.no_grad():
        replay_native, replay_logprob = agent.sample_native(obs, source)
        torch.testing.assert_close(replay_native, native, rtol=4e-5, atol=2e-6)
        torch.testing.assert_close(replay_logprob, old_logprob, rtol=4e-5, atol=1e-5)
        torch.testing.assert_close(agent.logprob(obs, native), old_logprob, rtol=4e-5, atol=1e-5)
        np.testing.assert_allclose(action_np, (agent.action_bias + agent.action_scale * native.tanh()).cpu().numpy(), rtol=3e-6, atol=3e-7)
        agent.couplings[0][-1].bias.add_(0.2)
        _, preserved, drift = flow.rollout_statistics(agent, obs, native, old_logprob)
        torch.testing.assert_close(preserved, old_logprob, rtol=0, atol=0)
        assert drift[1].item() > 0.01
        sampler.refresh()
        next_native, next_action = sampler(observations, rng)
        assert next_native is native_np and next_action is action_np
        torch.testing.assert_close(agent.logprob(obs, torch.from_numpy(next_native.copy()).cuda()),
                                   torch.from_numpy(sampler.logprob.copy()).cuda(), rtol=4e-5, atol=1e-5)


def test_compiled_ppo_uses_signed_clipped_joint_ratio_and_unweighted_critic():
    args = flow.Args(norm_adv=False)
    agent = nontrivial_agent()
    obs = torch.randn(32, 3, device="cuda")
    native = torch.randn(32, 3, device="cuda") * 0.3
    with torch.no_grad():
        logprob = agent.logprob(obs, native)
        # Deliberately include ratios below and above BOTH clipping boundaries,
        # with both advantage signs. Incorrect weighted regression cannot pass.
        ratio = torch.tensor([0.5, 0.9, 1.1, 1.5], device="cuda").repeat(8)
        old_logprob = logprob - ratio.log()
        advantages = torch.tensor([-1.0, 1.0], device="cuda").repeat_interleave(4).repeat(4)
        old_values = agent.get_value(obs).flatten()
        returns = old_values + torch.linspace(-1, 1, 32, device="cuda")
        expected_policy = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
        expected_value = 0.5 * (returns - old_values).square().mean()
    compiled = torch.compile(lambda o, a, lp, adv, ret, val: flow.ppo_loss(agent, o, a, lp, adv, ret, val, args), fullgraph=True)
    loss, metrics = compiled(obs, native, old_logprob, advantages, returns, old_values)
    torch.testing.assert_close(metrics[0], expected_policy, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(metrics[1], expected_value, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(loss, expected_policy + args.vf_coef * expected_value, rtol=2e-5, atol=2e-6)
    loss.backward()
    assert torch.isfinite(agent.couplings[0][-1].weight.grad).all()
    assert agent.couplings[0][-1].weight.grad.norm().item() > 1e-5
