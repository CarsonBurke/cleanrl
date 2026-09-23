"""FPO algorithm/gradient and host-device contracts; CUDA execution is queue-only."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_fpo_reference_actor_v2 as fpo


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def deterministic_runtime():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(71)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def environments():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.array([-3.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 5.5], dtype=np.float32),
        ),
    )


def make_args(**kwargs):
    return fpo.validate_args(fpo.Args(num_envs=2, num_steps=4, num_minibatches=2, **kwargs))


@pytest.mark.parametrize("mse", ["epsilon", "velocity"])
def test_stored_pairs_give_unit_ratios_and_remain_frozen_across_compiled_updates(mse):
    args = make_args(mse=mse)
    agent = fpo.Agent(environments(), args).cuda()
    observations = torch.randn(args.batch_size, 3, device="cuda")
    native = torch.randn(args.batch_size, 2, device="cuda")
    cache = fpo.RolloutCFM(args, 2, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(231)

    def statistics(obs, targets, times, noise):
        return agent.get_value(obs).flatten(), fpo.cfm_loss(agent, obs, targets, times, noise, args.mse)

    stats = fpo.graph_compile(statistics)
    cache.refresh(stats, observations, native, generator)
    frozen = [value.clone() for value in (cache.times, cache.noise, cache.old_loss, cache.old_values, native)]
    indices = torch.tensor([7, 1, 5, 0], device="cuda")
    # Reordered subsets must use their original pairs, not resample MC draws.
    with torch.no_grad():
        before = stats(observations[indices], native[indices], cache.times[indices], cache.noise[indices])[1]
        torch.testing.assert_close((cache.old_loss[indices] - before).exp(), torch.ones_like(before), rtol=0, atol=2e-6)
    advantages = torch.tensor([-2.0, 1.0, -1.0, 3.0], device="cuda")
    returns = cache.old_values[indices] + advantages

    def learner(obs, targets, times, noise, old_loss, adv, ret, values):
        return fpo.fpo_loss(agent, obs, targets, times, noise, old_loss, adv, ret, values, args)

    learner = torch.compile(learner, mode="reduce-overhead", fullgraph=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5, fused=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        loss, _ = learner(observations[indices], native[indices], cache.times[indices], cache.noise[indices],
                          cache.old_loss[indices], advantages, returns, cache.old_values[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
    with torch.no_grad():
        after = stats(observations[indices], native[indices], cache.times[indices], cache.noise[indices])[1]
        assert torch.any((cache.old_loss[indices] - after).abs() > 1e-6)
    for actual, expected in zip((cache.times, cache.noise, cache.old_loss, cache.old_values, native), frozen, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad


def test_action_mean_then_mc_mean_precedes_exponentiation_and_clipping():
    args = make_args(norm_adv=False, clip_coef=0.05)
    agent = fpo.Agent(environments(), args).cuda()
    with torch.no_grad():
        agent.actor[-1].weight.zero_()
        agent.actor[-1].bias.zero_()
    observations = torch.zeros(2, 3, device="cuda")
    native = torch.tensor([[2.0, 4.0], [1.0, 3.0]], device="cuda")
    noise = torch.zeros(2, 2, 2, device="cuda")
    times = torch.tensor([[[0.25], [0.75]], [[0.2], [0.8]]], device="cuda")
    expected_errors = torch.tensor([[5.625, 0.625], [3.2, 0.2]], device="cuda")
    errors = fpo.cfm_errors(agent, observations, native, times, noise, "epsilon")
    torch.testing.assert_close(errors, expected_errors)
    per_action = fpo.cfm_loss(agent, observations, native, times, noise, "epsilon")
    torch.testing.assert_close(per_action, expected_errors.mean(-1))
    old_loss = per_action.detach() + torch.tensor([0.1, -0.1], device="cuda")
    advantages = torch.tensor([2.0, -3.0], device="cuda")
    with torch.no_grad():
        values = agent.get_value(observations).flatten()
    total_loss, metrics = fpo.fpo_loss(agent, observations, native, times, noise, old_loss,
                                       advantages, values, values, args)
    expected_ratio = (old_loss - expected_errors.mean(-1)).exp()
    wrong_ratio = (old_loss[:, None] - expected_errors).exp().mean(-1)
    assert torch.all(wrong_ratio > expected_ratio + 0.5)
    expected_loss = torch.maximum(-advantages * expected_ratio, -advantages * expected_ratio.clamp(0.95, 1.05)).mean()
    torch.testing.assert_close(total_loss, expected_loss)
    torch.testing.assert_close(metrics[5], expected_ratio.mean())
    torch.testing.assert_close(metrics[4], torch.ones((), device="cuda"))


def test_signed_clipping_only_blocks_the_improving_side_and_freezes_old_statistics():
    # Positive advantages stop at the upper clip, negative ones at the lower;
    # the opposite sides retain gradients to recover from harmful updates.
    ratios = torch.tensor([1.2, 0.8, 0.8, 1.2, 1.0, 1.0], device="cuda")
    new_loss = (-ratios.log()).requires_grad_()
    old_loss = torch.zeros_like(new_loss, requires_grad=True)
    advantages = torch.tensor([2.0, 2.0, -3.0, -3.0, 2.0, -3.0], device="cuda", requires_grad=True)
    loss, _, ratio = fpo.clipped_surrogate(new_loss, old_loss, advantages, 0.05)
    loss.backward()
    torch.testing.assert_close(ratio, ratios)
    expected_gradient = torch.tensor([0.0, 1.6, 0.0, -3.6, 2.0, -3.0], device="cuda") / 6
    torch.testing.assert_close(new_loss.grad, expected_gradient)
    assert old_loss.grad is None
    assert advantages.grad is None


def test_epsilon_velocity_relation_endpoints_and_detached_rollout_targets():
    args = make_args()
    agent = fpo.Agent(environments(), args).cuda()
    observations = torch.randn(3, 3, device="cuda", requires_grad=True)
    native = torch.randn(3, 2, device="cuda", requires_grad=True)
    noise = torch.randn(3, 4, 2, device="cuda", requires_grad=True)
    times = torch.tensor([0.0, 0.2, 0.7, 1.0], device="cuda").reshape(1, 4, 1).expand(3, -1, -1).clone().requires_grad_()
    states, target_velocity = fpo.flow_interpolant(native[:, None, :], noise, times)
    torch.testing.assert_close(states[:, 0], native)
    torch.testing.assert_close(states[:, -1], noise[:, -1])
    torch.testing.assert_close(target_velocity, noise - native[:, None, :])
    assert not states.requires_grad and not target_velocity.requires_grad
    epsilon_errors = fpo.cfm_errors(agent, observations, native, times, noise, "epsilon")
    velocity_errors = fpo.cfm_errors(agent, observations, native, times, noise, "velocity")
    torch.testing.assert_close(epsilon_errors, (1.0 - times.detach().squeeze(-1)).square() * velocity_errors,
                               rtol=2e-5, atol=2e-6)
    epsilon_errors.mean().backward()
    assert agent.actor[-1].weight.grad.norm() > 0
    for target in (observations, native, noise, times):
        assert target.grad is None


def test_value_clip_is_independent_of_actor_clip_and_retains_baseline_gradient():
    args = make_args(norm_adv=False, value_clip_coef=0.2, clip_coef=0.05)
    agent = fpo.Agent(environments(), args).cuda()
    with torch.no_grad():
        agent.critic[-1].weight.zero_()
        agent.critic[-1].bias.fill_(0.1)
    observations = torch.zeros(2, 3, device="cuda")
    native = torch.zeros(2, 2, device="cuda")
    times = torch.full((2, 2, 1), 0.5, device="cuda")
    noise = torch.ones(2, 2, 2, device="cuda")
    with torch.no_grad():
        old_loss = fpo.cfm_loss(agent, observations, native, times, noise, args.mse)
    zeros, returns = torch.zeros(2, device="cuda"), torch.ones(2, device="cuda")
    loss, metrics = fpo.fpo_loss(agent, observations, native, times, noise, old_loss, zeros, returns, zeros, args)
    loss.backward()
    torch.testing.assert_close(metrics[1], torch.tensor(0.405, device="cuda"))
    # Accidentally using actor clip=.05 would clip the value and zero this gradient.
    torch.testing.assert_close(agent.critic[-1].bias.grad, torch.tensor([-0.9 * args.vf_coef], device="cuda"))


@pytest.mark.parametrize("transform", ["raw", "tanh"])
def test_host_euler_parity_refresh_and_original_saturated_native_rollout_storage(transform):
    args = make_args(action_transform=transform)
    agent = fpo.Agent(environments(), args).cuda()
    sampler = fpo.HostSampler(agent, 2, args.flow_steps)
    # A changed velocity with state/time dependence catches stale mirrors and
    # wrong integration direction/grid, rather than merely comparing zero heads.
    with torch.no_grad():
        agent.actor[-1].weight.copy_(torch.linspace(-0.15, 0.2, agent.actor[-1].weight.numel(), device="cuda").reshape_as(agent.actor[-1].weight))
        agent.actor[-1].bias.copy_(torch.tensor([0.2, -0.3], device="cuda"))
    sampler.refresh()
    observations = np.array([[0.3, -0.8, 1.0], [-1.1, 0.6, 0.2]], dtype=np.float32)
    noise = np.array([[0.7, -1.2], [-0.4, 1.3]], dtype=np.float32)
    original_noise = noise.copy()
    native, action = sampler(observations, None, noise=noise)
    gpu_native, gpu_action = agent.sample(torch.from_numpy(observations).cuda(), torch.from_numpy(noise).cuda())
    np.testing.assert_allclose(native, gpu_native.cpu().numpy(), rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(action, gpu_action.cpu().numpy(), rtol=2e-5, atol=3e-6)
    np.testing.assert_array_equal(noise, original_noise)
    assert not np.shares_memory(native, action)
    assert not np.shares_memory(native, noise)

    transfer = fpo.RolloutTransfer(2, 2, (3,), torch.device("cuda"),
                                   fields={"observations": (3,), "native_actions": (2,)})
    try:
        saturated_noise = np.array([[20.0, -20.0], [-25.0, 25.0]], dtype=np.float32)
        native, action = sampler(observations, None, noise=saturated_noise)
        expected_native = native.copy()
        if transform == "tanh":
            normalized_action = (action - sampler.bias) / sampler.scale
            np.testing.assert_array_equal(np.abs(normalized_action), np.ones_like(normalized_action))
        else:
            np.testing.assert_array_equal(action, expected_native)
            assert (action[:, 0] > environments().single_action_space.high[0]).any()
        transfer.push(0, np.zeros(2), np.zeros(2), np.zeros(2), observations=observations, native_actions=native)
        next_native, _ = sampler(observations, None, noise=-saturated_noise)
        assert np.shares_memory(native, next_native)
        transfer.push(1, np.zeros(2), np.zeros(2), np.zeros(2), observations=observations, native_actions=next_native)
        stored = transfer.upload().fields["native_actions"]
        torch.testing.assert_close(stored[0], torch.from_numpy(expected_native).cuda(), rtol=0, atol=0)
        torch.testing.assert_close(stored[1], torch.from_numpy(next_native.copy()).cuda(), rtol=0, atol=0)
        assert torch.isfinite(stored).all()
        # Recovering native targets from saturated actions would produce infinities
        # (or clipped inverses near 7), not the original large ODE output.
        assert (stored[0].abs() > 10).all()
    finally:
        transfer.close()


def test_one_step_discrete_epsilon_rejects_zero_gradient_training():
    # At the only Euler time t=1, epsilon_hat=noise independently of weights.
    with pytest.raises(ValueError, match="zero actor gradient"):
        make_args(flow_steps=1)
    args = make_args(flow_steps=1, discrete_training_times=False)
    agent = fpo.Agent(environments(), args).cuda()
    cache = fpo.RolloutCFM(args, 2, "cuda")
    obs = torch.randn(args.batch_size, 3, device="cuda")
    native = torch.randn(args.batch_size, 2, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(19)
    cache.refresh(lambda s, a, t, e: (agent.get_value(s).flatten(),
                                     fpo.cfm_loss(agent, s, a, t, e, args.mse)),
                  obs, native, generator)
    fpo.cfm_loss(agent, obs, native, cache.times, cache.noise, args.mse).mean().backward()
    assert agent.actor[-1].weight.grad.norm() > 0


def test_reference_velocity_loss_and_euler_match_independent_equations():
    """Reference equations, not two calls sharing the same sampler/velocity."""
    args = make_args()
    agent = fpo.Agent(environments(), args).cuda()
    obs = torch.randn(4, 3, device="cuda")
    noise = torch.randn(4, 2, device="cuda")
    targets = torch.randn(4, 2, device="cuda")

    def reference_velocity(states, times):
        frequencies = torch.tensor([1., 2., 4., 8.], device="cuda")
        phases = times * frequencies
        x = torch.cat((obs, states, phases.cos(), phases.sin()), -1)
        # Independent LeCun MLP evaluation; no Agent.velocity or Sequential call.
        weights = list(agent.actor.parameters())
        for index in range(0, len(weights), 2):
            x = torch.nn.functional.linear(x, weights[index], weights[index + 1])
            if index < len(weights) - 2:
                x = x * torch.sigmoid(x)
        return 0.25 * x

    with torch.no_grad():
        expected = noise.clone()
        for step in range(args.flow_steps):
            t = torch.full((4, 1), 1. - step / args.flow_steps, device="cuda")
            expected -= reference_velocity(expected, t) / args.flow_steps
        sampled, action = agent.sample(obs, noise)
        torch.testing.assert_close(sampled, expected)
        torch.testing.assert_close(action, expected)
        t = torch.tensor([[1.], [.9], [.4], [.1]], device="cuda")
        interpolated = t * noise + (1. - t) * targets
        velocity = reference_velocity(interpolated, t)
        predicted_eps = interpolated + (1. - t) * velocity
        expected_loss = (predicted_eps - noise).square().mean(-1)
        actual = fpo.cfm_loss(agent, obs, targets, t[:, None], noise[:, None], "epsilon")
        torch.testing.assert_close(actual, expected_loss)


def test_fused_silu_matches_cuda_across_negative_tail_and_refresh():
    network = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.SiLU()).cuda()
    with torch.no_grad():
        network[0].weight.fill_(1.)
        network[0].bias.zero_()
    inputs = np.array([-100., -85., -20., -3., -1., 0., 1., 3., 20., 100.], dtype=np.float32)[:, None]
    mirror = fpo.make_host_mirror(network, len(inputs))
    for shift in (0., .25):
        with torch.no_grad():
            network[0].bias.fill_(shift)
            expected = network(torch.from_numpy(inputs).cuda()).cpu().numpy()
        mirror.refresh()
        np.testing.assert_allclose(mirror(inputs), expected, rtol=2e-5, atol=1e-7)
