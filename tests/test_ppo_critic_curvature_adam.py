"""Queue these critic-only curvature and baseline actor CUDA checks through mlq."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_critic_curvature_adam_v3 import (
    Agent,
    CriticCurvatureAdam,
    critic_objective,
    ppo_loss,
    value_loss,
)
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def parameters(actor=(2.0, -1.0), critic=(0.5, 3.0)):
    return [torch.nn.Parameter(torch.tensor(value, device="cuda")) for value in (actor, critic)]


def quadratic(critic, curvature, target=0.0):
    return (0.5 * (curvature * (critic - target).square()).sum()).reshape(1)


def history(optimizer, parameter):
    return {key: optimizer.state[parameter][key].clone() for key in ("step", "exp_avg", "exp_avg_sq")}


def assert_history(optimizer, parameter, expected):
    for key, value in expected.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)


def test_external_actor_adam_matches_joint_baseline_across_changing_gradients_and_lrs():
    actor, critic = parameters((3.0, -4.0, 1e-4), (2.0, -1.0))
    refs = [torch.nn.Parameter(parameter.detach().clone()) for parameter in (actor, critic)]
    optimizer = CriticCurvatureAdam([actor], [critic], lr=0.4, compile=False)
    actor_optimizer = torch.optim.Adam([actor], lr=3e-4, eps=1e-5, fused=True)
    reference = torch.optim.Adam(refs, lr=3e-4, eps=1e-5, fused=True)
    prescribed = (
        ((300.0, -400.0, 0.01), (900.0, -200.0), 3e-4),
        ((-700.0, 20.0, 4.0), (30.0, 800.0), 2.1e-4),
        ((10.0, 600.0, -2.0), (-500.0, -100.0), 9e-5),
        ((-50.0, -90.0, 100.0), (4000.0, -3000.0), 3e-5),
    )
    for actor_gradient, value_gradient, lr in prescribed:
        actor.grad = actor.new_tensor(actor_gradient)
        # This is already the gradient of the weighted value objective.
        raw_critic = critic.new_tensor(value_gradient) * 0.37
        critic.grad = raw_critic.clone()
        for ref, parameter in zip(refs, (actor, critic)):
            ref.grad = parameter.grad.clone()
        torch.nn.utils.clip_grad_norm_(refs, 0.5)
        reference.param_groups[0]["lr"] = lr
        actor_optimizer.param_groups[0]["lr"] = lr
        reference.step()
        actor_origin, critic_origin = actor.detach().clone(), critic.detach().clone()
        target = critic_origin - raw_critic / 100.0
        before = quadratic(critic, 100.0, target).detach()

        optimizer.probe()
        torch.testing.assert_close(actor, actor_origin, rtol=0, atol=0)
        torch.testing.assert_close(actor.grad, refs[0].grad, rtol=2e-6, atol=1e-8)
        torch.testing.assert_close(critic.grad, raw_critic, rtol=0, atol=0)
        displacement = critic_origin - critic.detach()
        prediction = (raw_critic * displacement).sum().reshape(1)
        torch.testing.assert_close(optimizer.probe_prediction, prediction, rtol=2e-6, atol=1e-8)
        clipped_prediction = (refs[1].grad * displacement).sum().reshape(1)
        assert torch.all(prediction > 10 * clipped_prediction)
        actor_optimizer.step()
        torch.testing.assert_close(actor, refs[0], rtol=0, atol=5e-7)
        for parameter, ref, adam in ((actor, refs[0], actor_optimizer), (critic, refs[1], optimizer)):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(adam.state[parameter][key], reference.state[ref][key], rtol=2e-6, atol=1e-8)
        assert set(optimizer.state) == {critic}
        actor_updated = actor.detach().clone()
        actor_history, critic_history = history(actor_optimizer, actor), history(optimizer, critic)
        probe = quadratic(critic, 100.0, target).detach()
        optimizer.propose(before, probe)
        torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
        candidate = quadratic(critic, 100.0, target).detach()
        optimizer.finish(before, probe, candidate)
        torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
        assert_history(actor_optimizer, actor, actor_history)
        assert_history(optimizer, critic, critic_history)


def test_critic_quadratic_optimum_observes_moments_once_and_leaves_actor_step_intact():
    actor, critic = parameters()
    curvature = critic.new_tensor([9.0, 2.0])
    optimizer = CriticCurvatureAdam([actor], [critic], lr=0.4, compile=False)
    actor_optimizer = torch.optim.Adam([actor], eps=1e-5, fused=True)
    before = quadratic(critic, curvature)
    before.sum().backward()
    before = before.detach()
    actor.grad = actor.new_tensor([200.0, -300.0])
    actor_origin, origin, raw_gradient = actor.detach().clone(), critic.detach().clone(), critic.grad.clone()
    optimizer.probe()
    torch.testing.assert_close(actor, actor_origin, rtol=0, atol=0)
    probe = quadratic(critic, curvature).detach()
    direction = (origin - critic.detach()) / 0.4
    optimum = (raw_gradient * direction).sum() / (curvature * direction.square()).sum()
    observed = history(optimizer, critic)
    assert observed["step"].item() == 1
    actor_optimizer.step()
    actor_updated = actor.detach().clone()

    optimizer.propose(before, probe)
    torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
    torch.testing.assert_close(critic, origin - optimum * direction, rtol=2e-5, atol=2e-5)
    predicted = (raw_gradient * (origin - critic.detach())).sum().reshape(1)
    torch.testing.assert_close(optimizer.candidate_prediction, predicted, rtol=2e-5, atol=1e-6)
    assert_history(optimizer, critic, observed)
    candidate = quadratic(critic, curvature).detach()
    candidate_weights = critic.detach().clone()
    metrics = optimizer.finish(before, probe, candidate)
    assert metrics.shape == (1, 10)
    torch.testing.assert_close(metrics[:, 7], before.new_tensor([2.0]))
    torch.testing.assert_close(metrics[:, 9], before - candidate)
    torch.testing.assert_close(optimizer.lrs, optimum.reshape(1), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(critic, candidate_weights, rtol=0, atol=0)
    torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
    assert_history(optimizer, critic, observed)
    for name in ("lrs", "probe_lrs", "probe_prediction", "candidate_lrs", "candidate_prediction", "fallback"):
        assert getattr(optimizer, name).shape == (1,)


@pytest.mark.parametrize("finite_probe,invalid_candidate", [(True, float("nan")), (False, float("inf"))])
def test_nonfinite_candidate_restores_checked_critic_only(finite_probe, invalid_candidate):
    actor, critic = parameters(critic=(2.0, 3.0))
    optimizer = CriticCurvatureAdam([actor], [critic], lr=0.4, compile=False)
    actor_optimizer = torch.optim.Adam([actor], eps=1e-5, fused=True)
    before = quadratic(critic, 1.0)
    before.sum().backward()
    before = before.detach()
    actor.grad = actor.new_tensor([100.0, -80.0])
    origin = critic.detach().clone()
    optimizer.probe()
    probe = quadratic(critic, 1.0).detach()
    assert torch.all(probe < before)
    expected = critic.detach().clone() if finite_probe else origin
    if not finite_probe:
        probe.fill_(float("inf"))
    observed = history(optimizer, critic)
    actor_optimizer.step()
    actor_updated = actor.detach().clone()
    optimizer.propose(before, probe)
    torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
    with torch.no_grad():
        critic.fill_(invalid_candidate)
    candidate = quadratic(critic, 1.0).detach()
    metrics = optimizer.finish(before, probe, candidate)
    torch.testing.assert_close(critic, expected, rtol=0, atol=0)
    torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
    torch.testing.assert_close(metrics[:, 7], before.new_tensor([float(finite_probe)]))
    accepted = probe if finite_probe else before
    torch.testing.assert_close(metrics[:, 9], before - accepted)
    torch.testing.assert_close(optimizer.lrs, before.new_tensor([0.4 if finite_probe else 0.1]))
    assert observed["step"].item() == 1
    assert_history(optimizer, critic, observed)


@pytest.mark.parametrize("clipped", [False, True])
def test_value_loss_flattens_and_chooses_the_pessimistic_clipped_branch(clipped):
    values = torch.tensor([[1.0], [-1.0], [2.0], [0.1]], device="cuda", requires_grad=True)
    returns = values.new_tensor([2.0, -2.0, 0.0, 0.0])
    old_values = torch.zeros_like(returns)
    result = value_loss(values, returns, old_values, 0.2, clipped)
    # The clipped branch wins the first two samples; the raw branch wins the third.
    expected = 10.49 / 8 if clipped else 6.01 / 8
    torch.testing.assert_close(result, values.new_tensor(expected))
    result.backward()
    expected_gradient = [0.0, 0.0, 0.5, 0.025] if clipped else [-0.25, 0.25, 0.5, 0.025]
    torch.testing.assert_close(values.grad, values.new_tensor(expected_gradient).reshape(-1, 1))


@pytest.mark.parametrize("clipped", [False, True])
def test_checked_critic_objective_matches_ppo_without_evaluating_actor(device, monkeypatch, clipped):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-3.0, 1.0], np.float32), np.array([2.0, 5.0], np.float32)
        ),
    )
    with torch.device(device):
        agent = Agent(spaces)
    args = SimpleNamespace(clip_coef=0.2, clip_vloss=clipped, vf_coef=0.37, ent_coef=0.013, norm_adv=False)
    observations = torch.linspace(-1.0, 1.0, 12, device=device).reshape(4, 3)
    native_actions = torch.linspace(0.1, 0.9, 8, device=device).reshape(4, 2)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
    old_values = values.flatten() + values.new_tensor([0.5, -0.5, 0.5, -0.5])
    returns = values.flatten() + values.new_tensor([2.0, -2.0, 1.0, -1.0])
    advantages = values.new_tensor([2.0, -1.0, 3.0, -4.0])
    loss, metrics, objective = ppo_loss(
        agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args
    )
    assert metrics.shape == (6,)
    assert objective.shape == (1,)
    expected = args.vf_coef * value_loss(values, returns, old_values, args.clip_coef, clipped)
    torch.testing.assert_close(objective, expected.reshape(1))
    torch.testing.assert_close(loss, metrics[0] - args.ent_coef * metrics[2] + expected)

    def forbidden_actor(*args, **kwargs):
        raise AssertionError("checked value trials must not evaluate the actor")

    monkeypatch.setattr(agent.actor, "forward", forbidden_actor)
    monkeypatch.setattr(agent, "get_policy_and_value", forbidden_actor)
    checked = critic_objective(agent, observations, returns, old_values, args)
    torch.testing.assert_close(checked, objective)
    checked.sum().backward()
    assert all(parameter.grad is None for parameter in agent.actor.parameters())


def test_compiled_cycles_keep_external_actor_steps_without_recompile_or_cuda_sync():
    from torch._inductor.compile_fx import compile_fx

    compilations = 0

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return compile_fx(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    real_compile = torch.compile
    actor, critic = parameters()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=backend, **kwargs))
        optimizer = CriticCurvatureAdam([actor], [critic])
    actor_optimizer = torch.optim.Adam([actor], eps=1e-5, fused=True)

    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(8):
            actor_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            target = -0.2 * index
            before = quadratic(critic, 7.0, target)
            (before.sum() + 100 * (actor - 0.1 * index).square().sum()).backward()
            before = before.detach()
            optimizer.lrs.copy_(before.new_tensor([0.3 - 0.01 * index]))
            actor_optimizer.param_groups[0]["lr"] = 3e-4 * (1 - index / 8)
            actor_origin = actor.detach().clone()
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                optimizer.probe()
                actor_at_probe = actor.detach().clone()
                actor_optimizer.step()
                actor_updated = actor.detach().clone()
                probe = quadratic(critic, 7.0, target).detach()
                optimizer.propose(before, probe)
                actor_at_candidate = actor.detach().clone()
                candidate = quadratic(critic, 7.0, target).detach()
                metrics = optimizer.finish(before, probe, candidate)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            if index == 0:
                warm_compilations = compilations
                assert warm_compilations > 0
            assert compilations == warm_compilations
            torch.testing.assert_close(actor_at_probe, actor_origin, rtol=0, atol=0)
            torch.testing.assert_close(actor_at_candidate, actor_updated, rtol=0, atol=0)
            torch.testing.assert_close(actor, actor_updated, rtol=0, atol=0)
            expected = torch.stack((before, probe, candidate)).min(dim=0).values
            torch.testing.assert_close(quadratic(critic, 7.0, target).detach(), expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(metrics[:, 9], before - expected, rtol=2e-5, atol=2e-6)
            for adam, parameter in ((actor_optimizer, actor), (optimizer, critic)):
                assert adam.state[parameter]["step"].item() == index + 1
