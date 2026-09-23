"""CUDA contracts for action-space projection through the INTACT law."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_intact_quotient_wml_v9 as model
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Submit contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def agent_and_args():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2, -1, -0.5, -3, -1, -4], dtype=np.float32),
            np.array([1, 3, 2, 0.5, 4, 2], dtype=np.float32),
        ),
    )
    args = model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    return model.Agent(envs, args).cuda(), args


def batch(agent, rows=32):
    observations = torch.randn(rows, agent.input_dim, device="cuda")
    native = torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1
    weights, _ = model.factual_goal_weights(torch.randn(rows, device="cuda"), 0.1, "weighted")
    return observations, native, weights


def test_actor_fits_exact_executed_action_likelihood_only(agent_and_args):
    agent, args = agent_and_args
    observations, native, weights = batch(agent)
    alpha, beta = agent.direct_policy(observations)
    expected = -(weights * agent.action_logprob(alpha, beta, native)).mean()
    actual = model.actor_objective(agent, observations, native, weights, args)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    world, actor, critic = agent.parameter_groups()
    assert all(parameter.grad is None for parameter in (*world, *critic))
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in actor)


def test_no_reward_dynamics_or_value_derivative_enters_actor(agent_and_args):
    agent, args = agent_and_args
    data = batch(agent)
    before = model.actor_objective(agent, *data, args)
    gradients_before = torch.autograd.grad(before, tuple(agent.prescriber.parameters()))
    with torch.no_grad():
        for module in (agent.reward_head, agent.continuation_head, agent.dynamics, agent.pred_proj, agent.critic):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * 10)
    after = model.actor_objective(agent, *data, args)
    gradients_after = torch.autograd.grad(after, tuple(agent.prescriber.parameters()))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for first, second in zip(gradients_before, gradients_after):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_weights_act_on_executed_likelihood_not_goals(agent_and_args):
    agent, args = agent_and_args
    observations, native, weights = batch(agent)
    uniform = torch.ones_like(weights)
    weighted_loss = model.actor_objective(agent, observations, native, weights, args)
    uniform_loss = model.actor_objective(agent, observations, native, uniform, args)
    weighted_grad = torch.cat([g.flatten() for g in torch.autograd.grad(weighted_loss, tuple(agent.prescriber.parameters()))])
    uniform_grad = torch.cat([g.flatten() for g in torch.autograd.grad(uniform_loss, tuple(agent.prescriber.parameters()))])
    assert (weighted_grad - uniform_grad).norm() > 1e-8


def test_host_policy_matches_cuda_and_refreshes(agent_and_args):
    agent, _ = agent_and_args
    observations, _, _ = batch(agent, rows=16)
    host = model.HostIntactActor(agent, 16)
    for mutate in (False, True):
        if mutate:
            with torch.no_grad():
                for parameter in agent.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.01)
            host.refresh()
        with torch.no_grad():
            alpha, beta = agent.direct_policy(observations)
            logits = torch.from_numpy(host(observations.cpu().numpy()).copy()).cuda()
            host_alpha, host_beta = (torch.nn.functional.softplus(logits) + 1).chunk(2, -1)
            torch.testing.assert_close(host_alpha, alpha, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(host_beta, beta, rtol=3e-5, atol=3e-6)


def test_compiled_actor_gradient_and_full_training_owners(agent_and_args):
    agent, args = agent_and_args
    observations, native, weights = batch(agent)
    actor = lambda x, a, w: model.actor_objective(agent, x, a, w, args)
    expected = actor(observations, native, weights)
    expected_grad = torch.autograd.grad(expected, tuple(agent.prescriber.parameters()))
    compiled = torch.compile(actor, fullgraph=True, mode="reduce-overhead")
    actual = compiled(observations, native, weights)
    actual_grad = torch.autograd.grad(actual, tuple(agent.prescriber.parameters()))
    torch.testing.assert_close(actual, expected)
    for first, second in zip(actual_grad, expected_grad):
        torch.testing.assert_close(first, second, rtol=3e-4, atol=3e-6)

    following = torch.randn_like(observations)
    goals = torch.randn_like(observations)
    returns, values, rewards = [torch.randn(32, device="cuda") for _ in range(3)]
    terms = torch.zeros(32, device="cuda")
    full = torch.compile(lambda *data: model.training_loss(agent, *data, args), fullgraph=True, mode="reduce-overhead")
    loss, metrics = full(observations, native, weights, returns, values, following, goals, rewards, terms)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    ownership = []
    for group in agent.parameter_groups():
        ownership.append(set(map(id, group)))
        gradients = [parameter.grad for parameter in group if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(map(len, ownership)) == len(set.union(*ownership))
    assert set.union(*ownership) == set(map(id, agent.parameters()))
