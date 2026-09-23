"""CUDA contracts for combining future-target stopping with SF consumption."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_successor_features_v8 as reference
from cleanrl import ppo_continuous_action_jepa_successor_target_stop_v11 as model
from cleanrl.shared.runtime import configure_runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; run through mlq")


@pytest.fixture(autouse=True)
def runtime():
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
                           single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32))


def args_for(**kwargs):
    return model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8, **kwargs)


@pytest.mark.parametrize("target_gradient", ["attached", "stopped"])
def test_prediction_target_stop_is_independent_of_sigreg_and_successor_teacher(envs, target_gradient):
    args = args_for(prediction_target_gradient=target_gradient)
    agent = model.Agent(envs, args).cuda()
    observations = torch.randn(32, 17, device="cuda", requires_grad=True)
    following = torch.randn_like(observations, requires_grad=True)
    actions = torch.rand(32, 6, device="cuda", requires_grad=True)
    rewards = torch.randn(32, device="cuda", requires_grad=True)
    teachers = torch.randn(32, agent.successor_dim, device="cuda", requires_grad=True)
    components, _ = model.representation_components(agent, observations, actions, following, rewards, teachers, args)
    future_prediction_gradient, = torch.autograd.grad(components[0], following, retain_graph=True)
    assert bool(future_prediction_gradient.abs().sum() > 0) == (target_gradient == "attached")
    future_regularizer_gradient, = torch.autograd.grad(components[1], following, retain_graph=True)
    assert bool(future_regularizer_gradient.abs().sum() > 0)
    for component in (components[2], components[3]):
        gradient, = torch.autograd.grad(component, observations, retain_graph=True)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
    components.sum().backward()
    assert rewards.grad is None and actions.grad is None and teachers.grad is None
    assert bool(observations.grad.abs().sum() > 0)


@pytest.mark.parametrize("successor_targets", ["reward", "joint"])
def test_attached_prediction_control_matches_v8_losses_gradients_and_value(envs, successor_targets):
    args = args_for(prediction_target_gradient="attached", successor_targets=successor_targets)
    torch.manual_seed(7)
    old = reference.Agent(envs, args).cuda()
    torch.manual_seed(7)
    new = model.Agent(envs, args).cuda()
    observations = torch.randn(32, 17, device="cuda")
    following = torch.randn_like(observations)
    actions = torch.rand(32, 6, device="cuda")
    rewards = torch.randn(32, device="cuda")
    teachers = torch.randn(32, new.successor_dim, device="cuda")
    torch.manual_seed(13)
    old_components, _ = reference.representation_components(old, observations, actions, following, rewards, teachers, args)
    old_rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(13)
    new_components, _ = model.representation_components(new, observations, actions, following, rewards, teachers, args)
    torch.testing.assert_close(torch.cuda.get_rng_state(), old_rng, rtol=0, atol=0)
    torch.testing.assert_close(new_components, old_components, rtol=0, atol=0)
    old_components.sum().backward()
    new_components.sum().backward()
    for old_parameter, new_parameter in zip(old.parameters(), new.parameters()):
        if old_parameter.grad is None:
            assert new_parameter.grad is None
        else:
            torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(new.get_value(observations), old.get_value(observations), rtol=0, atol=0)


@pytest.mark.parametrize("successor_targets", ["reward", "joint"])
def test_combined_stopped_target_compiled_ppo_and_world_update(envs, successor_targets):
    args = args_for(prediction_target_gradient="stopped", successor_targets=successor_targets)
    agent = model.Agent(envs, args).cuda()
    observations = torch.randn(512, 17, device="cuda")
    following = torch.randn_like(observations)
    actions = torch.rand(512, 6, device="cuda") * 0.8 + 0.1
    rewards = torch.randn(512, device="cuda")
    teachers = torch.randn(512, agent.successor_dim, device="cuda")
    advantages = torch.randn(512, device="cuda")
    with torch.no_grad():
        latent = agent.encoder(observations)
        cached = agent.critic_features(observations, latent).clone()
        alpha, beta, values = agent.get_policy_and_value(observations, critic_inputs=cached)
        oldlog = agent.action_logprob(alpha, beta, actions)
        oldvalues = values.flatten().clone()
        returns = oldvalues + torch.randn_like(oldvalues)

    def objective(obs, next_obs, native, real_rewards, fixed_targets, inputs):
        outputs = agent.get_policy_value_latents(obs, critic_inputs=inputs)
        ppo, _ = model._policy_objective(agent, outputs[:3], native, oldlog, advantages, returns, oldvalues, args)
        components, _ = model.representation_components(agent, obs, native, next_obs, real_rewards,
                                                         fixed_targets, args, latent=outputs[3])
        return ppo, components

    groups = agent.parameter_groups()
    optimizers = tuple(torch.optim.Adam(parameters, lr=3e-4) for parameters in groups)
    compiled = torch.compile(objective, fullgraph=True, options={"triton.cudagraphs": False})
    policy, components = compiled(observations, following, actions, rewards, teachers, cached)
    policy.backward(retain_graph=True)
    for group in groups[1:]:
        assert all(parameter.grad is None or bool((parameter.grad == 0).all()) for parameter in group)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    components.sum().backward()
    before = agent.get_successor(observations).detach().clone()
    norms = model.optimizer_step(agent, optimizers, args, policy_step=False, parameters=groups)
    assert bool(torch.isfinite(norms).all())
    assert not torch.equal(agent.get_successor(observations), before)
