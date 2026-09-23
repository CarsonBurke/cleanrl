"""CUDA contracts for latent depth, attached conditioning, and loss balance; use mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_attached_v1 as v1
from cleanrl import ppo_continuous_action_jepa_latent_balance_v2 as jepa
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )


def batch(agent):
    observations = torch.randn(32, 5, device="cuda")
    native = torch.rand(32, 2, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return (
        observations, native, old_logprobs, torch.randn(32, device="cuda"),
        values.flatten() + torch.randn(32, device="cuda"), values.flatten(),
    )


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_first_layer_preserves_v1_objective_and_gradients(envs, mode):
    torch.manual_seed(1)
    reference = v1.Agent(envs, mode).cuda()
    torch.manual_seed(1)
    agent = jepa.Agent(envs, mode, "first").cuda()
    data = batch(agent)
    next_obs = torch.randn_like(data[0])
    expected, expected_metrics = v1.ppo_loss(reference, *data, v1.Args(jepa_mode=mode), next_obs)
    actual, actual_metrics = jepa.ppo_loss(agent, *data, jepa.Args(jepa_mode=mode, jepa_layer="first"), next_obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for parameter, reference_parameter in zip(agent.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("layer", ["first", "final"])
@pytest.mark.parametrize("mode", ["actor", "critic", "both"])
def test_auxiliary_reaches_selected_depth_at_both_times(envs, layer, mode):
    agent = jepa.Agent(envs, mode, layer).cuda()
    current = torch.randn(32, 5, device="cuda", requires_grad=True)
    following = torch.randn(32, 5, device="cuda", requires_grad=True)
    native = torch.rand(32, 2, device="cuda")
    _, _, value, actor_latent, critic_latent = agent.get_policy_value_latents(current)
    actor_loss, critic_loss, _ = agent.jepa_losses(actor_latent, critic_latent, following, native, value)
    (actor_loss + critic_loss).backward()
    assert current.grad.norm() > 0 and following.grad.norm() > 0
    for name in ("actor", "critic"):
        trunk = getattr(agent, name)
        enabled = mode in (name, "both")
        if enabled:
            assert trunk[0].weight.grad.norm() > 0
            if layer == "final":
                assert trunk[2].weight.grad.norm() > 0
            else:
                assert trunk[2].weight.grad is None
        else:
            assert all(parameter.grad is None for parameter in trunk.parameters())
        assert trunk[4].weight.grad is None


@pytest.mark.parametrize("projection", ["concat", "adaln"])
def test_conditioning_uses_chosen_action_and_attached_current_value(envs, projection):
    agent = jepa.Agent(envs, "both", "final", projection, "value").cuda()
    current = torch.randn(32, 5, device="cuda")
    following = torch.randn_like(current)
    native = torch.rand(32, 2, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    # AdaLN-zero starts with closed gates, as in le-wm. One update opens them.
    _, _, value, actor_latent, critic_latent = agent.get_policy_value_latents(current)
    actor_loss, critic_loss, _ = agent.jepa_losses(actor_latent, critic_latent, following, native, value)
    (actor_loss + critic_loss).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    native.grad = None
    _, _, value, actor_latent, critic_latent = agent.get_policy_value_latents(current)
    value.retain_grad()
    actor_loss, critic_loss, _ = agent.jepa_losses(actor_latent, critic_latent, following, native, value)
    (actor_loss + critic_loss).backward()
    assert native.grad.norm() > 0
    assert value.grad.norm() > 0
    assert agent.critic[4].weight.grad.norm() > 0
    with torch.no_grad():
        actor_prediction = agent.actor_predictor(actor_latent, 2 * native - 1)
        changed_action = agent.actor_predictor(actor_latent, 1 - 2 * native)
        critic_prediction = agent.critic_predictor(critic_latent, value)
        changed_value = agent.critic_predictor(critic_latent, value + 1)
    assert not torch.allclose(actor_prediction, changed_action)
    assert not torch.allclose(critic_prediction, changed_value)


@pytest.mark.parametrize("mode,projection,condition", [
    ("both", "concat", "action"),
    ("both", "adaln", "value"),
    ("actor", "adaln", "action"),
    ("critic", "concat", "value"),
    ("none", "concat", "action"),
])
def test_compiled_objective_and_diagnostics_preserve_update(envs, mode, projection, condition):
    agent = jepa.Agent(envs, mode, "final", projection, condition).cuda()
    data = batch(agent)
    next_obs = torch.randn_like(data[0])
    args = jepa.Args(jepa_mode=mode, jepa_projection=projection, jepa_critic_condition=condition)

    def components(*inputs):
        return jepa.loss_components(agent, *inputs, args, next_obs)[0]

    def objective(*inputs):
        return jepa.ppo_loss(agent, *inputs, args, next_obs)

    expected, expected_metrics = objective(*data)
    expected.backward()
    expected_gradients = [parameter.grad.clone() for parameter in agent.parameters()]
    compiled_components = torch.compile(components, fullgraph=True, options={"triton.cudagraphs": False})
    measured = compiled_components(*data)
    metrics = jepa.gradient_balance(agent, measured)
    assert all(torch.isfinite(value) for value in metrics.values())
    # Diagnostics must preserve pre-existing gradients exactly, not silently overwrite them.
    for parameter, gradient in zip(agent.parameters(), expected_gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=True)
    actual, actual_metrics = compiled(*data)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-5, atol=2e-6)
    for parameter, gradient in zip(agent.parameters(), expected_gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=3e-6)
    for trunk in (agent.actor, agent.critic):
        for index in (0, 2, 4):
            assert trunk[index].weight.grad.norm() > 0


def test_balance_accounts_for_coefficients_and_detects_zero_signal(envs):
    agent = jepa.Agent(envs, "both", "final", "concat", "value").cuda()
    data = batch(agent)
    next_obs = torch.randn_like(data[0])

    def measure(coef, vf_coef):
        components, _ = jepa.loss_components(
            agent, *data, jepa.Args(jepa_coef=coef, vf_coef=vf_coef), next_obs,
        )
        return components, jepa.gradient_balance(agent, components)

    components, normal = measure(1.0, 0.5)
    _, stronger_auxiliary = measure(3.0, 0.5)
    _, weaker_value = measure(1.0, 0.125)
    _, disabled = measure(0.0, 0.0)
    for branch in ("actor", "critic"):
        ratio = f"balance/{branch}_jepa_to_ppo_grad_ratio"
        torch.testing.assert_close(stronger_auxiliary[ratio], 3 * normal[ratio])
        assert disabled[f"balance/{branch}_jepa_grad_norm"] == 0
        assert torch.isfinite(disabled[ratio])
    torch.testing.assert_close(
        weaker_value["balance/critic_jepa_to_ppo_grad_ratio"],
        4 * normal["balance/critic_jepa_to_ppo_grad_ratio"],
    )
    # Value-conditioned JEPA also owns the critic output head: include it in the norm.
    for index, name in enumerate(("actor", "critic")):
        trunk = getattr(agent, name)
        parameters = tuple(trunk[:4].parameters()) if name == "actor" else tuple(trunk.parameters())
        primary = torch.cat([grad.flatten() for grad in torch.autograd.grad(components[index], parameters, retain_graph=True)])
        auxiliary = torch.cat([grad.flatten() for grad in torch.autograd.grad(components[index + 2], parameters, retain_graph=True)])
        torch.testing.assert_close(normal[f"balance/{name}_ppo_grad_norm"], primary.norm())
        torch.testing.assert_close(normal[f"balance/{name}_jepa_grad_norm"], auxiliary.norm())
        torch.testing.assert_close(
            normal[f"balance/{name}_grad_cosine"],
            torch.dot(primary, auxiliary) / (primary.norm() * auxiliary.norm()),
        )
    assert all(parameter.grad is None for parameter in agent.parameters())
