"""Attached JEPA gradient contracts; run CUDA tests through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action as baseline
from cleanrl import ppo_continuous_action_jepa_attached_v1 as jepa
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run these CUDA contracts through mlq"
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


def test_none_matches_baseline_loss_and_optimizer_step(envs):
    torch.manual_seed(1)
    reference = baseline.Agent(envs).cuda()
    torch.manual_seed(1)
    agent = jepa.Agent(envs, "none").cuda()
    data = batch(reference)
    expected_loss, expected_metrics = baseline.ppo_loss(reference, *data, baseline.Args())
    loss, metrics = jepa.ppo_loss(agent, *data, jepa.Args(jepa_mode="none"))
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(metrics, expected_metrics, rtol=0, atol=0)
    loss.backward()
    expected_loss.backward()
    for model in (agent, reference):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5, fused=True).step()
    actual_policy = agent.get_policy_and_value(data[0])
    expected_policy = reference.get_policy_and_value(data[0])
    for actual, expected in zip(actual_policy, expected_policy):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both"])
def test_auxiliary_reaches_both_times_and_only_selected_trunks(envs, mode):
    agent = jepa.Agent(envs, mode).cuda()
    observations = torch.randn(32, 5, device="cuda", requires_grad=True)
    next_observations = torch.randn(32, 5, device="cuda", requires_grad=True)
    native = torch.rand(32, 2, device="cuda", requires_grad=True)
    _, _, _, actor_latent, critic_latent = agent.get_policy_value_latents(observations)
    auxiliary, _ = agent.jepa_loss(actor_latent, critic_latent, next_observations, native)
    auxiliary.backward()
    # A stop-gradient on either temporal branch breaks these contracts.
    for tensor in (observations, next_observations, native):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.norm() > 0
    for name in ("actor", "critic"):
        trunk = getattr(agent, name)
        enabled = mode in (name, "both")
        grad = trunk[0].weight.grad
        if enabled:
            assert grad is not None and grad.norm() > 0
            for parameter in getattr(agent, name + "_predictor").parameters():
                assert parameter.grad is not None and parameter.grad.norm() > 0
        else:
            assert grad is None
        # The predictor branches at layer one; downstream layers belong to PPO.
        assert trunk[2].weight.grad is None and trunk[4].weight.grad is None


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_compiled_joint_loss_preserves_full_ppo_gradients(envs, mode):
    agent = jepa.Agent(envs, mode).cuda()
    data = batch(agent)
    next_observations = torch.randn_like(data[0])
    args = jepa.Args(jepa_mode=mode, jepa_coef=0.7)

    def objective(*inputs):
        return jepa.ppo_loss(agent, *inputs, args, next_observations)

    expected, expected_metrics = objective(*data)
    expected.backward()
    gradients = [parameter.grad.clone() for parameter in agent.parameters()]
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, fullgraph=True, mode="reduce-overhead")
    actual, metrics = compiled(*data)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(metrics, expected_metrics, rtol=2e-5, atol=2e-6)
    for parameter, gradient in zip(agent.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=3e-6)
        assert parameter.grad.norm() > 0
    # PPO must still reach all layers, not just the JEPA encoder.
    for trunk in (agent.actor, agent.critic):
        for index in (0, 2, 4):
            assert trunk[index].weight.grad.norm() > 0


def test_terminal_and_truncated_targets_are_not_autoreset_states(envs):
    normalizer = VectorObsNorm(3, (5,))
    reference_normalizer = VectorObsNorm(3, (5,))
    initial = np.zeros((3, 5), dtype=np.float32)
    current = normalizer.normalize(initial)
    reference_normalizer.normalize(initial)
    reset_or_next = np.array([[10.0] * 5, [-10.0] * 5, [2.0] * 5], dtype=np.float32)
    true_next = np.array([[-1.0] * 5, [1.0] * 5, [2.0] * 5], dtype=np.float32)
    terminations = np.array([True, False, False])
    truncations = np.array([False, True, False])
    infos = {"final_observation": true_next, "_final_observation": np.array([True, True, False])}
    reset_obs, transition_obs = normalizer.normalize_step(reset_or_next, terminations, truncations, infos)
    expected = reference_normalizer.normalize(true_next)
    transfer = RolloutTransfer(1, 3, (5,), "cuda", fields={"next_observations": (5,)})
    try:
        transfer.push(0, np.zeros(3), terminations, truncations, next_observations=transition_obs)
        targets = transfer.upload().fields["next_observations"].flatten(0, 1)
        agent = jepa.Agent(envs, "both").cuda()
        states = torch.as_tensor(current, device="cuda")
        native = torch.full((3, 2), 0.5, device="cuda")
        _, _, _, actor_latent, critic_latent = agent.get_policy_value_latents(states)
        actual, _ = agent.jepa_loss(actor_latent, critic_latent, targets, native)
        expected_loss, _ = agent.jepa_loss(
            actor_latent, critic_latent, torch.as_tensor(expected, device="cuda"), native,
        )
        reset_loss, _ = agent.jepa_loss(
            actor_latent, critic_latent, torch.as_tensor(reset_obs, device="cuda"), native,
        )
        torch.testing.assert_close(actual, expected_loss)
        assert not torch.isclose(actual, reset_loss)
    finally:
        transfer.close()
