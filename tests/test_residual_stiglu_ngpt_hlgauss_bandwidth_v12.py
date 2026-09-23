"""HL-Gauss frame transitions, CE gradients, and real PPO/host integration."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_scaled_asymclip_v7 import Agent as ScalarAgent
from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_hlgauss_bandwidth_v12 import (
    Agent, Args, ResidualHostMirror, categorical_value_loss, ppo_loss, target_entropy,
)
from cleanrl.shared.runtime import configure_runtime


def test_excess_ce_removes_label_floor_without_creating_a_gradient_floor():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    labels = torch.tensor([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0]], device="cuda")
    logits = labels.clamp_min(1e-30).log().requires_grad_()
    floor_fn = torch.compile(target_entropy, fullgraph=True, options={"triton.cudagraphs": False})
    floor = floor_fn(labels)
    expected = torch.tensor((np.log(2) - 0.25*np.log(0.25) - 0.75*np.log(0.75))/2, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(floor, expected)
    loss = categorical_value_loss(logits, labels)
    torch.testing.assert_close(loss - floor, torch.zeros((), device="cuda"), atol=1e-7, rtol=0)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)


def test_compiled_ppo_updates_preserve_actor_initialization_and_host_policy_parity():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    observations = np.random.default_rng(1).normal(size=(16, 17)).astype(np.float32)
    x = torch.as_tensor(observations, device="cuda")
    torch.manual_seed(1)
    scalar = ScalarAgent(spaces).cuda()
    scalar.normalize_matrices()
    with torch.no_grad():
        initial_policy = scalar.actor(x).clone()
    del scalar
    torch.manual_seed(1)
    args = Args(value_atoms=201, value_sigma_bins=1.5)
    agent = Agent(spaces, args).cuda()
    agent.normalize_matrices()
    with torch.no_grad():
        torch.testing.assert_close(agent.actor(x), initial_policy, atol=0, rtol=0)
    mirror = ResidualHostMirror(agent.actor, 16)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0024, eps=1e-5, fused=True)
    loss_fn = torch.compile(ppo_loss, fullgraph=True, options={"triton.cudagraphs": False})
    value_fn = torch.compile(agent.get_value, fullgraph=True, options={"triton.cudagraphs": False})
    native = torch.full((16, 6), 0.5, device="cuda")
    for shift in (0.0, 2.0):
        with torch.no_grad():
            alpha, beta, old_values = agent.get_policy_and_value(x)
            old_logprobs = agent.action_logprob(alpha, beta, native)
            returns = old_values.flatten() + torch.linspace(-0.5, 1.5, 16, device="cuda") + shift
            advantages = returns - old_values.flatten()
            agent.histogram.observe(returns)
            labels = agent.histogram.project(returns)
        for _ in range(2):
            loss, metrics = loss_fn(agent, x, native, old_logprobs, advantages, labels, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in agent.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            # A small, nonzero categorical gain must let the critic trunk learn immediately.
            assert agent.critic.first[0].gate.weight.grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            agent.normalize_matrices()
        mirror.refresh()
        with torch.no_grad():
            expected = agent.actor(x).cpu().numpy()
            torch.testing.assert_close(value_fn(x), agent.get_policy_and_value(x)[2])
            assert torch.isfinite(metrics).all()
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
