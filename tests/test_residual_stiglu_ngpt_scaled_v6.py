"""Normalized residuals must preserve directions and match the rollout policy."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_scaled_v6 import (
    Agent, ResidualHostMirror, ResidualMLP,
)
from cleanrl.shared.runtime import configure_runtime


def test_compiled_normalized_policy_matches_host_after_projected_adam_updates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(spaces).cuda()
    agent.normalize_matrices()
    mirror = ResidualHostMirror(agent.actor, 16)
    actor = torch.compile(agent.actor, fullgraph=True, options={"triton.cudagraphs": False})
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0024)
    observations = np.random.default_rng(1).normal(size=(16, 17)).astype(np.float32)
    x = torch.as_tensor(observations, device="cuda")
    zeros = np.zeros_like(observations)
    for _ in range(3):
        with torch.no_grad():
            agent.actor.readout_gain.mul_(torch.linspace(0.8, 1.2, 12, device="cuda"))
        mirror.refresh()
        with torch.no_grad():
            expected = actor(x).cpu().numpy()
            expected_zero = actor(torch.zeros_like(x)).cpu().numpy()
        assert np.isfinite(expected).all() and np.isfinite(expected_zero).all()
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(mirror(zeros), expected_zero, rtol=2e-4, atol=2e-5)
        # A zero-input call must not corrupt the permanent normalization buffers.
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
        optimizer.zero_grad(set_to_none=True)
        loss = (actor(x) - 1).square().mean() + (agent.critic(x) - 1).square().mean()
        loss.backward()
        for parameter in agent.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        agent.normalize_matrices()


def test_normalized_directions_retain_scale_and_readout_exceeds_unit_bound():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    trunk = ResidualMLP(17, 64, output_std=1.0).cuda()
    x = torch.randn(16, 17, device="cuda")
    with torch.no_grad():
        # Identity readout makes residual geometry directly observable.
        trunk.head[0].weight.copy_(torch.eye(64, device="cuda"))
        trunk.head[0].bias.zero_()
        original = trunk(x)
        torch.testing.assert_close(original.norm(dim=-1), torch.full((16,), 8.0, device="cuda"))
        trunk.first[0].down.weight.mul_(3.0)
        trunk.second[0].down.weight.mul_(7.0)
        torch.testing.assert_close(trunk(x), original, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(trunk(torch.zeros_like(x)), torch.zeros_like(original))

        # A unit-norm readout aligned with the hidden direction is not capped at 1.
        trunk.head[0].weight[0].copy_(original[0] / 8.0)
        torch.testing.assert_close(trunk(x)[0, 0], torch.tensor(8.0, device="cuda"))
        trunk.readout_gain[0].fill_(3.0)
        torch.testing.assert_close(trunk(x)[0, 0], torch.tensor(24.0, device="cuda"))
