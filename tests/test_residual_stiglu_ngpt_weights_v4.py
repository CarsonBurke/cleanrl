"""Projection axes and CUDA/host policy agreement are training invariants."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_weights_v4 import Agent, ResidualHostMirror
from cleanrl.shared.runtime import configure_runtime


def test_compiled_ngpt_projection_after_adam_preserves_axes_biases_and_behavior():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(spaces).cuda()
    project = torch.compile(agent.normalize_matrices, fullgraph=True,
                            options={"triton.cudagraphs": False})
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0024)
    observations = np.random.default_rng(1).normal(size=(16, 17)).astype(np.float32)
    x = torch.as_tensor(observations, device="cuda")
    mirror = ResidualHostMirror(agent.actor, 16)
    for _ in range(2):
        biases = [trunk.head[0].bias.detach().clone() for trunk in (agent.actor, agent.critic)]
        project()
        for trunk, bias in zip((agent.actor, agent.critic), biases):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1), (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    norms = torch.linalg.vector_norm(weight, dim=dim)
                    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-6, atol=1e-6)
            norms = torch.linalg.vector_norm(trunk.head[0].weight, dim=1)
            torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(trunk.head[0].bias, bias, rtol=0, atol=0)
        mirror.refresh()
        with torch.no_grad():
            expected = agent.actor(x).cpu().numpy()
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
        optimizer.zero_grad(set_to_none=True)
        loss = (agent.actor(x) - 1).square().mean() + (agent.critic(x) - 1).square().mean()
        loss.backward()
        optimizer.step()
