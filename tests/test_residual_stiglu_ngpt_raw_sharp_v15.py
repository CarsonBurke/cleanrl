"""Sharp labels must preserve means between nodes, not only at large values."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_sharp_v15 import (
    Agent, Args, make_target_projector,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.mark.parametrize("atoms", [51, 101])
def test_sharp_projection_preserves_small_and_large_raw_means(atoms):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args(value_atoms=atoms, value_sigma_bins=0.25)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(spaces, args).cuda()
    project = torch.compile(make_target_projector(args, agent.histogram), fullgraph=True,
                            options={"triton.cudagraphs": False})
    targets = torch.tensor([-25000., -19000., -1000., -1., -0.1, -0.05, -0.02,
                            0., 0.02, 0.05, 0.1, 1., 100., 1000., 5000., 15000., 19000., 25000.],
                           device="cuda")
    with torch.no_grad():
        labels = project(targets)
        torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets), atol=1e-6, rtol=1e-6)
        decoded = agent.histogram.probs_to_scalar(labels)
        torch.testing.assert_close(decoded, targets.clamp(args.value_min, args.value_max),
                                   atol=2e-6, rtol=2e-6)
        mixture = (labels[12] + labels[13]) / 2
        torch.testing.assert_close(agent.histogram.probs_to_scalar(mixture),
                                   torch.tensor(550., device="cuda"), atol=2e-6, rtol=2e-6)
