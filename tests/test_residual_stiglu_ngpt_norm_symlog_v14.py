"""Mean matching must remain valid in small normalized-return units."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_norm_symlog_v14 import (
    Agent, Args, make_target_projector,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.mark.parametrize("value_target", ["hlgauss", "twohot"])
def test_small_support_means_include_near_boundary_targets(value_target):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    args = Args(value_target=value_target, value_atoms=501, value_sigma_bins=1.5,
                value_min=-10.0, value_max=10.0)
    agent = Agent(spaces, args).cuda()
    project = torch.compile(make_target_projector(args, agent.histogram), fullgraph=True,
                            options={"triton.cudagraphs": False})
    targets = torch.tensor([-12., -9.99999, -9.99, -5., -3., -0.01, 0., 0.01,
                            1., 3., 9., 9.99, 9.99999, 12.], device="cuda")
    with torch.no_grad():
        labels = project(targets)
        torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets), atol=1e-6, rtol=1e-6)
        decoded = agent.histogram.probs_to_scalar(labels)
        torch.testing.assert_close(decoded, targets.clamp(-10, 10), atol=2e-6, rtol=2e-6)
        mixture = (labels[8] + labels[10]) / 2
        torch.testing.assert_close(agent.histogram.probs_to_scalar(mixture),
                                   torch.tensor(5., device="cuda"), atol=2e-6, rtol=2e-6)
