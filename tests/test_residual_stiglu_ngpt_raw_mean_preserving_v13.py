"""Raw target means must survive categorical projection and conditional mixing."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_mean_preserving_v13 import (
    Agent, Args, make_target_projector,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.mark.parametrize("value_target", ["hlgauss", "twohot"])
def test_compiled_labels_preserve_raw_means_and_conditional_mixtures(value_target):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    args = Args(value_target=value_target)
    agent = Agent(spaces, args).cuda()
    project = torch.compile(make_target_projector(args, agent.histogram), fullgraph=True,
                            options={"triton.cudagraphs": False})
    targets = torch.tensor([-25000., -1000., -300., -0.1, 0., 0.1,
                            100., 300., 900., 1000., 25000.], device="cuda")
    with torch.no_grad():
        labels = project(targets)
        decoded = agent.histogram.probs_to_scalar(labels)
        torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets), atol=1e-6, rtol=1e-6)
        # Includes overflow clipping, zero-crossing and nonlinear raw support.
        torch.testing.assert_close(decoded, targets.clamp(args.value_min, args.value_max),
                                   atol=0.002, rtol=2e-6)
        mixture = (labels[6] + labels[8]) / 2
        torch.testing.assert_close(agent.histogram.probs_to_scalar(mixture),
                                   torch.tensor(500., device="cuda"), atol=0.002, rtol=2e-6)
