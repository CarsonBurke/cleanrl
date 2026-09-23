"""Policy ratio asymmetry must not widen the critic's clipping radius."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_scaled_asymclip_v7 import (
    Agent, Args, ppo_loss,
)
from cleanrl.shared.runtime import configure_runtime


def test_compiled_asymmetric_objective_gradients_and_independent_value_clip():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(spaces).cuda()
    agent.normalize_matrices()
    observations = torch.randn(4, 17, device="cuda")
    native_actions = torch.full((4, 6), 0.5, device="cuda")
    ratios = torch.tensor([1.24, 1.40, 0.75, 0.85], device="cuda")
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0], device="cuda")
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        logprobs = agent.action_logprob(alpha, beta, native_actions)
        # The new critic prediction is perfect; the old value is one unit below.
        returns = values.flatten().clone()
        old_values = returns - 1.0
    old_logprobs = (logprobs - ratios.log()).requires_grad_()
    args = Args(norm_adv=False, clip_coef=0.2, clip_coef_upper=0.28, clip_vloss=True)
    loss_fn = torch.compile(ppo_loss, fullgraph=True, options={"triton.cudagraphs": False})
    loss, metrics = loss_fn(
        agent, observations, native_actions, old_logprobs,
        advantages, returns, old_values, args,
    )
    loss.backward()
    # Positive advantages stop improving above 1.28; negative ones below 0.80.
    expected_pg = torch.tensor((-1.24 - 1.28 + 0.80 + 0.85) / 4, device="cuda")
    torch.testing.assert_close(metrics[0], expected_pg)
    torch.testing.assert_close(
        old_logprobs.grad, torch.tensor([1.24, 0.0, 0.0, -0.85], device="cuda") / 4,
    )
    torch.testing.assert_close(metrics[5], torch.tensor(0.5, device="cuda"))
    # Value clipping still permits only +0.2, leaving an error of 0.8.
    torch.testing.assert_close(metrics[1], torch.tensor(0.5 * 0.8**2, device="cuda"))
    torch.testing.assert_close(loss, expected_pg + args.vf_coef * metrics[1])
