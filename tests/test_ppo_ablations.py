"""CPU contracts for the Beta PPO clip-higher and PopArt ablations."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl import ppo_continuous_action as ppo
from cleanrl import ppo_continuous_action_clip_higher_v1 as clip_higher
from cleanrl import ppo_continuous_action_popart_v1 as popart
from cleanrl.shared.sampling import sample_beta_actions


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-1.0, -2.0], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
        ),
    )


def test_clip_higher_defaults_are_asymmetric_and_value_clip_stays_symmetric():
    args = clip_higher.Args()
    assert (args.clip_coef, args.clip_coef_low, args.clip_coef_high) == (0.2, 0.2, 0.28)
    assert args.clip_vloss is True


def test_clip_higher_allows_ratios_between_symmetric_and_high_clip():
    torch.manual_seed(0)
    agent = clip_higher.Agent(_spaces())
    n = 8
    observations = torch.zeros(n, 11)
    alpha, beta, _ = agent.get_policy_and_value(observations)
    native, _ = sample_beta_actions(alpha, beta, agent.action_low, agent.action_high)
    old_logprob = agent.action_logprob(alpha, beta, native).detach()
    # Ratio 1.25 is clipped by symmetric 0.2 and unclipped by clip-higher 0.28.
    shifted_logprob = old_logprob - float(np.log(1.25))
    advantages = torch.ones(n)
    zeros = torch.zeros(n)
    high_args = SimpleNamespace(
        clip_coef=0.2, clip_coef_low=0.2, clip_coef_high=0.28,
        clip_vloss=False, norm_adv=False, ent_coef=0.0, vf_coef=0.0,
    )
    base_args = SimpleNamespace(
        clip_coef=0.2, clip_vloss=False, norm_adv=False, ent_coef=0.0, vf_coef=0.0,
    )
    high_loss, high_metrics = clip_higher.ppo_loss(
        agent, observations, native, shifted_logprob, advantages, zeros, zeros, high_args,
    )
    sym_loss, _ = ppo.ppo_loss(
        agent, observations, native, shifted_logprob, advantages, zeros, zeros, base_args,
    )
    assert high_metrics[5].item() == 0.0
    assert high_loss.item() < sym_loss.item()


def test_popart_rescale_preserves_unnormalized_values_and_adam_moments():
    torch.manual_seed(0)
    agent = popart.Agent(_spaces())
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    observations = torch.randn(16, 11)
    agent.get_value(observations).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    before = agent.get_value(observations).detach().clone()
    weight_state = optimizer.state[agent.critic[-1].weight]
    bias_state = optimizer.state[agent.critic[-1].bias]
    old_avg = weight_state["exp_avg"].clone()
    old_avg_sq = weight_state["exp_avg_sq"].clone()
    old_bias_avg = bias_state["exp_avg"].clone()

    returns = torch.linspace(-8.0, 12.0, 32)
    old_std = agent.popart_std.clone()
    normalized = agent.update_popart(returns, rate=0.5, std_min=1e-2, std_max=1e6, optimizer=optimizer)
    scale = old_std / agent.popart_std

    torch.testing.assert_close(agent.get_value(observations), before, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight_state["exp_avg"], old_avg * scale)
    torch.testing.assert_close(weight_state["exp_avg_sq"], old_avg_sq * scale.square())
    torch.testing.assert_close(bias_state["exp_avg"], old_bias_avg * scale)
    torch.testing.assert_close(normalized, (returns - agent.popart_mean) / agent.popart_std)
    assert agent.popart_std.item() != old_std.item()
