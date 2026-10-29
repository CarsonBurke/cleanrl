"""Consequence-tilt contracts; run CUDA work exclusively through mlq."""
from types import SimpleNamespace

import torch

from cleanrl.tilt.ppo_continuous_action_sepclip_consequence_tilt_v3 import (
    ConsequenceModel, beta_mean_statistics, consequence_loss, considered_slope, natural_step,
)


def test_considered_slope_recovers_the_tilt_of_a_gain_linear_in_sufficient_statistics():
    torch.manual_seed(1)
    states, dims = 8, 3
    alpha, beta = (1.0 + 4.0 * torch.rand(2, states, dims, device="cuda", dtype=torch.float64)).unbind(0)
    weight = torch.randn(states, 2 * dims, device="cuda")

    def consequences(observations, actions):
        statistics = torch.cat((actions.log(), torch.log1p(-actions)), dim=-1)
        return observations, (weight * statistics).sum(-1), torch.zeros_like(actions[..., 0])

    agent = SimpleNamespace(consequence=SimpleNamespace(consequences=consequences),
                            critic=lambda observations: observations[..., :1] * 0.0)
    slope, linear_share = considered_slope(agent, torch.zeros(states, 5, device="cuda"), alpha, beta, 0.99, 800_000)
    # u . z reproduces the centered gain, so the natural step is the weight up to the gain's RMS scale.
    step_alpha, step_beta = natural_step(alpha, beta, slope)
    recovered = torch.cat((step_alpha, step_beta), dim=-1)
    scale = (recovered * weight).sum() / weight.double().square().sum()
    torch.testing.assert_close(recovered, scale * weight.double(), atol=0.03 * float(scale * weight.abs().max()), rtol=0.0)
    assert linear_share > 0.99


def test_standardized_consequences_round_trip_and_mask_episode_boundaries():
    torch.manual_seed(1)
    model = ConsequenceModel(5, 3, 32).cuda()
    observations, actions = torch.randn(64, 5, device="cuda"), torch.rand(64, 3, device="cuda")
    changes, rewards = 3.0 * torch.randn(64, 5, device="cuda"), 0.1 * torch.randn(64, device="cuda")
    valid = (torch.arange(64, device="cuda") % 4 != 0).float()
    standard_changes, standard_rewards = model.standardize(changes, rewards, valid)
    torch.testing.assert_close((standard_changes.square() * valid[:, None]).sum(0) / valid.sum(), torch.ones(5, device="cuda"))
    next_observations, reward, terminal = model.consequences(observations, actions)
    prediction = model(observations, actions)
    torch.testing.assert_close(next_observations, observations + prediction[:, :5] * model.scale[:5])
    torch.testing.assert_close(reward, prediction[:, 5] * model.scale[5])
    targets = torch.cat((standard_changes, standard_rewards[:, None], torch.zeros(64, 1, device="cuda")), dim=-1)
    agent = SimpleNamespace(consequence=model)
    corrupted = targets.clone()
    corrupted[valid == 0, :5] = 1e6
    torch.testing.assert_close(consequence_loss(agent, observations, actions, corrupted, valid)[0],
                               consequence_loss(agent, observations, actions, targets, valid)[0])
