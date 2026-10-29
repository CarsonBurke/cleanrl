"""Surprise-tilt contracts; run CUDA work exclusively through mlq."""
from types import SimpleNamespace

import torch

from cleanrl.shared.ppo_loop import compute_gae
from cleanrl.tilt.ppo_continuous_action_sepclip_surprise_tilt_v4 import (
    ConsequenceModel, consequence_loss, considered_gains, natural_step, residual_loss,
)


def linear_agent(weight, value_weight):
    def consequences(observations, actions):
        statistics = torch.cat((actions.log(), torch.log1p(-actions)), dim=-1)
        return observations, (weight * statistics).sum(-1), torch.zeros_like(actions[..., 0])

    return SimpleNamespace(consequence=SimpleNamespace(consequences=consequences),
                           critic=lambda observations: observations @ value_weight[:, None])


def test_considered_gains_return_the_taken_actions_own_effect_and_the_exact_slope():
    torch.manual_seed(1)
    states, dims = 8, 3
    alpha, beta = (1.0 + 4.0 * torch.rand(2, states, dims, device="cuda", dtype=torch.float64)).unbind(0)
    weight = torch.randn(states, 2 * dims, device="cuda")
    observations, taken = torch.randn(states, 5, device="cuda"), torch.rand(states, dims, device="cuda")
    divisor = 0.5 + torch.rand(states, device="cuda")
    agent = linear_agent(weight, torch.randn(5, device="cuda"))
    own, slope, linear_share = considered_gains(agent, observations, alpha, beta, taken, divisor, 0.99, 200_000)
    # The state-only value cancels; the gain is linear in sufficient statistics, so the natural
    # step is the reward weight in normalized-reward units.
    expected = (weight / divisor[:, None]).double()
    step_alpha, step_beta = natural_step(alpha, beta, slope)
    torch.testing.assert_close(torch.cat((step_alpha, step_beta), dim=-1), expected,
                               atol=0.03 * float(expected.abs().max()), rtol=0.0)
    statistics = torch.cat((taken.log(), torch.log1p(-taken)), dim=-1).double()
    mean = torch.cat((torch.digamma(alpha) - torch.digamma(alpha + beta),
                      torch.digamma(beta) - torch.digamma(alpha + beta)), dim=-1)
    torch.testing.assert_close(own, (expected * (statistics - mean)).sum(-1), atol=0.03 * float(own.abs().max()), rtol=0.0)
    assert linear_share > 0.99


def test_reward_substitution_in_gae_leaves_exactly_the_surprise_sum():
    torch.manual_seed(1)
    steps, envs, discount = 64, 4, 0.99 * 0.95
    rewards, values, bootstrap, own = (torch.randn(steps, envs, device="cuda") for _ in range(4))
    terminations = (torch.rand(steps, envs, device="cuda") < 0.05).float()
    truncations = (torch.rand(steps, envs, device="cuda") < 0.05).float() * (1.0 - terminations)
    rest = (values, terminations, truncations, bootstrap, torch.randn(envs, device="cuda"), 0.99, 0.95)
    advantages, surprise = compute_gae(rewards, *rest)[0], compute_gae(rewards - own, *rest)[0]
    # advantage - surprise is the discounted sum of own effects: this step's plus later luck.
    trace, running = torch.zeros_like(own), torch.zeros(envs, device="cuda")
    for step in reversed(range(steps)):
        running = own[step] + discount * (1.0 - torch.maximum(terminations[step], truncations[step])) * running
        trace[step] = running
    torch.testing.assert_close(advantages - surprise, trace, atol=1e-4, rtol=1e-4)


def test_losses_use_stored_units_and_a_state_baseline():
    torch.manual_seed(1)
    model = ConsequenceModel(5, 3, 32).cuda()
    observations, actions = torch.randn(64, 5, device="cuda"), torch.rand(64, 3, device="cuda")
    changes, rewards = 3.0 * torch.randn(64, 5, device="cuda"), 40.0 * torch.randn(64, device="cuda")
    valid = (torch.arange(64, device="cuda") % 4 != 0).float()
    model.calibrate(changes, rewards, valid)
    torch.testing.assert_close(model.scale[-1], rewards.square().mean().sqrt())
    agent = SimpleNamespace(consequence=model)
    targets = torch.cat((changes, rewards[:, None], torch.zeros(64, 1, device="cuda")), dim=-1)
    corrupted = targets.clone()
    corrupted[valid == 0, :5] = 1e6
    loss = consequence_loss(agent, observations, actions, targets, valid)
    torch.testing.assert_close(consequence_loss(agent, observations, actions, corrupted, valid)[0], loss[0])
    assert 0.5 < loss[1][1] < 2.0  # raw rewards are standardized inside the loss

    scores, surprise = torch.randn(64, 6, device="cuda"), torch.randn(64, device="cuda")
    output = torch.randn(64, 7, device="cuda")
    fitted = SimpleNamespace(residual=lambda _: output, critic=lambda x: x[:, :1])
    metrics = residual_loss(fitted, observations, scores, surprise, observations[:, 0], SimpleNamespace(vf_coef=0.5))[1]
    torch.testing.assert_close(metrics[0], ((surprise - output[:, 0] - (output[:, 1:] * scores).sum(-1)) ** 2).mean())
    assert metrics[1] == 0.0
