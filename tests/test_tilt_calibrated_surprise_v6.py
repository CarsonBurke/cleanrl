"""Calibrated-surprise contracts; run CUDA work exclusively through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.awr.ppo_continuous_action_ppo_sepclip_control_v1 import ppo_loss
from cleanrl.tilt.ppo_continuous_action_sepclip_calibrated_surprise_v6 import (
    Agent, Args, calibrate_own, considered_gains, surprise_ppo_loss,
)


def test_calibration_recovers_the_trust_each_part_deserves_and_leaves_the_rest_as_surprise():
    torch.manual_seed(1)
    steps, envs, discount = 256, 8, 0.99 * 0.95
    own = torch.randn(steps, envs, 2, device="cuda")
    continuing = (torch.rand(steps, envs, device="cuda") > 0.02).float()
    truth = torch.tensor([0.9, 0.05], device="cuda")
    sums, running = torch.empty(steps, envs, 2, device="cuda"), torch.zeros(envs, 2, device="cuda")
    for step in reversed(range(steps)):
        running = own[step] + discount * continuing[step, :, None] * running
        sums[step] = running
    noise = 0.3 * torch.randn(steps, envs, device="cuda")
    trust, surprise, explained = calibrate_own(sums @ truth + noise, own, continuing, discount)
    torch.testing.assert_close(trust, truth, atol=0.02, rtol=0.0)
    torch.testing.assert_close(surprise, sums @ truth + noise - sums @ trust, atol=1e-4, rtol=1e-4)
    assert 0.0 < 1.0 - explained < 0.2
    # A part unrelated to the advantage earns no trust and removes nothing.
    trust, surprise, explained = calibrate_own(noise, own, continuing, discount)
    assert trust.abs().max() < 0.02 and explained < 0.01


def fixture(batch=256, considered=8):
    torch.manual_seed(1)
    env = SimpleNamespace(single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
                          single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(env, 64).cuda()
    agent.normalize_matrices()
    observations = torch.randn(batch, 17, device="cuda")
    native = 0.1 + 0.8 * torch.rand(batch, 6, device="cuda")
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old = agent.action_logprob(alpha, beta, native) - torch.tensor([0.5, 0.9, 1.1, 1.5], device="cuda").repeat(batch // 4).log()
    return agent, observations, native, old, alpha, beta, values.flatten(), considered


def test_considered_gains_are_centred_on_the_alternatives_and_shaped_per_state():
    agent, observations, native, _, alpha, beta, _, considered = fixture()
    divisor = 0.5 + torch.rand(len(observations), device="cuda")
    actions, logprobs, own, gains = considered_gains(agent, observations, alpha, beta, native, divisor, 0.99, considered, chunk=3)
    assert actions.shape == (256, considered, 6) and logprobs.shape == (256, considered)
    assert own.shape == (256, 2) and gains.shape == (256, considered, 2)
    torch.testing.assert_close(gains.mean(1), torch.zeros_like(own), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(logprobs, agent.action_logprob(alpha[:, None], beta[:, None], actions))
    _, reward, terminal = agent.consequence.consequences(observations, native)
    taken = torch.stack((reward / divisor, 0.99 * (1.0 - terminal) * agent.critic(
        agent.consequence.consequences(observations, native)[0]).squeeze(-1)), dim=-1)
    # own + baseline is the taken action's gain; the baseline is the considered mean.
    baseline = taken - own
    torch.testing.assert_close((gains * (considered - 1.0) / considered + baseline[:, None]).mean(1), baseline, atol=1e-5, rtol=1e-4)


def test_loss_without_own_effects_is_the_control_and_own_effects_add_their_expected_surrogate():
    agent, observations, native, old, alpha, beta, values, considered = fixture()
    args = Args()
    advantages, returns = torch.randn(256, device="cuda"), values + torch.randn(256, device="cuda")
    actions = torch.distributions.Beta(alpha, beta).sample((considered,)).transpose(0, 1).clamp(1e-6, 1 - 1e-6)
    logprobs = agent.action_logprob(alpha[:, None], beta[:, None], actions).detach()
    zero = torch.zeros(256, considered, device="cuda")
    loss, metrics = surprise_ppo_loss(agent, observations, native, old, advantages, returns, values, actions, logprobs, zero, args)
    control = ppo_loss(agent, observations, native, old, advantages, returns, values, args)
    torch.testing.assert_close(loss, control[0])
    torch.testing.assert_close(metrics[:6], control[1])
    own = torch.randn(256, considered, device="cuda")
    with_own, metrics = surprise_ppo_loss(agent, observations, native, old, advantages, returns, values, actions, logprobs, own, args)
    # At the old policy every considered ratio is one, so the term is minus the mean own effect.
    torch.testing.assert_close(with_own - loss, -own.mean(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(metrics[6], -own.mean(), atol=1e-5, rtol=1e-4)
