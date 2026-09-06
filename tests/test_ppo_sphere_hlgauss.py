"""Numerical CUDA contracts for the matched HalfCheetah histogram critic.

Run through mlq; these use synthetic 17-observation/6-action batches, not
shortened environment training runs. The full-batch test records CUDA timing
without asserting machine-dependent throughput.
"""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Beta

from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_v2 as histogram,
)
from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_v1 as baseline,
)
from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.ppo_loop import compute_gae
from cleanrl.shared.vector_norm import VectorRewardNorm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


@pytest.fixture
def device():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    with torch.random.fork_rng():
        torch.manual_seed(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield torch.device("cuda")
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
    )


def _observations(batch_size, device):
    generator = np.random.default_rng(1)
    return torch.as_tensor(generator.standard_normal((batch_size, 17)).astype(np.float32), device=device)


def _inputs(agent, observations, *, clipped=False):
    count = observations.shape[0]
    offsets = torch.linspace(-1.0, 1.0, count, device=observations.device)
    native_actions = torch.linspace(0.07, 0.93, count * 6, device=observations.device).reshape(count, 6)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
        if clipped:
            old_logprobs = old_logprobs - 0.6 * offsets
    return native_actions, old_logprobs, 0.7 + 2.3 * offsets


def _nonuniform_critic(agent):
    # A learned-head probe is needed: a uniform prediction has identical CE
    # against every normalized label and hides incorrect target handling.
    with torch.no_grad():
        head = agent.critic[-1]
        head.bias.copy_(torch.linspace(-2.0, 3.0, head.out_features, device=head.bias.device))
        coordinates = torch.arange(head.weight.numel(), device=head.weight.device).reshape_as(head.weight)
        head.weight.copy_(0.05 * torch.sin(coordinates * 0.13))


def test_initial_policy_densities_and_sampled_actions_match_baseline(device):
    observations = _observations(256, device)
    torch.manual_seed(1)
    reference = baseline.Agent(_spaces()).to(device)
    torch.manual_seed(1)
    candidate = histogram.Agent(_spaces()).to(device)
    native_actions, _, _ = _inputs(reference, observations)
    with torch.no_grad():
        ref_alpha, ref_beta, _ = reference.get_policy_and_value(observations)
        alpha, beta, _ = candidate.get_policy_and_value(observations)
        torch.testing.assert_close(alpha, ref_alpha, rtol=0, atol=0)
        torch.testing.assert_close(beta, ref_beta, rtol=0, atol=0)
        torch.testing.assert_close(
            candidate.action_logprob(alpha, beta, native_actions),
            reference.action_logprob(ref_alpha, ref_beta, native_actions),
            rtol=0,
            atol=0,
        )
        torch.manual_seed(1)
        expected = reference.get_action_and_value(observations)
        torch.manual_seed(1)
        actual = candidate.get_action_and_value(observations)
        for observed, wanted in zip(actual[:3], expected[:3]):
            torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def test_public_scalar_values_are_bounded_and_ce_updates_critic(device):
    config = HLGaussConfig(v_min=-9.0, v_max=9.0)
    agent = histogram.Agent(_spaces(), value_config=config).to(device)
    observations = _observations(256, device)
    native, old_logprobs, advantages = _inputs(agent, observations)
    args = histogram.Args(value=config, vf_coef=1.0, ent_coef=0.0)
    targets = agent.value_support.project(torch.full((256,), 6.0, device=device)).detach()
    with torch.no_grad():
        values = agent.get_value(observations)
        assert values.shape == (256, 1)
        assert torch.all((values >= config.v_min) & (values <= config.v_max))
        torch.testing.assert_close(agent.get_policy_and_value(observations)[2], values)
        torch.testing.assert_close(agent.get_action_and_value(observations, 2 * native - 1)[3], values)
    loss, metrics = histogram.ppo_loss(agent, observations, native, old_logprobs, advantages * 0, targets, args)
    _, _, logits = agent.get_policy_and_value_logits(observations)
    expected_ce = -(targets * logits.log_softmax(-1)).sum(-1).mean()
    torch.testing.assert_close(loss, expected_ce)
    torch.testing.assert_close(metrics[1], expected_ce.detach())
    optimizer = torch.optim.SGD(agent.critic.parameters(), lr=0.1)
    loss.backward()
    gradient_norm = torch.stack([p.grad.square().sum() for p in agent.critic.parameters() if p.grad is not None]).sum()
    assert torch.isfinite(gradient_norm) and gradient_norm > 0
    optimizer.step()
    with torch.no_grad():
        updated = agent.get_value(observations)
        _, after = histogram.ppo_loss(agent, observations, native, old_logprobs, advantages * 0, targets, args)
        assert after[1] < metrics[1]
        assert updated.mean() > values.mean()
        assert torch.all((updated >= config.v_min) & (updated <= config.v_max))


def test_target_labels_do_not_change_unclipped_actor_gradient(device):
    agent = histogram.Agent(_spaces()).to(device)
    observations = _observations(512, device)
    native, old_logprobs, advantages = _inputs(agent, observations)
    args = histogram.Args(norm_adv=False, vf_coef=0.5, ent_coef=0.01)
    gradients = []
    critic_gradients = []
    for target in (-25.0, 25.0):
        agent.zero_grad(set_to_none=True)
        labels = agent.value_support.project(torch.full((512,), target, device=device)).detach()
        loss, metrics = histogram.ppo_loss(agent, observations, native, old_logprobs, advantages, labels, args)
        assert metrics[-1] == 0, "isolate labels from PPO and global gradient clipping"
        loss.backward()
        gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in agent.actor.parameters()]))
        critic_gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in agent.critic.parameters()]))
    assert gradients[0].norm() > 0
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert not torch.allclose(critic_gradients[0], critic_gradients[1])


def test_normalized_reward_gae_labels_retain_return_units_in_loss(device):
    agent = histogram.Agent(_spaces()).to(device)
    _nonuniform_critic(agent)
    steps, envs = 8, 16
    observations = _observations(steps * envs, device)
    generator = np.random.default_rng(1)
    normalizer = VectorRewardNorm(envs, gamma=0.99)
    for raw in generator.uniform(4.0, 8.0, size=(256, envs)):
        normalizer.normalize(raw, np.zeros(envs))
    raw_rewards = generator.uniform(4.0, 8.0, size=(steps, envs))
    terms = np.zeros((steps, envs), dtype=np.float32)
    truncs = np.zeros_like(terms)
    terms[3, 0] = 1
    truncs[5, 1] = 1
    rewards = np.stack([normalizer.normalize(raw, terminal) for raw, terminal in zip(raw_rewards, terms)])
    with torch.no_grad():
        values = agent.get_value(observations).reshape(steps, envs)
        tail = torch.linspace(-0.5, 0.5, envs, device=device)
        bootstrap = torch.full_like(values, 1.25)
        advantages, returns = compute_gae(
            torch.as_tensor(rewards, device=device),
            values,
            torch.as_tensor(terms, device=device),
            torch.as_tensor(truncs, device=device),
            bootstrap,
            tail,
            0.99,
            0.95,
        )
        value_array, next_tail = values.cpu().numpy(), tail.cpu().numpy()
        expected_advantages = np.zeros_like(rewards)
        running = np.zeros(envs, dtype=np.float32)
        for step in reversed(range(steps)):
            next_value = next_tail if step == steps - 1 else value_array[step + 1]
            next_value = np.where(truncs[step], 1.25, next_value)
            delta = rewards[step] + 0.99 * (1 - terms[step]) * next_value - value_array[step]
            running = delta + 0.99 * 0.95 * (1 - np.maximum(terms[step], truncs[step])) * running
            expected_advantages[step] = running
        expected_returns = torch.as_tensor(expected_advantages + value_array, device=device).flatten()
        torch.testing.assert_close(returns.flatten(), expected_returns, rtol=2e-5, atol=2e-6)
        targets = agent.value_support.project(returns.flatten()).detach()
    native, old_logprobs, _ = _inputs(agent, observations)
    args = histogram.Args(norm_adv=False, vf_coef=0.5, ent_coef=0.0)
    loss, metrics = histogram.ppo_loss(
        agent,
        observations,
        native,
        old_logprobs,
        advantages.flatten(),
        targets,
        args,
    )
    _, _, logits = agent.get_policy_and_value_logits(observations)
    log_probs = logits.log_softmax(-1)
    expected_ce = -(agent.value_support.project(expected_returns) * log_probs).sum(-1).mean()
    torch.testing.assert_close(metrics[1], expected_ce.detach())
    torch.testing.assert_close(loss, -advantages.mean() + args.vf_coef * expected_ce)
    # Ensure this numerical probe distinguishes both historical unit mistakes.
    standardized = (expected_returns - expected_returns.mean()) / expected_returns.std()
    for wrong_units in (standardized, expected_returns * 1000):
        wrong_ce = -(agent.value_support.project(wrong_units) * log_probs).sum(-1).mean()
        assert (expected_ce - wrong_ce).abs() > 1e-3


def test_support_override_changes_scalar_boundaries_and_target_loss(device):
    observations = _observations(128, device)
    losses = []
    for lower, upper in ((-2.0, 3.0), (-80.0, 120.0)):
        config = HLGaussConfig(v_min=lower, v_max=upper, num_bins=31)
        agent = histogram.Agent(_spaces(), value_config=config).to(device)
        with torch.no_grad():
            agent.critic[-1].weight.zero_()
            for endpoint, expected in ((0, lower), (-1, upper)):
                agent.critic[-1].bias.fill_(-80)
                agent.critic[-1].bias[endpoint] = 80
                torch.testing.assert_close(
                    agent.get_value(observations),
                    torch.full((128, 1), expected, device=device),
                    rtol=2e-6,
                    atol=2e-6,
                )
        _nonuniform_critic(agent)
        native, old_logprobs, advantages = _inputs(agent, observations)
        returns = torch.full((128,), 1.5, device=device)
        labels = agent.value_support.project(returns).detach()
        _, metrics = histogram.ppo_loss(
            agent,
            observations,
            native,
            old_logprobs,
            advantages,
            labels,
            histogram.Args(value=config),
        )
        _, _, logits = agent.get_policy_and_value_logits(observations)
        torch.testing.assert_close(metrics[1], agent.value_support.loss(logits, returns).detach())
        losses.append(metrics[1])
    assert (losses[0] - losses[1]).abs() > 1e-3


def test_full_batch_compiled_loss_gradients_and_repeated_updates(device, record_property):
    """The real 2048 x 16 learner shape, including steady-state CUDA replay."""
    agent = histogram.Agent(_spaces()).to(device)
    _nonuniform_critic(agent)
    reference = copy.deepcopy(agent)
    args = histogram.Args(num_envs=16, num_steps=2048, num_minibatches=1, update_epochs=10, norm_adv=False, ent_coef=0.01)
    observations = _observations(2048 * 16, device)
    native, old_logprobs, advantages = _inputs(agent, observations, clipped=True)
    targets = agent.value_support.project(torch.linspace(-60, 60, observations.shape[0], device=device)).detach()
    inputs = (observations, native, old_logprobs, advantages, targets)
    compiled = torch.compile(
        lambda *batch: histogram.ppo_loss(agent, *batch, args), mode="reduce-overhead", fullgraph=True, dynamic=False
    )
    expected_loss, expected_metrics = histogram.ppo_loss(reference, *inputs, args)
    torch.compiler.cudagraph_mark_step_begin()
    actual_loss, actual_metrics = compiled(*inputs)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-5, atol=3e-6)
    assert expected_metrics[-1] > 0.5, "full-batch probe must exercise PPO clipping"
    expected_loss.backward()
    actual_loss.backward()
    for (name, expected), (_, actual) in zip(reference.named_parameters(), agent.named_parameters()):
        assert expected.grad is not None and actual.grad is not None, name
        assert torch.isfinite(actual.grad).all(), name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=5e-6, msg=name)
    del actual_loss, actual_metrics, expected_loss, expected_metrics
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    snapshots = torch.empty((args.update_epochs, 7), device=device)
    before_policy = agent.actor(observations[:16]).detach().clone()
    before_value = agent.get_value(observations[:16]).detach().clone()
    # Two updates let Inductor's backward compilation and CUDA graph capture
    # settle before timing the ten-epoch production-shape replay workload.
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compiled(*inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, foreach=True)
        optimizer.step()
        del loss, metrics
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(args.update_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compiled(*inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, foreach=True)
        optimizer.step()
        snapshots[epoch, 0].copy_(loss.detach())
        snapshots[epoch, 1:].copy_(metrics)
        del loss, metrics
    stop.record()
    stop.synchronize()
    record_property("compiled_full_batch_update_ms", start.elapsed_time(stop) / args.update_epochs)
    record_property("transitions_per_update", observations.shape[0])
    assert torch.isfinite(snapshots).all()
    assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())
    with torch.no_grad():
        assert not torch.allclose(agent.actor(observations[:16]), before_policy)
        assert not torch.allclose(agent.get_value(observations[:16]), before_value)


def test_evaluation_normalization_preserves_termination_and_truncation_semantics():
    class Episodes(gym.Env):
        observation_space = gym.spaces.Box(-100, 100, (2,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def __init__(self):
            self.episode = 0
            self.age = 0

        def reset(self, **kwargs):
            self.episode += 1
            self.age = 0
            return np.array([self.episode, -2], np.float32), {}

        def step(self, action):
            self.age += 1
            terminal = self.age == 2 and self.episode % 2 == 0
            truncated = self.age == 2 and not terminal
            return np.array([self.episode, self.age], np.float32), float(self.age + 3), terminal, truncated, {}

    reference = gym.wrappers.TransformReward(
        gym.wrappers.NormalizeReward(
            gym.wrappers.TransformObservation(
                gym.wrappers.NormalizeObservation(Episodes()), lambda obs: np.clip(obs, -10, 10)
            ),
            gamma=0.99,
        ),
        lambda reward: np.clip(reward, -10, 10),
    )
    candidate = histogram._EvaluationNorm(Episodes(), 0.99)
    try:
        for _ in range(4):
            np.testing.assert_allclose(candidate.reset()[0], reference.reset()[0], rtol=1e-6, atol=1e-6)
            for _ in range(2):
                actual, expected = candidate.step(np.zeros(1)), reference.step(np.zeros(1))
                np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
                assert actual[1] == pytest.approx(expected[1], rel=1e-6, abs=1e-6)
                assert actual[2:4] == expected[2:4]
    finally:
        candidate.close()
        reference.close()
