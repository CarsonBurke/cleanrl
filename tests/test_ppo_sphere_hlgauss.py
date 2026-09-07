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
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_decoupled_v5 as histogram_decoupled_v5,
)
from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_default_v3 as histogram_default_v3,
)
from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_raw_symlog_v4 as histogram_raw_symlog_v4,
)
from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_v2 as histogram_v2,
)
from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_widecritic_v6 as histogram_widecritic_v6,
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


V5_RAW_CONFIG = HLGaussConfig(
    v_min=-2500.0, v_max=2500.0, num_bins=1001, sigma_ratio=0.75, transform="linear", bin_type="edges"
)
V5_NORMALIZED = pytest.param((histogram_decoupled_v5, None, True), id="decoupled_v5_normalized")
V5_RAW = pytest.param((histogram_decoupled_v5, V5_RAW_CONFIG, False), id="decoupled_v5_raw")
V6_NORMALIZED = pytest.param((histogram_widecritic_v6, None, True), id="widecritic_v6_normalized")
V6_RAW = pytest.param((histogram_widecritic_v6, V5_RAW_CONFIG, False), id="widecritic_v6_raw")


@pytest.fixture(
    params=[
        pytest.param((histogram_v2, None, True), id="v2"),
        pytest.param((histogram_default_v3, None, True), id="default_v3"),
        pytest.param((histogram_raw_symlog_v4, None, False), id="raw_symlog_v4"),
        V5_NORMALIZED,
        V5_RAW,
        V6_NORMALIZED,
        V6_RAW,
    ]
)
def histogram_spec(request):
    module, value_config, reward_norm = request.param
    return SimpleNamespace(module=module, value_config=value_config, reward_norm=reward_norm)


@pytest.fixture
def histogram(histogram_spec):
    return histogram_spec.module


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


def test_initial_policy_densities_and_sampled_actions_match_baseline(histogram, histogram_spec, device):
    observations = _observations(256, device)
    torch.manual_seed(1)
    reference = baseline.Agent(_spaces()).to(device)
    torch.manual_seed(1)
    candidate = histogram.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
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


def test_host_mirror_policy_statistics_match_cuda_after_update(histogram, histogram_spec, device):
    agent = histogram.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
    observations = _observations(16, device)
    host_observations = observations.cpu().numpy()
    mirror_factory = {
        histogram_v2: histogram_v2.HostSiTUSphereActor,
        histogram_default_v3: histogram_default_v3.make_host_mirror,
        histogram_raw_symlog_v4: histogram_raw_symlog_v4.make_host_mirror,
        histogram_decoupled_v5: histogram_decoupled_v5.make_host_mirror,
        histogram_widecritic_v6: histogram_widecritic_v6.make_host_mirror,
    }[histogram]
    mirror = mirror_factory(agent.actor, 16)
    native, _, _ = _inputs(agent, observations)

    @torch.no_grad()
    def check_statistics():
        mirror.refresh()
        concentration = np.logaddexp(0.0, mirror(host_observations)) + 1.0
        host_alpha, host_beta = torch.as_tensor(concentration, device=device).chunk(2, dim=-1)
        alpha, beta, _ = agent.get_policy_and_value(observations)
        host_distribution = Beta(host_alpha, host_beta, validate_args=False)
        distribution = Beta(alpha, beta, validate_args=False)
        torch.testing.assert_close(host_distribution.mean, distribution.mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(host_distribution.variance, distribution.variance, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            agent.action_logprob(host_alpha, host_beta, native),
            agent.action_logprob(alpha, beta, native),
            rtol=2e-5,
            atol=5e-6,
        )
        return distribution.mean.clone()

    before = check_statistics()
    optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    alpha, beta, _ = agent.get_policy_and_value(observations)
    loss = -agent.action_logprob(alpha, beta, native).mean()
    loss.backward()
    optimizer.step()
    after = check_statistics()
    assert not torch.allclose(after, before)


def test_public_scalar_values_are_bounded_and_ce_updates_critic(histogram, device):
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


def test_target_labels_do_not_change_unclipped_actor_gradient(histogram, histogram_spec, device):
    agent = histogram.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
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


@pytest.mark.parametrize(
    "histogram_spec",
    [
        pytest.param((histogram_v2, None, True), id="v2"),
        pytest.param((histogram_default_v3, None, True), id="default_v3"),
        V5_NORMALIZED,
    ],
    indirect=True,
)
def test_normalized_reward_gae_labels_retain_return_units_in_loss(histogram, device):
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


@pytest.mark.parametrize(
    "histogram_spec",
    [pytest.param((histogram_raw_symlog_v4, None, False), id="raw_symlog_v4"), V5_RAW],
    indirect=True,
)
def test_raw_reward_gae_labels_retain_return_units_in_loss(histogram, histogram_spec, device):
    agent = histogram.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
    support = agent.value_support
    with torch.no_grad():
        agent.critic[-1].weight.zero_()
        agent.critic[-1].bias.copy_(support.project(torch.tensor(100.0, device=device)).clamp_min(1e-6).log())
    observations = _observations(8, device)
    rewards = torch.tensor([[25.0, -40.0], [-50.0, 75.0], [60.0, -30.0], [-80.0, 45.0]], device=device)
    terms = torch.zeros_like(rewards)
    truncs = torch.zeros_like(rewards)
    terms[1, 0] = 1
    truncs[1, 1] = 1
    # A true terminal wins even when the environment also reports a time limit.
    terms[3, 0] = truncs[3, 0] = 1
    gamma, gae_lambda = 0.99, 0.95
    with torch.no_grad():
        values = agent.get_value(observations).reshape_as(rewards)
        tail = agent.get_value(_observations(2, device)).flatten()
        bootstrap = agent.get_value(_observations(8, device)).reshape_as(rewards)
        advantages, returns = compute_gae(rewards, values, terms, truncs, bootstrap, tail, gamma, gae_lambda)
        reward_array, value_array = rewards.cpu().numpy(), values.cpu().numpy()
        term_array, trunc_array = terms.cpu().numpy(), truncs.cpu().numpy()
        bootstrap_array, tail_array = bootstrap.cpu().numpy(), tail.cpu().numpy()
        expected_advantages = np.zeros_like(reward_array)
        running = np.zeros(2, dtype=np.float32)
        for step in reversed(range(4)):
            next_value = tail_array if step == 3 else value_array[step + 1]
            next_value = np.where(trunc_array[step], bootstrap_array[step], next_value)
            delta = reward_array[step] + gamma * (1 - term_array[step]) * next_value - value_array[step]
            running = delta + gamma * gae_lambda * (1 - np.maximum(term_array[step], trunc_array[step])) * running
            expected_advantages[step] = running
        expected_returns = torch.as_tensor(expected_advantages + value_array, device=device)
        torch.testing.assert_close(advantages, torch.as_tensor(expected_advantages, device=device))
        torch.testing.assert_close(returns, expected_returns)
        torch.testing.assert_close(returns[1, 0], rewards[1, 0])
        torch.testing.assert_close(returns[3, 0], rewards[3, 0])
        torch.testing.assert_close(returns[1, 1], rewards[1, 1] + gamma * bootstrap[1, 1])
        labels = support.project(returns.flatten())
        # Coordinate smoothing introduces a small raw-mean bias, not a unit conversion.
        torch.testing.assert_close(support.probs_to_scalar(labels), returns.flatten(), rtol=1e-3, atol=2e-3)
    native, old_logprobs, _ = _inputs(agent, observations)
    args = histogram.Args(norm_adv=False, vf_coef=0.5, ent_coef=0.0)
    loss, metrics = histogram.ppo_loss(agent, observations, native, old_logprobs, advantages.flatten(), labels, args)
    _, _, logits = agent.get_policy_and_value_logits(observations)
    log_probs = logits.log_softmax(-1)
    expected_ce = -(support.project(expected_returns.flatten()) * log_probs).sum(-1).mean()
    torch.testing.assert_close(metrics[1], expected_ce.detach())
    torch.testing.assert_close(loss, -advantages.mean() + args.vf_coef * expected_ce)
    clipped_returns = compute_gae(rewards.clamp(-10, 10), values, terms, truncs, bootstrap, tail, gamma, gae_lambda)[1]
    standardized_returns = (expected_returns - expected_returns.mean()) / expected_returns.std()
    for wrong_units in (clipped_returns, standardized_returns):
        wrong_ce = -(support.project(wrong_units.flatten()) * log_probs).sum(-1).mean()
        assert (expected_ce - wrong_ce).abs() > 1e-3


def test_raw_symlog_projection_and_scalar_expectation_preserve_realistic_scale(device):
    support = histogram_raw_symlog_v4.Agent(_spaces()).to(device).value_support
    magnitudes = torch.tensor([1.0, 10.0, 100.0, 1000.0, 5000.0], device=device)
    targets = torch.cat((-magnitudes.flip(0), torch.zeros(1, device=device), magnitudes))
    labels = support.project(targets)
    assert torch.isfinite(labels).all() and torch.all(labels >= 0)
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets))
    decoded = support.probs_to_scalar(labels)
    # Gaussian smoothing in symlog coordinates is not a mean-preserving projection.
    # This tolerance covers its small outward bias, but not clipping or wrong units.
    torch.testing.assert_close(decoded, targets, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(support.to_scalar(labels.log()), decoded, rtol=2e-6, atol=2e-5)
    assert torch.all(decoded.diff() > 0)
    # A broad asymmetric histogram separates E[raw] from symexp(E[symlog]).
    # Narrow single-target labels alone cannot reliably distinguish these decoders.
    mixture_targets = torch.tensor([1.0, 5000.0], device=device)
    mixture = support.project(mixture_targets).mean(0)
    expected_mean = mixture_targets.mean()
    torch.testing.assert_close(support.probs_to_scalar(mixture), expected_mean, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(support.to_scalar(mixture.log()), expected_mean, rtol=1e-3, atol=2e-3)


def test_support_override_changes_scalar_boundaries_and_target_loss(histogram, device):
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


@pytest.mark.parametrize("varied_component", ["actor", "critic"])
@pytest.mark.parametrize("histogram_spec", [V5_NORMALIZED, V5_RAW], indirect=True)
def test_decoupled_adam_updates_ignore_other_loss_magnitude(histogram_spec, varied_component, device):
    reference = histogram_decoupled_v5.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
    _nonuniform_critic(reference)
    candidate = copy.deepcopy(reference)
    args = histogram_decoupled_v5.Args(
        learning_rate=2e-3, critic_learning_rate=7e-4, vf_coef=0.5, ent_coef=0.0, norm_adv=False
    )
    optimizers = [histogram_decoupled_v5.make_optimizer(agent, args) for agent in (reference, candidate)]
    observations = _observations(128, device)
    native, old_logprobs, advantages = _inputs(reference, observations)
    extent = 1000.0 if not histogram_spec.reward_norm else 5.0
    returns = torch.linspace(-extent, extent, observations.shape[0], device=device, requires_grad=True)
    original_returns = returns.detach().clone()
    labels = reference.value_support.project(returns)
    original_labels = labels.detach().clone()
    fixed_component = "critic" if varied_component == "actor" else "actor"
    fixed_reference = getattr(reference, fixed_component)
    fixed_candidate = getattr(candidate, fixed_component)
    before = [parameter.detach().clone() for parameter in fixed_reference.parameters()]
    # Vary the interference across steps: Adam can conceal a constant rescaling
    # through its moments, but cannot conceal a changing shared clip budget.
    for step, magnitude in enumerate((1e5, 1e2, 1e7, 1e3)):
        for index, (agent, optimizer) in enumerate(zip((reference, candidate), optimizers)):
            multiplier = magnitude if index else 1.0
            step_args = copy.copy(args)
            step_advantages = advantages.roll(step * 7)
            if varied_component == "actor":
                step_advantages = step_advantages * multiplier
            else:
                step_args.vf_coef *= multiplier
            loss, _ = histogram_decoupled_v5.ppo_loss(
                agent, observations, native, old_logprobs, step_advantages, labels, step_args
            )
            histogram_decoupled_v5.optimizer_step(loss, optimizer, args.max_grad_norm)
        for expected, actual in zip(fixed_reference.parameters(), fixed_candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Critic supervision must not backpropagate through the return producer,
        # nor modify or detach its input graph in place.
        assert returns.grad is None and returns.requires_grad
        torch.testing.assert_close(returns, original_returns, rtol=0, atol=0)
        torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    assert any(not torch.equal(parameter, initial) for parameter, initial in zip(fixed_reference.parameters(), before))


def test_percentile_return_scale_matches_ema_without_changing_targets(device):
    scale = histogram_decoupled_v5.PercentileReturnScale().to(device)
    shifted_scale = histogram_decoupled_v5.PercentileReturnScale().to(device)
    expected_bounds = np.zeros(2, dtype=np.float64)
    batches = (
        np.linspace(-10, 10, 41),
        np.linspace(-2000, 1000, 41),  # Asymmetry distinguishes percentiles from a magnitude estimate.
        np.full(41, 700.0),
        np.linspace(-400, 800, 41),
    )
    for values in batches:
        returns = torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)
        original = returns.detach().clone()
        expected_bounds = 0.99 * expected_bounds + 0.01 * np.quantile(values, [0.05, 0.95])
        expected = max(1.0, expected_bounds[1] - expected_bounds[0])
        denominator = scale(returns)
        shifted_denominator = shifted_scale(returns + 1234.0)
        torch.testing.assert_close(denominator, torch.tensor(expected, dtype=torch.float32, device=device))
        torch.testing.assert_close(shifted_denominator, denominator, rtol=2e-6, atol=2e-6)
        assert not denominator.requires_grad
        assert returns.requires_grad
        torch.testing.assert_close(returns, original, rtol=0, atol=0)
        weights = torch.linspace(-0.7, 1.3, returns.numel(), device=device)
        # Only the actor's advantages are divided; gradients through their
        # producer survive, while percentile statistics are stop-gradient.
        gradient = torch.autograd.grad((returns * weights / denominator).sum(), returns)[0]
        torch.testing.assert_close(gradient, weights / expected)


def test_full_batch_compiled_loss_gradients_and_repeated_updates(histogram, histogram_spec, device, record_property):
    """The real 2048 x 16 learner shape, including steady-state CUDA replay."""
    agent = histogram.Agent(_spaces(), value_config=histogram_spec.value_config).to(device)
    _nonuniform_critic(agent)
    reference = copy.deepcopy(agent)
    args = histogram.Args(
        num_envs=16,
        num_steps=2048,
        num_minibatches=1,
        update_epochs=10,
        norm_adv=False,
        ent_coef=0.01,
        value=agent.value_support.config,
    )
    observations = _observations(2048 * 16, device)
    native, old_logprobs, advantages = _inputs(agent, observations, clipped=True)
    extent = 2000 if histogram_spec.value_config is V5_RAW_CONFIG else 60
    returns = torch.linspace(-extent, extent, observations.shape[0], device=device)
    expected_targets = agent.value_support.project(returns).detach()
    compiled_project = torch.compile(agent.value_support.project, mode="reduce-overhead", fullgraph=True, dynamic=False)
    torch.compiler.cudagraph_mark_step_begin()
    targets = compiled_project(returns).detach().clone()
    torch.testing.assert_close(targets, expected_targets, rtol=2e-5, atol=2e-6)
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
    if histogram in (histogram_decoupled_v5, histogram_widecritic_v6):
        optimizer = histogram.make_optimizer(agent, args)

        def update(loss):
            return histogram.optimizer_step(loss, optimizer, args.max_grad_norm)

    else:
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

        def update(loss):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, foreach=True)
            optimizer.step()

    snapshots = torch.empty((args.update_epochs, 7), device=device)
    before_policy = agent.actor(observations[:16]).detach().clone()
    before_value = agent.get_value(observations[:16]).detach().clone()
    # Two updates let Inductor's backward compilation and CUDA graph capture
    # settle before timing the ten-epoch production-shape replay workload.
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        agent.zero_grad(set_to_none=True)
        loss, metrics = compiled(*inputs)
        update(loss)
        del loss, metrics
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(args.update_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        agent.zero_grad(set_to_none=True)
        loss, metrics = compiled(*inputs)
        update(loss)
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


def test_evaluation_normalization_preserves_termination_and_truncation_semantics(histogram, histogram_spec):
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
            reward = float(25 * self.age * (-1 if self.age == 2 else 1))
            return np.array([self.episode, self.age], np.float32), reward, terminal, truncated, {}

    reference = gym.wrappers.TransformObservation(
        gym.wrappers.NormalizeObservation(Episodes()), lambda obs: np.clip(obs, -10, 10)
    )
    raw_rewards = not histogram_spec.reward_norm
    if histogram in (histogram_decoupled_v5, histogram_widecritic_v6):
        candidate = histogram._EvaluationNorm(Episodes(), 0.99, reward_norm=histogram_spec.reward_norm)
    elif raw_rewards:
        candidate = histogram._EvaluationNorm(Episodes())
    if not raw_rewards:
        reference = gym.wrappers.TransformReward(
            gym.wrappers.NormalizeReward(reference, gamma=0.99),
            lambda reward: np.clip(reward, -10, 10),
        )
        if histogram not in (histogram_decoupled_v5, histogram_widecritic_v6):
            candidate = histogram._EvaluationNorm(Episodes(), 0.99)
    try:
        for _ in range(4):
            np.testing.assert_allclose(candidate.reset()[0], reference.reset()[0], rtol=1e-6, atol=1e-6)
            for step in range(2):
                actual, expected = candidate.step(np.zeros(1)), reference.step(np.zeros(1))
                np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
                assert actual[1] == pytest.approx(expected[1], rel=1e-6, abs=1e-6)
                if raw_rewards:
                    assert actual[1] == (25.0 if step == 0 else -50.0)
                assert actual[2:4] == expected[2:4]
    finally:
        candidate.close()
        reference.close()
