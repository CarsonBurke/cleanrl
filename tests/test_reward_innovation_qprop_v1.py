import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl" / "ppo_continuous_action_reward_innovation_qprop_v1.py"
REFERENCE = ROOT / "cleanrl" / "ppo_continuous_action_sffactor_rewardanchor_v4_rngpaired.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module("reward_innovation_qprop_v1", SCRIPT)
REFERENCE_MODULE = load_module("rewardanchor_v4_reference", REFERENCE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def test_scalar_td0_handles_terminal_valid_truncation_and_missing_final_state():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    next_values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    terminations = torch.tensor([0.0, 1.0, 0.0, 0.0])
    valids = torch.tensor([1.0, 0.0, 1.0, 0.0])

    targets, valid = MODULE.scalar_td0_target(
        rewards, next_values, terminations, valids, gamma=0.9
    )

    torch.testing.assert_close(targets, torch.tensor([10.0, 2.0, 30.0, 0.0]))
    torch.testing.assert_close(valid, torch.tensor([True, True, True, False]))


def test_tail_innovation_handles_terminal_valid_truncation_and_missing_final_state():
    current = torch.tensor([1.0, 2.0, 3.0, 4.0])
    next_last = torch.tensor([10.0, 20.0, 30.0, 40.0])
    next_tail = torch.tensor([5.0, 6.0, 7.0, 8.0])
    terminations = torch.tensor([0.0, 1.0, 0.0, 0.0])
    valids = torch.tensor([1.0, 0.0, 1.0, 0.0])

    targets, valid = MODULE.tail_innovation_target(
        current, next_last, next_tail, terminations, valids, gamma=0.9
    )

    torch.testing.assert_close(
        targets, torch.tensor([10.0 + 0.9 * 5.0 - 1.0, -2.0, 30.0 + 0.9 * 7.0 - 3.0, 0.0])
    )
    torch.testing.assert_close(valid, torch.tensor([True, True, True, False]))


def test_future_reward_targets_stop_at_boundaries_and_rollout_tail():
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    boundaries = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    targets, valid = MODULE.build_future_reward_targets(rewards, boundaries, horizon=4)

    torch.testing.assert_close(targets[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(valid[0, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(valid[1, 0], torch.tensor([True, False, False, False]))
    torch.testing.assert_close(valid[2, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(valid[3, 0], torch.tensor([True, False, False, False]))


def test_action_future_targets_fill_known_zeros_after_true_terminal():
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    terminations = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    boundaries = terminations.clone()

    targets, valid = MODULE.build_action_future_reward_targets(
        rewards, terminations, boundaries, horizon=4
    )

    torch.testing.assert_close(targets[0, 0], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    torch.testing.assert_close(valid[0, 0], torch.tensor([True, True, True, True]))
    torch.testing.assert_close(targets[1, 0], torch.tensor([2.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(valid[1, 0], torch.tensor([True, True, True, True]))
    torch.testing.assert_close(valid[2, 0], torch.tensor([True, True, False, False]))


def test_action_future_targets_censor_after_time_limit():
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    terminations = torch.zeros_like(rewards)
    boundaries = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    targets, valid = MODULE.build_action_future_reward_targets(
        rewards, terminations, boundaries, horizon=4
    )

    torch.testing.assert_close(targets[0, 0], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    torch.testing.assert_close(valid[0, 0], torch.tensor([True, True, False, False]))


def test_scalar_gae_uses_true_rewards_and_censors_missing_final_state():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_values = torch.tensor([[4.0], [5.0], [6.0]])
    terminations = torch.tensor([[0.0], [0.0], [1.0]])
    boundaries = torch.ones_like(rewards)
    valids = torch.tensor([[1.0], [0.0], [0.0]])

    advantages, returns = MODULE.scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.9,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(advantages[0], rewards[0] + 0.9 * next_values[0] - values[0])
    torch.testing.assert_close(advantages[1], torch.zeros(1))
    torch.testing.assert_close(returns[1], values[1])
    torch.testing.assert_close(advantages[2], rewards[2] - values[2])


def test_value_is_exact_discounted_prefix_plus_discounted_tail():
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        predictive_rank=3,
        prediction_horizon=5,
        gamma=0.8,
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    with torch.no_grad():
        agent.coefficient_head.weight.fill_(0.1)
        agent.coefficient_head.bias.fill_(0.2)
        agent.temporal_basis.weight.fill_(0.3)
        agent.temporal_basis.bias.copy_(torch.arange(5, dtype=torch.float32))
        agent.tail_head.weight.fill_(0.05)
        agent.tail_head.bias.fill_(0.4)

    _, future_rewards, tail, value = agent.get_critic(torch.randn(7, 3))
    expected = future_rewards @ agent.discount_weights + (0.8**5) * tail

    torch.testing.assert_close(value, expected)
    assert future_rewards.shape == (7, 5)
    centered = future_rewards - future_rewards.mean(dim=0)
    assert torch.linalg.matrix_rank(centered) <= args.predictive_rank


def test_zero_initialized_coefficients_and_tail_start_with_zero_value():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, predictive_rank=3, prediction_horizon=5)
    agent = MODULE.Agent(DummyVectorEnv(), args)

    coefficients, future_rewards, tail, value = agent.get_critic(torch.randn(7, 3))

    torch.testing.assert_close(coefficients, torch.zeros_like(coefficients))
    torch.testing.assert_close(future_rewards, torch.zeros_like(future_rewards))
    torch.testing.assert_close(tail, torch.zeros_like(tail))
    torch.testing.assert_close(value, torch.zeros_like(value))


def test_actor_and_global_rng_are_paired_with_reference():
    new_args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, predictive_rank=3, prediction_horizon=5)
    reference_args = REFERENCE_MODULE.Args(hidden=8, k_blocks=1, n_experts=2)

    torch.manual_seed(19)
    new_agent = MODULE.Agent(DummyVectorEnv(), new_args)
    new_rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(19)
    reference_agent = REFERENCE_MODULE.Agent(DummyVectorEnv(), reference_args)
    reference_rng_after = torch.random.get_rng_state().clone()

    new_actor = dict(new_agent.named_parameters())
    reference_actor = dict(reference_agent.named_parameters())
    for name in (
        "actor_alpha_head.weight",
        "actor_alpha_head.bias",
        "actor_beta_head.weight",
        "actor_beta_head.bias",
    ):
        torch.testing.assert_close(new_actor[name], reference_actor[name])
    for name, parameter in new_agent.trunk.named_parameters():
        torch.testing.assert_close(
            parameter, dict(reference_agent.trunk.named_parameters())[name]
        )
    torch.testing.assert_close(new_rng_after, reference_rng_after)


def test_action_innovation_construction_can_preserve_global_rng():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, prediction_horizon=5)
    torch.manual_seed(23)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    rng_before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        MODULE.ActionInnovation(
            obs_dim=3,
            act_dim=2,
            horizon=5,
            rank=3,
            hidden=8,
            discount_weights=agent.discount_weights,
            tail_discount=agent.tail_discount,
        )
    rng_after = torch.random.get_rng_state()

    torch.testing.assert_close(rng_after, rng_before)


def test_masked_mse_and_advantage_normalization_exclude_invalid_rows():
    prediction = torch.tensor([1.0, 100.0, 5.0], requires_grad=True)
    target = torch.tensor([0.0, -100.0, 1.0])
    valid = torch.tensor([True, False, True])

    loss = MODULE.masked_mse(prediction, target, valid)
    normalized = MODULE.normalize_valid_advantages(prediction.detach(), valid)

    torch.testing.assert_close(loss, torch.tensor(8.5))
    torch.testing.assert_close(normalized, torch.tensor([-1.0, 0.0, 1.0]))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.0, 0.0, 4.0]))


def test_actor_backward_does_not_reach_predictive_heads():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, predictive_rank=3, prediction_horizon=5)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    _, _, log_prob, entropy, _, _, _ = agent.get_action_and_value(torch.randn(16, 3))

    (-(log_prob + 0.01 * entropy).mean()).backward()

    for module in (agent.coefficient_head, agent.temporal_basis, agent.tail_head):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_shift_targets_are_frozen_when_used_as_old_predictions():
    prediction = torch.randn(4, 6, requires_grad=True)
    old_next_prediction = torch.randn(4, 6, requires_grad=True)
    loss = torch.nn.functional.mse_loss(
        prediction[:, 1:], old_next_prediction.detach()[:, :-1]
    )

    loss.backward()

    assert prediction.grad is not None
    assert old_next_prediction.grad is None


def test_scalar_td0_alone_reaches_prefix_tail_and_shared_trunk():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, predictive_rank=3, prediction_horizon=5)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    _, _, _, value = agent.get_critic(torch.randn(32, 3))

    torch.nn.functional.mse_loss(value, torch.randn_like(value)).backward()

    assert any(parameter.grad is not None for parameter in agent.trunk.parameters())
    assert any(parameter.grad is not None for parameter in agent.coefficient_head.parameters())
    assert any(parameter.grad is not None for parameter in agent.temporal_basis.parameters())
    assert any(parameter.grad is not None for parameter in agent.tail_head.parameters())


def test_discounted_projection_detects_correlated_sequence_error_and_masks_rows():
    weights = torch.tensor([1.0, 0.5, 0.25])
    target = torch.zeros(2, 3)
    prediction = torch.tensor([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]])
    valid = torch.tensor([[True, True, True], [True, False, True]])

    loss = MODULE.discounted_projection_mse(prediction, target, valid, weights)

    torch.testing.assert_close(loss, torch.tensor(1.75**2))


def test_action_innovation_is_zero_at_behavior_mean_and_projection_is_exact():
    model = MODULE.ActionInnovation(
        obs_dim=3,
        act_dim=2,
        horizon=4,
        rank=3,
        hidden=8,
        discount_weights=torch.tensor([1.0, 0.8, 0.64, 0.512]),
        tail_discount=0.4096,
    )
    observations = torch.randn(6, 3)
    behavior_means = torch.randn(6, 2)

    future_zero, tail_zero, total_zero, total_slopes = model(
        observations, behavior_means, behavior_means
    )
    torch.testing.assert_close(future_zero, torch.zeros_like(future_zero))
    torch.testing.assert_close(tail_zero, torch.zeros_like(tail_zero))
    torch.testing.assert_close(total_zero, torch.zeros_like(total_zero))

    with torch.no_grad():
        model.coefficient_slope_head.weight.normal_()
        model.tail_slope_head.weight.normal_()
    actions = torch.randn(6, 2)
    future, tail, total, total_slopes = model(observations, actions, behavior_means)
    fast_total, fast_slopes = model.total(observations, actions, behavior_means)
    expected = future @ model.discount_weights + model.tail_discount * tail
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(fast_total, total)
    torch.testing.assert_close(fast_slopes, total_slopes)
    torch.testing.assert_close(
        total, ((actions - behavior_means) * total_slopes).sum(dim=-1)
    )


def test_affine_beta_expectation_matches_action_slope_formula():
    torch.manual_seed(7)
    alpha = torch.tensor([[2.5, 1.7]])
    beta = torch.tensor([[3.2, 4.1]])
    action_low = torch.tensor([-1.0, -2.0])
    action_high = torch.tensor([1.0, 2.0])
    old_mean = torch.tensor([[0.1, -0.2]])
    slopes = torch.tensor([[0.7, -1.3]])
    native = torch.distributions.Beta(alpha, beta).sample((200000,))
    actions = action_low + (action_high - action_low) * native
    empirical = ((actions - old_mean) * slopes).sum(dim=-1).mean(dim=0)
    new_mean = action_low + (action_high - action_low) * alpha / (alpha + beta)
    analytic = (slopes * (new_mean - old_mean)).sum(dim=-1)
    torch.testing.assert_close(empirical, analytic, atol=8e-3, rtol=0.0)


def test_masked_standardized_mse_is_scale_invariant():
    prediction = torch.tensor([1.0, 2.0, 100.0])
    target = torch.tensor([2.0, 4.0, -100.0])
    valid = torch.tensor([True, True, False])
    base = MODULE.masked_standardized_mse(prediction, target, valid)
    scaled = MODULE.masked_standardized_mse(10.0 * prediction, 10.0 * target, valid)
    torch.testing.assert_close(base, scaled)


def test_qprop_term_has_zero_population_value_and_gradient():
    # Exact two-action analogue with a uniform behavior policy. This verifies the
    # sampled-importance minus analytic-expectation signs used by the Beta actor.
    logit = torch.tensor(0.7, requires_grad=True)
    new_probability = logit.sigmoid()
    ratio = torch.stack([2.0 * (1.0 - new_probability), 2.0 * new_probability])
    slope = torch.tensor(1.3)
    sampled_innovation = slope * torch.tensor([-0.5, 0.5])
    expected_new_innovation = slope * (new_probability - 0.5)
    loss = MODULE.qprop_control_variate_loss(
        ratio,
        sampled_innovation,
        expected_new_innovation.unsqueeze(0),
        advantage_scale=torch.tensor(2.0),
        coefficient=1.0,
    )

    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(logit.grad, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_qprop_actor_backward_keeps_innovation_and_critic_heads_frozen():
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        predictive_rank=3,
        prediction_horizon=5,
    )
    agent = MODULE.Agent(DummyVectorEnv(), args)
    innovation = MODULE.ActionInnovation(
        obs_dim=3,
        act_dim=2,
        horizon=5,
        rank=3,
        hidden=8,
        discount_weights=agent.discount_weights,
        tail_discount=agent.tail_discount,
    )
    with torch.no_grad():
        innovation.coefficient_slope_head.weight.normal_()
        innovation.tail_slope_head.weight.normal_()
    observations = torch.randn(32, 3)
    action, z, old_logprob, _, _, _, old_mean = agent.get_action_and_value(observations)
    with torch.no_grad():
        sampled_h, slopes = innovation.total(observations, action, old_mean)
    _, _, new_logprob, _, _, _, new_mean = agent.get_action_and_value(observations, z)
    ratio = (new_logprob - old_logprob.detach()).exp()
    expected_h = (slopes * (new_mean - old_mean.detach())).sum(dim=-1)
    loss = MODULE.qprop_control_variate_loss(
        ratio, sampled_h, expected_h, torch.tensor(1.0), coefficient=1.0
    )

    loss.backward()

    assert any(parameter.grad is not None for parameter in agent.trunk.parameters())
    assert any(parameter.grad is not None for parameter in agent.actor_alpha_head.parameters())
    assert all(parameter.grad is None for parameter in innovation.parameters())
    for module in (agent.coefficient_head, agent.temporal_basis, agent.tail_head):
        assert all(parameter.grad is None for parameter in module.parameters())
