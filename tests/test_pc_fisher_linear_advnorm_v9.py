import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from cleanrl.ppo_continuous_action_pc_fisher_linear_advnorm_v9 import (
    Args,
    AugmentedLinearAscent,
    AugmentedAdamW,
    LocalPredictor,
    OutputPredictor,
    PCHierarchy,
    PostsynapticNeuronScoreRMS,
    StreamingAdvantageMoments,
    TraceBank,
    beta_fisher_metric,
    beta_head_scores,
    bootstrap_observations,
    metric_times,
    natural_score_target,
    normalized_actor_directions,
)


def directional_finite_difference(fn, value, direction, eps=1e-5):
    plus = fn(value + eps * direction)
    minus = fn(value - eps * direction)
    return (plus - minus) / (2.0 * eps)


def test_neuron_score_rms_is_lagged_and_shared_across_incoming_synapses():
    template = torch.empty(2, 2, 3)
    normalizer = PostsynapticNeuronScoreRMS([template], decay=0.5, minimum=0.1)
    first = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 0.0, -2.0]],
            [[-1.0, 1.0, 2.0], [1.0, 3.0, 1.0]],
        ]
    )
    preconditioned, scales = normalizer.precondition_and_update(
        [first], torch.tensor([True, True])
    )
    torch.testing.assert_close(preconditioned[0], first)
    torch.testing.assert_close(scales[0], torch.ones(2, 1))
    expected_moment = first.square().mean(dim=(0, 2)).unsqueeze(1)
    torch.testing.assert_close(normalizer.mean_squares[0], expected_moment)

    second = torch.full_like(first, 100.0)
    preconditioned, scales = normalizer.precondition_and_update(
        [second], torch.tensor([True, True])
    )
    expected_scale = expected_moment.sqrt().reciprocal()
    torch.testing.assert_close(scales[0], expected_scale)
    torch.testing.assert_close(preconditioned[0], second * expected_scale)
    assert scales[0].shape == (2, 1)


def test_advantage_clipping_is_identical_for_proxy_and_reward_modulation():
    moments = StreamingAdvantageMoments(
        torch.device("cpu"), num_envs=2, decay=0.5, minimum=0.1, stats_winsor=0.0
    )
    mean, std, actor_delta = moments.assign_and_update(
        torch.tensor([10.0, -10.0]),
        torch.tensor([False, False]),
        trace_decay=0.7,
        clip=2.0,
    )
    torch.testing.assert_close(mean, torch.tensor(0.0))
    torch.testing.assert_close(std, torch.tensor(1.0))
    torch.testing.assert_close(actor_delta, torch.tensor([2.0, -2.0]))
    # The exact tensor returned for reward modulation also enters the proxy.
    torch.testing.assert_close(moments.proxy, actor_delta)
    torch.testing.assert_close(moments.mean, torch.tensor(0.0))
    torch.testing.assert_close(moments.variance, torch.tensor(4.0))


def test_nonfinite_advantages_and_scores_do_not_contaminate_moments():
    moments = StreamingAdvantageMoments(
        torch.device("cpu"), num_envs=2, decay=0.9, minimum=0.1, stats_winsor=0.0
    )
    moments.assign_and_update(
        torch.tensor([1.0, 3.0]),
        torch.tensor([False, False]),
        trace_decay=0.8,
        clip=0.0,
    )
    old_mean = moments.mean.clone()
    old_variance = moments.variance.clone()
    _, _, actor_delta = moments.assign_and_update(
        torch.tensor([float("nan"), float("inf")]),
        torch.tensor([False, False]),
        trace_decay=0.8,
        clip=2.0,
    )
    torch.testing.assert_close(actor_delta, torch.zeros(2))
    torch.testing.assert_close(moments.proxy, torch.zeros(2))
    torch.testing.assert_close(moments.mean, old_mean)
    torch.testing.assert_close(moments.variance, old_variance)

    normalizer = PostsynapticNeuronScoreRMS(
        [torch.empty(2, 1, 3)], decay=0.9, minimum=0.1
    )
    normalizer.precondition_and_update(
        [torch.ones(2, 1, 3)], torch.tensor([True, True])
    )
    old_score_moment = normalizer.mean_squares[0].clone()
    preconditioned, _ = normalizer.precondition_and_update(
        [torch.full((2, 1, 3), float("nan"))], torch.tensor([False, False])
    )
    torch.testing.assert_close(preconditioned[0], torch.zeros(2, 1, 3))
    torch.testing.assert_close(normalizer.mean_squares[0], old_score_moment)


def test_cumulative_linear_parameter_updates_equal_offline_normalized_returns():
    trace_decay = 0.7
    raw_scores = torch.tensor(
        [
            [[1.0, -0.5, 0.2]],
            [[0.4, 2.0, -1.0]],
            [[-1.0, 0.3, 0.7]],
            [[0.8, -0.2, 1.5]],
            [[2.0, 0.5, -0.4]],
            [[-0.3, 1.2, 0.6]],
        ]
    )
    raw_deltas = torch.tensor([4.0, -3.0, 0.8, 2.5, -4.0, 1.1])
    done = torch.tensor([False, True, False, False, False, False])
    creation_lrs = torch.tensor([0.08, 0.06, 0.05, 0.04, 0.03, 0.02])

    edge = LocalPredictor(2, 1, "identity", std=0.1)
    initial = torch.cat([edge.weight, edge.bias[:, None]], dim=1).clone()
    updater = AugmentedLinearAscent([edge])
    reward_trace = TraceBank([torch.empty(1, 1, 3)])
    baseline_trace = TraceBank([torch.empty(1, 1, 3)])
    score_rms = PostsynapticNeuronScoreRMS(
        [torch.empty(1, 1, 3)], decay=0.5, minimum=0.2
    )
    moments = StreamingAdvantageMoments(
        torch.device("cpu"), num_envs=1, decay=0.5, minimum=0.2, stats_winsor=0.0
    )

    assigned_scores = []
    assigned_means = []
    assigned_stds = []
    actor_deltas = []
    for step in range(len(raw_deltas)):
        step_done = done[step].view(1)
        mean, std, actor_delta = moments.assign_and_update(
            raw_deltas[step].view(1), step_done, trace_decay, clip=1.5
        )
        preconditioned, _ = score_rms.precondition_and_update(
            [raw_scores[step].unsqueeze(0)], torch.tensor([True])
        )
        assigned_score = preconditioned[0]
        creation_scale = creation_lrs[step] / std
        reward_trace.accumulate([assigned_score * creation_scale], trace_decay)
        baseline_trace.accumulate(
            [assigned_score * (creation_scale * mean)], trace_decay
        )
        direction = normalized_actor_directions(
            reward_trace,
            baseline_trace,
            actor_delta,
            step_done | (step == len(raw_deltas) - 1),
            trace_decay,
        )
        updater.step(direction)
        reward_trace.reset(step_done)
        baseline_trace.reset(step_done)
        assigned_scores.append(assigned_score.squeeze(0).clone())
        assigned_means.append(mean.clone())
        assigned_stds.append(std.clone())
        actor_deltas.append(actor_delta.squeeze().clone())

    offline = torch.zeros_like(initial)
    episode_start = 0
    for episode_end in [1, len(raw_deltas) - 1]:
        for score_step in range(episode_start, episode_end + 1):
            lambda_return = sum(
                trace_decay ** (future_step - score_step) * actor_deltas[future_step]
                for future_step in range(score_step, episode_end + 1)
            )
            offline += creation_lrs[score_step] * assigned_scores[score_step] * (
                (lambda_return - assigned_means[score_step])
                / assigned_stds[score_step]
            )
        episode_start = episode_end + 1
    actual = torch.cat([edge.weight, edge.bias[:, None]], dim=1) - initial
    assert any(
        not torch.equal(raw_deltas[idx], actor_deltas[idx])
        for idx in range(len(raw_deltas))
    )
    torch.testing.assert_close(actual, offline, rtol=2e-6, atol=2e-6)


def test_invalid_score_flushes_prior_credit_and_baseline_before_trace_reset():
    trace_decay = 0.5
    edge = LocalPredictor(1, 1, "identity", std=0.1)
    with torch.no_grad():
        edge.weight.zero_()
        edge.bias.zero_()
    updater = AugmentedLinearAscent([edge])
    reward_trace = TraceBank([torch.empty(2, 1, 2)])
    baseline_trace = TraceBank([torch.empty(2, 1, 2)])

    first_scores = torch.tensor([[[0.0, 2.0]], [[0.0, 3.0]]])
    reward_trace.accumulate([first_scores], trace_decay)
    baseline_trace.accumulate([first_scores], trace_decay)
    updater.step(
        normalized_actor_directions(
            reward_trace,
            baseline_trace,
            torch.tensor([4.0, 5.0]),
            torch.tensor([False, False]),
            trace_decay,
        )
    )

    # Environment zero has no valid current score. Its zero score is skipped,
    # but its prior eligibility still receives delta=6 and a terminal baseline
    # coefficient before the row is reset. Environment one ends normally.
    valid = torch.tensor([False, True])
    second_scores = torch.tensor([[[0.0, 0.0]], [[0.0, 7.0]]])
    reward_trace.accumulate([second_scores], trace_decay)
    baseline_trace.accumulate([second_scores], trace_decay)
    updater.step(
        normalized_actor_directions(
            reward_trace,
            baseline_trace,
            torch.tensor([6.0, 8.0]),
            ~valid | torch.tensor([False, True]),
            trace_decay,
        )
    )
    reward_trace.reset(~valid | torch.tensor([False, True]))
    baseline_trace.reset(~valid | torch.tensor([False, True]))

    expected_env_zero = 2.0 * ((4.0 + trace_decay * 6.0) - 1.0)
    expected_env_one = (
        3.0 * ((5.0 + trace_decay * 8.0) - 1.0)
        + 7.0 * (8.0 - 1.0)
    )
    torch.testing.assert_close(
        edge.bias,
        torch.tensor([(expected_env_zero + expected_env_one) / 2.0]),
    )
    torch.testing.assert_close(reward_trace.traces[0], torch.zeros(2, 1, 2))
    torch.testing.assert_close(baseline_trace.traces[0], torch.zeros(2, 1, 2))


def test_v9_defaults_to_no_decay_and_calibrated_actor_rate():
    args = Args()
    assert args.weight_decay == 0.0
    assert args.actor_learning_rate == 8e-4


def test_beta_raw_score_matches_directional_finite_difference():
    torch.manual_seed(0)
    raw = torch.randn(4, 6, dtype=torch.float64)
    alpha_raw, beta_raw = raw.chunk(2, dim=1)
    action = torch.rand(4, 3, dtype=torch.float64).clamp(0.05, 0.95)
    alpha, beta = 1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)
    score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, action)
    direction = torch.randn_like(raw)

    def objective(candidate):
        candidate_alpha, candidate_beta = candidate.chunk(2, dim=1)
        dist = Beta(1.0 + F.softplus(candidate_alpha), 1.0 + F.softplus(candidate_beta))
        return dist.log_prob(action).sum()

    finite_difference = directional_finite_difference(objective, raw, direction)
    torch.testing.assert_close(finite_difference, (score * direction).sum(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("scale", [0.1, 1.0, 8.0])
def test_beta_fisher_is_spd_and_natural_target_has_exact_score_force(scale):
    torch.manual_seed(1)
    alpha_raw = scale * torch.randn(7, 3)
    beta_raw = scale * torch.randn(7, 3)
    alpha, beta = 1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)
    action = torch.rand(7, 3).clamp(1e-5, 1.0 - 1e-5)
    score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, action)
    metric = beta_fisher_metric(alpha_raw, beta_raw, alpha, beta, damping=1e-3)
    torch.linalg.cholesky(metric)
    output = torch.cat([alpha_raw, beta_raw], dim=1)
    nudge = 0.037
    target = natural_score_target(output, metric, score, nudge)
    recovered = metric_times(metric, target - output) / nudge
    torch.testing.assert_close(recovered, score, rtol=3e-4, atol=3e-4)


def test_beta_fisher_matches_score_covariance_monte_carlo():
    torch.manual_seed(2)
    alpha_raw = torch.tensor([[0.4]], dtype=torch.float64)
    beta_raw = torch.tensor([[-0.7]], dtype=torch.float64)
    alpha, beta = 1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)
    samples = Beta(alpha, beta).sample((300_000,)).view(-1, 1)
    repeated_a = alpha_raw.expand_as(samples)
    repeated_b = beta_raw.expand_as(samples)
    scores = beta_head_scores(
        repeated_a, repeated_b, alpha.expand_as(samples), beta.expand_as(samples), samples
    )
    empirical = scores.T @ scores / scores.shape[0]
    analytic = beta_fisher_metric(alpha_raw, beta_raw, alpha, beta, damping=1e-12)[0]
    torch.testing.assert_close(empirical, analytic, rtol=1.5e-2, atol=1.5e-3)


def test_free_states_have_zero_residual_without_hidden_clamps():
    torch.manual_seed(3)
    args = Args(hidden_size=5, pc_num_hidden_layers=3)
    hierarchy = PCHierarchy(4, args.hidden_size, args)
    with torch.no_grad():
        hierarchy.edges[0].weight.mul_(20.0)
        hierarchy.edges[0].bias.fill_(9.0)
    observation = torch.randn(6, 4)
    free = hierarchy.initial_states(observation)
    assert free[0].abs().max() > 5.0
    for residual in hierarchy.residuals(observation, free):
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_top_curvature_contains_full_output_metric():
    torch.manual_seed(4)
    args = Args(hidden_size=4, pc_num_hidden_layers=2, pc_curvature_damping=0.07)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 3, std=0.2)
    observation = torch.randn(8, 3)
    states = hierarchy.initial_states(observation)
    raw = torch.randn(8, 3, 3)
    metric = raw @ raw.transpose(1, 2) + 0.2 * torch.eye(3)
    factors, _, _ = hierarchy.curvature_factors(states, output, metric, args)
    recovered = factors[-1] @ factors[-1].T
    expected = (
        (1.0 + args.pc_curvature_damping) * torch.eye(args.hidden_size)
        + output.weight.T @ metric.mean(0) @ output.weight
    )
    torch.testing.assert_close(recovered, expected, rtol=2e-6, atol=2e-6)


def _settled_directions(nudge):
    torch.manual_seed(5)
    args = Args(
        hidden_size=5,
        pc_num_hidden_layers=3,
        pc_inference_steps=20,
        pc_inference_scale=1.0,
        pc_curvature_damping=0.05,
    )
    hierarchy = PCHierarchy(4, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 2, std=0.15)
    observation = 0.3 * torch.randn(9, 4)
    free = hierarchy.initial_states(observation)
    prediction = output(free[-1])
    metric = torch.tensor([[1.7, 0.25], [0.25, 0.8]]).expand(9, -1, -1).clone()
    force = torch.randn(9, 2)
    target = prediction + nudge * torch.linalg.solve(metric, force.unsqueeze(-1)).squeeze(-1)
    states, _, _ = hierarchy.settle(observation, output, target, metric, args)
    directions = hierarchy.local_scores(observation, states, nudge)
    output_error = metric_times(metric, target - output(states[-1]))
    directions.append(
        output_error.unsqueeze(2) * output.augmented_features(states[-1]).unsqueeze(1) / nudge
    )
    return directions


def test_inverse_nudge_compensation_has_linear_small_nudge_limit():
    medium = _settled_directions(2e-3)
    small = _settled_directions(1e-3)
    for medium_direction, small_direction in zip(medium, small):
        cosine = F.cosine_similarity(medium_direction.flatten(), small_direction.flatten(), dim=0)
        norm_ratio = medium_direction.norm() / small_direction.norm().clamp_min(1e-12)
        assert cosine > 0.9999
        assert 0.995 < norm_ratio < 1.005


def _actor_pc_and_exact_score_cosine(head_scale):
    torch.manual_seed(42)
    args = Args(
        hidden_size=16,
        pc_num_hidden_layers=6,
        pc_inference_steps=10,
        pc_actor_nudge=0.05,
    )
    hierarchy = PCHierarchy(7, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.01)
    with torch.no_grad():
        output.weight.mul_(head_scale)
    observation = 0.2 * torch.randn(16, 7)
    free = hierarchy.initial_states(observation)
    raw = output(free[-1])
    alpha_raw, beta_raw = raw.chunk(2, dim=1)
    alpha, beta = 1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)
    action = torch.rand(16, 2).clamp(0.05, 0.95)
    score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, action)
    metric = beta_fisher_metric(alpha_raw, beta_raw, alpha, beta, args.pc_fisher_damping)
    target = natural_score_target(raw, metric, score, args.pc_actor_nudge)
    states, _, _ = hierarchy.settle(observation, output, target, metric, args)
    pc_directions = hierarchy.local_scores(observation, states, args.pc_actor_nudge)
    output_error = metric_times(metric, target - output(states[-1]))
    pc_directions.append(
        output_error.unsqueeze(2)
        * output.augmented_features(states[-1]).unsqueeze(1)
        / args.pc_actor_nudge
    )
    pc_flat = torch.cat([direction.sum(0).flatten() for direction in pc_directions])

    parameters = []
    for edge in [*hierarchy.edges, output]:
        edge.weight.requires_grad_(True)
        edge.bias.requires_grad_(True)
        parameters.extend([edge.weight, edge.bias])
    source = observation
    for edge in hierarchy.edges:
        source = edge(source)
    exact_raw = output(source)
    exact_alpha_raw, exact_beta_raw = exact_raw.chunk(2, dim=1)
    objective = Beta(
        1.0 + F.softplus(exact_alpha_raw), 1.0 + F.softplus(exact_beta_raw)
    ).log_prob(action).sum()
    gradients = torch.autograd.grad(objective, parameters)
    exact_augmented = [
        torch.cat([weight_gradient, bias_gradient[:, None]], dim=1).flatten()
        for weight_gradient, bias_gradient in zip(gradients[::2], gradients[1::2])
    ]
    exact_flat = torch.cat(exact_augmented)
    return F.cosine_similarity(pc_flat, exact_flat, dim=0)


def test_six_layer_compensated_actor_direction_tracks_exact_score_at_runtime_nudge():
    ordinary_cosine = _actor_pc_and_exact_score_cosine(head_scale=1.0)
    high_sensitivity_cosine = _actor_pc_and_exact_score_cosine(head_scale=100.0)
    assert ordinary_cosine > 0.99
    # Strong output sensitivity changes the coherent PC equilibrium geometry;
    # the direction must remain positively aligned even in this stress regime.
    assert high_sensitivity_cosine > 0.70
    assert high_sensitivity_cosine < ordinary_cosine


def test_ten_sweeps_reduce_total_coherent_pc_energy():
    torch.manual_seed(61)
    args = Args(hidden_size=8, pc_num_hidden_layers=6, pc_inference_steps=10)
    hierarchy = PCHierarchy(5, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 4, std=0.2)
    observation = torch.randn(16, 5)
    free = hierarchy.initial_states(observation)
    prediction = output(free[-1])
    raw_metric = torch.randn(16, 4, 4)
    metric = raw_metric @ raw_metric.transpose(1, 2) + 0.1 * torch.eye(4)
    force = torch.randn(16, 4)
    target = prediction + 0.05 * torch.linalg.solve(metric, force.unsqueeze(-1)).squeeze(-1)

    def energy(states):
        hidden = sum(0.5 * residual.square().sum(1) for residual in hierarchy.residuals(observation, states))
        output_residual = target - output(states[-1])
        return hidden + 0.5 * torch.einsum(
            "bi,bij,bj->b", output_residual, metric, output_residual
        )

    before = energy(free)
    settled, _, _ = hierarchy.settle(observation, output, target, metric, args)
    after = energy(settled)
    assert after.sum() < before.sum()
    assert torch.all(after <= before + 1e-6)


def test_quadratic_output_local_score_is_negative_energy_direction():
    torch.manual_seed(6)
    output = OutputPredictor(3, 2, std=0.2).double()
    source = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    raw = torch.randn(5, 2, 2, dtype=torch.float64)
    metric = raw @ raw.transpose(1, 2) + 0.3 * torch.eye(2, dtype=torch.float64)
    precision_error = metric_times(metric, target - output(source))
    phi = output.augmented_features(source)
    direction = (precision_error.unsqueeze(2) * phi.unsqueeze(1)).sum(0)
    augmented = torch.cat([output.weight, output.bias[:, None]], dim=1).detach()

    def log_likelihood(candidate):
        prediction = F.linear(source, candidate[:, :-1], candidate[:, -1])
        residual = target - prediction
        return -0.5 * torch.einsum("bi,bij,bj->", residual, metric, residual)

    finite_difference = directional_finite_difference(log_likelihood, augmented, direction)
    torch.testing.assert_close(finite_difference, direction.square().sum(), rtol=2e-6, atol=2e-6)


def test_trace_is_per_environment_and_terminal_delta_precedes_reset():
    trace = TraceBank([torch.empty(3, 1)])
    trace.accumulate([torch.tensor([[1.0], [2.0], [3.0]])], decay=0.5)
    trace.accumulate([torch.tensor([[2.0], [0.0], [-1.0]])], decay=0.5)
    update = trace.modulated_mean(torch.tensor([0.0, 6.0, 0.0]))[0]
    torch.testing.assert_close(update, torch.tensor([2.0]))
    trace.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(trace.traces[0].squeeze(), torch.tensor([2.5, 0.0, 0.5]))


def test_augmented_adamw_has_parameter_moments_and_does_not_decay_bias():
    edge = LocalPredictor(2, 1, "identity", std=0.1)
    with torch.no_grad():
        edge.weight.fill_(2.0)
        edge.bias.fill_(3.0)
    optimizer = AugmentedAdamW([edge], beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.1)
    direction = torch.tensor([[4.0, -2.0, 0.0]])
    optimizer.step([direction], learning_rate=0.01)
    torch.testing.assert_close(edge.weight, torch.tensor([[2.008, 1.988]]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(edge.bias, torch.tensor([3.0]), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.first[0], 0.1 * direction)
    torch.testing.assert_close(optimizer.second[0], 0.001 * direction.square())


def test_all_local_parameters_are_frozen_to_autograd():
    args = Args(hidden_size=4, pc_num_hidden_layers=2)
    hierarchy = PCHierarchy(3, args.hidden_size, args)
    output = OutputPredictor(args.hidden_size, 2, std=0.1)
    assert all(not parameter.requires_grad for parameter in hierarchy.parameters())
    assert all(not parameter.requires_grad for parameter in output.parameters())


def test_truncation_bootstraps_from_final_observation():
    next_obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    infos = {
        "final_observation": np.array([np.array([8.0, 9.0]), None], dtype=object),
        "_final_observation": np.array([True, False]),
    }
    result = bootstrap_observations(next_obs, np.array([True, False]), infos)
    np.testing.assert_array_equal(result[0], np.array([8.0, 9.0]))
    np.testing.assert_array_equal(result[1], next_obs[1])
