import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_temporal_residual_pcopt_v22.py"
SPEC = importlib.util.spec_from_file_location("temporal_residual_pcopt_v22", MODULE_PATH)
V22 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = V22
SPEC.loader.exec_module(V22)
V20_SPEC = importlib.util.spec_from_file_location(
    "multiscale_nextlat_v20_reference_for_pcopt_v22",
    ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_multiscale_nextlat_v20.py",
)
V20 = importlib.util.module_from_spec(V20_SPEC)
assert V20_SPEC.loader is not None
sys.modules[V20_SPEC.name] = V20
V20_SPEC.loader.exec_module(V20)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _agent(**overrides):
    values = dict(
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
        temporal_pc_macro_hidden=12,
    )
    values.update(overrides)
    return V22.Agent(_DummyEnvs(), V22.Args(**values))


def _problem(agent, *, batch=5):
    generator = torch.Generator().manual_seed(719)
    source = torch.randn(batch, 8, generator=generator)
    actions = torch.randn(16, batch, 3, generator=generator)
    targets = torch.randn(16, batch, 8, generator=generator)
    masks = torch.ones(16, batch)
    scales = torch.arange(1, 17, dtype=torch.float32).sqrt()
    decoder = V22.snapshot_policy_decoder(agent)
    anchor_targets = torch.stack(
        [targets[horizon - 1] for horizon in V22.TEMPORAL_PC_ANCHOR_HORIZONS]
    )
    first, second = V22.policy_parameters_from_snapshot(
        agent, anchor_targets.flatten(0, 1), decoder
    )
    first = first.reshape(len(V22.TEMPORAL_PC_ANCHOR_HORIZONS), batch, -1)
    second = second.reshape(len(V22.TEMPORAL_PC_ANCHOR_HORIZONS), batch, -1)
    policy_scales = torch.ones(len(V22.TEMPORAL_PC_ANCHOR_HORIZONS))
    return source, actions, targets, masks, scales, first, second, policy_scales, decoder


def _settle(agent, problem):
    return V22.temporal_pc_settle_forward(agent, *problem)


def test_defaults_are_one_fixed_temporal_pc_intervention():
    args = V22.Args()
    assert args.temporal_pc_enabled
    assert args.temporal_pc_inference_steps == 4
    assert args.temporal_pc_inference_damping == 0.5
    assert args.temporal_pc_trunk_grad_clip == 0.025
    assert args.temporal_pc_policy_precision == 0.1
    assert V22.TEMPORAL_PC_ANCHOR_HORIZONS == (1, 2, 4, 8, 16)
    assert V22.TEMPORAL_PC_MACRO_HORIZONS == (2, 4, 8, 16)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_disabled_mode_is_exact_v20_off_task_initialization_rng_and_adam(actor_dist):
    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist=actor_dist)
    torch.manual_seed(491)
    reference = V20.Agent(_DummyEnvs(), V20.Args(pc_mode="off", **common))
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(491)
    candidate = V22.Agent(
        _DummyEnvs(), V22.Args(temporal_pc_enabled=False, **common)
    )
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(candidate_rng, reference_rng)
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)
    assert not candidate.temporal_pc_parameters()
    assert candidate.temporal_pc_forward_flops() == 0
    assert len(candidate.task_parameters()) == len(list(candidate.parameters()))

    observations = torch.randn(6, 7)
    reference_outputs = V20.policy_model_forward(reference, observations)
    candidate_outputs = V22.policy_model_forward(candidate, observations)
    for actual, expected in zip(candidate_outputs, reference_outputs):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    reference_optimizer = torch.optim.Adam(
        reference.task_parameters(), lr=3e-4, eps=1e-5
    )
    candidate_optimizer = torch.optim.Adam(
        candidate.task_parameters(), lr=3e-4, eps=1e-5
    )
    sum(output.square().mean() for output in reference_outputs).backward()
    sum(output.square().mean() for output in candidate_outputs).backward()
    reference_optimizer.step()
    candidate_optimizer.step()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_dense_boundary_mask_matches_transition_oracle():
    generator = torch.Generator().manual_seed(787)
    boundaries = (torch.rand(23, 4, generator=generator) < 0.2).float()
    actual = V22.build_temporal_pc_mask(boundaries, 16)
    expected = torch.zeros_like(actual)
    for step in range(23):
        for env in range(4):
            for horizon in range(1, 17):
                expected[step, env, horizon - 1] = float(
                    step + horizon < 23
                    and not boundaries[step : step + horizon, env].bool().any()
                )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dense_h16_indices_follow_t_major_layout_without_future_leakage():
    sources = torch.tensor([0, 1, 5, 62, 63]).numpy()
    action, target = V22.make_temporal_pc_indices(sources, 4, 64, 16)
    expected_action = torch.as_tensor(sources)[None] + 4 * torch.arange(16)[:, None]
    expected_target = torch.as_tensor(sources)[None] + 4 * torch.arange(1, 17)[:, None]
    expected_action.clamp_(max=63)
    expected_target.clamp_(max=63)
    torch.testing.assert_close(torch.from_numpy(action), expected_action)
    torch.testing.assert_close(torch.from_numpy(target), expected_target)


def test_residual_modules_are_zero_identity_initialized_and_vectorized():
    agent = _agent()
    torch.testing.assert_close(
        agent.temporal_pc_transition.output.weight,
        torch.zeros_like(agent.temporal_pc_transition.output.weight),
    )
    torch.testing.assert_close(
        agent.temporal_pc_macro_factors.output.weight,
        torch.zeros_like(agent.temporal_pc_macro_factors.output.weight),
    )
    source, actions, _, masks, *_ = _problem(agent)
    rollout = V22.temporal_pc_chain_rollout(
        agent, source, actions, masks, frozen=False
    )
    torch.testing.assert_close(rollout, source.unsqueeze(0).expand_as(rollout))
    transition_source = inspect.getsource(V22.temporal_pc_energy_terms)
    assert "transition.forward_frozen(safe_chain_sources, safe_actions)" in transition_source
    assert "macro.forward_frozen(activities[0], safe_actions)" in transition_source


def test_activity_gradient_and_jacobi_step_match_autograd_oracle():
    agent = _agent(temporal_pc_policy_precision=0.0)
    problem = _problem(agent, batch=3)
    source, actions, targets, masks, scales, first, second, policy_scales, decoder = problem
    activities = V22.temporal_pc_chain_rollout(
        agent, source, actions, masks, frozen=True
    ).detach()
    gradient, value = V22._temporal_pc_activity_gradient_and_value(
        activities,
        agent,
        source,
        actions,
        targets,
        masks,
        scales,
        first,
        second,
        policy_scales,
        decoder,
    )
    leaf = activities.clone().requires_grad_(True)
    expected_value = V22._temporal_pc_activity_energy(
        leaf,
        agent,
        source,
        actions,
        targets,
        masks,
        scales,
        first,
        second,
        policy_scales,
        decoder,
    )
    expected_gradient = torch.autograd.grad(expected_value, leaf)[0]
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(gradient, expected_gradient)

    diagonal = V22.temporal_pc_activity_diagonal(agent, masks, scales, 8)
    expected_step = (
        activities
        - agent.temporal_pc_inference_damping
        * torch.where(diagonal > 0, expected_gradient / diagonal.clamp_min(1e-30), 0.0)
    ).detach()
    one_step_agent = _agent(
        temporal_pc_policy_precision=0.0,
        temporal_pc_inference_steps=1,
    )
    one_step_agent.load_state_dict(agent.state_dict())
    one_step_agent.temporal_pc_inference_steps = 1
    actual, _ = _settle(one_step_agent, problem)
    torch.testing.assert_close(actual, expected_step)


def test_settled_activities_are_detached_and_energy_decreases():
    agent = _agent()
    settled, energies = _settle(agent, _problem(agent))
    assert not settled.requires_grad
    assert not energies.requires_grad
    assert settled.grad_fn is None
    assert energies.shape == (5,)
    assert torch.all(energies[1:] <= energies[:-1] + 1e-6)


def test_local_parameter_graph_isolation_and_breaker_semantics():
    agent = _agent()
    problem = _problem(agent)
    source, actions, _, masks, scales, *_ = problem
    settled, _ = _settle(agent, problem)
    live = source.clone().requires_grad_(True)
    losses = V22.compute_temporal_pc_local_losses(
        agent, live, settled, actions, masks, scales
    )
    losses["predictor"].backward()
    assert live.grad is None
    assert all(parameter.grad is None for parameter in agent.temporal_pc_trunk_parameters())
    assert any(parameter.grad is not None for parameter in agent.temporal_pc_predictor_parameters())

    agent.zero_grad(set_to_none=True)
    losses = V22.compute_temporal_pc_local_losses(
        agent, live, settled, actions, masks, scales
    )
    losses["trunk"].backward()
    assert live.grad is not None
    assert all(parameter.grad is None for parameter in agent.temporal_pc_predictor_parameters())
    assert not V22.temporal_pc_source_feature(live, trunk_active=False).requires_grad
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.temporal_pc_predictor_parameters()}
    assert task_ids.isdisjoint(predictor_ids)


def test_closed_form_pc_feature_gradient_matches_local_root_autograd():
    agent = _agent()
    problem = _problem(agent, batch=4)
    source, actions, _, masks, scales, *_ = problem
    masks[0, 2] = 0.0
    settled, _ = _settle(agent, problem)
    leaf = source.clone().requires_grad_(True)
    coefficient = 0.7
    losses = V22.compute_temporal_pc_local_losses(
        agent, leaf, settled, actions, masks, scales
    )
    expected = torch.autograd.grad(
        coefficient * agent.temporal_pc_root_precision * losses["root"], leaf
    )[0]
    actual = V22.temporal_pc_local_root_feature_gradient(
        source,
        settled[0],
        masks[0],
        scales[0],
        coefficient=coefficient,
        precision=agent.temporal_pc_root_precision,
    )
    torch.testing.assert_close(actual, expected)
    reference = torch.randn_like(actual)
    cosine, ratio, actual_norm, reference_norm = (
        V22.temporal_pc_feature_gradient_statistics(actual, reference)
    )
    torch.testing.assert_close(
        cosine,
        torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        ),
    )
    torch.testing.assert_close(ratio, actual.norm() / reference.norm())
    torch.testing.assert_close(actual_norm, actual.norm())
    torch.testing.assert_close(reference_norm, reference.norm())


def test_parameter_vjp_exposes_nonisometric_jacobian_and_is_side_effect_free():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
    outputs = torch.stack((parameter[0], 10.0 * parameter[1]))
    pc_feature_gradient = torch.tensor([1.0, 1.0])
    bptt_feature_gradient = torch.tensor([1.0, -1.0])
    feature_cosine = torch.nn.functional.cosine_similarity(
        pc_feature_gradient, bptt_feature_gradient, dim=0
    )
    parameter_cosine, ratio, pc_norm, bptt_norm = (
        V22.temporal_pc_parameter_vjp_statistics(
            outputs,
            [parameter],
            pc_feature_gradient,
            bptt_feature_gradient,
        )
    )
    assert abs(float(parameter_cosine - feature_cosine)) > 0.5
    torch.testing.assert_close(
        parameter_cosine, torch.tensor(-99.0 / 101.0)
    )
    torch.testing.assert_close(ratio, torch.tensor(1.0))
    torch.testing.assert_close(pc_norm, torch.sqrt(torch.tensor(101.0)))
    torch.testing.assert_close(bptt_norm, torch.sqrt(torch.tensor(101.0)))
    assert parameter.grad is None


def test_real_trunk_parameter_vjps_do_not_accumulate_grads_or_mutate_adam():
    agent = _agent()
    task_parameters = agent.task_parameters()
    optimizer = torch.optim.Adam(task_parameters, lr=3e-4, eps=1e-5)
    observations = torch.randn(4, 7)
    warm_feature = agent.get_actor_feat(observations)
    warm_feature.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_before = {
        parameter: {
            name: value.clone() if torch.is_tensor(value) else value
            for name, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }
    live_feature = agent.get_actor_feat(observations)
    pc_feature_gradient = torch.randn_like(live_feature)
    bptt_feature_gradient = torch.randn_like(live_feature)
    statistics = V22.temporal_pc_parameter_vjp_statistics(
        live_feature,
        agent.temporal_pc_trunk_parameters(),
        pc_feature_gradient,
        bptt_feature_gradient,
    )
    assert all(torch.isfinite(value) for value in statistics)
    assert all(parameter.grad is None for parameter in agent.parameters())
    assert optimizer.state.keys() == optimizer_before.keys()
    for parameter, expected_state in optimizer_before.items():
        for name, expected in expected_state.items():
            actual = optimizer.state[parameter][name]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            else:
                assert actual == expected


def test_shared_trunk_critic_predictor_root_actor_backward_sequence_keeps_graph():
    agent = _agent()
    problem = _problem(agent, batch=4)
    _, actions, _, masks, scales, *_ = problem
    observations = torch.randn(4, 7)
    actor_feature, critic_feature = agent._trunks(observations)
    critic_loss = agent.critic_head(critic_feature).square().mean()
    actor_loss = (
        agent.actor_alpha_head(actor_feature).square().mean()
        + agent.actor_beta_head(actor_feature).square().mean()
    )
    settled = torch.randn(17, 4, 8)
    local = V22.compute_temporal_pc_local_losses(
        agent, actor_feature, settled, actions, masks, scales
    )

    agent.zero_grad(set_to_none=True)
    critic_loss.backward(retain_graph=True)
    agent.zero_grad(set_to_none=True)
    local["predictor"].backward()
    agent.zero_grad(set_to_none=True)
    local["trunk"].backward(retain_graph=True)
    agent.zero_grad(set_to_none=True)
    actor_loss.backward()
    assert any(parameter.grad is not None for parameter in agent.actor_parameters())


def test_invalid_poison_cannot_change_energy_settling_or_local_losses():
    agent = _agent(temporal_pc_policy_precision=0.0)
    clean = list(_problem(agent, batch=4))
    clean[3][5:, 1] = 0.0
    poison = [tensor.clone() if torch.is_tensor(tensor) else tensor for tensor in clean]
    poison[1][5:, 1] = float("nan")
    poison[2][5:, 1] = float("nan")
    clean_settled, clean_energy = _settle(agent, tuple(clean))
    poison_settled, poison_energy = _settle(agent, tuple(poison))
    torch.testing.assert_close(poison_energy, clean_energy)
    torch.testing.assert_close(poison_settled[:, [0, 2, 3]], clean_settled[:, [0, 2, 3]])
    assert torch.isfinite(poison_settled).all()
    clean_losses = V22.compute_temporal_pc_local_losses(
        agent, clean[0], clean_settled, clean[1], clean[3], clean[4]
    )
    poison_losses = V22.compute_temporal_pc_local_losses(
        agent, poison[0], poison_settled, poison[1], poison[3], poison[4]
    )
    for key in clean_losses:
        torch.testing.assert_close(poison_losses[key], clean_losses[key])


def test_movement_scales_are_scalar_relative_floored_and_scale_equivariant():
    generator = torch.Generator().manual_seed(321)
    features = torch.randn(80, 8, generator=generator)
    masks = torch.ones(80, 16)
    scales, raw, floor = V22.compute_rollout_movement_scales(
        features, masks, 2, 1e-3
    )
    scaled, scaled_raw, scaled_floor = V22.compute_rollout_movement_scales(
        7.0 * features, masks, 2, 1e-3
    )
    assert scales.shape == (16,)
    assert not scales.requires_grad
    torch.testing.assert_close(scaled, 7.0 * scales, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(scaled_raw, 7.0 * raw, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(scaled_floor, 7.0 * floor, rtol=2e-6, atol=2e-6)

    zero_scales, zero_raw, zero_floor = V22.compute_rollout_movement_scales(
        torch.zeros_like(features), masks, 2, 1e-3
    )
    assert torch.isfinite(zero_scales).all()
    assert torch.all(zero_scales > 0)
    assert torch.all(zero_scales.square() > 0)
    torch.testing.assert_close(zero_raw, torch.zeros_like(zero_raw))
    torch.testing.assert_close(zero_scales, zero_floor.expand_as(zero_scales))


def test_zero_policy_persistence_has_finite_additively_regularized_scale():
    agent = _agent()
    features = torch.zeros(80, 8)
    masks = torch.ones(80, 16)
    decoder = V22.snapshot_policy_decoder(agent)
    effective, raw, regularizer = V22.compute_rollout_policy_scales(
        agent, features, masks, 2, decoder, 1e-3
    )
    assert torch.isfinite(effective).all()
    assert torch.all(effective > 0)
    torch.testing.assert_close(raw, torch.zeros_like(raw))
    torch.testing.assert_close(regularizer, torch.zeros_like(regularizer))
    assert not effective.requires_grad


def test_policy_snapshot_is_cloned_and_semantic_gradient_never_reaches_head():
    agent = _agent()
    decoder = V22.snapshot_policy_decoder(agent)
    before = tuple(tensor.clone() for tensor in decoder)
    with torch.no_grad():
        agent.actor_alpha_head.weight.add_(10.0)
    for actual, expected in zip(decoder, before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    features = torch.randn(6, 8, requires_grad=True)
    first, second = V22.policy_parameters_from_snapshot(agent, features.detach(), decoder)
    loss = V22.policy_kl_from_snapshot(agent, features, first, second, decoder).mean()
    loss.backward()
    assert features.grad is not None
    assert all(parameter.grad is None for parameter in agent.actor_alpha_head.parameters())
    assert all(parameter.grad is None for parameter in agent.actor_beta_head.parameters())


def test_local_losses_equal_fixed_activity_energy_weight_partials():
    agent = _agent(
        temporal_pc_evidence_precision=0.0,
        temporal_pc_policy_precision=0.0,
    )
    problem = _problem(agent)
    source, actions, targets, masks, scales, first, second, policy_scales, decoder = problem
    settled, _ = _settle(agent, problem)
    local = V22.compute_temporal_pc_local_losses(
        agent, source, settled, actions, masks, scales
    )
    terms = V22.temporal_pc_energy_terms(
        agent,
        settled,
        source,
        actions,
        targets,
        masks,
        scales,
        first,
        second,
        policy_scales,
        decoder,
        frozen_factors=False,
    )
    torch.testing.assert_close(local["root"], terms[0].mean())
    torch.testing.assert_close(local["chain"], terms[1].mean())
    torch.testing.assert_close(local["macro"], terms[2].mean())


def test_free_forward_endpoint_metric_distinguishes_prediction_from_persistence():
    source = torch.zeros(4, 3)
    targets = torch.ones(16, 4, 3)
    predictions = targets.clone()
    predictions[1] = source
    masks = torch.ones(16, 4)
    # Invalid poison must not contaminate any scalar diagnostic.
    masks[4, 2] = 0.0
    targets[4, 2] = float("nan")
    predictions[4, 2] = float("nan")
    errors = V22.normalized_endpoint_errors_vs_persistence(
        predictions, source, targets, masks
    )
    assert torch.isfinite(errors).all()
    torch.testing.assert_close(errors[0], torch.tensor(0.0))
    torch.testing.assert_close(errors[1], torch.tensor(1.0))


def test_static_settle_wrapper_compiles_real_inductor_without_kl_dispatch(
    monkeypatch,
):
    agent = _agent()
    problem = _problem(agent, batch=3)

    def forbidden_distribution_dispatch(*_args, **_kwargs):
        raise AssertionError("compiled temporal PC must not call distribution KL dispatch")

    monkeypatch.setattr(
        torch.distributions,
        "kl_divergence",
        forbidden_distribution_dispatch,
    )
    compiled = torch.compile(
        lambda *values: V22.temporal_pc_settle_forward(agent, *values),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    expected = _settle(agent, problem)
    actual = compiled(*problem)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    retained = V22.retain_graph_output(actual[0], compiled=True)
    actual[0].zero_()
    torch.testing.assert_close(retained, expected[0])


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_closed_form_policy_kl_matches_distribution_reference_and_gradients(actor_dist):
    agent = _agent(actor_dist=actor_dist)
    decoder = V22.snapshot_policy_decoder(agent)
    target_features = torch.randn(7, 8, requires_grad=True)
    activity_features = torch.randn(7, 8, requires_grad=True)
    target_first, target_second = V22.policy_parameters_from_snapshot(
        agent, target_features, decoder
    )
    actual = V22.policy_kl_from_snapshot(
        agent,
        activity_features,
        target_first,
        target_second,
        decoder,
    )
    activity_first, activity_second = V22.policy_parameters_from_snapshot(
        agent, activity_features, decoder
    )
    if actor_dist == "gaussian":
        target_distribution = torch.distributions.Normal(
            target_first.detach(), (0.5 * target_second.detach()).exp()
        )
        activity_distribution = torch.distributions.Normal(
            activity_first, (0.5 * activity_second).exp()
        )
    else:
        target_distribution = torch.distributions.Beta(
            target_first.detach(), target_second.detach()
        )
        activity_distribution = torch.distributions.Beta(
            activity_first, activity_second
        )
    expected = torch.distributions.kl_divergence(
        target_distribution, activity_distribution
    ).sum(dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_gradient = torch.autograd.grad(
        actual.sum(), activity_features, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.sum(), activity_features)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6
    )
    assert torch.autograd.grad(
        actual.sum(), target_features, allow_unused=True
    )[0] is None
    helper_source = inspect.getsource(V22.policy_kl_from_snapshot)
    assert "kl_divergence" not in helper_source
    assert "torch.distributions" not in helper_source


def test_source_contains_feature_bptt_diagnostic_and_no_training_through_settling():
    source = MODULE_PATH.read_text()
    assert "temporal_pc_conversion/feature_gradient_cosine" in source
    assert "temporal_pc_conversion/feature_gradient_norm_ratio" in source
    assert "temporal_pc_conversion/trunk_parameter_gradient_cosine" in source
    assert "temporal_pc_conversion/trunk_parameter_gradient_norm_ratio" in source
    assert '"temporal_pc_free_forward/objective"' in source
    assert "bptt_activities[1:].detach()" in source
    assert "free_forward_normalized_error_vs_persistence" in source
    assert "source_reference = diagnostic_source.detach().requires_grad_(True)" in source
    assert "torch.autograd.grad(\n                args.temporal_pc_coef * bptt_loss, source_reference" in source
    assert "settled_activities = retain_graph_output" in source
    assert "temporal_pc_settle_fn = torch.compile(" in source
    assert "fullgraph=True" in source
    assert "args.compile_mode" in source
    assert "temporal_pc_inference/energy_ratio_" in source
    assert "diagnostic_source_live = agent.get_actor_feat(" in source
    assert "temporal_pc_parameter_vjp_statistics(" in source
    settle_source = inspect.getsource(V22.temporal_pc_settle_forward)
    assert ").detach()" in settle_source
    local_source = inspect.getsource(V22.compute_temporal_pc_local_losses)
    assert "settled = settled_activities.detach()" in local_source
