import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_freegoal_exact_cost_v3.py"
)
SPEC = importlib.util.spec_from_file_location("freegoal_exact_cost_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def small_args():
    return MODULE.Args(
        history_length=4,
        latent_dim=4,
        hidden_dim=16,
        transformer_layers=1,
        transformer_heads=2,
        predictor_heads=2,
        value_heads=2,
    )


def test_straight_through_is_factual_forward_and_predicted_backward():
    source = torch.tensor([2.0], requires_grad=True)
    predicted = source.square()
    actual = torch.tensor([11.0])
    training_next = MODULE.straight_through_actual(predicted, actual)
    torch.testing.assert_close(training_next, actual)
    training_next.backward()
    torch.testing.assert_close(source.grad, torch.tensor([4.0]))


def test_deterministic_action_is_repeatable():
    args = small_args()
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args).eval()
    observations = torch.randn(5, args.history_length, 3)
    actions = torch.randn(5, args.history_length, 2)
    first = agent.act(observations, actions)
    second = agent.act(observations, actions)
    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_action_excitation_identifies_every_action_coordinate():
    action_dim = 6
    period = 257
    probes = torch.stack(
        [
            MODULE.deterministic_action_excitation(
                step,
                num_envs=1,
                action_dim=action_dim,
                amplitude=0.12,
                period=period,
            )[0]
            for step in range(period)
        ]
    )
    centered = probes - probes.mean(dim=0, keepdim=True)
    assert torch.linalg.matrix_rank(centered) == action_dim


def test_goal_excitation_identifies_every_latent_coordinate():
    latent_dim = 32
    period = 379
    probes = torch.stack(
        [
            MODULE.deterministic_goal_excitation(
                step,
                num_envs=1,
                latent_dim=latent_dim,
                amplitude=0.15,
                period=period,
            )[0]
            for step in range(period)
        ]
    )
    centered = probes - probes.mean(dim=0, keepdim=True)
    assert torch.linalg.matrix_rank(centered) == latent_dim


def test_post_norm_projector_restores_variable_belief_magnitude():
    args = small_args()
    world = MODULE.CausalHistoryWorldModel(
        obs_dim=3,
        action_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        predictor_heads=args.predictor_heads,
    )
    observations = torch.randn(32, args.history_length, 3)
    actions = torch.randn(32, args.history_length, 2)
    beliefs = world.belief_raw(observations, actions)
    assert beliefs.norm(dim=-1).std() > 1e-3

    sigreg = MODULE.SIGReg(projections=16, knots=5, reference_samples=32)
    sigreg(beliefs).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in world.belief_projector.parameters()
    )


def test_goal_is_unbounded_and_not_directionally_projected():
    goal = MODULE.FreeGoalNetwork(3, 8)
    with torch.no_grad():
        goal.network[-1].weight.zero_()
        goal.network[-1].bias.fill_(9.0)
    output = goal(torch.zeros(2, 3))
    torch.testing.assert_close(output, torch.full((2, 3), 9.0))
    assert torch.all(output.norm(dim=-1) > 15)


def test_state_value_signature_is_state_only():
    parameters = list(inspect.signature(MODULE.StateValueEnsemble.forward).parameters)
    assert parameters == ["self", "belief"]


def test_reward_signature_is_state_transition_only():
    parameters = list(
        inspect.signature(MODULE.TransitionRewardEnsemble.forward).parameters
    )
    assert parameters == ["self", "belief", "next_belief"]


def test_actor_gradients_reach_goal_and_policy_but_not_frozen_models():
    args = small_args()
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    beliefs = torch.randn(7, args.latent_dim)
    next_beliefs = torch.randn(7, args.latent_dim)
    loss, _, _ = MODULE.factual_st_actor_loss(
        agent,
        beliefs,
        next_beliefs,
        torch.zeros(7, 2),
        torch.zeros(7, args.latent_dim),
        0.1,
        args,
    )
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.goal.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.policy.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (agent.world, agent.reward, agent.value)
        for parameter in module.parameters()
    )


def test_delivery_does_not_directly_regress_goal_to_actual_next():
    args = small_args()
    args.uncertainty_coef = 0.0
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.zero_()
        for parameter in agent.value.parameters():
            parameter.zero_()
        for parameter in agent.reward.parameters():
            parameter.zero_()
    beliefs = torch.randn(7, args.latent_dim)
    next_beliefs = torch.randn(7, args.latent_dim)
    loss, _, _ = MODULE.factual_st_actor_loss(
        agent,
        beliefs,
        next_beliefs,
        torch.zeros(7, 2),
        torch.zeros(7, args.latent_dim),
        0.0,
        args,
    )
    loss.backward()
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in agent.goal.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.policy.parameters()
    )


def test_exact_control_cost_matches_halfcheetah_units_and_gradient():
    actions = torch.tensor(
        [[1.0, -0.5, 0.25, 0.0, -1.0, 0.5]], requires_grad=True
    )
    cost = MODULE.exact_control_cost(
        actions, ctrl_cost_weight=0.1, reward_scale=0.1
    )
    expected = 0.01 * actions.detach().square().sum(dim=-1)
    torch.testing.assert_close(cost, expected)
    cost.sum().backward()
    torch.testing.assert_close(actions.grad, 0.02 * actions.detach())


def test_exact_control_cost_uses_environment_configured_weight():
    wrapped = [
        SimpleNamespace(
            unwrapped=SimpleNamespace(_ctrl_cost_weight=0.1)
        )
        for _ in range(3)
    ]
    vector_env = SimpleNamespace(envs=wrapped)
    assert MODULE.configured_ctrl_cost_weight(vector_env) == 0.1


def test_forward_reward_reconstructs_total_without_double_counting():
    total = torch.tensor([-0.3, 1.2])
    actions = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    control = MODULE.exact_control_cost(actions, ctrl_cost_weight=0.1)
    forward = MODULE.forward_reward_from_total(
        total, actions, ctrl_cost_weight=0.1
    )
    torch.testing.assert_close(forward - control, total)


def test_actor_objective_subtracts_exact_scaled_cost_once():
    args = small_args()
    args.goal_delivery_coef = 0.0
    args.uncertainty_coef = 0.0
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    with torch.no_grad():
        for module in (agent.reward, agent.value):
            for parameter in module.parameters():
                parameter.zero_()
        agent.policy.network[-1].weight.zero_()
        agent.policy.network[-1].bias.fill_(0.5)
    beliefs = torch.randn(7, args.latent_dim)
    next_beliefs = torch.randn(7, args.latent_dim)
    loss, metrics, action = MODULE.factual_st_actor_loss(
        agent,
        beliefs,
        next_beliefs,
        torch.zeros(7, 2),
        torch.zeros(7, args.latent_dim),
        0.1,
        args,
    )
    expected = MODULE.exact_control_cost(
        action, 0.1, args.reward_scale
    ).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        metrics["exact_scaled_control_cost"], expected
    )


def test_route_diagnostics_are_observational_complete_and_finite():
    args = small_args()
    plain_agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    diagnosed_agent = copy.deepcopy(plain_agent)
    beliefs = torch.randn(7, args.latent_dim)
    next_beliefs = torch.randn(7, args.latent_dim)
    action_excitation = torch.zeros(7, 2)
    goal_excitation = torch.zeros(7, args.latent_dim)
    plain_loss, plain_metrics, plain_actions = MODULE.factual_st_actor_loss(
        plain_agent,
        beliefs,
        next_beliefs,
        action_excitation,
        goal_excitation,
        0.1,
        args,
    )
    diagnosed_loss, metrics, diagnosed_actions = MODULE.factual_st_actor_loss(
        diagnosed_agent,
        beliefs,
        next_beliefs,
        action_excitation,
        goal_excitation,
        0.1,
        args,
        diagnose_routes=True,
    )
    expected = {
        "return_action_pressure_norm",
        "delivery_action_pressure_norm",
        "exact_cost_action_pressure_norm",
        "uncertainty_action_pressure_norm",
        "return_raw_policy_pressure_norm",
        "delivery_raw_policy_pressure_norm",
        "exact_cost_raw_policy_pressure_norm",
        "uncertainty_raw_policy_pressure_norm",
        "return_exact_cost_pressure_cosine",
        "return_delivery_pressure_cosine",
        "return_uncertainty_pressure_cosine",
    }
    assert not expected.intersection(plain_metrics)
    assert expected <= metrics.keys()
    assert all(torch.isfinite(metrics[key]) for key in expected)
    torch.testing.assert_close(plain_loss, diagnosed_loss)
    torch.testing.assert_close(plain_actions, diagnosed_actions)

    plain_loss.backward()
    diagnosed_loss.backward()
    for plain_parameter, diagnosed_parameter in zip(
        list(plain_agent.goal.parameters())
        + list(plain_agent.policy.parameters()),
        list(diagnosed_agent.goal.parameters())
        + list(diagnosed_agent.policy.parameters()),
    ):
        torch.testing.assert_close(
            plain_parameter.grad, diagnosed_parameter.grad
        )


def test_route_diagnostics_default_and_validation_are_explicit():
    assert MODULE.Args().route_diagnostics_interval == 32
    source = SCRIPT.read_text()
    assert 'raise ValueError("route_diagnostics_interval must be positive")' in source
    assert "iteration % args.route_diagnostics_interval == 0" in source


def test_representation_prediction_target_receives_gradient():
    predicted = torch.randn(2, 6, 4, requires_grad=True)
    actual = torch.randn(6, 4, requires_grad=True)
    loss = MODULE.prediction_mse_with_live_target(predicted, actual)
    loss.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert actual.grad is not None and actual.grad.abs().sum() > 0


def test_attached_target_gradient_reaches_history_encoder_with_fixed_prediction():
    args = small_args()
    world = MODULE.CausalHistoryWorldModel(
        obs_dim=3,
        action_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        predictor_heads=args.predictor_heads,
    )
    target_observations = torch.randn(
        5, args.history_length, 3, requires_grad=True
    )
    target_actions = torch.randn(5, args.history_length, 2)
    actual = world.belief(target_observations, target_actions)
    fixed_prediction = torch.zeros(
        args.predictor_heads, 5, args.latent_dim, requires_grad=True
    )
    loss = MODULE.prediction_mse_with_live_target(
        fixed_prediction, actual
    )
    loss.backward()
    assert (
        target_observations.grad is not None
        and target_observations.grad.abs().sum() > 0
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in world.observation_encoder.parameters()
    )


def test_representation_objective_contains_exactly_prediction_and_sigreg():
    predicted = torch.tensor([[[2.0]]], requires_grad=True)
    actual = torch.tensor([[1.0]], requires_grad=True)
    sigreg = torch.tensor(3.0, requires_grad=True)
    total, prediction = MODULE.representation_objective(
        predicted, actual, sigreg, sigreg_coef=0.25
    )
    torch.testing.assert_close(prediction, torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(1.75))


def test_optimizer_parameter_ownership_is_disjoint_and_complete():
    args = small_args()
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    groups = {
        "world": {id(parameter) for parameter in agent.world.parameters()},
        "value": {id(parameter) for parameter in agent.value.parameters()},
        "reward": {id(parameter) for parameter in agent.reward.parameters()},
        "actor": {
            id(parameter)
            for module in (agent.goal, agent.policy)
            for parameter in module.parameters()
        },
    }
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert groups[name].isdisjoint(groups[other])
    assert set().union(*groups.values()) == {
        id(parameter) for parameter in agent.parameters()
    }


def test_value_and_reward_backwards_leave_world_gradients_none():
    args = small_args()
    agent = MODULE.Agent(obs_dim=3, action_dim=2, args=args)
    observations = torch.randn(6, args.history_length, 3)
    actions = torch.randn(6, args.history_length, 2)
    with torch.no_grad():
        belief = agent.world.belief(observations, actions)
        next_belief = belief + 0.1
    value_loss = agent.value(belief.detach()).square().mean()
    reward_loss = agent.reward(
        belief.detach(), next_belief.detach()
    ).square().mean()
    value_loss.backward()
    reward_loss.backward()
    assert all(parameter.grad is None for parameter in agent.world.parameters())


def test_structured_excitation_is_deterministic_and_phase_diverse():
    first = MODULE.deterministic_action_excitation(17, 4, 3, 0.1, 257)
    second = MODULE.deterministic_action_excitation(17, 4, 3, 0.1, 257)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[0], first[1])
    assert torch.all(first.abs() <= 0.1)


def test_goal_excitation_is_deterministic_and_coordinate_diverse():
    first = MODULE.deterministic_goal_excitation(23, 3, 5, 0.2, 379)
    second = MODULE.deterministic_goal_excitation(23, 3, 5, 0.2, 379)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[:, 0], first[:, 1])
    assert torch.all(first.abs() <= 0.2)


def test_causal_mask_blocks_future_but_history_changes_latest_belief():
    args = small_args()
    world = MODULE.CausalHistoryWorldModel(
        obs_dim=3,
        action_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        predictor_heads=args.predictor_heads,
    ).eval()
    observations = torch.randn(1, args.history_length, 3)
    actions = torch.randn(1, args.history_length, 2)
    altered_future = observations.clone()
    altered_future[:, -1] += 100
    original_sequence = world.belief_sequence_raw(observations, actions)
    altered_sequence = world.belief_sequence_raw(altered_future, actions)
    torch.testing.assert_close(
        original_sequence[:, :-1], altered_sequence[:, :-1]
    )

    altered_past = observations.clone()
    altered_past[:, 0] += 10
    changed_latest = world.belief_sequence_raw(altered_past, actions)[:, -1]
    assert not torch.allclose(original_sequence[:, -1], changed_latest)


def test_timeout_bootstraps_but_stops_advantage_trace():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[4.0], [8.0]])
    bootstrap = torch.ones_like(rewards)
    trace = torch.tensor([[0.0], [1.0]])
    advantages = MODULE.factual_gae(
        rewards, values, next_values, bootstrap, trace, gamma=0.5, gae_lambda=1.0
    )
    torch.testing.assert_close(advantages[0], torch.tensor([3.0]))
    torch.testing.assert_close(advantages[1], torch.tensor([6.0]))


def test_done_history_resets_live_state_but_factual_history_keeps_terminal():
    observations = np.zeros((2, 3, 1), dtype=np.float32)
    actions = np.zeros((2, 3, 1), dtype=np.float32)
    chosen = np.array([[1.0], [2.0]], dtype=np.float32)
    terminal = np.array([[7.0], [8.0]], dtype=np.float32)
    factual_obs, factual_actions = MODULE.transition_next_histories(
        observations, actions, chosen, terminal
    )
    autoreset = np.array([[70.0], [80.0]], dtype=np.float32)
    live_obs, live_actions = MODULE.online_successor_histories(
        factual_obs,
        factual_actions,
        autoreset,
        np.array([True, False]),
    )
    assert factual_obs[0, -1, 0] == 7
    assert np.all(live_obs[0, :, 0] == 70)
    assert np.all(live_actions[0] == 0)
    assert live_obs[1, -1, 0] == 8


def test_source_excludes_forbidden_mechanisms():
    source = SCRIPT.read_text()
    forbidden = (
        "old_logprob",
        "clip_coef",
        "action_value",
        "target_network",
        "mirrored_population",
        "continuous_local_goal",
        "cosine_similarity",
        "Categorical(",
        "Normal(",
        "optimize_action",
        "action_cost_coef",
    )
    assert all(token not in source for token in forbidden)
