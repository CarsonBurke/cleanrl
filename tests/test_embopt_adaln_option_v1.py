import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_adaln_option_v1.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_adaln_option_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def small_args():
    return MODULE.Args(
        history_length=3,
        latent_dim=4,
        hidden_dim=16,
        transformer_layers=2,
        transformer_heads=2,
        option_heads=3,
        minibatch_size=8,
    )


def world():
    args = small_args()
    return MODULE.CausalActionWorldModel(
        obs_dim=5,
        action_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    )


def activate_adaln(model):
    with torch.no_grad():
        for block in model.predictor:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)


def test_action_alignment_and_causality():
    model = world().eval()
    activate_adaln(model)
    observations = torch.randn(2, 3, 5)
    actions = torch.randn(2, 3, 2)
    baseline = model.predict_sequence(observations, actions)

    changed_last_action = actions.clone()
    changed_last_action[:, 2] += 20
    changed = model.predict_sequence(observations, changed_last_action)
    torch.testing.assert_close(baseline[:, :2], changed[:, :2])
    assert not torch.allclose(baseline[:, 2], changed[:, 2])

    changed_future_observation = observations.clone()
    changed_future_observation[:, 2] += 20
    changed = model.predict_sequence(changed_future_observation, actions)
    torch.testing.assert_close(baseline[:, :2], changed[:, :2])


def test_sequence_shapes_pair_each_source_with_outgoing_action_and_next_target():
    model = world()
    observations = torch.randn(7, 4, 5)
    actions = torch.randn(7, 3, 2)
    encoded = model.encode(observations)
    predicted = model.predict_from_latents(encoded[:, :-1], actions)
    assert predicted.shape == encoded[:, 1:].shape == (7, 3, 4)


def test_adaln_is_zero_initialized_then_learns_action_gradient():
    model = world()
    assert all(
        torch.count_nonzero(block.adaLN_modulation[-1].weight) == 0
        and torch.count_nonzero(block.adaLN_modulation[-1].bias) == 0
        for block in model.predictor
    )
    observations = torch.randn(16, 3, 5)
    actions = torch.randn(16, 3, 2)
    target = torch.randn(16, 3, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad()
        (model.predict_sequence(observations, actions) - target).square().mean().backward()
        optimizer.step()
    probe = actions.detach().clone().requires_grad_(True)
    model.predict_final(observations, probe).square().mean().backward()
    assert probe.grad[:, -1].abs().sum() > 0
    assert MODULE.adaln_gate_magnitude(model) > 0


def test_world_target_is_attached_and_invalid_padding_is_excluded():
    model = world()
    sigreg = MODULE.SIGReg(projections=8, knots=5, reference_samples=8)
    observations = torch.randn(4, 4, 5, requires_grad=True)
    actions = torch.randn(4, 3, 2)
    valid = torch.tensor(
        [[False, False, True, True]] * 4,
        dtype=torch.bool,
    )
    loss, _, _ = MODULE.masked_world_objective(
        model, sigreg, observations, actions, valid, sigreg_coef=0.0
    )
    loss.backward()
    assert observations.grad[:, -1].abs().sum() > 0
    assert observations.grad[:, :2].abs().sum() == 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.observation_encoder.parameters()
    )


def test_world_optimizer_ownership_is_exact_and_disjoint():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    groups = MODULE.optimizer_parameter_groups(agent)
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert groups[name].isdisjoint(groups[other])
    assert set().union(*groups.values()) == set(agent.parameters())
    expected_world = {
        id(parameter)
        for module in (
            agent.world.observation_encoder,
            agent.world.encoder_projector,
            agent.world.action_encoder,
            agent.world.predictor,
            agent.world.predictor_projector,
        )
        for parameter in module.parameters()
    } | {id(agent.world.position)}
    assert expected_world <= MODULE.world_parameter_ids(agent)


def test_procrustes_alignment_restores_old_global_chart():
    raw_new = torch.randn(64, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    global_old = raw_new @ q
    alignment = MODULE.procrustes_alignment(raw_new, global_old)
    torch.testing.assert_close(raw_new @ alignment, global_old, atol=1e-5, rtol=1e-5)


def test_raw_goal_persists_and_resets_only_on_episode_boundary():
    goals = MODULE.PersistentGoals(3, 2)
    proposed = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    used = goals.apply(proposed, np.asarray([True, True, False]))
    np.testing.assert_array_equal(used[:2], proposed[:2])
    previous = used.copy()
    used = goals.apply(proposed + 100, np.asarray([False, False, True]))
    np.testing.assert_array_equal(used[:2], previous[:2])
    goals.reset(np.asarray([False, True, True]))
    assert goals.has_goal.tolist() == [True, False, False]


def test_zero_initialized_heads_stop_on_tie_and_force_initial_proposal():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    z = torch.randn(9, 4)
    goal = torch.randn(9, 4)
    assert not MODULE.continue_decision(
        agent.h(z), agent.c(z, goal), args.pessimism_coef
    ).any()
    persistent = MODULE.PersistentGoals(9, 5)
    used, proposed = MODULE.option_decision(
        agent,
        torch.randn(9, 5),
        persistent,
        torch.zeros(9, 5),
        args,
    )
    assert proposed.all()
    assert used.shape == (9, 5)


def test_pessimism_is_on_correlated_per_head_surplus():
    h = torch.tensor([[0.0], [10.0], [20.0]])
    c = h + torch.tensor([[1.0], [2.0], [3.0]])
    expected = torch.tensor([2.0 - np.std([1.0, 2.0, 3.0])])
    torch.testing.assert_close(
        MODULE.pessimistic_surplus(h, c, 1.0), expected.float(), atol=1e-6, rtol=1e-6
    )
    shifted = torch.tensor([[100.0], [-50.0], [7.0]])
    torch.testing.assert_close(
        MODULE.pessimistic_surplus(h + shifted, c + shifted, 1.0),
        MODULE.pessimistic_surplus(h, c, 1.0),
    )


def test_next_branch_is_one_shared_decision_but_preserves_head_values():
    h = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    c = torch.tensor([[4.0, 9.0], [5.0, 19.0], [6.0, 29.0]])
    selected, continuing = MODULE.select_next_branch(h, c, coefficient=0.0)
    assert continuing.tolist() == [True, False]
    torch.testing.assert_close(selected[:, 0], c[:, 0])
    torch.testing.assert_close(selected[:, 1], h[:, 1])


def test_continuation_target_algebra_uses_same_goal_branch():
    rewards = torch.tensor([2.0, 3.0])
    next_h = torch.tensor([[1.0, 4.0], [2.0, 5.0]])
    next_c = torch.tensor([[5.0, 1.0], [6.0, 2.0]])
    targets, continuing = MODULE.continuation_targets(
        rewards,
        next_h,
        next_c,
        discount=1.0,
        bootstrap=torch.ones(2),
        coefficient=0.0,
    )
    assert continuing.tolist() == [True, False]
    torch.testing.assert_close(targets[:, 0], rewards[0] + next_c[:, 0])
    torch.testing.assert_close(targets[:, 1], rewards[1] + next_h[:, 1])


def test_manager_lambda_return_exact_recurrence_and_trace_stop():
    rewards = torch.tensor([[1.0], [2.0]])
    next_values = torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]]])
    bootstrap = torch.ones(2, 1)
    trace = torch.tensor([[1.0], [0.0]])
    returns = MODULE.manager_lambda_returns(
        rewards, next_values, bootstrap, trace, 1.0, 0.5
    )
    torch.testing.assert_close(returns[1, 0], torch.tensor([32.0, 42.0]))
    torch.testing.assert_close(returns[0, 0], torch.tensor([22.0, 32.0]))


def test_average_reward_boundary_is_regenerative_and_discounted_boundary_is_episodic():
    shape = (2, 3)
    next_h = torch.ones(shape)
    next_c = torch.full(shape, 2.0)
    live_h = torch.full(shape, 7.0)
    factual_h = torch.full(shape, 11.0)
    terminated = torch.tensor([True, False, False])
    truncated = torch.tensor([False, True, False])
    selected, bootstrap, _ = MODULE.boundary_next_option_values(
        next_h, next_c, live_h, factual_h, terminated, truncated, True, 0.0
    )
    torch.testing.assert_close(selected[:, :2], live_h[:, :2])
    assert bootstrap.tolist() == [1.0, 1.0, 1.0]
    selected, bootstrap, _ = MODULE.boundary_next_option_values(
        next_h, next_c, live_h, factual_h, terminated, truncated, False, 0.0
    )
    torch.testing.assert_close(selected[:, 1], factual_h[:, 1])
    assert bootstrap.tolist() == [0.0, 1.0, 1.0]


def test_cumulative_rate_is_exact_non_ema_telemetry():
    rate = MODULE.CumulativeRewardRate()
    assert rate.update([1.0, 3.0]) == 2.0
    assert rate.update([8.0]) == 4.0
    assert rate.reward_sum == 12.0
    assert rate.step_count == 3


def test_system_identification_excitation_has_a_hard_finite_boundary():
    args = MODULE.Args()
    assert not MODULE.excitation_active(
        args.warmup_steps - 1, args.warmup_steps, args.excitation_steps
    )
    assert MODULE.excitation_active(
        args.warmup_steps, args.warmup_steps, args.excitation_steps
    )
    assert MODULE.excitation_active(
        args.warmup_steps + args.excitation_steps - 1,
        args.warmup_steps,
        args.excitation_steps,
    )
    assert not MODULE.excitation_active(
        args.warmup_steps + args.excitation_steps,
        args.warmup_steps,
        args.excitation_steps,
    )


def test_runtime_validation_rejects_invalid_history_and_excitation_window():
    args = MODULE.Args(history_length=0)
    try:
        MODULE.validate_runtime_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("zero history must be rejected")
    args = MODULE.Args(excitation_steps=-1)
    try:
        MODULE.validate_runtime_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("negative excitation window must be rejected")


def test_h_optimizer_is_entirely_skipped_without_proposals():
    args = small_args()
    h = MODULE.StateBiasEnsemble(4, 16, 3)
    optimizer = torch.optim.AdamW(h.parameters(), lr=1e-2, weight_decay=0.1)
    before = [parameter.detach().clone() for parameter in h.parameters()]
    loss, gradient = MODULE.update_proposal_h(
        h,
        torch.randn(8, 4),
        torch.randn(3, 8),
        torch.zeros(8, dtype=torch.bool),
        optimizer,
        args.bootstrap_probability,
        args.max_grad_norm,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    for old, parameter in zip(before, h.parameters()):
        torch.testing.assert_close(old, parameter)


def test_goal_update_owns_only_g_and_flows_through_frozen_encoder_and_critics():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    with torch.no_grad():
        for head in agent.c.heads:
            torch.nn.init.normal_(head[-1].weight)
    loss, _ = MODULE.goal_proposal_loss(
        agent,
        torch.randn(8, 5),
        torch.randn(8, 5) * 0.01,
        torch.tensor([True] * 4 + [False] * 4),
        args.pessimism_coef,
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.goal.parameters())
    assert all(
        p.grad is None
        for module in (agent.world, agent.h, agent.c, agent.policy)
        for p in module.parameters()
    )


def test_policy_progress_st_is_factual_forward_model_backward_and_policy_only():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    activate_adaln(agent.world)
    context = torch.randn(7, 3, 5)
    past = torch.randn(7, 2, 2)
    factual = torch.randn(7, 5)
    goals = torch.randn(7, 5)
    loss, actions, metrics = MODULE.policy_progress_loss(
        agent,
        context,
        past,
        factual,
        goals,
        torch.zeros(7, 2),
        control_cost_weight=0.1,
        reward_scale=0.1,
    )
    loss.backward()
    assert actions.shape == (7, 2)
    assert torch.isfinite(torch.stack(list(metrics.values()))).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.parameters())
    assert all(
        p.grad is None
        for module in (agent.world, agent.h, agent.c, agent.goal)
        for p in module.parameters()
    )
    # Prediction telemetry reads the raw model output, whereas the optimized
    # forward progress is factual through the straight-through construction.
    assert not torch.allclose(
        metrics["predicted_goal_mse"], metrics["actual_goal_mse"]
    )


def test_exact_action_and_goal_recompute_are_zero_before_updates():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    context = torch.randn(6, 3, 5)
    past = torch.randn(6, 2, 2)
    factual = torch.randn(6, 5)
    current = context[:, -1]
    goal_excitation = torch.randn(6, 5) * 0.01
    with torch.no_grad():
        z = agent.world.encode(current)
        goals = current + agent.goal(z) + goal_excitation
        goal_z = agent.world.encode(goals)
        excitation = torch.randn(6, 2) * 0.01
        stored_actions = agent.policy(z, goal_z, excitation)
    _, recomputed, _ = MODULE.policy_progress_loss(
        agent, context, past, factual, goals, excitation, 0.1, 0.1
    )
    torch.testing.assert_close(recomputed, stored_actions)
    goal_mse = MODULE.goal_recompute_mse(
        agent,
        current,
        goal_excitation,
        goals,
        torch.ones(6, dtype=torch.bool),
    )
    torch.testing.assert_close(goal_mse, torch.zeros_like(goal_mse))


def test_replay_contains_only_factual_sequences_not_goals_or_rewards():
    parameters = list(inspect.signature(MODULE.SequenceReplay.add).parameters)
    assert parameters == ["self", "observations", "actions", "valid"]
    replay = MODULE.SequenceReplay(16, 3, 5, 2)
    assert not hasattr(replay, "goals")


def test_no_forbidden_ppo_q_ema_or_planning_implementation():
    source = SCRIPT.read_text().lower()
    tree = ast.parse(source)
    forbidden_names = {
        "logprob",
        "ratio",
        "clip_coef",
        "target_network",
        "soft_update",
        "planner",
        "planning",
        "ema",
        "q_network",
    }
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert forbidden_names.isdisjoint(identifiers)
    assert "cosine_similarity" not in source
    assert "normalize(" not in source
