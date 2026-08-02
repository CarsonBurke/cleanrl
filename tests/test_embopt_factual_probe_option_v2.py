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
    / "ppo_continuous_action_embopt_factual_probe_option_v2.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_factual_probe_option_v2", SCRIPT)
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


def test_world_target_is_attached_and_invalid_padding_is_excluded():
    model = world()
    sigreg = MODULE.SIGReg(projections=8, knots=5, reference_samples=8)
    observations = torch.randn(4, 4, 5, requires_grad=True)
    actions = torch.randn(4, 3, 2)
    valid = torch.tensor([[False, False, True, True]] * 4)
    loss, _, _ = MODULE.masked_world_objective(
        model, sigreg, observations, actions, valid, sigreg_coef=0.0
    )
    loss.backward()
    assert observations.grad[:, -1].abs().sum() > 0
    assert observations.grad[:, :2].abs().sum() == 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.observation_encoder.parameters()
    )


def test_action_alignment_and_causal_mask():
    model = world().eval()
    with torch.no_grad():
        for block in model.predictor:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
    observations = torch.randn(2, 3, 5)
    actions = torch.randn(2, 3, 2)
    baseline = model.predict_sequence(observations, actions)
    changed = actions.clone()
    changed[:, -1] += 10
    prediction = model.predict_sequence(observations, changed)
    torch.testing.assert_close(baseline[:, :-1], prediction[:, :-1])
    assert not torch.allclose(baseline[:, -1], prediction[:, -1])


def test_padding_mask_blocks_invalid_observations_and_actions_from_valid_predictions():
    model = world().eval()
    with torch.no_grad():
        for block in model.predictor:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    observations = torch.randn(4, 3, 5)
    actions = torch.randn(4, 3, 2)
    valid = torch.tensor([[False, False, True]] * 4)
    baseline = model.predict_sequence(observations, actions, valid)
    corrupted_observations = observations.clone()
    corrupted_actions = actions.clone()
    corrupted_observations[:, :2] += 1_000
    corrupted_actions[:, :2] -= 1_000
    changed = model.predict_sequence(
        corrupted_observations, corrupted_actions, valid
    )
    assert torch.isfinite(changed).all()
    torch.testing.assert_close(baseline[:, -1], changed[:, -1])


def test_favorable_probe_residual_has_correct_policy_gradient_sign():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    means = parameter.expand(4, 1)
    probes = torch.tensor([[0.1], [-0.1], [0.1], [-0.1]])
    # Positive utility along +probe and negative utility along -probe.
    residuals = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, covariance, _ = MODULE.directional_policy_surrogate(
        means, residuals, probes
    )
    loss.backward()
    assert parameter.grad < 0
    torch.testing.assert_close(covariance, torch.tensor([[0.01]]))


def test_unfavorable_positive_probe_reverses_policy_gradient_sign():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    means = parameter.expand(4, 1)
    probes = torch.tensor([[0.1], [-0.1], [0.1], [-0.1]])
    residuals = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    loss, _, _ = MODULE.directional_policy_surrogate(means, residuals, probes)
    loss.backward()
    assert parameter.grad > 0


def test_factual_policy_loss_updates_policy_but_never_world_or_other_modules():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    count = 16
    current = torch.randn(count, 5)
    factual_next = torch.randn(count, 5)
    goals = torch.randn(count, 4)
    with torch.no_grad():
        z = agent.world.encode(current)
        raw_actions = agent.policy.raw_action(z, goals)
        requested = MODULE.deterministic_action_probe(
            3, count, 2, 0.08
        )
        executed, _, bounded_deltas = MODULE.raw_probe_action(
            raw_actions, requested
        )
    policy_loss, _, metrics = MODULE.factual_probe_policy_loss(
        agent,
        current,
        factual_next,
        goals,
        raw_actions,
        executed,
        requested,
        bounded_deltas,
        torch.randn(count),
        control_cost_weight=0.1,
        reward_scale=0.1,
        reward_utility_coef=0.0,
    )
    policy_loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in agent.policy.parameters()
    )
    assert all(
        p.grad is None
        for module in (agent.world, agent.h, agent.c, agent.goal, agent.reward_rate)
        for p in module.parameters()
    )
    assert torch.isfinite(torch.stack(list(metrics.values()))).all()


def test_balanced_probe_gradient_is_invariant_to_constant_utility_offset():
    probes = torch.cat(
        [
            MODULE.deterministic_action_probe(step, 16, 2, 0.08)
            for step in (7, 8)
        ]
    )
    parameter = torch.nn.Parameter(torch.zeros(2))
    raw_actions = parameter.expand(32, 2)
    utilities = torch.randn(32)
    loss, _, _ = MODULE.directional_policy_surrogate(
        raw_actions, utilities, probes
    )
    gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    group_constant = torch.cat([torch.full((16,), 3.0), torch.full((16,), -2.0)])
    shifted_loss, _, _ = MODULE.directional_policy_surrogate(
        raw_actions, utilities - group_constant, probes
    )
    shifted_gradient = torch.autograd.grad(shifted_loss, parameter)[0]
    torch.testing.assert_close(gradient, shifted_gradient, atol=1e-5, rtol=1e-5)


def test_agent_has_no_learned_factual_utility_baseline():
    agent = MODULE.Agent(5, 2, small_args())
    assert not hasattr(agent, "utility_baseline")


def test_default_low_level_utility_cannot_bypass_goals_with_task_reward():
    assert MODULE.Args().utility_reward_coef == 0.0


def test_balanced_action_probes_have_full_rank_covariance():
    probes = MODULE.deterministic_action_probe(
        step=11,
        num_envs=16,
        action_dim=6,
        amplitude=0.08,
    )
    covariance = MODULE.probe_covariance(probes)
    assert torch.linalg.matrix_rank(covariance) == 6
    assert covariance.diagonal().min() > 0
    assert covariance.diagonal().max() / covariance.diagonal().min() < 1.01
    assert (covariance - torch.diag(covariance.diagonal())).abs().max() < 1e-6


def test_raw_logit_probe_records_actual_bounded_delta():
    raw_actions = torch.tensor([[4.0, -4.0], [0.0, 0.0]])
    requested = torch.tensor([[0.1, -0.1], [0.1, -0.1]])
    executed, nominal, actual = MODULE.raw_probe_action(raw_actions, requested)
    assert executed.abs().max() <= 1
    torch.testing.assert_close(executed - nominal, actual)
    assert actual[0, 0] < requested[0, 0]


def test_latent_goal_persists_and_resets_only_at_episode_boundary():
    goals = MODULE.PersistentGoals(3, 4)
    proposed = np.arange(12, dtype=np.float32).reshape(3, 4)
    used = goals.apply(proposed, np.asarray([True, True, False]))
    np.testing.assert_array_equal(used[:2], proposed[:2])
    previous = used.copy()
    used = goals.apply(proposed + 100, np.asarray([False, False, True]))
    np.testing.assert_array_equal(used[:2], previous[:2])
    assert goals.latents.shape == (3, 4)
    assert not hasattr(goals, "observations")
    goals.reset(np.asarray([False, True, True]))
    assert goals.has_goal.tolist() == [True, False, False]


def test_goal_proposal_is_free_latent_displacement_and_only_updates_g():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    with torch.no_grad():
        for head in agent.c.heads:
            torch.nn.init.normal_(head[-1].weight)
    loss, metrics = MODULE.goal_proposal_loss(
        agent,
        torch.randn(8, 5),
        torch.randn(8, 4) * 0.5,
        torch.tensor([True] * 4 + [False] * 4),
        args.pessimism_coef,
    )
    loss.backward()
    assert metrics["latent_displacement_norm"] > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in agent.goal.parameters()
    )
    assert all(
        p.grad is None
        for module in (agent.world, agent.h, agent.c, agent.policy, agent.reward_rate)
        for p in module.parameters()
    )


def test_stop_on_tie_forces_initial_proposal_in_latent_space():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    persistent = MODULE.PersistentGoals(9, 4)
    used, proposed = MODULE.option_decision(
        agent,
        torch.randn(9, 5),
        persistent,
        torch.zeros(9, 4),
        args,
    )
    assert proposed.all()
    assert used.shape == (9, 4)


def test_optimizer_ownership_is_exact_and_disjoint():
    agent = MODULE.Agent(5, 2, small_args())
    groups = MODULE.optimizer_parameter_groups(agent)
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert groups[name].isdisjoint(groups[other])
    assert set().union(*groups.values()) == set(agent.parameters())
    assert set(agent.reward_rate.parameters()) == groups["reward_rate"]


def test_world_optimizer_contains_world_and_only_world():
    agent = MODULE.Agent(5, 2, small_args())
    assert MODULE.world_parameter_ids(agent) == {
        id(parameter) for parameter in agent.world.parameters()
    }


def test_procrustes_alignment_restores_existing_global_chart():
    raw_new = torch.randn(64, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    global_old = raw_new @ q
    alignment = MODULE.procrustes_alignment(raw_new, global_old)
    torch.testing.assert_close(raw_new @ alignment, global_old, atol=1e-5, rtol=1e-5)


def test_cumulative_reward_rate_is_exact_and_not_a_moving_average():
    rate = MODULE.CumulativeRewardRate()
    assert rate.update([1.0, 3.0]) == 2.0
    assert rate.update([8.0]) == 4.0
    assert rate.reward_sum == 12.0
    assert rate.step_count == 3


def test_learned_reward_rate_uses_detached_bellman_residual():
    rate = MODULE.DifferentialRewardRate()
    scaled_rewards = torch.tensor([1.0, 3.0], requires_grad=True)
    bootstrap = torch.tensor([1.0, 0.0])
    selected_next = torch.tensor(
        [[4.0, 8.0], [6.0, 10.0]], requires_grad=True
    )
    current_c = torch.tensor(
        [[2.0, 1.0], [4.0, 3.0]], requires_grad=True
    )
    loss, target = MODULE.differential_rate_loss(
        rate, scaled_rewards, bootstrap, selected_next, current_c
    )
    torch.testing.assert_close(target, torch.tensor([3.0, 1.0]))
    loss.backward()
    assert rate.value.grad < 0
    assert scaled_rewards.grad is None
    assert selected_next.grad is None
    assert current_c.grad is None


def test_option_targets_use_detached_learned_rate_not_cumulative_telemetry():
    rate = MODULE.DifferentialRewardRate()
    with torch.no_grad():
        rate.value.fill_(1.25)
    rewards = torch.tensor([2.0, 4.0], requires_grad=True)
    adjusted = MODULE.centered_option_rewards(rewards, rate, average_reward=True)
    torch.testing.assert_close(adjusted, torch.tensor([0.75, 2.75]))
    adjusted.sum().backward()
    assert rate.value.grad is None
    torch.testing.assert_close(
        MODULE.centered_option_rewards(rewards.detach(), rate, average_reward=False),
        rewards.detach(),
    )


def test_average_and_discounted_boundaries():
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


def test_factual_terminal_observation_replaces_autoreset_observation():
    live = np.asarray([[1.0], [2.0]], dtype=np.float32)
    infos = {
        "final_observation": np.asarray(
            [np.asarray([9.0], dtype=np.float32), None], dtype=object
        ),
        "_final_observation": np.asarray([True, False]),
    }
    factual = MODULE.factual_next_observations(live, infos)
    np.testing.assert_array_equal(factual[0], [9.0])
    np.testing.assert_array_equal(factual[1], [2.0])


def test_h_optimizer_skips_decay_without_proposals():
    args = small_args()
    h = MODULE.StateBiasEnsemble(4, 16, 3)
    optimizer = torch.optim.AdamW(h.parameters(), lr=1e-2, weight_decay=0.1)
    before = [parameter.detach().clone() for parameter in h.parameters()]
    MODULE.update_proposal_h(
        h,
        torch.randn(8, 4),
        torch.randn(3, 8),
        torch.zeros(8, dtype=torch.bool),
        optimizer,
        args.bootstrap_probability,
        args.max_grad_norm,
    )
    for old, parameter in zip(before, h.parameters()):
        torch.testing.assert_close(old, parameter)


def test_replay_has_only_factual_model_data():
    parameters = list(inspect.signature(MODULE.SequenceReplay.add).parameters)
    assert parameters == ["self", "observations", "actions", "valid"]
    replay = MODULE.SequenceReplay(16, 3, 5, 2)
    assert not hasattr(replay, "goals")
    assert not hasattr(replay, "rewards")


def test_no_forbidden_policy_or_planning_method():
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
        "straight_through_actual",
        "policy_progress_loss",
    }
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert forbidden_names.isdisjoint(identifiers)
    assert "cosine_similarity" not in source
    assert "normalize(" not in source
