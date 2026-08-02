import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch


PATH = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_state_mpc_v1.py"
)
SPEC = importlib.util.spec_from_file_location("embopt_state_mpc_v1", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def make_agent():
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -1.0, 1.0, (5,), dtype=float
        ),
        single_action_space=gym.spaces.Box(
            -1.0, 1.0, (2,), dtype=float
        ),
    )
    args = SimpleNamespace(
        latent_dim=8,
        goal_dim=4,
        hidden_dim=16,
        value_heads=3,
    )
    return M.Agent(env, args)


def test_route_rate_uses_endogenous_remaining_steps():
    reward = torch.tensor([10.0, -2.0])
    command_rate = torch.tensor([2.0, 4.0])
    log_steps = torch.tensor([3.0, 1.0]).log()

    result = M.route_rate(reward, command_rate, log_steps)

    torch.testing.assert_close(
        result,
        torch.tensor([(10.0 + 3.0 * 2.0) / 4.0, (-2.0 + 4.0) / 2.0]),
    )


def test_route_rate_remains_finite_for_extreme_predicted_durations():
    result = M.route_rate(
        torch.tensor([3.0, 3.0]),
        torch.tensor([7.0, 7.0]),
        torch.tensor([-1e6, 1e6]),
    )

    torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))
    assert result.isfinite().all()


def test_factual_transition_observations_restore_autoreset_terminals():
    reset_observations = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    infos = {
        "final_observation": np.asarray(
            [np.asarray([9.0, 8.0]), None], dtype=object
        ),
        "_final_observation": np.asarray([True, False]),
    }

    factual = M.factual_transition_observations(reset_observations, infos)

    np.testing.assert_array_equal(factual, [[9.0, 8.0], [3.0, 4.0]])
    np.testing.assert_array_equal(reset_observations, [[1.0, 2.0], [3.0, 4.0]])


def test_cem_candidates_are_bounded_and_mix_local_with_global():
    center = torch.tensor([[0.25, -0.5], [-0.25, 0.5]])
    std = torch.full_like(center, 0.2)
    generator = torch.Generator().manual_seed(7)

    candidates = M.sample_cem_candidates(
        center,
        std,
        population=20,
        global_fraction=0.25,
        generator=generator,
    )

    assert candidates.shape == (2, 20, 2)
    assert (candidates >= -1.0).all() and (candidates <= 1.0).all()
    torch.testing.assert_close(candidates[:, 0], center)
    # The final five are the explicitly global population.
    assert (candidates[:, -5:].sub(center[:, None]).abs() > 0.2).any()


def test_cem_elite_update_selects_top_actions_and_best_sample():
    candidates = torch.tensor(
        [[[0.0], [0.25], [0.75], [-0.5]], [[-0.8], [0.1], [0.4], [0.9]]]
    )
    scores = torch.tensor([[0.0, 2.0, 5.0, -1.0], [4.0, 0.0, 1.0, 3.0]])
    center = torch.zeros(2, 1)
    std = torch.ones(2, 1)

    next_center, next_std, best, elites = M.cem_elite_update(
        candidates,
        scores,
        elite_count=2,
        center=center,
        std=std,
        update_rate=1.0,
        minimum_std=0.05,
    )

    torch.testing.assert_close(best, torch.tensor([[0.75], [-0.8]]))
    torch.testing.assert_close(next_center, torch.tensor([[0.5], [0.05]]))
    torch.testing.assert_close(next_std, elites.std(dim=1, unbiased=False))


def test_transition_reward_has_state_only_signature():
    parameters = list(inspect.signature(M.Agent.transition_reward).parameters)

    assert parameters == ["self", "y", "y_next", "detach_inputs"]
    assert "action" not in parameters


def test_command_heads_return_rate_and_log_remaining_steps():
    agent = make_agent()
    y = torch.randn(6, 4)
    goal = torch.randn(6, 4)

    rate, log_steps = agent.command_values(y, goal)

    assert rate.shape == (6, 3)
    assert log_steps.shape == (6, 3)
    assert rate.isfinite().all() and log_steps.isfinite().all()


def test_state_objectives_are_gradient_isolated_and_no_action_value_exists():
    agent = make_agent()
    y = torch.randn(6, 4, requires_grad=True)
    y_next = torch.randn(6, 4, requires_grad=True)
    goal = torch.randn(6, 4, requires_grad=True)

    rate, log_steps = agent.command_values(y, goal)
    reward = agent.transition_reward(y, y_next)
    (rate.mean() + log_steps.mean() + reward.mean()).backward()

    assert y.grad is None and y_next.grad is None and goal.grad is None
    assert any(
        parameter.grad is not None
        for parameter in agent.command_value_heads.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in agent.transition_reward_model.parameters()
    )
    assert not hasattr(agent, "controller")
    assert not hasattr(agent, "policy")
    assert not hasattr(agent, "q")
    assert not hasattr(agent, "q_value")


def test_candidate_scoring_cannot_backpropagate_into_actions_or_models():
    agent = make_agent()
    obs = torch.randn(3, 5)
    z = agent.encode(obs)
    y = agent.goal_encode(z)
    goal = torch.randn(3, 4)
    actions = torch.randn(3, 8, 2, requires_grad=True)

    scores, reward = M.score_action_candidates(
        agent,
        z,
        y,
        goal,
        actions,
        torch.tensor([0, 1, 2]),
    )

    assert scores.shape == (3, 8)
    assert reward.shape == (3, 8)
    assert not scores.requires_grad and not reward.requires_grad
    assert actions.grad is None
    assert all(parameter.grad is None for parameter in agent.parameters())


def test_proposer_objective_is_scoped_away_from_chart_and_value_heads():
    agent = make_agent()
    z = agent.encode(torch.randn(6, 5))
    y = agent.goal_encode(z)
    head = torch.tensor([0, 1, 2, 0, 1, 2])
    goal = agent.propose(y, head)
    rates, _ = agent.command_values(y, goal, detach_inputs=False)
    score = M.gather_value_head(rates, head).mean()
    proposer_parameters = list(agent.proposer_heads.parameters())

    score.backward(inputs=proposer_parameters)

    assert any(parameter.grad is not None for parameter in proposer_parameters)
    assert all(
        parameter.grad is None for parameter in agent.encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.goal_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.command_value_heads.parameters()
    )


def test_completed_suffixes_keep_failures_and_censor_open_command():
    reward = torch.tensor([[1.0], [2.0], [-3.0], [-4.0]])
    done = torch.zeros(5, 1)
    switched = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    rate, length, valid = M.completed_command_suffix_targets(
        reward, done, switched
    )

    torch.testing.assert_close(
        rate.squeeze(), torch.tensor([1.5, 2.0, -3.5, -4.0])
    )
    torch.testing.assert_close(
        length.squeeze(), torch.tensor([2.0, 1.0, 2.0, 1.0])
    )
    torch.testing.assert_close(
        valid.squeeze(), torch.tensor([1.0, 1.0, 0.0, 0.0])
    )


def test_switch_uses_evaluator_uncertainty_and_initializes_any_goal():
    current = torch.tensor(
        [[1.9, 2.0, 2.1], [1.9, 2.0, 2.1], [100.0, 2.0, -100.0]]
    )
    candidate = torch.tensor(
        [[2.9, 3.0, 3.1], [2.0, 2.1, 2.2], [-100.0, 100.0, 100.0]]
    )

    switch, challenger, _, _ = M.evidence_goal_switch(
        current,
        candidate,
        torch.tensor([True, True, False]),
        error_floor=0.2,
    )

    torch.testing.assert_close(switch, torch.tensor([True, False, True]))
    torch.testing.assert_close(challenger, torch.tensor([True, False, False]))


def test_generating_head_is_excluded_from_goal_evidence():
    values = torch.tensor([[100.0, 2.0, 3.0], [4.0, 100.0, 6.0]])

    evaluators = M.leave_one_out(values, torch.tensor([0, 1]))

    torch.testing.assert_close(
        evaluators, torch.tensor([[2.0, 3.0], [4.0, 6.0]])
    )


def test_procrustes_frame_alignment_recovers_persistent_chart():
    torch.manual_seed(2)
    old = torch.randn(64, 4)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    raw_new = old @ rotation

    alignment = M.procrustes_alignment(raw_new, old)

    torch.testing.assert_close(
        raw_new @ alignment, old, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        alignment.T @ alignment,
        torch.eye(4),
        atol=1e-5,
        rtol=1e-5,
    )
