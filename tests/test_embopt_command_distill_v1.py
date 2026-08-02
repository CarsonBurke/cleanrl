import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


PATH = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_command_distill_v1.py"
)
SPEC = importlib.util.spec_from_file_location("command_distill", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def make_agent():
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -1, 1, (5,), dtype=float
        ),
        single_action_space=gym.spaces.Box(
            -1, 1, (2,), dtype=float
        ),
    )
    args = SimpleNamespace(
        latent_dim=8,
        goal_dim=4,
        hidden_dim=16,
        value_heads=3,
    )
    return M.Agent(env, args)


def test_completed_suffixes_exclude_censored_tail_and_keep_failures():
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


def test_episode_boundary_completes_command_suffix():
    reward = torch.tensor([[1.0], [-2.0], [5.0]])
    done = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    switched = torch.tensor([[1.0], [0.0], [1.0]])

    rate, length, valid = M.completed_command_suffix_targets(
        reward, done, switched
    )

    torch.testing.assert_close(
        rate.squeeze(), torch.tensor([-0.5, -2.0, 5.0])
    )
    torch.testing.assert_close(
        length.squeeze(), torch.tensor([2.0, 1.0, 1.0])
    )
    torch.testing.assert_close(
        valid.squeeze(), torch.tensor([1.0, 1.0, 0.0])
    )


def test_evidence_switch_has_no_age_or_fixed_score_margin():
    current = torch.tensor(
        [[1.9, 2.0, 2.1], [1.9, 2.0, 2.1], [100.0, 2.0, -100.0]]
    )
    candidate = torch.tensor(
        [[2.9, 3.0, 3.1], [2.0, 2.1, 2.2], [-100.0, 100.0, 100.0]]
    )

    switch, challenger, old_score, new_score = M.evidence_goal_switch(
        current,
        candidate,
        torch.tensor([True, True, False]),
        error_floor=0.2,
    )

    torch.testing.assert_close(switch, torch.tensor([True, False, True]))
    torch.testing.assert_close(challenger, torch.tensor([True, False, False]))
    torch.testing.assert_close(old_score, current.mean(dim=-1))
    torch.testing.assert_close(new_score, candidate.mean(dim=-1))


def test_generating_head_cannot_validate_its_own_goal():
    values = torch.tensor(
        [[100.0, 2.0, 3.0], [4.0, 100.0, 6.0]]
    )
    evaluators = M.leave_one_out(values, torch.tensor([0, 1]))

    torch.testing.assert_close(
        evaluators, torch.tensor([[2.0, 3.0], [4.0, 6.0]])
    )


def test_bootstrap_mask_covers_every_example():
    generator = torch.Generator().manual_seed(3)
    mask = M.bootstrap_mask(
        128, 3, 0.2, torch.device("cpu"), generator
    )
    assert mask.shape == (128, 3)
    assert mask.any(dim=-1).all()
    assert not torch.equal(mask[:, 0], mask[:, 1])


def test_path_weights_are_bounded_masked_normalized_and_monotone():
    rate = torch.tensor([-2.0, 0.0, 4.0, 100.0])
    valid = torch.tensor([1.0, 1.0, 1.0, 0.0])
    weights = M.path_weights(rate, valid, 1.0, 0.25, 1.75)

    assert 0.25 <= weights[0] < weights[1] < weights[2] <= 1.75
    assert weights[3] == 0
    torch.testing.assert_close(weights[:3].mean(), torch.tensor(1.0))


def test_dependent_modules_are_gradient_isolated():
    agent = make_agent()
    z = agent.encode(torch.randn(6, 5))
    y = agent.goal_encode(z)
    goal = agent.propose(y, torch.tensor([0, 1, 2, 0, 1, 2]))
    action = agent.control(z, y, goal)
    loss = agent.command_values(y, goal).mean() + action.mean()
    loss.backward()

    assert all(
        parameter.grad is None for parameter in agent.encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.goal_projector.parameters()
    )
    assert all(
        parameter.grad is None for parameter in agent.proposer_heads.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in agent.command_value_heads.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in agent.controller.parameters()
    )


def test_proposer_objective_cannot_reach_geometry_or_values_when_scoped():
    agent = make_agent()
    z = agent.encode(torch.randn(6, 5))
    y = agent.goal_encode(z)
    head = torch.tensor([0, 1, 2, 0, 1, 2])
    goal = agent.propose(y, head)
    score = M.gather_value_head(
        agent.command_values(y, goal, detach_inputs=False), head
    ).mean()
    proposer_parameters = list(agent.proposer_heads.parameters())
    score.backward(inputs=proposer_parameters)

    assert any(
        parameter.grad is not None for parameter in proposer_parameters
    )
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


def test_procrustes_keeps_global_coordinates_persistent():
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
