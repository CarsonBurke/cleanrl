import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_freegoal_svg_v1.py"
)
SPEC = importlib.util.spec_from_file_location("freegoal_svg_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ridge_solver_recovers_full_rank_target():
    control = torch.eye(3).unsqueeze(0)
    target = torch.tensor([[2.0, -1.0, 0.5]])
    result = MODULE.ridge_solve(control, target, ridge=0.01)
    torch.testing.assert_close(result, target / 1.01)


def test_euclidean_mse_preserves_magnitude_information():
    origin = torch.zeros(1, 3)
    small = torch.tensor([[1.0, 0.0, 0.0]])
    large = torch.tensor([[3.0, 0.0, 0.0]])
    small_error = (small - origin).square().mean()
    large_error = (large - origin).square().mean()
    assert large_error == 9 * small_error


def test_state_value_has_state_only_signature():
    parameters = list(inspect.signature(MODULE.StateValueEnsemble.forward).parameters)
    assert parameters == ["self", "y"]


def test_procrustes_transports_rotated_chart_without_changing_magnitude():
    generator = torch.Generator().manual_seed(11)
    raw_old = torch.randn(64, 4, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    raw_new = raw_old @ rotation
    alignment = MODULE.procrustes_alignment(raw_new, raw_old)
    transported = raw_new @ alignment
    torch.testing.assert_close(transported, raw_old, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        transported.norm(dim=-1), raw_new.norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_goal_actor_output_is_not_projected_or_clipped():
    actor = MODULE.GoalActor(3, 16, -4.0, 1.0)
    with torch.no_grad():
        actor.trunk[-1].weight.zero_()
        actor.trunk[-1].bias[:3].fill_(7.0)
    mean, _ = actor(torch.zeros(2, 3))
    torch.testing.assert_close(mean, torch.full((2, 3), 7.0))
    assert torch.all(mean.norm(dim=-1) > 10)


def test_actor_receives_gradient_through_controller_and_models():
    args = MODULE.Args(
        latent_dim=3,
        hidden_dim=16,
        ensemble_size=2,
        goal_delivery_coef=0.1,
    )
    agent = MODULE.Agent(obs_dim=4, action_dim=2, args=args)
    y = agent.encode(torch.randn(8, 4)).detach()
    factual_goals = torch.randn(8, 3)
    advantages = torch.linspace(-1, 1, 8)
    loss, _ = MODULE.fresh_actor_loss(
        agent, y, factual_goals, advantages, args
    )
    loss.backward()
    gradient = sum(
        parameter.grad.abs().sum()
        for parameter in agent.goal_actor.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0
    assert all(
        parameter.grad is None
        for module in (
            agent.controller,
            agent.forward_ensemble,
            agent.value_ensemble,
        )
        for parameter in module.parameters()
    )


def test_score_gradient_changes_sign_with_factual_advantage():
    actor = MODULE.GoalActor(2, 8, -4.0, 1.0)
    y = torch.randn(5, 2)
    goals = torch.randn(5, 2)

    positive_loss, _ = MODULE.goal_score_loss(
        actor, y, goals, torch.ones(5)
    )
    positive_gradient = torch.autograd.grad(
        positive_loss, tuple(actor.parameters())
    )
    negative_loss, _ = MODULE.goal_score_loss(
        actor, y, goals, -torch.ones(5)
    )
    negative_gradient = torch.autograd.grad(
        negative_loss, tuple(actor.parameters())
    )
    for positive, negative in zip(positive_gradient, negative_gradient):
        torch.testing.assert_close(positive, -negative)


def test_factual_gae_stops_trace_at_episode_boundary_but_bootstraps_timeout():
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


def test_source_has_no_replay_goal_or_spherical_goal_mechanism():
    source = SCRIPT.read_text()
    forbidden = (
        "mirrored_population",
        "normalized_es_gradient",
        "continuous_local_goal",
        "goal_alignment",
        "target_network",
        "old_logprob",
        "clip_coef",
    )
    assert all(token not in source for token in forbidden)
