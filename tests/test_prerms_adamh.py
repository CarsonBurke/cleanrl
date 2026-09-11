"""Keep the v6 control exact while restricting Hyperball to hidden branch matrices."""

from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_clip_toggle_v6 as baseline
from cleanrl import ppo_continuous_action_prerms_adamh_v22 as trainer
from cleanrl.shared.norm_residual import NormResidualTrunk
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    return device

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture
def spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    )


def make_agent(module, spaces):
    with torch.device("cuda"):
        return module.Agent(spaces)


def ppo_batch(agent):
    observations = torch.randn(64, 17, device="cuda")
    native_actions = 0.1 + 0.8 * torch.rand(64, 6, device="cuda")
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
        old_values = values.flatten()
    advantages = torch.randn(64, device="cuda")
    targets = old_values + 1.0 + 0.1 * torch.randn_like(old_values)
    return observations, native_actions, old_logprobs, advantages, targets, old_values


def test_initial_model_and_policy_exactly_match_best_baseline(spaces):
    torch.manual_seed(19)
    reference = make_agent(baseline, spaces)
    torch.manual_seed(19)
    candidate = make_agent(trainer, spaces)
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    assert candidate_state.keys() == reference_state.keys()
    for name in reference_state:
        torch.testing.assert_close(candidate_state[name], reference_state[name], rtol=0, atol=0)
    observations = torch.randn(64, 17, device="cuda")
    with torch.no_grad():
        for actual, expected in zip(
            candidate.get_policy_and_value(observations), reference.get_policy_and_value(observations)
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adam_control_preserves_original_ppo_updates(spaces):
    torch.manual_seed(23)
    reference = make_agent(baseline, spaces)
    torch.manual_seed(23)
    candidate = make_agent(trainer, spaces)
    args = trainer.Args(optimizer_kind="adam")
    reference_args = baseline.Args(grad_clip=False)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    optimizer, hyperball_optimizer = trainer.make_optimizers(candidate, args)
    assert hyperball_optimizer is None
    batch = ppo_batch(reference)
    for fraction in (1.0, 0.5, 0.1):
        reference_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
        optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
        reference_loss, reference_metrics = baseline.ppo_loss(reference, *batch, reference_args)
        candidate_loss, candidate_metrics = trainer.ppo_loss(candidate, *batch, args)
        torch.testing.assert_close(candidate_loss, reference_loss, rtol=0, atol=0)
        torch.testing.assert_close(candidate_metrics, reference_metrics, rtol=0, atol=0)
        reference_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        reference_loss.backward()
        candidate_loss.backward()
        reference_optimizer.step()
        optimizer.step()
        for actual, expected in zip(candidate.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_only_hidden_branch_weights_keep_initial_frobenius_radius(spaces):
    torch.manual_seed(29)
    agent = make_agent(trainer, spaces)
    optimizer, hyperball_optimizer = trainer.make_optimizers(agent, trainer.Args())
    assert hyperball_optimizer is not None
    hidden = trainer.hidden_branch_parameters(agent)
    initial_hidden = [parameter.detach().clone() for parameter in hidden]
    radii = [torch.linalg.vector_norm(parameter) for parameter in initial_hidden]
    unconstrained = tuple(
        parameter
        for network in (agent.actor, agent.critic)
        for parameter in (cast(NormResidualTrunk, network[0]).in_proj.weight,
                          cast(torch.nn.Linear, network[-1]).weight,
                          *cast(NormResidualTrunk, network[0]).block_gates)
    )
    unconstrained_radii = [torch.linalg.vector_norm(parameter.detach()) for parameter in unconstrained]
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        hyperball_optimizer.zero_grad(set_to_none=True)
        for parameter in agent.parameters():
            parameter.grad = parameter.detach().clone()
        optimizer.step()
        hyperball_optimizer.step()
    for parameter, initial, radius in zip(hidden, initial_hidden, radii):
        torch.testing.assert_close(torch.linalg.vector_norm(parameter), radius, rtol=3e-7, atol=0)
        assert not torch.allclose(parameter, initial)
    for parameter, radius in zip(unconstrained, unconstrained_radii):
        assert not torch.isclose(torch.linalg.vector_norm(parameter), radius, rtol=1e-5, atol=1e-7)
    assert trainer.geometry_metrics(hyperball_optimizer)["geometry/max_radius_relative_error"] < 3e-7


def test_compiled_real_ppo_loss_updates_both_optimizers_without_radius_drift(spaces):
    torch.manual_seed(31)
    agent = make_agent(trainer, spaces)
    args = trainer.Args()
    optimizer, hyperball_optimizer = trainer.make_optimizers(agent, args)
    assert hyperball_optimizer is not None
    batch = ppo_batch(agent)
    with torch.no_grad():
        before = tuple(value.clone() for value in agent.get_policy_and_value(batch[0]))

    def loss_model(observations, native_actions, old_logprobs, advantages, targets, old_values):
        return trainer.ppo_loss(agent, observations, native_actions, old_logprobs, advantages, targets, old_values, args)

    compiled_loss = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
    for fraction in (1.0, 0.5, 0.1):
        optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
        hyperball_optimizer.set_lr(fraction * args.hyperball_learning_rate)
        torch.compiler.cudagraph_mark_step_begin()
        loss, metrics = compiled_loss(*batch)
        optimizer.zero_grad(set_to_none=True)
        hyperball_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.clip_gradients(*trainer.clipping_parameters(agent, args.clip_heads), args.max_grad_norm,
                               enabled=args.grad_clip)
        optimizer.step()
        hyperball_optimizer.step()
        assert torch.isfinite(loss)
        assert torch.isfinite(metrics).all()
    with torch.no_grad():
        after = agent.get_policy_and_value(batch[0])
        for actual, initial in zip(after, before):
            assert torch.isfinite(actual).all()
            assert not torch.allclose(actual, initial)
    diagnostics = trainer.geometry_metrics(hyperball_optimizer)
    assert diagnostics["geometry/max_radius_relative_error"] < 3e-7
    torch.testing.assert_close(
        diagnostics["charts/hyperball_learning_rate"],
        torch.tensor(0.1 * args.hyperball_learning_rate, device="cuda"),
    )


def test_adamh_rejects_unsupported_activation():
    with pytest.raises(ValueError, match="stiglu"):
        trainer.validate_args(trainer.Args(activation="lrelusq"))
