"""CUDA contracts for independent actor/critic clipping; run only through mlq."""

import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_dreamer_twohot_v3 as frozen_twohot
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_indclip_v3 as base
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4 as categorical
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_v2 as frozen_base
from test_ppo_normres_twohot import (
    SELECTED_TRUNK,
    _agent,
    _assert_parameters,
    _observations,
    _policy_inputs,
    device,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


@pytest.fixture(params=[base, categorical], ids=["base", "categorical"])
def trainer(request):
    return request.param


def _parameters(device):
    with torch.device(device):
        return tuple(
            torch.nn.Parameter(torch.linspace(-0.2, 0.3, count).reshape(shape))
            for count, shape in [(6, (2, 3)), (3, (3,))]
        )


def _set_gradients(parameters, step, norm):
    # Non-collinear, sign-changing steps expose Adam moment coupling that a
    # constant gradient rescaling can conceal, especially on its first step.
    flat = torch.arange(sum(p.numel() for p in parameters), device=parameters[0].device).float()
    flat = (flat * 0.7 + step * 1.3).sin()
    flat = flat * (norm / torch.linalg.vector_norm(flat))
    for parameter, gradient in zip(parameters, flat.split([p.numel() for p in parameters])):
        parameter.grad = gradient.reshape_as(parameter).clone()


def _manual_clip(parameters, threshold):
    # Independent vector reference, not another call to clip_grad_norm_.
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in parameters]))
    coefficient = (threshold / (norm + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(coefficient)
    return norm


def test_norms_and_gradients_match_separate_vector_clips(device, trainer):
    actor, critic = _parameters(device), _parameters(device)
    expected_actor, expected_critic = _parameters(device), _parameters(device)
    # Opposite sides of the boundary, a non-default threshold, and the exact
    # boundary catch global clipping, swapped groups, and a hard-coded limit.
    for step, (threshold, actor_norm, critic_norm) in enumerate([(0.5, 0.2, 2.0), (0.125, 3.0, 0.05), (0.5, 0.5, 0.5)]):
        for parameters in (actor, expected_actor):
            _set_gradients(parameters, step, actor_norm)
        for parameters in (critic, expected_critic):
            _set_gradients(parameters, step + 3, critic_norm)
        expected_norms = (
            _manual_clip(expected_actor, threshold),
            _manual_clip(expected_critic, threshold),
        )
        actual_norms = trainer.clip_gradients(actor, critic, threshold)
        torch.testing.assert_close(actual_norms, expected_norms, rtol=2e-6, atol=2e-7)
        for actual_group, expected_group in ((actor, expected_actor), (critic, expected_critic)):
            for actual, expected in zip(actual_group, expected_group):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("fixed_group", [0, 1], ids=["actor-invariant", "critic-invariant"])
def test_other_network_gradient_scale_cannot_change_joint_adam_updates(device, trainer, fixed_group):
    control = (_parameters(device), _parameters(device))
    perturbed = (_parameters(device), _parameters(device))
    # A global-clip counterfactual proves the gradient sequence can detect the
    # original coupling despite Adam's approximate scale invariance.
    coupled = (_parameters(device), _parameters(device))
    optimizers = [
        torch.optim.Adam(groups[0] + groups[1], lr=0.0096, eps=1e-5, fused=True)
        for groups in (control, perturbed, coupled)
    ]
    for step, (fixed_norm, other_norm, factor) in enumerate(
        [(0.2, 0.1, 10000.0), (2.0, 3.0, 0.0001), (0.5, 0.3, 100.0), (0.1, 2.0, 0.01)]
    ):
        for index, (groups, optimizer) in enumerate(zip((control, perturbed, coupled), optimizers)):
            optimizer.zero_grad(set_to_none=True)
            _set_gradients(groups[fixed_group], step, fixed_norm)
            _set_gradients(groups[1 - fixed_group], step + 3, other_norm * (1.0 if index == 0 else factor))
            if index == 2:
                torch.nn.utils.clip_grad_norm_(groups[0] + groups[1], 0.5, foreach=True)
            else:
                trainer.clip_gradients(*groups, 0.5)
            optimizer.step()
        for expected, actual in zip(control[fixed_group], perturbed[fixed_group]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any(not torch.allclose(a, b) for a, b in zip(control[fixed_group], coupled[fixed_group]))
    assert any(not torch.allclose(a, b) for a, b in zip(control[1 - fixed_group], perturbed[1 - fixed_group]))


@pytest.mark.parametrize(
    "module,reference_module,value_loss,reference_loss",
    [
        (base, frozen_base, None, None),
        (categorical, frozen_twohot, "twohot", "twohot"),
        (categorical, frozen_twohot, "mse", "mse_unclipped"),
    ],
    ids=["base-unclipped", "dreamer-twohot", "scalar-mse-unclipped"],
)
def test_initialization_and_weighted_loss_gradients_preserve_frozen_trainers(
    device, module, reference_module, value_loss, reference_loss
):
    actual_options = {} if value_loss is None else dict(value_loss=value_loss, value_max_abs=20000.0)
    reference_options = {} if reference_loss is None else dict(value_loss=reference_loss, value_max_abs=20000.0)
    candidate = _agent(module, device, **actual_options)
    reference = _agent(reference_module, device, **reference_options)
    _assert_parameters(candidate, reference, exact=True)
    observations = _observations(256, device)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate.get_policy_and_value(observations),
            reference.get_policy_and_value(observations),
            rtol=0,
            atol=0,
        )
    common = dict(**SELECTED_TRUNK, norm_adv=False, ent_coef=0.03, vf_coef=0.7, clip_coef=0.2)
    actual_args = module.Args(**common, **actual_options)
    reference_args = reference_module.Args(**common, **reference_options, clip_vloss=False)
    actions, logprobs, advantages, initial_values = _policy_inputs(reference, observations)
    direction = torch.where(torch.arange(256, device=device) % 2 == 0, 1.0, -1.0)
    old_values = initial_values + direction
    returns = initial_values - 2.0 * direction
    # Value clipping would be active here, so accidentally retaining it fails.
    clipped = old_values + (initial_values - old_values).clamp(-0.2, 0.2)
    assert torch.all((clipped - returns).square() > (initial_values - returns).square())
    if value_loss == "twohot":
        targets = candidate.value_support.project(returns)
        torch.testing.assert_close(targets, reference.value_support.project(returns), rtol=0, atol=0)
    else:
        targets = returns
    saved_targets, saved_advantages = targets.clone(), advantages.clone()
    actor_parameters, critic_parameters = tuple(candidate.actor.parameters()), tuple(candidate.critic.parameters())
    reference_actor, reference_critic = tuple(reference.actor.parameters()), tuple(reference.critic.parameters())
    optimizers = [
        torch.optim.Adam(agent.parameters(), lr=0.0096, eps=1e-5, fused=True) for agent in (candidate, reference)
    ]
    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        # The reference receives old values that would activate value clipping.
        # The new unclipped objectives no longer accept or gather old values.
        batch = observations, actions, logprobs, advantages, targets
        expected, expected_metrics = reference_module.ppo_loss(
            reference, *batch, old_values + step * 20.0, reference_args
        )
        actual, actual_metrics = module.ppo_loss(candidate, *batch, actual_args)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-6, atol=2e-7)
        expected.backward()
        actual.backward()
        _assert_parameters(candidate, reference, gradients=True)
        # After the zero-initialized categorical head's first update, its
        # trunk receives a real gradient too; comparing only step zero misses it.
        if step == 1:
            for network in (candidate.actor[0], candidate.critic[0]):
                assert sum(p.grad.square().sum() for p in network.parameters()) > 0
        expected_norms = (
            torch.nn.utils.clip_grad_norm_(reference_actor, 0.5, foreach=True),
            torch.nn.utils.clip_grad_norm_(reference_critic, 0.5, foreach=True),
        )
        actual_norms = module.clip_gradients(actor_parameters, critic_parameters, 0.5)
        torch.testing.assert_close(actual_norms, expected_norms, rtol=2e-6, atol=2e-7)
        _assert_parameters(candidate, reference, gradients=True)
        for optimizer in optimizers:
            optimizer.step()
        _assert_parameters(candidate, reference)
    torch.testing.assert_close(targets, saved_targets, rtol=0, atol=0)
    torch.testing.assert_close(advantages, saved_advantages, rtol=0, atol=0)


def test_compiled_loss_feeds_independent_clipping_and_joint_fused_adam(device, trainer):
    candidate, reference = _agent(trainer, device), _agent(trainer, device)
    args = trainer.Args(**SELECTED_TRUNK, norm_adv=False, ent_coef=0.03, vf_coef=0.7)
    observations = _observations(256, device)
    actions, logprobs, advantages, old_values = _policy_inputs(candidate, observations)
    returns = observations.new_tensor([-3.0, 3.0]).repeat(128)
    targets = candidate.value_support.project(returns) if trainer is categorical else returns
    batch = observations, actions, logprobs, advantages, targets
    compiled_loss = torch.compile(
        lambda *inputs: trainer.ppo_loss(candidate, *inputs, args),
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    groups = [(tuple(agent.actor.parameters()), tuple(agent.critic.parameters())) for agent in (candidate, reference)]
    optimizers = [
        torch.optim.Adam(agent.parameters(), lr=0.0096, eps=1e-5, fused=True) for agent in (candidate, reference)
    ]
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        actual, actual_metrics = compiled_loss(*batch)
        expected, expected_metrics = trainer.ppo_loss(reference, *batch, args)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-4, atol=2e-5)
        actual.backward()
        expected.backward()
        actual_norms = trainer.clip_gradients(*groups[0], 0.5)
        expected_norms = tuple(torch.nn.utils.clip_grad_norm_(group, 0.5, foreach=True) for group in groups[1])
        torch.testing.assert_close(actual_norms, expected_norms, rtol=2e-4, atol=2e-5)
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            torch.testing.assert_close(
                candidate.get_policy_and_value(observations),
                reference.get_policy_and_value(observations),
                rtol=3e-4,
                atol=3e-5,
            )
        del actual, actual_metrics, actual_norms
