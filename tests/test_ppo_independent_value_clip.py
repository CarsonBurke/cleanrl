"""CUDA contracts for scalar value clipping alongside independent gradient clipping."""

from dataclasses import replace

import pytest
import torch

from cleanrl import ppo_continuous_action as frozen_ground
from cleanrl import ppo_continuous_action_indclip_v1 as ground
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_indclip_v4 as scalar
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4 as frozen_twohot
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v5 as categorical
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_v2 as frozen_scalar
from test_ppo_normres_independent_clip import _manual_clip, _set_gradients
from test_ppo_normres_twohot import (
    SELECTED_TRUNK,
    _agent,
    _assert_parameters,
    _observations,
    _policy_inputs,
    _spaces,
    device,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]

SCALAR_CASES = [
    pytest.param(ground, frozen_ground, {}, id="ground"),
    pytest.param(scalar, frozen_scalar, {}, id="normres-scalar"),
    pytest.param(categorical, frozen_scalar, {"value_loss": "mse"}, id="normres-categorical-mse"),
]
ALL_CASES = [
    pytest.param(ground, {}, id="ground"),
    pytest.param(scalar, {}, id="normres-scalar"),
    pytest.param(categorical, {"value_loss": "mse"}, id="normres-categorical-mse"),
    pytest.param(categorical, {"value_loss": "twohot"}, id="normres-twohot"),
]


def _make_agent(module, device, options):
    if module in (ground, frozen_ground):
        torch.manual_seed(1)
        with torch.device(device):
            return module.Agent(_spaces())
    return _agent(module, device, **options)


def _args(module, options, *, clip_vloss=True):
    trunk = {} if module in (ground, frozen_ground) else SELECTED_TRUNK
    return module.Args(
        **trunk,
        **options,
        norm_adv=False,
        ent_coef=0.03,
        vf_coef=0.7,
        clip_coef=0.2,
        clip_vloss=clip_vloss,
    )


def _groups(agent):
    return tuple(agent.actor.parameters()), tuple(agent.critic.parameters())


def _optimizer(agent):
    return torch.optim.Adam(agent.parameters(), lr=0.0096, eps=1e-5, fused=True)


def _scalar_batch(agent, observations):
    actions, logprobs, advantages, initial_values = _policy_inputs(agent, observations)
    index = torch.arange(observations.shape[0], device=observations.device)
    direction = torch.where(index % 2 == 0, 1.0, -1.0)
    active = index % 4 < 2
    old_values = initial_values + torch.where(active, direction, 0.0)
    targets = initial_values - 4.0 * direction - 2.0
    clipped = old_values + (initial_values - old_values).clamp(-0.2, 0.2)
    # Half the examples have a saturated, strictly dominant clipped error;
    # the remainder keep a live critic gradient, including its trunk.
    assert torch.all((clipped - targets).square()[active] > (initial_values - targets).square()[active])
    torch.testing.assert_close(clipped[~active], initial_values[~active], rtol=0, atol=0)
    return observations, actions, logprobs, advantages * 10.0, targets, old_values


@pytest.mark.parametrize("module,reference_module,options", SCALAR_CASES)
@pytest.mark.parametrize("clip_vloss", [True, False], ids=["clipped", "explicit-unclipped"])
def test_scalar_initialization_objective_and_gradients_match_frozen_reference(
    device, module, reference_module, options, clip_vloss
):
    candidate = _make_agent(module, device, options)
    reference = _make_agent(reference_module, device, {})
    _assert_parameters(candidate, reference, exact=True)
    observations = _observations(256, device)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate.get_policy_and_value(observations),
            reference.get_policy_and_value(observations),
            rtol=0,
            atol=0,
        )
        # Compare executed hidden representations, not module names or source.
        for actual, expected in ((candidate.actor, reference.actor), (candidate.critic, reference.critic)):
            torch.testing.assert_close(actual[:-1](observations), expected[:-1](observations), rtol=0, atol=0)
    args = _args(module, options, clip_vloss=clip_vloss)
    reference_args = _args(reference_module, {}, clip_vloss=clip_vloss)
    batch = _scalar_batch(reference, observations)
    snapshots = tuple(tensor.clone() for tensor in batch[3:])
    actual, actual_metrics = module.ppo_loss(candidate, *batch, args)
    expected, expected_metrics = reference_module.ppo_loss(reference, *batch, reference_args)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-6, atol=2e-7)
    actual.backward()
    expected.backward()
    _assert_parameters(candidate, reference, gradients=True)
    for network in (candidate.actor[:-1], candidate.critic[:-1]):
        assert sum(parameter.grad.square().sum() for parameter in network.parameters()) > 0

    # A passing frozen-reference comparison must not hide an inert clip branch.
    opposite, opposite_metrics = module.ppo_loss(candidate, *batch, replace(args, clip_vloss=not clip_vloss))
    clipped_metrics, unclipped_metrics = (
        (actual_metrics, opposite_metrics) if clip_vloss else (opposite_metrics, actual_metrics)
    )
    assert clipped_metrics[1] > unclipped_metrics[1]
    assert not torch.isclose(actual, opposite, rtol=1e-5, atol=1e-6)
    opposite_gradients = torch.autograd.grad(opposite, tuple(candidate.critic.parameters()))
    assert any(
        not torch.allclose(parameter.grad, opposite_gradient, rtol=1e-5, atol=1e-6)
        for parameter, opposite_gradient in zip(candidate.critic.parameters(), opposite_gradients)
    )
    for tensor, snapshot in zip(batch[3:], snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


def test_twohot_loss_and_gradients_ignore_old_values_and_value_clip_toggle(device):
    options = dict(value_loss="twohot", value_max_abs=20000.0)
    candidate = _make_agent(categorical, device, options)
    reference = _make_agent(frozen_twohot, device, options)
    _assert_parameters(candidate, reference, exact=True)
    observations = _observations(256, device)
    actions, logprobs, advantages, old_values = _policy_inputs(reference, observations)
    targets = candidate.value_support.project(observations.new_tensor([-6000.0, 6000.0]).repeat(128))
    torch.testing.assert_close(
        targets, reference.value_support.project(observations.new_tensor([-6000.0, 6000.0]).repeat(128)), rtol=0, atol=0
    )
    batch = observations, actions, logprobs, advantages, targets
    snapshots = tuple(tensor.clone() for tensor in (advantages, targets, old_values))
    args = _args(categorical, options)
    # The frozen independent v4 objective has neither clip_vloss nor old_values.
    reference_args = frozen_twohot.Args(
        placement="pre",
        norm_kind="rms",
        activation="stiglu",
        value_loss="twohot",
        value_max_abs=20000.0,
        norm_adv=False,
        ent_coef=0.03,
        vf_coef=0.7,
        clip_coef=0.2,
    )
    optimizers = [_optimizer(agent) for agent in (candidate, reference)]
    extreme = torch.finfo(old_values.dtype).max
    old_value_cases = (old_values, torch.full_like(old_values, extreme), torch.full_like(old_values, -extreme))
    for step in range(3):
        optimizers[1].zero_grad(set_to_none=True)
        expected, expected_metrics = frozen_twohot.ppo_loss(reference, *batch, reference_args)
        expected.backward()
        for clip_vloss in (True, False):
            for values in old_value_cases:
                optimizers[0].zero_grad(set_to_none=True)
                actual, actual_metrics = categorical.ppo_loss(
                    candidate, *batch, values, replace(args, clip_vloss=clip_vloss)
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_metrics, expected_metrics, rtol=0, atol=0)
                actual.backward()
                _assert_parameters(candidate, reference, gradients=True, exact=True)
        if step == 1:
            # The initial zero categorical head masks trunk regressions; update it.
            assert sum(parameter.grad.square().sum() for parameter in candidate.critic[0].parameters()) > 0
        categorical.clip_gradients(*_groups(candidate), 0.5)
        frozen_twohot.clip_gradients(*_groups(reference), 0.5)
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            torch.testing.assert_close(
                candidate.get_policy_and_value(observations),
                reference.get_policy_and_value(observations),
                rtol=0,
                atol=0,
            )
    for tensor, snapshot in zip((advantages, targets, old_values), snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("module,options", ALL_CASES)
@pytest.mark.parametrize("fixed_group", [0, 1], ids=["actor-isolated", "critic-isolated"])
def test_actual_agent_adam_updates_are_isolated_from_other_network_gradients(device, module, options, fixed_group):
    control, perturbed, coupled = [_make_agent(module, device, options) for _ in range(3)]
    agents = control, perturbed, coupled
    groups = [_groups(agent) for agent in agents]
    optimizers = [_optimizer(agent) for agent in agents]
    observations = _observations(32, device)
    # Reuse the sign-changing Adam fixture on complete real agent networks.
    # A globally clipped counterfactual must fail the same observable contract.
    for step, (fixed_norm, other_norm, factor) in enumerate(
        [(0.2, 0.1, 10000.0), (2.0, 3.0, 0.0001), (0.5, 0.3, 100.0), (0.1, 2.0, 0.01)]
    ):
        for index, (networks, optimizer) in enumerate(zip(groups, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            _set_gradients(networks[fixed_group], step, fixed_norm)
            _set_gradients(networks[1 - fixed_group], step + 3, other_norm * (1.0 if index == 0 else factor))
            if index == 2:
                torch.nn.utils.clip_grad_norm_(networks[0] + networks[1], 0.5, foreach=True)
            else:
                module.clip_gradients(*networks, 0.5)
            optimizer.step()
        with torch.no_grad():
            expected = control.get_policy_and_value(observations)
            actual = perturbed.get_policy_and_value(observations)
            selected = slice(0, 2) if fixed_group == 0 else slice(2, 3)
            torch.testing.assert_close(actual[selected], expected[selected], rtol=0, atol=0)
            if step == 3:
                counterfactual = coupled.get_policy_and_value(observations)
                assert any(not torch.allclose(a, b) for a, b in zip(expected[selected], counterfactual[selected]))
                other = slice(2, 3) if fixed_group == 0 else slice(0, 2)
                assert any(not torch.allclose(a, b) for a, b in zip(expected[other], actual[other]))


@pytest.mark.parametrize("module,reference_module,options", SCALAR_CASES)
def test_compiled_scalar_ppo_independent_clipping_and_adam_match_exact_reference(
    device, module, reference_module, options
):
    candidate = _make_agent(module, device, options)
    reference = _make_agent(reference_module, device, {})
    args, reference_args = _args(module, options), _args(reference_module, {})
    observations = _observations(256, device)
    batch = _scalar_batch(reference, observations)
    snapshots = tuple(tensor.clone() for tensor in batch[3:])
    with torch.no_grad():
        initial_outputs = tuple(tensor.clone() for tensor in reference.get_policy_and_value(observations))
    compiled_loss = torch.compile(
        lambda *inputs: module.ppo_loss(candidate, *inputs, args),
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    groups = [_groups(agent) for agent in (candidate, reference)]
    optimizers = [_optimizer(agent) for agent in (candidate, reference)]
    for step in range(4):
        # One graph lifetime per optimizer update; rollout old_values are fixed.
        torch.compiler.cudagraph_mark_step_begin()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        actual, actual_metrics = compiled_loss(*batch)
        expected, expected_metrics = reference_module.ppo_loss(reference, *batch, reference_args)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-4, atol=2e-5)
        actual.backward()
        expected.backward()
        expected_norms = tuple(_manual_clip(network, 0.5) for network in groups[1])
        actual_norms = module.clip_gradients(*groups[0], 0.5)
        torch.testing.assert_close(actual_norms, expected_norms, rtol=2e-4, atol=2e-5)
        if step == 0:
            assert all(norm > 0.5 for norm in expected_norms)
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            torch.testing.assert_close(
                candidate.get_policy_and_value(observations),
                reference.get_policy_and_value(observations),
                rtol=3e-4,
                atol=3e-5,
            )
        # Do not carry cudagraph-owned forward outputs into its next replay.
        del actual, actual_metrics, actual_norms
    with torch.no_grad():
        final_outputs = candidate.get_policy_and_value(observations)
        assert not torch.allclose(final_outputs[0], initial_outputs[0])
        assert not torch.allclose(final_outputs[2], initial_outputs[2])
    for tensor, snapshot in zip(batch[3:], snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
