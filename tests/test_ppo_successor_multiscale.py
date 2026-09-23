"""CUDA behavioral contracts for v15. Execute through mlq, never on CPU."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_successor_multiscale_v15 as model
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.runtime import configure_runtime
from scripts.audit_successor_multiscale import finite_returns


MODES = ("scalar", "successor", "multiscale")
AUXILIARY_NAMES = ("contrast99", "value90", "contrast90", "value97", "contrast97")


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-2.0, 3.0, (6,), dtype=np.float32),
    )


def make_agent(envs, mode):
    torch.manual_seed(17)
    return model.Agent(envs, model.Args(critic_mode=mode)).cuda()


def physical_factors(coordinates):
    """Independent inverse of [value, contrast] head coordinates."""
    pairs = coordinates.reshape(*coordinates.shape[:-1], -1, 2)
    value, contrast = pairs.unbind(-1)
    return torch.stack((value / 2 + contrast / np.sqrt(2),
                        value / 2 - contrast / np.sqrt(2)), -1).flatten(-2)


def manual_auxiliary(predictions):
    if predictions.shape[-1] == 1:
        return predictions[..., :0]
    pairs = predictions.reshape(*predictions.shape[:-1], -1, 2)
    run, control = pairs.unbind(-1)
    contrasts = (run - control) / np.sqrt(2)
    coordinates = [contrasts[..., 0]]
    for horizon in range(1, pairs.shape[-2]):
        coordinates.extend(((run[..., horizon] + control[..., horizon]) / np.sqrt(2),
                            contrasts[..., horizon]))
    return torch.stack(coordinates, -1)


def reference_targets(features, predictions, next_predictions, terms, truncs, discounts, lam):
    """Explicit finite trace, independent of both production target/GAE functions."""
    width = features.shape[-1]
    result = torch.empty_like(predictions)
    for horizon, gamma in enumerate(discounts):
        block = slice(horizon * width, (horizon + 1) * width)
        advantage = torch.zeros_like(features[0])
        for step in reversed(range(features.shape[0])):
            bootstrap = (1 - terms[step]).unsqueeze(-1)
            trace = ((1 - terms[step]) * (1 - truncs[step])).unsqueeze(-1)
            delta = (features[step] + gamma * bootstrap * next_predictions[step, :, block]
                     - predictions[step, :, block])
            advantage = delta + gamma * lam * trace * advantage
            result[step, :, block] = predictions[step, :, block] + advantage
    return result.detach()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_independent_horizon_targets_respect_factual_boundaries_and_tail(mode, compiled):
    # Nondefault task gamma also catches an accidentally hardcoded .99 horizon.
    discounts = (0.995, 0.90, 0.97) if mode == "multiscale" else (0.995,)
    width = 1 if mode == "scalar" else 2
    features = torch.arange(1, 41, device="cuda", dtype=torch.float32).reshape(4, 5, 2)
    if width == 1:
        features = features.sum(-1, keepdim=True)
    features.requires_grad_()
    predictions = torch.randn(4, 5, width * len(discounts), device="cuda", requires_grad=True)
    following = (7 * torch.randn_like(predictions)).requires_grad_()
    terms = torch.zeros(4, 5, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[0, 0] = terms[0, 2] = terms[1, 3] = 1
    truncs[0, 1] = truncs[0, 2] = 1
    gae = get_gae_fn(compiled=compiled, explicit_next_values=True)
    targets = model.multiscale_lambda_targets(
        features, predictions, following, terms, truncs, discounts, 0.63, gae,
    )
    expected = reference_targets(features, predictions, following, terms, truncs, discounts, 0.63)
    torch.testing.assert_close(targets, expected, rtol=3e-6, atol=2e-5)
    assert targets.grad_fn is None and not targets.requires_grad
    for horizon, gamma in enumerate(discounts):
        block = slice(horizon * width, (horizon + 1) * width)
        torch.testing.assert_close(targets[0, 0, block], features[0, 0])
        torch.testing.assert_close(targets[0, 1, block], features[0, 1] + gamma * following[0, 1, block])
        torch.testing.assert_close(targets[0, 2, block], features[0, 2])
        torch.testing.assert_close(targets[-1, :, block], features[-1] + gamma * following[-1, :, block])
    # Reset trajectories must never leak across termination, truncation, or both.
    changed_features, changed_predictions, changed_next = (
        tensor.detach().clone() for tensor in (features, predictions, following)
    )
    changed_features[1:, :3] += 1000
    changed_predictions[1:, :3] -= 300
    changed_next[1:, :3] += 200
    changed_next[0, (0, 2)] += 50
    changed = model.multiscale_lambda_targets(
        changed_features, changed_predictions, changed_next, terms, truncs, discounts, 0.63, gae,
    )
    torch.testing.assert_close(changed[0, :3], targets[0, :3])
    changed_next[0, 1] += 10
    changed = model.multiscale_lambda_targets(
        changed_features, changed_predictions, changed_next, terms, truncs, discounts, 0.63, gae,
    )
    for horizon, gamma in enumerate(discounts):
        block = slice(horizon * width, (horizon + 1) * width)
        torch.testing.assert_close(changed[0, 1, block] - targets[0, 1, block],
                                   torch.full((width,), gamma * 10, device="cuda"))
    scalar_features = features if width == 1 else features.sum(-1, keepdim=True)
    scalar_target = model.successor_lambda_targets(
        scalar_features, model.value_readout(predictions).unsqueeze(-1),
        model.value_readout(following).unsqueeze(-1), terms, truncs, discounts[0], 0.63, gae,
    )
    torch.testing.assert_close(model.value_readout(targets), scalar_target[..., 0], rtol=3e-6, atol=3e-5)


def test_auxiliary_horizons_cannot_change_task_returns_or_readout(envs):
    agent = make_agent(envs, "multiscale")
    observations = torch.randn(20, 17, device="cuda")
    before = agent.get_successor(observations).detach()
    expected = before[:, 0] + before[:, 1]
    torch.testing.assert_close(model.value_readout(before), expected)
    changed = before.clone().requires_grad_()
    gradient, = torch.autograd.grad(model.value_readout(changed).sum(), changed)
    torch.testing.assert_close(gradient[:, :2], torch.ones_like(gradient[:, :2]))
    torch.testing.assert_close(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))
    with torch.no_grad():
        agent.critic.head[0].weight[2:].mul_(100)
        agent.critic.head[0].bias[2:].add_(40)
    after = agent.get_successor(observations)
    assert bool((after[:, 2:] != before[:, 2:]).any())
    torch.testing.assert_close(agent.get_value(observations).flatten(), expected, rtol=0, atol=0)
    torch.testing.assert_close(agent.get_policy_and_value(observations)[2].flatten(), expected, rtol=0, atol=0)
    features = torch.randn(4, 5, 2, device="cuda")
    masks = torch.zeros(4, 5, device="cuda")
    gae = get_gae_fn(explicit_next_values=True)
    original_targets = model.multiscale_lambda_targets(
        features, before.reshape(4, 5, 6), before.reshape(4, 5, 6) + 0.3,
        masks, masks, agent.discounts, 0.95, gae,
    )
    changed_targets = model.multiscale_lambda_targets(
        features, after.reshape(4, 5, 6), after.reshape(4, 5, 6) + 0.3,
        masks, masks, agent.discounts, 0.95, gae,
    )
    torch.testing.assert_close(model.value_readout(changed_targets), model.value_readout(original_targets),
                               rtol=0, atol=0)
    torch.testing.assert_close(model.value_readout(changed_targets) - model.value_readout(after.reshape(4, 5, 6)),
                               model.value_readout(original_targets) - model.value_readout(before.reshape(4, 5, 6)),
                               rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_auxiliary_metric_is_uncentered_per_coordinate_and_loss_only(mode):
    channels = {"scalar": 1, "successor": 2, "multiscale": 6}[mode]
    targets = (torch.randn(12, channels, device="cuda") + torch.arange(channels, device="cuda") * 3).requires_grad_()
    predictions = torch.randn_like(targets, requires_grad=True)
    old_values = (model.value_readout(predictions).detach() - 0.4).requires_grad_()
    args = model.Args(critic_mode=mode, clip_vloss=True)
    expected_auxiliary = manual_auxiliary(targets.detach())
    scale = model.successor_loss_scale(targets)
    torch.testing.assert_close(model.auxiliary_coordinates(targets), expected_auxiliary)
    torch.testing.assert_close(scale, (expected_auxiliary.square().mean(0) + 1e-8).sqrt())
    assert scale.grad_fn is None and not scale.requires_grad
    scale.requires_grad_()
    snapshots = [tensor.detach().clone() for tensor in (predictions, targets, old_values)]
    value, auxiliary = model.critic_losses(predictions, targets, old_values, args, scale)
    values, returns = model.value_readout(predictions), model.value_readout(targets.detach())
    clipped = old_values.detach() + (values - old_values.detach()).clamp(-args.clip_coef, args.clip_coef)
    expected_value = 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
    torch.testing.assert_close(value, expected_value)
    if channels == 1:
        torch.testing.assert_close(auxiliary, torch.zeros((), device="cuda"))
    else:
        error = manual_auxiliary(predictions) - expected_auxiliary
        torch.testing.assert_close(auxiliary, 0.5 * (error / scale.detach()).square().mean())
        changed_scale = scale.detach().clone()
        changed_scale[0] *= 7
        other_value, other_auxiliary = model.critic_losses(predictions, targets, old_values, args, changed_scale)
        torch.testing.assert_close(other_value, value, rtol=0, atol=0)
        torch.testing.assert_close(other_auxiliary, 0.5 * (error / changed_scale).square().mean())
        # Shifting only the task value is not counted again as an auxiliary target.
        shifted = predictions.detach().clone()
        shifted[:, :2] += 2
        _, shifted_auxiliary = model.critic_losses(shifted, targets, old_values, args, scale)
        torch.testing.assert_close(shifted_auxiliary, auxiliary, rtol=2e-6, atol=2e-6)
    (value + auxiliary).backward()
    assert targets.grad is None and old_values.grad is None and scale.grad is None
    for tensor, snapshot in zip((predictions, targets, old_values), snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("successor", "multiscale"))
def test_fixed_auxiliary_budget_and_task_coefficient_do_not_grow_with_horizons(mode):
    channels = 2 if mode == "successor" else 6
    # Every standardized auxiliary coordinate has unit error: total loss is .5,
    # regardless of whether there is one or five coordinates.
    coordinates = torch.ones(8, channels, device="cuda")
    coordinates[:, 0] = 2
    coordinates[:, 2::2] = np.sqrt(2)
    predictions = physical_factors(coordinates).requires_grad_()
    targets = torch.zeros_like(predictions)
    scale = torch.ones(channels - 1, device="cuda")
    value, auxiliary = model.critic_losses(predictions, targets, torch.zeros(8, device="cuda"),
                                           model.Args(clip_vloss=False), scale)
    torch.testing.assert_close(value, torch.tensor(2.0, device="cuda"))
    torch.testing.assert_close(auxiliary, torch.tensor(0.5, device="cuda"))
    gradient, = torch.autograd.grad(value, predictions)
    expected = torch.zeros_like(predictions)
    expected[:, :2] = 2 / 8
    torch.testing.assert_close(gradient, expected)


def test_every_auxiliary_head_and_shared_stage_has_correct_gradient_ownership(envs):
    agent = make_agent(envs, "multiscale")
    observations = torch.randn(24, 17, device="cuda", requires_grad=True)
    actor, critic = agent.parameter_groups()
    args = model.Args(critic_mode="multiscale", clip_vloss=False)
    scale = torch.arange(1, 6, device="cuda", dtype=torch.float32)
    shared = tuple(agent.critic.first.parameters()) + tuple(agent.critic.second.parameters())
    for coordinate in range(5):
        agent.zero_grad(set_to_none=True)
        predictions = agent.get_successor(observations)
        # Only one auxiliary coordinate is wrong in each pass. Every head must
        # train its own coordinate and both shared stages, never the task head.
        head_targets = agent.critic(observations.detach()).detach().clone()
        head_targets[:, coordinate + 1] += 3
        targets = physical_factors(head_targets)
        _, auxiliary = model.critic_losses(predictions, targets, model.value_readout(predictions).detach(), args, scale)
        manual = 0.5 * ((manual_auxiliary(predictions) - manual_auxiliary(targets)) / scale).square().mean()
        expected = torch.autograd.grad(manual, critic, retain_graph=True)
        auxiliary.backward()
        for parameter, gradient in zip(critic, expected, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-7)
        assert all(parameter.grad is None for parameter in actor)
        assert observations.grad is None
        assert all(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in shared)
        head_gradient = agent.critic.head[0].weight.grad
        assert bool(head_gradient[coordinate + 1].abs().sum() > 0)
        other_rows = [index for index in range(6) if index != coordinate + 1]
        torch.testing.assert_close(head_gradient[other_rows], torch.zeros_like(head_gradient[other_rows]),
                                   rtol=0, atol=2e-7)
    agent.zero_grad(set_to_none=True)
    predictions = agent.get_successor(observations)
    targets = predictions.detach().clone()
    targets[:, :2] += 1
    value, _ = model.critic_losses(predictions, targets, model.value_readout(predictions).detach(), args, scale)
    value.backward()
    assert all(parameter.grad is None for parameter in actor)
    assert all(bool(parameter.grad.abs().sum() > 0) for parameter in shared)
    assert bool(agent.critic.head[0].weight.grad[0].abs().sum() > 0)
    torch.testing.assert_close(agent.critic.head[0].weight.grad[1:],
                               torch.zeros_like(agent.critic.head[0].weight.grad[1:]), rtol=0, atol=2e-7)


def test_all_modes_match_initialization_rng_and_scalar_adam_updates(envs):
    agents, random_states = [], []
    for mode in MODES:
        agents.append(make_agent(envs, mode))
        random_states.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
    observations = torch.randn(64, 17, device="cuda")
    returns = torch.randn(64, device="cuda")
    for agent, states, dimension in zip(agents, random_states, (1, 2, 6), strict=True):
        assert agent.successor_dim == dimension
        assert agent.discounts == ((0.99, 0.90, 0.97) if dimension == 6 else (0.99,))
        for actual, expected in zip(states, random_states[0], strict=True):
            assert torch.equal(actual, expected)
        for actual, expected in zip(agent.actor.parameters(), agents[0].actor.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(agent.get_policy_and_value(observations),
                                    agents[0].get_policy_and_value(observations), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-7)
    optimizers = [torch.optim.Adam(agent.critic.parameters(), lr=3e-4, eps=1e-5) for agent in agents]
    for _ in range(4):
        for agent, optimizer in zip(agents, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * (agent.get_value(observations).flatten() - returns).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            optimizer.step()
        for agent in agents[1:]:
            torch.testing.assert_close(agent.get_value(observations), agents[0].get_value(observations),
                                       rtol=3e-5, atol=3e-6)


def training_batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda", requires_grad=True)
    native = (torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1).requires_grad_()
    with torch.no_grad():
        predictions, logprobs = model.rollout_statistics(agent, observations, native)
    targets = (predictions + torch.randn_like(predictions)).requires_grad_()
    old_values = (model.value_readout(predictions) + 0.4).requires_grad_()
    advantages = torch.randn(rows, device="cuda", requires_grad=True)
    old_logprobs = logprobs.detach().requires_grad_()
    scale = model.successor_loss_scale(targets).requires_grad_()
    return observations, native, old_logprobs, advantages, targets, old_values, scale


def test_auxiliary_objective_cannot_change_actor_gradient_or_clipped_adam_step(envs):
    agents = [make_agent(envs, mode) for mode in MODES]
    data = training_batch(agents[-1])
    observations, native, old_logprobs, advantages, full_targets, old_values, _ = data
    gradients = []
    args = model.Args(clip_vloss=False)
    for agent in agents:
        targets = (model.value_readout(full_targets).unsqueeze(-1) if agent.successor_dim == 1
                   else full_targets[:, :agent.successor_dim])
        # Deliberately large auxiliary gradients expose accidental global clipping.
        scale = model.successor_loss_scale(targets) * 0.001
        objective, _ = model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                         targets, old_values, args, scale)
        objective.backward()
        gradients.append([parameter.grad.clone() for parameter in agent.actor.parameters()])
        groups = agent.parameter_groups()
        assert set(map(id, groups[0])).isdisjoint(map(id, groups[1]))
        assert set(map(id, groups[0] + groups[1])) == set(map(id, agent.parameters()))
        optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
        model.optimizer_step(optimizers, groups, args.max_grad_norm, torch.empty(2, device="cuda"))
    for agent, actor_gradients in zip(agents[1:], gradients[1:], strict=True):
        for actual, expected in zip(actor_gradients, gradients[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(agent.actor.parameters(), agents[0].actor.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(tensor.grad is None for tensor in data)


@pytest.mark.parametrize("mode", MODES)
def test_compiled_objective_gradients_and_separately_owned_updates_match_eager(envs, mode):
    agent = make_agent(envs, mode)
    reference = deepcopy(agent)
    args = model.Args(critic_mode=mode, clip_vloss=True)
    data = training_batch(agent)
    expected, expected_metrics = model.policy_loss(reference, *data[:-1], args, data[-1])
    expected.backward()

    def objective(observations, native, old_logprobs, advantages, targets, old_values, scale):
        return model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                  targets, old_values, args, scale)

    compiled = torch.compile(objective, fullgraph=True, options={"triton.cudagraphs": False})
    actual, actual_metrics = compiled(*data)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for name in expected_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=3e-5, atol=3e-6)
    _, auxiliary = model.critic_losses(agent.get_successor(data[0]), data[4], data[5], args, data[6])
    torch.testing.assert_close(actual_metrics["successor/auxiliary_loss"], auxiliary, rtol=3e-5, atol=3e-6)
    actual.backward()
    for parameter, counterpart in zip(agent.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, counterpart.grad, rtol=5e-4, atol=5e-6)
    groups, reference_groups = agent.parameter_groups(), reference.parameter_groups()
    optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
    reference_optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True)
                                 for group in reference_groups)
    expected_norms = torch.stack([torch.nn.utils.clip_grad_norm_(group, args.max_grad_norm)
                                  for group in reference_groups])
    for optimizer in reference_optimizers:
        optimizer.step()
    norms = torch.empty(2, device="cuda")
    before = [parameter.detach().clone() for parameter in agent.parameters()]
    model.optimizer_step(optimizers, groups, args.max_grad_norm, norms)
    torch.testing.assert_close(norms, expected_norms, rtol=5e-4, atol=5e-6)
    for parameter, counterpart, previous in zip(agent.parameters(), reference.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, counterpart, rtol=5e-4, atol=5e-6)
        assert bool(torch.isfinite(parameter).all())
        assert bool((parameter != previous).any())
    assert all(tensor.grad is None for tensor in data)


@pytest.mark.parametrize("mode", MODES)
def test_gradient_diagnostics_report_shared_trunk_without_touching_gradients_or_rng(envs, mode):
    agent = make_agent(envs, mode)
    observations, _, _, _, targets, old_values, scale = training_batch(agent)
    args = model.Args(critic_mode=mode, clip_vloss=False)
    shared = tuple(agent.critic.first.parameters()) + tuple(agent.critic.second.parameters())
    if mode != "scalar":
        value, auxiliary = model.critic_losses(agent.get_successor(observations), targets, old_values, args, scale)
        value_gradient = torch.cat([gradient.flatten() for gradient in
                                    torch.autograd.grad(value, shared, retain_graph=True)])
        auxiliary_gradient = torch.cat([gradient.flatten() for gradient in torch.autograd.grad(auxiliary, shared)])
        dot = (value_gradient * auxiliary_gradient).sum()
        expected = {
            "gradients/value_norm": value_gradient.norm(),
            "gradients/auxiliary_norm": auxiliary_gradient.norm(),
            "gradients/value_auxiliary_dot": dot,
            "gradients/value_auxiliary_cosine": dot / (value_gradient.norm() * auxiliary_gradient.norm()),
            "diagnostics/auxiliary_loss": auxiliary.detach(),
        }
        coordinates = manual_auxiliary(targets.detach())
        for index, name in enumerate(AUXILIARY_NAMES[:coordinates.shape[-1]]):
            expected[f"diagnostics/{name}/target_variance"] = coordinates[:, index].var(unbiased=False)
            expected[f"diagnostics/{name}/target_rms_squared"] = coordinates[:, index].square().mean()
    else:
        expected = {name: torch.zeros((), device="cuda") for name in (
            "gradients/value_norm", "gradients/auxiliary_norm", "gradients/value_auxiliary_dot",
            "gradients/value_auxiliary_cosine", "diagnostics/auxiliary_loss",
        )}
        value, _ = model.critic_losses(agent.get_successor(observations), targets, old_values, args, scale)
        value_gradient = torch.cat([gradient.flatten() for gradient in torch.autograd.grad(value, shared)])
        expected["gradients/value_norm"] = value_gradient.norm()
    # Preserve both existing accumulated gradients and None slots, not just a
    # freshly zeroed model. Diagnostics must not perform a hidden backward.
    gradient_snapshots = []
    for index, parameter in enumerate(agent.parameters()):
        parameter.grad = torch.full_like(parameter, 0.125) if index % 2 else None
        gradient_snapshots.append(None if parameter.grad is None else parameter.grad.clone())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    for _ in range(2):
        actual = model.gradient_diagnostics(agent, observations, targets, old_values, args, scale)
        assert set(actual) == set(expected)
        for name, value in actual.items():
            assert value.ndim == 0 and not value.requires_grad and value.grad_fn is None
            torch.testing.assert_close(value, expected[name], rtol=5e-4, atol=5e-6)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        for parameter, snapshot in zip(agent.parameters(), gradient_snapshots, strict=True):
            if snapshot is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, snapshot, rtol=0, atol=0)
    assert all(tensor.grad is None for tensor in (observations, targets, old_values, scale))


@pytest.mark.parametrize("compiled", [False, True])
def test_audit_finite_returns_match_discounted_sums_and_exclude_boundary_endpoints(compiled):
    factors = torch.arange(1, 55, device="cuda", dtype=torch.float32).reshape(9, 3, 2)
    boundaries = torch.zeros(9, 3, device="cuda", dtype=torch.bool)
    boundaries[3, 0] = True
    boundaries[7, 1] = True
    horizon, discount = 4, 0.93
    expected = torch.stack([
        sum(discount ** offset * factors[start + offset] for offset in range(horizon))
        for start in range(factors.shape[0] - horizon + 1)
    ])
    expected_valid = torch.stack([
        ~boundaries[start:start + horizon].any(0)
        for start in range(factors.shape[0] - horizon + 1)
    ])
    function = (torch.compile(finite_returns, fullgraph=True, options={"triton.cudagraphs": False})
                if compiled else finite_returns)
    actual, valid = function(factors, boundaries, horizon, discount)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(valid, expected_valid)
    # At env 0, starts 0/1/3 end at/cross/start at the boundary; start 4 is fresh.
    assert not bool(valid[0, 0]) and not bool(valid[1, 0]) and not bool(valid[3, 0])
    assert bool(valid[4, 0])
    assert bool(valid[:4, 1].all()) and not bool(valid[4:, 1].any())
    assert bool(valid[:, 2].all())
