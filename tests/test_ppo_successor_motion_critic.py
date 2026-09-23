"""CUDA behavioral contracts for critic-only v18; execute only through mlq."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from scipy.special import roots_jacobi

from cleanrl import ppo_continuous_action_successor_motion_critic_v18 as model
from cleanrl import ppo_continuous_action_successor_score_geometry_v17 as baseline
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.runtime import configure_runtime


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contracts require mlq; no CPU fallback")
MODES = ("ppo", "one_step", "successor")


@pytest.fixture(autouse=True)
def cuda_runtime():
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-2.0, 3.0, (6,), dtype=np.float32),
    )


def tensor(values, **kwargs):
    return torch.tensor(values, device="cuda", dtype=torch.float64, **kwargs)


def make_agent(envs, mode="successor"):
    torch.manual_seed(17)
    return model.Agent(envs, model.Args(aux_mode=mode)).cuda()


def prime_auxiliary(agent):
    # Zero heads legitimately give no trunk gradient on their first update.
    with torch.no_grad():
        for head in (agent.state_head, agent.action_head):
            head.weight.normal_(std=0.03)
            head.bias.normal_(std=0.1)


def assert_tensors_close(actual, expected, *, rtol=0, atol=0):
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=atol)


def gradients(parameters):
    return tuple(parameter.grad.detach().clone() for parameter in parameters)


def optimizers_for(groups, args):
    return tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)


def reference_targets(features, predictions, following, terms, truncs, gamma, lam):
    """Finite sum of TD errors, independent of the production backward recurrence."""
    features, predictions, following = (value.detach() for value in (features, predictions, following))
    result = torch.empty_like(predictions)
    for start in range(features.shape[0]):
        total = predictions[start].clone()
        alive = torch.ones_like(terms[start]).unsqueeze(-1)
        for step in range(start, features.shape[0]):
            delta = ((1 - gamma) * features[step]
                     + gamma * (1 - terms[step]).unsqueeze(-1) * following[step] - predictions[step])
            total = total + (gamma * lam) ** (step - start) * alive * delta
            alive = alive * ((1 - terms[step]) * (1 - truncs[step])).unsqueeze(-1)
        result[start] = total
    return result


def training_batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda", requires_grad=True)
    native = (0.1 + 0.8 * torch.rand(rows, 6, device="cuda")).requires_grad_()
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        distribution = torch.distributions.Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - agent.log_action_scale).sum(-1)
        basis = model.beta_action_basis(alpha, beta, native)
        # Both signs cross the clipping interval; old values also exercise value clipping.
        old_logprob = logprob - torch.linspace(-0.7, 0.6, rows, device="cuda")
        old_values = values.squeeze(-1) + torch.linspace(-0.5, 0.5, rows, device="cuda")
        returns = values.squeeze(-1) + torch.linspace(-3.0, 3.0, rows, device="cuda")
    advantages = 200 * torch.sin(torch.arange(rows, device="cuda") * 1.3) + 25
    state_targets = torch.randn(rows, 45, device="cuda") + torch.linspace(-2.0, 4.0, 45, device="cuda")
    action_targets = torch.randn(rows, 45, device="cuda") * 2 - state_targets / 3
    state_scale = (state_targets.square().mean(0) + 1e-8).sqrt()
    action_scale = (action_targets.square().mean(0) + 1e-8).sqrt()
    return tuple(value.detach().requires_grad_() for value in (
        observations, native, old_logprob, advantages, returns, old_values,
        basis, state_targets, action_targets, state_scale, action_scale,
    ))


def reference_ppo(agent, args, data):
    """Ordinary reward PPO only: no production objective or auxiliary helper."""
    observations, native, old_logprob, advantages, returns, old_values = data[:6]
    alpha, beta, values = agent.get_policy_and_value(observations.detach())
    distribution = torch.distributions.Beta(alpha, beta, validate_args=False)
    logprob = (distribution.log_prob(native.detach()) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    logratio = logprob - old_logprob.detach()
    ratio = logratio.exp()
    advantage = advantages.detach()
    if args.norm_adv:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    policy = torch.maximum(-advantage * ratio,
                           -advantage * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    values = values.squeeze(-1)
    squared_error = (values - returns.detach()).square()
    if args.clip_vloss:
        clipped = old_values.detach() + (values - old_values.detach()).clamp(-args.clip_coef, args.clip_coef)
        squared_error = torch.maximum(squared_error, (clipped - returns.detach()).square())
    value = 0.5 * squared_error.mean()
    metrics = {
        "losses/policy_loss": policy,
        "losses/value_loss": value,
        "losses/entropy": entropy,
        "losses/old_approx_kl": -logratio.mean(),
        "losses/approx_kl": ((ratio - 1) - logratio).mean(),
        "losses/clipfrac": ((ratio - 1).abs() > args.clip_coef).float().mean(),
    }
    return policy - args.ent_coef * entropy + args.vf_coef * value, metrics


def test_motion_features_retain_physical_units_and_joint_product_order():
    raw = torch.linspace(-3.1, 5.4, 2 * 3 * 17, device="cuda", dtype=torch.float64).reshape(2, 3, 17)
    raw[..., 0] += 2.0
    raw[..., 8:11] *= 9.0
    snapshot = raw.clone()
    expected = torch.stack([
        raw[..., 0],
        *(raw[..., index].sin() for index in range(1, 8)),
        *(raw[..., index].cos() for index in range(1, 8)),
        *(raw[..., index] for index in range(8, 17)),
        *(raw[..., first] * raw[..., second] for first in range(11, 17) for second in range(first, 17)),
    ], -1)
    features = model.motion_features(raw)
    torch.testing.assert_close(features, expected, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(raw, snapshot, rtol=0, atol=0)
    changed = raw.clone()
    changed[..., 1:8] += 2 * torch.pi
    changed[..., 8:11] += 1000
    changed_features = model.motion_features(changed)
    torch.testing.assert_close(changed_features[..., :15], features[..., :15], rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(changed_features[..., 18:], features[..., 18:], rtol=0, atol=0)
    torch.testing.assert_close(changed_features[..., 15:18], features[..., 15:18] + 1000, rtol=0, atol=0)


def test_factual_features_use_final_not_autoreset_observations_at_every_boundary():
    reset = np.arange(4 * 17, dtype=np.float32).reshape(4, 17) / 10
    snapshot = reset.copy()
    empty = np.zeros(4, dtype=bool)
    assert model.factual_next_observations(reset, empty, empty, {}) is reset
    terms = np.array([True, False, True, False])
    truncs = np.array([False, True, True, False])
    finals = np.empty(4, dtype=object)
    finals[:] = None
    for index in range(3):
        finals[index] = reset[index] + 100 * (index + 1)
    infos = {"final_observation": finals, "_final_observation": terms | truncs}
    factual = model.factual_next_observations(reset, terms, truncs, infos)
    expected = reset.copy()
    expected[:3] = np.stack(finals[:3])
    np.testing.assert_array_equal(factual, expected)
    np.testing.assert_array_equal(reset, snapshot)
    # This is the actual downstream physical-feature path, not merely an array substitution.
    actual_features = model.motion_features(torch.as_tensor(factual, device="cuda"))
    expected_features = model.motion_features(torch.as_tensor(expected, device="cuda"))
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    assert not torch.equal(actual_features[:3], model.motion_features(torch.as_tensor(reset[:3], device="cuda")))


@pytest.mark.parametrize("invalid", ["missing", "masked", "none", "shape", "nonfinite"])
def test_boundary_without_a_valid_factual_final_is_rejected(invalid):
    reset = np.zeros((2, 17), dtype=np.float32)
    terms, truncs = np.array([True, False]), np.array([False, True])
    finals = np.empty(2, dtype=object)
    finals[0], finals[1] = np.ones(17, dtype=np.float32), np.full(17, 2.0, dtype=np.float32)
    infos = {"final_observation": finals, "_final_observation": np.ones(2, dtype=bool)}
    if invalid == "missing":
        infos = {}
    elif invalid == "masked":
        infos["_final_observation"][1] = False
    elif invalid == "none":
        finals[0] = None
    elif invalid == "shape":
        finals[1] = np.ones(16, dtype=np.float32)
    else:
        finals[0][3] = np.nan
    with pytest.raises(FloatingPointError if invalid == "nonfinite" else RuntimeError):
        model.factual_next_observations(reset, terms, truncs, infos)
    np.testing.assert_array_equal(reset, np.zeros_like(reset))


@pytest.mark.parametrize("compiled", [False, True])
def test_vector_lambda_targets_match_finite_sum_and_cut_resets_but_bootstrap_truncations(compiled):
    features = torch.linspace(-4, 8, 4 * 5 * 45, device="cuda").reshape(4, 5, 45).requires_grad_()
    predictions = torch.randn_like(features, requires_grad=True)
    following = (7 * torch.randn_like(features)).requires_grad_()
    terms = torch.zeros(4, 5, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[0, 0] = terms[0, 2] = terms[1, 3] = 1
    truncs[0, 1] = truncs[0, 2] = 1
    gamma, lam = 0.87, 0.61
    gae = get_gae_fn(compiled=compiled, explicit_next_values=True)
    actual = model.successor_lambda_targets(features, predictions, following, terms, truncs, gamma, lam, gae)
    expected = reference_targets(features, predictions, following, terms, truncs, gamma, lam)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    assert not actual.requires_grad and actual.grad_fn is None
    torch.testing.assert_close(actual[0, (0, 2)], (1 - gamma) * features[0, (0, 2)], rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual[0, 1], (1 - gamma) * features[0, 1] + gamma * following[0, 1])
    torch.testing.assert_close(actual[-1], (1 - gamma) * features[-1] + gamma * following[-1])
    altered = [value.detach().clone() for value in (features, predictions, following)]
    altered[0][1:, :3] += 1000
    altered[1][1:, :3] -= 200
    altered[2][1:, :3] += 500
    altered[2][0, (0, 2)] -= 100
    changed = model.successor_lambda_targets(*altered, terms, truncs, gamma, lam, gae)
    torch.testing.assert_close(changed[0, :3], actual[0, :3], rtol=0, atol=0)
    altered[2][0, 1] += 10
    changed = model.successor_lambda_targets(*altered, terms, truncs, gamma, lam, gae)
    torch.testing.assert_close(changed[0, 1] - actual[0, 1], torch.full((45,), gamma * 10, device="cuda"))


def test_one_step_and_successor_targets_have_distinct_meaning_and_frozen_physical_rms():
    features = torch.linspace(-3, 17, 3 * 2 * 45, device="cuda").reshape(3, 2, 45)
    predictions = 2 + torch.randn_like(features)
    following = torch.randn_like(features) * 3
    for value in (features, predictions, following):
        value[..., 0] = 0
        value.requires_grad_()
    terms = torch.zeros(3, 2, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[1, 0], truncs[0, 1] = 1, 1
    gae = get_gae_fn(explicit_next_values=True)
    gamma, lam = 0.91, 0.73
    expected_successor = reference_targets(features, predictions, following, terms, truncs, gamma, lam)
    results = {}
    for mode in MODES:
        args = model.Args(aux_mode=mode, gamma=gamma, gae_lambda=lam)
        result = model.auxiliary_targets(features, predictions, following, terms, truncs, args, gae)
        results[mode] = result
        state, action, state_scale, action_scale = result
        expected_state = features.detach() if mode == "one_step" else expected_successor
        expected_action = expected_state - predictions.detach()
        torch.testing.assert_close(state, expected_state, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(action, expected_action, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(state_scale, (expected_state.square().mean((0, 1)) + 1e-8).sqrt())
        torch.testing.assert_close(action_scale, (expected_action.square().mean((0, 1)) + 1e-8).sqrt())
        torch.testing.assert_close(state_scale[0], torch.tensor(1e-4, device="cuda"))
        assert all(not value.requires_grad and value.grad_fn is None for value in result)
    assert_tensors_close(results["ppo"], results["successor"])
    assert not torch.allclose(results["one_step"][0], results["successor"][0])
    snapshots = {mode: tuple(value.clone() for value in result) for mode, result in results.items()}
    with torch.no_grad():
        features.add_(123)
        predictions.sub_(40)
        following.mul_(7)
    for mode, result in results.items():
        assert_tensors_close(result, snapshots[mode])


def test_beta_polynomial_basis_matches_product_quadrature_and_is_conditionally_orthonormal():
    # Three Gauss-Jacobi points integrate every per-action polynomial through degree five.
    # A basis Gram entry has degree at most four, so this is exact integration, not sampling.
    alpha_values = np.array([1.001, 1.2, 2.5, 7.0, 18.0, 3.1])
    beta_values = np.array([1.001, 9.0, 3.5, 2.0, 2.3, 13.0])
    one_dimensional = [roots_jacobi(3, b - 1, a - 1) for a, b in zip(alpha_values, beta_values, strict=True)]
    nodes = np.stack([(points + 1) / 2 for points, _ in one_dimensional])
    weights = np.stack([mass / mass.sum() for _, mass in one_dimensional])
    indices = np.indices((3,) * 6).reshape(6, -1).T
    native = tensor(np.stack([nodes[dim, indices[:, dim]] for dim in range(6)], -1), requires_grad=True)
    product_weights = tensor(np.prod(np.stack([weights[dim, indices[:, dim]] for dim in range(6)], -1), -1))
    alpha, beta = tensor(alpha_values, requires_grad=True), tensor(beta_values, requires_grad=True)
    actual = model.beta_action_basis(alpha.expand_as(native), beta.expand_as(native), native)

    # Independent Gram-Schmidt under the quadrature measure: no analytic skew/kurtosis helper.
    node_tensor, weight_tensor = tensor(nodes), tensor(weights)
    mean = (weight_tensor * node_tensor).sum(-1)
    centered_nodes = node_tensor - mean[:, None]
    variance = (weight_tensor * centered_nodes.square()).sum(-1)
    standardized_nodes = centered_nodes / variance.sqrt()[:, None]
    third = (weight_tensor * standardized_nodes.pow(3)).sum(-1)
    residual_nodes = standardized_nodes.square() - 1 - third[:, None] * standardized_nodes
    residual_norm = (weight_tensor * residual_nodes.square()).sum(-1).sqrt()
    standardized = (native.detach() - mean) / variance.sqrt()
    quadratic = (standardized.square() - 1 - third * standardized) / residual_norm
    expected = torch.cat((standardized, quadratic, torch.stack([
        standardized[:, first] * standardized[:, second] for first in range(6) for second in range(first + 1, 6)
    ], -1)), -1)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(product_weights @ actual, torch.zeros(27, device="cuda", dtype=torch.float64),
                               rtol=0, atol=2e-11)
    torch.testing.assert_close(actual.T @ (product_weights[:, None] * actual),
                               torch.eye(27, device="cuda", dtype=torch.float64), rtol=2e-11, atol=2e-11)
    assert not actual.requires_grad and actual.grad_fn is None
    snapshot = actual.clone()
    with torch.no_grad():
        alpha.add_(4)
        beta.mul_(2)
        native.mul_(0.8)
    torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)


def test_all_modes_preserve_v17_actor_scalar_initialization_rng_and_exclusive_owners(envs):
    torch.manual_seed(17)
    canonical = baseline.Agent(envs, baseline.Args(control_mode="ppo")).cuda()
    expected_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    agents = []
    for mode in MODES:
        agent = make_agent(envs, mode)
        agents.append(agent)
        assert torch.equal(torch.get_rng_state(), expected_rng[0])
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng[1])
        assert_tensors_close(agent.actor.parameters(), canonical.actor.parameters())
        assert_tensors_close(agent.critic.parameters(), canonical.critic.parameters())
        groups = agent.parameter_groups()
        owners = tuple(set(map(id, group)) for group in groups)
        assert len(owners) == 3
        assert owners[0] == set(map(id, agent.actor.parameters()))
        assert owners[1] == set(map(id, agent.critic.parameters()))
        assert owners[2] == set(map(id, (*agent.state_head.parameters(), *agent.action_head.parameters())))
        assert all(owners[first].isdisjoint(owners[second]) for first in range(3) for second in range(first))
        assert sum(map(len, groups)) == len(set.union(*owners))
        assert set.union(*owners) == set(map(id, agent.parameters()))
    observations = torch.randn(8, 17, device="cuda", requires_grad=True)
    sampling_rng = torch.cuda.get_rng_state()
    expected = canonical.get_action_and_value(observations)
    for agent in agents:
        torch.cuda.set_rng_state(sampling_rng)
        assert_tensors_close(agent.get_action_and_value(observations), expected)
        assert_tensors_close(agent.get_policy_and_value(observations), canonical.get_policy_and_value(observations))
        state, action = agent.auxiliary_predictions(observations, torch.randn(8, 27, device="cuda"))
        torch.testing.assert_close(state, torch.zeros(8, 45, device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(action, torch.zeros_like(state), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ppo", "successor"])
def test_auxiliary_predictions_are_action_centered_with_only_critic_side_gradients(envs, mode):
    agent = make_agent(envs, mode)
    prime_auxiliary(agent)
    observations = torch.randn(12, 17, device="cuda", requires_grad=True)
    basis = torch.randn(12, 27, device="cuda", requires_grad=True)
    state, action = agent.auxiliary_predictions(observations, basis)
    opposite_state, opposite_action = agent.auxiliary_predictions(observations, -basis)
    zero_state, zero_action = agent.auxiliary_predictions(observations, torch.zeros_like(basis))
    torch.testing.assert_close(opposite_state, state, rtol=0, atol=0)
    torch.testing.assert_close(zero_state, state, rtol=0, atol=0)
    torch.testing.assert_close(opposite_action, -action, rtol=0, atol=0)
    torch.testing.assert_close(zero_action, torch.zeros_like(action), rtol=0, atol=0)
    (state.square().mean() + action.square().mean()).backward()
    assert observations.grad is None and basis.grad is None
    assert all(parameter.grad is None for parameter in agent.actor.parameters())
    assert all(parameter.grad is None for parameter in agent.critic.head.parameters())
    for head in (agent.state_head, agent.action_head):
        assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in head.parameters())
    if mode == "ppo":
        assert all(parameter.grad is None for parameter in agent.critic.parameters())
    else:
        for stage in (agent.critic.first, agent.critic.second):
            assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in stage.parameters())


@pytest.mark.parametrize("clip_vloss", [False, True])
def test_objective_has_standard_ppo_and_fixed_two_head_auxiliary_budget(envs, clip_vloss):
    agent = make_agent(envs)
    prime_auxiliary(agent)
    args = model.Args(aux_mode="successor", ent_coef=0.013, vf_coef=0.37, clip_vloss=clip_vloss, norm_adv=True)
    data = training_batch(agent)
    ordinary, expected = reference_ppo(agent, args, data)
    state, action = agent.auxiliary_predictions(data[0], data[6])
    state_loss = 0.5 * ((state - data[7].detach()) / data[9].detach()).square().mean()
    action_loss = 0.5 * ((action - data[8].detach()) / data[10].detach()).square().mean()
    auxiliary = (state_loss + action_loss) / 2
    expected.update({"successor/state_loss": state_loss, "successor/action_loss": action_loss,
                     "successor/auxiliary_loss": auxiliary})
    actual, metrics = model.minibatch_objective(agent, args, *data)
    torch.testing.assert_close(actual, ordinary + args.vf_coef * auxiliary)
    torch.testing.assert_close(metrics, torch.stack([expected[name].detach() for name in model.MINIBATCH_METRIC_NAMES]))
    assert not metrics.requires_grad
    changed_data = (*data[:10], (data[10].detach() * 7).requires_grad_())
    changed, changed_metrics = model.minibatch_objective(agent, args, *changed_data)
    torch.testing.assert_close(changed - actual, args.vf_coef * action_loss * (1 / 49 - 1) / 2)
    named = dict(zip(model.MINIBATCH_METRIC_NAMES, metrics, strict=True))
    changed_named = dict(zip(model.MINIBATCH_METRIC_NAMES, changed_metrics, strict=True))
    for name in ("losses/policy_loss", "losses/value_loss", "losses/entropy", "successor/state_loss"):
        torch.testing.assert_close(changed_named[name], named[name], rtol=0, atol=0)
    actual.backward()
    assert all(value.grad is None for value in data)


@pytest.mark.parametrize("mode", MODES)
def test_changed_auxiliary_targets_cannot_change_actor_gradients_or_clipped_adam_step(envs, mode):
    first = make_agent(envs, mode)
    prime_auxiliary(first)
    second = deepcopy(first)
    args = model.Args(aux_mode=mode, norm_adv=False, ent_coef=0.017, clip_vloss=True)
    data = training_batch(first)
    changed = list(data)
    changed[7] = (1000 + 11 * data[7].detach().square()).requires_grad_()
    changed[8] = (-2000 - 17 * data[8].detach().square()).requires_grad_()
    changed[9] = (data[9].detach() * 0.01).requires_grad_()
    changed[10] = (data[10].detach() * 0.003).requires_grad_()
    groups = first.parameter_groups(), second.parameter_groups()
    optimizers = optimizers_for(groups[0], args), optimizers_for(groups[1], args)
    for update in range(2):
        for optimizer in (*optimizers[0], *optimizers[1]):
            optimizer.zero_grad(set_to_none=True)
        model.minibatch_objective(first, args, *data)[0].backward()
        model.minibatch_objective(second, args, *changed)[0].backward()
        assert_tensors_close(gradients(first.actor.parameters()), gradients(second.actor.parameters()))
        if mode == "ppo":
            assert_tensors_close(gradients(first.critic.parameters()), gradients(second.critic.parameters()))
        else:
            assert not torch.allclose(first.critic.first[0].weight.grad, second.critic.first[0].weight.grad)
        if update == 0:
            # Auxiliary gradients may reach the shared trunk, never the scalar readout itself.
            assert_tensors_close(gradients(first.critic.head.parameters()), gradients(second.critic.head.parameters()))
        norms = torch.empty(3, device="cuda"), torch.empty(3, device="cuda")
        for owned_optimizers, owned_groups, result in zip(optimizers, groups, norms, strict=True):
            model.optimizer_step(owned_optimizers, owned_groups, args.max_grad_norm, result)
        assert float(norms[0][0]) > args.max_grad_norm
        torch.testing.assert_close(norms[0][0], norms[1][0], rtol=0, atol=0)
        assert_tensors_close(first.actor.parameters(), second.actor.parameters())
        if mode == "ppo":
            assert_tensors_close(first.critic.parameters(), second.critic.parameters())
        else:
            assert not torch.allclose(first.critic.first[0].weight, second.critic.first[0].weight, rtol=0, atol=1e-7)
    assert all(value.grad is None for value in (*data, *changed))


def test_detached_ppo_auxiliary_training_preserves_v17_actor_and_scalar_adam_trajectory(envs):
    agent = make_agent(envs, "ppo")
    prime_auxiliary(agent)
    torch.manual_seed(17)
    canonical = baseline.Agent(envs, baseline.Args(control_mode="ppo")).cuda()
    args = model.Args(aux_mode="ppo", ent_coef=0.013, clip_vloss=True)
    data = training_batch(agent)
    groups, canonical_groups = agent.parameter_groups(), canonical.parameter_groups()[:2]
    optimizers, canonical_optimizers = optimizers_for(groups, args), optimizers_for(canonical_groups, args)
    head_before = tuple(parameter.detach().clone() for parameter in groups[2])
    for _ in range(3):
        for optimizer in (*optimizers, *canonical_optimizers):
            optimizer.zero_grad(set_to_none=True)
        model.minibatch_objective(agent, args, *data)[0].backward()
        reference_ppo(canonical, args, data)[0].backward()
        assert_tensors_close(gradients(agent.actor.parameters()), gradients(canonical.actor.parameters()))
        assert_tensors_close(gradients(agent.critic.parameters()), gradients(canonical.critic.parameters()))
        for parameter in groups[2]:
            parameter.grad.mul_(1e6)
        model.optimizer_step(optimizers, groups, args.max_grad_norm, torch.empty(3, device="cuda"))
        for group in canonical_groups:
            torch.nn.utils.clip_grad_norm_(group, args.max_grad_norm)
        for optimizer in canonical_optimizers:
            optimizer.step()
        assert_tensors_close(agent.actor.parameters(), canonical.actor.parameters())
        assert_tensors_close(agent.critic.parameters(), canonical.critic.parameters())
    assert any(not torch.equal(after, before) for after, before in zip(groups[2], head_before, strict=True))
    assert all(value.grad is None for value in data)


@pytest.mark.parametrize("mode", MODES)
def test_repeated_fullgraph_objective_gradients_and_owned_adam_updates_match_eager(envs, mode):
    agent = make_agent(envs, mode)
    prime_auxiliary(agent)
    reference = deepcopy(agent)
    args = model.Args(aux_mode=mode, ent_coef=0.013, clip_vloss=True)
    data = training_batch(agent)
    groups, reference_groups = agent.parameter_groups(), reference.parameter_groups()
    optimizers, reference_optimizers = optimizers_for(groups, args), optimizers_for(reference_groups, args)
    initial = tuple(tuple(parameter.detach().clone() for parameter in group) for group in groups)

    def objective(*inputs):
        return model.minibatch_objective(agent, args, *inputs)

    compiled = torch.compile(objective, fullgraph=True, options={"triton.cudagraphs": False})
    # Reuse the compiled graph after a real update, including carried Adam moments.
    for _ in range(2):
        for optimizer in (*optimizers, *reference_optimizers):
            optimizer.zero_grad(set_to_none=True)
        expected, expected_metrics = model.minibatch_objective(reference, args, *data)
        expected.backward()
        actual, metrics = compiled(*data)
        torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-6)
        torch.testing.assert_close(metrics, expected_metrics, rtol=3e-4, atol=2e-5)
        actual.backward()
        assert_tensors_close(gradients(agent.parameters()), gradients(reference.parameters()), rtol=8e-4, atol=1e-5)
        expected_norms = torch.stack([torch.nn.utils.clip_grad_norm_(group, args.max_grad_norm)
                                      for group in reference_groups])
        for optimizer in reference_optimizers:
            optimizer.step()
        norms = torch.empty(3, device="cuda")
        model.optimizer_step(optimizers, groups, args.max_grad_norm, norms)
        torch.testing.assert_close(norms, expected_norms, rtol=8e-4, atol=1e-5)
        assert_tensors_close(agent.parameters(), reference.parameters(), rtol=8e-4, atol=1e-5)
    for group, before in zip(groups, initial, strict=True):
        assert any(not torch.equal(actual, previous) for actual, previous in zip(group, before, strict=True))
    assert all(value.grad is None for value in data)
