"""CUDA contracts for v16. Run through mlq; CPU is used only for quadrature nodes."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.special import roots_jacobi

from cleanrl import ppo_continuous_action_successor_multiscale_v15 as baseline
from cleanrl import ppo_continuous_action_successor_score_control_v16 as model
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-2.0, 3.0, (2,), dtype=np.float32),
    )


def make_agent(envs, mode="corrected"):
    torch.manual_seed(17)
    return model.Agent(envs, model.Args(control_mode=mode)).cuda()


def tensor(values, **kwargs):
    return torch.tensor(values, device="cuda", dtype=torch.float64, **kwargs)


def quadrature(alpha, beta, order=96):
    """Tensor-product Gauss-Jacobi measure, normalized without a beta-function oracle."""
    rules = [roots_jacobi(order, b - 1, a - 1) for a, b in zip(alpha, beta, strict=True)]
    nodes = np.stack(np.meshgrid(*[(x + 1) / 2 for x, _ in rules], indexing="ij"), -1)
    weights = np.prod(np.stack(np.meshgrid(*[w / w.sum() for _, w in rules], indexing="ij")), axis=0)
    return tensor(nodes.reshape(-1, len(alpha))), tensor(weights.reshape(-1))


def reference_moments(alpha, beta):
    total = torch.digamma(alpha + beta)
    return torch.cat((torch.digamma(alpha) - total, torch.digamma(beta) - total), -1)


def reference_fisher(alpha, beta):
    """Autograd Hessian of log partition; deliberately not the production block formula."""
    matrices = []
    for a, b in zip(alpha, beta, strict=True):
        concentrations = torch.cat((a, b)).detach().requires_grad_()

        def partition(value):
            first, second = value.chunk(2)
            return (torch.lgamma(first) + torch.lgamma(second) - torch.lgamma(first + second)).sum()

        matrices.append(torch.autograd.functional.hessian(partition, concentrations))
    return torch.stack(matrices).detach()


def actor_output_jacobians(actor, observations):
    """Explicit per-example/per-output gradients of all weights AND biases."""
    parameters = tuple(actor.parameters())
    rows = []
    for observation in observations.detach():
        output = actor(observation.unsqueeze(0))[0]
        gradients = []
        for coordinate in range(output.numel()):
            derivatives = torch.autograd.grad(output[coordinate], parameters, retain_graph=coordinate + 1 < output.numel())
            gradients.append(torch.cat([value.flatten() for value in derivatives]))
        rows.append(torch.stack(gradients))
    return torch.stack(rows).detach()


def reference_output_gradient(advantages, score, alpha, beta, coefficients):
    fisher = reference_fisher(alpha, beta)
    concentration = torch.cat((alpha, beta), -1)
    # Invert softplus and differentiate it, independently of the expm1 identity.
    logits = torch.log(torch.expm1(concentration - 1))
    jacobian = torch.sigmoid(logits)
    control = coefficients[:, 0] + (coefficients[:, 1:] * score).sum(-1)
    response = torch.einsum("bij,bj->bi", fisher, coefficients[:, 1:])
    return jacobian * (score * (advantages - control).unsqueeze(-1) + response)


def prime_control(agent):
    # Nonzero head exposes accidental trunk and actor/critic gradient leakage.
    with torch.no_grad():
        agent.control.head[0].weight.normal_(std=0.08)
        agent.control.head[0].bias.copy_(torch.linspace(-0.3, 0.4, 1 + 2 * agent.action_dim, device="cuda"))


def training_batch(agent, rows=8):
    observations = torch.randn(rows, 3, device="cuda", requires_grad=True)
    native = (0.15 + 0.7 * torch.rand(rows, agent.action_dim, device="cuda")).requires_grad_()
    old = model.rollout_statistics(agent, observations, native)
    # Live leaves, rather than already-detached inputs, catch missing ownership boundaries.
    old = {name: value.detach().clone().requires_grad_() for name, value in old.items()}
    advantages = torch.linspace(-2.1, 3.8, rows, device="cuda").requires_grad_()
    returns = (old["values"].detach() + torch.linspace(-1.5, 2.0, rows, device="cuda")).requires_grad_()
    return observations, native, advantages, returns, old


def objective(agent, data, args):
    observations, native, advantages, returns, old = data
    return model.policy_loss(agent, observations, native, advantages, returns, old, args)


def teacher_tensors(data):
    observations, native, advantages, returns, old = data
    return (observations, native, advantages, returns, *old.values())


def gradients(module):
    return tuple(parameter.grad.detach().clone() for parameter in module.parameters())


def assert_gradients(actual, expected, rtol=2e-5, atol=2e-7):
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=atol)


def test_beta_moments_fisher_and_expectation_derivative_match_log_partition_autograd():
    alpha = tensor([[1.03, 2.8], [7.2, 1.4], [3.6, 5.1]], requires_grad=True)
    beta = tensor([[4.2, 1.07], [1.2, 5.8], [2.2, 8.3]], requires_grad=True)
    partition = (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)).sum()
    expected_moments = torch.cat(torch.autograd.grad(partition, (alpha, beta)), -1)
    torch.testing.assert_close(model.beta_log_moments(alpha, beta), expected_moments, rtol=2e-12, atol=2e-12)
    fisher = reference_fisher(alpha, beta)
    vector = tensor([[0.8, -1.3, 0.4, 2.1], [-0.2, 0.5, 1.7, -0.3], [2.0, 0.1, -0.9, 0.6]])
    expected_product = torch.einsum("bij,bj->bi", fisher, vector)
    torch.testing.assert_close(model.beta_fisher_product(alpha, beta, vector), expected_product, rtol=2e-12, atol=2e-12)
    # A perturbation of alpha_0 must affect beta_0, not alpha_1 or beta_1.
    impulse = torch.zeros_like(vector)
    impulse[:, 0] = 1
    product = model.beta_fisher_product(alpha, beta, impulse)
    torch.testing.assert_close(product[:, [1, 3]], torch.zeros_like(product[:, [1, 3]]), rtol=0, atol=0)
    assert bool((product[:, 2] < 0).all())
    coefficients = torch.cat((tensor([[0.3], [-0.8], [1.2]]), vector), -1).requires_grad_()
    old_moments = reference_moments(alpha.detach() + 0.7, beta.detach() + 0.2).requires_grad_()
    expectation = model.control_expectation(coefficients, alpha, beta, old_moments)
    first, second, coefficient_grad, old_grad = torch.autograd.grad(
        expectation.sum(), (alpha, beta, coefficients, old_moments), allow_unused=True,
    )
    torch.testing.assert_close(torch.cat((first, second), -1), expected_product, rtol=3e-12, atol=3e-12)
    assert coefficient_grad is None and old_grad is None
    logits = torch.log(torch.expm1(torch.cat((alpha, beta), -1) - 1)).detach().requires_grad_()
    derivative, = torch.autograd.grad((F.softplus(logits) + 1).sum(), (logits,))
    torch.testing.assert_close(model.beta_logit_jacobian(alpha, beta), derivative, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("current", [([3.4, 4.1], [4.7, 3.2]), ([4.2, 3.5], [3.6, 4.3])])
def test_integrated_control_has_zero_population_value_and_gradient_under_policy_shift(current):
    native, weights = quadrature([3.4, 4.1], [4.7, 3.2])
    old_alpha, old_beta = tensor([[3.4, 4.1]]), tensor([[4.7, 3.2]])
    alpha, beta = tensor([current[0]], requires_grad=True), tensor([current[1]], requires_grad=True)
    coefficients = tensor([[0.7, -1.2, 0.4, 0.9, -0.6]])
    old_moments = reference_moments(old_alpha, old_beta)
    old_score = torch.cat((native.log(), torch.log1p(-native)), -1) - old_moments
    torch.testing.assert_close(model.beta_score(native, old_alpha, old_beta), old_score, rtol=2e-12, atol=2e-12)
    sampled = model.control_sample(coefficients, old_score)
    old_logprob = torch.distributions.Beta(old_alpha, old_beta).log_prob(native).sum(-1)
    new_logprob = torch.distributions.Beta(alpha, beta).log_prob(native).sum(-1)
    integrated = model.control_expectation(coefficients, alpha, beta, old_moments)
    residual = (weights * (new_logprob - old_logprob).exp() * sampled).sum() - integrated.sum()
    derivative = torch.autograd.grad(residual, (alpha, beta))
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=3e-7)
    for value in derivative:
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=2e-6)


def test_behavior_policy_corrected_loss_gradient_has_the_positive_objective_sign():
    logits = tensor([[0.2, -0.7, 0.8, 1.1], [1.3, 0.4, -0.1, -0.6]], requires_grad=True)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    native = tensor([[0.18, 0.72], [0.61, 0.28]])
    advantages = tensor([1.7, -0.9])
    coefficients = tensor([[0.3, -0.7, 0.8, 1.1, -0.2], [-0.4, 0.5, -0.3, 0.6, 0.9]])
    old_moments = reference_moments(alpha.detach(), beta.detach())
    score = torch.cat((native.log(), torch.log1p(-native)), -1) - old_moments
    sampled = coefficients[:, 0] + (coefficients[:, 1:] * score).sum(-1)
    logprob = torch.distributions.Beta(alpha, beta).log_prob(native).sum(-1)
    ratio = (logprob - logprob.detach()).exp()
    expectation = coefficients[:, 0] + (coefficients[:, 1:] * (reference_moments(alpha, beta) - old_moments)).sum(-1)
    loss = (-ratio * advantages + ratio * sampled - expectation).sum()
    loss_gradient, = torch.autograd.grad(loss, (logits,))
    actual = model.corrected_output_gradient(advantages, score, alpha, beta, coefficients)
    torch.testing.assert_close(actual, -loss_gradient, rtol=3e-12, atol=3e-12)


def test_actor_gram_and_gradient_moments_match_all_parameter_autograd():
    actor = model.TaskFFN(3, 4, 0.7).cuda().double()
    observations = tensor([[0.4, -0.7, 1.2], [-0.3, 0.6, 0.9], [1.1, -1.4, 0.2]], requires_grad=True)
    output_gradients = tensor([[0.2, -0.8, 1.1, 0.4], [1.3, 0.6, -0.2, -0.9], [-0.5, 0.1, 0.7, 1.4]], requires_grad=True)
    jacobians = actor_output_jacobians(actor, observations)
    expected_gram = jacobians @ jacobians.transpose(-1, -2)
    actual_gram = model.actor_jacobian_gram(actor, observations)
    torch.testing.assert_close(actual_gram, expected_gram, rtol=2e-11, atol=2e-11)
    per_example = torch.einsum("bcp,bc->bp", jacobians, output_gradients.detach())
    expected = per_example.square().sum(-1).mean(), per_example.mean(0).square().sum()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual = model.actor_gradient_moments(actor, observations, output_gradients)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-11, atol=2e-11)
        assert not value.requires_grad
    assert not actual_gram.requires_grad
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert observations.grad is None and output_gradients.grad is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_compatible_advantage_removes_action_variance_without_changing_mean_gradient():
    native, weights = quadrature([3.7], [4.4], order=160)
    alpha, beta = tensor([[3.7]]), tensor([[4.4]])
    coefficients = tensor([[0.6, -1.1, 0.8]])
    score = model.beta_score(native, alpha, beta)
    advantages = coefficients[:, 0] + (coefficients[:, 1:] * score).sum(-1)
    corrected = model.corrected_output_gradient(advantages, score, alpha, beta, coefficients)
    expected = model.beta_logit_jacobian(alpha, beta) * model.beta_fisher_product(alpha, beta, coefficients[:, 1:])
    torch.testing.assert_close(corrected, expected.expand_as(corrected), rtol=3e-12, atol=3e-12)
    base = model.beta_logit_jacobian(alpha, beta) * score * advantages[:, None]
    mean_base = (weights[:, None] * base).sum(0)
    mean_corrected = (weights[:, None] * corrected).sum(0)
    torch.testing.assert_close(mean_base, mean_corrected, rtol=2e-6, atol=2e-7)
    # Compare actor parameter gradients as well: this is not just output-space variance.
    actor = model.TaskFFN(3, 2, 0.6).cuda().double()
    jacobian = actor_output_jacobians(actor, tensor([[0.2, -0.7, 0.9]]))[0]
    centered = (corrected - mean_corrected) @ jacobian
    base_centered = (base - mean_base) @ jacobian
    torch.testing.assert_close((weights[:, None] * centered.square()).sum(), tensor(0.0), rtol=0, atol=1e-20)
    assert float((weights[:, None] * base_centered.square()).sum()) > 0.1


def test_variance_loss_matches_explicit_parameter_gradient_second_moment_and_detaches_teachers():
    actor = model.TaskFFN(3, 4, 0.7).cuda().double()
    observations = tensor([[0.1, -0.4, 0.8], [0.9, 0.6, -0.3], [-0.7, 1.1, 0.2]])
    jacobians = actor_output_jacobians(actor, observations)
    gram = (jacobians @ jacobians.transpose(-1, -2)).requires_grad_()
    alpha = tensor([[1.4, 2.1], [3.2, 1.3], [2.4, 4.1]], requires_grad=True)
    beta = tensor([[2.3, 1.7], [1.8, 3.3], [4.2, 1.6]], requires_grad=True)
    score = tensor([[-0.5, 0.2, 0.9, -0.6], [0.8, -0.4, -0.2, 1.1], [0.4, 0.7, -0.9, -0.3]], requires_grad=True)
    advantages = tensor([1.8, -0.6, 2.7], requires_grad=True)
    coefficients = tensor([[0.2, -0.8, 0.3, 0.9, -0.4], [-0.5, 0.7, -0.6, 0.2, 1.0], [0.8, -0.2, 0.5, -0.7, 0.4]], requires_grad=True)
    parameter_count = jacobians.shape[-1]
    reference_g = reference_output_gradient(advantages.detach(), score.detach(), alpha.detach(), beta.detach(), coefficients)
    parameter_gradients = torch.einsum("bcp,bc->bp", jacobians, reference_g)
    reference_loss = 0.5 * parameter_gradients.square().sum(-1).mean() / parameter_count
    expected_gradient, = torch.autograd.grad(reference_loss, (coefficients,))
    actual = model.control_variance_loss(coefficients, advantages, score, alpha, beta, gram, parameter_count)
    torch.testing.assert_close(actual, reference_loss, rtol=3e-11, atol=3e-12)
    actual.backward()
    torch.testing.assert_close(coefficients.grad, expected_gradient, rtol=3e-11, atol=3e-12)
    assert all(value.grad is None for value in (advantages, score, alpha, beta, gram))
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_all_arms_preserve_scalar_initialization_sampling_rng_and_zero_head(envs):
    torch.manual_seed(17)
    scalar = baseline.Agent(envs, baseline.Args(critic_mode="scalar")).cuda()
    states = torch.get_rng_state(), torch.cuda.get_rng_state()
    agents = []
    for mode in ("ppo", "baseline", "corrected"):
        agent = make_agent(envs, mode)
        agents.append(agent)
        assert torch.equal(states[0], torch.get_rng_state())
        assert torch.equal(states[1], torch.cuda.get_rng_state())
        for owner in ("actor", "critic"):
            for actual, expected in zip(getattr(agent, owner).parameters(), getattr(scalar, owner).parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    observations = torch.randn(5, 3, device="cuda")
    sample_state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(sample_state)
    expected_action = scalar.get_action_and_value(observations)
    for agent in agents:
        torch.testing.assert_close(agent.control(observations), torch.zeros(5, 5, device="cuda"), rtol=0, atol=0)
        torch.cuda.set_rng_state(sample_state)
        for actual, expected in zip(agent.get_action_and_value(observations), expected_action, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        groups = agent.parameter_groups()
        identities = [set(map(id, group)) for group in groups]
        assert len(groups) == 3
        assert all(identities[i].isdisjoint(identities[j]) for i in range(3) for j in range(i))
        assert set.union(*identities) == set(map(id, agent.parameters()))
        for actual, owner in zip(identities, (agent.actor, agent.critic, agent.control), strict=True):
            assert actual == set(map(id, owner.parameters()))


@pytest.mark.parametrize("mode", ["ppo", "baseline", "corrected"])
def test_zero_head_or_disabled_correction_matches_scalar_ppo_clipped_adam_updates(envs, mode):
    agent = make_agent(envs, mode)
    if mode == "ppo":
        prime_control(agent)
    torch.manual_seed(17)
    scalar = baseline.Agent(envs, baseline.Args(critic_mode="scalar")).cuda()
    data = training_batch(agent)
    observations, native, advantages, returns, old = data
    args = model.Args(control_mode=mode, clip_vloss=True, ent_coef=0.013)
    baseline_args = baseline.Args(critic_mode="scalar", clip_vloss=True, ent_coef=args.ent_coef)
    groups, reference_groups = agent.parameter_groups(), scalar.parameter_groups()
    optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
    reference_optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in reference_groups)
    for _ in range(3):
        for optimizer in (*optimizers, *reference_optimizers):
            optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(agent, data, args)
        reference, _ = baseline.policy_loss(scalar, observations, native, old["logprobs"], advantages,
                                             returns[:, None], old["values"], baseline_args,
                                             torch.ones(1, device="cuda"))
        loss.backward()
        reference.backward()
        assert_gradients(gradients(agent.actor), gradients(scalar.actor), rtol=0, atol=0)
        assert_gradients(gradients(agent.critic), gradients(scalar.critic), rtol=0, atol=0)
        # Huge auxiliary gradients must not enter the actor or critic clipping norm.
        for parameter in agent.control.parameters():
            parameter.grad.mul_(1e6)
        model.optimizer_step(optimizers, groups, args.max_grad_norm, torch.empty(3, device="cuda"))
        baseline.optimizer_step(reference_optimizers, reference_groups, args.max_grad_norm, torch.empty(2, device="cuda"))
        for actual, expected in zip((*agent.actor.parameters(), *agent.critic.parameters()), scalar.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(value.grad is None for value in teacher_tensors(data))


@pytest.mark.parametrize("mode", ["baseline", "corrected"])
@pytest.mark.parametrize("norm_adv", [False, True])
def test_actual_control_fit_reaches_only_control_and_joint_objective_respects_all_owners(envs, norm_adv, mode):
    agent = make_agent(envs, mode)
    prime_control(agent)
    data = training_batch(agent)
    observations, native, advantages, returns, old = data
    args = model.Args(control_mode=mode, norm_adv=norm_adv, clip_vloss=False, ent_coef=0.017)
    reference = deepcopy(agent)
    alpha, beta, value = reference.get_policy_and_value(observations.detach())
    distribution = torch.distributions.Beta(alpha, beta)
    logprob = (distribution.log_prob(native.detach()) - reference.log_action_scale).sum(-1)
    ratio = (logprob - old["logprobs"].detach()).exp()
    normalized = advantages.detach()
    if args.norm_adv:
        normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-8)
    pg = torch.maximum(-normalized * ratio, -normalized * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    frozen_coefficients = old["coefficients"].detach()
    sample = frozen_coefficients[:, 0] + (frozen_coefficients[:, 1:] * old["score"].detach()).sum(-1)
    expected = frozen_coefficients[:, 0] + (frozen_coefficients[:, 1:] * (reference_moments(alpha, beta) - old["log_moments"].detach())).sum(-1)
    entropy = (distribution.entropy() + reference.log_action_scale).sum(-1).mean()
    correction = (ratio * sample - expected).mean()
    if mode == "baseline":
        correction = ((ratio - 1) * frozen_coefficients[:, 0]).mean()
    actor_loss = pg - args.ent_coef * entropy + correction
    actor_loss.backward()
    critic_loss = args.vf_coef * 0.5 * (value.flatten() - returns.detach()).square().mean()
    critic_loss.backward()
    current = reference.control(observations.detach())
    jacobians = actor_output_jacobians(reference.actor, observations)
    g = reference_output_gradient(normalized, old["score"].detach(), old["alpha"].detach(), old["beta"].detach(), current)
    full_gradient = torch.einsum("bcp,bc->bp", jacobians, g)
    control_loss = 0.5 * full_gradient.square().sum(-1).mean() / jacobians.shape[-1]
    before_actor, before_critic = gradients(reference.actor), gradients(reference.critic)
    control_loss.backward()
    assert_gradients(gradients(reference.actor), before_actor, rtol=0, atol=0)
    assert_gradients(gradients(reference.critic), before_critic, rtol=0, atol=0)
    actual, _ = objective(agent, data, args)
    actual.backward()
    torch.testing.assert_close(actual, actor_loss + critic_loss + control_loss, rtol=2e-5, atol=2e-6)
    for owner in ("actor", "critic", "control"):
        assert_gradients(gradients(getattr(agent, owner)), gradients(getattr(reference, owner)), rtol=8e-5, atol=2e-6)
    assert all(value.grad is None for value in teacher_tensors(data))


def test_prefit_coefficients_remain_frozen_and_correction_is_independent_of_current_advantages(envs):
    agent = make_agent(envs)
    prime_control(agent)
    data = training_batch(agent)
    data = (*data[:4], model.rollout_statistics(agent, data[0], data[1]))
    assert all(not value.requires_grad for value in data[4].values())
    observations, _, _, _, old = data
    snapshots = {name: value.detach().clone() for name, value in old.items()}
    diagnose = torch.compile(
        lambda obs, advantages, snapshot: model.control_diagnostics(agent, obs, advantages, snapshot, exact=True),
        fullgraph=True, options={"triton.cudagraphs": False},
    )
    prefit_diagnostics = diagnose(observations, data[2], old)
    corrected = model.Args(control_mode="corrected")
    off = model.Args(control_mode="ppo")
    old_loss, _ = objective(agent, data, corrected)
    old_loss.backward()
    expected_actor = gradients(agent.actor)
    control_optimizer = torch.optim.Adam(agent.control.parameters(), lr=0.03, eps=1e-5, fused=True)
    torch.nn.utils.clip_grad_norm_(agent.control.parameters(), 0.5)
    control_optimizer.step()
    agent.zero_grad(set_to_none=True)
    assert not torch.allclose(agent.control(observations.detach()), snapshots["coefficients"], rtol=1e-5, atol=1e-6)
    frozen_diagnostics = diagnose(observations, data[2], old)
    for name, expected in prefit_diagnostics.items():
        torch.testing.assert_close(frozen_diagnostics[name], expected, rtol=0, atol=0)
        assert not frozen_diagnostics[name].requires_grad
    refit_loss, _ = objective(agent, data, corrected)
    refit_loss.backward()
    assert_gradients(gradients(agent.actor), expected_actor, rtol=0, atol=0)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(old[name], snapshot, rtol=0, atol=0)
    # Move the actor off behavior so the correction has nonzero value as well as derivative.
    with torch.no_grad():
        agent.actor.head[0].bias.add_(torch.tensor([0.4, -0.2, 0.3, -0.5], device="cuda"))
    corrected_loss, _ = objective(agent, data, corrected)
    off_loss, _ = objective(agent, data, off)
    amplified_advantages = (data[0], data[1], data[2] * 11 + 23, data[3], data[4])
    amplified_on, _ = objective(agent, amplified_advantages, corrected)
    amplified_off, _ = objective(agent, amplified_advantages, off)
    torch.testing.assert_close(amplified_on - amplified_off, corrected_loss - off_loss, rtol=3e-4, atol=3e-6)
    assert all(value.grad is None for value in teacher_tensors(data))


def test_compiled_analytic_helpers_and_coefficient_gradients_match_eager():
    actor = model.TaskFFN(3, 4, 0.4).cuda()
    observations = torch.randn(7, 3, device="cuda")
    alpha = torch.linspace(1.2, 4.5, 14, device="cuda").reshape(7, 2).requires_grad_()
    beta = torch.linspace(3.7, 1.1, 14, device="cuda").reshape(7, 2).requires_grad_()
    native = torch.linspace(0.07, 0.93, 14, device="cuda").reshape(7, 2)
    advantages = torch.linspace(-2.0, 2.8, 7, device="cuda")
    coefficients = torch.randn(7, 5, device="cuda", requires_grad=True)
    old_moments = reference_moments(alpha.detach() + 0.2, beta.detach() + 0.3)
    parameter_count = sum(parameter.numel() for parameter in actor.parameters())

    def kernel(a, b, coeff):
        score = model.beta_score(native, a, b)
        gram = model.actor_jacobian_gram(actor, observations)
        g = model.corrected_output_gradient(advantages, score, a, b, coeff)
        moments = model.actor_gradient_moments(actor, observations, g)
        expectation = model.control_expectation(coeff, a, b, old_moments)
        loss = model.control_variance_loss(coeff, advantages, score, a, b, gram, parameter_count)
        return (loss + expectation.mean(), score, gram, g, *moments,
                model.beta_fisher_product(a, b, coeff[:, 1:]), model.beta_logit_jacobian(a, b))

    eager = kernel(alpha, beta, coefficients)
    expected_gradients = torch.autograd.grad(eager[0], (alpha, beta, coefficients))
    compiled = torch.compile(kernel, fullgraph=True, options={"triton.cudagraphs": False})
    actual = compiled(alpha, beta, coefficients)
    actual_gradients = torch.autograd.grad(actual[0], (alpha, beta, coefficients))
    for first, second in zip(actual, eager, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-4, atol=1e-5)
    assert_gradients(actual_gradients, expected_gradients, rtol=3e-4, atol=1e-5)
    assert all(parameter.grad is None for parameter in actor.parameters())


@pytest.mark.parametrize("mode", ["ppo", "baseline", "corrected"])
def test_compiled_objective_gradients_and_exclusive_clipped_adam_updates_match_eager(envs, mode):
    agent = make_agent(envs, mode)
    prime_control(agent)
    reference = deepcopy(agent)
    args = model.Args(control_mode=mode, ent_coef=0.01)
    data = training_batch(agent)
    # Exercise nontrivial ratios and value clipping, not only the behavior-policy point.
    with torch.no_grad():
        agent.actor.head[0].bias.add_(torch.tensor([0.3, -0.4, 0.2, -0.1], device="cuda"))
        agent.critic.head[0].bias.add_(0.5)
        reference.load_state_dict(agent.state_dict())
    expected, expected_metrics = objective(reference, data, args)
    expected.backward()

    def loss(observations, native, advantages, returns, old):
        return model.policy_loss(agent, observations, native, advantages, returns, old, args)

    compiled = torch.compile(loss, fullgraph=True, options={"triton.cudagraphs": False})
    actual, actual_metrics = compiled(*data)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    assert actual_metrics.keys() == expected_metrics.keys()
    for name in expected_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=2e-4, atol=1e-5)
    actual.backward()
    for owner in ("actor", "critic", "control"):
        assert_gradients(gradients(getattr(agent, owner)), gradients(getattr(reference, owner)), rtol=4e-4, atol=5e-6)
    groups, reference_groups = agent.parameter_groups(), reference.parameter_groups()
    optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
    reference_optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in reference_groups)
    expected_norms = torch.stack([torch.nn.utils.clip_grad_norm_(group, args.max_grad_norm) for group in reference_groups])
    for optimizer in reference_optimizers:
        optimizer.step()
    norms = torch.empty(3, device="cuda")
    model.optimizer_step(optimizers, groups, args.max_grad_norm, norms)
    torch.testing.assert_close(norms, expected_norms, rtol=4e-4, atol=5e-6)
    for actual_parameter, expected_parameter in zip(agent.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=5e-4, atol=5e-6)
    assert all(value.grad is None for value in teacher_tensors(data))
