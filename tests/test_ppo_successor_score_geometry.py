"""CUDA contracts for v17; execute through mlq, with CPU only for quadrature nodes."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from scipy.special import roots_jacobi

from cleanrl import ppo_continuous_action_successor_multiscale_v15 as baseline
from cleanrl import ppo_continuous_action_successor_score_control_v16 as sampled_reference
from cleanrl import ppo_continuous_action_successor_score_geometry_v17 as model
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


def tensor(values, **kwargs):
    return torch.tensor(values, device="cuda", dtype=torch.float64, **kwargs)


def quadrature(alpha, beta, order=128):
    """Independent tensor-product Beta measure, not samples or production moments."""
    rules = [roots_jacobi(order, b - 1, a - 1) for a, b in zip(alpha, beta, strict=True)]
    nodes = np.stack(np.meshgrid(*[(x + 1) / 2 for x, _ in rules], indexing="ij"), -1)
    weights = np.prod(np.stack(np.meshgrid(*[w / w.sum() for _, w in rules], indexing="ij")), axis=0)
    return tensor(nodes.reshape(-1, len(alpha))), tensor(weights.reshape(-1))


def reference_moments(alpha, beta):
    common = torch.digamma(alpha + beta)
    return torch.cat((torch.digamma(alpha) - common, torch.digamma(beta) - common), -1)


def reference_fisher(alpha, beta):
    """Log-partition Hessian instead of the production polygamma block assembly."""
    matrices = []
    for a, b in zip(alpha, beta, strict=True):
        concentrations = torch.cat((a, b)).detach().requires_grad_()

        def partition(value):
            first, second = value.chunk(2)
            return (torch.lgamma(first) + torch.lgamma(second) - torch.lgamma(first + second)).sum()

        matrices.append(torch.autograd.functional.hessian(partition, concentrations))
    return torch.stack(matrices).detach()


def reference_basis(fisher):
    """Dense Cholesky inverse in alpha-all, beta-all order; no packed-whitening helper."""
    size = fisher.shape[-1]
    identity = torch.eye(size, device="cuda", dtype=fisher.dtype).expand_as(fisher)
    inverse = torch.linalg.solve_triangular(torch.linalg.cholesky(fisher), identity, upper=False)
    basis = torch.zeros(fisher.shape[0], size + 1, size + 1, device="cuda", dtype=fisher.dtype)
    basis[:, 0, 0] = 1
    basis[:, 1:, 1:] = inverse.transpose(-1, -2)
    return basis


def reference_jacobian(alpha, beta):
    concentrations = torch.cat((alpha, beta), -1)
    return torch.sigmoid(torch.log(torch.expm1(concentrations - 1)))


@pytest.fixture
def population():
    alpha, beta = tensor([[3.4, 5.7]]), tensor([[4.6, 3.2]])
    native, weights = quadrature([3.4, 5.7], [4.6, 3.2])
    score = torch.cat((native.log(), torch.log1p(-native)), -1) - reference_moments(alpha, beta)
    fisher = reference_fisher(alpha, beta)
    basis = reference_basis(fisher)
    # The metric couples different actions and the alpha/beta coordinates of each action.
    root = tensor([[1.8, 0.4, -0.3, 0.7], [0.2, 1.1, 0.6, -0.5],
                   [-0.4, 0.3, 1.5, 0.2], [0.6, -0.2, 0.4, 1.3]])
    gram = (root @ root.T).unsqueeze(0)
    jacobian = reference_jacobian(alpha, beta)
    features = torch.cat((score.unsqueeze(-1), score.unsqueeze(-1) * score.unsqueeze(-2) - fisher), -1)
    design = jacobian.unsqueeze(-1) * features
    white_design = design @ basis
    white_score = score @ basis[0, 1:, 1:]
    return SimpleNamespace(alpha=alpha, beta=beta, native=native, weights=weights, score=score,
                           fisher=fisher, basis=basis, gram=gram, jacobian=jacobian,
                           design=design, white_design=white_design, white_score=white_score)


def integrated_loss(coefficients, advantages, normal, eligibility, score_power, weights, parameter_count):
    # The public objective is a minibatch mean: multiply each teacher by N*w to integrate it.
    mass = weights * weights.numel()
    return model.analytic_control_loss(
        coefficients.expand(weights.numel(), -1), advantages,
        normal * mass[:, None, None], eligibility * mass[:, None], score_power * mass,
        parameter_count,
    )


def test_population_normal_matrix_matches_direct_design_integral_including_cross_action_cumulants(population):
    p = population
    whitening, actual = model.beta_control_geometry(p.alpha, p.beta, p.gram)
    expected = torch.einsum("n,nki,kl,nlj->ij", p.weights, p.white_design, p.gram[0], p.white_design)
    torch.testing.assert_close(actual[0], expected, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(actual, actual.transpose(-1, -2), rtol=0, atol=2e-12)

    inverse_basis = torch.linalg.inv(p.basis[0])
    natural = inverse_basis.T @ actual[0] @ inverse_basis
    natural_integral = torch.einsum("n,nki,kl,nlj->ij", p.weights, p.design, p.gram[0], p.design)
    metric = p.jacobian[0, :, None] * p.gram[0] * p.jacobian[0, None, :]
    fisher = p.fisher[0]
    power = torch.einsum("ni,ij,nj->n", p.score, metric, p.score)
    third_integral = torch.einsum("n,n,ni->i", p.weights, power, p.score)
    torch.testing.assert_close(natural[0, 1:], third_integral, rtol=3e-6, atol=3e-7)
    gaussian_part = torch.trace(metric @ fisher) * fisher + fisher @ metric @ fisher
    fourth = natural[1:, 1:] - gaussian_part
    fourth_integral = natural_integral[1:, 1:] - gaussian_part
    torch.testing.assert_close(fourth, fourth_integral, rtol=3e-6, atol=3e-7)
    # Independence removes fourth cumulants across actions, NOT the entire normal block.
    cross_action = tensor([[False, True, False, True], [True, False, True, False],
                           [False, True, False, True], [True, False, True, False]]).bool()
    torch.testing.assert_close(fourth[cross_action], torch.zeros_like(fourth[cross_action]), rtol=0, atol=2e-12)
    assert float(natural_integral[1:, 1:][cross_action].abs().max()) > 1e-3
    assert float(fourth_integral[0, 2]) < -1e-4
    assert float(third_integral[0]) < -1e-3

    coefficients = tensor([[0.6, -0.8, 0.3, 0.9, -0.4]])
    natural_coefficients = model.natural_coefficients(coefficients, whitening)
    expected_coefficients = (p.basis @ coefficients.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(natural_coefficients, expected_coefficients, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(p.white_score.T @ (p.weights[:, None] * p.white_score),
                               torch.eye(4, device="cuda", dtype=torch.float64), rtol=2e-6, atol=2e-7)
    white_control = coefficients[:, 0] + (coefficients[:, 1:] * p.white_score).sum(-1)
    torch.testing.assert_close(model.control_sample(natural_coefficients, p.score), white_control,
                               rtol=2e-12, atol=2e-12)


def test_whitened_control_analytic_expectation_and_policy_derivative_match_shifted_quadrature(population):
    p = population
    whitening, _ = model.beta_control_geometry(p.alpha, p.beta, p.gram)
    white = tensor([[0.7, -1.2, 0.4, 0.9, -0.6]], requires_grad=True)
    natural = model.natural_coefficients(white, whitening)
    alpha, beta = tensor([[4.2, 4.6]], requires_grad=True), tensor([[3.8, 4.4]], requires_grad=True)
    old_moments = reference_moments(p.alpha, p.beta).requires_grad_()
    control = (white[:, 0] + (white[:, 1:] * p.white_score).sum(-1)).detach()
    old_logprob = torch.distributions.Beta(p.alpha, p.beta).log_prob(p.native).sum(-1)
    new_logprob = torch.distributions.Beta(alpha, beta).log_prob(p.native).sum(-1)
    integral = (p.weights * (new_logprob - old_logprob).exp() * control).sum()
    expectation = model.control_expectation(natural, alpha, beta, old_moments).sum()
    torch.testing.assert_close(expectation, integral, rtol=3e-6, atol=3e-7)
    expected = torch.autograd.grad(integral, (alpha, beta))
    actual = torch.autograd.grad(expectation, (alpha, beta, white, old_moments), allow_unused=True)
    for first, second in zip(actual[:2], expected, strict=True):
        torch.testing.assert_close(first, second, rtol=8e-6, atol=2e-6)
    assert actual[2:] == (None, None)


def test_linear_features_and_analytic_objective_gradient_match_integrated_sampled_second_moment(population):
    p = population
    whitening, normal = model.beta_control_geometry(p.alpha, p.beta, p.gram)
    power, eligibility = model.control_linear_features(p.score, p.alpha, p.beta, p.gram, whitening)
    base = p.jacobian * p.score
    expected_power = torch.einsum("ni,ij,nj->n", base, p.gram[0], base)
    expected_eligibility = torch.einsum("nki,kl,nl->ni", p.white_design, p.gram[0], base)
    torch.testing.assert_close(power, expected_power, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(eligibility, expected_eligibility, rtol=3e-12, atol=3e-12)
    # Deliberately outside the compatible linear-score span.
    advantages = (0.4 + 1.7 * p.native[:, 0].square() - 0.8 * p.native[:, 1]
                  + torch.sin(4 * p.native[:, 0] * p.native[:, 1]))
    coefficients = tensor([[0.2, -0.7, 0.8, 0.3, -0.4]], requires_grad=True)
    parameter_count = 37
    gradient = base * advantages[:, None] - (p.white_design @ coefficients[0])
    expected = 0.5 * torch.einsum("n,ni,ij,nj->", p.weights, gradient, p.gram[0], gradient) / parameter_count
    actual = integrated_loss(coefficients, advantages, normal, eligibility, power, p.weights, parameter_count)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-8)
    expected_gradient, = torch.autograd.grad(expected, (coefficients,))
    actual_gradient, = torch.autograd.grad(actual, (coefficients,))
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-6, atol=3e-8)


def test_planted_compatible_advantage_is_stationary_and_removes_action_variance(population):
    p = population
    whitening, normal = model.beta_control_geometry(p.alpha, p.beta, p.gram)
    power, eligibility = model.control_linear_features(p.score, p.alpha, p.beta, p.gram, whitening)
    planted = tensor([[0.6, -1.1, 0.5, 0.8, -0.3]], requires_grad=True)
    advantages = (planted[:, 0] + (planted[:, 1:] * p.white_score).sum(-1)).detach()
    actual = integrated_loss(planted, advantages, normal, eligibility, power, p.weights, 37)
    derivative, = torch.autograd.grad(actual, (planted,))
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), rtol=0, atol=3e-8)
    natural = (p.basis @ planted.detach().unsqueeze(-1)).squeeze(-1)
    gradient = p.jacobian * p.score * advantages[:, None] - p.white_design @ planted.detach()[0]
    expected = p.jacobian * (p.fisher @ natural[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(gradient, expected.expand_as(gradient), rtol=3e-11, atol=3e-11)
    base = p.jacobian * p.score * advantages[:, None]
    torch.testing.assert_close((p.weights[:, None] * base).sum(0), expected[0], rtol=3e-6, atol=3e-7)
    assert float(torch.linalg.eigvalsh(normal[0]).min()) > 0


def test_rare_action_analytic_estimate_may_be_negative_without_losing_its_gradient():
    alpha, beta, gram = tensor([[3.4]]), tensor([[4.7]]), torch.eye(2, device="cuda", dtype=torch.float64)[None]
    whitening, normal = model.beta_control_geometry(alpha, beta, gram)
    native = tensor([[0.001]])
    score = torch.cat((native.log(), torch.log1p(-native)), -1) - reference_moments(alpha, beta)
    power, eligibility = model.control_linear_features(score, alpha, beta, gram, whitening)
    coefficients = (0.5 * torch.linalg.solve(normal, eligibility.unsqueeze(-1)).squeeze(-1)).requires_grad_()
    actual = model.analytic_control_loss(coefficients, tensor([1.0]), normal, eligibility, power, 11)
    expected = (0.5 * power - (coefficients * eligibility).sum(-1)
                + 0.5 * torch.einsum("bi,bij,bj->b", coefficients, normal, coefficients)).mean() / 11
    assert float(expected) < -1.0
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    actual.backward()
    expected_gradient = ((normal @ coefficients.detach().unsqueeze(-1)).squeeze(-1) - eligibility) / 11
    torch.testing.assert_close(coefficients.grad, expected_gradient, rtol=2e-12, atol=2e-12)
    assert float(coefficients.grad.norm()) > 0.1


def test_float64_geometry_is_cast_after_whitening_and_every_teacher_is_detached():
    alpha = torch.tensor([[1.0001, 128.0], [16384.0, 2.0]], device="cuda", requires_grad=True)
    beta = torch.tensor([[32.0, 1.0002], [8192.0, 4096.0]], device="cuda", requires_grad=True)
    root = torch.tensor([[2.0, 0.3, -0.7, 0.5], [0.1, 1.4, 0.6, -0.4],
                         [0.8, -0.5, 1.1, 0.2], [-0.3, 0.4, 0.7, 1.5]], device="cuda")
    gram = (root @ root.T).expand(2, -1, -1).clone().requires_grad_()
    whitening, normal = model.beta_control_geometry(alpha, beta, gram)
    expected_whitening, expected_normal = model.beta_control_geometry(alpha.double(), beta.double(), gram.double())
    for actual, expected in ((whitening, expected_whitening), (normal, expected_normal)):
        assert actual.dtype == torch.float32 and actual.device.type == "cuda"
        assert torch.isfinite(actual).all() and not actual.requires_grad
        torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=1e-7)
    score = torch.tensor([[-0.4, 0.1, 0.9, -0.8], [0.3, -0.7, -0.5, 0.2]], device="cuda", requires_grad=True)
    live_whitening = whitening.clone().requires_grad_()
    power, eligibility = model.control_linear_features(score, alpha, beta, gram, live_whitening)
    assert not power.requires_grad and not eligibility.requires_grad
    coefficients = torch.randn(2, 5, device="cuda", requires_grad=True)
    natural = model.natural_coefficients(coefficients, live_whitening)
    basis = reference_basis(reference_fisher(alpha.double(), beta.double())).float()
    expected_natural = (basis @ coefficients.unsqueeze(-1)).squeeze(-1)
    expected_derivative, = torch.autograd.grad(expected_natural.square().sum(), (coefficients,))
    natural.square().sum().backward()
    torch.testing.assert_close(coefficients.grad, expected_derivative, rtol=3e-5, atol=3e-5)
    assert live_whitening.grad is None

    coefficients.grad = None
    live_normal, live_eligibility, live_power = (value.clone().requires_grad_() for value in (normal, eligibility, power))
    advantages = torch.tensor([-0.8, 1.4], device="cuda", requires_grad=True)
    actual = model.analytic_control_loss(coefficients, advantages, live_normal, live_eligibility, live_power, 17)
    expected_derivative = ((normal @ coefficients.detach().unsqueeze(-1)).squeeze(-1)
                           - advantages.detach()[:, None] * eligibility) / (17 * coefficients.shape[0])
    actual.backward()
    torch.testing.assert_close(coefficients.grad, expected_derivative, rtol=3e-5, atol=1e-6)
    assert all(value.grad is None for value in (alpha, beta, gram, score, live_whitening,
                                               live_normal, live_eligibility, live_power, advantages))


def make_agent(envs, mode="analytic"):
    torch.manual_seed(17)
    return model.Agent(envs, model.Args(control_mode=mode)).cuda()


def prime_control(agent):
    with torch.no_grad():
        agent.control.head[0].weight.normal_(std=0.08)
        agent.control.head[0].bias.copy_(torch.linspace(-0.3, 0.4, 1 + 2 * agent.action_dim, device="cuda"))


def training_batch(agent, rows=8):
    observations = torch.randn(rows, 3, device="cuda", requires_grad=True)
    native = (0.15 + 0.7 * torch.rand(rows, agent.action_dim, device="cuda")).requires_grad_()
    old = model.rollout_statistics(agent, observations, native)
    # Live teacher leaves expose missing detach boundaries that no-grad snapshots would hide.
    old = {name: value.clone().requires_grad_() for name, value in old.items()}
    advantages = torch.linspace(-2.1, 3.8, rows, device="cuda").requires_grad_()
    returns = (old["values"].detach() + torch.linspace(-1.5, 2.0, rows, device="cuda")).requires_grad_()
    return observations, native, advantages, returns, old


def objective(agent, data, args):
    return model.policy_loss(agent, *data, args)


def teachers(data):
    return (*data[:4], *data[4].values())


def gradients(module):
    return tuple(parameter.grad.detach().clone() for parameter in module.parameters())


def assert_gradients(actual, expected, rtol=3e-5, atol=2e-6):
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=atol)


def test_all_modes_preserve_scalar_initialization_sampling_rng_and_zero_control(envs):
    torch.manual_seed(17)
    scalar = baseline.Agent(envs, baseline.Args(critic_mode="scalar")).cuda()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    agents = []
    for mode in ("ppo", "sampled", "analytic"):
        agents.append(make_agent(envs, mode))
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    observations = torch.randn(5, 3, device="cuda")
    sample_rng = torch.cuda.get_rng_state()
    expected_action = scalar.get_action_and_value(observations)
    for agent in agents:
        for owner in ("actor", "critic"):
            for actual, expected in zip(getattr(agent, owner).parameters(), getattr(scalar, owner).parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        old = model.rollout_statistics(agent, observations, torch.full((5, 2), 0.4, device="cuda"))
        torch.testing.assert_close(old["control_samples"], torch.zeros(5, device="cuda"), rtol=0, atol=0)
        torch.cuda.set_rng_state(sample_rng)
        for actual, expected in zip(agent.get_action_and_value(observations), expected_action, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        groups = [set(map(id, group)) for group in agent.parameter_groups()]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
        assert set.union(*groups) == set(map(id, agent.parameters()))


@pytest.mark.parametrize("mode", ["ppo", "sampled", "analytic"])
def test_zero_head_or_disabled_control_matches_scalar_ppo_adam_updates(envs, mode):
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
                                             returns[:, None], old["values"], baseline_args, torch.ones(1, device="cuda"))
        loss.backward()
        reference.backward()
        assert_gradients(gradients(agent.actor), gradients(scalar.actor), rtol=0, atol=0)
        assert_gradients(gradients(agent.critic), gradients(scalar.critic), rtol=0, atol=0)
        # Auxiliary gradients cannot change either policy/value clipping radius.
        for parameter in agent.control.parameters():
            parameter.grad.mul_(1e6)
        model.optimizer_step(optimizers, groups, args.max_grad_norm, torch.empty(3, device="cuda"))
        baseline.optimizer_step(reference_optimizers, reference_groups, args.max_grad_norm, torch.empty(2, device="cuda"))
        for actual, expected in zip((*agent.actor.parameters(), *agent.critic.parameters()), scalar.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(value.grad is None for value in teachers(data))


@pytest.mark.parametrize("mode", ["ppo", "sampled", "analytic"])
@pytest.mark.parametrize("norm_adv", [False, True])
def test_joint_objective_uses_ppo_advantage_units_and_exclusive_gradient_owners(envs, mode, norm_adv):
    agent = make_agent(envs, mode)
    prime_control(agent)
    data = training_batch(agent)
    observations, native, advantages, returns, old = data
    with torch.no_grad():
        agent.actor.head[0].bias.add_(torch.tensor([0.3, -0.4, 0.2, -0.1], device="cuda"))
    args = model.Args(control_mode=mode, norm_adv=norm_adv, clip_vloss=False, ent_coef=0.017)
    reference = deepcopy(agent)
    alpha, beta, value = reference.get_policy_and_value(observations.detach())
    distribution = torch.distributions.Beta(alpha, beta)
    logprob = (distribution.log_prob(native.detach()) - reference.log_action_scale).sum(-1)
    ratio = (logprob - old["logprobs"].detach()).exp()
    normalized = advantages.detach()
    if norm_adv:
        normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-8)
    pg = torch.maximum(-normalized * ratio, -normalized * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    frozen = old["coefficients"].detach()
    sample = frozen[:, 0] + (frozen[:, 1:] * old["score"].detach()).sum(-1)
    expected = frozen[:, 0] + (frozen[:, 1:] * (reference_moments(alpha, beta) - old["log_moments"].detach())).sum(-1)
    correction = (ratio * sample - expected).mean() if mode != "ppo" else pg.new_zeros(())
    entropy = (distribution.entropy() + reference.log_action_scale).sum(-1).mean()
    actor_loss = pg + correction - args.ent_coef * entropy
    actor_loss.backward()
    critic_loss = args.vf_coef * 0.5 * (value.flatten() - returns.detach()).square().mean()
    critic_loss.backward()
    white = reference.control(observations.detach())
    if mode == "sampled":
        basis = reference_basis(reference_fisher(old["alpha"].double(), old["beta"].double())).float()
        natural = (basis @ white.unsqueeze(-1)).squeeze(-1)
        fit = sampled_reference.control_variance_loss(natural, normalized, old["score"], old["alpha"],
                                                       old["beta"], old["gram"], reference.actor_parameter_count,
                                                       fisher_diagonal=old["fisher_diagonal"],
                                                       fisher_cross=old["fisher_cross"],
                                                       logit_jacobian=old["logit_jacobian"])
    else:
        fit = (0.5 * normalized.square() * old["score_power"].detach()
               - normalized * (white * old["eligibility"].detach()).sum(-1)
               + 0.5 * torch.einsum("bi,bij,bj->b", white, old["normal"].detach(), white)).mean() / reference.actor_parameter_count
    before_actor, before_critic = gradients(reference.actor), gradients(reference.critic)
    fit.backward()
    assert_gradients(gradients(reference.actor), before_actor, rtol=0, atol=0)
    assert_gradients(gradients(reference.critic), before_critic, rtol=0, atol=0)
    actual, metrics = objective(agent, data, args)
    actual.backward()
    torch.testing.assert_close(actual, actor_loss + critic_loss + fit, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(metrics["control/fit_objective"], fit.detach(), rtol=3e-5, atol=3e-6)
    for owner in ("actor", "critic", "control"):
        assert_gradients(gradients(getattr(agent, owner)), gradients(getattr(reference, owner)), rtol=8e-5, atol=3e-6)
    assert all(value.grad is None for value in teachers(data))


def test_prefit_natural_coefficients_and_actual_variance_diagnostics_are_frozen(envs):
    agent = make_agent(envs)
    prime_control(agent)
    data = training_batch(agent)
    observations, native, advantages, returns, _ = data
    old = model.rollout_statistics(agent, observations, native)
    data = observations, native, advantages, returns, old
    assert all(not value.requires_grad for value in old.values())
    snapshots = {name: value.clone() for name, value in old.items()}
    basis = reference_basis(reference_fisher(old["alpha"].double(), old["beta"].double())).float()
    expected_natural = (basis @ agent.control(observations.detach()).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(old["coefficients"], expected_natural, rtol=3e-6, atol=3e-6)
    diagnose = torch.compile(
        lambda obs, adv, snapshot: model.control_diagnostics(agent, obs, adv, snapshot, exact=True),
        fullgraph=True, options={"triton.cudagraphs": False},
    )
    before = diagnose(observations, advantages, old)
    normalized = (advantages.detach() - advantages.detach().mean()) / (advantages.detach().std() + 1e-8)
    sampled_moment = sampled_reference.control_variance_loss(
        old["coefficients"], normalized, old["score"], old["alpha"], old["beta"],
        old["gram"], agent.actor_parameter_count, fisher_diagonal=old["fisher_diagonal"],
        fisher_cross=old["fisher_cross"], logit_jacobian=old["logit_jacobian"],
    )
    torch.testing.assert_close(before["control/preupdate_sampled_second_moment"], sampled_moment,
                               rtol=3e-5, atol=3e-6)
    args = model.Args(control_mode="analytic", norm_adv=False)
    loss, _ = objective(agent, data, args)
    loss.backward()
    expected_actor = gradients(agent.actor)
    optimizer = torch.optim.Adam(agent.control.parameters(), lr=0.03, eps=1e-5, fused=True)
    torch.nn.utils.clip_grad_norm_(agent.control.parameters(), 0.5)
    optimizer.step()
    agent.zero_grad(set_to_none=True)
    current = (basis @ agent.control(observations.detach()).unsqueeze(-1)).squeeze(-1)
    assert not torch.allclose(current, snapshots["coefficients"], rtol=1e-5, atol=1e-6)
    after = diagnose(observations, advantages, old)
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, rtol=0, atol=0)
        assert not after[name].requires_grad
    loss, _ = objective(agent, data, args)
    loss.backward()
    assert_gradients(gradients(agent.actor), expected_actor, rtol=0, atol=0)
    for name, expected in snapshots.items():
        torch.testing.assert_close(old[name], expected, rtol=0, atol=0)
    # Moving off behavior makes the unscaled correction's value nonzero as well as its derivative.
    with torch.no_grad():
        agent.actor.head[0].bias.add_(torch.tensor([0.4, -0.2, 0.3, -0.5], device="cuda"))
    off = model.Args(control_mode="ppo", norm_adv=False)
    on_loss, _ = objective(agent, data, args)
    off_loss, _ = objective(agent, data, off)
    rescaled = observations, native, advantages * 11 + 23, returns, old
    scaled_on, _ = objective(agent, rescaled, args)
    scaled_off, _ = objective(agent, rescaled, off)
    torch.testing.assert_close(scaled_on - scaled_off, on_loss - off_loss, rtol=3e-4, atol=1e-5)
    assert all(value.grad is None for value in teachers(data))


def test_fullgraph_geometry_linear_features_and_both_fit_gradients_match_eager():
    actor = model.TaskFFN(3, 4, 0.4).cuda()
    observations = torch.randn(7, 3, device="cuda")
    alpha = torch.linspace(1.2, 4.5, 14, device="cuda").reshape(7, 2).requires_grad_()
    beta = torch.linspace(3.7, 1.1, 14, device="cuda").reshape(7, 2).requires_grad_()
    native = torch.linspace(0.07, 0.93, 14, device="cuda").reshape(7, 2)
    advantages = torch.linspace(-2.0, 2.8, 7, device="cuda", requires_grad=True)
    coefficients = torch.randn(7, 5, device="cuda", requires_grad=True)
    parameter_count = sum(parameter.numel() for parameter in actor.parameters())
    old_moments = reference_moments(alpha.detach() + 0.2, beta.detach() + 0.3)

    def kernel(a, b, coeff, adv):
        score = model.beta_score(native, a, b)
        gram = model.actor_jacobian_gram(actor, observations)
        whitening, normal = model.beta_control_geometry(a, b, gram)
        power, eligibility = model.control_linear_features(score, a, b, gram, whitening)
        natural = model.natural_coefficients(coeff, whitening)
        analytic = model.analytic_control_loss(coeff, adv, normal, eligibility, power, parameter_count)
        sampled = model.control_variance_loss(natural, adv, score, a, b, gram, parameter_count)
        expectation = model.control_expectation(natural, a, b, old_moments)
        return analytic + sampled + expectation.mean(), analytic, sampled, whitening, normal, power, eligibility, natural

    eager = kernel(alpha, beta, coefficients, advantages)
    expected_gradients = torch.autograd.grad(eager[0], (alpha, beta, coefficients, advantages), allow_unused=True)
    compiled = torch.compile(kernel, fullgraph=True, options={"triton.cudagraphs": False})
    actual = compiled(alpha, beta, coefficients, advantages)
    actual_gradients = torch.autograd.grad(actual[0], (alpha, beta, coefficients, advantages), allow_unused=True)
    for first, second in zip(actual, eager, strict=True):
        torch.testing.assert_close(first, second, rtol=3e-4, atol=2e-5)
    assert expected_gradients[-1] is None and actual_gradients[-1] is None
    assert_gradients(actual_gradients[:3], expected_gradients[:3], rtol=4e-4, atol=2e-5)
    assert all(parameter.grad is None for parameter in actor.parameters())


@pytest.mark.parametrize("mode", ["ppo", "sampled", "analytic"])
def test_fullgraph_objective_and_exclusive_clipped_adam_updates_match_eager(envs, mode):
    agent = make_agent(envs, mode)
    prime_control(agent)
    data = training_batch(agent)
    args = model.Args(control_mode=mode, ent_coef=0.01, clip_vloss=True)
    with torch.no_grad():
        agent.actor.head[0].bias.add_(torch.tensor([0.3, -0.4, 0.2, -0.1], device="cuda"))
        agent.critic.head[0].bias.add_(0.5)
    reference = deepcopy(agent)
    groups, reference_groups = agent.parameter_groups(), reference.parameter_groups()
    optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
    reference_optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in reference_groups)

    def loss(observations, native, advantages, returns, old):
        return model.policy_loss(agent, observations, native, advantages, returns, old, args)

    compiled = torch.compile(loss, fullgraph=True, options={"triton.cudagraphs": False})
    # Two updates exercise both Adam initialization and its carried moment state.
    for _ in range(2):
        for optimizer in (*optimizers, *reference_optimizers):
            optimizer.zero_grad(set_to_none=True)
        expected, expected_metrics = objective(reference, data, args)
        expected.backward()
        actual, actual_metrics = compiled(*data)
        torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-6)
        for name in expected_metrics:
            torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=3e-4, atol=2e-5)
        actual.backward()
        for owner in ("actor", "critic", "control"):
            assert_gradients(gradients(getattr(agent, owner)), gradients(getattr(reference, owner)), rtol=8e-4, atol=1e-5)
        expected_norms = torch.stack([torch.nn.utils.clip_grad_norm_(group, args.max_grad_norm) for group in reference_groups])
        for optimizer in reference_optimizers:
            optimizer.step()
        norms = torch.empty(3, device="cuda")
        model.optimizer_step(optimizers, groups, args.max_grad_norm, norms)
        torch.testing.assert_close(norms, expected_norms, rtol=8e-4, atol=1e-5)
        for actual_parameter, expected_parameter in zip(agent.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=8e-4, atol=1e-5)
    assert all(value.grad is None for value in teachers(data))
