"""Queue these policy-geometry and unclipped-critic CUDA contracts through mlq."""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

from cleanrl.ppo_continuous_action_critic_curvature_unclipped_v4 import (
    CriticCurvatureAdam as AblationCriticCurvatureAdam,
)
from cleanrl.ppo_continuous_action_tangent_policy_analytic_kl_v5 import (
    Agent,
    CriticCurvatureAdam,
    TangentPolicyOptimizer,
    apply_beta_fisher,
    beta_fisher_metric,
    beta_kl,
)
from test_ppo_normres_twohot import device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


def tiny_actor(device, *, symmetric=False):
    with torch.device(device):
        actor = nn.Sequential(nn.Linear(2, 3, dtype=torch.float64), nn.Tanh(),
                              nn.Linear(3, 4, dtype=torch.float64))
    with torch.no_grad():
        for index, parameter in enumerate(actor.parameters()):
            coordinates = torch.arange(parameter.numel(), device=device, dtype=parameter.dtype)
            parameter.copy_((0.2 * (coordinates + index + 1).sin()).reshape_as(parameter))
        if symmetric:
            # All four concentrations agree initially, so mirrored action pairs
            # have equal old density and opposite, analytically aligned labels.
            actor[-1].weight.copy_(actor[-1].weight[:1].clone().expand_as(actor[-1].weight))
            actor[-1].bias.fill_(0.1)
    return actor


def actual_actor(device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        single_action_space=gym.spaces.Box(-2.0, 3.0, (2,), np.float32),
    )
    with torch.device(device):
        actor = Agent(spaces).actor
    # Exercise the real 64x64 actor, with nonzero nonlinear hidden-layer
    # derivatives and no acceptance assertion depending on random initialization.
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor[0].weight[:2].copy_(actor[0].weight.new_tensor([[0.6, 0.2], [-0.1, 0.5]]))
        actor[0].bias[:2].copy_(actor[0].bias.new_tensor([0.15, -0.1]))
        actor[2].weight[:2, :2].copy_(actor[2].weight.new_tensor([[0.7, -0.2], [0.3, 0.6]]))
        actor[2].bias[:2].copy_(actor[2].bias.new_tensor([0.1, 0.2]))
        actor[4].weight[:, :2].copy_(actor[4].weight.new_tensor(
            [[0.3, 0.2], [-0.2, 0.4], [-0.3, -0.2], [0.2, -0.4]]
        ))
        actor[4].bias.copy_(actor[4].bias.new_tensor([0.1, -0.15, -0.2, 0.05]))
    return actor


def rollout(actor, num_envs):
    reference = next(actor.parameters())
    states = reference.new_tensor([
        [-0.75, 0.25], [-0.75, 0.25], [0.2, -0.5], [0.2, -0.5],
        [0.8, 0.4], [0.8, 0.4], [-0.2, 0.7], [-0.2, 0.7],
    ])
    first = reference.new_tensor([0.2, 0.8, 0.3, 0.7, 0.25, 0.75, 0.35, 0.65])
    actions = torch.stack((first, 1.0 - first), dim=-1)
    # The score for increasing alpha and decreasing beta is proportional to
    # log(u / (1-u)); the second action coordinate is its paired complement.
    advantages = first.log() - (1.0 - first).log()
    return (
        states[:, None, :].expand(-1, num_envs, -1).reshape(-1, 2).clone(),
        actions[:, None, :].expand(-1, num_envs, -1).reshape(-1, 2).clone(),
        advantages[:, None].expand(-1, num_envs).reshape(-1).clone(),
    )


def split_streams(num_envs, iteration):
    order = [(slot - iteration) % num_envs for slot in range(num_envs)]
    boundary = num_envs - max(1, num_envs // 4)
    return order[:boundary], order[boundary:]


def normalized_advantages(advantages, num_envs, iteration):
    fit, _ = split_streams(num_envs, iteration)
    values = advantages.view(-1, num_envs)
    fitted = values[:, fit]
    return ((values - fitted.mean()) / (fitted.std() + 1e-8)).reshape(-1)


def observed_policy_change(old_logits, new_logits, native, advantages, num_envs):
    # Independent double-precision distribution calculation, including the
    # action sum and time mean for each whole environment stream.
    old_alpha, old_beta = (F.softplus(old_logits.double()) + 1.0).chunk(2, -1)
    alpha, beta = (F.softplus(new_logits.double()) + 1.0).chunk(2, -1)
    old = Beta(old_alpha, old_beta, validate_args=False)
    new = Beta(alpha, beta, validate_args=False)
    ratio_change = torch.expm1((new.log_prob(native.double()) - old.log_prob(native.double())).sum(-1))
    labels = advantages.double()
    gain = torch.minimum(ratio_change * labels, ratio_change.clamp(-0.2, 0.2) * labels)
    kl = kl_divergence(old, new).sum(-1)
    return gain.view(-1, num_envs).mean(0), kl.view(-1, num_envs).mean(0)


def metric(metrics, name):
    return metrics["policy_model/" + name]


def assert_parameters(actor, expected, *, exact=True):
    for parameter, saved in zip(actor.parameters(), expected):
        torch.testing.assert_close(parameter, saved, rtol=0 if exact else 3e-4,
                                   atol=0 if exact else 3e-6)


def test_beta_fisher_pullback_is_exact_kl_hessian_with_cross_action_and_state_reductions(device):
    logits = torch.tensor([
        [-3.0, 1.2, 0.8, -0.4], [0.2, -1.1, 1.7, 0.3], [6.0, 2.5, 3.2, 8.0],
    ], device=device, dtype=torch.float64)
    old_alpha, old_beta = (F.softplus(logits) + 1.0).chunk(2, -1)
    old = Beta(old_alpha, old_beta, validate_args=False)

    def exact_kl(flat):
        alpha, beta = (F.softplus(flat.reshape_as(logits)) + 1.0).chunk(2, -1)
        return kl_divergence(old, Beta(alpha, beta, validate_args=False)).sum(-1).mean()

    hessian = torch.autograd.functional.hessian(exact_kl, logits.flatten())
    basis = torch.eye(logits.numel(), device=device, dtype=logits.dtype)
    geometry = beta_fisher_metric(logits)
    # Every basis direction, rather than just a diagonal or a scalar quadratic
    # form, detects missing alpha/beta cross terms and coordinate mixing.
    pullback = torch.stack([
        apply_beta_fisher(geometry, vector.reshape_as(logits)).flatten() / logits.shape[0]
        for vector in basis
    ], dim=1)
    torch.testing.assert_close(pullback, hessian, rtol=2e-9, atol=2e-11)

    displaced = logits + logits.new_tensor([0.35, -0.2, 0.1, 0.45])
    alpha, beta = (F.softplus(displaced) + 1.0).chunk(2, -1)
    expected_per_state = kl_divergence(old, Beta(alpha, beta, validate_args=False)).sum(-1)
    torch.testing.assert_close(beta_kl(old_alpha, old_beta, alpha, beta), expected_per_state,
                               rtol=2e-12, atol=2e-12)


def test_parameter_fisher_product_matches_exact_and_finite_difference_kl_hessians(device):
    actor = tiny_actor(device)
    optimizer = TangentPolicyOptimizer(actor, num_envs=4, num_steps=2, compile=False)
    observations = next(actor.parameters()).new_tensor([
        [-0.8, 0.3], [0.2, -0.5], [0.9, 0.4], [-0.2, 0.7], [0.5, -0.9], [0.1, 0.6],
    ])
    names, parameters = zip(*actor.named_parameters())
    origins = tuple(parameter.detach().clone() for parameter in parameters)
    sizes = [parameter.numel() for parameter in parameters]
    flat = torch.cat([parameter.flatten() for parameter in origins])
    vector = torch.linspace(-0.35, 0.5, flat.numel(), device=device, dtype=flat.dtype)

    def unpack(value):
        return tuple(part.reshape_as(parameter) for part, parameter in zip(value.split(sizes), origins))

    def logits_at(value):
        return torch.func.functional_call(actor, dict(zip(names, unpack(value))), (observations,), strict=True)

    old_logits = logits_at(flat).detach()
    old_alpha, old_beta = (F.softplus(old_logits) + 1.0).chunk(2, -1)
    old = Beta(old_alpha, old_beta, validate_args=False)

    def exact_kl(value):
        alpha, beta = (F.softplus(logits_at(value)) + 1.0).chunk(2, -1)
        return kl_divergence(old, Beta(alpha, beta, validate_args=False)).sum(-1).mean()

    actual = torch.cat([part.flatten() for part in optimizer._fisher(
        origins, unpack(vector), observations, beta_fisher_metric(old_logits)
    )])
    expected = torch.autograd.functional.hessian(exact_kl, flat) @ vector
    torch.testing.assert_close(actual, expected, rtol=3e-8, atol=2e-10)
    epsilon = 1e-4
    gradient = torch.func.grad(exact_kl)
    finite_difference = (gradient(flat + epsilon * vector) - gradient(flat - epsilon * vector)) / (2 * epsilon)
    torch.testing.assert_close(actual, finite_difference, rtol=2e-6, atol=2e-9)


def capture_full_batch_proposal(monkeypatch, optimizer, batch_size):
    evaluations = []
    forward = optimizer._logits

    def observe(parameters, observations):
        output = forward(parameters, observations)
        if observations.shape[0] == batch_size:
            evaluations.append(tuple(parameter.detach().clone() for parameter in parameters))
        return output

    # Observe actual evaluated proposals without changing losses or returning
    # mock predictions. The commit/rejection and independent density checks
    # below are the externally visible contract, not the number of calls.
    monkeypatch.setattr(optimizer, "_logits", observe)
    return evaluations


@pytest.mark.parametrize("num_envs,iteration", [(4, 0), (8, 3)])
def test_aligned_step_improves_both_splits_and_heldout_labels_cannot_change_proposal(
    device, monkeypatch, num_envs, iteration,
):
    actor = tiny_actor(device, symmetric=True)
    rejected_actor = deepcopy(actor)
    observations, native, advantages = rollout(actor, num_envs)
    origins = tuple(parameter.detach().clone() for parameter in actor.parameters())
    old_logits = actor(observations).detach()
    options = dict(num_envs=num_envs, num_steps=8, kl_budget=0.004,
                   cg_iters=10, backtracks=8, compile=False)
    optimizer = TangentPolicyOptimizer(actor, **options)
    rejected_optimizer = TangentPolicyOptimizer(rejected_actor, **options)
    aligned_proposals = capture_full_batch_proposal(monkeypatch, optimizer, observations.shape[0])
    rejected_proposals = capture_full_batch_proposal(monkeypatch, rejected_optimizer, observations.shape[0])
    fit, held = split_streams(num_envs, iteration)
    anti_aligned = advantages.view(8, num_envs).clone()
    anti_aligned[:, held] *= -1

    accepted = optimizer.step(observations, native, advantages, iteration)
    rejected = rejected_optimizer.step(observations, native, anti_aligned.reshape(-1), iteration)

    assert metric(accepted, "accepted").item() == 1
    assert metric(rejected, "accepted").item() == 0
    assert metric(rejected, "fit_selected").item() == 1
    assert metric(rejected, "candidate_heldout_gain").item() < 0
    assert metric(rejected, "step_scale").item() == 0
    assert metric(rejected, "committed_full_kl").item() == 0
    assert_parameters(rejected_actor, origins)
    assert_parameters(actor, aligned_proposals[-1])
    for proposed, rejected_proposal in zip(aligned_proposals[-1], rejected_proposals[-1]):
        # Exact candidate equality guards both the direction and its length,
        # even though the public committed step scale becomes zero on rejection.
        torch.testing.assert_close(proposed, rejected_proposal, rtol=0, atol=0)
    for name in ("candidate_fit_gain", "candidate_fit_kl", "predicted_fit_gain", "predicted_fit_kl",
                 "linear_predicted_gain", "quadratic_predicted_kl", "gradient_norm", "cg_relative_residual"):
        torch.testing.assert_close(metric(accepted, name), metric(rejected, name), rtol=0, atol=0)

    gain, kl = observed_policy_change(old_logits, actor(observations).detach(), native,
                                     normalized_advantages(advantages, num_envs, iteration), num_envs)
    assert torch.all(gain[fit] > 0)
    assert torch.all(gain[held] > 0)
    assert kl[fit].mean().item() <= optimizer.kl_budget
    assert kl[held].mean().item() <= optimizer.kl_budget
    for split, streams in (("fit", fit), ("heldout", held)):
        torch.testing.assert_close(metric(accepted, "candidate_" + split + "_gain"), gain[streams].mean(),
                                   rtol=2e-8, atol=2e-10)
        torch.testing.assert_close(metric(accepted, "candidate_" + split + "_kl"), kl[streams].mean(),
                                   rtol=2e-8, atol=2e-10)


def test_zero_advantage_rejects_without_nonfinite_or_stale_updates(device):
    actor = tiny_actor(device, symmetric=True)
    observations, native, advantages = rollout(actor, 4)
    optimizer = TangentPolicyOptimizer(actor, num_envs=4, num_steps=8, kl_budget=0.004,
                                       cg_iters=10, backtracks=8, compile=False)
    accepted = optimizer.step(observations, native, advantages, iteration=0)
    assert metric(accepted, "accepted").item() == 1
    origins = tuple(parameter.detach().clone() for parameter in actor.parameters())

    rejected = optimizer.step(observations, native, torch.zeros_like(advantages), iteration=1)

    assert_parameters(actor, origins)
    for value in rejected.values():
        assert torch.isfinite(value).all()
    for name in ("accepted", "fit_selected", "step_scale", "committed_full_kl", "linear_predicted_gain"):
        assert metric(rejected, name).item() == 0


def critic_history(optimizer, parameter):
    return {name: optimizer.state[parameter][name].clone() for name in ("step", "exp_avg", "exp_avg_sq")}


def assert_critic_history(optimizer, parameter, expected):
    for name, value in expected.items():
        torch.testing.assert_close(optimizer.state[parameter][name], value, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_class", [CriticCurvatureAdam, AblationCriticCurvatureAdam],
                         ids=["tangent_policy", "unclipped_critic_ablation"])
def test_unclipped_critic_finds_raw_gradient_quadratic_optimum_with_one_moment_update(device, optimizer_class):
    parameter = nn.Parameter(torch.tensor([2.0, -1.0, 0.25], device=device))
    reference = nn.Parameter(parameter.detach().clone())
    curvature = parameter.new_tensor([90.0, 6.0, 40.0])
    lr, betas, eps = 0.4, (0.6, 0.8), 1e-5
    optimizer = optimizer_class([parameter], lr=lr, betas=betas, eps=eps, compile=False)
    adam = torch.optim.Adam([reference], lr=lr, betas=betas, eps=eps, fused=True)

    def objective():
        return (0.5 * (curvature * parameter.square()).sum()).reshape(1)

    before = objective()
    before.sum().backward()
    before = before.detach()
    origin, raw_gradient = parameter.detach().clone(), parameter.grad.clone()
    reference.grad = raw_gradient.clone()
    adam.step()
    optimizer.probe()
    torch.testing.assert_close(parameter, reference, rtol=2e-6, atol=5e-7)
    torch.testing.assert_close(parameter.grad, raw_gradient, rtol=0, atol=0)
    for name in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(optimizer.state[parameter][name], adam.state[reference][name],
                                   rtol=2e-6, atol=1e-7)
    observed = critic_history(optimizer, parameter)
    assert observed["step"].item() == 1
    direction = (origin - parameter.detach()) / lr
    optimum = (raw_gradient * direction).sum() / (curvature * direction.square()).sum()
    probe = objective().detach()
    torch.testing.assert_close(optimizer.probe_prediction,
                               (raw_gradient * (origin - parameter.detach())).sum().reshape(1))

    optimizer.propose(before, probe)
    torch.testing.assert_close(parameter, origin - optimum * direction, rtol=2e-5, atol=2e-5)
    assert_critic_history(optimizer, parameter, observed)
    candidate = objective().detach()
    candidate_weights = parameter.detach().clone()
    optimizer.finish(before, probe, candidate)

    torch.testing.assert_close(parameter, candidate_weights, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.lrs, optimum.reshape(1), rtol=2e-5, atol=2e-5)
    assert objective().item() < probe.item() < before.item()
    assert_critic_history(optimizer, parameter, observed)


@pytest.mark.parametrize("optimizer_class", [CriticCurvatureAdam, AblationCriticCurvatureAdam],
                         ids=["tangent_policy", "unclipped_critic_ablation"])
def test_critic_domain_invalid_trials_restore_origin_but_observe_raw_gradient_once(device, optimizer_class):
    parameter = nn.Parameter(torch.tensor([0.1], device=device))
    optimizer = optimizer_class([parameter], lr=0.4, betas=(0.6, 0.8), compile=False)
    before = parameter.log()
    before.sum().backward()
    before = before.detach()
    origin, gradient = parameter.detach().clone(), parameter.grad.clone()

    optimizer.probe()
    observed = critic_history(optimizer, parameter)
    torch.testing.assert_close(observed["exp_avg"], 0.4 * gradient)
    torch.testing.assert_close(observed["exp_avg_sq"], 0.2 * gradient.square())
    assert observed["step"].item() == 1
    probe = parameter.detach().log()
    optimizer.propose(before, probe)
    candidate = parameter.detach().log()
    assert torch.isnan(probe).all() and torch.isnan(candidate).all()
    optimizer.finish(before, probe, candidate)

    torch.testing.assert_close(parameter, origin, rtol=0, atol=0)
    assert torch.isfinite(parameter).all()
    assert_critic_history(optimizer, parameter, observed)


def test_compiled_actual_actor_cycles_match_eager_without_postwarmup_cuda_sync(device, monkeypatch):
    from torch._inductor.compile_fx import compile_fx

    compilations = 0

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return compile_fx(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    actor = actual_actor(device)
    eager_actor = deepcopy(actor)
    options = dict(num_envs=4, num_steps=8, kl_budget=0.004, cg_iters=10, backtracks=8)
    real_compile = torch.compile
    with monkeypatch.context() as patch:
        patch.setattr(torch, "compile", lambda function, **kwargs: real_compile(function, backend=backend, **kwargs))
        optimizer = TangentPolicyOptimizer(actor, **options)
    eager = TangentPolicyOptimizer(eager_actor, **options, compile=False)
    observations, native, advantages = rollout(actor, 4)

    for iteration, outcome in enumerate(("aligned", "anti_aligned", "zero", "aligned")):
        current_observations = observations * (1.0 + 0.02 * iteration)
        labels = advantages.clone()
        fit, held = split_streams(4, iteration)
        if outcome == "anti_aligned":
            labels.view(8, 4)[:, held] *= -1
        elif outcome == "zero":
            labels.zero_()
        origins = tuple(parameter.detach().clone() for parameter in actor.parameters())
        old_logits = actor(current_observations).detach()
        expected = eager.step(current_observations, native, labels, iteration)
        previous_sync_mode = torch.cuda.get_sync_debug_mode()
        try:
            if iteration:
                torch.cuda.set_sync_debug_mode("error")
                with torch._dynamo.config.patch(error_on_recompile=True):
                    actual = optimizer.step(current_observations, native, labels, iteration)
            else:
                # Warm every actual JVP/VJP, fit/full-batch check and CG graph;
                # compilation and its synchronization are allowed only here.
                # Earlier reference calculations must not hide the cold-process
                # KL-dispatch failure observed by the first real v4 benchmark.
                monkeypatch.setattr(torch.distributions.kl, "_KL_MEMOIZE", {})
                actual = optimizer.step(current_observations, native, labels, iteration)
        finally:
            torch.cuda.set_sync_debug_mode(previous_sync_mode)

        if iteration == 0:
            warm_compilations = compilations
            assert warm_compilations > 0
        assert compilations == warm_compilations
        should_accept = outcome == "aligned"
        assert metric(actual, "accepted").item() == float(should_accept)
        assert metric(expected, "accepted").item() == float(should_accept)
        assert_parameters(actor, tuple(eager_actor.parameters()), exact=False)
        for name in ("candidate_fit_gain", "candidate_fit_kl", "candidate_heldout_gain", "candidate_heldout_kl",
                     "linear_predicted_gain", "quadratic_predicted_kl", "step_scale", "committed_full_kl"):
            torch.testing.assert_close(metric(actual, name), metric(expected, name), rtol=3e-4, atol=3e-6)
        for value in actual.values():
            assert torch.isfinite(value).all()
        if should_accept:
            gain, kl = observed_policy_change(old_logits, actor(current_observations).detach(), native,
                                             normalized_advantages(labels, 4, iteration), 4)
            assert torch.all(gain[fit] > 0) and torch.all(gain[held] > 0)
            # Forward parameters are FP32; allow only its roundoff against the
            # independent FP64 KL calculation, not a relaxed trust-region size.
            assert kl[fit].mean().item() <= optimizer.kl_budget + 1e-6
            assert kl[held].mean().item() <= optimizer.kl_budget + 1e-6
        else:
            assert_parameters(actor, origins)
