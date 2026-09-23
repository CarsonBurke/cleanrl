"""Behavioral CUDA contracts for factual-goal policy fitting; run through mlq."""

from types import SimpleNamespace
import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_intact_factual_goals_v8 as model
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.rollout_graph import graph_compile


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Submit these contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2, -1, -0.5, -3, -1, -4], dtype=np.float32),
            np.array([1, 3, 2, 0.5, 4, 2], dtype=np.float32),
        ),
    )


def test_hindsight_never_uses_reset_or_later_episode():
    following = torch.arange(6 * 2 * 3, device="cuda").reshape(6, 2, 3).float()
    terms = torch.zeros(6, 2, device="cuda", dtype=torch.bool)
    truncs = torch.zeros_like(terms)
    terms[1, 0] = True
    truncs[3, 1] = True
    goals = model.make_goal_observations(following, terms, truncs, 4)
    endpoints = [[1, 3], [1, 3], [5, 3], [5, 3], [5, 5], [5, 5]]
    for step in range(6):
        for env in range(2):
            torch.testing.assert_close(goals[step, env], following[endpoints[step][env], env])


def test_one_step_goals_are_exact_factual_successors():
    following = torch.randn(7, 3, 23, device="cuda")
    boundary = torch.rand(7, 3, device="cuda") > 0.5
    torch.testing.assert_close(
        model.make_goal_observations(following, boundary, ~boundary, 1), following,
    )


def test_parameter_owners_are_disjoint_and_complete(envs):
    args = model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    agent = model.Agent(envs, args).cuda()
    groups = [set(map(id, group)) for group in agent.parameter_groups()]
    assert len(groups) == 3
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert set.union(*groups) == set(map(id, agent.parameters()))
    assert groups[1] == set(map(id, agent.prescriber.parameters()))
    assert groups[2] == set(map(id, agent.critic.parameters()))


@pytest.mark.parametrize("budget", [0.0, 0.01, 0.1, 1.0])
def test_weights_solve_empirical_kl_and_preserve_advantage_order(budget):
    advantages = torch.linspace(-3, 3, 512, device="cuda")
    weights, metrics = model.factual_goal_weights(advantages, budget, "weighted")
    assert not weights.requires_grad
    torch.testing.assert_close(weights.mean(), weights.new_tensor(1.0))
    assert torch.all(weights[1:] >= weights[:-1])
    torch.testing.assert_close(metrics["selection/empirical_kl"], weights.new_tensor(budget), atol=3e-6, rtol=3e-5)
    transformed, _ = model.factual_goal_weights(advantages * 7 + 13, budget, "weighted")
    torch.testing.assert_close(weights, transformed, atol=2e-5, rtol=2e-5)


def test_degenerate_advantages_and_uniform_ablation_are_finite():
    for advantages, mode in [
        (torch.ones(32, device="cuda"), "weighted"),
        (torch.randn(32, device="cuda"), "uniform"),
        (torch.ones(1, device="cuda"), "weighted"),
    ]:
        weights, metrics = model.factual_goal_weights(advantages, 0.1, mode)
        torch.testing.assert_close(weights, torch.ones_like(weights))
        assert all(torch.isfinite(value) for value in metrics.values())


def test_compiled_rollout_weight_solver_matches_eager():
    advantages = torch.randn(16 * 1024, device="cuda")
    compiled = graph_compile(lambda x: model.factual_goal_weights(x, 0.1, "weighted"))
    expected, expected_metrics = model.factual_goal_weights(advantages, 0.1, "weighted")
    actual, metrics = compiled(advantages)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    for name in metrics:
        torch.testing.assert_close(metrics[name], expected_metrics[name], rtol=3e-5, atol=3e-5)


def test_goal_density_matches_torch_mixture_and_updates_only_prescriber(envs):
    agent = model.Agent(envs, model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)).cuda()
    latent = torch.randn(32, 64, device="cuda")
    previous = torch.randn(32, agent.action_dim, device="cuda")
    intent = torch.randn_like(latent)
    logits, means, stds = agent.goal_distribution(latent, previous)
    expected = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means, stds[..., None]), 1),
    ).log_prob(intent)
    actual = agent.goal_logprob(latent, previous, intent)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    (-actual.mean()).backward()
    world, actor, critic = agent.parameter_groups()
    assert all(parameter.grad is None for parameter in (*world, *critic))
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in actor)


def test_host_sampling_and_conditional_beta_match_gpu_after_refresh(envs):
    args = model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    agent = model.Agent(envs, args).cuda()
    rows, seed = 16, 713
    observations = torch.randn(rows, agent.input_dim, device="cuda")
    mirror = model.HostIntactActor(agent, rows, seed)
    rng = np.random.default_rng(seed)
    for mutate in [False, True]:
        if mutate:
            with torch.no_grad():
                for parameter in agent.parameters():
                    parameter.add_(0.002 * torch.randn_like(parameter))
            mirror.refresh()
        host_logits = mirror(observations.cpu().numpy()).copy()
        with torch.no_grad():
            latent = agent.encode(observations)
            previous = observations[:, agent.observation_dim:]
            logits, means, stds = agent.goal_distribution(latent, previous)
            probabilities = logits.softmax(-1).cpu().numpy()
            cumulative = probabilities.cumsum(-1)
            cumulative[:, -1] = 1.0
            component = (rng.random((rows, 1)) > cumulative).sum(-1)
            noise = rng.standard_normal((rows, 64), dtype=np.float32)
            selected = torch.as_tensor(component, device="cuda")
            indices = torch.arange(rows, device="cuda")
            intent = means[indices, selected] + stds[indices, selected, None] * torch.from_numpy(noise).cuda()
            torch.testing.assert_close(torch.from_numpy(mirror._law_input[:, 64:128].copy()).cuda(), intent, rtol=3e-5, atol=3e-6)
            alpha, beta = agent.law_from_intent(latent, intent, previous)
            host_alpha, host_beta = (torch.nn.functional.softplus(torch.from_numpy(host_logits).cuda()) + 1).chunk(2, -1)
            torch.testing.assert_close(host_alpha, alpha, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(host_beta, beta, rtol=3e-5, atol=3e-6)


def test_compiled_joint_objective_matches_eager_and_gradients(envs):
    args = model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    agent = model.Agent(envs, args).cuda()
    rows = 32
    observations = torch.randn(rows, agent.input_dim, device="cuda")
    following = torch.randn_like(observations)
    native = torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1
    weights, _ = model.factual_goal_weights(torch.randn(rows, device="cuda"), 0.1, "weighted")
    with torch.no_grad():
        latent = agent.encode(observations)
        intent = agent.encode(following) - latent
    # Isolate deterministic joint factor fitting from randomized SIGReg.
    def objective(z, g, a, w, previous):
        goal_nll = -(w * agent.goal_logprob(z, previous, g)).mean()
        alpha, beta = agent.law_from_intent(z, g, previous)
        law_nll = -(w * agent.action_logprob(alpha, beta, a)).mean()
        return goal_nll + args.goal_nll_coef * law_nll

    inputs = (latent, intent, native, weights, observations[:, agent.observation_dim:])
    eager = objective(*inputs)
    fitted_parameters = tuple(
        parameter for module in (agent.prescriber, agent.previous_action_embedding, agent.action_law)
        for parameter in module.parameters()
    )
    eager_grad = torch.autograd.grad(eager, fitted_parameters)
    compiled = torch.compile(objective, fullgraph=True, mode="reduce-overhead")
    actual = compiled(*inputs)
    actual_grad = torch.autograd.grad(actual, fitted_parameters)
    torch.testing.assert_close(actual, eager, rtol=3e-5, atol=3e-5)
    # If FP32 reductions disagree, retain a higher-precision oracle in the
    # failure output rather than blindly relaxing a per-element tolerance.
    oracle = copy.deepcopy(agent).double()
    z64, g64, a64, w64, previous64 = (value.double() for value in inputs)
    oracle_alpha, oracle_beta = oracle.law_from_intent(z64, g64, previous64)
    oracle_loss = -(w64 * oracle.goal_logprob(z64, previous64, g64)).mean()
    oracle_loss -= args.goal_nll_coef * (w64 * oracle.action_logprob(oracle_alpha, oracle_beta, a64)).mean()
    oracle_parameters = tuple(
        parameter for module in (oracle.prescriber, oracle.previous_action_embedding, oracle.action_law)
        for parameter in module.parameters()
    )
    oracle_grad = torch.autograd.grad(oracle_loss, oracle_parameters)
    for label, gradients in (("eager", eager_grad), ("compiled", actual_grad)):
        delta = torch.cat([(value.double() - reference).flatten() for value, reference in zip(gradients, oracle_grad)])
        reference = torch.cat([value.flatten() for value in oracle_grad])
        print(f"{label} FP64 gradient relative L2 error: {float(delta.norm() / reference.norm()):.8g}")
    for actual_value, expected_value in zip(actual_grad, eager_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=3e-4, atol=3e-5)


def test_compiled_full_world_and_critic_objective(envs):
    args = model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    agent = model.Agent(envs, args).cuda()
    rows = 32
    observations = torch.randn(rows, agent.input_dim, device="cuda")
    following = torch.randn_like(observations)
    native = torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1
    weights, _ = model.factual_goal_weights(torch.randn(rows, device="cuda"), 0.1, "weighted")
    with torch.no_grad():
        latent = agent.encode(observations)
        intent = agent.encode(following) - latent
    returns = torch.randn(rows, device="cuda")
    old_values = torch.randn_like(returns)
    rewards = torch.randn_like(returns)
    terms = torch.zeros(rows, device="cuda")
    full = torch.compile(
        lambda *batch: model.training_loss(agent, *batch, args),
        fullgraph=True, mode="reduce-overhead",
    )
    loss, metrics = full(observations, native, weights, returns, old_values, following, latent, intent, rewards, terms)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    for group in agent.parameter_groups():
        gradients = [parameter.grad for parameter in group if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
