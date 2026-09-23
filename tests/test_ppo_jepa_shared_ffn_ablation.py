"""CUDA scientific controls for shared-FFN ablations; execute only through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_jepa_shared_ffn_ablation_v6 as model
from cleanrl import ppo_continuous_action_jepa_shared_ffn_v4 as reference
from cleanrl.shared.ppo_loop import device_minibatches
from cleanrl.shared.runtime import configure_runtime


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


def args_for(**kwargs):
    return model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8, **kwargs)


def make_agent(envs, args, seed=13):
    torch.manual_seed(seed)
    return model.Agent(envs, args).cuda()


def batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda")
    actions = torch.rand(rows, 6, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        logprobs = agent.action_logprob(alpha, beta, actions)
    return (observations, actions, logprobs, torch.randn(rows, device="cuda"),
            values.flatten() + torch.randn(rows, device="cuda"), values.flatten())


def snapshots(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def assert_unchanged(parameters, before):
    for parameter, expected in zip(parameters, before, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def assert_changed(parameters, before):
    assert any(not torch.equal(parameter, expected)
               for parameter, expected in zip(parameters, before, strict=True))


def assert_no_grad(parameters):
    assert all(parameter.grad is None or not bool(parameter.grad.any()) for parameter in parameters)


def make_optimizers(agent, args):
    policy, representation, reward = agent.parameter_groups()
    return (
        torch.optim.Adam(policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True) if representation else None,
        torch.optim.AdamW(reward, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True) if reward else None,
    )


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_off_control_preserves_v4_rng_predictions_losses_gradients_and_update(envs, mode):
    args = args_for(jepa_mode=mode, reward_mode="off")
    torch.manual_seed(13)
    old = reference.Agent(envs, mode, sigreg_num_proj=16, sigreg_proj_chunk=8).cuda()
    old_rng = torch.get_rng_state().clone()
    agent = make_agent(envs, args)
    torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
    policy, representation, reward = agent.parameter_groups()
    groups = [set(map(id, group)) for group in (policy, representation, reward)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert set.union(*groups) == set(map(id, agent.parameters()))
    for group in (policy, representation, reward):
        assert len(group) == len(set(map(id, group)))
    if mode == "none":
        assert agent.encoder is None and agent.ssl is None
        assert not representation and not reward
    data = batch(agent)
    following = torch.randn_like(data[0])
    rewards = torch.randn(data[0].shape[0], device="cuda")
    for actual, expected in zip(agent.get_policy_value_latents(data[0]),
                                old.get_policy_value_latents(data[0]), strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.manual_seed(29)
    expected_policy, expected_ssl, _ = reference.ppo_loss(old, *data, args, following)
    torch.manual_seed(29)
    actual_policy, _ = model.policy_loss(agent, *data, args)
    actual_ssl, actual_reward, _ = model.representation_loss(
        agent, data[0], data[1], following, rewards, args)
    torch.testing.assert_close(actual_policy, expected_policy, rtol=0, atol=0)
    torch.testing.assert_close(actual_ssl, expected_ssl, rtol=0, atol=0)
    assert actual_reward == 0
    (actual_policy + actual_ssl + actual_reward).backward()
    (expected_policy + expected_ssl).backward()
    old_policy, old_representation = old.parameter_groups()
    for actual, expected in zip(policy + representation, old_policy + old_representation, strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert_no_grad(reward)
    old_optimizers = (
        torch.optim.Adam(old_policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(old_representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True) if old_representation else None,
    )
    for parameters, optimizer in zip((old_policy, old_representation), old_optimizers, strict=True):
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
    model.optimizer_step(agent, make_optimizers(agent, args), args)
    assert_unchanged(policy + representation, snapshots(old_policy + old_representation))
    for actual, expected in zip(agent.get_policy_and_value(data[0]),
                                old.get_policy_and_value(data[0]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both"])
def test_reward_uses_current_latent_stored_action_and_real_reward_with_owned_gradients(envs, mode):
    detached_args = args_for(jepa_mode=mode, reward_mode="detached")
    attached_args = args_for(jepa_mode=mode, reward_mode="attached")
    detached = make_agent(envs, detached_args)
    attached = make_agent(envs, attached_args)
    observations = torch.randn(32, 17, device="cuda", requires_grad=True)
    actions = torch.rand(32, 6, device="cuda", requires_grad=True)
    following = torch.randn_like(observations, requires_grad=True)
    rewards = torch.linspace(-1.5, 2.5, 32, device="cuda", requires_grad=True)
    head_gradients = []
    for agent, args in ((detached, detached_args), (attached, attached_args)):
        policy, representation, reward = agent.parameter_groups()
        _, reward_loss, metrics = model.representation_loss(
            agent, observations, actions, following, rewards, args)
        latent = agent.encoder(observations)
        prediction = agent.reward_head(torch.cat((latent, 2 * actions.detach() - 1), dim=-1)).flatten()
        expected = F.mse_loss(prediction, rewards.detach()) * args.reward_coef
        torch.testing.assert_close(reward_loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics["reward/mse"], expected / args.reward_coef, rtol=0, atol=0)
        expected_ev = 1 - (rewards.detach() - prediction.detach()).var(unbiased=False) / rewards.detach().var(unbiased=False)
        torch.testing.assert_close(metrics["reward/explained_variance"], expected_ev, rtol=0, atol=0)
        assert not metrics["reward/mse"].requires_grad
        assert not metrics["reward/explained_variance"].requires_grad
        expected_head = torch.autograd.grad(expected, reward, retain_graph=True)
        expected_encoder = torch.autograd.grad(expected, tuple(agent.encoder.parameters()))
        reward_loss.backward()
        for parameter, expected_gradient in zip(reward, expected_head, strict=True):
            torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
        head_gradients.append([parameter.grad.clone() for parameter in reward])
        assert_no_grad(policy)
        assert_no_grad(agent.ssl.parameters())
        assert_no_grad((actions, rewards, following))
        if args.reward_mode == "attached":
            assert observations.grad.norm() > 0
            for parameter, expected_gradient in zip(agent.encoder.parameters(), expected_encoder, strict=True):
                torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
            assert any(bool(parameter.grad.any()) for parameter in agent.encoder.parameters())
        else:
            assert_no_grad(representation)
            assert_no_grad((observations,))
        _, changed_future_loss, _ = model.representation_loss(
            agent, observations, actions, following + 100, rewards, args)
        torch.testing.assert_close(changed_future_loss, reward_loss, rtol=0, atol=0)
        _, _, constant_metrics = model.representation_loss(
            agent, observations, actions, following, torch.ones_like(rewards), args)
        assert torch.isnan(constant_metrics["reward/explained_variance"])
    for actual, expected in zip(*head_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_ppo_remains_detached_with_reward_enabled(envs, mode):
    args = args_for(jepa_mode=mode, reward_mode="attached")
    agent = make_agent(envs, args)
    policy, representation, reward = agent.parameter_groups()
    model.policy_loss(agent, *batch(agent), args)[0].backward()
    assert_no_grad(representation + reward)
    for net in (agent.actor, agent.critic):
        assert all(parameter.grad is not None and bool(parameter.grad.any())
                   for parameter in net.parameters() if parameter.ndim == 2)


def test_detached_reward_clipping_cannot_scale_encoder_update(envs):
    off_args = args_for(reward_mode="off", max_grad_norm=1e-3)
    detached_args = args_for(reward_mode="detached", max_grad_norm=1e-3)
    off, detached = make_agent(envs, off_args), make_agent(envs, detached_args)
    data = batch(off)
    following, rewards = torch.randn_like(data[0]), torch.full((32,), 1e6, device="cuda")
    for agent, args in ((off, off_args), (detached, detached_args)):
        torch.manual_seed(37)
        ssl, reward_loss, _ = model.representation_loss(agent, data[0], data[1], following, rewards, args)
        (ssl + reward_loss).backward()
    detached_reward = detached.parameter_groups()[2]
    assert sum(parameter.grad.square().sum() for parameter in detached_reward).sqrt() > 1e4
    for agent, args in ((off, off_args), (detached, detached_args)):
        policy = agent.parameter_groups()[0]
        before_policy = snapshots(policy)
        model.optimizer_step(agent, make_optimizers(agent, args), args, policy_step=False)
        assert_unchanged(policy, before_policy)
    assert_unchanged(detached.parameter_groups()[1], snapshots(off.parameter_groups()[1]))
    assert_changed(detached_reward, snapshots(off.parameter_groups()[2]))


def test_residual_tanh_adds_only_identity_skip_without_changing_capacity(envs):
    plain_args, residual_args = args_for(), args_for(task_residual=True)
    plain, residual = make_agent(envs, plain_args), make_agent(envs, residual_args)
    assert plain.parameter_counts() == residual.parameter_counts()
    assert_unchanged(residual.parameters(), snapshots(plain.parameters()))
    observations = torch.randn(32, 17, device="cuda")
    for plain_net, residual_net in ((plain.actor, residual.actor), (plain.critic, residual.critic)):
        features = plain.encoder(observations).detach()
        hidden = residual_net.first(features)
        expected = residual_net.head(hidden + residual_net.second(hidden))
        torch.testing.assert_close(residual_net(features), expected, rtol=0, atol=0)
        assert not torch.equal(plain_net(features), expected)
        with torch.no_grad():
            for parameter in residual_net.second.parameters():
                parameter.zero_()
        torch.testing.assert_close(residual_net(features), residual_net.head(hidden), rtol=0, atol=0)


@pytest.mark.parametrize("ppo_size", [512, 2048, 4096])
def test_scheduler_holds_ssl_exposure_and_shuffle_fixed_and_routes_policy_once(ppo_size):
    size, ssl_size = 4096, 512
    generator = torch.Generator(device="cuda").manual_seed(73)
    reference_generator = torch.Generator(device="cuda").manual_seed(73)
    original_global_rng = torch.cuda.get_rng_state().clone()
    all_epochs = []
    for _ in range(2):
        schedule = list(model.iter_update_batches(size, ppo_size, ssl_size, "cuda", generator))
        expected = device_minibatches(size, ssl_size, "cuda", reference_generator)
        assert len(schedule) == size // ssl_size
        ssl_rows = [ssl for _, ssl in schedule]
        assert all(row.numel() == ssl_size and row.is_cuda for row in ssl_rows)
        torch.testing.assert_close(torch.cat(ssl_rows), torch.cat(expected), rtol=0, atol=0)
        torch.testing.assert_close(torch.cat(ssl_rows).sort().values,
                                   torch.arange(size, device="cuda"), rtol=0, atol=0)
        policy_rows = [ppo for ppo, _ in schedule if ppo is not None]
        assert len(policy_rows) == size // ppo_size
        torch.testing.assert_close(torch.cat(policy_rows), torch.cat(ssl_rows), rtol=0, atol=0)
        stride = ppo_size // ssl_size
        for offset in range(0, len(schedule), stride):
            ppo, first_ssl = schedule[offset]
            assert ppo is not None and ppo.numel() == ppo_size
            torch.testing.assert_close(first_ssl, ppo[:ssl_size], rtol=0, atol=0)
            assert all(row[0] is None for row in schedule[offset + 1:offset + stride])
            torch.testing.assert_close(ppo, torch.cat(ssl_rows[offset:offset + stride]), rtol=0, atol=0)
        all_epochs.append(torch.cat(ssl_rows))
    assert not torch.equal(*all_epochs)
    torch.testing.assert_close(torch.cuda.get_rng_state(), original_global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 1, "num_steps": 1537, "num_minibatches": 1},
    {"num_envs": 1, "num_steps": 4096, "num_minibatches": 3},
    {"num_envs": 1, "num_steps": 4096, "num_minibatches": 16},
    {"num_envs": 1, "num_steps": 4096, "num_minibatches": 1, "target_kl": 0.01},
    {"num_envs": 1, "num_steps": 4096, "num_minibatches": 1, "ssl_minibatch_size": 1024},
])
def test_invalid_batch_or_early_stop_cannot_change_ssl_exposure(kwargs):
    with pytest.raises(ValueError):
        model.validate_args(args_for(**kwargs))


@pytest.mark.parametrize("activation,residual,projection,mode", [
    ("tanh", False, "none", "none"),
    ("tanh", True, "hidden", "both"),
    ("tanh", False, "all", "actor"),
    ("stiglu", False, "none", "critic"),
    ("stiglu", True, "hidden", "actor"),
    ("stiglu", True, "all", "both"),
])
def test_native_host_matches_compiled_policy_and_refreshes_after_owned_updates(
        envs, activation, residual, projection, mode):
    args = args_for(task_activation=activation, task_residual=residual,
                    weight_projection=projection, jepa_mode=mode, reward_mode="attached")
    agent = make_agent(envs, args)
    data = batch(agent)
    host = model.HostPolicy(agent, data[0].shape[0])
    host_observations = data[0].cpu().numpy()
    compiled = torch.compile(agent.get_policy_and_value, fullgraph=True,
                             options={"triton.cudagraphs": False})
    initial_host_logits = host(host_observations).copy()
    for updated in (False, True):
        if updated:
            following = torch.randn_like(data[0])
            rewards = torch.randn(data[0].shape[0], device="cuda")
            ppo, _ = model.policy_loss(agent, *data, args)
            ssl, reward, _ = model.representation_loss(agent, data[0], data[1], following, rewards, args)
            (ppo + ssl + reward).backward()
            model.optimizer_step(agent, make_optimizers(agent, args), args)
            host.refresh()
        with torch.no_grad():
            alpha, beta, value = compiled(data[0])
            actual_logits = host(host_observations).copy()
            host_alpha, host_beta = (F.softplus(torch.from_numpy(actual_logits).cuda()) + 1).chunk(2, dim=-1)
            torch.testing.assert_close(host_alpha, alpha, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(host_beta, beta, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(agent.get_value(data[0]), value, rtol=3e-5, atol=3e-6)
            if updated:
                assert not np.array_equal(actual_logits, initial_host_logits)


def projection_matrices(agent, activation, projection):
    matrices = []
    for net in (agent.actor, agent.critic):
        for stage in (net.first, net.second):
            if activation == "tanh":
                matrices.append((stage[0].weight, 1))
            else:
                branch = stage[0]
                matrices.extend(((branch.gate.weight, 1), (branch.up.weight, 1),
                                 (branch.down.weight, 0)))
        if projection == "all":
            matrices.append((net.head[0].weight, 1))
    return matrices


@pytest.mark.parametrize("activation", ["tanh", "stiglu"])
@pytest.mark.parametrize("projection", ["hidden", "all"])
def test_projection_axes_scope_initial_head_scale_and_adam_state(envs, activation, projection):
    args = args_for(task_activation=activation, task_residual=True,
                    weight_projection=projection, reward_mode="attached")
    unprojected = make_agent(envs, args_for(task_activation=activation, task_residual=True,
                                          reward_mode="attached"))
    agent = make_agent(envs, args)
    matrices = projection_matrices(agent, activation, projection)
    for weight, dim in matrices:
        torch.testing.assert_close(weight.norm(dim=dim), torch.ones(weight.shape[1 - dim], device="cuda"),
                                   rtol=2e-6, atol=2e-6)
    if projection == "hidden":
        assert_unchanged((agent.actor.head[0].weight, agent.critic.head[0].weight),
                         snapshots((unprojected.actor.head[0].weight, unprojected.critic.head[0].weight)))
        torch.testing.assert_close(agent.actor.head[0].weight.norm(dim=1),
                                   torch.full((12,), 0.01, device="cuda"), rtol=2e-6, atol=2e-7)
    else:
        assert not torch.equal(agent.actor.head[0].weight, unprojected.actor.head[0].weight)
    assert_unchanged(agent.parameter_groups()[1] + agent.parameter_groups()[2],
                     snapshots(unprojected.parameter_groups()[1] + unprojected.parameter_groups()[2]))
    data = batch(agent)
    optimizers = make_optimizers(agent, args)
    ppo, _ = model.policy_loss(agent, *data, args)
    ssl, reward, _ = model.representation_loss(
        agent, data[0], data[1], torch.randn_like(data[0]), torch.randn(32, device="cuda"), args)
    (ppo + ssl + reward).backward()
    before_policy = snapshots(agent.parameter_groups()[0])
    compiled_projection = torch.compile(
        agent.project_policy_weights, fullgraph=True, options={"triton.cudagraphs": False},
    )
    model.optimizer_step(agent, optimizers, args, projection=compiled_projection)
    assert_changed(agent.parameter_groups()[0], before_policy)
    for weight, dim in matrices:
        torch.testing.assert_close(weight.norm(dim=dim), torch.ones(weight.shape[1 - dim], device="cuda"),
                                   rtol=2e-6, atol=2e-6)
    with torch.no_grad():
        for weight, _ in matrices:
            weight.mul_(torch.linspace(0.3, 2.7, weight.numel(), device="cuda").reshape_as(weight))
    projected_ids = {id(weight) for weight, _ in matrices}
    untouched = tuple(parameter for parameter in agent.parameters() if id(parameter) not in projected_ids)
    before_untouched = snapshots(untouched)
    before_matrices = snapshots(weight for weight, _ in matrices)
    state_tensors = [value for optimizer in optimizers if optimizer is not None
                     for state in optimizer.state.values() for value in state.values()
                     if isinstance(value, torch.Tensor)]
    before_state = snapshots(state_tensors)
    agent.project_policy_weights()
    for (weight, dim), before in zip(matrices, before_matrices, strict=True):
        expected = before / before.norm(dim=dim, keepdim=True)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    assert_unchanged(untouched, before_untouched)
    assert_unchanged(state_tensors, before_state)


def test_ssl_only_step_never_projects_or_updates_task_ffns(envs):
    args = args_for(task_activation="stiglu", task_residual=True,
                    weight_projection="all", reward_mode="attached")
    agent = make_agent(envs, args)
    policy, representation, reward = agent.parameter_groups()
    with torch.no_grad():
        for weight, _ in projection_matrices(agent, "stiglu", "all"):
            weight.mul_(2)
    before_policy, before_representation = snapshots(policy), snapshots(representation)
    before_reward = snapshots(reward)
    data = batch(agent)
    ssl, reward_loss, _ = model.representation_loss(
        agent, data[0], data[1], torch.randn_like(data[0]), torch.randn(32, device="cuda"), args)
    (ssl + reward_loss).backward()
    model.optimizer_step(agent, make_optimizers(agent, args), args, policy_step=False)
    assert_unchanged(policy, before_policy)
    assert_changed(representation, before_representation)
    assert_changed(reward, before_reward)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_compiled_attached_losses_preserve_owner_gradients_and_optimizer_updates(envs, mode):
    args = args_for(jepa_mode=mode, reward_mode="attached")
    agent = make_agent(envs, args)
    data = batch(agent)
    following, rewards = torch.randn_like(data[0]), torch.randn(32, device="cuda")

    def objectives(observations, actions, old_logprobs, advantages, returns, old_values, next_obs, real_rewards):
        policy, _ = model.policy_loss(agent, observations, actions, old_logprobs,
                                      advantages, returns, old_values, args)
        ssl, reward, _ = model.representation_loss(agent, observations, actions, next_obs, real_rewards, args)
        return policy, ssl, reward

    eager_policy, _, eager_reward = objectives(*data, following, rewards)
    (eager_policy + eager_reward).backward()
    policy, representation, reward = agent.parameter_groups()
    expected_policy = [parameter.grad.clone() for parameter in policy]
    expected_reward = [parameter.grad.clone() for parameter in reward]
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objectives, fullgraph=True, options={"triton.cudagraphs": False})
    actual_policy, actual_ssl, actual_reward = compiled(*data, following, rewards)
    torch.testing.assert_close(actual_policy, eager_policy, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_reward, eager_reward, rtol=3e-5, atol=3e-6)
    (actual_policy + actual_ssl + actual_reward).backward()
    for parameter, expected in zip(policy + reward, expected_policy + expected_reward, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=5e-4, atol=5e-6)
    before = [snapshots(group) for group in (policy, representation, reward)]
    model.optimizer_step(agent, make_optimizers(agent, args), args)
    for group, previous in zip((policy, representation, reward), before, strict=True):
        if group:
            assert_changed(group, previous)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.parameters())


@pytest.mark.parametrize("reward_mode", ["off", "detached", "attached"])
def test_compiled_sparse_diagnostics_measure_weighted_gradients_without_mutating_grad(envs, reward_mode):
    args = args_for(reward_mode=reward_mode, reward_coef=2.3, sigreg_weight=0.13)
    agent = make_agent(envs, args)
    data = batch(agent)
    following, rewards = torch.randn_like(data[0]), torch.randn(32, device="cuda")
    policy, representation, reward = agent.parameter_groups()
    model.policy_loss(agent, *data, args)[0].backward()

    def components(observations, actions, next_obs, real_rewards):
        return model.representation_components(agent, observations, actions, next_obs, real_rewards, args)[0]

    compiled = torch.compile(components, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
    measured = compiled(data[0], data[1], following, rewards)
    encoder = tuple(agent.encoder.parameters())
    term_gradients = []
    for term in measured:
        gradients = torch.autograd.grad(term, encoder, retain_graph=True, allow_unused=True)
        term_gradients.append(tuple(torch.zeros_like(parameter) if gradient is None else gradient
                                    for parameter, gradient in zip(encoder, gradients, strict=True)))
    norms = [torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
             for gradients in term_gradients]

    def cosine(left, right):
        dot = torch.stack([(a * b).sum() for a, b in
                           zip(term_gradients[left], term_gradients[right], strict=True)]).sum()
        return dot / (norms[left] * norms[right]).clamp_min(1e-12)

    expected = {
        "balance/shared_prediction_grad_norm": norms[0],
        "balance/shared_sigreg_grad_norm": norms[1],
        "balance/shared_reward_grad_norm": norms[2],
        "balance/shared_sigreg_to_prediction_grad_ratio": norms[1] / norms[0].clamp_min(1e-12),
        "balance/shared_reward_to_prediction_grad_ratio": norms[2] / norms[0].clamp_min(1e-12),
        "balance/shared_grad_cosine": cosine(0, 1),
        "balance/shared_prediction_reward_grad_cosine": cosine(0, 2),
        "balance/shared_sigreg_reward_grad_cosine": cosine(1, 2),
    }
    parameters = tuple(agent.parameters())
    before = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    rng_before = torch.cuda.get_rng_state().clone()
    for _ in range(2):
        actual = model.gradient_balance(agent, measured)
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=3e-5, atol=3e-6)
        for parameter, previous in zip(parameters, before, strict=True):
            if previous is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, previous, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before, rtol=0, atol=0)
    assert norms[0] > 0 and norms[1] > 0
    if reward_mode == "attached":
        assert norms[2] > 0
    else:
        assert norms[2] == 0
    before_policy = snapshots(parameter.grad for parameter in policy)
    measured.sum().backward()
    for index, parameter in enumerate(encoder):
        expected_gradient = sum(gradients[index] for gradients in term_gradients)
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=5e-4, atol=5e-6)
    assert_unchanged((parameter.grad for parameter in policy), before_policy)
    before_representation, before_reward = snapshots(representation), snapshots(reward)
    model.optimizer_step(agent, make_optimizers(agent, args), args)
    assert_changed(representation, before_representation)
    if reward_mode == "off":
        assert_unchanged(reward, before_reward)
    else:
        assert_changed(reward, before_reward)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_parameter_accounting_separates_inference_and_auxiliary_capacity(envs, mode):
    agents = [make_agent(envs, args_for(jepa_mode=mode, task_activation=activation))
              for activation in ("tanh", "stiglu")]
    for agent in agents:
        counts = agent.parameter_counts()
        assert counts["actor_ffn"] == sum(parameter.numel() for parameter in agent.actor.parameters())
        assert counts["critic_ffn"] == sum(parameter.numel() for parameter in agent.critic.parameters())
        expected_encoder = 0 if agent.encoder is None else sum(parameter.numel() for parameter in agent.encoder.parameters())
        expected_jepa = 0 if agent.ssl is None else sum(parameter.numel() for parameter in agent.ssl.parameters())
        expected_reward = sum(parameter.numel() for parameter in agent.parameter_groups()[2])
        assert counts["shared_encoder"] == expected_encoder
        assert counts["jepa_auxiliary"] == expected_jepa
        assert counts["reward_head"] == expected_reward
        assert counts["inference"] == counts["actor_ffn"] + counts["critic_ffn"] + expected_encoder
        assert counts["training_only"] == expected_jepa + expected_reward
        assert counts["total"] == sum(parameter.numel() for parameter in agent.parameters())
        assert counts["total"] == counts["inference"] + counts["training_only"]
    tanh, stiglu = [agent.parameter_counts() for agent in agents]
    assert stiglu["actor_ffn"] > tanh["actor_ffn"]
    assert stiglu["critic_ffn"] > tanh["critic_ffn"]
    assert stiglu["shared_encoder"] == tanh["shared_encoder"]
    assert stiglu["training_only"] == tanh["training_only"]
    assert_unchanged(agents[1].parameter_groups()[1] + agents[1].parameter_groups()[2],
                     snapshots(agents[0].parameter_groups()[1] + agents[0].parameter_groups()[2]))
