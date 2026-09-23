"""CUDA behavioral contracts for v8. Run through mlq, never on a CPU fallback."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_jepa_successor_features_v8 as model
from cleanrl import ppo_continuous_action_jepa_shared_ffn_ablation_v6 as reference
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorRewardNorm


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


def snapshots(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def assert_unchanged(parameters, previous):
    for parameter, expected in zip(parameters, previous, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def assert_changed(parameters, previous):
    assert any(not torch.equal(parameter, expected)
               for parameter, expected in zip(parameters, previous, strict=True))


def assert_no_grad(parameters):
    assert all(parameter.grad is None or not bool(parameter.grad.any()) for parameter in parameters)


def make_optimizers(agent, args):
    policy, representation, reward, successor = agent.parameter_groups()
    return (
        torch.optim.Adam(policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
        torch.optim.AdamW(reward, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True) if args.reward_mode != "off" else None,
        torch.optim.AdamW(successor, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
    )


def ppo_batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda")
    actions = torch.rand(rows, 6, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        latent = agent.encoder(observations)
        predictions = agent.successor_head(latent)
        critic_inputs = agent.critic_features(observations, latent, predictions).clone()
        alpha, beta, values = agent.get_policy_and_value(observations, critic_inputs=critic_inputs)
        logprobs = agent.action_logprob(alpha, beta, actions)
    data = (observations, actions, logprobs, torch.randn(rows, device="cuda"),
            values.flatten() + torch.randn(rows, device="cuda"), values.flatten())
    return data, predictions, critic_inputs


@pytest.mark.parametrize("compiled", [False, True])
def test_vector_lambda_targets_respect_boundaries_factual_bootstrap_and_rollout_tail(compiled):
    gamma, lam = 0.8, 0.6
    features = torch.arange(1, 46, device="cuda", dtype=torch.float32).reshape(3, 5, 3)
    predictions = torch.randn_like(features)
    factual_next = torch.randn_like(features) * 7
    terms = torch.zeros(3, 5, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[0, 0] = terms[0, 2] = terms[1, 3] = 1
    truncs[0, 1] = truncs[0, 2] = 1
    gae = get_gae_fn(compiled=compiled, explicit_next_values=True)
    targets = model.successor_lambda_targets(
        features, predictions, factual_next, terms, truncs, gamma, lam, gae,
    )
    immediate = (1 - gamma) * features
    torch.testing.assert_close(targets[0, 0], immediate[0, 0])
    torch.testing.assert_close(targets[0, 1], immediate[0, 1] + gamma * factual_next[0, 1])
    torch.testing.assert_close(targets[0, 2], immediate[0, 2])
    torch.testing.assert_close(targets[0, 3], immediate[0, 3] + gamma * factual_next[0, 3]
                               + gamma * lam * (immediate[1, 3] - predictions[1, 3]))
    torch.testing.assert_close(targets[-1], immediate[-1] + gamma * factual_next[-1])
    delta1 = immediate[1, 4] + gamma * factual_next[1, 4] - predictions[1, 4]
    delta2 = immediate[2, 4] + gamma * factual_next[2, 4] - predictions[2, 4]
    torch.testing.assert_close(targets[0, 4], immediate[0, 4] + gamma * factual_next[0, 4]
                               + gamma * lam * (delta1 + gamma * lam * delta2))

    # Later episodes cannot leak backwards across either kind of boundary.
    changed_features, changed_predictions, changed_next = features.clone(), predictions.clone(), factual_next.clone()
    changed_features[1:, :3] += 1000
    changed_predictions[1:, :3] -= 300
    changed_next[1:, :3] += 200
    changed_next[0, 0] += 50
    changed_next[0, 2] += 50
    changed = model.successor_lambda_targets(
        changed_features, changed_predictions, changed_next, terms, truncs, gamma, lam, gae,
    )
    torch.testing.assert_close(changed[0, :3], targets[0, :3])
    changed_next[0, 1] += 10
    changed = model.successor_lambda_targets(
        changed_features, changed_predictions, changed_next, terms, truncs, gamma, lam, gae,
    )
    torch.testing.assert_close(changed[0, 1] - targets[0, 1], torch.full((3,), gamma * 10, device="cuda"))


def test_successor_feature_map_preserves_raw_reward_and_centered_action_moments():
    raw = torch.tensor([[20.0, -3.0]], device="cuda", requires_grad=True)
    observations = torch.randn(1, 2, 17, device="cuda", requires_grad=True)
    native = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0, 0.2],
                            [0.1, 0.4, 0.6, 0.8, 0.3, 0.9]]], device="cuda", requires_grad=True)
    features = model.successor_features(raw, observations, native)
    torch.testing.assert_close(features[..., 0], raw)
    torch.testing.assert_close(features[..., 1:18], observations)
    centered = 2 * native - 1
    torch.testing.assert_close(features[..., 18:24], centered)
    torch.testing.assert_close(features[..., 24:30], centered.square())
    torch.testing.assert_close(features[..., 30], torch.ones_like(raw))
    assert not features.requires_grad


@pytest.mark.parametrize("mode", ["actor", "both"])
def test_value_and_forecast_are_state_only_even_when_stored_actions_change(envs, mode):
    agent = make_agent(envs, args_for(jepa_mode=mode))
    observations = torch.randn(16, 17, device="cuda")
    first_actions = torch.full((16, 6), -1.0, device="cuda")
    other_actions = torch.full((16, 6), 2.0, device="cuda")
    with torch.no_grad():
        first = agent.get_action_and_value(observations, first_actions)
        other = agent.get_action_and_value(observations, other_actions)
        psi = agent.get_successor(observations)
        # Stored future actions change training features, never inference V(s).
        phi1 = model.successor_features(torch.ones(16, device="cuda"), observations,
                                        (first_actions + 2) / 5)
        phi2 = model.successor_features(torch.ones(16, device="cuda"), observations,
                                        (other_actions + 2) / 5)
        torch.testing.assert_close(first[3], other[3], rtol=0, atol=0)
        torch.testing.assert_close(psi, agent.get_successor(observations), rtol=0, atol=0)
        assert not torch.equal(first[1], other[1])
        assert not torch.equal(phi1[:, 18:30], phi2[:, 18:30])


def test_reward_only_ignores_inactive_targets_and_consumer_channels(envs):
    agent = make_agent(envs, args_for(successor_targets="reward"))
    observations = torch.randn(24, 17, device="cuda")
    latent = agent.encoder(observations)
    predictions = agent.successor_prediction(latent)
    target = torch.randn_like(predictions)
    other = target.clone()
    other[:, 1:] += 1000
    loss = model.successor_objective(agent, predictions, target)
    other_loss = model.successor_objective(agent, predictions, other)
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)
    first_grad = torch.autograd.grad(loss, predictions, retain_graph=True)[0]
    other_grad = torch.autograd.grad(other_loss, predictions)[0]
    torch.testing.assert_close(first_grad, other_grad, rtol=0, atol=0)
    assert not bool(first_grad[:, 1:].any())
    torch.testing.assert_close(loss, F.mse_loss(predictions[:, 0], target[:, 0]))
    with torch.no_grad():
        inputs = agent.critic_features(observations, latent, predictions)
        changed = predictions.clone()
        changed[:, 1:] += 1000
        unchanged_value = agent.critic(agent.critic_features(observations, latent, changed))
        torch.testing.assert_close(agent.critic(inputs), unchanged_value, rtol=0, atol=0)
        assert not bool(inputs[:, -agent.successor_dim + 1:].any())
        changed[:, 0] += 5
        changed_value = agent.critic(agent.critic_features(observations, latent, changed))
        assert not torch.equal(agent.critic(inputs), changed_value)
    joint = make_agent(envs, args_for(successor_targets="joint"))
    assert not torch.equal(model.successor_objective(joint, predictions, target),
                           model.successor_objective(joint, predictions, other))


@pytest.mark.parametrize("mode", ["actor", "both"])
def test_auxiliary_and_consumer_match_capacity_but_only_consumer_responds_to_forecasts(envs, mode):
    auxiliary = make_agent(envs, args_for(jepa_mode=mode, successor_consumer="auxiliary"))
    consumer = make_agent(envs, args_for(jepa_mode=mode, successor_consumer="critic"))
    assert_unchanged(auxiliary.parameters(), snapshots(consumer.parameters()))
    observations = torch.randn(32, 17, device="cuda")
    for agent in (auxiliary, consumer):
        with torch.no_grad():
            before = agent.get_value(observations).clone()
            agent.successor_head[-1].bias.add_(2)
            after = agent.get_value(observations)
        if agent is auxiliary:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        else:
            assert not torch.equal(before, after)
    assert sum(p.numel() for p in auxiliary.parameters()) == sum(p.numel() for p in consumer.parameters())


@pytest.mark.parametrize("mode", ["actor", "both"])
@pytest.mark.parametrize("cached", [False, True])
def test_ppo_cannot_train_encoder_or_successor_even_with_attached_student(envs, mode, cached):
    args = args_for(jepa_mode=mode, successor_encoder_gradients="attached", reward_mode="attached")
    agent = make_agent(envs, args)
    data, _, critic_inputs = ppo_batch(agent)
    loss, _ = model.policy_loss(agent, *data, args, critic_inputs=critic_inputs if cached else None)
    loss.backward()
    policy, representation, reward, successor = agent.parameter_groups()
    assert_no_grad(representation + reward + successor)
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in policy)


@pytest.mark.parametrize("path", ["stopped", "attached"])
def test_successor_student_encoder_flag_never_attaches_teacher_targets(envs, path):
    args = args_for(successor_encoder_gradients=path)
    agent = make_agent(envs, args)
    observations = torch.randn(3, 4, 17, device="cuda")
    following = torch.randn_like(observations, requires_grad=True)
    raw = torch.randn(3, 4, device="cuda", requires_grad=True)
    actions = torch.rand(3, 4, 6, device="cuda", requires_grad=True)
    teacher = agent.get_successor(observations)
    next_teacher = agent.get_successor(following)
    teacher.retain_grad()
    next_teacher.retain_grad()
    features = model.successor_features(raw, observations, actions)
    targets = model.successor_lambda_targets(
        features, teacher, next_teacher, torch.zeros_like(raw), torch.zeros_like(raw),
        args.gamma, args.gae_lambda, get_gae_fn(explicit_next_values=True),
    )
    assert not targets.requires_grad
    # Explicit detach also protects callers supplying an attached external target.
    external_target = targets.detach().clone().requires_grad_()
    predicted = agent.get_successor(observations)
    loss = model.successor_objective(agent, predicted, external_target)
    loss.backward()
    assert external_target.grad is None and teacher.grad is None and next_teacher.grad is None
    assert following.grad is None and raw.grad is None and actions.grad is None
    assert_no_grad(tuple(agent.actor.parameters()) + tuple(agent.critic.parameters()) + tuple(agent.ssl.parameters()))
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.successor_head.parameters())
    if path == "stopped":
        assert_no_grad(agent.encoder.parameters())
    else:
        assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.encoder.parameters())


@pytest.mark.parametrize("mode", ["actor", "both"])
@pytest.mark.parametrize("consumer", ["auxiliary", "critic"])
def test_cached_critic_fit_survives_encoder_and_successor_changes(envs, mode, consumer):
    agent = make_agent(envs, args_for(jepa_mode=mode, successor_consumer=consumer))
    data, old_predictions, critic_inputs = ppo_batch(agent)
    observations = data[0]
    with torch.no_grad():
        alpha, _, old_value = agent.get_policy_and_value(observations, critic_inputs=critic_inputs)
        agent.encoder[0].weight.add_(0.1)
        agent.successor_head[-1].bias.add_(2)
        new_alpha, _, cached_value = agent.get_policy_and_value(observations, critic_inputs=critic_inputs)
        online_value = agent.get_value(observations)
        torch.testing.assert_close(cached_value, old_value, rtol=0, atol=0)
        assert not torch.equal(alpha, new_alpha)
        diagnose = torch.compile(
            lambda obs, inputs, predictions, targets: model.successor_fit_diagnostics(
                agent, obs, inputs, predictions, targets, 0.99,
            ), fullgraph=True, options={"triton.cudagraphs": False},
        )
        diagnostics = diagnose(observations, critic_inputs, old_predictions, torch.randn_like(old_predictions))
        expected_drift = (online_value - cached_value).square().mean().sqrt()
        torch.testing.assert_close(diagnostics["drift/current_critic_value_rmse"], expected_drift)
        if mode == "actor" and consumer == "auxiliary":
            torch.testing.assert_close(online_value, old_value, rtol=0, atol=0)
        else:
            assert bool(expected_drift > 0)


def test_raw_successor_targets_do_not_replace_normalized_ppo_or_reward_head_units(envs):
    args = args_for()
    agent = make_agent(envs, args)
    raw = np.array([20.0, -4.0], dtype=np.float32)
    normalized = VectorRewardNorm(2, args.gamma).normalize(raw, np.ones(2, dtype=bool))
    transfer = RolloutTransfer(1, 2, (17,), "cuda", fields={
        "observations": (17,), "native_actions": (6,), "raw_rewards": (),
    })
    try:
        transfer.push(0, normalized, np.ones(2), np.zeros(2),
                      observations=np.zeros((2, 17)), native_actions=np.full((2, 6), 0.7), raw_rewards=raw)
        rollout = transfer.upload()
        features = model.successor_features(rollout.fields["raw_rewards"],
                                            rollout.fields["observations"], rollout.fields["native_actions"])
        zeros = torch.zeros_like(features)
        targets = model.successor_lambda_targets(
            features, zeros, zeros, rollout.terminations, rollout.truncations,
            args.gamma, args.gae_lambda, get_gae_fn(explicit_next_values=True),
        )
        _, ppo_returns = get_gae_fn(explicit_next_values=True)(
            rollout.rewards, torch.zeros_like(rollout.rewards), rollout.terminations,
            rollout.truncations, torch.zeros_like(rollout.rewards), args.gamma, args.gae_lambda,
        )
        torch.testing.assert_close(targets[..., 0], (1 - args.gamma) * torch.as_tensor(raw, device="cuda")[None])
        torch.testing.assert_close(ppo_returns[0], torch.as_tensor(normalized, device="cuda"))
        assert not torch.allclose(ppo_returns, targets[..., 0] / (1 - args.gamma))
        obs = rollout.fields["observations"].flatten(0, 1)
        native = rollout.fields["native_actions"].flatten(0, 1)
        _, reward_loss, _, _ = model.representation_loss(
            agent, obs, native, obs, rollout.rewards.flatten(), targets.flatten(0, 1), args,
        )
        expected = F.mse_loss(agent.reward_prediction(agent.encoder(obs), native), rollout.rewards.flatten())
        torch.testing.assert_close(reward_loss, expected)
    finally:
        transfer.close()


@pytest.mark.parametrize("activation", ["tanh", "stiglu"])
def test_actor_representation_and_ssl_initialization_pair_across_modes_and_treatments(envs, activation):
    baseline_args = args_for(task_activation=activation, task_residual=True)
    torch.manual_seed(19)
    baseline = reference.Agent(envs, baseline_args).cuda()
    expected_rng = torch.get_rng_state()
    for mode, consumer, targets, path in (
        ("both", "auxiliary", "reward", "stopped"),
        ("actor", "critic", "joint", "attached"),
        ("both", "critic", "joint", "stopped"),
        ("actor", "auxiliary", "reward", "attached"),
    ):
        agent = make_agent(envs, args_for(jepa_mode=mode, task_activation=activation, task_residual=True,
                                        successor_consumer=consumer, successor_targets=targets,
                                        successor_encoder_gradients=path), seed=19)
        for name in ("actor", "encoder", "ssl", "reward_head"):
            assert_unchanged(getattr(agent, name).parameters(), snapshots(getattr(baseline, name).parameters()))
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        # Nonzero state sensitivity avoids a shared hard-zero forecast cold start.
        observations = torch.randn(16, 17, device="cuda")
        with torch.no_grad():
            predictions = agent.get_successor(observations)
        assert bool((predictions.std(dim=0) > 0).all())


def test_successor_optimizer_has_exclusive_ownership_and_fixed_clip(envs):
    args = args_for(max_grad_norm=0.03)
    agent = make_agent(envs, args)
    groups = agent.parameter_groups()
    ids = [id(parameter) for group in groups for parameter in group]
    assert len(ids) == len(set(ids)) == len(tuple(agent.parameters()))
    assert {id(parameter) for parameter in agent.encoder.parameters()} <= {id(parameter) for parameter in groups[1]}
    optimizers = make_optimizers(agent, args)
    before = [snapshots(group) for group in groups]
    observations = torch.randn(32, 17, device="cuda")
    predictions = agent.get_successor(observations)
    model.successor_objective(agent, predictions, torch.full_like(predictions, 1000)).backward()
    # Only the head optimizer steps; weight decay on unrelated owners is absent.
    norms = model.optimizer_step(agent, (None, None, None, optimizers[3]), args, policy_step=False)
    for group, previous in zip(groups[:3], before[:3], strict=True):
        assert_unchanged(group, previous)
    assert_changed(groups[3], before[3])
    assert bool(norms[3] > 0.5)
    clipped = torch.stack([parameter.grad.square().sum() for parameter in groups[3]]).sum().sqrt()
    torch.testing.assert_close(clipped, torch.tensor(0.5, device="cuda"), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode,path,targets,consumer", [
    ("both", "stopped", "joint", "critic"),
    ("actor", "attached", "reward", "auxiliary"),
])
def test_compiled_cuda_forward_backward_and_owned_update(envs, mode, path, targets, consumer):
    args = args_for(jepa_mode=mode, successor_encoder_gradients=path,
                    successor_targets=targets, successor_consumer=consumer)
    agent = make_agent(envs, args)
    data, _, critic_inputs = ppo_batch(agent)
    following = torch.randn_like(data[0])
    rewards = torch.randn(32, device="cuda")
    teachers = torch.randn(32, agent.successor_dim, device="cuda")

    def objective(observations, native, old_logprobs, advantages, returns, old_values,
                  next_observations, normalized_rewards, fixed_targets, fixed_inputs):
        ppo, _ = model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                   returns, old_values, args, critic_inputs=fixed_inputs)
        components, _ = model.representation_components(
            agent, observations, native, next_observations, normalized_rewards, fixed_targets, args,
        )
        return ppo, components

    policy, components = objective(*data, following, rewards, teachers, critic_inputs)
    # SIGReg draws fresh directions; compare deterministic heads, not random draws.
    (policy + components[2:].sum()).backward()
    groups = agent.parameter_groups()
    expected_gradients = snapshots([parameter.grad for parameter in groups[0] + groups[2] + groups[3]])
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, fullgraph=True, options={"triton.cudagraphs": False})
    actual_policy, actual_components = compiled(*data, following, rewards, teachers, critic_inputs)
    torch.testing.assert_close(actual_policy, policy, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_components[2:], components[2:], rtol=3e-5, atol=3e-6)
    (actual_policy + actual_components.sum()).backward()
    for parameter, expected in zip(groups[0] + groups[2] + groups[3], expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=5e-4, atol=5e-6)
    before = [snapshots(group) for group in groups]
    model.optimizer_step(agent, make_optimizers(agent, args), args)
    for group, previous in zip(groups, before, strict=True):
        assert_changed(group, previous)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.parameters())


@pytest.mark.parametrize("ppo_size", [512, 2048, 16384])
def test_successor_and_ssl_exposure_are_independent_of_ppo_batch_size(ppo_size):
    rows = 16 * 1024
    args = model.validate_args(args_for(num_envs=16, num_steps=1024,
                                        num_minibatches=rows // ppo_size, update_epochs=10))
    generator = torch.Generator(device="cuda").manual_seed(3)
    ppo_count = world_count = 0
    for _ in range(args.update_epochs):
        exposures = torch.zeros(rows, dtype=torch.int32, device="cuda")
        for ppo_indices, ssl_indices in model.iter_update_batches(
                rows, args.minibatch_size, args.ssl_minibatch_size, "cuda", generator):
            if ppo_indices is not None:
                ppo_count += 1
            exposures[ssl_indices] += 1
            world_count += 1
        torch.testing.assert_close(exposures, torch.ones_like(exposures))
    assert world_count == 320
    assert ppo_count == args.update_epochs * rows // ppo_size
