"""CUDA scientific contracts for geometry/drift v7; execute only through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_jepa_geometry_drift_v7 as model
from cleanrl import ppo_continuous_action_jepa_shared_ffn_ablation_v6 as reference
from cleanrl.shared.runtime import configure_runtime


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; run through mlq")


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


def args_for(**kwargs):
    return model.Args(sigreg_num_proj=16, sigreg_proj_chunk=8, **kwargs)


def make_agent(envs, args, seed=13):
    torch.manual_seed(seed)
    return model.Agent(envs, args).cuda()


def batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda")
    actions = torch.rand(rows, 6, device="cuda") * 0.8 + 0.1
    values, logprobs, features = model.rollout_statistics(agent, observations, actions)
    data = (observations, actions, logprobs, torch.randn(rows, device="cuda"),
            values + torch.randn(rows, device="cuda"), values)
    return data, features


def assert_no_grad(parameters):
    assert all(parameter.grad is None or not bool(parameter.grad.any()) for parameter in parameters)


def optimizers_for(agent, args):
    policy, representation, reward = agent.parameter_groups()
    return (
        torch.optim.Adam(policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
        torch.optim.AdamW(reward, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
    )


def test_huber_matches_mse_loss_and_gradient_through_unit_boundary():
    residual = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device="cuda", requires_grad=True)
    target = torch.zeros_like(residual)
    mse = model.prediction_objective(residual, target, "mse")
    huber = model.prediction_objective(residual, target, "huber")
    torch.testing.assert_close(huber, mse, rtol=0, atol=0)
    mse_grad = torch.autograd.grad(mse, residual)[0]
    huber_grad = torch.autograd.grad(huber, residual)[0]
    torch.testing.assert_close(huber_grad, mse_grad, rtol=0, atol=0)
    torch.testing.assert_close(huber_grad, 2 * residual / residual.numel())


def test_huber_has_linear_not_quadratic_tails():
    residual = torch.tensor([-4.0, -2.0, 2.0, 4.0], device="cuda", requires_grad=True)
    target = torch.zeros_like(residual)
    huber = model.prediction_objective(residual, target, "huber")
    torch.testing.assert_close(huber, (2 * residual.abs() - 1).mean(), rtol=0, atol=0)
    gradient = torch.autograd.grad(huber, residual)[0]
    torch.testing.assert_close(gradient, 2 * residual.sign() / residual.numel(), rtol=0, atol=0)
    assert huber < model.prediction_objective(residual, target, "mse")


@pytest.mark.parametrize("loss_kind", ["mse", "huber"])
def test_target_stop_is_after_projector_and_sigreg_still_trains_both_times(envs, loss_kind):
    gradients = {}
    for target_gradient in ("attached", "stopped"):
        args = args_for(prediction_loss=loss_kind, prediction_target_gradient=target_gradient)
        agent = make_agent(envs, args)
        current = torch.randn(32, 17, device="cuda", requires_grad=True)
        following = torch.randn_like(current, requires_grad=True)
        actions = torch.rand(32, 6, device="cuda", requires_grad=True)
        rewards = torch.randn(32, device="cuda", requires_grad=True)
        captured = {}

        def capture_projection(module, inputs, output):
            captured["embeddings"] = output

        def capture_prediction(module, inputs, output):
            captured["prediction"] = output

        projector_hook = agent.ssl.projector.register_forward_hook(capture_projection)
        predictor_hook = agent.ssl.pred_proj.register_forward_hook(capture_prediction)
        rng_before = torch.cuda.get_rng_state().clone()
        components, metrics = model.representation_components(agent, current, actions, following, rewards, args)
        rng_after = torch.cuda.get_rng_state().clone()
        projector_hook.remove()
        predictor_hook.remove()
        embeddings = captured["embeddings"]
        prediction = captured["prediction"]
        prediction_grads = torch.autograd.grad(
            components[0], (current, following, embeddings), retain_graph=True,
        )
        assert prediction_grads[0].norm() > 0
        assert prediction_grads[2][0].norm() > 0
        if target_gradient == "stopped":
            # Stopping before the projector would leave embeddings[1]'s gradient nonzero.
            assert not bool(prediction_grads[1].any())
            assert not bool(prediction_grads[2][1].any())
        else:
            assert prediction_grads[1].norm() > 0
            assert prediction_grads[2][1].norm() > 0
        sigreg_grads = torch.autograd.grad(components[1], (current, following), retain_graph=True)
        assert all(gradient.norm() > 0 for gradient in sigreg_grads)
        gradients[target_gradient] = sigreg_grads
        expected_target = embeddings[1].detach() if target_gradient == "stopped" else embeddings[1]
        expected_prediction = model.prediction_objective(prediction, expected_target, loss_kind)
        expected_projector_grad = torch.autograd.grad(
            expected_prediction, tuple(agent.ssl.projector.parameters()), retain_graph=True,
        )
        actual_projector_grad = torch.autograd.grad(
            components[0], tuple(agent.ssl.projector.parameters()), retain_graph=True,
        )
        for actual, expected in zip(actual_projector_grad, expected_projector_grad, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            residual = prediction - embeddings[1]
            squared = residual.square()
            tails = residual.abs() > 1
            torch.testing.assert_close(metrics["ssl/shared_prediction_loss"], expected_prediction)
            torch.testing.assert_close(metrics["ssl/shared_prediction_raw_mse"], squared.mean())
            torch.testing.assert_close(metrics["ssl/shared_residual_gt1_fraction"], tails.float().mean())
            torch.testing.assert_close(metrics["ssl/shared_residual_gt1_squared_error_share"],
                                       squared[tails].sum() / squared.sum().clamp_min(1e-12))
            # Exactly one SIGReg draw: metrics must not advance the training RNG again.
            torch.cuda.set_rng_state(rng_before)
            expected_regularization = agent.ssl.sigreg(embeddings.detach())
            torch.testing.assert_close(metrics["ssl/shared_sigreg_loss"], expected_regularization)
            torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after, rtol=0, atol=0)
        assert all(not metric.requires_grad for metric in metrics.values())
        # Default detached reward learns the real target without changing any representation owner.
        components[2].backward(retain_graph=True)
        assert_no_grad(agent.parameter_groups()[0] + agent.parameter_groups()[1])
        assert any(parameter.grad is not None and bool(parameter.grad.any())
                   for parameter in agent.reward_head.parameters())
        assert_no_grad((current, following, actions, rewards))
    for attached, stopped in zip(gradients["attached"], gradients["stopped"], strict=True):
        torch.testing.assert_close(attached, stopped, rtol=0, atol=0)


@pytest.mark.parametrize("mode,features", [("actor", "online"), ("both", "online"), ("both", "rollout")])
def test_ppo_cannot_train_encoder_ssl_reward_or_cache(envs, mode, features):
    args = args_for(jepa_mode=mode, critic_feature_updates=features, reward_mode="attached")
    agent = make_agent(envs, args)
    data, snapshot = batch(agent)
    cache = snapshot.requires_grad_() if features == "rollout" else None
    model.policy_loss(agent, *data, args, critic_features=cache)[0].backward()
    policy, representation, reward = agent.parameter_groups()
    assert_no_grad(representation + reward)
    if cache is not None:
        assert cache.grad is None
    for network in (agent.actor, agent.critic):
        assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in network.parameters())
    groups = [set(map(id, group)) for group in (policy, representation, reward)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert set.union(*groups) == set(map(id, agent.parameters()))


def test_cached_critic_stays_fixed_while_online_actor_and_inference_move(envs):
    agent = make_agent(envs, args_for(critic_feature_updates="rollout"))
    data, snapshot = batch(agent)
    observations, actions = data[:2]
    assert not snapshot.requires_grad
    before = agent.get_policy_value_latents(observations, snapshot)
    with torch.no_grad():
        agent.encoder[0].weight.mul_(0.3)
        agent.encoder[2].bias.add_(0.7)
    after = agent.get_policy_value_latents(observations, snapshot)
    torch.testing.assert_close(after[2], before[2], rtol=0, atol=0)
    assert not torch.equal(after[0], before[0])
    assert not torch.equal(after[3], before[3])
    assert not torch.equal(agent.get_value(observations), before[2])
    new_values, _, new_snapshot = model.rollout_statistics(agent, observations, actions)
    torch.testing.assert_close(new_values, agent.get_value(observations).flatten())
    assert not torch.equal(new_snapshot, snapshot)
    # The next rollout deliberately adopts new coordinates; this is not a long-term drift cure.
    torch.testing.assert_close(agent.get_policy_and_value(observations, new_snapshot)[2].flatten(), new_values)
    model.policy_loss(agent, *data, args_for(critic_feature_updates="rollout"),
                      critic_features=snapshot)[0].backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.critic.parameters())


@pytest.mark.parametrize("mode", ["actor", "both"])
def test_drift_holds_current_head_fixed_and_does_not_consume_rng(envs, mode):
    agent = make_agent(envs, args_for(jepa_mode=mode))
    data, snapshot = batch(agent, rows=256)
    rng_before = torch.cuda.get_rng_state().clone()
    indices = model.drift_probe_indices(data[0].shape[0], "cuda")
    torch.testing.assert_close(indices, model.drift_probe_indices(256, "cuda"), rtol=0, atol=0)
    observations, pre_features = data[0][indices], snapshot[indices]
    with torch.no_grad():
        # Critic-only changes must not masquerade as encoder-coordinate drift.
        agent.critic.head[0].weight.mul_(2.0)
        unchanged = model.coordinate_drift(agent, observations, pre_features)
        # Rollout/probe GEMMs use different row counts; tolerate FP32 rounding only.
        torch.testing.assert_close(unchanged["drift/representation_relative_rms"],
                                   snapshot.new_zeros(()), rtol=0, atol=1e-6)
        torch.testing.assert_close(unchanged["drift/critic_coordinate_value_rms"],
                                   snapshot.new_zeros(()), rtol=0, atol=1e-6)
        agent.encoder[2].bias.add_(0.4)
        post_features = agent.encoder(observations)
        # Change head again: the counterfactual must use this head on BOTH sets of features.
        agent.critic.head[0].weight.mul_(1.7)
        actual = model.coordinate_drift(agent, observations, pre_features)
        expected_representation = ((post_features - pre_features).square().mean()
                                   / pre_features.square().mean()).sqrt()
        torch.testing.assert_close(actual["drift/representation_relative_rms"], expected_representation)
        assert actual["drift/representation_relative_rms"] > 0
        assert actual["drift/critic_uses_encoder"] == float(mode == "both")
        if mode == "both":
            pre_value, post_value = agent.critic(torch.cat((pre_features, post_features))).chunk(2)
            expected = (post_value - pre_value).square().mean().sqrt()
            torch.testing.assert_close(actual["drift/critic_coordinate_value_rms"], expected)
            torch.testing.assert_close(actual["drift/critic_coordinate_value_relative_rms"],
                                       expected / pre_value.square().mean().sqrt())
            assert expected > 0
        else:
            assert actual["drift/critic_coordinate_value_rms"] == 0
            assert actual["drift/critic_coordinate_value_relative_rms"] == 0
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before, rtol=0, atol=0)
    assert all(not value.requires_grad for value in actual.values())


@pytest.mark.parametrize("activation,residual,projection", [("tanh", False, "none"), ("stiglu", True, "all")])
def test_actor_both_initialization_is_paired_for_all_shared_owners(envs, activation, residual, projection):
    shared = dict(task_activation=activation, task_residual=residual, weight_projection=projection)
    both = make_agent(envs, args_for(jepa_mode="both", **shared))
    rng_after_both = torch.get_rng_state().clone()
    actor = make_agent(envs, args_for(jepa_mode="actor", **shared))
    torch.testing.assert_close(torch.get_rng_state(), rng_after_both, rtol=0, atol=0)
    for name in ("actor", "encoder", "ssl", "reward_head"):
        for key, tensor in getattr(both, name).state_dict().items():
            torch.testing.assert_close(getattr(actor, name).state_dict()[key], tensor, rtol=0, atol=0)
    treatment = make_agent(envs, args_for(prediction_loss="huber", prediction_target_gradient="stopped",
                                         critic_feature_updates="rollout", reward_mode="attached", **shared))
    torch.testing.assert_close(torch.get_rng_state(), rng_after_both, rtol=0, atol=0)
    for key, tensor in both.state_dict().items():
        torch.testing.assert_close(treatment.state_dict()[key], tensor, rtol=0, atol=0)


def test_default_both_matches_v6_detached_control_loss_gradients_and_update(envs):
    args = args_for()
    torch.manual_seed(13)
    old = reference.Agent(envs, reference.Args(reward_mode="detached", sigreg_num_proj=16, sigreg_proj_chunk=8)).cuda()
    old_rng = torch.get_rng_state().clone()
    agent = make_agent(envs, args)
    torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
    for key, tensor in old.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[key], tensor, rtol=0, atol=0)
    data, _ = batch(agent)
    following, rewards = torch.randn_like(data[0]), torch.randn(32, device="cuda")
    old_ppo, _ = reference.policy_loss(old, *data, args)
    actual_ppo, _ = model.policy_loss(agent, *data, args)
    torch.manual_seed(29)
    old_components, old_metrics = reference.representation_components(old, data[0], data[1], following, rewards, args)
    rng_after_old = torch.cuda.get_rng_state().clone()
    torch.manual_seed(29)
    actual_components, actual_metrics = model.representation_components(agent, data[0], data[1], following, rewards, args)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after_old, rtol=0, atol=0)
    torch.testing.assert_close(actual_ppo, old_ppo, rtol=0, atol=0)
    torch.testing.assert_close(actual_components, old_components, rtol=0, atol=0)
    for name, value in old_metrics.items():
        torch.testing.assert_close(actual_metrics[name], value, rtol=0, atol=0)
    (old_ppo + old_components.sum()).backward()
    (actual_ppo + actual_components.sum()).backward()
    for actual, expected in zip(agent.parameters(), old.parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    reference.optimizer_step(old, optimizers_for(old, args), args)
    model.optimizer_step(agent, optimizers_for(agent, args), args)
    for actual, expected in zip(agent.parameters(), old.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rollout_cache_does_not_change_terminal_bootstrapping_or_trace_boundaries(envs):
    agent = make_agent(envs, args_for(critic_feature_updates="rollout"))
    observations = torch.randn(2, 4, 17, device="cuda")
    actions = torch.full((8, 6), 0.5, device="cuda")
    values, _, _ = model.rollout_statistics(agent, observations.flatten(0, 1), actions)
    values = values.reshape(2, 4)
    finals = np.arange(68, dtype=np.float32).reshape(4, 17) / 13
    terms = torch.tensor([[1., 0., 1., 0.], [0., 0., 0., 0.]], device="cuda")
    truncs = torch.tensor([[0., 1., 1., 0.], [0., 0., 0., 0.]], device="cuda")
    cache = model.TruncationBootstrapCache(2, 4, (17,))
    cache.push_normalized(0, np.array([False, True, True, False]), finals)
    rewards = torch.tensor([[1., 2., 3., 4.], [100., 200., 300., 400.]], device="cuda")
    gamma, lam = 0.9, 0.7
    with torch.no_grad():
        final_values = cache.resolve(agent.get_value, torch.device("cuda"))
        tail = agent.get_value(torch.full((4, 17), -3.0, device="cuda")).flatten()
        expected_final = agent.get_value(torch.as_tensor(finals, device="cuda")).flatten()
        torch.testing.assert_close(final_values[0, 1:3], expected_final[1:3])
        advantages, returns = model.get_gae_fn(compiled=False)(
            rewards, values, terms, truncs, final_values, tail, gamma, lam,
        )
        expected_last = rewards[1] + gamma * tail - values[1]
        torch.testing.assert_close(advantages[1], expected_last)
        expected_first = rewards[0] - values[0]
        expected_first[1] += gamma * expected_final[1]
        expected_first[3] += gamma * values[1, 3] + gamma * lam * expected_last[3]
        torch.testing.assert_close(advantages[0], expected_first)
        torch.testing.assert_close(returns, advantages + values)
    cache.close()


@pytest.mark.parametrize("kwargs", [
    {"jepa_mode": "actor", "critic_feature_updates": "rollout"},
    {"jepa_mode": "none"},
    {"prediction_loss": "cosine"},
    {"prediction_target_gradient": "ema"},
    {"critic_feature_updates": "frozen"},
])
def test_invalid_treatment_cannot_silently_select_a_different_experiment(envs, kwargs):
    with pytest.raises(ValueError):
        model.validate_args(args_for(**kwargs))
    with pytest.raises(ValueError):
        model.Agent(envs, args_for(**kwargs))


@pytest.mark.parametrize("loss_kind,target_gradient,features", [
    ("mse", "attached", "online"), ("huber", "stopped", "rollout"),
])
def test_fullgraph_real_joint_loss_backward_and_cached_policy_on_cuda(envs, loss_kind, target_gradient, features):
    args = args_for(prediction_loss=loss_kind, prediction_target_gradient=target_gradient,
                    critic_feature_updates=features)
    agent = make_agent(envs, args)
    data, snapshot = batch(agent, rows=512)
    following, rewards = torch.randn_like(data[0]), torch.randn(512, device="cuda")
    cache = snapshot if features == "rollout" else None

    def objective(observations, actions, logprobs, advantages, returns, values, next_obs, real_rewards, cached):
        return model.joint_loss(agent, observations, actions, logprobs, advantages, returns, values,
                                next_obs, real_rewards, args, critic_features=cached)

    expected_policy, expected_components, _, _ = objective(*data, following, rewards, cache)
    (expected_policy + expected_components[2]).backward()
    policy, representation, reward = agent.parameter_groups()
    expected_gradients = [parameter.grad.clone() for parameter in policy + reward]
    assert_no_grad(representation)
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    actual_policy, components, _, metrics = compiled(*data, following, rewards, cache)
    torch.testing.assert_close(actual_policy, expected_policy, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(components[[0, 2]], expected_components[[0, 2]], rtol=3e-5, atol=3e-6)
    (actual_policy + components.sum()).backward()
    for parameter, expected in zip(policy + reward, expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=5e-4, atol=5e-6)
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.encoder.parameters())
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    before = [[parameter.detach().clone() for parameter in group] for group in (policy, representation, reward)]
    model.optimizer_step(agent, optimizers_for(agent, args), args)
    for group, previous in zip((policy, representation, reward), before, strict=True):
        assert any(not torch.equal(parameter, old) for parameter, old in zip(group, previous, strict=True))

    compiled_policy = torch.compile(agent.get_policy_and_value, fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False})
    with torch.no_grad():
        before_outputs = compiled_policy(data[0], cache)
        agent.encoder[2].bias.add_(0.6)
        after_outputs = compiled_policy(data[0], cache)
        assert not torch.equal(after_outputs[0], before_outputs[0])
        if cache is not None:
            torch.testing.assert_close(after_outputs[2], before_outputs[2], rtol=0, atol=0)
        else:
            assert not torch.equal(after_outputs[2], before_outputs[2])
        compiled_statistics = torch.compile(lambda obs, native: model.rollout_statistics(agent, obs, native),
                                             fullgraph=True, options={"triton.cudagraphs": False})
        new_values, _, new_features = compiled_statistics(data[0], data[1])
        torch.testing.assert_close(new_values, agent.get_value(data[0]).flatten(), rtol=3e-5, atol=3e-6)
        assert not new_features.requires_grad
        compiled_drift = torch.compile(lambda obs, pre: model.coordinate_drift(agent, obs, pre),
                                        fullgraph=True, options={"triton.cudagraphs": False})
        measured = compiled_drift(data[0][:128], snapshot[:128])
        assert measured["drift/representation_relative_rms"] > 0
        assert measured["drift/critic_coordinate_value_rms"] > 0
