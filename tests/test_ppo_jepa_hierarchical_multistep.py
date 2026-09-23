"""CUDA scientific contracts for hierarchical v10; execute only through mlq."""
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_jepa_geometry_drift_v7 as reference
from cleanrl import ppo_continuous_action_jepa_hierarchical_multistep_v10 as model
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


def assert_no_grad(parameters):
    assert all(parameter.grad is None or not bool(parameter.grad.any()) for parameter in parameters)


def assert_zero_or_unused(gradients):
    assert all(gradient is None or not bool(gradient.any()) for gradient in gradients)


def world_batch(agent, rows=512):
    observations = torch.randn(rows, 17, device="cuda", requires_grad=True)
    actions = torch.rand(rows, agent.ssl.max_horizon, 6, device="cuda", requires_grad=True)
    targets = torch.randn(rows, len(agent.ssl.target_horizons), 17, device="cuda", requires_grad=True)
    valid = torch.ones(rows, len(agent.ssl.target_horizons), dtype=torch.bool, device="cuda")
    rewards = torch.randn(rows, device="cuda", requires_grad=True)
    return observations, actions, targets, valid, rewards


def optimizers_for(agent, args):
    policy, representation, reward = agent.parameter_groups()
    return (
        torch.optim.Adam(policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
        torch.optim.AdamW(reward, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
    )


def test_windows_use_factual_endpoints_exclude_terminal_endpoint_and_preserve_env():
    steps, env_count, action_dim = 7, 3, 2
    horizons = (1, 2, 4, 8)
    terms = torch.zeros(steps, env_count, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[2, 0] = 1
    truncs[3, 1] = 1
    terms[-1, 2] = 1
    windows = model.build_window_indices(terms, truncs, horizons)
    actions = torch.arange(steps * env_count * action_dim, device="cuda").reshape(-1, action_dim)
    # These are factual terminal observations, deliberately unlike reset states.
    factual = 1000 + torch.arange(steps * env_count, device="cuda")[:, None]
    sequence, targets, valid = model.gather_windows(actions, factual, windows)
    expected_valid, expected_targets, expected_actions = [], [], []
    boundaries = {(2, 0), (3, 1), (6, 2)}
    for time in range(steps):
        for env in range(env_count):
            expected_valid.append([
                time + horizon <= steps
                and not any((crossing, env) in boundaries for crossing in range(time, time + horizon - 1))
                for horizon in horizons
            ])
            expected_targets.append([min(time + horizon - 1, steps - 1) * env_count + env for horizon in horizons])
            expected_actions.append([min(time + offset, steps - 1) * env_count + env for offset in range(max(horizons))])
    torch.testing.assert_close(valid, torch.tensor(expected_valid, device="cuda"))
    torch.testing.assert_close(targets, factual[torch.tensor(expected_targets, device="cuda")])
    torch.testing.assert_close(sequence, actions[torch.tensor(expected_actions, device="cuda")])
    assert bool(valid[:, 0].all())
    assert bool(valid[1 * env_count + 0, 1])  # Arrives at terminal transition 2.
    assert not bool(valid[2 * env_count + 0, 1])  # Crosses its reset.
    assert bool(valid[0 * env_count + 1, 2])  # Arrives at truncated transition 3.
    assert not bool(valid[1 * env_count + 1, 2])
    assert not bool(valid[:, -1].any())  # No fabricated rollout-tail endpoints.
    torch.testing.assert_close(windows.action_indices % env_count,
                               torch.arange(steps * env_count, device="cuda")[:, None].expand_as(windows.action_indices) % env_count)


def test_ordered_chunks_and_both_recurrences_match_composed_step_math(envs):
    args = args_for(fine_horizons=(1, 4, 8), coarse_horizons=(4, 8))
    agent = make_agent(envs, args)
    branch = agent.ssl
    # AdaLN-zero initially masks action effects; activate gates to test ordering.
    with torch.no_grad():
        branch.predictor.modulation[-1].weight.normal_(std=0.04)
        branch.coarse_predictor.modulation[-1].weight.normal_(std=0.04)
    fine = torch.randn(8, 64, device="cuda", requires_grad=True)
    coarse = torch.randn_like(fine, requires_grad=True)
    actions = torch.rand(8, 8, 6, device="cuda", requires_grad=True)
    actual_fine, actual_coarse = branch.rollouts(fine, coarse, actions)
    fine_step, coarse_step = fine, coarse
    for horizon in range(1, 9):
        condition = branch.action_encoder(2 * actions[:, horizon - 1].detach() - 1)
        fine_step = branch.pred_proj(branch.predictor(fine_step, condition))
        if horizon in branch.target_horizons:
            torch.testing.assert_close(actual_fine[horizon], fine_step)
    for horizon in (4, 8):
        ordered_chunk = (2 * actions[:, horizon - 4:horizon].detach() - 1).flatten(1)
        condition = branch.coarse_action_encoder(ordered_chunk)
        coarse_step = branch.coarse_pred_proj(branch.coarse_predictor(coarse_step, condition))
        torch.testing.assert_close(actual_coarse[horizon], coarse_step)
    permuted = actions.detach().reshape(8, 2, 4, 6).flip(2).reshape(8, 8, 6)
    _, reordered_coarse = branch.rollouts(fine, coarse, permuted)
    assert not torch.allclose(actual_coarse[8], reordered_coarse[8])
    (actual_fine[8].square().mean() + actual_coarse[8].square().mean()).backward()
    assert fine.grad.norm() > 0 and coarse.grad.norm() > 0
    assert_no_grad((actions,))


@pytest.mark.parametrize("loss_kind", ["mse", "huber"])
def test_masked_losses_ignore_invalid_rows_and_empty_horizons_without_gradient(loss_kind):
    prediction = torch.tensor([[3.0, -2.0], [9.0, -8.0], [0.5, -0.5]], device="cuda", requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [-7.0, 6.0], [0.0, 0.0]], device="cuda", requires_grad=True)
    valid = torch.tensor([True, False, True], device="cuda")
    loss = model.masked_prediction_objective(prediction, target, valid, loss_kind)
    expected = (F.mse_loss(prediction[valid], target[valid]) if loss_kind == "mse"
                else 2 * F.smooth_l1_loss(prediction[valid], target[valid], beta=1.0))
    torch.testing.assert_close(loss, expected)
    gradients = torch.autograd.grad(loss, (prediction, target), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (prediction, target), retain_graph=True)
    for actual, reference_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference_gradient)
        assert not bool(actual[~valid].any())
    empty = model.masked_prediction_objective(prediction, target, torch.zeros_like(valid), loss_kind)
    assert empty == 0
    assert_zero_or_unused(torch.autograd.grad(empty, (prediction, target)))


@pytest.mark.parametrize("loss_kind", ["mse", "huber"])
def test_observed_stop_after_each_projector_leaves_h1_sigreg_attached_and_teacher_stopped(envs, loss_kind):
    consistency_gradients, regularization_gradients = {}, {}
    for attachment in ("attached", "stopped"):
        args = args_for(prediction_loss=loss_kind, prediction_target_gradient=attachment)
        agent = make_agent(envs, args)
        with torch.no_grad():
            agent.ssl.predictor.modulation[-1].weight.normal_(std=0.04)
            agent.ssl.coarse_predictor.modulation[-1].weight.normal_(std=0.04)
        data = world_batch(agent)
        current, actions, targets, valid, rewards = data
        # Invalid long windows are still fixed-shaped, but cannot train targets.
        valid[256:, 1:] = False
        fine_outputs, coarse_outputs, teacher_outputs, fine_predictions = [], [], [], []
        handles = [
            agent.ssl.projector.register_forward_hook(lambda module, inputs, output: fine_outputs.append(output)),
            agent.ssl.coarse_projector.register_forward_hook(lambda module, inputs, output: coarse_outputs.append(output)),
            agent.ssl.coarse_pred_proj.register_forward_hook(lambda module, inputs, output: teacher_outputs.append(output)),
            agent.ssl.pred_proj.register_forward_hook(lambda module, inputs, output: fine_predictions.append(output)),
        ]
        components, metrics = model.representation_components(agent, *data, args)
        for handle in handles:
            handle.remove()
        fine, coarse = fine_outputs[0], coarse_outputs[0]
        anchor_gradients = torch.autograd.grad(components[0], (current, targets, fine, coarse), retain_graph=True)
        assert anchor_gradients[0].norm() > 0
        if attachment == "stopped":
            assert not bool(anchor_gradients[1].any())
            assert not bool(anchor_gradients[2][:, 1:].any())
            assert not bool(anchor_gradients[3][:, 1:].any())
        else:
            assert anchor_gradients[1][:256].norm() > 0
            assert anchor_gradients[2][:256, 1:].norm() > 0
            assert anchor_gradients[3][:256, 1:].norm() > 0
        assert not bool(anchor_gradients[1][256:, 1:].any())
        reg_gradients = torch.autograd.grad(components[1], (current, targets), retain_graph=True)
        assert reg_gradients[0].norm() > 0 and reg_gradients[1][:, 0].norm() > 0
        assert not bool(reg_gradients[1][:, 1:].any())
        regularization_gradients[attachment] = reg_gradients
        fine_parameters = tuple(agent.encoder.parameters()) + tuple(agent.ssl.projector.parameters()) + tuple(agent.ssl.predictor.parameters()) + tuple(agent.ssl.pred_proj.parameters())
        coarse_projector = tuple(agent.ssl.coarse_projector.parameters())
        coarse_dynamics = tuple(agent.ssl.coarse_action_encoder.parameters()) + tuple(agent.ssl.coarse_predictor.parameters()) + tuple(agent.ssl.coarse_pred_proj.parameters())
        fine_grads = torch.autograd.grad(components[3], fine_parameters, retain_graph=True)
        projector_grads = torch.autograd.grad(components[3], coarse_projector, retain_graph=True)
        assert sum(gradient.square().sum() for gradient in fine_grads) > 0
        assert sum(gradient.square().sum() for gradient in projector_grads) > 0
        action_gradients = torch.autograd.grad(components[3], tuple(agent.ssl.action_encoder.parameters()), retain_graph=True)
        assert sum(gradient.square().sum() for gradient in action_gradients) > 0
        assert_zero_or_unused(torch.autograd.grad(components[3], coarse_dynamics + tuple(teacher_outputs) + (targets,), retain_graph=True, allow_unused=True))
        consistency_gradients[attachment] = fine_grads + projector_grads
        # The consistency derivative must equal a stopped coarse teacher even
        # when the observed-anchor target attachment setting is "attached".
        agreements = []
        for index, horizon in enumerate(args.coarse_horizons):
            agreements.append(model.masked_prediction_objective(
                coarse_outputs[1][:, index], teacher_outputs[horizon // args.coarse_stride - 1].detach(),
                valid[:, agent.ssl.target_horizons.index(horizon)], loss_kind,
            ))
        expected_consistency = torch.stack(agreements).mean()
        torch.testing.assert_close(components[3], expected_consistency)
        expected_grads = torch.autograd.grad(expected_consistency, fine_parameters + coarse_projector, retain_graph=True)
        for actual, expected in zip(consistency_gradients[attachment], expected_grads, strict=True):
            torch.testing.assert_close(actual, expected)
        for prediction in (fine_predictions[3], fine_predictions[15], teacher_outputs[0], teacher_outputs[-1]):
            gradient, = torch.autograd.grad(components[0] + components[3], (prediction,), retain_graph=True)
            assert not bool(gradient[256:].any())
        # Same-forward persistence and zero baselines use exactly the anchor mask.
        level_anchors = []
        for level, embeddings, horizons, predictions, stride in (
            ("fine", fine, args.fine_horizons, fine_predictions, 1),
            ("coarse", coarse, args.coarse_horizons, teacher_outputs, args.coarse_stride),
        ):
            horizon_anchors = []
            for horizon in horizons:
                index = agent.ssl.target_horizons.index(horizon)
                keep = valid[:, index]
                observed = embeddings[:, index + 1].detach()
                predicted = predictions[horizon // stride - 1].detach()
                anchor = (F.mse_loss(predicted[keep], observed[keep]) if loss_kind == "mse" else
                          2 * F.smooth_l1_loss(predicted[keep], observed[keep], beta=1.0))
                horizon_anchors.append(anchor)
                torch.testing.assert_close(metrics[f"ssl/{level}_h{horizon}_anchor_loss"], anchor)
                torch.testing.assert_close(metrics[f"ssl/{level}_h{horizon}_raw_mse"],
                                           (predicted[keep] - observed[keep]).square().mean())
                torch.testing.assert_close(metrics[f"ssl/{level}_h{horizon}_persistence_mse"],
                                           (embeddings[keep, 0].detach() - observed[keep]).square().mean())
                torch.testing.assert_close(metrics[f"ssl/{level}_h{horizon}_zero_mse"], observed[keep].square().mean())
            level_anchors.append(torch.stack(horizon_anchors).mean())
        torch.testing.assert_close(components[0], 0.5 * sum(level_anchors))
        assert_zero_or_unused(torch.autograd.grad(components.sum(), (actions, rewards), retain_graph=True, allow_unused=True))
        components[2].backward()
        assert_no_grad(agent.parameter_groups()[0] + agent.parameter_groups()[1])
        assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.reward_head.parameters())
        assert all(not metric.requires_grad for metric in metrics.values())
    for values in (consistency_gradients, regularization_gradients):
        for attached, stopped in zip(values["attached"], values["stopped"], strict=True):
            torch.testing.assert_close(attached, stopped, rtol=0, atol=0)


@pytest.mark.parametrize("activation,residual,projection", [("tanh", False, "none"), ("stiglu", True, "all")])
def test_off_on_and_raw_critic_keep_canonical_initialization_and_rng(envs, activation, residual, projection):
    args = args_for(task_activation=activation, task_residual=residual, weight_projection=projection)
    coupled = make_agent(envs, args)
    coupled_rng = torch.get_rng_state().clone()
    uncoupled = make_agent(envs, replace(args, hierarchy_consistency="off"))
    torch.testing.assert_close(torch.get_rng_state(), coupled_rng, rtol=0, atol=0)
    for key, value in coupled.state_dict().items():
        torch.testing.assert_close(value, uncoupled.state_dict()[key], rtol=0, atol=0)
    assert coupled.parameter_counts() == uncoupled.parameter_counts()
    raw = make_agent(envs, replace(args, jepa_mode="actor"))
    for key, value in coupled.state_dict().items():
        if not key.startswith("critic."):
            torch.testing.assert_close(value, raw.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(13)
    frozen = reference.Agent(envs, reference.Args(
        sigreg_num_proj=16, sigreg_proj_chunk=8, task_activation=activation,
        task_residual=residual, weight_projection=projection,
    )).cuda()
    torch.testing.assert_close(torch.get_rng_state(), coupled_rng, rtol=0, atol=0)
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(coupled.state_dict()[key], value, rtol=0, atol=0)


def test_off_on_same_anchors_two_fixed_marginal_draws_and_same_forward_diagnostics(envs):
    args = args_for()
    on = make_agent(envs, args)
    off = make_agent(envs, replace(args, hierarchy_consistency="off"))
    data = world_batch(on)
    data[3][256:, 1:] = False
    captured = []
    hooks = [on.ssl.sigreg.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0])),
             on.ssl.coarse_sigreg.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))]
    rng_before = torch.cuda.get_rng_state().clone()
    on_components, on_metrics = model.representation_components(on, *data, args)
    rng_after = torch.cuda.get_rng_state().clone()
    for hook in hooks:
        hook.remove()
    assert [tuple(tensor.shape) for tensor in captured] == [(2, 512, 64), (2, 512, 64)]
    torch.cuda.set_rng_state(rng_before)
    off_components, off_metrics = model.representation_components(off, *data, replace(args, hierarchy_consistency="off"))
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after, rtol=0, atol=0)
    torch.testing.assert_close(on_components[:3], off_components[:3], rtol=0, atol=0)
    assert off_components[3] == 0 and on_components[3] > 0
    for name in on_metrics:
        if name != "ssl/shared_consistency_loss":
            torch.testing.assert_close(on_metrics[name], off_metrics[name], rtol=0, atol=0)
    torch.cuda.set_rng_state(rng_before)
    with torch.no_grad():
        fine_reg = on.ssl.sigreg(captured[0].detach())
        coarse_reg = on.ssl.coarse_sigreg(captured[1].detach())
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after, rtol=0, atol=0)
    torch.testing.assert_close(on_components[1], 0.09 * 0.5 * (fine_reg + coarse_reg))
    expected_anchor = 0.5 * (
        torch.stack([on_metrics[f"ssl/fine_h{horizon}_anchor_loss"] for horizon in args.fine_horizons]).mean()
        + torch.stack([on_metrics[f"ssl/coarse_h{horizon}_anchor_loss"] for horizon in args.coarse_horizons]).mean()
    )
    torch.testing.assert_close(on_components[0], expected_anchor)
    assert set(on_metrics) == set(model.representation_metric_names(args))
    balance = model.gradient_balance(on, on_components)
    off_balance = model.gradient_balance(off, off_components)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after, rtol=0, atol=0)
    assert balance["balance/shared_consistency_grad_norm"] > 0
    assert off_balance["balance/shared_consistency_grad_norm"] == 0
    assert balance["balance/shared_reward_grad_norm"] == 0
    gradients = [torch.autograd.grad(component, tuple(on.encoder.parameters()), retain_graph=True)
                 for component in (on_components[0], on_components[3])]
    flat = [torch.cat([gradient.flatten() for gradient in row]) for row in gradients]
    torch.testing.assert_close(balance["balance/shared_consistency_grad_norm"], flat[1].norm())
    torch.testing.assert_close(balance["balance/shared_prediction_consistency_grad_cosine"],
                               F.cosine_similarity(flat[0], flat[1], dim=0))
    assert_no_grad(tuple(on.parameters()) + tuple(off.parameters()))
    (on_components.sum()).backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in on.encoder.parameters())


@pytest.mark.parametrize("mode,features", [("actor", "online"), ("both", "online"), ("both", "rollout")])
def test_ppo_detaches_encoder_and_cache_and_never_uses_future_training_actions(envs, mode, features):
    args = args_for(jepa_mode=mode, critic_feature_updates=features, reward_mode="attached")
    agent = make_agent(envs, args)
    observations, sequence, targets, valid, rewards = world_batch(agent)
    native = sequence[:, 0].detach().clamp(0.1, 0.9)
    values, logprobs, snapshot = model.rollout_statistics(agent, observations, native)
    cache = snapshot.clone().requires_grad_() if features == "rollout" else None
    ppo, components, _, _ = model.joint_loss(
        agent, observations, native, logprobs, torch.randn_like(values), values + 1, values,
        sequence, targets, valid, rewards, args, critic_features=cache,
    )
    ppo.backward()
    assert_no_grad(agent.parameter_groups()[1] + agent.parameter_groups()[2])
    assert_no_grad((sequence, targets, rewards))
    if mode == "both":
        assert_no_grad((observations,))
    else:
        assert observations.grad.norm() > 0  # The raw critic still differentiates its raw input.
    if cache is not None:
        assert_no_grad((cache,))
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.actor.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.critic.parameters())
    groups = [set(map(id, group)) for group in agent.parameter_groups()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(map(id, agent.parameters()))


def test_fullbatch_ppo_retains_ssl512_and_320_exposures_without_dropping_tail_rows():
    for ppo_size in (512, 16384):
        generator = torch.Generator(device="cuda").manual_seed(41)
        policy_steps = ssl_steps = 0
        for _ in range(10):
            seen = torch.zeros(16384, dtype=torch.long, device="cuda")
            for ppo_rows, ssl_rows in model.iter_update_batches(16384, ppo_size, 512, "cuda", generator):
                assert ssl_rows.numel() == 512
                seen.index_add_(0, ssl_rows, torch.ones_like(ssl_rows))
                ssl_steps += 1
                if ppo_rows is not None:
                    assert ppo_rows.numel() == ppo_size
                    torch.testing.assert_close(ssl_rows, ppo_rows[:512], rtol=0, atol=0)
                    policy_steps += 1
            torch.testing.assert_close(seen, torch.ones_like(seen), rtol=0, atol=0)
        assert ssl_steps == 320
        assert policy_steps == (320 if ppo_size == 512 else 10)


@pytest.mark.parametrize("loss_kind,attachment,features,consistency", [
    ("mse", "attached", "online", "on"), ("huber", "stopped", "rollout", "off"),
])
def test_compiled_cuda_real_fullbatch_joint_and_world_only_update(envs, loss_kind, attachment, features, consistency):
    args = model.validate_args(args_for(
        num_envs=16, num_steps=1024, num_minibatches=1, prediction_loss=loss_kind,
        prediction_target_gradient=attachment, critic_feature_updates=features, hierarchy_consistency=consistency,
    ))
    agent = make_agent(envs, args)
    observations = torch.randn(args.batch_size, 17, device="cuda")
    native = torch.rand(args.batch_size, 6, device="cuda") * 0.8 + 0.1
    following = torch.randn_like(observations)
    terms = torch.zeros(1024, 16, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[15, 0] = 1
    truncs[-1, 1] = 1
    windows = model.build_window_indices(terms, truncs, agent.ssl.target_horizons)
    sequences, targets, valid = model.gather_windows(native, following, windows)
    values, logprobs, snapshot = model.rollout_statistics(agent, observations, native)
    cache = snapshot if features == "rollout" else None
    advantages = torch.randn_like(values)
    returns = values + torch.randn_like(values)
    rewards = torch.randn_like(values)
    row = (observations, native, logprobs, advantages, returns, values,
           sequences[:512], targets[:512], valid[:512], rewards[:512], cache)

    def joint(obs, actions, old_logprobs, adv, ret, old_values, action_sequence, endpoints, validity, real_rewards, cached):
        return model.joint_loss(agent, obs, actions, old_logprobs, adv, ret, old_values,
                                action_sequence, endpoints, validity, real_rewards, args, critic_features=cached)

    expected_policy, expected_components, _, _ = joint(*row)
    (expected_policy + expected_components[2]).backward()
    policy, representation, reward = agent.parameter_groups()
    expected_gradients = [parameter.grad.clone() for parameter in policy + reward]
    assert_no_grad(representation)
    agent.zero_grad(set_to_none=True)
    compiled_joint = torch.compile(joint, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    actual_policy, components, _, metrics = compiled_joint(*row)
    torch.testing.assert_close(actual_policy, expected_policy, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(components[[0, 2, 3]], expected_components[[0, 2, 3]], rtol=5e-5, atol=5e-6)
    (actual_policy + components.sum()).backward()
    for parameter, expected in zip(policy + reward, expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=5e-4, atol=5e-6)
    assert all(bool(torch.isfinite(metric)) for metric in metrics.values())
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.encoder.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in agent.ssl.coarse_predictor.parameters())
    owners = optimizers_for(agent, args)
    model.optimizer_step(agent, owners, args)
    before_world = [[parameter.detach().clone() for parameter in group] for group in (policy, representation, reward)]
    for owner in owners:
        owner.zero_grad(set_to_none=True)

    def world(obs, action_sequence, endpoints, validity, real_rewards):
        return model.representation_components(agent, obs, action_sequence, endpoints, validity, real_rewards, args)

    compiled_world = torch.compile(world, fullgraph=True, dynamic=False, mode="reduce-overhead")
    torch.compiler.cudagraph_mark_step_begin()
    world_components, world_metrics = compiled_world(observations[512:1024], sequences[512:1024],
                                                     targets[512:1024], valid[512:1024], rewards[512:1024])
    world_components.sum().backward()
    model.optimizer_step(agent, owners, args, policy_step=False)
    for parameter, expected in zip(policy, before_world[0], strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    for group, previous in zip((representation, reward), before_world[1:], strict=True):
        assert any(not torch.equal(parameter, old) for parameter, old in zip(group, previous, strict=True))
    assert set(world_metrics) == set(model.representation_metric_names(args))
    with torch.no_grad():
        before = agent.get_policy_and_value(observations, cache)
        agent.encoder[2].bias.add_(0.6)
        after = agent.get_policy_and_value(observations, cache)
        assert not torch.equal(before[0], after[0])
        if cache is not None:
            torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
        else:
            assert not torch.equal(before[2], after[2])


def test_drift_probe_covers_envs_time_and_small_rollout_without_rng_draw():
    before = torch.cuda.get_rng_state().clone()
    indices = model.drift_probe_indices(1024, 16, "cuda")
    torch.testing.assert_close(torch.cuda.get_rng_state(), before, rtol=0, atol=0)
    assert indices.unique().numel() == 128
    torch.testing.assert_close(torch.bincount(indices % 16, minlength=16),
                               torch.full((16,), 8, device="cuda", dtype=torch.long))
    times = indices // 16
    assert int(times.min()) < 16 and int(times.max()) >= 1000
    assert (times // 128).unique().numel() == 8
    small = model.drift_probe_indices(3, 7, "cuda")
    torch.testing.assert_close(small.sort().values, torch.arange(21, device="cuda"), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"fine_horizons": ()}, {"fine_horizons": (4, 16)}, {"fine_horizons": (1, 4, 4, 16)},
    {"fine_horizons": (1, 17)}, {"coarse_horizons": ()}, {"coarse_horizons": (3, 16)},
    {"coarse_stride": 0}, {"hierarchy_consistency": "unknown"},
])
def test_invalid_horizon_or_treatment_contract_is_rejected_before_model_construction(envs, kwargs):
    with pytest.raises(ValueError):
        model.validate_args(args_for(**kwargs))
    with pytest.raises(ValueError):
        model.Agent(envs, args_for(**kwargs))
