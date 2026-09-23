"""CUDA scientific contracts for flat multi-step v9; run only through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_jepa_geometry_drift_v7 as reference
from cleanrl import ppo_continuous_action_jepa_multistep_v9 as model
from cleanrl.shared.rollout_transfer import RolloutTransfer
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


def make_batch(agent, horizons=(1, 4, 16), steps=32, env_count=16):
    rows = steps * env_count
    observations = torch.randn(rows, 17, device="cuda")
    factual_next = torch.randn_like(observations)
    native = torch.rand(rows, 6, device="cuda") * 0.8 + 0.1
    terms = torch.zeros((steps, env_count), dtype=torch.bool, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[0, 0] = True
    terms[steps // 2, 1] = True
    truncs[2, 2] = True
    terms[3, 3] = truncs[3, 3] = True
    windows = model.prediction_windows(terms, truncs, horizons)
    sequences, targets, validity = model.gather_prediction_windows(native, factual_next, windows)
    rewards = torch.randn(rows, device="cuda")
    return observations, native, targets, rewards, sequences, validity


def ppo_data(agent, observations, native):
    values, logprobs, features = model.rollout_statistics(agent, observations, native)
    rows = observations.shape[0]
    return ((observations, native, logprobs, torch.randn(rows, device="cuda"),
             values + torch.randn(rows, device="cuda"), values), features)


def make_optimizers(agent, args):
    policy, representation, reward = agent.parameter_groups()
    return (
        torch.optim.Adam(policy, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.AdamW(representation, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
        torch.optim.AdamW(reward, lr=args.ssl_learning_rate,
                          weight_decay=args.ssl_weight_decay, fused=True),
    )


def assert_no_grad(parameters):
    assert all(parameter.grad is None or not bool(parameter.grad.any()) for parameter in parameters)


@pytest.mark.parametrize("horizons", [(), (0, 1), (1, 1), (4, 1), (2, 4), (1, 1024), (1, 2.0), (True, 4)])
def test_invalid_horizons_cannot_silently_select_a_different_experiment(horizons):
    with pytest.raises(ValueError):
        model.validate_args(args_for(prediction_horizons=horizons))


def test_uploaded_windows_exclude_crossed_resets_but_include_terminal_endpoints_and_keep_env_identity():
    steps, env_count, horizons = 6, 4, (1, 2, 4)
    terms = np.zeros((steps, env_count), dtype=bool)
    truncs = np.zeros_like(terms)
    terms[0, 0] = True
    terms[3, 1] = True
    truncs[1, 2] = True
    terms[2, 3] = truncs[2, 3] = True
    fields = {"native_actions": (1,), "next_observations": (1,)}
    transfer = RolloutTransfer(steps, env_count, (1,), torch.device("cuda"), fields=fields)
    try:
        for time in range(steps):
            transfer.push(time, np.zeros(env_count, dtype=np.float32), terms[time], truncs[time],
                          native_actions=(np.arange(env_count) + time / 10).astype(np.float32)[:, None],
                          next_observations=(100 + np.arange(env_count) * 1000 + time).astype(np.float32)[:, None])
        uploaded = transfer.upload()
        windows = model.prediction_windows(uploaded.terminations, uploaded.truncations, horizons)
        actions, targets, validity = model.gather_prediction_windows(
            uploaded.fields["native_actions"].flatten(0, 1),
            uploaded.fields["next_observations"].flatten(0, 1), windows,
        )
        expected_valid, expected_actions, expected_targets = [], [], []
        for time in range(steps):
            for environment in range(env_count):
                expected_valid.append([
                    time + horizon <= steps
                    and not bool((terms | truncs)[time:time + horizon - 1, environment].any())
                    for horizon in horizons
                ])
                expected_actions.append([[environment + min(time + offset, steps - 1) / 10]
                                         for offset in range(max(horizons))])
                expected_targets.append([[100 + 1000 * environment + min(time + horizon - 1, steps - 1)]
                                         for horizon in horizons])
        torch.testing.assert_close(validity, torch.tensor(expected_valid, device="cuda"))
        torch.testing.assert_close(actions, torch.tensor(expected_actions, device="cuda"))
        torch.testing.assert_close(targets, torch.tensor(expected_targets, device="cuda", dtype=torch.float32))
        assert bool(validity[:, 0].all())
        assert bool(validity[1, 2])  # t=0, env=1: terminal endpoint at transition 3.
        assert bool(validity[2, 1])  # t=0, env=2: truncated endpoint at transition 1.
        assert bool(validity[env_count + 3, 1])  # t=1, env=3: simultaneous endpoint at 2.
        assert not bool(validity[3, 2])  # Crosses that simultaneous reset.
        assert not bool(validity[-1, 1:].any())  # No fabricated tail horizon.
        expected_envs = torch.arange(steps * env_count, device="cuda").remainder(env_count)[:, None]
        torch.testing.assert_close(windows.action_indices.remainder(env_count), expected_envs.expand_as(windows.action_indices))
        torch.testing.assert_close(windows.target_indices.remainder(env_count), expected_envs.expand_as(windows.target_indices))
    finally:
        transfer.close()


@pytest.mark.parametrize("loss_kind", ["mse", "huber"])
@pytest.mark.parametrize("target_gradient", ["attached", "stopped"])
def test_masked_objective_same_forward_comparators_and_target_stop_leave_sigreg_online(envs, loss_kind, target_gradient):
    args = args_for(prediction_loss=loss_kind, prediction_target_gradient=target_gradient)
    agent = make_agent(envs, args)
    observations, native, targets, rewards, sequences, validity = make_batch(agent)
    observations.requires_grad_()
    targets.requires_grad_()
    sequences.requires_grad_()
    captured, predictions, sigreg_inputs = {}, [], []

    def projection_hook(module, inputs, output):
        captured["embeddings"] = output

    def prediction_hook(module, inputs, output):
        predictions.append(output)

    def sigreg_hook(module, inputs):
        sigreg_inputs.append(inputs[0])

    handles = [agent.ssl.projector.register_forward_hook(projection_hook),
               agent.ssl.pred_proj.register_forward_hook(prediction_hook),
               agent.ssl.sigreg.register_forward_pre_hook(sigreg_hook)]
    rng_before = torch.cuda.get_rng_state().clone()
    components, metrics = model.representation_components(
        agent, observations, native, targets, rewards, args,
        action_sequences=sequences, validity=validity,
    )
    rng_after = torch.cuda.get_rng_state().clone()
    for handle in handles:
        handle.remove()
    embeddings = captured["embeddings"]
    expected_losses = []
    expected_tail_fractions, expected_tail_shares = [], []
    for index, horizon in enumerate(args.prediction_horizons):
        mask = validity[:, index]
        actual_target = embeddings[index + 1]
        training_target = actual_target.detach() if target_gradient == "stopped" else actual_target
        predicted = predictions[horizon - 1]
        objective = model.prediction_objective(predicted[mask], training_target[mask], loss_kind)
        expected_losses.append(objective)
        residual = (predicted - actual_target)[mask]
        squared, tails = residual.square(), residual.abs() > 1.0
        expected_tail_fractions.append(tails.float().mean())
        expected_tail_shares.append((squared * tails).sum() / squared.sum().clamp_min(1e-12))
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/prediction_loss"], objective)
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/raw_mse"], residual.square().mean())
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/huber"],
                                   2 * F.smooth_l1_loss(predicted[mask], actual_target[mask], beta=1.0))
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/persistence_mse"],
                                   (embeddings[0] - actual_target)[mask].square().mean())
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/zero_mse"], actual_target[mask].square().mean())
        torch.testing.assert_close(metrics[f"ssl/h{horizon}/valid_samples"], mask.sum())
    expected_loss = torch.stack(expected_losses).mean()
    torch.testing.assert_close(components[0], expected_loss)
    torch.testing.assert_close(metrics["ssl/shared_residual_gt1_fraction"],
                               torch.stack(expected_tail_fractions).mean())
    torch.testing.assert_close(metrics["ssl/shared_residual_gt1_squared_error_share"],
                               torch.stack(expected_tail_shares).mean())
    expected_projector_grads = torch.autograd.grad(expected_loss, tuple(agent.ssl.projector.parameters()), retain_graph=True)
    actual_projector_grads = torch.autograd.grad(components[0], tuple(agent.ssl.projector.parameters()), retain_graph=True)
    for actual, expected in zip(actual_projector_grads, expected_projector_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
    prediction_grads = torch.autograd.grad(components[0], (observations, targets, embeddings), retain_graph=True)
    assert prediction_grads[0].norm() > 0
    assert prediction_grads[2][0].norm() > 0
    if target_gradient == "stopped":
        assert not bool(prediction_grads[1].any())
        assert not bool(prediction_grads[2][1:].any())
    else:
        assert prediction_grads[1][validity].norm() > 0
        assert bool((prediction_grads[2][1:].flatten(1).norm(dim=1) > 0).all())
    assert not bool(prediction_grads[1][~validity].any())
    sigreg_grads = torch.autograd.grad(components[1], (observations, targets), retain_graph=True)
    assert sigreg_grads[0].norm() > 0
    assert sigreg_grads[1][:, 0].norm() > 0
    assert not bool(sigreg_grads[1][:, 1:].any())
    assert len(sigreg_inputs) == 1 and sigreg_inputs[0].shape == (2, 512, 64)
    torch.testing.assert_close(sigreg_inputs[0], embeddings[:2], rtol=0, atol=0)
    torch.cuda.set_rng_state(rng_before)
    expected_sigreg = agent.ssl.sigreg(embeddings[:2].detach())
    torch.testing.assert_close(components[1], args.sigreg_weight * expected_sigreg)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_after, rtol=0, atol=0)
    rng_before_diagnostics = torch.cuda.get_rng_state().clone()
    model.gradient_balance(agent, components)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before_diagnostics, rtol=0, atol=0)
    assert all(not metric.requires_grad for metric in metrics.values())
    components.sum().backward()
    assert sequences.grad is None  # Factual future actions are training data, never differentiable controls.


@pytest.mark.parametrize("target_gradient", ["attached", "stopped"])
def test_invalid_sequence_values_cannot_change_valid_losses_or_gradients_even_with_empty_horizons(envs, target_gradient):
    args = args_for(prediction_target_gradient=target_gradient)
    agent = make_agent(envs, args)
    observations, native, targets, rewards, sequences, validity = make_batch(agent)
    validity[:, -1] = False  # Empty H16 must contribute zero, without renormalizing away that horizon.
    unused_actions = ~validity[:, agent.ssl.action_horizon_indices]
    changed_sequences = sequences.masked_fill(unused_actions.unsqueeze(-1), float("nan"))
    changed_targets = targets.masked_fill((~validity).unsqueeze(-1), float("nan"))
    parameters = tuple(agent.encoder.parameters()) + tuple(agent.ssl.parameters())
    results = []
    for target_values, action_values in ((targets, sequences), (changed_targets, changed_sequences)):
        torch.manual_seed(31)
        components, metrics = model.representation_components(
            agent, observations, native, target_values, rewards, args,
            action_sequences=action_values, validity=validity,
        )
        gradients = torch.autograd.grad(components[:2].sum(), parameters)
        assert metrics["ssl/h16/valid_samples"] == 0
        for name in ("prediction_loss", "raw_mse", "huber", "persistence_mse", "zero_mse"):
            assert metrics[f"ssl/h16/{name}"] == 0
        torch.testing.assert_close(components[0],
                                   (metrics["ssl/h1/prediction_loss"] + metrics["ssl/h4/prediction_loss"]) / 3)
        results.append((components, gradients, metrics))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for actual, expected in zip(results[0][1], results[1][1], strict=True):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol=0, atol=0)


def test_long_horizon_predictions_depend_on_ordered_actions_without_changing_h1(envs):
    args = args_for()
    agent = make_agent(envs, args)
    # Canonical AdaLN-zero starts action-independent; exercise a nonzero learned
    # modulation rather than changing its canonical initialization for this test.
    with torch.no_grad():
        modulation = agent.ssl.predictor.modulation[-1]
        modulation.weight.copy_(torch.linspace(-0.2, 0.2, modulation.weight.numel(), device="cuda")
                                .reshape_as(modulation.weight))
        modulation.bias[128:].fill_(0.3)
    current = torch.randn(512, 64, device="cuda")
    following = torch.randn(512, 3, 64, device="cuda")
    actions = torch.rand(512, 16, 6, device="cuda")
    reordered = actions.clone()
    reordered[:, 1], reordered[:, 2] = actions[:, 2], actions[:, 1]
    runs = []
    for values in (actions, reordered):
        predictions = []
        handle = agent.ssl.pred_proj.register_forward_hook(lambda module, inputs, output: predictions.append(output.detach().clone()))
        agent.ssl(current, following, values)
        handle.remove()
        runs.append(predictions)
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    assert not torch.allclose(runs[0][3], runs[1][3], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("loss_kind,target_gradient", [("mse", "attached"), ("huber", "stopped")])
@pytest.mark.parametrize("mode", ["actor", "both"])
def test_h1_matches_frozen_v7_parameters_initialization_rng_losses_and_gradients(envs, loss_kind, target_gradient, mode):
    common = dict(sigreg_num_proj=16, sigreg_proj_chunk=8, prediction_loss=loss_kind,
                  prediction_target_gradient=target_gradient, jepa_mode=mode)
    old_args = reference.Args(**common)
    args = model.Args(prediction_horizons=(1,), **common)
    torch.manual_seed(13)
    old = reference.Agent(envs, old_args).cuda()
    old_cpu_rng, old_cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    agent = make_agent(envs, args)
    torch.testing.assert_close(torch.get_rng_state(), old_cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), old_cuda_rng, rtol=0, atol=0)
    assert agent.parameter_counts() == old.parameter_counts()
    assert agent.state_dict().keys() == old.state_dict().keys()
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(agent.state_dict()[name], expected, rtol=0, atol=0)
    observations, native, targets, rewards, _, _ = make_batch(agent, horizons=(1,))
    data, _ = ppo_data(agent, observations, native)
    old_ppo = reference.policy_loss(old, *data, old_args)[0]
    actual_ppo = model.policy_loss(agent, *data, args)[0]
    torch.manual_seed(29)
    old_components, old_metrics = reference.representation_components(old, observations, native, targets[:, 0], rewards, old_args)
    old_rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(29)
    actual_components, actual_metrics = model.representation_components(agent, observations, native, targets[:, 0], rewards, args)
    torch.testing.assert_close(torch.cuda.get_rng_state(), old_rng, rtol=0, atol=0)
    torch.testing.assert_close(actual_ppo, old_ppo, rtol=0, atol=0)
    torch.testing.assert_close(actual_components, old_components, rtol=2e-5, atol=2e-6)
    for name, expected in old_metrics.items():
        torch.testing.assert_close(actual_metrics[name], expected, rtol=2e-5, atol=2e-6)
    (old_ppo + old_components.sum()).backward()
    (actual_ppo + actual_components.sum()).backward()
    for actual, expected in zip(agent.parameters(), old.parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("activation,residual,projection", [("tanh", False, "none"), ("stiglu", True, "all")])
def test_default_multistep_retains_canonical_raw_and_latent_initialization(envs, activation, residual, projection):
    common = dict(sigreg_num_proj=16, sigreg_proj_chunk=8, task_activation=activation,
                  task_residual=residual, weight_projection=projection)
    torch.manual_seed(13)
    old = reference.Agent(envs, reference.Args(**common)).cuda()
    rng_after_old = torch.get_rng_state().clone()
    for mode in ("both", "actor"):
        agent = make_agent(envs, model.Args(jepa_mode=mode, **common))
        torch.testing.assert_close(torch.get_rng_state(), rng_after_old, rtol=0, atol=0)
        for name in ("actor", "encoder", "ssl", "reward_head"):
            for key, expected in getattr(old, name).state_dict().items():
                torch.testing.assert_close(getattr(agent, name).state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode,features", [("actor", "online"), ("both", "online"), ("both", "rollout")])
def test_joint_ppo_gradient_isolation_and_state_only_critic(envs, mode, features):
    args = args_for(jepa_mode=mode, critic_feature_updates=features, reward_mode="attached")
    agent = make_agent(envs, args)
    observations, native, targets, rewards, sequences, validity = make_batch(agent)
    data, snapshot = ppo_data(agent, observations, native)
    cache = snapshot.requires_grad_() if features == "rollout" else None
    before = agent.get_value(observations).detach().clone()
    ppo, _, _, _ = model.joint_loss(agent, *data, targets, rewards, args, critic_features=cache,
                                   action_sequences=sequences, validity=validity)
    ppo.backward()
    policy, representation, reward = agent.parameter_groups()
    assert_no_grad(representation + reward)
    if cache is not None:
        assert cache.grad is None
    for network in (agent.actor, agent.critic):
        assert any(parameter.grad is not None and bool(parameter.grad.any()) for parameter in network.parameters())
    groups = [set(map(id, group)) for group in (policy, representation, reward)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert set.union(*groups) == set(map(id, agent.parameters()))
    model.representation_components(agent, observations, native, targets.flip(1), rewards, args,
                                    action_sequences=sequences.flip(1), validity=validity)
    torch.testing.assert_close(agent.get_value(observations), before, rtol=0, atol=0)


@pytest.mark.parametrize("loss_kind,target_gradient,features", [
    ("mse", "attached", "online"), ("huber", "stopped", "rollout"),
])
def test_fullgraph_cuda_joint_and_world_only_updates_with_real_ssl512(envs, loss_kind, target_gradient, features):
    args = args_for(prediction_loss=loss_kind, prediction_target_gradient=target_gradient,
                    critic_feature_updates=features)
    agent = make_agent(envs, args)
    observations, native, targets, rewards, sequences, validity = make_batch(agent, steps=64)
    data, snapshot = ppo_data(agent, observations, native)
    cache = snapshot if features == "rollout" else None
    ssl_inputs = (targets[:512], rewards[:512], sequences[:512], validity[:512])

    def joint(observations, actions, logprobs, advantages, returns, values, targets, rewards, sequences, validity, cache):
        return model.joint_loss(agent, observations, actions, logprobs, advantages, returns, values,
                                targets, rewards, args, critic_features=cache,
                                action_sequences=sequences, validity=validity)

    def world(observations, native, targets, rewards, sequences, validity):
        return model.representation_components(agent, observations, native, targets, rewards, args,
                                               action_sequences=sequences, validity=validity)

    eager_ppo, eager_components, _, _ = joint(*data, *ssl_inputs, cache)
    compiled_joint = torch.compile(joint, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    compiled_world = torch.compile(world, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    ppo, components, _, metrics = compiled_joint(*data, *ssl_inputs, cache)
    torch.testing.assert_close(ppo, eager_ppo, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(components[[0, 2]], eager_components[[0, 2]], rtol=5e-5, atol=5e-6)
    (ppo + components.sum()).backward()
    groups = agent.parameter_groups()
    optimizers = make_optimizers(agent, args)
    before = [[parameter.detach().clone() for parameter in group] for group in groups]
    model.optimizer_step(agent, optimizers, args)
    for group, previous in zip(groups, before, strict=True):
        assert any(not torch.equal(parameter, old) for parameter, old in zip(group, previous, strict=True))
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    for owner in optimizers:
        owner.zero_grad(set_to_none=True)
    before_world = [[parameter.detach().clone() for parameter in group] for group in groups]
    world_components, world_metrics = compiled_world(
        observations[512:], native[512:], targets[512:], rewards[512:], sequences[512:], validity[512:],
    )
    world_components.sum().backward()
    assert_no_grad(groups[0])
    model.optimizer_step(agent, optimizers, args, policy_step=False)
    for parameter, previous in zip(groups[0], before_world[0], strict=True):
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
    for group, previous in zip(groups[1:], before_world[1:], strict=True):
        assert any(not torch.equal(parameter, old) for parameter, old in zip(group, previous, strict=True))
    assert all(bool(torch.isfinite(value)) for value in world_metrics.values())
    assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.parameters())


@pytest.mark.parametrize("ppo_size", [512, 2048, 16384])
def test_full_batch_ppo_cannot_change_ssl512_or_320_exposures(ppo_size):
    rows, epochs = 16 * 1024, 10
    args = model.validate_args(args_for(num_envs=16, num_minibatches=rows // ppo_size, update_epochs=epochs))
    generator = torch.Generator(device="cuda").manual_seed(7)
    ppo_count = ssl_count = 0
    exposure = torch.zeros(rows, device="cuda", dtype=torch.long)
    for _ in range(epochs):
        for ppo_rows, ssl_rows in model.iter_update_batches(rows, ppo_size, 512, "cuda", generator):
            assert ssl_rows.shape == (512,)
            exposure.index_add_(0, ssl_rows, torch.ones_like(ssl_rows))
            ssl_count += 1
            if ppo_rows is not None:
                assert ppo_rows.shape == (ppo_size,)
                torch.testing.assert_close(ssl_rows, ppo_rows[:512], rtol=0, atol=0)
                ppo_count += 1
    torch.testing.assert_close(exposure, torch.full_like(exposure, epochs), rtol=0, atol=0)
    assert ssl_count == 320
    assert ppo_count == args.update_epochs * rows // ppo_size


def test_drift_probe_covers_environments_and_time_without_advancing_any_rng():
    cpu_before, cuda_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    indices = model.drift_probe_indices(1024, 16, "cuda")
    assert indices.shape == (128,)
    assert torch.unique(indices).numel() == 128
    torch.testing.assert_close(torch.bincount(indices.remainder(16), minlength=16),
                               torch.full((16,), 8, device="cuda", dtype=torch.long))
    torch.testing.assert_close(torch.bincount((indices // 16) // 128, minlength=8),
                               torch.full((8,), 16, device="cuda", dtype=torch.long))
    torch.testing.assert_close(model.drift_probe_indices(4, 3, "cuda"), torch.arange(12, device="cuda"))
    torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_before, rtol=0, atol=0)
