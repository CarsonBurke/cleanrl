"""CUDA contracts for direct SF v14. Run only through mlq; no CPU model fallback."""
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_successor_reward_factored_v14 as model
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


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


def make_agent(envs, mode):
    torch.manual_seed(17)
    return model.Agent(envs, model.Args(critic_mode=mode)).cuda()


def test_native_reward_factors_use_displacement_physical_control_and_final_info():
    env = make_mujoco_vector_env("HalfCheetah-v4", 2, backend="native", num_threads=2,
                                 max_episode_steps=2)
    try:
        raw_obs, _ = env.reset(seed=1)
        obs_norm = VectorObsNorm(2, (17,))
        obs_norm.normalize(raw_obs)
        # Actual environment action bounds, not the Beta [0,1] sample space.
        low, high = env.single_action_space.low, env.single_action_space.high
        native = np.array([[0.1, 0.3, 0.8, 0.4, 0.7, 0.9],
                           [0.9, 0.7, 0.2, 0.6, 0.3, 0.1]], dtype=np.float32)
        physical = low + native * (high - low)
        for step in range(2):
            before = np.array([outer.unwrapped.data.qpos[0] for outer in env.envs])
            # The last transition also verifies the real ClipAction control cost.
            action = physical if step == 0 else physical * 3
            raw_obs, raw_reward, terms, truncs, infos = env.step(action)
            factors = model.observed_reward_factors(raw_reward, terms, truncs, infos)
            np.testing.assert_allclose(factors.sum(-1), raw_reward, rtol=0, atol=1e-14)
            clipped = np.clip(action, low, high)
            expected_cost = np.array([outer.unwrapped.control_cost(row)
                                      for outer, row in zip(env.envs, clipped, strict=True)])
            np.testing.assert_allclose(factors[:, 1], -expected_cost, rtol=0, atol=0)
            if step == 0:
                assert not np.allclose(factors[:, 1], -0.1 * np.square(native).sum(-1))
            next_obs, factual = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
            for index, outer in enumerate(env.envs):
                info = infos["final_info"][index] if truncs[index] else {
                    key: infos[key][index] for key in ("x_position", "x_velocity")
                }
                base = outer.unwrapped
                displacement_reward = base._forward_reward_weight * (info["x_position"] - before[index]) / base.dt
                np.testing.assert_allclose(factors[index, 0], displacement_reward, rtol=1e-12, atol=1e-12)
            if step == 1:
                assert np.all(truncs) and not np.any(terms)
                assert not np.allclose(next_obs, factual)
                missing = deepcopy(infos)
                del missing["final_info"][0]["reward_ctrl"]
                with pytest.raises(RuntimeError, match="final_info missing"):
                    model.observed_reward_factors(raw_reward, terms, truncs, missing)
    finally:
        env.close()


def test_normalized_factor_readout_preserves_zero_rewards_clipping_and_transfer():
    norm = VectorRewardNorm(4, 0.99)
    norm.counts[:] = 1e9
    norm.variances[:] = (0.01, 0.02, 0.03, 16)
    raw_factors = np.array([[10000.5, -0.5], [-9999.5, -0.5], [2, -2], [2.5, -0.5]])
    raw = raw_factors.sum(-1)
    terms = np.array([False, False, True, False])
    truncs = np.array([False, True, False, False])
    reward = norm.normalize(raw, terms)
    factors = model.normalize_reward_factors(raw_factors, raw, reward, norm)
    assert reward[0] == 10 and reward[1] == -10 and reward[2] == 0
    np.testing.assert_allclose(factors.sum(-1), reward, rtol=1e-7, atol=1e-7)
    assert np.isfinite(factors).all() and factors[2, 0] > 0 and factors[2, 1] < 0
    np.testing.assert_allclose(factors[:, 0] / raw_factors[:, 0], factors[:, 1] / raw_factors[:, 1])
    transfer = RolloutTransfer(1, 4, (17,), "cuda", fields={"reward_factors": (2,)})
    try:
        transfer.push(0, reward, terms, truncs, reward_factors=factors)
        batch = transfer.upload()
        torch.testing.assert_close(batch.fields["reward_factors"].sum(-1), batch.rewards, rtol=2e-6, atol=2e-7)
    finally:
        transfer.close()
    # A future normalizer update cannot reinterpret an already collected factor.
    frozen = factors.copy()
    norm.normalize(raw * 4, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(factors, frozen)


@pytest.mark.parametrize("compiled", [False, True])
def test_vector_scalar_return_identity_and_factual_boundary_masks(compiled):
    gamma, lam = 0.8, 0.6
    features = torch.arange(1, 31, device="cuda", dtype=torch.float32).reshape(3, 5, 2).requires_grad_()
    predictions = torch.randn_like(features, requires_grad=True)
    factual_next = (torch.randn_like(features) * 7).requires_grad_()
    terms = torch.zeros(3, 5, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[0, 0] = terms[0, 2] = terms[1, 3] = 1
    truncs[0, 1] = truncs[0, 2] = 1
    gae = get_gae_fn(compiled=compiled, explicit_next_values=True)
    targets = model.successor_lambda_targets(features, predictions, factual_next, terms, truncs, gamma, lam, gae)
    scalar_targets = model.successor_lambda_targets(
        features.sum(-1, keepdim=True), predictions.sum(-1, keepdim=True), factual_next.sum(-1, keepdim=True),
        terms, truncs, gamma, lam, gae,
    )
    assert not targets.requires_grad
    torch.testing.assert_close(targets.sum(-1), scalar_targets[..., 0], rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(targets.sum(-1) - predictions.detach().sum(-1),
                               scalar_targets[..., 0] - predictions.detach().sum(-1), rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(targets[0, 0], features[0, 0])
    torch.testing.assert_close(targets[0, 1], features[0, 1] + gamma * factual_next[0, 1])
    torch.testing.assert_close(targets[0, 2], features[0, 2])
    torch.testing.assert_close(targets[0, 3], features[0, 3] + gamma * factual_next[0, 3]
                               + gamma * lam * (features[1, 3] - predictions[1, 3]))
    torch.testing.assert_close(targets[-1], features[-1] + gamma * factual_next[-1])
    changed_features, changed_predictions, changed_next = (x.detach().clone() for x in (features, predictions, factual_next))
    changed_features[1:, :3] += 1000
    changed_predictions[1:, :3] -= 300
    changed_next[1:, :3] += 200
    changed_next[0, 0] += 50
    changed_next[0, 2] += 50
    changed = model.successor_lambda_targets(changed_features, changed_predictions, changed_next,
                                            terms, truncs, gamma, lam, gae)
    torch.testing.assert_close(changed[0, :3], targets[0, :3])
    changed_next[0, 1] += 10
    changed = model.successor_lambda_targets(changed_features, changed_predictions, changed_next,
                                            terms, truncs, gamma, lam, gae)
    torch.testing.assert_close(changed[0, 1] - targets[0, 1], torch.full((2,), gamma * 10, device="cuda"))


def test_loss_normalization_does_not_reparameterize_bootstraps_or_task_direction(envs):
    agent = make_agent(envs, "successor")
    observations = torch.randn(12, 17, device="cuda")
    predictions = agent.get_successor(observations).detach()
    features = torch.randn(3, 4, 2, device="cuda")
    terms = torch.zeros(3, 4, device="cuda")
    gae = get_gae_fn(explicit_next_values=True)
    following = predictions.reshape(3, 4, 2) + 0.2
    targets = model.successor_lambda_targets(features, predictions.reshape(3, 4, 2), following,
                                            terms, terms, 0.99, 0.95, gae).flatten(0, 1)
    old_values = predictions.sum(-1)
    args = model.Args(clip_vloss=False)
    scale = model.successor_loss_scale(targets)
    value, auxiliary = model.critic_losses(predictions, targets, old_values, args, scale)
    other_value, other_auxiliary = model.critic_losses(predictions, targets, old_values, args, scale * 2)
    torch.testing.assert_close(other_value, value, rtol=0, atol=0)
    torch.testing.assert_close(other_auxiliary, auxiliary / 4)
    torch.testing.assert_close(agent.get_successor(observations), predictions, rtol=0, atol=0)
    repeated = model.successor_lambda_targets(features, predictions.reshape(3, 4, 2), following,
                                             terms, terms, 0.99, 0.95, gae).flatten(0, 1)
    torch.testing.assert_close(repeated, targets, rtol=0, atol=0)
    # Explicit coordinate rescaling must transform BOTH immediate factors and
    # bootstraps. A normalized regression metric then has invariant auxiliary loss.
    multiplier = 7.0
    rescaled = model.successor_lambda_targets(features * multiplier, predictions.reshape(3, 4, 2) * multiplier,
                                             following * multiplier, terms, terms, 0.99, 0.95, gae).flatten(0, 1)
    torch.testing.assert_close(rescaled / multiplier, targets, rtol=2e-6, atol=2e-6)
    rescaled_scale = model.successor_loss_scale(rescaled)
    scaled_value, scaled_auxiliary = model.critic_losses(predictions * multiplier, rescaled,
                                                        old_values * multiplier, args, rescaled_scale)
    torch.testing.assert_close(scaled_value / multiplier ** 2, value, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(scaled_auxiliary, auxiliary, rtol=2e-6, atol=2e-6)


def test_reward_direction_loss_and_gradient_do_not_shrink_with_auxiliary_dimension():
    args = model.Args(clip_vloss=False)
    for channels in (1, 2, 7):
        prediction = torch.full((8, channels), 2.0 / channels, device="cuda", requires_grad=True)
        target = torch.zeros_like(prediction)
        value, _ = model.critic_losses(prediction, target, prediction.detach().sum(-1), args,
                                       torch.ones((), device="cuda"))
        torch.testing.assert_close(value, torch.tensor(2.0, device="cuda"))
        value.backward()
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 2.0 / 8))


def test_matched_actor_initial_values_native_mirror_and_physical_action_logprob(envs):
    scalar, successor = (make_agent(envs, mode) for mode in ("scalar", "successor"))
    observations = torch.randn(32, 17, device="cuda")
    native = torch.linspace(0.1, 0.9, 32 * 6, device="cuda").reshape(32, 6)
    physical = scalar.action_low + native * scalar.action_scale
    a0, b0, value0 = scalar.get_policy_and_value(observations)
    a1, b1, value1 = successor.get_policy_and_value(observations)
    torch.testing.assert_close(a0, a1, rtol=0, atol=0)
    torch.testing.assert_close(b0, b1, rtol=0, atol=0)
    torch.testing.assert_close(value0, value1, rtol=2e-5, atol=5e-7)
    _, physical_logprob, _, _ = scalar.get_action_and_value(observations, physical)
    torch.testing.assert_close(physical_logprob, scalar.action_logprob(a0, b0, native), rtol=2e-6, atol=2e-6)
    host = model.HostPolicy(successor, 32)
    native_logits = host(observations.cpu().numpy()).copy()
    np.testing.assert_allclose(native_logits, successor.actor(observations).detach().cpu().numpy(), rtol=2e-5, atol=2e-7)
    with torch.no_grad():
        successor.actor.head[0].bias.add_(0.25)
    host.refresh()
    np.testing.assert_allclose(host(observations.cpu().numpy()),
                               successor.actor(observations).detach().cpu().numpy(), rtol=2e-5, atol=2e-7)


def test_adding_factor_contrast_preserves_scalar_adam_value_updates(envs):
    scalar, successor = (make_agent(envs, mode) for mode in ("scalar", "successor"))
    observations = torch.randn(64, 17, device="cuda")
    returns = torch.randn(64, device="cuda")
    optimizers = [torch.optim.Adam(agent.critic.parameters(), lr=3e-4, eps=1e-5)
                  for agent in (scalar, successor)]
    for _ in range(4):
        for agent, optimizer in zip((scalar, successor), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * (agent.get_value(observations).flatten() - returns).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            optimizer.step()
        torch.testing.assert_close(scalar.get_value(observations), successor.get_value(observations),
                                   rtol=3e-5, atol=3e-6)


def training_batch(agent, rows=32):
    observations = torch.randn(rows, 17, device="cuda", requires_grad=True)
    native = (torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1).requires_grad_()
    with torch.no_grad():
        predictions, logprobs = model.rollout_statistics(agent, observations, native)
    targets = (predictions + torch.randn_like(predictions)).requires_grad_()
    old_values = predictions.sum(-1).requires_grad_()
    advantages = torch.randn(rows, device="cuda", requires_grad=True)
    old_logprobs = logprobs.detach().requires_grad_()
    scale = model.successor_loss_scale(targets).requires_grad_()
    return observations, native, old_logprobs, advantages, targets, old_values, scale


def test_exact_gradient_owners_and_detached_teachers(envs):
    agent = make_agent(envs, "successor")
    args = model.Args(clip_vloss=False)
    data = training_batch(agent)
    observations, native, old_logprobs, advantages, targets, old_values, scale = data
    objective, _ = model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                     targets, old_values, args, scale)
    objective.backward()
    actor, critic = agent.parameter_groups()
    assert set(map(id, actor)).isdisjoint(map(id, critic))
    assert set(map(id, actor + critic)) == set(map(id, agent.parameters()))
    assert all(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in actor + critic)
    assert all(tensor.grad is None for tensor in data)
    # Critic gradients reach both actual representation stages with either
    # task-direction supervision or reward-nullspace supervision alone.
    for component in (0, 1):
        agent.zero_grad(set_to_none=True)
        predictions = agent.get_successor(observations)
        losses = model.critic_losses(predictions, targets, old_values, args, scale)
        losses[component].backward()
        assert all(parameter.grad is None for parameter in actor)
        assert all(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in critic)
    # Both matched actors get identical gradients for the same PPO batch, even
    # though one critic has an auxiliary loss. Owner clipping is separate too.
    scalar = make_agent(envs, "scalar")
    agent.zero_grad(set_to_none=True)
    scalar_objective, _ = model.policy_loss(scalar, observations, native, old_logprobs, advantages,
                                            targets.sum(-1, keepdim=True), old_values, args, scale)
    sf_objective, _ = model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                       targets, old_values, args, scale)
    scalar_objective.backward()
    sf_objective.backward()
    for first, second in zip(scalar.actor.parameters(), agent.actor.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["scalar", "successor"])
def test_compiled_cuda_objective_backward_and_owned_optimizer_update(envs, mode):
    agent = make_agent(envs, mode)
    args = model.Args(critic_mode=mode, clip_vloss=False)
    data = training_batch(agent)

    def objective(observations, native, old_logprobs, advantages, targets, old_values, scale):
        return model.policy_loss(agent, observations, native, old_logprobs, advantages,
                                  targets, old_values, args, scale)

    expected, expected_metrics = objective(*data)
    expected.backward()
    expected_gradients = [parameter.grad.detach().clone() for parameter in agent.parameters()]
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, fullgraph=True, options={"triton.cudagraphs": False})
    actual, actual_metrics = compiled(*data)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for name in expected_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=3e-5, atol=3e-6)
    actual.backward()
    for parameter, expected_gradient in zip(agent.parameters(), expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=5e-4, atol=5e-6)
    groups = agent.parameter_groups()
    optimizers = tuple(torch.optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in groups)
    before = [parameter.detach().clone() for parameter in agent.parameters()]
    model.optimizer_step(optimizers, groups, args.max_grad_norm, torch.empty(2, device="cuda"))
    for parameter, previous in zip(agent.parameters(), before, strict=True):
        assert bool((parameter != previous).any())
        assert bool(torch.isfinite(parameter).all())
    assert all(tensor.grad is None for tensor in data)
