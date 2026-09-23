"""CUDA behavioral contracts; submit through mlq, never a shortened training run."""

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Beta

from cleanrl import ppo_continuous_action_jepa_intact_model_control_v5 as model
from cleanrl.shared.ppo_loop import TruncationBootstrapCache
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    # Asymmetric non-unit bounds expose accidental native/physical-action mixing.
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2, -1, -0.5, -3, -1, -4], dtype=np.float32),
            np.array([1, 3, 2, 0.5, 4, 2], dtype=np.float32),
        ),
    )


def make_agent(envs, mode="both"):
    args = model.Args(control_mode=mode, sigreg_num_proj=16, sigreg_proj_chunk=8)
    return model.Agent(envs, args).cuda(), args


def observations(agent, rows=32, requires_grad=False):
    state = torch.randn(rows, agent.observation_dim, device="cuda")
    previous = agent.action_low + agent.action_scale * torch.rand(rows, agent.action_dim, device="cuda")
    return torch.cat((state, previous), dim=-1).requires_grad_(requires_grad)


def world_batch(agent, requires_grad=False):
    current = observations(agent, requires_grad=requires_grad)
    native = torch.rand(32, agent.action_dim, device="cuda") * 0.8 + 0.1
    following = observations(agent)
    following[:, agent.observation_dim:] = agent.action_low + agent.action_scale * native
    following.requires_grad_(requires_grad)
    goal = observations(agent, requires_grad=requires_grad)
    rewards = torch.randn(32, device="cuda")
    terminated = torch.arange(32, device="cuda") % 5 == 0
    return current, native, following, goal, rewards, terminated


def assert_useful_gradients(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.square().sum() for g in gradients) > 0


def assert_no_gradients(parameters):
    assert all(p.grad is None or not bool(p.grad.any()) for p in parameters)


def manual_imagination(agent, x, args):
    z = agent.encode(x).detach()
    previous = x[:, agent.observation_dim:].detach()
    states, rewards, continuations, following_values = [], [], [], []
    for _ in range(args.imagination_horizon):
        states.append((z, previous))
        alpha, beta = agent.policy_from_latent(z, previous)
        native = Beta(alpha, beta, validate_args=False).rsample().clamp(model.SAMPLE_EPS, 1 - model.SAMPLE_EPS)
        action = agent.action_low + agent.action_scale * native
        rewards.append(agent.predict_reward(z, action, frozen=True))
        continuations.append(agent.predict_continuation(z, action, frozen=True))
        z = agent.predict_next(z, action, frozen=True)
        previous = action
        following_values.append(agent.value_from_latent(z, previous, frozen=True).flatten())
    return states, rewards, continuations, following_values


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_world_update_is_identical_across_ablation_and_owns_only_world(envs, mode):
    agent, args = make_agent(envs, mode)
    world, actor, critic = agent.parameter_groups()
    groups = [set(map(id, group)) for group in (world, actor, critic)]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert set.union(*groups) == set(map(id, agent.parameters()))
    assert groups[1] == set(map(id, agent.prescriber.parameters()))
    assert groups[2] == set(map(id, agent.critic.parameters()))
    data = world_batch(agent)
    torch.manual_seed(17)
    loss, _ = model.world_loss(agent, *data, args)
    torch.manual_seed(17)
    matched, _ = model.world_loss(agent, *data, replace(args, control_mode="none"))
    torch.testing.assert_close(loss, matched, rtol=0, atol=0)
    loss.backward()
    assert_useful_gradients(world)
    for module in (agent.encoder, agent.dynamics, agent.action_law,
                   agent.previous_action_embedding, agent.reward_head, agent.continuation_head):
        assert_useful_gradients(module.parameters())
    assert_no_gradients(actor)
    assert_no_gradients(critic)
    before = [p.detach().clone() for p in (*actor, *critic)]
    torch.optim.Adam(world, lr=5e-5).step()
    for parameter, saved in zip((*actor, *critic), before):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)


def test_forward_prediction_attaches_both_temporal_encodings(envs):
    agent, args = make_agent(envs)
    data = world_batch(agent, requires_grad=True)
    loss, _ = model.world_loss(
        agent, *data, replace(args, local_nll_coef=0, goal_nll_coef=0, sigreg_weight=0)
    )
    loss.backward()
    for temporal in (data[0], data[2]):
        assert temporal.grad[:, :agent.observation_dim].norm() > 0
    assert data[3].grad is None or not bool(data[3].grad.any())
    assert_no_gradients(agent.prescriber.parameters())
    assert_no_gradients(agent.critic.parameters())


def test_real_value_anchor_trains_only_critic(envs):
    agent, _ = make_agent(envs)
    x = observations(agent, requires_grad=True)
    value = agent.get_value(x).flatten()
    target = torch.randn_like(value)
    (0.5 * (value - target).square().mean()).backward()
    world, actor, critic = agent.parameter_groups()
    assert_useful_gradients(critic)
    assert_no_gradients(world)
    assert_no_gradients(actor)
    if x.grad is not None:
        torch.testing.assert_close(x.grad[:, :agent.observation_dim], torch.zeros_like(x[:, :agent.observation_dim]), rtol=0, atol=0)


@pytest.mark.parametrize("branch", ["local", "goal"])
def test_intent_nll_has_attached_temporal_branches_and_detached_goal_anchor(envs, branch):
    agent, args = make_agent(envs)
    data = world_batch(agent, requires_grad=True)
    current, native, following, goal, _, _ = data
    baseline_args = replace(args, local_nll_coef=0, goal_nll_coef=0, sigreg_weight=0)
    active_args = replace(baseline_args, **{f"{branch}_nll_coef": 1.0})
    active, _ = model.world_loss(agent, *data, active_args)
    baseline, _ = model.world_loss(agent, *data, baseline_args)
    z = agent.encode(current)
    intent = agent.encode(following) - z if branch == "local" else agent.encode(goal).detach() - z
    alpha, beta = agent.law_from_intent(z, intent, current[:, agent.observation_dim:], frozen=False)
    expected = -(Beta(alpha, beta, validate_args=False).log_prob(native) - agent.log_action_scale).sum(-1).mean()
    inputs = (current, following, goal, *agent.encoder.parameters(), *agent.action_law.parameters())
    actual_gradients = torch.autograd.grad(active - baseline, inputs, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, inputs, allow_unused=True)
    for source, actual, wanted in zip(inputs, actual_gradients, expected_gradients):
        actual = torch.zeros_like(source) if actual is None else actual
        wanted = torch.zeros_like(source) if wanted is None else wanted
        torch.testing.assert_close(actual, wanted, rtol=2e-3, atol=2e-6)
    assert actual_gradients[0][:, :agent.observation_dim].norm() > 0
    if branch == "local":
        assert actual_gradients[1][:, :agent.observation_dim].norm() > 0
    assert actual_gradients[2] is None or not bool(actual_gradients[2].any())


def test_reward_and_continuation_supervision_cannot_shape_encoder(envs):
    agent, args = make_agent(envs)
    args = replace(args, local_nll_coef=0, goal_nll_coef=0, sigreg_weight=0)
    data = world_batch(agent, requires_grad=True)
    first, _ = model.world_loss(agent, *data, args)
    changed = (*data[:4], data[4] + 2, ~data[5])
    second, _ = model.world_loss(agent, *changed, args)
    (second - first).backward()
    for parameter in agent.encoder.parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=1e-6)
    assert_useful_gradients(agent.reward_head.parameters())
    assert_useful_gradients(agent.continuation_head.parameters())


def test_policy_context_ignores_previous_action_in_encoder_but_not_actor_or_value(envs):
    agent, _ = make_agent(envs)
    first = observations(agent)
    second = first.clone()
    second[:, agent.observation_dim:] = agent.action_low + agent.action_high - first[:, agent.observation_dim:]
    torch.testing.assert_close(agent.encode(first), agent.encode(second), rtol=0, atol=0)
    alpha_a, beta_a, value_a = agent.get_policy_and_value(first)
    alpha_b, beta_b, value_b = agent.get_policy_and_value(second)
    assert not torch.equal(alpha_a, alpha_b) and not torch.equal(beta_a, beta_b)
    assert not torch.equal(value_a, value_b)
    torch.testing.assert_close(agent.get_value(first), value_a, rtol=0, atol=0)
    torch.testing.assert_close(agent.direct_policy(first)[0], alpha_a, rtol=0, atol=0)
    (-Beta(alpha_a, beta_a, validate_args=False).log_prob(torch.full_like(alpha_a, 0.3)).mean()).backward()
    world, actor, critic = agent.parameter_groups()
    assert_useful_gradients(actor)
    assert_no_gradients(world)
    assert_no_gradients(critic)


def test_actor_optimizes_stochastic_model_return_through_frozen_weights(envs):
    agent, args = make_agent(envs)
    x = observations(agent, requires_grad=True)
    torch.manual_seed(39)
    loss = model.actor_objective(agent, x, args)
    loss.backward()
    world, actor, critic = agent.parameter_groups()
    actual_gradients = [p.grad.detach().clone() for p in actor]
    assert_useful_gradients(actor)
    assert_no_gradients(world)
    assert_no_gradients(critic)
    assert x.grad is None or not bool(x.grad.any())
    agent.zero_grad(set_to_none=True)
    torch.manual_seed(39)
    _, rewards, continuations, values = manual_imagination(agent, x, args)
    discount = torch.ones_like(rewards[0])
    expected_return = torch.zeros_like(discount)
    for reward, continuation in zip(rewards, continuations):
        expected_return = expected_return + discount * reward
        discount = discount * args.gamma * continuation
    expected = -(expected_return + discount * values[-1]).mean()
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-6)
    for parameter, wanted in zip(actor, actual_gradients):
        torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=3e-7)
    # A fresh sample must not silently become a deterministic mean-action rollout.
    torch.manual_seed(40)
    assert not torch.equal(model.actor_objective(agent, x, args), loss)


def test_imagined_critic_regresses_detached_lambda_returns_with_physical_history(envs):
    agent, args = make_agent(envs)
    x = observations(agent, requires_grad=True)
    torch.manual_seed(51)
    actual = model.imagined_critic_loss(agent, x, args)
    actual.backward()
    world, actor, critic = agent.parameter_groups()
    gradients = [p.grad.detach().clone() for p in critic]
    assert_useful_gradients(critic)
    assert_no_gradients(world)
    assert_no_gradients(actor)
    assert x.grad is None or not bool(x.grad.any())
    agent.zero_grad(set_to_none=True)
    torch.manual_seed(51)
    with torch.no_grad():
        states, rewards, continuations, next_values = manual_imagination(agent, x, args)
        target = next_values[-1]
        targets = []
        for step in reversed(range(args.imagination_horizon)):
            target = rewards[step] + args.gamma * continuations[step] * (
                (1 - args.gae_lambda) * next_values[step] + args.gae_lambda * target
            )
            targets.append(target)
        targets.reverse()
    z = torch.cat([state[0] for state in states])
    previous = torch.cat([state[1] for state in states])
    prediction = agent.value_from_latent(z, previous).flatten()
    expected = 0.5 * (prediction - torch.cat(targets)).square().mean()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for parameter, wanted in zip(critic, gradients):
        torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("horizon", [1, 3, 20])
def test_goals_stop_at_termination_time_limit_and_rollout_end(horizon):
    # These are factual next observations, already corrected for auto-reset.
    following = torch.arange(7 * 2 * 23, device="cuda", dtype=torch.float32).reshape(7, 2, 23)
    terminated = torch.zeros(7, 2, device="cuda", dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[1, 0] = True
    truncated[4, 0] = True
    truncated[2, 1] = True
    terminated[5, 1] = True
    actual = model.make_goal_observations(following, terminated, truncated, horizon)
    expected = torch.empty_like(following)
    for start in range(7):
        for env in range(2):
            endpoint = start
            for candidate in range(start, min(start + horizon, 7)):
                endpoint = candidate
                if bool(terminated[candidate, env] or truncated[candidate, env]):
                    break
            expected[start, env] = following[endpoint, env]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_host_actor_refreshes_encoder_prescriber_and_shared_law(envs):
    agent, _ = make_agent(envs)
    x = observations(agent, rows=16)
    host_x = x.detach().cpu().numpy()
    mirror = model.HostIntactActor(agent, 16)
    for module in (None, agent.encoder, agent.prescriber, agent.action_law, agent.previous_action_embedding):
        if module is not None:
            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.02)
            mirror.refresh()
        with torch.no_grad():
            alpha, beta = agent.direct_policy(x)
            logits = torch.from_numpy(mirror(host_x).copy()).cuda()
            host_alpha, host_beta = (torch.nn.functional.softplus(logits) + 1).chunk(2, dim=-1)
            torch.testing.assert_close(host_alpha, alpha, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(host_beta, beta, rtol=3e-5, atol=3e-6)


def test_frozen_world_and_action_law_remain_differentiable_in_their_inputs(envs):
    agent, _ = make_agent(envs)
    # AdaLN-zero initially gates action conditioning off. Move away from that
    # initialization to exercise the action Jacobian a learned model must retain.
    with torch.no_grad():
        for parameter in agent.dynamics.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.03)
    z = agent.encode(observations(agent)).detach().requires_grad_()
    action = (agent.action_low + agent.action_scale * torch.rand(32, agent.action_dim, device="cuda")).requires_grad_()
    outputs = (
        agent.predict_next(z, action, frozen=True),
        agent.predict_reward(z, action, frozen=True),
        agent.predict_continuation(z, action, frozen=True),
        agent.value_from_latent(z, action, frozen=True),
    )
    for output in outputs:
        gradients = torch.autograd.grad(output.sum(), (z, action), retain_graph=True)
        assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)
    intent = torch.randn_like(z, requires_grad=True)
    alpha, beta = agent.law_from_intent(z, intent, action, frozen=True)
    (alpha.square().mean() + beta.square().mean()).backward()
    assert z.grad.norm() > 0 and intent.grad.norm() > 0 and action.grad.norm() > 0
    assert_no_gradients(agent.parameters())


def test_fullgraph_stochastic_objectives_keep_separate_gradient_owners(envs):
    agent, args = make_agent(envs)
    data = world_batch(agent)

    def objectives(current, native, following, goal, reward, terminated):
        world, _ = model.world_loss(agent, current, native, following, goal, reward, terminated, args)
        actor = model.actor_objective(agent, current, args)
        critic = model.imagined_critic_loss(agent, current, args)
        return world, actor, critic

    compiled = torch.compile(objectives, mode="reduce-overhead", fullgraph=True)
    owners = agent.parameter_groups()
    optimizers = [torch.optim.Adam(parameters, lr=5e-5) for parameters in owners]
    previous_actor_loss = None
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        losses = compiled(*data)
        assert all(torch.isfinite(loss) for loss in losses)
        actor_loss = losses[1].detach().clone()
        if previous_actor_loss is not None:
            assert not torch.equal(actor_loss, previous_actor_loss)
        previous_actor_loss = actor_loss
        saved_gradients = []
        for index, loss in enumerate(losses):
            agent.zero_grad(set_to_none=True)
            loss.backward(retain_graph=index < len(losses) - 1)
            assert_useful_gradients(owners[index])
            for other, parameters in enumerate(owners):
                if other != index:
                    assert_no_gradients(parameters)
            saved_gradients.append([None if p.grad is None else p.grad.detach().clone() for p in owners[index]])
        # Detached functional weights share storage; update only after every
        # backward has consumed the same forward's saved model tensors.
        for parameters, gradients, optimizer in zip(owners, saved_gradients, optimizers):
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            optimizer.step()


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_training_routes_only_the_selected_learning_signals(envs, mode):
    agent, args = make_agent(envs, mode)
    args = replace(args, ent_coef=0.03, norm_adv=True, clip_vloss=True)
    current, native, following, goal, reward, terminated = world_batch(agent)
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(current)
        old_logprob = agent.action_logprob(alpha, beta, native) - torch.linspace(-0.4, 0.4, 32, device="cuda")
        old_value = value.flatten() + torch.linspace(-0.5, 0.5, 32, device="cuda")
        returns = torch.randn_like(old_value)
    advantages = torch.randn(32, device="cuda")
    torch.manual_seed(71)
    actual, _ = model.training_loss(
        agent, current, native, old_logprob, advantages, returns, old_value,
        following, goal, reward, terminated, args,
    )
    _, actor, critic = agent.parameter_groups()
    actual_gradients = torch.autograd.grad(actual, (*actor, *critic))
    torch.manual_seed(71)
    alpha, beta, value = agent.get_policy_and_value(current)
    ratio = (agent.action_logprob(alpha, beta, native) - old_logprob).exp()
    normalized_advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg = torch.maximum(-normalized_advantage * ratio, -normalized_advantage * ratio.clamp(
        1 - args.clip_coef, 1 + args.clip_coef
    )).mean()
    entropy = (Beta(alpha, beta, validate_args=False).entropy() + agent.log_action_scale).sum(-1).mean()
    selected_actor = model.actor_objective(agent, current, args) if mode in {"actor", "both"} else pg - args.ent_coef * entropy
    clipped_value = old_value + (value.flatten() - old_value).clamp(-args.clip_coef, args.clip_coef)
    real_critic = 0.5 * torch.maximum((value.flatten() - returns).square(), (clipped_value - returns).square()).mean()
    extra_critic = model.imagined_critic_loss(agent, current, args) if mode in {"critic", "both"} else 0
    selected_critic = args.vf_coef * (real_critic + extra_critic)
    expected_world, _ = model.world_loss(agent, current, native, following, goal, reward, terminated, args)
    expected = selected_actor + selected_critic + expected_world
    expected_gradients = torch.autograd.grad(expected, (*actor, *critic))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-4, atol=3e-6)


def test_collector_preserves_factual_actions_and_resets_only_policy_history(envs):
    agent, _ = make_agent(envs)
    rng = np.random.default_rng(9)
    next_obs = rng.standard_normal((4, agent.observation_dim)).astype(np.float32)
    final_obs = next_obs.copy()
    final_obs[1:] += 3
    physical = rng.uniform(envs.single_action_space.low, envs.single_action_space.high, size=(4, 6)).astype(np.float32)
    terminated = np.array([False, True, False, True])
    truncated = np.array([False, False, True, True])
    next_out = np.empty((4, agent.input_dim), dtype=np.float32)
    final_out = np.empty_like(next_out)
    model.augment_transition_observations(
        next_obs, final_obs, physical, terminated, truncated,
        next_out=next_out, final_out=final_out,
    )
    np.testing.assert_array_equal(next_out[:, :agent.observation_dim], next_obs)
    np.testing.assert_array_equal(final_out[:, :agent.observation_dim], final_obs)
    np.testing.assert_array_equal(final_out[:, agent.observation_dim:], physical)
    np.testing.assert_array_equal(next_out[0, agent.observation_dim:], physical[0])
    np.testing.assert_array_equal(next_out[1:, agent.observation_dim:], np.zeros((3, 6), dtype=np.float32))
    cache = TruncationBootstrapCache(2, 4, (agent.input_dim,))
    cache.push_normalized(0, truncated & ~terminated, final_out)
    with torch.no_grad():
        expected = agent.get_value(torch.from_numpy(final_out.copy()).cuda()).flatten()
        reset_value = agent.get_value(torch.from_numpy(next_out.copy()).cuda()).flatten()
    assert not torch.equal(expected[2], reset_value[2])
    # Collector outputs and native sampler actions are reusable staging buffers.
    # The cache must retain the factual augmented state before their next use.
    final_out.fill(0)
    physical.fill(0)
    with torch.no_grad():
        bootstraps = cache.resolve(lambda x: agent.get_value(x).flatten(), torch.device("cuda"))
    wanted = torch.zeros(2, 4, device="cuda")
    wanted[0, 2] = expected[2]
    torch.testing.assert_close(bootstraps, wanted, rtol=2e-5, atol=2e-6)
