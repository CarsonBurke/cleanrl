"""Queue through mlq: CUDA contracts, not shortened environment training runs.

These exercise the v7 phase helpers at the production learner batch size.
The main-loop ordering itself still needs the parent's actual training run.
Evaluation normalization is already covered by the predecessor's suite.
"""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import (
    ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_hlgauss_criticfirst_v7 as trainer,
)
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_v1 as baseline
from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.ppo_loop import compute_gae

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


@pytest.fixture(
    params=[
        pytest.param((201, 10.0), id="normalized"),
        pytest.param((1001, 2500.0), id="raw"),
    ]
)
def value_config(request):
    bins, extent = request.param
    return HLGaussConfig(
        num_bins=bins,
        v_min=-extent,
        v_max=extent,
        sigma_ratio=2.0,
        transform="linear",
        bin_type="edges",
    )


@pytest.fixture
def device():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    with torch.random.fork_rng():
        torch.manual_seed(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield torch.device("cuda")
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _spaces():
    # Non-unit, asymmetric bounds expose accidentally treating native samples
    # as physical actions, or omitting the density/entropy Jacobian.
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-3, -2, 1, -4, 0, -1], np.float32),
            np.array([2, 4, 5, -1, 2, 8], np.float32),
        ),
    )


def _observations(count, device):
    generator = np.random.default_rng(1)
    return torch.as_tensor(generator.standard_normal((count, 17)).astype(np.float32), device=device)


def _policy_inputs(agent, observations, clipped):
    count = observations.shape[0]
    native = torch.linspace(0.03, 0.97, count * 6, device=observations.device).reshape(count, 6)
    native[0, ::2] = trainer.SAMPLE_EPS
    native[0, 1::2] = 1.0 - trainer.SAMPLE_EPS
    with torch.no_grad():
        alpha, beta = (torch.nn.functional.softplus(agent.actor(observations)) + 1).chunk(2, dim=-1)
        old_logprobs = agent.action_logprob(alpha, beta, native)
        if clipped:
            # All combinations of positive/negative advantage and low/high
            # ratio occur, including samples on the active unclipped branch.
            logratios = observations.new_tensor([0.0, 0.6, -0.6, 0.6, -0.6, 0.0])
            old_logprobs = old_logprobs - logratios.repeat((count + 5) // 6)[:count]
    advantages = observations.new_tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    return native, old_logprobs, advantages.repeat((count + 5) // 6)[:count]


def _snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_unchanged(module, snapshot):
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0, msg=name)


def _step(loss, optimizer, parameters, max_grad_norm):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, foreach=True)
    optimizer.step()
    return norm.detach()


@pytest.mark.parametrize("clipped", [False, True], ids=["unclipped", "clipped"])
def test_actor_loss_matches_frozen_baseline_without_a_critic(value_config, device, clipped):
    agent = trainer.Agent(_spaces(), value_config=value_config).to(device)
    reference = baseline.Agent(_spaces()).to(device)
    reference.actor.load_state_dict(agent.actor.state_dict())
    observations = _observations(192, device)
    native, old_logprobs, supplied_advantages = _policy_inputs(agent, observations, clipped)
    # Keep the same graph alive across backwards. Updating the actor must not
    # backpropagate through the critic that produced refreshed advantages.
    advantages = supplied_advantages + agent.get_value(observations).flatten()
    advantages.retain_grad()
    saved_advantages = advantages.detach().clone()
    producer = agent.critic
    del agent.critic
    args = trainer.Args(value=value_config, norm_adv=False, ent_coef=0.017)
    reference_args = baseline.Args(norm_adv=False, ent_coef=args.ent_coef, vf_coef=0.0, clip_vloss=False)
    zero_values = torch.zeros_like(supplied_advantages)
    for _ in range(2):
        agent.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        actual, metrics = trainer.actor_loss(agent, observations, native, old_logprobs, advantages, args)
        expected, baseline_metrics = baseline.ppo_loss(
            reference,
            observations,
            native,
            old_logprobs,
            saved_advantages,
            zero_values,
            zero_values,
            reference_args,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(metrics, baseline_metrics[[0, 2, 3, 4, 5]], rtol=2e-5, atol=3e-6)
        assert (metrics[-1] > 0.5) if clipped else (metrics[-1] == 0)
        actual.backward()
        expected.backward()
        for (name, candidate), (_, frozen) in zip(agent.actor.named_parameters(), reference.actor.named_parameters()):
            assert candidate.grad is not None and frozen.grad is not None, name
            torch.testing.assert_close(candidate.grad, frozen.grad, rtol=1e-4, atol=5e-6, msg=name)
        assert advantages.grad is None
        assert all(parameter.grad is None for parameter in producer.parameters())
        assert advantages.requires_grad
        torch.testing.assert_close(advantages, saved_advantages, rtol=0, atol=0)


def test_repeated_phases_preserve_fixed_label_graph_and_other_network(value_config, device):
    agent = trainer.Agent(_spaces(), value_config=value_config).to(device)
    args = trainer.Args(value=value_config)
    actor_optimizer, critic_optimizer = trainer.make_optimizers(agent, args)
    observations = _observations(128, device)
    # Nonuniform predictions avoid a constant cross-entropy hiding an unwanted
    # derivative into a normalized probability producer.
    with torch.no_grad():
        agent.critic[-1].bias.copy_(torch.linspace(-1, 1, value_config.num_bins, device=device))
    returns = torch.linspace(-0.4 * value_config.v_max, 0.4 * value_config.v_max, 128, device=device)
    returns.requires_grad_()
    labels = agent.value_support.project(returns)
    labels.retain_grad()
    saved_labels, saved_returns = labels.detach().clone(), returns.detach().clone()
    native, old_logprobs, advantages = _policy_inputs(agent, observations, clipped=True)
    initial_actor = _snapshot(agent.actor)
    for _ in range(4):
        actor_before = _snapshot(agent.actor)
        with torch.no_grad():
            policy_before = agent.actor(observations).clone()
        loss, ce = trainer.critic_loss(agent, observations, labels, args)
        assert not ce.requires_grad
        torch.testing.assert_close(loss.detach(), args.vf_coef * ce)
        _step(loss, critic_optimizer, agent.critic.parameters(), args.max_grad_norm)
        assert returns.grad is None and labels.grad is None
        assert labels.requires_grad and returns.requires_grad
        _assert_unchanged(agent.actor, actor_before)
        with torch.no_grad():
            torch.testing.assert_close(agent.actor(observations), policy_before, rtol=0, atol=0)
            values_before = agent.get_value(observations).clone()
        critic_before = _snapshot(agent.critic)
        # Do not clear the other network's gradients: real sequential phases
        # leave them behind, so overlapping optimizers would silently step it.
        actor_loss, _ = trainer.actor_loss(agent, observations, native, old_logprobs, advantages, args)
        _step(actor_loss, actor_optimizer, agent.actor.parameters(), args.max_grad_norm)
        _assert_unchanged(agent.critic, critic_before)
        with torch.no_grad():
            torch.testing.assert_close(agent.get_value(observations), values_before, rtol=0, atol=0)
    assert any(not torch.equal(value, initial_actor[name]) for name, value in agent.actor.state_dict().items())
    torch.testing.assert_close(labels, saved_labels, rtol=0, atol=0)
    torch.testing.assert_close(returns, saved_returns, rtol=0, atol=0)


def test_compiled_full_batch_phases_are_independent_and_refit_fixed_targets(value_config, device):
    """Real 2048 x 16 shapes, backward capture and ten epochs per phase."""
    agent = trainer.Agent(_spaces(), value_config=value_config).to(device)
    reference = copy.deepcopy(agent)
    args = trainer.Args(
        value=value_config,
        learning_rate=0.0096,
        critic_learning_rate=0.003,
        norm_adv=False,
        max_grad_norm=0.5,
        ent_coef=0.01,
        update_epochs=10,
    )
    actor_optimizer, critic_optimizer = trainer.make_optimizers(agent, args)
    # Two distinguishable states, each repeated through the production batch,
    # make value regression learnable without asserting arbitrary convergence.
    prototypes = _observations(2, device)
    observations = prototypes.repeat(16384, 1)
    returns = observations.new_tensor([-0.3, 0.3]).repeat(16384) * value_config.v_max
    labels = agent.value_support.project(returns).detach()
    saved_labels = labels.clone()
    native, old_logprobs, advantages = _policy_inputs(agent, observations, clipped=True)
    saved_native, saved_logprobs = native.clone(), old_logprobs.clone()
    compiled_actor = torch.compile(
        lambda obs, actions, logprobs, adv: trainer.actor_loss(agent, obs, actions, logprobs, adv, args),
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    compiled_critic = torch.compile(
        lambda obs, targets: trainer.critic_loss(agent, obs, targets, args),
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    # Independent eager CE oracle; the baseline actor oracle is tested above.
    expected = args.vf_coef * reference.value_support.loss(reference.critic(observations), returns)
    expected.backward()
    torch.compiler.cudagraph_mark_step_begin()
    actual, ce = compiled_critic(observations, labels)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    actual.backward()
    for (name, candidate), (_, eager) in zip(agent.critic.named_parameters(), reference.critic.named_parameters()):
        assert candidate.grad is not None and eager.grad is not None, name
        torch.testing.assert_close(candidate.grad, eager.grad, rtol=1e-4, atol=5e-6, msg=name)
    del actual, ce, expected
    agent.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    expected, expected_metrics = trainer.actor_loss(reference, observations, native, old_logprobs, advantages, args)
    expected.backward()
    torch.compiler.cudagraph_mark_step_begin()
    actual, actual_metrics = compiled_actor(observations, native, old_logprobs, advantages)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-5, atol=3e-6)
    actual.backward()
    for (name, candidate), (_, eager) in zip(agent.actor.named_parameters(), reference.actor.named_parameters()):
        assert candidate.grad is not None and eager.grad is not None, name
        torch.testing.assert_close(candidate.grad, eager.grad, rtol=1e-4, atol=5e-6, msg=name)
    del actual, actual_metrics, expected, expected_metrics, reference
    agent.zero_grad(set_to_none=True)
    with torch.no_grad():
        before_values = agent.get_value(prototypes).flatten().clone()
        before_ce = agent.value_support.loss(agent.critic(prototypes), returns[:2]).clone()
        before_policy = agent.actor(prototypes).clone()
        alpha, beta = (torch.nn.functional.softplus(before_policy) + 1).chunk(2, dim=-1)
        before_logprobs = agent.action_logprob(alpha, beta, native[:2]).clone()
    actor_before = _snapshot(agent.actor)
    critic_stats = torch.empty((args.update_epochs, 2), device=device)
    for epoch in range(args.update_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        loss, ce = compiled_critic(observations, labels)
        norm = _step(loss, critic_optimizer, agent.critic.parameters(), args.max_grad_norm)
        critic_stats[epoch, 0].copy_(ce)
        critic_stats[epoch, 1].copy_(norm)
        del loss, ce, norm
        _assert_unchanged(agent.actor, actor_before)
        with torch.no_grad():
            policy = agent.actor(prototypes)
            torch.testing.assert_close(policy, before_policy, rtol=0, atol=0)
            alpha, beta = (torch.nn.functional.softplus(policy) + 1).chunk(2, dim=-1)
            torch.testing.assert_close(agent.action_logprob(alpha, beta, native[:2]), before_logprobs, rtol=0, atol=0)
    with torch.no_grad():
        after_values = agent.get_value(prototypes).flatten().clone()
        after_ce = agent.value_support.loss(agent.critic(prototypes), returns[:2])
        assert after_ce < before_ce
        assert (after_values - returns[:2]).square().mean() < (before_values - returns[:2]).square().mean()
    critic_before = _snapshot(agent.critic)
    agent.zero_grad(set_to_none=True)
    actor_stats = torch.empty((args.update_epochs, 6), device=device)
    for epoch in range(args.update_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        loss, metrics = compiled_actor(observations, native, old_logprobs, advantages)
        norm = _step(loss, actor_optimizer, agent.actor.parameters(), args.max_grad_norm)
        actor_stats[epoch, :5].copy_(metrics)
        actor_stats[epoch, 5].copy_(norm)
        del loss, metrics, norm
        _assert_unchanged(agent.critic, critic_before)
        with torch.no_grad():
            torch.testing.assert_close(agent.get_value(prototypes).flatten(), after_values, rtol=0, atol=0)
    assert torch.isfinite(critic_stats).all() and torch.isfinite(actor_stats).all()
    assert (critic_stats[:, 1] > 0).all() and (actor_stats[:, 5] > 0).all()
    with torch.no_grad():
        assert not torch.allclose(agent.actor(prototypes), before_policy)
    torch.testing.assert_close(labels, saved_labels, rtol=0, atol=0)
    torch.testing.assert_close(native, saved_native, rtol=0, atol=0)
    torch.testing.assert_close(old_logprobs, saved_logprobs, rtol=0, atol=0)


def test_gae_refresh_respects_terminal_time_limit_and_tail_dependencies(device):
    rewards = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4) / 4
    old_values = torch.zeros_like(rewards)
    fresh_values = 0.2 + rewards / 3
    terminations = torch.zeros_like(rewards)
    truncations = torch.zeros_like(rewards)
    terminations[1, 0] = 1
    truncations[1, 1] = 1
    terminations[1, 3] = truncations[1, 3] = 1
    old_bootstraps = torch.zeros_like(rewards)
    fresh_bootstraps = torch.full_like(rewards, 3.0)
    old_tail = torch.zeros(4, device=device)
    fresh_tail = torch.full_like(old_tail, 4.0)

    def gae(values, bootstraps, tail, rollout_rewards=rewards):
        return compute_gae(
            rollout_rewards,
            values,
            terminations,
            truncations,
            bootstraps,
            tail,
            0.99,
            0.95,
        )

    old_advantages, _ = gae(old_values, old_bootstraps, old_tail)
    fresh_advantages, fresh_returns = gae(fresh_values, fresh_bootstraps, fresh_tail)
    assert not torch.allclose(fresh_advantages, old_advantages)
    # Failing to refresh either terminal observations or the rollout tail
    # changes exactly the corresponding predecessor chain, not other envs.
    stale_time_limits, _ = gae(fresh_values, old_bootstraps, fresh_tail)
    torch.testing.assert_close(stale_time_limits[:, [0, 2, 3]], fresh_advantages[:, [0, 2, 3]])
    torch.testing.assert_close(stale_time_limits[2], fresh_advantages[2])
    assert torch.all(stale_time_limits[:2, 1] != fresh_advantages[:2, 1])
    stale_tail, _ = gae(fresh_values, fresh_bootstraps, old_tail)
    torch.testing.assert_close(stale_tail[:2, [0, 1, 3]], fresh_advantages[:2, [0, 1, 3]])
    assert torch.all(stale_tail[:, 2] != fresh_advantages[:, 2])
    # Rewards in reset episodes cannot leak backward through either kind of
    # done; ordinary transitions must still propagate that same perturbation.
    later_rewards = rewards.clone()
    later_rewards[2] += 100
    perturbed, _ = gae(fresh_values, fresh_bootstraps, fresh_tail, later_rewards)
    torch.testing.assert_close(perturbed[:2, [0, 1, 3]], fresh_advantages[:2, [0, 1, 3]])
    assert torch.all(perturbed[:2, 2] != fresh_advantages[:2, 2])
    # True terminal returns cannot acquire a value bootstrap even when the
    # environment also labels the same transition as time-limited.
    _, shifted_returns = gae(fresh_values + 20, fresh_bootstraps + 100, fresh_tail + 100)
    torch.testing.assert_close(shifted_returns[1, [0, 3]], fresh_returns[1, [0, 3]], rtol=0, atol=2e-6)
