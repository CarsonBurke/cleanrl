import copy
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "sf_vlam_v9_for_v14",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v14_geometric_event_mdn",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v14_geometric_event_mdn.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_event_autoencoder_has_exact_linear_reward_anchor():
    args = MODULE.Args()
    content_dim = args.emb_dim + 17 + 2 * 6 + 1
    autoencoder = MODULE.EventAutoencoder(content_dim, args).cuda()
    reward = torch.randn(64, device="cuda") * 7.0
    content = torch.randn(64, content_dim, device="cuda")
    latent = autoencoder.encode(reward, content)

    assert latent.shape == (64, args.event_latent_dim)
    for coordinate in range(args.event_reward_dims):
        torch.testing.assert_close(
            latent[:, coordinate], reward / args.event_reward_scale
        )
    torch.testing.assert_close(autoencoder.decode_reward(latent), reward)
    torch.testing.assert_close(
        autoencoder.decode_reward(torch.zeros_like(latent)), torch.zeros_like(reward)
    )


def test_distribution_value_is_only_linear_mixture_reward_expectation():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    batch = 19
    absorb_logit = torch.randn(batch, device="cuda")
    logits = torch.randn(batch, args.mdn_components, device="cuda")
    means = torch.randn(
        batch, args.mdn_components, args.event_latent_dim, device="cuda"
    )
    params = absorb_logit, logits, means

    weights = torch.softmax(logits, dim=-1)
    alive_reward = (
        weights.unsqueeze(-1) * means[..., : args.event_reward_dims]
    ).sum(1).mean(-1) * args.event_reward_scale
    expected = (
        torch.sigmoid(-absorb_logit) * alive_reward / (1.0 - args.gamma)
    )
    torch.testing.assert_close(agent.distribution_value(params), expected)

    all_absorbed = (torch.full_like(absorb_logit, 100.0), logits, means)
    torch.testing.assert_close(
        agent.distribution_value(all_absorbed),
        torch.zeros_like(expected),
        atol=1e-30,
        rtol=0.0,
    )


def test_geometric_targets_bootstrap_truncation_and_tail_but_absorb_termination():
    time_steps, num_envs, latent_dim, components = 8, 3, 5, 2
    gamma = 0.99
    events = torch.arange(
        time_steps * num_envs * latent_dim, device="cuda", dtype=torch.float32
    ).view(time_steps, num_envs, latent_dim)
    terminations = torch.zeros(time_steps, num_envs, device="cuda")
    boundaries = torch.zeros_like(terminations)
    valids = torch.ones_like(terminations)
    terminations[3, 0] = 1.0
    boundaries[3, 0] = 1.0
    boundaries[3, 1] = 1.0  # TimeLimit: final observation is valid.

    absorb_logits = torch.full((time_steps, num_envs), -100.0, device="cuda")
    absorb_logits[:, 1] = 100.0  # Truncation bootstrap deterministically absorbs.
    logits = torch.zeros(time_steps, num_envs, components, device="cuda")
    means = torch.zeros(
        time_steps, num_envs, components, latent_dim, device="cuda"
    )
    means[:, 2] = 7.0  # Tail bootstrap for env 2 is an alive all-seven event.
    bootstrap = (
        absorb_logits.reshape(-1),
        logits.reshape(-1, components),
        means.reshape(-1, components, latent_dim),
    )

    seed = 981
    expected_generator = torch.Generator(device="cpu").manual_seed(seed)
    uniform = torch.rand(
        (time_steps, num_envs), generator=expected_generator, dtype=torch.float64
    )
    horizons = torch.floor(torch.log1p(-uniform) / np.log(gamma)).long().cuda()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets, valid, absorbed, actual_horizons, bootstrapped = (
        MODULE.sample_geometric_future_targets(
            events,
            terminations,
            boundaries,
            valids,
            bootstrap,
            0.0,
            gamma,
            generator,
        )
    )
    torch.testing.assert_close(actual_horizons, horizons)
    assert valid.all()

    source = torch.arange(time_steps, device="cuda").unsqueeze(1)
    expected_term_absorb = ((source <= 3) & (source + horizons > 3))[:, 0]
    torch.testing.assert_close(absorbed[:, 0], expected_term_absorb)
    torch.testing.assert_close(
        targets[:, 0][expected_term_absorb],
        torch.zeros_like(targets[:, 0][expected_term_absorb]),
    )

    expected_trunc_bootstrap = torch.where(
        source <= 3, source + horizons > 3, source + horizons >= time_steps
    )[:, 1]
    torch.testing.assert_close(bootstrapped[:, 1], expected_trunc_bootstrap)
    torch.testing.assert_close(absorbed[:, 1], expected_trunc_bootstrap)

    expected_tail_bootstrap = (source + horizons >= time_steps)[:, 2]
    torch.testing.assert_close(bootstrapped[:, 2], expected_tail_bootstrap)
    torch.testing.assert_close(
        targets[:, 2][expected_tail_bootstrap],
        torch.full_like(targets[:, 2][expected_tail_bootstrap], 7.0),
    )


def test_fixed_covariance_mdn_rewards_correct_conditioning():
    target = torch.tensor([[-2.0], [2.0]], device="cuda")
    good_means = target.view(2, 1, 1)
    bad_means = good_means.roll(1, dims=0)
    logits = torch.zeros(2, 1, device="cuda")
    good = MODULE.mdn_nll(logits, good_means, target, 0.5)
    bad = MODULE.mdn_nll(logits, bad_means, target, 0.5)
    assert torch.all(good < bad)


def test_reward_moment_recursion_vectorizes_terminal_truncation_and_rollout_tail():
    gamma = 0.5
    rewards = torch.tensor(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ],
        device="cuda",
    )
    terminations = torch.zeros_like(rewards)
    boundaries = torch.zeros_like(rewards)
    valids = torch.ones_like(rewards)
    next_values = torch.tensor(
        [
            [11.0, 110.0, 1100.0, 11000.0],
            [12.0, 120.0, 1200.0, 12000.0],
            [13.0, 130.0, 1300.0, 13000.0],
        ],
        device="cuda",
    )

    # Environment 0 terminates after its middle reward: no terminal bootstrap.
    terminations[1, 0] = 1.0
    boundaries[1, 0] = 1.0
    # Environment 1 truncates after its middle reward: bootstrap final_observation.
    boundaries[1, 1] = 1.0
    # Environment 2 has no boundary: bootstrap only at the rollout tail.
    # Environment 3 truncates without a final observation: its prefix is unknowable.
    boundaries[1, 3] = 1.0
    valids[1, 3] = 0.0

    actual, actual_valid = MODULE.rao_blackwell_reward_moment(
        rewards, terminations, boundaries, valids, next_values, gamma
    )
    expected_returns = torch.tensor(
        [
            [1.0 + 0.5 * 2.0, 10.0 + 0.5 * (20.0 + 0.5 * 120.0), 100.0 + 0.5 * (200.0 + 0.5 * (300.0 + 0.5 * 1300.0))],
            [2.0, 20.0 + 0.5 * 120.0, 200.0 + 0.5 * (300.0 + 0.5 * 1300.0)],
            [3.0 + 0.5 * 13.0, 30.0 + 0.5 * 130.0, 300.0 + 0.5 * 1300.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(actual[:, :3], (1.0 - gamma) * expected_returns)
    torch.testing.assert_close(
        actual_valid,
        torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
            device="cuda",
        ),
    )


def test_factorized_nll_weights_alive_term_by_valid_alive_fraction():
    target = torch.tensor([[1.0], [0.0], [-1.0]], device="cuda")
    absorb_logit = torch.zeros(3, device="cuda")
    logits = torch.zeros(3, 1, device="cuda")
    means = torch.zeros(3, 1, 1, device="cuda")
    valid = torch.tensor([True, True, False], device="cuda")
    absorbed = torch.tensor([False, True, False], device="cuda")
    joint, absorb_loss, alive_nll, alive_fraction = MODULE.factorized_event_nll(
        absorb_logit, logits, means, target, valid, absorbed, 0.5
    )
    torch.testing.assert_close(alive_fraction, torch.tensor(0.5, device="cuda"))
    torch.testing.assert_close(joint, absorb_loss + 0.5 * alive_nll)


def test_missing_truncation_final_observation_is_explicitly_censored():
    time_steps, latent_dim = 2, 3
    events = torch.randn(time_steps, 1, latent_dim, device="cuda")
    terminations = torch.zeros(time_steps, 1, device="cuda")
    boundaries = torch.zeros_like(terminations)
    boundaries[0] = 1.0
    valids = torch.ones_like(terminations)
    valids[0] = 0.0
    bootstrap = (
        torch.full((time_steps,), 100.0, device="cuda"),
        torch.zeros(time_steps, 1, device="cuda"),
        torch.zeros(time_steps, 1, latent_dim, device="cuda"),
    )
    generator = torch.Generator(device="cpu").manual_seed(73)
    _, valid, absorbed, horizons, bootstrapped = MODULE.sample_geometric_future_targets(
        events,
        terminations,
        boundaries,
        valids,
        bootstrap,
        0.0,
        0.999,
        generator,
    )
    assert horizons[0, 0] > 0
    assert bootstrapped[0, 0]
    assert absorbed[0, 0]
    assert not valid[0, 0]


def test_slow_event_state_and_event_targets_stabilize_full_target_frame():
    torch.manual_seed(144)
    args = MODULE.Args()
    content_dim = args.emb_dim + 17 + 2 * 6 + 1
    source_state = MODULE.StateEncoder(17, args.emb_dim, args.ssl_hidden).cuda()
    target_state = copy.deepcopy(source_state).requires_grad_(False)
    online_event = MODULE.EventAutoencoder(content_dim, args).cuda()
    target_event = copy.deepcopy(online_event).requires_grad_(False)

    next_obs = torch.randn(128, 17, device="cuda")
    action = torch.randn(128, 6, device="cuda").tanh()
    continuation = torch.randint(0, 2, (128,), device="cuda").float()
    reward = torch.randn(128, device="cuda")
    with torch.no_grad():
        before_content = MODULE.event_content(
            target_state(next_obs), next_obs, action, continuation
        )
        before = target_event.encode(reward, before_content)
        for parameter in source_state.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
        for parameter in online_event.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
        hard_content = MODULE.event_content(
            source_state(next_obs), next_obs, action, continuation
        )
        hard_after = online_event.encode(reward, hard_content)

    # Exercise CompiledModule-aware source unwrapping used after SSL compilation.
    wrapped_source = MODULE.CompiledModule(source_state, cudagraphs=False)
    MODULE.ema_update(target_event, online_event, args.event_target_decay)
    MODULE.ema_update(
        target_state, wrapped_source, args.event_state_target_decay
    )
    with torch.no_grad():
        after_content = MODULE.event_content(
            target_state(next_obs), next_obs, action, continuation
        )
        after = target_event.encode(reward, after_content)
        ema_drift = (after - before).square().mean().sqrt()
        hard_drift = (hard_after - before).square().mean().sqrt()
    assert ema_drift < 0.1 * hard_drift
    torch.testing.assert_close(
        after[:, : args.event_reward_dims],
        before[:, : args.event_reward_dims],
        atol=0.0,
        rtol=0.0,
    )


def test_v9_trunk_actor_and_rng_pairing_is_exact():
    args_base = BASE.Args()
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), args_base)
    expected_rng = torch.get_rng_state()

    args = MODULE.Args()
    torch.manual_seed(918)
    agent = MODULE.Agent(_FakeEnv(), args)
    actual_rng = torch.get_rng_state()

    torch.testing.assert_close(actual_rng, expected_rng)
    for actual, expected in zip(
        agent.trunk.parameters(), base.trunk.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for name in ("actor_alpha_head", "actor_beta_head"):
        actual_head = getattr(agent, name)
        expected_head = getattr(base, name)
        torch.testing.assert_close(actual_head.weight, expected_head.weight)
        torch.testing.assert_close(actual_head.bias, expected_head.bias)


def test_initial_value_is_exactly_zero_with_low_absorbing_prior():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    obs = torch.randn(128, 17, device="cuda")
    with torch.no_grad():
        params = agent.get_value(obs)
        absorb_logit, logits, means = params
        value = agent.distribution_value(params)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0.0, rtol=0.0)
    torch.testing.assert_close(logits, torch.zeros_like(logits), atol=0.0, rtol=0.0)
    assert torch.sigmoid(absorb_logit).max() < 0.01
    assert means[..., args.event_reward_dims :].std() > 0.0


def test_compiled_distributional_loss_backward_is_finite():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    agent.trunk = MODULE.CompiledModule(agent.trunk, cudagraphs=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    actor_parameters = agent.actor_parameters()
    critic_parameters = agent.critic_parameters()
    obs = torch.randn(128, 17, device="cuda")
    with torch.no_grad():
        _, native_action, _, _, _ = agent.get_action_and_value(obs)
    _, _, newlogprob, entropy, value_dist = agent.get_action_and_value(
        obs, native_action
    )
    absorb_logit, logits, means = value_dist
    target = torch.randn(128, args.event_latent_dim, device="cuda")
    valid = torch.ones(128, dtype=torch.bool, device="cuda")
    absorbed = torch.zeros_like(valid)
    geometric_nll, _, _, _ = MODULE.factorized_event_nll(
        absorb_logit,
        logits,
        means,
        target,
        valid,
        absorbed,
        args.mdn_fixed_std,
    )
    value_loss = geometric_nll + torch.nn.functional.smooth_l1_loss(
        (1.0 - args.gamma) * agent.distribution_value(value_dist),
        torch.randn(128, device="cuda"),
    )
    policy_loss = -(newlogprob * torch.randn_like(newlogprob)).mean() - 0.01 * entropy.mean()

    optimizer.zero_grad(set_to_none=True)
    value_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(critic_parameters, args.critic_grad_clip)
    value_grads = [
        (parameter, parameter.grad.detach().clone())
        for parameter in critic_parameters
        if parameter.grad is not None
    ]
    optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_parameters, args.actor_grad_clip)
    for parameter, gradient in value_grads:
        parameter.grad = (
            gradient if parameter.grad is None else parameter.grad + gradient
        )
    optimizer.step()

    assert torch.isfinite(value_loss)
    assert torch.isfinite(policy_loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.critic_parameters()
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_event_autoencoder_has_exact_linear_reward_anchor()
    test_distribution_value_is_only_linear_mixture_reward_expectation()
    test_geometric_targets_bootstrap_truncation_and_tail_but_absorb_termination()
    test_fixed_covariance_mdn_rewards_correct_conditioning()
    test_reward_moment_recursion_vectorizes_terminal_truncation_and_rollout_tail()
    test_factorized_nll_weights_alive_term_by_valid_alive_fraction()
    test_missing_truncation_final_observation_is_explicitly_censored()
    test_slow_event_state_and_event_targets_stabilize_full_target_frame()
    test_v9_trunk_actor_and_rng_pairing_is_exact()
    test_initial_value_is_exactly_zero_with_low_absorbing_prior()
    test_compiled_distributional_loss_backward_is_finite()
