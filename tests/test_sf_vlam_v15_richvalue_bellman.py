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
    "sf_vlam_v9_for_v15",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v15_richvalue_bellman",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v15_richvalue_bellman.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_reward_code_and_bellman_composition_decode_exactly():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    encoder = MODULE.RewardValueEncoder(
        args.value_latent_dim,
        args.reward_hidden,
        args.reward_input_scale,
        args.value_scale,
    ).cuda()
    reward = torch.randn(257, device="cuda") * 8.0
    next_latent = torch.randn(257, args.value_latent_dim, device="cuda")
    continuation = torch.randint(0, 2, (257,), device="cuda").float()

    reward_code = encoder(reward, agent.decoder_direction)
    torch.testing.assert_close(
        agent.decode_value(reward_code), reward, atol=2e-5, rtol=2e-5
    )
    target = reward_code + args.gamma * continuation[:, None] * next_latent
    expected = reward + args.gamma * continuation * agent.decode_value(next_latent)
    torch.testing.assert_close(
        agent.decode_value(target), expected, atol=5e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        agent.decoder_null_basis.T @ agent.decoder_null_basis,
        torch.eye(args.value_latent_dim - 1, device="cuda"),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        agent.decoder_direction @ agent.decoder_null_basis,
        torch.zeros(args.value_latent_dim - 1, device="cuda"),
        atol=1e-5,
        rtol=1e-5,
    )


def test_critic_is_primary_latent_and_initial_value_is_zero():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    obs = torch.randn(31, 17, device="cuda")
    latent = agent.get_value_latent(obs)
    value = agent.get_value(obs)

    assert latent.shape == (31, args.value_latent_dim)
    assert agent.critic_head.out_features == args.value_latent_dim
    torch.testing.assert_close(value, agent.decode_value(latent))
    torch.testing.assert_close(value, torch.zeros_like(value))


def test_v15_preserves_v9_actor_trunk_and_global_rng():
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), BASE.Args())
    expected_rng = torch.get_rng_state()

    torch.manual_seed(918)
    agent = MODULE.Agent(_FakeEnv(), MODULE.Args())
    actual_rng = torch.get_rng_state()

    torch.testing.assert_close(actual_rng, expected_rng)
    for actual, expected in zip(agent.trunk.parameters(), base.trunk.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for name in ("actor_alpha_head", "actor_beta_head"):
        actual_head = getattr(agent, name)
        expected_head = getattr(base, name)
        torch.testing.assert_close(actual_head.weight, expected_head.weight)
        torch.testing.assert_close(actual_head.bias, expected_head.bias)


def test_scalar_gae_censors_missing_truncation_but_keeps_terminal_and_valid_truncation():
    rewards = torch.tensor([[3.0, 3.0, 3.0]], device="cuda")
    values = torch.tensor([[1.0, 1.0, 1.0]], device="cuda")
    next_values = torch.tensor([[9.0, 9.0, 9.0]], device="cuda")
    terminations = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    boundaries = torch.ones_like(terminations)
    transition_valids = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    advantages, returns = MODULE.compute_scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        transition_valids,
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.0, 2.0, 3.0 + 0.99 * 9.0 - 1.0]], device="cuda"),
    )
    torch.testing.assert_close(returns, advantages + values)

    bootstrap = (1.0 - terminations) * transition_valids
    latent_valid = torch.logical_or(terminations.bool(), transition_valids.bool())

    torch.testing.assert_close(
        bootstrap, torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    )
    assert latent_valid.tolist() == [[False, True, True]]


def test_compiled_dual_backward_reward_grounding_and_ema_are_finite():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    target_critic = MODULE.TargetCritic(agent).cuda()
    agent.trunk = MODULE.CompiledModule(agent.trunk, cudagraphs=False)
    target_critic.trunk = MODULE.CompiledModule(target_critic.trunk, cudagraphs=False)
    reward_encoder = MODULE.RewardValueEncoder(
        args.value_latent_dim,
        args.reward_hidden,
        args.reward_input_scale,
        args.value_scale,
    ).cuda()
    obs = torch.randn(128, 17, device="cuda")
    native_action = torch.rand(128, 6, device="cuda").clamp(1e-4, 1.0 - 1e-4)
    _, _, logprob, entropy, value, latent = agent.get_action_and_value(obs, native_action)
    target = torch.randn_like(latent)

    scalar_loss = (value / args.value_scale).square().mean()
    rich_loss = MODULE.project_null(
        latent - target, agent.decoder_direction.detach()
    ).square().mean()
    value_loss = scalar_loss + rich_loss
    policy_loss = -0.01 * logprob.mean() - 0.01 * entropy.mean()
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    agent.zero_grad(set_to_none=True)
    value_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
    value_grads = [
        (parameter, parameter.grad.detach().clone())
        for parameter in critic_params
        if parameter.grad is not None
    ]
    agent.zero_grad(set_to_none=True)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
    for parameter, gradient in value_grads:
        parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient

    reward = torch.randn(128, device="cuda")
    reward_code = reward_encoder(reward, agent.decoder_direction)
    reward_reg, variance_loss, covariance_loss, mean_loss = (
        MODULE.reward_feature_regularization(
            reward_code, agent.decoder_null_basis, args.reward_feature_std
        )
    )
    reward_reg.backward()

    assert torch.isfinite(value_loss + policy_loss + reward_reg)
    assert variance_loss.item() > 0.0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in reward_encoder.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.parameters()
    )
    target_before = target_critic.critic_head.weight.detach().clone()
    with torch.no_grad():
        agent.critic_head.weight.add_(1.0)
    MODULE.ema_update(
        target_critic.trunk, agent.trunk, args.target_ema_rate
    )
    MODULE.ema_update(
        target_critic.critic_head, agent.critic_head, args.target_ema_rate
    )
    assert not torch.equal(target_critic.critic_head.weight, target_before)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_reward_code_and_bellman_composition_decode_exactly()
    test_critic_is_primary_latent_and_initial_value_is_zero()
    test_v15_preserves_v9_actor_trunk_and_global_rng()
    test_scalar_gae_censors_missing_truncation_but_keeps_terminal_and_valid_truncation()
    test_compiled_dual_backward_reward_grounding_and_ema_are_finite()
