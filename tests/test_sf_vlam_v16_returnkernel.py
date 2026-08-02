import importlib.util
import sys
from copy import deepcopy
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
    "sf_vlam_v9_for_v16",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v16_returnkernel",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v16_returnkernel.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_latent_composition_has_exact_isolated_scalar_coordinate():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    scalar = torch.randn(257, device="cuda")
    rich = torch.randn(257, args.value_latent_dim - 1, device="cuda")
    latent = agent.compose_value_latent(scalar, rich)

    torch.testing.assert_close(latent @ agent.decoder_direction, scalar)
    torch.testing.assert_close(latent @ agent.decoder_null_basis, rich)
    agent.value_mean.fill_(12.0)
    agent.value_std.fill_(3.5)
    torch.testing.assert_close(agent.decode_value(latent), 12.0 + 3.5 * scalar)


def test_initial_scalar_and_rich_predictions_are_zero():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    obs = torch.randn(31, 17, device="cuda")
    latent = agent.get_value_latent(obs)

    assert latent.shape == (31, args.value_latent_dim)
    torch.testing.assert_close(agent.get_value(obs), torch.zeros(31, device="cuda"))
    torch.testing.assert_close(latent, torch.zeros_like(latent))


def test_v16_preserves_v9_actor_trunk_and_global_rng():
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


def test_value_stat_update_preserves_every_raw_prediction():
    agent = MODULE.Agent(_FakeEnv(), MODULE.Args()).cuda()
    obs = torch.randn(127, 17, device="cuda")
    with torch.no_grad():
        agent.scalar_value_head.weight.normal_()
        agent.scalar_value_head.bias.fill_(0.7)
        before = agent.get_value(obs)
        agent.update_value_stats(
            torch.tensor(43.0, device="cuda"),
            torch.tensor(17.0, device="cuda"),
            rate=0.6,
        )
        after = agent.get_value(obs)

    torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)


def test_sampled_return_respects_termination_truncation_and_missing_observation():
    rewards = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        device="cuda",
    )
    next_values = torch.tensor(
        [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]],
        device="cuda",
    )
    terminations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device="cuda",
    )
    boundaries = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device="cuda",
    )
    valids = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        device="cuda",
    )
    returns, valid = MODULE.compute_sampled_returns(
        rewards,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.5,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0 + 0.5 * 8.0, 3.0 + 0.5 * 9.0, 4.0 + 0.5 * (40.0 + 0.5 * 14.0)],
            [10.0 + 0.5 * 11.0, 20.0 + 0.5 * 12.0, 30.0 + 0.5 * 13.0, 40.0 + 0.5 * 14.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(returns, expected)
    assert valid.tolist() == [[True, True, False, True], [True, True, True, True]]


def test_regularizer_strongly_penalizes_duplicate_and_scalar_coordinates():
    scalar = torch.linspace(-2.0, 2.0, 4096, device="cuda")
    duplicate = scalar[:, None].repeat(1, 7)
    total, variance, correlation, scalar_correlation, mean = (
        MODULE.return_code_regularization(duplicate, scalar)
    )

    assert torch.isfinite(total)
    assert correlation.item() > 0.8
    assert scalar_correlation.item() > 0.8
    assert variance.item() < 0.1
    assert mean.item() < 1e-6


def test_auxiliary_trunk_gradient_is_projected_and_capped():
    parameter = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    scalar = {parameter: torch.tensor([1.0, 0.0, 0.0], device="cuda")}
    actor = {parameter: torch.tensor([1.0, 1.0, 0.0], device="cuda")}
    primary = {parameter: scalar[parameter] + actor[parameter]}
    auxiliary = {parameter: torch.tensor([-1.0, 2.0, 4.0], device="cuda")}
    cosines, retained = MODULE.protect_auxiliary_trunk_gradient(
        [scalar, actor],
        auxiliary,
        [parameter],
        primary,
        max_ratio=0.25,
    )

    assert cosines[0] < 0.0
    assert 0.0 < retained < 1.0
    assert torch.dot(scalar[parameter], auxiliary[parameter]).abs() < 1e-7
    assert torch.dot(actor[parameter], auxiliary[parameter]).abs() < 1e-7
    assert auxiliary[parameter].norm() <= 0.25 * primary[parameter].norm() + 1e-7


def test_compiled_three_way_backward_and_return_encoder_ema_are_finite():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    agent.trunk = MODULE.CompiledModule(agent.trunk, cudagraphs=False)
    encoder = MODULE.ReturnKernelEncoder(
        args.value_latent_dim - 1, args.return_hidden
    ).cuda()
    target_encoder = deepcopy(encoder).requires_grad_(False)
    obs = torch.randn(128, 17, device="cuda")
    native_action = torch.rand(128, 6, device="cuda").clamp(1e-4, 1.0 - 1e-4)
    _, _, logprob, entropy, _, latent = agent.get_action_and_value(
        obs, native_action, rich_trunk_enabled=True
    )
    scalar_target = torch.randn(128, device="cuda")
    return_sample = torch.randn(128, device="cuda")
    rich_target = target_encoder(return_sample)
    scalar_loss = (
        latent @ agent.decoder_direction - scalar_target
    ).square().mean()
    rich_loss = (
        latent @ agent.decoder_null_basis - rich_target.detach()
    ).square().mean()
    policy_loss = -0.01 * logprob.mean() - 0.01 * entropy.mean()

    agent.zero_grad(set_to_none=True)
    scalar_loss.backward(retain_graph=True)
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(
        agent.scalar_value_parameters(), args.critic_grad_clip
    ))
    agent.zero_grad(set_to_none=True)
    rich_loss.backward(retain_graph=True)
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(
        agent.rich_value_parameters(), args.rich_grad_clip
    ))
    agent.zero_grad(set_to_none=True)
    policy_loss.backward()
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(
        agent.actor_parameters(), args.actor_grad_clip
    ))

    code = encoder(return_sample)
    reg, *_ = MODULE.return_code_regularization(code, return_sample)
    (code.square().mean() + reg).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    before = next(target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(encoder.parameters()).add_(1.0)
    MODULE.ema_update(target_encoder, encoder, args.return_encoder_ema_rate)
    assert not torch.equal(next(target_encoder.parameters()), before)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_latent_composition_has_exact_isolated_scalar_coordinate()
    test_initial_scalar_and_rich_predictions_are_zero()
    test_v16_preserves_v9_actor_trunk_and_global_rng()
    test_value_stat_update_preserves_every_raw_prediction()
    test_sampled_return_respects_termination_truncation_and_missing_observation()
    test_regularizer_strongly_penalizes_duplicate_and_scalar_coordinates()
    test_auxiliary_trunk_gradient_is_projected_and_capped()
    test_compiled_three_way_backward_and_return_encoder_ema_are_finite()
