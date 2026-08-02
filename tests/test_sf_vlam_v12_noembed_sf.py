import importlib.util
import inspect
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
    "sf_vlam_v9_leakyrelu05sq_for_v12",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v12_noembed_sf",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v12_noembed_sf.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_phi_is_exact_reward_sufficient_basis_without_embedding_argument():
    assert list(inspect.signature(MODULE.phi_features).parameters) == ["obs", "action"]
    obs = torch.randn(19, 17, device="cuda")
    action = torch.randn(19, 6, device="cuda")
    phi = MODULE.phi_features(obs, action)

    assert phi.shape == (19, 17 + 6 + 6 + 1)
    torch.testing.assert_close(phi[:, :17], obs)
    torch.testing.assert_close(phi[:, 17:23], action)
    torch.testing.assert_close(phi[:, 23:29], action.square())
    torch.testing.assert_close(phi[:, -1], torch.ones_like(phi[:, -1]))


def test_linear_readout_exactly_represents_quadratic_control_reward():
    obs = torch.randn(97, 17, device="cuda")
    action = torch.randn(97, 6, device="cuda")
    phi = MODULE.phi_features(obs, action)

    weights = torch.zeros(phi.shape[-1], device="cuda")
    weights[8] = 2.75
    weights[23:29] = -0.1
    weights[-1] = 0.4
    reward = 2.75 * obs[:, 8] - 0.1 * action.square().sum(-1) + 0.4
    torch.testing.assert_close(phi @ weights, reward)


def test_reduced_head_shape_and_actor_trunk_rng_pairing_with_v9():
    args_base = BASE.Args()
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), args_base)
    expected_rng = torch.get_rng_state()

    args = MODULE.Args()
    torch.manual_seed(918)
    agent = MODULE.Agent(_FakeEnv(), args)
    actual_rng = torch.get_rng_state()

    assert agent.sf_dim == 17 + 2 * 6 + 1
    assert agent.critic_head.out_features == args.critic_mtp_horizon * agent.sf_dim
    torch.testing.assert_close(actual_rng, expected_rng)
    for actual, expected in zip(agent.trunk.parameters(), base.trunk.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for name in ("actor_alpha_head", "actor_beta_head"):
        actual_head = getattr(agent, name)
        expected_head = getattr(base, name)
        torch.testing.assert_close(actual_head.weight, expected_head.weight)
        torch.testing.assert_close(actual_head.bias, expected_head.bias)


def test_embedding_width_cannot_change_phi_or_sf_head_width():
    args = MODULE.Args(emb_dim=7)
    agent = MODULE.Agent(_FakeEnv(), args)
    assert agent.sf_dim == 30

    args = MODULE.Args(emb_dim=113)
    agent = MODULE.Agent(_FakeEnv(), args)
    assert agent.sf_dim == 30


def test_compiled_critic_forward_backward_is_finite():
    agent = MODULE.Agent(_FakeEnv(), MODULE.Args()).cuda()
    get_value = torch.compile(agent.get_value, mode="reduce-overhead")
    obs = torch.randn(128, 17, device="cuda", requires_grad=True)
    value_sf = get_value(obs)
    assert value_sf.shape == (128, 6, 30)

    # The production head starts at zero by design. Give it a nonzero value here so the
    # test exercises gradients through both the reduced head and the shared trunk.
    with torch.no_grad():
        agent.critic_head.weight.normal_(std=0.01)
    value_sf = get_value(obs)
    value_sf.square().mean().backward()
    assert torch.isfinite(value_sf).all()
    assert torch.isfinite(obs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.critic_parameters()
    )


def test_scalar_lambda_remains_default():
    assert MODULE.Args().per_dim_lambda is False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_phi_is_exact_reward_sufficient_basis_without_embedding_argument()
    test_linear_readout_exactly_represents_quadratic_control_reward()
    test_reduced_head_shape_and_actor_trunk_rng_pairing_with_v9()
    test_embedding_width_cannot_change_phi_or_sf_head_width()
    test_compiled_critic_forward_backward_is_finite()
    test_scalar_lambda_remains_default()
