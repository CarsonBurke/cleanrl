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
    "sf_vlam_v9_leakyrelu05sq_for_v13",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v13_rewardanchored",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_phi_has_exact_reward_anchor_and_expected_auxiliary_layout():
    assert list(inspect.signature(MODULE.phi_features).parameters) == [
        "reward",
        "emb",
        "obs",
        "action",
    ]
    reward = torch.randn(19, device="cuda")
    emb = torch.randn(19, 32, device="cuda")
    obs = torch.randn(19, 17, device="cuda")
    action = torch.randn(19, 6, device="cuda")
    phi = MODULE.phi_features(reward, emb, obs, action)

    assert phi.shape == (19, 1 + 32 + 17 + 6 + 6 + 1)
    torch.testing.assert_close(phi[:, 0], reward)
    torch.testing.assert_close(phi[:, 1:33], emb)
    torch.testing.assert_close(phi[:, 33:50], obs)
    torch.testing.assert_close(phi[:, 50:56], action)
    torch.testing.assert_close(phi[:, 56:62], action.square())
    torch.testing.assert_close(phi[:, -1], torch.ones_like(phi[:, -1]))

    reward_readout = torch.zeros(phi.shape[-1], device="cuda")
    reward_readout[0] = 1.0
    torch.testing.assert_close(phi @ reward_readout, reward)


def test_reward_successor_coordinate_is_exactly_scalar_gae():
    torch.manual_seed(31)
    t, b, d = 13, 5, 9
    reward = torch.randn(t, b, device="cuda")
    phi = torch.randn(t, b, d, device="cuda")
    phi[..., 0] = reward
    psi_cur = torch.randn(t, b, d, device="cuda")
    psi_next = torch.randn(t, b, d, device="cuda")
    terminations = torch.zeros(t, b, device="cuda")
    boundaries = torch.zeros(t, b, device="cuda")
    valids = torch.ones(t, b, device="cuda")
    terminations[4, 2] = 1.0
    boundaries[4, 2] = 1.0
    boundaries[8, 1] = 1.0
    valids[8, 1] = 1.0
    gamma, gae_lambda = 0.99, 0.95
    lam_vec = torch.full((d,), gae_lambda, device="cuda")

    residual = MODULE.successor_lambda_residual(
        phi,
        psi_cur,
        psi_next,
        terminations,
        boundaries,
        valids,
        gamma,
        lam_vec,
    )

    expected = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for step in reversed(range(t)):
        boot = (1.0 - terminations[step]) * valids[step]
        cont = 1.0 - boundaries[step]
        delta = reward[step] + gamma * boot * psi_next[step, :, 0] - psi_cur[step, :, 0]
        last = delta + gamma * gae_lambda * cont * last
        expected[step] = last
    torch.testing.assert_close(residual[..., 0], expected)


def test_wider_head_preserves_v9_actor_trunk_and_global_rng():
    args_base = BASE.Args()
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), args_base)
    expected_rng = torch.get_rng_state()

    args = MODULE.Args()
    torch.manual_seed(918)
    agent = MODULE.Agent(_FakeEnv(), args)
    actual_rng = torch.get_rng_state()

    assert agent.sf_dim == 1 + 32 + 17 + 2 * 6 + 1
    assert agent.critic_head.out_features == args.critic_mtp_horizon * agent.sf_dim
    torch.testing.assert_close(actual_rng, expected_rng)
    for actual, expected in zip(agent.trunk.parameters(), base.trunk.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for name in ("actor_alpha_head", "actor_beta_head"):
        actual_head = getattr(agent, name)
        expected_head = getattr(base, name)
        torch.testing.assert_close(actual_head.weight, expected_head.weight)
        torch.testing.assert_close(actual_head.bias, expected_head.bias)


def test_compiled_critic_and_grouped_loss_backward_are_finite():
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args).cuda()
    get_value = torch.compile(agent.get_value, mode="reduce-overhead")
    obs = torch.randn(128, 17, device="cuda", requires_grad=True)
    with torch.no_grad():
        agent.critic_head.weight.normal_(std=0.01)
    value_sf = get_value(obs)
    assert value_sf.shape == (128, 6, 63)

    target = torch.randn_like(value_sf)
    mask = torch.ones(value_sf.shape[:2], device="cuda")
    err = value_sf - target
    direct_value_loss = err[:, 0, 0].square().mean()
    occupancy_loss = (err[..., 1:].square().mean(-1) * mask).sum() / mask.sum()
    future_mask = mask[:, 1:]
    future_value_loss = (err[:, 1:, 0].square() * future_mask).sum() / future_mask.sum()
    aux_loss = 0.5 * (occupancy_loss + future_value_loss)
    loss = mask.sum(-1).mean() * (
        direct_value_loss + args.aux_sf_coef * aux_loss
    ) / (1.0 + args.aux_sf_coef)
    loss.backward()

    assert torch.isfinite(value_sf).all()
    assert torch.isfinite(obs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.critic_parameters()
    )


def test_anchored_defaults_use_scalar_policy_gae():
    args = MODULE.Args()
    assert args.per_dim_lambda is False
    assert args.vector_adv is False
    assert args.aux_sf_coef == 1.0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_phi_has_exact_reward_anchor_and_expected_auxiliary_layout()
    test_reward_successor_coordinate_is_exactly_scalar_gae()
    test_wider_head_preserves_v9_actor_trunk_and_global_rng()
    test_compiled_critic_and_grouped_loss_backward_are_finite()
    test_anchored_defaults_use_scalar_policy_gae()
