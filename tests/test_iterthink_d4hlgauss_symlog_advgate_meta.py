import gymnasium as gym
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_advgate_meta_v1 import (
    Agent,
    Args,
    gradient_alignment_loss,
    ppo_actor_loss_from_adv,
)


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_meta_gate_defaults_use_beta_multiplier_without_rank_bc():
    args = Args()

    assert args.adv_gate is True
    assert args.adv_gate_coef == 0.02
    assert args.adv_gate_grad_clip == 0.25


def test_adv_gate_beta_head_is_action_conditioned_and_attached_to_trunk():
    args = Args(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    agent = Agent(DummyVecEnv(), args)
    obs = torch.randn(8, 4)
    z1 = torch.full((8, 2), 0.2)
    z2 = torch.full((8, 2), 0.8)

    dist1 = agent.get_adv_gate_dist(obs, z1)
    dist2 = agent.get_adv_gate_dist(obs, z2)

    assert dist1.mean.shape == (8, 1)
    assert torch.all(dist1.concentration1 > 1.0)
    assert torch.all(dist1.concentration0 > 1.0)
    assert not torch.allclose(dist1.mean, dist2.mean)

    loss = -dist1.mean.mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()

    trunk = agent.trunk if args.share_backbone else agent.critic_trunk
    assert any(p.grad is not None for p in trunk.parameters())
    assert any(p.grad is not None for p in agent.adv_gate_parameters())


def test_gradient_alignment_loss_trains_gate_from_ppo_gradient_geometry():
    args = Args(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    agent = Agent(DummyVecEnv(), args)
    obs = torch.randn(16, 4)
    z = torch.rand(16, 2).clamp(1e-4, 1 - 1e-4)
    _, _, logprob, _, _ = agent.get_action_and_value(obs, z)
    ratio = (logprob - logprob.detach()).exp()
    adv = torch.linspace(-1.0, 1.0, 16)
    gate = agent.get_adv_gate_dist(obs, z).mean.squeeze(-1)

    train_loss = ppo_actor_loss_from_adv(adv[:8] * gate[:8], ratio[:8], 0.2, 0.28)
    valid_loss = ppo_actor_loss_from_adv(adv[8:], ratio[8:], 0.2, 0.28)
    gate_loss = gradient_alignment_loss(train_loss, valid_loss, agent.actor_parameters(), 1e-8)

    agent.zero_grad(set_to_none=True)
    gate_loss.backward()

    assert torch.isfinite(gate_loss)
    assert any(p.grad is not None for p in agent.adv_gate_parameters())
