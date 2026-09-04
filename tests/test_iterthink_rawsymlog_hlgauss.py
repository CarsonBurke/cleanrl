from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.iterthink.critic_variants.ppo_continuous_action_iterthink_v24_beta_rawsymlog_hlgauss_v1 import (
    Args,
    value_support_bounds,
)
from cleanrl.iterthink.critic_variants.ppo_continuous_action_iterthink_v24_beta_rawsymlog_hlgauss_nocriticbias_v2 import (
    Agent as NoCriticBiasAgent,
    Args as NoCriticBiasArgs,
    value_support_bounds as nocriticbias_value_support_bounds,
)
from cleanrl.iterthink.critic_variants.ppo_continuous_action_iterthink_v24_beta_rawsymlog_hlgauss_criticclip05_v3 import (
    Args as CriticClipArgs,
)
from cleanrl.shared.hl_gauss import HLGaussSupport, symlog


def test_raw_return_symlog_support_uses_transformed_bounds():
    args = SimpleNamespace(v_min=-20000.0, v_max=20000.0, value_symlog=True)
    support_min, support_max = value_support_bounds(args)
    expected = symlog(torch.tensor([args.v_min, args.v_max])).tolist()

    assert support_min == expected[0]
    assert support_max == expected[1]

    support = HLGaussSupport(
        num_bins=511,
        v_min=support_min,
        v_max=support_max,
        sigma_ratio=Args.value_sigma_to_bin_ratio,
        device=torch.device("cpu"),
        use_symlog=True,
        support_is_edges=True,
    )
    assert support.support.abs().max() < 11.0

    targets = torch.tensor([0.0, 20000.0, 30000.0])
    probs = support.project(targets)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(probs[1], probs[2], atol=1e-6)

    logits = torch.full((1, support.num_bins), -100.0)
    zero_bin = support.support.abs().argmin()
    logits[0, zero_bin] = 100.0
    decoded = support.to_scalar(logits)
    assert decoded.abs().item() < 0.05


def test_rawsymlog_criticclip_v3_only_relaxes_critic_clip_default():
    v2_args = NoCriticBiasArgs()
    v3_args = CriticClipArgs()

    assert v3_args.actor_grad_clip == v2_args.actor_grad_clip == 0.25
    assert v2_args.critic_grad_clip == 0.25
    assert v3_args.critic_grad_clip == 0.5
    assert v3_args.normalize_reward is False
    assert v3_args.clip_reward is False


def test_rawsymlog_ablation_disables_reward_return_normalization_by_default():
    args = Args()
    assert args.normalize_reward is False
    assert args.clip_reward is False

    v2_args = NoCriticBiasArgs()
    assert v2_args.normalize_reward is False
    assert v2_args.clip_reward is False
    assert v2_args.value_symlog is True
    assert v2_args.value_sigma_to_bin_ratio == 0.5

    support_min, support_max = nocriticbias_value_support_bounds(v2_args)
    expected = symlog(torch.tensor([v2_args.v_min, v2_args.v_max])).tolist()
    assert support_min == expected[0]
    assert support_max == expected[1]


def test_rawsymlog_nocriticbias_v2_starts_with_uniform_value_logits():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    args = NoCriticBiasArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    agent = NoCriticBiasAgent(DummyVecEnv(), args)
    obs = torch.zeros(3, 4)

    logits = agent.get_value(obs)

    assert agent.critic_head.bias is None
    assert torch.allclose(agent.critic_head.weight, torch.zeros_like(agent.critic_head.weight))
    assert torch.allclose(logits, torch.zeros_like(logits))
