from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.dg.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dg_paper_v1 import (
    Agent as DGAgent,
    Args as DGArgs,
    dg_action_density_logprob,
    delightful_gate,
    value_support_bounds as dg_value_support_bounds,
)
from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_v1 import (
    Agent as BaseAgent,
    Args as BaseArgs,
    value_support_bounds as base_value_support_bounds,
)


def test_dg_defaults_preserve_value_and_policy_defaults():
    base = BaseArgs()
    dg = DGArgs()

    assert dg.delightful_pg is True
    assert dg.dg_eta == 1.0
    assert dg.dg_surprisal_clip == 10.0
    assert dg.dg_surprisal_source == "current"

    assert dg.actor_dist == base.actor_dist == "beta"
    assert base.adv_transform == "rankgauss"
    assert dg.adv_transform == "v10"
    assert base.norm_adv is True
    assert dg.norm_adv is False
    assert dg.target_kl is None
    assert dg.clip_coef == base.clip_coef == 0.2
    assert dg.clip_coef_high == base.clip_coef_high == 0.28
    assert dg.value_symlog == base.value_symlog is True
    assert dg.num_bins == base.num_bins == 511
    assert dg.v_min == base.v_min == -10.0
    assert dg.v_max == base.v_max == 10.0
    assert dg.value_sigma_to_bin_ratio == base.value_sigma_to_bin_ratio == 2.0
    assert dg.critic_init_tau == base.critic_init_tau == 0.5


def test_dg_preserves_symlog_support_and_critic_init():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    base_args = BaseArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31)
    dg_args = DGArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31)

    assert dg_value_support_bounds(dg_args) == base_value_support_bounds(base_args)

    torch.manual_seed(123)
    base_agent = BaseAgent(DummyVecEnv(), base_args)
    torch.manual_seed(123)
    dg_agent = DGAgent(DummyVecEnv(), dg_args)

    assert torch.allclose(dg_agent.critic_head.weight, base_agent.critic_head.weight)
    assert torch.allclose(dg_agent.critic_head.bias, base_agent.critic_head.bias)


def test_delightful_gate_matches_algorithm_2_with_surprisal_clipping():
    advantages = torch.tensor([2.0, -2.0, 0.5, -0.5])
    logprob = torch.tensor([-12.0, -12.0, 15.0, 15.0])

    gate, clipped_surprisal, raw_surprisal, delight = delightful_gate(
        advantages,
        logprob,
        eta=1.0,
        clip_bound=10.0,
    )

    expected_raw_surprisal = torch.tensor([12.0, 12.0, -15.0, -15.0])
    expected_surprisal = torch.tensor([10.0, 10.0, -10.0, -10.0])
    expected_delight = advantages * expected_surprisal
    expected_gate = torch.sigmoid(expected_delight)

    assert torch.allclose(raw_surprisal, expected_raw_surprisal)
    assert torch.allclose(clipped_surprisal, expected_surprisal)
    assert torch.allclose(delight, expected_delight)
    assert torch.allclose(gate, expected_gate)


def test_dg_gate_is_detachable_for_score_function_estimator():
    advantage = torch.tensor([1.5])
    logprob = torch.tensor([-2.0], requires_grad=True)
    gate, _, _, _ = delightful_gate(advantage, logprob, eta=1.0, clip_bound=10.0)

    loss = -(gate.detach() * advantage * logprob).sum()
    loss.backward()

    assert torch.allclose(logprob.grad, -gate.detach() * advantage)


def test_paper_faithful_loss_has_no_ratio_or_ppo_clip_gradient():
    advantage = torch.tensor([1.5])
    old_logprob = torch.tensor([-10.0])
    new_logprob = torch.tensor([-2.0], requires_grad=True)
    gate, _, _, _ = delightful_gate(advantage, new_logprob.detach(), eta=1.0, clip_bound=10.0)

    loss = -(gate.detach() * advantage * new_logprob).mean()
    loss.backward()

    ratio = (new_logprob.detach() - old_logprob).exp()
    assert ratio.item() > 1.2
    assert torch.allclose(new_logprob.grad, -gate.detach() * advantage)


def test_beta_dg_surprisal_uses_action_density_not_ratio_logprob():
    agent = SimpleNamespace(
        actor_dist="beta",
        action_low=torch.tensor([-1.0, -2.0]),
        action_high=torch.tensor([1.0, 2.0]),
    )
    ppo_logprob = torch.tensor([3.0, -1.0])

    action_logprob = dg_action_density_logprob(agent, ppo_logprob)

    expected_logdet = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(4.0))
    assert torch.allclose(action_logprob, ppo_logprob - expected_logdet)
