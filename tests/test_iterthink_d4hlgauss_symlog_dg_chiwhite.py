from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.dg.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dg_chiwhite_v1 import (
    Agent as DGAgent,
    Args as DGArgs,
    dg_action_density_logprob,
    dg_policy_loss,
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
    assert dg.dg_chi_whiten is True
    assert dg.dg_chi_whiten_center is True
    assert dg.dg_chi_whiten_eps == 1e-8
    assert dg.update_epochs == 1

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

    gate, clipped_surprisal, raw_surprisal, chi_raw, chi_gate = delightful_gate(
        advantages,
        logprob,
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=False,
    )

    expected_raw_surprisal = torch.tensor([12.0, 12.0, -15.0, -15.0])
    expected_surprisal = torch.tensor([10.0, 10.0, -10.0, -10.0])
    expected_delight = advantages * expected_surprisal
    expected_gate = torch.sigmoid(expected_delight)

    assert torch.allclose(raw_surprisal, expected_raw_surprisal)
    assert torch.allclose(clipped_surprisal, expected_surprisal)
    assert torch.allclose(chi_raw, expected_delight)
    assert torch.allclose(chi_gate, expected_delight)
    assert torch.allclose(gate, expected_gate)


def test_chi_whitening_standardizes_only_gate_input():
    advantages = torch.tensor([1.0, 2.0, -1.0, -2.0])
    logprob = torch.tensor([-1.0, -2.0, -3.0, -4.0])

    gate, _, _, chi_raw, chi_gate = delightful_gate(
        advantages,
        logprob,
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=True,
        center_chi=True,
    )

    expected_raw = torch.tensor([1.0, 4.0, -3.0, -8.0])
    expected_gate_input = (expected_raw - expected_raw.mean()) / expected_raw.std(unbiased=False)

    assert torch.allclose(chi_raw, expected_raw)
    assert torch.allclose(chi_gate, expected_gate_input)
    assert torch.allclose(chi_gate.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(chi_gate.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(gate, torch.sigmoid(expected_gate_input))


def test_dg_gate_is_detachable_for_score_function_estimator():
    advantage = torch.tensor([1.5])
    logprob = torch.tensor([-2.0], requires_grad=True)
    gate, _, _, _, _ = delightful_gate(
        advantage,
        logprob,
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=False,
    )

    loss = dg_policy_loss(advantage, logprob, gate) * advantage.numel()
    loss.backward()

    assert torch.allclose(logprob.grad, -gate.detach() * advantage)


def test_paper_faithful_loss_has_no_ratio_or_ppo_clip_gradient():
    advantage = torch.tensor([1.5])
    old_logprob = torch.tensor([-10.0])
    new_logprob = torch.tensor([-2.0], requires_grad=True)
    gate, _, _, _, _ = delightful_gate(
        advantage,
        new_logprob.detach(),
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=False,
    )

    loss = dg_policy_loss(advantage, new_logprob, gate)
    loss.backward()

    ratio = (new_logprob.detach() - old_logprob).exp()
    assert ratio.item() > 1.2
    assert torch.allclose(new_logprob.grad, -gate.detach() * advantage)


def test_chi_whitened_loss_still_updates_with_raw_advantage():
    advantage = torch.tensor([0.25, 2.0, -1.0])
    new_logprob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    gate, _, _, chi_raw, chi_gate = delightful_gate(
        advantage,
        new_logprob.detach(),
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=True,
        center_chi=True,
    )

    loss = dg_policy_loss(advantage, new_logprob, gate)
    loss.backward()

    assert not torch.allclose(chi_raw, chi_gate)
    assert torch.allclose(new_logprob.grad, -gate.detach() * advantage / advantage.numel())


def test_dg_loss_uses_raw_update_advantage_not_normalized_surrogate():
    raw_advantage = torch.tensor([0.25, 2.0, -1.0])
    normalized_surrogate = (raw_advantage - raw_advantage.mean()) / raw_advantage.std()
    new_logprob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    gate, _, _, _, _ = delightful_gate(
        raw_advantage,
        new_logprob.detach(),
        eta=1.0,
        clip_bound=10.0,
        whiten_chi=True,
        center_chi=True,
    )

    loss = dg_policy_loss(raw_advantage, new_logprob, gate)
    loss.backward()

    raw_grad = -gate.detach() * raw_advantage / raw_advantage.numel()
    normalized_grad = -gate.detach() * normalized_surrogate / raw_advantage.numel()
    assert torch.allclose(new_logprob.grad, raw_grad)
    assert not torch.allclose(new_logprob.grad, normalized_grad)


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
