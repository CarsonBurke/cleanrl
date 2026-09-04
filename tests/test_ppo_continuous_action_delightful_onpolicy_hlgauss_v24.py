import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "delightful" / "onpolicy" / "ppo_continuous_action_delightful_onpolicy_hlgauss_v24.py"
SPEC = importlib.util.spec_from_file_location("delightful_onpolicy_hlgauss_v24", SCRIPT)
v24 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v24)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_defaults_define_one_fresh_256_transition_actor_batch():
    args = v24.Args()

    assert args.num_envs * args.num_steps == 256
    assert args.target_actor_batch_size == 256
    assert args.num_minibatches == 1
    assert args.sigma_mode == "state"
    assert args.critic_mtp_horizon == 1


def test_delightful_gate_matches_detached_definition():
    advantages = torch.tensor([2.0, -3.0], requires_grad=True)
    action_logprobs = torch.tensor([-0.5, -2.0], requires_grad=True)

    gate, surprisal, delight = v24.delightful_gate(advantages, action_logprobs, eta=2.0)

    torch.testing.assert_close(surprisal, torch.tensor([0.5, 2.0]))
    torch.testing.assert_close(delight, torch.tensor([1.0, -6.0]))
    torch.testing.assert_close(gate, torch.sigmoid(torch.tensor([0.5, -3.0])))
    assert not gate.requires_grad
    assert not surprisal.requires_grad
    assert not delight.requires_grad


def test_delightful_gate_clips_surprisal_and_validates_scale():
    _, surprisal, _ = v24.delightful_gate(
        torch.ones(2),
        torch.tensor([-100.0, 100.0]),
        surprisal_clip=10.0,
    )

    torch.testing.assert_close(surprisal, torch.tensor([10.0, -10.0]))
    with pytest.raises(ValueError, match="dg_eta"):
        v24.delightful_gate(torch.ones(1), torch.zeros(1), eta=0.0)
    with pytest.raises(ValueError, match="dg_surprisal_clip"):
        v24.delightful_gate(torch.ones(1), torch.zeros(1), surprisal_clip=0.0)


def test_single_critic_head_has_neutral_value_initialization():
    args = v24.Args()
    agent = v24.Agent(DummyVectorEnv(), args)
    observations = torch.randn(5, 3)

    logits = agent.get_value_logits(observations)
    values = agent.get_value(observations)

    assert logits.shape == (5, 1, args.value_num_bins)
    assert torch.count_nonzero(agent.critic_head.weight) == 0
    torch.testing.assert_close(values, torch.zeros_like(values), atol=2e-5, rtol=0.0)


def test_critic_decodes_expectation_in_symlog_coordinates_before_symexp():
    args = v24.Args(value_num_bins=3, value_min=-2.0, value_max=2.0)
    support = torch.tensor([-1.0, 0.0, 1.0])
    agent = v24.Agent(DummyVectorEnv(), args, value_support=support)
    probabilities = torch.tensor([[0.1, 0.2, 0.7]])
    logits = probabilities.log()

    decoded = agent.decode_value_logits(logits)
    expected = v24.symexp((probabilities * support).sum(dim=-1))
    wrong_raw_space_expectation = (probabilities * v24.symexp(support)).sum(dim=-1)

    torch.testing.assert_close(decoded, expected)
    assert not torch.allclose(decoded, wrong_raw_space_expectation)


def test_gae_bootstraps_truncation_but_stops_trace():
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    values = torch.zeros_like(rewards)
    terminations = torch.zeros_like(rewards)
    truncations = torch.tensor([[0.0], [1.0], [0.0]])
    truncation_values = torch.tensor([[0.0], [10.0], [0.0]])

    advantages, returns = v24.compute_gae(
        rewards,
        values,
        terminations,
        truncations,
        truncation_values,
        rollout_tail_value=torch.tensor([0.0]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages[:, 0], torch.tensor([10.9, 11.0, 100.0]))
    torch.testing.assert_close(returns, advantages)


def test_actor_update_source_is_single_pass_without_off_policy_machinery():
    source = SCRIPT.read_text()

    assert source.count("actor_optimizer.step()") == 1
    assert "actor_loss = -(actor_weights * gaussian_logprob).mean()" in source
    assert "actor_weights = (gate * b_advantages).detach()" in source
    assert "action_logprob = action_logprob.clone()" in source
    assert "ReplayBuffer" not in source
    assert "importance_ratio" not in source
    assert "backtrack_parameters" not in source
    assert "max_mean_kl" not in source
