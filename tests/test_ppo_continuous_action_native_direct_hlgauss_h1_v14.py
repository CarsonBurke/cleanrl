import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "native_pg" / "ppo_continuous_action_native_direct_hlgauss_h1_v14.py"
SPEC = importlib.util.spec_from_file_location("native_direct_hlgauss_h1_v14", SCRIPT)
v14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v14)

V12_SCRIPT = Path(__file__).parents[1] / "cleanrl" / "native_pg" / "ppo_continuous_action_native_direct_trust_v12.py"
V12_SPEC = importlib.util.spec_from_file_location("native_direct_trust_v12_for_v14_test", V12_SCRIPT)
v12 = importlib.util.module_from_spec(V12_SPEC)
V12_SPEC.loader.exec_module(v12)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_defaults_keep_state_sigma_disable_kl_stop_and_disable_mtp():
    args = v14.Args()

    assert args.sigma_mode == "state"
    assert args.max_mean_kl == 0.0
    assert args.critic_mtp_horizon == 1
    assert args.value_num_bins == 511


def test_single_critic_head_has_neutral_value_initialization():
    args = v14.Args()
    agent = v14.Agent(DummyVectorEnv(), args)
    observations = torch.randn(5, 3)

    logits = agent.get_value_logits(observations)
    values = agent.get_value(observations)

    assert logits.shape == (5, 1, args.value_num_bins)
    assert torch.count_nonzero(agent.critic_head.weight) == 0
    torch.testing.assert_close(values, torch.zeros_like(values), atol=2e-5, rtol=0.0)


def test_state_actor_initialization_is_identical_to_v12():
    torch.manual_seed(1)
    baseline = v12.Agent(DummyVectorEnv(), sigma_mode="state")
    torch.manual_seed(1)
    candidate = v14.Agent(DummyVectorEnv(), v14.Args())

    baseline_actor = {name: parameter.detach() for name, parameter in baseline.named_parameters() if name.startswith("actor_")}
    candidate_actor = {
        name: parameter.detach() for name, parameter in candidate.named_parameters() if name.startswith("actor_")
    }

    assert baseline_actor.keys() == candidate_actor.keys()
    for name in baseline_actor:
        assert torch.equal(candidate_actor[name], baseline_actor[name]), name


def test_horizon_one_contains_only_the_current_return():
    returns = torch.randn(8, 3)
    boundaries = torch.ones_like(returns)

    targets, mask = v14.build_mtp_targets(returns, boundaries, horizon=1)

    torch.testing.assert_close(targets[..., 0], returns)
    assert torch.all(mask)
