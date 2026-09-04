import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cleanrl.shared.hl_gauss import HLGaussSupport, symexp

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "native_pg" / "ppo_continuous_action_native_direct_hlgauss_v13.py"
SPEC = importlib.util.spec_from_file_location("native_direct_hlgauss_v13", SCRIPT)
v13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v13)

V12_SCRIPT = Path(__file__).parents[1] / "cleanrl" / "native_pg" / "ppo_continuous_action_native_direct_trust_v12.py"
V12_SPEC = importlib.util.spec_from_file_location("native_direct_trust_v12_for_v13_test", V12_SCRIPT)
v12 = importlib.util.module_from_spec(V12_SPEC)
V12_SPEC.loader.exec_module(v12)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def make_support(args):
    return HLGaussSupport(
        args.value_num_bins,
        args.value_min,
        args.value_max,
        args.value_sigma_to_bin_ratio,
        "cpu",
        use_symlog=True,
        support_is_edges=True,
    )


def test_defaults_match_the_proven_raw_return_hlgauss_geometry():
    args = v13.Args()

    assert args.value_num_bins == 511
    assert np.isclose(args.value_min, -np.log1p(20_000.0))
    assert np.isclose(args.value_max, np.log1p(20_000.0))
    assert args.value_sigma_to_bin_ratio == 2.0
    assert args.critic_mtp_horizon == 6
    assert args.num_steps == 128
    assert args.num_envs * args.num_steps == args.target_actor_batch_size == 2048


def test_neutral_head_decodes_zero_and_preserves_expected_shapes():
    torch.manual_seed(1)
    args = v13.Args()
    agent = v13.Agent(DummyVectorEnv(), args)
    observations = torch.randn(7, 3)

    logits = agent.get_value_logits(observations)
    values = agent.get_value(observations)

    assert logits.shape == (7, args.critic_mtp_horizon, args.value_num_bins)
    assert values.shape == (7, 1)
    assert agent.critic_head.bias is None
    assert torch.count_nonzero(agent.critic_head.weight) == 0
    torch.testing.assert_close(values, torch.zeros_like(values), atol=2e-5, rtol=0.0)


def test_actor_initialization_is_identical_to_v12_at_the_same_seed():
    torch.manual_seed(1)
    baseline = v12.Agent(DummyVectorEnv(), sigma_mode="state")
    torch.manual_seed(1)
    candidate = v13.Agent(DummyVectorEnv(), v13.Args())

    baseline_actor = {name: parameter.detach() for name, parameter in baseline.named_parameters() if name.startswith("actor_")}
    candidate_actor = {
        name: parameter.detach() for name, parameter in candidate.named_parameters() if name.startswith("actor_")
    }

    assert baseline_actor.keys() == candidate_actor.keys()
    for name in baseline_actor:
        assert torch.equal(candidate_actor[name], baseline_actor[name]), name


def test_decode_is_expected_raw_value_not_inverse_of_mean_coordinate():
    args = v13.Args()
    agent = v13.Agent(DummyVectorEnv(), args)
    logits = torch.full((1, args.value_num_bins), -100.0)
    logits[0, args.value_num_bins // 2] = 0.0
    logits[0, -1] = 0.0

    probabilities = torch.softmax(logits, dim=-1)
    decoded = agent.decode_value_logits(logits)
    expected_raw = (probabilities * agent.raw_value_support).sum(dim=-1)
    support = make_support(args)
    inverse_mean_coordinate = symexp((probabilities * support.support).sum(dim=-1))

    torch.testing.assert_close(decoded, expected_raw)
    assert not torch.allclose(decoded, inverse_mean_coordinate)


def test_hlgauss_targets_are_smooth_normalized_and_inside_support():
    args = v13.Args()
    support = make_support(args)
    targets = torch.tensor([-1_000.0, 0.0, 500.0])

    probabilities = support.project(targets)

    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3))
    assert torch.all((probabilities > 0).sum(dim=-1) > 1)
    assert torch.all(probabilities[:, 0] + probabilities[:, -1] < 1e-6)


def test_mtp_targets_never_cross_termination_or_truncation_boundaries():
    returns = torch.arange(10.0).reshape(5, 2)
    boundaries = torch.zeros_like(returns)
    boundaries[1, 0] = 1.0
    boundaries[2, 1] = 1.0

    targets, mask = v13.build_mtp_targets(returns, boundaries, horizon=4)

    torch.testing.assert_close(targets[0, 0], torch.tensor([0.0, 2.0, 4.0, 6.0]))
    assert torch.equal(mask[0, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(targets[1, 1], torch.tensor([3.0, 5.0, 7.0, 9.0]))
    assert torch.equal(mask[1, 1], torch.tensor([True, True, False, False]))
    assert torch.all(mask[:, :, 0])
    assert not torch.any(mask[-1, :, 1:])


def test_source_changes_only_the_critic_objective_and_keeps_raw_rewards():
    source = SCRIPT.read_text()

    assert "NormalizeReward" not in source
    assert "TransformReward" not in source
    assert "actor_loss = -(b_advantages * gaussian_logprob).mean()" in source
    assert "HLGaussSupport(" in source
    assert "value_cross_entropy" in source
    assert "nn.functional.mse_loss" not in source
    assert "losses/value_mse" in source
    assert "bootstrap_observations(next_obs_np, truncations, infos)" in source
