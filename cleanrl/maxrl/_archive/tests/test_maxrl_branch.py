"""Contracts for the per-state-group MaxRL variant (ppo_continuous_action_maxrl_branch_v1).

The whole construction rests on two mechanical claims that are easy to get wrong and
invisible in a loss curve: that cloning really makes a group share one state, and that
the (segments, groups, members) reshape lines each branch's outcome up with its own
timesteps. Both are pinned here.
"""

import importlib

import numpy as np
import pytest
import torch

branch = importlib.import_module("cleanrl.ppo_continuous_action_maxrl_branch_v1")
pure = importlib.import_module("cleanrl.ppo_continuous_action_maxrl_pure_v1")


def make_args(**overrides):
    args = branch.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_maclaurin_weights_agree_with_the_pure_variant():
    """Both files carry the estimator inline (single-file rule); they must not drift."""
    for group_size in (2, 4, 8, 16):
        successes = torch.arange(0, group_size + 1, dtype=torch.float64)
        for order in (1, 2, 8, 32):
            ours = branch.maclaurin_weights(successes, group_size, order)
            theirs = pure.maclaurin_weights(successes, group_size, order)
            for actual, expected in zip(ours, theirs):
                torch.testing.assert_close(actual, expected)


def test_pass_rate_is_taken_within_a_group_but_the_bar_is_global():
    """Cross-group variation in p is the entire reason this file exists."""
    args = make_args(group_size=4, maxrl_estimator="maxrl")
    # One easy group (3 of 4 beat the bar) and one hard group (1 of 4).
    returns = torch.tensor([[[3.0, 3.0, 3.0, -1.0], [3.0, -1.0, -1.0, -1.0]]])
    advantages, diagnostics = branch.group_advantages(returns, torch.tensor(0.0), args)
    assert advantages.shape == returns.shape
    # 1/p amplifies the hard group's success far beyond the easy group's.
    assert advantages[0, 1].max() > advantages[0, 0].max()
    assert diagnostics[2] > 0.0, "pass rates must differ across groups"


def test_uninformative_groups_are_dropped():
    """All-fail and all-pass groups carry no preference; the paper drops those prompts."""
    args = make_args(group_size=4, maxrl_estimator="maxrl")
    returns = torch.tensor([[[-1.0] * 4, [3.0] * 4, [3.0, 3.0, -1.0, -1.0]]])
    advantages, diagnostics = branch.group_advantages(returns, torch.tensor(0.0), args)
    torch.testing.assert_close(advantages[0, 0], torch.zeros(4))
    torch.testing.assert_close(advantages[0, 1], torch.zeros(4))
    assert advantages[0, 2].abs().sum() > 0
    assert diagnostics[3].item() == pytest.approx(1 / 3)   # groups_all_failed
    assert diagnostics[4].item() == pytest.approx(1 / 3)   # groups_all_passed
    assert diagnostics[5].item() == pytest.approx(1 / 3)   # informative


@pytest.mark.parametrize("estimator", branch.ESTIMATORS)
def test_every_estimator_stays_finite_at_degenerate_pass_rates(estimator):
    args = make_args(group_size=8, maxrl_estimator=estimator)
    for value in (-5.0, 0.0, 5.0):
        returns = torch.full((2, 3, 8), value)
        advantages, diagnostics = branch.group_advantages(returns, torch.tensor(0.0), args)
        assert torch.isfinite(advantages).all(), estimator
        assert torch.isfinite(diagnostics).all(), estimator


def test_outcome_broadcast_lands_on_the_right_timesteps():
    """Reproduce the main loop's reshape and check every element's provenance.

    An off-by-one in this index algebra would train each branch on another branch's
    outcome and would never show up as an error, only as a silently worse run.
    """
    args = make_args(num_envs=6, group_size=3, num_steps=4, segment_length=2)
    segments, length, groups, members = 2, 2, 2, 3
    # Reward marks its own (step, env) so the mapping is checkable.
    rewards = torch.arange(args.num_steps * args.num_envs, dtype=torch.float32).view(
        args.num_steps, args.num_envs)
    shaped = rewards.view(segments, length, groups, members)
    for s in range(segments):
        for l in range(length):
            for g in range(groups):
                for m in range(members):
                    step, env = s * length + l, g * members + m
                    assert shaped[s, l, g, m] == rewards[step, env]

    advantages = torch.arange(segments * groups * members, dtype=torch.float32).view(
        segments, groups, members)
    broadcast = advantages.unsqueeze(1).expand(-1, length, -1, -1).reshape(
        args.num_steps, args.num_envs)
    for step in range(args.num_steps):
        for env in range(args.num_envs):
            s, g, m = step // length, env // members, env % members
            assert broadcast[step, env] == advantages[s, g, m]


def test_segment_return_is_discounted_within_the_segment():
    args = make_args(num_envs=2, group_size=2, num_steps=4, segment_length=2, gamma=0.5)
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    discounts = args.gamma ** torch.arange(args.segment_length, dtype=torch.float32)
    shaped = rewards.view(2, 2, 1, 2)
    segment_returns = (shaped * discounts[None, :, None, None]).sum(dim=1)
    # env 0: 1 + 0.5*1 = 1.5 in segment 0, 0 in segment 1. env 1: 0 then 2 + 0.5*2 = 3.
    torch.testing.assert_close(segment_returns.flatten(), torch.tensor([1.5, 0.0, 0.0, 3.0]))


def test_cloning_shares_one_state_and_leaves_the_leader_alone():
    """The defining mechanic, checked on the real env: a cloned group must be identical."""
    envs = branch.make_mujoco_vector_env("HalfCheetah-v4", 4, backend="native", num_threads=1)
    try:
        obs, _ = envs.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs, _, _, _, _ = envs.step(rng.uniform(-1, 1, size=(4, 6)).astype(np.float32))
        assert not np.allclose(obs[0], obs[1]), "environments must differ before cloning"

        leader_before = envs._bases[0].data.qpos.copy()
        normalized = obs.astype(np.float32).copy()
        branch.clone_group_states(envs._bases, normalized, group_size=4)

        # The leader is the source and must be untouched: its episode stays valid.
        assert np.allclose(envs._bases[0].data.qpos, leader_before)
        for index in range(1, 4):
            assert np.allclose(envs._bases[index].data.qpos, leader_before)
            assert np.allclose(normalized[index], normalized[0])

        action = rng.uniform(-1, 1, size=(1, 6)).astype(np.float32).repeat(4, axis=0)
        stepped, rewards, _, _, _ = envs.step(action)
        for index in range(1, 4):
            assert np.allclose(stepped[0], stepped[index]), "clones must evolve identically"
            assert rewards[0] == pytest.approx(rewards[index])
    finally:
        envs.close()


def test_cloning_keeps_groups_independent():
    """Group 1 must not be contaminated by group 0's leader."""
    envs = branch.make_mujoco_vector_env("HalfCheetah-v4", 4, backend="native", num_threads=1)
    try:
        obs, _ = envs.reset(seed=3)
        rng = np.random.default_rng(1)
        for _ in range(15):
            obs, _, _, _, _ = envs.step(rng.uniform(-1, 1, size=(4, 6)).astype(np.float32))
        normalized = obs.astype(np.float32).copy()
        branch.clone_group_states(envs._bases, normalized, group_size=2)
        assert np.allclose(normalized[1], normalized[0])
        assert np.allclose(normalized[3], normalized[2])
        assert not np.allclose(normalized[0], normalized[2])
    finally:
        envs.close()


def test_segment_and_rollout_must_tile_the_episode():
    """A segment straddling an autoreset would mix two episodes into one outcome."""
    with pytest.raises(ValueError, match="segments"):
        branch.validate_args(make_args(num_steps=250, segment_length=99))
    # Tiles the rollout but not the 1000-step episode, so segment 8 would straddle a reset.
    with pytest.raises(ValueError, match="horizon"):
        branch.validate_args(make_args(num_steps=256, segment_length=128))
    with pytest.raises(ValueError, match="whole groups"):
        branch.validate_args(make_args(num_envs=10, group_size=4))
    with pytest.raises(ValueError, match="continuations"):
        branch.validate_args(make_args(group_size=1))
    with pytest.raises(ValueError, match="native"):
        branch.validate_args(make_args(env_backend="sync"))
    good = branch.validate_args(make_args())
    assert good.num_groups == 8 and good.segments_per_rollout == 2
