"""Structural tests for the v10 fullbatch version.

Covers what the v8 functional tests cannot: split-LR defaults, unshared
backbone default, full-batch defaults, and the optimizer-group invariant the
v10 build asserts (disjoint actor/critic groups covering all params exactly
once -- required so no parameter is stepped at two learning rates).
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import pytest
import torch

ROOT = Path(__file__).parents[1]


def _load_v10():
    spec = importlib.util.spec_from_file_location(
        "nextlat_v10_fullbatch",
        ROOT / "cleanrl/iterthink/v24_d3bucket/ppo/ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_nextlat_v10_fullbatch.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def _tiny_args(module, **overrides):
    return replace(
        module.Args(),
        hidden=8,
        k_blocks=1,
        n_experts=2,
        num_bins=7,
        critic_mtp_horizon=2,
        actor_dist="beta",
        **overrides,
    )


def test_v10_fullbatch_defaults():
    module = _load_v10()
    args = module.Args()
    assert args.share_backbone is False
    assert args.num_envs == 64
    assert args.num_steps == 39
    assert args.num_minibatches == 1
    assert args.update_epochs == 1
    assert args.actor_lr == pytest.approx(3e-3)
    assert args.critic_lr == pytest.approx(1e-3)
    assert args.target_kl == pytest.approx(0.03)


def _optimizer_groups(agent):
    actor = list(agent.nextlat_actor_parameters())
    critic = list(agent.nextlat_critic_parameters()) + list(agent.critic_head.parameters())
    return actor, critic


@pytest.mark.parametrize("share_backbone", [False, True])
def test_optimizer_group_partition_matches_guard(share_backbone):
    torch.manual_seed(0)
    module = _load_v10()
    agent = module.Agent(_DummyEnvs(), _tiny_args(module, share_backbone=share_backbone))
    actor, critic = _optimizer_groups(agent)
    actor_ids = {id(p) for p in actor}
    critic_ids = {id(p) for p in critic}
    all_ids = {id(p) for p in agent.parameters()}
    if not share_backbone:
        assert not (actor_ids & critic_ids)
        assert all_ids == (actor_ids | critic_ids)
    else:
        # Shared trunk lands in both groups: the v10 build guard must fire.
        assert actor_ids & critic_ids


def test_unshared_trunks_are_distinct_objects():
    torch.manual_seed(0)
    module = _load_v10()
    agent = module.Agent(_DummyEnvs(), _tiny_args(module, share_backbone=False))
    assert agent.actor_trunk is not agent.critic_trunk
    x = torch.randn(4, 4)
    assert not torch.equal(agent.actor_trunk(x), agent.critic_trunk(x))
