from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import torch

import cleanrl.ppo_continuous_action_pc_fisher_dreamer_retnorm_v11 as v11
import cleanrl.ppo_continuous_action_pc_fisher_rawreward_td_lambda_v10 as v10


def make_retnorm(**overrides):
    values = dict(rate=0.01, limit=1.0, perclo=5.0, perchi=95.0)
    values.update(overrides)
    return v11.RunningReturnRange(torch.device("cpu"), **values)


def test_percentile_ema_matches_dreamer_zero_initialization_and_current_update_use():
    norm = make_retnorm()
    targets = torch.tensor([0.0, 200.0])
    offset, scale = norm.update(targets)

    # Linear percentiles are 10 and 190. Dreamer uses a zero-initialized,
    # non-debiased EMA, so the first lo/hi are 0.1 and 1.9 rather than 10 and 190.
    torch.testing.assert_close(offset, torch.tensor(0.1))
    torch.testing.assert_close(norm.hi, torch.tensor(1.9))
    torch.testing.assert_close(scale, torch.tensor(1.8))

    second_targets = torch.tensor([100.0, 300.0])
    second_quantiles = torch.quantile(second_targets, torch.tensor([0.05, 0.95]))
    second_offset, second_scale = norm.update(second_targets)
    expected_lo = 0.99 * torch.tensor(0.1) + 0.01 * second_quantiles[0]
    expected_hi = 0.99 * torch.tensor(1.9) + 0.01 * second_quantiles[1]
    torch.testing.assert_close(second_offset, expected_lo)
    torch.testing.assert_close(norm.hi, expected_hi)
    torch.testing.assert_close(second_scale, expected_hi - expected_lo)

    result = v11.compute_td_modulations(
        reward=targets,
        terminated=torch.tensor([True, True]),
        next_value=torch.zeros(2),
        value=torch.zeros(2),
        gamma=0.99,
        retnorm=make_retnorm(),
        actor_clip=1_000.0,
        critic_clip=1_000.0,
    )
    _, td_error, actor_delta, critic_delta, _, used_scale = result
    torch.testing.assert_close(used_scale, torch.tensor(1.8))
    torch.testing.assert_close(actor_delta, td_error / used_scale)
    torch.testing.assert_close(critic_delta, td_error)


def test_percentiles_use_all_and_only_finite_target_rows():
    norm = make_retnorm(rate=0.2, limit=0.01, perclo=25.0, perchi=75.0)
    offset, scale = norm.update(
        torch.tensor([float("nan"), 1.0, float("inf"), 3.0, 5.0, 7.0])
    )
    finite = torch.tensor([1.0, 3.0, 5.0, 7.0])
    expected = torch.quantile(finite, torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(offset, 0.2 * expected[0])
    torch.testing.assert_close(norm.hi, 0.2 * expected[1])
    torch.testing.assert_close(scale, 0.2 * (expected[1] - expected[0]))


def test_all_nonfinite_targets_leave_persistent_stats_unchanged():
    norm = make_retnorm()
    norm.update(torch.tensor([-2.0, 4.0, 10.0]))
    before_lo = norm.lo.clone()
    before_hi = norm.hi.clone()
    offset, scale = norm.update(torch.tensor([float("nan"), float("inf")]))
    torch.testing.assert_close(norm.lo, before_lo, rtol=0, atol=0)
    torch.testing.assert_close(norm.hi, before_hi, rtol=0, atol=0)
    torch.testing.assert_close(offset, before_lo, rtol=0, atol=0)
    torch.testing.assert_close(scale, (before_hi - before_lo).clamp_min(1.0), rtol=0, atol=0)


def test_unclipped_bootstrap_targets_update_norm_actor_only_and_critic_stays_raw():
    norm = make_retnorm(rate=1.0, limit=1.0)
    reward = torch.tensor([5.0, 5.0, 5.0])
    terminated = torch.tensor([True, False, False])
    next_value = torch.tensor([100.0, 100.0, -100.0])
    value = torch.tensor([1.0, 20.0, -20.0])
    result = v11.compute_td_modulations(
        reward,
        terminated,
        next_value,
        value,
        gamma=0.99,
        retnorm=norm,
        actor_clip=10.0,
        critic_clip=10.0,
    )
    td_target, td_error, actor_delta, critic_delta, offset, scale = result

    expected_target = torch.tensor([5.0, 104.0, -94.0])
    torch.testing.assert_close(td_target, expected_target)
    expected_percentiles = torch.quantile(expected_target, torch.tensor([0.05, 0.95]))
    torch.testing.assert_close(offset, expected_percentiles[0])
    torch.testing.assert_close(scale, expected_percentiles[1] - expected_percentiles[0])
    torch.testing.assert_close(actor_delta, (td_error / scale).clamp(-10.0, 10.0))
    torch.testing.assert_close(critic_delta, td_error.clamp(-10.0, 10.0))
    assert norm.hi > 10.0  # Percentile inputs were not clipped at the TD clamp.


def test_constant_return_targets_keep_dreamer_minimum_scale_without_centering():
    norm = make_retnorm(rate=1.0, limit=1.0)
    offset, scale = norm.update(torch.full((16,), 17.0))
    torch.testing.assert_close(offset, torch.tensor(17.0))
    torch.testing.assert_close(norm.hi, torch.tensor(17.0))
    torch.testing.assert_close(scale, torch.tensor(1.0))

    _, td_error, actor_delta, _, _, _ = v11.compute_td_modulations(
        reward=torch.full((16,), 17.0),
        terminated=torch.ones(16, dtype=torch.bool),
        next_value=torch.zeros(16),
        value=torch.zeros(16),
        gamma=0.99,
        retnorm=norm,
        actor_clip=100.0,
        critic_clip=100.0,
    )
    # The lo offset is tracked but deliberately not subtracted from actor TD.
    torch.testing.assert_close(actor_delta, td_error)


def test_v11_defaults_only_replace_v10_td_rms_with_dreamer_retnorm():
    old = asdict(v10.Args())
    new = asdict(v11.Args())
    assert old.pop("exp_name").endswith("pc_fisher_rawreward_td_lambda_v10")
    assert new.pop("exp_name").endswith("pc_fisher_dreamer_retnorm_v11")
    assert old.pop("td_rms_decay") == 0.999
    assert old.pop("td_rms_min") == 0.1
    assert new.pop("retnorm_rate") == 0.01
    assert new.pop("retnorm_limit") == 1.0
    assert new.pop("retnorm_perclo") == 5.0
    assert new.pop("retnorm_perchi") == 95.0
    assert new == old


def test_v11_network_initialization_and_rng_match_v10_exactly():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -float("inf"), float("inf"), shape=(7,), dtype=float
        ),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=float),
    )
    common = dict(hidden_size=8, pc_num_hidden_layers=3)
    torch.manual_seed(91)
    old_agent = v10.Agent(envs, v10.Args(**common))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    new_agent = v11.Agent(envs, v11.Args(**common))
    new_rng = torch.get_rng_state().clone()

    torch.testing.assert_close(new_rng, old_rng, rtol=0, atol=0)
    assert new_agent.state_dict().keys() == old_agent.state_dict().keys()
    for name, old_value in old_agent.state_dict().items():
        torch.testing.assert_close(new_agent.state_dict()[name], old_value, rtol=0, atol=0)
