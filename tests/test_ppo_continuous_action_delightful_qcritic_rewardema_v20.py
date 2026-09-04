import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "delightful" / "qcritic" / "ppo_continuous_action_delightful_qcritic_rewardema_v20.py"
SPEC = importlib.util.spec_from_file_location("delightful_qcritic_rewardema_v20", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reward_normalizer_uses_bias_corrected_immediate_reward_rms():
    normalizer = MODULE.EMARewardNormalizer(decay=0.5, epsilon=1e-8)
    first_rms = normalizer.update(np.array([3.0, 4.0], dtype=np.float32))
    assert np.isclose(first_rms, np.sqrt(12.5) + 1e-8)

    second_rms = normalizer.update(np.array([0.0, 2.0], dtype=np.float32))
    expected_m2 = (0.5 * 6.25 + 0.5 * 2.0) / (1.0 - 0.5**2)
    assert np.isclose(second_rms, np.sqrt(expected_m2) + 1e-8)


def test_reward_normalizer_preserves_signed_gym_reward_order():
    rewards = np.array([-2.0, -0.5, 0.0, 3.0], dtype=np.float32)
    normalizer = MODULE.EMARewardNormalizer()
    reward_rms = normalizer.update(rewards)
    normalized = rewards / reward_rms
    assert np.all(np.diff(normalized) > 0)
    assert normalized[0] < 0 < normalized[-1]


def test_delightful_gate_matches_algorithm_two_and_detaches_inputs():
    advantages = torch.tensor([2.0, -2.0], requires_grad=True)
    logprobs = torch.tensor([-3.0, -3.0], requires_grad=True)
    gate, surprisal, delight = MODULE.delightful_gate(advantages, logprobs)
    torch.testing.assert_close(surprisal, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(delight, torch.tensor([6.0, -6.0]))
    torch.testing.assert_close(gate, torch.sigmoid(torch.tensor([6.0, -6.0])))
    assert not gate.requires_grad


def test_delightful_gate_clips_signed_continuous_density_surprisal():
    _, surprisal, _ = MODULE.delightful_gate(
        torch.ones(3),
        torch.tensor([-100.0, 100.0, -1.0]),
        surprisal_clip=10.0,
    )
    torch.testing.assert_close(surprisal, torch.tensor([10.0, -10.0, 1.0]))


def test_replay_wraparound_and_sample_shapes():
    replay = MODULE.ReplayBuffer(5, (2,), (1,), seed=3)
    for offset in (0, 3):
        observations = np.arange(offset, offset + 6, dtype=np.float32).reshape(3, 2)
        replay.add(
            observations,
            np.ones((3, 1), dtype=np.float32) * offset,
            np.arange(offset, offset + 3, dtype=np.float32),
            observations + 1,
            np.ones(3, dtype=np.float32) * 0.99,
        )
    assert replay.size == 5
    assert replay.position == 1
    batch = replay.sample(4, torch.device("cpu"))
    assert [tuple(tensor.shape) for tensor in batch] == [(4, 2), (4, 1), (4,), (4, 2), (4,)]
    retained_observations = {tuple(row) for row in replay.observations[: replay.size]}
    assert retained_observations == {(2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (7.0, 8.0)}


def test_replay_warmup_must_be_reachable_and_cover_a_batch():
    MODULE.validate_replay_config(replay_capacity=100, critic_batch_size=32, learning_starts=64)
    with pytest.raises(ValueError, match="at least critic_batch_size"):
        MODULE.validate_replay_config(replay_capacity=100, critic_batch_size=32, learning_starts=16)
    with pytest.raises(ValueError, match="cannot exceed replay_capacity"):
        MODULE.validate_replay_config(replay_capacity=100, critic_batch_size=32, learning_starts=101)


def test_oversized_replay_add_retains_exactly_the_newest_transitions():
    replay = MODULE.ReplayBuffer(3, (1,), (1,), seed=1)
    values = np.arange(5, dtype=np.float32).reshape(5, 1)
    replay.add(values, values, values[:, 0], values, np.ones(5, dtype=np.float32))
    assert replay.size == 3
    assert replay.position == 2
    assert set(replay.observations[:, 0]) == {2.0, 3.0, 4.0}


def test_qcritic_is_action_conditioned():
    critic = MODULE.QCritic(obs_dim=3, action_dim=2)
    first_layer = critic.network[0]
    assert first_layer.in_features == 5
    observations = torch.zeros(4, 3)
    actions = torch.zeros(4, 2)
    assert critic(observations, actions).shape == (4,)


def test_actor_has_state_dependent_diagonal_scale_and_bounded_actions():
    actor = MODULE.Actor(
        obs_dim=3,
        action_dim=2,
        action_low=np.array([-1.0, -2.0], dtype=np.float32),
        action_high=np.array([1.0, 2.0], dtype=np.float32),
    )
    observations = torch.randn(7, 3)
    mean, logstd = actor.parameters_for(observations)
    actions, raw_actions = actor.sample(observations)
    assert mean.shape == logstd.shape == raw_actions.shape == actions.shape == (7, 2)
    assert torch.all(actions[:, 0].abs() <= 1.0)
    assert torch.all(actions[:, 1].abs() <= 2.0)
    assert actor.raw_scale.weight.shape == (2, 256)
    assert torch.all(logstd.exp() >= MODULE.MIN_POLICY_STD)
    torch.testing.assert_close(
        logstd.exp().mean(),
        torch.tensor(MODULE.INITIAL_POLICY_STD),
        atol=0.05,
        rtol=0.0,
    )


def test_hard_update_copies_online_q_exactly():
    source = MODULE.QCritic(obs_dim=2, action_dim=1)
    target = MODULE.QCritic(obs_dim=2, action_dim=1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(1.0)
    MODULE.hard_update(target, source)
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_time_limit_bootstrap_uses_final_observation():
    next_obs = np.array([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
    infos = {
        "final_observation": np.array([None, np.array([2.0, 3.0], dtype=np.float32)], dtype=object),
        "_final_observation": np.array([False, True]),
    }
    result = MODULE.bootstrap_observations(next_obs, np.array([False, True]), infos)
    np.testing.assert_allclose(result, np.array([[10.0, 11.0], [2.0, 3.0]], dtype=np.float32))


def test_one_step_target_bootstraps_truncation_but_not_termination():
    rewards = torch.tensor([0.2, 0.2])
    # First transition is a true termination; second is a time-limit
    # truncation whose final observation has already been retained.
    discounts = torch.tensor([0.0, 0.99])
    next_q = torch.tensor([7.0, 7.0])
    target = MODULE.one_step_q_target(rewards, discounts, next_q)
    torch.testing.assert_close(target, torch.tensor([0.2, 7.13]))


def test_latent_and_transformed_fixed_action_scores_have_identical_actor_gradients():
    actor = MODULE.Actor(
        obs_dim=3,
        action_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    observations = torch.randn(5, 3)
    raw_actions = torch.randn(5, 2)
    gaussian_logprob, action_logprob, _, _ = actor.logprobs(observations, raw_actions)
    parameters = tuple(actor.parameters())
    gaussian_grads = torch.autograd.grad(gaussian_logprob.sum(), parameters, retain_graph=True)
    action_grads = torch.autograd.grad(action_logprob.sum(), parameters)
    for gaussian_grad, action_grad in zip(gaussian_grads, action_grads, strict=True):
        torch.testing.assert_close(gaussian_grad, action_grad)


def test_v20_uses_current_ema_scaled_raw_rewards_and_no_retrace():
    source = SCRIPT.read_text()
    assert "td_target = one_step_q_target(" in source
    assert "b_rewards / reward_rms" in source
    assert "raw_rewards.reshape(args.batch_size) / reward_rms" in source
    assert "reward_normalizer.update(raw_reward)" in source
    assert "dmc_cheetah_run_reward" not in source
    assert "def retrace" not in source.lower()
    assert "importance" not in source.lower()
    assert "if learner_step == 0:" in source
    assert "q_taken = critic_value(b_obs, b_actions).clone()" in source
    assert "baseline_actions, _ = actor.sample(b_obs)" in source
    assert "q_baseline = critic(b_obs, baseline_actions)" in source
    assert "q_policy_baseline" not in source
    assert "actor_baseline_samples" not in source
    assert "next_actions = next_actions.clone()" in source


def test_v20_uses_256_fresh_actor_samples_and_preserves_replay_ratio():
    args = MODULE.Args()
    fresh_batch_size = args.num_envs * args.num_steps
    critic_samples = args.critic_updates_per_iteration * args.critic_batch_size
    transitions_per_target_update = args.target_update_interval / args.critic_updates_per_iteration * fresh_batch_size
    ema_updates_per_transition = args.num_steps / fresh_batch_size
    assert fresh_batch_size == 256
    assert args.critic_batch_size == 256
    assert critic_samples / fresh_batch_size == 4
    assert transitions_per_target_update == 6_400
    assert ema_updates_per_transition == 1 / 16
    assert args.learning_starts % fresh_batch_size == 0


def test_policy_scale_floor_and_latent_gaussian_score():
    raw_scale = torch.tensor([-1_000.0, 0.0, 1_000.0])
    scale = MODULE.state_dependent_logstd(raw_scale).exp()
    assert torch.all(scale >= MODULE.MIN_POLICY_STD - 1e-6)
    assert torch.all(scale <= MODULE.MAX_POLICY_STD + 1e-6)

    actor = MODULE.Actor(
        obs_dim=3,
        action_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    observations = torch.randn(5, 3)
    raw_actions = torch.randn(5, 2) * 3.0
    gaussian_logprob, action_logprob, mean, _ = actor.logprobs(observations, raw_actions)
    torch.testing.assert_close(action_logprob, gaussian_logprob)
    assert torch.all(mean.abs() <= 1.0)
    assert torch.all(actor.transform(raw_actions).abs() <= 1.0)
