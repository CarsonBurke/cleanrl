import inspect
from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import cleanrl.ppo_continuous_action_pc_fisher_clamped_td_lambda_v3 as v3
import cleanrl.ppo_continuous_action_pc_fisher_dreamer_retnorm_twohot_v13 as v13
import cleanrl.ppo_continuous_action_pc_fisher_rewardnorm_twohot_v14 as v14


class ConstantRewardEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.ones(3, dtype=np.float32), 123.5, False, False, {}


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env, transform):
        super().__init__(env)
        self.transform = transform

    def observation(self, observation):
        return self.transform(observation)


class TransformReward(gym.RewardWrapper):
    def __init__(self, env, transform):
        super().__init__(env)
        self.transform = transform

    def reward(self, reward):
        return self.transform(reward)


def wrapper_names(env):
    names = []
    while True:
        names.append(type(env).__name__)
        if not hasattr(env, "env"):
            return names
        env = env.env


def make_agent(module, seed=1):
    torch.manual_seed(seed)
    args = module.Args(
        hidden_size=8,
        pc_num_hidden_layers=2,
        pc_inference_steps=3,
        compile=False,
    )
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )
    return module.Agent(envs, args), args


def test_defaults_are_v3_no_decay_with_only_obsolete_scalar_critic_clip_removed():
    old = asdict(v3.Args())
    new = asdict(v14.Args())
    assert old.pop("exp_name").endswith("pc_fisher_clamped_td_lambda_v3")
    assert new.pop("exp_name").endswith("pc_fisher_rewardnorm_twohot_v14")
    assert old.pop("weight_decay") == 0.01
    assert new.pop("weight_decay") == 0.0
    assert old.pop("critic_td_clip") == 10.0
    assert "critic_td_clip" not in new
    assert old == new


def test_environment_wrapper_stack_exactly_matches_v3_reward_normalization(monkeypatch):
    assert inspect.getsource(v14.make_env) == inspect.getsource(v3.make_env)
    monkeypatch.setattr(v14.gym, "make", lambda *args, **kwargs: ConstantRewardEnv())
    monkeypatch.setattr(v14.gym.wrappers, "TransformObservation", TransformObservation)
    monkeypatch.setattr(v14.gym.wrappers, "TransformReward", TransformReward)
    old = v3.make_env("unused", 0, False, "test", gamma=0.99)()
    new = v14.make_env("unused", 0, False, "test", gamma=0.99)()
    try:
        assert wrapper_names(new) == wrapper_names(old)
        names = wrapper_names(new)
        assert "NormalizeReward" in names
        assert "TransformReward" in names
        old_obs, _ = old.reset(seed=4)
        new_obs, _ = new.reset(seed=4)
        np.testing.assert_array_equal(new_obs, old_obs)
        action = np.array([0.25, -0.5], dtype=np.float32)
        old_transition = old.step(action)
        new_transition = new.step(action)
        for actual, expected in zip(new_transition[:4], old_transition[:4]):
            np.testing.assert_array_equal(actual, expected)
    finally:
        old.close()
        new.close()


def test_actor_parameters_and_pc_directions_are_exactly_v3():
    old, old_args = make_agent(v3, seed=23)
    new, new_args = make_agent(v14, seed=23)
    for name, expected in old.actor_pc.state_dict().items():
        torch.testing.assert_close(new.actor_pc.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in old.actor_output.state_dict().items():
        torch.testing.assert_close(new.actor_output.state_dict()[name], expected, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(51)
    observations = torch.randn(6, 4, generator=generator)
    action_z = torch.rand(6, 2, generator=generator).clamp(0.01, 0.99)
    old_result = old.settle_actor(
        observations, action_z, old_args, collect_diagnostics=False
    )
    new_result = new.settle_actor(
        observations, action_z, new_args, collect_diagnostics=False
    )
    for old_group, new_group in zip(old_result[:3], new_result[:3]):
        if isinstance(old_group, list):
            for expected, actual in zip(old_group, new_group):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(new_group, old_group, rtol=0, atol=0)


def test_td_rms_is_behavior_identical_to_v3_and_terminal_target_is_robust():
    old = v3.RunningTDRMS(torch.device("cpu"), decay=0.9, minimum=0.1)
    new = v14.RunningTDRMS(torch.device("cpu"), decay=0.9, minimum=0.1)
    for delta in (
        torch.tensor([1.0, -2.0, 3.0]),
        torch.tensor([float("nan"), 4.0, -1.0]),
        torch.tensor([0.5, 0.25, -0.75]),
    ):
        torch.testing.assert_close(new.normalize(delta, 10.0), old.normalize(delta, 10.0))
        torch.testing.assert_close(new.mean_square, old.mean_square)
        torch.testing.assert_close(new.initialized, old.initialized)

    robust = v14.RunningTDRMS(torch.device("cpu"), decay=0.999, minimum=0.1)
    target, error, actor_delta = v14.compute_td_modulations(
        reward=torch.tensor([2.0, 1.0]),
        terminated=torch.tensor([True, False]),
        next_value=torch.tensor([float("nan"), 3.0]),
        value=torch.tensor([0.5, 0.25]),
        gamma=0.9,
        td_rms=robust,
        actor_clip=10.0,
    )
    torch.testing.assert_close(target, torch.tensor([2.0, 3.7]))
    torch.testing.assert_close(error, target - torch.tensor([0.5, 0.25]))
    assert torch.isfinite(actor_delta).all()


def test_twohot_primitives_are_bit_exact_v13_machinery():
    bins = v14.dreamer_twohot_bins()
    torch.testing.assert_close(bins, v13.dreamer_twohot_bins(), rtol=0, atol=0)
    assert torch.equal(bins, -bins.flip(0))
    assert bins[127].item() == 0.0
    logits = torch.randn(7, 255)
    values = torch.tensor([-10_000.0, -1.0, 0.0, 0.2, 4.0, 100.0, 20_000.0])
    torch.testing.assert_close(
        v14.twohot_encode(values, bins),
        v13.twohot_encode(values, bins),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v14.twohot_decode(logits, bins),
        v13.twohot_decode(logits, bins),
        rtol=0,
        atol=0,
    )
    for actual, expected in zip(
        v14.twohot_ce_force(logits, values, bins),
        v13.twohot_ce_force(logits, values, bins),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_categorical_critic_initialization_and_pc_directions_are_exactly_v13():
    old, old_args = make_agent(v13, seed=37)
    new, new_args = make_agent(v14, seed=37)
    for name, expected in old.critic_pc.state_dict().items():
        torch.testing.assert_close(new.critic_pc.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in old.critic_output.state_dict().items():
        torch.testing.assert_close(new.critic_output.state_dict()[name], expected, rtol=0, atol=0)
    assert torch.count_nonzero(new.critic_output.weight) == 0
    assert torch.count_nonzero(new.critic_output.bias) == 0

    observations = torch.randn(6, 4)
    targets = torch.tensor([-100.0, -2.0, 0.0, 1.0, 8.0, 200.0])
    old_result = old.settle_critic(
        observations, targets, old_args, collect_diagnostics=False
    )
    new_result = new.settle_critic(
        observations, targets, new_args, collect_diagnostics=False
    )
    for old_group, new_group in zip(old_result[:3], new_result[:3]):
        if isinstance(old_group, list):
            for expected, actual in zip(old_group, new_group):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(new_group, old_group, rtol=0, atol=0)


def test_v14_has_no_retnorm_or_categorical_critic_trace_path():
    source = inspect.getsource(v14)
    assert "RunningReturnRange" not in source
    assert "retnorm" not in source.lower()
    assert "critic_trace" not in source
    assert "critic_delta" not in source
    assert "immediate_critic_directions" in source
