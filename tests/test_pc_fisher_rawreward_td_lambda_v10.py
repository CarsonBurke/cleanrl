from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import cleanrl.pc.ppo_continuous_action_pc_fisher_clamped_td_lambda_v3 as v3
import cleanrl.pc.ppo_continuous_action_pc_fisher_rawreward_td_lambda_v10 as v10


class RawRewardEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.ones(3, dtype=np.float32), 123.5, False, False, {}


class TransformObservation(gym.ObservationWrapper):
    """Compatibility shim for the v3 two-argument wrapper call under newer Gymnasium."""

    def __init__(self, env, transform):
        super().__init__(env)
        self.transform = transform

    def observation(self, observation):
        return self.transform(observation)


def wrapper_names(env):
    names = []
    while isinstance(env, gym.Wrapper):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_v10_uses_raw_rewards_but_preserves_observation_wrappers(monkeypatch):
    monkeypatch.setattr(v10.gym, "make", lambda *args, **kwargs: RawRewardEnv())
    monkeypatch.setattr(v10.gym.wrappers, "TransformObservation", TransformObservation)
    env = v10.make_env("unused", 0, False, "test", gamma=0.99)()
    names = wrapper_names(env)
    assert "NormalizeReward" not in names
    assert "TransformReward" not in names
    assert "NormalizeObservation" in names
    assert "TransformObservation" in names
    assert "ClipAction" in names
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert reward == 123.5
    env.close()


def test_v10_defaults_only_change_name_and_weight_decay():
    old = asdict(v3.Args())
    new = asdict(v10.Args())
    assert old.pop("exp_name").endswith("pc_fisher_clamped_td_lambda_v3")
    assert new.pop("exp_name").endswith("pc_fisher_rawreward_td_lambda_v10")
    assert old.pop("weight_decay") == 0.01
    assert new.pop("weight_decay") == 0.0
    assert new == old


def test_v10_network_initialization_and_rng_match_v3_exactly():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float32
        ),
        single_action_space=gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        ),
    )
    common = dict(hidden_size=8, pc_num_hidden_layers=3)
    torch.manual_seed(91)
    old_agent = v3.Agent(envs, v3.Args(**common))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    new_agent = v10.Agent(envs, v10.Args(**common))
    new_rng = torch.get_rng_state().clone()

    torch.testing.assert_close(new_rng, old_rng, rtol=0, atol=0)
    assert new_agent.state_dict().keys() == old_agent.state_dict().keys()
    for name, old_value in old_agent.state_dict().items():
        torch.testing.assert_close(
            new_agent.state_dict()[name], old_value, rtol=0, atol=0
        )


def test_default_zero_decay_does_not_move_parameters_without_a_direction():
    edge = v10.LocalPredictor(2, 1, "identity", std=0.1)
    optimizer = v10.AugmentedAdamW(
        [edge],
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=v10.Args().weight_decay,
    )
    before_weight = edge.weight.clone()
    before_bias = edge.bias.clone()
    optimizer.step([torch.zeros(1, 3)], learning_rate=0.01)
    torch.testing.assert_close(edge.weight, before_weight, rtol=0, atol=0)
    torch.testing.assert_close(edge.bias, before_bias, rtol=0, atol=0)
