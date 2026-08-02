from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import cleanrl.ppo_continuous_action_pc_fisher_dreamer_retnorm_v11 as v11
import cleanrl.ppo_continuous_action_pc_fisher_rewardnorm_dreamer_retnorm_v12 as v12


class ConstantRewardEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.ones(3, dtype=np.float32), 123.5, False, False, {}


class TransformObservation(gym.ObservationWrapper):
    """Compatibility shim for the inherited two-argument wrapper call."""

    def __init__(self, env, transform):
        super().__init__(env)
        self.transform = transform

    def observation(self, observation):
        return self.transform(observation)


class TransformReward(gym.RewardWrapper):
    """Compatibility shim for the inherited two-argument wrapper call."""

    def __init__(self, env, transform):
        super().__init__(env)
        self.transform = transform

    def reward(self, reward):
        return self.transform(reward)


def wrappers(env):
    result = []
    while isinstance(env, gym.Wrapper):
        result.append(env)
        env = env.env
    return result


def test_v12_source_diff_is_exactly_header_and_v3_reward_wrappers():
    old = Path(v11.__file__).read_text()
    new = Path(v12.__file__).read_text()
    old_body = old[old.index("import math") :]
    new_body = new[new.index("import math") :]
    anchor = (
        "        env = gym.wrappers.TransformObservation("
        "env, lambda obs: np.clip(obs, -10, 10))\n"
    )
    expected = old_body.replace(
        anchor,
        anchor
        + "        env = gym.wrappers.NormalizeReward(env, gamma=gamma)\n"
        + "        env = gym.wrappers.TransformReward("
        "env, lambda reward: np.clip(reward, -10, 10))\n",
        1,
    )
    assert new_body == expected


def test_v12_restores_normalization_then_reward_clipping(monkeypatch):
    monkeypatch.setattr(v12.gym, "make", lambda *args, **kwargs: ConstantRewardEnv())
    monkeypatch.setattr(v12.gym.wrappers, "TransformObservation", TransformObservation)
    monkeypatch.setattr(v12.gym.wrappers, "TransformReward", TransformReward)
    env = v12.make_env("unused", 0, False, "test", gamma=0.87)()
    chain = wrappers(env)
    names = [type(wrapper).__name__ for wrapper in chain]
    assert names.index("TransformReward") < names.index("NormalizeReward")
    assert names.index("NormalizeReward") < names.index("TransformObservation")
    normalize = next(
        wrapper for wrapper in chain if type(wrapper).__name__ == "NormalizeReward"
    )
    assert normalize.gamma == 0.87
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert np.isfinite(reward)
    assert -10.0 <= reward <= 10.0
    env.close()


def test_v12_defaults_match_v11_except_experiment_name():
    old = asdict(v11.Args())
    new = asdict(v12.Args())
    assert old.pop("exp_name").endswith("pc_fisher_dreamer_retnorm_v11")
    assert new.pop("exp_name").endswith(
        "pc_fisher_rewardnorm_dreamer_retnorm_v12"
    )
    assert new == old
    assert new["weight_decay"] == 0.0


def test_v12_network_initialization_and_rng_match_v11_exactly():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float32
        ),
        single_action_space=gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        ),
    )
    common = dict(hidden_size=8, pc_num_hidden_layers=3)
    torch.manual_seed(121)
    old_agent = v11.Agent(envs, v11.Args(**common))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(121)
    new_agent = v12.Agent(envs, v12.Args(**common))
    new_rng = torch.get_rng_state().clone()

    torch.testing.assert_close(new_rng, old_rng, rtol=0, atol=0)
    assert new_agent.state_dict().keys() == old_agent.state_dict().keys()
    for name, old_value in old_agent.state_dict().items():
        torch.testing.assert_close(
            new_agent.state_dict()[name], old_value, rtol=0, atol=0
        )


def test_v12_retains_dreamer_actor_scaling_and_unscaled_critic_td():
    old_norm = v11.RunningReturnRange(
        torch.device("cpu"), rate=1.0, limit=1.0, perclo=5.0, perchi=95.0
    )
    new_norm = v12.RunningReturnRange(
        torch.device("cpu"), rate=1.0, limit=1.0, perclo=5.0, perchi=95.0
    )
    arguments = dict(
        reward=torch.tensor([0.0, 200.0]),
        terminated=torch.tensor([True, True]),
        next_value=torch.zeros(2),
        value=torch.zeros(2),
        gamma=0.99,
        actor_clip=1_000.0,
        critic_clip=1_000.0,
    )
    old = v11.compute_td_modulations(retnorm=old_norm, **arguments)
    new = v12.compute_td_modulations(retnorm=new_norm, **arguments)
    for actual, expected in zip(new, old):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, td_error, actor_delta, critic_delta, _, scale = new
    torch.testing.assert_close(actor_delta, td_error / scale)
    torch.testing.assert_close(critic_delta, td_error)
    assert scale > 1.0
