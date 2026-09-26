"""Policy-coordinate regressions; these checks do not run a training loop."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_beta import Agent, make_env


class ActionRecorder(gym.Env):
    def __init__(self, action_space):
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.last_action = action.copy()
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_native_actions_map_to_physical_bounds(monkeypatch, shape):
    recorder = ActionRecorder(gym.spaces.Box(-3.0, 7.0, shape, dtype=np.float32))
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: recorder)
    env = make_env("record-actions", 0, False, "test", 0.99)()
    try:
        env.reset(seed=1)
        assert np.all(env.action_space.low == 0)
        assert np.all(env.action_space.high == 1)
        for value in (0.0, 0.25, 1.0):
            action = np.full(shape, value, dtype=np.float32)
            env.step(action)
            np.testing.assert_allclose(recorder.last_action, -3.0 + 10.0 * value)
            np.testing.assert_array_equal(action, np.full(shape, value, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize(
    "space",
    [
        gym.spaces.Discrete(2),
        gym.spaces.Box(0, 1, (2,), dtype=np.int32),
        gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        gym.spaces.Box(1.0, 1.0, (2,), dtype=np.float32),
        gym.spaces.Box(-np.finfo(np.float32).max, np.finfo(np.float32).max, (2,), dtype=np.float32),
        gym.spaces.Box(0.0, 1.0, (0,), dtype=np.float32),
    ],
)
def test_invalid_physical_action_spaces_are_rejected(monkeypatch, space):
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: ActionRecorder(space))
    with pytest.raises(ValueError, match="Beta policies require"):
        make_env("invalid-actions", 0, False, "test", 0.99)()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA policy regression")
@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_native_sample_replay_gradients_and_checkpoint(tmp_path, shape):
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32),
    )
    with torch.device("cuda"):
        agent = Agent(spaces)
        restored = Agent(spaces)
    observations = torch.randn(32, 3, device="cuda")
    with torch.no_grad():
        action, old_logprob, _, _ = agent.get_action_and_value(observations)
    assert action.shape == (32,) + shape
    assert torch.all((action > 0) & (action < 1))
    replay, logprob, entropy, value = agent.get_action_and_value(observations, action)
    assert replay is action
    torch.testing.assert_close(logprob, old_logprob, rtol=0, atol=0)
    loss = -(logprob * torch.linspace(-1, 1, 32, device="cuda")).mean() + value.square().mean() - entropy.mean()
    loss.backward()
    for parameter in agent.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    checkpoint = tmp_path / "beta.cleanrl_model"
    torch.save(agent.state_dict(), checkpoint)
    restored.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    with torch.no_grad():
        _, restored_logprob, restored_entropy, restored_value = restored.get_action_and_value(observations, action)
    torch.testing.assert_close(restored_logprob, logprob)
    torch.testing.assert_close(restored_entropy, entropy)
    torch.testing.assert_close(restored_value, value)
