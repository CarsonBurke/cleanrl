"""Landscape pipeline against ground truth.

1. An analytic linear-quadratic task: observations do not depend on actions and reward is -|a - M s|^2, so the
   expected return of a linear policy a = W s + scale sigma eps is known in closed form along any direction.
   The pipeline's paired slices must recover the true noise cost, the true peak and curvature of a slice, and
   the critic ascent direction must be the true policy gradient when the critic is exact.
2. The REPPO adapter against the trainer that saved the snapshot: same moments, critic, entropy term and
   observation normalization (environment 0's statistics, as the trainer's own evaluate()), and on the real
   HalfCheetah-v4 the adapter's deterministic returns equal a plain gymnasium rollout of the trainer's actor.
"""

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared import landscape as ls

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "cleanrl/reppo/ppo_continuous_action_reppo_exacttarget_v61.py"
OBS, ACT, HORIZON, SIGMA = 4, 2, 6, 0.3
ACT_HC = 6  # HalfCheetah-v4 action dimension
TARGET = np.random.default_rng(11).normal(size=(ACT, OBS))  # M


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuadraticEnv:
    """obs_t = features(seed, t), independent of actions; reward -|a - M obs_t|^2; fixed horizon."""

    def __init__(self, count):
        self.count = count

    def features(self):
        return np.stack([np.random.default_rng(int(s) * 1000 + self.t).normal(size=OBS) for s in self.seeds])

    def reset(self, seed):
        self.seeds, self.t = np.asarray(seed), 0
        self.obs = self.features()
        return self.obs.astype(np.float32), {}

    def step(self, actions):
        reward = -np.square(actions - self.obs @ TARGET.T).sum(1)
        self.t += 1
        self.obs = self.features()
        done = np.full(self.count, self.t >= HORIZON)
        return self.obs.astype(np.float32), reward, np.zeros(self.count, bool), done, {}

    def close(self):
        pass


def quadratic_adapter(weight):
    module = nn.Linear(OBS, ACT, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
    return ls.Adapter(module=module, obs_mean=torch.zeros(OBS), obs_inv_std=torch.ones(OBS), obs_clip=1e9,
                      action_dim=ACT, act=lambda out, noise, scale: out + scale * SIGMA * noise,
                      critic=lambda obs, actions: -(actions - obs @ torch.as_tensor(TARGET, dtype=torch.float32).T
                                                    ).square().sum(-1))


def episode_states(episodes, seed):
    return [np.stack([np.random.default_rng((seed + e) * 1000 + t).normal(size=OBS) for t in range(HORIZON)])
            for e in range(episodes)]


def expected_return(weight, states, scale):
    """E_eps of the return, averaged over the given episodes' states."""
    return float(np.mean([-np.square(s @ (weight - TARGET).T).sum() - HORIZON * ACT * (scale * SIGMA) ** 2
                          for s in states]))


@pytest.fixture(scope="module")
def setup():
    weight = np.random.default_rng(3).normal(size=(ACT, OBS))
    adapter = quadratic_adapter(weight)
    return weight, adapter, ls.base_parameters(adapter.module)


def test_noise_slice_measures_the_true_noise_cost(setup):
    weight, adapter, base = setup
    points = {k: v[None].expand(2, *v.shape) for k, v in base.items()}
    episodes = 256
    run = ls.episode_returns(QuadraticEnv, adapter, points, episodes, [0.0, 1.0], 7)
    # Scale 0 is deterministic: exactly the closed form on the same episodes.
    states = episode_states(episodes, 7)
    np.testing.assert_allclose(run.returns[0].mean(), expected_return(weight, states, 0.0), rtol=1e-4)
    summary = ls.paired_summary(run.returns, 0)
    true_cost = -HORIZON * ACT * SIGMA ** 2
    assert abs(summary["diff"][1] - true_cost) < 4 * summary["diff_se"][1]
    # Pairing must help: the paired se is far below the unpaired one.
    unpaired = np.sqrt(summary["se"][0] ** 2 + summary["se"][1] ** 2)
    assert summary["diff_se"][1] < 0.3 * unpaired


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_update_slice_recovers_true_peak_and_curvature(setup, scale):
    weight, adapter, base = setup
    direction = {"weight": torch.as_tensor(TARGET - weight, dtype=torch.float32)}  # the optimum sits at c = 1
    coefficients = np.linspace(-0.5, 2.0, 11)
    center = 2  # c = 0
    points = ls.grid_points(base, [direction], torch.as_tensor(coefficients[:, None], dtype=torch.float32))
    episodes = 64
    run = ls.episode_returns(QuadraticEnv, adapter, points, episodes, scale, 5)
    summary = ls.paired_summary(run.returns, center)
    fit = ls.paired_quadratic(coefficients, summary["diff"], summary["diff_se"], center)
    states = episode_states(episodes, 5)
    true = ls.quadratic_peak(coefficients, np.array(
        [expected_return(weight + c * (TARGET - weight), states, scale) for c in coefficients]))
    assert true["peak"] == pytest.approx(1.0)
    tolerance = 1e-3 if scale == 0 else 4 * fit["curvature_se"]
    assert abs(fit["curvature"] - true["curvature"]) <= tolerance * (1 if scale else abs(true["curvature"]))
    assert fit["peak"] == pytest.approx(1.0, abs=1e-3 if scale == 0 else 0.1)


def test_critic_ascent_is_the_true_policy_gradient_under_an_exact_critic(setup):
    weight, adapter, base = setup
    states = torch.as_tensor(np.concatenate(episode_states(128, 9)), dtype=torch.float32)
    direction = ls.critic_ascent_direction(adapter, base, states, 64, 1.0, 0)["weight"].numpy()
    # d/dW E[-|W s + sigma eps - M s|^2] = -2 (W - M) E[s s^T]; the noise term has zero mean.
    s = states.numpy()
    truth = -2.0 * (weight - TARGET) @ (s.T @ s / len(s))
    cosine = (direction * truth).sum() / (np.linalg.norm(direction) * np.linalg.norm(truth))
    assert cosine > 0.99
    assert np.linalg.norm(direction) == pytest.approx(np.linalg.norm(truth), rel=0.05)


# ---------------------------------------------------------------- REPPO adapter vs its trainer

@pytest.fixture(scope="module")
def reppo():
    trainer = load(TRAINER, "reppo_trainer_under_test")
    landscape = load(REPO / "scripts/landscape.py", "landscape_script_under_test")
    torch.manual_seed(0)
    args = trainer.Args()
    env = gym.make("HalfCheetah-v4")
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    actor = trainer.Actor(obs_dim, act_dim, args)
    critic = trainer.Critic(obs_dim, act_dim, args, torch.device("cpu"))
    rng = np.random.default_rng(0)
    # Per-environment statistics that differ, as VectorObsNorm's do.
    means = rng.normal(scale=0.3, size=(3, obs_dim))
    variances = rng.uniform(0.5, 2.0, size=(3, obs_dim))
    snapshot = {"args": vars(args), "actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
                "obs_means": means, "obs_variances": variances, "obs_epsilon": 1e-8, "temperature": [0.05] * act_dim}
    adapter, env_id = landscape.reppo_adapter(snapshot, str(TRAINER), torch.device("cpu"))
    return trainer, landscape, args, actor, critic, snapshot, adapter, env_id


def trainer_normalize(snapshot, args, raw):
    """The trainer's evaluate(): environment 0's statistics, clipped."""
    mean = snapshot["obs_means"][0]
    inv_std = 1.0 / np.sqrt(snapshot["obs_variances"][0] + snapshot["obs_epsilon"])
    return np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)


def test_reppo_adapter_matches_trainer_networks(reppo):
    trainer, _, args, actor, critic, snapshot, adapter, env_id = reppo
    assert env_id == args.env_id
    raw = np.random.default_rng(1).normal(scale=3.0, size=(64, adapter.obs_mean.numel())).astype(np.float32)
    obs = torch.as_tensor(trainer_normalize(snapshot, args, raw))
    torch.testing.assert_close(adapter.normalize(torch.as_tensor(raw)), obs, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        mean, std = trainer.policy_moments(actor.net, obs)
        outputs = adapter.module(obs)
        torch.testing.assert_close(outputs, torch.cat((mean, std), -1))
        noise = torch.randn(mean.shape)
        actions = adapter.act(outputs, noise, 1.0)
        torch.testing.assert_close(actions, torch.tanh(mean + std * noise))
        torch.testing.assert_close(adapter.critic(obs, actions), critic.value(obs, actions))
        pre = mean + std * noise
        logprob = -0.5 * noise.square() - std.log() - trainer.LOG_SQRT_2PI - trainer.tanh_log_det(pre)
        expected = -(torch.tensor(snapshot["temperature"]) * logprob).sum(-1)
        torch.testing.assert_close(adapter.regularizer(outputs, noise, 1.0), expected, rtol=1e-5, atol=1e-5)


def test_reppo_adapter_acts_like_the_trainer_along_a_real_trajectory(reppo):
    """Along a real HalfCheetah-v4 episode of the trainer's evaluate() policy, the landscape's batched
    deterministic action agrees to float32 rounding. Returns are not compared: HalfCheetah amplifies 1e-7
    action differences into ~10% return differences over 1000 steps, which is also why paired landscape
    differences keep a chaotic floor even for infinitesimal parameter steps."""
    trainer, _, args, actor, _, snapshot, adapter, env_id = reppo
    points = {k: v[None] for k, v in ls.base_parameters(adapter.module).items()}
    env = gym.make(env_id)
    raw, _ = env.reset(seed=123)
    worst, done = 0.0, False
    while not done:
        with torch.no_grad():
            mean, _ = trainer.policy_moments(actor.net, torch.as_tensor(trainer_normalize(snapshot, args, raw))[None])
            obs = adapter.normalize(torch.as_tensor(raw, dtype=torch.float32))[None, None]
            ours = adapter.act(ls.batched_outputs(adapter.module, points, obs), torch.zeros(1, 1, ACT_HC), 0.0)
        worst = max(worst, (ours[0, 0] - torch.tanh(mean[0])).abs().max().item())
        raw, _, terminated, truncated, _ = env.step(torch.tanh(mean[0]).numpy())
        done = terminated or truncated
    env.close()
    assert worst < 1e-5
