import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared import landscape as ls


def make_adapter(module, action_dim=2, critic=None):
    return ls.Adapter(module=module, obs_mean=torch.zeros(3), obs_inv_std=torch.ones(3), obs_clip=10.0,
                      action_dim=action_dim, act=lambda out, noise, scale: torch.tanh(out + scale * noise),
                      subsets={"last": lambda name, value: torch.full_like(value, float(name.startswith("1.")))},
                      critic=critic)


class FakeVectorEnv:
    """reward = -|a - target(seed)|^2, episode length fixed; obs carries the reset seed."""

    def __init__(self, count, length=5):
        self.count, self.length = count, length

    def reset(self, seed):
        self.seeds = np.asarray(seed, dtype=np.float64)
        self.t = 0
        return self.obs(), {}

    def obs(self):
        return np.stack([self.seeds, np.full(self.count, self.t), np.ones(self.count)], 1).astype(np.float32)

    def step(self, actions):
        self.t += 1
        target = np.sin(self.seeds + self.t)[:, None]
        reward = -np.square(actions - target).sum(1)
        done = np.full(self.count, self.t >= self.length)
        return self.obs(), reward, np.zeros(self.count, bool), done, {}

    def close(self):
        pass


def test_filter_normalize_matches_row_norms_and_drops_vectors():
    base = {"w": torch.randn(4, 3), "b": torch.randn(4)}
    direction = ls.filter_normalize({"w": torch.randn(4, 3), "b": torch.randn(4)}, base)
    torch.testing.assert_close(direction["w"].norm(dim=1), base["w"].norm(dim=1))
    assert torch.all(direction["b"] == 0)


def test_random_direction_is_seeded_and_respects_mask():
    base = {"w": torch.randn(4, 3), "v": torch.randn(2, 2)}
    mask = {"w": torch.ones(4, 3), "v": torch.zeros(2, 2)}
    a, b = ls.random_direction(base, 3, mask), ls.random_direction(base, 3, mask)
    torch.testing.assert_close(a["w"], b["w"])
    assert torch.all(a["v"] == 0)
    assert not torch.allclose(a["w"], ls.random_direction(base, 4, mask)["w"])


def test_orthogonalize_and_grid_points():
    base = {"w": torch.randn(3, 2)}
    d1, d2 = {"w": torch.randn(3, 2)}, {"w": torch.randn(3, 2)}
    d2 = ls.orthogonalize(d2, d1)
    assert abs(float(ls.dot(d1, d2))) < 1e-5
    points = ls.grid_points(base, [d1, d2], torch.tensor([[0.0, 0.0], [1.0, -2.0]]))
    torch.testing.assert_close(points["w"][0], base["w"])
    torch.testing.assert_close(points["w"][1], base["w"] + d1["w"] - 2.0 * d2["w"])


def test_subset_mask_and_unknown_subset():
    module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    adapter = make_adapter(module)
    mask = ls.subset_mask(adapter, ls.base_parameters(module), "last")
    assert torch.all(mask["0.weight"] == 0) and torch.all(mask["1.weight"] == 1)
    assert ls.subset_mask(adapter, ls.base_parameters(module), "all") is None
    with pytest.raises(ValueError):
        ls.subset_mask(adapter, ls.base_parameters(module), "nope")


def test_batched_outputs_matches_per_point_calls():
    module = nn.Sequential(nn.Linear(3, 4), nn.RMSNorm(4), nn.SiLU(), nn.Linear(4, 2))
    base = ls.base_parameters(module)
    points = ls.grid_points(base, [ls.random_direction(base, 0)], torch.tensor([[0.0], [0.5]]))
    obs = torch.randn(2, 5, 3)
    out = ls.batched_outputs(module, points, obs)
    for p in range(2):
        expected = torch.func.functional_call(module, ls.take(points, p), (obs[p],))
        torch.testing.assert_close(out[p], expected)


def test_episode_returns_common_random_numbers():
    module = nn.Sequential(nn.Linear(3, 2))
    adapter = make_adapter(module)
    base = ls.base_parameters(module)
    points = {k: v[None].expand(3, *v.shape) for k, v in base.items()}
    # Identical points must give identical returns across chunks: resets and noise are shared per step.
    run = ls.episode_returns(lambda n: FakeVectorEnv(n), adapter, points, 4, 1.0, 7, max_envs=4,
                             record_point=1)
    np.testing.assert_allclose(run.returns, np.broadcast_to(run.returns[0], run.returns.shape))
    assert run.lengths.min() == 5 and run.states.shape[1] == 3
    # Per-point scales: scale 0 removes the noise, so the point differs from the noisy ones.
    mixed = ls.episode_returns(lambda n: FakeVectorEnv(n), adapter, points, 4, [1.0, 0.0, 1.0], 7)
    np.testing.assert_allclose(mixed.returns[0], mixed.returns[2])
    assert not np.allclose(mixed.returns[0], mixed.returns[1])


def test_critic_ascent_direction_increases_surrogate():
    module = nn.Sequential(nn.Linear(3, 2))
    adapter = make_adapter(module, critic=lambda obs, actions: -(actions - 0.3).square().sum(-1))
    base = ls.base_parameters(module)
    states = torch.randn(64, 3)
    direction = ls.critic_ascent_direction(adapter, base, states, 4, 1.0, 0)
    points = ls.grid_points(base, [direction], torch.tensor([[0.0], [1e-3]]))
    values = ls.critic_objective(adapter, points, states, 4, 1.0, 0)
    assert values[1] > values[0]


def test_paired_summary_and_quadratic_peak():
    returns = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.5]])
    summary = ls.paired_summary(returns, 0)
    np.testing.assert_allclose(summary["diff"], [0.0, 1.1666666], rtol=1e-6)
    c = np.linspace(-1, 2, 7)
    fit = ls.quadratic_peak(c, 3.0 + 2.0 * c - 0.5 * 4.0 * c ** 2)
    assert fit["curvature"] == pytest.approx(-4.0) and fit["peak"] == pytest.approx(0.5)
    assert np.isnan(ls.quadratic_peak(c, c ** 2)["peak"])


def test_paired_quadratic_survives_zero_se_center_and_large_curvature():
    c = np.linspace(-0.02, 0.02, 11)
    diff = 3e3 * c - 0.5 * 4e5 * c ** 2
    se = np.full(11, 500.0)
    se[5] = 0.0
    fit = ls.paired_quadratic(c, diff, se, 5)
    assert fit["curvature"] == pytest.approx(-4e5, rel=1e-6) and fit["peak"] == pytest.approx(3e3 / 4e5)
    shifted = ls.paired_quadratic(c + 1.0, diff, se, 5)
    assert shifted["peak"] == pytest.approx(1.0 + 3e3 / 4e5)


class StaggeredEnv(FakeVectorEnv):
    """Ends each env on a seed-dependent step and autoresets it (gym 0.29 same-step semantics)."""

    def step(self, actions):
        obs, reward, terminated, _, info = super().step(actions)
        terminated = self.t == 2 + (self.seeds.astype(int) % 3)
        return obs, reward, terminated, np.full(self.count, self.t >= 6), info


def test_staggered_termination_counts_only_first_episode():
    module = nn.Sequential(nn.Linear(3, 2))
    adapter = make_adapter(module)
    points = {k: v[None].expand(2, *v.shape) for k, v in ls.base_parameters(module).items()}
    run = ls.episode_returns(lambda n: StaggeredEnv(n), adapter, points, 3, 0.0, 0)
    np.testing.assert_array_equal(run.lengths, [[2, 3, 4], [2, 3, 4]])
    np.testing.assert_allclose(run.returns[0], run.returns[1])
