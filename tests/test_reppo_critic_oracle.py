import math

import numpy as np
import pytest
import torch
from scipy import stats

from scripts import reppo_critic_oracle as oracle

awr = oracle.awr


def keys(*parts):
    return awr.stream_key(12345, *parts)


def test_hash_normals_are_standard_normal():
    z = oracle.hash_normals(keys(oracle.TEST, torch.arange(100_000)), 4).double().numpy()
    assert np.isfinite(z).all()
    assert stats.kstest(z[:, 0], "norm").pvalue > 1e-3
    assert stats.kstest(z[:, 3], "norm").pvalue > 1e-3
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.02
    assert np.abs(np.corrcoef(z.T) - np.eye(4)).max() < 0.02


def test_hash_normals_common_random_numbers():
    sid, rollout = torch.arange(50).repeat_interleave(4), torch.arange(4).repeat(50)
    draw = lambda t: oracle.hash_normals(keys(oracle.BRANCH, sid, rollout, t), 6)  # noqa: E731
    # Every action of a state draws from (state, rollout, t) only: recreating the key recreates the noise.
    assert torch.equal(draw(7), draw(7))
    by_state = draw(7).view(50, 4, 6)
    assert torch.equal(by_state[:, 0], oracle.hash_normals(keys(oracle.BRANCH, torch.arange(50), 0, 7), 6))
    # Different rollouts, steps and streams are independent.
    assert abs(np.corrcoef(draw(7).flatten(), draw(8).flatten())[0, 1]) < 0.1
    assert not torch.equal(by_state[:, 0], by_state[:, 1])
    assert not torch.equal(draw(7), oracle.hash_normals(keys(oracle.TAIL, sid, rollout, 7), 6))


def test_orthonormal_directions():
    u = oracle.orthonormal_directions(keys(oracle.DIRECTIONS, torch.arange(200)), 6, 6).double()
    assert torch.allclose(u @ u.transpose(-1, -2), torch.eye(6, dtype=torch.float64).expand(200, 6, 6), atol=1e-6)
    # Haar: the first direction's coordinates are uniform on the sphere, mean zero and variance 1/6.
    first = u[:, 0].numpy()
    assert abs(first.mean()) < 0.05 and abs((first ** 2).mean() - 1 / 6) < 0.03
    assert oracle.orthonormal_directions(keys(oracle.DIRECTIONS, torch.arange(3)), 6, 4).shape == (3, 4, 6)


def test_soft_return_accounting():
    """Horizon 3, time limit at s_2: Q through the limit and Q cut at it, soft and hard."""
    gamma, horizon = 0.9, 3
    r = [1.0, 2.0, 3.0]  # r_0, r_1, r_2
    b = [None, 0.5, 0.25]  # b(s_t, a_t) at t = 1, 2; none at s_0 (its action is given)
    tail_soft, tail_hard, limit_soft, limit_hard = 10.0, 8.0, 20.0, 16.0
    to_limit = torch.tensor([2, 7])  # the second branch never reaches the limit within H
    totals = torch.zeros((4, 2), dtype=torch.float64)
    for t in range(1, horizon + 1):
        oracle.account_reward(totals, t, gamma, torch.full((2,), r[t - 1], dtype=torch.float64), to_limit == t)
        if t == 1:  # a checkpoint at s_1 bootstraps before s_1's bonus: r_0 + gamma V'(s_1)
            assert torch.allclose(totals[0], torch.full((2,), r[0], dtype=torch.float64))
        if t < horizon:
            oracle.account_bonus(totals, t, gamma, torch.full((2,), b[t], dtype=torch.float64))
    full = lambda value: torch.full((2,), value, dtype=torch.float64)  # noqa: E731
    q, weight = oracle.bootstrap(totals, to_limit, horizon, gamma, full(tail_soft), full(tail_hard), full(limit_soft),
                                 full(limit_hard))
    soft = r[0] + gamma * (r[1] + b[1]) + gamma ** 2 * (r[2] + b[2]) + gamma ** 3 * tail_soft
    hard = r[0] + gamma * r[1] + gamma ** 2 * r[2] + gamma ** 3 * tail_hard
    # Cut at s_2: r_0, r_1 and s_1's bonus, then V'(s_2), which already holds s_2's bonus.
    soft_limit = r[0] + gamma * (r[1] + b[1]) + gamma ** 2 * limit_soft
    hard_limit = r[0] + gamma * r[1] + gamma ** 2 * limit_hard
    expected = torch.tensor([[soft, soft], [hard, hard], [soft_limit, soft], [hard_limit, hard]], dtype=torch.float64)
    assert torch.allclose(q, expected)
    assert torch.allclose(weight, torch.tensor([gamma ** 2, gamma ** 3], dtype=torch.float64))


def analytic(a):
    """Q(a) = sin(w . a) + a^T B a / 2 with its gradient."""
    w = torch.tensor([0.7, -1.1, 0.3, 0.5, -0.2, 0.9], dtype=torch.float64)
    b = torch.diag(torch.tensor([1.0, -2.0, 0.5, 0.3, -0.7, 1.5], dtype=torch.float64))
    value = torch.sin(a @ w) + 0.5 * ((a @ b) * a).sum(-1)
    return value, torch.cos(a @ w)[..., None] * w + a @ b


def test_difference_pairs_recover_gradients():
    n = 64
    pre0 = 1.5 * oracle.hash_normals(keys(oracle.TEST, torch.arange(n)), 6).double()
    a0 = torch.tanh(pre0)
    u = oracle.orthonormal_directions(keys(oracle.DIRECTIONS, torch.arange(n)), 6, 6).double()
    steps = torch.tensor([0.003, 0.03], dtype=torch.float64)
    pairs, h, valid = oracle.difference_actions(a0, pre0, u, steps)
    assert pairs.shape == (n, 2, 2, 6, 2, 6) and h.shape == valid.shape == (n, 2, 2, 6)
    # Action-space pairs never step past |a| = 1 - MARGIN (a0 beyond it drops them); pre-space pairs are exact.
    inside = (pairs[:, 0].abs() <= 1 - oracle.MARGIN + 1e-12).all(-1).all(-1)
    assert inside[valid[:, 0]].all() and not valid[:, 0].all()
    assert (h[:, 0] <= steps[:, None]).all() and (h[:, 1] == steps[:, None]).all() and valid[:, 1].all()
    q, _ = analytic(pairs)
    d = oracle.directional_derivatives(q[..., 0].numpy(), q[..., 1].numpy(), h.numpy())
    _, grad = analytic(a0)
    grad_pre = grad * (1 - a0 ** 2)
    for space, g in ((0, grad), (1, grad_pre)):
        for i in range(2):
            ok = valid[:, space, i].all(-1).numpy()
            assert ok.mean() > 0.3
            # J = dims orthonormal directions: the projections determine the whole gradient.
            recovered = np.einsum("njd,nj->nd", u.numpy()[ok], d[ok, space, i])
            tolerance = 1e-5 if i == 0 else 1e-2
            np.testing.assert_allclose(recovered, g.numpy()[ok], atol=tolerance * max(1.0, np.abs(g.numpy()).max()))


def test_leave_one_out_baseline_is_unbiased():
    rng = np.random.default_rng(0)
    f = lambda a: np.sin(3 * a) + a ** 2  # noqa: E731
    value = f(rng.normal(size=2_000_000)).mean()  # V = E_{a ~ N(0, 1)} f(a)
    fixed, count, trials = 0.8, 4, 200_000
    q = np.concatenate([np.full((trials, 1), f(fixed)), f(rng.normal(size=(trials, count - 1)))], 1)
    advantage = oracle.leave_one_out(q)[:, 0]
    se = advantage.std() / math.sqrt(trials)
    assert abs(advantage.mean() - (f(fixed) - value)) < 4 * se
    naive = (q - q.mean(1, keepdims=True))[:, 0]  # the plain mean shrinks it by (K - 1) / K
    assert abs(naive.mean() - (count - 1) / count * (f(fixed) - value)) < 4 * se


def test_gradient_stats_remove_rollout_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(4000, 6))
    samples = truth[..., None] + 1.5 * rng.normal(size=(4000, 6, 16))  # SE 0.375 per direction
    result = oracle.gradient_stats(samples, 2.0 * truth, 2.0 * truth)
    assert result["pooled_corr"] < 0.97
    assert result["pooled_corr_corrected"] == pytest.approx(1.0, abs=0.01)
    assert result["slope_true_on_critic"] == pytest.approx(0.5, abs=0.01)
    assert result["ascent_ratio"] == pytest.approx(0.5, abs=0.01)
    assert result["norm_ratio_critic_over_true"] == pytest.approx(2.0, abs=0.02)
    assert result["cos_ceiling_median"] == pytest.approx(np.median(oracle.cosines(samples.mean(-1), truth)), abs=0.02)
    both = oracle.linearity_stats(samples, truth[..., None] + 1.5 * rng.normal(size=samples.shape))
    assert both["corr_corrected"] == pytest.approx(1.0, abs=0.02)


def test_within_stats_remove_rollout_noise():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(3000, 8)) + 10 * rng.normal(size=(3000, 1))  # state levels do not matter
    common = rng.normal(size=(3000, 1, 16))  # CRN: noise shared by every action of a state and rollout
    samples = q[..., None] + 3.0 * common + 0.8 * rng.normal(size=(3000, 8, 16))
    result = oracle.within_stats(q, samples)
    assert result["corr"] < 0.99
    assert result["corr_corrected"] == pytest.approx(1.0, abs=0.01)
    assert result["true_sd"] == pytest.approx(1.0, abs=0.03) and result["sd_ratio"] == pytest.approx(1.0, abs=0.03)


def test_gradient_stats_standard_errors_hold_under_shared_noise():
    """Directions of a state share their rollouts, so their noise correlates; the SEs must still be right."""
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(200, 6))
    critic = truth + rng.normal(size=truth.shape)
    slopes, ascents, slope_ses, ascent_ses = [], [], [], []
    for _ in range(300):
        shared = rng.normal(size=(200, 1, 8))
        samples = truth[..., None] + 4.0 * shared + 2.0 * rng.normal(size=(200, 6, 8))
        result = oracle.gradient_stats(samples, critic, critic)
        slopes.append(result["slope_true_on_critic"])
        ascents.append(result["ascent_ratio"])
        slope_ses.append(result["slope_se"])
        ascent_ses.append(result["ascent_se"])
    assert np.std(slopes) == pytest.approx(np.mean(slope_ses), rel=0.15)
    assert np.std(ascents) == pytest.approx(np.mean(ascent_ses), rel=0.15)
    # Pure noise hides |d|^2: the corrected statistics decline to answer instead of exploding.
    noisy = oracle.gradient_stats(0.01 * truth[..., None] + rng.normal(size=(200, 6, 8)), critic, critic)
    assert noisy["pooled_corr_corrected"] is None and noisy["norm_ratio_critic_over_true"] is None


def test_critic_pick_gain_is_unbiased():
    """A perfect critic's favourite of K gains E[max of K] - V: 1.4236 true sds for K = 8 normal actions."""
    rng = np.random.default_rng(4)
    q = rng.normal(size=(40_000, 8))
    result = oracle.within_stats(q, q[..., None] + 0.1 * rng.normal(size=(40_000, 8, 4)))
    assert result["critic_pick_gain"] == pytest.approx(1.4236, abs=0.02)
