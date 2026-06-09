import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dgae_v2 import (
    Args,
    distributional_lambda_target,
)
from cleanrl.shared.hl_gauss import HLGaussSupport


def _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam):
    """Reference scalar GAE lambda-return, mirroring the in-file recursion."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm = 1.0 - next_done
            nextval = next_value
        else:
            nonterm = 1.0 - dones[t + 1]
            nextval = values[t + 1]
        delta = rewards[t] + gamma * nextval * nonterm - values[t]
        adv[t] = last = delta + gamma * lam * nonterm * last
    return adv + values


def _make_support():
    # Linear (non-symlog) support so HL-Gauss projection preserves the raw mean;
    # wide + central so edge truncation bias is negligible.
    return HLGaussSupport(
        num_bins=201, v_min=-50.0, v_max=50.0, sigma_ratio=0.5,
        device=torch.device("cpu"), use_symlog=False, support_is_edges=True,
    )


def _random_probs(shape, n, generator):
    logits = torch.randn(*shape, n, generator=generator)
    return torch.softmax(logits, dim=-1)


def test_distributional_lambda_target_mean_matches_scalar_lambda_return():
    g = torch.Generator().manual_seed(0)
    T, B = 6, 3
    support = _make_support()
    raw_support = support.support  # linear => raw == coord
    n = support.num_bins
    gamma, lam = 0.99, 0.95

    value_probs = _random_probs((T, B), n, g)
    bootstrap_probs = _random_probs((B,), n, g)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)        # no episode boundaries in this test
    next_done = torch.zeros(B)

    lam_probs, target_std = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, raw_support, gamma, lam, support
    )

    # Mean of the distributional target should equal the scalar lambda-return.
    values = (value_probs * raw_support).sum(-1)
    next_value = (bootstrap_probs * raw_support).sum(-1)
    scalar_ret = _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam)
    dist_mean = (lam_probs * raw_support).sum(-1)

    assert torch.allclose(dist_mean, scalar_ret, atol=0.05), (dist_mean - scalar_ret).abs().max()
    # Probabilities are valid and the target carries genuine spread.
    assert torch.allclose(lam_probs.sum(-1), torch.ones(T, B), atol=1e-5)
    assert (target_std > 0).all()


def test_terminal_step_collapses_to_reward_point_mass():
    g = torch.Generator().manual_seed(1)
    T, B = 4, 2
    support = _make_support()
    raw_support = support.support
    n = support.num_bins
    gamma, lam = 0.99, 0.95

    value_probs = _random_probs((T, B), n, g)
    bootstrap_probs = _random_probs((B,), n, g)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)
    dones[2] = 1.0  # s_2 is terminal/reset => transition at t=1 must not bootstrap
    next_done = torch.zeros(B)

    lam_probs, _ = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, raw_support, gamma, lam, support
    )

    # At t=1 the bootstrap is masked off, so the target mean is just r_1.
    dist_mean = (lam_probs * raw_support).sum(-1)
    assert torch.allclose(dist_mean[1], rewards[1], atol=0.05), (dist_mean[1] - rewards[1]).abs().max()


def test_lambda_zero_point_mass_recovers_scalar_one_step_target():
    # At lambda=0 (pure 1-step, no mixture) with point-mass value dists, the
    # distributional target collapses to v1's project(1-step return) exactly.
    support = _make_support()
    raw_support = support.support
    n = support.num_bins
    gamma, lam = 0.99, 0.0
    T, B = 5, 2

    def point_mass(idx):
        p = torch.zeros(n)
        p[idx] = 1.0
        return p

    value_probs = torch.stack([torch.stack([point_mass(100 + t) for _ in range(B)]) for t in range(T)])
    bootstrap_probs = torch.stack([point_mass(100) for _ in range(B)])
    rewards = torch.randn(T, B, generator=torch.Generator().manual_seed(2))
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)

    lam_probs, _ = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, raw_support, gamma, lam, support
    )

    values = (value_probs * raw_support).sum(-1)
    next_value = (bootstrap_probs * raw_support).sum(-1)
    scalar_ret = _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam)
    v1_target = support.project(scalar_ret)

    assert torch.allclose(lam_probs, v1_target, atol=1e-4)


def test_distributional_target_is_wider_than_scalar_projection():
    # The whole point of v2: with genuine (non-point-mass) value dists, the TD(lambda)
    # mixture carries strictly more spread than v1's smoothed-point-mass scalar target.
    g = torch.Generator().manual_seed(3)
    T, B = 6, 3
    support = _make_support()
    raw_support = support.support
    n = support.num_bins
    gamma, lam = 0.99, 0.95

    value_probs = _random_probs((T, B), n, g)
    bootstrap_probs = _random_probs((B,), n, g)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)

    lam_probs, target_std = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, raw_support, gamma, lam, support
    )

    values = (value_probs * raw_support).sum(-1)
    next_value = (bootstrap_probs * raw_support).sum(-1)
    scalar_ret = _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam)
    v1_target = support.project(scalar_ret)
    mean_raw = (v1_target * raw_support).sum(-1, keepdim=True)
    v1_std = (v1_target * (raw_support - mean_raw) ** 2).sum(-1).clamp_min(0).sqrt()

    assert (target_std > v1_std).float().mean() > 0.9


def test_v2_defaults():
    args = Args()
    assert args.dist_lambda_target is True
    assert args.dgae_policy_adv is True
    assert args.value_symlog is True
