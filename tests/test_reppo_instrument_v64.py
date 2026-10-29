"""Contracts of v64's pure instrument pieces; no environments, CPU only."""
import math

import numpy as np
import torch

from cleanrl.reppo.ppo_continuous_action_reppo_instrument_v64 import (
    action_noise_rng, centered_log_ratio, gaussian_log_ratio, inverted_rows, lane_standard_error,
    realised_gain_terms, top_energy_share, tumbled_envs,
)


def host_draws(rng):
    """The trainer's draw: float32 normals written into a reused buffer."""
    out = np.empty((16, 6), dtype=np.float32)
    return [rng.standard_normal(out.shape, dtype=np.float32, out=out).copy() for _ in range(3)]


def test_stream_zero_is_the_v60_sampler():
    for seed in (1, 7, 123456):
        np.testing.assert_array_equal(host_draws(action_noise_rng(seed, 0)), host_draws(np.random.default_rng(seed)))


def test_other_streams_differ_from_stream_zero_and_each_other():
    draws = [np.stack(host_draws(action_noise_rng(1, stream))) for stream in range(4)]
    for i in range(4):
        for j in range(i):
            assert not np.array_equal(draws[i], draws[j])
            assert abs(np.corrcoef(draws[i].ravel(), draws[j].ravel())[0, 1]) < 0.2


def test_posture_uses_the_wrapped_pitch():
    half = math.pi / 2
    pitch = torch.tensor([0.0, half - 0.01, half + 0.01, math.pi, 3 * half - 0.01, 3 * half + 0.01,
                          2 * math.pi, -math.pi, -2 * math.pi, 5 * math.pi])
    expected = torch.tensor([False, False, True, True, True, False, False, True, False, True])
    assert torch.equal(inverted_rows(pitch), expected)


def test_tumbles_count_upright_to_inverted_steps_only():
    # Columns: tips over; starts inverted and recovers; inverted until a reset; stays upright.
    inverted = torch.tensor([[False, True, True, False],
                             [False, True, True, False],
                             [True, False, False, False]])
    # The last step of column 2's episode ends in its (inverted) final observation, not the reset one.
    next_inverted = torch.tensor([[False, True, True, False],
                                  [True, False, True, False],
                                  [True, False, False, False]])
    assert torch.equal(tumbled_envs(inverted, next_inverted), torch.tensor([True, False, False, False]))


def test_top_energy_share_on_a_spiked_field():
    energy = torch.ones(200)
    energy[[17, 140]] = 100.0
    share, rows = top_energy_share(energy, 0.01)
    torch.testing.assert_close(share, torch.tensor(200.0 / 398.0))
    assert sorted(rows.tolist()) == [17, 140]
    flat_share, _ = top_energy_share(torch.ones(200), 0.01)
    torch.testing.assert_close(flat_share, torch.tensor(0.01))


def test_log_ratio_matches_the_gaussian_densities():
    generator = torch.Generator().manual_seed(0)
    shape = (64, 6)
    noise, mean, old_mean = (torch.randn(shape, generator=generator, dtype=torch.float64) for _ in range(3))
    std, old_std = (torch.rand(shape, generator=generator, dtype=torch.float64) + 0.2 for _ in range(2))
    pre = mean + std * noise
    expected = (torch.distributions.Normal(mean, std).log_prob(pre)
                - torch.distributions.Normal(old_mean, old_std).log_prob(pre)).sum(-1)
    torch.testing.assert_close(gaussian_log_ratio(noise, mean, std, old_mean, old_std), expected)
    torch.testing.assert_close(gaussian_log_ratio(noise, mean, std, mean, std), torch.zeros(64, dtype=torch.float64))


def whitened_noise(rows, dims, generator):
    """A sample whose mean is exactly zero and whose covariance is exactly the identity."""
    noise = torch.randn((rows, dims), generator=generator, dtype=torch.float64)
    noise = noise - noise.mean(0)
    return noise @ torch.linalg.inv(torch.linalg.cholesky(noise.T @ noise / rows)).T


def test_realised_gain_recovers_a_linear_advantage_exactly():
    # A(u) = slope . (u - mu_new) is the advantage of pi_new under a linear reward, for which moving the mean
    # from mu_old to mu_new gains exactly slope . (mu_new - mu_old), whatever the (shared) std.
    generator = torch.Generator().manual_seed(1)
    rows, dims = 4096, 6
    noise = whitened_noise(rows, dims, generator)
    slope = torch.randn(dims, generator=generator, dtype=torch.float64)
    mean = torch.randn(dims, generator=generator, dtype=torch.float64).expand(rows, dims)
    old_mean = mean - 0.3 * torch.randn(dims, generator=generator, dtype=torch.float64)
    std = (torch.rand(dims, generator=generator, dtype=torch.float64) + 0.3).expand(rows, dims)
    advantages = (slope * std * noise).sum(-1) + 5.0  # the level must not leak in
    gain = realised_gain_terms(advantages, gaussian_log_ratio(noise, mean, std, old_mean, std)).mean()
    torch.testing.assert_close(gain, (slope * (mean[0] - old_mean[0])).sum())
    # No update, no gain.
    unchanged = realised_gain_terms(advantages, gaussian_log_ratio(noise, mean, std, mean, std))
    torch.testing.assert_close(unchanged.mean(), torch.zeros((), dtype=torch.float64))


def test_centered_log_ratio_keeps_baseline_error_out_of_the_gain():
    # States differ in how far their policy moved, and the advantage estimate G - b(s) carries a baseline
    # error that grows with that move. The raw log ratio's per-state mean, KL(new || old) > 0, turns the
    # error into gain; the centered one has zero mean over the action at every state.
    generator = torch.Generator().manual_seed(4)
    states, draws, dims = 512, 256, 6
    shape = (states, draws, dims)
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    mean = torch.zeros(shape, dtype=torch.float64)
    std = (torch.rand((states, 1, dims), generator=generator, dtype=torch.float64) + 0.3).expand(shape)
    move = torch.rand((states, 1, 1), generator=generator, dtype=torch.float64)  # per-state step size
    old_mean = (mean - move * std).expand(shape)
    slope = torch.randn(dims, generator=generator, dtype=torch.float64)
    true_advantage = (slope * std * noise).sum(-1)
    baseline_error = 20.0 * move.squeeze(-1).expand(states, draws)
    raw = gaussian_log_ratio(noise, mean, std, old_mean, std)
    centered = centered_log_ratio(noise, mean, std, old_mean, std)
    # Zero mean over the action at every state (Monte Carlo over 256 draws), unlike the raw ratio.
    assert centered.mean(1).abs().max() < 0.5
    assert centered.mean().abs() < 0.01 < raw.mean()
    exact = (slope * (mean - old_mean)[:, 0]).sum(-1).mean()  # linear advantage, shared std: slope . move
    estimate = realised_gain_terms((true_advantage + baseline_error).flatten(), centered.flatten()).mean()
    leaky = realised_gain_terms((true_advantage + baseline_error).flatten(), raw.flatten()).mean()
    torch.testing.assert_close(estimate, exact, atol=0.1, rtol=0.0)
    assert (leaky - exact).abs() > 1.0


def test_lane_standard_error_is_the_lane_jackknife():
    generator = torch.Generator().manual_seed(2)
    steps, envs = 50, 8
    terms = torch.randn((steps, envs), generator=generator, dtype=torch.float64) + torch.arange(envs)
    flat = terms.flatten()  # rows in (step, env) order
    leave_one_out = torch.stack([terms[:, [e for e in range(envs) if e != lane]].mean() for lane in range(envs)])
    jackknife = ((envs - 1) / envs * (leave_one_out - leave_one_out.mean()).square().sum()).sqrt()
    torch.testing.assert_close(lane_standard_error(flat, envs), jackknife)
