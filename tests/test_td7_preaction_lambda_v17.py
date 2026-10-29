"""Analytical tensor tests only: no model construction or ML execution."""
import torch

from cleanrl.td7.latentq.td7_latentq_preaction_lambda_v17 import Args, lambda_returns


def compute(rewards, values, masks=None, boundaries=None, gamma=0.5, trace_lambda=0.5, horizon=32):
    rewards = torch.tensor(rewards, dtype=torch.float64).reshape(-1, 1, 1)
    values = torch.tensor(values, dtype=torch.float64).reshape_as(rewards)
    masks = torch.ones_like(rewards) if masks is None else torch.tensor(masks).reshape_as(rewards)
    boundaries = torch.zeros_like(rewards) if boundaries is None else torch.tensor(boundaries).reshape_as(rewards)
    return lambda_returns(rewards, masks, boundaries, values, gamma, trace_lambda, horizon).flatten()


def test_two_step_lambda_return_and_final_bootstrap():
    actual = compute([1.0, 2.0], [10.0, 20.0])
    torch.testing.assert_close(actual, torch.tensor([6.5, 12.0], dtype=torch.float64))


def test_lambda_zero_is_frozen_one_step_for_every_row():
    actual = compute([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], trace_lambda=0)
    torch.testing.assert_close(actual, torch.tensor([6.0, 12.0, 18.0], dtype=torch.float64))


def test_thirty_two_step_segments_cut_without_shortening_rollout():
    length = 1024
    actual = compute([1.0] * length, [0.0] * length, gamma=1, trace_lambda=1)
    expected = torch.arange(32, 0, -1, dtype=torch.float64).repeat(length // 32)
    torch.testing.assert_close(actual, expected)
    assert Args().num_steps == 1024 and Args().trace_horizon == 32


def test_segment_end_bootstraps_its_own_next_value():
    actual = compute([1.0, 2.0, 100.0], [10.0, 20.0, 30.0], trace_lambda=1, horizon=2)
    torch.testing.assert_close(actual, torch.tensor([7.0, 12.0, 115.0], dtype=torch.float64))


def test_terminal_and_truncation_stop_autoreset_trace():
    terminal = compute([1.0, 1000.0], [10.0, 20.0], masks=[0, 1], boundaries=[1, 0], trace_lambda=1)
    truncated = compute([1.0, 1000.0], [10.0, 20.0], masks=[1, 1], boundaries=[1, 0], trace_lambda=1)
    assert terminal[0] == 1.0
    assert truncated[0] == 6.0
    assert terminal[1] == truncated[1] == 1010.0


def test_parallel_lanes_are_independent_and_short_final_segment_is_valid():
    rewards = torch.tensor([[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]], dtype=torch.float64)
    values = torch.full_like(rewards, 4.0)
    result = lambda_returns(rewards, torch.ones_like(rewards), torch.zeros_like(rewards), values, 1, 1, 2)
    expected = torch.tensor([[[7.0], [34.0]], [[6.0], [24.0]], [[7.0], [34.0]]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_warmup_mu_and_sigma_tensor_shape_matches_all_rollout_rows():
    # The warmup bootstrap is a Gaussian center query, not a sampled action.
    staged_mu = torch.empty(1024 * 16, 6)
    mu = torch.zeros_like(staged_mu)
    sigma = torch.full_like(mu, Args().warmup_sigma)
    assert mu.shape == sigma.shape == (16384, 6)
    assert torch.count_nonzero(mu) == 0
    torch.testing.assert_close(sigma, torch.full((16384, 6), 0.6))
