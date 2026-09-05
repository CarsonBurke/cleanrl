"""CUDA sampler identity: queue this module through mlq with other GPU tests."""

import pytest
import torch
from torch.distributions import Beta, Distribution

from cleanrl.shared.sampling import sample_beta_actions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 6), (16, 3), (2496, 6)])
def test_beta_sampler_preserves_draws_rng_and_global_validation(shape):
    alpha = torch.linspace(1.01, 80.0, shape[0], device="cuda")[:, None].expand(shape)
    beta = alpha.flip(0)
    low = torch.full((shape[-1],), -1.0, device="cuda")
    high = -low
    validation_before = Distribution._validate_args
    torch.cuda.manual_seed(1)
    rng_before = torch.cuda.get_rng_state()
    expected_native = Beta(alpha, beta).sample().clamp(1e-6, 1 - 1e-6)
    expected_action = low + (high - low) * expected_native
    expected_rng = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_before)
    native, action = sample_beta_actions(alpha, beta, low, high)
    assert torch.equal(native, expected_native)
    assert torch.equal(action, expected_action)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    assert Distribution._validate_args == validation_before
