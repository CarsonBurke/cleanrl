"""Beta sampler identity: the CUDA path (queue with mlq) and the fused host path."""

import warnings

import numpy as np
import pytest
import torch
from torch.distributions import Beta, Distribution

from cleanrl.shared.sampling import (
    make_beta_sampler, sample_beta_actions, sample_beta_actions_host,
)


def host_case(index):
    """A (logits, low, high, num_envs, act_dim) case; every third is extreme."""
    meta = np.random.default_rng(4000 + index)
    num_envs, act_dim = int(meta.integers(1, 65)), int(meta.integers(1, 25))
    logits = (meta.normal(size=(num_envs, 2 * act_dim))
              * float(meta.choice([0.5, 4.0, 40.0]))).astype(np.float32)
    if index % 3 == 0:
        # Denormals, both signed zeros, the logaddexp branch point and inputs
        # far enough out that softplus saturates to 0 and to the identity.
        extreme = np.array([0.0, -0.0, 1e-45, -1e-45, 1e-38, 120.0, -120.0, 1e30,
                            -1e30, 0.6931472, 87.9, -87.9], dtype=np.float32)
        flat = logits.reshape(-1)
        flat[: min(flat.size, extreme.size)] = extreme[: min(flat.size, extreme.size)]
    if index % 3 == 1:
        low = meta.uniform(-9.0, 0.0, size=act_dim).astype(np.float32)
        high = meta.uniform(0.0, 9.0, size=act_dim).astype(np.float32)
    elif index % 3 == 2:
        low, high = np.float32(-2.5), np.float32(0.75)
    else:
        low = np.full(act_dim, -1.0, np.float32)
        high = np.full(act_dim, 1.0, np.float32)
    return logits, low, high, num_envs, act_dim


def test_fused_beta_head_matches_the_numpy_path_bitwise():
    """Not "within tolerance": the draws feed the learner, so max|delta| is 0."""
    compared = 0
    for index in range(300):
        logits, low, high, num_envs, act_dim = host_case(index)
        sampler = make_beta_sampler(num_envs, act_dim, low, high)
        assert sampler.fused, sampler.fallback_reason
        reference_rng = np.random.default_rng(index)
        fused_rng = np.random.default_rng(index)
        for _ in range(5):
            want_native, want_action = sample_beta_actions_host(
                logits, low, high, reference_rng)
            got_native, got_action = sampler(logits, fused_rng)
            want_action = np.asarray(want_action, dtype=np.float32)
            assert np.array_equal(want_native.view(np.uint32), got_native.view(np.uint32))
            assert np.array_equal(want_action.view(np.uint32), got_action.view(np.uint32))
            assert np.max(np.abs(want_native.astype(np.float64)
                                 - got_native.astype(np.float64))) == 0.0
            assert np.max(np.abs(want_action.astype(np.float64)
                                 - got_action.astype(np.float64))) == 0.0
            compared += want_native.size + want_action.size
    assert compared > 1_000_000, compared


def test_fused_beta_head_leaves_the_generator_stream_untouched():
    """Same generator, same values, same call count: state must match exactly."""
    logits, low, high, num_envs, act_dim = host_case(1)
    sampler = make_beta_sampler(num_envs, act_dim, low, high)
    reference_rng = np.random.default_rng(99)
    fused_rng = np.random.default_rng(99)
    for _ in range(8):
        sample_beta_actions_host(logits, low, high, reference_rng)
        sampler(logits, fused_rng)
        assert fused_rng.bit_generator.state == reference_rng.bit_generator.state
    assert np.array_equal(reference_rng.random(64), fused_rng.random(64))
    assert np.array_equal(reference_rng.standard_normal((4, 5)),
                          fused_rng.standard_normal((4, 5)))


def test_make_beta_sampler_warns_once_with_the_kernel_reason_when_it_cannot_fuse():
    """A silent 4-6us/step regression is exactly what must not ship."""
    logits, _, _, num_envs, act_dim = host_case(2)
    low = np.full(act_dim, -1.0, np.float64)  # FP64 bounds: NumPy's promotion
    high = np.full(act_dim, 1.0, np.float64)  # decides the rounding, so decline
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = make_beta_sampler(num_envs, act_dim, low, high)
    assert sampler.fused is False
    assert "float32 action bounds" in sampler.fallback_reason
    assert len(caught) == 1 and caught[0].category is RuntimeWarning
    assert sampler.fallback_reason in str(caught[0].message)
    reference_rng, fallback_rng = np.random.default_rng(5), np.random.default_rng(5)
    want = sample_beta_actions_host(logits, low, high, reference_rng)
    got = sampler(logits, fallback_rng)
    for expected, actual in zip(want, got):
        assert np.array_equal(np.asarray(expected), np.asarray(actual))
    assert reference_rng.bit_generator.state == fallback_rng.bit_generator.state
    # Opting out is deliberate, so it stays quiet.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opted_out = make_beta_sampler(num_envs, act_dim, low, high, fused=False)
    assert opted_out.fused is False and opted_out.fallback_reason is None
    assert [str(entry.message) for entry in caught] == []


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
