import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "streaming_switch_regression_v2.py"
)
spec = importlib.util.spec_from_file_location("streaming_switch_regression_v2", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
make_filter_step = module.make_filter_step
make_stream = module.make_stream


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "streaming filter tests require CUDA"
    return torch.device("cuda")


def small_args(method):
    return Args(
        method=method,
        condition="switching_outlier",
        total_steps=2_000,
        output_dim=2,
        min_regime_steps=500,
        max_regime_steps=600,
    )


def test_prediction_is_prequential_and_q_changes_only_next_update(device):
    adaptive_args = small_args("robust_adaptive")
    fixed_args = small_args("robust_fixed")
    adaptive_step, adaptive_weight, adaptive_state = make_filter_step(adaptive_args, device)
    fixed_step, fixed_weight, fixed_state = make_filter_step(fixed_args, device)
    x = torch.zeros(adaptive_args.input_dim, device=device)
    x[0] = 1.0
    observed = torch.full((adaptive_args.output_dim,), 20.0, device=device)
    latent = torch.zeros_like(observed)

    adaptive_result = adaptive_step(x, observed, latent)
    fixed_result = fixed_step(x, observed, latent)

    torch.testing.assert_close(adaptive_result[1], fixed_result[1], rtol=0, atol=0)
    torch.testing.assert_close(adaptive_weight, fixed_weight, rtol=0, atol=0)
    assert not torch.equal(adaptive_state["log_q"], fixed_state["log_q"])


def test_student_measurement_moves_less_than_gaussian_on_outlier(device):
    robust_args = small_args("robust_fixed")
    gaussian_args = small_args("gaussian_filter")
    robust_step, robust_weight, _ = make_filter_step(robust_args, device)
    gaussian_step, gaussian_weight, _ = make_filter_step(gaussian_args, device)
    x = torch.zeros(robust_args.input_dim, device=device)
    x[0] = 1.0
    observed = torch.full((robust_args.output_dim,), 100.0, device=device)
    latent = torch.zeros_like(observed)

    robust_step(x, observed, latent)
    gaussian_step(x, observed, latent)

    assert robust_weight.square().sum() < gaussian_weight.square().sum()


def test_joseph_diagonal_stays_positive(device):
    args = small_args("robust_adaptive")
    step, _, state = make_filter_step(args, device)
    generator = torch.Generator(device=device).manual_seed(4)
    for _ in range(1_000):
        x = torch.randn(args.input_dim, device=device, generator=generator)
        x = x / x.norm()
        observed = torch.randn(args.output_dim, device=device, generator=generator)
        step(x, observed, observed)

    assert torch.isfinite(state["posterior_var"]).all()
    assert torch.all(state["posterior_var"] > 0)
    assert torch.isfinite(state["log_q"]).all()


def test_stream_has_unit_norm_and_isolated_outliers(device):
    args = small_args("robust_adaptive")
    stream = make_stream(args, device)
    norms = stream["x"].square().sum(dim=1)
    events = stream["outlier_mask"].any(dim=1)

    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6)
    assert not torch.any(events[1:] & events[:-1])
    assert not torch.any(events & stream["switch_mask"])
    assert stream["gram_min"].min() > 0
