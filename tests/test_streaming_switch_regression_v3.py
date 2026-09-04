import importlib.util
import math
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "streaming_switch_regression_v3.py"
)
spec = importlib.util.spec_from_file_location("streaming_switch_regression_v3", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
make_temporal_hyperfilter_step = module.make_temporal_hyperfilter_step


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "streaming filter tests require CUDA"
    return torch.device("cuda")


def hyper_args(process_variance):
    return Args(
        method="robust_hyperq",
        output_dim=2,
        filter_process_variance=process_variance,
        filter_q_rate=0.0,
        filter_q_prior=0.0,
        filter_q_min=1e-6,
        filter_q_max=1.0,
    )


def prequential_student_nll(weight, posterior_var, log_q, x, observed, args):
    process_var = log_q.exp()
    prediction = torch.nn.functional.linear(x, weight)
    residual = observed - prediction
    innovation_var = args.noise_std**2 + (
        (posterior_var + process_var.unsqueeze(1)) * x.square().unsqueeze(0)
    ).sum(1)
    delta = residual.square() / innovation_var
    return (
        0.5 * innovation_var.log()
        + 0.5 * (args.filter_student_df + 1.0) * torch.log1p(delta / args.filter_student_df)
    ).sum()


def test_temporal_q_score_matches_two_step_finite_difference(device):
    q = 0.05
    epsilon = 1e-3
    x0 = torch.zeros(32, device=device)
    x0[:3] = torch.tensor((0.8, -0.5, 0.3), device=device)
    x0 /= x0.norm()
    y0 = torch.tensor((1.2, -0.7), device=device)
    x1 = torch.zeros(32, device=device)
    x1[:3] = torch.tensor((-0.2, 0.9, 0.4), device=device)
    x1 /= x1.norm()
    y1 = torch.tensor((-0.4, 1.1), device=device)

    losses = []
    for offset in (-epsilon, epsilon):
        args = hyper_args(q * math.exp(offset))
        step, weight, state = make_temporal_hyperfilter_step(args, device)
        step(x0, y0, y0)
        losses.append(
            prequential_student_nll(
                weight,
                state["posterior_var"],
                state["log_q"],
                x1,
                y1,
                args,
            )
        )
    finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)

    args = hyper_args(q)
    step, _, _ = make_temporal_hyperfilter_step(args, device)
    step(x0, y0, y0)
    _, _, diagnostics = step(x1, y1, y1)
    reported_directional_derivative = diagnostics[5] * args.output_dim

    torch.testing.assert_close(
        reported_directional_derivative,
        finite_difference,
        rtol=2e-2,
        atol=2e-4,
    )


def test_temporal_sensitivities_and_variance_remain_finite(device):
    args = hyper_args(1e-4)
    args.filter_q_rate = 0.1
    step, weight, state = make_temporal_hyperfilter_step(args, device)
    generator = torch.Generator(device=device).manual_seed(17)

    for _ in range(1_000):
        x = torch.randn(args.input_dim, device=device, generator=generator)
        x /= x.norm()
        observed = torch.randn(args.output_dim, device=device, generator=generator)
        step(x, observed, observed)

    assert torch.isfinite(weight).all()
    assert torch.isfinite(state["posterior_var"]).all()
    assert torch.all(state["posterior_var"] > 0)
    assert torch.isfinite(state["mean_sensitivity"]).all()
    assert torch.isfinite(state["var_sensitivity"]).all()
    assert torch.isfinite(state["log_q"]).all()
