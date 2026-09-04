import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "streaming_switch_regression_v4.py"
)
spec = importlib.util.spec_from_file_location("streaming_switch_regression_v4", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
make_mixture_filter_step = module.make_mixture_filter_step


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "mixture filter tests require CUDA"
    return torch.device("cuda")


def small_args():
    return Args(
        method="bayes_mixture",
        output_dim=2,
        total_steps=2_000,
    )


def test_prediction_is_causal_before_observation_update(device):
    args = small_args()
    first_step, first_weight, first_state = make_mixture_filter_step(args, device)
    second_step, second_weight, second_state = make_mixture_filter_step(args, device)
    x = torch.zeros(args.input_dim, device=device)
    x[0] = 1.0
    latent = torch.zeros(args.output_dim, device=device)

    first = first_step(x, torch.full_like(latent, 20.0), latent)
    second = second_step(x, torch.full_like(latent, 2.0), latent)

    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert not torch.equal(first_weight, second_weight)
    assert not torch.equal(first_state["log_model_prob"], second_state["log_model_prob"])


def test_model_probabilities_normalize_and_variances_stay_positive(device):
    args = small_args()
    step, weight, state = make_mixture_filter_step(args, device)
    generator = torch.Generator(device=device).manual_seed(31)

    for _ in range(1_000):
        x = torch.randn(args.input_dim, device=device, generator=generator)
        x /= x.norm()
        observed = torch.randn(args.output_dim, device=device, generator=generator)
        step(x, observed, observed)

    probabilities = state["log_model_prob"].exp()
    torch.testing.assert_close(
        probabilities.sum(0),
        torch.ones(args.output_dim, device=device),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.isfinite(weight).all()
    assert torch.isfinite(state["posterior_var"]).all()
    assert torch.all(state["posterior_var"] > 0)


def test_student_evidence_limits_isolated_outlier_update(device):
    args = small_args()
    step, weight, _ = make_mixture_filter_step(args, device)
    x = torch.zeros(args.input_dim, device=device)
    x[0] = 1.0
    target = torch.zeros(args.output_dim, device=device)

    _, _, diagnostics = step(x, torch.full_like(target, 1_000.0), target)

    assert diagnostics[0] < 0.1
    assert torch.isfinite(weight).all()
    assert weight.square().mean().sqrt() < 1.0


def test_process_noise_bank_is_strictly_ordered(device):
    args = small_args()
    _, _, state = make_mixture_filter_step(args, device)
    process_vars = state["process_vars"]

    assert torch.all(process_vars > 0)
    assert torch.all(process_vars[1:] > process_vars[:-1])
