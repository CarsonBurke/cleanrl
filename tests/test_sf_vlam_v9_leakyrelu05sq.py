import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "sf_vlam_v2",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v2.py",
)
MODULE = _load(
    "sf_vlam_v9_leakyrelu05sq",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)


def test_activation_matches_requested_function_and_has_negative_gradient():
    inputs = torch.tensor([-3.0, -0.5, 0.5, 3.0], device="cuda", requires_grad=True)
    actual = MODULE.LeakyReLUSquared()(inputs)
    expected = F.leaky_relu(inputs, negative_slope=0.5).square()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.all(inputs.grad != 0)
    assert torch.all(inputs.grad[:2] < 0)


def test_branch_is_parameter_matched_and_preserves_rng_position():
    torch.manual_seed(7)
    base_branch = BASE._branch_body(64)
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    branch = MODULE._branch_body(64)
    actual_rng_state = torch.get_rng_state()

    assert sum(p.numel() for p in branch.parameters()) == sum(
        p.numel() for p in base_branch.parameters()
    )
    torch.testing.assert_close(actual_rng_state, expected_rng_state)


def test_variance_matched_init_preserves_relu_squared_output_rms():
    inputs = torch.randn(65_536, 64, device="cuda")

    torch.manual_seed(11)
    base_branch = BASE._branch_body(64).cuda()
    base_output = base_branch(inputs)

    torch.manual_seed(11)
    branch = MODULE._branch_body(64).cuda()
    output = branch(inputs)

    rms_ratio = output.square().mean().sqrt() / base_output.square().mean().sqrt()
    torch.testing.assert_close(rms_ratio, torch.ones_like(rms_ratio), atol=0.05, rtol=0.05)


def test_compiled_branch_forward_and_backward_are_finite():
    branch = torch.compile(MODULE._branch_body(64).cuda(), mode="reduce-overhead")
    inputs = torch.randn(128, 64, device="cuda", requires_grad=True)
    output = branch(inputs)
    output.square().mean().backward()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in branch.parameters()
    )


def test_scalar_lambda_is_the_default():
    assert MODULE.Args().per_dim_lambda is False


if __name__ == "__main__":
    test_activation_matches_requested_function_and_has_negative_gradient()
    test_branch_is_parameter_matched_and_preserves_rng_position()
    test_variance_matched_init_preserves_relu_squared_output_rms()
    test_compiled_branch_forward_and_backward_are_finite()
    test_scalar_lambda_is_the_default()
