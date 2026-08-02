import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
V3_SCRIPT = (
    ROOT
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v3_situglu.py"
)
V7_SCRIPT = (
    ROOT
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v7_rownorm_situglu.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V3 = _load_module("sf_vlam_v3_situglu_for_v7_test", V3_SCRIPT)
V7 = _load_module("sf_vlam_v7_rownorm_situglu", V7_SCRIPT)


def test_v7_keeps_official_situ_glu_equation_and_bound():
    gate = torch.linspace(-10_000.0, 10_000.0, 100_001, device="cuda")
    up = gate.flip(0)
    expected = (
        4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    ) * (25.0 * torch.tanh(up / 25.0))
    actual = V7.situ_glu(gate, up)
    torch.testing.assert_close(actual, expected)
    assert actual.abs().max() <= 100.0


def test_fixed_row_norm_linear_is_invariant_to_positive_row_rescaling():
    torch.manual_seed(3)
    layer = V7.layer_init(V7.FixedRowNormLinear(64, 43)).cuda()
    inputs = torch.randn(257, 64, device="cuda")
    expected = layer(inputs)

    row_scales = torch.logspace(-2, 2, 43, device="cuda").unsqueeze(1)
    with torch.no_grad():
        layer.weight.mul_(row_scales)

    actual = layer(inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    effective_weight = layer.weight * (
        layer.target_row_norm
        / layer.weight.norm(dim=1, keepdim=True).clamp_min(
            torch.finfo(layer.weight.dtype).eps
        )
    )
    expected_norm = torch.full((43,), 2.0**0.5, device="cuda")
    torch.testing.assert_close(
        effective_weight.norm(dim=1), expected_norm, atol=2e-6, rtol=2e-6
    )


def test_v7_initial_branch_matches_v3():
    torch.manual_seed(11)
    v3_branch = V3.SiTUGLUBranch(64).cuda()
    torch.manual_seed(11)
    v7_branch = V7.SiTUGLUBranch(64).cuda()

    for v3_parameter, v7_parameter in zip(
        v3_branch.parameters(), v7_branch.parameters(), strict=True
    ):
        torch.testing.assert_close(v7_parameter, v3_parameter, atol=0.0, rtol=0.0)

    inputs = torch.randn(4096, 64, device="cuda")
    torch.testing.assert_close(
        v7_branch(inputs), v3_branch(inputs), atol=2e-5, rtol=2e-5
    )


def test_v7_branch_preserves_parameters_and_v2_rng_position():
    branch = V7.SiTUGLUBranch(64)
    assert branch.hidden_dim == 43
    assert sum(parameter.numel() for parameter in branch.parameters()) == 8_256
    assert branch.gate.bias is None
    assert branch.up.bias is None
    assert branch.down.bias is None

    torch.manual_seed(7)
    V7.layer_init(torch.nn.Linear(64, 64))
    V7.layer_init(torch.nn.Linear(64, 64))
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    V7.SiTUGLUBranch(64)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng_state)


def test_v7_compiled_branch_forward_and_backward_are_finite():
    branch = torch.compile(V7.SiTUGLUBranch(64).cuda(), mode="reduce-overhead")
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


if __name__ == "__main__":
    test_v7_keeps_official_situ_glu_equation_and_bound()
    test_fixed_row_norm_linear_is_invariant_to_positive_row_rescaling()
    test_v7_initial_branch_matches_v3()
    test_v7_branch_preserves_parameters_and_v2_rng_position()
    test_v7_compiled_branch_forward_and_backward_are_finite()
