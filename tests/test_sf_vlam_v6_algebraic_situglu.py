import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v6_algebraic_situglu.py"
)
SPEC = importlib.util.spec_from_file_location("sf_vlam_v6_algebraic_situglu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_situ_glu_matches_algebraic_equation_and_bound():
    gate = torch.linspace(
        -10_000.0, 10_000.0, 100_001, device="cuda", dtype=torch.float64
    )
    up = gate.flip(0)
    gate_cap = 4.0 * gate / torch.hypot(gate, torch.full_like(gate, 4.0))
    gate_factor = torch.where(gate <= 0.0, gate, gate_cap)
    up_cap = 25.0 * up / torch.hypot(up, torch.full_like(up, 25.0))
    expected = (gate_factor * torch.sigmoid(gate)) * up_cap
    actual = MODULE.situ_glu(gate, up)
    torch.testing.assert_close(actual, expected)
    assert actual.abs().max() <= 100.0


def test_algebraic_caps_retain_nonzero_extreme_gradients():
    gate = torch.tensor(10_000.0, device="cuda", requires_grad=True)
    up = torch.tensor(10_000.0, device="cuda", requires_grad=True)
    MODULE.situ_glu(gate, up).backward()
    assert torch.isfinite(gate.grad)
    assert torch.isfinite(up.grad)
    assert gate.grad > 0.0
    assert up.grad > 0.0


def test_algebraic_cap_has_correct_second_derivative():
    value = torch.tensor(3.0, device="cuda", dtype=torch.float64, requires_grad=True)
    capped = MODULE.algebraic_cap(value, 4.0)
    first, = torch.autograd.grad(capped, value, create_graph=True)
    second, = torch.autograd.grad(first, value)
    expected_first = (1.0 + (value / 4.0).square()).pow(-1.5)
    expected_second = (
        -3.0 * value / 16.0 * (1.0 + (value / 4.0).square()).pow(-2.5)
    )
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second, expected_second)


def test_algebraic_situ_second_moment_is_recomputed():
    recomputed = MODULE._algebraic_situ_gaussian_second_moment()
    assert recomputed == MODULE.ALGEBRAIC_SITU_SECOND_MOMENT
    assert abs(recomputed - 1.1716758983237182) < 1e-14


def test_situ_branch_is_parameter_matched_and_bias_free():
    branch = MODULE.SiTUGLUBranch(64).cuda()
    parameter_count = sum(parameter.numel() for parameter in branch.parameters())
    old_parameter_count = 2 * (64 * 64 + 64)
    assert branch.hidden_dim == 43
    assert parameter_count == 8_256
    assert abs(parameter_count - old_parameter_count) / old_parameter_count < 0.01
    assert branch.gate.bias is None
    assert branch.up.bias is None
    assert branch.down.bias is None


def test_situ_branch_preserves_v2_rng_position():
    torch.manual_seed(7)
    MODULE.layer_init(torch.nn.Linear(64, 64))
    MODULE.layer_init(torch.nn.Linear(64, 64))
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    MODULE.SiTUGLUBranch(64)
    actual_rng_state = torch.get_rng_state()
    torch.testing.assert_close(actual_rng_state, expected_rng_state)


def test_situ_branch_matches_v2_initial_output_rms():
    torch.manual_seed(11)
    old_in = MODULE.layer_init(torch.nn.Linear(64, 64)).cuda()
    old_out = MODULE.layer_init(torch.nn.Linear(64, 64)).cuda()
    inputs = torch.randn(65_536, 64, device="cuda")
    old_output = old_out(torch.relu(old_in(inputs)).square())

    torch.manual_seed(11)
    branch = MODULE.SiTUGLUBranch(64).cuda()
    situ_output = branch(inputs)
    rms_ratio = situ_output.square().mean().sqrt() / old_output.square().mean().sqrt()
    torch.testing.assert_close(rms_ratio, torch.ones_like(rms_ratio), atol=0.05, rtol=0.05)


def test_compiled_situ_branch_forward_and_backward_are_finite():
    branch = torch.compile(MODULE.SiTUGLUBranch(64).cuda(), mode="reduce-overhead")
    inputs = torch.randn(128, 64, device="cuda", requires_grad=True)
    output = branch(inputs)
    loss = output.square().mean()
    loss.backward()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in branch.parameters()
    )


if __name__ == "__main__":
    test_situ_glu_matches_algebraic_equation_and_bound()
    test_algebraic_caps_retain_nonzero_extreme_gradients()
    test_algebraic_cap_has_correct_second_derivative()
    test_algebraic_situ_second_moment_is_recomputed()
    test_situ_branch_is_parameter_matched_and_bias_free()
    test_situ_branch_preserves_v2_rng_position()
    test_situ_branch_matches_v2_initial_output_rms()
    test_compiled_situ_branch_forward_and_backward_are_finite()
