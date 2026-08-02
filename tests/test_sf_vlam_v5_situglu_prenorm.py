import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v5_situglu_prenorm.py"
)
SPEC = importlib.util.spec_from_file_location("sf_vlam_v5_situglu_prenorm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_situ_glu_matches_reference_equation_and_bound():
    gate = torch.linspace(-10_000.0, 10_000.0, 100_001, device="cuda")
    up = gate.flip(0)
    expected = (
        4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    ) * (25.0 * torch.tanh(up / 25.0))
    actual = MODULE.situ_glu(gate, up)
    torch.testing.assert_close(actual, expected)
    assert actual.abs().max() <= 100.0


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
    torch.testing.assert_close(rms_ratio, torch.ones_like(rms_ratio), atol=0.025, rtol=0.025)


def test_situ_branch_forward_and_backward_are_finite():
    branch = MODULE.SiTUGLUBranch(64).cuda()
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


def test_preactivation_norm_preserves_initial_rms_and_removes_weight_scale():
    branch = MODULE.SiTUGLUBranch(64).cuda()
    inputs = torch.randn(512, 64, device="cuda")
    gate_before, up_before = branch.preactivations(inputs)
    target_rms = 2.0**0.5
    torch.testing.assert_close(
        gate_before.square().mean(-1).sqrt(),
        torch.full((inputs.shape[0],), target_rms, device="cuda"),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        up_before.square().mean(-1).sqrt(),
        torch.full((inputs.shape[0],), target_rms, device="cuda"),
        atol=2e-5,
        rtol=2e-5,
    )

    with torch.no_grad():
        branch.gate.weight.mul_(10.0)
        branch.up.weight.mul_(7.0)
    gate_after, up_after = branch.preactivations(inputs)
    torch.testing.assert_close(gate_after, gate_before, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(up_after, up_before, atol=2e-5, rtol=2e-5)
    assert branch.gate_norm.elementwise_affine is False
    assert branch.up_norm.elementwise_affine is False


if __name__ == "__main__":
    test_situ_glu_matches_reference_equation_and_bound()
    test_situ_branch_is_parameter_matched_and_bias_free()
    test_situ_branch_preserves_v2_rng_position()
    test_situ_branch_matches_v2_initial_output_rms()
    test_situ_branch_forward_and_backward_are_finite()
    test_preactivation_norm_preserves_initial_rms_and_removes_weight_scale()
