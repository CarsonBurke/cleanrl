import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v4_situglu_stablemoe.py"
)
SPEC = importlib.util.spec_from_file_location("sf_vlam_v4_situglu_stablemoe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

V3_SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v3_situglu.py"
)
V3_SPEC = importlib.util.spec_from_file_location("sf_vlam_v3_situglu_for_v4_test", V3_SCRIPT)
V3 = importlib.util.module_from_spec(V3_SPEC)
assert V3_SPEC.loader is not None
V3_SPEC.loader.exec_module(V3)

V2_SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v2.py"
)
V2_SPEC = importlib.util.spec_from_file_location("sf_vlam_v2_for_v4_test", V2_SCRIPT)
V2 = importlib.util.module_from_spec(V2_SPEC)
assert V2_SPEC.loader is not None
V2_SPEC.loader.exec_module(V2)


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
    torch.testing.assert_close(rms_ratio, torch.ones_like(rms_ratio), atol=0.05, rtol=0.05)


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


def test_stable_moe_is_parameter_matched_and_seed_paired():
    torch.manual_seed(19)
    v2_trunk = V2.ThinkTrunk(17, 64, 3, 16)
    v2_parameter_count = sum(parameter.numel() for parameter in v2_trunk.parameters())

    torch.manual_seed(19)
    v3_trunk = V3.ThinkTrunk(17, 64, 3, 16)
    v3_rng_state = torch.get_rng_state()

    torch.manual_seed(19)
    v4_trunk = MODULE.ThinkTrunk(17, 64, 3, 16)
    v4_rng_state = torch.get_rng_state()
    v4_parameter_count = sum(parameter.numel() for parameter in v4_trunk.parameters())

    assert v4_trunk.blocks[0].dense.hidden_dim == 43
    assert v4_trunk.blocks[0].experts[0].hidden_dim == 42
    assert v4_trunk.blocks[0].moe_out_norm.elementwise_affine is False
    assert v4_trunk.blocks[0].moe_out_proj.bias is None
    assert v2_parameter_count - v4_parameter_count == 192
    torch.testing.assert_close(v4_rng_state, v3_rng_state)
    torch.testing.assert_close(v4_trunk.entry.weight, v3_trunk.entry.weight)
    torch.testing.assert_close(v4_trunk.out_proj.weight, v3_trunk.out_proj.weight)
    for v4_block, v3_block in zip(v4_trunk.blocks, v3_trunk.blocks):
        torch.testing.assert_close(v4_block.in_proj.weight, v3_block.in_proj.weight)
        torch.testing.assert_close(v4_block.gate.weight, v3_block.gate.weight)


def test_stable_moe_outer_projection_has_matched_rms_and_finite_gradients():
    block = MODULE.ThinkBlock(64, 64, 16).cuda()
    cat_feats = torch.randn(256, 64, device="cuda", requires_grad=True)
    x0 = torch.randn(256, 64, device="cuda", requires_grad=True)

    m_in = block.moe_norm(cat_feats)
    weights = torch.softmax(block.gate(m_in), dim=-1)
    all_out = torch.stack([expert(m_in) for expert in block.experts], dim=1)
    routed = (weights.unsqueeze(-1) * all_out).sum(dim=1)
    stable_routed = block.moe_out_proj(block.moe_out_norm(routed))
    expected_rms = (12.0 * 0.203988672) ** 0.5
    assert abs(float(stable_routed.square().mean().sqrt()) - expected_rms) < 1e-3

    output = block(cat_feats, x0)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(cat_feats.grad).all()
    assert torch.isfinite(x0.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )


def test_stable_moe_initial_routed_rms_matches_v3():
    inputs = torch.randn(2_048, 64, device="cuda")
    torch.manual_seed(23)
    v3_block = V3.ThinkBlock(64, 64, 16).cuda()
    torch.manual_seed(23)
    v4_block = MODULE.ThinkBlock(64, 64, 16).cuda()

    def routed_output(block, stable):
        m_in = block.moe_norm(inputs)
        weights = torch.softmax(block.gate(m_in), dim=-1)
        all_out = torch.stack([expert(m_in) for expert in block.experts], dim=1)
        routed = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return block.moe_out_proj(block.moe_out_norm(routed)) if stable else routed

    v3_rms = routed_output(v3_block, stable=False).square().mean().sqrt()
    v4_rms = routed_output(v4_block, stable=True).square().mean().sqrt()
    torch.testing.assert_close(v4_rms / v3_rms, torch.ones((), device="cuda"), atol=0.025, rtol=0.025)


if __name__ == "__main__":
    test_situ_glu_matches_reference_equation_and_bound()
    test_situ_branch_is_parameter_matched_and_bias_free()
    test_situ_branch_preserves_v2_rng_position()
    test_situ_branch_matches_v2_initial_output_rms()
    test_situ_branch_forward_and_backward_are_finite()
    test_stable_moe_is_parameter_matched_and_seed_paired()
    test_stable_moe_outer_projection_has_matched_rms_and_finite_gradients()
    test_stable_moe_initial_routed_rms_matches_v3()
