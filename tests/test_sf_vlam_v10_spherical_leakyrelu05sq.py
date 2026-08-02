import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "sf_vlam_v9_for_spherical_test",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v10_spherical_leakyrelu05sq",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v10_spherical_leakyrelu05sq.py",
)


def test_spherical_linear_projects_requested_weight_axis():
    row_layer = MODULE.spherical_linear(64, 43, norm_dim=1).cuda()
    column_layer = MODULE.spherical_linear(43, 64, norm_dim=0).cuda()
    torch.testing.assert_close(
        row_layer.weight.norm(dim=1),
        torch.ones(43, device="cuda"),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        column_layer.weight.norm(dim=0),
        torch.ones(43, device="cuda"),
        atol=2e-6,
        rtol=2e-6,
    )
    assert row_layer.bias is None
    assert column_layer.bias is None


def test_unit_state_and_unit_rows_hard_bound_squared_activation():
    layer = MODULE.spherical_linear(64, 64, norm_dim=1).cuda()
    # Include the tight Cauchy-Schwarz case by aligning one input with one row.
    inputs = torch.cat(
        [
            layer.weight[:1].detach(),
            MODULE.unit_sphere(torch.randn(1023, 64, device="cuda")),
        ]
    )
    preactivations = layer(inputs)
    activations = MODULE.LeakyReLUSquared()(preactivations)
    assert preactivations.abs().max() <= 1.0 + 2e-6
    assert activations.max() <= 1.0 + 4e-6
    torch.testing.assert_close(
        preactivations[0, 0], torch.tensor(1.0, device="cuda"), atol=2e-6, rtol=2e-6
    )


def test_branch_and_block_outputs_remain_on_unit_sphere():
    branch = MODULE.SphericalLeakySquaredBranch(64).cuda()
    inputs = MODULE.unit_sphere(torch.randn(256, 64, device="cuda"))
    branch_output = branch(inputs)
    torch.testing.assert_close(
        branch_output.norm(dim=-1),
        torch.ones(256, device="cuda"),
        atol=2e-5,
        rtol=2e-5,
    )

    block = MODULE.ThinkBlock(128, 64, 16).cuda()
    x0 = MODULE.unit_sphere(torch.randn(256, 64, device="cuda"))
    cat_feats = torch.cat(
        [x0, MODULE.unit_sphere(torch.randn(256, 64, device="cuda"))], dim=-1
    )
    block_output = block(cat_feats, x0)
    torch.testing.assert_close(
        block_output.norm(dim=-1),
        torch.ones(256, device="cuda"),
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        block.dense_alpha,
        torch.full((64,), 0.05, device="cuda"),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        block.moe_alpha,
        torch.full((64,), 0.05, device="cuda"),
        atol=0.0,
        rtol=0.0,
    )


def test_projection_retraction_restores_optimizer_drift():
    trunk = MODULE.ThinkTrunk(17, 64, 1, 2).cuda()
    layers = [
        module for module in trunk.modules() if isinstance(module, MODULE.SphericalLinear)
    ]
    optimizer = torch.optim.Adam(trunk.parameters(), lr=3e-4)
    loss = trunk(torch.randn(128, 17, device="cuda")).square().sum(dim=-1).mean()
    # The fixed output radius makes the above loss constant; use a directional loss.
    loss = loss + trunk(torch.randn(128, 17, device="cuda"))[:, 0].mean()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.allclose(
            layer.weight.norm(dim=layer.norm_dim),
            torch.ones_like(layer.weight.norm(dim=layer.norm_dim)),
            atol=2e-6,
            rtol=2e-6,
        )
        for layer in layers
    )
    for layer in layers:
        layer.project_()
        torch.testing.assert_close(
            layer.weight.norm(dim=layer.norm_dim),
            torch.ones_like(layer.weight.norm(dim=layer.norm_dim)),
            atol=2e-6,
            rtol=2e-6,
        )


def test_trunk_preserves_rng_and_nearly_matches_parameter_budget():
    torch.manual_seed(7)
    base = BASE.ThinkTrunk(17, 64, 3, 16)
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    spherical = MODULE.ThinkTrunk(17, 64, 3, 16)
    actual_rng_state = torch.get_rng_state()

    torch.testing.assert_close(actual_rng_state, expected_rng_state)
    base_count = sum(parameter.numel() for parameter in base.parameters())
    spherical_count = sum(parameter.numel() for parameter in spherical.parameters())
    assert spherical_count == base_count - 6_512
    assert abs(spherical_count - base_count) / base_count < 0.02


def test_actual_compiled_trunk_path_is_finite_and_head_scale_matched():
    trunk = MODULE.ThinkTrunk(17, 64, 3, 16).cuda()
    compiled = MODULE.CompiledModule(
        trunk, mode="reduce-overhead", cudagraphs=False
    )
    inputs = torch.randn(128, 17, device="cuda", requires_grad=True)
    output = compiled(inputs)
    output[:, 0].mean().backward()

    expected_norm = torch.full((128,), (2.0 * 64) ** 0.5, device="cuda")
    torch.testing.assert_close(
        output.norm(dim=-1), expected_norm, atol=3e-4, rtol=3e-5
    )
    rms = output.square().mean().sqrt()
    torch.testing.assert_close(
        rms, torch.tensor(2.0**0.5, device="cuda"), atol=3e-5, rtol=3e-5
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trunk.parameters()
    )


def test_scalar_lambda_is_the_default():
    assert MODULE.Args().per_dim_lambda is False


if __name__ == "__main__":
    test_spherical_linear_projects_requested_weight_axis()
    test_unit_state_and_unit_rows_hard_bound_squared_activation()
    test_branch_and_block_outputs_remain_on_unit_sphere()
    test_projection_retraction_restores_optimizer_drift()
    test_trunk_preserves_rng_and_nearly_matches_parameter_budget()
    test_actual_compiled_trunk_path_is_finite_and_head_scale_matched()
    test_scalar_lambda_is_the_default()
