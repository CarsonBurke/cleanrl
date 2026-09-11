"""CUDA-only nGPT geometry and donor equations; queue execution through mlq."""

import math

from typing import cast
import pytest
import torch
from torch import nn

from cleanrl.shared.ngpt import NGPTBlock, NGPTHead, NGPTTrunk, project_ngpt_weights
from test_ppo_normres_twohot import device

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _unit(x, dim=-1):
    # Independent equation: no production normalization helper or forwards.
    return x / x.square().sum(dim, keepdim=True).sqrt().clamp_min(1e-12)


def _block_equation(h, weights, prefix, width):
    h = _unit(h)
    up = h @ weights[prefix + "up.weight"].T
    gate = h @ weights[prefix + "gate.weight"].T
    up = up * weights[prefix + "suv"][0] * math.sqrt(width)
    gate = gate * weights[prefix + "suv"][1] * math.sqrt(width)
    branch = _unit((up * gate * gate.sigmoid()) @ weights[prefix + "down.weight"].T)
    step = (weights[prefix + "alpha"] * (0.05 / (width ** -0.5))).abs()
    return _unit((1 - step) * h + step * branch)


def _equations(x, weights, width, blocks):
    # Homogeneous projection as an affine equation instead of concatenation.
    projection = weights["0.in_proj.weight"]
    h = _unit(x @ projection[:, :-1].T + projection[:, -1])
    for index in range(blocks):
        h = _block_equation(h, weights, f"0.blocks.{index}.", width)
    return (h @ weights["1.weight"].T) * (weights["1.scale"] / (width ** -0.5))


def _assert_unit_matrices(model):
    for name, parameter in model.named_parameters():
        if name.endswith("weight"):
            dim = 0 if name.endswith("down.weight") else 1
            norms = parameter.norm(dim=dim)
            torch.testing.assert_close(norms, torch.ones_like(norms), rtol=2e-6, atol=2e-7, msg=name)


def test_projection_preserves_geometry_gains_and_adam_moments_after_updates(device):
    torch.manual_seed(913)
    with torch.device("cuda"):
        model = nn.Sequential(NGPTTrunk(5, width=8, n_blocks=2), NGPTHead(8, 3))
        unrelated = nn.Linear(3, 2)
        container = nn.ModuleDict({"ngpt": model, "unrelated": unrelated})
        x = torch.randn(17, 5)
        target = torch.randn(17, 3)
    with torch.no_grad():
        for block in cast(NGPTTrunk, model[0]).blocks:
            block = cast(NGPTBlock, block)
            block.alpha.copy_(torch.linspace(-8, 10, 8, device="cuda"))
            block.suv.copy_(torch.linspace(-2, 3, 64, device="cuda").reshape(2, 32))
        cast(NGPTHead, model[1]).scale.copy_(torch.tensor([-0.7, 0.0, 1.4], device="cuda"))
    _assert_unit_matrices(model)
    unrelated_before = {name: value.detach().clone() for name, value in unrelated.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    # Compile the real projection path: graph breaks from tensor reads or
    # data-dependent traversal must fail rather than silently executing eagerly.
    project = torch.compile(project_ngpt_weights, fullgraph=True)
    forward = torch.compile(model, fullgraph=True)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (forward(x) - target).square().mean().backward()
        optimizer.step()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        moments = {
            parameter: {name: value.clone() for name, value in state.items()}
            for parameter, state in optimizer.state.items()
        }
        project(container)
        _assert_unit_matrices(model)
        for name, parameter in model.named_parameters():
            if name.endswith("weight"):
                dim = 0 if name.endswith("down.weight") else 1
                torch.testing.assert_close(parameter, _unit(before[name], dim), rtol=2e-6, atol=2e-7, msg=name)
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0, msg=name)
        for parameter, state in optimizer.state.items():
            for name, value in state.items():
                torch.testing.assert_close(value, moments[parameter][name], rtol=0, atol=0)
    for name, parameter in unrelated.named_parameters():
        torch.testing.assert_close(parameter, unrelated_before[name], rtol=0, atol=0)


def test_outputs_and_all_gradients_match_independent_donor_equations(device):
    torch.manual_seed(914)
    with torch.device("cuda"):
        model = nn.Sequential(NGPTTrunk(5, width=8, n_blocks=2), NGPTHead(8, 3)).double()
        x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        for index, block in enumerate(cast(NGPTTrunk, model[0]).blocks):
            block = cast(NGPTBlock, block)
            # Both signs and effective steps above one expose abs/sigmoid/clamp
            # substitutions. Large, signed suv exposes capped SiTU substitutes.
            block.alpha.copy_(torch.linspace(-12, 14, 8, device="cuda", dtype=torch.float64) + index)
            block.suv.copy_(torch.linspace(-7, 9, 64, device="cuda", dtype=torch.float64).reshape(2, 32))
        cast(NGPTHead, model[1]).scale.copy_(torch.tensor([-0.7, 0.0, 1.4], device="cuda", dtype=torch.float64))
    reference_x = x.detach().clone().requires_grad_()
    weights = dict(model.named_parameters())
    actual = model(x)
    expected = _equations(reference_x, weights, 8, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (x, *weights.values()), upstream)
    reference_gradients = torch.autograd.grad(expected, (reference_x, *weights.values()), upstream)
    for name, actual_gradient, reference_gradient in zip(("input", *weights), actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=2e-8, atol=2e-9, msg=name)


def test_block_normalizes_nonunit_inputs_before_branch_and_residual(device):
    torch.manual_seed(915)
    with torch.device("cuda"):
        block = NGPTBlock(8).double()
        x = 7 * torch.randn(4, 8, dtype=torch.float64)
    with torch.no_grad():
        block.alpha.copy_(torch.linspace(-12, 14, 8, device="cuda", dtype=torch.float64))
    actual = block(x)
    expected = _block_equation(x, dict(block.named_parameters()), "", 8)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(actual, block(x / 19), rtol=2e-10, atol=2e-11)


def test_zero_observation_has_unit_features_and_adam_safe_gradients(device):
    torch.manual_seed(916)
    with torch.device("cuda"):
        model = nn.Sequential(NGPTTrunk(17), NGPTHead(64, 3))
        observations = torch.zeros(2, 17, requires_grad=True)
        probe = torch.tensor([-0.2, 0.1, 0.3])
    features = model[0](observations)
    torch.testing.assert_close(features.norm(dim=-1), torch.ones(2, device="cuda"), rtol=2e-6, atol=2e-7)
    outputs = model[1](features)
    assert torch.isfinite(outputs).all()
    gradients = torch.autograd.grad((outputs * probe).sum(), (observations, *model.parameters()))
    for gradient in gradients:
        # Finite gradients alone miss an exploding squared Adam accumulator.
        assert torch.isfinite(gradient.square()).all()
    assert gradients[0].abs().max() > 0
    # The homogeneous-coordinate column must itself remain learnable.
    assert gradients[1][:, -1].abs().max() > 0


def test_head_scale_can_learn_through_zero_and_remain_negative_after_projection(device):
    torch.manual_seed(917)
    with torch.device("cuda"):
        head = NGPTHead(8, 3)
    with torch.no_grad():
        head.scale.zero_()
    # Every diagonal linear output is one, guaranteeing nonzero scale gradients
    # even though every output starts at zero. abs/exp gains cannot satisfy this.
    x = head.weight.detach().clone()
    optimizer = torch.optim.Adam([head.scale], lr=0.02)
    head(x).diagonal().sum().backward()
    torch.testing.assert_close(head.scale.grad, torch.full_like(head.scale, math.sqrt(8)), rtol=2e-6, atol=2e-7)
    optimizer.step()
    project_ngpt_weights(head)
    assert (head.scale < 0).all()
    torch.testing.assert_close(head(x).diagonal(), head.scale / head.base_scale, rtol=2e-6, atol=2e-7)
