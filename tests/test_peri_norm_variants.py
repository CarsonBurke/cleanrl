"""Peri sphere/Derf branches preserve additive streams and native rollout statistics."""

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.norm_residual import Derf, make_norm_residual_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture
def isolated_compiler():
    # Real trainers run in separate processes, not one shared Dynamo cache.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _set_derf_parameters(trunk, phase):
    with torch.no_grad():
        for index, norm in enumerate((*trunk.block_norms, *trunk.block_output_norms)):
            norm.alpha.fill_(0.35 + 0.07 * index + 0.2 * phase)
            norm.shift.fill_(-0.25 + 0.08 * index - 0.15 * phase)
            norm.weight.copy_(torch.linspace(-0.7, 1.3, trunk.width, device="cuda") + 0.2 * phase)
            norm.bias.copy_(torch.linspace(-0.15, 0.2, trunk.width, device="cuda") + 0.05 * index + 0.1 * phase)


def _reference_norm(x, norm, kind):
    if kind == "sphere":
        return x / x.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return torch.erf(norm.alpha * x + norm.shift) * norm.weight + norm.bias


@pytest.mark.parametrize("norm_kind", ["sphere", "derf"])
@pytest.mark.parametrize("activation", ["lrelusq", "stiglu"])
def test_peri_matches_branch_only_additive_definition(norm_kind, activation):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(11)
    trunk = make_norm_residual_trunk(17, 43, placement="peri", norm_kind=norm_kind,
                                     activation=activation).cuda()
    if norm_kind == "derf":
        _set_derf_parameters(trunk, 0)
    x = torch.randn(16, 17, device="cuda")
    x[0] = 0
    x[1] *= 1e-14
    with torch.no_grad():
        for index, gate in enumerate(trunk.block_gates):
            gate.copy_(torch.linspace(-2, 1.5, trunk.width, device="cuda") + index * 0.25)
        h = trunk.in_proj(x)
        for block, gate, norm, output_norm in zip(
            trunk.blocks, trunk.block_gates, trunk.block_norms, trunk.block_output_norms
        ):
            branch = block(_reference_norm(h, norm, norm_kind))
            h = h + gate.sigmoid() * _reference_norm(branch, output_norm, norm_kind)
        # No input projection normalization, post-add projection, dense skip,
        # branch input multiplier, or final norm may enter this ablation.
        torch.testing.assert_close(trunk(x), h / math.sqrt(43), rtol=1e-5, atol=1e-6)


@pytest.mark.usefixtures("isolated_compiler")
@pytest.mark.parametrize("norm_kind", ["sphere", "derf"])
@pytest.mark.parametrize("activation", ["lrelusq", "stiglu"])
def test_fused_rollout_matches_compiled_policy_and_parameter_refresh(norm_kind, activation):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(13)
    # Odd dimensions exercise native vector tails and normalization reductions.
    trunk = make_norm_residual_trunk(11, 43, placement="peri", norm_kind=norm_kind,
                                     activation=activation)
    policy = nn.Sequential(trunk, nn.Linear(43, 12)).cuda()
    if norm_kind == "derf":
        _set_derf_parameters(trunk, 0)
    mirror = make_host_mirror(policy, 13)
    assert isinstance(mirror, HostGraphActor)
    x = np.random.default_rng(13).standard_normal((13, 11)).astype(np.float32)
    x[0] = 0
    x[1] *= 1e-14
    device_x = torch.as_tensor(x, device="cuda")
    compiled = torch.compile(policy, fullgraph=True, backend="inductor")
    native_actions = torch.linspace(0.1, 0.9, 6, device="cuda").expand(13, -1)

    def assert_rollout_statistics():
        with torch.no_grad():
            logits = compiled(device_x)
            host_logits = torch.as_tensor(mirror(x).copy(), device="cuda")
            torch.testing.assert_close(host_logits, logits, rtol=2e-4, atol=2e-5)
            alpha, beta = (F.softplus(logits) + 1).chunk(2, dim=-1)
            host_alpha, host_beta = (F.softplus(host_logits) + 1).chunk(2, dim=-1)
            # PPO consumes the joint old-policy likelihood, not only head logits.
            expected = torch.distributions.Beta(alpha, beta).log_prob(native_actions).sum(-1)
            actual = torch.distributions.Beta(host_alpha, host_beta).log_prob(native_actions).sum(-1)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    assert_rollout_statistics()
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    target = torch.linspace(-1, 1, 12, device="cuda").expand(13, -1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = compiled(device_x)
        (prediction - target).square().mean().backward()
        optimizer.step()
    mirror.refresh()
    assert_rollout_statistics()

    # Explicit changes prevent small optimizer deltas from hiding stale gates or
    # any of Derf's four learned fields, at either Peri normalization site.
    with torch.no_grad():
        for index, gate in enumerate(trunk.block_gates):
            gate.copy_(torch.linspace(-2, 1.5, trunk.width, device="cuda") + index * 0.25)
    if norm_kind == "derf":
        _set_derf_parameters(trunk, 1)
    mirror.refresh()
    assert_rollout_statistics()


def test_sphere_norm_zero_tiny_and_unit_rows():
    trunk = make_norm_residual_trunk(3, 3, placement="peri", norm_kind="sphere").cuda()
    x = torch.tensor([[0, 0, 0], [3e-14, 4e-14, 0], [3, 4, 0]], device="cuda")
    expected = torch.tensor([[0, 0, 0], [0.03, 0.04, 0], [0.6, 0.8, 0]], device="cuda")
    for norm in (*trunk.block_norms, *trunk.block_output_norms):
        torch.testing.assert_close(norm(x), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("activation", ["lrelusq", "stiglu"])
def test_sphere_peri_is_invariant_to_positive_branch_output_rescaling(activation):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(17)
    trunk = make_norm_residual_trunk(17, 43, placement="peri", norm_kind="sphere",
                                     activation=activation).cuda()
    x = torch.randn(16, 17, device="cuda")
    with torch.no_grad():
        # Nonzero offsets exercise normalization of the complete affine branch.
        for block in trunk.blocks:
            projection = block.down if activation == "stiglu" else block.lin2
            assert isinstance(projection, nn.Linear)
            if projection.bias is not None:
                projection.bias.copy_(torch.linspace(-1, 1, trunk.width, device="cuda"))
        expected = trunk(x)
        for index, block in enumerate(trunk.blocks):
            projection = block.down if activation == "stiglu" else block.lin2
            assert isinstance(projection, nn.Linear)
            factor = 2.0 ** (index + 1)
            projection.weight.mul_(factor)
            if projection.bias is not None:
                projection.bias.mul_(factor)
        torch.testing.assert_close(trunk(x), expected, rtol=1e-5, atol=2e-6)


def test_derf_elementwise_formula_and_learned_parameter_gradients():
    norm = Derf(5).cuda().double()
    with torch.no_grad():
        norm.alpha.fill_(-0.7)
        norm.shift.fill_(0.3)
        norm.weight.copy_(torch.tensor([-1.2, 0.5, 1.3, -0.4, 0.8], device="cuda"))
        norm.bias.copy_(torch.tensor([0.2, -0.1, 0.4, 0.1, -0.3], device="cuda"))
    x = torch.tensor([[0, 1e-14, -0.8, 1.5, 40], [-40, -1e-14, 0.6, -2, 0]],
                     dtype=torch.float64, device="cuda", requires_grad=True)
    upstream = torch.linspace(-0.9, 1.1, 10, dtype=torch.float64, device="cuda").reshape(2, 5)
    result = norm(x)
    z = norm.alpha * x + norm.shift
    torch.testing.assert_close(result, torch.erf(z) * norm.weight + norm.bias, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(result, (x, norm.alpha, norm.shift, norm.weight, norm.bias), upstream)
    with torch.no_grad():
        dz = upstream * norm.weight * (2 / math.sqrt(math.pi)) * torch.exp(-z.square())
        expected = (
            dz * norm.alpha,
            (dz * x).sum().reshape_as(norm.alpha),
            dz.sum().reshape_as(norm.shift),
            (upstream * torch.erf(z)).sum(0),
            upstream.sum(0),
        )
        for actual, reference in zip(gradients, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
