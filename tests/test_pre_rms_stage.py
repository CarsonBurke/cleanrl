"""Single-stage pre-RMS policies preserve their definition and fused rollout."""

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.pre_rms import make_pre_rms_trunk
from cleanrl.shared.pre_rms_stage import make_pre_rms_stage_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_compiler():
    # Real trainers compile one architecture per process, rather than sharing
    # Dynamo's code-object specialization allowance across parametrized cases.
    torch.compiler.reset()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("architecture", ["stage-residual", "stage-plain", "pair-plain"])
@pytest.mark.parametrize(
    "learned_input_gain,final_norm,branch_input_scale",
    [(False, True, 1.0), (False, False, 0.125), (True, True, 0.125), (True, False, 1.0)],
    ids=["fixed-normalized", "fixed-radial-scaled", "affine-normalized-scaled", "affine-radial"],
)
def test_fused_rollout_tracks_compiled_updates(
    architecture, learned_input_gain, final_norm, branch_input_scale
):
    torch.manual_seed(41)
    factory = make_pre_rms_trunk if architecture == "pair-plain" else make_pre_rms_stage_trunk
    n_blocks = 3 if architecture == "pair-plain" else 6
    # Ragged batch and hidden dimensions exercise native GEMM and RMS tails.
    trunk = factory(11, 43, n_blocks, residual=architecture == "stage-residual",
                    learned_input_gain=learned_input_gain, final_norm=final_norm,
                    branch_input_scale=branch_input_scale)
    policy = nn.Sequential(trunk, nn.Linear(43, 12)).cuda()
    mirror = make_host_mirror(policy, 13)
    assert isinstance(mirror, HostGraphActor)
    x = np.random.default_rng(41).standard_normal((13, 11)).astype(np.float32)
    x[0] = 0
    x[1] *= 1e-7
    device_x = torch.as_tensor(x, device="cuda")
    compiled = torch.compile(policy, fullgraph=True, backend="inductor")

    def device_logits():
        with torch.no_grad():
            return compiled(device_x).cpu().numpy()

    def assert_mirrored(expected):
        actual = mirror(x)
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5, equal_nan=False)

    # Exercise zero and epsilon-dominated rows before updates introduce biases.
    assert_mirrored(device_logits())
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    target = torch.linspace(-1, 1, 12, device="cuda").expand(13, -1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (compiled(device_x) - target).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    trained = device_logits()
    mirror.refresh()
    assert_mirrored(trained)

    if learned_input_gain:
        # Refresh only the norm gains after a successful whole-policy refresh:
        # a stale gain must fail even if Linear weights are copied correctly.
        with torch.no_grad():
            factors = torch.linspace(-0.75, 1.5, 43, device="cuda")
            for norm in trunk.block_norms:
                norm.weight.mul_(factors)
        changed = device_logits()
        assert np.max(np.abs(changed - trained)) > 1e-4
        mirror.refresh()
        assert_mirrored(changed)


@pytest.mark.parametrize("residual", [True, False], ids=["residual", "plain"])
@pytest.mark.parametrize("final_norm", [True, False], ids=["normalized-readout", "radial-readout"])
def test_single_stages_match_repeated_rms_linear_squared_activation(residual, final_norm):
    torch.manual_seed(43)
    width = 43
    branch_input_scale = 0.125
    trunk = make_pre_rms_stage_trunk(11, width, 6, residual=residual,
                                    learned_input_gain=True, final_norm=final_norm,
                                    branch_input_scale=branch_input_scale).cuda()
    x = torch.randn(13, 11, device="cuda")
    x[0] = 0
    x[1] *= 1e-7

    def rms(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)

    with torch.no_grad():
        # Nonuniform, signed gains and nonzero biases prevent scale cancellation
        # or the all-zero stream from concealing misplaced/missing operations.
        channels = torch.linspace(-0.5, 1.25, width, device="cuda")
        for index in range(6):
            trunk.block_norms[index].weight.copy_(channels + index * 0.05)
            trunk.blocks[index].lin.bias.copy_(channels * (index + 1) * 0.03)
            if residual:
                trunk.block_gates[index].copy_(channels - 2 + index * 0.1)

        h = F.linear(x, trunk.in_proj.weight, trunk.in_proj.bias)
        for index in range(6):
            # Do not invoke stage.forward, its activation, or a trunk norm:
            # CUDA and host could otherwise agree on the same wrong structure.
            u = rms(h) * trunk.block_norms[index].weight * branch_input_scale
            linear = trunk.blocks[index].lin
            z = F.linear(u, linear.weight, linear.bias)
            branch = torch.where(z >= 0, z, 0.5 * z).square() * math.sqrt(0.25 / 6.375)
            if residual:
                h = h + trunk.block_gates[index].sigmoid() * branch
            else:
                # Plain means a full replacement, not a residual or a gated
                # branch. Its observable values defend the absence of gates.
                h = branch
        expected = (rms(h) if final_norm else h) / math.sqrt(width)
        torch.testing.assert_close(trunk(x), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("final_norm", [True, False], ids=["normalized-readout", "radial-readout"])
def test_plain_pair_blocks_replace_the_stream_without_gating(final_norm):
    torch.manual_seed(47)
    width = 43
    branch_input_scale = 0.125
    trunk = make_pre_rms_trunk(11, width, 3, residual=False, learned_input_gain=True,
                               final_norm=final_norm, branch_input_scale=branch_input_scale).cuda()
    x = torch.randn(13, 11, device="cuda")
    x[0] = 0
    x[1] *= 1e-7

    def rms(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)

    with torch.no_grad():
        channels = torch.linspace(-0.5, 1.25, width, device="cuda")
        for index in range(3):
            trunk.block_norms[index].weight.copy_(channels + index * 0.1)
            trunk.blocks[index].lin1.bias.copy_(channels * 0.03)
            trunk.blocks[index].lin2.bias.copy_(channels * 0.07)
        h = F.linear(x, trunk.in_proj.weight, trunk.in_proj.bias)
        for index in range(3):
            block = trunk.blocks[index]
            u = rms(h) * trunk.block_norms[index].weight * branch_input_scale
            z = F.linear(u, block.lin1.weight, block.lin1.bias)
            activation = torch.where(z >= 0, z, 0.5 * z).square()
            h = F.linear(activation, block.lin2.weight, block.lin2.bias)
        expected = (rms(h) if final_norm else h) / math.sqrt(width)
        torch.testing.assert_close(trunk(x), expected, rtol=2e-5, atol=2e-6)
