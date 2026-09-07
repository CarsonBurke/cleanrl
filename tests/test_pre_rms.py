"""Pre-RMS policy variants preserve initialization and fused CUDA/host behavior."""

import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.pre_rms import make_pre_rms_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_compiler():
    # Trainers compile one architecture per process; parametrized tests must not
    # exhaust the shared Python code objects' Dynamo specialization allowance.
    torch.compiler.reset()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    yield
    torch.compiler.reset()


def test_same_seed_baseline_preserves_existing_pre_rms_policy():
    torch.manual_seed(17)
    baseline = nn.Sequential(
        make_norm_residual_trunk(11, 64, 3, placement="pre", norm_kind="rms"),
        nn.Linear(64, 12),
    ).cuda()
    torch.manual_seed(17)
    candidate = nn.Sequential(make_pre_rms_trunk(11, 64, 3), nn.Linear(64, 12)).cuda()
    x = torch.randn(13, 11, device="cuda")
    x[0] = 0
    x[1] *= 1e-7
    with torch.no_grad():
        # Exact equality also defends RNG initialization order, including the
        # subsequently initialized policy head, not merely matching statistics.
        torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)


@pytest.mark.parametrize("norm_position", ["input", "preact", "postact", "branch"])
@pytest.mark.parametrize("final_norm", [True, False], ids=["normalized-readout", "radial-readout"])
def test_fused_rollout_tracks_compiled_updates_and_affine_gains(norm_position, final_norm):
    torch.manual_seed(23)
    # Ragged rows/width exercise native GEMM tails and RMS reduction tails.
    trunk = make_pre_rms_trunk(11, 43, 3, norm_position=norm_position,
                               learned_input_gain=True, final_norm=final_norm)
    policy = nn.Sequential(trunk, nn.Linear(43, 12)).cuda()
    mirror = make_host_mirror(policy, 13)
    assert isinstance(mirror, HostGraphActor)
    x = np.random.default_rng(23).standard_normal((13, 11)).astype(np.float32)
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

    # Check epsilon-dominated rows before training introduces nonzero biases.
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

    # Change only CUDA affine gains after an otherwise successful refresh. This
    # isolates stale/missing gain transfers from ordinary Linear weight refresh.
    with torch.no_grad():
        factors = torch.linspace(-0.75, 1.5, 43, device="cuda")
        for norm in trunk.block_norms:
            norm.weight.mul_(factors)
    changed = device_logits()
    assert np.max(np.abs(changed - trained)) > 1e-4
    mirror.refresh()
    assert_mirrored(changed)


def test_disabling_final_norm_retains_radius_and_matches_fused_rollout():
    x = np.zeros((3, 11), dtype=np.float32)
    x[0, 0] = np.sqrt(11)
    x[1] = 4 * x[0]
    device_x = torch.as_tensor(x, device="cuda")
    outputs = {}
    for final_norm in (True, False):
        torch.manual_seed(29)
        trunk = make_pre_rms_trunk(11, 43, 3, final_norm=final_norm)
        policy = nn.Sequential(trunk, nn.Linear(43, 12)).cuda()
        with torch.no_grad():
            # A zero residual branch leaves just the identity stream: removing
            # final RMS must expose its radius, without removing readout scaling.
            for block in trunk.blocks:
                block.lin2.weight.zero_()
                block.lin2.bias.zero_()
            policy[1].bias.zero_()
            outputs[final_norm] = trunk(device_x)
            logits = policy(device_x).cpu().numpy()
        mirror = make_host_mirror(policy, 3)
        assert isinstance(mirror, HostGraphActor)
        np.testing.assert_allclose(mirror(x), logits, rtol=2e-4, atol=2e-5, equal_nan=False)
        if not final_norm:
            np.testing.assert_allclose(logits[1], 4 * logits[0], rtol=1e-6, atol=1e-7)

    # Orthogonal projection gain sqrt(width/in_dim), followed by width**-.5,
    # sends this input direction to radius one. Final RMS erases the factor four.
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[False], dim=-1),
                               torch.tensor([1., 4., 0.], device="cuda"), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.linalg.vector_norm(outputs[True], dim=-1),
                               torch.tensor([1., 1., 0.], device="cuda"), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("norm_position", ["input", "preact", "postact", "branch"])
def test_branch_input_reparameterization_preserves_initial_function(norm_position):
    torch.manual_seed(31)
    ordinary = make_pre_rms_trunk(11, 64, 3, norm_position=norm_position,
                                  learned_input_gain=True, final_norm=False).cuda()
    torch.manual_seed(31)
    scaled = make_pre_rms_trunk(11, 64, 3, norm_position=norm_position,
                                learned_input_gain=True, final_norm=False,
                                branch_input_scale=0.125).cuda()
    x = torch.randn(13, 11, device="cuda")
    x[0] = 0
    x[1] *= 1e-7
    with torch.no_grad():
        # Raw output prevents final RMS from hiding an incorrect overall scale.
        torch.testing.assert_close(scaled(x), ordinary(x), rtol=1e-5, atol=1e-6)
