"""Normalized residual policies must agree between CUDA learning and host rollout."""

import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture
def isolated_compiler():
    # Architecture cases share Python code objects, but real trainers are
    # separate processes. Do not consume each other's Dynamo specialization cap.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.usefixtures("isolated_compiler")
@pytest.mark.parametrize("placement", ["pre", "post"])
@pytest.mark.parametrize("norm_kind", ["layer", "rms"])
@pytest.mark.parametrize("activation", ["lrelusq", "stiglu"])
@pytest.mark.parametrize("branch_input_scale", [1.0, 0.125])
def test_normalized_rollout_matches_learner_after_updates(placement, norm_kind, activation, branch_input_scale):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    # Odd width exercises native row tails as well as normalization reductions.
    trunk = make_norm_residual_trunk(11, 43, 3, placement=placement,
                                     norm_kind=norm_kind, activation=activation,
                                     branch_input_scale=branch_input_scale)
    policy = nn.Sequential(trunk, nn.Linear(43, 12)).cuda()
    mirror = make_host_mirror(policy, 13)
    assert isinstance(mirror, HostGraphActor)
    x = np.random.default_rng(1).standard_normal((13, 11)).astype(np.float32)
    x[0] = 0
    x[1] *= 1e-7  # epsilon-dominated normalization must not produce NaNs.
    device_x = torch.as_tensor(x, device="cuda")
    compiled = torch.compile(policy, fullgraph=True, backend="inductor")
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    target = torch.linspace(-1, 1, 12, device="cuda").expand(13, -1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = compiled(device_x)
        loss = (prediction - target).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    with torch.no_grad():
        expected = policy(device_x).cpu().numpy()
    mirror.refresh()
    np.testing.assert_allclose(mirror(x), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("placement", ["pre", "post"])
@pytest.mark.parametrize("norm_kind", ["layer", "rms"])
def test_normalized_trunk_matches_additive_residual_definition(placement, norm_kind):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    trunk = make_norm_residual_trunk(17, 64, placement=placement, norm_kind=norm_kind).cuda()
    x = torch.randn(16, 17, device="cuda")

    def normalize(value):
        centered = value - value.mean(-1, keepdim=True) if norm_kind == "layer" else value
        return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)

    with torch.no_grad():
        h = trunk.in_proj(x)
        if placement == "post":
            h = normalize(h)
        for block, gate in zip(trunk.blocks, trunk.block_gates):
            branch_input = normalize(h) if placement == "pre" else h
            h = h + gate.sigmoid() * block(branch_input)
            if placement == "post":
                h = normalize(h)
        if placement == "pre":
            h = normalize(h)
        torch.testing.assert_close(trunk(x), h / 8, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("activation", ["lrelusq", "stiglu"])
def test_parameter_scale_preserves_initial_policy_function(activation):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    ordinary = make_norm_residual_trunk(17, 64, placement="post", activation=activation).cuda()
    torch.manual_seed(1)
    scaled = make_norm_residual_trunk(17, 64, placement="post", activation=activation,
                                     branch_input_scale=0.125).cuda()
    x = torch.randn(16, 17, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(scaled(x), ordinary(x), rtol=1e-5, atol=1e-6)
