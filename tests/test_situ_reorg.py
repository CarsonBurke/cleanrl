"""SiTU norm relocation preserves capacity, initialization, and native rollout parity."""

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.shared.host_actor import SITU_GLU_MEAN_SQUARE, SiTUGLUBranch
from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.situ_reorg import make_situ_reorg_trunk

VARIANTS = ("baseline", "gate_only", "up_only", "product_norm", "branch_norm", "blend", "anchor")
pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_compiler():
    # Each trainer is a separate process; cases must not share Dynamo's limit.
    torch.compiler.reset()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    try:
        yield
    finally:
        torch.compiler.reset()


def _halfcheetah_networks(factory):
    networks = nn.ModuleDict()
    # Match the winner's construction order, including its Beta actor head.
    for name, outputs, gain in (("critic", 1, 1.0), ("actor", 12, 0.01)):
        trunk = factory(17, 64, 3)
        head = nn.Linear(64, outputs)
        nn.init.orthogonal_(head.weight, gain)
        nn.init.zeros_(head.bias)
        networks[name] = nn.Sequential(trunk, head)
    return networks


def _capacity(module):
    return (
        sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        sum(isinstance(child, nn.Linear) for child in module.modules()),
        sum(isinstance(child, nn.RMSNorm) for child in module.modules()),
        sum(isinstance(child, SiTUGLUBranch) for child in module.modules()),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_capacity_initialization_and_baseline_outputs(variant):
    with torch.device("cuda"):
        torch.manual_seed(1)
        original = _halfcheetah_networks(
            lambda *args: make_norm_residual_trunk(*args, placement="pre", norm_kind="rms", activation="stiglu")
        )
        torch.manual_seed(1)
        reorganized = _halfcheetah_networks(lambda *args: make_situ_reorg_trunk(*args, variant=variant))

    assert _capacity(original) == (53069, 22, 8, 6)
    assert _capacity(reorganized) == _capacity(original)
    for name in ("critic", "actor"):
        assert _capacity(reorganized[name][0]) == _capacity(original[name][0]) == (26112, 10, 4, 3)

    # Include both heads: matching trunk weights alone would miss extra RNG
    # draws which change the rest of the agent's initialization.
    original_parameters = list(original.named_parameters())
    reorganized_parameters = list(reorganized.named_parameters())
    assert [name for name, _ in reorganized_parameters] == [name for name, _ in original_parameters]
    for (_, actual), (_, expected) in zip(reorganized_parameters, original_parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if variant == "baseline":
        observations = torch.randn(13, 17, device="cuda")
        observations[0] = 0
        observations[1] *= 1e-7
        with torch.no_grad():
            for name in ("critic", "actor"):
                old_forward = torch.compile(original[name], fullgraph=True, backend="inductor")
                new_forward = torch.compile(reorganized[name], fullgraph=True, backend="inductor")
                torch.testing.assert_close(new_forward(observations), old_forward(observations), rtol=0, atol=0)


def _equation_reference(trunk, observations, variant):
    # Deliberately avoid trunk/block/norm forward methods and situ_glu so that
    # learner and native lowering cannot agree on the same misplaced norm.
    def rms(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)

    def linear(value, layer):
        return F.linear(value, layer.weight, layer.bias)

    initial = linear(observations, trunk.in_proj)
    hidden = initial
    for block, logits in zip(trunk.blocks, trunk.block_gates):
        gate_input = rms(hidden) if variant in ("baseline", "gate_only", "blend", "anchor") else hidden
        up_input = rms(hidden) if variant in ("baseline", "up_only", "blend", "anchor") else hidden
        gate = linear(gate_input, block.gate)
        up = linear(up_input, block.up)
        product = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))
        if variant == "product_norm":
            product = math.sqrt(SITU_GLU_MEAN_SQUARE) * rms(product)
        branch = linear(product, block.down)
        if variant == "branch_norm":
            branch = math.sqrt(0.5) * rms(branch)
        weight = torch.sigmoid(logits)
        if variant == "blend":
            hidden = hidden + weight * (branch - hidden)
        elif variant == "anchor":
            hidden = initial + weight * branch
        else:
            hidden = hidden + weight * branch
    return rms(hidden) / math.sqrt(trunk.width)


@pytest.mark.parametrize("variant", VARIANTS)
def test_compiled_equations_and_native_rollout_refresh(variant):
    torch.manual_seed(19)
    with torch.device("cuda"):
        # Width 43 also produces an odd branch width (29), covering both native
        # linear row tails and product-normalization reduction tails.
        trunk = make_situ_reorg_trunk(11, 43, 3, variant=variant)
        policy = nn.Sequential(trunk, nn.Linear(43, 12))
    with torch.no_grad():
        for index, logits in enumerate(trunk.block_gates):
            logits.copy_(torch.linspace(-2.4, 1.7, 43, device="cuda").roll(7 * index))

    observations = np.random.default_rng(19).standard_normal((13, 11)).astype(np.float32)
    observations[0] = 0
    observations[1] *= 1e-7
    observations[2] *= 9.0
    observations[3] *= 0.03
    device_observations = torch.as_tensor(observations, device="cuda")
    compiled = torch.compile(policy, fullgraph=True, backend="inductor")
    mirror = make_host_mirror(policy, len(observations), fused=True)
    assert isinstance(mirror, HostGraphActor)

    @torch.no_grad()
    def assert_outputs():
        prediction = compiled(device_observations)
        expected = F.linear(_equation_reference(trunk, device_observations, variant),
                            policy[1].weight, policy[1].bias)
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction, expected, rtol=2e-4, atol=2e-5)
        actual = mirror(observations).copy()
        np.testing.assert_allclose(actual, prediction.cpu().numpy(), rtol=2e-4, atol=2e-5)
        return actual

    before_updates = assert_outputs()
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    target = torch.linspace(-1.0, 1.0, 12, device="cuda").expand(13, -1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (compiled(device_observations) - target).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    mirror.refresh()
    after_updates = assert_outputs()
    assert np.max(np.abs(after_updates - before_updates)) > 1e-4

    # Changing gates alone distinguishes a working gate refresh from refreshing
    # only the linear weights. It also catches scalar/per-block gate lowering.
    with torch.no_grad():
        for index, logits in enumerate(trunk.block_gates):
            logits.copy_(torch.linspace(2.2, -1.9, 43, device="cuda").roll(11 * index))
    mirror.refresh()
    after_gate_refresh = assert_outputs()
    assert np.max(np.abs(after_gate_refresh - after_updates)) > 1e-4
