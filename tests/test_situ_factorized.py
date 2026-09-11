"""Fixed-capacity SiTU direction/amplitude coordinates and refreshed rollout parity."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.shared.host_actor import SiTUGLUBranch
from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.situ_factorized import make_situ_factorized_trunk

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
    # Preserve the original trainer's critic-first order and Beta head gains.
    for name, outputs, gain in (("critic", 1, 1.0), ("actor", 12, 0.01)):
        trunk = factory(17, 64, 3)
        head = nn.Linear(64, outputs)
        nn.init.orthogonal_(head.weight, gain)
        nn.init.zeros_(head.bias)
        networks[name] = nn.Sequential(trunk, head)
    return networks


def _capacity(module):
    return (
        sum(parameter.numel() for parameter in module.parameters()),
        sum(isinstance(child, nn.Linear) for child in module.modules()),
        sum(isinstance(child, nn.RMSNorm) for child in module.modules()),
        sum(isinstance(child, SiTUGLUBranch) for child in module.modules()),
    )


def _equation_reference(trunk, observations):
    # Do not call trunk/block/norm forward, situ_glu, or F.normalize: the
    # compiled learner and native lowering must satisfy an independent equation.
    def rms(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)

    hidden = F.linear(observations, trunk.in_proj.weight, trunk.in_proj.bias)
    for block, raw_amplitude in zip(trunk.blocks, trunk.block_gates):
        normalized = rms(hidden)
        gate = F.linear(normalized, block.gate.weight)
        up = F.linear(normalized, block.up.weight)
        product = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))
        row_norm = block.down.weight.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        effective_weight = F.softplus(raw_amplitude)[:, None] * block.down.weight / row_norm
        hidden = hidden + F.linear(product, effective_weight)
    return rms(hidden) * trunk.output_scale


def test_exact_capacity_and_function_preserving_initialization():
    with torch.device("cuda"):
        torch.manual_seed(1)
        original = _halfcheetah_networks(
            lambda *args: make_norm_residual_trunk(*args, placement="pre", norm_kind="rms", activation="stiglu")
        )
        torch.manual_seed(1)
        factorized = _halfcheetah_networks(make_situ_factorized_trunk)

    assert _capacity(original) == _capacity(factorized) == (53069, 22, 8, 6)
    original_parameters = dict(original.named_parameters())
    factorized_parameters = dict(factorized.named_parameters())
    assert factorized_parameters.keys() == original_parameters.keys()
    for name, actual in factorized_parameters.items():
        if ".block_gates." not in name:
            # Includes both heads, so extra factory RNG draws cannot slip by.
            torch.testing.assert_close(actual, original_parameters[name], rtol=0, atol=0)

    observations = torch.randn(13, 17, device="cuda")
    observations[0] = 0
    observations[1] *= 1e-7
    observations[2] *= 9.0
    with torch.no_grad():
        for name in ("critic", "actor"):
            old_trunk, new_trunk = original[name][0], factorized[name][0]
            assert _capacity(old_trunk) == _capacity(new_trunk) == (26112, 10, 4, 3)
            for old_block, old_gate, new_block, new_gate in zip(
                old_trunk.blocks, old_trunk.block_gates, new_trunk.blocks, new_trunk.block_gates
            ):
                old_effective = old_gate.sigmoid()[:, None] * old_block.down.weight
                new_effective = F.softplus(new_gate)[:, None] * F.normalize(new_block.down.weight, dim=1)
                torch.testing.assert_close(new_effective, old_effective, rtol=2e-6, atol=2e-7)
            old_forward = torch.compile(original[name], fullgraph=True, backend="inductor")
            new_forward = torch.compile(factorized[name], fullgraph=True, backend="inductor")
            expected = F.linear(
                _equation_reference(new_trunk, observations), factorized[name][1].weight, factorized[name][1].bias
            )
            torch.testing.assert_close(new_forward(observations), old_forward(observations), rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(new_forward(observations), expected, rtol=2e-4, atol=2e-6)


def test_positive_down_row_scaling_preserves_compiled_function():
    torch.manual_seed(7)
    with torch.device("cuda"):
        trunk = make_situ_factorized_trunk(11, 43, 3)
    observations = torch.randn(13, 11, device="cuda")
    compiled = torch.compile(trunk, fullgraph=True, backend="inductor")
    with torch.no_grad():
        for index, raw_amplitude in enumerate(trunk.block_gates):
            raw_amplitude.copy_(torch.linspace(-2.4, 1.7, 43, device="cuda").roll(7 * index))
        before = compiled(observations).clone()
        torch.testing.assert_close(before, _equation_reference(trunk, observations), rtol=2e-4, atol=2e-5)
        for index, block in enumerate(trunk.blocks):
            # Independent row factors, all safely outside the zero/epsilon branch.
            factors = torch.logspace(-3, 3, 43, device="cuda").roll(5 * index)
            block.down.weight.mul_(factors[:, None])
        after = compiled(observations)
        torch.testing.assert_close(after, before, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(after, _equation_reference(trunk, observations), rtol=2e-4, atol=2e-5)


def test_compiled_down_gradients_are_tangent_to_nonzero_rows():
    torch.manual_seed(11)
    with torch.device("cuda"):
        trunk = make_situ_factorized_trunk(11, 43, 3)
        policy = nn.Sequential(trunk, nn.Linear(43, 12))
    with torch.no_grad():
        for index, raw_amplitude in enumerate(trunk.block_gates):
            raw_amplitude.copy_(torch.linspace(-1.8, 1.4, 43, device="cuda").roll(9 * index))
        for block in trunk.blocks:
            block.down.weight.mul_(torch.logspace(-1, 1, 43, device="cuda")[:, None])
    observations = torch.randn(19, 11, device="cuda")
    targets = torch.randn(19, 12, device="cuda")
    compiled = torch.compile(policy, fullgraph=True, backend="inductor")
    (compiled(observations) - targets).square().mean().backward()

    for block in trunk.blocks:
        weight, gradient = block.down.weight.detach(), block.down.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        weight_norm = weight.norm(dim=1)
        gradient_norm = gradient.norm(dim=1)
        assert (weight_norm > 1e-8).all()
        assert (gradient_norm > 1e-10).all()
        # Scale-free radial derivative: a detached/missing row denominator would
        # retain a nonzero projection even when forward values initially match.
        radial_fraction = (weight * gradient).sum(1) / (weight_norm * gradient_norm)
        torch.testing.assert_close(radial_fraction, torch.zeros_like(radial_fraction), rtol=0, atol=3e-5)


def test_native_refresh_matches_equation_after_adam_and_coordinate_changes():
    torch.manual_seed(19)
    with torch.device("cuda"):
        # Odd trunk and branch widths exercise native linear/reduction tails.
        trunk = make_situ_factorized_trunk(11, 43, 3)
        policy = nn.Sequential(trunk, nn.Linear(43, 12))
    with torch.no_grad():
        for index, raw_amplitude in enumerate(trunk.block_gates):
            raw_amplitude.copy_(torch.linspace(-2.4, 1.7, 43, device="cuda").roll(7 * index))

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
        expected = F.linear(_equation_reference(trunk, device_observations), policy[1].weight, policy[1].bias)
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction, expected, rtol=2e-4, atol=2e-5)
        actual = mirror(observations).copy()
        np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=2e-4, atol=2e-5)
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

    # Force a large row-norm change rather than relying on small Adam steps to
    # expose a refresh that reuses stale row denominators.
    with torch.no_grad():
        for index, block in enumerate(trunk.blocks):
            factors = torch.logspace(-2, 2, 43, device="cuda").roll(11 * index)
            block.down.weight.mul_(factors[:, None])
    mirror.refresh()
    after_rescaling = assert_outputs()
    np.testing.assert_allclose(after_rescaling, after_updates, rtol=2e-4, atol=2e-5)

    # Gates must be transformed and refreshed per channel, not once at creation.
    with torch.no_grad():
        for index, raw_amplitude in enumerate(trunk.block_gates):
            raw_amplitude.copy_(torch.linspace(2.2, -1.9, 43, device="cuda").roll(11 * index))
    mirror.refresh()
    after_gate_refresh = assert_outputs()
    assert np.max(np.abs(after_gate_refresh - after_rescaling)) > 1e-4


def test_compiled_beta_forward_kl_matches_fp64_distribution_reference():
    from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_factorized_v7 import beta_forward_kl

    # Include skew, large concentration, and small movement at high concentration;
    # FP32 lgamma cancellation can swamp the last case's genuine small KL.
    old_alpha = torch.tensor(
        [[1.01, 80.0], [10000.0, 20000.0], [10000.0, 2.0]], device="cuda"
    )
    old_beta = torch.tensor(
        [[80.0, 1.01], [10000.0, 2.0], [10000.0, 4.0]], device="cuda"
    )
    alpha = torch.tensor(
        [[1.2, 65.0], [11000.0, 18000.0], [10000.5, 2.0001]], device="cuda"
    )
    beta = torch.tensor(
        [[65.0, 1.2], [9000.0, 3.0], [9999.5, 3.9999]], device="cuda"
    )
    old = torch.distributions.Beta(old_alpha.double(), old_beta.double(), validate_args=False)
    new = torch.distributions.Beta(alpha.double(), beta.double(), validate_args=False)
    expected = torch.distributions.kl_divergence(old, new).sum(-1)
    compiled = torch.compile(beta_forward_kl, fullgraph=True, backend="inductor")
    actual = compiled(old_alpha, old_beta, alpha, beta)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-9)
    torch.testing.assert_close(
        compiled(old_alpha, old_beta, old_alpha, old_beta), torch.zeros_like(expected), rtol=0, atol=2e-9
    )


def test_policy_geometry_separates_mean_and_concentration_movements():
    from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_factorized_v7 import policy_geometry

    # A real linear actor exposes chosen Beta parameters through its logits.
    # Double precision makes the interventions pure rather than approximately
    # preserving a concentration/mean through FP32 inverse-softplus roundoff.
    with torch.device("cuda"):
        actor = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        actor.weight.copy_(torch.eye(4, device="cuda", dtype=torch.float64))
    old_mean = torch.tensor([[0.25, 0.5], [0.65, 0.4], [0.35, 0.7]], device="cuda", dtype=torch.float64)
    old_concentration = torch.tensor(
        [[24.0, 30.0], [40.0, 18.0], [32.0, 28.0]], device="cuda", dtype=torch.float64
    )
    old_alpha = old_mean * old_concentration
    old_beta = (1 - old_mean) * old_concentration
    compiled = torch.compile(
        lambda observations: policy_geometry(actor, observations, old_alpha, old_beta),
        fullgraph=True, backend="inductor",
    )
    old = torch.distributions.Beta(old_alpha, old_beta, validate_args=False)
    for mean, concentration, moving_index, stationary_index in (
        (old_mean + 0.08, old_concentration, 6, 7),
        (old_mean, old_concentration * 1.7, 7, 6),
    ):
        alpha, beta = mean * concentration, (1 - mean) * concentration
        positive_logits = torch.cat((alpha, beta), dim=-1) - 1.0
        observations = positive_logits + torch.log(-torch.expm1(-positive_logits))
        geometry = compiled(observations)
        expected_kl = torch.distributions.kl_divergence(
            old, torch.distributions.Beta(alpha, beta, validate_args=False)
        ).sum(-1)
        torch.testing.assert_close(geometry[0], expected_kl.mean(), rtol=2e-7, atol=2e-10)
        torch.testing.assert_close(geometry[moving_index], geometry[0], rtol=2e-7, atol=2e-10)
        torch.testing.assert_close(
            geometry[stationary_index], torch.zeros_like(geometry[0]), rtol=0, atol=2e-10
        )
        standardized_mean_move = ((mean - old_mean).square() / (old_mean * (1 - old_mean)
                                  / (old_concentration + 1))).mean().sqrt()
        log_concentration_move = (concentration / old_concentration).log().square().mean().sqrt()
        torch.testing.assert_close(geometry[4], standardized_mean_move, rtol=2e-7, atol=2e-10)
        torch.testing.assert_close(geometry[5], log_concentration_move, rtol=2e-7, atol=2e-10)
