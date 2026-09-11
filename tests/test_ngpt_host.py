"""nGPT CUDA policies and native Beta rollouts agree after projected updates."""

from typing import cast
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from cleanrl.shared.host_graph import BetaHeadGraph, HostGraphActor, make_host_mirror
from cleanrl.shared.ngpt import NGPTBlock, NGPTHead, NGPTTrunk, project_ngpt_weights
from test_ppo_normres_twohot import device

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def isolated_compiler(device):
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("rows,width", [(16, 64), (5, 21)])
def test_projected_policy_scales_and_beta_probabilities_match_native(rows, width):
    # The odd geometry exercises GEMM and norm tails; width 64 is the trainer.
    torch.manual_seed(71)
    with torch.device("cuda"):
        trunk = NGPTTrunk(17, width=width, n_blocks=3)
        head = NGPTHead(width, 12)
        policy = nn.Sequential(trunk, head)
    mirror = make_host_mirror(policy, rows)
    assert isinstance(mirror, HostGraphActor)
    with pytest.raises(ValueError, match="requires the fused host mirror"):
        make_host_mirror(policy, rows, fused=False)
    # A plain Linear would silently lose the learned nGPT readout scale.
    with pytest.raises(ValueError):
        make_host_mirror(nn.Sequential(trunk, nn.Linear(width, 12, bias=False).cuda()), rows)

    rng = np.random.default_rng(71)
    observations = rng.standard_normal((rows, 17)).astype(np.float32)
    observations[0] = 0
    observations[1] *= 1e-14
    observations[2] *= 8.0
    device_observations = torch.as_tensor(observations, device="cuda")
    native = torch.linspace(0.02, 0.98, rows * 6, device="cuda").view(rows, 6)
    beta_head = BetaHeadGraph(rows, 6, np.float32(-1), np.float32(1))

    def statistics(obs, actions):
        logits = policy(obs)
        alpha, beta = (1.0 + F.softplus(logits)).chunk(2, dim=-1)
        logprob = Beta(alpha, beta, validate_args=False).log_prob(actions).sum(-1)
        return logits, logprob

    compiled = torch.compile(statistics, fullgraph=True, backend="inductor")

    @torch.no_grad()
    def assert_outputs():
        logits, fixed_logprob = compiled(device_observations, native)
        actual = mirror(observations).copy()
        np.testing.assert_allclose(actual, logits.cpu().numpy(), rtol=2e-4, atol=2e-5,
                                   equal_nan=False)
        alpha, beta = beta_head.concentration(actual)
        host_distribution = Beta(torch.as_tensor(alpha, device="cuda"),
                                 torch.as_tensor(beta, device="cuda"), validate_args=False)
        torch.testing.assert_close(host_distribution.log_prob(native).sum(-1),
                                   fixed_logprob, rtol=2e-4, atol=2e-4)
        # Exercise the actual host concentration/sampling/rescaling path. Compare
        # densities at identical native actions, not unrelated CPU/CUDA RNGs.
        draw = rng.beta(alpha, beta)
        draw[0, :2] = (0.0, 1.0)  # Include sampler clipping boundaries.
        sampled_native, _ = beta_head.rescale(draw)
        sampled = torch.as_tensor(sampled_native, device="cuda")
        _, expected_logprob = compiled(device_observations, sampled)
        actual_logprob = host_distribution.log_prob(sampled).sum(-1)
        torch.testing.assert_close(actual_logprob, expected_logprob, rtol=2e-4, atol=2e-4)
        # PPO's no-update likelihood ratio must remain unity, even at boundaries.
        torch.testing.assert_close((actual_logprob - expected_logprob).exp(),
                                   torch.ones_like(expected_logprob), rtol=0, atol=5e-4)

    assert_outputs()
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    advantages = torch.linspace(-1.0, 1.0, rows, device="cuda")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, logprob = compiled(device_observations, native)
        (-(advantages * logprob).mean()).backward()
        optimizer.step()
        project_ngpt_weights(policy)
        mirror.refresh()
        assert_outputs()

    # Only the homogeneous column changes. Missing bias refresh specifically
    # corrupts the zero-observation direction, even if all regular GEMMs agree.
    with torch.no_grad():
        trunk.in_proj.weight[:, -1].copy_(torch.linspace(-2.0, 1.0, width, device="cuda"))
        project_ngpt_weights(policy)
    mirror.refresh()
    assert_outputs()

    # Signed, unequal up/gate gains expose swapped suv rows or folded gains in
    # the wrong order. Large channels also distinguish true SwiGLU from SiTU.
    with torch.no_grad():
        for index, block in enumerate(trunk.blocks):
            block = cast(NGPTBlock, block)
            block.suv[0].copy_(torch.linspace(-8.0, 12.0, 4 * width, device="cuda").roll(index * 7))
            block.suv[1].copy_(torch.linspace(9.0, -6.0, 4 * width, device="cuda").roll(index * 11))
    mirror.refresh()
    assert_outputs()

    # These are effective interpolation coefficients, not raw alpha values:
    # both negative alpha and magnitudes above one must survive without caps.
    with torch.no_grad():
        for index, block in enumerate(trunk.blocks):
            block = cast(NGPTBlock, block)
            coefficients = torch.linspace(-1.8, 2.2, width, device="cuda").roll(index * 3)
            block.alpha.copy_(coefficients * (block.base_scale / 0.05))
    mirror.refresh()
    assert_outputs()

    # Head gains are signed, unconstrained, and divided by the base scale.
    with torch.no_grad():
        head.scale.copy_(torch.linspace(-3.0, 5.0, 12, device="cuda") * head.base_scale)
    mirror.refresh()
    assert_outputs()


def test_uncapped_negative_tail_does_not_create_a_spurious_spherical_branch():
    with torch.device("cuda"):
        trunk = NGPTTrunk(17, width=64, n_blocks=1)
        head = NGPTHead(64, 12)
        policy = nn.Sequential(trunk, head)
    with torch.no_grad():
        # Every observation embeds to the same positive direction. All gate
        # channels are deeply negative, so the true SwiGLU branch is zero.
        trunk.in_proj.weight.zero_()
        trunk.in_proj.weight[:, -1].fill_(1.0)
        block = cast(NGPTBlock, trunk.blocks[0])
        block.gate.weight.fill_(-0.125)
        block.up.weight.fill_(0.125)
        block.suv[1].fill_(1e30)
        project_ngpt_weights(policy)
    observations = np.zeros((16, 17), dtype=np.float32)
    mirror = make_host_mirror(policy, 16)
    with torch.no_grad():
        expected = policy(torch.as_tensor(observations, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(observations), expected, rtol=2e-5, atol=2e-6)
