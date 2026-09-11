"""Paper token policies agree with fused native rollouts through real updates."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from cleanrl.shared.host_graph import BetaHeadGraph, HostGraphActor, make_host_mirror
from cleanrl.shared.paper_norm import PaperNormBlock, make_paper_norm_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
METHODS = ("peri", "span", "hybrid", "siamese")


@pytest.fixture(autouse=True)
def isolated_compiler():
    torch.compiler.reset()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    try:
        yield
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rows,width,tokens,heads", [(16, 32, 4, 2), (5, 21, 3, 3)])
def test_compiled_policy_and_joint_logprob_match_native_after_refresh(method, rows, width, tokens, heads):
    # The second geometry has 15 token rows, 45 head rows, head_dim=7 and
    # FFN hidden_dim=14: row, column, and normalization vector tails all matter.
    torch.manual_seed(53)
    with torch.device("cuda"):
        trunk = make_paper_norm_trunk(17, width, tokens, 3, heads, method=method)
        head = nn.Linear(trunk.out_dim, 12)
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.zeros_(head.bias)
        policy = nn.Sequential(trunk, head)
    mirror = make_host_mirror(policy, rows, fused=True)
    assert isinstance(mirror, HostGraphActor)
    with pytest.raises(ValueError, match="requires the fused host mirror"):
        make_host_mirror(policy, rows, fused=False)

    observations = np.random.default_rng(53).standard_normal((rows, 17)).astype(np.float32)
    observations[0] = 0
    observations[1] *= 1e-7
    observations[2] *= 8.0
    device_observations = torch.as_tensor(observations, device="cuda")
    # Fixed native actions compare densities, not unrelated CUDA/NumPy RNGs.
    native = torch.linspace(1e-6, 1.0 - 1e-6, rows * 6, device="cuda").view(rows, 6)
    beta_head = BetaHeadGraph(rows, 6, np.float32(-1.0), np.float32(1.0))

    def statistics(obs, actions):
        logits = policy(obs)
        alpha, beta = (1.0 + F.softplus(logits)).chunk(2, dim=-1)
        logprob = Beta(alpha, beta, validate_args=False).log_prob(actions).sum(-1)
        return logits, logprob

    compiled = torch.compile(statistics, fullgraph=True, backend="inductor")

    @torch.no_grad()
    def assert_outputs():
        logits, logprob = compiled(device_observations, native)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(logprob).all()
        actual = mirror(observations).copy()
        np.testing.assert_allclose(actual, logits.cpu().numpy(), rtol=2e-4, atol=2e-5)
        # Use the actual native Beta concentration op, then evaluate both
        # parameter sets with CUDA's density implementation to isolate rollout
        # error from differing lgamma/log implementations.
        alpha, beta = beta_head.concentration(actual)
        host_logprob = Beta(torch.as_tensor(alpha, device="cuda"),
                            torch.as_tensor(beta, device="cuda"),
                            validate_args=False).log_prob(native).sum(-1)
        torch.testing.assert_close(host_logprob, logprob, rtol=2e-4, atol=2e-4)
        return actual

    before = assert_outputs()
    optimizer = torch.optim.Adam(policy.parameters(), lr=9.6e-3, eps=1e-5, fused=True)
    advantages = torch.linspace(-1.0, 1.0, rows, device="cuda")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, logprob = compiled(device_observations, native)
        loss = -(advantages * logprob).mean()
        loss.backward()
        optimizer.step()
    mirror.refresh()
    updated = assert_outputs()
    assert np.max(np.abs(updated - before)) > 1e-4

    # Change ONLY learned norm gains; this catches missing refresh registrations
    # independently of linears and gates, including per-head Q/K/V RMS gains.
    with torch.no_grad():
        for index, norm in enumerate(module for module in trunk.modules() if isinstance(module, nn.RMSNorm)):
            norm.weight.copy_(torch.linspace(0.45, 1.8, norm.weight.numel(), device="cuda")
                              .roll(index % norm.weight.numel()))
    mirror.refresh()
    normalized = assert_outputs()
    assert np.max(np.abs(normalized - updated)) > 1e-4

    # Distinct channel patterns and depth offsets expose scalar gates, a gate
    # reused between sublayers, or Siamese's delta being gated a second time.
    with torch.no_grad():
        for index, block in enumerate(trunk.blocks):
            assert isinstance(block, PaperNormBlock)
            block.attn_gate.copy_(torch.linspace(-2.7, 1.6, width, device="cuda").roll(3 * index))
            block.ffn_gate.copy_(torch.linspace(2.1, -1.8, width, device="cuda").roll(5 * index))
    mirror.refresh()
    gated = assert_outputs()
    assert np.max(np.abs(gated - normalized)) > 1e-4

    # Force peaked but finite attention scores. Unstabilized exp(scores) would
    # overflow here; Q/K norm gains must be scaled instead in normalized modes.
    with torch.no_grad():
        for block in trunk.blocks:
            assert isinstance(block, PaperNormBlock)
            if method in ("hybrid", "siamese"):
                assert isinstance(block.attn.q_norm, nn.RMSNorm)
                assert isinstance(block.attn.k_norm, nn.RMSNorm)
                block.attn.q_norm.weight.mul_(16.0)
                block.attn.k_norm.weight.mul_(16.0)
            else:
                block.attn.q_proj.weight.mul_(16.0)
                block.attn.k_proj.weight.mul_(16.0)
    mirror.refresh()
    peaked = assert_outputs()

    # Changing one environment must never change another environment's logits.
    # A shared [batch*tokens, width] softmax or incorrect sample stride fails.
    changed = observations.copy()
    changed[2] = np.linspace(-3.0, 4.0, 17, dtype=np.float32)
    actual = mirror(changed).copy()
    untouched = np.arange(rows) != 2
    np.testing.assert_array_equal(actual[untouched], peaked[untouched])
    with torch.no_grad():
        expected, _ = compiled(torch.as_tensor(changed, device="cuda"), native)
    np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=2e-4, atol=2e-5)
    assert np.max(np.abs(actual[2] - peaked[2])) > 1e-4
