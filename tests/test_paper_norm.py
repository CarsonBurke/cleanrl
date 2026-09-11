"""CUDA-only paper equations and attention invariants; run through mlq.

The reference uses tensor equations and parameters, never production forwards.
Native/compiled parity is covered separately by test_paper_norm_host.py.
"""

import math

import pytest
import torch
from torch.nn import functional as F

from cleanrl.shared.host_actor import SITU_GLU_MEAN_SQUARE
from cleanrl.shared.paper_norm import PaperNormBlock, make_paper_norm_trunk
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
METHODS = ("peri", "span", "hybrid", "siamese")


def _rms(x, weight):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight


def _attention(x, weights, prefix, heads, normalize_qkv):
    # Keep token-major heads and use einsum, unlike production's transposed
    # head-major matmul. All leading axes represent independent observations.
    head_dim = x.shape[-1] // heads
    q, k, v = (
        F.linear(x, weights[prefix + name + "_proj.weight"]).unflatten(-1, (heads, head_dim))
        for name in ("q", "k", "v")
    )
    if normalize_qkv:
        q, k, v = (
            _rms(value, weights[prefix + name + "_norm.weight"])
            for name, value in zip(("q", "k", "v"), (q, k, v))
        )
    logits = torch.einsum("...thd,...shd->...hts", q, k) / math.sqrt(head_dim)
    probabilities = torch.exp(logits - logits.amax(-1, keepdim=True))
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    values = torch.einsum("...hts,...shd->...thd", probabilities, v).flatten(-2)
    return F.linear(values, weights[prefix + "out_proj.weight"])


def _ffn(x, weights, prefix):
    gate = F.linear(x, weights[prefix + "gate.weight"])
    up = F.linear(x, weights[prefix + "up.weight"])
    product = (4 * torch.tanh(gate / 4)) * gate.sigmoid() * (25 * torch.tanh(up / 25))
    return F.linear(product, weights[prefix + "down.weight"])


def _equations(x, weights, method, width, tokens, blocks, heads):
    h = F.linear(x, weights["in_proj.weight"], weights["in_proj.bias"]).unflatten(-1, (tokens, width))
    post, identity = h, h
    for index in range(blocks):
        prefix = f"blocks.{index}."
        ga, gf = (weights[prefix + name + "_gate"].sigmoid() for name in ("attn", "ffn"))
        if method == "siamese":
            for sublayer, gate in (("attn", ga), ("ffn", gf)):
                normalized_identity = _rms(identity, weights[prefix + sublayer + "_pre_norm.weight"])
                merged = _rms(post + normalized_identity, weights[prefix + sublayer + "_input_norm.weight"])
                update = gate * (
                    _attention(merged, weights, prefix + "attn.", heads, True)
                    if sublayer == "attn" else _ffn(merged, weights, prefix + "ffn.")
                )
                post = _rms(post + update / math.sqrt(index + 1), weights[prefix + sublayer + "_post_norm.weight"])
                identity = identity + update
        elif method == "span":
            saved = h
            first_input = _rms(h, weights["initial_norm.weight"]) if index == 0 else h
            y = _rms(saved + ga * _attention(first_input, weights, prefix + "attn.", heads, False),
                     weights[prefix + "attn_post_norm.weight"])
            h = _rms(saved + gf * _ffn(y, weights, prefix + "ffn."), weights[prefix + "ffn_post_norm.weight"])
        elif method == "hybrid":
            y = h + ga * _attention(h, weights, prefix + "attn.", heads, True)
            z = _rms(y, weights[prefix + "ffn_pre_norm.weight"])
            h = z + gf * _ffn(z, weights, prefix + "ffn.")
        else:
            a = _attention(_rms(h, weights[prefix + "attn_pre_norm.weight"]), weights,
                           prefix + "attn.", heads, False)
            y = h + ga * _rms(a, weights[prefix + "attn_post_norm.weight"])
            f = _ffn(_rms(y, weights[prefix + "ffn_pre_norm.weight"]), weights, prefix + "ffn.")
            h = y + gf * _rms(f, weights[prefix + "ffn_post_norm.weight"])
    if method == "siamese":
        h = post + _rms(identity, weights["final_norm.weight"])
    return h.flatten(-2) / math.sqrt(tokens * width)


@pytest.mark.parametrize("method", METHODS)
def test_outputs_and_all_gradients_match_independent_equations(method):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(831)
    trunk = make_paper_norm_trunk(7, width=8, n_tokens=4, n_blocks=3, n_heads=2, method=method).cuda().double()
    with torch.no_grad():
        # Nonconstant, nonunit gains/gates expose ordering errors which unit
        # initialization could hide, including gating inside an RMSNorm.
        for index, (name, parameter) in enumerate(trunk.named_parameters()):
            if "norm.weight" in name:
                parameter.copy_(torch.linspace(0.45, 1.7, parameter.numel(), device="cuda", dtype=torch.float64)
                                + index * 0.003)
            elif name.endswith("_gate"):
                parameter.copy_(torch.linspace(-2.1, 0.9, 8, device="cuda", dtype=torch.float64) + index * 0.01)
    x = torch.randn(2, 3, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = dict(trunk.named_parameters())
    actual = trunk(x)
    expected = _equations(reference_x, weights, method, 8, 4, 3, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (x, *weights.values()), upstream)
    reference_gradients = torch.autograd.grad(expected, (reference_x, *weights.values()), upstream)
    # No allow_unused: every registered parameter must participate. This also
    # catches a detached Siamese stream, an omitted norm, or discarded gates.
    for name, actual_gradient, reference_gradient in zip(("input", *weights), actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=2e-8, atol=2e-9, msg=name)


@pytest.mark.parametrize("method", ["peri", "hybrid"])
def test_attention_is_sample_independent_token_equivariant_and_noncausal(method):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(832)
    block = make_paper_norm_trunk(7, width=8, n_heads=2, method=method).blocks[0]
    assert isinstance(block, PaperNormBlock)
    attn = block.attn.cuda().double()
    tokens = torch.randn(2, 3, 4, 8, device="cuda", dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.no_grad():
        original = attn(tokens)
        torch.testing.assert_close(attn(tokens[..., permutation, :]), original[..., permutation, :],
                                   rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(attn(tokens[1, 2]), original[1, 2], rtol=1e-11, atol=1e-12)
        changed = tokens.clone()
        changed[0, 0, -1] += torch.linspace(-2, 3, 8, device="cuda", dtype=torch.float64)
        result = attn(changed)
        torch.testing.assert_close(result[1], original[1], rtol=0, atol=0)
        torch.testing.assert_close(result[0, 1:], original[0, 1:], rtol=0, atol=0)
        # A last-token intervention must reach the first token: neither a
        # causal mask nor a pointwise faux-attention implementation can pass.
        assert not torch.allclose(result[0, 0, 0], original[0, 0, 0], rtol=1e-5, atol=1e-6)


def test_methods_share_seeded_weights_and_depth_scaled_initialization():
    trunks = {}
    for method in METHODS:
        torch.manual_seed(833)
        trunks[method] = make_paper_norm_trunk(7, width=8, n_tokens=4, n_blocks=3, n_heads=2, method=method).cuda()
    common = {name: value for name, value in trunks["peri"].named_parameters() if "norm" not in name}
    for trunk in trunks.values():
        parameters = dict(trunk.named_parameters())
        for name, expected in common.items():
            torch.testing.assert_close(parameters[name], expected, rtol=0, atol=0)
    # Orthogonal matrix Gram invariants verify the actual std scaling rather
    # than comparing to another invocation of the same initializer.
    trunk = trunks["peri"]
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    projection = trunk.in_proj.weight
    torch.testing.assert_close(projection.T @ projection, torch.eye(7, device="cuda") * (32 / 7),
                               rtol=2e-5, atol=2e-6)
    for block in trunk.blocks:
        for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj):
            torch.testing.assert_close(projection.weight.T @ projection.weight, torch.eye(8, device="cuda"),
                                       rtol=2e-5, atol=2e-6)
        output = block.attn.out_proj.weight
        torch.testing.assert_close(output.T @ output, torch.eye(8, device="cuda") / 3, rtol=2e-5, atol=2e-6)
        down = block.ffn.down.weight
        hidden = block.ffn.hidden_dim
        gain_squared = 0.5 * 8 / (hidden * SITU_GLU_MEAN_SQUARE * 3)
        torch.testing.assert_close(down.T @ down, torch.eye(hidden, device="cuda") * gain_squared,
                                   rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("method", METHODS)
def test_zero_observation_gradients_fit_fp32_adam_second_moment(method):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(53)
    with torch.device("cuda"):
        trunk = make_paper_norm_trunk(17, method=method)
        observations = torch.zeros(2, 17, requires_grad=True)
        features = trunk(observations)
        probe = torch.linspace(-0.1, 0.1, trunk.out_dim)
    gradients = torch.autograd.grad((features * probe).sum(),
                                    (observations, *trunk.parameters()))
    for gradient in gradients:
        # Repeated RMS at an all-zero embedding can yield finite 1e26
        # gradients whose squared Adam accumulator immediately overflows.
        assert torch.isfinite(gradient.square()).all()
