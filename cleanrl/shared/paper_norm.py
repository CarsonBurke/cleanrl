"""Observation-token PPO adaptations, not reproductions of full LLM recipes.

SpanNorm (https://arxiv.org/abs/2601.22580, Eqs. 5-8) spans the FFN skip
across the whole block. HybridNorm (https://arxiv.org/abs/2503.04598,
Algorithm 1) normalizes per-head Q/K/V and uses N(y) + FFN(N(y)).
SiameseNorm (https://arxiv.org/abs/2602.08064, Algorithm 1 and auxiliary
mechanisms) shares each update across identity/post-norm streams, with
normalized merged input and block-depth scaling only on the post stream.

All methods use affine RMSNorm, existing SiTU-GLU FFNs, learned channel gates,
and the same depth-scaled output initialization, including the Peri control.
Attention is noncausal over learned tokens from ONE observation, never over
samples, environments, or history. There are no positions, masks, or dropout.
"""

import math

import torch
from torch import nn

from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch


def _norm(width, active):
    return nn.RMSNorm(width, eps=1e-5, elementwise_affine=True) if active else nn.Identity()


class PaperAttention(nn.Module):
    """Explicit multi-head token attention; Q/K/V norms act within each head."""

    def __init__(self, width, n_heads, *, method="peri"):
        super().__init__()
        self.width = int(width)
        self.n_heads = int(n_heads)
        if self.width < 1 or self.n_heads < 1 or self.width % self.n_heads:
            raise ValueError("attention width must be positive and divisible by n_heads")
        self.head_dim = self.width // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.out_proj = nn.Linear(self.width, self.width, bias=False)
        qkv_norm = method in ("hybrid", "siamese")
        self.q_norm = _norm(self.head_dim, qkv_norm)
        self.k_norm = _norm(self.head_dim, qkv_norm)
        self.v_norm = _norm(self.head_dim, qkv_norm)

    def forward(self, x):
        shape = (*x.shape[:-1], self.n_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).reshape(shape)).transpose(-3, -2)
        k = self.k_norm(self.k_proj(x).reshape(shape)).transpose(-3, -2)
        v = self.v_norm(self.v_proj(x).reshape(shape)).transpose(-3, -2)
        probabilities = (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attended = torch.matmul(probabilities, v).transpose(-3, -2).contiguous()
        return self.out_proj(attended.view(x.shape))


class PaperNormBlock(nn.Module):
    """Shared sublayers; only normalization sites used by the method own gains."""

    def __init__(self, width, n_heads, *, method="peri"):
        super().__init__()
        self.attn = PaperAttention(width, n_heads, method=method)
        self.ffn = SiTUGLUBranch(width, width)
        self.attn_gate = nn.Parameter(torch.full((width,), -1.5))
        self.ffn_gate = nn.Parameter(torch.full((width,), -1.5))
        self.attn_pre_norm = _norm(width, method in ("peri", "siamese"))
        self.attn_post_norm = _norm(width, method in ("peri", "span", "siamese"))
        self.ffn_pre_norm = _norm(width, method in ("peri", "hybrid", "siamese"))
        self.ffn_post_norm = _norm(width, method in ("peri", "span", "siamese"))
        self.attn_input_norm = _norm(width, method == "siamese")
        self.ffn_input_norm = _norm(width, method == "siamese")


class PaperNormTrunk(nn.Module):
    """Project observations to tokens, apply the selected topology, flatten.

    Gates multiply sublayer outputs (after the output norm for Peri), not inputs.
    Span's initial norm affects the first attention input but NOT its skip.
    Hybrid normalizes the FFN identity path as well as the FFN input.
    Siamese sends the SAME gated update to both streams, scaling only the post
    stream by 1/sqrt(block_index+1); its readout is p + final_norm(u).
    Other methods have no final norm. All readouts scale by out_dim**-0.5.
    Use the factory to apply the common training initialization.
    """

    def __init__(self, in_dim, width=32, n_tokens=4, n_blocks=3, n_heads=2, *, method="peri"):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_tokens = int(n_tokens)
        self.n_blocks = int(n_blocks)
        self.n_heads = int(n_heads)
        self.method = method
        if min(self.in_dim, self.width, self.n_tokens, self.n_blocks, self.n_heads) < 1:
            raise ValueError("PaperNormTrunk dimensions and block/head counts must be positive")
        if self.width % self.n_heads:
            raise ValueError("width must be divisible by n_heads")
        if self.method not in ("peri", "span", "hybrid", "siamese"):
            raise ValueError("method must be 'peri', 'span', 'hybrid', or 'siamese'")
        self.out_dim = self.n_tokens * self.width
        self.output_scale = self.out_dim ** -0.5
        self.in_proj = nn.Linear(self.in_dim, self.out_dim)
        self.blocks = nn.ModuleList(
            PaperNormBlock(self.width, self.n_heads, method=method) for _ in range(self.n_blocks)
        )
        self.initial_norm = _norm(self.width, method == "span")
        self.final_norm = _norm(self.width, method == "siamese")

    def forward(self, x):
        h = self.in_proj(x).reshape(*x.shape[:-1], self.n_tokens, self.width)
        if self.method == "siamese":
            p, u = h, h
            for index, block in enumerate(self.blocks):
                assert isinstance(block, PaperNormBlock)
                scale = (index + 1) ** -0.5
                merged = block.attn_input_norm(p + block.attn_pre_norm(u))
                delta = block.attn_gate.sigmoid() * block.attn(merged)
                p = block.attn_post_norm(p + scale * delta)
                u = u + delta
                merged = block.ffn_input_norm(p + block.ffn_pre_norm(u))
                delta = block.ffn_gate.sigmoid() * block.ffn(merged)
                p = block.ffn_post_norm(p + scale * delta)
                u = u + delta
            h = p + self.final_norm(u)
        elif self.method == "span":
            for index, block in enumerate(self.blocks):
                assert isinstance(block, PaperNormBlock)
                saved = h
                attn_input = self.initial_norm(h) if index == 0 else h
                y = block.attn_post_norm(saved + block.attn_gate.sigmoid() * block.attn(attn_input))
                h = block.ffn_post_norm(saved + block.ffn_gate.sigmoid() * block.ffn(y))
        elif self.method == "hybrid":
            for block in self.blocks:
                assert isinstance(block, PaperNormBlock)
                y = h + block.attn_gate.sigmoid() * block.attn(h)
                z = block.ffn_pre_norm(y)
                h = z + block.ffn_gate.sigmoid() * block.ffn(z)
        else:
            for block in self.blocks:
                assert isinstance(block, PaperNormBlock)
                branch = block.attn_post_norm(block.attn(block.attn_pre_norm(h)))
                h = h + block.attn_gate.sigmoid() * branch
                branch = block.ffn_post_norm(block.ffn(block.ffn_pre_norm(h)))
                h = h + block.ffn_gate.sigmoid() * branch
        return h.flatten(start_dim=-2) * self.output_scale


def make_paper_norm_trunk(in_dim, width=32, n_tokens=4, n_blocks=3, n_heads=2, *, method="peri"):
    """Identical seeded shared weights across methods; output std scales 1/sqrt(L).

    RMSNorm and gate initialization are constant and consume no RNG. Linear
    construction and initialization order are independent of normalization mode.
    """
    trunk = PaperNormTrunk(in_dim, width, n_tokens, n_blocks, n_heads, method=method)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.out_dim / trunk.in_dim))
    # Learned token offsets keep a zero normalized observation away from the
    # all-zero fixed point of repeated RMS norms (whose Jacobian scales as
    # eps**-0.5 at each site). Shared across methods, not an input special case.
    nn.init.normal_(trunk.in_proj.bias, std=1.0)
    output_gain = trunk.n_blocks ** -0.5
    for block in trunk.blocks:
        assert isinstance(block, PaperNormBlock)
        for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj):
            nn.init.orthogonal_(projection.weight, 1.0)
        nn.init.orthogonal_(block.attn.out_proj.weight, output_gain)
        init_situglu_branch(block.ffn, target_out_var=0.5)
        with torch.no_grad():
            block.ffn.down.weight.mul_(output_gain)
    return trunk
