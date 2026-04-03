"""
Dreamer4 World Model — Transformer Modules
============================================
AxialSpaceTimeTransformer with:
- Spatial attention (within each timestep)
- Temporal attention (causal, across timesteps)
- SwiGLU feedforward
- GRU temporal recurrence
- Rotary positional embeddings for time
- Multi-head RMSNorm on Q/K
- Value residual connections
- Attention gating (AlphaFold style)
- Belief attention (BeliefFormer)
- Attention residual connections (Kimi)
- Special token masking (agent tokens can see everything, modality cannot see agent)
"""
from __future__ import annotations
from typing import Callable
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor, cat, stack, zeros, ones, arange, randn
from torch.nn import Module, ModuleList, Linear, Parameter, Sequential, RMSNorm

from .utils import (
    exists, default, divisible_by, l2norm, softclamp, safe_stack,
)

LinearNoBias = partial(Linear, bias=False)

AttentionIntermediates = namedtuple('AttentionIntermediates', ('next_kv_cache', 'normed_inputs'))
TransformerIntermediates = namedtuple(
    'TransformerIntermediates',
    ('next_kv_cache', 'normed_time_inputs', 'normed_space_inputs', 'next_rnn_hiddens', 'layer_hiddens'),
    defaults=(None,)
)


# ──────────────────────────────────────────────
# Rotary Embeddings (1D, for time)
# ──────────────────────────────────────────────

class Rotary1D(Module):
    def __init__(self, dim_head, theta=10000.):
        super().__init__()
        inv_freq = 1.0 / (theta ** (arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, offset=0):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype
        t = torch.arange(seq_len, device=device).type(dtype) + offset
        freqs = torch.outer(t, self.inv_freq)
        return cat((freqs, freqs), dim=-1)


def apply_rotations(rotations, t):
    """
    rotations: (seq, dim_head) or (heads, seq, dim_head)
    t: (batch, heads, seq, dim_head)
    """
    seq_len, dtype = t.shape[2], t.dtype
    rot_seq = rotations.shape[-2] if rotations.ndim >= 2 else rotations.shape[0]

    if rot_seq > seq_len:
        rotations = rotations[..., -seq_len:, :]

    t = t.float()

    # handle GQA
    if rotations.ndim == 3 and rotations.shape[0] < t.shape[1]:
        heads = t.shape[1]
        rot_heads = rotations.shape[0]
        assert divisible_by(heads, rot_heads)
        groups = heads // rot_heads
        rotations = rotations.repeat_interleave(groups, dim=0)

    x1, x2 = t.chunk(2, dim=-1)
    rotated_half = cat((-x2, x1), dim=-1)
    rotated = t * rotations.cos() + rotated_half * rotations.sin()
    return rotated.type(dtype)


# ──────────────────────────────────────────────
# Multi-Head RMSNorm
# ──────────────────────────────────────────────

class MultiHeadRMSNorm(Module):
    def __init__(self, dim_head, heads=8):
        super().__init__()
        self.scale = dim_head ** 0.5
        self.gamma = Parameter(torch.zeros(heads, dim_head))

    def forward(self, x):
        # x: (b, h, n, d)
        normed = l2norm(x)
        scale = (self.gamma + 1.) * self.scale
        return normed * scale.unsqueeze(0).unsqueeze(2)  # (1, h, 1, d)


# ──────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        query_heads=None,
        heads=8,
        pre_rmsnorm=True,
        pre_context_rmsnorm=False,
        gate_values=True,
        rmsnorm_key=True,
        value_residual=True,
        belief_attn=True,
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.norm_context = RMSNorm(dim) if pre_context_rmsnorm else nn.Identity()

        query_heads = default(query_heads, heads)
        assert query_heads >= heads and divisible_by(query_heads, heads)

        self.heads = heads
        self.query_heads = query_heads
        self.dim_head = dim_head

        dim_q_inner = dim_head * query_heads
        dim_kv_inner = dim_head * heads

        self.to_q = LinearNoBias(dim, dim_q_inner)
        self.to_k = LinearNoBias(dim, dim_kv_inner)
        self.to_v = LinearNoBias(dim, dim_kv_inner)
        self.to_out = LinearNoBias(dim_q_inner, dim)

        # gating
        self.to_gates = None
        if gate_values:
            self.to_gates = Sequential(
                LinearNoBias(dim, query_heads),
                nn.Sigmoid()
            )

        # QK normalization
        self.k_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads=heads) if rmsnorm_key else nn.Identity()

        # value residual
        self.to_learned_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid()
        ) if value_residual else None

        # belief attention
        self.belief_attn = belief_attn

    def _split_heads(self, x, num_heads):
        b, n, _ = x.shape
        return x.reshape(b, n, num_heads, self.dim_head).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d)

    def forward(
        self,
        tokens,
        context=None,
        kv_cache=None,
        return_intermediates=False,
        rotary_pos_emb=None,
        residual_values=None,
        attend_fn=None,
    ):
        was_3d = tokens.ndim == 3
        if not was_3d:
            orig_shape = tokens.shape
            tokens = tokens.reshape(-1, *tokens.shape[-2:])

        normed = self.norm(tokens)
        q = self.to_q(normed)

        has_context = exists(context)
        if has_context:
            if context.ndim > 3:
                context = context.reshape(-1, *context.shape[-2:])
            context = self.norm_context(context)
        else:
            context = normed

        k, v = self.to_k(context), self.to_v(context)

        q = self._split_heads(q, self.query_heads)
        k = self._split_heads(k, self.heads)
        v = self._split_heads(v, self.heads)

        # value residual
        if exists(residual_values) and exists(self.to_learned_value_residual_mix):
            if residual_values.ndim == 4:
                # (b, n, h, d) -> (b, h, n, d)
                rv = residual_values.permute(0, 2, 1, 3) if residual_values.shape[-2] == self.heads else residual_values
            else:
                rv = residual_values.reshape(-1, *residual_values.shape[-3:])
                rv = rv.permute(0, 2, 1, 3)

            mix = self.to_learned_value_residual_mix(normed)
            mix = mix.permute(0, 2, 1).unsqueeze(-1)  # (b, h, n, 1)
            v = v.lerp(rv, mix)

        # K normalization
        k = self.k_heads_rmsnorm(k)

        # rotary
        if exists(rotary_pos_emb):
            q = apply_rotations(rotary_pos_emb, q)
            k = apply_rotations(rotary_pos_emb, k)

        # save for belief attention
        if self.belief_attn and not has_context:
            v_for_belief = v

        # KV caching
        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim=-2)
            v = cat((cv, v), dim=-2)

        # attend
        if exists(attend_fn):
            out = attend_fn(q, k, v)
        else:
            out = _naive_attend(q, k, v)

        # belief attention: remove parallel component
        if self.belief_attn and not has_context:
            v_normed = l2norm(v_for_belief)
            # handle GQA
            if v_normed.shape[1] < out.shape[1]:
                groups = out.shape[1] // v_normed.shape[1]
                v_normed = v_normed.repeat_interleave(groups, dim=1)
            parallel = (out * v_normed).sum(dim=-1, keepdim=True) * v_normed
            out = out - parallel

        # gate
        if exists(self.to_gates):
            gates = self.to_gates(normed)
            gates = gates.permute(0, 2, 1).unsqueeze(-1)  # (b, h, n, 1)
            out = out * gates

        out = self._merge_heads(out)
        out = self.to_out(out)

        if not was_3d:
            out = out.reshape(orig_shape)

        if not return_intermediates:
            return out

        return out, AttentionIntermediates(stack((k, v)), normed)


def _naive_attend(q, k, v, softclamp_value=50., causal=False, causal_block_size=1, mask=None):
    """Standard multi-head attention with GQA support."""
    scale = q.shape[-1] ** -0.5

    # GQA
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        q = q.reshape(q.shape[0], k.shape[1], groups, *q.shape[2:])
        sim = torch.einsum('b h g i d, b h j d -> b h g i j', q, k)
    else:
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    sim = sim * scale

    if exists(softclamp_value):
        sim = softclamp(sim, softclamp_value)

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        if causal_block_size > 1:
            bi = (i + causal_block_size - 1) // causal_block_size
            bj = (j + causal_block_size - 1) // causal_block_size
            causal_mask = torch.ones((bi, bj), dtype=torch.bool, device=sim.device).triu(bj - bi + 1)
            causal_mask = causal_mask.repeat_interleave(causal_block_size, dim=0).repeat_interleave(causal_block_size, dim=1)
            causal_mask = causal_mask[:i, :j]
        else:
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=sim.device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

    attn = sim.softmax(dim=-1)

    if groups > 1:
        out = torch.einsum('b h g i j, b h j d -> b h g i d', attn, v)
        out = out.reshape(out.shape[0], -1, *out.shape[3:])
    else:
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

    return out


def get_attend_fn(
    causal=False,
    softclamp_value=50.,
    num_special_tokens=0,
    special_seq_len=None,
    special_attend_only_itself=False,
    device=None,
    seq_len=None,
    k_seq_len=None,
):
    """Factory for spatial/temporal attend functions with masking."""
    mask = None

    if num_special_tokens > 0 and exists(seq_len):
        sl = default(special_seq_len, seq_len)
        q_seq = torch.arange(seq_len, device=device).unsqueeze(1)
        k_seq = torch.arange(default(k_seq_len, seq_len), device=device).unsqueeze(0)

        is_special_start = sl - num_special_tokens
        q_is_special = q_seq >= is_special_start
        k_is_special = k_seq >= is_special_start

        if special_attend_only_itself:
            mask = ~(q_is_special & ~k_is_special)
        else:
            mask = ~(~q_is_special & k_is_special)

    return partial(_naive_attend, causal=causal, softclamp_value=softclamp_value, mask=mask)


# ──────────────────────────────────────────────
# SwiGLU Feedforward
# ──────────────────────────────────────────────

class SwiGLUFeedforward(Module):
    def __init__(self, dim, expansion_factor=4, pre_rmsnorm=True):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        dim_inner = int(dim * expansion_factor * 2 / 3)
        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(self, x):
        x = self.norm(x)
        x, gates = self.proj_in(x).chunk(2, dim=-1)
        x = x * F.gelu(gates)
        return self.proj_out(x)


# ──────────────────────────────────────────────
# GRU Layer
# ──────────────────────────────────────────────

class GRULayer(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gru = nn.GRU(dim, dim_out, batch_first=True)

    def forward(self, x, prev_hiddens=None):
        x = self.norm(x)
        x, hiddens = self.gru(x, prev_hiddens)
        return x, hiddens


# ──────────────────────────────────────────────
# Attention Residual (Kimi-style)
# ──────────────────────────────────────────────

class AttentionResidual(Module):
    """Cross-attend to all previous layer hidden states."""

    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.attn = Attention(
            dim=dim, heads=heads, dim_head=dim_head,
            gate_values=False, value_residual=False, rmsnorm_key=True
        )

    def forward(self, x, hiddens: list[Tensor] | None = None, **kwargs):
        assert exists(hiddens), 'hiddens must be passed to AttentionResidual'
        context = stack(hiddens, dim=-2)  # (..., num_layers, dim)
        queries = x.unsqueeze(-2)  # (..., 1, dim)
        out = self.attn(queries, context=context)
        return out.squeeze(-2)


# ──────────────────────────────────────────────
# Axial Space-Time Transformer
# ──────────────────────────────────────────────

class AxialSpaceTimeTransformer(Module):
    """
    Processes tokens with shape (batch, time, space, dim).
    - Most blocks do spatial attention (within each timestep)
    - Every `time_block_every`-th block does temporal attention (causal, across timesteps)
    - GRU recurrence on temporal blocks
    - Attention residual connections across all layers
    """

    def __init__(
        self,
        dim,
        depth,
        attn_heads=8,
        attn_dim_head=64,
        attn_softclamp_value=50.,
        time_block_every=4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        num_special_spatial_tokens=1,
        special_attend_only_itself=False,
        final_norm=True,
        value_residual=True,
        rnn_time=True,
    ):
        super().__init__()
        assert depth >= time_block_every

        self.attn_softclamp_value = attn_softclamp_value
        self.special_attend_only_itself = special_attend_only_itself

        # time rotary
        self.time_rotary = Rotary1D(attn_dim_head)

        # value residual
        self.value_residual = value_residual
        if value_residual:
            dim_inner = attn_dim_head * attn_heads
            self.to_value_residual = nn.Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_inner),
            )
            self.vr_heads = attn_heads
            self.vr_dim_head = attn_dim_head

        self.rnn_time = rnn_time

        layers = []
        rnn_layers = []
        is_time = []
        rnn_attn_residuals = []
        attn_attn_residuals = []
        ff_attn_residuals = []

        ar_kwargs = dict(heads=4, dim_head=min(64, attn_dim_head))

        for i in range(depth):
            layer_index = i + 1
            is_time_block = divisible_by(layer_index, time_block_every)
            is_time.append(is_time_block)

            layers.append(ModuleList([
                Attention(dim=dim, heads=attn_heads, dim_head=attn_dim_head,
                         value_residual=value_residual, **attn_kwargs),
                SwiGLUFeedforward(dim=dim, **ff_kwargs)
            ]))

            rnn_layers.append(GRULayer(dim, dim) if is_time_block and rnn_time else None)
            rnn_attn_residuals.append(AttentionResidual(dim, **ar_kwargs) if is_time_block and rnn_time else None)
            attn_attn_residuals.append(AttentionResidual(dim, **ar_kwargs))
            ff_attn_residuals.append(AttentionResidual(dim, **ar_kwargs))

        self.layers = ModuleList(layers)
        self.rnn_layers = ModuleList(rnn_layers)
        self.is_time = is_time
        self.rnn_attn_residuals = ModuleList(rnn_attn_residuals)
        self.attn_attn_residuals = ModuleList(attn_attn_residuals)
        self.ff_attn_residuals = ModuleList(ff_attn_residuals)
        self.final_attn_residual = AttentionResidual(dim, **ar_kwargs)

        self.final_norm = RMSNorm(dim) if final_norm else nn.Identity()
        self.num_special_spatial_tokens = num_special_spatial_tokens

    def forward(
        self,
        tokens,  # (b, t, s, d)
        cache=None,
        return_intermediates=False,
    ):
        batch, time, space_seq_len, dim, device = *tokens.shape, tokens.device
        assert tokens.ndim == 4

        kv_cache = rnn_prev_hiddens = None
        if exists(cache):
            kv_cache = cache.next_kv_cache
            rnn_prev_hiddens = cache.next_rnn_hiddens

        has_kv_cache = exists(kv_cache)

        # attend functions
        attend_kwargs = dict(
            softclamp_value=self.attn_softclamp_value,
            special_attend_only_itself=self.special_attend_only_itself,
            device=device,
        )

        space_attend = get_attend_fn(
            causal=False, seq_len=space_seq_len,
            num_special_tokens=self.num_special_spatial_tokens,
            **attend_kwargs
        )

        time_attend = get_attend_fn(
            causal=True, seq_len=time,
            **attend_kwargs
        )

        # cache prep
        time_attn_kv_caches = []
        rnn_hiddens = []

        if has_kv_cache:
            past_tokens, tokens = tokens[:, :-1], tokens[:, -1:]
            rotary_seq_len = 1
            rotary_pos_offset = past_tokens.shape[1]
        else:
            rotary_seq_len = time
            rotary_pos_offset = 0

        kv_cache_iter = iter(default(kv_cache, (None,)))
        rnn_prev_iter = iter(default(rnn_prev_hiddens, (None,)))

        time_pos_emb = self.time_rotary(rotary_seq_len, offset=rotary_pos_offset)

        # value residual
        residual_values = None
        if self.value_residual:
            vr = self.to_value_residual(tokens)
            residual_values = vr.reshape(*vr.shape[:-1], self.vr_heads, self.vr_dim_head)

        normed_time_attn_inputs = []
        normed_space_attn_inputs = []

        layer_hiddens = [tokens]
        hiddens = []

        for (attn, ff), maybe_rnn, layer_is_time, rnn_attn_res, attn_attn_res, ff_attn_res in zip(
            self.layers, self.rnn_layers, self.is_time,
            self.rnn_attn_residuals, self.attn_attn_residuals, self.ff_attn_residuals
        ):
            # RNN block (only on time layers)
            if layer_is_time and exists(maybe_rnn):
                tokens = rnn_attn_res(tokens, layer_hiddens)

                # rearrange for temporal: (b, t, s, d) -> (b*s, t, d)
                b, t, s, d = tokens.shape
                tokens_t = tokens.permute(0, 2, 1, 3).reshape(b * s, t, d)
                tokens_t, layer_rnn_hiddens = maybe_rnn(tokens_t, next(rnn_prev_iter, None))
                tokens = tokens_t.reshape(b, s, t, d).permute(0, 2, 1, 3)

                rnn_hiddens.append(layer_rnn_hiddens)
                layer_hiddens.append(tokens)

            # Attention block (no explicit residual — attn_attn_res handles skip connections)
            tokens = attn_attn_res(tokens, layer_hiddens)

            if layer_is_time:
                # temporal attention: (b, t, s, d) -> (b*s, t, d)
                b, t, s, d = tokens.shape
                tokens_r = tokens.permute(0, 2, 1, 3).reshape(b * s, t, d)

                maybe_kv_cache = next(kv_cache_iter, None) if layer_is_time else None

                # value residual for temporal
                layer_rv = None
                if exists(residual_values):
                    # (b, t, s, h, dh) -> (b*s, t, h, dh)
                    rv_perm = residual_values.permute(0, 2, 1, 3, 4)
                    layer_rv = rv_perm.reshape(b * s, t, self.vr_heads, self.vr_dim_head)

                tokens, attn_inter = attn(
                    tokens_r,
                    rotary_pos_emb=time_pos_emb,
                    attend_fn=time_attend,
                    kv_cache=maybe_kv_cache,
                    residual_values=layer_rv,
                    return_intermediates=True,
                )

                tokens = tokens.reshape(b, s, t, d).permute(0, 2, 1, 3)
                time_attn_kv_caches.append(attn_inter.next_kv_cache)
                normed_time_attn_inputs.append(attn_inter.normed_inputs)

            else:
                # spatial attention: (b, t, s, d) -> (b*t, s, d)
                b, t, s, d = tokens.shape
                tokens_r = tokens.reshape(b * t, s, d)

                layer_rv = None
                if exists(residual_values):
                    rv_r = residual_values.reshape(b * t, s, self.vr_heads, self.vr_dim_head)
                    layer_rv = rv_r

                tokens, attn_inter = attn(
                    tokens_r,
                    attend_fn=space_attend,
                    residual_values=layer_rv,
                    return_intermediates=True,
                )

                tokens = tokens.reshape(b, t, s, d)
                normed_space_attn_inputs.append(attn_inter.normed_inputs)

            layer_hiddens.append(tokens)

            # Feedforward (no explicit residual — ff_attn_res handles skip connections)
            tokens = ff_attn_res(tokens, layer_hiddens)
            tokens = ff(tokens)
            layer_hiddens.append(tokens)
            hiddens.append(tokens)

        tokens = self.final_attn_residual(tokens, layer_hiddens)
        out = self.final_norm(tokens)

        if has_kv_cache:
            out = cat((past_tokens, out), dim=1)

        if not return_intermediates:
            return out

        intermediates = TransformerIntermediates(
            safe_stack(time_attn_kv_caches) if time_attn_kv_caches else None,
            safe_stack(normed_time_attn_inputs),
            safe_stack(normed_space_attn_inputs),
            safe_stack(rnn_hiddens),
            hiddens,
        )
        return out, intermediates
