# Online Dreamer-style latent imagination with tokenized transformer context.
#
# v23 keeps the scaled SDPA + GQA transformer from v22, and adds rotary time
# positions plus per-layer KV caches for incremental observe / imagine steps.
# This matches Dreamer4 more closely on the causal-time-attention path.
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins: int, bin_range: float = 20.0) -> torch.Tensor:
    if num_bins % 2 == 1:
        half = torch.linspace(-bin_range, 0.0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(0)])
    half = torch.linspace(-bin_range, 0.0, num_bins // 2)
    half = symexp(half)
    return torch.cat([half, -half.flip(0)])


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    below = (bins <= x.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    below_val = bins[below]
    above_val = bins[above]
    weight = (x - below_val) / (above_val - below_val + 1e-8)
    weight = weight.clamp(0, 1)
    result = torch.zeros(*x.shape, len(bins), device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))
    return result


def twohot_predict(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    two_hot = twohot_encode(target, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(two_hot * log_probs).sum(-1)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, act="silu", norm=True, out_scale=1.0):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        dims = [in_dim] + [hidden] * layers + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    mods.append(RMSNorm(dims[i + 1]))
                mods.append(act_fn())
        self.net = nn.Sequential(*mods)
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        if out_scale == 0.0:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            fan_in = last.weight.shape[1]
            nn.init.trunc_normal_(last.weight, std=out_scale / math.sqrt(fan_in))
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim_head, heads, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(heads, dim_head))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale.unsqueeze(0).unsqueeze(2)


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)


def apply_rotary(x, freqs_cos, freqs_sin):
    return (x * freqs_cos) + (rotate_half(x) * freqs_sin)


class BlockLinear(nn.Module):
    def __init__(self, in_features, out_features, blocks):
        super().__init__()
        assert in_features % blocks == 0 and out_features % blocks == 0
        self.blocks = blocks
        self.in_per = in_features // blocks
        self.out_per = out_features // blocks
        self.weight = nn.Parameter(torch.empty(blocks, self.in_per, self.out_per))
        self.bias = nn.Parameter(torch.zeros(blocks, self.out_per))
        for b in range(blocks):
            nn.init.xavier_uniform_(self.weight[b])

    def forward(self, x):
        *batch, _ = x.shape
        x = x.reshape(*batch, self.blocks, self.in_per)
        out = torch.einsum('...bi,bio->...bo', x, self.weight) + self.bias
        return out.reshape(*batch, self.blocks * self.out_per)


# ---------------------------------------------------------------------------
# Transformer World Model
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, heads, query_heads=None):
        super().__init__()
        query_heads = heads if query_heads is None else query_heads
        assert query_heads >= heads and query_heads % heads == 0
        assert dim % query_heads == 0

        self.heads = heads
        self.query_heads = query_heads
        self.dim_head = dim // query_heads
        self.use_gqa = query_heads != heads

        dim_q = self.dim_head * query_heads
        dim_kv = self.dim_head * heads

        self.to_q = nn.Linear(dim, dim_q, bias=False)
        self.to_k = nn.Linear(dim, dim_kv, bias=False)
        self.to_v = nn.Linear(dim, dim_kv, bias=False)
        self.to_out = nn.Linear(dim_q, dim, bias=False)
        self.to_gates = nn.Sequential(
            nn.Linear(dim, query_heads, bias=False),
            nn.Sigmoid(),
        )
        self.k_norm = MultiHeadRMSNorm(self.dim_head, heads)
        assert self.dim_head % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_head, 2).float() / self.dim_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotary_freqs(self, positions, num_heads):
        freqs = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).unsqueeze(1).expand(-1, num_heads, -1, -1)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).unsqueeze(1).expand(-1, num_heads, -1, -1)
        return cos, sin

    def _append_kv_cache(self, k_new, v_new, kv_cache, cache_lengths, max_seq_len):
        if kv_cache is None:
            return k_new, v_new

        k_prev, v_prev = kv_cache
        batch = k_new.shape[0]
        new_len = k_new.shape[-2]
        total_len = min(max_seq_len, k_prev.shape[-2] + new_len)
        k_all = k_new.new_zeros(batch, self.heads, total_len, self.dim_head)
        v_all = v_new.new_zeros(batch, self.heads, total_len, self.dim_head)

        for i in range(batch):
            prev_len = int(cache_lengths[i].item())
            prev_len = min(prev_len, k_prev.shape[-2], total_len - new_len)
            keep_from = max(0, prev_len - (total_len - new_len))
            kept = prev_len - keep_from
            if kept > 0:
                k_all[i, :, :kept] = k_prev[i, :, keep_from:keep_from + kept]
                v_all[i, :, :kept] = v_prev[i, :, keep_from:keep_from + kept]
            k_all[i, :, kept:kept + new_len] = k_new[i]
            v_all[i, :, kept:kept + new_len] = v_new[i]
        return k_all, v_all

    def forward(
        self,
        x,
        positions,
        attn_mask=None,
        key_padding_mask=None,
        kv_cache=None,
        cache_lengths=None,
        max_seq_len=None,
        return_kv_cache=False,
    ):
        bsz, seq_len, _ = x.shape

        q = self.to_q(x).reshape(bsz, seq_len, self.query_heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).reshape(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).reshape(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2)

        k = self.k_norm(k)
        q_cos, q_sin = self.rotary_freqs(positions, self.query_heads)
        k_cos, k_sin = self.rotary_freqs(positions, self.heads)
        q = apply_rotary(q, q_cos, q_sin)
        k = apply_rotary(k, k_cos, k_sin)

        if kv_cache is not None:
            if cache_lengths is None or max_seq_len is None:
                raise ValueError("cache_lengths and max_seq_len are required when kv_cache is used")
            k, v = self._append_kv_cache(k, v, kv_cache, cache_lengths, max_seq_len)

        allowed_mask = None
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("GroupedQueryAttention expects a boolean attn_mask")
            if attn_mask.ndim == 2:
                allowed_mask = ~attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                allowed_mask = ~attn_mask.unsqueeze(1)
            else:
                raise ValueError("GroupedQueryAttention attn_mask must be 2D or 3D")
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("GroupedQueryAttention expects a boolean key_padding_mask")
            key_allowed = ~key_padding_mask[:, None, None, :]
            allowed_mask = key_allowed if allowed_mask is None else (allowed_mask & key_allowed)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed_mask,
            dropout_p=0.0,
            enable_gqa=self.use_gqa,
        )

        gates = self.to_gates(x).transpose(1, 2).unsqueeze(-1)
        attn_out = attn_out * gates
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self.query_heads * self.dim_head)
        out = self.to_out(attn_out)
        if return_kv_cache:
            return out, (k, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, query_heads=None, mlp_ratio=4.0, act="silu"):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, heads, query_heads=query_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_fn(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(
        self,
        x,
        positions,
        attn_mask=None,
        key_padding_mask=None,
        kv_cache=None,
        cache_lengths=None,
        max_seq_len=None,
        return_kv_cache=False,
    ):
        h = self.norm1(x)
        attn_out = self.attn(
            h,
            positions,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            kv_cache=kv_cache,
            cache_lengths=cache_lengths,
            max_seq_len=max_seq_len,
            return_kv_cache=return_kv_cache,
        )
        kv_cache_out = None
        if return_kv_cache:
            attn_out, kv_cache_out = attn_out
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        if return_kv_cache:
            return x, kv_cache_out
        return x


class TransformerWorldModel(nn.Module):
    def __init__(self, token_dim, act_dim, dim=256, heads=4, query_heads=None, layers=4, context_len=64, act="silu"):
        super().__init__()
        self.dim = dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.feat_dim = dim
        self.token_dim = token_dim
        self.tokens_per_step = 4
        self.obs_proj = nn.Linear(token_dim, dim)
        self.act_proj = nn.Linear(act_dim, dim)
        self.rew_proj = nn.Linear(1, dim)
        self.agent_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.type_emb = nn.Embedding(self.tokens_per_step, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, query_heads=query_heads, act=act) for _ in range(layers)]
        )
        self.final_norm = RMSNorm(dim)
        self.transition = MLP(dim + act_dim, dim, dim, 2, act=act, out_scale=0.0)
        self.next_obs_token = MLP(dim, token_dim, dim, 1, act=act, out_scale=0.0)

    def initial(self, batch_size, device):
        return {
            'history': torch.zeros(batch_size, self.context_len, self.tokens_per_step, self.dim, device=device),
            'steps': torch.zeros(batch_size, dtype=torch.long, device=device),
            'agent': torch.zeros(batch_size, self.dim, device=device),
            'token_count': torch.zeros(batch_size, dtype=torch.long, device=device),
            'kv_caches': None,
        }

    def get_feat(self, state):
        return state['agent']

    def build_step_tokens(self, obs_token, prev_action, prev_reward, is_first):
        keep = (~is_first).float().unsqueeze(-1)
        obs_tok = self.obs_proj(obs_token) + self.type_emb.weight[0]
        act_tok = self.act_proj(prev_action * keep) + self.type_emb.weight[1]
        rew_tok = self.rew_proj(symlog(prev_reward).unsqueeze(-1) * keep) + self.type_emb.weight[2]
        agent_tok = self.agent_token.unsqueeze(0).expand(obs_token.shape[0], -1) + self.type_emb.weight[3]
        return torch.stack([obs_tok, act_tok, rew_tok, agent_tok], dim=1)

    def _append(self, history, steps, step_tokens, reset_mask=None, active_mask=None):
        new_history = history.clone()
        new_steps = steps.clone()
        batch = history.shape[0]
        if reset_mask is None:
            reset_mask = torch.zeros(batch, dtype=torch.bool, device=history.device)
        if active_mask is None:
            active_mask = torch.ones(batch, dtype=torch.bool, device=history.device)
        for i in range(batch):
            if not active_mask[i]:
                continue
            if reset_mask[i]:
                new_history[i].zero_()
                new_steps[i] = 0
            length = int(new_steps[i].item())
            if length < self.context_len:
                new_history[i, length] = step_tokens[i]
                new_steps[i] = length + 1
            else:
                new_history[i, :-1] = new_history[i, 1:].clone()
                new_history[i, -1] = step_tokens[i]
        return new_history, new_steps

    def _positions_for_history(self, steps, token_count):
        batch, device = steps.shape[0], steps.device
        seq_len = self.context_len * self.tokens_per_step
        valid_tokens = steps.clamp(min=0) * self.tokens_per_step
        start = (token_count - valid_tokens).unsqueeze(1)
        offsets = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        return start + offsets

    def _run_transformer(self, history, steps, token_count, return_kv_caches=False):
        batch, ctx, per_step, dim = history.shape
        seq_len = ctx * per_step
        x = history.reshape(batch, seq_len, dim)
        positions = self._positions_for_history(steps, token_count)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=history.device, dtype=torch.bool),
            diagonal=1,
        )
        valid_tokens = (steps.clamp(min=1) * per_step).unsqueeze(1)
        key_padding_mask = torch.arange(seq_len, device=history.device).unsqueeze(0) >= valid_tokens
        kv_caches = [] if return_kv_caches else None
        for block in self.blocks:
            x = block(
                x,
                positions,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                return_kv_cache=return_kv_caches,
            )
            if return_kv_caches:
                x, kv_cache = x
                kv_caches.append(kv_cache)
        x = self.final_norm(x).reshape(batch, ctx, per_step, dim)
        if return_kv_caches:
            return x, kv_caches
        return x

    def _run_transformer_cached(self, step_tokens, state):
        batch, new_seq_len, _ = step_tokens.shape
        positions = state['token_count'].unsqueeze(1) + torch.arange(new_seq_len, device=step_tokens.device).unsqueeze(0)
        cache_lengths = state['steps'] * self.tokens_per_step
        max_seq_len = self.context_len * self.tokens_per_step
        step_causal = torch.triu(
            torch.ones(new_seq_len, new_seq_len, device=step_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_mask = torch.ones(batch, new_seq_len, max_seq_len, device=step_tokens.device, dtype=torch.bool)
        for i in range(batch):
            prefix = min(int(cache_lengths[i].item()), max_seq_len)
            total_len = min(max_seq_len, prefix + new_seq_len)
            kept = total_len - new_seq_len
            attn_mask[i, :, :kept] = False
            attn_mask[i, :, kept:kept + new_seq_len] = step_causal

        x = step_tokens
        new_kv_caches = []
        for idx, block in enumerate(self.blocks):
            x, next_cache = block(
                x,
                positions,
                attn_mask=attn_mask,
                kv_cache=None if state['kv_caches'] is None else state['kv_caches'][idx],
                cache_lengths=cache_lengths,
                max_seq_len=max_seq_len,
                return_kv_cache=True,
            )
            new_kv_caches.append(next_cache)
        x = self.final_norm(x)
        return x[:, -1], new_kv_caches

    def _extract_agent(self, history_out, steps):
        agent = history_out[:, :, -1]
        last_idx = (steps - 1).clamp(min=0)
        agent = agent[torch.arange(agent.shape[0], device=agent.device), last_idx]
        return agent * (steps > 0).unsqueeze(-1)

    def observe_step(self, state, obs_token, prev_action, prev_reward, is_first, active_mask=None):
        step_tokens = self.build_step_tokens(obs_token, prev_action, prev_reward, is_first)
        history, steps = self._append(
            state['history'], state['steps'], step_tokens, reset_mask=is_first, active_mask=active_mask
        )
        token_count = state['token_count'].clone()
        active = torch.ones_like(is_first) if active_mask is None else active_mask
        token_count[active & is_first] = self.tokens_per_step
        token_count[active & ~is_first] += self.tokens_per_step

        use_cache = (
            state['kv_caches'] is not None
            and bool(active.all().item())
            and not bool(is_first.any().item())
        )
        if use_cache:
            agent, kv_caches = self._run_transformer_cached(step_tokens, state)
        else:
            history_out, kv_caches = self._run_transformer(history, steps, token_count, return_kv_caches=True)
            agent = self._extract_agent(history_out, steps)

        new_state = {
            'history': history,
            'steps': steps,
            'agent': agent,
            'token_count': token_count,
            'kv_caches': kv_caches,
        }
        return new_state, agent

    def observe(self, obs_tokens, prev_actions, prev_rewards, is_first, valid_mask=None):
        batch, time, _ = obs_tokens.shape
        if valid_mask is None:
            valid_mask = torch.ones(batch, time, dtype=torch.bool, device=obs_tokens.device)

        step_tokens = self.build_step_tokens(
            obs_tokens.reshape(batch * time, -1),
            prev_actions.reshape(batch * time, -1),
            prev_rewards.reshape(batch * time),
            is_first.reshape(batch * time),
        ).reshape(batch, time, self.tokens_per_step, self.dim)

        episode_ids = torch.full((batch, time), -1, dtype=torch.long, device=obs_tokens.device)
        step_positions = torch.zeros((batch, time), dtype=torch.long, device=obs_tokens.device)
        for b in range(batch):
            ep = -1
            pos = 0
            prev_valid = False
            for t in range(time):
                if not bool(valid_mask[b, t].item()):
                    prev_valid = False
                    continue
                if (not prev_valid) or bool(is_first[b, t].item()):
                    ep += 1
                    pos = 0
                else:
                    pos += 1
                episode_ids[b, t] = ep
                step_positions[b, t] = pos
                prev_valid = True

        seq_len = time * self.tokens_per_step
        token_offsets = torch.arange(self.tokens_per_step, device=obs_tokens.device).view(1, 1, self.tokens_per_step)
        positions = (step_positions.unsqueeze(-1) * self.tokens_per_step + token_offsets).reshape(batch, seq_len)
        episode_flat = episode_ids.unsqueeze(-1).expand(-1, -1, self.tokens_per_step).reshape(batch, seq_len)
        valid_flat = valid_mask.unsqueeze(-1).expand(-1, -1, self.tokens_per_step).reshape(batch, seq_len)

        x = step_tokens.reshape(batch, seq_len, self.dim)
        idx = torch.arange(seq_len, device=obs_tokens.device)
        causal_mask = idx.unsqueeze(0) > idx.unsqueeze(1)
        same_episode = episode_flat[:, :, None] == episode_flat[:, None, :]
        allowed = same_episode & valid_flat[:, None, :]
        disallow = ~allowed | causal_mask.unsqueeze(0) | (~valid_flat[:, :, None])

        for block in self.blocks:
            x = block(x, positions, attn_mask=disallow)
        x = self.final_norm(x).reshape(batch, time, self.tokens_per_step, self.dim)
        agents = x[:, :, -1] * valid_mask.unsqueeze(-1)
        return {'agent': agents}

    def seed_from_sequence(self, obs_tokens, prev_actions, prev_rewards, is_first, lengths):
        state = self.initial(obs_tokens.shape[0], obs_tokens.device)
        max_len = obs_tokens.shape[1]
        for t in range(max_len):
            active_mask = t < lengths
            state, _ = self.observe_step(
                state, obs_tokens[:, t], prev_actions[:, t], prev_rewards[:, t], is_first[:, t], active_mask=active_mask
            )
        return state

    def transition_features(self, agent, action):
        return self.transition(torch.cat([agent, action], dim=-1))

    def predict_next_obs_token(self, transition_feat):
        return self.next_obs_token(transition_feat)

    def imagine_step(self, state, action, reward_head, continue_head):
        transition_feat = self.transition_features(state['agent'], action)
        next_obs_token = self.predict_next_obs_token(transition_feat)
        reward = reward_head.predict(transition_feat)
        cont_logit = continue_head(transition_feat)
        next_state, agent = self.observe_step(
            state,
            next_obs_token,
            action,
            reward,
            torch.zeros(action.shape[0], dtype=torch.bool, device=action.device),
        )
        next_state['agent'] = agent
        return next_state, transition_feat, next_obs_token, reward, cont_logit


# ---------------------------------------------------------------------------
# Encoder / Decoder / Heads
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(obs_dim, units, units, layers, act=act, out_scale=1.0)

    def forward(self, obs):
        return self.mlp(symlog(obs))


class Decoder(nn.Module):
    def __init__(self, feat_dim, obs_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, obs_dim, units, layers, act=act, out_scale=1.0)

    def forward(self, feat):
        return self.mlp(feat)

    def loss(self, feat, obs_target):
        pred = self.forward(feat)
        return 0.5 * (pred - symlog(obs_target)).pow(2).sum(-1)


class RewardHead(nn.Module):
    def __init__(self, feat_dim, units, num_bins=255, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, num_bins, units, 1, act=act, out_scale=0.0)
        self.register_buffer('bins', build_symexp_bins(num_bins))

    def forward(self, feat):
        return self.mlp(feat)

    def loss(self, feat, target):
        logits = self.forward(feat)
        return twohot_loss(logits, target, self.bins)

    def predict(self, feat):
        logits = self.forward(feat)
        return twohot_predict(logits, self.bins)


class ContinueHead(nn.Module):
    def __init__(self, feat_dim, units, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, 1, units, 1, act=act, out_scale=1.0)

    def forward(self, feat):
        return self.mlp(feat).squeeze(-1)

    def loss(self, feat, target):
        logits = self.forward(feat)
        return F.binary_cross_entropy_with_logits(logits, target)

    def predict(self, feat):
        return torch.sigmoid(self.forward(feat))


class Actor(nn.Module):
    def __init__(self, feat_dim, act_dim, units, layers, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, 2 * act_dim, units, layers, act=act, out_scale=0.01)
        self.act_dim = act_dim

    def stats(self, feat):
        out = self.mlp(feat)
        mean, log_var = out.split(self.act_dim, -1)
        return mean, log_var

    def dist_from_stats(self, mean, log_var):
        std = (0.5 * log_var.clamp(-10, 4)).exp()  # clamp for numerical safety
        return Normal(mean, std)

    def forward(self, feat):
        mean, log_var = self.stats(feat)
        return self.dist_from_stats(mean, log_var)


class Critic(nn.Module):
    def __init__(self, feat_dim, units, layers, num_bins=255, act="silu"):
        super().__init__()
        self.mlp = MLP(feat_dim, num_bins, units, layers, act=act, out_scale=0.0)
        self.register_buffer('bins', build_symexp_bins(num_bins))

    def forward(self, feat):
        return self.mlp(feat)

    def predict(self, feat):
        return twohot_predict(self.forward(feat), self.bins)

    def loss(self, logits, target):
        return twohot_loss(logits, target, self.bins)


# ---------------------------------------------------------------------------
# Percentile Return Normalization
# ---------------------------------------------------------------------------

class EMAReturnNorm:
    """EMA-based return normalization (dreamer4 style).
    Tracks mean/std of returns via EMA, normalizes both returns and values."""
    def __init__(self, decay=0.998, perclo=5.0, perchi=95.0):
        self.decay = decay
        self.perclo = perclo
        self.perchi = perchi
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, returns):
        flat = returns.detach().float()
        # Quantile clamp (dreamer4 style) to reduce outlier influence
        lo = torch.quantile(flat, self.perclo / 100.0)
        hi = torch.quantile(flat, self.perchi / 100.0)
        clamped = flat.clamp(lo, hi)
        batch_mean = clamped.mean().item()
        batch_var = clamped.var().item()
        if not self.initialized:
            self.mean = batch_mean
            self.var = max(batch_var, 1e-8)
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1 - self.decay) * batch_var

    def normalize(self, x):
        std = max(self.var ** 0.5, 1e-4)
        return (x - self.mean) / std


# ---------------------------------------------------------------------------
# GAE Returns
# ---------------------------------------------------------------------------

def compute_gae_returns(rew, val, gamma, lam, discounts=None):
    horizon = rew.shape[1]
    gae = torch.zeros_like(rew[:, 0])
    returns = []
    for t in reversed(range(horizon)):
        discount_t = gamma if discounts is None else discounts[:, t]
        delta = rew[:, t] + discount_t * val[:, t + 1] - val[:, t]
        gae = delta + discount_t * lam * gae
        returns.append(gae + val[:, t])
    returns.reverse()
    return torch.stack(returns, 1)


# ---------------------------------------------------------------------------
# Replay Buffer (episode-based, like dreamer4)
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.episodes = []
        self.total_steps = 0

    @property
    def num_steps(self):
        return self.total_steps

    def add_episode(self, episode):
        length = len(episode['reward'])
        if length < 2:
            return
        self.episodes.append(episode)
        self.total_steps += length
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed['reward'])

    def can_sample(self, batch_length):
        return len(self.episodes) > 0

    def sample(self, batch_size, batch_length):
        valid = self.episodes

        keys = tuple(valid[0].keys())
        sample_ep = valid[0]
        batch = {}
        for k in keys:
            shape = sample_ep[k].shape[1:]
            batch[k] = np.zeros((batch_size, batch_length) + shape, dtype=sample_ep[k].dtype)
        batch['valid'] = np.zeros((batch_size, batch_length), dtype=bool)
        batch['length'] = np.zeros((batch_size,), dtype=np.int64)

        for batch_index in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            ep_len = len(ep['reward'])
            if ep_len >= batch_length:
                start = np.random.randint(0, ep_len - batch_length + 1)
            else:
                start = 0
            end = start + batch_length
            chunk_len = min(ep_len - start, batch_length)
            batch['valid'][batch_index, :chunk_len] = True
            batch['length'][batch_index] = chunk_len
            for k in keys:
                chunk = ep[k][start:end]
                batch[k][batch_index, : len(chunk)] = chunk

        return batch

    def sample_tensors(self, batch_size, batch_length, device: torch.device):
        batch_np = self.sample(batch_size, batch_length)
        batch = {}
        for key, value in batch_np.items():
            tensor = torch.from_numpy(value)
            if tensor.dtype == torch.bool:
                batch[key] = tensor.to(device=device)
            elif key == 'length':
                batch[key] = tensor.to(device=device, dtype=torch.long)
            else:
                batch[key] = tensor.to(device=device, dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 8

    # Transformer world model
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_query_heads: int = 8
    transformer_layers: int = 4
    transformer_context: int = 32
    dyn_scale: float = 1.0
    rep_scale: float = 0.1

    # Networks
    mlp_units: int = 256
    mlp_layers: int = 2
    num_bins: int = 255

    # Training
    learning_rate: float = 3e-4
    wm_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    eps: float = 1e-8
    batch_size_per_env: int = 16  # d4 ref: 16 at 1 env, scales with num_envs
    batch_length: int = 32
    train_ratio: float = 64.0
    imag_last: int = 8
    max_grad_norm: float = 100.0
    compile: bool = True

    # Imagination
    imag_horizon: int = 15

    # Actor-Critic
    horizon: int = 333
    lam: float = 0.95
    actor_entropy: float = 0.01
    pmpo_pos_neg_weight: float = 0.5  # α in PMPO: weight for positive advantages
    pmpo_kl_weight: float = 0.3  # reverse KL constraint weight
    gae_discount: float = 0.997
    keep_return_ema_stats: bool = False
    return_norm_decay: float = 0.998
    return_norm_perclo: float = 5.0
    return_norm_perchi: float = 95.0
    value_clip: float = 0.4  # clipped value loss range (dreamer4)

    # Replay
    replay_size: int = 1_000_000
    prefill_steps: int = 5000

    num_iterations: int = 0


# ---------------------------------------------------------------------------
# make_env (no normalization wrappers - symlog handles scale)
# ---------------------------------------------------------------------------

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args_class=Args):
    args = tyro.cli(args_class)
    args.batch_size = args.batch_size_per_env * args.num_envs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    # Build models
    encoder = Encoder(obs_dim, args.mlp_units, args.mlp_layers).to(device)
    token_dim = args.mlp_units

    world_model = TransformerWorldModel(
        token_dim=token_dim,
        act_dim=act_dim,
        dim=args.transformer_dim,
        heads=args.transformer_heads,
        query_heads=args.transformer_query_heads,
        layers=args.transformer_layers,
        context_len=args.transformer_context,
    ).to(device)

    feat_dim = world_model.feat_dim

    decoder = Decoder(token_dim, obs_dim, args.mlp_units, args.mlp_layers).to(device)
    reward_head = RewardHead(feat_dim, args.mlp_units, args.num_bins).to(device)
    continue_head = ContinueHead(feat_dim, args.mlp_units).to(device)
    actor = Actor(feat_dim, act_dim, args.mlp_units, args.mlp_layers).to(device)
    critic = Critic(feat_dim, args.mlp_units, args.mlp_layers, args.num_bins).to(device)

    # Optimizers (separate for WM, actor, critic)
    wm_params = (list(encoder.parameters()) + list(world_model.parameters()) +
                 list(decoder.parameters()) + list(reward_head.parameters()) +
                 list(continue_head.parameters()))
    wm_opt = optim.Adam(wm_params, lr=args.wm_learning_rate, eps=args.eps)
    actor_opt = optim.Adam(actor.parameters(), lr=args.actor_learning_rate, eps=args.eps)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_learning_rate, eps=args.eps)

    # torch.compile
    # Compile actor MLP separately (Normal distribution breaks torch.compile)
    actor_mlp_compiled = actor.mlp
    if args.compile:
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)
        reward_head = torch.compile(reward_head)
        continue_head = torch.compile(continue_head)
        critic = torch.compile(critic)
        actor_mlp_compiled = torch.compile(actor.mlp)

    bins = build_symexp_bins(args.num_bins).to(device)
    return_norm = EMAReturnNorm(
        decay=args.return_norm_decay,
        perclo=args.return_norm_perclo,
        perchi=args.return_norm_perchi,
    )

    replay = EpisodeReplayBuffer(args.replay_size)
    cont_target_val = 1.0 - 1.0 / args.horizon

    # Episode tracking per env
    ongoing = [None] * args.num_envs

    def init_ongoing(env_id, obs_np):
        ongoing[env_id] = {
            'obs': [obs_np.copy()],
            'prev_act': [np.zeros(act_dim, dtype=np.float32)],
            'reward': [0.0],
            'is_first': [True],
            'is_terminal': [False],
        }

    def finish_ongoing(env_id, final_obs, last_action, last_reward, terminated):
        ep = ongoing[env_id]
        ep['obs'].append(final_obs.copy())
        ep['prev_act'].append(last_action.copy())
        ep['reward'].append(float(last_reward))
        ep['is_first'].append(False)
        ep['is_terminal'].append(bool(terminated))
        episode = {k: np.array(v, dtype=np.float32 if k not in ('is_first', 'is_terminal') else bool)
                   for k, v in ep.items()}
        replay.add_episode(episode)

    def add_step(env_id, next_obs, action, reward, is_first, is_terminal):
        ep = ongoing[env_id]
        ep['obs'].append(next_obs.copy())
        ep['prev_act'].append(action.copy())
        ep['reward'].append(float(reward))
        ep['is_first'].append(bool(is_first))
        ep['is_terminal'].append(bool(is_terminal))

    # --- Training step ---
    def train_step():
        batch = replay.sample_tensors(args.batch_size, args.batch_length, device)
        obs = batch['obs']
        prev_act = batch['prev_act']
        rew = batch['reward']
        is_first = batch['is_first']
        is_term = batch['is_terminal']
        valid = batch['valid']
        lengths = batch['length']
        B, T = obs.shape[:2]

        is_first = is_first.clone()
        is_first[:, 0] = True

        # ===== World Model =====
        wm_opt.zero_grad()

        tokens = encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)
        states = world_model.observe(tokens, prev_act, rew, is_first, valid_mask=valid)
        feat = states['agent']

        src_feat = feat[:, :-1]
        src_act = prev_act[:, 1:]
        target_tokens = tokens[:, 1:]
        transition_mask = valid[:, 1:]
        transition_feat = world_model.transition_features(
            src_feat.reshape(B * (T - 1), -1), src_act.reshape(B * (T - 1), -1)
        )
        pred_next_token = world_model.predict_next_obs_token(transition_feat).reshape(B, T - 1, -1)

        rec_loss_t = decoder.loss(
            pred_next_token.reshape(B * (T - 1), -1), obs[:, 1:].reshape(B * (T - 1), -1)
        ).reshape(B, T - 1)
        rew_loss_t = reward_head.loss(
            transition_feat, rew[:, 1:].reshape(B * (T - 1))
        ).reshape(B, T - 1)

        con_target = (~is_term[:, 1:]).float() * cont_target_val
        con_logits = continue_head(transition_feat)
        con_loss_t = F.binary_cross_entropy_with_logits(
            con_logits, con_target.reshape(B * (T - 1)), reduction='none'
        ).reshape(B, T - 1)

        dyn_loss_t = 0.5 * (pred_next_token - target_tokens.detach()).pow(2).mean(-1)
        rep_loss_t = 0.5 * (target_tokens - pred_next_token.detach()).pow(2).mean(-1)
        rec_loss = rec_loss_t[transition_mask].mean()
        rew_loss = rew_loss_t[transition_mask].mean()
        con_loss = con_loss_t[transition_mask].mean()
        dyn_loss = dyn_loss_t[transition_mask].mean()
        rep_loss = rep_loss_t[transition_mask].mean()

        wm_loss = rec_loss + rew_loss + con_loss + args.dyn_scale * dyn_loss + args.rep_scale * rep_loss

        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
        wm_opt.step()

        # ===== Imagination =====
        with torch.no_grad():
            K = min(args.imag_last, T) if args.imag_last > 0 else T
            BK = B * K
            hist_len = min(args.transformer_context, T)
            seed_obs_tokens = tokens.new_zeros(BK, hist_len, token_dim)
            seed_prev_act = prev_act.new_zeros(BK, hist_len, act_dim)
            seed_prev_rew = rew.new_zeros(BK, hist_len)
            seed_is_first = torch.ones(BK, hist_len, dtype=torch.bool, device=device)
            seed_len = torch.zeros(BK, dtype=torch.long, device=device)
            seed_valid = torch.zeros(BK, dtype=torch.bool, device=device)
            for b in range(B):
                valid_len = int(lengths[b].item())
                num_seed = min(K, valid_len)
                start_seed = max(0, valid_len - num_seed)
                for k in range(K):
                    t = start_seed + min(k, max(num_seed - 1, 0))
                    start = max(0, t - hist_len + 1)
                    length = t - start + 1
                    idx = b * K + k
                    seed_obs_tokens[idx, :length] = tokens[b, start:t + 1]
                    seed_prev_act[idx, :length] = prev_act[b, start:t + 1]
                    seed_prev_rew[idx, :length] = rew[b, start:t + 1]
                    seed_is_first[idx, :length] = is_first[b, start:t + 1]
                    seed_len[idx] = length
                    seed_valid[idx] = k < num_seed
            state = world_model.seed_from_sequence(
                seed_obs_tokens, seed_prev_act, seed_prev_rew, seed_is_first, seed_len
            )
            cur_feat = world_model.get_feat(state)
            img_feats = [cur_feat]
            img_acts = []
            img_old_means = []
            img_old_logvars = []
            img_rews = []
            img_conts = []
            for h in range(args.imag_horizon):
                # Manual actor forward (compiled MLP + manual Normal sample).
                actor_out = actor_mlp_compiled(cur_feat)
                act_mean, act_logvar = actor_out.split(act_dim, -1)
                act_logvar = act_logvar.clamp(-10, 4)
                act_std = (0.5 * act_logvar).exp()
                action = (act_mean + act_std * torch.randn_like(act_std)).clamp(-1, 1)
                img_old_means.append(act_mean)
                img_old_logvars.append(act_logvar)
                img_acts.append(action)
                state, transition_feat, _, pred_reward, cont_logit = world_model.imagine_step(
                    state, action, reward_head, continue_head
                )
                img_rews.append(pred_reward)
                img_conts.append(torch.sigmoid(cont_logit))
                cur_feat = world_model.get_feat(state)
                img_feats.append(cur_feat)

            img_feats = torch.stack(img_feats, 1)
            img_acts = torch.stack(img_acts, 1)
            img_old_means = torch.stack(img_old_means, 1)
            img_old_logvars = torch.stack(img_old_logvars, 1)
            img_rews = torch.stack(img_rews, 1)
            img_conts = torch.stack(img_conts, 1)
            seed_rollout_mask = seed_valid.unsqueeze(1).expand(-1, args.imag_horizon)

            H1 = args.imag_horizon + 1
            img_feats_flat = img_feats.reshape(BK * H1, -1)
            img_old_val = twohot_predict(critic(img_feats_flat), bins).reshape(BK, H1).float()
            returns = compute_gae_returns(
                img_rews, img_old_val, args.gae_discount, args.lam,
                discounts=args.gae_discount * img_conts.clamp(0.0, 1.0)
            )

            if args.keep_return_ema_stats:
                return_norm.update(returns[seed_valid])
                advantages = return_norm.normalize(returns) - return_norm.normalize(img_old_val[:, :-1])
                ret_scale = return_norm.var ** 0.5 if return_norm.initialized else 1.0
            else:
                advantages = returns - img_old_val[:, :-1]
                ret_scale = 1.0

            imag_actor_feats = img_feats[:, :-1].reshape(BK * args.imag_horizon, -1)
            imag_acts_flat = img_acts.reshape(BK * args.imag_horizon, -1)
            old_val_for_critic = img_old_val[:, :-1].reshape(-1)

        # ===== Actor (PMPO) =====
        actor_opt.zero_grad()

        policy_dist = actor(imag_actor_feats.detach())
        logpi = policy_dist.log_prob(imag_acts_flat.detach()).reshape(BK, args.imag_horizon, act_dim)
        entropy = policy_dist.entropy().reshape(BK, args.imag_horizon, act_dim).sum(-1)

        adv_detached = advantages.detach()
        pos_mask = (adv_detached >= 0) & seed_rollout_mask
        neg_mask = (adv_detached < 0) & seed_rollout_mask

        alpha = args.pmpo_pos_neg_weight
        pos_term = (logpi * pos_mask.unsqueeze(-1)).sum() / (pos_mask.sum().clamp(min=1) * act_dim)
        neg_term = -(logpi * neg_mask.unsqueeze(-1)).sum() / (neg_mask.sum().clamp(min=1) * act_dim)
        pmpo_loss = -(alpha * pos_term + (1 - alpha) * neg_term)

        entropy_loss = -entropy[seed_rollout_mask].float().mean()

        if args.pmpo_kl_weight > 0:
            old_mu = img_old_means.detach().reshape(BK * args.imag_horizon, -1)
            old_logvar = img_old_logvars.detach().reshape(BK * args.imag_horizon, -1)
            new_mu, new_logvar = actor.stats(imag_actor_feats.detach())
            new_logvar = new_logvar.clamp(-10, 4)
            old_var = old_logvar.exp()
            new_var = new_logvar.exp()
            kl_div = 0.5 * (
                new_logvar - old_logvar +
                (old_var + (old_mu - new_mu).pow(2)) / new_var - 1.0
            ).sum(-1)
            kl_loss = kl_div.reshape(BK, args.imag_horizon)[seed_rollout_mask].mean()
        else:
            kl_loss = torch.zeros(1, device=device)

        actor_loss = pmpo_loss + args.actor_entropy * entropy_loss + args.pmpo_kl_weight * kl_loss

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_opt.step()

        # ===== Critic (clipped value loss, rollout-time baseline) =====
        critic_opt.zero_grad()

        val_logits = critic(imag_actor_feats.detach())
        returns_flat = returns.reshape(-1).detach()
        return_twohot = twohot_encode(returns_flat, bins)
        loss_unclipped = -(return_twohot * F.log_softmax(val_logits, dim=-1)).sum(-1).reshape(
            BK, args.imag_horizon
        )
        val_pred = twohot_predict(val_logits, bins).reshape(BK, args.imag_horizon)
        old_val = old_val_for_critic.detach().reshape(BK, args.imag_horizon)
        clipped_val = old_val + (val_pred - old_val).clamp(-args.value_clip, args.value_clip)
        clipped_twohot = twohot_encode(clipped_val.reshape(-1), bins)
        loss_clipped = -(return_twohot * F.log_softmax(clipped_twohot, dim=-1)).sum(-1).reshape(
            BK, args.imag_horizon
        )

        loss_returns = torch.max(loss_unclipped, loss_clipped)
        critic_loss = loss_returns[seed_rollout_mask].mean()

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        critic_opt.step()

        return {
            'wm_loss': wm_loss.item(),
            'rec_loss': rec_loss.item(),
            'rew_loss': rew_loss.item(),
            'con_loss': con_loss.item(),
            'dyn_kl': dyn_loss.item(),
            'rep_kl': rep_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy[seed_rollout_mask].mean().item(),
            'returns': returns[seed_valid].mean().item(),
            'ret_scale': ret_scale,
        }

    # === Flush stdout for background task visibility ===
    import builtins
    _print = builtins.print
    def print(*a, **kw):
        kw.setdefault('flush', True)
        _print(*a, **kw)

    print(f"Starting latent_imagination_v23 on {args.env_id} with {args.num_envs} envs")
    total_params = sum(p.numel() for p in wm_params) + sum(
        p.numel() for p in actor.parameters()) + sum(
        p.numel() for p in critic.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Token Transformer WM: dim={args.transformer_dim}, kv_heads={args.transformer_heads}, "
          f"query_heads={args.transformer_query_heads}, layers={args.transformer_layers}, "
          f"ctx={args.transformer_context}, feat_dim={feat_dim}")
    print(f"Batch: {args.batch_size} ({args.batch_size_per_env}/env × {args.num_envs} envs), "
          f"length={args.batch_length}, train_ratio={args.train_ratio}")

    obs, _ = envs.reset(seed=args.seed)
    for i in range(args.num_envs):
        init_ongoing(i, obs[i])

    wm_state = world_model.initial(args.num_envs, device)
    prev_action = torch.zeros(args.num_envs, act_dim, device=device)
    prev_reward = torch.zeros(args.num_envs, device=device)
    is_first_flag = np.ones(args.num_envs, dtype=bool)

    global_step = 0
    train_steps = 0
    imagined_steps = 0
    imag_steps_per_train = args.batch_size * (min(args.imag_last, args.batch_length) if args.imag_last > 0 else args.batch_length) * args.imag_horizon
    start_time = time.time()
    steps_per_train = args.batch_size * args.batch_length

    while global_step < args.total_timesteps:
        # --- Collect one step ---
        prefilling = global_step < args.prefill_steps

        with torch.no_grad():
            if prefilling:
                actions_np = np.array([envs.single_action_space.sample()
                                       for _ in range(args.num_envs)])
                action_tensor = torch.tensor(actions_np, device=device, dtype=torch.float32)
            else:
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
                is_first_tensor = torch.tensor(is_first_flag, device=device, dtype=torch.bool)
                tokens = encoder(obs_tensor)
                wm_state, _ = world_model.observe_step(
                    wm_state, tokens, prev_action, prev_reward, is_first_tensor
                )
                feat = world_model.get_feat(wm_state)
                dist = actor(feat)
                action_tensor = dist.sample().float().clamp(-1, 1)
                actions_np = action_tensor.cpu().numpy()

        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions_np)
        dones = terminateds | truncateds
        global_step += args.num_envs

        for i in range(args.num_envs):
            if dones[i]:
                if "final_observation" in infos and infos["final_observation"][i] is not None:
                    final_obs = infos["final_observation"][i]
                else:
                    final_obs = next_obs[i]
                finish_ongoing(i, final_obs, actions_np[i], rewards[i], terminateds[i])
                init_ongoing(i, next_obs[i])

                if "final_info" in infos and infos["final_info"][i] is not None:
                    ep_info = infos["final_info"][i].get("episode", None)
                    if ep_info is not None:
                        ep_return = float(ep_info["r"])
                        ep_length = int(ep_info["l"])
                        print(f"global_step={global_step}, episodic_return={ep_return:.1f}, "
                              f"episodic_length={ep_length}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        total_steps = global_step + imagined_steps
                        writer.add_scalar("charts_total/episodic_return", ep_return, total_steps)
                        writer.add_scalar("charts_total/total_steps", total_steps, global_step)
            else:
                add_step(i, next_obs[i], actions_np[i], rewards[i], False, False)

        prev_action = action_tensor
        prev_reward = torch.tensor(rewards, device=device, dtype=torch.float32)
        is_first_flag = dones.copy()
        obs = next_obs

        # --- Training ---
        if not prefilling and replay.can_sample(args.batch_length):
            effective_step = global_step - args.prefill_steps
            target_train_steps = int(effective_step * args.train_ratio / steps_per_train)
            # Cap training burst (match dreamer4 ref)
            max_steps_per_cycle = max(1, int(args.num_envs * args.train_ratio / steps_per_train)) * 2
            steps_this_cycle = 0
            while train_steps < target_train_steps and steps_this_cycle < max_steps_per_cycle:
                stats = train_step()
                train_steps += 1
                imagined_steps += imag_steps_per_train
                steps_this_cycle += 1

                if train_steps % 100 == 0:
                    writer.add_scalar("losses/wm_loss", stats['wm_loss'], global_step)
                    writer.add_scalar("losses/rec_loss", stats['rec_loss'], global_step)
                    writer.add_scalar("losses/rew_loss", stats['rew_loss'], global_step)
                    writer.add_scalar("losses/con_loss", stats['con_loss'], global_step)
                    writer.add_scalar("losses/dyn_kl", stats['dyn_kl'], global_step)
                    writer.add_scalar("losses/rep_kl", stats['rep_kl'], global_step)
                    writer.add_scalar("losses/actor_loss", stats['actor_loss'], global_step)
                    writer.add_scalar("losses/critic_loss", stats['critic_loss'], global_step)
                    writer.add_scalar("imagination/entropy", stats['entropy'], global_step)
                    writer.add_scalar("imagination/returns", stats['returns'], global_step)
                    writer.add_scalar("imagination/ret_scale", stats['ret_scale'], global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("charts/train_steps", train_steps, global_step)
                    writer.add_scalar("charts/imagined_steps", imagined_steps, global_step)
                    writer.add_scalar("charts_total/SPS", int((global_step + imagined_steps) / (time.time() - start_time)), global_step)

        if global_step % 10000 < args.num_envs:
            sps = int(global_step / (time.time() - start_time))
            print(f"SPS: {sps}, global_step={global_step}, replay={replay.num_steps}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        state_dict = {
            'encoder': encoder.state_dict() if not isinstance(encoder, torch._dynamo.eval_frame.OptimizedModule) else encoder._orig_mod.state_dict(),
            'world_model': world_model.state_dict(),
            'decoder': decoder.state_dict() if not isinstance(decoder, torch._dynamo.eval_frame.OptimizedModule) else decoder._orig_mod.state_dict(),
            'reward_head': reward_head.state_dict() if not isinstance(reward_head, torch._dynamo.eval_frame.OptimizedModule) else reward_head._orig_mod.state_dict(),
            'continue_head': continue_head.state_dict() if not isinstance(continue_head, torch._dynamo.eval_frame.OptimizedModule) else continue_head._orig_mod.state_dict(),
            'actor': actor.state_dict(),
            'critic': critic.state_dict() if not isinstance(critic, torch._dynamo.eval_frame.OptimizedModule) else critic._orig_mod.state_dict(),
        }
        torch.save(state_dict, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
