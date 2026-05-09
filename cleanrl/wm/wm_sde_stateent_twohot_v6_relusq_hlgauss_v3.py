# PPO + WM + SDE + state entropy + relu^2 heads + HL-Gauss value/reward encoding.
#
# v3 changes over v2:
#   - Unified encode trunk. The trunk has a single base layout
#     `[obs_tokens, action, reward, agent, cls]`. Transition prediction extends
#     that same layout with per-frame eval-action tokens and a per-frame
#     transition token.
#   - Autoregressive WM training over horizon=16 (Dreamer4-aligned). The trunk's
#     temporal dimension is extended from CONTEXT_LEN to CONTEXT_LEN + dyn_horizon
#     during WM training. Frames 0..CONTEXT_LEN-1 are the seed warmup; frames
#     CONTEXT_LEN-1..T-2 are predict frames each carrying an eval-action token
#     for action a_{t+h} and a transition token whose readout predicts
#     (obs_{t+h+1}, reward_{t+h}, continue_{t+h}). Temporal attention is causal,
#     so transition_token at frame k only sees frames 0..k — the trunk handles
#     autoregression in a single forward pass under teacher forcing on real
#     ground-truth obs at each imagine frame.
#   - Per-frame prediction heads. The horizon-indexed MTP head from earlier
#     drafts is gone: `to_state_pred`, `reward_head_bins`, and `continue_head`
#     are now single-step per-frame heads, applied independently to each
#     transition token in the extended window. With the time dimension carrying
#     the horizon, no horizon multiplexing in the head output is needed.
#   - The dream loop still uses the single-window encode (CONTEXT_LEN sliding
#     window, h=0 prediction), now consuming the per-frame heads directly.
#
# v2 changes over v1:
#   1. SymExp HL-Gauss for both value and reward, support [-5, 5] in symlog space
#      (covers ~[-148, 148] in real return space, fixing critic saturation).
#   2. Per-call equal-weight loss rescaling (loss / |loss.detach()|) so each
#      category contributes ~unit gradient magnitude regardless of natural scale.
#   3. Imagined-rollout returns are computed once per iteration under no_grad,
#      freezing the bootstrap target during the inner SGD epochs without a
#      separate EMA target copy.
#   4. Larger WM backbone (embed=128, heads=8) and wider head MLPs. The
#      tokenizer backbone keeps the original v1 dims so existing pretrained
#      checkpoints still load; STSTSCLSBackbone is now parameterised.
#   5. Dynamics-value heads trained in the PPO phase on the already-encoded,
#      frozen latent (Dreamer4-style: RL learns heads, not the WM trunk).
#      WM supervision and imagination both use a 16-step horizon.
#   6. bf16 autocast around the major forward passes; SDPA dispatches to
#      FlashAttention.
#   7. AsyncVectorEnv for parallel env stepping at high num_envs.
#   8. WM aux objective now mirrors dreamer4: action-conditioned dynamics
#      outputs are decoded against the FROZEN tokenizer's next-obs embeddings
#      (state_pred_loss).
#      The collapse-prone transition_loss against the trainable trunk's own
#      next-bundle is dropped; sigreg goes with it (no longer needed).
#   9. DreamerV3-style asymmetric continue label smoothing
#      (target = (1 - terminal).clamp(max=gamma)) so the head's asymptote is
#      bounded on envs that never terminate.
#  10. Per-frame trunk anchor: dyn_to_obs head decodes WM dyn_flat against
#      the frozen tokenizer's CURRENT-step obs tokens.
#  11. Value head is HL-Gauss bin only — `get_value` returns the expected
#      value via softmax-over-bins. The MSE critic head was producing
#      unbounded `0.5*(newvalue - returns)^2` losses that dominated the
#      shared trunk's gradient (KL exploded to >2 in iter 1 with
#      `clip_vloss=False`). Softmax CE has bounded gradients per logit, so
#      no value clipping is needed and none is implemented.
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cleanrl.shared.hl_gauss import HLGaussSupport


# --- WM backbone (trainable, larger capacity) ---
# 4x v1's embed (32 → 128); 8 heads (head_dim=16) for RoPE comfort; FFN_MULT=4.
WM_EMBED_DIM = 128
WM_NUM_HEADS = 8
WM_FFN_MULT = 4
WM_NUM_SPATIAL_BLOCKS = 3
WM_NUM_TEMPORAL_BLOCKS = 2

# --- Tokenizer backbone (frozen target; keeps v1 dims to load existing ckpts) ---
TOK_EMBED_DIM = 32
TOK_NUM_HEADS = 4
TOK_FFN_MULT = 2
TOK_NUM_SPATIAL_BLOCKS = 3
TOK_NUM_TEMPORAL_BLOCKS = 2

CONTEXT_LEN = 5
DYN_TOKEN_NAMES = ['dyn0', 'dyn1', 'dyn2', 'dyn3']
WM_CLS_NAMES = DYN_TOKEN_NAMES
DYN_TOKEN_COUNT = len(DYN_TOKEN_NAMES)
AGENT_TOKEN_COUNT = 2
WM_DYN_FLAT_DIM = WM_EMBED_DIM * DYN_TOKEN_COUNT
WM_AGENT_FLAT_DIM = WM_EMBED_DIM * AGENT_TOKEN_COUNT

# Wider head MLPs; with the bigger trunk the readouts need to keep up.
ACTOR_HIDDEN = 256
CRITIC_HIDDEN = 256
DYN_VALUE_HIDDEN = 256
REWARD_HIDDEN = 512
CONTINUE_HIDDEN = 256

STD_MIN = 0.05
WM_COEF = 1.0
DYNAMICS_VALUE_BINS_COEF = 0.5
DYNAMICS_SHAPE_COEF = 0.5
STATE_PRED_LOSS_COEF = 0.1
SQUASH_EPS = 1e-6

def equal_weight(loss, eps=1e-6):
    """Rescale a scalar loss to ~unit magnitude in value space.

    `loss / |loss.detach()|` standardizes each contributor so the sum across
    heterogeneous categories has equal-weight value semantics; gradient
    direction is unchanged because the divisor is detached. Sentinel zeros
    pass through cleanly (0 / eps = 0) — no running state to corrupt.
    """
    return loss / loss.detach().abs().clamp_min(eps)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    compile: bool = False
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 64
    rollout_batch_size: int = 2048
    # Derived from rollout_batch_size // num_envs after CLI parsing.
    num_steps: int = 0
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 1
    norm_adv: bool = True
    spo_eps_low: float = 0.40
    spo_eps_high: float = 0.56
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.15
    # SymExp HL-Gauss: bin centers are linear in symlog space; targets are
    # symlogged before projection and outputs symexp'd. Support [-5, 5] in
    # symlog space covers ~[-148, 148] in real return / reward space, which
    # is wide enough that lambda-returns over a 15-step horizon at gamma=0.99
    # don't saturate the support.
    num_bins: int = 51
    v_min: float = -5.0
    v_max: float = 5.0
    sigma_ratio: float = 0.75
    use_symlog: bool = True
    reward_num_bins: int = 51
    reward_v_min: float = -5.0
    reward_v_max: float = 5.0
    reward_sigma_ratio: float = 0.75
    reward_use_symlog: bool = True
    tokenizer_path: str = ""
    tokenizer_checkpoint_prefix: str = ""
    dyn_horizon: int = 16
    wm_batch_size: int = 512
    wm_update_epochs: int = 1
    imagination_horizon: int = 16
    imagine_start_step: int = 0
    imagine_update_epochs: int = 1
    imagine_actor_coef: float = 1.0
    imagine_critic_coef: float = 1.0
    imagine_actor_ent_coef: float = 0.0
    # Flattened imagined transition samples optimized per dream batch.
    dream_batch_size: int = 2048
    dream_batches: int = 8

    batch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Tanh-squashed Gaussian already bounds actions.
        env = gym.wrappers.NormalizeObservation(env)
        clipped_obs_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=clipped_obs_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env.observation_space = clipped_obs_space
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=None, bias_const=0.0):
    fan_in = layer.weight.shape[1]
    if std is None:
        std = 1.0 / fan_in**0.5
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def orthogonal_layer_init(layer, std=None, bias_const=0.0):
    fan_in = layer.weight.shape[1]
    if std is None:
        std = 1.0 / fan_in**0.5
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ReluSq(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def relusq_mlp(in_dim, out_dim, hidden_dim, out_std=None):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_dim)),
        ReluSq(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        ReluSq(),
        layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
    )


def relusq_mlp_orthogonal(in_dim, out_dim, hidden_dim, out_std=None):
    return nn.Sequential(
        orthogonal_layer_init(nn.Linear(in_dim, hidden_dim)),
        ReluSq(),
        orthogonal_layer_init(nn.Linear(hidden_dim, hidden_dim)),
        ReluSq(),
        orthogonal_layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
    )


def relusq_mlp_prenorm_out(in_dim, out_dim, hidden_dim, out_std=None):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_dim)),
        ReluSq(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        ReluSq(),
        RMSNorm(hidden_dim),
        layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(dtype=dtype)


def build_rope_cache(seq_len, head_dim, device):
    assert head_dim % 2 == 0
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult=2, init_scale=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_pre_norm = RMSNorm(dim)
        self.attn_post_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.ffn_pre_norm = RMSNorm(dim)
        self.ffn_post_norm = RMSNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_value = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=False)
        self.ffn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        for module in [self.q_proj, self.k_proj, self.v_proj, self.ffn_gate, self.ffn_value]:
            fan_in = module.weight.shape[1]
            std = 1.0 / fan_in**0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        fan_in = self.out_proj.weight.shape[1]
        std = 0.1 * init_scale / fan_in**0.5
        nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-2 * std, b=2 * std)
        fan_in = self.ffn_out.weight.shape[1]
        std = init_scale / fan_in**0.5
        nn.init.trunc_normal_(self.ffn_out.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x, x0, rope_cos=None, rope_sin=None, attn_mask=None, is_causal=False):
        batch, steps, width = x.shape
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0].view(1, 1, width) * x + mix[1].view(1, 1, width) * x0

        h = self.attn_pre_norm(x)
        q = self.q_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn = attn.transpose(1, 2).reshape(batch, steps, width)
        attn_out = self.attn_post_norm(self.out_proj(attn))
        x = x + self.attn_scale.to(dtype=x.dtype).view(1, 1, width) * attn_out

        h = self.ffn_pre_norm(x)
        ffn = F.silu(self.ffn_gate(h)) * self.ffn_value(h)
        ffn_out = self.ffn_post_norm(self.ffn_out(ffn))
        x = x + self.ffn_scale.to(dtype=x.dtype).view(1, 1, width) * ffn_out
        return x


class STSTSCLSBackbone(nn.Module):
    def __init__(
        self,
        obs_dim,
        context_len,
        cls_names,
        embed_dim,
        num_heads,
        ffn_mult,
        num_spatial_blocks,
        num_temporal_blocks,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_len = context_len
        self.cls_names = list(cls_names)
        self.num_cls_tokens = len(self.cls_names)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.num_spatial_blocks = num_spatial_blocks
        self.num_temporal_blocks = num_temporal_blocks
        for i, name in enumerate(self.cls_names):
            setattr(self, f'{name}_cls_index', obs_dim + i)

        self.value_proj = layer_init(nn.Linear(1, embed_dim), std=1.0)
        self.input_norm = RMSNorm(embed_dim)
        self.dim_id_embed = nn.Embedding(obs_dim, embed_dim)
        self.register_buffer("dim_indices", torch.arange(obs_dim))

        cls_std = 1.0 / embed_dim**0.5
        self.cls_params = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim)) for _ in range(self.num_cls_tokens)
        ])
        for p in self.cls_params:
            nn.init.trunc_normal_(p, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

        init_scale = 1.0 / (2 * (num_spatial_blocks + num_temporal_blocks)) ** 0.5
        self.s_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, ffn_mult, init_scale) for _ in range(num_spatial_blocks)]
        )
        self.t_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, ffn_mult, init_scale) for _ in range(num_temporal_blocks)]
        )
        self.final_norm = RMSNorm(embed_dim)

        head_dim = embed_dim // num_heads
        temporal_cos, temporal_sin = build_rope_cache(context_len, head_dim, torch.device("cpu"))
        self.register_buffer("temporal_cos", temporal_cos)
        self.register_buffer("temporal_sin", temporal_sin)

    def _spatial(self, tokens, tokens0, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.reshape(batch * time_steps, slots, width)
        x0 = tokens0.reshape(batch * time_steps, slots, width)
        x = block(x, x0)
        return x.reshape(batch, time_steps, slots, width)

    def _temporal(self, tokens, tokens0, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x0 = tokens0.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x = block(x, x0, rope_cos=self.temporal_cos, rope_sin=self.temporal_sin)
        x = x.reshape(batch, slots, time_steps, width).permute(0, 2, 1, 3)
        return x

    def forward(self, obs_seq, return_obs_tokens=False, return_all_obs_tokens=False):
        batch, time_steps, _ = obs_seq.shape
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1))
        dtype = obs_tokens.dtype
        obs_tokens = obs_tokens + self.dim_id_embed(self.dim_indices).to(dtype=dtype).view(
            1, 1, self.obs_dim, self.embed_dim
        )

        cls_tokens = torch.stack(list(self.cls_params), dim=0).to(dtype=dtype)
        cls_tokens = cls_tokens.view(1, 1, self.num_cls_tokens, self.embed_dim).expand(batch, time_steps, -1, -1)
        tokens = torch.cat([obs_tokens, cls_tokens], dim=2)
        tokens = self.input_norm(tokens)
        tokens0 = tokens

        # Interleave spatial / temporal blocks; assumes num_spatial_blocks ==
        # num_temporal_blocks + 1 (3 + 2 here). Generic loop supports both
        # backbone configurations used in this file.
        s_iter = iter(self.s_blocks)
        t_iter = iter(self.t_blocks)
        for s_block in s_iter:
            tokens = self._spatial(tokens, tokens0, s_block)
            t_block = next(t_iter, None)
            if t_block is not None:
                tokens = self._temporal(tokens, tokens0, t_block)
        tokens = self.final_norm(tokens)

        out = {}
        for i, name in enumerate(self.cls_names):
            out[name] = tokens[:, -1, self.obs_dim + i]
        if return_obs_tokens:
            out['obs_tokens'] = tokens[:, -1, :self.obs_dim]
        if return_all_obs_tokens:
            out['all_obs_tokens'] = tokens[:, :, :self.obs_dim]
        return out


class TokenActionSTSBackbone(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        context_len,
        cls_names,
        agent_token_count,
        input_token_dim,
        embed_dim,
        num_heads,
        ffn_mult,
        num_spatial_blocks,
        num_temporal_blocks,
        max_temporal_steps=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        # RoPE cache covers up to max_temporal_steps; defaults to context_len
        # for compat. Set larger by callers that run extended-window training.
        self.max_temporal_steps = max_temporal_steps if max_temporal_steps is not None else context_len
        self.cls_names = list(cls_names)
        self.num_cls_tokens = len(self.cls_names)
        self.num_agent_tokens = agent_token_count
        self.embed_dim = embed_dim
        self.reward_slots = 1
        for i, name in enumerate(self.cls_names):
            setattr(self, f'{name}_cls_index', obs_dim + action_dim + self.reward_slots + self.num_agent_tokens + i)

        self.obs_token_proj = layer_init(nn.Linear(input_token_dim, embed_dim))
        self.input_norm = RMSNorm(embed_dim)
        self.obs_dim_embed = nn.Embedding(obs_dim, embed_dim)
        self.action_proj = layer_init(nn.Linear(1, embed_dim))
        self.action_dim_embed = nn.Embedding(action_dim, embed_dim)
        self.reward_proj = layer_init(nn.Linear(1, embed_dim))
        self.reward_embed = nn.Parameter(torch.empty(embed_dim))
        self.agent_params = nn.Parameter(torch.empty(agent_token_count, embed_dim))
        self.eval_action_embed = nn.Parameter(torch.empty(embed_dim))
        self.transition_param = nn.Parameter(torch.empty(embed_dim))
        self.register_buffer("obs_dim_indices", torch.arange(obs_dim))
        self.register_buffer("action_dim_indices", torch.arange(action_dim))

        cls_std = 1.0 / embed_dim**0.5
        self.cls_params = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim)) for _ in range(self.num_cls_tokens)
        ])
        for p in self.cls_params:
            nn.init.trunc_normal_(p, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.agent_params, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.eval_action_embed, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.transition_param, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.obs_dim_embed.weight, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.action_dim_embed.weight, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.reward_embed, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

        init_scale = 1.0 / (2 * (num_spatial_blocks + num_temporal_blocks)) ** 0.5
        self.s_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, ffn_mult, init_scale) for _ in range(num_spatial_blocks)]
        )
        self.t_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, ffn_mult, init_scale) for _ in range(num_temporal_blocks)]
        )
        self.final_norm = RMSNorm(embed_dim)

        head_dim = embed_dim // num_heads
        temporal_cos, temporal_sin = build_rope_cache(self.max_temporal_steps, head_dim, torch.device("cpu"))
        self.register_buffer("temporal_cos", temporal_cos)
        self.register_buffer("temporal_sin", temporal_sin)

    def _spatial(self, tokens, tokens0, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.reshape(batch * time_steps, slots, width)
        x0 = tokens0.reshape(batch * time_steps, slots, width)
        x = block(x, x0)
        return x.reshape(batch, time_steps, slots, width)

    def _temporal(self, tokens, tokens0, block, is_causal=True):
        batch, time_steps, slots, width = tokens.shape
        # Slice RoPE cache to the active T (cache is sized for the extended
        # window; single-window mode just uses the first context_len entries).
        cos = self.temporal_cos[:time_steps]
        sin = self.temporal_sin[:time_steps]
        x = tokens.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x0 = tokens0.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x = block(x, x0, rope_cos=cos, rope_sin=sin, is_causal=is_causal)
        x = x.reshape(batch, slots, time_steps, width).permute(0, 2, 1, 3)
        return x

    def _run(self, tokens, causal_temporal=True):
        tokens = self.input_norm(tokens)
        tokens0 = tokens
        s_iter = iter(self.s_blocks)
        t_iter = iter(self.t_blocks)
        for s_block in s_iter:
            tokens = self._spatial(tokens, tokens0, s_block)
            t_block = next(t_iter, None)
            if t_block is not None:
                tokens = self._temporal(tokens, tokens0, t_block, is_causal=causal_temporal)
        return self.final_norm(tokens)

    def _action_tokens(self, action_history):
        action_history = action_history.to(dtype=self.action_proj.weight.dtype)
        action_tokens = self.action_proj(action_history.unsqueeze(-1))
        dtype = action_tokens.dtype
        return action_tokens + self.action_dim_embed(self.action_dim_indices).to(dtype=dtype).view(
            1, 1, self.action_dim, self.embed_dim
        )

    def _eval_action_tokens(self, transition_action, time_steps):
        """Build per-frame eval-action tokens.

        `transition_action` is either:
          - (B, action_dim): single action placed at the last frame (single-window
            mode used by the dream loop and the rollout-time head readouts);
          - (B, T, action_dim) with T == time_steps: per-frame eval actions
            covering the imagine extension during WM training.
        Frames with no eval action carry zeros (the action proj turns those into
        learned-bias-only tokens that the trunk can downweight).
        """
        batch = transition_action.shape[0]
        if transition_action.ndim == 2:
            action_history = torch.zeros(
                batch,
                time_steps,
                self.action_dim,
                device=transition_action.device,
                dtype=transition_action.dtype,
            )
            action_history[:, -1] = transition_action.reshape(batch, self.action_dim)
        elif transition_action.ndim == 3:
            assert transition_action.shape[1] == time_steps, (
                f"transition_action time dim {transition_action.shape[1]} != time_steps {time_steps}"
            )
            action_history = transition_action
        else:
            raise ValueError(
                f"transition_action must be (B, A) or (B, T, A); got shape {tuple(transition_action.shape)}"
            )
        action_tokens = self._action_tokens(action_history)
        return action_tokens + self.eval_action_embed.to(dtype=action_tokens.dtype).view(
            1,
            1,
            1,
            self.embed_dim,
        )

    def _reward_tokens(self, reward_history):
        reward_history = reward_history.to(dtype=self.reward_proj.weight.dtype)
        reward_tokens = self.reward_proj(reward_history.unsqueeze(-1).unsqueeze(-1))
        return reward_tokens + self.reward_embed.to(dtype=reward_tokens.dtype).view(1, 1, 1, self.embed_dim)

    def _agent_tokens(self, batch, time_steps, dtype=None):
        tokens = self.agent_params.view(1, 1, self.num_agent_tokens, self.embed_dim).expand(
            batch,
            time_steps,
            -1,
            -1,
        )
        if dtype is not None:
            tokens = tokens.to(dtype=dtype)
        return tokens

    def encode_obs_tokens(
        self,
        obs_tokens,
        action_history=None,
        reward_history=None,
        transition_action=None,
        return_obs_tokens=False,
        return_transition_token=False,
        return_transition_seq=False,
        cls_index=-1,
    ):
        batch, time_steps, obs_dim, _ = obs_tokens.shape
        assert time_steps <= self.max_temporal_steps, (
            f"time_steps={time_steps} exceeds max_temporal_steps={self.max_temporal_steps}"
        )
        assert obs_dim == self.obs_dim
        if action_history is None:
            action_history = torch.zeros(
                batch,
                time_steps,
                self.action_dim,
                device=obs_tokens.device,
                dtype=obs_tokens.dtype,
            )
        if reward_history is None:
            reward_history = torch.zeros(
                batch,
                time_steps,
                device=obs_tokens.device,
                dtype=obs_tokens.dtype,
            )

        obs_tokens = self.obs_token_proj(obs_tokens)
        dtype = obs_tokens.dtype
        obs_tokens = obs_tokens + self.obs_dim_embed(self.obs_dim_indices).to(dtype=dtype).view(
            1, 1, self.obs_dim, self.embed_dim
        )
        action_tokens = self._action_tokens(action_history)
        reward_tokens = self._reward_tokens(reward_history)
        agent_tokens = self._agent_tokens(batch, time_steps, dtype=dtype)
        cls_tokens = torch.stack(list(self.cls_params), dim=0).to(dtype=dtype)
        cls_tokens = cls_tokens.view(1, 1, self.num_cls_tokens, self.embed_dim).expand(batch, time_steps, -1, -1)
        token_parts = [obs_tokens, action_tokens, reward_tokens, agent_tokens, cls_tokens]
        transition_start = None
        if transition_action is not None:
            eval_action_tokens = self._eval_action_tokens(transition_action, time_steps).to(dtype=dtype)
            transition_start = sum(part.shape[2] for part in token_parts) + self.action_dim
            transition_token = self.transition_param.to(dtype=dtype).view(1, 1, 1, self.embed_dim).expand(
                batch,
                time_steps,
                -1,
                -1,
            )
            token_parts.extend([eval_action_tokens, transition_token])
        tokens = torch.cat(token_parts, dim=2)
        tokens = self._run(tokens, causal_temporal=True)

        out = {}
        agent_start = self.obs_dim + self.action_dim + self.reward_slots
        # `cls_index` selects which time index the agent / cls / obs / single
        # transition readouts come from. Default -1 (last frame) matches the
        # rollout / dream / single-window callers; extended-window WM training
        # passes cls_index=context_len-1 so the trunk-anchor readout comes
        # from the end of the seed (the current state) rather than the end of
        # the imagine extension.
        out['agent'] = tokens[:, cls_index, agent_start:agent_start + self.num_agent_tokens]
        for i, name in enumerate(self.cls_names):
            out[name] = tokens[:, cls_index, agent_start + self.num_agent_tokens + i]
        if return_obs_tokens:
            out['obs_tokens'] = tokens[:, cls_index, :self.obs_dim]
        if return_transition_token:
            if transition_start is None:
                raise ValueError("return_transition_token requires transition_action.")
            out['transition'] = tokens[:, cls_index, transition_start]
        if return_transition_seq:
            if transition_start is None:
                raise ValueError("return_transition_seq requires transition_action.")
            out['transition_seq'] = tokens[:, :, transition_start]
        return out

class Agent(nn.Module):
    def __init__(
        self,
        envs,
        num_envs,
        num_bins,
        reward_num_bins,
        dyn_horizon,
        tokenizer_path="",
        tokenizer_checkpoint_prefix="",
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.obs_dim = obs_dim
        # Tokenizer is the frozen target; state-pred targets live in TOK space.
        self.tok_obs_embed_dim = obs_dim * TOK_EMBED_DIM
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN
        self.dyn_horizon = dyn_horizon
        self.reward_num_bins = reward_num_bins

        # Extended-window WM training spans the seed (CONTEXT_LEN) plus
        # dyn_horizon imagine frames so frame k carries (obs_{t+k-CONTEXT_LEN+1},
        # action_{t+k-CONTEXT_LEN}, reward_{t+k-CONTEXT_LEN}) under teacher
        # forcing. Causal temporal attention turns this into a single-pass
        # autoregressive forward over the full horizon.
        self.wm_max_temporal_steps = self.context_len + dyn_horizon
        self.wm_backbone = TokenActionSTSBackbone(
            obs_dim,
            action_dim,
            self.context_len,
            WM_CLS_NAMES,
            AGENT_TOKEN_COUNT,
            input_token_dim=TOK_EMBED_DIM,
            embed_dim=WM_EMBED_DIM,
            num_heads=WM_NUM_HEADS,
            ffn_mult=WM_FFN_MULT,
            num_spatial_blocks=WM_NUM_SPATIAL_BLOCKS,
            num_temporal_blocks=WM_NUM_TEMPORAL_BLOCKS,
            max_temporal_steps=self.wm_max_temporal_steps,
        )
        self.tokenizer_backbone = STSTSCLSBackbone(
            obs_dim, self.context_len, WM_CLS_NAMES,
            embed_dim=TOK_EMBED_DIM,
            num_heads=TOK_NUM_HEADS,
            ffn_mult=TOK_FFN_MULT,
            num_spatial_blocks=TOK_NUM_SPATIAL_BLOCKS,
            num_temporal_blocks=TOK_NUM_TEMPORAL_BLOCKS,
        )
        if tokenizer_path:
            self._load_tokenizer_checkpoint(tokenizer_path, tokenizer_checkpoint_prefix)
        else:
            raise ValueError("This variant requires --tokenizer-path from tokenizer pretraining.")
        self.tokenizer_backbone.eval().requires_grad_(False)

        # State-dependent (mean, std) head with a softplus reparameterization.
        # actor_head emits 2·action_dim outputs; the second slice maps to
        # `std = softplus(raw) + STD_MIN`. softplus is a smooth R→R+ bijection
        # so std is naturally floored away from 0 without any clamp on raw —
        # this is what keeps the per-state (z-mean)/std² PPO gradient bounded
        # (the failure mode that breaks the unclamped exp(.) parameterization
        # under SDE: each transition can collapse its own std and explode its
        # own gradient). Init: small (out_std=0.01) weights + zero bias →
        # raw ≈ 0 → std ≈ softplus(0) + STD_MIN = log(2) + STD_MIN ≈ 0.74.
        self.actor_head = relusq_mlp_orthogonal(WM_AGENT_FLAT_DIM, 2 * action_dim, ACTOR_HIDDEN, out_std=0.01)

        # HL-Gauss bin critic — trained by softmax CE; expected-value
        # readout via `to_value(critic_bins(...))` is naturally bounded
        # by the support range, so no MSE head and no value clipping
        # are needed (a separate MSE head with unbounded loss would
        # dominate the shared trunk's gradient).
        self.critic_bins = relusq_mlp_orthogonal(WM_AGENT_FLAT_DIM, num_bins, CRITIC_HIDDEN, out_std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

        # Dreamer4-style transition readout: the WM backbone appends an
        # eval-action token plus a learned transition token at every frame.
        # Heads are per-frame single-step (predict the NEXT obs/reward/continue
        # from each frame's transition token); the time dimension carries the
        # horizon, so no horizon multiplexing in the head output is needed.
        self.to_state_pred = nn.Sequential(
            RMSNorm(WM_EMBED_DIM),
            nn.Linear(WM_EMBED_DIM, self.tok_obs_embed_dim * 2),
        )

        # Per-frame trunk anchor: WM dyn_flat at the current step is decoded
        # against the FROZEN tokenizer's current-step obs tokens.
        self.dyn_to_obs = nn.Sequential(
            RMSNorm(WM_DYN_FLAT_DIM),
            nn.Linear(WM_DYN_FLAT_DIM, self.tok_obs_embed_dim),
        )

        # Dynamics-value HL-Gauss bin head — trained on real GAE returns,
        # then used as the value bootstrap/readout inside imagined rollouts.
        # The dream loop computes lambda returns once per iteration under
        # no_grad, freezing the bootstrap target for that iteration's SGD —
        # so a separate EMA target copy isn't needed.
        self.dynamics_value_bins = relusq_mlp(WM_AGENT_FLAT_DIM, num_bins, DYN_VALUE_HIDDEN, out_std=1.0)
        self.reward_head_bins = relusq_mlp(WM_EMBED_DIM, reward_num_bins, REWARD_HIDDEN)
        self.continue_head = relusq_mlp(WM_EMBED_DIM, 1, CONTINUE_HIDDEN)
        nn.init.constant_(self.continue_head[-1].bias, 5.0)

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))
        self.register_buffer("action_history", torch.zeros(num_envs, self.context_len, action_dim))
        self.register_buffer("reward_history", torch.zeros(num_envs, self.context_len))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
            self.action_history.zero_()
            self.reward_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0
            self.action_history[env_mask] = 0.0
            self.reward_history[env_mask] = 0.0

    def update_history(self, obs, action=None, reward=None):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)
        if action is None:
            action = torch.zeros(self.obs_history.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype)
        else:
            action = action.to(dtype=self.action_history.dtype)
        if reward is None:
            reward = torch.zeros(self.obs_history.shape[0], device=obs.device, dtype=self.reward_history.dtype)
        else:
            reward = reward.to(dtype=self.reward_history.dtype)
        self.action_history = torch.cat([self.action_history[:, 1:], action.unsqueeze(1)], dim=1)
        self.reward_history = torch.cat([self.reward_history[:, 1:], reward.unsqueeze(1)], dim=1)

    def _load_tokenizer_checkpoint(self, tokenizer_path, tokenizer_checkpoint_prefix=""):
        path = Path(tokenizer_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            for key in ("tokenizer_backbone", "state_dict", "model", "agent"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Tokenizer checkpoint {path} did not contain a state dict.")

        target_keys = set(self.tokenizer_backbone.state_dict().keys())
        auto_prefixes = (
            "",
            "tokenizer_backbone.",
            "wm_backbone.",
            "module.tokenizer_backbone.",
            "module.wm_backbone.",
            "agent.tokenizer_backbone.",
            "agent.wm_backbone.",
        )
        if tokenizer_checkpoint_prefix:
            prefix = tokenizer_checkpoint_prefix
            if prefix in ("root", ".", "<root>"):
                prefixes = ("",)
            else:
                prefixes = (prefix if prefix.endswith(".") else f"{prefix}.",)
        else:
            prefixes = auto_prefixes

        candidates = []
        for prefix in prefixes:
            stripped = {}
            for key, value in checkpoint.items():
                if prefix:
                    if not key.startswith(prefix):
                        continue
                    stripped_key = key[len(prefix):]
                else:
                    stripped_key = key
                if stripped_key in target_keys:
                    stripped[stripped_key] = value
            if stripped:
                candidates.append((prefix, stripped))

        if not candidates:
            raise RuntimeError(f"Tokenizer checkpoint {path} had no compatible backbone keys.")

        best_count = max(len(state) for _, state in candidates)
        best_candidates = [(prefix, state) for prefix, state in candidates if len(state) == best_count]
        if len(best_candidates) > 1 and not tokenizer_checkpoint_prefix:
            prefixes_found = ", ".join(repr(prefix) for prefix, _ in best_candidates)
            raise RuntimeError(
                f"Tokenizer checkpoint {path} is ambiguous across prefixes {prefixes_found}; "
                "set --tokenizer-checkpoint-prefix explicitly."
            )

        best_prefix, best_state = best_candidates[0]
        missing, unexpected = self.tokenizer_backbone.load_state_dict(best_state, strict=False)
        parameter_keys = set(dict(self.tokenizer_backbone.named_parameters()).keys())
        missing_parameters = [key for key in missing if key in parameter_keys]
        if missing_parameters:
            raise RuntimeError(
                f"Tokenizer checkpoint {path} is missing parameter tensors: "
                f"{missing_parameters[:8]}"
            )
        print(
            f"Loaded tokenizer target from {path} "
            f"({len(best_state)} tensors, prefix='{best_prefix}', "
            f"missing={len(missing)}, unexpected={len(unexpected)})."
        )

    def _encode_wm(self, obs_seq, action_seq=None, reward_seq=None, return_obs_tokens=False):
        with bf16_autocast(obs_seq.device):
            tokenizer = self._encode_tokenizer(obs_seq, return_all_obs_tokens=True)
            return self.wm_backbone.encode_obs_tokens(
                tokenizer['all_obs_tokens'],
                action_history=action_seq,
                reward_history=reward_seq,
                return_obs_tokens=return_obs_tokens,
            )

    def _encode_tokenizer(self, obs_seq, return_obs_tokens=False, return_all_obs_tokens=False):
        self.tokenizer_backbone.eval()
        with torch.no_grad(), bf16_autocast(obs_seq.device):
            return self.tokenizer_backbone(
                obs_seq,
                return_obs_tokens=return_obs_tokens,
                return_all_obs_tokens=return_all_obs_tokens,
            )

    def _dyn_bundle_from_cls(self, cls):
        return torch.stack([cls[name] for name in DYN_TOKEN_NAMES], dim=1)

    def _dyn_flat_from_cls(self, cls):
        return self._dyn_bundle_from_cls(cls).flatten(1)

    def _agent_bundle_from_cls(self, cls):
        return cls['agent']

    def _agent_flat_from_cls(self, cls):
        return self._agent_bundle_from_cls(cls).flatten(1)

    def _transition_token_from_obs_tokens(self, obs_tokens, action_history, reward_history, action):
        cls = self.wm_backbone.encode_obs_tokens(
            obs_tokens,
            action_history=action_history,
            reward_history=reward_history,
            transition_action=action,
            return_transition_token=True,
        )
        return cls['transition']

    def _transition_predictions_from_token(self, transition_token):
        """Apply per-frame heads to one or many transition tokens.

        Accepts either:
          - (B, D): single-frame readout used by the dream loop;
          - (B, T, D): per-frame readout used by extended-window WM training.
        Returns predictions of corresponding rank: (B, ...) or (B, T, ...).
        Each transition token at frame k predicts (obs_{k+1}, reward_k,
        continue_k); the caller is responsible for selecting / masking the
        valid prediction frames.
        """
        pred = self.to_state_pred(transition_token)  # (..., 2 * tok_obs_embed_dim)
        reward_logits = self.reward_head_bins(transition_token)  # (..., reward_num_bins)
        continue_logits = self.continue_head(transition_token).squeeze(-1)  # (...)
        return pred, reward_logits, continue_logits

    def _predict_transition_from_obs_tokens(self, obs_tokens, action_history, reward_history, action):
        transition_token = self._transition_token_from_obs_tokens(
            obs_tokens,
            action_history,
            reward_history,
            action,
        )
        return self._transition_predictions_from_token(transition_token)

    def _encode_next_obs_window(self, obs_window, action_history, reward_history):
        """Encode a window whose final obs slot is a forward prediction.

        Same trunk path as a real-frame encode; the predicted reward at the
        final slot is already available before encoding s_{t+1}.
        """
        return self.wm_backbone.encode_obs_tokens(
            obs_window,
            action_history=action_history,
            reward_history=reward_history,
        )

    def _actor_mean_std(self, agent_flat):
        """State-dependent (mean, std). std = softplus(raw) + STD_MIN — smooth, floored, no clamp."""
        mean, raw_std = self.actor_head(agent_flat).chunk(2, dim=-1)
        std = F.softplus(raw_std) + STD_MIN
        return mean, std

    def _u_to_action(self, u):
        center = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return center + half_range * u

    def _action_to_u(self, action):
        center = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return ((action - center) / half_range).clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)

    def _squashed_log_prob_entropy(self, mean, std, z=None):
        """log_prob of a squashed Normal action.

        At rollout time `z` is None — sample a fresh pre-squash latent.
        At PPO update time the rollout's `z` is passed back in directly. The
        seemingly-equivalent `z = atanh(action_to_u(action))` round-trip is
        numerically broken for saturated actions: `tanh(z)` clips to
        ±(1 - SQUASH_EPS) and the inverse can't recover the original large |z|,
        so log_prob silently diverges from the rollout value.

        The squash correction uses the SAC-stable identity
        `log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2z))` to avoid the
        catastrophic cancellation that wrecks `log(1 - u^2)` once |z| ≳ 4 —
        especially in bf16, where 1 - tanh(5)^2 underflows to zero. log_prob is
        accumulated in fp32 to keep it deterministic across autocast contexts.
        """
        with torch.amp.autocast(mean.device.type, enabled=False):
            mean_f = mean.float()
            std_f = std.float()
            normal = Normal(mean_f, std_f)
            if z is None:
                z = normal.rsample()
            else:
                z = z.float()
            u = torch.tanh(z)
            action = self._u_to_action(u)
            log_prob_z = normal.log_prob(z).sum(-1)
            squash_correction = (
                2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
            ).sum(-1)
            log_prob = log_prob_z - squash_correction
            entropy = normal.entropy().sum(-1)
        return action, log_prob, entropy, z

    def to_value(self, logits):
        """Expected value from HL-Gauss bin logits — set self.hl_support after construction."""
        return self.hl_support.to_scalar(logits)

    def get_value(self, obs_seq, action_seq=None, reward_seq=None):
        cls = self._encode_wm(obs_seq, action_seq, reward_seq)
        return self.to_value(self.critic_bins(self._agent_flat_from_cls(cls).detach()))

    def get_action_and_value(self, obs_seq, action_seq=None, reward_seq=None):
        # Actor, critic_bins, and dynamics_value_bins all read a detached
        # view of the WM trunk (Dreamer4 default — only train heads from
        # PPO/value losses). The trunk is shaped exclusively by the WM aux
        # losses (state-pred / reward / continue / dyn_to_obs / transition),
        # which keeps PPO from staling its own ratios mid-epoch by moving
        # the latent representation under stored old log-probs.
        wm_cls = self._encode_wm(obs_seq, action_seq, reward_seq)
        dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
        agent_bundle = self._agent_bundle_from_cls(wm_cls)
        dyn_flat = dyn_bundle.flatten(1)
        agent_flat = agent_bundle.flatten(1)
        agent_flat_d = agent_flat.detach()
        action_mean, action_std = self._actor_mean_std(agent_flat_d)
        action, log_prob, entropy, z = self._squashed_log_prob_entropy(action_mean, action_std)
        return action, log_prob, entropy, self.to_value(self.critic_bins(agent_flat_d)), dyn_bundle, dyn_flat, agent_bundle, agent_flat, z

    def get_all_for_update(self, obs_seq, action_seq, reward_seq, z):
        # Dreamer4-style RL boundary: replay the WM encoder without building a
        # trainable trunk graph, then learn actor/value heads from frozen latents.
        with torch.no_grad():
            wm_cls = self._encode_wm(obs_seq, action_seq, reward_seq)
            dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
            agent_flat_d = self._agent_flat_from_cls(wm_cls)
        action_mean, action_std = self._actor_mean_std(agent_flat_d)
        _, log_prob, entropy, _ = self._squashed_log_prob_entropy(action_mean, action_std, z=z)
        value_logits = self.critic_bins(agent_flat_d)
        return (
            log_prob,
            entropy,
            self.to_value(value_logits),                # expected value from bins
            value_logits,                               # bin logits for HL-Gauss CE
            agent_flat_d,                               # detached — all PPO agent heads consume detached trunk
        )

    def get_imagined_action_and_value(self, agent_bundle, z=None):
        # Both actor and dynamics_value_bins read a detached view — Dreamer4
        # default: only train heads from PPO/value losses. agent_bundle is itself
        # a detached imagined latent; we re-detach defensively.
        agent_flat = agent_bundle.flatten(1).detach()
        action_mean, action_std = self._actor_mean_std(agent_flat)
        action, log_prob, entropy, z_out = self._squashed_log_prob_entropy(action_mean, action_std, z=z)
        value_logits = self.dynamics_value_bins(agent_flat)
        return action, log_prob, entropy, self.hl_support.to_scalar(value_logits), value_logits, z_out

    def get_imagined_value(self, agent_bundle, hl_support):
        return hl_support.to_scalar(self.get_imagined_value_logits(agent_bundle))

    def get_imagined_value_logits(self, agent_bundle):
        # Defensive detach for symmetry with get_imagined_action_and_value —
        # imagined latents are stored detached, but we re-detach so the head
        # never accidentally backprops through the trunk via this path.
        return self.dynamics_value_bins(agent_bundle.flatten(1).detach())


def clip_agent_grad_norms(agent, max_grad_norm):
    """Clip shared trunk and each head independently to avoid cross-head starvation."""
    groups = [
        agent.wm_backbone.parameters(),
        agent.actor_head.parameters(),
        agent.critic_bins.parameters(),
        agent.dynamics_value_bins.parameters(),
        agent.to_state_pred.parameters(),
        agent.dyn_to_obs.parameters(),
        agent.reward_head_bins.parameters(),
        agent.continue_head.parameters(),
    ]
    for params in groups:
        nn.utils.clip_grad_norm_(params, max_grad_norm)


@contextmanager
def bf16_autocast(device):
    """bf16 autocast on CUDA, no-op elsewhere; SDPA dispatches to FlashAttention."""
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


@contextmanager
def frozen_module_params(*modules):
    params = [param for module in modules for param in module.parameters()]
    old_requires_grad = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in zip(params, old_requires_grad):
            param.requires_grad_(requires_grad)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.rollout_batch_size % args.num_envs != 0:
        raise ValueError("--rollout-batch-size must be divisible by --num-envs")
    if args.dream_batches < 0:
        raise ValueError("--dream-batches must be non-negative")
    if args.dream_batch_size <= 0:
        raise ValueError("--dream-batch-size must be positive")
    if args.imagination_horizon <= 0:
        raise ValueError("--imagination-horizon must be positive")
    if args.wm_batch_size <= 0:
        raise ValueError("--wm-batch-size must be positive")
    args.num_steps = args.rollout_batch_size // args.num_envs
    args.batch_size = int(args.rollout_batch_size)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        from torch.backends.cuda import (
            enable_cudnn_sdp,
            enable_flash_sdp,
            enable_math_sdp,
            enable_mem_efficient_sdp,
        )

        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(True)
        enable_cudnn_sdp(False)

    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        args.num_envs,
        args.num_bins,
        args.reward_num_bins,
        args.dyn_horizon,
        args.tokenizer_path,
        args.tokenizer_checkpoint_prefix,
    ).to(device)
    if args.compile:
        print("[warn] --compile is ignored; this variant uses native Flash SDPA instead of torch.compile.")
    optimizer = optim.Adam((p for p in agent.parameters() if p.requires_grad), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.sigma_ratio,
        device,
        use_symlog=args.use_symlog,
    )
    agent.hl_support = hl_support
    reward_support = HLGaussSupport(
        args.reward_num_bins,
        args.reward_v_min,
        args.reward_v_max,
        args.reward_sigma_ratio,
        device,
        use_symlog=args.reward_use_symlog,
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    next_obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    action_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, action_dim), device=device)
    next_action_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, action_dim), device=device)
    reward_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len), device=device)
    next_reward_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    # Pre-squash sample stored alongside the action so PPO recompute uses the
    # exact `z` from rollout — `z = atanh(action_to_u(action))` is wrong for
    # saturated samples and was the cause of iter-0 KL spikes.
    pre_actions = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    raw_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.ones((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)
    agent.reset_history()
    agent.update_history(next_obs)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_seqs[step] = agent.obs_history.clone()
            action_seqs[step] = agent.action_history.clone()
            reward_seqs[step] = agent.reward_history.clone()
            dones[step] = next_done
            with torch.no_grad(), bf16_autocast(device):
                action, logprob, _, value, _, _, _, _, z = agent.get_action_and_value(
                    agent.obs_history,
                    agent.action_history,
                    agent.reward_history,
                )
                values[step] = value.flatten()
            actions[step] = action
            pre_actions[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            raw_rewards[step] = reward_tensor
            rewards[step] = reward_tensor
            terminations_buf[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
            boundaries[step] = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
            transition_next_obs = np.array(next_obs, copy=True)
            transition_valid_np = np.ones(args.num_envs, dtype=np.float32)
            if "final_observation" in infos:
                final_obs = infos["final_observation"]
                for env_idx, done in enumerate(next_done_np):
                    if done and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                    elif done:
                        transition_valid_np[env_idx] = 0.0
            else:
                transition_valid_np[next_done_np] = 0.0
            transition_valids[step] = torch.as_tensor(transition_valid_np, device=device)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            transition_next_obs = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
            transition_next_history = torch.cat([agent.obs_history[:, 1:], transition_next_obs.unsqueeze(1)], dim=1)
            transition_next_action_history = torch.cat([agent.action_history[:, 1:], action.unsqueeze(1)], dim=1)
            transition_next_reward_history = torch.cat([agent.reward_history[:, 1:], reward_tensor.unsqueeze(1)], dim=1)
            next_obs_seqs[step] = transition_next_history
            next_action_seqs[step] = transition_next_action_history
            next_reward_seqs[step] = transition_next_reward_history
            history_action = action.clone()
            history_action[next_done.bool()] = 0.0
            history_reward = reward_tensor.clone()
            history_reward[next_done.bool()] = 0.0
            if next_done.any():
                agent.reset_history(next_done.bool())
            agent.update_history(next_obs, history_action, history_reward)

            if "final_info" in infos:
                episode_returns = []
                episode_lengths = []
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        episode_returns.append(float(np.asarray(info["episode"]["r"]).reshape(-1)[0]))
                        episode_lengths.append(float(np.asarray(info["episode"]["l"]).reshape(-1)[0]))
                if episode_returns:
                    episode_returns_np = np.asarray(episode_returns, dtype=np.float32)
                    episode_lengths_np = np.asarray(episode_lengths, dtype=np.float32)
                    print(
                        f"global_step={global_step}, episodes={len(episode_returns)}, "
                        f"episodic_return_mean={episode_returns_np.mean():.3f}, "
                        f"episodic_return_min={episode_returns_np.min():.3f}, "
                        f"episodic_return_max={episode_returns_np.max():.3f}, "
                        f"episodic_length_mean={episode_lengths_np.mean():.1f}"
                    )
                    writer.add_scalar("charts/episodic_return", float(episode_returns_np.mean()), global_step)
                    writer.add_scalar("charts/episodic_return_min", float(episode_returns_np.min()), global_step)
                    writer.add_scalar("charts/episodic_return_max", float(episode_returns_np.max()), global_step)
                    writer.add_scalar("charts/episodic_length", float(episode_lengths_np.mean()), global_step)
                    writer.add_scalar("charts/episodes_completed", len(episode_returns), global_step)
                    writer.add_histogram("charts/episodic_return_distribution", episode_returns_np, global_step)

        # GAE: forward under autocast, accumulation in fp32 (the loop body is
        # pure scalar arithmetic — autocast wouldn't affect it, but doing it
        # in fp32 avoids accidental bf16 silent promotions on `next_value`).
        with torch.no_grad(), bf16_autocast(device):
            next_value = agent.get_value(agent.obs_history, agent.action_history, agent.reward_history).reshape(1, -1)
        next_value = next_value.float()
        with torch.no_grad():
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs_seqs = obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_next_obs_seqs = next_obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_action_seqs = action_seqs.reshape(-1, agent.context_len, action_dim)
        b_next_action_seqs = next_action_seqs.reshape(-1, agent.context_len, action_dim)
        b_reward_seqs = reward_seqs.reshape(-1, agent.context_len)
        b_next_reward_seqs = next_reward_seqs.reshape(-1, agent.context_len)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_pre_actions = pre_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_raw_rewards = raw_rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        b_terminations = terminations_buf.reshape(-1)

        b_wm_valid = transition_valids.reshape(-1)

        # Pre-project scalar returns into HL-Gauss categorical targets.
        with torch.no_grad():
            b_return_bins = hl_support.project(b_returns)

        # PPO update
        b_inds = np.arange(args.batch_size)
        spo_penalties = []
        stop_policy_updates = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            mb_inds = b_inds

            if not stop_policy_updates:
                with bf16_autocast(device):
                    (
                        newlogprob,
                        entropy,
                        newvalue,
                        value_logits,
                        agent_flat,
                    ) = agent.get_all_for_update(
                        b_obs_seqs[mb_inds],
                        b_action_seqs[mb_inds],
                        b_reward_seqs[mb_inds],
                        b_pre_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        if iteration <= 2 and epoch == 0:
                            print(
                                f"[diag] iter={iteration} approx_kl={approx_kl.item():.6f} "
                                f"old_approx_kl={old_approx_kl.item():.6f} ratio_mean={ratio.mean().item():.6f} "
                                f"logratio_abs_max={logratio.abs().max().item():.6f}"
                            )

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # SPO asymmetric quadratic penalty (parity with dream).
                    # spo_eps_high relaxes positive-advantage updates, low
                    # tightens negative-advantage updates — same form as the
                    # imagination loop so real and dream pull on the policy
                    # with the same shape, just on different samples.
                    ratio_diff = ratio - 1.0
                    spo_eps = torch.where(
                        (mb_advantages * ratio_diff) > 0,
                        torch.full_like(mb_advantages, args.spo_eps_high),
                        torch.full_like(mb_advantages, args.spo_eps_low),
                    )
                    spo_penalty = mb_advantages.abs() * ratio_diff.square() / (2.0 * spo_eps)
                    pg_loss = -(mb_advantages * ratio - spo_penalty).mean()
                    spo_penalties.append(spo_penalty.mean().item())
                    entropy_loss = entropy.mean()
                    newvalue = newvalue.view(-1)
                    v_loss_bins = F.cross_entropy(value_logits, b_return_bins[mb_inds], reduction='mean')

                    # Dynamics-value bin head also reads a detached view
                    # (Dreamer4 default — only learn heads during PPO; the
                    # WM trunk is shaped by the WM aux losses below).
                    dynamics_value_logits = agent.dynamics_value_bins(agent_flat)
                    dynamics_value_loss_bins = F.cross_entropy(
                        dynamics_value_logits, b_return_bins[mb_inds], reduction='mean'
                    )

                    # Actor, critic_bins, and dynamics_value_bins all read
                    # frozen mixed agent tokens (see get_all_for_update), so PPO only
                    # updates heads. The WM trunk is shaped exclusively by
                    # the WM aux losses below, preventing PPO ratio staleness
                    # from representation drift. HL-Gauss CE on critic_bins
                    # is naturally bounded so no value clipping is needed.
                    policy_loss = (
                        pg_loss - args.ent_coef * entropy_loss
                        + v_loss_bins
                        + DYNAMICS_VALUE_BINS_COEF * dynamics_value_loss_bins
                    )
                optimizer.zero_grad()
                policy_loss.backward()
                clip_agent_grad_norms(agent, args.max_grad_norm)
                optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_policy_updates = True

        # World-model phase: run after PPO so aux updates do not invalidate
        # rollout log-prob ratios inside the PPO epoch.
        wm_horizon_weight = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_latent_weight = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_reward_mse = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_reward_bins = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_continue = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_state_pred = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_trunk_anchor = torch.zeros(args.dyn_horizon, device=device)
        for wm_epoch in range(args.wm_update_epochs):
            wm_seed_inds = np.arange(args.batch_size)
            np.random.shuffle(wm_seed_inds)
            wm_batch_size = min(args.wm_batch_size, args.batch_size)

            for start in range(0, args.batch_size, wm_batch_size):
                mb_inds_np = wm_seed_inds[start:start + wm_batch_size]
                mb_inds = torch.as_tensor(mb_inds_np, device=device, dtype=torch.long)
                mb_size = mb_inds.numel()
                mb_step_inds = mb_inds // args.num_envs
                mb_env_inds = mb_inds % args.num_envs

                with torch.no_grad():
                    horizon_offsets = torch.arange(args.dyn_horizon, device=device)
                    future_step_inds = mb_step_inds[:, None] + horizon_offsets[None, :]
                    in_rollout = future_step_inds < args.num_steps
                    safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                    env_inds = mb_env_inds[:, None].expand_as(safe_step_inds)
                    horizon_flat_inds = safe_step_inds * args.num_envs + env_inds

                    future_boundaries = boundaries[safe_step_inds, env_inds]
                    prev_continues = torch.cat(
                        [torch.ones(mb_size, 1, device=device), 1.0 - future_boundaries[:, :-1]],
                        dim=1,
                    )
                    horizon_step_weight = torch.cumprod(prev_continues, dim=1) * in_rollout.float()
                    needed_target_inds, inverse_target_inds = torch.unique(
                        horizon_flat_inds.reshape(-1),
                        sorted=False,
                        return_inverse=True,
                    )
                    inverse_target_inds = inverse_target_inds.view(mb_size, args.dyn_horizon)

                with bf16_autocast(device):
                    with torch.no_grad():
                        # Tokenize the seed window (5 frames ending at o_t).
                        seed_obs_tokens = agent._encode_tokenizer(
                            b_obs_seqs[mb_inds],
                            return_all_obs_tokens=True,
                        )['all_obs_tokens']
                        # Tokenize each unique imagine window once (deduped via
                        # `needed_target_inds`), then take the last frame to
                        # get the obs token at o_{t+1+h}.
                        imagine_full_tok = agent._encode_tokenizer(
                            b_next_obs_seqs[needed_target_inds],
                            return_all_obs_tokens=True,
                        )['all_obs_tokens']
                        imagine_obs_unique = imagine_full_tok[:, -1]
                        imagine_obs_tokens = imagine_obs_unique[inverse_target_inds]

                    # Per-frame action / reward histories for the imagine
                    # extension. action_history at imagine frame h is the
                    # action that LED to that imagine obs (= a_{t+h}); reward
                    # similarly.
                    imagine_action_history = b_next_action_seqs[
                        horizon_flat_inds.reshape(-1)
                    ][:, -1].view(mb_size, args.dyn_horizon, action_dim)
                    imagine_reward_history = b_next_reward_seqs[
                        horizon_flat_inds.reshape(-1)
                    ][:, -1].view(mb_size, args.dyn_horizon)

                    extended_obs_tokens = torch.cat(
                        [seed_obs_tokens, imagine_obs_tokens], dim=1
                    )
                    extended_action_history = torch.cat(
                        [b_action_seqs[mb_inds], imagine_action_history], dim=1
                    )
                    extended_reward_history = torch.cat(
                        [b_reward_seqs[mb_inds], imagine_reward_history], dim=1
                    )

                    # Per-frame eval-action sequence: at frame predict_start+h
                    # the eval action is a_{t+h} (the action taken FROM that
                    # frame's obs to produce the next obs). Frames outside the
                    # predict window carry zero eval action.
                    predict_start = agent.context_len - 1
                    extended_eval_action_seq = torch.zeros(
                        mb_size,
                        agent.wm_max_temporal_steps,
                        action_dim,
                        device=device,
                        dtype=extended_action_history.dtype,
                    )
                    extended_eval_action_seq[:, predict_start:predict_start + args.dyn_horizon] = (
                        b_actions[horizon_flat_inds]
                    )

                    # Single-pass extended encode: returns the per-frame
                    # transition token sequence for predictions and (via
                    # cls_index=context_len-1) the dyn / agent readouts at the
                    # last seed frame for the trunk anchor. Causal temporal
                    # attention means under teacher forcing the frame-4 dyn is
                    # identical to a dedicated single-window encode on the seed
                    # alone, so this is one autoregressive forward instead of
                    # two stacked forwards.
                    current_obs_embed_flat = seed_obs_tokens[:, -1].flatten(1)
                    ext_cls = agent.wm_backbone.encode_obs_tokens(
                        extended_obs_tokens,
                        action_history=extended_action_history,
                        reward_history=extended_reward_history,
                        transition_action=extended_eval_action_seq,
                        return_transition_seq=True,
                        cls_index=predict_start,
                    )
                    current_flat = agent._dyn_bundle_from_cls(ext_cls).flatten(1)
                    transition_seq = ext_cls['transition_seq']
                    predict_transitions = transition_seq[
                        :, predict_start:predict_start + args.dyn_horizon
                    ]
                    pred, pred_reward_logits, pred_continue_logits = (
                        agent._transition_predictions_from_token(predict_transitions)
                    )

                    target_next_obs_embed = imagine_obs_tokens.flatten(2)

                    pred_mean = pred[:, :, :agent.tok_obs_embed_dim]
                    pred_log_var = pred[:, :, agent.tok_obs_embed_dim:]
                    with torch.amp.autocast(device.type, enabled=False):
                        pred_var = pred_log_var.float().clamp(-10.0, 10.0).exp()
                        state_pred_losses = F.gaussian_nll_loss(
                            pred_mean.float(),
                            target_next_obs_embed.float(),
                            var=pred_var,
                            reduction='none',
                        ).mean(-1)

                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        reward_bins = reward_support.project(b_raw_rewards[needed_target_inds].float())
                    target_reward_bins = reward_bins[inverse_target_inds]
                    reward_losses = F.cross_entropy(
                        pred_reward_logits.reshape(-1, args.reward_num_bins),
                        target_reward_bins.reshape(-1, args.reward_num_bins),
                        reduction='none',
                    ).view(mb_size, args.dyn_horizon)
                    with torch.no_grad():
                        pred_reward_values = reward_support.to_scalar(
                            pred_reward_logits.reshape(-1, args.reward_num_bins).float()
                        ).view(mb_size, args.dyn_horizon)
                        target_rewards = b_raw_rewards[horizon_flat_inds]
                        reward_mse = (pred_reward_values - target_rewards).square()

                    continue_target = 1.0 - b_terminations[horizon_flat_inds]
                    continue_losses = F.binary_cross_entropy_with_logits(
                        pred_continue_logits,
                        continue_target,
                        reduction="none",
                    )

                    trunk_anchor_pred = agent.dyn_to_obs(current_flat)
                    trunk_anchor_flat = F.mse_loss(
                        trunk_anchor_pred,
                        current_obs_embed_flat,
                        reduction='none',
                    ).mean(-1)

                    # All H horizons are trained. horizon_step_weight gates by
                    # in-rollout AND no-prior-boundary; b_wm_valid gates by
                    # per-step transition validity.
                    horizon_latent_weight = horizon_step_weight * b_wm_valid[horizon_flat_inds]
                    step_mask = horizon_step_weight > 0
                    latent_mask = horizon_latent_weight > 0
                    reward_losses = torch.where(step_mask, reward_losses, torch.zeros_like(reward_losses))
                    continue_losses = torch.where(step_mask, continue_losses, torch.zeros_like(continue_losses))
                    reward_mse = torch.where(step_mask, reward_mse, torch.zeros_like(reward_mse))
                    state_pred_losses = torch.where(
                        latent_mask,
                        state_pred_losses,
                        torch.zeros_like(state_pred_losses),
                    )
                    trunk_anchor_weight = horizon_step_weight[:, 0]
                    trunk_anchor_loss = (
                        trunk_anchor_flat * trunk_anchor_weight
                    ).sum() / (trunk_anchor_weight.sum() + 1e-8)

                    reward_loss_bins = (
                        reward_losses * horizon_step_weight
                    ).sum() / (horizon_step_weight.sum() + 1e-8)
                    continue_loss = (
                        continue_losses * horizon_step_weight
                    ).sum() / (horizon_step_weight.sum() + 1e-8)
                    state_pred_loss = (
                        state_pred_losses * horizon_latent_weight
                    ).sum() / (horizon_latent_weight.sum() + 1e-8)

                    with torch.no_grad():
                        wm_horizon_weight += horizon_step_weight.sum(dim=0)
                        wm_horizon_latent_weight += horizon_latent_weight.sum(dim=0)
                        wm_horizon_reward_mse += (
                            reward_mse.detach() * horizon_step_weight
                        ).sum(dim=0)
                        wm_horizon_reward_bins += (
                            reward_losses.detach() * horizon_step_weight
                        ).sum(dim=0)
                        wm_horizon_continue += (
                            continue_losses.detach() * horizon_step_weight
                        ).sum(dim=0)
                        wm_horizon_state_pred += (
                            state_pred_losses.detach() * horizon_latent_weight
                        ).sum(dim=0)
                        wm_horizon_trunk_anchor[0] += (
                            trunk_anchor_flat.detach() * trunk_anchor_weight
                        ).sum()

                    aux_loss = WM_COEF * (
                        equal_weight(reward_loss_bins)
                        + equal_weight(continue_loss)
                        + STATE_PRED_LOSS_COEF * equal_weight(state_pred_loss)
                        + DYNAMICS_SHAPE_COEF * equal_weight(trunk_anchor_loss)
                    )

                optimizer.zero_grad()
                aux_loss.backward()
                clip_agent_grad_norms(agent, args.max_grad_norm)
                optimizer.step()

        # Imagination phase: v54-style fixed dream buffer, then PPO/SPO actor
        # and value updates on imagined latent rollouts.
        imagine_actor_loss = torch.tensor(0.0, device=device)
        imagine_critic_loss = torch.tensor(0.0, device=device)
        imagine_approx_kl = torch.tensor(0.0, device=device)
        imagine_logprob_delta_abs = torch.tensor(0.0, device=device)
        imagined_return_mean = torch.tensor(0.0, device=device)
        dream_horizon_return_mean = None
        dream_horizon_reward_mean = None
        dream_horizon_continue_mean = None
        use_imagination = (
            global_step >= args.imagine_start_step
            and (args.imagine_actor_coef != 0.0 or args.imagine_critic_coef != 0.0)
            and args.dream_batches > 0
        )
        if use_imagination:
            imagine_actor_losses = []
            imagine_critic_losses = []
            imagine_approx_kls = []
            imagine_logprob_delta_abses = []
            imagined_return_means = []
            dream_horizon_return_sum = torch.zeros(args.imagination_horizon, device=device)
            dream_horizon_reward_sum = torch.zeros(args.imagination_horizon, device=device)
            dream_horizon_continue_sum = torch.zeros(args.imagination_horizon, device=device)
            dream_horizon_count = torch.zeros(args.imagination_horizon, device=device)

            for _dream_batch in range(args.dream_batches):
                with torch.no_grad(), bf16_autocast(device):
                    seed_pool = b_obs_seqs.shape[0]
                    dream_seed_count = math.ceil(args.dream_batch_size / args.imagination_horizon)
                    if dream_seed_count < seed_pool:
                        seed_inds = torch.randperm(seed_pool, device=device)[:dream_seed_count]
                    elif dream_seed_count == seed_pool:
                        seed_inds = torch.arange(seed_pool, device=device)
                    else:
                        seed_inds = torch.randint(0, seed_pool, (dream_seed_count,), device=device)
                    dream_seed_obs = b_obs_seqs[seed_inds]
                    dream_seed_actions = b_action_seqs[seed_inds]
                    dream_seed_rewards = b_reward_seqs[seed_inds]
                    dream_seed_alive = b_wm_valid[seed_inds]

                    seed_cls = agent._encode_wm(
                        dream_seed_obs,
                        dream_seed_actions,
                        dream_seed_rewards,
                    )
                    seed_obs_tokens = agent._encode_tokenizer(
                        dream_seed_obs,
                        return_all_obs_tokens=True,
                    )['all_obs_tokens']
                    z = agent._dyn_bundle_from_cls(seed_cls).detach()
                    agent_z = agent._agent_bundle_from_cls(seed_cls).detach()
                    alive = dream_seed_alive.float()
                    dream_states = []
                    dream_pre_actions = []
                    dream_old_logprobs = []
                    dream_values = []
                    dream_rewards = []
                    dream_continues = []
                    dream_learn_masks = []

                    # Sliding context_len-long windows ending at the current
                    # imagined state. Each loop iteration slides them by one
                    # step and re-encodes through the unified trunk, treating
                    # the predicted-obs slab as if it came from the tokenizer.
                    obs_token_window = seed_obs_tokens.detach()
                    action_window = dream_seed_actions.detach()
                    reward_window = dream_seed_rewards.detach()

                    for _ in range(args.imagination_horizon):
                        dream_states.append(agent_z.detach())
                        action, old_logprob, _, _, _, action_z = agent.get_imagined_action_and_value(agent_z)
                        value = agent.get_imagined_value(agent_z, hl_support)

                        pred_state, reward_logits, continue_logits = agent._predict_transition_from_obs_tokens(
                            obs_token_window,
                            action_window,
                            reward_window,
                            action,
                        )
                        continue_prob = continue_logits.float().sigmoid()
                        reward_hat = reward_support.to_scalar(reward_logits) * alive
                        continue_hat = continue_prob * alive

                        # Prepare next-step (s_{k+1}) windows: shift everything
                        # left, append the sampled action and predicted reward,
                        # then encode the next imagined state from the predicted
                        # obs token. Dream rollouts condition future decisions
                        # on predicted rewards just as real rollouts condition
                        # on observed rewards.
                        next_obs_mean = pred_state[:, :agent.tok_obs_embed_dim]
                        next_obs_token = next_obs_mean.view(
                            next_obs_mean.shape[0],
                            1,
                            agent.obs_dim,
                            TOK_EMBED_DIM,
                        ).to(
                            dtype=obs_token_window.dtype
                        )
                        next_obs_window = torch.cat(
                            [obs_token_window[:, 1:], next_obs_token], dim=1
                        )
                        next_action_window = torch.cat(
                            [action_window[:, 1:], action.detach().unsqueeze(1)], dim=1
                        )
                        next_reward_window = torch.cat(
                            [reward_window[:, 1:], reward_hat.detach().unsqueeze(1)],
                            dim=1,
                        )
                        cls_next = agent._encode_next_obs_window(
                            next_obs_window,
                            action_history=next_action_window,
                            reward_history=next_reward_window,
                        )
                        next_z = agent._dyn_bundle_from_cls(cls_next)
                        next_agent_z = agent._agent_bundle_from_cls(cls_next)

                        dream_pre_actions.append(action_z.detach())
                        dream_old_logprobs.append(old_logprob.detach())
                        dream_values.append(value.detach())
                        dream_rewards.append(reward_hat.detach())
                        dream_continues.append(continue_hat.detach())
                        dream_learn_masks.append(alive.bool())
                        sampled_terminal = (torch.bernoulli((1.0 - continue_prob).clamp(0.0, 1.0)) > 0).float()
                        alive = alive * (1.0 - sampled_terminal)

                        # Slide windows forward into s_{k+1} for the next iter.
                        obs_token_window = next_obs_window.detach()
                        action_window = next_action_window
                        reward_window = next_reward_window
                        z = next_z.detach()
                        agent_z = next_agent_z.detach()

                    bootstrap_value = agent.get_imagined_value(agent_z, hl_support).detach()
                    dream_returns = []
                    dream_gae = torch.zeros_like(bootstrap_value)
                    dream_values_with_bootstrap = dream_values + [bootstrap_value]
                    for h in reversed(range(args.imagination_horizon)):
                        delta = (
                            dream_rewards[h]
                            + args.gamma * dream_continues[h] * dream_values_with_bootstrap[h + 1]
                            - dream_values_with_bootstrap[h]
                        )
                        dream_gae = delta + args.gamma * args.gae_lambda * dream_continues[h] * dream_gae
                        dream_gae = torch.where(dream_learn_masks[h], dream_gae, torch.zeros_like(dream_gae))
                        dream_returns.append(dream_gae + dream_values_with_bootstrap[h])
                    dream_returns.reverse()

                    with torch.no_grad():
                        horizon_items = zip(
                            dream_returns,
                            dream_rewards,
                            dream_continues,
                            dream_learn_masks,
                        )
                        for horizon, (horizon_return, horizon_reward, horizon_continue, horizon_mask) in enumerate(horizon_items):
                            mask = horizon_mask.float()
                            count = mask.sum()
                            dream_horizon_return_sum[horizon] += (horizon_return * mask).sum()
                            dream_horizon_reward_sum[horizon] += (horizon_reward * mask).sum()
                            dream_horizon_continue_sum[horizon] += (horizon_continue * mask).sum()
                            dream_horizon_count[horizon] += count

                    dream_states = torch.cat(dream_states, dim=0)
                    dream_pre_actions = torch.cat(dream_pre_actions, dim=0)
                    dream_old_logprobs = torch.cat(dream_old_logprobs, dim=0)
                    dream_values = torch.cat(dream_values, dim=0)
                    dream_returns = torch.cat(dream_returns, dim=0)
                    dream_advantages = dream_returns - dream_values
                    dream_learn_masks = torch.cat(dream_learn_masks, dim=0)
                    if dream_states.shape[0] > args.dream_batch_size:
                        train_inds = torch.randperm(dream_states.shape[0], device=device)[: args.dream_batch_size]
                        dream_states = dream_states[train_inds]
                        dream_pre_actions = dream_pre_actions[train_inds]
                        dream_old_logprobs = dream_old_logprobs[train_inds]
                        dream_values = dream_values[train_inds]
                        dream_returns = dream_returns[train_inds]
                        dream_advantages = dream_advantages[train_inds]
                        dream_learn_masks = dream_learn_masks[train_inds]
                    if bool(dream_learn_masks.any()):
                        imagined_return_means.append(dream_returns[dream_learn_masks].mean())
                    else:
                        imagined_return_means.append(dream_returns.mean())

                dream_inds = np.arange(dream_states.shape[0])
                for _ in range(args.imagine_update_epochs):
                    np.random.shuffle(dream_inds)
                    mb_inds = dream_inds
                    mb_mask = dream_learn_masks[mb_inds]
                    has_targets = bool(mb_mask.any())

                    with bf16_autocast(device):
                        if args.imagine_actor_coef != 0.0:
                            _, new_logprob, dream_entropy, _, _, _ = agent.get_imagined_action_and_value(
                                dream_states[mb_inds],
                                z=dream_pre_actions[mb_inds],
                            )
                            dream_logratio = new_logprob - dream_old_logprobs[mb_inds]
                            dream_ratio = dream_logratio.exp()
                            with torch.no_grad():
                                if has_targets:
                                    valid_logratio = dream_logratio[mb_mask]
                                    valid_ratio = dream_ratio[mb_mask]
                                    dream_approx_kl = ((valid_ratio - 1.0) - valid_logratio).mean()
                                    imagine_approx_kls.append(dream_approx_kl.item())
                                    imagine_logprob_delta_abses.append(valid_logratio.abs().mean().item())

                            mb_advantages = dream_advantages[mb_inds]
                            if args.norm_adv and has_targets:
                                valid_advantages = mb_advantages[mb_mask]
                                mb_advantages = (mb_advantages - valid_advantages.mean()) / (
                                    valid_advantages.std(unbiased=False) + 1e-8
                                )

                            ratio_diff = dream_ratio - 1.0
                            dream_spo_eps = torch.where(
                                (mb_advantages * ratio_diff) > 0,
                                torch.full_like(mb_advantages, args.spo_eps_high),
                                torch.full_like(mb_advantages, args.spo_eps_low),
                            )
                            dream_spo_penalty = mb_advantages.abs() * ratio_diff.square() / (2.0 * dream_spo_eps)
                            dream_policy_loss = -(mb_advantages * dream_ratio - dream_spo_penalty)
                            if has_targets:
                                imagine_actor_loss = dream_policy_loss[mb_mask].mean()
                                imagine_actor_loss = imagine_actor_loss - args.imagine_actor_ent_coef * dream_entropy[mb_mask].mean()
                            else:
                                imagine_actor_loss = dream_policy_loss.sum() * 0.0
                        else:
                            imagine_actor_loss = torch.tensor(0.0, device=device)

                        dream_return_targets = dream_returns[mb_inds]
                        with torch.amp.autocast("cuda", enabled=False):
                            dream_return_probs = hl_support.project(dream_return_targets.float())
                        dream_value_logits = agent.get_imagined_value_logits(dream_states[mb_inds])
                        dream_value_loss_bins = F.cross_entropy(
                            dream_value_logits,
                            dream_return_probs.detach(),
                            reduction='none',
                        )
                        if has_targets:
                            imagine_critic_loss = dream_value_loss_bins[mb_mask].mean()
                        else:
                            imagine_critic_loss = dream_value_loss_bins.sum() * 0.0

                        imagine_loss = (
                            args.imagine_actor_coef * imagine_actor_loss
                            + args.imagine_critic_coef * imagine_critic_loss
                        )
                    optimizer.zero_grad()
                    imagine_loss.backward()
                    clip_agent_grad_norms(agent, args.max_grad_norm)
                    optimizer.step()

                    imagine_actor_losses.append(imagine_actor_loss.item())
                    imagine_critic_losses.append(imagine_critic_loss.item())

            if imagine_actor_losses:
                imagine_actor_loss = torch.tensor(float(np.mean(imagine_actor_losses)), device=device)
            if imagine_critic_losses:
                imagine_critic_loss = torch.tensor(float(np.mean(imagine_critic_losses)), device=device)
            if imagine_approx_kls:
                imagine_approx_kl = torch.tensor(float(np.mean(imagine_approx_kls)), device=device)
            if imagine_logprob_delta_abses:
                imagine_logprob_delta_abs = torch.tensor(float(np.mean(imagine_logprob_delta_abses)), device=device)
            if imagined_return_means:
                imagined_return_mean = torch.stack(imagined_return_means).mean()
            if bool((dream_horizon_count > 0).any()):
                dream_horizon_denom = dream_horizon_count.clamp_min(1.0)
                dream_horizon_return_mean = dream_horizon_return_sum / dream_horizon_denom
                dream_horizon_reward_mean = dream_horizon_reward_sum / dream_horizon_denom
                dream_horizon_continue_mean = dream_horizon_continue_sum / dream_horizon_denom

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_bins", v_loss_bins.item(), global_step)
        writer.add_scalar("losses/dynamics_value_loss_bins", dynamics_value_loss_bins.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/spo_penalty", np.mean(spo_penalties) if spo_penalties else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        wm_horizon_denom = wm_horizon_weight.clamp_min(1e-8)
        wm_horizon_latent_denom = wm_horizon_latent_weight.clamp_min(1e-8)
        wm_reward_mse = wm_horizon_reward_mse.sum() / wm_horizon_weight.sum().clamp_min(1e-8)
        wm_reward_loss_bins = wm_horizon_reward_bins.sum() / wm_horizon_weight.sum().clamp_min(1e-8)
        wm_continue_loss = wm_horizon_continue.sum() / wm_horizon_weight.sum().clamp_min(1e-8)
        wm_trunk_anchor_loss = wm_horizon_trunk_anchor[0] / wm_horizon_weight[0].clamp_min(1e-8)
        wm_state_pred_loss = wm_horizon_state_pred.sum() / wm_horizon_latent_weight.sum().clamp_min(1e-8)
        writer.add_scalar("worldmodel/reward_mse", wm_reward_mse.item(), global_step)
        writer.add_scalar("worldmodel/reward_loss_bins", wm_reward_loss_bins.item(), global_step)
        writer.add_scalar("worldmodel/continue_loss", wm_continue_loss.item(), global_step)
        writer.add_scalar("worldmodel/trunk_anchor_loss", wm_trunk_anchor_loss.item(), global_step)
        writer.add_scalar("state_pred/loss", wm_state_pred_loss.item(), global_step)
        wm_horizon_reward_mse_mean = wm_horizon_reward_mse / wm_horizon_denom
        wm_horizon_reward_bins_mean = wm_horizon_reward_bins / wm_horizon_denom
        wm_horizon_continue_mean = wm_horizon_continue / wm_horizon_denom
        wm_horizon_trunk_anchor_mean = wm_horizon_trunk_anchor / wm_horizon_denom
        wm_horizon_state_pred_mean = wm_horizon_state_pred / wm_horizon_latent_denom
        for horizon in range(args.dyn_horizon):
            horizon_tag = f"h{horizon + 1:02d}"
            writer.add_scalar(
                f"worldmodel_horizon/reward_mse_{horizon_tag}",
                wm_horizon_reward_mse_mean[horizon].item(),
                global_step,
            )
            writer.add_scalar(
                f"worldmodel_horizon/reward_bins_{horizon_tag}",
                wm_horizon_reward_bins_mean[horizon].item(),
                global_step,
            )
            writer.add_scalar(
                f"worldmodel_horizon/continue_{horizon_tag}",
                wm_horizon_continue_mean[horizon].item(),
                global_step,
            )
            writer.add_scalar(
                f"worldmodel_horizon/state_pred_{horizon_tag}",
                wm_horizon_state_pred_mean[horizon].item(),
                global_step,
            )
            if horizon == 0:
                writer.add_scalar(
                    f"worldmodel_horizon/trunk_anchor_{horizon_tag}",
                    wm_horizon_trunk_anchor_mean[horizon].item(),
                    global_step,
                )
        writer.add_scalar("imagination/actor_loss", imagine_actor_loss.item(), global_step)
        writer.add_scalar("imagination/critic_loss", imagine_critic_loss.item(), global_step)
        writer.add_scalar("imagination/post_actor_approx_kl", imagine_approx_kl.item(), global_step)
        writer.add_scalar("imagination/post_actor_logprob_delta_abs", imagine_logprob_delta_abs.item(), global_step)
        with torch.no_grad():
            writer.add_scalar("imagination/mean_return", imagined_return_mean.item(), global_step)
            if dream_horizon_return_mean is not None:
                for horizon in range(args.imagination_horizon):
                    horizon_tag = f"h{horizon + 1:02d}"
                    writer.add_scalar(
                        f"imagination_horizon/return_{horizon_tag}",
                        dream_horizon_return_mean[horizon].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f"imagination_horizon/reward_{horizon_tag}",
                        dream_horizon_reward_mean[horizon].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f"imagination_horizon/continue_{horizon_tag}",
                        dream_horizon_continue_mean[horizon].item(),
                        global_step,
                    )
            with bf16_autocast(device):
                cls = agent._encode_wm(agent.obs_history, agent.action_history, agent.reward_history)
                agent_flat = agent._agent_flat_from_cls(cls)
                _, action_std = agent._actor_mean_std(agent_flat)
            writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
