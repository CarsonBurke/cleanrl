# PPO + WM + SDE + state entropy + relu^2 heads + HL-Gauss value/reward encoding.
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
#   8. WM aux objective now mirrors dreamer4: predictor outputs are decoded
#      against the FROZEN tokenizer's next-obs embeddings (state_pred_loss).
#      The collapse-prone transition_loss against the trainable trunk's own
#      next-bundle is dropped; sigreg goes with it (no longer needed).
#   9. DreamerV3-style asymmetric continue label smoothing
#      (target = (1 - terminal).clamp(max=gamma)) so the head's asymptote is
#      bounded on envs that never terminate.
#  10. Per-frame trunk anchor: dyn_to_obs head decodes WM dyn_flat against
#      the frozen tokenizer's CURRENT-step obs tokens. Without it the
#      action-conditioned predictor can carry all the burden via state_pred,
#      letting the dyn CLS go lazy.
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
WM_DYN_FLAT_DIM = WM_EMBED_DIM * DYN_TOKEN_COUNT
TOK_DYN_FLAT_DIM = TOK_EMBED_DIM * DYN_TOKEN_COUNT

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
STATE_ENT_COEF = 0.0
STATE_PRED_LOSS_COEF = 0.1
SQUASH_EPS = 1e-6
PRED_NUM_BLOCKS = 2

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
    compile: bool = True
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
    num_envs: int = 1
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
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
    pred_context: int = 5
    wm_update_epochs: int = 1
    imagination_horizon: int = 16
    imagine_start_step: int = 0
    imagine_update_epochs: int = 1
    imagine_actor_coef: float = 1.0
    imagine_critic_coef: float = 1.0
    imagine_actor_ent_coef: float = 0.0
    # Cap on the number of seed states imagined from each rollout buffer.
    # 0 means "use all" (the v1 behaviour). Subsampling avoids OOM at large
    # num_envs/num_steps and is standard practice for Dreamer imagination.
    dream_batch_size: int = 2048

    batch_size: int = 0
    minibatch_size: int = 0
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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
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
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def build_rope_cache(seq_len, head_dim, device):
    assert head_dim % 2 == 0
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
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

    def forward(self, x, x0, rope_cos=None, rope_sin=None, attn_mask=None):
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

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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

    def forward(self, obs_seq, return_obs_tokens=False):
        batch, time_steps, _ = obs_seq.shape
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1))
        obs_tokens = obs_tokens + self.dim_id_embed(self.dim_indices).view(1, 1, self.obs_dim, self.embed_dim)

        cls_tokens = torch.stack(list(self.cls_params), dim=0)
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
        return out


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        num_envs,
        num_bins,
        reward_num_bins,
        tokenizer_path="",
        tokenizer_checkpoint_prefix="",
        pred_context=CONTEXT_LEN,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.obs_dim = obs_dim
        # Tokenizer is the frozen target; state-pred targets live in TOK space.
        self.tok_obs_embed_dim = obs_dim * TOK_EMBED_DIM
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN
        self.pred_context = pred_context

        self.wm_backbone = STSTSCLSBackbone(
            obs_dim, self.context_len, WM_CLS_NAMES,
            embed_dim=WM_EMBED_DIM,
            num_heads=WM_NUM_HEADS,
            ffn_mult=WM_FFN_MULT,
            num_spatial_blocks=WM_NUM_SPATIAL_BLOCKS,
            num_temporal_blocks=WM_NUM_TEMPORAL_BLOCKS,
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
        self.actor_head = relusq_mlp_orthogonal(WM_DYN_FLAT_DIM, 2 * action_dim, ACTOR_HIDDEN, out_std=0.01)

        # HL-Gauss bin critic — trained by softmax CE; expected-value
        # readout via `to_value(critic_bins(...))` is naturally bounded
        # by the support range, so no MSE head and no value clipping
        # are needed (a separate MSE head with unbounded loss would
        # dominate the shared trunk's gradient).
        self.critic_bins = relusq_mlp_orthogonal(WM_DYN_FLAT_DIM, num_bins, CRITIC_HIDDEN, out_std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

        # State prediction grounds dynamics in next-observation token embeddings
        # produced by the (frozen, smaller) tokenizer backbone.
        self.to_state_pred = nn.Sequential(
            RMSNorm(WM_DYN_FLAT_DIM),
            nn.Linear(WM_DYN_FLAT_DIM, self.tok_obs_embed_dim * 2),
        )

        # Per-frame trunk anchor: WM dyn_flat at the current step is decoded
        # against the FROZEN tokenizer's current-step obs tokens. This forces
        # every dyn CLS to encode the current state directly — without it the
        # predictor (which is the only trunk gradient path through state_pred)
        # could carry the burden and let the trunk go lazy.
        self.dyn_to_obs = nn.Sequential(
            RMSNorm(WM_DYN_FLAT_DIM),
            nn.Linear(WM_DYN_FLAT_DIM, self.tok_obs_embed_dim),
        )

        # Action-token contextual dynamics predictor (lives in WM space).
        self.pred_action_proj = layer_init(nn.Linear(1, WM_EMBED_DIM))
        self.pred_action_dim_embed = nn.Embedding(action_dim, WM_EMBED_DIM)
        nn.init.trunc_normal_(self.pred_action_dim_embed.weight, std=1.0 / WM_EMBED_DIM**0.5)
        self.pred_pos_embed = nn.Parameter(
            torch.empty(self.pred_context * (action_dim + DYN_TOKEN_COUNT), WM_EMBED_DIM)
        )
        nn.init.trunc_normal_(self.pred_pos_embed, std=1.0 / WM_EMBED_DIM**0.5)
        # Learnable "no-history" tokens prepended when the real history is
        # shorter than pred_context. Keeps the predictor's seq dim fixed at
        # pred_context * tokens_per_step so torch.compile sees one shape and
        # SDPA dispatches to FlashAttention with no mask.
        self.pred_pad_latent = nn.Parameter(torch.zeros(1, 1, DYN_TOKEN_COUNT, WM_EMBED_DIM))
        nn.init.trunc_normal_(self.pred_pad_latent, std=1.0 / WM_EMBED_DIM**0.5)
        self.pred_pad_action = nn.Parameter(torch.zeros(1, 1, action_dim))
        pred_init_scale = 1.0 / (2 * PRED_NUM_BLOCKS) ** 0.5
        self.pred_blocks = nn.ModuleList(
            [SelfAttentionBlock(WM_EMBED_DIM, WM_NUM_HEADS, WM_FFN_MULT, pred_init_scale)
             for _ in range(PRED_NUM_BLOCKS)]
        )
        self.pred_final_norm = RMSNorm(WM_EMBED_DIM)
        self.pred_next_proj = layer_init(nn.Linear(WM_EMBED_DIM, WM_EMBED_DIM))

        # Dynamics-value HL-Gauss bin head — trained on real GAE returns,
        # then used as the value bootstrap/readout inside imagined rollouts.
        # The dream loop computes lambda returns once per iteration under
        # no_grad, freezing the bootstrap target for that iteration's SGD —
        # so a separate EMA target copy isn't needed.
        self.dynamics_value_bins = relusq_mlp(WM_DYN_FLAT_DIM, num_bins, DYN_VALUE_HIDDEN, out_std=1.0)
        # Reward reads current latent, action features, and predicted next latent.
        reward_input_dim = WM_DYN_FLAT_DIM * 2 + action_dim * 2 + 1
        self.reward_head = relusq_mlp(reward_input_dim, 1, REWARD_HIDDEN)
        self.reward_head_bins = relusq_mlp(reward_input_dim, reward_num_bins, REWARD_HIDDEN)
        self.continue_head = relusq_mlp(WM_DYN_FLAT_DIM, 1, CONTINUE_HIDDEN)
        nn.init.constant_(self.continue_head[-1].bias, 5.0)

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0

    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def _load_tokenizer_checkpoint(self, tokenizer_path, tokenizer_checkpoint_prefix=""):
        path = Path(tokenizer_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
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

    def _encode_wm(self, obs_seq, return_obs_tokens=False):
        return self.wm_backbone(obs_seq, return_obs_tokens=return_obs_tokens)

    def _encode_tokenizer(self, obs_seq, return_obs_tokens=False):
        self.tokenizer_backbone.eval()
        with torch.no_grad():
            return self.tokenizer_backbone(obs_seq, return_obs_tokens=return_obs_tokens)

    def _dyn_bundle_from_cls(self, cls):
        return torch.stack([cls[name] for name in DYN_TOKEN_NAMES], dim=1)

    def _dyn_flat_from_cls(self, cls):
        return self._dyn_bundle_from_cls(cls).flatten(1)

    def _reward_input(self, current_flat, action, next_flat):
        action_energy = action.square().sum(dim=-1, keepdim=True)
        return torch.cat([current_flat, next_flat, action, action.square(), action_energy], dim=-1)

    def _predict_next_bundle_from_history(self, latent_history, action_history):
        batch, context_len, num_tokens, width = latent_history.shape
        if context_len > self.pred_context:
            latent_history = latent_history[:, -self.pred_context:]
            action_history = action_history[:, -self.pred_context:]
        elif context_len < self.pred_context:
            pad = self.pred_context - context_len
            latent_history = torch.cat(
                [self.pred_pad_latent.expand(batch, pad, num_tokens, width), latent_history],
                dim=1,
            )
            action_history = torch.cat(
                [self.pred_pad_action.expand(batch, pad, self.action_dim), action_history],
                dim=1,
            )
        context_len = self.pred_context
        action_tokens = self.pred_action_proj(action_history.unsqueeze(-1))
        action_tokens = action_tokens + self.pred_action_dim_embed.weight.view(1, 1, self.action_dim, WM_EMBED_DIM)
        tokens = torch.cat([action_tokens, latent_history], dim=2)
        tokens_per_step = self.action_dim + num_tokens
        tokens = tokens.reshape(batch, context_len * tokens_per_step, width)
        tokens = tokens + self.pred_pos_embed.view(1, context_len * tokens_per_step, width)
        x0 = tokens
        # No attn_mask: every token in this call is from a strictly-past step
        # (or learnable padding) relative to the prediction target, so full
        # attention is correct and SDPA dispatches to FlashAttention.
        for block in self.pred_blocks:
            tokens = block(tokens, x0)
        tokens = self.pred_next_proj(self.pred_final_norm(tokens))
        tokens = tokens.reshape(batch, context_len, tokens_per_step, width)
        return tokens[:, -1, self.action_dim:]

    def predict_next_bundles_teacher_forced(self, latent_history, action_history):
        _, horizon, _, _ = latent_history.shape
        preds = []
        for step in range(horizon):
            start = max(0, step + 1 - self.pred_context)
            preds.append(
                self._predict_next_bundle_from_history(
                    latent_history[:, start:step + 1],
                    action_history[:, start:step + 1],
                )
            )
        return torch.stack(preds, dim=1)

    def _actor_mean_std(self, dyn_flat):
        """State-dependent (mean, std). std = softplus(raw) + STD_MIN — smooth, floored, no clamp."""
        mean, raw_std = self.actor_head(dyn_flat).chunk(2, dim=-1)
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

    def get_state_entropy_bonus(self, state_pred_latent):
        pred = self.to_state_pred(state_pred_latent)
        log_var = pred[:, self.tok_obs_embed_dim:]
        return log_var.mean(dim=-1) * STATE_ENT_COEF

    def to_value(self, logits):
        """Expected value from HL-Gauss bin logits — set self.hl_support after construction."""
        return self.hl_support.to_scalar(logits)

    def get_value(self, obs_seq):
        cls = self._encode_wm(obs_seq)
        return self.to_value(self.critic_bins(self._dyn_flat_from_cls(cls).detach()))

    def get_action_and_value(self, obs_seq):
        # Actor, critic_bins, and dynamics_value_bins all read a detached
        # view of the WM trunk (Dreamer4 default — only train heads from
        # PPO/value losses). The trunk is shaped exclusively by the WM aux
        # losses (state-pred / reward / continue / dyn_to_obs / predictor),
        # which keeps PPO from staling its own ratios mid-epoch by moving
        # the latent representation under stored old log-probs.
        wm_cls = self._encode_wm(obs_seq)
        dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
        dyn_flat = dyn_bundle.flatten(1)
        dyn_flat_d = dyn_flat.detach()
        action_mean, action_std = self._actor_mean_std(dyn_flat_d)
        action, log_prob, entropy, z = self._squashed_log_prob_entropy(action_mean, action_std)
        return action, log_prob, entropy, self.to_value(self.critic_bins(dyn_flat_d)), dyn_bundle, dyn_flat, z

    def get_all_for_update(self, obs_seq, z):
        # Dreamer4-style RL boundary: replay the WM encoder without building a
        # trainable trunk graph, then learn actor/value heads from frozen latents.
        with torch.no_grad():
            wm_cls = self._encode_wm(obs_seq)
            dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
            dyn_flat_d = dyn_bundle.flatten(1)
        action_mean, action_std = self._actor_mean_std(dyn_flat_d)
        _, log_prob, entropy, _ = self._squashed_log_prob_entropy(action_mean, action_std, z=z)
        value_logits = self.critic_bins(dyn_flat_d)
        return (
            log_prob,
            entropy,
            self.to_value(value_logits),                # expected value from bins
            value_logits,                               # bin logits for HL-Gauss CE
            dyn_flat_d,                                 # detached — all PPO heads consume detached trunk
        )

    def get_imagined_action_and_value(self, dyn_bundle, z=None):
        # Both actor and dynamics_value_bins read a detached view — Dreamer4
        # default: only train heads from PPO/value losses. dyn_bundle is itself
        # a detached imagined latent; we re-detach defensively.
        dyn_flat = dyn_bundle.flatten(1).detach()
        action_mean, action_std = self._actor_mean_std(dyn_flat)
        action, log_prob, entropy, z_out = self._squashed_log_prob_entropy(action_mean, action_std, z=z)
        value_logits = self.dynamics_value_bins(dyn_flat)
        return action, log_prob, entropy, self.hl_support.to_scalar(value_logits), value_logits, z_out

    def get_imagined_value(self, dyn_bundle, hl_support):
        return hl_support.to_scalar(self.get_imagined_value_logits(dyn_bundle))

    def get_imagined_value_logits(self, dyn_bundle):
        # Defensive detach for symmetry with get_imagined_action_and_value —
        # imagined latents are stored detached, but we re-detach so the head
        # never accidentally backprops through the trunk via this path.
        return self.dynamics_value_bins(dyn_bundle.flatten(1).detach())


def clip_agent_grad_norms(agent, max_grad_norm):
    """Clip shared trunk and each head independently to avoid cross-head starvation."""
    groups = [
        agent.wm_backbone.parameters(),
        agent.actor_head.parameters(),
        agent.critic_bins.parameters(),
        agent.dynamics_value_bins.parameters(),
        agent.to_state_pred.parameters(),
        agent.dyn_to_obs.parameters(),
        agent.pred_action_proj.parameters(),
        agent.pred_action_dim_embed.parameters(),
        agent.pred_blocks.parameters(),
        agent.pred_final_norm.parameters(),
        agent.pred_next_proj.parameters(),
        [agent.pred_pos_embed, agent.pred_pad_latent, agent.pred_pad_action],
        agent.reward_head.parameters(),
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
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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

    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        args.num_envs,
        args.num_bins,
        args.reward_num_bins,
        args.tokenizer_path,
        args.tokenizer_checkpoint_prefix,
        args.pred_context,
    ).to(device)
    if args.compile and device.type == "cuda":
        # All transformers see only batch-dim variation across phases — their
        # internal seq dim is fixed (CONTEXT_LEN × slots for the backbones,
        # pred_context × tokens_per_step for pred_blocks now that history is
        # padded with `pred_pad_*`). `dynamic=False, fullgraph=True` recompiles
        # once per distinct batch size then hits the cache.
        agent.wm_backbone = torch.compile(agent.wm_backbone, dynamic=False, fullgraph=True)
        agent.tokenizer_backbone = torch.compile(agent.tokenizer_backbone, dynamic=False, fullgraph=True)
        agent.pred_blocks = nn.ModuleList(
            [torch.compile(block, dynamic=False, fullgraph=True) for block in agent.pred_blocks]
        )
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
            dones[step] = next_done
            with torch.no_grad(), bf16_autocast(device):
                action, logprob, _, value, _, state_pred_latent, z = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
                ent_bonus = agent.get_state_entropy_bonus(state_pred_latent)
            actions[step] = action
            pre_actions[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            reward_tensor = torch.as_tensor(reward, device=device).view(-1)
            raw_rewards[step] = reward_tensor
            rewards[step] = reward_tensor + ent_bonus
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
            next_obs_seqs[step] = transition_next_history
            if next_done.any():
                agent.reset_history(next_done.bool())
            agent.update_history(next_obs)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE: forward under autocast, accumulation in fp32 (the loop body is
        # pure scalar arithmetic — autocast wouldn't affect it, but doing it
        # in fp32 avoids accidental bf16 silent promotions on `next_value`).
        with torch.no_grad(), bf16_autocast(device):
            next_value = agent.get_value(agent.obs_history).reshape(1, -1)
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
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if not stop_policy_updates:
                    with bf16_autocast(device):
                        (
                            newlogprob,
                            entropy,
                            newvalue,
                            value_logits,
                            dyn_flat,
                        ) = agent.get_all_for_update(b_obs_seqs[mb_inds], b_pre_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1.0) - logratio).mean()
                            if iteration <= 2 and epoch == 0 and start == 0:
                                print(
                                    f"[diag] iter={iteration} mb=0 approx_kl={approx_kl.item():.6f} "
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
                        dynamics_value_logits = agent.dynamics_value_bins(dyn_flat)
                        dynamics_value_loss_bins = F.cross_entropy(
                            dynamics_value_logits, b_return_bins[mb_inds], reduction='mean'
                        )

                        # Actor, critic_bins, and dynamics_value_bins all read
                        # frozen latents (see get_all_for_update), so PPO only
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
        # rollout log-prob ratios inside the PPO epoch. Mirrors the PPO
        # minibatch size — the (mb × dyn_horizon)-batch encode then matches
        # PPO's encode shape modulo the horizon multiplier, which keeps the
        # number of WM-aux minibatches at args.num_minibatches.
        wm_minibatch_size = args.minibatch_size
        wm_horizon_weight = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_latent_weight = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_reward_mse = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_reward_bins = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_continue = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_state_pred = torch.zeros(args.dyn_horizon, device=device)
        wm_horizon_trunk_anchor = torch.zeros(args.dyn_horizon, device=device)
        for wm_epoch in range(args.wm_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, wm_minibatch_size):
                end = start + wm_minibatch_size
                mb_inds = b_inds[start:end]

                # Teacher-forced multi-step dynamics, inspired by v54: encode
                # a real rollout window, predict each next latent from the
                # corresponding real current latent/action, and mask across
                # rollout edges and episode boundaries.
                mb_size = len(mb_inds)
                mb_step_inds = torch.as_tensor(mb_inds // args.num_envs, device=device, dtype=torch.long)
                mb_env_inds = torch.as_tensor(mb_inds % args.num_envs, device=device, dtype=torch.long)
                horizon_offsets = torch.arange(args.dyn_horizon, device=device)
                future_step_inds = mb_step_inds[:, None] + horizon_offsets[None, :]
                in_rollout = (future_step_inds < args.num_steps).float()
                safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                env_inds = mb_env_inds[:, None].expand_as(safe_step_inds)

                future_actions = actions[safe_step_inds, env_inds]
                future_rewards = raw_rewards[safe_step_inds, env_inds]
                future_terminations = terminations_buf[safe_step_inds, env_inds]
                future_boundaries = boundaries[safe_step_inds, env_inds]
                future_valids = transition_valids[safe_step_inds, env_inds]
                future_next_obs_seqs = next_obs_seqs[safe_step_inds, env_inds]

                with bf16_autocast(device):
                    # Encode current state plus the first horizon-1 next states
                    # (no horizon+1 needed since transition_loss is gone — we
                    # only need current_bundles to feed the predictor).
                    window_obs_seqs = torch.cat(
                        [b_obs_seqs[mb_inds].unsqueeze(1), future_next_obs_seqs[:, :-1]], dim=1
                    )
                    window_cls = agent._encode_wm(window_obs_seqs.reshape(-1, agent.context_len, obs_dim))
                    current_bundles = agent._dyn_bundle_from_cls(window_cls).reshape(
                        mb_size, args.dyn_horizon, DYN_TOKEN_COUNT, WM_EMBED_DIM
                    )
                    current_flat = current_bundles.reshape(mb_size * args.dyn_horizon, WM_DYN_FLAT_DIM)
                    flat_future_actions = future_actions.reshape(mb_size * args.dyn_horizon, action_dim)

                    pred_next_bundles = agent.predict_next_bundles_teacher_forced(
                        current_bundles,
                        future_actions,
                    )
                    pred_next_flat = pred_next_bundles.reshape(mb_size * args.dyn_horizon, WM_DYN_FLAT_DIM)
                    reward_input = agent._reward_input(current_flat, flat_future_actions, pred_next_flat)
                    pred_rewards = agent.reward_head(reward_input).view(mb_size, args.dyn_horizon)
                    pred_reward_logits = agent.reward_head_bins(reward_input).view(
                        mb_size, args.dyn_horizon, args.reward_num_bins
                    )
                    pred_continue_logits = agent.continue_head(pred_next_flat).view(mb_size, args.dyn_horizon)

                    prev_continues = torch.cat(
                        [torch.ones(mb_size, 1, device=device), 1.0 - future_boundaries[:, :-1]],
                        dim=1,
                    )
                    step_weight = torch.cumprod(prev_continues, dim=1) * in_rollout
                    latent_weight = step_weight * future_valids

                    with torch.no_grad():
                        tokenizer_next = agent._encode_tokenizer(
                            future_next_obs_seqs.reshape(-1, agent.context_len, obs_dim),
                            return_obs_tokens=True,
                        )
                        next_obs_embed_flat = tokenizer_next['obs_tokens'].flatten(1).reshape(
                            mb_size, args.dyn_horizon, agent.tok_obs_embed_dim
                        )
                        # Current-step obs embeds for the per-frame trunk anchor.
                        # Window step h's "current" obs equals window step h-1's "next"
                        # obs (== `future_next_obs_seqs[:, h-1]`), so only step 0 needs
                        # a fresh tokenizer pass — the rest reuse `next_obs_embed_flat`.
                        first_step_tokens = agent._encode_tokenizer(
                            b_obs_seqs[mb_inds],
                            return_obs_tokens=True,
                        )['obs_tokens'].flatten(1)
                        current_obs_embed_flat = torch.cat(
                            [first_step_tokens.unsqueeze(1), next_obs_embed_flat[:, :-1]],
                            dim=1,
                        )

                    reward_loss = (
                        (pred_rewards - future_rewards).square() * step_weight
                    ).sum() / (step_weight.sum() + 1e-8)

                    # HL-Gauss reward loss. Project in fp32 — bf16 erf
                    # underflows the inner-bin probability mass for targets
                    # near the support boundary.
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        mb_reward_bins = reward_support.project(future_rewards.reshape(-1).float())
                    reward_ce = F.cross_entropy(
                        pred_reward_logits.reshape(-1, args.reward_num_bins),
                        mb_reward_bins,
                        reduction='none',
                    ).view(mb_size, args.dyn_horizon)
                    reward_loss_bins = (reward_ce * step_weight).sum() / (step_weight.sum() + 1e-8)

                    # DreamerV3-style asymmetric label smoothing on the
                    # non-termination class: continue target is clamped DOWN
                    # to gamma (= 1 - eps) so the head's asymptote is bounded
                    # at logit ~ ln(gamma/(1-gamma)) instead of +∞ on envs
                    # that never terminate. Termination targets stay at 0.
                    continue_target = (1.0 - future_terminations).clamp(max=args.gamma)
                    continue_loss_raw = F.binary_cross_entropy_with_logits(
                        pred_continue_logits,
                        continue_target,
                        reduction="none",
                    )
                    continue_loss = (continue_loss_raw * step_weight).sum() / (step_weight.sum() + 1e-8)

                    # State prediction loss against frozen tokenizer's next
                    # obs embeds — this is the dreamer4-style "predict next
                    # tokenizer embeds" objective. It anchors the WM trunk
                    # against a fixed target, so collapse is impossible and
                    # no separate sigreg / transition-MSE is needed.
                    pred = agent.to_state_pred(pred_next_flat)
                    pred_mean, pred_log_var = pred[:, :agent.tok_obs_embed_dim], pred[:, agent.tok_obs_embed_dim:]
                    pred_var = pred_log_var.exp()
                    sp_loss_raw = F.gaussian_nll_loss(
                        pred_mean,
                        next_obs_embed_flat.reshape(mb_size * args.dyn_horizon, agent.tok_obs_embed_dim),
                        var=pred_var,
                        reduction='none',
                    ).mean(-1).view(mb_size, args.dyn_horizon)
                    state_pred_loss = (sp_loss_raw * latent_weight).sum() / (latent_weight.sum() + 1e-8)

                    # Per-frame trunk anchor: dyn_flat at each window step
                    # decodes to the frozen tokenizer's current-step obs
                    # tokens. Forces every dyn CLS to faithfully encode the
                    # current state — a tighter constraint than the predictor
                    # path alone provides.
                    trunk_anchor_pred = agent.dyn_to_obs(current_flat)
                    trunk_anchor_raw = F.mse_loss(
                        trunk_anchor_pred,
                        current_obs_embed_flat.reshape(mb_size * args.dyn_horizon, agent.tok_obs_embed_dim),
                        reduction='none',
                    ).mean(-1).view(mb_size, args.dyn_horizon)
                    trunk_anchor_loss = (trunk_anchor_raw * step_weight).sum() / (step_weight.sum() + 1e-8)

                    with torch.no_grad():
                        step_weight_d = step_weight.detach()
                        latent_weight_d = latent_weight.detach()
                        wm_horizon_weight += step_weight_d.sum(dim=0)
                        wm_horizon_latent_weight += latent_weight_d.sum(dim=0)
                        wm_horizon_reward_mse += (
                            (pred_rewards - future_rewards).square().detach() * step_weight_d
                        ).sum(dim=0)
                        wm_horizon_reward_bins += (reward_ce.detach() * step_weight_d).sum(dim=0)
                        wm_horizon_continue += (continue_loss_raw.detach() * step_weight_d).sum(dim=0)
                        wm_horizon_state_pred += (sp_loss_raw.detach() * latent_weight_d).sum(dim=0)
                        wm_horizon_trunk_anchor += (trunk_anchor_raw.detach() * step_weight_d).sum(dim=0)

                    # Equal-weight aggregation across heterogeneous categories.
                    aux_loss = (
                        equal_weight(reward_loss)
                        + equal_weight(reward_loss_bins)
                        + equal_weight(continue_loss)
                        + equal_weight(state_pred_loss)
                        + equal_weight(trunk_anchor_loss)
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
        )
        if use_imagination:
            with torch.no_grad(), bf16_autocast(device):
                # Subsample seed states for imagination — running the dream
                # loop over the full PPO rollout buffer would OOM on large
                # num_envs/num_steps configurations and is unnecessary for
                # Dreamer-style imagination.
                seed_pool = b_obs_seqs.shape[0]
                if 0 < args.dream_batch_size < seed_pool:
                    seed_inds = torch.randperm(seed_pool, device=device)[: args.dream_batch_size]
                    dream_seed_obs = b_obs_seqs[seed_inds]
                    dream_seed_alive = b_wm_valid[seed_inds]
                else:
                    dream_seed_obs = b_obs_seqs
                    dream_seed_alive = b_wm_valid

                # Chunk the seed-encode pass: at full capacity the entire
                # rollout buffer (num_envs * num_steps sequences) does not
                # fit through the WM transformer in a single forward.
                seed_chunk = max(args.minibatch_size, 256)
                seed_bundles = []
                for s_start in range(0, dream_seed_obs.shape[0], seed_chunk):
                    s_end = s_start + seed_chunk
                    chunk_cls = agent._encode_wm(dream_seed_obs[s_start:s_end])
                    seed_bundles.append(agent._dyn_bundle_from_cls(chunk_cls))
                z = torch.cat(seed_bundles, dim=0).detach()
                alive = dream_seed_alive.float()
                dream_states = []
                dream_actions = []
                dream_pre_actions = []
                dream_old_logprobs = []
                dream_values = []
                dream_rewards = []
                dream_continues = []
                dream_learn_masks = []
                summary_history = [z.detach()]
                action_history = []

                for _ in range(args.imagination_horizon):
                    dream_states.append(z.detach())
                    action, old_logprob, _, _, _, action_z = agent.get_imagined_action_and_value(z)
                    # Lambda returns are computed once per iteration under
                    # no_grad, freezing the bootstrap target during SGD.
                    value = agent.get_imagined_value(z, hl_support)
                    action_history.append(action.detach())
                    context_len = min(agent.pred_context, len(summary_history), len(action_history))
                    pred_context = torch.stack(summary_history[-context_len:], dim=1)
                    action_context = torch.stack(action_history[-context_len:], dim=1)
                    next_z = agent._predict_next_bundle_from_history(pred_context, action_context)
                    next_flat = next_z.flatten(1)
                    reward_input = agent._reward_input(z.flatten(1), action, next_flat)
                    reward_logits = agent.reward_head_bins(reward_input)
                    continue_logits = agent.continue_head(next_flat).squeeze(-1)
                    # Sigmoid in fp32 — bf16 (1 - p) underflows when p ~ 1.
                    continue_prob = continue_logits.float().sigmoid()
                    reward_hat = reward_support.to_scalar(reward_logits) * alive
                    continue_hat = continue_prob * alive
                    dream_actions.append(action.detach())
                    dream_pre_actions.append(action_z.detach())
                    dream_old_logprobs.append(old_logprob.detach())
                    dream_values.append(value.detach())
                    dream_rewards.append(reward_hat.detach())
                    dream_continues.append(continue_hat.detach())
                    dream_learn_masks.append(alive.bool())
                    sampled_terminal = (torch.bernoulli((1.0 - continue_prob).clamp(0.0, 1.0)) > 0).float()
                    alive = alive * (1.0 - sampled_terminal)
                    z = next_z.detach()
                    summary_history.append(z)

                bootstrap_value = agent.get_imagined_value(z, hl_support).detach()
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

                dream_horizon_return_mean = []
                dream_horizon_reward_mean = []
                dream_horizon_continue_mean = []
                for horizon_return, horizon_reward, horizon_continue, horizon_mask in zip(
                    dream_returns,
                    dream_rewards,
                    dream_continues,
                    dream_learn_masks,
                ):
                    if bool(horizon_mask.any()):
                        dream_horizon_return_mean.append(horizon_return[horizon_mask].mean())
                        dream_horizon_reward_mean.append(horizon_reward[horizon_mask].mean())
                        dream_horizon_continue_mean.append(horizon_continue[horizon_mask].mean())
                    else:
                        dream_horizon_return_mean.append(horizon_return.mean())
                        dream_horizon_reward_mean.append(horizon_reward.mean())
                        dream_horizon_continue_mean.append(horizon_continue.mean())
                dream_horizon_return_mean = torch.stack(dream_horizon_return_mean)
                dream_horizon_reward_mean = torch.stack(dream_horizon_reward_mean)
                dream_horizon_continue_mean = torch.stack(dream_horizon_continue_mean)

                dream_states = torch.cat(dream_states, dim=0)
                dream_actions = torch.cat(dream_actions, dim=0)
                dream_pre_actions = torch.cat(dream_pre_actions, dim=0)
                dream_old_logprobs = torch.cat(dream_old_logprobs, dim=0)
                dream_values = torch.cat(dream_values, dim=0)
                dream_returns = torch.cat(dream_returns, dim=0)
                dream_advantages = dream_returns - dream_values
                dream_learn_masks = torch.cat(dream_learn_masks, dim=0)
                if bool(dream_learn_masks.any()):
                    imagined_return_mean = dream_returns[dream_learn_masks].mean()
                else:
                    imagined_return_mean = dream_returns.mean()

            dream_inds = np.arange(dream_states.shape[0])
            dream_minibatch_size = max(args.minibatch_size, args.minibatch_size * args.imagination_horizon)
            imagine_actor_losses = []
            imagine_critic_losses = []
            imagine_approx_kls = []
            imagine_logprob_delta_abses = []
            for _ in range(args.imagine_update_epochs):
                np.random.shuffle(dream_inds)
                for start in range(0, dream_states.shape[0], dream_minibatch_size):
                    end = start + dream_minibatch_size
                    mb_inds = dream_inds[start:end]
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
                        # Project in fp32 (bf16 erf is too lossy near boundaries),
                        # then let F.cross_entropy handle the autocast promotion
                        # for the logits-vs-soft-target reduction.
                        with torch.amp.autocast("cuda", enabled=False):
                            dream_return_probs = hl_support.project(dream_return_targets.float())
                        dream_value_logits = agent.get_imagined_value_logits(dream_states[mb_inds])
                        dream_value_loss_bins = F.cross_entropy(
                            dream_value_logits, dream_return_probs.detach(), reduction='none'
                        )
                        if has_targets:
                            imagine_critic_loss = dream_value_loss_bins[mb_mask].mean()
                        else:
                            imagine_critic_loss = dream_value_loss_bins.sum() * 0.0

                        # equal_weight(0) = 0, so the sentinel paths produce no
                        # gradient — no special-casing needed.
                        imagine_loss = (
                            args.imagine_actor_coef * equal_weight(imagine_actor_loss)
                            + args.imagine_critic_coef * equal_weight(imagine_critic_loss)
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
        writer.add_scalar("worldmodel/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("worldmodel/reward_loss_bins", reward_loss_bins.item(), global_step)
        writer.add_scalar("worldmodel/continue_loss", continue_loss.item(), global_step)
        writer.add_scalar("worldmodel/trunk_anchor_loss", trunk_anchor_loss.item(), global_step)
        writer.add_scalar("state_pred/loss", state_pred_loss.item(), global_step)
        wm_horizon_denom = wm_horizon_weight.clamp_min(1e-8)
        wm_horizon_latent_denom = wm_horizon_latent_weight.clamp_min(1e-8)
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
            cls = agent._encode_wm(agent.obs_history)
            dyn_flat = agent._dyn_flat_from_cls(cls)
            _, action_std = agent._actor_mean_std(dyn_flat)
            writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
            sp_pred = agent.to_state_pred(dyn_flat)
            sp_log_var = sp_pred[:, agent.tok_obs_embed_dim:]
            writer.add_scalar("state_pred/log_var_mean", sp_log_var.mean().item(), global_step)
            writer.add_scalar("state_pred/entropy_bonus_mean", (sp_log_var.mean(-1) * STATE_ENT_COEF).mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
