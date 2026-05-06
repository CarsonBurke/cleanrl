# PPO + WM + SDE + state entropy + relu^2 heads + HL-Gauss value/reward encoding.
#
# Fork of the v4 wm_sde_stateent_twohot_v6 lineage.
# Replaces the auxiliary two-hot categorical targets with HL-Gauss projections
# and uses relu(x)^2 hidden layers in the actor, value/reward readout heads,
# and transition head. The transition head normalizes the final hidden state
# before projection to reduce closed-loop imagined latent scale drift.
# The policy, critic, reward, continue, and transition heads all consume a
# shared 4-token WM dynamics bundle. Real and imagined actions/values use the
# trainable WM latent basis for PPO/critic/dynamics, with a frozen pretrained
# tokenizer used only as an external next-observation embedding target.
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


EMBED_DIM = 32
NUM_HEADS = 4
FFN_MULT = 2
CONTEXT_LEN = 5
NUM_SPATIAL_BLOCKS = 3
NUM_TEMPORAL_BLOCKS = 2
DYN_TOKEN_NAMES = ['dyn0', 'dyn1', 'dyn2', 'dyn3']
WM_CLS_NAMES = DYN_TOKEN_NAMES
DYN_TOKEN_COUNT = len(DYN_TOKEN_NAMES)
DYN_FLAT_DIM = EMBED_DIM * DYN_TOKEN_COUNT
LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
WM_COEF = 1.0
DYNAMICS_VALUE_COEF = 0.5
DYNAMICS_VALUE_BINS_COEF = 0.5
DYNAMICS_SHAPE_COEF = 0.5
STATE_ENT_COEF = 0.0
STATE_PRED_LOSS_COEF = 0.1
SQUASH_EPS = 1e-6
PRED_NUM_BLOCKS = 2


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
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef_low: float = 0.2
    clip_coef_high: float = 0.28
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.15
    num_bins: int = 51
    v_min: float = -5.0
    v_max: float = 5.0
    sigma_ratio: float = 0.5
    use_symlog: bool = True
    reward_num_bins: int = 51
    reward_v_min: float = -10.0
    reward_v_max: float = 10.0
    reward_sigma_ratio: float = 0.75
    reward_use_symlog: bool = False
    tokenizer_path: str = ""
    tokenizer_checkpoint_prefix: str = ""
    dyn_horizon: int = 5
    pred_context: int = 5
    wm_update_epochs: int = 1
    sigreg_coef: float = 0.03
    sigreg_num_proj: int = 256
    sigreg_knots: int = 17
    sigreg_min_valid: int = 32
    imagination_horizon: int = 15
    imagine_start_step: int = 0
    imagine_update_epochs: int = 4
    imagine_actor_coef: float = 1.0
    imagine_critic_coef: float = 0.5
    imagine_actor_ent_coef: float = 0.0
    dream_spo_eps_low: float = 0.40
    dream_spo_eps_high: float = 0.56

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


def temporal_causal_mask(context_len, tokens_per_step, device):
    total_tokens = context_len * tokens_per_step
    token_steps = torch.arange(total_tokens, device=device) // tokens_per_step
    future = token_steps.unsqueeze(0) > token_steps.unsqueeze(1)
    mask = torch.zeros(total_tokens, total_tokens, device=device)
    return mask.masked_fill(future, float("-inf"))


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer for latent anti-collapse."""

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def sample_projection(self, dim, device, dtype):
        proj = torch.randn(dim, self.num_proj, device=device, dtype=dtype)
        return proj.div_(proj.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8))

    def forward(self, x):
        # x: [time, batch, dim]
        proj = self.sample_projection(x.size(-1), x.device, x.dtype)
        t = self.t.to(device=x.device, dtype=x.dtype)
        phi = self.phi.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        x_t = (x @ proj).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * x.size(-2)
        return statistic.mean()


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
    def __init__(self, obs_dim, context_len, cls_names):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_len = context_len
        self.cls_names = list(cls_names)
        self.num_cls_tokens = len(self.cls_names)
        for i, name in enumerate(self.cls_names):
            setattr(self, f'{name}_cls_index', obs_dim + i)

        self.value_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.input_norm = RMSNorm(EMBED_DIM)
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        self.register_buffer("dim_indices", torch.arange(obs_dim))

        cls_std = 1.0 / EMBED_DIM**0.5
        self.cls_params = nn.ParameterList([
            nn.Parameter(torch.empty(EMBED_DIM)) for _ in range(self.num_cls_tokens)
        ])
        for p in self.cls_params:
            nn.init.trunc_normal_(p, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

        init_scale = 1.0 / (2 * (NUM_SPATIAL_BLOCKS + NUM_TEMPORAL_BLOCKS)) ** 0.5
        self.s_blocks = nn.ModuleList(
            [SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_SPATIAL_BLOCKS)]
        )
        self.t_blocks = nn.ModuleList(
            [SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_TEMPORAL_BLOCKS)]
        )
        self.final_norm = RMSNorm(EMBED_DIM)

        head_dim = EMBED_DIM // NUM_HEADS
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
        obs_tokens = obs_tokens + self.dim_id_embed(self.dim_indices).view(1, 1, self.obs_dim, EMBED_DIM)

        cls_tokens = torch.stack(list(self.cls_params), dim=0)
        cls_tokens = cls_tokens.view(1, 1, self.num_cls_tokens, EMBED_DIM).expand(batch, time_steps, -1, -1)
        tokens = torch.cat([obs_tokens, cls_tokens], dim=2)
        tokens = self.input_norm(tokens)
        tokens0 = tokens

        tokens = self._spatial(tokens, tokens0, self.s_blocks[0])
        tokens = self._temporal(tokens, tokens0, self.t_blocks[0])
        tokens = self._spatial(tokens, tokens0, self.s_blocks[1])
        tokens = self._temporal(tokens, tokens0, self.t_blocks[1])
        tokens = self._spatial(tokens, tokens0, self.s_blocks[2])
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
        self.obs_embed_dim = obs_dim * EMBED_DIM
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN
        self.pred_context = pred_context

        self.wm_backbone = STSTSCLSBackbone(obs_dim, self.context_len, WM_CLS_NAMES)
        self.tokenizer_backbone = STSTSCLSBackbone(obs_dim, self.context_len, WM_CLS_NAMES)
        if tokenizer_path:
            self._load_tokenizer_checkpoint(tokenizer_path, tokenizer_checkpoint_prefix)
        else:
            raise ValueError("This variant requires --tokenizer-path from tokenizer pretraining.")
        self.tokenizer_backbone.eval().requires_grad_(False)

        # Actor and critic read the full shared WM dynamics-token bundle.
        self.actor_mean = relusq_mlp(DYN_FLAT_DIM, action_dim, 64, out_std=0.01)

        # Scalar critic — used for real-rollout GAE / PPO value loss
        self.critic = relusq_mlp(DYN_FLAT_DIM, 1, 64, out_std=1.0)

        # Bin critic — trained by HL-Gauss CE
        self.critic_bins = relusq_mlp(DYN_FLAT_DIM, num_bins, 64, out_std=1.0)

        # SDE on the full dynamics bundle.
        self.log_std_param = nn.Parameter(torch.zeros(DYN_FLAT_DIM, action_dim))
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

        # State prediction grounds dynamics in next-observation token embeddings.
        self.to_state_pred = nn.Sequential(
            RMSNorm(DYN_FLAT_DIM),
            nn.Linear(DYN_FLAT_DIM, self.obs_embed_dim * 2),
        )

        # Action-token contextual dynamics predictor.
        self.pred_action_proj = layer_init(nn.Linear(1, EMBED_DIM))
        self.pred_action_dim_embed = nn.Embedding(action_dim, EMBED_DIM)
        nn.init.trunc_normal_(self.pred_action_dim_embed.weight, std=1.0 / EMBED_DIM**0.5)
        self.pred_pos_embed = nn.Parameter(torch.empty(self.pred_context * (action_dim + DYN_TOKEN_COUNT), EMBED_DIM))
        nn.init.trunc_normal_(self.pred_pos_embed, std=1.0 / EMBED_DIM**0.5)
        pred_init_scale = 1.0 / (2 * PRED_NUM_BLOCKS) ** 0.5
        self.pred_blocks = nn.ModuleList(
            [SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, pred_init_scale) for _ in range(PRED_NUM_BLOCKS)]
        )
        self.pred_final_norm = RMSNorm(EMBED_DIM)
        self.pred_next_proj = layer_init(nn.Linear(EMBED_DIM, EMBED_DIM))

        # One-step fallback head is retained for compatibility diagnostics.
        self.transition = relusq_mlp_prenorm_out(EMBED_DIM * DYN_TOKEN_COUNT + action_dim, EMBED_DIM * DYN_TOKEN_COUNT, 256)
        # Dynamics-value heads — trained on real GAE returns,
        # then used as the value bootstrap/readout inside imagined rollouts.
        self.dynamics_value = relusq_mlp(DYN_FLAT_DIM, 1, 64, out_std=1.0)
        self.dynamics_value_bins = relusq_mlp(DYN_FLAT_DIM, num_bins, 64, out_std=1.0)
        # Reward reads current latent, action features, and predicted next latent.
        reward_input_dim = DYN_FLAT_DIM * 2 + action_dim * 2 + 1
        self.reward_head = relusq_mlp(reward_input_dim, 1, 128)
        self.reward_head_bins = relusq_mlp(reward_input_dim, reward_num_bins, 128)
        self.continue_head = relusq_mlp(DYN_FLAT_DIM, 1, 128)
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
            context_len = self.pred_context
        action_tokens = self.pred_action_proj(action_history.unsqueeze(-1))
        action_tokens = action_tokens + self.pred_action_dim_embed.weight.view(1, 1, self.action_dim, EMBED_DIM)
        tokens = torch.cat([action_tokens, latent_history], dim=2)
        tokens_per_step = self.action_dim + num_tokens
        tokens = tokens.reshape(batch, context_len * tokens_per_step, width)
        tokens = tokens + self.pred_pos_embed[: tokens.shape[1]].view(1, tokens.shape[1], width)
        x0 = tokens
        attn_mask = temporal_causal_mask(context_len, tokens_per_step, tokens.device)
        for block in self.pred_blocks:
            tokens = block(tokens, x0, attn_mask=attn_mask)
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

    def transition_step(self, z_bundle, action):
        z_flat = z_bundle.flatten(1)
        next_bundle = self._predict_next_bundle_from_history(z_bundle.unsqueeze(1), action.unsqueeze(1))
        next_flat = next_bundle.flatten(1)
        reward_input = self._reward_input(z_flat, action, next_flat)
        reward = self.reward_head(reward_input).squeeze(-1)
        reward_logits = self.reward_head_bins(reward_input)
        continue_logits = self.continue_head(next_flat).squeeze(-1)
        return next_bundle, reward, reward_logits, continue_logits

    def _get_action_std(self, sde_latent):
        sde_latent = torch.tanh(sde_latent)
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std_sq = log_std.exp().pow(2)
        action_var = sde_latent.pow(2) @ std_sq
        return (action_var + 1e-6).sqrt()

    def _action_std_fixed(self):
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return log_std.exp().pow(2).mean(0).sqrt()

    def _u_to_action(self, u):
        center = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return center + half_range * u

    def _action_to_u(self, action):
        center = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return ((action - center) / half_range).clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)

    def _squashed_log_prob_entropy(self, mean, std, action=None):
        normal = Normal(mean, std)
        if action is None:
            z = normal.rsample()
            u = torch.tanh(z)
            action = self._u_to_action(u)
        else:
            u = self._action_to_u(action)
            z = torch.atanh(u)
        log_prob_z = normal.log_prob(z).sum(-1)
        squash_correction = torch.log(1.0 - u.pow(2) + SQUASH_EPS).sum(-1)
        log_prob = log_prob_z - squash_correction
        entropy = normal.entropy().sum(-1)
        return action, log_prob, entropy

    def get_state_entropy_bonus(self, state_pred_latent):
        pred = self.to_state_pred(state_pred_latent)
        log_var = pred[:, self.obs_embed_dim:]
        return log_var.mean(dim=-1) * STATE_ENT_COEF

    def get_value(self, obs_seq):
        cls = self._encode_wm(obs_seq)
        return self.critic(self._dyn_flat_from_cls(cls))

    def get_action_and_value(self, obs_seq, action=None):
        wm_cls = self._encode_wm(obs_seq)
        dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
        dyn_flat = dyn_bundle.flatten(1)
        action_mean = self.actor_mean(dyn_flat)
        action_std = self._get_action_std(dyn_flat)
        action, log_prob, entropy = self._squashed_log_prob_entropy(action_mean, action_std, action)
        return action, log_prob, entropy, self.critic(dyn_flat), dyn_bundle, dyn_flat

    def get_all_for_update(self, obs_seq, action):
        wm_cls = self._encode_wm(obs_seq)
        dyn_bundle = self._dyn_bundle_from_cls(wm_cls)
        dyn_flat = dyn_bundle.flatten(1)
        action_mean = self.actor_mean(dyn_flat)
        action_std = self._get_action_std(dyn_flat)
        _, log_prob, entropy = self._squashed_log_prob_entropy(action_mean, action_std, action)
        return (
            log_prob,
            entropy,
            self.critic(dyn_flat),                      # scalar value for GAE
            self.critic_bins(dyn_flat),                 # bin logits for HL-Gauss CE
            dyn_bundle,                                 # 4-token dynamics bundle
            self.dynamics_value(dyn_flat),
            self.dynamics_value_bins(dyn_flat),
            dyn_flat,                                   # reward-bin head input
            dyn_flat,                                   # state-pred head input
        )

    def get_imagined_action_and_value(self, dyn_bundle, action=None):
        dyn_flat = dyn_bundle.flatten(1)
        action_mean = self.actor_mean(dyn_flat)
        action_std = self._get_action_std(dyn_flat)
        action, log_prob, entropy = self._squashed_log_prob_entropy(action_mean, action_std, action)
        return action, log_prob, entropy, self.dynamics_value(dyn_flat).squeeze(-1), self.dynamics_value_bins(dyn_flat)

    def get_imagined_value(self, dyn_bundle, hl_support):
        return hl_support.to_scalar(self.get_imagined_value_logits(dyn_bundle))

    def get_imagined_value_logits(self, dyn_bundle):
        return self.dynamics_value_bins(dyn_bundle.flatten(1))

    def imagine_rollout(self, z_start, horizon, gamma, gae_lambda):
        """Roll out latent dynamics and compute Dreamer-style lambda targets."""
        action_std = self._action_std_fixed().detach()
        z = z_start
        latents = []
        rewards = []
        values = []
        continues = []

        for _ in range(horizon):
            latents.append(z)
            z_flat = z.flatten(1)
            a_mean = self.actor_mean(z_flat)
            a = self._u_to_action(torch.tanh(Normal(a_mean, action_std).rsample()))
            z_next, reward, _, continue_logits = self.transition_step(z, a)
            rewards.append(reward)
            continues.append(continue_logits.sigmoid())
            values.append(self.dynamics_value(z_flat).squeeze(-1))
            z = z_next

        values.append(self.dynamics_value(z.flatten(1)).squeeze(-1))

        imagine_return = values[-1]
        lambda_returns = []
        for t in reversed(range(horizon)):
            imagine_return = rewards[t] + gamma * continues[t] * (
                gae_lambda * imagine_return + (1.0 - gae_lambda) * values[t + 1]
            )
            lambda_returns.append(imagine_return)
        lambda_returns.reverse()

        rewards = torch.stack(rewards)
        continues = torch.stack(continues)
        values = torch.stack(values)
        latents = torch.stack(latents)
        lambda_returns = torch.stack(lambda_returns)
        discounts = gamma * continues.detach()
        weights = torch.cat([
            torch.ones_like(discounts[:1]),
            torch.cumprod(discounts[:-1], dim=0),
        ], dim=0)

        return {
            "latents": latents,
            "rewards": rewards,
            "continues": continues,
            "values": values[:-1],
            "bootstrap": values[-1],
            "lambda_returns": lambda_returns,
            "weights": weights,
        }

    def imagine(self, z_start, horizon, gamma, gae_lambda):
        rollout = self.imagine_rollout(z_start, horizon, gamma, gae_lambda)
        return rollout["lambda_returns"], list(rollout["values"])

    def imagine_value_targets(self, z_start, horizon, gamma, gae_lambda):
        """Roll out frozen dynamics and train dynamics_value on imagined lambda returns."""
        with torch.no_grad():
            rollout = self.imagine_rollout(z_start, horizon, gamma, gae_lambda)

        return rollout["latents"], rollout["lambda_returns"], rollout["weights"]


def clip_agent_grad_norms(agent, max_grad_norm):
    """Clip shared trunk and each head independently to avoid cross-head starvation."""
    groups = [
        agent.wm_backbone.parameters(),
        agent.actor_mean.parameters(),
        [agent.log_std_param],
        agent.critic.parameters(),
        agent.critic_bins.parameters(),
        agent.dynamics_value.parameters(),
        agent.dynamics_value_bins.parameters(),
        agent.to_state_pred.parameters(),
        agent.pred_action_proj.parameters(),
        agent.pred_action_dim_embed.parameters(),
        agent.pred_blocks.parameters(),
        agent.pred_final_norm.parameters(),
        agent.pred_next_proj.parameters(),
        agent.transition.parameters(),
        agent.reward_head.parameters(),
        agent.reward_head_bins.parameters(),
        agent.continue_head.parameters(),
    ]
    for params in groups:
        nn.utils.clip_grad_norm_(params, max_grad_norm)


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

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
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
    optimizer = optim.Adam((p for p in agent.parameters() if p.requires_grad), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.sigma_ratio,
        device,
        use_symlog=args.use_symlog,
    )
    reward_support = HLGaussSupport(
        args.reward_num_bins,
        args.reward_v_min,
        args.reward_v_max,
        args.reward_sigma_ratio,
        device,
        use_symlog=args.reward_use_symlog,
    )
    sigreg = SIGReg(args.sigreg_knots, args.sigreg_num_proj).to(device)

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    next_obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
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
            with torch.no_grad():
                action, logprob, _, value, _, state_pred_latent = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
                ent_bonus = agent.get_state_entropy_bonus(state_pred_latent)
            actions[step] = action
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

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(agent.obs_history).reshape(1, -1)
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
        clipfracs = []
        stop_policy_updates = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if not stop_policy_updates:
                    (
                        newlogprob,
                        entropy,
                        newvalue,
                        value_logits,
                        _dynamics_latent_unused,
                        _dynamics_value_unused,
                        _dynamics_value_logits_unused,
                        _reward_bins_latent_unused,
                        _state_pred_latent_unused,
                    ) = agent.get_all_for_update(b_obs_seqs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipped_low = ratio < (1.0 - args.clip_coef_low)
                        clipped_high = ratio > (1.0 + args.clip_coef_high)
                        clipfracs += [(clipped_low | clipped_high).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef_low,
                        1.0 + args.clip_coef_high,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef_low,
                            args.clip_coef_low,
                        )
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    v_loss_bins = F.cross_entropy(value_logits, b_return_bins[mb_inds], reduction='mean')

                    # Policy update also shapes the trainable WM latent basis.
                    policy_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + v_loss_bins
                    optimizer.zero_grad()
                    policy_loss.backward()
                    clip_agent_grad_norms(agent, args.max_grad_norm)
                    optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        stop_policy_updates = True

        # World-model phase: run after PPO so aux updates do not invalidate
        # rollout log-prob ratios inside the PPO epoch.
        for wm_epoch in range(args.wm_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
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

                window_obs_seqs = torch.cat([b_obs_seqs[mb_inds].unsqueeze(1), future_next_obs_seqs], dim=1)
                window_cls = agent._encode_wm(window_obs_seqs.reshape(-1, agent.context_len, obs_dim))
                window_bundle = agent._dyn_bundle_from_cls(window_cls).reshape(
                    mb_size, args.dyn_horizon + 1, DYN_TOKEN_COUNT, EMBED_DIM
                )
                dynamics_bundle = window_bundle[:, 0]
                dynamics_flat = dynamics_bundle.flatten(1)
                current_bundles = window_bundle[:, :-1]
                target_next_bundles = window_bundle[:, 1:]
                current_flat = current_bundles.reshape(mb_size * args.dyn_horizon, DYN_FLAT_DIM)
                flat_future_actions = future_actions.reshape(mb_size * args.dyn_horizon, action_dim)

                pred_next_bundles = agent.predict_next_bundles_teacher_forced(
                    current_bundles,
                    future_actions,
                )
                pred_next_flat = pred_next_bundles.reshape(mb_size * args.dyn_horizon, DYN_FLAT_DIM)
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
                        mb_size, args.dyn_horizon, agent.obs_embed_dim
                    )

                mb_return_bins = b_return_bins[mb_inds]

                # Dynamics-return loss. This is the value head
                # used in imagined rollouts; reward prediction remains immediate.
                dynamics_value_input = dynamics_flat.detach()
                dynamics_value = agent.dynamics_value(dynamics_value_input).view(-1)
                dynamics_value_logits = agent.dynamics_value_bins(dynamics_value_input)
                dynamics_value_loss = 0.5 * ((dynamics_value - b_returns[mb_inds]) ** 2).mean()
                dynamics_value_loss_bins = F.cross_entropy(dynamics_value_logits, mb_return_bins, reduction='mean')

                mb_valid = b_wm_valid[mb_inds]

                per_step_transition_loss = F.mse_loss(
                    pred_next_bundles,
                    target_next_bundles,
                    reduction="none",
                ).mean(dim=(2, 3))
                transition_loss = (
                    per_step_transition_loss * latent_weight
                ).sum() / (latent_weight.sum() + 1e-8)
                sigreg_terms = []
                window_flat = window_bundle.reshape(mb_size, args.dyn_horizon + 1, DYN_FLAT_DIM)
                sigreg_masks = [torch.ones(mb_size, device=device, dtype=torch.bool)]
                sigreg_masks.extend([latent_weight[:, horizon_idx] > 0.0 for horizon_idx in range(args.dyn_horizon)])
                for horizon_idx, sigreg_mask in enumerate(sigreg_masks):
                    if int(sigreg_mask.sum().item()) >= args.sigreg_min_valid:
                        sigreg_terms.append(sigreg(window_flat[sigreg_mask, horizon_idx].unsqueeze(0)))
                sigreg_loss = torch.stack(sigreg_terms).mean() if sigreg_terms else window_flat.sum() * 0.0
                reward_loss = (
                    (pred_rewards - future_rewards).square() * step_weight
                ).sum() / (step_weight.sum() + 1e-8)

                # HL-Gauss reward loss
                with torch.no_grad():
                    mb_reward_bins = reward_support.project(future_rewards.reshape(-1))
                reward_ce = F.cross_entropy(
                    pred_reward_logits.reshape(-1, args.reward_num_bins),
                    mb_reward_bins,
                    reduction='none',
                ).view(mb_size, args.dyn_horizon)
                reward_loss_bins = (reward_ce * step_weight).sum() / (step_weight.sum() + 1e-8)

                continue_loss_raw = F.binary_cross_entropy_with_logits(
                    pred_continue_logits,
                    1.0 - future_terminations,
                    reduction="none",
                )
                continue_loss = (continue_loss_raw * step_weight).sum() / (step_weight.sum() + 1e-8)
                wm_loss = transition_loss + reward_loss + continue_loss + reward_loss_bins

                # State prediction loss
                pred = agent.to_state_pred(pred_next_flat)
                pred_mean, pred_log_var = pred[:, :agent.obs_embed_dim], pred[:, agent.obs_embed_dim:]
                pred_var = pred_log_var.exp()
                sp_loss_raw = F.gaussian_nll_loss(
                    pred_mean,
                    next_obs_embed_flat.reshape(mb_size * args.dyn_horizon, agent.obs_embed_dim),
                    var=pred_var,
                    reduction='none',
                ).mean(-1).view(mb_size, args.dyn_horizon)
                state_pred_loss = (sp_loss_raw * latent_weight).sum() / (latent_weight.sum() + 1e-8)

                critic_loss = DYNAMICS_VALUE_COEF * dynamics_value_loss + DYNAMICS_VALUE_BINS_COEF * dynamics_value_loss_bins
                dynamics_shape_loss = (
                    + WM_COEF * wm_loss
                    + STATE_PRED_LOSS_COEF * state_pred_loss
                    + args.sigreg_coef * sigreg_loss
                )
                aux_loss = critic_loss + DYNAMICS_SHAPE_COEF * dynamics_shape_loss

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
        use_imagination = (
            global_step >= args.imagine_start_step
            and (args.imagine_actor_coef != 0.0 or args.imagine_critic_coef != 0.0)
        )
        if use_imagination:
            with torch.no_grad():
                seed_cls = agent._encode_wm(b_obs_seqs)
                z = agent._dyn_bundle_from_cls(seed_cls).detach()
                alive = b_wm_valid.float()
                dream_states = []
                dream_actions = []
                dream_old_logprobs = []
                dream_values = []
                dream_rewards = []
                dream_continues = []
                dream_learn_masks = []
                summary_history = [z.detach()]
                action_history = []

                for _ in range(args.imagination_horizon):
                    dream_states.append(z.detach())
                    action, old_logprob, _, _, _ = agent.get_imagined_action_and_value(z)
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
                    continue_prob = continue_logits.sigmoid()
                    reward_hat = reward_support.to_scalar(reward_logits) * alive
                    continue_hat = continue_prob * alive
                    dream_actions.append(action.detach())
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

                dream_states = torch.cat(dream_states, dim=0)
                dream_actions = torch.cat(dream_actions, dim=0)
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

                    if args.imagine_actor_coef != 0.0:
                        _, new_logprob, dream_entropy, _, _ = agent.get_imagined_action_and_value(
                            dream_states[mb_inds],
                            dream_actions[mb_inds],
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
                            torch.full_like(mb_advantages, args.dream_spo_eps_high),
                            torch.full_like(mb_advantages, args.dream_spo_eps_low),
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
                    dream_return_probs = hl_support.project(dream_return_targets)
                    dream_value_logits = agent.get_imagined_value_logits(dream_states[mb_inds])
                    dream_value_loss_bins = -(
                        dream_return_probs.detach() * torch.log_softmax(dream_value_logits, dim=-1)
                    ).sum(dim=-1)
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

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/value_loss_bins", v_loss_bins.item(), global_step)
        writer.add_scalar("losses/dynamics_value_loss", dynamics_value_loss.item(), global_step)
        writer.add_scalar("losses/dynamics_value_loss_bins", dynamics_value_loss_bins.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("worldmodel/transition_loss", transition_loss.item(), global_step)
        writer.add_scalar("worldmodel/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("worldmodel/reward_loss_bins", reward_loss_bins.item(), global_step)
        writer.add_scalar("worldmodel/continue_loss", continue_loss.item(), global_step)
        writer.add_scalar("worldmodel/sigreg_loss", sigreg_loss.item(), global_step)
        writer.add_scalar("worldmodel/total_loss", wm_loss.item(), global_step)
        writer.add_scalar("state_pred/loss", state_pred_loss.item(), global_step)
        writer.add_scalar("imagination/actor_loss", imagine_actor_loss.item(), global_step)
        writer.add_scalar("imagination/critic_loss", imagine_critic_loss.item(), global_step)
        writer.add_scalar("imagination/post_actor_approx_kl", imagine_approx_kl.item(), global_step)
        writer.add_scalar("imagination/post_actor_logprob_delta_abs", imagine_logprob_delta_abs.item(), global_step)
        with torch.no_grad():
            writer.add_scalar("imagination/mean_return", imagined_return_mean.item(), global_step)
            action_std = agent._action_std_fixed()
            writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
            cls = agent._encode_wm(agent.obs_history)
            dyn_flat = agent._dyn_flat_from_cls(cls)
            sp_pred = agent.to_state_pred(dyn_flat)
            sp_log_var = sp_pred[:, agent.obs_embed_dim:]
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
