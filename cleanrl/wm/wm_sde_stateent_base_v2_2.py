# PPO + WM + direct-var SDE + state entropy bonus.
#
# Fork of worldmodel_sde4cls_stateent with simplified variance parameterization.
# Instead of the indirect quadratic form (tanh(sde)² @ exp(W)²), the SDE CLS
# is projected directly to per-action log_var via a single linear layer.
# The backbone handles all state-dependent feature extraction; the head just reads it out.
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


EMBED_DIM = 32
NUM_HEADS = 4
FFN_MULT = 2
CONTEXT_LEN = 5
NUM_SPATIAL_BLOCKS = 3
NUM_TEMPORAL_BLOCKS = 2
NUM_CLS_TOKENS = 4
LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
IMAGINATION_HORIZON = 15
WM_COEF = 1.0
IMAGINE_COEF = 0.1
IMAGINE_CRITIC_COEF = 0.5
N_IMAGINE_SEEDS = 256
STATE_ENT_COEF = 0.01
STATE_PRED_LOSS_COEF = 0.1


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
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

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
        env = gym.wrappers.ClipAction(env)
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

        self.ffn_pre_norm = RMSNorm(dim)
        self.ffn_post_norm = RMSNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_value = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=False)

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

    def forward(self, x, rope_cos=None, rope_sin=None, is_causal=False):
        batch, steps, width = x.shape
        h = self.attn_pre_norm(x)
        q = self.q_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).reshape(batch, steps, width)
        x = x + self.attn_post_norm(self.out_proj(attn))

        h = self.ffn_pre_norm(x)
        ffn = F.silu(self.ffn_gate(h)) * self.ffn_value(h)
        x = x + self.ffn_post_norm(self.ffn_out(ffn))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult=2, init_scale=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_pre_norm = RMSNorm(dim)
        self.kv_pre_norm = RMSNorm(dim)
        self.attn_post_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.ffn_pre_norm = RMSNorm(dim)
        self.ffn_post_norm = RMSNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_value = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=False)

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

    def forward(self, x, context):
        batch, steps, width = x.shape
        context_steps = context.shape[1]
        q = self.q_proj(self.q_pre_norm(x)).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_pre_norm(context)
        k = self.k_proj(kv).view(batch, context_steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch, context_steps, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, steps, width)
        x = x + self.attn_post_norm(self.out_proj(attn))

        h = self.ffn_pre_norm(x)
        ffn = F.silu(self.ffn_gate(h)) * self.ffn_value(h)
        x = x + self.ffn_post_norm(self.ffn_out(ffn))
        return x


class STSTSCLSBackbone(nn.Module):
    def __init__(self, obs_dim, action_dim, context_len):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.num_modality_tokens = obs_dim + 2

        self.value_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.action_proj = layer_init(nn.Linear(action_dim, EMBED_DIM), std=1.0)
        self.reward_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.input_norm = RMSNorm(EMBED_DIM)
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        self.register_buffer("dim_indices", torch.arange(obs_dim))
        cls_std = 1.0 / EMBED_DIM**0.5
        self.action_token = nn.Parameter(torch.empty(EMBED_DIM))
        self.reward_token = nn.Parameter(torch.empty(EMBED_DIM))
        self.actor_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.critic_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.dynamics_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.state_pred_cls = nn.Parameter(torch.empty(EMBED_DIM))
        for p in [
            self.action_token,
            self.reward_token,
            self.actor_cls,
            self.critic_cls,
            self.dynamics_cls,
            self.state_pred_cls,
        ]:
            nn.init.trunc_normal_(p, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

        init_scale = 1.0 / (2 * (NUM_SPATIAL_BLOCKS + NUM_TEMPORAL_BLOCKS)) ** 0.5
        self.modality_s_blocks = nn.ModuleList(
            [SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_SPATIAL_BLOCKS)]
        )
        self.cls_cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_SPATIAL_BLOCKS)]
        )
        self.cls_s_blocks = nn.ModuleList(
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

    def _spatial(self, tokens, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.reshape(batch * time_steps, slots, width)
        x = block(x)
        return x.reshape(batch, time_steps, slots, width)

    def _spatial_cross(self, query_tokens, context_tokens, block):
        batch, time_steps, query_slots, width = query_tokens.shape
        context_slots = context_tokens.shape[2]
        q = query_tokens.reshape(batch * time_steps, query_slots, width)
        ctx = context_tokens.reshape(batch * time_steps, context_slots, width)
        x = block(q, ctx)
        return x.reshape(batch, time_steps, query_slots, width)

    def _temporal(self, tokens, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x = block(x, rope_cos=self.temporal_cos, rope_sin=self.temporal_sin, is_causal=True)
        x = x.reshape(batch, slots, time_steps, width).permute(0, 2, 1, 3)
        return x

    def encode_state(self, obs_seq):
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1))
        return obs_tokens + self.dim_id_embed(self.dim_indices).view(1, 1, self.obs_dim, EMBED_DIM)

    def forward_state_tokens(self, state_tokens, action_seq, reward_seq):
        batch, time_steps = state_tokens.shape[:2]
        action_tokens = self.action_proj(action_seq).unsqueeze(2) + self.action_token.view(1, 1, 1, EMBED_DIM)
        reward_tokens = self.reward_proj(reward_seq).unsqueeze(2) + self.reward_token.view(1, 1, 1, EMBED_DIM)

        cls_tokens = torch.stack(
            [self.actor_cls, self.critic_cls, self.dynamics_cls, self.state_pred_cls], dim=0
        )
        cls_tokens = cls_tokens.view(1, 1, NUM_CLS_TOKENS, EMBED_DIM).expand(batch, time_steps, -1, -1)
        modality_tokens = torch.cat([state_tokens, action_tokens, reward_tokens], dim=2)
        tokens = torch.cat([modality_tokens, cls_tokens], dim=2)
        tokens = self.input_norm(tokens)
        modality_tokens = tokens[:, :, :self.num_modality_tokens]
        cls_tokens = tokens[:, :, self.num_modality_tokens:]

        for block_idx in range(NUM_SPATIAL_BLOCKS):
            modality_tokens = self._spatial(modality_tokens, self.modality_s_blocks[block_idx])
            cls_tokens = self._spatial_cross(cls_tokens, modality_tokens, self.cls_cross_blocks[block_idx])
            cls_tokens = self._spatial(cls_tokens, self.cls_s_blocks[block_idx])
            if block_idx < NUM_TEMPORAL_BLOCKS:
                tokens = torch.cat([modality_tokens, cls_tokens], dim=2)
                tokens = self._temporal(tokens, self.t_blocks[block_idx])
                modality_tokens = tokens[:, :, :self.num_modality_tokens]
                cls_tokens = tokens[:, :, self.num_modality_tokens:]

        tokens = self.final_norm(torch.cat([modality_tokens, cls_tokens], dim=2))
        modality_tokens = tokens[:, :, :self.num_modality_tokens]
        cls_tokens = tokens[:, :, self.num_modality_tokens:]
        state_tokens = modality_tokens[:, :, :self.obs_dim]

        actor_cls = cls_tokens[:, -1, 0]
        critic_cls = cls_tokens[:, -1, 1]
        dynamics_cls = cls_tokens[:, -1, 2]
        state_pred_cls = cls_tokens[:, -1, 3]
        return actor_cls, critic_cls, dynamics_cls, state_pred_cls, state_tokens

    def forward(self, obs_seq, action_seq, reward_seq):
        state_tokens = self.encode_state(obs_seq)
        return self.forward_state_tokens(state_tokens, action_seq, reward_seq)


class Agent(nn.Module):
    def __init__(self, envs, num_envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN

        self.backbone = STSTSCLSBackbone(obs_dim, action_dim, self.context_len)
        self.policy_mean_log_var = layer_init(nn.Linear(EMBED_DIM, action_dim * 2), std=0.01)
        with torch.no_grad():
            self.policy_mean_log_var.bias[action_dim:].fill_(2.0 * LOG_STD_INIT)
        self.critic = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)

        # State prediction head: Gaussian (mean, log_var) over next dynamics latent
        self.to_state_pred = nn.Sequential(
            RMSNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, EMBED_DIM * 2),
        )

        # World model heads
        self.next_state_token_head = nn.Sequential(
            RMSNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, EMBED_DIM * 2),
            nn.SiLU(),
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
        )
        cls_std = 1.0 / EMBED_DIM**0.5
        self.reward_readout_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.continue_readout_cls = nn.Parameter(torch.empty(EMBED_DIM))
        nn.init.trunc_normal_(self.reward_readout_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.continue_readout_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        self.next_state_readout = CrossAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale=1.0)
        self.reward_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))
        self.register_buffer("action_history", torch.zeros(num_envs, self.context_len, action_dim))
        self.register_buffer("reward_history", torch.zeros(num_envs, self.context_len, 1))

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
        if action is None:
            action = torch.zeros(obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype)
        if reward is None:
            reward = torch.zeros(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)
        self.action_history = torch.cat([self.action_history[:, 1:], action.unsqueeze(1)], dim=1)
        self.reward_history = torch.cat([self.reward_history[:, 1:], reward.unsqueeze(1)], dim=1)

    def _encode(self, obs_seq, action_seq, reward_seq):
        return self.backbone(obs_seq, action_seq, reward_seq)

    def encode_state(self, obs_seq):
        return self.backbone.encode_state(obs_seq)

    def _encode_state_tokens(self, state_token_seq, action_seq, reward_seq):
        return self.backbone.forward_state_tokens(state_token_seq, action_seq, reward_seq)

    def _get_policy_dist(self, actor_latent):
        action_mean, action_log_var = self.policy_mean_log_var(actor_latent).chunk(2, dim=-1)
        clamped_log_var = -6.0 + 5.0 * torch.sigmoid(action_log_var)
        action_std = (0.5 * clamped_log_var).exp()
        return Normal(action_mean, action_std), action_mean, action_log_var, action_std

    def _action_std_fixed(self):
        """Fixed std for imagination (no SDE CLS available)."""
        return torch.full((self.action_dim,), torch.exp(torch.tensor(LOG_STD_INIT)).item(),
                          device=next(self.parameters()).device)

    def get_state_entropy_bonus(self, state_pred_latent):
        pred = self.to_state_pred(state_pred_latent)
        log_var = pred[:, EMBED_DIM:]
        return log_var.mean(dim=-1) * STATE_ENT_COEF

    def get_value(self, obs_seq, action_seq, reward_seq):
        _, critic_latent, _, _, _ = self._encode(obs_seq, action_seq, reward_seq)
        return self.critic(critic_latent)

    def get_action_and_value(self, obs_seq, action_seq, reward_seq, action=None):
        actor_latent, critic_latent, dynamics_latent, state_pred_latent, _ = self._encode(
            obs_seq, action_seq, reward_seq
        )
        probs, action_mean, action_log_var, action_std = self._get_policy_dist(actor_latent)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(critic_latent),
            dynamics_latent,
            state_pred_latent,
        )

    def get_all_for_update(self, obs_seq, action_seq, reward_seq, action):
        actor_latent, critic_latent, dynamics_latent, state_pred_latent, state_tokens = self._encode(
            obs_seq, action_seq, reward_seq
        )
        probs, action_mean, action_log_var, action_std = self._get_policy_dist(actor_latent)
        return (
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(critic_latent),
            dynamics_latent,
            state_pred_latent,
            state_tokens[:, -1],
        )

    def predict_next_state_tokens(self, state_tokens):
        token_delta = self.next_state_token_head(state_tokens)
        return state_tokens + token_delta

    def readout_next_state(self, state_tokens):
        queries = torch.stack([self.reward_readout_cls, self.continue_readout_cls], dim=0)
        queries = queries.unsqueeze(0).expand(state_tokens.shape[0], -1, -1)
        readouts = self.next_state_readout(queries, state_tokens)
        return readouts[:, 0], readouts[:, 1]

    def imagine(self, state_token_seq, action_seq, reward_seq, horizon, gamma, gae_lambda):
        rewards = []
        values = []
        continues = []

        for _ in range(horizon):
            actor_latent, critic_latent, dynamics_latent, _, state_tokens = self._encode_state_tokens(
                state_token_seq, action_seq, reward_seq
            )
            dist, a_mean, _, a_std = self._get_policy_dist(actor_latent)
            a = dist.rsample()
            next_state_tokens = self.predict_next_state_tokens(state_tokens[:, -1])
            reward_latent, continue_latent = self.readout_next_state(next_state_tokens)
            rewards.append(self.reward_head(reward_latent).squeeze(-1))
            continues.append(self.continue_head(continue_latent).sigmoid().squeeze(-1))
            values.append(self.critic(critic_latent).squeeze(-1))
            state_token_seq = torch.cat([state_token_seq[:, 1:], next_state_tokens.unsqueeze(1)], dim=1)
            action_seq = torch.cat([action_seq[:, 1:], a.unsqueeze(1)], dim=1)
            reward_seq = torch.cat([reward_seq[:, 1:], rewards[-1].unsqueeze(1).unsqueeze(-1)], dim=1)

        _, critic_latent, _, _, _ = self._encode_state_tokens(state_token_seq, action_seq, reward_seq)
        values.append(self.critic(critic_latent).squeeze(-1))

        imagine_return = values[-1]
        lambda_returns = []
        for t in reversed(range(horizon)):
            imagine_return = rewards[t] + gamma * continues[t] * (
                gae_lambda * imagine_return + (1.0 - gae_lambda) * values[t + 1]
            )
            lambda_returns.append(imagine_return)
        lambda_returns.reverse()

        return lambda_returns, values[:-1]


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

    agent = Agent(envs, args.num_envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    action_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, action_dim), device=device)
    reward_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, 1), device=device)
    next_obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    next_action_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, action_dim), device=device)
    next_reward_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, 1), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_dones = torch.zeros((args.num_steps, args.num_envs), device=device)
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
            with torch.no_grad():
                action, logprob, _, value, _, state_pred_latent = agent.get_action_and_value(
                    agent.obs_history,
                    agent.action_history,
                    agent.reward_history,
                )
                values[step] = value.flatten()
                ent_bonus = agent.get_state_entropy_bonus(state_pred_latent)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            reward_tensor = torch.as_tensor(reward, device=device).view(-1)
            rewards[step] = reward_tensor + ent_bonus
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
            next_dones[step] = next_done
            history_action = action.detach()
            history_reward = rewards[step].unsqueeze(-1)
            if next_done.any():
                done_mask = next_done.bool()
                agent.reset_history(done_mask)
                history_action = history_action.clone()
                history_reward = history_reward.clone()
                history_action[done_mask] = 0.0
                history_reward[done_mask] = 0.0
            agent.update_history(next_obs, history_action, history_reward)
            next_obs_seqs[step] = agent.obs_history.clone()
            next_action_seqs[step] = agent.action_history.clone()
            next_reward_seqs[step] = agent.reward_history.clone()

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_return_time", info["episode"]["r"], int(time.time() - start_time))
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(agent.obs_history, agent.action_history, agent.reward_history).reshape(1, -1)
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
        b_action_seqs = action_seqs.reshape(-1, agent.context_len, action_dim)
        b_reward_seqs = reward_seqs.reshape(-1, agent.context_len, 1)
        b_next_obs_seqs = next_obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_next_action_seqs = next_action_seqs.reshape(-1, agent.context_len, action_dim)
        b_next_reward_seqs = next_reward_seqs.reshape(-1, agent.context_len, 1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)

        # Mask out transitions whose returned observation is already from the next episode.
        wm_valid = 1.0 - next_dones
        b_wm_valid = wm_valid.reshape(-1)

        # PPO update with world model + state pred loss
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                newlogprob, entropy, newvalue, dynamics_latent, state_pred_latent, state_tokens = agent.get_all_for_update(
                    b_obs_seqs[mb_inds],
                    b_action_seqs[mb_inds],
                    b_reward_seqs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # World model loss on next encoded state tokens
                with torch.no_grad():
                    next_state_tokens_target = agent.encode_state(b_next_obs_seqs[mb_inds])[:, -1]
                    _, _, next_dynamics, _, _ = agent._encode(
                        b_next_obs_seqs[mb_inds], b_next_action_seqs[mb_inds], b_next_reward_seqs[mb_inds]
                    )
                mb_state_pred = agent.predict_next_state_tokens(state_tokens)
                reward_latent, continue_latent = agent.readout_next_state(mb_state_pred)
                mb_r_pred = agent.reward_head(reward_latent).squeeze(-1)
                mb_c_pred = agent.continue_head(continue_latent).squeeze(-1)
                mb_valid = b_wm_valid[mb_inds]

                transition_loss = (
                    ((mb_state_pred - next_state_tokens_target) ** 2).mean(dim=(-1, -2)) * mb_valid
                ).sum() / (mb_valid.sum() + 1e-8)
                reward_loss = ((mb_r_pred - b_rewards[mb_inds]) ** 2 * mb_valid).sum() / (mb_valid.sum() + 1e-8)
                continue_target = mb_valid
                continue_loss = F.binary_cross_entropy_with_logits(mb_c_pred, continue_target, reduction="mean")
                wm_loss = transition_loss + reward_loss + continue_loss

                # State prediction loss: Gaussian NLL on next dynamics latent
                pred = agent.to_state_pred(state_pred_latent)
                pred_mean, pred_log_var = pred[:, :EMBED_DIM], pred[:, EMBED_DIM:]
                pred_var = pred_log_var.exp()
                sp_loss_raw = F.gaussian_nll_loss(pred_mean, next_dynamics, var=pred_var, reduction='none')
                state_pred_loss = (sp_loss_raw.mean(-1) * mb_valid).sum() / (mb_valid.sum() + 1e-8)

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + WM_COEF * wm_loss + STATE_PRED_LOSS_COEF * state_pred_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Imagination phase
        n_seeds = min(N_IMAGINE_SEEDS, args.batch_size)
        seed_inds = torch.randperm(args.batch_size, device=device)[:n_seeds]
        with torch.no_grad():
            state_token_seeds = agent.encode_state(b_obs_seqs[seed_inds]).detach()

        lambda_returns, imagined_values = agent.imagine(
            state_token_seeds,
            b_action_seqs[seed_inds].detach(),
            b_reward_seqs[seed_inds].detach(),
            IMAGINATION_HORIZON,
            args.gamma,
            args.gae_lambda,
        )

        imagine_actor_loss = -torch.stack(lambda_returns).mean()

        imagine_critic_loss = 0.0
        for t in range(IMAGINATION_HORIZON):
            imagine_critic_loss = imagine_critic_loss + ((imagined_values[t] - lambda_returns[t].detach()) ** 2).mean()
        imagine_critic_loss = imagine_critic_loss / IMAGINATION_HORIZON

        imagine_loss = IMAGINE_COEF * imagine_actor_loss + IMAGINE_CRITIC_COEF * imagine_critic_loss

        optimizer.zero_grad()
        imagine_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("worldmodel/transition_loss", transition_loss.item(), global_step)
        writer.add_scalar("worldmodel/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("worldmodel/continue_loss", continue_loss.item(), global_step)
        writer.add_scalar("worldmodel/total_loss", wm_loss.item(), global_step)
        writer.add_scalar("state_pred/loss", state_pred_loss.item(), global_step)
        writer.add_scalar("imagination/actor_loss", imagine_actor_loss.item(), global_step)
        writer.add_scalar("imagination/critic_loss", imagine_critic_loss.item() if isinstance(imagine_critic_loss, torch.Tensor) else imagine_critic_loss, global_step)
        with torch.no_grad():
            writer.add_scalar("imagination/mean_return", torch.stack(lambda_returns).mean().item(), global_step)
            actor_latent, _, _, _, _ = agent._encode(agent.obs_history, agent.action_history, agent.reward_history)
            _, _, _, action_std = agent._get_policy_dist(actor_latent)
            writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
            _, _, _, sp_latent, _ = agent._encode(agent.obs_history, agent.action_history, agent.reward_history)
            sp_pred = agent.to_state_pred(sp_latent)
            sp_log_var = sp_pred[:, EMBED_DIM:]
            writer.add_scalar("state_pred/log_var_mean", sp_log_var.mean().item(), global_step)
            writer.add_scalar("state_pred/entropy_bonus_mean", (sp_log_var.sum(-1) * STATE_ENT_COEF).mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
