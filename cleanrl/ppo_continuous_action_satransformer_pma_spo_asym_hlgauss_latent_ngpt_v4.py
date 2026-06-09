# nGPT hypersphere satransformer v4: v3 aligned to iterthink_v24_beta action
# distribution + persistent per-observation identity tokens.
#
# Diff vs previous v4 draft: the capped mode/nu Beta is removed. The default
# actor is exactly the iterthink_v24_beta unimodal Beta:
#     alpha = 1 + softplus(alpha_head); beta = 1 + softplus(beta_head)
# with native z in (0,1), linear action rescale, z-replay during PPO updates,
# and no action-rescale Jacobian in the PPO ratio. This keeps the action path
# aligned with the iterthink_v24_beta lineage rather than introducing a new
# exploration-floor parameterization.
#
# Diff vs latent_ngpt_v3: RoPE stays removed, but observation tokens are no
# longer effectively identity-free near x=0. Each scalar observation is encoded
# as justnorm(obs_id[i] + x_i * obs_value[i]); both row embeddings are projected
# to the hypersphere after optimizer steps. This preserves fixed body/muscle
# identity without imposing permutation invariance, while still letting the
# scalar value move the token direction.
#
# Diff vs latent_ngpt_v1 (shared with v2/v3):
#   - 4 nGPT SA blocks (was 1) + 1 nGPT PMA readout block. nGPT's central
#     claim is that hypersphere residuals (bounded spherical interpolation
#     steps) make depth trainable without warmup/norm tricks — v1's 1-block
#     backbone couldn't test that; this does.
#   - Per-action PMA actor seeds + one global actor seed + one critic seed.
#     Each actuator gets its own readout query, while a small global actor
#     residual keeps whole-body coordination available. The separate sde seed
#     is gone with the sde head.
#   - Action distribution ported from iterthink_v24_beta_d4hlgauss (dreamer4
#     faithful, --actor-dist toggle):
#       beta (DEFAULT): two concentration heads, alpha/beta=1+softplus(head);
#         native support (0,1) rescaled linearly to the action range; the
#         buffer stores the NATIVE z and replays it; the constant rescale
#         Jacobian drops out of the PPO ratio. Bounded support => no squash
#         saturation, no 1/sigma^2 blow-up, no bang-bang.
#       gaussian (control): mean head + state-dependent log-VARIANCE head,
#         soft tanh-rescale bound to [logvar_min, logvar_max] (no dead
#         gradient at the bound), std = exp(lv/2), tanh-squash + stable
#         Jacobian, z-replay.
#   - No entropy coefficient: the loss is pg_loss + vf_coef * v_loss.
#     Entropy is logged as a diagnostic only.
#
# nGPT mechanics (unchanged from v1): justnorm everywhere instead of RMSNorm,
# eigen-LR spherical residuals, sqk on normalized q/k with sqrt(head_dim)
# softmax scale, SwiGLU with s_u/s_v sqrt(D) scalings, post-step weight
# re-projection onto the hypersphere (rows for input projections and token
# embeddings, cols for output projections), plain Adam (no weight decay).
# Heads stay unconstrained; token embeddings and seeds are projected. Seed
# features are scaled by sqrt(D) before the heads (RMS-1 head-input scale,
# stand-in for nGPT's s_z).
# Known deviation from the nGPT recipe (audited): Adam betas stay at the PPO
# convention (0.9, 0.999) rather than nGPT's (0.9, 0.95).
#
# Hypothesis. v1 @2M trailed the non-nGPT baseline ~40% with a matched
# 1M->2M slope, and 3.3x LR recovered little of it, so the deficit is
# structural, not step-size. The two structural suspects this version
# addresses: (a) one SA block is too shallow for the eigen-LR residual
# scheme to pay off — nGPT's gains come from depth it can finally afford;
# (b) the tanh-squashed Gaussian's 1/sigma^2 mean-gradient pathology, which
# the iterthink line already showed the unimodal Beta fixes.

import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.tensorboard import SummaryWriter

# Make the cleanrl package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.hl_gauss import HLGaussSupport


MODEL_DIM = 64
NUM_HEADS = 4
FFN_MULT = 2
NUM_SA_BLOCKS = 4
NUM_LATENTS = 3  # learnable scratch tokens that participate in SA (no readout role)
NUM_GLOBAL_PMA_SEEDS = 2  # readout queries after per-action actor seeds: global actor, critic
OBS_VALUE_SCALE = 0.5
GLOBAL_ACTOR_MIX = 0.25

# nGPT scale parameterization: param stored at init_scaling, used at
# param * (init_value / init_scaling) — keeps all learnable scales at a
# magnitude Adam handles well while their effective values differ.
BASE_SCALE = MODEL_DIM ** -0.5
ALPHA_INIT_VALUE = 0.05  # effective eigen-LR at init
ALPHA_INIT_SCALING = BASE_SCALE
SQK_INIT_VALUE = 1.0
SQK_INIT_SCALING = BASE_SCALE
SUV_INIT_VALUE = 1.0
SUV_INIT_SCALING = 1.0

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))
LOG_2 = float(np.log(2.0))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantage normalization (PPO-style minibatch standardization)."""
    spo_eps_low: float = 0.40
    """SPO penalty bound when drift opposes advantage sign (constraining)"""
    spo_eps_high: float = 0.56
    """SPO penalty bound when drift agrees with advantage sign (permissive)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # Action distribution (ported from iterthink_v24_beta)
    actor_dist: str = "beta"
    """action distribution: "beta" (dreamer4 unimodal Beta) or "gaussian" (log-variance head)"""
    logvar_min: float = -8.0
    """gaussian: soft log-variance lower bound (symmetric bounds => std=1 at init)"""
    logvar_max: float = 8.0
    """gaussian: soft log-variance upper bound"""

    # HL-Gauss critic
    num_bins: int = 51
    """number of bins in the categorical value head"""
    v_min: float = -5.0
    """min of the value support (in symlog space when use_symlog=True)"""
    v_max: float = 5.0
    """max of the value support (in symlog space when use_symlog=True)"""
    sigma_ratio: float = 0.5
    """HL-Gauss sigma as a fraction of bin width"""
    use_symlog: bool = True
    """apply symlog/symexp around the categorical support (DreamerV3-style)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def rescale(x, from_range, to_range):
    lo1, hi1 = from_range
    lo2, hi2 = to_range
    return (x - lo1) / (hi1 - lo1) * (hi2 - lo2) + lo2


def justnorm(x, dim=-1):
    """L2-normalize onto the unit hypersphere. Reduction in fp32 (safe under
    bf16 autocast), cast back to input dtype. eps guards any accidental
    all-zero vector before projection."""
    dtype = x.dtype
    x = x.float()
    return (x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-12)).to(dtype)


def flash_attention(q, k, v, scale):
    """Flash SDPA on CUDA with low-precision Q/K/V, matching nGPT's explicit
    bf16 flash-attn call. nGPT passes scale=sqrt(head_dim): q·k is a bounded
    cosine after justnorm, so it needs amplification, not damping."""
    if q.is_cuda:
        attn_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q.to(dtype=attn_dtype),
                        k.to(dtype=attn_dtype),
                        v.to(dtype=attn_dtype),
                        dropout_p=0.0,
                        scale=scale,
                    )
                return out.to(dtype=q.dtype)
            except RuntimeError as err:
                raise RuntimeError("FlashAttention SDPA failed on CUDA; not falling back silently.") from err
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)


def init_linear_(module, init_scale=1.0):
    fan_in = module.weight.shape[1]
    std = init_scale * fan_in ** -0.5
    nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class NGPTSABlock(nn.Module):
    """nGPT self-attention block over unit-norm tokens. NoPE: token identity
    comes from per-observation identity embeddings, not positions. Per-head
    justnorm + sqk scaling on Q/K, sqrt(head_dim) softmax scale, SwiGLU with
    s_u/s_v scalings, spherical-interpolation residuals with learnable
    per-dim eigen learning rates. No pre-norms, no biases."""

    def __init__(self, dim, num_heads, ffn_mult):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_scale = self.head_dim ** 0.5

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)  # silu (v) branch
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)  # linear (u) branch
        self.suv_factor = (SUV_INIT_VALUE / SUV_INIT_SCALING) * dim ** 0.5

        self.attn_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.mlp_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.sqk = nn.Parameter(SQK_INIT_SCALING * torch.ones(dim))
        self.s_u = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))
        self.s_v = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))

        for module in (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2, self.w3):
            init_linear_(module, init_scale=1.0)

    def forward(self, x):
        B, S, D = x.shape

        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        sqk = (self.sqk * (SQK_INIT_VALUE / SQK_INIT_SCALING)).view(1, self.num_heads, 1, self.head_dim)
        q = sqk * justnorm(q)
        k = sqk * justnorm(k)

        attn_out = flash_attention(q, k, v, self.softmax_scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        h_att = self.wo(attn_out)

        lr = (self.attn_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs()
        a_norm = justnorm(x)
        x = justnorm(a_norm + lr * (justnorm(h_att) - a_norm))

        u = (self.s_u * self.suv_factor) * self.w3(x)
        sv = (self.s_v * self.suv_factor) * self.w1(x)
        h_mlp = self.w2(u * F.silu(sv))

        lr = (self.mlp_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs()
        a_norm = justnorm(x)
        x = justnorm(a_norm + lr * (justnorm(h_mlp) - a_norm))
        return x


class NGPTPMABlock(nn.Module):
    """nGPT PMA block: unit-norm seeds (Q) cross-attend unit-norm tokens
    (K, V). NoPE throughout (seed role comes from learned query identity;
    observation-token identity comes from per-observation embeddings).
    Same justnorm Q/K + sqk, sqrt(head_dim) scale, SwiGLU s_u/s_v, and
    eigen-LR spherical residuals on the seed stream as the SA block."""

    def __init__(self, dim, num_heads, ffn_mult):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_scale = self.head_dim ** 0.5

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)  # silu (v) branch
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)  # linear (u) branch
        self.suv_factor = (SUV_INIT_VALUE / SUV_INIT_SCALING) * dim ** 0.5

        self.attn_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.mlp_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.sqk = nn.Parameter(SQK_INIT_SCALING * torch.ones(dim))
        self.s_u = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))
        self.s_v = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))

        for module in (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2, self.w3):
            init_linear_(module, init_scale=1.0)

    def forward(self, seeds, kv_input):
        B, S_q, D = seeds.shape
        S_kv = kv_input.shape[1]

        q = self.wq(seeds).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(kv_input).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv_input).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        sqk = (self.sqk * (SQK_INIT_VALUE / SQK_INIT_SCALING)).view(1, self.num_heads, 1, self.head_dim)
        q = sqk * justnorm(q)
        k = sqk * justnorm(k)

        attn_out = flash_attention(q, k, v, self.softmax_scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, D)
        h_att = self.wo(attn_out)

        lr = (self.attn_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs()
        a_norm = justnorm(seeds)
        seeds = justnorm(a_norm + lr * (justnorm(h_att) - a_norm))

        u = (self.s_u * self.suv_factor) * self.w3(seeds)
        sv = (self.s_v * self.suv_factor) * self.w1(seeds)
        h_mlp = self.w2(u * F.silu(sv))

        lr = (self.mlp_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs()
        a_norm = justnorm(seeds)
        seeds = justnorm(a_norm + lr * (justnorm(h_mlp) - a_norm))
        return seeds


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.action_dim = int(np.prod(envs.single_action_space.shape))
        self.num_bins = args.num_bins
        self.actor_dist = args.actor_dist

        # Tokenizer: each obs dim is a fixed body/muscle channel, not an
        # exchangeable set member. A persistent identity direction survives
        # x_i ~= 0, while the scalar value moves that token along its own
        # learned value direction.
        embed_std = MODEL_DIM ** -0.5
        self.obs_id_embed = nn.Parameter(torch.empty(self.obs_dim, MODEL_DIM))
        self.obs_value_embed = nn.Parameter(torch.empty(self.obs_dim, MODEL_DIM))
        nn.init.trunc_normal_(self.obs_id_embed, std=embed_std, a=-2 * embed_std, b=2 * embed_std)
        nn.init.trunc_normal_(self.obs_value_embed, std=embed_std, a=-2 * embed_std, b=2 * embed_std)

        # SA latent tokens: positionless learnable scratch slots that
        # participate in self-attention with the obs tokens. No readout role.
        latent_std = MODEL_DIM ** -0.5
        self.latent_tokens = nn.Parameter(torch.randn(NUM_LATENTS, MODEL_DIM) * latent_std)

        # PMA seed queries: action-local actor readouts plus global actor and
        # critic readouts. These DO NOT appear inside SA — they only act as PMA Q.
        seed_std = MODEL_DIM ** -0.5
        self.action_seeds = nn.Parameter(torch.randn(self.action_dim, MODEL_DIM) * seed_std)
        self.global_actor_seed = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)
        self.critic_seed = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)

        self.sa_blocks = nn.ModuleList(
            [NGPTSABlock(MODEL_DIM, NUM_HEADS, FFN_MULT) for _ in range(NUM_SA_BLOCKS)]
        )
        self.pma_block = NGPTPMABlock(MODEL_DIM, NUM_HEADS, FFN_MULT)

        # Linear-only heads (no MLP), unconstrained fp32. Actor heads are
        # applied per action-local PMA feature.
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(MODEL_DIM, 1), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(MODEL_DIM, 1), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # iterthink_v24_beta / dreamer4 unimodal Beta: two concentration
            # heads, alpha,beta = 1 + softplus(head).
            self.actor_alpha_head = layer_init(nn.Linear(MODEL_DIM, 1), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(MODEL_DIM, 1), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        self.critic_head = layer_init(nn.Linear(MODEL_DIM, args.num_bins), std=1.0)

        # NoPE: no positional encoding anywhere. Obs identity comes from the
        # per-dim body/muscle embeddings; latents and seeds are positionless.

        # Start on the hypersphere (nGPT normalizes at init too).
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self):
        """Project backbone weights back onto the unit hypersphere. Called at
        init and after every optimizer.step() (nGPT's normalize_matrices).
        Input projections per row (each output neuron's input weights),
        output projections per column (each input channel's contribution),
        embeddings per row. Heads stay unconstrained."""
        for block in (*self.sa_blocks, self.pma_block):
            for module in (block.wq, block.wk, block.wv, block.w1, block.w3):
                module.weight.data.copy_(justnorm(module.weight.data, dim=1))
            for module in (block.wo, block.w2):
                module.weight.data.copy_(justnorm(module.weight.data, dim=0))
        self.obs_id_embed.data.copy_(justnorm(self.obs_id_embed.data, dim=-1))
        self.obs_value_embed.data.copy_(justnorm(self.obs_value_embed.data, dim=-1))
        self.latent_tokens.data.copy_(justnorm(self.latent_tokens.data, dim=-1))
        self.action_seeds.data.copy_(justnorm(self.action_seeds.data, dim=-1))
        self.global_actor_seed.data.copy_(justnorm(self.global_actor_seed.data, dim=-1))
        self.critic_seed.data.copy_(justnorm(self.critic_seed.data, dim=-1))

    def _encode(self, x):
        """Tokenize → cat[latent, obs] → justnorm → 4x SA → PMA(action,
        global-actor, critic seeds over full SA output K/V) → sqrt(D) head
        scale. Returns actor features (B, action_dim, D) and critic feature
        (B, D)."""
        B = x.shape[0]
        # Tokenize: (B, obs_dim) → (B, obs_dim, MODEL_DIM) via per-dim
        # identity + scalar value displacement.
        obs_tokens = (
            self.obs_id_embed.unsqueeze(0)
            + OBS_VALUE_SCALE * x.unsqueeze(-1) * self.obs_value_embed.unsqueeze(0)
        )

        # Concat SA latent scratch tokens upfront (positionless).
        latent = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, NUM_LATENTS, D)
        sa_tokens = torch.cat([latent, obs_tokens], dim=1)  # (B, NUM_LATENTS+obs_dim, D)
        sa_tokens = justnorm(sa_tokens)
        for sa_block in self.sa_blocks:
            sa_tokens = sa_block(sa_tokens)

        # PMA: action-local and global seed queries cross-attend the FULL SA output.
        seed_bank = torch.cat([self.action_seeds, self.global_actor_seed, self.critic_seed], dim=0)
        seeds = seed_bank.unsqueeze(0).expand(B, -1, -1)  # (B, action_dim+2, D)
        seeds = self.pma_block(seeds, sa_tokens)
        action_feats = seeds[:, : self.action_dim]
        global_actor_feat = seeds[:, self.action_dim]
        critic_feat = seeds[:, self.action_dim + 1]
        # Mix global coordination before the head-input scale so actor features
        # stay on the same hypersphere as the critic readout.
        actor_feats = justnorm(action_feats + GLOBAL_ACTOR_MIX * global_actor_feat.unsqueeze(1))
        actor_feats = actor_feats * (MODEL_DIM ** 0.5)
        critic_feat = critic_feat * (MODEL_DIM ** 0.5)
        return actor_feats, critic_feat

    def _encoder_with_autocast(self, x):
        if x.is_cuda and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actor_feat, critic_feat = self._encode(x)
        else:
            actor_feat, critic_feat = self._encode(x)
        return actor_feat.float(), critic_feat.float()

    def _actor_dist(self, actor_feat):
        """Build the action distribution and the native-space transforms
        (iterthink_v24_beta). Returns (dist, to_action, log_det_fn) where:
          to_action(z): map a NATIVE sample z to the env action.
          log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
                         from dist.log_prob(z) (0 where the map is volume-constant)."""
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat).squeeze(-1)
            raw_lv = self.actor_logvar_head(actor_feat).squeeze(-1)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: (2.0 * (LOG_2 - z - F.softplus(-2.0 * z))).sum(1)
            return dist, to_action, log_det_fn
        # beta: iterthink_v24_beta / dreamer4 unimodal parameterization.
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat).squeeze(-1))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat).squeeze(-1))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def get_value_logits(self, x):
        _, critic_feat = self._encoder_with_autocast(x)
        return self.critic_head(critic_feat)

    def get_value(self, x, hl_support):
        return hl_support.to_scalar(self.get_value_logits(x))

    def get_action_distribution(self, x):
        actor_feat, _ = self._encoder_with_autocast(x)
        dist, _, _ = self._actor_dist(actor_feat)
        return dist

    def get_action_and_value(self, x, z=None):
        """z is the distribution-NATIVE sample (pre-tanh for gaussian; in
        (0,1) for beta). When replaying from the buffer it is passed back in;
        log_prob is recomputed at the same native sample (z-replay)."""
        actor_feat, critic_feat = self._encoder_with_autocast(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        value_logits = self.critic_head(critic_feat)  # (B, num_bins)

        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

        env_action = to_action(z)
        log_prob = dist.log_prob(z).sum(1) - log_det_fn(z)
        entropy = dist.entropy().sum(1)  # diagnostic only (no entropy term in the loss)
        return env_action, z, log_prob, entropy, value_logits


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, args):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs, args).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            env_action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
        next_obs, _, _, _, infos = envs.step(env_action.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs.close()
    return episodic_returns


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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    # Adam (no weight decay) — nGPT runs without weight decay; the post-step
    # hypersphere projection is the regularizer.
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=args.use_symlog,
    )

    # ALGO Logic: Storage setup
    # `actions` stores the distribution-NATIVE sample z (pre-tanh for
    # gaussian, (0,1) for beta); the env action is recomputed via to_action.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                env_action, z, logprob, _, value_logits = agent.get_action_and_value(next_obs)
                values[step] = hl_support.to_scalar(value_logits)
            actions[step] = z  # store native sample (used for replay)
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(env_action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, hl_support).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        spo_penalty_mean = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # b_actions[mb_inds] is the stored native sample; pass as `z`.
                _, _, newlogprob, entropy, newvalue_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # KL(old||new) approximations http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # SPO with asymmetric ε. Per-sample bound is ε_high when
                # drift direction agrees with advantage sign, ε_low otherwise.
                ratio_diff = ratio - 1.0
                with_adv = (mb_advantages * ratio_diff) > 0
                eps = torch.where(
                    with_adv,
                    torch.full_like(mb_advantages, args.spo_eps_high),
                    torch.full_like(mb_advantages, args.spo_eps_low),
                )
                pg_surrogate = mb_advantages * ratio
                spo_penalty = mb_advantages.abs() * ratio_diff.pow(2) / (2.0 * eps)
                pg_loss = -(pg_surrogate - spo_penalty).mean()
                spo_penalty_mean = spo_penalty.detach().mean()

                # Value loss: HL-Gauss cross-entropy on projected returns.
                # No vclip — categorical doesn't have a clean ratio analogue.
                target_probs = hl_support.project(b_returns[mb_inds])
                log_probs_v = torch.log_softmax(newvalue_logits, dim=-1)
                v_loss = -(target_probs * log_probs_v).sum(dim=-1).mean()

                entropy_loss = entropy.mean()  # logged only; no entropy term in the loss
                loss = pg_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                # nGPT: project backbone weights back onto the hypersphere
                # after every optimizer step.
                agent.normalize_weights()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Distribution diagnostics.
        with torch.no_grad():
            dist_diag = agent.get_action_distribution(b_obs)
            diag_entropy = dist_diag.entropy().sum(1).mean().item()
            if args.actor_dist == "beta":
                diag_conc_a = dist_diag.concentration1.mean().item()
                diag_conc_b = dist_diag.concentration0.mean().item()
            else:
                diag_conc_a = dist_diag.mean.abs().mean().item()
                diag_conc_b = dist_diag.stddev.mean().item()
            # nGPT eigen-LR diagnostics: effective residual step size per stage.
            sa_attn_lr = np.mean(
                [(b.attn_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs().mean().item() for b in agent.sa_blocks]
            )
            sa_mlp_lr = np.mean(
                [(b.mlp_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs().mean().item() for b in agent.sa_blocks]
            )
            pma_attn_lr = (agent.pma_block.attn_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs().mean().item()
            pma_mlp_lr = (agent.pma_block.mlp_alpha * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING)).abs().mean().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("diag/spo_penalty", spo_penalty_mean.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/dist_entropy", diag_entropy, global_step)
        writer.add_scalar("diag/conc_alpha_or_absmean", diag_conc_a, global_step)
        writer.add_scalar("diag/conc_beta_or_std", diag_conc_b, global_step)
        writer.add_scalar("diag/ngpt_sa_attn_lr", sa_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_sa_mlp_lr", sa_mlp_lr, global_step)
        writer.add_scalar("diag/ngpt_pma_attn_lr", pma_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_pma_mlp_lr", pma_mlp_lr, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        episodic_returns = evaluate_policy(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            model=Agent,
            device=device,
            gamma=args.gamma,
            args=args,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
