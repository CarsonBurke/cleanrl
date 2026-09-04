# nGPT hypersphere satransformer v8-hlcdf: v7 decoupled role-token
# encoder/decoders with critic-CDF policy advantages.
#
# Diff vs v6: actor/critic readout tokens no longer enter the shared SA stream
# from layer 0. The shared encoder first processes only obs tokens plus learned
# latent workspace tokens. Separate actor and critic cross-attention decoders
# then read that memory without writing back into it. Actor action tokens get a
# final action-only refinement block with the actor global/context tokens, so
# action coordination remains learnable without body topology, masks, RoPE, or
# permutation invariance.
#
# Diff vs v7: replace batch rankgauss advantages with an uncertainty-aware
# transform from the HL-Gauss critic distribution. For each rollout sample,
# u = P_critic(V(s) <= lambda_return), then the policy advantage is
# tanh(probit(u) / kappa). This removes empirical batch ranking, preserves the
# critic's distributional uncertainty, and compresses extreme update pressure.
# No target-KL stopping. HL-Gauss remains Dreamer4-style edge-bin support with
# sigma_to_bin_ratio=2 and our reward-normalized raw range [-10, 10].
#
# Diff vs v4/v5: keep the best role-token architecture, but replace
#     justnorm(obs_id[i] + 0.5 * x_i * obs_value[i])
# with a factorized token:
#     justnorm(channel_id[i] + obs_type
#              + shared_value_encoder(phi(x_i))
#              + channel_value_encoder_i(phi(x_i)))
# where phi is a bounded, non-periodic scalar basis over the clipped normalized
# observation value. This gives attention/FFNs stable identity, token-role
# type, transferable numeric features, and per-channel numeric semantics,
# without RoPE, body graph masks, hand-coded topology, or permutation
# invariance. Role tokens are likewise factored into learned ID + role type.
#
# Diff vs previous v4 draft: the capped mode/nu Beta is removed. The default
# actor is exactly the iterthink_v24_beta unimodal Beta:
#     alpha = 1 + softplus(alpha_head); beta = 1 + softplus(beta_head)
# with native z in (0,1), linear action rescale, z-replay during PPO updates,
# and no action-rescale Jacobian in the PPO ratio. This keeps the action path
# aligned with the iterthink_v24_beta lineage rather than introducing a new
# exploration-floor parameterization.
#
# Diff vs latent_ngpt_v3: RoPE stays removed. Fixed body/muscle identity is
# represented by learned channel IDs plus obs token-type embeddings, while
# scalar values use shared and per-channel bounded value encoders. This avoids
# the v4 failure mode where large |x| rotates a token almost entirely away from
# its stable identity direction.
#
# Diff vs latent_ngpt_v1 (shared with v2/v3):
#   - 4 nGPT SA blocks (was 1). Actor/critic readouts are now learned role
#     tokens inserted into the SA stream from layer 0, then picked from the
#     final sequence. nGPT's central
#     claim is that hypersphere residuals (bounded spherical interpolation
#     steps) make depth trainable without warmup/norm tricks — v1's 1-block
#     backbone couldn't test that; this does.
#   - Per-action actor role tokens + one global actor token + one critic token.
#     Each actuator gets its own final token feature, while a small global
#     actor residual keeps whole-body coordination available. The separate sde
#     seed is gone with the sde head. For gaussian control, learned log-variance
#     is projected from the same action role token; for beta, concentration
#     heads learn bounded-support variance through alpha/beta.
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
# Heads stay unconstrained; token embeddings and role tokens are projected. Role
# features are scaled by sqrt(D) before the heads (RMS-1 head-input scale,
# stand-in for nGPT's s_z).
# Optimizer defaults stay at the PPO convention, but Adam betas are CLI
# configurable so the hotter ../ngpt setting (lr=15e-4, beta2=0.95) can be
# tested as a clean follow-up run without changing architecture.
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
from cleanrl.shared.hl_gauss import HLGaussSupport, symlog


MODEL_DIM = 64
NUM_HEADS = 4
FFN_MULT = 2
NUM_ENCODER_BLOCKS = 4
NUM_DECODER_BLOCKS = 2
NUM_ACTION_REFINEMENT_BLOCKS = 1
NUM_LATENTS = 3  # learnable scratch tokens that participate in SA (no readout role)
OBS_VALUE_FEATURE_DIM = 6
ACTOR_READOUT_TOKENS = 3  # per-action role + two global actor role tokens
CRITIC_READOUT_TOKENS = 2
ACTOR_HEAD_DIM = MODEL_DIM * ACTOR_READOUT_TOKENS
CRITIC_HEAD_DIM = MODEL_DIM * CRITIC_READOUT_TOKENS

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
SYMLOG_10 = float(np.log(11.0))


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
    adam_beta1: float = 0.9
    """Adam beta1"""
    adam_beta2: float = 0.999
    """Adam beta2"""
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
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles post-shaping z-score advantage normalization."""
    adv_rms_norm: bool = True
    """sign-preserving post-shaping RMS scaling for policy advantages"""
    adv_transform: str = "hlcdf_tanh"
    """policy advantage shaping: 'hlcdf_tanh', 'hlcdf_probit', 'rankgauss', or 'v10'"""
    cdf_probit_clamp: float = 0.995
    """CDF/probit clamp before erfinv"""
    hlcdf_tanh_kappa: float = 1.5
    """temperature for tanh(probit(critic_cdf) / kappa)"""
    clip_coef: float = 0.2
    """PPO lower clip coefficient"""
    clip_coef_high: float = 0.28
    """PPO upper clip coefficient (clip-higher)"""
    norm_adv_scope: str = "minibatch"
    """advantage normalization scope: 'minibatch' or 'batch'"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for global gradient clipping when separate_grad_clip=False"""
    separate_grad_clip: bool = True
    """clip policy and value gradients separately before summing shared-trunk gradients"""
    actor_grad_clip: float = 0.25
    """max norm for policy gradient clipping when separate_grad_clip=True"""
    critic_grad_clip: float = 0.25
    """max norm for value gradient clipping when separate_grad_clip=True"""

    # Action distribution (ported from iterthink_v24_beta)
    actor_dist: str = "beta"
    """action distribution: "beta" (dreamer4 unimodal Beta) or "gaussian" (log-variance head)"""
    logvar_min: float = -8.0
    """gaussian: soft log-variance lower bound (symmetric bounds => std=1 at init)"""
    logvar_max: float = 8.0
    """gaussian: soft log-variance upper bound"""

    # HL-Gauss critic
    num_bins: int = 511
    """number of bins in the categorical value head"""
    v_min: float = -10.0
    """min raw scalar value covered by the critic support"""
    v_max: float = 10.0
    """max raw scalar value covered by the critic support"""
    sigma_ratio: float = 2.0
    """HL-Gauss sigma as a fraction of bin width"""
    use_symlog: bool = True
    """use symlog support coordinates for raw scalar value bounds"""
    critic_init_tau: float = 0.5
    """initial categorical critic width in support coordinates"""

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


def scalar_value_features(x):
    """Bounded, non-periodic scalar basis for normalized MuJoCo observations.

    These are value features, not token-position features: no RoPE or sequence
    index signal is introduced. Every component is bounded to keep value
    information from erasing channel identity before the transformer can learn
    how much to use it.
    """
    z = x.clamp(-10.0, 10.0)
    mag = (z.abs() / 10.0).clamp(0.0, 1.0)
    signed_sqrt = z.sign() * mag.sqrt()
    signed_square = z.sign() * mag.square()
    return torch.stack(
        (
            z / 10.0,
            torch.tanh(z),
            symlog(z) / SYMLOG_10,
            signed_sqrt,
            signed_square,
            mag,
        ),
        dim=-1,
    )


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


def value_support_bounds(args):
    """Support bounds in the coordinate used for categorical bins.

    Args v_min/v_max are raw scalar values so `[-10, 10]` means the linear
    critic range the user sees. With symlog enabled, the categorical bins live
    in symlog coordinates over those raw bounds, matching the iterthink d4
    HL-Gauss lineage.
    """
    if not args.use_symlog:
        return args.v_min, args.v_max
    bounds = torch.tensor([args.v_min, args.v_max], dtype=torch.float32)
    return symlog(bounds).tolist()


@torch.no_grad()
def init_hl_gauss_critic_head_(head, args):
    support_min, support_max = value_support_bounds(args)
    edge_width = (support_max - support_min) / args.num_bins
    z = torch.linspace(
        support_min + 0.5 * edge_width,
        support_max - 0.5 * edge_width,
        args.num_bins,
        dtype=head.bias.dtype,
        device=head.bias.device,
    )
    head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)


def shape_advantage(gae, args, device):
    if args.adv_transform == "v10":
        return gae
    if args.adv_transform != "rankgauss":
        raise ValueError(f"unknown adv_transform {args.adv_transform}")
    n = gae.numel()
    ranks = gae.argsort().argsort().to(torch.float32)
    uq = (ranks + 0.5) / n
    centered = (2.0 * uq - 1.0).clamp(-args.cdf_probit_clamp, args.cdf_probit_clamp)
    return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)


def hl_gauss_cdf_advantage(logits, returns, hl_support, args):
    """Distribution-aware advantage from the critic CDF at each return target.

    The categorical critic predicts P(V(s) in bin_i). The CDF value u at the
    lambda-return says how surprising-good the target is under the current
    value distribution. This preserves critic uncertainty instead of replacing
    samples with empirical rollout ranks.
    """
    probs = torch.softmax(logits, dim=-1)
    targets = symlog(returns) if hl_support.use_symlog else returns
    targets = targets.clamp(hl_support.v_min, hl_support.v_max)

    edges = hl_support.edges
    if edges is None:
        half_w = hl_support.bin_width / 2.0
        edges = torch.cat(
            [
                hl_support.support[:1] - half_w,
                hl_support.support + half_w,
            ]
        )
    bin_idx = torch.bucketize(targets, edges).sub(1).clamp(0, hl_support.num_bins - 1)
    cdf = torch.cumsum(probs, dim=-1)
    cdf_before = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., :-1]], dim=-1)
    before = cdf_before.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    bin_prob = probs.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    left = edges[bin_idx]
    frac = ((targets - left) / hl_support.bin_width).clamp(0.0, 1.0)
    u = (before + bin_prob * frac).clamp(1e-6, 1.0 - 1e-6)

    centered = (2.0 * u - 1.0).clamp(-args.cdf_probit_clamp, args.cdf_probit_clamp)
    z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(returns.device)
    if args.adv_transform == "hlcdf_probit":
        adv = z
    elif args.adv_transform == "hlcdf_tanh":
        adv = torch.tanh(z / args.hlcdf_tanh_kappa)
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")
    return adv, u, z


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


class NGPTCrossAttentionBlock(nn.Module):
    """nGPT cross-attention block. Query tokens are updated; memory tokens are
    read-only. This preserves the same hypersphere residual, sqk, Flash SDPA,
    and SwiGLU mechanics as NGPTSABlock while separating readout-token credit
    assignment from shared encoder computation."""

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
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.suv_factor = (SUV_INIT_VALUE / SUV_INIT_SCALING) * dim ** 0.5

        self.attn_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.mlp_alpha = nn.Parameter(ALPHA_INIT_SCALING * torch.ones(dim))
        self.sqk = nn.Parameter(SQK_INIT_SCALING * torch.ones(dim))
        self.s_u = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))
        self.s_v = nn.Parameter(SUV_INIT_SCALING * torch.ones(ffn_dim))

        for module in (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2, self.w3):
            init_linear_(module, init_scale=1.0)

    def forward(self, x, memory):
        B, S, D = x.shape
        M = memory.shape[1]

        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

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


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.action_dim = int(np.prod(envs.single_action_space.shape))
        self.num_bins = args.num_bins
        self.actor_dist = args.actor_dist

        # Tokenizer: each obs dim is a fixed body/muscle channel, not an
        # exchangeable set member. Identity, token type, shared scalar
        # semantics, and per-channel scalar semantics are separate learnable
        # factors so the transformer gets stable labels without RoPE/topology.
        embed_std = MODEL_DIM ** -0.5
        self.obs_id_embed = nn.Parameter(torch.empty(self.obs_dim, MODEL_DIM))
        self.obs_channel_value_embed = nn.Parameter(
            torch.empty(self.obs_dim, OBS_VALUE_FEATURE_DIM, MODEL_DIM)
        )
        self.obs_shared_value_encoder = nn.Linear(OBS_VALUE_FEATURE_DIM, MODEL_DIM, bias=False)
        # Branch scales are learned, not fixed: all start equal so v5 does not
        # hard-code how much identity/type/value information should dominate.
        # The final token direction is normalized, so these relative weights
        # control initial geometry and remain free to adapt.
        self.obs_branch_scale = nn.Parameter(torch.ones(4))
        self.obs_type_embed = nn.Parameter(torch.empty(1, MODEL_DIM))
        self.latent_type_embed = nn.Parameter(torch.empty(1, MODEL_DIM))
        self.action_role_type_embed = nn.Parameter(torch.empty(1, MODEL_DIM))
        self.global_actor_type_embed = nn.Parameter(torch.empty(1, MODEL_DIM))
        self.critic_type_embed = nn.Parameter(torch.empty(1, MODEL_DIM))
        nn.init.trunc_normal_(self.obs_id_embed, std=embed_std, a=-2 * embed_std, b=2 * embed_std)
        nn.init.trunc_normal_(
            self.obs_channel_value_embed,
            std=embed_std,
            a=-2 * embed_std,
            b=2 * embed_std,
        )
        for token in (
            self.obs_type_embed,
            self.latent_type_embed,
            self.action_role_type_embed,
            self.global_actor_type_embed,
            self.critic_type_embed,
        ):
            nn.init.trunc_normal_(token, std=embed_std, a=-2 * embed_std, b=2 * embed_std)
        init_linear_(self.obs_shared_value_encoder, init_scale=1.0)

        # SA latent tokens: positionless learnable scratch slots that
        # participate in self-attention with the obs tokens. No readout role.
        latent_std = MODEL_DIM ** -0.5
        self.latent_tokens = nn.Parameter(torch.randn(NUM_LATENTS, MODEL_DIM) * latent_std)

        # Role tokens: action-local actor readouts plus global actor and critic
        # readouts. In v7 these are decoder queries, not shared encoder tokens:
        # they read obs/latent memory through cross-attention and cannot write
        # back into the shared body/muscle representation.
        seed_std = MODEL_DIM ** -0.5
        self.action_role_tokens = nn.Parameter(torch.randn(self.action_dim, MODEL_DIM) * seed_std)
        self.global_actor_token = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)
        self.actor_context_token = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)
        self.critic_token = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)
        self.critic_context_token = nn.Parameter(torch.randn(1, MODEL_DIM) * seed_std)

        self.sa_blocks = nn.ModuleList(
            [NGPTSABlock(MODEL_DIM, NUM_HEADS, FFN_MULT) for _ in range(NUM_ENCODER_BLOCKS)]
        )
        self.actor_cross_blocks = nn.ModuleList(
            [NGPTCrossAttentionBlock(MODEL_DIM, NUM_HEADS, FFN_MULT) for _ in range(NUM_DECODER_BLOCKS)]
        )
        self.critic_cross_blocks = nn.ModuleList(
            [NGPTCrossAttentionBlock(MODEL_DIM, NUM_HEADS, FFN_MULT) for _ in range(NUM_DECODER_BLOCKS)]
        )
        self.actor_refine_blocks = nn.ModuleList(
            [NGPTSABlock(MODEL_DIM, NUM_HEADS, FFN_MULT) for _ in range(NUM_ACTION_REFINEMENT_BLOCKS)]
        )

        # Linear-only heads (no MLP), unconstrained fp32. Actor heads are
        # applied per action-local role-token feature.
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(ACTOR_HEAD_DIM, 1), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(ACTOR_HEAD_DIM, 1), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # iterthink_v24_beta / dreamer4 unimodal Beta: two concentration
            # heads, alpha,beta = 1 + softplus(head).
            self.actor_alpha_head = layer_init(nn.Linear(ACTOR_HEAD_DIM, 1), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(ACTOR_HEAD_DIM, 1), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        self.critic_head = layer_init(nn.Linear(CRITIC_HEAD_DIM, args.num_bins), std=0.1)
        init_hl_gauss_critic_head_(self.critic_head, args)

        # NoPE: no positional encoding anywhere. Obs identity comes from the
        # per-dim body/muscle embeddings; latents and role tokens are
        # positionless.

        # Start on the hypersphere (nGPT normalizes at init too).
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self):
        """Project backbone weights back onto the unit hypersphere. Called at
        init and after every optimizer.step() (nGPT's normalize_matrices).
        Input projections per row (each output neuron's input weights),
        output projections per column (each input channel's contribution),
        embeddings per row. Heads stay unconstrained."""
        ngpt_blocks = (
            list(self.sa_blocks)
            + list(self.actor_cross_blocks)
            + list(self.critic_cross_blocks)
            + list(self.actor_refine_blocks)
        )
        for block in ngpt_blocks:
            for module in (block.wq, block.wk, block.wv, block.w1, block.w3):
                module.weight.data.copy_(justnorm(module.weight.data, dim=1))
            for module in (block.wo, block.w2):
                module.weight.data.copy_(justnorm(module.weight.data, dim=0))
        self.obs_id_embed.data.copy_(justnorm(self.obs_id_embed.data, dim=-1))
        self.obs_channel_value_embed.data.copy_(justnorm(self.obs_channel_value_embed.data, dim=-1))
        # Linear weight is (D, F); columns are the D-dimensional basis
        # directions for scalar features. Column projection makes the shared
        # branch geometry comparable to obs_channel_value_embed[:, f, :].
        self.obs_shared_value_encoder.weight.data.copy_(
            justnorm(self.obs_shared_value_encoder.weight.data, dim=0)
        )
        self.obs_type_embed.data.copy_(justnorm(self.obs_type_embed.data, dim=-1))
        self.latent_type_embed.data.copy_(justnorm(self.latent_type_embed.data, dim=-1))
        self.action_role_type_embed.data.copy_(justnorm(self.action_role_type_embed.data, dim=-1))
        self.global_actor_type_embed.data.copy_(justnorm(self.global_actor_type_embed.data, dim=-1))
        self.critic_type_embed.data.copy_(justnorm(self.critic_type_embed.data, dim=-1))
        self.latent_tokens.data.copy_(justnorm(self.latent_tokens.data, dim=-1))
        self.action_role_tokens.data.copy_(justnorm(self.action_role_tokens.data, dim=-1))
        self.global_actor_token.data.copy_(justnorm(self.global_actor_token.data, dim=-1))
        self.actor_context_token.data.copy_(justnorm(self.actor_context_token.data, dim=-1))
        self.critic_token.data.copy_(justnorm(self.critic_token.data, dim=-1))
        self.critic_context_token.data.copy_(justnorm(self.critic_context_token.data, dim=-1))

    def _encode(self, x):
        """Tokenize obs/latents → shared SA encoder → separate actor/critic
        cross-attention decoders → actor-only action refinement → sqrt(D) head
        scale. Returns actor features (B, action_dim, 3D) and critic feature
        (B, 2D)."""
        B = x.shape[0]
        # Tokenize: (B, obs_dim) → (B, obs_dim, MODEL_DIM) via stable channel
        # identity + token type + shared and channel-local bounded value bases.
        value_features = scalar_value_features(x)
        shared_value = self.obs_shared_value_encoder(value_features)
        channel_value = torch.einsum(
            "bif,ifd->bid",
            value_features,
            self.obs_channel_value_embed,
        )
        obs_tokens = (
            self.obs_branch_scale[0] * self.obs_id_embed.unsqueeze(0)
            + self.obs_branch_scale[1] * self.obs_type_embed.view(1, 1, MODEL_DIM)
            + self.obs_branch_scale[2] * shared_value
            + self.obs_branch_scale[3] * channel_value
        )

        # Shared encoder memory is only obs plus latent workspace. Actor and
        # critic readout tokens are deliberately excluded so they cannot write
        # into the representation they both consume.
        latent = (self.latent_tokens + self.latent_type_embed).unsqueeze(0).expand(B, -1, -1)
        memory = torch.cat([latent, obs_tokens], dim=1)
        memory = justnorm(memory)
        for sa_block in self.sa_blocks:
            memory = sa_block(memory)

        action_roles = self.action_role_tokens + self.action_role_type_embed
        global_actor = self.global_actor_token + self.global_actor_type_embed
        actor_context = self.actor_context_token + self.global_actor_type_embed
        actor_tokens = torch.cat([action_roles, global_actor, actor_context], dim=0)
        actor_tokens = justnorm(actor_tokens).unsqueeze(0).expand(B, -1, -1)
        for block in self.actor_cross_blocks:
            actor_tokens = block(actor_tokens, memory)
        for block in self.actor_refine_blocks:
            actor_tokens = block(actor_tokens)

        critic = self.critic_token + self.critic_type_embed
        critic_context = self.critic_context_token + self.critic_type_embed
        critic_tokens = torch.cat([critic, critic_context], dim=0)
        critic_tokens = justnorm(critic_tokens).unsqueeze(0).expand(B, -1, -1)
        for block in self.critic_cross_blocks:
            critic_tokens = block(critic_tokens, memory)

        action_feats = actor_tokens[:, : self.action_dim]
        global_actor_feat = actor_tokens[:, self.action_dim]
        actor_context_feat = actor_tokens[:, self.action_dim + 1]
        critic_feat = critic_tokens[:, 0]
        critic_context_feat = critic_tokens[:, 1]
        # Concatenate multiple learned role readouts instead of imposing a
        # fixed additive global residual. Each role feature is unit-scale from
        # the nGPT stream; sqrt(D) keeps per-component head input RMS near 1.
        actor_feats = torch.cat(
            [
                action_feats,
                global_actor_feat.unsqueeze(1).expand(-1, self.action_dim, -1),
                actor_context_feat.unsqueeze(1).expand(-1, self.action_dim, -1),
            ],
            dim=-1,
        ) * (MODEL_DIM ** 0.5)
        critic_feat = torch.cat([critic_feat, critic_context_feat], dim=-1) * (MODEL_DIM ** 0.5)
        return actor_feats, critic_feat

    def _encoder_with_autocast(self, x):
        if x.is_cuda and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actor_feat, critic_feat = self._encode(x)
        else:
            actor_feat, critic_feat = self._encode(x)
        return actor_feat.float(), critic_feat.float()

    def shared_parameters(self):
        """Parameters intentionally shared by policy and value gradients."""
        params = [
            self.obs_id_embed,
            self.obs_channel_value_embed,
            self.obs_branch_scale,
            self.obs_type_embed,
            self.latent_type_embed,
            self.latent_tokens,
        ]
        params += list(self.obs_shared_value_encoder.parameters())
        params += list(self.sa_blocks.parameters())
        return params

    def actor_parameters(self):
        params = self.shared_parameters()
        params += [
            self.action_role_tokens,
            self.global_actor_token,
            self.actor_context_token,
            self.action_role_type_embed,
            self.global_actor_type_embed,
        ]
        params += list(self.actor_cross_blocks.parameters())
        params += list(self.actor_refine_blocks.parameters())
        if self.actor_dist == "gaussian":
            params += list(self.actor_head.parameters())
            params += list(self.actor_logvar_head.parameters())
        else:
            params += list(self.actor_alpha_head.parameters())
            params += list(self.actor_beta_head.parameters())
        return params

    def critic_parameters(self):
        params = self.shared_parameters()
        params += [
            self.critic_token,
            self.critic_context_token,
            self.critic_type_embed,
        ]
        params += list(self.critic_cross_blocks.parameters())
        params += list(self.critic_head.parameters())
        return params

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
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-5,
    )
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    support_min, support_max = value_support_bounds(args)
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.sigma_ratio,
        device,
        use_symlog=args.use_symlog,
        support_is_edges=True,
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
        b_hlcdf_u = None
        b_hlcdf_z = None
        if args.adv_transform in ("hlcdf_tanh", "hlcdf_probit"):
            with torch.no_grad():
                value_logits_chunks = []
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    value_logits_chunks.append(agent.get_value_logits(b_obs[start:end]))
                b_value_logits = torch.cat(value_logits_chunks, dim=0)
                b_policy_advantages, b_hlcdf_u, b_hlcdf_z = hl_gauss_cdf_advantage(
                    b_value_logits,
                    b_returns,
                    hl_support,
                    args,
                )
        else:
            b_policy_advantages = shape_advantage(b_advantages, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_advantages = (
                (b_policy_advantages - b_policy_advantages.mean())
                / (b_policy_advantages.std() + 1e-8)
            )
        with torch.no_grad():
            raw_adv_z = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            shaped_adv_z = (b_policy_advantages - b_policy_advantages.mean()) / (
                b_policy_advantages.std() + 1e-8
            )
            adv_corr = (raw_adv_z * shaped_adv_z).mean()
            adv_sign_agree = (
                torch.sign(raw_adv_z) == torch.sign(shaped_adv_z)
            ).float().mean()
            adv_raw_sign_agree = (
                torch.sign(b_advantages) == torch.sign(b_policy_advantages)
            ).float().mean()
            adv_mean_abs = b_policy_advantages.abs().mean()
            adv_saturation = (b_policy_advantages.abs() > 0.95).float().mean()
            if b_hlcdf_u is not None:
                hlcdf_u_mean = b_hlcdf_u.mean()
                hlcdf_u_std = b_hlcdf_u.std()
                hlcdf_edge_frac = ((b_hlcdf_u < 0.01) | (b_hlcdf_u > 0.99)).float().mean()
                hlcdf_z_abs = b_hlcdf_z.abs().mean()
            else:
                hlcdf_u_mean = torch.zeros((), device=device)
                hlcdf_u_std = torch.zeros((), device=device)
                hlcdf_edge_frac = torch.zeros((), device=device)
                hlcdf_z_abs = torch.zeros((), device=device)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        clipfrac_mean = 0.0
        actor_gn = torch.zeros((), device=device)
        critic_gn = torch.zeros((), device=device)
        update_adv_sign_agrees = []
        update_adv_mean_abs = []
        update_adv_saturations = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            clipfracs = []
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
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_policy_advantages[mb_inds]
                if args.adv_rms_norm:
                    mb_advantages = mb_advantages / mb_advantages.square().mean().sqrt().clamp_min(1e-8)
                elif args.norm_adv and args.norm_adv_scope == "minibatch":
                    mb_advantages = (
                        (mb_advantages - mb_advantages.mean())
                        / (mb_advantages.std() + 1e-8)
                    )
                with torch.no_grad():
                    update_adv_sign_agrees.append(
                        (
                            torch.sign(b_advantages[mb_inds]) == torch.sign(mb_advantages)
                        ).float().mean().item()
                    )
                    update_adv_mean_abs.append(mb_advantages.abs().mean().item())
                    update_adv_saturations.append((mb_advantages.abs() > 0.95).float().mean().item())

                clip_hi = args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + clip_hi,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: HL-Gauss cross-entropy on projected returns.
                # No vclip — categorical doesn't have a clean ratio analogue.
                target_probs = hl_support.project(b_returns[mb_inds])
                log_probs_v = torch.log_softmax(newvalue_logits, dim=-1)
                v_loss = -(target_probs * log_probs_v).sum(dim=-1).mean()

                entropy_loss = entropy.mean()  # logged only; no entropy term in the loss

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    critic_grads = [
                        (p, p.grad.detach().clone())
                        for p in critic_params
                        if p.grad is not None
                    ]

                    optimizer.zero_grad(set_to_none=True)
                    pg_loss.backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, g in critic_grads:
                        p.grad = g if p.grad is None else p.grad + g
                else:
                    loss = pg_loss + v_loss * args.vf_coef
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_gn = critic_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                # nGPT: project backbone weights back onto the hypersphere
                # after every optimizer step.
                agent.normalize_weights()
            clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else 0.0
        update_adv_sign_agree = float(np.mean(update_adv_sign_agrees)) if update_adv_sign_agrees else 0.0
        update_adv_mean_abs = float(np.mean(update_adv_mean_abs)) if update_adv_mean_abs else 0.0
        update_adv_saturation = float(np.mean(update_adv_saturations)) if update_adv_saturations else 0.0

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
            def block_lr(blocks, attr):
                return np.mean(
                    [
                        (getattr(b, attr) * (ALPHA_INIT_VALUE / ALPHA_INIT_SCALING))
                        .abs()
                        .mean()
                        .item()
                        for b in blocks
                    ]
                )

            sa_attn_lr = block_lr(agent.sa_blocks, "attn_alpha")
            sa_mlp_lr = block_lr(agent.sa_blocks, "mlp_alpha")
            actor_cross_attn_lr = block_lr(agent.actor_cross_blocks, "attn_alpha")
            actor_cross_mlp_lr = block_lr(agent.actor_cross_blocks, "mlp_alpha")
            critic_cross_attn_lr = block_lr(agent.critic_cross_blocks, "attn_alpha")
            critic_cross_mlp_lr = block_lr(agent.critic_cross_blocks, "mlp_alpha")
            actor_refine_attn_lr = block_lr(agent.actor_refine_blocks, "attn_alpha")
            actor_refine_mlp_lr = block_lr(agent.actor_refine_blocks, "mlp_alpha")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
        writer.add_scalar("diag/adv_corr_shaped", adv_corr.item(), global_step)
        writer.add_scalar("diag/adv_sign_agree_shaped_z", adv_sign_agree.item(), global_step)
        writer.add_scalar("diag/adv_sign_agree_raw", adv_raw_sign_agree.item(), global_step)
        writer.add_scalar("diag/adv_mean_abs", adv_mean_abs.item(), global_step)
        writer.add_scalar("diag/adv_saturation", adv_saturation.item(), global_step)
        writer.add_scalar("diag/update_adv_sign_agree_raw", update_adv_sign_agree, global_step)
        writer.add_scalar("diag/update_adv_mean_abs", update_adv_mean_abs, global_step)
        writer.add_scalar("diag/update_adv_saturation", update_adv_saturation, global_step)
        writer.add_scalar("diag/adv_hlcdf_u_mean", hlcdf_u_mean.item(), global_step)
        writer.add_scalar("diag/adv_hlcdf_u_std", hlcdf_u_std.item(), global_step)
        writer.add_scalar("diag/adv_hlcdf_edge_frac", hlcdf_edge_frac.item(), global_step)
        writer.add_scalar("diag/adv_hlcdf_z_abs", hlcdf_z_abs.item(), global_step)
        writer.add_scalar("grad/actor_norm", actor_gn.item(), global_step)
        writer.add_scalar("grad/critic_norm", critic_gn.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/dist_entropy", diag_entropy, global_step)
        writer.add_scalar("diag/conc_alpha_or_absmean", diag_conc_a, global_step)
        writer.add_scalar("diag/conc_beta_or_std", diag_conc_b, global_step)
        writer.add_scalar("diag/ngpt_sa_attn_lr", sa_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_sa_mlp_lr", sa_mlp_lr, global_step)
        writer.add_scalar("diag/ngpt_actor_cross_attn_lr", actor_cross_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_actor_cross_mlp_lr", actor_cross_mlp_lr, global_step)
        writer.add_scalar("diag/ngpt_critic_cross_attn_lr", critic_cross_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_critic_cross_mlp_lr", critic_cross_mlp_lr, global_step)
        writer.add_scalar("diag/ngpt_actor_refine_attn_lr", actor_refine_attn_lr, global_step)
        writer.add_scalar("diag/ngpt_actor_refine_mlp_lr", actor_refine_mlp_lr, global_step)
        writer.add_scalar("diag/obs_branch_id_scale", agent.obs_branch_scale[0].item(), global_step)
        writer.add_scalar("diag/obs_branch_type_scale", agent.obs_branch_scale[1].item(), global_step)
        writer.add_scalar("diag/obs_branch_shared_value_scale", agent.obs_branch_scale[2].item(), global_step)
        writer.add_scalar("diag/obs_branch_channel_value_scale", agent.obs_branch_scale[3].item(), global_step)
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
