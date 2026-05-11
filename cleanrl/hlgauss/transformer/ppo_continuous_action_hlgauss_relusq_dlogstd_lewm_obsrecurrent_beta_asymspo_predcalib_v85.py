# PPO + HL-Gauss with a LeWM-style action-conditioned latent world model.
#
# Key ideas:
# - a linear observation mixer maps the full raw observation vector to 8 recurrent latent tokens
# - an encoder transformer refines the mixed observation tokens directly
# - a separate predictor transformer rolls latent tokens forward from latent/action history
# - a standard relu-squared MLP PPO agent acts on detached mixed-observation latent tokens
# - Xavier/Glorot init on tokenizer and transformer layers
# - LeWM-style next-encoder-latent MSE: pred(z_t, a_t) targets encoder(o_{t+1})
# - 5-step teacher-forced WM training masked across episode boundaries
# - LeWM-style SIGReg regularizes the full encoded latent sequence toward an isotropic Gaussian
# - v59 predicts one state+outcome target embedding per transition:
#   target(obs_{t+1}, reward_t, continuation_t), with reward/continuation
#   entering the predicted embedding rather than detached dynamics heads
# - imagined actor uses asymmetric half-strength SPO on detached world-model latent rollouts
# - imagined critic uses an HL-Gauss value head for Dreamer-style lambda returns
# - v44: dream construction runs the WM in eval mode, termination uses soft continuation
#   for GAE without also masking by sampled terminal, and the unused dynamics value loss is disabled
# - v45: immediate reward readout was action-aware: r_hat = g(z_t, a_t, z_hat_{t+1})
# - v46: agent critic used a scalar ReLU-squared value head to avoid early HL-Gauss
#   support-edge bootstraps poisoning imagined PPO; v54 returns to HL-Gauss value targets
# - v47: imagined PPO uses one fixed dream buffer per rollout iteration, and the detached
#   latent agent input is RMS-normalized before the standard ReLU-squared actor/critic
# - v48: imagined lambda returns reset across non-learnable sampled-terminal steps; dreamed
#   diagnostics track advantage/action correlation and policy-neighborhood reward sensitivity
# - v51: dreamed rollouts are prompted with recent same-episode real latent/action context
#   before generated policy actions, matching Dreamer-style prompted generation
# - v52: predictor emits next latents through a LeWM-style projection instead of a
#   zero-initialized residual delta, removing the identity shortcut that made dreams
#   nearly action-indifferent
# - v53: expands the rollout bottleneck to 8 latent tokens and represents continuous
#   actions as one explicit predictor token per action dimension
# - v54: replaces SiLU/gated hidden activations with ReLU-squared throughout
#   the world model, predictor, action embedder, and readout projections
#   and uses standard Pre-LN residuals with parameter-golf-style residual mixing
# - v55: actor/critic use ReLU-squared MLP heads with RMSNorm after each hidden
#   activation, and real rollout PPO is restored as an anchor using stored
#   rollout latent tokens with real GAE advantages/returns
# - v56: widens actor/critic hidden layers to 256, removes the actor mean/logstd
#   clamps, and stores real rollout logprobs per action dimension like dreams
# - v57: aligns real rollout actor training with imagined actor training by using
#   the same asymmetric SPO objective; v67 restores asymmetric half-strength SPO
# - v58: separates actor and critic latent-input normalization so critic/value
#   gradients cannot move the actor feature scale, and logs real KL like SPO refs
# - v60: restores LeWM-style target-gradient flow for summary prediction while
#   detaching future target summaries when they are used as teacher-forced context
# - v61: restores LeWM-style multi-step predictor context and per-layer action
#   conditioning via AdaLN, and regularizes the full state+outcome summary
# - v62: replaces per-observation-dimension tokens with 8 mixed observation tokens:
#   obs_dim -> Linear(NUM_OBS_TOKENS * MODEL_DIM)
# - v63: constructs the full dreamed PPO batch in GPU chunks, stages it on CPU,
#   then streams shuffled dream minibatches back to CUDA for multi-epoch PPO
# - v64: removes learned summary/state query tokens; the recurrent world-model state
#   is now 8 mixed observation latent tokens plus 2 reward/continuation outcome tokens
# - v65: logs real and imagined PPO return targets separately for each agent minibatch
# - v66: replaces tanh-squashed Gaussian actions with D4-style Beta policies on
#   normalized action coordinates, linearly mapped to the environment action box
# - v67: restores asymmetric half-strength SPO for real and imagined Beta-policy updates
# - v68: fixes time-limit bootstrapping, boundary target validity, direct outcome-token
#   prediction gradients, observation-only SIGReg, and survival-weighted soft dream continuations
# - v69: keeps learned outcome target projections inside the JEPA embedding system:
#   full state+outcome summaries get SIGReg and reward/continue are decoded by
#   distance to split learned outcome-token codebooks, not a trained CE/BCE dynamics head
# - v82: returns to the v69 backbone but decodes reward/termination through detached
#   supervised probes from predicted outcome tokens, avoiding codebook-distance decode drift
# - v83: removes predicted-token outcome probes; reward/termination scalars are decoded
#   through inverse heads calibrated only on target outcome tokens, while predicted outcome
#   tokens remain trained by the JEPA embedding objective
# - v85: calibrates the same outcome decoder on both target tokens and detached predicted
#   tokens, closing the encoded-vs-predicted readout gap without reward/termination
#   gradients entering the JEPA predictor
import os
import random
import sys
import time
import warnings
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cleanrl.shared.hl_gauss import HLGaussSupport


MODEL_DIM = 64
REWARD_FEATURE_DIM = 512
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
FFN_MULT = 2
DYN_NUM_LAYERS = 2
AGENT_NUM_LAYERS = 2
PRED_NUM_LAYERS = 2
PRED_DROPOUT = 0.0
PRED_CONTEXT = 5
DEFAULT_PRED_CONTEXT = 5
NUM_OBS_TOKENS = 8
NUM_OUTCOME_TOKENS = 2
NUM_LATENT_TOKENS = NUM_OBS_TOKENS + NUM_OUTCOME_TOKENS
SCALAR_EMBED_DIM = 32
AGENT_INPUT_DIM = NUM_OBS_TOKENS * MODEL_DIM
AGENT_HIDDEN_DIM = 256
SAMPLE_EPS = 1e-7


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
    async_vector_env: bool = True
    """use subprocess vector envs to overlap CPU MuJoCo stepping"""
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
    wm_update_epochs: int = 1
    """number of world-model epochs per rollout iteration after warmup starts"""
    agent_update_epochs: int = 4
    """number of PPO epochs per rollout iteration once agent training is enabled"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """PPO reference clip coefficient used for KL/clipfrac diagnostics"""
    spo_eps_low: float = 0.40
    """SPO bound when ratio drift opposes the advantage direction; half-strength vs 0.20"""
    spo_eps_high: float = 0.56
    """SPO bound when ratio drift agrees with the advantage direction; half-strength vs 0.28"""
    clip_vloss: bool = True
    """retained for CLI compatibility; imagined value loss uses HL-Gauss targets"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    actor_mean_scale: float = 3.0
    """retained for CLI compatibility; v66 uses Beta concentration heads"""
    use_pmpo: bool = False
    """retained for CLI compatibility; v67 uses advantage-magnitude SPO, not PMPO"""
    detach_world_model_from_agent: bool = True
    """if toggled, PPO and dreamed agent losses see detached world-model latent tokens"""

    # HL-Gauss specific
    num_bins: int = 51
    """number of bins for the categorical value head"""
    v_min: float = -5.0
    """minimum value of the support (in symlog space)"""
    v_max: float = 5.0
    """maximum value of the support (in symlog space)"""
    sigma_ratio: float = 0.5
    """sigma / bin_width ratio for HL-Gauss target smoothing"""

    # LeWM dynamics auxiliary
    dyn_horizon: int = 5
    """teacher-forced dynamics horizon"""
    pred_context: int = DEFAULT_PRED_CONTEXT
    """number of summary steps the predictor can attend over; v61 defaults to the full dynamics horizon"""
    dyn_latent_coef: float = 1.0
    """weight on next-dynamics-token prediction"""
    dyn_reward_coef: float = 0.25
    """weight on target reward-outcome inverse decoder calibration"""
    dyn_termination_coef: float = 0.25
    """weight on target continuation-outcome inverse decoder calibration"""
    outcome_decode_temp: float = 0.25
    """retained for CLI compatibility; v83 uses target-calibrated inverse outcome decoders"""
    reward_num_bins: int = 51
    """number of bins for the learned reward outcome token"""
    reward_v_min: float = -10.0
    """minimum reward support for the learned reward outcome token"""
    reward_v_max: float = 10.0
    """maximum reward support for the learned reward outcome token"""
    reward_sigma_ratio: float = 0.75
    """sigma / bin_width ratio for the auxiliary reward support"""
    imagine_horizon: int = 16
    """dream rollout horizon for Dreamer-style imagined GAE"""
    dream_prompt_len: int = DEFAULT_PRED_CONTEXT
    """real same-episode summary/action context length used to prompt dreamed rollouts"""
    imagine_actor_coef: float = 1.0
    """weight on the imagined-rollout actor objective"""
    imagine_critic_coef: float = 0.5
    """weight on the imagined-rollout critic objective"""
    imagine_actor_ent_coef: float = 0.0
    """entropy bonus for the dreamed PPO actor update"""
    imagine_update_epochs: int = 4
    """number of PPO epochs over the fixed imagined rollout buffer"""
    imagine_start_step: int = 0
    """global step at which dreamed updates become active"""
    wm_warmup_steps: int = 100000
    """number of env steps to train only the world model before enabling agent updates"""
    sigreg_coef: float = 0.09
    """weight on the SIGReg latent anti-collapse regularizer"""
    sigreg_num_proj: int = 1024
    """number of random projections used by SIGReg"""
    sigreg_knots: int = 17
    """number of quadrature knots used by SIGReg"""
    sigreg_min_valid: int = 32
    """minimum valid samples required for a masked timestep to contribute to SIGReg"""
    dynamics_diagnostic_batch: int = 1024
    """number of real rollout starts used for detached dynamics diagnostics"""
    imagination_diagnostic_batch: int = 512
    """number of dreamed starts used for detached imagination control-signal diagnostics"""
    dream_build_batch_size: int = 16384
    """number of real rollout starts to dream at once before staging tensors on CPU"""
    action_sensitivity_samples: int = 8
    """number of random actions per state for reward action-sensitivity diagnostics"""

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


def xavier_init_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def set_requires_grad(modules, requires_grad):
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(requires_grad)


def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ReluSq(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class ReluSqRMSHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_std=1.0):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            ReluSq(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            ReluSq(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, output_dim), std=output_std),
        )

    def forward(self, x):
        return self.net(x)


def relu_sq(x):
    return torch.relu(x).square()


class SIGReg(nn.Module):
    """LeWM-style Sketched Isotropic Gaussian Regularizer."""

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
        A = torch.randn(dim, self.num_proj, device=device, dtype=dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8))
        return A

    def forward(self, proj, A=None):
        # proj: (T, B, D)
        if A is None:
            A = self.sample_projection(proj.size(-1), proj.device, proj.dtype)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


def build_rope_cache(num_obs_tokens, num_special_tokens, head_dim, device, base=10000.0):
    assert head_dim % 2 == 0
    total_tokens = num_obs_tokens + num_special_tokens
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(num_obs_tokens, device=device).float()
    freqs = torch.outer(positions, theta)

    cos = torch.ones(total_tokens, head_dim // 2, device=device)
    sin = torch.zeros(total_tokens, head_dim // 2, device=device)
    cos[num_special_tokens:] = torch.cos(freqs)
    sin[num_special_tokens:] = torch.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def temporal_causal_mask(context_len, tokens_per_step, device):
    total_tokens = context_len * tokens_per_step
    token_steps = torch.arange(total_tokens, device=device) // tokens_per_step
    future = token_steps.unsqueeze(0) > token_steps.unsqueeze(1)
    mask = torch.zeros(total_tokens, total_tokens, device=device)
    return mask.masked_fill(future, float("-inf"))


def attention(q, k, v, dropout_p=0.0, attn_mask=None):
    if q.is_cuda and attn_mask is None:
        attn_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype), dropout_p=dropout_p
                    )
                return out.to(q.dtype)
            except RuntimeError:
                pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_mult=2):
        super().__init__()
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = None

        for module in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            xavier_init_linear(module)

    def forward(self, x, rope_cos, rope_sin, *, x0, attn_mask=None):
        batch, seq_len, width = x.shape
        mix = self.resid_mix.to(dtype=x.dtype, device=x.device)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        h = self.attn_norm(x)
        q = self.wq(h).view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, seq_len, self.head_dim)
            k = k.reshape(batch, self.num_q_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, seq_len, self.head_dim)
            v = v.reshape(batch, self.num_q_heads, seq_len, self.head_dim)

        attn_out = attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        x = x + self.attn_scale.to(dtype=x.dtype, device=x.device)[None, None, :] * self.wo(attn_out)

        h = self.ffn_norm(x)
        x = x + self.ffn_scale.to(dtype=x.dtype, device=x.device)[None, None, :] * self.w2(relu_sq(self.w1(h)))
        return x


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_mult=2, dropout=0.1):
        super().__init__()
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads
        self.dropout = dropout

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = None
        self.adaln = nn.Sequential(ReluSq(), nn.Linear(dim, 6 * dim))

        for module in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            xavier_init_linear(module)
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def _modulate(self, x, shift, scale):
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return x * (1.0 + scale) + shift

    def forward(self, x, action_features, rope_cos, rope_sin, attn_mask=None):
        batch, seq_len, width = x.shape
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln(action_features).chunk(6, dim=-1)

        h = self._modulate(self.attn_norm(x), shift_attn, scale_attn)
        q = self.wq(h).view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, seq_len, self.head_dim)
            k = k.reshape(batch, self.num_q_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, seq_len, self.head_dim)
            v = v.reshape(batch, self.num_q_heads, seq_len, self.head_dim)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = attention(q, k, v, dropout_p=dropout_p, attn_mask=attn_mask)
        attn_out = self.wo(attn_out.transpose(1, 2).reshape(batch, seq_len, width))
        attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        if gate_attn.dim() == 2:
            gate_attn = gate_attn.unsqueeze(1)
            gate_ffn = gate_ffn.unsqueeze(1)
        x = x + gate_attn * attn_out

        h = self._modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        ffn_out = self.w2(relu_sq(self.w1(h)))
        ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x = x + gate_ffn * ffn_out
        return x


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        num_bins,
        reward_num_bins,
        detach_world_model_from_agent=True,
        actor_mean_scale=3.0,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.act_dim = act_dim
        self.reward_num_bins = reward_num_bins
        self.detach_world_model_from_agent = detach_world_model_from_agent
        self.actor_mean_scale = actor_mean_scale
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

        self.obs_input_norm = RMSNorm(obs_dim)
        self.obs_mix_proj = xavier_init_linear(nn.Linear(obs_dim, NUM_OBS_TOKENS * MODEL_DIM))
        self.obs_token_norm = RMSNorm(MODEL_DIM)

        self.dyn_embed_norm = RMSNorm(MODEL_DIM)
        self.dyn_layers = nn.ModuleList(
            [TransformerBlock(MODEL_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT) for _ in range(DYN_NUM_LAYERS)]
        )
        self.dyn_final_norm = RMSNorm(MODEL_DIM)
        self.dyn_next_proj = xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM))

        self.pred_action_in_proj = xavier_init_linear(nn.Linear(1, SCALAR_EMBED_DIM))
        self.pred_action_out_proj = xavier_init_linear(nn.Linear(SCALAR_EMBED_DIM, MODEL_DIM))
        self.pred_action_dim_embed = nn.Parameter(torch.empty(act_dim, MODEL_DIM))
        nn.init.xavier_uniform_(self.pred_action_dim_embed)
        self.pred_action_cond_proj = xavier_init_linear(nn.Linear(act_dim, MODEL_DIM))
        self.pred_layers = nn.ModuleList(
            [
                AdaLNTransformerBlock(MODEL_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT, dropout=PRED_DROPOUT)
                for _ in range(PRED_NUM_LAYERS)
            ]
        )
        self.pred_final_norm = RMSNorm(MODEL_DIM)
        self.pred_next_proj = xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM))

        head_dim = MODEL_DIM // NUM_Q_HEADS
        dyn_rope_cos, dyn_rope_sin = build_rope_cache(
            NUM_OBS_TOKENS, 0, head_dim, torch.device("cpu")
        )
        pred_tokens_per_step = act_dim + NUM_LATENT_TOKENS
        pred_rope_cos, pred_rope_sin = build_rope_cache(
            PRED_CONTEXT * pred_tokens_per_step, 0, head_dim, torch.device("cpu")
        )
        self.register_buffer("dyn_rope_cos", dyn_rope_cos)
        self.register_buffer("dyn_rope_sin", dyn_rope_sin)
        self.register_buffer("pred_rope_cos", pred_rope_cos)
        self.register_buffer("pred_rope_sin", pred_rope_sin)

        self.critic = ReluSqRMSHead(AGENT_INPUT_DIM, AGENT_HIDDEN_DIM, num_bins, output_std=1.0)
        self.actor_beta = ReluSqRMSHead(AGENT_INPUT_DIM, AGENT_HIDDEN_DIM, 2 * act_dim, output_std=0.01)
        self.actor_input_norm = RMSNorm(AGENT_INPUT_DIM)
        self.critic_input_norm = RMSNorm(AGENT_INPUT_DIM)
        self.reward_outcome_input_norm = RMSNorm(reward_num_bins)
        self.reward_outcome_proj = xavier_init_linear(nn.Linear(reward_num_bins, MODEL_DIM))
        self.continuation_outcome_input_norm = RMSNorm(1)
        self.continuation_outcome_proj = xavier_init_linear(nn.Linear(1, MODEL_DIM))
        self.outcome_token_norm = RMSNorm(MODEL_DIM)
        self.outcome_decode_temp = 0.25
        self.register_buffer("reward_codebook_probs", torch.eye(reward_num_bins), persistent=False)
        self.reward_outcome_unproj = xavier_init_linear(nn.Linear(MODEL_DIM, reward_num_bins))
        self.continuation_outcome_unproj = xavier_init_linear(nn.Linear(MODEL_DIM, 1))
        nn.init.constant_(self.continuation_outcome_unproj.bias, -5.0)

    def _action_distribution(self, agent_input):
        beta_head = self.actor_beta(agent_input)
        head_alpha, head_beta = beta_head.chunk(2, dim=-1)
        alpha = 1.0 + F.softplus(head_alpha)
        beta = 1.0 + F.softplus(head_beta)
        return Beta(alpha, beta)

    def _z_to_action(self, action_z):
        return self.action_low + (self.action_high - self.action_low) * action_z

    def _action_to_z(self, action):
        action_z = (action - self.action_low) / (self.action_high - self.action_low)
        return action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def _beta_action_logprob_entropy(self, dist, action=None, action_z=None, sum_logprob=True):
        if action_z is None:
            if action is not None:
                action_z = self._action_to_z(action)
            else:
                action_z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                action = self._z_to_action(action_z)
        else:
            action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(action_z)
        logprob_per_dim = dist.log_prob(action_z)
        entropy_per_dim = dist.entropy()
        if sum_logprob:
            logprob = logprob_per_dim.sum(1)
            entropy = entropy_per_dim.sum(1)
        else:
            logprob = logprob_per_dim
            entropy = entropy_per_dim
        return action, action_z, logprob, entropy

    def _action_mean_from_dist(self, dist):
        return self._z_to_action(dist.mean)

    def _action_std_from_dist(self, dist):
        return dist.stddev * (self.action_high - self.action_low)

    def _encode_dynamics_tokens(self, x):
        batch = x.shape[0]
        obs_flat = x.reshape(batch, -1)
        obs_tokens = self.obs_mix_proj(self.obs_input_norm(obs_flat))
        obs_tokens = self.obs_token_norm(obs_tokens.reshape(batch, NUM_OBS_TOKENS, MODEL_DIM))
        dyn_tokens = self.dyn_embed_norm(obs_tokens)

        dyn_x0 = dyn_tokens
        for layer in self.dyn_layers:
            dyn_tokens = layer(dyn_tokens, self.dyn_rope_cos, self.dyn_rope_sin, x0=dyn_x0)

        return self.dyn_final_norm(dyn_tokens)

    def _encode_obs_latent_tokens(self, x):
        obs_tokens = self._encode_dynamics_tokens(x)
        return self.dyn_next_proj(obs_tokens)

    def _reward_outcome_token(self, reward_probs):
        return self.outcome_token_norm(
            self.reward_outcome_proj(self.reward_outcome_input_norm(reward_probs))
        )

    def _continuation_outcome_token(self, continuations):
        continuation_input = continuations.unsqueeze(-1)
        return self.outcome_token_norm(
            self.continuation_outcome_proj(self.continuation_outcome_input_norm(continuation_input))
        )

    def _outcome_tokens_from_labels(self, reward_probs, continuations):
        reward_token = self._reward_outcome_token(reward_probs)
        continuation_token = self._continuation_outcome_token(continuations)
        return torch.stack([reward_token, continuation_token], dim=1)

    def _outcome_tokens(self, obs_tokens, reward_probs, continuations):
        return self._outcome_tokens_from_labels(reward_probs, continuations)

    def _neutral_outcome_tokens(self, obs_tokens):
        reward_probs = self.reward_codebook_probs.to(
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )[self.reward_num_bins // 2].expand(obs_tokens.shape[0], -1)
        continuations = obs_tokens.new_ones((obs_tokens.shape[0],))
        return self._outcome_tokens(obs_tokens, reward_probs, continuations)

    def _latent_from_obs(self, obs_tokens, outcome_tokens):
        return torch.cat([obs_tokens, outcome_tokens], dim=1)

    def _encode_online_summary(self, x):
        obs_tokens = self._encode_obs_latent_tokens(x)
        return self._latent_from_obs(obs_tokens, self._neutral_outcome_tokens(obs_tokens))

    def encode_target_summary(self, x, reward_probs, continuations):
        obs_tokens = self._encode_obs_latent_tokens(x)
        outcome_tokens = self._outcome_tokens(obs_tokens, reward_probs, continuations)
        return self._latent_from_obs(obs_tokens, outcome_tokens)

    def decode_outcomes(self, summary_tokens, detach_summary=True):
        outcome_tokens = summary_tokens[:, NUM_OBS_TOKENS:]
        if detach_summary:
            outcome_tokens = outcome_tokens.detach()
        reward_logits = self.reward_outcome_unproj(outcome_tokens[:, 0])
        termination_logits = self.continuation_outcome_unproj(outcome_tokens[:, 1]).squeeze(-1)
        return reward_logits, termination_logits

    def _actor_features_from_latents(self, latent_tokens):
        obs_tokens = latent_tokens[:, :NUM_OBS_TOKENS]
        return self.actor_input_norm(obs_tokens.reshape(obs_tokens.shape[0], -1))

    def _critic_features_from_latents(self, latent_tokens):
        obs_tokens = latent_tokens[:, :NUM_OBS_TOKENS]
        return self.critic_input_norm(obs_tokens.reshape(obs_tokens.shape[0], -1))

    def _value_from_agent_input(self, agent_input, hl_support):
        if hl_support is None:
            raise ValueError("hl_support is required for HL-Gauss value decoding")
        return hl_support.to_scalar(self.critic(agent_input))

    def _encode_critic_features(self, x):
        latent_tokens = self._encode_online_summary(x)
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        return self._critic_features_from_latents(latent_tokens)

    def get_agent_latents(self, x):
        latent_tokens = self._encode_online_summary(x)
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        return latent_tokens

    def predict_dynamics(self, x, action):
        summary_tokens = self.get_summary_targets(x)
        return self.dynamics_step(summary_tokens, action)

    def predict_next_latents(self, latent_tokens, action):
        latent_history = latent_tokens.unsqueeze(1)
        action_history = action.unsqueeze(1)
        return self.predict_next_latents_from_history(latent_history, action_history)

    def predict_next_latents_all_from_history(self, latent_history, action_history):
        batch, context_len, num_latents, width = latent_history.shape
        if context_len > PRED_CONTEXT:
            raise ValueError(f"context_len={context_len} exceeds PRED_CONTEXT={PRED_CONTEXT}")

        action_tokens = self.pred_action_out_proj(relu_sq(self.pred_action_in_proj(action_history.unsqueeze(-1))))
        action_tokens = action_tokens + self.pred_action_dim_embed.view(1, 1, self.act_dim, width)
        tokens_per_step = self.act_dim + num_latents
        pred_tokens = torch.cat([action_tokens, latent_history], dim=2)
        pred_tokens = pred_tokens.reshape(batch, context_len * tokens_per_step, width)
        action_features = self.pred_action_cond_proj(action_history)
        action_features = action_features.unsqueeze(2).expand(batch, context_len, tokens_per_step, width)
        action_features = action_features.reshape(batch, context_len * tokens_per_step, width)
        rope_cos = self.pred_rope_cos[: context_len * tokens_per_step]
        rope_sin = self.pred_rope_sin[: context_len * tokens_per_step]
        attn_mask = temporal_causal_mask(context_len, tokens_per_step, pred_tokens.device)
        for layer in self.pred_layers:
            pred_tokens = layer(pred_tokens, action_features, rope_cos, rope_sin, attn_mask=attn_mask)
        pred_latents = self.pred_next_proj(self.pred_final_norm(pred_tokens))
        pred_latents = pred_latents.reshape(batch, context_len, tokens_per_step, width)
        return pred_latents[:, :, self.act_dim :]

    def predict_next_latents_from_history(self, latent_history, action_history):
        pred_latents = self.predict_next_latents_all_from_history(latent_history, action_history)
        return pred_latents[:, -1]

    def dynamics_teacher_forced(self, latent_history, action_history):
        batch, horizon, num_latents, width = latent_history.shape
        pred_next_latents = self.predict_next_latents_all_from_history(latent_history, action_history)
        pred_reward_logits, pred_termination_logits = self.decode_outcomes(
            pred_next_latents.reshape(batch * horizon, num_latents, width),
        )
        pred_reward_logits = pred_reward_logits.reshape(batch, horizon, -1)
        pred_termination_logits = pred_termination_logits.reshape(batch, horizon)
        return pred_next_latents, pred_reward_logits, pred_termination_logits

    def dynamics_step_from_history(self, summary_history, action_history):
        latent_history = summary_history
        pred_next_latents = self.predict_next_latents_from_history(latent_history, action_history)
        pred_next_summary = pred_next_latents
        pred_reward_logits, pred_termination_logits = self.decode_outcomes(pred_next_latents)
        return pred_next_summary, pred_reward_logits, pred_termination_logits

    def dynamics_step(self, summary_tokens, action):
        return self.dynamics_step_from_history(summary_tokens.unsqueeze(1), action.unsqueeze(1))

    def get_imagined_action_dist(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        return self._action_distribution(agent_input)

    def get_imagined_action_mean(self, summary_tokens):
        dist = self.get_imagined_action_dist(summary_tokens)
        return self._action_mean_from_dist(dist)

    def get_imagined_action_std(self, summary_tokens):
        dist = self.get_imagined_action_dist(summary_tokens)
        return self._action_std_from_dist(dist)

    def get_imagined_raw_action_mean(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        return self.actor_beta(agent_input)

    def get_imagined_value(self, summary_tokens, hl_support=None):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self._value_from_agent_input(agent_input, hl_support)

    def get_imagined_value_logits(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self.critic(agent_input)

    def get_imagined_action_and_value(self, summary_tokens, hl_support, action=None, action_z=None):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        actor_input = self._actor_features_from_latents(latent_tokens)
        critic_input = self._critic_features_from_latents(latent_tokens)
        dist = self._action_distribution(actor_input)
        action, action_z, logprob, entropy = self._beta_action_logprob_entropy(
            dist, action=action, action_z=action_z, sum_logprob=False
        )
        value = self._value_from_agent_input(critic_input, hl_support)
        return action, action_z, logprob, entropy, value

    def get_summary_targets(self, x):
        return self._encode_online_summary(x)

    def get_value(self, x, hl_support=None):
        agent_input = self._encode_critic_features(x)
        return self._value_from_agent_input(agent_input, hl_support)

    def get_value_logits(self, x):
        agent_input = self._encode_critic_features(x)
        return self.critic(agent_input)

    def get_value_logits_from_latents(self, latent_tokens):
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self.critic(agent_input)

    def get_action_and_value_from_latents(
        self,
        latent_tokens,
        hl_support,
        action=None,
        action_z=None,
        sum_logprob=True,
    ):
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        actor_input = self._actor_features_from_latents(latent_tokens)
        critic_input = self._critic_features_from_latents(latent_tokens)
        dist = self._action_distribution(actor_input)
        action, action_z, logprob, entropy = self._beta_action_logprob_entropy(
            dist,
            action=action,
            action_z=action_z,
            sum_logprob=sum_logprob,
        )
        value = self._value_from_agent_input(critic_input, hl_support)
        return action, action_z, logprob, entropy, value

    def get_action_logprob_entropy_from_latents(
        self,
        latent_tokens,
        action=None,
        action_z=None,
        sum_logprob=True,
    ):
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        dist = self._action_distribution(agent_input)
        return self._beta_action_logprob_entropy(
            dist,
            action=action,
            action_z=action_z,
            sum_logprob=sum_logprob,
        )

    def get_action_and_value(self, x, hl_support, action=None, action_z=None):
        latent_tokens = self.get_agent_latents(x)
        return self.get_action_and_value_from_latents(
            latent_tokens,
            hl_support,
            action=action,
            action_z=action_z,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.pred_context < 1 or args.pred_context > PRED_CONTEXT:
        raise ValueError(f"--pred-context must be in [1, {PRED_CONTEXT}]")
    if args.dyn_horizon < 1:
        raise ValueError("--dyn-horizon must be at least 1")
    if args.dyn_horizon > args.pred_context:
        raise ValueError("--dyn-horizon must be <= --pred-context for teacher-forced contextual prediction")
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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda")

    # env setup
    vector_env_cls = gym.vector.AsyncVectorEnv if args.async_vector_env else gym.vector.SyncVectorEnv
    envs = vector_env_cls(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        args.num_bins,
        args.reward_num_bins,
        detach_world_model_from_agent=args.detach_world_model_from_agent,
        actor_mean_scale=args.actor_mean_scale,
    ).to(device)
    agent.outcome_decode_temp = args.outcome_decode_temp
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def module_parameters(*modules):
        params = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    grad_clip_groups = [
        (
            "wm",
            [
                *module_parameters(agent.obs_input_norm, agent.obs_mix_proj, agent.obs_token_norm),
                *module_parameters(agent.dyn_embed_norm, agent.dyn_layers, agent.dyn_final_norm, agent.dyn_next_proj),
                *module_parameters(
                    agent.reward_outcome_input_norm,
                    agent.reward_outcome_proj,
                    agent.continuation_outcome_input_norm,
                    agent.continuation_outcome_proj,
                    agent.outcome_token_norm,
                ),
                agent.pred_action_in_proj.weight,
                agent.pred_action_in_proj.bias,
                agent.pred_action_out_proj.weight,
                agent.pred_action_out_proj.bias,
                agent.pred_action_dim_embed,
                agent.pred_action_cond_proj.weight,
                agent.pred_action_cond_proj.bias,
                *module_parameters(agent.pred_layers, agent.pred_final_norm, agent.pred_next_proj),
            ],
        ),
        (
            "outcome_inverse",
            module_parameters(agent.reward_outcome_unproj, agent.continuation_outcome_unproj),
        ),
        ("actor", module_parameters(agent.actor_input_norm, agent.actor_beta)),
        ("critic", module_parameters(agent.critic_input_norm, agent.critic)),
    ]

    def clip_grad_groups():
        for _, params in grad_clip_groups:
            params_with_grad = [param for param in params if param.grad is not None]
            if params_with_grad:
                nn.utils.clip_grad_norm_(params_with_grad, args.max_grad_norm)

    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device)
    hl_support = HLGaussSupport(args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=True)
    reward_support = HLGaussSupport(
        args.reward_num_bins,
        args.reward_v_min,
        args.reward_v_max,
        args.reward_sigma_ratio,
        device,
        use_symlog=False,
    )
    agent.reward_codebook_probs = reward_support.project(reward_support.support).detach()
    action_low = torch.tensor(envs.single_action_space.low, device=device)
    action_high = torch.tensor(envs.single_action_space.high, device=device)

    def masked_time_major_sigreg(sigreg_latents, sigreg_valids):
        sigreg_losses = []
        A = sigreg.sample_projection(
            sigreg_latents[0].shape[-1],
            sigreg_latents[0].device,
            sigreg_latents[0].dtype,
        )
        for latents, valids in zip(sigreg_latents, sigreg_valids):
            valid_latents = latents[valids]
            if valid_latents.shape[0] >= args.sigreg_min_valid:
                sigreg_losses.append(sigreg(valid_latents.unsqueeze(0), A=A))
        if sigreg_losses:
            return torch.stack(sigreg_losses).mean()
        return sigreg_latents[0].sum() * 0.0

    def imagined_lambda_returns(rewards_hat, continues_hat, values_hat, learn_masks):
        returns = []
        gae = torch.zeros_like(values_hat[-1])
        for step in reversed(range(len(rewards_hat))):
            delta = rewards_hat[step] + args.gamma * continues_hat[step] * values_hat[step + 1] - values_hat[step]
            gae = delta + args.gamma * args.gae_lambda * continues_hat[step] * gae
            gae = torch.where(learn_masks[step], gae, torch.zeros_like(gae))
            returns.append(gae + values_hat[step])
        returns.reverse()
        return returns

    def pearson_corr(x, y):
        if x.numel() <= 1 or y.numel() <= 1:
            return x.sum() * 0.0
        x = x - x.mean()
        y = y - y.mean()
        denom = x.square().mean().sqrt() * y.square().mean().sqrt()
        return (x * y).mean() / denom.clamp_min(1e-8)

    def weighted_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    @torch.no_grad()
    def build_dream_prompt_context(flat_obs, rollout_actions, rollout_boundaries):
        prompt_len = max(1, min(args.dream_prompt_len, args.pred_context))
        flat_inds = torch.arange(args.batch_size, device=device)
        step_inds = flat_inds // args.num_envs
        env_inds = flat_inds % args.num_envs
        prompt_valids = step_inds >= (prompt_len - 1)

        prompt_summary_history = []
        for offset in range(prompt_len):
            hist_step = step_inds - (prompt_len - 1 - offset)
            safe_hist_step = hist_step.clamp(min=0)
            hist_flat_inds = safe_hist_step * args.num_envs + env_inds
            prompt_summary_history.append(agent.get_summary_targets(flat_obs[hist_flat_inds]).detach())

        prompt_action_history = []
        for offset in range(prompt_len - 1):
            action_step = step_inds - (prompt_len - 1 - offset)
            safe_action_step = action_step.clamp(min=0)
            prompt_action_history.append(rollout_actions[safe_action_step, env_inds].detach())

        for back_offset in range(prompt_len - 1):
            boundary_step = step_inds - 1 - back_offset
            safe_boundary_step = boundary_step.clamp(min=0)
            prompt_valids = prompt_valids & (boundary_step >= 0)
            prompt_valids = prompt_valids & (~rollout_boundaries[safe_boundary_step, env_inds].bool())

        return prompt_summary_history, prompt_action_history, prompt_valids

    def build_dream_batch(prompt_summary_history, prompt_action_history, prompt_valids):
        states = []
        raw_actions = []
        action_zs = []
        old_logprobs = []
        values = []
        learn_masks = []
        learn_weights = []
        rewards_hat = []
        continues_hat = []
        policy_reward_sensitivity_stds = []
        policy_reward_sensitivity_ranges = []
        policy_latent_sensitivity_stds = []
        policy_latent_sensitivity_ranges = []
        summary_history = [summary.detach() for summary in prompt_summary_history]
        action_history = [action.detach() for action in prompt_action_history]
        alive = prompt_valids.float()
        diagnostic_n = min(args.imagination_diagnostic_batch, summary_history[-1].shape[0])
        sensitivity_k = max(2, args.action_sensitivity_samples)
        with torch.no_grad():
            for _ in range(args.imagine_horizon):
                summary_state = summary_history[-1].detach()
                states.append(summary_state)
                dream_action, dream_action_z, old_logprob, _, value = agent.get_imagined_action_and_value(
                    summary_state, hl_support
                )
                action_history.append(dream_action.detach())
                context_len = min(args.pred_context, len(summary_history), len(action_history))
                pred_context = torch.stack(summary_history[-context_len:], dim=1)
                action_context = torch.stack(action_history[-context_len:], dim=1)
                pred_next_summary, pred_reward_logits, pred_termination_logits = agent.dynamics_step_from_history(
                    pred_context, action_context
                )
                if diagnostic_n > 0:
                    diag_alive = alive[:diagnostic_n].bool()
                    diag_cpu_rng_state = torch.random.get_rng_state()
                    diag_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    try:
                        diag_state = summary_state[:diagnostic_n]
                        diag_dist = agent.get_imagined_action_dist(diag_state)
                        diag_action_zs = diag_dist.sample((sensitivity_k,)).transpose(0, 1)
                        diag_action_zs = diag_action_zs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                        diag_actions = agent._z_to_action(diag_action_zs)
                        diag_pred_context = torch.stack(
                            [summary[:diagnostic_n] for summary in summary_history[-context_len:]],
                            dim=1,
                        )
                        diag_pred_context = diag_pred_context.unsqueeze(1).expand(
                            -1, sensitivity_k, -1, -1, -1
                        ).reshape(
                            diagnostic_n * sensitivity_k,
                            context_len,
                            NUM_LATENT_TOKENS,
                            MODEL_DIM,
                        )
                        if context_len > 1:
                            previous_actions = torch.stack(
                                [action[:diagnostic_n] for action in action_history[-context_len:-1]],
                                dim=1,
                            )
                            previous_actions = previous_actions.unsqueeze(1).expand(
                                -1, sensitivity_k, -1, -1
                            )
                            diag_action_context = torch.cat(
                                [previous_actions, diag_actions.unsqueeze(2)],
                                dim=2,
                            )
                        else:
                            diag_action_context = diag_actions.unsqueeze(2)
                        diag_action_context = diag_action_context.reshape(
                            diagnostic_n * sensitivity_k,
                            context_len,
                            agent.act_dim,
                        )
                        diag_next_summary, diag_reward_logits, _ = agent.dynamics_step_from_history(
                            diag_pred_context,
                            diag_action_context,
                        )
                        diag_rewards = reward_support.to_scalar(diag_reward_logits).reshape(
                            diagnostic_n,
                            sensitivity_k,
                        )
                        diag_next_latents = diag_next_summary[:, :NUM_LATENT_TOKENS].reshape(
                            diagnostic_n,
                            sensitivity_k,
                            -1,
                        )
                        if bool(diag_alive.any()):
                            alive_diag_rewards = diag_rewards[diag_alive]
                            alive_diag_latents = diag_next_latents[diag_alive]
                            policy_reward_sensitivity_stds.append(
                                alive_diag_rewards.std(dim=1, unbiased=False).mean()
                            )
                            policy_reward_sensitivity_ranges.append(
                                (
                                    alive_diag_rewards.max(dim=1).values
                                    - alive_diag_rewards.min(dim=1).values
                                ).mean()
                            )
                            policy_latent_sensitivity_stds.append(
                                alive_diag_latents.std(dim=1, unbiased=False).norm(dim=-1).mean()
                            )
                            policy_latent_sensitivity_ranges.append(
                                (
                                    alive_diag_latents.max(dim=1).values
                                    - alive_diag_latents.min(dim=1).values
                                ).norm(dim=-1).mean()
                            )
                    finally:
                        torch.random.set_rng_state(diag_cpu_rng_state)
                        if diag_cuda_rng_state is not None:
                            torch.cuda.set_rng_state_all(diag_cuda_rng_state)
                raw_actions.append(dream_action)
                action_zs.append(dream_action_z)
                old_logprobs.append(old_logprob)
                values.append(value)
                learn_masks.append(alive > 1e-6)
                learn_weights.append(alive)
                pred_reward = reward_support.to_scalar(pred_reward_logits)
                termination_prob = torch.sigmoid(pred_termination_logits)
                pred_continue = 1.0 - termination_prob
                rewards_hat.append(pred_reward)
                continues_hat.append(pred_continue)
                alive = alive * pred_continue
                summary_history.append(pred_next_summary.detach())
            bootstrap_value = agent.get_imagined_value(summary_history[-1].detach(), hl_support)

        returns = imagined_lambda_returns(rewards_hat, continues_hat, values + [bootstrap_value], learn_masks)
        states = torch.cat(states, dim=0)
        raw_actions = torch.cat(raw_actions, dim=0)
        action_zs = torch.cat(action_zs, dim=0)
        old_logprobs = torch.cat(old_logprobs, dim=0)
        values = torch.cat(values, dim=0)
        learn_masks = torch.cat(learn_masks, dim=0)
        learn_weights = torch.cat(learn_weights, dim=0)
        returns = torch.cat(returns, dim=0)
        advantages = returns - values
        rewards_flat = torch.cat(rewards_hat, dim=0)
        continues_flat = torch.cat(continues_hat, dim=0)
        with torch.no_grad():
            if bool(learn_masks.any()):
                diag_rewards = rewards_flat[learn_masks]
                diag_continues = continues_flat[learn_masks]
                diag_values = values[learn_masks]
                diag_returns = returns[learn_masks]
                diag_advantages = advantages[learn_masks]
                diag_actions = raw_actions[learn_masks]
                diag_action_zs = action_zs[learn_masks]
            else:
                diag_rewards = rewards_flat
                diag_continues = continues_flat
                diag_values = values
                diag_returns = returns
                diag_advantages = advantages
                diag_actions = raw_actions
                diag_action_zs = action_zs
            action_dim_corrs = []
            action_z_dim_corrs = []
            for action_dim in range(agent.act_dim):
                action_dim_corrs.append(pearson_corr(diag_advantages, diag_actions[:, action_dim]).abs())
                action_z_dim_corrs.append(
                    pearson_corr(diag_advantages, diag_action_zs[:, action_dim]).abs()
                )
            action_dim_corrs = torch.stack(action_dim_corrs)
            action_z_dim_corrs = torch.stack(action_z_dim_corrs)
            action_norm = diag_actions.norm(dim=1)
            action_energy = diag_actions.square().sum(dim=1)
            diagnostics = {
                "reward_mean": diag_rewards.mean().item(),
                "reward_std": diag_rewards.std(unbiased=False).item(),
                "continue_mean": diag_continues.mean().item(),
                "learn_mask_frac": learn_masks.float().mean().item(),
                "learn_weight_mean": learn_weights.mean().item(),
                "prompt_valid_frac": prompt_valids.float().mean().item(),
                "value_mean": diag_values.mean().item(),
                "value_max": diag_values.max().item(),
                "bootstrap_value_mean": bootstrap_value.mean().item(),
                "return_mean": diag_returns.mean().item(),
                "return_std": diag_returns.std(unbiased=False).item(),
                "return_max": diag_returns.max().item(),
                "advantage_abs_mean": diag_advantages.abs().mean().item(),
                "advantage_std": diag_advantages.std(unbiased=False).item(),
                "advantage_action_norm_corr": pearson_corr(diag_advantages, action_norm).item(),
                "advantage_action_energy_corr": pearson_corr(diag_advantages, action_energy).item(),
                "advantage_action_dim_abs_corr_mean": action_dim_corrs.mean().item(),
                "advantage_action_dim_abs_corr_max": action_dim_corrs.max().item(),
                "advantage_action_z_dim_abs_corr_mean": action_z_dim_corrs.mean().item(),
                "advantage_action_z_dim_abs_corr_max": action_z_dim_corrs.max().item(),
            }
            if policy_reward_sensitivity_stds:
                diagnostics.update(
                    reward_policy_action_sensitivity_std=torch.stack(policy_reward_sensitivity_stds).mean().item(),
                    reward_policy_action_sensitivity_range=torch.stack(policy_reward_sensitivity_ranges).mean().item(),
                    latent_policy_action_sensitivity_std=torch.stack(policy_latent_sensitivity_stds).mean().item(),
                    latent_policy_action_sensitivity_range=torch.stack(policy_latent_sensitivity_ranges).mean().item(),
                )
        return (
            states,
            raw_actions,
            action_zs,
            old_logprobs,
            values,
            advantages,
            returns,
            learn_masks,
            learn_weights,
            diagnostics,
        )

    def build_dream_batch_eval(prompt_summary_history, prompt_action_history, prompt_valids):
        was_training = agent.training
        agent.eval()
        try:
            total_starts = prompt_valids.shape[0]
            chunk_size = args.dream_build_batch_size if args.dream_build_batch_size > 0 else total_starts
            tensor_chunks = [[] for _ in range(9)]
            diagnostic_sums = {}
            diagnostic_weight = 0

            for start in range(0, total_starts, chunk_size):
                end = min(start + chunk_size, total_starts)
                chunk_summaries = [summary[start:end] for summary in prompt_summary_history]
                chunk_actions = [action[start:end] for action in prompt_action_history]
                chunk_valids = prompt_valids[start:end]
                chunk_batch = build_dream_batch(chunk_summaries, chunk_actions, chunk_valids)
                for idx, tensor in enumerate(chunk_batch[:9]):
                    tensor_chunks[idx].append(tensor.detach().cpu())
                chunk_weight = end - start
                for key, value in chunk_batch[9].items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value * chunk_weight
                diagnostic_weight += chunk_weight
                del chunk_batch

            tensors = tuple(torch.cat(chunks, dim=0) for chunks in tensor_chunks)
            diagnostics = {
                key: value / max(1, diagnostic_weight)
                for key, value in diagnostic_sums.items()
            }
            return tensors + (diagnostics,)
        finally:
            agent.train(was_training)

    @torch.no_grad()
    def dynamics_diagnostics(flat_obs, rollout_rewards, rollout_actions, rollout_terminations, rollout_boundaries, rollout_valids):
        num_starts = min(args.dynamics_diagnostic_batch, flat_obs.shape[0])
        if num_starts <= 0:
            return {}

        was_training = agent.training
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        agent.eval()
        try:
            sample_inds = torch.randperm(flat_obs.shape[0], device=device)[:num_starts]
            mb_step_inds = sample_inds // args.num_envs
            mb_env_inds = sample_inds % args.num_envs
            summary_history = [agent.get_summary_targets(flat_obs[sample_inds]).detach()]
            action_history = []
            alive = torch.ones(num_starts, device=device)
            pred_reward_sum = torch.zeros(num_starts, device=device)
            true_reward_sum = torch.zeros(num_starts, device=device)
            pred_discounted_return = torch.zeros(num_starts, device=device)
            true_discounted_return = torch.zeros(num_starts, device=device)
            valid_any = torch.zeros(num_starts, device=device, dtype=torch.bool)
            step_abs_errors = []
            step_biases = []
            term_briers = []
            horizon = min(args.dyn_horizon, args.imagine_horizon)

            for horizon_idx in range(horizon):
                future_step_inds = mb_step_inds + horizon_idx
                in_rollout = (future_step_inds < args.num_steps).float()
                safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                future_actions = rollout_actions[safe_step_inds, mb_env_inds]
                future_rewards = rollout_rewards[safe_step_inds, mb_env_inds]
                future_terminations = rollout_terminations[safe_step_inds, mb_env_inds]
                future_boundaries = rollout_boundaries[safe_step_inds, mb_env_inds]
                future_valids = rollout_valids[safe_step_inds, mb_env_inds]
                step_weight = alive * in_rollout * future_valids
                valid_mask = step_weight > 0.0

                action_history.append(future_actions)
                context_len = min(args.pred_context, len(summary_history), len(action_history))
                pred_context = torch.stack(summary_history[-context_len:], dim=1)
                action_context = torch.stack(action_history[-context_len:], dim=1)
                pred_next_summary, pred_reward_logits, pred_termination_logits = agent.dynamics_step_from_history(
                    pred_context, action_context
                )
                pred_reward = reward_support.to_scalar(pred_reward_logits)
                terminal_prob = torch.sigmoid(pred_termination_logits)

                if bool(valid_mask.any()):
                    reward_error = pred_reward - future_rewards
                    step_abs_errors.append(reward_error[valid_mask].abs().mean())
                    step_biases.append(reward_error[valid_mask].mean())
                    term_briers.append((terminal_prob[valid_mask] - future_terminations[valid_mask]).square().mean())

                discount = args.gamma ** horizon_idx
                pred_reward_sum = pred_reward_sum + pred_reward * step_weight
                true_reward_sum = true_reward_sum + future_rewards * step_weight
                pred_discounted_return = pred_discounted_return + pred_reward * step_weight * discount
                true_discounted_return = true_discounted_return + future_rewards * step_weight * discount
                valid_any |= valid_mask
                alive = alive * (1.0 - future_boundaries)
                summary_history.append(pred_next_summary.detach())

            metrics = {}
            if bool(valid_any.any()):
                metrics.update(
                    rollout_reward_step_mae=torch.stack(step_abs_errors).mean().item() if step_abs_errors else 0.0,
                    rollout_reward_step_bias=torch.stack(step_biases).mean().item() if step_biases else 0.0,
                    rollout_reward_sum_mae=(pred_reward_sum[valid_any] - true_reward_sum[valid_any]).abs().mean().item(),
                    rollout_reward_sum_bias=(pred_reward_sum[valid_any] - true_reward_sum[valid_any]).mean().item(),
                    rollout_discounted_return_mae=(
                        pred_discounted_return[valid_any] - true_discounted_return[valid_any]
                    ).abs().mean().item(),
                    rollout_discounted_return_bias=(
                        pred_discounted_return[valid_any] - true_discounted_return[valid_any]
                    ).mean().item(),
                    rollout_terminal_brier=torch.stack(term_briers).mean().item() if term_briers else 0.0,
                    rollout_valid_frac=valid_any.float().mean().item(),
                )

            sensitivity_n = min(256, num_starts)
            sensitivity_k = max(2, args.action_sensitivity_samples)
            sensitivity_summary = summary_history[0][:sensitivity_n]
            random_actions = action_low + torch.rand(
                sensitivity_n, sensitivity_k, agent.act_dim, device=device
            ) * (action_high - action_low)
            repeated_summary = sensitivity_summary.repeat_interleave(sensitivity_k, dim=0)
            flat_actions = random_actions.reshape(sensitivity_n * sensitivity_k, agent.act_dim)
            sensitivity_next_summary, sensitivity_reward_logits, _ = agent.dynamics_step_from_history(
                repeated_summary.unsqueeze(1),
                flat_actions.unsqueeze(1),
            )
            sensitivity_rewards = reward_support.to_scalar(sensitivity_reward_logits).reshape(sensitivity_n, sensitivity_k)
            sensitivity_latents = sensitivity_next_summary[:, :NUM_LATENT_TOKENS].reshape(sensitivity_n, sensitivity_k, -1)
            metrics.update(
                reward_action_sensitivity_std=sensitivity_rewards.std(dim=1, unbiased=False).mean().item(),
                reward_action_sensitivity_range=(
                    sensitivity_rewards.max(dim=1).values - sensitivity_rewards.min(dim=1).values
                ).mean().item(),
                latent_action_sensitivity_std=sensitivity_latents.std(dim=1, unbiased=False).norm(dim=-1).mean().item(),
                latent_action_sensitivity_range=(
                    sensitivity_latents.max(dim=1).values - sensitivity_latents.min(dim=1).values
                ).norm(dim=-1).mean().item(),
            )
            return metrics
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            agent.train(was_training)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    transition_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    agent_latents = torch.zeros((args.num_steps, args.num_envs, NUM_LATENT_TOKENS, MODEL_DIM)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    imagined_steps = 0
    imagined_learnable_steps = 0
    real_minibatch_step = 0
    imagination_minibatch_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                rollout_latents = agent.get_agent_latents(next_obs)
                action, action_z, logprob, _, value = agent.get_action_and_value_from_latents(
                    rollout_latents,
                    hl_support,
                    sum_logprob=False,
                )
                values[step] = value.flatten()
                agent_latents[step] = rollout_latents.detach()
            actions[step] = action
            action_zs[step] = action_z
            env_action = torch.clamp(action, action_low, action_high)
            transition_actions[step] = env_action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(env_action.cpu().numpy())
            transition_termination = terminations
            transition_boundary = np.logical_or(terminations, truncations)
            transition_next_obs = np.array(next_obs, copy=True)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
            next_done = transition_boundary
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            transition_terminations[step] = torch.tensor(transition_termination, device=device, dtype=torch.float32)
            transition_boundaries[step] = torch.tensor(transition_boundary, device=device, dtype=torch.float32)
            transition_valids[step] = torch.tensor(transition_valid, device=device)
            next_obses[step] = torch.tensor(transition_next_obs, device=device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_transition_values = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                hl_support,
            ).reshape(args.num_steps, args.num_envs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_zs = action_zs.reshape((-1,) + envs.single_action_space.shape)
        b_agent_latents = agent_latents.reshape(-1, NUM_LATENT_TOKENS, MODEL_DIM)
        b_transition_actions = transition_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_transition_terminations = transition_terminations.reshape(-1)
        b_transition_boundaries = transition_boundaries.reshape(-1)
        b_transition_valids = transition_valids.reshape(-1)
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)

        world_model_only = global_step < args.wm_warmup_steps

        wm_b_inds = np.arange(args.batch_size)
        dyn_losses = []
        dyn_latent_losses = []
        dyn_reward_losses = []
        dyn_termination_losses = []
        pred_reward_decode_losses = []
        pred_termination_decode_losses = []
        dyn_sigreg_losses = []
        lejepa_losses = []
        dyn_reward_mses = []
        dyn_termination_accs = []
        lejepa_pred_mses = []
        lejepa_obs_pred_mses = []
        lejepa_outcome_pred_mses = []

        for epoch in range(args.wm_update_epochs):
            np.random.shuffle(wm_b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = wm_b_inds[start:end]

                mb_size = len(mb_inds)
                mb_step_inds = torch.as_tensor(mb_inds // args.num_envs, device=device, dtype=torch.long)
                mb_env_inds = torch.as_tensor(mb_inds % args.num_envs, device=device, dtype=torch.long)
                horizon_offsets = torch.arange(args.dyn_horizon, device=device)
                future_step_inds = mb_step_inds[:, None] + horizon_offsets[None, :]
                in_rollout = (future_step_inds < args.num_steps).float()
                safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                env_inds = mb_env_inds[:, None].expand_as(safe_step_inds)

                future_actions = transition_actions[safe_step_inds, env_inds]
                future_rewards = rewards[safe_step_inds, env_inds]
                future_terminations = transition_terminations[safe_step_inds, env_inds]
                future_boundaries = transition_boundaries[safe_step_inds, env_inds]
                future_valids = transition_valids[safe_step_inds, env_inds]
                future_next_obs = next_obses[safe_step_inds, env_inds]

                initial_summary = agent.get_summary_targets(b_obs[mb_inds])
                reward_target_probs = reward_support.project(future_rewards.reshape(-1)).reshape(
                    mb_size * args.dyn_horizon,
                    -1,
                )
                future_continues = (1.0 - future_terminations).reshape(-1)
                target_future_summaries = agent.encode_target_summary(
                    future_next_obs.reshape((-1,) + envs.single_observation_space.shape),
                    reward_target_probs,
                    future_continues,
                ).reshape(mb_size, args.dyn_horizon, NUM_LATENT_TOKENS, MODEL_DIM)
                sigreg_window_latents = torch.cat([initial_summary.unsqueeze(1), target_future_summaries], dim=1)
                teacher_window_latents = torch.cat(
                    [initial_summary.unsqueeze(1), target_future_summaries.detach()],
                    dim=1,
                )
                latent_history = teacher_window_latents[:, :-1]
                target_next_latents = target_future_summaries

                pred_next_latents, pred_reward_logits, pred_termination_logits = agent.dynamics_teacher_forced(
                    latent_history,
                    future_actions,
                )
                target_reward_logits, target_termination_logits = agent.decode_outcomes(
                    target_next_latents.reshape(mb_size * args.dyn_horizon, NUM_LATENT_TOKENS, MODEL_DIM),
                )
                target_reward_logits = target_reward_logits.reshape(mb_size, args.dyn_horizon, -1)
                target_termination_logits = target_termination_logits.reshape(mb_size, args.dyn_horizon)

                prev_continues = torch.cat(
                    [
                        torch.ones(mb_size, 1, device=device),
                        1.0 - future_boundaries[:, :-1],
                    ],
                    dim=1,
                )
                step_weight = torch.cumprod(prev_continues, dim=1) * in_rollout
                latent_weight = step_weight * future_valids

                sigreg_latents = [
                    sigreg_window_latents[:, horizon_idx].reshape(mb_size, -1)
                    for horizon_idx in range(args.dyn_horizon + 1)
                ]
                sigreg_valids = [torch.ones(mb_size, device=device, dtype=torch.bool)]
                sigreg_valids.extend([latent_weight[:, horizon_idx] > 0.0 for horizon_idx in range(args.dyn_horizon)])

                per_step_latent_loss = F.mse_loss(
                    pred_next_latents,
                    target_next_latents,
                    reduction="none",
                ).mean(dim=(-1, -2))
                dyn_latent_loss = (
                    per_step_latent_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                reward_target_probs = reward_target_probs.reshape(
                    mb_size,
                    args.dyn_horizon,
                    -1,
                )
                target_reward_decode_loss = -(
                    reward_target_probs.detach() * torch.log_softmax(target_reward_logits, dim=-1)
                ).sum(dim=-1)
                pred_reward_decode_loss = -(
                    reward_target_probs.detach() * torch.log_softmax(pred_reward_logits, dim=-1)
                ).sum(dim=-1)
                dyn_target_reward_loss = (
                    target_reward_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                dyn_pred_reward_loss = (
                    pred_reward_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                dyn_reward_loss = 0.5 * (dyn_target_reward_loss + dyn_pred_reward_loss)
                target_termination_decode_loss = F.binary_cross_entropy_with_logits(
                    target_termination_logits,
                    future_terminations,
                    reduction="none",
                )
                pred_termination_decode_loss = F.binary_cross_entropy_with_logits(
                    pred_termination_logits,
                    future_terminations,
                    reduction="none",
                )
                dyn_target_termination_loss = (
                    target_termination_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                dyn_pred_termination_loss = (
                    pred_termination_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                dyn_termination_loss = 0.5 * (dyn_target_termination_loss + dyn_pred_termination_loss)
                pred_reward_decode_losses.append(dyn_pred_reward_loss.item())
                pred_termination_decode_losses.append(dyn_pred_termination_loss.item())
                with torch.no_grad():
                    reward_pred = reward_support.to_scalar(pred_reward_logits)
                    termination_pred = (torch.sigmoid(pred_termination_logits) >= 0.5).float()
                    for horizon_idx in range(args.dyn_horizon):
                        horizon_weight = step_weight[:, horizon_idx]
                        denom = horizon_weight.sum().clamp_min(1.0)
                        reward_mse = (
                            (reward_pred[:, horizon_idx] - future_rewards[:, horizon_idx]).square()
                            * horizon_weight
                        ).sum() / denom
                        termination_acc = (
                            (termination_pred[:, horizon_idx] == future_terminations[:, horizon_idx]).float()
                            * horizon_weight
                        ).sum() / denom
                        dyn_reward_mses.append(reward_mse.item())
                        dyn_termination_accs.append(termination_acc.item())
                    obs_pred_mse = (
                        F.mse_loss(
                            pred_next_latents[:, :, :NUM_OBS_TOKENS],
                            target_next_latents[:, :, :NUM_OBS_TOKENS].detach(),
                            reduction="none",
                        ).mean(dim=(-1, -2))
                        * latent_weight
                    ).sum() / latent_weight.sum().clamp_min(1.0)
                    outcome_pred_mse = (
                        F.mse_loss(
                            pred_next_latents[:, :, NUM_OBS_TOKENS:],
                            target_next_latents[:, :, NUM_OBS_TOKENS:].detach(),
                            reduction="none",
                        ).mean(dim=(-1, -2))
                        * latent_weight
                    ).sum() / latent_weight.sum().clamp_min(1.0)
                    lejepa_obs_pred_mses.append(obs_pred_mse.item())
                    lejepa_outcome_pred_mses.append(outcome_pred_mse.item())
                dyn_sigreg_loss = masked_time_major_sigreg(sigreg_latents, sigreg_valids)
                lejepa_loss = args.dyn_latent_coef * dyn_latent_loss + args.sigreg_coef * dyn_sigreg_loss
                outcome_inverse_loss = (
                    args.dyn_reward_coef * dyn_reward_loss
                    + args.dyn_termination_coef * dyn_termination_loss
                )
                wm_loss = lejepa_loss + outcome_inverse_loss

                optimizer.zero_grad()
                wm_loss.backward()
                clip_grad_groups()
                optimizer.step()

                dyn_losses.append(wm_loss.item())
                dyn_latent_losses.append(dyn_latent_loss.item())
                dyn_reward_losses.append(dyn_reward_loss.item())
                dyn_termination_losses.append(dyn_termination_loss.item())
                dyn_sigreg_losses.append(dyn_sigreg_loss.item())
                lejepa_losses.append(lejepa_loss.item())
                lejepa_pred_mses.append(dyn_latent_loss.item())

        dyn_diagnostics = dynamics_diagnostics(
            b_obs,
            rewards,
            transition_actions,
            transition_terminations,
            transition_boundaries,
            transition_valids,
        )

        prompt_summary_history = None
        prompt_action_history = None
        prompt_valids = None
        use_imagination_updates = args.imagine_actor_coef != 0.0 or args.imagine_critic_coef != 0.0
        if (not world_model_only) and use_imagination_updates and global_step >= args.imagine_start_step:
            with torch.no_grad():
                prompt_summary_history, prompt_action_history, prompt_valids = build_dream_prompt_context(
                    b_obs,
                    transition_actions,
                    transition_boundaries,
                )

        # Agent phase on a frozen world-model interface. Real rollouts provide
        # the PPO anchor; imagined rollouts add model-based updates.
        b_inds = np.arange(args.batch_size)
        dream_inds = np.arange(args.batch_size * args.imagine_horizon) if prompt_summary_history is not None else None
        dream_minibatch_size = args.minibatch_size * args.imagine_horizon if prompt_summary_history is not None else None
        clipfracs = []
        action_clipfracs = []
        real_approx_kls = []
        real_cleanrl_approx_kls = []
        real_logratio_abs_means = []
        real_logratio_max_abses = []
        real_action_logratio_abs_means = []
        real_actor_stds = []
        dream_clipfracs = []
        imagine_actor_losses = []
        imagine_actor_returns = []
        imagine_critic_losses = []
        imagine_old_approx_kls = []
        imagine_approx_kls = []
        imagine_cleanrl_approx_kls = []
        imagine_logratio_abs_means = []
        imagine_logratio_max_abses = []
        imagine_action_logratio_abs_means = []
        dream_action_clipfracs = []
        imagine_action_sat_fracs = []
        imagine_actor_mean_abs_means = []
        imagine_actor_mean_max_abses = []
        imagine_raw_actor_beta_head_abs_means = []
        imagine_raw_actor_beta_head_max_abses = []
        imagine_actor_stds = []
        real_spo_penalties = []
        imagine_spo_penalties = []
        pg_loss = None
        v_loss = None
        entropy_loss = None
        old_approx_kl = None
        approx_kl = None
        dream_diagnostics = {}
        dream_diagnostic_values = {}
        imagine_explained_var = None
        imagine_explained_vars = []

        if not world_model_only:
            for epoch in range(args.agent_update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    with torch.no_grad():
                        writer.add_scalar(
                            "charts/real_return",
                            b_returns[mb_inds].mean().item(),
                            real_minibatch_step,
                        )
                        real_minibatch_step += 1

                    _, _, newlogprob, entropy = agent.get_action_logprob_entropy_from_latents(
                        b_agent_latents[mb_inds],
                        b_actions[mb_inds],
                        b_action_zs[mb_inds],
                        sum_logprob=False,
                    )
                    action_logratio = newlogprob - b_logprobs[mb_inds]
                    action_ratio = action_logratio.exp()
                    logratio = action_logratio.sum(1)
                    ratio = logratio.exp()
                    action_approx_kl = ((action_ratio - 1) - action_logratio).sum(1)

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = action_approx_kl.mean()
                        cleanrl_approx_kl = ((ratio - 1) - logratio).mean()
                        real_approx_kls.append(approx_kl.item())
                        real_cleanrl_approx_kls.append(cleanrl_approx_kl.item())
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        action_clipfracs += [
                            ((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]
                        real_logratio_abs_means.append(logratio.abs().mean().item())
                        real_logratio_max_abses.append(logratio.abs().max().item())
                        real_action_logratio_abs_means.append(action_logratio.abs().mean().item())
                        real_actor_stds.append(
                            agent.get_imagined_action_std(b_agent_latents[mb_inds]).mean().item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std(unbiased=False) + 1e-8
                        )

                    ratio_diff = ratio - 1.0
                    spo_eps = torch.where(
                        (mb_advantages * ratio_diff) > 0,
                        torch.full_like(mb_advantages, args.spo_eps_high),
                        torch.full_like(mb_advantages, args.spo_eps_low),
                    )
                    spo_penalty = mb_advantages.abs() * ratio_diff.square() / (2.0 * spo_eps)
                    pg_loss = -(mb_advantages * ratio - spo_penalty).mean()
                    real_spo_penalties.append(spo_penalty.mean().item())

                    value_logits = agent.get_value_logits_from_latents(b_agent_latents[mb_inds])
                    return_probs = hl_support.project(b_returns[mb_inds])
                    v_loss = -(
                        return_probs.detach() * torch.log_softmax(value_logits, dim=-1)
                    ).sum(dim=-1).mean()
                    entropy_loss = entropy.sum(1).mean()

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_groups()
                    optimizer.step()

        if prompt_summary_history is not None:
            dream_batch = build_dream_batch_eval(prompt_summary_history, prompt_action_history, prompt_valids)
            imagined_steps += args.batch_size * args.imagine_horizon
            imagined_learnable_steps += int(dream_batch[7].sum().item())
            (
                dream_states,
                dream_actions,
                dream_action_zs,
                dream_old_logprobs,
                dream_values,
                dream_advantages,
                dream_returns,
                dream_learn_masks,
                dream_learn_weights,
                dream_diagnostics,
            ) = dream_batch
            for key, value in dream_diagnostics.items():
                dream_diagnostic_values.setdefault(key, []).append(value)
            for epoch in range(args.imagine_update_epochs):
                np.random.shuffle(dream_inds)
                for start in range(0, dream_states.shape[0], dream_minibatch_size):
                    end = start + dream_minibatch_size
                    mb_inds = dream_inds[start:end]
                    mb_inds_t = torch.as_tensor(mb_inds, dtype=torch.long)
                    mb_dream_states = dream_states[mb_inds_t].to(device)
                    mb_dream_actions = dream_actions[mb_inds_t].to(device)
                    mb_dream_action_zs = dream_action_zs[mb_inds_t].to(device)
                    mb_dream_old_logprobs = dream_old_logprobs[mb_inds_t].to(device)
                    mb_dream_advantages = dream_advantages[mb_inds_t].to(device)
                    mb_dream_returns = dream_returns[mb_inds_t].to(device)
                    mb_dream_learn_mask = dream_learn_masks[mb_inds_t].to(device)
                    mb_dream_learn_weights = dream_learn_weights[mb_inds_t].to(device)
                    has_dream_targets = bool(mb_dream_learn_mask.any())
                    if not has_dream_targets:
                        continue
                    valid_weights = mb_dream_learn_weights[mb_dream_learn_mask]

                    with torch.no_grad():
                        writer.add_scalar(
                            "charts/imagination_return",
                            weighted_mean(
                                mb_dream_returns[mb_dream_learn_mask],
                                valid_weights,
                            ).item(),
                            imagination_minibatch_step,
                        )
                        imagination_minibatch_step += 1

                    _, _, dream_action_newlogprob, dream_entropy, _ = agent.get_imagined_action_and_value(
                        mb_dream_states,
                        hl_support,
                        mb_dream_actions,
                        mb_dream_action_zs,
                    )
                    dream_action_logratio = dream_action_newlogprob - mb_dream_old_logprobs
                    dream_action_ratio = dream_action_logratio.exp()
                    dream_logratio = dream_action_logratio.sum(1)
                    dream_ratio = dream_logratio.exp()
                    dream_action_approx_kl = ((dream_action_ratio - 1) - dream_action_logratio).sum(1)

                    with torch.no_grad():
                        valid_dream_action_logratio = dream_action_logratio[mb_dream_learn_mask]
                        valid_dream_action_ratio = dream_action_ratio[mb_dream_learn_mask]
                        valid_dream_logratio = dream_logratio[mb_dream_learn_mask]
                        valid_dream_ratio = dream_ratio[mb_dream_learn_mask]
                        dream_old_approx_kl = (-valid_dream_logratio).mean()
                        dream_approx_kl = dream_action_approx_kl[mb_dream_learn_mask].mean()
                        dream_cleanrl_approx_kl = ((valid_dream_ratio - 1) - valid_dream_logratio).mean()
                        imagine_cleanrl_approx_kls.append(dream_cleanrl_approx_kl.item())
                        dream_clipfracs += [
                            ((valid_dream_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]
                        dream_action_clipfracs += [
                            ((valid_dream_action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]
                        imagine_logratio_abs_means.append(valid_dream_logratio.abs().mean().item())
                        imagine_logratio_max_abses.append(valid_dream_logratio.abs().max().item())
                        imagine_action_logratio_abs_means.append(valid_dream_action_logratio.abs().mean().item())
                        valid_actions = mb_dream_actions[mb_dream_learn_mask]
                        imagine_action_sat_fracs.append((valid_actions.abs() > 0.98).float().mean().item())
                        valid_dream_states = mb_dream_states[mb_dream_learn_mask]
                        dream_action_mean = agent.get_imagined_action_mean(valid_dream_states)
                        raw_dream_beta_head = agent.get_imagined_raw_action_mean(valid_dream_states)
                        imagine_actor_mean_abs_means.append(dream_action_mean.abs().mean().item())
                        imagine_actor_mean_max_abses.append(dream_action_mean.abs().max().item())
                        imagine_raw_actor_beta_head_abs_means.append(raw_dream_beta_head.abs().mean().item())
                        imagine_raw_actor_beta_head_max_abses.append(raw_dream_beta_head.abs().max().item())
                        imagine_actor_stds.append(agent.get_imagined_action_std(valid_dream_states).mean().item())

                    if args.norm_adv:
                        valid_advantages = mb_dream_advantages[mb_dream_learn_mask]
                        advantage_mean = weighted_mean(valid_advantages, valid_weights)
                        advantage_var = weighted_mean((valid_advantages - advantage_mean).square(), valid_weights)
                        mb_dream_advantages = (mb_dream_advantages - advantage_mean) / (
                            advantage_var.sqrt() + 1e-8
                        )

                    dream_ratio_diff = dream_ratio - 1.0
                    dream_spo_eps = torch.where(
                        (mb_dream_advantages * dream_ratio_diff) > 0,
                        torch.full_like(mb_dream_advantages, args.spo_eps_high),
                        torch.full_like(mb_dream_advantages, args.spo_eps_low),
                    )
                    dream_spo_penalty = (
                        mb_dream_advantages.abs() * dream_ratio_diff.square() / (2.0 * dream_spo_eps)
                    )
                    imagine_policy_loss = -(mb_dream_advantages * dream_ratio - dream_spo_penalty)
                    imagine_actor_loss = imagine_policy_loss - args.imagine_actor_ent_coef * dream_entropy.sum(1)
                    imagine_actor_loss = weighted_mean(
                        imagine_actor_loss[mb_dream_learn_mask],
                        valid_weights,
                    )
                    imagine_spo_penalties.append(
                        weighted_mean(dream_spo_penalty[mb_dream_learn_mask], valid_weights).item()
                    )

                    dream_value_logits = agent.get_imagined_value_logits(mb_dream_states)
                    dream_return_probs = hl_support.project(mb_dream_returns)
                    imagine_value_loss = -(
                        dream_return_probs.detach() * torch.log_softmax(dream_value_logits, dim=-1)
                    ).sum(dim=-1)
                    imagine_critic_loss = weighted_mean(
                        imagine_value_loss[mb_dream_learn_mask],
                        valid_weights,
                    )

                    imagine_loss = (
                        args.imagine_actor_coef * imagine_actor_loss
                        + args.imagine_critic_coef * imagine_critic_loss
                    )

                    optimizer.zero_grad()
                    imagine_loss.backward()
                    clip_grad_groups()
                    optimizer.step()

                    imagine_actor_losses.append(imagine_actor_loss.item())
                    imagine_actor_returns.append(
                        weighted_mean(mb_dream_returns[mb_dream_learn_mask], valid_weights).item()
                    )
                    imagine_old_approx_kls.append(dream_old_approx_kl.item())
                    imagine_approx_kls.append(dream_approx_kl.item())
                    imagine_critic_losses.append(imagine_critic_loss.item())

                with torch.no_grad():
                    valid_inds = torch.nonzero(dream_learn_masks, as_tuple=False).flatten()
                    valid_returns = dream_returns[valid_inds].float()
                    if valid_returns.numel() > 1:
                        value_chunks = []
                        for value_start in range(0, valid_inds.numel(), dream_minibatch_size):
                            value_end = value_start + dream_minibatch_size
                            value_inds = valid_inds[value_start:value_end]
                            value_states = dream_states[value_inds].to(device)
                            value_chunks.append(agent.get_imagined_value(value_states, hl_support).detach().cpu())
                        valid_values = torch.cat(value_chunks, dim=0).float()
                        var_returns = torch.var(valid_returns, unbiased=False)
                        if var_returns.item() == 0.0:
                            imagine_explained_vars.append(np.nan)
                        else:
                            imagine_explained_vars.append(
                                (
                                    1 - torch.var(valid_returns - valid_values, unbiased=False) / var_returns
                                ).item()
                            )
            if imagine_explained_vars:
                imagine_explained_var = float(np.nanmean(imagine_explained_vars))

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/world_model_only", float(world_model_only), global_step)
        if v_loss is not None:
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            if real_approx_kls:
                writer.add_scalar("losses/approx_kl_update_mean", np.mean(real_approx_kls), global_step)
                writer.add_scalar("losses/approx_kl_update_max", np.max(real_approx_kls), global_step)
            if real_cleanrl_approx_kls:
                writer.add_scalar("losses/cleanrl_approx_kl", real_cleanrl_approx_kls[-1], global_step)
                writer.add_scalar(
                    "losses/cleanrl_approx_kl_update_mean", np.mean(real_cleanrl_approx_kls), global_step
                )
                writer.add_scalar(
                    "losses/cleanrl_approx_kl_update_max", np.max(real_cleanrl_approx_kls), global_step
                )
            if action_clipfracs:
                writer.add_scalar("losses/action_clipfrac", np.mean(action_clipfracs), global_step)
            if real_logratio_abs_means:
                writer.add_scalar("real_rollout/logratio_abs_mean", np.mean(real_logratio_abs_means), global_step)
                writer.add_scalar("real_rollout/logratio_max_abs", np.mean(real_logratio_max_abses), global_step)
                writer.add_scalar(
                    "real_rollout/action_logratio_abs_mean", np.mean(real_action_logratio_abs_means), global_step
                )
                writer.add_scalar("real_rollout/actor_std_mean", np.mean(real_actor_stds), global_step)
            if real_spo_penalties:
                writer.add_scalar("losses/spo_penalty", np.mean(real_spo_penalties), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        dyn_loss_mean = safe_mean(dyn_losses)
        dyn_latent_loss_mean = safe_mean(dyn_latent_losses)
        dyn_reward_loss_mean = safe_mean(dyn_reward_losses)
        dyn_termination_loss_mean = safe_mean(dyn_termination_losses)
        pred_reward_decode_loss_mean = safe_mean(pred_reward_decode_losses)
        pred_termination_decode_loss_mean = safe_mean(pred_termination_decode_losses)
        dyn_sigreg_loss_mean = safe_mean(dyn_sigreg_losses)
        lejepa_loss_mean = safe_mean(lejepa_losses)
        writer.add_scalar("losses/dyn_loss", dyn_loss_mean, global_step)
        writer.add_scalar("losses/lejepa_loss", lejepa_loss_mean, global_step)
        writer.add_scalar("losses/dyn_latent_loss", dyn_latent_loss_mean, global_step)
        writer.add_scalar("losses/outcome_inverse_reward_loss", dyn_reward_loss_mean, global_step)
        writer.add_scalar("losses/outcome_inverse_termination_loss", dyn_termination_loss_mean, global_step)
        writer.add_scalar("losses/dyn_sigreg_loss", dyn_sigreg_loss_mean, global_step)
        writer.add_scalar("lejepa/loss", lejepa_loss_mean, global_step)
        writer.add_scalar("lejepa/prediction_loss", dyn_latent_loss_mean, global_step)
        writer.add_scalar("lejepa/prediction_mse", safe_mean(lejepa_pred_mses), global_step)
        writer.add_scalar("lejepa/obs_prediction_mse", safe_mean(lejepa_obs_pred_mses), global_step)
        writer.add_scalar("lejepa/outcome_prediction_mse", safe_mean(lejepa_outcome_pred_mses), global_step)
        writer.add_scalar("lejepa/sigreg_loss", dyn_sigreg_loss_mean, global_step)
        writer.add_scalar("dynamics/loss", dyn_loss_mean, global_step)
        writer.add_scalar("dynamics/target_reward_inverse_loss", dyn_reward_loss_mean, global_step)
        writer.add_scalar("dynamics/pred_reward_loss", pred_reward_decode_loss_mean, global_step)
        writer.add_scalar("dynamics/reward_mse", safe_mean(dyn_reward_mses), global_step)
        writer.add_scalar("dynamics/target_termination_inverse_loss", dyn_termination_loss_mean, global_step)
        writer.add_scalar("dynamics/pred_termination_loss", pred_termination_decode_loss_mean, global_step)
        writer.add_scalar("dynamics/termination_accuracy", safe_mean(dyn_termination_accs), global_step)
        for key, value in dyn_diagnostics.items():
            writer.add_scalar(f"dynamics/{key}", value, global_step)
        if imagine_critic_losses:
            writer.add_scalar("losses/imagine_critic_loss", np.mean(imagine_critic_losses), global_step)
        if imagine_actor_losses:
            writer.add_scalar("losses/imagine_actor_loss", np.mean(imagine_actor_losses), global_step)
        if imagine_spo_penalties:
            writer.add_scalar("losses/imagine_spo_penalty", np.mean(imagine_spo_penalties), global_step)
        if imagine_actor_returns:
            writer.add_scalar("imagination/returns", np.mean(imagine_actor_returns), global_step)
        if imagine_explained_var is not None:
            writer.add_scalar("losses/imagine_explained_variance", imagine_explained_var, global_step)
        writer.add_scalar("losses/real_rollout_explained_variance", explained_var, global_step)
        for key, diagnostic_values in dream_diagnostic_values.items():
            writer.add_scalar(f"imagination/{key}", safe_mean(diagnostic_values), global_step)
        if imagine_approx_kls:
            writer.add_scalar("losses/imagine_old_approx_kl", np.mean(imagine_old_approx_kls), global_step)
            writer.add_scalar("losses/imagine_approx_kl", np.mean(imagine_approx_kls), global_step)
            writer.add_scalar("losses/imagine_clipfrac", np.mean(dream_clipfracs), global_step)
        if imagine_cleanrl_approx_kls:
            writer.add_scalar("losses/imagine_cleanrl_approx_kl", np.mean(imagine_cleanrl_approx_kls), global_step)
            writer.add_scalar("losses/imagine_action_clipfrac", np.mean(dream_action_clipfracs), global_step)
        if imagine_logratio_abs_means:
            writer.add_scalar("imagination/logratio_abs_mean", np.mean(imagine_logratio_abs_means), global_step)
            writer.add_scalar("imagination/logratio_max_abs", np.mean(imagine_logratio_max_abses), global_step)
            writer.add_scalar(
                "imagination/action_logratio_abs_mean", np.mean(imagine_action_logratio_abs_means), global_step
            )
            writer.add_scalar("imagination/action_saturation_frac", np.mean(imagine_action_sat_fracs), global_step)
            writer.add_scalar("imagination/actor_mean_abs_mean", np.mean(imagine_actor_mean_abs_means), global_step)
            writer.add_scalar("imagination/actor_mean_max_abs", np.mean(imagine_actor_mean_max_abses), global_step)
            writer.add_scalar(
                "imagination/raw_actor_beta_head_abs_mean",
                np.mean(imagine_raw_actor_beta_head_abs_means),
                global_step,
            )
            writer.add_scalar(
                "imagination/raw_actor_beta_head_max_abs",
                np.mean(imagine_raw_actor_beta_head_max_abses),
                global_step,
            )
            writer.add_scalar("imagination/actor_std_mean", np.mean(imagine_actor_stds), global_step)
        writer.add_scalar("charts/imagined_steps", imagined_steps, global_step)
        writer.add_scalar("charts/imagined_learnable_steps", imagined_learnable_steps, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts_total/SPS", int((global_step + imagined_steps) / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
