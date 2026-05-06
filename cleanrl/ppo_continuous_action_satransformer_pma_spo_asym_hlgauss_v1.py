# SPO-asym + shared SA→PMA transformer + HL-Gauss categorical critic.
#
# Sibling of satransformer_pma_spo_asym_v1; only the value head changes:
#   critic_head: Linear(D, num_bins)           # logits over discretized support
#   value loss: cross-entropy on HL-Gauss-projected returns (no vclip; PPO's
#                                                ratio-style vclip doesn't
#                                                fit a categorical head)
#   bootstrap : E[support · softmax(logits)]   (with optional symlog inversion)
#
# HL-Gauss support utility (cleanrl/shared/hl_gauss.py):
#   project: scalar return -> Gaussian-smoothed categorical over fixed bins
#   to_scalar: logits -> scalar via expectation under softmax
#   use_symlog=True maps real-valued returns into a compressed range so a
#   small fixed [v_min, v_max] window covers wide reward distributions.
#
# (header below preserved from base file)
# SPO-asym + shared SA→PMA transformer backbone, direct log-std, tanh-squashed.
#
# Same SPO-asymmetric trust region as ppo_continuous_action_pmpo_d4_beta_relusq_spo_asym_v1.py
# (winner of HC sweep at ε=0.40/0.56). Replaces the ReluSq-MLP Beta agent with:
#
#   tokenizer (per-dim affine)
#   embed RMSNorm
#   1 self-attention block over obs tokens (RoPE, QK-norm, SwiGLU FFN, pre-norm,
#                                            learnable scalar residual gates)
#   1 PMA cross-attention block with 3 learnable seeds (actor / sde / critic)
#                                  attending the SA output
#                                  (QK-norm, SwiGLU FFN, pre-norm, no RoPE,
#                                   learnable scalar residual gates)
#   final RMSNorm
#   linear-only heads from each seed (no MLP):
#       mean    = actor_head(actor_seed)
#       log_std = sde_head(sde_seed)         # direct log-std (separate seed)
#       value   = critic_head(critic_seed)
#
# Distribution: tanh-squashed Normal(mean, exp(t) + STD_MIN).
#   z      ~ Normal(mean, std)
#   action = tanh(z)                         (bounded to [-1, 1] = action space)
#   log_prob(z) = Normal.log_prob(z) - sum(log(1 - tanh(z)^2))
#                                            (Jacobian correction; numerically
#                                             stable form via softplus)
#
# Std parameterization: std = exp(t) + STD_MIN, where t is the raw sde-head
# output and STD_MIN = 1e-3 provides a hard lower bound on std (preventing
# log_prob → ∞ from a vanishing-variance delta). No upper clamp: tanh squash
# absorbs saturated samples naturally (sech² ≈ 0 in saturation), so unbounded
# growth in std doesn't blow up actions or gradients through the action.
#
# To avoid atanh(action) round-trip bias in the PPO ratio, the rollout buffer
# stores the pre-squash sample `z`; tanh(z) is sent to the env.
#
# SDE design vs Dreamer4 (dreamer4/dreamer4.py:398-404, 1090-1142):
#   - We use a separate seed (sde_seed); Dreamer4 uses one head producing a
#     joint (mean, log_var) tensor and unbinds dim=-1.
#   - We parameterize std as exp(t) + STD_MIN (lower-bounded, no upper clamp);
#     Dreamer4 parameterizes log_var (std = exp(log_var/2)) with no bounds.
#   - We tanh-squash the action; Dreamer4 emits unbounded Gaussian samples.
#   - We use dist.sample(); Dreamer4 manually reparameterizes
#     (mean + std * randn_like * temperature).
#
# bf16 + FlashAttention via torch.autocast on encoder forward. Heads run in
# fp32 for distribution / log-prob stability.
#
# Hypothesis. Per-dim obs tokens give the policy a natural inductive bias for
# proprioceptive state (each joint sensed independently then mixed by attention).
# Separating the std readout onto its own seed lets exploration adapt without
# stealing capacity from the action-mean computation, and keeps gradient
# flow through SDE distinct from gradient flow through the actor mean.

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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Make the cleanrl package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.hl_gauss import HLGaussSupport


MODEL_DIM = 64
NUM_HEADS = 4
FFN_MULT = 2
NUM_BLOCKS = 2  # one SA + one PMA — used for output-projection init scale

STD_MIN = 1e-3  # hard lower bound on std (prevents log_prob → ∞ at vanishing variance)
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
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Promote to fp32 for the reduction; safe under bf16 autocast.
        x_f = x.float()
        rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + self.eps)
        out = x_f / rms
        return (out * self.weight).to(x.dtype)


def build_rope_cache(num_positions, head_dim, device, base=10000.0):
    """RoPE cos/sin cache for `num_positions` positions, half-dim split."""
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(num_positions, device=device).float()
    freqs = torch.outer(positions, theta)  # (num_positions, head_dim//2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply RoPE to last-dim of `x`. cos/sin are (S, head_dim//2).
    Cast cos/sin to x.dtype so RoPE doesn't upcast bf16 q/k under autocast
    (which would break the FlashAttention path)."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def flash_attention(q, k, v):
    """SDPA call; PyTorch dispatches to FlashAttention when bf16/fp16 inputs
    + supported head_dim. Under bf16 autocast this hits the FA path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def init_linear_(module, init_scale=1.0):
    fan_in = module.weight.shape[1]
    std = init_scale * fan_in ** -0.5
    nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class SABlock(nn.Module):
    """Pre-norm SA + SwiGLU FFN over obs tokens. RoPE on Q/K. QK-norm.
    Learnable scalar gate per residual ('residual mixing')."""

    def __init__(self, dim, num_heads, ffn_mult, init_scale):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

        # residual mixing: learnable scalar gates, init=1.0
        self.attn_gate = nn.Parameter(torch.ones(()))
        self.ffn_gate = nn.Parameter(torch.ones(()))

        for module in (self.wq, self.wk, self.wv, self.w1, self.w3):
            init_linear_(module, init_scale=1.0)
        for module in (self.wo, self.w2):
            init_linear_(module, init_scale=init_scale)

    def forward(self, x, rope_cos, rope_sin):
        B, S, D = x.shape

        h = self.attn_norm(x)
        q = self.wq(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        attn_out = flash_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.attn_gate * self.wo(attn_out)

        h = self.ffn_norm(x)
        ffn_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + self.ffn_gate * ffn_out
        return x


class PMABlock(nn.Module):
    """PMA = pooling by multi-head cross-attention. Seeds (Q) attend
    obs tokens (K, V). Pre-norm on both Q and KV streams. QK-norm.
    No RoPE (seeds are positionless; obs token positional info is already
    baked in via the upstream SA block + per-dim tokenizer embeddings).
    SwiGLU FFN on the seed stream. Learnable scalar residual gates."""

    def __init__(self, dim, num_heads, ffn_mult, init_scale):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_pre_norm = RMSNorm(dim)
        self.kv_pre_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

        self.attn_gate = nn.Parameter(torch.ones(()))
        self.ffn_gate = nn.Parameter(torch.ones(()))

        for module in (self.wq, self.wk, self.wv, self.w1, self.w3):
            init_linear_(module, init_scale=1.0)
        for module in (self.wo, self.w2):
            init_linear_(module, init_scale=init_scale)

    def forward(self, seeds, kv_input):
        B, S_q, D = seeds.shape
        S_kv = kv_input.shape[1]

        h_q = self.q_pre_norm(seeds)
        h_kv = self.kv_pre_norm(kv_input)

        q = self.wq(h_q).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = flash_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, D)
        seeds = seeds + self.attn_gate * self.wo(attn_out)

        h = self.ffn_norm(seeds)
        ffn_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        seeds = seeds + self.ffn_gate * ffn_out
        return seeds


class Agent(nn.Module):
    def __init__(self, envs, num_bins):
        super().__init__()
        self.obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.action_dim = int(np.prod(envs.single_action_space.shape))
        self.num_bins = num_bins

        # Tokenizer: per-dim affine (each obs dim → its own token).
        embed_std = MODEL_DIM ** -0.5
        self.obs_embed_w = nn.Parameter(torch.empty(self.obs_dim, MODEL_DIM))
        nn.init.trunc_normal_(self.obs_embed_w, std=embed_std, a=-2 * embed_std, b=2 * embed_std)
        self.obs_embed_b = nn.Parameter(torch.zeros(self.obs_dim, MODEL_DIM))

        self.embed_norm = RMSNorm(MODEL_DIM)

        # 3 PMA seeds: actor (mean), sde (log_std), critic (value).
        seed_std = MODEL_DIM ** -0.5
        self.seeds = nn.Parameter(torch.randn(3, MODEL_DIM) * seed_std)

        init_scale = 1.0 / (2 * NUM_BLOCKS) ** 0.5  # 1/sqrt(4) = 0.5
        self.sa_block = SABlock(MODEL_DIM, NUM_HEADS, FFN_MULT, init_scale)
        self.pma_block = PMABlock(MODEL_DIM, NUM_HEADS, FFN_MULT, init_scale)
        self.final_norm = RMSNorm(MODEL_DIM)

        # Linear-only heads (no MLP).
        self.actor_mean_head = layer_init(nn.Linear(MODEL_DIM, self.action_dim), std=0.01)
        self.sde_logstd_head = layer_init(nn.Linear(MODEL_DIM, self.action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(MODEL_DIM, num_bins), std=1.0)

        # RoPE cache for SA over obs token positions [0, obs_dim).
        head_dim = MODEL_DIM // NUM_HEADS
        rope_cos, rope_sin = build_rope_cache(self.obs_dim, head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def _encode(self, x):
        """Tokenize → embed_norm → SA → PMA → final_norm. Returns the three
        seed features (actor, sde, critic) each shape (B, MODEL_DIM)."""
        B = x.shape[0]
        # (B, obs_dim) → (B, obs_dim, MODEL_DIM) via per-dim affine.
        obs_tokens = x.unsqueeze(-1) * self.obs_embed_w + self.obs_embed_b
        obs_tokens = self.embed_norm(obs_tokens)
        obs_tokens = self.sa_block(obs_tokens, self.rope_cos, self.rope_sin)

        seeds = self.seeds.unsqueeze(0).expand(B, -1, -1)  # (B, 3, D)
        seeds = self.pma_block(seeds, obs_tokens)
        seeds = self.final_norm(seeds)
        return seeds[:, 0], seeds[:, 1], seeds[:, 2]

    def _encoder_with_autocast(self, x):
        if x.is_cuda and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actor_feat, sde_feat, critic_feat = self._encode(x)
        else:
            actor_feat, sde_feat, critic_feat = self._encode(x)
        return actor_feat.float(), sde_feat.float(), critic_feat.float()

    def get_value_logits(self, x):
        _, _, critic_feat = self._encoder_with_autocast(x)
        return self.critic_head(critic_feat)

    def get_value(self, x, hl_support):
        return hl_support.to_scalar(self.get_value_logits(x))

    def get_action_distribution(self, x):
        actor_feat, sde_feat, _ = self._encoder_with_autocast(x)
        mean = self.actor_mean_head(actor_feat)
        std = self.sde_logstd_head(sde_feat).exp() + STD_MIN
        return Normal(mean, std), std

    def get_action_and_value(self, x, z=None):
        """If `z` is None: sample fresh z, return (env_action, z, log_prob, ...).
        Else: treat `z` as the pre-squash sample (replay) and recompute log_prob.

        Storing z (rather than tanh(z)) avoids the atanh round-trip bias when
        replaying; the squash Jacobian correction depends only on z, so it
        cancels in the PPO ratio anyway, but using exact z keeps the
        underlying Normal log-prob exact too."""
        actor_feat, sde_feat, critic_feat = self._encoder_with_autocast(x)
        mean = self.actor_mean_head(actor_feat)
        std = self.sde_logstd_head(sde_feat).exp() + STD_MIN
        value_logits = self.critic_head(critic_feat)  # (B, num_bins)
        dist = Normal(mean, std)

        if z is None:
            z = dist.sample()

        # log|det dy/dz| where y = tanh(z): sum_i log(1 - tanh(z_i)^2).
        # Numerically stable form: log(1 - tanh(z)^2) = 2*(log2 - z - softplus(-2z)).
        log_squash = (2.0 * (LOG_2 - z - F.softplus(-2.0 * z))).sum(-1)
        log_prob = dist.log_prob(z).sum(-1) - log_squash

        env_action = torch.tanh(z)
        entropy = dist.entropy().sum(-1)  # underlying-Normal entropy (squash not corrected)
        return env_action, z, log_prob, entropy, value_logits, std


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, num_bins):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs, num_bins).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            env_action, _, _, _, _, _ = agent.get_action_and_value(obs_tensor)
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

    agent = Agent(envs, args.num_bins).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=args.use_symlog,
    )

    # ALGO Logic: Storage setup
    # `actions` stores the pre-squash sample z (Normal noise), not tanh(z).
    # tanh(z) is what the env sees; z is what we replay through the policy.
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
                env_action, z, logprob, _, value_logits, _ = agent.get_action_and_value(next_obs)
                values[step] = hl_support.to_scalar(value_logits)
            actions[step] = z  # store pre-squash sample (used for replay)
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

                # b_actions[mb_inds] is the stored pre-squash z; pass as `z`.
                _, _, newlogprob, entropy, newvalue_logits, _ = agent.get_action_and_value(
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Distribution diagnostics (state-dependent std).
        with torch.no_grad():
            _, std_diag = agent.get_action_distribution(b_obs)
            mean_std = std_diag.mean().item()
            min_std = std_diag.min().item()
            max_std = std_diag.max().item()
            mean_logstd = std_diag.log().mean().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("diag/spo_penalty", spo_penalty_mean.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/mean_std", mean_std, global_step)
        writer.add_scalar("diag/min_std", min_std, global_step)
        writer.add_scalar("diag/max_std", max_std, global_step)
        writer.add_scalar("diag/mean_logstd", mean_logstd, global_step)
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
            num_bins=args.num_bins,
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
