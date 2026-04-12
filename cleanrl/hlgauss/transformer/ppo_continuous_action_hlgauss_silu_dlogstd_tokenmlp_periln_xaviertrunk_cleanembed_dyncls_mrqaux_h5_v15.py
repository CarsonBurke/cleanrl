# PPO + HL-Gauss with an LLM-style shared self-attention backbone.
#
# Key ideas:
# - one learned token per observation dimension via a small scalar MLP embedder
# - four learned CLS tokens: actor, critic, exploration/log-std, and dynamics
# - 2-layer shared transformer with RoPE, QK-norm, RMSNorm Peri-LN, and SwiGLU
# - Xavier/Glorot init on tokenizer and transformer layers
# - MR.Q-style auxiliary dynamics head from a dedicated dynamics CLS, conditioned on the clipped env action
# - 5-step autoregressive dynamics unroll masked across episode boundaries
# - no extra CLS/head norm; rely on RMSNorm Peri-LN blocks
import copy
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cleanrl.shared.hl_gauss import HLGaussSupport


MODEL_DIM = 64
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
FFN_MULT = 2
NUM_LAYERS = 2
NUM_SPECIAL_TOKENS = 4
SCALAR_EMBED_DIM = 32


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
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # HL-Gauss specific
    num_bins: int = 51
    """number of bins for the categorical value head"""
    v_min: float = -5.0
    """minimum value of the support (in symlog space)"""
    v_max: float = 5.0
    """maximum value of the support (in symlog space)"""
    sigma_ratio: float = 0.5
    """sigma / bin_width ratio for HL-Gauss target smoothing"""

    # Dynamics auxiliary (MR.Q-inspired)
    dyn_coef: float = 0.1
    """overall scale on the dynamics auxiliary loss"""
    dyn_horizon: int = 5
    """autoregressive dynamics horizon"""
    dyn_latent_coef: float = 1.0
    """weight on next-dynamics-token prediction"""
    dyn_reward_coef: float = 0.25
    """weight on reward prediction from the dynamics token"""
    dyn_done_coef: float = 0.25
    """weight on boundary prediction from the dynamics token"""
    dyn_target_tau: float = 0.01
    """EMA update rate for the target encoder used in dynamics prediction"""
    reward_num_bins: int = 51
    """number of bins for the auxiliary reward head"""
    reward_v_min: float = -10.0
    """minimum reward support for the auxiliary reward head"""
    reward_v_max: float = 10.0
    """maximum reward support for the auxiliary reward head"""
    reward_sigma_ratio: float = 0.75
    """sigma / bin_width ratio for the auxiliary reward support"""

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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


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


def attention(q, k, v):
    if q.is_cuda:
        attn_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype), dropout_p=0.0
                    )
                return out.to(q.dtype)
            except RuntimeError:
                pass
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_mult=2):
        super().__init__()
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads

        self.attn_pre_norm = RMSNorm(dim)
        self.attn_post_norm = RMSNorm(dim)
        self.ffn_pre_norm = RMSNorm(dim)
        self.ffn_post_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

        for module in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2, self.w3]:
            xavier_init_linear(module)

    def forward(self, x, rope_cos, rope_sin):
        batch, seq_len, width = x.shape

        h = self.attn_pre_norm(x)
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

        attn_out = attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        x = x + self.attn_post_norm(self.wo(attn_out))

        h = self.ffn_pre_norm(x)
        x = x + self.ffn_post_norm(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x


class Agent(nn.Module):
    def __init__(self, envs, num_bins):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.act_dim = act_dim

        self.obs_in_proj = xavier_init_linear(nn.Linear(1, SCALAR_EMBED_DIM))
        self.obs_out_proj = xavier_init_linear(nn.Linear(SCALAR_EMBED_DIM, MODEL_DIM))

        self.obs_dim_embed = nn.Parameter(torch.empty(obs_dim, MODEL_DIM))
        nn.init.xavier_uniform_(self.obs_dim_embed)

        cls = torch.empty(NUM_SPECIAL_TOKENS, MODEL_DIM)
        nn.init.xavier_uniform_(cls)
        self.actor_cls = nn.Parameter(cls[0:1].unsqueeze(0))
        self.critic_cls = nn.Parameter(cls[1:2].unsqueeze(0))
        self.sde_cls = nn.Parameter(cls[2:3].unsqueeze(0))
        self.dyn_cls = nn.Parameter(cls[3:4].unsqueeze(0))

        self.embed_norm = RMSNorm(MODEL_DIM)
        self.layers = nn.ModuleList(
            [TransformerBlock(MODEL_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT) for _ in range(NUM_LAYERS)]
        )
        self.final_norm = RMSNorm(MODEL_DIM)

        head_dim = MODEL_DIM // NUM_Q_HEADS
        rope_cos, rope_sin = build_rope_cache(obs_dim, NUM_SPECIAL_TOKENS, head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.actor_mean_head = layer_init(nn.Linear(MODEL_DIM, act_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(MODEL_DIM, num_bins), std=1.0)
        self.sde_logstd_head = layer_init(nn.Linear(MODEL_DIM, act_dim), std=0.01)
        self.dyn_action_proj = xavier_init_linear(nn.Linear(act_dim, MODEL_DIM))
        self.dyn_joint_proj = xavier_init_linear(nn.Linear(MODEL_DIM * 2, MODEL_DIM))
        self.dyn_next_head = xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM))
        self.dyn_reward_head = xavier_init_linear(nn.Linear(MODEL_DIM, num_bins))
        self.dyn_done_head = xavier_init_linear(nn.Linear(MODEL_DIM, 1))

    def _encode(self, x):
        batch = x.shape[0]
        obs_tokens = self.obs_out_proj(F.silu(self.obs_in_proj(x.unsqueeze(-1))))
        obs_tokens = obs_tokens + self.obs_dim_embed.unsqueeze(0)
        special_tokens = torch.cat(
            [
                self.actor_cls.expand(batch, -1, -1),
                self.critic_cls.expand(batch, -1, -1),
                self.sde_cls.expand(batch, -1, -1),
                self.dyn_cls.expand(batch, -1, -1),
            ],
            dim=1,
        )
        tokens = torch.cat([special_tokens, obs_tokens], dim=1)
        tokens = self.embed_norm(tokens)

        for layer in self.layers:
            tokens = layer(tokens, self.rope_cos, self.rope_sin)

        tokens = self.final_norm(tokens)
        return tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]

    def _dynamics_latent(self, dyn_features, action):
        action_features = F.silu(self.dyn_action_proj(action))
        return F.silu(self.dyn_joint_proj(torch.cat([dyn_features, action_features], dim=-1)))

    def predict_dynamics(self, x, action):
        dyn_features = self.get_dyn_target(x)
        return self.dynamics_step(dyn_features, action)

    def dynamics_step(self, dyn_features, action):
        dyn_latent = self._dynamics_latent(dyn_features, action)
        pred_next_dyn = self.dyn_next_head(dyn_latent)
        pred_reward_logits = self.dyn_reward_head(dyn_latent)
        pred_done_logits = self.dyn_done_head(dyn_latent).squeeze(-1)
        return pred_next_dyn, pred_reward_logits, pred_done_logits

    def get_dyn_target(self, x):
        return self._encode(x)[3]

    def get_value_logits(self, x):
        _, critic_features, _, _ = self._encode(x)
        return self.critic_head(critic_features)

    def get_value(self, x, hl_support):
        return hl_support.to_scalar(self.get_value_logits(x))

    def get_action_and_value(self, x, hl_support, action=None):
        actor_features, critic_features, sde_features, _ = self._encode(x)
        action_mean = self.actor_mean_head(actor_features)
        action_logstd = torch.clamp(self.sde_logstd_head(sde_features), -5.0, 2.0)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = hl_support.to_scalar(self.critic_head(critic_features))
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


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
    target_agent = copy.deepcopy(agent).to(device)
    target_agent.eval()
    for param in target_agent.parameters():
        param.requires_grad_(False)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=True)
    reward_support = HLGaussSupport(
        args.reward_num_bins,
        args.reward_v_min,
        args.reward_v_max,
        args.reward_sigma_ratio,
        device,
        use_symlog=False,
    )
    action_low = torch.tensor(envs.single_action_space.low, device=device)
    action_high = torch.tensor(envs.single_action_space.high, device=device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    transition_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
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
                action, logprob, _, value = agent.get_action_and_value(next_obs, hl_support)
                values[step] = value.flatten()
            actions[step] = action
            env_action = torch.clamp(action, action_low, action_high)
            transition_actions[step] = env_action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(env_action.cpu().numpy())
            transition_done = terminations
            transition_boundary = np.logical_or(terminations, truncations)
            transition_next_obs = np.array(next_obs, copy=True)
            transition_valid = np.ones(args.num_envs, dtype=np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
            next_done = transition_boundary
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            transition_dones[step] = torch.tensor(transition_done, device=device, dtype=torch.float32)
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
        b_transition_actions = transition_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_transition_dones = transition_dones.reshape(-1)
        b_transition_boundaries = transition_boundaries.reshape(-1)
        b_transition_valids = transition_valids.reshape(-1)
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        dyn_losses = []
        dyn_latent_losses = []
        dyn_reward_losses = []
        dyn_done_losses = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], hl_support, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: HL-Gauss cross-entropy (no vclip — categorical doesn't support it cleanly)
                newvalue_logits = agent.get_value_logits(b_obs[mb_inds])
                target_probs = hl_support.project(b_returns[mb_inds])
                log_probs = torch.log_softmax(newvalue_logits, dim=-1)
                v_loss = -(target_probs * log_probs).sum(dim=-1).mean()

                dyn_features = agent.get_dyn_target(b_obs[mb_inds])
                mb_step_inds = torch.as_tensor(mb_inds // args.num_envs, device=device, dtype=torch.long)
                mb_env_inds = torch.as_tensor(mb_inds % args.num_envs, device=device, dtype=torch.long)
                dyn_alive = torch.ones(len(mb_inds), device=device)
                dyn_latent_loss = torch.zeros((), device=device)
                dyn_reward_loss = torch.zeros((), device=device)
                dyn_done_loss = torch.zeros((), device=device)
                for horizon_idx in range(args.dyn_horizon):
                    future_step_inds = mb_step_inds + horizon_idx
                    in_rollout = (future_step_inds < args.num_steps).float()
                    safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)

                    future_actions = transition_actions[safe_step_inds, mb_env_inds]
                    future_rewards = rewards[safe_step_inds, mb_env_inds]
                    future_dones = transition_dones[safe_step_inds, mb_env_inds]
                    future_boundaries = transition_boundaries[safe_step_inds, mb_env_inds]
                    future_valids = transition_valids[safe_step_inds, mb_env_inds]
                    future_next_obs = next_obses[safe_step_inds, mb_env_inds]

                    pred_next_dyn, pred_reward_logits, pred_done_logits = agent.dynamics_step(dyn_features, future_actions)
                    with torch.no_grad():
                        target_next_dyn = target_agent.get_dyn_target(future_next_obs)

                    step_weight = dyn_alive * in_rollout
                    latent_weight = step_weight * future_valids
                    dyn_latent_loss = dyn_latent_loss + (
                        F.mse_loss(pred_next_dyn, target_next_dyn, reduction="none").mean(dim=-1) * latent_weight
                    ).mean()
                    reward_target_probs = reward_support.project(future_rewards)
                    dyn_reward_loss = dyn_reward_loss + (
                        -(reward_target_probs * torch.log_softmax(pred_reward_logits, dim=-1)).sum(dim=-1) * step_weight
                    ).mean()
                    dyn_done_loss = dyn_done_loss + (
                        F.binary_cross_entropy_with_logits(pred_done_logits, future_dones, reduction="none") * step_weight
                    ).mean()

                    dyn_features = pred_next_dyn
                    dyn_alive = dyn_alive * (1.0 - future_boundaries)

                dyn_latent_loss = dyn_latent_loss / args.dyn_horizon
                dyn_reward_loss = dyn_reward_loss / args.dyn_horizon
                dyn_done_loss = dyn_done_loss / args.dyn_horizon
                dyn_loss = (
                    args.dyn_latent_coef * dyn_latent_loss
                    + args.dyn_reward_coef * dyn_reward_loss
                    + args.dyn_done_coef * dyn_done_loss
                )

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.dyn_coef * dyn_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                        target_param.lerp_(param, args.dyn_target_tau)

                dyn_losses.append(dyn_loss.item())
                dyn_latent_losses.append(dyn_latent_loss.item())
                dyn_reward_losses.append(dyn_reward_loss.item())
                dyn_done_losses.append(dyn_done_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

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
        writer.add_scalar("losses/dyn_loss", np.mean(dyn_losses), global_step)
        writer.add_scalar("losses/dyn_latent_loss", np.mean(dyn_latent_losses), global_step)
        writer.add_scalar("losses/dyn_reward_loss", np.mean(dyn_reward_losses), global_step)
        writer.add_scalar("losses/dyn_done_loss", np.mean(dyn_done_losses), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
