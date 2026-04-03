# PPO with Hindsight Teacher — Bidirectional trajectory reasoning for actor training.
#
# Fork of ststs_shared_cls. Adds a "teacher" that sees complete trajectories
# with full hindsight (bidirectional attention) and produces per-step target
# logits (μ*, σ*) for the actor to distill from.
#
# Architecture:
#   Backbone (causal STSTS): [actor_cls, critic_cls, teacher_cls] → per-step latents
#   Teacher: bidirectional transformer over (teacher_latent_t, a_t, R_t) for t=1..T
#            → μ*_t, log_σ*_t at each timestep
#   Actor: trains via distillation from teacher targets (NO policy gradient)
#   Critic: standard value regression for computing returns
#
# Training loop per iteration:
#   1. Rollout with actor (collect obs, actions, rewards)
#   2. Compute returns/advantages with critic (standard GAE)
#   3. PPO update for critic only (value regression)
#   4. Teacher processes full trajectories bidirectionally → μ*, σ* targets
#   5. Actor trains: ||μ_actor - μ*||² + ||log_σ_actor - log_σ*||²
#   6. Teacher trains: return-weighted regression on same data
#   Both teacher and actor losses flow back through backbone (unfrozen).
#
# The teacher's advantage over the causal actor: it can see future outcomes
# and reason about which actions led to good/bad results. It amortizes
# cross-timestep credit assignment into per-step action targets.
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
NUM_CLS_TOKENS = 3  # actor, critic, teacher

# Teacher config
TEACHER_LAYERS = 2
TEACHER_INPUT_DIM = EMBED_DIM + 1  # teacher_cls + return-to-go (action is projected separately)
TEACHER_COEF = 1.0          # weight of teacher's advantage-weighted regression loss
DISTILL_COEF = 0.5          # weight of actor distillation loss
DISTILL_WARMUP_ITERS = 10   # iterations before distillation kicks in (let teacher learn first)
PG_COEF = 0.5               # PG weight after warmup (always >0 so actor_logstd keeps training)


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


class STSTSCLSBackbone(nn.Module):
    def __init__(self, obs_dim, context_len):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_len = context_len
        self.actor_cls_index = obs_dim
        self.critic_cls_index = obs_dim + 1
        self.teacher_cls_index = obs_dim + 2

        self.value_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.input_norm = RMSNorm(EMBED_DIM)
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        self.register_buffer("dim_indices", torch.arange(obs_dim))
        cls_std = 1.0 / EMBED_DIM**0.5
        self.actor_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.critic_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.teacher_cls = nn.Parameter(torch.empty(EMBED_DIM))
        nn.init.trunc_normal_(self.actor_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.critic_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.teacher_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

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

    def _spatial(self, tokens, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.reshape(batch * time_steps, slots, width)
        x = block(x)
        return x.reshape(batch, time_steps, slots, width)

    def _temporal(self, tokens, block):
        batch, time_steps, slots, width = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * slots, time_steps, width)
        x = block(x, rope_cos=self.temporal_cos, rope_sin=self.temporal_sin)
        x = x.reshape(batch, slots, time_steps, width).permute(0, 2, 1, 3)
        return x

    def forward(self, obs_seq):
        batch, time_steps, _ = obs_seq.shape
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1))
        obs_tokens = obs_tokens + self.dim_id_embed(self.dim_indices).view(1, 1, self.obs_dim, EMBED_DIM)

        cls_tokens = torch.stack([self.actor_cls, self.critic_cls, self.teacher_cls], dim=0)
        cls_tokens = cls_tokens.view(1, 1, NUM_CLS_TOKENS, EMBED_DIM).expand(batch, time_steps, -1, -1)
        tokens = torch.cat([obs_tokens, cls_tokens], dim=2)
        tokens = self.input_norm(tokens)

        tokens = self._spatial(tokens, self.s_blocks[0])
        tokens = self._temporal(tokens, self.t_blocks[0])
        tokens = self._spatial(tokens, self.s_blocks[1])
        tokens = self._temporal(tokens, self.t_blocks[1])
        tokens = self._spatial(tokens, self.s_blocks[2])
        tokens = self.final_norm(tokens)

        actor_cls = tokens[:, -1, self.actor_cls_index]
        critic_cls = tokens[:, -1, self.critic_cls_index]
        teacher_cls = tokens[:, -1, self.teacher_cls_index]
        return actor_cls, critic_cls, teacher_cls

    def forward_all_steps(self, obs_seq):
        """Return CLS tokens at ALL timesteps (not just last), for teacher."""
        batch, time_steps, _ = obs_seq.shape
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1))
        obs_tokens = obs_tokens + self.dim_id_embed(self.dim_indices).view(1, 1, self.obs_dim, EMBED_DIM)

        cls_tokens = torch.stack([self.actor_cls, self.critic_cls, self.teacher_cls], dim=0)
        cls_tokens = cls_tokens.view(1, 1, NUM_CLS_TOKENS, EMBED_DIM).expand(batch, time_steps, -1, -1)
        tokens = torch.cat([obs_tokens, cls_tokens], dim=2)
        tokens = self.input_norm(tokens)

        tokens = self._spatial(tokens, self.s_blocks[0])
        tokens = self._temporal(tokens, self.t_blocks[0])
        tokens = self._spatial(tokens, self.s_blocks[1])
        tokens = self._temporal(tokens, self.t_blocks[1])
        tokens = self._spatial(tokens, self.s_blocks[2])
        tokens = self.final_norm(tokens)

        # Return teacher CLS at all timesteps
        teacher_cls_all = tokens[:, :, self.teacher_cls_index]  # (batch, time_steps, EMBED_DIM)
        return teacher_cls_all


class BidirectionalTeacher(nn.Module):
    """Bidirectional teacher that sees full trajectories + actor's current policy
    and outputs improvement deltas for the actor.

    Input per timestep: (teacher_cls, action_taken, return_to_go, actor_mu)
    Output: delta_mu (direction to move actor's mean), log_std (confidence)

    The teacher's "action" is the delta — trained via REINFORCE where the
    reward is the advantage. The teacher learns: "given where the actor is
    and what happened in this trajectory, which direction should the actor move?"
    """
    def __init__(self, action_dim):
        super().__init__()
        # Input: teacher_cls + action + RTG + actor_mu
        self.input_proj = layer_init(nn.Linear(EMBED_DIM + action_dim + 1 + action_dim, EMBED_DIM))
        self.input_norm = RMSNorm(EMBED_DIM)

        # Bidirectional self-attention layers (NO causal mask)
        init_scale = 1.0 / (2 * TEACHER_LAYERS) ** 0.5
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(TEACHER_LAYERS)]
        )
        self.output_norm = RMSNorm(EMBED_DIM)

        # Output: delta to apply to actor's mean, and log_std for REINFORCE
        self.delta_head = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)

    def forward(self, teacher_latents, actions, returns_to_go, actor_mu):
        """
        Args:
            teacher_latents: (batch, seq_len, EMBED_DIM) from backbone teacher CLS
            actions: (batch, seq_len, action_dim) actions taken during rollout
            returns_to_go: (batch, seq_len, 1)
            actor_mu: (batch, seq_len, action_dim) actor's current mean (detached)
        Returns:
            delta_mu: (batch, seq_len, action_dim) improvement direction
            log_std: (batch, seq_len, action_dim) teacher's confidence
        """
        x = torch.cat([teacher_latents, actions, returns_to_go, actor_mu], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x, is_causal=False)  # bidirectional!

        x = self.output_norm(x)
        return self.delta_head(x), self.log_std_head(x)


class Agent(nn.Module):
    def __init__(self, envs, num_envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN

        self.backbone = STSTSCLSBackbone(obs_dim, self.context_len)
        self.actor_mean = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)

        self.teacher = BidirectionalTeacher(action_dim)

        # Small trainable projection for teacher's backbone input
        # Since backbone is detached during teacher forward (for memory),
        # this gives the teacher a learnable transform on the frozen features
        self.teacher_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            RMSNorm(EMBED_DIM),
        )

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0

    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def get_value(self, obs_seq):
        _, critic_latent, _ = self.backbone(obs_seq)
        return self.critic(critic_latent)

    def get_action_and_value(self, obs_seq, action=None):
        actor_latent, critic_latent, _ = self.backbone(obs_seq)
        action_mean = self.actor_mean(actor_latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_latent)

    def get_actor_logits(self, obs_seq):
        """Get actor's current μ and log_σ for distillation loss."""
        actor_latent, _, _ = self.backbone(obs_seq)
        action_mean = self.actor_mean(actor_latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def get_teacher_output(self, obs_seq_trajectory, actions_trajectory, returns_to_go):
        """
        Run the teacher on full trajectories. Also computes actor's current μ
        at each step (detached) so the teacher can see where the actor is.

        Args:
            obs_seq_trajectory: (num_envs, num_steps, context_len, obs_dim)
            actions_trajectory: (num_envs, num_steps, action_dim)
            returns_to_go: (num_envs, num_steps, 1)
        Returns:
            delta_mu: (num_envs, num_steps, action_dim) improvement direction
            log_std: (num_envs, num_steps, action_dim) teacher confidence
            actor_mu: (num_envs, num_steps, action_dim) actor's current mean (detached)
        """
        num_envs, num_steps = actions_trajectory.shape[:2]
        obs_flat = obs_seq_trajectory.reshape(num_envs * num_steps, self.context_len, -1)

        # Get teacher CLS and actor mu for all steps.
        # Backbone outputs are DETACHED — teacher gradients only flow through
        # the bidirectional teacher layers (not the full backbone graph).
        # The backbone still gets gradients from the PPO critic/actor loop.
        CHUNK = 512
        teacher_latent_chunks = []
        actor_mu_chunks = []
        with torch.no_grad():
            for i in range(0, obs_flat.shape[0], CHUNK):
                actor_latent, _, chunk_teacher = self.backbone(obs_flat[i:i + CHUNK])
                teacher_latent_chunks.append(chunk_teacher)
                actor_mu_chunks.append(self.actor_mean(actor_latent))
        teacher_latents = torch.cat(teacher_latent_chunks, dim=0).reshape(num_envs, num_steps, EMBED_DIM)
        actor_mu = torch.cat(actor_mu_chunks, dim=0).reshape(num_envs, num_steps, self.action_dim)

        # Trainable projection on detached backbone features
        # This gives the teacher a learnable transform it can adapt
        teacher_latents = self.teacher_proj(teacher_latents)

        # Run bidirectional teacher: sees trajectory + where actor currently is
        delta_mu, log_std = self.teacher(
            teacher_latents, actions_trajectory, returns_to_go, actor_mu
        )
        return delta_mu, log_std, actor_mu


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
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
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
                action, logprob, _, value = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
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

        # Compute returns-to-go for teacher input (per env, per step)
        # returns_to_go[t] = sum of discounted rewards from t to end
        with torch.no_grad():
            returns_to_go = torch.zeros_like(rewards)
            rtg = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                # Reset RTG at episode boundaries: dones[t+1] means episode ended after step t
                if t < args.num_steps - 1:
                    mask = 1.0 - dones[t + 1]
                else:
                    mask = 1.0 - next_done
                rtg = rewards[t] + args.gamma * rtg * mask
                returns_to_go[t] = rtg

        # === Teacher phase: process full trajectories bidirectionally ===
        # Reshape to (num_envs, num_steps, ...) for per-env trajectory processing
        obs_seqs_by_env = obs_seqs.permute(1, 0, 2, 3)  # (num_envs, num_steps, ctx, obs_dim)
        actions_by_env = actions.permute(1, 0, 2)         # (num_envs, num_steps, act_dim)
        rtg_by_env = returns_to_go.permute(1, 0).unsqueeze(-1)  # (num_envs, num_steps, 1)

        # Normalize returns-to-go per env for stable teacher input
        with torch.no_grad():
            rtg_mean = rtg_by_env.mean(dim=1, keepdim=True)
            rtg_std = rtg_by_env.std(dim=1, keepdim=True) + 1e-8
            rtg_normed = (rtg_by_env - rtg_mean) / rtg_std

        # Get teacher output: delta_mu (improvement direction) + log_std + actor's current mu
        delta_mu, teacher_log_std, actor_mu_by_env = agent.get_teacher_output(
            obs_seqs_by_env, actions_by_env.detach(), rtg_normed
        )
        # Teacher's proposed target: actor's current mean + delta
        mu_target = actor_mu_by_env + delta_mu

        # Teacher loss: advantage-weighted regression.
        # The teacher's proposed target (actor_mu + delta) should be close to
        # high-advantage actions. This trains the teacher to point the actor
        # toward actions that worked well, using its bidirectional hindsight
        # to identify which actions at which timesteps were responsible.
        with torch.no_grad():
            adv_by_env = advantages.permute(1, 0)  # (num_envs, num_steps)
            # Use positive advantages only — pull toward good actions, ignore bad ones
            teacher_weights = torch.clamp(adv_by_env, min=0.0)
            w_sum = teacher_weights.sum(dim=1, keepdim=True) + 1e-8
            teacher_weights = teacher_weights / w_sum * args.num_steps

        # MSE between teacher's proposed target and the observed high-advantage actions
        target_error = (mu_target - actions_by_env.detach()) ** 2  # (num_envs, num_steps, act_dim)
        teacher_loss = (teacher_weights.unsqueeze(-1) * target_error).mean()

        # Teacher backward immediately (fresh graph, no staleness)
        optimizer.zero_grad()
        (TEACHER_COEF * teacher_loss).backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # Detach mu_target for use in distillation (teacher already updated)
        mu_target = mu_target.detach()

        # === Critic + actor distillation update ===
        b_obs_seqs = obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_seqs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy gradient loss (can be blended with distillation via PG_COEF)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Actor distillation loss: pull actor mean toward teacher targets
                # Only distill mean — let actor learn its own exploration via log_std
                distill_loss = torch.zeros(1, device=device)
                if iteration > DISTILL_WARMUP_ITERS and DISTILL_COEF > 0:
                    actor_mu, _ = agent.get_actor_logits(b_obs_seqs[mb_inds])
                    # Map flat indices back to (env, step) for teacher targets
                    # obs_seqs is (num_steps, num_envs, ...) → flat index i: step=i//num_envs, env=i%num_envs
                    env_idx = mb_inds % args.num_envs
                    step_idx = mb_inds // args.num_envs
                    mb_mu_target = mu_target[env_idx, step_idx].detach()
                    distill_loss = ((actor_mu - mb_mu_target) ** 2).mean()

                # During warmup, use PG so actor isn't frozen; after warmup, blend or use pure distillation
                warmup = iteration <= DISTILL_WARMUP_ITERS
                pg_weight = 1.0 if warmup else PG_COEF

                entropy_loss = entropy.mean()
                loss = (
                    pg_weight * pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + DISTILL_COEF * distill_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

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
        writer.add_scalar("losses/teacher_loss", teacher_loss.item(), global_step)
        writer.add_scalar("losses/distill_loss", distill_loss.item(), global_step)
        with torch.no_grad():
            # Log teacher delta magnitude and target distance
            delta_mag = delta_mu.abs().mean()
            writer.add_scalar("teacher/delta_magnitude", delta_mag.item(), global_step)
            n_sample = min(256, args.batch_size)
            actor_mu_sample, _ = agent.get_actor_logits(b_obs_seqs[:n_sample])
            sample_env_idx = torch.arange(n_sample, device=device) % args.num_envs
            sample_step_idx = torch.arange(n_sample, device=device) // args.num_envs
            teacher_mu_sample = mu_target[sample_env_idx, sample_step_idx]
            target_dist = (actor_mu_sample - teacher_mu_sample).abs().mean()
            writer.add_scalar("teacher/mean_target_distance", target_dist.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
