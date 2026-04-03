"""Axial transformer: separate space and time attention blocks.

Architecture:
- Input: (B, CONTEXT_LEN, obs_dim)
- Embedding: learned ID embedding per obs dim + linear value projection (additive)
  This gives each sensor a unique identity ("I am knee angle") modulated by its reading.
  Spatial blocks use NO positional encoding (obs dims have no natural ordering).
- Alternating blocks:
  1. Spatial block: attention across D dims within each timestep (no RoPE)
  2. Temporal block: attention across T timesteps for each dim (with RoPE)
- Readout: take last timestep, mean-pool across dims -> (B, EMBED_DIM)
- Separate actor/critic backbones
"""
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
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
FFN_MULT = 2
NUM_LAYERS = 2  # 2 spatial + 2 temporal = 4 total blocks
CONTEXT_LEN = 5


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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    clip_vloss: bool = True
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
        std = 1.0 / fan_in ** 0.5
    nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def build_rope_cache(seq_len, head_dim, device):
    assert head_dim % 2 == 0
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQATransformerBlock(nn.Module):
    """Transformer++ block: sandwich norm, QK-norm, RoPE, GQA, SwiGLU."""

    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_mult=2, init_scale=1.0):
        super().__init__()
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.attn_pre_norm = RMSNorm(dim)
        self.attn_post_norm = RMSNorm(dim)
        self.ffn_pre_norm = RMSNorm(dim)
        self.ffn_post_norm = RMSNorm(dim)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

        self._init_weights(init_scale)

    def _init_weights(self, init_scale):
        for module in [self.wq, self.wk, self.wv, self.w1, self.w3]:
            fan_in = module.weight.shape[1]
            std = 1.0 / fan_in ** 0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
        for module in [self.wo, self.w2]:
            fan_in = module.weight.shape[1]
            std = init_scale / fan_in ** 0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, D = x.shape

        h = self.attn_pre_norm(x)
        q = self.wq(h).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_group_size, T, self.head_dim)
            k = k.reshape(B, self.num_q_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_group_size, T, self.head_dim)
            v = v.reshape(B, self.num_q_heads, T, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.wo(attn_out)

        x = x + self.attn_post_norm(attn_out)

        h = self.ffn_pre_norm(x)
        ffn_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + self.ffn_post_norm(ffn_out)

        return x


class AxialTransformerBackbone(nn.Module):
    """Axial transformer: alternating spatial and temporal attention blocks.

    Input: (B, T, D, EMBED_DIM) where T=timesteps, D=obs_dims
    Spatial blocks: attend across D for each T independently
    Temporal blocks: attend across T for each D independently
    """

    def __init__(self, obs_dim, num_layers):
        super().__init__()
        self.obs_dim = obs_dim

        # ID embedding: each obs dim gets a unique learned vector
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        # Value projection: scalar reading -> EMBED_DIM (additive with ID)
        self.value_proj = nn.Linear(1, EMBED_DIM)
        std_vp = 1.0  # fan_in=1
        nn.init.trunc_normal_(self.value_proj.weight, std=std_vp, a=-2*std_vp, b=2*std_vp)

        self.embed_norm = RMSNorm(EMBED_DIM)

        # Total blocks = 2 * num_layers (alternating spatial, temporal)
        total_blocks = 2 * num_layers
        init_scale = 1.0 / (2 * total_blocks) ** 0.5

        self.spatial_blocks = nn.ModuleList([
            GQATransformerBlock(EMBED_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT, init_scale)
            for _ in range(num_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            GQATransformerBlock(EMBED_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT, init_scale)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(EMBED_DIM)

        # RoPE only for temporal blocks (timesteps have natural ordering)
        head_dim = EMBED_DIM // NUM_Q_HEADS
        temporal_cos, temporal_sin = build_rope_cache(CONTEXT_LEN, head_dim, torch.device('cpu'))
        self.register_buffer('temporal_cos', temporal_cos)
        self.register_buffer('temporal_sin', temporal_sin)

        # Pre-compute dim indices for embedding lookup
        self.register_buffer('dim_indices', torch.arange(obs_dim))

    def forward(self, obs_seq):
        """obs_seq: (B, T, D_obs). Returns (B, EMBED_DIM)."""
        B, T, D = obs_seq.shape

        # ID embedding + value projection: (B, T, D) -> (B, T, D, E)
        id_embed = self.dim_id_embed(self.dim_indices)  # (D, E)
        val_embed = self.value_proj(obs_seq.unsqueeze(-1))  # (B, T, D, E)
        tokens = id_embed + val_embed  # broadcast: (B, T, D, E)
        tokens = self.embed_norm(tokens)

        # Alternating spatial and temporal blocks
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: attend across D dims, independently per timestep (no RoPE)
            tokens = tokens.reshape(B * T, D, EMBED_DIM)
            tokens = spatial_block(tokens)  # no positional encoding
            tokens = tokens.reshape(B, T, D, EMBED_DIM)

            # Temporal: attend across T timesteps, independently per dim (with RoPE)
            tokens = tokens.permute(0, 2, 1, 3).reshape(B * D, T, EMBED_DIM)
            tokens = temporal_block(tokens, self.temporal_cos, self.temporal_sin)
            tokens = tokens.reshape(B, D, T, EMBED_DIM).permute(0, 2, 1, 3)  # back to (B, T, D, E)

        tokens = self.final_norm(tokens)

        # Readout: last timestep, mean-pool across dims
        last_step = tokens[:, -1]  # (B, D, E)
        return last_step.mean(dim=1)  # (B, E)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.obs_dim = obs_dim
        self.num_envs = args.num_envs

        # Separate actor/critic axial transformer backbones
        self.actor_backbone = AxialTransformerBackbone(obs_dim, NUM_LAYERS)
        self.critic_backbone = AxialTransformerBackbone(obs_dim, NUM_LAYERS)

        # Actor head
        self.actor_out = layer_init(nn.Linear(EMBED_DIM, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Critic head
        self.value_out = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)

        # History buffer
        self.register_buffer('obs_history', torch.zeros(args.num_envs, CONTEXT_LEN, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0

    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def get_value(self, obs_seq):
        return self.value_out(self.critic_backbone(obs_seq))

    def get_action_and_value(self, obs_seq, action=None):
        actor_features = self.actor_backbone(obs_seq)
        critic_features = self.critic_backbone(obs_seq)

        action_mean = self.actor_out(actor_features)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.value_out(critic_features)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    obs_seqs = torch.zeros((args.num_steps, args.num_envs, CONTEXT_LEN, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    agent.reset_history()
    agent.update_history(next_obs)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_seqs[step] = agent.obs_history.clone()
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            if next_done.any():
                agent.reset_history(next_done.bool())
            agent.update_history(next_obs)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(agent.obs_history).reshape(1, -1)
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

        b_obs_seqs = obs_seqs.reshape(-1, CONTEXT_LEN, obs_dim)
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
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio < (1 - args.clip_coef)) | (ratio > (1 + args.clip_coef))).float().mean().item()]
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
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate
        episodic_returns = evaluate(model_path, make_env, args.env_id, eval_episodes=10,
                                    run_name=f"{run_name}-eval", Model=Agent, device=device, gamma=args.gamma)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
