# PPO + Dreamer-style differentiable world model with imagination rollouts.
#
# Shared axial space-time transformer backbone with 3 CLS tokens:
# actor, critic, dynamics. SDE exploration removed — the imagination
# rollouts provide gradient signal about future consequences directly.
#
# World model: transition(z_t, a_t) -> z_{t+1}, reward(z_t, a_t) -> r,
# continue(z_t, a_t) -> p(continue). Actor trained via backprop through
# imagined rollouts (Dreamer-v3 style) + PPO on real data.
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
NUM_CLS_TOKENS = 3
LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
IMAGINATION_HORIZON = 15
WM_COEF = 1.0
IMAGINE_COEF = 0.1
IMAGINE_CRITIC_COEF = 0.5
N_IMAGINE_SEEDS = 256


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

    def forward(self, x, rope_cos=None, rope_sin=None):
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

        attn = F.scaled_dot_product_attention(q, k, v)
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
        self.dynamics_cls_index = obs_dim + 2

        self.value_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.input_norm = RMSNorm(EMBED_DIM)
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        self.register_buffer("dim_indices", torch.arange(obs_dim))
        cls_std = 1.0 / EMBED_DIM**0.5
        self.actor_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.critic_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.dynamics_cls = nn.Parameter(torch.empty(EMBED_DIM))
        nn.init.trunc_normal_(self.actor_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.critic_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)
        nn.init.trunc_normal_(self.dynamics_cls, std=cls_std, a=-2 * cls_std, b=2 * cls_std)

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

        cls_tokens = torch.stack([self.actor_cls, self.critic_cls, self.dynamics_cls], dim=0)
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
        dynamics_cls = tokens[:, -1, self.dynamics_cls_index]
        return actor_cls, critic_cls, dynamics_cls


class Agent(nn.Module):
    def __init__(self, envs, num_envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.context_len = CONTEXT_LEN

        self.backbone = STSTSCLSBackbone(obs_dim, self.context_len)
        self.actor_mean = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), LOG_STD_INIT))
        self.critic = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)

        # World model heads
        self.transition = nn.Sequential(
            nn.Linear(EMBED_DIM + action_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, EMBED_DIM),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(EMBED_DIM + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(EMBED_DIM + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0

    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def _encode(self, obs_seq):
        return self.backbone(obs_seq)

    def _action_std(self):
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def get_value(self, obs_seq):
        _, critic_latent, _ = self._encode(obs_seq)
        return self.critic(critic_latent)

    def get_action_and_value(self, obs_seq, action=None):
        actor_latent, critic_latent, dynamics_latent = self._encode(obs_seq)
        action_mean = self.actor_mean(actor_latent)
        action_std = self._action_std()
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_latent), dynamics_latent

    def get_action_and_value_no_latent(self, obs_seq, action=None):
        """For PPO update where we don't need dynamics latent recomputed."""
        actor_latent, critic_latent, _ = self._encode(obs_seq)
        action_mean = self.actor_mean(actor_latent)
        action_std = self._action_std()
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_latent)

    def imagine(self, z_start, horizon, gamma, gae_lambda):
        """Differentiable imagination rollout from seed latents.

        Returns imagined lambda-returns for actor loss and per-step values for critic loss.
        """
        action_std = self._action_std()
        z = z_start
        rewards = []
        values = []
        continues = []

        for _ in range(horizon):
            a_mean = self.actor_mean(z)
            a = Normal(a_mean, action_std).rsample()
            za = torch.cat([z, a], dim=-1)
            rewards.append(self.reward_head(za).squeeze(-1))
            continues.append(self.continue_head(za).sigmoid().squeeze(-1))
            values.append(self.critic(z).squeeze(-1))
            z = self.transition(za)

        # Bootstrap value at horizon
        values.append(self.critic(z).squeeze(-1))

        # Compute lambda-returns (all differentiable)
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
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    latents = torch.zeros((args.num_steps, args.num_envs, EMBED_DIM), device=device)

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
                action, logprob, _, value, dynamics_latent = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
                latents[step] = dynamics_latent
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

        b_obs_seqs = obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_latents = latents.reshape(-1, EMBED_DIM)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)

        # Build next-latent targets and validity mask for world model
        # Shift latents by 1 within each env, masking episode boundaries
        next_latents = torch.zeros_like(latents)
        next_latents[:-1] = latents[1:]
        # Last step: get latent from current obs_history
        with torch.no_grad():
            _, _, final_dynamics = agent._encode(agent.obs_history)
            next_latents[-1] = final_dynamics
        # Mask: transition is invalid if next step is a new episode
        wm_valid = torch.ones(args.num_steps, args.num_envs, device=device)
        wm_valid[:-1] = 1.0 - dones[1:]  # invalid if next step starts new episode
        b_next_latents = next_latents.reshape(-1, EMBED_DIM)
        b_wm_valid = wm_valid.reshape(-1)

        # PPO update with world model loss
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value_no_latent(b_obs_seqs[mb_inds], b_actions[mb_inds])
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

                # World model loss on this minibatch
                mb_za = torch.cat([b_latents[mb_inds], b_actions[mb_inds]], dim=-1)
                mb_z_pred = agent.transition(mb_za)
                mb_r_pred = agent.reward_head(mb_za).squeeze(-1)
                mb_c_pred = agent.continue_head(mb_za).squeeze(-1)
                mb_valid = b_wm_valid[mb_inds]

                transition_loss = (((mb_z_pred - b_next_latents[mb_inds].detach()) ** 2).mean(-1) * mb_valid).sum() / (mb_valid.sum() + 1e-8)
                reward_loss = ((mb_r_pred - b_rewards[mb_inds]) ** 2 * mb_valid).sum() / (mb_valid.sum() + 1e-8)
                # Continue target: 1 if next step is NOT a terminal, 0 if terminal
                continue_target = mb_valid  # reuse: valid transitions are exactly the continuing ones
                continue_loss = F.binary_cross_entropy_with_logits(mb_c_pred, continue_target, reduction="mean")
                wm_loss = transition_loss + reward_loss + continue_loss

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + WM_COEF * wm_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Imagination phase: differentiable rollouts for actor + critic
        n_seeds = min(N_IMAGINE_SEEDS, args.batch_size)
        seed_inds = torch.randperm(args.batch_size, device=device)[:n_seeds]
        z_seeds = b_latents[seed_inds].detach()

        lambda_returns, imagined_values = agent.imagine(
            z_seeds, IMAGINATION_HORIZON, args.gamma, args.gae_lambda
        )

        # Actor loss: maximize imagined returns (gradient flows through dynamics)
        imagine_actor_loss = -torch.stack(lambda_returns).mean()

        # Critic loss: regress values toward lambda-return targets (detached)
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
        writer.add_scalar("imagination/actor_loss", imagine_actor_loss.item(), global_step)
        writer.add_scalar("imagination/critic_loss", imagine_critic_loss.item() if isinstance(imagine_critic_loss, torch.Tensor) else imagine_critic_loss, global_step)
        with torch.no_grad():
            writer.add_scalar("imagination/mean_return", torch.stack(lambda_returns).mean().item(), global_step)
            action_std = agent._action_std()
            writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
            writer.add_scalar("policy/log_std_mean", agent.log_std.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
