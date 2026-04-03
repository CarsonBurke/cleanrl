# PPO temporal shared-backbone ablation: a single transformer processes the
# observation history plus three learned query tokens, one each for actor,
# critic, and state-dependent exploration.
#
# Key idea:
# - shared temporal transformer backbone
# - one token per history step
# - separate CLS tokens for actor, critic, and SDE
# - discard the history token outputs and use only CLS readouts
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


LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
SDE_EPS = 1e-6
SDE_PRESCALE = 1.5
EMBED_DIM = 32
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
FFN_MULT = 2
NUM_LAYERS = 2
CONTEXT_LEN = 5
NUM_SPECIAL_TOKENS = 3


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
    update_epochs: int = 10
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
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
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

        for module in [self.wq, self.wk, self.wv, self.w1, self.w3]:
            fan_in = module.weight.shape[1]
            std = 1.0 / fan_in**0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        for module in [self.wo, self.w2]:
            fan_in = module.weight.shape[1]
            std = init_scale / fan_in**0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x, rope_cos, rope_sin):
        batch, steps, width = x.shape
        h = self.attn_pre_norm(x)
        q = self.wq(h).view(batch, steps, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(batch, steps, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(batch, steps, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, steps, self.head_dim)
            k = k.reshape(batch, self.num_q_heads, steps, self.head_dim)
            v = v.unsqueeze(2).expand(batch, self.num_kv_heads, self.kv_group_size, steps, self.head_dim)
            v = v.reshape(batch, self.num_q_heads, steps, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, steps, width)
        x = x + self.attn_post_norm(self.wo(attn_out))

        h = self.ffn_pre_norm(x)
        ffn_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + self.ffn_post_norm(ffn_out)
        return x


class Agent(nn.Module):
    def __init__(self, envs, num_envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.context_len = CONTEXT_LEN

        self.embed_fc1 = layer_init(nn.Linear(obs_dim, EMBED_DIM))
        self.embed_fc2 = layer_init(nn.Linear(EMBED_DIM, EMBED_DIM))
        self.embed_norm = RMSNorm(EMBED_DIM)

        cls_std = 1.0 / EMBED_DIM**0.5
        self.actor_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * cls_std)
        self.critic_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * cls_std)
        self.sde_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * cls_std)

        head_dim = EMBED_DIM // NUM_Q_HEADS
        rope_cos, rope_sin = build_rope_cache(self.context_len + NUM_SPECIAL_TOKENS, head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        init_scale = 1.0 / (2 * NUM_LAYERS) ** 0.5
        self.layers = nn.ModuleList(
            [GQATransformerBlock(EMBED_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT, init_scale) for _ in range(NUM_LAYERS)]
        )
        self.final_norm = RMSNorm(EMBED_DIM)

        self.actor_mean = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)
        self.sde_proj = layer_init(nn.Linear(EMBED_DIM, EMBED_DIM), std=1.0)
        self.log_std_param = nn.Parameter(torch.zeros(EMBED_DIM, action_dim))
        self.log_std_offset_proj = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)

        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None:
            self.obs_history.zero_()
        else:
            self.obs_history[env_mask] = 0.0

    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)

    def _encode(self, obs_seq):
        batch = obs_seq.shape[0]
        tokens = F.silu(self.embed_fc1(obs_seq))
        tokens = self.embed_fc2(tokens)
        tokens = self.embed_norm(tokens)
        special = torch.cat(
            [
                self.actor_cls.expand(batch, -1, -1),
                self.critic_cls.expand(batch, -1, -1),
                self.sde_cls.expand(batch, -1, -1),
            ],
            dim=1,
        )
        tokens = torch.cat([special, tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, self.rope_cos, self.rope_sin)
        tokens = self.final_norm(tokens)
        return tokens[:, 0], tokens[:, 1], tokens[:, 2]

    def _action_std(self, sde_features):
        sde_latent = torch.tanh(self.sde_proj(sde_features) / SDE_PRESCALE)
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        action_var = (sde_latent.pow(2)) @ log_std.exp().pow(2)
        log_std_state = (self.log_std_offset_proj(sde_features) + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        action_var = action_var + log_std_state.exp().pow(2)
        return (action_var + SDE_EPS).sqrt()

    def get_value(self, obs_seq):
        _, critic_features, _ = self._encode(obs_seq)
        return self.critic(critic_features)

    def get_action_and_value(self, obs_seq, action=None):
        actor_features, critic_features, sde_features = self._encode(obs_seq)
        action_mean = self.actor_mean(actor_features)
        action_std = self._action_std(sde_features)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_features)


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.num_envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
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

        log_std_eff = (agent.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        writer.add_scalar("sde/log_std_mean", log_std_eff.mean().item(), global_step)
        writer.add_scalar("sde/log_std_std", log_std_eff.std().item(), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
