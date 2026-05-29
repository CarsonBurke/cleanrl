# PPO + IterThink (v1). Branched from ppo_continuous_action_fire_v19, FIRE stripped.
#
# Iterative "thinking" trunk: a single shared block applied K times to a
# residual stream. Each step's input is conditioned on a per-step embedding
# (concat + project) so the shared weights can still implement different
# behavior per step. The block has three parallel paths:
#
#   1. Dense path  — standard MLP transform (Linear -> ReLU^2 -> Linear).
#   2. Soft MoE    — n_experts parallel MLPs, weighted by softmax(gate); all
#                    experts run every forward (no top-k dispatch).
#   3. Spatial conv — Conv1d kernels that slide across the H-axis (feature
#                    axis as 1D spatial); one set of small kernels reused at
#                    every position. Pressure encourages the input projection
#                    to organize features into locally meaningful groupings.
#
# All three deltas sum into the residual stream:
#     x_k = x_{k-1} + d_dense + d_moe + d_conv
#
# Both actor and critic get their own IterTrunk (separate weights). Output
# is normed before the head. Distribution: tanh-squashed Gaussian with
# stable log-Jacobian and state-independent logstd, same as v19.
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
from math import log
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    num_envs: int = 16
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

    hidden: int = 64
    k_steps: int = 3
    n_experts: int = 4
    conv_channels: int = 8
    conv_kernel: int = 3

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
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class IterBlock(nn.Module):
    """Shared-weight block applied K times. Step embedding concat+project
    differentiates iterations. Three parallel paths sum into the stream."""

    def __init__(self, H, K, n_experts, conv_channels, conv_kernel):
        super().__init__()
        self.norm = nn.RMSNorm(H)
        # Per-step embedding; concat with normed features then project back to H.
        self.step_emb = nn.Parameter(torch.zeros(K, H))
        self.step_proj = layer_init(nn.Linear(2 * H, H))
        # Dense path.
        self.dense = nn.Sequential(
            layer_init(nn.Linear(H, H)),
            ReLUSquared(),
            layer_init(nn.Linear(H, H)),
        )
        # Soft MoE: parallel experts, softmax gating, all experts run.
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    layer_init(nn.Linear(H, H)),
                    ReLUSquared(),
                    layer_init(nn.Linear(H, H)),
                )
                for _ in range(n_experts)
            ]
        )
        # Spatial conv path: kernels slide across the H (feature) axis.
        self.conv_in = layer_init(nn.Conv1d(1, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2))
        self.conv_act = ReLUSquared()
        self.conv_out = layer_init(nn.Conv1d(conv_channels, 1, kernel_size=1))

    def step(self, x, k):
        """One iteration; returns updated residual stream value."""
        h_norm = self.norm(x)
        step_e = self.step_emb[k].unsqueeze(0).expand(h_norm.size(0), -1)  # (B, H)
        h = self.step_proj(torch.cat([h_norm, step_e], dim=-1))             # (B, H)
        # Dense path.
        d_dense = self.dense(h)
        # MoE path.
        w = torch.softmax(self.gate(h), dim=-1)                             # (B, E)
        e_out = torch.stack([e(h) for e in self.experts], dim=-1)           # (B, H, E)
        d_moe = (w.unsqueeze(1) * e_out).sum(dim=-1)                        # (B, H)
        # Spatial conv path: H is the 1D spatial axis.
        z = self.conv_act(self.conv_in(h.unsqueeze(1)))                     # (B, C, H)
        d_conv = self.conv_out(z).squeeze(1)                                # (B, H)
        return x + d_dense + d_moe + d_conv


class IterTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts, conv_channels, conv_kernel):
        super().__init__()
        self.K = K
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.block = IterBlock(H, K, n_experts, conv_channels, conv_kernel)
        self.out_norm = nn.RMSNorm(H)

    def forward(self, x):
        x = self.in_proj(x)
        for k in range(self.K):
            x = self.block.step(x, k)
        return self.out_norm(x)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        K = args.k_steps
        self.critic_trunk = IterTrunk(obs_dim, H, K, args.n_experts, args.conv_channels, args.conv_kernel)
        self.critic_head = layer_init(nn.Linear(H, 1), std=1.0)
        self.actor_trunk = IterTrunk(obs_dim, H, K, args.n_experts, args.conv_channels, args.conv_kernel)
        self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic_head(self.critic_trunk(x))

    def get_action_and_value(self, x, z=None):
        mean = self.actor_head(self.actor_trunk(x))
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if z is None:
            z = probs.sample()
        action = torch.tanh(z)
        log_det = 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = (probs.log_prob(z) - log_det).sum(1)
        value = self.critic_head(self.critic_trunk(x))
        return action, z, log_prob, probs.entropy().sum(1), value


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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
                action, z, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
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

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_latent_zs[mb_inds])
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
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
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

    envs.close()
    writer.close()
