# SAC + Distributional (DSAC, Ma et al. 2020 / JAIR 2025).
#
# Branches from sac_continuous_action_v1.py — keeps v1's stable tanh
# log-Jacobian, state-dependent logstd, autotuned alpha, twin critics,
# Polyak target updates.
#
# Algorithmic change: scalar Q is replaced by a quantile distribution
# Z_{tau_hat_i}(s, a; theta) parameterised by N fixed QR-DQN-style
# midpoint fractions tau_hat_i = (i + 0.5) / N. Twin distributional
# critics retained; conservative target is per-quantile min across
# twins (Algorithm 1).
#
# Critic loss (per twin k, Eq. 14 with uniform tau partition):
#     L_Z = (1/N) Σ_i Σ_j |tau_hat_j - 1_{delta_ij < 0}| * Huber_kappa(delta_ij)
# where delta_ij = y_i - Z_{tau_hat_j}(s, a; theta_k), y_i is the i-th
# target quantile (after min over twin targets + SAC entropy term),
# and tau_hat_j is the current critic's predicted quantile fraction.
# kappa = 1 -> Huber matches F.smooth_l1_loss(beta=1).
#
# Actor loss: SAC objective with Q derived from quantile average,
# Q(s, a; theta_k) = (1/N) Σ_i Z_{tau_hat_i}(s, a; theta_k), and
# min over twins.
import math
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    num_quantiles: int = 32
    """N in DSAC: number of QR-DQN-style fixed quantile midpoints."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QuantileSoftQNetwork(nn.Module):
    """Distributional critic: outputs N quantile values for each (s, a)."""

    def __init__(self, env, num_quantiles):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_quantiles)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # (B, N)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        y_t = torch.tanh(z)
        action = y_t * self.action_scale + self.action_bias
        log_det_tanh = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = normal.log_prob(z) - log_det_tanh - torch.log(self.action_scale)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def quantile_huber_loss(z_pred, y, tau_hat):
    """DSAC critic loss (Eq. 14) with kappa=1 Huber.

    Args:
        z_pred: (B, N) — current critic's quantile predictions.
        y:      (B, N) — target distribution (already includes SAC entropy term).
        tau_hat: (N,)  — fixed quantile midpoint fractions for the CURRENT critic.

    Returns scalar mean loss.
    """
    # delta[b, i, j] = y[b, i] - z_pred[b, j]
    delta = y.unsqueeze(2) - z_pred.unsqueeze(1)  # (B, N_target, N_pred)
    huber = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="none")
    tau_hat_j = tau_hat.view(1, 1, -1)  # (1, 1, N_pred)
    asym_weight = torch.abs(tau_hat_j - (delta.detach() < 0).float())
    # (1/N_target) Σ_i Σ_j weight_ij * huber_ij averaged over batch.
    per_sample = (asym_weight * huber).mean(dim=1).sum(dim=1)  # (B,)
    return per_sample.mean()


if __name__ == "__main__":

    args = tyro.cli(Args)
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    N = args.num_quantiles
    tau_hat = ((torch.arange(N, dtype=torch.float32, device=device) + 0.5) / N)

    actor = Actor(envs).to(device)
    qf1 = QuantileSoftQNetwork(envs, N).to(device)
    qf2 = QuantileSoftQNetwork(envs, N).to(device)
    qf1_target = QuantileSoftQNetwork(envs, N).to(device)
    qf2_target = QuantileSoftQNetwork(envs, N).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                z1_next = qf1_target(data.next_observations, next_state_actions)  # (B, N)
                z2_next = qf2_target(data.next_observations, next_state_actions)
                # Per-quantile min over twin targets (Algorithm 1 line: y_i = min_k Z_{tau_hat_i}).
                z_next = torch.min(z1_next, z2_next)
                # SAC soft target: subtract alpha * log_pi (broadcast over N).
                z_next = z_next - alpha * next_state_log_pi  # (B, N)
                # Distributional Bellman target.
                r = data.rewards.view(-1, 1)
                not_done = (1.0 - data.dones.view(-1, 1))
                y = r + not_done * args.gamma * z_next  # (B, N)

            z1_pred = qf1(data.observations, data.actions)  # (B, N)
            z2_pred = qf2(data.observations, data.actions)
            qf1_loss = quantile_huber_loss(z1_pred, y, tau_hat)
            qf2_loss = quantile_huber_loss(z2_pred, y, tau_hat)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    z1_pi = qf1(data.observations, pi)  # (B, N)
                    z2_pi = qf2(data.observations, pi)
                    # Scalar Q via quantile average (uniform partition -> mean).
                    q1_pi = z1_pi.mean(dim=1, keepdim=True)
                    q2_pi = z2_pi.mean(dim=1, keepdim=True)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = ((alpha * log_pi) - min_q_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                # Log expected Q (averaged over quantiles) for comparability with scalar SAC.
                q1_mean = z1_pred.mean(dim=1)
                q2_mean = z2_pred.mean(dim=1)
                writer.add_scalar("losses/qf1_values", q1_mean.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", q2_mean.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
