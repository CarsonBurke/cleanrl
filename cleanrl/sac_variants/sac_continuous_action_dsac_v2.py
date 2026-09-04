# SAC + DSAC v2: closer to the reference (Ma et al. 2020).
#
# Branches from sac_continuous_action_dsac.py. Adds three changes per
# the official repo (https://github.com/xtma/dsac, HalfCheetah config):
#   (1) IQN-style critic with cosine tau embedding, no LayerNorm
#       (skipped to minimize independent variables vs SAC baseline).
#       Reference: QuantileMlp in networks.py.
#   (2) zf_lr = 3e-4 (their HalfCheetah config; canonical SAC uses 1e-3).
#   (4) Polyak-averaged target actor; next-state actions in the Bellman
#       target are sampled from the target actor, not the live one.
# Skipped: LayerNorm (3); fixed alpha (5). Autotune stays on.
#
# Critic per-call:
#   presum_tau ~ Uniform(0,1) + 0.1, normalized; tau = cumsum;
#   tau_hat   = midpoints. Independent samples for current critic and
#   target critic (next_tau_hat). Loss weights by next_presum_tau (the
#   target's per-quantile mass widths).
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
    q_lr: float = 3e-4  # (2) was 1e-3 in canonical SAC; DSAC ref uses 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    num_quantiles: int = 32
    tau_embedding_size: int = 64


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


HIDDEN = 256


class QuantileMlp(nn.Module):
    """IQN-style distributional critic (no LayerNorm)."""

    def __init__(self, env, embedding_size):
        super().__init__()
        in_dim = int(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape))
        self.embedding_size = embedding_size
        # base_fc: state-action features
        self.base_fc = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(inplace=True),
        )
        # tau_fc: cosine tau embedding -> gating signal
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, HIDDEN),
            nn.Sigmoid(),
        )
        # merge_fc: after multiplicative merge with tau embedding
        self.merge_fc = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(HIDDEN, 1)
        self.register_buffer("const_vec", torch.arange(1, embedding_size + 1, dtype=torch.float32))

    def forward(self, state, action, tau):
        """tau: (B, N) — quantile fractions. Returns (B, N) quantile values."""
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)                                        # (B, HIDDEN)
        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * math.pi)  # (B, N, embedding_size)
        x = self.tau_fc(x)                                          # (B, N, HIDDEN)
        h = x * h.unsqueeze(-2)                                     # (B, N, HIDDEN), broadcast
        h = self.merge_fc(h)                                        # (B, N, HIDDEN)
        return self.last_fc(h).squeeze(-1)                          # (B, N)


def sample_iqn_tau(batch_size, num_quantiles, device):
    """IQN-style stochastic tau (Ma et al. dsac.py get_tau).
    Returns (presum_tau, tau, tau_hat), all shape (B, N)."""
    presum_tau = torch.rand(batch_size, num_quantiles, device=device) + 0.1
    presum_tau = presum_tau / presum_tau.sum(dim=1, keepdim=True)
    tau = torch.cumsum(presum_tau, dim=1)                    # (B, N) in (0, 1]
    # midpoints
    tau_hat = torch.zeros_like(tau)
    tau_hat[:, 0] = tau[:, 0] / 2.0
    tau_hat[:, 1:] = (tau[:, :-1] + tau[:, 1:]) / 2.0
    return presum_tau, tau, tau_hat


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc_mean = nn.Linear(HIDDEN, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(HIDDEN, np.prod(env.single_action_space.shape))
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


def quantile_huber_loss(z_pred, z_target, tau_hat_pred, presum_tau_target):
    """DSAC critic loss (matches xtma/dsac quantile_regression_loss).

    Args:
        z_pred:           (B, N_pred)   — current critic's quantile predictions.
        z_target:         (B, N_target) — target distribution (already includes SAC entropy term).
        tau_hat_pred:     (B, N_pred)   — current critic's quantile midpoints.
        presum_tau_target:(B, N_target) — target's quantile mass widths (IQN weights).

    Returns scalar mean loss.
    """
    # delta[b, j, i] = pred[b, j] - target[b, i]
    pred = z_pred.unsqueeze(-1)                            # (B, N_pred, 1)
    target = z_target.detach().unsqueeze(-2)               # (B, 1, N_target)
    tau = tau_hat_pred.detach().unsqueeze(-1)              # (B, N_pred, 1)
    weight = presum_tau_target.detach().unsqueeze(-2)      # (B, 1, N_target)
    delta = pred - target                                  # (B, N_pred, N_target)
    huber = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="none")
    # sign = 1 if pred > target else 0   (matches ref: sign(pred - target)/2 + 0.5)
    sign = torch.sign(delta) / 2.0 + 0.5
    rho = torch.abs(tau - sign) * huber * weight           # (B, N_pred, N_target)
    return rho.sum(dim=-1).mean()


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
    actor = Actor(envs).to(device)
    target_actor = Actor(envs).to(device)         # (4) Polyak-averaged target actor
    target_actor.load_state_dict(actor.state_dict())
    for p in target_actor.parameters():
        p.requires_grad = False

    qf1 = QuantileMlp(envs, args.tau_embedding_size).to(device)
    qf2 = QuantileMlp(envs, args.tau_embedding_size).to(device)
    qf1_target = QuantileMlp(envs, args.tau_embedding_size).to(device)
    qf2_target = QuantileMlp(envs, args.tau_embedding_size).to(device)
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
            B = data.observations.shape[0]

            # Independent IQN tau samples for target (next) and current critic.
            next_presum_tau, _, next_tau_hat = sample_iqn_tau(B, N, device)
            _, _, tau_hat = sample_iqn_tau(B, N, device)

            with torch.no_grad():
                # (4) Use target actor to sample next-state action.
                next_state_actions, next_state_log_pi, _ = target_actor.get_action(data.next_observations)
                z1_next = qf1_target(data.next_observations, next_state_actions, next_tau_hat)  # (B, N)
                z2_next = qf2_target(data.next_observations, next_state_actions, next_tau_hat)
                z_next = torch.min(z1_next, z2_next)
                z_next = z_next - alpha * next_state_log_pi                                       # (B, N)
                r = data.rewards.view(-1, 1)
                not_done = 1.0 - data.dones.view(-1, 1)
                z_target = r + not_done * args.gamma * z_next                                     # (B, N)

            z1_pred = qf1(data.observations, data.actions, tau_hat)                               # (B, N)
            z2_pred = qf2(data.observations, data.actions, tau_hat)
            qf1_loss = quantile_huber_loss(z1_pred, z_target, tau_hat, next_presum_tau)
            qf2_loss = quantile_huber_loss(z2_pred, z_target, tau_hat, next_presum_tau)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    # Fresh tau sample for actor evaluation.
                    _, _, tau_hat_actor = sample_iqn_tau(B, N, device)
                    z1_pi = qf1(data.observations, pi, tau_hat_actor)
                    z2_pi = qf2(data.observations, pi, tau_hat_actor)
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
                # (4) Polyak-average target actor.
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
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
