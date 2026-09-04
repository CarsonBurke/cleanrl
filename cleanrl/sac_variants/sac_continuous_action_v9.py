# SAC v9: v4 optimistic HL-Gauss critic + denser same-range support.
#
# Keeps v4's pure CE loss, support midpoint, and sigma/bin smoothing ratio.
# Only doubles support resolution to test whether late plateau is from coarse
# actor-facing Q ranking. No scalar auxiliary and no zero-Q init.
#
# Diagnostics log projection bias, support usage, scalar TD error, and critic
# disagreement. CUDA is required.
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
from cleanrl.shared.hl_gauss import HLGaussSupport, symexp, symlog


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

    # HL-Gauss critic — raw support sized to HalfCheetah's SAC Q-range.
    # symlog dropped: SAC's actor backprops Q's gradient through to_scalar;
    # with symlog, the symexp Jacobian amplifies by exp(|z|) ~= |Q|,
    # destabilizing the actor.
    num_bins: int = 201
    v_min: float = -500.0
    v_max: float = 3000.0
    sigma_ratio: float = 0.75
    value_symlog: bool = False


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


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


HIDDEN = 256


class HLGaussSoftQNetwork(nn.Module):
    """Distributional critic: outputs num_bins logits per (s, a)."""

    def __init__(self, env, num_bins):
        super().__init__()
        in_dim = int(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape))
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            ReLUSquared(),
            nn.Linear(HIDDEN, HIDDEN),
            ReLUSquared(),
        )
        self.head = nn.Linear(HIDDEN, num_bins)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.head(self.trunk(x))  # logits (B, num_bins)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            ReLUSquared(),
            nn.Linear(HIDDEN, HIDDEN),
            ReLUSquared(),
        )
        self.fc_mean = nn.Linear(HIDDEN, act_dim)
        self.fc_logstd = nn.Linear(HIDDEN, act_dim)
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
        h = self.trunk(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
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


def logits_to_probs(logits):
    return torch.softmax(logits, dim=-1)


def probs_to_scalar(probs, hl_support):
    value = (probs * hl_support.support).sum(dim=-1)
    if hl_support.use_symlog:
        value = symexp(value)
    return value


def transformed_clip_fraction(values, hl_support):
    if hl_support.use_symlog:
        values = symlog(values)
    return ((values < hl_support.v_min) | (values > hl_support.v_max)).float().mean()


def categorical_js_divergence(p, q, eps):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=-1)
        + (q * (q.log() - m.log())).sum(dim=-1)
    )


if __name__ == "__main__":

    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This SAC ablation is CUDA-only; run with CUDA available and --cuda=True.")

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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    hl_support = HLGaussSupport(
        num_bins=args.num_bins,
        v_min=args.v_min,
        v_max=args.v_max,
        sigma_ratio=args.sigma_ratio,
        device=device,
        use_symlog=args.value_symlog,
    )

    actor = Actor(envs).to(device)
    qf1 = HLGaussSoftQNetwork(envs, args.num_bins).to(device)
    qf2 = HLGaussSoftQNetwork(envs, args.num_bins).to(device)
    qf1_target = HLGaussSoftQNetwork(envs, args.num_bins).to(device)
    qf2_target = HLGaussSoftQNetwork(envs, args.num_bins).to(device)
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
                qf1_next_logits = qf1_target(data.next_observations, next_state_actions)
                qf2_next_logits = qf2_target(data.next_observations, next_state_actions)
                qf1_next = hl_support.to_scalar(qf1_next_logits)
                qf2_next = hl_support.to_scalar(qf2_next_logits)
                min_qf_next_target = torch.min(qf1_next, qf2_next) - alpha * next_state_log_pi.squeeze(-1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target

            # Cross-entropy vs HL-Gauss projection of scalar target.
            target_probs = hl_support.project(next_q_value)                 # (B, num_bins)
            qf1_logits = qf1(data.observations, data.actions)               # (B, num_bins)
            qf2_logits = qf2(data.observations, data.actions)
            qf1_log_probs = F.log_softmax(qf1_logits, dim=-1)
            qf2_log_probs = F.log_softmax(qf2_logits, dim=-1)
            qf1_loss = -(target_probs * qf1_log_probs).sum(dim=-1).mean()
            qf2_loss = -(target_probs * qf2_log_probs).sum(dim=-1).mean()
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = hl_support.to_scalar(qf1(data.observations, pi))   # (B,)
                    qf2_pi = hl_support.to_scalar(qf2(data.observations, pi))
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi.squeeze(-1)) - min_qf_pi).mean()

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
                with torch.no_grad():
                    qf1_probs = logits_to_probs(qf1_logits)
                    qf2_probs = logits_to_probs(qf2_logits)
                    q1_scalar_values = probs_to_scalar(qf1_probs, hl_support)
                    q2_scalar_values = probs_to_scalar(qf2_probs, hl_support)
                    target_probs_for_diag = hl_support.project(next_q_value)
                    target_scalar_values = probs_to_scalar(target_probs_for_diag, hl_support)
                    q1_scalar = q1_scalar_values.mean().item()
                    q2_scalar = q2_scalar_values.mean().item()
                    scalar_td_abs = 0.5 * (
                        (q1_scalar_values - next_q_value).abs().mean()
                        + (q2_scalar_values - next_q_value).abs().mean()
                    )
                    projection_mean_bias = (target_scalar_values - next_q_value).mean()
                    q_edge_mass = 0.5 * (
                        (qf1_probs[:, 0] + qf1_probs[:, -1]).mean()
                        + (qf2_probs[:, 0] + qf2_probs[:, -1]).mean()
                    )
                    q_entropy = 0.5 * (
                        -(qf1_probs * qf1_probs.clamp_min(hl_support.eps).log()).sum(dim=-1).mean()
                        -(qf2_probs * qf2_probs.clamp_min(hl_support.eps).log()).sum(dim=-1).mean()
                    )
                    target_dist_entropy = (
                        -(target_probs_for_diag * target_probs_for_diag.clamp_min(hl_support.eps).log())
                        .sum(dim=-1)
                        .mean()
                    )
                    target_clip_frac = transformed_clip_fraction(next_q_value, hl_support)
                    target_edge_mass = (target_probs_for_diag[:, 0] + target_probs_for_diag[:, -1]).mean()
                    q_scalar_disagreement = (q1_scalar_values - q2_scalar_values).abs().mean()
                    q_dist_js = categorical_js_divergence(qf1_probs, qf2_probs, hl_support.eps).mean()
                writer.add_scalar("losses/qf1_values", q1_scalar, global_step)
                writer.add_scalar("losses/qf2_values", q2_scalar, global_step)
                writer.add_scalar("losses/target_values", next_q_value.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/scalar_td_abs", scalar_td_abs.item(), global_step)
                writer.add_scalar("losses/projection_mean_bias", projection_mean_bias.item(), global_step)
                writer.add_scalar("losses/target_clip_frac", target_clip_frac.item(), global_step)
                writer.add_scalar("losses/target_edge_mass", target_edge_mass.item(), global_step)
                writer.add_scalar("losses/q_edge_mass", q_edge_mass.item(), global_step)
                writer.add_scalar("losses/target_entropy", target_dist_entropy.item(), global_step)
                writer.add_scalar("losses/q_entropy", q_entropy.item(), global_step)
                writer.add_scalar("losses/q_scalar_disagreement", q_scalar_disagreement.item(), global_step)
                writer.add_scalar("losses/q_dist_js", q_dist_js.item(), global_step)
                writer.add_scalar("losses/target_min", next_q_value.min().item(), global_step)
                writer.add_scalar("losses/target_max", next_q_value.max().item(), global_step)
                writer.add_scalar("losses/target_p95", next_q_value.quantile(0.95).item(), global_step)
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
