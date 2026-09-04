# PMPO-D4 + Beta + ReluSq + SPO asym (half-strength) + FIRE
#
# Base: pmpo_d4_beta_relusq_spo_asym_v1, but with the SPO penalty bounds
# DOUBLED — ε_low=0.40, ε_high=0.56 — which halves the effective penalty
# strength (penalty = |A|·(r−1)²/(2ε) shrinks as ε grows). This matches the
# "halfstrength" sibling that ran on this codebase previously.
#
# Addition: FIRE re-orthogonalization at every 1M global_step. Each Linear
# weight is replaced with its nearest semi-orthogonal matrix via Newton-Schulz
# (Muon-style sqrt(d_out/d_in) scaling). Adam state is rebuilt at each event
# so stale moments do not immediately undo the reinit.
#
# Hypothesis: the half-strength SPO trust region permits larger per-step
# drift, which over an 8M horizon may degrade plasticity / effective rank
# faster than tighter SPO. FIRE every 1M restores isometry repeatedly,
# letting later updates keep finding new directions instead of saturating.
#
# Reference: https://isaac7778.github.io/fire/  (ICLR 2026)

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

SAMPLE_EPS = 1e-7


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
    clip_vloss: bool = True
    clip_coef: float = 0.2
    spo_eps_low: float = 0.40
    """SPO penalty bound — half-strength (2x baseline 0.20)"""
    spo_eps_high: float = 0.56
    """SPO penalty bound — half-strength (2x baseline 0.28)"""
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # FIRE knobs — reinit every 1M global_step from 1M..7M (8M horizon)
    fire_steps: tuple[int, ...] = (
        1_000_000, 2_000_000, 3_000_000, 4_000_000,
        5_000_000, 6_000_000, 7_000_000,
    )
    fire_ns_iters: int = 10

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


class ReluSq(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def newton_schulz(matrix: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """Quintic-free Newton-Schulz iteration for nearest semi-orthogonal matrix.
    Coefficients (1.5, -0.5) match the original FIRE reference implementation."""
    assert matrix.ndim == 2
    a, b = 1.5, -0.5
    do_transpose = matrix.size(1) > matrix.size(0)
    X = matrix.T if do_transpose else matrix
    X = X / (X.norm() + 1e-8)
    for _ in range(num_iters):
        A = X.T @ X
        X = a * X + b * X @ A
    return X.T if do_transpose else X


@torch.no_grad()
def fire_reinit(model: nn.Module, num_iters: int = 10):
    """Replace each Linear weight with its nearest semi-orthogonal matrix
    (Newton-Schulz approx), scaled Muon-style by sqrt(d_out / d_in).
    Biases are left untouched."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            d_out, d_in = W.shape
            ortho = newton_schulz(W.detach().clone(), num_iters=num_iters)
            scale = (d_out / d_in) ** 0.5
            m.weight.data.copy_(ortho * scale)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 2 * action_dim), std=0.01),
        )
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        head = self.actor(x)
        head_alpha, head_beta = head.chunk(2, dim=-1)
        alpha = 1.0 + nn.functional.softplus(head_alpha)
        beta = 1.0 + nn.functional.softplus(head_beta)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        return ((action - self.action_low) / (self.action_high - self.action_low)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, action=None):
        dist = self.get_action_distribution(x)
        if action is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        log_prob = dist.log_prob(z).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(x), dist


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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    fire_pending = sorted(set(int(s) for s in args.fire_steps))

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
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
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
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
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        spo_penalty_mean = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

        # FIRE: re-orthogonalize linear weights at the configured global_step thresholds
        # and reset the Adam optimizer state so momentum from the pre-FIRE trajectory
        # does not immediately overwrite the new weights.
        while fire_pending and global_step >= fire_pending[0]:
            threshold = fire_pending.pop(0)
            print(f"[FIRE] global_step={global_step} reinit at threshold={threshold}")
            fire_reinit(agent, num_iters=args.fire_ns_iters)
            optimizer = optim.Adam(agent.parameters(), lr=optimizer.param_groups[0]["lr"], eps=1e-5)
            writer.add_scalar("fire/applied_at_step", threshold, global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        with torch.no_grad():
            dist_diag = agent.get_action_distribution(b_obs)
            mean_alpha = dist_diag.concentration1.mean().item()
            mean_beta = dist_diag.concentration0.mean().item()
            sum_conc = (dist_diag.concentration1 + dist_diag.concentration0).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("diag/spo_penalty", spo_penalty_mean.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/beta_mean_alpha", mean_alpha, global_step)
        writer.add_scalar("diag/beta_mean_beta", mean_beta, global_step)
        writer.add_scalar("diag/beta_concentration_sum", sum_conc, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
