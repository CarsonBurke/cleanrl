# Pan Goal Solver v23: Forward-Backward successor representations — the
# occupancy quantity made load-bearing.
#
# v14-v22 all trained goal-followers by hindsight behavioral cloning, which
# has NO improvement operator: it can only retrieve behavior already in the
# data, and every version measured a dead or bias-token goal channel with
# the commanded-goal loop never closing. Two independent reviews (2026-07-17,
# recorded in FAMILY.md) traced the ceiling to the family's own "no
# action-value critic" clause and to Pan-1's successor-measure quantity
# being implemented but consumed by nothing.
#
# v23 pivots the family core to Forward-Backward representations (Touati &
# Ollivier 2021; Touati, Rapin & Ollivier 2023): the discounted successor
# measure is factored as M^pi_z(s,a, ds_f) ~ F(s,a,z)^T B(s_f) rho(ds_f).
# - F^T B IS Pan-1's central quantity #1 (goal occupancy), promoted from an
#   inert InfoNCE readout to the object everything runs on.
# - The improvement operator is the measure-Bellman backup (TD with target
#   networks, twin F's, clipped double targets) plus a deterministic actor
#   ascending on OCCUPANCY F(s, pi(s,z), z)^T z — reward never trains the
#   actor. FAMILY.md amended accordingly: the ban is on REWARD critics and
#   reward policy gradients; an action-conditioned successor critic honors
#   the family's reward-containment spirit.
# - The goal-target problem is solved exactly, replacing the entire
#   anchor/aspiration/ratchet/reward-head proposer stack:
#   z_task = sqrt(d) * normalize( E_replay[ r * B(s') ] ) — a dot product
#   against observed reward. Reward's ONLY entry point. Acting is one
#   forward pass a = pi(s, z_task); no search, no planning.
# - Collection is z-diverse by construction (fresh z per episode AND every
#   z_resample_steps within episodes, mixing sphere noise / B(replay state)
#   goal vectors / z_task): the policy is z-conditioned from the start, so
#   goal-directed data exists from the start — closing the
#   I(action; goal | belief) ~ 0 identifiability hole of v14-v22.
# - The LeJEPA world model is parked for this version (single-variable test
#   on a fully observable env; raw observations keep the critic's inputs
#   stationary under TD). B is the goal encoder now: a commanded
#   observation s_g maps to z = sqrt(d) * normalize(B(s_g)), which restores
#   Pan-style observation goals whenever wanted.
#
# Pre-registered 1.5M falsifiers (failure kills FB here, not just a tweak):
# 1. evaluation/task_return clearly above the family's ~1500-1800 ceiling
#    and still climbing.
# 2. diagnostics/z_conditioning_gap sustained > 0 (matched vs shuffled z).
# 3. B covariance ~ I and bounded F/B norms (no TD divergence, no collapse).

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
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    num_steps: int = 512
    replay_size: int = 1_000_000
    batch_size: int = 512
    learning_rate: float = 1e-4
    # Gradient updates per collected env step (TD3-style 1:1).
    updates_per_env_step: float = 1.0
    max_grad_norm: float = 1.0

    z_dim: int = 64
    f_hidden: int = 1024
    b_hidden: int = 256
    actor_hidden: int = 1024
    gamma: float = 0.98
    tau: float = 0.01
    ortho_coef: float = 1.0
    # Target-policy smoothing (TD3): noise on the bootstrap action only.
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise_std: float = 0.2

    # z mixtures. Training-z: task fraction first, remainder split between
    # uniform-sphere and B(replay state) goal vectors. Behavior-z: same
    # structure; resampled at episode reset and every z_resample_steps so
    # commands switch WITHIN episodes at Pan-like horizons.
    train_z_task_fraction: float = 0.2
    train_z_goal_fraction: float = 0.5
    behavior_z_task_fraction: float = 0.3
    behavior_z_goal_fraction: float = 0.4
    z_resample_steps: int = 256

    # Task inference: z_task = sqrt(d) * normalize(E[r * B(s')]) over this
    # many replay transitions, refreshed every iteration. Reward's only job.
    task_inference_samples: int = 4096

    random_warmup_steps: int = 25_000
    eval_interval: int = 100_000
    eval_envs: int = 4


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def xavier_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def mlp(sizes, out_gain=1.0):
    layers = [nn.LayerNorm(sizes[0])]
    for i in range(len(sizes) - 2):
        layers += [xavier_linear(nn.Linear(sizes[i], sizes[i + 1])), nn.SiLU()]
    layers.append(xavier_linear(nn.Linear(sizes[-2], sizes[-1]), gain=out_gain))
    return nn.Sequential(*layers)


class BackwardMap(nn.Module):
    """B(s): state -> R^d. F^T B is the goal-occupancy successor measure;
    B is also the goal encoder (z_g = sqrt(d) * normalize(B(s_g)))."""

    def __init__(self, obs_dim, z_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, z_dim])

    def forward(self, obs):
        return self.net(obs)


class ForwardMap(nn.Module):
    """Twin F(s,a,z) -> R^d each; min over the pair gives clipped double
    successor targets (TD3-style), the occupancy analogue of double-Q."""

    def __init__(self, obs_dim, act_dim, z_dim, hidden):
        super().__init__()
        in_dim = obs_dim + act_dim + z_dim
        self.f1 = mlp([in_dim, hidden, hidden, z_dim])
        self.f2 = mlp([in_dim, hidden, hidden, z_dim])

    def forward(self, obs, action, z):
        x = torch.cat([obs, action, z], dim=-1)
        return self.f1(x), self.f2(x)


class Actor(nn.Module):
    """pi(s, z): deterministic z-conditioned policy, trained by ascent on
    occupancy F(s, pi, z)^T z. Reward never appears in its objective."""

    def __init__(self, obs_dim, act_dim, z_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim + z_dim, hidden, hidden, act_dim], out_gain=0.05)

    def forward(self, obs, z):
        return torch.tanh(self.net(torch.cat([obs, z], dim=-1)))


def project_z(v, z_dim):
    """Scale to the sqrt(d)-radius sphere FB samples its tasks from."""
    return math.sqrt(z_dim) * F.normalize(v, dim=-1)


class FlatReplayBuffer:
    """Plain uniform transition buffer (per-env rings, no histories)."""

    def __init__(self, total_capacity, num_envs, obs_dim, act_dim):
        self.num_envs = num_envs
        self.capacity = max(2, total_capacity // num_envs)
        self.obs = np.zeros((num_envs, self.capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((num_envs, self.capacity, obs_dim), np.float32)
        self.action = np.zeros((num_envs, self.capacity, act_dim), np.float32)
        self.reward = np.zeros((num_envs, self.capacity), np.float32)
        self.terminal = np.zeros((num_envs, self.capacity), np.float32)
        self.total_steps = 0

    @property
    def size(self):
        return min(self.total_steps, self.capacity) * self.num_envs

    def add(self, obs, action, reward, next_obs, terminal):
        slot = self.total_steps % self.capacity
        self.obs[:, slot] = obs
        self.action[:, slot] = action
        self.reward[:, slot] = reward
        self.next_obs[:, slot] = next_obs
        self.terminal[:, slot] = terminal
        self.total_steps += 1

    def _indices(self, count, rng):
        steps = min(self.total_steps, self.capacity)
        if steps == 0:
            raise RuntimeError("replay buffer is empty")
        env = rng.integers(0, self.num_envs, size=count)
        slot = rng.integers(0, steps, size=count)
        return env, slot

    def sample(self, count, rng):
        env, slot = self._indices(count, rng)
        return {
            "obs": self.obs[env, slot],
            "action": self.action[env, slot],
            "reward": self.reward[env, slot],
            "next_obs": self.next_obs[env, slot],
            "terminal": self.terminal[env, slot],
        }

    def sample_states(self, count, rng):
        env, slot = self._indices(count, rng)
        return self.next_obs[env, slot]


def as_tensor(batch, device):
    return torch.as_tensor(batch, device=device, dtype=torch.float32)


def sample_z_mixture(
    backward,
    replay,
    count,
    z_dim,
    task_fraction,
    goal_fraction,
    z_task,
    rng,
    device,
):
    """Task z / B(replay state) goal z / uniform sphere z, in that priority.
    Goal-vector z's tie the task distribution to achieved states (FB's
    goal-based mix); sphere z's keep coverage of the whole task space."""
    z = project_z(torch.randn(count, z_dim, device=device), z_dim)
    draw = torch.as_tensor(rng.random(count), device=device, dtype=torch.float32)
    goal_states = as_tensor(replay.sample_states(count, rng), device)
    with torch.no_grad():
        goal_z = project_z(backward(goal_states), z_dim)
    task_cut = task_fraction if z_task is not None else 0.0
    goal_mask = (draw >= task_cut) & (draw < task_cut + goal_fraction)
    z = torch.where(goal_mask.unsqueeze(-1), goal_z, z)
    if z_task is not None:
        z = torch.where((draw < task_cut).unsqueeze(-1), z_task.unsqueeze(0), z)
    return z


def train_fb_step(
    forward_map,
    backward,
    actor,
    forward_target,
    backward_target,
    actor_target,
    fb_optimizer,
    actor_optimizer,
    batch,
    z,
    args,
    device,
):
    obs = as_tensor(batch["obs"], device)
    action = as_tensor(batch["action"], device)
    next_obs = as_tensor(batch["next_obs"], device)
    terminal = as_tensor(batch["terminal"], device)

    # Measure-Bellman backup with in-batch future states: rows are (s,a,z),
    # columns are candidate futures s_f = next_obs of the batch. Diagonal is
    # the on-transition positive.
    with torch.no_grad():
        noise = (
            torch.randn_like(action) * args.target_noise_std
        ).clamp(-args.target_noise_clip, args.target_noise_clip)
        next_action = (actor_target(next_obs, z) + noise).clamp(-1.0, 1.0)
        target_f1, target_f2 = forward_target(next_obs, next_action, z)
        target_b = backward_target(next_obs)
        target_m1 = target_f1 @ target_b.T
        target_m2 = target_f2 @ target_b.T
        target_m = torch.min(target_m1, target_m2)
        target_m = args.gamma * (1.0 - terminal).unsqueeze(-1) * target_m
    f1, f2 = forward_map(obs, action, z)
    b_next = backward(next_obs)
    m1 = f1 @ b_next.T
    m2 = f2 @ b_next.T
    fb_offdiag = 0.5 * ((m1 - target_m).square().mean() + (m2 - target_m).square().mean())
    fb_diag = -(m1.diagonal().mean() + m2.diagonal().mean())
    # Orthonormality: keeps B's embedding of the state distribution
    # near-isotropic — the FB analogue of the family's collapse guards.
    covariance = b_next.T @ b_next / b_next.shape[0]
    identity = torch.eye(args.z_dim, device=device)
    ortho_loss = (covariance - identity).square().sum()
    fb_loss = fb_offdiag + fb_diag + args.ortho_coef * ortho_loss
    fb_optimizer.zero_grad(set_to_none=True)
    fb_loss.backward()
    fb_grad = nn.utils.clip_grad_norm_(
        list(forward_map.parameters()) + list(backward.parameters()), args.max_grad_norm
    )
    fb_optimizer.step()

    # Actor: ascend occupancy of its own commanded z. Gradients flow through
    # F but only actor parameters update (separate optimizer); z detached.
    pi_action = actor(obs, z)
    q1, q2 = forward_map(obs, pi_action, z)
    q = torch.min((q1 * z).sum(-1), (q2 * z).sum(-1))
    actor_loss = -q.mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
    actor_optimizer.step()

    with torch.no_grad():
        for target, online in (
            (forward_target, forward_map),
            (backward_target, backward),
            (actor_target, actor),
        ):
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.lerp_(op, args.tau)
        # Successor ranking: does F^T B put the true next state above the
        # in-batch negatives? Chance = 1/batch.
        ranking_accuracy = (
            (m1.argmax(dim=-1) == torch.arange(m1.shape[0], device=device))
            .float()
            .mean()
        )
        cov_error = (covariance - identity).square().mean()
    return {
        "fb/loss": fb_loss.item(),
        "fb/offdiag": fb_offdiag.item(),
        "fb/diag_positive": -fb_diag.item() / 2.0,
        "fb/ortho": ortho_loss.item(),
        "fb/cov_error": cov_error.item(),
        "fb/ranking_accuracy": ranking_accuracy.item(),
        "fb/f_norm": f1.norm(dim=-1).mean().item(),
        "fb/b_norm": b_next.norm(dim=-1).mean().item(),
        "fb/grad_norm": float(fb_grad),
        "actor/loss": actor_loss.item(),
        "actor/occupancy_q": q.mean().item(),
        "actor/grad_norm": float(actor_grad),
    }


@torch.no_grad()
def infer_task_z(backward, replay, args, rng, device):
    """z_task = sqrt(d) * normalize(E[r * B(s')]). Reward's only entry
    point into the whole system."""
    batch = replay.sample(args.task_inference_samples, rng)
    states = as_tensor(batch["next_obs"], device)
    rewards = as_tensor(batch["reward"], device)
    z_raw = (rewards.unsqueeze(-1) * backward(states)).mean(dim=0)
    if float(z_raw.norm()) < 1e-8:
        return None
    return project_z(z_raw, args.z_dim)


@torch.no_grad()
def evaluate_policy(actor, args, device, run_name, z, seed_offset):
    """Deterministic rollout of pi(., z) over eval_envs episodes."""
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, run_name + "-eval") for index in range(args.eval_envs)]
    )
    obs, _ = envs.reset(seed=args.seed + seed_offset)
    obs = np.asarray(obs, np.float32)
    z_batch = z.unsqueeze(0).expand(args.eval_envs, -1)
    returns = []
    while len(returns) < args.eval_envs:
        action = actor(as_tensor(obs, device), z_batch)
        next_obs, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    returns.append(float(info["episode"]["r"]))
        obs = np.asarray(next_obs, np.float32)
    envs.close()
    return float(np.mean(returns[: args.eval_envs]))


def main():
    args = tyro.cli(Args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Pan Goal Solver training requires CUDA")
    for name, task_frac, goal_frac in (
        ("train", args.train_z_task_fraction, args.train_z_goal_fraction),
        ("behavior", args.behavior_z_task_fraction, args.behavior_z_goal_fraction),
    ):
        if not (0 <= task_frac and 0 <= goal_frac and task_frac + goal_frac <= 1):
            raise ValueError(f"{name} z mixture must be a sub-distribution")
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
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name) for index in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("only continuous Box actions are supported")
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    forward_map = ForwardMap(obs_dim, act_dim, args.z_dim, args.f_hidden).to(device)
    backward = BackwardMap(obs_dim, args.z_dim, args.b_hidden).to(device)
    actor = Actor(obs_dim, act_dim, args.z_dim, args.actor_hidden).to(device)
    forward_target = ForwardMap(obs_dim, act_dim, args.z_dim, args.f_hidden).to(device)
    backward_target = BackwardMap(obs_dim, args.z_dim, args.b_hidden).to(device)
    actor_target = Actor(obs_dim, act_dim, args.z_dim, args.actor_hidden).to(device)
    forward_target.load_state_dict(forward_map.state_dict())
    backward_target.load_state_dict(backward.state_dict())
    actor_target.load_state_dict(actor.state_dict())
    for module in (forward_target, backward_target, actor_target):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    fb_optimizer = torch.optim.Adam(
        list(forward_map.parameters()) + list(backward.parameters()), lr=args.learning_rate
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    replay = FlatReplayBuffer(args.replay_size, args.num_envs, obs_dim, act_dim)
    z_task = None

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = np.asarray(next_obs, np.float32)
    behavior_z = project_z(torch.randn(args.num_envs, args.z_dim, device=device), args.z_dim)
    steps_since_resample = np.zeros(args.num_envs, np.int64)

    def resample_behavior_z(indices):
        """Fresh commands: task / achieved-state goal / sphere, switching
        at episode resets AND on a within-episode timer."""
        if not len(indices):
            return
        count = len(indices)
        fresh = project_z(torch.randn(count, args.z_dim, device=device), args.z_dim)
        draw = rng.random(count)
        task_cut = args.behavior_z_task_fraction if z_task is not None else 0.0
        if replay.total_steps > 0:
            goal_states = as_tensor(replay.sample_states(count, rng), device)
            with torch.no_grad():
                goal_z = project_z(backward(goal_states), args.z_dim)
            goal_mask = torch.as_tensor(
                (draw >= task_cut) & (draw < task_cut + args.behavior_z_goal_fraction),
                device=device,
            )
            fresh = torch.where(goal_mask.unsqueeze(-1), goal_z, fresh)
        if z_task is not None:
            task_mask = torch.as_tensor(draw < task_cut, device=device)
            fresh = torch.where(task_mask.unsqueeze(-1), z_task.unsqueeze(0), fresh)
        behavior_z[torch.as_tensor(indices, device=device)] = fresh
        steps_since_resample[indices] = 0

    updates_per_iteration = int(
        args.updates_per_env_step * args.num_envs * args.num_steps
    )
    global_step = 0
    start_time = time.time()
    next_eval_step = args.eval_interval
    num_iterations = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    for iteration in range(1, num_iterations + 1):
        rollout_steps = min(
            args.num_steps, (args.total_timesteps - global_step) // args.num_envs
        )
        for _ in range(rollout_steps):
            if global_step < args.random_warmup_steps:
                action = rng.uniform(-1, 1, size=(args.num_envs, act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    action = actor(as_tensor(next_obs, device), behavior_z).cpu().numpy()
                action = np.clip(
                    action
                    + args.exploration_noise_std
                    * rng.standard_normal(action.shape).astype(np.float32),
                    -1,
                    1,
                )
            step_obs, reward, terminations, truncations, infos = envs.step(action)
            done = np.logical_or(terminations, truncations)
            step_obs = np.asarray(step_obs, np.float32)
            # Bootstrap through truncations; cut only on true terminations.
            real_next = step_obs.copy()
            if "final_observation" in infos:
                for env_index, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        real_next[env_index] = np.asarray(final_obs, np.float32)
            replay.add(next_obs, action, reward, real_next, terminations.astype(np.float32))
            global_step += args.num_envs
            steps_since_resample += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episodic_return:.3f}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar(
                            "charts/episodic_length", float(info["episode"]["l"]), global_step
                        )
            next_obs = step_obs
            refresh = done | (steps_since_resample >= args.z_resample_steps)
            if refresh.any():
                resample_behavior_z(np.flatnonzero(refresh))

        metrics = []
        if global_step >= args.random_warmup_steps and replay.size >= args.batch_size * 2:
            for _ in range(updates_per_iteration):
                batch = replay.sample(args.batch_size, rng)
                z = sample_z_mixture(
                    backward,
                    replay,
                    args.batch_size,
                    args.z_dim,
                    args.train_z_task_fraction,
                    args.train_z_goal_fraction,
                    z_task,
                    rng,
                    device,
                )
                metrics.append(
                    train_fb_step(
                        forward_map,
                        backward,
                        actor,
                        forward_target,
                        backward_target,
                        actor_target,
                        fb_optimizer,
                        actor_optimizer,
                        batch,
                        z,
                        args,
                        device,
                    )
                )
            previous_z_task = z_task
            z_task = infer_task_z(backward, replay, args, rng, device)
            if z_task is not None:
                writer.add_scalar("task/z_norm_sanity", float(z_task.norm()), global_step)
                if previous_z_task is not None:
                    writer.add_scalar(
                        "task/z_drift_cosine",
                        float(F.cosine_similarity(z_task, previous_z_task, dim=0)),
                        global_step,
                    )

        if metrics:
            aggregated = {}
            for row in metrics:
                for key, value in row.items():
                    aggregated.setdefault(key, []).append(value)
            for key, values in aggregated.items():
                writer.add_scalar(key, float(np.mean(values)), global_step)
            # z-conditioning: the FB analogue of the family's goal-removal
            # test. Dead (=0) means the actor collapsed to z-independence —
            # the v14-v22 failure signature.
            with torch.no_grad():
                probe = as_tensor(replay.sample(args.batch_size, rng)["obs"], device)
                probe_z = sample_z_mixture(
                    backward, replay, args.batch_size, args.z_dim,
                    0.0, args.train_z_goal_fraction, None, rng, device,
                )
                matched = actor(probe, probe_z)
                shuffled = actor(probe, probe_z.roll(1, dims=0))
                writer.add_scalar(
                    "diagnostics/z_conditioning_gap",
                    (matched - shuffled).square().mean().item(),
                    global_step,
                )

        if args.eval_interval > 0 and global_step >= next_eval_step and z_task is not None:
            task_return = evaluate_policy(actor, args, device, run_name, z_task, 10_000)
            writer.add_scalar("evaluation/task_return", task_return, global_step)
            # Control arm: a random command. Task inference works iff the
            # task arm beats this decisively.
            random_z = project_z(torch.randn(args.z_dim, device=device), args.z_dim)
            random_return = evaluate_policy(actor, args, device, run_name, random_z, 10_000)
            writer.add_scalar("evaluation/random_z_return", random_return, global_step)
            writer.add_scalar(
                "evaluation/task_minus_random", task_return - random_return, global_step
            )
            next_eval_step += args.eval_interval

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/replay_size", replay.size, global_step)
        print(
            f"iteration={iteration}/{num_iterations}, step={global_step}, "
            f"SPS={sps}, replay={replay.size}"
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "forward": forward_map.state_dict(),
                "backward": backward.state_dict(),
                "actor": actor.state_dict(),
                "args": vars(args),
            },
            model_path,
        )
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
