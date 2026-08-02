# Embedding Optimization: Command Distillation v1
#
# Hypothesis: exact factual command outcomes plus achieved-endpoint path
# distillation can turn an online LeJEPA chart into a useful control interface
# without PPO, Q-values, TD/bootstrap targets, EMA, or fixed command horizons.
# Three bootstrapped command-value/proposer heads provide epistemic exploration.
# Projector-only geometry updates are gauge-fixed by Procrustes alignment before
# the isolated value, proposer, and direct-controller phases.
import math
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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    num_steps: int = 128
    replay_iters: int = 64
    recent_iters: int = 9
    minibatch_size: int = 512

    latent_dim: int = 64
    goal_dim: int = 16
    hidden_dim: int = 256
    value_heads: int = 3

    wm_lr: float = 1e-4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    wm_updates: int = 16
    geometry_updates: int = 8
    value_updates: int = 16
    controller_updates: int = 16
    proposer_updates: int = 8
    bootstrap_probability: float = 0.5
    sigreg_coef: float = 0.09
    path_weight_temperature: float = 1.0
    path_weight_low: float = 0.25
    path_weight_high: float = 1.75

    noise_rho: float = 0.95
    expl_noise: float = 0.2
    goal_noise: float = 0.1
    ghost_goal_radius: float = 0.1
    warmup_steps: int = 10_000
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden_dim, out_dim, out_std=1.0, layer_norm=False):
    layers = [layer_init(nn.Linear(in_dim, hidden_dim))]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.extend([nn.SiLU(), layer_init(nn.Linear(hidden_dim, hidden_dim))])
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.extend([nn.SiLU(), layer_init(nn.Linear(hidden_dim, out_dim), std=out_std)])
    return nn.Sequential(*layers)


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularizer used by the LeJEPA lineage."""

    def __init__(self, projections=256, knots=17, reference_samples=128):
        super().__init__()
        self.projections = projections
        self.reference_samples = reference_samples
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        quadrature = torch.full((knots,), 2 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", quadrature * gaussian_cf)

    def forward(self, x):
        axes = torch.randn(
            x.shape[-1], self.projections, device=x.device, dtype=x.dtype
        )
        axes = axes / axes.norm(dim=0, keepdim=True).clamp_min(1e-8)
        phase = (x @ axes).unsqueeze(-1) * self.t
        error = (
            (phase.cos().mean(0) - self.gaussian_cf).square()
            + phase.sin().mean(0).square()
        )
        return ((error @ self.weights) * self.reference_samples).mean()


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        dz, dy, hidden = args.latent_dim, args.goal_dim, args.hidden_dim

        self.encoder = mlp(obs_dim, hidden, dz, layer_norm=True)
        self.dynamics = mlp(
            dz + action_dim, hidden, dz, out_std=0.01, layer_norm=True
        )
        # A linear projector keeps SIGReg gauge drift predominantly orthogonal,
        # which the explicit Procrustes alignment can correct. A nonlinear
        # projector could silently warp persistent goal coordinates between
        # updates even when its marginal distribution remained Gaussian.
        self.goal_projector = nn.Linear(dz, dy, bias=False)
        nn.init.orthogonal_(self.goal_projector.weight)
        self.command_value_heads = nn.ModuleList(
            [mlp(3 * dy, hidden, 1) for _ in range(args.value_heads)]
        )
        self.proposer_heads = nn.ModuleList(
            [mlp(dy, hidden, dy) for _ in range(args.value_heads)]
        )
        self.controller = mlp(
            dz + 2 * dy, hidden, action_dim, out_std=0.01
        )
        self.register_buffer("goal_alignment", torch.eye(dy))
        self.goal_dim = dy
        self.value_heads = args.value_heads

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        return z + self.dynamics(torch.cat([z, action], dim=-1))

    def goal_encode(self, z):
        # Only the projector's SIGReg phase may change the goal chart.
        return self.goal_projector(z.detach()) @ self.goal_alignment

    @staticmethod
    def command_features(y, goal, detach=True):
        if detach:
            y, goal = y.detach(), goal.detach()
        return torch.cat([y, goal, goal - y], dim=-1)

    def command_values(self, y, goal, detach_inputs=True):
        features = self.command_features(y, goal, detach=detach_inputs)
        return torch.stack(
            [head(features).squeeze(-1) for head in self.command_value_heads],
            dim=-1,
        )

    def propose_all(self, y):
        y = y.detach()
        proposals = torch.stack([head(y) for head in self.proposer_heads], dim=1)
        return (
            math.sqrt(self.goal_dim)
            * proposals
            / proposals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )

    def propose(self, y, head=None):
        proposals = self.propose_all(y)
        if head is None:
            return proposals
        return gather_head(proposals, head)

    def control(self, z, y, goal):
        # Controller distillation cannot train LeJEPA, geometry, or proposer.
        z, y, goal = z.detach(), y.detach(), goal.detach()
        return torch.tanh(
            self.controller(torch.cat([z, goal, goal - y], dim=-1))
        )


def gather_head(values, head):
    """Select one bootstrap head per leading batch element."""
    index = head.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(*head.shape, 1, values.shape[-1])
    return values.gather(-2, index).squeeze(-2)


def gather_value_head(values, head):
    return values.gather(-1, head.unsqueeze(-1)).squeeze(-1)


def completed_command_suffix_targets(reward, done, switched):
    """Return factual reward/step for completed exact-command suffixes.

    A command completes only at a later switch or episode boundary. The unfinished
    command at the right data edge is censored. No success/support filter is used,
    so failed and off-support commands remain training examples.
    """

    steps, envs = reward.shape
    if done.shape != (steps + 1, envs) or switched.shape != (steps, envs):
        raise ValueError("expected reward/switch [T,E] and done [T+1,E]")
    rate = torch.zeros_like(reward)
    length = torch.zeros_like(reward)
    valid = torch.zeros_like(reward)
    reward_suffix = torch.zeros(envs, device=reward.device, dtype=reward.dtype)
    suffix_steps = torch.zeros_like(reward_suffix)
    has_completion = torch.zeros(envs, device=reward.device, dtype=torch.bool)
    for step in range(steps - 1, -1, -1):
        boundary = done[step + 1].bool()
        if step + 1 < steps:
            boundary = boundary | switched[step + 1].bool()
        reward_suffix = torch.where(
            boundary, reward[step], reward[step] + reward_suffix
        )
        suffix_steps = torch.where(
            boundary, torch.ones_like(suffix_steps), suffix_steps + 1
        )
        has_completion = boundary | has_completion
        rate[step] = reward_suffix / suffix_steps
        length[step] = suffix_steps
        valid[step] = has_completion.to(reward.dtype)
    return rate, length, valid


def bootstrap_mask(batch_size, heads, probability, device, generator=None):
    """Independent bootstrap membership, with at least one head per example."""

    if not 0 < probability <= 1:
        raise ValueError("bootstrap probability must lie in (0, 1]")
    mask = (
        torch.rand(
            batch_size,
            heads,
            device=device,
            generator=generator,
        )
        < probability
    )
    uncovered = ~mask.any(dim=-1)
    fallback = torch.randint(
        heads,
        (batch_size,),
        device=device,
        generator=generator,
    )
    fallback_membership = nn.functional.one_hot(fallback, heads).bool()
    mask |= uncovered.unsqueeze(-1) & fallback_membership
    return mask


def leave_one_out(values, excluded_head):
    """Return evaluator heads excluding the head that generated a proposal."""

    heads = values.shape[-1]
    keep = torch.arange(heads, device=values.device).unsqueeze(0)
    keep = keep != excluded_head.unsqueeze(-1)
    return values[keep].reshape(*values.shape[:-1], heads - 1)


def evidence_goal_switch(
    current_values,
    candidate_values,
    valid,
    error_floor=0.0,
):
    """Switch only when the ensemble credibly prefers the candidate.

    Invalid goals always initialize, ensuring unconstrained proposals are tried.
    Once a command is active, its duration is endogenous: no age, refresh period,
    or fixed score margin can end it. Requiring the candidate's lower estimate to
    exceed the incumbent's upper estimate prevents proposer optimization from
    turning infinitesimal score gains into one-step commands.
    """

    error_floor = torch.as_tensor(
        error_floor,
        device=current_values.device,
        dtype=current_values.dtype,
    )
    current_mean = current_values.mean(dim=-1)
    candidate_mean = candidate_values.mean(dim=-1)
    current_std = current_values.std(dim=-1, unbiased=False)
    candidate_std = candidate_values.std(dim=-1, unbiased=False)
    current_error = (current_std.square() + error_floor.square()).sqrt()
    candidate_error = (candidate_std.square() + error_floor.square()).sqrt()
    challenger = candidate_mean - candidate_error > current_mean + current_error
    switch = (~valid.bool()) | challenger
    return (
        switch,
        challenger & valid.bool(),
        current_mean,
        candidate_mean,
    )


def path_weights(
    reward_rate,
    valid,
    temperature,
    lower=0.25,
    upper=1.75,
):
    """Bounded monotone weights with exactly unit mean over valid paths."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= lower < 1 < upper:
        raise ValueError("weight bounds must straddle one")
    valid_bool = valid.bool()
    weights = torch.zeros_like(reward_rate)
    if not valid_bool.any():
        return weights
    valid_rate = reward_rate[valid_bool]
    standardized = (valid_rate - valid_rate.mean()) / valid_rate.std(
        unbiased=False
    ).clamp_min(1e-6)
    # Relative standardization keeps selection pressure alive when locomotion
    # moves from negative reward rates to 10+ reward/step. A sigmoid of the raw
    # rate would saturate precisely in the high-performance regime.
    score = torch.tanh(standardized / temperature)
    centered = score - score.mean()
    scale = centered.abs().max().clamp_min(1e-8)
    amplitude = min(1.0 - lower, upper - 1.0)
    normalized = 1.0 + amplitude * centered / scale
    weights[valid_bool] = normalized
    return weights


def sample_path_endpoints(trajectory_steps, batch_size, device, generator=None):
    """Sample unordered state pairs, then orient them forward in time."""

    endpoints = torch.randint(
        trajectory_steps + 1,
        (2, batch_size),
        device=device,
        generator=generator,
    )
    start = endpoints.min(dim=0).values
    end = endpoints.max(dim=0).values
    equal = start == end
    start = torch.where(equal & (start == trajectory_steps), start - 1, start)
    end = torch.where(equal, start + 1, end)
    return start, end


def procrustes_alignment(raw_new, global_old):
    """Orthogonal map from a new raw chart to persistent global coordinates."""

    if raw_new.ndim != 2 or raw_new.shape != global_old.shape:
        raise ValueError("anchor matrices must have the same [samples, goal_dim] shape")
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def participation_rank(x):
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def masked_correlation(x, y, mask):
    mask = mask.bool()
    if mask.sum() < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x, y = x[mask], y[mask]
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (
        x.square().mean().sqrt() * y.square().mean().sqrt()
    ).clamp_min(1e-8)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.cuda and torch.cuda.is_available(), "this experiment requires CUDA"
    assert args.value_heads == 3, "command-distill v1 is defined with three heads"
    assert 2 <= args.recent_iters <= args.replay_iters
    assert (
        args.recent_iters * args.num_steps >= 1000 + args.num_steps - 1
    ), "recent replay must retain any full target-environment episode"

    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
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
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    T, E, R = args.num_steps, args.num_envs, args.replay_iters
    dz, dy = args.latent_dim, args.goal_dim

    agent = Agent(envs, args).to(device)
    z_sigreg = SIGReg().to(device)
    y_sigreg = SIGReg().to(device)
    wm_params = list(agent.encoder.parameters()) + list(agent.dynamics.parameters())
    geometry_params = list(agent.goal_projector.parameters())
    value_params = list(agent.command_value_heads.parameters())
    proposer_params = list(agent.proposer_heads.parameters())
    controller_params = list(agent.controller.parameters())
    wm_optimizer = optim.AdamW(
        wm_params,
        lr=args.wm_lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    geometry_optimizer = optim.Adam(
        geometry_params, lr=args.learning_rate, eps=1e-5
    )
    value_optimizer = optim.Adam(value_params, lr=args.learning_rate, eps=1e-5)
    proposer_optimizer = optim.Adam(
        proposer_params, lr=args.learning_rate, eps=1e-5
    )
    controller_optimizer = optim.Adam(
        controller_params, lr=args.learning_rate, eps=1e-5
    )
    optimizers = (
        (wm_optimizer, args.wm_lr),
        (geometry_optimizer, args.learning_rate),
        (value_optimizer, args.learning_rate),
        (proposer_optimizer, args.learning_rate),
        (controller_optimizer, args.learning_rate),
    )

    def wm_loss_fn(obs, action, next_obs, valid):
        z = agent.encode(obs)
        next_z = agent.encode(next_obs)
        prediction = agent.predict_next(z, action)
        prediction_loss = masked_mean(
            (prediction - next_z).square().mean(dim=-1), valid
        )
        sigreg_loss = z_sigreg(z)
        return (
            prediction_loss + args.sigreg_coef * sigreg_loss,
            prediction_loss.detach(),
            sigreg_loss.detach(),
        )

    def geometry_loss_fn(z):
        y = agent.goal_encode(z)
        return args.sigreg_coef * y_sigreg(y)

    if args.compile:
        wm_loss_fn = torch.compile(
            wm_loss_fn, mode=args.compile_mode, dynamic=False
        )
        geometry_loss_fn = torch.compile(
            geometry_loss_fn, mode=args.compile_mode, dynamic=False
        )
        print(
            "[embopt_command_distill_v1] "
            f"torch.compile(mode={args.compile_mode!r}, dynamic=False)"
        )

    def mark_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

    buffer_obs = torch.zeros((R, T + 1, E, obs_dim), device=device)
    buffer_action = torch.zeros((R, T, E, action_dim), device=device)
    buffer_reward = torch.zeros((R, T, E), device=device)
    buffer_done = torch.zeros((R, T + 1, E), device=device)
    buffer_y = torch.zeros((R, T + 1, E, dy), device=device)
    buffer_goal = torch.zeros((R, T, E, dy), device=device)
    buffer_switch = torch.zeros((R, T, E), device=device)
    buffer_bootstrap = torch.zeros(
        (R, T, E, args.value_heads), dtype=torch.bool, device=device
    )
    buffer_proposal_score = torch.zeros((R, T, E), device=device)
    buffer_proposal_disagreement = torch.zeros((R, T, E), device=device)
    buffer_age = torch.zeros((R, T, E), device=device)
    filled = buffer_pointer = 0

    next_obs_numpy, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(
        next_obs_numpy, dtype=torch.float32, device=device
    )
    next_done = torch.zeros(E, device=device)
    active_goal = torch.zeros((E, dy), device=device)
    active_head = torch.zeros(E, dtype=torch.long, device=device)
    active_bootstrap = torch.ones(
        E, args.value_heads, dtype=torch.bool, device=device
    )
    goal_valid = torch.zeros(E, dtype=torch.bool, device=device)
    action_noise = torch.zeros(E, action_dim, device=device)
    command_age = torch.zeros(E, device=device)
    # Before the first completed causal commands exist, only invalid/reset goals
    # may initialize. Thereafter this becomes the current out-of-bootstrap value
    # error, learned afresh from replay rather than annealed or EMA-smoothed.
    switch_error_floor = torch.full((), float("inf"), device=device)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            for optimizer, base_learning_rate in optimizers:
                optimizer.param_groups[0]["lr"] = (
                    fraction * base_learning_rate
                )

        slot = buffer_pointer
        rollout_switches = torch.zeros((), device=device)
        rollout_challengers = torch.zeros((), device=device)
        rollout_ghosts = torch.zeros((), device=device)
        rollout_saturation = torch.zeros((), device=device)
        rollout_head_usage = torch.zeros(
            args.value_heads, device=device
        )
        for step in range(T):
            buffer_obs[slot, step] = next_obs
            buffer_done[slot, step] = next_done
            goal_valid &= ~next_done.bool()

            with torch.no_grad():
                z = agent.encode(next_obs)
                y = agent.goal_encode(z)
                # Each prospective command draws one posterior head. If accepted,
                # that head is retained as the selected command's identity.
                proposal_head = torch.randint(
                    args.value_heads, (E,), device=device
                )
                candidate_goal = agent.propose(y, proposal_head)
                candidate_goal = candidate_goal + (
                    args.goal_noise * torch.randn_like(candidate_goal)
                )
                candidate_goal = (
                    math.sqrt(dy)
                    * candidate_goal
                    / candidate_goal.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                )
                current_values = agent.command_values(y, active_goal)
                candidate_values = agent.command_values(y, candidate_goal)
                evaluator_current = leave_one_out(
                    current_values, active_head
                )
                evaluator_candidate = leave_one_out(
                    candidate_values, proposal_head
                )
                switched, challenger, current_score, candidate_score = (
                    evidence_goal_switch(
                        evaluator_current,
                        evaluator_candidate,
                        goal_valid,
                        switch_error_floor,
                    )
                )
                ghost = challenger & goal_valid & (
                    (candidate_goal - active_goal).norm(dim=-1)
                    < args.ghost_goal_radius
                )
                # Coordinate-equivalent proposals are not new causal commands.
                # Rejecting them preserves command membership and prevents
                # epsilon-scale value differences from manufacturing switches.
                switched = switched & ~ghost
                challenger = challenger & ~ghost
                active_goal = torch.where(
                    switched.unsqueeze(-1), candidate_goal, active_goal
                )
                active_head = torch.where(
                    switched, proposal_head, active_head
                )
                candidate_bootstrap = bootstrap_mask(
                    E,
                    args.value_heads,
                    args.bootstrap_probability,
                    device,
                )
                active_bootstrap = torch.where(
                    switched.unsqueeze(-1),
                    candidate_bootstrap,
                    active_bootstrap,
                )
                chosen_score = torch.where(
                    switched, candidate_score, current_score
                )
                action_noise = torch.where(
                    (switched | next_done.bool()).unsqueeze(-1),
                    torch.zeros_like(action_noise),
                    action_noise,
                )
                action = agent.control(z, y, active_goal)
                action_noise = (
                    args.noise_rho * action_noise
                    + math.sqrt(1 - args.noise_rho**2)
                    * torch.randn_like(action_noise)
                )
                action = (
                    action + args.expl_noise * action_noise
                ).clamp(-1, 1)
                if global_step < args.warmup_steps:
                    action = torch.empty_like(action).uniform_(-1, 1)

                goal_valid.fill_(True)
                command_age = torch.where(
                    switched, torch.zeros_like(command_age), command_age + 1
                )
                buffer_y[slot, step] = y
                buffer_goal[slot, step] = active_goal
                buffer_switch[slot, step] = switched.float()
                # Bootstrap membership belongs to the command, not the timestep.
                # All suffix labels from one causal experiment retain the same
                # membership so heads do not converge by eventually seeing every
                # fragment of every command.
                buffer_bootstrap[slot, step] = active_bootstrap
                buffer_proposal_score[slot, step] = chosen_score
                buffer_proposal_disagreement[slot, step] = (
                    candidate_values.std(dim=-1, unbiased=False)
                )
                buffer_age[slot, step] = command_age
                rollout_switches += switched.float().mean()
                rollout_challengers += challenger.float().mean()
                rollout_ghosts += ghost.float().mean()
                rollout_saturation += (
                    (action.abs() > 0.95).float().mean()
                )
                rollout_head_usage += torch.bincount(
                    active_head, minlength=args.value_heads
                )

            buffer_action[slot, step] = action
            global_step += E
            (
                next_obs_numpy,
                reward_numpy,
                terminations,
                truncations,
                infos,
            ) = envs.step(action.cpu().numpy())
            buffer_reward[slot, step] = torch.as_tensor(
                reward_numpy, dtype=torch.float32, device=device
            )
            next_obs = torch.as_tensor(
                next_obs_numpy, dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(
                np.logical_or(terminations, truncations),
                dtype=torch.float32,
                device=device,
            )
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )

        buffer_obs[slot, T] = next_obs
        buffer_done[slot, T] = next_done
        with torch.no_grad():
            buffer_y[slot, T] = agent.goal_encode(agent.encode(next_obs))

        filled = min(filled + 1, R)
        fresh_slot = buffer_pointer
        buffer_pointer = (buffer_pointer + 1) % R
        recent_count = min(args.recent_iters, filled)
        recent_newest = (
            torch.arange(
                fresh_slot,
                fresh_slot - recent_count,
                -1,
                device=device,
            )
            % R
        )
        recent_slots = recent_newest.flip(0)
        sequence_steps = recent_count * T
        obs_sequence = buffer_obs[recent_slots, :T].reshape(
            sequence_steps, E, obs_dim
        )
        obs_sequence = torch.cat(
            [obs_sequence, buffer_obs[fresh_slot, T].unsqueeze(0)], dim=0
        )
        action_sequence = buffer_action[recent_slots].reshape(
            sequence_steps, E, action_dim
        )
        reward_sequence = buffer_reward[recent_slots].reshape(
            sequence_steps, E
        )
        done_sequence = buffer_done[recent_slots, :T].reshape(
            sequence_steps, E
        )
        done_sequence = torch.cat(
            [done_sequence, buffer_done[fresh_slot, T].unsqueeze(0)], dim=0
        )
        goal_sequence = buffer_goal[recent_slots].reshape(
            sequence_steps, E, dy
        )
        switch_sequence = buffer_switch[recent_slots].reshape(
            sequence_steps, E
        )
        bootstrap_sequence = buffer_bootstrap[recent_slots].reshape(
            sequence_steps, E, args.value_heads
        )
        proposal_score_sequence = buffer_proposal_score[recent_slots].reshape(
            sequence_steps, E
        )
        proposal_disagreement_sequence = buffer_proposal_disagreement[
            recent_slots
        ].reshape(sequence_steps, E)
        episode_id = done_sequence.cumsum(dim=0)
        reward_prefix = torch.cat(
            [
                torch.zeros((1, E), device=device),
                reward_sequence.cumsum(dim=0),
            ],
            dim=0,
        )

        # Preserve the chart seen by rollout goals and replay before either
        # online LeJEPA or the projector changes its input/output coordinates.
        anchor_pool = buffer_obs[:filled, :T].reshape(-1, obs_dim)
        anchor_count = min(2048, anchor_pool.shape[0])
        anchor_index = torch.randperm(
            anchor_pool.shape[0], device=device
        )[:anchor_count]
        anchor_obs = anchor_pool[anchor_index]
        with torch.no_grad():
            anchor_global_before = agent.goal_encode(
                agent.encode(anchor_obs)
            )
            old_alignment = agent.goal_alignment.clone()

        wm_statistics = []
        for _ in range(args.wm_updates):
            flat_index = torch.randint(
                0, filled * T * E, (args.minibatch_size,), device=device
            )
            replay_slot = flat_index // (T * E)
            replay_step = (flat_index // E) % T
            env_index = flat_index % E
            replay_obs = buffer_obs[
                replay_slot, replay_step, env_index
            ]
            replay_next_obs = buffer_obs[
                replay_slot, replay_step + 1, env_index
            ]
            replay_action = buffer_action[
                replay_slot, replay_step, env_index
            ]
            replay_valid = (
                1.0
                - buffer_done[
                    replay_slot, replay_step + 1, env_index
                ]
            )
            mark_step()
            wm_loss, prediction_loss, z_sigreg_loss = wm_loss_fn(
                replay_obs,
                replay_action,
                replay_next_obs,
                replay_valid,
            )
            wm_optimizer.zero_grad(set_to_none=True)
            wm_loss.backward(inputs=wm_params)
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_optimizer.step()
            wm_statistics.append(
                (prediction_loss.item(), z_sigreg_loss.item())
            )

        geometry_statistics = []
        for _ in range(args.geometry_updates):
            time_index = torch.randint(
                sequence_steps + 1,
                (args.minibatch_size,),
                device=device,
            )
            env_index = torch.randint(
                E, (args.minibatch_size,), device=device
            )
            with torch.no_grad():
                z = agent.encode(obs_sequence[time_index, env_index])
            mark_step()
            geometry_loss = geometry_loss_fn(z)
            geometry_optimizer.zero_grad(set_to_none=True)
            geometry_loss.backward(inputs=geometry_params)
            nn.utils.clip_grad_norm_(
                geometry_params, args.max_grad_norm
            )
            geometry_optimizer.step()
            geometry_statistics.append(geometry_loss.item())

        with torch.no_grad():
            anchor_z_after = agent.encode(anchor_obs)
            raw_after = agent.goal_projector(anchor_z_after.detach())
            unaligned_after = raw_after @ old_alignment
            frame_drift = (
                unaligned_after - anchor_global_before
            ).square().mean()
            fit_count = max(anchor_count // 2, dy)
            alignment = procrustes_alignment(
                raw_after[:fit_count],
                anchor_global_before[:fit_count],
            )
            agent.goal_alignment.copy_(alignment)
            aligned_after = raw_after @ agent.goal_alignment
            heldout = slice(fit_count, None)
            frame_residual = (
                aligned_after[heldout] - anchor_global_before[heldout]
            ).square().mean()

            # Re-encode replay states in the newly aligned global chart. Stored
            # commanded goals remain meaningful global coordinates, but cached
            # state embeddings would retain avoidable nonlinear encoder drift.
            current_z_sequence = agent.encode(
                obs_sequence.reshape(-1, obs_dim)
            ).reshape(sequence_steps + 1, E, dz)
            current_y_sequence = agent.goal_encode(
                current_z_sequence.reshape(-1, dz)
            ).reshape(sequence_steps + 1, E, dy)

        # Command labels span rollout boundaries. Only the final incomplete
        # command is censored; every completed failure remains eligible.
        command_rate, command_length, command_valid = (
            completed_command_suffix_targets(
                reward_sequence,
                done_sequence,
                switch_sequence,
            )
        )
        flat_y = current_y_sequence[:-1].reshape(-1, dy)
        flat_goal = goal_sequence.reshape(-1, dy)
        flat_rate = command_rate.reshape(-1)
        flat_valid = command_valid.reshape(-1)
        flat_switch = switch_sequence.reshape(-1)
        flat_bootstrap = bootstrap_sequence.reshape(
            -1, args.value_heads
        )
        flat_proposal_score = proposal_score_sequence.reshape(-1)

        value_statistics = []
        value_head_counts = torch.zeros(
            args.value_heads, device=device
        )
        for _ in range(args.value_updates):
            index = torch.randint(
                sequence_steps * E,
                (args.minibatch_size,),
                device=device,
            )
            eligible = flat_valid[index].bool()
            membership = (
                flat_bootstrap[index] & eligible.unsqueeze(-1)
            )
            predictions = agent.command_values(
                flat_y[index], flat_goal[index]
            )
            per_head_error = nn.functional.smooth_l1_loss(
                predictions,
                flat_rate[index].unsqueeze(-1).expand_as(predictions),
                reduction="none",
            )
            denominator = membership.sum().clamp_min(1)
            value_loss = (
                per_head_error * membership.to(per_head_error.dtype)
            ).sum() / denominator
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward(inputs=value_params)
            nn.utils.clip_grad_norm_(
                value_params, args.max_grad_norm
            )
            value_optimizer.step()
            value_statistics.append(value_loss.item())
            value_head_counts += membership.sum(dim=0)

        controller_statistics = []
        high_rate_statistics = []
        low_rate_statistics = []
        for _ in range(args.controller_updates):
            start, end = sample_path_endpoints(
                sequence_steps,
                args.minibatch_size,
                device,
            )
            env_index = torch.randint(
                E, (args.minibatch_size,), device=device
            )
            valid_path = (
                episode_id[start, env_index]
                == episode_id[end, env_index]
            )
            segment_rate = (
                reward_prefix[end, env_index]
                - reward_prefix[start, env_index]
            ) / (end - start).to(reward_sequence.dtype)
            sample_weight = path_weights(
                segment_rate,
                valid_path,
                args.path_weight_temperature,
                args.path_weight_low,
                args.path_weight_high,
            )
            with torch.no_grad():
                z = current_z_sequence[start, env_index]
                y = current_y_sequence[start, env_index]
                achieved_goal = current_y_sequence[end, env_index]
            prediction = agent.control(z, y, achieved_goal)
            per_path_error = (
                prediction - action_sequence[start, env_index]
            ).square().mean(dim=-1)
            controller_loss = (
                per_path_error * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1)
            controller_optimizer.zero_grad(set_to_none=True)
            controller_loss.backward(inputs=controller_params)
            nn.utils.clip_grad_norm_(
                controller_params, args.max_grad_norm
            )
            controller_optimizer.step()
            controller_statistics.append(controller_loss.item())

            median_rate = (
                segment_rate[valid_path].median()
                if valid_path.any()
                else torch.zeros((), device=device)
            )
            high = valid_path & (segment_rate >= median_rate)
            low = valid_path & (segment_rate < median_rate)
            high_rate_statistics.append(
                masked_mean(per_path_error, high).item()
            )
            low_rate_statistics.append(
                masked_mean(per_path_error, low).item()
            )

        proposer_statistics = []
        proposer_head_counts = torch.zeros(
            args.value_heads, device=device
        )
        for _ in range(args.proposer_updates):
            time_index = torch.randint(
                sequence_steps,
                (args.minibatch_size,),
                device=device,
            )
            env_index = torch.randint(
                E, (args.minibatch_size,), device=device
            )
            sampled_head = torch.randint(
                args.value_heads,
                (args.minibatch_size,),
                device=device,
            )
            with torch.no_grad():
                z = agent.encode(obs_sequence[time_index, env_index])
                y = agent.goal_encode(z)
            proposal = agent.propose(y, sampled_head)
            # Values are a differentiable, fixed objective with respect to the
            # proposal coordinate. backward(inputs=...) isolates their parameters.
            proposal_values = agent.command_values(
                y, proposal, detach_inputs=False
            )
            proposal_score = gather_value_head(
                proposal_values, sampled_head
            )
            proposer_loss = -proposal_score.mean()
            proposer_optimizer.zero_grad(set_to_none=True)
            proposer_loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(
                proposer_params, args.max_grad_norm
            )
            proposer_optimizer.step()
            proposer_statistics.append(
                (
                    proposal_score.mean().item(),
                    proposal_values.std(
                        dim=-1, unbiased=False
                    ).mean().item(),
                )
            )
            proposer_head_counts += torch.bincount(
                sampled_head, minlength=args.value_heads
            )

        with torch.no_grad():
            factual_mask = flat_valid.bool()
            proposed_mask = factual_mask & flat_switch.bool()
            factual_values = agent.command_values(
                flat_y, flat_goal
            )
            factual_mean = factual_values.mean(dim=-1)
            factual_disagreement = factual_values.std(
                dim=-1, unbiased=False
            )
            factual_calibration = masked_mean(
                (factual_mean - flat_rate).abs(), factual_mask
            )
            proposed_calibration = masked_mean(
                (factual_mean - flat_rate).abs(), proposed_mask
            )
            proposed_disagreement = masked_mean(
                factual_disagreement, proposed_mask
            )
            factual_disagreement_mean = masked_mean(
                factual_disagreement, factual_mask
            )
            proposal_delivery_error = masked_mean(
                (flat_proposal_score - flat_rate).abs(),
                proposed_mask,
            )
            proposal_delivery_correlation = masked_correlation(
                flat_proposal_score,
                flat_rate,
                proposed_mask,
            )
            oob_membership = (
                ~flat_bootstrap
            ) & factual_mask.unsqueeze(-1)
            oob_abs_error = (
                factual_values - flat_rate.unsqueeze(-1)
            ).abs()
            if oob_membership.any():
                switch_error_floor = oob_abs_error[
                    oob_membership
                ].median()

            diagnostic_count = min(
                512, sequence_steps * E
            )
            diagnostic_obs = obs_sequence[:-1].reshape(
                -1, obs_dim
            )[:diagnostic_count]
            diagnostic_z = agent.encode(diagnostic_obs)
            diagnostic_y = agent.goal_encode(diagnostic_z)
            diagnostic_head = torch.arange(
                diagnostic_count, device=device
            ) % args.value_heads
            diagnostic_goal = agent.propose(
                diagnostic_y, diagnostic_head
            )
            swapped_goal = diagnostic_goal.roll(1, dims=0)
            action_original = agent.control(
                diagnostic_z,
                diagnostic_y,
                diagnostic_goal,
            )
            action_swapped = agent.control(
                diagnostic_z,
                diagnostic_y,
                swapped_goal,
            )
            goal_swap_sensitivity = (
                action_original - action_swapped
            ).square().mean()
            latent_rank = participation_rank(diagnostic_z)
            goal_rank = participation_rank(diagnostic_y)

        wm_array = np.asarray(wm_statistics)
        proposal_array = np.asarray(proposer_statistics)
        writer.add_scalar(
            "losses/lejepa_prediction",
            wm_array[:, 0].mean(),
            global_step,
        )
        writer.add_scalar(
            "losses/lejepa_sigreg",
            wm_array[:, 1].mean(),
            global_step,
        )
        writer.add_scalar(
            "losses/goal_geometry",
            np.mean(geometry_statistics),
            global_step,
        )
        writer.add_scalar(
            "losses/command_value",
            np.mean(value_statistics),
            global_step,
        )
        writer.add_scalar(
            "losses/controller",
            np.mean(controller_statistics),
            global_step,
        )
        writer.add_scalar(
            "controller/mse_high_reward_rate",
            np.mean(high_rate_statistics),
            global_step,
        )
        writer.add_scalar(
            "controller/mse_low_reward_rate",
            np.mean(low_rate_statistics),
            global_step,
        )
        writer.add_scalar(
            "controller/goal_swap_action_sensitivity",
            goal_swap_sensitivity,
            global_step,
        )
        writer.add_scalar(
            "command/factual_calibration_abs",
            factual_calibration,
            global_step,
        )
        writer.add_scalar(
            "command/factual_disagreement",
            factual_disagreement_mean,
            global_step,
        )
        writer.add_scalar(
            "command/proposed_calibration_abs",
            proposed_calibration,
            global_step,
        )
        writer.add_scalar(
            "command/proposed_disagreement",
            proposed_disagreement,
            global_step,
        )
        writer.add_scalar(
            "command/proposal_score",
            proposal_array[:, 0].mean(),
            global_step,
        )
        writer.add_scalar(
            "command/current_proposal_disagreement",
            proposal_array[:, 1].mean(),
            global_step,
        )
        writer.add_scalar(
            "command/selection_time_proposal_disagreement",
            proposal_disagreement_sequence.mean(),
            global_step,
        )
        writer.add_scalar(
            "command/oob_switch_error_floor",
            switch_error_floor,
            global_step,
        )
        writer.add_scalar(
            "command/realized_delivery_rate",
            masked_mean(flat_rate, proposed_mask),
            global_step,
        )
        writer.add_scalar(
            "command/proposal_delivery_abs_error",
            proposal_delivery_error,
            global_step,
        )
        writer.add_scalar(
            "command/proposal_delivery_correlation",
            proposal_delivery_correlation,
            global_step,
        )
        writer.add_scalar(
            "command/completed_suffix_steps",
            masked_mean(command_length.reshape(-1), factual_mask),
            global_step,
        )
        writer.add_scalar(
            "command/ghost_rejection_rate",
            rollout_ghosts / T,
            global_step,
        )
        writer.add_scalar(
            "commitment/switch_rate",
            rollout_switches / T,
            global_step,
        )
        writer.add_scalar(
            "commitment/challenger_rate",
            rollout_challengers / T,
            global_step,
        )
        writer.add_scalar(
            "commitment/age_mean",
            buffer_age[fresh_slot].mean(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/frame_alignment_residual",
            frame_residual,
            global_step,
        )
        writer.add_scalar(
            "diagnostics/frame_projector_drift",
            frame_drift,
            global_step,
        )
        writer.add_scalar(
            "diagnostics/latent_rank",
            latent_rank,
            global_step,
        )
        writer.add_scalar(
            "diagnostics/goal_rank",
            goal_rank,
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_saturation",
            rollout_saturation / T,
            global_step,
        )
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        for head in range(args.value_heads):
            writer.add_scalar(
                f"bootstrap/value_head_{head}_fraction",
                value_head_counts[head]
                / value_head_counts.sum().clamp_min(1),
                global_step,
            )
            writer.add_scalar(
                f"bootstrap/proposer_head_{head}_fraction",
                proposer_head_counts[head]
                / proposer_head_counts.sum().clamp_min(1),
                global_step,
            )
            writer.add_scalar(
                f"bootstrap/rollout_head_{head}_fraction",
                rollout_head_usage[head]
                / rollout_head_usage.sum().clamp_min(1),
                global_step,
            )

    envs.close()
    writer.close()
