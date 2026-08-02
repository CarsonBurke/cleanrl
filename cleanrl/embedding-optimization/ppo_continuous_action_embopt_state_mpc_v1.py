# Embedding Optimization: State MPC v1
#
# Hypothesis: a stable online LeJEPA chart can control locomotion without a
# policy by replanning every environment step. Two-iteration CEM shoots actions
# through one-step latent dynamics and ranks predicted state transitions by
# factual transition reward plus exact completed-command rate and duration.
# Bootstrapped state/goal values provide leave-generator-out goal evidence;
# proposer heads remain aspirational and are falsified by completed commands.
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
    reward_updates: int = 16
    proposer_updates: int = 8
    bootstrap_probability: float = 0.5
    sigreg_coef: float = 0.09

    action_center_rho: float = 0.90
    cem_population: int = 128
    cem_iterations: int = 2
    cem_elites: int = 16
    cem_global_fraction: float = 0.25
    cem_initial_std: float = 0.45
    cem_min_std: float = 0.08
    cem_update_rate: float = 0.7
    warmup_steps: int = 10_000
    goal_noise: float = 0.1
    ghost_goal_radius: float = 0.1
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


def factual_transition_observations(next_observations, infos):
    """Replace autoreset observations with factual terminal observations."""

    factual = np.array(next_observations, copy=True)
    if "final_observation" not in infos:
        return factual
    final_mask = infos.get(
        "_final_observation",
        np.asarray(
            [
                observation is not None
                for observation in infos["final_observation"]
            ],
            dtype=bool,
        ),
    )
    for env_index in np.flatnonzero(final_mask):
        factual[env_index] = infos["final_observation"][env_index]
    return factual


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
            [mlp(3 * dy, hidden, 2, out_std=0.01) for _ in range(args.value_heads)]
        )
        self.proposer_heads = nn.ModuleList(
            [mlp(dy, hidden, dy) for _ in range(args.value_heads)]
        )
        self.transition_reward_model = mlp(
            3 * dy, hidden, 1, out_std=0.01
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
        """Return state/goal reward-rate and log-remaining-step ensembles."""

        features = self.command_features(y, goal, detach=detach_inputs)
        outputs = torch.stack(
            [head(features) for head in self.command_value_heads],
            dim=-1,
        )
        return outputs[..., 0, :], outputs[..., 1, :]

    def transition_reward(self, y, y_next, detach_inputs=True):
        """Predict factual one-step reward from states and their difference."""

        if detach_inputs:
            y, y_next = y.detach(), y_next.detach()
        features = torch.cat([y, y_next, y_next - y], dim=-1)
        return self.transition_reward_model(features).squeeze(-1)

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

def gather_head(values, head):
    """Select one bootstrap head per leading batch element."""
    index = head.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(*head.shape, 1, values.shape[-1])
    return values.gather(-2, index).squeeze(-2)


def gather_value_head(values, head):
    return values.gather(-1, head.unsqueeze(-1)).squeeze(-1)


def route_rate(transition_reward, command_rate, log_remaining_steps):
    """Combine one factual transition and a state/goal command suffix.

    The command prediction is a per-step rate, so its contribution is weighted
    by its endogenous predicted duration. There is no discount, bootstrap, or
    fixed semantic horizon.
    """

    # Algebraically identical to weighting by exp(log_remaining_steps), while
    # remaining finite for arbitrary off-support duration predictions.
    suffix_weight = torch.sigmoid(log_remaining_steps)
    return (
        torch.sigmoid(-log_remaining_steps) * transition_reward
        + suffix_weight * command_rate
    )


def sample_cem_candidates(
    center,
    std,
    population,
    global_fraction,
    generator=None,
):
    """Draw bounded local candidates plus unbiased global actions."""

    if center.ndim != 2 or std.shape != center.shape:
        raise ValueError("center and std must have shape [batch, action_dim]")
    if population < 2:
        raise ValueError("population must be at least two")
    if not 0.0 <= global_fraction < 1.0:
        raise ValueError("global_fraction must lie in [0, 1)")
    global_count = min(population - 1, int(round(population * global_fraction)))
    local_count = population - global_count
    local = center.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
        center.shape[0],
        local_count,
        center.shape[1],
        device=center.device,
        dtype=center.dtype,
        generator=generator,
    )
    # The persistent center is always evaluated exactly.
    local[:, 0] = center
    if global_count:
        global_actions = 2.0 * torch.rand(
            center.shape[0],
            global_count,
            center.shape[1],
            device=center.device,
            dtype=center.dtype,
            generator=generator,
        ) - 1.0
        candidates = torch.cat([local, global_actions], dim=1)
    else:
        candidates = local
    return candidates.clamp(-1.0, 1.0)


def cem_elite_update(
    candidates,
    scores,
    elite_count,
    center,
    std,
    update_rate,
    minimum_std,
):
    """Return a soft elite moment update and the best sampled action."""

    if candidates.ndim != 3 or scores.shape != candidates.shape[:2]:
        raise ValueError("expected candidates [B,P,A] and scores [B,P]")
    if not 1 <= elite_count <= candidates.shape[1]:
        raise ValueError("invalid elite_count")
    if not 0.0 < update_rate <= 1.0:
        raise ValueError("update_rate must lie in (0, 1]")
    elite_index = scores.topk(elite_count, dim=1).indices
    elites = candidates.gather(
        1, elite_index.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
    )
    elite_center = elites.mean(dim=1)
    elite_std = elites.std(dim=1, unbiased=False)
    next_center = center.lerp(elite_center, update_rate).clamp(-1.0, 1.0)
    next_std = std.lerp(elite_std, update_rate).clamp_min(minimum_std)
    best_index = scores.argmax(dim=1)
    best = candidates.gather(
        1, best_index[:, None, None].expand(-1, 1, candidates.shape[-1])
    ).squeeze(1)
    return next_center, next_std, best, elites


@torch.no_grad()
def score_action_candidates(agent, z, y, goal, actions, excluded_head):
    """Score actions entirely through predicted state/state-difference facts."""

    batch, population, action_dim = actions.shape
    z_candidates = z.unsqueeze(1).expand(-1, population, -1).reshape(
        batch * population, -1
    )
    y_candidates = y.unsqueeze(1).expand(-1, population, -1).reshape(
        batch * population, -1
    )
    goals = goal.unsqueeze(1).expand(-1, population, -1).reshape(
        batch * population, -1
    )
    flat_actions = actions.reshape(batch * population, action_dim)
    z_next = agent.predict_next(z_candidates, flat_actions)
    y_next = agent.goal_encode(z_next)
    reward = agent.transition_reward(y_candidates, y_next)
    rates, log_steps = agent.command_values(y_next, goals)
    excluded = excluded_head.unsqueeze(1).expand(-1, population).reshape(-1)
    evaluator_rates = leave_one_out(rates, excluded).mean(dim=-1)
    evaluator_log_steps = leave_one_out(log_steps, excluded).mean(dim=-1)
    scores = route_rate(reward, evaluator_rates, evaluator_log_steps)
    return (
        scores.reshape(batch, population),
        reward.reshape(batch, population),
    )


@torch.no_grad()
def plan_one_step(agent, z, y, goal, excluded_head, persistent_center, args):
    """Two-iteration CEM shooting; scores are never differentiated."""

    center = persistent_center
    std = torch.full_like(center, args.cem_initial_std)
    first_random_mean = None
    final_scores = final_elites = final_actions = final_reward = None
    for _ in range(args.cem_iterations):
        actions = sample_cem_candidates(
            center,
            std,
            args.cem_population,
            args.cem_global_fraction,
        )
        scores, predicted_reward = score_action_candidates(
            agent, z, y, goal, actions, excluded_head
        )
        if first_random_mean is None:
            global_count = min(
                args.cem_population - 1,
                int(round(args.cem_population * args.cem_global_fraction)),
            )
            first_random_mean = (
                scores[:, -global_count:].mean(dim=1)
                if global_count
                else scores.mean(dim=1)
            )
        center, std, _, elites = cem_elite_update(
            actions,
            scores,
            args.cem_elites,
            center,
            std,
            args.cem_update_rate,
            args.cem_min_std,
        )
        final_actions, final_scores = actions, scores
        final_elites, final_reward = elites, predicted_reward
    best_index = final_scores.argmax(dim=1)
    selected = final_actions.gather(
        1, best_index[:, None, None].expand(-1, 1, final_actions.shape[-1])
    ).squeeze(1)
    selected_reward = final_reward.gather(1, best_index[:, None]).squeeze(1)
    diagnostics = {
        "score_spread": final_scores.std(dim=1, unbiased=False),
        "elite_std": final_elites.std(dim=1, unbiased=False).mean(dim=-1),
        "route_advantage": final_scores.max(dim=1).values - first_random_mean,
        "selected_score": final_scores.max(dim=1).values,
        "predicted_reward": selected_reward,
    }
    return selected, center, diagnostics


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
    assert args.value_heads >= 3, "state MPC needs leave-generator-out uncertainty"
    assert args.cem_iterations == 2, "state MPC v1 is defined by two CEM iterations"
    assert 1 <= args.cem_elites < args.cem_population
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
    reward_params = list(agent.transition_reward_model.parameters())
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
    reward_optimizer = optim.Adam(
        reward_params, lr=args.learning_rate, eps=1e-5
    )
    optimizers = (
        (wm_optimizer, args.wm_lr),
        (geometry_optimizer, args.learning_rate),
        (value_optimizer, args.learning_rate),
        (proposer_optimizer, args.learning_rate),
        (reward_optimizer, args.learning_rate),
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
            "[embopt_state_mpc_v1] "
            f"torch.compile(mode={args.compile_mode!r}, dynamic=False)"
        )

    def mark_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

    buffer_obs = torch.zeros((R, T + 1, E, obs_dim), device=device)
    # Vector environments return reset observations on terminal steps. Retain
    # the factual post-action observation separately for transition replay.
    buffer_transition_next_obs = torch.zeros(
        (R, T, E, obs_dim), device=device
    )
    buffer_action = torch.zeros((R, T, E, action_dim), device=device)
    buffer_reward = torch.zeros((R, T, E), device=device)
    buffer_done = torch.zeros((R, T + 1, E), device=device)
    buffer_goal = torch.zeros((R, T, E, dy), device=device)
    buffer_switch = torch.zeros((R, T, E), device=device)
    buffer_bootstrap = torch.zeros(
        (R, T, E, args.value_heads), dtype=torch.bool, device=device
    )
    buffer_proposal_score = torch.zeros((R, T, E), device=device)
    buffer_proposal_log_steps = torch.zeros((R, T, E), device=device)
    buffer_proposal_disagreement = torch.zeros((R, T, E), device=device)
    buffer_age = torch.zeros((R, T, E), device=device)
    buffer_predicted_reward = torch.zeros((R, T, E), device=device)
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
    action_center = torch.zeros(E, action_dim, device=device)
    command_age = torch.zeros(E, device=device)
    # Before the first completed causal commands exist, only invalid/reset goals
    # may initialize. Thereafter this becomes the current out-of-bootstrap value
    # error, learned afresh from replay rather than annealed or EMA-smoothed.
    switch_error_floor = torch.full((), float("inf"), device=device)
    has_command_evidence = False
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
        rollout_score_spread = torch.zeros((), device=device)
        rollout_elite_std = torch.zeros((), device=device)
        rollout_route_advantage = torch.zeros((), device=device)
        rollout_center_drift = torch.zeros((), device=device)
        rollout_goal_sensitivity = torch.zeros((), device=device)
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
                current_rates, current_log_steps = agent.command_values(
                    y, active_goal
                )
                candidate_rates, candidate_log_steps = agent.command_values(
                    y, candidate_goal
                )
                evaluator_current = leave_one_out(
                    current_rates, active_head
                )
                evaluator_candidate = leave_one_out(
                    candidate_rates, proposal_head
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
                current_log_duration = leave_one_out(
                    current_log_steps, active_head
                ).mean(dim=-1)
                candidate_log_duration = leave_one_out(
                    candidate_log_steps, proposal_head
                ).mean(dim=-1)
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
                chosen_log_duration = torch.where(
                    switched, candidate_log_duration, current_log_duration
                )
                action_center = torch.where(
                    next_done.bool().unsqueeze(-1),
                    torch.zeros_like(action_center),
                    action_center,
                )
                old_center = action_center
                if (
                    global_step < args.warmup_steps
                    or not has_command_evidence
                ):
                    # Initial causal coverage must not be selected by entirely
                    # untrained dynamics and reward models. This is data
                    # collection, not a fixed command horizon: the same
                    # commands, completion rules, and labels remain active.
                    action = torch.empty_like(action_center).uniform_(-1.0, 1.0)
                    warmup_scores, warmup_rewards = score_action_candidates(
                        agent,
                        z,
                        y,
                        active_goal,
                        action.unsqueeze(1),
                        active_head,
                    )
                    zero = torch.zeros(E, device=device)
                    plan_diagnostics = {
                        "score_spread": zero,
                        "elite_std": zero,
                        "route_advantage": zero,
                        "selected_score": warmup_scores[:, 0],
                        "predicted_reward": warmup_rewards[:, 0],
                    }
                else:
                    action, planned_center, plan_diagnostics = plan_one_step(
                        agent,
                        z,
                        y,
                        active_goal,
                        active_head,
                        action_center,
                        args,
                    )
                    action_center = old_center.lerp(
                        planned_center, 1.0 - args.action_center_rho
                    )
                swapped_goal = active_goal.roll(1, dims=0)
                swapped_scores, _ = score_action_candidates(
                    agent,
                    z,
                    y,
                    swapped_goal,
                    action.unsqueeze(1),
                    active_head.roll(1),
                )
                goal_sensitivity = (
                    plan_diagnostics["selected_score"] - swapped_scores[:, 0]
                ).abs()

                goal_valid.fill_(True)
                command_age = torch.where(
                    switched, torch.zeros_like(command_age), command_age + 1
                )
                buffer_goal[slot, step] = active_goal
                buffer_switch[slot, step] = switched.float()
                # Bootstrap membership belongs to the command, not the timestep.
                # All suffix labels from one causal experiment retain the same
                # membership so heads do not converge by eventually seeing every
                # fragment of every command.
                buffer_bootstrap[slot, step] = active_bootstrap
                buffer_proposal_score[slot, step] = chosen_score
                buffer_proposal_log_steps[slot, step] = chosen_log_duration
                buffer_proposal_disagreement[slot, step] = (
                    candidate_rates.std(dim=-1, unbiased=False)
                )
                buffer_age[slot, step] = command_age
                buffer_predicted_reward[slot, step] = plan_diagnostics[
                    "predicted_reward"
                ]
                rollout_switches += switched.float().mean()
                rollout_challengers += challenger.float().mean()
                rollout_ghosts += ghost.float().mean()
                rollout_saturation += (
                    (action.abs() > 0.95).float().mean()
                )
                rollout_score_spread += plan_diagnostics[
                    "score_spread"
                ].mean()
                rollout_elite_std += plan_diagnostics["elite_std"].mean()
                rollout_route_advantage += plan_diagnostics[
                    "route_advantage"
                ].mean()
                rollout_center_drift += (
                    action_center - old_center
                ).square().mean().sqrt()
                rollout_goal_sensitivity += goal_sensitivity.mean()
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
            transition_next_obs_numpy = factual_transition_observations(
                next_obs_numpy, infos
            )
            buffer_transition_next_obs[slot, step] = torch.as_tensor(
                transition_next_obs_numpy,
                dtype=torch.float32,
                device=device,
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
        proposal_log_steps_sequence = buffer_proposal_log_steps[
            recent_slots
        ].reshape(sequence_steps, E)
        proposal_disagreement_sequence = buffer_proposal_disagreement[
            recent_slots
        ].reshape(sequence_steps, E)
        predicted_reward_sequence = buffer_predicted_reward[
            recent_slots
        ].reshape(
            sequence_steps, E
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
            replay_next_obs = buffer_transition_next_obs[
                replay_slot, replay_step, env_index
            ]
            replay_action = buffer_action[
                replay_slot, replay_step, env_index
            ]
            replay_valid = torch.ones_like(replay_slot, dtype=torch.float32)
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
            transition_next_z_sequence = agent.encode(
                buffer_transition_next_obs[recent_slots].reshape(-1, obs_dim)
            ).reshape(sequence_steps, E, dz)
            transition_next_y_sequence = agent.goal_encode(
                transition_next_z_sequence.reshape(-1, dz)
            ).reshape(sequence_steps, E, dy)

        # Command labels span rollout boundaries. Only the final incomplete
        # command is censored; every completed failure remains eligible.
        command_rate, command_length, command_valid = (
            completed_command_suffix_targets(
                reward_sequence,
                done_sequence,
                switch_sequence,
            )
        )
        has_command_evidence = has_command_evidence or bool(
            command_valid.any().item()
        )
        flat_y = current_y_sequence[:-1].reshape(-1, dy)
        flat_y_next = transition_next_y_sequence.reshape(-1, dy)
        flat_goal = goal_sequence.reshape(-1, dy)
        flat_rate = command_rate.reshape(-1)
        flat_log_length = command_length.clamp_min(1).log().reshape(-1)
        flat_valid = command_valid.reshape(-1)
        flat_transition_valid = torch.ones_like(flat_valid)
        flat_reward = reward_sequence.reshape(-1)
        flat_switch = switch_sequence.reshape(-1)
        flat_bootstrap = bootstrap_sequence.reshape(
            -1, args.value_heads
        )
        flat_proposal_score = proposal_score_sequence.reshape(-1)
        flat_proposal_log_steps = proposal_log_steps_sequence.reshape(-1)

        reward_statistics = []
        for _ in range(args.reward_updates):
            flat_index = torch.randint(
                0, filled * T * E, (args.minibatch_size,), device=device
            )
            replay_slot = flat_index // (T * E)
            replay_step = (flat_index // E) % T
            env_index = flat_index % E
            with torch.no_grad():
                replay_z = agent.encode(
                    buffer_obs[replay_slot, replay_step, env_index]
                )
                replay_next_z = agent.encode(
                    buffer_transition_next_obs[
                        replay_slot, replay_step, env_index
                    ]
                )
                replay_y = agent.goal_encode(replay_z)
                replay_y_next = agent.goal_encode(replay_next_z)
                replay_model_y_next = agent.goal_encode(
                    agent.predict_next(
                        replay_z,
                        buffer_action[
                            replay_slot, replay_step, env_index
                        ],
                    )
                )
                replay_reward = buffer_reward[
                    replay_slot, replay_step, env_index
                ]
            factual_reward_prediction = agent.transition_reward(
                replay_y, replay_y_next
            )
            model_reward_prediction = agent.transition_reward(
                replay_y, replay_model_y_next
            )
            reward_loss = 0.5 * (
                nn.functional.smooth_l1_loss(
                    factual_reward_prediction, replay_reward
                )
                + nn.functional.smooth_l1_loss(
                    model_reward_prediction, replay_reward
                )
            )
            reward_optimizer.zero_grad(set_to_none=True)
            reward_loss.backward(inputs=reward_params)
            nn.utils.clip_grad_norm_(reward_params, args.max_grad_norm)
            reward_optimizer.step()
            reward_statistics.append(reward_loss.item())

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
            predicted_rates, predicted_log_steps = agent.command_values(
                flat_y[index], flat_goal[index]
            )
            rate_error = nn.functional.smooth_l1_loss(
                predicted_rates,
                flat_rate[index].unsqueeze(-1).expand_as(predicted_rates),
                reduction="none",
            )
            step_error = nn.functional.smooth_l1_loss(
                predicted_log_steps,
                flat_log_length[index]
                .unsqueeze(-1)
                .expand_as(predicted_log_steps),
                reduction="none",
            )
            denominator = membership.sum().clamp_min(1)
            value_loss = (
                (rate_error + step_error)
                * membership.to(rate_error.dtype)
            ).sum() / denominator
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward(inputs=value_params)
            nn.utils.clip_grad_norm_(
                value_params, args.max_grad_norm
            )
            value_optimizer.step()
            value_statistics.append(
                (
                    value_loss.item(),
                    masked_mean(rate_error, membership).item(),
                    masked_mean(step_error, membership).item(),
                )
            )
            value_head_counts += membership.sum(dim=0)

        proposer_statistics = []
        proposer_head_counts = torch.zeros(
            args.value_heads, device=device
        )
        proposer_update_count = (
            args.proposer_updates if has_command_evidence else 0
        )
        for _ in range(proposer_update_count):
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
            proposal_rates, _ = agent.command_values(
                y, proposal, detach_inputs=False
            )
            proposal_score = gather_value_head(
                proposal_rates, sampled_head
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
                    proposal_rates.std(
                        dim=-1, unbiased=False
                    ).mean().item(),
                )
            )
            proposer_head_counts += torch.bincount(
                sampled_head, minlength=args.value_heads
            )
        if not proposer_statistics:
            proposer_statistics.append((0.0, 0.0))

        with torch.no_grad():
            factual_mask = flat_valid.bool()
            proposed_mask = factual_mask & flat_switch.bool()
            factual_rates, factual_log_steps = agent.command_values(
                flat_y, flat_goal
            )
            factual_mean = factual_rates.mean(dim=-1)
            factual_mean_log_steps = factual_log_steps.mean(dim=-1)
            factual_disagreement = factual_rates.std(
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
            factual_step_calibration = masked_mean(
                (factual_mean_log_steps - flat_log_length).abs(),
                factual_mask,
            )
            proposal_step_delivery_error = masked_mean(
                (flat_proposal_log_steps - flat_log_length).abs(),
                proposed_mask,
            )
            oob_membership = (
                ~flat_bootstrap
            ) & factual_mask.unsqueeze(-1)
            oob_abs_error = (
                factual_rates - flat_rate.unsqueeze(-1)
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
            current_reward_predictions = agent.transition_reward(
                flat_y, flat_y_next
            )
            reward_calibration = masked_mean(
                (current_reward_predictions - flat_reward).abs(),
                flat_transition_valid,
            )
            selection_reward_calibration = masked_mean(
                (
                    predicted_reward_sequence.reshape(-1)
                    - flat_reward
                ).abs(),
                flat_transition_valid,
            )
            latent_rank = participation_rank(diagnostic_z)
            goal_rank = participation_rank(diagnostic_y)

        wm_array = np.asarray(wm_statistics)
        value_array = np.asarray(value_statistics)
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
            value_array[:, 0].mean(),
            global_step,
        )
        writer.add_scalar(
            "losses/command_rate",
            value_array[:, 1].mean(),
            global_step,
        )
        writer.add_scalar(
            "losses/command_log_steps",
            value_array[:, 2].mean(),
            global_step,
        )
        writer.add_scalar(
            "losses/transition_reward",
            np.mean(reward_statistics),
            global_step,
        )
        writer.add_scalar(
            "reward/selection_time_abs_error",
            selection_reward_calibration,
            global_step,
        )
        writer.add_scalar(
            "reward/current_model_abs_error",
            reward_calibration,
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
            "command/factual_log_steps_abs_error",
            factual_step_calibration,
            global_step,
        )
        writer.add_scalar(
            "command/proposal_log_steps_delivery_abs_error",
            proposal_step_delivery_error,
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
            "planner/candidate_score_spread",
            rollout_score_spread / T,
            global_step,
        )
        writer.add_scalar(
            "planner/elite_action_std",
            rollout_elite_std / T,
            global_step,
        )
        writer.add_scalar(
            "planner/selected_vs_random_route_advantage",
            rollout_route_advantage / T,
            global_step,
        )
        writer.add_scalar(
            "planner/action_center_drift",
            rollout_center_drift / T,
            global_step,
        )
        writer.add_scalar(
            "planner/action_goal_score_sensitivity",
            rollout_goal_sensitivity / T,
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
