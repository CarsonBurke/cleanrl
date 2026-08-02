# Embedding Optimization: Achieved-Goal Atlas Navigator v2
#
# Hypothesis: reusable locomotion commands are factual state-difference motifs,
# not unconstrained latent goals. An online LeJEPA chart indexes achieved replay
# segments; nearby motifs are transported to the current state and selected by
# pessimistic bootstrapped reward-rate and support evidence. A multimode factual
# navigator and inverse model execute persistent goals. There is no PPO, Q-value,
# TD target, EMA, fixed horizon, or learned goal proposer.
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
    minibatch_size: int = 1024

    latent_dim: int = 64
    goal_dim: int = 16
    hidden_dim: int = 256
    value_heads: int = 5
    navigator_modes: int = 4

    wm_lr: float = 1e-4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    wm_updates: int = 16
    geometry_updates: int = 8
    pair_updates: int = 32
    command_updates: int = 8
    bootstrap_probability: float = 0.6
    sigreg_coef: float = 0.09
    navigator_balance_coef: float = 0.05
    max_grad_norm: float = 0.5

    atlas_pool_size: int = 256
    atlas_candidates: int = 32
    atlas_realized_uncertainty: float = 0.25
    uncertainty_coef: float = 1.0
    support_score_coef: float = 0.15
    minimum_support: float = 0.55
    switch_margin: float = 0.05
    arrival_threshold: float = 1.5
    warmup_steps: int = 10_000
    minimum_evidence_transitions: int = 32_768
    minimum_head_examples: int = 8_192
    minimum_pair_updates: int = 64
    exploration_noise: float = 0.15

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
    """Replace vector autoreset observations with the factual terminal states."""
    factual = np.array(next_observations, copy=True)
    if "final_observation" not in infos:
        return factual
    mask = infos.get(
        "_final_observation",
        np.asarray([item is not None for item in infos["final_observation"]], dtype=bool),
    )
    for index in np.flatnonzero(mask):
        factual[index] = infos["final_observation"][index]
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
        # A linear chart makes SIGReg's principal gauge drift orthogonal, so an
        # explicit Procrustes update can preserve persistent global coordinates.
        self.goal_projector = nn.Linear(dz, dy, bias=False)
        nn.init.orthogonal_(self.goal_projector.weight)
        # Each head predicts factual reward/step, log duration, and support logit.
        self.pair_heads = nn.ModuleList(
            [mlp(4 * dy, hidden, 3, out_std=0.01) for _ in range(args.value_heads)]
        )
        self.navigator = mlp(
            4 * dy, hidden, args.navigator_modes * dy, out_std=0.01
        )
        self.inverse = mlp(dz + dy, hidden, action_dim, out_std=0.01)
        self.register_buffer("goal_alignment", torch.eye(dy))
        self.goal_dim = dy
        self.value_heads = args.value_heads
        self.navigator_modes = args.navigator_modes
        with torch.no_grad():
            for head in self.pair_heads:
                head[-1].bias[1] = math.log(32.0)

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        return z + self.dynamics(torch.cat([z, action], dim=-1))

    def goal_encode(self, z):
        return self.goal_projector(z.detach()) @ self.goal_alignment

    @staticmethod
    def pair_features(y, goal, goal_delta, detach=True):
        if detach:
            y, goal, goal_delta = y.detach(), goal.detach(), goal_delta.detach()
        return torch.cat([y, goal, goal_delta, goal - y], dim=-1)

    def pair_values(self, y, goal, goal_delta, detach_inputs=True):
        features = self.pair_features(
            y, goal, goal_delta, detach=detach_inputs
        )
        outputs = torch.stack([head(features) for head in self.pair_heads], dim=-1)
        return outputs[..., 0, :], outputs[..., 1, :], outputs[..., 2, :]

    def navigate(self, y, goal, goal_delta):
        shape = y.shape[:-1] + (self.navigator_modes, self.goal_dim)
        return self.navigator(
            self.pair_features(y, goal, goal_delta)
        ).reshape(shape)

    def act(self, z, desired_displacement):
        return torch.tanh(
            self.inverse(torch.cat([z.detach(), desired_displacement.detach()], dim=-1))
        )


def ordered_pair_targets(reward_prefix, episode_id, start, end, env):
    """Targets for inclusive, forward, factual transition segments."""
    duration = end - start + 1
    reward_sum = reward_prefix[end + 1, env] - reward_prefix[start, env]
    rate = reward_sum / duration.clamp_min(1).to(reward_sum.dtype)
    valid = (
        (end >= start)
        & (episode_id[start, env] == episode_id[end, env])
    )
    return rate, duration.clamp_min(1).float().log(), valid.float()


def sample_ordered_pairs(length, batch_size, envs, device, generator=None):
    first = torch.randint(length, (batch_size,), device=device, generator=generator)
    second = torch.randint(length, (batch_size,), device=device, generator=generator)
    start = torch.minimum(first, second)
    end = torch.maximum(first, second)
    env = torch.randint(envs, (batch_size,), device=device, generator=generator)
    return start, end, env


def realized_rate_optimal_endpoints(
    reward_prefix,
    episode_id,
    start,
    env,
    uncertainty_floor=0.0,
):
    """Select each factual start's best later endpoint by realized reward/step.

    Every later state in the same episode competes; this is optimal stopping on
    achieved evidence, not a horizon bin or a learned/value-gradient proposal.
    """
    length = episode_id.shape[0]
    endpoint = torch.arange(length, device=start.device)
    duration = endpoint.unsqueeze(0) - start.unsqueeze(1) + 1
    same_episode = (
        episode_id[endpoint.unsqueeze(0), env.unsqueeze(1)]
        == episode_id[start, env].unsqueeze(1)
    )
    valid = (duration > 0) & same_episode
    reward_sum = (
        reward_prefix[endpoint.unsqueeze(0) + 1, env.unsqueeze(1)]
        - reward_prefix[start, env].unsqueeze(1)
    )
    rates = reward_sum / duration.clamp_min(1).to(reward_sum.dtype)
    conservative_rate = rates - uncertainty_floor / duration.clamp_min(1).sqrt()
    conservative_rate = conservative_rate.masked_fill(~valid, -torch.inf)
    best_end = conservative_rate.argmax(dim=1)
    best_rate = rates.gather(1, best_end.unsqueeze(1)).squeeze(1)
    return best_end, best_rate, torch.isfinite(best_rate)


def bootstrap_mask(batch_size, heads, probability, device, generator=None):
    if not 0 < probability <= 1:
        raise ValueError("bootstrap probability must lie in (0, 1]")
    mask = torch.rand(
        batch_size, heads, device=device, generator=generator
    ) < probability
    uncovered = ~mask.any(dim=-1)
    fallback = torch.randint(heads, (batch_size,), device=device, generator=generator)
    mask |= uncovered.unsqueeze(-1) & nn.functional.one_hot(fallback, heads).bool()
    return mask


def pessimistic_pair_score(
    rates,
    support_logits,
    uncertainty_coef,
    support_score_coef,
    minimum_support,
    error_floor=0.0,
):
    """Support-aware lower confidence score for a state/state-difference goal."""
    floor = torch.as_tensor(error_floor, device=rates.device, dtype=rates.dtype)
    rate_mean = rates.mean(dim=-1)
    rate_error = (
        rates.std(dim=-1, unbiased=False).square() + floor.square()
    ).sqrt()
    support_mean = support_logits.mean(dim=-1)
    support_error = support_logits.std(dim=-1, unbiased=False)
    support_lower = support_mean - uncertainty_coef * support_error
    support_probability = support_lower.sigmoid()
    score = (
        rate_mean
        - uncertainty_coef * rate_error
        + support_score_coef * nn.functional.logsigmoid(support_lower)
    )
    credible = support_probability >= minimum_support
    return score.masked_fill(~credible, -torch.inf), credible, support_probability


def endogenous_goal_switch(
    current_rates,
    current_log_duration,
    current_support,
    candidate_rates,
    candidate_support,
    valid,
    uncertainty_coef,
    minimum_support,
    switch_margin,
    arrival_threshold,
    error_floor=0.0,
):
    """Switch on learned arrival or a credibly superior supported reward rate."""
    floor = torch.as_tensor(error_floor, device=current_rates.device, dtype=current_rates.dtype)
    current_error = (
        current_rates.std(-1, unbiased=False).square() + floor.square()
    ).sqrt()
    candidate_error = (
        candidate_rates.std(-1, unbiased=False).square() + floor.square()
    ).sqrt()
    current_upper = current_rates.mean(-1) + uncertainty_coef * current_error
    candidate_lower = candidate_rates.mean(-1) - uncertainty_coef * candidate_error
    candidate_support_lower = (
        candidate_support.mean(-1)
        - uncertainty_coef * candidate_support.std(-1, unbiased=False)
    ).sigmoid()
    current_supported = (
        current_support.mean(-1)
        - uncertainty_coef * current_support.std(-1, unbiased=False)
    ).sigmoid() >= minimum_support
    arrived = (
        current_log_duration.mean(-1) <= math.log(arrival_threshold)
    ) & current_supported & valid.bool()
    challenger = (
        candidate_lower > current_upper + switch_margin
    ) & (candidate_support_lower >= minimum_support) & valid.bool() & ~arrived
    switch = ~valid.bool() | arrived | challenger
    return switch, arrived, challenger


def resolve_goal_install(requested_switch, arrived, candidate_credible, valid):
    """Separate ending an arrived command from installing its replacement."""
    install = requested_switch & candidate_credible
    ended_without_replacement = arrived & ~install
    event = install | ended_without_replacement
    next_valid = (valid.bool() & ~arrived) | install
    return install, event, next_valid


def navigator_loss(candidates, target, valid, balance_coef):
    per_mode = (candidates - target.unsqueeze(-2)).square().mean(-1)
    winner = per_mode.argmin(-1)
    chosen = per_mode.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
    valid = valid.to(chosen.dtype)
    denom = valid.sum().clamp_min(1)
    fit = (chosen * valid).sum() / denom
    assignment = torch.softmax(-per_mode, dim=-1)
    usage = (assignment * valid.unsqueeze(-1)).sum(0) / denom
    balance = (usage * usage.clamp_min(1e-8).log()).sum()
    return fit + balance_coef * balance, fit.detach(), winner.detach(), usage.detach()


def procrustes_alignment(raw_new, global_old):
    if raw_new.ndim != 2 or raw_new.shape != global_old.shape:
        raise ValueError("anchor matrices must have identical [samples, goal_dim] shapes")
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def complete_evidence_gate(
    transitions,
    head_examples,
    pair_updates,
    minimum_transitions,
    minimum_head_examples,
    minimum_pair_updates,
):
    """Learned control starts only after every component has factual evidence."""
    return bool(
        transitions >= minimum_transitions
        and pair_updates >= minimum_pair_updates
        and torch.as_tensor(head_examples).min().item() >= minimum_head_examples
    )


def select_nearby_motifs(
    current_y,
    motif_start_y,
    motif_end_y,
    motif_valid,
    candidate_count,
):
    """Transport nearest factual replay displacements to each current state."""
    if motif_start_y.shape != motif_end_y.shape or motif_start_y.ndim != 2:
        raise ValueError("motif endpoints must share [pool, goal_dim] shape")
    candidate_count = min(candidate_count, motif_start_y.shape[0])
    distance = torch.cdist(current_y, motif_start_y)
    distance = distance.masked_fill(~motif_valid.bool().unsqueeze(0), torch.inf)
    nearest = distance.topk(candidate_count, largest=False, dim=-1).indices
    starts = motif_start_y[nearest]
    displacements = motif_end_y[nearest] - starts
    goals = current_y.unsqueeze(1) + displacements
    valid = motif_valid[nearest] & torch.isfinite(
        distance.gather(1, nearest)
    )
    return goals, nearest, valid, distance.gather(1, nearest)


def participation_rank(x):
    centered = x - x.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def masked_median(values, mask):
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return selected.median()


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


def completed_goal_suffix_rates(reward, done, switched):
    """Factual remaining reward/step until a later switch or episode boundary."""
    steps, envs = reward.shape
    if done.shape != reward.shape or switched.shape != reward.shape:
        raise ValueError("reward, done, and switched must share [steps, envs]")
    rate = torch.zeros_like(reward)
    valid = torch.zeros_like(reward)
    running_reward = torch.zeros(envs, device=reward.device, dtype=reward.dtype)
    running_steps = torch.zeros_like(running_reward)
    has_completion = torch.zeros(envs, dtype=torch.bool, device=reward.device)
    for step in range(steps - 1, -1, -1):
        boundary = done[step].bool()
        if step + 1 < steps:
            boundary |= switched[step + 1].bool()
        running_reward = torch.where(
            boundary, reward[step], reward[step] + running_reward
        )
        running_steps = torch.where(
            boundary, torch.ones_like(running_steps), running_steps + 1
        )
        has_completion |= boundary
        rate[step] = running_reward / running_steps.clamp_min(1)
        valid[step] = has_completion
    return rate, valid


def completed_goal_suffix_targets(reward, done, switched):
    """Causal reward-rate and log-duration labels for completed commands."""
    rate, valid = completed_goal_suffix_rates(reward, done, switched)
    steps, envs = reward.shape
    duration = torch.zeros_like(reward)
    running_steps = torch.zeros(envs, device=reward.device, dtype=reward.dtype)
    for step in range(steps - 1, -1, -1):
        boundary = done[step].bool()
        if step + 1 < steps:
            boundary |= switched[step + 1].bool()
        running_steps = torch.where(
            boundary, torch.ones_like(running_steps), running_steps + 1
        )
        duration[step] = running_steps
    return rate, duration.clamp_min(1).log(), valid


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.cuda and torch.cuda.is_available(), "this experiment requires CUDA"
    assert 1 <= args.recent_iters <= args.replay_iters
    assert args.value_heads >= 2 and args.navigator_modes >= 1
    assert 1 <= args.atlas_candidates <= args.atlas_pool_size

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
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    T, E, R = args.num_steps, args.num_envs, args.replay_iters
    dz, dy, mb = args.latent_dim, args.goal_dim, args.minibatch_size

    agent = Agent(envs, args).to(device)
    z_sigreg = SIGReg().to(device)
    y_sigreg = SIGReg().to(device)
    wm_params = list(agent.encoder.parameters()) + list(agent.dynamics.parameters())
    geometry_params = list(agent.goal_projector.parameters())
    pair_params = (
        list(agent.pair_heads.parameters())
        + list(agent.navigator.parameters())
        + list(agent.inverse.parameters())
    )
    value_params = list(agent.pair_heads.parameters())
    wm_opt = optim.AdamW(
        wm_params, lr=args.wm_lr, weight_decay=args.weight_decay, eps=1e-5
    )
    geometry_opt = optim.Adam(geometry_params, lr=args.learning_rate, eps=1e-5)
    pair_opt = optim.Adam(pair_params, lr=args.learning_rate, eps=1e-5)
    optimizers = (
        (wm_opt, args.wm_lr),
        (geometry_opt, args.learning_rate),
        (pair_opt, args.learning_rate),
    )

    def wm_loss_fn(obs, action, next_obs):
        z = agent.encode(obs)
        target = agent.encode(next_obs)
        prediction = agent.predict_next(z, action)
        prediction_loss = (prediction - target).square().mean()
        sigreg_loss = z_sigreg(torch.cat([z, target], dim=0))
        return (
            prediction_loss + args.sigreg_coef * sigreg_loss,
            prediction_loss.detach(),
            sigreg_loss.detach(),
        )

    def geometry_loss_fn(z):
        y = agent.goal_encode(z)
        return args.sigreg_coef * y_sigreg(y)

    if args.compile:
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        geometry_loss_fn = torch.compile(
            geometry_loss_fn, mode=args.compile_mode, dynamic=False
        )
        print(
            f"[embopt_atlas_navigator_v2] torch.compile("
            f"mode={args.compile_mode!r}, dynamic=False)"
        )

    def mark_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

    buffer_obs = torch.zeros((R, T, E, obs_dim), device=device)
    buffer_next_obs = torch.zeros_like(buffer_obs)
    buffer_action = torch.zeros((R, T, E, action_dim), device=device)
    buffer_reward = torch.zeros((R, T, E), device=device)
    buffer_done = torch.zeros((R, T, E), device=device)
    buffer_episode = torch.zeros((R, T, E), dtype=torch.long, device=device)
    buffer_goal = torch.zeros((R, T, E, dy), device=device)
    buffer_goal_delta = torch.zeros_like(buffer_goal)
    buffer_desired = torch.zeros_like(buffer_goal)
    buffer_switch = torch.zeros((R, T, E), device=device)
    buffer_arrived = torch.zeros_like(buffer_switch)
    buffer_challenger = torch.zeros_like(buffer_switch)
    buffer_mode = torch.zeros((R, T, E), dtype=torch.long, device=device)
    buffer_goal_rate = torch.zeros_like(buffer_switch)
    buffer_goal_support = torch.zeros_like(buffer_switch)
    buffer_goal_log_duration = torch.zeros_like(buffer_switch)
    buffer_control_gate = torch.zeros_like(buffer_switch)
    buffer_filled = 0
    buffer_pointer = 0

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    episode_counter = torch.zeros(E, dtype=torch.long, device=device)
    active_origin_obs = torch.zeros((E, obs_dim), device=device)
    active_motif_start_obs = torch.zeros_like(active_origin_obs)
    active_motif_terminal_obs = torch.zeros_like(active_origin_obs)
    active_motif_end_obs = torch.zeros_like(active_origin_obs)
    goal_valid = torch.zeros(E, dtype=torch.bool, device=device)
    head_examples = torch.zeros(args.value_heads, dtype=torch.long, device=device)
    completed_pair_updates = 0
    evidence_ready = False
    rate_error_floor = 0.0
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            for optimizer, base_lr in optimizers:
                optimizer.param_groups[0]["lr"] = fraction * base_lr

        slot = buffer_pointer
        rollout_motif_distance = []
        rollout_candidate_coverage = []
        rollout_mode_score_spread = []
        rollout_atlas_realized_rate = []
        if evidence_ready:
            recent_count = min(args.recent_iters, buffer_filled)
            newest = (
                torch.arange(
                    buffer_pointer - 1,
                    buffer_pointer - recent_count - 1,
                    -1,
                    device=device,
                )
                % R
            )
            atlas_slots = newest.flip(0)
            atlas_length = recent_count * T
            atlas_obs_sequence = buffer_obs[atlas_slots].reshape(
                atlas_length, E, obs_dim
            )
            atlas_next_sequence = buffer_next_obs[atlas_slots].reshape(
                atlas_length, E, obs_dim
            )
            atlas_reward_sequence = buffer_reward[atlas_slots].reshape(
                atlas_length, E
            )
            atlas_episode_sequence = buffer_episode[atlas_slots].reshape(
                atlas_length, E
            )
            atlas_reward_prefix = torch.cat(
                [
                    torch.zeros((1, E), device=device),
                    atlas_reward_sequence.cumsum(0),
                ],
                dim=0,
            )
            motif_start = torch.randint(
                atlas_length,
                (args.atlas_pool_size,),
                device=device,
            )
            motif_env = torch.randint(
                E, (args.atlas_pool_size,), device=device
            )
            (
                motif_end,
                motif_realized_rate,
                motif_valid_pool,
            ) = realized_rate_optimal_endpoints(
                atlas_reward_prefix,
                atlas_episode_sequence,
                motif_start,
                motif_env,
                args.atlas_realized_uncertainty,
            )
            motif_start_obs = atlas_obs_sequence[motif_start, motif_env]
            motif_terminal_obs = atlas_obs_sequence[motif_end, motif_env]
            motif_end_obs = atlas_next_sequence[motif_end, motif_env]
            with torch.no_grad():
                motif_start_y = agent.goal_encode(agent.encode(motif_start_obs))
                motif_terminal_y = agent.goal_encode(
                    agent.encode(motif_terminal_obs)
                )
                motif_end_y = agent.goal_encode(agent.encode(motif_end_obs))

        for step in range(T):
            buffer_obs[slot, step] = next_obs
            buffer_episode[slot, step] = episode_counter
            learned_action = torch.zeros((E, action_dim), device=device)
            switched = torch.zeros(E, dtype=torch.bool, device=device)
            arrived = torch.zeros_like(switched)
            challenger = torch.zeros_like(switched)
            chosen_rate = torch.zeros(E, device=device)
            chosen_support = torch.zeros(E, device=device)
            chosen_log_duration = torch.zeros(E, device=device)
            desired = torch.zeros((E, dy), device=device)
            mode = torch.zeros(E, dtype=torch.long, device=device)

            with torch.no_grad():
                mark_step()
                z = agent.encode(next_obs)
                y = agent.goal_encode(z)
                if evidence_ready:
                    candidate_goals, nearest, candidate_valid, motif_distance = (
                        select_nearby_motifs(
                            y,
                            motif_start_y,
                            motif_end_y,
                            motif_valid_pool,
                            args.atlas_candidates,
                        )
                    )
                    flat_y = y.unsqueeze(1).expand_as(candidate_goals).reshape(-1, dy)
                    flat_goal = candidate_goals.reshape(-1, dy)
                    candidate_goal_deltas = (
                        motif_end_y - motif_terminal_y
                    )[nearest]
                    (
                        candidate_rates,
                        candidate_logs,
                        candidate_supports,
                    ) = agent.pair_values(
                            flat_y,
                            flat_goal,
                            candidate_goal_deltas.reshape(-1, dy),
                        )
                    candidate_rates = candidate_rates.reshape(
                        E, args.atlas_candidates, args.value_heads
                    )
                    candidate_logs = candidate_logs.reshape_as(candidate_rates)
                    candidate_supports = candidate_supports.reshape_as(candidate_rates)
                    candidate_scores, candidate_credible, _ = (
                        pessimistic_pair_score(
                            candidate_rates,
                            candidate_supports,
                            args.uncertainty_coef,
                            args.support_score_coef,
                            args.minimum_support,
                            rate_error_floor,
                        )
                    )
                    candidate_scores = candidate_scores.masked_fill(
                        ~candidate_valid, -torch.inf
                    )
                    any_credible = torch.isfinite(candidate_scores).any(-1)
                    best = candidate_scores.argmax(-1)
                    batch = torch.arange(E, device=device)
                    candidate_rate_heads = candidate_rates[batch, best]
                    candidate_support_heads = candidate_supports[batch, best]

                    active_origin_y = agent.goal_encode(
                        agent.encode(active_origin_obs)
                    )
                    active_start_y = agent.goal_encode(
                        agent.encode(active_motif_start_obs)
                    )
                    active_terminal_y = agent.goal_encode(
                        agent.encode(active_motif_terminal_obs)
                    )
                    active_end_y = agent.goal_encode(
                        agent.encode(active_motif_end_obs)
                    )
                    active_goal = active_origin_y + active_end_y - active_start_y
                    active_goal_delta = active_end_y - active_terminal_y
                    current_rates, current_logs, current_supports = agent.pair_values(
                        y, active_goal, active_goal_delta
                    )
                    requested_switch, arrived, challenger = endogenous_goal_switch(
                        current_rates,
                        current_logs,
                        current_supports,
                        candidate_rate_heads,
                        candidate_support_heads,
                        goal_valid,
                        args.uncertainty_coef,
                        args.minimum_support,
                        args.switch_margin,
                        args.arrival_threshold,
                        rate_error_floor,
                    )
                    install, switched, goal_valid = resolve_goal_install(
                        requested_switch,
                        arrived,
                        any_credible,
                        goal_valid,
                    )
                    selected_pool_index = nearest[batch, best]
                    selected_start_obs = motif_start_obs[selected_pool_index]
                    selected_terminal_obs = motif_terminal_obs[selected_pool_index]
                    selected_end_obs = motif_end_obs[selected_pool_index]
                    active_origin_obs = torch.where(
                        install.unsqueeze(-1), next_obs, active_origin_obs
                    )
                    active_motif_start_obs = torch.where(
                        install.unsqueeze(-1),
                        selected_start_obs,
                        active_motif_start_obs,
                    )
                    active_motif_terminal_obs = torch.where(
                        install.unsqueeze(-1),
                        selected_terminal_obs,
                        active_motif_terminal_obs,
                    )
                    active_motif_end_obs = torch.where(
                        install.unsqueeze(-1),
                        selected_end_obs,
                        active_motif_end_obs,
                    )

                    active_origin_y = agent.goal_encode(
                        agent.encode(active_origin_obs)
                    )
                    active_start_y = agent.goal_encode(
                        agent.encode(active_motif_start_obs)
                    )
                    active_terminal_y = agent.goal_encode(
                        agent.encode(active_motif_terminal_obs)
                    )
                    active_end_y = agent.goal_encode(
                        agent.encode(active_motif_end_obs)
                    )
                    active_goal = active_origin_y + active_end_y - active_start_y
                    active_goal_delta = active_end_y - active_terminal_y
                    chosen_rates, chosen_logs, chosen_supports = agent.pair_values(
                        y, active_goal, active_goal_delta
                    )
                    chosen_rate = chosen_rates.mean(-1)
                    chosen_log_duration = chosen_logs.mean(-1)
                    chosen_support = chosen_supports.mean(-1).sigmoid()

                    displacements = agent.navigate(y, active_goal, active_goal_delta)
                    next_y = y.unsqueeze(1) + displacements
                    expanded_goal = active_goal.unsqueeze(1).expand_as(next_y)
                    expanded_goal_delta = active_goal_delta.unsqueeze(1).expand_as(
                        next_y
                    )
                    mode_rates, _, mode_supports = agent.pair_values(
                        next_y.reshape(-1, dy),
                        expanded_goal.reshape(-1, dy),
                        expanded_goal_delta.reshape(-1, dy),
                    )
                    mode_scores, _, _ = pessimistic_pair_score(
                        mode_rates,
                        mode_supports,
                        args.uncertainty_coef,
                        args.support_score_coef,
                        args.minimum_support,
                        rate_error_floor,
                    )
                    mode_scores = mode_scores.reshape(E, args.navigator_modes)
                    # If all modes are unsupported, choose the least pessimistic
                    # raw rate instead of manufacturing a supported action.
                    no_supported_mode = ~torch.isfinite(mode_scores).any(-1)
                    raw_mode = mode_rates.mean(-1).reshape(E, args.navigator_modes)
                    mode_scores = torch.where(
                        no_supported_mode.unsqueeze(-1), raw_mode, mode_scores
                    )
                    mode = mode_scores.argmax(-1)
                    desired = displacements[batch, mode]
                    learned_action = agent.act(z, desired)
                    rollout_motif_distance.append(
                        motif_distance[batch, best].mean().item()
                    )
                    rollout_candidate_coverage.append(
                        candidate_credible.float().mean().item()
                    )
                    rollout_mode_score_spread.append(
                        mode_scores.std(-1, unbiased=False).mean().item()
                    )
                    rollout_atlas_realized_rate.append(
                        motif_realized_rate[selected_pool_index].mean().item()
                    )

            random_action = torch.empty((E, action_dim), device=device).uniform_(
                -1.0, 1.0
            )
            control_enabled = (
                goal_valid & evidence_ready & (global_step >= args.warmup_steps)
            )
            noisy_learned_action = (
                learned_action
                + args.exploration_noise * torch.randn_like(learned_action)
            ).clamp(-1.0, 1.0)
            action = torch.where(
                control_enabled.unsqueeze(-1),
                noisy_learned_action,
                random_action,
            )

            buffer_action[slot, step] = action
            buffer_goal[slot, step] = (
                active_goal if evidence_ready else torch.zeros_like(buffer_goal[slot, step])
            )
            buffer_goal_delta[slot, step] = (
                active_goal_delta
                if evidence_ready
                else torch.zeros_like(buffer_goal_delta[slot, step])
            )
            buffer_desired[slot, step] = desired
            buffer_switch[slot, step] = switched.float()
            buffer_arrived[slot, step] = arrived.float()
            buffer_challenger[slot, step] = challenger.float()
            buffer_mode[slot, step] = mode
            buffer_goal_rate[slot, step] = chosen_rate
            buffer_goal_support[slot, step] = chosen_support
            buffer_goal_log_duration[slot, step] = chosen_log_duration
            buffer_control_gate[slot, step] = control_enabled.float()

            global_step += E
            next_obs_np, reward_np, termination, truncation, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(termination, truncation)
            factual_next_np = factual_transition_observations(next_obs_np, infos)
            buffer_next_obs[slot, step] = torch.as_tensor(
                factual_next_np, dtype=torch.float32, device=device
            )
            buffer_reward[slot, step] = torch.as_tensor(
                reward_np, dtype=torch.float32, device=device
            )
            buffer_done[slot, step] = torch.as_tensor(
                done_np, dtype=torch.float32, device=device
            )
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            done = torch.as_tensor(done_np, dtype=torch.bool, device=device)
            episode_counter += done.long()
            goal_valid &= ~done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        buffer_filled = min(buffer_filled + 1, R)
        fresh_slot = buffer_pointer
        buffer_pointer = (buffer_pointer + 1) % R
        transition_count = buffer_filled * T * E

        # Capture the persistent global frame before either online LeJEPA or the
        # projector changes. The later Procrustes solve therefore compensates for
        # both encoder and chart drift seen by every dependent goal model.
        anchor_count = min(1024, transition_count)
        anchor_flat = torch.randint(transition_count, (anchor_count,), device=device)
        anchor_slot = anchor_flat // (T * E)
        anchor_time = (anchor_flat // E) % T
        anchor_env = anchor_flat % E
        anchor_obs = buffer_obs[anchor_slot, anchor_time, anchor_env]
        with torch.no_grad():
            anchor_y_before = agent.goal_encode(agent.encode(anchor_obs))
            old_alignment = agent.goal_alignment.clone()

        wm_stats = []
        for _ in range(args.wm_updates):
            flat = torch.randint(transition_count, (mb,), device=device)
            replay_slot = flat // (T * E)
            replay_time = (flat // E) % T
            replay_env = flat % E
            mark_step()
            wm_loss, prediction_loss, z_reg = wm_loss_fn(
                buffer_obs[replay_slot, replay_time, replay_env],
                buffer_action[replay_slot, replay_time, replay_env],
                buffer_next_obs[replay_slot, replay_time, replay_env],
            )
            wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward(inputs=wm_params)
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((prediction_loss.item(), z_reg.item()))

        geometry_stats = []
        for _ in range(args.geometry_updates):
            flat = torch.randint(transition_count, (mb,), device=device)
            replay_slot = flat // (T * E)
            replay_time = (flat // E) % T
            replay_env = flat % E
            with torch.no_grad():
                z = agent.encode(buffer_obs[replay_slot, replay_time, replay_env])
            mark_step()
            geometry_loss = geometry_loss_fn(z)
            geometry_opt.zero_grad(set_to_none=True)
            geometry_loss.backward(inputs=geometry_params)
            nn.utils.clip_grad_norm_(geometry_params, args.max_grad_norm)
            geometry_opt.step()
            geometry_stats.append(geometry_loss.item())

        with torch.no_grad():
            anchor_z_after = agent.encode(anchor_obs)
            raw_after = agent.goal_projector(anchor_z_after.detach())
            unaligned_after = raw_after @ old_alignment
            unaligned_drift = (unaligned_after - anchor_y_before).square().mean()
            agent.goal_alignment.copy_(
                procrustes_alignment(raw_after, anchor_y_before)
            )
            aligned_after = raw_after @ agent.goal_alignment
            alignment_residual = (aligned_after - anchor_y_before).square().mean()

        recent_count = min(args.recent_iters, buffer_filled)
        newest = (
            torch.arange(
                buffer_pointer - 1,
                buffer_pointer - recent_count - 1,
                -1,
                device=device,
            )
            % R
        )
        recent_slots = newest.flip(0)
        sequence_length = recent_count * T
        obs_sequence = buffer_obs[recent_slots].reshape(sequence_length, E, obs_dim)
        next_sequence = buffer_next_obs[recent_slots].reshape(sequence_length, E, obs_dim)
        action_sequence = buffer_action[recent_slots].reshape(sequence_length, E, action_dim)
        reward_sequence = buffer_reward[recent_slots].reshape(sequence_length, E)
        episode_sequence = buffer_episode[recent_slots].reshape(sequence_length, E)
        reward_prefix = torch.cat(
            [torch.zeros((1, E), device=device), reward_sequence.cumsum(0)], dim=0
        )
        with torch.no_grad():
            z_sequence = agent.encode(obs_sequence.reshape(-1, obs_dim)).reshape(
                sequence_length, E, dz
            )
            next_z_sequence = agent.encode(next_sequence.reshape(-1, obs_dim)).reshape(
                sequence_length, E, dz
            )

        pair_stats = []
        for _ in range(args.pair_updates):
            pair_start, pair_end, pair_env = sample_ordered_pairs(
                sequence_length, mb, E, device
            )
            rate_target, log_duration_target, valid = ordered_pair_targets(
                reward_prefix,
                episode_sequence,
                pair_start,
                pair_end,
                pair_env,
            )
            z0 = z_sequence[pair_start, pair_env]
            zterminal = z_sequence[pair_end, pair_env]
            zgoal = next_z_sequence[pair_end, pair_env]
            znext = next_z_sequence[pair_start, pair_env]
            action_target = action_sequence[pair_start, pair_env]
            y0 = agent.goal_encode(z0).detach()
            yterminal = agent.goal_encode(zterminal).detach()
            ygoal = agent.goal_encode(zgoal).detach()
            ynext = agent.goal_encode(znext).detach()
            goal_delta = (ygoal - yterminal).detach()
            rates, log_durations, support_logits = agent.pair_values(
                y0, ygoal, goal_delta
            )
            membership = bootstrap_mask(
                mb,
                args.value_heads,
                args.bootstrap_probability,
                device,
            )
            positive_membership = membership & valid.bool().unsqueeze(-1)
            negative_goal = ygoal.roll(1, dims=0)
            negative_delta = goal_delta.roll(1, dims=0)
            _, _, negative_support = agent.pair_values(
                y0, negative_goal, negative_delta
            )
            losses = []
            rate_loss_total = torch.zeros((), device=device)
            duration_loss_total = torch.zeros((), device=device)
            support_loss_total = torch.zeros((), device=device)
            for head in range(args.value_heads):
                positive = positive_membership[:, head]
                denominator = positive.sum().clamp_min(1)
                rate_loss = (
                    nn.functional.smooth_l1_loss(
                        rates[:, head], rate_target, reduction="none"
                    )
                    * positive
                ).sum() / denominator
                duration_loss = (
                    nn.functional.smooth_l1_loss(
                        log_durations[:, head],
                        log_duration_target,
                        reduction="none",
                    )
                    * positive
                ).sum() / denominator
                member = membership[:, head]
                support_loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        support_logits[:, head],
                        torch.ones_like(support_logits[:, head]),
                        reduction="none",
                    )
                    * positive
                ).sum() / denominator
                negative_denominator = member.sum().clamp_min(1)
                support_loss = support_loss + (
                    nn.functional.binary_cross_entropy_with_logits(
                        negative_support[:, head],
                        torch.zeros_like(negative_support[:, head]),
                        reduction="none",
                    )
                    * member
                ).sum() / negative_denominator
                losses.append(rate_loss + duration_loss + support_loss)
                rate_loss_total += rate_loss.detach()
                duration_loss_total += duration_loss.detach()
                support_loss_total += support_loss.detach()
            value_loss = torch.stack(losses).mean()
            target_displacement = (ynext - y0).detach()
            candidates = agent.navigate(y0, ygoal, goal_delta)
            nav_loss, nav_fit, _, usage = navigator_loss(
                candidates,
                target_displacement,
                valid,
                args.navigator_balance_coef,
            )
            predicted_action = agent.act(z0, target_displacement)
            inverse_loss = masked_mean(
                (predicted_action - action_target).square().mean(-1), valid
            )
            pair_loss = value_loss + nav_loss + inverse_loss
            pair_opt.zero_grad(set_to_none=True)
            pair_loss.backward(inputs=pair_params)
            nn.utils.clip_grad_norm_(pair_params, args.max_grad_norm)
            pair_opt.step()
            head_examples += positive_membership.sum(0)
            completed_pair_updates += 1
            pair_stats.append(
                (
                    (rate_loss_total / args.value_heads).item(),
                    (duration_loss_total / args.value_heads).item(),
                    (support_loss_total / args.value_heads).item(),
                    nav_fit.item(),
                    inverse_loss.item(),
                    usage.std(unbiased=False).item(),
                )
            )

        # Hindsight achieved pairs teach reachability structure; these causal
        # labels calibrate the exact transported commands the controller actually
        # attempted, including poor and failed commands. Right-edge commands are
        # censored rather than assigned optimistic partial outcomes.
        (
            command_rate_target,
            command_log_duration_target,
            command_valid,
        ) = completed_goal_suffix_targets(
            reward_sequence,
            buffer_done[recent_slots].reshape(sequence_length, E),
            buffer_switch[recent_slots].reshape(sequence_length, E),
        )
        command_valid = command_valid.bool() & buffer_control_gate[
            recent_slots
        ].reshape(sequence_length, E).bool()
        command_y = agent.goal_encode(z_sequence).detach()
        command_goal = buffer_goal[recent_slots].reshape(sequence_length, E, dy)
        command_goal_delta = buffer_goal_delta[recent_slots].reshape(
            sequence_length, E, dy
        )
        eligible_command = torch.where(command_valid.reshape(-1))[0]
        command_stats = []
        if eligible_command.numel():
            for _ in range(args.command_updates):
                sampled = eligible_command[
                    torch.randint(
                        eligible_command.numel(),
                        (mb,),
                        device=device,
                    )
                ]
                sampled_y = command_y.reshape(-1, dy)[sampled]
                sampled_goal = command_goal.reshape(-1, dy)[sampled]
                sampled_delta = command_goal_delta.reshape(-1, dy)[sampled]
                sampled_rate = command_rate_target.reshape(-1)[sampled]
                sampled_log_duration = command_log_duration_target.reshape(-1)[sampled]
                rates, log_durations, _ = agent.pair_values(
                    sampled_y, sampled_goal, sampled_delta
                )
                membership = bootstrap_mask(
                    mb,
                    args.value_heads,
                    args.bootstrap_probability,
                    device,
                )
                head_losses = []
                for head in range(args.value_heads):
                    member = membership[:, head]
                    denominator = member.sum().clamp_min(1)
                    rate_loss = (
                        nn.functional.smooth_l1_loss(
                            rates[:, head],
                            sampled_rate,
                            reduction="none",
                        )
                        * member
                    ).sum() / denominator
                    duration_loss = (
                        nn.functional.smooth_l1_loss(
                            log_durations[:, head],
                            sampled_log_duration,
                            reduction="none",
                        )
                        * member
                    ).sum() / denominator
                    head_losses.append(rate_loss + duration_loss)
                command_loss = torch.stack(head_losses).mean()
                pair_opt.zero_grad(set_to_none=True)
                command_loss.backward(inputs=value_params)
                nn.utils.clip_grad_norm_(value_params, args.max_grad_norm)
                pair_opt.step()
                command_stats.append(command_loss.item())

        evidence_ready = evidence_ready or complete_evidence_gate(
            transition_count,
            head_examples,
            completed_pair_updates,
            args.minimum_evidence_transitions,
            args.minimum_head_examples,
            args.minimum_pair_updates,
        )

        with torch.no_grad():
            pair_start, pair_end, pair_env = sample_ordered_pairs(
                sequence_length, mb, E, device
            )
            rate_target, log_duration_target, valid = ordered_pair_targets(
                reward_prefix,
                episode_sequence,
                pair_start,
                pair_end,
                pair_env,
            )
            y0 = agent.goal_encode(z_sequence[pair_start, pair_env])
            yterminal = agent.goal_encode(z_sequence[pair_end, pair_env])
            ygoal = agent.goal_encode(next_z_sequence[pair_end, pair_env])
            ynext = agent.goal_encode(next_z_sequence[pair_start, pair_env])
            goal_delta = ygoal - yterminal
            rates, log_durations, supports = agent.pair_values(
                y0, ygoal, goal_delta
            )
            ensemble_rate = rates.mean(-1)
            ensemble_log_duration = log_durations.mean(-1)
            calibration_error = (ensemble_rate - rate_target).abs()
            rate_absolute_error = masked_mean(calibration_error, valid)
            duration_log_error = masked_mean(
                (ensemble_log_duration - log_duration_target).abs(), valid
            )
            # Recompute from the current factual calibration batch. Do not carry
            # a temporally smoothed target or statistic between updates.
            rate_error_floor = masked_median(calibration_error, valid).item()
            support_probability = supports.mean(-1).sigmoid()
            support_positive = masked_mean(support_probability, valid)
            nav_candidates = agent.navigate(y0, ygoal, goal_delta)
            nav_target = ynext - y0
            nav_per_mode = (
                nav_candidates - nav_target.unsqueeze(1)
            ).square().mean(-1)
            nav_best = nav_per_mode.min(-1).values
            nav_zero = nav_target.square().mean(-1)
            nav_error_vs_zero = (
                masked_mean(nav_best, valid) / masked_mean(nav_zero, valid).clamp_min(1e-8)
            )
            selected_mode = nav_per_mode.argmin(-1)
            selected_displacement = nav_candidates[
                torch.arange(mb, device=device), selected_mode
            ]
            nav_cosine = masked_mean(
                nn.functional.cosine_similarity(
                    selected_displacement, nav_target, dim=-1
                ),
                valid,
            )
            swapped_goal = ygoal.roll(1, dims=0)
            swapped_delta = goal_delta.roll(1, dims=0)
            swapped_candidates = agent.navigate(y0, swapped_goal, swapped_delta)
            bridge_goal_sensitivity = (
                nav_candidates - swapped_candidates
            ).square().mean().sqrt() / nav_target.square().mean().sqrt().clamp_min(1e-8)
            inverse_prediction = agent.act(
                z_sequence[pair_start, pair_env], nav_target
            )
            inverse_error = masked_mean(
                (
                    inverse_prediction
                    - action_sequence[pair_start, pair_env]
                ).square().mean(-1),
                valid,
            )
            action_variance = action_sequence[pair_start, pair_env].var(
                dim=0, unbiased=False
            ).mean()
            inverse_error_vs_variance = inverse_error / action_variance.clamp_min(1e-8)
            fresh_z = agent.encode(
                buffer_obs[fresh_slot].reshape(-1, obs_dim)
            ).reshape(T, E, dz)
            fresh_next_z = agent.encode(
                buffer_next_obs[fresh_slot].reshape(-1, obs_dim)
            ).reshape(T, E, dz)
            fresh_y = agent.goal_encode(fresh_z)
            fresh_next_y = agent.goal_encode(fresh_next_z)
            delivery_mask = buffer_control_gate[fresh_slot].bool()
            before_distance = (
                buffer_goal[fresh_slot] - fresh_y
            ).square().mean(-1)
            after_distance = (
                buffer_goal[fresh_slot] - fresh_next_y
            ).square().mean(-1)
            goal_progress = masked_mean(
                before_distance - after_distance, delivery_mask
            )
            delivered_rate, delivered_valid = completed_goal_suffix_rates(
                buffer_reward[fresh_slot],
                buffer_done[fresh_slot],
                buffer_switch[fresh_slot],
            )
            selected_delivered_correlation = masked_correlation(
                buffer_goal_rate[fresh_slot],
                delivered_rate,
                delivery_mask & delivered_valid.bool(),
            )
            y_sample = agent.goal_encode(
                z_sequence.reshape(-1, dz)[: min(4096, sequence_length * E)]
            )
            y_rank = participation_rank(y_sample)
            z_rank = participation_rank(
                z_sequence.reshape(-1, dz)[: min(4096, sequence_length * E)]
            )
            action_saturation = (
                (buffer_action[fresh_slot].abs() > 0.95).float().mean()
            )
            control_cost = (
                0.1 * buffer_action[fresh_slot].square().sum(-1)
            ).mean()

        wm_mean = np.mean(wm_stats, axis=0)
        pair_mean = np.mean(pair_stats, axis=0)
        writer.add_scalar("charts/learning_rate", pair_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/lejepa_prediction", wm_mean[0], global_step)
        writer.add_scalar("losses/lejepa_sigreg", wm_mean[1], global_step)
        writer.add_scalar("losses/goal_sigreg", np.mean(geometry_stats), global_step)
        writer.add_scalar("losses/pair_rate", pair_mean[0], global_step)
        writer.add_scalar("losses/pair_log_duration", pair_mean[1], global_step)
        writer.add_scalar("losses/pair_support", pair_mean[2], global_step)
        writer.add_scalar("losses/navigator", pair_mean[3], global_step)
        writer.add_scalar("losses/inverse", pair_mean[4], global_step)
        writer.add_scalar(
            "losses/causal_command_value",
            np.mean(command_stats) if command_stats else 0.0,
            global_step,
        )
        writer.add_scalar("value/rate_absolute_error", rate_absolute_error, global_step)
        writer.add_scalar("value/rate_error_floor", rate_error_floor, global_step)
        writer.add_scalar("value/log_duration_absolute_error", duration_log_error, global_step)
        writer.add_scalar("support/factual_probability", support_positive, global_step)
        writer.add_scalar("navigator/error_vs_zero", nav_error_vs_zero, global_step)
        writer.add_scalar("navigator/cosine", nav_cosine, global_step)
        writer.add_scalar("navigator/inverse_action_mse", inverse_error, global_step)
        writer.add_scalar("navigator/inverse_error_vs_action_variance", inverse_error_vs_variance, global_step)
        writer.add_scalar("navigator/mode_usage_std", pair_mean[5], global_step)
        writer.add_scalar("navigator/swapped_goal_sensitivity", bridge_goal_sensitivity, global_step)
        writer.add_scalar(
            "atlas/nearest_motif_distance",
            np.mean(rollout_motif_distance) if rollout_motif_distance else 0.0,
            global_step,
        )
        writer.add_scalar(
            "atlas/credible_candidate_fraction",
            np.mean(rollout_candidate_coverage) if rollout_candidate_coverage else 0.0,
            global_step,
        )
        writer.add_scalar(
            "atlas/mode_score_spread",
            np.mean(rollout_mode_score_spread) if rollout_mode_score_spread else 0.0,
            global_step,
        )
        writer.add_scalar(
            "atlas/source_realized_rate",
            np.mean(rollout_atlas_realized_rate) if rollout_atlas_realized_rate else 0.0,
            global_step,
        )
        writer.add_scalar("delivery/selected_realized_rate_correlation", selected_delivered_correlation, global_step)
        writer.add_scalar("delivery/goal_progress", goal_progress, global_step)
        writer.add_scalar("commitment/switch_rate", buffer_switch[fresh_slot].mean(), global_step)
        writer.add_scalar("commitment/arrival_rate", buffer_arrived[fresh_slot].mean(), global_step)
        writer.add_scalar("commitment/challenger_rate", buffer_challenger[fresh_slot].mean(), global_step)
        writer.add_scalar("commitment/predicted_rate", buffer_goal_rate[fresh_slot].mean(), global_step)
        writer.add_scalar("commitment/support", buffer_goal_support[fresh_slot].mean(), global_step)
        writer.add_scalar(
            "commitment/predicted_log_duration",
            buffer_goal_log_duration[fresh_slot].mean(),
            global_step,
        )
        writer.add_scalar("evidence/control_enabled_fraction", buffer_control_gate[fresh_slot].mean(), global_step)
        writer.add_scalar("evidence/ready", float(evidence_ready), global_step)
        writer.add_scalar("evidence/minimum_head_examples", head_examples.min(), global_step)
        writer.add_scalar("diagnostics/goal_frame_unaligned_drift", unaligned_drift, global_step)
        writer.add_scalar("diagnostics/goal_frame_alignment_residual", alignment_residual, global_step)
        writer.add_scalar("diagnostics/z_effective_rank", z_rank, global_step)
        writer.add_scalar("diagnostics/y_effective_rank", y_rank, global_step)
        writer.add_scalar("diagnostics/action_saturation", action_saturation, global_step)
        writer.add_scalar("diagnostics/control_cost", control_cost, global_step)
        print(
            f"iter={iteration} SPS={int(global_step / (time.time() - start_time))} "
            f"evidence={evidence_ready}"
        )

    envs.close()
    writer.close()
