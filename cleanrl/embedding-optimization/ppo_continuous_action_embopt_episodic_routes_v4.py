# Embedding Optimization: Episodic Routes v4
#
# Hypothesis: reusable locomotion routes are factual state-transition records,
# not unconstrained latent goals. An online LeJEPA chart indexes achieved replay
# segments. Retrieval uses their actual re-encoded terminal edge—never a
# transported latent displacement. A data-driven reward-stability score chooses
# factual routes across all realized durations. Every environment step retrieves
# anew and executes that route's actual achieved first waypoint through exact
# inverse dynamics; there is no learned endpoint navigator or commitment loop.
# Bootstrapped applicability validates delivery from the current state. There is
# no PPO, Q-value, TD target, EMA, fixed horizon, or learned goal proposer.
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

    wm_lr: float = 1e-4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    wm_updates: int = 16
    geometry_updates: int = 8
    pair_updates: int = 32
    selected_route_updates: int = 8
    bootstrap_probability: float = 0.6
    sigreg_coef: float = 0.09
    max_grad_norm: float = 0.5

    atlas_pool_size: int = 4096
    atlas_candidates: int = 32
    route_search_chunk: int = 256
    route_archive_capacity: int = 1024
    route_stability_coef: float = 1.5
    route_variance_prior: float = 1.0
    applicability_coef: float = 0.35
    uncertainty_coef: float = 1.0
    support_score_coef: float = 0.15
    minimum_support: float = 0.55
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


def chronological_replay_slots(filled, pointer, capacity, device=None):
    """Oldest-to-newest ring slots, including every filled replay segment."""
    if not 0 <= filled <= capacity:
        raise ValueError("filled must lie within replay capacity")
    if filled < capacity:
        return torch.arange(filled, device=device)
    return torch.arange(pointer, pointer + capacity, device=device) % capacity


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
        # Each head predicts current-state first-edge reward and delivery support.
        self.pair_heads = nn.ModuleList(
            [mlp(5 * dy, hidden, 2, out_std=0.01) for _ in range(args.value_heads)]
        )
        self.inverse = mlp(dz + dy, hidden, action_dim, out_std=0.01)
        self.register_buffer("goal_alignment", torch.eye(dy))
        self.goal_dim = dy
        self.value_heads = args.value_heads

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        return z + self.dynamics(torch.cat([z, action], dim=-1))

    def goal_encode(self, z):
        return self.goal_projector(z.detach()) @ self.goal_alignment

    @staticmethod
    def pair_features(y, goal, goal_delta, first_edge, detach=True):
        if detach:
            y = y.detach()
            goal = goal.detach()
            goal_delta = goal_delta.detach()
            first_edge = first_edge.detach()
        return torch.cat([y, goal, goal_delta, first_edge, goal - y], dim=-1)

    def pair_values(self, y, goal, goal_delta, first_edge, detach_inputs=True):
        features = self.pair_features(
            y, goal, goal_delta, first_edge, detach=detach_inputs
        )
        outputs = torch.stack([head(features) for head in self.pair_heads], dim=-1)
        return outputs[..., 0, :], outputs[..., 1, :]

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


def stability_adjusted_optimal_endpoints(
    reward_prefix,
    squared_reward_prefix,
    episode_id,
    start,
    env,
    stability_coef,
    variance_prior,
):
    """Select each factual start's best endpoint by stability-adjusted reward rate.

    This is a robustness heuristic over autocorrelated rewards. A same-episode
    empirical variance prior and standard-error-shaped penalty make one-step
    spikes pay for noisy evidence without excluding a stable one-step optimum.
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
    squared_reward_sum = (
        squared_reward_prefix[endpoint.unsqueeze(0) + 1, env.unsqueeze(1)]
        - squared_reward_prefix[start, env].unsqueeze(1)
    )
    duration_float = duration.clamp_min(1).to(reward_sum.dtype)
    mean = reward_sum / duration_float
    segment_m2 = (
        squared_reward_sum - reward_sum.square() / duration_float
    ).clamp_min(0)

    reward = reward_prefix[1:] - reward_prefix[:-1]
    squared_reward = squared_reward_prefix[1:] - squared_reward_prefix[:-1]
    episode_mask = (
        episode_id[:, env].T
        == episode_id[start, env].unsqueeze(1)
    )
    episode_count = episode_mask.sum(1).clamp_min(1).to(reward.dtype)
    episode_sum = (reward[:, env].T * episode_mask).sum(1)
    episode_square_sum = (squared_reward[:, env].T * episode_mask).sum(1)
    episode_variance = (
        episode_square_sum / episode_count
        - (episode_sum / episode_count).square()
    ).clamp_min(0)
    variance = (
        segment_m2 + variance_prior * episode_variance.unsqueeze(1)
    ) / (duration_float - 1 + variance_prior)
    stability_penalty_scale = (variance / duration_float).sqrt()
    stability_score = mean - stability_coef * stability_penalty_scale
    stability_score = stability_score.masked_fill(~valid, -torch.inf)
    best_end = stability_score.argmax(dim=1)
    best_mean = mean.gather(1, best_end.unsqueeze(1)).squeeze(1)
    best_stability_score = stability_score.gather(
        1, best_end.unsqueeze(1)
    ).squeeze(1)
    best_duration = duration_float.gather(
        1, best_end.unsqueeze(1)
    ).squeeze(1)
    return (
        best_end,
        best_mean,
        best_stability_score,
        best_duration,
        torch.isfinite(best_stability_score),
    )


def deduplicated_elite_indices(records, scores, valid, capacity):
    """Keep the strongest route for each exact raw start before truncation."""
    valid_indices = torch.where(valid.bool())[0]
    if valid_indices.numel() == 0 or capacity <= 0:
        return valid_indices[:0]
    _, inverse = torch.unique(
        records[valid_indices], dim=0, return_inverse=True
    )
    group_count = int(inverse.max().item()) + 1
    group_best_score = torch.full(
        (group_count,), -torch.inf, device=scores.device, dtype=scores.dtype
    )
    group_best_score.scatter_reduce_(
        0,
        inverse,
        scores[valid_indices],
        reduce="amax",
        include_self=True,
    )
    is_best = scores[valid_indices] == group_best_score[inverse]
    sentinel = records.shape[0]
    group_best_index = torch.full(
        (group_count,), sentinel, device=valid_indices.device, dtype=torch.long
    )
    group_best_index.scatter_reduce_(
        0,
        inverse,
        torch.where(
            is_best,
            valid_indices,
            torch.full_like(valid_indices, sentinel),
        ),
        reduce="amin",
        include_self=True,
    )
    keep_count = min(capacity, group_best_index.numel())
    return group_best_index[
        scores[group_best_index].topk(keep_count).indices
    ]


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


def gather_head(values, head):
    """Select one Thompson head per leading batch element."""
    return values.gather(-1, head.unsqueeze(-1)).squeeze(-1)


def leave_one_out(values, excluded_head):
    """Evaluator ensemble excluding the head that selected a route."""
    heads = values.shape[-1]
    if heads < 2:
        raise ValueError("leave-one-out evaluation requires at least two heads")
    keep = torch.arange(heads, device=values.device)
    view_shape = (1,) * excluded_head.ndim + (heads,)
    keep = keep.reshape(view_shape) != excluded_head.unsqueeze(-1)
    return values[keep].reshape(*values.shape[:-1], heads - 1)


def factual_thompson_score(
    source_rate,
    head_rate,
    head_support_logit,
    applicability_coef,
    support_score_coef,
    minimum_support,
):
    """Source facts lead; a sampled head corrects current-state applicability."""
    support_probability = head_support_logit.sigmoid()
    score = (
        (1.0 - applicability_coef) * source_rate
        + applicability_coef * head_rate
        + support_score_coef * nn.functional.logsigmoid(head_support_logit)
    )
    credible = support_probability >= minimum_support
    return score.masked_fill(~credible, -torch.inf), credible, support_probability


def conservative_route_values(source_score, applicability_rates, applicability_coef):
    """Combine a leading factual stability score with delivery evidence."""
    return (
        (1.0 - applicability_coef) * source_score.unsqueeze(-1)
        + applicability_coef * applicability_rates
    )


def pessimistic_pair_score(
    rates,
    support_logits,
    uncertainty_coef,
    support_score_coef,
    minimum_support,
    error_floor=0.0,
):
    """Support-aware pessimistic score for a state/endpoint route."""
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


def retrieve_absolute_routes(
    current_y,
    motif_start_y,
    motif_end_y,
    motif_valid,
    candidate_count,
):
    """Retrieve actual factual endpoints whose route starts are nearest."""
    if motif_start_y.shape != motif_end_y.shape or motif_start_y.ndim != 2:
        raise ValueError("motif endpoints must share [pool, goal_dim] shape")
    candidate_count = min(candidate_count, motif_start_y.shape[0])
    distance = torch.cdist(current_y, motif_start_y)
    distance = distance.masked_fill(~motif_valid.bool().unsqueeze(0), torch.inf)
    nearest = distance.topk(candidate_count, largest=False, dim=-1).indices
    goals = motif_end_y[nearest]
    valid = motif_valid[nearest] & torch.isfinite(
        distance.gather(1, nearest)
    )
    return goals, nearest, valid, distance.gather(1, nearest)


def factual_first_edge(source_start_y, source_next_y):
    """Exact achieved local edge; independent of the current query state."""
    if source_start_y.shape != source_next_y.shape:
        raise ValueError("source start and next state must share shape")
    return source_next_y - source_start_y


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


def waypoint_delivery_quality(desired, actual):
    """Continuous factual success label for an attempted first waypoint edge."""
    cosine = nn.functional.cosine_similarity(desired, actual, dim=-1)
    relative_error = (
        (actual - desired).square().mean(-1)
        / desired.square().mean(-1).clamp_min(1e-8)
    )
    quality = cosine.clamp_min(0) * (-relative_error).exp()
    return quality, relative_error, cosine


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.cuda and torch.cuda.is_available(), "this experiment requires CUDA"
    assert 1 <= args.recent_iters <= args.replay_iters
    assert args.value_heads >= 2
    assert args.route_variance_prior > 0 and args.route_stability_coef >= 0
    assert 1 <= args.atlas_candidates <= args.atlas_pool_size
    assert args.route_search_chunk >= 1 and args.route_archive_capacity >= 1

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
        + list(agent.inverse.parameters())
    )
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
            f"[embopt_episodic_routes_v4] torch.compile("
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
    buffer_goal_rate = torch.zeros((R, T, E), device=device)
    buffer_goal_support = torch.zeros_like(buffer_goal_rate)
    buffer_control_gate = torch.zeros_like(buffer_goal_rate)
    buffer_selection_head = torch.zeros((R, T, E), dtype=torch.long, device=device)
    buffer_source_score = torch.zeros_like(buffer_goal_rate)
    buffer_source_action = torch.zeros_like(buffer_action)
    buffer_source_start_obs = torch.zeros_like(buffer_obs)
    buffer_source_waypoint_obs = torch.zeros_like(buffer_obs)
    buffer_route_terminal_obs = torch.zeros_like(buffer_obs)
    buffer_route_end_obs = torch.zeros_like(buffer_obs)
    buffer_filled = 0
    buffer_pointer = 0
    archive_capacity = args.route_archive_capacity
    archive_start_obs = torch.zeros((archive_capacity, obs_dim), device=device)
    archive_waypoint_obs = torch.zeros_like(archive_start_obs)
    archive_terminal_obs = torch.zeros_like(archive_start_obs)
    archive_end_obs = torch.zeros_like(archive_start_obs)
    archive_source_action = torch.zeros(
        (archive_capacity, action_dim), device=device
    )
    archive_route_mean = torch.zeros(archive_capacity, device=device)
    archive_route_score = torch.zeros(archive_capacity, device=device)
    archive_route_duration = torch.zeros(archive_capacity, device=device)
    archive_filled = 0

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    episode_counter = torch.zeros(E, dtype=torch.long, device=device)
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
        rollout_candidate_score_spread = []
        rollout_route_mean = []
        rollout_route_stability_score = []
        rollout_route_duration = []
        atlas_replay_transitions = 0
        if evidence_ready:
            atlas_slots = chronological_replay_slots(
                buffer_filled, buffer_pointer, R, device
            )
            atlas_length = buffer_filled * T
            atlas_replay_transitions = atlas_length * E
            atlas_obs_sequence = buffer_obs[atlas_slots].reshape(
                atlas_length, E, obs_dim
            )
            atlas_next_sequence = buffer_next_obs[atlas_slots].reshape(
                atlas_length, E, obs_dim
            )
            atlas_action_sequence = buffer_action[atlas_slots].reshape(
                atlas_length, E, action_dim
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
            atlas_squared_reward_prefix = torch.cat(
                [
                    torch.zeros((1, E), device=device),
                    atlas_reward_sequence.square().cumsum(0),
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
            route_chunks = []
            for chunk_start in range(0, args.atlas_pool_size, args.route_search_chunk):
                chunk_end = min(
                    chunk_start + args.route_search_chunk,
                    args.atlas_pool_size,
                )
                route_chunks.append(
                    stability_adjusted_optimal_endpoints(
                        atlas_reward_prefix,
                        atlas_squared_reward_prefix,
                        atlas_episode_sequence,
                        motif_start[chunk_start:chunk_end],
                        motif_env[chunk_start:chunk_end],
                        args.route_stability_coef,
                        args.route_variance_prior,
                    )
                )
            motif_end = torch.cat([chunk[0] for chunk in route_chunks])
            motif_route_mean = torch.cat([chunk[1] for chunk in route_chunks])
            motif_route_stability_score = torch.cat(
                [chunk[2] for chunk in route_chunks]
            )
            motif_route_duration = torch.cat([chunk[3] for chunk in route_chunks])
            motif_valid_pool = torch.cat([chunk[4] for chunk in route_chunks])
            fresh_start_obs = atlas_obs_sequence[motif_start, motif_env]
            fresh_waypoint_obs = atlas_next_sequence[motif_start, motif_env]
            fresh_source_action = atlas_action_sequence[motif_start, motif_env]
            fresh_terminal_obs = atlas_obs_sequence[motif_end, motif_env]
            fresh_end_obs = atlas_next_sequence[motif_end, motif_env]

            if archive_filled:
                motif_start_obs = torch.cat(
                    [fresh_start_obs, archive_start_obs[:archive_filled]]
                )
                motif_waypoint_obs = torch.cat(
                    [fresh_waypoint_obs, archive_waypoint_obs[:archive_filled]]
                )
                motif_source_action = torch.cat(
                    [fresh_source_action, archive_source_action[:archive_filled]]
                )
                motif_terminal_obs = torch.cat(
                    [fresh_terminal_obs, archive_terminal_obs[:archive_filled]]
                )
                motif_end_obs = torch.cat(
                    [fresh_end_obs, archive_end_obs[:archive_filled]]
                )
                motif_route_mean = torch.cat(
                    [motif_route_mean, archive_route_mean[:archive_filled]]
                )
                motif_route_stability_score = torch.cat(
                    [
                        motif_route_stability_score,
                        archive_route_score[:archive_filled],
                    ]
                )
                motif_route_duration = torch.cat(
                    [
                        motif_route_duration,
                        archive_route_duration[:archive_filled],
                    ]
                )
                motif_valid_pool = torch.cat(
                    [
                        motif_valid_pool,
                        torch.ones(
                            archive_filled, dtype=torch.bool, device=device
                        ),
                    ]
                )
            else:
                motif_start_obs = fresh_start_obs
                motif_waypoint_obs = fresh_waypoint_obs
                motif_source_action = fresh_source_action
                motif_terminal_obs = fresh_terminal_obs
                motif_end_obs = fresh_end_obs

            archive_index = deduplicated_elite_indices(
                motif_start_obs,
                motif_route_stability_score,
                motif_valid_pool,
                archive_capacity,
            )
            archive_count = archive_index.numel()
            archive_start_obs[:archive_count] = motif_start_obs[archive_index]
            archive_waypoint_obs[:archive_count] = motif_waypoint_obs[archive_index]
            archive_source_action[:archive_count] = motif_source_action[archive_index]
            archive_terminal_obs[:archive_count] = motif_terminal_obs[archive_index]
            archive_end_obs[:archive_count] = motif_end_obs[archive_index]
            archive_route_mean[:archive_count] = motif_route_mean[archive_index]
            archive_route_score[:archive_count] = motif_route_stability_score[
                archive_index
            ]
            archive_route_duration[:archive_count] = motif_route_duration[
                archive_index
            ]
            archive_filled = archive_count
            with torch.no_grad():
                motif_start_y = agent.goal_encode(agent.encode(motif_start_obs))
                motif_waypoint_y = agent.goal_encode(agent.encode(motif_waypoint_obs))
                motif_terminal_y = agent.goal_encode(
                    agent.encode(motif_terminal_obs)
                )
                motif_end_y = agent.goal_encode(agent.encode(motif_end_obs))

        for step in range(T):
            buffer_obs[slot, step] = next_obs
            buffer_episode[slot, step] = episode_counter
            learned_action = torch.zeros((E, action_dim), device=device)
            route_selected = torch.zeros(E, dtype=torch.bool, device=device)
            chosen_rate = torch.zeros(E, device=device)
            chosen_support = torch.zeros(E, device=device)
            desired = torch.zeros((E, dy), device=device)
            selected_goal = torch.zeros((E, dy), device=device)
            selected_goal_delta = torch.zeros_like(selected_goal)
            proposal_head = torch.zeros(E, dtype=torch.long, device=device)
            selected_source_score = torch.zeros(E, device=device)
            selected_source_action = torch.zeros((E, action_dim), device=device)
            selected_source_start_obs = torch.zeros((E, obs_dim), device=device)
            selected_source_waypoint_obs = torch.zeros_like(
                selected_source_start_obs
            )
            selected_route_terminal_obs = torch.zeros_like(
                selected_source_start_obs
            )
            selected_route_end_obs = torch.zeros_like(selected_source_start_obs)

            with torch.no_grad():
                mark_step()
                z = agent.encode(next_obs)
                y = agent.goal_encode(z)
                if evidence_ready:
                    candidate_goals, nearest, candidate_valid, motif_distance = (
                        retrieve_absolute_routes(
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
                    candidate_first_edges = (
                        motif_waypoint_y - motif_start_y
                    )[nearest]
                    (
                        candidate_rates,
                        candidate_supports,
                    ) = agent.pair_values(
                            flat_y,
                            flat_goal,
                            candidate_goal_deltas.reshape(-1, dy),
                            candidate_first_edges.reshape(-1, dy),
                        )
                    candidate_rates = candidate_rates.reshape(
                        E, args.atlas_candidates, args.value_heads
                    )
                    candidate_supports = candidate_supports.reshape_as(candidate_rates)
                    proposal_head = torch.randint(
                        args.value_heads, (E,), device=device
                    )
                    proposal_head_candidates = proposal_head.unsqueeze(1).expand(
                        E, args.atlas_candidates
                    )
                    thompson_rates = gather_head(
                        candidate_rates, proposal_head_candidates
                    )
                    thompson_supports = gather_head(
                        candidate_supports, proposal_head_candidates
                    )
                    source_rates = motif_route_stability_score[nearest]
                    candidate_scores, candidate_credible, _ = factual_thompson_score(
                        source_rates,
                        thompson_rates,
                        thompson_supports,
                        args.applicability_coef,
                        args.support_score_coef,
                        args.minimum_support,
                    )
                    effective_candidate_rates = conservative_route_values(
                        source_rates,
                        candidate_rates,
                        args.applicability_coef,
                    )
                    evaluator_rates = leave_one_out(
                        effective_candidate_rates, proposal_head_candidates
                    )
                    evaluator_supports = leave_one_out(
                        candidate_supports, proposal_head_candidates
                    )
                    _, evaluator_credible, _ = pessimistic_pair_score(
                        evaluator_rates,
                        evaluator_supports,
                        args.uncertainty_coef,
                        args.support_score_coef,
                        args.minimum_support,
                        rate_error_floor,
                    )
                    candidate_scores = candidate_scores.masked_fill(
                        ~(candidate_valid & evaluator_credible), -torch.inf
                    )
                    any_credible = torch.isfinite(candidate_scores).any(-1)
                    best = candidate_scores.argmax(-1)
                    batch = torch.arange(E, device=device)
                    selected_pool_index = nearest[batch, best]
                    selected_goal = candidate_goals[batch, best]
                    selected_goal_delta = candidate_goal_deltas[batch, best]
                    selected_source_score = source_rates[batch, best]
                    selected_source_action = motif_source_action[selected_pool_index]
                    selected_source_start_obs = motif_start_obs[selected_pool_index]
                    selected_source_waypoint_obs = motif_waypoint_obs[
                        selected_pool_index
                    ]
                    selected_route_terminal_obs = motif_terminal_obs[
                        selected_pool_index
                    ]
                    selected_route_end_obs = motif_end_obs[selected_pool_index]
                    chosen_rate = evaluator_rates[batch, best].mean(-1)
                    chosen_support = evaluator_supports[
                        batch, best
                    ].mean(-1).sigmoid()
                    desired = candidate_first_edges[batch, best]
                    learned_action = agent.act(z, desired)
                    route_selected = any_credible
                    rollout_motif_distance.append(
                        motif_distance[batch, best].mean().item()
                    )
                    rollout_candidate_coverage.append(
                        candidate_credible.float().mean().item()
                    )
                    rollout_candidate_score_spread.append(
                        candidate_scores[
                            torch.isfinite(candidate_scores)
                        ].std(unbiased=False).item()
                        if torch.isfinite(candidate_scores).any()
                        else 0.0
                    )
                    if route_selected.any():
                        installed_pool_index = selected_pool_index[route_selected]
                        rollout_route_mean.append(
                            motif_route_mean[installed_pool_index].mean().item()
                        )
                        rollout_route_stability_score.append(
                            motif_route_stability_score[
                                installed_pool_index
                            ].mean().item()
                        )
                        rollout_route_duration.append(
                            motif_route_duration[installed_pool_index]
                        )

            random_action = torch.empty((E, action_dim), device=device).uniform_(
                -1.0, 1.0
            )
            control_enabled = (
                route_selected
                & evidence_ready
                & (global_step >= args.warmup_steps)
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
            buffer_goal_rate[slot, step] = chosen_rate
            buffer_goal_support[slot, step] = chosen_support
            buffer_control_gate[slot, step] = control_enabled.float()
            buffer_selection_head[slot, step] = proposal_head
            buffer_source_score[slot, step] = selected_source_score
            buffer_source_action[slot, step] = selected_source_action
            buffer_source_start_obs[slot, step] = selected_source_start_obs
            buffer_source_waypoint_obs[slot, step] = selected_source_waypoint_obs
            buffer_route_terminal_obs[slot, step] = selected_route_terminal_obs
            buffer_route_end_obs[slot, step] = selected_route_end_obs

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
            _, _, valid = ordered_pair_targets(
                reward_prefix,
                episode_sequence,
                pair_start,
                pair_end,
                pair_env,
            )
            rate_target = reward_sequence[pair_start, pair_env]
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
            target_displacement = (ynext - y0).detach()
            rates, support_logits = agent.pair_values(
                y0, ygoal, goal_delta, target_displacement
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
            negative_first_edge = target_displacement.roll(1, dims=0)
            _, negative_support = agent.pair_values(
                y0, negative_goal, negative_delta, negative_first_edge
            )
            losses = []
            rate_loss_total = torch.zeros((), device=device)
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
                losses.append(rate_loss + support_loss)
                rate_loss_total += rate_loss.detach()
                support_loss_total += support_loss.detach()
            value_loss = torch.stack(losses).mean()
            predicted_action = agent.act(z0, target_displacement)
            # The first transition and action are factual regardless of whether
            # the independently sampled endpoint crosses an episode boundary.
            inverse_loss = (predicted_action - action_target).square().mean()
            pair_loss = value_loss + inverse_loss
            pair_opt.zero_grad(set_to_none=True)
            pair_loss.backward(inputs=pair_params)
            nn.utils.clip_grad_norm_(pair_params, args.max_grad_norm)
            pair_opt.step()
            head_examples += positive_membership.sum(0)
            completed_pair_updates += 1
            pair_stats.append(
                (
                    (rate_loss_total / args.value_heads).item(),
                    (support_loss_total / args.value_heads).item(),
                    inverse_loss.item(),
                )
            )

        selected_y = agent.goal_encode(z_sequence).detach()
        selected_next_y = agent.goal_encode(next_z_sequence).detach()
        selected_source_start_y = agent.goal_encode(
            agent.encode(
                buffer_source_start_obs[recent_slots].reshape(-1, obs_dim)
            )
        ).reshape(sequence_length, E, dy).detach()
        selected_source_waypoint_y = agent.goal_encode(
            agent.encode(
                buffer_source_waypoint_obs[recent_slots].reshape(-1, obs_dim)
            )
        ).reshape(sequence_length, E, dy).detach()
        selected_route_terminal_y = agent.goal_encode(
            agent.encode(
                buffer_route_terminal_obs[recent_slots].reshape(-1, obs_dim)
            )
        ).reshape(sequence_length, E, dy).detach()
        selected_goal = agent.goal_encode(
            agent.encode(buffer_route_end_obs[recent_slots].reshape(-1, obs_dim))
        ).reshape(sequence_length, E, dy).detach()
        selected_goal_delta = (
            selected_goal - selected_route_terminal_y
        )
        selected_desired = factual_first_edge(
            selected_source_start_y,
            selected_source_waypoint_y,
        )
        selected_valid = buffer_control_gate[recent_slots].reshape(
            sequence_length, E
        ).bool()
        selected_index = torch.where(selected_valid.reshape(-1))[0]
        selected_route_stats = []
        if selected_index.numel():
            for _ in range(args.selected_route_updates):
                sampled = selected_index[
                    torch.randint(
                        selected_index.numel(),
                        (mb,),
                        device=device,
                    )
                ]
                sampled_y = selected_y.reshape(-1, dy)[sampled]
                sampled_next_y = selected_next_y.reshape(-1, dy)[sampled]
                sampled_goal = selected_goal.reshape(-1, dy)[sampled]
                sampled_goal_delta = selected_goal_delta.reshape(-1, dy)[sampled]
                sampled_desired = selected_desired.reshape(-1, dy)[sampled]
                sampled_reward = reward_sequence.reshape(-1)[sampled]
                actual_displacement = (sampled_next_y - sampled_y).detach()
                delivery_quality, relative_error, cosine = waypoint_delivery_quality(
                    sampled_desired, actual_displacement
                )
                rates, supports = agent.pair_values(
                    sampled_y,
                    sampled_goal,
                    sampled_goal_delta,
                    sampled_desired,
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
                    delivery_rate_loss = (
                        nn.functional.smooth_l1_loss(
                            rates[:, head],
                            sampled_reward,
                            reduction="none",
                        )
                        * member
                    ).sum() / denominator
                    delivery_support_loss = (
                        nn.functional.binary_cross_entropy_with_logits(
                            supports[:, head],
                            delivery_quality,
                            reduction="none",
                        )
                        * member
                    ).sum() / denominator
                    head_losses.append(delivery_rate_loss + delivery_support_loss)
                delivery_value_loss = torch.stack(head_losses).mean()
                selected_loss = delivery_value_loss
                pair_opt.zero_grad(set_to_none=True)
                selected_loss.backward(inputs=pair_params)
                nn.utils.clip_grad_norm_(pair_params, args.max_grad_norm)
                pair_opt.step()
                selected_route_stats.append(
                    (
                        delivery_value_loss.item(),
                        relative_error.mean().item(),
                        cosine.mean().item(),
                    )
                )

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
            _, _, valid = ordered_pair_targets(
                reward_prefix,
                episode_sequence,
                pair_start,
                pair_end,
                pair_env,
            )
            rate_target = reward_sequence[pair_start, pair_env]
            y0 = agent.goal_encode(z_sequence[pair_start, pair_env])
            yterminal = agent.goal_encode(z_sequence[pair_end, pair_env])
            ygoal = agent.goal_encode(next_z_sequence[pair_end, pair_env])
            ynext = agent.goal_encode(next_z_sequence[pair_start, pair_env])
            goal_delta = ygoal - yterminal
            factual_displacement = ynext - y0
            rates, supports = agent.pair_values(
                y0, ygoal, goal_delta, factual_displacement
            )
            ensemble_rate = rates.mean(-1)
            calibration_error = (ensemble_rate - rate_target).abs()
            rate_absolute_error = masked_mean(calibration_error, valid)
            # Recompute from the current factual calibration batch. Do not carry
            # a temporally smoothed target or statistic between updates.
            rate_error_floor = masked_median(calibration_error, valid).item()
            support_probability = supports.mean(-1).sigmoid()
            support_positive = masked_mean(support_probability, valid)
            inverse_prediction = agent.act(
                z_sequence[pair_start, pair_env], factual_displacement
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
            fresh_source_start_y = agent.goal_encode(
                agent.encode(
                    buffer_source_start_obs[fresh_slot].reshape(-1, obs_dim)
                )
            ).reshape(T, E, dy)
            fresh_source_waypoint_y = agent.goal_encode(
                agent.encode(
                    buffer_source_waypoint_obs[fresh_slot].reshape(-1, obs_dim)
                )
            ).reshape(T, E, dy)
            delivery_mask = buffer_control_gate[fresh_slot].bool()
            actual_displacement = fresh_next_y - fresh_y
            desired_displacement = factual_first_edge(
                fresh_source_start_y,
                fresh_source_waypoint_y,
            )
            (
                delivery_quality,
                waypoint_relative_error,
                waypoint_cosine_values,
            ) = waypoint_delivery_quality(
                desired_displacement,
                actual_displacement,
            )
            waypoint_mse = masked_mean(
                (actual_displacement - desired_displacement).square().mean(-1),
                delivery_mask,
            )
            waypoint_relative_mse = masked_mean(
                waypoint_relative_error, delivery_mask
            )
            waypoint_cosine = masked_mean(
                waypoint_cosine_values, delivery_mask
            )
            waypoint_quality = masked_mean(delivery_quality, delivery_mask)
            desired_edge_norm = masked_mean(
                desired_displacement.norm(dim=-1), delivery_mask
            )
            achieved_edge_norm = masked_mean(
                actual_displacement.norm(dim=-1), delivery_mask
            )
            achieved_desired_norm_ratio = (
                achieved_edge_norm / desired_edge_norm.clamp_min(1e-8)
            )
            source_delivered_correlation = masked_correlation(
                buffer_source_score[fresh_slot],
                buffer_reward[fresh_slot],
                delivery_mask,
            )
            source_action_mse = masked_mean(
                (
                    buffer_action[fresh_slot]
                    - buffer_source_action[fresh_slot]
                ).square().mean(-1),
                delivery_mask,
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
            controlled_heads = buffer_selection_head[fresh_slot][delivery_mask]
            if controlled_heads.numel():
                selected_head_histogram = torch.bincount(
                    controlled_heads,
                    minlength=args.value_heads,
                ).float()
                selected_head_probability = (
                    selected_head_histogram / selected_head_histogram.sum()
                )
                selected_head_entropy = -(
                    selected_head_probability
                    * selected_head_probability.clamp_min(1e-8).log()
                ).sum() / math.log(args.value_heads)
            else:
                selected_head_entropy = torch.zeros((), device=device)

        wm_mean = np.mean(wm_stats, axis=0)
        pair_mean = np.mean(pair_stats, axis=0)
        selected_route_mean = (
            np.mean(selected_route_stats, axis=0)
            if selected_route_stats
            else np.zeros(3)
        )
        if rollout_route_duration:
            selected_route_duration = torch.cat(rollout_route_duration).long()
            duration_histogram = torch.bincount(selected_route_duration)
            duration_probability = (
                duration_histogram.float() / duration_histogram.sum()
            )
            route_duration_entropy = -(
                duration_probability
                * duration_probability.clamp_min(1e-8).log()
            ).sum()
            route_duration_mean = selected_route_duration.float().mean()
        else:
            route_duration_entropy = torch.zeros((), device=device)
            route_duration_mean = torch.zeros((), device=device)
        writer.add_scalar("charts/learning_rate", pair_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/lejepa_prediction", wm_mean[0], global_step)
        writer.add_scalar("losses/lejepa_sigreg", wm_mean[1], global_step)
        writer.add_scalar("losses/goal_sigreg", np.mean(geometry_stats), global_step)
        writer.add_scalar("losses/first_edge_reward", pair_mean[0], global_step)
        writer.add_scalar("losses/pair_support", pair_mean[1], global_step)
        writer.add_scalar("losses/inverse", pair_mean[2], global_step)
        writer.add_scalar("losses/selected_route_delivery", selected_route_mean[0], global_step)
        writer.add_scalar("value/rate_absolute_error", rate_absolute_error, global_step)
        writer.add_scalar("value/rate_error_floor", rate_error_floor, global_step)
        writer.add_scalar("support/factual_probability", support_positive, global_step)
        writer.add_scalar("inverse/action_mse", inverse_error, global_step)
        writer.add_scalar("inverse/error_vs_action_variance", inverse_error_vs_variance, global_step)
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
            "atlas/candidate_score_spread",
            (
                np.mean(rollout_candidate_score_spread)
                if rollout_candidate_score_spread
                else 0.0
            ),
            global_step,
        )
        writer.add_scalar(
            "routes/selected_source_mean_rate",
            np.mean(rollout_route_mean) if rollout_route_mean else 0.0,
            global_step,
        )
        writer.add_scalar(
            "routes/selected_stability_adjusted_rate",
            (
                np.mean(rollout_route_stability_score)
                if rollout_route_stability_score
                else 0.0
            ),
            global_step,
        )
        writer.add_scalar("routes/selected_duration_mean", route_duration_mean, global_step)
        writer.add_scalar("routes/selected_duration_entropy", route_duration_entropy, global_step)
        writer.add_scalar("routes/archive_size", archive_filled, global_step)
        writer.add_scalar("routes/replay_transition_count", atlas_replay_transitions, global_step)
        writer.add_scalar("delivery/source_score_reward_correlation", source_delivered_correlation, global_step)
        writer.add_scalar("delivery/waypoint_mse", waypoint_mse, global_step)
        writer.add_scalar("delivery/waypoint_relative_mse", waypoint_relative_mse, global_step)
        writer.add_scalar("delivery/waypoint_cosine", waypoint_cosine, global_step)
        writer.add_scalar("delivery/waypoint_quality", waypoint_quality, global_step)
        writer.add_scalar("delivery/desired_edge_norm", desired_edge_norm, global_step)
        writer.add_scalar("delivery/achieved_edge_norm", achieved_edge_norm, global_step)
        writer.add_scalar("delivery/achieved_desired_norm_ratio", achieved_desired_norm_ratio, global_step)
        writer.add_scalar("delivery/source_action_mse", source_action_mse, global_step)
        writer.add_scalar("routes/predicted_first_edge_reward", buffer_goal_rate[fresh_slot].mean(), global_step)
        writer.add_scalar("routes/predicted_support", buffer_goal_support[fresh_slot].mean(), global_step)
        writer.add_scalar("routes/thompson_head_entropy", selected_head_entropy, global_step)
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
