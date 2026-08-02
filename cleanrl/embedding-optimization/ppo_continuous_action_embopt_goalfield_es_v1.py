# Embedding Optimization: Local Goal-Field ES v1
#
# Hypothesis: goal choice can be optimized from completed episodic reward without
# a policy gradient or value function. A 32-parameter field scores factual,
# state-local LeJEPA displacements; their continuous mixture is executed by exact
# inverse dynamics. Eight mirrored, common-random-number episodes estimate the
# field update. Representation, inverse, and edge atlas are frozen within each
# generation. There is no PPO, Q function, learned reward model, EMA, or horizon.
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
    wandb_entity: str | None = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    max_episode_steps: int = 1_000
    replay_capacity: int = 262_144
    minibatch_size: int = 1_024

    latent_dim: int = 8
    hidden_dim: int = 256
    field_features: int = 4
    field_dim: int = 32
    atlas_size: int = 4_096
    local_edges: int = 32
    edge_temperature: float = 0.20

    warmup_generations: int = 4
    model_refresh_generations: int = 8
    representation_updates: int = 64
    inverse_updates: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    sigreg_coef: float = 0.09
    max_grad_norm: float = 0.5

    es_sigma: float = 0.10
    es_learning_rate: float = 0.03
    field_norm: float = 1.0
    max_es_gradient_norm: float = 5.0

    compile: bool = True
    compile_mode: str = "reduce-overhead"


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
    layers.extend(
        [
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=out_std),
        ]
    )
    return nn.Sequential(*layers)


class SIGReg(nn.Module):
    """Characteristic-function regularizer used by the LeJEPA lineage."""

    def __init__(self, projections=256, knots=17, reference_samples=128):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        quadrature = torch.full((knots,), 2 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.projections = projections
        self.reference_samples = reference_samples
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
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.encoder = mlp(
            obs_dim, args.hidden_dim, args.latent_dim, layer_norm=True
        )
        self.dynamics = mlp(
            args.latent_dim + action_dim,
            args.hidden_dim,
            args.latent_dim,
            out_std=0.01,
            layer_norm=True,
        )
        self.inverse = mlp(
            2 * args.latent_dim,
            args.hidden_dim,
            action_dim,
            out_std=0.01,
        )
        self.register_buffer("alignment", torch.eye(args.latent_dim))

    def encode_raw(self, obs):
        return self.encoder(obs)

    def encode(self, obs):
        return self.encoder(obs) @ self.alignment

    def predict_next_raw(self, z, action):
        return z + self.dynamics(torch.cat([z, action], dim=-1))

    def act(self, y, desired_delta):
        return torch.tanh(
            self.inverse(torch.cat([y.detach(), desired_delta.detach()], dim=-1))
        )


class Replay:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.pointer = 0
        self.size = 0

    def add(self, obs, actions, next_obs, mask):
        indices = np.flatnonzero(mask)
        for index in indices:
            self.obs[self.pointer] = obs[index]
            self.actions[self.pointer] = actions[index]
            self.next_obs[self.pointer] = next_obs[index]
            self.pointer = (self.pointer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, count, rng):
        if self.size == 0:
            raise ValueError("cannot sample empty replay")
        return rng.integers(0, self.size, size=count)


def paired_reset_seeds(base_seed, generation, pairs):
    first = base_seed + generation * pairs
    return [first + pair for pair in range(pairs) for _ in range(2)]


def orthogonal_directions(field_dim, pairs, generator, device, tangent=None):
    if pairs > field_dim:
        raise ValueError("number of directions cannot exceed field dimension")
    matrix = torch.randn(field_dim, pairs, generator=generator, device=device)
    if tangent is not None and tangent.norm() > 1e-6:
        unit = tangent / tangent.norm()
        matrix = matrix - unit[:, None] * (unit @ matrix)[None]
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T * math.sqrt(field_dim)


def mirrored_population(theta, directions, sigma, target_norm=None):
    population = torch.empty(
        2 * directions.shape[0], theta.numel(), device=theta.device
    )
    population[0::2] = theta + sigma * directions
    population[1::2] = theta - sigma * directions
    if target_norm is not None:
        population = (
            target_norm
            * population
            / population.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
    return population


def normalized_es_gradient(returns, directions, sigma):
    differences = returns[0::2] - returns[1::2]
    # Across-seed population dispersion is the relevant noise scale. Normalizing
    # by the paired differences themselves would force the same update norm even
    # when the perturbations have essentially no behavioral effect.
    scale = returns.std(unbiased=False).clamp_min(1.0)
    gradient = (differences[:, None] * directions).mean(0) / (2 * sigma * scale)
    return gradient, differences, scale


def procrustes_alignment(raw_new, global_old):
    u, _, vh = torch.linalg.svd(raw_new.T @ global_old, full_matrices=False)
    return u @ vh


def field_features(y, count):
    if count < 1 or count > y.shape[-1] + 1:
        raise ValueError("field_features must include bias and fit latent state")
    features = torch.ones(*y.shape[:-1], count, device=y.device, dtype=y.dtype)
    if count > 1:
        features[..., 1:] = torch.tanh(y[..., : count - 1])
    return features


def continuous_local_goal(
    y,
    theta_population,
    atlas_y,
    atlas_delta,
    field_features_count,
    local_edges,
    temperature,
):
    """Return a convex mixture of factual edges near each current state."""
    distances = torch.cdist(y, atlas_y)
    k = min(local_edges, atlas_y.shape[0])
    neighbor_indices = distances.topk(k, largest=False).indices
    candidates = atlas_delta[neighbor_indices]

    features = field_features(y, field_features_count)
    matrices = theta_population.view(
        theta_population.shape[0], field_features_count, y.shape[-1]
    )
    query = torch.einsum("bf,bfd->bd", features, matrices)
    query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    candidate_unit = candidates / candidates.norm(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    logits = torch.einsum("bkd,bd->bk", candidate_unit, query) / temperature
    weights = logits.softmax(dim=-1)
    desired = torch.einsum("bk,bkd->bd", weights, candidates)
    return desired, weights, neighbor_indices, candidates


def factual_next_observations(next_obs, infos):
    factual = np.array(next_obs, copy=True)
    if "final_observation" not in infos:
        return factual
    mask = infos.get(
        "_final_observation",
        np.asarray([item is not None for item in infos["final_observation"]]),
    )
    for index in np.flatnonzero(mask):
        factual[index] = infos["final_observation"][index]
    return factual


@torch.no_grad()
def rebuild_atlas(agent, replay, atlas_size, rng, device):
    count = min(atlas_size, replay.size)
    indices = rng.choice(replay.size, size=count, replace=False)
    obs = torch.as_tensor(replay.obs[indices], device=device)
    next_obs = torch.as_tensor(replay.next_obs[indices], device=device)
    y = agent.encode(obs)
    next_y = agent.encode(next_obs)
    return y, next_y - y


def refresh_models(
    agent,
    representation_optimizer,
    inverse_optimizer,
    sigreg,
    replay,
    args,
    rng,
    device,
):
    if replay.size < args.minibatch_size:
        return {}

    anchor_count = min(2_048, replay.size)
    anchor_indices = rng.choice(replay.size, size=anchor_count, replace=False)
    anchor_obs = torch.as_tensor(replay.obs[anchor_indices], device=device)
    with torch.no_grad():
        old_global = agent.encode(anchor_obs)

    representation_loss = prediction_loss = regularization_loss = None
    for _ in range(args.representation_updates):
        indices = replay.sample_indices(args.minibatch_size, rng)
        obs = torch.as_tensor(replay.obs[indices], device=device)
        actions = torch.as_tensor(replay.actions[indices], device=device)
        next_obs = torch.as_tensor(replay.next_obs[indices], device=device)

        z = agent.encode_raw(obs)
        next_z = agent.encode_raw(next_obs)
        predicted = agent.predict_next_raw(z, actions)
        prediction_loss = nn.functional.smooth_l1_loss(predicted, next_z.detach())
        regularization_loss = 0.5 * (sigreg(z) + sigreg(next_z))
        representation_loss = prediction_loss + args.sigreg_coef * regularization_loss

        representation_optimizer.zero_grad(set_to_none=True)
        representation_loss.backward()
        nn.utils.clip_grad_norm_(
            list(agent.encoder.parameters()) + list(agent.dynamics.parameters()),
            args.max_grad_norm,
        )
        representation_optimizer.step()

    with torch.no_grad():
        new_raw = agent.encode_raw(anchor_obs)
        new_alignment = procrustes_alignment(new_raw, old_global)
        chart_residual = (
            (new_raw @ new_alignment - old_global).square().mean().sqrt()
        )
        agent.alignment.copy_(new_alignment)

    inverse_loss = action_variance = None
    for _ in range(args.inverse_updates):
        indices = replay.sample_indices(args.minibatch_size, rng)
        obs = torch.as_tensor(replay.obs[indices], device=device)
        actions = torch.as_tensor(replay.actions[indices], device=device)
        next_obs = torch.as_tensor(replay.next_obs[indices], device=device)
        with torch.no_grad():
            y = agent.encode(obs)
            delta = agent.encode(next_obs) - y
        predicted_action = agent.act(y, delta)
        inverse_loss = nn.functional.mse_loss(predicted_action, actions)
        action_variance = predicted_action.var(dim=0, unbiased=False).mean()

        inverse_optimizer.zero_grad(set_to_none=True)
        inverse_loss.backward()
        nn.utils.clip_grad_norm_(agent.inverse.parameters(), args.max_grad_norm)
        inverse_optimizer.step()

    return {
        "representation_loss": representation_loss.item(),
        "prediction_loss": prediction_loss.item(),
        "sigreg_loss": regularization_loss.item(),
        "inverse_loss": inverse_loss.item(),
        "inverse_action_variance": action_variance.item(),
        "chart_residual": chart_residual.item(),
    }


def main():
    args = tyro.cli(Args)
    if args.num_envs != 16:
        raise ValueError("goalfield ES v1 requires 16 envs for eight mirrored pairs")
    if args.field_dim != args.field_features * args.latent_dim:
        raise ValueError("field_dim must equal field_features * latent_dim")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("goalfield ES v1 requires CUDA")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    es_generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, action_dim, args).to(device)
    sigreg = SIGReg().to(device)
    representation_optimizer = optim.AdamW(
        list(agent.encoder.parameters()) + list(agent.dynamics.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    inverse_optimizer = optim.AdamW(
        agent.inverse.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def infer_control(obs, population, atlas_states, atlas_edges):
        y = agent.encode(obs)
        desired, weights, neighbor_indices, candidates = continuous_local_goal(
            y,
            population,
            atlas_states,
            atlas_edges,
            args.field_features,
            args.local_edges,
            args.edge_temperature,
        )
        return agent.act(y, desired), y, desired, weights, neighbor_indices, candidates

    if args.compile:
        infer_control = torch.compile(infer_control, mode=args.compile_mode)
        infer_encode = torch.compile(agent.encode, mode=args.compile_mode)
    else:
        infer_encode = agent.encode

    replay = Replay(args.replay_capacity, obs_dim, action_dim)
    theta = torch.zeros(args.field_dim, device=device)
    atlas_y = atlas_delta = None
    global_step = 0
    generation = 0
    start_time = time.time()
    latest_model_metrics = {}
    best_return = -float("inf")
    previous_gradient = None

    while global_step < args.total_timesteps:
        seeds = paired_reset_seeds(args.seed + 1_000_000, generation, 8)
        observations, _ = envs.reset(seed=seeds)
        pair_mismatch = max(
            float(np.max(np.abs(observations[2 * pair] - observations[2 * pair + 1])))
            for pair in range(8)
        )
        if pair_mismatch > 1e-6:
            raise RuntimeError(
                f"paired resets are not identical (max mismatch {pair_mismatch})"
            )

        directions = orthogonal_directions(
            args.field_dim, 8, es_generator, device, tangent=theta
        )
        population = mirrored_population(
            theta, directions, args.es_sigma, target_norm=args.field_norm
        )
        warmup = generation < args.warmup_generations or atlas_y is None
        active = np.ones(args.num_envs, dtype=bool)
        episode_returns = np.zeros(args.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
        action_saturation_sum = 0.0
        desired_norm_sum = 0.0
        achieved_cosine_sum = 0.0
        achieved_norm_ratio_sum = 0.0
        goal_entropy_sum = 0.0
        paired_action_sensitivity_sum = 0.0
        controlled_steps = 0

        for episode_step in range(args.max_episode_steps):
            obs_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            if warmup:
                pair_actions = rng.uniform(-1, 1, size=(8, action_dim)).astype(
                    np.float32
                )
                actions = np.repeat(pair_actions, 2, axis=0)
                desired = weights = current_y = None
            else:
                with torch.no_grad():
                    (
                        action_tensor,
                        current_y,
                        desired,
                        weights,
                        _,
                        _,
                    ) = infer_control(
                        obs_tensor,
                        population,
                        atlas_y,
                        atlas_delta,
                    )
                    # reduce-overhead CUDA graphs reuse their output storage on
                    # the next compiled call. These two tensors remain live
                    # until delivery is measured after the environment step.
                    current_y = current_y.clone()
                    desired = desired.clone()
                actions = action_tensor.cpu().numpy()
                desired_norm_sum += desired.norm(dim=-1).mean().item()
                paired_action_sensitivity_sum += (
                    action_tensor[0::2] - action_tensor[1::2]
                ).norm(dim=-1).mean().item()
                goal_entropy_sum += (
                    -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean().item()
                )
                controlled_steps += 1

            next_observations, rewards, terminations, truncations, infos = envs.step(
                actions
            )
            factual_next = factual_next_observations(next_observations, infos)
            was_active = active.copy()
            replay.add(observations, actions, factual_next, was_active)
            episode_returns[was_active] += rewards[was_active]
            episode_lengths[was_active] += 1
            action_saturation_sum += float(
                (np.abs(actions[was_active]) > 0.95).mean()
            )

            if not warmup:
                with torch.no_grad():
                    next_y = infer_encode(
                        torch.as_tensor(
                            factual_next, dtype=torch.float32, device=device
                        )
                    )
                    achieved = next_y - current_y
                    cosine = nn.functional.cosine_similarity(
                        desired, achieved, dim=-1
                    )
                    ratio = achieved.norm(dim=-1) / desired.norm(
                        dim=-1
                    ).clamp_min(1e-6)
                mask_tensor = torch.as_tensor(was_active, device=device)
                achieved_cosine_sum += cosine[mask_tensor].mean().item()
                achieved_norm_ratio_sum += ratio[mask_tensor].mean().item()

            done = np.logical_or(terminations, truncations)
            active &= ~done
            global_step += int(was_active.sum())
            observations = next_observations
            if not active.any() or global_step >= args.total_timesteps:
                break

        if active.any() and global_step < args.total_timesteps:
            raise RuntimeError(
                "some environments did not finish within max_episode_steps"
            )

        returns_tensor = torch.as_tensor(
            episode_returns, dtype=torch.float32, device=device
        )
        gradient, differences, return_scale = normalized_es_gradient(
            returns_tensor, directions, args.es_sigma
        )
        pair_difference_rms = differences.square().mean().sqrt()
        consecutive_gradient_cosine = (
            nn.functional.cosine_similarity(
                gradient[None], previous_gradient[None]
            ).item()
            if previous_gradient is not None
            else 0.0
        )
        if not warmup:
            gradient.mul_(
                min(
                    1.0,
                    args.max_es_gradient_norm
                    / gradient.norm().clamp_min(args.max_es_gradient_norm).item(),
                )
            )
            theta.add_(args.es_learning_rate * gradient)
            theta.mul_(
                args.field_norm / theta.norm().clamp_min(1e-6).item()
            )

        generation += 1
        refresh_due = generation >= args.warmup_generations and (
            (generation - args.warmup_generations)
            % args.model_refresh_generations
            == 0
        )
        if refresh_due:
            latest_model_metrics = refresh_models(
                agent,
                representation_optimizer,
                inverse_optimizer,
                sigreg,
                replay,
                args,
                rng,
                device,
            )
            atlas_y, atlas_delta = rebuild_atlas(
                agent, replay, args.atlas_size, rng, device
            )
            previous_gradient = None
        else:
            previous_gradient = gradient.detach().clone()

        mean_return = float(episode_returns.mean())
        best_return = max(best_return, float(episode_returns.max()))
        writer.add_scalar("charts/episodic_return", mean_return, global_step)
        writer.add_scalar(
            "charts/episodic_length", float(episode_lengths.mean()), global_step
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("es/best_population_return", best_return, global_step)
        writer.add_scalar(
            "es/pair_return_difference_rms",
            pair_difference_rms.item(),
            global_step,
        )
        writer.add_scalar("es/return_normalizer", return_scale.item(), global_step)
        writer.add_scalar(
            "es/pair_difference_snr",
            (pair_difference_rms / return_scale).item(),
            global_step,
        )
        writer.add_scalar(
            "es/consecutive_gradient_cosine",
            consecutive_gradient_cosine,
            global_step,
        )
        writer.add_scalar("es/gradient_norm", gradient.norm().item(), global_step)
        writer.add_scalar("es/field_norm", theta.norm().item(), global_step)
        writer.add_scalar("es/sigma", args.es_sigma, global_step)
        writer.add_scalar("diagnostics/paired_initial_mismatch", pair_mismatch, global_step)
        writer.add_scalar(
            "diagnostics/action_saturation",
            action_saturation_sum
            / max(1, int(episode_lengths.max())),
            global_step,
        )
        if controlled_steps:
            writer.add_scalar(
                "goals/desired_norm", desired_norm_sum / controlled_steps, global_step
            )
            writer.add_scalar(
                "goals/edge_weight_entropy",
                goal_entropy_sum / controlled_steps,
                global_step,
            )
            writer.add_scalar(
                "goals/paired_action_sensitivity",
                paired_action_sensitivity_sum / controlled_steps,
                global_step,
            )
            writer.add_scalar(
                "goals/achieved_cosine",
                achieved_cosine_sum / controlled_steps,
                global_step,
            )
            writer.add_scalar(
                "goals/achieved_norm_ratio",
                achieved_norm_ratio_sum / controlled_steps,
                global_step,
            )
        for key, value in latest_model_metrics.items():
            writer.add_scalar(f"model/{key}", value, global_step)

        print(
            f"generation={generation} step={global_step} "
            f"return={mean_return:.1f} best={best_return:.1f} "
            f"pair_rms={pair_difference_rms.item():.2f} "
            f"grad_cos={consecutive_gradient_cosine:.2f} warmup={warmup}"
        )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
