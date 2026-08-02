# Factorized geometric doubly-robust LeJEPA actor v5.
# =============================================================================
# A state-only model M(s) predicts normalized long-geometric successor outcomes.
# A separately learned one-step delta/reward model D(s,a) provides action
# dependence by the Bellman factorization
# C(s,a)=(1-gamma)*phi_hat+gamma*M(s+D_delta(s,a)).
# This concentrates scarce counterfactual supervision in short-horizon dynamics
# while M learns long horizons from every state.  Frozen C is used symmetrically
# for behavior and independent auxiliary actions in an exact vector doubly-
# robust policy estimator; the reward covector is applied only at the end.
#
# The reward residual is a coordinate, not a scalar critic.  After each rollout,
# an analytic gauge change transports both state-mean heads to the newly fitted
# reward probe, preserving r = [w, 1] dot [base_phi, residual] exactly.  There is
# no GAE, Q learner, EMA, contrastive loss, PopArt, or PPO clipping. LeJEPA
# targets remain attached; hard targets are copied only after actor consumption.
# Reward contraction makes C action-value-like, but no scalar Q is represented
# or trained and there is no Q-learning objective.
# A held-out vector alpha jointly minimizes the reward-conditioned covariance
# of every outcome coordinate's policy-head score contribution; the estimator
# remains vector-valued until its final task contraction. It is not claimed to
# optimize the shared trunk's full gradient covariance.
# =============================================================================
import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta, Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import ActionEncoder, ARPredictor, MLP, SIGReg, StateEncoder


SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.9970087504549047
    num_minibatches: int = 32
    update_epochs: int = 10
    ent_coef: float = 0.0
    actor_grad_clip: float = 0.5
    target_kl: Optional[float] = 0.02
    alpha_mode: str = "heldout"  # heldout, one, or zero
    alpha_holdout_fraction: float = 0.1
    auxiliary_actions: int = 4

    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    emb_dim: int = 32
    ssl_hidden: int = 256
    pred_depth: int = 2
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = 4
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    ssl_lr: float = 5e-5
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 1024
    ssl_epochs: int = 8
    ssl_grad_clip: float = 1.0

    model_hidden: int = 256
    model_lr: float = 1e-4
    model_weight_decay: float = 1e-4
    model_batch: int = 1024
    model_epochs: int = 8
    model_grad_clip: float = 1.0
    outcome_scale_floor: float = 0.1
    reward_ridge: float = 1e-3
    dynamics_hidden: int = 256
    dynamics_lr: float = 3e-4
    dynamics_weight_decay: float = 1e-4
    dynamics_batch: int = 1024
    dynamics_epochs: int = 8
    dynamics_grad_clip: float = 1.0
    delta_scale_floor: float = 0.05
    dynamics_reward_scale_floor: float = 0.1

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    normalize_reward: bool = False
    clip_reward: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
    def thunk():
        kwargs = {"render_mode": "rgb_array"} if capture_video and idx == 0 else {}
        env = gym.make(env_id, **kwargs)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def _branch_body(width):
    return nn.Sequential(
        layer_init(nn.Linear(width, width)),
        ReLUSquared(),
        layer_init(nn.Linear(width, width)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, width, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, width))
        self.residual_gate = nn.Parameter(torch.full((width,), 4.0))
        self.dense_norm = nn.RMSNorm(width, elementwise_affine=False)
        self.dense = _branch_body(width)
        self.expert_norm = nn.RMSNorm(width, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(width, n_experts))
        self.experts = nn.ModuleList(
            [_branch_body(width) for _ in range(n_experts)]
        )

    def forward(self, concatenated_features, x0):
        x = self.in_proj(concatenated_features)
        gate = torch.sigmoid(self.residual_gate)
        x = gate * x + (1.0 - gate) * x0
        dense = self.dense(self.dense_norm(x))
        expert_input = self.expert_norm(x)
        weights = self.gate(expert_input).softmax(-1)
        experts = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )
        return x + dense + (weights.unsqueeze(-1) * experts).sum(1)


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, width, n_blocks, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, width))
        self.blocks = nn.ModuleList(
            [
                ThinkBlock(width * (index + 1), width, n_experts)
                for index in range(n_blocks)
            ]
        )
        self.output_norm = nn.RMSNorm(
            width * (n_blocks + 1), elementwise_affine=False
        )
        self.output_projection = layer_init(
            nn.Linear(width * (n_blocks + 1), width)
        )

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, -1), x0))
        return self.output_projection(
            self.output_norm(torch.cat(features, -1))
        )


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_dist = args.actor_dist
        self.logvar_min = args.logvar_min
        self.logvar_max = args.logvar_max
        self.trunk = ThinkTrunk(
            obs_dim,
            args.hidden,
            args.k_blocks,
            args.n_experts,
        )
        if args.actor_dist == "beta":
            self.first_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
            self.second_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        elif args.actor_dist == "gaussian":
            self.first_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
            self.second_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        else:
            raise ValueError(f"unknown actor_dist={args.actor_dist!r}")

    def parameters_for_distribution(self, obs):
        feat = self.trunk(obs)
        first_raw = self.first_head(feat)
        second_raw = self.second_head(feat)
        if self.actor_dist == "beta":
            return F.softplus(first_raw) + 1.0, F.softplus(second_raw) + 1.0
        logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(
            second_raw
        )
        return first_raw, logvar

    def distribution(self, obs):
        first, second = self.parameters_for_distribution(obs)
        if self.actor_dist == "beta":
            return Beta(first, second), first, second
        return Normal(first, torch.exp(0.5 * second)), first, second

    def evaluate_latent(self, obs, latent):
        dist, first, second = self.distribution(obs)
        if self.actor_dist == "beta":
            z = latent.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = 2.0 * z - 1.0
            logprob = dist.log_prob(z).sum(-1) - z.shape[-1] * np.log(2.0)
            entropy = dist.entropy().sum(-1) + z.shape[-1] * np.log(2.0)
        else:
            z = latent
            action = torch.tanh(z)
            log_det = 2.0 * (np.log(2.0) - z - F.softplus(-2.0 * z))
            logprob = (dist.log_prob(z) - log_det).sum(-1)
            sample = dist.rsample()
            sample_det = 2.0 * (
                np.log(2.0) - sample - F.softplus(-2.0 * sample)
            )
            entropy = -(dist.log_prob(sample) - sample_det).sum(-1)
        return action, logprob, entropy, first, second

    def get_action(self, obs):
        dist, _, _ = self.distribution(obs)
        latent = dist.sample()
        action, logprob, entropy, first, second = self.evaluate_latent(obs, latent)
        return action, latent, logprob, entropy, first, second


class LeJepaSSL(nn.Module):
    """Attached-target LeJEPA prediction with SIGReg."""

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.encoder = StateEncoder(obs_dim, args.emb_dim, args.ssl_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.predictor = ARPredictor(
            num_frames=args.seq_len,
            depth=args.pred_depth,
            heads=args.pred_heads,
            mlp_dim=args.pred_mlp_dim,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            dim_head=args.pred_dim_head,
            dropout=0.0,
            emb_dropout=0.0,
        )
        self.pred_proj = MLP(args.emb_dim, args.ssl_hidden, args.emb_dim)
        self.sigreg = SIGReg(num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk)

    def forward(self, obs_seq, act_seq, mask_seq, sigreg_weight):
        embedding = self.encoder(obs_seq)
        action_embedding = self.action_encoder(act_seq)
        prediction = self.pred_proj(self.predictor(embedding, action_embedding))
        error = (prediction[:, :-1] - embedding[:, 1:]).square().mean(-1)
        prediction_loss = (error * mask_seq).sum() / mask_seq.sum().clamp_min(1.0)
        sigreg_loss = self.sigreg(embedding.transpose(0, 1))
        return (
            prediction_loss + sigreg_weight * sigreg_loss,
            prediction_loss,
            sigreg_loss,
        )


class StateGeometricOutcomeMean(nn.Module):
    """State-only normalized long-geometric successor-outcome mean M(s)."""

    def __init__(self, obs_dim, outcome_dim, hidden):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.SiLU(),
            nn.RMSNorm(hidden),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        # A zero frozen model is a neutral first-rollout tail and baseline.
        self.mean_head = layer_init(nn.Linear(hidden, outcome_dim), std=0.0)

    def forward(self, obs):
        return self.mean_head(self.body(obs))


class OneStepDynamics(nn.Module):
    """Predict normalized-observation delta and raw reward.

    Both heads are zero initialized, so the cold transition model is the
    identity and its cold reward prediction is neutral.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            nn.RMSNorm(hidden),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.delta_head = layer_init(nn.Linear(hidden, obs_dim), std=0.0)
        self.reward_head = layer_init(nn.Linear(hidden, 1), std=0.0)

    def forward(self, obs, action):
        hidden = self.body(torch.cat([obs, action], -1))
        return self.delta_head(hidden), self.reward_head(hidden).squeeze(-1)


def base_outcome_features(next_embedding, next_obs, action):
    """Use Gym's coordinate-normalized next_s without destroying linear velocity."""
    return torch.cat(
        [
            next_embedding.detach(),
            next_obs,
            action,
            action.square(),
            torch.ones_like(action[..., :1]),
        ],
        -1,
    )


def fit_reward_covector(base_phi, reward, ridge=1e-3, valid=None):
    """Ridge-fit w and define an algebraically exact residual r - w dot phi."""
    x = base_phi.reshape(-1, base_phi.shape[-1])
    y = reward.reshape(-1)
    if valid is not None:
        keep = valid.reshape(-1) > 0
        x, y = x[keep], y[keep]
    x64, y64 = x.double(), y.double()
    gram = x64.T @ x64
    scale = gram.diagonal().mean().clamp_min(1.0)
    eye = torch.eye(x.shape[-1], device=x.device, dtype=torch.float64)
    weight = torch.linalg.solve(gram + ridge * scale * eye, x64.T @ y64)
    weight = weight.to(base_phi.dtype)
    residual = reward - torch.einsum("...d,d->...", base_phi, weight)
    return weight, residual


def augment_reward_residual(base_phi, reward, reward_covector):
    residual = reward - torch.einsum("...d,d->...", base_phi, reward_covector)
    return torch.cat([base_phi, residual.unsqueeze(-1)], -1)


@torch.no_grad()
def factorized_action_control(
    state_model,
    dynamics,
    encoder,
    obs,
    action,
    frame_rotation,
    frame_offset,
    reward_covector,
    gamma,
):
    """Bellman-composed action control variate in normalized outcome units.

    Predicting raw reward makes the residual coordinate gauge exact:
    eps_hat = r_hat - w dot base_phi_hat. Consequently contracting the immediate
    outcome with [w, 1] recovers r_hat under every reward-probe gauge.
    """
    predicted_delta, predicted_reward = dynamics(obs, action)
    predicted_next_obs = obs + predicted_delta
    predicted_embedding = encoder(predicted_next_obs)
    aligned_embedding = predicted_embedding @ frame_rotation + frame_offset
    immediate_base = base_outcome_features(
        aligned_embedding, predicted_next_obs, action
    )
    predicted_residual = predicted_reward - torch.einsum(
        "...d,d->...", immediate_base, reward_covector
    )
    immediate = torch.cat(
        [immediate_base, predicted_residual.unsqueeze(-1)], -1
    )
    return (1.0 - gamma) * immediate + gamma * state_model(predicted_next_obs)


@torch.no_grad()
def transport_residual_gauge(model, old_covector, new_covector):
    """Change eps_old to eps_new without changing any represented reward."""
    delta = old_covector - new_covector
    weight = model.mean_head.weight
    bias = model.mean_head.bias
    old_last_weight = weight[-1].clone()
    old_last_bias = bias[-1].clone()
    weight[-1].copy_(old_last_weight + delta @ weight[:-1])
    bias[-1].copy_(old_last_bias + delta @ bias[:-1])


def full_suffix_rb_targets(
    outcomes, frozen_tail_means, terminations, boundaries, valids, gamma
):
    """Longest valid normalized suffix, frozen tail at truncation/rollout edge."""
    targets = torch.zeros_like(outcomes)
    target_valids = torch.zeros_like(valids, dtype=torch.bool)
    for t in range(outcomes.shape[0] - 1, -1, -1):
        row_valid = valids[t] > 0
        terminal = terminations[t] > 0
        boundary = boundaries[t] > 0
        if t == outcomes.shape[0] - 1:
            tail = frozen_tail_means[t]
            tail_valid = row_valid
        else:
            tail = torch.where(
                boundary.unsqueeze(-1), frozen_tail_means[t], targets[t + 1]
            )
            tail_valid = torch.where(boundary, row_valid, target_valids[t + 1])
        tail = torch.where(terminal.unsqueeze(-1), torch.zeros_like(tail), tail)
        tail_valid = torch.where(terminal, row_valid, tail_valid)
        targets[t] = (1.0 - gamma) * outcomes[t] + gamma * tail
        target_valids[t] = row_valid & tail_valid
    return targets, target_valids


def doubly_robust_vector_surrogate(
    data_ratio, target, baseline, data_mean, auxiliary_ratio, auxiliary_mean, alpha
):
    """Exact DR estimator centered by the independently predicted state mean."""
    data_term = data_ratio.unsqueeze(-1) * (
        target - baseline - alpha * (data_mean - baseline)
    )
    addback = alpha * (
        auxiliary_ratio.unsqueeze(-1)
        * (auxiliary_mean - baseline.unsqueeze(1))
    ).mean(dim=1)
    return data_term + addback


def distribution_score_proxy(
    actor_dist, latent, first, second, logvar_min=-8.0, logvar_max=8.0
):
    """Score wrt policy-head preactivations, used only to fit scalar alpha."""
    if actor_dist == "beta":
        z = latent.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        common = torch.digamma(first + second)
        score_first = z.log() - torch.digamma(first) + common
        score_second = (1.0 - z).log() - torch.digamma(second) + common
        # d softplus(raw)/d raw recovered from alpha=1+softplus(raw).
        score_first = score_first * (-torch.expm1(-(first - 1.0)))
        score_second = score_second * (-torch.expm1(-(second - 1.0)))
    else:
        variance = second.exp()
        centered = latent - first
        score_first = centered / variance
        score_logvar = 0.5 * (centered.square() / variance - 1.0)
        span = logvar_max - logvar_min
        sigmoid_raw = ((second - logvar_min) / span).clamp(0.0, 1.0)
        score_second = score_logvar * span * sigmoid_raw * (1.0 - sigmoid_raw)
    return torch.cat([score_first, score_second], -1)


def variance_optimal_coordinate_alpha(g0, coordinate_directions, ridge=1e-4):
    """Fit embedding-coordinate controls against policy-head score variance.

    g0 has shape [sample, score], while coordinate_directions has shape
    [sample, outcome, score]. The small dense solve retains cross-coordinate
    covariance instead of scalarizing the outcome before variance reduction.
    """
    g0_centered = g0 - g0.mean(0, keepdim=True)
    direction_centered = (
        coordinate_directions
        - coordinate_directions.mean(0, keepdim=True)
    )
    directions64 = direction_centered.double()
    g064 = g0_centered.double()
    gram = torch.einsum("ndp,nep->de", directions64, directions64)
    covariance = torch.einsum("ndp,np->d", directions64, g064)
    scale = gram.diagonal().mean().clamp_min(1e-12)
    regularized = gram + ridge * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    return torch.linalg.solve(regularized, -covariance).to(g0.dtype)


def analytic_policy_kl(actor_dist, old_first, old_second, new_first, new_second):
    if actor_dist == "beta":
        old_dist, new_dist = Beta(old_first, old_second), Beta(new_first, new_second)
    else:
        old_dist = Normal(old_first, torch.exp(0.5 * old_second))
        new_dist = Normal(new_first, torch.exp(0.5 * new_second))
    return kl_divergence(old_dist, new_dist).sum(-1)


def restore_training_state(module, optimizer, module_state, optimizer_state):
    """Rollback both parameters and Adam moments after a rejected policy epoch."""
    module.load_state_dict(module_state)
    optimizer.load_state_dict(optimizer_state)


def reset_optimizer_output_row(optimizer, linear, row=-1):
    """Zero only Adam moments whose output coordinate underwent the dense gauge."""
    for parameter in (linear.weight, linear.bias):
        if parameter is None:
            continue
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(key)
            if moment is not None:
                moment[row].zero_()


def heldout_env_split(valid, num_envs, fraction, offset=0):
    """Hold out complete independent environment streams, not overlapping suffixes."""
    valid_indices = torch.where(valid.reshape(-1))[0]
    if num_envs < 2:
        holdout_count = min(
            max(1, int(valid_indices.numel() * fraction)),
            max(1, valid_indices.numel() - 1),
        )
        return valid_indices[:holdout_count], valid_indices[holdout_count:]
    holdout_env_count = min(
        max(1, round(num_envs * fraction)), num_envs - 1
    )
    holdout_envs = (
        torch.arange(holdout_env_count, device=valid.device) + offset
    ).remainder(num_envs)
    env_ids = valid_indices.remainder(num_envs)
    holdout_mask = (env_ids[:, None] == holdout_envs[None]).any(1)
    return valid_indices[holdout_mask], valid_indices[~holdout_mask]


def chunk_sequences(x, seq_len):
    steps = (x.shape[0] // seq_len) * seq_len
    x = x[:steps]
    return x.reshape(steps // seq_len, seq_len, x.shape[1], *x.shape[2:]).transpose(
        1, 2
    ).reshape(-1, seq_len, *x.shape[2:])


def outcome_scales(outcomes, floor=0.1, valid=None):
    flat = outcomes.reshape(-1, outcomes.shape[-1])
    if valid is not None:
        flat = flat[valid.reshape(-1) > 0]
    return flat.std(0, unbiased=False).clamp_min(floor)


@torch.no_grad()
def hard_update(target, online):
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    target.eval()


def affine_frame_transport(after, before, current_rotation, current_offset):
    after_mean, before_mean = after.mean(0), before.mean(0)
    cross = (after - after_mean).double().T @ (before - before_mean).double()
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    step = (u @ vh).to(after.dtype)
    rotation = step @ current_rotation
    offset = before_mean @ current_rotation + current_offset - after_mean @ rotation
    return rotation, offset


def sample_auxiliary(agent, obs, count):
    with torch.no_grad():
        dist, first, second = agent.distribution(obs)
        latent = dist.sample((count,)).transpose(0, 1)
        n, k, act_dim = latent.shape
        repeated_obs = obs.unsqueeze(1).expand(n, k, obs.shape[-1]).reshape(n * k, -1)
        flat_latent = latent.reshape(n * k, act_dim)
        action, logprob, _, _, _ = agent.evaluate_latent(repeated_obs, flat_latent)
    return (
        action.reshape(n, k, act_dim),
        latent,
        logprob.reshape(n, k),
        first,
        second,
    )


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.alpha_mode not in {"heldout", "one", "zero"}:
        raise ValueError("--alpha-mode must be heldout, one, or zero")
    if args.auxiliary_actions < 1:
        raise ValueError("--auxiliary-actions must be positive")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_steps % args.seq_len:
        raise ValueError("num_steps must be divisible by seq_len")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
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
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    base_dim = args.emb_dim + obs_dim + 2 * act_dim + 1
    outcome_dim = base_dim + 1

    agent = Agent(envs, args).to(device)
    ssl = LeJepaSSL(obs_dim, act_dim, args).to(device)
    state_model = StateGeometricOutcomeMean(
        obs_dim, outcome_dim, args.model_hidden
    ).to(device)
    state_model_target = copy.deepcopy(state_model).to(device)
    state_model_target.requires_grad_(False)
    state_model_target.eval()
    dynamics = OneStepDynamics(
        obs_dim, act_dim, args.dynamics_hidden
    ).to(device)
    dynamics_target = copy.deepcopy(dynamics).to(device)
    dynamics_target.requires_grad_(False)
    dynamics_target.eval()
    if args.compile:
        agent.trunk = torch.compile(agent.trunk, mode=args.compile_mode)
        state_model.body = torch.compile(state_model.body, mode=args.compile_mode)
        state_model_target.body = torch.compile(
            state_model_target.body, mode=args.compile_mode
        )
        dynamics.body = torch.compile(dynamics.body, mode=args.compile_mode)
        dynamics_target.body = torch.compile(
            dynamics_target.body, mode=args.compile_mode
        )

    actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    state_model_optimizer = optim.AdamW(
        state_model.parameters(),
        lr=args.model_lr,
        weight_decay=args.model_weight_decay,
    )
    dynamics_optimizer = optim.AdamW(
        dynamics.parameters(),
        lr=args.dynamics_lr,
        weight_decay=args.dynamics_weight_decay,
    )

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    next_obs_buf = torch.zeros_like(obs_buf)
    action_buf = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    latent_buf = torch.zeros_like(action_buf)
    logprob_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    reward_buf = torch.zeros_like(logprob_buf)
    termination_buf = torch.zeros_like(logprob_buf)
    boundary_buf = torch.zeros_like(logprob_buf)
    valid_buf = torch.zeros_like(logprob_buf)

    global_step = 0
    start_time = time.time()
    frame_rotation = torch.eye(args.emb_dim, device=device)
    frame_offset = torch.zeros(args.emb_dim, device=device)
    reward_covector = torch.zeros(base_dim, device=device)
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            ssl_optimizer.param_groups[0]["lr"] = fraction * args.ssl_lr
            state_model_optimizer.param_groups[0]["lr"] = fraction * args.model_lr
            dynamics_optimizer.param_groups[0]["lr"] = fraction * args.dynamics_lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, latent, logprob, _, _, _ = agent.get_action(next_obs)
            action_buf[step], latent_buf[step], logprob_buf[step] = action, latent, logprob
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            boundary = np.logical_or(terminations, truncations)
            transition_next_obs = np.array(next_obs_np, copy=True)
            valid = (~boundary).astype(np.float32)
            final_obs, final_mask = infos.get("final_observation"), infos.get(
                "_final_observation"
            )
            if final_obs is not None:
                if final_mask is None:
                    final_mask = [item is not None for item in final_obs]
                for env_index, has_final in enumerate(final_mask):
                    if has_final and final_obs[env_index] is not None:
                        transition_next_obs[env_index] = final_obs[env_index]
                        valid[env_index] = 1.0
            next_obs_buf[step] = torch.as_tensor(
                transition_next_obs, device=device, dtype=torch.float32
            )
            reward_buf[step] = torch.as_tensor(reward, device=device)
            termination_buf[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            boundary_buf[step] = torch.as_tensor(
                boundary, device=device, dtype=torch.float32
            )
            valid_buf[step] = torch.as_tensor(valid, device=device)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_next_obs = next_obs_buf.reshape(-1, obs_dim)
        b_action = action_buf.reshape(-1, act_dim)
        b_latent = latent_buf.reshape(-1, act_dim)
        b_old_logprob = logprob_buf.reshape(-1)
        with torch.no_grad():
            raw_embedding = ssl.encoder(b_next_obs)
            aligned_embedding = raw_embedding @ frame_rotation + frame_offset
            base_phi = base_outcome_features(
                aligned_embedding, b_next_obs, b_action
            ).reshape(args.num_steps, args.num_envs, base_dim)
            outcomes = augment_reward_residual(
                base_phi, reward_buf, reward_covector
            )
            reward_task = torch.cat(
                [reward_covector, torch.ones(1, device=device)]
            )

            auxiliary_action, auxiliary_latent, auxiliary_old_logprob, old_first, old_second = (
                sample_auxiliary(agent, b_obs, args.auxiliary_actions)
            )
            flat_aux_obs = b_obs[:, None].expand(
                -1, args.auxiliary_actions, -1
            ).reshape(-1, obs_dim)
            auxiliary_mean = factorized_action_control(
                state_model_target,
                dynamics_target,
                ssl.encoder,
                flat_aux_obs,
                auxiliary_action.reshape(-1, act_dim),
                frame_rotation,
                frame_offset,
                reward_covector,
                args.gamma,
            ).reshape(args.batch_size, args.auxiliary_actions, outcome_dim)
            # M(s) is the policy-state successor mean.  It is the natural state
            # baseline and avoids injecting finite-auxiliary-sample noise into
            # every behavior-action residual.
            baseline = state_model_target(b_obs)
            data_mean = factorized_action_control(
                state_model_target,
                dynamics_target,
                ssl.encoder,
                b_obs,
                b_action,
                frame_rotation,
                frame_offset,
                reward_covector,
                args.gamma,
            )
            # M is state-conditioned, so rollout and truncation edges need no
            # sampled tail action. Both frozen models still predate this rollout's
            # fitting, preserving the actor/model temporal separation.
            frozen_tail = state_model_target(b_next_obs)
            targets, target_valids = full_suffix_rb_targets(
                outcomes,
                frozen_tail.reshape(args.num_steps, args.num_envs, outcome_dim),
                termination_buf,
                boundary_buf,
                valid_buf,
                args.gamma,
            )

            data_score = distribution_score_proxy(
                args.actor_dist,
                b_latent,
                old_first,
                old_second,
                args.logvar_min,
                args.logvar_max,
            )
            aux_first = old_first[:, None].expand(-1, args.auxiliary_actions, -1)
            aux_second = old_second[:, None].expand(-1, args.auxiliary_actions, -1)
            auxiliary_score = distribution_score_proxy(
                args.actor_dist,
                auxiliary_latent,
                aux_first,
                aux_second,
                args.logvar_min,
                args.logvar_max,
            )

        flat_targets = targets.reshape(-1, outcome_dim)
        flat_valid = target_valids.reshape(-1)
        alpha_indices, actor_indices = heldout_env_split(
            flat_valid,
            args.num_envs,
            args.alpha_holdout_fraction,
            offset=(iteration - 1) * max(
                1, round(args.num_envs * args.alpha_holdout_fraction)
            ),
        )
        if alpha_indices.numel() == 0 or actor_indices.numel() == 0:
            raise RuntimeError("rollout has too few valid transitions for held-out alpha")

        with torch.no_grad():
            primary_vector = flat_targets - baseline
            data_control_vector = data_mean - baseline
            auxiliary_control_vector = (
                auxiliary_mean - baseline.unsqueeze(1)
            )
            primary_scalar = torch.einsum(
                "nd,d->n", primary_vector, reward_task
            )
            g0 = data_score * primary_scalar.unsqueeze(-1)
            auxiliary_coordinate_direction = torch.einsum(
                "njp,njd->ndp",
                auxiliary_score,
                auxiliary_control_vector,
            ) / args.auxiliary_actions
            data_coordinate_direction = (
                data_control_vector.unsqueeze(-1)
                * data_score.unsqueeze(1)
            )
            coordinate_direction = (
                auxiliary_coordinate_direction - data_coordinate_direction
            ) * reward_task.view(1, -1, 1)
            if args.alpha_mode == "heldout":
                alpha = variance_optimal_coordinate_alpha(
                    g0[alpha_indices],
                    coordinate_direction[alpha_indices],
                )
            elif args.alpha_mode == "one":
                alpha = torch.ones(outcome_dim, device=device)
            else:
                alpha = torch.zeros(outcome_dim, device=device)
            base_variance = (
                g0[alpha_indices] - g0[alpha_indices].mean(0, keepdim=True)
            ).square().sum()
            adjusted = g0[alpha_indices] + torch.einsum(
                "ndp,d->np",
                coordinate_direction[alpha_indices],
                alpha,
            )
            adjusted_variance = (
                adjusted - adjusted.mean(0, keepdim=True)
            ).square().sum()
            heldout_target_residual = primary_scalar[alpha_indices]
            heldout_control_residual = torch.einsum(
                "nd,d->n",
                data_control_vector[alpha_indices],
                reward_task,
            )
            target_centered = (
                heldout_target_residual - heldout_target_residual.mean()
            )
            control_centered = (
                heldout_control_residual - heldout_control_residual.mean()
            )
            control_target_covariance = (
                target_centered * control_centered
            ).mean()
            control_target_correlation = control_target_covariance / (
                target_centered.square().mean().sqrt()
                * control_centered.square().mean().sqrt()
            ).clamp_min(1e-12)

        def evaluate_fixed_policy(index_set):
            objective_sum = torch.zeros((), device=device)
            empirical_sum = torch.zeros((), device=device)
            analytic_sum = torch.zeros((), device=device)
            count = 0
            with torch.no_grad():
                for eval_start in range(0, index_set.numel(), args.minibatch_size):
                    mb = index_set[eval_start : eval_start + args.minibatch_size]
                    _, new_logprob, _, new_first, new_second = agent.evaluate_latent(
                        b_obs[mb], b_latent[mb]
                    )
                    repeated_obs = b_obs[mb, None].expand(
                        -1, args.auxiliary_actions, -1
                    ).reshape(-1, obs_dim)
                    _, aux_new_logprob, _, _, _ = agent.evaluate_latent(
                        repeated_obs,
                        auxiliary_latent[mb].reshape(-1, act_dim),
                    )
                    data_logratio = new_logprob - b_old_logprob[mb]
                    aux_logratio = aux_new_logprob.reshape(
                        -1, args.auxiliary_actions
                    ) - auxiliary_old_logprob[mb]
                    data_ratio = data_logratio.exp()
                    vector = doubly_robust_vector_surrogate(
                        data_ratio,
                        flat_targets[mb],
                        baseline[mb],
                        data_mean[mb],
                        aux_logratio.exp(),
                        auxiliary_mean[mb],
                        alpha,
                    )
                    n = mb.numel()
                    objective_sum += torch.einsum(
                        "nd,d->", vector, reward_task
                    )
                    empirical_sum += ((data_ratio - 1.0) - data_logratio).sum()
                    analytic_sum += analytic_policy_kl(
                        args.actor_dist,
                        old_first[mb],
                        old_second[mb],
                        new_first,
                        new_second,
                    ).sum()
                    count += n
            denominator = max(count, 1)
            return (
                objective_sum / denominator,
                empirical_sum / denominator,
                analytic_sum / denominator,
            )

        approx_kl = analytic_kl = pg_loss = entropy_loss = torch.zeros((), device=device)
        actor_grad_norm = torch.zeros((), device=device)
        epochs_taken = 0
        rejected_epochs = 0
        for epoch in range(args.update_epochs):
            policy_snapshot = copy.deepcopy(agent.state_dict())
            optimizer_snapshot = copy.deepcopy(actor_optimizer.state_dict())
            objective_before, _, _ = evaluate_fixed_policy(actor_indices)
            permutation = actor_indices[
                torch.randperm(actor_indices.numel(), device=device)
            ]
            epoch_nonfinite = False
            for start in range(0, permutation.numel(), args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                _, new_logprob, entropy, _, _ = agent.evaluate_latent(
                    b_obs[mb], b_latent[mb]
                )
                repeated_obs = b_obs[mb, None].expand(
                    -1, args.auxiliary_actions, -1
                ).reshape(-1, obs_dim)
                _, aux_new_logprob, _, _, _ = agent.evaluate_latent(
                    repeated_obs,
                    auxiliary_latent[mb].reshape(-1, act_dim),
                )
                data_logratio = new_logprob - b_old_logprob[mb]
                auxiliary_logratio = aux_new_logprob.reshape(
                    -1, args.auxiliary_actions
                ) - auxiliary_old_logprob[mb]
                data_ratio = data_logratio.exp()
                auxiliary_ratio = auxiliary_logratio.exp()
                vector_surrogate = doubly_robust_vector_surrogate(
                    data_ratio,
                    flat_targets[mb],
                    baseline[mb],
                    data_mean[mb],
                    auxiliary_ratio,
                    auxiliary_mean[mb],
                    alpha,
                )
                # The full vector remains intact until this final reward contraction.
                scalar_surrogate = torch.einsum(
                    "nd,d->n", vector_surrogate, reward_task
                )
                pg_loss = -scalar_surrogate.mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss
                if not torch.isfinite(loss):
                    epoch_nonfinite = True
                    break
                actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.actor_grad_clip
                )
                if not torch.isfinite(actor_grad_norm):
                    epoch_nonfinite = True
                    actor_optimizer.zero_grad(set_to_none=True)
                    break
                actor_optimizer.step()
                if not all(
                    torch.isfinite(parameter).all()
                    for parameter in agent.parameters()
                ):
                    epoch_nonfinite = True
                    break

            parameters_finite = all(
                torch.isfinite(parameter).all() for parameter in agent.parameters()
            )
            if epoch_nonfinite or not parameters_finite:
                objective_after = candidate_empirical = candidate_analytic = torch.full(
                    (), float("inf"), device=device
                )
            else:
                try:
                    objective_after, candidate_empirical, candidate_analytic = (
                        evaluate_fixed_policy(actor_indices)
                    )
                except (RuntimeError, ValueError):
                    objective_after = candidate_empirical = candidate_analytic = (
                        torch.full((), float("inf"), device=device)
                    )
                    epoch_nonfinite = True
            metrics_finite = all(
                torch.isfinite(metric)
                for metric in (
                    objective_after,
                    candidate_empirical,
                    candidate_analytic,
                )
            )
            violates_kl = args.target_kl is not None and max(
                candidate_empirical.item(), candidate_analytic.item()
            ) > args.target_kl
            loses_surrogate = objective_after.item() < objective_before.item() - 1e-8
            if epoch_nonfinite or not metrics_finite or violates_kl or loses_surrogate:
                restore_training_state(
                    agent,
                    actor_optimizer,
                    policy_snapshot,
                    optimizer_snapshot,
                )
                rejected_epochs += 1
                _, approx_kl, analytic_kl = evaluate_fixed_policy(actor_indices)
                break
            epochs_taken = epoch + 1
            pg_loss = -objective_after
            approx_kl = candidate_empirical
            analytic_kl = candidate_analytic

        scales = outcome_scales(outcomes, args.outcome_scale_floor, target_valids)
        model_train_indices = actor_indices
        model_loss = model_grad_norm = torch.zeros((), device=device)
        for _ in range(args.model_epochs):
            permutation = model_train_indices[
                torch.randperm(model_train_indices.numel(), device=device)
            ]
            for start in range(0, permutation.numel(), args.model_batch):
                mb = permutation[start : start + args.model_batch]
                prediction = state_model(b_obs[mb])
                model_loss = ((prediction - flat_targets[mb]) / scales).square().mean()
                state_model_optimizer.zero_grad(set_to_none=True)
                model_loss.backward()
                model_grad_norm = nn.utils.clip_grad_norm_(
                    state_model.parameters(), args.model_grad_clip
                )
                state_model_optimizer.step()
        with torch.no_grad():
            heldout_prediction = state_model(b_obs[alpha_indices])
            heldout_mse = (
                (heldout_prediction - flat_targets[alpha_indices]) / scales
            ).square().mean()
            heldout_target = flat_targets[alpha_indices]
            marginal = heldout_target.mean(0, keepdim=True)
            marginal_mse = ((marginal - heldout_target) / scales).square().mean()

        flat_transition_valid = valid_buf.reshape(-1) > 0
        dynamics_holdout_indices, dynamics_train_indices = heldout_env_split(
            flat_transition_valid,
            args.num_envs,
            args.alpha_holdout_fraction,
            offset=(iteration - 1) * max(
                1, round(args.num_envs * args.alpha_holdout_fraction)
            ),
        )
        if (
            dynamics_holdout_indices.numel() == 0
            or dynamics_train_indices.numel() == 0
        ):
            raise RuntimeError("rollout has too few transitions for dynamics split")
        observed_delta = b_next_obs - b_obs
        delta_scales = observed_delta[dynamics_train_indices].std(
            0, unbiased=False
        ).clamp_min(args.delta_scale_floor)
        dynamics_reward_scale = reward_buf.reshape(-1)[
            dynamics_train_indices
        ].std(unbiased=False).clamp_min(args.dynamics_reward_scale_floor)
        dynamics_loss = dynamics_grad_norm = torch.zeros((), device=device)
        dynamics_delta_loss = dynamics_reward_loss = torch.zeros((), device=device)
        for _ in range(args.dynamics_epochs):
            permutation = dynamics_train_indices[
                torch.randperm(dynamics_train_indices.numel(), device=device)
            ]
            for start in range(0, permutation.numel(), args.dynamics_batch):
                mb = permutation[start : start + args.dynamics_batch]
                predicted_delta, predicted_reward = dynamics(
                    b_obs[mb], b_action[mb]
                )
                dynamics_delta_loss = (
                    (predicted_delta - observed_delta[mb]) / delta_scales
                ).square().mean()
                dynamics_reward_loss = (
                    (
                        predicted_reward
                        - reward_buf.reshape(-1)[mb]
                    )
                    / dynamics_reward_scale
                ).square().mean()
                dynamics_loss = dynamics_delta_loss + dynamics_reward_loss
                dynamics_optimizer.zero_grad(set_to_none=True)
                dynamics_loss.backward()
                dynamics_grad_norm = nn.utils.clip_grad_norm_(
                    dynamics.parameters(), args.dynamics_grad_clip
                )
                dynamics_optimizer.step()
        with torch.no_grad():
            heldout_delta = observed_delta[dynamics_holdout_indices]
            predicted_heldout_delta, predicted_heldout_reward = dynamics(
                b_obs[dynamics_holdout_indices],
                b_action[dynamics_holdout_indices],
            )
            dynamics_heldout_mse = (
                (predicted_heldout_delta - heldout_delta) / delta_scales
            ).square().mean()
            marginal_delta = observed_delta[dynamics_train_indices].mean(
                0, keepdim=True
            )
            dynamics_marginal_mse = (
                (marginal_delta - heldout_delta) / delta_scales
            ).square().mean()
            heldout_reward = reward_buf.reshape(-1)[dynamics_holdout_indices]
            dynamics_reward_heldout_mse = (
                (predicted_heldout_reward - heldout_reward)
                / dynamics_reward_scale
            ).square().mean()
            marginal_reward = reward_buf.reshape(-1)[
                dynamics_train_indices
            ].mean()
            dynamics_reward_marginal_mse = (
                (marginal_reward - heldout_reward) / dynamics_reward_scale
            ).square().mean()

        # Only now, after all actor epochs consumed the preceding snapshots, may
        # this rollout's learned models become the next rollout's hard targets.
        hard_update(state_model_target, state_model)
        hard_update(dynamics_target, dynamics)
        new_reward_covector, new_residual = fit_reward_covector(
            base_phi, reward_buf, args.reward_ridge, target_valids
        )
        transport_residual_gauge(
            state_model, reward_covector, new_reward_covector
        )
        transport_residual_gauge(
            state_model_target, reward_covector, new_reward_covector
        )
        # AdamW's diagonal moments have no exact covariance under this dense output
        # gauge.  Retain body state but restart only the transported output head.
        reset_optimizer_output_row(
            state_model_optimizer, state_model.mean_head
        )
        reward_covector = new_reward_covector

        frame_probe = b_obs[:: max(args.batch_size // 2048, 1)][:2048]
        with torch.no_grad():
            frame_before = ssl.encoder(frame_probe)
        seq_obs = chunk_sequences(obs_buf, args.seq_len)
        seq_action = chunk_sequences(action_buf, args.seq_len)
        seq_continue = chunk_sequences(1.0 - boundary_buf, args.seq_len)
        seq_mask = seq_continue.cumprod(1)[:, :-1]
        ssl_prediction = ssl_sigreg = ssl_grad_norm = torch.zeros((), device=device)
        ssl_steps = 0
        for _ in range(args.ssl_epochs):
            permutation = torch.randperm(seq_obs.shape[0], device=device)
            for start in range(0, seq_obs.shape[0] - args.ssl_batch + 1, args.ssl_batch):
                mb = permutation[start : start + args.ssl_batch]
                ssl_loss, ssl_prediction, ssl_sigreg = ssl(
                    seq_obs[mb], seq_action[mb], seq_mask[mb], args.sigreg_weight
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_grad_norm = nn.utils.clip_grad_norm_(
                    ssl.parameters(), args.ssl_grad_clip
                )
                ssl_optimizer.step()
                ssl_steps += 1
        with torch.no_grad():
            frame_after = ssl.encoder(frame_probe)
            frame_rotation, frame_offset = affine_frame_transport(
                frame_after, frame_before, frame_rotation, frame_offset
            )

        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("actor/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("actor/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("actor/empirical_kl", approx_kl.item(), global_step)
        writer.add_scalar("actor/analytic_kl", analytic_kl.item(), global_step)
        writer.add_scalar("actor/epochs_taken", epochs_taken, global_step)
        writer.add_scalar("actor/rejected_epochs", rejected_epochs, global_step)
        writer.add_scalar("actor/grad_norm", float(actor_grad_norm), global_step)
        writer.add_scalar("dr/alpha", alpha.mean().item(), global_step)
        writer.add_scalar("dr/alpha_std", alpha.std(unbiased=False).item(), global_step)
        writer.add_scalar("dr/alpha_rms", alpha.square().mean().sqrt().item(), global_step)
        writer.add_scalar(
            "dr/heldout_variance_ratio",
            (adjusted_variance / base_variance.clamp_min(1e-12)).item(),
            global_step,
        )
        with torch.no_grad():
            scalar_auxiliary_control = torch.einsum(
                "nkd,d->nk", auxiliary_mean, reward_task
            )
            control_action_spread = scalar_auxiliary_control.std(
                1, unbiased=False
            ).mean()
        writer.add_scalar(
            "dr/control_action_spread", control_action_spread.item(), global_step
        )
        writer.add_scalar(
            "dr/control_target_covariance",
            control_target_covariance.item(),
            global_step,
        )
        writer.add_scalar(
            "dr/control_target_correlation",
            control_target_correlation.item(),
            global_step,
        )
        writer.add_scalar("state_model/loss", model_loss.item(), global_step)
        writer.add_scalar(
            "state_model/heldout_mse", heldout_mse.item(), global_step
        )
        writer.add_scalar(
            "state_model/heldout_skill",
            (1.0 - heldout_mse / marginal_mse.clamp_min(1e-12)).item(),
            global_step,
        )
        writer.add_scalar(
            "state_model/grad_norm", float(model_grad_norm), global_step
        )
        writer.add_scalar("dynamics/loss", dynamics_loss.item(), global_step)
        writer.add_scalar(
            "dynamics/delta_loss", dynamics_delta_loss.item(), global_step
        )
        writer.add_scalar(
            "dynamics/reward_loss", dynamics_reward_loss.item(), global_step
        )
        writer.add_scalar(
            "dynamics/heldout_mse", dynamics_heldout_mse.item(), global_step
        )
        writer.add_scalar(
            "dynamics/heldout_delta_skill",
            (
                1.0
                - dynamics_heldout_mse
                / dynamics_marginal_mse.clamp_min(1e-12)
            ).item(),
            global_step,
        )
        writer.add_scalar(
            "dynamics/grad_norm", float(dynamics_grad_norm), global_step
        )
        writer.add_scalar(
            "dynamics/heldout_reward_skill",
            (
                1.0
                - dynamics_reward_heldout_mse
                / dynamics_reward_marginal_mse.clamp_min(1e-12)
            ).item(),
            global_step,
        )
        writer.add_scalar("reward/residual_std", new_residual.std().item(), global_step)
        writer.add_scalar("ssl/prediction_loss", ssl_prediction.item(), global_step)
        writer.add_scalar("ssl/sigreg", ssl_sigreg.item(), global_step)
        writer.add_scalar("ssl/grad_norm", float(ssl_grad_norm), global_step)
        writer.add_scalar("ssl/steps", ssl_steps, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
