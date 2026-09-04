# LeJEPA geometric doubly-robust policy gradient v4.
#
# An attached LeJEPA encoder (prediction MSE + SIGReg, no teacher) defines transition
# features.  An action-conditioned model predicts their normalized gamma-successor mean
# from the longest available same-episode suffix.  The Beta actor is optimized with an
# exact vector doubly-robust estimator: an independently sampled behavior action adds the
# model expectation back after subtracting the model at the observed action.  Only the
# batch-mean vector is contracted with the immediate-reward covector.  There is no scalar
# temporal critic, GAE, Q learning, flow model, EMA, contrastive loss, or PopArt.
import copy
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
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import ActionEncoder, SIGReg, StateEncoder


EPS = 1e-6
GEOMETRIC_GAMMA = 0.9970087504549047


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    model_lr: float = 5e-5
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = GEOMETRIC_GAMMA
    num_minibatches: int = 32
    update_epochs: int = 10
    reward_norm: bool = False

    ent_coef: float = 0.0
    target_kl: float = 0.01
    max_grad_norm: float = 0.5

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    emb_dim: int = 32
    model_hidden: int = 256
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    sigreg_batch: int = 1024
    model_weight_decay: float = 1e-3
    model_grad_clip: float = 1.0
    reward_ridge: float = 1e-5
    alpha_mode: str = "heldout"  # heldout, one, zero
    alpha_ridge: float = 1e-4

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=False):
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
        if reward_norm:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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


def _branch_body(hidden):
    return nn.Sequential(
        layer_init(nn.Linear(hidden, hidden)),
        ReLUSquared(),
        layer_init(nn.Linear(hidden, hidden)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, hidden, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, hidden))
        self.resid_gate = nn.Parameter(torch.full((hidden,), 4.0))
        self.dense_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.dense = _branch_body(hidden)
        self.moe_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(hidden, n_experts))
        self.experts = nn.ModuleList(
            [_branch_body(hidden) for _ in range(n_experts)]
        )

    def forward(self, cat_features, x0):
        x = self.in_proj(cat_features)
        gate = torch.sigmoid(self.resid_gate)
        x = gate * x + (1.0 - gate) * x0
        dense = self.dense(self.dense_norm(x))
        moe_input = self.moe_norm(x)
        weights = torch.softmax(self.gate(moe_input), dim=-1)
        experts = torch.stack([expert(moe_input) for expert in self.experts], dim=1)
        return x + dense + (weights.unsqueeze(-1) * experts).sum(1)


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, hidden, blocks, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.ModuleList(
            [
                ThinkBlock(hidden * (index + 1), hidden, n_experts)
                for index in range(blocks)
            ]
        )
        self.out_norm = nn.RMSNorm(
            hidden * (blocks + 1), elementwise_affine=False
        )
        self.out_proj = layer_init(nn.Linear(hidden * (blocks + 1), hidden))

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(features, dim=-1)))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.trunk = ThinkTrunk(
            obs_dim, args.hidden, args.k_blocks, args.n_experts
        )
        self.alpha_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)

    def distribution(self, obs):
        hidden = self.trunk(obs)
        alpha = 1.0 + F.softplus(self.alpha_head(hidden))
        beta = 1.0 + F.softplus(self.beta_head(hidden))
        return Beta(alpha, beta), alpha, beta

    def forward(self, obs, native_action=None):
        distribution, alpha, beta = self.distribution(obs)
        if native_action is None:
            native_action = distribution.sample().clamp(EPS, 1.0 - EPS)
        log_prob = distribution.log_prob(native_action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        action = 2.0 * native_action - 1.0
        return native_action, action, log_prob, entropy, alpha, beta


def transition_features(next_embedding, obs, next_obs, action):
    """Reward-complete vector: nonlinear state, finite difference, controls, intercept."""
    ones = next_embedding.new_ones(next_embedding.shape[:-1] + (1,))
    return torch.cat(
        [
            next_embedding,
            obs,
            next_obs - obs,
            action,
            action.square(),
            ones,
        ],
        dim=-1,
    )


class GeometricSuccessor(nn.Module):
    """Attached LeJEPA and an action-conditioned normalized successor mean."""

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.encoder = StateEncoder(obs_dim, args.emb_dim, args.model_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.feature_dim = args.emb_dim + 2 * obs_dim + 2 * act_dim + 1
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(2 * args.emb_dim, args.model_hidden)),
            nn.GELU(),
            layer_init(nn.Linear(args.model_hidden, args.model_hidden)),
            nn.GELU(),
            layer_init(nn.Linear(args.model_hidden, self.feature_dim), std=0.01),
        )
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj,
            proj_chunk=args.sigreg_proj_chunk,
        )

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, obs, action):
        embedding = self.encoder(obs)
        action_embedding = self.action_encoder(action)
        return self.predictor(torch.cat([embedding, action_embedding], dim=-1))

    def attached_loss(
        self,
        obs,
        next_obs,
        actions,
        continuation,
        bootstrap_mask,
        bootstrap_successor,
        gamma,
        sigreg_weight,
        sigreg_batch,
    ):
        time_steps, num_envs = obs.shape[:2]
        flat_obs = obs.flatten(0, 1)
        flat_next_obs = next_obs.flatten(0, 1)
        embedding = self.encoder(flat_obs).view(time_steps, num_envs, -1)
        next_embedding = self.encoder(flat_next_obs).view(time_steps, num_envs, -1)
        features = transition_features(
            next_embedding, obs, next_obs, actions
        )
        targets = longest_suffix_targets(
            features,
            bootstrap_successor,
            continuation,
            bootstrap_mask,
            gamma,
        )
        predictions = self.predict(flat_obs, actions.flatten(0, 1)).view_as(targets)
        prediction_loss = F.mse_loss(predictions, targets)

        sig_embeddings = torch.cat(
            [embedding.flatten(0, 1), next_embedding.flatten(0, 1)], dim=0
        )
        count = min(sigreg_batch, sig_embeddings.shape[0])
        sig_indices = torch.randperm(
            sig_embeddings.shape[0], device=sig_embeddings.device
        )[:count]
        # SIGReg requires (T, B, D); this is one population at one logical time.
        sigreg_loss = self.sigreg(sig_embeddings[sig_indices].unsqueeze(0))
        return (
            prediction_loss + sigreg_weight * sigreg_loss,
            prediction_loss,
            sigreg_loss,
            targets,
            features,
        )


def longest_suffix_targets(
    features,
    next_successor,
    continuation,
    bootstrap_mask,
    gamma,
):
    """Normalized longest suffix, bootstrapping only truncations and rollout edges."""
    if features.shape[:2] != continuation.shape:
        raise ValueError("features and continuation time/environment axes must match")
    targets = torch.empty_like(features)
    running = torch.zeros_like(features[0])
    for step in reversed(range(features.shape[0])):
        if step == features.shape[0] - 1:
            tail = bootstrap_mask[step].unsqueeze(-1) * next_successor[step]
        else:
            tail = (
                continuation[step].unsqueeze(-1) * running
                + bootstrap_mask[step].unsqueeze(-1) * next_successor[step]
            )
        running = (1.0 - gamma) * features[step] + gamma * tail
        targets[step] = running
    return targets


def beta_parameter_score(native_action, alpha, beta):
    total = alpha + beta
    score_alpha = (
        native_action.clamp_min(EPS).log()
        - torch.digamma(alpha)
        + torch.digamma(total)
    )
    score_beta = (
        (1.0 - native_action).clamp_min(EPS).log()
        - torch.digamma(beta)
        + torch.digamma(total)
    )
    return torch.cat([score_alpha, score_beta], dim=-1)


def crossfit_alpha(
    target,
    observed_model,
    auxiliary_model,
    observed_score,
    auxiliary_score,
    env_index,
    mode,
    ridge=1e-4,
):
    """Cross-fit the scalar minimizing native-Beta gradient variance per env fold."""
    if mode not in {"heldout", "one", "zero"}:
        raise ValueError("alpha_mode must be heldout, one, or zero")
    if mode == "one":
        return target.new_ones(target.shape[0]), target.new_ones(2)
    if mode == "zero":
        return target.new_zeros(target.shape[0]), target.new_zeros(2)

    anchor = observed_score.unsqueeze(-1) * target.unsqueeze(-2)
    control = (
        auxiliary_score.unsqueeze(-1) * auxiliary_model.unsqueeze(-2)
        - observed_score.unsqueeze(-1) * observed_model.unsqueeze(-2)
    )
    per_row = target.new_zeros(target.shape[0])
    fold_values = target.new_zeros(2)
    for fold in (0, 1):
        heldout = (env_index % 2) == fold
        source = ~heldout
        if source.sum() < 2:
            continue
        anchor_source = anchor[source]
        control_source = control[source]
        anchor_source = anchor_source - anchor_source.mean(0, keepdim=True)
        control_source = control_source - control_source.mean(0, keepdim=True)
        numerator = -(anchor_source * control_source).sum()
        denominator = control_source.square().sum()
        scale = denominator.detach() / max(int(source.sum()), 1)
        coefficient = numerator / (denominator + ridge * scale.clamp_min(EPS))
        coefficient = torch.where(
            torch.isfinite(coefficient), coefficient, coefficient.new_zeros(())
        )
        per_row[heldout] = coefficient
        fold_values[fold] = coefficient
    return per_row.detach(), fold_values.detach()


def vector_doubly_robust_surrogate(
    observed_ratio,
    auxiliary_ratio,
    target,
    observed_model,
    auxiliary_model,
    alpha,
):
    """Unbiased vector IS objective plus an exact independent-action correction."""
    alpha = alpha.unsqueeze(-1)
    corrected = (
        observed_ratio.unsqueeze(-1) * target
        + alpha
        * (
            auxiliary_ratio.unsqueeze(-1) * auxiliary_model
            - observed_ratio.unsqueeze(-1) * observed_model
        )
    )
    return corrected.mean(0)


def solve_reward_covector(features, rewards, ridge):
    """Rollout-local linear reward functional; no temporal scalar regression."""
    features64 = features.double()
    rewards64 = rewards.double()
    gram = features64.T @ features64
    scale = gram.diagonal().mean().clamp_min(1e-12)
    regularizer = ridge * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    return torch.linalg.solve(
        gram + regularizer, features64.T @ rewards64
    ).to(features.dtype)


def explained_variance(prediction, target):
    target_variance = target.var().clamp_min(1e-12)
    return 1.0 - (target - prediction).var() / target_variance


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.alpha_mode not in {"heldout", "one", "zero"}:
        raise ValueError("--alpha-mode must be heldout, one, or zero")
    if args.gamma != GEOMETRIC_GAMMA:
        raise ValueError(
            f"v4 fixes the geometric horizon at gamma={GEOMETRIC_GAMMA}"
        )
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

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
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    assert device.type == "cuda", "this research implementation requires CUDA"

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                index,
                args.capture_video,
                run_name,
                args.gamma,
                args.reward_norm,
            )
            for index in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    actor = Actor(obs_dim, act_dim, args).to(device)
    successor = GeometricSuccessor(obs_dim, act_dim, args).to(device)
    actor_optimizer = optim.Adam(
        actor.parameters(), lr=args.learning_rate, eps=1e-5
    )
    model_optimizer = optim.AdamW(
        successor.parameters(),
        lr=args.model_lr,
        weight_decay=args.model_weight_decay,
    )
    if args.compile:
        actor = torch.compile(actor, mode=args.compile_mode)

    shape_obs = (
        args.num_steps,
        args.num_envs,
    ) + envs.single_observation_space.shape
    shape_act = (
        args.num_steps,
        args.num_envs,
    ) + envs.single_action_space.shape
    observations = torch.zeros(shape_obs, device=device)
    next_observations = torch.zeros_like(observations)
    native_actions = torch.zeros(shape_act, device=device)
    actions = torch.zeros_like(native_actions)
    auxiliary_native_actions = torch.zeros_like(native_actions)
    auxiliary_actions = torch.zeros_like(actions)
    old_log_probs = torch.zeros(
        (args.num_steps, args.num_envs), device=device
    )
    auxiliary_old_log_probs = torch.zeros_like(old_log_probs)
    old_alphas = torch.zeros_like(native_actions)
    old_betas = torch.zeros_like(native_actions)
    rewards = torch.zeros_like(old_log_probs)
    continuation = torch.zeros_like(old_log_probs)
    bootstrap_mask = torch.zeros_like(old_log_probs)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(
        next_obs_np, device=device, dtype=torch.float32
    )
    print(
        "[lejepa_gdr_v4] normalized full-suffix successors + "
        f"vector DR actor (alpha={args.alpha_mode})"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = (
                fraction * args.learning_rate
            )
            model_optimizer.param_groups[0]["lr"] = fraction * args.model_lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            with torch.no_grad():
                (
                    native_action,
                    action,
                    log_prob,
                    _,
                    alpha,
                    beta,
                ) = actor(next_obs)
                (
                    auxiliary_native,
                    auxiliary_action,
                    auxiliary_log_prob,
                    _,
                    _,
                    _,
                ) = actor(next_obs)
            native_actions[step] = native_action
            actions[step] = action
            auxiliary_native_actions[step] = auxiliary_native
            auxiliary_actions[step] = auxiliary_action
            old_log_probs[step] = log_prob
            auxiliary_old_log_probs[step] = auxiliary_log_prob
            old_alphas[step] = alpha
            old_betas[step] = beta

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            boundary = np.logical_or(terminated, truncated)
            transition_next = np.array(next_obs_np, copy=True)
            final_observation = infos.get("final_observation")
            final_mask = infos.get("_final_observation")
            if final_observation is not None:
                if final_mask is None:
                    final_mask = [
                        item is not None for item in final_observation
                    ]
                for env_index, has_final in enumerate(final_mask):
                    if has_final and final_observation[env_index] is not None:
                        transition_next[env_index] = final_observation[env_index]

            rewards[step] = torch.as_tensor(
                reward, device=device, dtype=torch.float32
            )
            continuation[step] = torch.as_tensor(
                ~boundary, device=device, dtype=torch.float32
            )
            bootstrap_mask[step] = torch.as_tensor(
                truncated, device=device, dtype=torch.float32
            )
            next_observations[step] = torch.as_tensor(
                transition_next, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(
                next_obs_np, device=device, dtype=torch.float32
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={episodic_return}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return",
                            episodic_return,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )

        # The last unfinished row is the sole ordinary rollout-edge bootstrap.
        bootstrap_mask[-1] = torch.maximum(
            bootstrap_mask[-1], continuation[-1]
        )
        flat_obs = observations.flatten(0, 1)
        flat_next_obs = next_observations.flatten(0, 1)
        flat_action = actions.flatten(0, 1)
        flat_native = native_actions.flatten(0, 1)
        flat_aux_action = auxiliary_actions.flatten(0, 1)
        flat_aux_native = auxiliary_native_actions.flatten(0, 1)
        flat_reward = rewards.flatten()

        # Everything consumed by the actor is rollout-lagged and detached.
        with torch.no_grad():
            _, bootstrap_action, _, _, _, _ = actor(flat_next_obs)
            next_successor = successor.predict(
                flat_next_obs, bootstrap_action
            ).view(args.num_steps, args.num_envs, -1)
            next_embedding = successor.encode(flat_next_obs).view(
                args.num_steps, args.num_envs, -1
            )
            vector_features = transition_features(
                next_embedding,
                observations,
                next_observations,
                actions,
            )
            vector_target = longest_suffix_targets(
                vector_features,
                next_successor,
                continuation,
                bootstrap_mask,
                args.gamma,
            ).flatten(0, 1)
            observed_model = successor.predict(flat_obs, flat_action)
            auxiliary_model = successor.predict(flat_obs, flat_aux_action)
            reward_covector = solve_reward_covector(
                vector_features.flatten(0, 1),
                flat_reward,
                args.reward_ridge,
            )
            reward_prediction = (
                vector_features.flatten(0, 1) @ reward_covector
            )
            observed_score = beta_parameter_score(
                flat_native,
                old_alphas.flatten(0, 1),
                old_betas.flatten(0, 1),
            )
            auxiliary_score = beta_parameter_score(
                flat_aux_native,
                old_alphas.flatten(0, 1),
                old_betas.flatten(0, 1),
            )
            environment_index = (
                torch.arange(args.batch_size, device=device) % args.num_envs
            )
            alpha_rows, alpha_folds = crossfit_alpha(
                vector_target,
                observed_model,
                auxiliary_model,
                observed_score,
                auxiliary_score,
                environment_index,
                args.alpha_mode,
                args.alpha_ridge,
            )

        actor_snapshot = copy.deepcopy(actor.state_dict())
        optimizer_snapshot = copy.deepcopy(actor_optimizer.state_dict())
        permutation = np.arange(args.batch_size)
        actor_steps = 0
        policy_loss_value = 0.0
        entropy_value = 0.0
        for _ in range(args.update_epochs):
            np.random.shuffle(permutation)
            for start in range(0, args.batch_size, args.minibatch_size):
                minibatch = permutation[start : start + args.minibatch_size]
                (
                    _,
                    _,
                    new_log_prob,
                    entropy,
                    _,
                    _,
                ) = actor(flat_obs[minibatch], flat_native[minibatch])
                (
                    _,
                    _,
                    new_aux_log_prob,
                    _,
                    _,
                    _,
                ) = actor(flat_obs[minibatch], flat_aux_native[minibatch])
                observed_ratio = (
                    new_log_prob - old_log_probs.flatten()[minibatch]
                ).exp()
                auxiliary_ratio = (
                    new_aux_log_prob
                    - auxiliary_old_log_probs.flatten()[minibatch]
                ).exp()
                vector_objective = vector_doubly_robust_surrogate(
                    observed_ratio,
                    auxiliary_ratio,
                    vector_target[minibatch],
                    observed_model[minibatch],
                    auxiliary_model[minibatch],
                    alpha_rows[minibatch],
                )
                # This is deliberately the first scalarization of policy credit.
                policy_loss = -(
                    vector_objective @ reward_covector
                ) / (1.0 - args.gamma)
                policy_loss = policy_loss - args.ent_coef * entropy.mean()
                actor_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    actor.parameters(), args.max_grad_norm
                )
                actor_optimizer.step()
                actor_steps += 1
                policy_loss_value = float(policy_loss.detach())
                entropy_value = float(entropy.mean().detach())

        with torch.no_grad():
            kl_sum = 0.0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = min(start + args.minibatch_size, args.batch_size)
                _, new_alpha, new_beta = actor.distribution(
                    flat_obs[start:end]
                )
                old_distribution = Beta(
                    old_alphas.flatten(0, 1)[start:end],
                    old_betas.flatten(0, 1)[start:end],
                )
                kl_sum += float(
                    kl_divergence(
                        old_distribution, Beta(new_alpha, new_beta)
                    )
                    .sum(-1)
                    .sum()
                )
            analytic_kl = kl_sum / args.batch_size
        actor_rollback = int(
            not np.isfinite(analytic_kl)
            or analytic_kl > args.target_kl
        )
        if actor_rollback:
            actor.load_state_dict(actor_snapshot)
            actor_optimizer.load_state_dict(optimizer_snapshot)
            actor_steps = 0

        # Update the attached representation only after its lagged predictions are used.
        with torch.no_grad():
            model_bootstrap = successor.predict(
                flat_next_obs, bootstrap_action
            ).view(args.num_steps, args.num_envs, -1)
        (
            model_loss,
            prediction_loss,
            sigreg_loss,
            _,
            _,
        ) = successor.attached_loss(
            observations,
            next_observations,
            actions,
            continuation,
            bootstrap_mask,
            model_bootstrap,
            args.gamma,
            args.sigreg_weight,
            args.sigreg_batch,
        )
        model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        model_grad_norm = nn.utils.clip_grad_norm_(
            successor.parameters(), args.model_grad_clip
        )
        model_optimizer.step()

        writer.add_scalar(
            "charts/learning_rate",
            actor_optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        writer.add_scalar(
            "losses/vector_policy", policy_loss_value, global_step
        )
        writer.add_scalar("losses/analytic_kl", analytic_kl, global_step)
        writer.add_scalar(
            "losses/actor_rollback", actor_rollback, global_step
        )
        writer.add_scalar("losses/actor_steps", actor_steps, global_step)
        writer.add_scalar("losses/entropy", entropy_value, global_step)
        writer.add_scalar(
            "model/successor_prediction",
            float(prediction_loss.detach()),
            global_step,
        )
        writer.add_scalar(
            "model/sigreg", float(sigreg_loss.detach()), global_step
        )
        writer.add_scalar(
            "model/grad_norm", float(model_grad_norm), global_step
        )
        writer.add_scalar(
            "model/reward_probe_ev",
            float(explained_variance(reward_prediction, flat_reward)),
            global_step,
        )
        writer.add_scalar(
            "dr/alpha_fold_0", float(alpha_folds[0]), global_step
        )
        writer.add_scalar(
            "dr/alpha_fold_1", float(alpha_folds[1]), global_step
        )
        writer.add_scalar(
            "dr/correction_rms",
            float(
                (
                    alpha_rows.unsqueeze(-1)
                    * (auxiliary_model - observed_model)
                )
                .square()
                .mean()
                .sqrt()
            ),
            global_step,
        )

    envs.close()
    writer.close()
