# LeJEPA geometric control-variate policy gradient v2.
#
# The reward gradient uses the longest available same-trajectory TD return, bootstrapping
# only at the rollout edge: no GAE, Q learning, or moving return normalization. A lagged
# action-conditioned LeJEPA predicts both the next embedding and its normalized discounted
# successor embedding via TD. Their action Jacobian defines a low-rank linear control
# variate. The behavior-action term subtracts that variate and an analytic Beta-mean term
# adds its expectation back, so model error changes variance rather than the policy target.
# Coefficients and their variance gate are cross-fit across disjoint environments.
#
# The actor uses unclipped likelihood ratios, a full-rollout analytic KL leash, and
# rollback. The LeJEPA model is trained only after its lagged predictions are consumed.
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

from cleanrl.shared.lejepa import (
    ActionEncoder,
    SIGReg,
    StateEncoder,
)

EPS = 1e-6


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
    actor_lr: float = None
    v_lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    num_minibatches: int = 32
    actor_epochs: int = 16
    actor_backtracks: int = 4
    actor_backtrack_factor: float = 0.5
    actor_kl_fill: float = 0.8
    v_epochs: int = 4
    reward_norm: bool = False

    ent_coef: float = 0.0
    kl_target: float = 0.0075
    max_grad_norm: float = 0.5

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    v_hidden: int = 256

    ssl_warmup_iters: int = 4
    emb_dim: int = 32
    ssl_hidden: int = 256
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    ssl_lr: float = 5e-5
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 512
    ssl_steps: int = 32
    ssl_grad_clip: float = 1.0
    ssl_successor_weight: float = 1.0
    cv_rank: int = 16
    cv_ridge: float = 1e-2
    cv_fd_eps: float = 0.05
    cv_min_variance_gain: float = 0.02
    cv_feature_chunk: int = 4096

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
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([ex(m_in) for ex in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList([ThinkBlock(H * (k + 1), H, n_experts) for k in range(K)])
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class LeJepaSSL(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.encoder = StateEncoder(obs_dim, args.emb_dim, args.ssl_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.next_predictor = nn.Sequential(
            layer_init(nn.Linear(2 * args.emb_dim, args.ssl_hidden)),
            nn.GELU(),
            layer_init(nn.Linear(args.ssl_hidden, args.ssl_hidden)),
            nn.GELU(),
            layer_init(nn.Linear(args.ssl_hidden, args.emb_dim), std=0.01),
        )
        self.successor_predictor = copy.deepcopy(self.next_predictor)
        self.sigreg = SIGReg(num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk)

    def encode(self, obs):
        return self.encoder(obs)

    def _model_input(self, e, a):
        action_emb = self.action_encoder(a)
        return torch.cat([e, action_emb], dim=-1)

    def predict_next(self, e, a):
        return e + self.next_predictor(self._model_input(e, a))

    def predict_successor(self, e, a):
        return e + self.successor_predictor(self._model_input(e, a))

    def predict_features(self, e, a):
        return torch.cat(
            [self.predict_next(e, a), self.predict_successor(e, a)],
            dim=-1,
        )

    def forward(
        self,
        obs,
        actions,
        target_obs,
        successor_target,
        sigreg_weight,
        successor_weight,
    ):
        emb = self.encoder(obs)
        target_emb = self.encoder(target_obs)
        next_prediction = self.predict_next(emb, actions)
        successor_prediction = self.predict_successor(emb, actions)
        next_loss = F.mse_loss(next_prediction, target_emb)
        successor_loss = F.mse_loss(
            successor_prediction, successor_target
        )
        sig_embeddings = torch.cat([emb, target_emb], dim=0)
        sigreg_loss = self.sigreg(sig_embeddings.unsqueeze(0))
        return (
            next_loss
            + successor_weight * successor_loss
            + sigreg_weight * sigreg_loss,
            next_loss,
            successor_loss,
            sigreg_loss,
        )


class LongHorizonValue(nn.Module):
    """State-only baseline for the longest available rollout TD target."""

    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, 1), std=0.01),
        )

    def forward(self, e):
        return self.net(e).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)

    def forward(self, x, z=None):
        h = self.trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, entropy, alpha, beta


def longest_horizon_td_returns(rewards, boundaries, last_value, gamma):
    """Use every transition and bootstrap only unfinished paths at the rollout edge."""
    returns = torch.zeros_like(rewards)
    running = last_value
    for t in reversed(range(rewards.shape[0])):
        boundary = boundaries[t].bool()
        running = rewards[t] + gamma * (~boundary).to(rewards.dtype) * running
        returns[t] = running
    return returns


def successor_td_target(
    next_embedding, next_successor, boundaries, gamma
):
    """Normalized discounted successor embedding with zero continuation at boundaries."""
    return (
        (1.0 - gamma) * next_embedding
        + gamma
        * (1.0 - boundaries).unsqueeze(-1)
        * next_successor
    )


def beta_parameter_score(z, alpha, beta):
    """Score of a factorized Beta sample with respect to alpha and beta."""
    total = alpha + beta
    score_alpha = z.clamp_min(EPS).log() - torch.digamma(alpha) + torch.digamma(total)
    score_beta = (
        (1.0 - z).clamp_min(EPS).log()
        - torch.digamma(beta)
        + torch.digamma(total)
    )
    return torch.cat([score_alpha, score_beta], dim=-1)


def beta_action_mean_and_derivatives(alpha, beta):
    """Mean action in [-1,1] and derivatives with respect to native Beta parameters."""
    total_sq = (alpha + beta).square()
    mean = 2.0 * alpha / (alpha + beta) - 1.0
    d_alpha = 2.0 * beta / total_sq
    d_beta = -2.0 * alpha / total_sq
    return mean, d_alpha, d_beta


def fit_crossfit_control_variate(
    raw_features,
    raw_jacobian,
    advantage,
    z,
    alpha,
    beta,
    valid,
    env_index,
    rank,
    ridge,
    min_variance_gain,
):
    """Cross-fit a low-rank geometric control variate by Beta-score variance."""
    n, feature_dim = raw_features.shape
    act_dim = z.shape[-1]
    score = beta_parameter_score(z, alpha, beta)
    _, d_alpha, d_beta = beta_action_mean_and_derivatives(alpha, beta)
    behavior_control = raw_features.new_zeros(n)
    action_gradient = raw_features.new_zeros(n, act_dim)
    anchor_gradient = score * advantage.unsqueeze(-1)
    fitted = torch.zeros(n, dtype=torch.bool, device=raw_features.device)
    retained_ranks = []

    for fold in (0, 1):
        hold = valid & ((env_index % 2) == fold)
        source = valid & ((env_index % 2) != fold)
        # Fit and gate on disjoint source environments. Neither the coefficient nor
        # its on/off decision can therefore depend on a held-out action it affects.
        source_group = (env_index // 2) % 2
        train = source & (source_group == 0)
        gate = source & (source_group == 1)
        if (
            train.sum() <= max(rank, 2)
            or gate.sum() < 2
            or hold.sum() == 0
        ):
            continue
        x_train = raw_features[train]
        covariance = x_train.T @ x_train / x_train.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        keep = min(rank, int((eigenvalues > 1e-8).sum().item()))
        if keep == 0:
            continue
        values = eigenvalues[-keep:].clamp_min(1e-8)
        projection = eigenvectors[:, -keep:] / values.sqrt().unsqueeze(0)

        x_proj = raw_features[train] @ projection
        jac_proj = torch.einsum(
            "nfa,fr->nra", raw_jacobian[train], projection
        )
        add_alpha = (
            jac_proj.transpose(1, 2) * d_alpha[train].unsqueeze(-1)
        )
        add_beta = jac_proj.transpose(1, 2) * d_beta[train].unsqueeze(-1)
        add_matrix = torch.cat([add_alpha, add_beta], dim=1)
        control_matrix = (
            -score[train].unsqueeze(-1) * x_proj.unsqueeze(1) + add_matrix
        )
        y = anchor_gradient[train]
        normal = torch.einsum("npr,nps->rs", control_matrix, control_matrix)
        normal = normal + ridge * torch.eye(
            keep, device=normal.device, dtype=normal.dtype
        )
        rhs = -torch.einsum("npr,np->r", control_matrix, y)
        coefficient = torch.linalg.solve(normal, rhs)

        x_gate = raw_features[gate] @ projection
        jac_gate = torch.einsum(
            "nfa,fr->nra", raw_jacobian[gate], projection
        )
        gate_control = x_gate @ coefficient
        gate_action_gradient = torch.einsum(
            "nra,r->na", jac_gate, coefficient
        )
        gate_addback = torch.cat(
            [
                gate_action_gradient * d_alpha[gate],
                gate_action_gradient * d_beta[gate],
            ],
            dim=-1,
        )
        gate_corrected = (
            anchor_gradient[gate]
            - score[gate] * gate_control.unsqueeze(-1)
            + gate_addback
        )
        gate_anchor_var = (
            anchor_gradient[gate].var(dim=0, unbiased=False).sum().clamp_min(1e-12)
        )
        gate_ratio = float(
            gate_corrected.var(dim=0, unbiased=False).sum() / gate_anchor_var
        )
        if (
            not np.isfinite(gate_ratio)
            or gate_ratio > 1.0 - min_variance_gain
        ):
            continue

        x_hold = raw_features[hold] @ projection
        jac_hold = torch.einsum(
            "nfa,fr->nra", raw_jacobian[hold], projection
        )
        behavior_control[hold] = x_hold @ coefficient
        action_gradient[hold] = torch.einsum(
            "nra,r->na", jac_hold, coefficient
        )
        fitted[hold] = True
        retained_ranks.append(keep)

    evaluated = valid & fitted
    if evaluated.sum() < 2:
        return behavior_control.zero_(), action_gradient.zero_(), 1.0, 0
    anchor = anchor_gradient[evaluated]
    addback = torch.cat(
        [
            action_gradient[evaluated] * d_alpha[evaluated],
            action_gradient[evaluated] * d_beta[evaluated],
        ],
        dim=-1,
    )
    corrected = (
        anchor
        - score[evaluated] * behavior_control[evaluated].unsqueeze(-1)
        + addback
    )
    anchor_var = anchor.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    corrected_var = corrected.var(dim=0, unbiased=False).sum()
    variance_ratio = float(corrected_var / anchor_var)
    return (
        behavior_control,
        action_gradient,
        variance_ratio,
        min(retained_ranks),
    )


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
    assert device.type == "cuda"

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.reward_norm)
            for i in range(args.num_envs)
        ]
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    actor = Actor(obs_dim, act_dim, args).to(device)
    ssl = LeJepaSSL(obs_dim, act_dim, args).to(device)
    v_net = LongHorizonValue(obs_dim, args.v_hidden).to(device)

    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor.parameters(), lr=actor_base_lr, eps=1e-5)
    v_opt = optim.Adam(v_net.parameters(), lr=args.v_lr, eps=1e-5)
    ssl_opt = optim.AdamW(ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay)

    advantage_scale = torch.tensor(1.0, device=device)
    model_updates = 0

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    next_obses = torch.zeros_like(obs)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    actions = torch.zeros_like(zs)
    alphas = torch.zeros_like(zs)
    betas = torch.zeros_like(zs)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    print(
        "[lejepa_geocv_pg_v2] longest-horizon TD anchor + lagged local/successor "
        f"LeJEPA control variate; rank={args.cv_rank}"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            v_opt.param_groups[0]["lr"] = frac * args.v_lr
            ssl_opt.param_groups[0]["lr"] = frac * args.ssl_lr

        # The first rollout initializes an action-independent return scale and trains
        # the lagged baseline/model. Using its own scale would make the reward anchor
        # data-adaptive; policy optimization starts from iteration two.
        actor_active = iteration > 1

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                z, action, logprob, _, alpha, beta = actor(next_obs)
            zs[step] = z
            actions[step] = action
            alphas[step] = alpha
            betas[step] = beta
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            boundary = np.logical_or(terminations, truncations)
            transition_next = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_mask is None:
                    final_mask = [
                        item is not None for item in final_obs
                    ]
                for env_index, has_final in enumerate(final_mask):
                    if has_final and final_obs[env_index] is not None:
                        transition_next[env_index] = final_obs[env_index]

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            boundaries[step] = torch.as_tensor(boundary, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(
                transition_next, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ---- Longest-available TD anchor and lagged successor target ---------------
        with torch.no_grad():
            last_value = v_net(next_obs)
            td_returns = longest_horizon_td_returns(
                rewards, boundaries, last_value, args.gamma
            )
            flat_obs = obs.reshape(-1, obs_dim)
            flat_next_obs = next_obses.reshape(-1, obs_dim)
            flat_actions = actions.reshape(-1, act_dim)
            flat_z = zs.reshape(-1, act_dim)
            flat_alpha = alphas.reshape(-1, act_dim)
            flat_beta = betas.reshape(-1, act_dim)
            flat_boundaries = boundaries.reshape(-1)
            baseline = v_net(flat_obs).reshape(args.num_steps, args.num_envs)
            raw_advantage = (td_returns - baseline).reshape(-1)
            observed_scale = raw_advantage.std().clamp_min(1.0)
            if iteration == 1:
                advantage_scale = observed_scale
            scale_used = advantage_scale.detach().clone()
            advantage_scale = 0.95 * advantage_scale + 0.05 * observed_scale
            old_mean, _, _ = beta_action_mean_and_derivatives(
                flat_alpha, flat_beta
            )
            next_actions = torch.empty_like(actions)
            next_actions[:-1] = actions[1:]
            _, final_action, _, _, _, _ = actor(next_obs)
            next_actions[-1] = final_action
            flat_next_actions = next_actions.reshape(-1, act_dim)
            next_embedding = ssl.encode(flat_next_obs)
            successor_bootstrap = ssl.predict_successor(
                next_embedding, flat_next_actions
            )
            successor_target = successor_td_target(
                next_embedding,
                successor_bootstrap,
                flat_boundaries,
                args.gamma,
            )

        # ---- Lagged geometric LeJEPA control variate -------------------------------
        behavior_control = raw_advantage.new_zeros(args.batch_size)
        action_gradient = flat_actions.new_zeros(args.batch_size, act_dim)
        variance_ratio = 1.0
        cv_rank = 0
        action_jacobian_rms = 0.0
        if (
            model_updates >= args.ssl_warmup_iters
            and args.batch_size >= 4 * args.cv_rank
        ):
            feature_dim = 2 * args.emb_dim
            raw_jacobian = flat_actions.new_empty(
                args.batch_size, feature_dim, act_dim
            )
            with torch.no_grad():
                embedding = ssl.encode(flat_obs)
                for start in range(0, args.batch_size, args.cv_feature_chunk):
                    end = min(start + args.cv_feature_chunk, args.batch_size)
                    emb_chunk = embedding[start:end]
                    mean_chunk = old_mean[start:end]
                    for action_dim in range(act_dim):
                        plus = mean_chunk.clone()
                        minus = mean_chunk.clone()
                        plus[:, action_dim] = (
                            plus[:, action_dim] + args.cv_fd_eps
                        ).clamp(-1.0, 1.0)
                        minus[:, action_dim] = (
                            minus[:, action_dim] - args.cv_fd_eps
                        ).clamp(-1.0, 1.0)
                        denominator = (
                            plus[:, action_dim] - minus[:, action_dim]
                        ).clamp_min(1e-6)
                        pred_plus = ssl.predict_features(
                            emb_chunk, plus
                        )
                        pred_minus = ssl.predict_features(
                            emb_chunk, minus
                        )
                        raw_jacobian[start:end, :, action_dim] = (
                            pred_plus - pred_minus
                        ) / denominator.unsqueeze(-1)
                raw_features = torch.einsum(
                    "nfa,na->nf",
                    raw_jacobian,
                    flat_actions - old_mean,
                )
                env_index = torch.arange(
                    args.batch_size, device=device
                ) % args.num_envs
                (
                    behavior_control,
                    action_gradient,
                    variance_ratio,
                    cv_rank,
                ) = fit_crossfit_control_variate(
                    raw_features,
                    raw_jacobian,
                    raw_advantage,
                    flat_z,
                    flat_alpha,
                    flat_beta,
                    torch.ones(
                        args.batch_size,
                        dtype=torch.bool,
                        device=device,
                    ),
                    env_index,
                    args.cv_rank,
                    args.cv_ridge,
                    args.cv_min_variance_gain,
                )
                action_jacobian_rms = float(raw_jacobian.square().mean().sqrt())
            del raw_jacobian, raw_features

        # ---- Full-batch KL-trusted, bias-corrected policy update -------------------
        active = torch.arange(args.batch_size, device=device)
        actor_steps = 0
        actor_rollbacks = 0
        final_kl = 0.0
        final_anchor_gain = 0.0
        surrogate_value = 0.0
        entropy_value = 0.0
        if actor_active and active.numel() > 0:
            actor_obs = flat_obs[active]
            actor_z = flat_z[active]
            old_logprob = logprobs.reshape(-1)[active]
            old_alpha = flat_alpha[active]
            old_beta = flat_beta[active]
            old_action_mean = old_mean[active]
            scaled_residual = (
                raw_advantage[active] - behavior_control[active]
            ) / scale_used
            scaled_anchor = raw_advantage[active] / scale_used
            scaled_action_gradient = action_gradient[active] / scale_used
            initial_anchor = float(scaled_anchor.mean())
            actor_batch_size = actor_obs.shape[0]

            stop_actor = False
            for _ in range(args.actor_epochs):
                accepted = False
                for _ in range(args.actor_backtracks + 1):
                    actor_state = copy.deepcopy(actor.state_dict())
                    optimizer_state = copy.deepcopy(actor_opt.state_dict())
                    trial_lr = actor_opt.param_groups[0]["lr"]
                    actor_opt.zero_grad(set_to_none=True)
                    surrogate_sum = 0.0
                    proposal_finite = True
                    for start in range(
                        0, actor_batch_size, args.minibatch_size
                    ):
                        end = min(
                            start + args.minibatch_size,
                            actor_batch_size,
                        )
                        (
                            _,
                            _,
                            new_logprob,
                            entropy,
                            new_alpha,
                            new_beta,
                        ) = actor(actor_obs[start:end], actor_z[start:end])
                        log_ratio = (
                            new_logprob - old_logprob[start:end]
                        )
                        ratio = log_ratio.exp()
                        new_action_mean, _, _ = (
                            beta_action_mean_and_derivatives(
                                new_alpha, new_beta
                            )
                        )
                        behavior_sum = (
                            ratio * scaled_residual[start:end]
                        ).sum()
                        addback_sum = (
                            scaled_action_gradient[start:end]
                            * (
                                new_action_mean
                                - old_action_mean[start:end]
                            )
                        ).sum()
                        chunk_surrogate = behavior_sum + addback_sum
                        chunk_loss = (
                            -chunk_surrogate
                            - args.ent_coef * entropy.sum()
                        ) / actor_batch_size
                        if not bool(
                            torch.isfinite(chunk_loss)
                            & torch.isfinite(log_ratio).all()
                            & torch.isfinite(ratio).all()
                        ):
                            proposal_finite = False
                            break
                        chunk_loss.backward()
                        surrogate_sum += float(
                            chunk_surrogate.detach()
                        )
                    if proposal_finite:
                        grad_norm = nn.utils.clip_grad_norm_(
                            actor.parameters(), args.max_grad_norm
                        )
                        proposal_finite = bool(
                            torch.isfinite(grad_norm)
                        )
                    if proposal_finite:
                        actor_opt.step()
                        proposal_finite = all(
                            bool(torch.isfinite(parameter).all())
                            for parameter in actor.parameters()
                        )

                    checked_kl = float("nan")
                    anchor_gain = float("nan")
                    checked_entropy_sum = 0.0
                    if proposal_finite:
                        with torch.no_grad():
                            checked_kl_sum = 0.0
                            checked_anchor_sum = 0.0
                            for start in range(
                                0,
                                actor_batch_size,
                                args.minibatch_size,
                            ):
                                end = min(
                                    start + args.minibatch_size,
                                    actor_batch_size,
                                )
                                (
                                    _,
                                    _,
                                    checked_logprob,
                                    checked_entropy,
                                    checked_alpha,
                                    checked_beta,
                                ) = actor(
                                    actor_obs[start:end],
                                    actor_z[start:end],
                                )
                                checked_ratio = (
                                    checked_logprob
                                    - old_logprob[start:end]
                                ).exp()
                                checked_kl_sum += float(
                                    kl_divergence(
                                        Beta(
                                            old_alpha[start:end],
                                            old_beta[start:end],
                                        ),
                                        Beta(
                                            checked_alpha,
                                            checked_beta,
                                        ),
                                    )
                                    .sum(-1)
                                    .sum()
                                )
                                checked_anchor_sum += float(
                                    (
                                        checked_ratio
                                        * scaled_anchor[start:end]
                                    ).sum()
                                )
                                checked_entropy_sum += float(
                                    checked_entropy.sum()
                                )
                            checked_kl = (
                                checked_kl_sum / actor_batch_size
                            )
                            anchor_gain = (
                                checked_anchor_sum / actor_batch_size
                                - initial_anchor
                            )
                    accepted = (
                        proposal_finite
                        and np.isfinite(checked_kl)
                        and np.isfinite(anchor_gain)
                        and checked_kl <= args.kl_target
                        and anchor_gain >= -1e-4
                    )
                    if accepted:
                        break
                    actor.load_state_dict(actor_state)
                    actor_opt.load_state_dict(optimizer_state)
                    actor_opt.param_groups[0]["lr"] = (
                        trial_lr * args.actor_backtrack_factor
                    )
                    actor_rollbacks += 1
                if not accepted:
                    stop_actor = True
                    break
                actor_steps += 1
                final_kl = checked_kl
                final_anchor_gain = anchor_gain
                surrogate_value = (
                    surrogate_sum / actor_batch_size
                )
                entropy_value = (
                    checked_entropy_sum / actor_batch_size
                )
                if final_kl >= args.actor_kl_fill * args.kl_target:
                    break
            if stop_actor:
                actor_opt.zero_grad(set_to_none=True)

        # ---- Lagged state baseline: update only after actor use ---------------------
        baseline_loss = torch.zeros((), device=device)
        if active.numel() > 0:
            baseline_obs = flat_obs[active]
            baseline_target = td_returns.reshape(-1)[active].detach()
            baseline_indices = np.arange(active.numel())
            for _ in range(args.v_epochs):
                np.random.shuffle(baseline_indices)
                for start in range(
                    0, active.numel(), args.minibatch_size
                ):
                    mb = baseline_indices[
                        start : start + args.minibatch_size
                    ]
                    normalized_error = (
                        v_net(baseline_obs[mb]) - baseline_target[mb]
                    ) / scale_used
                    baseline_loss = (
                        0.5
                        * normalized_error.square().mean()
                        * scale_used
                    )
                    v_opt.zero_grad(set_to_none=True)
                    baseline_loss.backward()
                    nn.utils.clip_grad_norm_(
                        v_net.parameters(), args.max_grad_norm
                    )
                    v_opt.step()
            with torch.no_grad():
                baseline_prediction = v_net(baseline_obs)
                target_variance = baseline_target.var().clamp_min(1e-12)
                baseline_ev = float(
                    1.0
                    - (baseline_target - baseline_prediction).var()
                    / target_variance
                )
        else:
            baseline_ev = float("nan")

        # ---- Train LeJEPA after use; TD successor targets remain rollout-lagged ----
        current_obs = flat_obs
        current_actions = flat_actions
        target_obs = flat_next_obs
        ssl_next = ssl_successor = ssl_sig = ssl_grad = 0.0
        candidate_count = args.batch_size
        for _ in range(args.ssl_steps):
            indices = torch.randint(
                candidate_count,
                (min(args.ssl_batch, candidate_count),),
                device=device,
            )
            (
                ssl_loss,
                next_loss,
                successor_loss,
                sigreg_loss,
            ) = ssl(
                current_obs[indices],
                current_actions[indices],
                target_obs[indices],
                successor_target[indices],
                args.sigreg_weight,
                args.ssl_successor_weight,
            )
            ssl_opt.zero_grad(set_to_none=True)
            ssl_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                ssl.parameters(), args.ssl_grad_clip
            )
            ssl_opt.step()
            ssl_next += float(next_loss.detach())
            ssl_successor += float(successor_loss.detach())
            ssl_sig += float(sigreg_loss.detach())
            ssl_grad += float(grad_norm)
        model_updates += 1
        ssl_next /= args.ssl_steps
        ssl_successor /= args.ssl_steps
        ssl_sig /= args.ssl_steps
        ssl_grad /= args.ssl_steps

        writer.add_scalar(
            "charts/learning_rate",
            actor_opt.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/policy_surrogate", surrogate_value, global_step)
        writer.add_scalar("losses/baseline_loss", float(baseline_loss.detach()), global_step)
        writer.add_scalar("losses/baseline_ev_long_td", baseline_ev, global_step)
        writer.add_scalar("losses/final_analytic_kl", final_kl, global_step)
        writer.add_scalar("losses/reward_anchor_gain", final_anchor_gain, global_step)
        writer.add_scalar("losses/actor_steps", actor_steps, global_step)
        writer.add_scalar(
            "losses/actor_rollbacks", actor_rollbacks, global_step
        )
        writer.add_scalar("losses/entropy", entropy_value, global_step)
        writer.add_scalar("td/advantage_scale", scale_used.item(), global_step)
        writer.add_scalar("cv/variance_ratio", variance_ratio, global_step)
        writer.add_scalar("cv/active_rank", cv_rank, global_step)
        writer.add_scalar(
            "cv/behavior_std",
            behavior_control.std().item(),
            global_step,
        )
        writer.add_scalar("cv/action_jacobian_rms", action_jacobian_rms, global_step)
        writer.add_scalar("ssl/next_loss", ssl_next, global_step)
        writer.add_scalar(
            "ssl/successor_td_loss", ssl_successor, global_step
        )
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_grad, global_step)

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {"actor": actor.state_dict(), "ssl": ssl.state_dict(), "v_net": v_net.state_dict()},
            path,
        )
        print(f"model saved to {path}")

    envs.close()
    writer.close()
