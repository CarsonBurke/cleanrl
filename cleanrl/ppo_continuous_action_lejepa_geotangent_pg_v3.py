# LeJEPA geometric tangent-advantage PPO v3.
#
# No GAE, Q learning, or PopArt. A portfolio of fixed n-step TD advantages at geometric
# horizons keeps the standard clipped-PPO reward path competitive while avoiding a lambda
# trace. Direct action-conditioned LeJEPA heads predict future embeddings at those horizons.
# A lagged state-only latent value probe turns each predictor's local action tangent into a
# state- and actuator-specific embedding advantage. Its PPO gradient is conflict-projected
# against, and norm-bounded by, the reward gradient so learned geometry can add credit
# without reversing reward improvement. TimeLimit truncations bootstrap final observations.
#
# Intermediate horizons train and measure direct predictive skill; the farthest horizon
# with reproducible reward gradient owns half the portfolio. Model/probe train after use.
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
TD_HORIZONS = (1, 2, 4, 8, 16, 32, 64)


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
    update_epochs: int = 10
    v_epochs: int = 4
    reward_norm: bool = False

    ent_coef: float = 0.0
    clip_coef: float = 0.2
    clip_coef_high: float = 0.28
    embedding_clip_coef: float = 0.06
    target_kl: float = 0.03
    embedding_grad_fraction: float = 0.2
    horizon_agreement_floor: float = 0.05
    embedding_skill_floor: float = 0.05
    probe_ev_floor: float = 0.10
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
    probe_lr: float = 3e-4
    probe_steps: int = 32
    probe_batch: int = 1024
    tangent_chunk: int = 4096

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
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    layer_init(nn.Linear(2 * args.emb_dim, args.ssl_hidden)),
                    nn.GELU(),
                    layer_init(nn.Linear(args.ssl_hidden, args.ssl_hidden)),
                    nn.GELU(),
                    layer_init(
                        nn.Linear(args.ssl_hidden, args.emb_dim),
                        std=0.01,
                    ),
                )
                for _ in TD_HORIZONS
            ]
        )
        self.sigreg = SIGReg(num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk)

    def encode(self, obs):
        return self.encoder(obs)

    def _model_input(self, e, a):
        action_emb = self.action_encoder(a)
        return torch.cat([e, action_emb], dim=-1)

    def predict_all(self, e, a):
        model_input = self._model_input(e, a)
        return torch.stack(
            [e + predictor(model_input) for predictor in self.predictors],
            dim=1,
        )

    def forward(
        self,
        obs,
        actions,
        target_obs,
        masks,
        sigreg_weight,
    ):
        emb = self.encoder(obs)
        target_emb = self.encoder(target_obs.flatten(0, 1)).view(
            obs.shape[0], len(TD_HORIZONS), -1
        )
        predictions = self.predict_all(emb, actions)
        squared_error = (predictions - target_emb).square().mean(-1)
        persistence_error = (
            emb.unsqueeze(1) - target_emb
        ).square().mean(-1).detach()
        normalized_error = squared_error / persistence_error.clamp_min(1e-3)
        per_horizon = (
            normalized_error * masks
        ).sum(0) / masks.sum(0).clamp_min(1.0)
        prediction_loss = per_horizon.mean()
        sig_embeddings = torch.cat(
            [emb, target_emb.flatten(0, 1)], dim=0
        )
        sigreg_loss = self.sigreg(sig_embeddings.unsqueeze(0))
        return (
            prediction_loss + sigreg_weight * sigreg_loss,
            prediction_loss,
            sigreg_loss,
            per_horizon,
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


class LatentValueProbe(nn.Module):
    """State-only desirability on LeJEPA embeddings; never action-conditioned."""

    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(emb_dim, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, 1), std=0.01),
        )

    def forward(self, embedding):
        return self.net(embedding).squeeze(-1)


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
        per_logp = dist.log_prob(z)
        logp = per_logp.sum(1)
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, per_logp, entropy, alpha, beta


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


def beta_action_mean(alpha, beta):
    return 2.0 * alpha / (alpha + beta) - 1.0


def fixed_horizon_td_returns(
    rewards,
    boundaries,
    terminations,
    next_values,
    gamma,
    horizons=TD_HORIZONS,
):
    """Fixed n-step targets with correct terminal/truncation and rollout-edge bootstrap."""
    requested = set(horizons)
    outputs = []
    previous = None
    for depth in range(1, max(horizons) + 1):
        if previous is None:
            continuation_value = next_values
        else:
            continuation_value = next_values.clone()
            continuation_value[:-1] = torch.where(
                boundaries[:-1].bool(),
                next_values[:-1],
                previous[1:],
            )
        current = rewards + gamma * (1.0 - terminations) * continuation_value
        previous = current
        if depth in requested:
            outputs.append(current)
    return torch.stack(outputs, dim=0)


def horizon_portfolio(
    advantages,
    z,
    alpha,
    beta,
    minimum_agreement=0.05,
):
    """Weight fixed horizons only when two environment splits agree."""
    horizon_count, timesteps, envs = advantages.shape
    normalized = (
        advantages - advantages.mean(dim=(1, 2), keepdim=True)
    ) / advantages.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    score = beta_parameter_score(z, alpha, beta).view(
        timesteps, envs, -1
    )
    env_gradients = torch.einsum(
        "hte,teq->heq", normalized, score
    ) / timesteps
    def cosine(left, right):
        return (left * right).sum(-1) / (
            left.norm(dim=-1) * right.norm(dim=-1)
        ).clamp_min(1e-8)

    even_odd = cosine(
        env_gradients[:, 0::2].mean(1),
        env_gradients[:, 1::2].mean(1),
    )
    midpoint = envs // 2
    first_second = cosine(
        env_gradients[:, :midpoint].mean(1),
        env_gradients[:, midpoint:].mean(1),
    )
    agreement = torch.minimum(even_odd, first_second)
    reliability = (agreement - minimum_agreement).clamp_min(0.0)
    if float(reliability.sum()) > 0.0:
        weights = 0.5 * reliability / reliability.sum()
        farthest_reliable = int(
            (reliability > 0).nonzero(as_tuple=True)[0][-1]
        )
        weights[farthest_reliable] += 0.5
    else:
        weights = reliability
        weights[TD_HORIZONS.index(16)] = 1.0
    portfolio = torch.einsum("h,hte->te", weights, normalized)
    return portfolio, weights, agreement


def compose_projected_gradients(reward_grads, embedding_grads, fraction):
    """Project only conflicting embedding gradient and bound its global norm."""
    dot = sum(
        (reward * embedding).sum()
        for reward, embedding in zip(reward_grads, embedding_grads)
    )
    reward_sq = sum(gradient.square().sum() for gradient in reward_grads)
    embedding_sq = sum(
        gradient.square().sum() for gradient in embedding_grads
    )
    if float(embedding_sq) <= 1e-20 or fraction <= 0.0:
        return list(reward_grads), dot.new_zeros(()), dot.new_zeros(())
    raw_cosine = dot / (
        reward_sq.sqrt() * embedding_sq.sqrt()
    ).clamp_min(1e-12)
    coefficient = torch.minimum(dot, dot.new_zeros(())) / reward_sq.clamp_min(
        1e-12
    )
    projected = [
        embedding - coefficient * reward
        for reward, embedding in zip(reward_grads, embedding_grads)
    ]
    projected_sq = sum(gradient.square().sum() for gradient in projected)
    scale = torch.minimum(
        dot.new_ones(()),
        fraction
        * reward_sq.sqrt()
        / projected_sq.sqrt().clamp_min(1e-12),
    )
    combined = [
        reward + scale * embedding
        for reward, embedding in zip(reward_grads, projected)
    ]
    return combined, raw_cosine, scale


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
    latent_probe = LatentValueProbe(args.emb_dim, args.v_hidden).to(device)

    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor.parameters(), lr=actor_base_lr, eps=1e-5)
    v_opt = optim.Adam(v_net.parameters(), lr=args.v_lr, eps=1e-5)
    ssl_opt = optim.AdamW(ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay)
    probe_opt = optim.Adam(latent_probe.parameters(), lr=args.probe_lr, eps=1e-5)

    model_updates = 0

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    next_obses = torch.zeros_like(obs)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    actions = torch.zeros_like(zs)
    alphas = torch.zeros_like(zs)
    betas = torch.zeros_like(zs)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    per_logprobs = torch.zeros_like(actions)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations_buf = torch.zeros_like(boundaries)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    print(
        "[lejepa_geotangent_pg_v3] fixed geometric TD portfolio + "
        "state-conditioned per-actuator LeJEPA tangent PPO"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            v_opt.param_groups[0]["lr"] = frac * args.v_lr
            ssl_opt.param_groups[0]["lr"] = frac * args.ssl_lr
            probe_opt.param_groups[0]["lr"] = frac * args.probe_lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                (
                    z,
                    action,
                    logprob,
                    per_logprob,
                    _,
                    alpha,
                    beta,
                ) = actor(next_obs)
            zs[step] = z
            actions[step] = action
            alphas[step] = alpha
            betas[step] = beta
            logprobs[step] = logprob
            per_logprobs[step] = per_logprob

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
            terminations_buf[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
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

        # ---- Fixed geometric TD portfolio -----------------------------------------
        with torch.no_grad():
            flat_obs = obs.reshape(-1, obs_dim)
            flat_next_obs = next_obses.reshape(-1, obs_dim)
            flat_actions = actions.reshape(-1, act_dim)
            flat_z = zs.reshape(-1, act_dim)
            flat_alpha = alphas.reshape(-1, act_dim)
            flat_beta = betas.reshape(-1, act_dim)
            next_values = v_net(flat_next_obs).reshape(
                args.num_steps, args.num_envs
            )
            baseline = v_net(flat_obs).reshape(args.num_steps, args.num_envs)
            td_targets = fixed_horizon_td_returns(
                rewards,
                boundaries,
                terminations_buf,
                next_values,
                args.gamma,
            )
            horizon_advantages = td_targets - baseline.unsqueeze(0)
            policy_advantage, td_weights, td_agreement = horizon_portfolio(
                horizon_advantages,
                zs,
                alphas,
                betas,
                args.horizon_agreement_floor,
            )
            policy_advantage = (
                policy_advantage - policy_advantage.mean()
            ) / policy_advantage.std().clamp_min(1e-6)
            horizon_scales = horizon_advantages.std(
                dim=(1, 2)
            ).clamp_min(1.0)
            old_mean = beta_action_mean(flat_alpha, flat_beta)

        # ---- Direct geometric LeJEPA skill and tangent advantages ------------------
        max_horizon = max(TD_HORIZONS)
        base_steps = args.num_steps - max_horizon
        model_obs = obs[:base_steps].reshape(-1, obs_dim)
        model_actions = actions[:base_steps].reshape(-1, act_dim)
        model_targets = torch.stack(
            [obs[h : h + base_steps] for h in TD_HORIZONS],
            dim=2,
        ).reshape(-1, len(TD_HORIZONS), obs_dim)
        model_masks = []
        continuation = 1.0 - boundaries
        for horizon in TD_HORIZONS:
            alive = torch.ones(
                base_steps, args.num_envs, device=device
            )
            for offset in range(horizon):
                alive = alive * continuation[
                    offset : offset + base_steps
                ]
            model_masks.append(alive)
        model_masks = torch.stack(model_masks, dim=-1).reshape(
            -1, len(TD_HORIZONS)
        )
        with torch.no_grad():
            model_embedding = ssl.encode(model_obs)
            target_embedding = ssl.encode(
                model_targets.flatten(0, 1)
            ).view(model_obs.shape[0], len(TD_HORIZONS), -1)
            predictions = ssl.predict_all(model_embedding, model_actions)
            permuted_predictions = ssl.predict_all(
                model_embedding, model_actions.roll(1, dims=0)
            )
            prediction_mse = (
                predictions - target_embedding
            ).square().mean(-1)
            permuted_mse = (
                permuted_predictions - target_embedding
            ).square().mean(-1)
            persistence_mse = (
                model_embedding.unsqueeze(1) - target_embedding
            ).square().mean(-1)
            skill = 1.0 - (
                (prediction_mse * model_masks).sum(0)
                / (persistence_mse * model_masks)
                .sum(0)
                .clamp_min(1e-6)
            )
            action_skill = (
                ((permuted_mse - prediction_mse) * model_masks).sum(0)
                / (persistence_mse * model_masks)
                .sum(0)
                .clamp_min(1e-6)
            )
            predicted_values = latent_probe(
                predictions.flatten(0, 1)
            ).view(model_obs.shape[0], len(TD_HORIZONS))
            permuted_values = latent_probe(
                permuted_predictions.flatten(0, 1)
            ).view_as(predicted_values)
            target_values = latent_probe(
                target_embedding.flatten(0, 1)
            ).view_as(predicted_values)
            persistence_values = latent_probe(
                model_embedding
            ).unsqueeze(1)
            predicted_value_error = (
                predicted_values - target_values
            ).square()
            permuted_value_error = (
                permuted_values - target_values
            ).square()
            persistence_value_error = (
                persistence_values - target_values
            ).square()
            value_error_denominator = (
                persistence_value_error * model_masks
            ).sum(0).clamp_min(1e-6)
            value_skill = 1.0 - (
                (predicted_value_error * model_masks).sum(0)
                / value_error_denominator
            )
            action_value_skill = (
                (
                    (permuted_value_error - predicted_value_error)
                    * model_masks
                ).sum(0)
                / value_error_denominator
            )
            model_value_gap = (
                (predicted_values - target_values).abs() * model_masks
            ).sum(0) / model_masks.sum(0).clamp_min(1.0)
            full_embedding = ssl.encode(flat_obs)
            probe_pre_prediction = latent_probe(full_embedding)
            probe_pre_ev = float(
                1.0
                - (
                    td_targets[-1].reshape(-1)
                    - probe_pre_prediction
                ).var()
                / td_targets[-1].var().clamp_min(1e-12)
            )

        embedding_advantage = flat_actions.new_zeros(
            args.batch_size, act_dim
        )
        tangent_rms = 0.0
        embedding_confidence = 0.0
        embedding_horizon_weights = skill.new_zeros(len(TD_HORIZONS))
        probe_reliability = max(
            0.0,
            min(
                1.0,
                (probe_pre_ev - args.probe_ev_floor)
                / max(1.0 - args.probe_ev_floor, EPS),
            ),
        )
        joint_skill = torch.minimum(
            torch.minimum(skill, action_skill),
            torch.minimum(value_skill, action_value_skill),
        )
        reliable_skill = (
            joint_skill - args.embedding_skill_floor
        ).clamp_min(0.0)
        if (
            model_updates >= args.ssl_warmup_iters
            and float(reliable_skill.max()) > 0.0
            and probe_reliability > 0.0
        ):
            embedding_horizon_weights = (
                reliable_skill / reliable_skill.sum().clamp_min(1e-6)
            )
            farthest_reliable = int(
                (reliable_skill > 0).nonzero(as_tuple=True)[0][-1]
            )
            embedding_horizon_weights *= 0.5
            embedding_horizon_weights[farthest_reliable] += 0.5
            embedding_confidence = (
                float(
                    (
                        embedding_horizon_weights
                        * joint_skill.clamp(0.0, 1.0)
                    ).sum()
                )
                * probe_reliability
            )
            tangent = flat_actions.new_empty(
                args.batch_size, len(TD_HORIZONS), act_dim
            )
            embedding = ssl.encode(flat_obs).detach()
            for start in range(0, args.batch_size, args.tangent_chunk):
                end = min(start + args.tangent_chunk, args.batch_size)
                action_at_mean = (
                    old_mean[start:end].detach().clone().requires_grad_(True)
                )
                predicted = ssl.predict_all(
                    embedding[start:end], action_at_mean
                )
                for horizon_index in range(len(TD_HORIZONS)):
                    desirability = latent_probe(
                        predicted[:, horizon_index]
                    ).sum()
                    tangent[start:end, horizon_index] = torch.autograd.grad(
                        desirability,
                        action_at_mean,
                        retain_graph=horizon_index + 1 < len(TD_HORIZONS),
                    )[0].detach()
            with torch.no_grad():
                discount = flat_actions.new_tensor(
                    [args.gamma**horizon for horizon in TD_HORIZONS]
                )
                per_horizon_credit = (
                    tangent
                    * (flat_actions - old_mean).unsqueeze(1)
                    * discount.view(1, -1, 1)
                )
                embedding_advantage = torch.einsum(
                    "h,nha->na",
                    embedding_horizon_weights,
                    per_horizon_credit,
                )
                embedding_advantage = (
                    embedding_confidence
                    * embedding_advantage
                    / horizon_scales[-1]
                )
                tangent_rms = float(tangent.square().mean().sqrt())
            del tangent

        # ---- Clipped minibatch PPO + conflict-projected embedding gradient ---------
        actor_steps = 0
        approx_kl_value = 0.0
        reward_loss_value = 0.0
        embedding_loss_value = 0.0
        entropy_value = 0.0
        gradient_cosine_value = 0.0
        embedding_scale_value = 0.0
        actor_parameters = tuple(actor.parameters())
        old_joint_logprob = logprobs.reshape(-1)
        old_factor_logprob = per_logprobs.reshape(-1, act_dim)
        b_reward_advantage = policy_advantage.reshape(-1)
        embedding_active = embedding_confidence > 0.0
        actor_state = copy.deepcopy(actor.state_dict())
        actor_optimizer_state = copy.deepcopy(actor_opt.state_dict())
        indices = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            epoch_kls = []
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = indices[start : start + args.minibatch_size]
                (
                    _,
                    _,
                    new_logprob,
                    new_factor_logprob,
                    entropy,
                    _,
                    _,
                ) = actor(flat_obs[mb], flat_z[mb])
                log_ratio = new_logprob - old_joint_logprob[mb]
                ratio = log_ratio.exp()
                reward_advantage = b_reward_advantage[mb]
                reward_unclipped = -reward_advantage * ratio
                reward_clipped = -reward_advantage * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef_high,
                )
                reward_loss = torch.maximum(
                    reward_unclipped, reward_clipped
                ).mean() - args.ent_coef * entropy.mean()
                reward_grads = torch.autograd.grad(
                    reward_loss,
                    actor_parameters,
                    retain_graph=embedding_active,
                )

                if embedding_active:
                    factor_ratio = (
                        new_factor_logprob - old_factor_logprob[mb]
                    ).exp()
                    credit = embedding_advantage[mb]
                    embedding_unclipped = -credit * factor_ratio
                    embedding_clipped = -credit * torch.clamp(
                        factor_ratio,
                        1.0 - args.embedding_clip_coef,
                        1.0 + args.embedding_clip_coef,
                    )
                    embedding_loss = torch.maximum(
                        embedding_unclipped, embedding_clipped
                    ).mean()
                    embedding_grads = torch.autograd.grad(
                        embedding_loss, actor_parameters
                    )
                    combined, gradient_cosine, embedding_scale = (
                        compose_projected_gradients(
                            reward_grads,
                            embedding_grads,
                            args.embedding_grad_fraction
                            * embedding_confidence,
                        )
                    )
                    gradient_cosine_value = float(gradient_cosine)
                    embedding_scale_value = float(embedding_scale)
                    embedding_loss_value = float(embedding_loss)
                else:
                    combined = reward_grads
                actor_opt.zero_grad(set_to_none=True)
                for parameter, gradient in zip(
                    actor_parameters, combined
                ):
                    parameter.grad = gradient.detach()
                nn.utils.clip_grad_norm_(
                    actor_parameters, args.max_grad_norm
                )
                actor_opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                epoch_kls.append(float(approx_kl))
                actor_steps += 1
                approx_kl_value = float(approx_kl)
                reward_loss_value = float(reward_loss)
                entropy_value = float(entropy.mean())
            if np.mean(epoch_kls) > args.target_kl:
                break
        with torch.no_grad():
            analytic_kl_sum = 0.0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = min(start + args.minibatch_size, args.batch_size)
                _, _, _, _, _, new_alpha, new_beta = actor(
                    flat_obs[start:end], flat_z[start:end]
                )
                analytic_kl_sum += float(
                    kl_divergence(
                        Beta(
                            flat_alpha[start:end],
                            flat_beta[start:end],
                        ),
                        Beta(new_alpha, new_beta),
                    )
                    .sum(-1)
                    .sum()
                )
            final_analytic_kl = analytic_kl_sum / args.batch_size
        actor_rollback = int(
            not np.isfinite(final_analytic_kl)
            or final_analytic_kl > 1.5 * args.target_kl
        )
        if actor_rollback:
            actor.load_state_dict(actor_state)
            actor_opt.load_state_dict(actor_optimizer_state)
            actor_steps = 0

        # ---- State baseline trained against every fixed-horizon target -------------
        active = torch.arange(args.batch_size, device=device)
        baseline_loss = torch.zeros((), device=device)
        if active.numel() > 0:
            baseline_obs = flat_obs[active]
            baseline_targets = td_targets.reshape(
                len(TD_HORIZONS), -1
            ).detach()
            critic_weights = horizon_scales.new_full(
                (len(TD_HORIZONS),),
                0.5 / (len(TD_HORIZONS) - 1),
            )
            critic_weights[-1] = 0.5
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
                        v_net(baseline_obs[mb]).unsqueeze(0)
                        - baseline_targets[:, mb]
                    ) / horizon_scales.unsqueeze(-1)
                    baseline_loss = (
                        0.5
                        * (
                            critic_weights
                            * normalized_error.square().mean(-1)
                        ).sum()
                        * horizon_scales[-1]
                    )
                    v_opt.zero_grad(set_to_none=True)
                    baseline_loss.backward()
                    nn.utils.clip_grad_norm_(
                        v_net.parameters(), args.max_grad_norm
                    )
                    v_opt.step()
            with torch.no_grad():
                baseline_prediction = v_net(baseline_obs)
                primary_target = baseline_targets[-1]
                target_variance = primary_target.var().clamp_min(1e-12)
                baseline_ev = float(
                    1.0
                    - (primary_target - baseline_prediction).var()
                    / target_variance
                )
        else:
            baseline_ev = float("nan")

        # ---- Train direct geometric LeJEPA after actor use --------------------------
        ssl_prediction = ssl_sig = ssl_grad = 0.0
        per_horizon_loss = torch.zeros(len(TD_HORIZONS), device=device)
        candidate_count = model_obs.shape[0]
        for _ in range(args.ssl_steps):
            indices = torch.randint(
                candidate_count,
                (min(args.ssl_batch, candidate_count),),
                device=device,
            )
            (
                ssl_loss,
                prediction_loss,
                sigreg_loss,
                horizon_loss,
            ) = ssl(
                model_obs[indices],
                model_actions[indices],
                model_targets[indices],
                model_masks[indices],
                args.sigreg_weight,
            )
            ssl_opt.zero_grad(set_to_none=True)
            ssl_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                ssl.parameters(), args.ssl_grad_clip
            )
            ssl_opt.step()
            ssl_prediction += float(prediction_loss.detach())
            ssl_sig += float(sigreg_loss.detach())
            ssl_grad += float(grad_norm)
            per_horizon_loss += horizon_loss.detach()
        model_updates += 1
        ssl_prediction /= args.ssl_steps
        ssl_sig /= args.ssl_steps
        ssl_grad /= args.ssl_steps
        per_horizon_loss /= args.ssl_steps

        # Keep the probe in the encoder's post-update coordinate frame. Both remain
        # lagged relative to the actor because this happens after tangent consumption.
        with torch.no_grad():
            probe_embedding = ssl.encode(flat_obs)
        probe_target = td_targets[-1].reshape(-1).detach()
        probe_loss_value = 0.0
        for _ in range(args.probe_steps):
            probe_indices = torch.randint(
                args.batch_size,
                (min(args.probe_batch, args.batch_size),),
                device=device,
            )
            probe_error = (
                latent_probe(probe_embedding[probe_indices])
                - probe_target[probe_indices]
            ) / horizon_scales[-1]
            probe_loss = (
                0.5 * probe_error.square().mean() * horizon_scales[-1]
            )
            probe_opt.zero_grad(set_to_none=True)
            probe_loss.backward()
            nn.utils.clip_grad_norm_(
                latent_probe.parameters(), args.max_grad_norm
            )
            probe_opt.step()
            probe_loss_value += float(probe_loss.detach())
        probe_loss_value /= args.probe_steps
        with torch.no_grad():
            probe_prediction = latent_probe(probe_embedding)
            probe_ev = float(
                1.0
                - (probe_target - probe_prediction).var()
                / probe_target.var().clamp_min(1e-12)
            )

        writer.add_scalar(
            "charts/learning_rate",
            actor_opt.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/reward_policy", reward_loss_value, global_step)
        writer.add_scalar(
            "losses/embedding_policy", embedding_loss_value, global_step
        )
        writer.add_scalar("losses/baseline_loss", float(baseline_loss.detach()), global_step)
        writer.add_scalar("losses/baseline_ev_h64", baseline_ev, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl_value, global_step)
        writer.add_scalar(
            "losses/final_analytic_kl", final_analytic_kl, global_step
        )
        writer.add_scalar("losses/actor_rollback", actor_rollback, global_step)
        writer.add_scalar("losses/actor_steps", actor_steps, global_step)
        writer.add_scalar("losses/entropy", entropy_value, global_step)
        writer.add_scalar("probe/loss", probe_loss_value, global_step)
        writer.add_scalar("probe/pre_update_ev_h64", probe_pre_ev, global_step)
        writer.add_scalar("probe/ev_h64", probe_ev, global_step)
        writer.add_scalar(
            "embedding/confidence", embedding_confidence, global_step
        )
        writer.add_scalar(
            "embedding/tangent_rms", tangent_rms, global_step
        )
        writer.add_scalar(
            "embedding/reward_gradient_cosine",
            gradient_cosine_value,
            global_step,
        )
        writer.add_scalar(
            "embedding/gradient_scale", embedding_scale_value, global_step
        )
        writer.add_scalar("ssl/pred_loss", ssl_prediction, global_step)
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_grad, global_step)
        for horizon_index, horizon in enumerate(TD_HORIZONS):
            writer.add_scalar(
                f"td/weight_h{horizon}",
                td_weights[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"td/agreement_h{horizon}",
                td_agreement[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"ssl/skill_h{horizon}",
                skill[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"ssl/action_skill_h{horizon}",
                action_skill[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"ssl/value_skill_h{horizon}",
                value_skill[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"ssl/action_value_skill_h{horizon}",
                action_value_skill[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"probe/model_value_gap_h{horizon}",
                model_value_gap[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"ssl/pred_loss_h{horizon}",
                per_horizon_loss[horizon_index].item(),
                global_step,
            )
            writer.add_scalar(
                f"embedding/weight_h{horizon}",
                embedding_horizon_weights[horizon_index].item(),
                global_step,
            )

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "actor": actor.state_dict(),
                "ssl": ssl.state_dict(),
                "v_net": v_net.state_dict(),
                "latent_probe": latent_probe.state_dict(),
            },
            path,
        )
        print(f"model saved to {path}")

    envs.close()
    writer.close()
