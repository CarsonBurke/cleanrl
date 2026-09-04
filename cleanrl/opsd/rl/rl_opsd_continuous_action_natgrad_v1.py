# RL-OPSD v1: advantage-conditioned on-policy self-distillation for continuous control.
# =====================================================================================
# Base: HalfCheetah-v4__iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1__1__1782008060
# (mbpercnorm_v2 with --norm-adv --norm-adv-scope batch --no-ret-percnorm).
#
# The rollout records each state, sampled native Beta action, and old policy logits. After
# GAE, the actor turns each standardized advantage into a detached, action-conditioned
# teacher distribution:
#   1. compute the sampled action's score under the recorded Beta policy;
#   2. take a signed natural-gradient step in the two concentration-head logits;
#   3. cap the local teacher displacement by a per-action-coordinate KL radius;
#   4. fit the current policy to that teacher with forward KL, one pass over the rollout.
# Positive advantages move the teacher toward the sampled action; negative advantages move
# it away. This is full-distribution, per-coordinate self-distillation: no importance ratio,
# PPO clipping, policy epochs, reference model, or sampled-action policy loss.
#
# The v24 unimodal Beta actor, shared IterThink trunk, raw-reward GAE, and Dreamer3-bucket
# MTP critic are retained. Actor and critic gradients remain independently clipped before
# being summed on the shared trunk. A fixed-shape CUDA graph captures the complete forward
# and dual-backward update; the optimizer step remains outside the graph so LR annealing is
# dynamic. One shuffled pass makes the update roughly 10x cheaper than the 10-epoch base.
# =====================================================================================
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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6


def value_support_bounds(args):
    """Return critic support endpoints in the coordinate system used by bins."""
    return args.v_min, args.v_max




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
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32

    # RL-OPSD: one natural-gradient teacher step per rollout sample, then forward-KL
    # distillation. The cap is the local quadratic KL per action coordinate.
    teacher_step_size: float = 1.0
    teacher_kl_cap: float = 0.05
    fisher_damping: float = 1e-4

    vf_coef: float = 0.5
    actor_grad_clip: float = 0.25
    critic_grad_clip: float = 0.25

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
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
        env = gym.wrappers.TransformObservation(  # pyright: ignore[reportCallIssue]
            env, lambda observation: np.asarray(observation).clip(-10, 10)
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(
                env, lambda reward: min(max(float(reward), -10.0), 10.0)
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer(
            "action_low",
            torch.as_tensor(envs.single_action_space.low, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(envs.single_action_space.high, dtype=torch.float32).reshape(-1),
        )

    def _heads(self, x):
        feat = self.trunk(x)
        raw_alpha = self.actor_alpha_head(feat)
        raw_beta = self.actor_beta_head(feat)
        alpha = 1.0 + F.softplus(raw_alpha)
        beta = 1.0 + F.softplus(raw_beta)
        value_logits = self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return raw_alpha, raw_beta, alpha, beta, value_logits

    def get_value(self, x):
        feat = self.trunk(x)
        return self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x):
        raw_alpha, raw_beta, alpha, beta, value_logits = self._heads(x)
        z = Beta(alpha, beta, validate_args=False).sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        return action, z, raw_alpha, raw_beta, value_logits

    def get_policy_and_value(self, x):
        _, _, alpha, beta, value_logits = self._heads(x)
        return alpha, beta, value_logits

    def actor_parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.actor_alpha_head.parameters())
            + list(self.actor_beta_head.parameters())
        )

    def critic_parameters(self):
        return list(self.trunk.parameters()) + list(self.critic_head.parameters())


@torch.no_grad()
def make_advantage_conditioned_teacher(old_raw_alpha, old_raw_beta, z, advantage, args):
    """Take a detached natural policy-gradient step and return its Beta distribution."""
    old_alpha = 1.0 + F.softplus(old_raw_alpha)
    old_beta = 1.0 + F.softplus(old_raw_beta)
    total = old_alpha + old_beta

    score_alpha = z.log() - torch.digamma(old_alpha) + torch.digamma(total)
    score_beta = torch.log1p(-z) - torch.digamma(old_beta) + torch.digamma(total)
    jac_alpha = old_raw_alpha.sigmoid()
    jac_beta = old_raw_beta.sigmoid()
    score_raw_alpha = jac_alpha * score_alpha
    score_raw_beta = jac_beta * score_beta

    trigamma_total = torch.polygamma(1, total)
    fisher_aa = jac_alpha.square() * (torch.polygamma(1, old_alpha) - trigamma_total)
    fisher_bb = jac_beta.square() * (torch.polygamma(1, old_beta) - trigamma_total)
    fisher_ab = -jac_alpha * jac_beta * trigamma_total
    fisher_aa = fisher_aa + args.fisher_damping
    fisher_bb = fisher_bb + args.fisher_damping
    determinant = (fisher_aa * fisher_bb - fisher_ab.square()).clamp_min(1e-12)

    natural_alpha = (fisher_bb * score_raw_alpha - fisher_ab * score_raw_beta) / determinant
    natural_beta = (fisher_aa * score_raw_beta - fisher_ab * score_raw_alpha) / determinant
    signed_step = args.teacher_step_size * advantage.unsqueeze(-1)

    # The quadratic KL is local to one action coordinate, the continuous analogue of
    # OPSD's per-token control. It bounds the teacher target, not a policy ratio.
    natural_quadratic = (
        score_raw_alpha * natural_alpha + score_raw_beta * natural_beta
    ).clamp_min(0.0)
    local_kl = 0.5 * signed_step.square() * natural_quadratic
    radius_scale = torch.sqrt(args.teacher_kl_cap / (local_kl + 1e-8)).clamp(max=1.0)
    signed_step = signed_step * radius_scale

    teacher_alpha = 1.0 + F.softplus(old_raw_alpha + signed_step * natural_alpha)
    teacher_beta = 1.0 + F.softplus(old_raw_beta + signed_step * natural_beta)
    teacher_kl = kl_divergence(
        Beta(teacher_alpha, teacher_beta, validate_args=False),
        Beta(old_alpha, old_beta, validate_args=False),
    )
    capped_fraction = (radius_scale < 1.0).float().mean()
    return teacher_alpha, teacher_beta, teacher_kl, capped_fraction


class CudaGraphUpdater:
    """Fixed-shape full update: forward KL, value CE, and decoupled dual backward."""

    def __init__(self, agent, optimizer, args, obs_shape, act_dim, device):
        self.agent = agent
        self.optimizer = optimizer
        self.args = args
        mb = args.minibatch_size
        self.obs = torch.zeros((mb,) + obs_shape, device=device)
        self.teacher_alpha = torch.full((mb, act_dim), 2.0, device=device)
        self.teacher_beta = torch.full((mb, act_dim), 2.0, device=device)
        self.target_probs = torch.full(
            (mb, args.critic_mtp_horizon, args.num_bins),
            1.0 / args.num_bins,
            device=device,
        )
        self.target_mask = torch.ones((mb, args.critic_mtp_horizon), device=device)

        self.actor_params = agent.actor_parameters()
        self.critic_params = agent.critic_parameters()
        self.critic_grad_buffers = [torch.zeros_like(p) for p in self.critic_params]
        for parameter in agent.parameters():
            parameter.grad = torch.zeros_like(parameter)

        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._backward()
        current_stream.wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = self._backward()
        self.optimizer.zero_grad(set_to_none=False)

    def _backward(self):
        self.optimizer.zero_grad(set_to_none=False)
        student_alpha, student_beta, value_logits = self.agent.get_policy_and_value(self.obs)
        policy_loss = kl_divergence(
            Beta(self.teacher_alpha, self.teacher_beta, validate_args=False),
            Beta(student_alpha, student_beta, validate_args=False),
        ).sum(dim=-1).mean()
        entropy = Beta(
            student_alpha, student_beta, validate_args=False
        ).entropy().sum(dim=-1).mean()

        value_log_probs = torch.log_softmax(value_logits, dim=-1)
        value_ce = -(self.target_probs * value_log_probs).sum(dim=-1)
        value_loss = (value_ce * self.target_mask).sum(dim=-1).mean()

        (self.args.vf_coef * value_loss).backward(retain_graph=True)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic_params, self.args.critic_grad_clip)
        for parameter, buffer in zip(self.critic_params, self.critic_grad_buffers):
            buffer.copy_(parameter.grad)

        self.optimizer.zero_grad(set_to_none=False)
        policy_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_params, self.args.actor_grad_clip)
        for parameter, buffer in zip(self.critic_params, self.critic_grad_buffers):
            parameter.grad.add_(buffer)
        return policy_loss, value_loss, entropy, actor_grad_norm, critic_grad_norm

    def step(self, obs, teacher_alpha, teacher_beta, target_probs, target_mask):
        self.obs.copy_(obs)
        self.teacher_alpha.copy_(teacher_alpha)
        self.teacher_beta.copy_(teacher_beta)
        self.target_probs.copy_(target_probs)
        self.target_mask.copy_(target_mask)
        self.graph.replay()
        self.optimizer.step()
        return self.outputs


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.teacher_step_size > 0.0
    assert args.teacher_kl_cap > 0.0
    assert args.fisher_damping > 0.0

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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RL-OPSD")
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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    support_min, support_max = value_support_bounds(args)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))
    updater = CudaGraphUpdater(agent, optimizer, args, obs_shape, act_dim, device)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    old_raw_alphas = torch.zeros_like(latent_zs)
    old_raw_betas = torch.zeros_like(latent_zs)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, raw_alpha, raw_beta, value_logits = agent.get_action_and_value(next_obs)
                values[step] = hl_support.to_scalar(value_logits[:, 0])
            latent_zs[step] = z
            old_raw_alphas[step] = raw_alpha
            old_raw_betas[step] = raw_beta

            env_action = action.reshape((args.num_envs,) + action_shape)
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [item is not None for item in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_obses[step] = torch.as_tensor(
                transition_next_obs, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value_logits = agent.get_value(next_obses.reshape((-1,) + obs_shape))[:, 0]
            next_values = hl_support.to_scalar(next_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (
                    (1.0 - transition_terminations[t]) * transition_valids[t]
                )
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_values[t] * bootstrap_nonterminal - values[t]
                last_gae = (
                    delta
                    + args.gamma * args.gae_lambda * lambda_nonterminal * last_gae
                )
                advantages[t] = last_gae
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=device
            )
            for horizon in range(mtp):
                valid_len = args.num_steps - horizon
                valid_horizon = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for boundary_offset in range(horizon):
                    valid_horizon &= (
                        transition_boundaries[
                            boundary_offset : boundary_offset + valid_len
                        ]
                        == 0
                    )
                return_mtp[:valid_len, :, horizon] = returns[horizon:]
                return_mtp_mask[:valid_len, :, horizon] = valid_horizon
            target_probs = hl_support.project(return_mtp)

            batch_advantage = advantages.reshape(-1)
            batch_advantage = (
                batch_advantage - batch_advantage.mean()
            ) / (batch_advantage.std() + 1e-8)
            teacher_alpha, teacher_beta, teacher_kl, capped_fraction = (
                make_advantage_conditioned_teacher(
                    old_raw_alphas.reshape(-1, act_dim),
                    old_raw_betas.reshape(-1, act_dim),
                    latent_zs.reshape(-1, act_dim),
                    batch_advantage,
                    args,
                )
            )

        b_obs = obs.reshape((-1,) + obs_shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(
            -1, args.critic_mtp_horizon, args.num_bins
        )
        b_target_mask = return_mtp_mask.reshape(
            -1, args.critic_mtp_horizon
        ).to(dtype=torch.float32)

        permutation = torch.randperm(args.batch_size, device=device)
        metric_totals = torch.zeros(5, device=device)
        for start in range(0, args.batch_size, args.minibatch_size):
            mb_inds = permutation[start : start + args.minibatch_size]
            outputs = updater.step(
                b_obs[mb_inds],
                teacher_alpha[mb_inds],
                teacher_beta[mb_inds],
                b_target_probs[mb_inds],
                b_target_mask[mb_inds],
            )
            for metric_total, output in zip(metric_totals, outputs):
                metric_total.add_(output.detach())
        policy_loss, value_loss, entropy, actor_grad_norm, critic_grad_norm = (
            metric_totals / args.num_minibatches
        ).tolist()

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = (
            np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance
        )
        edge_per_horizon = b_target_probs[..., 0] + b_target_probs[..., -1]
        edge_mass = (
            (edge_per_horizon * b_target_mask).sum()
            / b_target_mask.sum().clamp_min(1.0)
        ).item()
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_forward_kl", policy_loss, global_step)
        writer.add_scalar("losses/value_loss", value_loss, global_step)
        writer.add_scalar("losses/entropy", entropy, global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("losses/actor_grad_norm", actor_grad_norm, global_step)
        writer.add_scalar("losses/critic_grad_norm", critic_grad_norm, global_step)
        writer.add_scalar("opsd/teacher_kl_mean", teacher_kl.mean().item(), global_step)
        writer.add_scalar("opsd/teacher_kl_max", teacher_kl.max().item(), global_step)
        writer.add_scalar("opsd/teacher_cap_fraction", capped_fraction.item(), global_step)
        writer.add_scalar("debug/advantage_mean", advantages.mean().item(), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
