# OPSD-Distill v1 -- PURE self-distillation. One loss on the actor: per-dim KL from an
# ANALYTIC teacher. No clone/BC term, no ratio, no clipping, no advantage-weighted gradient.
# =====================================================================================
# WHY THE EARLIER DISTILL-ONLY ATTEMPT WAS INERT (measured, not assumed)
#
# `opsd_advcond_distillonly_v1.py` removed the clone term from the AdvCond hybrid and kept
# a LEARNED privileged channel pi(.|s, delta). Result: `cond_gap = 0.000`, collapse by
# 352k. The reason is structural, not a tuning failure: nothing in a pure-KL objective
# ever teaches the network what the delta channel MEANS, so the trunk is free to ignore it
# -- and ignoring it is optimal, because then teacher == student and the loss is already 0.
# A learned conditional needs a likelihood term to become identifiable. That is exactly
# what clone was doing, and why removing it produced an inert method.
#
# THE FIX: stop learning the conditional. CONSTRUCT the teacher in closed form.
#
# The teacher is the rollout policy tilted toward the action it actually took, scaled by
# how well that action did:
#     g_t   = tanh(A_hat_t / tilt_temp)                 signed credit, bounded, g(0) = 0
#     m_T   = m_S + tilt_gain * g_t * (z_t - m_S)       move toward z_t if good, away if bad
#     sd_T  = sd_S * (1 - tilt_sharpen * |g_t|)         optional confidence sharpening
#     (alpha_T, beta_T) = moment-match(m_T, sd_T)       so the divergence stays closed-form
# and the ONLY actor loss is  sum_d clip( KL(Beta_T,d || Beta_S,d), tau ).  Teacher detached.
#
# This is the user's framing executed literally: replay each step, look at the advantage
# and the policy that chose it, and take a gradient on per-dim divergence. It is also the
# paper's "full-vocabulary" view rather than its sampled-token view -- we distill a whole
# distribution per dim, which the paper found decisively better than a per-sample score.
#
# WHY IT CANNOT GO INERT: the shift is computed, not learned, so it is nonzero whenever the
# advantage is. There is no channel to ignore.
#
# WHY IT IS BETTER CONDITIONED THAN A POLICY GRADIENT (the real hypothesis):
#   the shift is proportional to the REALIZED deviation (z_t - m_S), which is O(sigma), so
#       KL ~= (1/2) (delta_m / sigma)^2 = (1/2) (tilt_gain * g_t)^2
#   is SIGMA-INDEPENDENT. Constant information per step, with no dual and no schedule.
#   Vanilla PG moves the mean by A/sigma^2, so its KL step grows like (A/sigma)^2 and
#   EXPLODES as entropy collapses. This chassis's champion collapses to H = -13.0 while
#   scoring 8819, so it lives deep in the regime where that distinction matters. v7 failed
#   in this lineage by quoting its dose in delta units, which self-extinguished as delta's
#   spread shrank; here shift and scale shrink together, which is the whole point.
#
# INHERITED, UNCHANGED: the winning chassis (ThinkTrunk Beta policy, HL-Gauss MTP critic,
# decoupled actor/critic epochs, 511 bins) and the one variable that produced the 8819
# result -- `cond_scale=ema_rms` advantage scaling (scale-only, natural zero preserved).
# `cond_lambda=0` (1-step TD residual) is kept: it was measured to be the most
# action-attributable credit signal, an 80x stronger channel than GAE(0.95).
#
# REFERENCE (HalfCheetah-v4, seed 1): PPO 1576/3079/5012/6716/8278 @0.5/1/2/4/8M.
# AdvCond hybrid champion 994/3826/6267/7680/8819. This file must be judged against those.
#
# KILL-TELLS (logged): debug/tilt_shift ~ 0 -> no improvement pressure (raise tilt_gain).
# losses/distill_kl ~ 0 with flat return -> the student already matches the teacher.
# debug/tilt_kl_pred vs losses/distill_kl diverging -> the sigma-independence claim is
# false in practice and the step size is not actually constant.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6


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
    learning_rate: float = 1e-3
    num_envs: int = 16
    num_steps: int = 2048
    # --- throughput. Neither flag changes the algorithm: identical math, identical
    # per-env seeding. Measured on this chassis: the acting forward is LAUNCH-bound, not
    # compute-bound (eager act = 1149 us at batch 16, 1277 us at batch 256 -- flat), and it
    # dominated the 5.8 ms vec-step while env stepping was only 0.99 ms of it.
    #   compile mode=reduce-overhead: 1149 -> 347 us at batch 16.
    #   AsyncVectorEnv: raw env stepping 16131 -> 40357 samples/s.
    #   marginal end-to-end: 3025 -> 4300 SPS at 16 envs; 6013 at 128 envs.
    # NOTE the 8819 @8M champion was measured EAGER; arms after this patch are compiled, so
    # numerics are not bit-identical (well inside the +-498 CI95 on that measurement).
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1         # passes over the batch for the rationalization + distill
    #                               losses. Reuse here re-fits the SAME action per state, so
    #                               it sharpens the conditional (entropy drops).
    critic_epochs: int = 4        # passes for the value regression only. Reuse here is plain
    #                               supervised learning on fixed targets and is nearly free.

    # --- ANALYTIC TILT TEACHER (replaces the learned privileged channel) ---
    tilt_gain: float = 0.3        # eta: fraction of the realized deviation (z_t - m_S) the
                                  # teacher's mean moves, at saturated credit |g| = 1.
                                  # eta = 1 would place the teacher's mean exactly on the
                                  # taken action. This is the step size, and because the
                                  # shift is O(sigma) the resulting KL ~ (eta*g)^2/2 does
                                  # not depend on sigma -- see the header.
    tilt_temp: float = 1.0        # tau_A: advantage scale inside tanh. b_adv is already
                                  # unit-RMS from cond_scale=ema_rms, so 1.0 means "one RMS
                                  # of advantage saturates about 76% of the way".
    tilt_sharpen: float = 0.0     # zeta: shrink the teacher's sd by (1 - zeta*|g|). 0 = pure
                                  # mean shift. The hybrid's clone term sharpened as a SIDE
                                  # EFFECT and that was measured to be load-bearing
                                  # (removing it cost 5365 -> 4136 @2M), so expose it as an
                                  # explicit knob rather than a byproduct.
    teacher_conc_cap: float = 100.0  # cap on Beta concentration nu = alpha+beta for the
                                  # moment-matched teacher; prevents a near-boundary mean
                                  # from requesting an unrepresentable spike.
    adv_cond_clip: float = 3.0    # clamp on the scaled advantage before tanh
    cond_scale: str = "ema_rms"   # "ema_rms" | "batch" | "raw"; ema_rms is the 8819 setting
    cond_ema_beta: float = 0.99   # EMA horizon for the RMS scale (~100 iterations)
    cond_lambda: float = 0.0      # GAE lambda for the CREDIT signal driving the tilt; 0 =
                                  # 1-step TD residual, the most action-attributable signal
    distill_kl_clip: float = 2.0  # tau: the paper's per-token pointwise divergence clip

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

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


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form."""

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


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
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts
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
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            self.blocks.append(ThinkBlock(H * (k + 1), H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


MEAN_EPS = SAMPLE_EPS  # preserve legitimate sharp Beta means; z is already clamped here
SD_SAFETY = 0.995      # a Beta sd must stay below sqrt(m(1-m)); leave a margin


def beta_moments(alpha, beta):
    """Mean and sd of Beta(alpha, beta), per dim."""
    nu = alpha + beta
    mean = alpha / nu
    sd = (alpha * beta / (nu.square() * (nu + 1.0))).sqrt()
    return mean, sd


def moments_to_beta(mean, sd, conc_cap):
    """Moment-match (mean, sd) back to Beta(alpha, beta) with alpha, beta >= 1.

    Cap the CONCENTRATION nu = alpha + beta rather than alpha and beta separately. Capping
    them independently is mean-destroying: once a target is sharp enough for both to hit
    the cap, the distribution degenerates to Beta(cap, cap), whose mean is 0.5 regardless
    of what was requested -- so the teacher stops pointing anywhere and drags every action
    to the middle of the range. An earlier file in this lineage measurably entered that
    regime. Capping nu limits sharpness exactly as intended while preserving the mean.
    """
    mean = mean.clamp(MEAN_EPS, 1.0 - MEAN_EPS)
    sd_max = SD_SAFETY * (mean * (1.0 - mean)).sqrt()
    sd = torch.minimum(sd.clamp_min(1e-6), sd_max)
    nu = (mean * (1.0 - mean)) / sd.square() - 1.0                 # > 0 by construction
    nu_min = 1.0 / torch.minimum(mean, 1.0 - mean)                 # gives alpha, beta >= 1
    nu_cap = torch.as_tensor(2.0 * conc_cap, device=nu.device, dtype=nu.dtype)
    nu = torch.minimum(nu.clamp_min(1e-6).maximum(nu_min), torch.maximum(nu_cap, nu_min))
    return mean * nu, (1.0 - mean) * nu


class Agent(nn.Module):
    """ONE context. There is no privileged input block, because the teacher is not a
    forward pass through this network -- it is computed in closed form from the rollout's
    own Beta parameters, the action it sampled, and that action's credit. So the trunk
    takes obs only, and the actor and critic are exactly pi(.|s) and V(s).

    This also removes a whole context from every minibatch: the hybrid ran a 2N-wide
    forward (privileged + absent) plus an N-wide teacher snapshot pass; this runs N.
    """

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

    def _feat(self, obs):
        return self.trunk(obs)

    def policy(self, obs):
        feat = self._feat(obs)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def policy_and_value(self, obs):
        feat = self._feat(obs)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        value_logits = self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return alpha, beta, value_logits

    def act(self, obs):
        alpha, beta, value_logits = self.policy_and_value(obs)
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        return action, z, value_logits, alpha, beta

    def get_value(self, obs):
        """V(s): no action input ever enters the critic."""
        feat = self._feat(obs)
        return self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.tilt_gain > 0.0, "a zero tilt gain makes the teacher the rollout policy"
    assert 0.0 <= args.tilt_sharpen < 1.0, "tilt_sharpen must keep sd_T positive"

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
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    vector_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_cls(
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
    # Rollout forward only. The update stays eager: it is a small share of wall clock
    # (512 minibatch steps per iteration against 2048 acting steps), and graphing it would
    # complicate the dual/telemetry paths for little gain.
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    # The teacher needs the policy that actually chose each action. Recording it during the
    # rollout is exact and free; recomputing it later would only be equal by luck.
    roll_alphas = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    roll_betas = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
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

    # Slow RMS of the raw conditioning residual. Not a gradient statistic -- it exists only
    # to keep the Fourier features inside the range their fixed frequencies resolve.
    cond_ms = torch.zeros((), device=device)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, value_logits, roll_alpha, roll_beta = act_fn(next_obs)
                values[step] = hl_support.to_scalar(value_logits[:, 0])
            latent_zs[step] = z
            roll_alphas[step] = roll_alpha
            roll_betas[step] = roll_beta

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
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # ================= GAE on V(s), then the MTP critic targets =====================
        with torch.no_grad():
            next_value_logits = agent.get_value(next_obses.reshape((-1,) + obs_shape))[:, 0]
            next_values = hl_support.to_scalar(next_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # Two credit signals from the SAME V(s). GAE(gae_lambda) for critic targets,
            # and GAE(cond_lambda) for the privileged channel.
            advantages = torch.zeros_like(rewards)
            cond_adv = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs, device=device)
            last_cond = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_values[t] * bootstrap_nonterminal - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * lambda_nonterminal * last_gae
                advantages[t] = last_gae
                last_cond = delta + args.gamma * args.cond_lambda * lambda_nonterminal * last_cond
                cond_adv[t] = last_cond
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros((*returns.shape, mtp), dtype=torch.bool, device=device)
            for horizon in range(mtp):
                valid_len = args.num_steps - horizon
                valid_horizon = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for boundary_offset in range(horizon):
                    valid_horizon &= (
                        transition_boundaries[boundary_offset : boundary_offset + valid_len] == 0
                    )
                return_mtp[:valid_len, :, horizon] = returns[horizon:]
                return_mtp_mask[:valid_len, :, horizon] = valid_horizon
            target_probs = hl_support.project(return_mtp)

            # THE PRIVILEGED CHANNEL must actually IDENTIFY the action, or the network is
            # right to ignore it. Measured with GAE(0.95): cond_gap 0.002 and falling,
            # distill_kl 3e-4 -- the teacher was the student. The cause is informational,
            # not architectural: with gamma 0.99 and lambda 0.95 the advantage is dominated
            # by ~100 steps of downstream trajectory and value error, so I(a_t ; A_t | s_t)
            # is almost nil. The 1-step residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
            # is instead a near-deterministic function of a_t in a deterministic MuJoCo
            # transition, so conditioning on it is learnable, and it is exactly the classic
            # actor-critic advantage -- V(s) only, no action ever enters the critic.
            b_adv = cond_adv.reshape(-1)
            # SCALING A CONDITIONING INPUT IS NOT SCALING A GRADIENT. "batch" (v1-v5,
            # inherited from PPO adv-norm) is wrong twice over here:
            #   (a) mean subtraction destroys delta's natural zero. delta_t = 0 means
            #       "exactly as V expected"; subtracting the batch mean relabels the
            #       least-bad action of an all-bad batch as POSITIVE -- a false sign.
            #   (b) the batch sd makes the units non-stationary, and adv_boost is quoted
            #       in those units, so the same nominal margin means different things at
            #       different times.
            # Raw delta cannot be fed either: AdvEmbed's frequencies are FIXED (0.5..8).
            # Measured (v4, 131k steps): raw delta's RMS grew 0.61 -> 2.17 in 4 iterations
            # and already saturated the +-3 clip at 11%. Entropy preservation ordered
            # raw > ema_rms > batch, exactly as (a) predicts.
            if args.cond_scale == "batch":
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            elif args.cond_scale == "ema_rms":
                ms = b_adv.square().mean()
                cond_ms.mul_(args.cond_ema_beta).add_((1.0 - args.cond_ema_beta) * ms)
                bias = 1.0 - args.cond_ema_beta ** iteration
                b_adv = b_adv / (cond_ms / bias).sqrt().clamp_min(1e-8)
            elif args.cond_scale == "raw":
                pass
            else:
                raise ValueError(f"unknown cond_scale {args.cond_scale!r}")
            cond_scale_used = b_adv.square().mean().sqrt().item()
            b_adv_cond = b_adv.clamp(-args.adv_cond_clip, args.adv_cond_clip).unsqueeze(-1)
            cond_clipped = (b_adv.abs() >= args.adv_cond_clip).float().mean().item()

        b_obs = obs.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).to(torch.float32)

        # ===== THE TEACHER, IN CLOSED FORM ============================================
        # Computed once per iteration from the rollout's own Beta parameters, the action it
        # sampled, and that action's credit. No forward pass, no learned channel, nothing
        # to ignore. Detached by construction -- these are plain target tensors.
        #
        #   g    = tanh(A_hat / tilt_temp)                 in (-1, 1), and g(0) = 0 exactly,
        #                                                  so a state that performed exactly
        #                                                  as V expected asks for no change.
        #   m_T  = m_S + tilt_gain * g * (z - m_S)         toward the action if it paid off,
        #                                                  away from it if it did not.
        #   sd_T = sd_S * (1 - tilt_sharpen * |g|)
        #
        # The mean shift is proportional to the realized deviation, which is O(sd_S), so the
        # induced KL is ~ (tilt_gain * g)^2 / 2 regardless of how collapsed the policy is.
        # That is the property this file exists to test; `debug/tilt_kl_pred` logs the
        # prediction next to the measured `losses/distill_kl` so it is falsifiable.
        with torch.no_grad():
            b_roll_alpha = roll_alphas.reshape(-1, act_dim)
            b_roll_beta = roll_betas.reshape(-1, act_dim)
            m_s, sd_s = beta_moments(b_roll_alpha, b_roll_beta)
            g = torch.tanh(b_adv_cond / args.tilt_temp)          # (batch, 1), broadcasts
            m_t = m_s + args.tilt_gain * g * (b_z - m_s)
            sd_t = sd_s * (1.0 - args.tilt_sharpen * g.abs())
            b_t_alpha, b_t_beta = moments_to_beta(m_t, sd_t, args.teacher_conc_cap)
            tilt_shift = (m_t - m_s).abs().mean().item()
            tilt_kl_pred = (0.5 * (args.tilt_gain * g).square()).mean().item() * act_dim
            g_abs = g.abs().mean().item()

        distill_kls, v_losses, ents, tea_ents = [], [], [], []
        clip_fracs = []
        # ===== DECOUPLED ACTOR / CRITIC BUDGETS ========================================
        # e4 beat e1 by ~1260 at 1M on mb128 -- but e4 also handed the CRITIC 4x its
        # regression passes, and those two kinds of reuse are not the same thing:
        #   actor reuse re-fits the SAME sampled action for the same state. It adds no
        #     information, it only sharpens the conditional, and it is paid for in entropy.
        #   critic reuse is ordinary supervised regression onto FIXED bootstrap targets,
        #     where extra passes simply reduce fitting error.
        # So they get separate budgets. Epochs past actor_epochs take a critic-only path:
        # one zeroed-context forward, no policy heads, roughly half the compute.
        for epoch in range(max(args.actor_epochs, args.critic_epochs)):
            do_actor = epoch < args.actor_epochs
            do_critic = epoch < args.critic_epochs
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]

                if not do_actor:
                    value_logits = agent.get_value(b_obs[mb])
                    log_value_probs = torch.log_softmax(value_logits, dim=-1)
                    value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                    v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        v_losses.append(v_loss.item())
                    continue

                obs_mb, z_mb = b_obs[mb], b_z[mb]
                a_tea, b_tea = b_t_alpha[mb], b_t_beta[mb]

                # ONE context, ONE forward. pi(.|s) is both the acting policy and the only
                # thing being trained.
                a_stu, b_stu, value_logits = agent.policy_and_value(obs_mb)

                # THE ONLY ACTOR LOSS: per-dim clipped forward KL from the analytic teacher
                # into the student. No likelihood term, no ratio, no surrogate, no weights.
                kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()

                if do_critic:
                    log_value_probs = torch.log_softmax(value_logits, dim=-1)
                    value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                    v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()
                    loss = loss + args.vf_coef * v_loss
                    v_losses.append(v_loss.item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    ents.append(
                        Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    tea_ents.append(
                        Beta(a_tea, b_tea, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    clip_fracs.append(
                        (kl_dims >= args.distill_kl_clip - 1e-6).float().mean().item()
                    )

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("debug/tilt_shift", tilt_shift, global_step)
        writer.add_scalar("debug/tilt_kl_pred", tilt_kl_pred, global_step)
        writer.add_scalar("debug/tilt_g_abs", g_abs, global_step)
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(v_losses)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/kl_clip_frac", float(np.mean(clip_fracs)), global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/cond_scale_rms", cond_scale_used, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
