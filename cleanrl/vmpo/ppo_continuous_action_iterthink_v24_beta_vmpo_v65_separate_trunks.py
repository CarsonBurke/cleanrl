# V-MPO v65 — v62 with an optional separate critic trunk (--separate-trunks).
#
# v62 shares one ThinkTrunk between the Beta policy head and the HL-Gauss
# critic. In V-MPO the policy loss is a weighted maximum likelihood over the
# top half of a 2496-sample batch with centred weights of order 1/N, while
# the critic loss is a 51-way cross-entropy on every sample; the shared
# representation is therefore shaped almost entirely by the critic. The PPO
# recipes in this repo that reach 14.5k on HalfCheetah at 50M all use
# separate trunks. Hypothesis: with its own trunk the policy representation
# is free to specialise, and the E-step budget (epsilon_eta 0.03 is the peak
# of the v62 dose-response: 9315 / 10272 / 11048 / 9928 / 7951 for
# 0.01 / 0.02 / 0.03 / 0.05 / 0.1 at 50M) converts into more return.
# Defaults reproduce v62 exactly; --separate-trunks builds a second
# ThinkTrunk for the critic. The KL anchor still copies the whole agent but
# only its policy path is evaluated.
#
# --- v62 header follows ---
#
# Algorithm is v61 unchanged: online actor, EMA KL anchor (tau 0.1), exact eta
# by bisection, decoupled Beta trust region with Lagrangian duals, HL-Gauss
# critic with moment-matched labels, percentile-scaled GAE advantages, one
# full-batch Adam step per rollout, and the switchable E-step options
# (--center-weights, --topk-fraction, --epsilon-eta, --segment-normalize).
#
# Execution is rebuilt for throughput. v61 stepped a SyncVectorEnv of wrapped
# Gymnasium envs from Python, ran an eager Beta sampler and five small
# transfers per step, and normalised in NumPy. Measured 15k SPS under load.
# v62 uses the native batched MuJoCo backend with a small thread pool, the
# fused per-env observation/reward normalisers, one captured CUDA graph per
# step (upload, policy, sampling, storage scatter, action download; a single
# host sync), and one packed upload per rollout. Next-state critic values for
# GAE come from one full-batch forward over the stored transition
# observations, exactly as v30/v61 computed them.
# The learner paths compile in mode "default". On torch 2.12 the
# reduce-overhead cudagraph trees invalidate the manually captured rollout
# graph: the first replay after the first compiled update faults with an
# illegal memory access (bisected: default and eager survive, reduce-overhead
# fails with and without the split-cosine diagnostic). Measured cost of
# default mode is ~10 ms per 64x39 update; the run holds 54-58k SPS.
#
# One learner option is new: --dual-lr gives the two Lagrange multipliers
# their own Adam rate. At the shared 3e-4 they move at most 3e-4 per update,
# so in v61 the mean KL sat 2x over its bound and the concentration KL 13-70x
# over for the whole run; the trust region never bound and the policy
# concentration ran from 5 to 38 in 7M steps.
#
# Defaults reproduce v61 defaults (= v60: uncentered median-cut E-step).

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.collector import OnPolicyCollector
from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
DUAL_FLOOR = 1e-8


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 64
    num_steps: int = 39
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Native MuJoCo physics threads for the vector env; a per-run latency knob.
    env_threads: int = 2
    env_backend: str = "native"
    non_blocking_transfers: bool = True

    # EMA rate of the KL-anchor policy per learner update. 1.0 makes the anchor
    # the rollout snapshot (per-batch trust region); 0.1 lags ~10 updates.
    anchor_ema_tau: float = 0.1
    topk_fraction: float = 0.5
    center_weights: bool = False
    segment_normalize: bool = False
    # Split-half policy-gradient cosine on log iterations (eager FP32 autograd).
    split_cosine_diagnostic: bool = True
    epsilon_eta: float = 0.01
    # Neutral geometric midpoints of the paper's Gym log-uniform search ranges.
    epsilon_alpha_mean: float = 0.007071067811865476
    epsilon_alpha_concentration: float = 1.5811388300841898e-5
    initial_alpha_mean: float = 1.0
    initial_alpha_concentration: float = 1.0
    # Adam rate of the KL duals; None shares learning_rate with the network.
    dual_lr: float | None = None
    return_percentile_low: float = 0.05
    return_percentile_high: float = 0.95
    return_percentile_floor: float = 1.0
    num_value_bins: int = 51
    value_support_limit: float = 20_000.0
    value_sigma_to_bin_ratio: float = 0.75

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    # Give the critic its own ThinkTrunk instead of sharing the policy's.
    separate_trunks: bool = False

    compile: bool = True
    compile_mode: str = "default"
    log_interval: int = 10
    save_checkpoint: bool = True

    batch_size: int = 0
    topk_size: int = 0
    estep_rows: int = 0
    estep_columns: int = 0
    num_iterations: int = 0
    initial_phase_warmup_steps: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def branch_body(hidden):
    return nn.Sequential(
        layer_init(nn.Linear(hidden, hidden)),
        ReLUSquared(),
        layer_init(nn.Linear(hidden, hidden)),
    )


class FusedExperts(nn.Module):
    """Equivalent expert MLPs stored as two batched GEMMs."""

    def __init__(self, n_experts, hidden):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(n_experts, hidden, hidden))
        self.bias1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.weight2 = nn.Parameter(torch.empty(n_experts, hidden, hidden))
        self.bias2 = nn.Parameter(torch.zeros(n_experts, hidden))
        with torch.no_grad():
            for expert_index in range(n_experts):
                nn.init.orthogonal_(self.weight1[expert_index], np.sqrt(2))
                nn.init.orthogonal_(self.weight2[expert_index], np.sqrt(2))

    def forward(self, x):
        hidden = torch.einsum("bi,eoi->beo", x, self.weight1) + self.bias1
        hidden = torch.relu(hidden).square()
        return torch.einsum("bei,eoi->beo", hidden, self.weight2) + self.bias2


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, hidden, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, hidden))
        self.resid_gate = nn.Parameter(torch.full((hidden,), 4.0))
        self.dense_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.dense = branch_body(hidden)
        self.moe_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(hidden, n_experts))
        self.experts = FusedExperts(n_experts, hidden)

    def forward(self, features, x0):
        x = self.in_proj(features)
        gate = torch.sigmoid(self.resid_gate)
        residual = gate * x + (1.0 - gate) * x0
        dense = self.dense(self.dense_norm(residual))
        moe_input = self.moe_norm(residual)
        weights = torch.softmax(self.gate(moe_input), dim=-1)
        expert_outputs = self.experts(moe_input)
        moe = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return residual + dense + moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, hidden, k_blocks, n_experts):
        super().__init__()
        self.output_dim = hidden * (k_blocks + 1)
        self.entry = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.ModuleList(
            [ThinkBlock(hidden * (index + 1), hidden, n_experts) for index in range(k_blocks)]
        )
        self.out_norm = nn.RMSNorm(self.output_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(self.output_dim, self.output_dim))

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(features, dim=-1)))


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.value_trunk = (
            ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
            if args.separate_trunks
            else None
        )
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.actor_alpha = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(self.trunk.output_dim, 256)),
            ReLUSquared(),
        )
        self.value_head = nn.Linear(256, args.num_value_bins, bias=False)
        with torch.no_grad():
            self.value_head.weight.zero_()
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def policy(self, observations):
        features = self.policy_mlp(self.trunk(observations))
        alpha = 1.0 + F.softplus(self.actor_alpha(features))
        beta = 1.0 + F.softplus(self.actor_beta(features))
        return alpha, beta

    def forward(self, observations):
        features = self.trunk(observations)
        policy_features = self.policy_mlp(features)
        alpha = 1.0 + F.softplus(self.actor_alpha(policy_features))
        beta = 1.0 + F.softplus(self.actor_beta(policy_features))
        if self.value_trunk is not None:
            features = self.value_trunk(observations)
        value_logits = self.value_head(self.value_mlp(features))
        return alpha, beta, value_logits

    def value_logits(self, observations):
        trunk = self.trunk if self.value_trunk is None else self.value_trunk
        return self.value_head(self.value_mlp(trunk(observations)))



def beta_log_prob(alpha, beta, action):
    action = action.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    log_normalizer = (
        torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    )
    return (
        (alpha - 1.0) * action.log()
        + (beta - 1.0) * torch.log1p(-action)
        - log_normalizer
    ).sum(-1)


def beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    old_sum = old_alpha + old_beta
    new_sum = new_alpha + new_beta
    kl = (
        torch.lgamma(new_alpha)
        + torch.lgamma(new_beta)
        - torch.lgamma(new_sum)
        - torch.lgamma(old_alpha)
        - torch.lgamma(old_beta)
        + torch.lgamma(old_sum)
        + (old_alpha - new_alpha) * torch.digamma(old_alpha)
        + (old_beta - new_beta) * torch.digamma(old_beta)
        + (new_sum - old_sum) * torch.digamma(old_sum)
    )
    return kl.sum(-1)


def decoupled_beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    old_concentration = old_alpha + old_beta
    new_concentration = new_alpha + new_beta
    old_mean = old_alpha / old_concentration
    new_mean = new_alpha / new_concentration

    mean_alpha = (new_mean * old_concentration).clamp_min(SAMPLE_EPS)
    mean_beta = ((1.0 - new_mean) * old_concentration).clamp_min(SAMPLE_EPS)
    concentration_alpha = (old_mean * new_concentration).clamp_min(SAMPLE_EPS)
    concentration_beta = ((1.0 - old_mean) * new_concentration).clamp_min(SAMPLE_EPS)
    mean_kl = beta_kl(old_alpha, old_beta, mean_alpha, mean_beta)
    concentration_kl = beta_kl(
        old_alpha, old_beta, concentration_alpha, concentration_beta
    )
    return mean_kl, concentration_kl


# Order matches the metrics stack returned by update_loss_model.
METRIC_NAMES = (
    "losses/policy_loss",
    "losses/value_loss",
    "losses/temperature_loss",
    "vmpo/mean_kl",
    "vmpo/concentration_kl",
    "vmpo/anchor_full_kl",
    "vmpo/weight_ess_fraction",
    "vmpo/top_advantage_min",
    "debug/advantage_mean",
    "debug/advantage_std",
    "vmpo/e_step_kl",
    "vmpo/eta_stationarity",
    "vmpo/weight_perplexity_fraction",
    "vmpo/max_weight",
    "vmpo/weight_ess",
    "vmpo/mean_kl_residual",
    "vmpo/concentration_kl_residual",
    "debug/value_rmse",
    "debug/value_explained_variance",
    "critic/target_outside_support",
    "critic/target_edge_mass",
    "critic/prediction_edge_mass",
    "debug/policy_concentration",
    "debug/policy_native_variance",
    "vmpo/eta",
    "vmpo/rollout_full_kl",
    "debug/value_mean",
)


def validate_args(args):
    if args.num_steps <= 0 or args.num_envs <= 0:
        raise ValueError("num_steps and num_envs must be positive")
    if args.env_threads <= 0:
        raise ValueError("env_threads must be positive")
    if args.env_backend not in {"native", "threaded", "sync"}:
        raise ValueError("env_backend must be native, threaded or sync")
    args.batch_size = args.num_envs * args.num_steps
    # E-step normalisation groups are columns of a [rows, columns] view of the
    # flattened [T, N] rollout: one column for a batch-level target, N columns
    # of T steps for per-segment targets.
    if args.segment_normalize:
        args.estep_rows, args.estep_columns = args.num_steps, args.num_envs
    else:
        args.estep_rows, args.estep_columns = args.batch_size, 1
    args.topk_size = int(args.estep_rows * args.topk_fraction)
    if not 0 < args.topk_size <= args.estep_rows:
        raise ValueError("topk_fraction produces an invalid per-group top-k size")
    args.initial_phase_warmup_steps = episode_horizon(args.env_id)
    warmup_transitions = args.num_envs * args.initial_phase_warmup_steps
    if warmup_transitions >= args.total_timesteps:
        raise ValueError("total_timesteps must exceed the initial phase warmup")
    args.num_iterations = (args.total_timesteps - warmup_transitions) // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("total_timesteps yields zero full batches after warmup")
    if not 0.0 < args.anchor_ema_tau <= 1.0:
        raise ValueError("anchor_ema_tau must be in (0, 1]")
    if args.dual_lr is not None and args.dual_lr <= 0.0:
        raise ValueError("dual_lr must be positive")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if not 0.0 <= args.return_percentile_low < args.return_percentile_high <= 1.0:
        raise ValueError("return percentiles must satisfy 0 <= low < high <= 1")
    if args.return_percentile_floor <= 0.0:
        raise ValueError("return_percentile_floor must be positive")
    if args.num_value_bins < 3 or args.num_value_bins % 2 == 0:
        raise ValueError("num_value_bins must be odd and at least three")
    if args.value_support_limit <= 0.0 or args.value_sigma_to_bin_ratio <= 0.0:
        raise ValueError("value support limit and sigma ratio must be positive")
    return args


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required")
    configure_runtime(
        cudnn_deterministic=args.torch_deterministic, matmul_precision="high", allow_tf32=True
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16
    estep_rows, estep_columns = args.estep_rows, args.estep_columns

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )
    envs = make_mujoco_vector_env(
        args.env_id,
        args.num_envs,
        backend=args.env_backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video,
        run_name=run_name,
    )
    collector = None
    try:
        if not isinstance(envs.single_action_space, gym.spaces.Box):
            raise TypeError("V-MPO continuous control requires a Box action space")
        observation_shape = tuple(envs.single_observation_space.shape)

        symlog_limit = float(np.log1p(args.value_support_limit))
        hl_support = Dreamer3BucketHLGaussSupport(
            args.num_value_bins, -symlog_limit, symlog_limit, args.value_sigma_to_bin_ratio, device
        )
        agent = Agent(envs, args).to(device)
        # The online policy acts. The anchor is an EMA of the online parameters,
        # used only to evaluate the M-step KL prior; it never generates data.
        anchor_agent = copy.deepcopy(agent).requires_grad_(False)
        anchor_params = list(anchor_agent.parameters())
        online_params = list(agent.parameters())
        duals = nn.Parameter(
            torch.tensor(
                [args.initial_alpha_mean, args.initial_alpha_concentration], device=device
            )
        )
        dual_lr = args.learning_rate if args.dual_lr is None else args.dual_lr
        optimizer = optim.Adam(
            [
                {"params": list(agent.parameters())},
                {"params": [duals], "lr": dual_lr},
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )

        def rollout_model(observations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                alpha, beta, value_logits = agent(observations)
            values = hl_support.to_scalar(value_logits.float())
            return alpha.float(), beta.float(), values

        def value_model(observations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                value_logits = agent.value_logits(observations)
            return hl_support.to_scalar(value_logits.float())

        gae = get_gae_fn(compiled=args.compile, mode=args.compile_mode, explicit_next_values=True)

        def gae_model(transition_observations, rewards, values, terminations, truncations):
            # One full-batch next-state critic evaluation feeding the v30
            # recurrence: bootstrap through truncations, reset traces at every
            # boundary.
            next_values = value_model(transition_observations.flatten(0, 1)).view(
                args.num_steps, args.num_envs
            )
            return gae(
                rewards, values, terminations, truncations, next_values, args.gamma, args.gae_lambda
            )

        def update_loss_model(
            observations,
            native_actions,
            old_alpha,
            old_beta,
            advantages,
            value_targets,
        ):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                new_alpha, new_beta, value_logits = agent(observations)
                with torch.no_grad():
                    anchor_alpha, anchor_beta = anchor_agent.policy(observations)
            new_alpha = new_alpha.float()
            new_beta = new_beta.float()
            anchor_alpha = anchor_alpha.float()
            anchor_beta = anchor_beta.float()
            value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
            value_probs = value_log_probs.exp()
            values = hl_support.probs_to_scalar(value_probs)

            alpha_mean = duals[0].clamp_min(DUAL_FLOOR)
            alpha_concentration = duals[1].clamp_min(DUAL_FLOOR)

            with torch.no_grad():
                grouped = advantages.view(estep_rows, estep_columns)
                # Match RLax's inclusive kth threshold, including every cutoff tie.
                group_threshold = torch.sort(grouped, dim=0).values[-args.topk_size]
                selected = grouped >= group_threshold
                group_selected_count = selected.sum(dim=0).to(grouped.dtype)
                log_group_selected_count = group_selected_count.log()

                # Center each group at its maximum before division. This preserves
                # softmax weights and the eta derivative while avoiding a large
                # common logit that would erase near-maximum differences in FP32.
                group_maximum = grouped.max(dim=0).values
                centered_advantages = grouped - group_maximum
                group_span = group_maximum - group_threshold

                def group_target(log_eta):
                    logits = torch.where(
                        selected, centered_advantages / log_eta.exp(), -torch.inf
                    )
                    log_w = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                    w = log_w.exp()
                    safe_log_w = torch.where(selected, log_w, 0.0)
                    # Mean over groups: the KL of the equal-mass mixture of group
                    # targets to the equal-mass mixture of per-group uniforms.
                    kl = (w * (safe_log_w + log_group_selected_count)).sum(dim=0).mean()
                    return kl, w, logits

                # max span / epsilon is a feasible upper bracket for every group:
                # each selected logit range is then at most epsilon, so no group's
                # KL to its uniform can exceed epsilon. Bisection is geometric
                # because eta can span many orders of magnitude.
                log_eta_low = torch.full_like(group_span.max(), np.log(DUAL_FLOOR))
                log_eta_high = (
                    group_span.max().div(args.epsilon_eta).clamp_min(DUAL_FLOOR).log()
                )
                for _ in range(32):
                    log_eta_mid = 0.5 * (log_eta_low + log_eta_high)
                    mid_kl, _, _ = group_target(log_eta_mid)
                    infeasible = mid_kl > args.epsilon_eta
                    log_eta_low = torch.where(infeasible, log_eta_mid, log_eta_low)
                    log_eta_high = torch.where(infeasible, log_eta_high, log_eta_mid)
                eta = log_eta_high.exp()

                temperature_kl, group_weights, policy_logits = group_target(log_eta_high)
                temperature_loss = (
                    group_maximum
                    + eta
                    * (
                        args.epsilon_eta
                        + torch.logsumexp(policy_logits, dim=0)
                        - log_group_selected_count
                    )
                ).mean()
                # Each group carries mass 1 / columns, so the weights sum to one and
                # reshape back to the flattened [T, N] sample order.
                weights = (group_weights / estep_columns).reshape(-1)
                selected_count = group_selected_count.sum()
                topk_threshold = group_threshold.mean()
                if args.center_weights:
                    # Batch-mean score as a control variate: exact zero mean at the
                    # rollout policy, identically zero weights when psi is uniform.
                    loss_weights = weights - 1.0 / args.batch_size
                else:
                    loss_weights = weights

            log_prob = beta_log_prob(new_alpha, new_beta, native_actions)
            policy_loss = -(loss_weights * log_prob).sum()

            mean_kl, concentration_kl = decoupled_beta_kl(
                anchor_alpha, anchor_beta, new_alpha, new_beta
            )
            mean_kl_average = mean_kl.mean()
            concentration_kl_average = concentration_kl.mean()
            mean_constraint_loss = (
                alpha_mean * (args.epsilon_alpha_mean - mean_kl_average.detach())
                + alpha_mean.detach() * mean_kl_average
            )
            concentration_constraint_loss = (
                alpha_concentration
                * (
                    args.epsilon_alpha_concentration
                    - concentration_kl_average.detach()
                )
                + alpha_concentration.detach() * concentration_kl_average
            )
            with torch.no_grad():
                value_target_probs = hl_support.project_moment_matched(value_targets)
            value_loss = -(value_target_probs * value_log_probs).sum(dim=-1).mean()
            total_loss = (
                policy_loss
                + mean_constraint_loss
                + concentration_constraint_loss
                + value_loss
            )

            # KL to the rollout policy is the realised per-update movement; KL to
            # the anchor is what the duals constrain.
            rollout_full_kl = beta_kl(old_alpha, old_beta, new_alpha, new_beta).mean()
            anchor_full_kl = beta_kl(
                anchor_alpha, anchor_beta, new_alpha, new_beta
            ).mean()
            effective_sample_size = weights.square().sum().reciprocal()
            effective_sample_fraction = effective_sample_size / selected_count
            eta_stationarity = args.epsilon_eta - temperature_kl
            mean_kl_residual = mean_kl_average - args.epsilon_alpha_mean
            concentration_kl_residual = (
                concentration_kl_average - args.epsilon_alpha_concentration
            )
            value_error = values - value_targets
            value_rmse = value_error.square().mean().sqrt()
            explained_variance = 1.0 - value_error.var(unbiased=False) / (
                value_targets.var(unbiased=False) + 1e-8
            )
            target_outside_support = (
                value_targets.abs() > args.value_support_limit
            ).float().mean()
            target_edge_mass = (
                value_target_probs[:, 0] + value_target_probs[:, -1]
            ).mean()
            prediction_edge_mass = (
                value_probs[:, 0] + value_probs[:, -1]
            ).mean()
            policy_concentration = (new_alpha + new_beta).mean()
            policy_variance = (
                new_alpha
                * new_beta
                / (
                    (new_alpha + new_beta).square()
                    * (new_alpha + new_beta + 1.0)
                )
            ).mean()
            metrics = torch.stack(
                (
                    policy_loss.detach(),
                    value_loss.detach(),
                    temperature_loss.detach(),
                    mean_kl_average.detach(),
                    concentration_kl_average.detach(),
                    anchor_full_kl.detach(),
                    effective_sample_fraction.detach(),
                    topk_threshold.detach(),
                    advantages.mean().detach(),
                    advantages.std().detach(),
                    temperature_kl.detach(),
                    eta_stationarity.detach(),
                    (-temperature_kl).exp().detach(),
                    group_weights.max().detach(),
                    effective_sample_size.detach(),
                    mean_kl_residual.detach(),
                    concentration_kl_residual.detach(),
                    value_rmse.detach(),
                    explained_variance.detach(),
                    target_outside_support.detach(),
                    target_edge_mass.detach(),
                    prediction_edge_mass.detach(),
                    policy_concentration.detach(),
                    policy_variance.detach(),
                    eta.detach(),
                    rollout_full_kl.detach(),
                    values.mean().detach(),
                )
            )
            return total_loss, metrics, loss_weights


        if args.compile:
            rollout_model = graph_compile(rollout_model)
            gae_model = torch.compile(
                gae_model, mode=args.compile_mode, fullgraph=True, dynamic=False
            )
            update_loss_model = torch.compile(
                update_loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False
            )
            print(f"compiled BF16 fullgraph learner paths ({args.compile_mode})")

        def sample(observations):
            alpha, beta, value = rollout_model(observations)
            native, action = sample_beta_actions(alpha, beta, agent.action_low, agent.action_high)
            return dict(action=action, native_action=native, alpha=alpha, beta=beta, value=value)

        suppress = np.zeros(args.num_envs, dtype=bool)
        rollout_returns = []

        def log_episodes(infos, step):
            for index, info in enumerate(infos.get("final_info", ())):
                if info and "episode" in info:
                    if suppress[index]:
                        suppress[index] = False
                        continue
                    episode_return = float(info["episode"]["r"])
                    rollout_returns.append(episode_return)
                    print(f"global_step={step}, episodic_return={episode_return}")
                    writer.add_scalar("charts/episodic_return", episode_return, step)
                    writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), step)

        # One RunningMeanStd row per environment, as v30's independent wrappers.
        obs_norm = VectorObsNorm(args.num_envs, observation_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        collector = OnPolicyCollector(
            envs,
            args.num_steps,
            sample,
            obs_norm,
            rew_norm,
            non_blocking=args.non_blocking_transfers,
            episode_callback=log_episodes,
        )
        start_time = time.perf_counter()
        offsets = compute_phase_offsets(args.num_envs, args.initial_phase_warmup_steps, args.seed)
        writer.add_text("initial_phase_offsets", ",".join(map(str, offsets)))

        def warmup_action(observations):
            action = collector.graph.step(observations)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite warmup actions")
            return action

        warm = run_phase_warmup(
            envs,
            obs_norm=obs_norm,
            rew_norm=rew_norm,
            act_fn=warmup_action,
            horizon=args.initial_phase_warmup_steps,
            phase_offsets=offsets,
            seed=args.seed,
        )
        suppress[:] = warm.suppress_mask
        collector.set_observation(warm.next_obs, total_steps=warm.transitions)
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, collector.total_steps)

        policy_parameters = [
            parameter
            for name, parameter in agent.named_parameters()
            if not name.startswith("value_")
        ]
        # Whole segments stay together, so the halves are independent trajectories.
        split_mask = (
            (torch.arange(args.batch_size, device=device) % args.num_envs) % 2
        ) == 0

        def policy_gradient_split_cosine(loss_weights, flat_observations, flat_actions):
            """Cosine between policy gradients from even and odd environments.

            Eager FP32, log iterations only. Near one means the E-step target is
            consistent across independent trajectories; near zero means noise.
            Each half is centered on its own weight mean so that an uneven split
            of total weight between halves does not add an anti-correlated
            common component; the cosine is scale invariant so this is free.
            """
            halves = []
            for mask in (split_mask, ~split_mask):
                half_weights = loss_weights[mask]
                half_weights = half_weights - half_weights.mean()
                alpha, beta = agent.policy(flat_observations[mask])
                log_prob = beta_log_prob(alpha, beta, flat_actions[mask])
                loss = -(half_weights * log_prob).sum()
                grads = torch.autograd.grad(loss, policy_parameters)
                halves.append(torch.cat([grad.reshape(-1) for grad in grads]))
            return F.cosine_similarity(halves[0], halves[1], dim=0)


        timer = PhaseTimer()
        percentile_levels = torch.tensor(
            [args.return_percentile_low, args.return_percentile_high], device=device
        )
        interval_start, interval_step = time.perf_counter(), collector.total_steps
        for iteration in range(1, args.num_iterations + 1):
            batch = collector.collect()
            global_step = collector.total_steps
            with timer.span("gae"), torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                gae_advantages, returns = gae_model(
                    batch.transitions.transition_observations,
                    batch.transitions.rewards,
                    batch.policy["value"],
                    batch.transitions.terminations,
                    batch.transitions.truncations,
                )
                advantages = gae_advantages.flatten().clone()
                value_targets = returns.flatten().clone()
                del gae_advantages, returns
                percentiles = torch.quantile(value_targets, percentile_levels)
                return_percentile_scale = (percentiles[1] - percentiles[0]).clamp_min(
                    args.return_percentile_floor
                )
                advantages.div_(return_percentile_scale)
            flat_observations = batch.observations.flatten(0, 1)
            flat_actions = batch.policy["native_action"].flatten(0, 1)
            should_log = iteration % args.log_interval == 0 or iteration == 1
            duals_before = duals.detach().clone() if should_log else None
            split_cosine = None
            with timer.span("update"):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                total_loss, metrics, loss_weights = update_loss_model(
                    flat_observations,
                    flat_actions,
                    batch.policy["alpha"].flatten(0, 1),
                    batch.policy["beta"].flatten(0, 1),
                    advantages,
                    value_targets,
                )
                if should_log and args.split_cosine_diagnostic and args.num_envs >= 2:
                    # Measured at the pre-update parameters that produced the weights.
                    split_cosine = policy_gradient_split_cosine(
                        loss_weights.detach().clone(), flat_observations, flat_actions
                    )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    duals.clamp_(min=DUAL_FLOOR)
                    torch._foreach_lerp_(anchor_params, online_params, args.anchor_ema_tau)

            if should_log:
                assert duals_before is not None
                named = dict(zip(METRIC_NAMES, metrics)) | {
                    "vmpo/alpha_mean": duals[0],
                    "vmpo/alpha_concentration": duals[1],
                    "vmpo/alpha_mean_delta": duals[0] - duals_before[0],
                    "vmpo/alpha_concentration_delta": duals[1] - duals_before[1],
                    "debug/return_percentile_scale": return_percentile_scale,
                }
                if split_cosine is not None:
                    named["debug/policy_grad_split_cosine"] = split_cosine
                logged = gather_metrics(named)
                if not all(math.isfinite(value) for value in logged.values()):
                    raise FloatingPointError("nonfinite learner metric; see the last logged step")
                now = time.perf_counter()
                sps = global_step / (now - start_time)
                logged.update(
                    {
                        "charts/learning_rate": args.learning_rate,
                        "charts/SPS": sps,
                        "charts/interval_SPS": (global_step - interval_step) / (now - interval_start),
                        "charts/completed_episodes": float(len(rollout_returns)),
                        "vmpo/learner_updates": float(iteration),
                    }
                )
                if rollout_returns:
                    logged["charts/rollout_episodic_return"] = float(np.mean(rollout_returns))
                for phases in (collector.timer.summary(), timer.summary()):
                    for phase, measured in phases.items():
                        logged[f"timing/{phase}_s"] = float(measured["total_s"])
                for name, value in logged.items():
                    writer.add_scalar(name, value, global_step)
                print(f"SPS: {int(sps)}", flush=True)
                rollout_returns.clear()
                collector.timer.reset()
                timer.reset()
                interval_start, interval_step = now, global_step

        if args.save_checkpoint:
            # Learning state only: the native vector env and episode wrappers
            # are not serialised, so this is not an exact environment resume.
            checkpoint = {
                "version": 1,
                "agent": agent.state_dict(),
                "anchor_agent": anchor_agent.state_dict(),
                "duals": duals.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "steps": collector.total_steps,
                "args": asdict(args),
                "observation_normalizer": {
                    "means": obs_norm.means.copy(),
                    "variances": obs_norm.variances.copy(),
                    "counts": obs_norm.counts.copy(),
                },
                "reward_normalizer": {
                    "returns": rew_norm.returns.copy(),
                    "means": rew_norm.means.copy(),
                    "variances": rew_norm.variances.copy(),
                    "counts": rew_norm.counts.copy(),
                },
                "exact_environment_resume": False,
            }
            checkpoint_path = f"runs/{run_name}/final_learning_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
            print(f"learning checkpoint saved to {checkpoint_path}")
    finally:
        try:
            (collector.close if collector is not None else envs.close)()
        finally:
            writer.close()


if __name__ == "__main__":
    main()
