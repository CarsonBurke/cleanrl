# PPO with a learned kernel-mean embedding of a bootstrapped sampled return.
#
# A return encoder learns H(G), where G is one recursive bootstrapped return, and the state
# critic learns z(s)=E[H(G)|s]. One isolated coordinate is the standardized mean return
# consumed by PPO; the remaining coordinates learn decorrelated nonlinear return
# statistics. There are no actions, Q-values, fixed quantiles, horizon lists, future-state
# targets, or per-step reward occupancies.
#
# The scalar critic keeps its full standardized gradient budget. Rich gradients are
# separately clipped and cannot enter the shared trunk until their held-out prediction EV
# is positive; once enabled, opposing components are removed before combination with the
# scalar-value and policy gradients. An EMA return encoder supplies a single stable target
# frame for each rollout. The v9 LeakyReLU(0.5)^2 trunk, actor, and initialization stream
# are retained.
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from math import log
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import CompiledModule

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    # NOTE: the reference run this variant is built against (exp-name
    # "iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1", 8278 @8M) was NOT a separate
    # file -- it was mbpercnorm_v2 launched with --norm-adv --norm-adv-scope batch
    # --no-ret-percnorm. Those three are baked in as defaults here so the baseline is
    # reproduced by default rather than by remembering flags.
    norm_adv: bool = True            # ppoadvnorm: plain PPO advantage standardization...
    # --- Percentile advantage normalization (disabled: superseded by norm_adv) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's raw-reward target. A separate soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Alpha is auto-tuned by SAC's
    # exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # --- Learned total-return kernel ---------------------------------------------------
    value_latent_dim: int = 8        # one exact mean direction + seven learned statistics
    return_hidden: int = 64
    return_feature_reg_coef: float = 1.0
    return_consistency_coef: float = 1.0
    return_model_lr: float = 1e-4
    return_model_epochs: int = 4
    return_model_batch: int = 4096
    return_model_grad_clip: float = 1.0
    return_encoder_ema_rate: float = 0.20
    value_stat_ema: float = 0.01
    return_stat_ema: float = 0.01
    critic_loss_scale: float = 6.0   # v13-parity critic scale before its separate clip
    rich_value_coef: float = 1.0
    rich_grad_clip: float = 0.05     # representation shaping cannot consume scalar budget
    rich_trunk_ratio: float = 0.25   # auxiliary trunk norm relative to protected primary
    rich_holdout_envs: int = 2       # complete env streams, not adjacent-transition leakage
    rich_gate_ev: float = 0.05
    rich_gate_rank: float = 2.0
    rich_gate_prediction_rank: float = 1.5
    rich_gate_min_target_std: float = 0.25
    rich_gate_min_prediction_std: float = 0.05
    rich_gate_patience: int = 5
    rich_gate_fail_patience: int = 5
    mc_window: int = 500             # common unbootstrapped value-quality gate

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"    # ...standardized once over the whole rollout ("batch")

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- torch.compile ---------------------------------------------------------------
    # reduce-overhead (CUDA graph trees) is NOT usable on the actor/critic path: v_loss and
    # pg_loss share the trunk forward, and the update runs backward twice with
    # retain_graph=True, cloning and re-adding grads in between. Cudagraph outputs live in
    # the graph pool, clip_grad_norm_ mutates them in place, and the second backward replays
    # over live refs -> either "accessing tensor output of CUDAGraphs that has been
    # overwritten" or a re-record per minibatch (x320/iter), which is SLOWER than eager.
    # So both online and EMA trunks get inductor fusion without graphs.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"

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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


LEAKY_RELU_SLOPE = 0.5
LEAKY_RELU_SQUARED_OUT_GAIN = np.sqrt(2.0 / (1.0 + LEAKY_RELU_SLOPE**4))


class LeakyReLUSquared(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE).square()


def _branch_body(H):
    # ThinkBlock applies a non-affine RMSNorm before this branch.  For z ~ N(0, 2),
    # E[LeakyReLU_a(z)^4] = 6(1+a^4); this final gain keeps E[out^2] at 12,
    # matching the original ReLU^2 branch rather than changing residual scale.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        LeakyReLUSquared(),
        layer_init(nn.Linear(H, H), std=LEAKY_RELU_SQUARED_OUT_GAIN),
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


def project_null(x, decoder_direction):
    """Remove the one direction consumed by PPO's scalar value readout."""
    return x - (x @ decoder_direction).unsqueeze(-1) * decoder_direction


class ReturnKernelEncoder(nn.Module):
    """Learned nonlinear coordinates of one standardized total-return outcome."""

    def __init__(self, rich_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(1, hidden)),
            LeakyReLUSquared(),
            layer_init(nn.Linear(hidden, hidden)),
            LeakyReLUSquared(),
            layer_init(nn.Linear(hidden, rich_dim), std=0.01),
        )

    def forward(self, standardized_return):
        return self.net(standardized_return.unsqueeze(-1))


@torch.no_grad()
def ema_update(target, online, rate):
    target_params = list(target.parameters())
    online_params = list(online.parameters())
    if len(target_params) != len(online_params):
        raise RuntimeError("EMA modules do not have matching parameter structures")
    for target_param, online_param in zip(target_params, online_params, strict=True):
        target_param.lerp_(online_param, rate)


def return_code_regularization(coords, standardized_returns):
    """Keep nonlinear coordinates noncollapsed and distinct from the scalar return."""
    centered = coords - coords.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-6)
    variance_loss = F.relu(1.0 - std).mean()
    normalized = centered / std.detach().clamp_min(1e-4)
    corr = normalized.T @ normalized / max(coords.shape[0] - 1, 1)
    offdiag = corr - torch.diag_embed(torch.diagonal(corr))
    denom = max(coords.shape[-1] * (coords.shape[-1] - 1), 1)
    correlation_loss = offdiag.square().sum() / denom
    scalar = standardized_returns - standardized_returns.mean()
    scalar = scalar / scalar.std(unbiased=False).detach().clamp_min(1e-4)
    scalar_correlation_loss = (
        (normalized * scalar.unsqueeze(-1)).mean(0).square().mean()
    )
    mean_loss = coords.mean(0).square().mean()
    total = variance_loss + correlation_loss + scalar_correlation_loss + mean_loss
    return (
        total,
        variance_loss,
        correlation_loss,
        scalar_correlation_loss,
        mean_loss,
    )


def compute_sampled_returns(
    rewards,
    next_values,
    terminations,
    boundaries,
    bootstrap_valids,
    gamma,
):
    """One reverse return sample, censored when a required final observation is missing."""
    sampled_returns = torch.zeros_like(rewards)
    sampled_valids = torch.zeros_like(rewards, dtype=torch.bool)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            future_return = next_values[t]
            future_valid = bootstrap_valids[t].bool()
        else:
            crosses_boundary = boundaries[t].bool()
            future_return = torch.where(
                crosses_boundary, next_values[t], sampled_returns[t + 1]
            )
            future_valid = torch.where(
                crosses_boundary,
                bootstrap_valids[t].bool(),
                sampled_valids[t + 1],
            )
        current_valid = torch.logical_or(
            terminations[t].bool(),
            torch.logical_and(bootstrap_valids[t].bool(), future_valid),
        )
        sampled_returns[t] = rewards[t] + gamma * (1.0 - terminations[t]) * future_return
        sampled_valids[t] = current_valid
    return sampled_returns, sampled_valids


def effective_rank(x):
    centered = x - x.mean(0, keepdim=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    eig = eig[eig > 1e-10]
    if not eig.numel():
        return 0.0
    weights = eig / eig.sum().clamp_min(1e-12)
    return float(torch.exp(-(weights * (weights + 1e-12).log()).sum()))


@torch.no_grad()
def protect_auxiliary_trunk_gradient(
    protected_grads,
    auxiliary_grads,
    trunk_parameters,
    cap_reference_grads,
    max_ratio,
):
    """Remove the protected-gradient span, then cap the auxiliary trunk norm."""
    device = trunk_parameters[0].device
    reference_sq = sum(
        (
            gradient.double().square().sum()
            for gradient in cap_reference_grads.values()
        ),
        start=torch.zeros((), device=device, dtype=torch.float64),
    )
    auxiliary_sq = sum(
        (
            auxiliary_grads[parameter].double().square().sum()
            for parameter in trunk_parameters
            if parameter in auxiliary_grads
        ),
        start=torch.zeros((), device=device, dtype=torch.float64),
    )
    cosines = []
    orthogonal_basis = []
    for protected in protected_grads:
        protected_sq = sum(
            (gradient.double().square().sum() for gradient in protected.values()),
            start=torch.zeros((), device=device, dtype=torch.float64),
        )
        protected_dot = sum(
            (
                auxiliary_grads[parameter]
                .double()
                .mul(protected[parameter].double())
                .sum()
                for parameter in trunk_parameters
                if parameter in auxiliary_grads and parameter in protected
            ),
            start=torch.zeros((), device=device, dtype=torch.float64),
        )
        cosines.append(
            protected_dot
            / (protected_sq * auxiliary_sq).sqrt().clamp_min(1e-20)
        )
        basis = {
            parameter: gradient.detach().clone()
            for parameter, gradient in protected.items()
        }
        for previous, previous_sq in orthogonal_basis:
            coefficient = sum(
                (
                    basis[parameter]
                    .double()
                    .mul(previous[parameter].double())
                    .sum()
                    for parameter in trunk_parameters
                    if parameter in basis and parameter in previous
                ),
                start=torch.zeros((), device=device, dtype=torch.float64),
            ) / previous_sq.clamp_min(1e-20)
            for parameter in trunk_parameters:
                if parameter in basis and parameter in previous:
                    basis[parameter].sub_(
                        previous[parameter] * coefficient.to(basis[parameter].dtype)
                    )
        basis_sq = sum(
            (gradient.double().square().sum() for gradient in basis.values()),
            start=torch.zeros((), device=device, dtype=torch.float64),
        )
        coefficient = sum(
            (
                auxiliary_grads[parameter]
                .double()
                .mul(basis[parameter].double())
                .sum()
                for parameter in trunk_parameters
                if parameter in auxiliary_grads and parameter in basis
            ),
            start=torch.zeros((), device=device, dtype=torch.float64),
        ) / basis_sq.clamp_min(1e-20)
        for parameter in trunk_parameters:
            if parameter in auxiliary_grads and parameter in basis:
                auxiliary_grads[parameter].sub_(
                    basis[parameter]
                    * coefficient.to(basis[parameter].dtype)
                )
        orthogonal_basis.append((basis, basis_sq))
    projected_sq = sum(
        (
            auxiliary_grads[parameter].double().square().sum()
            for parameter in trunk_parameters
            if parameter in auxiliary_grads
        ),
        start=torch.zeros((), device=device, dtype=torch.float64),
    )
    cap = max_ratio * reference_sq.sqrt()
    scale = torch.clamp(
        cap / projected_sq.sqrt().clamp_min(1e-20), max=1.0
    )
    for parameter in trunk_parameters:
        if parameter in auxiliary_grads:
            auxiliary_grads[parameter].mul_(
                scale.to(auxiliary_grads[parameter].dtype)
            )
    retained = scale * projected_sq.sqrt() / auxiliary_sq.sqrt().clamp_min(1e-20)
    return cosines, retained


def compute_scalar_gae(
    rewards,
    values,
    next_values,
    terminations,
    boundaries,
    bootstrap_valids,
    gamma,
    gae_lambda,
):
    """Standard scalar GAE with correct Gymnasium termination/truncation semantics."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        valid_target = torch.logical_or(
            terminations[t].bool(), bootstrap_valids[t].bool()
        )
        # A truncation has a valid terminal observation and therefore bootstraps, while a
        # true MDP termination does not. Both boundaries cut the trace across the reset.
        bootstrap_nonterminal = (1.0 - terminations[t]) * bootstrap_valids[t]
        lambda_nonterminal = 1.0 - boundaries[t]
        delta = rewards[t] + gamma * next_values[t] * bootstrap_nonterminal - values[t]
        candidate = delta + gamma * gae_lambda * lambda_nonterminal * lastgaelam
        # A time-limit boundary without final_observation has no valid scalar target.
        # Zero its advantage and trace contribution instead of treating it as terminal.
        lastgaelam = torch.where(valid_target, candidate, torch.zeros_like(candidate))
        advantages[t] = lastgaelam
    return advantages, advantages + values


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # Initialize the treatment privately, then consume exactly the public CPU RNG
        # v9's 6 x 62 SF head consumed. Actor/trunk initialization remains bit-paired.
        reference_sf_dim = 32 + obs_dim + 2 * act_dim + 1
        with torch.random.fork_rng(devices=[]):
            self.scalar_value_head = layer_init(nn.Linear(H, 1), std=0.1)
            self.rich_value_head = layer_init(
                nn.Linear(H, args.value_latent_dim - 1, bias=False), std=0.1
            )
            with torch.no_grad():
                self.scalar_value_head.weight.zero_()
                self.scalar_value_head.bias.zero_()
                self.rich_value_head.weight.zero_()
            direction = torch.randn(args.value_latent_dim)
            direction = direction / direction.norm()
            null_basis = torch.linalg.svd(
                direction.unsqueeze(0), full_matrices=True
            ).Vh[1:].T.contiguous()
            self.register_buffer("decoder_direction", direction)
            self.register_buffer("decoder_null_basis", null_basis)
            self.register_buffer("value_mean", torch.tensor(0.0))
            self.register_buffer("value_std", torch.tensor(1.0))
        _rng_advance_only = layer_init(
            nn.Linear(H, 6 * reference_sf_dim, bias=False), std=0.1
        )
        del _rng_advance_only
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        _, critic_feat = self._trunks(x)
        standardized_value = self.scalar_value_head(critic_feat).squeeze(-1)
        return self.value_mean + self.value_std * standardized_value

    def get_value_latent(self, x, rich_trunk_enabled=True):
        _, critic_feat = self._trunks(x)
        scalar_value = self.scalar_value_head(critic_feat).squeeze(-1)
        rich_feat = critic_feat if rich_trunk_enabled else critic_feat.detach()
        rich_value = self.rich_value_head(rich_feat)
        return self.compose_value_latent(scalar_value, rich_value)

    def compose_value_latent(self, scalar_value, rich_value):
        return (
            scalar_value.unsqueeze(-1) * self.decoder_direction
            + rich_value @ self.decoder_null_basis.T
        )

    def decode_value(self, latent):
        return self.value_mean + self.value_std * (latent @ self.decoder_direction)

    @torch.no_grad()
    def update_value_stats(self, batch_mean, batch_std, rate):
        """Update the scalar gauge without changing any raw value prediction."""
        old_mean = self.value_mean.clone()
        old_std = self.value_std.clone()
        new_mean = torch.lerp(old_mean, batch_mean, rate)
        new_std = torch.lerp(old_std, batch_std, rate).clamp_min(1e-6)
        scale = old_std / new_std
        self.scalar_value_head.weight.mul_(scale)
        self.scalar_value_head.bias.copy_(
            (old_std * self.scalar_value_head.bias + old_mean - new_mean)
            / new_std
        )
        self.value_mean.copy_(new_mean)
        self.value_std.copy_(new_std)

    def get_action_and_value(self, x, z=None, rich_trunk_enabled=True):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        standardized_value = self.scalar_value_head(critic_feat).squeeze(-1)
        value = self.value_mean + self.value_std * standardized_value
        rich_feat = critic_feat if rich_trunk_enabled else critic_feat.detach()
        rich_value = self.rich_value_head(rich_feat)
        value_latent = self.compose_value_latent(standardized_value, rich_value)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value, value_latent

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return (
            list(trunk.parameters())
            + list(self.scalar_value_head.parameters())
            + list(self.rich_value_head.parameters())
        )

    def scalar_value_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.scalar_value_head.parameters())

    def rich_value_parameters(self, include_trunk=True):
        parameters = list(self.rich_value_head.parameters())
        if include_trunk:
            trunk = self.trunk if self.share_backbone else self.critic_trunk
            parameters = list(trunk.parameters()) + parameters
        return parameters

    def shared_trunk_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters())


def shape_advantage(gae, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has only a linear scalar readout. Feeding them a constant
    sigma=1 / u=0 placeholder would
    make cdf_probit return the same constant for every sample (a zero policy gradient
    after norm_adv) while logging a perfectly healthy-looking curve, so the branches are
    deleted and the arg is rejected at startup instead.
    """
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
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
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Fail loud at startup rather than degrading silently mid-run.
    if args.adv_transform in ("tanh_std", "cdf_probit"):
        raise ValueError(
            f"adv_transform={args.adv_transform!r} reads the critic's per-state value "
            "DISTRIBUTION, which this latent critic's linear readout does not have. "
            "Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    if args.value_latent_dim < 2:
        raise ValueError("value_latent_dim must leave at least one learned nullspace direction")
    if not 0 < args.rich_holdout_envs < args.num_envs:
        raise ValueError("rich_holdout_envs must be between zero and num_envs")
    if not args.separate_grad_clip:
        raise ValueError("v16 requires separate_grad_clip for protected rich gradients")

    agent = Agent(envs, args).to(device)
    with torch.random.fork_rng(devices=[]):
        return_encoder = ReturnKernelEncoder(
            args.value_latent_dim - 1, args.return_hidden
        ).to(device)
    target_return_encoder = deepcopy(return_encoder).requires_grad_(False)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    return_optimizer = optim.AdamW(
        return_encoder.parameters(), lr=args.return_model_lr, weight_decay=1e-3
    )
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    scalar_value_params = agent.scalar_value_parameters()
    rich_head_params = agent.rich_value_parameters(include_trunk=False)
    shared_trunk_params = agent.shared_trunk_parameters()

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    if args.compile:
        # retain_graph in the dual-backward is incompatible with donated buffers.
        import torch._functorch.config
        import torch._dynamo

        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.suppress_errors = True   # never let compile break the run
        torch._dynamo.config.recompile_limit = 64     # trunk alone sees 4 (shape, grad) combos
        if agent.share_backbone:
            agent.trunk = CompiledModule(agent.trunk, cudagraphs=False)
        else:
            agent.actor_trunk = CompiledModule(agent.actor_trunk, cudagraphs=False)
            agent.critic_trunk = CompiledModule(agent.critic_trunk, cudagraphs=False)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0
    value_stat_count = 0
    return_stat_count = 0
    return_mean = torch.zeros((), device=device)
    return_std = torch.ones((), device=device)
    rich_gate_count = 0
    rich_gate_fail_count = 0
    rich_trunk_enabled = False

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            return_optimizer.param_groups[0]["lr"] = frac * args.return_model_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            flat_obs_shape = (-1,) + envs.single_observation_space.shape
            next_transition_values = agent.get_value(
                next_obses.reshape(flat_obs_shape)
            ).reshape(args.num_steps, args.num_envs)
            advantages, returns = compute_scalar_gae(
                rewards,
                values,
                next_transition_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )

            if auto_alpha:
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
                soft_next_values = next_transition_values + next_value_bonus
                policy_adv, _ = compute_scalar_gae(
                    rewards,
                    values,
                    soft_next_values,
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                    args.gamma,
                    args.gae_lambda,
                )
            else:
                next_value_bonus = None
                policy_adv = advantages

            # Common unbootstrapped value gate used by the SF lineage. Availability
            # removes both rollout-tail and reset-boundary bias from the comparison.
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            if mc_mask.sum() >= 256:
                mc_target = mc_ret.reshape(-1)[mc_mask]
                mc_prediction = values.reshape(-1)[mc_mask]
                mc_var = mc_target.var().clamp_min(1e-12)
                unrigged_mc_ev = float(
                    1.0 - (mc_target - mc_prediction).var() / mc_var
                )
            else:
                unrigged_mc_ev = float("nan")

            scalar_valid = torch.logical_or(
                transition_terminations.bool(), transition_valids.bool()
            )
            sampled_returns, sampled_valid = compute_sampled_returns(
                rewards,
                next_transition_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
            )
            # V13-style O(1) scalar gauge. The first update exactly adopts the first
            # rollout's target statistics; later updates become a slow EMA.
            valid_scalar_returns = returns[scalar_valid]
            batch_value_mean = valid_scalar_returns.mean()
            batch_value_std = valid_scalar_returns.std().clamp_min(1e-6)
            value_stat_count += 1
            value_stat_rate = max(args.value_stat_ema, 1.0 / value_stat_count)
            agent.update_value_stats(
                batch_value_mean, batch_value_std, value_stat_rate
            )
            scalar_target = (returns - agent.value_mean) / agent.value_std
            # The full recursive sample has a substantially broader transient
            # distribution than the GAE target. Give H its own adaptive input
            # gauge so the squared encoder never sees avoidable extreme inputs.
            valid_sampled_returns = sampled_returns[sampled_valid]
            batch_return_mean = valid_sampled_returns.mean()
            batch_return_std = valid_sampled_returns.std().clamp_min(1e-6)
            return_stat_count += 1
            return_stat_rate = max(args.return_stat_ema, 1.0 / return_stat_count)
            return_mean.lerp_(batch_return_mean, return_stat_rate)
            return_std.lerp_(batch_return_std, return_stat_rate)
            standardized_sampled_return = (
                sampled_returns - return_mean
            ) / return_std
            rich_target = target_return_encoder(
                standardized_sampled_return.reshape(-1)
            ).reshape(
                args.num_steps,
                args.num_envs,
                args.value_latent_dim - 1,
            )

            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)[scalar_valid.reshape(-1)]
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_scalar_target = scalar_target.reshape(-1)
        b_rich_target = rich_target.reshape(-1, args.value_latent_dim - 1)
        b_scalar_valid = scalar_valid.reshape(-1)
        b_sampled_returns = sampled_returns.reshape(-1)
        b_standardized_sampled_return = standardized_sampled_return.reshape(-1)
        b_rich_valid = sampled_valid.reshape(-1)
        sample_index = torch.arange(args.batch_size, device=device)
        # Flattening is time-major, so modulo num_envs identifies a complete vector
        # environment stream. Hold out whole streams to avoid adjacent-transition
        # leakage in the representation gate.
        environment_index = sample_index.remainder(args.num_envs)
        b_rich_holdout = b_rich_valid & (
            environment_index < args.rich_holdout_envs
        )
        b_rich_train = b_rich_valid & ~b_rich_holdout
        b_policy_valid = b_scalar_valid
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = torch.zeros_like(gae)
        b_policy_adv[b_policy_valid] = shape_advantage(
            gae[b_policy_valid], args, device
        )
        if args.norm_adv and args.norm_adv_scope == "batch":
            valid_adv = b_policy_adv[b_policy_valid]
            b_policy_adv_normed = torch.zeros_like(b_policy_adv)
            b_policy_adv_normed[b_policy_valid] = (
                valid_adv - valid_adv.mean()
            ) / (valid_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = torch.zeros_like(b_policy_adv)
            b_policy_adv_normed[b_policy_valid] = b_policy_adv[b_policy_valid] / b_returns[
                b_policy_valid
            ].std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        scalar_grad_norms = []
        rich_grad_norms = []
        actor_grad_norms = []
        rich_scalar_cosines = []
        rich_actor_cosines = []
        rich_trunk_retained = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_valid = b_policy_valid[mb_inds]
                mb_valid_weight = mb_valid.to(b_returns.dtype)
                mb_valid_denom = mb_valid_weight.sum().clamp_min(1.0)

                _, _, newlogprob, entropy, newvalue, new_value_latent = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_latent_zs[mb_inds],
                    rich_trunk_enabled=rich_trunk_enabled,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = torch.zeros_like(b_advantages[mb_inds])
                    mb_raw_adv[mb_valid] = shape_advantage(
                        b_advantages[mb_inds][mb_valid], args, device
                    )
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds][
                            mb_valid
                        ].std().clamp(min=args.ret_perc_floor)
                    else:
                        valid_adv = mb_advantages[mb_valid]
                        normalized = (valid_adv - valid_adv.mean()) / (
                            valid_adv.std() + 1e-8
                        )
                        mb_advantages = torch.zeros_like(mb_advantages)
                        mb_advantages[mb_valid] = normalized
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds][mb_valid]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = (
                    torch.max(pg_loss1, pg_loss2) * mb_valid_weight
                ).sum() / mb_valid_denom

                scalar_valid = b_scalar_valid[mb_inds]
                scalar_weight = scalar_valid.to(newvalue.dtype)
                scalar_denom = scalar_weight.sum().clamp_min(1.0)
                scalar_prediction = new_value_latent @ agent.decoder_direction
                scalar_error = (
                    scalar_prediction - b_scalar_target[mb_inds]
                ).square()
                scalar_value_loss = (
                    scalar_error * scalar_weight
                ).sum() / scalar_denom

                rich_valid = b_rich_train[mb_inds]
                rich_weight = rich_valid.to(newvalue.dtype)
                rich_denom = rich_weight.sum().clamp_min(1.0)
                rich_prediction = (
                    new_value_latent @ agent.decoder_null_basis
                )
                rich_error = (
                    rich_prediction - b_rich_target[mb_inds]
                ).square().mean(-1)
                rich_value_loss = (
                    rich_error * rich_weight
                ).sum() / rich_denom
                # Preserve the scalar critic's full baseline weight. Rich supervision is
                # additive; toggling it off does not silently halve scalar learning.
                v_loss = scalar_value_loss + args.rich_value_coef * rich_value_loss

                entropy_loss = (entropy * mb_valid_weight).sum() / mb_valid_denom

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_per_sample = -log_alpha.exp() * (
                        newlogprob.detach() + target_entropy
                    )
                    alpha_loss = (
                        alpha_per_sample * mb_valid_weight
                    ).sum() / mb_valid_denom
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                if args.separate_grad_clip:
                    # Three independent backward passes protect the scalar critic and
                    # policy. The rich head always learns; its shared-trunk component
                    # is admitted only by the held-out gate, projected against a
                    # conflicting primary gradient, then norm-capped.
                    optimizer.zero_grad(set_to_none=True)
                    (
                        args.vf_coef
                        * args.critic_loss_scale
                        * scalar_value_loss
                    ).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(
                        scalar_value_params, args.critic_grad_clip
                    )
                    scalar_grads = {
                        p: p.grad.detach().clone()
                        for p in scalar_value_params
                        if p.grad is not None
                    }

                    optimizer.zero_grad(set_to_none=True)
                    (
                        args.vf_coef
                        * args.critic_loss_scale
                        * args.rich_value_coef
                        * rich_value_loss
                    ).backward(retain_graph=True)
                    rich_params = agent.rich_value_parameters(
                        include_trunk=rich_trunk_enabled
                    )
                    rich_gn = nn.utils.clip_grad_norm_(
                        rich_params, args.rich_grad_clip
                    )
                    rich_grads = {
                        p: p.grad.detach().clone()
                        for p in rich_params
                        if p.grad is not None
                    }

                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(
                        actor_params, args.actor_grad_clip
                    )

                    if rich_trunk_enabled:
                        # Primary is exactly what would update the trunk without the
                        # auxiliary. Remove only an opposing rich component, then cap
                        # its remaining norm to a fraction of the primary norm.
                        scalar_trunk_grads = {
                            p: scalar_grads[p]
                            for p in shared_trunk_params
                            if p in scalar_grads
                        }
                        actor_trunk_grads = {
                            p: p.grad.detach().clone()
                            for p in shared_trunk_params
                            if p.grad is not None
                        }
                        primary = {}
                        for p in shared_trunk_params:
                            actor_g = actor_trunk_grads.get(p)
                            scalar_g = scalar_trunk_grads.get(p)
                            if actor_g is not None or scalar_g is not None:
                                primary[p] = (
                                    torch.zeros_like(p)
                                    if actor_g is None else actor_g.detach().clone()
                                )
                                if scalar_g is not None:
                                    primary[p].add_(scalar_g)
                        cosines, retained = protect_auxiliary_trunk_gradient(
                            [scalar_trunk_grads, actor_trunk_grads],
                            rich_grads,
                            shared_trunk_params,
                            primary,
                            args.rich_trunk_ratio,
                        )
                        rich_scalar_cosines.append(cosines[0])
                        rich_actor_cosines.append(cosines[1])
                        rich_trunk_retained.append(retained)

                    # Policy grads are currently live. Add the protected scalar
                    # gradients, all rich-head gradients, and only the gated rich
                    # trunk gradients.
                    for p, g in scalar_grads.items():
                        p.grad = g if p.grad is None else p.grad + g
                    for p, g in rich_grads.items():
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                    scalar_grad_norms.append(critic_gn.detach())
                    rich_grad_norms.append(rich_gn.detach())
                    actor_grad_norms.append(actor_gn.detach())
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    rich_gn = critic_gn

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Learn H(G) after PPO, against the detached state predictor. Noncollapse
        # pressure acts only on this small encoder and therefore cannot spend any
        # actor/scalar gradient budget. Whole environment streams remain held out.
        with torch.no_grad():
            state_rich_for_model = (
                agent.get_value_latent(b_obs, rich_trunk_enabled=True)
                @ agent.decoder_null_basis
            )
        return_model_loss_sum = return_reg_loss_sum = return_grad_norm_sum = 0.0
        return_var_loss_sum = return_corr_loss_sum = 0.0
        return_scalar_corr_loss_sum = return_mean_loss_sum = 0.0
        return_model_steps = 0
        model_valid_inds = torch.nonzero(b_rich_train, as_tuple=False).squeeze(-1)
        for _ in range(args.return_model_epochs):
            perm = model_valid_inds[torch.randperm(model_valid_inds.numel(), device=device)]
            for start in range(0, perm.numel(), args.return_model_batch):
                idx = perm[start : start + args.return_model_batch]
                online_code = return_encoder(
                    b_standardized_sampled_return[idx]
                )
                return_model_loss = (
                    online_code - state_rich_for_model[idx]
                ).square().mean()
                (
                    return_reg_loss,
                    return_var_loss,
                    return_corr_loss,
                    return_scalar_corr_loss,
                    return_mean_loss,
                ) = return_code_regularization(
                    online_code,
                    b_standardized_sampled_return[idx],
                )
                return_loss = (
                    args.return_consistency_coef * return_model_loss
                    + args.return_feature_reg_coef * return_reg_loss
                )
                return_optimizer.zero_grad(set_to_none=True)
                return_loss.backward()
                return_grad_norm = nn.utils.clip_grad_norm_(
                    return_encoder.parameters(), args.return_model_grad_clip
                )
                return_optimizer.step()
                return_model_loss_sum += return_model_loss.item()
                return_reg_loss_sum += return_reg_loss.item()
                return_var_loss_sum += return_var_loss.item()
                return_corr_loss_sum += return_corr_loss.item()
                return_scalar_corr_loss_sum += return_scalar_corr_loss.item()
                return_mean_loss_sum += return_mean_loss.item()
                return_grad_norm_sum += float(return_grad_norm)
                return_model_steps += 1
        denominator = max(return_model_steps, 1)
        return_model_loss_mean = return_model_loss_sum / denominator
        return_reg_loss_mean = return_reg_loss_sum / denominator
        return_var_loss_mean = return_var_loss_sum / denominator
        return_corr_loss_mean = return_corr_loss_sum / denominator
        return_scalar_corr_loss_mean = return_scalar_corr_loss_sum / denominator
        return_mean_loss_mean = return_mean_loss_sum / denominator
        return_grad_norm_mean = return_grad_norm_sum / denominator

        # Advance the target frame only after this rollout's frozen targets have
        # finished being consumed.
        ema_update(
            target_return_encoder,
            return_encoder,
            args.return_encoder_ema_rate,
        )

        with torch.no_grad():
            post_target = target_return_encoder(b_standardized_sampled_return)
            post_prediction = (
                agent.get_value_latent(b_obs, rich_trunk_enabled=True)
                @ agent.decoder_null_basis
            )
            # Admission is measured against the exact frozen frame PPO consumed,
            # not the easier post-update EMA frame.
            held_target = b_rich_target[b_rich_holdout]
            held_prediction = post_prediction[b_rich_holdout]
            target_variance = held_target.var(0, unbiased=False).clamp_min(1e-8)
            coordinate_ev = 1.0 - (
                held_target - held_prediction
            ).square().mean(0) / target_variance
            heldout_rich_ev = float(coordinate_ev.mean())
            heldout_rich_best_ev = float(coordinate_ev.max())
            heldout_normalized_bias = float(
                (
                    (held_prediction.mean(0) - held_target.mean(0)).square()
                    / target_variance
                ).mean()
            )
            target_effective_rank = effective_rank(held_target)
            prediction_effective_rank = effective_rank(held_prediction)
            target_feature_std = float(held_target.std(0).mean())
            prediction_feature_std = float(held_prediction.std(0).mean())
            target_mean_abs = float(held_target.mean(0).abs().mean())
            encoder_frame_rmse = float(
                (
                    post_target[b_rich_holdout] - held_target
                ).square().mean().sqrt()
            )

        gate_passes = (
            heldout_rich_ev > args.rich_gate_ev
            and target_effective_rank > args.rich_gate_rank
            and prediction_effective_rank > args.rich_gate_prediction_rank
            and target_feature_std > args.rich_gate_min_target_std
            and prediction_feature_std > args.rich_gate_min_prediction_std
        )
        if rich_trunk_enabled:
            rich_gate_fail_count = 0 if gate_passes else rich_gate_fail_count + 1
            if rich_gate_fail_count >= args.rich_gate_fail_patience:
                rich_trunk_enabled = False
                rich_gate_count = 0
        else:
            rich_gate_count = rich_gate_count + 1 if gate_passes else 0
            rich_gate_fail_count = 0
            if rich_gate_count >= args.rich_gate_patience:
                rich_trunk_enabled = True

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/scalar_value_loss", scalar_value_loss.item(), global_step)
        writer.add_scalar("losses/rich_value_loss", rich_value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("gate/unrigged_mc_ev", unrigged_mc_ev, global_step)
        writer.add_scalar(
            "losses/actor_grad_norm",
            float(torch.stack(actor_grad_norms).mean())
            if actor_grad_norms
            else float(actor_gn),
            global_step,
        )
        writer.add_scalar(
            "losses/critic_grad_norm",
            float(torch.stack(scalar_grad_norms).mean())
            if scalar_grad_norms
            else float(critic_gn),
            global_step,
        )
        writer.add_scalar(
            "losses/rich_grad_norm",
            float(torch.stack(rich_grad_norms).mean())
            if rich_grad_norms
            else float(rich_gn),
            global_step,
        )
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/value_target_mean", agent.value_mean.item(), global_step)
        writer.add_scalar("debug/value_target_std", agent.value_std.item(), global_step)
        writer.add_scalar("debug/sampled_return_mean", return_mean.item(), global_step)
        writer.add_scalar("debug/sampled_return_std", return_std.item(), global_step)
        writer.add_scalar(
            "debug/standardized_sampled_return_absmax",
            b_standardized_sampled_return[b_rich_valid].abs().max().item(),
            global_step,
        )
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        writer.add_scalar("rich_value/return_model_loss", return_model_loss_mean, global_step)
        writer.add_scalar("rich_value/return_regularization_loss", return_reg_loss_mean, global_step)
        writer.add_scalar("rich_value/return_variance_loss", return_var_loss_mean, global_step)
        writer.add_scalar("rich_value/return_correlation_loss", return_corr_loss_mean, global_step)
        writer.add_scalar(
            "rich_value/return_scalar_correlation_loss",
            return_scalar_corr_loss_mean,
            global_step,
        )
        writer.add_scalar("rich_value/return_mean_loss", return_mean_loss_mean, global_step)
        writer.add_scalar("rich_value/return_model_grad_norm", return_grad_norm_mean, global_step)
        writer.add_scalar("rich_value/return_model_steps", return_model_steps, global_step)
        writer.add_scalar("rich_value/heldout_ev", heldout_rich_ev, global_step)
        writer.add_scalar("rich_value/heldout_best_ev", heldout_rich_best_ev, global_step)
        writer.add_scalar(
            "rich_value/heldout_normalized_bias",
            heldout_normalized_bias,
            global_step,
        )
        writer.add_scalar("rich_value/target_effective_rank", target_effective_rank, global_step)
        writer.add_scalar(
            "rich_value/prediction_effective_rank",
            prediction_effective_rank,
            global_step,
        )
        writer.add_scalar("rich_value/target_feature_std", target_feature_std, global_step)
        writer.add_scalar(
            "rich_value/prediction_feature_std",
            prediction_feature_std,
            global_step,
        )
        writer.add_scalar("rich_value/target_mean_abs", target_mean_abs, global_step)
        writer.add_scalar("rich_value/encoder_frame_rmse", encoder_frame_rmse, global_step)
        writer.add_scalar("rich_value/trunk_gate_count", rich_gate_count, global_step)
        writer.add_scalar(
            "rich_value/trunk_gate_fail_count", rich_gate_fail_count, global_step
        )
        writer.add_scalar(
            "rich_value/trunk_enabled", float(rich_trunk_enabled), global_step
        )
        writer.add_scalar(
            "rich_value/rich_scalar_cosine",
            float(torch.stack(rich_scalar_cosines).mean())
            if rich_scalar_cosines
            else 0.0,
            global_step,
        )
        writer.add_scalar(
            "rich_value/rich_actor_cosine",
            float(torch.stack(rich_actor_cosines).mean())
            if rich_actor_cosines
            else 0.0,
            global_step,
        )
        writer.add_scalar(
            "rich_value/rich_trunk_retained",
            float(torch.stack(rich_trunk_retained).mean())
            if rich_trunk_retained
            else 0.0,
            global_step,
        )
        writer.add_scalar(
            "rich_value/valid_fraction", b_rich_valid.float().mean().item(), global_step
        )
        writer.add_scalar(
            "rich_value/holdout_fraction",
            b_rich_holdout.float().mean().item(),
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
