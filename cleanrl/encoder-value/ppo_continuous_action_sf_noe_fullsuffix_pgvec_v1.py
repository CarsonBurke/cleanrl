# Full-Suffix Successor-Residual Per-Actuator PPO v1.
#
# Arm A deliberately removes learned trace selection and every moving target statistic.
# The critic predicts raw successor features phi=[s,a,a^2,1]. Its target and the policy's
# common credit are the complete available lambda=1 suffix, with a bootstrap only at a
# rollout edge or time-limit truncation and a hard cut at resets. A discounted reward-probe
# residual is appended to the policy vector, so its fixed covector projects exactly to the
# corresponding scalar Monte-Carlo residual; scalar GAE is logged only as a diagnostic.
#
# A one-rollout-lagged reward covector freezes the policy/value coordinate frame. A closed-form
# actuator router is fit across independent environment folds; source-only out-of-fold skill
# and destination prediction agreement jointly gate its zero-sum corrections. Those corrections
# enter marginal Beta PPO ratios, so the vector residual changes policy optimization rather than
# merely reconstructing scalar GAE. `--no-pgvec` is the matched joint-PPO control. No learned
# encoder, Q-function, EMA, contrastive objective, or PopArt-style output rescaling is present.
# =====================================================================================
import os
import random
import time
from dataclasses import dataclass
from math import log

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
    gae_lambda: float = 0.95  # diagnostics only; the actor and SF target always use lambda=1
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
    ret_perc_scope: str = "minibatch"
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional occupancy vector.
    critic_mtp_horizon: int = 6

    # --- successor-feature value pathway ---------------------------------------------
    # No encoder args: phi = [s, a, a*a, 1] and the whole SSL path is gone (see header).
    sf_alpha: float = 1.0            # 1.0 = pure occupancy prediction, ZERO scalar regression in the
    #                                  network. <1 mixes in MSE(w_r.psi, w_r.Lambda) -- the fallback
    #                                  if the pure form underfits, not the starting point.
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)
    pgvec: bool = True               # false = matched full-suffix joint-ratio PPO control
    pgvec_ridge: float = 1e-3
    pgvec_min_fold_samples: int = 256

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
    # So the trunk gets inductor fusion without graphs. (v2 also compiled the SSL nets with
    # reduce-overhead; there are none left here, so compile_mode is accepted and unused --
    # kept so the canonical mlq submit line is identical across the two runs.)
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # accepted but UNUSED (no SSL nets left to apply it to).
    #   Kept so the canonical submit line parses. NOTE parity is partial: --compile-ssl-cudagraphs
    #   was deleted with the SSL path, so that one v2 flag is a tyro error here.

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


def phi_features(obs, action):
    """phi = [s, a, a*a, 1] -- v2's basis with the learned-embedding block removed.

    The raw block is the already-NormalizeObservation-scaled state, which is what makes
    HalfCheetah's velocity term exactly linearly recoverable by w_r (v1, which had only
    the embedding, measured a lag-1 residual autocorrelation of 0.474 -- structured error
    injected into every value estimate; v2 with s present measured 0.254 at R^2 0.988).

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on s alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so s has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = obs[..., :1].new_ones(obs.shape[:-1] + (1,))
    return torch.cat([obs, action, action * action, ones], dim=-1)


def solve_reward_probe(phi, reward, ridge):
    """Closed-form ridge solve of w_r from phi -> immediate reward.

    Solved, not gradient-fit: it removes a learning rate, removes the cold-start phase
    where a random V would drive the actor, and makes "thin, fast, stationary readout"
    literal -- w_r is recomputed optimally every iteration. Done in float64; the normal
    equations square the condition number and phi's blocks have very different scales.
    """
    phi64 = phi.double()
    gram = phi64.T @ phi64
    rhs = phi64.T @ reward.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(phi.dtype)


def full_suffix_vector_residual(
    delta,
    continuation,
    gamma,
):
    """Complete available discounted suffix, cut only at reset boundaries."""
    if delta.shape[:2] != continuation.shape:
        raise ValueError("delta and continuation time/environment axes must match")
    residual = torch.zeros_like(delta)
    running = torch.zeros_like(delta[0])
    for t in reversed(range(delta.shape[0])):
        running = delta[t] + gamma * continuation[t].unsqueeze(-1) * running
        residual[t] = running
    return residual


def full_suffix_sf_credit(
    phi,
    psi,
    psi_next,
    terminations,
    transition_valids,
    boundaries,
    gamma,
):
    """Vector TD suffix with terminal cuts and time-limit bootstraps."""
    bootstrap = (1.0 - terminations) * transition_valids
    continuation = 1.0 - boundaries
    delta = phi + gamma * bootstrap.unsqueeze(-1) * psi_next - psi
    residual = full_suffix_vector_residual(delta, continuation, gamma)
    return residual, psi + residual


def project_policy_credit(sf_residual, reward_residual, reward_covector):
    """Append exact probe error and contract only after vector credit is built."""
    vector_residual = torch.cat(
        [sf_residual, reward_residual.unsqueeze(-1)], dim=-1
    )
    extended_covector = torch.cat(
        [reward_covector, reward_covector.new_ones(1)]
    )
    return vector_residual, vector_residual @ extended_covector


def standardize_policy_credit(policy_credit):
    """The actor's only scalarization path; diagnostic GAE is deliberately absent."""
    return (policy_credit - policy_credit.mean()) / (
        policy_credit.std() + 1e-8
    )


def pgvec_design(residual, residual_scale, standardized_action):
    normalized = residual / residual_scale
    return (
        standardized_action.unsqueeze(-1) * normalized.unsqueeze(1)
    ).reshape(residual.shape[0], -1)


def pgvec_solve(design, target, ridge):
    design64 = design.double()
    gram = design64.T @ design64
    scale = gram.diagonal().mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    return torch.linalg.solve(gram, design64.T @ target.double()).to(design.dtype)


def pgvec_logits(coefficients, residual, residual_scale, action_dim):
    return (residual / residual_scale) @ coefficients.view(action_dim, -1).T


def pgvec_rho(logits, standardized_action):
    contribution = logits * standardized_action
    return contribution - contribution.mean(-1, keepdim=True)


def per_actuator_clip_bounds(clip_low, clip_high, action_dim):
    """Marginal bounds under additive per-dimension KL/Fisher scaling."""
    exponent = 1.0 / np.sqrt(action_dim)
    return (1.0 - clip_low) ** exponent, (1.0 + clip_high) ** exponent


def per_actuator_ppo_loss(
    logratio_dim,
    common_advantage,
    redistribution,
    clip_low,
    clip_high,
):
    """Marginal PPO whose zero-redistribution gradient matches joint PPO at ratio one."""
    action_dim = logratio_dim.shape[-1]
    ratio_dim = logratio_dim.exp()
    lower_dim, upper_dim = per_actuator_clip_bounds(
        clip_low, clip_high, action_dim
    )
    advantage = common_advantage.unsqueeze(-1) + redistribution
    unclipped = -advantage * ratio_dim
    clipped = -advantage * ratio_dim.clamp(lower_dim, upper_dim)
    return torch.maximum(unclipped, clipped).sum(-1).mean()


def joint_ppo_loss(logratio, advantage, clip_low, clip_high):
    ratio = logratio.exp()
    unclipped = -advantage * ratio
    clipped = -advantage * ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    return torch.maximum(unclipped, clipped).mean()


def correlation(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (
        a.std().clamp_min(1e-12) * b.std().clamp_min(1e-12)
    )


def cross_fitted_pgvec(
    vector_residual,
    standardized_action,
    target,
    valid,
    ridge,
    min_fold_samples,
):
    """Fit each environment parity from the other parity with fresh reliability.

    The source parity is divided by environment index modulo four. Two independent
    source fits must predict one another's targets before their predictions are
    compared on the destination environments. Destination targets never affect the
    fitted router or its gate.
    """
    time_steps, num_envs, vector_dim = vector_residual.shape
    action_dim = standardized_action.shape[-1]
    if standardized_action.shape[:2] != (time_steps, num_envs):
        raise ValueError("action and residual time/environment axes must match")
    if target.shape != (time_steps, num_envs) or valid.shape != target.shape:
        raise ValueError("target and validity mask must have shape (T,B)")

    residual_flat = vector_residual.reshape(-1, vector_dim)
    action_flat = standardized_action.reshape(-1, action_dim)
    target_flat = target.reshape(-1)
    valid_flat = valid.reshape(-1)
    environment = torch.arange(num_envs, device=vector_residual.device)
    environment_flat = environment.view(1, -1).expand(time_steps, -1).reshape(-1)
    rho_flat = vector_residual.new_zeros(time_steps * num_envs, action_dim)
    reliabilities = []
    predictive_skills = []
    split_agreements = []
    raw_magnitudes = []
    coefficient_norms = []

    for destination_parity in (0, 1):
        destination = environment_flat.remainder(2).eq(destination_parity)
        source = ~destination
        source_group_a = source & environment_flat.remainder(4).eq(
            1 - destination_parity
        )
        source_group_b = source & ~source_group_a
        source_valid = source & valid_flat
        group_a_valid = source_group_a & valid_flat
        group_b_valid = source_group_b & valid_flat
        destination_valid = destination & valid_flat
        minimum = max(min_fold_samples, action_dim * vector_dim)
        if (
            group_a_valid.sum() < minimum
            or group_b_valid.sum() < minimum
            or destination_valid.sum() < min_fold_samples
        ):
            continue

        residual_scale = residual_flat[source_valid].std(0).clamp_min(1e-6)

        def solve(mask):
            target_scale = target_flat[mask].std().clamp_min(1e-6)
            design = pgvec_design(
                residual_flat[mask], residual_scale, action_flat[mask]
            )
            coefficients = pgvec_solve(
                design, target_flat[mask] / target_scale, ridge
            )
            return coefficients, target_scale

        coefficients_a, target_scale_a = solve(group_a_valid)
        coefficients_b, target_scale_b = solve(group_b_valid)
        coefficients, _ = solve(source_valid)

        def summed_contribution(coefficients_fold, mask):
            logits = pgvec_logits(
                coefficients_fold,
                residual_flat[mask],
                residual_scale,
                action_dim,
            )
            return (logits * action_flat[mask]).sum(-1)

        skill_a_to_b = correlation(
            summed_contribution(coefficients_a, group_b_valid),
            target_flat[group_b_valid] / target_scale_a,
        ).clamp(0.0, 1.0)
        skill_b_to_a = correlation(
            summed_contribution(coefficients_b, group_a_valid),
            target_flat[group_a_valid] / target_scale_b,
        ).clamp(0.0, 1.0)
        predictive_skill = 0.5 * (skill_a_to_b + skill_b_to_a)
        destination_residual = residual_flat[destination_valid]
        destination_action = action_flat[destination_valid]
        rho_a = pgvec_rho(
            pgvec_logits(
                coefficients_a,
                destination_residual,
                residual_scale,
                action_dim,
            ),
            destination_action,
        )
        rho_b = pgvec_rho(
            pgvec_logits(
                coefficients_b,
                destination_residual,
                residual_scale,
                action_dim,
            ),
            destination_action,
        )
        split_agreement = correlation(rho_a, rho_b).clamp(0.0, 1.0)
        reliability = predictive_skill * split_agreement
        destination_all = destination.nonzero(as_tuple=True)[0]
        rho_raw = pgvec_rho(
            pgvec_logits(
                coefficients,
                residual_flat[destination_all],
                residual_scale,
                action_dim,
            ),
            action_flat[destination_all],
        )
        rho_flat[destination_all] = reliability * rho_raw
        reliabilities.append(reliability)
        predictive_skills.append(predictive_skill)
        split_agreements.append(split_agreement)
        raw_magnitudes.append(rho_raw.abs().mean())
        coefficient_norms.append(coefficients.norm())

    zero = vector_residual.new_zeros(())
    reliability = torch.stack(reliabilities).mean() if reliabilities else zero
    predictive_skill = (
        torch.stack(predictive_skills).mean() if predictive_skills else zero
    )
    split_agreement = (
        torch.stack(split_agreements).mean() if split_agreements else zero
    )
    raw_magnitude = (
        torch.stack(raw_magnitudes).mean() if raw_magnitudes else zero
    )
    coefficient_norm = (
        torch.stack(coefficient_norms).mean() if coefficient_norms else zero
    )
    return (
        rho_flat.reshape(time_steps, num_envs, action_dim),
        reliability,
        predictive_skill,
        split_agreement,
        raw_magnitude,
        coefficient_norm,
        len(reliabilities),
    )


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


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
        # SUCCESSOR-FEATURE critic head. Same shape and MTP semantics as the HL-Gauss head
        # it replaces -- only the per-horizon target space changed, from 511 scalar-return
        # bucket logits to K = obs_dim + 2*act_dim + 1 occupancy dimensions.
        # Horizon h predicts Lambda_{t+h}, the vector-valued lambda-return of phi.
        # Zero-init is deliberate and not merely neutral: with psi_old == 0 the first
        # rollout's Lambda degenerates to the plain discounted sum of phi, i.e. a clean
        # Monte-Carlo target with no bootstrap from a random head.
        self.sf_dim = obs_dim + 2 * act_dim + 1  # [s, a, a*a, 1]
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * self.sf_dim, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
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
        # Returns successor features (B, mtp, sf_dim); horizon 0 is psi(s_t), whose
        # scalar readout w_r . psi_0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)

    def get_action_and_value(self, x, z=None):
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
        log_prob_dim = dist.log_prob(z) - log_det_fn(z)
        log_prob = log_prob_dim.sum(1)
        value_sf = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)
        standardized_action = (z - dist.mean) / dist.stddev.clamp_min(1e-6)
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
        return (
            action,
            z,
            log_prob,
            entropy,
            value_sf,
            log_prob_dim,
            standardized_action,
        )

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
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def shape_advantage(gae, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has no distributional critic to read -- the head
    predicts successor features. Feeding them a constant sigma=1 / u=0 placeholder would
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
    if (
        args.actor_dist != "beta"
        or not args.norm_adv
        or args.norm_adv_scope != "batch"
        or args.adv_transform != "v10"
        or args.ret_percnorm
        or args.sf_alpha != 1.0
    ):
        raise ValueError(
            "full-suffix pgvec requires Beta, batch advantage normalization, "
            "identity shaping, pure vector critic loss, and no percentile normalization"
        )
    if args.pgvec and (args.num_envs < 4 or args.num_envs % 4):
        raise ValueError("cross-fitted pgvec requires num_envs divisible by four")
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
            "DISTRIBUTION, which this variant does not have (the head predicts successor "
            "features, not bucket logits). Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # --- successor-feature value pathway ---------------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim

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

    # The head emits raw successor-feature units. A fresh rollout scale weights its
    # regression error but never changes the head's coordinate system.
    sf_center = torch.zeros(sf_dim, device=device)
    sf_scale = torch.ones(sf_dim, device=device)
    w_r = torch.zeros(sf_dim, device=device)

    def psi_raw(sf_prediction):
        return sf_prediction

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_dim = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    standardized_actions = torch.zeros(
        (args.num_steps, args.num_envs, act_dim)
    ).to(device)
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
    ret_perc_scale = 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                (
                    action,
                    z,
                    logprob,
                    ent,
                    value_sf,
                    logprob_dim,
                    standardized_action,
                ) = agent.get_action_and_value(next_obs)
                values[step] = psi_raw(value_sf[:, 0]) @ w_r
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            logprobs_dim[step] = logprob_dim
            standardized_actions[step] = standardized_action

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
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
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
            psi_next = psi_raw(agent.get_value(next_obses.reshape(flat_obs_shape))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            # Kept as trunk features, not just psi: the EV gate below needs the SAME
            # features to fit its detached scalar probe, and this variant already does two
            # full-batch trunk forwards per iteration where the base did one.
            critic_feat_buf = agent._trunks(obs.reshape(flat_obs_shape))[1]
            psi_cur = psi_raw(
                agent.critic_head(critic_feat_buf).view(-1, args.critic_mtp_horizon, sf_dim)[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)
            phi = phi_features(obs, actions)
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            # Freeze the coordinate frame used during collection. The current rollout
            # only solves the covector that the *next* rollout will consume.
            w_r_lagged = w_r
            reward_resid = flat_rew - flat_phi @ w_r_lagged
            reward_r2 = 1.0 - (reward_resid.var() / flat_rew.var().clamp_min(1e-12)).item()
            values = psi_cur @ w_r_lagged
            next_transition_values = psi_next @ w_r_lagged

            bootstrap = (
                (1.0 - transition_terminations) * transition_valids
            )
            continuation = 1.0 - transition_boundaries
            sf_residual, sf_target = full_suffix_sf_credit(
                phi,
                psi_cur,
                psi_next,
                transition_terminations,
                transition_valids,
                transition_boundaries,
                args.gamma,
            )

            reward_residual_tb = reward_resid.reshape(
                args.num_steps, args.num_envs
            )
            residual_trace = full_suffix_vector_residual(
                reward_residual_tb.unsqueeze(-1),
                continuation,
                args.gamma,
            ).squeeze(-1)
            policy_vector_residual, policy_adv = project_policy_credit(
                sf_residual,
                residual_trace,
                w_r_lagged,
            )

            # Scalar GAE remains a diagnostic and reporting target only.
            diagnostic_gae = torch.zeros_like(rewards)
            diagnostic_running = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                delta_reward = (
                    rewards[t]
                    + args.gamma
                    * bootstrap[t]
                    * next_transition_values[t]
                    - values[t]
                )
                diagnostic_running = (
                    delta_reward
                    + args.gamma
                    * args.gae_lambda
                    * continuation[t]
                    * diagnostic_running
                )
                diagnostic_gae[t] = diagnostic_running
            returns = diagnostic_gae + values

            # R^2 alone is the WRONG gate: r_t uses the frame-skip AVERAGED velocity while the
            # observation carries instantaneous qvel, so a structural residual of ~1-3% is
            # expected and harmless IF it is white. What costs EV is a gait-phase-locked,
            # speed-correlated residual, which shows up as autocorrelation, not as R^2.
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            reward_resid_ac = [
                ((resid_c[:-k] * resid_c[k:]).mean() / resid_var).item() for k in (1, 5, 10, 20)
            ]

            flat_tgt = sf_target.reshape(-1, sf_dim)
            sf_center = flat_tgt.mean(0)
            sf_scale = flat_tgt.std(0).clamp_min(1e-6)

            # MTP: horizon h regresses Lambda_{t+h}. Masks are the base's, verbatim -- a
            # future target is valid only when no reset boundary lies between source and
            # target state and it stays inside the rollout.
            mtp = args.critic_mtp_horizon
            sf_mtp = sf_target.new_zeros((args.num_steps, args.num_envs, mtp, sf_dim))
            return_mtp_mask = torch.zeros(
                (args.num_steps, args.num_envs, mtp), dtype=torch.bool, device=device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                sf_mtp[:valid_len, :, h] = sf_target[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h

            # ---- three-way EV gate: the falsifier, computed on an UNRIGGED target ----
            # The usual EV target `b_returns = advantages + values` CONTAINS the critic's
            # own bootstrapped values, and at dt=0.05 the errors are strongly
            # state-autocorrelated -- a critic scores well against a target built from
            # itself. Every predictor below is scored against the same truncated
            # Monte-Carlo discounted return instead.
            #
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition.
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_resid = torch.zeros_like(rewards)          # discounted PROBE residual
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            resid_run = torch.zeros_like(rewards[0])
            resid_tb2 = reward_resid.reshape(args.num_steps, args.num_envs)
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                resid_run = resid_tb2[t] + args.gamma * cont * resid_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
                mc_resid[t] = resid_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())
            if args.pgvec:
                (
                    rho,
                    pgvec_reliability,
                    pgvec_predictive_skill,
                    pgvec_split_agreement,
                    pgvec_rho_raw,
                    pgvec_coefficient_norm,
                    pgvec_active_folds,
                ) = cross_fitted_pgvec(
                    policy_vector_residual,
                    standardized_actions,
                    mc_ret - values,
                    mc_mask.reshape(args.num_steps, args.num_envs),
                    args.pgvec_ridge,
                    args.pgvec_min_fold_samples,
                )
            else:
                rho = standardized_actions.new_zeros(
                    args.num_steps, args.num_envs, act_dim
                )
                pgvec_reliability = rho.new_zeros(())
                pgvec_predictive_skill = rho.new_zeros(())
                pgvec_split_agreement = rho.new_zeros(())
                pgvec_rho_raw = rho.new_zeros(())
                pgvec_coefficient_norm = rho.new_zeros(())
                pgvec_active_folds = 0
            if n_mc >= 256:
                flat_mc = mc_ret.reshape(-1)[mc_mask]
                feat_mc = critic_feat_buf[mc_mask]
                ones_mc = feat_mc.new_ones(feat_mc.shape[0], 1)
                # (1) reference: what a scalar critic reading THIS trunk could achieve.
                #     Detached closed-form probe, so no scalar-return gradient ever enters
                #     the network -- it replaces the in-run HL-Gauss comparator that the
                #     head swap removed.
                trunk_feat_mc = torch.cat([feat_mc, ones_mc], dim=-1)
                ev_trunk_probe = ev_score(
                    trunk_feat_mc @ solve_reward_probe(trunk_feat_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # (3) how much reward-relevant signal is LINEARLY present in the
                #     INSTANTANEOUS state. Not a ceiling on psi: psi reads the trunk, not
                #     s_t, and accumulates phi over time, so (2) > (3) is expected and is
                #     in fact the successor-feature construction earning its keep.
                #     THIS IS THE SLOT v1/v2 USED FOR gate/ev_latent_cap (the same probe on
                #     e). Comparing that curve against this one is the cleanest read on
                #     what the encoder adds to a one-step linear readout -- but note it is
                #     only a ONE-STEP read and says nothing about the occupancy structure
                #     e contributes to Lambda, which is what this whole run is testing.
                s_mc = torch.cat([obs.reshape(flat_obs_shape)[mc_mask], ones_mc], dim=-1)
                ev_obs_probe = ev_score(
                    s_mc @ solve_reward_probe(s_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # THE decisive reward-probe metric, and the reason R^2 and the AC lags are
                # both kept only as secondary reads. Since V = w_r.Lambda and
                # Lambda = sum gamma^k phi, we have w_r.Lambda = G_t - sum gamma^k eps_{t+k}:
                # the value error IS the discounted sum of probe residuals. R^2 alone is
                # actively misleading here -- measured offline, adding s' to phi raises R^2
                # from 0.61 to 0.94 while making THIS number WORSE (0.77 -> 0.84), because
                # the discounted sum amplifies a small correlated residual far more than a
                # large white one. Ratio to the value's own spread; lower is better.
                value_err_frac = float(
                    mc_resid.reshape(-1)[mc_mask].std() / flat_mc.std().clamp_min(1e-12)
                )
                # (2) the treatment: the ONLINE value the actor actually consumed.
                ev_sf = ev_score(values.reshape(-1)[mc_mask], flat_mc)
                # (1) and (3) are IN-SAMPLE ridge fits, i.e. optimistic upper bounds; (2) is
                # an honest out-of-sample prediction. So (3) vs (1) is the apples-to-apples
                # comparison -- (3) << (1) means the reward BASIS is the bottleneck and no
                # amount of psi quality can rescue this. (2) << (3) with (3) ~ (1) instead
                # points at the SF machinery (recursion, standardization).
            else:
                ev_trunk_probe = ev_obs_probe = ev_sf = value_err_frac = float("nan")

            # Install only after every current-rollout actor/critic target and router
            # quantity is frozen. This becomes the collection covector next iteration.
            w_r_next = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            w_r = w_r_next

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_logprobs_dim = logprobs_dim.reshape(-1, act_dim)
        b_rho = rho.reshape(-1, act_dim)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)
        b_diagnostic_gae = diagnostic_gae.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        b_policy_adv = b_advantages
        b_policy_adv_normed = standardize_policy_credit(b_policy_adv)
        # Scalar GAE is observed here, never consumed by the actor.
        az = (b_diagnostic_gae - b_diagnostic_gae.mean()) / (
            b_diagnostic_gae.std() + 1e-8
        )
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_sf,
                    newlogprob_dim,
                    _,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = b_policy_adv_normed[mb_inds]

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
                logratio_dim = newlogprob_dim - b_logprobs_dim[mb_inds]
                if args.pgvec:
                    ratio_dim = logratio_dim.exp()
                    lower_dim, upper_dim = per_actuator_clip_bounds(
                        args.clip_coef, clip_hi, act_dim
                    )
                    pg_loss = per_actuator_ppo_loss(
                        logratio_dim,
                        mb_advantages,
                        b_rho[mb_inds],
                        args.clip_coef,
                        clip_hi,
                    )
                    with torch.no_grad():
                        clipfrac_dim = (
                            (ratio_dim < lower_dim) | (ratio_dim > upper_dim)
                        ).float().mean().item()
                else:
                    # Matched control: the same full-suffix vector projection drives
                    # the parent's ordinary joint-ratio PPO surrogate.
                    pg_loss = joint_ppo_loss(
                        logratio,
                        mb_advantages,
                        args.clip_coef,
                        clip_hi,
                    )
                    clipfrac_dim = clipfracs[-1]

                # SUCCESSOR-FEATURE value loss: per-horizon masked MSE against the
                # standardized OCCUPANCY lambda-return, summed over valid horizons per row.
                # No scalar-return regression anywhere at the default sf_alpha=1.
                sf_tgt = b_sf_target[mb_inds]
                normalization_center = sf_center.view(1, 1, -1)
                normalization_scale = sf_scale.view(1, 1, -1)
                sf_prediction_normalized = (
                    value_sf - normalization_center
                ) / normalization_scale
                sf_target_normalized = (
                    sf_tgt - normalization_center
                ) / normalization_scale
                sf_err = sf_prediction_normalized - sf_target_normalized
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)          # (B, mtp)
                v_loss = (sf_err.pow(2).mean(-1) * value_mask).sum(dim=-1).mean()
                # Per-horizon psi MSE (last minibatch of the last epoch is what gets logged).
                sf_per_h_mse = (sf_err.detach().pow(2).mean(-1) * value_mask).sum(0) / value_mask.sum(
                    0
                ).clamp_min(1)

                entropy_loss = entropy.mean()

                ent_coef_eff = args.ent_coef

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("pgvec/reliability", pgvec_reliability, global_step)
        writer.add_scalar(
            "pgvec/predictive_skill", pgvec_predictive_skill, global_step
        )
        writer.add_scalar(
            "pgvec/split_agreement", pgvec_split_agreement, global_step
        )
        writer.add_scalar("pgvec/rho_absmean_raw", pgvec_rho_raw, global_step)
        writer.add_scalar(
            "pgvec/coefficient_norm", pgvec_coefficient_norm, global_step
        )
        writer.add_scalar("pgvec/active_folds", pgvec_active_folds, global_step)
        writer.add_scalar("pgvec/rho_absmean", rho.abs().mean(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/clipfrac_actuator", clipfrac_dim, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/fullsuffix_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/fullsuffix_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar(
            "debug/fullsuffix_policy_std", policy_adv.std(), global_step
        )
        writer.add_scalar("debug/diagnostic_gae_std", diagnostic_gae.std(), global_step)

        # ---- successor-feature diagnostics -------------------------------------
        # R^2 is the weak gate; a structural ~1-3% residual is expected (frame-skip
        # averaged reward velocity vs instantaneous qvel). The residual's
        # AUTOCORRELATION is what actually costs EV: white residual at R^2=0.98
        # costs ~0.2% EV, a gait-phase-locked one at the same R^2 costs ~2%.
        writer.add_scalar("sf/value_err_frac", value_err_frac, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        for lag, ac in zip((1, 5, 10, 20), reward_resid_ac):
            writer.add_scalar(f"sf/reward_resid_ac_lag{lag}", ac, global_step)
        for h in range(args.critic_mtp_horizon):
            writer.add_scalar(f"sf/psi_mse_h{h}", sf_per_h_mse[h].item(), global_step)
        writer.add_scalar("sf/fresh_scale_obs", sf_scale[:obs_dim].mean(), global_step)
        writer.add_scalar("sf/fresh_scale_act", sf_scale[obs_dim:-1].mean(), global_step)
        writer.add_scalar("sf/fresh_center_obs", sf_center[:obs_dim].abs().mean(), global_step)
        writer.add_scalar("sf/target_absmean_obs", flat_tgt[:, :obs_dim].abs().mean(), global_step)
        writer.add_scalar("sf/w_r_lagged_norm", w_r_lagged.norm().item(), global_step)
        writer.add_scalar("sf/w_r_next_norm", w_r_next.norm().item(), global_step)

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        # (3) occupies v1/v2's gate/ev_latent_cap slot, on raw s instead of e -- the two
        # curves side by side are the one-step read on what the encoder was contributing.
        writer.add_scalar("gate/ev_obs_probe", ev_obs_probe, global_step)
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
