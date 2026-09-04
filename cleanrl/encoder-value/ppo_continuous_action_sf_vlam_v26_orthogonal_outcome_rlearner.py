# Orthogonalized latent-outcome policy improvement with a LeJEPA vector critic.
#
# A double-ML outcome model decomposes the full D-dimensional TD(lambda) target into a
# state baseline and action-caused residual. The residual uses a behavior-invariant
# Jacobi basis with exact current-Beta centering, exposing mean, concentration, and
# cross-action effects
# without letting state prediction absorb randomized action variation. Nearby higher-
# return trajectories contribute several baseline-relative latent innovations; return is
# used only for ordinal selection. The actor matches this complete reachable vector by
# changing the analytic moments of its Beta distribution. There is no scalar advantage,
# value readout, preference covector, sampled policy-score target, or likelihood ratio.
# Exact Beta KL and full epoch rollback provide the trust region. The outcome model is
# refit after the actor, critic, and aligned LeJEPA update for use one rollout later.
#
# Base: v23's 32D value representation and v9's RMSNorm -> Linear ->
# LeakyReLU(0.5)^2 IterThink trunk. Iteration one updates only the critic head and
# outcome model.
import copy
import os
import random
import time
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
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import (
    ActionEncoder,
    ARPredictor,
    CompiledModule,
    MLP,
    SIGReg,
    StateEncoder,
)
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
    ent_coef: float = 0.0
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit.
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional latent occupancy vector.
    critic_mtp_horizon: int = 1      # one recursively bootstrapped rich value object

    # --- LeJEPA successor-feature value pathway -------------------------------------
    emb_dim: int = 32                # d. Obs manifold is 17-dim, so e is rank <= 17 regardless;
    #                                  expect effective rank ~17, which is NOT a failure signal.
    ssl_hidden: int = 256            # encoder / projector MLP width
    pred_depth: int = 2              # causal transformer depth
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = 4                 # history_size(3) + num_preds(1), per the LeWM reference
    sigreg_weight: float = 0.09      # lambda. NOTE: the Epps-Pulley statistic scales with batch
    #                                  size, so the reference value does not transfer verbatim.
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256     # exact (statistic is a mean over directions); bounds memory
    ssl_lr: float = 5e-5             # LeWM reference optimizer
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 1024            # sequences per SSL minibatch (SIGReg is ~1.1GB here, ~10GB at 8192)
    ssl_epochs: int = 8              # passes over the rollout's sequences per iteration (=64 SSL
    #                                  steps at 8192 seqs / 1024 batch), OUTSIDE the PPO minibatch loop
    ssl_grad_clip: float = 1.0       # own clip (reference lewm.yaml gradient_clip_val), fully
    #                                  separate from PPO's -- see ssl/grad_norm for the pre-clip value
    # --- Cross-fit orthogonal latent-outcome R-learner -------------------------------
    outcome_hidden: int = 256
    outcome_lr: float = 3e-4
    outcome_baseline_epochs: int = 10
    outcome_effect_epochs: int = 10
    outcome_time_batch: int = 256
    outcome_grad_clip: float = 1.0
    elite_size: int = 4
    elite_horizon_tolerance: int = 0
    outcome_scale_floor: float = 0.05
    latent_actor_kl_coef: float = 1.0
    latent_actor_target_kl: float = 0.03
    latent_actor_kl_tolerance: float = 0.25
    per_dim_lambda: bool = False     # scalar-lambda chassis used by the named reference run
    #                                  mixing time instead of using one scalar gae_lambda
    lam_min: float = 0.0             # floor. NOT a safety margin: a white coordinate's
    #                                  correct lambda IS 0 (past k=0 its conditional mean
    #                                  is the stationary mean, which the critic learns
    #                                  trivially, so MC there is pure variance). A floor of
    #                                  0.5 would pin tau >= 2 and censor the action blocks,
    #                                  i.e. clamp away half the heterogeneity being tested.
    lam_max: float = 0.99            # ceiling: tau -> 1/(1-gamma) coordinates (the constant
    #                                  intercept) would otherwise ask for exactly lambda=1
    tau_ema: float = 0.05            # EMA rate for the per-dimension mixing-time estimate,
    #                                  with the same 1/count warmup as the Lambda standardizer
    sf_target_ema: float = 0.01      # EMA rate for per-dimension Lambda standardization
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- torch.compile ---------------------------------------------------------------
    # reduce-overhead (CUDA graph trees) is NOT usable on the actor/critic path: v_loss and
    # actor_loss and value loss share the trunk forward, and the update runs backward twice with
    # retain_graph=True, cloning and re-adding grads in between. Cudagraph outputs live in
    # the graph pool, clip_grad_norm_ mutates them in place, and the second backward replays
    # over live refs -> either "accessing tensor output of CUDAGraphs that has been
    # overwritten" or a re-record per minibatch (x320/iter), which is SLOWER than eager.
    # So the trunk gets inductor fusion without graphs; the SSL nets (separate optimizer,
    # single plain backward) do get reduce-overhead.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # applies to the SSL nets only; see above for why the
    #                                        actor/critic trunk cannot take cudagraphs at all
    compile_ssl_cudagraphs: bool = False   # DEFAULT OFF: LeJEPA chains
    #   encoder -> action encoder -> predictor, and each cudagraph-wrapped call issues its
    #   own cudagraph_mark_step_begin(), which invalidates the still-pending backward of the
    #   modules called earlier in the SAME forward. It raises "accessing tensor output of
    #   CUDAGraphs that has been overwritten" on the FIRST ssl_loss.backward() -- reproduced --
    #   and suppress_errors does NOT catch it (that only covers dynamo compile-time errors).
    #   Cloning outputs does not help either: the invalidated tensors are the saved intermediates.
    #   The SSL path is ~3% of wall clock, so inductor fusion without graphs costs ~nothing.

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


class LeJepaSSL(nn.Module):
    """Attached LeJEPA prediction plus SIGReg; no reward or return objective."""

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
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk
        )

    def forward(self, obs_seq, act_seq, mask_seq, sigreg_weight):
        emb = self.encoder(obs_seq)
        action_embedding = self.action_encoder(act_seq)
        prediction = self.pred_proj(
            self.predictor(emb, action_embedding)
        )
        temporal_error = (
            prediction[:, :-1] - emb[:, 1:]
        ).square().mean(-1)
        prediction_loss = (
            temporal_error * mask_seq
        ).sum() / mask_seq.sum().clamp_min(1.0)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        return (
            prediction_loss + sigreg_weight * sigreg_loss,
            prediction_loss,
            sigreg_loss,
        )


class OrthogonalOutcomeModel(nn.Module):
    """State nuisance plus a full-rank response to orthogonal Beta treatments."""

    def __init__(self, embedding_dim, basis_dim, hidden, scale_floor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.basis_dim = basis_dim
        self.scale_floor = scale_floor
        self.baseline_body = nn.Sequential(
            nn.RMSNorm(embedding_dim),
            layer_init(nn.Linear(embedding_dim, hidden)),
            LeakyReLUSquared(),
        )
        self.baseline_head = layer_init(nn.Linear(hidden, embedding_dim), std=0.01)
        self.effect_body = nn.Sequential(
            nn.RMSNorm(embedding_dim),
            layer_init(nn.Linear(embedding_dim, hidden)),
            LeakyReLUSquared(),
        )
        self.effect_head = layer_init(
            nn.Linear(hidden, embedding_dim * basis_dim), std=0.01
        )
        self.scale_head = layer_init(nn.Linear(hidden, 1), std=0.0, bias_const=0.54132485)

    def baseline(self, embeddings):
        embeddings = embeddings.detach()
        return self.baseline_head(self.baseline_body(embeddings))

    def effect_components(self, embeddings):
        embeddings = embeddings.detach()
        features = self.effect_body(embeddings)
        response = self.effect_head(features).view(
            *embeddings.shape[:-1], self.embedding_dim, self.basis_dim
        )
        scale = F.softplus(self.scale_head(features)).squeeze(-1) + self.scale_floor
        return response, scale

    def effect(self, embeddings, basis):
        response, scale = self.effect_components(embeddings)
        return torch.einsum("...dk,...k->...d", response, basis), scale

    def forward(self, embeddings, basis):
        effect, scale = self.effect(embeddings, basis)
        return self.baseline(embeddings) + effect, scale


def beta_from_raw(raw_parameters, action_dim):
    raw_alpha, raw_beta = raw_parameters.split(action_dim, dim=-1)
    return Beta(1.0 + F.softplus(raw_alpha), 1.0 + F.softplus(raw_beta))


def beta_mean_action(raw_parameters, action_low, action_high):
    action_dim = raw_parameters.shape[-1] // 2
    distribution = beta_from_raw(raw_parameters, action_dim)
    native_mean = distribution.mean
    return action_low + (action_high - action_low) * native_mean


def beta_moment_geometry(raw_parameters, action_dim):
    """Behavior moments needed for stable degree-1/2 Jacobi features."""
    distribution = beta_from_raw(raw_parameters, action_dim)
    alpha = distribution.concentration1
    beta = distribution.concentration0
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total.square() * (total + 1.0))
    std = variance.sqrt()
    skew = 2.0 * (beta - alpha) * (total + 1.0).sqrt() / (
        (total + 2.0) * (alpha * beta).sqrt()
    )
    excess_kurtosis = 6.0 * (
        (alpha - beta).square() * (total + 1.0) - alpha * beta * (total + 2.0)
    ) / (alpha * beta * (total + 2.0) * (total + 3.0))
    degree2_scale = (excess_kurtosis + 2.0 - skew.square()).clamp_min(1e-8).sqrt()
    return mean, variance, std, skew, degree2_scale


def beta_basis_values(native_actions, reference_raw, pair_indices=None):
    """Fixed Jacobi feature values in one behavior-invariant reference frame."""
    action_dim = native_actions.shape[-1]
    mean, _, std, skew, degree2_scale = beta_moment_geometry(
        reference_raw.detach(), action_dim
    )
    degree1 = (native_actions.detach() - mean) / std
    degree2 = (degree1.square() - 1.0 - skew * degree1) / degree2_scale
    pair = pair_indices
    if pair is None:
        pair = torch.triu_indices(action_dim, action_dim, offset=1, device=degree1.device)
    pairwise = degree1[..., pair[0]] * degree1[..., pair[1]]
    return torch.cat([degree1, degree2, pairwise], dim=-1)


def expected_beta_basis(reference_raw, candidate_raw, action_dim, pair_indices=None):
    """Expected fixed-reference Jacobi features under a candidate Beta."""
    old_mean, old_variance, old_std, old_skew, degree2_scale = beta_moment_geometry(
        reference_raw.detach(), action_dim
    )
    new_mean, new_variance, _, _, _ = beta_moment_geometry(candidate_raw, action_dim)
    degree1 = (new_mean - old_mean) / old_std
    expected_old_standard_square = (
        new_variance + (new_mean - old_mean).square()
    ) / old_variance
    degree2 = (
        expected_old_standard_square - 1.0 - old_skew * degree1
    ) / degree2_scale
    pair = pair_indices
    if pair is None:
        pair = torch.triu_indices(action_dim, action_dim, offset=1, device=degree1.device)
    pairwise = degree1[..., pair[0]] * degree1[..., pair[1]]
    return torch.cat([degree1, degree2, pairwise], dim=-1)


def realized_beta_treatment(
    native_actions, behavior_raw, reference_raw, pair_indices=None
):
    """Exactly center fixed features under the current randomized behavior policy."""
    action_dim = native_actions.shape[-1]
    return beta_basis_values(native_actions, reference_raw, pair_indices) - expected_beta_basis(
        reference_raw, behavior_raw.detach(), action_dim, pair_indices
    )


def expected_beta_treatment_shift(
    behavior_raw, candidate_raw, reference_raw, action_dim, pair_indices=None
):
    """Candidate-minus-behavior moments in the invariant treatment frame."""
    return expected_beta_basis(
        reference_raw, candidate_raw, action_dim, pair_indices
    ) - expected_beta_basis(
        reference_raw, behavior_raw.detach(), action_dim, pair_indices
    )


def beta_treatment_jacobian(
    behavior_raw, reference_raw, action_dim, pair_indices=None
):
    """Local map from Beta raw parameters to fixed-frame treatment moments."""

    def single(candidate, behavior):
        return expected_beta_treatment_shift(
            behavior.unsqueeze(0),
            candidate.unsqueeze(0),
            reference_raw,
            action_dim,
            pair_indices,
        ).squeeze(0)

    return torch.func.vmap(torch.func.jacrev(single, argnums=0))(
        behavior_raw, behavior_raw
    )


def latent_vector_error(prediction, target):
    """Rotation-invariant full-vector error; target coordinates are never scalarized."""
    return 0.5 * (prediction - target.detach()).square().mean(-1)


def outcome_baseline_loss(model, embeddings, outcomes, valid):
    prediction = model.baseline(embeddings)
    if not valid.any():
        return prediction.sum() * 0.0, prediction
    return latent_vector_error(prediction, outcomes)[valid].mean(), prediction


def outcome_effect_loss(model, embeddings, basis, residual_targets, valid):
    prediction, scale = model.effect(embeddings, basis.detach())
    error = latent_vector_error(prediction, residual_targets)
    nll = error / scale.square() + scale.log()
    if not valid.any():
        return prediction.sum() * 0.0, prediction, scale
    return nll[valid].mean(), prediction, scale


@torch.no_grad()
def outcome_metrics(model, embeddings, basis, outcomes, valid, baseline_override=None):
    if not valid.any():
        return (float("nan"),) * 6
    baseline = (
        model.baseline(embeddings)
        if baseline_override is None
        else baseline_override.detach()
    )
    action_effect, scale = model.effect(embeddings, basis)
    prediction = baseline + action_effect
    target_residual = outcomes - baseline
    full_error = latent_vector_error(prediction, outcomes)
    baseline_error = latent_vector_error(baseline, outcomes)
    cosine = F.cosine_similarity(action_effect, target_residual, dim=-1, eps=1e-8)
    effect_fraction = action_effect.norm(dim=-1) / target_residual.norm(dim=-1).clamp_min(1e-8)
    return (
        float(full_error[valid].mean()),
        float(baseline_error[valid].mean()),
        float((baseline_error[valid] - full_error[valid]).mean()),
        float(cosine[valid].mean()),
        float(effect_fraction[valid].mean()),
        float(scale[valid].mean()),
    )


@torch.no_grad()
def local_elite_innovation_goals(
    embeddings, horizons, valid, returns, innovations, elite_size, horizon_tolerance
):
    """Average several local, strictly better baseline-relative outcome innovations."""
    distances = torch.cdist(embeddings, embeddings)
    horizon_distance = (horizons.unsqueeze(2) - horizons.unsqueeze(1)).abs()
    candidate = (
        valid.unsqueeze(2)
        & valid.unsqueeze(1)
        & (returns.unsqueeze(2) < returns.unsqueeze(1))
        & (horizon_distance <= horizon_tolerance)
    )
    candidate.diagonal(dim1=1, dim2=2).fill_(False)
    distances.masked_fill_(~candidate, float("inf"))
    k = min(elite_size, embeddings.shape[1] - 1)
    selected_distance, selected = distances.topk(k, dim=-1, largest=False)
    selected_valid = selected_distance.isfinite()
    gathered = innovations.unsqueeze(1).expand(
        -1, embeddings.shape[1], -1, -1
    ).gather(
        2, selected.unsqueeze(-1).expand(-1, -1, -1, innovations.shape[-1])
    )
    count = selected_valid.sum(-1)
    goal = (
        gathered * selected_valid.unsqueeze(-1)
    ).sum(2) / count.clamp_min(1).unsqueeze(-1)
    return goal.detach(), count.gt(0), selected_distance, count


@torch.no_grad()
def mean_beta_kl(old_raw, new_raw, action_dim):
    return kl_divergence(
        beta_from_raw(old_raw, action_dim), beta_from_raw(new_raw, action_dim)
    ).sum(-1).mean()




@torch.no_grad()
def orthogonal_alignment(source, target):
    """Map source coordinates into target's frame without changing geometry."""
    source64 = source.double() - source.double().mean(0, keepdim=True)
    target64 = target.double() - target.double().mean(0, keepdim=True)
    left, _, right_t = torch.linalg.svd(
        source64.T @ target64, full_matrices=False
    )
    return (left @ right_t).to(source.dtype)


def effective_rank(features):
    detached = features.detach().double()
    centered = detached - detached.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    return float(
        torch.exp(
            -(probabilities * (probabilities + 1e-12).log()).sum()
        )
    )


def chunk_sequences(x, seq_len):
    """(T, B, D) -> (n*B, L, D) with time contiguous within a chunk and env fixed.

    The buffer is T-MAJOR, so the usual flatten gives index t*B + e. Chunking the
    FLATTENED tensor would produce sequences that walk across envs at a fixed timestep
    rather than across time -- and because adjacent envs look similar under
    NormalizeObservation, that yields a perfectly plausible prediction loss that no curve
    would ever catch. Chunk before flattening.

    Round-trip: x[n*L + l, b] == out[n*B + b, l].
    """
    t, b = x.shape[0], x.shape[1]
    tail = x.shape[2:]
    n = t // seq_len
    return (
        x[: n * seq_len]
        .view(n, seq_len, b, *tail)
        .permute(0, 2, 1, *range(3, 3 + len(tail)))
        .reshape(n * b, seq_len, *tail)
    )


def successor_lambda_residual(
    phi,
    psi_cur,
    psi_next,
    transition_terminations,
    transition_boundaries,
    transition_valids,
    gamma,
    lam_vec,
):
    """Vector TD(lambda) residual with distinct bootstrap and trace-boundary masks."""
    residual = torch.zeros_like(phi)
    last = torch.zeros_like(phi[0])
    for t in reversed(range(phi.shape[0])):
        boot = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
        cont = (1.0 - transition_boundaries[t]).unsqueeze(-1)
        delta = phi[t] + gamma * boot * psi_next[t] - psi_cur[t]
        last = delta + gamma * lam_vec * cont * last
        residual[t] = last
    return residual


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
        # One rich latent state-value object; no scalar coordinate or hand-built basis.
        # Zero-init is deliberate and not merely neutral: with psi_old == 0 the first
        # rollout's Lambda degenerates to the plain discounted sum of phi, i.e. a clean
        # Monte-Carlo target with no bootstrap from a random head.
        self.sf_dim = args.emb_dim
        self.critic_mtp_horizon = args.critic_mtp_horizon
        # Construct the wider treatment head without advancing the global RNG, then
        # consume exactly v9's original head initialization stream. This keeps every
        # downstream actor parameter and stochastic rollout paired with v9.
        with torch.random.fork_rng(devices=[]):
            self.critic_head = layer_init(
                nn.Linear(H, args.critic_mtp_horizon * self.sf_dim, bias=False), std=0.1
            )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        paired_v9_sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1
        _paired_v9_head = layer_init(
            nn.Linear(H, 6 * paired_v9_sf_dim, bias=False), std=0.1
        )
        del _paired_v9_head
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
            raw_policy = self.raw_policy_parameters(actor_feat)
            mean, raw_lv = raw_policy.chunk(2, dim=-1)
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
        raw_policy = self.raw_policy_parameters(actor_feat)
        dist = beta_from_raw(raw_policy, raw_policy.shape[-1] // 2)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def raw_policy_parameters(self, actor_feat):
        if self.actor_dist == "gaussian":
            return torch.cat(
                [self.actor_head(actor_feat), self.actor_logvar_head(actor_feat)], dim=-1
            )
        return torch.cat(
            [self.actor_alpha_head(actor_feat), self.actor_beta_head(actor_feat)], dim=-1
        )

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns successor features (B, mtp, sf_dim); horizon-0 coordinate 0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        raw_policy = self.raw_policy_parameters(actor_feat)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_sf = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)
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
        return action, z, log_prob, entropy, value_sf, raw_policy

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


def _uncompiled(module):
    return module._orig_mod if isinstance(module, CompiledModule) else module


@torch.no_grad()
def copy_policy(source, target):
    """Copy only the behavior policy, including a possibly compiled source trunk."""
    if source.share_backbone:
        target.trunk.load_state_dict(_uncompiled(source.trunk).state_dict())
    else:
        target.actor_trunk.load_state_dict(
            _uncompiled(source.actor_trunk).state_dict()
        )
    if source.actor_dist == "beta":
        target.actor_alpha_head.load_state_dict(source.actor_alpha_head.state_dict())
        target.actor_beta_head.load_state_dict(source.actor_beta_head.state_dict())
    else:
        target.actor_head.load_state_dict(source.actor_head.state_dict())
        target.actor_logvar_head.load_state_dict(source.actor_logvar_head.state_dict())


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.critic_mtp_horizon != 1:
        raise ValueError("this design has exactly one recursively bootstrapped rich value object")
    if args.num_envs < 2:
        raise ValueError("latent outcome improvement needs at least two parallel environments")
    if not 1 <= args.elite_size < args.num_envs:
        raise ValueError("elite_size must be between 1 and num_envs - 1")
    if args.outcome_scale_floor <= 0:
        raise ValueError("outcome_scale_floor must be positive")
    if args.actor_dist != "beta":
        raise ValueError("orthogonal outcome features require the Beta actor")
    if args.auto_entropy or args.ent_coef != 0.0:
        raise ValueError("latent outcome improvement has no scalar entropy-value objective")
    if args.latent_actor_target_kl <= 0:
        raise ValueError("latent_actor_target_kl must be positive")
    if not args.separate_grad_clip:
        raise ValueError("latent-outcome warm-up requires separate_grad_clip")
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

    n_seq_per_iter = (args.num_steps // args.seq_len) * args.num_envs
    if n_seq_per_iter < args.ssl_batch:
        raise ValueError(
            f"ssl_batch={args.ssl_batch} exceeds the {n_seq_per_iter} sequences a rollout "
            "yields ((num_steps // seq_len) * num_envs). The SSL loop drops the last ragged "
            "minibatch (SIGReg's statistic scales with batch size), so it would take ZERO "
            "steps: the encoder would stay at random init and every sf/ and gate/ metric "
            "would be computed on a frozen random projection."
        )

    agent = Agent(envs, args).to(device)
    # The outcome nuisance is trained under one behavior policy and consumed under the
    # next. Retain that exact policy so its baseline can be transported analytically.
    with torch.random.fork_rng(devices=[]):
        outcome_behavior_policy = Agent(envs, args).to(device)
    copy_policy(agent, outcome_behavior_policy)
    outcome_behavior_policy.requires_grad_(False)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        raise ValueError(
            "auto_entropy's scalar soft-value bootstrap is intentionally absent from the "
            "latent-successor design; use the Beta actor or disable auto_entropy"
        )
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # --- Reward-free latent successor value pathway ---------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim
    beta_pair_indices = torch.triu_indices(
        act_dim_sf, act_dim_sf, offset=1, device=device
    )
    beta_reference_raw = torch.zeros(1, 2 * act_dim_sf, device=device)

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    # Consume exactly v13's LeJEPA initialization stream, then isolate the outcome model
    # so actor sampling remains paired with the strongest control.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    outcome_basis_dim = 2 * act_dim_sf + act_dim_sf * (act_dim_sf - 1) // 2
    with torch.random.fork_rng(devices=[]):
        outcome_model = OrthogonalOutcomeModel(
            args.emb_dim,
            outcome_basis_dim,
            args.outcome_hidden,
            args.outcome_scale_floor,
        ).to(device)
    target_alignment = torch.eye(args.emb_dim, device=device)
    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    outcome_baseline_optimizer = optim.Adam(
        list(outcome_model.baseline_body.parameters())
        + list(outcome_model.baseline_head.parameters()),
        lr=args.outcome_lr,
        eps=1e-5,
    )
    outcome_effect_optimizer = optim.Adam(
        list(outcome_model.effect_body.parameters())
        + list(outcome_model.effect_head.parameters())
        + list(outcome_model.scale_head.parameters()),
        lr=args.outcome_lr,
        eps=1e-5,
    )
    outcome_generator = torch.Generator(device=device)
    outcome_generator.manual_seed(args.seed + 17021)

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
        for name in ("encoder", "action_encoder", "predictor"):
            setattr(ssl, name, CompiledModule(
                getattr(ssl, name), mode=args.compile_mode, cudagraphs=args.compile_ssl_cudagraphs
            ))
        outcome_model.baseline_body = CompiledModule(
            outcome_model.baseline_body, mode=args.compile_mode, cudagraphs=False
        )
        outcome_model.baseline_head = CompiledModule(
            outcome_model.baseline_head, mode=args.compile_mode, cudagraphs=False
        )
        outcome_model.effect_body = CompiledModule(
            outcome_model.effect_body, mode=args.compile_mode, cudagraphs=False
        )
        outcome_model.effect_head = CompiledModule(
            outcome_model.effect_head, mode=args.compile_mode, cudagraphs=False
        )
        outcome_model.scale_head = CompiledModule(
            outcome_model.scale_head, mode=args.compile_mode, cudagraphs=False
        )

    # Per-dimension EMA standardization of the SF target. The blocks of phi are wildly
    # heteroscedastic: for a whitened e block std(sum gamma^k e) ~ tau ~ 20 (the effective
    # autocorrelation horizon, NOT 1/(1-gamma)=100), while the a*a block is strictly
    # positive with a large mean and small variance, and the constant block is ~1/(1-gamma)
    # exactly. A single global rescale leaves the MSE dominated by the wrong block.
    sf_mean = torch.zeros(sf_dim, device=device)
    sf_std = torch.ones(sf_dim, device=device)
    sf_stat_count = 0
    # Per-dimension mixing time, initialised at the horizon the scalar gae_lambda implies
    # (tau = 1/(1-gamma*lambda) = 16.8 at the defaults) so iteration 1 starts from the
    # base's behaviour rather than an arbitrary one. The 1/count warmup overwrites it
    # exactly on the first measurement.
    tau_vec = torch.full(
        (sf_dim,), 1.0 / (1.0 - args.gamma * args.gae_lambda), device=device
    )
    tau_stat_count = 0
    lam_vec = torch.full((sf_dim,), args.gae_lambda, device=device)
    # Fixed INDEX set (not a fixed obs batch) for the frame-drift probe: the states are
    # re-drawn from each rollout, so the probe never goes stale w.r.t. the observation
    # normalizer. Stride 31 is COPRIME with num_envs; the buffer is T-major (i = t*B + e),
    # so a stride sharing a factor with num_envs (e.g. 32 at num_envs=16) would alias onto
    # a single environment and measure drift on one trajectory instead of the state marginal.
    drift_probe_idx = torch.arange(0, args.num_steps * args.num_envs, 31, device=device)[:1024]

    def psi_raw(sf_standardized):
        """Un-standardize the head output into normalized occupancy units."""
        return sf_standardized * sf_std + sf_mean

    def encode_target(observation):
        return ssl.encoder(observation) @ target_alignment

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    policy_raws = torch.zeros((args.num_steps, args.num_envs, 2 * act_dim_sf)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    outcome_ready = False

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            # The SSL optimizer is separate, so the base's anneal (which writes
            # param_groups[0] of `optimizer` only) does not reach it. Anneal it too: it
            # freezes the encoder frame late in training, which is exactly when the policy
            # is refining and a drifting frame would do the most damage to the bootstrap.
            ssl_optimizer.param_groups[0]["lr"] = frac * args.ssl_lr
            outcome_baseline_optimizer.param_groups[0]["lr"] = frac * args.outcome_lr
            outcome_effect_optimizer.param_groups[0]["lr"] = frac * args.outcome_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, _, raw_policy = agent.get_action_and_value(next_obs)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob
            policy_raws[step] = raw_policy

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
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None
            # The transported LeJEPA encoder defines the persistent target frame.
            # Multiplication by (1-gamma) makes psi a normalized geometric occupancy
            # embedding, so its scale remains O(1) as gamma approaches one.
            emb_buf = encode_target(obs.reshape(flat_obs_shape)).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            phi = (1.0 - args.gamma) * emb_buf

            # ---- truncated-MC scaffolding (moved AHEAD of Lambda) --------------------
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition. The EV
            # gate below still consumes mc_ret/mc_mask; the tau estimator needs them here.
            #
            # mc_phi is the SAME recursion run element-wise on phi, i.e. the lambda=1
            # discounted feature sum. It is what makes the tau estimate below
            # LAMBDA-INDEPENDENT, which is the whole reason it is computed separately
            # instead of being read off sf_std: sf_std is the spread of Lambda(lambda),
            # so deriving lambda from it would close a feedback loop whose only stable
            # point is lambda_max.
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_complete = torch.zeros_like(rewards, dtype=torch.bool)
            mc_phi = torch.zeros_like(phi)
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            complete_run = torch.zeros_like(rewards[0], dtype=torch.bool)
            phi_run = torch.zeros_like(phi[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                complete_run = transition_boundaries[t].bool() | complete_run
                phi_run = phi[t] + args.gamma * cont.unsqueeze(-1) * phi_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
                mc_complete[t] = complete_run
                mc_phi[t] = phi_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())

            # ---- PER-DIMENSION lambda from measured mixing time ----------------------
            # THE IDEA. GAE's single lambda is not a free knob -- it is an assumption about
            # ONE number, the effective credit horizon: tau = 1/(1 - gamma*lambda). The
            # default lambda=0.95 at gamma=0.99 asserts tau = 16.8 steps (0.84 s) for
            # EVERY coordinate of the problem. That is false here and measurably so: the
            # obs block mixes at tau ~ 12, the SIGReg-whitened e block at tau ~ 7, the
            # action blocks are nearly white (tau ~ 1), and the constant coordinate never
            # decorrelates at all (its discounted sum is deterministic given the episode
            # end, so Monte Carlo is EXACT and zero-variance for it -- tau -> 1/(1-gamma)).
            #
            # A scalar critic cannot express this: it has one target, so it gets one
            # lambda. A VECTOR target has K coordinates with K different mixing times, so
            # the bias/variance trade should be made per coordinate. Accumulating raw
            # samples past a coordinate's mixing time adds variance with NO bias
            # reduction, because past that point E[phi_{t+k}|s_t] is just the stationary
            # mean, which the critic learns trivially.
            #
            # tau_d is measured, not assumed: for coordinate d the ratio of the discounted
            # sum's spread to the feature's own spread IS the effective horizon,
            #     tau_d = std(sum_k gamma^k phi_d) / std(phi_d),
            # with no AR(1) or exponential-decay assumption -- oscillatory coordinates
            # (joint angles under a periodic gait) cancel correctly in the sum, which a
            # lag-1 autocorrelation estimate would get badly wrong.
            #
            # Inverting tau = 1/(1 - gamma*lambda) gives lambda_d = (1 - 1/tau_d)/gamma.
            # Sanity: tau = 16.8 -> lambda = 0.95 exactly, i.e. this REDUCES to the base's
            # scalar GAE when every coordinate happens to mix at the assumed rate.
            # GATED ON SAMPLE COUNT, at the same threshold the EV gate uses. Two distinct
            # failure modes, both silent:
            #   n_mc == 0  -- mc_avail only reaches mc_window on segments >= 500 steps. On
            #     Hopper/Walker early training that is NO segment, so per_dim_lambda would
            #     be a no-op for many iterations while advB/* reported measured values.
            #   0 < n_mc < 256 -- var_phi passes the degeneracy test, and at
            #     tau_stat_count == 1 the warmup rate is exactly 1.0, so a covariance
            #     estimated from a few dozen consecutive, highly autocorrelated samples off
            #     one or two trajectories would OVERWRITE all sf_dim mixing times outright
            #     and then persist for ~1/tau_ema = 20 iterations through the EMA -- into
            #     the critic target, not just the advantage.
            # tau_stat_count is incremented INSIDE the guard so "exact init at count == 1"
            # stays true: a dead iteration must not consume a warmup slot.
            if args.per_dim_lambda and n_mc >= 256:
                mcp = mc_phi.reshape(-1, sf_dim)[mc_mask]
                php = phi.reshape(-1, sf_dim)[mc_mask]
                php_c = php - php.mean(0)
                mcp_c = mcp - mcp.mean(0)
                var_phi = php_c.pow(2).mean(0)
                # COVARIANCE, NOT A RATIO OF STANDARD DEVIATIONS. This distinction is the
                # whole estimator and it is easy to get wrong:
                #     std(sum_k gamma^k phi_d) / std(phi_d)
                # measures the spread of the REALIZED return, which is inflated by the
                # accumulated future noise that lambda exists to avoid. For a white
                # coordinate it returns 1/sqrt(1-gamma^2) ~ 7.1 instead of the true 1 --
                # it would ask for heavy Monte Carlo on exactly the coordinates where MC
                # is pure variance. Verified numerically against AR(1) ground truth.
                #
                # What is wanted is the spread of the CONDITIONAL EXPECTATION,
                # Lambda(s) = sum_k gamma^k E[phi_{t+k}|s_t]. Since the future noise is
                # independent of phi_t it drops out of a covariance:
                #     Cov(phi_d[t], sum_k gamma^k phi_d[t+k]) / Var(phi_d[t])
                #       = sum_k gamma^k rho_d(k)  =  tau_d
                # i.e. the discounted integrated autocorrelation time, estimated as a
                # per-dimension OLS slope. White -> 1. AR(1) -> 1/(1-gamma*rho).
                # Oscillatory -> small, because the alternating rho_d(k) cancel in the sum,
                # which is the failure mode a lag-1 estimate cannot see.
                tau_raw = (php_c * mcp_c).mean(0) / var_phi.clamp_min(1e-8)
                # A coordinate with Var(phi_d) = 0 -- the intercept, and any dead obs dim --
                # gives 0/0. The estimator cannot see it, but its lambda is NOT irrelevant:
                # lambda_d multiplies the trace of delta_sf[...,d], and psi_cur[...,d] is a
                # learned state-varying quantity (for the intercept it encodes expected
                # discounted remaining episode length), so this coordinate moves both
                # adv_vector and the critic target.
                # The right answer is available analytically instead of empirically: if
                # phi_d is deterministic then so is sum_k gamma^k phi_d, i.e. Monte Carlo is
                # EXACT and zero-variance for it, so lambda_d = 1 with tau = 1/(1-gamma).
                # This is the coordinate the header names as the motivating example; falling
                # back to the scalar default here would silently contradict it.
                tau_det = 1.0 / (1.0 - args.gamma)
                tau_raw = torch.where(
                    var_phi > 1e-8, tau_raw, torch.full_like(tau_raw, tau_det)
                )
                # NOTE, load-bearing: this same branch is what makes an all-NaN var_phi
                # (n_mc == 0, mean over an empty tensor) safe, since NaN > 1e-8 is False.
                # The n_mc guard above makes that unreachable; do not invert the predicate.
                # tau < 1 means anti-persistent (the discounted sum partially cancels the
                # current value); the shortest meaningful credit horizon is one step.
                tau_raw = tau_raw.clamp(1.0, 1.0 / (1.0 - args.gamma))
                # Same 1/count warmup idiom as the Lambda standardizer: an exact running
                # mean early (and an exact init at count == 1), decaying into the EMA.
                tau_stat_count += 1
                tau_rate = max(args.tau_ema, 1.0 / tau_stat_count)
                tau_vec = tau_vec + tau_rate * (tau_raw - tau_vec)
                lam_vec = ((1.0 - 1.0 / tau_vec.clamp_min(1.0 + 1e-6)) / args.gamma).clamp(
                    args.lam_min, args.lam_max
                )
            else:
                lam_vec = torch.full((sf_dim,), args.gae_lambda, device=device)

            # ---- vector TD(lambda) ---------------------------------------------------
            # Element-wise on phi, mirroring the reward GAE. THE TWO MASKS ARE NOT THE SAME
            # MASK. They differ exactly at a TRUNCATION (term=0, valid=1, boundary=1):
            # bootstrap through it, but CUT the lambda trace. Collapsing them would let the
            # next episode's discounted phi-sum bleed backward into this episode's target
            # from a fresh, near-zero-velocity reset state -- ~1.6% of samples corrupted by
            # roughly the reset-vs-running value gap. Written in residual space so the
            # rollout tail (E_T = 0) is handled for free.
            # lam_vec is (sf_dim,) and broadcasts over (B, sf_dim); with per_dim_lambda off
            # it is a constant vector and this is bit-identical to v2.
            sf_residual = successor_lambda_residual(
                phi,
                psi_cur,
                psi_next,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                lam_vec,
            )
            sf_target = sf_residual + psi_cur                 # Lambda_t

            # Preserve the entire vector trace for policy credit. There is no scalar
            # value estimate and no scalar bootstrap in this design.
            vector_policy_trace = sf_residual
            returns = mc_ret
            if auto_alpha:
                raise AssertionError("auto_entropy is rejected at startup")
            # Per-dimension EMA standardization of the target (see setup for why).
            # The 1/count warmup is NOT cosmetic. e's own scale moves fast for the first
            # few dozen iterations while SIGReg whitens it, but 1/sf_target_ema = 100
            # iterations is 3.3M steps of lag -- the standardization would spend a third of
            # the run tracking a distribution the encoder left behind. rate = 1/count is an
            # exact running mean early (and an exact init at count == 1, replacing a
            # separate first-iteration branch) and decays into the plain EMA once the
            # encoder settles.
            sf_stat_count += 1
            sf_rate = max(args.sf_target_ema, 1.0 / sf_stat_count)
            flat_tgt = sf_target.reshape(-1, sf_dim)
            tgt_mean, tgt_std = flat_tgt.mean(0), flat_tgt.std(0).clamp_min(1e-6)
            sf_mean = sf_mean + sf_rate * (tgt_mean - sf_mean)
            sf_std = sf_std + sf_rate * (tgt_std - sf_std)

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
            sf_mtp = (sf_mtp - sf_mean) / sf_std              # head regresses standardized units

            # The randomized Beta sample becomes an exactly centered, unit-Gram
            # treatment. The action-free nuisance is predicted separately; elite goals
            # transfer only outcome innovations, never baseline state differences.
            model_behavior_features = outcome_behavior_policy._trunks(
                obs.reshape(flat_obs_shape)
            )[0]
            model_behavior_raw = outcome_behavior_policy.raw_policy_parameters(
                model_behavior_features
            ).reshape(args.num_steps, args.num_envs, 2 * act_dim_sf)
            behavior_basis = realized_beta_treatment(
                latent_zs,
                policy_raws,
                beta_reference_raw,
                beta_pair_indices,
            )
            frozen_outcome_baseline = outcome_model.baseline(emb_buf)
            frozen_outcome_response, frozen_outcome_scale = (
                outcome_model.effect_components(emb_buf)
            )
            model_to_current_basis = expected_beta_treatment_shift(
                model_behavior_raw,
                policy_raws,
                beta_reference_raw,
                act_dim_sf,
                beta_pair_indices,
            )
            transported_outcome_baseline = frozen_outcome_baseline + torch.einsum(
                "...dk,...k->...d", frozen_outcome_response, model_to_current_basis
            )
            outcome_innovations = (sf_target - transported_outcome_baseline).detach()
            elite_candidate = mc_complete & transition_boundaries.eq(0)
            (
                desired_outcome_effect,
                loser_mask,
                elite_distances,
                elite_count,
            ) = local_elite_innovation_goals(
                emb_buf,
                mc_avail,
                elite_candidate,
                returns,
                outcome_innovations,
                args.elite_size,
                args.elite_horizon_tolerance,
            )
            outcome_holdout_time = (
                (torch.arange(args.num_steps, device=device) + iteration) % 5 == 0
            )
            outcome_holdout_valid = transition_boundaries.eq(0) & outcome_holdout_time.unsqueeze(1)
            (
                preactor_outcome_error,
                preactor_outcome_baseline_error,
                preactor_outcome_action_gain,
                preactor_outcome_cosine,
                preactor_outcome_effect_fraction,
                preactor_outcome_scale,
            ) = outcome_metrics(
                outcome_model,
                emb_buf,
                behavior_basis,
                sf_target,
                outcome_holdout_valid,
                transported_outcome_baseline,
            )

        # Preserve the exact pre-update policy under which this rollout was collected.
        # The outcome model fitted below is relative to it and will transport from it on
        # the next rollout before any new actor target is constructed.
        copy_policy(agent, outcome_behavior_policy)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_outcome_response = frozen_outcome_response.reshape(
            -1, args.emb_dim, outcome_basis_dim
        )
        b_outcome_scale = frozen_outcome_scale.reshape(-1)
        b_desired_outcome_effect = desired_outcome_effect.reshape(-1, args.emb_dim)
        b_loser_mask = loser_mask.reshape(-1)
        b_old_raw = policy_raws.reshape(-1, 2 * act_dim_sf)
        b_returns = returns.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)

        b_inds = np.arange(args.batch_size)
        # Before a learned outcome response exists, update only the zero-initialized head.
        # Critic gradients into the shared trunk would otherwise change the actor policy
        # even though latent outcome improvement is intentionally off during warm-up.
        active_critic_params = (
            critic_params if outcome_ready else list(agent.critic_head.parameters())
        )
        for parameter in outcome_model.parameters():
            parameter.requires_grad_(False)
        actor_epochs_accepted = 0
        kl_guard_rejections = 0
        accepted_actor_grad_norm_sum = 0.0
        accepted_critic_grad_norm_sum = 0.0
        accepted_grad_steps = 0
        for epoch in range(args.update_epochs):
            epoch_actor_grad_norm_sum = 0.0
            epoch_critic_grad_norm_sum = 0.0
            epoch_grad_steps = 0
            if outcome_ready:
                # A critic update also moves the shared actor trunk. Snapshot the whole
                # joint step so the rollout-level exact-KL guard can reject an epoch,
                # including its Adam moments, rather than merely report an overshoot.
                epoch_parameters = [parameter.detach().clone() for parameter in agent.parameters()]
                epoch_optimizer_state = copy.deepcopy(optimizer.state_dict())
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, _, entropy, value_sf, current_raw = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                current_exact_kl = kl_divergence(
                    beta_from_raw(b_old_raw[mb_inds], act_dim_sf),
                    beta_from_raw(current_raw, act_dim_sf),
                ).sum(-1)
                if outcome_ready:
                    candidate_basis = expected_beta_treatment_shift(
                        b_old_raw[mb_inds],
                        current_raw,
                        beta_reference_raw,
                        act_dim_sf,
                        beta_pair_indices,
                    )
                    predicted_outcome_effect = torch.einsum(
                        "bdk,bk->bd", b_outcome_response[mb_inds], candidate_basis
                    )
                    outcome_scale = b_outcome_scale[mb_inds]
                    outcome_error = latent_vector_error(
                        predicted_outcome_effect,
                        b_desired_outcome_effect[mb_inds],
                    )
                    valid_actor = b_loser_mask[mb_inds]
                    improvement_loss = (
                        (
                            outcome_error / outcome_scale.square()
                            + outcome_scale.log()
                        )[valid_actor].mean()
                        if valid_actor.any()
                        else current_raw.sum() * 0.0
                    )
                    actor_loss = (
                        improvement_loss
                        + args.latent_actor_kl_coef * current_exact_kl.mean()
                    )
                else:
                    actor_loss = current_raw.sum() * 0.0
                    improvement_loss = actor_loss.detach()

                # Every coordinate belongs to the one rich state-value object.
                sf_tgt = b_sf_target[mb_inds]
                sf_err = value_sf - sf_tgt
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)
                v_loss = (
                    sf_err.square().mean(-1) * value_mask
                ).sum() / value_mask.sum().clamp_min(1.0)
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
                    critic_gn = nn.utils.clip_grad_norm_(active_critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [
                        (p, p.grad.detach().clone())
                        for p in active_critic_params
                        if p.grad is not None
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    (actor_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = actor_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                epoch_actor_grad_norm_sum += float(actor_gn)
                epoch_critic_grad_norm_sum += float(critic_gn)
                epoch_grad_steps += 1

            if outcome_ready:
                with torch.no_grad():
                    epoch_raw_chunks = []
                    for guard_start in range(0, args.batch_size, 4096):
                        guard_features = agent._trunks(
                            b_obs[guard_start : guard_start + 4096]
                        )[0]
                        epoch_raw_chunks.append(agent.raw_policy_parameters(guard_features))
                    epoch_raw = torch.cat(epoch_raw_chunks)
                    epoch_exact_kl = mean_beta_kl(b_old_raw, epoch_raw, act_dim_sf)
                if float(epoch_exact_kl) > (
                    args.latent_actor_target_kl
                    * (1.0 + args.latent_actor_kl_tolerance)
                ):
                    with torch.no_grad():
                        for parameter, saved in zip(
                            agent.parameters(), epoch_parameters, strict=True
                        ):
                            parameter.copy_(saved)
                    optimizer.load_state_dict(epoch_optimizer_state)
                    kl_guard_rejections += 1
                    break
            accepted_actor_grad_norm_sum += epoch_actor_grad_norm_sum
            accepted_critic_grad_norm_sum += epoch_critic_grad_norm_sum
            accepted_grad_steps += epoch_grad_steps
            actor_epochs_accepted += 1

        with torch.no_grad():
            final_raw_chunks = []
            final_value_chunks = []
            for metric_start in range(0, args.batch_size, 4096):
                actor_features, critic_features = agent._trunks(
                    b_obs[metric_start : metric_start + 4096]
                )
                final_raw_chunks.append(agent.raw_policy_parameters(actor_features))
                final_value_chunks.append(
                    agent.critic_head(critic_features).view(
                        -1, args.critic_mtp_horizon, sf_dim
                    )
                )
            final_raw = torch.cat(final_raw_chunks)
            final_value_sf = torch.cat(final_value_chunks)
            final_exact_kl = kl_divergence(
                beta_from_raw(b_old_raw, act_dim_sf),
                beta_from_raw(final_raw, act_dim_sf),
            ).sum(-1)
            final_mean_actions = beta_mean_action(
                final_raw, agent.action_low, agent.action_high
            )
            behavior_distribution = beta_from_raw(b_old_raw, act_dim_sf)
            final_distribution = beta_from_raw(final_raw, act_dim_sf)
            final_expected_basis = expected_beta_treatment_shift(
                b_old_raw,
                final_raw,
                beta_reference_raw,
                act_dim_sf,
                beta_pair_indices,
            )
            predicted_outcome_effect = torch.einsum(
                "bdk,bk->bd", b_outcome_response, final_expected_basis
            )
            actor_outcome_scale = b_outcome_scale
            old_outcome_error = latent_vector_error(
                torch.zeros_like(b_desired_outcome_effect), b_desired_outcome_effect
            )
            new_outcome_error = latent_vector_error(
                predicted_outcome_effect, b_desired_outcome_effect
            )
            if b_loser_mask.any():
                actor_old_error = float(old_outcome_error[b_loser_mask].mean())
                actor_new_error = float(new_outcome_error[b_loser_mask].mean())
                actor_predicted_improvement = actor_old_error - actor_new_error
                desired_outcome_effect_norm = float(
                    b_desired_outcome_effect[b_loser_mask].norm(dim=-1).mean()
                )
                desired_outcome_effect_rank = effective_rank(
                    b_desired_outcome_effect[b_loser_mask]
                )
                full_improvement_loss = float(
                    (
                        new_outcome_error / actor_outcome_scale.square()
                        + actor_outcome_scale.log()
                    )[b_loser_mask].mean()
                )
            else:
                actor_old_error = actor_new_error = actor_predicted_improvement = float("nan")
                desired_outcome_effect_norm = float("nan")
                desired_outcome_effect_rank = float("nan")
                full_improvement_loss = 0.0
            behavior_mean_actions = agent.action_low + (
                agent.action_high - agent.action_low
            ) * behavior_distribution.mean
            actor_mean_shift = float(
                (final_mean_actions - behavior_mean_actions).norm(dim=-1).mean()
            )
            actor_variance_shift = float(
                (final_distribution.variance - behavior_distribution.variance)
                .norm(dim=-1)
                .mean()
            )
            actor_basis_shift = float(final_expected_basis.norm(dim=-1).mean())
            actor_linear_basis_shift = float(
                final_expected_basis[:, :act_dim_sf].norm(dim=-1).mean()
            )
            actor_quadratic_basis_shift = float(
                final_expected_basis[:, act_dim_sf:].norm(dim=-1).mean()
            )
            beta_entropy = float(final_distribution.entropy().sum(-1).mean())
            beta_alpha_mean = float(final_distribution.concentration1.mean())
            beta_beta_mean = float(final_distribution.concentration0.mean())
            final_sf_error = final_value_sf - b_sf_target
            final_value_mask = b_target_mask.to(final_sf_error.dtype)
            full_value_loss = float(
                (final_sf_error.square().mean(-1) * final_value_mask).sum()
                / final_value_mask.sum().clamp_min(1.0)
            )
            sf_per_h_mse = (
                final_sf_error.square().mean(-1) * final_value_mask
            ).sum(0) / final_value_mask.sum(0).clamp_min(1.0)
            full_policy_loss = (
                full_improvement_loss
                + args.latent_actor_kl_coef * float(final_exact_kl.mean())
                if outcome_ready
                else 0.0
            )
            accepted_actor_grad_norm = (
                accepted_actor_grad_norm_sum / max(accepted_grad_steps, 1)
            )
            accepted_critic_grad_norm = (
                accepted_critic_grad_norm_sum / max(accepted_grad_steps, 1)
            )

            # Project through the actual 12-parameter Beta tangent, not the unrestricted
            # 27D polynomial response. Pairwise basis directions are second-order at the
            # behavior policy and must not inflate local controllability.
            loser_indices = b_loser_mask.nonzero(as_tuple=False).flatten()[:256]
            if loser_indices.numel():
                actor_response = b_outcome_response[loser_indices]
                actor_targets = b_desired_outcome_effect[loser_indices]
                with torch.enable_grad():
                    basis_jacobian = beta_treatment_jacobian(
                        b_old_raw[loser_indices],
                        beta_reference_raw,
                        act_dim_sf,
                        beta_pair_indices,
                    )
                local_actor_response = torch.einsum(
                    "ndk,nkp->ndp", actor_response, basis_jacobian.detach()
                )
                response_solution = torch.linalg.lstsq(
                    local_actor_response.float(), actor_targets.float().unsqueeze(-1)
                ).solution
                controllable_target = torch.matmul(
                    local_actor_response.float(), response_solution
                ).squeeze(-1)
                actor_controllable_fraction = float(
                    (
                        controllable_target.norm(dim=-1)
                        / actor_targets.float().norm(dim=-1).clamp_min(1e-8)
                    ).mean()
                )
                response_singular_values = torch.linalg.svdvals(
                    local_actor_response.float()
                )
                response_spectrum = response_singular_values / response_singular_values.sum(
                    -1, keepdim=True
                ).clamp_min(1e-12)
                actor_response_effective_rank = float(
                    torch.exp(
                        -(
                            response_spectrum
                            * (response_spectrum + 1e-12).log()
                        ).sum(-1)
                    ).mean()
                )
            else:
                actor_controllable_fraction = float("nan")
                actor_response_effective_rank = float("nan")
            latent_target_flat = sf_target.reshape(-1, sf_dim)
            latent_prediction_flat = psi_cur.reshape(-1, sf_dim)
            latent_target_variance = latent_target_flat.var(
                0, unbiased=False
            ).clamp_min(1e-8)
            latent_preupdate_ev = float(
                1.0
                - (
                    (latent_prediction_flat - latent_target_flat)
                    .square()
                    .mean(0)
                    / latent_target_variance
                ).mean()
            )
            latent_target_rank = effective_rank(latent_target_flat)
            latent_prediction_rank = effective_rank(latent_prediction_flat)

        # ---- LeJEPA step --------------------------------------------------------------
        # ONCE PER ITERATION, OUTSIDE the 320-minibatch PPO loop: inside it costs +100-200%
        # wall clock, outside it costs +10-20%. Placed AFTER the PPO update so the encoder
        # frame is frozen across the entire target-construction + critic-fitting phase. The
        # residual drift is one ITERATION's worth (ssl_epochs * (n_seq // ssl_batch) = 64
        # steps at defaults, not one), which is exactly what ssl/frame_drift_* measures.
        with torch.no_grad():
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            # Position l predicts l+1. Masking ONLY the crossing transition
            # (1 - boundaries[l]) is not enough: chunks are cut on fixed multiples of
            # seq_len in rollout time, not on episode boundaries, so a reset at intra-chunk
            # position 0 still leaves positions 1..L-2 predicting normally while causal
            # attention reads position 0 -- a state from the PREVIOUS episode. The CUMPROD
            # requires the whole context 0..l to be same-episode, which is the actual
            # precondition for the prediction to be well-posed. Prefix positions before the
            # first boundary are kept, so a straddling chunk is degraded, not discarded.
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
            seq_mask = seq_cont.cumprod(dim=1)[:, :-1]
        n_seq = seq_obs.shape[0]
        ssl_pred_l = ssl_sig_l = ssl_gn_sum = 0.0
        ssl_steps = 0
        # e BEFORE this iteration's SSL step, on states from THIS rollout. Paired with the
        # post-step embedding of the SAME inputs below, this isolates frame movement from
        # distribution shift without needing to keep a copy of the old encoder.
        with torch.no_grad():
            probe_emb_before = ssl.encoder(
                obs.reshape(flat_obs_shape)[drift_probe_idx]
            ).clone()
            target_probe_before = probe_emb_before @ target_alignment
        for _ in range(args.ssl_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l = ssl(
                    seq_obs[idx],
                    seq_act[idx],
                    seq_mask[idx],
                    args.sigreg_weight,
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_gn = nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_optimizer.step()
                ssl_pred_l += pred_l.item()
                ssl_sig_l += sig_l.item()
                ssl_gn_sum += float(ssl_gn)
                ssl_steps += 1
        ssl_pred_l /= max(ssl_steps, 1)
        ssl_sig_l /= max(ssl_steps, 1)
        ssl_gn_sum /= max(ssl_steps, 1)

        # ---- encoder health + frame drift ----------------------------------------------
        # SIGReg pins the DISTRIBUTION of e, not its coordinate frame: N(0,I) is
        # rotation-invariant and the prediction loss cannot pin the frame either (the
        # predictor co-rotates). But psi_old and w_r are functions of COORDINATES, so a
        # drifting frame makes the bootstrap target stale. This is a non-stationarity the
        # design INTRODUCES, so it is measured rather than pre-emptively patched.
        #   frame_drift_raw ~ 1 and frame_drift_rot ~ 1  -> stable, bootstrap is sound
        #   frame_drift_raw << 1 but frame_drift_rot ~ 1 -> pure rotation, the failure mode
        # frame_drift_rot is in [0,1] by von Neumann's trace inequality; frame_drift_raw is
        # in [-1,1] (it goes negative under anti-correlation). Both are 1 iff no drift.
        #
        # Both embeddings are of the SAME inputs drawn from THIS rollout, taken before and
        # after this iteration's SSL step. A probe frozen at iteration 1 would be wrong in
        # two ways: those are post-NormalizeObservation vectors carrying iteration-1
        # running stats, so by mid-training they decode to raw states the agent never
        # visits -- and OOD inputs drift MORE, so the metric would cry wolf about a frame
        # instability the actual bootstrap never sees.
        with torch.no_grad():
            probe_emb_after = ssl.encoder(obs.reshape(flat_obs_shape)[drift_probe_idx]).float()
            a = probe_emb_before - probe_emb_before.mean(0)
            b_ = probe_emb_after - probe_emb_after.mean(0)
            na, nb = a.pow(2).sum(), b_.pow(2).sum()
            denom = (0.5 * (na + nb)).clamp_min(1e-12)
            drift_raw = float(1.0 - (b_ - a).pow(2).sum() / (2.0 * denom))
            drift_rot = float(torch.linalg.svdvals(b_.T.double() @ a.double()).sum() / denom.double())
            # e's marginal: SIGReg should drive these to (0, I). Read from the POST-step
            # embedding, so this panel and ssl/sigreg_epps_pulley describe the same encoder
            # -- emb_buf is the PRE-step encoder and would lag by a full 64 SSL steps,
            # making the very first ssl/emb_std point describe the random init.
            # Effective rank is expected near 17 (the observation dimension), NOT 32: a
            # single-state encoder cannot exceed the manifold's rank, and that is not a
            # failure -- random 1-D projections of a rank-17 pushforward still look
            # near-Gaussian by mixing, so the statistic converges anyway.
            emb_mean_abs = float(probe_emb_after.mean(0).abs().mean())
            emb_std = float(probe_emb_after.std(0).mean())
            cov_e = (b_.T @ b_) / b_.shape[0]
            eig = torch.linalg.eigvalsh(cov_e.double()).clamp_min(0)
            p_eig = eig / eig.sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p_eig * (p_eig + 1e-12).log()).sum()))
            # LeJEPA has no EMA teacher. Transport its newly learned output directly
            # into the critic's persistent coordinate frame.
            target_probe_raw_after = probe_emb_after
            next_alignment = orthogonal_alignment(
                target_probe_raw_after, target_probe_before
            )
            target_alignment.copy_(next_alignment)
            target_probe_after = target_probe_raw_after @ target_alignment
            target_frame_drift = float(
                1.0
                - (target_probe_after - target_probe_before).square().sum()
                / (
                    target_probe_after.square().sum()
                    + target_probe_before.square().sum()
                ).clamp_min(1e-12)
            )

        # ---- fit the next rollout's orthogonal outcome model -------------------------
        # Recompute the complete Lambda outcome in the post-update persistent frame.
        # First fit the action-free nuisance; only then fit randomized action residuals.
        with torch.no_grad():
            outcome_embeddings = encode_target(obs.reshape(flat_obs_shape)).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            outcome_psi_cur = psi_raw(
                agent.get_value(obs.reshape(flat_obs_shape))[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)
            outcome_psi_next = psi_raw(
                agent.get_value(next_obses.reshape(flat_obs_shape))[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)
            outcome_phi = (1.0 - args.gamma) * outcome_embeddings
            outcome_targets = outcome_psi_cur + successor_lambda_residual(
                outcome_phi,
                outcome_psi_cur,
                outcome_psi_next,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                lam_vec,
            )
            outcome_basis = realized_beta_treatment(
                latent_zs,
                policy_raws,
                beta_reference_raw,
                beta_pair_indices,
            )
            outcome_valid = transition_boundaries.eq(0)
            holdout_outcome_valid = outcome_valid & outcome_holdout_time.unsqueeze(1)
            training_outcome_valid = outcome_valid & ~outcome_holdout_time.unsqueeze(1)
            crossfit_outcome_baseline = outcome_model.baseline(outcome_embeddings)
            crossfit_outcome_response, _ = outcome_model.effect_components(
                outcome_embeddings
            )
            crossfit_policy_shift = expected_beta_treatment_shift(
                model_behavior_raw,
                policy_raws,
                beta_reference_raw,
                act_dim_sf,
                beta_pair_indices,
            )
            crossfit_outcome_baseline = crossfit_outcome_baseline + torch.einsum(
                "...dk,...k->...d",
                crossfit_outcome_response,
                crossfit_policy_shift,
            )
            (
                transfer_outcome_error,
                transfer_outcome_baseline_error,
                transfer_outcome_action_gain,
                transfer_outcome_cosine,
                transfer_outcome_effect_fraction,
                transfer_outcome_scale,
            ) = outcome_metrics(
                outcome_model,
                outcome_embeddings,
                outcome_basis,
                outcome_targets,
                holdout_outcome_valid,
                crossfit_outcome_baseline,
            )
        for parameter in outcome_model.parameters():
            parameter.requires_grad_(True)
        # Cross-fit the R-learner nuisance: effect targets use the baseline frozen from
        # the previous rollout. Updating the baseline first would let a flexible network
        # memorize this rollout's randomized action consequences from unique states and
        # subtract away precisely the signal the response is meant to identify.
        with torch.no_grad():
            outcome_residual_targets = (
                outcome_targets - crossfit_outcome_baseline
            ).detach()
        effect_loss_sum = effect_grad_norm_sum = 0.0
        effect_steps = 0
        effect_parameters = (
            list(outcome_model.effect_body.parameters())
            + list(outcome_model.effect_head.parameters())
            + list(outcome_model.scale_head.parameters())
        )
        for _ in range(args.outcome_effect_epochs):
            time_permutation = torch.randperm(
                args.num_steps, device=device, generator=outcome_generator
            )
            for outcome_start in range(0, args.num_steps, args.outcome_time_batch):
                time_indices = time_permutation[
                    outcome_start : outcome_start + args.outcome_time_batch
                ]
                current_effect_loss, _, _ = outcome_effect_loss(
                    outcome_model,
                    outcome_embeddings[time_indices],
                    outcome_basis[time_indices],
                    outcome_residual_targets[time_indices],
                    training_outcome_valid[time_indices],
                )
                if not training_outcome_valid[time_indices].any():
                    continue
                outcome_effect_optimizer.zero_grad(set_to_none=True)
                current_effect_loss.backward()
                effect_grad_norm = nn.utils.clip_grad_norm_(
                    effect_parameters, args.outcome_grad_clip
                )
                outcome_effect_optimizer.step()
                effect_loss_sum += current_effect_loss.item()
                effect_grad_norm_sum += float(effect_grad_norm)
                effect_steps += 1
        effect_loss_mean = effect_loss_sum / max(effect_steps, 1)
        effect_grad_norm_mean = effect_grad_norm_sum / max(effect_steps, 1)

        # The nuisance is updated only after the effect learner consumed its cross-fit
        # residuals. This fitted baseline supplies elite innovations on the next rollout.
        baseline_loss_sum = baseline_grad_norm_sum = 0.0
        baseline_steps = 0
        baseline_parameters = (
            list(outcome_model.baseline_body.parameters())
            + list(outcome_model.baseline_head.parameters())
        )
        for _ in range(args.outcome_baseline_epochs):
            time_permutation = torch.randperm(
                args.num_steps, device=device, generator=outcome_generator
            )
            for outcome_start in range(0, args.num_steps, args.outcome_time_batch):
                time_indices = time_permutation[
                    outcome_start : outcome_start + args.outcome_time_batch
                ]
                current_baseline_loss, _ = outcome_baseline_loss(
                    outcome_model,
                    outcome_embeddings[time_indices],
                    outcome_targets[time_indices],
                    training_outcome_valid[time_indices],
                )
                if not training_outcome_valid[time_indices].any():
                    continue
                outcome_baseline_optimizer.zero_grad(set_to_none=True)
                current_baseline_loss.backward()
                baseline_grad_norm = nn.utils.clip_grad_norm_(
                    baseline_parameters, args.outcome_grad_clip
                )
                outcome_baseline_optimizer.step()
                baseline_loss_sum += current_baseline_loss.item()
                baseline_grad_norm_sum += float(baseline_grad_norm)
                baseline_steps += 1
        baseline_loss_mean = baseline_loss_sum / max(baseline_steps, 1)
        baseline_grad_norm_mean = baseline_grad_norm_sum / max(baseline_steps, 1)
        outcome_ready = (
            baseline_steps > 0 and effect_steps > 0
        ) or outcome_ready

        with torch.no_grad():
            (
                trained_outcome_error,
                trained_outcome_baseline_error,
                trained_outcome_action_gain,
                trained_outcome_cosine,
                trained_outcome_effect_fraction,
                trained_outcome_scale,
            ) = outcome_metrics(
                outcome_model,
                outcome_embeddings,
                outcome_basis,
                outcome_targets,
                holdout_outcome_valid,
            )

        with torch.no_grad():
            vector_trace_flat = vector_policy_trace.reshape(-1, sf_dim)
            vector_trace_norm = vector_trace_flat.norm(dim=-1)
            valid_neighbor_distance = elite_distances[elite_distances.isfinite()]
            neighbor_p50 = (
                float(valid_neighbor_distance.quantile(0.5))
                if valid_neighbor_distance.numel()
                else float("nan")
            )
            neighbor_p90 = (
                float(valid_neighbor_distance.quantile(0.9))
                if valid_neighbor_distance.numel()
                else float("nan")
            )
            flat_basis = outcome_basis.reshape(-1, outcome_basis_dim)[outcome_valid.reshape(-1)]
            basis_mean_abs = float(flat_basis.mean(0).abs().mean())
            basis_gram = flat_basis.T @ flat_basis / flat_basis.shape[0]
            basis_gram_drift = float(
                (basis_gram - torch.eye(outcome_basis_dim, device=device)).square().mean().sqrt()
            )
            basis_effective_rank = effective_rank(flat_basis)
            outcome_response, _ = outcome_model.effect_components(
                outcome_embeddings.reshape(-1, args.emb_dim)[::32]
            )
            first_order_effect_norm = float(
                outcome_response[..., :act_dim_sf].square().mean().sqrt()
            )
            higher_order_effect_norm = float(
                outcome_response[..., act_dim_sf:].square().mean().sqrt()
            )
            elite_size_mean = float(elite_count[loser_mask].float().mean()) if loser_mask.any() else float("nan")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", full_value_loss, global_step)
        writer.add_scalar("losses/policy_loss", full_policy_loss, global_step)
        writer.add_scalar("losses/entropy", beta_entropy, global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
        writer.add_scalar("losses/exact_kl_mean", final_exact_kl.mean().item(), global_step)
        writer.add_scalar("losses/exact_kl_p50", final_exact_kl.quantile(0.50).item(), global_step)
        writer.add_scalar("losses/exact_kl_p90", final_exact_kl.quantile(0.90).item(), global_step)
        writer.add_scalar("losses/exact_kl_p95", final_exact_kl.quantile(0.95).item(), global_step)
        writer.add_scalar("losses/exact_kl_p99", final_exact_kl.quantile(0.99).item(), global_step)
        writer.add_scalar("losses/exact_kl_max", final_exact_kl.max().item(), global_step)
        writer.add_scalar("losses/actor_epochs_accepted", actor_epochs_accepted, global_step)
        writer.add_scalar("losses/kl_guard_rejections", kl_guard_rejections, global_step)
        writer.add_scalar("losses/actor_grad_norm", accepted_actor_grad_norm, global_step)
        writer.add_scalar("losses/critic_grad_norm", accepted_critic_grad_norm, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)

        # ---- rich latent state-value diagnostics --------------------------------
        for h in range(args.critic_mtp_horizon):
            writer.add_scalar(f"sf/psi_mse_h{h}", sf_per_h_mse[h].item(), global_step)
        writer.add_scalar("sf/target_std_mean", sf_std.mean().item(), global_step)
        writer.add_scalar("sf/target_absmean", sf_mean.abs().mean().item(), global_step)
        writer.add_scalar("sf/preupdate_latent_ev", latent_preupdate_ev, global_step)
        writer.add_scalar("sf/target_effective_rank", latent_target_rank, global_step)
        writer.add_scalar("sf/prediction_effective_rank", latent_prediction_rank, global_step)
        writer.add_scalar("sf/lambda_mean", lam_vec.mean().item(), global_step)
        writer.add_scalar("sf/lambda_spread", (lam_vec.max() - lam_vec.min()).item(), global_step)

        # ---- direct orthogonal latent-outcome improvement ------------------------
        writer.add_scalar("vector_adv/trace_effective_rank", effective_rank(vector_trace_flat), global_step)
        writer.add_scalar("vector_adv/trace_norm_mean", vector_trace_norm.mean().item(), global_step)
        writer.add_scalar("vector_adv/trace_norm_std", vector_trace_norm.std().item(), global_step)
        writer.add_scalar("outcome_actor/improvement_nll", full_improvement_loss, global_step)
        writer.add_scalar("outcome_actor/loser_count", b_loser_mask.sum().item(), global_step)
        writer.add_scalar("outcome_actor/loser_fraction", b_loser_mask.float().mean().item(), global_step)
        writer.add_scalar("outcome_actor/old_vector_error", actor_old_error, global_step)
        writer.add_scalar("outcome_actor/new_vector_error", actor_new_error, global_step)
        writer.add_scalar("outcome_actor/predicted_error_reduction", actor_predicted_improvement, global_step)
        writer.add_scalar("outcome_actor/desired_effect_norm", desired_outcome_effect_norm, global_step)
        writer.add_scalar("outcome_actor/desired_effect_rank", desired_outcome_effect_rank, global_step)
        writer.add_scalar("outcome_actor/mean_action_shift", actor_mean_shift, global_step)
        writer.add_scalar("outcome_actor/variance_shift", actor_variance_shift, global_step)
        writer.add_scalar("outcome_actor/basis_shift", actor_basis_shift, global_step)
        writer.add_scalar("outcome_actor/linear_basis_shift", actor_linear_basis_shift, global_step)
        writer.add_scalar("outcome_actor/quadratic_basis_shift", actor_quadratic_basis_shift, global_step)
        writer.add_scalar("outcome_actor/controllable_fraction", actor_controllable_fraction, global_step)
        writer.add_scalar("outcome_actor/response_effective_rank", actor_response_effective_rank, global_step)
        writer.add_scalar("outcome_actor/beta_entropy", beta_entropy, global_step)
        writer.add_scalar("outcome_actor/beta_alpha_mean", beta_alpha_mean, global_step)
        writer.add_scalar("outcome_actor/beta_beta_mean", beta_beta_mean, global_step)
        writer.add_scalar("outcome_actor/elite_size_mean", elite_size_mean, global_step)
        writer.add_scalar("outcome_actor/neighbor_distance_p50", neighbor_p50, global_step)
        writer.add_scalar("outcome_actor/neighbor_distance_p90", neighbor_p90, global_step)

        # The held-out one-iteration-ahead panel is the causal test of whether the
        # frozen outcome model used by this actor identified current action effects.
        writer.add_scalar("outcome/baseline_loss", baseline_loss_mean, global_step)
        writer.add_scalar("outcome/effect_nll", effect_loss_mean, global_step)
        writer.add_scalar("outcome/baseline_grad_norm", baseline_grad_norm_mean, global_step)
        writer.add_scalar("outcome/effect_grad_norm", effect_grad_norm_mean, global_step)
        writer.add_scalar("outcome/baseline_steps", baseline_steps, global_step)
        writer.add_scalar("outcome/effect_steps", effect_steps, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_error", preactor_outcome_error, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_baseline_error", preactor_outcome_baseline_error, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_action_gain", preactor_outcome_action_gain, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_action_cosine", preactor_outcome_cosine, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_effect_fraction", preactor_outcome_effect_fraction, global_step)
        writer.add_scalar("outcome/heldout_one_iteration_ahead_scale", preactor_outcome_scale, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_error", transfer_outcome_error, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_baseline_error", transfer_outcome_baseline_error, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_action_gain", transfer_outcome_action_gain, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_action_cosine", transfer_outcome_cosine, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_effect_fraction", transfer_outcome_effect_fraction, global_step)
        writer.add_scalar("outcome/heldout_postupdate_transfer_scale", transfer_outcome_scale, global_step)
        writer.add_scalar("outcome/heldout_postfit_error", trained_outcome_error, global_step)
        writer.add_scalar("outcome/heldout_postfit_baseline_error", trained_outcome_baseline_error, global_step)
        writer.add_scalar("outcome/heldout_postfit_action_gain", trained_outcome_action_gain, global_step)
        writer.add_scalar("outcome/heldout_postfit_action_cosine", trained_outcome_cosine, global_step)
        writer.add_scalar("outcome/heldout_postfit_effect_fraction", trained_outcome_effect_fraction, global_step)
        writer.add_scalar("outcome/heldout_postfit_scale", trained_outcome_scale, global_step)
        writer.add_scalar("outcome/basis_mean_abs", basis_mean_abs, global_step)
        writer.add_scalar("outcome/basis_gram_drift", basis_gram_drift, global_step)
        writer.add_scalar("outcome/basis_effective_rank", basis_effective_rank, global_step)
        writer.add_scalar("outcome/first_order_effect_norm", first_order_effect_norm, global_step)
        writer.add_scalar("outcome/higher_order_effect_norm", higher_order_effect_norm, global_step)

        # ---- LeJEPA --------------------------------------------------------------
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        writer.add_scalar("ssl/target_frame_drift", target_frame_drift, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
