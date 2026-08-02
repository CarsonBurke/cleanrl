# PPO + one geometric successor-event distribution (MDN critic).
#
# A reward-anchored event autoencoder maps each observed transition to one latent event.
# The critic predicts a conditional fixed-covariance mixture over the event reached after
# K ~ Geometric(1-gamma), plus a separate absorbing atom. Direct sampled-future NLL trains
# that one normalized distribution; a full-suffix reward moment calibrates its expectation.
# V(s) is only the mixture's analytically decoded expected reward divided by 1-gamma.
# PPO uses ordinary scalar GAE; there is no scalar value head or multi-horizon target list.
# Terminations absorb, while truncations and rollout tails bootstrap by geometric
# memorylessness from the detached pre-update distribution at the final observation.
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
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
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

    # --- One normalized geometric successor-event distribution ----------------------
    mdn_components: int = 8
    event_latent_dim: int = 16
    event_reward_dims: int = 4
    event_reward_scale: float = 5.0
    event_hidden: int = 128
    event_ae_lr: float = 1e-4
    event_ae_weight_decay: float = 1e-4
    event_ae_epochs: int = 2
    event_ae_batch: int = 1024
    event_ae_grad_clip: float = 1.0
    event_target_decay: float = 0.995
    event_state_target_decay: float = 0.995
    mdn_fixed_std: float = 0.5
    reward_moment_coef: float = 1.0

    # --- LeJEPA event-state representation ------------------------------------------
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
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

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
    # So the trunk gets inductor fusion without graphs; the SSL nets (separate optimizer,
    # single plain backward) do get reduce-overhead.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # applies to the SSL nets only; see above for why the
    #                                        actor/critic trunk cannot take cudagraphs at all
    compile_ssl_cudagraphs: bool = False   # DEFAULT OFF, and not out of caution: LeJepaSSL.forward
    #   chains encoder -> action_encoder -> predictor, and each cudagraph-wrapped call issues its
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
    """Encoder + causal action-conditioned predictor + SIGReg.

    Trained by EXACTLY two terms: a next-embedding prediction MSE with BOTH branches
    attached (no stop-gradient on the target -- SIGReg is what prevents collapse, and a
    stop-grad or EMA teacher would re-introduce the second unanchored timescale this
    design exists to remove), and SIGReg itself.

    Its only job is to define what the state embedding means. It never sees reward and
    receives no gradient from the event autoencoder or distributional critic.
    """

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
        """obs_seq (N,L,obs); act_seq (N,L,act); mask_seq (N,L-1)."""
        emb = self.encoder(obs_seq)                              # (N,L,d)
        act_emb = self.action_encoder(act_seq)                   # (N,L,d)
        pred = self.pred_proj(self.predictor(emb, act_emb))      # (N,L,d)
        # Causal predictor: position l predicts the embedding at l+1. Target is attached.
        err = (pred[:, :-1] - emb[:, 1:]).pow(2).mean(-1)        # (N,L-1)
        # Masked mean, NOT .mean() over a zeroed tensor (which would divide by the full count).
        pred_loss = (err * mask_seq).sum() / mask_seq.sum().clamp_min(1.0)
        # THE TRANSPOSE IS LOAD-BEARING. SIGReg reduces the empirical characteristic
        # function over dim -3 and scales by size(-2); both resolve to the BATCH only in
        # (T, B, D) layout. Passing (N, L, d) would average the CF over L=4 samples,
        # silently disabling collapse protection while still logging a plausible number.
        sigreg_loss = self.sigreg(emb.transpose(0, 1))           # (L, N, d)
        return pred_loss + sigreg_weight * sigreg_loss, pred_loss, sigreg_loss


class EventAutoencoder(nn.Module):
    """Reward-anchored event embedding with a slow target encoder.

    Reward coordinates are copied, not learned, so the inverse reward map cannot drift.
    The learned coordinates must reconstruct transition content through a linear decoder;
    the EMA copy supplies a slowly moving likelihood target to the critic.
    """

    def __init__(self, content_dim, args):
        super().__init__()
        learned_dim = args.event_latent_dim - args.event_reward_dims
        if learned_dim < 1:
            raise ValueError("event_latent_dim must exceed event_reward_dims")
        self.reward_dims = args.event_reward_dims
        self.reward_scale = args.event_reward_scale
        self.encoder = nn.Sequential(
            nn.RMSNorm(content_dim + 1, elementwise_affine=False),
            layer_init(nn.Linear(content_dim + 1, args.event_hidden)),
            LeakyReLUSquared(),
            layer_init(
                nn.Linear(args.event_hidden, learned_dim),
                std=LEAKY_RELU_SQUARED_OUT_GAIN,
            ),
            nn.RMSNorm(learned_dim, elementwise_affine=False),
        )
        self.decoder = layer_init(nn.Linear(args.event_latent_dim, content_dim))

    def encode(self, reward, content):
        reward_coord = (reward / self.reward_scale).unsqueeze(-1)
        reward_block = reward_coord.expand(*reward_coord.shape[:-1], self.reward_dims)
        learned = self.encoder(torch.cat([reward_coord, content], dim=-1))
        return torch.cat([reward_block, learned], dim=-1)

    def decode_content(self, latent):
        return self.decoder(latent)

    def decode_reward(self, latent):
        return latent[..., : self.reward_dims].mean(-1) * self.reward_scale


def event_content(embedding, next_obs, action, continuation):
    """Unit-scale transition content used by the learned part of the event embedding."""
    emb = F.rms_norm(embedding, (embedding.shape[-1],))
    state = F.rms_norm(next_obs, (next_obs.shape[-1],))
    cont = 2.0 * continuation.unsqueeze(-1) - 1.0
    return torch.cat([emb, state, action, action.square(), cont], dim=-1)


def mdn_parameters(raw, components, latent_dim):
    """Factor an absorbing Bernoulli atom from a fixed-covariance alive-event mixture."""
    absorb_logit = raw[:, 0]
    alive_raw = raw[:, 1:].view(-1, components, 1 + latent_dim)
    logits = alive_raw[..., 0]
    means = alive_raw[..., 1:]
    return absorb_logit, logits, means


def mdn_nll(logits, means, target, fixed_std):
    """Exact fixed-covariance diagonal-Gaussian mixture negative log likelihood."""
    target = target.unsqueeze(1)
    logstd = np.log(fixed_std)
    log_component = -0.5 * ((target - means) / fixed_std).square()
    log_component = log_component - logstd - 0.5 * np.log(2.0 * np.pi)
    log_joint = F.log_softmax(logits, dim=-1) + log_component.sum(-1)
    return -torch.logsumexp(log_joint, dim=-1)


def mdn_expected_latent(logits, means):
    return (torch.softmax(logits, dim=-1).unsqueeze(-1) * means).sum(1)


def factorized_event_nll(
    absorb_logit,
    mixture_logits,
    mixture_means,
    target,
    valid,
    absorbed,
    fixed_std,
):
    """Joint NLL of the absorbing atom and alive mixture, averaged over valid rows."""
    valid_count = valid.sum().clamp_min(1)
    absorbed_target = absorbed.to(absorb_logit.dtype)
    absorb_loss = (
        F.binary_cross_entropy_with_logits(
            absorb_logit[valid], absorbed_target[valid], reduction="sum"
        )
        / valid_count
    )
    alive = valid & ~absorbed
    if alive.any():
        alive_nll = mdn_nll(
            mixture_logits[alive],
            mixture_means[alive],
            target[alive],
            fixed_std,
        ).mean()
    else:
        alive_nll = mixture_means.sum() * 0.0
    alive_fraction = alive.sum().to(absorb_logit.dtype) / valid_count
    joint_nll = absorb_loss + alive_fraction * alive_nll / target.shape[-1]
    return joint_nll, absorb_loss, alive_nll, alive_fraction


def unwrap_compiled(module):
    """Return trainable source module whether or not CompiledModule wraps it."""
    return module._orig_mod if isinstance(module, CompiledModule) else module


@torch.no_grad()
def ema_update(target, source, decay):
    source = unwrap_compiled(source)
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, 1.0 - decay)


def sample_geometric_future_targets(
    event_latents,
    transition_terminations,
    transition_boundaries,
    transition_valids,
    bootstrap_distribution,
    fixed_std,
    gamma,
    generator,
):
    """Sample one exact K~Geometric(1-gamma) future event per rollout state.

    A future strictly after a true termination is the absorbing zero event. A future
    after a time-limit truncation or the rollout tail is sampled from the detached
    pre-update critic at that boundary state. This is exact under geometric memorylessness.
    Only a boundary whose final observation is unavailable is censored.
    """
    time_steps, num_envs, latent_dim = event_latents.shape
    uniform = torch.rand((time_steps, num_envs), generator=generator, dtype=torch.float64)
    horizons = torch.floor(torch.log1p(-uniform) / np.log(gamma)).to(torch.long)

    device = event_latents.device
    horizons_device = horizons.to(device)
    source_t = torch.arange(time_steps, device=device).unsqueeze(1).expand(-1, num_envs)
    future_t = source_t + horizons_device

    sentinel = torch.full((num_envs,), time_steps, device=device, dtype=torch.long)
    next_boundary = torch.empty((time_steps, num_envs), device=device, dtype=torch.long)
    nearest = sentinel
    boundary_bool = transition_boundaries.bool()
    for t in reversed(range(time_steps)):
        nearest = torch.where(boundary_bool[t], torch.full_like(nearest, t), nearest)
        next_boundary[t] = nearest

    crosses_boundary = future_t > next_boundary
    boundary_index = next_boundary.clamp_max(time_steps - 1)
    boundary_is_terminal = transition_terminations.gather(0, boundary_index)
    true_terminal_absorbed = (
        crosses_boundary & (next_boundary < time_steps) & boundary_is_terminal.bool()
    )
    truncated = crosses_boundary & (next_boundary < time_steps) & ~boundary_is_terminal.bool()
    overflow = future_t >= time_steps
    bootstrapped = truncated | (overflow & ~true_terminal_absorbed)

    gather_t = future_t.clamp_max(time_steps - 1)
    gather_idx = gather_t.unsqueeze(-1).expand(-1, -1, latent_dim)
    targets = event_latents.gather(0, gather_idx)
    targets = torch.where(
        true_terminal_absorbed.unsqueeze(-1), torch.zeros_like(targets), targets
    )

    absorb_logits, mixture_logits, mixture_means = bootstrap_distribution
    absorb_logits = absorb_logits.view(time_steps, num_envs)
    mixture_logits = mixture_logits.view(time_steps, num_envs, mixture_logits.shape[-1])
    mixture_means = mixture_means.view(
        time_steps, num_envs, mixture_means.shape[-2], latent_dim
    )
    bootstrap_t = torch.where(
        truncated, boundary_index, torch.full_like(boundary_index, time_steps - 1)
    )
    bootstrap_absorb_logit = absorb_logits.gather(0, bootstrap_t)
    bootstrap_logits = mixture_logits.gather(
        0, bootstrap_t.unsqueeze(-1).expand_as(mixture_logits)
    )
    bootstrap_means = mixture_means.gather(
        0,
        bootstrap_t.unsqueeze(-1).unsqueeze(-1).expand_as(mixture_means),
    )
    # Sample on a dedicated CPU generator so target construction cannot perturb the
    # CUDA actor stream. The sampled event stays fixed through all PPO epochs.
    bootstrap_absorbed = (
        torch.rand((time_steps, num_envs), generator=generator).to(device)
        < torch.sigmoid(bootstrap_absorb_logit)
    )
    component = torch.multinomial(
        torch.softmax(bootstrap_logits, dim=-1).reshape(-1, bootstrap_logits.shape[-1]).cpu(),
        num_samples=1,
        generator=generator,
    ).to(device).view(time_steps, num_envs)
    bootstrap_sample = bootstrap_means.gather(
        2,
        component.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, latent_dim),
    ).squeeze(2)
    noise = torch.randn(
        (time_steps, num_envs, latent_dim), generator=generator
    ).to(device=device, dtype=bootstrap_sample.dtype)
    bootstrap_sample = bootstrap_sample + fixed_std * noise
    bootstrap_sample = torch.where(
        bootstrap_absorbed.unsqueeze(-1),
        torch.zeros_like(bootstrap_sample),
        bootstrap_sample,
    )
    targets = torch.where(bootstrapped.unsqueeze(-1), bootstrap_sample, targets)
    absorbed = true_terminal_absorbed | (bootstrapped & bootstrap_absorbed)

    bootstrap_valid = transition_valids.gather(0, bootstrap_t).bool()
    observed_valid = transition_valids.gather(0, gather_t).bool()
    # Crossing a true terminal needs no final observation: the absorbing event is exact.
    # A sampled real boundary event still needs its final observation for the event latent.
    valid = (
        true_terminal_absorbed
        | (bootstrapped & bootstrap_valid)
        | (~bootstrapped & observed_valid)
    )
    return targets, valid, absorbed, horizons_device, bootstrapped


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def rao_blackwell_reward_moment(
    rewards,
    transition_terminations,
    transition_boundaries,
    transition_valids,
    next_transition_values,
    gamma,
):
    """Integrate observed suffixes and censor rows with an unknowable continuation."""
    time_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    valid = torch.zeros_like(rewards, dtype=torch.bool)
    for t in reversed(range(time_steps)):
        continuation_observed = transition_valids[t].bool()
        if t == time_steps - 1:
            future_return = next_transition_values[t]
            future_valid = continuation_observed
        else:
            boundary = transition_boundaries[t].bool()
            future_return = torch.where(
                boundary,
                next_transition_values[t],
                returns[t + 1],
            )
            future_valid = torch.where(
                boundary,
                continuation_observed,
                continuation_observed & valid[t + 1],
            )
        terminal = transition_terminations[t].bool()
        bootstrap_nonterminal = (
            (1.0 - transition_terminations[t]) * transition_valids[t]
        )
        returns[t] = rewards[t] + gamma * bootstrap_nonterminal * future_return
        # True termination has a known zero tail even without final_observation.
        valid[t] = terminal | future_valid
    return (1.0 - gamma) * returns, valid


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


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.gamma = args.gamma
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.mdn_components = args.mdn_components
        self.event_latent_dim = args.event_latent_dim
        self.event_reward_dims = args.event_reward_dims
        self.event_reward_scale = args.event_reward_scale
        mdn_out_dim = 1 + args.mdn_components * (1 + args.event_latent_dim)
        # Construct the treatment head without advancing global RNG, then consume exactly
        # v9's old critic-head stream. This keeps every downstream actor parameter paired.
        with torch.random.fork_rng(devices=[]):
            self.critic_head = nn.Linear(H, mdn_out_dim)
            nn.init.normal_(self.critic_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.critic_head.bias)
            with torch.no_grad():
                self.critic_head.weight[0].zero_()
                self.critic_head.bias[0] = -5.0
                stride = 1 + args.event_latent_dim
                for component in range(args.mdn_components):
                    base = 1 + component * stride
                    self.critic_head.weight[base].zero_()  # uniform mixture logits
                    self.critic_head.bias[base] = 0.0
                    reward_rows = slice(
                        base + 1, base + 1 + args.event_reward_dims
                    )
                    self.critic_head.weight[reward_rows].zero_()
                    self.critic_head.bias[reward_rows].zero_()
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
        return self._critic_distribution(critic_feat)

    def _critic_distribution(self, critic_feat):
        return mdn_parameters(
            self.critic_head(critic_feat),
            self.mdn_components,
            self.event_latent_dim,
        )

    def distribution_value(self, params):
        absorb_logit, logits, means = params
        expected = mdn_expected_latent(logits, means)
        reward = expected[..., : self.event_reward_dims].mean(-1) * self.event_reward_scale
        alive_probability = torch.sigmoid(-absorb_logit)
        return alive_probability * reward / (1.0 - self.gamma)

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
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_dist = self._critic_distribution(critic_feat)
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
        return action, z, log_prob, entropy, value_dist

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
    the old critic's scalar-return bucket distribution (sigma in bin units and its CDF
    at the return). This variant instead has a latent-event mixture. Feeding the old
    transforms a constant sigma=1 / u=0 placeholder would
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
    if not (0.0 < args.gamma < 1.0):
        raise ValueError("gamma must be in (0, 1) for a normalized geometric distribution")
    if not (0 < args.event_reward_dims < args.event_latent_dim):
        raise ValueError("event_reward_dims must be in [1, event_latent_dim)")
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
            "DISTRIBUTION in scalar-return bucket coordinates, which this event MDN does "
            "not expose. Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    n_seq_per_iter = (args.num_steps // args.seq_len) * args.num_envs
    if n_seq_per_iter < args.ssl_batch:
        raise ValueError(
            f"ssl_batch={args.ssl_batch} exceeds the {n_seq_per_iter} sequences a rollout "
            "yields ((num_steps // seq_len) * num_envs). The SSL loop drops the last ragged "
            "minibatch (SIGReg's statistic scales with batch size), so it would take ZERO "
            "steps: the encoder would stay at random init and every geometric/ and gate/ metric "
            "would be computed on a frozen random projection."
        )

    agent = Agent(envs, args).to(device)
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
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # --- LeJEPA + reward-aware event embedding ---------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    # Slow state encoder used exclusively for event targets. Construction is RNG-isolated
    # and occurs after Agent, so it cannot alter the paired v9 policy initialization.
    with torch.random.fork_rng(devices=[]):
        event_state_target = StateEncoder(obs_dim, args.emb_dim, args.ssl_hidden).to(device)
    event_state_target.load_state_dict(ssl.encoder.state_dict())
    event_state_target.requires_grad_(False)
    content_dim = args.emb_dim + obs_dim + 2 * act_dim_sf + 1
    event_ae = EventAutoencoder(content_dim, args).to(device)
    event_target = EventAutoencoder(content_dim, args).to(device)
    event_target.load_state_dict(event_ae.state_dict())
    event_target.requires_grad_(False)
    event_optimizer = optim.AdamW(
        event_ae.parameters(),
        lr=args.event_ae_lr,
        weight_decay=args.event_ae_weight_decay,
    )
    # Dedicated CPU generators keep target sampling and AE shuffling from perturbing the
    # CUDA policy-sampling stream or NumPy's PPO minibatch order.
    geometric_generator = torch.Generator(device="cpu").manual_seed(args.seed + 14_000)
    event_batch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 14_001)

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

    # Fixed INDEX set (not a fixed obs batch) for the frame-drift probe: the states are
    # re-drawn from each rollout, so the probe never goes stale w.r.t. the observation
    # normalizer. Stride 31 is COPRIME with num_envs; the buffer is T-major (i = t*B + e),
    # so a stride sharing a factor with num_envs (e.g. 32 at num_envs=16) would alias onto
    # a single environment and measure drift on one trajectory instead of the state marginal.
    drift_probe_idx = torch.arange(0, args.num_steps * args.num_envs, 31, device=device)[:1024]

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
            event_optimizer.param_groups[0]["lr"] = frac * args.event_ae_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_dist = agent.get_action_and_value(next_obs)
                values[step] = agent.distribution_value(value_dist)
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
            next_dist = agent.get_value(next_obses.reshape(flat_obs_shape))
            next_transition_values = agent.distribution_value(next_dist).reshape(
                args.num_steps, args.num_envs
            )
            critic_feat_buf = agent._trunks(obs.reshape(flat_obs_shape))[1]
            current_dist = agent._critic_distribution(critic_feat_buf)

            if auto_alpha:
                _, _, boot_logprob, _, _ = agent.get_action_and_value(next_obses[-1])
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                next_value_bonus = None

            emb_buf = ssl.encoder(obs.reshape(flat_obs_shape)).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            next_emb_buf = event_state_target(next_obses.reshape(flat_obs_shape)).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            continuation = 1.0 - transition_terminations
            content_buf = event_content(next_emb_buf, next_obses, actions, continuation)
            event_latents = event_target.encode(rewards, content_buf)
            event_frame_probe_before = event_latents.reshape(
                -1, args.event_latent_dim
            )[drift_probe_idx].clone()
            (
                geometric_targets,
                geometric_valid,
                geometric_absorbed,
                sampled_horizons,
                geometric_bootstrapped,
            ) = (
                sample_geometric_future_targets(
                    event_latents,
                    transition_terminations,
                    transition_boundaries,
                    transition_valids,
                    next_dist,
                    args.mdn_fixed_std,
                    args.gamma,
                    geometric_generator,
                )
            )

            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (
                    (1.0 - transition_terminations[t]) * transition_valids[t]
                )
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = (
                    rewards[t]
                    + args.gamma * next_transition_values[t] * bootstrap_nonterminal
                    - values[t]
                )
                lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

            # Rao-Blackwellized reward moment of the SAME geometric distribution.
            # This integrates every observed suffix reward instead of using only sampled K.
            # True terminals have an exact zero tail; truncations and rollout-tail states
            # bootstrap from the final-observation distributional value.
            reward_moment_target, reward_moment_valid = rao_blackwell_reward_moment(
                rewards,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                next_transition_values,
                args.gamma,
            )

            if auto_alpha:
                policy_adv = torch.zeros_like(rewards)
                lastgaelam = torch.zeros_like(rewards[0])
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (
                        (1.0 - transition_terminations[t]) * transition_valids[t]
                    )
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = (
                        rewards[t]
                        + args.gamma
                        * (next_transition_values[t] + next_value_bonus[t])
                        * bootstrap_nonterminal
                        - values[t]
                    )
                    lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                    )
                    policy_adv[t] = lastgaelam
            else:
                policy_adv = advantages

            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        rate = args.ret_perc_rate
                        ema_ret_lo += rate * (lo - ema_ret_lo)
                        ema_ret_hi += rate * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale

            # Unrigged truncated Monte Carlo value diagnostic.
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
            n_mc = int(mc_mask.sum())
            ev_geometric = (
                ev_score(values.reshape(-1)[mc_mask], mc_ret.reshape(-1)[mc_mask])
                if n_mc >= 256
                else float("nan")
            )

            absorb_logits_buf, logits_buf, means_buf = current_dist
            expected_latent = mdn_expected_latent(logits_buf, means_buf)
            alive_expected_reward = (
                expected_latent[:, : args.event_reward_dims].mean(-1)
                * args.event_reward_scale
            )
            alive_probability = torch.sigmoid(-absorb_logits_buf)
            expected_reward = alive_probability * alive_expected_reward
            flat_reward_moment_target = reward_moment_target.reshape(-1)
            flat_reward_moment_valid = reward_moment_valid.reshape(-1)
            reward_moment_mae_preupdate = (
                (
                    expected_reward[flat_reward_moment_valid]
                    - flat_reward_moment_target[flat_reward_moment_valid]
                )
                .abs()
                .mean()
                if flat_reward_moment_valid.any()
                else expected_reward.new_zeros(())
            )
            manual_value = expected_reward / (1.0 - args.gamma)
            value_identity_maxerr = (
                manual_value - agent.distribution_value(current_dist)
            ).abs().max()
            target_reward_maxerr = (
                event_target.decode_reward(event_latents) - rewards
            ).abs().max()
            mixture_probs = torch.softmax(logits_buf, dim=-1)
            mixture_entropy = -(
                mixture_probs * mixture_probs.clamp_min(1e-12).log()
            ).sum(-1).mean()
            mixture_usage = mixture_probs.mean(0)
            mixture_usage_perplexity = torch.exp(
                -(mixture_usage * mixture_usage.clamp_min(1e-12).log()).sum()
            )
            reward_component_means = (
                means_buf[..., : args.event_reward_dims].mean(-1)
                * args.event_reward_scale
            )
            reward_between_var = (
                mixture_probs
                * (reward_component_means - alive_expected_reward.unsqueeze(-1)).square()
            ).sum(-1).mean()

        b_event_target = geometric_targets.reshape(-1, args.event_latent_dim)
        b_event_valid = geometric_valid.reshape(-1)
        b_event_absorbed = geometric_absorbed.reshape(-1)
        b_reward_moment_target = reward_moment_target.reshape(-1)
        b_reward_moment_valid = reward_moment_valid.reshape(-1)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
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

                _, _, newlogprob, entropy, value_dist = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
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
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                absorb_logit, mixture_logits, mixture_means = value_dist
                valid = b_event_valid[mb_inds]
                absorbed_target = b_event_absorbed[mb_inds]
                (
                    geometric_nll,
                    absorb_loss,
                    alive_nll,
                    alive_fraction,
                ) = factorized_event_nll(
                    absorb_logit,
                    mixture_logits,
                    mixture_means,
                    b_event_target[mb_inds],
                    valid,
                    absorbed_target,
                    args.mdn_fixed_std,
                )
                alive = valid & ~absorbed_target
                # Integrate the complete observed suffix to calibrate the reward moment.
                # This is the expectation of this exact mixture, not a scalar bypass head.
                predicted_reward_moment = (
                    (1.0 - args.gamma) * agent.distribution_value(value_dist)
                )
                moment_valid = b_reward_moment_valid[mb_inds]
                reward_moment_loss = (
                    F.smooth_l1_loss(
                        predicted_reward_moment[moment_valid],
                        b_reward_moment_target[mb_inds][moment_valid],
                    )
                    if moment_valid.any()
                    else predicted_reward_moment.sum() * 0.0
                )
                v_loss = (
                    geometric_nll
                    + args.reward_moment_coef * reward_moment_loss
                )
                with torch.no_grad():
                    shuffled_nll = (
                        mdn_nll(
                            mixture_logits[alive].roll(1, dims=0),
                            mixture_means[alive].roll(1, dims=0),
                            b_event_target[mb_inds][alive],
                            args.mdn_fixed_std,
                        ).mean()
                        if alive.any()
                        else alive_nll
                    )
                    conditional_nll_gain = shuffled_nll - alive_nll

                entropy_loss = entropy.mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
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

        # Train the reward-aware event autoencoder outside PPO. Its EMA target defines the
        # critic's coordinate frame and moves only after every likelihood target above has
        # been consumed, so a PPO epoch never chases a moving event representation.
        flat_content = content_buf.reshape(-1, content_dim)
        flat_reward = rewards.reshape(-1)
        event_recon_sum = event_grad_sum = 0.0
        event_steps = 0
        for _ in range(args.event_ae_epochs):
            permutation = torch.randperm(
                args.batch_size, generator=event_batch_generator
            )
            for start in range(0, args.batch_size - args.event_ae_batch + 1, args.event_ae_batch):
                index = permutation[start : start + args.event_ae_batch].to(device)
                latent = event_ae.encode(flat_reward[index], flat_content[index])
                reconstruction = event_ae.decode_content(latent)
                event_recon_loss = F.mse_loss(reconstruction, flat_content[index])
                event_optimizer.zero_grad(set_to_none=True)
                event_recon_loss.backward()
                event_grad = nn.utils.clip_grad_norm_(
                    event_ae.parameters(), args.event_ae_grad_clip
                )
                event_optimizer.step()
                event_recon_sum += event_recon_loss.item()
                event_grad_sum += float(event_grad)
                event_steps += 1
        with torch.no_grad():
            ema_update(event_target, event_ae, args.event_target_decay)
            event_probe = drift_probe_idx
            online_event_probe = event_ae.encode(
                flat_reward[event_probe], flat_content[event_probe]
            )
            target_event_probe = event_target.encode(
                flat_reward[event_probe], flat_content[event_probe]
            )
            event_teacher_online_cosine = F.cosine_similarity(
                online_event_probe[:, args.event_reward_dims :],
                target_event_probe[:, args.event_reward_dims :],
                dim=-1,
            ).mean()
        event_recon_mean = event_recon_sum / max(event_steps, 1)
        event_grad_mean = event_grad_sum / max(event_steps, 1)

        # ---- LeJEPA SSL step -----------------------------------------------------------
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
        probe_emb_before = emb_buf.reshape(-1, args.emb_dim)[drift_probe_idx].clone()
        for _ in range(args.ssl_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l = ssl(
                    seq_obs[idx], seq_act[idx], seq_mask[idx], args.sigreg_weight
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

        # Only now move the event-state teacher toward the fully updated SSL encoder.
        # Critic and AE targets for this iteration were all fitted in the old frozen frame.
        ema_update(event_state_target, ssl.encoder, args.event_state_target_decay)

        # ---- encoder health + frame drift ----------------------------------------------
        # SIGReg pins the DISTRIBUTION of e, not its coordinate frame: N(0,I) is
        # rotation-invariant and the prediction loss cannot pin the frame either (the
        # predictor co-rotates). The EMA event target and critic are functions of these
        # coordinates, so a drifting frame makes their targets stale. This is a non-stationarity the
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
            probe_next_obs = next_obses.reshape(flat_obs_shape)[drift_probe_idx]
            probe_action = actions.reshape(-1, act_dim_sf)[drift_probe_idx]
            probe_continuation = continuation.reshape(-1)[drift_probe_idx]
            probe_reward = rewards.reshape(-1)[drift_probe_idx]
            probe_target_state = event_state_target(probe_next_obs)
            probe_content_after = event_content(
                probe_target_state,
                probe_next_obs,
                probe_action,
                probe_continuation,
            )
            event_frame_probe_after = event_target.encode(
                probe_reward, probe_content_after
            )
            event_frame_rms_drift = (
                event_frame_probe_after - event_frame_probe_before
            ).square().mean().sqrt()
            event_frame_cosine = F.cosine_similarity(
                event_frame_probe_after[:, args.event_reward_dims :],
                event_frame_probe_before[:, args.event_reward_dims :],
                dim=-1,
            ).mean()
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/geometric_joint_nll", geometric_nll.item(), global_step)
        writer.add_scalar("losses/geometric_alive_nll", alive_nll.item(), global_step)
        writer.add_scalar("losses/absorbing_bce", absorb_loss.item(), global_step)
        writer.add_scalar("geometric/alive_fraction_valid", alive_fraction.item(), global_step)
        writer.add_scalar("losses/reward_moment", reward_moment_loss.item(), global_step)
        writer.add_scalar("losses/event_ae_reconstruction", event_recon_mean, global_step)
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
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        # ---- one-distribution identities, censoring, and conditional skill -------------
        writer.add_scalar("geometric/value_identity_maxerr", value_identity_maxerr.item(), global_step)
        writer.add_scalar("geometric/target_reward_maxerr", target_reward_maxerr.item(), global_step)
        writer.add_scalar("geometric/valid_fraction", geometric_valid.float().mean().item(), global_step)
        writer.add_scalar("geometric/censored_fraction", (~geometric_valid).float().mean().item(), global_step)
        writer.add_scalar("geometric/bootstrap_fraction", geometric_bootstrapped.float().mean().item(), global_step)
        writer.add_scalar("geometric/absorbed_fraction", geometric_absorbed.float().mean().item(), global_step)
        writer.add_scalar("geometric/sampled_horizon_mean", sampled_horizons.float().mean().item(), global_step)
        writer.add_scalar("geometric/sampled_horizon_max", sampled_horizons.max().item(), global_step)
        writer.add_scalar("geometric/conditional_nll_gain", conditional_nll_gain.item(), global_step)
        writer.add_scalar("geometric/mixture_entropy", mixture_entropy.item(), global_step)
        writer.add_scalar("geometric/mixture_usage_perplexity", mixture_usage_perplexity.item(), global_step)
        writer.add_scalar("geometric/absorbing_probability", torch.sigmoid(absorb_logits_buf).mean().item(), global_step)
        writer.add_scalar("geometric/reward_between_component_var", reward_between_var.item(), global_step)
        writer.add_scalar("geometric/reward_moment_mae_preupdate", reward_moment_mae_preupdate.item(), global_step)
        writer.add_scalar("geometric/reward_moment_valid_fraction", reward_moment_valid.float().mean().item(), global_step)
        writer.add_scalar("geometric/event_teacher_online_cosine", event_teacher_online_cosine.item(), global_step)
        writer.add_scalar("geometric/event_frame_rms_drift", event_frame_rms_drift.item(), global_step)
        writer.add_scalar("geometric/event_frame_cosine", event_frame_cosine.item(), global_step)
        writer.add_scalar("geometric/event_ae_grad_norm", event_grad_mean, global_step)
        writer.add_scalar("gate/ev_geometric_online", ev_geometric, global_step)
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)

        # ---- SSL path -------------------------------------------------------------------
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
