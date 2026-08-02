# PPO + LeJEPA full geometric successor-residual actor, v8.
#
# One state critic predicts the normalized full geometric successor mean M(s). Every
# rollout state uses its longest available same-episode suffix for supervision, with a
# frozen bootstrap only at a rollout or truncation edge and a zero tail at termination.
# The default actor directly uses the complete vector residual Y_suffix(s)-M(s), then
# contracts it with the lagged reward covector. This replaces GAE with the longest
# observed same-episode successor-feature target available at every state, while retaining
# the complete current-state baseline. A fixed 256-step vector Bellman trace is the
# matched control, selected with --credit-mode fixed. The critic and its frozen edge
# bootstrap retain the full gamma-geometric horizon in both modes.
# LeJEPA uses attached multi-horizon targets, action-trace conditioning, and SIGReg. Its
# aligned online embedding feeds both actor and critic. Analytic frame transport accepts
# representation updates in full or rejects them; there is no slowly updated copy or
# moving target statistic.
import copy
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
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
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
GEOMETRIC_GAMMA = 0.9970087504549047  # 95% mass lies in the first 1000 steps
LEJEPA_HORIZONS = (1, 2, 4, 8, 16, 32, 64)


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
    gamma: float = GEOMETRIC_GAMMA
    credit_mode: str = "full"        # "full" longest sampled suffix | "fixed" TD trace control
    credit_horizon: int = 256        # estimator unroll for credit_mode="fixed" only
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
    ret_perc_scope: str = "minibatch"  # "minibatch" or fresh whole-rollout "batch"
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    # Retained only as a fail-loud compatibility flag. Entropy-augmented temporal credit
    # would require adding entropy to the vector outcome basis.
    auto_entropy: bool = False
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # --- LeJEPA successor-feature value pathway -------------------------------------
    emb_dim: int = 32                # d. Obs manifold is 17-dim, so e is rank <= 17 regardless;
    #                                  expect effective rank ~17, which is NOT a failure signal.
    ssl_hidden: int = 256            # encoder / projector MLP width
    pred_depth: int = 2              # causal transformer depth
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = max(LEJEPA_HORIZONS) + 1
    sigreg_weight: float = 0.09      # lambda. NOTE: the Epps-Pulley statistic scales with batch
    #                                  size, so the reference value does not transfer verbatim.
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256     # exact (statistic is a mean over directions); bounds memory
    ssl_lr: float = 5e-5             # LeWM reference optimizer
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 128             # long sequences; keeps SIGReg and attention intermediates bounded
    ssl_epochs: int = 8
    ssl_grad_clip: float = 1.0       # own clip (reference lewm.yaml gradient_clip_val), fully
    #                                  separate from PPO's -- see ssl/grad_norm for the pre-clip value
    control_encoder_kl: float = 0.001
    control_encoder_emb_drift: float = 0.02
    control_encoder_sf_drift: float = 0.01
    control_encoder_value_drift: float = 0.01
    successor_scale_floor: float = 0.1  # fresh rollout coordinate scale
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping. "v10" is identity.
    #   "v10" | "tanh_adv" | "clip_z"
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


class LeJepaSSL(nn.Module):
    """Encoder + geometric action-conditioned predictors + SIGReg.

    Trained by EXACTLY two terms: a geometric embedding-prediction MSE with BOTH branches
    attached (no stop-gradient on the target -- SIGReg is what prevents collapse, and a
    stop-gradient target would re-introduce a second unanchored timescale), and SIGReg
    itself.

    Its only job is to define what the embedding e means. It never sees a reward, and it
    never receives gradient from the value path -- psi and w_r both read sg(e).
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
        self.pred_projs = nn.ModuleDict(
            {
                str(horizon): MLP(2 * args.emb_dim, args.ssl_hidden, args.emb_dim)
                for horizon in LEJEPA_HORIZONS
            }
        )
        self.action_trace_encoders = nn.ModuleDict(
            {
                str(horizon): nn.Conv1d(
                    args.emb_dim, args.emb_dim, kernel_size=horizon
                )
                for horizon in LEJEPA_HORIZONS
            }
        )
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk
        )

    def forward(self, obs_seq, act_seq, continuation_seq, sigreg_weight):
        """Predict attached future embeddings over geometric horizons.

        ``continuation_seq[:, t]`` says whether transition t -> t+1 stays in the same
        episode. A source-target pair is valid only when its full context prefix and
        intervening action trace do not cross a reset.
        """
        emb = self.encoder(obs_seq)                              # (N,L,d)
        act_emb = self.action_encoder(act_seq)                   # (N,L,d)
        context = self.predictor(emb, act_emb)                   # (N,L,d)
        horizon_losses = []
        for horizon in LEJEPA_HORIZONS:
            # A learned temporal kernel consumes exactly actions [t,t+h). Unlike a mean,
            # it distinguishes the order of stance/thrust phases. The final convolution
            # window begins at t=L-h and has no in-chunk target at t+h, so it is dropped.
            action_trace = self.action_trace_encoders[str(horizon)](
                act_emb.transpose(1, 2)
            ).transpose(1, 2)[:, :-1]
            pred = self.pred_projs[str(horizon)](
                torch.cat([context[:, :-horizon], action_trace], dim=-1)
            )
            err = (pred - emb[:, horizon:]).pow(2).mean(-1)
            # Each source needs only its own h intervening transitions. A prefix
            # cumprod would incorrectly discard valid pairs after an earlier reset in
            # the same chunk.
            valid = (
                continuation_seq.unfold(1, horizon, 1)
                .prod(-1)[:, :-1]
            )
            horizon_losses.append((err * valid).sum() / valid.sum().clamp_min(1.0))
        horizon_losses = torch.stack(horizon_losses)
        horizon_weights = horizon_losses.new_tensor(
            [horizon**-0.5 for horizon in LEJEPA_HORIZONS]
        )
        pred_loss = (horizon_weights * horizon_losses).sum() / horizon_weights.sum()
        # THE TRANSPOSE IS LOAD-BEARING. SIGReg reduces the empirical characteristic
        # function over dim -3 and scales by size(-2); both resolve to the BATCH only in
        # (T, B, D) layout. Passing (N, L, d) would average the CF over L=4 samples,
        # silently disabling collapse protection while still logging a plausible number.
        sigreg_loss = self.sigreg(emb.transpose(0, 1))           # (L, N, d)
        return (
            pred_loss + sigreg_weight * sigreg_loss,
            pred_loss,
            sigreg_loss,
            horizon_losses.detach(),
        )


def phi_features(emb, obs, action):
    """phi = [e, s, a, a*a, 1].

    The raw (already NormalizeObservation-scaled) state block is v2's whole change. See
    the header: without it a linear w_r cannot recover the velocity term from a whitened
    nonlinear latent, and v1 measured the resulting residual at lag-1 autocorrelation
    0.474 -- structured error, the expensive kind, injected into every value estimate.

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on e alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so e has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = emb[..., :1].new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, obs, action, action * action, ones], dim=-1)


def build_one_step_geometric_target(phi, next_mean, bootstrap, gamma):
    """Normalized one-step target for M(s)."""
    return (
        (1.0 - gamma) * phi
        + gamma * bootstrap.unsqueeze(-1) * next_mean
    )


def build_full_suffix_geometric_targets(
    phi, next_mean, bootstrap, continuation, gamma
):
    """Use every same-episode suffix; bootstrap only where observed data ends.

    `continuation[t]` selects the sampled target at t+1. At a reset edge it is zero:
    a valid truncation uses the frozen prediction at the supplied final observation,
    while a true termination has `bootstrap[t]==0` and therefore a zero tail.
    """
    targets = torch.empty_like(phi)
    running = torch.zeros_like(phi[0])
    for t in reversed(range(phi.shape[0])):
        if t == phi.shape[0] - 1:
            next_estimate = bootstrap[t].unsqueeze(-1) * next_mean[t]
        else:
            cont = continuation[t].unsqueeze(-1)
            boundary_bootstrap = (
                (1.0 - cont)
                * bootstrap[t].unsqueeze(-1)
                * next_mean[t]
            )
            next_estimate = cont * running + boundary_bootstrap
        running = (1.0 - gamma) * phi[t] + gamma * next_estimate
        targets[t] = running
    return targets


def fixed_horizon_sum(signal, continuation, gamma, horizon):
    """Discounted fixed-n sum that stops at reset boundaries.

    This is a direct finite sum, not a mixture over stopping times. Signal has shape
    (T,B,...); continuation[t]==0 prevents signal[t+1] from entering the sum at t.
    Rollout-tail rows naturally use shorter prefixes.
    """
    out = torch.zeros_like(signal)
    alive = torch.ones_like(continuation)
    for k in range(horizon):
        valid = signal.shape[0] - k
        if valid <= 0:
            break
        weight_shape = (valid, signal.shape[1]) + (1,) * (signal.ndim - 2)
        out[:valid] += (gamma**k) * alive.reshape(weight_shape) * signal[k:]
        if k + 1 < horizon and valid > 1:
            alive = alive[:-1] * continuation[k : k + valid - 1]
    return out


def same_episode_suffix_steps(continuation):
    """Number of sampled rewards in each full-suffix target before its edge."""
    available = torch.empty_like(continuation)
    running = torch.zeros_like(continuation[0])
    for t in reversed(range(continuation.shape[0])):
        running = 1.0 + continuation[t] * running
        available[t] = running
    return available


def align_latent_frame(source, reference):
    """Orthogonal Procrustes alignment with a translation for finite-sample means."""
    source_mean = source.mean(0)
    reference_mean = reference.mean(0)
    source_centered = source - source_mean
    reference_centered = reference - reference_mean
    u, _, vh = torch.linalg.svd(
        source_centered.T.double() @ reference_centered.double(),
        full_matrices=False,
    )
    rotation = (u @ vh).to(source.dtype)
    bias = reference_mean - source_mean @ rotation
    return rotation, bias, source @ rotation + bias


def affine_frame_transport(after, before, current_rotation, current_bias):
    """Compose a new raw-encoder frame with the prior aligned control frame."""
    after_mean, before_mean = after.mean(0), before.mean(0)
    cross = (after - after_mean).double().T @ (before - before_mean).double()
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    step = (u @ vh).to(after.dtype)
    rotation = step @ current_rotation
    bias = before_mean @ current_rotation + current_bias - after_mean @ rotation
    return rotation, bias


def augment_reward_residual(base_phi, reward, reward_covector):
    residual = reward - base_phi @ reward_covector
    return torch.cat([base_phi, residual.unsqueeze(-1)], dim=-1)


@torch.no_grad()
def transport_reward_gauge(critic_head, old_covector, new_covector):
    """Keep represented rewards fixed after changing the residual coordinate gauge."""
    delta = old_covector - new_covector
    residual_row = critic_head.weight.shape[0] - 1
    critic_head.weight[residual_row].add_(delta @ critic_head.weight[:-1])
    critic_head.bias[residual_row].add_(delta @ critic_head.bias[:-1])


@torch.no_grad()
def reset_optimizer_output_row(optimizer, layer):
    """Clear diagonal moments for the analytically transformed output row only."""
    row = layer.weight.shape[0] - 1
    for parameter in (layer.weight, layer.bias):
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment[row].zero_()


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


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


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
        agent_in_dim = obs_dim + args.emb_dim
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(agent_in_dim, H, args.k_blocks, args.n_experts)
        # One normalized full-horizon successor mean. The final coordinate is the
        # lagged reward-probe residual, making [w_r, 1] an exact reward covector.
        self.base_dim = args.emb_dim + obs_dim + 2 * act_dim + 1
        self.sf_dim = self.base_dim + 1  # [e, s, a, a*a, 1, reward residual]
        self.critic_head = layer_init(
            nn.Linear(H, self.sf_dim), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
            self.critic_head.bias.zero_()
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
        # Returns the normalized full geometric successor mean (B, sf_dim).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

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
        value_sf = self.critic_head(critic_feat)
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
        return action, z, log_prob, entropy, value_sf

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


def shape_advantage(raw_advantage, args, device):
    """Map raw geometric successor-residual credit according to args.adv_transform.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has no distributional critic to read -- the head
    predicts successor features. Feeding them a constant sigma=1 / u=0 placeholder would
    make cdf_probit return the same constant for every sample (a zero policy gradient
    after norm_adv) while logging a perfectly healthy-looking curve, so the branches are
    deleted and the arg is rejected at startup instead.
    """
    if args.adv_transform == "v10":
        return raw_advantage
    elif args.adv_transform == "tanh_adv":
        gz = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = raw_advantage.numel()
        ranks = raw_advantage.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed credit it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(raw_advantage)
        for side in (raw_advantage > 0, raw_advantage < 0):
            if side.any():
                g = raw_advantage[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (smaller kappa gives harder compression).
        n = raw_advantage.numel()
        ranks = raw_advantage.argsort().argsort().to(torch.float32)
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
        n = raw_advantage.numel()
        ranks = raw_advantage.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(raw_advantage) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if abs(args.gamma - GEOMETRIC_GAMMA) > 1e-12:
        raise ValueError(
            f"gamma must equal the geometric horizon decay ({GEOMETRIC_GAMMA:.16g}); "
            f"got {args.gamma}"
        )
    if args.seq_len != max(LEJEPA_HORIZONS) + 1:
        raise ValueError(
            f"seq_len must be {max(LEJEPA_HORIZONS) + 1} for the geometric LeJEPA grid"
        )
    if args.control_encoder_kl <= 0.0:
        raise ValueError("control_encoder_kl must be positive")
    if (
        args.control_encoder_emb_drift <= 0.0
        or args.control_encoder_sf_drift <= 0.0
        or args.control_encoder_value_drift <= 0.0
    ):
        raise ValueError("control encoder drift limits must be positive")
    if args.credit_mode not in ("full", "fixed"):
        raise ValueError("credit_mode must be full or fixed")
    if not 1 <= args.credit_horizon <= args.num_steps:
        raise ValueError("credit_horizon must be in [1, num_steps]")
    if args.successor_scale_floor <= 0.0:
        raise ValueError("successor_scale_floor must be positive")
    if args.ret_perc_scope not in ("minibatch", "batch"):
        raise ValueError("ret_perc_scope must be minibatch or batch")
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
            "DISTRIBUTION, which this variant does not have (the head predicts successor "
            "features, not bucket logits). Use v10 / tanh_adv / clip_z / rankgauss*."
        )
    if args.auto_entropy:
        raise ValueError("auto_entropy is not implemented for vector successor credit")
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # --- LeJEPA successor-feature value pathway -------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    # The online encoder is the control interface. A cumulative affine/orthogonal map
    # transports each newly learned raw frame back into the frame already consumed by
    # policy, critic, and reward probe.
    control_rotation = torch.eye(args.emb_dim, device=device)
    control_bias = torch.zeros(args.emb_dim, device=device)

    def encode_control(states):
        return ssl.encoder(states) @ control_rotation + control_bias

    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )

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

    reward_covector = torch.zeros(agent.base_dim, device=device)
    reward_task = torch.cat(
        [reward_covector, reward_covector.new_ones(1)]
    )

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
    ret_perc_scale = 1.0

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                next_emb = encode_control(next_obs)
                agent_obs = torch.cat([next_obs, next_emb], dim=-1)
                action, z, logprob, ent, value_sf = agent.get_action_and_value(agent_obs)
                values[step] = (value_sf @ reward_task) / (1.0 - args.gamma)
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
            flat_obs = obs.reshape(flat_obs_shape)
            flat_next_obs = next_obses.reshape(flat_obs_shape)
            emb_buf = encode_control(flat_obs).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            next_emb_buf = encode_control(flat_next_obs).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            flat_agent_obs = torch.cat([flat_obs, emb_buf.reshape(-1, args.emb_dim)], dim=-1)
            flat_next_agent_obs = torch.cat(
                [flat_next_obs, next_emb_buf.reshape(-1, args.emb_dim)], dim=-1
            )
            successor_next = agent.get_value(flat_next_agent_obs).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            critic_feat_buf = agent._trunks(flat_agent_obs)[1]
            successor_cur = agent.critic_head(critic_feat_buf).reshape(
                args.num_steps, args.num_envs, sf_dim
            )

            base_phi = phi_features(emb_buf, obs, actions)
            phi = augment_reward_residual(base_phi, rewards, reward_covector)
            reward_task = torch.cat(
                [reward_covector, reward_covector.new_ones(1)]
            )
            # This identity is exact by construction, including on the first rollout.
            if not torch.allclose(phi @ reward_task, rewards, atol=2e-5, rtol=2e-5):
                raise RuntimeError("reward residual coordinate lost exact reward identity")

            bootstrap = (1.0 - transition_terminations) * transition_valids
            continuation = 1.0 - transition_boundaries
            sf_target = build_full_suffix_geometric_targets(
                phi, successor_next, bootstrap, continuation, args.gamma
            )
            one_step_target = build_one_step_geometric_target(
                phi, successor_next, bootstrap, args.gamma
            )
            delta_sf = one_step_target - successor_cur
            full_vector_advantage = sf_target - successor_cur
            full_scalar_advantage = (
                full_vector_advantage @ reward_task
            ) / (1.0 - args.gamma)
            td0_scalar_advantage = (
                delta_sf @ reward_task
            ) / (1.0 - args.gamma)
            if args.credit_mode == "fixed":
                # The fixed control genuinely remains vector-valued through temporal
                # accumulation. This expensive trace is only materialized when selected.
                vector_advantage = fixed_horizon_sum(
                    delta_sf, continuation, args.gamma, args.credit_horizon
                )
                fixed_scalar_advantage = (
                    vector_advantage @ reward_task
                ) / (1.0 - args.gamma)
                advantages = fixed_scalar_advantage
            else:
                vector_advantage = full_vector_advantage
                advantages = full_scalar_advantage
                # Linearity makes this exactly the scalar contraction of the fixed vector
                # trace, without streaming 256 full successor tensors for a diagnostic.
                fixed_scalar_advantage = fixed_horizon_sum(
                    td0_scalar_advantage,
                    continuation,
                    args.gamma,
                    args.credit_horizon,
                )
            policy_adv = advantages
            returns = advantages + values

            _full = full_scalar_advantage.reshape(-1)
            _fixed = fixed_scalar_advantage.reshape(-1)
            _full_centered = _full - _full.mean()
            _fixed_centered = _fixed - _fixed.mean()
            full_fixed_corr = float(
                (_full_centered * _fixed_centered).mean()
                / (
                    _full_centered.square().mean().sqrt().clamp_min(1e-12)
                    * _fixed_centered.square().mean().sqrt().clamp_min(1e-12)
                )
            )
            full_fixed_std_ratio = float(
                _full.std(unbiased=False) / _fixed.std(unbiased=False).clamp_min(1e-12)
            )
            selected_coordinate_rms = vector_advantage.square().mean((0, 1)).sqrt()
            _a, _b = advantages.reshape(-1), td0_scalar_advantage.reshape(-1)
            selected_td0_corr = float(
                ((_a - _a.mean()) * (_b - _b.mean())).mean()
                / (_a.std().clamp_min(1e-12) * _b.std().clamp_min(1e-12))
            )

            if args.ret_percnorm and args.ret_perc_scope == "batch":
                flat_ret = returns.reshape(-1)
                qs = torch.tensor(
                    [args.ret_perc_lo, args.ret_perc_hi], device=device
                )
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                ret_perc_scale = max(args.ret_perc_floor, hi - lo)
                policy_adv = policy_adv / ret_perc_scale

            flat_base_phi = base_phi.reshape(-1, agent.base_dim)
            flat_rew = rewards.reshape(-1)
            new_reward_covector = solve_reward_probe(
                flat_base_phi, flat_rew, args.sf_ridge
            )
            reward_resid = flat_rew - flat_base_phi @ new_reward_covector
            reward_r2 = 1.0 - (
                reward_resid.var() / flat_rew.var().clamp_min(1e-12)
            ).item()
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            reward_resid_ac = [
                ((resid_c[:-k] * resid_c[k:]).mean() / resid_var).item()
                for k in (1, 5, 10, 20)
            ]

            # Targets are normalized O(1); a fresh rollout-only coordinate scale
            # balances heterogeneous coordinates without adding temporal state.
            sf_loss_scale = (
                sf_target.reshape(-1, sf_dim)
                .std(0, unbiased=False)
                .clamp_min(args.successor_scale_floor)
            )
            vector_whitened_rms = float(
                (vector_advantage / sf_loss_scale)
                .square()
                .mean()
                .sqrt()
            )
            # The snapshot predates this rollout, so this is a pre-update TD skill
            # diagnostic rather than in-sample fit. It is not a fully independent outcome
            # score because rollout/truncation edges bootstrap from the same snapshot.
            preupdate_scaled_mse = (
                (successor_cur - sf_target) / sf_loss_scale
            ).square().mean()
            marginal_target = sf_target.reshape(-1, sf_dim).mean(0)
            marginal_scaled_mse = (
                (sf_target - marginal_target) / sf_loss_scale
            ).square().mean()
            preupdate_skill = (
                1.0
                - preupdate_scaled_mse
                / marginal_scaled_mse.clamp_min(1e-12)
            )

            mc_ret = torch.zeros_like(rewards)
            mc_avail = same_episode_suffix_steps(continuation)
            mc_run = torch.zeros_like(rewards[0])
            mc_resid = torch.zeros_like(rewards)
            resid_run = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                cont = continuation[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                resid_run = resid_tb[t] + args.gamma * cont * resid_run
                mc_ret[t], mc_resid[t] = mc_run, resid_run
            suffix_steps = mc_avail
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())
            if n_mc >= 256:
                flat_mc = mc_ret.reshape(-1)[mc_mask]
                feat_mc = critic_feat_buf[mc_mask]
                ones_mc = feat_mc.new_ones(feat_mc.shape[0], 1)
                trunk_feat_mc = torch.cat([feat_mc, ones_mc], dim=-1)
                ev_trunk_probe = ev_score(
                    trunk_feat_mc
                    @ solve_reward_probe(
                        trunk_feat_mc, flat_mc, args.sf_ridge
                    ),
                    flat_mc,
                )
                e_mc = torch.cat(
                    [
                        emb_buf.reshape(-1, args.emb_dim)[mc_mask],
                        ones_mc,
                    ],
                    dim=-1,
                )
                ev_latent_cap = ev_score(
                    e_mc
                    @ solve_reward_probe(e_mc, flat_mc, args.sf_ridge),
                    flat_mc,
                )
                value_err_frac = float(
                    mc_resid.reshape(-1)[mc_mask].std()
                    / flat_mc.std().clamp_min(1e-12)
                )
                ev_sf = ev_score(values.reshape(-1)[mc_mask], flat_mc)
            else:
                ev_trunk_probe = ev_latent_cap = ev_sf = value_err_frac = float(
                    "nan"
                )
        b_obs = flat_agent_obs
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_target.reshape(-1, sf_dim)
        # Policy advantage: shape selected vector credit per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        raw_advantage = b_advantages
        b_policy_adv = shape_advantage(raw_advantage, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from raw credit?
        az = (raw_advantage - raw_advantage.mean()) / (
            raw_advantage.std() + 1e-8
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

                _, _, newlogprob, entropy, value_sf = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipped = (ratio < 1.0 - args.clip_coef) | (
                        ratio > 1.0 + args.clip_coef_high
                    )
                    clipfracs.append(clipped.float().mean().item())

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
                # returns, recomputed fresh each minibatch.
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

                # The sole critic head predicts one complete normalized successor mean.
                sf_tgt = b_sf_target[mb_inds]
                sf_raw_err = value_sf - sf_tgt
                sf_err = sf_raw_err / sf_loss_scale
                v_loss = sf_err.pow(2).mean()
                sf_mse = sf_err.detach().pow(2).mean()
                sf_raw_mse = sf_raw_err.detach().pow(2).mean()

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

        # Change the reward-residual coordinate only after every target and gradient that
        # used the lagged gauge is finished. The analytic output-row transform preserves
        # the represented scalar value exactly.
        transport_reward_gauge(
            agent.critic_head, reward_covector, new_reward_covector
        )
        reset_optimizer_output_row(optimizer, agent.critic_head)
        reward_covector = new_reward_covector
        reward_task = torch.cat(
            [reward_covector, reward_covector.new_ones(1)]
        )

        # ---- LeJEPA SSL step -----------------------------------------------------------
        # ONCE PER ITERATION, OUTSIDE the 320-minibatch PPO loop: inside it costs +100-200%
        # wall clock, outside it costs +10-20%. Placed AFTER the PPO update so the encoder
        # frame is frozen across the entire target-construction + critic-fitting phase. The
        # residual drift is one ITERATION's worth (ssl_epochs * (n_seq // ssl_batch) = 64
        # steps at defaults, not one), which is exactly what ssl/frame_drift_* measures.
        with torch.no_grad():
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
        n_seq = seq_obs.shape[0]
        ssl_pred_l = ssl_sig_l = ssl_gn_sum = 0.0
        ssl_horizon_l = torch.zeros(len(LEJEPA_HORIZONS), device=device)
        ssl_steps = 0
        # e BEFORE this iteration's SSL step, on states from THIS rollout. Paired with the
        # post-step embedding of the SAME inputs below, this isolates frame movement from
        # distribution shift without needing to keep a copy of the old encoder.
        probe_obs = obs.reshape(flat_obs_shape)[drift_probe_idx]
        probe_control_before = emb_buf.reshape(-1, args.emb_dim)[drift_probe_idx].clone()
        with torch.no_grad():
            probe_online_before = ssl.encoder(probe_obs).clone()
        ssl_state_before = copy.deepcopy(ssl.state_dict())
        ssl_optimizer_state_before = copy.deepcopy(ssl_optimizer.state_dict())
        for _ in range(args.ssl_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l, horizon_l = ssl(
                    seq_obs[idx], seq_act[idx], seq_cont[idx], args.sigreg_weight
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_gn = nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_optimizer.step()
                ssl_pred_l += pred_l.item()
                ssl_sig_l += sig_l.item()
                ssl_horizon_l += horizon_l
                ssl_gn_sum += float(ssl_gn)
                ssl_steps += 1
        ssl_pred_l /= max(ssl_steps, 1)
        ssl_sig_l /= max(ssl_steps, 1)
        ssl_horizon_l /= max(ssl_steps, 1)
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
            probe_emb_after = ssl.encoder(probe_obs).float()
            a = probe_online_before - probe_online_before.mean(0)
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

            old_agent_probe = torch.cat(
                [probe_obs, probe_control_before], dim=-1
            )
            old_actor_feat, old_critic_feat = agent._trunks(old_agent_probe)
            old_policy = agent._actor_dist(old_actor_feat)[0]
            old_successors = agent.critic_head(old_critic_feat)
            old_probe_values = (
                old_successors @ reward_task
            ) / (1.0 - args.gamma)

            candidate_rotation, candidate_bias = affine_frame_transport(
                probe_emb_after,
                probe_online_before,
                control_rotation,
                control_bias,
            )
            probe_control_after = (
                probe_emb_after @ candidate_rotation + candidate_bias
            )
            new_actor_feat, new_critic_feat = agent._trunks(
                torch.cat([probe_obs, probe_control_after], dim=-1)
            )
            new_policy = agent._actor_dist(new_actor_feat)[0]
            encoder_policy_kl = kl_divergence(
                old_policy, new_policy
            ).sum(-1).mean()
            encoder_emb_drift = (
                (probe_control_after - probe_control_before)
                .pow(2)
                .mean()
                .sqrt()
                / probe_control_before.pow(2).mean().sqrt().clamp_min(1e-6)
            )
            new_successors = agent.critic_head(new_critic_feat)
            encoder_sf_drift = (
                ((new_successors - old_successors) / sf_loss_scale)
                .pow(2)
                .mean()
                .sqrt()
            )
            new_probe_values = (
                new_successors @ reward_task
            ) / (1.0 - args.gamma)
            encoder_value_drift = (
                (new_probe_values - old_probe_values).pow(2).mean().sqrt()
                / old_probe_values.std().clamp_min(1.0)
            )
            accepted = bool(
                encoder_policy_kl <= args.control_encoder_kl
                and encoder_emb_drift <= args.control_encoder_emb_drift
                and encoder_sf_drift <= args.control_encoder_sf_drift
                and encoder_value_drift <= args.control_encoder_value_drift
            )
            if accepted:
                control_rotation = candidate_rotation
                control_bias = candidate_bias
            else:
                ssl.load_state_dict(ssl_state_before)
                ssl_optimizer.load_state_dict(ssl_optimizer_state_before)
                probe_control_after = probe_control_before

            ca = probe_control_before - probe_control_before.mean(0)
            cb = probe_control_after - probe_control_after.mean(0)
            control_denom = (
                0.5 * (ca.pow(2).sum() + cb.pow(2).sum())
            ).clamp_min(1e-12)
            control_drift_raw = float(
                1.0
                - (cb - ca).pow(2).sum() / (2.0 * control_denom)
            )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaping adds little; lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_raw_adv", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        # ---- successor-feature diagnostics -------------------------------------
        # R^2 is the weak gate; a structural ~1-3% residual is expected (frame-skip
        # averaged reward velocity vs instantaneous qvel). The residual's
        # AUTOCORRELATION is what actually costs EV: white residual at R^2=0.98
        # costs ~0.2% EV, a gait-phase-locked one at the same R^2 costs ~2%.
        writer.add_scalar("sf/value_err_frac", value_err_frac, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        for lag, ac in zip((1, 5, 10, 20), reward_resid_ac):
            writer.add_scalar(f"sf/reward_resid_ac_lag{lag}", ac, global_step)
        writer.add_scalar("geometric/mse_scaled", sf_mse.item(), global_step)
        writer.add_scalar("geometric/mse_raw", sf_raw_mse.item(), global_step)
        writer.add_scalar(
            "geometric/preupdate_skill", preupdate_skill.item(), global_step
        )
        writer.add_scalar(
            "geometric/target_rms",
            sf_target.pow(2).mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/coordinate_scale_mean",
            sf_loss_scale.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/coordinate_scale_max",
            sf_loss_scale.max().item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/effective_horizon",
            1.0 / (1.0 - args.gamma),
            global_step,
        )
        writer.add_scalar(
            "geometric/mass_first_1000",
            1.0 - args.gamma**1000,
            global_step,
        )
        writer.add_scalar(
            "sf/reward_covector_norm",
            reward_covector.norm().item(),
            global_step,
        )

        # Full-suffix vector credit diagnostics and fixed-horizon control comparison.
        writer.add_scalar(
            "tddelta/selected_td0_corr", selected_td0_corr, global_step
        )
        writer.add_scalar(
            "tddelta/reward_residual_std",
            phi[..., -1].std().item(),
            global_step,
        )
        writer.add_scalar(
            "tddelta/vector_advantage_std",
            advantages.std().item(),
            global_step,
        )
        writer.add_scalar(
            "tddelta/full_fixed_scalar_corr",
            full_fixed_corr,
            global_step,
        )
        writer.add_scalar(
            "tddelta/full_to_fixed_scalar_std",
            full_fixed_std_ratio,
            global_step,
        )
        writer.add_scalar(
            "tddelta/vector_coordinate_rms_mean",
            selected_coordinate_rms.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "tddelta/vector_coordinate_rms_max",
            selected_coordinate_rms.max().item(),
            global_step,
        )
        writer.add_scalar(
            "tddelta/vector_whitened_rms",
            vector_whitened_rms,
            global_step,
        )
        writer.add_scalar(
            "geometric/sampled_horizon_mean",
            suffix_steps.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/sampled_horizon_p50",
            torch.quantile(suffix_steps, 0.5).item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/sampled_horizon_p90",
            torch.quantile(suffix_steps, 0.9).item(),
            global_step,
        )
        writer.add_scalar(
            "geometric/sampled_horizon_max",
            suffix_steps.max().item(),
            global_step,
        )

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        writer.add_scalar("gate/ev_latent_cap", ev_latent_cap, global_step)    # (3) ceiling
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)

        # ---- SSL path -------------------------------------------------------------------
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        for horizon, loss_value in zip(LEJEPA_HORIZONS, ssl_horizon_l):
            writer.add_scalar(
                f"ssl/pred_loss_horizon_{horizon}", loss_value.item(), global_step
            )
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        writer.add_scalar("ssl/control_encoder_policy_kl", encoder_policy_kl.item(), global_step)
        writer.add_scalar("ssl/control_encoder_emb_drift", encoder_emb_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_sf_drift", encoder_sf_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_value_drift", encoder_value_drift.item(), global_step)
        writer.add_scalar("ssl/control_encoder_update_accepted", float(accepted), global_step)
        writer.add_scalar(
            "ssl/control_encoder_update_rate", float(accepted), global_step
        )
        writer.add_scalar("ssl/control_encoder_drift_raw", control_drift_raw, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
