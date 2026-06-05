# PPO + IterThink v31 (TD-FLOW CRITIC — common-random-numbers value + diagnostics). From v30.
#
# v30 fixed the velocity-net architecture but the policy STILL didn't learn. Two reasons,
# addressed here: (1) the flow VALUE was a fresh-noise Monte-Carlo estimate amplified 100x by
# 1/(1-g); since GAE's advantage contains -V(s_t) directly, that per-state value noise (std
# ~tens) swamped the O(1) reward signal and rankgauss ordered mostly noise. v31 uses COMMON
# RANDOM NUMBERS: a fixed K-sample noise basis (refreshed once per iteration) shared across all
# states, so V is a deterministic function of s and delta=r+gV(s')-V(s) measures true value
# DIFFERENCES (K bumped 16->64). (2) flow_loss was logged only as the (1-g)/g-weighted total,
# which hid whether the flow was learning; v31 logs flow_direct, flow_boot and value_std
# separately. TD2-CFM loss, decoupled flow encoder, and value readout are otherwise unchanged.
#
# v29 RAN but the flow loss stayed FLAT (~0.36) -> useless near-constant value -> policy
# barely learned. Root cause (vs paper arXiv 2503.09817 + working repo v120) was the velocity
# NET, not the TD2-CFM algorithm: (1) time was a raw Linear(1,h) rank-1 ramp that cannot
# represent a t-dependent field; (2) the conditioning carried gradient into the shared,
# policy-driven trunk, sabotaging the flow's representation; (3) the net was additive /
# Tanh / norm-free. v30 fixes all three: SINUSOIDAL time embedding -> nonlinear MLP, a
# DEDICATED observation encoder (flow fully decoupled from the policy trunk), CONCATENATED
# conditioning, SiLU + LayerNorm, and bootstrap integration steps 4 -> 8. The validated TD2-CFM
# loss and the 1/(1-g) successor-measure value readout are UNCHANGED.
#
# Replaces the distributional categorical critic with a Temporal-Difference-Flows critic
# (arXiv 2503.09817): a rectified-flow velocity net models the successor measure m^pi(.|s)
# (the discounted-future-OBSERVATION distribution), trained by a TD2 conditional-flow-matching
# loss = (1-g)*direct (flow toward the immediate next obs) + g*bootstrap (match an EMA target
# flow's velocity at s_{t+1}). The value is read off as V(s) = 1/(1-g) * E_{s'~m(.|s)}[r_hat(s')]
# by sampling future observations from the flow and scoring them with a learned reward model
# r_hat (regressed to observed rewards). NO categorical value head and NO value-regression loss
# — the trunk's value representation is shaped solely by the flow-matching objective. GAE, the
# beta policy, rankgauss, clip-higher and target_kl are unchanged; a triple decoupled grad clip
# (flow / reward / policy) keeps the flow from swamping the shared trunk.
# From v24:
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal Beta, exactly dreamer4's continuous_dist_type='beta' (which
#       forces unimodal=True) and our beta_relusq:
#           alpha = 1 + softplus(head_a);  beta = 1 + softplus(head_b)   (>=1 => unimodal)
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = "direct log std" done right):
#       dreamer4's Gaussian readout. A state-dependent log-VARIANCE head (not a
#       flat Parameter, not log-std), SOFT-bounded by dreamer4's tanh-rescale
#       (not a hard clamp, so the gradient never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink tanh-squash + SAC Jacobian on the sample (mean stays
#       raw), base-Normal entropy. (#1 soft-clamp + #2 log-var from the dreamer4
#       parity review; the standing entropy bonus #3 was judged not relevant.)
#
# PARITY NOTES (both dists): the rollout buffers the distribution-NATIVE sample
# (latent_zs) — pre-tanh z for gaussian, z in (0,1) for beta — and replays it on
# the update pass, so log_prob is recomputed at the same sample (identical to
# v21's z-replay). `actions` holds the env action (tanh(z) / rescaled z). The
# gaussian path is bit-identical to v21 except the flat logstd -> dreamer4 head.
# Bar to beat: v21 flat-Gaussian = 8774.
#
# --- inherited v21 notes ---
# PPO + IterThink v21 (SHARED BACKBONE + DECOUPLED GRAD CLIP). From v19.
#
# WHY v21. v19 used two independent ThinkTrunks (one actor, one critic). The
# classic MuJoCo-PPO result is that shared backbones LOSE, because the value
# loss gradient dominates the shared trunk and corrupts the policy's features.
# v21 tests whether we can have the representation-sharing benefit WITHOUT that
# cost, by decoupling the gradient magnitudes:
#   - share_backbone: one ThinkTrunk feeds both the actor head and the
#     (distributional) critic head; trunk is computed once per forward.
#   - separate_grad_clip: DUAL-BACKWARD clipping. The value gradient
#     (vf_coef * v_loss) and the policy gradient (pg_loss - ent) are each
#     backpropped and clipped to their OWN max-norm (critic_grad_clip /
#     actor_grad_clip), then summed on the shared trunk:
#         trunk.grad = clip_actor(d pg / d trunk) + clip_critic(d vl / d trunk)
#     so the distributional critic's large CE gradient can no longer swamp the
#     shared features. NOTE: the trunk's effective budget is the SUM of the two
#     clips, so each defaults to 0.25 (sum ~= v19's single 0.5 global clip).
# This is targeted: rankgauss already bounds the POLICY gradient (rank-only adv),
# so the dominant imbalance on a shared trunk is the critic -> clip it apart.
# Built on the v19 winner: adv_transform="rankgauss" + clip-higher (0.2/0.28).
# Both knobs are toggles, so this file also runs the {shared,separate} x
# {global,decoupled-clip} 2x2. The bar to beat: rankgauss_cliphigh ~= 8292 (towers).
#
# --- inherited v19 notes ---
# PPO + IterThink v19 (ADVANTAGE SHAPING — magnitude-preserving + attribution). From v17.
#
# WHY v19. A subagent review of v17 (CDF-rank distributional PG) found that in its
# STABLE regime the categorical critic is overconfident, so u=F_Z(G) is bimodal at
# 0/1, the probit saturates, and the advantage DEGENERATES to ≈sign(GAE) (corr 0.92);
# norm_adv then re-standardizes the ±3.3 spikes to ≈±1 binary. So v17 discards the
# advantage MAGNITUDE (the thing PPO needs) and is really a sign-of-TD-error update
# made trainable by KL control. v17's 5867@4M conflates THREE possible causes — the
# distribution, a bounded/outlier-robust advantage, and KL control — introduced at
# once. v19 disentangles them and adds the principled fix, via one `adv_transform`:
#
#   "v10"      : raw GAE (== v10 / dist_pg off). Baseline.
#   "cdf_probit": v17's CDF-rank u -> Phi^-1(u). Reference.
#   "tanh_std" : A~ = tanh( GAE_t / (kappa * sigma(s_t)) ).  THE FIX. Per-state
#                normalized by the critic's return std sigma(s) (v16's good idea),
#                but BOUNDED by tanh (fixes v16's blowup: tiny sigma -> saturate, not
#                explode) AND magnitude-preserving near 0 (fixes v17's sign-collapse:
#                linear in GAE for |GAE|<kappa*sigma). Note G_t-E[Z_t]=GAE_t exactly.
#   "tanh_gae" : A~ = tanh( zscore(GAE)_t / kappa ).  Robust-GAE CONTROL with NO
#                distribution — isolates "bounded/outlier-robust advantage" from the
#                distributional claim. If this matches v17, the distribution is
#                incidental and this is the cleaner lever.
#
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
import os
import copy
import math
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

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def sinusoidal_time_embed(t, dim):
    """Sinusoidal features for a continuous flow-time t in [0,1]. t:(B,1) -> (B,dim).
    Geometric frequencies from ~1 to ~128 cycles over the unit interval so the velocity
    net can resolve a SHARPLY t-dependent field. (v29's single Linear(1,h) mapped t to a
    rank-1 affine ramp, which cannot represent v(x,t) -> the flow loss never moved.)"""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(2.0 * math.pi), math.log(2.0 * math.pi * 128.0),
                                     half, device=t.device))
    args = t * freqs  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowVelocity(nn.Module):
    """Rectified-flow velocity v(x_t, t | s) for the TD-flow successor measure over
    OBSERVATIONS. Proven recipe (paper arXiv 2503.09817 + repo v120): SINUSOIDAL time
    embedding through a nonlinear MLP, a DEDICATED observation encoder (independent of the
    policy trunk, so the flow rides a stable representation), CONCATENATED conditioning,
    SiLU activations and LayerNorm. Replaces v29's single-Linear time ramp + additive,
    norm-free, Tanh net that could not represent a t-dependent field (flat flow loss)."""
    def __init__(self, obs_dim, hidden, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            layer_init(nn.Linear(t_dim, hidden)), nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
        )
        self.cond_enc = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
        )
        self.x_in = layer_init(nn.Linear(obs_dim, hidden))
        self.in_norm = nn.LayerNorm(3 * hidden)
        self.body = nn.Sequential(
            layer_init(nn.Linear(3 * hidden, hidden)), nn.SiLU(), nn.LayerNorm(hidden),
            layer_init(nn.Linear(hidden, hidden)), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.out = layer_init(nn.Linear(hidden, obs_dim), std=0.1)

    def forward(self, x, t, cond_obs):
        te = self.t_mlp(sinusoidal_time_embed(t, self.t_dim))
        ce = self.cond_enc(cond_obs)
        xe = self.x_in(x)
        h = self.in_norm(torch.cat([xe, te, ce], dim=-1))
        return self.out(self.body(h))


def integrate_flow(net, cond, x0, end_t, steps):
    """Euler-midpoint integrate dx/dtau = net(x, tau, cond) from tau=0 to per-sample end_t (B,1)."""
    x = x0
    dt = end_t / steps
    for i in range(steps):
        x = x + dt * net(x, (i + 0.5) * dt, cond)
    return x


def tdflow_loss(net, target_net, obs, next_obs, mask, gamma, steps):
    """TD2 rectified conditional-flow-matching loss (arXiv 2503.09817) for the discounted-
    future-observation successor measure. Conditioned on RAW observations (the flow has its
    own encoder): the direct term conditions the online field on the current state `obs` and
    flow-matches toward the immediate next state `next_obs` (coupled straight-line velocity);
    the bootstrap conditions the EMA-target field on `next_obs` (S') and the online field on
    `obs` (S). Fully independent of the policy trunk -> stable conditioning signal."""
    B = next_obs.shape[0]
    t = torch.rand(B, 1, device=next_obs.device)
    x0 = torch.randn_like(next_obs)
    xt = (1.0 - t) * x0 + t * next_obs
    direct = ((net(xt, t, obs) - (next_obs - x0)) ** 2).sum(-1)
    with torch.no_grad():
        b0 = torch.randn_like(next_obs)
        bxt = integrate_flow(target_net, next_obs, b0, t, steps)
        btgt = target_net(bxt, t, next_obs)
    boot = ((net(bxt, t, obs) - btgt) ** 2).sum(-1)
    loss = (1.0 - gamma) * direct + gamma * boot
    denom = mask.sum().clamp_min(1.0)
    # also return the (masked-mean) direct/boot components for diagnostics
    return (
        (loss * mask).sum() / denom,
        ((direct * mask).sum() / denom).detach(),
        ((boot * mask).sum() / denom).detach(),
    )


@torch.no_grad()
def ema_update(target_net, online_net, decay):
    for pt, po in zip(target_net.parameters(), online_net.parameters()):
        pt.mul_(decay).add_(po, alpha=1.0 - decay)


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
    norm_adv: bool = True
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    # v29 TD-FLOW CRITIC (arXiv 2503.09817): value comes ENTIRELY from a successor-measure
    # flow over observations + a learned reward model. No categorical value head.
    flow_gamma: float = 0.99          # successor-measure horizon; matches GAE gamma
    flow_hidden: int = 256            # velocity-net hidden width
    flow_t_dim: int = 128             # sinusoidal flow-time embedding dimension
    flow_steps: int = 8               # Euler steps for the TD bootstrap integration
    flow_value_steps: int = 8         # Euler steps when sampling future obs for V(s)
    flow_value_samples: int = 64      # # future-obs samples (common random numbers) averaged for V(s)
    flow_target_decay: float = 0.999  # EMA decay of the target velocity net
    flow_grad_clip: float = 0.5       # max-norm for the flow group (trunk + flow_net)
    reward_grad_clip: float = 0.5     # max-norm for the reward-model group
    flow_coef: float = 1.0            # weight of the TD-flow loss
    reward_coef: float = 1.0          # weight of the reward-regression loss
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 direct-log-std: state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
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
        # v24: action distribution. Both parameterizations are dreamer4-faithful.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 direct-log-std: mean head + state-dependent log-VARIANCE head.
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
        # v29 TD-FLOW CRITIC. Value = 1/(1-g) * E_{s'~m(.|s)}[ r_hat(s') ], no categorical head.
        self.flow_gamma = args.flow_gamma
        self.flow_net = FlowVelocity(obs_dim, args.flow_hidden, t_dim=args.flow_t_dim)
        self.reward_model = nn.Sequential(
            layer_init(nn.Linear(obs_dim, H)), nn.Tanh(),
            layer_init(nn.Linear(H, H)), nn.Tanh(),
            layer_init(nn.Linear(H, 1), std=1.0),
        )
        # Common-random-numbers noise basis for the flow value: K fixed source samples shared
        # across all states. Refreshed in-place once per training iteration (see main loop) so
        # V(s) is a deterministic function of s within an iteration.
        self.register_buffer("value_noise", torch.randn(args.flow_value_samples, obs_dim))

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

    def flow_value(self, x, steps):
        """V(s) via the successor-measure flow with COMMON RANDOM NUMBERS: every state is
        integrated from the SAME fixed noise basis `self.value_noise` (refreshed once per
        iteration), so V is a deterministic function of s within an iteration. This makes
        GAE's delta = r + gamma*V(s') - V(s) reflect true value DIFFERENCES rather than two
        independent Monte-Carlo draws, whose 1/(1-gamma)=100x-amplified noise otherwise
        swamps the O(1) reward signal in every advantage."""
        z = self.value_noise                         # (K, d) fixed this iteration
        K = z.shape[0]
        B = x.shape[0]
        cond_rep = x.repeat_interleave(K, dim=0)      # (B*K, d): [s0xK, s1xK, ...]
        z_rep = z.repeat(B, 1)                        # (B*K, d): [z0..z_{K-1}] tiled per state
        end = torch.ones(B * K, 1, device=x.device)
        s_future = integrate_flow(self.flow_net, cond_rep, z_rep, end, steps)
        r = self.reward_model(s_future).squeeze(-1).view(B, K).mean(dim=1)
        return r / (1.0 - self.flow_gamma)

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
        return action, z, log_prob, dist.entropy().sum(1)

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
        # Params receiving the VALUE gradient (incl. the shared trunk). v29 has no
        # categorical critic head; the value signal reaches the trunk via the flow loss.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, value_probs, bootstrap_probs, support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·nonterm·[ (1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1} ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones (T, B);
    value_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(mix, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    flow_target = copy.deepcopy(agent.flow_net)
    for p in flow_target.parameters():
        p.requires_grad_(False)
    flow_params = list(agent.flow_net.parameters())  # flow is fully decoupled from the policy trunk
    reward_params = list(agent.reward_model.parameters())

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    next_obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    flow_mask_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Refresh the common-random-numbers basis once per iteration so V(s) is a deterministic
        # function of s for this whole rollout+update (rollout values, GAE bootstrap, and any
        # value reads all share these K source samples).
        agent.value_noise.normal_()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, _ = agent.get_action_and_value(next_obs)
                values[step] = agent.flow_value(next_obs, args.flow_value_steps).flatten()
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_obs_buf[step] = next_obs          # s_{t+1}
            flow_mask_buf[step] = 1.0 - next_done  # mask transitions that crossed an episode boundary

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.flow_value(next_obs, args.flow_value_steps).reshape(1, -1)
            # Scalar GAE (means) — advantage baseline is unchanged from v7.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)            # raw GAE (mean-value)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_next_obs = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_flow_mask = flow_mask_buf.reshape(-1)
        b_rewards = rewards.reshape(-1)
        # Policy advantage: reshape the GAE / distribution per `adv_transform`.
        gae = b_advantages
        if args.adv_transform == "v10":
            b_policy_adv = gae
        elif args.adv_transform == "tanh_std":
            # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
            b_policy_adv = torch.tanh(gae / (args.tanh_kappa * b_sigma))
        elif args.adv_transform == "tanh_gae":
            gz = (gae - gae.mean()) / (gae.std() + 1e-8)
            b_policy_adv = torch.tanh(gz / args.tanh_kappa)
        elif args.adv_transform == "cdf_probit":
            centered = (2.0 * u.reshape(-1) - 1.0)
            c = args.cdf_probit_clamp
            b_policy_adv = ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
        elif args.adv_transform == "clip_z":
            # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
            # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
            # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
            gz = (gae - gae.mean()) / (gae.std() + 1e-8)
            b_policy_adv = gz.clamp(-args.clip_z_c, args.clip_z_c)
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
            b_policy_adv = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        elif args.adv_transform == "rankgauss_signed":
            # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
            # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
            # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
            # wrong sign to near-median samples — and PPO needs the sign right. Each sign
            # group is mapped to its half of the Gaussian by its within-group rank.
            c = args.cdf_probit_clamp
            b_policy_adv = torch.zeros_like(gae)
            for side in (gae > 0, gae < 0):
                if side.any():
                    g = gae[side]
                    r = g.argsort().argsort().to(torch.float32)
                    half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                    uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                    ctr = (2.0 * uq - 1.0).clamp(-c, c)
                    b_policy_adv[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
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
            b_policy_adv = torch.tanh(z / args.rank_tanh_kappa)
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
            b_policy_adv = torch.sign(gae) * mag
        else:
            raise ValueError(f"unknown adv_transform {args.adv_transform}")
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

                _, newz, newlogprob, entropy = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

                entropy_loss = entropy.mean()

                # v30 TD-FLOW CRITIC loss. Flow conditions on RAW observations via its own
                # encoder (decoupled from the policy trunk) -> a fully independent graph.
                flow_loss, flow_direct, flow_boot = tdflow_loss(
                    agent.flow_net, flow_target,
                    b_obs[mb_inds], b_next_obs[mb_inds], b_flow_mask[mb_inds],
                    args.flow_gamma, args.flow_steps,
                )
                r_pred = agent.reward_model(b_next_obs[mb_inds]).squeeze(-1)
                m = b_flow_mask[mb_inds]
                reward_loss = (((r_pred - b_rewards[mb_inds]) ** 2) * m).sum() / m.sum().clamp_min(1.0)

                # v30 backward. flow_net / reward_model / (trunk+actor) are now DISJOINT param
                # sets (the flow has its own obs encoder and the value no longer touches the
                # trunk), so a single backward populates each group's grads from only its own
                # loss; we clip each group to its own max-norm before one optimizer step.
                optimizer.zero_grad(set_to_none=True)
                (pg_loss - args.ent_coef * entropy_loss
                 + args.flow_coef * flow_loss + args.reward_coef * reward_loss).backward()
                actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                flow_gn = nn.utils.clip_grad_norm_(flow_params, args.flow_grad_clip)
                reward_gn = nn.utils.clip_grad_norm_(reward_params, args.reward_grad_clip)
                optimizer.step()
                ema_update(flow_target, agent.flow_net, args.flow_target_decay)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/flow_loss", flow_loss.item(), global_step)
        writer.add_scalar("losses/flow_direct", flow_direct.item(), global_step)
        writer.add_scalar("losses/flow_boot", flow_boot.item(), global_step)
        writer.add_scalar("losses/value_std", values.std().item(), global_step)
        writer.add_scalar("losses/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("losses/flow_grad_norm", float(flow_gn), global_step)
        writer.add_scalar("losses/value_mean", values.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/reward_grad_norm", float(reward_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
