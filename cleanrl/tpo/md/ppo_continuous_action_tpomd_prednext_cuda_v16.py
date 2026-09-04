# TPO-MD PredNext CUDA v16: graph-optimized v13 without an algorithm change.
#
# The exact v13 control and optimizer semantics are retained: K=8 raw-MuJoCo probes,
# one-sided dynamic KL, pure mirror descent, HL-Gauss critic, and optimizer-level
# PredNext admission. Static rollout, PPO-update, value, label-projection, and target-
# encoder forwards use torch.compile(reduce-overhead) with explicit CUDA-graph step
# boundaries. Long-lived graph outputs are cloned immediately; the full label table
# and epoch indices stay on CUDA. Packed telemetry and asynchronous fail-loud clipping
# remove avoidable minibatch synchronizations. High float32 matmul precision enables
# TF32 while preserving the task and forked probe RNG streams and every TPO decision.
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

from cleanrl.shared.hl_gauss import HLGaussSupport, symlog, symexp

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def clip_grad_norm_async_fail_loud_(parameters, max_norm, norm_type=2.0):
    """Clip exactly as PyTorch does and enqueue the finite check on-device."""
    total_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
    )
    torch._assert_async(
        torch.isfinite(total_norm),
        "The total gradient norm is non-finite; refusing the optimizer step",
    )
    return total_norm


@torch.no_grad()
def synchronize_scalar_telemetry(statistics):
    """Materialize CUDA scalar telemetry with one packed device-to-host copy."""
    if not statistics:
        return {}
    names = tuple(statistics)
    scalars = tuple(statistics.values())
    if not all(torch.is_tensor(value) for value in scalars):
        raise TypeError("telemetry values must be tensors")
    if not all(value.numel() == 1 for value in scalars):
        raise ValueError("telemetry values must be scalar tensors")
    if len({value.device for value in scalars}) != 1:
        raise ValueError("telemetry values must share one device")
    host_values = torch.stack(
        [value.detach().reshape(()) for value in scalars]
    ).cpu().tolist()
    return dict(zip(names, host_values))


@torch.no_grad()
def retain_graph_output(output, *, compiled):
    """Detach and clone output whose CUDA-graph storage will be replayed."""
    if not torch.is_tensor(output):
        raise TypeError("retained graph output must be a tensor")
    return output.detach().clone() if compiled else output.detach()


def value_support_bounds(args):
    """Support bounds in the coordinate used for categorical bins."""
    return args.v_min, args.v_max


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
    norm_adv: bool = True
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
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    # NOTE: target_kl epoch-stop would starve the (always-on) critic in this pure
    # variant; default None — the actor is leashed by tpo_kl_breaker instead.
    target_kl: Optional[float] = None

    # TPO mirror descent: probe-scored TPO target with MPO-style adaptive
    # temperature REPLACES the PPO surrogate. Probes run at EVERY rollout state.
    tpo_coef: float = 1.0        # weight of the TPO CE (the entire actor loss besides entropy); must be > 0
    tpo_eta: float = 6.0         # FIXED temperature, used only when tpo_adaptive_eta=False
    tpo_k: int = 8               # candidates per state (ALL probed, incl. the executed action as candidate 0)
    tpo_sigma_scale_coef: float = 1.0  # global score sigma = coef * EMA(one-step TD-residual RMS)
    tpo_eps: float = 0.03        # trust-region CAP / max KL per update (dyn-trust) OR fixed KL target (v1 mode)
    tpo_adaptive_eta: bool = True      # solve eta s.t. mean KL(p_old||q)=tpo_eps; False => fixed tpo_eta
    tpo_dyn_trust: bool = True   # one-sided KL cap on a fixed base temperature (v5 default). False => exact tpomd_v1 fixed-target dual
    tpo_eta_base: float = 1.0    # base temperature for the dynamic-cap path; natural KL at this eta is the signal-determined step (unused when tpo_dyn_trust=False)
    tpo_kl_breaker: float = 0.09 # actor circuit breaker: stop actor epochs when epoch-mean approx_kl exceeds (3x eps)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # v149-aligned distributional critic support. Bounds are already symlog
    # coordinates for raw-return support [-20000, 20000].
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 0.5  # requested sharper HL-Gauss projection sigma

    # Raw-return ablation: keep observations as in the source, but do not divide
    # rewards by NormalizeReward's running discounted-return std and do not clip
    # raw rewards before GAE.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
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

    # Depth-4 NextLat with optimizer-level task-priority admission. The predictive
    # coefficient is intentionally not a task-loss mixing coefficient: clipping and
    # update-space trust determine the representation perturbation's actual scale.
    nextlat: bool = True
    nextlat_depth: int = 4
    nextlat_coef: float = 1.0
    nextlat_kl_coef: float = 1.0
    nextlat_trunk_grad_clip: float = 0.25
    nextlat_predictor_grad_clip: float = 0.25
    prednext_trust_ratio: float = 0.05
    nextlat_target_batch_size: int = 8192

    # reduce-overhead enables CUDA graphs for every static neural forward. Raw MuJoCo
    # probes and all TPO control flow remain eager and behavior-equivalent to v13.
    compile: bool = False
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
        clipped_obs_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape, -10.0, dtype=env.observation_space.dtype),
            high=np.full(env.observation_space.shape, 10.0, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=clipped_obs_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    # Walk the .env wrapper chain looking for wrapper_type.
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class IndexedTransferBranch(nn.Module):
    def __init__(self, H, history_dim):
        super().__init__()
        if history_dim % H != 0:
            raise ValueError(f"history_dim={history_dim} must be divisible by H={H}")
        self.H = H
        self.history_slots = history_dim // H
        self.current_linear = layer_init(nn.Linear(H, H))
        self.act = ReLUSquared()
        self.out_linear = layer_init(nn.Linear(H, H))
        self.history_weight = nn.Parameter(torch.empty(self.history_slots, H))
        nn.init.normal_(self.history_weight, mean=0.0, std=np.sqrt(2.0 / (H + self.history_slots)))

    def forward(self, x, history):
        preact = self.current_linear(x)
        history = history.reshape(history.shape[0], self.history_slots, self.H)
        same_index_transfer = (history * self.history_weight.to(dtype=history.dtype).unsqueeze(0)).sum(dim=1)
        return self.out_linear(self.act(preact + same_index_transfer))


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
        self.dense = IndexedTransferBranch(H, in_dim)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([IndexedTransferBranch(H, in_dim) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in), cat_feats)        # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in, cat_feats) for e in self.experts], dim=1)  # (B, E, H)
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
        # v149 critic readout style, without MTP: biasless HL-Gauss value head.
        self.num_bins = args.num_bins
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
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
        # Isolate auxiliary initialization from the task RNG stream. This preserves
        # every base parameter and the post-Agent RNG state of TPO-MD v5 exactly.
        with torch.random.fork_rng(devices=[]):
            self.nextlat_predictor = nn.Sequential(
                layer_init(nn.Linear(H + act_dim, H)),
                ReLUSquared(),
                layer_init(nn.Linear(H, H), std=0.1),
            )

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

    def _actor_dist_frozen_head(self, actor_feat):
        """Decode a latent in policy geometry without updating policy-head weights."""
        if self.actor_dist == "gaussian":
            mean = F.linear(
                actor_feat,
                self.actor_head.weight.detach(),
                None if self.actor_head.bias is None else self.actor_head.bias.detach(),
            )
            raw_lv = F.linear(
                actor_feat,
                self.actor_logvar_head.weight.detach(),
                None
                if self.actor_logvar_head.bias is None
                else self.actor_logvar_head.bias.detach(),
            )
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            return (
                Normal(mean, (0.5 * lv).exp()),
                torch.tanh,
                lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z)),
            )
        alpha_raw = F.linear(
            actor_feat,
            self.actor_alpha_head.weight.detach(),
            None
            if self.actor_alpha_head.bias is None
            else self.actor_alpha_head.bias.detach(),
        )
        beta_raw = F.linear(
            actor_feat,
            self.actor_beta_head.weight.detach(),
            None
            if self.actor_beta_head.bias is None
            else self.actor_beta_head.bias.detach(),
        )
        return (
            Beta(1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw)),
            lambda z: self.action_low + (self.action_high - self.action_low) * z,
            lambda z: 0.0,
        )

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_actor_feat(self, x):
        return self._trunks(x)[0]

    def get_action_and_value(
        self, x, z=None, candidate_zs=None, return_dist=False, return_actor_feat=False
    ):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # TPO extensions (both default-off => base behavior/graph/RNG untouched):
        #   candidate_zs (B, K, A): also return per-candidate logprobs (B, K) from
        #     the SAME dist (one trunk forward, consumes no RNG — log_prob only,
        #     evaluated AFTER the gaussian entropy rsample so the RNG order of the
        #     base computation is preserved).
        #   return_dist: also return (dist, to_action, log_det_fn) so the rollout
        #     can sample probe candidates from the already-constructed dist.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat)
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
        out = (action, z, log_prob, entropy, value_logits)
        if candidate_zs is not None:
            # Evaluate as (K, B, A) so the dist's (B, A) batch shape broadcasts
            # over the K axis, then transpose back to (B, K).
            cz = candidate_zs.transpose(0, 1)
            candidate_log_probs = (dist.log_prob(cz) - log_det_fn(cz)).sum(-1).transpose(0, 1)
            out = out + (candidate_log_probs,)
        if return_dist:
            out = out + (dist, to_action, log_det_fn)
        if return_actor_feat:
            out = out + (actor_feat,)
        return out

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

    def nextlat_trunk_blocks(self):
        """Logical actor-trunk blocks used for local predictive trust regions."""
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        blocks = [[*trunk.entry.parameters()]]
        blocks.extend([list(block.parameters()) for block in trunk.blocks])
        blocks.append([*trunk.out_proj.parameters()])
        return blocks

    def nextlat_trunk_parameters(self):
        return [parameter for block in self.nextlat_trunk_blocks() for parameter in block]

    def nextlat_predictor_parameters(self):
        return list(self.nextlat_predictor.parameters())

    def task_parameters(self):
        predictor_ids = {id(parameter) for parameter in self.nextlat_predictor.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in predictor_ids]


def policy_model_forward(agent, observations):
    """Static neural policy/value path with no sampling or RNG side effects."""
    actor_feat, critic_feat = agent._trunks(observations)
    value_logits = agent.critic_head(critic_feat)
    if agent.actor_dist == "gaussian":
        first = agent.actor_head(actor_feat)
        raw_lv = agent.actor_logvar_head(actor_feat)
        second = rescale(
            (raw_lv / (agent.logvar_max - agent.logvar_min)).tanh(),
            (-1.0, 1.0),
            (agent.logvar_min, agent.logvar_max),
        )
    else:
        first = 1.0 + F.softplus(agent.actor_alpha_head(actor_feat))
        second = 1.0 + F.softplus(agent.actor_beta_head(actor_feat))
    return actor_feat, value_logits, first, second


def action_value_from_policy_outputs(
    agent,
    model_outputs,
    z=None,
    candidate_zs=None,
):
    """Apply v13's exact eager sampling/log-prob order to compiled model outputs."""
    actor_feat, value_logits, first, second = model_outputs
    if agent.actor_dist == "gaussian":
        dist = Normal(first, (0.5 * second).exp())
        to_action = torch.tanh
        log_det_fn = lambda sample: 2.0 * (
            log(2.0) - sample - F.softplus(-2.0 * sample)
        )
    else:
        dist = Beta(first, second)
        to_action = lambda sample: agent.action_low + (
            agent.action_high - agent.action_low
        ) * sample
        log_det_fn = lambda sample: 0.0
    if z is None:
        z = dist.sample()
        if agent.actor_dist == "beta":
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    action = to_action(z)
    log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
    if agent.actor_dist == "gaussian":
        zr = dist.rsample()
        entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
    else:
        entropy = dist.entropy().sum(1)
    out = (action, z, log_prob, entropy, value_logits)
    if candidate_zs is not None:
        candidate_transposed = candidate_zs.transpose(0, 1)
        candidate_log_probs = (
            dist.log_prob(candidate_transposed) - log_det_fn(candidate_transposed)
        ).sum(-1).transpose(0, 1)
        out = out + (candidate_log_probs,)
    return out + (actor_feat,), dist, to_action, log_det_fn


def value_forward(agent, observations):
    """Static value-logit wrapper used by transition and probe bootstraps."""
    return agent.get_value(observations)


def target_actor_feat_forward(agent, observations):
    """Static stopped-online target encoder wrapper."""
    return agent.get_actor_feat(observations)


def tpo_restricted_target(anchor_logp, score_signal, eta):
    """TPO-MD v5's anchored K-action mirror-descent target."""
    return torch.softmax(anchor_logp + score_signal / eta, dim=-1)


def tpo_reverse_kl(anchor_logp, score_signal, eta):
    """Batch mean KL(p_old || q_eta), the v5 one-sided-cap statistic."""
    p_old = anchor_logp.exp()
    log_q = F.log_softmax(anchor_logp + score_signal / eta, dim=-1)
    return (p_old * (anchor_logp - log_q)).sum(-1).mean()


def build_nextlat_mask(transition_boundaries, depth):
    """Validity of (state, outgoing actions, future-state) latent sequences."""
    num_steps, num_envs = transition_boundaries.shape
    mask = transition_boundaries.new_zeros((num_steps, num_envs, depth))
    for horizon in range(1, depth + 1):
        valid_len = num_steps - horizon
        if valid_len <= 0:
            break
        valid = torch.ones(
            (valid_len, num_envs), dtype=torch.bool, device=transition_boundaries.device
        )
        for offset in range(horizon):
            valid &= transition_boundaries[offset : offset + valid_len] == 0
        mask[:valid_len, :, horizon - 1] = valid.to(mask.dtype)
    return mask


def make_nextlat_indices(source_indices, num_envs, batch_size, depth):
    """T-major outgoing-action and future-state indices for recursive prediction."""
    action_offsets = np.arange(depth, dtype=np.int64)[:, None] * num_envs
    target_offsets = np.arange(1, depth + 1, dtype=np.int64)[:, None] * num_envs
    action_indices = np.clip(source_indices[None, :] + action_offsets, 0, batch_size - 1)
    target_indices = np.clip(source_indices[None, :] + target_offsets, 0, batch_size - 1)
    return action_indices, target_indices


@torch.no_grad()
def make_logical_block_layout(parameter_blocks):
    """Flatten trunk tensors while retaining contiguous logical-block segments."""
    if not parameter_blocks or any(not block for block in parameter_blocks):
        raise ValueError("each logical block must contain at least one parameter")
    parameters = [parameter for block in parameter_blocks for parameter in block]
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("all logical-block parameters must share a device")
    parameter_lengths = tuple(parameter.numel() for parameter in parameters)
    block_lengths_tuple = tuple(
        sum(parameter.numel() for parameter in block) for block in parameter_blocks
    )
    block_lengths = torch.tensor(block_lengths_tuple, device=device, dtype=torch.int64)
    block_ids = torch.repeat_interleave(
        torch.arange(len(parameter_blocks), device=device),
        block_lengths,
        output_size=sum(block_lengths_tuple),
    )
    return parameters, (parameter_lengths, block_lengths_tuple, block_lengths, block_ids)


def _segment_sum(values, lengths):
    return torch.segment_reduce(values, "sum", lengths=lengths)


def _all_finite(tensors):
    """Host decision used only at the eager optimizer-transaction boundary."""
    if not tensors:
        return True
    # One multi-tensor reduction avoids launching an isfinite kernel per trunk tensor.
    max_norms = torch.stack(torch._foreach_norm(tensors, float("inf")))
    return bool(torch.isfinite(max_norms).all().item())


@torch.no_grad()
def _zero_admission_result(task_updates, layout, *, nonfinite_veto):
    """Construct a finite zero admission directly, never by scaling unsafe values."""
    _, block_lengths_tuple, _, _ = layout
    zero = task_updates[0].new_zeros(())
    if _all_finite(task_updates):
        task_norm_candidate = torch.stack(torch._foreach_norm(task_updates)).norm()
        task_norm = task_norm_candidate if torch.isfinite(task_norm_candidate) else zero
    else:
        task_norm = zero
    block_zeros = task_updates[0].new_zeros(len(block_lengths_tuple))
    return [torch.zeros_like(update) for update in task_updates], {
        "task_norm": task_norm,
        "predictive_norm": zero,
        "admitted_norm": zero,
        "raw_cosine": zero,
        "accepted_fraction": zero,
        "global_cap_scale": zero,
        "actor_first_order": zero,
        "critic_first_order": zero,
        "block_actor_conflict": block_zeros.bool(),
        "block_critic_conflict": block_zeros.bool(),
        "block_admitted_ratio": block_zeros,
        "block_nonzero": block_zeros.bool(),
        "block_actor_first_order": block_zeros,
        "block_critic_first_order": block_zeros,
        "block_actor_tolerance": block_zeros,
        "block_critic_tolerance": block_zeros,
        "block_lengths": block_lengths_tuple,
        "nonfinite_veto": zero.new_tensor(float(nonfinite_veto)),
    }


def _admission_result_is_finite(admitted, stats):
    relevant_stats = [
        stats[key]
        for key in (
            "task_norm",
            "predictive_norm",
            "admitted_norm",
            "raw_cosine",
            "accepted_fraction",
            "global_cap_scale",
            "actor_first_order",
            "critic_first_order",
            "block_admitted_ratio",
            "block_actor_first_order",
            "block_critic_first_order",
            "block_actor_tolerance",
            "block_critic_tolerance",
            "nonfinite_veto",
        )
    ]
    return _all_finite([*admitted, *relevant_stats])


@torch.no_grad()
def admit_predictive_updates(
    task_updates,
    predictive_updates,
    max_ratio,
    *,
    actor_gradients,
    critic_gradients,
    layout,
):
    """Project and trust-cap a predictive Adam proposal per logical trunk block."""
    count = len(task_updates)
    if count == 0 or not (
        len(predictive_updates) == len(actor_gradients) == len(critic_gradients) == count
    ):
        raise ValueError("task, predictive, actor, and critic tensors must align")
    if max_ratio < 0.0:
        raise ValueError("max_ratio must be non-negative")
    parameter_lengths, block_lengths_tuple, block_lengths, block_ids = layout
    if tuple(update.numel() for update in task_updates) != parameter_lengths:
        raise ValueError("updates do not match the supplied logical-block layout")
    references = task_updates
    for tensors in (predictive_updates, actor_gradients, critic_gradients):
        for reference, tensor in zip(references, tensors):
            if tensor.shape != reference.shape or tensor.device != reference.device:
                raise ValueError("aligned update and gradient tensors must share shape and device")

    # Veto before concatenation/projection: multiplying an unsafe proposal by an
    # inactive-block zero is still NaN, so masking is not a finite-safety mechanism.
    if not _all_finite(
        [*task_updates, *predictive_updates, *actor_gradients, *critic_gradients]
    ):
        return _zero_admission_result(task_updates, layout, nonfinite_veto=True)

    flat_task = torch.cat([tensor.reshape(-1) for tensor in task_updates])
    flat_predictive = torch.cat([tensor.reshape(-1) for tensor in predictive_updates])
    flat_actor = torch.cat([tensor.reshape(-1) for tensor in actor_gradients])
    flat_critic = torch.cat([tensor.reshape(-1) for tensor in critic_gradients])

    task_sq_block = _segment_sum(flat_task.square(), block_lengths)
    pred_sq_block = _segment_sum(flat_predictive.square(), block_lengths)
    actor_sq = _segment_sum(flat_actor.square(), block_lengths)
    critic_sq = _segment_sum(flat_critic.square(), block_lengths)
    actor_critic = _segment_sum(flat_actor * flat_critic, block_lengths)
    actor_pred = _segment_sum(flat_actor * flat_predictive, block_lengths)
    critic_pred = _segment_sum(flat_critic * flat_predictive, block_lengths)
    actor_pred_abs = _segment_sum((flat_actor * flat_predictive).abs(), block_lengths)
    critic_pred_abs = _segment_sum((flat_critic * flat_predictive).abs(), block_lengths)
    actor_critic_abs = _segment_sum((flat_actor * flat_critic).abs(), block_lengths)

    # Exact active-set projection onto {delta: g_actor.delta <= 0,
    # g_critic.delta <= 0}. The zero proposal is the conservative fallback.
    tiny = torch.finfo(flat_task.dtype).tiny
    eps = torch.finfo(flat_task.dtype).eps
    # A pairwise dot reduction accumulates O(log2(n)) roundoff levels. The factor
    # covers the reductions plus the multiply/subtracts used to evaluate candidates;
    # multiplying by sum(abs(products)) makes the envelope scale- and cancellation-aware.
    reduction_levels = torch.ceil(torch.log2(block_lengths.to(flat_task.dtype))).clamp_min(1.0)
    roundoff_factor = 8.0 * eps * (reduction_levels + 2.0)

    def feasible(dot, absolute_term_bound):
        return dot <= roundoff_factor * absolute_term_bound + tiny

    best_distance = pred_sq_block.clone()
    best_actor_multiplier = torch.zeros_like(best_distance)
    best_critic_multiplier = torch.zeros_like(best_distance)
    best_kind = torch.zeros_like(best_distance, dtype=torch.int64)  # 0 => zero

    def select(distance, feasible, actor_multiplier, critic_multiplier, kind):
        nonlocal best_distance, best_actor_multiplier, best_critic_multiplier, best_kind
        choose = feasible & (distance < best_distance)
        best_distance = torch.where(choose, distance, best_distance)
        best_actor_multiplier = torch.where(choose, actor_multiplier, best_actor_multiplier)
        best_critic_multiplier = torch.where(choose, critic_multiplier, best_critic_multiplier)
        best_kind = torch.where(choose, torch.full_like(best_kind, kind), best_kind)

    zeros = torch.zeros_like(best_distance)
    select(
        zeros,
        feasible(actor_pred, actor_pred_abs)
        & feasible(critic_pred, critic_pred_abs),
        zeros,
        zeros,
        1,
    )

    actor_multiplier = actor_pred.clamp_min(0.0) / actor_sq.clamp_min(tiny)
    actor_candidate_actor = actor_pred - actor_multiplier * actor_sq
    actor_candidate_critic = critic_pred - actor_multiplier * actor_critic
    actor_candidate_actor_bound = actor_pred_abs + actor_multiplier.abs() * actor_sq
    actor_candidate_critic_bound = (
        critic_pred_abs + actor_multiplier.abs() * actor_critic_abs
    )
    select(
        actor_multiplier.square() * actor_sq,
        feasible(actor_candidate_actor, actor_candidate_actor_bound)
        & feasible(actor_candidate_critic, actor_candidate_critic_bound),
        actor_multiplier,
        zeros,
        2,
    )

    critic_multiplier = critic_pred.clamp_min(0.0) / critic_sq.clamp_min(tiny)
    critic_candidate_actor = actor_pred - critic_multiplier * actor_critic
    critic_candidate_critic = critic_pred - critic_multiplier * critic_sq
    critic_candidate_actor_bound = (
        actor_pred_abs + critic_multiplier.abs() * actor_critic_abs
    )
    critic_candidate_critic_bound = critic_pred_abs + critic_multiplier.abs() * critic_sq
    select(
        critic_multiplier.square() * critic_sq,
        feasible(critic_candidate_actor, critic_candidate_actor_bound)
        & feasible(critic_candidate_critic, critic_candidate_critic_bound),
        zeros,
        critic_multiplier,
        3,
    )

    # Both-active candidate, formed with blockwise Gram-Schmidt rather than the
    # normal-equation inverse. The latter subtracts two O(1/det) corrections when
    # gradients nearly oppose, destroying the finite projection through cancellation.
    actor_rank_valid = actor_sq > tiny
    critic_rank_valid = critic_sq > tiny
    actor_norm = actor_sq.sqrt()
    critic_norm = critic_sq.sqrt()
    safe_actor_norm = torch.where(
        actor_rank_valid, actor_norm, torch.ones_like(actor_norm)
    )
    safe_critic_norm = torch.where(
        critic_rank_valid, critic_norm, torch.ones_like(critic_norm)
    )
    flat_actor_unit = flat_actor / safe_actor_norm[block_ids]
    flat_critic_unit = flat_critic / safe_critic_norm[block_ids]
    gradient_correlation = actor_critic / (safe_actor_norm * safe_critic_norm)
    # Subtract the closest signed copy of u before orthogonalization. For nearly
    # opposite normals this forms v+u directly, retaining the small angular signal
    # instead of recovering it as v-rho*u after a cancellation-prone dot reduction.
    alignment_sign = torch.where(
        gradient_correlation >= 0.0,
        torch.ones_like(gradient_correlation),
        -torch.ones_like(gradient_correlation),
    )
    flat_angular_difference = (
        flat_critic_unit - alignment_sign[block_ids] * flat_actor_unit
    )
    difference_on_actor = _segment_sum(
        flat_actor_unit * flat_angular_difference, block_lengths
    )
    actor_unit_sq = _segment_sum(flat_actor_unit.square(), block_lengths)
    safe_actor_unit_sq = torch.where(
        actor_rank_valid, actor_unit_sq, torch.ones_like(actor_unit_sq)
    )
    difference_on_actor_multiplier = difference_on_actor / safe_actor_unit_sq
    flat_critic_orthogonal = (
        flat_angular_difference
        - difference_on_actor_multiplier[block_ids] * flat_actor_unit
    )
    critic_orthogonal_sq = _segment_sum(
        flat_critic_orthogonal.square(), block_lengths
    )
    critic_orthogonal_pred = _segment_sum(
        flat_critic_orthogonal * flat_predictive, block_lengths
    )
    rank_valid = actor_rank_valid & critic_rank_valid & (
        critic_orthogonal_sq > roundoff_factor.square()
    )
    safe_critic_orthogonal_sq = torch.where(
        rank_valid, critic_orthogonal_sq, torch.ones_like(critic_orthogonal_sq)
    )
    orthogonal_multiplier = critic_orthogonal_pred / safe_critic_orthogonal_sq
    orthogonal_multiplier = torch.where(
        rank_valid, orthogonal_multiplier, torch.zeros_like(orthogonal_multiplier)
    )
    actor_span_multiplier = _segment_sum(
        flat_actor_unit * flat_predictive, block_lengths
    ) / safe_actor_unit_sq
    critic_on_actor_multiplier = (
        alignment_sign + difference_on_actor_multiplier
    )
    # KKT multipliers in the ORIGINAL gradient scales.
    joint_critic_multiplier = orthogonal_multiplier / safe_critic_norm
    joint_actor_multiplier = (
        actor_span_multiplier
        - orthogonal_multiplier * critic_on_actor_multiplier
    ) / safe_actor_norm
    joint_actor_multiplier_tolerance = roundoff_factor * (
        actor_span_multiplier.abs()
        + (orthogonal_multiplier * critic_on_actor_multiplier).abs()
    ) / safe_actor_norm + tiny
    joint_critic_multiplier_tolerance = (
        roundoff_factor * orthogonal_multiplier.abs() / safe_critic_norm + tiny
    )
    flat_joint_projected = (
        flat_predictive
        - actor_span_multiplier[block_ids] * flat_actor_unit
        - orthogonal_multiplier[block_ids] * flat_critic_orthogonal
    )
    joint_actor = _segment_sum(flat_actor * flat_joint_projected, block_lengths)
    joint_critic = _segment_sum(flat_critic * flat_joint_projected, block_lengths)
    # Forward-error bounds must include the large intermediate corrections, not
    # merely abs(g * final_candidate): near opposition, O(1/angle) terms cancel.
    joint_actor_bound = (
        actor_pred_abs
        + joint_actor_multiplier.abs() * actor_sq
        + joint_critic_multiplier.abs() * actor_critic_abs
    )
    joint_critic_bound = (
        critic_pred_abs
        + joint_actor_multiplier.abs() * actor_critic_abs
        + joint_critic_multiplier.abs() * critic_sq
    )
    joint_distance = (
        actor_span_multiplier.square() * actor_unit_sq
        + orthogonal_multiplier.square() * critic_orthogonal_sq
    ).clamp_min(0.0)
    select(
        joint_distance,
        rank_valid
        & (joint_actor_multiplier >= -joint_actor_multiplier_tolerance)
        & (joint_critic_multiplier >= -joint_critic_multiplier_tolerance)
        & feasible(joint_actor, joint_actor_bound)
        & feasible(joint_critic, joint_critic_bound),
        zeros,
        zeros,
        4,
    )

    active = best_kind != 0
    flat_single_projected = (
        flat_predictive
        - best_actor_multiplier[block_ids] * flat_actor
        - best_critic_multiplier[block_ids] * flat_critic
    )
    flat_projected = torch.where(
        (best_kind == 4)[block_ids], flat_joint_projected, flat_single_projected
    ) * active[block_ids]
    post_actor = _segment_sum(flat_actor * flat_projected, block_lengths)
    post_critic = _segment_sum(flat_critic * flat_projected, block_lengths)
    # Correct positive residuals once per boundary. This removes the common one-ulp
    # sign flip without changing the active-set solution; near-opposed constraints may
    # reintroduce a tiny residual on the first boundary, handled by the error envelope.
    actor_residual_multiplier = post_actor.clamp_min(0.0) / actor_sq.clamp_min(tiny)
    flat_projected = flat_projected - actor_residual_multiplier[block_ids] * flat_actor
    post_critic_after_actor = _segment_sum(
        flat_critic * flat_projected, block_lengths
    )
    critic_residual_multiplier = (
        post_critic_after_actor.clamp_min(0.0) / critic_sq.clamp_min(tiny)
    )
    flat_projected = flat_projected - critic_residual_multiplier[block_ids] * flat_critic

    post_actor = _segment_sum(flat_actor * flat_projected, block_lengths)
    post_critic = _segment_sum(flat_critic * flat_projected, block_lengths)
    post_actor_abs = _segment_sum((flat_actor * flat_projected).abs(), block_lengths)
    post_critic_abs = _segment_sum((flat_critic * flat_projected).abs(), block_lengths)
    actor_tolerance = roundoff_factor * post_actor_abs + tiny
    critic_tolerance = roundoff_factor * post_critic_abs + tiny
    # Meaningful positive first-order changes still veto the entire logical block.
    safe = (post_actor <= actor_tolerance) & (post_critic <= critic_tolerance)
    flat_projected = flat_projected * safe[block_ids]

    projected_sq_block = _segment_sum(flat_projected.square(), block_lengths)
    task_norm_block = task_sq_block.sqrt()
    projected_norm_block = projected_sq_block.sqrt()
    local_scale = torch.clamp(
        max_ratio * task_norm_block / projected_norm_block.clamp_min(1e-20), max=1.0
    )
    local_scale *= (task_sq_block > 0.0).to(local_scale.dtype)
    flat_local = flat_projected * local_scale[block_ids]

    task_norm = task_sq_block.sum().sqrt()
    local_norm = flat_local.square().sum().sqrt()
    global_scale = torch.clamp(
        max_ratio * task_norm / local_norm.clamp_min(1e-20), max=1.0
    )
    global_scale *= (task_norm > 0.0).to(global_scale.dtype)
    flat_admitted = flat_local * global_scale
    admitted = [
        flat.view_as(reference)
        for flat, reference in zip(flat_admitted.split(parameter_lengths), references)
    ]

    admitted_sq_block = _segment_sum(flat_admitted.square(), block_lengths)
    admitted_norm = admitted_sq_block.sum().sqrt()
    predictive_norm = flat_predictive.square().sum().sqrt()
    raw_cosine = (flat_task * flat_predictive).sum() / (
        task_norm * predictive_norm
    ).clamp_min(1e-20)
    admitted_ratio_block = admitted_sq_block.sqrt() / task_norm_block.clamp_min(1e-20)
    stats = {
        "task_norm": task_norm,
        "predictive_norm": predictive_norm,
        "admitted_norm": admitted_norm,
        "raw_cosine": raw_cosine,
        "accepted_fraction": admitted_norm / predictive_norm.clamp_min(1e-20),
        "global_cap_scale": global_scale,
        "actor_first_order": (flat_actor * flat_admitted).sum(),
        "critic_first_order": (flat_critic * flat_admitted).sum(),
        "block_actor_conflict": actor_pred > 0.0,
        "block_critic_conflict": critic_pred > 0.0,
        "block_admitted_ratio": admitted_ratio_block,
        "block_nonzero": admitted_sq_block > 0.0,
        "block_actor_first_order": _segment_sum(flat_actor * flat_admitted, block_lengths),
        "block_critic_first_order": _segment_sum(flat_critic * flat_admitted, block_lengths),
        "block_actor_tolerance": actor_tolerance * local_scale * global_scale,
        "block_critic_tolerance": critic_tolerance * local_scale * global_scale,
        "block_lengths": block_lengths_tuple,
        "nonfinite_veto": flat_task.new_zeros(()),
    }
    # Finite inputs can still overflow a reduction. Keep the zero fallback outside
    # all unsafe arithmetic and let the transaction retain its exact task snapshot.
    if not _admission_result_is_finite(admitted, stats):
        return _zero_admission_result(task_updates, layout, nonfinite_veto=True)
    return admitted, stats


@torch.no_grad()
def apply_predictive_trunk_transaction(
    parameters,
    optimizer,
    gradients,
    task_updates,
    max_ratio,
    *,
    actor_gradients,
    critic_gradients,
    layout,
):
    """Advance predictive Adam state, then replace its proposal by the admitted delta."""
    if len(parameters) != len(gradients):
        raise ValueError("one predictive gradient is required per trunk parameter")
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    optimizer.step()
    raw_updates = [
        parameter.detach() - previous for parameter, previous in zip(parameters, before)
    ]
    for parameter, previous in zip(parameters, before):
        parameter.copy_(previous)
    if _all_finite(raw_updates):
        admitted_updates, stats = admit_predictive_updates(
            task_updates,
            raw_updates,
            max_ratio,
            actor_gradients=actor_gradients,
            critic_gradients=critic_gradients,
            layout=layout,
        )
    else:
        admitted_updates, stats = _zero_admission_result(
            task_updates, layout, nonfinite_veto=True
        )
    if not _admission_result_is_finite(admitted_updates, stats):
        admitted_updates, stats = _zero_admission_result(
            task_updates, layout, nonfinite_veto=True
        )
    vetoed = bool(stats["nonfinite_veto"].item())
    # On veto, do not even add a constructed zero: retaining the snapshot exactly
    # avoids any arithmetic involving the rejected proposal. Adam state already advanced.
    if not vetoed:
        for parameter, update in zip(parameters, admitted_updates):
            parameter.add_(update)
    optimizer.zero_grad(set_to_none=True)
    return raw_updates, admitted_updates, stats


@torch.no_grad()
def apply_private_optimizer_step(parameters, optimizer, gradients):
    """Apply the predictor's private Adam transaction and return its step norm."""
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("one gradient is required per private parameter")
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    optimizer.step()
    update_sq = parameters[0].new_zeros(())
    for parameter, previous in zip(parameters, before):
        update_sq += (parameter.detach() - previous).square().sum()
    optimizer.zero_grad(set_to_none=True)
    return update_sq.sqrt()


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
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
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
    assert args.tpo_coef > 0.0, "TPO-MD is the entire policy update; tpo_coef must be > 0"
    assert args.tpo_k >= 2, "TPO needs at least two candidates per group"
    assert args.tpo_eps > 0.0, "tpo_eps must be positive"
    assert args.tpo_eta_base > 0.0, "tpo_eta_base must be positive"
    assert args.tpo_kl_breaker > 0.0, "tpo_kl_breaker must be positive"
    assert args.nextlat_depth >= 1, "nextlat_depth must be positive"
    assert args.nextlat_target_batch_size >= 1, "nextlat_target_batch_size must be positive"
    assert 0.0 <= args.prednext_trust_ratio <= 1.0
    assert args.separate_grad_clip, "PredNext requires separate actor/critic gradients"
    # Probe rewards are RAW physics rewards; the critic must live in the same units.
    assert not args.normalize_reward, "TPO probe scores require raw rewards"
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
    # Ampere+ TF32 tensor-core matmuls are substantially faster. ``high`` retains
    # full-size float32 outputs/accumulators; only last-bit eager-v13 numerics may differ.
    torch.set_float32_matmul_precision("high")

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

    # TPO-MD probe machinery (always on: TPO-MD IS the policy update; every
    # (env, step) cell is probed — no state-frac mask).
    # Cache once: the raw physics env (walk the .env chain to the unwrapped
    # MujocoEnv) and the NormalizeObservation wrapper reference per env.
    probe_base_envs = [e.unwrapped for e in envs.envs]
    probe_obs_wrappers = [find_wrapper(e, gym.wrappers.NormalizeObservation) for e in envs.envs]
    assert all(w is not None for w in probe_obs_wrappers), "NormalizeObservation wrapper not found"
    probe_action_low = envs.single_action_space.low
    probe_action_high = envs.single_action_space.high
    # Persistent probe RNG stream: saved CPU+CUDA states restored inside
    # torch.random.fork_rng at every sampling site, so candidate sampling
    # never advances the MAIN RNG stream (the PPO trajectory of a tpo run
    # matches an unprobed run exactly).
    probe_cpu_rng_state = None
    probe_cuda_rng_state = None
    td_rms_ema = None  # EMA (decay 0.99) of the one-step TD-residual RMS

    agent = Agent(envs, args).to(device)
    # Independent moments are essential: task Adam is exactly the TPO-MD v5
    # optimizer, predictive trunk Adam proposes representation deltas, and the
    # predictor's private Adam never shares either geometry or trust budget.
    task_optimizer = optim.Adam(agent.task_parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    nextlat_trunk_blocks = agent.nextlat_trunk_blocks()
    nextlat_trunk_params, prednext_layout = make_logical_block_layout(nextlat_trunk_blocks)
    prednext_block_names = ["entry"] + [
        f"think_{index}" for index in range(args.k_blocks)
    ] + ["readout"]
    assert len(prednext_block_names) == len(nextlat_trunk_blocks)
    nextlat_predictor_params = agent.nextlat_predictor_parameters()
    assert [id(parameter) for parameter in nextlat_trunk_params] == [
        id(parameter)
        for parameter in (
            agent.trunk if agent.share_backbone else agent.actor_trunk
        ).parameters()
    ]
    predictive_trunk_optimizer = optim.Adam(
        nextlat_trunk_params, lr=args.learning_rate, eps=1e-5
    )
    predictor_optimizer = optim.Adam(
        nextlat_predictor_params, lr=args.learning_rate, eps=1e-5
    )

    def policy_rollout_fn(obs_):
        return policy_model_forward(agent, obs_)

    def policy_update_fn(obs_):
        return policy_model_forward(agent, obs_)

    def transition_value_fn(obs_):
        return value_forward(agent, obs_)

    def probe_value_fn(obs_):
        return value_forward(agent, obs_)

    def target_actor_feat_fn(obs_):
        return target_actor_feat_forward(agent, obs_)

    if args.compile:
        policy_rollout_fn = torch.compile(
            policy_rollout_fn, mode=args.compile_mode, dynamic=False
        )
        policy_update_fn = torch.compile(
            policy_update_fn, mode=args.compile_mode, dynamic=False
        )
        transition_value_fn = torch.compile(
            transition_value_fn, mode=args.compile_mode, dynamic=False
        )
        probe_value_fn = torch.compile(
            probe_value_fn, mode=args.compile_mode, dynamic=False
        )
        target_actor_feat_fn = torch.compile(
            target_actor_feat_fn, mode=args.compile_mode, dynamic=False
        )
        print(f"compiled static agent paths ({args.compile_mode})")

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

    support_min, support_max = value_support_bounds(args)
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    support = hl_support.support                       # (num_bins,) bin centers
    bin_width = hl_support.bin_width
    scalar_support = symexp(support) if args.value_symlog else support

    def value_logits_to_scalar(logits):
        return hl_support.to_expected_scalar(logits)

    scalar_bin_width = (
        (scalar_support[1:] - scalar_support[:-1]).abs().min()
        if args.value_symlog
        else bin_width
    )

    def project_value_targets_fn(targets_):
        return hl_support.project(targets_)

    if args.compile:
        project_value_targets_fn = torch.compile(
            project_value_targets_fn,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        print(f"compiled static target projection ({args.compile_mode})")

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_transition_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_transition_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # Preallocated probe storage: CPU numpy for the physics outputs, GPU
    # tensors for candidate z's/logprobs (written once per step, no per-env syncs).
    tpo_next_obs_np = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_observation_space.shape, dtype=np.float32
    )
    tpo_rewards_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_terms_np = np.zeros((args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32)
    tpo_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + envs.single_action_space.shape
    ).to(device)
    tpo_logprobs = torch.zeros((args.num_steps, args.num_envs, args.tpo_k)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            task_optimizer.param_groups[0]["lr"] = lrnow
            predictive_trunk_optimizer.param_groups[0]["lr"] = lrnow
            predictor_optimizer.param_groups[0]["lr"] = lrnow
        probe_seconds = 0.0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                rollout_outputs = policy_rollout_fn(next_obs)
                (
                    action,
                    z,
                    logprob,
                    ent,
                    value_logits,
                    roll_actor_feat,
                ), roll_dist, roll_to_action, roll_log_det_fn = action_value_from_policy_outputs(
                    agent,
                    rollout_outputs,
                )
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = value_logits_to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            # --- TPO probe: K candidates per env, one raw physics step each (every state) ---
            probe_start = time.time()
            with torch.no_grad():
                # Candidate sampling rides the PERSISTENT probe RNG stream inside
                # fork_rng: restore probe state, sample, save state back. The main
                # stream is untouched.
                with torch.random.fork_rng(devices=[device]):
                    if probe_cpu_rng_state is None:
                        torch.manual_seed(args.seed + 1_000_003)
                    else:
                        torch.set_rng_state(probe_cpu_rng_state)
                        torch.cuda.set_rng_state(probe_cuda_rng_state, device)
                    cand_z = roll_dist.sample(torch.Size([args.tpo_k]))   # (K, N, A)
                    probe_cpu_rng_state = torch.get_rng_state()
                    probe_cuda_rng_state = torch.cuda.get_rng_state(device)
                cand_z = cand_z.permute(1, 0, 2).contiguous()             # (N, K, A)
                if args.actor_dist == "beta":
                    cand_z = cand_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                cand_z[:, 0] = z                                          # executed action = candidate 0
                cz = cand_z.transpose(0, 1)                               # (K, N, A)
                cand_logprob = (roll_dist.log_prob(cz) - roll_log_det_fn(cz)).sum(-1).transpose(0, 1)
                tpo_zs[step] = cand_z
                tpo_logprobs[step] = cand_logprob                         # (N, K)
                # One transfer for the whole candidate block (no per-env GPU syncs).
                cand_actions_np = roll_to_action(cand_z).cpu().numpy()
            cand_actions_np = np.clip(cand_actions_np, probe_action_low, probe_action_high)
            for env_i in range(args.num_envs):
                base_env = probe_base_envs[env_i]
                obs_rms = probe_obs_wrappers[env_i].obs_rms
                saved_qpos = base_env.data.qpos.copy()
                saved_qvel = base_env.data.qvel.copy()
                saved_warm = base_env.data.qacc_warmstart.copy()
                saved_time = base_env.data.time
                for cand_i in range(args.tpo_k):
                    # Direct-assign restore (NO mj_forward, NEVER MujocoEnv.set_state):
                    # mj_step recomputes forward dynamics itself; restoring
                    # qacc_warmstart keeps the solver warmstart bit-identical so the
                    # REAL env.step below matches an unprobed run exactly.
                    base_env.data.qpos[:] = saved_qpos
                    base_env.data.qvel[:] = saved_qvel
                    base_env.data.qacc_warmstart[:] = saved_warm
                    base_env.data.time = saved_time
                    probe_obs, probe_rew, probe_term, _, _ = base_env.step(cand_actions_np[env_i, cand_i])
                    # FROZEN wrapper stats (stepping the raw env never updates
                    # obs_rms): float64 math, cast float32, then clip [-10, 10].
                    norm_obs = ((probe_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)).astype(np.float32)
                    tpo_next_obs_np[step, env_i, cand_i] = np.clip(norm_obs, -10.0, 10.0)
                    tpo_rewards_np[step, env_i, cand_i] = probe_rew       # RAW reward (base is raw-return)
                    tpo_terms_np[step, env_i, cand_i] = float(probe_term)
                base_env.data.qpos[:] = saved_qpos
                base_env.data.qvel[:] = saved_qvel
                base_env.data.qacc_warmstart[:] = saved_warm
                base_env.data.time = saved_time
            probe_seconds += time.time() - probe_start

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                transition_next_obs = np.array(next_obs_np, copy=True)
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
            else:
                transition_next_obs = next_obs_np
            transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                next_transition_logits = transition_value_fn(transition_next_obs_t)
                next_transition_values[step] = value_logits_to_scalar(next_transition_logits)
            next_transition_obses[step] = transition_next_obs_t
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
            transition_boundaries[step] = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_{t+1}) for each transition bootstrap entropy.
                # Use transition_next_obs, not rollout next_obs, so time-limit
                # truncations pair V(final_obs) with H(final_obs) rather than
                # accidentally reading entropy from the reset observation.
                _, _, next_transition_logprob, _, _ = agent.get_action_and_value(
                    next_transition_obses.reshape((-1,) + envs.single_observation_space.shape)
                )
                next_transition_logprob = next_transition_logprob.reshape(args.num_steps, args.num_envs)
                alpha_r = log_alpha.exp().detach()
                next_value_bonus = alpha_r * (-next_transition_logprob)
            else:
                next_value_bonus = None
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                    lambda_nonterminal = 1.0 - transition_boundaries[t]
                    delta = (
                        rewards[t]
                        + args.gamma
                        * (next_transition_values[t] + next_value_bonus[t])
                        * bootstrap_nonterminal
                        - values[t]
                    )
                    policy_adv[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                    )
            else:
                policy_adv = advantages
            # Critic target: Dreamer4-style scalar-return HL-Gauss. GAE computes
            # the scalar λ-return; the value encoder projects that scalar target
            # into a Gaussian-smoothed categorical distribution over fixed bins.
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            projected_targets = project_value_targets_fn(returns)
            # This full CUDA-resident table is indexed for all ten PPO epochs. A
            # compiled graph owns a reusable output buffer, so retain an independent
            # clone before any later compiled forward can replay its storage.
            target_probs = retain_graph_output(
                projected_targets,
                compiled=args.compile,
            )
            # Per-state return std sigma(s_t) in raw return units, matching the
            # GAE scale consumed by tanh_std.
            sigma = (value_probs * (scalar_support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * scalar_bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            returns_coord = symlog(returns) if args.value_symlog else returns
            cdf_frac = ((returns_coord.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean()  # calib probe (uniform≈0.1)

            # --- TPO-MD target construction (frozen pre-update critic; q fixed across epochs) ---
            # Running one-step TD-residual RMS over EXECUTED transitions -> GLOBAL score sigma.
            td_resid = (
                rewards
                + args.gamma * next_transition_values * (1.0 - transition_terminations) * transition_valids
                - values
            )
            td_rms = td_resid.pow(2).mean().sqrt().item()
            td_rms_ema = td_rms if td_rms_ema is None else 0.99 * td_rms_ema + 0.01 * td_rms
            tpo_sigma_global = max(args.tpo_sigma_scale_coef * td_rms_ema, 1e-6)

            b_tpo_zs = tpo_zs.reshape((-1, args.tpo_k) + envs.single_action_space.shape)
            obs_dim = int(np.array(envs.single_observation_space.shape).prod())
            flat_probe_obs = torch.as_tensor(
                tpo_next_obs_np.reshape(-1, obs_dim), device=device
            )
            # Four static 65,536-row critic forwards at the defaults. Clone each graph
            # output immediately: later replays reuse its otherwise-ephemeral storage.
            probe_value_chunks = []
            for chunk in flat_probe_obs.split(65536):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                chunk_logits = probe_value_fn(chunk)
                chunk_logits = retain_graph_output(
                    chunk_logits,
                    compiled=args.compile,
                )
                probe_value_chunks.append(value_logits_to_scalar(chunk_logits))
            v_next = torch.cat(probe_value_chunks).reshape(
                args.batch_size, args.tpo_k
            )
            r_probe = torch.as_tensor(tpo_rewards_np.reshape(-1, args.tpo_k), device=device)
            term_probe = torch.as_tensor(tpo_terms_np.reshape(-1, args.tpo_k), device=device)
            # Oracle score: raw probe reward + bootstrapped frozen value.
            scores = r_probe + args.gamma * (1.0 - term_probe) * v_next      # (B, K)
            # Center per group, scale by the ONE GLOBAL sigma: cross-state advantage
            # MAGNITUDE survives (per-group z-scoring would erase it); no floor gating —
            # weak groups just contribute u ~= 0 naturally.
            u_scores = (
                (scores - scores.mean(dim=-1, keepdim=True)) / tpo_sigma_global
            ).clamp(-5.0, 5.0)
            group_std = scores.std(dim=-1, unbiased=False)                   # (B,) diagnostics only
            anchor_logp = F.log_softmax(tpo_logprobs.reshape(-1, args.tpo_k), dim=-1)

            def tpo_mean_kl(eta):
                # batch-mean KL(p_old || q(eta)); monotone DECREASING in eta
                # (eta -> inf => q -> p_old => KL -> 0).
                return tpo_reverse_kl(anchor_logp, u_scores, eta).item()

            # kl_base = the NATURAL (uncapped) step the SNR signal produces at the
            # fixed base temperature. In SNR units this is large under real signal
            # and -> 0 when candidates are within the critic's noise floor. Used by
            # the dynamic-cap path; also logged for diagnostics.
            tpo_kl_base = tpo_mean_kl(args.tpo_eta_base)
            tpo_cap_engaged = 0.0
            if u_scores.abs().max().item() < 1e-8:
                # Degenerate scores: target collapses to the anchor regardless of eta.
                # In dyn-trust eta_base is the natural choice (q ~= p_old anyway); in
                # v1 mode the original code returned 1.0 here.
                tpo_eta_solved = args.tpo_eta_base if args.tpo_dyn_trust else 1.0
            elif args.tpo_dyn_trust:
                # One-sided KL cap on the fixed base temperature. KL(eta) is monotone
                # DECREASING in eta, so we only ever RAISE eta above eta_base to pull
                # an over-large natural step DOWN to the cap; we never lower it. Thus
                # eta_solved >= eta_base ALWAYS, the step is bounded above by eps_cap,
                # and is free to shrink to ~0 when kl_base falls below the cap. No
                # lower floor — intentional (this is the late-training fix).
                if tpo_kl_base <= args.tpo_eps:
                    tpo_eta_solved = args.tpo_eta_base       # weak signal: natural step already within cap
                else:
                    tpo_cap_engaged = 1.0                    # strong signal: cap binds
                    log_lo, log_hi = float(np.log(args.tpo_eta_base)), float(np.log(1e4))
                    if tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                        tpo_eta_solved = float(np.exp(log_hi))  # even max temperature can't reach cap -> clamp
                    else:
                        # KL(eta_base) > eps and KL(1e4) <= eps -> root bracketed.
                        for _ in range(40):
                            log_mid = 0.5 * (log_lo + log_hi)
                            if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                                log_lo = log_mid             # KL too big -> need larger eta
                            else:
                                log_hi = log_mid
                        tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            elif args.tpo_adaptive_eta:
                # MPO-style dual: bisect log-eta so mean KL(p_old||q) = tpo_eps.
                log_lo, log_hi = float(np.log(1e-2)), float(np.log(1e4))
                if tpo_mean_kl(float(np.exp(log_lo))) < args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_lo))   # weak scores: even max-strength < eps
                elif tpo_mean_kl(float(np.exp(log_hi))) > args.tpo_eps:
                    tpo_eta_solved = float(np.exp(log_hi))   # huge scores: clamp at max temperature
                else:
                    for _ in range(40):
                        log_mid = 0.5 * (log_lo + log_hi)
                        if tpo_mean_kl(float(np.exp(log_mid))) > args.tpo_eps:
                            log_lo = log_mid                 # KL too big -> need larger eta
                        else:
                            log_hi = log_mid
                    tpo_eta_solved = float(np.exp(0.5 * (log_lo + log_hi)))
            else:
                tpo_eta_solved = args.tpo_eta
            b_tpo_q = tpo_restricted_target(anchor_logp, u_scores, tpo_eta_solved).detach()
            tpo_kl_achieved = tpo_mean_kl(tpo_eta_solved)
            log_q = b_tpo_q.clamp_min(1e-12).log()
            tpo_group_kl = (b_tpo_q * (log_q - anchor_logp)).sum(-1).mean().item()
            tpo_q_entropy = (-(b_tpo_q * log_q).sum(-1)).mean().item()
            tpo_score_std_mean = group_std.mean().item()
            tpo_score_std_p90 = group_std.quantile(0.9).item()
            if args.nextlat:
                nextlat_mask = build_nextlat_mask(
                    transition_boundaries, args.nextlat_depth
                )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        if args.nextlat:
            b_nextlat_mask = nextlat_mask.reshape(-1, args.nextlat_depth)
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean()

        b_inds = np.arange(args.batch_size)
        if args.nextlat:
            nextlat_action_offsets = (
                torch.arange(
                    args.nextlat_depth,
                    device=device,
                    dtype=torch.int64,
                )[:, None]
                * args.num_envs
            )
            nextlat_target_offsets = (
                torch.arange(
                    1,
                    args.nextlat_depth + 1,
                    device=device,
                    dtype=torch.int64,
                )[:, None]
                * args.num_envs
            )
        # v13 converted every float32 minibatch mean to Python before NumPy's
        # float64 average. Accumulate the same sequence asynchronously on CUDA.
        clipfrac_sum = torch.zeros((), dtype=torch.float64, device=device)
        clipfrac_count = 0
        epochs_completed = 0
        actor_epochs_completed = 0
        actor_active = True  # flipped off by the tpo_kl_breaker; critic runs all epochs regardless
        actor_epoch_ce_means = []  # per-epoch mean CE while the actor is active (convergence probe)
        if args.nextlat:
            # One stopped ONLINE (not EMA) target table from the pre-update trunk.
            # Chunked compiled encoding makes all depths simple indexed reads and
            # prevents a moving teacher from chasing ten epochs of student updates.
            target_chunks = []
            with torch.no_grad():
                for target_obs in b_obs.split(args.nextlat_target_batch_size):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    # Every retained chunk needs independent storage because the next
                    # replay reuses the graph-backed output buffer.
                    target_chunks.append(
                        retain_graph_output(
                            target_actor_feat_fn(target_obs),
                            compiled=args.compile,
                        )
                    )
                frozen_actor_feats = torch.cat(target_chunks)
                latent_batch_std = frozen_actor_feats.std(dim=0).mean()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Keep NumPy's exact seeded permutation, paying one compact transfer per
            # epoch instead of implicit host-index conversion for every buffer read.
            epoch_inds = torch.as_tensor(b_inds, device=device)
            epoch_kl_sum = torch.zeros((), dtype=torch.float64, device=device)
            epoch_kl_count = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = epoch_inds[start:end]

                # Same trunk forward as the base; the candidate logprobs ride the
                # SAME dist (no second trunk pass, consumes no RNG).
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                update_outputs = policy_update_fn(b_obs[mb_inds])
                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    value_logits,
                    new_cand_logprobs,
                    mb_actor_feat,
                ), _, _, _ = action_value_from_policy_outputs(
                    agent,
                    update_outputs,
                    b_latent_zs[mb_inds],
                    b_tpo_zs[mb_inds],
                )

                with torch.no_grad():
                    # TELEMETRY ONLY: ratio / KL / clipfrac (and pg_loss below) never
                    # reach a backward — the actor update is the TPO CE alone.
                    logratio = newlogprob.detach() - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac_sum.add_(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    )
                    clipfrac_count += 1
                    epoch_kl_sum.add_(approx_kl)
                    epoch_kl_count += 1

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
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

                # TELEMETRY-ONLY clipped surrogate (kept for cross-run comparability;
                # ratio is already detached, so no PG gradient can exist anywhere).
                with torch.no_grad():
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # TPO CE on the K-restricted softmax over ALL states (every state is
                # probed). Targets q are frozen (solved once post-rollout, detached).
                mb_logp_new = F.log_softmax(new_cand_logprobs, dim=-1)
                tpo_ce = (-(b_tpo_q[mb_inds] * mb_logp_new).sum(-1)).mean()
                # PURE mirror descent: the CE is the entire actor objective.
                actor_loss = args.tpo_coef * tpo_ce

                # HL-Gauss value loss: cross-entropy to the fixed scalar-return
                # projection target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                if args.nextlat:
                    action_indices = (
                        mb_inds[None, :] + nextlat_action_offsets
                    ).clamp_max(args.batch_size - 1)
                    target_indices = (
                        mb_inds[None, :] + nextlat_target_offsets
                    ).clamp_max(args.batch_size - 1)
                    future_actions = b_actions[action_indices]
                    future_target_feats = frozen_actor_feats[target_indices]
                    h_hat = mb_actor_feat
                    pred_losses, kl_losses = [], []
                    for horizon in range(args.nextlat_depth):
                        h_hat = agent.nextlat_predictor(
                            torch.cat([h_hat, future_actions[horizon]], dim=-1)
                        )
                        target_feat = future_target_feats[horizon]
                        mask = b_nextlat_mask[mb_inds, horizon]
                        denominator = mask.sum().clamp_min(1.0)
                        prediction_error = F.smooth_l1_loss(
                            h_hat, target_feat, reduction="none"
                        ).mean(-1)
                        pred_losses.append((prediction_error * mask).sum() / denominator)
                        with torch.no_grad():
                            target_dist, _, _ = agent._actor_dist_frozen_head(target_feat)
                        predicted_dist, _, _ = agent._actor_dist_frozen_head(h_hat)
                        policy_kl = torch.distributions.kl_divergence(
                            target_dist, predicted_dist
                        ).sum(-1)
                        kl_losses.append((policy_kl * mask).sum() / denominator)
                    nextlat_pred_loss = torch.stack(pred_losses).mean()
                    nextlat_kl_loss = torch.stack(kl_losses).mean()
                    nextlat_loss = (
                        nextlat_pred_loss + args.nextlat_kl_coef * nextlat_kl_loss
                    )

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

                # Preserve TPO-MD v5's two independently clipped task gradients, then
                # admit the auxiliary only after observing the ACTUAL task-Adam step.
                agent.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward(
                    retain_graph=args.nextlat or actor_active
                )
                critic_gn = clip_grad_norm_async_fail_loud_(
                    critic_params, args.critic_grad_clip
                )
                value_grads = {
                    parameter: parameter.grad.detach().clone()
                    for parameter in critic_params
                    if parameter.grad is not None
                }

                if args.nextlat:
                    agent.zero_grad(set_to_none=True)
                    (args.nextlat_coef * nextlat_loss).backward(
                        retain_graph=actor_active
                    )
                    nextlat_trunk_gn = clip_grad_norm_async_fail_loud_(
                        nextlat_trunk_params, args.nextlat_trunk_grad_clip
                    )
                    nextlat_predictor_gn = clip_grad_norm_async_fail_loud_(
                        nextlat_predictor_params, args.nextlat_predictor_grad_clip
                    )
                    nextlat_trunk_grads = [
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_trunk_params
                    ]
                    nextlat_predictor_grads = [
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_predictor_params
                    ]

                agent.zero_grad(set_to_none=True)
                if actor_active:
                    # Pure TPO CE remains the entire actor backward.
                    (actor_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = clip_grad_norm_async_fail_loud_(
                        actor_params, args.actor_grad_clip
                    )
                else:
                    actor_gn = v_loss.new_zeros(())

                if args.nextlat:
                    actor_trunk_grads = [
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_trunk_params
                    ]
                    critic_trunk_grads = [
                        value_grads[parameter]
                        if parameter in value_grads
                        else torch.zeros_like(parameter)
                        for parameter in nextlat_trunk_params
                    ]

                # Add the clipped critic gradient exactly as v5 did. With a shared
                # trunk this sums the two task signals; actor/critic heads remain disjoint.
                for parameter, gradient in value_grads.items():
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad + gradient
                    )

                if args.nextlat:
                    with torch.no_grad():
                        task_before = [
                            parameter.detach().clone()
                            for parameter in nextlat_trunk_params
                        ]
                task_optimizer.step()

                if args.nextlat:
                    task_updates = [
                        parameter.detach() - previous
                        for parameter, previous in zip(
                            nextlat_trunk_params, task_before
                        )
                    ]
                    agent.zero_grad(set_to_none=True)
                    predictor_step_norm = apply_private_optimizer_step(
                        nextlat_predictor_params,
                        predictor_optimizer,
                        nextlat_predictor_grads,
                    )
                    if actor_active:
                        (
                            _raw_predictive_updates,
                            _admitted_predictive_updates,
                            prednext_stats,
                        ) = apply_predictive_trunk_transaction(
                            nextlat_trunk_params,
                            predictive_trunk_optimizer,
                            nextlat_trunk_grads,
                            task_updates,
                            args.prednext_trust_ratio,
                            actor_gradients=actor_trunk_grads,
                            critic_gradients=critic_trunk_grads,
                            layout=prednext_layout,
                        )
                    else:
                        # The TPO breaker is a policy freeze for the rest of this
                        # iteration. Even a critic-safe representation delta would alter
                        # the shared actor, so neither advance predictive-trunk Adam nor
                        # admit a delta. The private predictor can still learn.
                        _, prednext_stats = _zero_admission_result(
                            task_updates,
                            prednext_layout,
                            nonfinite_veto=False,
                        )

            epochs_completed = epoch + 1
            if actor_active:
                actor_epochs_completed = epoch + 1
                # Circuit breaker (NOT an epoch break): past 3x the per-update KL
                # budget the actor stops, but the critic keeps training all epochs.
                # One epoch-level control-flow synchronization replaces one sync per
                # minibatch while retaining v13's float64 mean and breaker boundary.
                epoch_mean_kl = epoch_kl_sum / epoch_kl_count
                if epoch_mean_kl.item() > args.tpo_kl_breaker:
                    actor_active = False
            # target_kl (default None here) would also stop the critic; kept only as
            # an explicit opt-in escape hatch.
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        var_y = b_returns.var(correction=0)
        explained_var = torch.where(
            var_y == 0,
            torch.full_like(var_y, float("nan")),
            1.0 - (b_returns - b_values).var(correction=0) / var_y,
        )
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean()
        device_telemetry = {
            "losses/value_loss": v_loss,
            "losses/policy_loss": pg_loss,
            "losses/entropy": entropy_loss,
            "losses/old_approx_kl": old_approx_kl,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac_sum / clipfrac_count,
            "losses/explained_variance": explained_var,
            "losses/actor_grad_norm": actor_gn,
            "losses/critic_grad_norm": critic_gn,
            "debug/returns_mean": b_returns.mean(),
            "debug/returns_std": b_returns.std(),
            "debug/returns_absmax": b_returns.abs().max(),
            "debug/target_edge_mass": edge_mass,
            "debug/distpg_corr_with_gae": adv_corr,
            "debug/distpg_sign_agree": adv_sign_agree,
            "debug/u_edge_frac": u_edge_frac,
            "debug/sigma_mean": b_sigma.mean(),
            "losses/tpo_ce": tpo_ce,
        }
        if auto_alpha:
            device_telemetry.update(
                {
                    "losses/alpha": log_alpha.exp(),
                    "debug/squashed_entropy": (-logprobs).mean(),
                    "debug/soft_bootstrap_bonus": next_value_bonus.mean(),
                    "debug/soft_adv_std_ratio": policy_adv.std()
                    / (advantages.std() + 1e-8),
                }
            )
        if args.nextlat:
            device_telemetry.update(
                {
                    "losses/nextlat_prediction": nextlat_pred_loss,
                    "losses/nextlat_policy_kl": nextlat_kl_loss,
                    "losses/nextlat_trunk_grad_norm": nextlat_trunk_gn,
                    "losses/nextlat_predictor_grad_norm": nextlat_predictor_gn,
                    "debug/nextlat_latent_std": latent_batch_std,
                    "prednext/task_step_norm": prednext_stats["task_norm"],
                    "prednext/raw_predictive_step_norm": prednext_stats["predictive_norm"],
                    "prednext/admitted_step_norm": prednext_stats["admitted_norm"],
                    "prednext/predictor_step_norm": predictor_step_norm,
                    "prednext/raw_task_cosine": prednext_stats["raw_cosine"],
                    "prednext/accepted_fraction": prednext_stats["accepted_fraction"],
                    "prednext/global_cap_scale": prednext_stats["global_cap_scale"],
                    "prednext/nonfinite_veto": prednext_stats["nonfinite_veto"],
                    "prednext/actor_first_order": prednext_stats["actor_first_order"],
                    "prednext/critic_first_order": prednext_stats["critic_first_order"],
                    "prednext/block_actor_conflict_fraction": prednext_stats[
                        "block_actor_conflict"
                    ].float().mean(),
                    "prednext/block_critic_conflict_fraction": prednext_stats[
                        "block_critic_conflict"
                    ].float().mean(),
                }
            )
            for block_index, block_name in enumerate(prednext_block_names):
                device_telemetry[
                    f"prednext_blocks/{block_name}_actor_conflict"
                ] = prednext_stats["block_actor_conflict"][block_index]
                device_telemetry[
                    f"prednext_blocks/{block_name}_critic_conflict"
                ] = prednext_stats["block_critic_conflict"][block_index]
                device_telemetry[
                    f"prednext_blocks/{block_name}_admitted_ratio"
                ] = prednext_stats["block_admitted_ratio"][block_index]

        host_telemetry = synchronize_scalar_telemetry(device_telemetry)
        writer.add_scalar(
            "charts/learning_rate", task_optimizer.param_groups[0]["lr"], global_step
        )
        if auto_alpha:
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
        for tag, value in host_telemetry.items():
            writer.add_scalar(tag, value, global_step)
        for tag, value in {
            "debug/epochs_completed": epochs_completed,
            "debug/actor_epochs_completed": actor_epochs_completed,
            "debug/tpo_eta_solved": tpo_eta_solved,
            "debug/tpo_kl_achieved": tpo_kl_achieved,
            "debug/tpo_kl_base": tpo_kl_base,
            "debug/tpo_cap_engaged": tpo_cap_engaged,
            "debug/tpo_group_kl": tpo_group_kl,
            "debug/tpo_score_std_mean": tpo_score_std_mean,
            "debug/tpo_score_std_p90": tpo_score_std_p90,
            "debug/tpo_sigma_global": tpo_sigma_global,
            "debug/tpo_q_entropy": tpo_q_entropy,
        }.items():
            writer.add_scalar(tag, value, global_step)
        writer.add_scalar("charts/probe_sps_overhead", probe_seconds, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        del target_probs, b_target_probs

    envs.close()
    writer.close()
