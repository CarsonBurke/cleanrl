# PPO + IterThink v36 (GHM: GEOMETRIC HORIZON MODEL CRITIC, MC-grounded). From v24.
#
# WHY v36. v29-v35 ran the paper TD-CFM bootstrap inside an on-policy loop and EV
# stuck near +0.2 (v35 plateaued at +0.25). Diagnosis: on on-policy data the
# bootstrap term of the TD-CFM mixture self-reinforces (the flow trained on its
# own integrate output) and the (1-gamma)=0.01 direct term cannot anchor it -> the
# successor measure collapses to the on-policy STATE MARGINAL, so V(s) = const +
# noise and the critic learns ~nothing useful about s. v36 takes the cleanest
# fix: DROP THE BOOTSTRAP. The flow trains on REAL discounted-horizon future
# states sampled at k~Geom(1-gamma) from each on-policy rollout (boundary
# -respecting in time and across episode terminals), so every target is by
# construction a state-conditional future -> marginal collapse is impossible.
# The critic IS the flow (no categorical scalar head): V(s) = (1-gamma)^{-1} *
# E_{X~m(.|s,pi(s))}[r_hat(X)] with K=64 CRN samples and a learned reward model.
# Conditioning is on TRUNK FEATURES (s = trunk_features(obs)), so the flow loss
# gradient shapes the shared trunk to be predictive of geometric futures -- the
# trunk gets the critic-side gradient it previously got from the categorical
# head. Hypothesis: state-conditional MC targets unlock useful EV (>0.5 minimum
# target vs v35's +0.25 plateau).
#
# --- inherited v24 notes ---
# v24 reaches 9498 on HalfCheetah-v4 (iterthink, Beta dist, rankgauss adv,
# distributional categorical critic, clip-higher 0.28, target_kl=0.03, shared
# backbone, dual-backward decoupled grad clip). v36 inherits everything EXCEPT
# the categorical critic, which is REPLACED by the GHM flow described above.
import math
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

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sinusoidal_time_embed(t, dim):
    """Sinusoidal features for a continuous flow-time t in [0,1]. t:(B,1) -> (B,dim).
    Geometric frequencies from ~1 to ~128 cycles over the unit interval so the velocity
    net can resolve a SHARPLY t-dependent field. (Lifted from v34.)"""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(2.0 * math.pi), math.log(2.0 * math.pi * 128.0),
                                     half, device=t.device))
    args = t * freqs  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowVelocity(nn.Module):
    """Rectified-flow velocity v(x_t, t | s, a) for the successor measure over OBSERVATIONS.
    v36 takes `s` as TRUNK FEATURES (not raw obs), so the flow loss gradient shapes the
    shared iterthink trunk. Otherwise the proven recipe (v34): SINUSOIDAL time embedding
    through a nonlinear MLP, a DEDICATED (s,a) conditioning encoder, CONCATENATED
    (x, t, cond) features, SiLU activations and LayerNorm.

    Args:
        cond_dim:  width of `cond_obs` features (here: iterthink trunk hidden = args.hidden).
        hidden:    velocity net hidden width (args.flow_hidden).
        act_dim:   action dimension.
        obs_dim:   observation dimension (x lives in obs-space).
        t_dim:     sinusoidal time-embed dimension.
    """
    def __init__(self, cond_dim, hidden, act_dim, obs_dim, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            layer_init(nn.Linear(t_dim, hidden)), nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
        )
        # Q-style conditioning: cond_in = cat([trunk_features, action]) along feature axis.
        self.cond_enc = nn.Sequential(
            layer_init(nn.Linear(cond_dim + act_dim, hidden)), nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
        )
        self.x_in = layer_init(nn.Linear(obs_dim, hidden))
        self.in_norm = nn.LayerNorm(3 * hidden)
        self.body = nn.Sequential(
            layer_init(nn.Linear(3 * hidden, hidden)), nn.SiLU(), nn.LayerNorm(hidden),
            layer_init(nn.Linear(hidden, hidden)), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.out = layer_init(nn.Linear(hidden, obs_dim), std=0.1)

    def forward(self, x, t, cond_obs, cond_act):
        te = self.t_mlp(sinusoidal_time_embed(t, self.t_dim))
        cond_in = torch.cat([cond_obs, cond_act], dim=-1)
        ce = self.cond_enc(cond_in)
        xe = self.x_in(x)
        h = self.in_norm(torch.cat([xe, te, ce], dim=-1))
        return self.out(self.body(h))


def integrate_flow_sa(net, cond_obs, cond_act, x0, end_t, steps):
    """Euler-midpoint integrate dx/dtau = net(x, tau, cond_obs, cond_act) from tau=0 to
    per-sample end_t (B,1). (Lifted from v34.)"""
    x = x0
    dt = end_t / steps
    for i in range(steps):
        x = x + dt * net(x, (i + 0.5) * dt, cond_obs, cond_act)
    return x


def sample_geometric_future_targets(obs_buf, next_obs_buf, dones_buf, gamma, device):
    """Sample one geometric-horizon future state X ~ m^pi_geom(.|s_t,a_t) per (t,e) from
    the on-policy rollout, clipped to within-buffer and within-episode.

    For each (t, e):
      - Draw k ~ Geom(1-gamma) (k>=0 steps after t to look) by walking j = t, t+1, ...
        with per-step Bernoulli(1-gamma) STOP probability.
      - If a terminal occurred at step j (dones_buf[j,e]==1), the last in-episode
        successor we can use is next_obs_buf[j-1,e] if j > t. If j == t, no valid
        successor: future_mask=0. STOP.
      - If we walk past the buffer edge (j > T-1), use next_obs_buf[T-1,e] if
        dones_buf[T-1,e]==0 (mask=1) else mask=0. STOP.

    This is the MC realization of the Bellman recursion for the successor measure,
    boundary-respecting. T=2048, N=16 so the Python loop is cheap relative to flow
    training itself.

    INPUT shapes:
      obs_buf:      (T, N, D)
      next_obs_buf: (T, N, D)   next_obs returned by envs.step (RESET obs on done)
      dones_buf:    (T, N)      dones_buf[t,e]=1 iff a terminal occurred at step t

    OUTPUT (on `device`, dtype=obs_buf.dtype):
      future_obs:  (T, N, D)
      future_mask: (T, N)       1 where a valid in-episode successor exists, else 0
    """
    # Vectorized boundary-respecting MC successor-state sampling. Same semantics as the
    # original per-(t,e) Python walk, but no .item() / CUDA syncs per step.
    T, N, D = obs_buf.shape
    p_stop = 1.0 - gamma
    is_done = dones_buf.to(torch.bool)                                   # (T, N)
    # k ~ Geom(p_stop), 0-based count of failures before first success. k=0 means stop at j=t.
    k_all = torch.distributions.Geometric(probs=torch.tensor(p_stop, device=device)).sample((T, N)).long()
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, N)     # (T, N)
    e_idx = torch.arange(N, device=device).unsqueeze(0).expand(T, N)     # (T, N)
    naive_j = (t_idx + k_all).clamp_max(T - 1)                            # naive geometric pick
    # next_done_idx[t,e] = smallest j >= t with dones[j,e]==1, else T (sentinel). Reverse scan.
    next_done_idx = torch.full((T, N), T, device=device, dtype=torch.long)
    last_done = torch.full((N,), T, device=device, dtype=torch.long)
    for j in range(T - 1, -1, -1):
        last_done = torch.where(is_done[j], torch.full_like(last_done, j), last_done)
        next_done_idx[j] = last_done
    # last_valid_idx[t,e]: highest j in [t, T-1] such that next_obs_buf[j,e] is a valid
    # in-episode successor of s_t. If next_done_idx[t,e] > t, that's next_done_idx-1
    # (the step BEFORE the next done — its next_obs is the last in-episode obs). If
    # next_done_idx[t,e] == T (no done before buffer end), it's T-1. If next_done_idx[t,e]
    # == t (t itself is terminal), there is no valid successor (handled by mask below).
    last_valid_idx = (next_done_idx - 1).clamp_min(0).clamp_max(T - 1)   # (T, N)
    # effective gather index: respect both the geometric pick AND the episode/buffer boundary.
    effective_j = torch.minimum(naive_j, last_valid_idx)                  # (T, N)
    # mask: valid iff t itself is not a terminal AND there exists a successor in [t, T-1]
    # in the same episode (i.e., last_valid_idx >= t).
    valid = (~is_done) & (last_valid_idx >= t_idx)                        # (T, N)
    future_obs = next_obs_buf[effective_j, e_idx]                         # (T, N, D)
    future_obs = future_obs * valid.to(obs_buf.dtype).unsqueeze(-1)       # zero where invalid
    future_mask = valid.to(obs_buf.dtype)
    return future_obs, future_mask


def ghm_mc_loss(flow_net, agent, b_obs, b_actions, b_future_obs, b_future_mask):
    """Pure MC-grounded conditional flow matching. NO bootstrap, NO target net.

    X1 = sampled real geometric-horizon future state (k ~ Geom(1-gamma), boundary-clipped).
    Conditioning is on trunk features -> flow loss gradient shapes the trunk."""
    f_s = agent.trunk_features(b_obs)
    x1 = b_future_obs
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    cfm = ((flow_net(xt, t, f_s, b_actions) - (x1 - x0)) ** 2).sum(-1)
    denom = b_future_mask.sum().clamp_min(1.0)
    return (cfm * b_future_mask).sum() / denom


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
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # v36 GHM CRITIC. Value comes ENTIRELY from a successor-measure flow trained on
    # MC samples of geometric-horizon future states + a learned reward model.
    flow_gamma: float = 0.99          # successor-measure horizon; matches GAE gamma
    flow_hidden: int = 256            # velocity-net hidden width
    flow_t_dim: int = 128             # sinusoidal flow-time embedding dimension
    flow_steps: int = 8               # Euler steps for integration (general)
    flow_value_steps: int = 8         # Euler steps when sampling future obs for V(s)
    flow_value_samples: int = 64      # # future-obs samples (common random numbers) averaged for V(s)
    flow_coef: float = 1.0            # weight of the flow MC loss
    reward_coef: float = 1.0          # weight of the reward-regression loss

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage. See header.
    #   "v10" | "tanh_gae" | "clip_z" | "rankgauss" | "rankgauss_signed"
    #   | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 direct-log-std: state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
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
        # init +4 -> g ~ 0.982 -> x_in ~ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel gamma).
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
            # One trunk feeds both the actor and the flow conditioning.
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # v24 action distribution heads.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
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

        # v36 GHM CRITIC. V(s) = 1/(1-g) * E_{a~pi}[ E_{X~m(.|s,a)}[ r_hat(X) ] ],
        # where m is the (s,a)-conditioned successor measure represented by `flow_net`.
        # Conditioning is on TRUNK FEATURES (cond_dim = H), so flow gradient shapes the trunk.
        self.flow_gamma = args.flow_gamma
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.flow_net = FlowVelocity(
            cond_dim=H, hidden=args.flow_hidden, act_dim=act_dim,
            obs_dim=obs_dim, t_dim=args.flow_t_dim,
        )
        # Reward model is s-only, exactly v34's spec.
        self.reward_model = nn.Sequential(
            layer_init(nn.Linear(obs_dim, args.flow_hidden)), nn.Tanh(),
            layer_init(nn.Linear(args.flow_hidden, args.flow_hidden)), nn.Tanh(),
            layer_init(nn.Linear(args.flow_hidden, 1), std=1.0),
        )
        # Common-random-numbers noise basis for the flow value; refreshed in-place once per
        # training iteration (see main loop) so V(s) is a deterministic function of s within
        # an iteration.
        self.register_buffer("value_noise", torch.randn(args.flow_value_samples, obs_dim))

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
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
        log_det_fn = lambda z: 0.0
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def trunk_features(self, x):
        """Return the iterthink trunk's final feature tensor (B, H).
        Used as the flow's conditioning `s`. With share_backbone=True this is the same
        feature tensor consumed by the actor heads, so the flow loss shapes the shared
        trunk (the same role v24's categorical critic head played)."""
        if self.share_backbone:
            return self.trunk(x)
        return self.critic_trunk(x)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for beta).
        # When replaying from the buffer it is passed back in; log_prob is recomputed at
        # the same native sample (v21's z-replay, generalized).
        actor_feat, _ = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        return action, z, log_prob, dist.entropy().sum(1)

    def flow_value(self, x, steps):
        """V(s) = (1-flow_gamma)^-1 * E_{a~pi}[ E_{X~m(.|s,a)}[ r_hat(X) ] ].

        One stochastic a~pi per state; K=value_noise.shape[0] CRN samples through the flow.
        The action sample is under no_grad -- V(s) treats pi as the current policy but
        blocks gradient flow from V back into the actor."""
        with torch.no_grad():
            a, _, _, _ = self.get_action_and_value(x)  # sample a ~ pi
        f_s = self.trunk_features(x)
        K = self.value_noise.shape[0]
        B = x.shape[0]
        f_rep = f_s.repeat_interleave(K, 0)
        a_rep = a.repeat_interleave(K, 0)
        z_rep = self.value_noise.repeat(B, 1)
        ones = torch.ones(B * K, 1, device=x.device)
        s_future = integrate_flow_sa(self.flow_net, f_rep, a_rep, z_rep, ones, steps)
        r = self.reward_model(s_future).squeeze(-1).view(B, K).mean(1)
        return r / (1.0 - self.flow_gamma)

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk, the flow_net,
        # and the reward_model). v36 replaces v24's categorical critic head with the
        # GHM flow + reward; the trunk receives the flow loss's gradient via cond_enc.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return (
            list(trunk.parameters())
            + list(self.flow_net.parameters())
            + list(self.reward_model.parameters())
        )


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
    # Param groups for decoupled clipping. With a shared backbone the trunk appears in
    # BOTH lists (it receives policy and critic gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # dones_buf[t,e] = 1 iff a terminal occurred at step t (i.e., next_obs_buf[t,e] is RESET).
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
        # function of s for this whole rollout+update.
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
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_t = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            next_obs_buf[step] = next_obs_t           # s_{t+1} (RESET obs if terminal at step t)
            dones_buf[step] = next_done               # 1 iff a terminal occurred at step t
            next_obs = next_obs_t

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            # Bootstrap V(s_T). flow_value samples a fresh a ~ pi internally.
            next_value = agent.flow_value(next_obs, args.flow_value_steps).reshape(1, -1)
            # Scalar GAE (means) -- unchanged from v24.
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
            # MC-grounded geometric-future-state sample for the flow target.
            future_obs_buf, future_mask_buf = sample_geometric_future_targets(
                obs, next_obs_buf, dones_buf, args.flow_gamma, device,
            )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_next_obs = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_rewards = rewards.reshape(-1)
        b_future_obs = future_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_future_mask = future_mask_buf.reshape(-1)
        b_dones = dones_buf.reshape(-1)
        # mask_rew: only train reward on transitions that did NOT cross an episode boundary
        # (lifted from v34 verbatim: `mask_rew = 1 - dones_buf`).
        b_reward_mask = 1.0 - b_dones

        # Policy advantage: reshape the GAE per `adv_transform`.
        gae = b_advantages
        if args.adv_transform == "v10":
            b_policy_adv = gae
        elif args.adv_transform == "tanh_gae":
            gz = (gae - gae.mean()) / (gae.std() + 1e-8)
            b_policy_adv = torch.tanh(gz / args.tanh_kappa)
        elif args.adv_transform == "clip_z":
            gz = (gae - gae.mean()) / (gae.std() + 1e-8)
            b_policy_adv = gz.clamp(-args.clip_z_c, args.clip_z_c)
        elif args.adv_transform == "rankgauss":
            n = gae.numel()
            ranks = gae.argsort().argsort().to(torch.float32)
            uq = (ranks + 0.5) / n
            c = args.cdf_probit_clamp
            centered = (2.0 * uq - 1.0).clamp(-c, c)
            b_policy_adv = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        elif args.adv_transform == "rankgauss_signed":
            c = args.cdf_probit_clamp
            b_policy_adv = torch.zeros_like(gae)
            for side in (gae > 0, gae < 0):
                if side.any():
                    g = gae[side]
                    r = g.argsort().argsort().to(torch.float32)
                    half = (r + 0.5) / float(g.numel())
                    uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)
                    ctr = (2.0 * uq - 1.0).clamp(-c, c)
                    b_policy_adv[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        elif args.adv_transform == "rankgauss_temp":
            n = gae.numel()
            ranks = gae.argsort().argsort().to(torch.float32)
            uq = (ranks + 0.5) / n
            c = args.cdf_probit_clamp
            centered = (2.0 * uq - 1.0).clamp(-c, c)
            z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
            b_policy_adv = torch.tanh(z / args.rank_tanh_kappa)
        elif args.adv_transform == "rankgauss_signmag":
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
        flow_losses, reward_losses = [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy = agent.get_action_and_value(
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

                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()

                # v36 critic-side losses. flow_loss touches the trunk (via trunk_features
                # inside ghm_mc_loss); reward_loss is local to reward_model.
                flow_loss = ghm_mc_loss(
                    agent.flow_net, agent,
                    b_obs[mb_inds], b_actions[mb_inds],
                    b_future_obs[mb_inds], b_future_mask[mb_inds],
                )
                r_pred = agent.reward_model(b_next_obs[mb_inds]).squeeze(-1)
                mrw = b_reward_mask[mb_inds]
                reward_loss = (((r_pred - b_rewards[mb_inds]) ** 2) * mrw).sum() / mrw.sum().clamp_min(1.0)
                critic_total = args.flow_coef * flow_loss + args.reward_coef * reward_loss

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop critic and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic gradient cannot
                    # swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    critic_total.backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped critic grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the stashed
                    # clip_critic(d critic / d trunk). flow_net / reward_model get critic grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + critic_total
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                flow_losses.append(flow_loss.item())
                reward_losses.append(reward_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/flow_loss", float(np.mean(flow_losses)), global_step)
        writer.add_scalar("losses/reward_loss", float(np.mean(reward_losses)), global_step)
        writer.add_scalar("losses/value_std", values.std().item(), global_step)
        writer.add_scalar("losses/value_mean", values.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/future_mask_frac", b_future_mask.mean().item(), global_step)
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
