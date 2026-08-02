# PPO + IterThink v24 Beta + SUCCESSOR-FEATURE FACTORIZED CRITIC (sffactor v1).
# =====================================================================================
# Base: the "ppoadvnorm_batch_v1" reference (8278 @8M on HalfCheetah-v4) — i.e.
# ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_mbpercnorm_v2.py launched with
# --norm-adv --norm-adv-scope batch --no-ret-percnorm. That config is BAKED IN here
# (norm_adv=True, norm_adv_scope="batch", ret_percnorm removed): raw GAE + one
# batch-scope z-score is the sole advantage treatment. Beta actor, shared ThinkTrunk,
# decoupled dual-backward grad clip (0.25/0.25), clip-higher (0.2/0.28), tkl03 — all
# unchanged.
#
# THE ONE CHANGE: the Dreamer3 511-bucket HL-Gauss MTP critic head is replaced by a
# successor-feature factorization of the value:
#
#     V(s) = psi(s) . w
#
#   psi(s)  d-dim SF head on the shared trunk, trained by vector TD(lambda) toward
#           Psi_t = phi(s_t) + gamma * [(1-lam) * psi_old(s_{t+1}) + lam * Psi_{t+1}]
#           (targets computed once per rollout from the OLD psi, exactly like PPO's
#           scalar lambda-returns; sf_lambda=0 recovers plain 1-step SF-TD).
#           CONVENTION (the off-by-one that matters): psi(s_t) INCLUDES the current
#           features — occupancy starting NOW — so the "reward" in the recursion is
#           phi(s_t), not phi(s_{t+1}).
#   w       learned linear reward weight, trained by regression r_t ~= phi(s_t) . w.
#           Consistency: V(s_t) = E[sum gamma^k phi(s_{t+k})] . w ~= E[sum gamma^k r_{t+k}].
#
# PPO is otherwise untouched: GAE/returns are computed from V = psi.w with the real
# env rewards; the actor loss never changes. The critic loss becomes two terms:
# vf_coef * SF-TD MSE on psi  +  sf_w_coef * reward-regression MSE on w.
#
# phi STAGES (this file implements 1 and 2; learned-phi + SIGReg is the v2+ experiment):
#   phi_source="obs"      (DEFAULT, stage 1 plumbing test): phi = [normalized obs, 1].
#                         Tests only the factorization plumbing. If this doesn't
#                         ~match the 8278 baseline, everything downstream is broken.
#   phi_source="randproj" (stage 2): phi = [fixed random projection of obs, 1].
#                         Uninformative-but-non-degenerate basis — does SF-TD still
#                         learn a usable V?
#
# KNOWN APPROXIMATION: HalfCheetah's step reward depends on (s_t, a_t) (ctrl cost),
# so phi(s_t).w has an irreducible residual. Watch debug/reward_r2 — it bounds how
# well ANY state-only phi can factor this reward.
#
# HYPOTHESIS: the vector TD target gives the critic d error signals per step vs 1,
# so value-loss convergence (losses/sf_td_loss, losses/explained_variance) should
# match or beat the scalar/bucket critic at equal steps, and the psi latent carries
# dense credit a scalar head can't (tested later via reward-transfer refits of w).
# Falsifiable at stage 1: if returns fall well short of 8278, the factorization
# itself (not the representation) is losing value accuracy.
#
# RED-TEAM REVISIONS (pre-run, from adversarial review):
#   - psi_head has a BIAS (zero-init). HalfCheetah never terminates, so the constant
#     feature's SF is ~1/(1-gamma)=100 in every state — pure bias. Bias-free, that 100
#     must be built out of trunk features, so any actor-driven trunk drift creates a
#     large coherent TD error the critic immediately fights (the critic anchors the
#     shared trunk against the policy gradient). The bias absorbs it for free.
#   - randproj default is RANK-DEFICIENT (8 < obs_dim 17). A full-rank projection
#     spans the same space as phi=obs — a vacuous stage 2. Only a rank-deficient (or
#     nonlinear) basis actually tests SF-TD under an impoverished reward span.
#   - Diagnostics: per-dim TD-loss split (debug/td_const_share — if the constant
#     channel still dominates past ~1M steps the anchoring mechanism is live),
#     closed-form ridge w* gap (debug/w_gap_rel, debug/reward_r2_ridge — measures
#     Adam lag on w without changing training), epochs completed per iteration
#     (KL early stop gates CRITIC step budget via ACTOR KL — must be visible),
#     iteration-AVERAGED losses (last-minibatch-only logging is biased/noisy).
#   - Paired control run "sffactor_scalarmse_control_v1": identical base with a plain
#     Linear(H,1) MSE critic on returns. Disambiguates stage 1: the CE->MSE loss-family
#     swap is a confound; ~baseline control + low sffactor => factorization at fault;
#     low control => 8278 was never the right bar for an MSE-family critic.
#
# PRE-REGISTERED stage-1 criterion (fixed before results exist): PASS = episodic
# return within 10% of the baseline run at matched steps 2M/4M/8M AND explained
# variance within 0.05 of baseline from 2M on. FAIL = below on BOTH metrics at two
# consecutive checkpoints. In between: judge against the scalarmse control.
# =====================================================================================
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
    # ppoadvnorm_batch_v1 config, baked in: plain advantage standardization, once over
    # the whole rollout (batch scope), as the SOLE advantage treatment.
    norm_adv: bool = True
    norm_adv_scope: str = "batch"    # "batch" (reference) | "minibatch" (idiomatic PPO)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    vf_coef: float = 0.5             # weight on the SF-TD vector MSE (the "value loss")
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and SF heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # --- Successor-feature factorization ---
    phi_source: str = "obs"          # "obs" (stage 1: phi = normalized obs) | "randproj" (stage 2)
    phi_const: bool = True           # append a constant-1 feature (lets w carry a reward bias)
    randproj_dim: int = 8            # projection width for "randproj"; MUST be < obs_dim to be a
    #                                  real stage-2 test (full-rank = same span as phi=obs, vacuous)
    sf_lambda: Optional[float] = None  # lambda for the vector SF-TD(lambda) targets;
    #                                  None => use gae_lambda (consistent with GAE); 0.0 => 1-step SF-TD
    sf_w_coef: float = 1.0           # weight on the reward-regression MSE for w

    # Keep observation normalization; reward stays RAW (matches the reference run).
    normalize_reward: bool = False
    clip_reward: bool = False

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0

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

        # --- SF factorization: V(s) = psi(s) . w ---
        # phi is a FIXED function of the (already normalized+clipped) observation:
        #   "obs":      phi = [obs, 1]
        #   "randproj": phi = [P obs, 1], P a fixed Gaussian projection scaled by 1/sqrt(obs_dim)
        self.phi_source = args.phi_source
        self.phi_const = args.phi_const
        if self.phi_source == "obs":
            phi_dim = obs_dim
        elif self.phi_source == "randproj":
            phi_dim = args.randproj_dim
            proj = torch.randn(obs_dim, args.randproj_dim) / (obs_dim ** 0.5)
            self.register_buffer("phi_proj", proj)
        else:
            raise ValueError(f"unknown phi_source {self.phi_source}")
        self.sf_dim = phi_dim + (1 if self.phi_const else 0)
        # psi head: zero-init => psi=0 => V=0 prior (matches the base's neutral head).
        # WITH bias: the constant feature's SF (~100, state-independent under pure
        # truncation) lands in the bias instead of being built from trunk features,
        # so it can't turn the critic into a trunk anchor (see header).
        self.psi_head = layer_init(nn.Linear(H, self.sf_dim, bias=True), std=0.1)
        with torch.no_grad():
            self.psi_head.weight.zero_()
            self.psi_head.bias.zero_()
        # w: linear reward weight, zero-init (reward model starts at 0, like the value prior).
        self.w = nn.Parameter(torch.zeros(self.sf_dim))

        # v24: action distribution (unchanged from base).
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

    def get_phi(self, x):
        # Fixed feature map phi(s). No trainable parameters on this path.
        if self.phi_source == "obs":
            phi = x
        else:  # randproj
            phi = x @ self.phi_proj
        if self.phi_const:
            phi = torch.cat([phi, torch.ones_like(phi[:, :1])], dim=-1)
        return phi

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
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_psi(self, x):
        # Successor features psi(s): (B, sf_dim).
        _, critic_feat = self._trunks(x)
        return self.psi_head(critic_feat)

    def get_value(self, x):
        # V(s) = psi(s) . w   (w intentionally NOT detached here; callers that must
        # not backprop through w use torch.no_grad()).
        return self.get_psi(x) @ self.w

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
        psi = self.psi_head(critic_feat)
        if self.actor_dist == "gaussian":
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, psi

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk). w is part of
        # the value function: it gets the reward-regression gradient, clipped in the
        # same critic budget.
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.psi_head.parameters()) + [self.w]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("batch", "minibatch")
    sf_lambda = args.gae_lambda if args.sf_lambda is None else args.sf_lambda
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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    psis = torch.zeros((args.num_steps, args.num_envs, agent.sf_dim)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, psi = agent.get_action_and_value(next_obs)
                psis[step] = psi
                values[step] = psi @ agent.w
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
            flat_next_obses = next_obses.reshape((-1,) + envs.single_observation_space.shape)
            next_psis = agent.get_psi(flat_next_obses).reshape(
                args.num_steps, args.num_envs, agent.sf_dim
            )
            next_transition_values = next_psis @ agent.w  # (T, B)

            # SCALAR GAE — UNCHANGED PPO. Real env rewards, V = psi.w.
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

            # VECTOR SF-TD(lambda) targets Psi_t: exactly the scalar lambda-return
            # recursion with rewards -> phi(s_t) and values -> psi_old. Computed once
            # per rollout from the OLD psi (like PPO's `returns`), fixed across epochs.
            #   Psi_t = phi_t + gamma * [(1-lam) psi_old(s'_{t+1}) + lam Psi_{t+1}]
            # implemented via the GAE form: Psi_t = psi_old_t + A_t,
            #   A_t = delta_t + gamma*lam*(1-boundary)*A_{t+1},
            #   delta_t = phi_t + gamma*bnt_t*psi_old(s'_{t+1}) - psi_old_t.
            # At a truncation boundary the recursion falls back to the pure bootstrap
            # through the final observation (bnt=1 there); at termination bnt=0.
            phis = agent.get_phi(obs.reshape((-1,) + envs.single_observation_space.shape)).reshape(
                args.num_steps, args.num_envs, agent.sf_dim
            )
            psi_targets = torch.zeros_like(psis)
            last_vec_gaelam = torch.zeros(args.num_envs, agent.sf_dim, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
                lambda_nonterminal = (1.0 - transition_boundaries[t]).unsqueeze(-1)
                vec_delta = phis[t] + args.gamma * next_psis[t] * bootstrap_nonterminal - psis[t]
                last_vec_gaelam = vec_delta + args.gamma * sf_lambda * lambda_nonterminal * last_vec_gaelam
                psi_targets[t] = psis[t] + last_vec_gaelam

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_phis = phis.reshape(-1, agent.sf_dim)
        b_psi_targets = psi_targets.reshape(-1, agent.sf_dim)
        # ppoadvnorm_batch: one z-score over the whole rollout.
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_adv_normed = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # Iteration-averaged loss accumulators (last-minibatch-only logging is
        # biased low and noisy — see red-team notes).
        acc = {"sf_td": 0.0, "td_const": 0.0, "w": 0.0, "pg": 0.0, "n": 0}
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, new_psi = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_adv_normed[mb_inds]
                    else:
                        mb_advantages = b_advantages[mb_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                else:
                    mb_advantages = b_advantages[mb_inds]

                # Asymmetric "clip-higher": looser upper bound gives positive-advantage
                # actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # SF critic loss: vector TD on psi + reward regression on w. These two
                # REPLACE the scalar/bucket value loss; GAE above already consumed V=psi.w.
                td_err_per_dim = (new_psi - b_psi_targets[mb_inds]).pow(2).mean(dim=0)  # (sf_dim,)
                sf_td_loss = td_err_per_dim.mean()
                reward_pred = b_phis[mb_inds] @ agent.w
                w_loss = F.mse_loss(reward_pred, b_rewards[mb_inds])
                v_loss = args.vf_coef * sf_td_loss + args.sf_w_coef * w_loss
                with torch.no_grad():
                    acc["sf_td"] += sf_td_loss.item()
                    if args.phi_const:
                        acc["td_const"] += (td_err_per_dim[-1] / td_err_per_dim.sum().clamp_min(1e-12)).item()
                    acc["w"] += w_loss.item()
                    acc["pg"] += pg_loss.item()
                    acc["n"] += 1

                entropy_loss = entropy.mean()

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping (unchanged from base): value and
                    # policy gradients are backpropped separately, each clipped to its
                    # own max-norm, then summed on the shared trunk.
                    optimizer.zero_grad(set_to_none=True)
                    v_loss.backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            epochs_completed = epoch + 1
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # Reward-model quality: R^2 of phi.w against the actual step rewards over
        # this rollout. Bounds how well ANY V = psi.w can track the true value.
        # Also the closed-form ridge w* on this rollout (18x18 solve): the gap
        # ||w - w*||/||w*|| measures the SGD/Adam lag on w without changing training,
        # and reward_r2_ridge is the R^2 ceiling a lag-free w would achieve.
        with torch.no_grad():
            rew_resid = b_phis @ agent.w - b_rewards
            var_r = b_rewards.var().item()
            reward_r2 = float("nan") if var_r == 0 else 1.0 - rew_resid.var().item() / var_r
            gram = b_phis.T @ b_phis + 1e-3 * torch.eye(agent.sf_dim, device=device)
            w_star = torch.linalg.solve(gram, b_phis.T @ b_rewards)
            w_gap_rel = ((agent.w - w_star).norm() / w_star.norm().clamp_min(1e-8)).item()
            ridge_resid = b_phis @ w_star - b_rewards
            reward_r2_ridge = float("nan") if var_r == 0 else 1.0 - ridge_resid.var().item() / var_r

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        n_mb = max(acc["n"], 1)
        writer.add_scalar(
            "losses/value_loss",
            args.vf_coef * acc["sf_td"] / n_mb + args.sf_w_coef * acc["w"] / n_mb,
            global_step,
        )
        writer.add_scalar("losses/sf_td_loss", acc["sf_td"] / n_mb, global_step)
        writer.add_scalar("losses/w_loss", acc["w"] / n_mb, global_step)
        writer.add_scalar("losses/policy_loss", acc["pg"] / n_mb, global_step)
        writer.add_scalar("debug/td_const_share", acc["td_const"] / n_mb, global_step)
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("debug/w_gap_rel", w_gap_rel, global_step)
        writer.add_scalar("debug/reward_r2_ridge", reward_r2_ridge, global_step)
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
        writer.add_scalar("debug/reward_r2", reward_r2, global_step)
        writer.add_scalar("debug/w_norm", agent.w.detach().norm().item(), global_step)
        writer.add_scalar("debug/psi_target_absmax", b_psi_targets.abs().max().item(), global_step)
        writer.add_scalar("debug/psi_norm_mean", psis.norm(dim=-1).mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
