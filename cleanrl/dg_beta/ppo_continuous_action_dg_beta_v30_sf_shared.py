# ============================================================================
# DG v30 -- v29 SF critic on v18 shell + SHARED ThinkTrunk (like lejepa_sf).
#
# Match of HalfCheetah-v4__dg_beta_v29_sf_v18_batch_nodg_v1 (SF + batch advnorm +
# nodg + raw rewards + score PG + KL trust) with:
#   * share_backbone=True: one ThinkTrunk for actor heads + SF critic head
#   * separate_grad_clip: critic/actor max-norms applied to their param sets
#   * critic fully updated BEFORE actor each iteration (already v29 order)
#
# Shared trunk Adam state is single-optimizer: freeze the other head's requires_grad
# per phase so trunk sees critic then actor, without dual Adam states on trunk.
# ============================================================================
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import (
    ActionEncoder,
    ARPredictor,
    MLP,
    SIGReg,
    StateEncoder,
)

EPS = 1e-6


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
    actor_lr: float = None
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1
    critic_epochs: int = 10
    norm_adv: bool = True
    # norm_adv_scope set below with v30 defaults (batch for nodg match)
    clip_coef: float = 0.2
    clip_coef_high: float = None
    use_ratio_clip: bool = False
    kl_trust: bool = True
    kl_target: float = 0.02
    kl_beta_init: float = 3.0
    kl_beta_min: float = 0.1
    kl_beta_max: float = 300.0
    kl_cap_ratio: float = 3.0
    kl_step_scale: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5  # fallback when separate_grad_clip=False
    target_kl: float = None

    # Shared backbone (v30 default True). separate_grad_clip uses distinct max-norms.
    share_backbone: bool = True
    separate_grad_clip: bool = True
    actor_grad_clip: float = 0.5
    critic_grad_clip: float = 0.5

    ret_ema_norm: bool = False
    ret_norm: str = "d3perc"
    ret_ema_decay: float = 0.998
    ret_quantile_lo: float = 0.05
    ret_quantile_hi: float = 0.95
    ret_perc_rate: float = 0.01
    ret_perc_floor: float = 1.0

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    reward_norm: bool = False  # v18: raw rewards

    dg_use_gate: bool = False  # match batch_nodg reference
    dg_surprisal: str = "peak_ref"
    dg_eta: float = 1.0
    dg_clip: float = 10.0
    dg_renorm: bool = False
    norm_adv_scope: str = "batch"  # match batch_nodg reference

    # --- SF / LeJEPA (from lejepa_sf_v2) ---
    emb_dim: int = 32
    ssl_hidden: int = 256
    pred_depth: int = 2
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = 4
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    ssl_lr: float = 5e-5
    ssl_weight_decay: float = 1e-3
    # T=128,B=16 -> n_seq=(128//4)*16=512; keep batch <= 512
    ssl_batch: int = 256
    ssl_epochs: int = 8
    ssl_grad_clip: float = 1.0
    sf_ridge: float = 1e-3
    sf_target_ema: float = 0.01
    # Per-dim standardization of Lambda targets for MSE / psi_raw readout:
    #   "ema"   = slow EMA (rate max(sf_target_ema, 1/count)) — lejepa_sf_v2 default
    #   "batch" = this-rollout mean/std only (no cross-iter EMA)
    #   "none"  = raw Lambda MSE; head lives in phi-accumulation units (no μ/σ)
    sf_std_mode: str = "ema"
    # Value for GAE: V = psi_raw @ w_r (reward-probe evaluation). If False, V = psi_raw.mean(-1)
    # (unweighted SF occupancy — ablates closed-form w_r rescaling of features → reward).
    # w_r is still solved each iter for logging (reward_probe_r2).
    sf_use_wr: bool = True
    sf_alpha: float = 1.0  # 1 = pure latent MSE; <1 mixes scalar direction
    mc_window: float = 50  # truncated-MC gate; shorter than lejepa (500) due to T=128
    critic_mtp_horizon: int = 6
    vf_coef: float = 0.5

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=False):
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
        if reward_norm:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            self.blocks.append(ThinkBlock(H * (k + 1), H, n_experts))
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
        self.sigreg = SIGReg(num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk)

    def forward(self, obs_seq, act_seq, mask_seq, sigreg_weight):
        emb = self.encoder(obs_seq)
        act_emb = self.action_encoder(act_seq)
        pred = self.pred_proj(self.predictor(emb, act_emb))
        err = (pred[:, :-1] - emb[:, 1:]).pow(2).mean(-1)
        pred_loss = (err * mask_seq).sum() / mask_seq.sum().clamp_min(1.0)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        return pred_loss + sigreg_weight * sigreg_loss, pred_loss, sigreg_loss


def phi_features(emb, obs, action):
    """phi = [e, s, a, a*a, 1] -- lejepa_sf_v2 reward-feature basis."""
    ones = emb[..., :1].new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, obs, action, action * action, ones], dim=-1)


def solve_reward_probe(phi, reward, ridge):
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
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        self.dg_surprisal = args.dg_surprisal
        self.share_backbone = args.share_backbone
        self.sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1
        self.mtp = args.critic_mtp_horizon
        if args.share_backbone:
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        # Zero-init SF head: first Lambda targets are pure discounted phi sums (MC).
        self.critic_head = layer_init(nn.Linear(H, self.mtp * self.sf_dim, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()

    def _actor_h(self, x):
        return self.trunk(x) if self.share_backbone else self.actor_trunk(x)

    def _critic_h(self, x):
        return self.trunk(x) if self.share_backbone else self.critic_trunk(x)

    def get_value_sf(self, x):
        h = self._critic_h(x)
        return self.critic_head(h).view(-1, self.mtp, self.sf_dim)

    def get_action_and_value(self, x, z=None):
        h = self._actor_h(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        if self.dg_surprisal == "mahalanobis":
            ell = (0.5 * ((z - dist.mean) / (dist.stddev + 1e-6)) ** 2).sum(1)
        else:
            mode = ((alpha - 1.0) / (alpha + beta - 2.0).clamp_min(EPS)).clamp(EPS, 1.0 - EPS)
            ell = dist.log_prob(mode).sum(1) - logp
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        # Value only when needed; actor path can skip second trunk pass if shared
        value_sf = self.get_value_sf(x)
        return z, action, logp, ell, entropy, value_sf, alpha, beta

    def actor_head_parameters(self):
        return list(self.alpha_head.parameters()) + list(self.beta_head.parameters())

    def critic_head_parameters(self):
        return list(self.critic_head.parameters())

    def trunk_parameters(self):
        if self.share_backbone:
            return list(self.trunk.parameters())
        return list(self.actor_trunk.parameters()) + list(self.critic_trunk.parameters())


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("minibatch", "batch", "batch_retstd")
    assert args.sf_std_mode in ("ema", "batch", "none"), f"bad sf_std_mode {args.sf_std_mode}"
    assert args.reward_norm is False or True  # explicit: default False (v18)
    n_seq = (args.num_steps // args.seq_len) * args.num_envs
    assert n_seq >= args.ssl_batch, f"ssl_batch={args.ssl_batch} > n_seq={n_seq}; lower ssl_batch"

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
    assert device.type == "cuda"

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma, reward_norm=args.reward_norm)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1

    agent = Agent(obs_dim, act_dim, args).to(device)
    ssl = LeJepaSSL(obs_dim, act_dim, args).to(device)

    trunk_params = agent.trunk_parameters()
    actor_head_params = agent.actor_head_parameters()
    critic_head_params = agent.critic_head_parameters()
    # Clip sets (include trunk for both phases when shared)
    actor_clip_params = trunk_params + actor_head_params if args.share_backbone else (
        list(agent.actor_trunk.parameters()) + actor_head_params
    )
    critic_clip_params = trunk_params + critic_head_params if args.share_backbone else (
        list(agent.critic_trunk.parameters()) + critic_head_params
    )
    # Single Adam so shared trunk has one momentum state; freeze other head per phase.
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    agent_opt = optim.Adam(
        [
            {"params": trunk_params, "lr": args.learning_rate},
            {"params": critic_head_params, "lr": args.learning_rate},
            {"params": actor_head_params, "lr": actor_base_lr},
        ],
        eps=1e-5,
    )
    ssl_opt = optim.AdamW(ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay)

    def set_phase(phase: str):
        """Critic-before-actor: only the active head + trunk require grad."""
        if phase == "critic":
            for p in actor_head_params:
                p.requires_grad_(False)
            for p in critic_head_params:
                p.requires_grad_(True)
            for p in trunk_params:
                p.requires_grad_(True)
        elif phase == "actor":
            for p in critic_head_params:
                p.requires_grad_(False)
            for p in actor_head_params:
                p.requires_grad_(True)
            for p in trunk_params:
                p.requires_grad_(True)
        else:
            raise ValueError(phase)

    kl_beta = args.kl_beta_init
    sf_mean = torch.zeros(sf_dim, device=device)
    sf_std = torch.ones(sf_dim, device=device)
    sf_stat_count = 0
    w_r = torch.zeros(sf_dim, device=device)

    def psi_raw(head_out):
        """Map head output into phi-accumulation units for V / TD bootstrap."""
        if args.sf_std_mode == "none":
            return head_out
        return head_out * sf_std + sf_mean

    def value_from_psi(psi):
        """Scalar V for GAE from psi in phi-accumulation units (B, sf_dim) or (..., sf_dim)."""
        if args.sf_use_wr:
            return psi @ w_r
        return psi.mean(dim=-1)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    alphas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    betas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros_like(obs)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    print(
        f"[v30] SF+shared_trunk: reward_norm={args.reward_norm} "
        f"T={args.num_steps} sf_dim={sf_dim} share={args.share_backbone} "
        f"sep_clip={args.separate_grad_clip} nodg={not args.dg_use_gate} "
        f"norm_adv={args.norm_adv}/{args.norm_adv_scope} "
        f"sf_std_mode={args.sf_std_mode} sf_use_wr={args.sf_use_wr}"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            agent_opt.param_groups[0]["lr"] = frac * args.learning_rate  # trunk
            agent_opt.param_groups[1]["lr"] = frac * args.learning_rate  # critic head
            agent_opt.param_groups[2]["lr"] = frac * actor_base_lr  # actor heads
            ssl_opt.param_groups[0]["lr"] = frac * args.ssl_lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                z, action, logprob, _, _, value_sf, alpha, beta = agent.get_action_and_value(next_obs)
                values[step] = value_from_psi(psi_raw(value_sf[:, 0]))
            zs[step] = z
            actions[step] = action
            alphas[step] = alpha
            betas[step] = beta
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            boundary = np.logical_or(terminations, truncations)
            valid = (~boundary).astype(np.float32)
            transition_next = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_mask is None:
                    final_mask = [fo is not None for fo in final_obs]
                for i, has in enumerate(final_mask):
                    if has and final_obs[i] is not None:
                        transition_next[i] = final_obs[i]
                        valid[i] = 1.0
                    elif boundary[i]:
                        valid[i] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(terminations, device=device, dtype=torch.float32)
            transition_boundaries[step] = torch.as_tensor(boundary, device=device, dtype=torch.float32)
            transition_valids[step] = torch.as_tensor(valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ---- SF targets + GAE with SF V -------------------------------------------------
        with torch.no_grad():
            flat = (-1,) + envs.single_observation_space.shape
            psi_next = psi_raw(agent.get_value_sf(next_obses.reshape(flat))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            psi_cur = psi_raw(agent.get_value_sf(obs.reshape(flat))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            next_v = value_from_psi(psi_next)

            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                boot = (1.0 - transition_terminations[t]) * transition_valids[t]
                cont = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_v[t] * boot - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * cont * lastgaelam
            returns = advantages + values

            emb_buf = ssl.encoder(obs.reshape(flat)).reshape(args.num_steps, args.num_envs, args.emb_dim)
            phi = phi_features(emb_buf, obs, actions)

            # Vector TD-lambda on phi (lejepa_sf residual form)
            sf_residual = torch.zeros_like(phi)
            last_sf = torch.zeros_like(phi[0])
            for t in reversed(range(args.num_steps)):
                boot = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
                cont = (1.0 - transition_boundaries[t]).unsqueeze(-1)
                delta_sf = phi[t] + args.gamma * boot * psi_next[t] - psi_cur[t]
                last_sf = delta_sf + args.gamma * args.gae_lambda * cont * last_sf
                sf_residual[t] = last_sf
            sf_target = sf_residual + psi_cur  # Lambda_t

            # w_r AFTER GAE so this iter's advantages used previous w_r (self-consistent)
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            w_r = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            reward_resid = flat_rew - flat_phi @ w_r
            reward_r2 = 1.0 - (reward_resid.var() / flat_rew.var().clamp_min(1e-12)).item()
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            resid_ac1 = ((resid_c[:-1] * resid_c[1:]).mean() / resid_var).item()

            sf_stat_count += 1
            flat_tgt = sf_target.reshape(-1, sf_dim)
            tgt_mean, tgt_std = flat_tgt.mean(0), flat_tgt.std(0).clamp_min(1e-6)
            if args.sf_std_mode == "batch":
                # No cross-iter EMA: standardize against this rollout only.
                sf_mean, sf_std = tgt_mean, tgt_std
            elif args.sf_std_mode == "ema":
                sf_rate = max(args.sf_target_ema, 1.0 / sf_stat_count)
                sf_mean = sf_mean + sf_rate * (tgt_mean - sf_mean)
                sf_std = sf_std + sf_rate * (tgt_std - sf_std)
            # "none": leave sf_mean/sf_std unused; head trains on raw Lambda.

            mtp = args.critic_mtp_horizon
            sf_mtp = sf_target.new_zeros((args.num_steps, args.num_envs, mtp, sf_dim))
            mtp_mask = torch.zeros((args.num_steps, args.num_envs, mtp), dtype=torch.bool, device=device)
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                sf_mtp[:valid_len, :, h] = sf_target[h : h + valid_len]
                mtp_mask[:valid_len, :, h] = valid_h
            if args.sf_std_mode != "none":
                sf_mtp = (sf_mtp - sf_mean) / sf_std

            # Unrigged EV gate: SF V vs truncated MC
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
            if int(mc_mask.sum()) >= 64:
                ev_sf = ev_score(values.reshape(-1)[mc_mask], mc_ret.reshape(-1)[mc_mask])
            else:
                ev_sf = float("nan")

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_alphas = alphas.reshape((-1,) + envs.single_action_space.shape)
        b_betas = betas.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, mtp, sf_dim)
        b_target_mask = mtp_mask.reshape(-1, mtp)

        raw_u_std = b_advantages.std().detach()
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            b_advantages = b_advantages / b_returns.std().clamp(min=args.ret_perc_floor)
        u_std = b_advantages.std().detach()

        b_inds = np.arange(args.batch_size)

        # ---- Critic FIRST: multi-epoch MSE on SF targets (trunk + critic head) --------
        set_phase("critic")
        v_loss = torch.zeros((), device=device)
        critic_gn = 0.0
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                value_sf = agent.get_value_sf(b_obs[mb])
                sf_err = value_sf - b_sf_target[mb]
                mask = b_target_mask[mb].to(sf_err.dtype)
                v_loss = (sf_err.pow(2).mean(-1) * mask).sum(-1).mean()
                if args.sf_alpha < 1.0:
                    wr_dir = w_r if args.sf_std_mode == "none" else (w_r * sf_std)
                    scalar_err = (sf_err @ wr_dir).pow(2)
                    v_loss = args.sf_alpha * v_loss + (1.0 - args.sf_alpha) * (
                        scalar_err * mask
                    ).sum(-1).mean()
                agent_opt.zero_grad(set_to_none=True)
                (args.vf_coef * v_loss).backward()
                clip = args.critic_grad_clip if args.separate_grad_clip else args.max_grad_norm
                critic_gn = float(nn.utils.clip_grad_norm_(critic_clip_params, clip))
                agent_opt.step()

        # ---- Actor SECOND: score + KL trust (trunk + actor heads) ---------------------
        set_phase("actor")
        gate_means, surp_means, chi_stds, clipfracs, kl_terms, scale_terms = [], [], [], [], [], []
        approx_kl = torch.zeros((), device=device)
        kl_cap = args.kl_target * args.kl_cap_ratio
        n_capped, n_steps, stop_actor = 0, 0, False
        pg_loss = torch.zeros((), device=device)
        actor_gn = 0.0

        for epoch in range(args.actor_epochs):
            if stop_actor:
                break
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                # Actor path only (still may call get_value_sf; freeze critic head grads)
                _, _, newlogprob, ell, entropy, _, new_a, new_b = agent.get_action_and_value(
                    b_obs[mb], b_zs[mb]
                )
                mb_adv = b_advantages[mb]
                if args.norm_adv and args.norm_adv_scope == "minibatch":
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                if args.dg_surprisal == "raw":
                    surprisal = (-newlogprob).clamp(-args.dg_clip, args.dg_clip)
                else:
                    surprisal = ell.clamp(0.0, args.dg_clip)
                chi = mb_adv * surprisal
                gate_diag = torch.sigmoid(chi / args.dg_eta).detach()
                w = gate_diag if args.dg_use_gate else torch.ones_like(gate_diag)
                if args.dg_renorm:
                    w = w / (w.mean() + 1e-8)

                if args.use_ratio_clip:
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    surr1 = -mb_adv * ratio
                    surr2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = (w * torch.max(surr1, surr2)).mean()
                else:
                    pg_loss = -(w * mb_adv * newlogprob).mean()

                if args.kl_trust:
                    kl = kl_divergence(Beta(b_alphas[mb], b_betas[mb]), Beta(new_a, new_b)).sum(1).mean()
                else:
                    kl = torch.zeros((), device=device)

                if args.kl_trust and args.kl_cap_ratio > 0 and args.kl_step_scale:
                    denom = max(kl_cap - args.kl_target, 1e-8)
                    scale = float(np.clip((kl_cap - kl.item()) / denom, 0.0, 1.0))
                    pg_loss = pg_loss * scale
                    scale_terms.append(scale)
                    if scale < 1.0:
                        n_capped = 1

                actor_loss = pg_loss + kl_beta * kl - args.ent_coef * entropy.mean()
                agent_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                clip = args.actor_grad_clip if args.separate_grad_clip else args.max_grad_norm
                actor_gn = float(nn.utils.clip_grad_norm_(actor_clip_params, clip))
                agent_opt.step()
                n_steps += 1

                with torch.no_grad():
                    gate_means.append(gate_diag.mean().item())
                    surp_means.append(surprisal.mean().item())
                    chi_stds.append(chi.std().item())
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    kl_terms.append(kl.item())
                    approx_kl = ((ratio - 1) - logratio).mean()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        mean_kl = float(np.mean(kl_terms)) if kl_terms else 0.0
        if args.kl_trust:
            if mean_kl > args.kl_target * 1.5:
                kl_beta = min(kl_beta * 2.0, args.kl_beta_max)
            elif mean_kl < args.kl_target / 1.5:
                kl_beta = max(kl_beta / 2.0, args.kl_beta_min)

        # ---- SSL once per iteration (after PPO) ----------------------------------------
        with torch.no_grad():
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
            seq_mask = seq_cont.cumprod(dim=1)[:, :-1]
        n_seq = seq_obs.shape[0]
        ssl_pred = ssl_sig = 0.0
        ssl_steps = 0
        for _ in range(args.ssl_epochs):
            perm = torch.randperm(n_seq, device=device)
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l = ssl(
                    seq_obs[idx], seq_act[idx], seq_mask[idx], args.sigreg_weight
                )
                ssl_opt.zero_grad(set_to_none=True)
                ssl_loss.backward()
                nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_opt.step()
                ssl_pred += pred_l.item()
                ssl_sig += sig_l.item()
                ssl_steps += 1
        ssl_pred /= max(ssl_steps, 1)
        ssl_sig /= max(ssl_steps, 1)

        # Legacy EV (rigged with GAE returns) + SF gate
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", agent_opt.param_groups[2]["lr"], global_step)
        writer.add_scalar("losses/actor_grad_norm", actor_gn, global_step)
        writer.add_scalar("losses/critic_grad_norm", critic_gn, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/analytic_kl", mean_kl, global_step)
        writer.add_scalar("losses/kl_beta", kl_beta, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/actor_steps", n_steps, global_step)
        writer.add_scalar("losses/kl_cap_hit", float(n_capped), global_step)
        writer.add_scalar("losses/kl_step_scale", float(np.mean(scale_terms)) if scale_terms else 1.0, global_step)
        writer.add_scalar("charts/u_std", u_std.item(), global_step)
        writer.add_scalar("charts/raw_u_std", raw_u_std.item(), global_step)
        writer.add_scalar("charts/dg_gate_mean", float(np.mean(gate_means)) if gate_means else 0.0, global_step)
        writer.add_scalar("gate/ev_sf_online", ev_sf if ev_sf == ev_sf else 0.0, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        writer.add_scalar("sf/reward_resid_ac_lag1", resid_ac1, global_step)
        writer.add_scalar("sf/w_r_norm", float(w_r.norm()), global_step)
        writer.add_scalar("sf/use_wr", float(args.sf_use_wr), global_step)
        writer.add_scalar("sf/std_mode_batch", float(args.sf_std_mode == "batch"), global_step)
        writer.add_scalar("sf/std_mode_none", float(args.sf_std_mode == "none"), global_step)
        writer.add_scalar("ssl/pred_loss", ssl_pred, global_step)
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        print(
            f"SPS: {int(global_step / (time.time() - start_time))}  "
            f"approx_kl={approx_kl.item():.4f} EV={explained_var:.3f} "
            f"ev_sf={ev_sf if ev_sf == ev_sf else float('nan'):.3f} r2={reward_r2:.3f} ac1={resid_ac1:.3f}"
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"agent": agent.state_dict(), "ssl": ssl.state_dict(), "w_r": w_r}, path)
        print(f"model saved to {path}")

    envs.close()
    writer.close()
