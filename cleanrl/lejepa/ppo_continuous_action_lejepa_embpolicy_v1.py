# ============================================================================
# LeJEPA embedding policy v1 -- NO GAE. Policy optimizes latent path quality.
#
# Motivation. In dg_beta_v29 / lejepa_sf, LeJEPA only supplies e inside φ=[e,s,a,a²,1]
# for SF+GAE. After raw s is in φ, e is optional; SSL is not load-bearing.
#
# Goal. Make the encoder+predictor the credit-assignment backbone:
#   1) NO GAE, NO scalar V^π, NO successor-feature TD on rewards.
#   2) Multi-step signal comes from PREDICTED embedding unrolls under the taken actions
#      (predictor must be accurate or advantages are wrong → LeJEPA is essential).
#   3) Policy score-gradient / clipped PG on embedding advantages vs a batch baseline.
#
# Algorithm (per iteration):
#   • Roll out π, store (s, a, z, logπ, r, done)
#   • e_t = stopgrad(encoder(s_t))
#   • Maintain e_star: EMA of mean e on high return-to-go states (or ridge u(e)=w·e)
#   • For each t, open-loop unroll ê with the PREDICTOR for H steps using a_{t:t+H-1}:
#         ê_0 = e_t
#         ê_{k+1} = g(ê_k, a_{t+k})     # LeJEPA next-embedding map
#         S_t = Σ_k γ^k u(ê_k)
#     Mask by episode boundaries so unrolls do not cross resets.
#   • A_t = S_t − mean(S)   (batch baseline; optional z-score)
#   • Actor: 1 epoch × minibatches of score -(A logπ) or PPO clip on A, + optional KL trust
#   • SSL: standard LeJEPA (pred MSE + SIGReg) once per iter on real (s,a) sequences
#
# u(e) modes:
#   "cos_good"  — cos(e, e_star); e_star EMA of e on top-quantile RTG states
#   "linear_r"  — w·e with closed-form ridge e→r (forces e to carry reward; still no GAE)
#
# Hypothesis. If predictor/encoder are useless, S_t is noise and learning fails. If they
# capture controllable structure, multi-step latent scores replace GAE without scalar bootstrap.
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
    num_minibatches: int = 32
    actor_epochs: int = 1
    norm_adv: bool = True  # z-score embedding advantages (batch)
    clip_coef: float = 0.2
    clip_coef_high: float = 0.28
    use_ratio_clip: bool = False
    kl_trust: bool = True
    kl_target: float = 0.02
    kl_beta_init: float = 3.0
    kl_beta_min: float = 0.1
    kl_beta_max: float = 300.0
    kl_cap_ratio: float = 3.0
    kl_step_scale: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = None
    reward_norm: bool = False

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- embedding credit (replaces GAE) ---
    emb_horizon: int = 8  # predictive unroll length H; must make multi-step pred matter
    emb_score: str = "cos_good"  # "cos_good" | "linear_r"
    good_rtg_quantile: float = 0.8  # top (1-q) states update e_star
    e_star_ema: float = 0.05
    emb_ridge: float = 1e-3
    # If True, also add a small bonus for matching REAL future e (diagnostic mix). Default off:
    # pure predictive score so SSL predictor is load-bearing.
    mix_real_future: float = 0.0

    # --- LeJEPA SSL ---
    emb_dim: int = 32
    ssl_hidden: int = 256
    pred_depth: int = 2
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    seq_len: int = 8  # >= emb_horizon for long contexts; SSL chunks this
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256
    ssl_lr: float = 5e-5
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 256
    ssl_epochs: int = 8
    ssl_grad_clip: float = 1.0

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
        self.blocks = nn.ModuleList([ThinkBlock(H * (k + 1), H, n_experts) for k in range(K)])
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
            num_frames=max(args.seq_len, args.emb_horizon + 1),
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

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, e, a):
        """e: (B,d), a: (B,act) -> ê': (B,d). Single-step latent dynamics."""
        e_seq = e.unsqueeze(1)
        a_emb = self.action_encoder(a).unsqueeze(1)
        pred = self.pred_proj(self.predictor(e_seq, a_emb, pos_offset=0))
        return pred[:, 0]

    def forward(self, obs_seq, act_seq, mask_seq, sigreg_weight):
        emb = self.encoder(obs_seq)
        act_emb = self.action_encoder(act_seq)
        pred = self.pred_proj(self.predictor(emb, act_emb))
        err = (pred[:, :-1] - emb[:, 1:]).pow(2).mean(-1)
        pred_loss = (err * mask_seq).sum() / mask_seq.sum().clamp_min(1.0)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        return pred_loss + sigreg_weight * sigreg_loss, pred_loss, sigreg_loss


def solve_ridge(x, y, ridge):
    x64 = x.double()
    gram = x64.T @ x64
    rhs = x64.T @ y.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(x.dtype)


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


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)

    def forward(self, x, z=None):
        h = self.trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, entropy, alpha, beta


def compute_rtg(rewards, boundaries, gamma):
    """Discounted return-to-go; cuts at episode boundaries."""
    T, B = rewards.shape
    rtg = torch.zeros_like(rewards)
    run = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        cont = 1.0 - boundaries[t]
        run = rewards[t] + gamma * cont * run
        rtg[t] = run
    return rtg


def predictive_scores(ssl, emb, actions, boundaries, args, u_fn):
    """S_t from open-loop predictor unrolls. emb/actions: (T,B,·). Returns (T,B)."""
    T, B, d = emb.shape
    H = args.emb_horizon
    device = emb.device
    scores = torch.zeros(T, B, device=device)
    # Vectorize over starting times by looping H (small) and T (moderate)
    # ê starts as e_t for each t; we process all (t,b) as batch TB
    e = emb.reshape(T * B, d)
    # Precompute validity: for each start t, step k is valid if no boundary in [t, t+k)
    for t in range(T):
        e_cur = emb[t].clone()  # (B,d)
        s = u_fn(e_cur)
        disc = torch.ones(B, device=device)
        valid = torch.ones(B, device=device)
        total = s * disc
        for k in range(H - 1):
            t_a = t + k
            if t_a >= T:
                break
            # cannot step across boundary at t_a
            valid = valid * (1.0 - boundaries[t_a])
            a = actions[t_a]
            e_cur = ssl.predict_next(e_cur, a)
            disc = disc * args.gamma
            total = total + valid * disc * u_fn(e_cur)
        scores[t] = total
    return scores


def real_future_scores(emb, boundaries, args, u_fn):
    """Same structure using REAL future embeddings (ablation / mix)."""
    T, B, d = emb.shape
    H = args.emb_horizon
    scores = torch.zeros(T, B, device=emb.device)
    for t in range(T):
        disc = torch.ones(B, device=emb.device)
        valid = torch.ones(B, device=emb.device)
        total = u_fn(emb[t])
        for k in range(1, H):
            t_f = t + k
            if t_f >= T:
                break
            valid = valid * (1.0 - boundaries[t_f - 1])
            disc = disc * args.gamma
            total = total + valid * disc * u_fn(emb[t_f])
        scores[t] = total
    return scores


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.emb_score in ("cos_good", "linear_r")
    assert args.emb_horizon >= 1
    n_seq = (args.num_steps // args.seq_len) * args.num_envs
    assert n_seq >= args.ssl_batch, f"ssl_batch={args.ssl_batch} > n_seq={n_seq}"

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
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.reward_norm)
            for i in range(args.num_envs)
        ]
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    actor = Actor(obs_dim, act_dim, args).to(device)
    ssl = LeJepaSSL(obs_dim, act_dim, args).to(device)
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor.parameters(), lr=actor_base_lr, eps=1e-5)
    ssl_opt = optim.AdamW(ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay)

    kl_beta = args.kl_beta_init
    e_star = F.normalize(torch.randn(args.emb_dim, device=device), dim=0)
    e_star_inited = False
    w_e = torch.zeros(args.emb_dim, device=device)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    actions = torch.zeros_like(zs)
    alphas = torch.zeros_like(zs)
    betas = torch.zeros_like(zs)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    print(
        f"[embpolicy_v1] NO-GAE latent PG: score={args.emb_score} H={args.emb_horizon} "
        f"mix_real={args.mix_real_future} kl_trust={args.kl_trust}"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            ssl_opt.param_groups[0]["lr"] = frac * args.ssl_lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                z, action, logprob, _, alpha, beta = actor(next_obs)
            zs[step] = z
            actions[step] = action
            alphas[step] = alpha
            betas[step] = beta
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            boundary = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            boundaries[step] = torch.as_tensor(boundary, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ---- Embedding advantages (NO GAE) --------------------------------------------
        with torch.no_grad():
            flat = obs.reshape(-1, obs_dim)
            emb = ssl.encode(flat).reshape(args.num_steps, args.num_envs, args.emb_dim)
            rtg = compute_rtg(rewards, boundaries, args.gamma)

            # Update e_star from high-RTG states
            flat_rtg = rtg.reshape(-1)
            flat_e = emb.reshape(-1, args.emb_dim)
            thr = torch.quantile(flat_rtg, args.good_rtg_quantile)
            good = flat_rtg >= thr
            if good.any():
                e_good = F.normalize(flat_e[good].mean(0), dim=0)
                if not e_star_inited:
                    e_star = e_good
                    e_star_inited = True
                else:
                    e_star = F.normalize(
                        (1 - args.e_star_ema) * e_star + args.e_star_ema * e_good, dim=0
                    )

            # linear_r probe on e (optional score; also logged)
            w_e = solve_ridge(flat_e, rewards.reshape(-1), args.emb_ridge)
            r2_e = 1.0 - ((rewards.reshape(-1) - flat_e @ w_e).var() / rewards.reshape(-1).var().clamp_min(1e-12)).item()

            def u_fn(e):
                # e: (B,d)
                if args.emb_score == "cos_good":
                    return (F.normalize(e, dim=-1) * e_star).sum(-1)
                return e @ w_e

            # Predictive scores REQUIRE the SSL predictor (load-bearing).
            S_pred = predictive_scores(ssl, emb, actions, boundaries, args, u_fn)
            S_real = real_future_scores(emb, boundaries, args, u_fn)  # diagnostic + optional mix
            if args.mix_real_future > 0:
                S = (1 - args.mix_real_future) * S_pred + args.mix_real_future * S_real
            else:
                S = S_pred

            # One-step prediction error (health of LeJEPA for policy)
            e_next_real = emb[1:].reshape(-1, args.emb_dim)
            e_cur = emb[:-1].reshape(-1, args.emb_dim)
            a_cur = actions[:-1].reshape(-1, act_dim)
            e_next_hat = ssl.predict_next(e_cur, a_cur)
            cont = (1.0 - boundaries[:-1]).reshape(-1)
            pred_mse = ((e_next_hat - e_next_real).pow(2).mean(-1) * cont).sum() / cont.sum().clamp_min(1.0)

            A = S - S.mean()
            if args.norm_adv:
                A = A / (A.std() + 1e-8)

        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape(-1, act_dim)
        b_alphas = alphas.reshape(-1, act_dim)
        b_betas = betas.reshape(-1, act_dim)
        b_adv = A.reshape(-1)
        b_inds = np.arange(args.batch_size)

        # ---- Actor -----------------------------------------------------------------
        kl_cap = args.kl_target * args.kl_cap_ratio
        kl_terms, scale_terms, clipfracs = [], [], []
        approx_kl = torch.zeros((), device=device)
        pg_loss = torch.zeros((), device=device)
        n_steps = 0
        for epoch in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, entropy, new_a, new_b = actor(b_obs[mb], b_zs[mb])
                mb_adv = b_adv[mb]
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                if args.use_ratio_clip:
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    surr1 = -mb_adv * ratio
                    surr2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(surr1, surr2).mean()
                else:
                    pg_loss = -(mb_adv * newlogprob).mean()

                if args.kl_trust:
                    kl = kl_divergence(Beta(b_alphas[mb], b_betas[mb]), Beta(new_a, new_b)).sum(1).mean()
                    if args.kl_cap_ratio > 0 and args.kl_step_scale:
                        denom = max(kl_cap - args.kl_target, 1e-8)
                        scale = float(np.clip((kl_cap - kl.item()) / denom, 0.0, 1.0))
                        pg_loss = pg_loss * scale
                        scale_terms.append(scale)
                else:
                    kl = torch.zeros((), device=device)

                loss = pg_loss + kl_beta * kl - args.ent_coef * entropy.mean()
                actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                actor_opt.step()
                n_steps += 1
                with torch.no_grad():
                    kl_terms.append(kl.item())
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    approx_kl = ((ratio - 1) - logratio).mean()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        mean_kl = float(np.mean(kl_terms)) if kl_terms else 0.0
        if args.kl_trust:
            if mean_kl > args.kl_target * 1.5:
                kl_beta = min(kl_beta * 2.0, args.kl_beta_max)
            elif mean_kl < args.kl_target / 1.5:
                kl_beta = max(kl_beta / 2.0, args.kl_beta_min)

        # ---- SSL (defines e and predictor — the credit model) ------------------------
        with torch.no_grad():
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            seq_cont = chunk_sequences(1.0 - boundaries, args.seq_len)
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

        # Correlation between embedding advantages and a GAE-free reward RTG (diagnostic)
        with torch.no_grad():
            a_flat = A.reshape(-1)
            r_flat = rtg.reshape(-1)
            az = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
            rz = (r_flat - r_flat.mean()) / (r_flat.std() + 1e-8)
            corr_rtg = (az * rz).mean().item()
            # Predictive vs real score agreement
            sp = S_pred.reshape(-1)
            sr = S_real.reshape(-1)
            spz = (sp - sp.mean()) / (sp.std() + 1e-8)
            srz = (sr - sr.mean()) / (sr.std() + 1e-8)
            corr_pred_real = (spz * srz).mean().item()

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/analytic_kl", mean_kl, global_step)
        writer.add_scalar("losses/kl_beta", kl_beta, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/actor_steps", n_steps, global_step)
        writer.add_scalar("emb/score_mean", S.mean().item(), global_step)
        writer.add_scalar("emb/adv_std", A.std().item(), global_step)
        writer.add_scalar("emb/corr_with_rtg", corr_rtg, global_step)
        writer.add_scalar("emb/corr_pred_vs_real", corr_pred_real, global_step)
        writer.add_scalar("emb/pred_next_mse", pred_mse.item(), global_step)
        writer.add_scalar("emb/e_to_r_r2", r2_e, global_step)
        writer.add_scalar("ssl/pred_loss", ssl_pred, global_step)
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        print(
            f"SPS: {int(global_step / (time.time() - start_time))}  "
            f"approx_kl={approx_kl.item():.4f} corr_rtg={corr_rtg:.3f} "
            f"corr_pred_real={corr_pred_real:.3f} pred_mse={pred_mse.item():.4f} "
            f"ssl_pred={ssl_pred:.4f} r2_e={r2_e:.3f}"
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"actor": actor.state_dict(), "ssl": ssl.state_dict(), "e_star": e_star, "w_e": w_e}, path)
        print(f"model saved to {path}")

    envs.close()
    writer.close()
