# ============================================================================
# DG v27 -- full-batch actor + separate actor/critic CUDA-graph compile + clip ablation.
#
# Base for clean ablations after v18's hot-U / sequential-score issues:
#   1) FULL-BATCH actor: exactly ONE optimizer step per rollout on all batch_size samples.
#      No 32 sequential score updates on stale off-policy minibatches.
#   2) Separate monolithic torch.compile(mode="reduce-overhead") on ActorNet and CriticNet
#      (not one fused agent graph). mark_step_begin + clone-for-backward each call.
#   3) Critic keeps HL-Gauss Dreamer3-bucket CE + multi-epoch minibatches (not MSE).
#   4) Clip ablation: --use-ratio-clip uses PPO clip-higher (0.2 / 0.28); default OFF is
#      pure score -(U log pi). KL trust OFF by default so clip vs noclip is not confounded.
#   5) Cool U default: batch_retstd (A / max(1, std(R))); gate OFF; raw rewards/returns.
#
# Hypothesis: single on-policy actor step removes within-iteration score bias; clip-higher
# arm isolates whether a ratio trust region still helps once updates are single-shot.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport, HLGaussSupport

EPS = 1e-6


def _cudagraph_step_begin(enabled: bool):
    if enabled:
        torch.compiler.cudagraph_mark_step_begin()


def _clone_for_cg(t: torch.Tensor) -> torch.Tensor:
    """Clone compiled outputs so the next CUDA-graph replay does not overwrite tensors
    still needed for backward under reduce-overhead."""
    return t.clone() if torch.is_tensor(t) else t


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

    # Separate actor / critic monolithic compile (CUDA graphs when reduce-overhead).
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    actor_lr: float = None
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    adv_mode: str = "gae"
    num_minibatches: int = 32  # critic only; actor is full-batch
    actor_epochs: int = 1  # always 1 full-batch step in this base
    critic_epochs: int = 10
    norm_adv: bool = True
    norm_adv_scope: str = "batch_retstd"  # cool U default
    clip_coef: float = 0.2
    clip_coef_high: float = 0.28  # clip-higher upper bound when use_ratio_clip
    use_ratio_clip: bool = False  # OFF = pure score; ON = PPO clip-higher
    # KL optional; OFF by default so clip ablation is clean.
    kl_trust: bool = False
    kl_target: float = 0.02
    kl_beta_init: float = 3.0
    kl_beta_min: float = 0.1
    kl_beta_max: float = 300.0
    kl_cap_ratio: float = 0.0
    kl_step_scale: bool = False
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = None

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
    critic_init_tau: float = 0.5

    critic_d3bucket: bool = True
    critic_num_bins: int = 511
    critic_v_min: float = -9.90353755128617
    critic_v_max: float = 9.90353755128617
    critic_sigma_ratio: float = 0.75
    critic_symlog: bool = True
    reward_norm: bool = False

    dg_use_gate: bool = False
    dg_surprisal: str = "peak_ref"
    dg_eta: float = 1.0
    dg_clip: float = 10.0
    dg_renorm: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=True):
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


class ActorNet(nn.Module):
    """Monolithic Beta actor (ThinkTrunk + alpha/beta heads). Tensor-only outputs."""

    def __init__(self, obs_dim, act_dim, hidden=64, k_blocks=3, n_experts=16, dg_surprisal="peak_ref"):
        super().__init__()
        self.dg_surprisal = dg_surprisal
        self.trunk = ThinkTrunk(obs_dim, hidden, k_blocks, n_experts)
        self.alpha_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)

    def forward(self, x, z=None):
        h = self.trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        if self.dg_surprisal == "mahalanobis":
            mean = dist.mean
            std = dist.stddev
            ell = (0.5 * ((z - mean) / (std + 1e-6)) ** 2).sum(1)
        else:
            mode = ((alpha - 1.0) / (alpha + beta - 2.0).clamp_min(EPS)).clamp(EPS, 1.0 - EPS)
            logp_mode = dist.log_prob(mode).sum(1)
            ell = logp_mode - logp
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, ell, entropy, alpha, beta


class CriticNet(nn.Module):
    """Monolithic HL-Gauss critic (ThinkTrunk + categorical head)."""

    def __init__(self, obs_dim, num_bins, hidden=64, k_blocks=3, n_experts=16,
                 v_min=-10.0, v_max=10.0, critic_init_tau=0.5):
        super().__init__()
        self.trunk = ThinkTrunk(obs_dim, hidden, k_blocks, n_experts)
        self.head = layer_init(nn.Linear(hidden, num_bins), std=0.1)
        with torch.no_grad():
            zc = torch.linspace(v_min, v_max, num_bins)
            self.head.bias.copy_(-0.5 * (zc / critic_init_tau) ** 2)

    def forward(self, x):
        return self.head(self.trunk(x))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("minibatch", "batch", "batch_retstd"), f"bad norm_adv_scope {args.norm_adv_scope}"
    assert args.actor_epochs == 1, "v27 actor is single full-batch step; actor_epochs must be 1"
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
    assert device.type == "cuda", "v27 requires CUDA"

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, reward_norm=args.reward_norm)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    if args.critic_d3bucket:
        hlg = Dreamer3BucketHLGaussSupport(
            num_bins=args.critic_num_bins,
            coord_min=args.critic_v_min,
            coord_max=args.critic_v_max,
            sigma_ratio=args.critic_sigma_ratio,
            device=device,
        )
    else:
        hlg = HLGaussSupport(
            num_bins=args.critic_num_bins,
            v_min=args.critic_v_min,
            v_max=args.critic_v_max,
            sigma_ratio=args.critic_sigma_ratio,
            device=device,
            use_symlog=args.critic_symlog,
        )

    actor = ActorNet(
        obs_dim, act_dim, hidden=args.hidden, k_blocks=args.k_blocks,
        n_experts=args.n_experts, dg_surprisal=args.dg_surprisal,
    ).to(device)
    critic = CriticNet(
        obs_dim, args.critic_num_bins, hidden=args.hidden, k_blocks=args.k_blocks,
        n_experts=args.n_experts, v_min=args.critic_v_min, v_max=args.critic_v_max,
        critic_init_tau=args.critic_init_tau,
    ).to(device)

    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor_params, lr=actor_base_lr, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    use_cg = bool(args.compile and args.compile_mode == "reduce-overhead")
    if args.compile:
        actor = torch.compile(actor, mode=args.compile_mode, dynamic=False)
        critic = torch.compile(critic, mode=args.compile_mode, dynamic=False)
        print(
            f"[v27] torch.compile separate ActorNet/CriticNet mode={args.compile_mode!r} "
            f"dynamic=False cg={use_cg} fullbatch_actor use_ratio_clip={args.use_ratio_clip}"
        )

    kl_beta = args.kl_beta_init
    ema_ret_mean, ema_ret_var, ema_ret_std, ema_ret_inited = 0.0, 1.0, 1.0, False
    ema_ret_lo, ema_ret_hi, ema_perc_inited = 0.0, 1.0, False

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    alphas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    betas = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            critic_opt.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                _cudagraph_step_begin(use_cg)
                z, action, logprob, _, _, alpha, beta = actor(next_obs)
                z = _clone_for_cg(z)
                action = _clone_for_cg(action)
                logprob = _clone_for_cg(logprob)
                alpha = _clone_for_cg(alpha)
                beta = _clone_for_cg(beta)
                _cudagraph_step_begin(use_cg)
                value_logits = _clone_for_cg(critic(next_obs))
                values[step] = hlg.to_scalar(value_logits).flatten()
            zs[step] = z
            alphas[step] = alpha
            betas[step] = beta
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            _cudagraph_step_begin(use_cg)
            next_value = hlg.to_scalar(_clone_for_cg(critic(next_obs))).reshape(1, -1)
            if args.adv_mode == "reward_minus_baseline":
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_ret = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_ret = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_ret
                advantages = returns - values
            else:
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
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            if args.ret_ema_norm:
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_quantile_lo, args.ret_quantile_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_norm == "d3perc":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    ema_ret_std = max(args.ret_perc_floor, ema_ret_hi - ema_ret_lo)
                    advantages = advantages / ema_ret_std
                else:
                    clamped = flat_ret.clamp(lo, hi)
                    batch_mean = clamped.mean().item()
                    batch_var = clamped.var(unbiased=False).item()
                    if not ema_ret_inited:
                        ema_ret_mean, ema_ret_var, ema_ret_inited = batch_mean, max(batch_var, 1.0), True
                    else:
                        d = 1.0 - args.ret_ema_decay
                        ema_ret_mean += d * (batch_mean - ema_ret_mean)
                        ema_ret_var += d * (batch_var - ema_ret_var)
                    ema_ret_std = max(ema_ret_var, 1e-10) ** 0.5
                    advantages = advantages / ema_ret_std

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_alphas = alphas.reshape((-1,) + envs.single_action_space.shape)
        b_betas = betas.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        raw_u_std = b_advantages.std().detach()
        raw_ret_std = b_returns.std().detach()

        if args.norm_adv and args.norm_adv_scope == "batch":
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            b_advantages = b_advantages / b_returns.std().clamp(min=args.ret_perc_floor)
        u_std = b_advantages.std().detach()

        b_inds = np.arange(args.batch_size)

        # ---- Critic: multi-epoch minibatches, HL-Gauss CE on raw returns ----
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _cudagraph_step_begin(use_cg)
                value_logits = _clone_for_cg(critic(b_obs[mb_inds]))
                target_probs = hlg.project(b_returns[mb_inds])
                v_loss = -(target_probs * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_opt.step()

        # ---- Actor: ONE full-batch step ----
        mb_adv = b_advantages
        if args.norm_adv and args.norm_adv_scope == "minibatch":
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        _cudagraph_step_begin(use_cg)
        _, _, newlogprob, ell, entropy, new_alpha, new_beta = actor(b_obs, b_zs)
        newlogprob = _clone_for_cg(newlogprob)
        ell = _clone_for_cg(ell)
        entropy = _clone_for_cg(entropy)
        new_alpha = _clone_for_cg(new_alpha)
        new_beta = _clone_for_cg(new_beta)

        logratio = newlogprob - b_logprobs
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
            from torch.distributions.kl import kl_divergence

            old_dist = Beta(b_alphas, b_betas)
            new_dist = Beta(new_alpha, new_beta)
            kl = kl_divergence(old_dist, new_dist).sum(1).mean()
        else:
            kl = torch.zeros((), device=device)

        actor_loss = pg_loss + kl_beta * kl - args.ent_coef * entropy.mean()
        # Pre-step clipfrac is meaningful for the surrogate; ratio≈1 before any prior
        # actor step on this batch (full-batch single update), so log after the step too.
        with torch.no_grad():
            gate_mean = gate_diag.mean().item()
            surp_mean = surprisal.mean().item()
            chi_std = chi.std().item()
            clipfrac_pre = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
        actor_opt.step()
        n_steps = 1

        # Post-step KL / clipfrac: measure the actual full-batch update vs rollout policy.
        with torch.no_grad():
            _cudagraph_step_begin(use_cg)
            _, _, post_logprob, _, _, post_alpha, post_beta = actor(b_obs, b_zs)
            post_logprob = _clone_for_cg(post_logprob)
            post_logratio = post_logprob - b_logprobs
            post_ratio = post_logratio.exp()
            approx_kl = ((post_ratio - 1) - post_logratio).mean()
            clipfrac = ((post_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
            if args.kl_trust:
                from torch.distributions.kl import kl_divergence

                mean_kl = float(
                    kl_divergence(Beta(b_alphas, b_betas), Beta(post_alpha, post_beta))
                    .sum(1)
                    .mean()
                    .item()
                )
            else:
                mean_kl = float(approx_kl.item())
            _ = clipfrac_pre  # kept for potential debug; post-step is the logged metric

        if args.kl_trust:
            if mean_kl > args.kl_target * 1.5:
                kl_beta = min(kl_beta * 2.0, args.kl_beta_max)
            elif mean_kl < args.kl_target / 1.5:
                kl_beta = max(kl_beta / 2.0, args.kl_beta_min)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/analytic_kl", mean_kl, global_step)
        writer.add_scalar("losses/kl_beta", kl_beta, global_step)
        writer.add_scalar("losses/ret_ema_std", ema_ret_std, global_step)
        writer.add_scalar("charts/raw_u_std", raw_u_std.item(), global_step)
        writer.add_scalar("charts/u_std", u_std.item(), global_step)
        writer.add_scalar("charts/ret_std", raw_ret_std.item(), global_step)
        writer.add_scalar("losses/actor_steps", n_steps, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/dg_gate_mean", gate_mean, global_step)
        writer.add_scalar("charts/dg_surprisal_mean", surp_mean, global_step)
        writer.add_scalar("charts/dg_chi_std", chi_std, global_step)
        writer.add_scalar("charts/use_ratio_clip", float(args.use_ratio_clip), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
