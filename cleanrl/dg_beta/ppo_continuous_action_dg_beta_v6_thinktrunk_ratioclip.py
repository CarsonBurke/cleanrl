# ============================================================================
# DG v6 -- v5 ThinkTrunk + a PPO RATIO-CLIP TRUST REGION on the actor.
#
# WHY. v5 grafted iterthink_v24_beta's high-capacity ThinkTrunk onto the faithful
# (no-trust-region) DG recipe. Capacity helped EARLY (climbed to ~1200 by ~180 eps)
# then COLLAPSED -- the unconstrained single-pass score-function over-concentrates the
# Beta into a degenerate deterministic policy. The faithful step-control sweep confirmed
# step size alone can't fix it:
#     actor_lr 3e-4 (default): hard collapse -> -582
#     actor_lr 1e-4 (alr1e4):  delayed collapse -> -329 @1M
#     actor_lr 3e-5 (alr3e5):  NO collapse but STALLS at ~850 @1M (far under the 4100 MLP ceiling)
# Lowering the LR either still collapses or kills the climb. That isolates the missing
# ingredient: not step SIZE but step DIRECTION control -- a trust region. The PPO+ThinkTrunk
# base (9490) has exactly this and does not collapse. This variant adds it back.
#
# WHAT. The faithful DG single score pass is replaced by the standard PPO clipped surrogate,
# trained for `actor_epochs` (>1, so the importance ratio meaningfully leaves 1 and the clip
# engages -- with a single on-policy pass ratio==1 and a clip is a no-op). The DG gate is kept
# as an OPTIONAL multiplicative factor on the surrogate (`gate_ppo` form): with --no-dg-use-gate
# (the established winning config) w==1 and this is EXACTLY PPO-clip; with the gate on it is the
# delight-reweighted PPO update. Everything else stays the faithful v5 recipe: separate
# actor/critic ThinkTrunks, HL-Gauss distributional critic (critic_epochs regression refits),
# raw advantages (norm_adv=False), ent_coef=0, separate optimizers.
#
# HYPOTHESIS. The ratio-clip trust region is what lets the high-capacity ThinkTrunk actor turn
# its fast early climb into a stable ascent past the ~4100 MLP ceiling instead of collapsing.
# If v6 climbs and holds, the trust region (not capacity, not step size) was the missing piece;
# if it still stalls, the bottleneck is elsewhere (advantage shaping / critic / exploration).
#
# ----------------------------------------------------------------------------
# DELIGHTFUL POLICY GRADIENT (DG) -- Beta-policy surprisal (kept for the optional gate).
# Paper: arXiv:2603.14608v1. ell_t = surprisal (>=0), chi_t = U_t*ell_t, w_t = sigmoid(chi/eta)
# DETACHED. "Amplify rare successes, suppress rare failures." For a Beta the literal -log pi
# inverts as the policy concentrates (log-normalizer -> -inf), so we use the peak-referenced
# surprisal ell = log pi(mode) - log pi(a) >= 0 (cancels the normalizer; the Beta analog of the
# Gaussian Mahalanobis term), or the moment-matched "mahalanobis" form 0.5||(z-mu)/sigma||^2.
# In v6 the gate, when enabled, multiplies the PPO clipped surrogate rather than a raw score.
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

from cleanrl.shared.hl_gauss import HLGaussSupport

EPS = 1e-6  # clamp for Beta samples / mode to keep log_prob finite


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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    actor_lr: float = None           # actor Adam LR; None -> learning_rate. With the ratio-clip
    #                                  trust region now bounding the step DIRECTION, the LR crutch
    #                                  is no longer load-bearing, but it stays available to bracket.
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 10           # MUST be >1: the ratio clip is a no-op on a single on-policy
    #                                  pass (ratio==1). 10 = standard PPO epoch reuse.
    critic_epochs: int = 10          # critic is pure regression -> refit more for good advantages
    norm_adv: bool = False           # faithful line: gate & surrogate see the RAW advantage U
    clip_coef: float = 0.2           # PPO ratio clip: LOWER bound 1-clip_coef (also upper if no high)
    clip_coef_high: float = None     # DAPO "clip-higher": looser UPPER bound 1+clip_coef_high. None ->
    #                                  symmetric. Loosening only the upper clip lets rare-but-good
    #                                  (low-density) actions raise their density -> preserves Beta spread
    #                                  -> directly counters the over-concentration collapse. Base uses 0.28.
    use_ratio_clip: bool = True      # True: PPO clipped-surrogate trust region (licenses actor_epochs>1
    #                                  data reuse). False: faithful DG score-function -(w*U*log pi), NO
    #                                  ratio/clip -> ONE on-policy pass only (use --actor-epochs 1).
    ent_coef: float = 0.0            # paper DG uses no entropy bonus
    max_grad_norm: float = 0.5       # standard PPO clip
    target_kl: float = None          # optional KL early-stop across actor epochs (off by default)

    # iterthink_v24_beta EXACT ThinkTrunk architecture (separate actor/critic trunks)
    hidden: int = 64                 # trunk hidden width H
    k_blocks: int = 3                # number of ThinkBlocks (DenseNet-style depth)
    n_experts: int = 16              # soft-MoE experts per ThinkBlock
    critic_init_tau: float = 0.5     # init value dist ~ N(0, tau^2): peaked-at-0 critic-head bias

    # HL-Gauss distributional critic
    critic_num_bins: int = 101       # categorical support size
    critic_v_min: float = -10.0      # support min (symlog space): symexp(-10) ~ -2.2e4 raw
    critic_v_max: float = 10.0       # support max (symlog space)
    critic_sigma_ratio: float = 0.75 # HL-Gauss label sigma as a fraction of bin width (paper sweet spot)
    critic_symlog: bool = True       # symlog-scale targets (DreamerV3-style) -> robust to value range

    # DG-specific
    dg_use_gate: bool = False        # v6 default OFF: established winning config is nogate (== pure PPO
    #                                  clip here). True => delight-gate the clipped surrogate (gate_ppo).
    dg_surprisal: str = "peak_ref"   # "peak_ref" (ell=logp(mode)-logp(a)>=0) | "mahalanobis" | "raw"
    dg_eta: float = 1.0              # temperature eta in w = sigmoid(chi/eta)
    dg_clip: float = 10.0            # paper C: clip on the surprisal ell
    dg_renorm: bool = True           # rescale gate to mean(w)=1 -> pure reallocation (no-op when nogate)

    # to be filled in runtime
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
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# iterthink_v24_beta EXACT backbone: the "ThinkTrunk" -- a DenseNet-style stack
# of K ThinkBlocks. Each block: bounded-convex residual gate mixing x_in and x0, a
# dense pre-act MLP branch (RMSNorm + ReLU^2), and a soft (full-softmax) MoE branch.
# Used here as SEPARATE actor/critic trunks (share_backbone=False).
# ============================================================================
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
    """Beta-policy actor + HL-Gauss distributional critic on iterthink_v24_beta's EXACT
    ThinkTrunk architecture, as SEPARATE actor/critic trunks (share_backbone=False).

    a = 2z - 1 (z the native Beta sample); the constant log-2 Jacobian cancels in both the
    score grad and the peak-referenced surprisal, so we work in native z-space throughout.
    """

    def __init__(self, envs, num_bins, hidden=64, k_blocks=3, n_experts=16,
                 v_min=-10.0, v_max=10.0, critic_init_tau=0.5):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = hidden
        self.critic_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.actor_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        # dreamer4 unimodal Beta: alpha, beta = 1 + softplus(.) > 1 (interior mode). std=0.01.
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        # Distributional value head over the HL-Gauss support: small weight + Gaussian-logit
        # bias so the initial value distribution is PEAKED at 0 (iterthink's critic init).
        self.critic_head = layer_init(nn.Linear(H, num_bins), std=0.1)
        with torch.no_grad():
            zc = torch.linspace(v_min, v_max, num_bins)
            self.critic_head.bias.copy_(-0.5 * (zc / critic_init_tau) ** 2)

    def get_value(self, x):
        """Returns raw value LOGITS over the HL-Gauss support (caller decodes/projects)."""
        return self.critic_head(self.critic_trunk(x))

    def _dist(self, x):
        h = self.actor_trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        return Beta(alpha, beta)

    def get_action_and_value(self, x, z=None):
        """Returns (z_native, action, logp, ell, entropy, value_logits)."""
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        if getattr(self, "dg_surprisal", "peak_ref") == "mahalanobis":
            mean = dist.mean
            std = dist.stddev
            ell = (0.5 * ((z - mean) / (std + 1e-6)) ** 2).sum(1)
        else:
            a, b = dist.concentration1, dist.concentration0
            mode = ((a - 1.0) / (a + b - 2.0).clamp_min(EPS)).clamp(EPS, 1.0 - EPS)
            logp_mode = dist.log_prob(mode).sum(1)
            ell = logp_mode - logp  # >= 0
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, ell, entropy, self.get_value(x)


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

    # HL-Gauss categorical support for the distributional value critic.
    hlg = HLGaussSupport(
        num_bins=args.critic_num_bins,
        v_min=args.critic_v_min,
        v_max=args.critic_v_max,
        sigma_ratio=args.critic_sigma_ratio,
        device=device,
        use_symlog=args.critic_symlog,
    )

    agent = Agent(
        envs, args.critic_num_bins, hidden=args.hidden, k_blocks=args.k_blocks,
        n_experts=args.n_experts, v_min=args.critic_v_min, v_max=args.critic_v_max,
        critic_init_tau=args.critic_init_tau,
    ).to(device)
    agent.dg_surprisal = args.dg_surprisal
    actor_params = list(agent.actor_trunk.parameters()) + list(agent.alpha_head.parameters()) + list(agent.beta_head.parameters())
    critic_params = list(agent.critic_trunk.parameters()) + list(agent.critic_head.parameters())
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor_params, lr=actor_base_lr, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    # Storage: store the NATIVE beta sample z (replayed to recompute logp at the same draw).
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
                z, action, logprob, _, _, value_logits = agent.get_action_and_value(next_obs)
                values[step] = hlg.to_scalar(value_logits).flatten()
            zs[step] = z
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

        # GAE (computed once from rollout values) -- U = raw advantage.
        with torch.no_grad():
            next_value = hlg.to_scalar(agent.get_value(next_obs)).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)

        # ---- Critic update: many epochs of pure HL-Gauss regression (no off-policy bias) ----
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                value_logits = agent.get_value(b_obs[mb_inds])
                target_probs = hlg.project(b_returns[mb_inds])
                v_loss = -(target_probs * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_opt.step()

        # ---- Actor update: PPO ratio-clip surrogate (the trust region), optionally delight-gated ----
        gate_means, surp_means, chi_stds, clipfracs = [], [], [], []
        approx_kl = torch.zeros((), device=device)
        for epoch in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, ell, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds])

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Optional DG delight gate (detached). nogate => w==1 => exact PPO clip.
                if args.dg_surprisal == "raw":
                    surprisal = (-newlogprob).clamp(-args.dg_clip, args.dg_clip)
                else:
                    surprisal = ell.clamp(0.0, args.dg_clip)
                chi = mb_adv * surprisal
                w = torch.sigmoid(chi / args.dg_eta).detach()
                if not args.dg_use_gate:
                    w = torch.ones_like(w)
                if args.dg_renorm:
                    w = w / (w.mean() + 1e-8)

                if args.use_ratio_clip:
                    # PPO clipped surrogate (trust region), gated by w. Asymmetric "clip-higher"
                    # when clip_coef_high is set: looser UPPER bound 1+clip_coef_high (lets rare-but-
                    # good actions raise density -> preserves Beta spread), tight LOWER bound 1-clip_coef.
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    surr1 = -mb_adv * ratio
                    surr2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = (w * torch.max(surr1, surr2)).mean()
                else:
                    # Faithful DG (paper Alg.2): dtheta += w * U * grad log pi <=> minimize -(w*U*log pi).
                    # No ratio, no clip -> only valid for a SINGLE on-policy pass (actor_epochs=1).
                    pg_loss = -(w * mb_adv * newlogprob).mean()
                actor_loss = pg_loss - args.ent_coef * entropy.mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_opt.step()

                with torch.no_grad():
                    gate_means.append(w.mean().item())
                    surp_means.append(surprisal.mean().item())
                    chi_stds.append(chi.std().item())
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    approx_kl = ((ratio - 1) - logratio).mean()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/dg_gate_mean", float(np.mean(gate_means)), global_step)
        writer.add_scalar("charts/dg_surprisal_mean", float(np.mean(surp_means)), global_step)
        writer.add_scalar("charts/dg_chi_std", float(np.mean(chi_stds)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
