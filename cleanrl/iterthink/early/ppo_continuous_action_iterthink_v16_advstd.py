# PPO + IterThink v16 (distributional critic + per-state advantage standardization). From v10.
#
# THE UNUSED DEGREE OF FREEDOM. Every variant in the line (v10 categorical, v12
# location-scale, v13/14 quantile) collapses the value DISTRIBUTION to a scalar
# (its mean) before it touches control. The distribution's SHAPE never reaches the
# policy gradient. v16 makes the distribution earn its keep at the control
# interface using a quantity the mean cannot provide: the per-state return std.
#
#   sigma(s) = sqrt( sum_i p_i(s) * (z_i - V(s))^2 )      (free from the categorical Z)
#
# PER-STATE ADVANTAGE STANDARDIZATION. Standard PPO standardizes advantages with a
# SINGLE global (per-minibatch) mean/std — implicitly assuming homoscedastic return
# noise. But Var[A_t | s_t] ≈ Var[G_t | s_t] = sigma(s_t)^2, which is strongly
# STATE-DEPENDENT on terminating tasks: near-fall states in Hopper/Walker have huge
# return spread (recover → high, fall → low), so their GAE advantage is both
# large-scale and high-variance (noisy), and under global normalization they
# dominate the policy gradient with low-SNR updates. v16 z-scores the advantage
# PER STATE first:
#
#     A~_t = A_t / max(sigma(s_t), floor)      then the usual global norm_adv.
#
# This equalizes gradient signal-to-noise across states (down-weights noisy
# high-variance states, up-weights confident ones) — a state-adaptive refinement of
# advantage normalization that is ONLY possible with a distributional critic. The
# floor (a fraction of the batch-mean sigma) caps the up-weight and guards against
# an over-confident critic. `adv_std_norm=False` recovers v10 EXACTLY (built-in
# ablation). Everything else — categorical λ-return target, peaked init, support,
# separate trunks — is v10 verbatim.
#
# Hypothesis: the gain is largest on the heteroscedastic terminating tasks
# (Hopper/Walker2d), where the iterthink distributional critic is here evaluated
# for the FIRST time. Control: iterthink_v10_distreturn (== adv_std_norm off).
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport


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
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

    # Per-state advantage standardization (the v16 extension). adv_std_norm=False
    # recovers v10 exactly. Divide A_t by sigma(s_t) clamped at adv_std_floor *
    # mean(sigma) so the per-state up-weight is capped at 1/adv_std_floor.
    adv_std_norm: bool = True
    adv_std_floor: float = 0.3

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
        self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        return self.critic_head(self.critic_trunk(x))

    def get_action_and_value(self, x, z=None):
        mean = self.actor_head(self.actor_trunk(x))
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if z is None:
            z = probs.sample()
        action = torch.tanh(z)
        log_det = 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = (probs.log_prob(z) - log_det).sum(1)
        value_logits = self.critic_head(self.critic_trunk(x))
        return action, z, log_prob, probs.entropy().sum(1), value_logits


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
    value_stds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

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
                action, z, logprob, _, value_logits = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                v_mean = (p * support).sum(dim=-1)
                values[step] = v_mean
                # Per-state return std sigma(s) from the categorical Z (the v16 signal).
                value_stds[step] = (p * (support - v_mean.unsqueeze(-1)) ** 2).sum(dim=-1).clamp_min(0).sqrt()
            actions[step] = action
            latent_zs[step] = z
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
            bootstrap_probs = torch.softmax(agent.get_value(next_obs), dim=-1)   # (B, n) = Z(s_T)
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
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
            # Distributional λ-return target (mean-matches `returns`).
            target_probs = distributional_lambda_returns(
                rewards, dones, next_done, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_value_stds = value_stds.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        # Floor for sigma(s): a fraction of the batch-mean std, fixed per iteration.
        # Caps the per-state advantage up-weight at 1 / adv_std_floor.
        adv_std_floor_val = args.adv_std_floor * b_value_stds.mean()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.adv_std_norm:
                    # Per-state standardization: equalize advantage SNR across states
                    # using the critic's predicted return std sigma(s_t).
                    mb_sigma = b_value_stds[mb_inds].clamp_min(adv_std_floor_val)
                    mb_advantages = mb_advantages / mb_sigma
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        writer.add_scalar("debug/value_std_mean", b_value_stds.mean().item(), global_step)
        # Coefficient of variation of sigma(s): how heteroscedastic the returns are
        # (≈0 → per-state norm ≈ global norm; large → v16 reweights strongly).
        writer.add_scalar(
            "debug/value_std_cv", (b_value_stds.std() / (b_value_stds.mean() + 1e-8)).item(), global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
