# PPO + IterThink v14 (MAXIMUM-ENTROPY soft distributional value). From v12/v13.
#
# v12's location-scale critic (scalar value mu + auxiliary distributional residual
# shape head) reaches ~6000-6800 peak on HalfCheetah. v13 added SAC-style maximum
# entropy via a SOFT reward but FAILED: with NormalizeReward making rewards unit-
# scale, SAC's default alpha=0.1 over-weighted entropy (the discounted bonus
# accumulated to ~40, dwarfing the unit-scale task value), AND the dual updated
# only ONCE per iteration with lr 3e-4 so alpha was effectively frozen at init.
# Net: the policy was paid ~0.4/step to stay random -> never learned (return -118).
#
# v14 fixes the OPERATING REGIME so max-ent acts as an entropy FLOOR, not an early
# objective hijack:
#
#   soft reward:   r~_t = r_t - alpha * log pi(a_t | s_t)
#
#   - alpha starts NEAR ZERO (0.01) and is auto-tuned SAC-style toward target
#     entropy H_bar = -act_dim, but the dual now steps EVERY MINIBATCH on the
#     CURRENT policy's logpi (≈128 updates/iter vs 1) with lr 1e-3. Early policy
#     entropy (~4) >> H_bar (-6) so alpha decays to ≈0: v14 behaves like v12 with
#     no entropy interference. Only once the policy would over-commit (entropy
#     dropping below H_bar) does alpha rise to HOLD entropy at the floor -- exactly
#     the regime that targets v12's late premature value-collapse / seed-variance
#     (the bad seed collapsed entropy/EV; a floor prevents that) without paying the
#     early over-exploration tax that killed v13.
#   - log pi carries the tanh log-det (stable form): correct squashed-policy entropy.
#   - The residual recursion runs on the SOFT GAE deltas, so the distributional
#     backup propagates the soft return distribution for free. No Q(s,a) needed.
#   - No separate ent_coef bonus (entropy lives inside the soft advantage).
#   - Reported episodic_return stays the RAW env return: benchmark metric unaffected.
# Control: iterthink_v12_locscale. Fixes: iterthink_v13_maxent.
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
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0   # entropy now enters via the SOFT reward, not a bonus
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Location-scale critic (v12). mu = scalar value head (fast mean). shape =
    # categorical over a tight relative support c modelling the return residual.
    num_bins: int = 511
    rel_range: float = 5.0
    critic_init_tau: float = 0.5
    shape_coef: float = 1.0

    # Maximum-entropy (v14). Soft reward r~ = r - alpha*logpi; alpha auto-tuned to
    # a target policy entropy (SAC dual), stepped per-minibatch. target_entropy
    # defaults to -act_dim. alpha starts near zero so it acts as a late entropy
    # FLOOR rather than an early objective (the v13 failure mode).
    alpha: float = 0.01           # initial temperature (near zero)
    autotune_alpha: bool = True
    target_entropy: float = None  # None -> -act_dim
    alpha_lr: float = 1e-3

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
        # LOCATION head: scalar value mu(s). Regressed by MSE -> fast mean tracking.
        self.value_head = layer_init(nn.Linear(H, 1), std=1.0)
        # SHAPE head: categorical residual R = G^lambda - mu over a tight relative
        # support c. PEAKED init (sharp at 0) so the initial residual is concentrated.
        self.shape_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            c = torch.linspace(-args.rel_range, args.rel_range, args.num_bins)
            self.shape_head.bias.copy_(-0.5 * (c / args.critic_init_tau) ** 2)
        self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        # Returns (mu scalar value (B,), residual-shape logits (B, num_bins)).
        f = self.critic_trunk(x)
        return self.value_head(f).squeeze(-1), self.shape_head(f)

    def get_action_and_value(self, x, z=None):
        mean = self.actor_head(self.actor_trunk(x))
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if z is None:
            z = probs.sample()
        action = torch.tanh(z)
        log_det = 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = (probs.log_prob(z) - log_det).sum(1)
        f = self.critic_trunk(x)
        value = self.value_head(f).squeeze(-1)
        shape_logits = self.shape_head(f)
        return action, z, log_prob, probs.entropy().sum(1), value, shape_logits


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are clamped to [v_min, v_max]).
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


def distributional_residual_returns(
    deltas, dones, next_done, shape_probs, bootstrap_shape, c, rel_range, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return RESIDUAL R = G^λ - mu,
    in the location-relative frame (probs over the relative support c):

        R_t =_D delta_t + γ·nonterm·[ (1-λ)·shape(s_{t+1}) + λ·R_{t+1} ]

    where delta_t = r~_t + γ·nonterm·mu(s_{t+1}) - mu(s_t) is the (SOFT, in v13)
    location TD-error == GAE delta on the soft reward. The residual mean tracks the
    advantage and vanishes at convergence; this is an AUXILIARY distributional
    target and does not feed control. Shapes: deltas/dones (T, B); shape_probs
    (T, B, n); bootstrap_shape (B, n) = residual shape of Z(s_T). Returns (T, B, n).
    """
    T = deltas.shape[0]
    target = torch.zeros_like(shape_probs)
    g_next = bootstrap_shape                            # R_T ≡ residual of bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_shape                     # shape(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = shape_probs[t + 1]                  # shape(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = deltas[t].unsqueeze(-1) + gn * c        # (B, n) relative atoms
        g_next = categorical_project(mix, atoms, c, -rel_range, rel_range, bin_width)
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

    # Maximum-entropy temperature (SAC dual). Optimise in log space to keep alpha>0.
    act_dim = int(np.prod(envs.single_action_space.shape))
    target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
    log_alpha = torch.tensor(float(np.log(args.alpha)), device=device, requires_grad=True)
    a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # Tight relative support for the residual shape (relocates with mu).
    c = torch.linspace(-args.rel_range, args.rel_range, args.num_bins, device=device)
    bin_width = (c[1] - c[0]).item()

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)          # mu(s)
    shape_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

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

        alpha = log_alpha.exp().item() if args.autotune_alpha else args.alpha

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, _, value, shape_logits = agent.get_action_and_value(next_obs)
                shape_probs[step] = torch.softmax(shape_logits, dim=-1)
                values[step] = value
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
            # SOFT reward: fold the (tanh-corrected) entropy bonus into the reward
            # the value/Bellman sees. r~ = r - alpha*logpi. Max-ent via the value.
            soft_rewards = rewards - alpha * logprobs
            next_value, next_shape_logits = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
            bootstrap_shape = torch.softmax(next_shape_logits, dim=-1)           # (B, n) residual of Z(s_T)
            # Scalar GAE on mu (the fast location) over the SOFT reward. Drives the
            # soft advantage AND yields the per-step soft TD-errors for the recursion.
            advantages = torch.zeros_like(rewards).to(device)
            deltas = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = soft_rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                deltas[t] = delta
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            # Distributional λ-return RESIDUAL target (soft, mean-matches advantages).
            shape_target = distributional_residual_returns(
                deltas, dones, next_done, shape_probs, bootstrap_shape,
                c, args.rel_range, bin_width, args.gamma, args.gae_lambda,
            )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_shape_target = shape_target.reshape(-1, args.num_bins)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value, shape_logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # LOCATION loss: direct scalar MSE on mu -> fast, lag-free soft value.
                v_loss_mu = 0.5 * ((value - b_returns[mb_inds]) ** 2).mean()
                # SHAPE loss: CE to the distributional residual target (relative).
                shape_log_probs = torch.log_softmax(shape_logits, dim=-1)
                v_loss_shape = -(b_shape_target[mb_inds] * shape_log_probs).sum(dim=-1).mean()
                v_loss = v_loss_mu + args.shape_coef * v_loss_shape

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # SAC temperature dual, stepped EVERY minibatch on the CURRENT
                # policy's entropy (-newlogprob). Frequent steps make alpha actually
                # track: it decays to ~0 while entropy > target (no interference),
                # then rises to hold entropy at the floor target_entropy.
                if args.autotune_alpha:
                    alpha_loss = -(log_alpha * (newlogprob.detach() + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        edge_mass = (b_shape_target[:, 0] + b_shape_target[:, -1]).mean().item()
        policy_entropy = (-b_logprobs).mean().item()   # tanh-corrected entropy estimate

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/value_loss_mu", v_loss_mu.item(), global_step)
        writer.add_scalar("losses/value_loss_shape", v_loss_shape.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("maxent/alpha", log_alpha.exp().item(), global_step)
        writer.add_scalar("maxent/policy_entropy", policy_entropy, global_step)
        writer.add_scalar("maxent/target_entropy", target_entropy, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/advantages_absmax", b_advantages.abs().max().item(), global_step)
        writer.add_scalar("debug/shape_edge_mass", edge_mass, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
