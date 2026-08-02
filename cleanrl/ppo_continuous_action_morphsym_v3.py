# PPO + Morphogenic Symmetry v3.
#
# METHOD. One learned positional field governs BOTH weight-sharing and routing.
# v1 showed static weight-tying + content routing (a transformer's discipline)
# learns well; v2 showed content-DEPENDENT weights (per-input weight morphing) is
# unstable and regresses. v3 keeps weights static and pushes all dynamics into
# routing, then unifies the two learned mechanisms through a shared learned
# positional geometry:
#   - LEARNED POSITIONS. Every unit has a learned position p_u lifted by a
#     learnable Fourier map phi(p) = [p, sin(p*Omega), cos(p*Omega)] (Omega
#     learnable) -- a PoPE-flavored basis that can express smooth (equivariant)
#     or sharp (specialized) positional structure.
#   - POSITIONAL WEIGHT-SHARING. A unit's FFN weights are a static entmax mixture
#     of shared templates keyed by phi(p_u): units near each other in position
#     feature space share weights, so the symmetry orbit is a region of the
#     learned position manifold. Static weights => stable optimization.
#   - POSITIONAL ROUTING. Multi-head attention over units with a disentangled
#     positional bias from the same phi(p) (content term + position term). Each
#     head's routing support is charged to compute, so active-head count emerges.
#   - NO-POOLING READOUT. Outputs are UNITS, not a pool. The actor graph holds one
#     output unit per action dim; the critic graph one value unit. They co-evolve
#     with hidden units through all ticks and gather their own rich D-vector via
#     the same positional attention (Perceiver-IO / DETR object-query style). A
#     weight-shared head maps each action unit to Beta (alpha, beta); the value
#     unit to a scalar. The gated residual lives in the tick dynamics.
#
# Distinct-template count, per-head attention support -- the weight-sharing and
# routing complexity -- are compute-priced, so the field discovers the minimal
# symmetry group. Everything foundational is emergent and learned.
#
# HYPOTHESIS. Grounding both weight-sharing and routing in one learned positional
# field, with static weights and a bottleneck-free readout, is the stable emergent
# realization of learned spatial symmetry and should exceed v1.
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

from cleanrl.ppo_continuous_action_morphcompute_v9 import (
    Args as V9Args,
    ReLUSquared,
    SAMPLE_EPS,
    effective_support,
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.ppo_continuous_action_morphcompute_v18 import entmax15, signed_loss_with_safe_compute


@dataclass
class Args(V9Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    num_units: int = 48
    """N hidden units per graph (output units are added on top)"""
    hidden_dim: int = 128
    """H: template MLP hidden width"""
    num_templates: int = 8
    """M: shared weight-template pool size (upper bound on distinct unit functions)"""
    pos_dim: int = 8
    """learned raw position width per unit"""
    num_freq: int = 12
    """learnable Fourier frequencies; phi width = pos_dim + 2*num_freq"""
    freq_scale: float = 1.0
    """init scale of learnable Fourier frequencies Omega"""
    num_heads: int = 4
    """attention heads in the positional unit mixer"""
    attn_dim: int = 32
    """total query/key width (split across heads); must be divisible by num_heads"""
    sym_ticks: int = 3
    """T: shared-weight recurrent ticks (temporal symmetry)"""
    init_tick_bias: float = 1.0
    """initial per-tick update gate logit"""
    init_null_bias: float = 0.0
    """initial null-route bias per head (units may attend to nothing)"""
    template_compute_weight: float = 1.0
    """weight on effective-distinct-template compute cost"""
    attn_compute_weight: float = 1.0
    """weight on attention-support (routing) compute cost"""


def participation_ratio(usage, eps=1e-6):
    return usage.sum().pow(2) / usage.pow(2).sum().clamp_min(eps)


class TiedFFN(nn.Module):
    """Per-unit MLP whose weights are a static entmax mixture over M shared templates,
    keyed by the unit's positional features phi(p). Units sharing a template share weights
    and pool gradients; the effective number of distinct templates is emergent."""

    def __init__(self, D, H, M, phi_dim):
        super().__init__()
        self.D, self.H, self.M = D, H, M
        self.assign = layer_init(nn.Linear(phi_dim, M), std=0.5)
        self.W1 = nn.Parameter(torch.empty(M, D, H))
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.W2 = nn.Parameter(torch.empty(M, H, D))
        self.b2 = nn.Parameter(torch.zeros(M, D))
        for m in range(M):
            nn.init.orthogonal_(self.W1[m], gain=np.sqrt(2.0))
            nn.init.orthogonal_(self.W2[m], gain=0.5)
        self.norm = nn.LayerNorm(D)

    def assignment(self, phi):
        return entmax15(self.assign(phi), dim=-1)  # (U, M)

    def forward(self, h, phi):
        A = self.assignment(phi)
        x = self.norm(h)
        hidden = ReLUSquared()(torch.einsum("bud,mdh->bumh", x, self.W1) + self.b1)
        out = torch.einsum("bumh,mhd->bumd", hidden, self.W2) + self.b2
        delta = torch.einsum("um,bumd->bud", A, out)
        return h + delta, A


class PositionalMultiHeadMixer(nn.Module):
    """Multi-head entmax attention over units with a disentangled positional bias from phi(p).
    Content routing (activations) + positional routing, all with static shared weights."""

    def __init__(self, U, D, phi_dim, attn_dim, num_heads, init_null_bias):
        super().__init__()
        assert attn_dim % num_heads == 0 and D % num_heads == 0
        self.U, self.heads = U, num_heads
        self.dk = attn_dim // num_heads
        self.dv = D // num_heads
        self.q = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.k = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.v = layer_init(nn.Linear(D, D), std=0.5)
        self.pos_q = layer_init(nn.Linear(phi_dim, attn_dim), std=0.5)
        self.pos_k = layer_init(nn.Linear(phi_dim, attn_dim), std=0.5)
        self.out = layer_init(nn.Linear(D, D), std=0.5)
        self.null_bias = nn.Parameter(torch.full((num_heads, U, 1), float(init_null_bias)))
        self.scale = np.sqrt(self.dk)
        self.norm = nn.LayerNorm(D)

    def forward(self, h, phi):
        B, U = h.shape[0], h.shape[1]
        hn = self.norm(h)
        q = self.q(hn).view(B, U, self.heads, self.dk).permute(0, 2, 1, 3)  # (B,Hd,U,dk)
        k = self.k(hn).view(B, U, self.heads, self.dk).permute(0, 2, 1, 3)
        v = self.v(hn).view(B, U, self.heads, self.dv).permute(0, 2, 1, 3)  # (B,Hd,U,dv)
        content = torch.einsum("bhid,bhjd->bhij", q, k)
        pq = self.pos_q(phi).view(U, self.heads, self.dk)
        pk = self.pos_k(phi).view(U, self.heads, self.dk)
        pos_bias = torch.einsum("ihd,jhd->hij", pq, pk)
        logits = (content + pos_bias[None]) / self.scale  # (B,Hd,U,U)
        null = self.null_bias[None].expand(B, -1, -1, -1)  # (B,Hd,U,1)
        route_full = entmax15(torch.cat([logits, null], dim=-1), dim=-1)
        route = route_full[..., :U]  # (B,Hd,U,U)
        msg = torch.einsum("bhij,bhjd->bhid", route, v)  # (B,Hd,U,dv)
        msg = msg.permute(0, 2, 1, 3).reshape(B, U, self.heads * self.dv)
        support = effective_support(route, dim=-1)  # (B,Hd,U)
        return h + self.out(msg), support


class SymGraph(nn.Module):
    """A graph of N hidden units + n_out output units, all with learned positions, evolved by
    shared-weight positional attention + tied FFN over T ticks. Output units' states are read
    directly (no pooling)."""

    def __init__(self, obs_dim, args, n_out):
        super().__init__()
        self.N = args.num_units
        self.n_out = n_out
        self.U = self.N + n_out
        self.D = args.cell_dim
        self.T = args.sym_ticks
        self.M = args.num_templates
        self.template_compute_weight = args.template_compute_weight
        self.attn_compute_weight = args.attn_compute_weight
        self.phi_dim = args.pos_dim + 2 * args.num_freq

        self.pos = nn.Parameter(torch.randn(self.U, args.pos_dim))
        self.omega = nn.Parameter(torch.randn(args.pos_dim, args.num_freq) * args.freq_scale)
        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.pos_embed = layer_init(nn.Linear(self.phi_dim, self.D), std=0.5)
        self.mixer = PositionalMultiHeadMixer(
            self.U, self.D, self.phi_dim, args.attn_dim, args.num_heads, args.init_null_bias
        )
        self.field = TiedFFN(self.D, args.hidden_dim, self.M, self.phi_dim)
        self.tick_bias = nn.Parameter(torch.full((self.T,), float(args.init_tick_bias)))

    def phi(self):
        proj = self.pos @ self.omega  # (U, num_freq)
        return torch.cat([self.pos, torch.sin(proj), torch.cos(proj)], dim=-1)  # (U, phi_dim)

    def forward(self, x):
        B = x.shape[0]
        phi = self.phi()
        h = self.input(x)[:, None, :] + self.pos_embed(phi)[None, :, :]  # (B,U,D)
        tick_gates = torch.sigmoid(self.tick_bias)

        attn_support_sum = x.new_zeros(B)
        last_A = None
        for t in range(self.T):
            mixed, support = self.mixer(h, phi)
            h = h + tick_gates[t] * (mixed - h)
            updated, A = self.field(h, phi)
            h = h + tick_gates[t] * (updated - h)
            attn_support_sum = attn_support_sum + tick_gates[t] * support.mean(dim=(1, 2))
            last_A = A

        out_units = h[:, self.N :, :]  # (B, n_out, D) -- read directly, no pooling

        template_usage = last_A.sum(dim=0)  # (M,)
        eff_templates = participation_ratio(template_usage)
        tick_frac = tick_gates.mean()
        template_cost = (eff_templates / self.M) * tick_frac
        attn_cost = (attn_support_sum / max(self.U, 1)) / max(self.T, 1)
        wsum = self.template_compute_weight + self.attn_compute_weight
        compute = (
            self.template_compute_weight * template_cost.expand(B) + self.attn_compute_weight * attn_cost
        ) / wsum
        compute = compute.clamp(min=0.0)

        assign_support = effective_support(last_A, dim=-1).mean()  # avg templates per unit
        stats = {
            "compute": compute,
            "eff_templates": eff_templates.expand(B),
            "assign_support": assign_support.expand(B),
            "attn_support": (attn_support_sum / max(self.T, 1)).clamp(min=0.0),
            "tick_frac": tick_frac.expand(B),
        }
        return out_units, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.actor = SymGraph(obs_dim, args, n_out=action_dim)
        self.critic = SymGraph(obs_dim, args, n_out=1)
        # weight-shared action head: each action unit's D-vector -> (alpha_logit, beta_logit)
        self.action_head = layer_init(nn.Linear(args.cell_dim, 2), std=0.01)
        self.value_head = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def get_value(self, x):
        value_units, _ = self.critic(x)  # (B, 1, D)
        return self.value_head(value_units.squeeze(1)).squeeze(-1)

    def _dist(self, action_units):
        params = self.action_head(action_units)  # (B, action_dim, 2)
        alpha = 1.0 + F.softplus(params[..., 0])
        beta = 1.0 + F.softplus(params[..., 1])
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action

    def get_action_and_value(self, x, z=None):
        action_units, actor_stats = self.actor(x)  # (B, action_dim, D)
        value_units, critic_stats = self.critic(x)  # (B, 1, D)
        dist, to_action = self._dist(action_units)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.value_head(value_units.squeeze(1)).squeeze(-1)
        stats = {"actor": dict(actor_stats), "critic": dict(critic_stats)}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats


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
    last_stats = None

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, z, logprob, _, value, _, _, last_stats = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
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

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, _, newlogprob, entropy, newvalue, actor_compute, critic_compute, last_stats = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
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

                actor_multiplier = agent.compute_multiplier(actor_compute, args)
                critic_multiplier = agent.compute_multiplier(critic_compute, args)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = signed_loss_with_safe_compute(pg_loss_per_sample, actor_multiplier, args.actor_compute_loss_floor)

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * critic_multiplier).mean()
                else:
                    v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) ** 2) * critic_multiplier).mean()

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

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if last_stats is not None:
            for group in ("actor", "critic"):
                writer.add_scalar(f"morph/{group}_compute", mean_stat(last_stats, group, "compute"), global_step)
                writer.add_scalar(f"morph/{group}_eff_templates", mean_stat(last_stats, group, "eff_templates"), global_step)
                writer.add_scalar(f"morph/{group}_assign_support", mean_stat(last_stats, group, "assign_support"), global_step)
                writer.add_scalar(f"morph/{group}_attn_support", mean_stat(last_stats, group, "attn_support"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
