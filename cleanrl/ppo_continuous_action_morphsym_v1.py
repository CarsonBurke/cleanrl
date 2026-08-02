# PPO + Morphogenic Symmetry v1.
#
# METHOD. Learned spatial weight-tying. The morphcompute lineage already has
# *temporal* symmetry: one update law is shared across recurrent ticks. It lacked
# the *spatial* analog, and both prior extremes were hard-coded: v18/v23 force one
# update MLP on every unit (mandatory full symmetry, like a shared conv kernel),
# while v24 gives every unit its own FiLM'd law (mandatory symmetry breaking).
# v1 makes the equivalence relation itself learnable. Units own NO weights. A pool
# of M shared weight TEMPLATES exists; each unit is routed (entmax, sparsifies to a
# hard partition) to templates by its learned coordinate. Units on the same
# template ARE the same function: they share weights and their gradients pool into
# that template -> consistent weights AND consistent weight updates. The EFFECTIVE
# number of distinct templates used is charged to compute, so the field discovers
# the MINIMAL symmetry group that solves the task.
#
# This subsumes the extremes: collapse to 1 template => full weight sharing (a
# CNN/transformer layer); spread across M => v24's per-unit distinctness. The
# router learns whichever orbit structure the task rewards (translation-like tying
# for images, permutation orbits for graphs). Connectivity is a learned, priced
# entmax mixer over units (the graph); temporal symmetry stays as shared-weight
# tick recurrence. Everything foundational is emergent and compute-bounded.
#
# HYPOTHESIS. Discovering how many distinct weight-sets a layer needs -- rather
# than fixing it at 1 (over-shared) or N (over-parameterized) -- gives weight
# sharing where the task is symmetric (amplified, low-variance gradients, better
# sample efficiency) and distinct circuits where it is not.
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
    """N: units per field. Distinct weight-sets are emergent via tying, not this count"""
    hidden_dim: int = 128
    """H: template MLP hidden width"""
    num_templates: int = 8
    """M: shared weight-template pool size (upper bound on distinct unit functions)"""
    coord_dim: int = 16
    """learned per-unit coordinate width; the substrate over which symmetry is defined"""
    sym_ticks: int = 3
    """T: shared-weight recurrent ticks (temporal symmetry)"""
    attn_dim: int = 32
    """unit-mixer query/key width"""
    init_tick_bias: float = 1.0
    """initial per-tick update gate logit"""
    init_null_bias: float = 0.0
    """initial null-route bias in the unit mixer (opt out of messages)"""
    template_compute_weight: float = 1.0
    """weight on effective-distinct-template compute cost"""
    mixer_compute_weight: float = 1.0
    """weight on unit-mixer connectivity compute cost"""


def participation_ratio(usage, eps=1e-6):
    """Effective number of nonzero entries: (sum)^2 / sum(sq). Ranges 1..len(usage)."""
    return usage.sum().pow(2) / usage.pow(2).sum().clamp_min(eps)


class TiedField(nn.Module):
    """A layer of N units whose per-unit update MLP is a learned entmax mixture over M shared
    weight templates. Units routed to the same template share weights and pool their gradients."""

    def __init__(self, N, D, H, M, coord_dim):
        super().__init__()
        self.N, self.D, self.H, self.M = N, D, H, M
        self.assign = layer_init(nn.Linear(coord_dim, M), std=0.5)
        self.W1 = nn.Parameter(torch.empty(M, D, H))
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.W2 = nn.Parameter(torch.empty(M, H, D))
        self.b2 = nn.Parameter(torch.zeros(M, D))
        for m in range(M):
            nn.init.orthogonal_(self.W1[m], gain=np.sqrt(2.0))
            nn.init.orthogonal_(self.W2[m], gain=0.5)
        self.norm = nn.LayerNorm(D)

    def assignment(self, coord):
        return entmax15(self.assign(coord), dim=-1)  # (N, M) sparse partition

    def forward(self, h, coord):
        A = self.assignment(coord)
        x = self.norm(h)
        hidden = ReLUSquared()(torch.einsum("bnd,mdh->bnmh", x, self.W1) + self.b1)
        out = torch.einsum("bnmh,mhd->bnmd", hidden, self.W2) + self.b2
        delta = torch.einsum("nm,bnmd->bnd", A, out)
        return h + delta, A


class UnitMixer(nn.Module):
    """Learned, entmax-sparse, compute-priced connectivity among units (the graph)."""

    def __init__(self, N, D, coord_dim, attn_dim, init_null_bias):
        super().__init__()
        self.N = N
        self.q = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.k = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.v = layer_init(nn.Linear(D, D), std=0.5)
        self.coord_src = layer_init(nn.Linear(coord_dim, attn_dim), std=0.5)
        self.coord_tgt = layer_init(nn.Linear(coord_dim, attn_dim), std=0.5)
        self.null_bias = nn.Parameter(torch.full((N, 1), float(init_null_bias)))
        self.scale = np.sqrt(attn_dim)
        self.norm = nn.LayerNorm(D)

    def forward(self, h, coord):
        B = h.shape[0]
        hn = self.norm(h)
        logits = torch.einsum("bnq,bmq->bnm", self.q(hn), self.k(hn)) / self.scale
        coord_bias = self.coord_tgt(coord) @ self.coord_src(coord).T / self.scale  # (N, N)
        logits = logits + coord_bias[None, :, :]
        eye = torch.eye(self.N, device=h.device, dtype=h.dtype)
        logits = logits.masked_fill(eye[None].bool(), -1e9)
        null = self.null_bias[None, :, :].expand(B, self.N, 1)
        route_full = entmax15(torch.cat([logits, null], dim=-1), dim=-1)
        route = route_full[:, :, : self.N]
        msg = torch.bmm(route, self.v(hn))
        support = effective_support(route, dim=-1)  # (B, N)
        return h + msg, support


class SymField(nn.Module):
    """Actor/critic body: a homogeneous unit field with learned spatial weight-tying and
    shared-weight temporal recurrence. No unit owns weights; identity is only its coordinate."""

    def __init__(self, obs_dim, args):
        super().__init__()
        self.N = args.num_units
        self.D = args.cell_dim
        self.T = args.sym_ticks
        self.M = args.num_templates
        self.template_compute_weight = args.template_compute_weight
        self.mixer_compute_weight = args.mixer_compute_weight

        self.coord = nn.Parameter(torch.randn(self.N, args.coord_dim))
        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.coord_embed = layer_init(nn.Linear(args.coord_dim, self.D), std=0.5)
        self.mixer = UnitMixer(self.N, self.D, args.coord_dim, args.attn_dim, args.init_null_bias)
        self.field = TiedField(self.N, self.D, args.hidden_dim, self.M, args.coord_dim)
        self.tick_bias = nn.Parameter(torch.full((self.T,), float(args.init_tick_bias)))
        self.read_query = layer_init(nn.Linear(args.coord_dim, 1), std=0.5)
        self.readout = nn.Sequential(layer_init(nn.Linear(self.D, self.D)), ReLUSquared())

    def forward(self, x):
        B = x.shape[0]
        h = self.input(x)[:, None, :] + self.coord_embed(self.coord)[None, :, :]
        tick_gates = torch.sigmoid(self.tick_bias)

        mixer_support_sum = x.new_zeros(B)
        last_A = None
        for t in range(self.T):
            mixed, support = self.mixer(h, self.coord)
            updated, A = self.field(mixed, self.coord)
            h = h + tick_gates[t] * (updated - h)
            mixer_support_sum = mixer_support_sum + tick_gates[t] * support.mean(dim=1)
            last_A = A

        read_w = torch.softmax(self.read_query(self.coord).squeeze(-1), dim=0)  # (N,)
        pooled = torch.einsum("n,bnd->bd", read_w, h)
        out = self.readout(pooled)

        template_usage = last_A.sum(dim=0)  # (M,)
        eff_templates = participation_ratio(template_usage)
        tick_frac = tick_gates.mean()
        template_cost = (eff_templates / self.M) * tick_frac
        mixer_cost = (mixer_support_sum / max(self.N, 1)) / max(self.T, 1)
        compute = (
            self.template_compute_weight * template_cost.expand(B)
            + self.mixer_compute_weight * mixer_cost
        ) / (self.template_compute_weight + self.mixer_compute_weight)
        compute = compute.clamp(min=0.0)

        assign_support = effective_support(last_A, dim=-1).mean()  # avg templates per unit
        stats = {
            "compute": compute,
            "eff_templates": eff_templates.expand(B),
            "assign_support": assign_support.expand(B),
            "mixer_support": (mixer_support_sum / max(self.T, 1)).clamp(min=0.0),
            "tick_frac": tick_frac.expand(B),
        }
        return out, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymField(obs_dim, args)
        self.critic = SymField(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def get_value(self, x):
        critic_features, _ = self.critic(x)
        return self.critic_value(critic_features).squeeze(-1)

    def _dist(self, actor_features):
        alpha = 1.0 + F.softplus(self.actor_alpha(actor_features))
        beta = 1.0 + F.softplus(self.actor_beta(actor_features))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action

    def get_action_and_value(self, x, z=None):
        actor_features, actor_stats = self.actor(x)
        critic_features, critic_stats = self.critic(x)
        dist, to_action = self._dist(actor_features)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic_value(critic_features).squeeze(-1)
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
                writer.add_scalar(f"morph/{group}_mixer_support", mean_stat(last_stats, group, "mixer_support"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
