# PPO + Morphogenic Symmetry v11.
#
# METHOD. Makes per-template token width a REAL compute lever via PER-TEMPLATE SIZED
# MATMULS. Each template m has a learned integer width k_m in [1, D]; its ReGLU^2 FFN is
# an actual (., k_m) x (k_m, H) -> (., H) -> (., k_m) matmul over the first k_m channels,
# looped over the M templates. A narrow template is therefore a genuinely smaller GEMM
# (fewer FLOPs), unlike v9/v10 where a "narrow" token still ran the full dense einsum and
# was only gated afterward. Width is differentiable via a straight-through soft boundary:
# w_m = D*sigmoid(width_logit[m]) (continuous), k_m = ceil(w_m) (hard, used for slicing),
# and the boundary channel carries a fractional soft weight so gradients reach width_logit.
#
# HONEST-COMPUTE STATUS. This finally makes the width SAVING real (narrow => smaller matmul).
# But two honest caveats (see chat): (1) we are currently ENV-STEPPING bound, so fewer GPU
# FLOPs do not speed wall-clock yet; (2) making the saving real does NOT by itself make width
# move -- the priced-cost gradient is unchanged -- so v11 uses a STRENGTHENED width tax
# (width_compute_weight) to actually exercise the sizing. Validate by: does eff_token_size
# carve down, and does mean k_m (real FFN width) drop, at what score cost. Payoff scales with
# M and D (compute-bound regime, e.g. the 1000-template count-sparse axis) -- this is
# scaffolding for that, not an immediate MuJoCo score win.
#
# TODO(future): attention is still full-D (only FFN templates are sized); and the recursive
# "substrate inside a token" (perceptrons with their own routing + ticks) is a later axis.
import os
import math
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v9 import (
    ReLUSquared,
    effective_support,
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import (
    entmax15,
    signed_loss_with_safe_compute,
)
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v3 import participation_ratio
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v9 import Args as V9Args, Agent as V9Agent, SymGraph9


@dataclass
class Args(V9Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    width_init_bias: float = 3.0
    """per-template width logit init; sigmoid(3)~0.95 -> starts near full width (k~=D)"""
    width_compute_weight: float = 6.0
    """STRENGTHENED honest width tax (matmul is now sized, so shrinking saves real FLOPs)"""


class SizedTiedFFN(nn.Module):
    """ReGLU^2 tied FFN where each template m runs a real (.,k_m)x(k_m,H)->(.,k_m) matmul over
    its first k_m channels; k_m is a learned integer width (straight-through soft boundary)."""

    def __init__(self, D, H, M, phi_dim, width_init_bias):
        super().__init__()
        self.D, self.H, self.M = D, H, M
        self.assign = layer_init(nn.Linear(phi_dim, M), std=0.5)
        self.W1 = nn.Parameter(torch.empty(M, D, H))  # gate branch
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.W3 = nn.Parameter(torch.empty(M, D, H))  # value branch
        self.b3 = nn.Parameter(torch.zeros(M, H))
        self.W2 = nn.Parameter(torch.empty(M, H, D))  # output projection
        self.b2 = nn.Parameter(torch.zeros(M, D))
        for m in range(M):
            nn.init.orthogonal_(self.W1[m], gain=math.sqrt(2.0))
            nn.init.orthogonal_(self.W3[m], gain=1.0)
            nn.init.orthogonal_(self.W2[m], gain=0.5)
        self.width_logit = nn.Parameter(torch.full((M,), float(width_init_bias)))
        self.norm = nn.LayerNorm(D)
        self.act = ReLUSquared()

    def assignment(self, phi):
        return entmax15(self.assign(phi), dim=-1)  # (U, M)

    def forward(self, h, phi):
        B, U, _ = h.shape
        A = self.assignment(phi)  # (U, M)
        x = self.norm(h)
        widths = self.D * torch.sigmoid(self.width_logit)  # (M,) continuous width in [0, D]
        delta = h.new_zeros(B, U, self.D)
        ks = []
        for m in range(self.M):
            w = widths[m]
            k = int(max(1, min(self.D, math.ceil(w.item()))))
            ks.append(k)
            xm = x[:, :, :k]  # (B,U,k) sized input
            gate = self.act(torch.einsum("buk,kh->buh", xm, self.W1[m, :k]) + self.b1[m])
            val = torch.einsum("buk,kh->buh", xm, self.W3[m, :k]) + self.b3[m]
            hidden = gate * val  # (B,U,H)
            outm = torch.einsum("buh,hk->buk", hidden, self.W2[m, :, :k]) + self.b2[m, :k]  # (B,U,k)
            # straight-through soft boundary: last channel carries fractional weight -> grad to width_logit
            ramp = torch.clamp(w - torch.arange(k, device=h.device), 0.0, 1.0)  # (k,) ~1 except boundary
            outm = outm * ramp
            contrib = A[:, m][None, :, None] * outm  # (B,U,k)
            delta[:, :, :k] = delta[:, :, :k] + contrib
        # per-unit effective (soft) width = mixture of assigned template widths
        unit_width = (A @ widths)  # (U,)
        avg_width = unit_width.mean()
        avg_k = torch.tensor(float(np.mean(ks)), device=h.device)  # real avg FFN width (FLOP proxy)
        return h + delta, A, avg_width, avg_k


class SymGraph11(SymGraph9):
    """v9 honest-tax graph, but the FFN is a SizedTiedFFN (real per-template sized matmuls) and
    token width lives in the FFN sizing (no separate delta gate). Width tax is strengthened and
    now backs a real FLOP saving."""

    def __init__(self, obs_dim, args, n_out):
        super().__init__(obs_dim, args, n_out)
        # replace the gated field with the sized field; drop the v9 per-template width gate
        self.field = SizedTiedFFN(self.D, args.hidden_dim, self.M, self.phi_dim, args.width_init_bias)
        self.width_compute_weight = args.width_compute_weight
        if hasattr(self, "template_width"):
            del self.template_width  # width now lives inside SizedTiedFFN

    def forward(self, x):
        B = x.shape[0]
        phi = self.phi()
        h = self.input(x)[:, None, :] + self.pos_embed(phi)[None, :, :]
        tick_gates = torch.sigmoid(self.tick_bias)

        attn_support_sum = x.new_zeros(B)
        last_A = None
        avg_width = None
        avg_k = None
        for t in range(self.T):
            mixed, support = self.mixer(h, phi)
            h = h + tick_gates[t] * (mixed - h)
            updated, A, avg_width, avg_k = self.field(h, phi)
            h = h + tick_gates[t] * (updated - h)
            attn_support_sum = attn_support_sum + tick_gates[t] * support.mean(dim=(1, 2))
            last_A = A

        out_units = h[:, self.N :, :]

        eff_templates = participation_ratio(last_A.sum(dim=0))  # diagnostic
        assign_support = effective_support(last_A, dim=-1).mean()
        eff_token_size = avg_width  # soft per-unit width (priced)
        tick_frac = tick_gates.mean()

        assign_cost = (assign_support / max(self.M, 1)) * tick_frac
        attn_cost = (attn_support_sum / max(self.U, 1)) / max(self.T, 1)
        size_cost = (eff_token_size / self.D) * tick_frac
        wsum = self.assign_compute_weight + self.attn_compute_weight + self.width_compute_weight
        compute = (
            self.assign_compute_weight * assign_cost.expand(B)
            + self.attn_compute_weight * attn_cost
            + self.width_compute_weight * size_cost.expand(B)
        ) / max(wsum, 1e-8)
        compute = compute.clamp(min=0.0)

        stats = {
            "compute": compute,
            "eff_templates": eff_templates.expand(B),
            "assign_support": assign_support.expand(B),
            "attn_support": (attn_support_sum / max(self.T, 1)).clamp(min=0.0),
            "eff_token_size": eff_token_size.expand(B),
            "avg_k": avg_k.expand(B),  # real avg FFN width (integer channels actually computed)
            "tick_frac": tick_frac.expand(B),
        }
        return out_units, stats


class Agent(V9Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymGraph11(obs_dim, args, n_out=action_dim)
        self.critic = SymGraph11(obs_dim, args, n_out=1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
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
                writer.add_scalar(f"morph/{group}_eff_token_size", mean_stat(last_stats, group, "eff_token_size"), global_step)
                writer.add_scalar(f"morph/{group}_avg_k", mean_stat(last_stats, group, "avg_k"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
