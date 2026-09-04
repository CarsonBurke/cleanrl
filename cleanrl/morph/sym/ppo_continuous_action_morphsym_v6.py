# PPO + Morphogenic Symmetry v6.
#
# METHOD. Identical architecture to v5 (gated ReGLU^2 positional graph with learned
# token sizes), but with ALL effective-usage "compute" costs DISABLED: the template
# cost (eff_templates), the attention-support cost, and the token-size cost are set
# to zero, so `compute` is identically 0 and the loss reduces to plain PPO on this
# architecture. `compute_coef` is deliberately left UNCHANGED (still 0.05) -- we are
# not zeroing the coefficient, we are removing the priced terms it multiplies.
#
# WHY. Those costs are mispriced: the forwards are DENSE (all M templates, the full
# U x U attention matrix, and the full-D FFN are always computed; entmax/gates only
# sparsify the OUTPUT). So minimizing eff_templates / attn_support / eff_token_size
# saves ZERO real FLOPs -- they are implicit MDL/sparsity regularizers wearing a
# "compute budget" label. Under them, v5's actor collapses to ~1.8 effective
# templates (near-total weight-tying) whether or not the task wants distinct weights,
# and the width gate sits inert at ~60/64. This is a tax on capacity that buys
# nothing.
#
# HYPOTHESIS. Removing the fake tax lets the network use the capacity it actually
# wants: eff_templates should rise well above ~1.8 and, if the tax was suppressing
# useful weight diversity, benchmark return should improve over v5 (tax on). The
# diagnostics (eff_templates, attn_support, eff_token_size) are still MEASURED and
# logged -- just not priced -- so we can watch what the untaxed network chooses.
#
# NOTE (design debt, tracked in forward): tick_frac and per-tick gates currently only
# SCALE cost; the tick loop always runs all T ticks, so they never change real FLOPs.
# For compute pricing to ever be honest, ticks (and templates/channels) must become
# genuinely sparse -- see the TODO in SymGraph6.forward.
import os
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
    effective_support,
    make_env,
    mean_stat,
)
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import signed_loss_with_safe_compute
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v3 import participation_ratio
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v5 import (
    Args as V5Args,
    Agent as V5Agent,
    SymGraph5,
)


@dataclass
class Args(V5Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    # v6: disable every effective-usage compute cost (compute becomes identically 0).
    # compute_coef is intentionally left at its inherited default (0.05) -- we remove
    # the priced terms, not the coefficient.
    template_compute_weight: float = 0.0
    attn_compute_weight: float = 0.0
    size_compute_weight: float = 0.0


class SymGraph6(SymGraph5):
    """v5 architecture with the compute tax removed: `compute` is identically 0.
    All usage diagnostics are still computed and logged, just not priced."""

    def forward(self, x):
        B = x.shape[0]
        phi = self.phi()
        g = torch.sigmoid(self.size_logits(phi) + self.size_init_bias)  # (U, D) per-unit width gate
        h = self.input(x)[:, None, :] + self.pos_embed(phi)[None, :, :]
        tick_gates = torch.sigmoid(self.tick_bias)
        # TODO(compute honesty): tick_gates / tick_frac only SCALE the (now-disabled)
        # compute cost -- the loop below always runs all self.T ticks, so a tick gate
        # near 0 still costs full FLOPs. At some point tick_frac should actually gate
        # WHETHER a tick is computed (adaptive computation time / early-exit), so that
        # fewer active ticks means genuinely less compute. Same applies to templates
        # (dense over M) and channels (dense over D): real pricing needs sparse compute.

        attn_support_sum = x.new_zeros(B)
        last_A = None
        for t in range(self.T):
            mixed, support = self.mixer(h, phi)
            h = h + tick_gates[t] * (g[None] * (mixed - h))
            updated, A = self.field(h, phi)
            h = h + tick_gates[t] * (g[None] * (updated - h))
            attn_support_sum = attn_support_sum + tick_gates[t] * support.mean(dim=(1, 2))
            last_A = A

        out_units = h[:, self.N :, :]  # (B, n_out, D)

        # Diagnostics only -- measured, not priced (compute tax disabled in v6).
        template_usage = last_A.sum(dim=0)
        eff_templates = participation_ratio(template_usage)
        eff_token_size = g.sum(dim=-1).mean()
        tick_frac = tick_gates.mean()
        assign_support = effective_support(last_A, dim=-1).mean()
        compute = x.new_zeros(B)  # v6: no compute cost -> loss reduces to plain PPO

        stats = {
            "compute": compute,
            "eff_templates": eff_templates.expand(B),
            "assign_support": assign_support.expand(B),
            "attn_support": (attn_support_sum / max(self.T, 1)).clamp(min=0.0),
            "eff_token_size": eff_token_size.expand(B),
            "tick_frac": tick_frac.expand(B),
        }
        return out_units, stats


class Agent(V5Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymGraph6(obs_dim, args, n_out=action_dim)
        self.critic = SymGraph6(obs_dim, args, n_out=1)


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
                writer.add_scalar(f"morph/{group}_eff_token_size", mean_stat(last_stats, group, "eff_token_size"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
