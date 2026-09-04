# PPO + Morphogenic Compute v24.
#
# METHOD. Learned symmetry breaking. v23 applies one shared update MLP to every
# cell: spatial weight symmetry is hard-coded, like a transformer layer shared
# across tokens. v24 makes that symmetry itself learnable: each cell's property
# vector FiLM-modulates the shared update law's hidden layer through a bounded
# learned gate (init ~0 => exactly v23's shared law). Specialization is a real,
# paid forward path: modulation strength is charged to compute, so cells only
# differentiate when task gradients justify the price.
#
# HYPOTHESIS. Full weight sharing caps per-cell role specialization; letting the
# network pay to break symmetry recovers distinct-circuit benefits (biology: one
# canonical microcircuit, regionally differentiated) without giving up the
# shared-law inductive bias early in training.
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
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import signed_loss_with_safe_compute
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v23 import (
    Args as V23Args,
    Agent as V23Agent,
    ConductanceSubstrate,
)


@dataclass
class Args(V23Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    spec_gate_bias: float = -2.0
    """initial logit for per-cell update-law specialization; sigmoid(-2) starts near shared law"""
    spec_gate_max: float = 0.5
    """maximum specialization gate: bounds how far any cell's law departs from the shared one"""


class SymmetryBreakUpdate(nn.Module):
    """Shared update MLP whose hidden layer is FiLM-modulated per cell by its property vector."""

    def __init__(self, base_update, cell_prop, property_dim, hidden_dim, gate_bias, gate_max):
        super().__init__()
        self.w1 = base_update[0]
        self.act = base_update[1]
        self.w2 = base_update[2]
        self.cell_prop = cell_prop
        self.spec_scale = layer_init(nn.Linear(property_dim, hidden_dim), std=0.01)
        self.spec_shift = layer_init(nn.Linear(property_dim, hidden_dim), std=0.01)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias)))
        self.gate_max = gate_max

    def gate(self):
        return self.gate_max * torch.sigmoid(self.gate_logit)

    def modulation(self):
        return self.spec_scale(self.cell_prop), self.spec_shift(self.cell_prop)

    def forward(self, x):
        hidden = self.act(self.w1(x))
        g = self.gate()
        mscale, mshift = self.modulation()
        hidden = hidden * (1.0 + g * mscale[None, :, :]) + g * mshift[None, :, :]
        return self.w2(hidden)


class SymmetryBreakSubstrate(ConductanceSubstrate):
    def __init__(self, obs_dim, args, num_readouts):
        super().__init__(obs_dim, args, num_readouts)
        self.update = SymmetryBreakUpdate(
            self.update,
            self.cell_prop,
            args.property_dim,
            2 * self.D,
            args.spec_gate_bias,
            args.spec_gate_max,
        )

    def forward(self, x):
        pooled, token_features, h, read_weights, stats = super().forward(x)
        g = self.update.gate()
        mscale, mshift = self.update.modulation()
        usage = torch.tanh(mscale.abs().mean() + mshift.abs().mean())
        spec_compute = g * usage
        stats = dict(stats)
        stats["compute"] = stats["compute"] + spec_compute.expand(x.shape[0])
        stats["spec_gate"] = g.expand(x.shape[0])
        stats["spec_usage"] = usage.expand(x.shape[0])
        return pooled, token_features, h, read_weights, stats


class Agent(V23Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymmetryBreakSubstrate(obs_dim, args, action_dim)
        self.critic = SymmetryBreakSubstrate(obs_dim, args, 1)


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
                writer.add_scalar(f"morph/{group}_active_cells", mean_stat(last_stats, group, "active_cells"), global_step)
                writer.add_scalar(f"morph/{group}_active_ticks", mean_stat(last_stats, group, "active_ticks"), global_step)
                writer.add_scalar(f"morph/{group}_expected_active_edges", mean_stat(last_stats, group, "expected_active_edges"), global_step)
                writer.add_scalar(f"morph/{group}_conductance_gate", mean_stat(last_stats, group, "conductance_gate"), global_step)
                writer.add_scalar(f"morph/{group}_readout_support", mean_stat(last_stats, group, "readout_support"), global_step)
                writer.add_scalar(f"morph/{group}_readout_gate", mean_stat(last_stats, group, "readout_gate"), global_step)
                writer.add_scalar(f"morph/{group}_spec_gate", mean_stat(last_stats, group, "spec_gate"), global_step)
                writer.add_scalar(f"morph/{group}_spec_usage", mean_stat(last_stats, group, "spec_usage"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
