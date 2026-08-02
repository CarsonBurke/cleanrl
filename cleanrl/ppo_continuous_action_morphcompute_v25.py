# PPO + Morphogenic Compute v25.
#
# METHOD. Learned sensory receptive fields. In v18/v23 every cell receives the
# identical broadcast projection of the observation: input wiring is hard-coded,
# so cells can only differentiate downstream. v25 treats each observation
# dimension as a sensory token with a learned property vector; cells route from
# senses through entmax (with a null opt-out), combining a static learned wiring
# diagram (per-cell receptive field) with a state-dependent gain over senses.
# The routed sensory message is a real, paid, gated residual on cell-state
# initialization: sensory edges are charged to compute, the gate starts small so
# training begins at v23's behavior.
#
# HYPOTHESIS. Differentiated input wiring is the missing spatial symmetry break
# at the sensory boundary (biology: receptive fields are emergent and remap).
# Cells that see only task-relevant state subspaces should specialize earlier
# and route less redundant information, improving credit assignment.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.ppo_continuous_action_morphcompute_v9 import (
    ReLUSquared,
    effective_support,
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.ppo_continuous_action_morphcompute_v18 import (
    entmax_route_with_floor,
    signed_loss_with_safe_compute,
)
from cleanrl.ppo_continuous_action_morphcompute_v23 import (
    Args as V23Args,
    Agent as V23Agent,
    ConductanceSubstrate,
)


@dataclass
class Args(V23Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    sense_gate_bias: float = -2.0
    """initial logit for the sensory routing residual on cell-state init"""
    sense_gate_max: float = 1.0
    """maximum sensory routing residual scale"""
    sense_dense_floor: float = 0.05
    """dense floor in sensory routing so every sense keeps credit"""
    sense_null_bias: float = 0.0
    """initial null-route bias letting cells opt out of sensory input"""


class SensorySubstrate(ConductanceSubstrate):
    def __init__(self, obs_dim, args, num_readouts):
        super().__init__(obs_dim, args, num_readouts)
        self.obs_dim = int(obs_dim)
        P = args.property_dim
        self.obs_prop = nn.Parameter(torch.randn(self.obs_dim, P) * 0.5)
        self.obs_gain = nn.Parameter(torch.ones(self.obs_dim))
        self.obs_bias = nn.Parameter(torch.zeros(self.obs_dim))
        self.sense_q = layer_init(nn.Linear(P, P), std=0.5)
        self.sense_k = layer_init(nn.Linear(P, P), std=0.5)
        self.sense_state = layer_init(nn.Linear(self.D, P), std=0.2)
        self.sense_value = layer_init(nn.Linear(P + 1, self.D), std=0.5)
        self.sense_relation = nn.Sequential(
            layer_init(nn.Linear(4 * P, P), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(P, 1), std=0.05),
        )
        self.sense_null_bias = nn.Parameter(torch.full((self.K, 1), args.sense_null_bias))
        self.sense_gate_logit = nn.Parameter(torch.tensor(float(args.sense_gate_bias)))
        self.sense_gate_max = args.sense_gate_max
        self.sense_dense_floor = args.sense_dense_floor
        self.sense_scale = np.sqrt(P)

    def sense_gate(self):
        return self.sense_gate_max * torch.sigmoid(self.sense_gate_logit)

    def _sense_relation_bias(self):
        c = self.cell_prop[:, None, :]
        o = self.obs_prop[None, :, :]
        pair = torch.cat(
            [
                c.expand(-1, self.obs_dim, -1),
                o.expand(self.K, -1, -1),
                c - o,
                c * o,
            ],
            dim=-1,
        )
        return self.sense_relation(pair).squeeze(-1)

    def _sense(self, x, base):
        B = x.shape[0]
        x_enc = x * self.obs_gain[None, :] + self.obs_bias[None, :]
        values = self.sense_value(
            torch.cat([self.obs_prop[None, :, :].expand(B, -1, -1), x_enc[:, :, None]], dim=-1)
        )
        sense_keys = self.sense_k(self.obs_prop)
        wiring_logits = self.sense_q(self.cell_prop) @ sense_keys.T / self.sense_scale
        wiring_logits = wiring_logits + self._sense_relation_bias()
        state_logits = self.sense_state(base) @ sense_keys.T / self.sense_scale
        logits = wiring_logits[None, :, :] + state_logits[:, None, :]
        null_logits = self.sense_null_bias[None, :, :].expand(B, self.K, 1)
        route_with_null, _ = entmax_route_with_floor(
            torch.cat([logits, null_logits], dim=-1), self.sense_dense_floor, self.entmax_alpha, dim=-1
        )
        route = route_with_null[:, :, : self.obs_dim]
        msg = torch.bmm(route, values)
        sense_support = effective_support(route, dim=-1)
        return msg, sense_support

    def _core_forward(self, x):
        B = x.shape[0]
        base = self.input(x)
        query = self.query(base)
        dist2_query = (query[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)

        prop_law = self.prop_law(self.cell_prop)
        growth_pressure = self.growth(base) + self.prop_law_scale * prop_law[None, :, 0]
        shrink_pressure = F.softplus(self.shrink(base))
        active_logits = growth_pressure - shrink_pressure - 0.25 * dist2_query
        active = torch.sigmoid(active_logits)

        plasticity_logits = self.plasticity(base) + self.prop_law_scale * prop_law[None, :, 1]
        plasticity = torch.sigmoid(plasticity_logits)
        edge_source = self.edge_source(base)
        edge_target = self.edge_target(base)
        prop_active_weights = active / active.sum(dim=1, keepdim=True).clamp_min(1e-6)
        prop_persistence_bias = (prop_active_weights * prop_law[None, :, 2]).sum(dim=1, keepdim=True)
        prop_budget_bias = (prop_active_weights * prop_law[None, :, 3]).sum(dim=1, keepdim=True)
        persistence = torch.sigmoid(self.persistence(base) + self.prop_law_scale * prop_persistence_bias)
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(self.temperature(base))
        budget = torch.sigmoid(self.budget(base) + self.prop_law_scale * prop_budget_bias)
        tick_gates = torch.sigmoid(self.tick(base) + self.tick_offsets)

        sense_msg, sense_support = self._sense(x, base)
        sense_gate = self.sense_gate()
        sense_edges = (active * sense_support).sum(dim=1)
        sense_compute = sense_gate * sense_edges / max(self.K * self.obs_dim, 1)

        h = base[:, None, :] + self.cell_seed[None, :, :] + sense_gate * sense_msg
        coord_dist2 = (self.cell_pos[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)
        eye = torch.eye(self.K, device=x.device, dtype=x.dtype)
        nonself = 1.0 - eye
        geometry_logits = -coord_dist2[None, :, :] / temp[:, None, :]
        prop_route_bias = self.property_route_bias()
        edge_logits = (
            geometry_logits
            + edge_source[:, :, None]
            + edge_target[:, None, :]
            + self.edge_bias[None, :, :]
            + self.prop_route_scale * prop_route_bias[None, :, :]
        )
        route_logits = edge_logits + torch.log(active[:, None, :] + 1e-6)
        route_logits = route_logits.masked_fill(eye[None, :, :].bool(), -1e9)
        current_null_logits = self.null_route_bias[None, :, :].expand(B, self.K, 1)
        route_with_null, sparse_route_with_null = entmax_route_with_floor(
            torch.cat([route_logits, current_null_logits], dim=-1),
            self.route_dense_floor,
            self.entmax_alpha,
            dim=-1,
        )
        route = route_with_null[:, :, : self.K]
        sparse_route = sparse_route_with_null[:, :, : self.K]
        route_support = (sparse_route > 1e-6).to(route.dtype) * nonself[None, :, :]
        route_effective_support = effective_support(route * nonself[None, :, :], dim=-1)

        read_weights = active / (active.sum(dim=1, keepdim=True) + 1e-6)
        active_scale = active[:, :, None]
        plasticity_scale = plasticity[:, :, None]
        update_scale = budget[:, :, None] * active_scale * plasticity_scale
        history = []
        active_lookback_edges = x.new_zeros(B)
        expected_active_lookback_edges = x.new_zeros(B)
        current_route_support_sum = (active * route_support.sum(dim=-1)).sum(dim=1)
        current_effective_support_sum = (active * route_effective_support).sum(dim=1)
        current_compute_support_sum = (
            current_route_support_sum.detach()
            + current_effective_support_sum
            - current_effective_support_sum.detach()
        )
        lookback_route_entropy_sum = x.new_zeros(B)
        lookback_route_entropy_count = 0

        for t in range(self.T):
            source_values = self.norm(h) * active[:, :, None]
            msg = torch.bmm(route, source_values)
            lookback_msg = torch.zeros_like(h)
            if history:
                past = torch.stack(history, dim=1)
                history_count = past.shape[1]
                past_norm = self.norm(past)
                h_norm = self.norm(h)
                source_key = self.lookback_source(past_norm)
                target_key = self.lookback_target(h_norm)
                state_logits = torch.einsum("bif,bljf->blij", target_key, source_key) / np.sqrt(self.F)
                lookback_logits = (
                    geometry_logits[:, None, :, :]
                    + state_logits
                    + self.lookback_edge_bias[None, None, :, :]
                    + self.prop_route_scale * prop_route_bias[None, None, :, :]
                )

                source_activity = active[:, None, None, :]
                lookback_route_logits = lookback_logits + torch.log(source_activity + 1e-6)
                lookback_route_logits = lookback_route_logits.permute(0, 2, 1, 3).reshape(
                    B, self.K, history_count * self.K
                )
                lookback_null_logits = self.lookback_null_route_bias[None, :, :].expand(B, self.K, 1)
                lookback_route_with_null, sparse_lookback_route_with_null = entmax_route_with_floor(
                    torch.cat([lookback_route_logits, lookback_null_logits], dim=-1),
                    self.route_dense_floor,
                    self.entmax_alpha,
                    dim=-1,
                )
                lookback_route = lookback_route_with_null[:, :, : history_count * self.K]
                sparse_lookback_route = sparse_lookback_route_with_null[:, :, : history_count * self.K]
                past_values = past_norm * active[:, None, :, None]
                lookback_msg = torch.bmm(lookback_route, past_values.reshape(B, history_count * self.K, self.D))
                lookback_support = (sparse_lookback_route > 1e-6).to(lookback_route.dtype)
                lookback_effective_support = effective_support(lookback_route, dim=-1)

                tick_gate = tick_gates[:, t]
                lookback_support_sum = tick_gate * (active * lookback_support.sum(dim=-1)).sum(dim=1)
                lookback_effective_support_sum = tick_gate * (active * lookback_effective_support).sum(dim=1)
                active_lookback_edges = active_lookback_edges + lookback_support_sum
                expected_active_lookback_edges = expected_active_lookback_edges + (
                    lookback_support_sum.detach()
                    + lookback_effective_support_sum
                    - lookback_effective_support_sum.detach()
                )
                lookback_route_entropy = -(lookback_route * torch.log(lookback_route + 1e-8)).sum(dim=-1)
                lookback_route_entropy = (lookback_route_entropy * read_weights).sum(dim=1)
                lookback_route_entropy_sum = lookback_route_entropy_sum + tick_gate * lookback_route_entropy
                lookback_route_entropy_count += history_count * self.K

            mixed = h + persistence[:, :, None] * (self.route_mix * msg + self.lookback_mix * lookback_msg)
            delta = self.update(self.norm(mixed))
            h = h + tick_gates[:, t, None, None] * update_scale * delta
            history.append(h)

        pooled = torch.sum(read_weights[:, :, None] * h, dim=1)
        out = self.readout(pooled)

        active_cells = active.sum(dim=1)
        active_ticks = tick_gates.sum(dim=1)
        active_edges = current_route_support_sum
        active_edge_frac = active_edges / max(self.K * (self.K - 1), 1)
        expected_active_edges = current_compute_support_sum
        expected_active_edge_frac = expected_active_edges / max(self.K * (self.K - 1), 1)
        lookback_edge_capacity = max((self.T * (self.T - 1) // 2) * self.K * self.K, 1)
        active_lookback_edge_frac = active_lookback_edges / lookback_edge_capacity
        expected_active_lookback_edge_frac = expected_active_lookback_edges / lookback_edge_capacity
        route_entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1)
        route_entropy = (route_entropy * read_weights).sum(dim=1) / np.log(max(self.K - 1, 2))
        lookback_edge_entropy = lookback_route_entropy_sum / np.log(max(lookback_route_entropy_count, 2))
        lookback_route_entropy = lookback_route_entropy_sum / np.log(max(lookback_route_entropy_count, 2))
        active_cell_frac = active_cells / self.K
        active_tick_frac = active_ticks / self.T
        node_tick_compute = active_cell_frac * active_tick_frac
        edge_tick_compute = expected_active_edge_frac * active_tick_frac
        base_compute = (node_tick_compute + self.edge_compute_weight * edge_tick_compute) / (1.0 + self.edge_compute_weight)
        edge_read_capacity = max(self.T * self.K * (self.K - 1), 1)
        lookback_edge_compute = expected_active_lookback_edges / edge_read_capacity
        compute = (0.5 + 0.5 * budget.squeeze(1)) * (base_compute + self.lookback_compute_weight * lookback_edge_compute)
        compute = compute + sense_compute
        compute = compute.clamp(min=0.0)

        stats = {
            "compute": compute,
            "active_cells": active_cells,
            "active_ticks": active_ticks,
            "active_edges": active_edges,
            "active_edge_frac": active_edge_frac,
            "expected_active_edges": expected_active_edges,
            "expected_active_edge_frac": expected_active_edge_frac,
            "active_lookback_edges": active_lookback_edges,
            "active_lookback_edge_frac": active_lookback_edge_frac,
            "expected_active_lookback_edges": expected_active_lookback_edges,
            "expected_active_lookback_edge_frac": expected_active_lookback_edge_frac,
            "edge_entropy": route_entropy,
            "expected_edge_entropy": route_entropy,
            "edge_noise": x.new_empty(B, 0, self.K, self.K),
            "lookback_edge_entropy": lookback_edge_entropy,
            "expected_lookback_edge_entropy": lookback_edge_entropy,
            "lookback_edge_noise": x.new_empty(B, 0, self.K, self.K),
            "route_entropy": route_entropy,
            "lookback_route_entropy": lookback_route_entropy,
            "growth_pressure": growth_pressure.mean(dim=1),
            "shrink_pressure": shrink_pressure.mean(dim=1),
            "plasticity": plasticity.mean(dim=1),
            "persistence": persistence.squeeze(1),
            "budget": budget.squeeze(1),
            "temperature": temp.squeeze(1),
            "prop_law_std": prop_law.std().expand(B),
            "prop_route_bias_std": prop_route_bias.std().expand(B),
            "sense_gate": sense_gate.expand(B),
            "sense_support": sense_support.mean(dim=1),
            "sense_compute": sense_compute,
        }
        return out, h, read_weights, stats

    def forward(self, x):
        pooled, h, read_weights, stats = self._core_forward(x)
        token_features, token_entropy, token_support, token_compute = self._readout_tokens(h, read_weights)
        stats = dict(stats)
        stats["compute"] = stats["compute"] + token_compute
        stats["conductance_gate"] = (self.conductance_gate_max * torch.sigmoid(self.conductance_gate_logit)).expand(x.shape[0])
        stats["readout_entropy"] = token_entropy
        stats["readout_support"] = token_support
        stats["readout_compute"] = token_compute
        return pooled, token_features, h, read_weights, stats


class Agent(V23Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SensorySubstrate(obs_dim, args, action_dim)
        self.critic = SensorySubstrate(obs_dim, args, 1)


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
                writer.add_scalar(f"morph/{group}_sense_gate", mean_stat(last_stats, group, "sense_gate"), global_step)
                writer.add_scalar(f"morph/{group}_sense_support", mean_stat(last_stats, group, "sense_support"), global_step)
                writer.add_scalar(f"morph/{group}_sense_compute", mean_stat(last_stats, group, "sense_compute"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
