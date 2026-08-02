# PPO + Morphogenic Compute v18.
#
# METHOD. Property-coordinated v9. Keep v9's trainable morphogenic substrate and
# direct pooled actor/critic path, then add learned property coordinates as small
# residual coordinators for cell laws, routing, and action/value IO. This avoids
# v16/v17's brittle full replacement: PPO still has the v9 path, while properties
# can learn reusable transformer-like roles that coordinate weights and IO across
# cells, actions, value queries, growth/shrink, plasticity, and route biases.
#
# HYPOTHESIS. The right way to satisfy the "everything learnable" principle under
# PPO is not to force all control through an untrained property transformer from
# step zero. Instead, expose property-conditioned laws as residual degrees of
# freedom on top of a substrate that already receives useful task gradients.
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

ENTMAX_SQRT_EPS = 1e-8


def entmax15(logits, dim=-1):
    logits = logits / 2.0
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, dim=dim, descending=True).values
    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.size(dim)
    rhos = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(range_shape)
    mean = zs.cumsum(dim) / rhos
    mean_sq = zs.pow(2).cumsum(dim) / rhos
    ss = rhos * (mean_sq - mean.pow(2))
    delta = ((1.0 - ss) / rhos).clamp_min(ENTMAX_SQRT_EPS)
    taus = mean - torch.sqrt(delta)
    support = taus <= zs
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_star = taus.gather(dim, support_size.long() - 1)
    return torch.clamp(logits - tau_star, min=0.0).pow(2)


def entmax_route_with_floor(logits, dense_floor, alpha, dim=-1):
    if alpha != 1.5:
        raise ValueError("v18 only supports static entmax_alpha=1.5")
    sparse_route = entmax15(logits, dim=dim)
    if dense_floor <= 0.0:
        return sparse_route, sparse_route
    soft_route = torch.softmax(logits, dim=dim)
    route = (1.0 - dense_floor) * sparse_route + dense_floor * soft_route
    return route, sparse_route


@dataclass
class Args(V9Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    property_dim: int = 32
    """learned property coordinate width for cells, actions, and value queries"""
    prop_law_scale: float = 0.1
    """initial scale for property residuals added to growth/plasticity/budget/persistence laws"""
    prop_route_scale: float = 0.1
    """initial scale for property-pair residuals added to route logits"""
    prop_io_gate_bias: float = -2.0
    """initial gate for property readout residual logits/value; sigmoid(-2) keeps v9 path dominant"""
    prop_io_dense_floor: float = 0.05
    """softmax floor in property readout so every cell keeps IO credit"""
    prop_io_max_scale: float = 0.5
    """upper bound on learned property IO residual gates"""


class PropertyReadout(nn.Module):
    def __init__(self, cell_dim, property_dim, dense_floor):
        super().__init__()
        self.dense_floor = dense_floor
        self.query = layer_init(nn.Linear(property_dim, property_dim), std=0.5)
        self.key = layer_init(nn.Linear(property_dim, property_dim), std=0.5)
        self.state_key = layer_init(nn.Linear(cell_dim, property_dim), std=0.2)
        self.value = layer_init(nn.Linear(cell_dim, cell_dim), std=0.5)
        self.relation = nn.Sequential(
            layer_init(nn.Linear(4 * property_dim, property_dim), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(property_dim, 1), std=0.05),
        )
        self.out = nn.Sequential(layer_init(nn.Linear(cell_dim, cell_dim)), ReLUSquared())

    def relation_bias(self, query_prop, cell_prop):
        q = query_prop[:, None, :]
        c = cell_prop[None, :, :]
        pair = torch.cat(
            [
                q.expand(-1, cell_prop.shape[0], -1),
                c.expand(query_prop.shape[0], -1, -1),
                q - c,
                q * c,
            ],
            dim=-1,
        )
        return self.relation(pair).squeeze(-1)

    def forward(self, query_prop, cell_prop, h, read_weights):
        logits = torch.einsum(
            "qp,bkp->bqk",
            self.query(query_prop),
            self.key(cell_prop)[None, :, :] + self.state_key(h),
        ) / np.sqrt(query_prop.shape[-1])
        logits = logits + self.relation_bias(query_prop, cell_prop)[None, :, :]
        logits = logits + torch.log(read_weights[:, None, :].clamp_min(1e-8))
        sparse_route = entmax_route_with_floor(logits, 0.0, 1.5, dim=-1)[0]
        if self.dense_floor > 0.0:
            route = (1.0 - self.dense_floor) * sparse_route + self.dense_floor * torch.softmax(logits, dim=-1)
        else:
            route = sparse_route
        features = torch.bmm(route, self.value(h))
        entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1).mean(dim=1) / np.log(max(cell_prop.shape[0], 2))
        support = effective_support(route, dim=-1).mean(dim=1)
        return self.out(features), entropy, support


class PropertyMorphogenicSubstrate(nn.Module):
    """v9 substrate with residual property-conditioned cell laws and route bias."""

    def __init__(self, obs_dim, args):
        super().__init__()
        self.K = args.max_cells
        self.D = args.cell_dim
        self.F = args.field_dim
        self.T = args.max_ticks
        self.temp_min = args.field_temp_min
        self.temp_max = args.field_temp_max
        self.route_mix = args.route_mix
        self.edge_compute_weight = args.edge_compute_weight
        self.lookback_compute_weight = args.lookback_compute_weight
        self.lookback_mix = args.lookback_mix
        self.route_dense_floor = args.route_dense_floor
        self.entmax_alpha = args.entmax_alpha
        self.prop_law_scale = args.prop_law_scale
        self.prop_route_scale = args.prop_route_scale

        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.cell_seed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.cell_pos = nn.Parameter(torch.randn(self.K, self.F) * 0.5)
        self.cell_prop = nn.Parameter(torch.randn(self.K, args.property_dim) * 0.5)
        self.edge_bias = nn.Parameter(torch.full((self.K, self.K), args.init_edge_bias))
        self.lookback_edge_bias = nn.Parameter(torch.full((self.K, self.K), args.init_lookback_edge_bias))
        self.null_route_bias = nn.Parameter(torch.full((self.K, 1), args.init_null_route_bias))
        self.lookback_null_route_bias = nn.Parameter(torch.full((self.K, 1), args.init_lookback_null_route_bias))

        self.query = layer_init(nn.Linear(self.D, self.F), std=0.1)
        self.growth = layer_init(nn.Linear(self.D, self.K), std=0.01, bias_const=args.init_active_bias)
        self.shrink = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.plasticity = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.edge_source = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.edge_target = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.lookback_source = layer_init(nn.Linear(self.D, self.F), std=0.01)
        self.lookback_target = layer_init(nn.Linear(self.D, self.F), std=0.01)
        self.persistence = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.temperature = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.budget = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.tick = layer_init(nn.Linear(self.D, self.T), std=0.01)

        tick_offsets = torch.linspace(0.0, -1.5, self.T)
        self.register_buffer("tick_offsets", tick_offsets)
        self.tick.bias.data.fill_(args.init_tick_bias)

        self.prop_law = layer_init(nn.Linear(args.property_dim, 4), std=0.01)
        self.prop_route = nn.Sequential(
            layer_init(nn.Linear(4 * args.property_dim, args.property_dim), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(args.property_dim, 1), std=0.01),
        )

        self.norm = nn.LayerNorm(self.D)
        self.update = nn.Sequential(
            layer_init(nn.Linear(self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.5),
        )
        self.readout = nn.Sequential(layer_init(nn.Linear(self.D, self.D)), ReLUSquared())

    def property_route_bias(self):
        target = self.cell_prop[:, None, :]
        source = self.cell_prop[None, :, :]
        pair = torch.cat(
            [
                target.expand(-1, self.K, -1),
                source.expand(self.K, -1, -1),
                target - source,
                target * source,
            ],
            dim=-1,
        )
        return self.prop_route(pair).squeeze(-1)

    def forward(self, x):
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

        h = base[:, None, :] + self.cell_seed[None, :, :]
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
        }
        return out, h, read_weights, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor = PropertyMorphogenicSubstrate(obs_dim, args)
        self.critic = PropertyMorphogenicSubstrate(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        self.action_prop = nn.Parameter(torch.randn(action_dim, args.property_dim) * 0.5)
        self.value_prop = nn.Parameter(torch.randn(1, args.property_dim) * 0.5)
        self.actor_prop_read = PropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor)
        self.critic_prop_read = PropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor)
        self.prop_alpha = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.prop_beta = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.prop_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_io_gate = nn.Parameter(torch.tensor(args.prop_io_gate_bias))
        self.critic_io_gate = nn.Parameter(torch.tensor(args.prop_io_gate_bias))
        self.prop_io_max_scale = args.prop_io_max_scale
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def _gate(self, gate):
        return self.prop_io_max_scale * torch.sigmoid(gate)

    def get_value(self, x):
        critic_features, critic_cells, critic_weights, _ = self.critic(x)
        prop_features, _, _ = self.critic_prop_read(self.value_prop, self.critic.cell_prop, critic_cells, critic_weights)
        base_value = self.critic_value(critic_features).squeeze(-1)
        prop_value = self.prop_value(prop_features).squeeze(-1).squeeze(1)
        return base_value + self._gate(self.critic_io_gate) * prop_value

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def _actor_dist(self, actor_features, actor_cells, actor_weights):
        base_alpha_logits = self.actor_alpha(actor_features)
        base_beta_logits = self.actor_beta(actor_features)
        prop_features, read_entropy, read_support = self.actor_prop_read(
            self.action_prop, self.actor.cell_prop, actor_cells, actor_weights
        )
        gate = self._gate(self.actor_io_gate)
        alpha = 1.0 + F.softplus(base_alpha_logits + gate * self.prop_alpha(prop_features).squeeze(-1))
        beta = 1.0 + F.softplus(base_beta_logits + gate * self.prop_beta(prop_features).squeeze(-1))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action, read_entropy, read_support

    def get_action_and_value(self, x, z=None):
        actor_features, actor_cells, actor_weights, actor_stats = self.actor(x)
        critic_features, critic_cells, critic_weights, critic_stats = self.critic(x)
        dist, to_action, actor_read_entropy, actor_read_support = self._actor_dist(actor_features, actor_cells, actor_weights)
        critic_prop_features, critic_read_entropy, critic_read_support = self.critic_prop_read(
            self.value_prop, self.critic.cell_prop, critic_cells, critic_weights
        )
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic_value(critic_features).squeeze(-1) + self._gate(self.critic_io_gate) * self.prop_value(
            critic_prop_features
        ).squeeze(-1).squeeze(1)
        actor_stats = dict(actor_stats)
        critic_stats = dict(critic_stats)
        actor_stats["compute"] = actor_stats["compute"] + self._gate(self.actor_io_gate) * actor_read_support / max(
            self.actor.K, 1
        )
        critic_stats["compute"] = critic_stats["compute"] + self._gate(self.critic_io_gate) * critic_read_support / max(
            self.critic.K, 1
        )
        actor_stats["prop_read_entropy"] = actor_read_entropy
        actor_stats["prop_read_support"] = actor_read_support
        actor_stats["prop_io_gate"] = self._gate(self.actor_io_gate).expand_as(logprob)
        critic_stats["prop_read_entropy"] = critic_read_entropy
        critic_stats["prop_read_support"] = critic_read_support
        critic_stats["prop_io_gate"] = self._gate(self.critic_io_gate).expand_as(logprob)
        stats = {"actor": actor_stats, "critic": critic_stats}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats


def signed_loss_with_safe_compute(task_loss_sample, multiplier, floor):
    task = (task_loss_sample * multiplier.detach()).mean()
    compute = ((task_loss_sample.detach().abs() + floor) * multiplier).mean()
    return task + compute - compute.detach()


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
                writer.add_scalar(f"morph/{group}_route_entropy", mean_stat(last_stats, group, "route_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_entropy", mean_stat(last_stats, group, "prop_read_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_support", mean_stat(last_stats, group, "prop_read_support"), global_step)
                writer.add_scalar(f"morph/{group}_prop_io_gate", mean_stat(last_stats, group, "prop_io_gate"), global_step)
                writer.add_scalar(f"morph/{group}_prop_law_std", mean_stat(last_stats, group, "prop_law_std"), global_step)
                writer.add_scalar(f"morph/{group}_prop_route_bias_std", mean_stat(last_stats, group, "prop_route_bias_std"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
