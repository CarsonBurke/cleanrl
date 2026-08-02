# PPO + Morphogenic Compute v16.
#
# METHOD. Property-symmetric morphogenic PPO. Observation dimensions, latent
# cells, action dimensions, value queries, and substrate time are all represented
# as property-bearing objects. Shared property-conditioned relation laws decide
# input binding, cell-cell routing, temporal lookback routing, readout binding,
# plastic writes, soft lifecycle viability, and compute use. There are no K-output
# cell-law heads, KxK learned edge tables, or action-dimension-specific output
# heads; object roles are learned coordinates in a common property manifold.
#
# V16. Replace v9's slot-indexed substrate with a transformer-like property
# substrate. Null routing is removed from entmax competition: real sources always
# keep a gradient-carrying sparse+dense support, while learned open gates decide
# how much a target reads. Compute cost charges active plastic writes and relation
# usage, and PPO losses use loss * (1 + compute_cost) semantics with a safe
# nonnegative actor compute-gradient proxy for signed policy losses.
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

SAMPLE_EPS = 1e-6
ENTMAX_SQRT_EPS = 1e-8


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Property-symmetric morphogenic substrate arguments
    max_cells: int = 32
    """maximum latent property objects available to each substrate"""
    cell_dim: int = 64
    """latent object state width"""
    property_dim: int = 32
    """learned property coordinate width shared by observations, cells, actions, value, and time"""
    max_ticks: int = 4
    """maximum recurrent compute ticks per forward pass"""
    compute_coef: float = 0.05
    """strength of differentiable compute multiplier on actor and critic losses"""
    min_compute_multiplier: float = 1.0
    """base multiplier added before compute pressure"""
    route_dense_floor: float = 0.05
    """softmax floor mixed into entmax routes so off-support relations keep recovery gradients"""
    init_existence_bias: float = 0.0
    """initial soft lifecycle viability logit for each latent cell"""
    init_active_bias: float = 0.0
    """initial state-conditioned active-use logit"""
    init_open_bias: float = 1.0
    """initial relation/readout open-gate bias"""
    init_tick_bias: float = 0.2
    """initial recurrent tick halting bias"""
    min_viability: float = 0.5
    """floor on learned lifecycle viability so property objects cannot die before receiving task credit"""
    min_active: float = 0.5
    """floor on state-conditioned cell usage; compute pressure still acts through plastic writes/routes"""
    min_plasticity: float = 0.25
    """floor on plastic write gates to preserve recurrent learning signal early"""
    min_budget: float = 0.5
    """floor on per-cell compute budget so the substrate cannot shut updates off entirely"""
    min_tick_gate: float = 0.35
    """floor on each recurrent tick gate so temporal compute cannot disappear early"""
    min_route_open: float = 0.25
    """floor on relation/readout open gates so property objects keep communication gradients"""
    actor_compute_loss_floor: float = 0.01
    """small nonnegative floor for actor morphogenesis pressure when PPO loss is near zero"""
    critic_compute_loss_floor: float = 0.01
    """small nonnegative floor for critic morphogenesis pressure when value loss is near zero"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


def route_with_floor(logits, mask, dense_floor, dim=-1):
    masked_logits = logits.masked_fill(~mask, -1e9)
    sparse_route = entmax15(masked_logits, dim=dim) * mask.to(logits.dtype)
    if dense_floor <= 0.0:
        return sparse_route, sparse_route
    soft_route = torch.softmax(masked_logits, dim=dim) * mask.to(logits.dtype)
    route = (1.0 - dense_floor) * sparse_route + dense_floor * soft_route
    return route, sparse_route


def effective_support(route, dim=-1):
    mass = route.sum(dim=dim)
    support = mass.pow(2) / route.pow(2).sum(dim=dim).clamp_min(1e-6)
    return mass * support


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class PropertySymmetricSubstrate(nn.Module):
    """A fixed-capacity set of learned property objects with shared relation laws."""

    def __init__(self, obs_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.K = args.max_cells
        self.D = args.cell_dim
        self.P = args.property_dim
        self.T = args.max_ticks
        self.route_dense_floor = args.route_dense_floor
        self.min_viability = args.min_viability
        self.min_active = args.min_active
        self.min_plasticity = args.min_plasticity
        self.min_budget = args.min_budget
        self.min_tick_gate = args.min_tick_gate
        self.min_route_open = args.min_route_open

        self.obs_prop = nn.Parameter(torch.randn(obs_dim, self.P) * 0.5)
        self.cell_prop = nn.Parameter(torch.randn(self.K, self.P) * 0.5)
        self.time_prop = nn.Parameter(torch.randn(self.T, self.P) * 0.2)
        self.cell_seed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.existence_logit = nn.Parameter(torch.full((self.K,), args.init_existence_bias))

        self.scalar_token = layer_init(nn.Linear(1, self.D))
        self.prop_state = layer_init(nn.Linear(self.P, self.D), std=0.5)
        self.prop_query = layer_init(nn.Linear(self.P, self.P), std=0.5)
        self.prop_key = layer_init(nn.Linear(self.P, self.P), std=0.5)
        self.state_query = layer_init(nn.Linear(self.D, self.P), std=0.2)
        self.state_key = layer_init(nn.Linear(self.D, self.P), std=0.2)
        self.pair_bias = nn.Sequential(
            layer_init(nn.Linear(4 * self.P, self.P), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(self.P, 1), std=0.05),
        )

        gate_dim = 2 * self.D
        self.active_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01, bias_const=args.init_active_bias)
        self.birth_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.death_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.plasticity_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.budget_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.persistence_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.open_gate = layer_init(nn.Linear(gate_dim, 1), std=0.01, bias_const=args.init_open_bias)
        self.tick_gate = layer_init(nn.Linear(2 * self.D, 1), std=0.01, bias_const=args.init_tick_bias)
        self.register_buffer("tick_offsets", torch.linspace(0.0, -1.0, self.T))

        self.norm = nn.LayerNorm(self.D)
        self.update = nn.Sequential(
            layer_init(nn.Linear(2 * self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.5),
        )

    def _pair_property_bias(self, target_prop, source_prop):
        target = target_prop[:, None, :]
        source = source_prop[None, :, :]
        pair = torch.cat(
            [
                target.expand(-1, source_prop.shape[0], -1),
                source.expand(target_prop.shape[0], -1, -1),
                target - source,
                target * source,
            ],
            dim=-1,
        )
        return self.pair_bias(pair).squeeze(-1)

    def _gate_input(self, h, prop):
        prop_state = self.prop_state(prop)[None, :, :].expand(h.shape[0], -1, -1)
        return torch.cat([self.norm(h), prop_state], dim=-1)

    def _relation_logits(self, target_h, target_prop, source_h, source_prop):
        target_q = self.prop_query(target_prop)[None, :, :] + self.state_query(self.norm(target_h))
        source_k = self.prop_key(source_prop)[None, :, :] + self.state_key(self.norm(source_h))
        content_logits = torch.einsum("bip,bjp->bij", target_q, source_k) / np.sqrt(self.P)
        return content_logits + self._pair_property_bias(target_prop, source_prop)[None, :, :]

    def _route(self, target_h, target_prop, source_h, source_prop, mask, readout=False):
        logits = self._relation_logits(target_h, target_prop, source_h, source_prop)
        gate_input = self._gate_input(target_h, target_prop)
        open_gate = self.open_gate(gate_input)
        route_open_raw = torch.sigmoid(open_gate).squeeze(-1)
        route_open = self.min_route_open + (1.0 - self.min_route_open) * route_open_raw
        route, sparse_route = route_with_floor(logits, mask[None, :, :].expand(logits.shape[0], -1, -1), self.route_dense_floor)
        route = route * route_open[:, :, None]
        sparse_route = sparse_route * route_open[:, :, None]
        return route, sparse_route, route_open

    def _cell_laws(self, h, prop):
        gate_input = self._gate_input(h, prop)
        birth = torch.sigmoid(self.birth_gate(gate_input)).squeeze(-1)
        death = torch.sigmoid(self.death_gate(gate_input)).squeeze(-1)
        viability_raw = torch.sigmoid(self.existence_logit[None, :] + birth - death)
        active_raw = torch.sigmoid(self.active_gate(gate_input)).squeeze(-1)
        plasticity_raw = torch.sigmoid(self.plasticity_gate(gate_input)).squeeze(-1)
        budget_raw = torch.sigmoid(self.budget_gate(gate_input)).squeeze(-1)
        viability = self.min_viability + (1.0 - self.min_viability) * viability_raw
        active = viability * (self.min_active + (1.0 - self.min_active) * active_raw)
        plasticity = self.min_plasticity + (1.0 - self.min_plasticity) * plasticity_raw
        budget = self.min_budget + (1.0 - self.min_budget) * budget_raw
        persistence = torch.sigmoid(self.persistence_gate(gate_input)).squeeze(-1)
        return viability, active, plasticity, budget, persistence, birth, death

    def _tick_gates(self, h):
        context = h.mean(dim=1)
        time_state = self.prop_state(self.time_prop)
        gates = []
        for t in range(self.T):
            tick_input = torch.cat([context, time_state[t][None, :].expand(h.shape[0], -1)], dim=-1)
            tick_raw = torch.sigmoid(self.tick_gate(tick_input).squeeze(-1) + self.tick_offsets[t])
            gates.append(self.min_tick_gate + (1.0 - self.min_tick_gate) * tick_raw)
        return torch.stack(gates, dim=1)

    def read(self, query_prop, h, cell_weight):
        B = h.shape[0]
        query_h = self.prop_state(query_prop)[None, :, :].expand(B, -1, -1)
        mask = torch.ones(query_prop.shape[0], self.K, device=h.device, dtype=torch.bool)
        route, sparse_route, _ = self._route(query_h, query_prop, h, self.cell_prop, mask, readout=True)
        source_values = self.norm(h) * cell_weight[:, :, None]
        read_features = self.norm(torch.bmm(route, source_values) + query_h)
        read_support = effective_support(route, dim=-1)
        read_compute = read_support.mean(dim=1) / max(self.K, 1)
        read_entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1).mean(dim=1) / np.log(max(self.K, 2))
        return read_features, read_compute, read_entropy, sparse_route

    def forward(self, x):
        B = x.shape[0]
        obs_tokens = self.scalar_token(x[..., None]) + self.prop_state(self.obs_prop)[None, :, :]
        cell_prop = self.cell_prop + self.time_prop[0][None, :]
        h = self.cell_seed[None, :, :] + self.prop_state(cell_prop)[None, :, :]

        input_mask = torch.ones(self.K, self.obs_dim, device=x.device, dtype=torch.bool)
        input_route, sparse_input_route, _ = self._route(h.expand(B, -1, -1), cell_prop, obs_tokens, self.obs_prop, input_mask)
        h = h + torch.bmm(input_route, obs_tokens)

        tick_gates = self._tick_gates(h)
        eye = torch.eye(self.K, device=x.device, dtype=torch.bool)
        nonself_mask = ~eye
        history = []
        history_active = []

        active_cells_accum = x.new_zeros(B)
        plasticity_accum = x.new_zeros(B)
        write_intensity_sum = x.new_zeros(B)
        route_support_sum = x.new_zeros(B)
        route_effective_sum = x.new_zeros(B)
        lookback_support_sum = x.new_zeros(B)
        lookback_effective_sum = x.new_zeros(B)
        route_entropy_sum = x.new_zeros(B)
        lookback_entropy_sum = x.new_zeros(B)
        lookback_entropy_count = 0
        last_viability = last_active = last_plasticity = last_budget = None
        last_birth = last_death = None

        for t in range(self.T):
            viability, active, plasticity, budget, persistence, birth, death = self._cell_laws(h, cell_prop)
            tick_gate = tick_gates[:, t]
            write_intensity = tick_gate[:, None] * active * plasticity * budget
            source_values = self.norm(h) * active[:, :, None]

            current_route, sparse_current_route, _ = self._route(h, cell_prop, h, cell_prop, nonself_mask)
            current_msg = torch.bmm(current_route, source_values)
            current_support = (sparse_current_route > 1e-6).to(h.dtype).sum(dim=-1)
            current_effective = effective_support(current_route, dim=-1)

            lookback_msg = torch.zeros_like(h)
            if history:
                past = torch.stack(history, dim=1)
                history_count = past.shape[1]
                source_h = past.reshape(B, history_count * self.K, self.D)
                source_props = []
                for age in range(1, history_count + 1):
                    source_props.append(self.cell_prop + self.time_prop[min(age, self.T - 1)][None, :])
                source_prop = torch.cat(source_props, dim=0)
                lookback_mask = torch.ones(self.K, history_count * self.K, device=x.device, dtype=torch.bool)
                lookback_route, sparse_lookback_route, _ = self._route(h, cell_prop, source_h, source_prop, lookback_mask)
                source_active = torch.stack(history_active, dim=1).reshape(B, history_count * self.K)
                lookback_values = self.norm(source_h) * source_active[:, :, None]
                lookback_msg = torch.bmm(lookback_route, lookback_values)
                lookback_support = (sparse_lookback_route > 1e-6).to(h.dtype).sum(dim=-1)
                lookback_effective = effective_support(lookback_route, dim=-1)
                lookback_support_sum = lookback_support_sum + (write_intensity * lookback_support).sum(dim=1)
                lookback_effective_sum = lookback_effective_sum + (write_intensity * lookback_effective).sum(dim=1)
                lookback_entropy = -(lookback_route * torch.log(lookback_route + 1e-8)).sum(dim=-1)
                lookback_entropy_sum = lookback_entropy_sum + (lookback_entropy * write_intensity).sum(dim=1)
                lookback_entropy_count += history_count * self.K

            mixed = h + persistence[:, :, None] * (current_msg + lookback_msg)
            prop_state = self.prop_state(cell_prop)[None, :, :].expand(B, -1, -1)
            delta = self.update(torch.cat([self.norm(mixed), prop_state], dim=-1))
            h = h + write_intensity[:, :, None] * delta
            history.append(h)

            active_cells_accum = active_cells_accum + tick_gate * active.sum(dim=1)
            plasticity_accum = plasticity_accum + tick_gate * plasticity.mean(dim=1)
            write_intensity_sum = write_intensity_sum + write_intensity.sum(dim=1)
            route_support_sum = route_support_sum + (write_intensity * current_support).sum(dim=1)
            route_effective_sum = route_effective_sum + (write_intensity * current_effective).sum(dim=1)
            route_entropy = -(current_route * torch.log(current_route + 1e-8)).sum(dim=-1)
            route_entropy_sum = route_entropy_sum + (route_entropy * write_intensity).sum(dim=1)
            last_viability, last_active, last_plasticity, last_budget = viability, active, plasticity, budget
            last_birth, last_death = birth, death

            history_active.append(active)

        cell_weight = last_active.clamp_min(0.0)
        cell_weight = cell_weight / cell_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        active_cells = last_active.sum(dim=1)
        active_ticks = tick_gates.sum(dim=1)
        active_edges = route_support_sum
        expected_active_edges = route_effective_sum
        active_lookback_edges = lookback_support_sum
        expected_active_lookback_edges = lookback_effective_sum

        input_effective = effective_support(input_route, dim=-1).sum(dim=1) / max(self.K * self.obs_dim, 1)
        write_compute = write_intensity_sum / max(self.K * self.T, 1)
        edge_compute = route_effective_sum / max(self.K * (self.K - 1) * self.T, 1)
        lookback_capacity = max((self.T * (self.T - 1) // 2) * self.K * self.K, 1)
        lookback_compute = lookback_effective_sum / lookback_capacity
        compute = (input_effective + write_compute + edge_compute + lookback_compute) / 4.0

        route_entropy = route_entropy_sum / write_intensity_sum.clamp_min(1e-6) / np.log(max(self.K, 2))
        lookback_route_entropy = lookback_entropy_sum / write_intensity_sum.clamp_min(1e-6)
        lookback_route_entropy = lookback_route_entropy / np.log(max(lookback_entropy_count, 2))

        stats = {
            "compute": compute.clamp(min=0.0),
            "active_cells": active_cells,
            "active_ticks": active_ticks,
            "active_edges": active_edges,
            "active_edge_frac": active_edges / max(self.K * (self.K - 1) * self.T, 1),
            "expected_active_edges": expected_active_edges,
            "expected_active_edge_frac": expected_active_edges / max(self.K * (self.K - 1) * self.T, 1),
            "active_lookback_edges": active_lookback_edges,
            "active_lookback_edge_frac": active_lookback_edges / lookback_capacity,
            "expected_active_lookback_edges": expected_active_lookback_edges,
            "expected_active_lookback_edge_frac": expected_active_lookback_edges / lookback_capacity,
            "edge_entropy": route_entropy,
            "expected_edge_entropy": route_entropy,
            "edge_noise": x.new_empty(B, 0, self.K, self.K),
            "lookback_edge_entropy": lookback_route_entropy,
            "expected_lookback_edge_entropy": lookback_route_entropy,
            "lookback_edge_noise": x.new_empty(B, 0, self.K, self.K),
            "route_entropy": route_entropy,
            "lookback_route_entropy": lookback_route_entropy,
            "growth_pressure": last_birth.mean(dim=1),
            "shrink_pressure": last_death.mean(dim=1),
            "plasticity": last_plasticity.mean(dim=1),
            "persistence": torch.sigmoid(self.persistence_gate(self._gate_input(h, cell_prop))).squeeze(-1).mean(dim=1),
            "budget": last_budget.mean(dim=1),
            "temperature": x.new_full((B,), 1.0),
            "viability": last_viability.mean(dim=1),
            "input_route_support": (sparse_input_route > 1e-6).to(x.dtype).sum(dim=-1).mean(dim=1),
        }
        return h, cell_weight, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor = PropertySymmetricSubstrate(obs_dim, args)
        self.critic = PropertySymmetricSubstrate(obs_dim, args)
        self.action_prop = nn.Parameter(torch.randn(action_dim, args.property_dim) * 0.5)
        self.value_prop = nn.Parameter(torch.randn(1, args.property_dim) * 0.5)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def _add_read_compute(self, stats, read_compute):
        stats = dict(stats)
        stats["compute"] = (stats["compute"] + read_compute).clamp(min=0.0)
        return stats

    def get_value(self, x):
        critic_cells, critic_weight, critic_stats = self.critic(x)
        value_features, value_read_compute, _, _ = self.critic.read(self.value_prop, critic_cells, critic_weight)
        critic_stats = self._add_read_compute(critic_stats, value_read_compute)
        return self.critic_value(value_features).squeeze(-1).squeeze(1)

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def _actor_dist(self, actor_cells, actor_weight):
        action_features, action_read_compute, action_read_entropy, _ = self.actor.read(
            self.action_prop, actor_cells, actor_weight
        )
        alpha = 1.0 + F.softplus(self.actor_alpha(action_features).squeeze(-1))
        beta = 1.0 + F.softplus(self.actor_beta(action_features).squeeze(-1))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action, action_read_compute, action_read_entropy

    def get_action_and_value(self, x, z=None):
        actor_cells, actor_weight, actor_stats = self.actor(x)
        critic_cells, critic_weight, critic_stats = self.critic(x)
        dist, to_action, action_read_compute, action_read_entropy = self._actor_dist(actor_cells, actor_weight)
        value_features, value_read_compute, value_read_entropy, _ = self.critic.read(
            self.value_prop, critic_cells, critic_weight
        )
        actor_stats = self._add_read_compute(actor_stats, action_read_compute)
        critic_stats = self._add_read_compute(critic_stats, value_read_compute)
        actor_stats["read_entropy"] = action_read_entropy
        critic_stats["read_entropy"] = value_read_entropy

        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)

        stats = {
            "actor": actor_stats,
            "critic": critic_stats,
        }
        return (
            action,
            z,
            logprob,
            entropy,
            self.critic_value(value_features).squeeze(-1).squeeze(1),
            actor_stats["compute"],
            critic_stats["compute"],
            stats,
        )


def mean_stat(stats, group, name):
    return stats[group][name].detach().mean().item()


def signed_loss_with_safe_compute(task_loss_sample, multiplier, floor):
    task = (task_loss_sample * multiplier.detach()).mean()
    compute = ((task_loss_sample.detach().abs() + floor) * multiplier).mean()
    return task + compute - compute.detach()


def positive_loss_with_safe_compute(task_loss_sample, multiplier, floor):
    task = (task_loss_sample * multiplier.detach()).mean()
    compute = ((task_loss_sample.detach() + floor) * multiplier).mean()
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
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                (
                    action,
                    z,
                    logprob,
                    _,
                    value,
                    _,
                    _,
                    last_stats,
                ) = agent.get_action_and_value(next_obs)
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

                (
                    _,
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    actor_compute,
                    critic_compute,
                    last_stats,
                ) = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds])
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
                pg_loss = signed_loss_with_safe_compute(
                    pg_loss_per_sample,
                    actor_multiplier,
                    args.actor_compute_loss_floor,
                )
                actor_loss_magnitude = pg_loss_per_sample.detach().abs() + args.actor_compute_loss_floor

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_per_sample = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss_per_sample = 0.5 * (newvalue - b_returns[mb_inds]) ** 2
                v_loss = positive_loss_with_safe_compute(
                    v_loss_per_sample,
                    critic_multiplier,
                    args.critic_compute_loss_floor,
                )

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
            writer.add_scalar("morph/actor_compute", mean_stat(last_stats, "actor", "compute"), global_step)
            writer.add_scalar("morph/critic_compute", mean_stat(last_stats, "critic", "compute"), global_step)
            writer.add_scalar("morph/actor_active_cells", mean_stat(last_stats, "actor", "active_cells"), global_step)
            writer.add_scalar("morph/critic_active_cells", mean_stat(last_stats, "critic", "active_cells"), global_step)
            writer.add_scalar("morph/actor_active_ticks", mean_stat(last_stats, "actor", "active_ticks"), global_step)
            writer.add_scalar("morph/critic_active_ticks", mean_stat(last_stats, "critic", "active_ticks"), global_step)
            writer.add_scalar("morph/actor_expected_active_edges", mean_stat(last_stats, "actor", "expected_active_edges"), global_step)
            writer.add_scalar("morph/critic_expected_active_edges", mean_stat(last_stats, "critic", "expected_active_edges"), global_step)
            writer.add_scalar("morph/actor_expected_active_lookback_edges", mean_stat(last_stats, "actor", "expected_active_lookback_edges"), global_step)
            writer.add_scalar("morph/critic_expected_active_lookback_edges", mean_stat(last_stats, "critic", "expected_active_lookback_edges"), global_step)
            writer.add_scalar("morph/actor_route_entropy", mean_stat(last_stats, "actor", "route_entropy"), global_step)
            writer.add_scalar("morph/critic_route_entropy", mean_stat(last_stats, "critic", "route_entropy"), global_step)
            writer.add_scalar("morph/actor_lookback_route_entropy", mean_stat(last_stats, "actor", "lookback_route_entropy"), global_step)
            writer.add_scalar("morph/critic_lookback_route_entropy", mean_stat(last_stats, "critic", "lookback_route_entropy"), global_step)
            writer.add_scalar("morph/actor_read_entropy", mean_stat(last_stats, "actor", "read_entropy"), global_step)
            writer.add_scalar("morph/critic_read_entropy", mean_stat(last_stats, "critic", "read_entropy"), global_step)
            writer.add_scalar("morph/growth_pressure_mean", mean_stat(last_stats, "actor", "growth_pressure"), global_step)
            writer.add_scalar("morph/shrink_pressure_mean", mean_stat(last_stats, "actor", "shrink_pressure"), global_step)
            writer.add_scalar("morph/plasticity_mean", mean_stat(last_stats, "actor", "plasticity"), global_step)
            writer.add_scalar("morph/persistence_mean", mean_stat(last_stats, "actor", "persistence"), global_step)
            writer.add_scalar("morph/budget_mean", mean_stat(last_stats, "actor", "budget"), global_step)
            writer.add_scalar("morph/viability_mean", mean_stat(last_stats, "actor", "viability"), global_step)
            writer.add_scalar("morph/compute_multiplier_actor", actor_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/compute_multiplier_critic", critic_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/actor_compute_loss_magnitude", actor_loss_magnitude.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        class EvalAgent(Agent):
            def __init__(self, envs):
                super().__init__(envs, args)

            def get_action_and_value(self, x, z=None):
                action, _, logprob, entropy, value, _, _, _ = super().get_action_and_value(x, z)
                return action, logprob, entropy, value

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=EvalAgent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
