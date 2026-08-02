# PPO + adaptive-capacity and adaptive-depth sparse perceptrons v3.
#
# Each trunk has a mandatory sparse stem and a learned hard prefix of sequential
# all-prior blocks. It executes the active prefix plus one zero-forward frontier
# block, giving depth a task gradient without paying for all dormant blocks.
# Actor and critic depths are independent.  Perceptron fan-in remains adaptive;
# Taylor utility |w*dL/dw| selects mature edges for pruning and random rewiring.
# Exact depth, topology, and integer capacities stay frozen for a rollout/update.
#
# Compute-aware optimization uses m=2**(C/compute_double_connections) and the
# shift-invariant proxy m*(B + L - stop_grad(L)).  Its gradients are
# m*dL + B*dm, so signed/arbitrary-offset PPO losses cannot reward extra edges.
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

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
    total_timesteps: int = 8_000_000
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
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    width: int = 256
    """perceptrons per hidden layer"""
    initial_connections: int = 64
    """initial allowed incoming connections per perceptron"""
    min_connections: int = 1
    """minimum allowed incoming connections per perceptron"""
    edge_capacity: int = 65_536
    """weight slots per layer; shared unevenly across perceptrons"""
    initial_hidden_layers: int = 2
    """initial total depth per trunk, including the mandatory stem"""
    min_hidden_layers: int = 1
    """minimum total depth per trunk"""
    max_hidden_layers: int = 8
    """preallocated maximum total depth per trunk"""
    depth_tau: float = 0.5
    """temperature of the ordered depth boundary gates"""
    depth_learning_rate: float = 3e-5
    """Adam learning rate for log-depth parameters"""
    layer_overhead_connections: float = 2_048.0
    """connection-equivalent activation/routing cost per optional active layer"""
    pool: str = "prior"
    """prev | prior source pool"""
    capacity_tau: float = 0.5
    """temperature of ordered boundary straight-through gates"""
    compute_double_connections: float = 20_000.0
    """each additional count doubles the compute multiplier"""
    compute_loss_baseline: float = 1.0
    """positive, loss-offset-invariant strength of the compute gradient"""
    utility_ema: float = 0.99
    """EMA decay for Taylor edge utility |w*dL/dw|"""
    utility_rewire_fraction: float = 0.0
    """optional mature-edge exploration churn; growth/shrink already rewires"""
    utility_age_min: int = 100
    """optimizer steps before an edge can be pruned or rewired"""
    compile: bool = False
    compile_mode: str = "default"

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
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AdaptiveSparseLinear(nn.Module):
    """Packed hard-edge layer with independently learned fan-in per output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_connections: int,
        min_connections: int,
        edge_capacity: int,
        capacity_tau: float,
        utility_ema: float,
        utility_rewire_fraction: float,
        utility_age_min: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.min_connections = int(min(min_connections, in_features))
        self.capacity_tau = float(capacity_tau)
        self.utility_ema = float(utility_ema)
        self.utility_rewire_fraction = float(utility_rewire_fraction)
        self.utility_age_min = int(utility_age_min)

        max_unique_edges = self.in_features * self.out_features
        requested_capacity = min(int(edge_capacity), max_unique_edges)
        # A complete arena needs no separately reserved probes: whenever a unit
        # is not dense, one of its missing edges is itself a free probe slot.
        self.full_arena = requested_capacity == max_unique_edges
        probe_reserve = 0 if self.full_arena else self.out_features
        min_storage = self.out_features * self.min_connections + probe_reserve
        if requested_capacity < min_storage:
            raise ValueError(
                f"edge_capacity={requested_capacity} must fit min edges plus probes={min_storage}"
            )
        self.edge_capacity = requested_capacity
        self.max_live_edges = requested_capacity - probe_reserve

        init_count = min(max(int(initial_connections), self.min_connections), self.in_features)
        if init_count * self.out_features > self.max_live_edges:
            raise ValueError(
                "edge_capacity cannot hold initial_connections for every perceptron plus probes"
            )
        # Direct capacities plus a straight-through projection retain recovery
        # gradients at both bounds.  Softplus would strand units near k_min.
        self.capacity_raw = nn.Parameter(
            torch.full((self.out_features,), float(init_count))
        )
        self.weight = nn.Parameter(torch.zeros(self.edge_capacity))
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        self.register_buffer("source", torch.zeros(self.edge_capacity, dtype=torch.long))
        self.register_buffer("destination", torch.zeros(self.edge_capacity, dtype=torch.long))
        self.register_buffer("utility", torch.zeros(self.edge_capacity))
        self.register_buffer("age", torch.zeros(self.edge_capacity))
        self.register_buffer("is_live", torch.zeros(self.edge_capacity, dtype=torch.bool))
        self.register_buffer("is_probe", torch.zeros(self.edge_capacity, dtype=torch.bool))
        self.register_buffer("rank", torch.zeros(self.edge_capacity))
        self.register_buffer(
            "live_count", torch.full((self.out_features,), init_count, dtype=torch.long)
        )
        # Derived, variable-length execution plan.  Persisting it makes a
        # checkpoint unloadable when saved and freshly initialized counts differ.
        self.register_buffer("executed_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_load_state_dict_post_hook(self._post_load_state_dict)
        self.last_grown = 0
        self.last_pruned = 0
        self.last_rewired = 0
        self._initialize_arena(init_count)

    @torch.no_grad()
    def _initialize_arena(self, init_count: int) -> None:
        next_slot = 0
        # The forward divides each sum by sqrt(fan-in), so raw edge weights use
        # fan-in-independent He scale.  Using sqrt(2/k) here would normalize
        # twice and collapse high-capacity perceptrons' activations.
        std = math.sqrt(2.0)
        for dst in range(self.out_features):
            sources = torch.randperm(self.in_features)
            live_sources = sources[:init_count]
            ids = torch.arange(next_slot, next_slot + init_count)
            self.source[ids] = live_sources
            self.destination[ids] = dst
            self.is_live[ids] = True
            self.rank[ids] = torch.arange(init_count, dtype=self.rank.dtype)
            self.weight.data[ids] = torch.randn(init_count) * std
            next_slot += init_count

            if init_count < self.in_features:
                probe_id = next_slot
                self.source[probe_id] = int(sources[init_count])
                self.destination[probe_id] = dst
                self.is_probe[probe_id] = True
                self.rank[probe_id] = init_count
                self.weight.data[probe_id] = torch.randn(()) * std
                next_slot += 1
        self._refresh_executed_ids()

    def soft_capacities(self) -> torch.Tensor:
        unbounded = self.capacity_raw
        bounded = unbounded.clamp(
            min=float(self.min_connections), max=float(self.in_features)
        )
        # Straight-through clamp: impossible forward counts stay bounded while
        # task gradients can recover a parameter from either boundary.
        raw = unbounded + (bounded - unbounded).detach()
        minimum_total = float(self.out_features * self.min_connections)
        available_extra = float(self.max_live_edges) - minimum_total
        extra = raw - self.min_connections
        scale = torch.clamp(available_extra / extra.sum().clamp_min(1e-6), max=1.0)
        projected = self.min_connections + extra * scale
        # A hard storage projection must not pin d(total compute)/d(capacity) to
        # zero at saturation.  Forward uses the feasible allocation; backward
        # retains the radial degree of freedom that can learn to shrink it.
        return raw + (projected - raw).detach()

    def expected_connections(self) -> torch.Tensor:
        return self.soft_capacities().sum()

    def hard_connections(self) -> int:
        return int(self.live_count.sum().item())

    def capacity_stats(self) -> tuple[float, float, float, float]:
        counts = self.live_count.float()
        return (
            counts.mean().item(),
            counts.min().item(),
            counts.max().item(),
            counts.std(unbiased=False).item(),
        )

    def _post_load_state_dict(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        self._refresh_executed_ids()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_segments([x])

    def forward_segments(self, segments: list[torch.Tensor]) -> torch.Tensor:
        """Gather once from a logical concatenation of prior activations."""
        if not segments:
            raise ValueError("sparse layer requires at least one input segment")
        total_features = sum(segment.shape[-1] for segment in segments)
        if total_features != self.in_features:
            raise ValueError(
                f"input segments contain {total_features} features, expected {self.in_features}"
            )
        ids = self.executed_ids
        if ids.numel() == 0:
            return self.bias.expand(segments[0].shape[0], -1)
        dst = self.destination[ids]
        src = self.source[ids]
        capacity = self.soft_capacities()[dst]
        soft_gate = torch.sigmoid((capacity - self.rank[ids] - 0.5) / self.capacity_tau)
        hard_gate = self.is_live[ids].to(segments[0].dtype)
        gate = hard_gate + soft_gate - soft_gate.detach()
        hard_normalizer = self.live_count[dst].to(segments[0].dtype).clamp_min(1).rsqrt()
        soft_normalizer = capacity.clamp_min(1).rsqrt()
        normalizer = hard_normalizer + soft_normalizer - soft_normalizer.detach()
        inputs = segments[0] if len(segments) == 1 else torch.cat(segments, dim=-1)
        contributions = inputs[:, src] * self.weight[ids] * gate * normalizer
        output = inputs.new_zeros((inputs.shape[0], self.out_features))
        output.index_add_(1, dst, contributions)
        return output + self.bias

    @torch.no_grad()
    def update_utility_from_grad(self) -> None:
        if self.weight.grad is None:
            return
        ids = self.is_live.nonzero(as_tuple=False).squeeze(-1)
        sample = (self.weight[ids] * self.weight.grad[ids]).abs()
        # Advanced indexing returns a copy, so these must be explicit writes.
        self.utility[ids] = self.utility[ids] * self.utility_ema + sample * (
            1.0 - self.utility_ema
        )
        self.age[ids] = self.age[ids] + 1

    @staticmethod
    def _reset_optimizer_slots(
        optimizer: Optional[optim.Optimizer], parameter: nn.Parameter, ids: torch.Tensor
    ) -> None:
        if optimizer is None or ids.numel() == 0:
            return
        state = optimizer.state.get(parameter)
        if state is None:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[ids] = 0

    @torch.no_grad()
    def _free_ids(self) -> torch.Tensor:
        return (~self.is_live & ~self.is_probe).nonzero(as_tuple=False).squeeze(-1)

    @torch.no_grad()
    def _unused_source(self, dst: int, preferred: Optional[int] = None) -> Optional[int]:
        live_ids = (self.is_live & (self.destination == dst)).nonzero(as_tuple=False).squeeze(-1)
        used = torch.zeros(self.in_features, dtype=torch.bool, device=self.source.device)
        used[self.source[live_ids]] = True
        if preferred is not None and not bool(used[preferred]):
            return int(preferred)
        candidates = (~used).nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            return None
        index = torch.randint(candidates.numel(), (), device=candidates.device)
        return int(candidates[index])

    @torch.no_grad()
    def _deactivate(self, ids: torch.Tensor, optimizer: Optional[optim.Optimizer]) -> None:
        if ids.numel() == 0:
            return
        self.is_live[ids] = False
        self.utility[ids] = 0
        self.age[ids] = 0
        self.weight.data[ids] = 0
        self._reset_optimizer_slots(optimizer, self.weight, ids)

    @torch.no_grad()
    def _activate(
        self,
        slot: int,
        dst: int,
        source: int,
        rank: int,
        optimizer: Optional[optim.Optimizer],
    ) -> None:
        slot_tensor = torch.tensor([slot], device=self.weight.device)
        self.destination[slot] = dst
        self.source[slot] = source
        self.rank[slot] = rank
        self.is_live[slot] = True
        self.is_probe[slot] = False
        self.utility[slot] = 0
        self.age[slot] = 0
        self.weight.data[slot] = torch.randn((), device=self.weight.device) * math.sqrt(2.0)
        self._reset_optimizer_slots(optimizer, self.weight, slot_tensor)

    @torch.no_grad()
    def _rewire_slot(self, slot: int, optimizer: Optional[optim.Optimizer]) -> bool:
        dst = int(self.destination[slot])
        source = self._unused_source(dst)
        if source is None:
            return False
        slot_tensor = torch.tensor([slot], device=self.weight.device)
        self.source[slot] = source
        self.utility[slot] = 0
        self.age[slot] = 0
        self.weight.data[slot] = torch.randn((), device=self.weight.device) * math.sqrt(2.0)
        self._reset_optimizer_slots(optimizer, self.weight, slot_tensor)
        return True

    @torch.no_grad()
    def _refresh_probes(self, optimizer: Optional[optim.Optimizer]) -> None:
        old_probes = self.is_probe.nonzero(as_tuple=False).squeeze(-1)
        self.is_probe[old_probes] = False
        self.weight.data[old_probes] = 0
        self._reset_optimizer_slots(optimizer, self.weight, old_probes)

        free = self._free_ids().tolist()
        cursor = 0
        for dst in range(self.out_features):
            source = self._unused_source(dst)
            if source is None:
                continue
            if cursor >= len(free):
                raise RuntimeError("edge arena has no reserved slot for capacity probes")
            slot = free[cursor]
            cursor += 1
            self.destination[slot] = dst
            self.source[slot] = source
            self.rank[slot] = int(self.live_count[dst])
            self.is_probe[slot] = True
            self.weight.data[slot] = torch.randn((), device=self.weight.device) * math.sqrt(2.0)

    @torch.no_grad()
    def _refresh_executed_ids(self) -> None:
        self.executed_ids = (self.is_live | self.is_probe).nonzero(as_tuple=False).squeeze(-1)

    @torch.no_grad()
    def materialize(self, optimizer: Optional[optim.Optimizer] = None) -> tuple[int, int, int]:
        """Apply learned counts, utility shrink/prune, and fresh growth probes."""
        self.capacity_raw.clamp_(
            min=float(self.min_connections), max=float(self.in_features)
        )
        target = self.soft_capacities().round().to(torch.long)
        target.clamp_(min=self.min_connections, max=self.in_features)
        overflow = int(target.sum()) - self.max_live_edges
        if overflow > 0:
            order = torch.argsort(self.soft_capacities() - target, descending=False)
            while overflow > 0:
                changed = False
                for dst in order.tolist():
                    if target[dst] > self.min_connections:
                        target[dst] -= 1
                        overflow -= 1
                        changed = True
                        if overflow == 0:
                            break
                if not changed:
                    raise RuntimeError("could not project hard capacities into edge arena")

        grown = 0
        pruned = 0
        rewired = 0
        # Capacity shrink keeps the most useful edges, protecting immature edges first.
        for dst in range(self.out_features):
            ids = (self.is_live & (self.destination == dst)).nonzero(as_tuple=False).squeeze(-1)
            desired = int(target[dst])
            if ids.numel() > desired:
                mature = self.age[ids] >= self.utility_age_min
                mature_ids = ids[mature]
                remove_n = min(int(ids.numel()) - desired, int(mature_ids.numel()))
                if remove_n == 0:
                    target[dst] = ids.numel()
                    continue
                remove_local = torch.topk(
                    self.utility[mature_ids], remove_n, largest=False
                ).indices
                remove = mature_ids[remove_local]
                self._deactivate(remove, optimizer)
                pruned += remove_n
                # Capacity changes wait for age eligibility rather than silently
                # violating the protection contract.
                target[dst] = ids.numel() - remove_n

        # Capacity growth preferentially promotes this iteration's random probe.
        free = self._free_ids().tolist()
        free_cursor = 0
        for dst in range(self.out_features):
            current = int((self.is_live & (self.destination == dst)).sum())
            desired = int(target[dst])
            probe_ids = (self.is_probe & (self.destination == dst)).nonzero(as_tuple=False).squeeze(-1)
            preferred = int(self.source[probe_ids[0]]) if probe_ids.numel() else None
            while current < desired:
                source = self._unused_source(dst, preferred if current == int(self.live_count[dst]) else None)
                if source is None:
                    break
                if current == int(self.live_count[dst]) and probe_ids.numel():
                    # Promote the differentiable growth probe that supplied the
                    # boundary signal for this perceptron.
                    slot = int(probe_ids[0])
                else:
                    if free_cursor >= len(free):
                        raise RuntimeError("edge arena exhausted while growing capacity")
                    slot = free[free_cursor]
                    free_cursor += 1
                self._activate(slot, dst, source, current, optimizer)
                current += 1
                grown += 1

        self.live_count.copy_(target)

        # Utility pruning is one-for-one and therefore independent of capacity.
        if self.utility_rewire_fraction > 0:
            mature_ids = (
                self.is_live & (self.age >= self.utility_age_min)
            ).nonzero(as_tuple=False).squeeze(-1)
            n = min(
                mature_ids.numel(),
                max(1, int(self.utility_rewire_fraction * mature_ids.numel())),
            )
            if n > 0:
                low = torch.topk(self.utility[mature_ids], n, largest=False).indices
                for slot in mature_ids[low].tolist():
                    rewired += int(self._rewire_slot(slot, optimizer))

        # Stable utility order defines marginal ranks for the next frozen topology.
        for dst in range(self.out_features):
            ids = (self.is_live & (self.destination == dst)).nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(self.utility[ids], descending=True, stable=True)
            self.rank[ids[order]] = torch.arange(
                ids.numel(), device=ids.device, dtype=self.rank.dtype
            )
        self._refresh_probes(optimizer)
        self._refresh_executed_ids()
        self.last_grown, self.last_pruned, self.last_rewired = grown, pruned, rewired
        return grown, pruned, rewired


class SparseTrunk(nn.Module):
    def __init__(self, obs_dim: int, args: Args):
        super().__init__()
        self.any_prior = args.pool == "prior"
        self.min_depth = int(args.min_hidden_layers)
        self.max_depth = int(args.max_hidden_layers)
        self.depth_tau = float(args.depth_tau)
        self.layer_overhead_connections = float(args.layer_overhead_connections)
        initial_depth = min(max(int(args.initial_hidden_layers), self.min_depth), self.max_depth)
        self.depth_log = nn.Parameter(torch.tensor(math.log(float(initial_depth))))
        self.register_buffer("hard_depth", torch.tensor(initial_depth, dtype=torch.long))
        # Python-side cache avoids a CUDA scalar extraction in every hot forward
        # and compute-cost call. The tensor buffer exists only for checkpoints.
        self._hard_depth = initial_depth
        self.register_load_state_dict_post_hook(self._post_load_state_dict)
        self.last_depth_change = 0

        layer_kw = dict(
            initial_connections=args.initial_connections,
            min_connections=args.min_connections,
            edge_capacity=args.edge_capacity,
            capacity_tau=args.capacity_tau,
            utility_ema=args.utility_ema,
            utility_rewire_fraction=args.utility_rewire_fraction,
            utility_age_min=args.utility_age_min,
        )
        self.stem = AdaptiveSparseLinear(obs_dim, args.width, **layer_kw)
        self.blocks = nn.ModuleList()
        for layer_index in range(1, self.max_depth):
            in_features = obs_dim + layer_index * args.width if self.any_prior else args.width
            self.blocks.append(
                AdaptiveSparseLinear(in_features, args.width, **layer_kw)
            )

    def _post_load_state_dict(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        self._hard_depth = int(self.hard_depth.detach().cpu())

    def soft_depth(self) -> torch.Tensor:
        unbounded = self.depth_log.exp()
        bounded = unbounded.clamp(min=float(self.min_depth), max=float(self.max_depth))
        return unbounded + (bounded - unbounded).detach()

    def depth_gate(self, layer_index: int) -> torch.Tensor:
        soft = torch.sigmoid(
            (self.soft_depth() - float(layer_index) - 0.5) / self.depth_tau
        )
        hard = soft.new_tensor(float(layer_index < self._hard_depth))
        return hard + soft - soft.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.stem(x))
        segments = [x, h]
        hard_depth = self._hard_depth
        executed_depth = min(hard_depth + 1, self.max_depth)
        for layer_index in range(1, executed_depth):
            block = self.blocks[layer_index - 1]
            inputs = segments if self.any_prior else [h]
            delta = F.silu(block.forward_segments(inputs))
            if layer_index == hard_depth:
                # The frontier is a fixed random probe: its output informs depth,
                # but its dormant weights/capacities must receive grad=None.
                delta = delta.detach()
            gate = self.depth_gate(layer_index)
            # Active gates reproduce the established SPK trunk exactly: h=delta.
            # The inactive frontier has h unchanged while d(h)/d(depth) probes
            # whether replacing h by the next block would improve the task loss.
            h = h + gate * (delta - h)
            segments.append(h)
        return h

    def expected_connections(self) -> torch.Tensor:
        total = self.stem.expected_connections()
        for layer_index, block in enumerate(self.blocks, start=1):
            if layer_index > self._hard_depth:
                break
            block_cost = block.expected_connections()
            if layer_index >= self._hard_depth:
                block_cost = block_cost.detach()
            total = total + self.depth_gate(layer_index) * (
                block_cost + self.layer_overhead_connections
            )
        return total

    def expected_edge_connections(self) -> torch.Tensor:
        """Expected deployed edges, excluding the explicit per-layer overhead."""
        total = self.stem.expected_connections()
        for layer_index, block in enumerate(self.blocks, start=1):
            if layer_index > self._hard_depth:
                break
            block_cost = block.expected_connections()
            if layer_index >= self._hard_depth:
                block_cost = block_cost.detach()
            total = total + self.depth_gate(layer_index) * block_cost
        return total

    def hard_connections(self) -> int:
        active_blocks = self.blocks[: max(self._hard_depth - 1, 0)]
        return self.stem.hard_connections() + sum(
            block.hard_connections() for block in active_blocks
        )

    def physical_connections(self) -> int:
        """Edges gathered in training, including the zero-forward frontier."""
        total = int(self.stem.executed_ids.numel())
        executed_blocks = self.blocks[: min(self._hard_depth, len(self.blocks))]
        return total + sum(int(block.executed_ids.numel()) for block in executed_blocks)

    def active_layers(self) -> list[tuple[str, AdaptiveSparseLinear]]:
        layers = [("stem", self.stem)]
        layers.extend(
            (f"block_{index}", block)
            for index, block in enumerate(
                self.blocks[: max(self._hard_depth - 1, 0)], start=1
            )
        )
        return layers

    def update_utility_from_grad(self) -> None:
        self.stem.update_utility_from_grad()
        for layer in self.blocks[: max(self._hard_depth - 1, 0)]:
            layer.update_utility_from_grad()

    @torch.no_grad()
    def materialize(self, optimizer: Optional[optim.Optimizer]) -> tuple[int, int, int]:
        totals = [0, 0, 0]
        for _, layer in self.active_layers():
            result = layer.materialize(optimizer)
            totals = [left + right for left, right in zip(totals, result)]
        old_depth = self._hard_depth
        self.depth_log.clamp_(
            min=math.log(float(self.min_depth)),
            max=math.log(float(self.max_depth)),
        )
        desired_depth = int(self.soft_depth().round().clamp(self.min_depth, self.max_depth))
        # Only the adjacent frontier is task-probed, so a materialization may
        # cross at most one boundary in either direction.
        new_depth = min(max(desired_depth, old_depth - 1), old_depth + 1)
        self.hard_depth.fill_(new_depth)
        self._hard_depth = new_depth
        self.last_depth_change = new_depth - old_depth
        return tuple(totals)


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        self.args = args
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_trunk = SparseTrunk(obs_dim, args)
        self.critic_trunk = SparseTrunk(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.width, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.width, action_dim), std=0.01)
        self.critic_out = layer_init(nn.Linear(args.width, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def expected_connections(self) -> torch.Tensor:
        return (
            self.actor_trunk.expected_edge_connections()
            + self.critic_trunk.expected_edge_connections()
        )

    def expected_compute_connections(self) -> torch.Tensor:
        return self.actor_trunk.expected_connections() + self.critic_trunk.expected_connections()

    def hard_connections(self) -> int:
        return self.actor_trunk.hard_connections() + self.critic_trunk.hard_connections()

    def physical_connections(self) -> int:
        return self.actor_trunk.physical_connections() + self.critic_trunk.physical_connections()

    def hard_depths(self) -> tuple[int, int]:
        return self.actor_trunk._hard_depth, self.critic_trunk._hard_depth

    def soft_depths(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_trunk.soft_depth(), self.critic_trunk.soft_depth()

    def compute_multiplier(self) -> torch.Tensor:
        return torch.exp2(
            self.expected_compute_connections() / self.args.compute_double_connections
        )

    def update_utility_from_grad(self) -> None:
        self.actor_trunk.update_utility_from_grad()
        self.critic_trunk.update_utility_from_grad()

    def materialize(self, optimizer: Optional[optim.Optimizer]) -> tuple[int, int, int]:
        actor = self.actor_trunk.materialize(optimizer)
        critic = self.critic_trunk.materialize(optimizer)
        return tuple(left + right for left, right in zip(actor, critic))

    def _dist(self, x: torch.Tensor) -> Beta:
        h = self.actor_trunk(x)
        return Beta(1.0 + F.softplus(self.actor_alpha(h)), 1.0 + F.softplus(self.actor_beta(h)))

    def _z_to_action(self, z: torch.Tensor) -> torch.Tensor:
        return self.action_low + (self.action_high - self.action_low) * z

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_out(self.critic_trunk(x))

    def get_beta_action_and_value(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self._z_to_action(z)
        return action, z, dist.log_prob(z).sum(1), dist.entropy().sum(1), self.get_value(x)


def log_capacity_distribution(
    writer: SummaryWriter,
    prefix: str,
    values: torch.Tensor,
    global_step: int,
) -> dict[str, float]:
    """Log the full perceptron fan-in distribution and return scalar summaries."""
    values = values.detach().float().cpu()
    quantile_levels = values.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    quantiles = torch.quantile(values, quantile_levels)
    stats = {
        "total": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(quantiles[2]),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "unique": float(values.unique().numel()),
    }
    for name, value in stats.items():
        writer.add_scalar(f"{prefix}_{name}", value, global_step)
    writer.add_histogram(f"{prefix}_histogram", values, global_step)
    return stats


def log_capacity_gap(
    writer: SummaryWriter,
    prefix: str,
    hard: torch.Tensor,
    soft: torch.Tensor,
    global_step: int,
) -> None:
    gap = (soft.detach().float() - hard.detach().float()).cpu()
    writer.add_scalar(f"{prefix}_mean", float(gap.mean()), global_step)
    writer.add_scalar(f"{prefix}_mean_abs", float(gap.abs().mean()), global_step)
    writer.add_scalar(f"{prefix}_max_abs", float(gap.abs().max()), global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.pool not in ("prev", "prior"):
        raise ValueError(f"pool must be prev|prior, got {args.pool}")
    if not (1 <= args.min_hidden_layers <= args.initial_hidden_layers <= args.max_hidden_layers):
        raise ValueError("depths must satisfy 1 <= min <= initial <= max")
    if args.depth_learning_rate <= 0 or args.depth_tau <= 0:
        raise ValueError("depth learning rate and temperature must be positive")
    if args.layer_overhead_connections < 0:
        raise ValueError("layer overhead cannot be negative")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("adaptive SPK experiments require CUDA")
    if args.compile:
        raise ValueError("packed adaptive edge counts and depth are dynamic; --compile is not supported in v3")
    if args.compute_double_connections <= 0 or args.compute_loss_baseline <= 0:
        raise ValueError("compute cost scale and baseline must be positive")

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
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
        "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("only continuous action spaces are supported")
    agent = Agent(envs, args).to(device)
    depth_parameters = [agent.actor_trunk.depth_log, agent.critic_trunk.depth_log]
    depth_parameter_ids = {id(parameter) for parameter in depth_parameters}
    base_parameters = [
        parameter for parameter in agent.parameters() if id(parameter) not in depth_parameter_ids
    ]
    optimizer = optim.Adam(
        [
            {"params": base_parameters, "lr": args.learning_rate},
            {"params": depth_parameters, "lr": args.depth_learning_rate},
        ],
        eps=1e-5,
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            optimizer.param_groups[1]["lr"] = fraction * args.depth_learning_rate

        # Topology and integer capacities remain fixed from here through optimizer epochs.
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, z, logprob, _, value = agent.get_beta_action_and_value(next_obs)
                values[step] = value.flatten()
            zs[step] = z
            logprobs[step] = logprob
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values

        batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        batch_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)
        indices = np.arange(args.batch_size)
        clip_fractions = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                minibatch = indices[start : start + args.minibatch_size]
                _, _, new_logprob, entropy, new_value = agent.get_beta_action_and_value(
                    batch_obs[minibatch], batch_zs[minibatch]
                )
                log_ratio = new_logprob - batch_logprobs[minibatch]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = batch_advantages[minibatch]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                policy_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
                ).mean()

                new_value = new_value.view(-1)
                if args.clip_vloss:
                    value_unclipped = (new_value - batch_returns[minibatch]).square()
                    value_clipped_pred = batch_values[minibatch] + (
                        new_value - batch_values[minibatch]
                    ).clamp(-args.clip_coef, args.clip_coef)
                    value_clipped = (value_clipped_pred - batch_returns[minibatch]).square()
                    value_loss = 0.5 * torch.max(value_unclipped, value_clipped).mean()
                else:
                    value_loss = 0.5 * (new_value - batch_returns[minibatch]).square().mean()
                entropy_loss = entropy.mean()
                task_loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                multiplier = agent.compute_multiplier()
                loss = multiplier * (
                    args.compute_loss_baseline + task_loss - task_loss.detach()
                )

                optimizer.zero_grad()
                loss.backward()
                agent.update_utility_from_grad()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        grown, pruned, rewired = agent.materialize(optimizer)
        predicted = batch_values.cpu().numpy()
        targets = batch_returns.cpu().numpy()
        target_variance = np.var(targets)
        explained_variance = (
            np.nan if target_variance == 0 else 1 - np.var(targets - predicted) / target_variance
        )

        with torch.no_grad():
            actor_layers = [
                (f"actor_{name}", layer)
                for name, layer in agent.actor_trunk.active_layers()
            ]
            critic_layers = [
                (f"critic_{name}", layer)
                for name, layer in agent.critic_trunk.active_layers()
            ]
            named_layers = actor_layers + critic_layers
            expected_connections = float(agent.expected_connections())
            expected_compute_connections = float(agent.expected_compute_connections())
            hard_connections = agent.hard_connections()
            physical_connections = agent.physical_connections()
            compute_multiplier = float(agent.compute_multiplier())
            actor_depth, critic_depth = agent.hard_depths()
            actor_soft_depth, critic_soft_depth = (
                float(depth) for depth in agent.soft_depths()
            )
        scopes = [("global", named_layers), ("actor", actor_layers), ("critic", critic_layers)]
        scopes.extend((layer_name, [(layer_name, layer)]) for layer_name, layer in named_layers)
        hard_stats = None
        for scope_name, scope_layers in scopes:
            scope_hard = torch.cat(
                [layer.live_count.float() for _, layer in scope_layers]
            )
            scope_soft = torch.cat(
                [layer.soft_capacities() for _, layer in scope_layers]
            )
            scope_hard_stats = log_capacity_distribution(
                writer,
                f"sparse/capacity/hard/{scope_name}",
                scope_hard,
                global_step,
            )
            log_capacity_distribution(
                writer,
                f"sparse/capacity/soft/{scope_name}",
                scope_soft,
                global_step,
            )
            log_capacity_gap(
                writer,
                f"sparse/capacity/gap/{scope_name}",
                scope_hard,
                scope_soft,
                global_step,
            )
            if scope_name == "global":
                hard_stats = scope_hard_stats
        for layer_name, layer in named_layers:
            writer.add_scalar(
                f"sparse/capacity/arena_utilization/{layer_name}",
                layer.hard_connections() / layer.max_live_edges,
                global_step,
            )
            writer.add_scalar(
                f"sparse/capacity/fraction_at_min/{layer_name}",
                float((layer.live_count == layer.min_connections).float().mean()),
                global_step,
            )
            writer.add_scalar(
                f"sparse/capacity/fraction_at_max/{layer_name}",
                float((layer.live_count == layer.in_features).float().mean()),
                global_step,
            )
        assert hard_stats is not None
        # Stable aliases used by score_runs and the earlier v2 run.  These now
        # describe the true global distribution rather than means of layer stats.
        writer.add_scalar("sparse/capacity_mean", hard_stats["mean"], global_step)
        writer.add_scalar("sparse/capacity_median", hard_stats["median"], global_step)
        writer.add_scalar("sparse/capacity_min", hard_stats["min"], global_step)
        writer.add_scalar("sparse/capacity_max", hard_stats["max"], global_step)
        writer.add_scalar("sparse/capacity_std", hard_stats["std"], global_step)
        writer.add_scalar("sparse/connections_expected", expected_connections, global_step)
        writer.add_scalar(
            "sparse/compute_connections_expected", expected_compute_connections, global_step
        )
        writer.add_scalar("sparse/connections_hard", hard_connections, global_step)
        writer.add_scalar("sparse/connections_physical", physical_connections, global_step)
        writer.add_scalar("sparse/compute_multiplier", compute_multiplier, global_step)
        writer.add_scalar("sparse/depth/actor_hard", actor_depth, global_step)
        writer.add_scalar("sparse/depth/critic_hard", critic_depth, global_step)
        writer.add_scalar("sparse/depth/actor_soft", actor_soft_depth, global_step)
        writer.add_scalar("sparse/depth/critic_soft", critic_soft_depth, global_step)
        writer.add_scalar(
            "sparse/depth/active_layers", actor_depth + critic_depth, global_step
        )
        writer.add_scalar(
            "sparse/depth/executed_layers",
            min(actor_depth + 1, args.max_hidden_layers)
            + min(critic_depth + 1, args.max_hidden_layers),
            global_step,
        )
        writer.add_scalar(
            "sparse/depth/actor_change", agent.actor_trunk.last_depth_change, global_step
        )
        writer.add_scalar(
            "sparse/depth/critic_change", agent.critic_trunk.last_depth_change, global_step
        )
        writer.add_scalar("sparse/grown", grown, global_step)
        writer.add_scalar("sparse/pruned", pruned, global_step)
        writer.add_scalar("sparse/rewired", rewired, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/depth_learning_rate", optimizer.param_groups[1]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fractions), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(
            f"SPS={sps} hard_connections={hard_connections} "
            f"capacity_mean={hard_stats['mean']:.1f} "
            f"median={hard_stats['median']:.1f} "
            f"std={hard_stats['std']:.1f} "
            f"range=[{hard_stats['min']:.0f},{hard_stats['max']:.0f}] "
            f"depth=({actor_depth},{critic_depth}) "
            f"multiplier={compute_multiplier:.3f}"
        )
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()
