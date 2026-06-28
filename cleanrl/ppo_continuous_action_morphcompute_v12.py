# PPO + Morphogenic Compute v12.
#
# METHOD. Replace PPO's fixed actor/critic MLPs with separate differentiable
# computational substrates. Each substrate owns a maximum pool of latent cells.
# For each observation it predicts continuous laws over that substrate: growth
# and shrink pressure, coordinate-field routing, plasticity, persistence,
# abstraction temperature, and an adaptive compute budget. These laws softly
# determine how many cells are active, how strongly they exchange information,
# and how many recurrent update ticks are used. The graph is not edited
# discretely; the learned field induces the active computation.
#
# COMPUTE OBJECTIVE. PPO's policy and value losses are multiplied by a learned
# logical compute estimate:
#   policy loss = mean(stopgrad(mult_actor(s)) * clipped_ppo_loss(s))
#   value loss  = mean(mult_critic(s) * clipped_value_loss(s))
# where mult = 1 + compute_coef * normalized_compute. The multiplier remains
# differentiable so the model can reshape its substrate to optimize the
# loss-compute tradeoff, not just reward. Because the PPO actor surrogate is
# signed, the actor's morphogenesis gradient uses a zero-value multiplier term
# weighted by detached |clipped_ppo_loss|; otherwise good positive-advantage
# samples would perversely reward spending more compute.
#
# V2. Replace v1's unsquashed Normal actor with v168's Dreamer-style unimodal
# Beta actor: native z in [0, 1] is mapped linearly to the action bounds, with
# alpha,beta = 1 + softplus(head) so each dimension remains unimodal.
#
# V3. Replace all Tanh activations in the morphogenic substrate scaffolding with
# ReLU^2. The substrate's field/gating laws remain unchanged; this tests whether
# the higher-curvature, non-saturating positive activation improves MuJoCo credit
# flow versus bounded tanh features.
#
# V4. Replace the remaining GELU in the shared cell update law with ReLU^2, so
# every explicit substrate nonlinearity uses the same activation family. The
# raw observation embedding is simplified to a single linear projection; all
# nonlinearity lives inside the substrate update/readout and field gates. The
# readout is also kept as a plain Linear -> ReLU^2 projection, with no readout
# LayerNorm.
#
# V5. Add differentiable sparse edge existence gates. Cell-to-cell connections
# are no longer free dense softmax links: a stretched hard-concrete gate decides
# whether each directed edge exists, message passing normalizes over surviving
# edges, and active-edge fraction contributes directly to the compute multiplier.
#
# V6. Edge gates are stochastic hard-concrete samples during training, not
# deterministic clipped gates. Message passing uses sampled gates; the compute
# multiplier charges the expected L0 edge probability, so edge probabilities get
# stable cost gradients while sampled-open edges can still receive task signal.
#
# V7. Add sparse paid lookback edges across substrate time. At every update tick,
# each cell may read from any cell in any previous realized substrate layer.
# These historical reads use the same stochastic hard-concrete / expected-L0
# discipline as same-layer edges, and the cost is only the additional active
# edge probability: there is no artificial penalty for temporal distance.
#
# V8. Replace stochastic hard-concrete edge gates with deterministic sparsemax
# routing. Every current cell and every previous substrate cell remains a
# candidate communication source, but each destination chooses a sparse support
# by a differentiable projection. PPO no longer has hidden internal routing
# noise to replay; compute pressure uses differentiable effective support while
# diagnostics log exact nonzero route support.
#
# V9. Replace sparsemax routing with static alpha=1.5 entmax routing. This keeps
# exact-zero sparse supports and no-read candidates, but makes the projection
# smoother than sparsemax so useful edges are less likely to be pruned before
# PPO can assign credit.
#
# V12. True dynamic morphogenesis. The substrate no longer uses a fixed reserve
# cell pool. It starts with an initial number of cells, then after PPO updates
# deletes cells whose activity has negative loss-time utility and splits cells
# whose utility is strongly positive. Growth is local mitosis: new cells inherit
# a parent's state, coordinate, route/readout laws, and small perturbations.
#
# HYPOTHESIS. MuJoCo policies benefit from state-dependent computational
# morphogenesis: easy states should use fewer cells/ticks, hard states should
# grow local computation and route through richer latent geometry. The compute
# multiplier should discourage unused structure while preserving the option to
# spend compute when it reduces actor or critic loss.
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

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))
ENTMAX_SQRT_EPS = 1e-8  # avoid infinite threshold gradients at support-boundary ties


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
    cuda_contract: bool = False
    """run a deterministic CUDA morphogenesis/Adam/finite-gradient contract and exit"""
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

    # Morphogenic substrate arguments
    initial_cells: int = 32
    """initial latent cells in each dynamic substrate"""
    min_cells: int = 4
    """minimum cells retained by dynamic morphogenesis"""
    cell_dim: int = 64
    """latent cell/state width"""
    field_dim: int = 8
    """coordinate dimension of the latent computational manifold"""
    max_ticks: int = 4
    """maximum recurrent compute ticks per forward pass"""
    compute_coef: float = 0.05
    """strength of differentiable compute multiplier on actor and critic losses"""
    min_compute_multiplier: float = 1.0
    """base multiplier added before compute pressure"""
    field_temp_min: float = 0.25
    """minimum routing/abstraction temperature"""
    field_temp_max: float = 4.0
    """maximum routing/abstraction temperature"""
    route_mix: float = 0.5
    """how strongly learned routing messages influence each cell update"""
    init_active_bias: float = -1.1
    """initial active-cell logit bias, about 25 percent active before input pressure"""
    init_tick_bias: float = -0.7
    """initial tick logit bias, favoring roughly one to two active ticks"""
    actor_compute_loss_floor: float = 0.01
    """small nonnegative floor for actor morphogenesis pressure when PPO loss is near zero"""
    init_edge_bias: float = -0.5
    """initial directed-edge logit bias before coordinate/state fields"""
    init_null_route_bias: float = 1.0
    """initial no-read logit for same-tick routing"""
    edge_compute_weight: float = 1.0
    """relative cost of active directed edges in the normalized compute multiplier"""
    init_lookback_edge_bias: float = -1.5
    """initial historical directed-edge logit bias before state/coordinate fields"""
    init_lookback_null_route_bias: float = 1.0
    """initial no-read logit for historical routing"""
    lookback_compute_weight: float = 1.0
    """relative cost of active lookback edges in the normalized compute multiplier"""
    lookback_mix: float = 0.5
    """how strongly historical messages influence each cell update"""
    route_dense_floor: float = 0.0
    """optional softmax mixing floor for off-support task gradient; default keeps actual routing sparse"""
    entmax_alpha: float = 1.5
    """static entmax alpha for sparse routing; v9 supports alpha=1.5"""
    morph_prune_threshold: float = -0.25
    """cells below this normalized loss-time utility are deleted after an update"""
    morph_grow_threshold: float = 0.25
    """cells above this normalized loss-time utility are split after an update"""
    morph_max_deaths_per_iter: int = 2
    """maximum deleted cells per substrate after one PPO update"""
    morph_max_births_per_iter: int = 2
    """maximum new cells per substrate after one PPO update"""
    morph_warmup_iterations: int = 0
    """PPO iterations before dynamic grow/delete is allowed"""
    morph_child_noise: float = 0.02
    """relative noise added to mitosis child parameters"""
    morph_utility_eps: float = 1e-8
    """small denominator for normalizing cell utility before thresholding"""

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
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
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


def sparsemax(logits, dim=-1):
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, dim=dim, descending=True).values
    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.size(dim)
    rhos = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(range_shape)
    cssv = zs.cumsum(dim)
    support = 1 + rhos * zs > cssv
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cssv.gather(dim, support_size.long() - 1) - 1) / support_size
    return torch.clamp(logits - tau, min=0.0)


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


def effective_support(route, dim=-1):
    mass = route.sum(dim=dim)
    support = mass.pow(2) / route.pow(2).sum(dim=dim).clamp_min(1e-6)
    return mass * support


def entmax_route_with_floor(logits, dense_floor, alpha, dim=-1):
    if alpha != 1.5:
        raise ValueError("v9 only supports static entmax_alpha=1.5")
    sparse_route = entmax15(logits, dim=dim)
    if dense_floor <= 0.0:
        return sparse_route, sparse_route
    soft_route = torch.softmax(logits, dim=dim)
    route = (1.0 - dense_floor) * sparse_route + dense_floor * soft_route
    return route, sparse_route


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class MorphogenicSubstrate(nn.Module):
    """A fixed-capacity substrate whose active computation is generated by fields."""

    CELL_VECTOR_NAMES = {
        "cell_seed",
        "cell_pos",
        "null_route_bias",
        "lookback_null_route_bias",
    }
    CELL_MATRIX_NAMES = {
        "edge_bias",
        "lookback_edge_bias",
    }
    CELL_OUTPUT_LAYERS = {
        "growth",
        "shrink",
        "plasticity",
        "edge_source",
        "edge_target",
    }

    def __init__(self, obs_dim, args):
        super().__init__()
        self.K = args.initial_cells
        self.initial_K = args.initial_cells
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

        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.cell_seed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.cell_pos = nn.Parameter(torch.randn(self.K, self.F) * 0.5)
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

        self.norm = nn.LayerNorm(self.D)
        self.update = nn.Sequential(
            layer_init(nn.Linear(self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.5),
        )
        self.update_norm = nn.LayerNorm(self.D)
        self.readout = nn.Sequential(
            layer_init(nn.Linear(self.D, self.D)),
            ReLUSquared(),
            nn.LayerNorm(self.D),
        )
        self.register_buffer("utility_sum", torch.zeros(self.K))
        self.register_buffer("utility_count", torch.zeros(()))

    def _set_parameter(self, name, value):
        setattr(self, name, nn.Parameter(value.detach().clone()))

    def _resize_linear_out(self, layer, keep_idx=None, parent_idx=None, noise_scale=0.0):
        weight = layer.weight.data
        bias = layer.bias.data
        if keep_idx is not None:
            layer.out_features = len(keep_idx)
            layer.weight = nn.Parameter(weight[keep_idx].detach().clone())
            layer.bias = nn.Parameter(bias[keep_idx].detach().clone())
        if parent_idx is not None:
            new_weight = weight[parent_idx : parent_idx + 1].detach().clone()
            new_bias = bias[parent_idx : parent_idx + 1].detach().clone()
            if noise_scale > 0.0:
                new_weight = new_weight + noise_scale * torch.randn_like(new_weight)
                new_bias = new_bias + noise_scale * torch.randn_like(new_bias)
            layer.out_features = layer.out_features + 1
            layer.weight = nn.Parameter(torch.cat([layer.weight.data, new_weight], dim=0))
            layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_bias], dim=0))

    def _resize_cell_vector(self, param, keep_idx=None, parent_idx=None, noise_scale=0.0):
        data = param.data
        if keep_idx is not None:
            return data[keep_idx].detach().clone()
        child = data[parent_idx : parent_idx + 1].detach().clone()
        if noise_scale > 0.0:
            child = child + noise_scale * torch.randn_like(child)
        return torch.cat([data, child], dim=0)

    def _resize_cell_matrix(self, param, keep_idx=None, parent_idx=None, noise_scale=0.0):
        data = param.data
        if keep_idx is not None:
            return data[keep_idx][:, keep_idx].detach().clone()
        parent_row = data[parent_idx : parent_idx + 1].detach().clone()
        child_col = data[:, parent_idx : parent_idx + 1].detach().clone()
        child_self = data[parent_idx : parent_idx + 1, parent_idx : parent_idx + 1].detach().clone()
        if noise_scale > 0.0:
            parent_row = parent_row + noise_scale * torch.randn_like(parent_row)
            child_col = child_col + noise_scale * torch.randn_like(child_col)
            child_self = child_self + noise_scale * torch.randn_like(child_self)
        expanded = torch.cat([data, child_col], dim=1)
        child_row = torch.cat([parent_row, child_self], dim=1)
        return torch.cat([expanded, child_row], dim=0)

    def _reset_utility(self):
        self.utility_sum = torch.zeros(self.K, device=self.cell_seed.device)
        self.utility_count = torch.zeros((), device=self.cell_seed.device)

    def _accumulate_cell_utility(self, active, grad):
        with torch.no_grad():
            utility = (-grad * active).sum(dim=0)
            if not torch.isfinite(utility).all():
                utility = torch.where(torch.isfinite(utility), utility, torch.zeros_like(utility))
            self.utility_sum[: utility.numel()] += utility
            self.utility_count += active.shape[0]

    def _delete_cells(self, delete_idx):
        if not delete_idx:
            return 0
        keep = [i for i in range(self.K) if i not in set(delete_idx)]
        keep_idx = torch.tensor(keep, device=self.cell_seed.device, dtype=torch.long)
        self.K = len(keep)
        self._set_parameter("cell_seed", self._resize_cell_vector(self.cell_seed, keep_idx=keep_idx))
        self._set_parameter("cell_pos", self._resize_cell_vector(self.cell_pos, keep_idx=keep_idx))
        self._set_parameter("edge_bias", self._resize_cell_matrix(self.edge_bias, keep_idx=keep_idx))
        self._set_parameter("lookback_edge_bias", self._resize_cell_matrix(self.lookback_edge_bias, keep_idx=keep_idx))
        self._set_parameter("null_route_bias", self._resize_cell_vector(self.null_route_bias, keep_idx=keep_idx))
        self._set_parameter(
            "lookback_null_route_bias", self._resize_cell_vector(self.lookback_null_route_bias, keep_idx=keep_idx)
        )
        for layer in (self.growth, self.shrink, self.plasticity, self.edge_source, self.edge_target):
            self._resize_linear_out(layer, keep_idx=keep_idx)
        self._reset_utility()
        return len(delete_idx)

    def _grow_cell(self, parent_idx, noise_scale):
        self._set_parameter("cell_seed", self._resize_cell_vector(self.cell_seed, parent_idx=parent_idx, noise_scale=noise_scale))
        self._set_parameter("cell_pos", self._resize_cell_vector(self.cell_pos, parent_idx=parent_idx, noise_scale=noise_scale))
        self._set_parameter("edge_bias", self._resize_cell_matrix(self.edge_bias, parent_idx=parent_idx, noise_scale=noise_scale))
        self._set_parameter(
            "lookback_edge_bias",
            self._resize_cell_matrix(self.lookback_edge_bias, parent_idx=parent_idx, noise_scale=noise_scale),
        )
        self._set_parameter(
            "null_route_bias",
            self._resize_cell_vector(self.null_route_bias, parent_idx=parent_idx, noise_scale=noise_scale),
        )
        self._set_parameter(
            "lookback_null_route_bias",
            self._resize_cell_vector(self.lookback_null_route_bias, parent_idx=parent_idx, noise_scale=noise_scale),
        )
        for layer in (self.growth, self.shrink, self.plasticity, self.edge_source, self.edge_target):
            self._resize_linear_out(layer, parent_idx=parent_idx, noise_scale=noise_scale)
        self.K += 1
        self._reset_utility()
        return 1

    def apply_morphogenesis(self, args):
        cell_map = list(range(self.K))
        if self.utility_count.item() <= 0:
            return {"births": 0, "deaths": 0, "cells": self.K, "cell_map": cell_map}
        raw_utility = self.utility_sum / self.utility_count.clamp_min(1.0)
        if not torch.isfinite(raw_utility).all():
            self._reset_utility()
            return {"births": 0, "deaths": 0, "cells": self.K, "cell_map": cell_map}
        utility_scale = raw_utility.abs().mean()
        if not torch.isfinite(utility_scale) or utility_scale.item() < args.morph_utility_eps:
            self._reset_utility()
            return {"births": 0, "deaths": 0, "cells": self.K, "cell_map": cell_map}
        utility = raw_utility / utility_scale
        deaths = []
        if self.K > args.min_cells and args.morph_max_deaths_per_iter > 0:
            death_candidates = torch.nonzero(utility < args.morph_prune_threshold).flatten()
            if death_candidates.numel() > 0:
                ranked = death_candidates[torch.argsort(utility[death_candidates])]
                max_deaths = min(args.morph_max_deaths_per_iter, self.K - args.min_cells)
                deaths = ranked[:max_deaths].tolist()

        birth_parents = []
        birth_count = 0
        if args.morph_max_births_per_iter > 0:
            birth_candidates = torch.nonzero(utility > args.morph_grow_threshold).flatten()
            if birth_candidates.numel() > 0:
                ranked = birth_candidates[torch.argsort(utility[birth_candidates], descending=True)]
                birth_parents = ranked[: args.morph_max_births_per_iter].tolist()
        death_count = self._delete_cells(deaths)
        if deaths:
            death_set = set(deaths)
            cell_map = [old_idx for old_idx in cell_map if old_idx not in death_set]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(cell_map)}
        for parent in birth_parents:
            parent_idx = old_to_new.get(parent)
            if parent_idx is not None and parent_idx < self.K:
                birth_count += self._grow_cell(parent_idx, args.morph_child_noise)
                cell_map.append(parent)
        self._reset_utility()
        return {"births": birth_count, "deaths": death_count, "cells": self.K, "cell_map": cell_map}

    def forward(self, x):
        B = x.shape[0]
        base = self.input(x)
        query = self.query(base)
        dist2_query = (query[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)

        growth_pressure = self.growth(base)
        shrink_pressure = F.softplus(self.shrink(base))
        active_logits = growth_pressure - shrink_pressure - 0.25 * dist2_query
        active = torch.sigmoid(active_logits)
        if torch.is_grad_enabled() and active.requires_grad:
            active.register_hook(lambda grad, active_detached=active.detach(): self._accumulate_cell_utility(active_detached, grad))

        plasticity = torch.sigmoid(self.plasticity(base))
        edge_source = self.edge_source(base)
        edge_target = self.edge_target(base)
        persistence = torch.sigmoid(self.persistence(base))
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(self.temperature(base))
        budget = torch.sigmoid(self.budget(base))
        tick_gates = torch.sigmoid(self.tick(base) + self.tick_offsets)

        h = base[:, None, :] + self.cell_seed[None, :, :]
        coord_dist2 = (self.cell_pos[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)
        eye = torch.eye(self.K, device=x.device, dtype=x.dtype)
        nonself = 1.0 - eye
        geometry_logits = -coord_dist2[None, :, :] / temp[:, None, :]
        edge_logits = (
            geometry_logits
            + edge_source[:, :, None]
            + edge_target[:, None, :]
            + self.edge_bias[None, :, :]
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
                lookback_support_sum = tick_gate * (
                    active * lookback_support.sum(dim=-1)
                ).sum(dim=1)
                lookback_effective_support_sum = tick_gate * (
                    active * lookback_effective_support
                ).sum(dim=1)
                active_lookback_edges = active_lookback_edges + lookback_support_sum
                expected_active_lookback_edges = expected_active_lookback_edges + (
                    lookback_support_sum.detach()
                    + lookback_effective_support_sum
                    - lookback_effective_support_sum.detach()
                )
                lookback_route_entropy = -(
                    lookback_route * torch.log(lookback_route + 1e-8)
                ).sum(dim=-1)
                lookback_route_entropy = (lookback_route_entropy * read_weights).sum(dim=1)
                lookback_route_entropy_sum = lookback_route_entropy_sum + tick_gate * lookback_route_entropy
                lookback_route_entropy_count += history_count * self.K

            mixed = h + persistence[:, :, None] * (
                self.route_mix * msg + self.lookback_mix * lookback_msg
            )
            delta = self.update_norm(self.update(self.norm(mixed)))
            h = h + tick_gates[:, t, None, None] * update_scale * delta
            history.append(h)

        pooled = torch.sum(read_weights[:, :, None] * h, dim=1)
        out = self.readout(pooled)

        active_cells = active.sum(dim=1)
        active_ticks = tick_gates.sum(dim=1)
        active_edges = current_route_support_sum
        initial_edge_capacity = max(self.initial_K * (self.initial_K - 1), 1)
        active_edge_frac = active_edges / initial_edge_capacity
        expected_active_edges = current_compute_support_sum
        expected_active_edge_frac = expected_active_edges / initial_edge_capacity
        lookback_edge_capacity = max((self.T * (self.T - 1) // 2) * self.initial_K * self.initial_K, 1)
        active_lookback_edge_frac = active_lookback_edges / lookback_edge_capacity
        expected_active_lookback_edge_frac = expected_active_lookback_edges / lookback_edge_capacity
        route_entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1)
        route_entropy = (route_entropy * read_weights).sum(dim=1) / np.log(max(self.K - 1, 2))
        edge_entropy = route_entropy
        expected_edge_entropy = route_entropy
        lookback_edge_entropy = lookback_route_entropy_sum / np.log(max(lookback_route_entropy_count, 2))
        expected_lookback_edge_entropy = lookback_edge_entropy
        lookback_route_entropy = lookback_route_entropy_sum / np.log(max(lookback_route_entropy_count, 2))
        active_cell_frac = active_cells / self.initial_K
        active_tick_frac = active_ticks / self.T
        node_tick_compute = active_cell_frac * active_tick_frac
        edge_tick_compute = expected_active_edge_frac * active_tick_frac
        base_compute = (
            node_tick_compute + self.edge_compute_weight * edge_tick_compute
        ) / (1.0 + self.edge_compute_weight)
        edge_read_capacity = max(self.T * self.initial_K * (self.initial_K - 1), 1)
        lookback_edge_compute = expected_active_lookback_edges / edge_read_capacity
        compute = (0.5 + 0.5 * budget.squeeze(1)) * (
            base_compute + self.lookback_compute_weight * lookback_edge_compute
        )
        compute = compute.clamp(min=0.0)

        stats = {
            "compute": compute,
            "current_cells": x.new_full((B,), float(self.K)),
            "active_cells": active_cells,
            "active_cell_frac": active_cell_frac,
            "active_ticks": active_ticks,
            "active_edges": active_edges,
            "active_edge_frac": active_edge_frac,
            "expected_active_edges": expected_active_edges,
            "expected_active_edge_frac": expected_active_edge_frac,
            "active_lookback_edges": active_lookback_edges,
            "active_lookback_edge_frac": active_lookback_edge_frac,
            "expected_active_lookback_edges": expected_active_lookback_edges,
            "expected_active_lookback_edge_frac": expected_active_lookback_edge_frac,
            "edge_entropy": edge_entropy,
            "expected_edge_entropy": expected_edge_entropy,
            "edge_noise": x.new_empty(B, 0, self.K, self.K),
            "lookback_edge_entropy": lookback_edge_entropy,
            "expected_lookback_edge_entropy": expected_lookback_edge_entropy,
            "lookback_edge_noise": x.new_empty(B, 0, self.K, self.K),
            "route_entropy": route_entropy,
            "lookback_route_entropy": lookback_route_entropy,
            "growth_pressure": growth_pressure.mean(dim=1),
            "shrink_pressure": shrink_pressure.mean(dim=1),
            "plasticity": plasticity.mean(dim=1),
            "persistence": persistence.squeeze(1),
            "budget": budget.squeeze(1),
            "temperature": temp.squeeze(1),
        }
        return out, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor = MorphogenicSubstrate(obs_dim, args)
        self.critic = MorphogenicSubstrate(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)

    def get_value(self, x):
        critic_features, _ = self.critic(x)
        return self.critic_value(critic_features)

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def apply_morphogenesis(self, args):
        actor = self.actor.apply_morphogenesis(args)
        critic = self.critic.apply_morphogenesis(args)
        changed = actor["births"] or actor["deaths"] or critic["births"] or critic["deaths"]
        return {"actor": actor, "critic": critic, "changed": bool(changed)}

    def _actor_dist(self, actor_features):
        if not torch.isfinite(actor_features).all():
            raise FloatingPointError("actor substrate produced non-finite features before Beta concentration heads")
        alpha = 1.0 + F.softplus(self.actor_alpha(actor_features))
        beta = 1.0 + F.softplus(self.actor_beta(actor_features))
        if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
            raise FloatingPointError("actor Beta concentration heads produced non-finite alpha/beta")
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action

    def get_action_and_value(self, x, z=None):
        actor_features, actor_stats = self.actor(x)
        critic_features, critic_stats = self.critic(x)
        dist, to_action = self._actor_dist(actor_features)
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
            self.critic_value(critic_features),
            actor_stats["compute"],
            critic_stats["compute"],
            stats,
        )


def mean_stat(stats, group, name):
    return stats[group][name].detach().mean().item()


def assert_finite_module(module, context):
    for name, param in module.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"{context}: non-finite parameter {name}")


def assert_finite_optimizer_state(optimizer, context):
    for param_idx, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"{context}: non-finite optimizer state {key} for parameter index {param_idx}")


def assert_finite_grads(module, context):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"{context}: non-finite gradient {name}")


def _substrate_cell_map(param_name, morph_stats):
    for prefix in ("actor", "critic"):
        qualified_prefix = f"{prefix}."
        if not param_name.startswith(qualified_prefix):
            continue
        local_name = param_name[len(qualified_prefix) :]
        cell_map = morph_stats[prefix].get("cell_map")
        if cell_map is None:
            return None, None
        if local_name in MorphogenicSubstrate.CELL_VECTOR_NAMES:
            return cell_map, "vector"
        if local_name in MorphogenicSubstrate.CELL_MATRIX_NAMES:
            return cell_map, "matrix"
        parts = local_name.split(".")
        if len(parts) == 2 and parts[0] in MorphogenicSubstrate.CELL_OUTPUT_LAYERS:
            return cell_map, "vector"
    return None, None


def _migrate_optimizer_tensor_state(param_name, value, old_param, new_param, morph_stats):
    cell_map, cell_kind = _substrate_cell_map(param_name, morph_stats)
    if cell_map is not None and value.shape == old_param.shape:
        index = torch.tensor(cell_map, device=value.device, dtype=torch.long)
        if cell_kind == "vector":
            return value.index_select(0, index).detach().clone()
        if cell_kind == "matrix":
            return value.index_select(0, index).index_select(1, index).detach().clone()
    if value.shape == new_param.shape:
        return value.detach().clone()
    if value.ndim == 0:
        return value.detach().clone()
    return torch.zeros_like(new_param)


def rebuild_adam_with_morphed_state(agent, optimizer, old_named_params, morph_stats):
    new_optimizer = optimizer.__class__(agent.parameters(), **optimizer.defaults)
    new_optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
    new_named_params = dict(agent.named_parameters())
    for name, new_param in new_named_params.items():
        old_param = old_named_params.get(name)
        if old_param is None or old_param not in optimizer.state:
            continue
        old_state = optimizer.state[old_param]
        new_state = {}
        for key, value in old_state.items():
            if torch.is_tensor(value):
                migrated = _migrate_optimizer_tensor_state(name, value, old_param, new_param, morph_stats)
                if migrated.ndim == 0:
                    new_state[key] = migrated
                    continue
                if migrated.is_floating_point() and new_param.is_floating_point():
                    migrated = migrated.to(device=new_param.device, dtype=new_param.dtype)
                else:
                    migrated = migrated.to(device=new_param.device)
                new_state[key] = migrated
            else:
                new_state[key] = value
        new_optimizer.state[new_param] = new_state
    return new_optimizer


def assert_morphed_optimizer_state_aliases(agent, optimizer, morph_stats, context):
    named_params = dict(agent.named_parameters())
    for name, param in named_params.items():
        cell_map, cell_kind = _substrate_cell_map(name, morph_stats)
        if cell_map is None:
            continue
        first_by_old_cell = {}
        duplicate_pairs = []
        for new_idx, old_idx in enumerate(cell_map):
            if old_idx in first_by_old_cell:
                duplicate_pairs.append((first_by_old_cell[old_idx], new_idx))
            else:
                first_by_old_cell[old_idx] = new_idx
        if not duplicate_pairs:
            continue
        state = optimizer.state.get(param)
        if not state:
            raise AssertionError(f"{context}: missing optimizer state for morphed parameter {name}")
        for key, value in state.items():
            if not torch.is_tensor(value) or value.ndim == 0 or value.shape != param.shape:
                continue
            for parent_new_idx, child_new_idx in duplicate_pairs:
                if cell_kind == "vector":
                    parent_state = value[parent_new_idx]
                    child_state = value[child_new_idx]
                elif cell_kind == "matrix":
                    parent_state = value[parent_new_idx]
                    child_state = value[child_new_idx]
                else:
                    continue
                if not torch.allclose(parent_state, child_state, atol=0.0, rtol=0.0):
                    raise AssertionError(
                        f"{context}: Adam {key} for {name} did not duplicate parent state "
                        f"from row {parent_new_idx} to child row {child_new_idx}"
                    )


class _ContractEnv:
    def __init__(self, obs_dim=17, action_dim=6):
        self.single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)


def _contract_args(args):
    contract_args = Args()
    contract_args.seed = args.seed
    contract_args.cuda = True
    contract_args.torch_deterministic = True
    contract_args.learning_rate = args.learning_rate
    contract_args.initial_cells = 8
    contract_args.min_cells = 4
    contract_args.cell_dim = 32
    contract_args.field_dim = 4
    contract_args.max_ticks = 3
    contract_args.num_envs = 8
    contract_args.num_steps = 8
    contract_args.num_minibatches = 2
    contract_args.batch_size = 64
    contract_args.minibatch_size = 32
    contract_args.morph_max_births_per_iter = 2
    contract_args.morph_max_deaths_per_iter = 2
    contract_args.morph_child_noise = 0.01
    return contract_args


def _fixed_contract_batch(agent, obs, action_dim):
    z_base = torch.linspace(0.15, 0.85, action_dim, device=obs.device)
    z = z_base.unsqueeze(0).expand(obs.shape[0], action_dim).contiguous()
    with torch.no_grad():
        _, _, old_logprob, _, value, _, _, _ = agent.get_action_and_value(obs, z)
    advantages = torch.linspace(-1.0, 1.0, obs.shape[0], device=obs.device)
    returns = value.flatten().detach() + 0.25 * advantages
    return z, old_logprob.detach(), advantages, returns, value.flatten().detach()


def _ppo_like_contract_step(agent, optimizer, args, obs, z, old_logprob, advantages, returns, old_values, step_idx):
    _, _, newlogprob, entropy, newvalue, actor_compute, critic_compute, _ = agent.get_action_and_value(obs, z)
    if not torch.isfinite(newlogprob).all():
        raise FloatingPointError(f"contract step {step_idx}: non-finite logprob")
    if not torch.isfinite(entropy).all():
        raise FloatingPointError(f"contract step {step_idx}: non-finite entropy")
    logratio = newlogprob - old_logprob
    ratio = logratio.exp()
    mb_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_multiplier = agent.compute_multiplier(actor_compute, args)
    critic_multiplier = agent.compute_multiplier(critic_compute, args)

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
    pg_task_loss = (pg_loss_per_sample * actor_multiplier.detach()).mean()
    actor_loss_magnitude = pg_loss_per_sample.detach().abs() + args.actor_compute_loss_floor
    actor_compute_loss = (actor_loss_magnitude * actor_multiplier).mean()
    pg_loss = pg_task_loss + actor_compute_loss - actor_compute_loss.detach()

    newvalue = newvalue.view(-1)
    v_loss_unclipped = (newvalue - returns) ** 2
    v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * critic_multiplier).mean()
    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"contract step {step_idx}: non-finite loss")

    optimizer.zero_grad()
    loss.backward()
    assert_finite_grads(agent, f"contract step {step_idx}")
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"contract step {step_idx}: non-finite clipped gradient norm")
    optimizer.step()
    assert_finite_module(agent, f"contract step {step_idx}")
    assert_finite_optimizer_state(optimizer, f"contract step {step_idx}")


def _force_contract_morphology(substrate):
    utility = torch.tensor([-2.0, -1.5, -0.1, 0.0, 0.1, 0.5, 1.5, 2.0], device=substrate.utility_sum.device)
    if substrate.K != utility.numel():
        raise AssertionError(f"contract expected {utility.numel()} cells before morphogenesis, found {substrate.K}")
    substrate.utility_sum.copy_(utility)
    substrate.utility_count.fill_(64.0)


def run_cuda_contract(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA contract requested but CUDA is not available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    contract_args = _contract_args(args)
    device = torch.device("cuda")
    envs = _ContractEnv()
    agent = Agent(envs, contract_args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=contract_args.learning_rate, eps=1e-5)
    obs = torch.randn(contract_args.batch_size, envs.single_observation_space.shape[0], device=device)
    z, old_logprob, advantages, returns, old_values = _fixed_contract_batch(
        agent, obs, envs.single_action_space.shape[0]
    )

    _ppo_like_contract_step(agent, optimizer, contract_args, obs, z, old_logprob, advantages, returns, old_values, 0)
    _force_contract_morphology(agent.actor)
    _force_contract_morphology(agent.critic)
    old_named_params = dict(agent.named_parameters())
    morph_stats = agent.apply_morphogenesis(contract_args)
    if morph_stats["actor"]["births"] != 2 or morph_stats["actor"]["deaths"] != 2:
        raise AssertionError(f"contract actor morphogenesis did not grow/delete as expected: {morph_stats['actor']}")
    if morph_stats["critic"]["births"] != 2 or morph_stats["critic"]["deaths"] != 2:
        raise AssertionError(f"contract critic morphogenesis did not grow/delete as expected: {morph_stats['critic']}")
    optimizer = rebuild_adam_with_morphed_state(agent, optimizer, old_named_params, morph_stats)
    assert_finite_module(agent, "contract morphogenesis")
    assert_finite_optimizer_state(optimizer, "contract morphogenesis")
    assert_morphed_optimizer_state_aliases(agent, optimizer, morph_stats, "contract morphogenesis")

    z, old_logprob, advantages, returns, old_values = _fixed_contract_batch(agent, obs, envs.single_action_space.shape[0])
    for step_idx in range(1, 6):
        _ppo_like_contract_step(
            agent, optimizer, contract_args, obs, z, old_logprob, advantages, returns, old_values, step_idx
        )
    print("CUDA morphcompute_v12 contract passed")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.cuda_contract:
        run_cuda_contract(args)
        raise SystemExit(0)
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    last_stats = None

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
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

            # TRY NOT TO MODIFY: execute the game and log data.
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

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_multiplier = agent.compute_multiplier(actor_compute, args)
                critic_multiplier = agent.compute_multiplier(critic_compute, args)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_task_loss = (pg_loss_per_sample * actor_multiplier.detach()).mean()
                actor_loss_magnitude = pg_loss_per_sample.detach().abs() + args.actor_compute_loss_floor
                actor_compute_loss = (actor_loss_magnitude * actor_multiplier).mean()
                pg_loss = pg_task_loss + actor_compute_loss - actor_compute_loss.detach()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * (v_loss_max * critic_multiplier).mean()
                else:
                    v_loss_per_sample = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (v_loss_per_sample * critic_multiplier).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became non-finite before backward")
                optimizer.zero_grad()
                loss.backward()
                assert_finite_grads(agent, "PPO backward")
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                assert_finite_module(agent, "optimizer step")
                assert_finite_optimizer_state(optimizer, "optimizer step")

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
            writer.add_scalar("morph/actor_current_cells", mean_stat(last_stats, "actor", "current_cells"), global_step)
            writer.add_scalar("morph/critic_current_cells", mean_stat(last_stats, "critic", "current_cells"), global_step)
            writer.add_scalar("morph/actor_active_cells", mean_stat(last_stats, "actor", "active_cells"), global_step)
            writer.add_scalar("morph/critic_active_cells", mean_stat(last_stats, "critic", "active_cells"), global_step)
            writer.add_scalar("morph/actor_active_cell_frac", mean_stat(last_stats, "actor", "active_cell_frac"), global_step)
            writer.add_scalar("morph/critic_active_cell_frac", mean_stat(last_stats, "critic", "active_cell_frac"), global_step)
            writer.add_scalar("morph/actor_active_ticks", mean_stat(last_stats, "actor", "active_ticks"), global_step)
            writer.add_scalar("morph/critic_active_ticks", mean_stat(last_stats, "critic", "active_ticks"), global_step)
            writer.add_scalar("morph/actor_active_tick_frac", mean_stat(last_stats, "actor", "active_ticks") / args.max_ticks, global_step)
            writer.add_scalar("morph/critic_active_tick_frac", mean_stat(last_stats, "critic", "active_ticks") / args.max_ticks, global_step)
            writer.add_scalar("morph/actor_active_edges", mean_stat(last_stats, "actor", "active_edges"), global_step)
            writer.add_scalar("morph/critic_active_edges", mean_stat(last_stats, "critic", "active_edges"), global_step)
            writer.add_scalar("morph/actor_active_edge_frac", mean_stat(last_stats, "actor", "active_edge_frac"), global_step)
            writer.add_scalar("morph/critic_active_edge_frac", mean_stat(last_stats, "critic", "active_edge_frac"), global_step)
            writer.add_scalar("morph/actor_expected_active_edges", mean_stat(last_stats, "actor", "expected_active_edges"), global_step)
            writer.add_scalar("morph/critic_expected_active_edges", mean_stat(last_stats, "critic", "expected_active_edges"), global_step)
            writer.add_scalar("morph/actor_expected_active_edge_frac", mean_stat(last_stats, "actor", "expected_active_edge_frac"), global_step)
            writer.add_scalar("morph/critic_expected_active_edge_frac", mean_stat(last_stats, "critic", "expected_active_edge_frac"), global_step)
            writer.add_scalar("morph/actor_active_lookback_edges", mean_stat(last_stats, "actor", "active_lookback_edges"), global_step)
            writer.add_scalar("morph/critic_active_lookback_edges", mean_stat(last_stats, "critic", "active_lookback_edges"), global_step)
            writer.add_scalar("morph/actor_active_lookback_edge_frac", mean_stat(last_stats, "actor", "active_lookback_edge_frac"), global_step)
            writer.add_scalar("morph/critic_active_lookback_edge_frac", mean_stat(last_stats, "critic", "active_lookback_edge_frac"), global_step)
            writer.add_scalar("morph/actor_expected_active_lookback_edges", mean_stat(last_stats, "actor", "expected_active_lookback_edges"), global_step)
            writer.add_scalar("morph/critic_expected_active_lookback_edges", mean_stat(last_stats, "critic", "expected_active_lookback_edges"), global_step)
            writer.add_scalar("morph/actor_expected_active_lookback_edge_frac", mean_stat(last_stats, "actor", "expected_active_lookback_edge_frac"), global_step)
            writer.add_scalar("morph/critic_expected_active_lookback_edge_frac", mean_stat(last_stats, "critic", "expected_active_lookback_edge_frac"), global_step)
            writer.add_scalar("morph/actor_edge_entropy", mean_stat(last_stats, "actor", "edge_entropy"), global_step)
            writer.add_scalar("morph/critic_edge_entropy", mean_stat(last_stats, "critic", "edge_entropy"), global_step)
            writer.add_scalar("morph/actor_expected_edge_entropy", mean_stat(last_stats, "actor", "expected_edge_entropy"), global_step)
            writer.add_scalar("morph/critic_expected_edge_entropy", mean_stat(last_stats, "critic", "expected_edge_entropy"), global_step)
            writer.add_scalar("morph/actor_lookback_edge_entropy", mean_stat(last_stats, "actor", "lookback_edge_entropy"), global_step)
            writer.add_scalar("morph/critic_lookback_edge_entropy", mean_stat(last_stats, "critic", "lookback_edge_entropy"), global_step)
            writer.add_scalar("morph/actor_expected_lookback_edge_entropy", mean_stat(last_stats, "actor", "expected_lookback_edge_entropy"), global_step)
            writer.add_scalar("morph/critic_expected_lookback_edge_entropy", mean_stat(last_stats, "critic", "expected_lookback_edge_entropy"), global_step)
            writer.add_scalar("morph/actor_route_entropy", mean_stat(last_stats, "actor", "route_entropy"), global_step)
            writer.add_scalar("morph/critic_route_entropy", mean_stat(last_stats, "critic", "route_entropy"), global_step)
            writer.add_scalar("morph/actor_lookback_route_entropy", mean_stat(last_stats, "actor", "lookback_route_entropy"), global_step)
            writer.add_scalar("morph/critic_lookback_route_entropy", mean_stat(last_stats, "critic", "lookback_route_entropy"), global_step)
            writer.add_scalar("morph/growth_pressure_mean", mean_stat(last_stats, "actor", "growth_pressure"), global_step)
            writer.add_scalar("morph/shrink_pressure_mean", mean_stat(last_stats, "actor", "shrink_pressure"), global_step)
            writer.add_scalar("morph/plasticity_mean", mean_stat(last_stats, "actor", "plasticity"), global_step)
            writer.add_scalar("morph/persistence_mean", mean_stat(last_stats, "actor", "persistence"), global_step)
            writer.add_scalar("morph/budget_mean", mean_stat(last_stats, "actor", "budget"), global_step)
            writer.add_scalar("morph/temperature_mean", mean_stat(last_stats, "actor", "temperature"), global_step)
            writer.add_scalar("morph/compute_multiplier_actor", actor_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/compute_multiplier_critic", critic_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/actor_compute_loss_magnitude", actor_loss_magnitude.mean().item(), global_step)
        if iteration > args.morph_warmup_iterations:
            old_named_params = dict(agent.named_parameters())
            morph_stats = agent.apply_morphogenesis(args)
            writer.add_scalar("morph/actor_births", morph_stats["actor"]["births"], global_step)
            writer.add_scalar("morph/actor_deaths", morph_stats["actor"]["deaths"], global_step)
            writer.add_scalar("morph/actor_cells", morph_stats["actor"]["cells"], global_step)
            writer.add_scalar("morph/critic_births", morph_stats["critic"]["births"], global_step)
            writer.add_scalar("morph/critic_deaths", morph_stats["critic"]["deaths"], global_step)
            writer.add_scalar("morph/critic_cells", morph_stats["critic"]["cells"], global_step)
            if morph_stats["changed"]:
                optimizer = rebuild_adam_with_morphed_state(agent, optimizer, old_named_params, morph_stats)
                assert_finite_module(agent, "morphogenesis")
                assert_finite_optimizer_state(optimizer, "morphogenesis")
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
