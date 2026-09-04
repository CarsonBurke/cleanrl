# TPO-MD ThinkTrunk primitive-DAG predictive-coding GEM v3.
#
# The deployed policy is exactly pure TPO-v5's shared Beta ThinkTrunk.  Its real
# primitive DAG is lifted into local activities for every learned affine output,
# residual gate, residual mix, branch pre/output, MoE gate, block output, shared
# trunk feature, actor raw, and critic logit.  Actor and critic terminal messages
# meet in one shared trunk graph and every fanout message is summed.
#
# One immutable rollout q and TD(lambda) label feed ten outer E/M cycles.  Each E
# step is ten detached reverse Gauss-Seidel sweeps with analytic local messages;
# each M step is a fresh whole-rollout float64 Moore-Penrose residual correction.
# Joint indexed-transfer solves preserve current/history covariance and every
# parameter correction is captured before one atomic mutation.

import os
import random
import time
from dataclasses import dataclass
from math import log
from typing import Callable, NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport


SAMPLE_EPS = 1e-6
PC_INFERENCE_STEPS = 10
OUTER_CYCLES = 10


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_reward: bool = False
    clip_reward: bool = False

    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    hidden: int = 64
    share_backbone: bool = True
    k_blocks: int = 3
    n_experts: int = 16
    pc_inference_steps: int = PC_INFERENCE_STEPS
    pc_chunk_size: int = 512
    outer_cycles: int = OUTER_CYCLES
    tpo_coef: float = 1.0
    vf_coef: float = 1.0
    ent_coef: float = 0.0
    auto_entropy: bool = False

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_symlog: bool = True
    value_sigma_to_bin_ratio: float = 0.5

    tpo_k: int = 8
    tpo_sigma_scale_coef: float = 1.0
    tpo_eps: float = 0.03
    tpo_adaptive_eta: bool = True
    tpo_dyn_trust: bool = True
    tpo_eta_base: float = 1.0
    tpo_eta: float = 6.0
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


def rescale(tensor: torch.Tensor, old_range: tuple[float, float], new_range: tuple[float, float]):
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def value_support_bounds(args: Args) -> tuple[float, float]:
    return args.v_min, args.v_max


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(
    env_id: str,
    index: int,
    capture_video: bool,
    run_name: str,
    gamma: float,
    normalize_reward: bool,
    clip_reward: bool,
):
    def thunk():
        if capture_video and index == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        clipped_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape, -10.0, dtype=env.observation_space.dtype),
            high=np.full(env.observation_space.shape, 10.0, dtype=env.observation_space.dtype),
            dtype=env.observation_space.dtype,
        )
        try:
            env = gym.wrappers.TransformObservation(
                env,
                lambda observation: np.clip(observation, -10, 10),
                observation_space=clipped_space,
            )
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda observation: np.clip(observation, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def _require_finite(label: str, *values: torch.Tensor) -> None:
    finite = torch.stack([torch.isfinite(value).all() for value in values]).all()
    if not bool(finite):
        raise FloatingPointError(f"non-finite {label}; refusing the PC projection")


class ReLUSquared(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).pow(2)


class IndexedTransferBranch(nn.Module):
    def __init__(self, H: int, history_dim: int):
        super().__init__()
        if history_dim % H != 0:
            raise ValueError(f"history_dim={history_dim} must be divisible by H={H}")
        self.H = H
        self.history_slots = history_dim // H
        self.current_linear = layer_init(nn.Linear(H, H))
        self.act = ReLUSquared()
        self.out_linear = layer_init(nn.Linear(H, H))
        self.history_weight = nn.Parameter(torch.empty(self.history_slots, H))
        nn.init.normal_(
            self.history_weight,
            mean=0.0,
            std=np.sqrt(2.0 / (H + self.history_slots)),
        )

    def forward(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        preact = self.current_linear(x)
        history = history.reshape(history.shape[0], self.history_slots, self.H)
        same_index_transfer = (
            history * self.history_weight.to(dtype=history.dtype).unsqueeze(0)
        ).sum(dim=1)
        return self.out_linear(self.act(preact + same_index_transfer))


class ThinkBlock(nn.Module):
    """Pure-TPO-v5 bounded residual plus dense and soft-MoE block."""

    def __init__(self, in_dim: int, H: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = IndexedTransferBranch(H, in_dim)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList(
            [IndexedTransferBranch(H, in_dim) for _ in range(n_experts)]
        )

    def forward(self, cat_feats: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in), cat_feats)
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack(
            [expert(m_in, cat_feats) for expert in self.experts], dim=1
        )
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim: int, H: int, K: int, n_experts: int):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for block_index in range(K):
            block_in_dim = H * (block_index + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    """Exact pure-TPO-v5 deployed model; the PC graph adds no learned state."""

    def __init__(self, envs, args: Args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        hidden = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = ThinkTrunk(
                obs_dim, hidden, args.k_blocks, args.n_experts
            )
        else:
            self.critic_trunk = ThinkTrunk(
                obs_dim, hidden, args.k_blocks, args.n_experts
            )
            self.actor_trunk = ThinkTrunk(
                obs_dim, hidden, args.k_blocks, args.n_experts
            )
        self.num_bins = args.num_bins
        self.critic_head = layer_init(
            nn.Linear(hidden, args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            self.actor_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(
                nn.Linear(hidden, act_dim), std=0.01
            )
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            self.actor_alpha_head = layer_init(
                nn.Linear(hidden, act_dim), std=0.01
            )
            self.actor_beta_head = layer_init(
                nn.Linear(hidden, act_dim), std=0.01
            )
            self.register_buffer(
                "action_low",
                torch.tensor(envs.single_action_space.low, dtype=torch.float32),
            )
            self.register_buffer(
                "action_high",
                torch.tensor(envs.single_action_space.high, dtype=torch.float32),
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _actor_dist(self, actor_feat: torch.Tensor):
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_log_variance = self.actor_logvar_head(actor_feat)
            log_variance = rescale(
                (raw_log_variance / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            distribution = Normal(mean, (0.5 * log_variance).exp())
            to_action = torch.tanh
            log_det = lambda latent: 2.0 * (
                log(2.0) - latent - F.softplus(-2.0 * latent)
            )
            return distribution, to_action, log_det
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        distribution = Beta(alpha, beta)
        to_action = lambda latent: self.action_low + (
            self.action_high - self.action_low
        ) * latent
        log_det = lambda latent: 0.0
        return distribution, to_action, log_det

    def _trunks(self, observations: torch.Tensor):
        if self.share_backbone:
            feature = self.trunk(observations)
            return feature, feature
        return self.actor_trunk(observations), self.critic_trunk(observations)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        _, critic_feature = self._trunks(observations)
        return self.critic_head(critic_feature)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        candidate_zs: Optional[torch.Tensor] = None,
        return_dist: bool = False,
    ):
        actor_feature, critic_feature = self._trunks(observations)
        distribution, to_action, log_det = self._actor_dist(actor_feature)
        if z is None:
            z = distribution.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_probability = (distribution.log_prob(z) - log_det(z)).sum(1)
        value_logits = self.critic_head(critic_feature)
        if self.actor_dist == "gaussian":
            entropy_latent = distribution.rsample()
            entropy = (
                distribution.log_prob(entropy_latent) - log_det(entropy_latent)
            ).sum(1).neg()
        else:
            entropy = distribution.entropy().sum(1)
        output = (action, z, log_probability, entropy, value_logits)
        if candidate_zs is not None:
            candidates = candidate_zs.transpose(0, 1)
            candidate_logprobs = (
                distribution.log_prob(candidates) - log_det(candidates)
            ).sum(-1).transpose(0, 1)
            output += (candidate_logprobs,)
        if return_dist:
            output += (distribution, to_action, log_det)
        return output

    def actor_parameters(self) -> list[nn.Parameter]:
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(
                self.actor_logvar_head.parameters()
            )
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(
                self.actor_beta_head.parameters()
            )
        return list(trunk.parameters()) + heads

    def critic_parameters(self) -> list[nn.Parameter]:
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())

    def task_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())


class DAGActivities(NamedTuple):
    main: torch.Tensor
    projection: torch.Tensor
    residual_logit: torch.Tensor
    x_in: torch.Tensor
    branch_pre: torch.Tensor
    branch_out: torch.Tensor
    moe_logits: torch.Tensor
    trunk: torch.Tensor
    actor_alpha_raw: torch.Tensor
    actor_beta_raw: torch.Tensor
    critic_logits: torch.Tensor


class DAGSettleResult(NamedTuple):
    activities: DAGActivities
    energies: torch.Tensor
    stationarity_rms: torch.Tensor
    actor_boundary_energies: torch.Tensor
    critic_boundary_energies: torch.Tensor


def _branch_modules(block: ThinkBlock) -> tuple[IndexedTransferBranch, ...]:
    return (block.dense, *tuple(block.experts))


def _indexed_preact(
    branch: IndexedTransferBranch,
    current: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    history_slots = history.reshape(
        history.shape[0], branch.history_slots, branch.H
    )
    transfer = (
        history_slots
        * branch.history_weight.to(dtype=history.dtype).unsqueeze(0)
    ).sum(dim=1)
    return branch.current_linear(current) + transfer


def free_dag_activities(agent: Agent, observations: torch.Tensor) -> DAGActivities:
    """Topological free trace of the exact deployed shared ThinkTrunk."""

    if not agent.share_backbone or agent.actor_dist != "beta":
        raise ValueError("the primitive PC graph requires the shared Beta ThinkTrunk")
    main = [agent.trunk.entry(observations)]
    projections, residual_logits, mixed_states = [], [], []
    branch_pres, branch_outs, moe_logits = [], [], []
    for block in agent.trunk.blocks:
        history = torch.cat(main, dim=-1)
        projection = block.in_proj(history)
        residual_logit = block.resid_gate.unsqueeze(0).expand_as(projection)
        gate = torch.sigmoid(residual_logit)
        mixed = gate * projection + (1.0 - gate) * main[0]

        dense_current = block.dense_norm(mixed)
        moe_current = block.moe_norm(mixed)
        logits = block.gate(moe_current)
        weights = torch.softmax(logits, dim=-1)
        pres = [_indexed_preact(block.dense, dense_current, history)]
        pres.extend(
            _indexed_preact(expert, moe_current, history)
            for expert in block.experts
        )
        outs = [
            branch.out_linear(torch.relu(pre).pow(2))
            for branch, pre in zip(_branch_modules(block), pres, strict=True)
        ]
        packed_out = torch.stack(outs, dim=1)
        block_output = (
            mixed
            + packed_out[:, 0]
            + (weights.unsqueeze(-1) * packed_out[:, 1:]).sum(dim=1)
        )
        projections.append(projection)
        residual_logits.append(residual_logit)
        mixed_states.append(mixed)
        branch_pres.append(torch.stack(pres, dim=1))
        branch_outs.append(packed_out)
        moe_logits.append(logits)
        main.append(block_output)
    packed_main = torch.stack(main, dim=1)
    trunk_feature = agent.trunk.out_proj(
        agent.trunk.out_norm(packed_main.flatten(1))
    )
    return DAGActivities(
        packed_main,
        torch.stack(projections, dim=1),
        torch.stack(residual_logits, dim=1),
        torch.stack(mixed_states, dim=1),
        torch.stack(branch_pres, dim=1),
        torch.stack(branch_outs, dim=1),
        torch.stack(moe_logits, dim=1),
        trunk_feature,
        agent.actor_alpha_head(trunk_feature),
        agent.actor_beta_head(trunk_feature),
        agent.critic_head(trunk_feature),
    )


def dag_predictions(
    agent: Agent,
    observations: torch.Tensor,
    activities: DAGActivities,
) -> DAGActivities:
    main_predictions = [agent.trunk.entry(observations)]
    projection_predictions, residual_predictions, mixed_predictions = [], [], []
    pre_predictions, out_predictions, logit_predictions = [], [], []
    for block_index, block in enumerate(agent.trunk.blocks):
        history = activities.main[:, : block_index + 1].flatten(1)
        projection = agent.trunk.blocks[block_index].in_proj(history)
        residual_logit = block.resid_gate.unsqueeze(0).expand_as(projection)
        gate = torch.sigmoid(activities.residual_logit[:, block_index])
        mixed = (
            gate * activities.projection[:, block_index]
            + (1.0 - gate) * activities.main[:, 0]
        )
        dense_current = block.dense_norm(activities.x_in[:, block_index])
        moe_current = block.moe_norm(activities.x_in[:, block_index])
        branches = _branch_modules(block)
        pres = [_indexed_preact(block.dense, dense_current, history)]
        pres.extend(
            _indexed_preact(expert, moe_current, history)
            for expert in block.experts
        )
        outs = [
            branch.out_linear(
                torch.relu(activities.branch_pre[:, block_index, branch_index]).pow(2)
            )
            for branch_index, branch in enumerate(branches)
        ]
        logits = block.gate(moe_current)
        weights = torch.softmax(activities.moe_logits[:, block_index], dim=-1)
        block_output = (
            activities.x_in[:, block_index]
            + activities.branch_out[:, block_index, 0]
            + (
                weights.unsqueeze(-1)
                * activities.branch_out[:, block_index, 1:]
            ).sum(dim=1)
        )
        main_predictions.append(block_output)
        projection_predictions.append(projection)
        residual_predictions.append(residual_logit)
        mixed_predictions.append(mixed)
        pre_predictions.append(torch.stack(pres, dim=1))
        out_predictions.append(torch.stack(outs, dim=1))
        logit_predictions.append(logits)
    trunk_prediction = agent.trunk.out_proj(
        agent.trunk.out_norm(activities.main.flatten(1))
    )
    return DAGActivities(
        torch.stack(main_predictions, dim=1),
        torch.stack(projection_predictions, dim=1),
        torch.stack(residual_predictions, dim=1),
        torch.stack(mixed_predictions, dim=1),
        torch.stack(pre_predictions, dim=1),
        torch.stack(out_predictions, dim=1),
        torch.stack(logit_predictions, dim=1),
        trunk_prediction,
        agent.actor_alpha_head(activities.trunk),
        agent.actor_beta_head(activities.trunk),
        agent.critic_head(activities.trunk),
    )


def dag_residuals(
    activities: DAGActivities, predictions: DAGActivities
) -> DAGActivities:
    return DAGActivities(
        *(activity - prediction for activity, prediction in zip(activities, predictions, strict=True))
    )


def dag_model_energy(residuals: DAGActivities) -> torch.Tensor:
    return 0.5 * sum(residual.square().sum() for residual in residuals)


def _rms_epsilon(norm: nn.RMSNorm, tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps if norm.eps is None else norm.eps


def rmsnorm_vjp(
    tensor: torch.Tensor, cotangent: torch.Tensor, epsilon: float
) -> torch.Tensor:
    dimension = tensor.shape[-1]
    scale = (tensor.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    radial = (tensor * cotangent).sum(dim=-1, keepdim=True)
    return scale * cotangent - scale.pow(3) * tensor * radial / dimension


def rmsnorm_empirical_gram(
    tensor: torch.Tensor, matrix: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Mean J_rms A J_rms without materializing a row-wise D by D by D tensor."""

    dimension = tensor.shape[-1]
    scale = (tensor.square().mean(dim=-1) + epsilon).rsqrt()
    radial_coefficient = scale.pow(3) / dimension
    matrix_tensor = tensor @ matrix.T
    cross_coefficient = scale * radial_coefficient
    radial_quadratic = (tensor * matrix_tensor).sum(-1)
    result = scale.square().mean() * matrix
    result = result - torch.einsum(
        "b,bi,bj->ij", cross_coefficient, tensor, matrix_tensor
    ) / tensor.shape[0]
    result = result - torch.einsum(
        "b,bi,bj->ij", cross_coefficient, matrix_tensor, tensor
    ) / tensor.shape[0]
    result = result + torch.einsum(
        "b,b,bi,bj->ij",
        radial_coefficient.square(),
        radial_quadratic,
        tensor,
        tensor,
    ) / tensor.shape[0]
    return result


def rmsnorm_slice_empirical_gram(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    start: int,
    width: int,
    epsilon: float,
) -> torch.Tensor:
    """One input-slot diagonal block of mean J_rms A J_rms."""

    dimension = tensor.shape[-1]
    scale = (tensor.square().mean(dim=-1) + epsilon).rsqrt()
    radial_coefficient = scale.pow(3) / dimension
    matrix_tensor = tensor @ matrix.T
    tensor_slice = tensor[:, start : start + width]
    matrix_slice = matrix_tensor[:, start : start + width]
    matrix_block = matrix[start : start + width, start : start + width]
    cross_coefficient = scale * radial_coefficient
    radial_quadratic = (tensor * matrix_tensor).sum(-1)
    result = scale.square().mean() * matrix_block
    result = result - torch.einsum(
        "b,bi,bj->ij", cross_coefficient, tensor_slice, matrix_slice
    ) / tensor.shape[0]
    result = result - torch.einsum(
        "b,bi,bj->ij", cross_coefficient, matrix_slice, tensor_slice
    ) / tensor.shape[0]
    result = result + torch.einsum(
        "b,b,bi,bj->ij",
        radial_coefficient.square(),
        radial_quadratic,
        tensor_slice,
        tensor_slice,
    ) / tensor.shape[0]
    return result


def rmsnorm_row_gram(
    tensor: torch.Tensor, matrix: torch.Tensor, epsilon: float
) -> torch.Tensor:
    dimension = tensor.shape[-1]
    scale = (tensor.square().mean(dim=-1) + epsilon).rsqrt()
    radial_coefficient = scale.pow(3) / dimension
    matrix_tensor = tensor @ matrix.T
    radial_quadratic = (tensor * matrix_tensor).sum(-1)
    result = scale.square()[:, None, None] * matrix.unsqueeze(0)
    result = result - torch.einsum(
        "b,bi,bj->bij", scale * radial_coefficient, tensor, matrix_tensor
    )
    result = result - torch.einsum(
        "b,bi,bj->bij", scale * radial_coefficient, matrix_tensor, tensor
    )
    result = result + torch.einsum(
        "b,b,bi,bj->bij",
        radial_coefficient.square(),
        radial_quadratic,
        tensor,
        tensor,
    )
    return result


def rmsnorm_slice_row_gram(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    start: int,
    width: int,
    epsilon: float,
) -> torch.Tensor:
    dimension = tensor.shape[-1]
    scale = (tensor.square().mean(dim=-1) + epsilon).rsqrt()
    radial_coefficient = scale.pow(3) / dimension
    matrix_tensor = tensor @ matrix.T
    tensor_slice = tensor[:, start : start + width]
    matrix_slice = matrix_tensor[:, start : start + width]
    matrix_block = matrix[start : start + width, start : start + width]
    radial_quadratic = (tensor * matrix_tensor).sum(-1)
    result = scale.square()[:, None, None] * matrix_block.unsqueeze(0)
    result = result - torch.einsum(
        "b,bi,bj->bij",
        scale * radial_coefficient,
        tensor_slice,
        matrix_slice,
    )
    result = result - torch.einsum(
        "b,bi,bj->bij",
        scale * radial_coefficient,
        matrix_slice,
        tensor_slice,
    )
    result = result + torch.einsum(
        "b,b,bi,bj->bij",
        radial_coefficient.square(),
        radial_quadratic,
        tensor_slice,
        tensor_slice,
    )
    return result


def relu_squared_derivative(tensor: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.relu(tensor)


def solve_rank_one_identity(
    gradient: torch.Tensor, direction: torch.Tensor, solve_dimension: int
) -> torch.Tensor:
    direction64 = direction.to(torch.float64)
    gradient64 = gradient.to(torch.float64)
    numerator = (direction64 * gradient64).sum(
        dim=solve_dimension, keepdim=True
    )
    denominator = 1.0 + direction64.square().sum(
        dim=solve_dimension, keepdim=True
    )
    return (
        gradient64 - direction64 * (numerator / denominator)
    ).to(gradient.dtype)


def solve_categorical_identity(
    probabilities: torch.Tensor,
    gradient: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    diagonal = 1.0 + coefficient * probabilities
    inverse_gradient = gradient / diagonal
    inverse_probability = probabilities / diagonal
    numerator = (probabilities * inverse_gradient).sum(-1, keepdim=True)
    denominator = 1.0 - coefficient * (
        probabilities * inverse_probability
    ).sum(-1, keepdim=True)
    return inverse_gradient + coefficient * inverse_probability * (
        numerator / denominator
    )


def _shared_spd_solve(
    matrix: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    solve_dtype = (
        torch.float64
        if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else matrix.dtype
    )
    solve_matrix = matrix.to(solve_dtype)
    solve_gradient = gradient.to(solve_dtype)
    cholesky, info = torch.linalg.cholesky_ex(
        solve_matrix, check_errors=False
    )
    torch._assert_async(
        info.eq(0).all(), "non-positive exact shared GN system"
    )
    if matrix.ndim == 2:
        result = torch.cholesky_solve(solve_gradient.T, cholesky).T
    else:
        result = torch.cholesky_solve(
            solve_gradient.permute(1, 2, 0), cholesky
        ).permute(2, 0, 1)
    return result.to(gradient.dtype)


def beta_candidate_logits(
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    candidate_zs: torch.Tensor,
) -> torch.Tensor:
    alpha = 1.0 + F.softplus(alpha_raw)
    beta = 1.0 + F.softplus(beta_raw)
    candidates = candidate_zs.transpose(0, 1)
    distribution = Beta(alpha, beta, validate_args=False)
    return distribution.log_prob(candidates).sum(-1).transpose(0, 1)


def beta_score_jacobian(
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    candidate_zs: torch.Tensor,
) -> torch.Tensor:
    alpha = 1.0 + F.softplus(alpha_raw)
    beta = 1.0 + F.softplus(beta_raw)
    candidates = candidate_zs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    shared = torch.digamma(alpha + beta).unsqueeze(1)
    alpha_jacobian = torch.sigmoid(alpha_raw).unsqueeze(1) * (
        candidates.log() - torch.digamma(alpha).unsqueeze(1) + shared
    )
    beta_jacobian = torch.sigmoid(beta_raw).unsqueeze(1) * (
        torch.log1p(-candidates) - torch.digamma(beta).unsqueeze(1) + shared
    )
    return torch.cat((alpha_jacobian, beta_jacobian), dim=-1)


def categorical_gn_from_jacobian(
    probabilities: torch.Tensor, jacobian: torch.Tensor
) -> torch.Tensor:
    mean = (probabilities.unsqueeze(-1) * jacobian).sum(dim=1, keepdim=True)
    centered = jacobian - mean
    return torch.einsum("bk,bki,bkj->bij", probabilities, centered, centered)


def actor_boundary_gradient_and_gn(
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = beta_candidate_logits(alpha_raw, beta_raw, candidate_zs)
    probabilities = torch.softmax(logits, dim=-1)
    jacobian = beta_score_jacobian(alpha_raw, beta_raw, candidate_zs)
    gradient = torch.einsum("bk,bkd->bd", probabilities - targets, jacobian)
    return gradient, categorical_gn_from_jacobian(
        probabilities.to(torch.float64), jacobian.to(torch.float64)
    )


def actor_boundary_gradient(
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = beta_candidate_logits(alpha_raw, beta_raw, candidate_zs)
    probabilities = torch.softmax(logits, dim=-1)
    jacobian = beta_score_jacobian(alpha_raw, beta_raw, candidate_zs)
    return torch.einsum("bk,bkd->bd", probabilities - targets, jacobian)


def actor_boundary_energy(
    alpha_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    candidate_zs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return -(
        targets
        * F.log_softmax(
            beta_candidate_logits(alpha_raw, beta_raw, candidate_zs), dim=-1
        )
    ).sum(-1)


def critic_boundary_energy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1)


def _softmax_mixture_message_and_columns(
    logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    child_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(logits, dim=-1)
    mixture = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    columns = weights.unsqueeze(-1) * (expert_outputs - mixture.unsqueeze(1))
    message = (columns * child_residual.unsqueeze(1)).sum(dim=-1)
    return message, columns


def softmax_mixture_message(
    logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    child_residual: torch.Tensor,
) -> torch.Tensor:
    return _softmax_mixture_message_and_columns(
        logits, expert_outputs, child_residual
    )[0]


def softmax_mixture_message_and_gn(
    logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    child_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    message, columns = _softmax_mixture_message_and_columns(
        logits, expert_outputs, child_residual
    )
    columns64 = columns.to(torch.float64)
    gram = torch.einsum("beh,bfh->bef", columns64, columns64)
    return message, gram


def _activity_lists(activities: DAGActivities):
    return (
        list(activities.main.unbind(dim=1)),
        list(activities.projection.unbind(dim=1)),
        list(activities.residual_logit.unbind(dim=1)),
        list(activities.x_in.unbind(dim=1)),
        list(activities.branch_pre.unbind(dim=1)),
        list(activities.branch_out.unbind(dim=1)),
        list(activities.moe_logits.unbind(dim=1)),
        activities.trunk,
        activities.actor_alpha_raw,
        activities.actor_beta_raw,
        activities.critic_logits,
    )


def _pack_activity_lists(
    main,
    projection,
    residual_logit,
    x_in,
    branch_pre,
    branch_out,
    moe_logits,
    trunk,
    actor_alpha_raw,
    actor_beta_raw,
    critic_logits,
) -> DAGActivities:
    return DAGActivities(
        torch.stack(main, dim=1),
        torch.stack(projection, dim=1),
        torch.stack(residual_logit, dim=1),
        torch.stack(x_in, dim=1),
        torch.stack(branch_pre, dim=1),
        torch.stack(branch_out, dim=1),
        torch.stack(moe_logits, dim=1),
        trunk,
        actor_alpha_raw,
        actor_beta_raw,
        critic_logits,
    )


def _history_gradient_and_preconditioner(
    agent: Agent,
    main: list[torch.Tensor],
    residual_logit: list[torch.Tensor],
    main_error: list[torch.Tensor],
    projection_error: list[torch.Tensor],
    x_error: list[torch.Tensor],
    branch_pre_error: list[torch.Tensor],
    trunk_error: torch.Tensor,
    main_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = _history_gradient(
        agent,
        main,
        residual_logit,
        main_error,
        projection_error,
        x_error,
        branch_pre_error,
        trunk_error,
        main_index,
    )
    hidden = main[0].shape[-1]
    identity = torch.eye(hidden, device=gradient.device, dtype=torch.float64)
    preconditioner = identity.unsqueeze(0).expand(
        gradient.shape[0], -1, -1
    ).clone()
    for later_index in range(main_index, len(agent.trunk.blocks)):
        block = agent.trunk.blocks[later_index]
        start = main_index * hidden
        weight_slice = block.in_proj.weight[:, start : start + hidden]
        weight_slice64 = weight_slice.to(torch.float64)
        preconditioner = preconditioner + (
            weight_slice64.T @ weight_slice64
        ).unsqueeze(0)
        for branch_index, branch in enumerate(_branch_modules(block)):
            history_weight = branch.history_weight[main_index]
            preconditioner = preconditioner + torch.diag(
                history_weight.to(torch.float64).square()
            ).unsqueeze(0)
    if main_index == 0:
        for block_index in range(len(agent.trunk.blocks)):
            gate = torch.sigmoid(
                residual_logit[block_index].to(torch.float64)
            )
            skip_derivative = 1.0 - gate
            preconditioner = preconditioner + torch.diag_embed(
                skip_derivative.square()
            )
    final_history_float = torch.cat(main, dim=-1)
    final_epsilon = _rms_epsilon(agent.trunk.out_norm, final_history_float)
    final_history = final_history_float.to(torch.float64)
    start = main_index * hidden
    final_weight = agent.trunk.out_proj.weight.to(torch.float64)
    final_gram = final_weight.T @ final_weight
    preconditioner = preconditioner + rmsnorm_slice_row_gram(
        final_history,
        final_gram,
        start,
        hidden,
        final_epsilon,
    )
    return gradient, preconditioner


def _history_gradient(
    agent: Agent,
    main: list[torch.Tensor],
    residual_logit: list[torch.Tensor],
    main_error: list[torch.Tensor],
    projection_error: list[torch.Tensor],
    x_error: list[torch.Tensor],
    branch_pre_error: list[torch.Tensor],
    trunk_error: torch.Tensor,
    main_index: int,
) -> torch.Tensor:
    hidden = main[0].shape[-1]
    gradient = main_error[main_index]
    for later_index in range(main_index, len(agent.trunk.blocks)):
        block = agent.trunk.blocks[later_index]
        start = main_index * hidden
        weight_slice = block.in_proj.weight[:, start : start + hidden]
        gradient = gradient - projection_error[later_index] @ weight_slice
        for branch_index, branch in enumerate(_branch_modules(block)):
            gradient = gradient - (
                branch_pre_error[later_index][:, branch_index]
                * branch.history_weight[main_index]
            )
    if main_index == 0:
        for block_index in range(len(agent.trunk.blocks)):
            gate = torch.sigmoid(residual_logit[block_index])
            gradient = gradient - (1.0 - gate) * x_error[block_index]
    final_history = torch.cat(main, dim=-1)
    final_message = rmsnorm_vjp(
        final_history,
        trunk_error @ agent.trunk.out_proj.weight,
        _rms_epsilon(agent.trunk.out_norm, final_history),
    )
    start = main_index * hidden
    return gradient - final_message[:, start : start + hidden]


def _branch_pre_gradient_and_factors(
    block: ThinkBlock,
    branch_pre: torch.Tensor,
    branch_pre_error: torch.Tensor,
    branch_out_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = _branch_modules(block)
    weights = torch.stack(
        [branch.out_linear.weight for branch in branches], dim=0
    )
    weights64 = weights.to(torch.float64)
    weight_grams = weights64.transpose(-1, -2) @ weights64
    derivative = relu_squared_derivative(branch_pre)
    outgoing = torch.einsum("bro,roi->bri", branch_out_error, weights)
    gradient = branch_pre_error - derivative * outgoing
    return gradient, derivative, weight_grams


def _branch_pre_gradient(
    block: ThinkBlock,
    branch_pre: torch.Tensor,
    branch_pre_error: torch.Tensor,
    branch_out_error: torch.Tensor,
) -> torch.Tensor:
    weights = torch.stack(
        [branch.out_linear.weight for branch in _branch_modules(block)], dim=0
    )
    outgoing = torch.einsum("bro,roi->bri", branch_out_error, weights)
    return branch_pre_error - relu_squared_derivative(branch_pre) * outgoing


def _branch_pre_gradient_and_preconditioner(
    block: ThinkBlock,
    branch_pre: torch.Tensor,
    branch_pre_error: torch.Tensor,
    branch_out_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient, derivative, weight_grams = _branch_pre_gradient_and_factors(
        block, branch_pre, branch_pre_error, branch_out_error
    )
    identity = torch.eye(
        branch_pre.shape[-1], device=branch_pre.device, dtype=torch.float64
    )
    derivative = derivative.to(torch.float64)
    preconditioner = identity + (
        derivative.unsqueeze(-1)
        * weight_grams.unsqueeze(0)
        * derivative.unsqueeze(-2)
    )
    return gradient, preconditioner


def _solve_branch_pre_exact(
    derivative: torch.Tensor,
    weight_grams: torch.Tensor,
    gradient: torch.Tensor,
    group_size: int = 4,
) -> torch.Tensor:
    """Solve the exact row-local branch systems in bounded-memory groups."""

    batch_size, branch_count, hidden = derivative.shape
    identity = torch.eye(
        hidden, device=gradient.device, dtype=weight_grams.dtype
    )
    steps = []
    for start in range(0, branch_count, group_size):
        stop = min(start + group_size, branch_count)
        group_derivative = derivative[:, start:stop].to(weight_grams.dtype)
        group_matrix = identity + (
            group_derivative.unsqueeze(-1)
            * weight_grams[start:stop].unsqueeze(0)
            * group_derivative.unsqueeze(-2)
        )
        group_matrix = group_matrix.reshape(-1, hidden, hidden)
        group_gradient = gradient[:, start:stop].reshape(-1, hidden)
        group_step = _batched_spd_solve(group_matrix, group_gradient)
        steps.append(group_step.reshape(batch_size, stop - start, hidden))
    return torch.cat(steps, dim=1)


def _x_gradient_and_preconditioner(
    block: ThinkBlock,
    x_in: torch.Tensor,
    x_error: torch.Tensor,
    block_output_error: torch.Tensor,
    branch_pre_error: torch.Tensor,
    moe_logit_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient, dense_message, moe_message = _x_gradient_and_messages(
        block,
        x_in,
        x_error,
        block_output_error,
        branch_pre_error,
        moe_logit_error,
    )
    branches = _branch_modules(block)
    dense_weight = block.dense.current_linear.weight.to(torch.float64)
    gate_weight = block.gate.weight.to(torch.float64)
    dense_gram = dense_weight.T @ dense_weight
    moe_gram = gate_weight.T @ gate_weight
    for expert_index, expert in enumerate(block.experts):
        expert_weight = expert.current_linear.weight.to(torch.float64)
        moe_gram = moe_gram + expert_weight.T @ expert_weight
    dense_epsilon = _rms_epsilon(block.dense_norm, x_in)
    moe_epsilon = _rms_epsilon(block.moe_norm, x_in)
    identity = torch.eye(
        x_in.shape[-1], device=x_in.device, dtype=torch.float64
    )
    preconditioner = 2.0 * identity.unsqueeze(0).expand(
        x_in.shape[0], -1, -1
    )
    preconditioner = preconditioner + rmsnorm_row_gram(
        x_in.to(torch.float64), dense_gram, dense_epsilon
    )
    preconditioner = preconditioner + rmsnorm_row_gram(
        x_in.to(torch.float64), moe_gram, moe_epsilon
    )
    return gradient, preconditioner


def _x_gradient_and_messages(
    block: ThinkBlock,
    x_in: torch.Tensor,
    x_error: torch.Tensor,
    block_output_error: torch.Tensor,
    branch_pre_error: torch.Tensor,
    moe_logit_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_message = branch_pre_error[:, 0] @ block.dense.current_linear.weight
    moe_message = moe_logit_error @ block.gate.weight
    for expert_index, expert in enumerate(block.experts):
        moe_message = moe_message + (
            branch_pre_error[:, expert_index + 1]
            @ expert.current_linear.weight
        )
    gradient = (
        x_error
        - block_output_error
        - rmsnorm_vjp(
            x_in,
            dense_message,
            _rms_epsilon(block.dense_norm, x_in),
        )
        - rmsnorm_vjp(
            x_in,
            moe_message,
            _rms_epsilon(block.moe_norm, x_in),
        )
    )
    return gradient, dense_message, moe_message


def _trunk_gradient_and_preconditioner(
    agent: Agent,
    trunk_error: torch.Tensor,
    alpha_error: torch.Tensor,
    beta_error: torch.Tensor,
    critic_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = (
        trunk_error
        - alpha_error @ agent.actor_alpha_head.weight
        - beta_error @ agent.actor_beta_head.weight
        - critic_error @ agent.critic_head.weight
    )
    identity = torch.eye(
        trunk_error.shape[-1], device=trunk_error.device, dtype=torch.float64
    )
    preconditioner = identity
    for head in (
        agent.actor_alpha_head,
        agent.actor_beta_head,
        agent.critic_head,
    ):
        weight = head.weight.to(torch.float64)
        preconditioner = preconditioner + weight.T @ weight
    return gradient, preconditioner


def _settlement_energy(
    residuals: DAGActivities,
    activities: DAGActivities,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    tpo_coefficient: float,
    value_coefficient: float,
) -> torch.Tensor:
    return (
        dag_model_energy(residuals)
        + tpo_coefficient
        * actor_boundary_energy(
            activities.actor_alpha_raw,
            activities.actor_beta_raw,
            candidate_zs,
            q_targets,
        ).sum()
        + value_coefficient
        * critic_boundary_energy(activities.critic_logits, value_targets).sum()
    )


def dag_stationarity_rms(
    agent: Agent,
    observations: torch.Tensor,
    activities: DAGActivities,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    tpo_coefficient: float = 1.0,
    value_coefficient: float = 1.0,
) -> torch.Tensor:
    predictions = dag_predictions(agent, observations, activities)
    residuals = dag_residuals(activities, predictions)
    (
        main,
        projection,
        residual_logit,
        x_in,
        branch_pre,
        branch_out,
        moe_logits,
        trunk,
        alpha_raw,
        beta_raw,
        critic_logits,
    ) = _activity_lists(activities)
    (
        main_error,
        projection_error,
        residual_error,
        x_error,
        branch_pre_error,
        branch_out_error,
        moe_logit_error,
        trunk_error,
        alpha_error,
        beta_error,
        critic_error,
    ) = _activity_lists(residuals)
    task_gradient = actor_boundary_gradient(
        alpha_raw, beta_raw, candidate_zs, q_targets
    )
    actor_gradient = torch.cat((alpha_error, beta_error), dim=-1) + (
        tpo_coefficient * task_gradient
    )
    critic_gradient = critic_error + value_coefficient * (
        torch.softmax(critic_logits, dim=-1) - value_targets
    )
    trunk_gradient = (
        trunk_error
        - alpha_error @ agent.actor_alpha_head.weight
        - beta_error @ agent.actor_beta_head.weight
        - critic_error @ agent.critic_head.weight
    )
    gradients = [actor_gradient, critic_gradient, trunk_gradient]
    for main_index in range(len(main)):
        history_gradient = _history_gradient(
            agent,
            main,
            residual_logit,
            main_error,
            projection_error,
            x_error,
            branch_pre_error,
            trunk_error,
            main_index,
        )
        gradients.append(history_gradient)
    for block_index, block in enumerate(agent.trunk.blocks):
        gate_message = softmax_mixture_message(
            moe_logits[block_index],
            branch_out[block_index][:, 1:],
            main_error[block_index + 1],
        )
        gradients.append(moe_logit_error[block_index] - gate_message)
        weights = torch.softmax(moe_logits[block_index], dim=-1)
        output_direction = torch.cat(
            (
                torch.ones_like(weights[:, :1]),
                weights,
            ),
            dim=-1,
        ).unsqueeze(-1)
        gradients.append(
            branch_out_error[block_index]
            - output_direction * main_error[block_index + 1].unsqueeze(1)
        )
        pre_gradient = _branch_pre_gradient(
            block,
            branch_pre[block_index],
            branch_pre_error[block_index],
            branch_out_error[block_index],
        )
        gradients.append(pre_gradient)
        mixed_gradient, _, _ = _x_gradient_and_messages(
            block,
            x_in[block_index],
            x_error[block_index],
            main_error[block_index + 1],
            branch_pre_error[block_index],
            moe_logit_error[block_index],
        )
        gradients.append(mixed_gradient)
        gate = torch.sigmoid(residual_logit[block_index])
        residual_direction = gate * (1.0 - gate) * (
            projection[block_index] - main[0]
        )
        pair_direction = torch.stack((gate, residual_direction), dim=1)
        pair_error = torch.stack(
            (projection_error[block_index], residual_error[block_index]), dim=1
        )
        gradients.append(
            pair_error - pair_direction * x_error[block_index].unsqueeze(1)
        )
    squared_sum = torch.stack(
        [gradient.square().sum() for gradient in gradients]
    ).sum()
    element_count = sum(gradient.numel() for gradient in gradients)
    return (squared_sum / element_count).sqrt()


def _batched_spd_solve(
    matrix: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    solve_dtype = (
        torch.float64
        if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else matrix.dtype
    )
    solve_matrix = matrix.to(solve_dtype)
    solve_gradient = gradient.to(solve_dtype)
    cholesky, info = torch.linalg.cholesky_ex(
        solve_matrix, check_errors=False
    )
    torch._assert_async(
        info.eq(0).all(), "non-positive exact row-local GN system"
    )
    result = torch.cholesky_solve(
        solve_gradient.unsqueeze(-1), cholesky
    ).squeeze(-1)
    return result.to(gradient.dtype)


def _block_pre_predictions(
    block: ThinkBlock,
    x_in: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    dense_current = block.dense_norm(x_in)
    moe_current = block.moe_norm(x_in)
    predictions = [_indexed_preact(block.dense, dense_current, history)]
    predictions.extend(
        _indexed_preact(expert, moe_current, history)
        for expert in block.experts
    )
    return torch.stack(predictions, dim=1)


def _block_out_predictions(
    block: ThinkBlock, branch_pre: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            branch.out_linear(
                torch.relu(branch_pre[:, branch_index]).pow(2)
            )
            for branch_index, branch in enumerate(_branch_modules(block))
        ],
        dim=1,
    )


def _block_output_prediction(
    x_in: torch.Tensor,
    branch_out: torch.Tensor,
    moe_logits: torch.Tensor,
) -> torch.Tensor:
    weights = torch.softmax(moe_logits, dim=-1)
    return (
        x_in
        + branch_out[:, 0]
        + (weights.unsqueeze(-1) * branch_out[:, 1:]).sum(dim=1)
    )


def _settle_dag_core(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    inference_steps: int,
    tpo_coefficient: float = 1.0,
    value_coefficient: float = 1.0,
) -> DAGSettleResult:
    """Detached reverse-GS PC with one full residual refresh per sweep."""

    with torch.no_grad():
        free = free_dag_activities(agent, observations)
        (
            main,
            projection,
            residual_logit,
            x_in,
            branch_pre,
            branch_out,
            moe_logits,
            trunk,
            alpha_raw,
            beta_raw,
            critic_logits,
        ) = _activity_lists(free)
        free_predictions = dag_predictions(agent, observations, free)
        free_residuals = dag_residuals(free, free_predictions)
        energies = [
            _settlement_energy(
                free_residuals,
                free,
                candidate_zs,
                q_targets,
                value_targets,
                tpo_coefficient,
                value_coefficient,
            )
        ]
        actor_energies = [
            actor_boundary_energy(
                free.actor_alpha_raw,
                free.actor_beta_raw,
                candidate_zs,
                q_targets,
            ).sum()
        ]
        critic_energies = [
            critic_boundary_energy(free.critic_logits, value_targets).sum()
        ]
        stationarity = [
            dag_stationarity_rms(
                agent,
                observations,
                free,
                candidate_zs,
                q_targets,
                value_targets,
                tpo_coefficient,
                value_coefficient,
            )
        ]

        for _ in range(inference_steps):
            current = _pack_activity_lists(
                main,
                projection,
                residual_logit,
                x_in,
                branch_pre,
                branch_out,
                moe_logits,
                trunk,
                alpha_raw,
                beta_raw,
                critic_logits,
            )
            residuals = dag_residuals(
                current, dag_predictions(agent, observations, current)
            )
            (
                main_error,
                projection_error,
                residual_error,
                x_error,
                branch_pre_error,
                branch_out_error,
                moe_logit_error,
                trunk_error,
                alpha_error,
                beta_error,
                critic_error,
            ) = _activity_lists(residuals)

            # Coupled alpha/beta task boundary: one exact per-row 2A solve.
            actor_task_gradient, actor_task_gn = actor_boundary_gradient_and_gn(
                alpha_raw, beta_raw, candidate_zs, q_targets
            )
            actor_error = torch.cat((alpha_error, beta_error), dim=-1)
            actor_gradient = actor_error + tpo_coefficient * actor_task_gradient
            actor_identity = torch.eye(
                actor_gradient.shape[-1],
                device=actor_gradient.device,
                dtype=actor_task_gn.dtype,
            ).expand_as(actor_task_gn)
            actor_step = _batched_spd_solve(
                actor_identity + tpo_coefficient * actor_task_gn,
                actor_gradient,
            )
            alpha_step, beta_step = actor_step.chunk(2, dim=-1)
            alpha_raw = (alpha_raw - alpha_step).detach()
            beta_raw = (beta_raw - beta_step).detach()
            alpha_error = alpha_error - alpha_step
            beta_error = beta_error - beta_step

            critic_probabilities = torch.softmax(critic_logits, dim=-1)
            critic_gradient = critic_error + value_coefficient * (
                critic_probabilities - value_targets
            )
            critic_step = solve_categorical_identity(
                critic_probabilities, critic_gradient, value_coefficient
            )
            critic_logits = (critic_logits - critic_step).detach()
            critic_error = critic_error - critic_step

            # One shared trunk receives the sum of both actor heads and critic.
            trunk_gradient, trunk_preconditioner = (
                _trunk_gradient_and_preconditioner(
                    agent,
                    trunk_error,
                    alpha_error,
                    beta_error,
                    critic_error,
                )
            )
            trunk_step = _shared_spd_solve(
                trunk_preconditioner, trunk_gradient
            )
            trunk = (trunk - trunk_step).detach()
            trunk_error = trunk_error - trunk_step
            alpha_error = alpha_raw - agent.actor_alpha_head(trunk)
            beta_error = beta_raw - agent.actor_beta_head(trunk)
            critic_error = critic_logits - agent.critic_head(trunk)

            for block_index in reversed(range(len(agent.trunk.blocks))):
                block = agent.trunk.blocks[block_index]
                main_index = block_index + 1

                history_gradient, history_preconditioner = (
                    _history_gradient_and_preconditioner(
                        agent,
                        main,
                        residual_logit,
                        main_error,
                        projection_error,
                        x_error,
                        branch_pre_error,
                        trunk_error,
                        main_index,
                    )
                )
                history_step = _batched_spd_solve(
                    history_preconditioner, history_gradient
                )
                main[main_index] = (main[main_index] - history_step).detach()
                main_error[main_index] = main_error[main_index] - history_step
                for later_index in range(
                    main_index, len(agent.trunk.blocks)
                ):
                    later_block = agent.trunk.blocks[later_index]
                    later_history = torch.cat(
                        main[: later_index + 1], dim=-1
                    )
                    projection_error[later_index] = (
                        projection[later_index]
                        - later_block.in_proj(later_history)
                    )
                    branch_pre_error[later_index] = (
                        branch_pre[later_index]
                        - _block_pre_predictions(
                            later_block,
                            x_in[later_index],
                            later_history,
                        )
                    )
                final_history = torch.cat(main, dim=-1)
                trunk_error = trunk - agent.trunk.out_proj(
                    agent.trunk.out_norm(final_history)
                )

                gate_message, gate_gn = softmax_mixture_message_and_gn(
                    moe_logits[block_index],
                    branch_out[block_index][:, 1:],
                    main_error[main_index],
                )
                gate_gradient = moe_logit_error[block_index] - gate_message
                gate_identity = torch.eye(
                    gate_gradient.shape[-1],
                    device=gate_gradient.device,
                    dtype=gate_gn.dtype,
                ).expand_as(gate_gn)
                gate_step = _batched_spd_solve(
                    gate_identity + gate_gn, gate_gradient
                )
                moe_logits[block_index] = (
                    moe_logits[block_index] - gate_step
                ).detach()
                moe_logit_error[block_index] = (
                    moe_logit_error[block_index] - gate_step
                )
                main_error[main_index] = main[main_index] - (
                    _block_output_prediction(
                        x_in[block_index],
                        branch_out[block_index],
                        moe_logits[block_index],
                    )
                )

                weights = torch.softmax(moe_logits[block_index], dim=-1)
                output_direction = torch.cat(
                    (torch.ones_like(weights[:, :1]), weights), dim=-1
                ).unsqueeze(-1)
                output_gradient = branch_out_error[block_index] - (
                    output_direction * main_error[main_index].unsqueeze(1)
                )
                output_step = solve_rank_one_identity(
                    output_gradient, output_direction, solve_dimension=1
                )
                branch_out[block_index] = (
                    branch_out[block_index] - output_step
                ).detach()
                branch_out_error[block_index] = (
                    branch_out_error[block_index] - output_step
                )
                main_error[main_index] = main[main_index] - (
                    _block_output_prediction(
                        x_in[block_index],
                        branch_out[block_index],
                        moe_logits[block_index],
                    )
                )

                pre_gradient, pre_derivative, pre_weight_grams = (
                    _branch_pre_gradient_and_factors(
                        block,
                        branch_pre[block_index],
                        branch_pre_error[block_index],
                        branch_out_error[block_index],
                    )
                )
                pre_step = _solve_branch_pre_exact(
                    pre_derivative, pre_weight_grams, pre_gradient
                )
                branch_pre[block_index] = (
                    branch_pre[block_index] - pre_step
                ).detach()
                branch_pre_error[block_index] = (
                    branch_pre_error[block_index] - pre_step
                )
                branch_out_error[block_index] = (
                    branch_out[block_index]
                    - _block_out_predictions(
                        block, branch_pre[block_index]
                    )
                )

                mixed_gradient, mixed_preconditioner = (
                    _x_gradient_and_preconditioner(
                        block,
                        x_in[block_index],
                        x_error[block_index],
                        main_error[main_index],
                        branch_pre_error[block_index],
                        moe_logit_error[block_index],
                    )
                )
                mixed_step = _batched_spd_solve(
                    mixed_preconditioner, mixed_gradient
                )
                x_in[block_index] = (
                    x_in[block_index] - mixed_step
                ).detach()
                x_error[block_index] = x_error[block_index] - mixed_step
                history = torch.cat(main[: block_index + 1], dim=-1)
                branch_pre_error[block_index] = (
                    branch_pre[block_index]
                    - _block_pre_predictions(
                        block, x_in[block_index], history
                    )
                )
                moe_logit_error[block_index] = (
                    moe_logits[block_index]
                    - block.gate(block.moe_norm(x_in[block_index]))
                )
                main_error[main_index] = main[main_index] - (
                    _block_output_prediction(
                        x_in[block_index],
                        branch_out[block_index],
                        moe_logits[block_index],
                    )
                )

                residual_gate = torch.sigmoid(residual_logit[block_index])
                residual_direction = (
                    residual_gate
                    * (1.0 - residual_gate)
                    * (projection[block_index] - main[0])
                )
                pair_direction = torch.stack(
                    (residual_gate, residual_direction), dim=1
                )
                pair_error = torch.stack(
                    (
                        projection_error[block_index],
                        residual_error[block_index],
                    ),
                    dim=1,
                )
                pair_gradient = pair_error - (
                    pair_direction * x_error[block_index].unsqueeze(1)
                )
                pair_step = solve_rank_one_identity(
                    pair_gradient, pair_direction, solve_dimension=1
                )
                projection_step = pair_step[:, 0]
                residual_step = pair_step[:, 1]
                projection[block_index] = (
                    projection[block_index] - projection_step
                ).detach()
                residual_logit[block_index] = (
                    residual_logit[block_index] - residual_step
                ).detach()
                projection_error[block_index] = (
                    projection_error[block_index] - projection_step
                )
                residual_error[block_index] = (
                    residual_error[block_index] - residual_step
                )
                residual_gate = torch.sigmoid(residual_logit[block_index])
                x_error[block_index] = x_in[block_index] - (
                    residual_gate * projection[block_index]
                    + (1.0 - residual_gate) * main[0]
                )

            # Entry is the final history node and refreshes all direct consumers.
            entry_gradient, entry_preconditioner = (
                _history_gradient_and_preconditioner(
                    agent,
                    main,
                    residual_logit,
                    main_error,
                    projection_error,
                    x_error,
                    branch_pre_error,
                    trunk_error,
                    0,
                )
            )
            entry_step = _batched_spd_solve(
                entry_preconditioner, entry_gradient
            )
            main[0] = (main[0] - entry_step).detach()
            main_error[0] = main_error[0] - entry_step
            for block_index, block in enumerate(agent.trunk.blocks):
                history = torch.cat(main[: block_index + 1], dim=-1)
                projection_error[block_index] = (
                    projection[block_index] - block.in_proj(history)
                )
                branch_pre_error[block_index] = (
                    branch_pre[block_index]
                    - _block_pre_predictions(
                        block, x_in[block_index], history
                    )
                )
                gate = torch.sigmoid(residual_logit[block_index])
                x_error[block_index] = x_in[block_index] - (
                    gate * projection[block_index]
                    + (1.0 - gate) * main[0]
                )
            final_history = torch.cat(main, dim=-1)
            trunk_error = trunk - agent.trunk.out_proj(
                agent.trunk.out_norm(final_history)
            )

            current = _pack_activity_lists(
                main,
                projection,
                residual_logit,
                x_in,
                branch_pre,
                branch_out,
                moe_logits,
                trunk,
                alpha_raw,
                beta_raw,
                critic_logits,
            )
            current_residuals = _pack_activity_lists(
                main_error,
                projection_error,
                residual_error,
                x_error,
                branch_pre_error,
                branch_out_error,
                moe_logit_error,
                trunk_error,
                alpha_error,
                beta_error,
                critic_error,
            )
            energies.append(
                _settlement_energy(
                    current_residuals,
                    current,
                    candidate_zs,
                    q_targets,
                    value_targets,
                    tpo_coefficient,
                    value_coefficient,
                )
            )
            actor_energies.append(
                actor_boundary_energy(
                    alpha_raw, beta_raw, candidate_zs, q_targets
                ).sum()
            )
            critic_energies.append(
                critic_boundary_energy(critic_logits, value_targets).sum()
            )
            stationarity.append(
                dag_stationarity_rms(
                    agent,
                    observations,
                    current,
                    candidate_zs,
                    q_targets,
                    value_targets,
                    tpo_coefficient,
                    value_coefficient,
                )
            )
    return DAGSettleResult(
        current,
        torch.stack(energies).detach(),
        torch.stack(stationarity).detach(),
        torch.stack(actor_energies).detach(),
        torch.stack(critic_energies).detach(),
    )


def validate_dag_settle_result(result: DAGSettleResult) -> DAGSettleResult:
    _require_finite(
        "ThinkTrunk DAG settlement",
        *result.activities,
        result.energies,
        result.stationarity_rms,
        result.actor_boundary_energies,
        result.critic_boundary_energies,
    )
    if result.energies.shape[0] >= 4:
        recent_growth = result.energies[-3:] > result.energies[-4:-1]
        if bool(recent_growth.all() and result.energies[-1] > result.energies[0]):
            raise RuntimeError("ThinkTrunk PC energy rose systematically")
    return result


def settle_dag(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    inference_steps: int = PC_INFERENCE_STEPS,
    tpo_coefficient: float = 1.0,
    value_coefficient: float = 1.0,
) -> DAGSettleResult:
    return validate_dag_settle_result(
        _settle_dag_core(
            agent,
            observations,
            candidate_zs,
            q_targets,
            value_targets,
            inference_steps,
            tpo_coefficient,
            value_coefficient,
        )
    )


class AffineStatistics(NamedTuple):
    covariance: torch.Tensor
    residual_cross: torch.Tensor
    residual_sse: torch.Tensor
    rows: int
    has_bias: bool


class BranchOutStatistics(NamedTuple):
    covariance: torch.Tensor
    residual_cross: torch.Tensor
    residual_sse: torch.Tensor
    rows: int


class JointBranchStatistics(NamedTuple):
    covariance: torch.Tensor
    residual_cross: torch.Tensor
    residual_sse: torch.Tensor
    rows: int


class ResidualGateStatistics(NamedTuple):
    residual_sum: torch.Tensor
    residual_sse: torch.Tensor
    rows: int


class TerminalMessageStatistics(NamedTuple):
    actor_squared_sum: torch.Tensor
    critic_squared_sum: torch.Tensor
    cross_sum: torch.Tensor
    elements: int


class DAGStatistics(NamedTuple):
    affine: dict[str, AffineStatistics]
    branch_out: list[BranchOutStatistics]
    joint_branch: list[JointBranchStatistics]
    residual_gate: list[ResidualGateStatistics]


class DAGMstepResult(NamedTuple):
    corrections: dict[str, torch.Tensor]
    sse_before: torch.Tensor
    sse_after: torch.Tensor
    min_rank_fraction: torch.Tensor
    max_condition: torch.Tensor
    factor_rank_fraction: dict[str, torch.Tensor]
    factor_condition: dict[str, torch.Tensor]
    factor_correction_norm: dict[str, torch.Tensor]


def _empty_affine_statistics(
    module: nn.Linear, device: torch.device
) -> AffineStatistics:
    feature_dimension = module.in_features + int(module.bias is not None)
    return AffineStatistics(
        torch.zeros(
            feature_dimension,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(
            module.out_features,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros((), dtype=torch.float64, device=device),
        0,
        module.bias is not None,
    )


def _accumulate_affine_statistics(
    statistics: AffineStatistics,
    features: torch.Tensor,
    residual: torch.Tensor,
) -> AffineStatistics:
    design = features.to(torch.float64)
    if statistics.has_bias:
        design = torch.cat(
            (design, torch.ones_like(design[:, :1])), dim=-1
        )
    stopped_residual = residual.to(torch.float64)
    return AffineStatistics(
        statistics.covariance + design.T @ design,
        statistics.residual_cross + stopped_residual.T @ design,
        statistics.residual_sse + stopped_residual.square().sum(),
        statistics.rows + design.shape[0],
        statistics.has_bias,
    )


def _empty_branch_out_statistics(
    branches: int, hidden: int, device: torch.device
) -> BranchOutStatistics:
    feature_dimension = hidden + 1
    return BranchOutStatistics(
        torch.zeros(
            branches,
            feature_dimension,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(
            branches,
            hidden,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(branches, dtype=torch.float64, device=device),
        0,
    )


def _accumulate_branch_out_statistics(
    statistics: BranchOutStatistics,
    branch_pre: torch.Tensor,
    residual: torch.Tensor,
) -> BranchOutStatistics:
    features = torch.relu(branch_pre).square().to(torch.float64)
    design = torch.cat(
        (features, torch.ones_like(features[..., :1])), dim=-1
    )
    stopped_residual = residual.to(torch.float64)
    return BranchOutStatistics(
        statistics.covariance
        + torch.einsum("brf,brg->rfg", design, design),
        statistics.residual_cross
        + torch.einsum("bro,brf->rof", stopped_residual, design),
        statistics.residual_sse
        + stopped_residual.square().sum(dim=(0, 2)),
        statistics.rows + design.shape[0],
    )


def _empty_joint_branch_statistics(
    branches: int,
    hidden: int,
    history_slots: int,
    device: torch.device,
) -> JointBranchStatistics:
    feature_dimension = hidden + history_slots + 1
    return JointBranchStatistics(
        torch.zeros(
            hidden,
            feature_dimension,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(
            hidden,
            branches,
            feature_dimension,
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(
            hidden, branches, dtype=torch.float64, device=device
        ),
        0,
    )


def joint_indexed_design(
    normalized_current: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    batch, history_slots, hidden = history.shape
    current = normalized_current.unsqueeze(1).expand(batch, hidden, hidden)
    same_index_history = history.transpose(1, 2)
    return torch.cat(
        (
            current,
            same_index_history,
            torch.ones(
                batch,
                hidden,
                1,
                dtype=normalized_current.dtype,
                device=normalized_current.device,
            ),
        ),
        dim=-1,
    )


def _accumulate_joint_branch_statistics(
    statistics: JointBranchStatistics,
    normalized_current: torch.Tensor,
    history: torch.Tensor,
    residual: torch.Tensor,
) -> JointBranchStatistics:
    design = joint_indexed_design(normalized_current, history).to(torch.float64)
    stopped_residual = residual.to(torch.float64).permute(0, 2, 1)
    return JointBranchStatistics(
        statistics.covariance
        + torch.einsum("bhf,bhg->hfg", design, design),
        statistics.residual_cross
        + torch.einsum("bhr,bhf->hrf", stopped_residual, design),
        statistics.residual_sse
        + stopped_residual.square().sum(dim=0),
        statistics.rows + design.shape[0],
    )


def _dag_affine_modules(agent: Agent) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {"trunk.entry": agent.trunk.entry}
    for block_index, block in enumerate(agent.trunk.blocks):
        modules[f"trunk.blocks.{block_index}.in_proj"] = block.in_proj
        modules[f"trunk.blocks.{block_index}.gate"] = block.gate
    modules["trunk.out_proj"] = agent.trunk.out_proj
    modules["actor_alpha_head"] = agent.actor_alpha_head
    modules["actor_beta_head"] = agent.actor_beta_head
    modules["critic_head"] = agent.critic_head
    return modules


def empty_dag_statistics(agent: Agent, device: torch.device) -> DAGStatistics:
    affine = {
        name: _empty_affine_statistics(module, device)
        for name, module in _dag_affine_modules(agent).items()
    }
    branch_out = []
    joint_branch = []
    residual_gate = []
    for block in agent.trunk.blocks:
        branch_count = 1 + len(block.experts)
        hidden = block.resid_gate.numel()
        branch_out.append(
            _empty_branch_out_statistics(branch_count, hidden, device)
        )
        joint_branch.append(
            _empty_joint_branch_statistics(
                branch_count,
                hidden,
                block.dense.history_slots,
                device,
            )
        )
        residual_gate.append(
            ResidualGateStatistics(
                torch.zeros(hidden, dtype=torch.float64, device=device),
                torch.zeros((), dtype=torch.float64, device=device),
                0,
            )
        )
    return DAGStatistics(
        affine,
        branch_out,
        joint_branch,
        residual_gate,
    )


def accumulate_dag_statistics(
    agent: Agent,
    statistics: DAGStatistics,
    observations: torch.Tensor,
    settled: DAGActivities,
) -> TerminalMessageStatistics:
    """Stopped whole-chunk sufficient statistics without live mutation."""

    with torch.no_grad():
        residuals = dag_residuals(
            settled, dag_predictions(agent, observations, settled)
        )
        statistics.affine["trunk.entry"] = _accumulate_affine_statistics(
            statistics.affine["trunk.entry"],
            observations,
            residuals.main[:, 0],
        )
        for block_index, block in enumerate(agent.trunk.blocks):
            history = settled.main[:, : block_index + 1]
            flat_history = history.flatten(1)
            in_name = f"trunk.blocks.{block_index}.in_proj"
            statistics.affine[in_name] = _accumulate_affine_statistics(
                statistics.affine[in_name],
                flat_history,
                residuals.projection[:, block_index],
            )
            gate_name = f"trunk.blocks.{block_index}.gate"
            statistics.affine[gate_name] = _accumulate_affine_statistics(
                statistics.affine[gate_name],
                block.moe_norm(settled.x_in[:, block_index]),
                residuals.moe_logits[:, block_index],
            )
            statistics.branch_out[block_index] = _accumulate_branch_out_statistics(
                statistics.branch_out[block_index],
                settled.branch_pre[:, block_index],
                residuals.branch_out[:, block_index],
            )
            statistics.joint_branch[block_index] = (
                _accumulate_joint_branch_statistics(
                    statistics.joint_branch[block_index],
                    block.dense_norm(settled.x_in[:, block_index]),
                    history,
                    residuals.branch_pre[:, block_index],
                )
            )
            gate_stats = statistics.residual_gate[block_index]
            gate_residual = residuals.residual_logit[:, block_index].to(
                torch.float64
            )
            statistics.residual_gate[block_index] = ResidualGateStatistics(
                gate_stats.residual_sum + gate_residual.sum(dim=0),
                gate_stats.residual_sse + gate_residual.square().sum(),
                gate_stats.rows + gate_residual.shape[0],
            )
        final_history = settled.main.flatten(1)
        statistics.affine["trunk.out_proj"] = _accumulate_affine_statistics(
            statistics.affine["trunk.out_proj"],
            agent.trunk.out_norm(final_history),
            residuals.trunk,
        )
        for name, residual in (
            ("actor_alpha_head", residuals.actor_alpha_raw),
            ("actor_beta_head", residuals.actor_beta_raw),
            ("critic_head", residuals.critic_logits),
        ):
            statistics.affine[name] = _accumulate_affine_statistics(
                statistics.affine[name], settled.trunk, residual
            )
        actor_message = (
            residuals.actor_alpha_raw @ agent.actor_alpha_head.weight
            + residuals.actor_beta_raw @ agent.actor_beta_head.weight
        )
        critic_message = residuals.critic_logits @ agent.critic_head.weight
        return TerminalMessageStatistics(
            actor_message.square().sum(),
            critic_message.square().sum(),
            (actor_message * critic_message).sum(),
            actor_message.numel(),
        )


def _rank_condition(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    threshold = (
        matrix.shape[-1]
        * torch.finfo(matrix.dtype).eps
        * eigenvalues[..., -1:]
    )
    retained = eigenvalues > threshold
    rank_fraction = retained.to(torch.float64).mean(dim=-1)
    smallest = torch.where(
        retained, eigenvalues, torch.full_like(eigenvalues, torch.inf)
    ).min(dim=-1).values
    condition = eigenvalues[..., -1] / smallest
    return rank_fraction, condition


def _affine_m_step(
    module: nn.Linear,
    statistics: AffineStatistics,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if statistics.rows == 0:
        raise ValueError("cannot solve empty affine statistics")
    if statistics.has_bias:
        count = statistics.covariance[-1, -1]
        feature_sum = statistics.covariance[:-1, -1]
        residual_sum = statistics.residual_cross[:, -1]
        covariance = (
            statistics.covariance[:-1, :-1]
            - feature_sum[:, None] * feature_sum[None, :] / count
        )
        cross = (
            statistics.residual_cross[:, :-1]
            - residual_sum[:, None] * feature_sum[None, :] / count
        )
        weight = cross @ torch.linalg.pinv(covariance, hermitian=True)
        bias = residual_sum / count - weight @ (feature_sum / count)
        correction = torch.cat((weight, bias[:, None]), dim=-1)
        result = {
            "weight": weight.to(module.weight.dtype),
            "bias": bias.to(module.bias.dtype),
        }
    else:
        covariance = statistics.covariance
        weight = statistics.residual_cross @ torch.linalg.pinv(
            covariance, hermitian=True
        )
        correction = weight
        result = {"weight": weight.to(module.weight.dtype)}
    next_sse = (
        statistics.residual_sse
        - 2.0 * (correction * statistics.residual_cross).sum()
        + (correction @ statistics.covariance * correction).sum()
    )
    rank, condition = _rank_condition(covariance)
    return result, statistics.residual_sse, next_sse, rank, condition


def _branch_out_m_step(
    block: ThinkBlock,
    statistics: BranchOutStatistics,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = statistics.covariance[:, -1, -1]
    feature_sum = statistics.covariance[:, :-1, -1]
    residual_sum = statistics.residual_cross[:, :, -1]
    covariance = (
        statistics.covariance[:, :-1, :-1]
        - feature_sum.unsqueeze(-1) * feature_sum.unsqueeze(-2)
        / count[:, None, None]
    )
    cross = (
        statistics.residual_cross[:, :, :-1]
        - residual_sum.unsqueeze(-1) * feature_sum.unsqueeze(1)
        / count[:, None, None]
    )
    weight = cross @ torch.linalg.pinv(covariance, hermitian=True)
    bias = residual_sum / count[:, None] - torch.einsum(
        "roh,rh->ro", weight, feature_sum / count[:, None]
    )
    correction = torch.cat((weight, bias.unsqueeze(-1)), dim=-1)
    next_sse = (
        statistics.residual_sse
        - 2.0 * (correction * statistics.residual_cross).sum(dim=(1, 2))
        + torch.einsum(
            "rof,rfg,rog->r", correction, statistics.covariance, correction
        )
    )
    corrections = {}
    for branch_index, branch in enumerate(_branch_modules(block)):
        prefix = (
            "dense"
            if branch_index == 0
            else f"experts.{branch_index - 1}"
        )
        corrections[f"{prefix}.out_linear.weight"] = weight[branch_index].to(
            branch.out_linear.weight.dtype
        )
        corrections[f"{prefix}.out_linear.bias"] = bias[branch_index].to(
            branch.out_linear.bias.dtype
        )
    rank, condition = _rank_condition(covariance)
    return corrections, statistics.residual_sse, next_sse, rank, condition


def joint_indexed_m_step(
    block: ThinkBlock,
    statistics: JointBranchStatistics,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One per-channel C+ shared across dense plus all expert RHSs."""

    count = statistics.covariance[:, -1, -1]
    feature_sum = statistics.covariance[:, :-1, -1]
    residual_sum = statistics.residual_cross[:, :, -1]
    covariance = (
        statistics.covariance[:, :-1, :-1]
        - feature_sum.unsqueeze(-1) * feature_sum.unsqueeze(-2)
        / count[:, None, None]
    )
    cross = (
        statistics.residual_cross[:, :, :-1]
        - residual_sum.unsqueeze(-1) * feature_sum.unsqueeze(1)
        / count[:, None, None]
    )
    coefficients = cross @ torch.linalg.pinv(covariance, hermitian=True)
    bias = residual_sum / count[:, None] - torch.einsum(
        "hrf,hf->hr", coefficients, feature_sum / count[:, None]
    )
    correction = torch.cat((coefficients, bias.unsqueeze(-1)), dim=-1)
    next_sse = (
        statistics.residual_sse
        - 2.0 * (correction * statistics.residual_cross).sum(dim=-1)
        + torch.einsum(
            "hrf,hfg,hrg->hr",
            correction,
            statistics.covariance,
            correction,
        )
    )
    hidden = block.resid_gate.numel()
    history_slots = block.dense.history_slots
    corrections = {}
    for branch_index, branch in enumerate(_branch_modules(block)):
        prefix = (
            "dense"
            if branch_index == 0
            else f"experts.{branch_index - 1}"
        )
        branch_coefficients = coefficients[:, branch_index]
        corrections[f"{prefix}.current_linear.weight"] = (
            branch_coefficients[:, :hidden].to(branch.current_linear.weight.dtype)
        )
        corrections[f"{prefix}.history_weight"] = (
            branch_coefficients[:, hidden : hidden + history_slots]
            .T.to(branch.history_weight.dtype)
        )
        corrections[f"{prefix}.current_linear.bias"] = bias[
            :, branch_index
        ].to(branch.current_linear.bias.dtype)
    rank, condition = _rank_condition(covariance)
    return (
        corrections,
        statistics.residual_sse,
        next_sse,
        rank,
        condition,
    )


def dag_m_step(agent: Agent, statistics: DAGStatistics) -> DAGMstepResult:
    corrections: dict[str, torch.Tensor] = {}
    sse_before, sse_after, ranks, conditions = [], [], [], []
    factor_ranks: dict[str, torch.Tensor] = {}
    factor_conditions: dict[str, torch.Tensor] = {}
    factor_norms: dict[str, torch.Tensor] = {}

    def record_factor(
        name: str,
        rank: torch.Tensor,
        condition: torch.Tensor,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        factor_ranks[name] = rank
        factor_conditions[name] = condition
        factor_norms[name] = torch.stack(
            [tensor.to(torch.float64).square().sum() for tensor in tensors]
        ).sum().sqrt()

    for prefix, module in _dag_affine_modules(agent).items():
        local, before, after, rank, condition = _affine_m_step(
            module, statistics.affine[prefix]
        )
        for suffix, correction in local.items():
            corrections[f"{prefix}.{suffix}"] = correction
        sse_before.append(before.reshape(-1))
        sse_after.append(after.reshape(-1))
        ranks.append(rank.reshape(-1))
        conditions.append(condition.reshape(-1))
        record_factor(prefix, rank, condition, tuple(local.values()))
    for block_index, block in enumerate(agent.trunk.blocks):
        block_prefix = f"trunk.blocks.{block_index}"
        local, before, after, rank, condition = _branch_out_m_step(
            block, statistics.branch_out[block_index]
        )
        for suffix, correction in local.items():
            corrections[f"{block_prefix}.{suffix}"] = correction
        sse_before.append(before.reshape(-1))
        sse_after.append(after.reshape(-1))
        ranks.append(rank.reshape(-1))
        conditions.append(condition.reshape(-1))
        for branch_index in range(1 + len(block.experts)):
            branch_name = (
                "dense"
                if branch_index == 0
                else f"experts.{branch_index - 1}"
            )
            record_factor(
                f"{block_prefix}.{branch_name}.out_linear",
                rank[branch_index],
                condition[branch_index],
                (
                    local[f"{branch_name}.out_linear.weight"],
                    local[f"{branch_name}.out_linear.bias"],
                ),
            )

        local, before, after, rank, condition = joint_indexed_m_step(
            block, statistics.joint_branch[block_index]
        )
        for suffix, correction in local.items():
            corrections[f"{block_prefix}.{suffix}"] = correction
        sse_before.append(before.reshape(-1))
        sse_after.append(after.reshape(-1))
        ranks.append(rank.reshape(-1))
        conditions.append(condition.reshape(-1))
        for channel in range(block.resid_gate.numel()):
            channel_tensors = []
            for branch_index in range(1 + len(block.experts)):
                branch_name = (
                    "dense"
                    if branch_index == 0
                    else f"experts.{branch_index - 1}"
                )
                channel_tensors.extend(
                    (
                        local[f"{branch_name}.current_linear.weight"][channel],
                        local[f"{branch_name}.current_linear.bias"][channel],
                        local[f"{branch_name}.history_weight"][:, channel],
                    )
                )
            record_factor(
                f"{block_prefix}.indexed.channel.{channel}",
                rank[channel],
                condition[channel],
                tuple(channel_tensors),
            )

        gate_stats = statistics.residual_gate[block_index]
        gate_correction = gate_stats.residual_sum / gate_stats.rows
        next_gate_sse = (
            gate_stats.residual_sse
            - 2.0 * (gate_correction * gate_stats.residual_sum).sum()
            + gate_stats.rows * gate_correction.square().sum()
        )
        corrections[f"{block_prefix}.resid_gate"] = gate_correction.to(
            block.resid_gate.dtype
        )
        sse_before.append(gate_stats.residual_sse.reshape(-1))
        sse_after.append(next_gate_sse.reshape(-1))
        ranks.append(torch.ones(1, dtype=torch.float64, device=gate_correction.device))
        conditions.append(torch.ones(1, dtype=torch.float64, device=gate_correction.device))
        record_factor(
            f"{block_prefix}.resid_gate",
            ranks[-1][0],
            conditions[-1][0],
            (gate_correction,),
        )
    named_parameters = dict(agent.named_parameters())
    if set(corrections) != set(named_parameters):
        missing = sorted(set(named_parameters).difference(corrections))
        extra = sorted(set(corrections).difference(named_parameters))
        raise RuntimeError(
            f"DAG correction coverage mismatch: missing={missing}, extra={extra}"
        )
    _require_finite("DAG C+ corrections", *corrections.values())
    all_sse_before = torch.cat(sse_before)
    all_sse_after = torch.cat(sse_after)
    tolerance = 2e-6 * (1.0 + all_sse_before.abs())
    if bool((all_sse_after - all_sse_before > tolerance).any()):
        raise RuntimeError("a local DAG C+ correction increased stopped SSE")
    return DAGMstepResult(
        corrections,
        all_sse_before,
        all_sse_after,
        torch.cat(ranks).min(),
        torch.cat(conditions).max(),
        factor_ranks,
        factor_conditions,
        factor_norms,
    )


def apply_atomic_dag_corrections(
    agent: Agent, corrections: dict[str, torch.Tensor]
) -> None:
    named_parameters = dict(agent.named_parameters())
    if tuple(corrections) != tuple(named_parameters):
        if set(corrections) != set(named_parameters):
            raise KeyError("atomic DAG update has incomplete parameter coverage")
    _require_finite("atomic DAG update", *corrections.values())
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            parameter.add_(corrections[name])


class DAGCycleDiagnostics(NamedTuple):
    energy_per_row: torch.Tensor
    stationarity_rms: torch.Tensor
    actor_settlement_energy_per_row: torch.Tensor
    critic_settlement_energy_per_row: torch.Tensor
    actor_boundary_ce_before: torch.Tensor
    actor_boundary_ce_after: torch.Tensor
    critic_boundary_ce_before: torch.Tensor
    critic_boundary_ce_after: torch.Tensor
    proposed_behavior_kl: torch.Tensor
    correction_norm: torch.Tensor
    local_sse_before: torch.Tensor
    local_sse_after: torch.Tensor
    min_rank_fraction: torch.Tensor
    max_condition: torch.Tensor
    factor_rank_fraction: dict[str, torch.Tensor]
    factor_condition: dict[str, torch.Tensor]
    factor_correction_norm: dict[str, torch.Tensor]
    actor_terminal_message_rms: torch.Tensor
    critic_terminal_message_rms: torch.Tensor
    terminal_message_cosine: torch.Tensor
    actor_terminal_energy_share: torch.Tensor
    critic_terminal_energy_share: torch.Tensor


class DAGCycleResult(NamedTuple):
    cycle_index: int
    diagnostics: DAGCycleDiagnostics
    statistics: DAGStatistics
    corrections: dict[str, torch.Tensor]


class CorrectedDAGParameters(NamedTuple):
    trunk: dict[str, torch.Tensor]
    alpha_weight: torch.Tensor
    alpha_bias: torch.Tensor
    beta_weight: torch.Tensor
    beta_bias: torch.Tensor
    critic_weight: torch.Tensor


def corrected_dag_parameters(
    agent: Agent,
    corrections: dict[str, torch.Tensor],
) -> CorrectedDAGParameters:
    trunk_parameters = {
        name: parameter + corrections[f"trunk.{name}"]
        for name, parameter in agent.trunk.named_parameters()
    }
    return CorrectedDAGParameters(
        trunk_parameters,
        agent.actor_alpha_head.weight + corrections["actor_alpha_head.weight"],
        agent.actor_alpha_head.bias + corrections["actor_alpha_head.bias"],
        agent.actor_beta_head.weight + corrections["actor_beta_head.weight"],
        agent.actor_beta_head.bias + corrections["actor_beta_head.bias"],
        agent.critic_head.weight + corrections["critic_head.weight"],
    )


def functional_dag_outputs(
    agent: Agent,
    observations: torch.Tensor,
    parameters: CorrectedDAGParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trunk = torch.func.functional_call(
        agent.trunk, parameters.trunk, (observations,)
    )
    alpha_raw = F.linear(
        trunk,
        parameters.alpha_weight,
        parameters.alpha_bias,
    )
    beta_raw = F.linear(
        trunk,
        parameters.beta_weight,
        parameters.beta_bias,
    )
    critic_logits = F.linear(
        trunk,
        parameters.critic_weight,
    )
    return alpha_raw, beta_raw, critic_logits


def proposed_cycle_diagnostics(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    rollout_latent_zs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    corrections: dict[str, torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actor_before, actor_after = [], []
    critic_before, critic_after, behavior_kl = [], [], []
    with torch.no_grad():
        corrected_parameters = corrected_dag_parameters(agent, corrections)
        for start in range(0, observations.shape[0], chunk_size):
            end = min(start + chunk_size, observations.shape[0])
            obs = observations[start:end]
            current = free_dag_activities(agent, obs)
            new_alpha, new_beta, new_critic = functional_dag_outputs(
                agent, obs, corrected_parameters
            )
            actor_before.append(
                actor_boundary_energy(
                    current.actor_alpha_raw,
                    current.actor_beta_raw,
                    candidate_zs[start:end],
                    q_targets[start:end],
                )
            )
            actor_after.append(
                actor_boundary_energy(
                    new_alpha,
                    new_beta,
                    candidate_zs[start:end],
                    q_targets[start:end],
                )
            )
            critic_before.append(
                critic_boundary_energy(
                    current.critic_logits, value_targets[start:end]
                )
            )
            critic_after.append(
                critic_boundary_energy(new_critic, value_targets[start:end])
            )
            alpha = 1.0 + F.softplus(new_alpha)
            beta = 1.0 + F.softplus(new_beta)
            distribution = Beta(alpha, beta, validate_args=False)
            new_logprobs = distribution.log_prob(
                rollout_latent_zs[start:end]
            ).sum(-1)
            logratio = (
                new_logprobs - rollout_logprobs[start:end]
            ).to(torch.float64)
            behavior_kl.append(torch.expm1(logratio) - logratio)
    return tuple(
        torch.cat(values).mean()
        for values in (
            actor_before,
            actor_after,
            critic_before,
            critic_after,
            behavior_kl,
        )
    )


SettleCore = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], DAGSettleResult
]


def propose_dag_cycle(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    rollout_latent_zs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    chunk_size: int,
    settle_core: SettleCore,
    cycle_index: int,
) -> DAGCycleResult:
    rows = observations.shape[0]
    statistics = empty_dag_statistics(agent, observations.device)
    energy_per_row = torch.zeros(
        PC_INFERENCE_STEPS + 1,
        device=observations.device,
        dtype=observations.dtype,
    )
    stationarity_squared = torch.zeros_like(energy_per_row)
    actor_settlement_energy = torch.zeros_like(energy_per_row)
    critic_settlement_energy = torch.zeros_like(energy_per_row)
    settlement_failed = torch.zeros(
        (), dtype=torch.bool, device=observations.device
    )
    actor_message_squared = torch.zeros(
        (), dtype=observations.dtype, device=observations.device
    )
    critic_message_squared = torch.zeros_like(actor_message_squared)
    message_cross = torch.zeros_like(actor_message_squared)
    message_elements = 0
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        result = settle_core(
            observations[start:end],
            candidate_zs[start:end],
            q_targets[start:end],
            value_targets[start:end],
        )
        message_statistics = accumulate_dag_statistics(
            agent, statistics, observations[start:end], result.activities
        )
        actor_message_squared += message_statistics.actor_squared_sum
        critic_message_squared += message_statistics.critic_squared_sum
        message_cross += message_statistics.cross_sum
        message_elements += message_statistics.elements
        finite = torch.stack(
            [
                *(activity.isfinite().all() for activity in result.activities),
                result.energies.isfinite().all(),
                result.stationarity_rms.isfinite().all(),
                result.actor_boundary_energies.isfinite().all(),
                result.critic_boundary_energies.isfinite().all(),
            ]
        ).all()
        systematic_growth = (
            (result.energies[-3:] > result.energies[-4:-1]).all()
            & (result.energies[-1] > result.energies[0])
        )
        settlement_failed |= ~finite | systematic_growth
        energy_per_row += result.energies / rows
        actor_settlement_energy += result.actor_boundary_energies / rows
        critic_settlement_energy += result.critic_boundary_energies / rows
        row_fraction = (end - start) / rows
        stationarity_squared += row_fraction * result.stationarity_rms.square()
    if bool(settlement_failed):
        raise FloatingPointError("invalid whole-rollout ThinkTrunk settlement")
    m_step = dag_m_step(agent, statistics)
    (
        actor_ce_before,
        actor_ce_after,
        critic_ce_before,
        critic_ce_after,
        behavior_kl,
    ) = proposed_cycle_diagnostics(
        agent,
        observations,
        candidate_zs,
        q_targets,
        value_targets,
        rollout_latent_zs,
        rollout_logprobs,
        m_step.corrections,
        chunk_size,
    )
    correction_norm = torch.stack(
        [correction.square().sum() for correction in m_step.corrections.values()]
    ).sum().sqrt()
    message_floor = torch.finfo(observations.dtype).tiny
    actor_message_rms = (actor_message_squared / message_elements).sqrt()
    critic_message_rms = (critic_message_squared / message_elements).sqrt()
    message_cosine = message_cross / (
        (actor_message_squared * critic_message_squared).sqrt() + message_floor
    )
    total_message_energy = (
        actor_message_squared + critic_message_squared + message_floor
    )
    diagnostics = DAGCycleDiagnostics(
        energy_per_row,
        stationarity_squared.sqrt(),
        actor_settlement_energy,
        critic_settlement_energy,
        actor_ce_before,
        actor_ce_after,
        critic_ce_before,
        critic_ce_after,
        behavior_kl,
        correction_norm,
        m_step.sse_before,
        m_step.sse_after,
        m_step.min_rank_fraction,
        m_step.max_condition,
        m_step.factor_rank_fraction,
        m_step.factor_condition,
        m_step.factor_correction_norm,
        actor_message_rms,
        critic_message_rms,
        message_cosine,
        actor_message_squared / total_message_energy,
        critic_message_squared / total_message_energy,
    )
    _require_finite(
        "outer DAG cycle diagnostics",
        energy_per_row,
        diagnostics.stationarity_rms,
        actor_settlement_energy,
        critic_settlement_energy,
        actor_ce_before,
        actor_ce_after,
        critic_ce_before,
        critic_ce_after,
        behavior_kl,
        correction_norm,
        m_step.sse_before,
        m_step.sse_after,
        m_step.min_rank_fraction,
        actor_message_rms,
        critic_message_rms,
        message_cosine,
    )
    return DAGCycleResult(
        cycle_index, diagnostics, statistics, m_step.corrections
    )


def run_dag_outer_gem(
    agent: Agent,
    observations: torch.Tensor,
    candidate_zs: torch.Tensor,
    q_targets: torch.Tensor,
    value_targets: torch.Tensor,
    rollout_latent_zs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    chunk_size: int,
    settle_core: Optional[SettleCore] = None,
) -> tuple[DAGCycleResult, ...]:
    if settle_core is None:
        settle_core = lambda obs, candidates, q, value: _settle_dag_core(
            agent,
            obs,
            candidates,
            q,
            value,
            PC_INFERENCE_STEPS,
        )
    leading_dimensions = (
        candidate_zs.shape[0],
        q_targets.shape[0],
        value_targets.shape[0],
        rollout_latent_zs.shape[0],
        rollout_logprobs.shape[0],
    )
    if observations.shape[0] == 0 or any(
        rows != observations.shape[0] for rows in leading_dimensions
    ):
        raise ValueError("outer DAG GEM requires aligned non-empty rollout tensors")
    _require_finite(
        "immutable DAG rollout evidence",
        observations,
        candidate_zs,
        q_targets,
        value_targets,
        rollout_latent_zs,
        rollout_logprobs,
    )
    target_versions = (q_targets._version, value_targets._version)
    results = []
    for cycle_index in range(OUTER_CYCLES):
        result = propose_dag_cycle(
            agent,
            observations,
            candidate_zs,
            q_targets,
            value_targets,
            rollout_latent_zs,
            rollout_logprobs,
            chunk_size,
            settle_core,
            cycle_index,
        )
        if (q_targets._version, value_targets._version) != target_versions:
            raise RuntimeError("outer DAG GEM target evidence was mutated")
        apply_atomic_dag_corrections(agent, result.corrections)
        results.append(result)
    return tuple(results)


# TASK_TARGETS_FOLLOW
class TpoTarget(NamedTuple):
    probabilities: torch.Tensor
    eta: float
    achieved_reverse_kl: float
    base_reverse_kl: float
    cap_engaged: float
    score_std_mean: float
    score_std_p90: float
    score_scale: float


def build_tpo_target(
    anchor_logprobs: torch.Tensor,
    scores: torch.Tensor,
    current_td_rms: torch.Tensor,
    args: Args,
) -> TpoTarget:
    """v5 reverse-KL cap target, without v5's EMA or score clamp."""

    score_scale = max(
        args.tpo_sigma_scale_coef * float(current_td_rms.detach()), 1e-6
    )
    centered_scores = (scores - scores.mean(dim=-1, keepdim=True)) / score_scale
    anchor = F.log_softmax(anchor_logprobs, dim=-1)
    old_probabilities = anchor.exp()

    def reverse_kl(temperature: float) -> float:
        log_target = F.log_softmax(anchor + centered_scores / temperature, dim=-1)
        return float(
            (old_probabilities * (anchor - log_target)).sum(-1).mean()
        )

    base_kl = reverse_kl(args.tpo_eta_base)
    cap_engaged = 0.0
    if bool(centered_scores.abs().max() < 1e-8):
        eta = args.tpo_eta_base if args.tpo_dyn_trust else 1.0
    elif args.tpo_dyn_trust:
        if base_kl <= args.tpo_eps:
            eta = args.tpo_eta_base
        else:
            cap_engaged = 1.0
            log_low, log_high = float(np.log(args.tpo_eta_base)), float(np.log(1e4))
            if reverse_kl(float(np.exp(log_high))) > args.tpo_eps:
                eta = float(np.exp(log_high))
            else:
                for _ in range(40):
                    midpoint = 0.5 * (log_low + log_high)
                    if reverse_kl(float(np.exp(midpoint))) > args.tpo_eps:
                        log_low = midpoint
                    else:
                        log_high = midpoint
                eta = float(np.exp(0.5 * (log_low + log_high)))
    elif args.tpo_adaptive_eta:
        log_low, log_high = float(np.log(1e-2)), float(np.log(1e4))
        if reverse_kl(float(np.exp(log_low))) < args.tpo_eps:
            eta = float(np.exp(log_low))
        elif reverse_kl(float(np.exp(log_high))) > args.tpo_eps:
            eta = float(np.exp(log_high))
        else:
            for _ in range(40):
                midpoint = 0.5 * (log_low + log_high)
                if reverse_kl(float(np.exp(midpoint))) > args.tpo_eps:
                    log_low = midpoint
                else:
                    log_high = midpoint
            eta = float(np.exp(0.5 * (log_low + log_high)))
    else:
        eta = args.tpo_eta
    probabilities = torch.softmax(anchor + centered_scores / eta, dim=-1).detach()
    group_std = scores.std(dim=-1, unbiased=False)
    return TpoTarget(
        probabilities,
        eta,
        reverse_kl(eta),
        base_kl,
        cap_engaged,
        float(group_std.mean()),
        float(group_std.quantile(0.9)),
        score_scale,
    )


def gae_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminations: torch.Tensor,
    boundaries: torch.Tensor,
    valids: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TD(lambda): bootstrap truncations, stop lambda across every boundary."""

    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        bootstrap = (1.0 - terminations[step]) * valids[step]
        lambda_continue = 1.0 - boundaries[step]
        delta = rewards[step] + gamma * next_values[step] * bootstrap - values[step]
        accumulator = delta + gamma * gae_lambda * lambda_continue * accumulator
        advantages[step] = accumulator
    return advantages, advantages + values


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.actor_dist != "beta" or not args.share_backbone:
        raise ValueError("v3 requires pure-TPO-v5's shared Beta ThinkTrunk")
    if (args.hidden, args.k_blocks, args.n_experts, args.num_bins) != (
        64,
        3,
        16,
        511,
    ):
        raise ValueError("v3 freezes the 64x3x16 ThinkTrunk and 511-bin critic")
    if args.pc_inference_steps != PC_INFERENCE_STEPS:
        raise ValueError("v3 requires exactly ten reverse-GS sweeps")
    if args.outer_cycles != OUTER_CYCLES:
        raise ValueError("v3 requires exactly ten outer generalized-EM cycles")
    if args.pc_chunk_size <= 0:
        raise ValueError("pc_chunk_size must be positive")
    if args.batch_size % args.pc_chunk_size:
        raise ValueError("compiled v3 requires batch_size divisible by pc_chunk_size")
    if args.tpo_k < 2:
        raise ValueError("TPO requires at least two candidates")
    if args.ent_coef != 0.0 or args.auto_entropy:
        raise ValueError("v3 fixes entropy regularization off")
    if args.tpo_coef <= 0.0 or args.vf_coef <= 0.0:
        raise ValueError("actor and critic terminal energy coefficients must be positive")
    if args.normalize_reward or args.clip_reward:
        raise ValueError("probe scores and critic targets must remain in raw reward units")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Ampere+ tensor-core matmuls avoid the TF32-disabled compile path while all
    # model tensors stay float32 and sufficient statistics stay float64.
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                index,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("only continuous action spaces are supported")

    agent = Agent(envs, args).to(device)
    if any(parameter.requires_grad for parameter in agent.task_parameters()):
        raise RuntimeError("task parameters must remain outside autograd")
    settle_core = lambda observation, candidates, q_target, value_target: _settle_dag_core(
        agent,
        observation,
        candidates,
        q_target,
        value_target,
        args.pc_inference_steps,
        args.tpo_coef,
        args.vf_coef,
    )
    if args.compile:
        settle_core = torch.compile(
            settle_core,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )

    support_min, support_max = value_support_bounds(args)
    hl_support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )

    def value_logits_to_scalar(logits: torch.Tensor) -> torch.Tensor:
        return hl_support.to_expected_scalar(logits)

    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    observations = torch.zeros((args.num_steps, args.num_envs) + observation_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape, device=device)
    latent_zs = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    values = torch.zeros_like(logprobs)
    transition_terminations = torch.zeros_like(logprobs)
    transition_boundaries = torch.zeros_like(logprobs)
    transition_valids = torch.ones_like(logprobs)
    next_transition_values = torch.zeros_like(logprobs)

    candidate_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + action_shape,
        device=device,
    )
    candidate_logprobs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_k), device=device
    )
    candidate_next_obs = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k) + observation_shape,
        dtype=np.float32,
    )
    candidate_rewards = np.zeros(
        (args.num_steps, args.num_envs, args.tpo_k), dtype=np.float32
    )
    candidate_terminations = np.zeros_like(candidate_rewards)

    probe_base_envs = [env.unwrapped for env in envs.envs]
    probe_observation_wrappers = [
        find_wrapper(env, gym.wrappers.NormalizeObservation) for env in envs.envs
    ]
    if any(wrapper is None for wrapper in probe_observation_wrappers):
        raise RuntimeError("NormalizeObservation wrapper not found")
    probe_action_low = envs.single_action_space.low
    probe_action_high = envs.single_action_space.high
    probe_cpu_rng_state = None
    probe_cuda_rng_state = None

    global_step = 0
    start_time = time.time()
    next_observation_np, _ = envs.reset(seed=args.seed)
    next_observation = torch.as_tensor(
        next_observation_np, dtype=torch.float32, device=device
    )

    for iteration in range(1, args.num_iterations + 1):
        probe_seconds = 0.0
        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_observation
            with torch.no_grad():
                (
                    action,
                    latent,
                    log_probability,
                    _,
                    value_logits,
                    rollout_distribution,
                    rollout_to_action,
                    rollout_log_det,
                ) = agent.get_action_and_value(next_observation, return_dist=True)
                values[step] = value_logits_to_scalar(value_logits)
            actions[step] = action
            latent_zs[step] = latent
            logprobs[step] = log_probability

            probe_start = time.time()
            with torch.no_grad():
                with torch.random.fork_rng(devices=[device]):
                    if probe_cpu_rng_state is None:
                        torch.manual_seed(args.seed + 1_000_003)
                    else:
                        torch.set_rng_state(probe_cpu_rng_state)
                        torch.cuda.set_rng_state(probe_cuda_rng_state, device)
                    sampled_candidates = rollout_distribution.sample(
                        torch.Size([args.tpo_k])
                    )
                    probe_cpu_rng_state = torch.get_rng_state()
                    probe_cuda_rng_state = torch.cuda.get_rng_state(device)
                sampled_candidates = sampled_candidates.permute(1, 0, 2).contiguous()
                if args.actor_dist == "beta":
                    sampled_candidates.clamp_(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                sampled_candidates[:, 0] = latent
                transposed_candidates = sampled_candidates.transpose(0, 1)
                sampled_logprobs = (
                    rollout_distribution.log_prob(transposed_candidates)
                    - rollout_log_det(transposed_candidates)
                ).sum(-1).transpose(0, 1)
                candidate_zs[step] = sampled_candidates
                candidate_logprobs[step] = sampled_logprobs
                candidate_actions_np = rollout_to_action(sampled_candidates).cpu().numpy()
            candidate_actions_np = np.clip(
                candidate_actions_np, probe_action_low, probe_action_high
            )
            for env_index in range(args.num_envs):
                base_env = probe_base_envs[env_index]
                observation_rms = probe_observation_wrappers[env_index].obs_rms
                saved_qpos = base_env.data.qpos.copy()
                saved_qvel = base_env.data.qvel.copy()
                saved_warmstart = base_env.data.qacc_warmstart.copy()
                saved_time = base_env.data.time
                for candidate_index in range(args.tpo_k):
                    base_env.data.qpos[:] = saved_qpos
                    base_env.data.qvel[:] = saved_qvel
                    base_env.data.qacc_warmstart[:] = saved_warmstart
                    base_env.data.time = saved_time
                    (
                        probe_observation,
                        probe_reward,
                        probe_terminated,
                        _,
                        _,
                    ) = base_env.step(candidate_actions_np[env_index, candidate_index])
                    normalized_observation = (
                        (probe_observation - observation_rms.mean)
                        / np.sqrt(observation_rms.var + 1e-8)
                    ).astype(np.float32)
                    candidate_next_obs[step, env_index, candidate_index] = np.clip(
                        normalized_observation, -10.0, 10.0
                    )
                    candidate_rewards[step, env_index, candidate_index] = probe_reward
                    candidate_terminations[step, env_index, candidate_index] = float(
                        probe_terminated
                    )
                base_env.data.qpos[:] = saved_qpos
                base_env.data.qvel[:] = saved_qvel
                base_env.data.qacc_warmstart[:] = saved_warmstart
                base_env.data.time = saved_time
            probe_seconds += time.time() - probe_start

            next_observation_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            boundary = np.logical_or(terminated, truncated)
            valid = (~boundary).astype(np.float32)
            final_observations = infos.get("final_observation")
            final_mask = infos.get("_final_observation")
            if final_observations is not None:
                transition_next_observation = np.array(next_observation_np, copy=True)
                if final_mask is None:
                    final_mask = [item is not None for item in final_observations]
                for env_index, has_final in enumerate(final_mask):
                    if has_final and final_observations[env_index] is not None:
                        transition_next_observation[env_index] = final_observations[env_index]
                        valid[env_index] = 1.0
                    elif boundary[env_index]:
                        valid[env_index] = 0.0
            else:
                transition_next_observation = next_observation_np
            transition_next_tensor = torch.as_tensor(
                transition_next_observation, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                next_transition_values[step] = value_logits_to_scalar(
                    agent.get_value(transition_next_tensor)
                )
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            transition_terminations[step] = torch.as_tensor(
                terminated, dtype=torch.float32, device=device
            )
            transition_boundaries[step] = torch.as_tensor(
                boundary, dtype=torch.float32, device=device
            )
            transition_valids[step] = torch.as_tensor(
                valid, dtype=torch.float32, device=device
            )
            next_observation = torch.as_tensor(
                next_observation_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar(
                            "charts/episodic_return", episode_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            advantages, returns = gae_lambda_returns(
                rewards,
                values,
                next_transition_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            value_targets = hl_support.project(returns).reshape(-1, args.num_bins)
            td_residual = (
                rewards
                + args.gamma
                * next_transition_values
                * (1.0 - transition_terminations)
                * transition_valids
                - values
            )
            current_td_rms = td_residual.square().mean().sqrt()

            flat_probe_observations = torch.as_tensor(
                candidate_next_obs.reshape(-1, int(np.prod(observation_shape))),
                dtype=torch.float32,
                device=device,
            )
            probe_next_values = torch.cat(
                [
                    value_logits_to_scalar(agent.get_value(chunk))
                    for chunk in flat_probe_observations.split(65_536)
                ]
            ).reshape(args.batch_size, args.tpo_k)
            probe_rewards = torch.as_tensor(
                candidate_rewards.reshape(args.batch_size, args.tpo_k),
                dtype=torch.float32,
                device=device,
            )
            probe_terminations = torch.as_tensor(
                candidate_terminations.reshape(args.batch_size, args.tpo_k),
                dtype=torch.float32,
                device=device,
            )
            scores = probe_rewards + args.gamma * (
                1.0 - probe_terminations
            ) * probe_next_values
            tpo_target = build_tpo_target(
                candidate_logprobs.reshape(args.batch_size, args.tpo_k),
                scores,
                current_td_rms,
                args,
            )

        flat_observations = observations.reshape(
            (args.batch_size,) + observation_shape
        )
        flat_candidates = candidate_zs.reshape(
            (args.batch_size, args.tpo_k) + action_shape
        )
        flat_latents = latent_zs.reshape((args.batch_size,) + action_shape)
        flat_old_logprobs = logprobs.reshape(args.batch_size)
        # These are the immutable E-step evidence for all ten outer cycles.  In
        # particular, neither the critic-derived probe q nor TD(lambda) support
        # projection is rebuilt after a parameter correction.
        rollout_q_targets = tpo_target.probabilities
        rollout_value_targets = value_targets
        outer_cycle_results = run_dag_outer_gem(
            agent,
            flat_observations,
            flat_candidates,
            rollout_q_targets,
            rollout_value_targets,
            flat_latents,
            flat_old_logprobs,
            args.pc_chunk_size,
            settle_core,
        )
        final_cycle = outer_cycle_results[-1]
        final_diagnostics = final_cycle.diagnostics

        returns_numpy = returns.cpu().numpy().reshape(-1)
        values_numpy = values.cpu().numpy().reshape(-1)
        target_variance = np.var(returns_numpy)
        explained_variance = (
            np.nan
            if target_variance == 0
            else 1.0 - np.var(returns_numpy - values_numpy) / target_variance
        )
        edge_mass = float(value_targets[:, [0, -1]].sum(-1).mean())

        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/returns_mean", float(returns.mean()), global_step)
        writer.add_scalar("debug/returns_std", float(returns.std()), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        writer.add_scalar("debug/current_td_rms", float(current_td_rms), global_step)
        writer.add_scalar("debug/tpo_eta_solved", tpo_target.eta, global_step)
        writer.add_scalar(
            "debug/tpo_reverse_kl_achieved",
            tpo_target.achieved_reverse_kl,
            global_step,
        )
        writer.add_scalar("debug/tpo_reverse_kl_base", tpo_target.base_reverse_kl, global_step)
        writer.add_scalar("debug/tpo_cap_engaged", tpo_target.cap_engaged, global_step)
        writer.add_scalar("debug/tpo_score_std_mean", tpo_target.score_std_mean, global_step)
        writer.add_scalar("debug/tpo_score_std_p90", tpo_target.score_std_p90, global_step)
        writer.add_scalar("debug/tpo_score_scale", tpo_target.score_scale, global_step)
        for cycle in outer_cycle_results:
            cycle_number = cycle.cycle_index + 1
            diagnostics = cycle.diagnostics
            prefix = f"pc_outer/cycle_{cycle_number:02d}"
            writer.add_scalar(
                f"{prefix}/energy_free",
                float(diagnostics.energy_per_row[0]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/energy_settled",
                float(diagnostics.energy_per_row[-1]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_energy_free",
                float(diagnostics.actor_settlement_energy_per_row[0]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_energy_settled",
                float(diagnostics.actor_settlement_energy_per_row[-1]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_energy_free",
                float(diagnostics.critic_settlement_energy_per_row[0]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_energy_settled",
                float(diagnostics.critic_settlement_energy_per_row[-1]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_boundary_ce_before",
                float(diagnostics.actor_boundary_ce_before),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_boundary_ce_after",
                float(diagnostics.actor_boundary_ce_after),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_boundary_ce_before",
                float(diagnostics.critic_boundary_ce_before),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_boundary_ce_after",
                float(diagnostics.critic_boundary_ce_after),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/proposed_behavior_kl",
                float(diagnostics.proposed_behavior_kl),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/correction_norm",
                float(diagnostics.correction_norm),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/entry_correction_norm",
                float(diagnostics.factor_correction_norm["trunk.entry"]),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/min_cov_rank_fraction",
                float(diagnostics.min_rank_fraction),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/max_cov_condition",
                float(diagnostics.max_condition),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/local_sse_ratio",
                float(
                    diagnostics.local_sse_after.sum()
                    / diagnostics.local_sse_before.sum().clamp_min(1e-30)
                ),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_terminal_message_rms",
                float(diagnostics.actor_terminal_message_rms),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_terminal_message_rms",
                float(diagnostics.critic_terminal_message_rms),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/terminal_message_cosine",
                float(diagnostics.terminal_message_cosine),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/actor_terminal_energy_share",
                float(diagnostics.actor_terminal_energy_share),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/critic_terminal_energy_share",
                float(diagnostics.critic_terminal_energy_share),
                global_step,
            )
            for sweep in range(args.pc_inference_steps + 1):
                writer.add_scalar(
                    f"{prefix}/energy_sweep_{sweep}",
                    float(diagnostics.energy_per_row[sweep]),
                    global_step,
                )
                writer.add_scalar(
                    f"{prefix}/stationarity_sweep_{sweep}",
                    float(diagnostics.stationarity_rms[sweep]),
                    global_step,
                )
        writer.add_scalar(
            "pc/energy_free", float(final_diagnostics.energy_per_row[0]), global_step
        )
        writer.add_scalar(
            "pc/energy_settled", float(final_diagnostics.energy_per_row[-1]), global_step
        )
        writer.add_scalar(
            "pc/stationarity_rms",
            float(final_diagnostics.stationarity_rms[-1]),
            global_step,
        )
        writer.add_scalar(
            "pc/proposed_behavior_kl",
            float(final_diagnostics.proposed_behavior_kl),
            global_step,
        )
        writer.add_scalar(
            "pc/parameter_correction_norm",
            float(final_diagnostics.correction_norm),
            global_step,
        )
        writer.add_scalar(
            "pc/local_sse_ratio",
            float(
                final_diagnostics.local_sse_after.sum()
                / final_diagnostics.local_sse_before.sum().clamp_min(1e-30)
            ),
            global_step,
        )
        for factor_name in sorted(final_diagnostics.factor_rank_fraction):
            factor_prefix = f"pc_final_factor/{factor_name.replace('.', '/')}"
            writer.add_scalar(
                f"{factor_prefix}/rank_fraction",
                float(final_diagnostics.factor_rank_fraction[factor_name]),
                global_step,
            )
            writer.add_scalar(
                f"{factor_prefix}/condition",
                float(final_diagnostics.factor_condition[factor_name]),
                global_step,
            )
            writer.add_scalar(
                f"{factor_prefix}/correction_norm",
                float(final_diagnostics.factor_correction_norm[factor_name]),
                global_step,
            )
        writer.add_scalar("charts/probe_seconds", probe_seconds, global_step)
        steps_per_second = int(global_step / (time.time() - start_time))
        print("SPS:", steps_per_second)
        writer.add_scalar("charts/SPS", steps_per_second, global_step)

    envs.close()
    writer.close()
