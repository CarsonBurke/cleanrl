# Grounded Recurrent Search v1
#
# Learn a probabilistic ensemble from ordinary raw transitions, then spend inference
# compute on persistent-member, action-conditioned search.  Search actions control the
# environment and supervise a member-conditioned proposal; values use exact completed-
# episode returns, while proposal distillation uses only the current search.  This
# first version makes no performance claim before measurement.

import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter


LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    buffer_size: int = 1_000_000
    model_learning_starts: int = 4_096
    planning_starts: int = 32_768
    batch_size: int = 256
    heldout_fraction: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.5
    updates_per_vector_step: int = 16

    ensemble_size: int = 4
    hidden_dim: int = 512
    bottleneck_dim: int = 128
    world_depth: int = 4
    value_depth: int = 3
    proposal_depth: int = 3
    std_floor: float = 1e-4

    root_candidates: int = 256
    beam_width: int = 4
    branch_factor: int = 4
    search_depth: int = 16
    elite_roots: int = 8

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    log_frequency: int = 10_000
    diagnostic_batch_size: int = 128
    tail_window: int = 512


def make_env(env_id: str, index: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and index == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def factual_next_observations(
    autoreset_observations: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Recover the observation produced by the action, before vector autoreset."""
    result = np.array(autoreset_observations, copy=True)
    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    if not np.any(done):
        return result
    final = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final is None:
        raise RuntimeError("finished transition is missing infos['final_observation']")
    for env_index in np.flatnonzero(done):
        if final_mask is not None and not bool(final_mask[env_index]):
            raise RuntimeError(f"environment {env_index} is missing its final observation")
        if final[env_index] is None:
            raise RuntimeError(f"environment {env_index} has a null final observation")
        result[env_index] = np.asarray(final[env_index], dtype=result.dtype)
    return result


def map_normalized_action(
    normalized_action: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
):
    """Affine map from the planner's [-1, 1] coordinates to the environment box."""
    return low + (normalized_action + 1.0) * 0.5 * (high - low)


def update_episode_members(
    members: torch.Tensor,
    done: torch.Tensor,
    generator: torch.Generator,
    ensemble_size: int,
) -> torch.Tensor:
    """Keep the online ensemble member fixed except at an observed episode boundary."""
    replacements = torch.randint(
        ensemble_size,
        members.shape,
        generator=generator,
        device=members.device,
    )
    return torch.where(done, replacements, members)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise negative log likelihood under a diagonal Gaussian."""
    return torch.log(std) + 0.5 * ((target - mean) / std).square() + 0.5 * LOG_2PI


class EnsembleLinear(nn.Module):
    """Independent linear layers stored in one tensor for fused ensemble execution."""

    def __init__(self, members: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.members = members
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(members, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(members, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for member in range(self.members):
            nn.init.orthogonal_(self.weight[member])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward_all(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[0] != self.members:
            raise ValueError("all-member inputs must have shape [members, batch, features]")
        outputs = torch.einsum("mbi,moi->mbo", inputs, self.weight)
        if self.bias is not None:
            outputs = outputs + self.bias[:, None]
        return outputs

    def forward_selected(self, inputs: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] != members.shape[0]:
            raise ValueError("the leading input and member dimensions must match")
        weight = self.weight[members]
        outputs = torch.einsum("e...i,eoi->e...o", inputs, weight)
        if self.bias is not None:
            bias_shape = (members.shape[0],) + (1,) * (inputs.ndim - 2) + (self.out_features,)
            outputs = outputs + self.bias[members].view(bias_shape)
        return outputs


class EnsembleResidualBlock(nn.Module):
    def __init__(self, members: int, width: int, bottleneck: int):
        super().__init__()
        self.down = EnsembleLinear(members, width, bottleneck)
        self.up = EnsembleLinear(members, bottleneck, width)

    def forward_all(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.rms_norm(inputs, (inputs.shape[-1],))
        hidden = F.silu(self.down.forward_all(hidden))
        return inputs + self.up.forward_all(hidden) / math.sqrt(2.0)

    def forward_selected(self, inputs: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        hidden = F.rms_norm(inputs, (inputs.shape[-1],))
        hidden = F.silu(self.down.forward_selected(hidden, members))
        return inputs + self.up.forward_selected(hidden, members) / math.sqrt(2.0)


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        members: int,
        input_dim: int,
        output_dim: int,
        width: int,
        bottleneck: int,
        depth: int,
    ):
        super().__init__()
        self.stem = EnsembleLinear(members, input_dim, width)
        self.blocks = nn.ModuleList(
            [EnsembleResidualBlock(members, width, bottleneck) for _ in range(depth)]
        )
        self.head = EnsembleLinear(members, width, output_dim)
        with torch.no_grad():
            self.head.weight.mul_(0.01)

    def forward_all(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.stem.forward_all(inputs))
        for block in self.blocks:
            hidden = block.forward_all(hidden)
        return self.head.forward_all(F.rms_norm(hidden, (hidden.shape[-1],)))

    def forward_selected(self, inputs: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.stem.forward_selected(inputs, members))
        for block in self.blocks:
            hidden = block.forward_selected(hidden, members)
        return self.head.forward_selected(F.rms_norm(hidden, (hidden.shape[-1],)), members)


class WorldEnsemble(nn.Module):
    """One-step raw observation residual, reward, and true-termination likelihood."""

    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.obs_dim = obs_dim
        self.std_floor = args.std_floor
        output_dim = 2 * obs_dim + 3
        self.network = EnsembleMLP(
            args.ensemble_size,
            obs_dim + action_dim,
            output_dim,
            args.hidden_dim,
            args.bottleneck_dim,
            args.world_depth,
        )

    def _decode(self, raw: torch.Tensor):
        d = self.obs_dim
        delta_mean = raw[..., :d]
        delta_std = F.softplus(raw[..., d : 2 * d]) + self.std_floor
        reward_mean = raw[..., 2 * d]
        reward_std = F.softplus(raw[..., 2 * d + 1]) + self.std_floor
        termination_logit = raw[..., 2 * d + 2]
        return delta_mean, delta_std, reward_mean, reward_std, termination_logit

    def predict_all(self, obs: torch.Tensor, action: torch.Tensor):
        inputs = torch.cat([obs, action], dim=-1)
        return self._decode(self.network.forward_all(inputs))

    def predict_selected(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        members: torch.Tensor,
    ):
        inputs = torch.cat([obs, action], dim=-1)
        return self._decode(self.network.forward_selected(inputs, members))


class MCValueEnsemble(nn.Module):
    """Member-specific return rate fitted only to completed factual episodes."""

    def __init__(self, obs_dim: int, args: Args):
        super().__init__()
        self.network = EnsembleMLP(
            args.ensemble_size,
            obs_dim + 1,
            1,
            args.hidden_dim,
            args.bottleneck_dim,
            args.value_depth,
        )

    def predict_all(self, obs: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([obs, remaining.unsqueeze(-1)], dim=-1)
        return self.network.forward_all(inputs).squeeze(-1)

    def predict_selected(
        self, obs: torch.Tensor, remaining: torch.Tensor, members: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([obs, remaining.unsqueeze(-1)], dim=-1)
        return self.network.forward_selected(inputs, members).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.down(F.rms_norm(inputs, (inputs.shape[-1],))))
        return inputs + self.up(hidden) / math.sqrt(2.0)


class Proposal(nn.Module):
    """A Gaussian search proposal conditioned on the persistent online member."""

    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.members = args.ensemble_size
        self.action_dim = action_dim
        self.std_floor = args.std_floor
        self.stem = nn.Linear(obs_dim + 1 + self.members, args.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(args.hidden_dim, args.bottleneck_dim) for _ in range(args.proposal_depth)]
        )
        self.head = nn.Linear(args.hidden_dim, 2 * action_dim)
        nn.init.orthogonal_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, obs: torch.Tensor, remaining: torch.Tensor, members: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(members, self.members).to(dtype=obs.dtype)
        one_hot_shape = (obs.shape[0],) + (1,) * (obs.ndim - 2) + (self.members,)
        one_hot = one_hot.view(one_hot_shape).expand(obs.shape[:-1] + (self.members,))
        inputs = torch.cat([obs, remaining.unsqueeze(-1), one_hot], dim=-1)
        hidden = F.silu(self.stem(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(F.rms_norm(hidden, (hidden.shape[-1],)))
        mean, raw_scale = raw.chunk(2, dim=-1)
        return mean, F.softplus(raw_scale) + self.std_floor


def world_nll_loss(
    world: WorldEnsemble,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    bootstrap_weight: torch.Tensor,
):
    delta_mean, delta_std, reward_mean, reward_std, termination_logit = world.predict_all(obs, action)
    obs_nll = gaussian_nll(next_obs - obs, delta_mean, delta_std).sum(dim=-1)
    reward_nll = gaussian_nll(reward, reward_mean, reward_std)
    termination_nll = F.binary_cross_entropy_with_logits(
        termination_logit, terminated, reduction="none"
    )
    joint = obs_nll + reward_nll + termination_nll
    denominators = bootstrap_weight.sum(dim=1)
    active = denominators > 0
    safe_denominators = torch.where(active, denominators, torch.ones_like(denominators))

    def active_mean(per_sample: torch.Tensor) -> torch.Tensor:
        per_member = (per_sample * bootstrap_weight).sum(dim=1) / safe_denominators
        active_float = active.to(per_sample.dtype)
        return (per_member * active_float).sum() / active_float.sum().clamp_min(1.0)

    total = active_mean(joint)
    obs_metric = active_mean(obs_nll)
    reward_metric = active_mean(reward_nll)
    termination_metric = active_mean(termination_nll)
    obs_floor_fraction = (delta_std <= 1.01 * world.std_floor).to(obs.dtype).mean()
    reward_floor_fraction = (reward_std <= 1.01 * world.std_floor).to(obs.dtype).mean()
    return (
        total,
        obs_metric,
        reward_metric,
        termination_metric,
        obs_floor_fraction,
        reward_floor_fraction,
    )


def value_mse_loss(
    value: MCValueEnsemble,
    obs: torch.Tensor,
    remaining: torch.Tensor,
    factual_return: torch.Tensor,
    supervision_mask: torch.Tensor,
    max_episode_steps: int,
):
    predicted_rate = value.predict_all(obs, remaining)
    factual_return = torch.where(
        supervision_mask.bool(), factual_return, torch.zeros_like(factual_return)
    )
    factual_rate = factual_return / max_episode_steps
    per_sample = 0.5 * (predicted_rate - factual_rate).square()
    denominator = supervision_mask.sum()
    loss = (per_sample * supervision_mask).sum() / denominator.clamp_min(1.0)
    return loss, predicted_rate


def proposal_elite_nll(
    proposal: Proposal,
    obs: torch.Tensor,
    remaining: torch.Tensor,
    members: torch.Tensor,
    elite_pre_tanh: torch.Tensor,
):
    mean, std = proposal(obs, remaining, members)
    nll = gaussian_nll(elite_pre_tanh, mean.unsqueeze(-2), std.unsqueeze(-2)).sum(dim=-1)
    floor_fraction = (std <= 1.01 * proposal.std_floor).to(obs.dtype).mean()
    entropy = (torch.log(std) + 0.5 * (1.0 + LOG_2PI)).sum(dim=-1).mean()
    return nll.mean(), mean, std, floor_fraction, entropy


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather dimension two from [env, root, candidate, ...] tensors."""
    suffix = values.shape[3:]
    gather_index = indices.view(indices.shape + (1,) * len(suffix)).expand(indices.shape + suffix)
    return torch.gather(values, 2, gather_index)


class DistinctRootPlanner(nn.Module):
    """Static lookahead that never lets one root consume another root's beam slots."""

    def __init__(
        self,
        world: WorldEnsemble,
        value: MCValueEnsemble,
        proposal: Proposal,
        max_episode_steps: int,
        root_candidates: int,
        beam_width: int,
        branch_factor: int,
        depth: int,
        elite_roots: int,
    ):
        super().__init__()
        if root_candidates < elite_roots:
            raise ValueError("root_candidates must be at least elite_roots")
        if depth < 1 or beam_width < 1 or branch_factor < 1:
            raise ValueError("search dimensions must be positive")
        self.world = world
        self.value = value
        self.proposal = proposal
        self.max_episode_steps = max_episode_steps
        self.root_candidates = root_candidates
        self.beam_width = beam_width
        self.branch_factor = branch_factor
        self.depth = depth
        self.elite_roots = elite_roots

    def _step(
        self,
        obs: torch.Tensor,
        remaining: torch.Tensor,
        action: torch.Tensor,
        members: torch.Tensor,
    ):
        delta, _, reward, _, termination_logit = self.world.predict_selected(obs, action, members)
        next_obs = obs + delta
        next_remaining = torch.maximum(
            remaining - 1.0 / self.max_episode_steps,
            torch.zeros((), dtype=remaining.dtype, device=remaining.device),
        )
        continuation = torch.sigmoid(-termination_logit) * (next_remaining > 0).to(obs.dtype)
        return next_obs, next_remaining, reward, continuation

    def _proposal_mean_rollout(
        self, obs: torch.Tensor, remaining: torch.Tensor, members: torch.Tensor
    ) -> torch.Tensor:
        cumulative = torch.zeros_like(remaining)
        survival = torch.ones_like(remaining)
        state = obs
        time_left = remaining
        for _ in range(self.depth):
            mean, _ = self.proposal(state, time_left, members)
            action = torch.tanh(mean)
            state, time_left, reward, continuation = self._step(
                state, time_left, action, members
            )
            cumulative = cumulative + survival * reward
            survival = survival * continuation
        return cumulative + survival * self.max_episode_steps * self.value.predict_selected(
            state, time_left, members
        )

    def forward(
        self,
        obs: torch.Tensor,
        remaining: torch.Tensor,
        members: torch.Tensor,
        root_noise: torch.Tensor,
        branch_noise: torch.Tensor,
    ):
        root_mean, root_std = self.proposal(obs, remaining, members)
        root_pre_tanh = root_mean[:, None] + root_std[:, None] * root_noise
        root_action = torch.tanh(root_pre_tanh)
        root_obs = obs[:, None].expand(-1, self.root_candidates, -1)
        root_remaining = remaining[:, None].expand(-1, self.root_candidates)
        state, time_left, first_reward, continuation = self._step(
            root_obs, root_remaining, root_action, members
        )
        cumulative = first_reward
        survival = continuation
        leaf_value = self.max_episode_steps * self.value.predict_selected(
            state, time_left, members
        )
        scores = cumulative + survival * leaf_value

        state = state.unsqueeze(2)
        time_left = time_left.unsqueeze(2)
        cumulative = cumulative.unsqueeze(2)
        survival = survival.unsqueeze(2)
        scores = scores.unsqueeze(2)
        active_beams = 1

        for depth_index in range(1, self.depth):
            mean, std = self.proposal(state, time_left, members)
            noise = branch_noise[:, depth_index - 1, :, :active_beams]
            child_pre_tanh = mean.unsqueeze(3) + std.unsqueeze(3) * noise
            child_action = torch.tanh(child_pre_tanh)
            expanded_state = state.unsqueeze(3).expand_as(
                child_action[..., :1].expand(child_action.shape[:-1] + (state.shape[-1],))
            )
            expanded_time = time_left.unsqueeze(3).expand(child_action.shape[:-1])
            next_state, next_time, reward, continuation = self._step(
                expanded_state, expanded_time, child_action, members
            )
            child_cumulative = cumulative.unsqueeze(3) + survival.unsqueeze(3) * reward
            child_survival = survival.unsqueeze(3) * continuation
            child_value = self.max_episode_steps * self.value.predict_selected(
                next_state, next_time, members
            )
            child_score = child_cumulative + child_survival * child_value

            candidate_count = active_beams * self.branch_factor
            next_state = next_state.flatten(2, 3)
            next_time = next_time.flatten(2, 3)
            child_cumulative = child_cumulative.flatten(2, 3)
            child_survival = child_survival.flatten(2, 3)
            child_score = child_score.flatten(2, 3)
            retained = min(self.beam_width, candidate_count)
            _, best_indices = torch.topk(child_score, retained, dim=2, sorted=False)
            state = _gather_candidates(next_state, best_indices)
            time_left = _gather_candidates(next_time, best_indices)
            cumulative = _gather_candidates(child_cumulative, best_indices)
            survival = _gather_candidates(child_survival, best_indices)
            scores = _gather_candidates(child_score, best_indices)
            active_beams = retained

        root_scores, best_beam_indices = scores.max(dim=2)
        root_prefix = torch.gather(
            cumulative, 2, best_beam_indices.unsqueeze(-1)
        ).squeeze(-1)
        root_survival = torch.gather(
            survival, 2, best_beam_indices.unsqueeze(-1)
        ).squeeze(-1)
        root_tail = root_scores - root_prefix
        elite_scores, elite_indices = torch.topk(
            root_scores, self.elite_roots, dim=1, sorted=True
        )
        elite_pre_tanh = torch.gather(
            root_pre_tanh,
            1,
            elite_indices.unsqueeze(-1).expand(-1, -1, root_pre_tanh.shape[-1]),
        )
        best_index = elite_indices[:, 0]
        best_action = torch.gather(
            root_action, 1, best_index[:, None, None].expand(-1, 1, root_action.shape[-1])
        ).squeeze(1)
        chosen_first_reward = torch.gather(first_reward, 1, best_index[:, None]).squeeze(1)
        chosen_survival = torch.gather(root_survival, 1, best_index[:, None]).squeeze(1)
        proposal_mean_score = self._proposal_mean_rollout(obs, remaining, members)
        action_change = (best_action - torch.tanh(root_mean)).square().sum(dim=-1).sqrt()
        elite_action_std = torch.tanh(elite_pre_tanh).std(dim=1, correction=0).mean(dim=-1)
        prefix_std = root_prefix.std(dim=1, correction=0)
        tail_std = root_tail.std(dim=1, correction=0)
        total_std = root_scores.std(dim=1, correction=0)
        reward_ranks = torch.argsort(torch.argsort(root_prefix, dim=1), dim=1).to(obs.dtype)
        total_ranks = torch.argsort(torch.argsort(root_scores, dim=1), dim=1).to(obs.dtype)
        reward_ranks = reward_ranks - reward_ranks.mean(dim=1, keepdim=True)
        total_ranks = total_ranks - total_ranks.mean(dim=1, keepdim=True)
        rank_denominator = reward_ranks.square().sum(dim=1).sqrt() * total_ranks.square().sum(
            dim=1
        ).sqrt()
        rank_correlation = torch.where(
            rank_denominator > 0,
            (reward_ranks * total_ranks).sum(dim=1) / rank_denominator,
            torch.zeros_like(rank_denominator),
        )
        top_root_agreement = (root_prefix.argmax(dim=1) == best_index).to(obs.dtype)
        prefix_variance = prefix_std.square()
        tail_variance = tail_std.square()
        tail_dominance_ratio = torch.where(
            prefix_variance + tail_variance > 0,
            tail_variance / (prefix_variance + tail_variance),
            torch.zeros_like(tail_variance),
        )
        return (
            best_action,
            elite_pre_tanh,
            elite_scores[:, 0],
            proposal_mean_score,
            prefix_std,
            tail_std,
            total_std,
            rank_correlation,
            top_root_agreement,
            root_tail.amin(dim=1),
            root_tail.amax(dim=1),
            tail_dominance_ratio,
            chosen_survival,
            action_change,
            elite_action_std,
            chosen_first_reward,
            root_scores,
            root_prefix,
            root_tail,
        )


class GPUVectorReplay:
    """CUDA ring with a vector-time axis and exact episode-finalized returns."""

    def __init__(
        self,
        total_capacity: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int,
        elite_roots: int,
        max_episode_steps: int,
        device: torch.device | str,
        seed: int = 0,
        heldout_fraction: float = 0.05,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.ensemble_size = ensemble_size
        self.max_episode_steps = max_episode_steps
        self.capacity = max(2, math.ceil(total_capacity / num_envs))
        shape = (self.capacity, num_envs)
        self.obs = torch.empty(shape + (obs_dim,), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty_like(self.obs)
        self.action = torch.empty(shape + (action_dim,), dtype=torch.float32, device=self.device)
        self.reward = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.truncated = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.remaining = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.member = torch.empty(shape, dtype=torch.long, device=self.device)
        self.bootstrap_weight = torch.empty(
            shape + (ensemble_size,), dtype=torch.float32, device=self.device
        )
        self.elite_pre_tanh = torch.empty(
            shape + (elite_roots, action_dim), dtype=torch.float32, device=self.device
        )
        self.predicted_score = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.planner_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.value_return = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.value_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.holdout = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.episode = torch.empty(shape, dtype=torch.long, device=self.device)
        self.absolute = torch.full(shape, -1, dtype=torch.long, device=self.device)

        self.total_vector_steps = 0
        self.pending: list[list[int]] = [[] for _ in range(num_envs)]
        self.bootstrap_rng = np.random.default_rng(seed + 31_337)
        self.episode_uid = np.arange(num_envs, dtype=np.int64)
        self.next_episode_uid = num_envs
        self.current_bootstrap = self.bootstrap_rng.poisson(
            1.0, size=(num_envs, ensemble_size)
        ).astype(np.float32)
        period = max(2, round(1.0 / heldout_fraction))
        self.current_holdout = self.episode_uid % period == 0
        self.holdout_period = period
        self.train_value_rows = np.zeros(ensemble_size, dtype=np.int64)
        self.heldout_value_rows = np.zeros(ensemble_size, dtype=np.int64)

    @property
    def size(self) -> int:
        return min(self.total_vector_steps, self.capacity) * self.num_envs

    @property
    def occupied_slots(self) -> int:
        return min(self.total_vector_steps, self.capacity)

    def _tensor(self, value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        next_obs: Any,
        terminated: Any,
        truncated: Any,
        remaining: Any,
        member: Any,
        elite_pre_tanh: Any,
        predicted_score: Any,
        planner_valid: Any,
    ) -> None:
        absolute = self.total_vector_steps
        slot = absolute % self.capacity
        for env_index, episode_steps in enumerate(self.pending):
            if episode_steps and episode_steps[0] == absolute - self.capacity:
                raise RuntimeError("replay capacity is shorter than an unfinished episode")

        self.obs[slot].copy_(self._tensor(obs, torch.float32))
        self.action[slot].copy_(self._tensor(action, torch.float32))
        self.reward[slot].copy_(self._tensor(reward, torch.float32))
        self.next_obs[slot].copy_(self._tensor(next_obs, torch.float32))
        self.terminated[slot].copy_(self._tensor(terminated, torch.float32))
        self.truncated[slot].copy_(self._tensor(truncated, torch.bool))
        self.remaining[slot].copy_(self._tensor(remaining, torch.float32))
        self.member[slot].copy_(self._tensor(member, torch.long))
        self.elite_pre_tanh[slot].copy_(self._tensor(elite_pre_tanh, torch.float32))
        self.predicted_score[slot].copy_(self._tensor(predicted_score, torch.float32))
        self.planner_valid[slot].copy_(self._tensor(planner_valid, torch.bool))
        self.bootstrap_weight[slot].copy_(self._tensor(self.current_bootstrap, torch.float32))
        self.holdout[slot].copy_(self._tensor(self.current_holdout, torch.bool))
        self.episode[slot].copy_(self._tensor(self.episode_uid, torch.long))
        self.absolute[slot].fill_(absolute)
        self.value_valid[slot].zero_()

        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        for env_index in range(self.num_envs):
            self.pending[env_index].append(absolute)
            if done[env_index]:
                self._finalize_episode(env_index)
                self.episode_uid[env_index] = self.next_episode_uid
                self.next_episode_uid += 1
                self.current_bootstrap[env_index] = self.bootstrap_rng.poisson(
                    1.0, size=self.ensemble_size
                )
                self.current_holdout[env_index] = (
                    self.episode_uid[env_index] % self.holdout_period == 0
                )
        self.total_vector_steps += 1

    def _finalize_episode(self, env_index: int) -> None:
        absolute_steps = self.pending[env_index]
        if not absolute_steps:
            raise RuntimeError("cannot finalize an empty episode")
        slots = torch.as_tensor(
            [step % self.capacity for step in absolute_steps],
            dtype=torch.long,
            device=self.device,
        )
        expected = torch.as_tensor(absolute_steps, dtype=torch.long, device=self.device)
        if not torch.equal(self.absolute[slots, env_index], expected):
            raise RuntimeError("unfinished episode was overwritten before return finalization")
        rewards = self.reward[slots, env_index]
        returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        self.value_return[slots, env_index] = returns
        self.value_valid[slots, env_index] = True
        model_index = int(self.member[slots[0], env_index].item())
        if bool(self.holdout[slots[0], env_index].item()):
            self.heldout_value_rows[model_index] += len(absolute_steps)
        else:
            self.train_value_rows[model_index] += len(absolute_steps)
        self.pending[env_index] = []

    def chronological(self, env_index: int, field: str) -> torch.Tensor:
        if self.occupied_slots == 0:
            return getattr(self, field)[:0, env_index]
        start = max(0, self.total_vector_steps - self.capacity)
        absolute = torch.arange(start, self.total_vector_steps, device=self.device)
        slots = absolute % self.capacity
        if not torch.equal(self.absolute[slots, env_index], absolute):
            raise RuntimeError("ring chronology invariant failed")
        return getattr(self, field)[slots, env_index]

    def _sample_indices(
        self, mask: torch.Tensor, count: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        occupied = self.occupied_slots
        if occupied == 0 or not bool(mask[:occupied].any()):
            raise RuntimeError("no eligible replay rows")
        selected_slots: list[torch.Tensor] = []
        selected_envs: list[torch.Tensor] = []
        collected = 0
        attempts = 0
        while collected < count:
            attempts += 1
            if attempts > 100:
                raise RuntimeError("eligible replay rows are too sparse to sample")
            proposal_count = max(64, 4 * (count - collected))
            slots = torch.randint(
                occupied, (proposal_count,), generator=generator, device=self.device
            )
            envs = torch.randint(
                self.num_envs, (proposal_count,), generator=generator, device=self.device
            )
            keep = mask[slots, envs]
            slots = slots[keep]
            envs = envs[keep]
            selected_slots.append(slots)
            selected_envs.append(envs)
            collected += slots.numel()
        return (
            torch.cat(selected_slots)[:count],
            torch.cat(selected_envs)[:count],
        )

    def _gather(self, slots: torch.Tensor, envs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs[slots, envs],
            "next_obs": self.next_obs[slots, envs],
            "action": self.action[slots, envs],
            "reward": self.reward[slots, envs],
            "terminated": self.terminated[slots, envs],
            "remaining": self.remaining[slots, envs],
            "member": self.member[slots, envs],
            "bootstrap_weight": self.bootstrap_weight[slots, envs],
            "elite_pre_tanh": self.elite_pre_tanh[slots, envs],
            "predicted_score": self.predicted_score[slots, envs],
            "value_return": self.value_return[slots, envs],
            "absolute": self.absolute[slots, envs],
        }

    def _uniform_indices(
        self,
        count: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.occupied_slots == 0:
            raise RuntimeError("replay is empty")
        return (
            torch.randint(
                self.occupied_slots, (count,), generator=generator, device=self.device
            ),
            torch.randint(
                self.num_envs, (count,), generator=generator, device=self.device
            ),
        )

    def sample_world(self, batch_size: int, generator: torch.Generator):
        rows = []
        for model_index in range(self.ensemble_size):
            slots, envs = self._uniform_indices(batch_size, generator)
            row = self._gather(slots, envs)
            row["training_weight"] = (
                row["bootstrap_weight"][:, model_index]
                * (~self.holdout[slots, envs]).to(torch.float32)
            )
            rows.append(row)
        return {
            key: torch.stack([row[key] for row in rows])
            for key in ("obs", "next_obs", "action", "reward", "terminated", "remaining")
        } | {
            "bootstrap_weight": torch.stack(
                [rows[m]["training_weight"] for m in range(self.ensemble_size)]
            )
        }

    def can_sample_value(self, heldout: bool = False) -> bool:
        counts = self.heldout_value_rows if heldout else self.train_value_rows
        return bool(np.all(counts > 0))

    def sample_value(
        self,
        batch_size: int,
        generator: torch.Generator,
        heldout: bool = False,
    ) -> dict[str, torch.Tensor]:
        rows = []
        for model_index in range(self.ensemble_size):
            if heldout:
                eligible = (
                    (self.absolute >= 0)
                    & self.value_valid
                    & self.holdout
                    & (self.member == model_index)
                )
                slots, envs = self._sample_indices(eligible, batch_size, generator)
            else:
                slots, envs = self._uniform_indices(batch_size, generator)
            row = self._gather(slots, envs)
            split = self.holdout[slots, envs] if heldout else ~self.holdout[slots, envs]
            row["supervision_mask"] = (
                self.value_valid[slots, envs]
                & split
                & (row["member"] == model_index)
            ).to(torch.float32)
            row["value_return"] = torch.where(
                row["supervision_mask"].bool(),
                row["value_return"],
                torch.zeros_like(row["value_return"]),
            )
            rows.append(row)
        return {
            "obs": torch.stack([row["obs"] for row in rows]),
            "remaining": torch.stack([row["remaining"] for row in rows]),
            "value_return": torch.stack([row["value_return"] for row in rows]),
            "supervision_mask": torch.stack([row["supervision_mask"] for row in rows]),
            "absolute": torch.stack([row["absolute"] for row in rows]),
        }

    def sample_prequential(self, batch_size: int, generator: torch.Generator):
        eligible = (self.absolute >= 0) & self.planner_valid & self.value_valid
        slots, envs = self._sample_indices(eligible, batch_size, generator)
        return self._gather(slots, envs)

    def sample_heldout(self, batch_size: int, generator: torch.Generator):
        eligible = (self.absolute >= 0) & self.holdout
        slots, envs = self._sample_indices(eligible, batch_size, generator)
        return self._gather(slots, envs)

    def sample_heldout_sequence(
        self, batch_size: int, horizon: int, generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        candidates: list[tuple[int, int]] = []
        earliest = max(0, self.total_vector_steps - self.capacity)
        latest = self.total_vector_steps - horizon
        if latest < earliest:
            raise RuntimeError("replay does not contain a full diagnostic sequence")
        attempts = 0
        while len(candidates) < batch_size:
            attempts += 1
            if attempts > 200:
                raise RuntimeError("not enough held-out within-episode sequences")
            proposal_count = max(64, 8 * (batch_size - len(candidates)))
            starts = torch.randint(
                earliest,
                latest + 1,
                (proposal_count,),
                generator=generator,
                device=self.device,
            )
            envs = torch.randint(
                self.num_envs,
                (proposal_count,),
                generator=generator,
                device=self.device,
            )
            offsets = torch.arange(horizon, device=self.device)
            absolute = starts[:, None] + offsets[None]
            slots = absolute % self.capacity
            gathered_absolute = self.absolute[slots, envs[:, None]]
            episodes = self.episode[slots, envs[:, None]]
            valid = (gathered_absolute == absolute).all(dim=1)
            valid &= (episodes == episodes[:, :1]).all(dim=1)
            valid &= self.holdout[slots[:, 0], envs]
            for start, env in zip(starts[valid].tolist(), envs[valid].tolist(), strict=True):
                candidates.append((start, env))
                if len(candidates) == batch_size:
                    break
        starts = torch.tensor([x[0] for x in candidates], device=self.device)
        envs = torch.tensor([x[1] for x in candidates], device=self.device)
        absolute = starts[:, None] + torch.arange(horizon, device=self.device)[None]
        slots = absolute % self.capacity
        return {
            "obs": self.obs[slots[:, 0], envs],
            "remaining": self.remaining[slots[:, 0], envs],
            "member": self.member[slots[:, 0], envs],
            "action": self.action[slots, envs[:, None]],
            "next_obs": self.next_obs[slots, envs[:, None]],
            "reward": self.reward[slots, envs[:, None]],
        }


@torch.no_grad()
def heldout_diagnostics(
    world: WorldEnsemble,
    value: MCValueEnsemble,
    replay: GPUVectorReplay,
    generator: torch.Generator,
    batch_size: int,
    max_episode_steps: int,
) -> dict[str, float]:
    batch = replay.sample_heldout(batch_size, generator)
    members = batch["member"]
    delta, delta_std, reward_mean, reward_std, termination_logit = world.predict_selected(
        batch["obs"], batch["action"], members
    )
    next_mean = batch["obs"] + delta
    metrics = {
        "heldout/obs_rmse_h1": float(F.mse_loss(next_mean, batch["next_obs"]).sqrt()),
        "heldout/reward_rmse_h1": float(F.mse_loss(reward_mean, batch["reward"]).sqrt()),
        "heldout/obs_std": float(delta_std.mean()),
        "heldout/reward_std": float(reward_std.mean()),
        "heldout/obs_scale_floor_fraction": float(
            (delta_std <= 1.01 * world.std_floor).float().mean()
        ),
        "heldout/reward_scale_floor_fraction": float(
            (reward_std <= 1.01 * world.std_floor).float().mean()
        ),
        "heldout/termination_brier": float(
            F.mse_loss(torch.sigmoid(termination_logit), batch["terminated"])
        ),
    }
    try:
        sequence = replay.sample_heldout_sequence(batch_size, 32, generator)
    except RuntimeError:
        sequence = None
    if sequence is not None:
        state = sequence["obs"]
        time_left = sequence["remaining"]
        members = sequence["member"]
        predicted_reward_sum = torch.zeros_like(time_left)
        factual_reward_sum = torch.zeros_like(time_left)
        for step in range(32):
            delta, _, predicted_reward, _, _ = world.predict_selected(
                state, sequence["action"][:, step], members
            )
            state = state + delta
            predicted_reward_sum = predicted_reward_sum + predicted_reward
            factual_reward_sum = factual_reward_sum + sequence["reward"][:, step]
            time_left = torch.maximum(
                time_left - 1.0 / max_episode_steps, torch.zeros_like(time_left)
            )
            horizon = step + 1
            if horizon in (1, 2, 4, 8, 16, 32):
                metrics[f"heldout/obs_rmse_h{horizon}"] = float(
                    F.mse_loss(state, sequence["next_obs"][:, step]).sqrt()
                )
                metrics[f"heldout/reward_rmse_h{horizon}"] = float(
                    F.mse_loss(predicted_reward, sequence["reward"][:, step]).sqrt()
                )
                factual_state = sequence["next_obs"][:, step]
                state_error = (state - factual_state).square().mean(dim=0)
                state_variance = factual_state.var(dim=0, correction=0)
                state_nmse = torch.where(
                    state_variance > 0,
                    state_error / state_variance,
                    torch.zeros_like(state_error),
                ).mean()
                metrics[f"heldout/obs_nmse_h{horizon}"] = float(state_nmse)
                cumulative_error = predicted_reward_sum - factual_reward_sum
                metrics[f"heldout/cumulative_reward_mae_h{horizon}"] = float(
                    cumulative_error.abs().mean()
                )
                predicted_centered = predicted_reward_sum - predicted_reward_sum.mean()
                factual_centered = factual_reward_sum - factual_reward_sum.mean()
                correlation_denominator = predicted_centered.square().sum().sqrt() * factual_centered.square().sum().sqrt()
                correlation = torch.where(
                    correlation_denominator > 0,
                    (predicted_centered * factual_centered).sum() / correlation_denominator,
                    torch.zeros_like(correlation_denominator),
                )
                metrics[f"heldout/cumulative_reward_corr_h{horizon}"] = float(correlation)
    if replay.can_sample_value(heldout=True):
        value_batch = replay.sample_value(batch_size, generator, heldout=True)
        predicted_rate = value.predict_all(value_batch["obs"], value_batch["remaining"])
        factual_rate = value_batch["value_return"] / max_episode_steps
        metrics["heldout/value_rate_rmse"] = float(
            F.mse_loss(predicted_rate, factual_rate).sqrt()
        )
        metrics["heldout/value_return_rmse"] = float(
            F.mse_loss(
                max_episode_steps * predicted_rate, value_batch["value_return"]
            ).sqrt()
        )
        predicted_return = max_episode_steps * predicted_rate
        factual_return = value_batch["value_return"]
        value_error = predicted_return - factual_return
        factual_variance = factual_return.var(correction=0)
        explained_variance = torch.where(
            factual_variance > 0,
            1.0 - value_error.var(correction=0) / factual_variance,
            torch.zeros_like(factual_variance),
        )
        metrics["heldout/value_return_bias"] = float(value_error.mean())
        metrics["heldout/value_explained_variance"] = float(explained_variance)
    return metrics


@torch.no_grad()
def prequential_diagnostics(
    replay: GPUVectorReplay,
    generator: torch.Generator,
    batch_size: int,
) -> dict[str, float]:
    """Compare scores saved before acting with the exact return revealed later."""
    batch = replay.sample_prequential(batch_size, generator)
    predicted = batch["predicted_score"]
    factual = batch["value_return"]
    error = predicted - factual
    predicted_centered = predicted - predicted.mean()
    factual_centered = factual - factual.mean()
    denominator = predicted_centered.square().sum().sqrt() * factual_centered.square().sum().sqrt()
    correlation = torch.where(
        denominator > 0,
        (predicted_centered * factual_centered).sum() / denominator,
        torch.zeros_like(denominator),
    )
    return {
        "planner/prequential_return_mae": float(error.abs().mean()),
        "planner/prequential_return_bias": float(error.mean()),
        "planner/prequential_return_correlation": float(correlation),
    }


def _clone_outputs(outputs):
    return tuple(output.clone() for output in outputs)


def is_evaluation_step(completed_steps: int) -> bool:
    return completed_steps in (100_000, 250_000, 500_000) or (
        completed_steps >= 1_000_000 and completed_steps % 1_000_000 == 0
    )


@torch.no_grad()
def evaluate_search(
    planner_fn,
    envs: gym.vector.SyncVectorEnv,
    args: Args,
    obs_dim: int,
    action_dim: int,
    max_episode_steps: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """One fixed-seed search episode per eval env; proposal-only evaluation is invalid."""
    if envs.num_envs != args.num_envs:
        raise ValueError("evaluation must retain the planner's static environment dimension")
    eval_generator = torch.Generator(device=device).manual_seed(args.seed + 4_000_003)
    observations, _ = envs.reset(
        seed=[args.seed + 50_000 + index for index in range(args.num_envs)]
    )
    if observations.shape != (args.num_envs, obs_dim):
        raise RuntimeError("evaluation observation shape changed")
    episode_steps = np.zeros(args.num_envs, dtype=np.int64)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    active = np.ones(args.num_envs, dtype=bool)
    members = torch.arange(args.num_envs, device=device) % args.ensemble_size
    for _ in range(max_episode_steps):
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        remaining = torch.as_tensor(
            1.0 - episode_steps.astype(np.float32) / max_episode_steps, device=device
        )
        root_noise = torch.randn(
            (args.num_envs, args.root_candidates, action_dim),
            generator=eval_generator,
            device=device,
        )
        branch_noise = torch.randn(
            (
                args.num_envs,
                args.search_depth - 1,
                args.root_candidates,
                args.beam_width,
                args.branch_factor,
                action_dim,
            ),
            generator=eval_generator,
            device=device,
        )
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        outputs = _clone_outputs(
            planner_fn(obs_tensor, remaining, members, root_noise, branch_noise)
        )
        env_action = map_normalized_action(outputs[0], action_low, action_high)
        torch._assert_async(
            torch.isfinite(env_action).all(), "nonfinite action reached evaluation environment"
        )
        next_observations, rewards, terminated, truncated, _ = envs.step(
            env_action.cpu().numpy()
        )
        episode_returns += np.asarray(rewards) * active
        finished = (np.asarray(terminated) | np.asarray(truncated)) & active
        active &= ~finished
        episode_steps = np.where(active, episode_steps + 1, 0)
        observations = next_observations
        if not active.any():
            return episode_returns
    if active.any():
        raise RuntimeError("evaluation exceeded the declared episode horizon")
    return episode_returns


def main():
    args = tyro.cli(Args)
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Grounded Recurrent Search requires CUDA")
    if args.num_envs != 16:
        raise ValueError("v1 is designed and compiled for exactly 16 vector environments")
    if args.ensemble_size != 4:
        raise ValueError("v1 requires four independently parameterized ensemble members")
    if args.root_candidates != 256 or args.beam_width != 4 or args.branch_factor != 4:
        raise ValueError("v1 fixes search at 256 distinct roots, beam width 4, branch factor 4")
    if args.search_depth != 16 or args.elite_roots != 8:
        raise ValueError("v1 fixes depth 16 and eight elite roots")
    if args.batch_size != 256 or args.updates_per_vector_step != 16:
        raise ValueError("v1 fixes batch 256 and 16 learner updates per vector step (UTD 1)")
    if args.model_learning_starts % args.num_envs or args.planning_starts % args.num_envs:
        raise ValueError("learning boundaries must align with complete vector steps")
    if args.planning_starts <= args.model_learning_starts:
        raise ValueError("planning must follow a nonempty model-only pretraining phase")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|parameter|value|\n|-|-|\n" + "\n".join(
        f"|{key}|{value}|" for key, value in asdict(args).items()
    ))
    writer.add_text(
        "method/structure",
        "raw factual transition likelihood; exact undiscounted completed-episode value; "
        "four persistent online-member labels (weights remain online, so this is not exact "
        "posterior sampling); 256 distinct roots x beam4 x branch4 x depth16; eight-root "
        "online proposal distillation; batch256 x 16 updates/vector-step = UTD1, 256 "
        "replay rows/member/real transition and 1024 member-row uses/real transition",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name) for index in range(args.num_envs)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, False, run_name + "_eval") for index in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("continuous Box actions are required")
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if max_episode_steps is None:
        raise RuntimeError("the environment must expose a finite TimeLimit")

    world = WorldEnsemble(obs_dim, action_dim, args).to(device)
    value = MCValueEnsemble(obs_dim, args).to(device)
    proposal = Proposal(obs_dim, action_dim, args).to(device)
    planner = DistinctRootPlanner(
        world,
        value,
        proposal,
        max_episode_steps,
        args.root_candidates,
        args.beam_width,
        args.branch_factor,
        args.search_depth,
        args.elite_roots,
    )
    world_optimizer = torch.optim.AdamW(
        world.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    value_optimizer = torch.optim.AdamW(
        value.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    proposal_optimizer = torch.optim.AdamW(
        proposal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    world_loss_fn = world_nll_loss
    value_loss_fn = value_mse_loss
    proposal_loss_fn = proposal_elite_nll
    planner_fn = planner
    if args.compile:
        world_loss_fn = torch.compile(
            world_loss_fn, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        value_loss_fn = torch.compile(
            value_loss_fn, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        proposal_loss_fn = torch.compile(
            proposal_loss_fn, mode=args.compile_mode, dynamic=False, fullgraph=True
        )
        planner_fn = torch.compile(
            planner_fn, mode=args.compile_mode, dynamic=False, fullgraph=True
        )

    replay = GPUVectorReplay(
        args.buffer_size,
        args.num_envs,
        obs_dim,
        action_dim,
        args.ensemble_size,
        args.elite_roots,
        max_episode_steps,
        device,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
    )
    behavior_generator = torch.Generator(device=device).manual_seed(args.seed + 1_000_003)
    training_generator = torch.Generator(device=device).manual_seed(args.seed + 2_000_003)
    diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed + 3_000_003)

    observations, _ = envs.reset(seed=[args.seed + index for index in range(args.num_envs)])
    episode_steps = np.zeros(args.num_envs, dtype=np.int64)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    recent_returns: deque[float] = deque(maxlen=args.tail_window)
    members = torch.arange(args.num_envs, device=device) % args.ensemble_size
    members = members[
        torch.randperm(args.num_envs, generator=behavior_generator, device=device)
    ]
    member_counts = torch.zeros(args.ensemble_size, dtype=torch.long, device=device)
    action_low = torch.as_tensor(envs.single_action_space.low, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device)
    start_time = time.time()
    search_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    model_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    search_candidates_since_log = 0
    model_member_rows_since_log = 0
    planner_metric_names = (
        "improvement",
        "reward_prefix_std",
        "tail_value_std",
        "total_score_std",
        "reward_total_rank_correlation",
        "reward_total_top1_agreement",
        "tail_value_min",
        "tail_value_max",
        "tail_dominance_ratio",
        "h16_survival",
        "action_change",
        "action_saturation_fraction",
        "elite_std",
        "predicted_reward",
        "chosen_reward_absolute_error",
        "chosen_reward_optimism_bias",
    )
    planner_metric_sums = {
        key: torch.zeros((), device=device) for key in planner_metric_names
    }
    planner_metric_count = 0
    world_update_count = 0
    value_update_count = 0
    proposal_update_count = 0
    expected_world_updates_at_takeover = (
        (args.planning_starts - args.model_learning_starts) // args.num_envs + 1
    ) * args.updates_per_vector_step
    takeover_checked = False
    last_metrics: dict[str, torch.Tensor] = {}
    for global_step in range(0, args.total_timesteps, args.num_envs):
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        remaining_np = 1.0 - episode_steps.astype(np.float32) / max_episode_steps
        remaining = torch.as_tensor(remaining_np, device=device)
        planner_active = global_step >= args.planning_starts
        if planner_active and not takeover_checked:
            if world_update_count != expected_world_updates_at_takeover:
                raise RuntimeError(
                    "model pretraining update count does not match the declared UTD schedule: "
                    f"{world_update_count} != {expected_world_updates_at_takeover}"
                )
            if value_update_count == 0:
                raise RuntimeError("planning cannot start before any completed-return value update")
            writer.add_scalar("updates/world_at_planner_takeover", world_update_count, global_step)
            writer.add_scalar("updates/value_at_planner_takeover", value_update_count, global_step)
            takeover_checked = True

        if planner_active:
            root_noise = torch.randn(
                (args.num_envs, args.root_candidates, action_dim),
                generator=behavior_generator,
                device=device,
            )
            branch_noise = torch.randn(
                (
                    args.num_envs,
                    args.search_depth - 1,
                    args.root_candidates,
                    args.beam_width,
                    args.branch_factor,
                    action_dim,
                ),
                generator=behavior_generator,
                device=device,
            )
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            search_start_event = torch.cuda.Event(enable_timing=True)
            search_end_event = torch.cuda.Event(enable_timing=True)
            search_start_event.record()
            with torch.no_grad():
                planner_outputs = _clone_outputs(
                    planner_fn(obs_tensor, remaining, members, root_noise, branch_noise)
                )
            search_end_event.record()
            search_event_pairs.append((search_start_event, search_end_event))
            search_candidates_since_log += args.num_envs * (
                args.root_candidates
                + args.root_candidates
                * args.branch_factor
                * (1 + args.beam_width * max(0, args.search_depth - 2))
            )
            (
                normalized_action,
                elite_pre_tanh,
                best_score,
                mean_score,
                reward_prefix_std,
                tail_value_std,
                total_score_std,
                rank_correlation,
                top1_agreement,
                tail_value_min,
                tail_value_max,
                tail_dominance_ratio,
                h16_survival,
                action_change,
                elite_std,
                predicted_reward,
                _,
                _,
                _,
            ) = planner_outputs
            planner_valid = np.ones(args.num_envs, dtype=bool)
            for name, values in (
                ("improvement", best_score - mean_score),
                ("reward_prefix_std", reward_prefix_std),
                ("tail_value_std", tail_value_std),
                ("total_score_std", total_score_std),
                ("reward_total_rank_correlation", rank_correlation),
                ("reward_total_top1_agreement", top1_agreement),
                ("tail_value_min", tail_value_min),
                ("tail_value_max", tail_value_max),
                ("tail_dominance_ratio", tail_dominance_ratio),
                ("h16_survival", h16_survival),
                ("action_change", action_change),
                (
                    "action_saturation_fraction",
                    (normalized_action.abs() > 0.95).to(normalized_action.dtype).mean(dim=-1),
                ),
                ("elite_std", elite_std),
                ("predicted_reward", predicted_reward),
            ):
                planner_metric_sums[name].add_(values.detach().sum())
            planner_metric_count += args.num_envs
            predicted_score = best_score
        else:
            normalized_action = 2.0 * torch.rand(
                (args.num_envs, action_dim), generator=behavior_generator, device=device
            ) - 1.0
            elite_pre_tanh = torch.zeros(
                (args.num_envs, args.elite_roots, action_dim), device=device
            )
            predicted_score = torch.zeros(args.num_envs, device=device)
            planner_valid = np.zeros(args.num_envs, dtype=bool)

        env_action = map_normalized_action(normalized_action, action_low, action_high)
        torch._assert_async(
            torch.isfinite(env_action).all(), "nonfinite action reached environment execution"
        )
        next_observations, rewards, terminated, truncated, infos = envs.step(
            env_action.detach().cpu().numpy()
        )
        factual_next = factual_next_observations(
            next_observations, terminated, truncated, infos
        )
        done = np.asarray(terminated) | np.asarray(truncated)
        replay.add(
            observations,
            normalized_action,
            rewards,
            factual_next,
            terminated,
            truncated,
            remaining_np,
            members,
            elite_pre_tanh,
            predicted_score,
            planner_valid,
        )
        member_counts.add_(torch.bincount(members, minlength=args.ensemble_size))
        episode_returns += rewards
        episode_steps += 1
        if planner_active:
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            planner_metric_sums["chosen_reward_absolute_error"].add_(
                (predicted_reward.detach() - reward_tensor).abs().sum()
            )
            planner_metric_sums["chosen_reward_optimism_bias"].add_(
                (predicted_reward.detach() - reward_tensor).sum()
            )
        for env_index in np.flatnonzero(done):
            episode_return = float(episode_returns[env_index])
            recent_returns.append(episode_return)
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar(
                "charts/episodic_length", int(episode_steps[env_index]), global_step
            )
            episode_returns[env_index] = 0.0
            episode_steps[env_index] = 0
        acted_members = members
        members = update_episode_members(
            members,
            torch.as_tensor(done, device=device),
            behavior_generator,
            args.ensemble_size,
        )
        observations = next_observations

        if replay.size >= args.model_learning_starts:
            model_start_event = torch.cuda.Event(enable_timing=True)
            model_end_event = torch.cuda.Event(enable_timing=True)
            model_start_event.record()
            for _ in range(args.updates_per_vector_step):
                world_batch = replay.sample_world(args.batch_size, training_generator)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                world_outputs = world_loss_fn(
                    world,
                    world_batch["obs"],
                    world_batch["action"],
                    world_batch["next_obs"],
                    world_batch["reward"],
                    world_batch["terminated"],
                    world_batch["bootstrap_weight"],
                )
                world_optimizer.zero_grad(set_to_none=True)
                world_outputs[0].backward()
                world_grad = nn.utils.clip_grad_norm_(
                    world.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                world_optimizer.step()
                world_update_count += 1
                last_metrics.update(
                    {
                        "loss/world": world_outputs[0].detach(),
                        "loss/world_obs_nll": world_outputs[1].detach(),
                        "loss/world_reward_nll": world_outputs[2].detach(),
                        "loss/world_termination_bce": world_outputs[3].detach(),
                        "scales/world_obs_floor_fraction": world_outputs[4].detach(),
                        "scales/world_reward_floor_fraction": world_outputs[5].detach(),
                        "grad/world": world_grad.detach(),
                    }
                )

                if replay.can_sample_value():
                    value_batch = replay.sample_value(
                        args.batch_size,
                        training_generator,
                    )
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    value_loss, predicted_rate = value_loss_fn(
                        value,
                        value_batch["obs"],
                        value_batch["remaining"],
                        value_batch["value_return"],
                        value_batch["supervision_mask"],
                        max_episode_steps,
                    )
                    value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    value_grad = nn.utils.clip_grad_norm_(
                        value.parameters(), args.max_grad_norm, error_if_nonfinite=True
                    )
                    value_optimizer.step()
                    value_update_count += 1
                    mask = value_batch["supervision_mask"]
                    denominator = mask.sum().clamp_min(1.0)
                    predicted_return = max_episode_steps * predicted_rate.detach()
                    value_error = predicted_return - value_batch["value_return"]
                    factual_mean = (value_batch["value_return"] * mask).sum() / denominator
                    error_mean = (value_error * mask).sum() / denominator
                    factual_variance = (
                        (value_batch["value_return"] - factual_mean).square() * mask
                    ).sum() / denominator
                    error_variance = ((value_error - error_mean).square() * mask).sum() / denominator
                    explained_variance = torch.where(
                        factual_variance > 0,
                        1.0 - error_variance / factual_variance,
                        torch.zeros_like(factual_variance),
                    )
                    age = (
                        replay.total_vector_steps - 1 - value_batch["absolute"]
                    ).to(torch.float32) * args.num_envs
                    last_metrics.update(
                        {
                            "loss/value_rate_half_mse": value_loss.detach(),
                            "grad/value": value_grad.detach(),
                            "value/train_return_bias": error_mean,
                            "value/train_explained_variance": explained_variance,
                            "value/sample_age_mean": (age * mask).sum() / denominator,
                            "value/supervised_fraction": mask.mean(),
                        }
                    )

            if planner_active:
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                proposal_outputs = proposal_loss_fn(
                    proposal,
                    obs_tensor,
                    remaining,
                    acted_members,
                    elite_pre_tanh.detach(),
                )
                proposal_optimizer.zero_grad(set_to_none=True)
                proposal_outputs[0].backward()
                proposal_grad = nn.utils.clip_grad_norm_(
                    proposal.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                proposal_optimizer.step()
                proposal_update_count += 1
                last_metrics.update(
                    {
                        "loss/proposal_elite_nll": proposal_outputs[0].detach(),
                        "scales/proposal_floor_fraction": proposal_outputs[3].detach(),
                        "scales/proposal_entropy": proposal_outputs[4].detach(),
                        "grad/proposal": proposal_grad.detach(),
                    }
                )
            model_end_event.record()
            model_event_pairs.append((model_start_event, model_end_event))
            model_member_rows_since_log += (
                args.updates_per_vector_step * args.ensemble_size * args.batch_size
            )

        completed_steps = global_step + args.num_envs
        if completed_steps % args.log_frequency == 0:
            torch.cuda.synchronize()
            for key, value_metric in last_metrics.items():
                writer.add_scalar(key, value_metric, completed_steps)
            if planner_metric_count:
                for key, metric_sum in planner_metric_sums.items():
                    writer.add_scalar(
                        f"planner/{key}", metric_sum / planner_metric_count, completed_steps
                    )
                    metric_sum.zero_()
                planner_metric_count = 0
            total_member = int(member_counts.sum().item())
            if total_member > 0:
                for model_index in range(args.ensemble_size):
                    writer.add_scalar(
                        f"members/usage_{model_index}",
                        member_counts[model_index].to(torch.float32) / total_member,
                        completed_steps,
                    )
            search_seconds = sum(
                start.elapsed_time(end) for start, end in search_event_pairs
            ) / 1_000.0
            model_seconds = sum(
                start.elapsed_time(end) for start, end in model_event_pairs
            ) / 1_000.0
            if search_seconds > 0:
                writer.add_scalar(
                    "throughput/search_candidates_per_second",
                    search_candidates_since_log / search_seconds,
                    completed_steps,
                )
            if model_seconds > 0:
                writer.add_scalar(
                    "throughput/model_member_rows_per_second",
                    model_member_rows_since_log / model_seconds,
                    completed_steps,
                )
            search_event_pairs.clear()
            model_event_pairs.clear()
            search_candidates_since_log = 0
            model_member_rows_since_log = 0
            writer.add_scalar("updates/world", world_update_count, completed_steps)
            writer.add_scalar("updates/value", value_update_count, completed_steps)
            writer.add_scalar("updates/proposal", proposal_update_count, completed_steps)
            writer.add_scalar(
                "utd/optimizer_updates_per_real_transition",
                args.updates_per_vector_step / args.num_envs,
                completed_steps,
            )
            writer.add_scalar(
                "utd/replay_rows_per_member_per_real_transition",
                args.updates_per_vector_step * args.batch_size / args.num_envs,
                completed_steps,
            )
            writer.add_scalar(
                "utd/member_row_uses_per_real_transition",
                args.updates_per_vector_step
                * args.batch_size
                * args.ensemble_size
                / args.num_envs,
                completed_steps,
            )
            writer.add_scalar(
                "charts/SPS", int(completed_steps / (time.time() - start_time)), completed_steps
            )
            if recent_returns:
                ordered = np.sort(np.asarray(recent_returns))
                tail_count = max(1, math.ceil(0.1 * ordered.size))
                writer.add_scalar("charts/return_mean", float(ordered.mean()), completed_steps)
                writer.add_scalar(
                    "charts/return_cvar10", float(ordered[:tail_count].mean()), completed_steps
                )
            try:
                diagnostics = heldout_diagnostics(
                    world,
                    value,
                    replay,
                    diagnostic_generator,
                    args.diagnostic_batch_size,
                    max_episode_steps,
                )
            except RuntimeError:
                diagnostics = {}
            try:
                diagnostics.update(
                    prequential_diagnostics(
                        replay, diagnostic_generator, args.diagnostic_batch_size
                    )
                )
            except RuntimeError:
                pass
            for key, metric in diagnostics.items():
                writer.add_scalar(key, metric, completed_steps)
            print(
                f"step={completed_steps} sps={int(completed_steps / (time.time() - start_time))} "
                f"episodes={len(recent_returns)}"
            )

        if is_evaluation_step(completed_steps):
            evaluation_returns = evaluate_search(
                planner_fn,
                eval_envs,
                args,
                obs_dim,
                action_dim,
                max_episode_steps,
                action_low,
                action_high,
                device,
            )
            writer.add_scalar("eval/search_return_mean", evaluation_returns.mean(), completed_steps)
            writer.add_scalar("eval/search_return_median", np.median(evaluation_returns), completed_steps)
            writer.add_scalar("eval/search_return_min", evaluation_returns.min(), completed_steps)
            for model_index in range(args.ensemble_size):
                writer.add_scalar(
                    f"eval/member_{model_index}_return_mean",
                    evaluation_returns[model_index :: args.ensemble_size].mean(),
                    completed_steps,
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "world": world.state_dict(),
                "value": value.state_dict(),
                "proposal": proposal.state_dict(),
                "args": asdict(args),
            },
            model_path,
        )
    envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
