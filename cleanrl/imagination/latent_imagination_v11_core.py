# Online Dreamer-style latent imagination.
#
# v11 tightens the implementation against the local `../dreamer4` reference:
# 1. behavior learning happens before imagination, using replay-action cloning,
# 2. PMPO prior KL uses detached rollout-time policy distributions,
# 3. reward and action heads get multi-token prediction supervision,
# 4. replay contexts are contiguous across the ring buffer,
# 5. imagination updates only policy/value parameters while the world model stays
#    under no-grad in the rollout path.
import copy
import os
import random
import time
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.imagination import latent_imagination_v5_core as v5


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins: int, bin_range: float) -> torch.Tensor:
    if num_bins % 2 == 1:
        half = torch.linspace(-bin_range, 0.0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(0)])
    half = torch.linspace(-bin_range, 0.0, num_bins // 2)
    half = symexp(half)
    return torch.cat([half, -half.flip(0)])


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(-1)
    below = (bins <= x).long().sum(-1) - 1
    below = below.clamp(0, len(bins) - 1)
    above = (below + 1).clamp(0, len(bins) - 1)
    equal = below == above
    dist_below = torch.where(equal, torch.ones_like(x.squeeze(-1)), torch.abs(bins[below] - x.squeeze(-1)))
    dist_above = torch.where(equal, torch.ones_like(x.squeeze(-1)), torch.abs(bins[above] - x.squeeze(-1)))
    total = dist_below + dist_above
    weight_below = dist_above / total
    weight_above = dist_below / total
    return (
        F.one_hot(below, len(bins)).float() * weight_below.unsqueeze(-1)
        + F.one_hot(above, len(bins)).float() * weight_above.unsqueeze(-1)
    )


def twohot_predict(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


def twohot_loss(logits: torch.Tensor, target_twohot: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -(target_twohot * log_probs).sum(-1)


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
    """the wandb project name"""
    wandb_entity: str = None
    """the wandb entity"""
    capture_video: bool = False
    """whether to capture videos"""
    save_model: bool = False
    """whether to save the model"""
    upload_model: bool = False
    """whether to upload the model"""
    hf_entity: str = ""
    """the user or org name for HF upload"""

    env_id: str = "HalfCheetah-v4"
    """the environment id"""
    total_timesteps: int = 8_000_000
    """total timesteps"""
    num_envs: int = 16
    """number of parallel envs"""
    num_steps: int = 256
    """steps collected per iteration"""
    anneal_lr: bool = False
    """anneal learning rates"""
    gamma: float = 0.99
    """discount factor"""
    learning_starts: int = 20_000
    """real env steps before training starts"""
    replay_steps: int = 50_000
    """time-axis length of the sequence replay buffer"""
    context_len: int = 16
    """number of real transitions used to build a context state"""
    policy_learning_starts: int = 200_000
    """real env steps before imagination policy/value training starts"""
    prior_snapshot_step: int = 200_000
    """real env step when behavior-only training begins decaying toward imagination"""

    learning_rate: float = 3e-4
    """default optimizer learning rate"""
    world_model_learning_rate: float = 3e-4
    """world model learning rate"""
    imagination_learning_rate: float = 3e-4
    """imagination actor/value learning rate"""
    world_model_batch_size: int = 256
    """world model batch size"""
    world_model_gradient_steps: int = 64
    """world model updates per iteration"""
    world_model_max_grad_norm: float = 1.0
    """world model gradient clipping"""
    imagination_batch_size: int = 256
    """number of starting contexts per imagination update"""
    imagination_gradient_steps: int = 16
    """imagination updates per iteration"""
    imagination_max_grad_norm: float = 1.0
    """imagination gradient clipping"""

    latent_dim: int = 64
    """latent state width"""
    model_hidden_dim: int = 128
    """world model hidden width"""
    context_hidden_dim: int = 64
    """recurrent context width"""
    model_min_std: float = 0.05
    """minimum std for the stochastic transition"""
    model_max_std: float = 1.0
    """maximum std for the stochastic transition"""
    target_encoder_tau: float = 0.99
    """EMA factor for the target encoder"""
    transition_coef: float = 1.0
    """weight on the transition loss"""
    reward_coef: float = 1.0
    """weight on the reward loss"""
    done_coef: float = 0.25
    """weight on the done loss"""
    use_done_model: bool = True
    """learn a done model"""
    multi_token_pred_len: int = 8
    """multi-token prediction length for behavior and reward heads"""
    reward_num_bins: int = 255
    """number of bins for the reward head"""
    reward_bin_range: float = 3.0
    """symlog-space range for reward bins"""

    imagination_horizon: int = 8
    """imagination rollout horizon"""
    imagination_lambda: float = 0.95
    """lambda-return parameter for imagined value learning"""
    imagination_alpha: float = 0.5
    """PMPO positive/negative balance weight"""
    imagination_value_coef: float = 0.5
    """weight on the imagination value loss"""
    imagination_prior_coef: float = 0.3
    """weight on reverse KL to the detached rollout-time policy distribution"""
    behavior_clone_coef: float = 1.0
    """weight on replay-action cloning for the actor"""
    behavior_clone_min_coef: float = 0.1
    """minimum replay-action cloning weight after decay"""
    behavior_clone_decay_steps: int = 1_000_000
    """linear decay length for replay-action cloning after imagination begins"""
    imagination_num_bins: int = 255
    """number of bins for the imagination value head"""
    imagination_bin_range: float = 3.0
    """symlog-space range for imagination value bins"""

    num_iterations: int = 0
    """computed in runtime"""


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


class SequenceReplayBuffer:
    def __init__(self, capacity_steps: int, obs_shape, action_shape, num_envs: int):
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.obs = np.zeros((capacity_steps, num_envs) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((capacity_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity_steps, num_envs) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((capacity_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((capacity_steps, num_envs), dtype=np.float32)
        self.step_ids = np.full((capacity_steps, num_envs), -1, dtype=np.int64)
        self.pos = 0
        self.full = False
        self.next_step_id = 0

    @property
    def size(self) -> int:
        return self.capacity_steps if self.full else self.pos

    def add(self, obs, next_obs, actions, rewards, dones):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.step_ids[self.pos] = self.next_step_id
        self.pos += 1
        self.next_step_id += 1
        if self.pos == self.capacity_steps:
            self.full = True
            self.pos = 0

    def can_sample(self, min_steps: int) -> bool:
        return self.size > min_steps

    def sample_indices(self, batch_size: int):
        size = self.size
        time_indices = np.random.randint(0, size, size=batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=batch_size)
        return time_indices, env_indices

    def _gather_step_ids(self, time_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        return self.step_ids[time_indices % self.capacity_steps, env_indices]

    def get_tensors(self, time_indices, env_indices, device: torch.device):
        obs = torch.tensor(self.obs[time_indices, env_indices], device=device, dtype=torch.float32)
        next_obs = torch.tensor(self.next_obs[time_indices, env_indices], device=device, dtype=torch.float32)
        actions = torch.tensor(self.actions[time_indices, env_indices], device=device, dtype=torch.float32)
        rewards = torch.tensor(self.rewards[time_indices, env_indices], device=device, dtype=torch.float32)
        dones = torch.tensor(self.dones[time_indices, env_indices], device=device, dtype=torch.float32)
        return obs, next_obs, actions, rewards, dones

    def build_context_states(
        self,
        agent: "Agent",
        time_indices: np.ndarray,
        env_indices: np.ndarray,
        context_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        context_state = agent.init_context_state(len(time_indices), device)
        current_step_ids = self._gather_step_ids(time_indices, env_indices)
        for offset in range(context_len, 0, -1):
            prev_indices = time_indices - offset
            prev_slots = np.mod(prev_indices, self.capacity_steps)
            valid = prev_indices >= 0 if not self.full else np.ones_like(prev_indices, dtype=bool)
            valid &= self.step_ids[prev_slots, env_indices] == (current_step_ids - offset)
            if not np.any(valid):
                continue
            prev_obs = torch.tensor(self.obs[prev_slots[valid], env_indices[valid]], device=device, dtype=torch.float32)
            prev_actions = torch.tensor(
                self.actions[prev_slots[valid], env_indices[valid]], device=device, dtype=torch.float32
            )
            prev_dones = torch.tensor(self.dones[prev_slots[valid], env_indices[valid]], device=device, dtype=torch.float32)
            updated = agent.update_context_state(context_state[valid], prev_obs, prev_actions)
            updated = updated * (1.0 - prev_dones.unsqueeze(-1))
            context_state[valid] = updated
        return context_state

    def get_mtp_targets(self, time_indices, env_indices, horizon: int, device: torch.device):
        batch_size = len(time_indices)
        action_shape = self.actions.shape[2:]
        action_targets = np.zeros((batch_size, horizon) + action_shape, dtype=np.float32)
        reward_targets = np.zeros((batch_size, horizon), dtype=np.float32)
        target_mask = np.zeros((batch_size, horizon), dtype=bool)
        current_step_ids = self._gather_step_ids(time_indices, env_indices)
        alive = np.ones(batch_size, dtype=bool)

        for offset in range(horizon):
            future_indices = time_indices + offset
            future_slots = np.mod(future_indices, self.capacity_steps)
            valid = alive.copy()
            if not self.full:
                valid &= future_indices < self.size
            valid &= self.step_ids[future_slots, env_indices] == (current_step_ids + offset)
            if np.any(valid):
                action_targets[valid, offset] = self.actions[future_slots[valid], env_indices[valid]]
                reward_targets[valid, offset] = self.rewards[future_slots[valid], env_indices[valid]]
                target_mask[valid, offset] = True
                alive[valid] &= self.dones[future_slots[valid], env_indices[valid]] < 0.5
            alive &= valid

        return (
            torch.tensor(action_targets, device=device, dtype=torch.float32),
            torch.tensor(reward_targets, device=device, dtype=torch.float32),
            torch.tensor(target_mask, device=device, dtype=torch.bool),
        )


class Agent(v5.Agent):
    def __init__(
        self,
        envs,
        latent_dim: int = 64,
        model_hidden_dim: int = 128,
        model_min_std: float = 0.05,
        model_max_std: float = 1.0,
        use_done_model: bool = False,
        context_hidden_dim: int = 64,
        imagination_num_bins: int = 255,
        imagination_bin_range: float = 3.0,
        reward_num_bins: int = 255,
        reward_bin_range: float = 3.0,
        multi_token_pred_len: int = 8,
    ):
        super().__init__(
            envs,
            latent_dim=latent_dim,
            model_hidden_dim=model_hidden_dim,
            model_min_std=model_min_std,
            model_max_std=model_max_std,
            use_done_model=use_done_model,
        )
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.context_hidden_dim = context_hidden_dim
        self.multi_token_pred_len = multi_token_pred_len
        self.context_rnn = nn.GRUCell(latent_dim + action_dim + 1, context_hidden_dim)
        self.model_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        self.actor_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        self.value_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        nn.init.zeros_(self.model_context_proj.weight)
        nn.init.zeros_(self.actor_context_proj.weight)
        nn.init.zeros_(self.value_context_proj.weight)

        self.behavior_mtp_head = nn.Sequential(
            v5.layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, action_dim * (multi_token_pred_len - 1)), std=0.01),
        )
        self.reward_mtp_head = nn.Sequential(
            v5.layer_init(nn.Linear(2 * latent_dim + action_dim, model_hidden_dim)),
            nn.SiLU(),
            v5.layer_init(nn.Linear(model_hidden_dim, reward_num_bins * multi_token_pred_len), std=0.01),
        )

        self.value_head = nn.Sequential(
            v5.layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, imagination_num_bins), std=0.01),
        )
        self.register_buffer("value_bins", build_symexp_bins(imagination_num_bins, imagination_bin_range))
        self.register_buffer("reward_bins", build_symexp_bins(reward_num_bins, reward_bin_range))

    def init_context_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.context_hidden_dim, device=device)

    def update_context_state(self, context_state: torch.Tensor, obs: torch.Tensor, env_action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.encode(obs)
        continue_feature = torch.ones(latent.shape[0], 1, device=latent.device)
        context_input = torch.cat([latent, env_action, continue_feature], dim=-1)
        return self.context_rnn(context_input, context_state)

    def step_context_state_from_latent(
        self,
        context_state: torch.Tensor,
        latent: torch.Tensor,
        env_action: torch.Tensor,
        continuation: torch.Tensor,
    ) -> torch.Tensor:
        context_input = torch.cat([latent.detach(), env_action, continuation.unsqueeze(-1)], dim=-1)
        return self.context_rnn(context_input, context_state)

    def get_actor_dist_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> Normal:
        actor_latent = latent + self.actor_context_proj(context_state)
        action_mean = self.actor_mean(actor_latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return Normal(action_mean, torch.exp(action_logstd))

    def get_behavior_mtp_means_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        actor_latent = latent + self.actor_context_proj(context_state)
        if self.multi_token_pred_len <= 1:
            return actor_latent.new_zeros(actor_latent.shape[0], 0, self.actor_logstd.shape[-1])
        mtp_means = self.behavior_mtp_head(actor_latent)
        return mtp_means.view(actor_latent.shape[0], self.multi_token_pred_len - 1, -1)

    def get_value_logits_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        value_latent = latent + self.value_context_proj(context_state)
        return self.value_head(value_latent)

    def get_value_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        logits = self.get_value_logits_from_latent_context(latent, context_state)
        return twohot_predict(logits, self.value_bins)

    def contextual_transition_params(
        self, latent: torch.Tensor, context_state: torch.Tensor, env_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transition_params(latent + self.model_context_proj(context_state), env_action)

    def contextual_predict_reward_done(
        self,
        latent: torch.Tensor,
        context_state: torch.Tensor,
        env_action: torch.Tensor,
        next_latent: torch.Tensor,
    ):
        model_latent = latent + self.model_context_proj(context_state)
        model_next_latent = next_latent + self.model_context_proj(context_state)
        return self.predict_reward_done(model_latent, env_action, model_next_latent)

    def contextual_predict_reward_logits(
        self,
        latent: torch.Tensor,
        context_state: torch.Tensor,
        env_action: torch.Tensor,
        next_latent: torch.Tensor,
    ) -> torch.Tensor:
        model_latent = latent + self.model_context_proj(context_state)
        model_next_latent = next_latent + self.model_context_proj(context_state)
        model_input = torch.cat([model_latent, env_action, model_next_latent], dim=-1)
        logits = self.reward_mtp_head(model_input)
        return logits.view(latent.shape[0], self.multi_token_pred_len, -1)


def world_model_parameters(agent: Agent):
    params = []
    params.extend(agent.encoder.parameters())
    params.extend(agent.transition_backbone.parameters())
    params.extend(agent.transition_mean.parameters())
    params.extend(agent.transition_logstd.parameters())
    params.extend(agent.reward_model.parameters())
    params.extend(agent.reward_mtp_head.parameters())
    params.extend(agent.context_rnn.parameters())
    params.extend(agent.model_context_proj.parameters())
    if agent.done_model is not None:
        params.extend(agent.done_model.parameters())
    return params


def imagination_parameters(agent: Agent):
    params = []
    params.extend(agent.actor_mean.parameters())
    params.append(agent.actor_logstd)
    params.extend(agent.behavior_mtp_head.parameters())
    params.extend(agent.actor_context_proj.parameters())
    params.extend(agent.value_head.parameters())
    params.extend(agent.value_context_proj.parameters())
    return params


def current_imagination_coef(args: Args, global_step: int) -> float:
    return 1.0 if global_step >= args.policy_learning_starts else 0.0


def prior_snapshot_step(args: Args) -> int:
    return max(args.policy_learning_starts, args.prior_snapshot_step)


def current_behavior_clone_coef(args: Args, global_step: int) -> float:
    if global_step < args.learning_starts:
        return 0.0
    start = prior_snapshot_step(args)
    if global_step <= start:
        return args.behavior_clone_coef
    progress = min((global_step - start) / max(args.behavior_clone_decay_steps, 1), 1.0)
    return args.behavior_clone_coef + progress * (args.behavior_clone_min_coef - args.behavior_clone_coef)


def compute_behavior_clone_loss(
    agent: Agent,
    obs: torch.Tensor,
    context_state: torch.Tensor,
    behavior_actions: torch.Tensor,
    behavior_action_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        latent = agent.encode(obs)
    dist = agent.get_actor_dist_from_latent_context(latent, context_state)
    current_loss = -dist.log_prob(behavior_actions[:, 0].detach()).sum(-1)
    current_loss = current_loss[behavior_action_mask[:, 0]].mean() if behavior_action_mask[:, 0].any() else current_loss.mean() * 0.0

    if agent.multi_token_pred_len <= 1:
        return current_loss

    future_means = agent.get_behavior_mtp_means_from_latent_context(latent, context_state)
    future_std = torch.exp(agent.actor_logstd).view(1, 1, -1).expand_as(future_means)
    future_dist = Normal(future_means, future_std)
    future_targets = behavior_actions[:, 1:].detach()
    future_mask = behavior_action_mask[:, 1:]
    future_nll = -future_dist.log_prob(future_targets).sum(-1)
    if future_mask.any():
        future_loss = future_nll[future_mask].mean()
    else:
        future_loss = current_loss * 0.0
    return current_loss + future_loss


def lambda_returns(
    rewards: torch.Tensor,
    continuations: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    next_values = torch.cat([values[1:].detach(), bootstrap.unsqueeze(0)], dim=0)
    last = bootstrap
    for t in reversed(range(rewards.shape[0])):
        mixed_bootstrap = (1.0 - lambda_) * next_values[t] + lambda_ * last
        last = rewards[t] + gamma * continuations[t] * mixed_bootstrap
        returns[t] = last
    return returns


def sample_imagined_latent_trajectory(
    agent: Agent,
    start_obs: torch.Tensor,
    start_context: torch.Tensor,
    args: Args,
):
    with torch.no_grad():
        latent = agent.encode(start_obs)
    context_state = start_context

    logprobs = []
    rewards = []
    continuations = []
    values = []
    value_logits = []
    prior_kls = []
    alive_weights = []

    alive = torch.ones(latent.shape[0], device=latent.device)
    return latent, context_state, logprobs, rewards, continuations, values, value_logits, prior_kls, alive_weights, alive


def compute_world_model_losses(
    agent: Agent,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    env_actions: torch.Tensor,
    reward_targets: torch.Tensor,
    reward_target_mask: torch.Tensor,
    dones: torch.Tensor,
    context_state: torch.Tensor,
    args: Args,
    device: torch.device,
):
    latent = agent.encode(obs)
    target_next_latent = agent.encode_target(next_obs)
    pred_next_mean, pred_next_std = agent.contextual_transition_params(latent, context_state, env_actions)
    reward_pred, done_logit = agent.contextual_predict_reward_done(latent, context_state, env_actions, target_next_latent)
    reward_logits = agent.contextual_predict_reward_logits(latent, context_state, env_actions, target_next_latent)

    transition_loss = v5.gaussian_nll(target_next_latent, pred_next_mean, pred_next_std).mean()
    reward_twohot = twohot_encode(reward_targets.reshape(-1), agent.reward_bins)
    reward_ce = twohot_loss(reward_logits.reshape(-1, reward_logits.shape[-1]), reward_twohot).view_as(reward_targets)
    if reward_target_mask.any():
        reward_loss = reward_ce[reward_target_mask].mean()
    else:
        reward_loss = reward_ce.mean() * 0.0
    if done_logit is None:
        done_loss = torch.zeros((), device=device)
    else:
        done_loss = F.binary_cross_entropy_with_logits(done_logit, dones)
    raw_loss = args.transition_coef * transition_loss + args.reward_coef * reward_loss + args.done_coef * done_loss
    return raw_loss, transition_loss, reward_loss, done_loss, pred_next_std.mean()


def compute_imagination_losses(
    agent: Agent,
    start_obs: torch.Tensor,
    start_context: torch.Tensor,
    args: Args,
):
    with torch.no_grad():
        latent = agent.encode(start_obs)
    context_state = start_context

    logprobs = []
    rewards = []
    continuations = []
    values = []
    value_logits = []
    prior_kls = []
    alive_weights = []

    alive = torch.ones(latent.shape[0], device=latent.device)
    for _ in range(args.imagination_horizon):
        dist = agent.get_actor_dist_from_latent_context(latent, context_state)
        prior_dist = Normal(dist.loc.detach(), dist.scale.detach())
        action = dist.sample()
        logprobs.append(dist.log_prob(action.detach()).sum(-1))

        step_value_logits = agent.get_value_logits_from_latent_context(latent, context_state)
        value_logits.append(step_value_logits)
        values.append(twohot_predict(step_value_logits, agent.value_bins))
        alive_weights.append(alive)

        prior_kls.append(kl_divergence(prior_dist, dist).sum(-1))

        with torch.no_grad():
            env_action = agent.clamp_action(action.detach())
            pred_next_mean, pred_next_std = agent.contextual_transition_params(latent, context_state, env_action)
            next_latent = pred_next_mean + torch.randn_like(pred_next_std) * pred_next_std
            reward_logits = agent.contextual_predict_reward_logits(latent, context_state, env_action, next_latent)
            reward_pred = twohot_predict(reward_logits[:, 0], agent.reward_bins)
            _, done_logit = agent.contextual_predict_reward_done(latent, context_state, env_action, next_latent)
            if done_logit is None:
                continuation = torch.ones_like(reward_pred)
            else:
                continuation = 1.0 - torch.sigmoid(done_logit)
            next_context = agent.step_context_state_from_latent(context_state, next_latent, env_action, continuation)

        rewards.append(reward_pred)
        continuations.append(continuation)
        alive = alive * continuation
        latent = next_latent
        context_state = next_context

    with torch.no_grad():
        bootstrap = agent.get_value_from_latent_context(latent, context_state)

    logprobs = torch.stack(logprobs)
    rewards = torch.stack(rewards)
    continuations = torch.stack(continuations)
    values = torch.stack(values)
    value_logits = torch.stack(value_logits)
    alive_weights = torch.stack(alive_weights).detach()
    prior_kls = torch.stack(prior_kls)

    returns = lambda_returns(
        rewards,
        continuations,
        values.detach(),
        bootstrap.detach(),
        args.gamma,
        args.imagination_lambda,
    )
    advantages = (returns - values).detach()

    flat_advantages = advantages.reshape(-1)
    flat_logprobs = logprobs.reshape(-1)
    flat_weights = alive_weights.reshape(-1)
    flat_prior_kls = prior_kls.reshape(-1)
    flat_value_logits = value_logits.reshape(-1, value_logits.shape[-1])
    flat_targets = twohot_encode(returns.reshape(-1), agent.value_bins)

    positive_mask = flat_advantages >= 0
    negative_mask = ~positive_mask
    zero = torch.zeros((), device=start_obs.device)

    if torch.any(positive_mask):
        pos_weights = flat_weights[positive_mask]
        pos_loss = -(pos_weights * flat_logprobs[positive_mask]).sum() / pos_weights.sum().clamp_min(1e-6)
    else:
        pos_loss = zero
    if torch.any(negative_mask):
        neg_weights = flat_weights[negative_mask]
        neg_loss = (neg_weights * flat_logprobs[negative_mask]).sum() / neg_weights.sum().clamp_min(1e-6)
    else:
        neg_loss = zero

    policy_loss = args.imagination_alpha * pos_loss + (1.0 - args.imagination_alpha) * neg_loss
    value_loss = (flat_weights * twohot_loss(flat_value_logits, flat_targets)).sum() / flat_weights.sum().clamp_min(1e-6)
    prior_kl_loss = (flat_weights * flat_prior_kls).sum() / flat_weights.sum().clamp_min(1e-6)
    total_loss = policy_loss + args.imagination_value_coef * value_loss + args.imagination_prior_coef * prior_kl_loss

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "prior_kl_loss": prior_kl_loss,
        "behavior_clone_loss": torch.zeros((), device=start_obs.device),
        "mean_return": returns[0].mean(),
        "mean_advantage": advantages.mean(),
        "positive_fraction": positive_mask.float().mean(),
    }


def main(args_class=Args):
    args = tyro.cli(args_class)
    assert args.context_len > 0
    assert args.imagination_horizon > 0
    args.num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)
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
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(
        envs,
        latent_dim=args.latent_dim,
        model_hidden_dim=args.model_hidden_dim,
        model_min_std=args.model_min_std,
        model_max_std=args.model_max_std,
        use_done_model=args.use_done_model,
        context_hidden_dim=args.context_hidden_dim,
        imagination_num_bins=args.imagination_num_bins,
        imagination_bin_range=args.imagination_bin_range,
        reward_num_bins=args.reward_num_bins,
        reward_bin_range=args.reward_bin_range,
        multi_token_pred_len=args.multi_token_pred_len,
    ).to(device)

    world_model_optimizer = optim.Adam(world_model_parameters(agent), lr=args.world_model_learning_rate, eps=1e-5)
    imagination_optimizer = optim.Adam(imagination_parameters(agent), lr=args.imagination_learning_rate, eps=1e-5)

    replay = SequenceReplayBuffer(
        args.replay_steps,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        args.num_envs,
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    online_context_state = agent.init_context_state(args.num_envs, device)
    last_world_model_stats = {
        "loss": torch.zeros((), device=device),
        "transition_loss": torch.zeros((), device=device),
        "reward_loss": torch.zeros((), device=device),
        "done_loss": torch.zeros((), device=device),
        "transition_std": torch.zeros((), device=device),
    }
    last_imag_stats = {
        "total_loss": torch.zeros((), device=device),
        "policy_loss": torch.zeros((), device=device),
        "value_loss": torch.zeros((), device=device),
        "prior_kl_loss": torch.zeros((), device=device),
        "behavior_clone_loss": torch.zeros((), device=device),
        "mean_return": torch.zeros((), device=device),
        "mean_advantage": torch.zeros((), device=device),
        "positive_fraction": torch.zeros((), device=device),
    }

    for iteration in range(1, args.num_iterations + 1):
        for _ in range(args.num_steps):
            global_step += args.num_envs
            if global_step < args.learning_starts:
                actions_np = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)], dtype=np.float32)
                actions = torch.tensor(actions_np, device=device, dtype=torch.float32)
            else:
                with torch.no_grad():
                    latent = agent.encode(next_obs)
                    dist = agent.get_actor_dist_from_latent_context(latent, online_context_state)
                    actions = agent.clamp_action(dist.sample())
                actions_np = actions.cpu().numpy()

            next_obs_np, rewards_np, terminations, truncations, infos = envs.step(actions_np)
            next_done_np = np.logical_or(terminations, truncations)
            model_next_obs_np = v5.extract_model_next_obs(next_obs_np, infos)

            replay.add(
                next_obs.cpu().numpy(),
                model_next_obs_np,
                actions_np,
                rewards_np,
                next_done_np.astype(np.float32),
            )

            with torch.no_grad():
                online_context_state = agent.update_context_state(online_context_state, next_obs, actions)
                online_context_state = online_context_state * (
                    1.0 - torch.tensor(next_done_np, device=device, dtype=torch.float32).unsqueeze(-1)
                )

            next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if not replay.can_sample(args.context_len + 1) or global_step < args.learning_starts:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            continue

        for _ in range(args.world_model_gradient_steps):
            time_indices, env_indices = replay.sample_indices(args.world_model_batch_size)
            obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = replay.get_tensors(
                time_indices, env_indices, device
            )
            context_batch = replay.build_context_states(agent, time_indices, env_indices, args.context_len, device)
            _, reward_target_batch, reward_target_mask = replay.get_mtp_targets(
                time_indices, env_indices, args.multi_token_pred_len, device
            )
            wm_loss, transition_loss, reward_loss, done_loss, transition_std = compute_world_model_losses(
                agent,
                obs_batch,
                next_obs_batch,
                actions_batch,
                reward_target_batch,
                reward_target_mask,
                dones_batch,
                context_batch,
                args,
                device,
            )
            world_model_optimizer.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(world_model_parameters(agent), args.world_model_max_grad_norm)
            world_model_optimizer.step()
            agent.update_target_encoder(args.target_encoder_tau)
            last_world_model_stats = {
                "loss": wm_loss.detach(),
                "transition_loss": transition_loss.detach(),
                "reward_loss": reward_loss.detach(),
                "done_loss": done_loss.detach(),
                "transition_std": transition_std.detach(),
            }

        if global_step >= args.learning_starts:
            for _ in range(args.imagination_gradient_steps):
                time_indices, env_indices = replay.sample_indices(args.imagination_batch_size)
                start_obs, _, _, _, _ = replay.get_tensors(time_indices, env_indices, device)
                start_context = replay.build_context_states(agent, time_indices, env_indices, args.context_len, device)
                action_target_batch, _, action_target_mask = replay.get_mtp_targets(
                    time_indices, env_indices, args.multi_token_pred_len, device
                )
                bc_loss = compute_behavior_clone_loss(agent, start_obs, start_context, action_target_batch, action_target_mask)
                imag_coef = current_imagination_coef(args, global_step)
                if imag_coef > 0.0:
                    imag_stats = compute_imagination_losses(agent, start_obs, start_context, args)
                else:
                    imag_stats = {key: value.detach().clone() for key, value in last_imag_stats.items()}
                    imag_stats["total_loss"] = torch.zeros((), device=device)
                    imag_stats["policy_loss"] = torch.zeros((), device=device)
                    imag_stats["value_loss"] = torch.zeros((), device=device)
                    imag_stats["prior_kl_loss"] = torch.zeros((), device=device)
                    imag_stats["mean_return"] = torch.zeros((), device=device)
                    imag_stats["mean_advantage"] = torch.zeros((), device=device)
                    imag_stats["positive_fraction"] = torch.zeros((), device=device)
                total_imag_loss = imag_stats["total_loss"] * imag_coef + current_behavior_clone_coef(args, global_step) * bc_loss
                imagination_optimizer.zero_grad()
                total_imag_loss.backward()
                nn.utils.clip_grad_norm_(imagination_parameters(agent), args.imagination_max_grad_norm)
                imagination_optimizer.step()
                last_imag_stats = {key: value.detach() for key, value in imag_stats.items()}
                last_imag_stats["behavior_clone_loss"] = bc_loss.detach()
                last_imag_stats["total_loss"] = total_imag_loss.detach()

        writer.add_scalar("charts/world_model_learning_rate", world_model_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/imagination_learning_rate", imagination_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/model_loss", last_world_model_stats["loss"].item(), global_step)
        writer.add_scalar("losses/transition_loss", last_world_model_stats["transition_loss"].item(), global_step)
        writer.add_scalar("losses/reward_loss", last_world_model_stats["reward_loss"].item(), global_step)
        writer.add_scalar("losses/done_loss", last_world_model_stats["done_loss"].item(), global_step)
        writer.add_scalar("imagination/coef", current_imagination_coef(args, global_step), global_step)
        writer.add_scalar("imagination/policy_loss", last_imag_stats["policy_loss"].item(), global_step)
        writer.add_scalar("imagination/value_loss", last_imag_stats["value_loss"].item(), global_step)
        writer.add_scalar("imagination/prior_kl", last_imag_stats["prior_kl_loss"].item(), global_step)
        writer.add_scalar("imagination/behavior_clone_loss", last_imag_stats["behavior_clone_loss"].item(), global_step)
        writer.add_scalar("imagination/total_loss", last_imag_stats["total_loss"].item(), global_step)
        writer.add_scalar("imagination/mean_return", last_imag_stats["mean_return"].item(), global_step)
        writer.add_scalar("imagination/mean_advantage", last_imag_stats["mean_advantage"].item(), global_step)
        writer.add_scalar("imagination/positive_fraction", last_imag_stats["positive_fraction"].item(), global_step)
        writer.add_scalar("imagination/transition_std", last_world_model_stats["transition_std"].item(), global_step)
        writer.add_scalar("imagination/prior_active", 1.0 if current_imagination_coef(args, global_step) > 0.0 else 0.0, global_step)
        writer.add_scalar("imagination/behavior_clone_coef", current_behavior_clone_coef(args, global_step), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=partial(
                Agent,
                latent_dim=args.latent_dim,
                model_hidden_dim=args.model_hidden_dim,
                model_min_std=args.model_min_std,
                model_max_std=args.model_max_std,
                use_done_model=args.use_done_model,
                context_hidden_dim=args.context_hidden_dim,
                imagination_num_bins=args.imagination_num_bins,
                imagination_bin_range=args.imagination_bin_range,
                reward_num_bins=args.reward_num_bins,
                reward_bin_range=args.reward_bin_range,
                multi_token_pred_len=args.multi_token_pred_len,
            ),
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
