"""
Dreamer4 World Model — MuJoCo Training Script
================================================
Follows the default Dreamer4 split adapted for MuJoCo:

  Phase 1: Collect real experience into replay
  Phase 2: Train tokenizer + world model on replay
  Phase 3: Train policy/value heads only on dreamed rollouts

Real environment interaction is used to gather data for the world model.
Policy and value learning happen in imagination, mirroring the default
DreamTrainer path rather than the real-experience SimTrainer path.

Usage:
    uv run python cleanrl/d4_wm/train_mujoco.py --env-id HalfCheetah-v4
"""
from __future__ import annotations

import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import tyro

from cleanrl.d4_wm.utils import (
    Experience, Actions, StateTokenizer,
    exists,
)
from cleanrl.d4_wm.world_model import DynamicsWorldModel
from cleanrl.d4_wm.transformer import TransformerIntermediates


@dataclass
class Args:
    # Experiment
    exp_name: str = "d4_wm"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # Environment
    env_id: str = "HalfCheetah-v4"
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    capture_video: bool = False
    normalize_observations: bool = False
    normalize_rewards: bool = False
    observation_clip: float = 0.0
    reward_clip: float = 0.0

    # World Model Architecture
    wm_dim: int = 64
    wm_dim_latent: int = 32
    wm_num_latent_tokens: int = 2
    wm_depth: int = 2
    wm_attn_heads: int = 4
    wm_attn_dim_head: int = 32
    wm_time_block_every: int = 2
    wm_num_register_tokens: int = 2
    wm_num_spatial_tokens: int = 2
    wm_max_steps: int = 8
    wm_multi_token_pred_len: int = 1
    wm_pred_orig_latent: bool = True
    wm_use_time_rnn: bool = True

    # State Tokenizer
    tok_dim_hidden: int = 64
    tok_encoder_depth: int = 2
    tok_decoder_depth: int = 2

    # Replay Buffer
    replay_buffer_size: int = 1_000_000
    min_replay_size: int = 2048
    replay_seq_len: int = 32

    # World Model Training
    wm_learning_rate: float = 3e-4
    wm_max_grad_norm: float = 0.5
    wm_weight_decay: float = 1e-4
    wm_batch_size: int = 32
    wm_updates_per_iter: int = 0
    """Legacy fixed WM updates per iteration. Set >0 to override ratio scheduling."""
    wm_updates_per_env_step: float = 2.0
    """WM updates to schedule per real env frame collected."""
    wm_probe_batch_size: int = 16
    wm_warmup_min_steps: int = 5_000
    wm_readiness_patience: int = 5
    wm_readiness_min_evals: int = 5
    wm_readiness_min_improvement: float = 0.01
    wm_readiness_loss_threshold: float = 0.0
    """Optional absolute WM probe-loss threshold for enabling dream RL."""

    # Policy
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    policy_max_grad_norm: float = 0.5
    ppo_eps_clip: float = 0.2
    gae_gamma: float = 0.997
    gae_lambda: float = 0.95
    policy_entropy_weight: float = 0.01
    use_pmpo: bool = True
    pmpo_kl_weight: float = 0.3

    # Interaction (SimTrainer-style)
    num_steps_denoising: int = 4
    """Denoising steps for action selection in interact_with_env."""
    max_episode_steps: int = 1000
    env_steps_per_iter: int = 250
    """Real env frames to collect before re-entering training."""
    env_actor_max_batch_size: int = 4
    """Max envs to forward together during stateful action selection."""

    # Dream (DreamTrainer-style)
    dream_horizon: int = 15
    dream_batch_size: int = 32
    dream_updates_per_iter: int = 0
    """Legacy fixed dream updates per iteration. Set >0 to override ratio scheduling."""
    dream_updates_per_env_step: float = 0.5
    """Dream policy/value updates to schedule per real env frame collected."""
    dream_num_steps: int = 4
    dream_start_timesteps: int = 0
    """Legacy minimum real-step warmup override for dream RL."""

    # Logging
    log_freq: int = 10000


# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────

class ReplayBuffer:
    """Stores raw transitions. Samples contiguous sequences for WM training."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)

        if obs.ndim > 1:
            batch = obs.shape[0]
            assert action.ndim > 1 and action.shape[0] == batch
            assert reward.shape[0] == batch and done.shape[0] == batch
            for i in range(batch):
                self.add(obs[i], action[i], reward[i], done[i])
            return

        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, dones):
        for i in range(obs.shape[0]):
            self.add(obs[i], actions[i], rewards[i], dones[i])

    def sample_sequences(self, batch_size: int, seq_len: int, device='cpu'):
        assert self.size >= seq_len, f"need at least seq_len={seq_len} items in replay, got size={self.size}"
        obs_seqs, act_seqs, rew_seqs = [], [], []
        for _ in range(batch_size):
            for _attempt in range(100):
                start = random.randint(0, self.size - seq_len)
                end = start + seq_len
                if self.size == self.capacity and start <= self.pos < end:
                    continue
                indices = np.arange(start, end)
                if not self.dones[indices[:-1]].any():
                    break
            else:
                start = random.randint(0, self.size - seq_len)
                indices = np.arange(start, start + seq_len)
            obs_seqs.append(self.obs[indices])
            act_seqs.append(self.actions[indices])
            rew_seqs.append(self.rewards[indices])
        return (
            torch.tensor(np.array(obs_seqs), dtype=torch.float32, device=device),
            torch.tensor(np.array(act_seqs), dtype=torch.float32, device=device),
            torch.tensor(np.array(rew_seqs), dtype=torch.float32, device=device),
        )


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    observation_clip: float = 0.0,
    reward_clip: float = 0.0,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize_observations:
            env = gym.wrappers.NormalizeObservation(env)
        if observation_clip > 0:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -observation_clip, observation_clip))
        if normalize_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        if reward_clip > 0:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -reward_clip, reward_clip))
        return env
    return thunk


# ──────────────────────────────────────────────
# Env actor — stateful MuJoCo adaptation of reference interact_with_env
# ──────────────────────────────────────────────

class EnvActor:
    """Stateful policy/value interface mirroring the reference env interaction pattern."""

    def __init__(
        self,
        world_model: DynamicsWorldModel,
        tokenizer: StateTokenizer,
        num_steps_denoising: int = 4,
        use_time_cache: bool = True,
        max_batch_size: int = 4,
    ):
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.num_steps_denoising = num_steps_denoising
        self.use_time_cache = use_time_cache
        self.max_batch_size = max_batch_size

        self.obs_histories: list[Tensor] = []
        self.action_histories: list[Tensor] = []
        self.reward_histories: list[Tensor] = []
        self.time_caches: list | None = []
        self.last_state_entropy_bonus: Tensor | None = None

    @property
    def device(self):
        return self.world_model.device

    def _step_size(self) -> int:
        assert self.num_steps_denoising > 0
        assert self.world_model.max_steps % self.num_steps_denoising == 0
        return self.world_model.max_steps // self.num_steps_denoising

    def _to_obs_tensor(self, obs: Tensor | np.ndarray) -> Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs

    def reset(self, obs: Tensor | np.ndarray):
        obs = self._to_obs_tensor(obs)
        batch = obs.shape[0]
        act_dim = self.world_model.action_embedder.num_actions

        self.obs_histories = [obs[i:i+1].clone() for i in range(batch)]
        self.action_histories = [torch.empty((0, act_dim), device=self.device) for _ in range(batch)]
        self.reward_histories = [torch.empty((0,), device=self.device) for _ in range(batch)]
        self.time_caches = [None for _ in range(batch)]
        self.last_state_entropy_bonus = torch.zeros(batch, device=self.device)

    def _ensure_initialized(self, obs: Tensor | np.ndarray):
        obs = self._to_obs_tensor(obs)
        if not self.obs_histories or len(self.obs_histories) != obs.shape[0]:
            self.reset(obs)
        elif all(
            hist.shape[0] == 1 and acts.numel() == 0 and rews.numel() == 0
            for hist, acts, rews in zip(self.obs_histories, self.action_histories, self.reward_histories)
        ):
            self.obs_histories = [obs[i:i+1].clone() for i in range(obs.shape[0])]
        return obs

    @torch.no_grad()
    def _slice_time_cache(self, time_cache, batch_index: int):
        if not exists(time_cache):
            return None

        next_kv_cache = None
        if exists(time_cache.next_kv_cache):
            next_kv_cache = time_cache.next_kv_cache[:, batch_index:batch_index + 1].contiguous()

        next_rnn_hiddens = None
        if exists(time_cache.next_rnn_hiddens):
            next_rnn_hiddens = time_cache.next_rnn_hiddens[:, :, batch_index:batch_index + 1].contiguous()

        return TransformerIntermediates(
            next_kv_cache,
            None,
            None,
            next_rnn_hiddens,
            [],
        )

    @torch.no_grad()
    def _stack_time_caches(self, env_indices: list[int]):
        if not self.use_time_cache:
            return None

        caches = [self.time_caches[idx] for idx in env_indices]
        if not caches or any(cache is None for cache in caches):
            return None

        next_kv_cache = None
        if exists(caches[0].next_kv_cache):
            next_kv_cache = torch.cat([cache.next_kv_cache for cache in caches], dim=1)

        next_rnn_hiddens = None
        if exists(caches[0].next_rnn_hiddens):
            next_rnn_hiddens = torch.cat([cache.next_rnn_hiddens for cache in caches], dim=2)

        return TransformerIntermediates(
            next_kv_cache,
            None,
            None,
            next_rnn_hiddens,
            [],
        )

    @torch.no_grad()
    def _forward_batch(
        self,
        obs_batch: Tensor,
        rewards: Tensor | None = None,
        continuous_actions: Tensor | None = None,
        time_cache=None,
        advance_cache: bool = False,
        env_indices: list[int] | None = None,
    ):
        latents = self.tokenizer.tokenize(obs_batch)

        _, (embeds, next_time_cache) = self.world_model(
            latents=latents,
            signal_levels=self.world_model.max_steps - 1,
            step_sizes=self._step_size(),
            rewards=rewards,
            continuous_actions=continuous_actions,
            time_cache=time_cache,
            latent_is_noised=True,
            return_pred_only=True,
            return_intermediates=True,
        )

        if advance_cache and self.use_time_cache and exists(env_indices) and exists(next_time_cache):
            split_caches = [self._slice_time_cache(next_time_cache, batch_index) for batch_index in range(obs_batch.shape[0])]
            for env_index, env_cache in zip(env_indices, split_caches):
                self.time_caches[env_index] = env_cache

        agent_embed = embeds.agent[:, -1, 0]
        value_bins = self.world_model.value_head(agent_embed)
        value = self.world_model.reward_encoder.bins_to_scalar_value(value_bins)
        policy_embed = self.world_model.policy_head(agent_embed)

        state_entropy_bonus = torch.zeros(obs_batch.shape[0], device=self.device)
        if self.world_model.add_state_entropy_bonus:
            state_pred = self.world_model.to_state_pred(embeds.state_pred)
            state_entropy_bonus = self.world_model.state_beta_dist(state_pred).entropy().reshape(obs_batch.shape[0], -1).sum(dim=-1)

        return policy_embed, value, state_entropy_bonus

    @torch.no_grad()
    def _forward_env_group(self, env_indices: list[int], advance_cache: bool = True):
        obs_batch = torch.stack([self.obs_histories[idx] for idx in env_indices], dim=0)

        rewards = None
        if self.reward_histories[env_indices[0]].numel() > 0:
            rewards = torch.stack([self.reward_histories[idx] for idx in env_indices], dim=0)

        continuous_actions = None
        if self.action_histories[env_indices[0]].numel() > 0:
            continuous_actions = torch.stack([self.action_histories[idx] for idx in env_indices], dim=0)

        return self._forward_batch(
            obs_batch,
            rewards=rewards,
            continuous_actions=continuous_actions,
            time_cache=self._stack_time_caches(env_indices),
            advance_cache=advance_cache,
            env_indices=env_indices,
        )

    def _group_env_indices(self) -> list[list[int]]:
        grouped: dict[tuple[int, bool], list[int]] = {}
        for env_index, obs_history in enumerate(self.obs_histories):
            has_cache = bool(self.use_time_cache and self.time_caches[env_index] is not None)
            key = (obs_history.shape[0], has_cache)
            grouped.setdefault(key, []).append(env_index)
        return list(grouped.values())

    def _chunk_env_indices(self, env_indices: list[int]) -> list[list[int]]:
        if self.max_batch_size <= 0 or len(env_indices) <= self.max_batch_size:
            return [env_indices]
        return [
            env_indices[start:start + self.max_batch_size]
            for start in range(0, len(env_indices), self.max_batch_size)
        ]

    @torch.no_grad()
    def get_action(self, obs: Tensor | np.ndarray | None = None, temperature: float = 1.0) -> Tensor:
        if exists(obs):
            obs = self._to_obs_tensor(obs)
            policy_embed, _, _ = self._forward_batch(obs.unsqueeze(1))
            return self.world_model.action_embedder.sample(
                policy_embed,
                pred_head_index=0,
                temperature=temperature,
            )

        num_envs = len(self.obs_histories)
        act_dim = self.world_model.action_embedder.num_actions
        actions = torch.empty((num_envs, act_dim), device=self.device)
        bonuses = torch.zeros(num_envs, device=self.device)

        for env_indices in self._group_env_indices():
            for env_chunk in self._chunk_env_indices(env_indices):
                policy_embed, _, state_entropy_bonus = self._forward_env_group(env_chunk, advance_cache=True)
                sampled_actions = self.world_model.action_embedder.sample(
                    policy_embed,
                    pred_head_index=0,
                    temperature=temperature,
                )
                actions[env_chunk] = sampled_actions
                bonuses[env_chunk] = state_entropy_bonus

        self.last_state_entropy_bonus = bonuses
        return actions

    @torch.no_grad()
    def get_value(self, obs: Tensor | np.ndarray | None = None) -> Tensor:
        if exists(obs):
            obs = self._to_obs_tensor(obs)
            _, values, _ = self._forward_batch(obs.unsqueeze(1))
            return values

        num_envs = len(self.obs_histories)
        values = torch.empty(num_envs, device=self.device)
        for env_indices in self._group_env_indices():
            for env_chunk in self._chunk_env_indices(env_indices):
                _, group_values, _ = self._forward_env_group(env_chunk, advance_cache=False)
                values[env_chunk] = group_values
        return values

    @torch.no_grad()
    def act(self, temperature: float = 1.0) -> Tensor:
        return self.get_action(obs=None, temperature=temperature)

    @torch.no_grad()
    def value(self) -> Tensor:
        return self.get_value(obs=None)

    @torch.no_grad()
    def observe(
        self,
        actions: Tensor | np.ndarray,
        rewards: Tensor | np.ndarray,
        next_obs: Tensor | np.ndarray,
        dones: Tensor | np.ndarray,
    ):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = self._to_obs_tensor(next_obs)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        for env_index in range(len(self.obs_histories)):
            action = actions[env_index:env_index + 1]
            reward = rewards[env_index:env_index + 1]
            obs = next_obs[env_index:env_index + 1]

            self.action_histories[env_index] = torch.cat((self.action_histories[env_index], action), dim=0)
            self.reward_histories[env_index] = torch.cat((self.reward_histories[env_index], reward), dim=0)

            if dones[env_index]:
                self.obs_histories[env_index] = obs.clone()
                self.action_histories[env_index] = self.action_histories[env_index][:0]
                self.reward_histories[env_index] = self.reward_histories[env_index][:0]
                self.time_caches[env_index] = None
            else:
                self.obs_histories[env_index] = torch.cat((self.obs_histories[env_index], obs), dim=0)


# ──────────────────────────────────────────────
# Collect env steps — raw transitions for replay + WM policy for actions
# ──────────────────────────────────────────────

@torch.no_grad()
def collect_env_steps(
    actor: EnvActor,
    envs,
    obs: np.ndarray,
    replay: ReplayBuffer,
    num_steps: int,
    writer: SummaryWriter,
    global_step: int,
    use_random_policy: bool = False,
):
    num_envs = obs.shape[0]
    actor._ensure_initialized(obs)

    steps_collected = 0
    while steps_collected < num_steps:
        if use_random_policy:
            actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)], dtype=np.float32)
            reward_bonus = np.zeros(num_envs, dtype=np.float32)
        else:
            action_tensor = actor.act(temperature=1.0)
            actions = action_tensor.cpu().numpy()
            actions = np.clip(actions, envs.single_action_space.low, envs.single_action_space.high)
            reward_bonus = actor.last_state_entropy_bonus.detach().cpu().numpy()

        next_obs, rewards_np, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        rewards_np = rewards_np + reward_bonus

        replay.add_batch(obs, actions, rewards_np, dones.astype(np.float32))
        actor.observe(actions, rewards_np, next_obs, dones)

        steps_collected += num_envs
        global_step += num_envs

        writer.add_scalar("charts/step_reward", float(np.mean(rewards_np)), global_step)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    ep_ret = float(info["episode"]["r"])
                    ep_len = float(info["episode"]["l"])
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)
                    print(f"global_step={global_step}, episodic_return={ep_ret:.1f}")

        obs = next_obs

    return obs, global_step


@torch.no_grad()
def evaluate_world_model_probe(
    tokenizer: StateTokenizer,
    world_model: DynamicsWorldModel,
    probe_batch: tuple[Tensor, Tensor, Tensor] | None,
    device: torch.device,
) -> Tensor | None:
    if not exists(probe_batch):
        return None

    probe_obs, probe_actions, probe_rewards = probe_batch

    tok_was_training = tokenizer.training
    wm_was_training = world_model.training

    tokenizer.eval()
    world_model.eval()

    latents = tokenizer.tokenize(probe_obs.to(device))
    probe_loss = world_model(
        latents=latents,
        rewards=probe_rewards.to(device),
        continuous_actions=probe_actions.to(device),
    )

    if tok_was_training:
        tokenizer.train()
    if wm_was_training:
        world_model.train()

    return probe_loss


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                normalize_observations=args.normalize_observations,
                normalize_rewards=args.normalize_rewards,
                observation_clip=args.observation_clip,
                reward_clip=args.reward_clip,
            )
            for i in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    print(f"Environment: {args.env_id}, obs_dim={obs_dim}, act_dim={act_dim}, num_envs={args.num_envs}")

    # Models
    tokenizer = StateTokenizer(
        dim_obs=obs_dim, dim_latent=args.wm_dim_latent,
        num_latent_tokens=args.wm_num_latent_tokens,
        dim_hidden=args.tok_dim_hidden,
        encoder_depth=args.tok_encoder_depth,
        decoder_depth=args.tok_decoder_depth,
    ).to(device)

    world_model = DynamicsWorldModel(
        dim=args.wm_dim, dim_latent=args.wm_dim_latent,
        state_tokenizer=tokenizer, dim_obs=obs_dim,
        max_steps=args.wm_max_steps,
        num_register_tokens=args.wm_num_register_tokens,
        num_spatial_tokens=args.wm_num_spatial_tokens,
        num_latent_tokens=args.wm_num_latent_tokens,
        depth=args.wm_depth, pred_orig_latent=args.wm_pred_orig_latent,
        time_block_every=args.wm_time_block_every,
        attn_heads=args.wm_attn_heads, attn_dim_head=args.wm_attn_dim_head,
        use_time_rnn=args.wm_use_time_rnn,
        num_continuous_actions=act_dim,
        multi_token_pred_len=args.wm_multi_token_pred_len,
        ppo_eps_clip=args.ppo_eps_clip,
        gae_discount_factor=args.gae_gamma,
        gae_lambda=args.gae_lambda,
        policy_entropy_weight=args.policy_entropy_weight,
        pmpo_kl_div_loss_weight=args.pmpo_kl_weight,
        use_delight_gating=True,
        value_head_mlp_depth=3,
        policy_head_mlp_depth=3,
    ).to(device)

    tok_params = sum(p.numel() for p in tokenizer.parameters())
    wm_total = sum(p.numel() for p in world_model.parameters())
    print(f"Tokenizer: {tok_params:,}, World model: {wm_total - tok_params:,}, Total: {wm_total:,}")

    # Optimizers
    tok_opt = optim.AdamW(tokenizer.parameters(), lr=args.wm_learning_rate, weight_decay=args.wm_weight_decay)
    wm_params_list = [p for n, p in world_model.named_parameters() if not n.startswith('state_tokenizer.')]
    wm_opt = optim.AdamW(wm_params_list, lr=args.wm_learning_rate, weight_decay=args.wm_weight_decay)
    policy_opt = optim.AdamW(world_model.policy_head_parameters(), lr=args.policy_learning_rate)
    value_opt = optim.AdamW(world_model.value_head_parameters(), lr=args.value_learning_rate)

    # Replay buffer
    replay = ReplayBuffer(args.replay_buffer_size, obs_dim, act_dim)

    # ─── Main loop ───
    obs, _ = envs.reset(seed=args.seed)
    actor = EnvActor(
        world_model,
        tokenizer,
        num_steps_denoising=args.num_steps_denoising,
        use_time_cache=True,
        max_batch_size=args.env_actor_max_batch_size,
    )
    actor.reset(obs)
    global_step = 0
    start_time = time.time()
    last_log = 0
    tok_loss = torch.tensor(float('nan'), device=device)
    wm_loss = torch.tensor(float('nan'), device=device)
    policy_loss = torch.tensor(float('nan'), device=device)
    value_loss = torch.tensor(float('nan'), device=device)
    wm_probe_loss = torch.tensor(float('nan'), device=device)
    wm_probe_batch = None
    best_wm_probe_loss = float('inf')
    wm_probe_bad_iters = 0
    wm_probe_evals = 0
    dream_ready = False
    wm_update_budget = 0.0
    dream_update_budget = 0.0
    min_dream_start_step = max(args.wm_warmup_min_steps, args.dream_start_timesteps)

    while global_step < args.total_timesteps:

        # ══════════════════════════════════════
        # Phase 1: Collect env experience
        # ══════════════════════════════════════
        use_random = (not dream_ready) or replay.size < args.min_replay_size
        prev_global_step = global_step

        obs, global_step = collect_env_steps(
            actor, envs, obs, replay,
            num_steps=args.env_steps_per_iter,
            writer=writer,
            global_step=global_step,
            use_random_policy=use_random,
        )
        steps_collected = global_step - prev_global_step
        if args.wm_updates_per_iter <= 0:
            wm_update_budget += steps_collected * args.wm_updates_per_env_step

        if replay.size < args.min_replay_size:
            continue

        if not exists(wm_probe_batch):
            min_probe_replay = max(args.min_replay_size, args.wm_probe_batch_size * args.replay_seq_len)
            if replay.size >= min_probe_replay:
                wm_probe_batch = tuple(
                    tensor.cpu() for tensor in replay.sample_sequences(
                        args.wm_probe_batch_size,
                        args.replay_seq_len,
                        device=device,
                    )
                )

        # ══════════════════════════════════════
        # Phase 2: Train world model (BehaviorClone-style)
        # ══════════════════════════════════════
        world_model.train()

        if args.wm_updates_per_iter > 0:
            wm_updates = args.wm_updates_per_iter
        else:
            wm_updates = int(wm_update_budget)
            wm_update_budget -= wm_updates

        for _ in range(wm_updates):
            obs_seq, act_seq, rew_seq = replay.sample_sequences(
                args.wm_batch_size, args.replay_seq_len, device=device
            )

            # Tokenizer
            tok_loss, _ = tokenizer(obs_seq)
            tok_opt.zero_grad()
            tok_loss.backward()
            nn.utils.clip_grad_norm_(tokenizer.parameters(), args.wm_max_grad_norm)
            tok_opt.step()

            # World model
            with torch.no_grad():
                latents = tokenizer.tokenize(obs_seq)

            wm_loss = world_model(
                latents=latents,
                rewards=rew_seq,
                continuous_actions=act_seq,
            )

            wm_opt.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(wm_params_list, args.wm_max_grad_norm)
            wm_opt.step()

        if exists(wm_probe_batch):
            wm_probe_loss_eval = evaluate_world_model_probe(
                tokenizer,
                world_model,
                wm_probe_batch,
                device,
            )
            if exists(wm_probe_loss_eval):
                wm_probe_loss = wm_probe_loss_eval.detach()
                writer.add_scalar("losses/world_model_probe", wm_probe_loss.item(), global_step)

                if (not dream_ready) and global_step >= min_dream_start_step:
                    wm_probe_evals += 1
                    probe_value = wm_probe_loss.item()

                    improved = False
                    if not np.isfinite(best_wm_probe_loss):
                        improved = True
                    else:
                        required_delta = abs(best_wm_probe_loss) * args.wm_readiness_min_improvement
                        improved = probe_value < (best_wm_probe_loss - required_delta)

                    if improved:
                        best_wm_probe_loss = probe_value
                        wm_probe_bad_iters = 0
                    else:
                        wm_probe_bad_iters += 1

                    ready_by_threshold = (
                        args.wm_readiness_loss_threshold > 0. and
                        probe_value <= args.wm_readiness_loss_threshold
                    )
                    ready_by_plateau = (
                        wm_probe_evals >= args.wm_readiness_min_evals and
                        wm_probe_bad_iters >= args.wm_readiness_patience
                    )

                    if ready_by_threshold or ready_by_plateau:
                        dream_ready = True
                        dream_update_budget = 0.0
                        writer.add_scalar("charts/dream_ready", 1.0, global_step)
                        print(
                            f"Dream training enabled at step={global_step} "
                            f"(wm_probe={probe_value:.4f}, best={best_wm_probe_loss:.4f})"
                        )

        # ══════════════════════════════════════
        # Phase 3: Dream + train policy (DreamTrainer-style)
        # ══════════════════════════════════════
        if dream_ready:
            if args.dream_updates_per_iter > 0:
                dream_updates = args.dream_updates_per_iter
            else:
                dream_update_budget += steps_collected * args.dream_updates_per_env_step
                dream_updates = int(dream_update_budget)
                dream_update_budget -= dream_updates

            for _ in range(dream_updates):
                world_model.eval()

                dreams = world_model.generate(
                    time_steps=args.dream_horizon + 1,  # +1 for bootstrap value
                    num_steps=args.dream_num_steps,
                    batch_size=args.dream_batch_size,
                    return_for_policy_optimization=True,
                    continuous_temperature=1.0,
                )

                world_model.train()

                policy_loss, value_loss = world_model.learn_from_experience(
                    dreams,
                    only_learn_policy_value_heads=True,
                    use_pmpo=args.use_pmpo,
                )

                # Policy update
                policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(world_model.policy_head_parameters(), args.policy_max_grad_norm)
                policy_opt.step()

                # Value update
                value_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(world_model.value_head_parameters(), args.policy_max_grad_norm)
                value_opt.step()

        # ══════════════════════════════════════
        # Logging
        # ══════════════════════════════════════
        if global_step - last_log >= args.log_freq:
            last_log = global_step
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/dream_ready", float(dream_ready), global_step)
            if torch.isfinite(tok_loss):
                writer.add_scalar("losses/tokenizer", tok_loss.item(), global_step)
                tok_str = f"{tok_loss.item():.4f}"
            else:
                tok_str = "pending"
            if torch.isfinite(wm_loss):
                writer.add_scalar("losses/world_model", wm_loss.item(), global_step)
                wm_str = f"{wm_loss.item():.4f}"
            else:
                wm_str = "pending"
            if torch.isfinite(wm_probe_loss):
                writer.add_scalar("losses/world_model_probe", wm_probe_loss.item(), global_step)
            if dream_ready and torch.isfinite(policy_loss) and torch.isfinite(value_loss):
                writer.add_scalar("losses/policy", policy_loss.item(), global_step)
                writer.add_scalar("losses/value", value_loss.item(), global_step)
                policy_str = f"{policy_loss.item():.4f}"
                value_str = f"{value_loss.item():.4f}"
            else:
                policy_str = "warmup"
                value_str = "warmup"
            probe_str = f"{wm_probe_loss.item():.4f}" if torch.isfinite(wm_probe_loss) else "pending"
            print(
                f"step={global_step}, SPS={sps}, tok={tok_str}, "
                f"wm={wm_str}, wm_probe={probe_str}, pi={policy_str}, v={value_str}"
            )

    envs.close()
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
