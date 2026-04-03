# v8_pmpo: Fork of v8 that replaces PPO's clipped surrogate in the behavior
# phase with PMPO (Policy Mirror Primal Optimization).
#
# Changes from v8:
# 1. Behavior policy loss uses PMPO sign-based advantage splitting instead of
#    PPO clipping: maximize log_prob for positive advantages, minimize for
#    negative, weighted by pmpo_alpha (default 0.5 = equal weight).
# 2. KL divergence regularization between old and new behavior policy replaces
#    the clipping mechanism as the trust region.
# 3. Old policy mean/logstd stored per rollout step for exact KL computation.
# 4. Imagination phase remains unchanged (already uses PMPO-style loss).
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
from cleanrl.imagination import latent_imagination_v6_core as v6


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins: int, bin_range: float) -> torch.Tensor:
    half = torch.linspace(-bin_range, 0.0, num_bins // 2 + 1)
    half = symexp(half)
    if num_bins % 2 == 1:
        return torch.cat([half, -half[:-1].flip(0)])
    return torch.cat([half[:-1], -half[:-1].flip(0)])


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
class Args(v6.Args):
    imagination_start_fraction: float = 0.25
    """fraction of training completed before imagination updates turn on"""
    imagination_ramp_fraction: float = 0.25
    """fraction of training used to ramp imagination loss strength"""
    imagination_loss_coef: float = 1.0
    """overall weight on the imagination-phase objective"""
    imagination_horizon: int = 8
    """latent rollout horizon for imagination training"""
    imagination_update_epochs: int = 1
    """number of imagination passes over sampled starting contexts per iteration"""
    imagination_num_contexts: int = 1024
    """number of real contexts sampled per imagination update"""
    imagination_lambda: float = 0.95
    """lambda-return parameter for imagined value learning"""
    imagination_value_coef: float = 0.5
    """weight on the imagination value loss"""
    imagination_prior_coef: float = 0.3
    """weight on KL to the frozen policy prior during imagination updates"""
    imagination_alpha: float = 0.5
    """PMPO positive/negative balance weight"""
    imagination_learning_rate: float = 0.0
    """imagination-optimizer learning rate; <=0 means use learning_rate"""
    imagination_anneal_lr: bool = False
    """anneal the imagination optimizer learning rate"""
    imagination_num_bins: int = 255
    """number of symexp-twohot bins for the imagination value head"""
    imagination_bin_range: float = 3.0
    """symlog-space range for imagination value bins"""
    context_hidden_dim: int = 64
    """hidden width of the recurrent temporal context state"""
    imagination_bc_coef: float = 1.0
    """behavior-cloning weight on the imagination actor before imagination starts"""
    imagination_bc_after_start_coef: float = 0.05
    """behavior-cloning weight on the imagination actor after imagination starts"""
    pmpo_alpha: float = 0.5
    """PMPO positive/negative advantage weight for behavior policy (0.5 = equal)"""
    pmpo_kl_coef: float = 0.3
    """KL divergence regularization coefficient for behavior PMPO"""
    pmpo_reverse_kl: bool = True
    """use reverse KL (KL(old||new)) for behavior PMPO regularization"""


@dataclass
class FlatBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    env_actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_dones: torch.Tensor
    context_states: torch.Tensor
    multi_action_sequences: torch.Tensor
    multi_reward_targets: torch.Tensor
    multi_continuation_targets: torch.Tensor
    multi_next_obs_targets: torch.Tensor
    multi_valid_masks: torch.Tensor


@dataclass
class BehaviorLosses:
    total_loss: torch.Tensor
    pg_loss: torch.Tensor
    v_loss: torch.Tensor
    entropy_loss: torch.Tensor
    old_approx_kl: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: float


@dataclass
class WorldModelLosses:
    scaled_loss: torch.Tensor
    raw_loss: torch.Tensor
    transition_loss: torch.Tensor
    reward_loss: torch.Tensor
    value_consistency_loss: torch.Tensor
    done_loss: torch.Tensor
    model_coef: float
    transition_std: torch.Tensor


@dataclass
class ImaginationPhaseLosses:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    prior_kl_loss: torch.Tensor
    mean_return: torch.Tensor
    mean_advantage: torch.Tensor
    positive_fraction: torch.Tensor


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
        self.context_rnn = nn.GRUCell(latent_dim + action_dim + 1, context_hidden_dim)
        self.model_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        self.imagination_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        self.imagination_value_context_proj = nn.Linear(context_hidden_dim, latent_dim, bias=False)
        nn.init.zeros_(self.model_context_proj.weight)
        nn.init.zeros_(self.imagination_context_proj.weight)
        nn.init.zeros_(self.imagination_value_context_proj.weight)

        self.imagination_actor_mean = copy.deepcopy(self.actor_mean)
        self.imagination_actor_logstd = nn.Parameter(self.actor_logstd.detach().clone())
        self.imagination_value_head = nn.Sequential(
            v5.layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, imagination_num_bins), std=0.01),
        )
        self.register_buffer("imagination_value_bins", build_symexp_bins(imagination_num_bins, imagination_bin_range))

    def init_context_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.context_hidden_dim, device=device)

    def update_context_state(self, context_state: torch.Tensor, obs: torch.Tensor, env_action: torch.Tensor) -> torch.Tensor:
        latent = self.encode(obs).detach()
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
        context_input = torch.cat([latent, env_action, continuation.unsqueeze(-1)], dim=-1)
        return self.context_rnn(context_input, context_state)

    def get_imagination_dist_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> Normal:
        contextual_latent = latent + self.imagination_context_proj(context_state)
        action_mean = self.imagination_actor_mean(contextual_latent)
        action_logstd = self.imagination_actor_logstd.expand_as(action_mean)
        return Normal(action_mean, torch.exp(action_logstd))

    def get_imagination_value_logits_from_latent_context(
        self, latent: torch.Tensor, context_state: torch.Tensor
    ) -> torch.Tensor:
        contextual_latent = latent + self.imagination_value_context_proj(context_state)
        return self.imagination_value_head(contextual_latent)

    def get_imagination_value_from_latent_context(self, latent: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        logits = self.get_imagination_value_logits_from_latent_context(latent, context_state)
        return twohot_predict(logits, self.imagination_value_bins)

    def contextual_transition_params(
        self, latent: torch.Tensor, context_state: torch.Tensor, env_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contextual_latent = latent + self.model_context_proj(context_state)
        return self.transition_params(contextual_latent, env_action)

    def contextual_predict_reward_done(
        self,
        latent: torch.Tensor,
        context_state: torch.Tensor,
        env_action: torch.Tensor,
        next_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        contextual_latent = latent + self.model_context_proj(context_state)
        contextual_next_latent = next_latent + self.model_context_proj(context_state)
        return self.predict_reward_done(contextual_latent, env_action, contextual_next_latent)

    def behavior_parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.actor_mean.parameters())
        params.append(self.actor_logstd)
        params.extend(self.critic.parameters())
        return params

    def world_model_parameters(self):
        params = []
        params.extend(self.transition_backbone.parameters())
        params.extend(self.transition_mean.parameters())
        params.extend(self.transition_logstd.parameters())
        params.extend(self.reward_model.parameters())
        if self.done_model is not None:
            params.extend(self.done_model.parameters())
        params.extend(self.context_rnn.parameters())
        params.extend(self.model_context_proj.parameters())
        return params

    def imagination_parameters(self):
        params = []
        params.extend(self.imagination_actor_mean.parameters())
        params.append(self.imagination_actor_logstd)
        params.extend(self.imagination_value_head.parameters())
        params.extend(self.imagination_context_proj.parameters())
        params.extend(self.imagination_value_context_proj.parameters())
        return params


def resolve_learning_rate(explicit_lr: float, fallback_lr: float) -> float:
    return fallback_lr if explicit_lr <= 0 else explicit_lr


def behavior_parameters(agent: Agent):
    return agent.behavior_parameters()


def world_model_parameters(agent: Agent):
    return agent.world_model_parameters()


def imagination_parameters(agent: Agent):
    return agent.imagination_parameters()


def current_imagination_phase_coef(args: Args, global_step: int) -> float:
    if args.imagination_loss_coef <= 0:
        return 0.0
    if args.imagination_ramp_fraction <= 0:
        return args.imagination_loss_coef
    start_step = int(args.total_timesteps * args.imagination_start_fraction)
    ramp_steps = max(1, int(args.total_timesteps * args.imagination_ramp_fraction))
    progress = (global_step - start_step) / ramp_steps
    progress = min(1.0, max(0.0, progress))
    return args.imagination_loss_coef * progress


def current_behavior_actor_coef(args: Args, global_step: int) -> float:
    return 1.0


def build_policy_prior(agent: Agent):
    prior_actor_mean = copy.deepcopy(agent.imagination_actor_mean).eval()
    prior_context_proj = copy.deepcopy(agent.imagination_context_proj).eval()
    for parameter in prior_actor_mean.parameters():
        parameter.requires_grad = False
    for parameter in prior_context_proj.parameters():
        parameter.requires_grad = False
    prior_logstd = agent.imagination_actor_logstd.detach().clone()
    return prior_actor_mean, prior_context_proj, prior_logstd


@torch.no_grad()
def recompute_rollout_context_states(agent: Agent, obs: torch.Tensor, env_actions: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    num_steps, num_envs = obs.shape[:2]
    context_states = torch.zeros((num_steps, num_envs, agent.context_hidden_dim), device=obs.device)
    context_state = agent.init_context_state(num_envs, obs.device)
    for step in range(num_steps):
        context_state = context_state * (1.0 - dones[step].unsqueeze(-1))
        context_states[step] = context_state
        context_state = agent.update_context_state(context_state, obs[step], env_actions[step])
    return context_states


def flatten_rollout_batch(
    args: Args,
    envs,
    obs: torch.Tensor,
    actions: torch.Tensor,
    env_actions: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_obses: torch.Tensor,
    next_dones: torch.Tensor,
    context_states: torch.Tensor,
) -> FlatBatch:
    (
        multi_action_sequences,
        multi_reward_targets,
        multi_continuation_targets,
        multi_next_obs_targets,
        multi_valid_masks,
    ) = v5.build_multi_horizon_targets(
        rewards,
        next_obses,
        next_dones,
        env_actions,
        args.multi_horizon_steps,
        args.gamma,
    )

    return FlatBatch(
        obs=obs.reshape((-1,) + envs.single_observation_space.shape),
        actions=actions.reshape((-1,) + envs.single_action_space.shape),
        env_actions=env_actions.reshape((-1,) + envs.single_action_space.shape),
        logprobs=logprobs.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=values.reshape(-1),
        rewards=rewards.reshape(-1),
        next_obs=next_obses.reshape((-1,) + envs.single_observation_space.shape),
        next_dones=next_dones.reshape(-1),
        context_states=context_states.reshape((-1, context_states.shape[-1])),
        multi_action_sequences=multi_action_sequences.reshape(
            multi_action_sequences.shape[0],
            args.batch_size,
            *envs.single_action_space.shape,
        ),
        multi_reward_targets=multi_reward_targets.reshape(len(args.multi_horizon_steps), args.batch_size),
        multi_continuation_targets=multi_continuation_targets.reshape(len(args.multi_horizon_steps), args.batch_size),
        multi_next_obs_targets=multi_next_obs_targets.reshape(
            (len(args.multi_horizon_steps), args.batch_size) + envs.single_observation_space.shape
        ),
        multi_valid_masks=multi_valid_masks.reshape(len(args.multi_horizon_steps), args.batch_size),
    )


def compute_behavior_losses(
    agent: Agent,
    batch: FlatBatch,
    mb_inds: np.ndarray,
    args: Args,
    actor_coef: float,
    behavior_prior: Agent = None,
) -> BehaviorLosses:
    clip_lo = getattr(args, 'clip_coef_low', args.clip_coef)
    clip_hi = getattr(args, 'clip_coef_high', args.clip_coef)
    if actor_coef > 0.0:
        # Single forward pass: get per-dim log_probs and distribution for PMPO
        latent = agent.encode(batch.obs[mb_inds])
        new_dist = agent.get_dist_from_latent(latent)
        log_prob_per_dim = new_dist.log_prob(batch.actions[mb_inds])  # (mb, act_dim)
        newlogprob = log_prob_per_dim.sum(1)
        entropy = new_dist.entropy().sum(1)
        newvalue = agent.get_value_from_latent(latent)

        logratio = newlogprob - batch.logprobs[mb_inds]
        ratio = logratio.exp()
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio < (1 - clip_lo)) | (ratio > (1 + clip_hi))).float().mean().item()

        mb_advantages = batch.advantages[mb_inds]
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # PMPO: sign-based advantage splitting with per-dim masked mean
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        pos_mask_expanded = pos_mask.unsqueeze(-1).expand_as(log_prob_per_dim)
        neg_mask_expanded = neg_mask.unsqueeze(-1).expand_as(log_prob_per_dim)

        alpha = args.pmpo_alpha
        pos_count = pos_mask_expanded.sum()
        neg_count = neg_mask_expanded.sum()

        pos_loss = log_prob_per_dim[pos_mask_expanded].sum() / pos_count.clamp(min=1)
        neg_loss = -log_prob_per_dim[neg_mask_expanded].sum() / neg_count.clamp(min=1)
        pg_loss = -(alpha * pos_loss + (1.0 - alpha) * neg_loss)

        # KL divergence regularization against frozen prior (d4-style)
        if args.pmpo_kl_coef > 0.0 and behavior_prior is not None:
            with torch.no_grad():
                prior_latent = behavior_prior.encode(batch.obs[mb_inds])
                prior_dist = behavior_prior.get_dist_from_latent(prior_latent)
            if args.pmpo_reverse_kl:
                kl = kl_divergence(prior_dist, new_dist).sum(-1).mean()
            else:
                kl = kl_divergence(new_dist, prior_dist).sum(-1).mean()
            pg_loss = pg_loss + args.pmpo_kl_coef * kl

        entropy_loss = entropy.mean()
    else:
        newvalue = agent.get_value(batch.obs[mb_inds])
        zero = torch.zeros((), device=batch.obs.device)
        old_approx_kl = zero
        approx_kl = zero
        clipfrac = 0.0
        pg_loss = zero
        entropy_loss = zero

    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
        v_clipped = batch.values[mb_inds] + torch.clamp(
            newvalue - batch.values[mb_inds],
            -clip_lo,
            clip_hi,
        )
        v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

    total_loss = actor_coef * (pg_loss - args.ent_coef * entropy_loss) + args.vf_coef * v_loss
    return BehaviorLosses(
        total_loss=total_loss,
        pg_loss=pg_loss,
        v_loss=v_loss,
        entropy_loss=entropy_loss,
        old_approx_kl=old_approx_kl,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
    )


def compute_world_model_losses(
    agent: Agent,
    batch: FlatBatch,
    mb_inds: np.ndarray,
    args: Args,
    device: torch.device,
    global_step: int,
) -> WorldModelLosses:
    latent = agent.encode(batch.obs[mb_inds])
    model_latent = latent.detach() if args.detach_model_encoder else latent
    context_state = batch.context_states[mb_inds]

    if args.use_multi_horizon_model_loss:
        transition_loss_terms = []
        reward_loss_terms = []
        done_loss_terms = []
        value_consistency_terms = []
        transition_std_value = torch.zeros((), device=device)

        for horizon_index, horizon in enumerate(args.multi_horizon_steps):
            valid_mask = batch.multi_valid_masks[horizon_index, mb_inds] > 0.5
            if not torch.any(valid_mask):
                continue

            horizon_latent = model_latent[valid_mask]
            horizon_context = context_state[valid_mask]
            discounted_reward_pred = torch.zeros(horizon_latent.shape[0], device=device)
            continuation_pred = torch.ones(horizon_latent.shape[0], device=device)
            discount = 1.0

            first_step_mean = None
            first_step_std = None
            for offset in range(horizon):
                horizon_actions = batch.multi_action_sequences[offset, mb_inds][valid_mask]
                pred_next_mean, pred_next_std = agent.contextual_transition_params(
                    horizon_latent, horizon_context, horizon_actions
                )
                reward_pred, done_logit = agent.contextual_predict_reward_done(
                    horizon_latent, horizon_context, horizon_actions, pred_next_mean
                )

                if offset == 0:
                    first_step_mean = pred_next_mean
                    first_step_std = pred_next_std
                    transition_std_value = pred_next_std.mean()

                discounted_reward_pred = discounted_reward_pred + discount * continuation_pred * reward_pred
                if done_logit is None:
                    step_continue_prob = torch.ones_like(reward_pred)
                else:
                    step_continue_prob = 1.0 - torch.sigmoid(done_logit)

                continuation_pred = continuation_pred * step_continue_prob
                discount *= args.gamma
                horizon_context = agent.step_context_state_from_latent(
                    horizon_context, pred_next_mean, horizon_actions, step_continue_prob
                )
                horizon_latent = pred_next_mean

            target_next_latent = agent.encode_target(batch.multi_next_obs_targets[horizon_index, mb_inds][valid_mask])
            target_reward = batch.multi_reward_targets[horizon_index, mb_inds][valid_mask]
            target_continuation = batch.multi_continuation_targets[horizon_index, mb_inds][valid_mask]

            if horizon == 1 and first_step_mean is not None and first_step_std is not None:
                transition_loss_terms.append(v5.gaussian_nll(target_next_latent, first_step_mean, first_step_std).mean())
            else:
                transition_loss_terms.append(0.5 * ((horizon_latent - target_next_latent) ** 2).mean())

            reward_loss_terms.append(F.mse_loss(discounted_reward_pred, target_reward))
            if agent.use_done_model:
                done_loss_terms.append(
                    F.binary_cross_entropy(
                        continuation_pred.clamp(1e-6, 1.0 - 1e-6),
                        target_continuation,
                    )
                )
            predicted_next_value = agent.get_value_from_latent(horizon_latent).view(-1)
            target_next_value = agent.get_value_from_latent(target_next_latent).view(-1).detach()
            value_consistency_terms.append(0.5 * ((predicted_next_value - target_next_value) ** 2).mean())

        transition_loss = (
            torch.stack(transition_loss_terms).mean() if transition_loss_terms else torch.zeros((), device=device)
        )
        reward_loss = torch.stack(reward_loss_terms).mean() if reward_loss_terms else torch.zeros((), device=device)
        done_loss = torch.stack(done_loss_terms).mean() if done_loss_terms else torch.zeros((), device=device)
        value_consistency_loss = (
            torch.stack(value_consistency_terms).mean() if value_consistency_terms else torch.zeros((), device=device)
        )
        raw_loss = args.multi_horizon_coef * (
            args.transition_coef * transition_loss
            + args.reward_coef * reward_loss
            + args.value_consistency_coef * value_consistency_loss
            + args.done_coef * done_loss
        )
    else:
        target_next_latent = agent.encode_target(batch.next_obs[mb_inds])
        pred_next_mean, pred_next_std = agent.contextual_transition_params(model_latent, context_state, batch.env_actions[mb_inds])
        reward_pred, done_logit = agent.contextual_predict_reward_done(
            model_latent,
            context_state,
            batch.env_actions[mb_inds],
            target_next_latent,
        )

        transition_loss = v5.gaussian_nll(target_next_latent, pred_next_mean, pred_next_std).mean()
        reward_loss = F.mse_loss(reward_pred, batch.rewards[mb_inds])
        if done_logit is None:
            done_loss = torch.zeros((), device=device)
        else:
            done_loss = F.binary_cross_entropy_with_logits(done_logit, batch.next_dones[mb_inds])
        predicted_next_value = agent.get_value_from_latent(pred_next_mean).view(-1)
        target_next_value = agent.get_value_from_latent(target_next_latent).view(-1).detach()
        value_consistency_loss = 0.5 * ((predicted_next_value - target_next_value) ** 2).mean()
        raw_loss = (
            args.transition_coef * transition_loss
            + args.reward_coef * reward_loss
            + args.value_consistency_coef * value_consistency_loss
            + args.done_coef * done_loss
        )
        transition_std_value = pred_next_std.mean()

    model_coef = v5.current_model_coef(args, global_step)
    return WorldModelLosses(
        scaled_loss=raw_loss * model_coef,
        raw_loss=raw_loss,
        transition_loss=transition_loss,
        reward_loss=reward_loss,
        value_consistency_loss=value_consistency_loss,
        done_loss=done_loss,
        model_coef=model_coef,
        transition_std=transition_std_value,
    )


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


def compute_imagination_bc_loss(
    agent: Agent,
    obs: torch.Tensor,
    context_states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    latent = agent.encode(obs).detach()
    dist = agent.get_imagination_dist_from_latent_context(latent, context_states)
    return -dist.log_prob(actions).sum(-1).mean()


def compute_imagination_phase_losses(
    agent: Agent,
    start_obs: torch.Tensor,
    start_context: torch.Tensor,
    args: Args,
    prior_actor_mean: nn.Module,
    prior_context_proj: nn.Module,
    prior_logstd: torch.Tensor,
) -> ImaginationPhaseLosses:
    with torch.no_grad():
        latent = agent.encode(start_obs)
        context_state = start_context

    logprobs = []
    prior_kls = []
    rewards = []
    continuations = []
    value_logits = []
    values = []
    alive_weights = []

    alive = torch.ones(latent.shape[0], device=latent.device)
    for _ in range(args.imagination_horizon):
        dist = agent.get_imagination_dist_from_latent_context(latent, context_state)
        action = dist.rsample()
        logprobs.append(dist.log_prob(action).sum(-1))

        prior_latent = latent + prior_context_proj(context_state)
        prior_mean = prior_actor_mean(prior_latent)
        prior_std = torch.exp(prior_logstd.expand_as(prior_mean))
        prior_dist = Normal(prior_mean, prior_std)
        prior_kls.append(kl_divergence(dist, prior_dist).sum(-1))

        step_value_logits = agent.get_imagination_value_logits_from_latent_context(latent, context_state)
        value_logits.append(step_value_logits)
        values.append(twohot_predict(step_value_logits, agent.imagination_value_bins))
        alive_weights.append(alive)

        with torch.no_grad():
            env_action = agent.clamp_action(action.detach())
            mean, std = agent.contextual_transition_params(latent, context_state, env_action)
            next_latent = mean + torch.randn_like(std) * std
            reward_pred, done_logit = agent.contextual_predict_reward_done(latent, context_state, env_action, next_latent)
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
        bootstrap = agent.get_imagination_value_from_latent_context(latent, context_state)

    logprobs = torch.stack(logprobs)
    prior_kls = torch.stack(prior_kls)
    rewards = torch.stack(rewards)
    continuations = torch.stack(continuations)
    value_logits = torch.stack(value_logits)
    values = torch.stack(values)
    alive_weights = torch.stack(alive_weights).detach()

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
    flat_prior_kls = prior_kls.reshape(-1)
    flat_weights = alive_weights.reshape(-1)

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

    flat_value_logits = value_logits.reshape(-1, value_logits.shape[-1])
    flat_return_targets = twohot_encode(returns.reshape(-1), agent.imagination_value_bins)
    value_loss = (flat_weights * twohot_loss(flat_value_logits, flat_return_targets)).sum() / flat_weights.sum().clamp_min(1e-6)
    prior_kl_loss = (flat_weights * flat_prior_kls).sum() / flat_weights.sum().clamp_min(1e-6)
    policy_loss = args.imagination_alpha * pos_loss + (1.0 - args.imagination_alpha) * neg_loss
    total_loss = policy_loss + args.imagination_value_coef * value_loss + args.imagination_prior_coef * prior_kl_loss

    return ImaginationPhaseLosses(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        prior_kl_loss=prior_kl_loss,
        mean_return=returns[0].mean(),
        mean_advantage=advantages.mean(),
        positive_fraction=positive_mask.float().mean(),
    )


def main(args_class=Args, agent_class=None):
    args = tyro.cli(args_class)
    assert args.imag_horizon > 0, "imag_horizon must be positive"
    assert args.imag_branches > 0, "imag_branches must be positive"
    assert args.imagination_horizon > 0, "imagination_horizon must be positive"
    args.multi_horizon_steps = tuple(sorted({int(step) for step in args.multi_horizon_steps if int(step) > 0}))
    assert len(args.multi_horizon_steps) > 0, "multi_horizon_steps must contain at least one positive horizon"
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
        [v5.make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    _agent_class = agent_class or Agent
    agent = _agent_class(
        envs,
        latent_dim=args.latent_dim,
        model_hidden_dim=args.model_hidden_dim,
        model_min_std=args.model_min_std,
        model_max_std=args.model_max_std,
        use_done_model=args.use_done_model,
        context_hidden_dim=args.context_hidden_dim,
        imagination_num_bins=args.imagination_num_bins,
        imagination_bin_range=args.imagination_bin_range,
    ).to(device)
    behavior_optimizer = optim.Adam(
        behavior_parameters(agent),
        lr=resolve_learning_rate(args.behavior_learning_rate, args.learning_rate),
        eps=1e-5,
    )
    world_model_optimizer = optim.Adam(
        world_model_parameters(agent),
        lr=resolve_learning_rate(args.world_model_learning_rate, args.learning_rate),
        eps=1e-5,
    )
    imagination_optimizer = optim.Adam(
        imagination_parameters(agent),
        lr=resolve_learning_rate(args.imagination_learning_rate, args.learning_rate),
        eps=1e-5,
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    env_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    online_context_state = agent.init_context_state(args.num_envs, device)
    imagination_prior = None

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            behavior_optimizer.param_groups[0]["lr"] = frac * resolve_learning_rate(args.behavior_learning_rate, args.learning_rate)
        if args.world_model_anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            world_model_optimizer.param_groups[0]["lr"] = frac * resolve_learning_rate(args.world_model_learning_rate, args.learning_rate)
        if args.imagination_anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            imagination_optimizer.param_groups[0]["lr"] = frac * resolve_learning_rate(args.imagination_learning_rate, args.learning_rate)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                value = value.flatten()
                executed_action = agent.clamp_action(action)
                values[step] = value
            actions[step] = action
            env_actions[step] = executed_action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(executed_action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            model_next_obs_np = v5.extract_model_next_obs(next_obs_np, infos)

            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obses[step] = torch.tensor(model_next_obs_np, device=device)
            next_dones[step] = torch.tensor(next_done_np, device=device)

            with torch.no_grad():
                online_context_state = agent.update_context_state(online_context_state, obs[step], executed_action)
                online_context_state = online_context_state * (
                    1.0 - torch.tensor(next_done_np, device=device, dtype=torch.float32).unsqueeze(-1)
                )

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

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
                    nextnonterminal = 1.0 - next_dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        with torch.no_grad():
            rollout_context_states = recompute_rollout_context_states(agent, obs, env_actions, dones)
        batch = flatten_rollout_batch(
            args,
            envs,
            obs,
            actions,
            env_actions,
            logprobs,
            advantages,
            returns,
            values,
            rewards,
            next_obses,
            next_dones,
            rollout_context_states,
        )

        imagination_coef = current_imagination_phase_coef(args, global_step)
        actor_coef = current_behavior_actor_coef(args, global_step)
        if imagination_coef > 0.0 and imagination_prior is None:
            imagination_prior = build_policy_prior(agent)
            imagination_prior = (
                imagination_prior[0].to(device),
                imagination_prior[1].to(device),
                imagination_prior[2].to(device),
            )

        behavior_inds = np.arange(args.batch_size)
        clipfracs = []
        last_behavior_losses = None

        # Frozen behavior prior for PMPO KL (d4-style: snapshot once before all epochs)
        behavior_prior = copy.deepcopy(agent).eval()
        for p in behavior_prior.parameters():
            p.requires_grad = False

        for epoch in range(args.update_epochs):
            np.random.shuffle(behavior_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = behavior_inds[start:end]

                behavior_losses = compute_behavior_losses(agent, batch, mb_inds, args, actor_coef, behavior_prior)
                behavior_optimizer.zero_grad()
                behavior_losses.total_loss.backward()
                nn.utils.clip_grad_norm_(behavior_parameters(agent), args.max_grad_norm)
                behavior_optimizer.step()
                agent.update_target_encoder(args.target_encoder_tau)

                clipfracs.append(behavior_losses.clipfrac)
                last_behavior_losses = behavior_losses

            if actor_coef > 0.0 and args.target_kl is not None and last_behavior_losses.approx_kl > args.target_kl:
                break

        world_model_inds = np.arange(args.batch_size)
        last_world_model_losses = None
        for _ in range(args.world_model_update_epochs):
            np.random.shuffle(world_model_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = world_model_inds[start:end]

                world_model_losses = compute_world_model_losses(agent, batch, mb_inds, args, device, global_step)
                world_model_optimizer.zero_grad()
                world_model_losses.scaled_loss.backward()
                nn.utils.clip_grad_norm_(world_model_parameters(agent), args.world_model_max_grad_norm)
                world_model_optimizer.step()
                last_world_model_losses = world_model_losses

        bc_coef = args.imagination_bc_coef if imagination_coef <= 0.0 else args.imagination_bc_after_start_coef
        last_bc_loss = torch.zeros((), device=device)
        if bc_coef > 0.0:
            bc_inds = np.random.permutation(args.batch_size)[: min(args.imagination_num_contexts, args.batch_size)]
            bc_loss = compute_imagination_bc_loss(agent, batch.obs[bc_inds], batch.context_states[bc_inds], batch.actions[bc_inds])
            imagination_optimizer.zero_grad()
            (bc_coef * bc_loss).backward()
            nn.utils.clip_grad_norm_(imagination_parameters(agent), args.max_grad_norm)
            imagination_optimizer.step()
            last_bc_loss = bc_loss.detach()

        zero = torch.zeros((), device=device)
        last_imagination_losses = ImaginationPhaseLosses(
            total_loss=zero,
            policy_loss=zero,
            value_loss=zero,
            prior_kl_loss=zero,
            mean_return=zero,
            mean_advantage=zero,
            positive_fraction=zero,
        )
        if imagination_coef > 0.0 and imagination_prior is not None:
            imagination_batch_size = min(args.imagination_num_contexts, args.batch_size)
            imagination_inds = np.arange(args.batch_size)
            for _ in range(args.imagination_update_epochs):
                np.random.shuffle(imagination_inds)
                sel = imagination_inds[:imagination_batch_size]
                imagination_losses = compute_imagination_phase_losses(
                    agent,
                    batch.obs[sel],
                    batch.context_states[sel],
                    args,
                    imagination_prior[0],
                    imagination_prior[1],
                    imagination_prior[2],
                )
                imagination_optimizer.zero_grad()
                (imagination_coef * imagination_losses.total_loss).backward()
                nn.utils.clip_grad_norm_(imagination_parameters(agent), args.max_grad_norm)
                imagination_optimizer.step()
                last_imagination_losses = imagination_losses

        y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if last_behavior_losses is None or last_world_model_losses is None:
            raise RuntimeError("behavior and world model losses must be computed at least once per iteration")

        writer.add_scalar("charts/learning_rate", behavior_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/world_model_learning_rate", world_model_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/imagination_learning_rate", imagination_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/value_loss", last_behavior_losses.v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", last_behavior_losses.pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", last_behavior_losses.entropy_loss.item(), global_step)
        writer.add_scalar("losses/behavior_loss", last_behavior_losses.total_loss.item(), global_step)
        writer.add_scalar("losses/model_loss", last_world_model_losses.raw_loss.item(), global_step)
        writer.add_scalar("losses/model_scaled_loss", last_world_model_losses.scaled_loss.item(), global_step)
        writer.add_scalar("losses/model_coef", last_world_model_losses.model_coef, global_step)
        writer.add_scalar("losses/transition_loss", last_world_model_losses.transition_loss.item(), global_step)
        writer.add_scalar("losses/reward_loss", last_world_model_losses.reward_loss.item(), global_step)
        writer.add_scalar("losses/value_consistency_loss", last_world_model_losses.value_consistency_loss.item(), global_step)
        writer.add_scalar("losses/done_loss", last_world_model_losses.done_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", last_behavior_losses.old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", last_behavior_losses.approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("imagination/phase_coef", imagination_coef, global_step)
        writer.add_scalar("imagination/bc_loss", last_bc_loss.item(), global_step)
        writer.add_scalar("imagination/policy_loss", last_imagination_losses.policy_loss.item(), global_step)
        writer.add_scalar("imagination/value_loss", last_imagination_losses.value_loss.item(), global_step)
        writer.add_scalar("imagination/prior_kl", last_imagination_losses.prior_kl_loss.item(), global_step)
        writer.add_scalar("imagination/total_loss", last_imagination_losses.total_loss.item(), global_step)
        writer.add_scalar("imagination/mean_return", last_imagination_losses.mean_return.item(), global_step)
        writer.add_scalar("imagination/mean_advantage", last_imagination_losses.mean_advantage.item(), global_step)
        writer.add_scalar("imagination/positive_fraction", last_imagination_losses.positive_fraction.item(), global_step)
        writer.add_scalar("imagination/transition_std", last_world_model_losses.transition_std.item(), global_step)
        writer.add_scalar("imagination/behavior_actor_coef", actor_coef, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            v5.make_env,
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
            ),
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


if __name__ == "__main__":
    main()
