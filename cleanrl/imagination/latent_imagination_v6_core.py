# World-model-first refactor for the latent-imagination lineage.
#
# v5 still behaves mostly like "PPO plus a heavy auxiliary world model":
# policy, value, and model losses are summed into one backward pass, which
# makes it hard to tell whether failures come from the world model itself or
# from gradient interference inside PPO's shared update.
#
# v6 keeps the same rollout collection and imagined-advantage recipe as a
# baseline, but restructures training around two explicit learners:
# 1. a behavior learner (encoder + actor + critic) updated from real PPO data
#    plus a small imagined correction, and
# 2. a world model (transition + reward + continuation heads) updated in a
#    separate phase with its own optimizer.
#
# This is still compatible with PPO, but it is a better substrate for the next
# steps where imagination becomes the main learner and PPO becomes only the
# trust-region shell around real-data behavior updates.
import os
import random
import time
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.imagination import latent_imagination_v5_core as v5


@dataclass
class Args(v5.Args):
    behavior_learning_rate: float = 0.0
    """actor-critic learning rate; <=0 means use learning_rate"""
    world_model_learning_rate: float = 0.0
    """world-model learning rate; <=0 means use learning_rate"""
    world_model_update_epochs: int = 1
    """additional passes over the batch for the world model each iteration"""
    world_model_anneal_lr: bool = False
    """anneal the world-model learning rate alongside PPO"""
    world_model_max_grad_norm: float = 1.0
    """gradient clipping for world-model-only updates"""


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
    multi_action_sequences: torch.Tensor
    multi_reward_targets: torch.Tensor
    multi_continuation_targets: torch.Tensor
    multi_next_obs_targets: torch.Tensor
    multi_valid_masks: torch.Tensor


@dataclass
class ImaginationStats:
    imag_coef: float
    policy_advantages: torch.Tensor
    imagined_returns: torch.Tensor
    imagined_stds: torch.Tensor
    imagined_advantages: torch.Tensor
    imagined_advantages_capped: torch.Tensor
    imagined_actor_signal: torch.Tensor
    conf_gate: torch.Tensor
    sign_gate: torch.Tensor
    total_gate: torch.Tensor


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


def resolve_learning_rate(explicit_lr: float, fallback_lr: float) -> float:
    return fallback_lr if explicit_lr <= 0 else explicit_lr


def actor_critic_parameters(agent: v5.Agent):
    params = []
    params.extend(agent.encoder.parameters())
    params.extend(agent.actor_mean.parameters())
    params.append(agent.actor_logstd)
    params.extend(agent.critic.parameters())
    return params


def world_model_parameters(agent: v5.Agent):
    params = []
    params.extend(agent.transition_backbone.parameters())
    params.extend(agent.transition_mean.parameters())
    params.extend(agent.transition_logstd.parameters())
    params.extend(agent.reward_model.parameters())
    if agent.done_model is not None:
        params.extend(agent.done_model.parameters())
    return params


def flatten_rollout_batch(args: Args, envs, obs, actions, env_actions, logprobs, advantages, returns, values, rewards, next_obses, next_dones):
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


@torch.no_grad()
def compute_imagination_stats(agent: v5.Agent, batch: FlatBatch, args: Args, global_step: int) -> ImaginationStats:
    imag_coef = v5.current_imagination_coef(args, global_step)
    if imag_coef <= 0.0:
        zeros = torch.zeros_like(batch.values)
        return ImaginationStats(
            imag_coef=0.0,
            policy_advantages=batch.advantages,
            imagined_returns=zeros,
            imagined_stds=zeros,
            imagined_advantages=zeros,
            imagined_advantages_capped=zeros,
            imagined_actor_signal=zeros,
            conf_gate=zeros,
            sign_gate=zeros,
            total_gate=zeros,
        )

    if args.use_multi_horizon_actor_stats:
        horizon_returns = agent.imagine_multihorizon_returns(
            batch.obs,
            batch.env_actions,
            horizons=args.multi_horizon_steps,
            branches=args.imag_branches,
            gamma=args.gamma,
        )
        horizon_returns_mean = horizon_returns.mean(dim=-1)
        imagined_returns = horizon_returns_mean.mean(dim=0)
        imagined_stds = horizon_returns_mean.std(dim=0, unbiased=False)
    else:
        imagined_returns, imagined_stds = agent.imagine_returns(
            batch.obs,
            batch.env_actions,
            horizon=args.imag_horizon,
            branches=args.imag_branches,
            gamma=args.gamma,
        )

    imagined_advantages = imagined_returns - batch.values
    (
        imagined_advantages_capped,
        conf_gate,
        sign_gate,
        total_gate,
    ) = v5.gated_imagination_advantages(
        args,
        batch.advantages,
        imagined_advantages,
        imagined_stds,
    )

    if args.sign_only_imag_actor:
        imagined_actor_signal = torch.sign(imagined_advantages_capped)
    else:
        imagined_actor_signal = imagined_advantages_capped

    if args.use_imag_conf_gate:
        policy_advantages = batch.advantages + imag_coef * total_gate * imagined_actor_signal
    else:
        policy_advantages = batch.advantages + imag_coef * imagined_actor_signal

    return ImaginationStats(
        imag_coef=imag_coef,
        policy_advantages=policy_advantages,
        imagined_returns=imagined_returns,
        imagined_stds=imagined_stds,
        imagined_advantages=imagined_advantages,
        imagined_advantages_capped=imagined_advantages_capped,
        imagined_actor_signal=imagined_actor_signal,
        conf_gate=conf_gate,
        sign_gate=sign_gate,
        total_gate=total_gate,
    )


def compute_behavior_losses(agent: v5.Agent, batch: FlatBatch, mb_inds: np.ndarray, args: Args, policy_advantages: torch.Tensor) -> BehaviorLosses:
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(batch.obs[mb_inds], batch.actions[mb_inds])
    logratio = newlogprob - batch.logprobs[mb_inds]
    ratio = logratio.exp()

    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    mb_advantages = policy_advantages[mb_inds]
    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
        v_clipped = batch.values[mb_inds] + torch.clamp(
            newvalue - batch.values[mb_inds],
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    total_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
    return BehaviorLosses(
        total_loss=total_loss,
        pg_loss=pg_loss,
        v_loss=v_loss,
        entropy_loss=entropy_loss,
        old_approx_kl=old_approx_kl,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
    )


def compute_world_model_losses(agent: v5.Agent, batch: FlatBatch, mb_inds: np.ndarray, args: Args, device: torch.device, global_step: int) -> WorldModelLosses:
    latent = agent.encode(batch.obs[mb_inds])
    model_latent = latent.detach() if args.detach_model_encoder else latent

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
            discounted_reward_pred = torch.zeros(horizon_latent.shape[0], device=device)
            continuation_pred = torch.ones(horizon_latent.shape[0], device=device)
            discount = 1.0

            first_step_mean = None
            first_step_std = None
            for offset in range(horizon):
                horizon_actions = batch.multi_action_sequences[offset, mb_inds][valid_mask]
                pred_next_mean, pred_next_std = agent.transition_params(horizon_latent, horizon_actions)
                reward_pred, done_logit = agent.predict_reward_done(horizon_latent, horizon_actions, pred_next_mean)

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
                horizon_latent = pred_next_mean

            target_next_latent = agent.encode_target(batch.multi_next_obs_targets[horizon_index, mb_inds][valid_mask])
            target_reward = batch.multi_reward_targets[horizon_index, mb_inds][valid_mask]
            target_continuation = batch.multi_continuation_targets[horizon_index, mb_inds][valid_mask]

            if horizon == 1 and first_step_mean is not None and first_step_std is not None:
                transition_loss_terms.append(v5.gaussian_nll(target_next_latent, first_step_mean, first_step_std).mean())
            else:
                transition_loss_terms.append(0.5 * ((horizon_latent - target_next_latent) ** 2).mean())

            reward_loss_terms.append(torch.nn.functional.mse_loss(discounted_reward_pred, target_reward))
            if agent.use_done_model:
                done_loss_terms.append(
                    torch.nn.functional.binary_cross_entropy(
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
        pred_next_mean, pred_next_std = agent.transition_params(model_latent, batch.env_actions[mb_inds])
        reward_pred, done_logit = agent.predict_reward_done(model_latent, batch.env_actions[mb_inds], target_next_latent)

        transition_loss = v5.gaussian_nll(target_next_latent, pred_next_mean, pred_next_std).mean()
        reward_loss = torch.nn.functional.mse_loss(reward_pred, batch.rewards[mb_inds])
        if done_logit is None:
            done_loss = torch.zeros((), device=device)
        else:
            done_loss = torch.nn.functional.binary_cross_entropy_with_logits(done_logit, batch.next_dones[mb_inds])
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


def main(args_class=Args):
    args = tyro.cli(args_class)
    assert args.imag_horizon > 0, "imag_horizon must be positive"
    assert args.imag_branches > 0, "imag_branches must be positive"
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

    agent = v5.Agent(
        envs,
        latent_dim=args.latent_dim,
        model_hidden_dim=args.model_hidden_dim,
        model_min_std=args.model_min_std,
        model_max_std=args.model_max_std,
        use_done_model=args.use_done_model,
    ).to(device)
    behavior_optimizer = optim.Adam(
        actor_critic_parameters(agent),
        lr=resolve_learning_rate(args.behavior_learning_rate, args.learning_rate),
        eps=1e-5,
    )
    world_model_optimizer = optim.Adam(
        world_model_parameters(agent),
        lr=resolve_learning_rate(args.world_model_learning_rate, args.learning_rate),
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

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            behavior_lr = frac * resolve_learning_rate(args.behavior_learning_rate, args.learning_rate)
            behavior_optimizer.param_groups[0]["lr"] = behavior_lr
        if args.world_model_anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            wm_lr = frac * resolve_learning_rate(args.world_model_learning_rate, args.learning_rate)
            world_model_optimizer.param_groups[0]["lr"] = wm_lr

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
        )
        imagination = compute_imagination_stats(agent, batch, args, global_step)

        behavior_inds = np.arange(args.batch_size)
        clipfracs = []
        last_behavior_losses = None
        for epoch in range(args.update_epochs):
            np.random.shuffle(behavior_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = behavior_inds[start:end]

                behavior_losses = compute_behavior_losses(agent, batch, mb_inds, args, imagination.policy_advantages)
                behavior_optimizer.zero_grad()
                behavior_losses.total_loss.backward()
                nn.utils.clip_grad_norm_(actor_critic_parameters(agent), args.max_grad_norm)
                behavior_optimizer.step()
                agent.update_target_encoder(args.target_encoder_tau)

                clipfracs.append(behavior_losses.clipfrac)
                last_behavior_losses = behavior_losses

            if args.target_kl is not None and last_behavior_losses.approx_kl > args.target_kl:
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

        y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        real_adv_np = batch.advantages.cpu().numpy()
        imag_adv_np = imagination.imagined_advantages.cpu().numpy()
        if np.std(real_adv_np) > 1e-8 and np.std(imag_adv_np) > 1e-8:
            imag_adv_corr = float(np.corrcoef(real_adv_np, imag_adv_np)[0, 1])
        else:
            imag_adv_corr = 0.0

        if last_behavior_losses is None or last_world_model_losses is None:
            raise RuntimeError("behavior and world model losses must be computed at least once per iteration")

        writer.add_scalar("charts/learning_rate", behavior_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/world_model_learning_rate", world_model_optimizer.param_groups[0]["lr"], global_step)
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("imagination/mix_coef", imagination.imag_coef, global_step)
        writer.add_scalar("imagination/imagined_return", imagination.imagined_returns.mean().item(), global_step)
        writer.add_scalar("imagination/imagined_return_std", imagination.imagined_stds.mean().item(), global_step)
        writer.add_scalar("imagination/imagined_advantage", imagination.imagined_advantages.mean().item(), global_step)
        writer.add_scalar(
            "imagination/imagined_advantage_capped",
            imagination.imagined_advantages_capped.mean().item(),
            global_step,
        )
        writer.add_scalar("imagination/imagined_actor_signal", imagination.imagined_actor_signal.mean().item(), global_step)
        writer.add_scalar("imagination/advantage_correlation", imag_adv_corr, global_step)
        writer.add_scalar("imagination/conf_gate", imagination.conf_gate.mean().item(), global_step)
        writer.add_scalar("imagination/sign_gate", imagination.sign_gate.mean().item(), global_step)
        writer.add_scalar("imagination/total_gate", imagination.total_gate.mean().item(), global_step)
        writer.add_scalar("imagination/transition_std", last_world_model_losses.transition_std.item(), global_step)
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
                v5.Agent,
                latent_dim=args.latent_dim,
                model_hidden_dim=args.model_hidden_dim,
                model_min_std=args.model_min_std,
                model_max_std=args.model_max_std,
                use_done_model=args.use_done_model,
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
