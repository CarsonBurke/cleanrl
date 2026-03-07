# Dreamer-style imagination training for the latent-imagination lineage.
#
# v6 showed that separating PPO and world-model optimization was necessary but
# not sufficient: once actor-side imagined advantages entered PPO directly, the
# imagined signal stayed nearly uncorrelated with real advantages and the real
# critic quality collapsed.
#
# v7 changes the contract:
# 1. PPO goes back to learning only from real GAE targets.
# 2. The world model is updated separately and is frozen during imagination
#    training.
# 3. The actor improves in a dedicated imagination phase using short latent
#    rollouts, a separate imagination value head, a Dreamer4-style sign-based
#    PMPO loss, and a KL prior to a frozen copy of the current policy.
#
# This keeps PPO as the real-data trust-region optimizer while letting imagined
# rollouts train the behavior through their own objective instead of trying to
# masquerade as on-policy advantages.
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
import torch.optim as optim
import tyro
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.imagination import latent_imagination_v5_core as v5
from cleanrl.imagination import latent_imagination_v6_core as v6


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


class Agent(v5.Agent):
    def __init__(
        self,
        envs,
        latent_dim: int = 64,
        model_hidden_dim: int = 128,
        model_min_std: float = 0.05,
        model_max_std: float = 1.0,
        use_done_model: bool = False,
    ):
        super().__init__(
            envs,
            latent_dim=latent_dim,
            model_hidden_dim=model_hidden_dim,
            model_min_std=model_min_std,
            model_max_std=model_max_std,
            use_done_model=use_done_model,
        )
        self.imagination_value = nn.Sequential(
            v5.layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            v5.layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_imagination_value_from_latent(self, latent):
        return self.imagination_value(latent)


@dataclass
class ImaginationPhaseLosses:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    prior_kl_loss: torch.Tensor
    mean_return: torch.Tensor
    mean_advantage: torch.Tensor
    positive_fraction: torch.Tensor


def behavior_parameters(agent: Agent):
    params = v6.actor_critic_parameters(agent)
    params.extend(agent.imagination_value.parameters())
    return params


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


def build_policy_prior(agent: Agent):
    prior_actor_mean = copy.deepcopy(agent.actor_mean).eval()
    for parameter in prior_actor_mean.parameters():
        parameter.requires_grad = False
    prior_logstd = agent.actor_logstd.detach().clone()
    return prior_actor_mean, prior_logstd


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


def compute_imagination_phase_losses(
    agent: Agent,
    start_obs: torch.Tensor,
    args: Args,
    prior_actor_mean: nn.Module,
    prior_logstd: torch.Tensor,
) -> ImaginationPhaseLosses:
    with torch.no_grad():
        latent = agent.encode(start_obs)

    logprobs = []
    prior_kls = []
    rewards = []
    continuations = []
    values = []
    alive_weights = []

    alive = torch.ones(latent.shape[0], device=latent.device)
    for _ in range(args.imagination_horizon):
        dist = agent.get_dist_from_latent(latent)
        action = dist.rsample()
        logprobs.append(dist.log_prob(action).sum(-1))

        prior_mean = prior_actor_mean(latent)
        prior_std = torch.exp(prior_logstd.expand_as(prior_mean))
        prior_dist = Normal(prior_mean, prior_std)
        prior_kls.append(kl_divergence(dist, prior_dist).sum(-1))

        values.append(agent.get_imagination_value_from_latent(latent).squeeze(-1))
        alive_weights.append(alive)

        with torch.no_grad():
            env_action = agent.clamp_action(action.detach())
            mean, std = agent.transition_params(latent, env_action)
            next_latent = mean + torch.randn_like(std) * std
            reward_pred, done_logit = agent.predict_reward_done(latent, env_action, next_latent)
            if done_logit is None:
                continuation = torch.ones_like(reward_pred)
            else:
                continuation = 1.0 - torch.sigmoid(done_logit)

        rewards.append(reward_pred)
        continuations.append(continuation)
        alive = alive * continuation
        latent = next_latent

    with torch.no_grad():
        bootstrap = agent.get_imagination_value_from_latent(latent).squeeze(-1)

    logprobs = torch.stack(logprobs)
    prior_kls = torch.stack(prior_kls)
    rewards = torch.stack(rewards)
    continuations = torch.stack(continuations)
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

    policy_loss = args.imagination_alpha * pos_loss + (1.0 - args.imagination_alpha) * neg_loss
    value_loss = 0.5 * (flat_weights * (values.reshape(-1) - returns.reshape(-1)) ** 2).sum() / flat_weights.sum().clamp_min(1e-6)
    prior_kl_loss = (flat_weights * flat_prior_kls).sum() / flat_weights.sum().clamp_min(1e-6)
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


def main(args_class=Args):
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

    agent = Agent(
        envs,
        latent_dim=args.latent_dim,
        model_hidden_dim=args.model_hidden_dim,
        model_min_std=args.model_min_std,
        model_max_std=args.model_max_std,
        use_done_model=args.use_done_model,
    ).to(device)
    behavior_optimizer = optim.Adam(
        behavior_parameters(agent),
        lr=v6.resolve_learning_rate(args.behavior_learning_rate, args.learning_rate),
        eps=1e-5,
    )
    world_model_optimizer = optim.Adam(
        v6.world_model_parameters(agent),
        lr=v6.resolve_learning_rate(args.world_model_learning_rate, args.learning_rate),
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
            behavior_lr = frac * v6.resolve_learning_rate(args.behavior_learning_rate, args.learning_rate)
            behavior_optimizer.param_groups[0]["lr"] = behavior_lr
        if args.world_model_anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            wm_lr = frac * v6.resolve_learning_rate(args.world_model_learning_rate, args.learning_rate)
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

        batch = v6.flatten_rollout_batch(
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

        behavior_inds = np.arange(args.batch_size)
        clipfracs = []
        last_behavior_losses = None
        for epoch in range(args.update_epochs):
            np.random.shuffle(behavior_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = behavior_inds[start:end]

                behavior_losses = v6.compute_behavior_losses(agent, batch, mb_inds, args, batch.advantages)
                behavior_optimizer.zero_grad()
                behavior_losses.total_loss.backward()
                nn.utils.clip_grad_norm_(behavior_parameters(agent), args.max_grad_norm)
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

                world_model_losses = v6.compute_world_model_losses(agent, batch, mb_inds, args, device, global_step)
                world_model_optimizer.zero_grad()
                world_model_losses.scaled_loss.backward()
                nn.utils.clip_grad_norm_(v6.world_model_parameters(agent), args.world_model_max_grad_norm)
                world_model_optimizer.step()

                last_world_model_losses = world_model_losses

        imagination_coef = current_imagination_phase_coef(args, global_step)
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
        if imagination_coef > 0.0:
            prior_actor_mean, prior_logstd = build_policy_prior(agent)
            prior_actor_mean = prior_actor_mean.to(device)
            imagination_batch_size = min(args.imagination_num_contexts, args.batch_size)
            imagination_inds = np.arange(args.batch_size)
            for _ in range(args.imagination_update_epochs):
                np.random.shuffle(imagination_inds)
                start_obs = batch.obs[imagination_inds[:imagination_batch_size]]
                imagination_losses = compute_imagination_phase_losses(
                    agent,
                    start_obs,
                    args,
                    prior_actor_mean,
                    prior_logstd,
                )
                behavior_optimizer.zero_grad()
                (imagination_coef * imagination_losses.total_loss).backward()
                nn.utils.clip_grad_norm_(behavior_parameters(agent), args.max_grad_norm)
                behavior_optimizer.step()
                last_imagination_losses = imagination_losses

        y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
        writer.add_scalar("imagination/phase_coef", imagination_coef, global_step)
        writer.add_scalar("imagination/policy_loss", last_imagination_losses.policy_loss.item(), global_step)
        writer.add_scalar("imagination/value_loss", last_imagination_losses.value_loss.item(), global_step)
        writer.add_scalar("imagination/prior_kl", last_imagination_losses.prior_kl_loss.item(), global_step)
        writer.add_scalar("imagination/total_loss", last_imagination_losses.total_loss.item(), global_step)
        writer.add_scalar("imagination/mean_return", last_imagination_losses.mean_return.item(), global_step)
        writer.add_scalar("imagination/mean_advantage", last_imagination_losses.mean_advantage.item(), global_step)
        writer.add_scalar("imagination/positive_fraction", last_imagination_losses.positive_fraction.item(), global_step)
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
                Agent,
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
