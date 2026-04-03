"""Imagination v8.7 — Dreamer-style straight-through actor optimization

v8.5/v8.6 failed because they treated the world model as a black-box simulator
(REINFORCE/detached transitions). The world model is DIFFERENTIABLE — we should
backprop through it.

v8.7: the imagination actor is trained by backpropagating predicted reward
through the frozen world model's forward pass into the actor's parameters.
No REINFORCE, no log-probs, no punishment. The gradient directly says
"shift your action this way to increase predicted reward."

Architecture:
- v8.3 base (RMSNorm+SiLU encoder, world model aux loss)
- Separate imagination actor (clone of real actor)
- Imagination training: roll out imagination actor through DIFFERENTIABLE
  world model (frozen params, gradients flow through forward pass)
- Loss = -mean(predicted_returns) over H-step rollouts
- Imagination actor's discovered improvements distilled to real actor via
  soft KL loss (real policy pulled toward imagination policy)
- No critic augmentation (v8.5's blending didn't help) — imagination
  purely improves the actor
"""
import os
import random
import time
import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter


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
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 1
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

    # World model
    latent_dim: int = 64
    model_hidden_dim: int = 128
    model_min_std: float = 0.05
    model_max_std: float = 1.0
    world_model_lr: float = 3e-4
    world_model_update_epochs: int = 1
    world_model_max_grad_norm: float = 1.0
    target_encoder_tau: float = 0.005

    # Imagination
    imagine_horizon: int = 5
    imagine_warmup_steps: int = 200000
    imagine_lr: float = 1e-4
    imagine_update_epochs: int = 3
    imagine_batch_size: int = 512
    imagine_entropy_coef: float = 1e-3
    distill_coef: float = 0.5

    # Computed
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
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        ld = args.latent_dim
        mhd = args.model_hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            RMSNorm(128),
            nn.SiLU(),
            layer_init(nn.Linear(128, 128)),
            RMSNorm(128),
            nn.SiLU(),
            layer_init(nn.Linear(128, ld)),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Real actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(ld, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(ld, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # World model
        self.transition_backbone = nn.Sequential(
            layer_init(nn.Linear(ld + act_dim, mhd)), nn.SiLU(),
            layer_init(nn.Linear(mhd, mhd)), nn.SiLU(),
        )
        self.transition_mean = layer_init(nn.Linear(mhd, ld), std=0.01)
        self.transition_logstd = layer_init(nn.Linear(mhd, ld), std=0.01)
        self.reward_model = nn.Sequential(
            layer_init(nn.Linear(2 * ld + act_dim, mhd)), nn.SiLU(),
            layer_init(nn.Linear(mhd, 1), std=0.01),
        )

        # Imagination actor — clone of real actor
        self.imagine_actor_mean = copy.deepcopy(self.actor_mean)
        self.imagine_actor_logstd = nn.Parameter(self.actor_logstd.detach().clone())

        self.model_min_std = args.model_min_std
        self.model_max_std = args.model_max_std
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def encode(self, obs):
        return self.encoder(obs)

    def encode_target(self, obs):
        with torch.no_grad():
            return self.target_encoder(obs)

    def clamp_action(self, action):
        return torch.max(torch.min(action, self.action_high), self.action_low)

    @torch.no_grad()
    def update_target_encoder(self, tau):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.lerp_(p.data, tau)

    def transition_forward(self, latent, env_action):
        """Differentiable forward pass through transition model. Returns next latent (mean)."""
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        next_latent = latent + self.transition_mean(hidden)
        return next_latent

    def transition_params(self, latent, env_action):
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        mean = latent + self.transition_mean(hidden)
        std_scale = torch.sigmoid(self.transition_logstd(hidden))
        std = self.model_min_std + (self.model_max_std - self.model_min_std) * std_scale
        return mean, std

    def reward_forward(self, latent, env_action, next_latent):
        """Differentiable forward pass through reward model."""
        model_input = torch.cat([latent, env_action, next_latent], dim=-1)
        return self.reward_model(model_input).squeeze(-1)

    def predict_reward(self, latent, env_action, next_latent):
        return self.reward_forward(latent, env_action, next_latent)

    def imagine_rollout(self, start_latent, horizon, gamma):
        """Roll out imagination actor through DIFFERENTIABLE world model.

        Gradients flow: imagine_actor → action → world_model.forward → reward
        World model params are frozen (not in imagination optimizer).
        Encoder is detached (start_latent should be detached).

        Returns total discounted predicted reward (differentiable w.r.t. imagine_actor).
        """
        current_latent = start_latent
        total_return = torch.zeros(start_latent.shape[0], device=start_latent.device)
        total_entropy = torch.zeros(start_latent.shape[0], device=start_latent.device)
        discount = 1.0

        for h in range(horizon):
            # Imagination actor produces action (rsample for reparameterized gradients)
            action_mean = self.imagine_actor_mean(current_latent)
            action_logstd = self.imagine_actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            action = dist.rsample()
            total_entropy = total_entropy + dist.entropy().sum(-1)

            env_action = self.clamp_action(action)

            # DIFFERENTIABLE world model forward (gradients flow through!)
            next_latent = self.transition_forward(current_latent, env_action)
            reward = self.reward_forward(current_latent, env_action, next_latent)

            total_return = total_return + discount * reward
            discount *= gamma
            current_latent = next_latent

        # Bootstrap from critic at horizon (detached — critic not trained here)
        with torch.no_grad():
            bootstrap = self.critic(current_latent).squeeze(-1)
        total_return = total_return + discount * bootstrap

        return total_return, total_entropy / horizon

    def get_value(self, obs):
        return self.critic(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        latent = self.encode(obs)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        value = self.critic(latent)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

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
        return params

    def imagination_parameters(self):
        params = []
        params.extend(self.imagine_actor_mean.parameters())
        params.append(self.imagine_actor_logstd)
        return params


def compute_distill_loss(agent, latent):
    """KL(real_policy || imagine_policy) — pull real actor toward imagination discoveries."""
    with torch.no_grad():
        img_mean = agent.imagine_actor_mean(latent)
        img_logstd = agent.imagine_actor_logstd.expand_as(img_mean)
        img_dist = Normal(img_mean, torch.exp(img_logstd))

    real_mean = agent.actor_mean(latent)
    real_logstd = agent.actor_logstd.expand_as(real_mean)
    real_dist = Normal(real_mean, torch.exp(real_logstd))

    # KL(real || imagine) — minimized by moving real toward imagine
    return kl_divergence(real_dist, img_dist).sum(-1).mean()


def compute_world_model_loss(agent, obs, actions, next_obs, rewards, device):
    latent = agent.encode(obs)
    env_action = agent.clamp_action(actions)
    target_next_latent = agent.encode_target(next_obs)

    pred_mean, pred_std = agent.transition_params(latent, env_action)
    trans_dist = Normal(pred_mean, pred_std)
    transition_loss = -trans_dist.log_prob(target_next_latent.detach()).mean()

    pred_reward = agent.predict_reward(latent.detach(), env_action, target_next_latent)
    reward_loss = F.mse_loss(pred_reward, rewards)

    return transition_loss + reward_loss, transition_loss, reward_loss


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    behavior_optimizer = optim.Adam(agent.behavior_parameters(), lr=args.learning_rate, eps=1e-5)
    world_model_optimizer = optim.Adam(agent.world_model_parameters(), lr=args.world_model_lr, eps=1e-5)
    imagination_optimizer = optim.Adam(agent.imagination_parameters(), lr=args.imagine_lr, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    use_imagination = False

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            behavior_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            world_model_optimizer.param_groups[0]["lr"] = frac * args.world_model_lr
            imagination_optimizer.param_groups[0]["lr"] = frac * args.imagine_lr

        use_imagination = (global_step >= args.imagine_warmup_steps)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            model_next_obs = next_obs_np.copy()
            if "final_observation" in infos:
                for i, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        model_next_obs[i] = final_obs
            next_obses[step] = torch.Tensor(model_next_obs).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE
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

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
        b_rewards = rewards.reshape(-1)

        # PPO update + distillation from imagination actor
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        last_distill_loss = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Distill imagination actor discoveries into real actor
                if use_imagination:
                    latent_for_distill = agent.encode(b_obs[mb_inds]).detach()
                    distill_loss = compute_distill_loss(agent, latent_for_distill)
                    loss = loss + args.distill_coef * distill_loss
                    last_distill_loss = distill_loss.detach()

                behavior_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.behavior_parameters(), args.max_grad_norm)
                behavior_optimizer.step()
                agent.update_target_encoder(args.target_encoder_tau)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # World model update
        wm_inds = np.arange(args.batch_size)
        last_wm_loss = last_trans_loss = last_rew_loss = torch.zeros((), device=device)
        for _ in range(args.world_model_update_epochs):
            np.random.shuffle(wm_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = wm_inds[start:end]
                wm_loss, trans_loss, rew_loss = compute_world_model_loss(
                    agent, b_obs[mb_inds], b_actions[mb_inds],
                    b_next_obs[mb_inds], b_rewards[mb_inds], device)
                world_model_optimizer.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(agent.world_model_parameters(), args.world_model_max_grad_norm)
                world_model_optimizer.step()
                last_wm_loss = wm_loss.detach()
                last_trans_loss = trans_loss.detach()
                last_rew_loss = rew_loss.detach()

        # Imagination actor update — straight-through differentiable rollout
        last_imagine_return = torch.zeros((), device=device)
        last_imagine_entropy = torch.zeros((), device=device)
        if use_imagination:
            for _ in range(args.imagine_update_epochs):
                sel = np.random.permutation(args.batch_size)[:args.imagine_batch_size]
                with torch.no_grad():
                    start_latents = agent.encode(b_obs[sel])

                # Differentiable rollout through frozen world model
                imagined_return, mean_entropy = agent.imagine_rollout(
                    start_latents, args.imagine_horizon, args.gamma)

                # Loss = maximize returns + entropy bonus
                imagine_loss = -imagined_return.mean() - args.imagine_entropy_coef * mean_entropy.mean()

                imagination_optimizer.zero_grad()
                imagine_loss.backward()
                nn.utils.clip_grad_norm_(agent.imagination_parameters(), args.max_grad_norm)
                imagination_optimizer.step()

                last_imagine_return = imagined_return.mean().detach()
                last_imagine_entropy = mean_entropy.mean().detach()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", behavior_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("world_model/total_loss", last_wm_loss.item(), global_step)
        writer.add_scalar("world_model/transition_loss", last_trans_loss.item(), global_step)
        writer.add_scalar("world_model/reward_loss", last_rew_loss.item(), global_step)
        writer.add_scalar("imagination/active", float(use_imagination), global_step)
        writer.add_scalar("imagination/mean_return", last_imagine_return.item(), global_step)
        writer.add_scalar("imagination/mean_entropy", last_imagine_entropy.item(), global_step)
        writer.add_scalar("imagination/distill_loss", last_distill_loss.item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps} img={'ON' if use_imagination else 'OFF'} img_ret={last_imagine_return.item():.1f} distill={last_distill_loss.item():.3f}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
