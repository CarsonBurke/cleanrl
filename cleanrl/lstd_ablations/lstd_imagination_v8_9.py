"""Imagination v8.9 — Pure Dreamer: AC trained only through imagination

No PPO. Real experience only trains the world model. Actor and critic learn
entirely through imagined rollouts via the differentiable world model.

Phase 1 (warmup): collect real experience, train world model only. Actor/critic frozen.
Phase 2: actor trained via straight-through gradients through world model.
         Critic trained on imagined lambda-returns.
         World model keeps training on real transitions.

Real experience is only ever used for world model training.
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
    max_grad_norm: float = 0.5

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
    imagine_horizon: int = 15
    imagine_warmup_steps: int = 200000
    imagine_actor_lr: float = 3e-5
    imagine_critic_lr: float = 3e-4
    imagine_entropy_coef: float = 3e-4
    imagine_lambda: float = 0.95
    imagine_update_epochs: int = 10
    imagine_batch_size: int = 512

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

        # Encoder (trained with world model)
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

        # Actor (trained only through imagination)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(ld, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Critic (trained only through imagination)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(ld, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # World model (trained on real transitions)
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
        """Differentiable transition: returns next latent mean."""
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        return latent + self.transition_mean(hidden)

    def transition_params(self, latent, env_action):
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        mean = latent + self.transition_mean(hidden)
        std_scale = torch.sigmoid(self.transition_logstd(hidden))
        std = self.model_min_std + (self.model_max_std - self.model_min_std) * std_scale
        return mean, std

    def reward_forward(self, latent, env_action, next_latent):
        model_input = torch.cat([latent, env_action, next_latent], dim=-1)
        return self.reward_model(model_input).squeeze(-1)

    def imagine_rollout(self, start_latent, horizon, gamma, lam):
        """Differentiable rollout. Returns per-step rewards, values, and entropy.

        Actor gets gradients via rsample → world model forward → reward.
        Critic gets gradients via value predictions vs lambda-return targets.
        World model is effectively frozen (separate optimizer).
        """
        current_latent = start_latent  # detached from encoder
        rewards = []
        values = []
        entropies = []

        for h in range(horizon):
            # Critic value at this state
            v = self.critic(current_latent).squeeze(-1)
            values.append(v)

            # Actor samples action
            action_mean = self.actor_mean(current_latent)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            dist = Normal(action_mean, torch.exp(action_logstd))
            action = dist.rsample()
            entropies.append(dist.entropy().sum(-1))

            env_action = self.clamp_action(action)

            # Differentiable world model step
            next_latent = self.transition_forward(current_latent, env_action)
            reward = self.reward_forward(current_latent, env_action, next_latent)
            rewards.append(reward)

            current_latent = next_latent

        # Bootstrap value at horizon
        bootstrap = self.critic(current_latent).squeeze(-1)

        rewards = torch.stack(rewards, dim=0)      # (H, batch)
        values = torch.stack(values, dim=0)         # (H, batch)
        entropies = torch.stack(entropies, dim=0)   # (H, batch)

        # Lambda-returns WITH gradient flow through rewards (for actor)
        # Values are detached so gradients only flow through reward path
        lambda_returns = torch.zeros_like(rewards)
        last = bootstrap.detach()
        for t in reversed(range(horizon)):
            next_v = values[t + 1].detach() if t < horizon - 1 else bootstrap.detach()
            last = rewards[t] + gamma * ((1 - lam) * next_v + lam * last)
            lambda_returns[t] = last

        # Detached lambda-returns for critic targets (no actor gradients)
        lambda_returns_detached = lambda_returns.detach()

        return rewards, values, entropies, lambda_returns, lambda_returns_detached, bootstrap

    def get_action(self, obs):
        """For rollout collection only."""
        with torch.no_grad():
            latent = self.encode(obs)
            action_mean = self.actor_mean(latent)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            dist = Normal(action_mean, torch.exp(action_logstd))
            return dist.sample()

    def actor_parameters(self):
        params = []
        params.extend(self.actor_mean.parameters())
        params.append(self.actor_logstd)
        return params

    def critic_parameters(self):
        return list(self.critic.parameters())

    def world_model_parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.transition_backbone.parameters())
        params.extend(self.transition_mean.parameters())
        params.extend(self.transition_logstd.parameters())
        params.extend(self.reward_model.parameters())
        return params


def compute_world_model_loss(agent, obs, actions, next_obs, rewards):
    latent = agent.encode(obs)
    env_action = agent.clamp_action(actions)
    target_next_latent = agent.encode_target(next_obs)

    pred_mean, pred_std = agent.transition_params(latent, env_action)
    trans_dist = Normal(pred_mean, pred_std)
    transition_loss = -trans_dist.log_prob(target_next_latent.detach()).mean()

    pred_reward = agent.reward_forward(latent.detach(), env_action, target_next_latent)
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
    world_model_optimizer = optim.Adam(agent.world_model_parameters(), lr=args.world_model_lr, eps=1e-5)
    actor_optimizer = optim.Adam(agent.actor_parameters(), lr=args.imagine_actor_lr, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic_parameters(), lr=args.imagine_critic_lr, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        use_imagination = (global_step >= args.imagine_warmup_steps)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            world_model_optimizer.param_groups[0]["lr"] = frac * args.world_model_lr
            actor_optimizer.param_groups[0]["lr"] = frac * args.imagine_actor_lr
            critic_optimizer.param_groups[0]["lr"] = frac * args.imagine_critic_lr

        # Rollout collection (just gathering data for world model)
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            action = agent.get_action(next_obs)
            actions[step] = action

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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
        b_rewards = rewards.reshape(-1)

        # World model update (always)
        wm_inds = np.arange(args.batch_size)
        last_wm_loss = last_trans_loss = last_rew_loss = torch.zeros((), device=device)
        for _ in range(args.world_model_update_epochs):
            np.random.shuffle(wm_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = wm_inds[start:end]
                wm_loss, trans_loss, rew_loss = compute_world_model_loss(
                    agent, b_obs[mb_inds], b_actions[mb_inds],
                    b_next_obs[mb_inds], b_rewards[mb_inds])
                world_model_optimizer.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(agent.world_model_parameters(), args.world_model_max_grad_norm)
                world_model_optimizer.step()
                last_wm_loss = wm_loss.detach()
                last_trans_loss = trans_loss.detach()
                last_rew_loss = rew_loss.detach()
        agent.update_target_encoder(args.target_encoder_tau)

        # Imagination AC update (after warmup)
        last_imagine_return = torch.zeros((), device=device)
        last_imagine_entropy = torch.zeros((), device=device)
        last_critic_loss = torch.zeros((), device=device)
        if use_imagination:
            for _ in range(args.imagine_update_epochs):
                sel = np.random.permutation(args.batch_size)[:args.imagine_batch_size]
                with torch.no_grad():
                    start_latents = agent.encode(b_obs[sel]).detach()

                rewards_t, values_t, entropies_t, lambda_returns, lambda_returns_det, bootstrap = agent.imagine_rollout(
                    start_latents, args.imagine_horizon, args.gamma, args.imagine_lambda)

                # Critic loss: predict detached lambda-returns (no actor gradients)
                critic_loss = 0.5 * ((values_t - lambda_returns_det) ** 2).mean()

                critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(agent.critic_parameters(), args.max_grad_norm)
                critic_optimizer.step()

                # Actor loss: maximize lambda-returns directly (straight-through)
                # Gradients flow: actor → action → world_model.forward → rewards → lambda_returns
                # Values are detached inside lambda_returns, so only reward path carries gradients
                actor_loss = -lambda_returns.mean() - args.imagine_entropy_coef * entropies_t.mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor_parameters(), args.max_grad_norm)
                actor_optimizer.step()

                last_imagine_return = lambda_returns_det[0].mean()
                last_imagine_entropy = entropies_t.mean().detach()
                last_critic_loss = critic_loss.detach()

        writer.add_scalar("world_model/total_loss", last_wm_loss.item(), global_step)
        writer.add_scalar("world_model/transition_loss", last_trans_loss.item(), global_step)
        writer.add_scalar("world_model/reward_loss", last_rew_loss.item(), global_step)
        writer.add_scalar("imagination/active", float(use_imagination), global_step)
        writer.add_scalar("imagination/mean_return", last_imagine_return.item(), global_step)
        writer.add_scalar("imagination/mean_entropy", last_imagine_entropy.item(), global_step)
        writer.add_scalar("imagination/critic_loss", last_critic_loss.item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        phase = "DREAM" if use_imagination else "WM"
        print(f"SPS: {sps} [{phase}] img_ret={last_imagine_return.item():.1f} wm={last_wm_loss.item():.3f} crit={last_critic_loss.item():.3f}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
