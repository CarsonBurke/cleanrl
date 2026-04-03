"""Imagination v8.1 — Simplified: PPO + World Model Auxiliary Loss

Strips v8 down to essentials:
- Encoder (obs → latent) with target encoder EMA
- World model (transition + reward prediction) as auxiliary representation loss
- Standard diagonal Gaussian actor + critic operating on latents
- No imagination actor, no BC loss, no phase scheduling, no context RNN

The hypothesis: the world model auxiliary loss is what makes v8 good, not the
imagination rollouts. This ablation tests that by removing all imagination machinery.
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
            nn.Tanh(),
            layer_init(nn.Linear(128, ld)),
            nn.Tanh(),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Standard actor on latents (diagonal Gaussian, same as v5/v8)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(ld, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Standard critic on latents
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

    def transition_params(self, latent, env_action):
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        mean = latent + self.transition_mean(hidden)
        std_scale = torch.sigmoid(self.transition_logstd(hidden))
        std = self.model_min_std + (self.model_max_std - self.model_min_std) * std_scale
        return mean, std

    def predict_reward(self, latent, env_action, next_latent):
        model_input = torch.cat([latent, env_action, next_latent], dim=-1)
        return self.reward_model(model_input).squeeze(-1)

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

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            behavior_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            world_model_optimizer.param_groups[0]["lr"] = frac * args.world_model_lr

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

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
