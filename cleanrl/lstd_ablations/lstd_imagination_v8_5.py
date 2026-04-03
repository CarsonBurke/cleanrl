"""Imagination v8.5 — v8.3 + Imagination-Augmented Critic

v8.3 trains a world model as auxiliary loss but never uses it for decision-making.
v8.5 makes the critic *use* the world model: instead of just V(s)=MLP(s), it also
imagines forward H steps using the world model, accumulates lambda-returns over the
imagined trajectory, and blends with the MLP critic via a learned gate.

Key design choices:
- Imagination rollout is fully no_grad for world model and actor — only encoder,
  MLP critic, and gate receive value-loss gradients
- Imagination only runs during rollout collection (get_value for GAE bootstrap).
  PPO minibatch updates use MLP critic only — imagination is baked into return targets
- Warmup: imagination disabled for first 200K steps while world model trains
- Gate initialized at ~0.1 (sigmoid(-2.2)) so MLP critic dominates early
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

    # Imagination
    imagine_horizon: int = 5
    imagine_lambda: float = 0.95
    imagine_warmup_steps: int = 200000

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

        # Encoder: deeper RMSNorm + SiLU (no final activation for unbounded latent space)
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

        # Imagination gate: sigmoid(-2.2) ≈ 0.1
        self.imagination_gate = nn.Parameter(torch.tensor(-2.2))

        self.model_min_std = args.model_min_std
        self.model_max_std = args.model_max_std
        self.imagine_horizon = args.imagine_horizon
        self.imagine_lambda = args.imagine_lambda
        self.gamma = args.gamma
        self.use_imagination = False
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

    def imagine_value(self, latent):
        """Roll out world model H steps from latent, compute lambda-returns.

        World model and actor are used under no_grad — only the MLP critic
        (for bootstrap) and imagination_gate receive gradients through get_value.
        """
        batch_size = latent.shape[0]
        H = self.imagine_horizon
        lam = self.imagine_lambda
        gamma = self.gamma

        with torch.no_grad():
            # Collect imagined rewards and values
            imagined_rewards = []
            current_latent = latent

            for h in range(H):
                # Deterministic action from policy mean
                action_mean = self.actor_mean(current_latent)
                env_action = self.clamp_action(action_mean)

                # Deterministic transition (use mean, not sample)
                next_mean, _ = self.transition_params(current_latent, env_action)

                # Predicted reward
                reward = self.predict_reward(current_latent, env_action, next_mean)
                imagined_rewards.append(reward)

                current_latent = next_mean

            # Bootstrap value at horizon (this needs grad for critic)
            # Stack rewards: (H, batch)
            imagined_rewards = torch.stack(imagined_rewards, dim=0)  # (H, batch)

        # Bootstrap from MLP critic at final imagined latent — WITH gradient
        bootstrap_value = self.critic(current_latent).squeeze(-1)  # (batch,)

        # Compute lambda-returns backwards (no grad needed for rewards, only bootstrap)
        # V_H = bootstrap_value
        # V_t = r_t + gamma * ((1-lam)*V_{t+1}_critic + lam*V_{t+1})
        # Simplified: standard lambda-return
        lambda_return = bootstrap_value
        for h in reversed(range(H)):
            lambda_return = imagined_rewards[h] + gamma * lambda_return
            # For proper lambda-returns, we'd need per-step critic values.
            # Use TD(lambda) style: blend one-step and multi-step
            if h > 0:
                # No per-step critic evaluation to keep it simple and fast.
                # This is effectively a H-step return with discount, bootstrapped.
                pass

        # Actually, let's do proper lambda-returns with per-step critic values
        # Re-do: we need critic values at each step
        with torch.no_grad():
            # Recompute the trajectory to get per-step latents
            step_latents = [latent]
            step_rewards = []
            current_latent = latent
            for h in range(H):
                action_mean = self.actor_mean(current_latent)
                env_action = self.clamp_action(action_mean)
                next_mean, _ = self.transition_params(current_latent, env_action)
                reward = self.predict_reward(current_latent, env_action, next_mean)
                step_rewards.append(reward)
                current_latent = next_mean
                step_latents.append(current_latent)

        # Get critic values at each step (WITH gradient for critic)
        # step_latents has H+1 entries: s_0, s_1, ..., s_H
        all_latents = torch.stack(step_latents, dim=0)  # (H+1, batch, ld)
        all_latents_flat = all_latents.reshape(-1, all_latents.shape[-1])
        # Detach latents from world model, but critic gets gradients
        all_values_flat = self.critic(all_latents_flat.detach()).squeeze(-1)
        all_values = all_values_flat.reshape(H + 1, batch_size)  # (H+1, batch)

        # Lambda-return computation (backward from horizon)
        step_rewards_t = torch.stack(step_rewards, dim=0)  # (H, batch)
        lambda_return = all_values[H]  # bootstrap at horizon
        for h in reversed(range(H)):
            # TD target at step h: r_h + gamma * V_{h+1}
            td_target = step_rewards_t[h] + gamma * all_values[h + 1]
            # Lambda-return: blend TD target and full return
            lambda_return = (1 - lam) * td_target + lam * (step_rewards_t[h] + gamma * lambda_return)

        return lambda_return  # (batch,)

    def get_value(self, obs):
        latent = self.encode(obs)
        mlp_value = self.critic(latent).squeeze(-1)  # (batch,)

        if not self.use_imagination:
            return mlp_value.unsqueeze(-1)

        imagination_value = self.imagine_value(latent)
        alpha = torch.sigmoid(self.imagination_gate)
        blended = (1 - alpha) * mlp_value + alpha * imagination_value
        return blended.unsqueeze(-1)

    def get_action_and_value(self, obs, action=None):
        latent = self.encode(obs)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        # PPO updates use MLP critic only (no imagination)
        value = self.critic(latent)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def behavior_parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.actor_mean.parameters())
        params.append(self.actor_logstd)
        params.extend(self.critic.parameters())
        params.append(self.imagination_gate)
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

        # Enable/disable imagination based on warmup
        agent.use_imagination = (global_step >= args.imagine_warmup_steps)

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

        # GAE — use imagination-augmented value for bootstrap
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

        # PPO update (MLP critic only, no imagination)
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

        gate_alpha = torch.sigmoid(agent.imagination_gate).item()
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
        writer.add_scalar("imagination/gate_alpha", gate_alpha, global_step)
        writer.add_scalar("imagination/active", float(agent.use_imagination), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)), f"gate_alpha={gate_alpha:.3f}", f"imagine={'ON' if agent.use_imagination else 'OFF'}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
