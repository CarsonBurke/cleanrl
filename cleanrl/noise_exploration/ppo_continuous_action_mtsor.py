# Multi-Timescale Stochastic Orthogonal Rotation (MTSOR) Exploration for PPO
#
# Goal: go beyond gSDE-style time correlation while keeping PPO's per-step
# diagonal-Gaussian log_prob exact and cheap.
#
# Key ideas:
# 1) Multi-timescale AR(1) noise bank: maintain K independent AR(1) latents with
#    different time constants (rho_k = exp(-1/tau_k)). Each latent has stationary
#    N(0, I) marginals.
# 2) State-dependent spectral mixing: compute weights w_k(s) and mix the K
#    latents into a single exploration noise vector; normalize by ||w||_2 to
#    preserve unit variance exactly (so noise_t ~ N(0, I) marginally).
# 3) Orthogonal rotation: apply a random orthogonal matrix R (resampled each
#    rollout) so the resulting exploration exhibits rich cross-actuator *lag*
#    correlations while keeping same-time covariance equal to identity.
#
# The resulting sampled action is:
#   a_t = mean(s_t) + std(s_t) * (R @ z_t),  z_t ~ N(0, I)
# so the policy distribution at each step is still diagonal Gaussian and we can
# use the exact Normal(mean, std) log_prob for PPO.
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # MTSOR exploration arguments
    num_scales: int = 6
    """number of AR(1) time scales in the noise bank"""
    tau_min: float = 1.0
    """minimum AR time constant in steps (smaller -> whiter component)"""
    tau_max: float = 128.0
    """maximum AR time constant in steps (larger -> smoother component)"""
    mix_temperature: float = 1.0
    """softmax temperature for scale weights (smaller -> more peaky)"""
    resample_rotation_every_rollout: bool = True
    """resample orthogonal rotation at each rollout start"""
    reset_noise_on_done: bool = True
    """reset AR states for envs that ended an episode"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
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


def _sample_orthogonal(action_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Sample a random orthogonal matrix via QR; stabilize sign ambiguity.
    a = torch.randn((action_dim, action_dim), device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    d[d == 0] = 1.0
    q = q * d.unsqueeze(0)
    return q


class Agent(nn.Module):
    def __init__(self, envs, num_scales: int, tau_min: float, tau_max: float, mix_temperature: float):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.num_scales = int(num_scales)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.mix_temperature = float(mix_temperature)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.actor_mix = layer_init(nn.Linear(64, self.num_scales), std=0.01)

        self.noise_states = None
        self.rotation = None
        self._rhos = None
        self._noise_scales = None

    def reset_noise(self, batch_size: int, resample_rotation: bool = True):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if self.num_scales < 1:
            raise ValueError("num_scales must be >= 1")
        if not (self.tau_min > 0 and self.tau_max >= self.tau_min):
            raise ValueError("tau_min must be > 0 and tau_max >= tau_min")

        # log-spaced time constants -> rho in (0, 1)
        if self.num_scales == 1:
            taus = torch.tensor([self.tau_max], device=device, dtype=dtype)
        else:
            taus = torch.exp(torch.linspace(math.log(self.tau_min), math.log(self.tau_max), self.num_scales, device=device, dtype=dtype))
        rhos = torch.exp(-1.0 / taus)
        rhos = torch.clamp(rhos, min=0.0, max=0.9995)
        self._rhos = rhos.view(self.num_scales, 1, 1)
        self._noise_scales = torch.sqrt(torch.clamp(1.0 - self._rhos**2, min=1e-6))

        self.noise_states = torch.randn((self.num_scales, batch_size, self.action_dim), device=device, dtype=dtype)
        if resample_rotation:
            self.rotation = _sample_orthogonal(self.action_dim, device=device, dtype=dtype)

    def reset_done_envs(self, done_mask: torch.Tensor):
        if self.noise_states is None:
            return
        if done_mask is None:
            return
        if done_mask.dtype != torch.bool:
            done_mask = done_mask.bool()
        if done_mask.numel() == 0 or not torch.any(done_mask):
            return
        self.noise_states[:, done_mask, :].normal_()

    def get_value(self, x):
        return self.critic(x)

    def _mix_weights(self, latent: torch.Tensor) -> torch.Tensor:
        logits = self.actor_mix(latent)
        temp = max(self.mix_temperature, 1e-6)
        return torch.softmax(logits / temp, dim=-1)

    def _sample_correlated_noise(self, weights: torch.Tensor, update_noise: bool = True) -> torch.Tensor:
        # weights: (B, K), noise_states: (K, B, A)
        if self.noise_states is None or self.noise_states.shape[1] != weights.shape[0]:
            self.reset_noise(batch_size=weights.shape[0], resample_rotation=False)

        if update_noise:
            eps = torch.randn_like(self.noise_states)
            self.noise_states = self._rhos * self.noise_states + self._noise_scales * eps

        mixed = torch.einsum("bk,kba->ba", weights, self.noise_states)
        denom = torch.sqrt(torch.sum(weights**2, dim=-1, keepdim=True) + 1e-8)
        mixed = mixed / denom
        if self.rotation is not None:
            mixed = mixed @ self.rotation.T
        return mixed

    def get_action_and_value(self, x, action=None, update_noise: bool = True):
        latent = self.actor_latent(x)
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            weights = self._mix_weights(latent)
            noise = self._sample_correlated_noise(weights, update_noise=update_noise)
            action = action_mean + noise * action_std

        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        num_scales=args.num_scales,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        mix_temperature=args.mix_temperature,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.reset_noise(batch_size=args.num_envs, resample_rotation=args.resample_rotation_every_rollout)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, update_noise=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if args.reset_noise_on_done:
                agent.reset_done_envs(next_done.bool())

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], update_noise=False)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
