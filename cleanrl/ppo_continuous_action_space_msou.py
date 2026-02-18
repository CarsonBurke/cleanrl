# SPACE-MSOU: SPACE v13 (LSMN) + Multi-Scale OU latent noise
#
# Motivation:
# - Our gSDE-MSOU variants underperformed vs SPACE v13 on HalfCheetah at 1M.
# - SPACE v13’s exploration parameters (noise_proj, noise_log_std) affect the
#   marginal diagonal action std, so PPO gradients actually shape exploration.
# - We can add smoother, higher-entropy, time-correlated exploration by making
#   the shared latent noise ε_t time-correlated, while keeping ε_t marginally
#   N(0, I). This preserves SPACE’s per-step log_prob approximation but improves
#   exploration trajectories.
#
# Method:
# - Maintain K independent OU/AR(1) latent noise states eps_k with time constants
#   tau_k (log-spaced). Each eps_k is stationary N(0, I).
# - Mix them with fixed power-law weights w_k ~ tau_k^{-p} normalized by ||w||_2
#   to produce eps = Σ_k w_k eps_k, which is also marginally N(0, I) but with
#   multi-timescale temporal correlation.
# - Use eps in SPACE’s noise: noise_j = Σ_i W_ji * h_i * σ_ij * eps_i.
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

    # SPACE (LSMN) specific arguments
    noise_log_std_init: float = 0.0
    """initial log std for noise scale (per latent-action pair)"""

    # MSOU latent noise arguments (time correlation)
    num_scales: int = 6
    """number of OU time scales for latent noise"""
    tau_min: float = 1.0
    """minimum OU time constant in steps"""
    tau_max: float = 128.0
    """maximum OU time constant in steps"""
    weight_power: float = 0.5
    """fixed mix weights w_k ~ tau_k^{-weight_power}"""
    sde_update_freq: int = 1
    """frequency of OU updates (1 = every step)"""
    renewal_prob: float = 0.0
    """per-step per-env probability to fully renew latent noise states"""
    reset_on_done: bool = True
    """renew latent noise for envs that had an episode termination"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


LATENT_DIM = 64


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


def _logspace_taus(num_scales: int, tau_min: float, tau_max: float, device, dtype):
    if num_scales == 1:
        return torch.tensor([tau_max], device=device, dtype=dtype)
    return torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), num_scales, device=device, dtype=dtype))


def _powerlaw_weights(taus: torch.Tensor, power: float, eps: float = 1e-6):
    w = taus ** (-power)
    return w / torch.sqrt(torch.sum(w**2) + eps)


class MultiScaleOUNoise:
    def __init__(self, num_scales: int, latent_dim: int, tau_min: float, tau_max: float, weight_power: float, eps: float = 1e-6):
        self.num_scales = int(num_scales)
        self.latent_dim = int(latent_dim)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.weight_power = float(weight_power)
        self.eps = float(eps)

        self.taus = None
        self.rhos = None
        self.weights = None
        self.states = None  # (K, B, D)

    def init_params(self, device, dtype):
        self.taus = _logspace_taus(self.num_scales, self.tau_min, self.tau_max, device=device, dtype=dtype)
        rhos = torch.exp(-1.0 / self.taus)
        self.rhos = torch.clamp(rhos, min=0.0, max=0.9995)
        self.weights = _powerlaw_weights(self.taus, power=self.weight_power, eps=self.eps)

    def reset(self, batch_size: int, device, dtype):
        if self.taus is None:
            self.init_params(device=device, dtype=dtype)
        self.states = torch.randn((self.num_scales, batch_size, self.latent_dim), device=device, dtype=dtype)

    def reset_envs(self, env_mask: torch.Tensor):
        if self.states is None:
            return
        if env_mask.dtype != torch.bool:
            env_mask = env_mask.bool()
        if env_mask.numel() == 0 or not torch.any(env_mask):
            return
        self.states[:, env_mask, :].normal_()

    def step(self, update: bool = True) -> torch.Tensor:
        if self.states is None:
            raise RuntimeError("Noise not initialized")
        if update:
            eps = torch.randn_like(self.states)
            r = self.rhos.view(self.num_scales, 1, 1).to(self.states)
            s = torch.sqrt(torch.clamp(1.0 - r**2, min=self.eps))
            self.states = r * self.states + s * eps
        w = self.weights.to(self.states).view(self.num_scales, 1, 1)
        return torch.sum(w * self.states, dim=0)  # (B, D)


class Agent(nn.Module):
    def __init__(self, envs, noise_log_std_init=-2.0):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))

        self.action_dim = action_dim
        self.latent_dim = LATENT_DIM

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, LATENT_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(LATENT_DIM, LATENT_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(LATENT_DIM, 1), std=1.0),
        )
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, LATENT_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(LATENT_DIM, LATENT_DIM)),
            nn.Tanh(),
        )
        self.mean_net = layer_init(nn.Linear(LATENT_DIM, action_dim), std=0.01)

        self.noise_proj = nn.Linear(LATENT_DIM, action_dim, bias=False)
        torch.nn.init.orthogonal_(self.noise_proj.weight, gain=1.0)

        self.noise_log_std = nn.Parameter(torch.ones(LATENT_DIM, action_dim) * noise_log_std_init)

        self.noise_eps = None  # (B, latent_dim)

    def _get_noise_std(self):
        return torch.exp(self.noise_log_std)

    def get_value(self, x):
        return self.critic(x)

    def set_noise_eps(self, eps: torch.Tensor):
        self.noise_eps = eps

    def get_action_and_value(self, x, action=None):
        latent = self.actor_backbone(x)
        mean = self.mean_net(latent)

        noise_std = self._get_noise_std()
        W_noise = self.noise_proj.weight

        action_var = (latent**2) @ (noise_std**2 * W_noise.T**2)
        action_std = torch.sqrt(action_var + 1e-6)
        dist = Normal(mean, action_std)

        if action is None:
            h_eps = latent * self.noise_eps
            combined = h_eps.unsqueeze(-1) * noise_std
            noise = torch.einsum("ai,bia->ba", W_noise, combined)
            action = mean + noise

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(x)


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, noise_log_std_init=args.noise_log_std_init).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    msou = MultiScaleOUNoise(
        num_scales=args.num_scales,
        latent_dim=LATENT_DIM,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        weight_power=args.weight_power,
    )
    msou.reset(batch_size=args.num_envs, device=device, dtype=torch.float32)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            update_noise = (step % args.sde_update_freq) == 0
            eps_latent = msou.step(update=update_noise)
            agent.set_noise_eps(eps_latent)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done_np).to(device)

            if update_noise and args.renewal_prob > 0.0:
                renew = torch.rand(args.num_envs, device=device) < args.renewal_prob
                msou.reset_envs(renew)
            if args.reset_on_done and next_done.any():
                msou.reset_envs(next_done.bool())

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
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
