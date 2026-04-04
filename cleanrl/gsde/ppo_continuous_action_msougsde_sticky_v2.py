# MSOU-gSDE Sticky v2: Refactor + FixedMix + Optional Renewal + Correlation Schedule
#
# Refactor goals:
# - Keep OU exploration state continuous across rollouts (sticky), resetting only
#   per-env on episode end (reduces destabilizing jumps).
# - Remove the dead learnable mixing head: when PPO uses gSDE's marginal
#   log_prob, mixture weights do not get policy gradients. Use a fixed power-law
#   multi-scale mixture instead.
# - Add optional per-step "renewal" resets (jump-diffusion) and a correlation
#   schedule to tune smoothness during training.
#
# Note: This stays in the CleanRL gSDE lineage (cheap diagonal Normal log_prob).
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

    # gSDE + multi-scale OU arguments
    gsde_log_std_init: float = -2.0
    """initial value for the gSDE log standard deviation"""
    full_std: bool = True
    """whether to use (latent_dim x action_dim) parameters for std instead of (latent_dim, 1)"""
    use_expln: bool = False
    """use expln() instead of exp() for positive std (cf gSDE paper)"""

    num_scales: int = 8
    """number of OU time scales in the exploration matrix bank"""
    tau_min: float = 1.0
    """minimum OU time constant in steps"""
    tau_max: float = 128.0
    """maximum OU time constant in steps"""
    weight_power: float = 0.5
    """fixed mix weights w_k ~ tau_k^{-weight_power}"""

    # sticky state management
    reset_on_done: bool = True
    """reset exploration matrices for envs that ended an episode"""
    reset_every_rollout: bool = False
    """if True, resample exploration matrices at rollout start (non-sticky)"""

    # OU update cadence + renewal
    sde_update_freq: int = 1
    """frequency of OU updates (1 = every step)"""
    renewal_prob: float = 0.0
    """per-step per-env probability to renew (reset) exploration matrices"""

    # correlation schedule (scales rho magnitudes)
    rho_alpha_init: float = 1.0
    """initial multiplier for rho magnitudes (0=white, 1=full OU)"""
    rho_alpha_final: float = 1.0
    """final multiplier for rho magnitudes"""
    rho_alpha_warmup_iters: int = 0
    """linearly ramp rho_alpha over this many PPO iterations"""

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


def _powerlaw_weights(taus: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    w = taus ** (-power)
    return w / torch.sqrt(torch.sum(w**2) + eps)


class StickyMultiScaleOU:
    def __init__(self, num_scales: int, epsilon: float = 1e-6):
        self.num_scales = int(num_scales)
        self.epsilon = float(epsilon)
        self.exploration_matrices = None  # (K,B,D,A)

    @staticmethod
    def _std_from_log_std(log_std, full_std, latent_sde_dim, action_dim, use_expln, eps):
        if use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + eps
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)
        if full_std:
            return std
        return torch.ones(latent_sde_dim, action_dim, device=log_std.device) * std

    def ensure_init(self, log_std, full_std, latent_sde_dim, action_dim, use_expln, batch_size):
        if self.exploration_matrices is not None and self.exploration_matrices.shape[1] == batch_size:
            return
        std = self._std_from_log_std(log_std, full_std, latent_sde_dim, action_dim, use_expln, self.epsilon)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_matrices = weights_dist.rsample((self.num_scales, batch_size))

    def reset_envs(self, log_std, full_std, latent_sde_dim, action_dim, use_expln, env_mask: torch.Tensor):
        if self.exploration_matrices is None:
            return
        if env_mask is None:
            return
        if env_mask.dtype != torch.bool:
            env_mask = env_mask.bool()
        if env_mask.numel() == 0 or not torch.any(env_mask):
            return
        std = self._std_from_log_std(log_std, full_std, latent_sde_dim, action_dim, use_expln, self.epsilon)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_matrices[:, env_mask, :, :] = weights_dist.rsample((self.num_scales, int(env_mask.sum().item())))

    def step(self, log_std, full_std, latent_sde_dim, action_dim, use_expln, rhos_eff: torch.Tensor):
        # rhos_eff: (K,)
        assert self.exploration_matrices is not None
        std = self._std_from_log_std(log_std, full_std, latent_sde_dim, action_dim, use_expln, self.epsilon)
        weights_dist = Normal(torch.zeros_like(std), std)
        eps = weights_dist.rsample((self.num_scales, self.exploration_matrices.shape[1]))
        r = rhos_eff.to(device=log_std.device, dtype=log_std.dtype).view(self.num_scales, 1, 1, 1)
        noise_scale = torch.sqrt(torch.clamp(1.0 - r**2, min=self.epsilon))
        self.exploration_matrices = r * self.exploration_matrices + noise_scale * eps


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        latent_dim = 64

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.full_std = bool(args.full_std)
        self.use_expln = bool(args.use_expln)

        if args.num_scales < 1:
            raise ValueError("num_scales must be >= 1")
        if not (args.tau_min > 0 and args.tau_max >= args.tau_min):
            raise ValueError("tau_min must be > 0 and tau_max >= tau_min")

        if args.num_scales == 1:
            taus = torch.tensor([args.tau_max], dtype=torch.float32)
        else:
            taus = torch.exp(torch.linspace(math.log(args.tau_min), math.log(args.tau_max), int(args.num_scales)))
        rhos = torch.exp(-1.0 / taus)
        rhos = torch.clamp(rhos, min=0.0, max=0.9995)
        weights = _powerlaw_weights(taus, power=args.weight_power, eps=1e-6)
        self.register_buffer("taus", taus)
        self.register_buffer("rhos", rhos)
        self.register_buffer("mix_w", weights)  # (K,)

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
        self.actor_mean = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)

        if self.full_std:
            self.log_std = nn.Parameter(torch.ones(latent_dim, action_dim) * args.gsde_log_std_init)
        else:
            self.log_std = nn.Parameter(torch.ones(latent_dim, 1) * args.gsde_log_std_init)

        self.ou = StickyMultiScaleOU(num_scales=int(args.num_scales))
        self._last_batch_size = None

    def reset_noise(self, batch_size: int):
        self._last_batch_size = int(batch_size)
        self.ou.ensure_init(
            self.log_std,
            full_std=self.full_std,
            latent_sde_dim=self.latent_dim,
            action_dim=self.action_dim,
            use_expln=self.use_expln,
            batch_size=self._last_batch_size,
        )

    def reset_envs(self, env_mask: torch.Tensor):
        self.ou.reset_envs(
            self.log_std,
            full_std=self.full_std,
            latent_sde_dim=self.latent_dim,
            action_dim=self.action_dim,
            use_expln=self.use_expln,
            env_mask=env_mask,
        )

    def get_value(self, x):
        return self.critic(x)

    def _diag_marginal_dist(self, mean_actions, latent_sde):
        std = StickyMultiScaleOU._std_from_log_std(
            self.log_std,
            full_std=self.full_std,
            latent_sde_dim=self.latent_dim,
            action_dim=self.action_dim,
            use_expln=self.use_expln,
            eps=1e-6,
        )
        variance = torch.mm(latent_sde**2, std**2)
        return Normal(mean_actions, torch.sqrt(variance + 1e-6))

    def _sample_action(self, mean_actions, latent_sde):
        w = self.mix_w.to(device=mean_actions.device, dtype=mean_actions.dtype).view(-1, 1, 1, 1)
        e_eff = torch.sum(w * self.ou.exploration_matrices, dim=0)  # (B,D,A)
        noise = torch.bmm(latent_sde.unsqueeze(1), e_eff).squeeze(1)
        return mean_actions + noise

    def get_action_and_value(self, x, action=None, update_noise=True, rhos_eff=None):
        latent_sde = self.actor_latent(x)
        mean_actions = self.actor_mean(latent_sde)

        if update_noise:
            self.ou.ensure_init(
                self.log_std,
                full_std=self.full_std,
                latent_sde_dim=self.latent_dim,
                action_dim=self.action_dim,
                use_expln=self.use_expln,
                batch_size=mean_actions.shape[0],
            )
            self.ou.step(
                self.log_std,
                full_std=self.full_std,
                latent_sde_dim=self.latent_dim,
                action_dim=self.action_dim,
                use_expln=self.use_expln,
                rhos_eff=rhos_eff,
            )

        if action is None:
            action = self._sample_action(mean_actions, latent_sde)

        dist = self._diag_marginal_dist(mean_actions, latent_sde)
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)


def rho_alpha(args: Args, iteration: int) -> float:
    if args.rho_alpha_warmup_iters <= 0:
        return float(args.rho_alpha_final)
    t = min(1.0, max(0.0, (iteration - 1) / float(args.rho_alpha_warmup_iters)))
    return float(args.rho_alpha_init + t * (args.rho_alpha_final - args.rho_alpha_init))


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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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

    agent.reset_noise(batch_size=args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.reset_every_rollout:
            agent.reset_noise(batch_size=args.num_envs)

        alpha = rho_alpha(args, iteration)
        rhos_eff = torch.clamp(agent.rhos * alpha, min=0.0, max=0.9995)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                update_noise = (step % args.sde_update_freq) == 0
                action, logprob, _, value = agent.get_action_and_value(next_obs, update_noise=update_noise, rhos_eff=rhos_eff)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done_np).to(device)

            if update_noise and args.renewal_prob > 0.0:
                renew = torch.rand(args.num_envs, device=device) < args.renewal_prob
                agent.reset_envs(renew)
            if args.reset_on_done:
                agent.reset_envs(next_done.bool())

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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], update_noise=False, rhos_eff=rhos_eff
                )
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

