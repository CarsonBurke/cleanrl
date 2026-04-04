# SPACE v23: LSMN with K/τ Annealing
#
# Clean v15 LSMN with scheduled reduction of noise persistence.
# K decreases linearly over training; τ increases linearly.
#
# Motivation: v15 4M results show very high peaks but late-stage collapse
#   in Walker2d (peak 4556, final 427) and Hopper (peak 3188, final 1058).
#   The persistent noise that enables early exploration causes late-stage
#   instability. Annealing K/τ gives exploration early and stability late.
#
# Schedule:
#   K: resample_interval_max → resample_interval_min (linear in progress)
#   τ: resample_blend → resample_blend_max (linear in progress)
#
# v15 4M baselines:
#   HC K=64 τ=0.5: peak 4783, final 4768 (stable!)
#   Walker2d K=16 τ=0.3: peak 4556, final 427 (volatile)
#   Hopper K=16 τ=0.3: peak 3188q, final 1058 (volatile)

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

    # LSMN noise arguments
    noise_log_std_init: float = 0.0
    """initial log std for noise scale (per latent-action pair, ~0.63 effective action std)"""
    resample_interval: int = 16
    """resampling interval for persistent latent noise (steps)"""
    resample_blend: float = 0.3
    """blend factor τ for soft resampling: 1.0=hard, 0.3=soft (keeps 84% old direction)"""
    resample_on_reset: bool = True
    """resample noise for envs that had an episode termination"""
    resample_interval_min: int = -1
    """final resample interval (for K annealing). -1 = no annealing (use resample_interval)"""
    resample_blend_max: float = -1.0
    """final blend factor (for τ annealing). -1 = no annealing (use resample_blend)"""

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


class Agent(nn.Module):
    def __init__(self, envs, noise_log_std_init=0.0):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

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

        # LSMN: separate noise projection matrix [action_dim, latent_dim]
        self.noise_proj = nn.Linear(LATENT_DIM, action_dim, bias=False)
        torch.nn.init.orthogonal_(self.noise_proj.weight, gain=1.0)

        # Per-(latent, action) noise scale [latent_dim, action_dim]
        self.noise_log_std = nn.Parameter(
            torch.ones(LATENT_DIM, action_dim) * noise_log_std_init
        )

        # Persistent latent noise: [num_envs, latent_dim]
        self.noise_eps = None

    def _get_noise_std(self):
        return torch.exp(self.noise_log_std)

    def reset_noise(self, num_envs):
        """Sample fresh latent noise vectors for all envs."""
        device = self.noise_log_std.device
        self.noise_eps = torch.randn(num_envs, self.latent_dim, device=device)

    def reset_noise_for_envs(self, env_mask):
        """Resample noise for terminated envs."""
        if not env_mask.any():
            return
        n_reset = env_mask.sum().item()
        self.noise_eps[env_mask] = torch.randn(n_reset, self.latent_dim, device=self.noise_eps.device)

    def resample_for_envs(self, env_mask, blend=1.0):
        """Soft-resample noise. blend=1.0 is hard, <1 blends with old noise."""
        if not env_mask.any():
            return
        n = env_mask.sum().item()
        fresh = torch.randn(n, self.latent_dim, device=self.noise_eps.device)
        if blend >= 1.0:
            self.noise_eps[env_mask] = fresh
        else:
            keep = math.sqrt(1.0 - blend)
            inject = math.sqrt(blend)
            self.noise_eps[env_mask] = keep * self.noise_eps[env_mask] + inject * fresh

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        latent = self.actor_backbone(x)  # [batch, latent_dim]
        mean = self.mean_net(latent)      # [batch, action_dim]

        # LSMN noise variance (diagonal approximation)
        noise_std = self._get_noise_std()       # [latent_dim, action_dim]
        W_noise = self.noise_proj.weight         # [action_dim, latent_dim]

        # Marginal variance: var_j = Σ_i W_ji² * h_i² * σ_ij²
        action_var = (latent ** 2) @ (noise_std ** 2 * W_noise.T ** 2)
        action_std = torch.sqrt(action_var + 1e-6)

        dist = Normal(mean, action_std)

        if action is None:
            h_eps = latent * self.noise_eps                   # [batch, latent_dim]
            combined = h_eps.unsqueeze(-1) * noise_std        # [batch, latent, action]
            noise = torch.einsum('ai,bia->ba', W_noise, combined)
            action = mean + noise

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        value = self.critic(x)

        return action, log_prob, entropy, value


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

    agent = Agent(envs, noise_log_std_init=args.noise_log_std_init).to(device)
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
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Compute annealed K and τ for this iteration
        progress = (iteration - 1) / max(args.num_iterations - 1, 1)
        if args.resample_interval_min >= 0:
            current_K = int(args.resample_interval + (args.resample_interval_min - args.resample_interval) * progress)
            current_K = max(current_K, max(args.resample_interval_min, 1))
        else:
            current_K = args.resample_interval
        if args.resample_blend_max >= 0:
            current_blend = args.resample_blend + (args.resample_blend_max - args.resample_blend) * progress
        else:
            current_blend = args.resample_blend

        # Fresh noise at start of each rollout
        agent.reset_noise(args.num_envs)
        steps_since_resample = torch.zeros(args.num_envs, dtype=torch.long, device=device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Periodic soft resampling (with annealed K and τ)
            steps_since_resample += 1
            resample_mask = steps_since_resample >= current_K
            if resample_mask.any():
                agent.resample_for_envs(resample_mask, blend=current_blend)
                steps_since_resample[resample_mask] = 0

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset noise for terminated envs
            if args.resample_on_reset and next_done.any():
                reset_mask = next_done.bool()
                agent.reset_noise_for_envs(reset_mask)
                steps_since_resample[reset_mask] = 0

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

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
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
        with torch.no_grad():
            writer.add_scalar("space/noise_log_std_mean", agent.noise_log_std.mean().item(), global_step)
            writer.add_scalar("space/noise_std_mean", agent._get_noise_std().mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
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
