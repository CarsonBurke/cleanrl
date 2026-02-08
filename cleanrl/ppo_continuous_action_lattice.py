# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# Lattice exploration based on: Latent Exploration for Reinforcement Learning https://arxiv.org/abs/2305.20065
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import MultivariateNormal, Normal
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

    # Lattice specific arguments
    lattice_alpha: float = 1.0
    """relative weight between action and latent noise (0 removes latent noise)"""
    lattice_std_clip_min: float = 1e-3
    """minimum clipping value for lattice noise standard deviation"""
    lattice_std_clip_max: float = 10.0
    """maximum clipping value for lattice noise standard deviation"""
    lattice_std_reg: float = 0.0
    """regularization to prevent collapsing to a deterministic policy"""
    lattice_full_std: bool = False
    """use full (latent_dim x (latent_dim + action_dim)) std params; False uses only (latent_dim x 2)"""
    lattice_use_expln: bool = True
    """use expln() instead of exp() for std (keeps variance above zero, prevents fast growth)"""
    lattice_learn_features: bool = True
    """allow gradients to flow through latent features into the variance/noise computation"""
    sde_sample_freq: int = 1
    """frequency of resampling lattice noise exploration matrices (1 = every step)"""

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


class LatticeDistribution:
    """
    Lattice exploration distribution for continuous action spaces.

    Creates correlated noise across actuators via a covariance matrix induced by
    the policy network's last layer weights, following:
    Latent Exploration for Reinforcement Learning (https://arxiv.org/abs/2305.20065)

    The covariance decomposes into:
      - Correlated part: alpha^2 * W @ diag(latent_corr_var) @ W^T
        (noise injected into latent space, then projected through W)
      - Independent part: diag(latent_ind_var + std_reg^2)
        (direct per-action noise added to the diagonal)

    :param action_dim: Dimension of the action space.
    :param latent_dim: Dimension of the latent (last hidden) layer.
    :param alpha: Weight of correlated latent noise (0 removes lattice effect).
    :param full_std: If True, use (latent_dim x (latent_dim + action_dim)) std params.
        If False, use (latent_dim x 2) and broadcast (fewer parameters).
    :param use_expln: Use expln() instead of exp() for positive std (cf gSDE paper).
    :param std_clip: (min, max) clipping range for standard deviations.
    :param std_reg: Regularization to prevent deterministic collapse.
    :param learn_features: If True, gradients flow through latent features into
        variance/noise computation. If False, latent features are detached.
    :param epsilon: Small value to avoid NaN in expln computation.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        alpha: float = 1.0,
        full_std: bool = True,
        use_expln: bool = False,
        std_clip: tuple = (1e-3, 1.0),
        std_reg: float = 0.0,
        learn_features: bool = True,
        epsilon: float = 1e-6,
    ):
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.full_std = full_std
        self.use_expln = use_expln
        self.min_std, self.max_std = std_clip
        self.std_reg = std_reg
        self.learn_features = learn_features
        self.epsilon = epsilon

        # Exploration noise matrices (sampled periodically)
        self.corr_exploration_mat = None
        self.ind_exploration_mat = None
        self.corr_exploration_matrices = None
        self.ind_exploration_matrices = None

    def get_std(self, log_std: torch.Tensor):
        """
        Compute correlated and independent standard deviations from log_std parameter.

        With full_std=True, log_std has shape (latent_dim, latent_dim + action_dim):
          - First latent_dim columns: correlated noise std
          - Last action_dim columns: independent noise std

        With full_std=False, log_std has shape (latent_dim, 2):
          - Column 0: broadcast to correlated noise std
          - Column 1: broadcast to independent noise std

        A correction of -0.5 * log(latent_dim) is applied to normalize
        the variance contribution per latent dimension.
        """
        log_std = log_std.clip(min=np.log(self.min_std), max=np.log(self.max_std))
        log_std = log_std - 0.5 * np.log(self.latent_dim)

        if self.use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)

        if self.full_std:
            corr_std = std[:, :self.latent_dim]
            ind_std = std[:, -self.action_dim:]
        else:
            corr_std = torch.ones(self.latent_dim, self.latent_dim, device=log_std.device) * std[:, 0:1]
            ind_std = torch.ones(self.latent_dim, self.action_dim, device=log_std.device) * std[:, 1:]
        return corr_std, ind_std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1):
        """
        Sample exploration matrices from centered Gaussians parameterized by the
        learned standard deviations. Uses the reparameterization trick for gradients.
        """
        corr_std, ind_std = self.get_std(log_std)
        corr_dist = Normal(torch.zeros_like(corr_std), corr_std)
        ind_dist = Normal(torch.zeros_like(ind_std), ind_std)

        self.corr_exploration_mat = corr_dist.rsample()
        self.ind_exploration_mat = ind_dist.rsample()
        self.corr_exploration_matrices = corr_dist.rsample((batch_size,))
        self.ind_exploration_matrices = ind_dist.rsample((batch_size,))

    def _get_noise(self, latent_sde, exploration_mat, exploration_matrices):
        """Compute noise via matrix multiplication of latent features and exploration matrix."""
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(exploration_matrices):
            return torch.mm(latent_sde, exploration_mat)
        latent_sde = latent_sde.unsqueeze(dim=1)
        noise = torch.bmm(latent_sde, exploration_matrices)
        return noise.squeeze(dim=1)

    def sample(self, mean_actions, action_mean_weight, latent_cov):
        """
        Sample actions using lattice noise.

        Applies correlated latent perturbation from a dedicated covariance latent,
        projects it through the covariance projection layer, and adds independent
        action-space noise:
          actions = mean_actions + W_cov @ (alpha * corr_noise) + ind_noise
        """
        latent_cov = latent_cov if self.learn_features else latent_cov.detach()
        latent_noise = self.alpha * self._get_noise(
            latent_cov, self.corr_exploration_mat, self.corr_exploration_matrices
        )
        action_noise = self._get_noise(
            latent_cov, self.ind_exploration_mat, self.ind_exploration_matrices
        )
        actions = mean_actions + F.linear(latent_noise, action_mean_weight, bias=None) + action_noise
        return actions

    def get_distribution(self, mean_actions, log_std, latent_cov, action_mean_weight):
        """
        Build a MultivariateNormal distribution with the lattice-induced covariance.

        The covariance has two components:
          sigma = alpha^2 * W_mean diag(corr_var) W_mean^T + diag(ind_var + std_reg^2)
        where W_mean are the policy mean projection weights, and corr_var / ind_var
        are computed from the learned log_std and the latent features.
        """
        latent_cov = latent_cov if self.learn_features else latent_cov.detach()
        corr_std, ind_std = self.get_std(log_std)
        latent_corr_variance = torch.mm(latent_cov**2, corr_std**2)
        latent_ind_variance = torch.mm(latent_cov**2, ind_std**2) + self.std_reg**2

        W = action_mean_weight
        sigma_mat = self.alpha**2 * (W * latent_corr_variance[:, None, :]).matmul(W.T)
        sigma_mat[:, range(self.action_dim), range(self.action_dim)] += latent_ind_variance

        return MultivariateNormal(loc=mean_actions, covariance_matrix=sigma_mat, validate_args=False)


class Agent(nn.Module):
    def __init__(self, envs, alpha=1.0, full_std=True, use_expln=False,
                 std_clip=(1e-3, 1.0), std_reg=0.0, learn_features=True):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = 64

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean_trunk = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.RMSNorm(64),
        )
        self.actor_cov_trunk = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.RMSNorm(64),
        )
        self.actor_mean = nn.Linear(latent_dim, action_dim)

        # Lattice log_std parameter shape depends on full_std mode
        # full_std=True:  (latent_dim, latent_dim + action_dim) — full parameterization
        # full_std=False: (latent_dim, 2) — reduced, broadcast per-dim
        log_std_shape = (latent_dim, latent_dim + action_dim) if full_std else (latent_dim, 2)
        self.actor_logstd = nn.Parameter(torch.zeros(log_std_shape))

        self.lattice = LatticeDistribution(
            action_dim=action_dim,
            latent_dim=latent_dim,
            alpha=alpha,
            full_std=full_std,
            use_expln=use_expln,
            std_clip=std_clip,
            std_reg=std_reg,
            learn_features=learn_features,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        latent_mean = self.actor_mean_trunk(x)
        latent_cov_raw = self.actor_cov_trunk(x)
        cov_rms = torch.sqrt(torch.mean(latent_cov_raw.pow(2), dim=-1, keepdim=True) + 1e-8)
        latent_cov = latent_cov_raw / cov_rms

        action_mean = self.actor_mean(latent_mean)

        dist = self.lattice.get_distribution(
            action_mean, self.actor_logstd, latent_cov, self.actor_mean.weight
        )

        if action is None:
            action = self.lattice.sample(action_mean, self.actor_mean.weight, latent_cov)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, self.critic(x)

    def sample_noise(self, batch_size=1):
        """Resample the lattice exploration matrices."""
        self.lattice.sample_weights(self.actor_logstd, batch_size=batch_size)


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
        alpha=args.lattice_alpha,
        full_std=args.lattice_full_std,
        use_expln=args.lattice_use_expln,
        std_clip=(args.lattice_std_clip_min, args.lattice_std_clip_max),
        std_reg=args.lattice_std_reg,
        learn_features=args.lattice_learn_features,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initial noise sampling
    agent.sample_noise(batch_size=args.num_envs)

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.sde_sample_freq > 0 and step % args.sde_sample_freq == 0:
                    agent.sample_noise(batch_size=args.num_envs)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

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

                # Resample lattice noise for each training minibatch
                agent.sample_noise(batch_size=len(mb_inds))

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
