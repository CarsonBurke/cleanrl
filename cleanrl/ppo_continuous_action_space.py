# SPACE v2: Structured Persistent Adaptive Correlated Exploration
#
# v2 CHANGE: Multi-Timescale OU Exploration
#
# HYPOTHESIS: Locomotion requires coordinated action at multiple timescales
# simultaneously. A single OU momentum (0.9, τ≈10) can't capture both slow gait
# persistence (τ≈100, critical for HalfCheetah) and fast balance corrections
# (τ≈2, critical for Hopper/Walker2d). This was empirically confirmed:
#   - ACE v6 ri=64 (slow): HalfCheetah 1526 but Walker2d 302
#   - ACE v6 ri=1 (fast):  HalfCheetah 634  but Walker2d 705
# No single timescale works universally.
#
# SOLUTION: Three parallel OU processes at different rates:
#   - SLOW  (α=0.99, τ≈100): Gait discovery and pattern locking
#   - MEDIUM (α=0.9,  τ≈10):  Step-to-step coordination
#   - FAST  (α=0.5,  τ≈2):   Rapid balance corrections
#
# Each scale has independent exploration matrices evolving at its own rate.
# Noise is the weighted sum: noise = Σ_k w_k * noise_k, where weights satisfy
# Σ w_k² = 1 for exact variance preservation (density computation unchanged).
# Weights are learned parameters — the agent discovers the optimal temporal
# allocation per task.
#
# MTSOR tried multi-timescale noise but used diagonal Gaussian log_prob with
# orthogonal rotation — a density mismatch that killed learning (HC: 240).
# SPACE v2 avoids this by using MultivariateNormal with learned Cholesky factor.
#
# Retained from v1:
#   - Cross-actuator Cholesky factor L for coordinated motor commands
#   - MultivariateNormal for exact density computation
#   - 256 hidden units for capacity
#   - Episode-boundary hard reset
#
# v1 results (16 envs, 1-2M steps): HC 800, Hopper 676, Walker2d 1504

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.multivariate_normal import MultivariateNormal
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

    # SPACE v2 specific arguments
    gsde_log_std_init: float = -2.0
    """initial log standard deviation for gSDE exploration matrices"""
    hidden_dim: int = 256
    """hidden layer dimension for actor and critic networks"""
    resample_on_reset: bool = True
    """resample exploration matrices for envs that had an episode termination"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Fixed OU momentums for the three timescales
NOISE_MOMENTUMS = (0.5, 0.9, 0.99)
NUM_SCALES = len(NOISE_MOMENTUMS)


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
    def __init__(self, envs, gsde_log_std_init=-2.0, hidden_dim=256):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = hidden_dim

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_scales = NUM_SCALES
        self.noise_momentums = NOISE_MOMENTUMS

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_net = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)

        # gSDE log_std: [latent_dim, action_dim]
        self.gsde_log_std = nn.Parameter(
            torch.ones(latent_dim, action_dim) * gsde_log_std_init
        )

        # Cross-actuator mixing: learned lower-triangular Cholesky factor
        self.cholesky_log_diag = nn.Parameter(torch.zeros(action_dim))
        self.cholesky_offdiag = nn.Parameter(torch.zeros(action_dim * (action_dim - 1) // 2))

        # Multi-timescale mixing weights (learned)
        # Parameterized as raw log-weights, normalized so Σ w_k² = 1
        self.scale_log_weights = nn.Parameter(torch.zeros(self.num_scales))

        # Per-env exploration matrices for each timescale:
        # List of [num_envs, latent_dim, action_dim] tensors
        self.exploration_matrices_list = [None] * self.num_scales

        # Precompute innovation scales for OU updates
        self.innovation_scales = [math.sqrt(1.0 - m * m) for m in self.noise_momentums]

    def _get_gsde_std(self):
        return torch.exp(self.gsde_log_std)

    def _get_cholesky(self):
        """Build the lower-triangular Cholesky factor L."""
        L = torch.zeros(self.action_dim, self.action_dim, device=self.cholesky_log_diag.device)
        L.diagonal().copy_(torch.exp(self.cholesky_log_diag))
        idx = torch.tril_indices(self.action_dim, self.action_dim, offset=-1)
        L[idx[0], idx[1]] = self.cholesky_offdiag
        return L

    def _get_scale_weights(self):
        """Compute normalized scale weights satisfying Σ w_k² = 1."""
        raw_w = torch.exp(self.scale_log_weights)
        # Normalize so sum of squares = 1 (variance preservation)
        return raw_w / torch.sqrt((raw_w ** 2).sum())

    def _get_gsde_variance(self, latent):
        """Per-action marginal variance: var_i = sum_j (h_j^2 * sigma_j_i^2)."""
        std = self._get_gsde_std()
        return (latent ** 2) @ (std ** 2)

    def reset_noise(self, num_envs):
        """Sample fresh exploration matrices for all envs at all scales."""
        std = self._get_gsde_std()
        dist = Normal(torch.zeros_like(std), std)
        for k in range(self.num_scales):
            self.exploration_matrices_list[k] = dist.rsample((num_envs,))

    def reset_noise_for_envs(self, env_mask):
        """Resample exploration matrices for terminated envs at all scales."""
        if not env_mask.any():
            return
        n_reset = env_mask.sum().item()
        std = self._get_gsde_std()
        dist = Normal(torch.zeros_like(std), std)
        for k in range(self.num_scales):
            new_mats = dist.rsample((n_reset,))
            self.exploration_matrices_list[k][env_mask] = new_mats

    def evolve_noise(self):
        """OU update for all timescales, each at its own rate."""
        std = self._get_gsde_std()
        dist = Normal(torch.zeros_like(std), std)
        n_envs = self.exploration_matrices_list[0].shape[0]
        for k in range(self.num_scales):
            fresh = dist.rsample((n_envs,))
            alpha = self.noise_momentums[k]
            innov = self.innovation_scales[k]
            self.exploration_matrices_list[k] = alpha * self.exploration_matrices_list[k] + innov * fresh

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        latent = self.actor_backbone(x)
        mean = self.mean_net(latent)

        # State-dependent diagonal variance from gSDE
        gsde_var = self._get_gsde_variance(latent)  # [batch, action_dim]
        var_sqrt = torch.sqrt(gsde_var + 1e-6)  # [batch, action_dim]

        # Build full covariance via Cholesky: L @ diag(var) @ L^T
        L = self._get_cholesky()  # [action_dim, action_dim]

        # Scale L columns by var_sqrt: effective_L[b] = L @ diag(var_sqrt[b])
        effective_L = L.unsqueeze(0) * var_sqrt.unsqueeze(1)

        # MultivariateNormal with scale_tril for correct cross-actuator density
        # Variance is preserved because Σ w_k² = 1
        dist = MultivariateNormal(mean, scale_tril=effective_L)

        if action is None:
            # Multi-timescale noise: weighted sum across all scales
            weights = self._get_scale_weights()  # [num_scales], Σ w² = 1
            noise_total = torch.zeros_like(mean)
            for k in range(self.num_scales):
                noise_k = torch.bmm(
                    latent.unsqueeze(1), self.exploration_matrices_list[k]
                ).squeeze(1)  # [batch, action_dim]
                noise_total = noise_total + weights[k] * noise_k

            # Apply cross-actuator mixing: noise = z @ L^T
            noise_mixed = noise_total @ L.t()
            action = mean + noise_mixed

        log_prob = dist.log_prob(action)  # [batch]
        entropy = dist.entropy()  # [batch]
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
        gsde_log_std_init=args.gsde_log_std_init,
        hidden_dim=args.hidden_dim,
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

        # Initial noise sample at start of rollout (all scales)
        agent.reset_noise(args.num_envs)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Multi-timescale OU noise evolution
            if step > 0:
                agent.evolve_noise()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Resample noise for envs that just terminated/truncated (all scales)
            if args.resample_on_reset and next_done.any():
                agent.reset_noise_for_envs(next_done.bool())

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

                # Resample noise for training (marginal variance is analytically computed)
                agent.reset_noise(len(mb_inds))

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

        # Log diagnostics
        with torch.no_grad():
            writer.add_scalar("space/gsde_std_mean", agent._get_gsde_std().mean().item(), global_step)
            writer.add_scalar("space/gsde_std_max", agent._get_gsde_std().max().item(), global_step)
            L = agent._get_cholesky()
            offdiag_norm = (L - torch.diag(L.diagonal())).norm().item()
            writer.add_scalar("space/cholesky_offdiag_norm", offdiag_norm, global_step)
            writer.add_scalar("space/cholesky_diag_mean", L.diagonal().mean().item(), global_step)
            # Log multi-timescale weights
            weights = agent._get_scale_weights()
            for k in range(agent.num_scales):
                writer.add_scalar(f"space/scale_weight_{k}_mom{agent.noise_momentums[k]}", weights[k].item(), global_step)

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
