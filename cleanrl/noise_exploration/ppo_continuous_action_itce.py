# ITCE: Independent Time-Correlated Exploration for PPO
#
# This implements a structured exploration strategy for continuous control that
# produces *correlated* noise across actuators via a learned covariance structure,
# while maintaining *temporal* correlation by reusing exploration matrices across
# steps within a rollout (the gSDE trick).
#
# Background and motivation
# -------------------------
# Standard PPO uses independent diagonal Gaussian noise per action dimension.
# This is sample-inefficient in high-dimensional action spaces (e.g. humanoid
# control) because coordinated joint movements are exponentially unlikely under
# independent noise.
#
# gSDE (Generalized State-Dependent Exploration, Raffin et al. 2022) improves on
# this by making noise state-dependent: noise = latent @ exploration_matrix, where
# the exploration matrix is resampled per rollout rather than per step. This gives
# temporal consistency (smooth exploration trajectories) and state-dependence, but
# the resulting action-space covariance is still effectively diagonal — there is no
# mechanism to correlate noise across actuators.
#
# Lattice (Chiappa et al., NeurIPS 2023) introduces cross-actuator correlation by
# routing noise through the actor's weight matrix W, producing covariance structure
# W @ D @ W^T. However, sharing W between the action mean (mu = W @ h) and the
# covariance creates a gradient conflict: the policy gradient wants W to be a
# precise reward-maximizing projection, while the entropy bonus wants W to have
# large spread singular values. This couples exploration quality to mean quality.
#
# ITCE resolves this by decoupling the mean and covariance projections into two
# independent weight matrices, eliminating the gradient conflict while retaining
# the full Lattice covariance structure and gSDE temporal correlation.
#
# Architecture
# ------------
#   actor backbone:  obs -> [64, tanh, 64, tanh] -> latent h(s)     (d-dimensional)
#
#   mean pathway:    mu(s)     = W_mean @ h(s)                       (n_a-dimensional)
#   cov pathway:     Sigma(s)  = alpha^2 * W_cov @ D(s) @ W_cov^T + diag(sigma^2_ind(s))
#
#   D(s) and sigma^2_ind(s) are state-dependent variances derived from h(s) and
#   learned log-std parameters, following the Lattice formulation.
#
#   Sampling uses gSDE temporal correlation:
#     exploration matrices are sampled once per rollout, then reused for all steps.
#     action = mu(s) + W_cov @ (h(s) @ corr_exploration_mat) * alpha + (h(s) @ ind_exploration_mat)
#
# Key differences from Lattice:
#   1. W_cov is independent from W_mean — no gradient conflict
#   2. Mean clipping (when used) applies only to mu, not mu+noise — unbiased log_prob
#   3. Fully self-contained — no SB3 dependency

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions import MultivariateNormal, Normal
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# ITCE Distribution
# ---------------------------------------------------------------------------

class ITCEDistribution(nn.Module):
    """Independent Time-Correlated Exploration distribution.

    Produces actions from a MultivariateNormal whose covariance has the structure:

        Sigma = alpha^2 * W_cov @ diag(v_corr(s)) @ W_cov^T + diag(v_ind(s))

    where W_cov is a *dedicated* covariance projection (independent of the mean
    projection W_mean), and v_corr / v_ind are state-dependent variances.

    Temporal correlation comes from the gSDE mechanism: exploration matrices are
    sampled once (per call to `sample_exploration_matrices`) and reused across
    all subsequent `sample()` calls until the next resample.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        log_std_init: float = 0.0,
        alpha: float = 1.0,
        std_reg: float = 0.0,
        min_std: float = 1e-3,
        max_std: float = 100.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.std_reg = std_reg
        self.min_std = min_std
        self.max_std = max_std

        # --- Mean pathway ---
        self.mean_net = nn.Linear(latent_dim, action_dim)

        # --- Covariance pathway ---
        # No bias: noise projections must be zero-mean.  Any constant offset
        # belongs in the mean pathway.
        self.cov_net = nn.Linear(latent_dim, action_dim, bias=False)

        # --- Log-std parameters (full parameterization) ---
        # Shape: [latent_dim, latent_dim + action_dim]
        #   columns 0..latent_dim-1    -> correlated noise std per (source, target) pair
        #   columns latent_dim..end    -> independent noise std per (source, action) pair
        #
        # This full parameterization allows each latent dimension to contribute
        # a different noise scale to each target dimension, giving the optimizer
        # independent control over the spectral profile of the covariance.
        # Without this, diag(v_corr) collapses to a scalar * I, and the
        # off-diagonal correlation structure is locked to W_cov @ W_cov^T
        # with no way to selectively amplify or suppress latent directions.
        self.log_std = nn.Parameter(
            torch.ones(latent_dim, latent_dim + action_dim) * log_std_init,
            requires_grad=True,
        )

        # Dimension scaling correction: prevents variance from growing with
        # latent dimension.  See Lattice paper appendix.
        self._log_dim_correction = 0.5 * math.log(latent_dim)

        # Exploration matrices (set by sample_exploration_matrices).
        # Stored as plain tensors (not parameters) — they are sampled, not learned.
        self.register_buffer("corr_exploration_mat", None)
        self.register_buffer("ind_exploration_mat", None)
        self.register_buffer("corr_exploration_matrices", None)
        self.register_buffer("ind_exploration_matrices", None)

        # Cache for the distribution built in get_action_distribution
        self._distribution: Optional[MultivariateNormal] = None

        self._init_weights()

    def _init_weights(self):
        # Mean net: small init for stable initial policy (cleanRL convention)
        nn.init.orthogonal_(self.mean_net.weight, gain=0.01)
        nn.init.zeros_(self.mean_net.bias)
        # Cov net: standard init — exploration should start with reasonable scale
        nn.init.orthogonal_(self.cov_net.weight, gain=1.0)

    # --- Std computation ---

    def _get_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute correlated and independent std from log_std parameter.

        Returns:
            corr_std: [latent_dim, latent_dim] — std for correlated noise
            ind_std:  [latent_dim, action_dim]  — std for independent noise
        """
        log_std = self.log_std.clamp(
            min=math.log(self.min_std), max=math.log(self.max_std)
        )
        log_std = log_std - self._log_dim_correction
        std = torch.exp(log_std)

        corr_std = std[:, :self.latent_dim]
        ind_std = std[:, self.latent_dim:]
        return corr_std, ind_std

    # --- Exploration matrix sampling (gSDE temporal correlation) ---

    def sample_exploration_matrices(self, batch_size: int = 1) -> None:
        """Sample new exploration matrices.

        Call this once per rollout to get temporal correlation (gSDE mechanism).
        The matrices are reused for all steps within the rollout.
        """
        corr_std, ind_std = self._get_std()

        corr_dist = Normal(torch.zeros_like(corr_std), corr_std)
        ind_dist = Normal(torch.zeros_like(ind_std), ind_std)

        # Single matrix for non-batched inference (e.g. env stepping)
        self.corr_exploration_mat = corr_dist.rsample()
        self.ind_exploration_mat = ind_dist.rsample()

        # Batch of matrices for parallel log_prob evaluation during training
        self.corr_exploration_matrices = corr_dist.rsample((batch_size,))
        self.ind_exploration_matrices = ind_dist.rsample((batch_size,))

    # --- Noise computation ---

    def _get_noise(
        self,
        latent: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state-dependent noise: latent @ exploration_matrix.

        The latent is detached so that the noise magnitude (which is a function
        of h(s)) does not push gradients into the actor backbone.  Noise
        scaling should only train cov_net and log_std, not the feature extractor.

        Uses the single exploration_mat for single observations, or batched
        bmm for parallel evaluation.
        """
        latent = latent.detach()
        if len(latent) == 1 or len(latent) != len(exploration_matrices):
            return latent @ exploration_mat
        return torch.bmm(
            latent.unsqueeze(1), exploration_matrices
        ).squeeze(1)

    # --- Core interface ---

    def _build_covariance(
        self, latent: torch.Tensor
    ) -> torch.Tensor:
        """Build the covariance matrix Sigma(s) from a (detached) latent.

        Sigma = alpha^2 * W_cov @ diag(v_corr(s)) @ W_cov^T + diag(v_ind(s))

        This is factored out so it can be called with or without gradients
        flowing through cov_net / log_std.
        """
        latent_detached = latent.detach()
        corr_std, ind_std = self._get_std()

        # State-dependent variances: h(s)^2 @ std^2
        latent_corr_var = (latent_detached ** 2) @ (corr_std ** 2)
        latent_ind_var = (latent_detached ** 2) @ (ind_std ** 2) + self.std_reg ** 2

        W = self.cov_net.weight  # [action_dim, latent_dim]
        sigma_mat = self.alpha ** 2 * (
            (W * latent_corr_var[:, None, :]) @ W.T
        )
        sigma_mat[:, range(self.action_dim), range(self.action_dim)] += latent_ind_var
        return sigma_mat

    def get_action_distribution(
        self, latent: torch.Tensor
    ) -> MultivariateNormal:
        """Build the MultivariateNormal for the given latent.

        The full covariance (with gradients through cov_net and log_std) is
        used for both ``log_prob`` and ``entropy``.  This mirrors standard
        diagonal-Gaussian PPO where ``log_std`` receives gradient through
        ``log_prob`` naturally:

            log p(a) = -0.5 * [ (a-μ)ᵀ Σ⁻¹ (a-μ)  +  log|Σ|  +  k·log(2π) ]

        The Mahalanobis term ``(a-μ)ᵀ Σ⁻¹ (a-μ)`` pushes Σ larger when
        actions are far from the mean (exploration signal).  The normalization
        term ``log|Σ|`` pushes Σ smaller (regularization signal).  These
        naturally balance, exactly as ``-log(σ)`` and ``(a-μ)²/σ²`` balance
        in the univariate case.

        No stored-covariance decomposition or entropy bonus is needed —
        ``cov_net`` and ``log_std`` receive the same push-pull gradient through
        ``log_prob`` that ``log_std`` gets in standard PPO.
        """
        self._latent = latent
        self._mean = self.mean_net(latent)
        sigma_mat = self._build_covariance(latent)
        self._distribution = MultivariateNormal(
            loc=self._mean, covariance_matrix=sigma_mat, validate_args=False
        )
        return self._distribution

    def sample(self, latent: torch.Tensor) -> torch.Tensor:
        """Sample an action using the gSDE exploration matrices.

        This is the *actual* sampling path used during rollouts.  It computes:
            action = W_mean @ h  +  W_cov @ (h @ corr_mat) * alpha  +  (h @ ind_mat)  +  reg_noise

        The mean and noise are cleanly separated — no coupling through clipping
        or shared weights.
        """
        mean = self.mean_net(latent)

        corr_noise = self.alpha * self._get_noise(
            latent, self.corr_exploration_mat, self.corr_exploration_matrices
        )
        ind_noise = self._get_noise(
            latent, self.ind_exploration_mat, self.ind_exploration_matrices
        )

        # Project correlated noise through the *covariance* network (not the mean network)
        correlated_action_noise = corr_noise @ self.cov_net.weight.T  # [batch, action_dim]

        actions = mean + correlated_action_noise + ind_noise

        # Add std_reg noise to match the covariance diagonal floor.
        # Without this, the density evaluation (which includes std_reg^2 on the
        # diagonal) would assume more variance than the sampling actually produces.
        if self.std_reg > 0:
            actions = actions + torch.randn_like(actions) * self.std_reg

        return actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability under the current distribution.

        Gradients flow through both mean_net (via the Mahalanobis term) and
        cov_net / log_std (via both the Mahalanobis and normalization terms),
        providing the natural push-pull that prevents entropy collapse.
        """
        return self._distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        """Analytical entropy of the MultivariateNormal.

        Entropy = 0.5 * (k * log(2πe) + log|Σ|).  Depends only on the
        covariance, so gradients flow through cov_net and log_std.
        """
        return self._distribution.entropy()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(self, envs, itce_alpha: float, itce_std_reg: float, itce_log_std_init: float):
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
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.dist = ITCEDistribution(
            latent_dim=latent_dim,
            action_dim=action_dim,
            log_std_init=itce_log_std_init,
            alpha=itce_alpha,
            std_reg=itce_std_reg,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        latent = self.actor_backbone(x)
        self.dist.get_action_distribution(latent)
        if action is None:
            action = self.dist.sample(latent)
        log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        return action, log_prob, entropy, self.critic(x)

    def sample_exploration_matrices(self, batch_size: int = 1):
        self.dist.sample_exploration_matrices(batch_size)


# ---------------------------------------------------------------------------
# Args & helpers (cleanRL standard)
# ---------------------------------------------------------------------------

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

    # ITCE specific arguments
    itce_alpha: float = 1.0
    """weight of correlated (cross-actuator) noise relative to independent noise; 0 disables correlation"""
    itce_std_reg: float = 0.0
    """minimum independent noise floor to prevent collapse to deterministic policy"""
    itce_log_std_init: float = 0.0
    """initial value for the log standard deviation parameters"""

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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

    agent = Agent(envs, args.itce_alpha, args.itce_std_reg, args.itce_log_std_init).to(device)
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

        # ITCE: resample exploration matrices once per rollout for temporal correlation.
        # This is the gSDE mechanism — the same noise structure is reused across all
        # steps within this rollout, producing smooth exploration trajectories.
        agent.sample_exploration_matrices(batch_size=args.num_envs)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds],
                )
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
