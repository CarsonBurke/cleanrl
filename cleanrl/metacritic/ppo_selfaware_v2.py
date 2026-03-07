"""PPO Self-Aware Critic v2: World Model Uncertainty as Self-Awareness

Merges three ideas:
1. lstd_linear_tanh actor (best SDE exploration architecture)
2. Latent world model as critic auxiliary (from imagination v2 — value consistency)
3. Self-awareness via model uncertainty: the world model's transition std tells
   the critic how predictable the future is from each state-action pair

Architecture:
  - Actor: lstd_linear_tanh SDE (RMSNorm, SiLU, learned log_std_param, mean_scale)
  - Critic: lstd_linear_tanh backbone, trained with PPO value loss + value consistency
  - World Model (separate optimizer, detached from actor/critic):
    * Encoder: obs -> 64-dim latent
    * Stochastic transition: (z, a) -> N(z'_mean, z'_std) — std is the uncertainty
    * Reward head: (z, a) -> predicted reward
    * Value head: z -> predicted value (for value consistency with critic)
    * Target encoder: EMA copy for stable transition targets

Self-Awareness Mechanism:
  During PPO update, for each (obs, action) in the minibatch:
  1. World model encodes obs -> z, predicts transition (mean, std)
  2. uncertainty = mean(std) per observation — how unpredictable the future is
  3. confidence = 1/(1 + alpha * uncertainty), re-normalized to preserve adv scale
  4. Advantages weighted by confidence: uncertain states get less gradient
  This focuses learning on states where the value estimates are reliable.

Key lessons applied:
  - No imagined advantages for actor (v2's lesson: imagination poisons the actor)
  - Separate optimizer for world model (v6's fix: no gradient interference)
  - Value consistency only (v2's success: critic-side auxiliary, not actor-side)
  - Confidence warmup: don't gate advantages until world model has trained enough
"""
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
SDE_EPS = 1e-6
SDE_PRESCALE = 1.5


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
    clip_coef_low: float = 0.2
    """the lower surrogate clipping coefficient"""
    clip_coef_high: float = 0.28
    """the upper surrogate clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""

    # World model arguments
    wm_lr: float = 3e-4
    """world model learning rate"""
    wm_latent_dim: int = 64
    """latent space dimension"""
    wm_hidden_dim: int = 128
    """world model hidden layer width"""
    wm_epochs: int = 2
    """world model update epochs per PPO iteration"""
    wm_max_grad_norm: float = 1.0
    """gradient clipping for world model"""
    wm_ema_tau: float = 0.99
    """EMA decay for target encoder"""
    wm_transition_std_min: float = 0.05
    """minimum transition std"""
    wm_transition_std_max: float = 1.0
    """maximum transition std"""

    # Value consistency
    value_consistency_coef: float = 0.1
    """coefficient for value consistency auxiliary loss on critic"""

    # Self-awareness
    confidence_start_frac: float = 0.1
    """fraction of training before confidence weighting activates"""
    confidence_alpha: float = 2.0
    """scaling factor for uncertainty -> confidence mapping"""

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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class WorldModel(nn.Module):
    """Latent world model with stochastic transitions."""

    def __init__(self, obs_dim, act_dim, latent_dim=64, hidden_dim=128,
                 std_min=0.05, std_max=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.std_min = std_min
        self.std_max = std_max

        # Encoder: obs -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        # Target encoder (EMA copy)
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        # Initialize target = encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Stochastic transition: (z, a) -> (z'_mean, z'_std)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.transition_mean = nn.Linear(hidden_dim, latent_dim)
        self.transition_logstd = nn.Linear(hidden_dim, latent_dim)

        # Reward head: (z, a) -> scalar
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value head: z -> scalar (for value consistency)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, obs):
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs):
        return self.target_encoder(obs)

    def predict_transition(self, z, a):
        """Returns (next_z_mean, next_z_std)."""
        h = self.transition(torch.cat([z, a], dim=-1))
        mean = self.transition_mean(h)
        logstd = self.transition_logstd(h).clamp(-4.0, 2.0)
        std = logstd.exp().clamp(self.std_min, self.std_max)
        return mean, std

    def predict_reward(self, z, a):
        return self.reward_head(torch.cat([z, a], dim=-1)).squeeze(-1)

    def predict_value(self, z):
        return self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

    @torch.no_grad()
    def get_uncertainty(self, obs, action):
        """Per-observation transition uncertainty (mean std across latent dims)."""
        z = self.encode(obs)
        _, std = self.predict_transition(z, action)
        return std.mean(dim=-1)  # (batch,)


class Agent(nn.Module):
    """lstd_linear_tanh actor + critic (unchanged from base)."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        hidden_dim = 64

        # Actor backbone
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_norm2 = RMSNorm(hidden_dim)
        self.actor_out = layer_init(nn.Linear(hidden_dim, act_dim), std=1.0)
        self.mean_scale = nn.Parameter(torch.tensor(0.01))

        # SDE noise
        self.sde_fc = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.sde_norm = RMSNorm(hidden_dim)
        self.sde_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.log_std_param = nn.Parameter(torch.zeros(hidden_dim, act_dim))

        # Critic backbone
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_norm2 = RMSNorm(hidden_dim)
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _actor_features(self, x):
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))
        h = F.silu(self.actor_norm2(self.actor_fc2(h)))
        return h

    def _get_action_std(self, h):
        sde_raw = self.sde_fc(h)
        sde_latent = (self.sde_fc2(self.sde_norm(sde_raw)) / SDE_PRESCALE).tanh()
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std_sq = log_std.exp().pow(2)
        action_var = (sde_latent.pow(2)) @ std_sq
        action_std = (action_var + SDE_EPS).sqrt()
        return action_std

    def _critic_features(self, x):
        h = F.silu(self.critic_norm1(self.critic_fc1(x)))
        h = F.silu(self.critic_norm2(self.critic_fc2(h)))
        return h

    def get_value(self, x):
        h = self._critic_features(x)
        return self.value_out(h)

    def get_action_and_value(self, x, action=None):
        h = self._actor_features(x)
        action_mean = self.actor_out(h) * self.mean_scale
        action_std = self._get_action_std(h)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args), name=run_name,
            monitor_gym=True, save_code=True,
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
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)

    agent = Agent(envs, args).to(device)
    world_model = WorldModel(
        obs_dim, act_dim,
        latent_dim=args.wm_latent_dim,
        hidden_dim=args.wm_hidden_dim,
        std_min=args.wm_transition_std_min,
        std_max=args.wm_transition_std_max,
    ).to(device)

    # Separate optimizers
    behavior_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    wm_optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr, eps=1e-5)

    # Storage
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

    confidence_start_step = int(args.confidence_start_frac * args.total_timesteps)

    for iteration in range(1, args.num_iterations + 1):
        # Anneal LR
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            behavior_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # === Rollout collection ===
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_obses[step] = next_obs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # === GAE ===
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
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)

        # === Compute confidence weights (detached, before PPO update) ===
        use_confidence = global_step >= confidence_start_step
        if use_confidence:
            with torch.no_grad():
                uncertainty = world_model.get_uncertainty(b_obs, b_actions)
                confidence = 1.0 / (1.0 + args.confidence_alpha * uncertainty)
                # Re-normalize to preserve mean advantage scale
                confidence = confidence / (confidence.mean() + 1e-8)
        else:
            confidence = torch.ones(args.batch_size, device=device)

        # === PPO update (behavior) ===
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        all_v_loss = []
        all_vc_loss = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio < (1 - args.clip_coef_low)) | (ratio > (1 + args.clip_coef_high)))
                        .float().mean().item()
                    ]

                # Self-aware advantage weighting
                mb_advantages = b_advantages[mb_inds] * confidence[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef_low, 1 + args.clip_coef_high)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef_low,
                        args.clip_coef_high,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Value consistency: pull critic toward world model's value estimate
                with torch.no_grad():
                    z = world_model.encode(b_obs[mb_inds])
                    wm_value = world_model.predict_value(z)
                vc_loss = args.value_consistency_coef * F.mse_loss(newvalue, wm_value)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + vc_loss

                behavior_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                behavior_optimizer.step()

                all_v_loss.append(v_loss.item())
                all_vc_loss.append(vc_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # === World model update (separate phase) ===
        wm_transition_losses = []
        wm_reward_losses = []
        wm_value_losses = []
        wm_transition_stds = []

        for wm_epoch in range(args.wm_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_next_obs = b_next_obs[mb_inds]
                mb_rewards = b_rewards[mb_inds]

                # Encode current obs
                z = world_model.encode(mb_obs)
                # Target encode next obs (stable target)
                z_next_target = world_model.encode_target(mb_next_obs)

                # Transition loss: NLL of z_next_target under predicted distribution
                z_next_mean, z_next_std = world_model.predict_transition(z, mb_actions)
                transition_dist = Normal(z_next_mean, z_next_std)
                transition_loss = -transition_dist.log_prob(z_next_target).mean()

                # Reward loss
                pred_reward = world_model.predict_reward(z, mb_actions)
                reward_loss = F.mse_loss(pred_reward, mb_rewards)

                # World model value loss: predict critic's value (detached)
                with torch.no_grad():
                    critic_value = agent.get_value(mb_obs).squeeze(-1)
                wm_value_pred = world_model.predict_value(z)
                wm_value_loss = F.mse_loss(wm_value_pred, critic_value)

                wm_loss = transition_loss + reward_loss + 0.5 * wm_value_loss

                wm_optimizer.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(world_model.parameters(), args.wm_max_grad_norm)
                wm_optimizer.step()

                # Update target encoder
                world_model.update_target(args.wm_ema_tau)

                wm_transition_losses.append(transition_loss.item())
                wm_reward_losses.append(reward_loss.item())
                wm_value_losses.append(wm_value_loss.item())
                wm_transition_stds.append(z_next_std.mean().item())

        # === Logging ===
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", behavior_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", np.mean(all_v_loss), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # World model metrics
        writer.add_scalar("worldmodel/transition_loss", np.mean(wm_transition_losses), global_step)
        writer.add_scalar("worldmodel/reward_loss", np.mean(wm_reward_losses), global_step)
        writer.add_scalar("worldmodel/value_loss", np.mean(wm_value_losses), global_step)
        writer.add_scalar("worldmodel/transition_std", np.mean(wm_transition_stds), global_step)
        writer.add_scalar("worldmodel/value_consistency_loss", np.mean(all_vc_loss), global_step)

        # Self-awareness metrics
        if use_confidence:
            writer.add_scalar("selfaware/confidence_mean", confidence.mean().item(), global_step)
            writer.add_scalar("selfaware/confidence_std", confidence.std().item(), global_step)
            writer.add_scalar("selfaware/uncertainty_mean", uncertainty.mean().item(), global_step)

        # lstd-specific
        log_std_eff = (agent.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        writer.add_scalar("tbot/log_std_mean", log_std_eff.mean().item(), global_step)
        writer.add_scalar("tbot/log_std_std", log_std_eff.std().item(), global_step)
        writer.add_scalar("tbot/mean_scale", agent.mean_scale.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
