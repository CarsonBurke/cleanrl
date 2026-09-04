# PMPO-D4 + ReluSq + TanhNormal action-cell policy with global std v1.
#
# Catbins showed the useful constraint: PMPO should optimize action probability
# mass, not continuous point density or tanh-Jacobian geometry. But discrete
# action bins lost smooth continuous-control structure. This variant keeps
# TanhNormal continuous rollout actions, then trains PMPO on the probability
# mass of the fixed action-space cell containing each sampled action:
#
#   log P(a in [cell_lo, cell_hi])
#     = log(Phi(atanh(cell_hi); mu, sigma)
#           - Phi(atanh(cell_lo); mu, sigma))
#
# Earlier action-cell variants found two failure modes: full categorical cell KL
# over-penalized rare tail cells, while state-dependent TanhNormal std still
# allowed huge rare-state latent KL spikes. This version keeps the action-cell
# PMPO actor signal and latent Normal KL trust region, but uses one learned
# global log-std vector like PPO's baseline. Hypothesis: remove the per-state
# sigma escape hatch while preserving smooth continuous actions.

import os
import random
import time
from dataclasses import dataclass
from math import log, pi
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.kl import kl_divergence
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
    total_timesteps: int = 1000000
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
    """Toggles PPO advantage normalization. PMPO intentionally ignores this."""
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
    """PPO-only target ratio KL threshold"""

    # PMPO-D4 specific arguments
    policy_objective: Literal["pmpo", "ppo"] = "pmpo"
    """actor objective: pmpo is D4-style PMPO, ppo is base clipped PPO"""
    pmpo_pos_to_neg_weight: float = 0.5
    """PMPO positive-advantage weight; negative side uses 1 - this value"""
    pmpo_kl_coef: float = 0.1
    """reverse KL(old rollout policy || new policy) coefficient"""
    pmpo_reverse_kl: bool = True
    """use KL(old||new); false uses KL(new||old) for ablations"""
    pmpo_target_kl: float = 0.05
    """early-stop update epochs when running-mean reverse_kl exceeds this. None disables."""
    action_cells: int = 51
    """number of fixed action-space cells used for PMPO mass objective and KL"""

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


def logdiffexp(a, b):
    return a + torch.log1p(-torch.exp(b - a))


def normal_log_cdf(x):
    out = torch.empty_like(x)
    central_mask = x > -5.0
    if central_mask.any():
        central_x = x[central_mask]
        out[central_mask] = torch.log(0.5 * (1.0 + torch.erf(central_x / np.sqrt(2.0))))
    tail_mask = ~central_mask
    if tail_mask.any():
        tail_x = x[tail_mask]
        abs_x = tail_x.abs()
        inv_x2 = 1.0 / (abs_x * abs_x)
        correction = torch.log1p(-inv_x2 + 3.0 * inv_x2.square())
        out[tail_mask] = -0.5 * tail_x.square() - torch.log(abs_x) - 0.5 * log(2.0 * pi) + correction
    return out


def normal_log_interval_prob(mean, std, edges):
    dtype = mean.dtype
    mean64 = mean.double()
    std64 = std.double()
    finite_edges = edges[1:-1].double()
    num_cells = edges.numel() - 1
    out = torch.empty(mean.shape + (num_cells,), dtype=torch.float64, device=mean.device)

    first_upper = (finite_edges[0] - mean64) / std64
    out[..., 0] = normal_log_cdf(first_upper)
    last_lower = (finite_edges[-1] - mean64) / std64
    out[..., -1] = normal_log_cdf(-last_lower)

    if num_cells > 2:
        lower = (finite_edges[:-1].view(1, 1, -1) - mean64.unsqueeze(-1)) / std64.unsqueeze(-1)
        upper = (finite_edges[1:].view(1, 1, -1) - mean64.unsqueeze(-1)) / std64.unsqueeze(-1)
        interior_out = torch.empty_like(lower)

        left_mask = lower <= 0.0
        if left_mask.any():
            log_cdf_upper = normal_log_cdf(upper[left_mask])
            log_cdf_lower = normal_log_cdf(lower[left_mask])
            interior_out[left_mask] = logdiffexp(log_cdf_upper, log_cdf_lower)

        right_mask = ~left_mask
        if right_mask.any():
            log_sf_lower = normal_log_cdf(-lower[right_mask])
            log_sf_upper = normal_log_cdf(-upper[right_mask])
            interior_out[right_mask] = logdiffexp(log_sf_lower, log_sf_upper)
        out[..., 1:-1] = interior_out
    return out.to(dtype)


def kl_from_log_probs(p_log_probs, q_log_probs):
    p_probs = p_log_probs.exp()
    diff = torch.where(p_probs > 0.0, p_log_probs - q_log_probs, torch.zeros_like(p_log_probs))
    return (p_probs * diff).sum(dim=-1)


def entropy_from_log_probs(log_probs):
    probs = log_probs.exp()
    safe_log_probs = torch.where(probs > 0.0, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


class ReluSq(nn.Module):
    """f(x) = relu(x)^2 — x^2 for x>=0, 0 for x<0."""
    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    def __init__(self, envs, action_cells=51):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.action_cells = action_cells

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor predicts only the state-dependent latent mean. A single learned
        # log-std vector avoids rare per-state sigma jumps that produced giant
        # KL spikes in the fully state-dependent action-cell variant.
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer(
            "action_center",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        # log(action_scale) for Jacobian (constant, cancels in ratios).
        self.register_buffer("log_action_scale", torch.log(self.action_scale))
        normalized_edges = torch.linspace(-1.0, 1.0, action_cells + 1, dtype=torch.float32)
        latent_edges = torch.empty_like(normalized_edges)
        latent_edges[0] = -torch.inf
        latent_edges[-1] = torch.inf
        latent_edges[1:-1] = torch.atanh(normalized_edges[1:-1])
        self.register_buffer("latent_cell_edges", latent_edges)

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        mean = self.actor(x)
        std = self.actor_logstd.exp().expand_as(mean)
        return Normal(mean, std)

    def squash_action(self, z):
        return self.action_center + self.action_scale * torch.tanh(z)

    def squash_log_abs_det_jacobian(self, z):
        # log|d a / d z| = log(action_scale) + log(1 - tanh(z)^2)
        # Numerically stable: log(1 - tanh(z)^2) = 2*(log 2 - z - softplus(-2z))
        return self.log_action_scale + 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))

    def get_action_cell_indices(self, z):
        return torch.bucketize(z, self.latent_cell_edges[1:-1].contiguous())

    def get_action_cell_log_probs(self, latent_dist):
        return normal_log_interval_prob(latent_dist.mean, latent_dist.stddev, self.latent_cell_edges)

    def get_action_and_value(self, x, z=None):
        # z is the pre-tanh latent. None → sample fresh; else re-evaluate at stored z.
        latent_dist = self.get_action_distribution(x)
        if z is None:
            z = latent_dist.sample()
        action = self.squash_action(z)
        log_det = self.squash_log_abs_det_jacobian(z)
        log_prob_per_dim = latent_dist.log_prob(z) - log_det
        entropy_per_dim = latent_dist.entropy()
        return action, z, log_prob_per_dim.sum(1), entropy_per_dim.sum(1), self.critic(x)

    def get_squashed_action_and_value(self, x, z=None):
        # PMPO branch — returns latent Normal dist for closed-form reverse-KL.
        latent_dist = self.get_action_distribution(x)
        if z is None:
            z = latent_dist.sample()
        action = self.squash_action(z)
        log_det = self.squash_log_abs_det_jacobian(z)
        log_prob_per_dim = latent_dist.log_prob(z) - log_det
        entropy_per_dim = latent_dist.entropy()
        return (
            action,
            z,
            log_prob_per_dim.sum(1),
            entropy_per_dim.sum(1),
            self.critic(x),
            latent_dist,
            log_prob_per_dim,
        )


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, policy_objective, action_cells):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs, action_cells).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            if policy_objective == "pmpo":
                actions, _, _, _, _, _, _ = agent.get_squashed_action_and_value(obs_tensor)
            else:
                actions, _, _, _, _ = agent.get_action_and_value(obs_tensor)
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs.close()
    return episodic_returns


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

    agent = Agent(envs, args.action_cells).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # TanhNormal stores latent Normal (μ, σ) per dim for closed-form KL.
    old_action_means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    old_action_stds = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # Pre-tanh latent z stored directly — no atanh roundtrip during update.
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_cell_indices = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.long
    ).to(device)

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
                if args.policy_objective == "pmpo":
                    action, z, logprob, _, value, action_dist, _ = agent.get_squashed_action_and_value(next_obs)
                    cell_indices = agent.get_action_cell_indices(z)
                    cell_log_probs = agent.get_action_cell_log_probs(action_dist)
                    logprob = cell_log_probs.gather(-1, cell_indices.unsqueeze(-1)).squeeze(-1).sum(1)
                else:
                    action, z, logprob, _, value = agent.get_action_and_value(next_obs)
                    action_dist = agent.get_action_distribution(next_obs)
                    cell_indices = agent.get_action_cell_indices(z)
                values[step] = value.flatten()
                old_action_means[step] = action_dist.mean
                old_action_stds[step] = action_dist.stddev
                latent_zs[step] = z
                action_cell_indices[step] = cell_indices
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
        b_old_action_means = old_action_means.reshape((-1,) + envs.single_action_space.shape)
        b_old_action_stds = old_action_stds.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_action_cell_indices = action_cell_indices.reshape((-1,) + envs.single_action_space.shape)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        reverse_kl = torch.zeros((), device=device)
        reverse_kl_sum = torch.zeros((), device=device)
        reverse_kl_count = 0
        reverse_kl_max = torch.zeros((), device=device)
        cell_reverse_kl = torch.zeros((), device=device)
        cell_reverse_kl_sum = torch.zeros((), device=device)
        cell_reverse_kl_count = 0
        cell_reverse_kl_max = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.policy_objective == "pmpo":
                    _, _, _, entropy, newvalue, new_dist, _ = agent.get_squashed_action_and_value(
                        b_obs[mb_inds], b_latent_zs[mb_inds]
                    )
                    new_cell_log_probs = agent.get_action_cell_log_probs(new_dist)
                    log_prob_per_dim = new_cell_log_probs.gather(
                        -1, b_action_cell_indices[mb_inds].unsqueeze(-1)
                    ).squeeze(-1)
                    newlogprob = log_prob_per_dim.sum(1)
                else:
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_latent_zs[mb_inds])
                    new_dist = None
                    log_prob_per_dim = None
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.policy_objective == "ppo":
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
                    adv_weight = mb_advantages.tanh().abs().unsqueeze(-1)
                    signed_log_probs = log_prob_per_dim * adv_weight
                    pos_mask = (mb_advantages >= 0).unsqueeze(-1).expand_as(log_prob_per_dim)
                    neg_mask = (mb_advantages < 0).unsqueeze(-1).expand_as(log_prob_per_dim)

                    pos_count = pos_mask.float().sum().clamp_min(1.0)
                    neg_count = neg_mask.float().sum().clamp_min(1.0)
                    pos_loss = signed_log_probs[pos_mask].sum() / pos_count
                    neg_loss = signed_log_probs[neg_mask].sum() / neg_count
                    pg_loss = (
                        -args.pmpo_pos_to_neg_weight * pos_loss
                        + (1.0 - args.pmpo_pos_to_neg_weight) * neg_loss
                    )

                    old_dist = Normal(b_old_action_means[mb_inds], b_old_action_stds[mb_inds])
                    old_cell_log_probs = agent.get_action_cell_log_probs(old_dist)
                    cell_reverse_kl_per_sample = kl_from_log_probs(
                        old_cell_log_probs.detach(), new_cell_log_probs.detach()
                    ).sum(-1)
                    cell_reverse_kl = cell_reverse_kl_per_sample.mean()
                    cell_reverse_kl_sum = cell_reverse_kl_sum + cell_reverse_kl.detach() * len(mb_inds)
                    cell_reverse_kl_count += len(mb_inds)
                    cell_reverse_kl_max = torch.maximum(
                        cell_reverse_kl_max, cell_reverse_kl_per_sample.detach().max()
                    )
                    if args.pmpo_reverse_kl:
                        reverse_kl_per_sample = kl_divergence(old_dist, new_dist).sum(-1)
                    else:
                        reverse_kl_per_sample = kl_divergence(new_dist, old_dist).sum(-1)
                    reverse_kl = reverse_kl_per_sample.mean()
                    reverse_kl_sum = reverse_kl_sum + reverse_kl.detach() * len(mb_inds)
                    reverse_kl_count += len(mb_inds)
                    reverse_kl_max = torch.maximum(reverse_kl_max, reverse_kl_per_sample.detach().max())
                    if args.pmpo_kl_coef > 0.0:
                        pg_loss = pg_loss + args.pmpo_kl_coef * reverse_kl

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

            if args.policy_objective == "ppo" and args.target_kl is not None and approx_kl > args.target_kl:
                break
            # Dynamic trust region: early-stop PMPO epochs on running-mean reverse_kl.
            if (
                args.policy_objective == "pmpo"
                and args.pmpo_target_kl is not None
                and reverse_kl_count > 0
                and (reverse_kl_sum / reverse_kl_count).item() > args.pmpo_target_kl
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.policy_objective == "pmpo" and reverse_kl_count > 0:
            reverse_kl = reverse_kl_sum / reverse_kl_count
        if args.policy_objective == "pmpo" and cell_reverse_kl_count > 0:
            cell_reverse_kl = cell_reverse_kl_sum / cell_reverse_kl_count
        policy_kl = approx_kl if args.policy_objective == "ppo" else reverse_kl

        # TanhNormal diagnostics — latent (μ, σ) statistics.
        with torch.no_grad():
            mean_mu = b_old_action_means.mean().item()
            abs_mean_mu = b_old_action_means.abs().mean().item()
            max_abs_mu = b_old_action_means.abs().max().item()
            mean_sigma = b_old_action_stds.mean().item()
            min_sigma = b_old_action_stds.min().item()
            max_sigma = b_old_action_stds.max().item()
            mean_logstd = b_old_action_stds.log().mean().item()
            old_diag_dist = Normal(b_old_action_means, b_old_action_stds)
            old_cell_log_probs = agent.get_action_cell_log_probs(old_diag_dist)
            old_cell_probs = old_cell_log_probs.exp()
            mean_cell_entropy = entropy_from_log_probs(old_cell_log_probs).mean().item()
            mean_cell_max_prob = old_cell_probs.max(dim=-1).values.mean().item()
            mean_edge_prob = (old_cell_probs[..., 0] + old_cell_probs[..., -1]).mean().item()
            selected_edge_frac = (
                ((b_action_cell_indices == 0) | (b_action_cell_indices == args.action_cells - 1)).float().mean().item()
            )
            mean_abs_z = b_latent_zs.abs().mean().item()
            mean_tanh_deriv = (1.0 - torch.tanh(b_latent_zs).square()).mean().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/reverse_kl", reverse_kl.item(), global_step)
        writer.add_scalar("losses/reverse_kl_max", reverse_kl_max.item(), global_step)
        writer.add_scalar("losses/policy_kl", policy_kl.item(), global_step)
        writer.add_scalar("losses/actioncell_cell_reverse_kl", cell_reverse_kl.item(), global_step)
        writer.add_scalar("losses/actioncell_cell_reverse_kl_max", cell_reverse_kl_max.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/tanhnormal_mean_mu", mean_mu, global_step)
        writer.add_scalar("diag/tanhnormal_abs_mean_mu", abs_mean_mu, global_step)
        writer.add_scalar("diag/tanhnormal_max_abs_mu", max_abs_mu, global_step)
        writer.add_scalar("diag/tanhnormal_mean_sigma", mean_sigma, global_step)
        writer.add_scalar("diag/tanhnormal_min_sigma", min_sigma, global_step)
        writer.add_scalar("diag/tanhnormal_max_sigma", max_sigma, global_step)
        writer.add_scalar("diag/tanhnormal_mean_logstd", mean_logstd, global_step)
        writer.add_scalar("diag/actioncell_mean_entropy", mean_cell_entropy, global_step)
        writer.add_scalar("diag/actioncell_mean_max_prob", mean_cell_max_prob, global_step)
        writer.add_scalar("diag/actioncell_mean_edge_prob", mean_edge_prob, global_step)
        writer.add_scalar("diag/actioncell_selected_edge_frac", selected_edge_frac, global_step)
        writer.add_scalar("diag/actioncell_mean_abs_z", mean_abs_z, global_step)
        writer.add_scalar("diag/actioncell_mean_tanh_deriv", mean_tanh_deriv, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        episodic_returns = evaluate_policy(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            model=Agent,
            device=device,
            gamma=args.gamma,
            policy_objective=args.policy_objective,
            action_cells=args.action_cells,
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
