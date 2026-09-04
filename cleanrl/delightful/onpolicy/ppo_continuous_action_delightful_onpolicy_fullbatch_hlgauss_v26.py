# Fully on-policy full-batch Delightful Gradient with HL-Gauss v26.
#
# Uses v14's 2,048-transition current-policy rollout, 128-step GAE, actor cadence,
# state-dependent sigma, and critic schedule. The sole algorithmic actor change
# is a detached DG gate applied to every fresh score term; the HL-Gauss decoder
# follows the shared transformed-coordinate contract. There is no replay,
# importance ratio, target actor, KL stop, or stale actor epoch.
# Hypothesis: if this fails against v14, DG weighting itself is the remaining
# cause rather than actor batch size, rollout horizon, cadence, or critic fit.

import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport, symexp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
INITIAL_LOG_STD = -1.5
RAW_LOG_STD_INIT = float(np.arctanh(2.0 * (INITIAL_LOG_STD - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) - 1.0))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    target_actor_batch_size: int = 2048
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    max_grad_norm: float = 0.5
    sigma_mode: Literal["state", "global"] = "state"
    dg_eta: float = 1.0
    dg_surprisal_clip: float = 10.0

    value_num_bins: int = 511
    value_min: float = -9.90353755128617
    value_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 2.0
    critic_mtp_horizon: int = 1

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    del gamma

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations with final observations at time limits."""
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs

    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")

    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    """Compute GAE with distinct bootstrap and reset-boundary semantics."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(
            truncations[t].bool(),
            truncation_bootstrap_values[t],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def build_mtp_targets(returns, boundaries, horizon):
    """Shift return targets without allowing an auxiliary head across a reset."""
    if returns.shape != boundaries.shape:
        raise ValueError("returns and boundaries must have the same shape")
    if returns.ndim != 2:
        raise ValueError("returns and boundaries must have shape (steps, envs)")
    if horizon < 1:
        raise ValueError("critic_mtp_horizon must be positive")

    targets = returns.new_zeros((*returns.shape, horizon))
    mask = torch.zeros((*returns.shape, horizon), dtype=torch.bool, device=returns.device)
    for offset in range(horizon):
        valid_steps = returns.shape[0] - offset
        if valid_steps <= 0:
            break
        valid = torch.ones((valid_steps, returns.shape[1]), dtype=torch.bool, device=returns.device)
        for transition_offset in range(offset):
            valid &= ~boundaries[transition_offset : transition_offset + valid_steps].bool()
        targets[:valid_steps, :, offset] = returns[offset : offset + valid_steps]
        mask[:valid_steps, :, offset] = valid
    return targets, mask


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def smooth_logstd_bound(raw_logstd):
    bounded = torch.tanh(raw_logstd)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (bounded + 1.0)


def transition_info_values(infos, key, num_envs):
    values = np.full(num_envs, np.nan, dtype=np.float32)
    if key in infos:
        current = np.asarray(infos[key], dtype=np.float32)
        mask = np.asarray(infos.get(f"_{key}", np.ones(num_envs, dtype=bool)), dtype=bool)
        values[mask] = current[mask]
    for index, final_info in enumerate(infos.get("final_info", ())):
        if final_info is not None and key in final_info:
            values[index] = final_info[key]
    return values


def finite_mean(values):
    finite = torch.isfinite(values)
    return values[finite].mean().item() if finite.any() else float("nan")


def delightful_gate(advantages, action_logprobs, eta=1.0, surprisal_clip=10.0):
    """Return the detached DG gate and its sampled-action diagnostics."""
    if eta <= 0:
        raise ValueError("dg_eta must be positive")
    if surprisal_clip <= 0:
        raise ValueError("dg_surprisal_clip must be positive")
    surprisal = (-action_logprobs.detach()).clamp(-surprisal_clip, surprisal_clip)
    delight = advantages.detach() * surprisal
    return torch.sigmoid(delight / eta), surprisal, delight


def parameter_norm(parameters):
    total = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        total += parameter.detach().float().square().sum()
    return total.sqrt()


class Agent(nn.Module):
    def __init__(self, envs, args, value_support=None):
        super().__init__()
        sigma_mode = args.sigma_mode
        if sigma_mode not in ("state", "global"):
            raise ValueError(f"unknown sigma mode: {sigma_mode!r}")
        self.sigma_mode = sigma_mode
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.value_num_bins = args.value_num_bins
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        # Consume exactly v12's scalar-head RNG before actor initialization so
        # seed-1 changes only the critic, not the initial policy.
        layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        if sigma_mode == "state":
            self.actor_logstd_head = layer_init(nn.Linear(64, action_dim), std=0.01, bias_const=RAW_LOG_STD_INIT)
            self.register_parameter("actor_logstd_param", None)
        else:
            self.actor_logstd_head = None
            self.actor_logstd_param = nn.Parameter(torch.full((1, action_dim), RAW_LOG_STD_INIT))
        self.critic_head = nn.Linear(
            64,
            args.critic_mtp_horizon * args.value_num_bins,
            bias=False,
        )
        nn.init.zeros_(self.critic_head.weight)
        if value_support is None:
            value_edges = torch.linspace(args.value_min, args.value_max, args.value_num_bins + 1)
            value_support = (value_edges[:-1] + value_edges[1:]) / 2.0
        if value_support.shape != (args.value_num_bins,):
            raise ValueError("value_support must contain one transformed coordinate per value bin")
        self.register_buffer("value_support", value_support.detach().clone())
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        if not torch.all(torch.isfinite(self.action_scale)) or not torch.all(self.action_scale > 0):
            raise ValueError("tanh Gaussian actions require finite nonzero bounds")

    def get_value_logits(self, observations):
        features = self.critic_trunk(observations)
        return self.critic_head(features).view(-1, self.critic_mtp_horizon, self.value_num_bins)

    def decode_value_logits(self, logits):
        transformed_value = (torch.softmax(logits, dim=-1) * self.value_support).sum(dim=-1)
        return symexp(transformed_value)

    def get_value(self, observations):
        return self.decode_value_logits(self.get_value_logits(observations)[:, 0]).unsqueeze(-1)

    def get_policy_parameters(self, observations):
        features = self.actor_trunk(observations)
        mean = self.actor_mean(features)
        if self.sigma_mode == "state":
            raw_logstd = self.actor_logstd_head(features)
        else:
            raw_logstd = self.actor_logstd_param.expand_as(mean)
        return mean, smooth_logstd_bound(raw_logstd)

    def action_logprobs(self, observations, raw_actions):
        mean, logstd = self.get_policy_parameters(observations)
        distribution = Normal(mean, logstd.exp())
        gaussian_logprob = distribution.log_prob(raw_actions).sum(dim=-1)
        log_tanh_jacobian = 2.0 * (np.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions))
        action_logprob = gaussian_logprob - (log_tanh_jacobian + torch.log(self.action_scale)).sum(dim=-1)
        return gaussian_logprob, action_logprob, mean, logstd

    def sample_action_and_value(self, observations):
        mean, logstd = self.get_policy_parameters(observations)
        raw_action = Normal(mean, logstd.exp()).sample()
        _, logprob, _, _ = self.action_logprobs(observations, raw_action)
        action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        return action, raw_action, logprob, self.get_value(observations)

    def get_action_and_value(self, observations, action=None, raw_action=None):
        if action is not None and raw_action is None:
            raise ValueError("raw_action is required when replaying a tanh action")
        if raw_action is None:
            mean, logstd = self.get_policy_parameters(observations)
            distribution = Normal(mean, logstd.exp())
            raw_action = distribution.sample()
        _, logprob, _, _ = self.action_logprobs(observations, raw_action)
        transformed = torch.tanh(raw_action) * self.action_scale + self.action_bias
        if action is None:
            action = transformed
        # A single-sample unbiased entropy estimate for the transformed policy.
        return action, logprob, -logprob.detach(), self.get_value(observations)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size != args.target_actor_batch_size:
        raise ValueError(f"fresh actor batch must be {args.target_actor_batch_size}, got {args.batch_size}")
    if args.batch_size % args.num_minibatches:
        raise ValueError("critic minibatches must divide the rollout batch")
    if args.dg_eta <= 0:
        raise ValueError("dg_eta must be positive")
    if args.dg_surprisal_clip <= 0:
        raise ValueError("dg_surprisal_clip must be positive")
    if args.value_num_bins < 3 or args.value_num_bins % 2 != 1:
        raise ValueError("value_num_bins must be an odd integer of at least three")
    if args.value_min >= 0 or args.value_max <= 0 or not np.isclose(-args.value_min, args.value_max):
        raise ValueError("value support must be symmetric around zero")
    if args.value_sigma_to_bin_ratio <= 0:
        raise ValueError("value_sigma_to_bin_ratio must be positive")
    if args.critic_mtp_horizon < 1:
        raise ValueError("critic_mtp_horizon must be positive")
    args.minibatch_size = args.batch_size // args.num_minibatches
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
    rows = "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{rows}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("this experiment requires CUDA")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name, args.gamma) for index in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    hlgauss_support = HLGaussSupport(
        args.value_num_bins,
        args.value_min,
        args.value_max,
        args.value_sigma_to_bin_ratio,
        device,
        use_symlog=True,
        support_is_edges=True,
    )
    agent = Agent(
        envs,
        args,
        value_support=hlgauss_support.support,
    ).to(device)
    sample_action_and_value = agent.sample_action_and_value
    action_logprobs = agent.action_logprobs
    value_function = agent.get_value
    value_logits_function = agent.get_value_logits
    if args.compile:
        sample_action_and_value = torch.compile(sample_action_and_value, mode=args.compile_mode, dynamic=False)
        action_logprobs = torch.compile(action_logprobs, mode=args.compile_mode, dynamic=False)
        value_function = torch.compile(value_function, mode=args.compile_mode, dynamic=False)
        value_logits_function = torch.compile(value_logits_function, mode=args.compile_mode, dynamic=False)
        print(f"compiled policy and value functions ({args.compile_mode})")

    actor_parameters = [parameter for name, parameter in agent.named_parameters() if name.startswith("actor_")]
    critic_parameters = list(agent.critic_trunk.parameters()) + list(agent.critic_head.parameters())
    actor_optimizer = optim.Adam(actor_parameters, lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic_parameters, lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    raw_actions = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    terminations_buffer = torch.zeros_like(logprobs)
    truncations_buffer = torch.zeros_like(logprobs)
    truncation_bootstrap_values = torch.zeros_like(logprobs)
    values = torch.zeros_like(logprobs)
    forward_speeds = torch.full_like(logprobs, torch.nan)
    control_costs = torch.full_like(logprobs, torch.nan)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            critic_optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, raw_action, logprob, value = sample_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            raw_actions[step] = raw_action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            forward_speeds[step] = torch.as_tensor(
                transition_info_values(infos, "x_velocity", args.num_envs),
                dtype=torch.float32,
                device=device,
            )
            control_costs[step] = torch.as_tensor(
                -transition_info_values(infos, "reward_ctrl", args.num_envs),
                dtype=torch.float32,
                device=device,
            )
            terminations_buffer[step] = torch.as_tensor(terminations, dtype=torch.float32, device=device)
            truncations_buffer[step] = torch.as_tensor(truncations, dtype=torch.float32, device=device)
            if np.any(truncations):
                final_obs_np = bootstrap_observations(next_obs_np, truncations, infos)
                final_obs = torch.as_tensor(final_obs_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    final_values = value_function(final_obs).flatten()
                truncation_mask = torch.as_tensor(truncations, dtype=torch.bool, device=device)
                truncation_bootstrap_values[step] = torch.where(
                    truncation_mask,
                    final_values,
                    torch.zeros_like(final_values),
                )
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            for info in infos.get("final_info", ()):
                if info and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            rollout_tail_value = value_function(next_obs).flatten()
            advantages, returns = compute_gae(
                rewards,
                values,
                terminations_buffer,
                truncations_buffer,
                truncation_bootstrap_values,
                rollout_tail_value,
                args.gamma,
                args.gae_lambda,
            )
            boundaries = torch.maximum(terminations_buffer, truncations_buffer)
            mtp_returns, mtp_mask = build_mtp_targets(
                returns,
                boundaries,
                args.critic_mtp_horizon,
            )
            mtp_target_probs = hlgauss_support.project(mtp_returns)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_raw_actions = raw_actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1).detach()
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mtp_target_probs = mtp_target_probs.reshape(
            -1,
            args.critic_mtp_horizon,
            args.value_num_bins,
        )
        b_mtp_mask = mtp_mask.reshape(-1, args.critic_mtp_horizon)

        # One score-function update on exactly the actions sampled by this
        # rollout. The DG gate is detached: it reweights the unbiased policy
        # score rather than adding a pathwise gradient through the gate.
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        gaussian_logprob, action_logprob, old_mean_graph, old_logstd_graph = action_logprobs(
            b_obs,
            b_raw_actions,
        )
        # Compiled reduce-overhead outputs alias reusable CUDA-graph buffers;
        # this diagnostic survives the post-update policy invocation below.
        action_logprob = action_logprob.clone()
        old_mean = old_mean_graph.detach().clone()
        old_logstd = old_logstd_graph.detach().clone()
        old_std = old_logstd.exp()
        gate, surprisal, delight = delightful_gate(
            b_advantages,
            action_logprob,
            eta=args.dg_eta,
            surprisal_clip=args.dg_surprisal_clip,
        )
        actor_weights = (gate * b_advantages).detach()
        actor_loss = -(actor_weights * gaussian_logprob).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm)
        actor_clip_scale = min(1.0, args.max_grad_norm / max(actor_grad_norm.item(), 1e-30))
        old_parameters = [parameter.detach().clone() for parameter in actor_parameters]
        actor_optimizer.step()

        # The tanh transform and action rescaling are unchanged bijections, so
        # latent Gaussian KL is also the exact transformed-action KL.
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            _, post_action_logprob, post_mean, post_logstd = action_logprobs(
                b_obs,
                b_raw_actions,
            )
            post_action_logprob = post_action_logprob.clone()
            post_mean = post_mean.clone()
            post_logstd = post_logstd.clone()
            exact_kl = kl_divergence(
                Normal(old_mean, old_std),
                Normal(post_mean, post_logstd.exp()),
            ).sum(dim=-1)

        post_logratio = post_action_logprob - b_logprobs
        post_ratio = post_logratio.exp()
        sampled_kl = ((post_ratio - 1.0) - post_logratio).mean()
        actor_delta_squared = torch.zeros((), device=device)
        for parameter, old_parameter in zip(actor_parameters, old_parameters, strict=True):
            actor_delta_squared += (parameter.detach().float() - old_parameter.float()).square().sum()
        actor_delta_norm = actor_delta_squared.sqrt()
        actor_parameter_norm = parameter_norm(actor_parameters)

        # The critic retains v10's schedule and sees only raw native GAE returns.
        # Actor and critic optimizers remain completely separate.
        b_inds = np.arange(args.batch_size)
        critic_losses = []
        critic_cross_entropies = []
        critic_grad_norms = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                value_logits = value_logits_function(b_obs[mb_inds])
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                value_cross_entropy = -(b_mtp_target_probs[mb_inds] * value_log_probs).sum(dim=-1)
                value_mask = b_mtp_mask[mb_inds].to(dtype=value_cross_entropy.dtype)
                critic_loss = (value_cross_entropy * value_mask).sum(dim=-1).mean()
                critic_cross_entropy = (value_cross_entropy * value_mask).sum() / value_mask.sum().clamp_min(1.0)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm)
                critic_optimizer.step()
                critic_losses.append(critic_loss.item())
                critic_cross_entropies.append(critic_cross_entropy.item())
                critic_grad_norms.append(critic_grad_norm.item())

        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            post_value_logits = value_logits_function(b_obs)
            post_values = agent.decode_value_logits(post_value_logits[:, 0])
            post_value_probs = torch.softmax(post_value_logits, dim=-1)

        value_errors = b_values - b_returns
        value_mse = value_errors.square().mean()
        value_rmse = value_mse.sqrt()
        return_std = b_returns.std(unbiased=False)
        value_normalized_rmse = value_rmse / return_std.clamp_min(1e-8)
        post_value_mse = (post_values - b_returns).square().mean()
        post_value_rmse = post_value_mse.sqrt()
        post_value_normalized_rmse = post_value_rmse / return_std.clamp_min(1e-8)

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        target_variance = np.var(y_true)
        explained_variance = np.nan if target_variance == 0 else 1.0 - np.var(y_true - y_pred) / target_variance
        post_y_pred = post_values.cpu().numpy()
        post_explained_variance = np.nan if target_variance == 0 else 1.0 - np.var(y_true - post_y_pred) / target_variance
        target_edge_mass = (
            (b_mtp_target_probs[..., 0] + b_mtp_target_probs[..., -1]) * b_mtp_mask.to(dtype=b_mtp_target_probs.dtype)
        ).sum() / b_mtp_mask.sum().clamp_min(1)
        predicted_edge_mass = (
            (post_value_probs[..., 0] + post_value_probs[..., -1]) * b_mtp_mask.to(dtype=post_value_probs.dtype)
        ).sum() / b_mtp_mask.sum().clamp_min(1)
        normalized_actions = (b_actions - agent.action_bias) / agent.action_scale
        per_env_kl = exact_kl.reshape(args.num_steps, args.num_envs).mean(dim=0)
        current_lr = actor_optimizer.param_groups[0]["lr"]
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/actor", actor_loss.item(), global_step)
        writer.add_scalar("losses/value_hlgauss_total_ce", np.mean(critic_losses), global_step)
        writer.add_scalar("losses/value_hlgauss_ce", np.mean(critic_cross_entropies), global_step)
        writer.add_scalar("losses/value_mse", value_mse.item(), global_step)
        writer.add_scalar("losses/value_rmse", value_rmse.item(), global_step)
        writer.add_scalar("losses/value_normalized_rmse", value_normalized_rmse.item(), global_step)
        writer.add_scalar("losses/value_post_update_mse", post_value_mse.item(), global_step)
        writer.add_scalar("losses/value_post_update_rmse", post_value_rmse.item(), global_step)
        writer.add_scalar(
            "losses/value_post_update_normalized_rmse",
            post_value_normalized_rmse.item(),
            global_step,
        )
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("losses/post_update_explained_variance", post_explained_variance, global_step)
        writer.add_scalar("losses/critic_grad_norm", np.mean(critic_grad_norms), global_step)
        writer.add_scalar("value/return_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("value/return_std", return_std.item(), global_step)
        writer.add_scalar("value/return_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("value/prediction_mean", b_values.mean().item(), global_step)
        writer.add_scalar("value/prediction_std", b_values.std(unbiased=False).item(), global_step)
        writer.add_scalar("value/target_edge_mass", target_edge_mass.item(), global_step)
        writer.add_scalar("value/predicted_edge_mass", predicted_edge_mass.item(), global_step)
        writer.add_scalar("native/reward_mean", rewards.mean().item(), global_step)
        writer.add_scalar("native/reward_std", rewards.std().item(), global_step)
        writer.add_scalar("native/forward_speed_mean", finite_mean(forward_speeds), global_step)
        writer.add_scalar("native/control_cost_mean", finite_mean(control_costs), global_step)
        writer.add_scalar("advantage/mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("advantage/rms", b_advantages.square().mean().sqrt().item(), global_step)
        writer.add_scalar("advantage/min", b_advantages.min().item(), global_step)
        writer.add_scalar("advantage/max", b_advantages.max().item(), global_step)
        writer.add_scalar("dg/gate_mean", gate.mean().item(), global_step)
        writer.add_scalar("dg/gate_std", gate.std(unbiased=False).item(), global_step)
        writer.add_scalar("dg/gate_min", gate.min().item(), global_step)
        writer.add_scalar("dg/gate_max", gate.max().item(), global_step)
        writer.add_scalar("dg/surprisal_mean", surprisal.mean().item(), global_step)
        writer.add_scalar(
            "dg/surprisal_clip_fraction",
            ((-action_logprob.detach()).abs() > args.dg_surprisal_clip).float().mean().item(),
            global_step,
        )
        writer.add_scalar("dg/delight_mean", delight.mean().item(), global_step)
        writer.add_scalar("dg/delight_rms", delight.square().mean().sqrt().item(), global_step)
        writer.add_scalar("dg/actor_weight_rms", actor_weights.square().mean().sqrt().item(), global_step)
        truncation_mask = truncations_buffer.bool()
        writer.add_scalar(
            "bootstrap/truncation_count",
            truncation_mask.sum().item(),
            global_step,
        )
        if truncation_mask.any():
            writer.add_scalar(
                "bootstrap/truncation_value_mean",
                truncation_bootstrap_values[truncation_mask].mean().item(),
                global_step,
            )
            writer.add_scalar(
                "advantage/truncation_rms",
                advantages[truncation_mask].square().mean().sqrt().item(),
                global_step,
            )

        writer.add_scalar("action/logstd_mean", post_logstd.mean().item(), global_step)
        writer.add_scalar("action/logstd_std", post_logstd.std().item(), global_step)
        writer.add_scalar(
            "action/logstd_cross_state_std",
            post_logstd.std(dim=0).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "action/logstd_cross_action_std",
            post_logstd.mean(dim=0).std().item(),
            global_step,
        )
        writer.add_scalar("action/logstd_min", post_logstd.min().item(), global_step)
        writer.add_scalar("action/logstd_max", post_logstd.max().item(), global_step)
        for quantile, name in ((0.01, "p01"), (0.5, "p50"), (0.99, "p99")):
            writer.add_scalar(f"action/logstd_{name}", torch.quantile(post_logstd, quantile).item(), global_step)
        writer.add_scalar(
            "action/logstd_lower_bound_fraction",
            (post_logstd < LOG_STD_MIN + 0.05).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "action/logstd_upper_bound_fraction",
            (post_logstd > LOG_STD_MAX - 0.05).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "action/saturation_fraction",
            (normalized_actions.abs() > 0.95).float().mean().item(),
            global_step,
        )

        writer.add_scalar("grad/actor_preclip_norm", actor_grad_norm.item(), global_step)
        writer.add_scalar("grad/actor_clip_scale", actor_clip_scale, global_step)
        writer.add_scalar("grad/critic_preclip_mean", np.mean(critic_grad_norms), global_step)
        writer.add_scalar("update/actor_delta_norm", actor_delta_norm.item(), global_step)
        writer.add_scalar(
            "update/actor_relative_delta_norm",
            (actor_delta_norm / actor_parameter_norm.clamp_min(1e-30)).item(),
            global_step,
        )
        writer.add_scalar("realized/exact_tanh_kl_mean", exact_kl.mean().item(), global_step)
        writer.add_scalar(
            "realized/exact_tanh_kl_p50",
            torch.quantile(exact_kl, 0.5).item(),
            global_step,
        )
        writer.add_scalar(
            "realized/exact_tanh_kl_p90",
            torch.quantile(exact_kl, 0.9).item(),
            global_step,
        )
        writer.add_scalar(
            "realized/exact_tanh_kl_p99",
            torch.quantile(exact_kl, 0.99).item(),
            global_step,
        )
        writer.add_scalar("realized/exact_tanh_kl_max", exact_kl.max().item(), global_step)
        writer.add_scalar("realized/per_env_kl_max", per_env_kl.max().item(), global_step)
        writer.add_scalar("realized/sampled_approx_kl", sampled_kl.item(), global_step)
        writer.add_scalar(
            "realized/ratio_clipfrac",
            ((post_ratio - 1.0).abs() > 0.2).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "realized/logratio_abs_max",
            post_logratio.abs().max().item(),
            global_step,
        )
        print("SPS:", sps)

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
            Model=partial(Agent, args=args),
            device=device,
            gamma=args.gamma,
        )
        for index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, index)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "FullyOnPolicyFullBatchDelightfulGradient",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
