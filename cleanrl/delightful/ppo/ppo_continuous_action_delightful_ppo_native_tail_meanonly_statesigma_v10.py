# Delightful PPO with native rewards and mean-only chi-square tail gating v10.
#
# A detached, affine-invariant latent Gaussian tail surprisal gates only the
# policy-mean score; the scale score retains a neutral half-weighted advantage.
# This is a novel scale-invariant native-objective DG variant, not paper parity.

import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
INITIAL_LOG_STD = -1.5
RAW_LOG_STD_INIT = float(
    np.arctanh(
        2.0 * (INITIAL_LOG_STD - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) - 1.0
    )
)
DEFAULT_LAYER_STD = np.sqrt(2.0)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """CUDA is required for training"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    compile: bool = False
    """compile the action and value functions with torch.compile"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """the user or org name of the model repository"""

    # Algorithm-specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiment"""
    learning_rate: float = 3e-4
    """the actor and critic learning rate"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps per environment in each fresh actor rollout"""
    target_actor_batch_size: int = 2048
    """required fresh full actor batch size"""
    anneal_lr: bool = False
    """toggle learning-rate annealing; disabled by default"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for generalized advantage estimation"""
    num_minibatches: int = 32
    """the number of critic minibatches; the actor always uses the full batch"""
    update_epochs: int = 10
    """the number of critic epochs; the actor always takes one step"""
    clip_coef: float = 0.2
    """diagnostic threshold for the realized sampled likelihood ratio"""
    max_grad_norm: float = 0.5
    """the maximum actor and critic gradient norm"""
    actor_weighting: Literal["tail-dg", "neutral"] = "tail-dg"
    """mean-score weighting; neutral is the matched half-weight control"""

    # Filled at runtime.
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    del gamma  # Kept in the signature for compatibility with CleanRL's evaluator.

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


def transition_info_values(infos, key, num_envs):
    """Extract a MuJoCo transition metric, including auto-reset final info."""
    values = np.full(num_envs, np.nan, dtype=np.float32)
    if key in infos:
        current = np.asarray(infos[key], dtype=np.float32)
        mask = np.asarray(
            infos.get(f"_{key}", np.ones(num_envs, dtype=bool)), dtype=bool
        )
        values[mask] = current[mask]
    for index, final_info in enumerate(infos.get("final_info", ())):
        if final_info is not None and key in final_info:
            values[index] = final_info[key]
    return values


def finite_mean(values):
    finite = torch.isfinite(values)
    if not finite.any():
        return float("nan")
    return values[finite].mean().item()


def masked_mean(values, mask):
    if not mask.any():
        return torch.full((), torch.nan, dtype=values.dtype, device=values.device)
    return values[mask].mean()


def pearson_correlation(x, y):
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.square().sum().sqrt() * y_centered.square().sum().sqrt()
    if denominator.item() == 0.0:
        return torch.full((), torch.nan, dtype=x.dtype, device=x.device)
    return (x_centered * y_centered).sum() / denominator


def parameter_grad_norm(parameters):
    squared_norm = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum()
    return squared_norm.sqrt()


def layer_init(layer, std=DEFAULT_LAYER_STD, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def latent_tail_statistics(raw_action, mean, std):
    """Return detached q and -log chi-square survival probability."""
    standardized = ((raw_action - mean) / std).detach()
    q = standardized.square().sum(dim=-1)
    # Float64 materially extends the survival-function range. The tiny floor is
    # solely an underflow guard, not an algorithmic surprisal cap.
    q64 = q.double()
    half_dof = torch.full_like(q64, 0.5 * raw_action.shape[-1])
    survival = torch.special.gammaincc(half_dof, 0.5 * q64)
    rho = -torch.log(survival.clamp_min(torch.finfo(survival.dtype).tiny))
    return q, rho.to(q.dtype)


def native_score_weights(advantages, rho, actor_weighting):
    """Build detached mean/scale weights and DG diagnostics."""
    advantages = advantages.detach()
    rho = rho.detach()
    delight = advantages * rho
    eta = delight.abs().mean().detach()
    if eta.item() == 0.0:
        normalized_delight = torch.zeros_like(delight)
        candidate_gate = torch.full_like(delight, 0.5)
    else:
        normalized_delight = delight / eta
        candidate_gate = torch.sigmoid(normalized_delight)

    neutral_weight = 0.5 * advantages
    if actor_weighting == "tail-dg":
        applied_mean_gate = candidate_gate
    elif actor_weighting == "neutral":
        applied_mean_gate = torch.full_like(candidate_gate, 0.5)
    else:
        raise ValueError(f"unknown actor weighting: {actor_weighting!r}")
    mean_weight = advantages * applied_mean_gate
    return (
        mean_weight.detach(),
        neutral_weight.detach(),
        applied_mean_gate,
        candidate_gate,
        delight,
        eta,
        normalized_delight,
    )


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = layer_init(
            nn.Linear(64, action_dim), std=0.01, bias_const=RAW_LOG_STD_INIT
        )
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
        if not torch.all(torch.isfinite(self.action_scale)) or not torch.all(
            self.action_scale > 0.0
        ):
            raise ValueError(
                "tanh Gaussian actions require finite, non-degenerate action bounds"
            )

    def get_value(self, x):
        return self.critic(x)

    def get_policy_parameters(self, x):
        features = self.actor_trunk(x)
        mean = self.actor_mean(features)
        bounded = torch.tanh(self.actor_logstd(features))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (bounded + 1.0)
        return mean, logstd

    def get_action_distribution(self, x):
        mean, logstd = self.get_policy_parameters(x)
        return Normal(mean, logstd.exp()), logstd

    def _action_and_logprob_from_raw(self, distribution, raw_action):
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias
        gaussian_logprob = distribution.log_prob(raw_action).sum(dim=-1)
        log_tanh_jacobian = 2.0 * (
            np.log(2.0)
            - raw_action
            - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        logprob = gaussian_logprob - (
            log_tanh_jacobian + torch.log(self.action_scale)
        ).sum(dim=-1)
        return action, logprob, gaussian_logprob

    def sample_action_and_value(self, x):
        distribution, _ = self.get_action_distribution(x)
        raw_action = distribution.rsample()
        action, logprob, gaussian_logprob = self._action_and_logprob_from_raw(
            distribution, raw_action
        )
        return action, raw_action, logprob, gaussian_logprob, self.critic(x)

    def get_action_and_value(self, x, action=None, raw_action=None):
        """CleanRL-compatible action interface, including stored raw actions."""
        if action is None and raw_action is None:
            distribution, _ = self.get_action_distribution(x)
            sampled_raw_action = distribution.sample()
            action, logprob, _ = self._action_and_logprob_from_raw(
                distribution, sampled_raw_action
            )
            value = self.critic(x)
            return action, logprob, -logprob.detach(), value
        if raw_action is None:
            raise ValueError(
                "raw_action is required when evaluating a stored tanh action"
            )
        distribution, _ = self.get_action_distribution(x)
        transformed_action, logprob, _ = self._action_and_logprob_from_raw(
            distribution, raw_action
        )
        if action is None:
            action = transformed_action
        return action, logprob, -logprob.detach(), self.critic(x)

    def exact_action_logprob(self, x, raw_action):
        mean, logstd = self.get_policy_parameters(x)
        distribution = Normal(mean, logstd.exp())
        _, logprob, _ = self._action_and_logprob_from_raw(distribution, raw_action)
        return logprob, mean, logstd

    def decomposed_score_logprobs(self, x, raw_action):
        mean, logstd = self.get_policy_parameters(x)
        std = logstd.exp()
        mean_score_logprob = Normal(mean, std.detach()).log_prob(raw_action).sum(dim=-1)
        scale_score_logprob = (
            Normal(mean.detach(), std).log_prob(raw_action).sum(dim=-1)
        )
        return mean_score_logprob, scale_score_logprob, mean, logstd


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size != args.target_actor_batch_size:
        raise ValueError(
            "this method requires a fresh actor batch of "
            f"{args.target_actor_batch_size} transitions, got "
            f"num_envs * num_steps = {args.batch_size}"
        )
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("critic minibatches must divide the rollout batch exactly")
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
    hyperparameter_rows = "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()
    )
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hyperparameter_rows}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        raise ValueError("this experiment requires CUDA; --no-cuda is unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action spaces are supported"
    )

    agent = Agent(envs).to(device)
    sample_action_and_value = agent.sample_action_and_value
    exact_action_logprob = agent.exact_action_logprob
    decomposed_score_logprobs = agent.decomposed_score_logprobs
    value_function = agent.get_value
    if args.compile:
        sample_action_and_value = torch.compile(
            sample_action_and_value, mode=args.compile_mode, dynamic=False
        )
        exact_action_logprob = torch.compile(
            exact_action_logprob, mode=args.compile_mode, dynamic=False
        )
        decomposed_score_logprobs = torch.compile(
            decomposed_score_logprobs, mode=args.compile_mode, dynamic=False
        )
        value_function = torch.compile(
            value_function, mode=args.compile_mode, dynamic=False
        )
        print(
            "compiled action, score, and value functions "
            f"(mode={args.compile_mode!r}, dynamic=False)"
        )

    actor_trunk_parameters = list(agent.actor_trunk.parameters())
    actor_mean_parameters = list(agent.actor_mean.parameters())
    actor_logstd_parameters = list(agent.actor_logstd.parameters())
    actor_parameters = (
        actor_trunk_parameters + actor_mean_parameters + actor_logstd_parameters
    )
    critic_parameters = list(agent.critic.parameters())
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
    forward_speeds = torch.full_like(logprobs, torch.nan)
    control_costs = torch.full_like(logprobs, torch.nan)
    dones = torch.zeros_like(logprobs)
    values = torch.zeros_like(logprobs)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            critic_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, raw_action, logprob, _, value = (
                    sample_action_and_value(next_obs)
                )
                values[step] = value.flatten()
            actions[step] = action
            raw_actions[step] = raw_action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            )
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
            next_done_np = np.logical_or(terminations, truncations)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            "global_step="
                            f"{global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            next_value = value_function(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_raw_actions = raw_actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # One direct, fresh, full-batch score-function actor step. The mean and
        # scale paths deliberately stop gradients through the other parameter.
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        mean_score_logprob, scale_score_logprob, old_mean, old_logstd = (
            decomposed_score_logprobs(b_obs, b_raw_actions)
        )
        # Compiled reduce-overhead calls may return views into reusable CUDA
        # graph output buffers. The post-update policy call below would
        # otherwise overwrite the behavior-policy parameters used by the gate
        # and exact KL diagnostic.
        old_mean = old_mean.detach().clone()
        old_logstd = old_logstd.detach().clone()
        old_std = old_logstd.exp()
        q, rho = latent_tail_statistics(b_raw_actions, old_mean, old_std)
        (
            mean_weight,
            scale_weight,
            gate,
            candidate_gate,
            delight,
            eta,
            normalized_delight,
        ) = native_score_weights(b_advantages, rho, args.actor_weighting)

        mean_loss = -(mean_weight * mean_score_logprob).mean()
        scale_loss = -(scale_weight * scale_score_logprob).mean()
        actor_loss = mean_loss + scale_loss

        latent_scale_score = (
            ((b_raw_actions - old_mean) / old_std).square() - 1.0
        ).sum(dim=-1)
        gated_sigma_score = b_advantages * candidate_gate * latent_scale_score
        applied_sigma_score = scale_weight * latent_scale_score

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_trunk_grad_norm = parameter_grad_norm(actor_trunk_parameters)
        actor_mean_grad_norm = parameter_grad_norm(actor_mean_parameters)
        actor_logstd_grad_norm = parameter_grad_norm(actor_logstd_parameters)
        actor_grad_norm = nn.utils.clip_grad_norm_(
            actor_parameters, args.max_grad_norm
        )
        actor_optimizer.step()

        # The tanh transform and action rescaling are the same bijection before
        # and after the update, so latent Gaussian KL is the exact action KL.
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            post_logprob, post_mean, post_logstd = exact_action_logprob(
                b_obs, b_raw_actions
            )
            # This diagnostic survives hundreds of subsequent compiled critic
            # calls, so it also needs storage independent of CUDA-graph outputs.
            post_logstd = post_logstd.clone()
            old_distribution = Normal(old_mean, old_std)
            post_distribution = Normal(post_mean, post_logstd.exp())
            realized_exact_tanh_kl = kl_divergence(
                old_distribution, post_distribution
            ).sum(dim=-1)
            post_logratio = post_logprob - b_logprobs
            post_ratio = post_logratio.exp()
            realized_sample_kl = ((post_ratio - 1.0) - post_logratio).mean()
            realized_clipfrac = (
                (post_ratio - 1.0).abs() > args.clip_coef
            ).float().mean()

        # The critic alone receives 10 x 32 minibatch regression updates. Its
        # target and inputs come directly from the unmodified Gym reward.
        b_inds = np.arange(args.batch_size)
        critic_losses = []
        critic_grad_norms = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                newvalue = value_function(b_obs[mb_inds]).view(-1)
                critic_loss = 0.5 * torch.nn.functional.mse_loss(
                    newvalue, b_returns[mb_inds]
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    critic_parameters, args.max_grad_norm
                )
                critic_optimizer.step()
                critic_losses.append(critic_loss.item())
                critic_grad_norms.append(critic_grad_norm.item())

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0.0 else 1.0 - np.var(y_true - y_pred) / var_y
        )

        advantage_positive = b_advantages > 0.0
        advantage_negative = b_advantages < 0.0
        radial_outer = q > b_raw_actions.shape[-1]
        radial_inner = ~radial_outer
        per_sample_logstd = old_logstd.mean(dim=-1)
        rho_logstd_correlation = pearson_correlation(rho, per_sample_logstd)
        normalized_actions = (b_actions - agent.action_bias) / agent.action_scale
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar(
            "charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/actor_total", actor_loss.item(), global_step)
        writer.add_scalar("losses/actor_mean_score", mean_loss.item(), global_step)
        writer.add_scalar("losses/actor_scale_score", scale_loss.item(), global_step)
        writer.add_scalar("losses/value_mse", np.mean(critic_losses), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        writer.add_scalar("native/reward_mean", rewards.mean().item(), global_step)
        writer.add_scalar("native/reward_std", rewards.std().item(), global_step)
        writer.add_scalar("native/reward_min", rewards.min().item(), global_step)
        writer.add_scalar("native/reward_max", rewards.max().item(), global_step)
        writer.add_scalar(
            "native/forward_speed_mean", finite_mean(forward_speeds), global_step
        )
        writer.add_scalar(
            "native/control_cost_mean", finite_mean(control_costs), global_step
        )

        writer.add_scalar(
            "advantage/mean", b_advantages.mean().item(), global_step
        )
        writer.add_scalar("advantage/std", b_advantages.std().item(), global_step)
        writer.add_scalar(
            "advantage/rms", b_advantages.square().mean().sqrt().item(), global_step
        )
        writer.add_scalar("advantage/min", b_advantages.min().item(), global_step)
        writer.add_scalar("advantage/max", b_advantages.max().item(), global_step)
        writer.add_scalar(
            "advantage/positive_fraction",
            advantage_positive.float().mean().item(),
            global_step,
        )

        for name, statistic in (("q", q), ("rho", rho)):
            writer.add_scalar(f"tail/{name}_mean", statistic.mean().item(), global_step)
            writer.add_scalar(f"tail/{name}_std", statistic.std().item(), global_step)
            writer.add_scalar(
                f"tail/{name}_rms", statistic.square().mean().sqrt().item(), global_step
            )
            writer.add_scalar(f"tail/{name}_min", statistic.min().item(), global_step)
            writer.add_scalar(f"tail/{name}_max", statistic.max().item(), global_step)
        writer.add_scalar("tail/eta_mean_abs_delight", eta.item(), global_step)
        writer.add_scalar(
            "tail/normalized_delight_mean",
            normalized_delight.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "tail/normalized_delight_std", normalized_delight.std().item(), global_step
        )
        writer.add_scalar(
            "tail/normalized_delight_rms",
            normalized_delight.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "tail/normalized_delight_abs_mean",
            normalized_delight.abs().mean().item(),
            global_step,
        )
        writer.add_scalar("tail/delight_mean", delight.mean().item(), global_step)
        writer.add_scalar("tail/gate_mean", gate.mean().item(), global_step)
        writer.add_scalar(
            "tail/gate_low_saturation", (gate < 0.01).float().mean().item(), global_step
        )
        writer.add_scalar(
            "tail/gate_high_saturation",
            (gate > 0.99).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_mean", candidate_gate.mean().item(), global_step
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_saturation",
            ((candidate_gate < 0.01) | (candidate_gate > 0.99)).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_positive_advantage_mean",
            masked_mean(candidate_gate, advantage_positive).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_negative_advantage_mean",
            masked_mean(candidate_gate, advantage_negative).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_inner_radius_mean",
            masked_mean(candidate_gate, radial_inner).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/candidate_dg_gate_outer_radius_mean",
            masked_mean(candidate_gate, radial_outer).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/gate_positive_advantage_mean",
            masked_mean(gate, advantage_positive).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/gate_negative_advantage_mean",
            masked_mean(gate, advantage_negative).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/gate_inner_radius_mean",
            masked_mean(gate, radial_inner).item(),
            global_step,
        )
        writer.add_scalar(
            "tail/gate_outer_radius_mean",
            masked_mean(gate, radial_outer).item(),
            global_step,
        )
        for sign_name, sign_mask in (
            ("positive", advantage_positive),
            ("negative", advantage_negative),
        ):
            for radius_name, radius_mask in (
                ("inner", radial_inner),
                ("outer", radial_outer),
            ):
                writer.add_scalar(
                    f"tail/gate_{sign_name}_{radius_name}_mean",
                    masked_mean(gate, sign_mask & radius_mask).item(),
                    global_step,
                )
        writer.add_scalar(
            "tail/rho_logstd_correlation", rho_logstd_correlation.item(), global_step
        )

        writer.add_scalar(
            "action/logstd_mean", post_logstd.mean().item(), global_step
        )
        writer.add_scalar("action/logstd_std", post_logstd.std().item(), global_step)
        writer.add_scalar("action/logstd_min", post_logstd.min().item(), global_step)
        writer.add_scalar("action/logstd_max", post_logstd.max().item(), global_step)
        writer.add_scalar(
            "action/saturation_fraction",
            (normalized_actions.abs() > 0.95).float().mean().item(),
            global_step,
        )

        writer.add_scalar("grad/actor_total", actor_grad_norm.item(), global_step)
        writer.add_scalar(
            "grad/actor_trunk", actor_trunk_grad_norm.item(), global_step
        )
        writer.add_scalar(
            "grad/actor_mean_head", actor_mean_grad_norm.item(), global_step
        )
        writer.add_scalar(
            "grad/actor_logstd_head", actor_logstd_grad_norm.item(), global_step
        )
        writer.add_scalar(
            "grad/critic_mean", np.mean(critic_grad_norms), global_step
        )

        for name, statistic in (
            ("gated", gated_sigma_score),
            ("applied", applied_sigma_score),
        ):
            writer.add_scalar(
                f"sigma_score/{name}_mean", statistic.mean().item(), global_step
            )
            writer.add_scalar(
                f"sigma_score/{name}_std", statistic.std().item(), global_step
            )
            writer.add_scalar(
                f"sigma_score/{name}_rms",
                statistic.square().mean().sqrt().item(),
                global_step,
            )

        writer.add_scalar(
            "realized/exact_tanh_kl_mean",
            realized_exact_tanh_kl.mean().item(),
            global_step,
        )
        writer.add_scalar(
            "realized/exact_tanh_kl_max",
            realized_exact_tanh_kl.max().item(),
            global_step,
        )
        writer.add_scalar(
            "realized/sampled_approx_kl", realized_sample_kl.item(), global_step
        )
        writer.add_scalar(
            "realized/clipfrac", realized_clipfrac.item(), global_step
        )
        writer.add_scalar(
            "realized/logratio_abs_max", post_logratio.abs().max().item(), global_step
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
            Model=Agent,
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
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
