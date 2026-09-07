# HL-Gauss integration study on the exact 32x-LR sphere PPO baseline.
# Paper: arxiv.org/abs/2403.03950, Appendix A (uniform edges, raw Gaussian CE).
# Decouple actor/critic clipping and LR; rescale unit-L2 critic features to RMS1.
# Optional Dreamer percentile scaling acts ONLY on advantages, never value targets.
# Frozen support and labels; normalized and raw reward experiments use explicit
# ranges. This is a PPO adaptation, not Dreamer or a claim of optimal settings.
import os
import random
import time
from argparse import Namespace
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HistogramGaussian, HLGaussConfig
from cleanrl.shared.host_actor import make_situ_sphere_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache,
    device_minibatches,
    explained_variance,
    gather_metrics,
    get_gae_fn,
)
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import (
    VectorObsNorm,
    VectorRewardNorm,
    make_raw_continuous_env,
)

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
# Frozen version: do not silently inherit subsequent shared-default changes.
DEFAULT_VALUE_CONFIG = HLGaussConfig(
    v_min=-10.0,
    v_max=10.0,
    num_bins=201,
    sigma_ratio=0.75,
    transform="linear",
    bin_type="edges",
    decode="scalar",
)


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
    wandb_entity: str | None = None
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
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 9.6e-3
    """actor learning rate; baseline 32x the 3e-4 PPO default"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """raw GAE in the surrogate; no minibatch standardization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    critic_learning_rate: float = 3e-3
    """independent categorical critic learning rate"""
    critic_feature_scale: float = 8.0
    """unit-L2 width64 features times8 have RMS1; actor is unchanged"""
    reward_norm: bool = True
    """normalize/clip rewards; false leaves environment rewards raw"""
    actor_return_scale: bool = False
    """Dreamer-style percentile EMA denominator for advantages only"""

    value: HLGaussConfig = DEFAULT_VALUE_CONFIG
    """fixed support in the chosen reward units; explicitly set range for raw runs"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads, capped at num_envs"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class _EvaluationNorm(gym.Wrapper):
    """Single-env evaluation adapter using the same shared normalization math."""

    def __init__(self, env, gamma, reward_norm=True):
        super().__init__(env)
        self.obs_norm = VectorObsNorm(1, env.observation_space.shape)
        self.rew_norm = VectorRewardNorm(1, gamma) if reward_norm else None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.obs_norm.normalize(np.asarray(observation)[None])[0], info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.obs_norm.normalize(np.asarray(observation)[None])[0]
        if self.rew_norm is not None:
            reward = self.rew_norm.normalize(np.asarray([reward]), np.asarray([terminated]))[0]
        return observation, float(reward), terminated, truncated, info


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=True):
    raw_factory = make_raw_continuous_env(env_id, idx, capture_video, run_name)

    def thunk():
        return _EvaluationNorm(raw_factory(), gamma, reward_norm)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PercentileReturnScale(nn.Module):
    """Dreamer's zero-initialized, non-debiased 5/95 percentile EMA."""

    percentiles: torch.Tensor
    bounds: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("percentiles", torch.tensor([0.05, 0.95]))
        self.register_buffer("bounds", torch.zeros(2))

    @torch.no_grad()
    def forward(self, returns):
        self.bounds.lerp_(torch.quantile(returns.detach().flatten(), self.percentiles), 0.01)
        return (self.bounds[1] - self.bounds[0]).clamp_min(1.0)


class ScaledValueHead(nn.Linear):
    """Control the input scale of the classifier without changing sphere geometry."""

    def __init__(self, in_features, out_features, feature_scale):
        super().__init__(in_features, out_features)
        self.feature_scale = feature_scale

    def forward(self, input):
        return F.linear(input * self.feature_scale, self.weight, self.bias)


def make_optimizer(agent, args):
    return optim.Adam(
        [
            {"params": list(agent.actor.parameters()), "lr": args.learning_rate},
            {"params": list(agent.critic.parameters()), "lr": args.critic_learning_rate},
        ],
        eps=1e-5,
        fused=True,
    )


def optimizer_step(loss, optimizer, max_grad_norm):
    """Independent clip budgets prevent actor reward units from clipping CE gradients."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_norm = nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_grad_norm)
    critic_norm = nn.utils.clip_grad_norm_(optimizer.param_groups[1]["params"], max_grad_norm)
    optimizer.step()
    return torch.stack((actor_norm.detach(), critic_norm.detach()))


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor

    def __init__(self, envs, value_config: HLGaussConfig | None = None, critic_feature_scale=8.0):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        # Consume exactly the baseline scalar critic's RNG before initializing
        # the actor; categorical head size must not change the initial policy.
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(observation_dim, 64),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        if value_config is None:
            value_config = DEFAULT_VALUE_CONFIG
        self.value_support = HistogramGaussian(value_config)
        value_head = ScaledValueHead(64, value_config.num_bins, critic_feature_scale)
        # Uniform logits are uninformative and decode near zero on default support.
        nn.init.zeros_(value_head.weight)
        nn.init.zeros_(value_head.bias)
        self.critic[-1] = value_head

    def get_value(self, x):
        return self.value_support.to_scalar(self.critic(x)).unsqueeze(-1)

    def get_policy_and_value_logits(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta, logits = self.get_policy_and_value_logits(x)
        return alpha, beta, self.value_support.to_scalar(logits).unsqueeze(-1)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, target_probs, args):
    """Clipped native-action PPO plus CE against once-per-rollout HL-Gauss labels."""
    alpha, beta, value_logits = agent.get_policy_and_value_logits(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    v_loss = -(target_probs.detach() * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac)
    )
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if not np.isfinite(args.critic_learning_rate) or args.critic_learning_rate <= 0:
        raise ValueError("critic_learning_rate must be finite and positive")
    if not np.isfinite(args.critic_feature_scale) or args.critic_feature_scale <= 0:
        raise ValueError("critic_feature_scale must be finite and positive")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id,
        args.num_envs,
        backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video,
        run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
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
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        writer.add_text(
            "policy",
            "Beta host actor; 32x LR; 1 minibatch; hypersphered SiTU-GLU trunk; no advantage normalization; HL-Gauss critic",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, value_config=args.value, critic_feature_scale=args.critic_feature_scale).to(device)
        optimizer = make_optimizer(agent, args)
        value_model = agent.get_value
        target_model = agent.value_support.project

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, logits = agent.get_policy_and_value_logits(observations)
            probs = logits.softmax(dim=-1)
            value = agent.value_support.probs_to_scalar(probs)
            endpoint_mass = (probs[:, 0] + probs[:, -1]).mean()
            return value, agent.action_logprob(alpha, beta, native), endpoint_mass

        def loss_model(observations, native, old_logprobs, advantages, target_probs):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, target_probs, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            target_model = graph_compile(target_model)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(
            args.num_steps,
            args.num_envs,
            obs_shape,
            device,
            non_blocking=args.non_blocking_transfers,
            fields={"observations": obs_shape, "native_actions": (agent.action_dim,)},
        )
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        actor_scale_model = PercentileReturnScale().to(device) if args.actor_return_scale else None
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        update_grad_norms = torch.empty((max_updates, 2), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(
                envs,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                act_fn=warmup_action,
                horizon=horizon,
                phase_offsets=phases,
                seed=args.seed,
            )
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
                optimizer.param_groups[1]["lr"] = fraction * args.critic_learning_rate
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms) if rew_norm is not None else raw_reward
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs, value_endpoint_mass = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards,
                    values,
                    batch.terminations,
                    batch.truncations,
                    truncation_values,
                    tail_value,
                    args.gamma,
                    args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
                actor_scale = actor_scale_model(b_returns) if actor_scale_model is not None else b_returns.new_ones(())
                unscaled_advantage_rms = b_advantages.square().mean().sqrt()
                b_advantages.div_(actor_scale)
                # Freeze labels once per rollout, reusing them across every PPO epoch.
                b_target_probs = target_model(b_returns).detach()
                projection_error = agent.value_support.probs_to_scalar(b_target_probs) - b_returns
                value_diagnostics = {
                    "hlgauss/return_mean": b_returns.mean(),
                    "hlgauss/return_std": b_returns.std(unbiased=False),
                    "hlgauss/return_min": b_returns.min(),
                    "hlgauss/return_max": b_returns.max(),
                    "hlgauss/support_overflow_fraction": (
                        (b_returns < args.value.v_min) | (b_returns > args.value.v_max)
                    )
                    .float()
                    .mean(),
                    "hlgauss/projection_target_bias": projection_error.mean(),
                    "hlgauss/projection_target_rmse": projection_error.square().mean().sqrt(),
                    "hlgauss/preupdate_decoded_value_mse": (b_values - b_returns).square().mean(),
                    "hlgauss/preupdate_value_endpoint_mass": value_endpoint_mass,
                    "hlgauss/target_endpoint_mass": (b_target_probs[:, 0] + b_target_probs[:, -1]).mean(),
                }
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_target_probs[indices],
                        )
                        update_grad_norms[updates].copy_(optimizer_step(loss, optimizer, args.max_grad_norm))
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            with torch.no_grad():
                postupdate_error = value_model(b_obs).flatten() - b_returns
            logged = gather_metrics(
                {
                    "losses/policy_loss": last[0],
                    "losses/value_loss": last[1],
                    "losses/entropy": last[2],
                    "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4],
                    "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "grad/actor_preclip_norm": update_grad_norms[:updates, 0].mean(),
                    "grad/critic_preclip_norm": update_grad_norms[:updates, 1].mean(),
                    "actor/return_scale": actor_scale,
                    "actor/unscaled_advantage_rms": unscaled_advantage_rms,
                    "actor/advantage_rms": b_advantages.square().mean().sqrt(),
                    "actor/advantage_mean": b_advantages.mean(),
                    "hlgauss/postupdate_value_bias": postupdate_error.mean(),
                    "hlgauss/postupdate_value_rmse": postupdate_error.square().mean().sqrt(),
                    **value_diagnostics,
                }
            )
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.ppo_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                partial(make_env, reward_norm=args.reward_norm),
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=partial(Agent, value_config=args.value, critic_feature_scale=args.critic_feature_scale),
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
                    Namespace(**vars(args)),
                    episodic_returns,
                    repo_id,
                    "PPO",
                    f"runs/{run_name}",
                    f"videos/{run_name}-eval",
                )
    finally:
        resources.close()


if __name__ == "__main__":
    main()
