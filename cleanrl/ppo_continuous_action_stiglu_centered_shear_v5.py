# Centered volume-preserving transport PPO: separate location/entropy from shape.
# Gaussian controls native mean and entropy; alternating ODD additive shears
# learn nonlinear dependence without state-only translation or scale redundancy.
# T(-e)=-T(e), det(J_T)=1; exact PPO density needs neither ODE nor logdet network.
# Hypothesis: remove the affine base/coupling competition observed in v4.
# Restriction: centrally symmetric native law; not arbitrary skew or global optimality.
# Rollout samples detached, full inverse score differentiated; NOT flow matching.
import hashlib
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_actor import make_situ_sphere_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    save_model: bool = False
    flow_layers: int = 2
    """alternating odd, volume-preserving shears; zero is the Gaussian control"""
    initial_std: float = 0.610376541075546
    """initial native per-coordinate standard deviation, before tanh"""
    whiten_mean: bool = False
    """parameterize the base mean in fixed initial-noise units"""
    diagnostic_interval: int = 8

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
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

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads; two balances latency with concurrent N16 runs"""
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gaussian_logprob(sample, mean, log_std):
    return (-0.5 * ((sample - mean) * (-log_std).exp()).square()
            - log_std - 0.5 * LOG_TWO_PI).sum(-1)


def gaussian_kl_parts(old_mean, old_log_std, mean, log_std):
    """Exact KL(old || new), also exact after the shared invertible tanh map."""
    log_scale_ratio = old_log_std - log_std
    mean_kl = 0.5 * ((mean - old_mean) * (-log_std).exp()).square().sum(-1)
    scale_kl = (0.5 * torch.expm1(2.0 * log_scale_ratio) - log_scale_ratio).sum(-1)
    return mean_kl, scale_kl


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Agent(nn.Module):
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("transport policy requires Box actions")
        low, high = np.asarray(space.low).reshape(-1), np.asarray(space.high).reshape(-1)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
            raise ValueError("finite ordered action bounds required")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.observation_dim = int(np.prod(envs.single_observation_space.shape))
        if self.action_dim < 2 and args.flow_layers:
            raise ValueError("alternating coupling requires at least two action coordinates")
        self.split = self.action_dim // 2
        self.std_bias = math.atanh((math.log(args.initial_std) + 1.5) / 3.5)
        self.mean_scale = args.initial_std if args.whiten_mean else 1.0
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2, dtype=torch.float32))
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action range must be finite and positive in FP32")
        # Identical random initialization to the matched v2 Gaussian BEFORE
        # allocating any extra transport parameters.
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(self.observation_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(self.observation_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.0),
        )
        self.couplings = nn.ModuleList()
        for stage in range(args.flow_layers):
            fixed_dim = self.split if stage % 2 == 0 else self.action_dim - self.split
            changed_dim = self.action_dim - fixed_dim
            self.couplings.append(nn.Sequential(
                make_situ_sphere_trunk(self.observation_dim + fixed_dim, 64, n_blocks=1),
                layer_init(nn.Linear(64, changed_dim), std=0.0),
            ))
            # Host mirror requires a bias slot. Keep its cancelled coefficient
            # fixed at zero rather than allocating an unidentifiable Adam state.
            self.couplings[-1][-1].bias.requires_grad_(False)

    def get_value(self, observations):
        return self.critic(observations)

    def policy_parameters(self, observations):
        mean, raw_std = self.actor(observations).chunk(2, dim=-1)
        return mean * self.mean_scale, -1.5 + 3.5 * torch.tanh(raw_std + self.std_bias)

    def transport(self, observations, source, inverse=False):
        """Odd additive shears: T(-epsilon)=-T(epsilon), det(J_T)=1.

        Each shift is the odd part of an observation-conditioned network.
        Conditional native mean is exactly mu under the symmetric source;
        native entropy depends only on base sigma, not transport parameters.
        Marginal variances/covariances remain free; volume is not variance.
        Inverse graph dependencies remain essential to the exact score.
        """
        value = source
        logdet = torch.zeros_like(source[..., 0])
        stages = range(len(self.couplings) - 1, -1, -1) if inverse else range(len(self.couplings))
        for stage in stages:
            left, right = value[..., :self.split], value[..., self.split:]
            fixed, changed = (left, right) if stage % 2 == 0 else (right, left)
            paired_inputs = torch.cat((torch.cat((observations, fixed), -1),
                                       torch.cat((observations, -fixed), -1)), 0)
            positive, negative = self.couplings[stage](paired_inputs).chunk(2, 0)
            shift = 0.5 * (positive - negative)
            changed = changed - shift if inverse else changed + shift
            value = torch.cat((fixed, changed), -1) if stage % 2 == 0 else torch.cat((changed, fixed), -1)
        return value, logdet

    def sample_native(self, observations, source):
        """Differentiable reference only; production sampling uses the host mirror."""
        mean, log_std = self.policy_parameters(observations)
        value, logdet = self.transport(observations, source)
        native = mean + log_std.exp() * value
        logprob = (-0.5 * source.square() - 0.5 * LOG_TWO_PI - log_std).sum(-1) - logdet
        return native, logprob

    def logprob(self, observations, native):
        # Freeze the sampled action, not the inverse graph used to score it.
        mean, log_std = self.policy_parameters(observations)
        standardized = (native.detach() - mean) * (-log_std).exp()
        source, logdet = self.transport(observations, standardized, inverse=True)
        return (-0.5 * source.square() - 0.5 * LOG_TWO_PI - log_std).sum(-1) - logdet

    def physical_logprob(self, observations, native):
        logjac = 2.0 * (math.log(2.0) - native - F.softplus(-2.0 * native))
        return self.logprob(observations, native) - (logjac + self.action_scale.log()).sum(-1)


class HostSampler:
    """Permanent borrowed buffers. Save forward behavior score, never device replay."""
    def __init__(self, agent, rows):
        self.actor = make_host_mirror(agent.actor, rows)
        self.couplings = [make_host_mirror(module, 2 * rows) for module in agent.couplings]
        self.rows = rows
        self.split = agent.split
        self.obs_dim = agent.observation_dim
        self.std_bias, self.mean_scale = np.float32(agent.std_bias), np.float32(agent.mean_scale)
        self.action_scale = agent.action_scale.detach().cpu().numpy().copy()
        self.action_bias = agent.action_bias.detach().cpu().numpy().copy()
        shape = (rows, agent.action_dim)
        self.source = np.empty(shape, np.float32)
        self.value = np.empty_like(self.source)
        self.mean = np.empty_like(self.source)
        self.log_std = np.empty_like(self.source)
        self.std = np.empty_like(self.source)
        self.native = np.empty_like(self.source)
        self.action = np.empty_like(self.source)
        self.logprob = np.empty(rows, np.float32)
        self.logdet = np.empty(rows, np.float32)
        self.inputs, self.shifts = [], []
        for stage in range(len(self.couplings)):
            fixed_dim = self.split if stage % 2 == 0 else agent.action_dim - self.split
            self.inputs.append(np.empty((2 * rows, self.obs_dim + fixed_dim), np.float32))
            self.shifts.append(np.empty((rows, agent.action_dim - fixed_dim), np.float32))

    def refresh(self):
        self.actor.refresh()
        for coupling in self.couplings:
            coupling.refresh()

    def __call__(self, observations, rng, source=None):
        logits = self.actor(observations)
        mean, raw_std = np.split(logits, 2, -1)
        np.multiply(mean, self.mean_scale, out=self.mean)
        np.add(raw_std, self.std_bias, out=self.log_std)
        np.tanh(self.log_std, out=self.log_std)
        self.log_std *= np.float32(3.5)
        self.log_std += np.float32(-1.5)
        np.exp(self.log_std, out=self.std)
        if source is None:
            rng.standard_normal(self.source.shape, dtype=np.float32, out=self.source)
        else:
            np.copyto(self.source, source)
        np.copyto(self.value, self.source)
        self.logdet.fill(0)
        for stage, coupling in enumerate(self.couplings):
            fixed = self.value[:, :self.split] if stage % 2 == 0 else self.value[:, self.split:]
            changed = self.value[:, self.split:] if stage % 2 == 0 else self.value[:, :self.split]
            inputs, shift = self.inputs[stage], self.shifts[stage]
            inputs[:self.rows, :self.obs_dim] = observations
            inputs[self.rows:, :self.obs_dim] = observations
            inputs[:self.rows, self.obs_dim:] = fixed
            np.negative(fixed, out=inputs[self.rows:, self.obs_dim:])
            positive, negative = np.split(coupling(inputs), 2, 0)
            np.subtract(positive, negative, out=shift)
            shift *= np.float32(0.5)
            changed += shift
        np.multiply(self.value, self.std, out=self.native)
        self.native += self.mean
        # Source and Jacobian were computed by the ACTUAL sampler. The small
        # FP32 rounding difference at the stored native action is diagnosed.
        np.square(self.source, out=self.action)
        self.action *= np.float32(-0.5)
        self.action -= np.float32(0.5 * LOG_TWO_PI)
        self.action -= self.log_std
        np.sum(self.action, axis=-1, out=self.logprob)
        self.logprob -= self.logdet
        np.tanh(self.native, out=self.action)
        self.action *= self.action_scale
        self.action += self.action_bias
        return self.native, self.action


def rollout_statistics(agent, observations, native, behavior_logprob):
    replay = agent.logprob(observations, native)
    logratio = replay - behavior_logprob
    drift = torch.stack((logratio.mean(), logratio.abs().max(), logratio.square().mean().sqrt()))
    return agent.get_value(observations).flatten(), behavior_logprob, drift


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    newlogprob = agent.logprob(observations, native_actions)
    logratio = newlogprob - old_logprobs.detach()
    ratio = logratio.exp()
    with torch.no_grad():
        native_cross_entropy = -newlogprob.mean()
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.maximum(-advantages * ratio,
                            -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
    newvalue = agent.get_value(observations).flatten()
    if args.clip_vloss:
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.maximum((newvalue - returns).square(), (v_clipped - returns).square()).mean()
    else:
        v_loss = 0.5 * (newvalue - returns).square().mean()
    return pg_loss + args.vf_coef * v_loss, torch.stack((pg_loss.detach(), v_loss.detach(),
        native_cross_entropy, old_approx_kl, approx_kl, clipfrac))


def provenance(agent, args):
    return {
        "config": vars(args).copy(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "critic_sha256": state_hash(agent.critic),
        "actor_sha256": state_hash(agent.actor),
        "actor_trunk_sha256": state_hash(agent.actor[0]),
        "actor_parameters": sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.couplings.parameters()),
        "initial_pre_tanh_std": args.initial_std,
        "likelihood": "exact unit-determinant inverse transport density; host source/base logstd behavior score; FP32 rounding drift logged",
        "method": "PPO on centered odd volume-preserving shears in standardized Gaussian coordinates; NOT flow matching",
        "gradient": "sampled action detached; exact score differentiates the inverse map, never the rollout sampling graph",
        "limitations": ["PPO clipping is not a hard KL bound", "sampled KL, not analytic KL", "volume preservation does not bound conditioning or marginal variances; centrally symmetric native policies cannot represent skew", "one seed is not an optimality result"],
    }


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if not args.compile:
        raise ValueError("the matched experiment requires compiled training")
    if args.flow_layers < 0 or args.diagnostic_interval <= 0:
        raise ValueError("flow_layers must be nonnegative and diagnostic_interval positive")
    if not math.exp(LOG_STD_MIN) < args.initial_std < math.exp(LOG_STD_MAX):
        raise ValueError("initial_std must be inside the open logstd bounds")
    if args.ent_coef != 0.0:
        raise ValueError("the matched experiment requires ent_coef=0")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )



def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}"
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    metric_file = resources.enter_context(open(f"runs/{run_name}/metrics.jsonl", "w"))
    with open(f"runs/{run_name}/config.json", "w") as config:
        json.dump(vars(args), config, indent=2)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "exact standardized coupling transport PPO; detached host samples; differentiable inverse scores")
        Path(f"runs/{run_name}/source.py").write_text(Path(__file__).read_text())
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args)
        with open(f"runs/{run_name}/provenance.json", "w") as output:
            json.dump(provenance(agent, args), output, indent=2)
        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def statistics_model(observations, native, behavior_logprob):
            return rollout_statistics(agent, observations, native, behavior_logprob)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        def geometry_model(observations, native, behavior_logprob):
            logratio = agent.logprob(observations, native) - behavior_logprob
            _, log_std = agent.policy_parameters(observations)
            return torch.stack(((-logratio).mean(), ((logratio.exp() - 1) - logratio).mean(),
                                logratio.square().mean().sqrt(), logratio.abs().max(), log_std.exp().mean()))

        def distribution_diagnostics(observations, source):
            native, forward_logprob = agent.sample_native(observations, source)
            inverse_logprob = agent.logprob(observations, native)
            mean, log_std = agent.policy_parameters(observations)
            standardized = (native - mean) * (-log_std).exp()
            recovered, _ = agent.transport(observations, standardized, inverse=True)
            conditional = native.view(16, 64, agent.action_dim)
            physical = native.tanh().view_as(conditional)
            jac = (2 * (math.log(2.0) - native - F.softplus(-2 * native)) + agent.action_scale.log()).sum(-1)
            return torch.stack((conditional.std(1).mean(), physical.std(1).mean(),
                (physical.abs() > 0.99).float().mean(), (standardized - source).square().mean().sqrt(),
                (recovered - source).square().mean().sqrt(), (forward_logprob - inverse_logprob).abs().max(),
                (-forward_logprob + jac).mean()))

        if args.compile:
            geometry_model = torch.compile(geometry_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            distribution_diagnostics = torch.compile(distribution_diagnostics, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
            statistics_model = graph_compile(statistics_model)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        sampler = np.random.default_rng(args.seed)
        sample_actions = HostSampler(agent, args.num_envs)
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed + 739)

        def act(observations):
            native, action = sample_actions(observations, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,),
                                           "behavior_logprob": ()})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
            episode_returns, episode_lengths = [], []
            bootstraps.reset()
            sample_actions.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native,
                                  behavior_logprob=sample_actions.logprob)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        episode_returns.append(episode_return)
                        episode_lengths.append(float(info["episode"]["l"]))
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_behavior = batch.fields["behavior_logprob"].flatten()
                b_values, b_logprobs, replay_drift = statistics_model(b_obs, b_native, b_behavior)
                # CUDA graph outputs are borrowed; freeze statistics until the next rollout.
                b_values, b_logprobs = b_values.clone(), b_logprobs.clone()
                replay_drift = replay_drift.clone()
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            updates = 0

            with timer.span("update"):
                for _ in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1

            with torch.no_grad():
                geometry = geometry_model(b_obs, b_native, b_logprobs)
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/native_cross_entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "geometry/sampled_kl": geometry[0], "geometry/nonnegative_kl_estimate": geometry[1],
                "geometry/logratio_rms": geometry[2], "geometry/logratio_max_abs": geometry[3],
                "geometry/base_std": geometry[4], "geometry/replay_mean_logratio": replay_drift[0],
                "geometry/replay_max_abs_logratio": replay_drift[1],
                "geometry/replay_rms_logratio": replay_drift[2],
            })
            if iteration == 1 or iteration % args.diagnostic_interval == 0 or iteration == args.num_iterations:
                with timer.span("diagnostics"), torch.no_grad():
                    probe_indices = torch.linspace(0, args.batch_size - 1, 16, device=device).long()
                    probe_obs = b_obs[probe_indices].repeat_interleave(64, 0)
                    source = torch.randn((1024, agent.action_dim), device=device, generator=diagnostic_generator)
                    diagnostics = distribution_diagnostics(probe_obs, source)
                names = ("conditional_native_std", "conditional_action_std", "saturation_fraction",
                         "transport_displacement_rms", "inverse_source_rms", "inverse_score_max_error", "physical_entropy")
                logged.update(gather_metrics({f"distribution/{name}": diagnostics[i] for i, name in enumerate(names)}))
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            logged["geometry/optimizer_steps"] = updates
            now = time.perf_counter()
            logged["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
            logged["charts/SPS"] = int(global_step / (now - start_time))
            logged["charts/interval_SPS"] = (global_step - interval_step) / (now - interval_start)
            for phase, timing in timer.summary().items():
                logged[f"timing/{phase}_s"] = timing["total_s"]
            metric_file.write(json.dumps({"step": global_step, "iteration": iteration,
                                         "flow_layers": args.flow_layers, "episode_returns": episode_returns,
                                         "episode_lengths": episode_lengths, **logged}) + "\n")
            metric_file.flush()
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        if args.save_model:
            torch.save({"state_dict": agent.state_dict(), "args": vars(args),
                        "optimizer": optimizer.state_dict(), "global_step": global_step,
                        "obs_norm": {"means": torch.from_numpy(obs_norm.means.copy()),
                                     "variances": torch.from_numpy(obs_norm.variances.copy()),
                                     "counts": torch.from_numpy(obs_norm.counts.copy()),
                                     "epsilon": obs_norm.epsilon, "clip": obs_norm.clip},
                        "rew_norm": {"means": torch.from_numpy(rew_norm.means.copy()),
                                     "variances": torch.from_numpy(rew_norm.variances.copy()),
                                     "counts": torch.from_numpy(rew_norm.counts.copy()),
                                     "returns": torch.from_numpy(rew_norm.returns.copy()),
                                     "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon, "clip": rew_norm.clip}},
                       f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
