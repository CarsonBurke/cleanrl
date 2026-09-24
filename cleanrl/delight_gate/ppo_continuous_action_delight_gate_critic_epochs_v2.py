# Delight gate v2: v1 plus extra full-batch, critic-only epochs per rollout.
# The actor still takes exactly one on-policy step (no ratios, no minibatches),
# and advantages still come from the pre-update critic. The critic then takes
# critic_epochs - 1 more full-batch steps on the same GAE returns.
# Why: DG's noise term is an entropy push of about Var(U)/(4 eta). A one-step
# critic makes U mostly noise, so better value fits should sharpen the gate's
# breakthrough/blunder signal more for DG than for the PG control.
#
# v1 header follows.
# Delightful Policy Gradient (Osband 2026) on a Beta policy, single on-policy step.
# Base: residual SiTU-GLU nGPT v7, no advantage normalization, no value clipping.
# No epochs, no minibatches, no ratios, V(s) critic, reward normalization on.
# Each score term is weighted by gate = sigmoid(U * s / eta), detached.
#
# Beta surprisal: raw -log density is unusable. It shifts with the action range
# and turns negative as the policy sharpens (the base's entropy reaches -8.6
# nats), which flips delight's sign. We use a calibrated tail surprisal:
#   s(a) = -log P_{A'~pi}(l(A') >= l(a)),   l = log pi(mode) - log pi(.) >= 0,
# using the Gamma that matches the analytic mean and variance of l per state.
# It is exact for Gaussians and in the concentrated-Beta limit, and invariant to
# action scale. s >= 0, so breakthroughs and blunders keep their sign. On-policy
# s ~ Exp(1) in every state and action dimension, so eta is in advantage units
# only. eta_eff = eta * RMS(U) (scale only, no centering).
# Hypothesis: suppressing high-surprisal failures keeps finite-batch noise off
# the update, preserving breakthroughs rebalances toward badly-solved states, and
# DG's implicit Var(U)-scaled entropy push delays Beta collapse. Together these
# should beat the matched single-step PG control (--policy-gradient pg).
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, explained_variance, gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
POLICY_GRADIENTS = ("delight", "pg", "enlightened")
SURPRISALS = ("tail", "mode")


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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0024
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 64
    """steps per environment per rollout; one full-batch gradient step per rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    critic_epochs: int = 10
    """full-batch critic steps per rollout (the first is joint with the single actor step)"""

    # Delight gate
    policy_gradient: str = "delight"
    """delight: sigmoid(U s / eta) gate; pg: gate = 1 (control); enlightened: gate = 1{U > 0} (eta -> 0)"""
    surprisal: str = "tail"
    """tail: calibrated -log tail probability (~Exp(1) on-policy); mode: log pi(mode) - log pi(a)"""
    delight_temperature: float = 1.0
    """eta, in units of the batch advantage RMS"""
    surprisal_cap: float = 20.0
    """upper bound on surprisal (tail probability floor exp(-cap))"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile policy statistics, the loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""
    log_interval: int = 8192
    """environment steps between learner-metric logs"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return self.readout_gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))


class ResidualHostMirror:
    """Compose fused FP32 stage mirrors without changing shared implementations.

    Stage outputs are permanent buffers; copy the first before calling the
    second, and add into our own permanent buffer before evaluating the head.
    """

    def __init__(self, actor, num_envs):
        self.first = make_host_mirror(actor.first, num_envs)
        self.second = make_host_mirror(actor.second, num_envs)
        self.head = make_host_mirror(actor.head, num_envs)
        self.hidden = np.empty((num_envs, 64), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_envs, 1), dtype=np.float32)
        self.actor = actor
        self.readout_gain = np.empty(actor.readout_gain.numel(), dtype=np.float32)
        self.refresh()

    def refresh(self):
        self.first.refresh()
        self.second.refresh()
        self.head.refresh()
        np.copyto(self.readout_gain, self.actor.readout_gain.detach().cpu().numpy())

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, observations):
        np.copyto(self.hidden, self.first(observations))
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.hidden)
        output = self.head(self.hidden)
        np.multiply(output, self.readout_gain, out=output)
        return output


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta policy requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta policy requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("log_action_scale", (self.action_high - self.action_low).log())
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing Adam moments or biases."""
        for trunk in (self.actor, self.critic):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1),
                                    (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            weight = trunk.head[0].weight
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)


def beta_log_density(alpha, beta, x):
    """Per-dimension native log density; xlogy keeps alpha == 1 at x == 0 finite."""
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return torch.xlogy(alpha - 1.0, x) + torch.xlogy(beta - 1.0, 1.0 - x) - log_norm


@torch.no_grad()
def beta_surprisal(alpha, beta, native, kind, cap):
    """Scale-invariant, non-negative surprisal of product-Beta samples.

    ``mode``: l = log pi(mode) - log pi(a) summed over dims (Jacobians cancel).
    ``tail``: -log P(l(A') >= l(a)) with l's law moment-matched by a Gamma, using
    E[l] = H + log pi(mode) and the trigamma variance of the log density. That is
    exact for Gaussians (l ~ Gamma(d/2, 1)) and in the concentrated-Beta limit.
    """
    alpha, beta, native = alpha.double(), beta.double(), native.double()
    concentration = alpha + beta
    mode = torch.where(concentration > 2.0 + 1e-9,
                       (alpha - 1.0) / (concentration - 2.0).clamp_min(1e-9),
                       torch.full_like(alpha, 0.5))
    mode_log_density = beta_log_density(alpha, beta, mode)
    relative = (mode_log_density - beta_log_density(alpha, beta, native)).sum(-1).clamp_min(0.0)
    if kind == "mode":
        return relative.clamp_max(cap).float()
    entropy = Beta(alpha, beta, validate_args=False).entropy()
    mean = (entropy + mode_log_density).sum(-1).clamp_min(1e-9)
    shared = torch.polygamma(1, concentration)
    variance = ((alpha - 1.0).square() * (torch.polygamma(1, alpha) - shared)
                + (beta - 1.0).square() * (torch.polygamma(1, beta) - shared)
                - 2.0 * (alpha - 1.0) * (beta - 1.0) * shared).sum(-1).clamp_min(1e-18)
    shape, scale = mean.square() / variance, variance / mean
    tail = torch.special.gammaincc(shape, relative / scale)
    return (-tail.clamp_min(float(np.exp(-cap))).log()).float()


@torch.no_grad()
def delight_gate(alpha, beta, native, advantages, args):
    """Detached per-sample gate and the effective temperature (0 when unused)."""
    zero = advantages.new_zeros(())
    if args.policy_gradient == "pg":
        return torch.ones_like(advantages), zero, zero
    if args.policy_gradient == "enlightened":
        return (advantages > 0).float(), zero, zero
    surprisal = beta_surprisal(alpha, beta, native, args.surprisal, args.surprisal_cap)
    temperature = args.delight_temperature * advantages.square().mean().sqrt().clamp_min(1e-8)
    return torch.sigmoid(advantages * surprisal / temperature), surprisal, temperature


def delight_loss(agent, observations, native_actions, gates, advantages, returns, args):
    """Gated score-function loss plus V(s) regression; no ratios, no clipping."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    logprob = distribution.log_prob(native_actions).sum(-1)
    pg_loss = -(gates * advantages * logprob).mean()
    v_loss = 0.5 * (newvalue.view(-1) - returns).square().mean()
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    loss = pg_loss + args.vf_coef * v_loss
    return loss, torch.stack((pg_loss.detach(), v_loss.detach(), entropy.detach()))


def critic_loss(agent, observations, returns, args):
    """Critic-only V(s) regression for the extra full-batch epochs."""
    v_loss = 0.5 * (agent.get_value(observations).view(-1) - returns).square().mean()
    return args.vf_coef * v_loss, v_loss.detach()


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.critic_epochs) <= 0:
        raise ValueError("environment and rollout counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.policy_gradient not in POLICY_GRADIENTS or args.surprisal not in SURPRISALS:
        raise ValueError(f"policy_gradient in {POLICY_GRADIENTS}, surprisal in {SURPRISALS}")
    if not args.delight_temperature > 0 or not args.surprisal_cap > 0:
        raise ValueError("delight temperature and surprisal cap must be positive")
    if not args.cuda:
        raise ValueError("the shared trainer requires CUDA")
    args.batch_size = args.num_envs * args.num_steps
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs), run_name=run_name,
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
    log_every = max(1, args.log_interval // args.batch_size)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); one on-policy gradient step per rollout; "
                        "delight gate sigmoid(U s / (eta RMS(U))) with calibrated Beta tail surprisal")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations):
            """Behaviour Beta parameters and values for the whole rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return alpha, beta, value.flatten()

        def loss_model(observations, native, gates, advantages, returns):
            return delight_loss(agent, observations, native, gates, advantages, returns, args)

        def critic_model(observations, returns):
            return critic_loss(agent, observations, returns, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            critic_model = torch.compile(critic_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        critic_parameters = list(agent.critic.parameters())
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = ResidualHostMirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
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
            bootstraps.reset()
            host_actor.refresh()
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
                b_alpha, b_beta, b_values = rollout_statistics(b_obs)
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
                # One step on fresh data: the learner policy is the behaviour policy.
                b_gates, b_surprisal, temperature = delight_gate(
                    b_alpha, b_beta, b_native, b_advantages, args)
            with timer.span("update"):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                loss, metrics = loss_model(b_obs, b_native, b_gates, b_advantages, b_returns)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                normalize_matrices()
                # Later graph replays may reuse the joint step's output buffers.
                metrics = metrics.clone()
                critic_value_loss = metrics[1]
                # Actor grads stay None, so Adam leaves the actor and its moments untouched.
                for _ in range(args.critic_epochs - 1):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss, critic_value_loss = critic_model(b_obs, b_returns)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm)
                    optimizer.step()
                    normalize_matrices()
                critic_value_loss = critic_value_loss.clone()

            if iteration % log_every and iteration != args.num_iterations:
                continue
            positive = b_advantages > 0
            negative = ~positive
            logged = gather_metrics({
                "losses/policy_loss": metrics[0], "losses/value_loss": metrics[1],
                "losses/value_loss_final": critic_value_loss,
                "losses/entropy": metrics[2], "losses/grad_norm": grad_norm,
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "delight/advantage_rms": b_advantages.square().mean().sqrt(),
                "delight/gate_mean": b_gates.mean(),
                "delight/gate_positive": (b_gates * positive).sum() / positive.sum().clamp_min(1),
                "delight/gate_negative": (b_gates * negative).sum() / negative.sum().clamp_min(1),
                "delight/surprisal_mean": b_surprisal.mean(),
                "delight/surprisal_tail3": (b_surprisal > 3.0).float().mean(),
                "delight/temperature": temperature,
                "delight/concentration": (b_alpha + b_beta).mean(),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (now - start_time))}")
            interval_start, interval_step = now, global_step

        transfer.close()
        envs.close()
    finally:
        resources.close()


if __name__ == "__main__":
    main()
