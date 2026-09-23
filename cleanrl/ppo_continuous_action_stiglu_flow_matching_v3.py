# Advantage-weighted conditional latent flow matching (arXiv:2307.08698v1, Eq.9).
# Paper convention: data u0, Gaussian noise u1, ut=(1-t)u0+t*u1, target u1-u0;
# sample the velocity ODE backwards from t=1 to0 with fixed-step Heun.
# RL adaptation (NOT PPO): fit exp(standardized GAE / temperature)-weighted
# behavior actions. This tilts the conditional target AND reweights states;
# it is not an exact per-state E-step, likelihood ratio, or KL constraint.
# Samples, targets and inner ODE states have no training graph. Gradients are
# local velocity regression gradients, with the same GAE/value learner as PPO.
# Gaussian-AWR mode isolates this improvement objective from the flow family.
# Fixed tanh decoder scale matches v2 Beta/Gaussian initial action variance.
import hashlib
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
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
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
# E[tanh(s*N(0,1))**2] = 1/(2*(1+log(2))+1), Gauss-Hermite128 calibration.
MATCHED_STD = 0.610376541075546
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    policy: Literal["flow", "gaussian_awr"] = "flow"
    temperature: float = 1.0
    """exponential weighting temperature in rollout-standardized advantage units"""
    diagnostic_interval: int = 8
    flow_steps: int = 8
    """Heun steps per sampled thought; two velocity evaluations per step"""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    save_model: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    """value clipping only; the actor objective has no PPO ratio"""
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    env_backend: str = "auto"
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def advantage_logits(advantages, temperature):
    """Scale before centering to avoid finite FP32 variance overflow."""
    values = advantages.detach()
    scaled = values / values.abs().max().clamp_min(torch.finfo(values.dtype).tiny)
    centered = scaled - scaled.mean()
    standardized = centered / centered.square().mean().sqrt().clamp_min(torch.finfo(values.dtype).eps)
    return standardized / temperature


def advantage_weights(advantages, temperature):
    """Positive normalized rollout weights; no clipping and no signed MSE."""
    weights = torch.softmax(advantage_logits(advantages, temperature), dim=0) * advantages.numel()
    ess_fraction = weights.mean().square() / weights.square().mean()
    return weights.detach(), ess_fraction.detach()


def flow_interpolant(targets, noise, times):
    """Paper Eq.9, with data at t=0 and source Gaussian at t=1."""
    targets, noise, times = targets.detach(), noise.detach(), times.detach()
    return (1.0 - times) * targets + times * noise, noise - targets


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("continuous flow control requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("finite ordered action bounds required")
        self.policy = args.policy
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)
        self.std_bias = math.atanh((math.log(MATCHED_STD) + 1.5) / 3.5)
        # Identical initialization order and dimensions to both v2 controls.
        self.critic = nn.Sequential(
            make_situ_sphere_trunk(self.obs_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        input_dim = self.obs_dim + self.action_dim + 1 if self.policy == "flow" else self.obs_dim
        output_dim = self.action_dim if self.policy == "flow" else 2 * self.action_dim
        self.actor = nn.Sequential(
            make_situ_sphere_trunk(input_dim, 64, n_blocks=3),
            layer_init(nn.Linear(64, output_dim), std=0.0),
        )

    def get_value(self, observations):
        return self.critic(observations)

    def velocity(self, observations, states, times):
        return self.actor(torch.cat((observations, states, times), dim=-1))

    def gaussian_parameters(self, observations):
        mean, raw_std = self.actor(observations).chunk(2, dim=-1)
        return mean, -1.5 + 3.5 * torch.tanh(raw_std + self.std_bias)


def policy_loss(agent, observations, native, weights, returns, old_values, args):
    native, weights = native.detach(), weights.detach()
    if agent.policy == "flow":
        # Simulation-free training: no ODE solve or differentiation through a sample.
        noise = torch.randn_like(native)
        times = torch.rand_like(native[:, :1])
        states, target_velocity = flow_interpolant(native, noise, times)
        prediction = agent.velocity(observations, states, times)
        per_sample = (prediction - target_velocity).square().sum(-1)
        actor_loss = (weights * per_sample).mean()
    else:
        mean, log_std = agent.gaussian_parameters(observations)
        # Native storage is unit source coordinates for both policies. The fixed
        # tanh-affine Jacobian does not affect weighted NLL gradients.
        actions = native * MATCHED_STD
        logprob = (-0.5 * ((actions - mean) * (-log_std).exp()).square()
                   - log_std - 0.5 * LOG_TWO_PI).sum(-1)
        per_sample = -logprob
        actor_loss = (weights * per_sample).mean()
    values = agent.get_value(observations).flatten()
    if args.clip_vloss:
        clipped = old_values + torch.clamp(values - old_values, -args.clip_coef, args.clip_coef)
        value_loss = 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
    else:
        value_loss = 0.5 * (values - returns).square().mean()
    loss = actor_loss + args.vf_coef * value_loss
    return loss, torch.stack((actor_loss.detach(), value_loss.detach(), per_sample.mean().detach()))


class HostSampler:
    """Borrowed FP32 buffers; detached Heun sampling or matched Gaussian sampling."""
    def __init__(self, agent, num_rows, steps):
        self.policy = agent.policy
        self.obs_dim = agent.obs_dim
        self.action_dim = agent.action_dim
        self.steps = steps
        self.std_bias = np.float32(agent.std_bias)
        self.scale = agent.action_scale.cpu().numpy().copy()
        self.bias = agent.action_bias.cpu().numpy().copy()
        self.actor = make_host_mirror(agent.actor, num_rows)
        shape = (num_rows, agent.action_dim)
        self.native = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)
        self.noise = np.empty(shape, dtype=np.float32)
        self.first_velocity = np.empty(shape, dtype=np.float32)
        self.work = np.empty(shape, dtype=np.float32)
        self.inputs = np.empty((num_rows, agent.obs_dim + agent.action_dim + 1), dtype=np.float32)

    def refresh(self):
        self.actor.refresh()

    def __call__(self, observations, rng, *, noise=None):
        if noise is None:
            rng.standard_normal(self.noise.shape, dtype=np.float32, out=self.noise)
        else:
            np.copyto(self.noise, noise)
        if self.policy == "gaussian_awr":
            mean, raw_std = np.split(self.actor(observations), 2, axis=-1)
            np.add(raw_std, self.std_bias, out=self.work)
            np.tanh(self.work, out=self.work)
            self.work *= np.float32(3.5)
            self.work += np.float32(-1.5)
            np.exp(self.work, out=self.work)
            np.multiply(self.work, self.noise, out=self.native)
            self.native += mean
            self.native /= np.float32(MATCHED_STD)
        else:
            np.copyto(self.native, self.noise)
            self.inputs[:, :self.obs_dim] = observations
            step_size = np.float32(-1.0 / self.steps)
            for step in range(self.steps):
                self.inputs[:, self.obs_dim:-1] = self.native
                self.inputs[:, -1] = np.float32(1.0 - step / self.steps)
                np.copyto(self.first_velocity, self.actor(self.inputs))
                np.multiply(self.first_velocity, step_size, out=self.work)
                self.work += self.native
                self.inputs[:, self.obs_dim:-1] = self.work
                self.inputs[:, -1] = np.float32(1.0 - (step + 1) / self.steps)
                np.add(self.first_velocity, self.actor(self.inputs), out=self.work)
                self.work *= step_size * np.float32(0.5)
                self.native += self.work
        np.multiply(self.native, np.float32(MATCHED_STD), out=self.action)
        np.tanh(self.action, out=self.action)
        self.action *= self.scale
        self.action += self.bias
        return self.native, self.action


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.flow_steps, args.diagnostic_interval) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch and solver counts must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size <= 0 or args.batch_size < 2:
        raise ValueError("minibatch cannot be empty; rollout needs two samples")
    if not args.cuda:
        raise ValueError("CUDA is required")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=args.capture_video, run_name=run_name)


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
        writer.add_text("policy", f"{args.policy}: paper Eq.9 advantage-weighted fitting; no PPO ratios or KL claims; SiTU-GLU")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args)
        with open(__file__, "rb") as source:
            source_bytes = source.read()
        provenance = {"critic_hash": state_hash(agent.critic), "actor_trunk_hash": state_hash(agent.actor[0]),
                      "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                      "actor_parameters": sum(p.numel() for p in agent.actor.parameters()),
                      "critic_parameters": sum(p.numel() for p in agent.critic.parameters()),
                      "initial_pre_tanh_std": MATCHED_STD, "initial_action_variance": 1/(2*(1+math.log(2))+1),
                      "paper": "https://arxiv.org/pdf/2307.08698v1", "args": vars(args)}
        with open(f"runs/{run_name}/provenance.json", "w") as handle:
            json.dump(provenance, handle, indent=2)
        with open(f"runs/{run_name}/source.py", "wb") as handle:
            handle.write(source_bytes)
        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations):
            return agent.get_value(observations).flatten()

        def loss_model(observations, native, weights, returns, old_values):
            return policy_loss(agent, observations, native, weights, returns, old_values, args)

        diagnostic_velocity = agent.velocity
        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            if args.policy == "flow":
                diagnostic_velocity = graph_compile(diagnostic_velocity)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        sampler = np.random.default_rng(args.seed)
        sample_actions = HostSampler(agent, args.num_envs, args.flow_steps)
        probe_rows, probe_states, probe_samples = 256, 8, 32
        probe_sampler = HostSampler(agent, probe_rows, args.flow_steps)
        fine_sampler = HostSampler(agent, probe_rows, 2 * args.flow_steps) if args.policy == "flow" else None
        reference_sampler = HostSampler(agent, probe_rows, 4 * args.flow_steps) if args.policy == "flow" else None
        probe_rng = np.random.default_rng(args.seed + 773)
        probe_generator = torch.Generator(device=device).manual_seed(args.seed + 991)

        def act(observations):
            native, action = sample_actions(observations, sampler)
            if not np.isfinite(native).all() or not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite samples")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 3), device=device)
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
                b_values = rollout_statistics(b_obs).clone()
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
                b_weights, weight_ess = advantage_weights(b_advantages, args.temperature)
                weight_diagnostics = torch.stack((weight_ess, b_weights.max()/args.batch_size,
                    b_weights.topk(max(1,args.batch_size//100)).values.sum()/args.batch_size,
                    b_advantages.std(unbiased=False)))
            diagnose = iteration == 1 or iteration % args.diagnostic_interval == 0 or iteration == args.num_iterations
            probe_observations = probe_noise = old_probe_action = None
            if diagnose:
                with timer.span("diagnostics", use_cuda=False):
                    state_indices = torch.linspace(0, args.batch_size-1, probe_states, device=device).long()
                    probe_observations = np.ascontiguousarray(np.repeat(b_obs[state_indices].cpu().numpy(), probe_samples, axis=0))
                    probe_noise = probe_rng.standard_normal((probe_rows, agent.action_dim), dtype=np.float32)
                    probe_sampler.refresh()
                    old_probe_action = probe_sampler(probe_observations, probe_rng, noise=probe_noise)[1].copy()
            updates = 0
            with timer.span("update"):
                for _ in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(b_obs[indices], b_native[indices], b_weights[indices],
                                                   b_returns[indices], b_values[indices])
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/actor_loss": last[0], "losses/value_loss": last[1],
                "losses/unweighted_fit_loss": last[2],
                "losses/explained_variance": explained_variance(b_values,b_returns),
                "weights/ess_fraction": weight_diagnostics[0],
                "weights/max_mass": weight_diagnostics[1], "weights/top_one_percent_mass": weight_diagnostics[2],
                "weights/raw_advantage_std": weight_diagnostics[3],
            })
            if diagnose:
                assert probe_observations is not None and probe_noise is not None and old_probe_action is not None
                with timer.span("diagnostics", use_cuda=False):
                    probe_sampler.refresh()
                    post_native, post_action = probe_sampler(probe_observations, probe_rng, noise=probe_noise)
                    normalized = (post_action - probe_sampler.bias) / probe_sampler.scale
                    paired_rms = np.sqrt(np.mean(((post_action-old_probe_action)/probe_sampler.scale)**2,axis=-1))
                    conditional = normalized.reshape(probe_states,probe_samples,agent.action_dim)
                    centered = conditional - conditional.mean(axis=1,keepdims=True)
                    cov = np.einsum("nki,nkj->nij",centered,centered)/(probe_samples-1)
                    eigenvalues = np.linalg.eigvalsh(cov)
                    std = np.sqrt(np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0))
                    denom = std[:,:,None]*std[:,None,:]
                    correlation = np.divide(cov,denom,out=np.zeros_like(cov),where=denom>1e-12)
                    off_diagonal = ~np.eye(agent.action_dim,dtype=bool)
                    logged.update({"sampling/paired_update_action_rms": float(paired_rms.mean()),
                        "sampling/paired_update_action_p95":float(np.quantile(paired_rms,.95)),
                        "sampling/conditional_action_std":float(std.mean()),
                        "sampling/conditional_cov_min_eigenvalue":float(eigenvalues[:,0].mean()),
                        "sampling/conditional_cov_max_eigenvalue":float(eigenvalues[:,-1].mean()),
                        "sampling/conditional_abs_correlation":float(np.abs(correlation[:,off_diagonal]).mean()) if agent.action_dim>1 else 0.0,
                        "sampling/saturation_fraction":float((np.abs(normalized)>.99).mean()),
                        "sampling/conditional_native_std":float(post_native.reshape(probe_states,probe_samples,agent.action_dim).std(axis=1,ddof=1).mean())})
                    if fine_sampler is not None and reference_sampler is not None:
                        fine_sampler.refresh()
                        reference_sampler.refresh()
                        fine_action = fine_sampler(probe_observations,probe_rng,noise=probe_noise)[1]
                        reference_action = reference_sampler(probe_observations,probe_rng,noise=probe_noise)[1]
                        for name,left,right in (("coarse_fine",post_action,fine_action),("fine_reference",fine_action,reference_action),
                                                ("coarse_reference",post_action,reference_action)):
                            diff=np.sqrt(np.mean(((left-right)/probe_sampler.scale)**2,axis=-1))
                            logged[f"solver/{name}_action_rms"]=float(diff.mean())
                            logged[f"solver/{name}_action_p95"]=float(np.quantile(diff,.95))
                if args.policy == "flow":
                    with timer.span("diagnostics"), torch.no_grad():
                        fit_indices=torch.linspace(0,args.batch_size-1,probe_rows,device=device).long()
                        fit_noise=torch.randn((probe_rows,agent.action_dim),device=device,generator=probe_generator)
                        fit_data=b_native[fit_indices]
                        fit_weights=torch.softmax(advantage_logits(b_advantages,args.temperature)[fit_indices],dim=0)
                        fit_losses=[]
                        for time_bin in range(4):
                            fit_time=torch.full((probe_rows,1),(time_bin+.5)/4,device=device)
                            fit_state,fit_target=flow_interpolant(fit_data,fit_noise,fit_time)
                            prediction=diagnostic_velocity(b_obs[fit_indices],fit_state,fit_time)
                            errors=(prediction-fit_target).square().sum(-1)
                            fit_losses.extend((errors.mean(),(fit_weights*errors).sum()))
                        logged.update(gather_metrics({f"flow/{kind}_mse_bin{index//2}":value for index,value in enumerate(fit_losses)
                                                      for kind in ["unweighted" if index%2==0 else "weighted"]}))
            if any(not np.isfinite(value) for name,value in logged.items() if name!="losses/explained_variance"):
                raise FloatingPointError("nonfinite policy learner metrics")
            logged["weights/temperature"]=args.temperature
            logged["sampling/nfe"]=2*args.flow_steps if args.policy=="flow" else 1
            logged["optimizer/steps"]=updates
            metric_file.write(json.dumps({"step":global_step,**logged})+"\n")
            metric_file.flush()
            for name,value in logged.items():
                writer.add_scalar(name,value,global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        if args.save_model:
            torch.save({"state_dict": agent.state_dict(), "args": vars(args)},
                       f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
